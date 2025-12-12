"""
Algorithm Adapters for Smart Parallel Module.

Provides adapters to bridge existing algorithms with the new ParallelCoordinator
interface, enabling gradual migration without breaking existing code.

Supported algorithms:
- agd (Alternating Gradient Descent)
- bigamp (Standard BiG-AMP)
- bigamp_spreading (BiG-AMP with Spreading, non-parallel)
- bigamp_spreading_parallel (BiG-AMP with Spreading, full parallel)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Tuple, Type
from dataclasses import dataclass
import torch
import logging

from .execution_modes import (
    EstimationParams,
    ExecutionPlan,
    BatchConfig,
    AllocationConfig,
    AllocationPresets,
)
from .memory_estimator import MemoryEstimator

logger = logging.getLogger(__name__)


class BaseAlgorithmAdapter(ABC):
    """
    Base class for algorithm adapters.
    
    Provides common interface for all algorithms to work with ParallelCoordinator.
    """
    
    # Class variable: algorithm key this adapter handles
    algorithm_key: str = ""
    
    def __init__(self, algorithm: Any):
        """
        Initialize adapter.
        
        Args:
            algorithm: The algorithm instance to wrap
        """
        self.algorithm = algorithm
        self._estimator = MemoryEstimator()
    
    @abstractmethod
    def create_estimation_params(
        self,
        N1: int,
        N2: int,
        M: int,
        S: int,
        alpha_values: List[float],
        **kwargs,
    ) -> EstimationParams:
        """Create EstimationParams for this algorithm."""
        pass
    
    @abstractmethod
    def run_with_plan(
        self,
        plan: ExecutionPlan,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: Optional[torch.Tensor],
        alpha_values: List[float],
        seed: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute algorithm according to the plan.
        
        Args:
            plan: Execution plan from ParallelCoordinator
            W_teacher, X_teacher, Y_teacher: Teacher data
            masks: Observation masks (for dense algorithms)
            alpha_values: Alpha values to train
            seed: Random seed
            
        Returns:
            W_students, X_students
        """
        pass
    
    def get_estimator(self) -> MemoryEstimator:
        """Get the memory estimator."""
        return self._estimator


class DenseAlgorithmAdapter(BaseAlgorithmAdapter):
    """
    Adapter for dense matrix algorithms (AGD, standard BiGAMP).
    
    These algorithms use full N1×N2 matrices for masks and intermediates.
    """
    
    def __init__(self, algorithm: Any, algorithm_key: str = ""):
        super().__init__(algorithm)
        self.algorithm_key = algorithm_key or self._detect_algorithm_key()
    
    def _detect_algorithm_key(self) -> str:
        """Detect algorithm key from algorithm class name."""
        class_name = type(self.algorithm).__name__.lower()
        if 'agd' in class_name:
            return 'agd'
        elif 'bigamp' in class_name:
            return 'bigamp'
        return 'unknown'
    
    def create_estimation_params(
        self,
        N1: int,
        N2: int,
        M: int,
        S: int,
        alpha_values: List[float],
        **kwargs,
    ) -> EstimationParams:
        """Create EstimationParams for dense algorithms."""
        return EstimationParams(
            N1=N1,
            N2=N2,
            M=M,
            S=S,
            alpha_values=alpha_values,
            algorithm_key=self.algorithm_key,
            use_compile=kwargs.get('use_compile', True),
            use_bf16=kwargs.get('use_bf16', True),
            f_distribution='rademacher',  # Not used for dense algorithms
        )
    
    def run_with_plan(
        self,
        plan: ExecutionPlan,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: Optional[torch.Tensor],
        alpha_values: List[float],
        seed: int,
        step_callback: Optional[Callable] = None,
        sample_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute dense algorithm according to plan.
        
        For dense algorithms, we can directly use train_batch_alphas
        as they already handle batching internally.
        """
        # Dense algorithms have simpler memory model - typically run all at once
        return self.algorithm.train_batch_alphas(
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            Y_teacher=Y_teacher,
            masks=masks,
            alpha_values=alpha_values,
            seed=seed,
            step_callback=step_callback,
            sample_callback=sample_callback,
            **kwargs,
        )


class SpreadingAlgorithmAdapter(BaseAlgorithmAdapter):
    """
    Adapter for spreading algorithms (bigamp_spreading, bigamp_spreading_parallel).
    
    These algorithms use sparse edge-based operations and support
    different F distributions (gaussian, rademacher).
    """
    
    def __init__(
        self,
        algorithm: Any,
        algorithm_key: str = "bigamp_spreading_parallel",
        f_distribution: str = "rademacher",
    ):
        super().__init__(algorithm)
        self.algorithm_key = algorithm_key
        self.f_distribution = f_distribution
    
    def create_estimation_params(
        self,
        N1: int,
        N2: int,
        M: int,
        S: int,
        alpha_values: List[float],
        **kwargs,
    ) -> EstimationParams:
        """Create EstimationParams for spreading algorithms."""
        return EstimationParams(
            N1=N1,
            N2=N2,
            M=M,
            S=S,
            alpha_values=alpha_values,
            algorithm_key=self.algorithm_key,
            use_compile=kwargs.get('use_compile', True),
            use_bf16=kwargs.get('use_bf16', True),
            f_distribution=kwargs.get('f_distribution', self.f_distribution),
        )
    
    def run_with_plan(
        self,
        plan: ExecutionPlan,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: Optional[torch.Tensor],
        alpha_values: List[float],
        seed: int,
        step_callback: Optional[Callable] = None,
        sample_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute spreading algorithm according to plan.
        
        Uses the plan's batching to guide execution.
        """
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.algorithm.config.training.samples_per_alpha
        A = len(alpha_values)
        device = self.algorithm.device
        
        # Allocate result tensors
        W_result = torch.zeros(A, S, N1, M, device=device)
        X_result = torch.zeros(A, S, M, N2, device=device)
        
        # For spreading_parallel, create global spreading data once
        if hasattr(self.algorithm, 'create_spreading_data'):
            spreading_data = self.algorithm.create_spreading_data(
                W_teacher, X_teacher, alpha_values, S, seed
            )
        else:
            spreading_data = None
        
        # Execute batches according to plan
        for batch_idx, batch in enumerate(plan.batches):
            alpha_start, alpha_end = batch.alpha_range
            batch_alpha_indices = list(range(alpha_start, alpha_end))
            batch_alpha_list = [alpha_values[i] for i in batch_alpha_indices]
            
            # Notify UI
            if sample_callback:
                sample_callback(batch_idx, plan.num_batches, batch_alpha_list)
            
            logger.debug(
                f"Executing batch {batch_idx + 1}/{plan.num_batches}: "
                f"alphas {alpha_start}-{alpha_end}"
            )
            
            # Execute based on algorithm type
            if hasattr(self.algorithm, 'train_full_parallel') and spreading_data:
                # spreading_parallel path
                W_batch, X_batch = self.algorithm.train_full_parallel(
                    spreading_data,
                    batch_alpha_indices=batch_alpha_indices,
                    verbose=False,
                    step_callback=step_callback,
                )
                # W_batch: (S, B, N1, M) -> (B, S, N1, M)
                W_result[alpha_start:alpha_end] = W_batch.transpose(0, 1)
                X_result[alpha_start:alpha_end] = X_batch.transpose(0, 1)
            else:
                # Standard train_batch_alphas path
                W_batch, X_batch = self.algorithm.train_batch_alphas(
                    W_teacher=W_teacher,
                    X_teacher=X_teacher,
                    Y_teacher=Y_teacher,
                    masks=masks,
                    alpha_values=batch_alpha_list,
                    seed=seed + batch_idx,  # Different seed per batch
                    step_callback=step_callback,
                )
                W_result[alpha_start:alpha_end] = W_batch
                X_result[alpha_start:alpha_end] = X_batch
            
            # Clean up between batches
            if batch_idx < plan.num_batches - 1:
                torch.cuda.empty_cache()
        
        return W_result, X_result


# =============================================================================
# Factory Functions
# =============================================================================

def get_adapter(algorithm: Any) -> BaseAlgorithmAdapter:
    """
    Factory function to create appropriate adapter for algorithm.
    
    Args:
        algorithm: Algorithm instance
        
    Returns:
        Appropriate adapter instance
    """
    class_name = type(algorithm).__name__.lower()
    
    # Detect algorithm type
    if 'spreading' in class_name and 'parallel' in class_name:
        return SpreadingAlgorithmAdapter(algorithm, 'bigamp_spreading_parallel')
    elif 'spreading' in class_name:
        return SpreadingAlgorithmAdapter(algorithm, 'bigamp_spreading')
    elif 'agd' in class_name:
        return DenseAlgorithmAdapter(algorithm, 'agd')
    elif 'bigamp' in class_name:
        return DenseAlgorithmAdapter(algorithm, 'bigamp')
    else:
        # Default to dense
        logger.warning(f"Unknown algorithm type: {class_name}, using DenseAdapter")
        return DenseAlgorithmAdapter(algorithm)


def create_adapter(
    algorithm: Any,
    algorithm_key: Optional[str] = None,
    f_distribution: str = 'rademacher',
) -> BaseAlgorithmAdapter:
    """
    Create adapter with explicit configuration.
    
    Args:
        algorithm: Algorithm instance
        algorithm_key: Override algorithm key detection
        f_distribution: F distribution for spreading algorithms
        
    Returns:
        Configured adapter instance
    """
    if algorithm_key:
        if 'spreading' in algorithm_key:
            return SpreadingAlgorithmAdapter(algorithm, algorithm_key, f_distribution)
        else:
            return DenseAlgorithmAdapter(algorithm, algorithm_key)
    
    return get_adapter(algorithm)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_memory_estimate(
    N1: int,
    N2: int,
    M: int,
    S: int,
    alpha_values: List[float],
    algorithm_key: str = "bigamp_spreading_parallel",
    f_distribution: str = "rademacher",
    use_compile: bool = True,
    use_bf16: bool = True,
) -> float:
    """
    Quick memory estimation helper.
    
    Args:
        N1, N2: Matrix dimensions
        M: Hidden dimension
        S: Number of samples
        alpha_values: Alpha values
        algorithm_key: Algorithm name
        f_distribution: 'gaussian' or 'rademacher'
        use_compile: Whether torch.compile is used
        use_bf16: Whether BF16 is used
        
    Returns:
        Estimated memory in GB
    """
    estimator = MemoryEstimator()
    params = EstimationParams(
        N1=N1,
        N2=N2,
        M=M,
        S=S,
        alpha_values=alpha_values,
        algorithm_key=algorithm_key,
        use_compile=use_compile,
        use_bf16=use_bf16,
        f_distribution=f_distribution,
    )
    estimate = estimator.estimate(params)
    return estimate.total_gb


def compare_f_distributions(
    N1: int,
    N2: int,
    M: int,
    S: int,
    alpha_values: List[float],
    algorithm_key: str = "bigamp_spreading_parallel",
) -> Dict[str, float]:
    """
    Compare memory requirements for different F distributions.
    
    Args:
        Parameters as above
        
    Returns:
        Dict with 'gaussian' and 'rademacher' memory estimates
    """
    return {
        'gaussian': get_memory_estimate(
            N1, N2, M, S, alpha_values, algorithm_key, 'gaussian'
        ),
        'rademacher': get_memory_estimate(
            N1, N2, M, S, alpha_values, algorithm_key, 'rademacher'
        ),
    }
