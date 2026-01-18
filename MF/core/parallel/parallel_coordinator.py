"""
Parallel Coordinator for Smart Parallel Module.

Provides intelligent execution planning and coordination for parallel algorithms,
with support for:
- Multiple parallel modes (LINEAR, SAMPLE_PARALLEL, ALPHA_PARALLEL, HYBRID, FULL_PARALLEL)
- Dynamic batch sizing based on memory constraints
- Conservative allocation with large safety buffers (user-requested feature)
- Runtime memory monitoring integration
"""
from typing import List, Callable, Optional, Any, Dict
from dataclasses import replace
import logging
import gc
import torch

from .execution_modes import (
    ParallelMode,
    EstimationParams,
    BatchConfig,
    AllocationConfig,
    AllocationPresets,
    ExecutionPlan,
)
from .memory_estimator import MemoryEstimator

logger = logging.getLogger(__name__)


class ParallelCoordinator:
    """
    Parallel execution coordinator.
    
    Plans and executes algorithms with intelligent memory management and
    automatic batching based on GPU memory constraints.
    
    Design Philosophy:
    - Precise estimation: Calculate memory requirements accurately
    - Conservative allocation: Use only 50-60% of available memory (user-requested)
    - Runtime monitoring: Detect and recover from memory pressure
    
    Example:
        coordinator = ParallelCoordinator(
            estimator=MemoryEstimator(),
            config=AllocationPresets.CONSERVATIVE  # Uses 55% allocation
        )
        
        params = EstimationParams(...)
        plan = coordinator.plan_execution(params)
        results = coordinator.execute(plan, algorithm)
    """
    
    def __init__(
        self,
        estimator: Optional[MemoryEstimator] = None,
        config: Optional[AllocationConfig] = None,
    ):
        """
        Initialize coordinator.
        
        Args:
            estimator: Memory estimator instance. Creates default if None.
            config: Allocation configuration. Uses CONSERVATIVE preset if None.
        """
        self.estimator = estimator or MemoryEstimator()
        self.config = config or AllocationPresets.CONSERVATIVE
        
        # Runtime state
        self.current_plan: Optional[ExecutionPlan] = None
        self.current_batch_idx: int = 0
        self._abort_flag: bool = False
        self._memory_guard = None  # Set by set_memory_guard()
        
        # Statistics
        self.stats = {
            "plans_created": 0,
            "batches_executed": 0,
            "oom_recoveries": 0,
        }
    
    def plan_execution(self, params: EstimationParams) -> ExecutionPlan:
        """
        Create execution plan for given parameters.
        
        Uses conservative allocation based on config.allocation_ratio to ensure
        sufficient buffer for torch.compile dynamic caches and memory spikes.
        
        Args:
            params: Estimation parameters
            
        Returns:
            ExecutionPlan with batches and estimated memory
        """
        available_gb = self._get_available_memory()
        
        # Apply maximum allocation limit if configured
        if self.config.max_allocation_gb is not None:
            available_gb = min(available_gb, self.config.max_allocation_gb)
        
        # Conservative allocation: use only allocation_ratio of available memory
        target_gb = available_gb * self.config.allocation_ratio
        
        logger.info(
            f"Planning execution: available={available_gb:.1f}GB, "
            f"target={target_gb:.1f}GB (ratio={self.config.allocation_ratio:.0%})"
        )
        # DEBUG: Print to terminal
        
        
        S = params.S
        A = len(params.alpha_values)
        
        # Strategy 1: Try full parallel
        full_estimate = self.estimator.estimate(params)
        
        if full_estimate.total_gb <= target_gb:
            plan = ExecutionPlan(
                mode=ParallelMode.FULL_PARALLEL,
                batches=[BatchConfig(
                    sample_range=(0, S),
                    alpha_range=(0, A),
                    estimated_memory_gb=full_estimate.total_gb,
                    alpha_values=params.alpha_values,
                )],
                total_estimated_memory_gb=full_estimate.total_gb,
                allocation_config=self.config,
                algorithm_key=params.algorithm_key,
                gpu_model=self.estimator.gpu_model,
                available_memory_gb=available_gb,
            )
            logger.info(f"Selected FULL_PARALLEL mode: {full_estimate.total_gb:.1f}GB")
            self.stats["plans_created"] += 1
            return plan
        
        # Strategy 2: Fixed S, split Alpha (prefer keeping Sample parallel)
        batches = self._compute_alpha_batches(params, target_gb)
        
        if batches:
            total_mem = max(b.estimated_memory_gb for b in batches)
            plan = ExecutionPlan(
                mode=ParallelMode.HYBRID,
                batches=batches,
                total_estimated_memory_gb=total_mem,
                allocation_config=self.config,
                algorithm_key=params.algorithm_key,
                gpu_model=self.estimator.gpu_model,
                available_memory_gb=available_gb,
            )
            logger.info(
                f"Selected HYBRID mode: {len(batches)} batches, "
                f"peak={total_mem:.1f}GB"
            )
            self.stats["plans_created"] += 1
            return plan
        
        # Strategy 3: Reduce S and split Alpha
        for s in range(S - 1, 0, -1):
            reduced_params = replace(params, S=s)
            batches = self._compute_alpha_batches(reduced_params, target_gb)
            
            if batches:
                # Need multiple rounds for different sample groups
                logger.info(f"Reducing samples from {S} to {s} for memory fit")
                total_mem = max(b.estimated_memory_gb for b in batches)
                plan = ExecutionPlan(
                    mode=ParallelMode.ALPHA_PARALLEL,
                    batches=batches,
                    total_estimated_memory_gb=total_mem,
                    allocation_config=self.config,
                    algorithm_key=params.algorithm_key,
                    gpu_model=self.estimator.gpu_model,
                    available_memory_gb=available_gb,
                )
                self.stats["plans_created"] += 1
                return plan
        
        # Strategy 4: Fallback to linear
        logger.warning("Falling back to LINEAR mode due to memory constraints")
        linear_plan = self._create_linear_plan(params, available_gb)
        
        # Strategy 5: Check if even the minimum (S=1, single alpha) fits
        # If not, the data is simply too large for this GPU
        if linear_plan.batches:
            min_batch_mem = min(b.estimated_memory_gb for b in linear_plan.batches)
            if min_batch_mem > target_gb:
                # Even linear mode can't fit - problem is too large
                total_gb = self._get_total_memory()
                error_msg = (
                    f"ERROR: Problem size too large for GPU!\n"
                    f"  Minimum memory required: {min_batch_mem:.2f} GB\n"
                    f"  Available allocation: {target_gb:.2f} GB (ratio={self.config.allocation_ratio:.0%})\n"
                    f"  GPU available: {available_gb:.2f} GB (after reserved)\n"
                    f"  GPU total: {total_gb:.2f} GB\n"
                    f"  Parameters: N1={params.N1}, N2={params.N2}, M={params.M}\n"
                    f"\n"
                    f"  Suggestions:\n"
                    f"    1. Reduce matrix size (N1, N2, or M)\n"
                    f"    2. Use BF16 precision (halves memory usage)\n"
                    f"    3. Use a GPU with more VRAM\n"
                    f"    4. Try 'smf resume' after freeing GPU memory"
                )
                logger.error(error_msg)
                raise MemoryError(error_msg)
        
        return linear_plan
    
    def execute(
        self,
        plan: ExecutionPlan,
        algorithm: Any,
        step_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute plan with the given algorithm.
        
        Args:
            plan: Execution plan from plan_execution()
            algorithm: Algorithm instance with run_batch() method
            step_callback: Optional callback(batch_idx, step, metrics)
            
        Returns:
            Dict mapping alpha values to results
        """
        self.current_plan = plan
        self.current_batch_idx = 0
        self._abort_flag = False
        
        results = {}
        
        logger.info(f"Executing plan: {plan.mode.name}, {plan.num_batches} batches")
        
        # Start memory guard if configured
        if self._memory_guard:
            self._memory_guard.start()
        
        try:
            for batch_idx, batch in enumerate(plan.batches):
                if self._abort_flag:
                    logger.warning(f"Execution aborted at batch {batch_idx}")
                    break
                
                self.current_batch_idx = batch_idx
                
                # Execute single batch
                batch_result = self._execute_batch(
                    batch, algorithm, step_callback
                )
                results.update(batch_result)
                
                # Inter-batch cleanup
                self._cleanup_between_batches()
                
                self.stats["batches_executed"] += 1
                
        finally:
            if self._memory_guard:
                self._memory_guard.stop()
            
            self.current_plan = None
        
        return results
    
    def abort_current_batch(self) -> None:
        """Signal to abort current batch execution."""
        self._abort_flag = True
        logger.warning("Abort signal received")
    
    def replan_with_safety(self, factor: float = 0.7) -> ExecutionPlan:
        """
        Create a more conservative plan after OOM.
        
        Args:
            factor: Multiplier for allocation ratio (< 1.0)
            
        Returns:
            New execution plan with reduced memory usage
        """
        if self.current_plan is None:
            raise RuntimeError("No current plan to replan from")
        
        # Create more conservative config
        new_config = replace(
            self.config,
            allocation_ratio=self.config.allocation_ratio * factor
        )
        
        old_config = self.config
        self.config = new_config
        
        try:
            # Reconstruct params from current plan
            params = EstimationParams(
                N1=0,  # These need to be passed differently
                N2=0,
                M=0,
                S=0,
                alpha_values=[],
                algorithm_key=self.current_plan.algorithm_key,
            )
            # Note: Full replan would need original params, this is a simplified version
            return self.current_plan  # Placeholder
        finally:
            self.config = old_config
    
    def set_memory_guard(self, guard) -> None:
        """Set the memory guard for runtime monitoring."""
        self._memory_guard = guard
    
    def _compute_alpha_batches(
        self, 
        params: EstimationParams, 
        target_gb: float
    ) -> Optional[List[BatchConfig]]:
        """
        Compute alpha batches using greedy algorithm.
        
        Groups alphas to fill target memory, using dynamic sizing based on
        alpha_max per batch (since memory scales with alpha).
        """
        alpha_values = sorted(params.alpha_values)
        if not alpha_values:
            return None
        
        batches = []
        current_start = 0
        n = len(alpha_values)
        
        while current_start < n:
            # Greedy: try to fit as many alphas as possible
            best_end = current_start + 1
            
            for end in range(n, current_start, -1):
                batch_alphas = alpha_values[current_start:end]
                batch_alpha_max = max(batch_alphas) if batch_alphas else 0.1
                
                # Skip zero alpha (no edges, negligible memory)
                if batch_alpha_max <= 0:
                    best_end = end
                    break
                
                batch_params = replace(
                    params,
                    alpha_values=batch_alphas,
                )
                
                estimate = self.estimator.estimate(batch_params)
                
                if estimate.total_gb <= target_gb:
                    logger.info(
                        f"Batch selected: alphas={len(batch_alphas)}, α_max={batch_alpha_max:.2f}, "
                        f"estimate={estimate.total_gb:.1f}GB <= target={target_gb:.1f}GB"
                    )
                    batches.append(BatchConfig(
                        sample_range=(0, params.S),
                        alpha_range=(current_start, end),
                        estimated_memory_gb=estimate.total_gb,
                        alpha_values=batch_alphas,
                    ))
                    current_start = end
                    break
            else:
                # Cannot fit even single alpha - need to reduce S or fail
                return None
        
        return batches if batches else None
    
    def _create_linear_plan(
        self, 
        params: EstimationParams, 
        available_gb: float
    ) -> ExecutionPlan:
        """Create fully sequential execution plan."""
        batches = []
        
        for i, alpha in enumerate(params.alpha_values):
            single_params = replace(params, alpha_values=[alpha], S=1)
            estimate = self.estimator.estimate(single_params)
            
            batches.append(BatchConfig(
                sample_range=(0, 1),
                alpha_range=(i, i + 1),
                estimated_memory_gb=estimate.total_gb,
                alpha_values=[alpha],
            ))
        
        total_mem = max(b.estimated_memory_gb for b in batches) if batches else 0
        
        return ExecutionPlan(
            mode=ParallelMode.LINEAR,
            batches=batches,
            total_estimated_memory_gb=total_mem,
            allocation_config=self.config,
            algorithm_key=params.algorithm_key,
            gpu_model=self.estimator.gpu_model,
            available_memory_gb=available_gb,
        )
    
    def _execute_batch(
        self,
        batch: BatchConfig,
        algorithm: Any,
        step_callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """Execute a single batch."""
        logger.debug(
            f"Executing batch: alphas={batch.alpha_range}, "
            f"samples={batch.sample_range}, "
            f"estimated={batch.estimated_memory_gb:.1f}GB"
        )
        
        # Call algorithm's batch execution method
        if hasattr(algorithm, 'run_batch'):
            return algorithm.run_batch(
                alpha_values=batch.alpha_values,
                sample_range=batch.sample_range,
                step_callback=step_callback,
            )
        elif hasattr(algorithm, 'train_batch_alphas'):
            # Legacy interface
            return algorithm.train_batch_alphas(
                alpha_values=batch.alpha_values,
                step_callback=step_callback,
            )
        else:
            raise TypeError(
                f"Algorithm {type(algorithm).__name__} must have "
                "'run_batch' or 'train_batch_alphas' method"
            )
    
    def _cleanup_between_batches(self) -> None:
        """Clean up GPU memory between batches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _get_available_memory(self, use_total: bool = False) -> float:
        """Get available GPU memory in GB.
        
        Args:
            use_total: If True, return physical total memory (for OOM recovery).
                       If False, return available memory (total - reserved).
        
        Subtracts fixed CUDA overhead (~500MB) for context, compile cache, etc.
        This is the ONLY place where overhead is accounted for.
        """
        if use_total:
            # Use physical total memory - useful for OOM recovery replanning
            raw_memory = self.estimator.get_total_memory() or 8.0
        else:
            raw_memory = self.estimator.get_available_memory() or 8.0  # Default for CPU
        CUDA_OVERHEAD_GB = 0.5  # Fixed 500MB for CUDA context
        return max(raw_memory - CUDA_OVERHEAD_GB, 1.0)
    
    def _get_total_memory(self) -> float:
        """Get total GPU memory in GB (physical total, not available)."""
        return self.estimator.get_total_memory() or 8.0


class BatchExecutionContext:
    """
    Context manager for batch execution with automatic cleanup.
    
    Usage:
        with BatchExecutionContext(coordinator, batch):
            # Execute batch operations
            result = algorithm.run(batch)
    """
    
    def __init__(self, coordinator: ParallelCoordinator, batch: BatchConfig):
        self.coordinator = coordinator
        self.batch = batch
        self._initial_memory = 0.0
    
    def __enter__(self):
        """Record initial memory state."""
        if torch.cuda.is_available():
            self._initial_memory = torch.cuda.memory_allocated() / (1024**3)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up and record memory usage."""
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Record actual usage for calibration
            if self.coordinator.estimator and peak_memory > 0:
                logger.debug(
                    f"Batch memory: estimated={self.batch.estimated_memory_gb:.2f}GB, "
                    f"actual_peak={peak_memory:.2f}GB"
                )
            
            # Reset peak stats for next batch
            torch.cuda.reset_peak_memory_stats()
        
        # Clean up
        self.coordinator._cleanup_between_batches()
        
        return False  # Don't suppress exceptions
