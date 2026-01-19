"""
BiG-AMP for N-dimensional tensor CP decomposition.

This module provides the main algorithm class for tensor spreading,
implementing BiG-AMP message passing for n-dimensional tensor 
factorization with random spreading coefficients.

Key features:
- Supports arbitrary tensor order n (n=2 reduces to matrix factorization)
- Simplified architecture: single-alpha, single-sample training
- Onsager correction optional (default OFF)
- Rademacher or Gaussian spreading coefficients
- Fully integrated with runner via AlgorithmBase interface
"""

import math
import torch
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from .tensor_data import TensorHypergraph, TensorSpreadingData
from .tensor_step import tensor_step, forward_pass_tensor
from .tensor_hypergraph import generate_tensor_hypergraph, generate_tensor_observations

from matrix_factorization.modules.registry import register_algorithm
from matrix_factorization.modules.algorithms.base import AlgorithmBase


@dataclass
class TensorSpreadingConfig:
    """Configuration for tensor spreading algorithm."""
    tensor_order: int = 3
    dims: Tuple[int, ...] = (50, 50, 50)
    M: int = 20
    max_steps: int = 200
    damping: float = 0.5
    noise_var: float = 1e-6
    f_distribution: str = 'rademacher'
    onsager_correction: bool = False  # Default OFF per user requirement


@register_algorithm(
    key="bigamp_tensor",
    name="BiG-AMP Tensor Spreading",
    description="N-dimensional tensor CP decomposition with random spreading",
    default_params={'damping': 0.5, 'noise_var': 1e-6},
)
class BiGAMPTensorSpreading(AlgorithmBase):
    """
    BiG-AMP for N-dimensional tensor CP decomposition.
    
    This algorithm extends the matrix BiG-AMP spreading to n-dimensional
    tensors. Fully integrated with runner via AlgorithmBase interface.
    
    Supports two initialization modes:
    1. From runner: __init__(config, device) - config has matrix, training, etc.
    2. Direct use: Use class methods with explicit parameters
    """
    
    def __init__(self, config, device: torch.device):
        """
        Initialize from runner config or direct parameters.
        
        Args:
            config: Either a full config object (from runner) or tensor_order int (legacy)
            device: torch device
        """
        # Check if config is a full config object or legacy integer
        if hasattr(config, 'matrix'):
            # === From runner ===
            self.config = config
            self.device = device
            
            # Get tensor_order from spreading config
            self.order = getattr(config.spreading, 'tensor_order', 3) if hasattr(config, 'spreading') else 3
            
            # N维张量：目前假设所有维度相等 (除了 matrix.M)
            if config.matrix.N1 != config.matrix.N2:
                # 如果 N1 != N2，目前 tensor 实现可能会有维度不匹配问题
                # (因为 factors[1] 是 X.T (N2, M)，但 dims 假设全是 N1)
                import logging
                logging.getLogger(__name__).warning(
                    f"BiGAMPTensorSpreading currently assumes isotropic dimensions (N1==N2). "
                    f"Got N1={config.matrix.N1}, N2={config.matrix.N2}. "
                    f"Will use N1 for all tensor dimensions."
                )
            
            N = config.matrix.N1
            # assert config.matrix.N1 == config.matrix.N2, "BiGAMPTensorSpreading requires N1 == N2"
            self.dims = tuple([N] * self.order)
            self.M = config.matrix.M
            self.max_steps = config.training.max_steps
            self.S = config.training.samples_per_alpha
            
            # Algorithm params
            self.damping = config.algorithm_params.damping
            self.noise_var = config.algorithm_params.noise_var
            
            # Spreading params
            if hasattr(config, 'spreading') and config.spreading:
                self.f_distribution = config.spreading.f_distribution
                self.onsager_correction = config.spreading.onsager_correction
            else:
                self.f_distribution = 'rademacher'
                self.onsager_correction = False
                
        elif isinstance(config, int):
            # === Legacy direct call (tensor_order as first arg) ===
            self.config = None
            self.device = device
            self.order = config
            self.dims = tuple([50] * self.order)
            self.M = 20
            self.max_steps = 200
            self.S = 1
            self.damping = 0.5
            self.noise_var = 1e-6
            self.f_distribution = 'rademacher'
            self.onsager_correction = False
        else:
            raise ValueError(f"config must be a config object or int, got {type(config)}")
        
        # Validate dims matches order
        if len(self.dims) != self.order:
            raise ValueError(f"dims length {len(self.dims)} must match tensor_order {self.order}")

    # =========================================================================
    # AlgorithmBase Interface Implementation
    # =========================================================================
    
    def train_single_alpha(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AlgorithmBase interface: Train for single alpha.
        
        For tensor mode, W_teacher and X_teacher are used to construct
        n-dimensional teacher factors.
        """
        # Create teacher factors from W, X
        teacher_factors = self._create_teacher_factors(W_teacher, X_teacher)
        
        # Train single sample
        result = self._train_single_internal(teacher_factors, alpha, seed, self.device)
        
        # Return dummy W, X tensors (metrics computed in train_batch_alphas)
        W_students = torch.zeros(self.S, self.dims[0], self.M, device=self.device)
        X_students = torch.zeros(self.S, self.M, self.dims[1] if self.order >= 2 else self.dims[0], device=self.device)
        
        # Store Q_Y in instance for metrics retrieval
        self._last_result = result
        
        return W_students, X_students
    
    def train_batch_alphas(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: torch.Tensor,
        alpha_values: List[float],
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        step_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AlgorithmBase interface: Train for multiple alpha values.
        
        This is the main entry point called by runner.
        """
        A = len(alpha_values)
        
        # Create teacher factors
        teacher_factors = self._create_teacher_factors(W_teacher, X_teacher)
        
        # Storage for results
        W_all = torch.zeros(A, self.S, self.dims[0], self.M, device=self.device)
        X_all = torch.zeros(A, self.S, self.M, self.dims[1] if self.order >= 2 else self.dims[0], device=self.device)
        
        # Also store metrics for runner
        self._batch_metrics = {}
        
        total_work = A * self.S * self.max_steps
        completed_steps = 0
        
        for alpha_idx, alpha in enumerate(alpha_values):
            alpha_q_y_list = []
            
            for s in range(self.S):
                sample_seed = seed + s * 1000 + int(alpha * 100)
                
                result = self._train_single_internal(
                    teacher_factors, alpha, sample_seed, self.device,
                    step_callback=lambda step, total: step_callback(completed_steps + step, total_work) if step_callback else None
                )
                
                alpha_q_y_list.append(result['Q_Y'])
                completed_steps += self.max_steps
            
            # Store aggregated metrics per alpha
            self._batch_metrics[alpha] = {
                'Q_Y_mean': sum(alpha_q_y_list) / len(alpha_q_y_list),
                'Q_Y_std': (sum((q - sum(alpha_q_y_list)/len(alpha_q_y_list))**2 for q in alpha_q_y_list) / max(1, len(alpha_q_y_list)-1)) ** 0.5 if len(alpha_q_y_list) > 1 else 0.0,
            }
        
        return W_all, X_all
    
    def supports_batch_training(self) -> bool:
        """Tensor spreading does support batch training across alphas."""
        return True
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _create_teacher_factors(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Create n teacher factors from W and X matrices.
        
        For n=2: [W, X.T]
        For n>=3: [W, X.T, ...additional random factors]
        """
        factors = []
        
        # Factor 0: W (N1, M)
        factors.append(W_teacher.to(self.device))
        
        if self.order >= 2:
            # Factor 1: X.T (N2, M) - note X is (M, N2), so we transpose
            factors.append(X_teacher.T.to(self.device))
        
        # For n > 2, create additional factors
        for d in range(2, self.order):
            N_d = self.dims[d]
            # Initialize from W's statistics
            scale = W_teacher.std().item()
            torch.manual_seed(42 + d)
            factor_d = torch.randn(N_d, self.M, device=self.device) * scale
            factors.append(factor_d)
        
        return factors
    
    def _train_single_internal(
        self,
        teacher_factors: List[torch.Tensor],
        alpha: float,
        seed: int,
        device: torch.device,
        verbose: bool = False,
        step_callback: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Internal training for one alpha value.
        """
        n = self.order
        M = teacher_factors[0].shape[1]
        
        # Move teacher to device
        teacher_factors = [t.to(device) for t in teacher_factors]
        
        # Generate hypergraph
        hg = generate_tensor_hypergraph(self.dims, alpha, M, seed, device)
        
        # Generate F and Y
        F, Y = generate_tensor_observations(
            teacher_factors, hg, seed + 1000, device, self.f_distribution
        )
        
        # Initialize student (random)
        factors = [torch.randn_like(t) * 0.1 for t in teacher_factors]
        factor_vars = [torch.ones_like(t) for t in teacher_factors]
        
        prev_s = None
        is_rademacher = (self.f_distribution == 'rademacher')
        
        # BiG-AMP iterations
        for step in range(self.max_steps):
            factors, factor_vars, prev_s = tensor_step(
                factors, factor_vars, Y, F, hg.indices,
                damping=self.damping,
                noise_var=self.noise_var,
                is_rademacher=is_rademacher,
                prev_s=prev_s if self.onsager_correction else None,
                onsager_correction=self.onsager_correction,
            )
            
            if step_callback and ((step + 1) % 50 == 0 or step == self.max_steps - 1):
                step_callback(step + 1, self.max_steps)
            
            if verbose and (step + 1) % 50 == 0:
                Y_pred = forward_pass_tensor(factors, F, hg.indices)
                mse = ((Y - Y_pred) ** 2).mean().item()
                print(f"  Step {step + 1}/{self.max_steps}: MSE = {mse:.6f}")
        
        # Compute final metrics
        Y_student = forward_pass_tensor(factors, F, hg.indices)
        mse = ((Y - Y_student) ** 2).mean().item()
        
        if Y.numel() > 1:
            y_var = Y.var().item() + 1e-10
        else:
            y_var = Y.abs().mean().item()**2 + 1e-10 # Fallback for single element
            
        Q_Y = max(0.0, 1.0 - mse / y_var)
        
        return {
            'Q_Y': Q_Y,
            'MSE': mse,
            'alpha': alpha,
            'C': hg.C,
        }
    
    # =========================================================================
    # Legacy Interface (for backward compatibility)
    # =========================================================================
    
    def train(
        self,
        teacher_factors: List[torch.Tensor],
        alpha_values: List[float],
        S: int,
        base_seed: int,
        device: torch.device,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Legacy train method for direct usage.
        """
        results = []
        
        for alpha_idx, alpha in enumerate(alpha_values):
            alpha_results = []
            
            for s in range(S):
                seed = base_seed + s * 1000 + int(alpha * 100)
                
                if verbose:
                    print(f"Alpha {alpha:.2f}, Sample {s + 1}/{S}")
                
                result = self._train_single_internal(
                    teacher_factors, alpha, seed, device, verbose=False
                )
                result['sample'] = s
                result['alpha_idx'] = alpha_idx
                alpha_results.append(result)
            
            q_y_mean = sum(r['Q_Y'] for r in alpha_results) / S
            if verbose:
                print(f"  Alpha {alpha:.2f}: Q_Y = {q_y_mean:.4f}")
            
            results.extend(alpha_results)
        
        return results
    
    def create_teacher(
        self,
        device: torch.device,
        seed: int = 42,
        scale: float = 0.1,
    ) -> List[torch.Tensor]:
        """Create random teacher factors."""
        torch.manual_seed(seed)
        return [
            torch.randn(N_d, self.M, device=device) * scale
            for N_d in self.dims
        ]
