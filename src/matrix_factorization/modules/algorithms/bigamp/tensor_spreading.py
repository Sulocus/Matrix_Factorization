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
"""

import math
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .tensor_data import TensorHypergraph, TensorSpreadingData
from .tensor_step import tensor_step, forward_pass_tensor
from .tensor_hypergraph import generate_tensor_hypergraph, generate_tensor_observations


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


class BiGAMPTensorSpreading:
    """
    BiG-AMP for N-dimensional tensor CP decomposition.
    
    This algorithm extends the matrix BiG-AMP spreading to n-dimensional
    tensors using a simplified architecture optimized for high-dimensional
    cases where single-alpha computation already saturates GPU resources.
    
    Args:
        tensor_order: n - dimension of the tensor (2=matrix, 3=3-tensor, etc.)
        dims: Tuple of n integers (N_1, ..., N_n) - factor dimensions
        M: Latent dimension
        max_steps: Maximum BiG-AMP iterations
        damping: Damping factor (0=no damping, 1=full damping)
        noise_var: Observation noise variance
        f_distribution: 'rademacher' or 'gaussian'
        onsager_correction: Whether to apply Onsager correction (default: False)
        
    Example:
        >>> # 3-dimensional tensor decomposition
        >>> algo = BiGAMPTensorSpreading(tensor_order=3, dims=(50, 50, 50), M=20)
        >>> teacher = [torch.randn(50, 20) * 0.1 for _ in range(3)]
        >>> result = algo.train_single_alpha(teacher, alpha=2.0, seed=42, device='cuda')
        >>> print(f"Q_Y: {result['Q_Y']:.4f}")
    """
    
    def __init__(
        self,
        tensor_order: int = 3,
        dims: Tuple[int, ...] = None,
        M: int = 20,
        max_steps: int = 200,
        damping: float = 0.5,
        noise_var: float = 1e-6,
        f_distribution: str = 'rademacher',
        onsager_correction: bool = False,
    ):
        self.order = tensor_order
        self.dims = dims if dims is not None else tuple([50] * tensor_order)
        self.M = M
        self.max_steps = max_steps
        self.damping = damping
        self.noise_var = noise_var
        self.f_distribution = f_distribution
        self.onsager_correction = onsager_correction
        
        # Validate dims matches order
        if len(self.dims) != self.order:
            raise ValueError(f"dims length {len(self.dims)} must match tensor_order {self.order}")
    
    def train_single_alpha(
        self,
        teacher_factors: List[torch.Tensor],
        alpha: float,
        seed: int,
        device: torch.device,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Train for one alpha value.
        
        Args:
            teacher_factors: List of n teacher factor matrices, each (N_d, M)
            alpha: Observation density (average node degree)
            seed: Random seed for reproducibility
            device: torch device
            verbose: Print progress
            
        Returns:
            Dictionary with metrics: Q_Y, MSE, alpha
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
            
            if verbose and (step + 1) % 50 == 0:
                Y_pred = forward_pass_tensor(factors, F, hg.indices)
                mse = ((Y - Y_pred) ** 2).mean().item()
                print(f"  Step {step + 1}/{self.max_steps}: MSE = {mse:.6f}")
        
        # Compute final metrics
        Y_student = forward_pass_tensor(factors, F, hg.indices)
        mse = ((Y - Y_student) ** 2).mean().item()
        y_var = Y.var().item() + 1e-10
        Q_Y = max(0.0, 1.0 - mse / y_var)
        
        return {
            'Q_Y': Q_Y,
            'MSE': mse,
            'alpha': alpha,
            'C': hg.C,
        }
    
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
        Train across multiple alpha values and samples.
        
        Args:
            teacher_factors: List of n teacher factor matrices
            alpha_values: List of alpha values to sweep
            S: Number of samples per alpha
            base_seed: Base random seed
            device: torch device
            verbose: Print progress
            
        Returns:
            List of result dictionaries, one per (alpha, sample)
        """
        results = []
        
        for alpha_idx, alpha in enumerate(alpha_values):
            alpha_results = []
            
            for s in range(S):
                # Unique seed for each (alpha, sample) combination
                seed = base_seed + s * 1000 + int(alpha * 100)
                
                if verbose:
                    print(f"Alpha {alpha:.2f}, Sample {s + 1}/{S}")
                
                result = self.train_single_alpha(
                    teacher_factors, alpha, seed, device, verbose=False
                )
                result['sample'] = s
                result['alpha_idx'] = alpha_idx
                alpha_results.append(result)
            
            # Compute alpha-level statistics
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
        """
        Create random teacher factors.
        
        Args:
            device: torch device
            seed: random seed
            scale: standard deviation of initialization
            
        Returns:
            List of n teacher factor matrices
        """
        torch.manual_seed(seed)
        return [
            torch.randn(N_d, self.M, device=device) * scale
            for N_d in self.dims
        ]
