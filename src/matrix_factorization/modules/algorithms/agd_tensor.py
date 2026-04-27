"""
Tensor AGD Algorithm Implementation.

This module provides an implementation of Alternating Gradient Descent (AGD)
specifically for high-order Tensor CP Decomposition.
"""

from typing import Tuple, List, Optional, Callable
import torch
import math

from ..registry import register_algorithm
from .base import AlgorithmBase
from ...core.config import Config
from ...core.experiment.config import resolve_normalization_profile


@register_algorithm(
    key="agd_tensor",
    name="Tensor AGD",
    description="Alternating Gradient Descent for Tensor CP Decomposition (AutoGrad)",
    default_params={
        'learning_rate': 0.01,
        'max_steps': 5000,
        'tensor_order': 3,
    },
)
class TensorAGD(AlgorithmBase):
    """
    Alternating Gradient Descent for Tensor CP Decomposition.
    
    Uses PyTorch AutoGrad for simplicity and correctness.
    Note: This algorithm ignores the matrix W/X passed by DataFactory and
    generates its own Tensor Teacher based on the specific dimensions and seed.
    """

    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, device)
        
        # Params
        self.lr = getattr(config.algorithm_params, 'learning_rate', 0.01)
        self.max_steps = getattr(config.algorithm_params, 'max_steps', 5000)
        self.normalization_profile = getattr(config.algorithm_params, "normalization_profile", "paper_sparse_sampling")
        
        # Tensor config
        # Try to infer n, N, M from config
        self.M = config.matrix.M
        
        # Determine order n
        self.n = 3
        if hasattr(config, 'tensor_order'):
            self.n = config.tensor_order
        elif hasattr(config.algorithm_params, 'tensor_order'):
            self.n = config.algorithm_params.tensor_order
            
        # Determine dimensions
        # Assuming N1=N2=...=N for simplicity in standard config
        # Or read from config.matrix.N1 etc.
        self.N = config.matrix.N1
        self.dims = tuple([self.N] * self.n)
        self._norm = resolve_normalization_profile(self.normalization_profile, self.M)

    def cp_contract_factors(self, factors: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute paper-scaled CP tensor:
            T = (1/sqrt(M)) sum_m outer(f0[:,m], f1[:,m], ..., fn[:,m])
        """
        n = len(factors)
        
        # Build einsum pattern: 'am,bm,cm->abc' for n=3
        letters = 'abcdefghij'[:n]
        pattern = ','.join([f'{letters[d]}m' for d in range(n)]) + '->' + letters
        
        return self._norm.interaction_scale * torch.einsum(pattern, *factors)

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
        Train AGD for a single alpha.
        
        NOTE: Ignores W/X/Y/mask arguments and regenerates Tensor data locally
        using the provided seed/alpha to ensure correctness for Order-n tensors.
        """
        device = self.device
        n = self.n
        M = self.M
        dims = self.dims
        
        # 1. Generate Teacher Factors
        # Use seed to ensure reproducibility independent of Alpha
        # (This is consistent with creating a single ground truth tensor)
        torch.manual_seed(42)  # Fixed seed for Teacher (Ground Truth)
        teacher_factors = [
            torch.randn(d, M, device=device) * self._norm.latent_std
            for d in dims
        ]
        
        # Normalize teacher factors to match the scale expected
        # typically we want E[y^2] = 1. 
        # If factors ~ N(0,1), then T ~ M * 1^n? No.
        # Let's stick to the generation logic we validated: usually factors ~ N(0,1)
        # and T_teacher is just the contraction.
        T_teacher = self.cp_contract_factors(teacher_factors)
        
        # 2. Generate Mask for this specific Alpha
        # We need a new seed component for the mask
        torch.manual_seed(seed)
        
        dof = sum(dims) * M  # Standard DOF definition
        # Or n*N*M? user uses alpha = C / (N*M) typically?
        # Let's stick to user's definition: alpha = C / (M * N)
        # N refers to N1.
        # C = alpha * M * dims[0]
        
        total_entries = 1
        for d in dims:
            total_entries *= d
            
        num_obs = int(alpha * M * dims[0])
        obs_ratio = num_obs / total_entries
        
        # Create mask
        mask_tensor = (torch.rand(dims, device=device) < obs_ratio).float()
        
        # 3. Generate Observations (Noiseless)
        Y_observed = T_teacher * mask_tensor
        
        # 4. Initialize Student
        # Random init follows the selected normalization profile.
        # Use seed+1 for student init
        torch.manual_seed(seed + 1)
        student_factors = [
            torch.nn.Parameter(torch.randn(d, M, device=device) * self._norm.student_init_std)
            for d in dims
        ]
        
        # 5. Optimizer
        optimizer = torch.optim.Adam(student_factors, lr=self.lr)
        
        # 6. Training Loop
        for step in range(self.max_steps):
            optimizer.zero_grad()
            
            # Forward
            T_student = self.cp_contract_factors(student_factors)
            
            # Loss (MSE on observed entries)
            # Avoid division by zero
            mask_sum = mask_tensor.sum()
            if mask_sum == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = ((T_student - T_teacher) ** 2 * mask_tensor).sum() / mask_sum
            
            # Backward
            loss.backward()
            
            # Clip
            torch.nn.utils.clip_grad_norm_(student_factors, max_norm=1.0)
            
            # Update
            optimizer.step()
        
        # 7. Final Result
        # Compute Q_Y metric to store in _last_result for logging
        with torch.no_grad():
            T_student = self.cp_contract_factors(student_factors)
            inner = (T_student * T_teacher).sum()
            norm_s = T_student.norm()
            norm_t = T_teacher.norm()
            q_y = (inner / (norm_s * norm_t + 1e-12)).item()
            mse = ((T_student - T_teacher) ** 2).mean().item()
            
            # Save for metrics reporting
            self._last_result = {
                'alpha': alpha,
                'Q_Y': q_y,
                'MSE': mse
            }
            
        # Return first two factors to satisfy Runner signature
        # Format: (S, N, M). Here S=1.
        w_out = student_factors[0].detach().unsqueeze(0) # (1, N, M)
        x_out = student_factors[1].detach().transpose(0, 1).unsqueeze(0) # (1, M, N) ?
        # Wait, runner expects X shape (S, M, N2). 
        # student_factors[1] is (N2, M). transpose -> (M, N2).
        
        return w_out, x_out.transpose(1, 2) 
