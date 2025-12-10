"""
Unified BiG-AMP Algorithm Module.

Consolidates:
1. Standard BiG-AMP (Dense)
2. Spreading BiG-AMP (Sequential, Sparse)
3. Spreading BiG-AMP (Parallel, Disjoint Union)
"""

from typing import Tuple, Optional, Callable, Dict, Literal
import math
import torch

from ..registry import register_algorithm
from .base import AlgorithmBase
from ...core.config import Config

# Mode definitions
AlgorithmMode = Literal["standard", "spreading", "spreading_parallel"]


# ============================================================================
# 1. Standard Step (Dense)
# ============================================================================

def _bigamp_step_standard(
    w_hat: torch.Tensor,
    x_hat: torch.Tensor,
    w_var: torch.Tensor,
    x_var: torch.Tensor,
    Y: torch.Tensor,
    A: torch.Tensor,
    alpha_scale: float,
    damping: float,
    noise_var: float,
    M: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard BiG-AMP step."""
    # Forward pass
    z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
    w_sq = w_hat ** 2
    x_sq = x_hat ** 2
    p_var = (alpha_scale ** 2) * (
        torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq)
    )
    V = torch.clamp(p_var + noise_var, min=1e-8)
    residual = (Y - z_hat) * A
    s = residual / V

    # Update W
    tau_W = (alpha_scale ** 2) * torch.matmul(A / V, x_sq.transpose(-2, -1))
    tau_W = torch.clamp(tau_W, min=1e-8)
    w_var_new = 1.0 / (M + tau_W)
    r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
    w_hat_new = w_hat + w_var_new * r_W
    w_hat = damping * w_hat + (1 - damping) * w_hat_new
    w_var = torch.clamp(
        damping * w_var + (1 - damping) * w_var_new,
        min=1e-8, max=1.0
    )

    # Update X
    z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
    w_sq2 = w_hat ** 2
    p_var2 = (alpha_scale ** 2) * (
        torch.matmul(w_sq2, x_var) + torch.matmul(w_var, x_sq)
    )
    V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
    residual2 = (Y - z_hat2) * A
    s2 = residual2 / V2

    tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A / V2)
    tau_X = torch.clamp(tau_X, min=1e-8)
    x_var_new = 1.0 / (M + tau_X)
    r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
    x_hat_new = x_hat + x_var_new * r_X
    x_hat = damping * x_hat + (1 - damping) * x_hat_new
    x_var = torch.clamp(
        damping * x_var + (1 - damping) * x_var_new,
        min=1e-8, max=1.0
    )

    return w_hat, x_hat, w_var, x_var


# ============================================================================
# 2. Spreading Helpers (Common)
# ============================================================================

def _scatter_add_2d(
    src: torch.Tensor,
    idx: torch.Tensor,
    dim_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """Scatter-add operation for 2D tensors."""
    if dim == 0:
        out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
        idx_expanded = idx.unsqueeze(1).expand_as(src)
        out.scatter_add_(0, idx_expanded, src)
    else:
        out = torch.zeros(src.shape[0], dim_size, device=src.device, dtype=src.dtype)
        idx_expanded = idx.unsqueeze(0).expand_as(src)
        out.scatter_add_(1, idx_expanded, src)
    return out


# ============================================================================
# 3. Parallel Spreading Step (Optimized)
# ============================================================================

def bigamp_step_disjoint_union_flat(
    W_flat: torch.Tensor,      # (A, S*N1, M)
    X_flat: torch.Tensor,      # (A, S*N2, M)
    W_var_flat: torch.Tensor,
    X_var_flat: torch.Tensor,
    Y_flat: torch.Tensor,      # (S*C_max,)
    F_flat: torch.Tensor,      # (S*C_max, M)
    i_offset: torch.Tensor,
    j_offset: torch.Tensor,
    alpha_mask_exp: torch.Tensor,
    S: int,
    N1: int,
    N2: int,
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized BiG-AMP step operating on flat tensors."""
    A = W_flat.shape[0]
    M = W_flat.shape[2]
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Gather
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    W_var_sel = W_var_flat[:, i_offset, :]
    X_var_sel = X_var_flat[:, j_offset, :]
    
    # Forward pass
    F_compute = F_flat.float() if F_flat.dtype == torch.int8 else F_flat
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    Z_hat = alpha_scale * (F_exp * W_sel * X_sel).sum(dim=2)  # (A, SC)
    Z_hat = Z_hat * alpha_mask_exp.float()
    
    # Variance
    if is_rademacher:
        V = alpha_scale_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel).sum(dim=2)
        F_sq_exp = None
    else:
        F_sq_exp = F_exp.pow(2)
        V = alpha_scale_sq * (F_sq_exp * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V * alpha_mask_exp.float() + 1e-10
    
    # Residuals
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    s_values = s_values * alpha_mask_exp.float()
    
    # Scatter
    s_exp = s_values.unsqueeze(2)
    mask_exp = alpha_mask_exp.unsqueeze(2).float()
    inv_V = (1.0 / denom).unsqueeze(2)
    SC = F_flat.shape[0]
    
    # W update
    r_W_contrib = alpha_scale * F_exp * X_sel * s_exp * mask_exp
    r_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=W_flat.dtype)
    idx_W = i_offset.view(1, SC, 1).expand(A, SC, M)
    r_W.scatter_add_(1, idx_W, r_W_contrib)
    
    if is_rademacher:
        tau_W_contrib = alpha_scale_sq * X_sel.pow(2) * inv_V * mask_exp
    else:
        tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V * mask_exp
    tau_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=W_flat.dtype)
    tau_W.scatter_add_(1, idx_W, tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)
    
    W_var_new = 1.0 / (1.0 + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_flat + W_var_new * r_W
    
    # X update
    r_X_contrib = alpha_scale * F_exp * W_sel * s_exp * mask_exp
    r_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=W_flat.dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_add_(1, idx_X, r_X_contrib)
    
    if is_rademacher:
        tau_X_contrib = alpha_scale_sq * W_sel.pow(2) * inv_V * mask_exp
    else:
        tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V * mask_exp
    tau_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=W_flat.dtype)
    tau_X.scatter_add_(1, idx_X, tau_X_contrib)
    tau_X = tau_X.clamp(min=1e-10)
    
    X_var_new = 1.0 / (1.0 + tau_X)
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_flat + X_var_new * r_X
    
    # Damping
    W_flat_out = damping * W_hat_new + (1 - damping) * W_flat
    X_flat_out = damping * X_hat_new + (1 - damping) * X_flat
    W_var_out = damping * W_var_new + (1 - damping) * W_var_flat
    X_var_out = damping * X_var_new + (1 - damping) * X_var_flat
    
    # NaN protection
    W_flat_out = torch.nan_to_num(W_flat_out, nan=0.0)
    X_flat_out = torch.nan_to_num(X_flat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)
    
    return W_flat_out, X_flat_out, W_var_out, X_var_out, s_values


# ============================================================================
# 4. Main Class
# ============================================================================

@register_algorithm(
    key="unified",
    name="Unified BiG-AMP",
    description="Unified algorithm supporting Standard, Spreading, and Parallel modes",
    default_params={'damping': 0.5, 'noise_var': 1e-10, 'mode': 'standard'},
)
class BiGAMPUnified(AlgorithmBase):
    """
    Unified BiG-AMP Algorithm.
    
    Supported Modes:
    - 'standard': Standard dense matrix factorization
    - 'spreading': Random spreading (sequential)
    - 'spreading_parallel': Random spreading (parallel across alphas)
    """

    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, device)
        self.damping = config.algorithm.damping
        self.noise_var = config.algorithm.noise_var
        self.max_steps = config.training.max_steps
        self.S = getattr(config.training, 'samples_per_alpha', 1)
        self.mode = getattr(config.algorithm, 'mode', 'standard')
        
    def step_standard(
        self,
        w_hat, x_hat, w_var, x_var,
        Y, A, alpha_scale, M
    ):
        return _bigamp_step_standard(
            w_hat, x_hat, w_var, x_var,
            Y, A, alpha_scale, self.damping, self.noise_var, M
        )
    
    def step_parallel(
        self,
        W_flat, X_flat, W_var_flat, X_var_flat,
        Y_flat, F_flat, i_offset, j_offset, alpha_mask_exp,
        S, N1, N2,
        is_rademacher=False
    ):
        return bigamp_step_disjoint_union_flat(
            W_flat, X_flat, W_var_flat, X_var_flat,
            Y_flat, F_flat, i_offset, j_offset, alpha_mask_exp,
            S, N1, N2,
            self.damping, self.noise_var, is_rademacher
        )
        
    def train_single_alpha(self, *args, **kwargs):
        """Legacy interface compliance."""
        raise NotImplementedError("Use runner.run_experiment for Unified BiG-AMP")
