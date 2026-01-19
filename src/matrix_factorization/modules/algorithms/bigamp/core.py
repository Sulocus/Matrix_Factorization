"""
Core parallel operations for BiG-AMP Spreading.
"""

import math
import torch
from typing import Tuple


def forward_pass_parallel(
    W_hat: torch.Tensor,
    X_hat: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    alpha_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel forward pass across all alphas.

    Z_hat[a, c] = (1/√M) Σ_μ F[c,μ] W_hat[a, i[c], μ] X_hat[a, μ, j[c]]
                  if mask[a, c] else 0

    Args:
        W_hat: (A, N1, M) student W estimates
        X_hat: (A, M, N2) student X estimates
        F: (C_max, M) spreading coefficients for this sample
        i_idx: (C_max,) row indices
        j_idx: (C_max,) column indices
        alpha_mask: (A, C_max) boolean mask

    Returns:
        Z_hat: (A, C_max) predicted Y values
    """
    A, N1, M = W_hat.shape
    C_max = F.shape[0]
    alpha_scale = 1.0 / math.sqrt(M)

    # Ensure indices are long
    i_idx = i_idx.long()
    j_idx = j_idx.long()

    # Gather: select W and X at edge positions
    # W_sel[a, c, μ] = W_hat[a, i_idx[c], μ]
    W_sel = W_hat[:, i_idx, :]  # (A, C_max, M)

    # X_sel[a, c, μ] = X_hat[a, μ, j_idx[c]]
    X_sel = X_hat[:, :, j_idx].transpose(1, 2)  # (A, C_max, M)

    # F is (C_max, M), broadcast to (1, C_max, M)
    F_expanded = F.unsqueeze(0)

    # Element-wise multiply and sum
    Z_raw = alpha_scale * (F_expanded * W_sel * X_sel).sum(dim=2)  # (A, C_max)

    # Apply mask
    Z_hat = Z_raw * alpha_mask.float()

    return Z_hat


def compute_variance_parallel(
    W_hat: torch.Tensor,
    X_hat: torch.Tensor,
    W_var: torch.Tensor,
    X_var: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    alpha_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute prediction variance at each edge.

    Uses E[F²] = 1 approximation (exact for both Gaussian and Rademacher).

    V[a, c] = (1/M) Σ_μ (W_var[a,i,μ] X²[a,μ,j] + W²[a,i,μ] X_var[a,μ,j])

    Args:
        W_hat, X_hat: (A, N, M) mean estimates
        W_var, X_var: (A, N, M) variance estimates
        F: (C_max, M) spreading coefficients
        i_idx, j_idx: (C_max,) edge indices
        alpha_mask: (A, C_max) mask

    Returns:
        V: (A, C_max) variance at each edge
    """
    A = W_hat.shape[0]
    M = W_hat.shape[2]
    alpha_scale_sq = 1.0 / M

    # Gather values
    W_sel = W_hat[:, i_idx, :]       # (A, C_max, M)
    X_sel = X_hat[:, :, j_idx].transpose(1, 2)  # (A, C_max, M)
    W_var_sel = W_var[:, i_idx, :]   # (A, C_max, M)
    X_var_sel = X_var[:, :, j_idx].transpose(1, 2)

    # F² - use actual F² values (critical for Gaussian spreading)
    F_sq = F.pow(2).unsqueeze(0)  # (1, C_max, M)
    
    # V = (1/M) Σ_μ F² * (W_var * X² + W² * X_var)
    V_raw = alpha_scale_sq * (
        F_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)
    ).sum(dim=2)  # (A, C_max)

    # Apply mask and add small epsilon for stability
    V = V_raw * alpha_mask.float() + 1e-10

    return V


def scatter_add_parallel(
    src: torch.Tensor,
    idx: torch.Tensor,
    target_size: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel scatter_add with masking.

    result[a, n, μ] = Σ_{c: idx[c]=n, mask[a,c]=1} src[a, c, μ]

    Args:
        src: (A, C_max, M) source values
        idx: (C_max,) target indices
        target_size: N (output dimension)
        mask: (A, C_max) boolean mask

    Returns:
        result: (A, N, M)
    """
    A, C_max, M = src.shape
    result = torch.zeros(A, target_size, M, device=src.device, dtype=src.dtype)

    # Apply mask
    src_masked = src * mask.unsqueeze(2).float()

    # Expand indices for scatter: (1, C_max, 1) -> (A, C_max, M)
    # Convert to int64 for scatter_add_ (required by PyTorch)
    idx_expanded = idx.long().view(1, C_max, 1).expand(A, C_max, M)

    # Scatter reduce (Phase 1 optimization)
    result.scatter_reduce_(1, idx_expanded, src_masked.contiguous(), reduce="sum", include_self=True)

    return result
