"""
BiG-AMP Step functions (Standard, Adaptive, General).
"""

import math
import torch
from typing import Tuple, Optional
from .core import forward_pass_parallel, compute_variance_parallel, scatter_add_parallel


def bigamp_spreading_step(
    W_hat: torch.Tensor,
    X_hat: torch.Tensor,
    W_var: torch.Tensor,
    X_var: torch.Tensor,
    Y_values: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    alpha_mask: torch.Tensor,
    damping: float,
    noise_var: float,
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single BiG-AMP step with parallel alpha processing.

    Args:
        W_hat: (A, N1, M) W mean estimates
        X_hat: (A, M, N2) X mean estimates
        W_var: (A, N1, M) W variance estimates
        X_var: (A, M, N2) X variance estimates
        Y_values: (C_max,) teacher Y values (shared across alphas)
        F: (C_max, M) spreading coefficients
        i_idx: (C_max,) row indices
        j_idx: (C_max,) column indices
        alpha_mask: (A, C_max) active edge mask
        damping: Damping factor
        noise_var: Observation noise variance
        prev_s: Previous s values for Onsager correction

    Returns:
        Updated (W_hat, X_hat, W_var, X_var, s_values)
    """
    A, N1, M = W_hat.shape
    _, _, N2 = X_hat.shape
    C_max = F.shape[0]
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # ===== Forward pass: compute predictions =====
    Z_hat = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)  # (A, C_max)

    # ===== Compute variance =====
    V = compute_variance_parallel(
        W_hat, X_hat, W_var, X_var, F, i_idx, j_idx, alpha_mask
    )  # (A, C_max)

    # ===== Compute residuals and beliefs =====
    # s = (Y - Z_hat) / (V + noise_var)
    Y_broadcast = Y_values.unsqueeze(0)  # (1, C_max)
    # Ensure denominator has minimum value for numerical stability
    denominator = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_broadcast - Z_hat) / denominator  # (A, C_max)

    # Clamp s_values to prevent explosion (critical for numerical stability)
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)

    # Apply mask
    s_values = s_values * alpha_mask.float()

    # ===== Onsager correction / Damping =====
    # REMOVED: s-damping (inconsistent with reference Wang/bigamp/train.py)
    # if prev_s is not None:
    #     s_values = damping * s_values + (1 - damping) * prev_s

    # ===== Update W =====
    # r_W[a,i,μ] = Σ_{c: i_idx[c]=i} F[c,μ] * X[a,μ,j_idx[c]] * s[a,c]
    X_sel = X_hat[:, :, j_idx].transpose(1, 2)  # (A, C_max, M)
    F_expanded = F.unsqueeze(0)  # (1, C_max, M)
    s_expanded = s_values.unsqueeze(2)  # (A, C_max, 1)

    r_W_contrib = alpha_scale * F_expanded * X_sel * s_expanded  # (A, C_max, M)
    r_W = scatter_add_parallel(r_W_contrib, i_idx, N1, alpha_mask)  # (A, N1, M)

    # tau_W = Σ_c (F²/V) * X²
    inv_V = (1.0 / denominator).unsqueeze(2)  # (A, C_max, 1)
    F_sq_expanded = F_expanded.pow(2)  # (1, C_max, M) - F² for correct variance weighting
    tau_W_contrib = alpha_scale_sq * F_sq_expanded * X_sel.pow(2) * inv_V  # (A, C_max, M)
    tau_W = scatter_add_parallel(tau_W_contrib, i_idx, N1, alpha_mask)  # (A, N1, M)
    tau_W = tau_W.clamp(min=1e-10)

    # W update with prior N(0, 1)
    W_var_new = 1.0 / (M + tau_W)  # CRITICAL FIX: M in denominator for numerical stability
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)  # Clamp r_W to prevent explosion
    W_hat_new = W_hat + W_var_new * r_W  # CRITICAL FIX: incremental update (was missing + W_hat)

    # ===== Update X =====
    # Similar logic for X
    W_sel = W_hat[:, i_idx, :]  # (A, C_max, M)

    r_X_contrib = alpha_scale * F_expanded * W_sel * s_expanded  # (A, C_max, M)
    # Need to transpose for X: aggregate by j_idx
    r_X_contrib_T = r_X_contrib.transpose(1, 2).contiguous()  # (A, M, C_max)

    # Scatter to (A, M, N2)
    r_X = torch.zeros(A, M, N2, device=W_hat.device, dtype=W_hat.dtype)
    j_idx_expanded = j_idx.long().view(1, 1, C_max).expand(A, M, C_max)
    mask_expanded_X = alpha_mask.unsqueeze(1).float()  # (A, 1, C_max)
    r_X.scatter_reduce_(2, j_idx_expanded, (r_X_contrib_T * mask_expanded_X).contiguous(), reduce="sum", include_self=True)

    tau_X_contrib = alpha_scale_sq * F_sq_expanded * W_sel.pow(2) * inv_V  # (A, C_max, M) - F² added
    tau_X_contrib_T = tau_X_contrib.transpose(1, 2).contiguous()  # (A, M, C_max)
    tau_X = torch.zeros(A, M, N2, device=W_hat.device, dtype=W_hat.dtype)
    tau_X.scatter_reduce_(2, j_idx_expanded, (tau_X_contrib_T * mask_expanded_X).contiguous(), reduce="sum", include_self=True)
    tau_X = tau_X.clamp(min=1e-10)

    X_var_new = 1.0 / (M + tau_X)  # CRITICAL FIX: M in denominator for numerical stability
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)  # Clamp r_X to prevent explosion
    X_hat_new = X_hat + X_var_new * r_X  # CRITICAL FIX: incremental update (was missing + X_hat)

    # ===== Damping =====
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var, min=1e-4, max=1.0)

    # Replace NaN with zero for numerical stability
    W_hat_out = torch.nan_to_num(W_hat_out, nan=0.0)
    X_hat_out = torch.nan_to_num(X_hat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)

    return W_hat_out, X_hat_out, W_var_out, X_var_out, s_values


def bigamp_step_disjoint_union(
    W_hat: torch.Tensor,      # (S, A, N1, M)
    X_hat: torch.Tensor,      # (S, A, M, N2)
    W_var: torch.Tensor,      # (S, A, N1, M)
    X_var: torch.Tensor,      # (S, A, M, N2)
    Y_super: torch.Tensor,    # (S, C_max)
    F_super: torch.Tensor,    # (S, C_max, M)
    i_offset: torch.Tensor,   # (S*C_max,) - precomputed offset indices
    j_offset: torch.Tensor,   # (S*C_max,) - precomputed offset indices
    alpha_mask: torch.Tensor, # (A, C_max)
    S: int,
    damping: float,
    noise_var: float,
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BiG-AMP step with Disjoint Union parallelization.

    All S samples are processed in parallel by:
    1. Flattening sample dimension into node indices (index offsetting)
    2. One large scatter_add for all S*C_max edges
    3. Reshape back to (S, A, N, M)

    This achieves true GPU parallelization across all samples.
    """
    _, A, N1, M = W_hat.shape
    N2 = X_hat.shape[3]
    C_max = F_super.shape[1]
    SC = S * C_max

    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # ===== 1. Flatten tensors for Disjoint Union =====
    # (S, A, N1, M) -> (A, S*N1, M)
    W_flat = W_hat.permute(1, 0, 2, 3).reshape(A, S * N1, M)
    X_flat = X_hat.permute(1, 0, 3, 2).reshape(A, S * N2, M)  # Note: (S,A,M,N2) -> (A,S*N2,M)
    W_var_flat = W_var.permute(1, 0, 2, 3).reshape(A, S * N1, M)
    X_var_flat = X_var.permute(1, 0, 3, 2).reshape(A, S * N2, M)

    # (S, C_max, M) -> (S*C_max, M)
    F_flat = F_super.reshape(SC, M)
    # (S, C_max) -> (S*C_max,)
    Y_flat = Y_super.reshape(SC)

    # alpha_mask: (A, C_max) -> (A, S*C_max) by repeating for each sample
    alpha_mask_exp = alpha_mask.unsqueeze(1).expand(A, S, C_max).reshape(A, SC)

    # ===== 2. Gather: one operation for all S*C_max edges =====
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    W_var_sel = W_var_flat[:, i_offset, :]  # (A, SC, M)
    X_var_sel = X_var_flat[:, j_offset, :]  # (A, SC, M)

    # ===== 3. Forward pass =====
    # Z_hat[a, sc] = (1/√M) Σ_μ F[sc,μ] W_sel[a,sc,μ] X_sel[a,sc,μ]
    Z_hat = alpha_scale * (F_flat.unsqueeze(0) * W_sel * X_sel).sum(dim=2)  # (A, SC)
    Z_hat = Z_hat * alpha_mask_exp.float()

    # ===== 4. Variance =====
    F_sq_flat = F_flat.pow(2).unsqueeze(0)  # (1, SC, M)
    # V = (1/M) Σ F² (...)
    V = alpha_scale_sq * (F_sq_flat * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V * alpha_mask_exp.float() + 1e-10

    # ===== 5. Residuals =====
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom  # (A, SC)
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    s_values = s_values * alpha_mask_exp.float()

    s_values = s_values * alpha_mask_exp.float()

    # Onsager correction
    # REMOVED: s-damping
    # if prev_s is not None:
    #     s_values = damping * s_values + (1 - damping) * prev_s

    # ===== 6. Scatter: one operation for all edges =====
    s_exp = s_values.unsqueeze(2)  # (A, SC, 1)
    mask_exp = alpha_mask_exp.unsqueeze(2).float()  # (A, SC, 1)
    F_exp = F_flat.unsqueeze(0)  # (1, SC, M)

    # W update
    r_W_contrib = alpha_scale * F_exp * X_sel * s_exp * mask_exp  # (A, SC, M)
    r_W = torch.zeros(A, S * N1, M, device=W_hat.device, dtype=W_hat.dtype)
    idx_W = i_offset.view(1, SC, 1).expand(A, SC, M)
    r_W.scatter_reduce_(1, idx_W, r_W_contrib, reduce="sum", include_self=True)

    inv_V = (1.0 / denom).unsqueeze(2)  # (A, SC, 1)
    F_sq_exp = F_exp.pow(2)  # (1, SC, M) - F² for correct variance weighting
    tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V * mask_exp
    tau_W = torch.zeros(A, S * N1, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_W.scatter_reduce_(1, idx_W, tau_W_contrib, reduce="sum", include_self=True)
    tau_W = tau_W.clamp(min=1e-10)

    W_var_new = 1.0 / (M + tau_W)  # CRITICAL FIX: M in denominator
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_flat + W_var_new * r_W  # CRITICAL FIX: incremental update (was missing + W_flat)

    # X update
    r_X_contrib = alpha_scale * F_exp * W_sel * s_exp * mask_exp  # (A, SC, M)
    r_X = torch.zeros(A, S * N2, M, device=W_hat.device, dtype=W_hat.dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_reduce_(1, idx_X, r_X_contrib, reduce="sum", include_self=True)

    tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V * mask_exp  # F² added
    tau_X = torch.zeros(A, S * N2, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_X.scatter_reduce_(1, idx_X, tau_X_contrib, reduce="sum", include_self=True)
    tau_X = tau_X.clamp(min=1e-10)

    X_var_new = 1.0 / (M + tau_X)  # CRITICAL FIX: M in denominator
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_flat + X_var_new * r_X  # CRITICAL FIX: incremental update (was missing + X_flat)

    # ===== 7. Reshape back: (A, S*N, M) -> (S, A, N, M) =====
    W_hat_new = W_hat_new.reshape(A, S, N1, M).permute(1, 0, 2, 3)
    W_var_new = W_var_new.reshape(A, S, N1, M).permute(1, 0, 2, 3)
    X_hat_new = X_hat_new.reshape(A, S, N2, M).permute(1, 0, 3, 2)  # -> (S, A, M, N2)
    X_var_new = X_var_new.reshape(A, S, N2, M).permute(1, 0, 3, 2)

    # ===== 8. Damping =====
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var, min=1e-4, max=1.0)

    # NaN protection
    W_hat_out = torch.nan_to_num(W_hat_out, nan=0.0)
    X_hat_out = torch.nan_to_num(X_hat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)

    return W_hat_out, X_hat_out, W_var_out, X_var_out, s_values


def compute_log_likelihood(Y_flat, Z_hat, V, noise_var):
    """
    Compute Log-Likelihood for each batch element (alpha).
    Y_flat: (SC,) - flattened observations (shared across batch in current implementation logic)
            Wait, Y_flat is (SC,) but Z_hat is (B, SC).
            We broadcast Y_flat to (B, SC) for per-batch calculation.
    Z_hat: (B, SC) - predicted means
    V: (B, SC) - predicted variances
    noise_var: scalar
    
    Returns:
        log_likelihood: (B,) tensor
    """
    # Y_flat (SC,) -> unsqueeze -> (1, SC) broadcasts to (B, SC)
    residual = Y_flat.unsqueeze(0) - Z_hat
    
    # Gaussian Log-Likelihood: -0.5 * log(2*pi*var) - 0.5 * (y-z)^2 / var
    # var = V + noise_var
    var = V + noise_var
    log_term = -0.5 * torch.log(2 * torch.pi * var)
    exp_term = -0.5 * (residual ** 2) / var
    
    # Sum over SC dimension (dim=1), keep Batch dimension (dim=0)
    return (log_term + exp_term).sum(dim=1)


def bigamp_step_disjoint_union_flat_adaptive(
    W_flat: torch.Tensor,      # (A, S*N1, M)
    X_flat: torch.Tensor,      # (A, S*N2, M)
    W_var_flat: torch.Tensor,  # (A, S*N1, M)
    X_var_flat: torch.Tensor,  # (A, S*N2, M)
    Y_flat: torch.Tensor,      # (S*C_max,)
    F_flat: torch.Tensor,      # (S*C_max, M)
    i_offset: torch.Tensor,    # (S*C_max,)
    j_offset: torch.Tensor,    # (S*C_max,)
    alpha_mask_exp: torch.Tensor,  # (A, S*C_max)
    S: int,
    N1: int,
    N2: int,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adaptive Step Function: Returns RAW updates + Z_hat + V for cost check.
    Does NOT apply damping.
    
    Returns:
        W_hat_new, X_hat_new, W_var_new, X_var_new, s_values, Z_hat, V
    """
    A = W_flat.shape[0]
    M = W_flat.shape[2]
    SC = F_flat.shape[0]
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Pre-cast mask
    mask_typed = alpha_mask_exp.to(W_flat.dtype)
    
    # ===== 1. Gather =====
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    W_var_sel = W_var_flat[:, i_offset, :]
    X_var_sel = X_var_flat[:, j_offset, :]
    
    # ===== 2. Forward pass =====
    F_compute = F_flat.to(W_flat.dtype)
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    
    Z_hat = alpha_scale * (F_exp * W_sel * X_sel).sum(dim=2)  # (A, SC)
    Z_hat = Z_hat * mask_typed
    
    # ===== 3. Variance =====
    if is_rademacher:
        V = alpha_scale_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel).sum(dim=2)
        F_sq_exp = None
    else:
        F_sq_exp = F_exp.pow(2)
        V = alpha_scale_sq * (F_sq_exp * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V * mask_typed + 1e-10

    # ===== 4. ONSAGER CORRECTION (Restored) =====
    # Save Z_raw (physical prediction) for Likelihood calculation before Onsager modification
    Z_raw = Z_hat.clone()  # Physical prediction: F*W*X
    if prev_s is not None:
        # Standard AMP Onsager: Cavity Z = Z_raw - V * prev_s
        # Cavity Z is used for computing residuals (s), not for Likelihood
        Z_hat = Z_hat - V * prev_s

    # ===== 5. Residuals =====
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    s_values = s_values * alpha_mask_exp.float()
    
    # ===== 6. Scatter =====
    s_exp = s_values.unsqueeze(2)
    inv_V = (1.0 / denom).unsqueeze(2)
    storage_dtype = W_flat.dtype
    mask_typed = mask_typed.unsqueeze(2)
    
    s_typed = s_exp.to(storage_dtype)
    inv_V_typed = inv_V.to(storage_dtype)
    
    # W update
    r_W_contrib = alpha_scale * F_exp * X_sel * s_typed * mask_typed
    r_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=storage_dtype)
    idx_W = i_offset.view(1, SC, 1).expand(A, SC, M)
    r_W.scatter_add_(1, idx_W, r_W_contrib)
    
    if is_rademacher:
        tau_W_contrib = alpha_scale_sq * X_sel.pow(2) * inv_V_typed * mask_typed
    else:
        tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V_typed * mask_typed
    
    tau_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=storage_dtype)
    tau_W.scatter_add_(1, idx_W, tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)
    
    W_var_new = 1.0 / (M + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_flat + W_var_new * r_W
    
    # X update
    r_X_contrib = alpha_scale * F_exp * W_sel * s_typed * mask_typed
    r_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=storage_dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_add_(1, idx_X, r_X_contrib)
    
    if is_rademacher:
        tau_X_contrib = alpha_scale_sq * W_sel.pow(2) * inv_V_typed * mask_typed
    else:
        tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V_typed * mask_typed
    
    tau_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=storage_dtype)
    tau_X.scatter_add_(1, idx_X, tau_X_contrib)
    tau_X = tau_X.clamp(min=1e-10)
    
    X_var_new = 1.0 / (M + tau_X)
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_flat + X_var_new * r_X
    
    return W_hat_new, X_hat_new, W_var_new, X_var_new, s_values, Z_raw, V


def bigamp_step_disjoint_union_flat(
    W_flat: torch.Tensor,      # (A, S*N1, M) - already flattened
    X_flat: torch.Tensor,      # (A, S*N2, M) - already flattened
    W_var_flat: torch.Tensor,  # (A, S*N1, M)
    X_var_flat: torch.Tensor,  # (A, S*N2, M)
    Y_flat: torch.Tensor,      # (S*C_max,) - flattened Y
    F_flat: torch.Tensor,      # (S*C_max, M) - flattened F
    i_offset: torch.Tensor,    # (S*C_max,) - precomputed offset indices
    j_offset: torch.Tensor,    # (S*C_max,) - precomputed offset indices
    alpha_mask_exp: torch.Tensor,  # (A, S*C_max) - expanded mask
    S: int,
    N1: int,
    N2: int,
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,  # Optimization: skip F² for Rademacher
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized BiG-AMP step operating on flat tensors.
    
    Key optimization: NO reshape/permute inside this function.
    All tensors remain in flat (A, S*N, M) format throughout.
    
    Args:
        W_flat: (A, S*N1, M) W estimates, already flattened
        X_flat: (A, S*N2, M) X estimates, already flattened  
        W_var_flat: (A, S*N1, M) W variance, flattened
        X_var_flat: (A, S*N2, M) X variance, flattened
        Y_flat: (S*C_max,) Y values, flattened
        F_flat: (S*C_max, M) F coefficients, flattened
        i_offset: (S*C_max,) row indices with sample offset
        j_offset: (S*C_max,) col indices with sample offset
        alpha_mask_exp: (A, S*C_max) expanded alpha mask
        S: number of samples
        N1, N2: original dimensions
        damping: damping factor
        noise_var: noise variance
        is_rademacher: if True, skip F² computation (F²=1)
        prev_s: previous s values (unused, kept for API compatibility)
    
    Returns:
        Updated (W_flat, X_flat, W_var_flat, X_var_flat, s_values)
        All in flat format (A, S*N, M)
    """
    A = W_flat.shape[0]
    M = W_flat.shape[2]
    SC = F_flat.shape[0]
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Pre-cast mask once for reuse (avoid multiple casts in function)
    mask_typed = alpha_mask_exp.to(W_flat.dtype)
    
    # ===== 1. Gather: one operation for all S*C_max edges =====
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    W_var_sel = W_var_flat[:, i_offset, :]  # (A, SC, M)
    X_var_sel = X_var_flat[:, j_offset, :]  # (A, SC, M)
    
    # ===== 2. Forward pass =====
    # ALWAYS convert F to compute dtype to ensure scatter_add_ dtype consistency
    # (needed for both int8 Rademacher and float32 Gaussian when using bf16 storage)
    F_compute = F_flat.to(W_flat.dtype)
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    # Direct BF16 computation (RTX 5090 native support, 2.7x faster than .float())
    Z_hat = alpha_scale * (F_exp * W_sel * X_sel).sum(dim=2)  # (A, SC)
    Z_hat = Z_hat * mask_typed
    
    # ===== 3. Variance =====
    if is_rademacher:
        # F² = 1 for Rademacher, skip pow(2) computation
        V = alpha_scale_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel).sum(dim=2)
        F_sq_exp = None  # Not needed
    else:
        F_sq_exp = F_exp.pow(2)  # (1, SC, M)
        V = alpha_scale_sq * (F_sq_exp * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V * mask_typed + 1e-10

    # ===== 4. ONSAGER CORRECTION (Non-Adaptive Path) =====
    if prev_s is not None:
        # Standard AMP Onsager: Z_hat = Z_hat - V * prev_s
        Z_hat = Z_hat - V * prev_s
    
    # ===== 5. Residuals =====
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom  # (A, SC)
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    s_values = s_values * alpha_mask_exp.float()
    
    # ===== 5. Scatter: one operation for all edges =====
    s_exp = s_values.unsqueeze(2)  # (A, SC, 1)
    inv_V = (1.0 / denom).unsqueeze(2)  # (A, SC, 1)
    
    # Direct BF16 computation (no .float() conversion - 2.7x faster)
    storage_dtype = W_flat.dtype
    # Reuse pre-casted mask (A, SC) -> (A, SC, 1)
    mask_typed = mask_typed.unsqueeze(2)
    
    s_typed = s_exp.to(storage_dtype)
    inv_V_typed = inv_V.to(storage_dtype)
    
    # W update
    r_W_contrib = alpha_scale * F_exp * X_sel * s_typed * mask_typed  # (A, SC, M)
    r_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=storage_dtype)
    idx_W = i_offset.view(1, SC, 1).expand(A, SC, M)
    r_W.scatter_add_(1, idx_W, r_W_contrib)
    
    if is_rademacher:
        tau_W_contrib = alpha_scale_sq * X_sel.pow(2) * inv_V_typed * mask_typed
    else:
        tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V_typed * mask_typed
    tau_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=storage_dtype)
    tau_W.scatter_add_(1, idx_W, tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)
    
    W_var_new = 1.0 / (M + tau_W)  # CRITICAL FIX: M in denominator
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_flat + W_var_new * r_W
    
    # X update
    r_X_contrib = alpha_scale * F_exp * W_sel * s_typed * mask_typed  # (A, SC, M)
    r_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=storage_dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_add_(1, idx_X, r_X_contrib)
    
    if is_rademacher:
        tau_X_contrib = alpha_scale_sq * W_sel.pow(2) * inv_V_typed * mask_typed
    else:
        tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V_typed * mask_typed
    tau_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=storage_dtype)
    tau_X.scatter_add_(1, idx_X, tau_X_contrib)
    tau_X = tau_X.clamp(min=1e-10)
    
    X_var_new = 1.0 / (M + tau_X)  # CRITICAL FIX: M in denominator
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_flat + X_var_new * r_X
    
    # ===== 6. Damping =====
    W_flat_out = damping * W_hat_new + (1 - damping) * W_flat
    X_flat_out = damping * X_hat_new + (1 - damping) * X_flat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var_flat, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var_flat, min=1e-4, max=1.0)
    
    # NaN protection
    W_flat_out = torch.nan_to_num(W_flat_out, nan=0.0)
    X_flat_out = torch.nan_to_num(X_flat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)
    
    return W_flat_out, X_flat_out, W_var_out, X_var_out, s_values


# Compiled version will be cached at module level
_compiled_general_kernel = None


def general_edge_kernel_chunk(
    V_a: torch.Tensor,       # (B, ChunkSize, M)
    V_b: torch.Tensor,       # (B, ChunkSize, M)
    V_a_var: torch.Tensor,   # (B, ChunkSize, M)
    V_b_var: torch.Tensor,   # (B, ChunkSize, M)
    F_chunk: torch.Tensor,   # (ChunkSize, M)
    Y_chunk: torch.Tensor,   # (ChunkSize,)
    alpha_mask_chunk: torch.Tensor,  # (B, ChunkSize)
    prev_s_chunk: Optional[torch.Tensor],  # (B, ChunkSize) or None
    alpha_scale: float,
    alpha_scale_sq: float,
    noise_var: float,
    is_rademacher: bool,
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused computational kernel for a single chunk of edges.
    Designed to be compiled by torch.compile for kernel fusion.
    
    Returns:
        s_chunk: (B, ChunkSize) - residual signals
        r_a_contrib: (B, ChunkSize, M) - message to node a
        r_b_contrib: (B, ChunkSize, M) - message to node b
        tau_a_contrib: (B, ChunkSize, M) - precision to node a
        tau_b_contrib: (B, ChunkSize, M) - precision to node b
        Z_hat: (B, ChunkSize) - predictions (for diagnostics)
    """
    # 1. Prepare Constants
    F_exp = F_chunk.to(compute_dtype).unsqueeze(0)  # (1, ChunkSize, M)
    mask_typed = alpha_mask_chunk.to(compute_dtype)  # (B, ChunkSize)
    
    # 2. Forward Pass (Z_hat)
    Z_hat = alpha_scale * (F_exp * V_a * V_b).sum(dim=2) * mask_typed
    
    # 3. Variance Calculation
    if is_rademacher:
        # F^2 = 1 for Rademacher
        V_val_raw = (V_a_var * V_b.pow(2) + V_a.pow(2) * V_b_var).sum(dim=2)
        F_sq_exp = None
    else:
        F_sq_exp = F_exp.pow(2)
        V_val_raw = (F_sq_exp * (V_a_var * V_b.pow(2) + V_a.pow(2) * V_b_var)).sum(dim=2)
        
    V_val = alpha_scale_sq * V_val_raw * mask_typed + 1e-10

    # 4. Onsager Correction
    if prev_s_chunk is not None:
        Z_hat = Z_hat - V_val * prev_s_chunk
    
    # 5. Residuals
    denom = V_val + noise_var
    denom = torch.clamp(denom, min=1e-6)
    s_chunk = (Y_chunk.unsqueeze(0) - Z_hat) / denom
    s_chunk = torch.clamp(s_chunk, min=-1e6, max=1e6) * alpha_mask_chunk.float()

    # 6. Compute Messages
    s_exp = s_chunk.to(compute_dtype).unsqueeze(2)     # (B, ChunkSize, 1)
    inv_V = (1.0 / denom).to(compute_dtype).unsqueeze(2)  # (B, ChunkSize, 1)
    mask_3d = mask_typed.unsqueeze(2)                   # (B, ChunkSize, 1)
    
    # Message to Node A: r_a = F * V_b * s
    r_a_contrib = alpha_scale * F_exp * V_b * s_exp * mask_3d
    
    # Message to Node B: r_b = F * V_a * s
    r_b_contrib = alpha_scale * F_exp * V_a * s_exp * mask_3d

    # Preconditioners (Tau)
    if is_rademacher:
        tau_common = alpha_scale_sq * inv_V * mask_3d
        tau_a_contrib = V_b.pow(2) * tau_common
        tau_b_contrib = V_a.pow(2) * tau_common
    else:
        tau_common = alpha_scale_sq * F_sq_exp * inv_V * mask_3d
        tau_a_contrib = V_b.pow(2) * tau_common
        tau_b_contrib = V_a.pow(2) * tau_common

    return s_chunk, r_a_contrib, r_b_contrib, tau_a_contrib, tau_b_contrib, Z_hat


def bigamp_step_general_chunked(
    V_flat: torch.Tensor,      # (B, S*N_total, M)
    V_var_flat: torch.Tensor,  # (B, S*N_total, M)
    Y_flat: torch.Tensor,      # (S*C_max,)
    F_flat: torch.Tensor,      # (S*C_max, M)
    a_offset: torch.Tensor,    # (S*C_max,)
    b_offset: torch.Tensor,    # (S*C_max,)
    alpha_mask_exp: torch.Tensor,  # (B, S*C_max)
    S: int,
    N_total: int,
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
    chunk_size: int = 131072,
    use_compile: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-optimized General BiG-AMP step using chunked streaming.
    
    Key features:
    - Processes edges in fixed-size chunks to bound peak memory
    - Uses index_add_ for in-place accumulation (zero residual memory)
    - Supports torch.compile for kernel fusion within each chunk
    
    Args:
        V_flat: (B, S*N_total, M) unified node vectors
        V_var_flat: (B, S*N_total, M) unified node variances
        Y_flat: (SC,) flattened observations
        F_flat: (SC, M) flattened spreading coefficients
        a_offset: (SC,) first node indices
        b_offset: (SC,) second node indices
        alpha_mask_exp: (B, SC) alpha mask
        S: number of samples
        N_total: N1 + N2
        damping: damping factor
        noise_var: noise variance
        is_rademacher: True if F is Rademacher (F²=1)
        prev_s: (B, SC) previous s values for Onsager correction
        chunk_size: edges per chunk
        use_compile: whether to use compiled kernel
        
    Returns:
        V_flat_out, V_var_out, s_new_all, Z_hat (placeholder), V_val (placeholder)
    """
    global _compiled_general_kernel
    
    B, SN, M = V_flat.shape
    SC = F_flat.shape[0]
    compute_dtype = V_flat.dtype
    device = V_flat.device
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # 1. Allocate Global Accumulators (Zero-initialized)
    # Size is only (B, Nodes, M), much smaller than Edges
    r_V = torch.zeros(B, SN, M, device=device, dtype=compute_dtype)
    tau_V = torch.zeros(B, SN, M, device=device, dtype=compute_dtype)
    
    # Container for s_values (needed for next step's Onsager)
    s_new_all = torch.empty(B, SC, device=device, dtype=torch.float32)
    
    # 2. Optional: Compile the kernel (cached at module level)
    kernel_fn = general_edge_kernel_chunk
    if use_compile and _compiled_general_kernel is None:
        try:
            torch._dynamo.reset()
            _compiled_general_kernel = torch.compile(
                general_edge_kernel_chunk,
                mode="default",  # Avoid CUDA Graphs issues
                fullgraph=False,
            )
        except Exception:
            _compiled_general_kernel = general_edge_kernel_chunk
    
    if use_compile and _compiled_general_kernel is not None:
        kernel_fn = _compiled_general_kernel
    
    # 3. Chunked Processing Loop
    num_chunks = (SC + chunk_size - 1) // chunk_size
    
    for i in range(0, SC, chunk_size):
        end = min(i + chunk_size, SC)
        chunk_len = end - i
        idx_slice = slice(i, end)
        
        # 3a. Gather Inputs for this chunk
        a_idx_chunk = a_offset[idx_slice]
        b_idx_chunk = b_offset[idx_slice]
        
        V_a = V_flat.index_select(1, a_idx_chunk)
        V_b = V_flat.index_select(1, b_idx_chunk)
        V_a_var = V_var_flat.index_select(1, a_idx_chunk)
        V_b_var = V_var_flat.index_select(1, b_idx_chunk)
        
        F_chunk = F_flat[idx_slice]
        Y_chunk = Y_flat[idx_slice]
        mask_chunk = alpha_mask_exp[:, idx_slice]
        prev_s_chunk = prev_s[:, idx_slice] if prev_s is not None else None

        # 3b. Execute Kernel
        s_chunk, r_a, r_b, tau_a, tau_b, _ = kernel_fn(
            V_a, V_b, V_a_var, V_b_var, F_chunk, Y_chunk, mask_chunk, prev_s_chunk,
            alpha_scale, alpha_scale_sq, noise_var, is_rademacher, compute_dtype
        )
        
        # 3c. Store s values
        s_new_all[:, idx_slice] = s_chunk
        
        # 3d. Scatter Aggregate (Use scatter_add_ with expanded 3D indices for better performance)
        # Expand indices for scatter_add_: (ChunkLen,) -> (B, ChunkLen, M)
        a_idx_exp = a_idx_chunk.view(1, -1, 1).expand(B, chunk_len, M)
        b_idx_exp = b_idx_chunk.view(1, -1, 1).expand(B, chunk_len, M)
        
        r_V.scatter_add_(1, a_idx_exp, r_a)
        r_V.scatter_add_(1, b_idx_exp, r_b)
        
        tau_V.scatter_add_(1, a_idx_exp, tau_a)
        tau_V.scatter_add_(1, b_idx_exp, tau_b)
        
        # Free expanded indices
        del a_idx_exp, b_idx_exp

    # 4. Final Node Updates
    tau_V = tau_V.clamp(min=1e-10)
    V_var_new = 1.0 / (M + tau_V)
    
    r_V = torch.clamp(r_V, min=-1e4, max=1e4)
    V_hat_new = V_flat + V_var_new * r_V
    
    # 5. Damping
    V_flat_out = damping * V_hat_new + (1 - damping) * V_flat
    V_var_out = torch.clamp(damping * V_var_new + (1 - damping) * V_var_flat, min=1e-4, max=1.0)
    
    # NaN protection
    V_flat_out = torch.nan_to_num(V_flat_out, nan=0.0)
    V_var_out = torch.nan_to_num(V_var_out, nan=1.0)
    
    # Return placeholders for Z_hat and V_val (not needed in training loop)
    return V_flat_out, V_var_out, s_new_all, None, None


def bigamp_step_disjoint_union_flat_general(
    V_flat: torch.Tensor,      # (A, S*N_total, M) - flattened unified vectors
    V_var_flat: torch.Tensor,  # (A, S*N_total, M)
    Y_flat: torch.Tensor,      # (S*C_max,)
    F_flat: torch.Tensor,      # (S*C_max, M)
    a_offset: torch.Tensor,    # (S*C_max,) - unified indices for first node
    b_offset: torch.Tensor,    # (S*C_max,) - unified indices for second node
    alpha_mask_exp: torch.Tensor,
    S: int,
    N_total: int,
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    General Graph optimized BiG-AMP step using Unified Vector V.
    
    V_flat contains both W and X nodes.
    a_offset and b_offset index into V_flat.
    """
    A = V_flat.shape[0]
    M = V_flat.shape[2]
    SC = F_flat.shape[0]
    SN = S * N_total
    storage_dtype = V_flat.dtype
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    mask_typed = alpha_mask_exp.to(V_flat.dtype)
    
    # ===== 1. Gather (Vectorized) =====
    # V_flat: (B, S*N_total, M)
    # Result: (B, SC, M)
    V_a = V_flat.index_select(1, a_offset)
    V_b = V_flat.index_select(1, b_offset)
    V_a_var = V_var_flat.index_select(1, a_offset)
    V_b_var = V_var_flat.index_select(1, b_offset)
    
    # ===== 2. Forward =====
    F_compute = F_flat.to(V_flat.dtype)
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    
    # Z_hat = (1/√M) Σ F * V_a * V_b
    Z_hat = alpha_scale * (F_exp * V_a * V_b).sum(dim=2) * mask_typed
    
    # ===== 3. Variance =====
    # MEMORY OPT: 预计算 F_sq_exp 用于后续 scatter update
    if is_rademacher:
        # F²=1
        V_val = alpha_scale_sq * (V_a_var * V_b.pow(2) + V_a.pow(2) * V_b_var).sum(dim=2)
        F_sq_exp = None  # Not needed for Rademacher
    else:
        F_sq_exp = F_exp.pow(2)  # 预计算，后续复用
        V_val = alpha_scale_sq * (F_sq_exp * (V_a_var * V_b.pow(2) + V_a.pow(2) * V_b_var)).sum(dim=2)
    
    # MEMORY OPT: 释放不再需要的 variance 张量
    del V_a_var, V_b_var
    
    V_val = V_val * mask_typed + 1e-10
    
    # ===== 4. Onsager =====
    if prev_s is not None:
        Z_hat = Z_hat - V_val * prev_s
        
    # ===== 5. Residuals =====
    denom = torch.clamp(V_val + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom
    s_values = torch.clamp(s_values, min=-1e6, max=1e6) * alpha_mask_exp.float()
    
    # ===== 6. Scatter Update =====
    s_exp = s_values.unsqueeze(2).to(V_flat.dtype)
    inv_V_typed = (1.0 / denom).unsqueeze(2).to(V_flat.dtype)
    mask_typed = mask_typed.unsqueeze(2)
    
    # ===== 6. Accumulation =====
    # 初始化累加器 (A, S*N_total, M)
    r_V = torch.zeros(A, SN, M, device=V_flat.device, dtype=storage_dtype)
    tau_V = torch.zeros(A, SN, M, device=V_flat.device, dtype=storage_dtype)
    
    # --- Phase A: 处理 'a' 节点更新 (使用 V_b) ---
    term = alpha_scale * F_exp * V_b * s_exp * mask_typed
    r_V.index_add_(1, a_offset, term)
    del term
    
    if is_rademacher:
        term = alpha_scale_sq * V_b.pow(2) * inv_V_typed * mask_typed
    else:
        term = alpha_scale_sq * F_sq_exp * V_b.pow(2) * inv_V_typed * mask_typed
    tau_V.index_add_(1, a_offset, term)
    del term
    
    del V_b
    
    # --- Phase B: 处理 'b' 节点更新 (使用 V_a) ---
    term = alpha_scale * F_exp * V_a * s_exp * mask_typed
    r_V.index_add_(1, b_offset, term)
    del term
    
    if is_rademacher:
        term = alpha_scale_sq * V_a.pow(2) * inv_V_typed * mask_typed
    else:
        term = alpha_scale_sq * F_sq_exp * V_a.pow(2) * inv_V_typed * mask_typed
    tau_V.index_add_(1, b_offset, term)
    del term
    
    del V_a
    
    tau_V = tau_V.clamp(min=1e-10)
    
    # Final Update
    V_var_new = 1.0 / (M + tau_V)
    
    r_V = torch.clamp(r_V, min=-1e4, max=1e4)
    V_hat_new = V_flat + V_var_new * r_V
    
    # Damping
    V_flat_out = damping * V_hat_new + (1 - damping) * V_flat
    V_var_out = torch.clamp(damping * V_var_new + (1 - damping) * V_var_flat, min=1e-4, max=1.0)
    
    V_flat_out = torch.nan_to_num(V_flat_out, nan=0.0)
    V_var_out = torch.nan_to_num(V_var_out, nan=1.0)
    
    return V_flat_out, V_var_out, s_values, Z_hat, V_val


def compute_offset_indices(
    i_idx: torch.Tensor,  # (S, C_max)
    j_idx: torch.Tensor,  # (S, C_max)
    N1: int,
    N2: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute offset indices for Disjoint Union.

    Maps each sample's local indices to global indices:
    i_offset[s, c] = s * N1 + i_idx[s, c]
    j_offset[s, c] = s * N2 + j_idx[s, c]

    Args:
        i_idx: (S, C_max) row indices per sample
        j_idx: (S, C_max) col indices per sample
        N1: number of rows
        N2: number of columns

    Returns:
        i_offset: (S*C_max,) flattened offset row indices
        j_offset: (S*C_max,) flattened offset col indices
    """
    S = i_idx.shape[0]
    device = i_idx.device

    # Sample offsets: [0, N1, 2*N1, ...]
    offsets_N1 = torch.arange(S, device=device) * N1  # (S,)
    offsets_N2 = torch.arange(S, device=device) * N2  # (S,)

    # Add offsets and flatten
    i_offset = (i_idx + offsets_N1.unsqueeze(1)).reshape(-1)  # (S*C_max,)
    j_offset = (j_idx + offsets_N2.unsqueeze(1)).reshape(-1)  # (S*C_max,)

    return i_offset, j_offset

def clear_step_cache():
    """Clear internal compiled kernels."""
    global _compiled_general_kernel
    _compiled_general_kernel = None
