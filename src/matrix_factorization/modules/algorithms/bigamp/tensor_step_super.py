"""
Alpha-parallel BiG-AMP step functions for N-dimensional tensor CP decomposition.

This module extends tensor_step_batch.py to support Alpha + Sample parallelization
using the TensorSuperGraph Disjoint Union format.

Key differences from tensor_step_batch.py:
- factors shape: (A, S*N_d, M) instead of (S, N_d, M)
- Uses alpha_mask to handle variable edge counts per alpha
- Supports batch processing across all alpha values

PERFORMANCE OPTIMIZATION (Phase 2):
- offset_indices are precomputed in TensorSuperGraph
- No repeated sample_offsets calculation in the training loop
- Similar to Bipartite's compute_offset_indices() approach
"""

import math
import torch
from typing import List, Tuple, Optional

from .conventions import blend_new_old, damped_state, gaussian_posterior_update


def forward_pass_tensor_super(
    factors: List[torch.Tensor],       # n tensors of (A, S*N_d, M)
    F_flat: torch.Tensor,              # (S*C_max, M)
    offset_indices: List[torch.Tensor], # n tensors of (S*C_max,) - PRECOMPUTED
    S: int,
    N_dims: List[int],                 # [N_1, N_2, ..., N_n]
) -> torch.Tensor:
    """
    Forward pass for Alpha + Sample parallel tensor BiG-AMP.

    Computes Z_hat = (1/sqrt(M)) * sum_m F[c,m] * prod_d factors[d][idx[c], m]

    Args:
        factors: n tensors of (A, S*N_d, M) - student factors in Disjoint Union format
        F_flat: (S*C_max, M) - spreading coefficients (shared across alphas)
        offset_indices: n tensors of (S*C_max,) - PRECOMPUTED hyperedge indices with sample offsets
        S: Number of samples
        N_dims: List of dimension sizes

    Returns:
        Z_hat: (A, S*C_max) predicted observations
    """
    A = factors[0].shape[0]
    SC = F_flat.shape[0]
    M = F_flat.shape[1]
    n = len(factors)

    alpha_scale = 1.0 / math.sqrt(M)

    # Gather from each dimension using precomputed offset_indices
    gathered_list = []
    for d in range(n):
        # Direct gather using precomputed offset_indices (no sample_offsets calculation!)
        gathered = factors[d][:, offset_indices[d].long()]  # (A, S*C_max, M)
        gathered_list.append(gathered)

    # Stack and compute product
    gathered = torch.stack(gathered_list)  # (n, A, S*C_max, M)
    product = gathered.prod(dim=0)  # (A, S*C_max, M)

    # Compute Z_hat
    # F_flat is (S*C_max, M), broadcast to (A, S*C_max, M)
    Z_hat = alpha_scale * (F_flat.unsqueeze(0) * product).sum(dim=2)  # (A, S*C_max)

    return Z_hat


def compute_variance_tensor_super(
    factors: List[torch.Tensor],       # n tensors of (A, S*N_d, M)
    factor_vars: List[torch.Tensor],   # n tensors of (A, S*N_d, M)
    F_flat: torch.Tensor,              # (S*C_max, M)
    offset_indices: List[torch.Tensor], # n tensors of (S*C_max,) - PRECOMPUTED
    S: int,
    N_dims: List[int],
    is_ising: bool = False,
    alpha_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute variance for Alpha + Sample parallel tensor BiG-AMP.

    pvar = (1/M) * sum_m F²[c,m] * [prod_d E[X_d²] - prod_d E[X_d]²]

    Args:
        factors: n tensors of (A, S*N_d, M)
        factor_vars: n tensors of (A, S*N_d, M)
        F_flat: (S*C_max, M)
        offset_indices: n tensors of (S*C_max,) - PRECOMPUTED
        S: Number of samples
        N_dims: Dimension sizes
        is_ising: If True, skip F² (F²=1 for Ising)

    Returns:
        V: (A, S*C_max) variance estimates
    """
    A = factors[0].shape[0]
    SC = F_flat.shape[0]
    M = F_flat.shape[1]
    n = len(factors)

    alpha_scale_sq = 1.0 / M

    # Gather using precomputed offset_indices (no sample_offsets calculation!)
    gathered_list = []
    gathered_var_list = []
    for d in range(n):
        gathered_list.append(factors[d][:, offset_indices[d].long()])
        gathered_var_list.append(factor_vars[d][:, offset_indices[d].long()])

    gathered = torch.stack(gathered_list)      # (n, A, S*C_max, M)
    gathered_var = torch.stack(gathered_var_list)  # (n, A, S*C_max, M)

    # F²
    F_sq = torch.ones_like(F_flat) if is_ising else F_flat.pow(2)

    mean_sq_product = gathered.pow(2).prod(dim=0)
    second_product = (gathered.pow(2) + gathered_var).prod(dim=0)
    product_var = torch.clamp(second_product - mean_sq_product, min=0.0)
    V = alpha_scale_sq * (F_sq.unsqueeze(0) * product_var).sum(dim=2) + 1e-10

    if alpha_mask is not None:
        V = V * alpha_mask.float()

    return V


def tensor_step_super(
    factors: List[torch.Tensor],
    factor_vars: List[torch.Tensor],
    Y_flat: torch.Tensor,              # (S*C_max,)
    F_flat: torch.Tensor,              # (S*C_max, M)
    offset_indices: List[torch.Tensor], # n tensors of (S*C_max,) - PRECOMPUTED
    S: int,
    N_dims: List[int],
    M: int,                            # Latent dimension - for variance update
    alpha_mask: torch.Tensor,          # (A, S*C_max) bool
    damping: float,
    noise_var: float,
    is_ising: bool = False,
    prev_s: Optional[torch.Tensor] = None,
    prev_svar: Optional[torch.Tensor] = None,
    onsager_correction: bool = False,
    prior_precision_base: float = 1.0,
    prior_variance: float = 1.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Complete Alpha + Sample parallel BiG-AMP step for N-dimensional tensors.

    OPTIMIZED version: Uses precomputed offset_indices to eliminate
    repeated sample_offsets calculation in the training loop.

    Args:
        factors: n tensors of (A, S*N_d, M) - student factors
        factor_vars: n tensors of (A, S*N_d, M) - variances
        Y_flat: (S*C_max,) - observations (shared across alphas)
        F_flat: (S*C_max, M) - spreading coefficients
        offset_indices: n tensors of (S*C_max,) - PRECOMPUTED hyperedge indices
        S: Number of samples
        N_dims: Dimension sizes
        M: Latent dimension (for variance update numerical stability)
        alpha_mask: (A, S*C_max) - valid edges per alpha
        damping: Damping coefficient
        noise_var: Observation noise variance
        is_ising: Whether F is Ising
        prev_s: Previous s_values for Onsager correction
        onsager_correction: Whether to apply Onsager correction

    Returns:
        new_factors: Updated student factors
        new_factor_vars: Updated variances
        s_values: Damped residual state (A, S*C_max)
        svar_values: Damped output precision state (A, S*C_max)
    """
    n = len(factors)
    A = factors[0].shape[0]
    SC = F_flat.shape[0]
    C_max = SC // S

    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # Forward pass (uses precomputed offset_indices)
    Z_hat = forward_pass_tensor_super(factors, F_flat, offset_indices, S, N_dims)

    # Gather factors for variance and backward pass.
    gathered_list = []
    gathered_var_list = []
    for d in range(n):
        gathered_list.append(factors[d][:, offset_indices[d].long()])
        gathered_var_list.append(factor_vars[d][:, offset_indices[d].long()])

    gathered = torch.stack(gathered_list)  # (n, A, S*C_max, M)
    gathered_var = torch.stack(gathered_var_list)

    F_sq = torch.ones_like(F_flat) if is_ising else F_flat.pow(2)
    mean_sq = gathered.pow(2)
    mean_sq_product = mean_sq.prod(dim=0)
    second_product = (mean_sq + gathered_var).prod(dim=0)
    pvar = alpha_scale_sq * (
        F_sq.unsqueeze(0) * torch.clamp(second_product - mean_sq_product, min=0.0)
    ).sum(dim=2) + 1e-10
    zvar_sum = torch.zeros(A, SC, M, device=F_flat.device, dtype=factors[0].dtype)
    for d in range(n):
        if n > 1:
            other_mean_sq_product = torch.stack([
                mean_sq[dd] for dd in range(n) if dd != d
            ]).prod(dim=0)
        else:
            other_mean_sq_product = torch.ones(A, SC, M, device=F_flat.device, dtype=factors[0].dtype)
        zvar_sum += gathered_var[d] * other_mean_sq_product
    zvar = alpha_scale_sq * (F_sq.unsqueeze(0) * zvar_sum).sum(dim=2) + 1e-10
    pvar = pvar * alpha_mask.float() + 1e-10
    zvar = zvar * alpha_mask.float() + 1e-10

    # Onsager correction
    phat = Z_hat
    if onsager_correction and prev_s is not None:
        phat = Z_hat - zvar * prev_s

    # Residual
    Y_exp = Y_flat.unsqueeze(0)  # (1, S*C_max) -> broadcast to (A, S*C_max)
    # ROBUSTNESS FIX: Add Variance Floor (1e-5) to prevent explosion when V->0
    denom = torch.clamp(pvar + noise_var, min=1e-5)

    # DEBUG: Print key statistics
    DEBUG_STEP = False
    if DEBUG_STEP:
        print(f"  DEBUG: pvar.mean={pvar.mean().item():.6f}, pvar.max={pvar.max().item():.6f}, denom.mean={denom.mean().item():.6f}")

    new_s_values = torch.clamp((Y_exp - phat) / denom, min=-1e6, max=1e6)

    # Apply alpha mask: zero out invalid edges
    new_s_values = new_s_values * alpha_mask.float()

    # TeS-AMP CRITICAL: Damping on s_values (L86-87 of tes_amp.py)
    # This prevents large initial residuals from destabilizing the algorithm
    new_svar_values = (1.0 / denom) * alpha_mask.float()
    s_values = damped_state(new_s_values, prev_s if onsager_correction else None, damping)
    svar_values = damped_state(new_svar_values, prev_svar if onsager_correction else None, damping)

    # Update each factor dimension
    new_factors = []
    new_vars = []

    for d in range(n):
        N_d = N_dims[d]
        SN_d = S * N_d

        # Other product (all dimensions except d)
        if n > 1:
            other_indices = [i for i in range(n) if i != d]
            other_product = torch.stack([gathered[i] for i in other_indices]).prod(dim=0)
            other_second_product = torch.stack([
                gathered[i].pow(2) + gathered_var[i] for i in other_indices
            ]).prod(dim=0)
        else:
            other_product = torch.ones(A, SC, M, device=F_flat.device, dtype=F_flat.dtype)
            other_second_product = torch.ones(A, SC, M, device=F_flat.device, dtype=F_flat.dtype)

        # r contribution: alpha_scale * F * other_product * s_values
        r_contrib = alpha_scale * F_flat.unsqueeze(0) * other_product * s_values.unsqueeze(2)

        other_sq_product = other_product.pow(2)
        other_var_product = torch.clamp(other_second_product - other_sq_product, min=0.0)

        # CRITICAL FIX: Apply mask to tau_contrib to prevent spurious precision from padding edges
        mask_exp = alpha_mask.unsqueeze(2).float() # (A, SC, 1)
        tau_contrib = alpha_scale_sq * F_sq.unsqueeze(0) * other_sq_product * svar_values.unsqueeze(2) * mask_exp
        gain_contrib = alpha_scale_sq * F_sq.unsqueeze(0) * other_var_product * svar_values.unsqueeze(2) * mask_exp

        # Scatter add
        r_d = torch.zeros(A, SN_d, M, device=F_flat.device, dtype=factors[0].dtype)
        tau_d = torch.zeros(A, SN_d, M, device=F_flat.device, dtype=factors[0].dtype)
        gain_d = torch.zeros(A, SN_d, M, device=F_flat.device, dtype=factors[0].dtype)

        # Use precomputed offset_indices for scatter (no sample_offsets calculation!)
        idx_exp = offset_indices[d].unsqueeze(0).unsqueeze(2).expand(A, -1, M)

        r_d.scatter_add_(1, idx_exp, r_contrib.to(dtype=r_d.dtype))
        tau_d.scatter_add_(1, idx_exp, tau_contrib.to(dtype=tau_d.dtype))
        gain_d.scatter_add_(1, idx_exp, gain_contrib.to(dtype=gain_d.dtype))

        tau_d = tau_d.clamp(min=1e-10)
        new_factor_d, new_var_d = gaussian_posterior_update(
            factors[d], r_d, tau_d, prior_precision_base, prior_variance, gain_d
        )
        new_factor_d = torch.clamp(new_factor_d, min=-10.0, max=10.0)


        # Damping: beta=1 fully accepts the new state; beta=0 freezes old state.
        new_factors.append(blend_new_old(new_factor_d, factors[d], damping))
        new_vars.append(torch.clamp(
            blend_new_old(new_var_d, factor_vars[d], damping),
            min=1e-8,
            max=prior_variance,
        ))

    return new_factors, new_vars, s_values, svar_values
