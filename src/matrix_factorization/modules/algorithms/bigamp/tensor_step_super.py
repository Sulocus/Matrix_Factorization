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
    is_rademacher: bool = False,
    alpha_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute variance for Alpha + Sample parallel tensor BiG-AMP.

    V = (1/M) * sum_m F²[c,m] * sum_d (var_d[idx,m] * prod_{d'≠d} factor²[d'][idx,m])

    Args:
        factors: n tensors of (A, S*N_d, M)
        factor_vars: n tensors of (A, S*N_d, M)
        F_flat: (S*C_max, M)
        offset_indices: n tensors of (S*C_max,) - PRECOMPUTED
        S: Number of samples
        N_dims: Dimension sizes
        is_rademacher: If True, skip F² (F²=1 for Rademacher)

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
    F_sq = torch.ones_like(F_flat) if is_rademacher else F_flat.pow(2)

    # N-GAMP FIX: Use second moments E[X²] = μ² + V instead of μ²
    # This is critical for Order 3+ tensors where μ ≈ 0 initially but V ≈ 1
    all_second_moment = gathered.pow(2) + gathered_var  # E[X²] = μ² + V
    product_all_second = all_second_moment.prod(dim=0)  # (A, S*C_max, M)

    # Sum over dimensions
    V_sum = torch.zeros(A, SC, M, device=F_flat.device, dtype=factor_vars[0].dtype)
    for d in range(n):
        other_second = product_all_second / (all_second_moment[d] + 1e-10)
        V_sum += gathered_var[d] * other_second

    # Final variance
    V = alpha_scale_sq * (F_sq.unsqueeze(0) * V_sum).sum(dim=2) + 1e-10  # (A, S*C_max)

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
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
    onsager_correction: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
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
        is_rademacher: Whether F is Rademacher
        prev_s: Previous s_values for Onsager correction
        onsager_correction: Whether to apply Onsager correction

    Returns:
        new_factors: Updated student factors
        new_factor_vars: Updated variances
        s_values: Residual values (A, S*C_max)
    """
    n = len(factors)
    A = factors[0].shape[0]
    SC = F_flat.shape[0]
    C_max = SC // S

    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # Forward pass (uses precomputed offset_indices)
    Z_hat = forward_pass_tensor_super(factors, F_flat, offset_indices, S, N_dims)

    # Variance (uses precomputed offset_indices)
    V = compute_variance_tensor_super(factors, factor_vars, F_flat, offset_indices, S, N_dims, is_rademacher, alpha_mask)

    # Onsager correction
    if onsager_correction and prev_s is not None:
        correction = torch.clamp(V * prev_s, min=-0.5, max=0.5)
        Z_hat = Z_hat - correction

    # Residual
    Y_exp = Y_flat.unsqueeze(0)  # (1, S*C_max) -> broadcast to (A, S*C_max)
    # ROBUSTNESS FIX: Add Variance Floor (1e-5) to prevent explosion when V->0
    denom = torch.clamp(V + noise_var, min=1e-5)

    # DEBUG: Print key statistics
    DEBUG_STEP = False
    if DEBUG_STEP:
        print(f"  DEBUG: V.mean={V.mean().item():.6f}, V.max={V.max().item():.6f}, denom.mean={denom.mean().item():.6f}")

    new_s_values = torch.clamp((Y_exp - Z_hat) / denom, min=-1e6, max=1e6)

    # Apply alpha mask: zero out invalid edges
    new_s_values = new_s_values * alpha_mask.float()

    # TeS-AMP CRITICAL: Damping on s_values (L86-87 of tes_amp.py)
    # This prevents large initial residuals from destabilizing the algorithm
    if prev_s is not None:
        s_values = damping * new_s_values + (1 - damping) * prev_s
    else:
        s_values = new_s_values

    # Gather factors for backward pass (using precomputed offset_indices)
    gathered_list = []
    for d in range(n):
        gathered_list.append(factors[d][:, offset_indices[d].long()])

    gathered = torch.stack(gathered_list)  # (n, A, S*C_max, M)

    # F²
    F_sq = torch.ones_like(F_flat) if is_rademacher else F_flat.pow(2)

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
        else:
            other_product = torch.ones(A, SC, M, device=F_flat.device, dtype=F_flat.dtype)

        # r contribution: alpha_scale * F * other_product * s_values
        r_contrib = alpha_scale * F_flat.unsqueeze(0) * other_product * s_values.unsqueeze(2)

        # REVERTED: User reported N=3 convergence failure with variance term.
        # Back to "TeS-AMP MATCH": use squared means only.
        if n > 1:
            other_sq_product = torch.stack([
                gathered[i].pow(2) for i in other_indices
            ]).prod(dim=0)  # (A, SC, M)
        else:
            other_sq_product = other_product.pow(2)

        # CRITICAL FIX: Apply mask to tau_contrib to prevent spurious precision from padding edges
        mask_exp = alpha_mask.unsqueeze(2).float() # (A, SC, 1)
        tau_contrib = alpha_scale_sq * F_sq.unsqueeze(0) * other_sq_product / denom.unsqueeze(2) * mask_exp

        # Scatter add
        r_d = torch.zeros(A, SN_d, M, device=F_flat.device, dtype=factors[0].dtype)
        tau_d = torch.zeros(A, SN_d, M, device=F_flat.device, dtype=factors[0].dtype)

        # Use precomputed offset_indices for scatter (no sample_offsets calculation!)
        idx_exp = offset_indices[d].unsqueeze(0).unsqueeze(2).expand(A, -1, M)

        r_d.scatter_add_(1, idx_exp, r_contrib.to(dtype=r_d.dtype))
        tau_d.scatter_add_(1, idx_exp, tau_contrib.to(dtype=tau_d.dtype))

        # Update with Prior N(0, 1) - Correct AMP Posterior Update
        # Reference: TeS-AMP (Algorithm 2), priors.py GaussianPrior.estimate()
        #
        # TeS-AMP formula:
        #   r_hat = factors + r_var * r_term (observation scaled by variance)
        #   z_hat = prior.estimate(r_hat, r_var)
        #         = r_hat / (r_var + 1) for N(0,1) prior
        #         = (factors + r_var * r_term) / (r_var + 1)
        #         = factors / (r_var + 1) + r * r_var / (r_var + 1)
        #         = new_var * (tau * factors + r)  (using tau = 1/r_var)
        #
        # ISSUE: tau*factors and r are negatively correlated, causing cancellation!
        # SOLUTION: Use incremental update instead of replacement update
        #   new_factor = factor + step_size * gradient
        #   where gradient = r - tau * factor (natural gradient for N(0,1) prior)
        tau_d = tau_d.clamp(min=1e-10)
        new_var_d = torch.clamp(1.0 / (1.0 + tau_d), min=1e-10, max=1.0)

        # DEBUG: For first dimension only (disabled)
        if d == 0 and False:
            print(f"    tau_d.mean={tau_d.mean().item():.4f}, r_d.mean={r_d.mean().item():.6f}, r_d.std={r_d.std().item():.4f}, new_var.mean={new_var_d.mean().item():.4f}")
            tau_f_r = tau_d * factors[d] + r_d
            print(f"    factors[d].std={factors[d].std().item():.4f}, (tau*f+r).std={(tau_f_r).std().item():.4f}, expected_new_std={(new_var_d * tau_f_r).std().item():.4f}")

        # INCREMENTAL UPDATE: Avoid tau*f + r cancellation
        # gradient = r - tau * factor  (for N(0,1) prior)
        # This pushes factor towards posterior mean while shrinking towards prior (0)
        step_size = new_var_d  # Use new_var as adaptive step size
        gradient = r_d - tau_d * factors[d]
        new_factor_d = factors[d] + step_size * gradient
        new_factor_d = torch.clamp(new_factor_d, min=-10.0, max=10.0)


        # Damping (consistent with Bipartite step.py: damping * new + (1-damping) * old)
        new_factors.append(damping * new_factor_d + (1 - damping) * factors[d])
        new_vars.append(damping * new_var_d + (1 - damping) * factor_vars[d])

    return new_factors, new_vars, s_values
