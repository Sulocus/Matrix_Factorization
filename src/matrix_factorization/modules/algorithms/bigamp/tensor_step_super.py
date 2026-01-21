"""
Alpha-parallel BiG-AMP step functions for N-dimensional tensor CP decomposition.

This module extends tensor_step_batch.py to support Alpha + Sample parallelization
using the TensorSuperGraph Disjoint Union format.

Key differences from tensor_step_batch.py:
- factors shape: (A, S*N_d, M) instead of (S, N_d, M)
- Uses alpha_mask to handle variable edge counts per alpha
- Supports batch processing across all alpha values
"""

import math
import torch
from typing import List, Tuple, Optional


def forward_pass_tensor_super(
    factors: List[torch.Tensor],      # n tensors of (A, S*N_d, M)
    F_flat: torch.Tensor,             # (S*C_max, M)
    indices_flat: List[torch.Tensor], # n tensors of (S*C_max,)
    S: int,
    N_dims: List[int],                # [N_1, N_2, ..., N_n]
) -> torch.Tensor:
    """
    Forward pass for Alpha + Sample parallel tensor BiG-AMP.
    
    Computes Z_hat = (1/sqrt(M)) * sum_m F[c,m] * prod_d factors[d][idx[c], m]
    
    Args:
        factors: n tensors of (A, S*N_d, M) - student factors in Disjoint Union format
        F_flat: (S*C_max, M) - spreading coefficients (shared across alphas)
        indices_flat: n tensors of (S*C_max,) - hyperedge indices
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
    
    # Compute offset for each sample in the Disjoint Union
    # For sample s, factor d, the offset is s * N_d
    gathered_list = []
    for d in range(n):
        N_d = N_dims[d]
        # Compute sample offsets: indices_flat contains values in [0, N_d)
        # Need to add s * N_d for sample s
        C_max = SC // S
        sample_offsets = torch.arange(S, device=factors[d].device).unsqueeze(1) * N_d  # (S, 1)
        sample_offsets = sample_offsets.expand(S, C_max).reshape(-1)  # (S*C_max,)
        
        # Add offsets to indices
        offset_indices = indices_flat[d] + sample_offsets  # (S*C_max,)
        
        # Gather from factors[d]: (A, S*N_d, M) -> (A, S*C_max, M)
        gathered = factors[d][:, offset_indices.long()]  # (A, S*C_max, M)
        gathered_list.append(gathered)
    
    # Stack and compute product
    gathered = torch.stack(gathered_list)  # (n, A, S*C_max, M)
    product = gathered.prod(dim=0)  # (A, S*C_max, M)
    
    # Compute Z_hat
    # F_flat is (S*C_max, M), broadcast to (A, S*C_max, M)
    Z_hat = alpha_scale * (F_flat.unsqueeze(0) * product).sum(dim=2)  # (A, S*C_max)
    
    return Z_hat


def compute_variance_tensor_super(
    factors: List[torch.Tensor],      # n tensors of (A, S*N_d, M)
    factor_vars: List[torch.Tensor],  # n tensors of (A, S*N_d, M)
    F_flat: torch.Tensor,             # (S*C_max, M)
    indices_flat: List[torch.Tensor], # n tensors of (S*C_max,)
    S: int,
    N_dims: List[int],
    is_rademacher: bool = False,
) -> torch.Tensor:
    """
    Compute variance for Alpha + Sample parallel tensor BiG-AMP.
    
    V = (1/M) * sum_m F²[c,m] * sum_d (var_d[idx,m] * prod_{d'≠d} factor²[d'][idx,m])
    
    Args:
        factors: n tensors of (A, S*N_d, M)
        factor_vars: n tensors of (A, S*N_d, M)
        F_flat: (S*C_max, M)
        indices_flat: n tensors of (S*C_max,)
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
    C_max = SC // S
    
    alpha_scale_sq = 1.0 / M
    
    # Compute offset indices
    gathered_list = []
    gathered_var_list = []
    for d in range(n):
        N_d = N_dims[d]
        sample_offsets = torch.arange(S, device=factors[d].device).unsqueeze(1) * N_d
        sample_offsets = sample_offsets.expand(S, C_max).reshape(-1)
        offset_indices = indices_flat[d] + sample_offsets
        
        gathered_list.append(factors[d][:, offset_indices.long()])
        gathered_var_list.append(factor_vars[d][:, offset_indices.long()])
    
    gathered = torch.stack(gathered_list)      # (n, A, S*C_max, M)
    gathered_var = torch.stack(gathered_var_list)  # (n, A, S*C_max, M)
    
    # F²
    F_sq = torch.ones_like(F_flat) if is_rademacher else F_flat.pow(2)
    
    # Product of all squared factors
    all_sq = gathered.pow(2)  # (n, A, S*C_max, M)
    product_all_sq = all_sq.prod(dim=0)  # (A, S*C_max, M)
    
    # Sum over dimensions
    V_sum = torch.zeros(A, SC, M, device=F_flat.device, dtype=F_flat.dtype)
    for d in range(n):
        other_product = product_all_sq / (all_sq[d] + 1e-10)
        V_sum += gathered_var[d] * other_product
    
    # Final variance
    V = alpha_scale_sq * (F_sq.unsqueeze(0) * V_sum).sum(dim=2) + 1e-10  # (A, S*C_max)
    
    return V


def tensor_step_super(
    factors: List[torch.Tensor],
    factor_vars: List[torch.Tensor],
    Y_flat: torch.Tensor,             # (S*C_max,)
    F_flat: torch.Tensor,             # (S*C_max, M)
    indices_flat: List[torch.Tensor], # n tensors of (S*C_max,)
    S: int,
    N_dims: List[int],
    alpha_mask: torch.Tensor,         # (A, S*C_max) bool
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
    onsager_correction: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Complete Alpha + Sample parallel BiG-AMP step for N-dimensional tensors.
    
    Args:
        factors: n tensors of (A, S*N_d, M) - student factors
        factor_vars: n tensors of (A, S*N_d, M) - variances
        Y_flat: (S*C_max,) - observations (shared across alphas)
        F_flat: (S*C_max, M) - spreading coefficients
        indices_flat: n tensors of (S*C_max,) - hyperedge indices
        S: Number of samples
        N_dims: Dimension sizes
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
    M = F_flat.shape[1]
    C_max = SC // S
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Forward pass
    Z_hat = forward_pass_tensor_super(factors, F_flat, indices_flat, S, N_dims)
    
    # Variance
    V = compute_variance_tensor_super(factors, factor_vars, F_flat, indices_flat, S, N_dims, is_rademacher)
    
    # Onsager correction
    if onsager_correction and prev_s is not None:
        correction = torch.clamp(V * prev_s, min=-0.5, max=0.5)
        Z_hat = Z_hat - correction
    
    # Residual
    Y_exp = Y_flat.unsqueeze(0)  # (1, S*C_max) -> broadcast to (A, S*C_max)
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = torch.clamp((Y_exp - Z_hat) / denom, min=-1e6, max=1e6)
    
    # Apply alpha mask: zero out invalid edges
    s_values = s_values * alpha_mask.float()
    
    # Gather factors for backward pass
    gathered_list = []
    for d in range(n):
        N_d = N_dims[d]
        sample_offsets = torch.arange(S, device=factors[d].device).unsqueeze(1) * N_d
        sample_offsets = sample_offsets.expand(S, C_max).reshape(-1)
        offset_indices = indices_flat[d] + sample_offsets
        gathered_list.append(factors[d][:, offset_indices.long()])
    
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
        
        # tau contribution
        other_sq_product = other_product.pow(2) if n > 1 else other_product
        tau_contrib = alpha_scale_sq * F_sq.unsqueeze(0) * other_sq_product / denom.unsqueeze(2)
        
        # Scatter add
        r_d = torch.zeros(A, SN_d, M, device=F_flat.device, dtype=F_flat.dtype)
        tau_d = torch.zeros(A, SN_d, M, device=F_flat.device, dtype=F_flat.dtype)
        
        # Compute offset indices for scatter
        sample_offsets = torch.arange(S, device=factors[d].device).unsqueeze(1) * N_d
        sample_offsets = sample_offsets.expand(S, C_max).reshape(-1)
        offset_indices = indices_flat[d] + sample_offsets
        
        # Expand indices for (A, S*C_max, M) -> scatter to (A, S*N_d, M)
        idx_exp = offset_indices.unsqueeze(0).unsqueeze(2).expand(A, -1, M)
        
        r_d.scatter_add_(1, idx_exp, r_contrib)
        tau_d.scatter_add_(1, idx_exp, tau_contrib)
        
        # Update
        tau_d = tau_d.clamp(min=1e-10)
        new_var_d = torch.clamp(1.0 / tau_d, max=1.0)
        new_factor_d = torch.clamp(
            new_var_d * (tau_d * factors[d] + r_d),
            min=-10.0, max=10.0
        )
        
        # Damping
        new_factors.append(damping * factors[d] + (1 - damping) * new_factor_d)
        new_vars.append(damping * factor_vars[d] + (1 - damping) * new_var_d)
    
    return new_factors, new_vars, s_values
