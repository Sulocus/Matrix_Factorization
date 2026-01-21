"""
Batched n-dimensional tensor BiG-AMP step functions.

Supports batch dimension S for parallel sample processing.
All tensors have an additional leading batch dimension S compared to tensor_step.py.

Key functions:
- forward_pass_tensor_batch: (S, C) predicted values
- compute_variance_tensor_batch: (S, C) variance at each hyperedge
- tensor_step_batch: Complete batched BiG-AMP iteration
"""

import math
import torch
from typing import List, Optional, Tuple


def forward_pass_tensor_batch(
    factors: List[torch.Tensor],  # n tensors of (S, N_d, M)
    F: torch.Tensor,              # (S, C, M)
    indices: List[torch.Tensor],  # n tensors of (C,) - shared across samples
) -> torch.Tensor:
    """
    Batched forward pass for n-dimensional tensor.
    
    Z_hat[s, c] = (1/√M) Σ_μ F[s, c, μ] ∏_d factors[d][s, indices[d][c], μ]
    
    Args:
        factors: List of n factor matrices, each (S, N_d, M)
        F: (S, C, M) spreading coefficients
        indices: List of n index tensors, each (C,) - SHARED across all S samples
        
    Returns:
        Z_hat: (S, C) predicted values
    """
    S, C, M = F.shape
    alpha_scale = 1.0 / math.sqrt(M)
    
    # Gather factors at edge positions: (n, S, C, M)
    gathered = torch.stack([
        factors[d][:, indices[d].long()]  # (S, C, M)
        for d in range(len(factors))
    ])  # (n, S, C, M)
    
    # Product across factors
    product = gathered.prod(dim=0)  # (S, C, M)
    
    # Z_hat = (1/√M) Σ_μ F * product
    Z_hat = alpha_scale * (F * product).sum(dim=2)  # (S, C)
    
    return Z_hat


def compute_variance_tensor_batch(
    factors: List[torch.Tensor],      # n tensors of (S, N_d, M)
    factor_vars: List[torch.Tensor],  # n tensors of (S, N_d, M)
    F: torch.Tensor,                  # (S, C, M)
    indices: List[torch.Tensor],      # n tensors of (C,)
    is_rademacher: bool = False,
) -> torch.Tensor:
    """
    Batched variance computation.
    
    V[s, c] = (1/M) Σ_μ F²[s, c, μ] Σ_d [Var_d * ∏_{d'≠d} factor_d'^2]
    
    Args:
        factors: List of n factor matrices, each (S, N_d, M)
        factor_vars: List of n variance matrices, each (S, N_d, M)
        F: (S, C, M) spreading coefficients
        indices: List of n index tensors, each (C,)
        is_rademacher: If True, skip F² computation (F²=1 for Rademacher)
        
    Returns:
        V: (S, C) variance at each hyperedge
    """
    S, C, M = F.shape
    n = len(factors)
    alpha_scale_sq = 1.0 / M
    
    # Gather factors and variances at edge positions
    gathered = torch.stack([
        factors[d][:, indices[d].long()] for d in range(n)
    ])  # (n, S, C, M)
    gathered_var = torch.stack([
        factor_vars[d][:, indices[d].long()] for d in range(n)
    ])  # (n, S, C, M)
    
    # F² or 1 for Rademacher
    F_sq = torch.ones_like(F) if is_rademacher else F.pow(2)
    
    # Compute product of all factors squared
    all_sq = gathered.pow(2)  # (n, S, C, M)
    product_all_sq = all_sq.prod(dim=0)  # (S, C, M)
    
    # Sum over d: Var_d * (product_all_sq / factor_d^2)
    V_sum = torch.zeros(S, C, M, device=F.device, dtype=F.dtype)
    for d in range(n):
        # product_all_sq / factor_d^2 = ∏_{d'≠d} factor_d'^2
        # Add small epsilon to avoid division by zero
        other_product = product_all_sq / (all_sq[d] + 1e-10)
        V_sum += gathered_var[d] * other_product
    
    V = alpha_scale_sq * (F_sq * V_sum).sum(dim=2) + 1e-10
    return V


def tensor_step_batch(
    factors: List[torch.Tensor],      # n tensors of (S, N_d, M)
    factor_vars: List[torch.Tensor],  # n tensors of (S, N_d, M)
    Y: torch.Tensor,                  # (S, C)
    F: torch.Tensor,                  # (S, C, M)
    indices: List[torch.Tensor],      # n tensors of (C,)
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
    onsager_correction: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Complete batched n-dimensional BiG-AMP step.
    
    Performs one iteration of the BiG-AMP algorithm for n-dimensional
    tensor CP decomposition, processing all S samples in parallel.
    
    Args:
        factors: List of n factor matrices, each (S, N_d, M)
        factor_vars: List of n variance matrices, each (S, N_d, M)
        Y: (S, C) observed values for all samples
        F: (S, C, M) spreading coefficients for all samples
        indices: List of n index tensors, each (C,) - SHARED across samples
        damping: Damping factor (0 = no damping, 1 = full damping)
        noise_var: Observation noise variance
        is_rademacher: If True, F is Rademacher (F²=1)
        prev_s: (S, C) Previous residual for Onsager correction
        onsager_correction: Whether to apply Onsager correction
        
    Returns:
        new_factors: Updated factor estimates, each (S, N_d, M)
        new_factor_vars: Updated variance estimates, each (S, N_d, M)
        s_values: (S, C) Current residual (for next Onsager correction)
    """
    n = len(factors)
    S, C, M = F.shape
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Forward pass
    Z_hat = forward_pass_tensor_batch(factors, F, indices)  # (S, C)
    
    # Variance
    V = compute_variance_tensor_batch(factors, factor_vars, F, indices, is_rademacher)  # (S, C)
    
    # Onsager correction (optional, default OFF)
    if onsager_correction and prev_s is not None:
        correction = V * prev_s
        # Stability check: limit correction magnitude to prevent explosion
        correction = torch.clamp(correction, min=-0.5, max=0.5)
        Z_hat = Z_hat - correction
    
    # Residual
    denom = torch.clamp(V + noise_var, min=1e-6)  # (S, C)
    s_values = (Y - Z_hat) / denom
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    
    # Gather for backward pass
    gathered = torch.stack([
        factors[d][:, indices[d].long()] for d in range(n)
    ])  # (n, S, C, M)
    F_sq = torch.ones_like(F) if is_rademacher else F.pow(2)
    
    # Update each factor
    new_factors = []
    new_vars = []
    
    for d in range(n):
        N_d = factors[d].shape[1]
        
        # Product of other factors
        if n > 1:
            other_indices = [dd for dd in range(n) if dd != d]
            other_gathered = torch.stack([gathered[dd] for dd in other_indices])
            other_product = other_gathered.prod(dim=0)  # (S, C, M)
        else:
            other_product = torch.ones(S, C, M, device=F.device, dtype=F.dtype)
        
        # r contribution: (1/√M) F * other_product * s
        # s_values is (S, C), need to expand to (S, C, M)
        r_contrib = alpha_scale * F * other_product * s_values.unsqueeze(2)  # (S, C, M)
        
        # tau contribution
        if n > 1:
            other_sq = torch.stack([gathered[dd].pow(2) for dd in other_indices])
            other_sq_product = other_sq.prod(dim=0)  # (S, C, M)
        else:
            other_sq_product = torch.ones(S, C, M, device=F.device, dtype=F.dtype)
        tau_contrib = alpha_scale_sq * F_sq * other_sq_product / denom.unsqueeze(2)  # (S, C, M)
        
        # Scatter add with batch dimension
        r_d = torch.zeros(S, N_d, M, device=F.device, dtype=F.dtype)
        tau_d = torch.zeros(S, N_d, M, device=F.device, dtype=F.dtype)
        # Expand indices for batch scatter: (C,) -> (S, C, M)
        idx_exp = indices[d].long().unsqueeze(0).unsqueeze(2).expand(S, -1, M)
        r_d.scatter_add_(1, idx_exp, r_contrib)
        tau_d.scatter_add_(1, idx_exp, tau_contrib)
        tau_d = tau_d.clamp(min=1e-10)
        
        # Update
        new_var_d = 1.0 / tau_d
        new_var_d = new_var_d.clamp(max=1.0)
        new_factor_d = new_var_d * (tau_d * factors[d] + r_d)
        
        # Stability check: clamp factors to prevent explosion
        new_factor_d = torch.clamp(new_factor_d, min=-10.0, max=10.0)
        
        # Damping
        new_factor_d = damping * factors[d] + (1 - damping) * new_factor_d
        new_var_d = damping * factor_vars[d] + (1 - damping) * new_var_d
        
        new_factors.append(new_factor_d)
        new_vars.append(new_var_d)
    
    return new_factors, new_vars, s_values
