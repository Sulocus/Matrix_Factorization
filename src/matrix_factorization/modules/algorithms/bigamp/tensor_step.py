"""
N-dimensional tensor BigAMP step functions.

This module implements the core computational functions for n-dimensional
tensor CP decomposition using the BiG-AMP message passing algorithm.

Key functions:
- forward_pass_tensor: Compute Z_hat = (1/√M) Σ_μ F[c,μ] ∏_d V^{(d)}[i_d,μ]
- compute_variance_tensor: Compute prediction variance
- tensor_step: Complete BiG-AMP iteration
"""

import math
import torch
from typing import List, Optional, Tuple


def forward_pass_tensor(
    factors: List[torch.Tensor],  # n tensors of (N_d, M)
    F: torch.Tensor,              # (C, M)
    indices: List[torch.Tensor],  # n tensors of (C,)
) -> torch.Tensor:
    """
    Compute Z_hat for n-dimensional tensor.
    
    Z_hat[c] = (1/√M) Σ_μ F[c,μ] ∏_d factors[d][indices[d][c], μ]
    
    Args:
        factors: List of n factor matrices, each (N_d, M)
        F: (C, M) spreading coefficients
        indices: List of n index tensors, each (C,)
        
    Returns:
        Z_hat: (C,) predicted values
    """
    C, M = F.shape
    alpha_scale = 1.0 / math.sqrt(M)
    
    # Gather factors at edge positions: (n, C, M)
    gathered = torch.stack([
        factors[d][indices[d].long()]  # (C, M)
        for d in range(len(factors))
    ])  # (n, C, M)
    
    # Product across factors
    product = gathered.prod(dim=0)  # (C, M)
    
    # Z_hat = (1/√M) Σ_μ F * product
    Z_hat = alpha_scale * (F * product).sum(dim=1)  # (C,)
    
    return Z_hat


def compute_variance_tensor(
    factors: List[torch.Tensor],      # n tensors of (N_d, M)
    factor_vars: List[torch.Tensor],  # n tensors of (N_d, M)
    F: torch.Tensor,                  # (C, M)
    indices: List[torch.Tensor],      # n tensors of (C,)
    is_rademacher: bool = False,
) -> torch.Tensor:
    """
    Compute variance V for n-dimensional tensor.
    
    V[c] = (1/M) Σ_μ F²[c,μ] Σ_d [Var_d * ∏_{d'≠d} factor_d'^2]
    
    This is derived from first-order Taylor expansion of the variance
    of a product of random variables.
    
    Args:
        factors: List of n factor matrices, each (N_d, M)
        factor_vars: List of n variance matrices, each (N_d, M)
        F: (C, M) spreading coefficients
        indices: List of n index tensors, each (C,)
        is_rademacher: If True, skip F² computation (F²=1 for Rademacher)
        
    Returns:
        V: (C,) variance at each hyperedge
    """
    C, M = F.shape
    n = len(factors)
    alpha_scale_sq = 1.0 / M
    
    # Gather factors and variances at edge positions
    gathered = torch.stack([
        factors[d][indices[d].long()] for d in range(n)
    ])  # (n, C, M)
    gathered_var = torch.stack([
        factor_vars[d][indices[d].long()] for d in range(n)
    ])  # (n, C, M)
    
    # F² or 1 for Rademacher
    F_sq = torch.ones_like(F) if is_rademacher else F.pow(2)
    
    # Compute product of all factors squared
    all_sq = gathered.pow(2)  # (n, C, M)
    product_all_sq = all_sq.prod(dim=0)  # (C, M)
    
    # Sum over d: Var_d * (product_all_sq / factor_d^2)
    V_sum = torch.zeros(C, M, device=F.device, dtype=F.dtype)
    for d in range(n):
        # product_all_sq / factor_d^2 = ∏_{d'≠d} factor_d'^2
        # Add small epsilon to avoid division by zero
        other_product = product_all_sq / (all_sq[d] + 1e-10)
        V_sum += gathered_var[d] * other_product
    
    V = alpha_scale_sq * (F_sq * V_sum).sum(dim=1) + 1e-10
    return V


def tensor_step(
    factors: List[torch.Tensor],
    factor_vars: List[torch.Tensor],
    Y: torch.Tensor,
    F: torch.Tensor,
    indices: List[torch.Tensor],
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
    onsager_correction: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Complete n-dimensional BiG-AMP step.
    
    Performs one iteration of the BiG-AMP algorithm for n-dimensional
    tensor CP decomposition.
    
    Args:
        factors: List of n factor matrices, each (N_d, M)
        factor_vars: List of n variance matrices, each (N_d, M)
        Y: (C,) observed values
        F: (C, M) spreading coefficients
        indices: List of n index tensors, each (C,)
        damping: Damping factor (0 = no damping, 1 = full damping)
        noise_var: Observation noise variance
        is_rademacher: If True, F is Rademacher (F²=1)
        prev_s: Previous residual for Onsager correction
        onsager_correction: Whether to apply Onsager correction (default: False)
        
    Returns:
        new_factors: Updated factor estimates
        new_factor_vars: Updated variance estimates
        s_values: Current residual (for next Onsager correction)
    """
    n = len(factors)
    C, M = F.shape
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Forward pass
    Z_hat = forward_pass_tensor(factors, F, indices)
    
    # Variance
    V = compute_variance_tensor(factors, factor_vars, F, indices, is_rademacher)
    
    # Onsager correction (optional, default OFF)
    if onsager_correction and prev_s is not None:
        Z_hat = Z_hat - V * prev_s
    
    # Residual
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y - Z_hat) / denom
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    
    # Gather for backward pass
    gathered = torch.stack([
        factors[d][indices[d].long()] for d in range(n)
    ])  # (n, C, M)
    F_sq = torch.ones_like(F) if is_rademacher else F.pow(2)
    
    # Update each factor
    new_factors = []
    new_vars = []
    
    for d in range(n):
        N_d = factors[d].shape[0]
        
        # Product of other factors
        if n > 1:
            other_indices = [dd for dd in range(n) if dd != d]
            other_gathered = torch.stack([gathered[dd] for dd in other_indices])
            other_product = other_gathered.prod(dim=0)  # (C, M)
        else:
            other_product = torch.ones(C, M, device=F.device, dtype=F.dtype)
        
        # r contribution: (1/√M) F * other_product * s
        r_contrib = alpha_scale * F * other_product * s_values.unsqueeze(1)  # (C, M)
        
        # tau contribution
        if n > 1:
            other_sq = torch.stack([gathered[dd].pow(2) for dd in other_indices])
            other_sq_product = other_sq.prod(dim=0)  # (C, M)
        else:
            other_sq_product = torch.ones(C, M, device=F.device, dtype=F.dtype)
        tau_contrib = alpha_scale_sq * F_sq * other_sq_product / denom.unsqueeze(1)  # (C, M)
        
        # Scatter add
        r_d = torch.zeros(N_d, M, device=F.device, dtype=F.dtype)
        tau_d = torch.zeros(N_d, M, device=F.device, dtype=F.dtype)
        idx_exp = indices[d].long().unsqueeze(1).expand(-1, M)
        r_d.scatter_add_(0, idx_exp, r_contrib)
        tau_d.scatter_add_(0, idx_exp, tau_contrib)
        tau_d = tau_d.clamp(min=1e-10)
        
        # Update
        new_var_d = 1.0 / tau_d
        new_var_d = new_var_d.clamp(max=1.0)
        new_factor_d = new_var_d * (tau_d * factors[d] + r_d)
        
        # Damping
        new_factor_d = damping * factors[d] + (1 - damping) * new_factor_d
        new_var_d = damping * factor_vars[d] + (1 - damping) * new_var_d
        
        new_factors.append(new_factor_d)
        new_vars.append(new_var_d)
    
    return new_factors, new_vars, s_values
