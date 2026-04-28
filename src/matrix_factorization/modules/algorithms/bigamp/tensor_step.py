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

from .conventions import blend_new_old, damped_state, gaussian_posterior_update


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
    is_ising: bool = False,
) -> torch.Tensor:
    """
    Compute output variance pvar for n-dimensional tensor.
    
    pvar[c] = (1/M) Σ_μ F²[c,μ] Var(∏_d X_d)
            = (1/M) Σ_μ F²[c,μ] [∏_d E[X_d²] - ∏_d E[X_d]²]
    
    Args:
        factors: List of n factor matrices, each (N_d, M)
        factor_vars: List of n variance matrices, each (N_d, M)
        F: (C, M) spreading coefficients
        indices: List of n index tensors, each (C,)
        is_ising: If True, skip F² computation (F²=1 for Ising)
        
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
    
    # F² or 1 for Ising
    F_sq = torch.ones_like(F) if is_ising else F.pow(2)
    
    mean_sq_product = gathered.pow(2).prod(dim=0)
    second_product = (gathered.pow(2) + gathered_var).prod(dim=0)
    product_var = torch.clamp(second_product - mean_sq_product, min=0.0)
    V = alpha_scale_sq * (F_sq * product_var).sum(dim=1) + 1e-10
    return V


def tensor_step(
    factors: List[torch.Tensor],
    factor_vars: List[torch.Tensor],
    Y: torch.Tensor,
    F: torch.Tensor,
    indices: List[torch.Tensor],
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
    Complete n-dimensional BiG-AMP step.
    
    Performs one iteration of the BiG-AMP algorithm for n-dimensional
    tensor CP decomposition.
    
    Args:
        factors: List of n factor matrices, each (N_d, M)
        factor_vars: List of n variance matrices, each (N_d, M)
        Y: (C,) observed values
        F: (C, M) spreading coefficients
        indices: List of n index tensors, each (C,)
        damping: BiG-AMP beta; 1 fully accepts the new state, 0 freezes
        noise_var: Observation noise variance
        is_ising: If True, F is Ising (F²=1)
        prev_s: Previous residual for Onsager correction
        onsager_correction: Whether to apply Onsager correction (default: False)
        
    Returns:
        new_factors: Updated factor estimates
        new_factor_vars: Updated variance estimates
        s_values: Current damped residual state (for next Onsager correction)
        svar_values: Current damped output precision state
    """
    n = len(factors)
    C, M = F.shape
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Forward pass
    Z_hat = forward_pass_tensor(factors, F, indices)
    
    gathered = torch.stack([
        factors[d][indices[d].long()] for d in range(n)
    ])  # (n, C, M)
    gathered_var = torch.stack([
        factor_vars[d][indices[d].long()] for d in range(n)
    ])
    F_sq = torch.ones_like(F) if is_ising else F.pow(2)
    mean_sq = gathered.pow(2)
    mean_sq_product = mean_sq.prod(dim=0)
    second_product = (mean_sq + gathered_var).prod(dim=0)
    pvar = alpha_scale_sq * (F_sq * torch.clamp(second_product - mean_sq_product, min=0.0)).sum(dim=1) + 1e-10
    zvar_sum = torch.zeros(C, M, device=F.device, dtype=gathered.dtype)
    for d in range(n):
        if n > 1:
            other_mean_sq_product = torch.stack([
                mean_sq[dd] for dd in range(n) if dd != d
            ]).prod(dim=0)
        else:
            other_mean_sq_product = torch.ones(C, M, device=F.device, dtype=gathered.dtype)
        zvar_sum += gathered_var[d] * other_mean_sq_product
    zvar = alpha_scale_sq * (F_sq * zvar_sum).sum(dim=1) + 1e-10

    # Onsager correction (optional, default OFF)
    phat = Z_hat
    if onsager_correction and prev_s is not None:
        phat = Z_hat - zvar * prev_s

    # Residual
    denom = torch.clamp(pvar + noise_var, min=1e-6)
    s_new = torch.clamp((Y - phat) / denom, min=-1e6, max=1e6)
    svar_new = 1.0 / denom
    s_values = damped_state(s_new, prev_s if onsager_correction else None, damping)
    svar_values = damped_state(svar_new, prev_svar if onsager_correction else None, damping)
    
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
            other_second_product = torch.stack([
                gathered[dd].pow(2) + gathered_var[dd] for dd in other_indices
            ]).prod(dim=0)
        else:
            other_product = torch.ones(C, M, device=F.device, dtype=F.dtype)
            other_second_product = torch.ones(C, M, device=F.device, dtype=F.dtype)
        
        # r contribution: (1/√M) F * other_product * s
        r_contrib = alpha_scale * F * other_product * s_values.unsqueeze(1)  # (C, M)
        
        # tau contribution
        other_sq_product = other_product.pow(2)
        other_var_product = torch.clamp(other_second_product - other_sq_product, min=0.0)
        tau_contrib = alpha_scale_sq * F_sq * other_sq_product * svar_values.unsqueeze(1)
        gain_contrib = alpha_scale_sq * F_sq * other_var_product * svar_values.unsqueeze(1)
        
        # Scatter add
        r_d = torch.zeros(N_d, M, device=F.device, dtype=F.dtype)
        tau_d = torch.zeros(N_d, M, device=F.device, dtype=F.dtype)
        gain_d = torch.zeros(N_d, M, device=F.device, dtype=F.dtype)
        idx_exp = indices[d].long().unsqueeze(1).expand(-1, M)
        r_d.scatter_add_(0, idx_exp, r_contrib)
        tau_d.scatter_add_(0, idx_exp, tau_contrib)
        gain_d.scatter_add_(0, idx_exp, gain_contrib)
        tau_d = tau_d.clamp(min=1e-10)
        
        # Update
        new_factor_d, new_var_d = gaussian_posterior_update(
            factors[d], r_d, tau_d, prior_precision_base, prior_variance, gain_d
        )
        
        # Stability check: clamp factors to prevent explosion
        new_factor_d = torch.clamp(new_factor_d, min=-10.0, max=10.0)
        
        # Damping
        new_factor_d = blend_new_old(new_factor_d, factors[d], damping)
        new_var_d = torch.clamp(
            blend_new_old(new_var_d, factor_vars[d], damping),
            min=1e-8,
            max=prior_variance,
        )
        
        new_factors.append(new_factor_d)
        new_vars.append(new_var_d)
    
    return new_factors, new_vars, s_values, svar_values
