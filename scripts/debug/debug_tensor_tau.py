#!/usr/bin/env python3
"""
Debug Script: Analyze tau values to understand Prior constraint strength.

This script examines why Q_Y can reach 1.0 in underdetermined systems
by analyzing the precision (tau) values in the AMP update.
"""

import torch
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from matrix_factorization.modules.algorithms.bigamp.tensor_supergraph import (
    create_tensor_supergraph, create_tensor_superdata
)
from matrix_factorization.modules.algorithms.bigamp.tensor_step_super import (
    forward_pass_tensor_super, compute_variance_tensor_super
)


def main():
    """Analyze tau values."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ========== Configuration ==========
    dims = (50, 50, 50)
    n = 3
    M = 10
    S = 4
    alpha_values = [0.1, 0.5, 1.0, 2.0, 4.0]
    A = len(alpha_values)
    seed = 42
    
    print(f"\n=== Configuration ===")
    print(f"Dims: {dims}, M: {M}, S: {S}")
    print(f"Alpha values: {alpha_values}")
    
    # ========== Create Teacher and Data ==========
    torch.manual_seed(seed)
    teacher_factors = [
        torch.randn(dims[d], M, device=device) / math.sqrt(dims[d])
        for d in range(n)
    ]
    for d in range(n):
        teacher_factors[d] = teacher_factors[d] / teacher_factors[d].std()
    
    supergraph = create_tensor_supergraph(dims, alpha_values, M, S, seed, device)
    superdata = create_tensor_superdata(supergraph, teacher_factors, 'rademacher', seed + 1000)
    
    F_flat, Y_flat = superdata.get_flat_tensors()
    offset_indices = supergraph.get_offset_indices()
    alpha_mask_exp = superdata.alpha_mask_exp
    
    # ========== Initialize Random Student ==========
    torch.manual_seed(12345)
    factors = [
        torch.randn(A, S * dims[d], M, device=device)
        for d in range(n)
    ]
    factor_vars = [torch.ones_like(f) for f in factors]
    
    # ========== Compute tau manually ==========
    print(f"\n=== Tau Analysis ===")
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    SC = F_flat.shape[0]
    noise_var = 1.0
    
    # Forward pass
    Z_hat = forward_pass_tensor_super(factors, F_flat, offset_indices, S, list(dims))
    V = compute_variance_tensor_super(factors, factor_vars, F_flat, offset_indices, S, list(dims), is_rademacher=True, alpha_mask=alpha_mask_exp)
    
    denom = torch.clamp(V + noise_var, min=1e-5)
    
    print(f"V (Variance) stats: mean={V.mean():.4f}, max={V.max():.4f}")
    print(f"Denom stats: mean={denom.mean():.4f}, max={denom.max():.4f}")
    
    # Gather factors
    gathered = torch.stack([factors[d][:, offset_indices[d].long()] for d in range(n)])
    gathered_var = torch.stack([factor_vars[d][:, offset_indices[d].long()] for d in range(n)])
    
    F_sq = torch.ones_like(F_flat)  # Rademacher
    
    # Compute tau for dimension 0
    d = 0
    N_d = dims[d]
    SN_d = S * N_d
    
    other_indices = [i for i in range(n) if i != d]
    other_var_list = [factor_vars[i][:, offset_indices[i].long()] for i in other_indices]
    other_second_moment = torch.stack([
        gathered[i].pow(2) + other_var_list[j] 
        for j, i in enumerate(other_indices)
    ]).prod(dim=0)
    
    print(f"\nother_second_moment stats: mean={other_second_moment.mean():.4f}")
    
    mask_exp = alpha_mask_exp.unsqueeze(2).float()
    tau_contrib = alpha_scale_sq * F_sq.unsqueeze(0) * other_second_moment / denom.unsqueeze(2) * mask_exp
    
    print(f"tau_contrib per edge stats: mean={tau_contrib.mean():.6f}, max={tau_contrib.max():.6f}")
    
    # Scatter add to get tau per node
    tau_d = torch.zeros(A, SN_d, M, device=device, dtype=factors[0].dtype)
    idx_exp = offset_indices[d].unsqueeze(0).unsqueeze(2).expand(A, -1, M)
    tau_d.scatter_add_(1, idx_exp, tau_contrib.to(dtype=tau_d.dtype))
    
    print(f"\ntau_d (dimension 0) after scatter_add:")
    print(f"  Mean: {tau_d.mean():.6f}")
    print(f"  Max: {tau_d.max():.6f}")
    print(f"  Min (non-zero): {tau_d[tau_d > 0].min():.6f if (tau_d > 0).any() else 0.0}")
    
    # Posterior variance
    new_var_d = 1.0 / (1.0 + tau_d)
    print(f"\nposterior_var = 1/(1+tau):")
    print(f"  Mean: {new_var_d.mean():.6f}")
    print(f"  Min: {new_var_d.min():.6f}")
    
    # How much of factor space is constrained?
    non_zero_tau_fraction = (tau_d > 1e-6).float().mean()
    print(f"\nFraction of factor elements with non-trivial tau: {non_zero_tau_fraction:.4f}")
    
    # ========== Key Insight: Per-Alpha Statistics ==========
    print(f"\n=== Per-Alpha Tau Statistics ===")
    for a_idx, alpha in enumerate(alpha_values):
        tau_a = tau_d[a_idx]
        non_zero_fraction = (tau_a > 1e-6).float().mean().item()
        mean_tau = tau_a.mean().item()
        mean_var = new_var_d[a_idx].mean().item()
        print(f"Alpha={alpha:.1f}: non_zero={non_zero_fraction:.4f}, mean_tau={mean_tau:.6f}, mean_posterior_var={mean_var:.6f}")
    
    # ========== Calculate Effective Prior Strength ==========
    print(f"\n=== Effective Prior Strength ===")
    print(f"Prior precision = 1.0")
    print(f"If tau << 1, Prior dominates (posterior_var ≈ 1, small updates)")
    print(f"If tau >> 1, Observation dominates (posterior_var ≈ 1/tau, large updates)")
    
    for a_idx, alpha in enumerate(alpha_values):
        tau_a = tau_d[a_idx]
        mean_tau = tau_a.mean().item()
        prior_strength = 1.0 / (1.0 + mean_tau)  # How much Prior contributes
        print(f"Alpha={alpha:.1f}: Prior strength = {prior_strength:.4f} (1.0 = fully Prior, 0.0 = fully Observation)")
    
    print(f"\n=== Done ===")


if __name__ == '__main__':
    main()
