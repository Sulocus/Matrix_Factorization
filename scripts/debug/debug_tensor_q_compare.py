#!/usr/bin/env python3
"""
Debug Script: Compare Q_Y vs Q_Full_Tensor

Validates user hypothesis:
- Q_Y (observation-only) should be ≈ 1 (perfect fit to observed edges)
- Q_Full_Tensor (full tensor) should be << 1 in Cold Start (no learning)
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
    forward_pass_tensor_super, tensor_step_super
)
from matrix_factorization.modules.metrics.tensor_metrics import (
    compute_tensor_reconstruction_q, compute_tensor_cosine
)


def main():
    """Compare Q_Y vs Q_Full_Tensor."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ========== Configuration ==========
    dims = (50, 50, 50)
    n = 3
    M = 10
    S = 4
    alpha_values = [0.1, 1.0, 4.0]
    A = len(alpha_values)
    seed = 42
    
    print(f"\n=== Configuration ===")
    print(f"Dims: {dims}, M: {M}, S: {S}, Total Tensor Elements: {dims[0]**3}")
    print(f"Alpha values: {alpha_values}")
    
    # ========== Create Teacher ==========
    torch.manual_seed(seed)
    teacher_factors = [
        torch.randn(dims[d], M, device=device)
        for d in range(n)
    ]
    for d in range(n):
        teacher_factors[d] = teacher_factors[d] / teacher_factors[d].std()
    
    # ========== Create Data ==========
    supergraph = create_tensor_supergraph(dims, alpha_values, M, S, seed, device)
    superdata = create_tensor_superdata(supergraph, teacher_factors, 'rademacher', seed + 1000)
    
    F_flat, Y_flat = superdata.get_flat_tensors()
    offset_indices = supergraph.get_offset_indices()
    alpha_mask_exp = superdata.alpha_mask_exp
    
    print(f"\nC_per_alpha: {supergraph.C_per_alpha}")
    print(f"C_max: {supergraph.C_max}")
    
    # ========== Initialize Cold Start Student ==========
    torch.manual_seed(12345)
    factors = [torch.randn(A, S * dims[d], M, device=device) for d in range(n)]
    factor_vars = [torch.ones_like(f) for f in factors]
    
    # ========== Run 200 steps of AMP ==========
    print(f"\n=== Running 200 AMP steps ===")
    
    for step in range(200):
        factors, factor_vars, _ = tensor_step_super(
            factors, factor_vars, Y_flat, F_flat, offset_indices,
            S, list(dims), M, alpha_mask_exp,
            damping=0.5, noise_var=1.0, is_rademacher=True,
            prev_s=None, onsager_correction=False
        )
    
    # ========== Compute Q_Y (observation-only) ==========
    print(f"\n=== Q_Y (Observation-Only MSE) ===")
    
    Z_hat = forward_pass_tensor_super(factors, F_flat, offset_indices, S, list(dims))
    Y_exp = Y_flat.unsqueeze(0).expand(A, -1)
    
    diff_sq = ((Z_hat - Y_exp) ** 2) * alpha_mask_exp.float()
    edge_counts = alpha_mask_exp.sum(dim=1).float()
    mse_per_alpha = diff_sq.sum(dim=1) / (edge_counts + 1e-10)
    
    y_var = Y_flat.var().item()
    Q_Y_per_alpha = (1.0 - mse_per_alpha / y_var).cpu().tolist()
    
    for a_idx, alpha in enumerate(alpha_values):
        print(f"  Alpha={alpha}: Q_Y = {Q_Y_per_alpha[a_idx]:.6f} (C={supergraph.C_per_alpha[a_idx]} observations)")
    
    # ========== Compute Q_Full_Tensor (full tensor) ==========
    print(f"\n=== Q_Full_Tensor (Full Tensor Reconstruction) ===")
    
    # Reshape factors: (A, S*N_d, M) -> (A, S, N_d, M) -> mean over S -> (A, N_d, M)
    factors_reshaped = []
    for d in range(n):
        N_d = dims[d]
        f_d = factors[d].view(A, S, N_d, -1)
        f_d_mean = f_d.mean(dim=1)  # (A, N_d, M)
        factors_reshaped.append(f_d_mean)
    
    for a_idx, alpha in enumerate(alpha_values):
        student_factors_a = [f[a_idx] for f in factors_reshaped]
        q_full = compute_tensor_reconstruction_q(teacher_factors, student_factors_a)
        cosine = compute_tensor_cosine(teacher_factors, student_factors_a)
        
        print(f"  Alpha={alpha}: Q_Full = {q_full:.6f}, Cosine = {cosine:.6f}")
    
    # ========== Summary ==========
    print(f"\n=== Summary ===")
    print(f"Q_Y measures: How well does Student fit the {supergraph.C_per_alpha[0]}-{supergraph.C_per_alpha[-1]} observed edges?")
    print(f"Q_Full measures: How well does Student reconstruct the {dims[0]**3} element full tensor?")
    print(f"")
    print(f"If Q_Y ≈ 1 but Q_Full << 1:")
    print(f"  -> Student overfits observations but doesn't learn the Teacher")
    print(f"  -> This is EXPECTED in underdetermined systems")
    print(f"  -> Report Q_Full, not Q_Y!")


if __name__ == '__main__':
    main()
