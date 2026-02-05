#!/usr/bin/env python3
"""
Debug Script: Multi-Step AMP Iteration Test

This script runs multiple AMP steps to observe MSE evolution and detect
when the spurious Q≈1 convergence starts happening.
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


def main():
    """Run multi-step AMP test."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ========== Configuration ==========
    dims = (50, 50, 50)  # N x N x N tensor
    n = 3
    M = 10
    S = 4  # samples per alpha
    alpha_values = [0.1, 0.5, 1.0, 2.0, 4.0]
    A = len(alpha_values)
    seed = 42
    
    MAX_STEPS = 200   # More steps than single-step test
    PRINT_EVERY = 20
    DAMPING = 0.5
    NOISE_VAR_FIXED = 1.0  # Fixed noise, no annealing
    USE_ONSAGER = False    # Disable Onsager correction
    
    print(f"\n=== Configuration ===")
    print(f"Dims: {dims}, M: {M}, S: {S}")
    print(f"Alpha values: {alpha_values}")
    print(f"MAX_STEPS: {MAX_STEPS}, DAMPING: {DAMPING}, NOISE_VAR: {NOISE_VAR_FIXED}")
    print(f"Onsager Correction: {USE_ONSAGER}")
    
    # ========== Create Teacher Factors ==========
    torch.manual_seed(seed)
    teacher_factors = [
        torch.randn(dims[d], M, device=device)
        for d in range(n)
    ]
    # Normalize to unit
    for d in range(n):
        teacher_factors[d] = teacher_factors[d] / teacher_factors[d].std()
    
    # ========== Create Supergraph and Superdata ==========
    supergraph = create_tensor_supergraph(dims, alpha_values, M, S, seed, device)
    superdata = create_tensor_superdata(supergraph, teacher_factors, 'rademacher', seed + 1000)
    
    F_flat, Y_flat = superdata.get_flat_tensors()
    offset_indices = supergraph.get_offset_indices()
    alpha_mask_exp = superdata.alpha_mask_exp
    
    print(f"\n=== Supergraph ===")
    print(f"  C_max: {supergraph.C_max}, C_per_alpha: {supergraph.C_per_alpha}")
    
    # ========== Initialize Random Student Factors (Cold Start) ==========
    torch.manual_seed(12345)  # Different seed from teacher
    factors = [
        torch.randn(A, S * dims[d], M, device=device)
        for d in range(n)
    ]
    factor_vars = [torch.ones_like(f) for f in factors]
    
    print(f"\n=== Initial State ===")
    for d in range(n):
        print(f"  Factor {d}: mean={factors[d].mean():.4f}, std={factors[d].std():.4f}")
    
    # Store Y for computing metrics  
    Y_exp = Y_flat.unsqueeze(0).expand(A, -1)
    y_var = Y_flat.var().item()
    print(f"  Y variance: {y_var:.4f}")
    
    # ========== Multi-Step AMP Loop ==========
    print(f"\n=== Multi-Step AMP Iteration ===")
    print(f"{'Step':>6} | {'MSE_a0':>10} | {'MSE_a4':>10} | {'Q_a0':>10} | {'Q_a4':>10} | {'FactorStd':>10}")
    print("-" * 70)
    
    prev_s = None
    
    for step in range(MAX_STEPS):
        # NO NOISE ANNEALING - Fixed noise_var
        current_noise_var = NOISE_VAR_FIXED
        
        # AMP Step
        factors, factor_vars, new_s = tensor_step_super(
            factors=factors,
            factor_vars=factor_vars,
            Y_flat=Y_flat,
            F_flat=F_flat,
            offset_indices=offset_indices,
            S=S,
            N_dims=list(dims),
            M=M,
            alpha_mask=alpha_mask_exp,
            damping=DAMPING,
            noise_var=current_noise_var,
            is_rademacher=True,
            prev_s=prev_s if USE_ONSAGER else None,
            onsager_correction=USE_ONSAGER,
        )
        
        if USE_ONSAGER:
            prev_s = new_s
        
        # Print progress
        if step % PRINT_EVERY == 0 or step == MAX_STEPS - 1:
            with torch.no_grad():
                Z_hat = forward_pass_tensor_super(factors, F_flat, offset_indices, S, list(dims))
                diff_sq = ((Z_hat - Y_exp) ** 2) * alpha_mask_exp.float()
                edge_counts = alpha_mask_exp.sum(dim=1).float()
                mse_per_alpha = diff_sq.sum(dim=1) / (edge_counts + 1e-10)
                Q_per_alpha = 1.0 - mse_per_alpha / y_var
                
                mse_a0 = mse_per_alpha[0].item()
                mse_a4 = mse_per_alpha[4].item()
                q_a0 = Q_per_alpha[0].item()
                q_a4 = Q_per_alpha[4].item()
                f_std = factors[0].std().item()
                
                print(f"{step:>6} | {mse_a0:>10.4f} | {mse_a4:>10.4f} | {q_a0:>10.4f} | {q_a4:>10.4f} | {f_std:>10.4f}")
    
    # ========== Final State ==========
    print(f"\n=== Final State (Step {MAX_STEPS}) ===")
    with torch.no_grad():
        Z_hat = forward_pass_tensor_super(factors, F_flat, offset_indices, S, list(dims))
        diff_sq = ((Z_hat - Y_exp) ** 2) * alpha_mask_exp.float()
        edge_counts = alpha_mask_exp.sum(dim=1).float()
        mse_per_alpha = diff_sq.sum(dim=1) / (edge_counts + 1e-10)
        Q_per_alpha = 1.0 - mse_per_alpha / y_var
        
        print(f"MSE per alpha: {mse_per_alpha.tolist()}")
        print(f"Q per alpha: {Q_per_alpha.tolist()}")
        
        for d in range(n):
            print(f"Factor {d}: mean={factors[d].mean():.4f}, std={factors[d].std():.4f}")
    
    # ========== Correlation with Teacher ==========
    print(f"\n=== Correlation with Teacher ===")
    # Reshape factors to (A, S, N_d, M) and average over S
    for d in range(n):
        f_reshaped = factors[d].view(A, S, dims[d], M)
        f_mean = f_reshaped.mean(dim=1)  # (A, N_d, M)
        
        # For alpha=0.1 (index 0), compute correlation with teacher
        student_d_a0 = f_mean[0].flatten()  # (N_d * M,)
        teacher_d = teacher_factors[d].flatten()
        
        # Normalize
        student_norm = student_d_a0 - student_d_a0.mean()
        teacher_norm = teacher_d - teacher_d.mean()
        
        corr = (student_norm * teacher_norm).sum() / (student_norm.norm() * teacher_norm.norm() + 1e-10)
        print(f"  Factor {d} (Alpha=0.1) correlation with Teacher: {corr.item():.4f}")
    
    print(f"\n=== Done ===")


if __name__ == '__main__':
    main()
