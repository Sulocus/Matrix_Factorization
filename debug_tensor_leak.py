#!/usr/bin/env python3
"""
Debug Script: Tensor Algorithm Data Leakage Detection

This script isolates and tests the tensor algorithm initialization and 
forward pass to detect if there is data leakage causing Q=1 for all alphas.
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
    """Run diagnostic tests."""
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
    
    print(f"\n=== Configuration ===")
    print(f"Dims: {dims}, M: {M}, S: {S}")
    print(f"Alpha values: {alpha_values}")
    
    # ========== Create Teacher Factors ==========
    torch.manual_seed(seed)
    teacher_factors = [
        torch.randn(dims[d], M, device=device)
        for d in range(n)
    ]
    # Normalize to unit
    for d in range(n):
        teacher_factors[d] = teacher_factors[d] / teacher_factors[d].std()
    
    print(f"\n=== Teacher Factors ===")
    for d, t in enumerate(teacher_factors):
        print(f"  Factor {d}: shape={t.shape}, mean={t.mean():.4f}, std={t.std():.4f}")
    
    # ========== Create Supergraph and Superdata ==========
    supergraph = create_tensor_supergraph(dims, alpha_values, M, S, seed, device)
    superdata = create_tensor_superdata(supergraph, teacher_factors, 'rademacher', seed + 1000)
    
    F_flat, Y_flat = superdata.get_flat_tensors()
    offset_indices = supergraph.get_offset_indices()
    
    print(f"\n=== Supergraph ===")
    print(f"  C_max: {supergraph.C_max}, S*C_max: {S * supergraph.C_max}")
    print(f"  C_per_alpha: {supergraph.C_per_alpha}")
    print(f"  alpha_mask shape: {supergraph.alpha_mask.shape}")
    print(f"  alpha_mask[0] sum (Alpha=0.1): {supergraph.alpha_mask[0].sum().item()} (expected: {supergraph.C_per_alpha[0]})")
    
    print(f"\n=== Y Statistics ===")
    print(f"  Y_flat shape: {Y_flat.shape}")
    print(f"  Y_flat mean: {Y_flat.mean():.6f}, std: {Y_flat.std():.6f}")
    
    # ========== Test 1: Z_hat with Teacher Factors ==========
    # If we use teacher factors directly, Z_hat should equal Y
    print(f"\n=== Test 1: Z_hat with Teacher Factors ===")
    
    # Expand teacher to student format: (A, S*N_d, M)
    teacher_as_student = []
    for d in range(n):
        t_exp = teacher_factors[d].unsqueeze(0).unsqueeze(0).expand(A, S, -1, -1)
        t_exp = t_exp.reshape(A, S * dims[d], M)
        teacher_as_student.append(t_exp)
    
    Z_teacher = forward_pass_tensor_super(teacher_as_student, F_flat, offset_indices, S, list(dims))
    
    Y_exp = Y_flat.unsqueeze(0).expand(A, -1)
    diff = (Z_teacher - Y_exp).abs()
    print(f"  Z_teacher shape: {Z_teacher.shape}")
    print(f"  Max |Z_teacher - Y|: {diff.max().item():.2e} (should be ~0)")
    
    # ========== Test 2: Z_hat with Random Factors (Cold Start) ==========
    print(f"\n=== Test 2: Z_hat with Random Factors (Cold Start) ===")
    
    torch.manual_seed(12345)  # Different seed
    random_factors = [
        torch.randn(A, S * dims[d], M, device=device)
        for d in range(n)
    ]
    
    Z_random = forward_pass_tensor_super(random_factors, F_flat, offset_indices, S, list(dims))
    diff_random = (Z_random - Y_exp).abs()
    print(f"  Z_random shape: {Z_random.shape}")
    print(f"  Max |Z_random - Y|: {diff_random.max().item():.4f} (should be >> 0)")
    
    # Compute MSE per alpha
    alpha_mask_exp = superdata.alpha_mask_exp
    diff_sq_masked = ((Z_random - Y_exp) ** 2) * alpha_mask_exp.float()
    edge_counts = alpha_mask_exp.sum(dim=1).float()
    mse_per_alpha = diff_sq_masked.sum(dim=1) / (edge_counts + 1e-10)
    print(f"  MSE per alpha: {mse_per_alpha[:5].tolist()}")
    
    # Q = 1 - MSE / Var(Y)
    y_var = Y_flat.var()
    Q_random = 1.0 - mse_per_alpha / y_var
    print(f"  Q per alpha (should be << 1): {Q_random[:5].tolist()}")
    
    # ========== Test 3: Single AMP Step ==========
    print(f"\n=== Test 3: Single AMP Step ===")
    
    factor_vars = [torch.ones_like(f) for f in random_factors]
    
    new_factors, new_vars, s_values = tensor_step_super(
        factors=random_factors,
        factor_vars=factor_vars,
        Y_flat=Y_flat,
        F_flat=F_flat,
        offset_indices=offset_indices,
        S=S,
        N_dims=list(dims),
        M=M,
        alpha_mask=alpha_mask_exp,
        damping=0.5,
        noise_var=1e-5,
        is_rademacher=True,
        prev_s=None,
        onsager_correction=False,
    )
    
    Z_after_step = forward_pass_tensor_super(new_factors, F_flat, offset_indices, S, list(dims))
    diff_after = (Z_after_step - Y_exp).abs()
    
    print(f"  Max |Z_after_step - Y|: {diff_after.max().item():.4f}")
    
    # ========== Test 4: Check offset_indices correctness ==========
    print(f"\n=== Test 4: offset_indices Range Check ===")
    for d in range(n):
        idx = offset_indices[d]
        max_valid = S * dims[d] - 1
        print(f"  Dim {d}: offset_indices range: [{idx.min().item()}, {idx.max().item()}], valid range: [0, {max_valid}]")
        if idx.max().item() > max_valid:
            print(f"    ⚠️ WARNING: indices exceed valid range!")
    
    # ========== Test 5: Check if different Alphas share Factor updates ==========
    print(f"\n=== Test 5: Alpha Independence Check ===")
    
    # After AMP step, check if Alpha 0 and Alpha 4 have different factors
    for d in range(n):
        f0 = new_factors[d][0]  # Alpha 0.1
        f4 = new_factors[d][4]  # Alpha 4.0
        correlation = torch.corrcoef(torch.stack([f0.flatten(), f4.flatten()]))[0, 1].item()
        print(f"  Dim {d}: Correlation between Alpha 0.1 and Alpha 4.0 factors: {correlation:.4f}")
    
    print(f"\n=== Done ===")


if __name__ == '__main__':
    main()
