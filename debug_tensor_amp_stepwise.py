#!/usr/bin/env python3
"""
Debug Tensor AMP Step-by-Step

Run AMP for a few steps and log detailed statistics at each step.
"""

import torch
import sys
sys.path.insert(0, "/home/sucia/Matrix_Factorization/src")

from matrix_factorization.modules.algorithms.bigamp.tensor_supergraph import (
    create_tensor_supergraph, create_tensor_superdata
)
from matrix_factorization.modules.algorithms.bigamp.tensor_step_super import (
    tensor_step_super, forward_pass_tensor_super
)

def cp_contract_factors(factors):
    """Compute CP tensor from factors."""
    n = len(factors)
    letters = 'abcdefghij'[:n]
    pattern = ','.join([f'{letters[d]}m' for d in range(n)]) + '->' + letters
    return torch.einsum(pattern, *factors)

def compute_cosine_similarity(student_factors, teacher_factors):
    """Compute Q_Y (cosine similarity) between student and teacher tensors."""
    T_student = cp_contract_factors(student_factors)
    T_teacher = cp_contract_factors(teacher_factors)
    inner = (T_student * T_teacher).sum()
    norm_s = T_student.norm()
    norm_t = T_teacher.norm()
    return (inner / (norm_s * norm_t + 1e-12)).item()

def run_debug():
    device = torch.device('cuda')
    
    # Small problem for fast debugging
    N = 20
    M = 5
    n = 3
    alpha = 8.0  # High alpha where AGD works
    S = 1  # single sample
    
    print("="*60)
    print(f"Debug Tensor AMP: N={N}, M={M}, n={n}, alpha={alpha}")
    print("="*60)
    
    # Generate teacher
    torch.manual_seed(42)
    teacher_factors = [torch.randn(N, M, device=device) for _ in range(n)]
    
    # Create supergraph using the proper API
    dims = (N,) * n
    supergraph = create_tensor_supergraph(
        dims=dims,
        alpha_values=[alpha],  # Single alpha
        M=M,
        S=S,
        seed=42,
        device=device
    )
    
    print(f"Supergraph: C_max={supergraph.C_max} edges")
    
    # Create superdata with teacher
    superdata = create_tensor_superdata(
        supergraph=supergraph,
        teacher_factors=teacher_factors,
        f_distribution='rademacher',
        seed=12345
    )
    
    # Get flat tensors
    F_flat, Y_flat = superdata.get_flat_tensors()
    offset_indices = supergraph.get_offset_indices()
    
    print(f"F_flat shape: {F_flat.shape}, Y_flat shape: {Y_flat.shape}")
    print(f"Y norm: {Y_flat.norm():.4f}")
    
    # Initialize student randomly - std=1.0 to match teacher!
    torch.manual_seed(123)
    student_factors = [torch.randn(1, S*N, M, device=device) * 1.0 for _ in range(n)]  # Changed from 0.1 to 1.0
    factor_vars = [torch.ones(1, S*N, M, device=device) for _ in range(n)]
    
    q_init = compute_cosine_similarity(
        [f[0, :N, :] for f in student_factors], teacher_factors  # First sample, original dims
    )
    print(f"Initial Q_Y: {q_init:.6f}")
    
    # Create alpha_mask (all True for single alpha)
    alpha_mask = superdata.alpha_mask_exp  # (A, S*C_max)
    
    # Run AMP steps
    prev_s = None
    damping = 1.0  # Disable damping to test pure algorithm
    noise_var = 1e-5
    
    print("\n" + "="*60)
    print("AMP Step Debug:")
    print("="*60)
    
    for step in range(20):
        # Single AMP step
        new_factors, new_vars, s_values = tensor_step_super(
            factors=student_factors,
            factor_vars=factor_vars,
            Y_flat=Y_flat,
            F_flat=F_flat,
            offset_indices=offset_indices,
            S=S,
            N_dims=dims,
            M=M,
            alpha_mask=alpha_mask,
            damping=damping,
            noise_var=noise_var,
            is_rademacher=True,
            prev_s=prev_s,
            onsager_correction=False
        )
        
        # Update
        student_factors = new_factors
        factor_vars = new_vars
        prev_s = s_values
        
        # Statistics
        q_y = compute_cosine_similarity(
            [f[0, :N, :] for f in student_factors], teacher_factors
        )
        
        factor_means = [f.mean().item() for f in student_factors]
        factor_stds = [f.std().item() for f in student_factors]
        s_mean = s_values.mean().item()
        s_std = s_values.std().item()
        
        print(f"Step {step+1:2d}: Q_Y={q_y:+.6f} | "
              f"f_std=[{factor_stds[0]:.3f},{factor_stds[1]:.3f},{factor_stds[2]:.3f}] | "
              f"s_mean={s_mean:+.3f}, s_std={s_std:.3f}")
    
    print("\n" + "="*60)
    print(f"Final Q_Y: {q_y:.6f}")
    if q_y < 0.1:
        print("FAILURE: AMP is not converging!")
        print("\nDiagnosis suggestions:")
        print("  - Check if factor_std is decreasing (factors collapsing to 0)")
        print("  - Check if s_values are exploding")
        print("  - Check if Q_Y is oscillating")
    print("="*60)

if __name__ == "__main__":
    run_debug()
