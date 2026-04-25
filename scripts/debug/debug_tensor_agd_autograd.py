#!/usr/bin/env python3
"""
Tensor AGD Baseline Test (AutoGrad Version)

Uses PyTorch autograd for gradient computation to ensure correctness.
"""

import torch
import math
from typing import List

def cp_contract_factors(factors: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute CP tensor from factors: T = sum_m outer(f0[:,m], f1[:,m], ..., fn[:,m])
    Uses opt_einsum for efficiency.
    """
    n = len(factors)
    M = factors[0].shape[1]
    
    # Build einsum pattern: 'am,bm,cm->abc' for n=3
    letters = 'abcdefghij'[:n]
    pattern = ','.join([f'{letters[d]}m' for d in range(n)]) + '->' + letters
    
    return torch.einsum(pattern, *factors)

def run_tensor_agd_test_autograd(N: int = 20, M: int = 5, n: int = 3, 
                                  alpha: float = 4.0, lr: float = 0.01, 
                                  max_steps: int = 2000):
    """
    Run Tensor AGD test using PyTorch autograd.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Test: N={N}, M={M}, n={n}, alpha={alpha}")
    
    # Generate teacher factors ~ N(0, 1)
    torch.manual_seed(42)
    teacher_factors = [torch.randn(N, M, device=device) for _ in range(n)]
    
    # Generate teacher tensor
    T_teacher = cp_contract_factors(teacher_factors)
    print(f"Teacher tensor shape: {T_teacher.shape}")
    print(f"Teacher tensor norm: {T_teacher.norm():.4f}")
    
    # Generate observation mask
    dof = n * N * M
    num_obs = int(alpha * dof)
    total_entries = N ** n
    obs_ratio = num_obs / total_entries
    print(f"Observation ratio: {obs_ratio:.4f} ({num_obs}/{total_entries})")
    
    mask = (torch.rand(tuple([N] * n), device=device) < obs_ratio).float()
    actual_obs = mask.sum().item()
    print(f"Actual observations: {int(actual_obs)}")
    
    # Observed data (no noise for simplicity)
    Y = T_teacher * mask
    
    # Initialize student factors - requires_grad for optimization
    torch.manual_seed(123)
    student_factors = [
        torch.nn.Parameter(torch.randn(N, M, device=device) * 0.1)
        for _ in range(n)
    ]
    
    # Optimizer
    optimizer = torch.optim.Adam(student_factors, lr=lr)
    
    # Initial Q_Y
    with torch.no_grad():
        T_student = cp_contract_factors(student_factors)
        inner = (T_student * T_teacher).sum()
        norm_s = T_student.norm()
        norm_t = T_teacher.norm()
        q_init = (inner / (norm_s * norm_t + 1e-12)).item()
        print(f"Initial Q_Y (Cosine): {q_init:.6f}")
    
    # Training loop
    print(f"\nRunning {max_steps} AGD steps (Adam optimizer)...")
    for step in range(max_steps):
        optimizer.zero_grad()
        
        # Forward: compute student tensor
        T_student = cp_contract_factors(student_factors)
        
        # Loss: MSE on observed entries
        loss = ((T_student - T_teacher) ** 2 * mask).sum() / mask.sum()
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(student_factors, max_norm=1.0)
        
        # Update
        optimizer.step()
        
        # Log every 200 steps
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                T_student = cp_contract_factors(student_factors)
                inner = (T_student * T_teacher).sum()
                norm_s = T_student.norm()
                norm_t = T_teacher.norm()
                q_y = (inner / (norm_s * norm_t + 1e-12)).item()
                mse = ((T_student - T_teacher) ** 2).mean().item()
                print(f"Step {step+1}: Q_Y={q_y:.6f}, Full MSE={mse:.6f}, Loss={loss.item():.6f}")
    
    # Final Q_Y
    with torch.no_grad():
        T_student = cp_contract_factors(student_factors)
        inner = (T_student * T_teacher).sum()
        norm_s = T_student.norm()
        norm_t = T_teacher.norm()
        q_final = (inner / (norm_s * norm_t + 1e-12)).item()
        final_mse = ((T_student - T_teacher) ** 2).mean().item()
    
    print(f"\n{'='*60}")
    print(f"Final Q_Y (Cosine): {q_final:.6f}")
    print(f"Final MSE (Full Tensor): {final_mse:.6f}")
    print(f"{'='*60}")
    
    return q_final

if __name__ == "__main__":
    print("="*60)
    print("Tensor AGD Baseline Test (AutoGrad Version)")
    print("="*60)
    
    # Test with N=50, M=10, n=3, alpha=5.0 (Physical Limit)
    q_y = run_tensor_agd_test_autograd(N=50, M=10, n=3, alpha=5.0, lr=0.01, max_steps=5000)
    
    print("\n" + "="*60)
    if q_y > 0.9:
        print("✓ SUCCESS: AGD converged, Q_Y metric calculation is CORRECT!")
        print("  Problem is in AMP algorithm, not metrics.")
    elif q_y > 0.5:
        print("? PARTIAL: AGD partially converged")
        print("  May need more steps, higher alpha, or different hyperparams.")
    else:
        print("✗ FAILURE: AGD did not converge!")
        print("  Problem may be in tensor construction or underdetermined regime.")
    print("="*60)
