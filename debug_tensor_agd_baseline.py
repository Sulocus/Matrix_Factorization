#!/usr/bin/env python3
"""
Tensor AGD (Alternating Gradient Descent) Baseline Test

Purpose: Validate tensor Q_Y metric calculation using a simple, known-correct algorithm.
If Q_Y fails to reach 1.0 with AGD, the problem is in the metric calculation, not AMP.

Algorithm: For each factor d, minimize || Y - T(factors) ||^2 using gradient descent.
"""

import torch
import math
from typing import List, Tuple

def cp_contract_factors(factors: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute CP tensor from factors: T = sum_m outer(f0[:,m], f1[:,m], ..., fn[:,m])
    
    Args:
        factors: List of n tensors, each of shape (N_d, M)
        
    Returns:
        Tensor of shape (N_0, N_1, ..., N_{n-1})
    """
    n = len(factors)
    M = factors[0].shape[1]
    
    # Build contracted tensor
    result = None
    for m in range(M):
        # Outer product of all factors for rank m
        outer = torch.ones(1, device=factors[0].device, dtype=factors[0].dtype)
        for d in range(n):
            outer = torch.outer(outer.flatten(), factors[d][:, m]).reshape(
                *outer.shape, factors[d].shape[0]
            )
        outer = outer.squeeze(0)  # Remove initial dim
        
        if result is None:
            result = outer
        else:
            result = result + outer
    
    return result

def compute_tensor_cosine(student_factors: List[torch.Tensor], 
                           teacher_factors: List[torch.Tensor]) -> float:
    """Compute cosine similarity between student and teacher tensors."""
    # Contract to full tensors
    T_student = cp_contract_factors(student_factors)
    T_teacher = cp_contract_factors(teacher_factors)
    
    # Cosine similarity
    inner = (T_student * T_teacher).sum()
    norm_s = (T_student ** 2).sum().sqrt()
    norm_t = (T_teacher ** 2).sum().sqrt()
    
    return (inner / (norm_s * norm_t + 1e-12)).item()

def tensor_agd_step(factors: List[torch.Tensor], 
                    Y: torch.Tensor, 
                    mask: torch.Tensor,
                    lr: float) -> List[torch.Tensor]:
    """
    One step of Tensor AGD.
    For each factor d, compute gradient and update.
    """
    n = len(factors)
    new_factors = []
    
    for d in range(n):
        # Compute current tensor
        T = cp_contract_factors(factors)
        
        # Residual on observed entries
        residual = (Y - T) * mask
        
        # Gradient for factor d: ∂L/∂factor[d] = -2 * (residual contracted with other factors)
        # For CP: gradient[d][:,m] = sum over other dims of residual * prod of other factors
        M = factors[d].shape[1]
        N_d = factors[d].shape[0]
        
        grad_d = torch.zeros_like(factors[d])
        
        for m in range(M):
            # Build contraction pattern
            # Other factors for this rank
            other_factors_m = [factors[j][:, m] for j in range(n) if j != d]
            
            # Contract residual with other factors
            # Result should be (N_d,)
            contracted = residual.clone()
            for j_idx, j in enumerate([jj for jj in range(n) if jj != d]):
                # Sum over dimension j, weighted by factor
                # Determine which axis corresponds to j
                axis = j if j < d else j - 1
                contracted = (contracted * other_factors_m[j_idx].reshape(
                    *([1] * axis + [-1] + [1] * (contracted.dim() - axis - 1))
                )).sum(dim=axis, keepdim=True).squeeze(axis)
            
            grad_d[:, m] = -2.0 * contracted.squeeze()
        
        # Gradient clipping for stability
        grad_d = torch.clamp(grad_d, min=-10.0, max=10.0)
        
        # Update with NaN check
        new_factor = factors[d] - lr * grad_d
        if torch.isnan(new_factor).any():
            print(f"WARNING: NaN detected in factor {d}, keeping old value")
            new_factors.append(factors[d])
        else:
            new_factors.append(new_factor)
    
    return new_factors

def run_tensor_agd_test(N: int = 30, M: int = 10, n: int = 3, 
                         alpha: float = 2.0, lr: float = 0.01, 
                         max_steps: int = 5000):
    """
    Run Tensor AGD test and return final Q_Y.
    
    Args:
        N: Dimension size (all dimensions equal for simplicity)
        M: Latent rank
        n: Tensor order (n=3 for 3rd order tensor)
        alpha: Connectivity = num_observations / degrees_of_freedom
        lr: Learning rate
        max_steps: Number of gradient descent steps
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
    
    # Generate observation mask
    # Degrees of freedom = n * N * M
    # Number of observations = alpha * dof
    dof = n * N * M
    num_obs = int(alpha * dof)
    total_entries = N ** n
    obs_ratio = num_obs / total_entries
    print(f"Observation ratio: {obs_ratio:.4f} ({num_obs}/{total_entries})")
    
    # Random mask
    mask = (torch.rand(tuple([N] * n), device=device) < obs_ratio).float()
    actual_obs = mask.sum().item()
    print(f"Actual observations: {int(actual_obs)}")
    
    # Observed data
    Y = T_teacher * mask
    
    # Initialize student factors randomly
    student_factors = [torch.randn(N, M, device=device) * 0.1 for _ in range(n)]
    
    # Initial Q_Y
    q_init = compute_tensor_cosine(student_factors, teacher_factors)
    print(f"Initial Q_Y (Cosine): {q_init:.6f}")
    
    # AGD iterations
    print(f"\nRunning {max_steps} AGD steps...")
    for step in range(max_steps):
        student_factors = tensor_agd_step(student_factors, Y, mask, lr)
        
        # Log every 500 steps
        if (step + 1) % 500 == 0:
            q_y = compute_tensor_cosine(student_factors, teacher_factors)
            T_student = cp_contract_factors(student_factors)
            mse = ((T_student - T_teacher) ** 2).mean().item()
            print(f"Step {step+1}: Q_Y={q_y:.6f}, MSE={mse:.6f}")
    
    # Final Q_Y
    q_final = compute_tensor_cosine(student_factors, teacher_factors)
    print(f"\n{'='*60}")
    print(f"Final Q_Y (Cosine): {q_final:.6f}")
    print(f"{'='*60}")
    
    return q_final

if __name__ == "__main__":
    print("="*60)
    print("Tensor AGD Baseline Test")
    print("="*60)
    
    # Test with moderate problem size
    q_y = run_tensor_agd_test(N=20, M=5, n=3, alpha=4.0, lr=0.0001, max_steps=5000)
    
    print("\n" + "="*60)
    if q_y > 0.9:
        print("✓ SUCCESS: AGD converged, Q_Y metric is correct!")
        print("  Problem is in AMP algorithm, not metrics.")
    elif q_y > 0.5:
        print("? PARTIAL: AGD partially converged")
        print("  May need more steps or tuning.")
    else:
        print("✗ FAILURE: AGD did not converge!")
        print("  Problem may be in metric calculation or tensor construction.")
    print("="*60)
