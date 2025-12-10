"""
Run legacy bigamp_spreading_parallel.py directly - minimal config.
"""

import sys
import os

# ONLY use legacy path
for p in list(sys.path):
    if 'Sparse-Matrix' in p and '_legacy' not in p:
        sys.path.remove(p)
sys.path.insert(0, '/home/sucia/Sparse-Matrix/_legacy/backup_pre_refactor_20251209')

import torch

def main():
    from smf.modules.graphs.supergraph import create_supergraph
    from smf.modules.algorithms.bigamp_spreading_parallel import (
        generate_F_super,
        compute_Y_super,
        bigamp_spreading_parallel_step,
    )
    from smf.modules.metrics.spreading import compute_qy_spreading_parallel
    
    device = torch.device("cuda")
    N, M = 100, 25
    S = 2
    alpha_values = [2.0, 2.5, 3.0, 3.5, 4.0]
    seed = 42
    max_steps = 5000
    damping = 0.5
    noise_var = 1e-10
    
    print("=" * 70)
    print("RUNNING LEGACY CODE WITH MANUAL TRAINING")
    print("=" * 70)
    print(f"N={N}, M={M}, max_steps={max_steps}")
    print(f"Alphas: {alpha_values}")
    print()
    
    # Create Teacher
    torch.manual_seed(seed)
    W_true = torch.randn(N, M, device=device)
    X_true = torch.randn(M, N, device=device)
    
    # Create SuperGraph
    supergraph = create_supergraph(N, N, M, alpha_values, S, seed, device)
    
    # Generate F (rademacher)
    F_super = generate_F_super(supergraph, M, 12345, device, 'rademacher')
    
    # Compute Y
    Y_super = compute_Y_super(W_true, X_true, supergraph, F_super)
    
    A = len(alpha_values)
    
    # Manual training for sample 0
    print("Training sample 0...")
    
    F = F_super[0].float()  # (C_max, M)
    Y = Y_super[0]  # (C_max,)
    i_idx, j_idx = supergraph.get_sample_indices(0)
    alpha_mask = supergraph.alpha_mask  # (A, C_max)
    
    # Initialize student
    torch.manual_seed(9999)
    W_hat = torch.randn(A, N, M, device=device) * 0.1
    X_hat = torch.randn(A, M, N, device=device) * 0.1
    W_var = torch.ones(A, N, M, device=device)
    X_var = torch.ones(A, M, N, device=device)
    
    prev_s = None
    for step in range(max_steps):
        W_hat, X_hat, W_var, X_var, prev_s = bigamp_spreading_parallel_step(
            W_hat=W_hat,
            X_hat=X_hat,
            W_var=W_var,
            X_var=X_var,
            Y_values=Y,
            F=F,
            i_idx=i_idx,
            j_idx=j_idx,
            alpha_mask=alpha_mask,
            damping=damping,
            noise_var=noise_var,
            prev_s=prev_s,
        )
        
        if (step + 1) % 1000 == 0:
            print(f"  Step {step + 1}/{max_steps}")
    
    # Compute Q_Y (manually)
    from smf.modules.algorithms.bigamp_spreading_parallel import forward_pass_parallel
    
    print("\n" + "=" * 70)
    print("LEGACY CODE RESULTS")
    print("=" * 70)
    
    for a, alpha in enumerate(alpha_values):
        C_k = supergraph.get_active_edges(a)
        if C_k == 0:
            print(f"Alpha {alpha:.1f}: No edges")
            continue
        
        # Compute Y_student
        Y_student = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)
        
        y_t = Y[:C_k]
        y_s = Y_student[a, :C_k]
        
        dot = (y_t * y_s).sum()
        norm_t = y_t.norm()
        norm_s = y_s.norm()
        qy = (dot / (norm_t * norm_s + 1e-12)).item()
        
        # Q_W
        w_s = W_hat[a]  # (N, M)
        w_t = W_true  # (N, M)
        # Gram matrix similarity
        gram_s = w_s @ w_s.T  # (N, N)
        gram_t = w_t @ w_t.T  # (N, N)
        qw = (torch.trace(gram_s @ gram_t) / (gram_s.norm() * gram_t.norm() + 1e-12)).item()
        
        print(f"Alpha {alpha:.1f}: Q_Y = {qy:.4f}, Q_W (gram) = {qw:.4f}")

if __name__ == "__main__":
    main()
