"""
Test with inline variance monitoring to confirm clamp is applied.
"""

import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

import torch

from smf.modules.graphs.supergraph import create_supergraph
from smf.modules.algorithms.bigamp_spreading_parallel import (
    generate_F_super, compute_Y_super, forward_pass_parallel, bigamp_spreading_parallel_step
)

def main():
    device = torch.device("cuda")
    N, M = 100, 25
    alpha = 3.0
    seed = 42
    max_steps = 1000
    damping = 0.5
    noise_var = 1e-10
    
    print("=" * 70)
    print("VERIFY VARIANCE CLAMP IS WORKING")
    print("=" * 70)
    
    # Create Teacher
    torch.manual_seed(seed)
    W_true = torch.randn(N, M, device=device)
    X_true = torch.randn(M, N, device=device)
    
    # Create SuperGraph for just this alpha
    supergraph = create_supergraph(N, N, M, [alpha], 1, seed, device)
    
    # Generate F
    F_super = generate_F_super(supergraph, M, 12345, device, 'rademacher')
    
    # Compute Y
    Y_super = compute_Y_super(W_true, X_true, supergraph, F_super)
    
    A = 1
    
    F = F_super[0].float()
    Y = Y_super[0]
    i_idx, j_idx = supergraph.get_sample_indices(0)
    alpha_mask = supergraph.alpha_mask
    
    # Initialize student
    torch.manual_seed(9999)
    W_hat = torch.randn(A, N, M, device=device) * 0.1
    X_hat = torch.randn(A, M, N, device=device) * 0.1
    W_var = torch.ones(A, N, M, device=device)
    X_var = torch.ones(A, M, N, device=device)
    
    prev_s = None
    
    print(f"{'Step':<8} {'Q_Y':<10} {'W_var_min':<12} {'W_var_max':<12}")
    print("-" * 50)
    
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
        
        if (step + 1) % 100 == 0 or step < 5:
            Y_student = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)
            C_k = supergraph.get_active_edges(0)
            y_t = Y[:C_k]
            y_s = Y_student[0, :C_k]
            qy = ((y_t * y_s).sum() / (y_t.norm() * y_s.norm() + 1e-12)).item()
            
            print(f"{step+1:<8} {qy:<10.4f} {W_var.min().item():<12.8f} {W_var.max().item():<12.8f}")
    
    print("\nExpected: W_var_min should be >= 1e-8 if clamp is working")

if __name__ == "__main__":
    main()
