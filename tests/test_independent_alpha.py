"""
Test training each alpha INDEPENDENTLY (not parallel) to rule out cross-alpha interference.
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
    S = 1
    alpha_values = [2.0, 3.0, 4.0]
    seed = 42
    max_steps = 3000
    damping = 0.5
    noise_var = 1e-10
    
    print("=" * 70)
    print("TEST INDEPENDENT ALPHA TRAINING (ONE AT A TIME)")
    print("=" * 70)
    
    # Create Teacher
    torch.manual_seed(seed)
    W_true = torch.randn(N, M, device=device)
    X_true = torch.randn(M, N, device=device)
    
    for alpha in alpha_values:
        # Create SuperGraph for just this alpha
        supergraph = create_supergraph(N, N, M, [alpha], S, seed, device)
        
        # Generate F
        F_super = generate_F_super(supergraph, M, 12345, device, 'rademacher')
        
        # Compute Y
        Y_super = compute_Y_super(W_true, X_true, supergraph, F_super)
        
        A = 1  # Only 1 alpha
        
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
        
        # Compute Q_Y
        C_k = supergraph.get_active_edges(0)
        Y_student = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)
        
        y_t = Y[:C_k]
        y_s = Y_student[0, :C_k]
        qy = ((y_t * y_s).sum() / (y_t.norm() * y_s.norm() + 1e-12)).item()
        
        status = "✅" if qy > 0.99 else "❌"
        print(f"Alpha {alpha:.1f}: Q_Y = {qy:.4f} {status}")
    
    print("\n" + "=" * 70)
    print("EXPECTED: All Q_Y should be 1.0 when trained independently")
    print("=" * 70)

if __name__ == "__main__":
    main()
