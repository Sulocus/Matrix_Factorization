"""
Test spreading with damping = 1.0 (fully accept new values) to see if convergence improves.
"""

import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

import torch

from smf.core.config import Config
from smf.modules.teachers import TeacherGenerator, SpreadingDataParallel
from smf.modules.graphs.supergraph import create_supergraph
from smf.modules.algorithms.bigamp_spreading_parallel import (
    generate_F_super, compute_Y_super, bigamp_spreading_parallel_step, forward_pass_parallel
)

def main():
    device = torch.device("cuda")
    N, M = 100, 25
    S = 1
    alpha_values = [2.0, 3.0, 4.0]
    seed = 42
    max_steps = 3000
    
    # Test different damping values
    damping_values = [0.5, 0.8, 1.0]
    noise_var = 1e-10
    
    print("=" * 70)
    print("TEST DIFFERENT DAMPING VALUES")
    print("=" * 70)
    
    # Create Teacher
    torch.manual_seed(seed)
    W_true = torch.randn(N, M, device=device)
    X_true = torch.randn(M, N, device=device)
    
    # Create SuperGraph
    supergraph = create_supergraph(N, N, M, alpha_values, S, seed, device)
    
    # Generate F
    F_super = generate_F_super(supergraph, M, 12345, device, 'rademacher')
    
    # Compute Y
    Y_super = compute_Y_super(W_true, X_true, supergraph, F_super)
    
    A = len(alpha_values)
    
    F = F_super[0].float()
    Y = Y_super[0]
    i_idx, j_idx = supergraph.get_sample_indices(0)
    alpha_mask = supergraph.alpha_mask
    
    for damping in damping_values:
        print(f"\n--- Damping = {damping} ---")
        
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
        Y_student = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)
        
        for a, alpha in enumerate(alpha_values):
            C_k = supergraph.get_active_edges(a)
            y_t = Y[:C_k]
            y_s = Y_student[a, :C_k]
            qy = ((y_t * y_s).sum() / (y_t.norm() * y_s.norm() + 1e-12)).item()
            
            status = "✅" if qy > 0.99 else "❌"
            print(f"  Alpha {alpha:.1f}: Q_Y = {qy:.4f} {status}")

if __name__ == "__main__":
    main()
