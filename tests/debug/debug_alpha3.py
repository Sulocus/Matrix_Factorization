"""
Deep dive into alpha=3.0: Why doesn't it converge to Q_Y=1?
Monitor step-by-step to see if it's stuck or oscillating.
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
    alpha = 3.0  # Problem case
    seed = 42
    max_steps = 5000
    damping = 0.5
    noise_var = 1e-10
    
    print("=" * 70)
    print(f"DEEP DIVE: Alpha = {alpha}")
    print("=" * 70)
    
    # Create Teacher
    torch.manual_seed(seed)
    W_true = torch.randn(N, M, device=device)
    X_true = torch.randn(M, N, device=device)
    
    # Create SuperGraph
    supergraph = create_supergraph(N, N, M, [alpha], S, seed, device)
    
    # Generate F
    F_super = generate_F_super(supergraph, M, 12345, device, 'rademacher')
    
    # Compute Y
    Y_super = compute_Y_super(W_true, X_true, supergraph, F_super)
    
    A = 1
    C_k = supergraph.get_active_edges(0)
    
    print(f"C_k (edges) = {C_k}")
    print(f"DOF (N*M + M*N) = {N*M + M*N}")
    print(f"Ratio DOF/C_k = {(N*M + M*N) / C_k:.2f}")
    print()
    
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
    
    print("Step-by-step monitoring:")
    print(f"{'Step':<8} {'Q_Y':<10} {'MSE':<12} {'|W|':<10} {'|s|':<10}")
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
        
        # Monitor every 500 steps
        if (step + 1) % 500 == 0 or step < 5:
            Y_student = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)
            
            y_t = Y[:C_k]
            y_s = Y_student[0, :C_k]
            
            qy = ((y_t * y_s).sum() / (y_t.norm() * y_s.norm() + 1e-12)).item()
            mse = ((y_t - y_s) ** 2).mean().item()
            w_norm = W_hat.norm().item()
            s_norm = prev_s.norm().item() if prev_s is not None else 0
            
            print(f"{step+1:<8} {qy:<10.6f} {mse:<12.6f} {w_norm:<10.2f} {s_norm:<10.4f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    Y_student = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)
    y_t = Y[:C_k]
    y_s = Y_student[0, :C_k]
    
    residual = y_t - y_s
    print(f"Final residual mean: {residual.mean().item():.6f}")
    print(f"Final residual std: {residual.std().item():.6f}")
    print(f"Final residual max: {residual.abs().max().item():.6f}")

if __name__ == "__main__":
    main()
