"""
Monitor V (variance) and W_var to find why s explodes.
"""

import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

import torch

from smf.modules.graphs.supergraph import create_supergraph
from smf.modules.algorithms.bigamp_spreading_parallel import (
    generate_F_super, compute_Y_super, forward_pass_parallel,
    compute_variance_parallel
)
import math

def main():
    device = torch.device("cuda")
    N, M = 100, 25
    S = 1
    alpha = 3.0
    seed = 42
    max_steps = 1000
    damping = 0.5
    noise_var = 1e-10
    
    print("=" * 70)
    print(f"VARIANCE MONITORING: Alpha = {alpha}")
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
    
    F = F_super[0].float()
    Y = Y_super[0]
    i_idx, j_idx = supergraph.get_sample_indices(0)
    alpha_mask = supergraph.alpha_mask
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Initialize student
    torch.manual_seed(9999)
    W_hat = torch.randn(A, N, M, device=device) * 0.1
    X_hat = torch.randn(A, M, N, device=device) * 0.1
    W_var = torch.ones(A, N, M, device=device)
    X_var = torch.ones(A, M, N, device=device)
    
    print(f"{'Step':<8} {'Q_Y':<10} {'W_var_min':<12} {'W_var_max':<12} {'V_min':<12} {'V_max':<12}")
    print("-" * 70)
    
    prev_s = None
    for step in range(max_steps):
        # Forward pass
        Z_hat = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)
        
        # Compute variance
        V = compute_variance_parallel(W_hat, X_hat, W_var, X_var, F, i_idx, j_idx, alpha_mask)
        
        # Residual
        Y_broadcast = Y.unsqueeze(0)
        denominator = torch.clamp(V + noise_var, min=1e-6)
        s_values = (Y_broadcast - Z_hat) / denominator
        s_values = torch.clamp(s_values, min=-1e6, max=1e6)
        s_values = s_values * alpha_mask.float()
        
        # W update
        X_sel = X_hat[:, :, j_idx].transpose(1, 2)
        F_expanded = F.unsqueeze(0)
        s_expanded = s_values.unsqueeze(2)
        
        # r_W
        r_W_contrib = alpha_scale * F_expanded * X_sel * s_expanded
        r_W = torch.zeros(A, N, M, device=device)
        idx_W = i_idx.view(1, -1, 1).expand(A, -1, M)
        mask_W = alpha_mask.unsqueeze(2).float()
        r_W.scatter_add_(1, idx_W, r_W_contrib * mask_W)
        
        # tau_W
        inv_V = (1.0 / denominator).unsqueeze(2)
        F_sq_expanded = F_expanded.pow(2)
        tau_W_contrib = alpha_scale_sq * F_sq_expanded * X_sel.pow(2) * inv_V
        tau_W = torch.zeros(A, N, M, device=device)
        tau_W.scatter_add_(1, idx_W, tau_W_contrib * mask_W)
        tau_W = tau_W.clamp(min=1e-10)
        
        # Update
        W_var_new = 1.0 / (1.0 + tau_W)
        r_W = torch.clamp(r_W, min=-1e4, max=1e4)
        W_hat_new = W_hat + W_var_new * r_W
        
        # X update (similar)
        W_sel = W_hat[:, i_idx, :]
        r_X_contrib = alpha_scale * F_expanded * W_sel * s_expanded
        r_X_contrib_T = r_X_contrib.transpose(1, 2)
        r_X = torch.zeros(A, M, N, device=device)
        j_idx_expanded = j_idx.view(1, 1, -1).expand(A, M, -1)
        mask_X = alpha_mask.unsqueeze(1).float()
        r_X.scatter_add_(2, j_idx_expanded, r_X_contrib_T * mask_X)
        
        tau_X_contrib = alpha_scale_sq * F_sq_expanded * W_sel.pow(2) * inv_V
        tau_X_contrib_T = tau_X_contrib.transpose(1, 2)
        tau_X = torch.zeros(A, M, N, device=device)
        tau_X.scatter_add_(2, j_idx_expanded, tau_X_contrib_T * mask_X)
        tau_X = tau_X.clamp(min=1e-10)
        
        X_var_new = 1.0 / (1.0 + tau_X)
        r_X = torch.clamp(r_X, min=-1e4, max=1e4)
        X_hat_new = X_hat + X_var_new * r_X
        
        # Damping
        W_hat = damping * W_hat_new + (1 - damping) * W_hat
        X_hat = damping * X_hat_new + (1 - damping) * X_hat
        W_var = damping * W_var_new + (1 - damping) * W_var
        X_var = damping * X_var_new + (1 - damping) * X_var
        
        prev_s = s_values
        
        # Monitor
        if (step + 1) % 100 == 0 or step < 5:
            Y_student = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)
            y_t = Y[:C_k]
            y_s = Y_student[0, :C_k]
            qy = ((y_t * y_s).sum() / (y_t.norm() * y_s.norm() + 1e-12)).item()
            
            # Active edges only
            V_active = V[0, :C_k]
            
            print(f"{step+1:<8} {qy:<10.4f} {W_var.min().item():<12.6f} {W_var.max().item():<12.6f} "
                  f"{V_active.min().item():<12.6f} {V_active.max().item():<12.6f}")

if __name__ == "__main__":
    main()
