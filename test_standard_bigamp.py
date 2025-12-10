"""
Test standard BiG-AMP (not spreading) to confirm Q_Y = 1 after phase transition.
This gives us a known-correct baseline.
"""

import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix/_legacy/backup_pre_refactor_20251209')

import torch

def main():
    from smf.modules.algorithms.bigamp import _bigamp_step
    
    device = torch.device("cuda")
    N, M = 100, 25
    S = 2
    alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    seed = 42
    max_steps = 3000
    damping = 0.5
    noise_var = 1e-10
    alpha_scale = 1.0 / (M ** 0.5)
    
    print("=" * 70)
    print("STANDARD BIGAMP (DENSE MATRIX) BASELINE TEST")
    print("=" * 70)
    print(f"N={N}, M={M}, max_steps={max_steps}")
    print(f"Alphas: {alpha_values}")
    print()
    
    # Create Teacher
    torch.manual_seed(seed)
    W_true = torch.randn(N, M, device=device)
    X_true = torch.randn(M, N, device=device)
    Y_true = alpha_scale * W_true @ X_true  # (N, N)
    
    print("Training for each alpha...")
    
    for alpha in alpha_values:
        # Create mask
        torch.manual_seed(seed + int(alpha * 100))
        mask = (torch.rand(N, N, device=device) < alpha).float()
        
        # Initialize student
        torch.manual_seed(9999)
        w_hat = torch.randn(S, N, M, device=device) / (M ** 0.5)
        x_hat = torch.randn(S, M, N, device=device) / (M ** 0.5)
        w_var = torch.ones(S, N, M, device=device) / M
        x_var = torch.ones(S, M, N, device=device) / M
        
        A = mask.unsqueeze(0)  # (1, N, N)
        Y = Y_true.unsqueeze(0)  # (1, N, N)
        
        # Training
        for step in range(max_steps):
            w_hat, x_hat, w_var, x_var = _bigamp_step(
                w_hat, x_hat, w_var, x_var,
                Y, A, alpha_scale, damping, noise_var, M
            )
        
        # Compute Q_Y (on observed)
        Y_student = alpha_scale * w_hat @ x_hat  # (S, N, N)
        Y_teacher = Y_true.unsqueeze(0)  # (1, N, N)
        
        # Observed Q_Y
        mask_exp = mask.unsqueeze(0)  # (1, N, N)
        y_t_obs = Y_teacher[0][mask > 0]
        y_s_obs = Y_student.mean(0)[mask > 0]  # Average over samples
        
        dot = (y_t_obs * y_s_obs).sum()
        norm_t = y_t_obs.norm()
        norm_s = y_s_obs.norm()
        qy_obs = (dot / (norm_t * norm_s + 1e-12)).item()
        
        # Q_W (Gram)
        w_s = w_hat.mean(0)  # (N, M)
        gram_s = w_s @ w_s.T
        gram_t = W_true @ W_true.T
        qw = (torch.trace(gram_s @ gram_t) / (gram_s.norm() * gram_t.norm() + 1e-12)).item()
        
        status = "✅" if qy_obs > 0.99 else "❌"
        print(f"Alpha {alpha:.1f}: Q_Y_obs = {qy_obs:.4f} {status}, Q_W = {qw:.4f}")
    
    print("\n" + "=" * 70)
    print("EXPECTED: Q_Y_obs should be 1.0 for all alphas where alpha > 0")
    print("=" * 70)

if __name__ == "__main__":
    main()
