#!/usr/bin/env python3
"""Extended memory benchmark with more sizes and alphas."""

import torch
import gc
import time

def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def measure(N1, N2, M, S, B, alpha, device='cuda'):
    reset_memory()
    time.sleep(0.2)
    
    C = max(1, int(alpha * M * N1))
    SC = S * C
    
    try:
        # Allocate all tensors as in algorithm
        W_hat = torch.randn(S, B, N1, M, device=device)
        X_hat = torch.randn(S, B, M, N2, device=device)
        W_var = torch.ones(S, B, N1, M, device=device)
        X_var = torch.ones(S, B, M, N2, device=device)
        F_super = torch.randn(S, C, M, device=device)
        Y_super = torch.randn(S, C, device=device)
        i_offset = torch.randint(0, S * N1, (SC,), device=device, dtype=torch.long)
        j_offset = torch.randint(0, S * N2, (SC,), device=device, dtype=torch.long)
        
        W_flat = W_hat.reshape(B, S * N1, M)
        X_flat = X_hat.permute(1, 0, 3, 2).reshape(B, S * N2, M)
        W_var_flat = W_var.reshape(B, S * N1, M)
        X_var_flat = X_var.permute(1, 0, 3, 2).reshape(B, S * N2, M)
        
        W_sel = W_flat[:, i_offset, :]
        X_sel = X_flat[:, j_offset, :]
        W_var_sel = W_var_flat[:, i_offset, :]
        X_var_sel = X_var_flat[:, j_offset, :]
        
        alpha_scale = 1.0 / (M ** 0.5)
        F_flat = F_super.reshape(SC, M)
        Z_hat = alpha_scale * (F_flat.unsqueeze(0) * W_sel * X_sel).sum(dim=2)
        F_sq = F_flat.pow(2).unsqueeze(0)
        V = alpha_scale**2 * (F_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
        denom = torch.clamp(V + 1e-10, min=1e-6)
        s_values = (Y_super.reshape(1, SC) - Z_hat) / denom
        
        r_W = torch.zeros(B, S * N1, M, device=device)
        tau_W = torch.zeros(B, S * N1, M, device=device)
        r_X = torch.zeros(B, S * N2, M, device=device)
        tau_X = torch.zeros(B, S * N2, M, device=device)
        
        s_exp = s_values.unsqueeze(2)
        idx_W = i_offset.view(1, SC, 1).expand(B, SC, M)
        r_W_contrib = alpha_scale * F_flat.unsqueeze(0) * X_sel * s_exp
        r_W.scatter_add_(1, idx_W, r_W_contrib)
        
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        
        return peak
    except RuntimeError:
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Extended test cases
    tests = [
        # Small N, various alpha
        (300, 300, 50, 1, 5, 0.5),
        (300, 300, 50, 1, 5, 1.0),
        (300, 300, 50, 1, 5, 2.0),
        (300, 300, 50, 1, 5, 3.0),
        (300, 300, 50, 1, 5, 4.0),
        
        # Medium N, various B
        (500, 500, 100, 1, 1, 2.0),
        (500, 500, 100, 1, 3, 2.0),
        (500, 500, 100, 1, 5, 2.0),
        (500, 500, 100, 1, 8, 2.0),
        (500, 500, 100, 1, 10, 2.0),
        
        # Various alpha at medium size
        (500, 500, 100, 1, 5, 0.5),
        (500, 500, 100, 1, 5, 1.0),
        (500, 500, 100, 1, 5, 1.5),
        (500, 500, 100, 1, 5, 2.5),
        (500, 500, 100, 1, 5, 3.5),
        
        # Larger M
        (500, 500, 200, 1, 3, 2.0),
        (500, 500, 200, 1, 5, 2.0),
        
        # Larger N
        (800, 800, 150, 1, 3, 2.0),
        (800, 800, 150, 1, 5, 2.0),
        
        # Large scale
        (1000, 1000, 250, 1, 2, 3.0),
        (1000, 1000, 250, 1, 3, 3.0),
        (1000, 1000, 250, 1, 4, 3.0),
        (1000, 1000, 250, 1, 5, 3.0),
    ]
    
    results = []
    print(f"\n{'N1':>5} {'M':>4} {'B':>3} {'α':>4} | {'Actual':>7}")
    print("=" * 35)
    
    for N1, N2, M, S, B, alpha in tests:
        peak = measure(N1, N2, M, S, B, alpha)
        if peak:
            results.append((N1, N2, M, S, B, alpha, peak))
            print(f"{N1:>5} {M:>4} {B:>3} {alpha:>4.1f} | {peak:>6.2f}G")
        else:
            print(f"{N1:>5} {M:>4} {B:>3} {alpha:>4.1f} | OOM")
        time.sleep(0.3)
    
    # Output in format for verification
    print("\n\nData for verification:")
    print("results = [")
    for r in results:
        print(f"    {r},")
    print("]")
