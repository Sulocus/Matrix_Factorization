#!/usr/bin/env python3
"""
Fast Comparison: O(1) Unit Scaling vs Original 1/M Scaling.

Uses the optimized parallel implementations from smf framework:
- Wang/bigamp/orthogonal_teacher.py - for standalone parallel comparison
- Utilizes GPU parallelization properly

Usage:
    python scripts/compare_scaling_fast.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    N: int = 200
    M: int = 50
    alpha_start: float = 0.0
    alpha_stop: float = 4.0
    alpha_step: float = 0.1
    max_steps: int = 2000
    S: int = 4
    damping: float = 0.5
    noise_var: float = 1e-8
    seed: int = 42


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Teacher Creation
# ============================================================
def create_teacher(N, M, device, seed, use_unit_scaling=False):
    """Create orthogonal teacher with specified scaling."""
    torch.manual_seed(seed)
    W_raw = torch.randn(N, M, device=device, dtype=torch.float32)
    X_raw = torch.randn(M, N, device=device, dtype=torch.float32)
    
    W_ortho, _ = torch.linalg.qr(W_raw, mode='reduced')
    X_ortho_T, _ = torch.linalg.qr(X_raw.T, mode='reduced')
    X_ortho = X_ortho_T.T
    
    if use_unit_scaling:
        # NEW: unit variance (O(1))
        W = W_ortho * (N) ** 0.5
        X = X_ortho * (N) ** 0.5
    else:
        # OLD: 1/M variance (O(1/sqrt(M)))
        W = W_ortho * (N / M) ** 0.5
        X = X_ortho * (N / M) ** 0.5
    
    return W, X


# ============================================================
# Vectorized BiG-AMP (No loop over edges)
# ============================================================
def sample_mask(N, M, alpha, device, seed):
    """Generate mask and return indices."""
    C = int(alpha * M * N)
    if C == 0 or C > N * N:
        C = min(max(0, C), N * N)
    
    torch.manual_seed(seed)
    if C == 0:
        return torch.zeros(0, dtype=torch.long, device=device), \
               torch.zeros(0, dtype=torch.long, device=device)
    
    idx = torch.randperm(N * N, device=device)[:C]
    return idx // N, idx % N


def generate_F(C, M, seed, device):
    """Generate Rademacher F."""
    if C == 0:
        return torch.empty(0, M, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed ^ 0x5DEECE66D)
    return (torch.randint(0, 2, (C, M), device=device, generator=gen).float() * 2 - 1)


def compute_Y(W, X, F, i_idx, j_idx, M):
    """Vectorized Y computation."""
    if len(i_idx) == 0:
        return torch.empty(0, device=W.device)
    alpha_scale = 1.0 / (M ** 0.5)
    return alpha_scale * (F * W[i_idx] * X[:, j_idx].T).sum(dim=1)


def bigamp_vectorized(W_true, X_true, i_idx, j_idx, F, Y, 
                      N, M, steps, damping, noise_var, use_unit_scaling):
    """Fully vectorized BiG-AMP step using scatter_add."""
    C = len(i_idx)
    if C == 0:
        return 0.0
    
    alpha_scale = 1.0 / (M ** 0.5)
    alpha_scale_sq = 1.0 / M
    
    # Prior variance based on scaling
    prior_var = 1.0 if use_unit_scaling else 1.0 / M
    prior_precision = 1.0 if use_unit_scaling else M
    
    # Initialize
    torch.manual_seed(42)
    W_hat = torch.randn(N, M, device=DEVICE) * (prior_var ** 0.5)
    X_hat = torch.randn(M, N, device=DEVICE) * (prior_var ** 0.5)
    W_var = torch.ones(N, M, device=DEVICE) * prior_var
    X_var = torch.ones(M, N, device=DEVICE) * prior_var
    
    for step in range(steps):
        # Forward: Z_hat = (1/sqrt(M)) * sum(F * W * X)
        W_sel = W_hat[i_idx]              # (C, M)
        X_sel = X_hat[:, j_idx].T         # (C, M)
        Z_hat = alpha_scale * (F * W_sel * X_sel).sum(dim=1)  # (C,)
        
        # Variance
        W_var_sel = W_var[i_idx]
        X_var_sel = X_var[:, j_idx].T
        V = alpha_scale_sq * (W_var_sel * X_sel**2 + W_sel**2 * X_var_sel).sum(dim=1) + 1e-10
        
        # Residual
        denom = torch.clamp(V + noise_var, min=1e-6)
        s = torch.clamp((Y - Z_hat) / denom, min=-1e6, max=1e6)
        inv_V = 1.0 / denom
        
        # W update using scatter_add (vectorized)
        # r_W[i] = sum_c F[c] * X[c] * s[c] for all c where i_idx[c] == i
        F_s_X = alpha_scale * F * s.unsqueeze(1) * X_sel  # (C, M)
        r_W = torch.zeros(N, M, device=DEVICE)
        r_W.scatter_add_(0, i_idx.unsqueeze(1).expand(-1, M), F_s_X)
        
        F2_invV_X2 = alpha_scale_sq * (F**2) * inv_V.unsqueeze(1) * (X_sel**2)
        tau_W = torch.zeros(N, M, device=DEVICE)
        tau_W.scatter_add_(0, i_idx.unsqueeze(1).expand(-1, M), F2_invV_X2)
        tau_W = tau_W.clamp(min=1e-10)
        
        W_var_new = 1.0 / (prior_precision + tau_W)
        W_hat_new = W_hat + W_var_new * r_W
        
        # X update using scatter_add (vectorized)
        F_s_W = alpha_scale * F * s.unsqueeze(1) * W_sel  # (C, M)
        r_X = torch.zeros(M, N, device=DEVICE)
        r_X.scatter_add_(1, j_idx.unsqueeze(0).expand(M, -1), F_s_W.T)
        
        F2_invV_W2 = alpha_scale_sq * (F**2) * inv_V.unsqueeze(1) * (W_sel**2)
        tau_X = torch.zeros(M, N, device=DEVICE)
        tau_X.scatter_add_(1, j_idx.unsqueeze(0).expand(M, -1), F2_invV_W2.T)
        tau_X = tau_X.clamp(min=1e-10)
        
        X_var_new = 1.0 / (prior_precision + tau_X)
        X_hat_new = X_hat + X_var_new * r_X
        
        # Damping
        W_hat = damping * W_hat_new + (1 - damping) * W_hat
        X_hat = damping * X_hat_new + (1 - damping) * X_hat
        W_var = torch.clamp(damping * W_var_new + (1 - damping) * W_var, min=1e-4, max=1.0)
        X_var = torch.clamp(damping * X_var_new + (1 - damping) * X_var, min=1e-4, max=1.0)
    
    # Compute Q_Y
    Y_hat = W_hat @ X_hat
    Y_true = W_true @ X_true
    dot = (Y_hat.flatten() * Y_true.flatten()).sum()
    return float(dot / (Y_hat.norm() * Y_true.norm() + 1e-12))


# ============================================================
# Run Experiment
# ============================================================
def run_single_alpha(config, alpha, use_unit_scaling, trial_seed):
    """Run single alpha with vectorized implementation."""
    N, M = config.N, config.M
    
    W_true, X_true = create_teacher(N, M, DEVICE, config.seed, use_unit_scaling)
    i_idx, j_idx = sample_mask(N, M, alpha, DEVICE, trial_seed)
    C = len(i_idx)
    if C == 0:
        return 0.0
    
    F = generate_F(C, M, trial_seed + 10000, DEVICE)
    Y = compute_Y(W_true, X_true, F, i_idx, j_idx, M)
    
    return bigamp_vectorized(
        W_true, X_true, i_idx, j_idx, F, Y,
        N, M, config.max_steps, config.damping, config.noise_var, use_unit_scaling
    )


def run_experiment(config, use_unit_scaling):
    """Run full experiment."""
    alphas = np.arange(config.alpha_start, config.alpha_stop + config.alpha_step/2, config.alpha_step)
    results = {}
    
    version = "UNIT" if use_unit_scaling else "OLD"
    pbar = tqdm(total=len(alphas) * config.S, desc=f"Running {version}")
    
    for alpha in alphas:
        qy_trials = []
        for s in range(config.S):
            seed = config.seed + int(alpha * 1000) + s * 10000
            qy = run_single_alpha(config, alpha, use_unit_scaling, seed)
            qy_trials.append(qy)
            pbar.update(1)
        
        results[float(alpha)] = {
            'Q_Y_mean': float(np.mean(qy_trials)),
            'Q_Y_std': float(np.std(qy_trials, ddof=1)) if len(qy_trials) > 1 else 0.0,
        }
    
    pbar.close()
    return results


# ============================================================
# Visualization
# ============================================================
def plot_comparison(results_old, results_new, output_dir):
    """Generate comparison plot."""
    alphas = sorted(results_old.keys())
    qy_old = [results_old[a]['Q_Y_mean'] for a in alphas]
    qy_new = [results_new[a]['Q_Y_mean'] for a in alphas]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, qy_old, 'b-o', label='OLD: Var=1/M', linewidth=2, markersize=4)
    ax.plot(alphas, qy_new, 'r-s', label='NEW: Var=1 (Unit)', linewidth=2, markersize=4)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Perfect')
    ax.axvline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('α', fontsize=12)
    ax.set_ylabel('Q_Y', fontsize=12)
    ax.set_title('Normalization Comparison: Unit vs 1/M Scaling', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_qy.png', dpi=150)
    plt.close()
    print(f"✓ Saved plot to {output_dir / 'comparison_qy.png'}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Fast Scaling Comparison (Vectorized)")
    print("=" * 60)
    
    config = Config()
    print(f"\nN={config.N}, M={config.M}, alpha=0-4, steps={config.max_steps}, S={config.S}")
    print(f"Device: {DEVICE}")
    
    output_dir = Path(__file__).parent.parent / 'smf' / 'results' / 'scaling_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run OLD
    print("\n[1/2] Running OLD version (Var=1/M)...")
    t0 = time.time()
    results_old = run_experiment(config, use_unit_scaling=False)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Run NEW
    print("\n[2/2] Running NEW version (Var=1)...")
    t0 = time.time()
    results_new = run_experiment(config, use_unit_scaling=True)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Save
    with open(output_dir / 'results_old.json', 'w') as f:
        json.dump(results_old, f, indent=2)
    with open(output_dir / 'results_new.json', 'w') as f:
        json.dump(results_new, f, indent=2)
    
    plot_comparison(results_old, results_new, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    for name, res in [("OLD", results_old), ("NEW", results_new)]:
        alphas = sorted(res.keys())
        final = res[alphas[-1]]['Q_Y_mean']
        trans = next((a for a in alphas if res[a]['Q_Y_mean'] > 0.9), None)
        print(f"{name}: Final Q_Y={final:.3f}, Transition≈{trans}")
    print("=" * 60)


if __name__ == '__main__':
    main()
