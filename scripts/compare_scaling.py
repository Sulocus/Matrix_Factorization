#!/usr/bin/env python3
"""
Comparison Experiment: O(1) Unit Scaling vs Original 1/M Scaling.

This script compares the Q_Y phase transition behavior between:
1. OLD: orthogonal teacher (1/M variance) + bigamp_spreading_parallel (M+tau)
2. NEW: orthogonal_unit teacher (unit variance) + bigamp_spreading_parallel_unit (1+tau)

Based on prompt.md analysis:
- Old version: signal Y ~ O(1/√M), too weak to trigger phase transition
- New version: signal Y ~ O(1), matches paper standard

Usage:
    python scripts/compare_scaling.py
    
Parameters:
    N = 200, M = 50
    alpha = 0.0 to 4.0, step = 0.1
    max_steps = 2000
    S = 4 samples per alpha

Output:
    smf/results/scaling_comparison/
    - comparison_qy.png: Q_Y curves overlay
    - results_old.json: raw data for old version
    - results_new.json: raw data for new version
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================
@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    N: int = 200
    M: int = 50
    alpha_start: float = 0.0
    alpha_stop: float = 4.0
    alpha_step: float = 0.1
    max_steps: int = 2000
    S: int = 4  # samples per alpha
    damping: float = 0.5
    noise_var: float = 1e-8
    seed: int = 42
    f_distribution: str = 'rademacher'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Teacher Functions
# ============================================================
def create_orthogonal_teacher_old(N, M, device, seed=42):
    """
    Original orthogonal teacher with 1/M variance.
    W_true = W_ortho * sqrt(N/M)
    """
    torch.manual_seed(seed)
    W_raw = torch.randn(N, M, device=device, dtype=torch.float32)
    X_raw = torch.randn(M, N, device=device, dtype=torch.float32)
    
    W_ortho, _ = torch.linalg.qr(W_raw, mode='reduced')
    X_ortho_T, _ = torch.linalg.qr(X_raw.T, mode='reduced')
    X_ortho = X_ortho_T.T
    
    # OLD scaling: variance = 1/M
    W_true = W_ortho * (N / M) ** 0.5
    X_true = X_ortho * (N / M) ** 0.5
    
    return W_true, X_true


def create_orthogonal_teacher_new(N, M, device, seed=42):
    """
    New orthogonal teacher with unit variance.
    W_true = W_ortho * sqrt(N)
    """
    torch.manual_seed(seed)
    W_raw = torch.randn(N, M, device=device, dtype=torch.float32)
    X_raw = torch.randn(M, N, device=device, dtype=torch.float32)
    
    W_ortho, _ = torch.linalg.qr(W_raw, mode='reduced')
    X_ortho_T, _ = torch.linalg.qr(X_raw.T, mode='reduced')
    X_ortho = X_ortho_T.T
    
    # NEW scaling: variance = 1 (unit)
    W_true = W_ortho * (N) ** 0.5
    X_true = X_ortho * (N) ** 0.5
    
    return W_true, X_true


# ============================================================
# Mask Generation
# ============================================================
def sample_mask_random(N1, N2, M, alpha, device, seed):
    """Generate random observation mask."""
    C = int(alpha * M * N1)
    if C == 0:
        return torch.zeros(0, dtype=torch.long, device=device), \
               torch.zeros(0, dtype=torch.long, device=device)
    
    total = N1 * N2
    if C > total:
        C = total
    
    torch.manual_seed(seed)
    idx = torch.randperm(total, device=device)[:C]
    i_idx = idx // N2
    j_idx = idx % N2
    
    return i_idx, j_idx


# ============================================================
# F Generation
# ============================================================
def generate_F_rademacher(C, M, seed, device):
    """Generate Rademacher F ~ {-1, +1}."""
    if C == 0:
        return torch.empty(0, M, device=device, dtype=torch.float32)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed ^ 0x5DEECE66D)
    bits = torch.randint(0, 2, (C, M), device=device, dtype=torch.float32, generator=gen)
    return bits * 2 - 1


# ============================================================
# Y Computation
# ============================================================
def compute_Y(W, X, F, i_idx, j_idx):
    """Compute Y = (1/√M) Σ F * W * X at observed positions."""
    if len(i_idx) == 0:
        return torch.empty(0, device=W.device, dtype=torch.float32)
    
    M = W.shape[1]
    alpha_scale = 1.0 / (M ** 0.5)
    
    W_sel = W[i_idx]  # (C, M)
    X_sel = X[:, j_idx].T  # (C, M)
    
    Y = alpha_scale * (F * W_sel * X_sel).sum(dim=1)
    return Y


# ============================================================
# BiG-AMP Training Functions
# ============================================================
def bigamp_step_old(W_hat, X_hat, W_var, X_var, Y, F, i_idx, j_idx, 
                    M, N1, N2, damping, noise_var):
    """
    Old BiG-AMP step with M in denominator (for 1/M variance prior).
    Variance update: 1/(M + tau)
    """
    C = len(i_idx)
    if C == 0:
        return W_hat, X_hat, W_var, X_var
    
    alpha_scale = 1.0 / (M ** 0.5)
    alpha_scale_sq = 1.0 / M
    
    # Forward pass
    W_sel = W_hat[i_idx]
    X_sel = X_hat[:, j_idx].T
    Z_hat = alpha_scale * (F * W_sel * X_sel).sum(dim=1)
    
    # Variance
    W_var_sel = W_var[i_idx]
    X_var_sel = X_var[:, j_idx].T
    V = alpha_scale_sq * (W_var_sel * X_sel**2 + W_sel**2 * X_var_sel).sum(dim=1)
    V = V + 1e-10
    
    # Residual
    denom = torch.clamp(V + noise_var, min=1e-6)
    s = (Y - Z_hat) / denom
    s = torch.clamp(s, min=-1e6, max=1e6)
    
    # W update
    inv_V = 1.0 / denom
    r_W = torch.zeros_like(W_hat)
    tau_W = torch.zeros_like(W_hat)
    for c in range(C):
        i, j = i_idx[c].item(), j_idx[c].item()
        r_W[i] += alpha_scale * F[c] * X_hat[:, j] * s[c]
        tau_W[i] += alpha_scale_sq * (F[c]**2) * (X_hat[:, j]**2) * inv_V[c]
    
    tau_W = tau_W.clamp(min=1e-10)
    W_var_new = 1.0 / (M + tau_W)  # OLD: M in denominator
    W_hat_new = W_hat + W_var_new * r_W
    
    # X update
    r_X = torch.zeros_like(X_hat)
    tau_X = torch.zeros_like(X_hat)
    for c in range(C):
        i, j = i_idx[c].item(), j_idx[c].item()
        r_X[:, j] += alpha_scale * F[c] * W_hat[i] * s[c]
        tau_X[:, j] += alpha_scale_sq * (F[c]**2) * (W_hat[i]**2) * inv_V[c]
    
    tau_X = tau_X.clamp(min=1e-10)
    X_var_new = 1.0 / (M + tau_X)  # OLD: M in denominator
    X_hat_new = X_hat + X_var_new * r_X
    
    # Damping
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var, min=1e-4, max=1.0)
    
    return W_hat_out, X_hat_out, W_var_out, X_var_out


def bigamp_step_new(W_hat, X_hat, W_var, X_var, Y, F, i_idx, j_idx,
                    M, N1, N2, damping, noise_var):
    """
    New BiG-AMP step with 1 in denominator (for unit variance prior).
    Variance update: 1/(1 + tau)
    """
    C = len(i_idx)
    if C == 0:
        return W_hat, X_hat, W_var, X_var
    
    alpha_scale = 1.0 / (M ** 0.5)
    alpha_scale_sq = 1.0 / M
    
    # Forward pass
    W_sel = W_hat[i_idx]
    X_sel = X_hat[:, j_idx].T
    Z_hat = alpha_scale * (F * W_sel * X_sel).sum(dim=1)
    
    # Variance
    W_var_sel = W_var[i_idx]
    X_var_sel = X_var[:, j_idx].T
    V = alpha_scale_sq * (W_var_sel * X_sel**2 + W_sel**2 * X_var_sel).sum(dim=1)
    V = V + 1e-10
    
    # Residual
    denom = torch.clamp(V + noise_var, min=1e-6)
    s = (Y - Z_hat) / denom
    s = torch.clamp(s, min=-1e6, max=1e6)
    
    # W update
    inv_V = 1.0 / denom
    r_W = torch.zeros_like(W_hat)
    tau_W = torch.zeros_like(W_hat)
    for c in range(C):
        i, j = i_idx[c].item(), j_idx[c].item()
        r_W[i] += alpha_scale * F[c] * X_hat[:, j] * s[c]
        tau_W[i] += alpha_scale_sq * (F[c]**2) * (X_hat[:, j]**2) * inv_V[c]
    
    tau_W = tau_W.clamp(min=1e-10)
    W_var_new = 1.0 / (1.0 + tau_W)  # NEW: 1 in denominator
    W_hat_new = W_hat + W_var_new * r_W
    
    # X update
    r_X = torch.zeros_like(X_hat)
    tau_X = torch.zeros_like(X_hat)
    for c in range(C):
        i, j = i_idx[c].item(), j_idx[c].item()
        r_X[:, j] += alpha_scale * F[c] * W_hat[i] * s[c]
        tau_X[:, j] += alpha_scale_sq * (F[c]**2) * (W_hat[i]**2) * inv_V[c]
    
    tau_X = tau_X.clamp(min=1e-10)
    X_var_new = 1.0 / (1.0 + tau_X)  # NEW: 1 in denominator
    X_hat_new = X_hat + X_var_new * r_X
    
    # Damping
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var, min=1e-4, max=1.0)
    
    return W_hat_out, X_hat_out, W_var_out, X_var_out


# ============================================================
# Q_Y Computation
# ============================================================
def compute_qy(W_hat, X_hat, W_true, X_true):
    """Compute Q_Y = <Y_hat, Y_true> / (||Y_hat|| ||Y_true||)."""
    Y_hat = W_hat @ X_hat
    Y_true = W_true @ X_true
    
    Y_hat_flat = Y_hat.flatten()
    Y_true_flat = Y_true.flatten()
    
    dot = (Y_hat_flat * Y_true_flat).sum()
    norm_hat = Y_hat_flat.norm()
    norm_true = Y_true_flat.norm()
    
    return float(dot / (norm_hat * norm_true + 1e-12))


# ============================================================
# Single Run
# ============================================================
def run_single_alpha(config: ExperimentConfig, alpha: float, 
                     version: str, trial_seed: int) -> float:
    """Run a single (alpha, version) configuration and return Q_Y."""
    N, M = config.N, config.M
    
    # Create teacher based on version
    if version == 'old':
        W_true, X_true = create_orthogonal_teacher_old(N, M, DEVICE, config.seed)
        step_fn = bigamp_step_old
        init_var = 1.0 / M  # Prior variance = 1/M
    else:
        W_true, X_true = create_orthogonal_teacher_new(N, M, DEVICE, config.seed)
        step_fn = bigamp_step_new
        init_var = 1.0  # Prior variance = 1
    
    # Create mask
    i_idx, j_idx = sample_mask_random(N, N, M, alpha, DEVICE, trial_seed)
    C = len(i_idx)
    
    if C == 0:
        return 0.0  # No observations
    
    # Generate F
    F = generate_F_rademacher(C, M, trial_seed + 10000, DEVICE)
    
    # Compute Y
    Y = compute_Y(W_true, X_true, F, i_idx, j_idx)
    
    # Initialize students
    torch.manual_seed(trial_seed + 20000)
    scale = (init_var) ** 0.5
    W_hat = torch.randn(N, M, device=DEVICE) * scale
    X_hat = torch.randn(M, N, device=DEVICE) * scale
    W_var = torch.ones(N, M, device=DEVICE) * init_var
    X_var = torch.ones(M, N, device=DEVICE) * init_var
    
    # Training loop
    for step in range(config.max_steps):
        W_hat, X_hat, W_var, X_var = step_fn(
            W_hat, X_hat, W_var, X_var,
            Y, F, i_idx, j_idx,
            M, N, N, config.damping, config.noise_var
        )
    
    # Compute Q_Y
    qy = compute_qy(W_hat, X_hat, W_true, X_true)
    return qy


# ============================================================
# Main Experiment
# ============================================================
def run_experiment(config: ExperimentConfig, version: str) -> Dict:
    """Run full experiment for one version."""
    alpha_values = np.arange(config.alpha_start, config.alpha_stop + config.alpha_step/2, config.alpha_step)
    
    results = {}
    total_runs = len(alpha_values) * config.S
    
    pbar = tqdm(total=total_runs, desc=f"Running {version}")
    
    for alpha in alpha_values:
        qy_trials = []
        for s in range(config.S):
            trial_seed = config.seed + int(alpha * 1000) + s * 10000
            qy = run_single_alpha(config, alpha, version, trial_seed)
            qy_trials.append(qy)
            pbar.update(1)
        
        results[float(alpha)] = {
            'Q_Y_mean': float(np.mean(qy_trials)),
            'Q_Y_std': float(np.std(qy_trials, ddof=1)) if len(qy_trials) > 1 else 0.0,
            'Q_Y_trials': qy_trials,
        }
    
    pbar.close()
    return results


# ============================================================
# Visualization
# ============================================================
def plot_comparison(results_old: Dict, results_new: Dict, output_dir: Path):
    """Generate comparison plots."""
    alphas_old = sorted(results_old.keys())
    alphas_new = sorted(results_new.keys())
    
    qy_old = [results_old[a]['Q_Y_mean'] for a in alphas_old]
    qy_new = [results_new[a]['Q_Y_mean'] for a in alphas_new]
    
    qy_old_std = [results_old[a]['Q_Y_std'] for a in alphas_old]
    qy_new_std = [results_new[a]['Q_Y_std'] for a in alphas_new]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot with error bands
    ax.plot(alphas_old, qy_old, 'b-o', label='OLD: Var=1/M, 1/(M+τ)', linewidth=2, markersize=4)
    ax.fill_between(alphas_old, 
                    [q - s for q, s in zip(qy_old, qy_old_std)],
                    [q + s for q, s in zip(qy_old, qy_old_std)],
                    alpha=0.2, color='blue')
    
    ax.plot(alphas_new, qy_new, 'r-s', label='NEW: Var=1, 1/(1+τ)', linewidth=2, markersize=4)
    ax.fill_between(alphas_new,
                    [q - s for q, s in zip(qy_new, qy_new_std)],
                    [q + s for q, s in zip(qy_new, qy_new_std)],
                    alpha=0.2, color='red')
    
    # Reference line
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Recovery')
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='α = 1')
    
    ax.set_xlabel('α (observation density)', fontsize=12)
    ax.set_ylabel('Q_Y (overlap)', fontsize=12)
    ax.set_title('Normalization Correction Comparison\nO(1) Scaling vs O(1/√M) Scaling', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_qy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Saved comparison plot to {output_dir / 'comparison_qy.png'}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Scaling Comparison Experiment")
    print("=" * 60)
    
    config = ExperimentConfig()
    
    print(f"\nConfiguration:")
    print(f"  N = {config.N}, M = {config.M}")
    print(f"  Alpha: {config.alpha_start} to {config.alpha_stop}, step {config.alpha_step}")
    print(f"  Steps: {config.max_steps}")
    print(f"  Samples per alpha: {config.S}")
    print(f"  Device: {DEVICE}")
    
    # Output directory
    output_dir = Path(__file__).parent.parent / 'smf' / 'results' / 'scaling_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run OLD version
    print(f"\n[1/2] Running OLD version (Var=1/M, 1/(M+τ))...")
    start_old = time.time()
    results_old = run_experiment(config, 'old')
    time_old = time.time() - start_old
    print(f"  Completed in {time_old:.1f}s")
    
    # Run NEW version
    print(f"\n[2/2] Running NEW version (Var=1, 1/(1+τ))...")
    start_new = time.time()
    results_new = run_experiment(config, 'new')
    time_new = time.time() - start_new
    print(f"  Completed in {time_new:.1f}s")
    
    # Save results
    with open(output_dir / 'results_old.json', 'w') as f:
        json.dump(results_old, f, indent=2)
    with open(output_dir / 'results_new.json', 'w') as f:
        json.dump(results_new, f, indent=2)
    print(f"\n[✓] Saved results to {output_dir}")
    
    # Generate comparison plot
    plot_comparison(results_old, results_new, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Check phase transition point
    for version, results in [('OLD', results_old), ('NEW', results_new)]:
        alphas = sorted(results.keys())
        transition_alpha = None
        for a in alphas:
            if results[a]['Q_Y_mean'] > 0.9:
                transition_alpha = a
                break
        
        final_qy = results[alphas[-1]]['Q_Y_mean']
        print(f"\n{version} version:")
        print(f"  Phase transition: α ≈ {transition_alpha if transition_alpha else 'NOT FOUND'}")
        print(f"  Final Q_Y (α=4): {final_qy:.4f}")
    
    print(f"\nTotal runtime: {time_old + time_new:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
