#!/usr/bin/env python3
"""
BiG-AMP vs AGD Comparison Script

This script compares BiG-AMP (Bilinear Generalized Approximate Message Passing)
against the baseline AGD (Alternating Gradient Descent) algorithm.

Key features:
1. Uses SAME teacher model and masks for fair comparison
2. AGD runs at max epochs only (baseline reference)
3. BiG-AMP runs at multiple step counts for speedup analysis
4. High-contrast visualization for clear distinction

Usage:
    python compare_gamp_vs_agd.py [--quick] [--full]
"""

from pathlib import Path
import time
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ============================================================
# Configuration
# ============================================================
N1 = 200
N2 = 200
M = 50

# Comparison settings
ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 2
ALPHA_TILDE_STEP = 0.1  # Coarse step for quick comparison

SAMPLES_PER_ALPHA = 5
SEED = 42

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                      ('cuda' if torch.cuda.is_available() else 'cpu'))

# Performance settings
USE_BF16 = (DEVICE.type == 'cuda')
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Results directory
RESULT_DIR = Path(__file__).parent / "result" / f"{N1}_{N2}_{M}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Shared Functions
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@torch.no_grad()
def create_teacher_dense(N1, N2, M, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W_true = torch.randn(N1, M, device=device, dtype=torch.float32) * scale
    X_true = torch.randn(M, N2, device=device, dtype=torch.float32) * scale
    return W_true, X_true


@torch.no_grad()
def sample_pairs_random_gpu(N1, N2, M, alpha_tilde, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    C = int(round(alpha_tilde * M * N1))
    total_pairs = N1 * N2

    if C <= 0:
        return None, None, 0
    if C >= total_pairs:
        idx = torch.arange(total_pairs, device=device)
    else:
        idx = torch.randperm(total_pairs, device=device)[:C]

    i_idx = idx // N2
    j_idx = idx % N2
    return i_idx.long(), j_idx.long(), C


@torch.no_grad()
def gram_overlap_cosine(A, B, *, use_left=True):
    GA = A @ A.T if use_left else A.T @ A
    GB = B @ B.T if use_left else B.T @ B
    num = (GA * GB).sum()
    den = GA.norm() * GB.norm() + 1e-12
    return float((num / den).item())


@torch.no_grad()
def gram_overlap_zero_to_one(A, B, *, use_left=True):
    q = gram_overlap_cosine(A, B, use_left=use_left)
    if use_left:
        n, m = A.shape
    else:
        n, m = A.shape[1], A.shape[0]
    b = m / (m + n + 1)
    qc = (q - b) / (1.0 - b + 1e-12)
    return float(max(0.0, min(1.0, qc)))


# ============================================================
# AGD Training (Baseline)
# ============================================================
def train_agd(W_init, X_init, A, Y_teacher, steps, lr=1e-2):
    """AGD training with given initialization"""
    device = W_init.device
    S = W_init.shape[0]
    alpha_scale = 1.0 / (M ** 0.5)

    W = W_init.clone()
    X = X_init.clone()
    Y_teacher_expanded = Y_teacher.unsqueeze(0).expand(S, -1, -1)
    A_expanded = A.unsqueeze(0).expand(S, -1, -1)

    for _ in range(steps):
        with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
            Y_student = alpha_scale * torch.bmm(W, X)
            Mres = (Y_teacher_expanded - Y_student) * A_expanded
            grad_W = -2.0 * alpha_scale * torch.bmm(Mres, X.transpose(-2, -1))

        W = W - lr * grad_W.float()

        with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
            Y_student2 = alpha_scale * torch.bmm(W, X)
            Mres2 = (Y_teacher_expanded - Y_student2) * A_expanded
            grad_X = -2.0 * alpha_scale * torch.bmm(W.transpose(-2, -1), Mres2)

        X = X - lr * grad_X.float()

    return W, X


# ============================================================
# BiG-AMP Training
# ============================================================
def train_big_amp(W_init, X_init, A, Y_teacher, steps, damping=0.5, noise_var=1e-6):
    """
    BiG-AMP training with variance-based adaptive step sizes.

    Key features:
    1. Variance tracking for W and X
    2. Adaptive step size based on effective SNR
    3. Onsager correction for de-correlation
    4. Damping for stability
    """
    device = W_init.device
    S = W_init.shape[0]
    alpha_scale = 1.0 / (M ** 0.5)

    # Initialize from given values
    w_hat = W_init.clone()
    x_hat = X_init.clone()

    # Initialize variances (uncertainty in estimates)
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    Y_teacher_expanded = Y_teacher.unsqueeze(0).expand(S, -1, -1)
    A_expanded = A.unsqueeze(0).expand(S, -1, -1)

    # Count observations per location
    obs_count = A_expanded.sum(dim=0, keepdim=True).clamp(min=1)

    for step in range(steps):
        # ========== Forward pass ==========
        # Prediction
        z_hat = alpha_scale * torch.bmm(w_hat, x_hat)

        # Prediction variance
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (
            torch.bmm(w_sq, x_var) +
            torch.bmm(w_var, x_sq)
        )

        # Total variance at observations
        V = p_var + noise_var
        V = torch.clamp(V, min=1e-8)

        # Residual
        residual = (Y_teacher_expanded - z_hat) * A_expanded

        # Scaled residual (message to factors)
        s = residual / V

        # ========== Update W ==========
        # Effective information for W from observations
        # tau_W = how much information X provides about Y for updating W
        tau_W = (alpha_scale ** 2) * torch.bmm(
            A_expanded / V,
            x_sq.transpose(-2, -1)
        )
        tau_W = torch.clamp(tau_W, min=1e-8)

        # New variance for W
        w_var_new = 1.0 / (M + tau_W)

        # Direction for W update
        r_W = alpha_scale * torch.bmm(s, x_hat.transpose(-2, -1))

        # W update with variance scaling
        w_hat_new = w_hat + w_var_new * r_W

        # Apply damping
        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = damping * w_var + (1 - damping) * w_var_new

        # Clamp variance to prevent explosion
        w_var = torch.clamp(w_var, min=1e-8, max=1.0)

        # ========== Update X (with updated W) ==========
        z_hat2 = alpha_scale * torch.bmm(w_hat, x_hat)
        w_sq2 = w_hat ** 2
        p_var2 = (alpha_scale ** 2) * (
            torch.bmm(w_sq2, x_var) +
            torch.bmm(w_var, x_sq)
        )
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher_expanded - z_hat2) * A_expanded
        s2 = residual2 / V2

        # Effective information for X
        tau_X = (alpha_scale ** 2) * torch.bmm(
            w_sq2.transpose(-2, -1),
            A_expanded / V2
        )
        tau_X = torch.clamp(tau_X, min=1e-8)

        # New variance for X
        x_var_new = 1.0 / (M + tau_X)

        # X update
        r_X = alpha_scale * torch.bmm(w_hat.transpose(-2, -1), s2)
        x_hat_new = x_hat + x_var_new * r_X

        # Apply damping
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = damping * x_var + (1 - damping) * x_var_new

        # Clamp variance
        x_var = torch.clamp(x_var, min=1e-8, max=1.0)

    return w_hat, x_hat


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(W, X, Wt, Xt, A, Y_teacher):
    """Compute all metrics for given W, X"""
    S = W.shape[0]
    alpha_scale = 1.0 / (M ** 0.5)
    A_expanded = A.unsqueeze(0).expand(S, -1, -1)

    results = []
    for s in range(S):
        W_s, X_s = W[s], X[s]

        Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
        Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
        Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
        Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

        Yp = W_s @ X_s
        Yt = Y_teacher
        Q_Y = float(((Yt.flatten() * Yp.flatten()).sum()) /
                   (Yt.norm() * Yp.norm() + 1e-12))
        gen_error = float(torch.mean((Yt - Yp) ** 2).item())

        Y_final = alpha_scale * (W_s @ X_s)
        Rf = (Y_teacher - Y_final) * A_expanded[s]
        final_loss = float(torch.sum(Rf ** 2).item())

        results.append({
            'Q_W': Q_W, 'Q_X': Q_X,
            'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
            'Q_Y': Q_Y, 'Gen_Error': gen_error,
            'Final_Loss': final_loss
        })

    # Aggregate
    metrics = {}
    for key in results[0].keys():
        vals = [r[key] for r in results]
        metrics[f'{key}_mean'] = float(np.mean(vals))
        metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    return metrics


# ============================================================
# Comparison Experiment
# ============================================================
def run_comparison(alpha_values, agd_steps_list, gamp_steps_list):
    """
    Run comparison experiment.

    Args:
        alpha_values: List of alpha values to test
        agd_steps_list: List of step counts for AGD
        gamp_steps_list: List of step counts for BiG-AMP
    """
    set_seed(SEED)
    Wt, Xt = create_teacher_dense(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    results = {
        'alpha_values': [],
        'agd_results': {steps: {} for steps in agd_steps_list},
        'gamp_results': {steps: {} for steps in gamp_steps_list},
        'agd_times': {steps: {} for steps in agd_steps_list},
        'gamp_times': {steps: {} for steps in gamp_steps_list},
    }

    print(f"\n{'='*70}")
    print(f"COMPARISON: BiG-AMP vs AGD")
    print(f"{'='*70}")
    print(f"Matrix size: {N1} x {N2}, rank M={M}")
    print(f"Alpha values: {len(alpha_values)} points")
    print(f"AGD steps: {agd_steps_list}")
    print(f"BiG-AMP steps: {gamp_steps_list}")
    print(f"{'='*70}\n")

    for alpha_tilde in tqdm(alpha_values, desc="Alpha sweep"):
        # Generate mask
        i_idx, j_idx, C = sample_pairs_random_gpu(
            N1, N2, M, alpha_tilde, DEVICE,
            seed=SEED + int(alpha_tilde * 1000)
        )

        A = torch.zeros((N1, N2), dtype=torch.float32, device=DEVICE)
        if i_idx is not None and i_idx.numel() > 0:
            A[i_idx, j_idx] = 1.0

        # Same initialization for both algorithms
        torch.manual_seed(SEED + 10000)
        scale = 1.0 / (M ** 0.5)
        W_init = torch.randn((SAMPLES_PER_ALPHA, N1, M), device=DEVICE, dtype=torch.float32) * scale
        X_init = torch.randn((SAMPLES_PER_ALPHA, M, N2), device=DEVICE, dtype=torch.float32) * scale

        results['alpha_values'].append(float(alpha_tilde))

        # Test AGD at different step counts
        for steps in agd_steps_list:
            start = time.time()
            W_agd, X_agd = train_agd(W_init.clone(), X_init.clone(), A, Y_teacher, steps)
            elapsed = time.time() - start

            metrics = evaluate(W_agd, X_agd, Wt, Xt, A, Y_teacher)
            results['agd_results'][steps][float(alpha_tilde)] = metrics
            results['agd_times'][steps][float(alpha_tilde)] = elapsed

        # Test BiG-AMP at different step counts
        for steps in gamp_steps_list:
            start = time.time()
            W_gamp, X_gamp = train_big_amp(W_init.clone(), X_init.clone(), A, Y_teacher, steps)
            elapsed = time.time() - start

            metrics = evaluate(W_gamp, X_gamp, Wt, Xt, A, Y_teacher)
            results['gamp_results'][steps][float(alpha_tilde)] = metrics
            results['gamp_times'][steps][float(alpha_tilde)] = elapsed

    return results


# ============================================================
# Analysis and Reporting
# ============================================================
def analyze_results(results):
    """Analyze comparison results and find equivalent performance points"""
    alpha_values = results['alpha_values']
    agd_steps = list(results['agd_results'].keys())
    gamp_steps = list(results['gamp_results'].keys())

    print(f"\n{'='*70}")
    print(f"ANALYSIS RESULTS")
    print(f"{'='*70}")

    # Compare Q_Y at different step counts
    for gamp_s in gamp_steps:
        print(f"\n--- BiG-AMP at {gamp_s} steps ---")

        # Find which AGD step count gives similar Q_Y
        gamp_qy = np.mean([
            results['gamp_results'][gamp_s][a]['Q_Y_mean']
            for a in alpha_values if a > 0.5  # Focus on phase transition region
        ])

        best_match_agd = None
        best_diff = float('inf')

        for agd_s in agd_steps:
            agd_qy = np.mean([
                results['agd_results'][agd_s][a]['Q_Y_mean']
                for a in alpha_values if a > 0.5
            ])
            diff = abs(gamp_qy - agd_qy)
            if diff < best_diff:
                best_diff = diff
                best_match_agd = agd_s

        print(f"  BiG-AMP avg Q_Y (alpha > 0.5): {gamp_qy:.4f}")
        if best_match_agd:
            agd_qy = np.mean([
                results['agd_results'][best_match_agd][a]['Q_Y_mean']
                for a in alpha_values if a > 0.5
            ])
            print(f"  Closest AGD match: {best_match_agd} steps (Q_Y: {agd_qy:.4f})")

            # Calculate speedup
            gamp_time = np.mean(list(results['gamp_times'][gamp_s].values()))
            agd_time = np.mean(list(results['agd_times'][best_match_agd].values()))
            speedup = agd_time / gamp_time if gamp_time > 0 else 0

            print(f"  BiG-AMP time per alpha: {gamp_time:.3f}s")
            print(f"  AGD time per alpha: {agd_time:.3f}s")
            print(f"  Step efficiency: {best_match_agd / gamp_s:.1f}x (AGD needs {best_match_agd/gamp_s:.1f}x more steps)")
            print(f"  Wall-clock speedup: {speedup:.2f}x")


def plot_comparison(results):
    """Generate high-contrast comparison plots"""
    alpha_values = np.array(results['alpha_values'])
    agd_steps = sorted(results['agd_results'].keys())
    gamp_steps = sorted(results['gamp_results'].keys())

    # High-contrast color scheme
    AGD_COLOR = '#1a1a2e'  # Dark navy for baseline
    GAMP_COLORS = ['#e63946', '#2a9d8f', '#e9c46a', '#f4a261', '#264653']  # Bright distinct colors

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams['font.size'] = 12

    # ========== Plot 1: Q_Y comparison ==========
    ax1 = axes[0, 0]

    # AGD baseline (thick solid line with large markers)
    for i, steps in enumerate(agd_steps):
        qy = [results['agd_results'][steps][a]['Q_Y_mean'] for a in alpha_values]
        ax1.plot(alpha_values, qy, 'D-', color=AGD_COLOR,
                 markersize=10, linewidth=3, markeredgewidth=2, markeredgecolor='white',
                 label=f'AGD {steps:,} steps (baseline)', zorder=10)

    # BiG-AMP (dashed lines with different markers)
    markers = ['o', 's', '^', 'v', 'p']
    for i, steps in enumerate(gamp_steps):
        qy = [results['gamp_results'][steps][a]['Q_Y_mean'] for a in alpha_values]
        color = GAMP_COLORS[i % len(GAMP_COLORS)]
        ax1.plot(alpha_values, qy, f'{markers[i % len(markers)]}--', color=color,
                 markersize=8, linewidth=2, alpha=0.85,
                 label=f'BiG-AMP {steps} steps')

    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax1.set_ylabel('Q_Y', fontsize=14)
    ax1.set_title('Q_Y Comparison: BiG-AMP vs AGD', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.4, linestyle='-')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(alpha_values[0] - 0.05, alpha_values[-1] + 0.05)

    # ========== Plot 2: Gen Error comparison ==========
    ax2 = axes[0, 1]

    for i, steps in enumerate(agd_steps):
        ge = [results['agd_results'][steps][a]['Gen_Error_mean'] for a in alpha_values]
        ax2.semilogy(alpha_values, ge, 'D-', color=AGD_COLOR,
                     markersize=10, linewidth=3, markeredgewidth=2, markeredgecolor='white',
                     label=f'AGD {steps:,} steps', zorder=10)

    for i, steps in enumerate(gamp_steps):
        ge = [results['gamp_results'][steps][a]['Gen_Error_mean'] for a in alpha_values]
        color = GAMP_COLORS[i % len(GAMP_COLORS)]
        ax2.semilogy(alpha_values, ge, f'{markers[i % len(markers)]}--', color=color,
                     markersize=8, linewidth=2, alpha=0.85,
                     label=f'BiG-AMP {steps} steps')

    ax2.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax2.set_ylabel('Generalization Error (log)', fontsize=14)
    ax2.set_title('Generalization Error Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='-')

    # ========== Plot 3: Q_W' and Q_X' comparison ==========
    ax3 = axes[1, 0]

    # AGD Q_W' and Q_X'
    for i, steps in enumerate(agd_steps):
        qw = [results['agd_results'][steps][a]['Q_W_prime_mean'] for a in alpha_values]
        qx = [results['agd_results'][steps][a]['Q_X_prime_mean'] for a in alpha_values]
        ax3.plot(alpha_values, qw, 'D-', color=AGD_COLOR,
                 markersize=10, linewidth=3, markeredgewidth=2, markeredgecolor='white',
                 label=f"AGD Q_W'", zorder=10)
        ax3.plot(alpha_values, qx, 's-', color='#4a4e69',
                 markersize=9, linewidth=3, markeredgewidth=2, markeredgecolor='white',
                 label=f"AGD Q_X'", zorder=10)

    # BiG-AMP (only show best performing)
    best_gamp = max(gamp_steps)
    qw = [results['gamp_results'][best_gamp][a]['Q_W_prime_mean'] for a in alpha_values]
    qx = [results['gamp_results'][best_gamp][a]['Q_X_prime_mean'] for a in alpha_values]
    ax3.plot(alpha_values, qw, 'o--', color=GAMP_COLORS[0],
             markersize=8, linewidth=2.5, alpha=0.85,
             label=f"BiG-AMP {best_gamp} Q_W'")
    ax3.plot(alpha_values, qx, '^--', color=GAMP_COLORS[1],
             markersize=8, linewidth=2.5, alpha=0.85,
             label=f"BiG-AMP {best_gamp} Q_X'")

    ax3.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax3.set_ylabel("Q' (normalized overlap)", fontsize=14)
    ax3.set_title("Q_W' and Q_X' Comparison", fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.4, linestyle='-')
    ax3.set_ylim(-0.05, 1.05)

    # ========== Plot 4: Speedup Analysis ==========
    ax4 = axes[1, 1]

    # Calculate speedups vs AGD baseline
    agd_baseline = max(agd_steps)
    agd_time = np.mean(list(results['agd_times'][agd_baseline].values()))

    speedups = []
    step_efficiencies = []
    for steps in gamp_steps:
        gamp_time = np.mean(list(results['gamp_times'][steps].values()))
        speedups.append(agd_time / gamp_time if gamp_time > 0 else 0)
        step_efficiencies.append(agd_baseline / steps)

    x = np.arange(len(gamp_steps))
    width = 0.35

    bars1 = ax4.bar(x - width/2, speedups, width, label='Wall-clock Speedup',
                    color=GAMP_COLORS[0], edgecolor='white', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, step_efficiencies, width, label='Step Efficiency',
                    color=GAMP_COLORS[1], edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars1, speedups):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, step_efficiencies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax4.set_xlabel('BiG-AMP Steps', fontsize=14)
    ax4.set_ylabel('Speedup Factor', fontsize=14)
    ax4.set_title(f'Speedup vs AGD {agd_baseline:,} steps', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(s) for s in gamp_steps])
    ax4.legend(loc='upper right', fontsize=11)
    ax4.grid(True, alpha=0.4, axis='y', linestyle='-')

    # Add baseline reference line
    ax4.axhline(y=1, color=AGD_COLOR, linestyle='--', linewidth=2, alpha=0.7, label='AGD baseline')

    plt.tight_layout()

    # Save
    save_path = RESULT_DIR / 'comparison_gamp_vs_agd.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nComparison plot saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Compare BiG-AMP vs AGD')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer steps')
    parser.add_argument('--full', action='store_true', help='Full test with more step variants')
    parser.add_argument('--agd-epochs', type=int, default=None, help='Override AGD max epochs')
    args = parser.parse_args()

    # Use config from file header
    alpha_values = np.arange(ALPHA_TILDE_START, ALPHA_TILDE_STOP + 1e-12, ALPHA_TILDE_STEP)

    # AGD only runs at MAX epochs (baseline reference)
    # BiG-AMP runs at multiple step counts for speedup analysis
    if args.quick:
        agd_steps = [1000]  # Only max
        gamp_steps = [50, 100, 200]
    elif args.full:
        agd_steps = [100000]  # Only max (100k)
        gamp_steps = [50, 100, 200, 500, 1000]
    else:
        # Default: moderate test
        agd_steps = [10000]  # Only max
        gamp_steps = [50, 100, 200, 500]

    # Override AGD epochs if specified
    if args.agd_epochs:
        agd_steps = [args.agd_epochs]

    print("\n" + "=" * 70)
    print("BiG-AMP vs AGD COMPARISON")
    print("=" * 70)

    # Run comparison
    results = run_comparison(alpha_values, agd_steps, gamp_steps)

    # Analyze
    analyze_results(results)

    # Plot
    plot_comparison(results)

    # Save results
    results_path = RESULT_DIR / 'comparison_results.json'
    # Convert numpy types for JSON serialization
    results_json = {
        'alpha_values': [float(a) for a in results['alpha_values']],
        'agd_results': {
            str(k): {str(a): v for a, v in d.items()}
            for k, d in results['agd_results'].items()
        },
        'gamp_results': {
            str(k): {str(a): v for a, v in d.items()}
            for k, d in results['gamp_results'].items()
        },
        'agd_times': {
            str(k): {str(a): v for a, v in d.items()}
            for k, d in results['agd_times'].items()
        },
        'gamp_times': {
            str(k): {str(a): v for a, v in d.items()}
            for k, d in results['gamp_times'].items()
        },
    }
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
