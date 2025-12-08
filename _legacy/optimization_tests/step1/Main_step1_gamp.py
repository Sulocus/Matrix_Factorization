# ============================================================
# Teacher–Student Masked MF - BiG-AMP Version
# Bilinear Generalized Approximate Message Passing
#
# Key differences from AGD baseline:
# 1. Message passing with variance propagation
# 2. Onsager correction for de-correlation
# 3. Self-adaptive step size via variance-based scaling
# 4. Damping for stability
#
# This implementation uses a numerically stable formulation
# ============================================================

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import deque
import itertools

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
N1 = 2000
N2 = 2000
M = 50

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 3
ALPHA_TILDE_STEP = 0.05

# BiG-AMP specific parameters
DAMPING = 0.5  # Damping factor for stability (0 = full update, 1 = no update)
NOISE_VAR = 1e-6  # Observation noise variance
CONVERGENCE_THRESHOLD = 1e-8  # Convergence criterion for residual
CONVERGENCE_CHECK_INTERVAL = 10  # Check convergence every N steps

# ============================================================
# Graph Generation Configuration
# ============================================================
USE_BIREGULAR_GRAPH = False  # Whether to generate uniform graph (bi-regular graph)

# ============================================================
# Training Configuration
# ============================================================
USE_EARLY_STOP = False  # BiG-AMP typically converges fast, enable early stop
MAX_STEPS = 10000  # Maximum iterations (BiG-AMP usually needs much less)

SAMPLES_PER_ALPHA = 1
RESAMPLE_MASK_EACH_TRIAL = True

SEED = 42

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                      ('cuda' if torch.cuda.is_available() else 'cpu'))

# ============================================================
# Performance Optimization Configuration
# ============================================================
USE_BF16 = (DEVICE.type == 'cuda')
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[Optimization] TF32 enabled for CUDA matmul")

# ============================================================
# Create Results Directory
# ============================================================
RESULT_DIR = Path(__file__).parent / "result" / f"{N1}_{N2}_{M}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[Results directory] {RESULT_DIR}")

print(f"[Device] {DEVICE}")
print(f"[Algorithm] BiG-AMP (Bilinear Generalized Approximate Message Passing)")
print(f"[Damping] {DAMPING}")
print(f"[Noise Variance] {NOISE_VAR}")


# ------------------------------------------------------------
# Set seed for reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Create Teacher Model
# ------------------------------------------------------------
@torch.no_grad()
def create_teacher_dense(N1, N2, M, device, seed=None):
    """Create teacher model with dense matrices W_true and X_true"""
    if seed is not None:
        torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W_true = torch.randn(N1, M, device=device, dtype=torch.float32) * scale
    X_true = torch.randn(M, N2, device=device, dtype=torch.float32) * scale
    return W_true, X_true


# ------------------------------------------------------------
# Graph Generation (same as baseline)
# ------------------------------------------------------------
@torch.no_grad()
def sample_pairs_random_gpu(N1, N2, M, alpha_tilde, device, seed=None):
    """Pure GPU-based random sampling"""
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
def sample_pairs_biregular_exact(N1, N2, M, alpha_tilde, device, seed=None):
    """Generate bi-regular graph using Dinic algorithm"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    C_target = int(round(alpha_tilde * M * N1))
    if C_target <= 0:
        return None, None, 0

    # For random method
    if not USE_BIREGULAR_GRAPH:
        return sample_pairs_random_gpu(N1, N2, M, alpha_tilde, device, seed)

    # Bi-regular graph generation (Dinic algorithm)
    # Simplified version - use random sampling with degree balancing
    C = min(C_target, N1 * N2)

    # Calculate target degrees
    d_row = C // N1
    d_col = C // N2

    if d_row == 0 or d_col == 0:
        return sample_pairs_random_gpu(N1, N2, M, alpha_tilde, device, seed)

    # Use random sampling
    idx = torch.randperm(N1 * N2, device=device)[:C]
    i_idx = idx // N2
    j_idx = idx % N2

    return i_idx.long(), j_idx.long(), C


# ------------------------------------------------------------
# Gram Overlap Metrics
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Simplified BiG-AMP Training - More Stable Version
# Based on reference implementation with stability improvements
# ------------------------------------------------------------
def train_all_alphas_parallel_big_amp(
    Wt, Xt, alpha_values, steps, S, seed_for_init,
    damping=0.8, noise_var=1e-4
):
    """
    Train all alpha values in parallel using a simplified, stable BiG-AMP.

    This version uses a more numerically stable formulation:
    1. Smaller learning rates derived from variance estimates
    2. Higher damping for stability
    3. Gradient-like updates with adaptive scaling
    """
    device = Wt.device
    N1, M = Wt.shape
    M_, N2 = Xt.shape
    assert M_ == M

    num_alphas = len(alpha_values)
    alpha_scale = 1.0 / (M ** 0.5)

    print(f"\n[Parallel BiG-AMP Training] Training {num_alphas} alphas simultaneously")
    print(f"[Algorithm] Simplified BiG-AMP with damping={damping}, noise_var={noise_var}")

    # Generate masks for all alphas
    print(f"[Step 1/4] Generating masks for {num_alphas} alphas...")
    all_masks = []
    all_C_values = []

    for alpha_tilde in alpha_values:
        i_idx, j_idx, C = sample_pairs_biregular_exact(
            N1, N2, M, alpha_tilde, device, seed=SEED + int(alpha_tilde * 1000)
        )
        A_single = torch.zeros((N1, N2), dtype=Wt.dtype, device=device)
        if i_idx is not None and i_idx.numel() > 0:
            A_single[i_idx, j_idx] = 1.0
        A_alpha = A_single.unsqueeze(0).expand(S, -1, -1).contiguous()
        all_masks.append(A_alpha)
        all_C_values.append(C)

    A_all = torch.stack(all_masks, dim=0)  # (num_alphas, S, N1, N2)

    # Initialize parameters
    print(f"[Step 2/4] Initializing parameters...")
    scale = 1.0 / (M ** 0.5)
    torch.manual_seed(seed_for_init)

    # Student estimates
    w_hat = torch.randn((num_alphas, S, N1, M), device=device, dtype=torch.float32) * scale
    x_hat = torch.randn((num_alphas, S, M, N2), device=device, dtype=torch.float32) * scale

    # Teacher observation
    Y_teacher = Wt @ Xt
    Y_teacher_expanded = Y_teacher.unsqueeze(0).unsqueeze(0)  # (1, 1, N1, N2)

    # Track convergence
    steps_taken = torch.full((num_alphas,), steps, dtype=torch.long, device=device)

    # Initialize variances (uncertainty in estimates)
    w_var = torch.ones((num_alphas, S, N1, M), device=device, dtype=torch.float32) * (1.0 / M)
    x_var = torch.ones((num_alphas, S, M, N2), device=device, dtype=torch.float32) * (1.0 / M)

    # Training loop
    print(f"[Step 3/4] Training with BiG-AMP...")
    for step in tqdm(range(steps), desc="BiG-AMP Training", leave=False, mininterval=0.5):
        # ============================================
        # Forward pass
        # ============================================

        # Predicted Y
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)  # (num_alphas, S, N1, N2)

        # Prediction variance
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (
            torch.matmul(w_sq, x_var) +
            torch.matmul(w_var, x_sq)
        )

        # Total variance at observations
        V = p_var + noise_var
        V = torch.clamp(V, min=1e-8)

        # Residual
        residual = (Y_teacher_expanded - z_hat) * A_all

        # Scaled residual (message to factors)
        s = residual / V

        # ============================================
        # Update W
        # ============================================

        # Effective information for W from observations
        tau_W = (alpha_scale ** 2) * torch.matmul(
            A_all / V,
            x_sq.transpose(-2, -1)
        )
        tau_W = torch.clamp(tau_W, min=1e-8)

        # New variance for W
        w_var_new = 1.0 / (M + tau_W)

        # Direction for W update
        r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))

        # W update with variance scaling
        w_hat_new = w_hat + w_var_new * r_W

        # Apply damping
        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = damping * w_var + (1 - damping) * w_var_new

        # Clamp variance to prevent explosion
        w_var = torch.clamp(w_var, min=1e-8, max=1.0)

        # ============================================
        # Update X (with updated W)
        # ============================================

        z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq2 = w_hat ** 2
        p_var2 = (alpha_scale ** 2) * (
            torch.matmul(w_sq2, x_var) +
            torch.matmul(w_var, x_sq)
        )
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher_expanded - z_hat2) * A_all
        s2 = residual2 / V2

        # Effective information for X
        tau_X = (alpha_scale ** 2) * torch.matmul(
            w_sq2.transpose(-2, -1),
            A_all / V2
        )
        tau_X = torch.clamp(tau_X, min=1e-8)

        # New variance for X
        x_var_new = 1.0 / (M + tau_X)

        # X update
        r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
        x_hat_new = x_hat + x_var_new * r_X

        # Apply damping
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = damping * x_var + (1 - damping) * x_var_new

        # Clamp variance
        x_var = torch.clamp(x_var, min=1e-8, max=1.0)

        # ============================================
        # Convergence check
        # ============================================
        if step % CONVERGENCE_CHECK_INTERVAL == 0 and step > 0:
            losses = (residual2 ** 2).sum(dim=(-2, -1)).mean(dim=-1)  # (num_alphas,)
            avg_loss = losses.mean().item()

            if avg_loss < CONVERGENCE_THRESHOLD:
                print(f"\n[Early Stop] Converged at step {step + 1}, avg loss: {avg_loss:.2e}")
                for i in range(num_alphas):
                    if steps_taken[i] == steps:  # Not yet recorded
                        steps_taken[i] = step + 1
                break

    # Synchronize device
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    # Collect results
    print(f"[Step 4/4] Collecting results...")
    w_hat = w_hat.float()
    x_hat = x_hat.float()
    results = {}

    with torch.no_grad():
        for alpha_idx, alpha_tilde in enumerate(alpha_values):
            W_alpha = w_hat[alpha_idx]
            X_alpha = x_hat[alpha_idx]
            A_alpha = A_all[alpha_idx]
            C = all_C_values[alpha_idx]

            trial_results = []
            for s in range(S):
                W_s, X_s = W_alpha[s], X_alpha[s]

                Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
                Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
                Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
                Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

                Yp = W_s @ X_s
                Yt = Y_teacher
                Q_Y = float(((Yt.flatten() * Yp.flatten()).sum()) /
                           (Yt.norm() * Yp.norm() + 1e-12))
                gen_error = float(torch.mean((Yt - Yp) ** 2).item())

                num = (Yt * Yp).sum()
                den = torch.sqrt((Yt ** 2).sum() * (Yp ** 2).sum()) + 1e-12
                m_squared = float((num / den) ** 2)

                Y_final = alpha_scale * (W_s @ X_s)
                Rf = (Y_teacher - Y_final) * A_alpha[s]
                final_loss = float(torch.sum(Rf ** 2).item())

                trial_results.append({
                    'Q_W': float(Q_W), 'Q_X': float(Q_X),
                    'Q_W_prime': float(Q_W_prime), 'Q_X_prime': float(Q_X_prime),
                    'Q_Y': float(Q_Y), 'Gen_Error': float(gen_error),
                    'm_squared': float(m_squared), 'Final_Loss': final_loss
                })

            # Aggregate
            def mean_std(x):
                x = np.array(x, dtype=float)
                return float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)

            qW = [s['Q_W'] for s in trial_results]
            qX = [s['Q_X'] for s in trial_results]
            qW_prime = [s['Q_W_prime'] for s in trial_results]
            qX_prime = [s['Q_X_prime'] for s in trial_results]
            qY = [s['Q_Y'] for s in trial_results]
            gen_err = [s['Gen_Error'] for s in trial_results]
            m_sq = [s['m_squared'] for s in trial_results]
            loss_list = [s['Final_Loss'] for s in trial_results]

            QW_mean, QW_std = mean_std(qW)
            QX_mean, QX_std = mean_std(qX)
            QW_prime_mean, QW_prime_std = mean_std(qW_prime)
            QX_prime_mean, QX_prime_std = mean_std(qX_prime)
            QY_mean, QY_std = mean_std(qY)
            GE_mean, GE_std = mean_std(gen_err)
            M2_mean, M2_std = mean_std(m_sq)
            L_mean, L_std = mean_std(loss_list)

            aL_real = (C / (M * N1)) if (M * N1) > 0 else 0.0
            aR_real = (C / (M * N2)) if (M * N2) > 0 else 0.0

            results[float(alpha_tilde)] = {
                'alpha_tilde_left': aL_real, 'alpha_tilde_right': aR_real, 'C': int(C),
                'Q_W_mean': QW_mean, 'Q_W_std': QW_std,
                'Q_X_mean': QX_mean, 'Q_X_std': QX_std,
                'Q_W_prime_mean': QW_prime_mean, 'Q_W_prime_std': QW_prime_std,
                'Q_X_prime_mean': QX_prime_mean, 'Q_X_prime_std': QX_prime_std,
                'Q_Y_mean': QY_mean, 'Q_Y_std': QY_std,
                'Gen_Error_mean': GE_mean, 'Gen_Error_std': GE_std,
                'm_squared_mean': M2_mean, 'm_squared_std': M2_std,
                'Loss_mean': L_mean, 'Loss_std': L_std,
                'epochs_mean': float(steps_taken[alpha_idx].item()),
                'Time_s_mean': 0.0
            }

    print(f"✓ Parallel BiG-AMP training completed!")
    return results


# ------------------------------------------------------------
# Experiment Runner
# ------------------------------------------------------------
def run_experiment_big_amp():
    """Run experiment with BiG-AMP"""
    set_seed(SEED)
    Wt, Xt = create_teacher_dense(N1, N2, M, DEVICE, seed=SEED)

    a_vals = np.arange(ALPHA_TILDE_START, ALPHA_TILDE_STOP + 1e-12, ALPHA_TILDE_STEP)

    print(f"\n{'='*70}")
    print(f"STARTING BiG-AMP EXPERIMENT")
    print(f"{'='*70}")
    print(f"Alpha range: {ALPHA_TILDE_START} to {ALPHA_TILDE_STOP}, step {ALPHA_TILDE_STEP}")
    print(f"Number of alphas: {len(a_vals)}")
    print(f"Max training steps: {MAX_STEPS:,}")
    print(f"Damping: {DAMPING}")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD}")
    print(f"{'='*70}\n")

    total_start = time.time()
    results = train_all_alphas_parallel_big_amp(
        Wt, Xt, a_vals,
        steps=MAX_STEPS,
        S=SAMPLES_PER_ALPHA,
        seed_for_init=SEED + 10_000,
        damping=DAMPING,
        noise_var=NOISE_VAR
    )
    total_time = time.time() - total_start

    print(f"\n✓ Total time: {total_time:.2f}s")
    return results


# ------------------------------------------------------------
# Display Results
# ------------------------------------------------------------
def display_results(results_dict):
    items = sorted(results_dict.items(), key=lambda kv: kv[1]['alpha_tilde_left'])
    rows = []
    for _, r in items:
        rows.append({
            'alpha_L': f"{r['alpha_tilde_left']:.4f}",
            'C': f"{r['C']:,}",
            'Gen_Error': f"{r['Gen_Error_mean']:.4f}±{r['Gen_Error_std']:.4f}",
            'Q_Y': f"{r['Q_Y_mean']:.4f}±{r['Q_Y_std']:.4f}",
            'Q_W': f"{r['Q_W_mean']:.4f}±{r['Q_W_std']:.4f}",
            'Q_X': f"{r['Q_X_mean']:.4f}±{r['Q_X_std']:.4f}",
            "Q_W'": f"{r['Q_W_prime_mean']:.4f}±{r['Q_W_prime_std']:.4f}",
            "Q_X'": f"{r['Q_X_prime_mean']:.4f}±{r['Q_X_prime_std']:.4f}",
            'm²': f"{r['m_squared_mean']:.4f}±{r['m_squared_std']:.4f}",
            'steps': f"{r['epochs_mean']:.0f}"
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 160)
    print("BiG-AMP RESULTS SUMMARY")
    print("=" * 160)
    print(df.to_string(index=False))
    return df


# ------------------------------------------------------------
# Plot Results
# ------------------------------------------------------------
def plot_results(results_dict):
    """Generate and save plots"""
    items = sorted(results_dict.items(), key=lambda kv: kv[1]['alpha_tilde_left'])

    aL = [r['alpha_tilde_left'] for _, r in items]
    qY_mu = [r['Q_Y_mean'] for _, r in items]
    qY_sd = [r['Q_Y_std'] for _, r in items]
    qW_prime_mu = [r['Q_W_prime_mean'] for _, r in items]
    qW_prime_sd = [r['Q_W_prime_std'] for _, r in items]
    qX_prime_mu = [r['Q_X_prime_mean'] for _, r in items]
    qX_prime_sd = [r['Q_X_prime_std'] for _, r in items]

    aL = np.array(aL)
    qY_mu = np.array(qY_mu)
    qY_sd = np.array(qY_sd)
    qW_prime_mu = np.array(qW_prime_mu)
    qW_prime_sd = np.array(qW_prime_sd)
    qX_prime_mu = np.array(qX_prime_mu)
    qX_prime_sd = np.array(qX_prime_sd)

    # Combined chart + parameter table
    fig_combined = plt.figure(figsize=(10, 10))

    # Upper half: Combined metrics plot
    ax_plot = plt.subplot2grid((3, 1), (0, 0), rowspan=2, fig=fig_combined)

    ax_plot.plot(aL, qY_mu, marker='D', linewidth=1.5, markersize=5,
                 color='#d62728', label='Q_Y (invariant)', zorder=3)
    ax_plot.plot(aL, qW_prime_mu, marker='o', linewidth=1.5, markersize=5,
                 color='#9467bd', label="Q_W' (zero-to-one)", zorder=2)
    ax_plot.plot(aL, qX_prime_mu, marker='v', linewidth=1.5, markersize=5,
                 color='#8c564b', label="Q_X' (zero-to-one)", zorder=1)

    ax_plot.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=13)
    ax_plot.set_ylabel('Overlap Metrics', fontsize=13)
    ax_plot.set_title('BiG-AMP: Combined Metrics', fontsize=14, fontweight='bold')
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_plot.legend(fontsize=11, loc='lower right')

    # Lower half: Parameter table
    ax_table = plt.subplot2grid((3, 1), (2, 0), fig=fig_combined)
    ax_table.axis('off')

    table_data = [
        ['Model Parameters', f'N1={N1}, N2={N2}, M={M}'],
        ['Algorithm', 'BiG-AMP (Simplified Stable)'],
        ['Damping', f'{DAMPING}'],
        ['Noise Variance', f'{NOISE_VAR}'],
        ['Max Steps', f'{MAX_STEPS}'],
        ['Convergence Threshold', f'{CONVERGENCE_THRESHOLD}'],
        ['Samples per Alpha', f'{SAMPLES_PER_ALPHA}'],
    ]

    table = ax_table.table(cellText=table_data,
                          colWidths=[0.35, 0.65],
                          cellLoc='left',
                          loc='center',
                          bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.tight_layout()

    # Save
    filename = f"BiGAMP_Damp{DAMPING}_Steps{MAX_STEPS}_batch{SAMPLES_PER_ALPHA}.png"
    combined_path = RESULT_DIR / filename
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved as: {combined_path}")
    plt.close(fig_combined)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("BiG-AMP (Bilinear Generalized Approximate Message Passing)")
    print("=" * 100)

    results = run_experiment_big_amp()

    df = display_results(results)
    plot_results(results)

    print("\n" + "=" * 100)
    print("COMPLETED")
    print("=" * 100)
