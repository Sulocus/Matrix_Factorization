# ============================================================
# Teacher–Student Masked MF - Smart Parallel Version
# Automatically adjusts parallelism based on device memory
#
# Features:
# 1. Auto-detect CUDA/MPS/CPU and available memory
# 2. Estimate memory per alpha and choose optimal parallelism
# 3. Support both AGD and BiG-AMP algorithms
# 4. Batch processing when full parallelism isn't possible
# ============================================================

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional, Literal
import json

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
N1 = 10000
N2 = 10000
M = 100

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 2
ALPHA_TILDE_STEP = 0.1

# Training configuration
EPOCHS_PER_ALPHA = 200  # For BiG-AMP, this is very small
SAMPLES_PER_ALPHA = 1
SEED = 42

# Algorithm selection: "agd" or "bigamp"
ALGORITHM = "bigamp"

# AGD specific
AGD_LEARNING_RATE = 1e-2

# BiG-AMP specific
BIGAMP_DAMPING = 0.5
BIGAMP_NOISE_VAR = 1e-6

# Graph generation
USE_BIREGULAR_GRAPH = False
RESAMPLE_MASK_EACH_TRIAL = True

# ============================================================
# Device and Memory Detection
# ============================================================
@dataclass
class DeviceInfo:
    device: torch.device
    device_type: str  # 'cuda', 'mps', 'cpu'
    total_memory_gb: float
    available_memory_gb: float
    compute_dtype: torch.dtype
    use_bf16: bool


def get_device_info() -> DeviceInfo:
    """Detect device and available memory"""

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'

        # Get GPU memory
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_memory
        reserved_mem = torch.cuda.memory_reserved(0)
        allocated_mem = torch.cuda.memory_allocated(0)

        total_gb = total_mem / (1024**3)
        # Use 80% of free memory to be safe
        available_gb = (total_mem - reserved_mem) * 0.8 / (1024**3)

        use_bf16 = True
        compute_dtype = torch.bfloat16

        # Enable TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'

        # MPS doesn't have direct memory query, estimate from system
        import subprocess
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                   capture_output=True, text=True)
            total_mem = int(result.stdout.strip())
            total_gb = total_mem / (1024**3)
            # MPS uses unified memory, be conservative (use 30%)
            available_gb = total_gb * 0.3
        except:
            total_gb = 16.0  # Default assumption
            available_gb = 4.0

        use_bf16 = False  # MPS doesn't support BF16 well
        compute_dtype = torch.float32

    else:
        device = torch.device('cpu')
        device_type = 'cpu'

        import psutil
        total_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3) * 0.5

        use_bf16 = False
        compute_dtype = torch.float32

    return DeviceInfo(
        device=device,
        device_type=device_type,
        total_memory_gb=total_gb,
        available_memory_gb=available_gb,
        compute_dtype=compute_dtype,
        use_bf16=use_bf16
    )


def estimate_memory_per_alpha(N1: int, N2: int, M: int, S: int,
                               algorithm: str, dtype_bytes: int = 4) -> float:
    """
    Estimate GPU memory needed per alpha in GB.

    Main tensors per alpha:
    - W: (S, N1, M)
    - X: (S, M, N2)
    - A: (S, N1, N2)
    - Y_student: (S, N1, N2)
    - residual: (S, N1, N2)
    - gradients: similar sizes

    For BiG-AMP, add variance tensors (2x W, X)
    """

    # Base tensors
    W_size = S * N1 * M
    X_size = S * M * N2
    A_size = S * N1 * N2
    Y_size = S * N1 * N2

    # AGD needs: W, X, A, Y_student, residual, grad_W, grad_X
    agd_elements = W_size + X_size + A_size + 3 * Y_size + W_size + X_size

    # BiG-AMP needs additional variance tensors
    bigamp_elements = agd_elements + 2 * W_size + 2 * X_size  # w_var, x_var

    if algorithm == "bigamp":
        total_elements = bigamp_elements
    else:
        total_elements = agd_elements

    # Add 50% overhead for intermediate computations
    total_elements *= 1.5

    # Convert to GB
    memory_gb = total_elements * dtype_bytes / (1024**3)

    return memory_gb


def calculate_optimal_parallelism(device_info: DeviceInfo,
                                   N1: int, N2: int, M: int, S: int,
                                   num_alphas: int, algorithm: str,
                                   user_override: Optional[int] = None) -> int:
    """
    Calculate optimal number of alphas to process in parallel.

    Returns the number of alphas to batch together.
    """

    if user_override is not None:
        print(f"[Parallelism] User override: {user_override} alphas")
        return min(user_override, num_alphas)

    dtype_bytes = 2 if device_info.use_bf16 else 4
    mem_per_alpha = estimate_memory_per_alpha(N1, N2, M, S, algorithm, dtype_bytes)

    # Also need memory for teacher model and masks (shared)
    shared_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)  # Always FP32

    available = device_info.available_memory_gb - shared_mem

    if available <= 0:
        print(f"[Warning] Very limited memory, using sequential processing")
        return 1

    max_parallel = int(available / mem_per_alpha)
    max_parallel = max(1, min(max_parallel, num_alphas))

    print(f"\n[Memory Analysis]")
    print(f"  Device: {device_info.device_type.upper()}")
    print(f"  Total memory: {device_info.total_memory_gb:.1f} GB")
    print(f"  Available: {device_info.available_memory_gb:.1f} GB")
    print(f"  Memory per alpha: {mem_per_alpha*1000:.1f} MB")
    print(f"  Optimal parallelism: {max_parallel} alphas")

    return max_parallel


# ============================================================
# Setup
# ============================================================
DEVICE_INFO = get_device_info()
DEVICE = DEVICE_INFO.device
COMPUTE_DTYPE = DEVICE_INFO.compute_dtype
USE_BF16 = DEVICE_INFO.use_bf16

RESULT_DIR = Path(__file__).parent / "result" / f"{N1}_{N2}_{M}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[Configuration]")
print(f"  Device: {DEVICE}")
print(f"  Algorithm: {ALGORITHM.upper()}")
print(f"  Matrix: {N1}×{N2}, M={M}")
print(f"  Results: {RESULT_DIR}")


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


# ------------------------------------------------------------
# Training Functions
# ------------------------------------------------------------
def train_batch_agd(Wt, Xt, alpha_values, steps, S, seed_for_init, lr=1e-2):
    """AGD training for a batch of alphas"""
    device = Wt.device
    N1, M = Wt.shape
    M_, N2 = Xt.shape
    num_alphas = len(alpha_values)
    alpha_scale = 1.0 / (M ** 0.5)

    # Generate masks
    all_masks = []
    all_C_values = []
    for alpha_tilde in alpha_values:
        i_idx, j_idx, C = sample_pairs_random_gpu(
            N1, N2, M, alpha_tilde, device,
            seed=SEED + int(alpha_tilde * 1000)
        )
        A_single = torch.zeros((N1, N2), dtype=torch.float32, device=device)
        if i_idx is not None and i_idx.numel() > 0:
            A_single[i_idx, j_idx] = 1.0
        A_alpha = A_single.unsqueeze(0).expand(S, -1, -1).contiguous()
        all_masks.append(A_alpha)
        all_C_values.append(C)

    A_all = torch.stack(all_masks, dim=0)

    # Initialize
    scale = 1.0 / (M ** 0.5)
    torch.manual_seed(seed_for_init)
    W_all = torch.randn((num_alphas, S, N1, M), device=device, dtype=torch.float32) * scale
    X_all = torch.randn((num_alphas, S, M, N2), device=device, dtype=torch.float32) * scale

    Y_teacher = Wt @ Xt
    Y_teacher_expanded = Y_teacher.unsqueeze(0).unsqueeze(0)

    # Training loop
    for _ in tqdm(range(steps), desc="AGD", leave=False, mininterval=0.5):
        with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
            Y_student = alpha_scale * torch.matmul(W_all, X_all)
            Mres = (Y_teacher_expanded - Y_student) * A_all
            grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X_all.transpose(-2, -1))

        W_all = W_all - lr * grad_W.float()

        with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
            Y_student2 = alpha_scale * torch.matmul(W_all, X_all)
            Mres2 = (Y_teacher_expanded - Y_student2) * A_all
            grad_X = -2.0 * alpha_scale * torch.matmul(W_all.transpose(-2, -1), Mres2)

        X_all = X_all - lr * grad_X.float()

    return W_all, X_all, A_all, all_C_values, Y_teacher


def train_batch_bigamp(Wt, Xt, alpha_values, steps, S, seed_for_init,
                       damping=0.5, noise_var=1e-6):
    """BiG-AMP training for a batch of alphas"""
    device = Wt.device
    N1, M = Wt.shape
    M_, N2 = Xt.shape
    num_alphas = len(alpha_values)
    alpha_scale = 1.0 / (M ** 0.5)

    # Generate masks
    all_masks = []
    all_C_values = []
    for alpha_tilde in alpha_values:
        i_idx, j_idx, C = sample_pairs_random_gpu(
            N1, N2, M, alpha_tilde, device,
            seed=SEED + int(alpha_tilde * 1000)
        )
        A_single = torch.zeros((N1, N2), dtype=torch.float32, device=device)
        if i_idx is not None and i_idx.numel() > 0:
            A_single[i_idx, j_idx] = 1.0
        A_alpha = A_single.unsqueeze(0).expand(S, -1, -1).contiguous()
        all_masks.append(A_alpha)
        all_C_values.append(C)

    A_all = torch.stack(all_masks, dim=0)

    # Initialize
    scale = 1.0 / (M ** 0.5)
    torch.manual_seed(seed_for_init)
    w_hat = torch.randn((num_alphas, S, N1, M), device=device, dtype=torch.float32) * scale
    x_hat = torch.randn((num_alphas, S, M, N2), device=device, dtype=torch.float32) * scale

    # Variances
    w_var = torch.ones((num_alphas, S, N1, M), device=device, dtype=torch.float32) * (1.0 / M)
    x_var = torch.ones((num_alphas, S, M, N2), device=device, dtype=torch.float32) * (1.0 / M)

    Y_teacher = Wt @ Xt
    Y_teacher_expanded = Y_teacher.unsqueeze(0).unsqueeze(0)

    # Training loop
    for _ in tqdm(range(steps), desc="BiG-AMP", leave=False, mininterval=0.5):
        # Forward
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (
            torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq)
        )
        V = torch.clamp(p_var + noise_var, min=1e-8)

        residual = (Y_teacher_expanded - z_hat) * A_all
        s = residual / V

        # Update W
        tau_W = (alpha_scale ** 2) * torch.matmul(A_all / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
        w_hat_new = w_hat + w_var_new * r_W

        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        # Update X
        z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq2 = w_hat ** 2
        p_var2 = (alpha_scale ** 2) * (
            torch.matmul(w_sq2, x_var) + torch.matmul(w_var, x_sq)
        )
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher_expanded - z_hat2) * A_all
        s2 = residual2 / V2

        tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A_all / V2)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new = 1.0 / (M + tau_X)
        r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
        x_hat_new = x_hat + x_var_new * r_X

        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = torch.clamp(damping * x_var + (1 - damping) * x_var_new, min=1e-8, max=1.0)

    return w_hat, x_hat, A_all, all_C_values, Y_teacher


def evaluate_results(W_all, X_all, A_all, all_C_values, Wt, Xt, Y_teacher,
                     alpha_values, S, steps):
    """Evaluate and aggregate results"""
    alpha_scale = 1.0 / (M ** 0.5)
    results = {}

    with torch.no_grad():
        for alpha_idx, alpha_tilde in enumerate(alpha_values):
            W_alpha = W_all[alpha_idx]
            X_alpha = X_all[alpha_idx]
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

                Y_final = alpha_scale * (W_s @ X_s)
                Rf = (Y_teacher - Y_final) * A_alpha[s]
                final_loss = float(torch.sum(Rf ** 2).item())

                trial_results.append({
                    'Q_W': Q_W, 'Q_X': Q_X,
                    'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
                    'Q_Y': Q_Y, 'Gen_Error': gen_error,
                    'Final_Loss': final_loss
                })

            # Aggregate
            def mean_std(x):
                x = np.array(x, dtype=float)
                return float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)

            qW = [t['Q_W'] for t in trial_results]
            qX = [t['Q_X'] for t in trial_results]
            qW_prime = [t['Q_W_prime'] for t in trial_results]
            qX_prime = [t['Q_X_prime'] for t in trial_results]
            qY = [t['Q_Y'] for t in trial_results]
            gen_err = [t['Gen_Error'] for t in trial_results]
            loss_list = [t['Final_Loss'] for t in trial_results]

            QW_mean, QW_std = mean_std(qW)
            QX_mean, QX_std = mean_std(qX)
            QW_prime_mean, QW_prime_std = mean_std(qW_prime)
            QX_prime_mean, QX_prime_std = mean_std(qX_prime)
            QY_mean, QY_std = mean_std(qY)
            GE_mean, GE_std = mean_std(gen_err)
            L_mean, L_std = mean_std(loss_list)

            aL_real = (C / (M * N1)) if (M * N1) > 0 else 0.0

            results[float(alpha_tilde)] = {
                'alpha_tilde_left': aL_real, 'C': int(C),
                'Q_W_mean': QW_mean, 'Q_W_std': QW_std,
                'Q_X_mean': QX_mean, 'Q_X_std': QX_std,
                'Q_W_prime_mean': QW_prime_mean, 'Q_W_prime_std': QW_prime_std,
                'Q_X_prime_mean': QX_prime_mean, 'Q_X_prime_std': QX_prime_std,
                'Q_Y_mean': QY_mean, 'Q_Y_std': QY_std,
                'Gen_Error_mean': GE_mean, 'Gen_Error_std': GE_std,
                'Loss_mean': L_mean, 'Loss_std': L_std,
                'epochs_mean': float(steps)
            }

    return results


# ------------------------------------------------------------
# Main Training Function with Smart Parallelism
# ------------------------------------------------------------
def run_experiment(algorithm: str = "bigamp",
                   user_parallelism: Optional[int] = None):
    """
    Run experiment with smart parallelism.

    Args:
        algorithm: "agd" or "bigamp"
        user_parallelism: Override automatic parallelism calculation
    """
    set_seed(SEED)
    Wt, Xt = create_teacher_dense(N1, N2, M, DEVICE, seed=SEED)

    alpha_values = np.arange(ALPHA_TILDE_START, ALPHA_TILDE_STOP + 1e-12, ALPHA_TILDE_STEP)
    num_alphas = len(alpha_values)

    # Calculate optimal parallelism
    parallel_alphas = calculate_optimal_parallelism(
        DEVICE_INFO, N1, N2, M, SAMPLES_PER_ALPHA,
        num_alphas, algorithm, user_parallelism
    )

    print(f"\n{'='*70}")
    print(f"SMART PARALLEL TRAINING - {algorithm.upper()}")
    print(f"{'='*70}")
    print(f"Alpha range: {ALPHA_TILDE_START} to {ALPHA_TILDE_STOP}")
    print(f"Total alphas: {num_alphas}")
    print(f"Parallel batch size: {parallel_alphas}")
    print(f"Number of batches: {(num_alphas + parallel_alphas - 1) // parallel_alphas}")
    print(f"Steps per alpha: {EPOCHS_PER_ALPHA}")
    print(f"{'='*70}\n")

    all_results = {}
    total_start = time.time()

    # Process in batches
    for batch_start in range(0, num_alphas, parallel_alphas):
        batch_end = min(batch_start + parallel_alphas, num_alphas)
        batch_alphas = alpha_values[batch_start:batch_end]

        print(f"\n[Batch {batch_start//parallel_alphas + 1}] "
              f"Processing alphas {batch_start+1}-{batch_end} of {num_alphas}")

        # Train
        if algorithm == "bigamp":
            W_all, X_all, A_all, C_values, Y_teacher = train_batch_bigamp(
                Wt, Xt, batch_alphas, EPOCHS_PER_ALPHA, SAMPLES_PER_ALPHA,
                SEED + 10000 + batch_start,
                damping=BIGAMP_DAMPING, noise_var=BIGAMP_NOISE_VAR
            )
        else:
            W_all, X_all, A_all, C_values, Y_teacher = train_batch_agd(
                Wt, Xt, batch_alphas, EPOCHS_PER_ALPHA, SAMPLES_PER_ALPHA,
                SEED + 10000 + batch_start,
                lr=AGD_LEARNING_RATE
            )

        # Evaluate
        batch_results = evaluate_results(
            W_all, X_all, A_all, C_values, Wt, Xt, Y_teacher,
            batch_alphas, SAMPLES_PER_ALPHA, EPOCHS_PER_ALPHA
        )
        all_results.update(batch_results)

        # Clear memory
        del W_all, X_all, A_all
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\n✓ Total time: {total_time:.2f}s")

    return all_results


# ------------------------------------------------------------
# Display and Plot
# ------------------------------------------------------------
def display_results(results_dict):
    items = sorted(results_dict.items(), key=lambda kv: kv[1]['alpha_tilde_left'])
    rows = []
    for _, r in items:
        rows.append({
            'alpha_L': f"{r['alpha_tilde_left']:.4f}",
            'Q_Y': f"{r['Q_Y_mean']:.4f}±{r['Q_Y_std']:.4f}",
            "Q_W'": f"{r['Q_W_prime_mean']:.4f}±{r['Q_W_prime_std']:.4f}",
            "Q_X'": f"{r['Q_X_prime_mean']:.4f}±{r['Q_X_prime_std']:.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    return df


def plot_results(results_dict, algorithm):
    items = sorted(results_dict.items(), key=lambda kv: kv[1]['alpha_tilde_left'])

    aL = np.array([r['alpha_tilde_left'] for _, r in items])
    qY = np.array([r['Q_Y_mean'] for _, r in items])
    qW = np.array([r['Q_W_prime_mean'] for _, r in items])
    qX = np.array([r['Q_X_prime_mean'] for _, r in items])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(aL, qY, 'D-', linewidth=2, markersize=4, color='#d62728', label='Q_Y')
    ax.plot(aL, qW, 'o-', linewidth=2, markersize=4, color='#9467bd', label="Q_W'")
    ax.plot(aL, qX, 'v-', linewidth=2, markersize=4, color='#2ca02c', label="Q_X'")

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax.set_ylabel('Overlap', fontsize=14)
    ax.set_title(f'{algorithm.upper()} Results ({N1}×{N2}, M={M})', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f"{algorithm}_{N1}x{N2}_M{M}_E{EPOCHS_PER_ALPHA}.png"
    save_path = RESULT_DIR / filename
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Smart Parallel Training')
    parser.add_argument('--algorithm', '-a', choices=['agd', 'bigamp'],
                        default=ALGORITHM, help='Algorithm to use')
    parser.add_argument('--parallel', '-p', type=int, default=None,
                        help='Override automatic parallelism')
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS_PER_ALPHA,
                        help='Training epochs')
    args = parser.parse_args()

    EPOCHS_PER_ALPHA = args.epochs

    results = run_experiment(args.algorithm, args.parallel)
    display_results(results)
    plot_results(results, args.algorithm)

    # Save results
    results_path = RESULT_DIR / f"{args.algorithm}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
