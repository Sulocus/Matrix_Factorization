"""
BiG-AMP Optimized - Final Production Version

Features:
1. BiG-AMP (Bilinear Generalized Approximate Message Passing) algorithm
2. Smart memory control with 3 modes:
   - parallel: Standard batched processing (N <= 10000)
   - optimized: No mask pre-storage, sequential alpha (N <= 20000)
   - extreme: FP16 storage + chunked computation (N <= 25000)
3. Intelligent parallelism based on available GPU memory
4. Complete evaluation metrics (Q_Y, Q_W, Q_X, Q_W', Q_X', Gen_Error)

Usage:
    python Main_bigamp_optimized.py                          # Default settings
    python Main_bigamp_optimized.py --n1 20000 --m 141       # Large matrix
    python Main_bigamp_optimized.py --memory-mode extreme    # Force extreme mode
    python Main_bigamp_optimized.py --steps 500              # Custom step count
"""

from pathlib import Path
import time
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataclasses import dataclass

# ============================================================
# Default Parameters
# ============================================================
N1 = 300
N2 = 300
M = 100

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 4
ALPHA_TILDE_STEP = 0.1

# BiG-AMP parameters
DAMPING = 0.5
NOISE_VAR = 1e-10
MAX_STEPS = 1000

SAMPLES_PER_ALPHA = 1
SEED = 42

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))

# Precision settings
USE_BF16 = DEVICE.type == 'cuda'
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Result directory
RESULT_DIR = Path(__file__).parent / "Result" / f"{N1}_{N2}_{M}"


# ============================================================
# Device Info
# ============================================================
@dataclass
class DeviceInfo:
    device_type: str
    available_memory_gb: float
    device_name: str


def get_device_info() -> DeviceInfo:
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            device_type='cuda',
            available_memory_gb=props.total_memory / (1024**3),
            device_name=props.name
        )
    elif DEVICE.type == 'mps':
        return DeviceInfo(
            device_type='mps',
            available_memory_gb=32.0,  # Approximate for Apple Silicon
            device_name='Apple Silicon'
        )
    else:
        return DeviceInfo(
            device_type='cpu',
            available_memory_gb=64.0,
            device_name='CPU'
        )


DEVICE_INFO = get_device_info()


# ============================================================
# Utility Functions
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_teacher(N1, N2, M, device, seed=42):
    """Create teacher model W_true and X_true"""
    torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale
    return W, X


def sample_mask(N1, N2, M, alpha, device, seed=None):
    """Generate random observation mask"""
    if seed is not None:
        torch.manual_seed(seed)

    # Expected number of observations per row/column
    c = alpha * M

    # Generate mask with probability c/N2 for each entry
    prob = min(c / N2, 1.0)
    mask = (torch.rand((N1, N2), device=device) < prob).float()

    return mask, c


# ============================================================
# Evaluation Metrics
# ============================================================
@torch.no_grad()
def gram_overlap_cosine(A, B, use_left=True):
    """Compute Gram matrix overlap using cosine similarity"""
    if use_left:
        G_A = A @ A.T
        G_B = B @ B.T
    else:
        G_A = A.T @ A
        G_B = B.T @ B

    G_A_flat = G_A.flatten()
    G_B_flat = G_B.flatten()

    dot = (G_A_flat * G_B_flat).sum()
    norm_A = G_A_flat.norm()
    norm_B = G_B_flat.norm()

    return float(dot / (norm_A * norm_B + 1e-12))


@torch.no_grad()
def gram_overlap_zero_to_one(A, B, use_left=True):
    """
    Compute normalized Gram overlap in [0, 1] range with baseline correction.

    Uses baseline b = m/(m+n+1) which is the expected cosine for random matrices.
    This ensures random initialization gives Q' ≈ 0, and perfect match gives Q' = 1.
    """
    q = gram_overlap_cosine(A, B, use_left)
    if use_left:
        n, m = A.shape
    else:
        n, m = A.shape[1], A.shape[0]
    b = m / (m + n + 1)  # baseline: expected cosine for random matrices
    qc = (q - b) / (1.0 - b + 1e-12)  # baseline correction
    return float(max(0.0, min(1.0, qc)))


# ============================================================
# Memory Management
# ============================================================
def estimate_memory_per_alpha(N1, N2, M, S, dtype_bytes=4):
    """Estimate GPU memory needed per alpha value for BiG-AMP

    BiG-AMP creates many N1×N2 tensors simultaneously in each step:
    - W update: z_hat, p_var, V, residual, s, A/V temp (6)
    - X update: z_hat2, p_var2, V2, residual2, s2, A/V2 temp (6)
    - Plus A_all mask (1)
    Total: ~13 N1×N2 tensors per alpha batch + safety margin = 16
    """
    student_params = 2 * (S * N1 * M + S * M * N2)  # w_hat, x_hat, w_var, x_var

    # CRITICAL: BiG-AMP needs ~13 N1×N2 tensors simultaneously, use 16 for safety
    intermediate = 16 * S * N1 * N2

    total_elements = student_params + intermediate
    return total_elements * dtype_bytes / (1024**3)


def select_memory_mode(N1, N2, M, S, num_alphas, mode_override='auto'):
    """Select optimal memory mode based on matrix size"""
    MAX_GPU_MEMORY_GB = min(DEVICE_INFO.available_memory_gb, 32.0)
    RESERVED_MEMORY_GB = 3.0
    effective_available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

    # Memory estimates
    per_alpha_mem = estimate_memory_per_alpha(N1, N2, M, S)
    teacher_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)
    single_mask_mem = N1 * N2 * 4 / (1024**3)

    print("\n[Memory Mode Selection]")
    print(f"  Matrix: {N1}x{N2}, M={M}, S={S}")
    print(f"  Available: {effective_available:.1f} GB")
    print(f"  Per-alpha training: {per_alpha_mem:.2f} GB")
    print(f"  Single mask: {single_mask_mem:.2f} GB")

    if mode_override != 'auto':
        print(f"  Mode override: {mode_override}")
        return mode_override

    # Calculate how many alphas can fit in parallel mode
    # Each batch needs: batch_masks + teacher + per_alpha_mem * batch_size
    # Solve for max batch_size: batch_size * (per_alpha_mem + single_mask_mem) + teacher_mem < available * 0.85
    usable_mem = effective_available * 0.85 - teacher_mem
    mem_per_batch_alpha = per_alpha_mem + single_mask_mem
    max_batch = max(1, int(usable_mem / mem_per_batch_alpha))

    if max_batch >= 2:
        mode = "parallel"
        print(f"  Selected: parallel (batch={min(max_batch, num_alphas)})")
    elif per_alpha_mem + single_mask_mem < effective_available * 0.8:
        mode = "optimized"
        print("  Selected: optimized (sequential, on-demand masks)")
    else:
        mode = "extreme"
        print("  Selected: extreme (FP16 + sequential)")

    return mode


def calculate_smart_parallelism(N1, N2, M, S, num_alphas):
    """Calculate optimal parallelism based on memory"""
    MAX_GPU_MEMORY_GB = min(DEVICE_INFO.available_memory_gb, 32.0)
    RESERVED_MEMORY_GB = 3.0
    available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

    per_alpha_mem = estimate_memory_per_alpha(N1, N2, M, S)
    teacher_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)
    single_mask_mem = N1 * N2 * 4 / (1024**3)

    # Each batch alpha needs: training memory + mask storage
    mem_per_batch_alpha = per_alpha_mem + single_mask_mem
    usable_mem = available * 0.85 - teacher_mem

    if mem_per_batch_alpha <= 0:
        return num_alphas

    max_parallel = max(1, min(int(usable_mem / mem_per_batch_alpha), num_alphas))
    return max_parallel


# ============================================================
# BiG-AMP Training - Parallel Mode
# ============================================================
def train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, alpha_values, steps, S,
                          damping=0.5, noise_var=1e-6):
    """BiG-AMP training with parallel alpha processing"""
    device = Wt.device
    N1, M = Wt.shape
    N2 = Xt.shape[1]
    num_alphas = len(alpha_values)
    alpha_scale = 1.0 / (M ** 0.5)
    scale = 1.0 / (M ** 0.5)

    # Initialize estimates
    w_hat = torch.randn((num_alphas, S, N1, M), device=device) * scale
    x_hat = torch.randn((num_alphas, S, M, N2), device=device) * scale
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    Y_teacher_exp = Y_teacher.unsqueeze(0).unsqueeze(0)

    for step in tqdm(range(steps), desc="BiG-AMP Training", leave=False, mininterval=1.0):
        # Forward pass
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher_exp - z_hat) * A_all
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
        p_var2 = (alpha_scale ** 2) * (torch.matmul(w_sq2, x_var) + torch.matmul(w_var, x_sq))
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher_exp - z_hat2) * A_all
        s2 = residual2 / V2

        tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A_all / V2)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new = 1.0 / (M + tau_X)
        r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
        x_hat_new = x_hat + x_var_new * r_X
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = torch.clamp(damping * x_var + (1 - damping) * x_var_new, min=1e-8, max=1.0)

    return w_hat, x_hat


# ============================================================
# BiG-AMP Training - Memory Optimized (Sequential)
# ============================================================
def train_bigamp_single(Wt, Xt, Y_teacher, alpha, steps, S, seed,
                        damping=0.5, noise_var=1e-6, use_fp16=False):
    """Memory-optimized BiG-AMP training for single alpha"""
    device = Wt.device
    N1, M = Wt.shape
    N2 = Xt.shape[1]
    alpha_scale = 1.0 / (M ** 0.5)
    scale = 1.0 / (M ** 0.5)

    # Generate mask on-demand
    A, _ = sample_mask(N1, N2, M, alpha, device, seed=seed)
    A = A.unsqueeze(0)

    # Initialize
    torch.manual_seed(seed + 10000)
    storage_dtype = torch.float16 if use_fp16 else torch.float32

    w_hat = (torch.randn((S, N1, M), device=device) * scale).to(storage_dtype)
    x_hat = (torch.randn((S, M, N2), device=device) * scale).to(storage_dtype)
    w_var = (torch.ones_like(w_hat) * (1.0 / M))
    x_var = (torch.ones_like(x_hat) * (1.0 / M))

    for _ in tqdm(range(steps), desc=f"BiG-AMP α={alpha:.2f}", leave=False, mininterval=1.0):
        w_f, x_f = w_hat.float(), x_hat.float()
        w_v, x_v = w_var.float(), x_var.float()

        # Forward
        z_hat = alpha_scale * torch.matmul(w_f, x_f)
        w_sq = w_f ** 2
        x_sq = x_f ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_v) + torch.matmul(w_v, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher - z_hat) * A
        s = residual / V

        # Update W
        tau_W = (alpha_scale ** 2) * torch.matmul(A / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.matmul(s, x_f.transpose(-2, -1))
        w_hat_new = w_f + w_var_new * r_W
        w_f = damping * w_f + (1 - damping) * w_hat_new
        w_v = torch.clamp(damping * w_v + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        w_hat = w_f.to(storage_dtype)
        w_var = w_v.to(storage_dtype)

        # Update X
        z_hat2 = alpha_scale * torch.matmul(w_f, x_f)
        w_sq2 = w_f ** 2
        p_var2 = (alpha_scale ** 2) * (torch.matmul(w_sq2, x_v) + torch.matmul(w_v, x_sq))
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher - z_hat2) * A
        s2 = residual2 / V2

        tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A / V2)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new = 1.0 / (M + tau_X)
        r_X = alpha_scale * torch.matmul(w_f.transpose(-2, -1), s2)
        x_hat_new = x_f + x_var_new * r_X
        x_f = damping * x_f + (1 - damping) * x_hat_new
        x_v = torch.clamp(damping * x_v + (1 - damping) * x_var_new, min=1e-8, max=1.0)

        x_hat = x_f.to(storage_dtype)
        x_var = x_v.to(storage_dtype)

    return w_hat.float(), x_hat.float()


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_batch(W, X, Wt, Xt, Y_teacher, alpha_values, S):
    """Evaluate metrics for all alphas"""
    results = {}

    for a_idx, alpha in enumerate(alpha_values):
        trial_results = []

        for s in range(S):
            W_s = W[a_idx, s] if W.dim() == 4 else W[s]
            X_s = X[a_idx, s] if X.dim() == 4 else X[s]

            Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
            Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
            Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
            Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

            Yp = W_s @ X_s
            Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                       (Y_teacher.norm() * Yp.norm() + 1e-12))
            gen_error = float(torch.mean((Y_teacher - Yp) ** 2))

            trial_results.append({
                'Q_W': Q_W, 'Q_X': Q_X,
                'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
                'Q_Y': Q_Y, 'Gen_Error': gen_error
            })

        # Aggregate
        metrics = {}
        for key in trial_results[0].keys():
            vals = [r[key] for r in trial_results]
            metrics[f'{key}_mean'] = float(np.mean(vals))
            metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        # Compute replica overlap when S > 1
        if S > 1:
            W_alpha = W[a_idx] if W.dim() == 4 else W  # (S, N1, M)
            X_alpha = X[a_idx] if X.dim() == 4 else X  # (S, M, N2)
            Q_W_rep, Q_W_rep_std, Q_X_rep, Q_X_rep_std = compute_replica_overlap(W_alpha, X_alpha)
            metrics['Q_W_replica_mean'] = Q_W_rep
            metrics['Q_W_replica_std'] = Q_W_rep_std
            metrics['Q_X_replica_mean'] = Q_X_rep
            metrics['Q_X_replica_std'] = Q_X_rep_std
        else:
            metrics['Q_W_replica_mean'] = 0.0
            metrics['Q_W_replica_std'] = 0.0
            metrics['Q_X_replica_mean'] = 0.0
            metrics['Q_X_replica_std'] = 0.0

        results[float(alpha)] = metrics

    return results


@torch.no_grad()
def evaluate_single(W, X, Wt, Xt, Y_teacher, S):
    """Evaluate single alpha result"""
    trial_results = []

    for s in range(S):
        W_s, X_s = W[s], X[s]
        Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
        Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
        Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
        Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

        Yp = W_s @ X_s
        Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                   (Y_teacher.norm() * Yp.norm() + 1e-12))
        gen_error = float(torch.mean((Y_teacher - Yp) ** 2))

        trial_results.append({
            'Q_W': Q_W, 'Q_X': Q_X,
            'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
            'Q_Y': Q_Y, 'Gen_Error': gen_error
        })

    metrics = {}
    for key in trial_results[0].keys():
        vals = [r[key] for r in trial_results]
        metrics[f'{key}_mean'] = float(np.mean(vals))
        metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    # Compute replica overlap when S > 1
    if S > 1:
        Q_W_rep, Q_W_rep_std, Q_X_rep, Q_X_rep_std = compute_replica_overlap(W, X)
        metrics['Q_W_replica_mean'] = Q_W_rep
        metrics['Q_W_replica_std'] = Q_W_rep_std
        metrics['Q_X_replica_mean'] = Q_X_rep
        metrics['Q_X_replica_std'] = Q_X_rep_std
    else:
        metrics['Q_W_replica_mean'] = 0.0
        metrics['Q_W_replica_std'] = 0.0
        metrics['Q_X_replica_mean'] = 0.0
        metrics['Q_X_replica_std'] = 0.0

    return metrics


@torch.no_grad()
def compute_replica_overlap(W_all, X_all):
    """
    Compute pairwise Gram overlap between S replicas (no normalization).

    Args:
        W_all: (S, N1, M) - S replicas of W
        X_all: (S, M, N2) - S replicas of X

    Returns:
        Q_W_mean, Q_W_std, Q_X_mean, Q_X_std (raw cosine values)
    """
    S = W_all.shape[0]
    if S < 2:
        return 0.0, 0.0, 0.0, 0.0

    Q_W_list, Q_X_list = [], []

    for i in range(S):
        for j in range(i + 1, S):
            Q_W_list.append(gram_overlap_cosine(W_all[i], W_all[j], use_left=True))
            Q_X_list.append(gram_overlap_cosine(X_all[i], X_all[j], use_left=False))

    Q_W_mean = float(np.mean(Q_W_list))
    Q_W_std = float(np.std(Q_W_list, ddof=1)) if len(Q_W_list) > 1 else 0.0
    Q_X_mean = float(np.mean(Q_X_list))
    Q_X_std = float(np.std(Q_X_list, ddof=1)) if len(Q_X_list) > 1 else 0.0

    return Q_W_mean, Q_W_std, Q_X_mean, Q_X_std


# ============================================================
# Main Training Functions
# ============================================================
def run_parallel_mode(alpha_values, steps, S):
    """Run with parallel alpha processing"""
    set_seed(SEED)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)
    max_parallel = calculate_smart_parallelism(N1, N2, M, S, num_alphas)

    print(f"\n{'='*70}")
    print("BiG-AMP TRAINING - PARALLEL MODE")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({num_alphas} points)")
    print(f"Steps: {steps}, Samples: {S}")
    print(f"Parallelism: {max_parallel} alphas")
    print(f"{'='*70}\n")

    all_results = {}
    total_start = time.time()

    # Process in batches
    for batch_start in range(0, num_alphas, max_parallel):
        batch_end = min(batch_start + max_parallel, num_alphas)
        batch_alphas = alpha_values[batch_start:batch_end]
        batch_size = len(batch_alphas)

        print(f"[Batch {batch_start//max_parallel + 1}] Alpha {batch_alphas[0]:.2f} - {batch_alphas[-1]:.2f}")

        # Pre-generate masks
        A_all = torch.zeros((batch_size, 1, N1, N2), device=DEVICE)
        for i, alpha in enumerate(batch_alphas):
            mask_seed = SEED + int(alpha * 1000)
            A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
            A_all[i, 0] = A

        # Train
        W, X = train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, batch_alphas, steps, S,
                                      damping=DAMPING, noise_var=NOISE_VAR)

        # Evaluate
        batch_results = evaluate_batch(W, X, Wt, Xt, Y_teacher, batch_alphas, S)
        all_results.update(batch_results)

        del A_all, W, X
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")

    return all_results, total_time


def run_sequential_mode(alpha_values, steps, S, use_fp16=False):
    """Run with sequential alpha processing (memory optimized)"""
    set_seed(SEED)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)
    mode_name = "EXTREME" if use_fp16 else "OPTIMIZED"

    print(f"\n{'='*70}")
    print(f"BiG-AMP TRAINING - {mode_name} MODE (Sequential)")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({num_alphas} points)")
    print(f"Steps: {steps}, Samples: {S}")
    print(f"FP16 storage: {use_fp16}")
    print(f"{'='*70}\n")

    all_results = {}
    total_start = time.time()

    for i, alpha in enumerate(alpha_values):
        alpha_seed = SEED + int(alpha * 1000)
        print(f"[{i+1}/{num_alphas}] Alpha = {alpha:.2f}")

        W, X = train_bigamp_single(Wt, Xt, Y_teacher, alpha, steps, S, alpha_seed,
                                    damping=DAMPING, noise_var=NOISE_VAR, use_fp16=use_fp16)

        metrics = evaluate_single(W, X, Wt, Xt, Y_teacher, S)
        all_results[float(alpha)] = metrics

        del W, X
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")

    return all_results, total_time


# ============================================================
# Visualization
# ============================================================
def plot_results(results, alpha_values, save_path):
    """
    Generate result plots (Main.py style combined chart + parameter table)

    Creates a combined figure with:
    - Upper part: Q_Y, Q_W', Q_X' curves
    - Lower part: Parameter table
    """
    # Extract data
    aL = np.array(alpha_values)
    qY_mu = np.array([results[a]['Q_Y_mean'] for a in alpha_values])
    qW_prime_mu = np.array([results[a]['Q_W_prime_mean'] for a in alpha_values])
    qX_prime_mu = np.array([results[a]['Q_X_prime_mean'] for a in alpha_values])

    # ============================================================
    # Combined chart + parameter table (Two-tier layout)
    # ============================================================
    fig_combined = plt.figure(figsize=(10, 10))

    # Upper half: Combined metrics plot
    ax_plot = plt.subplot2grid((3, 1), (0, 0), rowspan=2, fig=fig_combined)

    # Three lines, thinner lines, no variance bands
    ax_plot.plot(aL, qY_mu, marker='D', linewidth=1.5, markersize=5,
                 color='#d62728', label='Q_Y (invariant)', zorder=3)
    ax_plot.plot(aL, qW_prime_mu, marker='o', linewidth=1.5, markersize=5,
                 color='#9467bd', label="Q_W' (zero-to-one)", zorder=2)
    ax_plot.plot(aL, qX_prime_mu, marker='v', linewidth=1.5, markersize=5,
                 color='#8c564b', label="Q_X' (zero-to-one)", zorder=1)

    ax_plot.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=13)
    ax_plot.set_ylabel('Overlap Metrics', fontsize=13)
    ax_plot.set_title('Combined Metrics: Q_Y and Zero-to-One Method', fontsize=14, fontweight='bold')
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_plot.legend(fontsize=11, loc='lower right')

    # Lower half: Parameter table
    ax_table = plt.subplot2grid((3, 1), (2, 0), fig=fig_combined)
    ax_table.axis('off')

    # Prepare parameter table data
    table_data = [
        ['Model Parameters', f'N1={N1}, N2={N2}, M={M}'],
        ['Algorithm', 'BiG-AMP'],
        ['Damping', f'{DAMPING}'],
        ['Noise Variance', f'{NOISE_VAR}'],
        ['Steps', f'{MAX_STEPS}'],
        ['Samples per Alpha', f'{SAMPLES_PER_ALPHA}'],
    ]

    # Create table
    table = ax_table.table(cellText=table_data,
                          colWidths=[0.35, 0.65],
                          cellLoc='left',
                          loc='center',
                          bbox=(0, 0, 1, 1))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Set table style
    for i in range(len(table_data)):
        table[(i, 0)].set_facecolor('#e6f2ff')
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_facecolor('#f0f0f0')

    plt.tight_layout()

    # Save chart
    fig_combined.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close(fig_combined)


def plot_replica_comparison(results, alpha_values, save_path):
    """
    Generate replica comparison plot

    Compares:
    - Q_W', Q_X' (teacher vs student, normalized to 0-1)
    - Q_W_replica, Q_X_replica (replica vs replica, raw cosine)
    """
    # Extract data
    aL = np.array(alpha_values)
    qW_prime_mu = np.array([results[a]['Q_W_prime_mean'] for a in alpha_values])
    qX_prime_mu = np.array([results[a]['Q_X_prime_mean'] for a in alpha_values])
    qW_rep_mu = np.array([results[a].get('Q_W_replica_mean', 0.0) for a in alpha_values])
    qX_rep_mu = np.array([results[a].get('Q_X_replica_mean', 0.0) for a in alpha_values])

    # Check if replica data exists
    has_replica = np.any(qW_rep_mu != 0) or np.any(qX_rep_mu != 0)
    if not has_replica:
        print("No replica overlap data (S=1), skipping replica comparison plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Upper plot: Q_W comparison
    ax1.plot(aL, qW_prime_mu, marker='o', linewidth=1.5, markersize=5,
             color='#9467bd', label="Q_W' (teacher-student, normalized)", zorder=2)
    ax1.plot(aL, qW_rep_mu, marker='s', linewidth=1.5, markersize=5,
             color='#2ca02c', label="Q_W_replica (replica-replica, raw cosine)", zorder=1)

    ax1.set_ylabel('W Overlap', fontsize=13)
    ax1.set_title('W Overlap Comparison: Teacher-Student vs Replica-Replica', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=10, loc='lower right')

    # Lower plot: Q_X comparison
    ax2.plot(aL, qX_prime_mu, marker='v', linewidth=1.5, markersize=5,
             color='#8c564b', label="Q_X' (teacher-student, normalized)", zorder=2)
    ax2.plot(aL, qX_rep_mu, marker='^', linewidth=1.5, markersize=5,
             color='#17becf', label="Q_X_replica (replica-replica, raw cosine)", zorder=1)

    ax2.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=13)
    ax2.set_ylabel('X Overlap', fontsize=13)
    ax2.set_title('X Overlap Comparison: Teacher-Student vs Replica-Replica', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Replica comparison plot saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    global N1, N2, M, ALPHA_TILDE_STEP, MAX_STEPS, SAMPLES_PER_ALPHA, RESULT_DIR

    parser = argparse.ArgumentParser(description='BiG-AMP Optimized Training')
    parser.add_argument('--n1', type=int, default=N1, help='Matrix N1 dimension')
    parser.add_argument('--n2', type=int, default=None, help='Matrix N2 dimension (default: same as N1)')
    parser.add_argument('--m', type=int, default=M, help='Latent dimension M')
    parser.add_argument('--steps', type=int, default=MAX_STEPS, help='BiG-AMP steps')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_ALPHA, help='Samples per alpha')
    parser.add_argument('--alpha-step', type=float, default=ALPHA_TILDE_STEP, help='Alpha step size')
    parser.add_argument('--alpha-stop', type=float, default=ALPHA_TILDE_STOP, help='Alpha max value')
    parser.add_argument('--memory-mode', type=str, default='auto',
                        choices=['auto', 'parallel', 'optimized', 'extreme'],
                        help='Memory mode')
    parser.add_argument('--damping', type=float, default=DAMPING, help='BiG-AMP damping factor')
    args = parser.parse_args()

    # Apply args
    N1 = args.n1
    N2 = args.n2 if args.n2 else args.n1
    M = args.m
    MAX_STEPS = args.steps
    SAMPLES_PER_ALPHA = args.samples
    ALPHA_TILDE_STEP = args.alpha_step

    RESULT_DIR = Path(__file__).parent / "Result" / f"{N1}_{N2}_{M}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    alpha_values = np.arange(ALPHA_TILDE_START, args.alpha_stop + 1e-12, ALPHA_TILDE_STEP)

    # Select mode
    mode = select_memory_mode(N1, N2, M, SAMPLES_PER_ALPHA, len(alpha_values), args.memory_mode)

    # Run training
    if mode == 'parallel':
        results, total_time = run_parallel_mode(alpha_values, MAX_STEPS, SAMPLES_PER_ALPHA)
    elif mode == 'optimized':
        results, total_time = run_sequential_mode(alpha_values, MAX_STEPS, SAMPLES_PER_ALPHA, use_fp16=False)
    else:  # extreme
        results, total_time = run_sequential_mode(alpha_values, MAX_STEPS, SAMPLES_PER_ALPHA, use_fp16=True)

    # Save results
    results_data = {
        'config': {
            'N1': N1, 'N2': N2, 'M': M,
            'steps': MAX_STEPS,
            'samples_per_alpha': SAMPLES_PER_ALPHA,
            'damping': DAMPING,
            'noise_var': NOISE_VAR,
            'mode': mode,
            'total_time': total_time
        },
        'alpha_values': [float(a) for a in alpha_values],
        'results': {str(k): v for k, v in results.items()}
    }

    results_path = RESULT_DIR / f'bigamp_results_steps{MAX_STEPS}.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved: {results_path}")

    # Plot 1: Teacher-student (Main.py style: Q_Y + Q_W' + Q_X' with parameter table)
    plot_path1 = RESULT_DIR / f'bigamp_teacher_student_steps{MAX_STEPS}.png'
    plot_results(results, [float(a) for a in alpha_values], plot_path1)

    # Plot 2: Replica comparison (Q_W'/Q_X' vs Q_W_replica/Q_X_replica)
    plot_path2 = RESULT_DIR / f'bigamp_replica_comparison_steps{MAX_STEPS}.png'
    plot_replica_comparison(results, [float(a) for a in alpha_values], plot_path2)

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Mode: {mode}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results: {results_path}")
    print(f"Plot 1: {plot_path1}")
    print(f"Plot 2: {plot_path2}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
