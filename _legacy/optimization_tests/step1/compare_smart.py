#!/usr/bin/env python3
"""
Smart Comparison: BiG-AMP vs AGD with Intelligent Parallelism

Features:
1. Auto-detect device memory and optimal parallelism
2. Batched processing for large matrices (10000x10000)
3. AGD runs at max epochs only (baseline reference)
4. BiG-AMP runs at multiple step counts
5. High-contrast visualization
6. Memory-optimized mode for ultra-large matrices (N > 15000)

Memory Modes:
- parallel: Standard batched processing (N <= 10000)
- optimized: No mask pre-storage, sequential alpha (N <= 23000)
- extreme: FP16 storage + chunked Y computation (N <= 28000)
- auto: Automatically select based on memory requirements
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
from typing import Optional, List, Dict

# ============================================================
# Configuration
# ============================================================
N1 = 10000
N2 = 10000
M = 100

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 2
ALPHA_TILDE_STEP = 0.1

SAMPLES_PER_ALPHA = 5
SEED = 42

# ============================================================
# Device Detection
# ============================================================
@dataclass
class DeviceInfo:
    device: torch.device
    device_type: str
    total_memory_gb: float
    available_memory_gb: float
    compute_dtype: torch.dtype
    use_bf16: bool


def get_device_info() -> DeviceInfo:
    """Detect device and available memory"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_memory
        reserved_mem = torch.cuda.memory_reserved(0)
        total_gb = total_mem / (1024**3)
        available_gb = (total_mem - reserved_mem) * 0.8 / (1024**3)
        use_bf16 = True
        compute_dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'
        import subprocess
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                   capture_output=True, text=True)
            total_mem = int(result.stdout.strip())
            total_gb = total_mem / (1024**3)
            available_gb = total_gb * 0.3
        except:
            total_gb = 16.0
            available_gb = 4.0
        use_bf16 = False
        compute_dtype = torch.float32
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        import psutil
        total_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3) * 0.5
        use_bf16 = False
        compute_dtype = torch.float32

    return DeviceInfo(device, device_type, total_gb, available_gb, compute_dtype, use_bf16)


DEVICE_INFO = get_device_info()
DEVICE = DEVICE_INFO.device
COMPUTE_DTYPE = DEVICE_INFO.compute_dtype
USE_BF16 = DEVICE_INFO.use_bf16

RESULT_DIR = Path(__file__).parent / "result" / f"{N1}_{N2}_{M}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_memory_per_alpha(N1, N2, M, S, algorithm, dtype_bytes=4):
    """Estimate GPU memory per alpha in GB - CORRECTED VERSION

    For large matrices, the N1×N2 tensors dominate:
    - Mask A: (S, N1, N2)
    - Y_student: (S, N1, N2)
    - Residual: (S, N1, N2)
    - Mres: (S, N1, N2)
    """
    W_size = S * N1 * M
    X_size = S * M * N2

    # CRITICAL: N1×N2 tensors are the memory bottleneck for large matrices
    NxN_size = S * N1 * N2  # This is huge for 10000×10000!

    # AGD needs: W, X, A (mask), Y_student, Mres, grad_W, grad_X
    # Y_student, Mres are both (S, N1, N2)
    agd_elements = (
        W_size +          # W
        X_size +          # X
        NxN_size +        # Mask A
        NxN_size +        # Y_student
        NxN_size +        # Mres (residual * mask)
        W_size +          # grad_W
        X_size            # grad_X
    )

    # BiG-AMP needs additional variance tensors + more N×N intermediates
    bigamp_elements = agd_elements + (
        W_size +          # w_var
        X_size +          # x_var
        NxN_size +        # p_var
        NxN_size +        # V
        NxN_size          # s (scaled residual)
    )

    total_elements = bigamp_elements if algorithm == "bigamp" else agd_elements

    # Add 100% overhead for intermediate computations and PyTorch fragmentation
    total_elements *= 2.0

    return total_elements * dtype_bytes / (1024**3)


def calculate_optimal_parallelism(N1, N2, M, S, num_alphas, algorithm):
    """Calculate optimal parallelism based on available memory

    CRITICAL: Must account for:
    1. Pre-stored masks: num_alphas × N1 × N2 × 4 bytes (FP32)
    2. Per-alpha batch memory
    3. Safety margin (reserve 3GB)

    NOTE: ALWAYS use FP32 (4 bytes) for estimation because:
    - W, X tensors are stored in FP32
    - autocast only affects computation, not storage
    - Many intermediate tensors remain FP32
    """
    # Hard limits for GPU memory
    MAX_GPU_MEMORY_GB = 28.0  # 5090 upper limit (32GB total, reserve some)
    RESERVED_MEMORY_GB = 3.0  # Safety margin

    effective_available = min(DEVICE_INFO.available_memory_gb, MAX_GPU_MEMORY_GB) - RESERVED_MEMORY_GB

    # ALWAYS use FP32 (4 bytes) - autocast doesn't reduce storage memory!
    dtype_bytes = 4  # FP32 always
    mem_per_alpha = estimate_memory_per_alpha(N1, N2, M, S, algorithm, dtype_bytes)

    # Shared memory: Teacher (Wt, Xt, Y_teacher) + ALL pre-stored masks
    teacher_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)  # FP32
    masks_mem = num_alphas * N1 * N2 * 4 / (1024**3)  # All masks pre-stored
    shared_mem = teacher_mem + masks_mem

    available = effective_available - shared_mem

    print(f"  [Memory Debug]")
    print(f"    Effective limit: {effective_available:.1f} GB")
    print(f"    Teacher memory: {teacher_mem:.2f} GB")
    print(f"    All masks memory: {masks_mem:.2f} GB")
    print(f"    Shared memory: {shared_mem:.2f} GB")
    print(f"    Available for batches: {available:.2f} GB")
    print(f"    Per-alpha estimate: {mem_per_alpha:.2f} GB")

    if available <= 0:
        print(f"    WARNING: Not enough memory!")
        return 1

    max_parallel = max(1, min(int(available / mem_per_alpha), num_alphas))
    print(f"    Calculated parallelism: {max_parallel}")

    return max_parallel


def select_memory_mode(N1, N2, M, S, num_alphas, mode_override='auto'):
    """Select optimal memory mode based on matrix size and available memory

    Modes:
    - parallel: Standard batched processing, pre-store all masks
    - optimized: No mask pre-storage, sequential alpha processing
    - extreme: FP16 storage + chunked computation

    Returns: (mode, estimated_max_parallel)
    """
    MAX_GPU_MEMORY_GB = 28.0
    RESERVED_MEMORY_GB = 3.0
    effective_available = min(DEVICE_INFO.available_memory_gb, MAX_GPU_MEMORY_GB) - RESERVED_MEMORY_GB

    # Memory estimates
    masks_mem = num_alphas * N1 * N2 * 4 / (1024**3)
    teacher_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)
    per_alpha_mem = estimate_memory_per_alpha(N1, N2, M, S, "agd", 4)

    print(f"\n[Memory Mode Selection]")
    print(f"  Matrix: {N1}x{N2}, M={M}")
    print(f"  Available: {effective_available:.1f} GB")
    print(f"  All masks: {masks_mem:.2f} GB")
    print(f"  Per-alpha: {per_alpha_mem:.2f} GB")

    if mode_override != 'auto':
        print(f"  Mode override: {mode_override}")
        return mode_override

    # Auto-select based on memory requirements
    total_parallel_mem = masks_mem + teacher_mem + per_alpha_mem
    total_optimized_mem = teacher_mem + per_alpha_mem  # No mask pre-storage
    total_extreme_mem = teacher_mem + per_alpha_mem * 0.5  # FP16 + chunked

    if total_parallel_mem < effective_available * 0.9:
        mode = "parallel"
        print(f"  Selected: parallel (fits in memory)")
    elif total_optimized_mem < effective_available * 0.9:
        mode = "optimized"
        print(f"  Selected: optimized (no mask pre-storage)")
    elif total_extreme_mem < effective_available * 0.9:
        mode = "extreme"
        print(f"  Selected: extreme (FP16 + chunked)")
    else:
        mode = "extreme"
        print(f"  WARNING: Matrix too large! Using extreme mode anyway.")

    return mode


# ============================================================
# Memory-Optimized Training Functions
# ============================================================
def train_single_agd_memory_optimized(Wt, Xt, Y_teacher, alpha, steps, S, seed,
                                       lr=1e-2, use_fp16=False, chunk_size=None):
    """Memory-optimized AGD training for a single alpha

    Optimizations:
    1. Generate mask on-demand (not pre-stored)
    2. Optional FP16 storage for W, X
    3. Optional chunked Y computation
    """
    device = Wt.device
    N1, M = Wt.shape
    N2 = Xt.shape[1]
    alpha_scale = 1.0 / (M ** 0.5)

    # Generate mask on-demand
    A, _ = sample_mask(N1, N2, M, alpha, device, seed=seed)
    A = A.unsqueeze(0)  # (1, N1, N2) for broadcasting

    # Initialize W, X
    torch.manual_seed(seed + 10000)
    scale = 1.0 / (M ** 0.5)
    storage_dtype = torch.float16 if use_fp16 else torch.float32

    W = (torch.randn((S, N1, M), device=device) * scale).to(storage_dtype)
    X = (torch.randn((S, M, N2), device=device) * scale).to(storage_dtype)

    if chunk_size is None or chunk_size >= N1:
        # Standard training (no chunking)
        for _ in tqdm(range(steps), desc=f"AGD α={alpha:.2f}", leave=False, mininterval=1.0):
            W_f, X_f = W.float(), X.float()

            with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
                Y_student = alpha_scale * torch.matmul(W_f, X_f)
                Mres = (Y_teacher - Y_student) * A
                grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X_f.transpose(-2, -1))
            W = (W_f - lr * grad_W.float()).to(storage_dtype)

            W_f = W.float()
            with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
                Y_student2 = alpha_scale * torch.matmul(W_f, X_f)
                Mres2 = (Y_teacher - Y_student2) * A
                grad_X = -2.0 * alpha_scale * torch.matmul(W_f.transpose(-2, -1), Mres2)
            X = (X_f - lr * grad_X.float()).to(storage_dtype)
    else:
        # Chunked training (extreme mode)
        for _ in tqdm(range(steps), desc=f"AGD α={alpha:.2f} (chunked)", leave=False, mininterval=1.0):
            W_f, X_f = W.float(), X.float()

            # Compute grad_W in chunks
            grad_W = torch.zeros_like(W_f)
            for i in range(0, N1, chunk_size):
                i_end = min(i + chunk_size, N1)
                W_chunk = W_f[:, i:i_end]  # (S, chunk, M)
                A_chunk = A[:, i:i_end]  # (1, chunk, N2)
                Y_t_chunk = Y_teacher[i:i_end]  # (chunk, N2)

                Y_chunk = alpha_scale * (W_chunk @ X_f)  # (S, chunk, N2)
                Mres_chunk = (Y_t_chunk - Y_chunk) * A_chunk
                grad_W[:, i:i_end] = -2.0 * alpha_scale * (Mres_chunk @ X_f.transpose(-2, -1))

            W = (W_f - lr * grad_W).to(storage_dtype)

            # Compute grad_X in chunks (along N2)
            W_f = W.float()
            grad_X = torch.zeros_like(X_f)
            for j in range(0, N2, chunk_size):
                j_end = min(j + chunk_size, N2)
                X_chunk = X_f[:, :, j:j_end]  # (S, M, chunk)
                A_chunk = A[:, :, j:j_end]  # (1, N1, chunk)
                Y_t_chunk = Y_teacher[:, j:j_end]  # (N1, chunk)

                Y_chunk = alpha_scale * (W_f @ X_chunk)  # (S, N1, chunk)
                Mres_chunk = (Y_t_chunk - Y_chunk) * A_chunk
                grad_X[:, :, j:j_end] = -2.0 * alpha_scale * (W_f.transpose(-2, -1) @ Mres_chunk)

            X = (X_f - lr * grad_X).to(storage_dtype)

    return W.float(), X.float()


def train_single_bigamp_memory_optimized(Wt, Xt, Y_teacher, alpha, steps, S, seed,
                                          damping=0.5, noise_var=1e-6,
                                          use_fp16=False, chunk_size=None):
    """Memory-optimized BiG-AMP training for a single alpha"""
    device = Wt.device
    N1, M = Wt.shape
    N2 = Xt.shape[1]
    alpha_scale = 1.0 / (M ** 0.5)

    # Generate mask on-demand
    A, _ = sample_mask(N1, N2, M, alpha, device, seed=seed)
    A = A.unsqueeze(0)  # (1, N1, N2)

    # Initialize
    torch.manual_seed(seed + 10000)
    scale = 1.0 / (M ** 0.5)
    storage_dtype = torch.float16 if use_fp16 else torch.float32

    w_hat = (torch.randn((S, N1, M), device=device) * scale).to(storage_dtype)
    x_hat = (torch.randn((S, M, N2), device=device) * scale).to(storage_dtype)
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    # Standard BiG-AMP (no chunking for now - BiG-AMP is complex)
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


@torch.no_grad()
def evaluate_single(W, X, Wt, Xt, Y_teacher, alpha, S):
    """Evaluate metrics for a single alpha (memory-optimized version)"""
    results = []

    for s in range(S):
        W_s, X_s = W[s], X[s]
        Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
        Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
        Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
        Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

        Yp = W_s @ X_s
        Q_Y = float(((Y_teacher.flatten() * Yp.flatten()).sum()) /
                   (Y_teacher.norm() * Yp.norm() + 1e-12))
        gen_error = float(torch.mean((Y_teacher - Yp) ** 2).item())

        results.append({
            'Q_W': Q_W, 'Q_X': Q_X,
            'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
            'Q_Y': Q_Y, 'Gen_Error': gen_error
        })

    metrics = {}
    for key in results[0].keys():
        vals = [r[key] for r in results]
        metrics[f'{key}_mean'] = float(np.mean(vals))
        metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    return metrics


# ============================================================
# Utility Functions
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@torch.no_grad()
def create_teacher(N1, N2, M, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W_true = torch.randn(N1, M, device=device, dtype=torch.float32) * scale
    X_true = torch.randn(M, N2, device=device, dtype=torch.float32) * scale
    return W_true, X_true


@torch.no_grad()
def sample_mask(N1, N2, M, alpha_tilde, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    C = int(round(alpha_tilde * M * N1))
    total_pairs = N1 * N2
    if C <= 0:
        return torch.zeros((N1, N2), dtype=torch.float32, device=device), 0
    if C >= total_pairs:
        return torch.ones((N1, N2), dtype=torch.float32, device=device), total_pairs
    idx = torch.randperm(total_pairs, device=device)[:C]
    A = torch.zeros((N1, N2), dtype=torch.float32, device=device)
    A.view(-1)[idx] = 1.0
    return A, C


@torch.no_grad()
def gram_overlap_cosine(A, B, use_left=True):
    GA = A @ A.T if use_left else A.T @ A
    GB = B @ B.T if use_left else B.T @ B
    num = (GA * GB).sum()
    den = GA.norm() * GB.norm() + 1e-12
    return float((num / den).item())


@torch.no_grad()
def gram_overlap_zero_to_one(A, B, use_left=True):
    q = gram_overlap_cosine(A, B, use_left=use_left)
    n, m = A.shape if use_left else (A.shape[1], A.shape[0])
    b = m / (m + n + 1)
    qc = (q - b) / (1.0 - b + 1e-12)
    return float(max(0.0, min(1.0, qc)))


# ============================================================
# Training Functions with Batching
# ============================================================
def train_batch_agd(Wt, Xt, Y_teacher, batch_alphas, A_batch, steps, S, lr=1e-2):
    """Train AGD for a batch of alphas"""
    device = Wt.device
    N1, M = Wt.shape
    N2 = Xt.shape[1]
    num_alphas = len(batch_alphas)
    alpha_scale = 1.0 / (M ** 0.5)

    scale = 1.0 / (M ** 0.5)
    W = torch.randn((num_alphas, S, N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((num_alphas, S, M, N2), device=device, dtype=torch.float32) * scale

    Y_exp = Y_teacher.unsqueeze(0).unsqueeze(0)

    for _ in tqdm(range(steps), desc="AGD", leave=False, mininterval=1.0):
        with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
            Y_student = alpha_scale * torch.matmul(W, X)
            Mres = (Y_exp - Y_student) * A_batch
            grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X.transpose(-2, -1))
        W = W - lr * grad_W.float()

        with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
            Y_student2 = alpha_scale * torch.matmul(W, X)
            Mres2 = (Y_exp - Y_student2) * A_batch
            grad_X = -2.0 * alpha_scale * torch.matmul(W.transpose(-2, -1), Mres2)
        X = X - lr * grad_X.float()

    return W, X


def train_batch_bigamp(Wt, Xt, Y_teacher, batch_alphas, A_batch, steps, S,
                       damping=0.5, noise_var=1e-6):
    """Train BiG-AMP for a batch of alphas"""
    device = Wt.device
    N1, M = Wt.shape
    N2 = Xt.shape[1]
    num_alphas = len(batch_alphas)
    alpha_scale = 1.0 / (M ** 0.5)

    scale = 1.0 / (M ** 0.5)
    w_hat = torch.randn((num_alphas, S, N1, M), device=device, dtype=torch.float32) * scale
    x_hat = torch.randn((num_alphas, S, M, N2), device=device, dtype=torch.float32) * scale

    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    Y_exp = Y_teacher.unsqueeze(0).unsqueeze(0)

    for _ in tqdm(range(steps), desc="BiG-AMP", leave=False, mininterval=1.0):
        # Forward
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_exp - z_hat) * A_batch
        s = residual / V

        # Update W
        tau_W = (alpha_scale ** 2) * torch.matmul(A_batch / V, x_sq.transpose(-2, -1))
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
        residual2 = (Y_exp - z_hat2) * A_batch
        s2 = residual2 / V2

        tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A_batch / V2)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new = 1.0 / (M + tau_X)
        r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
        x_hat_new = x_hat + x_var_new * r_X
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = torch.clamp(damping * x_var + (1 - damping) * x_var_new, min=1e-8, max=1.0)

    return w_hat, x_hat


@torch.no_grad()
def evaluate_batch(W, X, Wt, Xt, Y_teacher, batch_alphas, A_batch, S):
    """Evaluate metrics for a batch"""
    alpha_scale = 1.0 / (M ** 0.5)
    results = {}

    for i, alpha in enumerate(batch_alphas):
        trial_results = []
        for s in range(S):
            W_s, X_s = W[i, s], X[i, s]
            Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
            Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
            Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
            Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

            Yp = W_s @ X_s
            Q_Y = float(((Y_teacher.flatten() * Yp.flatten()).sum()) /
                       (Y_teacher.norm() * Yp.norm() + 1e-12))
            gen_error = float(torch.mean((Y_teacher - Yp) ** 2).item())

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
        results[float(alpha)] = metrics

    return results


# ============================================================
# Memory-Optimized Comparison
# ============================================================
def run_comparison_memory_optimized(alpha_values, agd_steps, gamp_steps_list, mode='optimized'):
    """Run comparison with memory-optimized training (sequential alpha processing)

    Args:
        mode: 'optimized' (FP32, no mask pre-storage) or 'extreme' (FP16 + chunked)
    """
    set_seed(SEED)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)
    use_fp16 = (mode == 'extreme')
    chunk_size = 2000 if mode == 'extreme' else None

    print(f"\n{'='*70}")
    print(f"MEMORY-OPTIMIZED COMPARISON: BiG-AMP vs AGD")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Mode: {mode} (FP16={use_fp16}, chunk={chunk_size})")
    print(f"Device: {DEVICE_INFO.device_type.upper()}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Alpha values: {num_alphas} points (sequential processing)")
    print(f"AGD: {agd_steps} steps")
    print(f"BiG-AMP: {gamp_steps_list} steps")
    print(f"{'='*70}\n")

    results = {
        'alpha_values': list(alpha_values),
        'agd_results': {agd_steps: {}},
        'gamp_results': {s: {} for s in gamp_steps_list},
        'agd_times': {agd_steps: {}},
        'gamp_times': {s: {} for s in gamp_steps_list},
        'mode': mode
    }

    # ========== Run AGD (sequential) ==========
    print(f"[1/2] Running AGD ({agd_steps} steps)...")
    agd_start = time.time()

    for i, alpha in enumerate(alpha_values):
        alpha_seed = SEED + int(alpha * 1000)
        print(f"  Alpha {i+1}/{num_alphas}: α={alpha:.2f}")

        W, X = train_single_agd_memory_optimized(
            Wt, Xt, Y_teacher, alpha, agd_steps, SAMPLES_PER_ALPHA, alpha_seed,
            use_fp16=use_fp16, chunk_size=chunk_size
        )

        metrics = evaluate_single(W, X, Wt, Xt, Y_teacher, alpha, SAMPLES_PER_ALPHA)
        results['agd_results'][agd_steps][float(alpha)] = metrics

        del W, X
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    agd_total_time = time.time() - agd_start
    for alpha in alpha_values:
        results['agd_times'][agd_steps][float(alpha)] = agd_total_time / num_alphas
    print(f"  AGD completed in {agd_total_time:.1f}s")

    # ========== Run BiG-AMP (sequential) ==========
    print(f"\n[2/2] Running BiG-AMP...")

    for gamp_steps in gamp_steps_list:
        print(f"\n  BiG-AMP {gamp_steps} steps:")
        gamp_start = time.time()

        for i, alpha in enumerate(alpha_values):
            alpha_seed = SEED + int(alpha * 1000)

            W, X = train_single_bigamp_memory_optimized(
                Wt, Xt, Y_teacher, alpha, gamp_steps, SAMPLES_PER_ALPHA, alpha_seed,
                use_fp16=use_fp16, chunk_size=chunk_size
            )

            metrics = evaluate_single(W, X, Wt, Xt, Y_teacher, alpha, SAMPLES_PER_ALPHA)
            results['gamp_results'][gamp_steps][float(alpha)] = metrics

            del W, X
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        gamp_total_time = time.time() - gamp_start
        for alpha in alpha_values:
            results['gamp_times'][gamp_steps][float(alpha)] = gamp_total_time / num_alphas
        print(f"    Completed in {gamp_total_time:.1f}s")

    return results


# ============================================================
# Main Comparison (Standard Parallel Mode)
# ============================================================
def run_comparison(alpha_values, agd_steps, gamp_steps_list):
    """Run comparison with smart batching"""
    set_seed(SEED)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)

    # Calculate parallelism for each algorithm
    agd_parallel = calculate_optimal_parallelism(N1, N2, M, SAMPLES_PER_ALPHA, num_alphas, "agd")
    gamp_parallel = calculate_optimal_parallelism(N1, N2, M, SAMPLES_PER_ALPHA, num_alphas, "bigamp")

    print(f"\n{'='*70}")
    print(f"SMART COMPARISON: BiG-AMP vs AGD")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Device: {DEVICE_INFO.device_type.upper()}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB available")
    print(f"Alpha values: {num_alphas} points")
    print(f"AGD: {agd_steps} steps, parallel={agd_parallel}")
    print(f"BiG-AMP: {gamp_steps_list} steps, parallel={gamp_parallel}")
    print(f"{'='*70}\n")

    results = {
        'alpha_values': list(alpha_values),
        'agd_results': {agd_steps: {}},
        'gamp_results': {s: {} for s in gamp_steps_list},
        'agd_times': {agd_steps: {}},
        'gamp_times': {s: {} for s in gamp_steps_list},
    }

    # Pre-generate all masks (stored without sample expansion to save memory)
    print("[1/3] Generating masks...")
    all_masks = []
    for alpha in tqdm(alpha_values, desc="Masks"):
        A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=SEED + int(alpha * 1000))
        all_masks.append(A)  # Shape: (N1, N2), not expanded

    # ========== Run AGD ==========
    print(f"\n[2/3] Running AGD ({agd_steps} steps)...")
    agd_start = time.time()

    for batch_start in range(0, num_alphas, agd_parallel):
        batch_end = min(batch_start + agd_parallel, num_alphas)
        batch_alphas = alpha_values[batch_start:batch_end]
        # Stack masks as (num_alphas, 1, N1, N2) for broadcasting - saves memory!
        batch_masks = torch.stack([all_masks[i].unsqueeze(0)
                                   for i in range(batch_start, batch_end)], dim=0)

        print(f"  Batch {batch_start//agd_parallel + 1}: alphas {batch_start+1}-{batch_end}")

        torch.manual_seed(SEED + 10000 + batch_start)
        W, X = train_batch_agd(Wt, Xt, Y_teacher, batch_alphas, batch_masks,
                               agd_steps, SAMPLES_PER_ALPHA)
        batch_results = evaluate_batch(W, X, Wt, Xt, Y_teacher, batch_alphas,
                                       batch_masks, SAMPLES_PER_ALPHA)
        results['agd_results'][agd_steps].update(batch_results)

        del W, X, batch_masks
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    agd_total_time = time.time() - agd_start
    for alpha in alpha_values:
        results['agd_times'][agd_steps][float(alpha)] = agd_total_time / num_alphas

    print(f"  AGD completed in {agd_total_time:.1f}s")

    # ========== Run BiG-AMP ==========
    print(f"\n[3/3] Running BiG-AMP...")

    for gamp_steps in gamp_steps_list:
        print(f"\n  BiG-AMP {gamp_steps} steps:")
        gamp_start = time.time()

        for batch_start in range(0, num_alphas, gamp_parallel):
            batch_end = min(batch_start + gamp_parallel, num_alphas)
            batch_alphas = alpha_values[batch_start:batch_end]
            # Stack masks as (num_alphas, 1, N1, N2) for broadcasting - saves memory!
            batch_masks = torch.stack([all_masks[i].unsqueeze(0)
                                       for i in range(batch_start, batch_end)], dim=0)

            print(f"    Batch {batch_start//gamp_parallel + 1}: alphas {batch_start+1}-{batch_end}")

            torch.manual_seed(SEED + 10000 + batch_start)
            W, X = train_batch_bigamp(Wt, Xt, Y_teacher, batch_alphas, batch_masks,
                                      gamp_steps, SAMPLES_PER_ALPHA)
            batch_results = evaluate_batch(W, X, Wt, Xt, Y_teacher, batch_alphas,
                                           batch_masks, SAMPLES_PER_ALPHA)
            results['gamp_results'][gamp_steps].update(batch_results)

            del W, X, batch_masks
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        gamp_total_time = time.time() - gamp_start
        for alpha in alpha_values:
            results['gamp_times'][gamp_steps][float(alpha)] = gamp_total_time / num_alphas
        print(f"    Completed in {gamp_total_time:.1f}s")

    return results


def analyze_results(results):
    """Analyze and print results"""
    alpha_values = results['alpha_values']
    agd_steps = list(results['agd_results'].keys())[0]
    gamp_steps_list = list(results['gamp_results'].keys())

    print(f"\n{'='*70}")
    print(f"ANALYSIS RESULTS")
    print(f"{'='*70}")

    # AGD baseline
    agd_qy = np.mean([results['agd_results'][agd_steps][a]['Q_Y_mean']
                      for a in alpha_values if a > 0.5])
    agd_time = np.mean(list(results['agd_times'][agd_steps].values())) * len(alpha_values)

    print(f"\nAGD {agd_steps} steps (baseline):")
    print(f"  Avg Q_Y (alpha > 0.5): {agd_qy:.4f}")
    print(f"  Total time: {agd_time:.1f}s")

    for gamp_steps in gamp_steps_list:
        gamp_qy = np.mean([results['gamp_results'][gamp_steps][a]['Q_Y_mean']
                          for a in alpha_values if a > 0.5])
        gamp_time = np.mean(list(results['gamp_times'][gamp_steps].values())) * len(alpha_values)

        step_efficiency = agd_steps / gamp_steps
        wall_speedup = agd_time / gamp_time if gamp_time > 0 else 0

        print(f"\nBiG-AMP {gamp_steps} steps:")
        print(f"  Avg Q_Y (alpha > 0.5): {gamp_qy:.4f}")
        print(f"  Total time: {gamp_time:.1f}s")
        print(f"  Step efficiency: {step_efficiency:.0f}x")
        print(f"  Wall-clock speedup: {wall_speedup:.1f}x")


def plot_comparison(results):
    """Generate high-contrast comparison plot"""
    alpha_values = np.array(results['alpha_values'])
    agd_steps = list(results['agd_results'].keys())[0]
    gamp_steps_list = sorted(results['gamp_results'].keys())

    AGD_COLOR = '#1a1a2e'
    GAMP_COLORS = ['#e63946', '#2a9d8f', '#e9c46a', '#f4a261', '#264653']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Q_Y
    ax1 = axes[0, 0]
    qy_agd = [results['agd_results'][agd_steps][a]['Q_Y_mean'] for a in alpha_values]
    ax1.plot(alpha_values, qy_agd, 'D-', color=AGD_COLOR,
             markersize=8, linewidth=3, markeredgewidth=2, markeredgecolor='white',
             label=f'AGD {agd_steps:,} steps', zorder=10)

    markers = ['o', 's', '^', 'v', 'p']
    for i, steps in enumerate(gamp_steps_list):
        qy = [results['gamp_results'][steps][a]['Q_Y_mean'] for a in alpha_values]
        ax1.plot(alpha_values, qy, f'{markers[i % 5]}--', color=GAMP_COLORS[i % 5],
                 markersize=6, linewidth=2, alpha=0.85, label=f'BiG-AMP {steps}')

    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax1.set_ylabel('Q_Y', fontsize=14)
    ax1.set_title(f'Q_Y: BiG-AMP vs AGD ({N1}x{N2})', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Gen Error
    ax2 = axes[0, 1]
    ge_agd = [results['agd_results'][agd_steps][a]['Gen_Error_mean'] for a in alpha_values]
    ax2.semilogy(alpha_values, ge_agd, 'D-', color=AGD_COLOR,
                 markersize=8, linewidth=3, markeredgewidth=2, markeredgecolor='white',
                 label=f'AGD {agd_steps:,}', zorder=10)

    for i, steps in enumerate(gamp_steps_list):
        ge = [results['gamp_results'][steps][a]['Gen_Error_mean'] for a in alpha_values]
        ax2.semilogy(alpha_values, ge, f'{markers[i % 5]}--', color=GAMP_COLORS[i % 5],
                     markersize=6, linewidth=2, alpha=0.85, label=f'BiG-AMP {steps}')

    ax2.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax2.set_ylabel('Gen Error (log)', fontsize=14)
    ax2.set_title('Generalization Error', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.4)

    # Plot 3: Q_W' and Q_X'
    ax3 = axes[1, 0]
    qw_agd = [results['agd_results'][agd_steps][a]['Q_W_prime_mean'] for a in alpha_values]
    qx_agd = [results['agd_results'][agd_steps][a]['Q_X_prime_mean'] for a in alpha_values]
    ax3.plot(alpha_values, qw_agd, 'D-', color=AGD_COLOR, markersize=8, linewidth=3,
             markeredgewidth=2, markeredgecolor='white', label=f"AGD Q_W'", zorder=10)
    ax3.plot(alpha_values, qx_agd, 's-', color='#4a4e69', markersize=7, linewidth=3,
             markeredgewidth=2, markeredgecolor='white', label=f"AGD Q_X'", zorder=10)

    best_gamp = max(gamp_steps_list)
    qw_gamp = [results['gamp_results'][best_gamp][a]['Q_W_prime_mean'] for a in alpha_values]
    qx_gamp = [results['gamp_results'][best_gamp][a]['Q_X_prime_mean'] for a in alpha_values]
    ax3.plot(alpha_values, qw_gamp, 'o--', color=GAMP_COLORS[0], markersize=6, linewidth=2,
             alpha=0.85, label=f"BiG-AMP {best_gamp} Q_W'")
    ax3.plot(alpha_values, qx_gamp, '^--', color=GAMP_COLORS[1], markersize=6, linewidth=2,
             alpha=0.85, label=f"BiG-AMP {best_gamp} Q_X'")

    ax3.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax3.set_ylabel("Q'", fontsize=14)
    ax3.set_title("Q_W' and Q_X'", fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.4)
    ax3.set_ylim(-0.05, 1.05)

    # Plot 4: Speedup
    ax4 = axes[1, 1]
    agd_time = np.mean(list(results['agd_times'][agd_steps].values())) * len(alpha_values)

    speedups = []
    step_effs = []
    for steps in gamp_steps_list:
        gamp_time = np.mean(list(results['gamp_times'][steps].values())) * len(alpha_values)
        speedups.append(agd_time / gamp_time if gamp_time > 0 else 0)
        step_effs.append(agd_steps / steps)

    x = np.arange(len(gamp_steps_list))
    width = 0.35
    bars1 = ax4.bar(x - width/2, speedups, width, label='Wall-clock', color=GAMP_COLORS[0])
    bars2 = ax4.bar(x + width/2, step_effs, width, label='Step Efficiency', color=GAMP_COLORS[1])

    for bar, val in zip(bars1, speedups):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.0f}x', ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, step_effs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.0f}x', ha='center', fontsize=10, fontweight='bold')

    ax4.set_xlabel('BiG-AMP Steps', fontsize=14)
    ax4.set_ylabel('Speedup', fontsize=14)
    ax4.set_title(f'Speedup vs AGD {agd_steps:,} steps', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(s) for s in gamp_steps_list])
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    save_path = RESULT_DIR / 'comparison_smart.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved: {save_path}")
    plt.close(fig)


def main():
    global N1, N2, M, ALPHA_TILDE_STEP, RESULT_DIR

    parser = argparse.ArgumentParser(description='Smart BiG-AMP vs AGD Comparison')
    parser.add_argument('--n1', type=int, default=N1, help='Matrix N1 dimension')
    parser.add_argument('--n2', type=int, default=N2, help='Matrix N2 dimension')
    parser.add_argument('--m', type=int, default=M, help='Latent dimension M')
    parser.add_argument('--agd-steps', type=int, default=5000, help='AGD training steps')
    parser.add_argument('--alpha-step', type=float, default=0.1, help='Alpha step size')
    parser.add_argument('--memory-mode', type=str, default='auto',
                        choices=['auto', 'parallel', 'optimized', 'extreme'],
                        help='Memory mode: auto (default), parallel, optimized, extreme')
    args = parser.parse_args()

    N1 = args.n1
    N2 = args.n2
    M = args.m
    ALPHA_TILDE_STEP = args.alpha_step
    RESULT_DIR = Path(__file__).parent / "result" / f"{N1}_{N2}_{M}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    alpha_values = np.arange(ALPHA_TILDE_START, ALPHA_TILDE_STOP + 1e-12, ALPHA_TILDE_STEP)
    gamp_steps_list = [10, 20, 50, 100, 200]

    # Select memory mode
    mode = select_memory_mode(N1, N2, M, SAMPLES_PER_ALPHA, len(alpha_values), args.memory_mode)

    # Run comparison based on mode
    if mode == 'parallel':
        results = run_comparison(alpha_values, args.agd_steps, gamp_steps_list)
    else:
        results = run_comparison_memory_optimized(alpha_values, args.agd_steps, gamp_steps_list, mode)

    analyze_results(results)
    plot_comparison(results)

    # Save
    results_path = RESULT_DIR / 'comparison_smart.json'
    results_json = {
        'alpha_values': [float(a) for a in results['alpha_values']],
        'agd_results': {str(k): {str(a): v for a, v in d.items()} for k, d in results['agd_results'].items()},
        'gamp_results': {str(k): {str(a): v for a, v in d.items()} for k, d in results['gamp_results'].items()},
        'agd_times': {str(k): {str(a): v for a, v in d.items()} for k, d in results['agd_times'].items()},
        'gamp_times': {str(k): {str(a): v for a, v in d.items()} for k, d in results['gamp_times'].items()},
    }
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
