"""
BiG-AMP Orthogonal Teacher Comparison

Compares standard Gaussian teacher vs orthogonal teacher (QR-decomposed)
to verify that the M/N linear bias in low-alpha region is a finite-size effect.

Features:
1. Standard teacher: W, X ~ N(0, 1/sqrt(M))
2. Orthogonal teacher: W^T W = I_M, X X^T = I_M (eliminates finite-size fluctuations)
3. Computes Q_Y against both teachers for comparison
4. Three output folders:
   - folder1_per_config: Same (N,M), compare Q_Y/Q_Y_unobs/Q_Y_ortho
   - folder2_cross_config: Same metric, compare across (N,M)
   - folder3_ortho_focused: Q_Y_ortho with zoomed alpha range

Usage:
    # Default configurations
    python bigamp_ortho_teacher.py

    # Custom configurations
    python bigamp_ortho_teacher.py --sizes "500:50,1000:50,2000:50"

    # Custom alpha focus range
    python bigamp_ortho_teacher.py --alpha-focus-max 2.0
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
from typing import List, Tuple
from collections import deque
import itertools

# ============================================================
# Default Parameters
# ============================================================
# Default (N, M) configurations to compare
DEFAULT_SIZES = [
    (1000, 100),
    (2000, 100),
    (3000, 100),
    (4000, 100),
]

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 3
ALPHA_TILDE_STEP = 0.1

# BiG-AMP parameters
DAMPING = 0.5
NOISE_VAR = 1e-10
MAX_STEPS = 5000

SAMPLES_PER_ALPHA = 4
RESAMPLE_MASK_EACH_TRIAL = True
SEED = 42

# ============================================================
# Graph Generation Configuration
# ============================================================
USE_BIREGULAR_GRAPH = False

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))

# Precision settings
USE_BF16 = DEVICE.type == 'cuda'
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
            available_memory_gb=32.0,
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
    """Create standard Gaussian teacher model W_true and X_true"""
    torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale
    return W, X


def create_orthogonal_teacher(N1, N2, M, device, seed=42):
    """
    Create orthogonal teacher model using QR decomposition.

    Enforces W^T W = I_M and X X^T = I_M (up to scaling),
    eliminating finite-size fluctuations in the overlap metrics.

    Mathematical properties:
    - W_ortho has orthonormal columns: W^T W = I_M
    - X_ortho has orthonormal rows: X X^T = I_M
    - Scaling: ||W||_F^2 = N1, ||X||_F^2 = N2 (same as standard)

    This removes the 2*alpha*M/N linear bias in low-alpha region,
    simulating the thermodynamic limit (N -> infinity) behavior.
    """
    torch.manual_seed(seed)

    # Generate random matrices
    W_raw = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_raw = torch.randn(M, N2, device=device, dtype=torch.float32)

    # QR decomposition for W (thin QR: N1 x M -> Q: N1 x M, R: M x M)
    # After QR: W_ortho^T @ W_ortho = I_M
    W_ortho, _ = torch.linalg.qr(W_raw, mode='reduced')

    # QR decomposition for X^T, then transpose back
    # After QR: X_ortho @ X_ortho^T = I_M
    X_ortho_T, _ = torch.linalg.qr(X_raw.T, mode='reduced')
    X_ortho = X_ortho_T.T

    # Scale to match expected Frobenius norm of standard teacher
    # Standard: E[||W||_F^2] = N1 * M * (1/M) = N1
    # Orthogonal: ||W_ortho||_F^2 = M (since orthonormal columns)
    # Scale factor: sqrt(N1/M) to get ||W||_F^2 = N1
    W_true = W_ortho * (N1 / M) ** 0.5
    X_true = X_ortho * (N2 / M) ** 0.5

    return W_true, X_true


def sample_pairs_random_gpu(N1, N2, C, device, seed=None):
    """Pure random mask generation (entirely on GPU)"""
    if seed is not None:
        torch.manual_seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"Requested edge count C={C} exceeds matrix total size {N1}x{N2}={total}")

    if C == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    idx = torch.randperm(total, device=device)[:C]
    i_idx = idx // N2
    j_idx = idx % N2

    return i_idx, j_idx, C


def sample_pairs_biregular_exact(N1, N2, M, alpha_tilde_left, device, seed=None):
    """Graph generation function (choose method based on USE_BIREGULAR_GRAPH)"""
    deg_left = int(round(alpha_tilde_left * M))
    deg_left = max(0, min(deg_left, N2))
    total_edges = N1 * deg_left

    if not USE_BIREGULAR_GRAPH:
        return sample_pairs_random_gpu(N1, N2, total_edges, device, seed)

    # Bi-regular graph generation using Dinic algorithm
    if deg_left == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    if deg_left > N2:
        raise RuntimeError(f"deg_left={deg_left} > N2={N2}, infeasible")
    deg_right_exact = total_edges // N2
    if total_edges % N2 == 0 and deg_right_exact > N1:
        raise RuntimeError(f"deg_right={deg_right_exact} > N1={N1}, infeasible")

    if seed is not None:
        rng = np.random.RandomState(seed + 12345 + int(round(alpha_tilde_left * 1e6)))
    else:
        rng = np.random.RandomState()

    base = total_edges // N2
    rem = total_edges % N2
    right_target = np.full(N2, base, dtype=int)
    if rem > 0:
        idx = np.arange(N2)
        rng.shuffle(idx)
        right_target[idx[:rem]] += 1

    if right_target.max() > N1:
        raise RuntimeError(f"Some right node target degree {right_target.max()} > N1={N1}, infeasible")

    class Dinic:
        __slots__ = ("n", "g", "lvl", "it")
        def __init__(self, n):
            self.n = n
            self.g = [[] for _ in range(n)]
        def add_edge(self, u, v, cap):
            self.g[u].append([v, cap, len(self.g[v])])
            self.g[v].append([u, 0, len(self.g[u]) - 1])
        def bfs(self, s, t):
            self.lvl = [-1] * self.n
            q = deque([s])
            self.lvl[s] = 0
            while q:
                u = q.popleft()
                for v, cap, _ in self.g[u]:
                    if cap > 0 and self.lvl[v] < 0:
                        self.lvl[v] = self.lvl[u] + 1
                        q.append(v)
            return self.lvl[t] >= 0
        def dfs(self, u, t, f):
            if u == t: return f
            for i in range(self.it[u], len(self.g[u])):
                self.it[u] = i
                v, cap, rev = self.g[u][i]
                if cap > 0 and self.lvl[u] + 1 == self.lvl[v]:
                    d = self.dfs(v, t, min(f, cap))
                    if d > 0:
                        self.g[u][i][1] -= d
                        self.g[v][rev][1] += d
                        return d
            return 0
        def max_flow(self, s, t):
            flow = 0
            INF = 10**9
            while self.bfs(s, t):
                self.it = [0] * self.n
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0: break
                    flow += f
            return flow

    S, L_off, R_off = 0, 1, 1 + N1
    T = R_off + N2
    din = Dinic(T + 1)

    for i in range(N1):
        din.add_edge(S, L_off + i, deg_left)

    all_pairs = list(itertools.product(range(N1), range(N2)))
    rng.shuffle(all_pairs)

    for i, j in all_pairs:
        ui = L_off + i
        vj = R_off + j
        din.add_edge(ui, vj, 1)

    for j in range(N2):
        din.add_edge(R_off + j, T, int(right_target[j]))

    f = din.max_flow(S, T)
    if f != total_edges:
        raise RuntimeError(f"maxflow only got {f}/{total_edges}, degree sequence infeasible")

    i_list, j_list = [], []
    for i in range(N1):
        u = L_off + i
        for v, cap, rev in din.g[u]:
            if (R_off <= v < R_off + N2):
                if din.g[v][rev][1] > 0:
                    j = v - R_off
                    i_list.append(i)
                    j_list.append(j)

    i_idx = torch.tensor(i_list, dtype=torch.long, device=device)
    j_idx = torch.tensor(j_list, dtype=torch.long, device=device)
    C = len(i_list)
    return i_idx, j_idx, C


def sample_mask(N1, N2, M, alpha, device, seed=None):
    """Generate observation mask"""
    c = alpha * M
    i_idx, j_idx, C = sample_pairs_biregular_exact(N1, N2, M, alpha, device, seed)
    mask = torch.zeros((N1, N2), device=device, dtype=torch.float32)
    if C > 0:
        mask[i_idx, j_idx] = 1.0
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
    """Compute normalized Gram overlap in [0, 1] range with baseline correction."""
    q = gram_overlap_cosine(A, B, use_left)
    if use_left:
        n, m = A.shape
    else:
        n, m = A.shape[1], A.shape[0]
    b = m / (m + n + 1)
    qc = (q - b) / (1.0 - b + 1e-12)
    return float(max(0.0, min(1.0, qc)))


# ============================================================
# Memory Management
# ============================================================
def estimate_memory_per_alpha(N1, N2, M, S, dtype_bytes=4):
    """Estimate GPU memory needed per alpha value"""
    student_params = 2 * (S * N1 * M + S * M * N2)
    intermediate = 16 * S * N1 * N2
    total_elements = student_params + intermediate
    return total_elements * dtype_bytes / (1024**3)


def calculate_smart_parallelism(N1, N2, M, S, num_alphas):
    """Calculate optimal parallelism based on memory"""
    MAX_GPU_MEMORY_GB = min(DEVICE_INFO.available_memory_gb, 32.0)
    RESERVED_MEMORY_GB = 3.0
    available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

    per_alpha_mem = estimate_memory_per_alpha(N1, N2, M, S)
    teacher_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)
    single_mask_mem = N1 * N2 * 4 / (1024**3)

    mem_per_batch_alpha = per_alpha_mem + single_mask_mem
    usable_mem = available * 0.85 - teacher_mem

    if mem_per_batch_alpha <= 0:
        return num_alphas

    max_parallel = max(1, min(int(usable_mem / mem_per_batch_alpha), num_alphas))
    return max_parallel


# ============================================================
# BiG-AMP Training
# ============================================================
def train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, alpha_values, steps, S, M,
                          damping=0.5, noise_var=1e-6):
    """BiG-AMP training with parallel alpha processing"""
    device = Wt.device
    N1 = Wt.shape[0]
    N2 = Xt.shape[1]
    num_alphas = len(alpha_values)
    alpha_scale = 1.0 / (M ** 0.5)
    scale = 1.0 / (M ** 0.5)

    w_hat = torch.randn((num_alphas, S, N1, M), device=device) * scale
    x_hat = torch.randn((num_alphas, S, M, N2), device=device) * scale
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    Y_teacher_exp = Y_teacher.unsqueeze(0).unsqueeze(0)

    for step in tqdm(range(steps), desc="BiG-AMP Training", leave=False, mininterval=1.0):
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher_exp - z_hat) * A_all
        s = residual / V

        tau_W = (alpha_scale ** 2) * torch.matmul(A_all / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
        w_hat_new = w_hat + w_var_new * r_W
        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new, min=1e-8, max=1.0)

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


def train_bigamp_single(Wt, Xt, Y_teacher, alpha, steps, S, M, seed,
                        damping=0.5, noise_var=1e-6, resample_mask=False):
    """Memory-optimized BiG-AMP training for single alpha"""
    device = Wt.device
    N1 = Wt.shape[0]
    N2 = Xt.shape[1]
    alpha_scale = 1.0 / (M ** 0.5)
    scale = 1.0 / (M ** 0.5)

    if resample_mask and S > 1:
        A = torch.zeros((S, N1, N2), device=device)
        for s in range(S):
            mask_seed = seed + s * 10000
            A_s, _ = sample_mask(N1, N2, M, alpha, device, seed=mask_seed)
            A[s] = A_s
    else:
        A, _ = sample_mask(N1, N2, M, alpha, device, seed=seed)
        A = A.unsqueeze(0)

    torch.manual_seed(seed + 10000)
    w_hat = torch.randn((S, N1, M), device=device) * scale
    x_hat = torch.randn((S, M, N2), device=device) * scale
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    for _ in tqdm(range(steps), desc=f"BiG-AMP alpha={alpha:.2f}", leave=False, mininterval=1.0):
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher - z_hat) * A
        s = residual / V

        tau_W = (alpha_scale ** 2) * torch.matmul(A / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
        w_hat_new = w_hat + w_var_new * r_W
        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
        w_sq2 = w_hat ** 2
        p_var2 = (alpha_scale ** 2) * (torch.matmul(w_sq2, x_var) + torch.matmul(w_var, x_sq))
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher - z_hat2) * A
        s2 = residual2 / V2

        tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A / V2)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new = 1.0 / (M + tau_X)
        r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
        x_hat_new = x_hat + x_var_new * r_X
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = torch.clamp(damping * x_var + (1 - damping) * x_var_new, min=1e-8, max=1.0)

    return w_hat, x_hat


# ============================================================
# Evaluation (Original version from bigamp_multi_size.py)
# ============================================================
@torch.no_grad()
def evaluate_batch(W, X, Wt, Xt, Y_teacher, alpha_values, S, A_all=None):
    """Evaluate metrics for all alphas

    Args:
        A_all: observation masks, shape depends on RESAMPLE_MASK_EACH_TRIAL:
               - (num_alphas, S, N1, N2) if resample each trial
               - (num_alphas, 1, N1, N2) if shared mask
    """
    results = {}

    for a_idx, alpha in enumerate(alpha_values):
        trial_results = []

        for s in range(S):
            W_s = W[a_idx, s] if W.dim() == 4 else W[s]
            X_s = X[a_idx, s] if X.dim() == 4 else X[s]

            Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
            Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

            Yp = W_s @ X_s
            Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                       (Y_teacher.norm() * Yp.norm() + 1e-12))

            # Compute Q_Y_unobserved: overlap only on unobserved positions (mask=0)
            if A_all is not None:
                # Get mask for this alpha and trial
                if A_all.shape[1] == S:
                    mask = A_all[a_idx, s]  # (N1, N2)
                else:
                    mask = A_all[a_idx, 0]  # Shared mask
                unobs_mask = 1.0 - mask
                Y_unobs = Y_teacher * unobs_mask
                Yp_unobs = Yp * unobs_mask
                Y_unobs_norm = Y_unobs.norm()
                Yp_unobs_norm = Yp_unobs.norm()
                if Y_unobs_norm > 1e-12 and Yp_unobs_norm > 1e-12:
                    Q_Y_unobserved = float((Y_unobs.flatten() * Yp_unobs.flatten()).sum() /
                                          (Y_unobs_norm * Yp_unobs_norm))
                else:
                    Q_Y_unobserved = 0.0
            else:
                Q_Y_unobserved = Q_Y

            trial_results.append({
                'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
                'Q_Y': Q_Y, 'Q_Y_unobserved': Q_Y_unobserved
            })

        metrics = {}
        for key in trial_results[0].keys():
            vals = [r[key] for r in trial_results]
            metrics[f'{key}_mean'] = float(np.mean(vals))
            metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        results[float(alpha)] = metrics

    return results


@torch.no_grad()
def evaluate_single(W, X, Wt, Xt, Y_teacher, S, A=None):
    """Evaluate single alpha result

    Args:
        A: observation mask, shape depends on RESAMPLE_MASK_EACH_TRIAL:
           - (S, N1, N2) if resample each trial
           - (1, N1, N2) if shared mask
    """
    trial_results = []

    for s in range(S):
        W_s, X_s = W[s], X[s]
        Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
        Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

        Yp = W_s @ X_s
        Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                   (Y_teacher.norm() * Yp.norm() + 1e-12))

        # Compute Q_Y_unobserved: overlap only on unobserved positions (mask=0)
        if A is not None:
            if A.shape[0] == S:
                mask = A[s]  # (N1, N2)
            else:
                mask = A[0]  # Shared mask
            unobs_mask = 1.0 - mask
            Y_unobs = Y_teacher * unobs_mask
            Yp_unobs = Yp * unobs_mask
            Y_unobs_norm = Y_unobs.norm()
            Yp_unobs_norm = Yp_unobs.norm()
            if Y_unobs_norm > 1e-12 and Yp_unobs_norm > 1e-12:
                Q_Y_unobserved = float((Y_unobs.flatten() * Yp_unobs.flatten()).sum() /
                                      (Y_unobs_norm * Yp_unobs_norm))
            else:
                Q_Y_unobserved = 0.0
        else:
            Q_Y_unobserved = Q_Y

        trial_results.append({
            'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
            'Q_Y': Q_Y, 'Q_Y_unobserved': Q_Y_unobserved
        })

    metrics = {}
    for key in trial_results[0].keys():
        vals = [r[key] for r in trial_results]
        metrics[f'{key}_mean'] = float(np.mean(vals))
        metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    return metrics


# ============================================================
# Training for Single Configuration (Two Independent Training Runs)
# ============================================================
def run_single_config(N, M, alpha_values, steps, S, damping, noise_var, seed):
    """
    Run training for a single (N, M) configuration.

    Two independent training runs:
    1. Standard teacher training -> Q_Y, Q_Y_unobserved
    2. Orthogonal teacher training (same masks) -> Q_Y_ortho

    This ensures Q_Y_ortho is a fair comparison with Q_Y, as both students
    are trained against their respective teachers.
    """
    N1 = N2 = N
    set_seed(seed)

    # Create both teachers
    Wt_std, Xt_std = create_teacher(N1, N2, M, DEVICE, seed=seed)
    Wt_ortho, Xt_ortho = create_orthogonal_teacher(N1, N2, M, DEVICE, seed=seed)

    Y_teacher_std = Wt_std @ Xt_std
    Y_teacher_ortho = Wt_ortho @ Xt_ortho

    # Verify orthogonality
    G_std = (M / N1) * (Wt_std.T @ Wt_std)
    G_ortho = (M / N1) * (Wt_ortho.T @ Wt_ortho)
    std_dev = float((G_std - torch.eye(M, device=DEVICE)).abs().mean())
    ortho_dev = float((G_ortho - torch.eye(M, device=DEVICE)).abs().mean())
    print(f"  Standard teacher G deviation: {std_dev:.6f}")
    print(f"  Orthogonal teacher G deviation: {ortho_dev:.6f}")

    num_alphas = len(alpha_values)
    max_parallel = calculate_smart_parallelism(N1, N2, M, S, num_alphas)

    print(f"\n  Training N={N}, M={M}")
    print(f"  Parallelism: {max_parallel} alphas")

    all_results = {}

    if max_parallel >= 2:
        # Parallel mode
        for batch_start in range(0, num_alphas, max_parallel):
            batch_end = min(batch_start + max_parallel, num_alphas)
            batch_alphas = alpha_values[batch_start:batch_end]
            batch_size = len(batch_alphas)

            # Generate masks ONCE (shared by both training runs)
            if RESAMPLE_MASK_EACH_TRIAL:
                A_all = torch.zeros((batch_size, S, N1, N2), device=DEVICE)
                for i, alpha in enumerate(batch_alphas):
                    for s in range(S):
                        mask_seed = seed + int(alpha * 1000) + s * 10000
                        A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
                        A_all[i, s] = A
            else:
                A_all = torch.zeros((batch_size, 1, N1, N2), device=DEVICE)
                for i, alpha in enumerate(batch_alphas):
                    mask_seed = seed + int(alpha * 1000)
                    A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
                    A_all[i, 0] = A

            # === Training 1: Standard teacher ===
            W_std, X_std = train_bigamp_parallel(Wt_std, Xt_std, Y_teacher_std, A_all,
                                                  batch_alphas, steps, S, M,
                                                  damping=damping, noise_var=noise_var)
            results_std = evaluate_batch(W_std, X_std, Wt_std, Xt_std, Y_teacher_std,
                                          batch_alphas, S, A_all=A_all)
            del W_std, X_std

            # === Training 2: Orthogonal teacher (same masks) ===
            W_ortho, X_ortho = train_bigamp_parallel(Wt_ortho, Xt_ortho, Y_teacher_ortho, A_all,
                                                      batch_alphas, steps, S, M,
                                                      damping=damping, noise_var=noise_var)
            results_ortho = evaluate_batch(W_ortho, X_ortho, Wt_ortho, Xt_ortho, Y_teacher_ortho,
                                            batch_alphas, S, A_all=A_all)
            del W_ortho, X_ortho

            # Merge results: Q_Y/Q_Y_unobserved from std, Q_Y_ortho/Q_Y_ortho_unobserved from ortho
            for alpha in batch_alphas:
                a = float(alpha)
                all_results[a] = {
                    'Q_W_prime_mean': results_std[a]['Q_W_prime_mean'],
                    'Q_W_prime_std': results_std[a]['Q_W_prime_std'],
                    'Q_X_prime_mean': results_std[a]['Q_X_prime_mean'],
                    'Q_X_prime_std': results_std[a]['Q_X_prime_std'],
                    # Standard teacher metrics
                    'Q_Y_mean': results_std[a]['Q_Y_mean'],
                    'Q_Y_std': results_std[a]['Q_Y_std'],
                    'Q_Y_unobserved_mean': results_std[a]['Q_Y_unobserved_mean'],
                    'Q_Y_unobserved_std': results_std[a]['Q_Y_unobserved_std'],
                    # Orthogonal teacher metrics (both full and unobserved)
                    'Q_Y_ortho_mean': results_ortho[a]['Q_Y_mean'],
                    'Q_Y_ortho_std': results_ortho[a]['Q_Y_std'],
                    'Q_Y_ortho_unobserved_mean': results_ortho[a]['Q_Y_unobserved_mean'],
                    'Q_Y_ortho_unobserved_std': results_ortho[a]['Q_Y_unobserved_std'],
                }

            del A_all
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    else:
        # Sequential mode
        for alpha in alpha_values:
            alpha_seed = seed + int(alpha * 1000)

            # Generate masks ONCE
            if RESAMPLE_MASK_EACH_TRIAL:
                A = torch.zeros((S, N1, N2), device=DEVICE)
                for s in range(S):
                    mask_seed = alpha_seed + s * 10000
                    A_s, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
                    A[s] = A_s
            else:
                A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=alpha_seed)
                A = A.unsqueeze(0)

            # === Training 1: Standard teacher ===
            W_std, X_std = train_bigamp_single(Wt_std, Xt_std, Y_teacher_std, alpha,
                                                steps, S, M, alpha_seed,
                                                damping=damping, noise_var=noise_var,
                                                resample_mask=RESAMPLE_MASK_EACH_TRIAL)
            results_std = evaluate_single(W_std, X_std, Wt_std, Xt_std, Y_teacher_std, S, A=A)
            del W_std, X_std

            # === Training 2: Orthogonal teacher (same masks) ===
            W_ortho, X_ortho = train_bigamp_single(Wt_ortho, Xt_ortho, Y_teacher_ortho, alpha,
                                                    steps, S, M, alpha_seed,
                                                    damping=damping, noise_var=noise_var,
                                                    resample_mask=RESAMPLE_MASK_EACH_TRIAL)
            results_ortho = evaluate_single(W_ortho, X_ortho, Wt_ortho, Xt_ortho,
                                             Y_teacher_ortho, S, A=A)
            del W_ortho, X_ortho

            # Merge results
            all_results[float(alpha)] = {
                'Q_W_prime_mean': results_std['Q_W_prime_mean'],
                'Q_W_prime_std': results_std['Q_W_prime_std'],
                'Q_X_prime_mean': results_std['Q_X_prime_mean'],
                'Q_X_prime_std': results_std['Q_X_prime_std'],
                # Standard teacher metrics
                'Q_Y_mean': results_std['Q_Y_mean'],
                'Q_Y_std': results_std['Q_Y_std'],
                'Q_Y_unobserved_mean': results_std['Q_Y_unobserved_mean'],
                'Q_Y_unobserved_std': results_std['Q_Y_unobserved_std'],
                # Orthogonal teacher metrics (both full and unobserved)
                'Q_Y_ortho_mean': results_ortho['Q_Y_mean'],
                'Q_Y_ortho_std': results_ortho['Q_Y_std'],
                'Q_Y_ortho_unobserved_mean': results_ortho['Q_Y_unobserved_mean'],
                'Q_Y_ortho_unobserved_std': results_ortho['Q_Y_unobserved_std'],
            }

            del A
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    return all_results


# ============================================================
# Visualization
# ============================================================
def get_color_palette(n_colors):
    """Get a list of distinct colors for plotting"""
    colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Gray
    ]
    if n_colors <= len(colors):
        return colors[:n_colors]
    return [colors[i % len(colors)] for i in range(n_colors)]


# ============================================================
# Folder 1: Per-config comparison (same N,M, different metrics)
# ============================================================
def plot_per_config_comparison(results, N, M, alpha_values, save_dir, steps):
    """
    Plot all 4 metrics on the same figure for one (N,M) config:
    - Q_Y, Q_Y_unobserved (standard teacher)
    - Q_Y_ortho, Q_Y_ortho_unobserved (orthogonal teacher)

    Purpose: Verify whether orthogonal teacher eliminates the linear offset
    in the low-alpha region.
    """
    aL = np.array(alpha_values)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Q_Y (standard teacher) - solid red
    qY = np.array([results[a]['Q_Y_mean'] for a in alpha_values])
    qY_err = np.array([results[a]['Q_Y_std'] for a in alpha_values])
    ax.plot(aL, qY, 'o-', color='#e41a1c', linewidth=2, markersize=4,
            label=r'$Q_Y$ (standard)')
    ax.fill_between(aL, qY - qY_err, qY + qY_err, color='#e41a1c', alpha=0.15)

    # Q_Y_unobserved (standard teacher) - dashed red
    qY_unobs = np.array([results[a]['Q_Y_unobserved_mean'] for a in alpha_values])
    qY_unobs_err = np.array([results[a]['Q_Y_unobserved_std'] for a in alpha_values])
    ax.plot(aL, qY_unobs, 's--', color='#e41a1c', linewidth=2, markersize=4,
            label=r'$Q_Y^{unobs}$ (standard)')
    ax.fill_between(aL, qY_unobs - qY_unobs_err, qY_unobs + qY_unobs_err,
                    color='#e41a1c', alpha=0.1)

    # Q_Y_ortho (orthogonal teacher) - solid green
    qY_ortho = np.array([results[a]['Q_Y_ortho_mean'] for a in alpha_values])
    qY_ortho_err = np.array([results[a]['Q_Y_ortho_std'] for a in alpha_values])
    ax.plot(aL, qY_ortho, '^-', color='#4daf4a', linewidth=2, markersize=4,
            label=r'$Q_Y^{ortho}$ (orthogonal)')
    ax.fill_between(aL, qY_ortho - qY_ortho_err, qY_ortho + qY_ortho_err,
                    color='#4daf4a', alpha=0.15)

    # Q_Y_ortho_unobserved (orthogonal teacher) - dashed green
    qY_ortho_unobs = np.array([results[a]['Q_Y_ortho_unobserved_mean'] for a in alpha_values])
    qY_ortho_unobs_err = np.array([results[a]['Q_Y_ortho_unobserved_std'] for a in alpha_values])
    ax.plot(aL, qY_ortho_unobs, 'v--', color='#4daf4a', linewidth=2, markersize=4,
            label=r'$Q_Y^{ortho,unobs}$ (orthogonal)')
    ax.fill_between(aL, qY_ortho_unobs - qY_ortho_unobs_err, qY_ortho_unobs + qY_ortho_unobs_err,
                    color='#4daf4a', alpha=0.1)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax.set_ylabel(r'Overlap', fontsize=14)
    ax.set_title(f'Metric Comparison: N={N}, M={M}\n(BiG-AMP, {steps} steps)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right', ncol=2)

    save_path = save_dir / f'metrics_comparison_N{N}_M{M}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ============================================================
# Folder 2: Cross-config comparison (different N,M, same metric)
# ============================================================
def plot_qy_comparison(all_results, sizes, alpha_values, save_path, steps):
    """Plot Q_Y comparison across different (N, M) configurations"""
    n_configs = len(sizes)
    colors = get_color_palette(n_configs)
    aL = np.array(alpha_values)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qY_mu = np.array([results[a]['Q_Y_mean'] for a in alpha_values])

        label = f'N={N}, M={M}'
        ax.plot(aL, qY_mu, marker='o', linewidth=2, markersize=4,
                color=colors[i], label=label)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax.set_ylabel(r'$Q_Y$', fontsize=14)
    ax.set_title(f'Q_Y (Standard Teacher) Comparison\n(BiG-AMP, {steps} steps)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y comparison plot saved: {save_path}")
    plt.close(fig)


def plot_qy_unobserved_comparison(all_results, sizes, alpha_values, save_path, steps):
    """Plot Q_Y_unobserved comparison across different (N, M) configurations"""
    n_configs = len(sizes)
    colors = get_color_palette(n_configs)
    aL = np.array(alpha_values)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qY_unobs_mu = np.array([results[a].get('Q_Y_unobserved_mean', results[a]['Q_Y_mean'])
                                for a in alpha_values])

        label = f'N={N}, M={M}'
        ax.plot(aL, qY_unobs_mu, marker='o', linewidth=2, markersize=4,
                color=colors[i], label=label)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax.set_ylabel(r'$Q_Y^{unobserved}$', fontsize=14)
    ax.set_title(f'Q_Y_unobserved (Generalization) Comparison\n(BiG-AMP, {steps} steps)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y_unobserved comparison plot saved: {save_path}")
    plt.close(fig)


def plot_qy_ortho_comparison(all_results, sizes, alpha_values, save_path, steps):
    """Plot Q_Y_ortho comparison across different (N, M) configurations"""
    n_configs = len(sizes)
    colors = get_color_palette(n_configs)
    aL = np.array(alpha_values)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qY_ortho_mu = np.array([results[a]['Q_Y_ortho_mean'] for a in alpha_values])

        label = f'N={N}, M={M}'
        ax.plot(aL, qY_ortho_mu, marker='o', linewidth=2, markersize=4,
                color=colors[i], label=label)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax.set_ylabel(r'$Q_Y^{ortho}$', fontsize=14)
    ax.set_title(f'Q_Y (Orthogonal Teacher) Comparison\n(BiG-AMP, {steps} steps)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y_ortho comparison plot saved: {save_path}")
    plt.close(fig)


def plot_qy_ortho_unobserved_comparison(all_results, sizes, alpha_values, save_path, steps):
    """Plot Q_Y_ortho_unobserved comparison across different (N, M) configurations"""
    n_configs = len(sizes)
    colors = get_color_palette(n_configs)
    aL = np.array(alpha_values)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qY_ortho_unobs_mu = np.array([results[a]['Q_Y_ortho_unobserved_mean']
                                       for a in alpha_values])

        label = f'N={N}, M={M}'
        ax.plot(aL, qY_ortho_unobs_mu, marker='o', linewidth=2, markersize=4,
                color=colors[i], label=label)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax.set_ylabel(r'$Q_Y^{ortho,unobs}$', fontsize=14)
    ax.set_title(f'Q_Y_unobserved (Orthogonal Teacher) Comparison\n(BiG-AMP, {steps} steps)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y_ortho_unobserved comparison plot saved: {save_path}")
    plt.close(fig)


# ============================================================
# Folder 3: Q_Y_ortho focused plot (zoomed alpha range)
# ============================================================
def plot_qy_ortho_focused(all_results, sizes, alpha_values, save_path, steps, alpha_max=1.5):
    """
    Plot Q_Y_ortho with focused alpha range for readability.

    Features:
    - X-axis: alpha = 0 to alpha_max (focus on low-alpha region)
    - Y-axis: Auto-scaled to data range (not forced to [0, 1])
    - Shows phase transition more clearly without being squashed to x-axis
    """
    n_configs = len(sizes)
    colors = get_color_palette(n_configs)

    # Filter alpha values
    alpha_filtered = [a for a in alpha_values if a <= alpha_max]
    aL = np.array(alpha_filtered)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_min, y_max = float('inf'), float('-inf')

    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qY_ortho = np.array([results[a]['Q_Y_ortho_mean'] for a in alpha_filtered])
        qY_ortho_err = np.array([results[a]['Q_Y_ortho_std'] for a in alpha_filtered])

        label = f'N={N}, M={M}'
        ax.plot(aL, qY_ortho, marker='o', linewidth=2, markersize=4,
                color=colors[i], label=label)
        ax.fill_between(aL, qY_ortho - qY_ortho_err, qY_ortho + qY_ortho_err,
                        color=colors[i], alpha=0.15)

        y_min = min(y_min, (qY_ortho - qY_ortho_err).min())
        y_max = max(y_max, (qY_ortho + qY_ortho_err).max())

    # Auto-scale Y-axis with padding
    y_padding = (y_max - y_min) * 0.15
    ax.set_ylim(max(-0.05, y_min - y_padding), y_max + y_padding)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax.set_ylabel(r'$Q_Y^{ortho}$', fontsize=14)
    ax.set_title(f'Q_Y (Orthogonal Teacher) - Focused View\n'
                 f'(BiG-AMP, {steps} steps, $\\alpha \\leq {alpha_max}$)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y_ortho focused plot saved: {save_path}")
    plt.close(fig)


def plot_qy_ortho_unobserved_focused(all_results, sizes, alpha_values, save_path, steps, alpha_max=1.5):
    """
    Plot Q_Y_ortho_unobserved with focused alpha range for readability.

    Features:
    - X-axis: alpha = 0 to alpha_max (focus on low-alpha region)
    - Y-axis: Auto-scaled to data range (not forced to [0, 1])
    - Shows phase transition more clearly without being squashed to x-axis
    """
    n_configs = len(sizes)
    colors = get_color_palette(n_configs)

    # Filter alpha values
    alpha_filtered = [a for a in alpha_values if a <= alpha_max]
    aL = np.array(alpha_filtered)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_min, y_max = float('inf'), float('-inf')

    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qY_ortho_unobs = np.array([results[a]['Q_Y_ortho_unobserved_mean'] for a in alpha_filtered])
        qY_ortho_unobs_err = np.array([results[a]['Q_Y_ortho_unobserved_std'] for a in alpha_filtered])

        label = f'N={N}, M={M}'
        ax.plot(aL, qY_ortho_unobs, marker='o', linewidth=2, markersize=4,
                color=colors[i], label=label)
        ax.fill_between(aL, qY_ortho_unobs - qY_ortho_unobs_err, qY_ortho_unobs + qY_ortho_unobs_err,
                        color=colors[i], alpha=0.15)

        y_min = min(y_min, (qY_ortho_unobs - qY_ortho_unobs_err).min())
        y_max = max(y_max, (qY_ortho_unobs + qY_ortho_unobs_err).max())

    # Auto-scale Y-axis with padding
    y_padding = (y_max - y_min) * 0.15
    ax.set_ylim(max(-0.05, y_min - y_padding), y_max + y_padding)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax.set_ylabel(r'$Q_Y^{ortho,unobs}$', fontsize=14)
    ax.set_title(f'Q_Y_unobserved (Orthogonal Teacher) - Focused View\n'
                 f'(BiG-AMP, {steps} steps, $\\alpha \\leq {alpha_max}$)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y_ortho_unobserved focused plot saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def parse_sizes(sizes_str: str) -> List[Tuple[int, int]]:
    """Parse sizes string like '200:50,400:100' into list of (N, M) tuples"""
    sizes = []
    for pair in sizes_str.split(','):
        pair = pair.strip()
        if ':' in pair:
            n, m = pair.split(':')
            sizes.append((int(n.strip()), int(m.strip())))
        else:
            raise ValueError(f"Invalid size format: {pair}. Expected 'N:M'")
    return sizes


def main():
    parser = argparse.ArgumentParser(description='BiG-AMP Orthogonal Teacher Comparison')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated N:M pairs, e.g., "500:50,1000:50,2000:50"')
    parser.add_argument('--steps', type=int, default=MAX_STEPS, help='BiG-AMP steps')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_ALPHA, help='Samples per alpha')
    parser.add_argument('--alpha-step', type=float, default=ALPHA_TILDE_STEP, help='Alpha step size')
    parser.add_argument('--alpha-stop', type=float, default=ALPHA_TILDE_STOP, help='Alpha max value')
    parser.add_argument('--alpha-focus-max', type=float, default=1.5,
                        help='Alpha max for focused Q_ortho plot (folder 3)')
    parser.add_argument('--damping', type=float, default=DAMPING, help='BiG-AMP damping factor')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Parse sizes
    if args.sizes:
        sizes = parse_sizes(args.sizes)
    else:
        sizes = DEFAULT_SIZES

    # Create alpha values
    alpha_values = list(np.arange(ALPHA_TILDE_START, args.alpha_stop + 1e-12, args.alpha_step))

    # Output directory
    if args.output_dir:
        result_dir = Path(args.output_dir)
    else:
        sizes_str = "_".join([f"{n}x{m}" for n, m in sizes])
        result_dir = Path(__file__).parent / "Result_ortho_compare" / sizes_str
    result_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different plot types
    folder1 = result_dir / "folder1_per_config"
    folder2 = result_dir / "folder2_cross_config"
    folder3 = result_dir / "folder3_ortho_focused"
    folder1.mkdir(exist_ok=True)
    folder2.mkdir(exist_ok=True)
    folder3.mkdir(exist_ok=True)

    print("=" * 70)
    print("BiG-AMP ORTHOGONAL TEACHER COMPARISON")
    print("=" * 70)
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Configurations: {sizes}")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({len(alpha_values)} points)")
    print(f"Steps: {args.steps}, Samples: {args.samples}")
    print(f"Resample mask each trial: {RESAMPLE_MASK_EACH_TRIAL}")
    print(f"Alpha focus max (folder 3): {args.alpha_focus_max}")
    print(f"Output: {result_dir}")
    print("=" * 70)

    # Run training for each configuration
    all_results = {}
    total_start = time.time()

    for i, (N, M) in enumerate(sizes):
        print(f"\n[{i+1}/{len(sizes)}] Configuration: N={N}, M={M}")
        config_start = time.time()

        results = run_single_config(N, M, alpha_values, args.steps, args.samples,
                                     args.damping, NOISE_VAR, SEED)
        all_results[(N, M)] = results

        # Generate per-config comparison plot (folder 1)
        plot_per_config_comparison(results, N, M, alpha_values, folder1, args.steps)

        config_time = time.time() - config_start
        print(f"  Completed in {config_time:.1f}s")

    total_time = time.time() - total_start

    # Save results (JSON)
    results_data = {
        'config': {
            'sizes': [[n, m] for n, m in sizes],
            'steps': args.steps,
            'samples_per_alpha': args.samples,
            'resample_mask_each_trial': RESAMPLE_MASK_EACH_TRIAL,
            'damping': args.damping,
            'noise_var': NOISE_VAR,
            'alpha_focus_max': args.alpha_focus_max,
            'total_time': total_time
        },
        'alpha_values': [float(a) for a in alpha_values],
        'results': {f"{n}x{m}": {str(k): v for k, v in results.items()}
                    for (n, m), results in all_results.items()}
    }

    results_path = result_dir / f'ortho_results_steps{args.steps}.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Folder 2: Cross-config comparison plots
    plot_qy_comparison(all_results, sizes, alpha_values,
                       folder2 / f'qy_comparison.png', args.steps)
    plot_qy_unobserved_comparison(all_results, sizes, alpha_values,
                                   folder2 / f'qy_unobserved_comparison.png', args.steps)
    plot_qy_ortho_comparison(all_results, sizes, alpha_values,
                              folder2 / f'qy_ortho_comparison.png', args.steps)
    plot_qy_ortho_unobserved_comparison(all_results, sizes, alpha_values,
                                         folder2 / f'qy_ortho_unobserved_comparison.png', args.steps)

    # Folder 3: Q_Y_ortho focused plots (both full and unobserved)
    plot_qy_ortho_focused(all_results, sizes, alpha_values,
                           folder3 / f'qy_ortho_focused_alpha{args.alpha_focus_max}.png',
                           args.steps, alpha_max=args.alpha_focus_max)
    plot_qy_ortho_unobserved_focused(all_results, sizes, alpha_values,
                                      folder3 / f'qy_ortho_unobserved_focused_alpha{args.alpha_focus_max}.png',
                                      args.steps, alpha_max=args.alpha_focus_max)

    # Summary
    print(f"\n{'='*70}")
    print("ORTHOGONAL TEACHER COMPARISON COMPLETED")
    print(f"{'='*70}")
    print(f"Configurations tested: {len(sizes)}")
    for N, M in sizes:
        print(f"  - N={N}, M={M}")
    print(f"Total time: {total_time:.1f}s")
    print(f"\nOutput structure:")
    print(f"  {result_dir}/")
    print(f"    ├── folder1_per_config/     (4 metrics per config)")
    print(f"    ├── folder2_cross_config/   (Cross-config comparison)")
    print(f"    │   ├── qy_comparison.png")
    print(f"    │   ├── qy_unobserved_comparison.png")
    print(f"    │   ├── qy_ortho_comparison.png")
    print(f"    │   └── qy_ortho_unobserved_comparison.png")
    print(f"    ├── folder3_ortho_focused/  (Zoomed view)")
    print(f"    │   ├── qy_ortho_focused_alpha{args.alpha_focus_max}.png")
    print(f"    │   └── qy_ortho_unobserved_focused_alpha{args.alpha_focus_max}.png")
    print(f"    └── ortho_results_steps{args.steps}.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
