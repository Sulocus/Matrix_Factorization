"""
BiG-AMP Multi-Size Comparison

Runs BiG-AMP training for multiple (N, M) configurations and plots
results on the same figure for comparison.

Features:
1. Accept multiple (N, M) pairs (assumes N1=N2=N)
2. Plot Q_Y from different sizes on one figure
3. Plot Q_W', Q_X' from different sizes on another figure
4. Automatic color assignment for different configurations

Usage:
    # Default configurations
    python bigamp_multi_size.py

    # Custom configurations (comma-separated N:M pairs)
    python bigamp_multi_size.py --sizes "200:50,400:100,800:200"

    # With custom steps
    python bigamp_multi_size.py --sizes "200:50,400:100" --steps 500
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
    (500,50),
    (1000,50),
    (1500,50),
    (2000,50),
]

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 3
ALPHA_TILDE_STEP = 0.05

# BiG-AMP parameters
DAMPING = 0.5
NOISE_VAR = 1e-10
MAX_STEPS = 10000

SAMPLES_PER_ALPHA = 4
RESAMPLE_MASK_EACH_TRIAL = True  # True: each trial gets different mask, False: all trials share same mask
SEED = 42

# ============================================================
# Graph Generation Configuration
# ============================================================
USE_BIREGULAR_GRAPH = False  # Whether to generate uniform graph (bi-regular graph)
# True:  Use Dinic algorithm to generate strict bi-regular/near-regular graph (uniform degree)
# False: Use pure random method to quickly generate random graph (entirely on GPU, supports any N1≠N2)

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
    """Create teacher model W_true and X_true"""
    torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale
    return W, X


def sample_pairs_random_gpu(N1, N2, C, device, seed=None):
    """
    Pure random mask generation (entirely on GPU, supports any N1≠N2)

    Approach:
    1. Map all positions of N1×N2 matrix to 1D index [0, N1*N2-1]
    2. Use torch.randperm to randomly shuffle all positions on GPU (automatically no duplicate edges)
    3. Take first C positions as observation points
    4. Restore 1D index to (i,j) coordinates

    Parameters:
        N1, N2: Matrix dimensions
        C: Number of edges needed (number of observation points)
        device: Device
        seed: Random seed

    Returns:
        i_idx, j_idx, C: Row indices, column indices, and edge count of observation points
    """
    if seed is not None:
        torch.manual_seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"Requested edge count C={C} exceeds matrix total size {N1}×{N2}={total}")

    if C == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    # Randomly shuffle all position indices on GPU
    idx = torch.randperm(total, device=device)[:C]  # Take first C

    # Restore 1D index to 2D coordinates
    i_idx = idx // N2  # Row index
    j_idx = idx % N2   # Column index

    return i_idx, j_idx, C


def sample_pairs_biregular_exact(N1, N2, M, alpha_tilde_left, device, seed=None):
    """
    Main graph generation function (choose method based on USE_BIREGULAR_GRAPH switch)

    USE_BIREGULAR_GRAPH=True:
      Strict bi-regular/near-regular construction (Dinic algorithm):
      - Fast path (N1==N2, divisible) remains unchanged.
      - General path (Dinic) forcibly introduces randomness by shuffling the order
        of adding L->R edges, eliminating the construction gap between
        rem=0 (deterministic) and rem>0 (randomness).

    USE_BIREGULAR_GRAPH=False:
      Pure random method (entirely on GPU, fast and supports any N1≠N2)
    """
    deg_left = int(round(alpha_tilde_left * M))
    deg_left = max(0, min(deg_left, N2))

    # Calculate total edges
    total_edges = N1 * deg_left

    # ============================================================
    # Choose graph generation method based on switch
    # ============================================================
    if not USE_BIREGULAR_GRAPH:
        # Use pure random method (GPU fast generation)
        return sample_pairs_random_gpu(N1, N2, total_edges, device, seed)

    # ============================================================
    # Below is the original Dinic bi-regular graph generation algorithm
    # ============================================================
    if deg_left == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    if deg_left > N2:
        raise RuntimeError(f"deg_left={deg_left} > N2={N2}, infeasible")
    deg_right_exact = total_edges // N2
    if total_edges % N2 == 0 and deg_right_exact > N1:
        raise RuntimeError(f"deg_right={deg_right_exact} > N1={N1}, infeasible")

    # ---------- Dinic max flow algorithm, generate random bi-regular graph ----------

    # 1. Uniformly create random number generator, ensure all random operations from same seed
    if seed is not None:
        rng = np.random.RandomState(seed + 12345 + int(round(alpha_tilde_left * 1e6)))
    else:
        rng = np.random.RandomState()  # Use non-fixed seed

    base = total_edges // N2
    rem = total_edges % N2
    right_target = np.full(N2, base, dtype=int)
    if rem > 0:
        idx = np.arange(N2)
        rng.shuffle(idx)
        right_target[idx[:rem]] += 1

    if right_target.max() > N1:
        raise RuntimeError(f"Some right node target degree {right_target.max()} > N1={N1}, infeasible")

    # Dinic implementation (lightweight version)
    class Dinic:
        __slots__ = ("n", "g", "lvl", "it")
        def __init__(self, n):
            self.n = n
            self.g = [[] for _ in range(n)]
        def add_edge(self, u, v, cap):
            self.g[u].append([v, cap, len(self.g[v])])
            self.g[v].append([u, 0,   len(self.g[u]) - 1])
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

    # Core modification: Randomize L->R edge addition order to forcibly introduce randomness
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
        raise RuntimeError(f"maxflow only got {f}/{total_edges}, degree sequence infeasible or implementation error")

    i_list, j_list = [], []
    for i in range(N1):
        u = L_off + i
        for v, cap, rev in din.g[u]:
            if (R_off <= v < R_off + N2):
                if din.g[v][rev][1] > 0:
                    j = v - R_off
                    i_list.append(i)
                    j_list.append(j)

    assert len(i_list) == total_edges, "Extracted edge count not equal to total_edges"
    i_np = np.array(i_list, dtype=int)
    j_np = np.array(j_list, dtype=int)
    left_deg = np.bincount(i_np, minlength=N1)
    right_deg = np.bincount(j_np, minlength=N2)
    assert np.all(left_deg == deg_left), "Left degree inconsistent (should all be equal)"
    assert np.all(right_deg == right_target), "Right degree doesn't match target"
    assert len(set(zip(i_np, j_np))) == len(i_np), "Duplicate edges exist"

    i_idx = torch.tensor(i_list, dtype=torch.long, device=device)
    j_idx = torch.tensor(j_list, dtype=torch.long, device=device)
    C = len(i_list)
    return i_idx, j_idx, C


def sample_mask(N1, N2, M, alpha, device, seed=None):
    """
    Generate observation mask using the configured graph generation method

    Returns:
        mask: Binary observation mask (N1, N2)
        c: Expected degree per left node (alpha * M)
    """
    c = alpha * M

    # Use graph generation function
    i_idx, j_idx, C = sample_pairs_biregular_exact(N1, N2, M, alpha, device, seed)

    # Convert edge list to mask matrix
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
    """
    Compute normalized Gram overlap in [0, 1] range with baseline correction.
    """
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
        # Generate S different masks (S, N1, N2)
        A = torch.zeros((S, N1, N2), device=device)
        for s in range(S):
            mask_seed = seed + s * 10000
            A_s, _ = sample_mask(N1, N2, M, alpha, device, seed=mask_seed)
            A[s] = A_s
    else:
        # Generate one mask, broadcast to all trials (1, N1, N2)
        A, _ = sample_mask(N1, N2, M, alpha, device, seed=seed)
        A = A.unsqueeze(0)

    torch.manual_seed(seed + 10000)
    w_hat = torch.randn((S, N1, M), device=device) * scale
    x_hat = torch.randn((S, M, N2), device=device) * scale
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    for _ in tqdm(range(steps), desc=f"BiG-AMP α={alpha:.2f}", leave=False, mininterval=1.0):
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
# Evaluation
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
# Training for Single Configuration
# ============================================================
def run_single_config(N, M, alpha_values, steps, S, damping, noise_var, seed):
    """Run training for a single (N, M) configuration"""
    N1 = N2 = N
    set_seed(seed)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=seed)
    Y_teacher = Wt @ Xt

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

            if RESAMPLE_MASK_EACH_TRIAL:
                # Generate S different masks for each alpha (num_alphas, S, N1, N2)
                A_all = torch.zeros((batch_size, S, N1, N2), device=DEVICE)
                for i, alpha in enumerate(batch_alphas):
                    for s in range(S):
                        mask_seed = seed + int(alpha * 1000) + s * 10000
                        A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
                        A_all[i, s] = A
            else:
                # Generate one mask per alpha, broadcast to all trials (num_alphas, 1, N1, N2)
                A_all = torch.zeros((batch_size, 1, N1, N2), device=DEVICE)
                for i, alpha in enumerate(batch_alphas):
                    mask_seed = seed + int(alpha * 1000)
                    A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
                    A_all[i, 0] = A

            W, X = train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, batch_alphas, steps, S, M,
                                          damping=damping, noise_var=noise_var)
            # Pass A_all for Q_Y_unobserved calculation
            batch_results = evaluate_batch(W, X, Wt, Xt, Y_teacher, batch_alphas, S, A_all=A_all)
            all_results.update(batch_results)

            del A_all, W, X
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    else:
        # Sequential mode
        for alpha in alpha_values:
            alpha_seed = seed + int(alpha * 1000)
            W, X = train_bigamp_single(Wt, Xt, Y_teacher, alpha, steps, S, M, alpha_seed,
                                        damping=damping, noise_var=noise_var,
                                        resample_mask=RESAMPLE_MASK_EACH_TRIAL)

            # Regenerate masks for Q_Y_unobserved evaluation
            if RESAMPLE_MASK_EACH_TRIAL:
                A = torch.zeros((S, N1, N2), device=DEVICE)
                for s in range(S):
                    mask_seed = alpha_seed + s * 10000
                    A_s, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
                    A[s] = A_s
            else:
                A, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=alpha_seed)
                A = A.unsqueeze(0)  # (1, N1, N2)

            metrics = evaluate_single(W, X, Wt, Xt, Y_teacher, S, A=A)
            all_results[float(alpha)] = metrics

            del W, X, A
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    return all_results


# ============================================================
# Visualization
# ============================================================
def get_color_palette(n_colors):
    """Get a list of distinct colors for plotting"""
    # High-contrast color palette
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
    # If need more colors, cycle through
    return [colors[i % len(colors)] for i in range(n_colors)]


def plot_qy_comparison(all_results, sizes, alpha_values, save_path, steps):
    """
    Plot Q_Y comparison across different (N, M) configurations
    """
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
    ax.set_title(f'Q_Y Comparison Across Different Matrix Sizes\n(BiG-AMP, {steps} steps)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y comparison plot saved: {save_path}")
    plt.close(fig)


def plot_qwx_comparison(all_results, sizes, alpha_values, save_path, steps):
    """
    Plot Q_W' and Q_X' comparison across different (N, M) configurations
    """
    n_configs = len(sizes)
    colors = get_color_palette(n_configs)
    aL = np.array(alpha_values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Q_W' comparison
    ax1 = axes[0]
    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qW_mu = np.array([results[a]['Q_W_prime_mean'] for a in alpha_values])

        label = f'N={N}, M={M}'
        ax1.plot(aL, qW_mu, marker='o', linewidth=2, markersize=4,
                 color=colors[i], label=label)

    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax1.set_ylabel(r"$Q_W'$", fontsize=14)
    ax1.set_title(r"$Q_W'$ Comparison", fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=10, loc='lower right')

    # Right: Q_X' comparison
    ax2 = axes[1]
    for i, (N, M) in enumerate(sizes):
        results = all_results[(N, M)]
        qX_mu = np.array([results[a]['Q_X_prime_mean'] for a in alpha_values])

        label = f'N={N}, M={M}'
        ax2.plot(aL, qX_mu, marker='s', linewidth=2, markersize=4,
                 color=colors[i], label=label)

    ax2.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax2.set_ylabel(r"$Q_X'$", fontsize=14)
    ax2.set_title(r"$Q_X'$ Comparison", fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=10, loc='lower right')

    plt.suptitle(f"Gram Overlap Comparison Across Different Matrix Sizes\n(BiG-AMP, {steps} steps)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_W'/Q_X' comparison plot saved: {save_path}")
    plt.close(fig)


def plot_qy_unobserved_comparison(all_results, sizes, alpha_values, save_path, steps):
    """
    Plot Q_Y_unobserved comparison across different (N, M) configurations

    This is the generalization metric - overlap only on unobserved positions.
    Uses same style as plot_qy_comparison.
    """
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
    ax.set_title(f'Q_Y_unobserved (Generalization) Comparison Across Different Matrix Sizes\n(BiG-AMP, {steps} steps)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y_unobserved comparison plot saved: {save_path}")
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
    parser = argparse.ArgumentParser(description='BiG-AMP Multi-Size Comparison')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated N:M pairs, e.g., "200:50,400:100,800:200"')
    parser.add_argument('--steps', type=int, default=MAX_STEPS, help='BiG-AMP steps')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_ALPHA, help='Samples per alpha')
    parser.add_argument('--alpha-step', type=float, default=ALPHA_TILDE_STEP, help='Alpha step size')
    parser.add_argument('--alpha-stop', type=float, default=ALPHA_TILDE_STOP, help='Alpha max value')
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

    # Output directory - use Result_compareNM for multi-size comparison results
    if args.output_dir:
        result_dir = Path(args.output_dir)
    else:
        # Use sizes string for directory name
        sizes_str = "_".join([f"{n}x{m}" for n, m in sizes])
        result_dir = Path(__file__).parent / "Result_compareNM" / sizes_str
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BiG-AMP MULTI-SIZE COMPARISON")
    print("=" * 70)
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Configurations: {sizes}")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({len(alpha_values)} points)")
    print(f"Steps: {args.steps}, Samples: {args.samples}")
    print(f"Resample mask each trial: {RESAMPLE_MASK_EACH_TRIAL}")
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

        config_time = time.time() - config_start
        print(f"  Completed in {config_time:.1f}s")

    total_time = time.time() - total_start

    # Save results
    results_data = {
        'config': {
            'sizes': [[n, m] for n, m in sizes],
            'steps': args.steps,
            'samples_per_alpha': args.samples,
            'resample_mask_each_trial': RESAMPLE_MASK_EACH_TRIAL,
            'damping': args.damping,
            'noise_var': NOISE_VAR,
            'total_time': total_time
        },
        'alpha_values': [float(a) for a in alpha_values],
        'results': {f"{n}x{m}": {str(k): v for k, v in results.items()}
                    for (n, m), results in all_results.items()}
    }

    results_path = result_dir / f'multi_size_results_steps{args.steps}.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Plot 1: Q_Y comparison
    plot_path1 = result_dir / f'multi_size_qy_steps{args.steps}.png'
    plot_qy_comparison(all_results, sizes, alpha_values, plot_path1, args.steps)

    # Plot 2: Q_W' and Q_X' comparison
    plot_path2 = result_dir / f'multi_size_qwx_steps{args.steps}.png'
    plot_qwx_comparison(all_results, sizes, alpha_values, plot_path2, args.steps)

    # Plot 3: Q_Y_unobserved comparison (generalization)
    plot_path3 = result_dir / f'multi_size_qy_unobserved_steps{args.steps}.png'
    plot_qy_unobserved_comparison(all_results, sizes, alpha_values, plot_path3, args.steps)

    # Summary
    print(f"\n{'='*70}")
    print("MULTI-SIZE COMPARISON COMPLETED")
    print(f"{'='*70}")
    print(f"Configurations tested: {len(sizes)}")
    for N, M in sizes:
        print(f"  - N={N}, M={M}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results: {results_path}")
    print(f"Plot 1 (Q_Y): {plot_path1}")
    print(f"Plot 2 (Q_W'/Q_X'): {plot_path2}")
    print(f"Plot 3 (Q_Y_unobserved): {plot_path3}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
