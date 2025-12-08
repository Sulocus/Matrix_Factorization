"""
BiG-AMP Replica Overlap Analysis

Measures the consistency of solutions across S independent replicas
trained on the same mask (same alpha, same graph).

Key difference from Main_bigamp_optimized.py:
- Instead of comparing W_student vs W_teacher, we compute pairwise
  overlap between all S replicas: W_i vs W_j for all i < j
- This reveals whether the algorithm converges to a unique solution
  or multiple local optima

Physical interpretation:
- Replica overlap ≈ 1: All replicas find the same solution
- Replica overlap < 1: Multiple distinct local optima exist

Usage:
    python Main_bigamp_replica_overlap.py                    # Default S=100
    python Main_bigamp_replica_overlap.py --samples 50       # Custom replica count
    python Main_bigamp_replica_overlap.py --n1 400 --m 50    # Custom matrix size
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
from collections import deque
import itertools

# ============================================================
# Default Parameters
# ============================================================
N1 = 200
N2 = 200
M = 50

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 4
ALPHA_TILDE_STEP = 0.1

# BiG-AMP parameters
DAMPING = 0.5
NOISE_VAR = 1e-10
MAX_STEPS = 1000

# Number of independent replicas per alpha (all share same mask)
SAMPLES_PER_ALPHA = 100
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

# Result directory - use Result_replica for replica overlap analysis
RESULT_DIR = Path(__file__).parent / "Result_replica" / f"{N1}_{N2}_{M}"


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


@torch.no_grad()
def compute_replica_overlap(W_all, X_all):
    """
    Compute pairwise Gram overlap between S replicas.

    Args:
        W_all: (S, N1, M) - S replicas of W
        X_all: (S, M, N2) - S replicas of X

    Returns:
        Q_W_replica: Mean W overlap across all pairs (raw cosine)
        Q_X_replica: Mean X overlap across all pairs (raw cosine)
        Q_Y_replica: Mean Y overlap across all pairs
        Q_W_replica_norm: Mean W overlap with baseline correction (same as teacher-student)
        Q_X_replica_norm: Mean X overlap with baseline correction (same as teacher-student)
        Plus standard deviations for all
    """
    S = W_all.shape[0]

    # Compute all Y = W @ X
    Y_all = torch.bmm(W_all, X_all)  # (S, N1, N2)

    # Collect pairwise overlaps
    Q_W_list = []
    Q_X_list = []
    Q_Y_list = []
    Q_W_norm_list = []  # With baseline correction (same as teacher-student)
    Q_X_norm_list = []  # With baseline correction (same as teacher-student)

    for i in range(S):
        for j in range(i + 1, S):
            # W overlap (Gram matrix cosine - raw)
            Q_W_list.append(gram_overlap_cosine(W_all[i], W_all[j], use_left=True))

            # X overlap (Gram matrix cosine - raw)
            Q_X_list.append(gram_overlap_cosine(X_all[i], X_all[j], use_left=False))

            # W overlap with baseline correction (same normalization as teacher-student)
            Q_W_norm_list.append(gram_overlap_zero_to_one(W_all[i], W_all[j], use_left=True))

            # X overlap with baseline correction (same normalization as teacher-student)
            Q_X_norm_list.append(gram_overlap_zero_to_one(X_all[i], X_all[j], use_left=False))

            # Y overlap (direct cosine similarity)
            Y_i_flat = Y_all[i].flatten()
            Y_j_flat = Y_all[j].flatten()
            Q_Y = float((Y_i_flat * Y_j_flat).sum() /
                       (Y_i_flat.norm() * Y_j_flat.norm() + 1e-12))
            Q_Y_list.append(Q_Y)

    # Number of pairs = S*(S-1)/2
    Q_W_mean = float(np.mean(Q_W_list))
    Q_X_mean = float(np.mean(Q_X_list))
    Q_Y_mean = float(np.mean(Q_Y_list))
    Q_W_norm_mean = float(np.mean(Q_W_norm_list))
    Q_X_norm_mean = float(np.mean(Q_X_norm_list))

    Q_W_std = float(np.std(Q_W_list, ddof=1)) if len(Q_W_list) > 1 else 0.0
    Q_X_std = float(np.std(Q_X_list, ddof=1)) if len(Q_X_list) > 1 else 0.0
    Q_Y_std = float(np.std(Q_Y_list, ddof=1)) if len(Q_Y_list) > 1 else 0.0
    Q_W_norm_std = float(np.std(Q_W_norm_list, ddof=1)) if len(Q_W_norm_list) > 1 else 0.0
    Q_X_norm_std = float(np.std(Q_X_norm_list, ddof=1)) if len(Q_X_norm_list) > 1 else 0.0

    return {
        'Q_W_replica_mean': Q_W_mean,
        'Q_X_replica_mean': Q_X_mean,
        'Q_Y_replica_mean': Q_Y_mean,
        'Q_W_replica_std': Q_W_std,
        'Q_X_replica_std': Q_X_std,
        'Q_Y_replica_std': Q_Y_std,
        # Normalized replica overlap (same baseline correction as teacher-student)
        'Q_W_replica_norm_mean': Q_W_norm_mean,
        'Q_X_replica_norm_mean': Q_X_norm_mean,
        'Q_W_replica_norm_std': Q_W_norm_std,
        'Q_X_replica_norm_std': Q_X_norm_std,
        'n_pairs': len(Q_W_list)
    }


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
# Simplified Training for Replica Overlap
# ============================================================
def train_replicas_single_alpha(Y_teacher, alpha, steps, S, seed,
                                 damping=0.5, noise_var=1e-6):
    """
    Train S replicas for a single alpha value with the SAME mask.

    All replicas share the same observation mask but have independent
    random initializations. This allows measuring solution consistency.
    """
    device = Y_teacher.device
    N1, N2 = Y_teacher.shape
    alpha_scale = 1.0 / (M ** 0.5)
    scale = 1.0 / (M ** 0.5)

    # Generate ONE mask for all replicas (RESAMPLE_MASK = False)
    A, _ = sample_mask(N1, N2, M, alpha, device, seed=seed)
    A = A.unsqueeze(0)  # (1, N1, N2) - broadcast to all replicas

    # Initialize S independent replicas with DIFFERENT random seeds
    torch.manual_seed(seed + 10000)
    w_hat = torch.randn((S, N1, M), device=device) * scale
    x_hat = torch.randn((S, M, N2), device=device) * scale
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    for _ in tqdm(range(steps), desc=f"BiG-AMP α={alpha:.2f}", leave=False, mininterval=1.0):
        # Forward
        z_hat = alpha_scale * torch.bmm(w_hat, x_hat)
        w_sq = w_hat ** 2
        x_sq = x_hat ** 2
        p_var = (alpha_scale ** 2) * (torch.bmm(w_sq, x_var) + torch.bmm(w_var, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher.unsqueeze(0) - z_hat) * A
        s = residual / V

        # Update W
        tau_W = (alpha_scale ** 2) * torch.bmm(A.expand(S, -1, -1) / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.bmm(s, x_hat.transpose(-2, -1))
        w_hat_new = w_hat + w_var_new * r_W
        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        # Update X
        z_hat2 = alpha_scale * torch.bmm(w_hat, x_hat)
        w_sq2 = w_hat ** 2
        p_var2 = (alpha_scale ** 2) * (torch.bmm(w_sq2, x_var) + torch.bmm(w_var, x_sq))
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        residual2 = (Y_teacher.unsqueeze(0) - z_hat2) * A
        s2 = residual2 / V2

        tau_X = (alpha_scale ** 2) * torch.bmm(w_sq2.transpose(-2, -1), A.expand(S, -1, -1) / V2)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new = 1.0 / (M + tau_X)
        r_X = alpha_scale * torch.bmm(w_hat.transpose(-2, -1), s2)
        x_hat_new = x_hat + x_var_new * r_X
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = torch.clamp(damping * x_var + (1 - damping) * x_var_new, min=1e-8, max=1.0)

    return w_hat, x_hat


# ============================================================
# Main Training Function
# ============================================================
def run_replica_overlap_analysis(alpha_values, steps, S):
    """
    Run replica overlap analysis for all alpha values.

    For each alpha:
    1. Generate ONE mask (shared by all replicas)
    2. Train S independent replicas
    3. Compute pairwise overlap between all replica pairs
    """
    set_seed(SEED)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)
    n_pairs = S * (S - 1) // 2

    print(f"\n{'='*70}")
    print("BiG-AMP REPLICA OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({num_alphas} points)")
    print(f"Steps: {steps}")
    print(f"Replicas per alpha: {S}")
    print(f"Pairs per alpha: {n_pairs} = {S}*({S}-1)/2")
    print(f"{'='*70}\n")

    all_results = {}
    total_start = time.time()

    for i, alpha in enumerate(alpha_values):
        alpha_seed = SEED + int(alpha * 1000)
        print(f"[{i+1}/{num_alphas}] Alpha = {alpha:.2f}")

        # Train S replicas with same mask
        W, X = train_replicas_single_alpha(Y_teacher, alpha, steps, S, alpha_seed,
                                            damping=DAMPING, noise_var=NOISE_VAR)

        # Compute replica overlap (W_i vs W_j, X_i vs X_j for all i<j)
        metrics = compute_replica_overlap(W, X)

        # Compute overlap with teacher for reference
        Q_Y_teacher_list = []
        Q_W_prime_list = []
        Q_X_prime_list = []
        for s in range(S):
            # Q_Y (teacher vs student)
            Yp = W[s] @ X[s]
            Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                       (Y_teacher.norm() * Yp.norm() + 1e-12))
            Q_Y_teacher_list.append(Q_Y)

            # Q_W' and Q_X' (normalized, with baseline correction)
            Q_W_prime_list.append(gram_overlap_zero_to_one(W[s], Wt, use_left=True))
            Q_X_prime_list.append(gram_overlap_zero_to_one(X[s], Xt, use_left=False))

        metrics['Q_Y_teacher_mean'] = float(np.mean(Q_Y_teacher_list))
        metrics['Q_Y_teacher_std'] = float(np.std(Q_Y_teacher_list, ddof=1))
        metrics['Q_W_prime_mean'] = float(np.mean(Q_W_prime_list))
        metrics['Q_W_prime_std'] = float(np.std(Q_W_prime_list, ddof=1))
        metrics['Q_X_prime_mean'] = float(np.mean(Q_X_prime_list))
        metrics['Q_X_prime_std'] = float(np.std(Q_X_prime_list, ddof=1))

        all_results[float(alpha)] = metrics

        # Print progress
        print(f"    Q_Y_replica = {metrics['Q_Y_replica_mean']:.4f} ± {metrics['Q_Y_replica_std']:.4f}")
        print(f"    Q_Y_teacher = {metrics['Q_Y_teacher_mean']:.4f} ± {metrics['Q_Y_teacher_std']:.4f}")

        del W, X
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")

    return all_results, total_time


# ============================================================
# Visualization
# ============================================================
def plot_replica_results(results, alpha_values, save_path, S):
    """Generate replica overlap result plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors
    REPLICA_COLOR = '#2563eb'    # Blue for replica overlap
    TEACHER_COLOR = '#dc2626'    # Red for teacher overlap
    W_COLOR = '#16a34a'          # Green for W
    X_COLOR = '#9333ea'          # Purple for X

    # Plot 1: Q_Y_replica vs Q_Y_teacher
    ax1 = axes[0, 0]
    qy_replica = [results[a]['Q_Y_replica_mean'] for a in alpha_values]
    qy_replica_std = [results[a]['Q_Y_replica_std'] for a in alpha_values]
    qy_teacher = [results[a]['Q_Y_teacher_mean'] for a in alpha_values]
    qy_teacher_std = [results[a]['Q_Y_teacher_std'] for a in alpha_values]

    ax1.errorbar(alpha_values, qy_replica, yerr=qy_replica_std, fmt='o-',
                 color=REPLICA_COLOR, capsize=3, markersize=6, linewidth=2,
                 label='Q_Y (replica)')
    ax1.errorbar(alpha_values, qy_teacher, yerr=qy_teacher_std, fmt='s--',
                 color=TEACHER_COLOR, capsize=3, markersize=5, linewidth=1.5,
                 label='Q_Y (teacher)', alpha=0.7)
    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax1.set_ylabel('Q_Y', fontsize=12)
    ax1.set_title('Y Overlap: Replica vs Teacher', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Q_W_replica and Q_X_replica
    ax2 = axes[0, 1]
    qw_replica = [results[a]['Q_W_replica_mean'] for a in alpha_values]
    qx_replica = [results[a]['Q_X_replica_mean'] for a in alpha_values]
    ax2.plot(alpha_values, qw_replica, 'o-', color=W_COLOR, markersize=6,
             linewidth=2, label='Q_W (replica)')
    ax2.plot(alpha_values, qx_replica, 's-', color=X_COLOR, markersize=6,
             linewidth=2, label='Q_X (replica)')
    ax2.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax2.set_ylabel('Q (cosine)', fontsize=12)
    ax2.set_title('W and X Replica Overlap', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Plot 3: Q_Y_replica only (main result)
    ax3 = axes[1, 0]
    ax3.errorbar(alpha_values, qy_replica, yerr=qy_replica_std, fmt='o-',
                 color=REPLICA_COLOR, capsize=3, markersize=8, linewidth=2.5)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect consistency')
    ax3.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax3.set_ylabel('Q_Y (replica)', fontsize=12)
    ax3.set_title('Replica Consistency (Main Result)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)

    # Plot 4: Standard deviation of replica overlap
    ax4 = axes[1, 1]
    ax4.plot(alpha_values, qy_replica_std, 'o-', color=REPLICA_COLOR,
             markersize=6, linewidth=2, label='Q_Y std')
    ax4.plot(alpha_values, [results[a]['Q_W_replica_std'] for a in alpha_values],
             's-', color=W_COLOR, markersize=5, linewidth=1.5, label='Q_W std', alpha=0.7)
    ax4.plot(alpha_values, [results[a]['Q_X_replica_std'] for a in alpha_values],
             '^-', color=X_COLOR, markersize=5, linewidth=1.5, label='Q_X std', alpha=0.7)
    ax4.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax4.set_ylabel('Standard Deviation', fontsize=12)
    ax4.set_title('Overlap Variance Across Pairs', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    n_pairs = S * (S - 1) // 2
    plt.suptitle(f'Replica Overlap: {N1}×{N2}, M={M}, S={S} replicas, {n_pairs} pairs',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved: {save_path}")
    plt.close(fig)


def plot_comparison(results, alpha_values, save_path):
    """
    Generate comparison plot: Teacher-Student vs Replica-Replica overlap.

    Shows Q_W and Q_X together in one plot, comparing:
    - Teacher-student overlap (normalized)
    - Replica-replica overlap (raw cosine, for reference)
    """
    aL = np.array(alpha_values)

    # Teacher-student overlap (normalized)
    qW_prime_mu = np.array([results[a].get('Q_W_prime_mean', 0.0) for a in alpha_values])
    qX_prime_mu = np.array([results[a].get('Q_X_prime_mean', 0.0) for a in alpha_values])

    # Replica-replica overlap (raw cosine)
    qW_rep_mu = np.array([results[a]['Q_W_replica_mean'] for a in alpha_values])
    qX_rep_mu = np.array([results[a]['Q_X_replica_mean'] for a in alpha_values])

    fig, ax = plt.subplots(figsize=(10, 7))

    # Teacher-student (normalized)
    ax.plot(aL, qW_prime_mu, marker='o', linewidth=1.5, markersize=4,
            color='#9467bd', label="Q_W' (teacher-student)", alpha=0.7)
    ax.plot(aL, qX_prime_mu, marker='v', linewidth=1.5, markersize=4,
            color='#8c564b', label="Q_X' (teacher-student)", alpha=0.7)

    # Replica-replica (raw cosine)
    ax.plot(aL, qW_rep_mu, marker='s', linewidth=1.5, markersize=4,
            color='#2ca02c', label="Q_W_replica (replica-replica)", alpha=0.7)
    ax.plot(aL, qX_rep_mu, marker='^', linewidth=1.5, markersize=4,
            color='#17becf', label="Q_X_replica (replica-replica)", alpha=0.7)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=13)
    ax.set_ylabel('Overlap', fontsize=13)
    ax.set_title('Teacher-Student vs Replica-Replica Overlap\n(Note: different normalization methods)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved: {save_path}")
    plt.close(fig)


def plot_normalized_comparison(results, alpha_values, save_path, N1, N2, M):
    """
    Generate comparison plot with SAME normalization for both teacher-student and replica-replica.

    Both use baseline correction: Q' = (q - b) / (1 - b) where b = m/(m+n+1)
    This allows fair comparison between teacher-student and replica-replica overlap.
    """
    aL = np.array(alpha_values)

    # Teacher-student overlap (normalized with baseline correction)
    qW_teacher = np.array([results[a].get('Q_W_prime_mean', 0.0) for a in alpha_values])
    qX_teacher = np.array([results[a].get('Q_X_prime_mean', 0.0) for a in alpha_values])

    # Replica-replica overlap (ALSO normalized with same baseline correction)
    qW_replica = np.array([results[a].get('Q_W_replica_norm_mean', 0.0) for a in alpha_values])
    qX_replica = np.array([results[a].get('Q_X_replica_norm_mean', 0.0) for a in alpha_values])

    # Compute baseline values for display
    b_W = M / (M + N1 + 1)
    b_X = M / (M + N2 + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Q_W comparison
    ax1 = axes[0]
    ax1.plot(aL, qW_teacher, 'o-', linewidth=2, markersize=5,
             color='#dc2626', label="Q_W' (teacher-student)")
    ax1.plot(aL, qW_replica, 's--', linewidth=2, markersize=5,
             color='#2563eb', label="Q_W' (replica-replica)")
    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=13)
    ax1.set_ylabel("Q_W' (normalized)", fontsize=13)
    ax1.set_title(f"W Overlap Comparison\n(baseline b = {b_W:.4f})",
                  fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='lower right')

    # Right plot: Q_X comparison
    ax2 = axes[1]
    ax2.plot(aL, qX_teacher, 'o-', linewidth=2, markersize=5,
             color='#dc2626', label="Q_X' (teacher-student)")
    ax2.plot(aL, qX_replica, 's--', linewidth=2, markersize=5,
             color='#2563eb', label="Q_X' (replica-replica)")
    ax2.set_xlabel(r'$\tilde{\alpha}$', fontsize=13)
    ax2.set_ylabel("Q_X' (normalized)", fontsize=13)
    ax2.set_title(f"X Overlap Comparison\n(baseline b = {b_X:.4f})",
                  fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='lower right')

    plt.suptitle(f"Fair Comparison: Same Normalization (baseline correction)\n"
                 f"Matrix: {N1}×{N2}, M={M}",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Normalized comparison plot saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    global N1, N2, M, ALPHA_TILDE_STEP, MAX_STEPS, SAMPLES_PER_ALPHA, RESULT_DIR, DAMPING

    parser = argparse.ArgumentParser(description='BiG-AMP Replica Overlap Analysis')
    parser.add_argument('--n1', type=int, default=N1, help='Matrix N1 dimension')
    parser.add_argument('--n2', type=int, default=None, help='Matrix N2 dimension (default: same as N1)')
    parser.add_argument('--m', type=int, default=M, help='Latent dimension M')
    parser.add_argument('--steps', type=int, default=MAX_STEPS, help='BiG-AMP steps')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_ALPHA, help='Number of replicas per alpha')
    parser.add_argument('--alpha-step', type=float, default=ALPHA_TILDE_STEP, help='Alpha step size')
    parser.add_argument('--alpha-stop', type=float, default=ALPHA_TILDE_STOP, help='Alpha max value')
    parser.add_argument('--damping', type=float, default=DAMPING, help='BiG-AMP damping factor')
    args = parser.parse_args()

    # Apply args
    N1 = args.n1
    N2 = args.n2 if args.n2 else args.n1
    M = args.m
    MAX_STEPS = args.steps
    SAMPLES_PER_ALPHA = args.samples
    ALPHA_TILDE_STEP = args.alpha_step
    DAMPING = args.damping

    RESULT_DIR = Path(__file__).parent / "Result_replica" / f"{N1}_{N2}_{M}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    alpha_values = np.arange(ALPHA_TILDE_START, args.alpha_stop + 1e-12, ALPHA_TILDE_STEP)

    # Run replica overlap analysis
    results, total_time = run_replica_overlap_analysis(alpha_values, MAX_STEPS, SAMPLES_PER_ALPHA)

    # Save results
    n_pairs = SAMPLES_PER_ALPHA * (SAMPLES_PER_ALPHA - 1) // 2
    results_data = {
        'config': {
            'N1': N1, 'N2': N2, 'M': M,
            'steps': MAX_STEPS,
            'replicas_per_alpha': SAMPLES_PER_ALPHA,
            'pairs_per_alpha': n_pairs,
            'damping': DAMPING,
            'noise_var': NOISE_VAR,
            'total_time': total_time
        },
        'alpha_values': [float(a) for a in alpha_values],
        'results': {str(k): v for k, v in results.items()}
    }

    results_path = RESULT_DIR / f'replica_overlap_S{SAMPLES_PER_ALPHA}_steps{MAX_STEPS}.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved: {results_path}")

    # Plot 1: Replica overlap results (original 4-panel plot)
    plot_path1 = RESULT_DIR / f'replica_overlap_S{SAMPLES_PER_ALPHA}_steps{MAX_STEPS}.png'
    plot_replica_results(results, [float(a) for a in alpha_values], plot_path1, SAMPLES_PER_ALPHA)

    # Plot 2: Comparison plot (teacher-student vs replica-replica, different normalization)
    plot_path2 = RESULT_DIR / f'replica_comparison_S{SAMPLES_PER_ALPHA}_steps{MAX_STEPS}.png'
    plot_comparison(results, [float(a) for a in alpha_values], plot_path2)

    # Plot 3: Fair comparison with SAME normalization (baseline correction for both)
    plot_path3 = RESULT_DIR / f'replica_normalized_comparison_S{SAMPLES_PER_ALPHA}_steps{MAX_STEPS}.png'
    plot_normalized_comparison(results, [float(a) for a in alpha_values], plot_path3, N1, N2, M)

    # Summary
    print(f"\n{'='*70}")
    print("REPLICA OVERLAP ANALYSIS COMPLETED")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Replicas: {SAMPLES_PER_ALPHA}, Pairs: {n_pairs}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results: {results_path}")
    print(f"Plot 1 (replica): {plot_path1}")
    print(f"Plot 2 (comparison, different norm): {plot_path2}")
    print(f"Plot 3 (comparison, same norm): {plot_path3}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
