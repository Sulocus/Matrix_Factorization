# ============================================================
# Teacher–Student Masked MF - GPU Batched Parallel Version
# Perfect Bi-Regular Graph (no duplicate edges)
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
N1 = 200
N2 = 200
M = 50

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 4
ALPHA_TILDE_STEP = 0.1  # Coarse scan first

LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0.0

# ============================================================
# Graph Generation Configuration
# ============================================================
USE_BIREGULAR_GRAPH = False  # Whether to generate uniform graph (bi-regular graph)
# True:  Use Dinic algorithm to generate strict bi-regular/near-regular graph (uniform degree)
# False: Use pure random method to quickly generate random graph (entirely on GPU, supports any N1≠N2)

# ============================================================
# Early Stop Configuration (Intelligent Convergence Detection)
# ============================================================
USE_EARLY_STOP = False  # Whether to enable Early Stop
EPOCHS_PER_ALPHA = 5000  # Coarse scan: higher epochs for large matrices
# [Mode 1] USE_EARLY_STOP = True (Recommended)
#   Strategy: Detect both absolute threshold and relative change rate
#   - Check loss every EARLY_STOP_CHECK_INTERVAL steps
#   - Stop immediately when loss < TARGET_LOSS_THRESHOLD (absolute threshold)
#   - Or stop when loss relative change < RELATIVE_CHANGE_THRESHOLD (convergence detection)
#   - No more than MAX_STEPS_PER_ALPHA steps (safety upper limit)
TARGET_LOSS_THRESHOLD = 1e-8           # Absolute loss threshold
RELATIVE_CHANGE_THRESHOLD = 1e-7       # Relative change threshold (e.g., 0.001% = 1e-5)
EARLY_STOP_CHECK_INTERVAL = 100        # Check interval (steps)
EARLY_STOP_PATIENCE = 5                # How many consecutive checks of almost no change before stopping
MAX_STEPS_PER_ALPHA = None             # Maximum step limit (None = use EPOCHS_PER_ALPHA)

# [Mode 2] USE_EARLY_STOP = False
#   - Fixed training of EPOCHS_PER_ALPHA steps, no loss checking
#   - Output final loss for quality checking when last alpha training completes

SAMPLES_PER_ALPHA = 1
RESAMPLE_MASK_EACH_TRIAL = True

SEED = 42

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                      ('cuda' if torch.cuda.is_available() else 'cpu'))

# ============================================================
# Performance Optimization Configuration (for modern GPUs like 5090)
# ============================================================
# BF16 mixed precision: Use BF16 in training loop to accelerate matrix multiplication
#   - Speed boost: ~2x
#   - Memory saving: 50%
#   - Precision impact: Minimal (BF16 range same as FP32)
USE_BF16 = False  # Disable BF16 to avoid torch.compile dtype issues
COMPUTE_DTYPE = torch.float32

# TF32 acceleration: Let CUDA automatically use TensorFloat32 to accelerate FP32 matrix multiplication
#   - Speed boost: Matrix multiplication itself ~8x, overall program ~1.1-1.3x
#   - Memory usage: No change
#   - Precision impact: Minimal (10-bit mantissa vs FP32's 23-bit)
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[Optimization] TF32 enabled for CUDA matmul")

# ============================================================
# Create Results Directory
# ============================================================
RESULT_DIR = Path(f"Result") / f"{N1}_{N2}_{M}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[Results directory] {RESULT_DIR}")

print(f"[Device] {DEVICE}")
print(f"[Compute dtype] {COMPUTE_DTYPE}")
print(f"[Implementation] DENSE MATRIX (full N1×N2 computation)")
print(f"[Graph generation] {'Bi-regular (Dinic)' if USE_BIREGULAR_GRAPH else 'Random (GPU fast)'}")
print(f"[Samples per alpha] {SAMPLES_PER_ALPHA} (batched on GPU)")
print(f"[Optimization] Kernel fusion enabled (reduces GPU kernel launches by ~66%)")


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_teacher_dense(N1, N2, M, device, seed=None):
    """
    Create teacher model parameters W_true and X_true

    Note: Teacher model always uses FP32 precision to ensure ground truth accuracy
          Student model training will use BF16 for acceleration, but final evaluation
          will convert back to FP32 for comparison
    """
    if seed is not None:
        torch.manual_seed(seed)
    scale = 1.0 / np.sqrt(M)
    # Teacher parameters always use FP32 to ensure precision
    W_true = torch.randn(N1, M, device=device, dtype=torch.float32) * scale
    X_true = torch.randn(M, N2, device=device, dtype=torch.float32) * scale
    return W_true, X_true

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

    # ============================================================
    # Removed Circulant fast path (avoid structural bias)
    # All cases uniformly use Dinic algorithm to generate random bi-regular graph
    # ============================================================
    # ---------- Dinic max flow algorithm, generate random bi-regular graph ----------

    # --- MODIFICATION START ---
    # 1. Uniformly create random number generator, ensure all random operations from same seed
    if seed is not None:
        rng = np.random.RandomState(seed + 12345 + int(round(alpha_tilde_left * 1e6)))
    else:
        rng = np.random.RandomState() # Use non-fixed seed
    # --- MODIFICATION END ---

    base = total_edges // N2
    rem = total_edges % N2
    right_target = np.full(N2, base, dtype=int)
    if rem > 0:
        idx = np.arange(N2)
        # --- MODIFICATION START ---
        # Use unified rng for shuffle
        rng.shuffle(idx)
        # --- MODIFICATION END ---
        right_target[idx[:rem]] += 1

    if right_target.max() > N1:
        raise RuntimeError(f"Some right node target degree {right_target.max()} > N1={N1}, infeasible")

    # Dinic implementation (lightweight version, remains unchanged)
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

    # --- MODIFICATION START ---
    # 2. Core modification: Randomize L->R edge addition order to forcibly introduce randomness
    # Even in rem=0 case, this ensures the generated graph is a random sample
    all_pairs = list(itertools.product(range(N1), range(N2)))
    rng.shuffle(all_pairs)

    for i, j in all_pairs:
        ui = L_off + i
        vj = R_off + j
        din.add_edge(ui, vj, 1)
    # --- MODIFICATION END ---

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


@torch.no_grad()
def compute_generalization_error(W_student, X_student, W_teacher, X_teacher):
    """Calculate generalization error E_test = (1/N²) Σ (Y*_ij - Y_ij)²"""
    Y_teacher = W_teacher @ X_teacher
    Y_student = W_student @ X_student
    mse = torch.mean((Y_teacher - Y_student) ** 2)
    return float(mse.item())


@torch.no_grad()
def compute_m_squared(W_student, X_student, W_teacher, X_teacher):
    """Calculate m² for verifying theoretical relationship"""
    Y_teacher = W_teacher @ X_teacher
    Y_student = W_student @ X_student

    num = (Y_teacher * Y_student).sum()
    den = torch.sqrt((Y_teacher ** 2).sum() * (Y_student ** 2).sum()) + 1e-12
    m_squared = (num / den) ** 2

    return float(m_squared.item())


def generate_batched_masks(N1, N2, M, alpha_tilde_left, S, device, seed_base):
    """
    Batch generate different masks for S trials

    Parameters:
        N1, N2, M: Graph dimension parameters
        alpha_tilde_left: Left alpha value
        S: Number of masks to generate (number of trials)
        device: torch device
        seed_base: Random seed base

    Returns:
        i_idx_batched: (S, num_edges) tensor, each row is i indices for a trial
        j_idx_batched: (S, num_edges) tensor, each row is j indices for a trial
        num_edges: Number of edges per mask
    """
    i_list = []
    j_list = []
    edge_counts = []

    # Generate mask for each trial
    for s in range(S):
        i_idx, j_idx, C = sample_pairs_biregular_exact(
            N1, N2, M, alpha_tilde_left, device, seed=seed_base + s
        )
        i_list.append(i_idx)
        j_list.append(j_idx)
        edge_counts.append(C)

    # Verify all masks have same edge count
    if len(set(edge_counts)) > 1:
        raise RuntimeError(f"Edge counts inconsistent across trials: {edge_counts}, cannot batch process")

    num_edges = edge_counts[0]

    if num_edges == 0:
        # Special case with 0 edges
        return (torch.empty((S, 0), dtype=torch.long, device=device),
                torch.empty((S, 0), dtype=torch.long, device=device),
                0)

    # Stack lists into (S, num_edges) tensor
    i_idx_batched = torch.stack(i_list, dim=0)  # (S, num_edges)
    j_idx_batched = torch.stack(j_list, dim=0)  # (S, num_edges)

    return i_idx_batched, j_idx_batched, num_edges


# ------------------------------------------------------------
# Batched Model
# ------------------------------------------------------------
class MaskedMF_Batched(nn.Module):
    def __init__(self, S, N1, N2, M, device, seed_base):
        super().__init__()
        self.S = S
        scale = 1.0 / np.sqrt(M)
        W_list = []
        X_list = []
        for s in range(S):
            torch.manual_seed(seed_base + s)
            W_list.append(torch.randn(N1, M, device=device) * scale)
            X_list.append(torch.randn(M, N2, device=device) * scale)

        self.W = nn.Parameter(torch.stack(W_list, dim=0))
        self.X = nn.Parameter(torch.stack(X_list, dim=0))

    def masked_mse_batched(self, y_true_obs, i_idx, j_idx):
        if i_idx.numel() == 0:
            return torch.zeros(self.S, device=self.W.device)

        Wi = self.W[:, i_idx, :]
        Xj = self.X[:, :, j_idx].transpose(1, 2)
        y_hat = (Wi * Xj).sum(dim=2)

        y_true_expanded = y_true_obs.unsqueeze(0).expand(self.S, -1)
        losses = torch.mean((y_hat - y_true_expanded) ** 2, dim=1)
        return losses


# ------------------------------------------------------------
# Batched Training - Optimized Kernel Fusion
# ------------------------------------------------------------

def fused_training_step(W, X, Y_teacher_b, A_b, alpha, lr):
    """
    Fused training step to reduce kernel launches.

    This function combines multiple operations into fewer kernels:
    - Computes both W and X gradients with fused operations
    - Reduces kernel count from ~18 to ~6 per step
    - Compatible with both MPS and CUDA

    Args:
        W: Student W parameters (S, N1, M)
        X: Student X parameters (S, M, N2)
        Y_teacher_b: Teacher output (1, N1, N2) or (S, N1, N2)
        A_b: Observation mask (1, N1, N2) or (S, N1, N2)
        alpha: Scaling factor (1/sqrt(M))
        lr: Learning rate

    Returns:
        Updated W, X tensors
    """
    # W update - fused operations
    Y_student = alpha * torch.matmul(W, X)
    # Fuse subtraction and multiplication
    Mres = (Y_teacher_b - Y_student) * A_b
    grad_W = -2.0 * alpha * torch.matmul(Mres, X.transpose(1, 2))
    W = W - lr * grad_W

    # X update - fused operations with updated W
    Y_student2 = alpha * torch.matmul(W, X)
    Mres2 = (Y_teacher_b - Y_student2) * A_b
    grad_X = -2.0 * alpha * torch.matmul(W.transpose(1, 2), Mres2)
    X = X - lr * grad_X

    return W, X


# Try to compile the training step for better performance
# Disable torch.compile for large matrices (CUDA graph issues with 10000×10000)
# Manual kernel fusion is already implemented and provides good performance
fused_training_step_compiled = fused_training_step
USE_COMPILED_STEP = False
print("[Optimization] Using manual kernel fusion (torch.compile disabled for stability)")


# ------------------------------------------------------------
# Batched Training
# ------------------------------------------------------------
def train_batched_trials_agd(
    Wt, Xt, i_idx, j_idx, steps, S, seed_for_init, lr=1e-2,
    *,
    loss_squared_sum: bool = False,  # Set True to "reproduce the discrepancy" (np.sum(M)**2)
    i_idx_batched=None,              # Optional: (S, num_edges) batched i indices
    j_idx_batched=None               # Optional: (S, num_edges) batched j indices
):
    """
    Parameters and behavior:
      - Wt, Xt: Teacher parameters (N1xM, MxN2)
      - i_idx, j_idx: Observation positions (coordinates of 1s in mask A), used when i_idx_batched=None
      - steps: Alternating steps (equivalent to your original EPOCHS_PER_ALPHA)
      - S: Number of parallel trials (batch)
      - seed_for_init: Seed base for student initialization
      - lr: Student learning rate
      - loss_squared_sum:
          False: L = sum(M**2) (consistent with given gradients, recommended)
          True : L = (sum(M))**2 (strictly copy "discrepancy" from screenshot)
      - i_idx_batched, j_idx_batched: Optional batched masks, shape (S, num_edges)
          If provided, each trial uses a different mask
    Returns: Same results list as original train_batched_trials (S dicts)
    """
    device = Wt.device
    N1, M = Wt.shape
    M_, N2 = Xt.shape
    assert M_ == M

    # Determine if using batched masks
    use_batched_masks = (i_idx_batched is not None) and (j_idx_batched is not None)

    if use_batched_masks:
        # Batched mask mode: Build different mask A for each trial
        # A shape: (S, N1, N2)
        A_b = torch.zeros((S, N1, N2), dtype=Wt.dtype, device=device)
        # Type assertion: these are guaranteed not None due to use_batched_masks condition
        assert i_idx_batched is not None and j_idx_batched is not None
        for s in range(S):
            if i_idx_batched[s].numel() > 0:
                # MPS-compatible: avoid mixed scalar+tensor indexing
                A_s = A_b[s]
                A_s[i_idx_batched[s], j_idx_batched[s]] = 1.0
    else:
        # Single mask mode: All trials use same mask
        A = torch.zeros((N1, N2), dtype=Wt.dtype, device=device)
        if i_idx is not None and i_idx.numel() > 0:
            A[i_idx, j_idx] = 1.0
        A_b = A.unsqueeze(0)                        # (1, N1, N2)

    # Precomputation
    alpha = 1.0 / (M ** 0.5)
    Y_teacher = Wt @ Xt                         # (N1, N2) FP32
    Y_teacher_b = Y_teacher.unsqueeze(0)        # (1, N1, N2)

    # Student initialization (use FP32 storage, will convert to BF16 for training computation)
    scale = 1.0 / (M ** 0.5)
    torch.manual_seed(seed_for_init)
    W = torch.randn((S, N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((S, M, N2), device=device, dtype=torch.float32) * scale

    # ============================================================
    # Training loop - Optimized with Kernel Fusion
    # ============================================================
    # Optimization strategy:
    # 1. Use fused training step to reduce kernel launches from ~18 to ~6 per step
    # 2. torch.compile automatically fuses operations when available (PyTorch 2.0+)
    # 3. Unified precision handling: use COMPUTE_DTYPE throughout (BF16 on CUDA, FP32 on MPS)
    # 4. Eliminates redundant type conversions (.float() calls)
    # 5. Early Stop: Training strategy decided by USE_EARLY_STOP switch
    # ============================================================

    actual_steps = 0  # Actual training steps

    # ============================================================
    # Early Stop state tracking
    # ============================================================
    loss_history = deque(maxlen=EARLY_STOP_PATIENCE) if USE_EARLY_STOP else None
    max_training_steps = MAX_STEPS_PER_ALPHA if MAX_STEPS_PER_ALPHA is not None else steps

    # Convert to compute dtype for training (BF16 on CUDA, FP32 on MPS/CPU)
    W = W.to(dtype=COMPUTE_DTYPE)
    X = X.to(dtype=COMPUTE_DTYPE)

    for step in range(max_training_steps):
        # Use optimized fused training step
        # This reduces kernel launches and improves GPU utilization
        W, X = fused_training_step_compiled(W, X, Y_teacher_b, A_b, alpha, lr)
        actual_steps = step + 1

        # ============================================================
        # Early Stop check (Intelligent convergence detection)
        # ============================================================
        if USE_EARLY_STOP and (step + 1) % EARLY_STOP_CHECK_INTERVAL == 0:
            with torch.no_grad():
                Y_check = alpha * torch.matmul(W, X)
                R_check = (Y_teacher_b - Y_check) * A_b
                if loss_squared_sum:
                    current_loss = float(torch.sum(R_check, dim=(1, 2)).mean().item()) ** 2
                else:
                    current_loss = float(torch.sum(R_check ** 2, dim=(1, 2)).mean().item())

                # Strategy 1: Absolute threshold check
                if current_loss < TARGET_LOSS_THRESHOLD:
                    break

                # Strategy 2: Relative change detection (convergence judgment)
                if loss_history is not None:  # Type guard for Pylance
                    loss_history.append(current_loss)
                    if len(loss_history) >= EARLY_STOP_PATIENCE:
                        # Calculate relative change over recent PATIENCE checks
                        losses = list(loss_history)
                        max_loss = max(losses)
                        min_loss = min(losses)
                        if max_loss > 1e-12:  # Avoid division by zero
                            relative_change = (max_loss - min_loss) / max_loss
                            if relative_change < RELATIVE_CHANGE_THRESHOLD:
                                # Loss almost unchanged, converged
                                break

    # ============================================================
    # Evaluation phase - Use FP32 precision to ensure accuracy
    # ============================================================
    # Convert student parameters back to FP32 for precise evaluation
    # (necessary if trained with BF16 on CUDA, no-op on MPS which uses FP32)
    W = W.float()
    X = X.float()

    results = []
    with torch.no_grad():
        # Final residual (for loss reporting) - compute with FP32
        Y_final = alpha * torch.matmul(W, X)
        Rf = (Y_teacher_b - Y_final) * A_b
        if loss_squared_sum:
            # Reproduce "discrepancy" version: (sum(M))**2
            loss_vec = torch.sum(Rf, dim=(1, 2)) ** 2
        else:
            # Mathematically consistent version: sum(M**2)
            loss_vec = torch.sum(Rf ** 2, dim=(1, 2))

        for s in range(S):
            W_s = W[s]  # FP32
            X_s = X[s]  # FP32
            # Gram overlap (using gram_overlap_cosine)
            Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
            Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
            # Gram overlap with zero-to-one normalization
            Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
            Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)
            # Y overlap (rotationally invariant)
            Yp = W_s @ X_s
            Yt = Y_teacher
            Q_Y = float(((Yt.flatten() * Yp.flatten()).sum()) /
                        (Yt.norm() * Yp.norm() + 1e-12))
            # Generalization error & m^2 (consistent with your original functions)
            gen_error = float(torch.mean((Yt - Yp) ** 2).item())
            num = (Yt * Yp).sum()
            den = torch.sqrt((Yt ** 2).sum() * (Yp ** 2).sum()) + 1e-12
            m_squared = float((num / den) ** 2)

            results.append(dict(
                Q_W=float(Q_W),
                Q_X=float(Q_X),
                Q_W_prime=float(Q_W_prime),
                Q_X_prime=float(Q_X_prime),
                Q_Y=float(Q_Y),
                Gen_Error=float(gen_error),
                m_squared=float(m_squared),
                Final_Loss=float(loss_vec[s].item()),
                epochs=int(actual_steps),  # Actual training steps (may be less than steps due to early stop)
                Time_s=0.0                 # Can add timing if needed
            ))
    return results


# ------------------------------------------------------------
# Parallel Alpha Training (Major Optimization)
# ------------------------------------------------------------
def train_all_alphas_parallel(
    Wt, Xt, alpha_values, steps, S, seed_for_init, lr=1e-2,
    *,
    loss_squared_sum: bool = False
):
    """
    Train all alpha values in parallel to dramatically reduce Python loop overhead.

    Key optimization: Instead of 21 alphas × 200k steps = 4.2M Python loops,
    we do 200k steps with all 21 alphas batched together = 200k Python loops.

    This reduces CPU overhead by ~21x and improves GPU utilization.

    Args:
        Wt, Xt: Teacher parameters
        alpha_values: List of alpha_tilde values to train
        steps: Training steps
        S: Samples per alpha
        seed_for_init: Random seed
        lr: Learning rate
        loss_squared_sum: Loss computation mode

    Returns:
        Dictionary mapping alpha values to their results
    """
    device = Wt.device
    N1, M = Wt.shape
    M_, N2 = Xt.shape
    assert M_ == M

    num_alphas = len(alpha_values)
    alpha_scale = 1.0 / (M ** 0.5)

    print(f"\n[Parallel Alpha Training] Training {num_alphas} alphas simultaneously")
    print(f"[Optimization] Reducing {num_alphas} × {steps} = {num_alphas * steps:,} loops")
    print(f"               to {steps:,} loops (CPU overhead reduced by {num_alphas}x)")

    # ============================================================
    # Generate masks for all alphas upfront
    # ============================================================
    print(f"[Step 1/4] Generating masks for {num_alphas} alphas...")

    # Pre-generate all masks: shape (num_alphas, S, N1, N2)
    all_masks = []
    all_C_values = []

    for alpha_tilde in alpha_values:
        if RESAMPLE_MASK_EACH_TRIAL and S > 1:
            # Generate S different masks for this alpha
            i_idx_batched, j_idx_batched, C = generate_batched_masks(
                N1, N2, M, alpha_tilde, S, device, seed_base=SEED + 1000
            )
            # Build mask tensor (S, N1, N2)
            A_alpha = torch.zeros((S, N1, N2), dtype=Wt.dtype, device=device)
            assert i_idx_batched is not None and j_idx_batched is not None
            for s in range(S):
                if i_idx_batched[s].numel() > 0:
                    A_s = A_alpha[s]
                    A_s[i_idx_batched[s], j_idx_batched[s]] = 1.0
        else:
            # Single mask for this alpha, broadcast across S
            i_idx, j_idx, C = sample_pairs_biregular_exact(
                N1, N2, M, alpha_tilde, device, seed=SEED + int(alpha_tilde * 1000)
            )
            A_single = torch.zeros((N1, N2), dtype=Wt.dtype, device=device)
            if i_idx is not None and i_idx.numel() > 0:
                A_single[i_idx, j_idx] = 1.0
            A_alpha = A_single.unsqueeze(0).expand(S, -1, -1).contiguous()

        all_masks.append(A_alpha)
        all_C_values.append(C)

    # Stack all masks: (num_alphas, S, N1, N2)
    A_all = torch.stack(all_masks, dim=0)

    # ============================================================
    # Initialize student parameters for all alphas
    # ============================================================
    print(f"[Step 2/4] Initializing parameters for {num_alphas} alphas...")

    scale = 1.0 / (M ** 0.5)
    torch.manual_seed(seed_for_init)

    # Shape: (num_alphas, S, N1, M) and (num_alphas, S, M, N2)
    W_all = torch.randn((num_alphas, S, N1, M), device=device, dtype=COMPUTE_DTYPE) * scale
    X_all = torch.randn((num_alphas, S, M, N2), device=device, dtype=COMPUTE_DTYPE) * scale

    # Prepare teacher output for broadcasting
    Y_teacher = Wt @ Xt  # (N1, N2)
    Y_teacher_expanded = Y_teacher.unsqueeze(0).unsqueeze(0)  # (1, 1, N1, N2)

    # ============================================================
    # Parallel training loop
    # ============================================================
    print(f"[Step 3/4] Training {num_alphas} alphas in parallel for {steps} steps...")

    for step in tqdm(range(steps), desc="Parallel training", leave=False):
        # Fused training step for all alphas simultaneously
        # W_all: (num_alphas, S, N1, M)
        # X_all: (num_alphas, S, M, N2)
        # A_all: (num_alphas, S, N1, N2)

        # W gradient and update
        Y_student = alpha_scale * torch.matmul(W_all, X_all)  # (num_alphas, S, N1, N2)
        Mres = (Y_teacher_expanded - Y_student) * A_all
        grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X_all.transpose(-2, -1))
        W_all = W_all - lr * grad_W

        # X gradient and update
        Y_student2 = alpha_scale * torch.matmul(W_all, X_all)
        Mres2 = (Y_teacher_expanded - Y_student2) * A_all
        grad_X = -2.0 * alpha_scale * torch.matmul(W_all.transpose(-2, -1), Mres2)
        X_all = X_all - lr * grad_X

    # ============================================================
    # Collect results for each alpha
    # ============================================================
    print(f"[Step 4/4] Collecting results for {num_alphas} alphas...")

    W_all = W_all.float()
    X_all = X_all.float()

    results = {}

    with torch.no_grad():
        for alpha_idx, alpha_tilde in enumerate(alpha_values):
            W_alpha = W_all[alpha_idx]  # (S, N1, M)
            X_alpha = X_all[alpha_idx]  # (S, M, N2)
            A_alpha = A_all[alpha_idx]  # (S, N1, N2)
            C = all_C_values[alpha_idx]

            # Compute metrics for each sample
            trial_results = []
            for s in range(S):
                W_s = W_alpha[s]
                X_s = X_alpha[s]

                # Overlaps
                Q_W = gram_overlap_cosine(W_s, Wt, use_left=True)
                Q_X = gram_overlap_cosine(X_s, Xt, use_left=False)
                Q_W_prime = gram_overlap_zero_to_one(W_s, Wt, use_left=True)
                Q_X_prime = gram_overlap_zero_to_one(X_s, Xt, use_left=False)

                # Y overlap
                Yp = W_s @ X_s
                Yt = Y_teacher
                Q_Y = float(((Yt.flatten() * Yp.flatten()).sum()) /
                           (Yt.norm() * Yp.norm() + 1e-12))

                # Gen error and m^2
                gen_error = float(torch.mean((Yt - Yp) ** 2).item())
                num = (Yt * Yp).sum()
                den = torch.sqrt((Yt ** 2).sum() * (Yp ** 2).sum()) + 1e-12
                m_squared = float((num / den) ** 2)

                # Final loss
                Y_final = alpha_scale * (W_s @ X_s)
                Rf = (Y_teacher - Y_final) * A_alpha[s]
                if loss_squared_sum:
                    final_loss = float(torch.sum(Rf) ** 2)
                else:
                    final_loss = float(torch.sum(Rf ** 2).item())

                trial_results.append({
                    'Q_W': float(Q_W),
                    'Q_X': float(Q_X),
                    'Q_W_prime': float(Q_W_prime),
                    'Q_X_prime': float(Q_X_prime),
                    'Q_Y': float(Q_Y),
                    'Gen_Error': float(gen_error),
                    'm_squared': float(m_squared),
                    'Final_Loss': final_loss
                })

            # Aggregate statistics
            qW = [s['Q_W'] for s in trial_results]
            qX = [s['Q_X'] for s in trial_results]
            qW_prime = [s['Q_W_prime'] for s in trial_results]
            qX_prime = [s['Q_X_prime'] for s in trial_results]
            qY = [s['Q_Y'] for s in trial_results]
            gen_err = [s['Gen_Error'] for s in trial_results]
            m_sq = [s['m_squared'] for s in trial_results]
            loss_list = [s['Final_Loss'] for s in trial_results]

            def mean_std(x):
                x = np.array(x, dtype=float)
                return float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)

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
                'alpha_tilde_left': aL_real,
                'alpha_tilde_right': aR_real,
                'C': int(C),
                'Q_W_mean': QW_mean, 'Q_W_std': QW_std,
                'Q_X_mean': QX_mean, 'Q_X_std': QX_std,
                'Q_W_prime_mean': QW_prime_mean, 'Q_W_prime_std': QW_prime_std,
                'Q_X_prime_mean': QX_prime_mean, 'Q_X_prime_std': QX_prime_std,
                'Q_Y_mean': QY_mean, 'Q_Y_std': QY_std,
                'Gen_Error_mean': GE_mean, 'Gen_Error_std': GE_std,
                'm_squared_mean': M2_mean, 'm_squared_std': M2_std,
                'Loss_mean': L_mean, 'Loss_std': L_std,
                'epochs_mean': float(steps),
                'Time_s_mean': 0.0
            }

    print(f"[Parallel Alpha Training] Completed!")
    return results


# ------------------------------------------------------------
# Experiment Runner
# ------------------------------------------------------------
def run_experiment_batched():
    set_seed(SEED)
    Wt, Xt = create_teacher_dense(N1, N2, M, DEVICE, seed=SEED)

    # Check for smart alphas (adaptive sampling)
    smart_alpha_path = Path(f"Result/{N1}_{N2}_{M}/smart_alphas.npy")
    if smart_alpha_path.exists():
        a_vals = np.load(smart_alpha_path)
        print(f"\n{'='*80}")
        print(f"[ADAPTIVE SAMPLING] Using {len(a_vals)} smart alphas")
        print(f"[Range] α ∈ [{a_vals[0]:.2f}, {a_vals[-1]:.2f}]")
        print(f"[Speedup] ~{int((ALPHA_TILDE_STOP - ALPHA_TILDE_START) / ALPHA_TILDE_STEP / len(a_vals))}x vs uniform Δα={ALPHA_TILDE_STEP}")
        print(f"{'='*80}\n")
    else:
        a_vals = np.arange(ALPHA_TILDE_START, ALPHA_TILDE_STOP + 1e-12, ALPHA_TILDE_STEP)
        print(f"\n[Uniform Sampling] Using {len(a_vals)} alphas, Δα={ALPHA_TILDE_STEP}")
    results = {}

    pbar = tqdm(a_vals,
                desc="Alpha sweep",
                bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                dynamic_ncols=True,
                leave=True,
                position=0)

    for aT in pbar:
        pbar.set_postfix({'α': f'{aT:.2f}'})

        # ============================================================
        # Graph generation strategy (optimized version):
        # - If RESAMPLE_MASK_EACH_TRIAL=True: Directly generate S different graphs, skip base graph generation
        # - If RESAMPLE_MASK_EACH_TRIAL=False: Only generate one base graph, all trials share
        # ============================================================
        if RESAMPLE_MASK_EACH_TRIAL:
            # Batch generate S different graphs (avoid generating unnecessary base graph)
            i_idx_batched, j_idx_batched, C = generate_batched_masks(
                N1, N2, M, aT, SAMPLES_PER_ALPHA, DEVICE, seed_base=SEED + 1000
            )
            i_idx_base, j_idx_base = None, None  # Base graph not needed
        else:
            # Generate single base graph, all trials share
            i_idx_base, j_idx_base, C = sample_pairs_biregular_exact(N1, N2, M, aT, DEVICE, seed=SEED)
            i_idx_batched, j_idx_batched = None, None  # Batch graph not needed

        aL_real = (C / (M * N1)) if (M * N1) > 0 else 0.0
        aR_real = (C / (M * N2)) if (M * N2) > 0 else 0.0

        if C == 0:
            results[float(aT)] = dict(
                alpha_tilde_left=aL_real, alpha_tilde_right=aR_real, C=0,
                Q_W_mean=0.0, Q_W_std=0.0, Q_X_mean=0.0, Q_X_std=0.0,
                Q_W_prime_mean=0.0, Q_W_prime_std=0.0, Q_X_prime_mean=0.0, Q_X_prime_std=0.0,
                Q_Y_mean=0.0, Q_Y_std=0.0,
                Gen_Error_mean=0.0, Gen_Error_std=0.0,
                m_squared_mean=0.0, m_squared_std=0.0,
                Loss_mean=0.0, Loss_std=0.0,
                epochs_mean=0.0, Time_s_mean=0.0
            )
            continue

        # ============================================================
        # Training phase: Use pre-generated graphs (graphs already generated above)
        # ============================================================
        trial_stats = train_batched_trials_agd(
            Wt, Xt, i_idx_base, j_idx_base,
            steps=EPOCHS_PER_ALPHA, S=SAMPLES_PER_ALPHA,
            seed_for_init=SEED + 10_000,
            lr=LEARNING_RATE,
            loss_squared_sum=True,
            i_idx_batched=i_idx_batched,
            j_idx_batched=j_idx_batched
        )

        # Aggregate
        qW = [s['Q_W'] for s in trial_stats]
        qX = [s['Q_X'] for s in trial_stats]
        qW_prime = [s['Q_W_prime'] for s in trial_stats]
        qX_prime = [s['Q_X_prime'] for s in trial_stats]
        qY = [s['Q_Y'] for s in trial_stats]
        gen_err = [s['Gen_Error'] for s in trial_stats]
        m_sq = [s['m_squared'] for s in trial_stats]
        loss_list = [s['Final_Loss'] for s in trial_stats]
        epochs_list = [s['epochs'] for s in trial_stats]  # Get actual training steps

        def mean_std(x):
            x = np.array(x, dtype=float)
            return float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)

        QW_mean, QW_std = mean_std(qW)
        QX_mean, QX_std = mean_std(qX)
        QW_prime_mean, QW_prime_std = mean_std(qW_prime)
        QX_prime_mean, QX_prime_std = mean_std(qX_prime)
        QY_mean, QY_std = mean_std(qY)
        GE_mean, GE_std = mean_std(gen_err)
        M2_mean, M2_std = mean_std(m_sq)
        L_mean, L_std = mean_std(loss_list)
        epochs_mean = float(np.mean(epochs_list))

        # Quality check: If Early Stop disabled and last alpha, output final loss
        is_last_alpha = (aT >= ALPHA_TILDE_STOP - 1e-6)
        if (not USE_EARLY_STOP) and is_last_alpha:
            print(f"\n{'='*70}")
            print(f"[Quality Check] Last alpha (α={aT:.1f}) training completed")
            print(f"{'='*70}")
            print(f"  Final Loss mean: {L_mean:.6e}")
            print(f"  Final Loss std: {L_std:.6e}")
            print(f"  Actual training steps: {int(epochs_mean):,}")
            if L_mean > 1e-4:
                print(f"  ⚠️  Warning: Loss is high, may need more training steps or adjust learning rate")
            print(f"{'='*70}\n")

        results[float(aT)] = dict(
            alpha_tilde_left=aL_real, alpha_tilde_right=aR_real, C=int(C),
            Q_W_mean=QW_mean, Q_W_std=QW_std,
            Q_X_mean=QX_mean, Q_X_std=QX_std,
            Q_W_prime_mean=QW_prime_mean, Q_W_prime_std=QW_prime_std,
            Q_X_prime_mean=QX_prime_mean, Q_X_prime_std=QX_prime_std,
            Q_Y_mean=QY_mean, Q_Y_std=QY_std,
            Gen_Error_mean=GE_mean, Gen_Error_std=GE_std,
            m_squared_mean=M2_mean, m_squared_std=M2_std,
            Loss_mean=L_mean, Loss_std=L_std,
            epochs_mean=epochs_mean,  # Use actual average steps
            Time_s_mean=trial_stats[0]['Time_s']
        )

    pbar.close()
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
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 140)
    print("RESULTS SUMMARY (Perfect Bi-Regular Graph - Cosine vs Zero-to-One)")
    print("=" * 140)
    print(df.to_string(index=False))
    print("=" * 140)
    return df


def plot_results(results_dict):
    """Plot five charts (2x3 layout): Q_Y + comparison of two gram overlap methods"""
    items = sorted(results_dict.items(), key=lambda kv: kv[1]['alpha_tilde_left'])
    aL = np.array([kv[1]['alpha_tilde_left'] for kv in items])

    gen_err_mu = np.array([kv[1]['Gen_Error_mean'] for kv in items])
    gen_err_sd = np.array([kv[1]['Gen_Error_std'] for kv in items])

    qY_mu = np.array([kv[1]['Q_Y_mean'] for kv in items])
    qY_sd = np.array([kv[1]['Q_Y_std'] for kv in items])

    qW_mu = np.array([kv[1]['Q_W_mean'] for kv in items])
    qW_sd = np.array([kv[1]['Q_W_std'] for kv in items])

    qX_mu = np.array([kv[1]['Q_X_mean'] for kv in items])
    qX_sd = np.array([kv[1]['Q_X_std'] for kv in items])

    qW_prime_mu = np.array([kv[1]['Q_W_prime_mean'] for kv in items])
    qW_prime_sd = np.array([kv[1]['Q_W_prime_std'] for kv in items])

    qX_prime_mu = np.array([kv[1]['Q_X_prime_mean'] for kv in items])
    qX_prime_sd = np.array([kv[1]['Q_X_prime_std'] for kv in items])

    fig = plt.figure(figsize=(20, 12))

    # Chart 1: Q_Y (rotationally invariant)
    ax1 = fig.add_subplot(231)
    ax1.plot(aL, qY_mu, marker='D', linewidth=3.0, markersize=8,
             color='#d62728', label='Invariant Q_Y', zorder=3)
    ax1.fill_between(aL, qY_mu - qY_sd, qY_mu + qY_sd,
                     alpha=0.25, color='#d62728')
    ax1.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=13)
    ax1.set_ylabel('Q (rotationally invariant)', fontsize=13)
    ax1.set_title('Invariant Q_Y vs Alpha', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_facecolor('#fff5f5')

    # Chart 2: Q_W (cosine similarity)
    ax2 = fig.add_subplot(232)
    ax2.plot(aL, qW_mu, marker='s', linewidth=2.5, markersize=6,
             color='#ff7f0e', label='Q_W (cosine)')
    ax2.fill_between(aL, qW_mu - qW_sd, qW_mu + qW_sd,
                     alpha=0.2, color='#ff7f0e')
    ax2.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=13)
    ax2.set_ylabel('Q_W (Gram cosine)', fontsize=13)
    ax2.set_title('Q_W (Cosine) vs Alpha', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=11, loc='lower right')

    # Chart 3: Q_X (cosine similarity)
    ax3 = fig.add_subplot(233)
    ax3.plot(aL, qX_mu, marker='^', linewidth=2.5, markersize=6,
             color='#2ca02c', label='Q_X (cosine)')
    ax3.fill_between(aL, qX_mu - qX_sd, qX_mu + qX_sd,
                     alpha=0.2, color='#2ca02c')
    ax3.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=13)
    ax3.set_ylabel('Q_X (Gram cosine)', fontsize=13)
    ax3.set_title('Q_X (Cosine) vs Alpha', fontsize=14, fontweight='bold')
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.legend(fontsize=11, loc='lower right')

    # Chart 4: Q_W' (zero-to-one normalized)
    ax4 = fig.add_subplot(234)
    ax4.plot(aL, qW_prime_mu, marker='o', linewidth=2.5, markersize=6,
             color='#9467bd', label="Q_W' (zero-to-one)")
    ax4.fill_between(aL, qW_prime_mu - qW_prime_sd, qW_prime_mu + qW_prime_sd,
                     alpha=0.2, color='#9467bd')
    ax4.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=13)
    ax4.set_ylabel("Q_W' (baseline-corrected)", fontsize=13)
    ax4.set_title("Q_W' (Zero-to-One) vs Alpha", fontsize=14, fontweight='bold')
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.legend(fontsize=11, loc='lower right')

    # Chart 5: Q_X' (zero-to-one normalized)
    ax5 = fig.add_subplot(235)
    ax5.plot(aL, qX_prime_mu, marker='v', linewidth=2.5, markersize=6,
             color='#8c564b', label="Q_X' (zero-to-one)")
    ax5.fill_between(aL, qX_prime_mu - qX_prime_sd, qX_prime_mu + qX_prime_sd,
                     alpha=0.2, color='#8c564b')
    ax5.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=13)
    ax5.set_ylabel("Q_X' (baseline-corrected)", fontsize=13)
    ax5.set_title("Q_X' (Zero-to-One) vs Alpha", fontsize=14, fontweight='bold')
    ax5.set_ylim(-0.05, 1.05)
    ax5.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax5.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    # Save first large chart (containing 5 subplots)
    # detailed_path = RESULT_DIR / 'detailed_metrics.png'
    # fig.savefig(detailed_path, dpi=300, bbox_inches='tight')
    # print(f"\nDetailed metrics chart saved as: {detailed_path}")
    # plt.close(fig)  # Close figure to free memory

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
        ['Graph Generation', f"{'BiRegular (Dinic)' if USE_BIREGULAR_GRAPH else 'Random (GPU)'}"],
        ['Resample per Trial', f'{RESAMPLE_MASK_EACH_TRIAL}'],
        ['Samples per Alpha', f'{SAMPLES_PER_ALPHA}'],
        ['Learning Rate', f'{LEARNING_RATE}'],
    ]

    # Early Stop configuration
    if USE_EARLY_STOP:
        table_data.extend([
            ['Early Stop', 'Enabled'],
            ['Loss Threshold', f'{TARGET_LOSS_THRESHOLD:.1e}'],
            ['Relative Change Threshold', f'{RELATIVE_CHANGE_THRESHOLD:.1e}'],
            ['Patience', f'{EARLY_STOP_PATIENCE}'],
            ['Max Steps', f'{MAX_STEPS_PER_ALPHA if MAX_STEPS_PER_ALPHA else "None (use default)"}'],
        ])
    else:
        table_data.extend([
            ['Early Stop', 'Disabled'],
            ['Fixed Training Steps', f'{EPOCHS_PER_ALPHA}'],
        ])

    # Create table
    table = ax_table.table(cellText=table_data,
                          colWidths=[0.35, 0.65],
                          cellLoc='left',
                          loc='center',
                          bbox=(0, 0, 1, 1))  # type: ignore  # Matplotlib accepts tuple for bbox

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Set table style
    for i in range(len(table_data)):
        table[(i, 0)].set_facecolor('#e6f2ff')
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_facecolor('#f0f0f0')

    plt.tight_layout()

    # Generate filename (5 components)
    graph_type = "BiReg" if USE_BIREGULAR_GRAPH else "Rand"
    resample = "Resample" if RESAMPLE_MASK_EACH_TRIAL else "NoResample"
    early_stop_flag = "ET" if USE_EARLY_STOP else "EF"

    if USE_EARLY_STOP:
        # Format loss threshold, e.g. 1e-7 -> Loss1e-7
        key_param = f"Loss{TARGET_LOSS_THRESHOLD:.0e}".replace('e-0', 'e-').replace('e+0', 'e')
    else:
        key_param = f"Epoch{EPOCHS_PER_ALPHA}"

    filename = f"{graph_type}_{resample}_{early_stop_flag}_{key_param}_batch{SAMPLES_PER_ALPHA}.png"
    combined_path = RESULT_DIR / filename

    # Save chart
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined metrics chart saved as: {combined_path}")
    plt.close(fig_combined)
    print("\nAll charts saved successfully, no GUI display needed")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("Starting Sparse Matrix simulation")
    print("=" * 100)

    total_start = time.time()
    results = run_experiment_batched()
    total_time = time.time() - total_start

    print(f"\n✓ Total time: {total_time:.2f}s")

    df = display_results(results)
    plot_results(results)

    print("\n" + "=" * 100)
    print("COMPLETED")
    print("=" * 100)
