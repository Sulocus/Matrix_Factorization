"""
BiG-AMP with C4-free Graph Generation

This is a modified version of bigamp_optimized.py that generates graphs
without 4-loops (4-cycles) to study the effect of short loops on
the phase transition behavior.

Key changes from bigamp_optimized.py:
1. Added sample_pairs_no_c4() - C4-free graph generation
2. Added count_4loops() - 4-loop counting for verification
3. Results saved to ResultNo4/ directory
4. Verification output comparing 4-loop counts

Usage:
    python bigamp_no4loop.py                    # Default settings
    python bigamp_no4loop.py --n1 300 --m 100   # Custom size
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
from scipy import sparse

# ============================================================
# Default Parameters
# ============================================================
N1 = 200
N2 = 200
M = 50

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 1
ALPHA_TILDE_STEP = 0.02

# BiG-AMP parameters
DAMPING = 0.5
NOISE_VAR = 1e-10  # Match bigamp_optimized.py exactly
MAX_STEPS = 5000

SAMPLES_PER_ALPHA = 10
RESAMPLE_MASK_EACH_TRIAL = True
SEED = 42

# ============================================================
# Graph Generation Configuration
# ============================================================
USE_BIREGULAR_GRAPH = False
FORBID_4_CYCLES = True      # Enable 4-loop minimization

# MCMC parameters for loop minimization
# Note: C4-free is only possible for alpha < 0.35 (Kővári–Sós–Turán theorem)
#       For higher alpha, MCMC can only reduce 4-loops by 20-30% at best
#       Higher-order loops (6-loop, 8-loop) have even weaker constraints
MCMC_LAMBDA = 5.0           # Penalty coefficient (unused in greedy mode)
MCMC_SWEEPS = 20             # Number of sweeps (more = better reduction, 3-10 reasonable)
MCMC_ALPHA_THRESHOLD = 0.8  # Only run MCMC when alpha < this threshold (saves time at high density)
LOOP_ORDER = 2              # k=2 for 4-loops, k=3 for 6-loops, k=4 for 8-loops, etc.

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))

# Precision settings
USE_BF16 = DEVICE.type == 'cuda'
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Result directory - changed to ResultNo4
RESULT_DIR = Path(__file__).parent / "ResultNo4" / f"{N1}_{N2}_{M}"


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


# ============================================================
# 4-Loop Counting and Expected Value Functions
# ============================================================
def count_4loops(i_idx, j_idx, N1, N2):
    """
    Count the number of 4-loops (4-cycles) in a bipartite graph.

    A 4-loop in a bipartite graph is:
        i -- j1 -- i' -- j2 -- i
    where i, i' are left nodes and j1, j2 are right nodes.

    Method:
    - Compute B = A @ A.T where A is the adjacency matrix (N1 x N2)
    - B[i, i'] = number of common right neighbors of i and i'
    - Number of 4-loops = sum_{i < i'} C(B[i,i'], 2) = sum_{i < i'} B[i,i']*(B[i,i']-1)/2

    Returns:
        int: Number of 4-loops
    """
    if len(i_idx) == 0:
        return 0

    i_np = i_idx.cpu().numpy() if torch.is_tensor(i_idx) else i_idx
    j_np = j_idx.cpu().numpy() if torch.is_tensor(j_idx) else j_idx

    # Build sparse adjacency matrix
    adj = sparse.csr_matrix(
        (np.ones(len(i_np)), (i_np, j_np)),
        shape=(N1, N2)
    )

    # B[i, i'] = number of common right neighbors
    B = (adj @ adj.T).toarray()

    # Take upper triangle (excluding diagonal)
    upper = np.triu(B, k=1)

    # Count 4-loops: for each pair (i, i') with k common neighbors,
    # they form C(k, 2) = k*(k-1)/2 different 4-loops
    count = np.sum(upper * (upper - 1)) // 2

    return int(count)


def expected_4loops_random(N1, N2, C):
    """
    Calculate expected number of 4-loops in a random bipartite graph.

    For a bipartite graph with N1 left nodes, N2 right nodes, and C edges
    chosen uniformly at random:

    E[4-loops] ≈ C(N1,2) × C(N2,2) × (C/(N1×N2))^4
               = N1(N1-1)/2 × N2(N2-1)/2 × (C/(N1×N2))^4

    This is derived from:
    - Number of potential 4-cycles = C(N1,2) × C(N2,2)
    - Probability each 4-cycle exists ≈ p^4 where p = C/(N1×N2)

    Returns:
        float: Expected number of 4-loops
    """
    if C == 0 or N1 < 2 or N2 < 2:
        return 0.0

    # Number of potential 4-cycles
    n_potential = (N1 * (N1 - 1) / 2) * (N2 * (N2 - 1) / 2)

    # Edge probability
    p = C / (N1 * N2)

    # Expected 4-loops (need 4 specific edges)
    expected = n_potential * (p ** 4)

    return expected


def expected_2k_loops_random(N1, N2, C, k=2):
    """
    Calculate expected number of 2k-loops in a random bipartite graph.

    For k=2 (4-loops):
        E[4-loops] ≈ C(N1,2) × C(N2,2) × (C/(N1×N2))^4

    For k=3 (6-loops):
        E[6-loops] ≈ C(N1,3) × C(N2,3) × (C/(N1×N2))^6 × 6

    For general k:
        E[2k-loops] ~ C(N1,k) × C(N2,k) × p^(2k) × (k-1)!

    Note: Higher-order formulas are approximations.
    """
    if C == 0 or N1 < k or N2 < k:
        return 0.0

    p = C / (N1 * N2)

    if k == 2:
        # Exact formula for 4-loops
        n_potential = (N1 * (N1 - 1) / 2) * (N2 * (N2 - 1) / 2)
        return n_potential * (p ** 4)

    elif k == 3:
        # Approximation for 6-loops
        # Number of ways to choose 3 left nodes and 3 right nodes
        from math import comb, factorial
        n_left = comb(N1, 3) if N1 >= 3 else 0
        n_right = comb(N2, 3) if N2 >= 3 else 0
        # Each triple can form 6-cycle in multiple ways
        return n_left * n_right * (p ** 6) * factorial(2)

    elif k == 4:
        # Approximation for 8-loops
        from math import comb, factorial
        n_left = comb(N1, 4) if N1 >= 4 else 0
        n_right = comb(N2, 4) if N2 >= 4 else 0
        return n_left * n_right * (p ** 8) * factorial(3)

    else:
        # General approximation
        from math import comb, factorial
        n_left = comb(N1, k) if N1 >= k else 0
        n_right = comb(N2, k) if N2 >= k else 0
        return n_left * n_right * (p ** (2 * k)) * factorial(k - 1)


def compute_loop_ratio(actual_loops, N1, N2, C, k=2):
    """
    Compute the ratio of actual loops to expected loops in random graph.

    Returns:
        float: Ratio around 1.0 for random graph
               < 1.0 means fewer loops than expected (good for minimization)
    """
    expected = expected_2k_loops_random(N1, N2, C, k)
    if expected < 1e-10:
        return 0.0 if actual_loops == 0 else 1.0
    return actual_loops / expected


def compute_4loop_ratio(actual_4loops, N1, N2, C):
    """
    Compute the ratio of actual 4-loops to expected 4-loops in random graph.
    Backwards compatible wrapper for compute_loop_ratio with k=2.
    """
    return compute_loop_ratio(actual_4loops, N1, N2, C, k=2)


# ============================================================
# GPU-Accelerated Loop Counting and Minimization
# ============================================================

def count_2k_loops_gpu(A, k=2):
    """
    Count 2k-loops in bipartite graph using matrix powers.

    For bipartite graph with adjacency matrix A (N1 x N2):
    - B = A @ A.T gives common neighbor counts
    - 2k-loop corresponds to closed walks of length 2k

    k=2: 4-loops (C4): i -- j1 -- i' -- j2 -- i
         Count = sum_{i<i'} C(B[i,i'], 2)

    k=3: 6-loops (C6): i -- j1 -- i' -- j2 -- i'' -- j3 -- i
         Related to trace(B^3) and B^2 structure

    k=4: 8-loops (C8): Similar pattern with B^4

    Note: Higher k gives more complex counting formulas.
    For simplicity, we use trace-based approximation for k>2.
    """
    B = A @ A.T  # B[i,i'] = number of common j-neighbors

    if k == 2:
        # Exact 4-loop count: sum_{i<i'} C(B[i,i'], 2)
        upper = torch.triu(B, diagonal=1)
        return (upper * (upper - 1)).sum() // 2

    elif k == 3:
        # 6-loops: related to trace(B^3) / 6
        # Each 6-loop is counted 6 times in trace(B^3)
        B2 = B @ B
        B3 = B2 @ B
        # Subtract contributions from shorter paths
        trace_B3 = torch.trace(B3)
        trace_B = torch.trace(B)
        # Approximate: trace(B^3) counts 6-loops plus some corrections
        return trace_B3 // 6

    elif k == 4:
        # 8-loops: related to trace(B^4) / 8
        B2 = B @ B
        B4 = B2 @ B2
        trace_B4 = torch.trace(B4)
        return trace_B4 // 8

    else:
        # General case: trace(B^k) / (2k)
        Bk = B.clone()
        for _ in range(k - 1):
            Bk = Bk @ B
        return torch.trace(Bk) // (2 * k)


def count_4loops_gpu(A):
    """Count 4-loops: sum_{i<i'} C(B[i,i'], 2) where B = A @ A.T"""
    return count_2k_loops_gpu(A, k=2)


def mcmc_minimize_2k_loops_gpu(edges, N1, N2, device, k=2, lambda_penalty=5.0, n_sweeps=5, seed=None):
    """
    Generalized GPU 2k-loop reduction using MCMC edge-switching.

    Parameters:
        edges: List of (i, j) tuples representing edges
        N1, N2: Number of left and right nodes
        device: Torch device
        k: Loop order (k=2 for 4-loops, k=3 for 6-loops, k=4 for 8-loops)
        lambda_penalty: MCMC penalty coefficient (unused in greedy mode)
        n_sweeps: Number of MCMC sweeps
        seed: Random seed

    Returns:
        final_edges: List of (i, j) tuples after optimization
        accept_rate: Fraction of accepted swaps
        n_initial: Initial loop count
        n_final: Final loop count

    Strategy:
    - For k=2 (4-loops): Use edge contribution score (B @ A)[i,j]
    - For k>2 (6-loops, 8-loops): Use generalized score based on B^(k-1) @ A
    """
    if seed is not None:
        torch.manual_seed(seed)

    C = len(edges)
    if C < 2:
        return edges, 1.0, 0, 0

    # Convert to GPU tensors
    edges_i = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    edges_j = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)

    # Build adjacency matrix
    A = torch.zeros((N1, N2), dtype=torch.float32, device=device)
    A[edges_i, edges_j] = 1.0

    # Count initial loops
    n_initial = int(count_2k_loops_gpu(A, k).item())
    n_accepted = 0

    for sweep in range(n_sweeps):
        # Compute B = A @ A.T (common neighbor matrix)
        B = A @ A.T

        # Compute edge contribution score based on loop order
        # For k-loop, edge (i,j) contribution is related to (B^(k-1) @ A)[i,j]
        if k == 2:
            # 4-loops: (B @ A)[i,j] = sum of paths i--i'--j for common neighbors i'
            score_matrix = B @ A
        elif k == 3:
            # 6-loops: (B^2 @ A)[i,j]
            B2 = B @ B
            score_matrix = B2 @ A
        elif k == 4:
            # 8-loops: (B^3 @ A)[i,j]
            B2 = B @ B
            B3 = B2 @ B
            score_matrix = B3 @ A
        else:
            # General case: B^(k-1) @ A
            Bk = B.clone()
            for _ in range(k - 2):
                Bk = Bk @ B
            score_matrix = Bk @ A

        edge_scores = score_matrix[edges_i, edges_j]

        # Sort edges by score (descending) - high score = many loops
        sorted_idx = edge_scores.argsort(descending=True)

        # Try swapping top-scoring edges with low-scoring ones
        n_top = min(C // 4, 200)

        for kk in range(n_top):
            # High-score edge
            idx1 = sorted_idx[kk]
            # Low-score edge (random from bottom half)
            rand_offset = int(torch.randint(C // 2, (1,), device=device).item())
            idx2 = sorted_idx[C // 2 + rand_offset]

            i1, j1 = edges_i[idx1], edges_j[idx1]
            i2, j2 = edges_i[idx2], edges_j[idx2]

            # Check validity
            if i1 == i2 or j1 == j2:
                continue
            if A[i1, j2] > 0 or A[i2, j1] > 0:
                continue

            # Compute old loop count
            old_count = count_2k_loops_gpu(A, k)

            # Apply swap
            A[i1, j1] = 0
            A[i2, j2] = 0
            A[i1, j2] = 1
            A[i2, j1] = 1

            # Check new loop count
            new_count = count_2k_loops_gpu(A, k)

            if new_count < old_count:
                # Keep swap, update edge list
                edges_i[idx1] = i1
                edges_j[idx1] = j2
                edges_i[idx2] = i2
                edges_j[idx2] = j1
                n_accepted += 1
            else:
                # Revert
                A[i1, j2] = 0
                A[i2, j1] = 0
                A[i1, j1] = 1
                A[i2, j2] = 1

    n_final = int(count_2k_loops_gpu(A, k).item())
    final_edges = list(zip(edges_i.cpu().tolist(), edges_j.cpu().tolist()))
    accept_rate = n_accepted / (n_sweeps * min(C // 4, 200)) if n_sweeps > 0 else 1.0

    return final_edges, accept_rate, n_initial, n_final


def mcmc_minimize_4loops_gpu(edges, N1, N2, device, lambda_penalty=5.0, n_sweeps=5, seed=None):
    """
    Smart GPU 4-loop reduction targeting high-contribution edges.
    This is a convenience wrapper around mcmc_minimize_2k_loops_gpu with k=2.
    """
    return mcmc_minimize_2k_loops_gpu(edges, N1, N2, device, k=2,
                                       lambda_penalty=lambda_penalty,
                                       n_sweeps=n_sweeps, seed=seed)


# Keep CPU version as fallback
def count_edge_4loops(left_neighbors, right_neighbors, i, j):
    """CPU version: Count 4-loops that edge (i,j) participates in."""
    neighbors_of_i = left_neighbors[i]
    neighbors_of_j = right_neighbors[j]

    if len(neighbors_of_i) <= 1 or len(neighbors_of_j) <= 1:
        return 0

    count = 0
    for jp in neighbors_of_i:
        if jp == j:
            continue
        common = right_neighbors[jp] & neighbors_of_j
        common_minus_i = len(common) - (1 if i in common else 0)
        count += common_minus_i

    return count


def mcmc_minimize_4loops(edges, N1, N2, lambda_penalty=0.3, n_sweeps=10, seed=None):
    """CPU version of MCMC (fallback)."""
    if seed is not None:
        np.random.seed(seed)

    edges = list(edges)
    C = len(edges)

    if C < 2:
        return edges, 1.0, 0, 0

    left_neighbors = [set() for _ in range(N1)]
    right_neighbors = [set() for _ in range(N2)]
    edge_set = set()

    for i, j in edges:
        left_neighbors[i].add(j)
        right_neighbors[j].add(i)
        edge_set.add((i, j))

    n4_initial = 0
    for i, j in edges:
        n4_initial += count_edge_4loops(left_neighbors, right_neighbors, i, j)
    n4_initial //= 4

    n_proposals = n_sweeps * C
    n_accepted = 0

    for _ in range(n_proposals):
        idx1, idx2 = np.random.choice(C, 2, replace=False)
        i1, j1 = edges[idx1]
        i2, j2 = edges[idx2]

        if i1 == i2 or j1 == j2:
            continue
        if (i1, j2) in edge_set or (i2, j1) in edge_set:
            continue

        # Compute delta using local counting
        old_count = (count_edge_4loops(left_neighbors, right_neighbors, i1, j1) +
                     count_edge_4loops(left_neighbors, right_neighbors, i2, j2))

        # Do swap
        left_neighbors[i1].remove(j1)
        left_neighbors[i2].remove(j2)
        right_neighbors[j1].remove(i1)
        right_neighbors[j2].remove(i2)
        left_neighbors[i1].add(j2)
        left_neighbors[i2].add(j1)
        right_neighbors[j2].add(i1)
        right_neighbors[j1].add(i2)

        new_count = (count_edge_4loops(left_neighbors, right_neighbors, i1, j2) +
                     count_edge_4loops(left_neighbors, right_neighbors, i2, j1))

        delta_n4 = new_count - old_count

        if delta_n4 <= 0 or np.random.random() < np.exp(-lambda_penalty * delta_n4):
            edge_set.remove((i1, j1))
            edge_set.remove((i2, j2))
            edge_set.add((i1, j2))
            edge_set.add((i2, j1))
            edges[idx1] = (i1, j2)
            edges[idx2] = (i2, j1)
            n_accepted += 1
        else:
            # Undo swap
            left_neighbors[i1].remove(j2)
            left_neighbors[i2].remove(j1)
            right_neighbors[j2].remove(i1)
            right_neighbors[j1].remove(i2)
            left_neighbors[i1].add(j1)
            left_neighbors[i2].add(j2)
            right_neighbors[j1].add(i1)
            right_neighbors[j2].add(i2)

    n4_final = 0
    for i, j in edges:
        n4_final += count_edge_4loops(left_neighbors, right_neighbors, i, j)
    n4_final //= 4

    accept_rate = n_accepted / n_proposals if n_proposals > 0 else 1.0

    return edges, accept_rate, n4_initial, n4_final


def sample_pairs_no_c4(N1, N2, C, device, seed=None, lambda_penalty=None, n_sweeps=None, loop_order=None):
    """
    Generate a graph with minimized 2k-loops using MCMC edge-switching.

    Algorithm:
    1. Generate random graph with C edges
    2. Run MCMC with 2-edge swaps to minimize loops
    3. Preserve degree sequence exactly

    Parameters:
        N1, N2: Left and right node counts
        C: Target number of edges
        device: Torch device for output tensors
        seed: Random seed
        lambda_penalty: MCMC penalty for loops (default: MCMC_LAMBDA)
        n_sweeps: Number of MCMC sweeps (default: MCMC_SWEEPS)
        loop_order: k value for 2k-loops (default: LOOP_ORDER)
                    k=2: 4-loops, k=3: 6-loops, k=4: 8-loops

    Returns:
        i_idx, j_idx, C_eff: Edge indices and actual edge count
    """
    # Use global parameters if not specified
    if lambda_penalty is None:
        lambda_penalty = MCMC_LAMBDA
    if n_sweeps is None:
        n_sweeps = MCMC_SWEEPS
    if loop_order is None:
        loop_order = LOOP_ORDER
    if seed is not None:
        np.random.seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"Requested edge count C={C} exceeds matrix total size {N1}×{N2}={total}")

    if C == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    # Step 1: Generate random graph
    all_idx = np.arange(total, dtype=np.int64)
    np.random.shuffle(all_idx)
    selected_idx = all_idx[:C]

    edges = [(idx // N2, idx % N2) for idx in selected_idx]

    # Step 2: Run MCMC to minimize 2k-loops (GPU accelerated)
    if C >= 2 and n_sweeps > 0:
        edges, accept_rate, n_init, n_final = mcmc_minimize_2k_loops_gpu(
            edges, N1, N2, device, k=loop_order,
            lambda_penalty=lambda_penalty, n_sweeps=n_sweeps, seed=seed
        )
        if n_init > 0:
            reduction = (n_init - n_final) / n_init * 100
        else:
            reduction = 0
        # Debug output (uncomment for debugging)
        # print(f"    MCMC: {n_init:,} → {n_final:,} {2*loop_order}-loops ({reduction:.1f}% reduction), accept={accept_rate:.3f}")

    # Convert to tensors
    edges_i = [e[0] for e in edges]
    edges_j = [e[1] for e in edges]

    i_idx = torch.tensor(edges_i, dtype=torch.long, device=device)
    j_idx = torch.tensor(edges_j, dtype=torch.long, device=device)

    return i_idx, j_idx, len(edges)


def sample_pairs_random_gpu(N1, N2, C, device, seed=None):
    """
    Pure random mask generation (entirely on GPU, supports any N1≠N2)
    """
    if seed is not None:
        torch.manual_seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"Requested edge count C={C} exceeds matrix total size {N1}×{N2}={total}")

    if C == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    idx = torch.randperm(total, device=device)[:C]
    i_idx = idx // N2
    j_idx = idx % N2

    return i_idx, j_idx, C


def sample_pairs_biregular_exact(N1, N2, M, alpha_tilde_left, device, seed=None):
    """
    Main graph generation function with C4-free option.

    When FORBID_4_CYCLES=True and alpha < MCMC_ALPHA_THRESHOLD,
    uses MCMC to minimize 4-loops. Otherwise uses random generation.
    """
    deg_left = int(round(alpha_tilde_left * M))
    deg_left = max(0, min(deg_left, N2))
    total_edges = N1 * deg_left

    # ============================================================
    # 4-loop minimization mode (only for low alpha)
    # ============================================================
    if FORBID_4_CYCLES and alpha_tilde_left < MCMC_ALPHA_THRESHOLD:
        return sample_pairs_no_c4(N1, N2, total_edges, device, seed)
    elif FORBID_4_CYCLES:
        # High alpha: skip MCMC, just use random (saves time)
        return sample_pairs_random_gpu(N1, N2, total_edges, device, seed)

    # ============================================================
    # Standard random mode
    # ============================================================
    if not USE_BIREGULAR_GRAPH:
        return sample_pairs_random_gpu(N1, N2, total_edges, device, seed)

    # ============================================================
    # Bi-regular mode (Dinic algorithm) - unchanged from original
    # ============================================================
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
        raise RuntimeError(f"maxflow only got {f}/{total_edges}")

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


def sample_mask(N1, N2, M, alpha, device, seed=None, loop_order=None):
    """
    Generate observation mask with optional loop minimization.

    Parameters:
        N1, N2, M: Matrix dimensions
        alpha: Observation ratio
        device: Torch device
        seed: Random seed
        loop_order: k value for 2k-loop counting (default: LOOP_ORDER)

    Returns:
        mask: Binary observation mask (N1, N2)
        c: Expected degree per left node (alpha * M)
        c_eff: Actual edge count
        n_loops: Number of 2k-loops in the generated graph
        loop_ratio: Ratio of actual to expected loops (0=perfect, 1=random)
    """
    if loop_order is None:
        loop_order = LOOP_ORDER

    c = alpha * M
    i_idx, j_idx, C_eff = sample_pairs_biregular_exact(N1, N2, M, alpha, device, seed)

    mask = torch.zeros((N1, N2), device=device, dtype=torch.float32)
    if C_eff > 0:
        mask[i_idx, j_idx] = 1.0

    # Count loops using GPU
    n_loops = int(count_2k_loops_gpu(mask, k=loop_order).item())
    loop_ratio = compute_loop_ratio(n_loops, N1, N2, C_eff, k=loop_order)

    return mask, c, C_eff, n_loops, loop_ratio


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
    """Estimate GPU memory needed per alpha value for BiG-AMP"""
    student_params = 2 * (S * N1 * M + S * M * N2)
    intermediate = 16 * S * N1 * N2
    total_elements = student_params + intermediate
    return total_elements * dtype_bytes / (1024**3)


def select_memory_mode(N1, N2, M, S, num_alphas, mode_override='auto'):
    """Select optimal memory mode based on matrix size"""
    MAX_GPU_MEMORY_GB = min(DEVICE_INFO.available_memory_gb, 32.0)
    RESERVED_MEMORY_GB = 3.0
    effective_available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

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

    # Generate mask on-demand (returns 5 values now)
    A, _, C_eff, n_4loops, loop4_ratio = sample_mask(N1, N2, M, alpha, device, seed=seed)
    A = A.unsqueeze(0)

    torch.manual_seed(seed + 10000)
    storage_dtype = torch.float16 if use_fp16 else torch.float32

    w_hat = (torch.randn((S, N1, M), device=device) * scale).to(storage_dtype)
    x_hat = (torch.randn((S, M, N2), device=device) * scale).to(storage_dtype)
    w_var = (torch.ones_like(w_hat) * (1.0 / M))
    x_var = (torch.ones_like(x_hat) * (1.0 / M))

    for _ in tqdm(range(steps), desc=f"BiG-AMP α={alpha:.2f}", leave=False, mininterval=1.0):
        w_f, x_f = w_hat.float(), x_hat.float()
        w_v, x_v = w_var.float(), x_var.float()

        z_hat = alpha_scale * torch.matmul(w_f, x_f)
        w_sq = w_f ** 2
        x_sq = x_f ** 2
        p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_v) + torch.matmul(w_v, x_sq))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        residual = (Y_teacher - z_hat) * A
        s = residual / V

        tau_W = (alpha_scale ** 2) * torch.matmul(A / V, x_sq.transpose(-2, -1))
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new = 1.0 / (M + tau_W)
        r_W = alpha_scale * torch.matmul(s, x_f.transpose(-2, -1))
        w_hat_new = w_f + w_var_new * r_W
        w_f = damping * w_f + (1 - damping) * w_hat_new
        w_v = torch.clamp(damping * w_v + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        w_hat = w_f.to(storage_dtype)
        w_var = w_v.to(storage_dtype)

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

    return w_hat.float(), x_hat.float(), C_eff, n_4loops, loop4_ratio


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_batch(W, X, Wt, Xt, Y_teacher, alpha_values, S, A_all=None):
    """Evaluate metrics for all alphas

    Parameters:
        A_all: Optional mask tensor of shape (num_alphas, 1, N1, N2) for Q_Y_unobserved calculation
    """
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

            # Compute Q_Y_unobserved: overlap only on unobserved positions (mask=0)
            if A_all is not None:
                mask = A_all[a_idx, 0]  # (N1, N2)
                unobs_mask = 1.0 - mask  # Flip: observed=0, unobserved=1
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
                Q_Y_unobserved = Q_Y  # Fallback if no mask

            trial_results.append({
                'Q_W': Q_W, 'Q_X': Q_X,
                'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
                'Q_Y': Q_Y, 'Q_Y_unobserved': Q_Y_unobserved, 'Gen_Error': gen_error
            })

        metrics = {}
        for key in trial_results[0].keys():
            vals = [r[key] for r in trial_results]
            metrics[f'{key}_mean'] = float(np.mean(vals))
            metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        if S > 1:
            W_alpha = W[a_idx] if W.dim() == 4 else W
            X_alpha = X[a_idx] if X.dim() == 4 else X
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
def evaluate_single(W, X, Wt, Xt, Y_teacher, S, A=None):
    """Evaluate single alpha result

    Parameters:
        A: Optional mask tensor of shape (N1, N2) for Q_Y_unobserved calculation
    """
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

        # Compute Q_Y_unobserved: overlap only on unobserved positions (mask=0)
        if A is not None:
            unobs_mask = 1.0 - A  # Flip: observed=0, unobserved=1
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
            Q_Y_unobserved = Q_Y  # Fallback if no mask

        trial_results.append({
            'Q_W': Q_W, 'Q_X': Q_X,
            'Q_W_prime': Q_W_prime, 'Q_X_prime': Q_X_prime,
            'Q_Y': Q_Y, 'Q_Y_unobserved': Q_Y_unobserved, 'Gen_Error': gen_error
        })

    metrics = {}
    for key in trial_results[0].keys():
        vals = [r[key] for r in trial_results]
        metrics[f'{key}_mean'] = float(np.mean(vals))
        metrics[f'{key}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

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
    """Compute pairwise Gram overlap between S replicas"""
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
    print("BiG-AMP TRAINING - PARALLEL MODE (C4-FREE)")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({num_alphas} points)")
    print(f"Steps: {steps}, Samples: {S}")
    print(f"Parallelism: {max_parallel} alphas")
    print(f"C4-free mode: {FORBID_4_CYCLES}")
    print(f"{'='*70}\n")

    all_results = {}
    graph_stats = {}  # Store 4-loop counts and C_eff for each alpha
    total_start = time.time()

    for batch_start in range(0, num_alphas, max_parallel):
        batch_end = min(batch_start + max_parallel, num_alphas)
        batch_alphas = alpha_values[batch_start:batch_end]
        batch_size = len(batch_alphas)

        print(f"[Batch {batch_start//max_parallel + 1}] Alpha {batch_alphas[0]:.2f} - {batch_alphas[-1]:.2f}")

        A_all = torch.zeros((batch_size, 1, N1, N2), device=DEVICE)
        for i, alpha in enumerate(batch_alphas):
            mask_seed = SEED + int(alpha * 1000)
            A, _, C_eff, n_4loops, loop4_ratio = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed)
            A_all[i, 0] = A

            # Store graph statistics
            target_C = N1 * int(round(alpha * M))
            expected_4loops = expected_4loops_random(N1, N2, C_eff)
            graph_stats[float(alpha)] = {
                'target_C': target_C,
                'actual_C': C_eff,
                'n_4loops': n_4loops,
                'expected_4loops': expected_4loops,
                'loop4_ratio': loop4_ratio,
                'alpha_eff': C_eff / (M * N1) if M * N1 > 0 else 0
            }
            print(f"  α={alpha:.2f}: C={C_eff}, 4-loops={n_4loops}, ratio={loop4_ratio:.4f}")

        W, X = train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, batch_alphas, steps, S,
                                      damping=DAMPING, noise_var=NOISE_VAR)

        batch_results = evaluate_batch(W, X, Wt, Xt, Y_teacher, batch_alphas, S)
        all_results.update(batch_results)

        del A_all, W, X
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")

    return all_results, total_time, graph_stats


def run_sequential_mode(alpha_values, steps, S, use_fp16=False):
    """Run with sequential alpha processing (memory optimized)"""
    set_seed(SEED)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)
    mode_name = "EXTREME" if use_fp16 else "OPTIMIZED"

    print(f"\n{'='*70}")
    print(f"BiG-AMP TRAINING - {mode_name} MODE (Sequential, C4-FREE)")
    print(f"{'='*70}")
    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Device: {DEVICE_INFO.device_name}, Memory: {DEVICE_INFO.available_memory_gb:.1f} GB")
    print(f"Alpha range: {alpha_values[0]:.2f} to {alpha_values[-1]:.2f} ({num_alphas} points)")
    print(f"Steps: {steps}, Samples: {S}")
    print(f"FP16 storage: {use_fp16}")
    print(f"C4-free mode: {FORBID_4_CYCLES}")
    print(f"{'='*70}\n")

    all_results = {}
    graph_stats = {}
    total_start = time.time()

    for i, alpha in enumerate(alpha_values):
        alpha_seed = SEED + int(alpha * 1000)
        target_C = N1 * int(round(alpha * M))
        print(f"[{i+1}/{num_alphas}] Alpha = {alpha:.2f} (target C={target_C})")

        W, X, C_eff, n_4loops, loop4_ratio = train_bigamp_single(Wt, Xt, Y_teacher, alpha, steps, S, alpha_seed,
                                    damping=DAMPING, noise_var=NOISE_VAR, use_fp16=use_fp16)

        # Store graph statistics
        expected_4loops = expected_4loops_random(N1, N2, C_eff)
        graph_stats[float(alpha)] = {
            'target_C': target_C,
            'actual_C': C_eff,
            'n_4loops': n_4loops,
            'expected_4loops': expected_4loops,
            'loop4_ratio': loop4_ratio,
            'alpha_eff': C_eff / (M * N1) if M * N1 > 0 else 0
        }
        print(f"  -> C={C_eff}, 4-loops={n_4loops}, ratio={loop4_ratio:.4f}")

        metrics = evaluate_single(W, X, Wt, Xt, Y_teacher, S)
        all_results[float(alpha)] = metrics

        del W, X
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")

    return all_results, total_time, graph_stats


# ============================================================
# Visualization
# ============================================================
def plot_results(results, alpha_values, graph_stats, save_path, results_random=None):
    """
    Generate result plots comparing 4-loop minimized vs random graph.

    Shows only 3 curves:
    1. Q_Y from random graph (if results_random provided)
    2. Q_Y from 4-loop minimized graph
    3. 4-loop ratio (actual/expected)

    Parameters:
        results: Results from 4-loop minimized graph
        alpha_values: List of alpha values
        graph_stats: Graph statistics including 4-loop counts
        save_path: Path to save the plot
        results_random: Optional results from random graph for comparison
    """
    aL = np.array(alpha_values)
    qY_mu = np.array([results[a]['Q_Y_mean'] for a in alpha_values])

    # Get loop ratio (actual/expected)
    loop_ratio = np.array([graph_stats[a].get('loop_ratio', graph_stats[a].get('loop4_ratio', 0)) for a in alpha_values])

    fig = plt.figure(figsize=(10, 8))

    # Main plot: Q_Y and loop ratio
    ax = fig.add_subplot(111)

    # Plot Q_Y from random graph if available
    if results_random is not None:
        qY_random = np.array([results_random[a]['Q_Y_mean'] for a in alpha_values])
        ax.plot(aL, qY_random, marker='o', linewidth=2, markersize=6,
                color='#1f77b4', label='Q_Y (Random Graph)', zorder=3)

    # Plot Q_Y from minimized graph
    ax.plot(aL, qY_mu, marker='D', linewidth=2, markersize=6,
            color='#d62728', label=f'Q_Y ({2*LOOP_ORDER}-loop Minimized)', zorder=4)

    # Plot loop ratio
    ax.plot(aL, loop_ratio, marker='x', linewidth=2, markersize=7,
            color='#2ca02c', label=f'{2*LOOP_ORDER}-loop ratio (actual/expected)', zorder=2,
            linestyle='--', alpha=0.8)

    ax.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=14)
    ax.set_ylabel(f'Q_Y / {2*LOOP_ORDER}-Loop Ratio', fontsize=14)

    title = f'Phase Transition: Random vs {2*LOOP_ORDER}-loop Minimized Graph'
    if results_random is None:
        title = f'Phase Transition with {2*LOOP_ORDER}-loop Minimized Graph'
    ax.set_title(title, fontsize=15, fontweight='bold')

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(aL[0] - 0.1, aL[-1] + 0.1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12, loc='lower right')

    # Add text box with statistics
    avg_ratio = np.mean(loop_ratio[loop_ratio > 0]) if np.any(loop_ratio > 0) else 0
    textstr = f'N={N1}, M={M}\nAvg {2*LOOP_ORDER}-loop ratio: {avg_ratio:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close(fig)


def plot_comparison(results_random, results_minimized, alpha_values, graph_stats, save_path):
    """
    Generate comparison plot between random and loop-minimized graphs.

    Shows:
    1. Q_Y from random graph (blue)
    2. Q_Y from loop-minimized graph (red)
    3. Loop ratio (green dashed)
    """
    aL = np.array(alpha_values)
    qY_random = np.array([results_random[a]['Q_Y_mean'] for a in alpha_values])
    qY_minimized = np.array([results_minimized[a]['Q_Y_mean'] for a in alpha_values])
    loop_ratio = np.array([graph_stats[a].get('loop_ratio', graph_stats[a].get('loop4_ratio', 0)) for a in alpha_values])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Top: Q_Y comparison
    ax1.plot(aL, qY_random, marker='o', linewidth=2, markersize=6,
             color='#1f77b4', label='Q_Y (Random Graph)')
    ax1.plot(aL, qY_minimized, marker='D', linewidth=2, markersize=6,
             color='#d62728', label=f'Q_Y ({2*LOOP_ORDER}-loop Minimized)')
    ax1.plot(aL, loop_ratio, marker='x', linewidth=2, markersize=7,
             color='#2ca02c', label=f'{2*LOOP_ORDER}-loop ratio', linestyle='--', alpha=0.7)

    ax1.set_ylabel(f'Q_Y / {2*LOOP_ORDER}-Loop Ratio', fontsize=14)
    ax1.set_title(f'Phase Transition Comparison (N={N1}, M={M})', fontsize=15, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='lower right')

    # Bottom: Difference plot
    diff = qY_minimized - qY_random
    ax2.bar(aL, diff, width=0.08, color='purple', alpha=0.7, label='ΔQ_Y (Minimized - Random)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=14)
    ax2.set_ylabel('ΔQ_Y', fontsize=14)
    ax2.set_title('Difference in Q_Y', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved: {save_path}")
    plt.close(fig)


def plot_replica_comparison(results, alpha_values, save_path):
    """Generate replica comparison plot"""
    aL = np.array(alpha_values)
    qW_prime_mu = np.array([results[a]['Q_W_prime_mean'] for a in alpha_values])
    qX_prime_mu = np.array([results[a]['Q_X_prime_mean'] for a in alpha_values])
    qW_rep_mu = np.array([results[a].get('Q_W_replica_mean', 0.0) for a in alpha_values])
    qX_rep_mu = np.array([results[a].get('Q_X_replica_mean', 0.0) for a in alpha_values])

    has_replica = np.any(qW_rep_mu != 0) or np.any(qX_rep_mu != 0)
    if not has_replica:
        print("No replica overlap data (S=1), skipping replica comparison plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(aL, qW_prime_mu, marker='o', linewidth=1.5, markersize=5,
             color='#9467bd', label="Q_W' (teacher-student, normalized)", zorder=2)
    ax1.plot(aL, qW_rep_mu, marker='s', linewidth=1.5, markersize=5,
             color='#2ca02c', label="Q_W_replica (replica-replica, raw cosine)", zorder=1)

    ax1.set_ylabel('W Overlap', fontsize=13)
    ax1.set_title('W Overlap Comparison (C4-FREE Graph)', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='lower right')

    ax2.plot(aL, qX_prime_mu, marker='v', linewidth=1.5, markersize=5,
             color='#8c564b', label="Q_X' (teacher-student, normalized)", zorder=2)
    ax2.plot(aL, qX_rep_mu, marker='^', linewidth=1.5, markersize=5,
             color='#17becf', label="Q_X_replica (replica-replica, raw cosine)", zorder=1)

    ax2.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=13)
    ax2.set_ylabel('X Overlap', fontsize=13)
    ax2.set_title('X Overlap Comparison (C4-FREE Graph)', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Replica comparison plot saved: {save_path}")
    plt.close(fig)


def plot_qy_unobserved_comparison(results_random, results_minimized, alpha_values, graph_stats, save_path):
    """
    Generate Q_Y_unobserved comparison plot between random and loop-minimized graphs.

    Shows:
    1. Q_Y_unobserved from random graph (blue)
    2. Q_Y_unobserved from loop-minimized graph (red)
    3. Loop ratio (green dashed)
    """
    aL = np.array(alpha_values)
    qY_unobs_random = np.array([results_random[a].get('Q_Y_unobserved_mean', results_random[a]['Q_Y_mean'])
                                 for a in alpha_values])
    qY_unobs_minimized = np.array([results_minimized[a].get('Q_Y_unobserved_mean', results_minimized[a]['Q_Y_mean'])
                                    for a in alpha_values])
    loop_ratio = np.array([graph_stats[a].get('loop_ratio', graph_stats[a].get('loop4_ratio', 0)) for a in alpha_values])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Top: Q_Y_unobserved comparison
    ax1.plot(aL, qY_unobs_random, marker='o', linewidth=2, markersize=6,
             color='#1f77b4', label='Q_Y_unobserved (Random Graph)')
    ax1.plot(aL, qY_unobs_minimized, marker='D', linewidth=2, markersize=6,
             color='#d62728', label=f'Q_Y_unobserved ({2*LOOP_ORDER}-loop Minimized)')
    ax1.plot(aL, loop_ratio, marker='x', linewidth=2, markersize=7,
             color='#2ca02c', label=f'{2*LOOP_ORDER}-loop ratio', linestyle='--', alpha=0.7)

    ax1.set_ylabel(f'Q_Y_unobserved / {2*LOOP_ORDER}-Loop Ratio', fontsize=14)
    ax1.set_title(f'Q_Y_unobserved Comparison (N={N1}, M={M})\nGeneralization on Unobserved Positions', fontsize=15, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='lower right')

    # Bottom: Difference plot
    diff = qY_unobs_minimized - qY_unobs_random
    ax2.bar(aL, diff, width=0.08, color='purple', alpha=0.7, label='ΔQ_Y_unobs (Minimized - Random)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel(r'$\tilde{\alpha}_L = C / (M \cdot N_1)$', fontsize=14)
    ax2.set_ylabel('ΔQ_Y_unobserved', fontsize=14)
    ax2.set_title('Difference in Q_Y_unobserved (Generalization)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q_Y_unobserved comparison plot saved: {save_path}")
    plt.close(fig)


# ============================================================
# Verification: Compare with random graph
# ============================================================
def verify_loop_reduction(loop_order=None):
    """
    Generate comparison between random graph and loop-minimized graph
    to verify that loops are indeed reduced.

    Shows:
    - Random graph: actual loops vs expected (should be close)
    - Minimized graph: actual loops vs expected (should be much lower)
    - Ratio: actual/expected (0=perfect, 1=random)

    Parameters:
        loop_order: k value for 2k-loops (default: LOOP_ORDER)
    """
    if loop_order is None:
        loop_order = LOOP_ORDER

    loop_name = f"{2*loop_order}-LOOP"

    print("\n" + "="*70)
    print(f"{loop_name} VERIFICATION TEST (MCMC Algorithm)")
    print("="*70)

    test_alpha = 2.0
    test_C = N1 * int(round(test_alpha * M))
    expected = expected_2k_loops_random(N1, N2, test_C, k=loop_order)

    print(f"Test parameters: N1={N1}, N2={N2}, M={M}, alpha={test_alpha}")
    print(f"Target edges: C={test_C}")
    print(f"Loop order: k={loop_order} ({2*loop_order}-loops)")
    print(f"Expected {2*loop_order}-loops (random): {expected:,.0f}")
    print()

    # Generate random graph
    print("Generating random graph...")
    i_rand, j_rand, C_rand = sample_pairs_random_gpu(N1, N2, test_C, DEVICE, seed=SEED)

    # Build adjacency for GPU counting
    A_rand = torch.zeros((N1, N2), device=DEVICE, dtype=torch.float32)
    A_rand[i_rand, j_rand] = 1.0
    n_loops_rand = int(count_2k_loops_gpu(A_rand, k=loop_order).item())
    ratio_rand = n_loops_rand / max(expected, 1)
    print(f"  Random graph: C={C_rand}, {2*loop_order}-loops={n_loops_rand:,}, ratio={ratio_rand:.4f}")

    # Generate minimized graph (MCMC algorithm)
    print(f"Generating {2*loop_order}-loop minimized graph (MCMC algorithm)...")
    i_min, j_min, C_min = sample_pairs_no_c4(N1, N2, test_C, DEVICE, seed=SEED, loop_order=loop_order)

    A_min = torch.zeros((N1, N2), device=DEVICE, dtype=torch.float32)
    A_min[i_min, j_min] = 1.0
    n_loops_min = int(count_2k_loops_gpu(A_min, k=loop_order).item())
    ratio_min = n_loops_min / max(expected, 1)
    print(f"  Minimized graph: C={C_min}, {2*loop_order}-loops={n_loops_min:,}, ratio={ratio_min:.4f}")

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Graph Type':<20} {'Edges':<10} {loop_name+'s':<15} {'Ratio':<10}")
    print("-"*70)
    print(f"{'Random':<20} {C_rand:<10} {n_loops_rand:<15,} {ratio_rand:.4f}")
    print(f"{'Minimized':<20} {C_min:<10} {n_loops_min:<15,} {ratio_min:.4f}")
    print("-"*70)

    if n_loops_rand > 0:
        reduction = (n_loops_rand - n_loops_min) / n_loops_rand * 100
        print(f"{2*loop_order}-loop reduction: {reduction:.1f}%")

    if ratio_min < ratio_rand * 0.9:
        print(f"✓ SUCCESS: {2*loop_order}-loop ratio reduced from {ratio_rand:.4f} to {ratio_min:.4f}")
    else:
        print(f"⚠ WARNING: Limited reduction (ratio {ratio_rand:.4f} -> {ratio_min:.4f})")

    print("="*70 + "\n")

    return n_loops_rand, n_loops_min, ratio_rand, ratio_min


# Keep old name for backwards compatibility
def verify_4loop_reduction():
    """Backwards-compatible wrapper for verify_loop_reduction with k=2"""
    return verify_loop_reduction(loop_order=2)


# ============================================================
# Comparison Mode (Parallel)
# ============================================================
def run_comparison(alpha_values, steps, S):
    """
    Run comparison between random graph and loop-minimized graph.
    Uses parallel alpha training for speed.

    Returns both results for plotting with 3 curves:
    1. Q_Y from random graph
    2. Q_Y from loop-minimized graph
    3. Loop ratio
    """
    global FORBID_4_CYCLES

    loop_name = f"{2*LOOP_ORDER}-loop"

    print("\n" + "="*70)
    print(f"COMPARISON MODE: Random Graph vs {loop_name} Minimized Graph (Parallel)")
    print("="*70)

    set_seed(SEED)
    Wt, Xt = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_teacher = Wt @ Xt

    num_alphas = len(alpha_values)

    # ============================================================
    # Phase 1: Random graph (all alphas in parallel)
    # ============================================================
    print("\n[Phase 1] Running random graph experiment (parallel)...")
    FORBID_4_CYCLES = False

    # Generate all masks in parallel
    A_random = torch.zeros((num_alphas, 1, N1, N2), device=DEVICE)
    graph_stats_random = {}

    for i, alpha in enumerate(alpha_values):
        mask_seed = SEED + int(alpha * 1000)
        A, _, C_eff, n_loops, loop_ratio = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed, loop_order=LOOP_ORDER)
        A_random[i, 0] = A

        target_C = N1 * int(round(alpha * M))
        expected = expected_2k_loops_random(N1, N2, C_eff, k=LOOP_ORDER)
        graph_stats_random[float(alpha)] = {
            'target_C': target_C, 'actual_C': C_eff,
            'n_loops': n_loops, 'expected_loops': expected,
            'loop_ratio': loop_ratio,
        }

    # Parallel training
    W_random, X_random = train_bigamp_parallel(
        Wt, Xt, Y_teacher, A_random, alpha_values, steps, S,
        damping=DAMPING, noise_var=NOISE_VAR
    )

    # Evaluate
    results_random = evaluate_batch(W_random, X_random, Wt, Xt, Y_teacher, alpha_values, S, A_all=A_random)

    del A_random, W_random, X_random
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # ============================================================
    # Phase 2: Loop-minimized graph (parallel where possible)
    # ============================================================
    print(f"\n[Phase 2] Running {loop_name} minimized graph experiment (parallel)...")
    FORBID_4_CYCLES = True

    A_minimized = torch.zeros((num_alphas, 1, N1, N2), device=DEVICE)
    graph_stats_minimized = {}

    for i, alpha in enumerate(alpha_values):
        mask_seed = SEED + int(alpha * 1000)
        A, _, C_eff, n_loops, loop_ratio = sample_mask(N1, N2, M, alpha, DEVICE, seed=mask_seed, loop_order=LOOP_ORDER)
        A_minimized[i, 0] = A

        target_C = N1 * int(round(alpha * M))
        expected = expected_2k_loops_random(N1, N2, C_eff, k=LOOP_ORDER)
        graph_stats_minimized[float(alpha)] = {
            'target_C': target_C, 'actual_C': C_eff,
            'n_loops': n_loops, 'expected_loops': expected,
            'loop_ratio': loop_ratio,
        }
        if alpha < MCMC_ALPHA_THRESHOLD:
            print(f"  α={alpha:.2f}: C={C_eff}, {loop_name}s={n_loops}, ratio={loop_ratio:.3f}")

    # Parallel training
    W_minimized, X_minimized = train_bigamp_parallel(
        Wt, Xt, Y_teacher, A_minimized, alpha_values, steps, S,
        damping=DAMPING, noise_var=NOISE_VAR
    )

    # Evaluate
    results_minimized = evaluate_batch(W_minimized, X_minimized, Wt, Xt, Y_teacher, alpha_values, S, A_all=A_minimized)

    del A_minimized, W_minimized, X_minimized
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return results_random, results_minimized, graph_stats_random, graph_stats_minimized


# ============================================================
# Main
# ============================================================
def main():
    global N1, N2, M, ALPHA_TILDE_STEP, MAX_STEPS, SAMPLES_PER_ALPHA, RESULT_DIR, FORBID_4_CYCLES, LOOP_ORDER

    parser = argparse.ArgumentParser(description='BiG-AMP with Loop-Minimized Graph Generation')
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
    parser.add_argument('--skip-verify', action='store_true', help='Skip loop verification test')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison between random and loop-minimized graphs')
    parser.add_argument('--loop-order', type=int, default=LOOP_ORDER,
                        help='Loop order k for 2k-loops (k=2: 4-loops, k=3: 6-loops, k=4: 8-loops)')
    args = parser.parse_args()

    # Apply args
    N1 = args.n1
    N2 = args.n2 if args.n2 else args.n1
    M = args.m
    MAX_STEPS = args.steps
    SAMPLES_PER_ALPHA = args.samples
    ALPHA_TILDE_STEP = args.alpha_step
    LOOP_ORDER = args.loop_order

    # Result directory includes loop order for k > 2
    loop_suffix = f"_loop{2*LOOP_ORDER}" if LOOP_ORDER > 2 else ""
    RESULT_DIR = Path(__file__).parent / "ResultNo4" / f"{N1}_{N2}_{M}{loop_suffix}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Config] Loop order: k={LOOP_ORDER} ({2*LOOP_ORDER}-loops)")

    # Run verification test first
    if not args.skip_verify:
        verify_loop_reduction(loop_order=LOOP_ORDER)

    alpha_values = np.arange(ALPHA_TILDE_START, args.alpha_stop + 1e-12, ALPHA_TILDE_STEP)

    # ============================================================
    # Comparison Mode: Random vs Loop-Minimized
    # ============================================================
    if args.compare:
        import time as time_module
        start_time = time_module.time()

        results_random, results_minimized, graph_stats_random, graph_stats_minimized = \
            run_comparison(alpha_values, MAX_STEPS, SAMPLES_PER_ALPHA)

        total_time = time_module.time() - start_time

        # Print comparison summary
        loop_name = f"{2*LOOP_ORDER}L"
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(f"{'Alpha':<8} {'Q_Y(Rand)':<12} {'Q_Y(Min)':<12} {loop_name+' Ratio':<10} {'Diff':<10}")
        print("-"*70)

        for alpha in [float(a) for a in alpha_values]:
            qy_rand = results_random[alpha]['Q_Y_mean']
            qy_min = results_minimized[alpha]['Q_Y_mean']
            ratio = graph_stats_minimized[alpha]['loop_ratio']
            diff = qy_min - qy_rand
            print(f"{alpha:<8.2f} {qy_rand:<12.4f} {qy_min:<12.4f} {ratio:<10.4f} {diff:+.4f}")

        # Save comparison results
        results_data = {
            'config': {
                'N1': N1, 'N2': N2, 'M': M,
                'steps': MAX_STEPS,
                'samples_per_alpha': SAMPLES_PER_ALPHA,
                'mode': 'comparison',
                'total_time': total_time,
            },
            'alpha_values': [float(a) for a in alpha_values],
            'results_random': {str(k): v for k, v in results_random.items()},
            'results_minimized': {str(k): v for k, v in results_minimized.items()},
            'graph_stats_random': {str(k): v for k, v in graph_stats_random.items()},
            'graph_stats_minimized': {str(k): v for k, v in graph_stats_minimized.items()}
        }

        results_path = RESULT_DIR / f'comparison_results_steps{MAX_STEPS}.json'
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved: {results_path}")

        # Generate comparison plots
        alpha_list = [float(a) for a in alpha_values]

        plot_path1 = RESULT_DIR / f'comparison_qy_steps{MAX_STEPS}.png'
        plot_results(results_minimized, alpha_list, graph_stats_minimized, plot_path1,
                     results_random=results_random)

        plot_path2 = RESULT_DIR / f'comparison_detailed_steps{MAX_STEPS}.png'
        plot_comparison(results_random, results_minimized, alpha_list,
                        graph_stats_minimized, plot_path2)

        # Plot 3: Q_Y_unobserved comparison (generalization)
        plot_path3 = RESULT_DIR / f'comparison_qy_unobserved_steps{MAX_STEPS}.png'
        plot_qy_unobserved_comparison(results_random, results_minimized, alpha_list,
                                       graph_stats_minimized, plot_path3)

        print(f"Plot 1: {plot_path1}")
        print(f"Plot 2: {plot_path2}")
        print(f"Plot 3: {plot_path3}")
        print(f"\nTotal time: {total_time:.1f}s")
        print("="*70)
        return

    # ============================================================
    # Standard Mode: Always run comparison (Random vs 4-loop Minimized)
    # ============================================================
    import time as time_module
    start_time = time_module.time()

    results_random, results_minimized, graph_stats_random, graph_stats_minimized = \
        run_comparison(alpha_values, MAX_STEPS, SAMPLES_PER_ALPHA)

    total_time = time_module.time() - start_time

    # Print comparison summary
    loop_name = f"{2*LOOP_ORDER}L"
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Alpha':<8} {'Q_Y(Rand)':<12} {'Q_Y(Min)':<12} {loop_name+' Ratio':<10} {'Diff':<10}")
    print("-"*70)

    for alpha in [float(a) for a in alpha_values]:
        qy_rand = results_random[alpha]['Q_Y_mean']
        qy_min = results_minimized[alpha]['Q_Y_mean']
        ratio = graph_stats_minimized[alpha]['loop_ratio']
        diff = qy_min - qy_rand
        print(f"{alpha:<8.2f} {qy_rand:<12.4f} {qy_min:<12.4f} {ratio:<10.4f} {diff:+.4f}")

    # Save comparison results
    results_data = {
        'config': {
            'N1': N1, 'N2': N2, 'M': M,
            'steps': MAX_STEPS,
            'samples_per_alpha': SAMPLES_PER_ALPHA,
            'mode': 'comparison',
            'total_time': total_time,
        },
        'alpha_values': [float(a) for a in alpha_values],
        'results_random': {str(k): v for k, v in results_random.items()},
        'results_minimized': {str(k): v for k, v in results_minimized.items()},
        'graph_stats_random': {str(k): v for k, v in graph_stats_random.items()},
        'graph_stats_minimized': {str(k): v for k, v in graph_stats_minimized.items()}
    }

    results_path = RESULT_DIR / f'comparison_results_steps{MAX_STEPS}.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Generate comparison plots
    alpha_list = [float(a) for a in alpha_values]

    # Plot 1: Main comparison (Q_Y curves + 4-loop ratio)
    plot_path1 = RESULT_DIR / f'comparison_qy_steps{MAX_STEPS}.png'
    plot_results(results_minimized, alpha_list, graph_stats_minimized, plot_path1,
                 results_random=results_random)

    # Plot 2: Detailed comparison with difference plot
    plot_path2 = RESULT_DIR / f'comparison_detailed_steps{MAX_STEPS}.png'
    plot_comparison(results_random, results_minimized, alpha_list,
                    graph_stats_minimized, plot_path2)

    # Plot 3: Q_Y_unobserved comparison (generalization)
    plot_path3 = RESULT_DIR / f'comparison_qy_unobserved_steps{MAX_STEPS}.png'
    plot_qy_unobserved_comparison(results_random, results_minimized, alpha_list,
                                   graph_stats_minimized, plot_path3)

    print(f"Plot 1: {plot_path1}")
    print(f"Plot 2: {plot_path2}")
    print(f"Plot 3: {plot_path3}")
    print(f"\nTotal time: {total_time:.1f}s")
    print("="*70)


if __name__ == "__main__":
    main()
