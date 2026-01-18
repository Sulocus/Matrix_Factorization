"""
Low-loop graph generation using MCMC edge-switching.

Generates graphs with minimized short cycles (4-loops, 6-loops, etc.)
to study the effect of loops on phase transition behavior.
"""

from typing import Tuple
import torch
import numpy as np

from ..registry import register_graph
from .base import GraphBase


def count_2k_loops_gpu(A: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Count 2k-loops in bipartite graph using matrix powers.

    For bipartite graph with adjacency matrix A (N1 x N2):
    - B = A @ A.T gives common neighbor counts
    - 2k-loop corresponds to closed walks of length 2k

    Args:
        A: Adjacency matrix (N1, N2)
        k: Loop order (k=2 for 4-loops, k=3 for 6-loops, k=4 for 8-loops)

    Returns:
        Number of 2k-loops
    """
    B = A @ A.T  # B[i,i'] = number of common j-neighbors

    if k == 2:
        # Exact 4-loop count: sum_{i<i'} C(B[i,i'], 2)
        upper = torch.triu(B, diagonal=1)
        return (upper * (upper - 1)).sum() // 2

    elif k == 3:
        # 6-loops: related to trace(B^3) / 6
        B2 = B @ B
        B3 = B2 @ B
        trace_B3 = torch.trace(B3)
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


def mcmc_minimize_2k_loops_gpu(
    edges: list,
    N1: int,
    N2: int,
    device: torch.device,
    k: int = 2,
    n_sweeps: int = 5,
    seed: int = None,
) -> Tuple[list, float, int, int]:
    """
    GPU-accelerated 2k-loop reduction using MCMC edge-switching.

    Algorithm:
    1. Compute edge contribution scores based on loop participation
    2. Sort edges by score (high = many loops)
    3. Try swapping high-score edges with low-score edges
    4. Accept swap if it reduces total loop count

    Args:
        edges: List of (i, j) tuples representing edges
        N1, N2: Number of left and right nodes
        device: Torch device
        k: Loop order (k=2 for 4-loops, k=3 for 6-loops, k=4 for 8-loops)
        n_sweeps: Number of MCMC sweeps
        seed: Random seed

    Returns:
        final_edges: List of (i, j) tuples after optimization
        accept_rate: Fraction of accepted swaps
        n_initial: Initial loop count
        n_final: Final loop count
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
    n_attempts = 0

    for sweep in range(n_sweeps):
        # Compute B = A @ A.T (common neighbor matrix)
        B = A @ A.T

        # Compute edge contribution score based on loop order
        if k == 2:
            score_matrix = B @ A
        elif k == 3:
            B2 = B @ B
            score_matrix = B2 @ A
        elif k == 4:
            B2 = B @ B
            B3 = B2 @ B
            score_matrix = B3 @ A
        else:
            Bk = B.clone()
            for _ in range(k - 2):
                Bk = Bk @ B
            score_matrix = Bk @ A

        edge_scores = score_matrix[edges_i, edges_j]

        # Sort edges by score (descending)
        sorted_idx = edge_scores.argsort(descending=True)

        # Try swapping top-scoring edges with low-scoring ones
        n_top = min(C // 4, 200)

        for kk in range(n_top):
            n_attempts += 1
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
    accept_rate = n_accepted / n_attempts if n_attempts > 0 else 1.0

    return final_edges, accept_rate, n_initial, n_final


@register_graph(
    key="low_loop",
    name="Low-Loop Graph (MCMC)",
    description="Minimized short cycles via MCMC edge-switching",
    default_params={
        "loop_order": 2,
        "n_sweeps": 5,
        "alpha_threshold": 0.8,
    },
)
class LowLoopGraph(GraphBase):
    """
    Low-loop graph generator using MCMC edge-switching.

    Generates graphs with minimized short cycles (4-loops by default)
    to study the effect of loops on phase transition behavior.

    Theory:
    - C4-free graphs are only possible for alpha < 0.35 (Kovari-Sos-Turan theorem)
    - For higher alpha, MCMC can reduce 4-loops by 20-30%
    - Useful for studying finite-size effects and AMP convergence

    Args:
        loop_order: k value for 2k-loops (default 2 for 4-loops)
        n_sweeps: Number of MCMC sweeps (more = better reduction)
        alpha_threshold: Only run MCMC when alpha < threshold (saves time)
    """

    def __init__(
        self,
        loop_order: int = 2,
        n_sweeps: int = 5,
        alpha_threshold: float = 0.8,
    ):
        self.loop_order = loop_order
        self.n_sweeps = n_sweeps
        self.alpha_threshold = alpha_threshold

    def generate(
        self,
        N1: int,
        N2: int,
        M: int,
        alpha: float,
        device: torch.device,
        seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generate edge indices with minimized short loops.

        Args:
            N1: Number of rows
            N2: Number of columns
            M: Latent dimension
            alpha: Sparsity parameter
            device: Torch device
            seed: Random seed

        Returns:
            i_idx, j_idx, C: Edge indices and count
        """
        # Calculate degree and total edges
        deg_left = int(round(alpha * M))
        deg_left = max(0, min(deg_left, N2))
        C = N1 * deg_left

        if seed is not None:
            np.random.seed(seed)

        total = N1 * N2
        if C > total:
            raise RuntimeError(
                f"Requested edge count C={C} exceeds matrix total size {N1}×{N2}={total}"
            )

        if C == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                0,
            )

        # Step 1: Generate random graph
        all_idx = np.arange(total, dtype=np.int64)
        np.random.shuffle(all_idx)
        selected_idx = all_idx[:C]
        edges = [(idx // N2, idx % N2) for idx in selected_idx]

        # Step 2: Run MCMC to minimize loops (only for low alpha)
        if C >= 2 and self.n_sweeps > 0 and alpha < self.alpha_threshold:
            edges, _, _, _ = mcmc_minimize_2k_loops_gpu(
                edges, N1, N2, device,
                k=self.loop_order,
                n_sweeps=self.n_sweeps,
                seed=seed,
            )

        # Convert to tensors
        edges_i = [e[0] for e in edges]
        edges_j = [e[1] for e in edges]

        i_idx = torch.tensor(edges_i, dtype=torch.long, device=device)
        j_idx = torch.tensor(edges_j, dtype=torch.long, device=device)

        return i_idx, j_idx, len(edges)


# Utility functions for external use
def count_4loops(A: torch.Tensor) -> int:
    """Count 4-loops in adjacency matrix."""
    return int(count_2k_loops_gpu(A, k=2).item())


def count_6loops(A: torch.Tensor) -> int:
    """Count 6-loops in adjacency matrix."""
    return int(count_2k_loops_gpu(A, k=3).item())


def expected_2k_loops_random(N1: int, N2: int, C: int, k: int = 2) -> float:
    """
    Expected number of 2k-loops in a random bipartite graph.

    For k=2 (4-loops):
    E[#4-loops] ≈ C(C,2) * C(N2,2) / (N1*N2)^2 * N1 * (N1-1) / 2
               ≈ (C^2 * (C-1)^2) / (4 * N1 * N2^2)

    Args:
        N1, N2: Number of left and right nodes
        C: Number of edges
        k: Loop order

    Returns:
        Expected number of 2k-loops
    """
    if C < 2 or N1 < 2 or N2 < 2:
        return 0.0

    p = C / (N1 * N2)  # Edge probability

    if k == 2:
        # 4-loops: pairs of left nodes sharing pairs of right neighbors
        # E[4-loops] ≈ C(N1,2) * C(N2,2) * p^4
        return (N1 * (N1 - 1) / 2) * (N2 * (N2 - 1) / 2) * (p ** 4)
    else:
        # Rough approximation for higher order loops
        return (N1 ** k) * (N2 ** k) * (p ** (2 * k)) / (2 * k)
