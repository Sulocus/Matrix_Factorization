"""
Unified Graph Generator module.

Supports all graph topology types:
1. random: Pure random sampling (GPU-accelerated)
2. uniform: Bi-regular graphs via Dinic's max-flow algorithm
3. low_loop: Minimized short cycles via MCMC edge-switching
"""

from collections import deque
from typing import Tuple, Literal
import itertools
import numpy as np
import torch

from ..registry import register_graph
from .base import GraphBase

GraphType = Literal["random", "uniform", "low_loop"]


# ============================================================================
# Dinic's Max-Flow Algorithm (for Uniform graphs)
# ============================================================================

class Dinic:
    """Dinic's algorithm for maximum flow (used by Uniform mode)."""
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
        if u == t:
            return f
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
                if f == 0:
                    break
                flow += f
        return flow


# ============================================================================
# Low-Loop MCMC Utilities
# ============================================================================

def count_2k_loops_gpu(A: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Count 2k-loops in bipartite graph using matrix powers.
    
    Args:
        A: Adjacency matrix (N1, N2)
        k: Loop order (k=2 for 4-loops, k=3 for 6-loops)
    
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
    
    Returns:
        final_edges, accept_rate, n_initial, n_final
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


# ============================================================================
# Main Graph Generator
# ============================================================================

@register_graph(
    key="generator",
    name="Unified Graph Generator",
    description="Unified generator for Random, Uniform (Bi-Regular), and Low-Loop graphs",
    default_params={"type": "random"},
)
class GraphGenerator(GraphBase):
    """
    Unified Graph Generator.
    
    Supports:
    1. random: GPU-accelerated pure random sampling
    2. uniform: Dinic's algorithm for bi-regular graphs
    3. low_loop: MCMC edge-switching to minimize short cycles
    """

    def __init__(
        self, 
        type: GraphType = "random",
        loop_order: int = 2,        # k for 2k-loops
        n_sweeps: int = 5,          # MCMC sweeps
        alpha_threshold: float = 0.8,  # Only run MCMC when alpha < threshold
    ):
        self.type = type
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
        """Generate graph edges based on configured type."""
        if self.type == "random":
            return self._generate_random(N1, N2, M, alpha, device, seed)
        elif self.type == "uniform":
            return self._generate_uniform(N1, N2, M, alpha, device, seed)
        elif self.type == "low_loop":
            return self._generate_low_loop(N1, N2, M, alpha, device, seed)
        else:
            raise ValueError(f"Unknown graph type: {self.type}")

    def _generate_random(
        self,
        N1: int,
        N2: int,
        M: int,
        alpha: float,
        device: torch.device,
        seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Generate random graph (GPU)."""
        deg_left = int(round(alpha * M))
        deg_left = max(0, min(deg_left, N2))
        C = N1 * deg_left

        if seed is not None:
            torch.manual_seed(seed)

        total = N1 * N2
        if C > total:
            raise RuntimeError(f"C={C} > N1*N2={total}")

        if C == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                0,
            )

        idx = torch.randperm(total, device=device)[:C]
        i_idx = idx // N2
        j_idx = idx % N2
        return i_idx, j_idx, C

    def _generate_uniform(
        self,
        N1: int,
        N2: int,
        M: int,
        alpha: float,
        device: torch.device,
        seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Generate bi-regular graph (CPU Dinic)."""
        deg_left = int(round(alpha * M))
        deg_left = max(0, min(deg_left, N2))
        total_edges = N1 * deg_left

        if deg_left == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                0,
            )

        deg_right_exact = total_edges // N2
        if total_edges % N2 == 0 and deg_right_exact > N1:
            raise RuntimeError(f"Infeasible bi-regular degree")

        if seed is not None:
            rng = np.random.RandomState(seed + 12345 + int(round(alpha * 1e6)))
        else:
            rng = np.random.RandomState()

        base = total_edges // N2
        rem = total_edges % N2
        right_target = np.full(N2, base, dtype=int)
        if rem > 0:
            idx = np.arange(N2)
            rng.shuffle(idx)
            right_target[idx[:rem]] += 1

        S, L_off, R_off = 0, 1, 1 + N1
        T = R_off + N2
        din = Dinic(T + 1)

        for i in range(N1):
            din.add_edge(S, L_off + i, deg_left)

        all_pairs = list(itertools.product(range(N1), range(N2)))
        rng.shuffle(all_pairs)

        for i, j in all_pairs:
            din.add_edge(L_off + i, R_off + j, 1)

        for j in range(N2):
            din.add_edge(R_off + j, T, int(right_target[j]))

        f = din.max_flow(S, T)
        if f != total_edges:
            raise RuntimeError(f"Maxflow failed: {f}/{total_edges}")

        i_list, j_list = [], []
        for i in range(N1):
            u = L_off + i
            for v, cap, rev in din.g[u]:
                if R_off <= v < R_off + N2:
                    if din.g[v][rev][1] > 0:
                        i_list.append(i)
                        j_list.append(v - R_off)

        i_idx = torch.tensor(i_list, dtype=torch.long, device=device)
        j_idx = torch.tensor(j_list, dtype=torch.long, device=device)
        return i_idx, j_idx, len(i_list)

    def _generate_low_loop(
        self,
        N1: int,
        N2: int,
        M: int,
        alpha: float,
        device: torch.device,
        seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generate low-loop graph via MCMC edge-switching.
        
        1. Start with a random graph
        2. If alpha < threshold, run MCMC to minimize short cycles
        """
        deg_left = int(round(alpha * M))
        deg_left = max(0, min(deg_left, N2))
        C = N1 * deg_left

        if seed is not None:
            np.random.seed(seed)

        total = N1 * N2
        if C > total:
            raise RuntimeError(f"C={C} > N1*N2={total}")

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


# ============================================================================
# Utility Functions
# ============================================================================

def count_4loops(A: torch.Tensor) -> int:
    """Count 4-loops in adjacency matrix."""
    return int(count_2k_loops_gpu(A, k=2).item())


def count_6loops(A: torch.Tensor) -> int:
    """Count 6-loops in adjacency matrix."""
    return int(count_2k_loops_gpu(A, k=3).item())
