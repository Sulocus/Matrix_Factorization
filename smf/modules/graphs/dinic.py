"""
Bi-regular graph generation using Dinic's max-flow algorithm.

Generates graphs with strict degree constraints:
- Each left node has exactly the target degree
- Right node degrees are as uniform as possible
"""

from typing import Tuple
from collections import deque
import itertools
import numpy as np
import torch

from ..registry import register_graph
from .base import GraphBase


class DinicMaxFlow:
    """
    Dinic's algorithm for maximum flow.

    Used to construct bi-regular bipartite graphs by modeling
    the edge assignment as a max-flow problem.
    """
    __slots__ = ("n", "g", "lvl", "it")

    def __init__(self, n: int):
        """Initialize flow network with n nodes."""
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int):
        """Add edge from u to v with capacity cap."""
        self.g[u].append([v, cap, len(self.g[v])])
        self.g[v].append([u, 0, len(self.g[u]) - 1])

    def bfs(self, s: int, t: int) -> bool:
        """Build level graph using BFS. Returns True if t is reachable."""
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

    def dfs(self, u: int, t: int, f: int) -> int:
        """Find augmenting path using DFS."""
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

    def max_flow(self, s: int, t: int) -> int:
        """Compute maximum flow from s to t."""
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


@register_graph(
    key="dinic",
    name="Bi-regular Graph (Dinic)",
    description="Strict bi-regular construction using max-flow algorithm",
)
class DinicGraph(GraphBase):
    """
    Bi-regular graph generator using Dinic's max-flow algorithm.

    Properties:
    - Each left node has exactly deg_left = round(alpha * M) edges
    - Right node degrees are as uniform as possible
    - Total edges = N1 * deg_left

    This is useful for studying theoretical properties where
    degree regularity matters for analysis.

    Note: Slower than random graph generation but provides
    exact degree constraints.
    """

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
        Generate bi-regular graph edge indices.

        Args:
            N1: Number of left nodes (rows)
            N2: Number of right nodes (columns)
            M: Latent dimension
            alpha: Sparsity parameter (left degree = alpha * M)
            device: Torch device
            seed: Random seed

        Returns:
            i_idx, j_idx, C: Edge indices and count

        Raises:
            RuntimeError: If degree constraints are infeasible
        """
        # Calculate target degrees
        deg_left = int(round(alpha * M))
        deg_left = max(0, min(deg_left, N2))
        total_edges = N1 * deg_left

        if deg_left == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                0,
            )

        # Check feasibility
        if deg_left > N2:
            raise RuntimeError(f"deg_left={deg_left} > N2={N2}, infeasible")

        deg_right_exact = total_edges // N2
        if total_edges % N2 == 0 and deg_right_exact > N1:
            raise RuntimeError(f"deg_right={deg_right_exact} > N1={N1}, infeasible")

        # Initialize RNG
        if seed is not None:
            rng = np.random.RandomState(seed + 12345 + int(round(alpha * 1e6)))
        else:
            rng = np.random.RandomState()

        # Calculate target right degrees (as uniform as possible)
        base = total_edges // N2
        rem = total_edges % N2
        right_target = np.full(N2, base, dtype=int)
        if rem > 0:
            idx = np.arange(N2)
            rng.shuffle(idx)
            right_target[idx[:rem]] += 1

        if right_target.max() > N1:
            raise RuntimeError(
                f"Some right node target degree {right_target.max()} > N1={N1}, infeasible"
            )

        # Build flow network
        # Nodes: 0=source, 1..N1=left nodes, N1+1..N1+N2=right nodes, N1+N2+1=sink
        S = 0
        L_off = 1
        R_off = 1 + N1
        T = R_off + N2

        din = DinicMaxFlow(T + 1)

        # Source -> left nodes (capacity = deg_left)
        for i in range(N1):
            din.add_edge(S, L_off + i, deg_left)

        # Left nodes -> right nodes (capacity = 1, random order)
        all_pairs = list(itertools.product(range(N1), range(N2)))
        rng.shuffle(all_pairs)

        for i, j in all_pairs:
            ui = L_off + i
            vj = R_off + j
            din.add_edge(ui, vj, 1)

        # Right nodes -> sink (capacity = target degree)
        for j in range(N2):
            din.add_edge(R_off + j, T, int(right_target[j]))

        # Compute max flow
        f = din.max_flow(S, T)
        if f != total_edges:
            raise RuntimeError(
                f"maxflow only got {f}/{total_edges}, degree sequence infeasible"
            )

        # Extract edges from flow result
        i_list, j_list = [], []
        for i in range(N1):
            u = L_off + i
            for v, cap, rev in din.g[u]:
                if R_off <= v < R_off + N2:
                    # Check if edge was used (reverse edge has capacity)
                    if din.g[v][rev][1] > 0:
                        j = v - R_off
                        i_list.append(i)
                        j_list.append(j)

        i_idx = torch.tensor(i_list, dtype=torch.long, device=device)
        j_idx = torch.tensor(j_list, dtype=torch.long, device=device)
        C = len(i_list)

        return i_idx, j_idx, C
