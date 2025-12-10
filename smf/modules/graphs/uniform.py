"""
Uniform (bi-regular) graph generation using Dinic algorithm.
"""

from collections import deque
from typing import Tuple
import itertools
import numpy as np
import torch

from ..registry import register_graph
from .base import GraphBase


class Dinic:
    """Dinic's algorithm for maximum flow."""
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


@register_graph(
    key="uniform",
    name="Uniform Graph (Dinic)",
    description="Strict bi-regular graph, uniform degree distribution, suitable for theoretical analysis",
)
class UniformGraph(GraphBase):
    """
    Bi-regular graph generation using Dinic's max-flow algorithm.

    Creates a strict bi-regular (or near-regular) graph where:
    - All left nodes have the same degree
    - Right nodes have degrees as uniform as possible
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
        """Generate bi-regular graph edge indices."""
        deg_left = int(round(alpha * M))
        deg_left = max(0, min(deg_left, N2))
        total_edges = N1 * deg_left

        if deg_left == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                0,
            )

        if deg_left > N2:
            raise RuntimeError(f"deg_left={deg_left} > N2={N2}, infeasible")

        deg_right_exact = total_edges // N2
        if total_edges % N2 == 0 and deg_right_exact > N1:
            raise RuntimeError(f"deg_right={deg_right_exact} > N1={N1}, infeasible")

        # Setup random number generator
        if seed is not None:
            rng = np.random.RandomState(seed + 12345 + int(round(alpha * 1e6)))
        else:
            rng = np.random.RandomState()

        # Calculate target right degrees
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
        S, L_off, R_off = 0, 1, 1 + N1
        T = R_off + N2
        din = Dinic(T + 1)

        # Source to left nodes
        for i in range(N1):
            din.add_edge(S, L_off + i, deg_left)

        # Randomize L->R edge addition order for randomness
        all_pairs = list(itertools.product(range(N1), range(N2)))
        rng.shuffle(all_pairs)

        for i, j in all_pairs:
            ui = L_off + i
            vj = R_off + j
            din.add_edge(ui, vj, 1)

        # Right nodes to sink
        for j in range(N2):
            din.add_edge(R_off + j, T, int(right_target[j]))

        # Run max flow
        f = din.max_flow(S, T)
        if f != total_edges:
            raise RuntimeError(
                f"maxflow only got {f}/{total_edges}, degree sequence infeasible"
            )

        # Extract edges from residual graph
        i_list, j_list = [], []
        for i in range(N1):
            u = L_off + i
            for v, cap, rev in din.g[u]:
                if R_off <= v < R_off + N2:
                    if din.g[v][rev][1] > 0:
                        j = v - R_off
                        i_list.append(i)
                        j_list.append(j)

        assert len(i_list) == total_edges, "Extracted edge count mismatch"

        # Verify degrees (debug)
        i_np = np.array(i_list, dtype=int)
        j_np = np.array(j_list, dtype=int)
        left_deg = np.bincount(i_np, minlength=N1)
        right_deg = np.bincount(j_np, minlength=N2)
        assert np.all(left_deg == deg_left), "Left degree inconsistent"
        assert np.all(right_deg == right_target), "Right degree doesn't match target"
        assert len(set(zip(i_np, j_np))) == len(i_np), "Duplicate edges exist"

        i_idx = torch.tensor(i_list, dtype=torch.long, device=device)
        j_idx = torch.tensor(j_list, dtype=torch.long, device=device)
        C = len(i_list)

        return i_idx, j_idx, C
