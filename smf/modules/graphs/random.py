"""
Random graph generation (GPU-based).
"""

from typing import Tuple
import torch

from ..registry import register_graph
from .base import GraphBase


@register_graph(
    key="random",
    name="Random Graph (GPU)",
    description="Pure random sampling, fast, supports any N1≠N2",
)
class RandomGraph(GraphBase):
    """
    Pure random mask generation (entirely on GPU).

    Approach:
    1. Map all positions of N1×N2 matrix to 1D index [0, N1*N2-1]
    2. Use torch.randperm to randomly shuffle all positions on GPU
    3. Take first C positions as observation points
    4. Restore 1D index to (i,j) coordinates
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
        """Generate random edge indices."""
        # Calculate degree and total edges
        deg_left = int(round(alpha * M))
        deg_left = max(0, min(deg_left, N2))
        C = N1 * deg_left

        if seed is not None:
            torch.manual_seed(seed)

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

        # Randomly shuffle all position indices on GPU
        idx = torch.randperm(total, device=device)[:C]

        # Restore 1D index to 2D coordinates
        i_idx = idx // N2  # Row index
        j_idx = idx % N2   # Column index

        return i_idx, j_idx, C
