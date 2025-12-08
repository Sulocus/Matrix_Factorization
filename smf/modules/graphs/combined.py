"""
Combined graph generator for flexible configuration.

Supports mixing multiple graph features:
- random: Pure random sampling
- dinic: Bi-regular using max-flow
- low_loop: MCMC-optimized low short-cycle graphs
- uniform: Near-uniform degree distribution

This module enables LLM-driven configuration to select graph types
based on natural language descriptions.
"""

import re
from typing import Tuple, Dict, Any
import torch

from ..registry import register_graph
from .base import GraphBase
from .random import RandomGraph
from .dinic import DinicGraph
from .low_loop import LowLoopGraph


@register_graph(
    key="combined",
    name="Combined Graph",
    description="Flexible graph supporting multiple generation strategies",
    default_params={
        "method": "random",
        "low_loop": False,
        "loop_order": 2,
        "n_sweeps": 5,
    },
)
class CombinedGraph(GraphBase):
    """
    Flexible graph generator that combines multiple features.

    Supports:
    - Base method selection (random, dinic)
    - Optional low-loop optimization via MCMC
    - Configurable loop order (4-loop, 6-loop, 8-loop)

    Example configurations:
    - {"method": "random"} -> Pure random graph
    - {"method": "dinic"} -> Strict bi-regular graph
    - {"low_loop": True, "loop_order": 2} -> Random + 4-loop minimization
    - {"method": "dinic", "low_loop": True} -> Bi-regular + loop minimization

    This is the recommended graph generator for LLM-driven configuration.
    """

    def __init__(
        self,
        method: str = "random",
        low_loop: bool = False,
        loop_order: int = 2,
        n_sweeps: int = 5,
        alpha_threshold: float = 0.8,
    ):
        """
        Initialize combined graph generator.

        Args:
            method: Base generation method ("random" or "dinic")
            low_loop: Whether to apply MCMC loop minimization
            loop_order: k value for 2k-loops (2=4-loops, 3=6-loops)
            n_sweeps: Number of MCMC sweeps for loop minimization
            alpha_threshold: Only run MCMC when alpha < threshold
        """
        self.method = method.lower()
        self.low_loop = low_loop
        self.loop_order = loop_order
        self.n_sweeps = n_sweeps
        self.alpha_threshold = alpha_threshold

        # Create base generator
        if self.method == "dinic":
            self._base = DinicGraph()
        else:
            self._base = RandomGraph()

        # Create low-loop optimizer if needed
        if self.low_loop:
            self._low_loop = LowLoopGraph(
                loop_order=loop_order,
                n_sweeps=n_sweeps,
                alpha_threshold=alpha_threshold,
            )

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
        Generate graph edge indices with combined features.

        Args:
            N1, N2: Matrix dimensions
            M: Latent dimension
            alpha: Sparsity parameter
            device: Torch device
            seed: Random seed

        Returns:
            i_idx, j_idx, C: Edge indices and count
        """
        # If low_loop is enabled, use LowLoopGraph directly
        # (it already starts with random and applies MCMC)
        if self.low_loop:
            return self._low_loop.generate(N1, N2, M, alpha, device, seed)

        # Otherwise use the base method
        return self._base.generate(N1, N2, M, alpha, device, seed)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombinedGraph":
        """
        Create graph from configuration dictionary.

        Args:
            config: Dictionary with keys:
                - method: str ("random", "dinic")
                - low_loop: bool
                - loop_order: int
                - n_sweeps: int
                - alpha_threshold: float

        Returns:
            CombinedGraph instance
        """
        return cls(
            method=config.get("method", "random"),
            low_loop=config.get("low_loop", False),
            loop_order=config.get("loop_order", 2),
            n_sweeps=config.get("n_sweeps", 5),
            alpha_threshold=config.get("alpha_threshold", 0.8),
        )

    @classmethod
    def from_natural_language(cls, description: str) -> "CombinedGraph":
        """
        Parse natural language description to create graph generator.

        Supports:
        - "random graph"
        - "bi-regular graph", "dinic graph"
        - "low loop graph", "no 4-cycles"
        - "6-loop free graph"
        - Chinese: "随机图", "双正则图", "低循环图"

        Args:
            description: Natural language description

        Returns:
            CombinedGraph instance
        """
        desc_lower = description.lower()
        method = "random"
        low_loop = False
        loop_order = 2

        # Check for bi-regular/dinic
        if any(kw in desc_lower for kw in ["bi-regular", "biregular", "dinic", "双正则"]):
            method = "dinic"

        # Check for low-loop
        if any(kw in desc_lower for kw in [
            "low loop", "low-loop", "no 4-cycle", "c4-free", "c4 free",
            "低循环", "无4环"
        ]):
            low_loop = True

        # Check for specific loop order
        loop_match = re.search(r"(\d+)-?loop", desc_lower)
        if loop_match:
            detected_loop = int(loop_match.group(1))
            if detected_loop in [4, 6, 8]:
                loop_order = detected_loop // 2
                low_loop = True

        return cls(method=method, low_loop=low_loop, loop_order=loop_order)

    def __repr__(self) -> str:
        features = [f"method={self.method}"]
        if self.low_loop:
            features.append(f"low_loop(k={self.loop_order})")
        return f"CombinedGraph({', '.join(features)})"
