"""
Base class for graph/mask generators.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch


class GraphBase(ABC):
    """Base class for graph/mask generation."""

    @abstractmethod
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
        Generate edge indices for the observation mask.

        Args:
            N1: Number of rows
            N2: Number of columns
            M: Latent dimension (used to compute degree)
            alpha: Sparsity parameter (degree = alpha * M)
            device: Torch device
            seed: Random seed (optional)

        Returns:
            i_idx: Row indices of observed entries (1D tensor)
            j_idx: Column indices of observed entries (1D tensor)
            C: Number of edges (len(i_idx))
        """
        pass

    def generate_mask(
        self,
        N1: int,
        N2: int,
        M: int,
        alpha: float,
        device: torch.device,
        seed: int = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate binary observation mask.

        Args:
            Same as generate()

        Returns:
            mask: Binary mask tensor (N1, N2)
            c: Expected degree per left node (alpha * M)
        """
        c = alpha * M
        i_idx, j_idx, C = self.generate(N1, N2, M, alpha, device, seed)

        mask = torch.zeros((N1, N2), device=device, dtype=torch.float32)
        if C > 0:
            mask[i_idx, j_idx] = 1.0

        return mask, c
