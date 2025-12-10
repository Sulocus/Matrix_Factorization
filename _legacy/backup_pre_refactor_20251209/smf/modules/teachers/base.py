"""
Base class for teacher model initialization.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch


class TeacherBase(ABC):
    """Base class for teacher model initialization."""

    @abstractmethod
    def create(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create teacher model W_true and X_true.

        Args:
            N1: Number of rows in W
            N2: Number of columns in X
            M: Latent dimension
            device: Torch device
            seed: Random seed

        Returns:
            W_true: Teacher W matrix (N1, M)
            X_true: Teacher X matrix (M, N2)
        """
        pass

    def create_with_Y(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create teacher model and compute Y = W @ X.

        Returns:
            W_true, X_true, Y_true
        """
        W, X = self.create(N1, N2, M, device, seed)
        Y = W @ X
        return W, X, Y
