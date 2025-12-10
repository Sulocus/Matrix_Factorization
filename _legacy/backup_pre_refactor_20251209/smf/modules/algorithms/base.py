"""
Base class for training algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Callable
import torch

from ...core.config import Config
from ..graphs.base import GraphBase
from ..teachers.base import TeacherBase


class AlgorithmBase(ABC):
    """Base class for training algorithms."""

    def __init__(self, config: Config, device: torch.device):
        """
        Initialize algorithm.

        Args:
            config: Experiment configuration
            device: Torch device
        """
        self.config = config
        self.device = device

    @abstractmethod
    def train_single_alpha(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train for a single alpha value.

        Args:
            W_teacher: Teacher W matrix (N1, M)
            X_teacher: Teacher X matrix (M, N2)
            Y_teacher: Teacher Y = W @ X (N1, N2)
            mask: Observation mask (N1, N2)
            alpha: Sparsity parameter
            seed: Random seed for student initialization

        Returns:
            W_student: Trained W matrices (S, N1, M)
            X_student: Trained X matrices (S, M, N2)
        """
        pass

    def train_batch_alphas(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: torch.Tensor,
        alpha_values: list[float],
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train for multiple alpha values in parallel (if supported).

        Default implementation runs single_alpha in sequence.
        Subclasses can override for parallel processing.

        Args:
            masks: Shape (num_alphas, N1, N2)
            alpha_values: List of alpha values

        Returns:
            W_student: (num_alphas, S, N1, M)
            X_student: (num_alphas, S, M, N2)
        """
        results_W = []
        results_X = []

        for i, alpha in enumerate(alpha_values):
            W, X = self.train_single_alpha(
                W_teacher, X_teacher, Y_teacher,
                masks[i], alpha, seed + i * 1000
            )
            results_W.append(W)
            results_X.append(X)

        return torch.stack(results_W), torch.stack(results_X)

    def supports_batch_training(self) -> bool:
        """Check if algorithm supports efficient batch training."""
        return False

    def estimate_memory_per_alpha(self, N1: int, N2: int, M: int, S: int) -> float:
        """
        Estimate GPU memory needed per alpha value (in GB).

        Subclasses should override with accurate estimates.
        """
        # Default conservative estimate
        student_params = 2 * (S * N1 * M + S * M * N2)
        intermediate = 10 * S * N1 * N2
        total_elements = student_params + intermediate
        return total_elements * 4 / (1024**3)  # 4 bytes per float32
