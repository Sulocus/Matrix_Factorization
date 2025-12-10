"""
Standard Gaussian teacher initialization.
"""

from typing import Tuple
import torch

from ..registry import register_teacher
from .base import TeacherBase


@register_teacher(
    key="standard",
    name="Standard Gaussian",
    description="W,X ~ N(0, 1/√M), standard initialization",
)
class StandardTeacher(TeacherBase):
    """
    Standard Gaussian teacher model initialization.

    W and X are initialized with N(0, 1/sqrt(M)) distribution,
    ensuring Y = W @ X has reasonable scale.
    """

    def create(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create standard Gaussian teacher matrices."""
        torch.manual_seed(seed)

        scale = 1.0 / (M ** 0.5)
        W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
        X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale

        return W, X
