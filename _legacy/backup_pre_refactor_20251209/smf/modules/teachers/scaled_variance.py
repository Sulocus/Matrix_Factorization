"""
Scaled variance teacher initialization.

Allows adjusting the variance scaling factor for W and X.
"""

from typing import Tuple
import torch

from ..registry import register_teacher
from .base import TeacherBase


@register_teacher(
    key="scaled_variance",
    name="Scaled Variance Gaussian",
    description="W,X ~ N(0, k/√M), supports variance scaling",
    default_params={'variance_scale': 1.0},
)
class ScaledVarianceTeacher(TeacherBase):
    """
    Teacher model with adjustable variance scaling.

    Standard initialization uses scale = 1/√M.
    This class allows using scale = k/√M where k is configurable.

    For example:
    - variance_scale = 1.0: standard 1/√M
    - variance_scale = 2.0: doubled variance 2/√M
    - variance_scale = 0.5: halved variance 0.5/√M
    """

    def __init__(self, variance_scale: float = 1.0):
        """
        Initialize with variance scale factor.

        Args:
            variance_scale: Multiplier for the standard 1/√M scaling.
                           1.0 = standard, 2.0 = double variance, etc.
        """
        self.variance_scale = variance_scale

    def create(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create teacher matrices with scaled variance."""
        torch.manual_seed(seed)

        # scale = k / √M where k is variance_scale
        scale = self.variance_scale / (M ** 0.5)

        W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
        X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale

        return W, X

    def get_description(self) -> str:
        """Get human-readable description."""
        if self.variance_scale == 1.0:
            return "Standard Gaussian N(0, 1/√M)"
        else:
            return f"Scaled Gaussian N(0, {self.variance_scale}/√M)"
