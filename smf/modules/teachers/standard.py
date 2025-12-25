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
        init_distribution: str = "gaussian",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create standard teacher matrices.
        
        Args:
            N1, N2, M: Matrix dimensions
            device: Torch device
            seed: Random seed
            init_distribution: "gaussian" (default) or "rademacher" (±1, Ising-like)
        """
        torch.manual_seed(seed)

        scale = 1.0 / (M ** 0.5)
        
        if init_distribution == "rademacher":
            # Rademacher: ±1 with equal probability, scaled by 1/√M
            W = (2 * torch.randint(0, 2, (N1, M), device=device, dtype=torch.float32) - 1) * scale
            X = (2 * torch.randint(0, 2, (M, N2), device=device, dtype=torch.float32) - 1) * scale
        else:
            # Gaussian: N(0, 1/√M)
            W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
            X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale

        return W, X
