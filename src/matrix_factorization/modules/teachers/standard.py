"""
Standard Gaussian teacher initialization.
"""

from typing import Tuple
import torch

from ..registry import register_teacher
from .base import TeacherBase
from ...core.experiment.config import resolve_normalization_profile


@register_teacher(
    key="standard",
    name="Standard Gaussian",
    description="W,X follow the selected normalization_profile",
)
class StandardTeacher(TeacherBase):
    """
    Standard Gaussian teacher model initialization.

    W and X follow algorithm_params.normalization_profile when passed.
    The default is paper_sparse_sampling, so entries have variance 1.
    """

    def create(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
        init_distribution: str = "gaussian",
        normalization_profile: str = "paper_sparse_sampling",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create standard teacher matrices.
        
        Args:
            N1, N2, M: Matrix dimensions
            device: Torch device
            seed: Random seed
            init_distribution: "gaussian" (default) or "rademacher" (±1, Ising-like)
        """
        torch.manual_seed(seed)

        scale = resolve_normalization_profile(normalization_profile, M).latent_std
        
        if init_distribution == "rademacher":
            # Rademacher: ±1 with equal probability in the selected latent scale.
            W = (2 * torch.randint(0, 2, (N1, M), device=device, dtype=torch.float32) - 1) * scale
            X = (2 * torch.randint(0, 2, (M, N2), device=device, dtype=torch.float32) - 1) * scale
        else:
            # Gaussian in the selected latent scale.
            W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
            X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale

        return W, X
