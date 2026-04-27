"""
Orthogonal teacher initialization using QR decomposition.

Eliminates finite-size fluctuations by enforcing orthonormality,
simulating the thermodynamic limit (N -> infinity) behavior.
"""

from typing import Tuple
import torch

from ..registry import register_teacher
from .base import TeacherBase
from ...core.experiment.config import resolve_normalization_profile


@register_teacher(
    key="orthogonal",
    name="Orthogonal Teacher",
    description="W,X with orthonormal columns/rows via QR decomposition",
)
class OrthogonalTeacher(TeacherBase):
    """
    Orthogonal teacher model using QR decomposition.

    Mathematical properties:
    - W^T W = I_M (orthonormal columns)
    - X X^T = I_M (orthonormal rows)
    - Scaling: ||W||_F^2 = N1, ||X||_F^2 = N2 (same as standard)

    Benefits:
    - Removes the 2*alpha*M/N linear bias in low-alpha region
    - Simulates thermodynamic limit behavior
    - More stable overlap metrics
    """

    def create(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
        normalization_profile: str = "paper_sparse_sampling",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create orthogonal teacher matrices.

        Uses QR decomposition to create matrices with orthonormal
        columns (W) and rows (X), then scales to match expected
        Frobenius norm of standard teacher.

        Args:
            N1: Number of rows in W
            N2: Number of columns in X
            M: Latent dimension (must be <= min(N1, N2))
            device: Torch device
            seed: Random seed

        Returns:
            W_true: Orthogonal teacher W matrix (N1, M)
            X_true: Orthogonal teacher X matrix (M, N2)
        """
        torch.manual_seed(seed)

        # Generate random matrices
        W_raw = torch.randn(N1, M, device=device, dtype=torch.float32)
        X_raw = torch.randn(M, N2, device=device, dtype=torch.float32)

        # QR decomposition for W (thin QR: N1 x M -> Q: N1 x M, R: M x M)
        # After QR: W_ortho^T @ W_ortho = I_M
        W_ortho, _ = torch.linalg.qr(W_raw, mode='reduced')

        # QR decomposition for X^T, then transpose back
        # After QR: X_ortho @ X_ortho^T = I_M
        X_ortho_T, _ = torch.linalg.qr(X_raw.T, mode='reduced')
        X_ortho = X_ortho_T.T

        # Scale to the selected entry variance.  paper_sparse_sampling gives
        # Var(entry)=1; internal_normalized_legacy gives Var(entry)=1/M.
        latent_std = resolve_normalization_profile(normalization_profile, M).latent_std
        W_true = W_ortho * (N1 ** 0.5) * latent_std
        X_true = X_ortho * (N2 ** 0.5) * latent_std

        return W_true, X_true
