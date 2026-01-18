"""
Orthogonal teacher initialization with UNIT variance (O(1) scaling).

This is the corrected version following Paper standards:
- Variables W, X have variance = 1 (O(1) magnitude)
- Signal Y is O(1), enabling proper phase transition

Reference: prompt.md - "方案 A：遵循论文标准"
"""

from typing import Tuple
import torch

from ..registry import register_teacher
from .base import TeacherBase


@register_teacher(
    key="orthogonal_unit",
    name="Orthogonal Teacher (Unit Variance)",
    description="W,X with orthonormal columns/rows, unit variance (O(1) scaling)",
)
class OrthogonalUnitTeacher(TeacherBase):
    """
    Orthogonal teacher model with UNIT variance.

    This corrects the scaling issue identified in prompt.md:
    - Original: Var(w_ij) = 1/M → signal too weak
    - Corrected: Var(w_ij) = 1 → signal O(1), matches paper

    Mathematical properties:
    - W^T W = N1 * I_M (scaled orthonormal columns)
    - X X^T = N2 * I_M (scaled orthonormal rows)
    - Scaling: ||W||_F^2 = N1*M, ||X||_F^2 = N2*M
    - Element variance: Var(w_ij) ≈ 1

    Key differences from orthogonal.py:
    - Scale factor: sqrt(N1) instead of sqrt(N1/M)
    - Algorithm must use 1/(1+tau) instead of 1/(M+tau)
    """

    def create(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create orthogonal teacher matrices with unit variance.

        Uses QR decomposition to create matrices with orthonormal
        columns (W) and rows (X), then scales to achieve unit variance.

        Args:
            N1: Number of rows in W
            N2: Number of columns in X
            M: Latent dimension (must be <= min(N1, N2))
            device: Torch device
            seed: Random seed

        Returns:
            W_true: Orthogonal teacher W matrix (N1, M) with Var(w_ij) ≈ 1
            X_true: Orthogonal teacher X matrix (M, N2) with Var(x_ij) ≈ 1
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

        # Scale to achieve UNIT variance (O(1) scaling)
        # Orthogonal: ||W_ortho||_F^2 = M (since orthonormal columns)
        # Target: ||W||_F^2 = N1 * M for unit variance
        # Scale factor: sqrt(N1) to get ||W||_F^2 = N1 * M
        #
        # Verification: Var(w_ij) = ||W||_F^2 / (N1*M) = N1*M / (N1*M) = 1 ✓
        W_true = W_ortho * (N1) ** 0.5
        X_true = X_ortho * (N2) ** 0.5

        return W_true, X_true
