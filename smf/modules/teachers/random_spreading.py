"""
Random spreading teacher model for disordered matrix factorization.

Implements Y_ij = (1/√M) Σ_μ F_ij,μ W_iμ X_μj where F ~ N(0,1)
to eliminate finite-size loop effects in the dense limit.

Physical motivation:
- Quenched randomness F breaks loop correlations
- Approaches mean-field behavior faster with finite N
- Cleaner phase transition at α_c
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch

from ..registry import register_teacher
from .base import TeacherBase


@dataclass
class SpreadingData:
    """
    Sparse storage for random spreading coefficients.

    Only stores F for observed edges, not the full N1 x N2 x M tensor.
    Memory: O(C × M) instead of O(N1 × N2 × M)

    Attributes:
        i_idx: (C,) row indices of observed edges
        j_idx: (C,) column indices of observed edges
        F: (C, M) spreading coefficients F_ij,μ ~ N(0,1)
        Y_values: (C,) Y values at observed positions
        seed: Global seed for F generation (ensures reproducibility)
        M: Hidden dimension
    """
    i_idx: torch.Tensor
    j_idx: torch.Tensor
    F: torch.Tensor
    Y_values: torch.Tensor
    seed: int
    M: int

    @property
    def num_edges(self) -> int:
        """Number of observed edges."""
        return len(self.i_idx)

    @property
    def device(self) -> torch.device:
        """Device of tensors."""
        return self.F.device

    def to(self, device: torch.device) -> "SpreadingData":
        """Move all tensors to specified device."""
        return SpreadingData(
            i_idx=self.i_idx.to(device),
            j_idx=self.j_idx.to(device),
            F=self.F.to(device),
            Y_values=self.Y_values.to(device),
            seed=self.seed,
            M=self.M,
        )

    def clone(self) -> "SpreadingData":
        """Create a deep copy."""
        return SpreadingData(
            i_idx=self.i_idx.clone(),
            j_idx=self.j_idx.clone(),
            F=self.F.clone(),
            Y_values=self.Y_values.clone(),
            seed=self.seed,
            M=self.M,
        )


def generate_spreading_coefficients(
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    M: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate deterministic spreading coefficients F for observed edges.

    F[c, μ] ~ N(0, 1) for each observed edge c and hidden dimension μ.

    The generation is deterministic: same (seed, edges) → same F.
    This is critical for:
    1. Reproducibility across runs
    2. Consistent evaluation (student and teacher must use same F)

    Args:
        i_idx: (C,) row indices of observed edges
        j_idx: (C,) column indices of observed edges
        M: Hidden dimension
        seed: Global random seed
        device: Target device (CPU/CUDA)

    Returns:
        F: (C, M) tensor where F[c, μ] ~ N(0, 1)
    """
    C = len(i_idx)

    if C == 0:
        return torch.empty(0, M, device=device, dtype=torch.float32)

    # Use combined seed for reproducibility
    # XOR with prime to avoid seed collision patterns
    combined_seed = seed ^ 0x5DEECE66D

    # Create generator for deterministic generation
    gen = torch.Generator(device=device)
    gen.manual_seed(combined_seed)

    F = torch.randn(C, M, device=device, dtype=torch.float32, generator=gen)

    return F


def generate_spreading_coefficients_per_edge(
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    M: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate F with per-edge deterministic seeding.

    Slower but guarantees exact reproducibility per edge.
    Use this for testing and verification when edge ordering may differ.

    Each edge (i, j) gets a unique seed derived from (global_seed, i, j),
    so the same edge always produces the same F vector regardless of
    the order of edges in the batch.

    Args:
        i_idx: (C,) row indices
        j_idx: (C,) column indices
        M: Hidden dimension
        seed: Global random seed
        device: Target device

    Returns:
        F: (C, M) tensor
    """
    C = len(i_idx)

    if C == 0:
        return torch.empty(0, M, device=device, dtype=torch.float32)

    F = torch.empty(C, M, device=device, dtype=torch.float32)

    # Large primes for hash mixing
    P1, P2, P3 = 1000000007, 1009, 1013

    # Move indices to CPU for iteration
    i_cpu = i_idx.cpu().numpy() if i_idx.is_cuda else i_idx.numpy()
    j_cpu = j_idx.cpu().numpy() if j_idx.is_cuda else j_idx.numpy()

    for c in range(C):
        i, j = int(i_cpu[c]), int(j_cpu[c])
        # Deterministic hash combining seed, i, j
        edge_seed = (seed * P1 + i * P2 + j * P3) % (2**31)
        gen = torch.Generator(device=device)
        gen.manual_seed(edge_seed)
        F[c, :] = torch.randn(M, device=device, dtype=torch.float32, generator=gen)

    return F


def compute_sparse_Y(
    W: torch.Tensor,
    X: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Y values at observed positions with spreading.

    Y[c] = (1/√M) Σ_μ F[c,μ] W[i_idx[c], μ] X[μ, j_idx[c]]

    This is a vectorized implementation avoiding Python loops.

    Args:
        W: (N1, M) teacher W matrix
        X: (M, N2) teacher X matrix
        F: (C, M) spreading coefficients
        i_idx: (C,) row indices of observed edges
        j_idx: (C,) column indices of observed edges

    Returns:
        Y_values: (C,) Y at observed positions
    """
    C = len(i_idx)

    if C == 0:
        return torch.empty(0, device=W.device, dtype=W.dtype)

    M = W.shape[1]
    alpha_scale = 1.0 / (M ** 0.5)

    # Vectorized selection
    W_selected = W[i_idx, :]           # (C, M)
    X_selected = X[:, j_idx].T         # (C, M) - transpose to align

    # Y[c] = (1/√M) × Σ_μ F[c,μ] × W[i,μ] × X[μ,j]
    # Element-wise multiply then sum over M dimension
    Y_values = alpha_scale * (F * W_selected * X_selected).sum(dim=1)  # (C,)

    return Y_values


def compute_sparse_Y_batched(
    W: torch.Tensor,
    X: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Y values for batched W and X.

    Args:
        W: (S, N1, M) batched W matrices
        X: (S, M, N2) batched X matrices
        F: (C, M) spreading coefficients (shared across batch)
        i_idx: (C,) row indices
        j_idx: (C,) column indices

    Returns:
        Y_values: (S, C) Y at observed positions for each sample
    """
    S = W.shape[0]
    C = len(i_idx)

    if C == 0:
        return torch.empty(S, 0, device=W.device, dtype=W.dtype)

    M = W.shape[2]
    alpha_scale = 1.0 / (M ** 0.5)

    # W_selected: (S, C, M)
    W_selected = W[:, i_idx, :]
    # X_selected: (S, C, M)
    X_selected = X[:, :, j_idx].transpose(1, 2)

    # F is (C, M), broadcast to (1, C, M)
    F_expanded = F.unsqueeze(0)

    # Y_values: (S, C)
    Y_values = alpha_scale * (F_expanded * W_selected * X_selected).sum(dim=2)

    return Y_values


@register_teacher(
    key="random_spreading",
    name="Random Spreading Teacher",
    description="Y = (1/√M) Σ F_ij,μ W_iμ X_μj with quenched F ~ N(0,1)",
    default_params={"spreading_seed": 12345},
)
class RandomSpreadingTeacher(TeacherBase):
    """
    Random spreading teacher for disordered matrix factorization.

    Generates W, X with standard Gaussian distribution (same as StandardTeacher),
    but computes Y with random spreading coefficients F.

    Physical motivation:
    - Quenched randomness F breaks loop correlations in factor graphs
    - Reduces finite-size effects in the dense limit
    - Phase transition at α_c becomes sharper and closer to theoretical prediction

    Usage:
        teacher = RandomSpreadingTeacher(spreading_seed=12345)

        # Generate observed edges (from graph module)
        i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)

        # Create teacher with spreading
        W, X, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed
        )

        # Use spreading_data in training and evaluation
    """

    def __init__(self, spreading_seed: int = 12345):
        """
        Initialize Random Spreading Teacher.

        Args:
            spreading_seed: Seed for F generation.
                           Separate from W, X seed for independent control.
                           Same spreading_seed ensures identical F across runs.
        """
        self.spreading_seed = spreading_seed

    def create(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create standard Gaussian teacher matrices W, X.

        Same as StandardTeacher - W, X ~ N(0, 1/√M).

        Args:
            N1: Number of rows in W
            N2: Number of columns in X
            M: Hidden/latent dimension
            device: Target device (CPU/CUDA)
            seed: Random seed for W, X generation

        Returns:
            W: (N1, M) teacher W matrix
            X: (M, N2) teacher X matrix
        """
        torch.manual_seed(seed)

        scale = 1.0 / (M ** 0.5)
        W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
        X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale

        return W, X

    def create_with_Y(
        self,
        N1: int,
        N2: int,
        M: int,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create W, X and full Y matrix.

        WARNING: For random spreading, full Y requires mask/edge information.
        This method falls back to standard Y = W @ X (no spreading).

        For proper spreading computation, use create_with_spreading() instead.

        Returns:
            W: (N1, M) teacher W
            X: (M, N2) teacher X
            Y: (N1, N2) standard matrix product (no spreading)
        """
        W, X = self.create(N1, N2, M, device, seed)
        # Fall back to standard Y (no spreading without edge information)
        Y = W @ X
        return W, X, Y

    def create_with_spreading(
        self,
        N1: int,
        N2: int,
        M: int,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
        device: torch.device,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, SpreadingData]:
        """
        Create W, X and SpreadingData for observed edges.

        This is the main interface for random spreading model.

        Args:
            N1: Number of rows in W
            N2: Number of columns in X
            M: Hidden dimension
            i_idx: (C,) observed edge row indices
            j_idx: (C,) observed edge column indices
            device: Target device
            seed: Random seed for W, X

        Returns:
            W: (N1, M) teacher W matrix
            X: (M, N2) teacher X matrix
            spreading_data: SpreadingData containing F and Y_values
        """
        # Create W, X
        W, X = self.create(N1, N2, M, device, seed)

        # Ensure indices are on correct device
        i_idx = i_idx.to(device)
        j_idx = j_idx.to(device)

        # Generate spreading coefficients
        F = generate_spreading_coefficients(
            i_idx, j_idx, M, self.spreading_seed, device
        )

        # Compute Y at observed positions with spreading
        Y_values = compute_sparse_Y(W, X, F, i_idx, j_idx)

        spreading_data = SpreadingData(
            i_idx=i_idx,
            j_idx=j_idx,
            F=F,
            Y_values=Y_values,
            seed=self.spreading_seed,
            M=M,
        )

        return W, X, spreading_data

    def regenerate_spreading_data(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
    ) -> SpreadingData:
        """
        Regenerate SpreadingData for existing W, X with new edges.

        Useful when edge set changes (e.g., different alpha values).

        Args:
            W: Existing teacher W matrix
            X: Existing teacher X matrix
            i_idx: New edge row indices
            j_idx: New edge column indices

        Returns:
            SpreadingData for the new edge set
        """
        device = W.device
        M = W.shape[1]

        i_idx = i_idx.to(device)
        j_idx = j_idx.to(device)

        F = generate_spreading_coefficients(
            i_idx, j_idx, M, self.spreading_seed, device
        )

        Y_values = compute_sparse_Y(W, X, F, i_idx, j_idx)

        return SpreadingData(
            i_idx=i_idx,
            j_idx=j_idx,
            F=F,
            Y_values=Y_values,
            seed=self.spreading_seed,
            M=M,
        )
