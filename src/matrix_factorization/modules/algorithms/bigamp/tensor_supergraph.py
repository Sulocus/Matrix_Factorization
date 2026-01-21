"""
TensorSuperGraph: Disjoint Union data structure for N-dimensional tensor parallelization.

This module provides the data structures and utilities for Alpha + Sample
parallel processing in tensor CP decomposition, mirroring the SuperGraph
architecture from the Bipartite spreading algorithm.

Key concepts:
- TensorSuperGraph: Index structure shared across all alphas and samples
- TensorSuperData: Complete training data with F, Y, and teacher factors
- Disjoint Union format: (A, S*N_d, M) enables parallel processing
"""

import math
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TensorSuperGraph:
    """
    N-dimensional tensor Disjoint Union index structure.
    
    This structure enables Alpha + Sample parallel processing by:
    1. Sharing hyperedge indices across all alphas (up to C_max edges)
    2. Using alpha_mask to handle variable edge counts per alpha
    3. Storing indices in flat (S*C_max) format for efficient gathering
    
    Attributes:
        dims: Tensor dimensions (N_1, N_2, ..., N_n)
        n: Tensor order (len(dims))
        A: Number of alpha values (parallel)
        S: Number of samples (parallel)
        M: Latent dimension
        C_per_alpha: Number of hyperedges per alpha value
        C_max: Maximum edges for any alpha (for padding)
        indices: n tensors of shape (S, C_max) - hyperedge indices per dimension
        alpha_mask: (A, C_max) bool - which edges are valid for each alpha
    """
    dims: Tuple[int, ...]
    n: int
    A: int
    S: int
    M: int
    C_per_alpha: List[int]
    C_max: int
    indices: List[torch.Tensor]  # n × (S, C_max)
    alpha_mask: torch.Tensor     # (A, C_max)
    device: torch.device
    
    @property
    def SC(self) -> int:
        """Total edges per alpha batch: S * C_max"""
        return self.S * self.C_max
    
    def get_flat_indices(self) -> List[torch.Tensor]:
        """
        Get indices in flat (S*C_max,) format for gather operations.
        
        Returns:
            n tensors of shape (S*C_max,)
        """
        return [idx.reshape(-1) for idx in self.indices]


@dataclass
class TensorSuperData:
    """
    Complete training data for TensorSuperGraph parallel training.
    
    Contains:
    - SuperGraph index structure
    - F_super: Spreading coefficients for all samples
    - Y_super: Observations for all samples
    - Teacher factors for metric computation
    
    Tensor formats:
    - F_super: (S, C_max, M) - shared across alphas (same random realization)
    - Y_super: (S, C_max) - observations per sample
    - alpha_mask_exp: (A, S*C_max) - expanded for flat operations
    """
    supergraph: TensorSuperGraph
    F_super: torch.Tensor       # (S, C_max, M)
    Y_super: torch.Tensor       # (S, C_max)
    teacher_factors: List[torch.Tensor]  # n × (N_d, M)
    alpha_mask_exp: torch.Tensor  # (A, S*C_max) - expanded alpha mask
    
    @property
    def S(self) -> int:
        return self.supergraph.S
    
    @property
    def A(self) -> int:
        return self.supergraph.A
    
    @property
    def M(self) -> int:
        return self.supergraph.M
    
    @property
    def C_max(self) -> int:
        return self.supergraph.C_max
    
    def get_flat_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get F and Y in flat (S*C_max, M) and (S*C_max,) format.
        
        Returns:
            F_flat: (S*C_max, M)
            Y_flat: (S*C_max,)
        """
        SC = self.supergraph.SC
        F_flat = self.F_super.reshape(SC, self.M)
        Y_flat = self.Y_super.reshape(SC)
        return F_flat, Y_flat


def create_tensor_supergraph(
    dims: Tuple[int, ...],
    alpha_values: List[float],
    M: int,
    S: int,
    seed: int,
    device: torch.device,
) -> TensorSuperGraph:
    """
    Create TensorSuperGraph with shared index structure.
    
    The indices are generated once for C_max edges, and alpha_mask
    indicates which edges are valid for each alpha value.
    
    Args:
        dims: Tensor dimensions (N_1, ..., N_n)
        alpha_values: List of alpha values to parallelize
        M: Latent dimension
        S: Number of samples
        seed: Random seed
        device: Target device
        
    Returns:
        TensorSuperGraph with shared indices and alpha mask
    """
    n = len(dims)
    A = len(alpha_values)
    
    # Compute degrees of freedom
    dof = sum(dims) * M
    
    # Compute C for each alpha
    C_per_alpha = [max(1, int(alpha * dof)) for alpha in alpha_values]
    C_max = max(C_per_alpha)
    
    # Generate indices for C_max edges (shared across alphas)
    torch.manual_seed(seed)
    indices = [
        torch.randint(0, dims[d], (S, C_max), device=device)
        for d in range(n)
    ]
    
    # Create alpha mask: (A, C_max)
    # True if edge index < C for this alpha
    edge_indices = torch.arange(C_max, device=device).unsqueeze(0)  # (1, C_max)
    C_thresholds = torch.tensor(C_per_alpha, device=device).unsqueeze(1)  # (A, 1)
    alpha_mask = edge_indices < C_thresholds  # (A, C_max)
    
    logger.debug(f"Created TensorSuperGraph: dims={dims}, A={A}, S={S}, C_max={C_max}")
    
    return TensorSuperGraph(
        dims=dims,
        n=n,
        A=A,
        S=S,
        M=M,
        C_per_alpha=C_per_alpha,
        C_max=C_max,
        indices=indices,
        alpha_mask=alpha_mask,
        device=device,
    )


def create_tensor_superdata(
    supergraph: TensorSuperGraph,
    teacher_factors: List[torch.Tensor],
    f_distribution: str = 'rademacher',
    seed: int = 12345,
) -> TensorSuperData:
    """
    Create TensorSuperData with F and Y tensors.
    
    Args:
        supergraph: TensorSuperGraph with index structure
        teacher_factors: n tensors of (N_d, M) teacher factors
        f_distribution: 'rademacher' or 'gaussian'
        seed: Random seed for F generation
        
    Returns:
        TensorSuperData with F_super, Y_super, and expanded alpha mask
    """
    S = supergraph.S
    C_max = supergraph.C_max
    M = supergraph.M
    n = supergraph.n
    device = supergraph.device
    
    # Generate F: (S, C_max, M)
    torch.manual_seed(seed)
    if f_distribution == 'rademacher':
        F_super = (torch.randint(0, 2, (S, C_max, M), device=device) * 2 - 1).float()
    else:
        F_super = torch.randn(S, C_max, M, device=device)
    
    # Compute Y using teacher factors
    # Y[s, c] = (1/sqrt(M)) * sum_m F[s,c,m] * prod_d teacher[d][idx[s,c], m]
    alpha_scale = 1.0 / math.sqrt(M)
    
    # Gather teacher factors
    gathered = torch.stack([
        teacher_factors[d][supergraph.indices[d].long()]  # (S, C_max, M)
        for d in range(n)
    ])  # (n, S, C_max, M)
    
    # Product across dimensions
    product = gathered.prod(dim=0)  # (S, C_max, M)
    
    # Compute Y
    Y_super = alpha_scale * (F_super * product).sum(dim=2)  # (S, C_max)
    
    # Expand alpha mask: (A, C_max) -> (A, S*C_max)
    A = supergraph.A
    alpha_mask_exp = supergraph.alpha_mask.unsqueeze(1).expand(A, S, C_max).reshape(A, S * C_max)
    
    return TensorSuperData(
        supergraph=supergraph,
        F_super=F_super,
        Y_super=Y_super,
        teacher_factors=teacher_factors,
        alpha_mask_exp=alpha_mask_exp,
    )
