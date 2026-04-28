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

import hashlib
import math
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from matrix_factorization.core.distributions import (
    F_DISTRIBUTION_GAUSSIAN,
    F_DISTRIBUTION_ISING,
    normalize_f_distribution,
)

logger = logging.getLogger(__name__)


def stable_partition_seed(base_seed: int, *parts: object) -> int:
    """Return a deterministic torch seed independent of Python hash randomization."""
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


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
    offset_indices: List[torch.Tensor]  # n × (S*C_max,) - precomputed for gather
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
    
    def get_offset_indices(self) -> List[torch.Tensor]:
        """
        Get precomputed offset indices for gather/scatter operations.
        
        These indices include sample offsets, ready for direct use:
            gathered = factors[d][:, offset_indices[d]]
        
        Returns:
            n tensors of shape (S*C_max,) with sample offsets applied
        """
        return self.offset_indices


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
    partition_invariant: bool = False,
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
    
    # Compute degrees of freedom (User Definition)
    # User constraint: Avg Degree should be alpha * M.
    # Total Nodes = n * N. Total Edges = C.
    # Avg Degree = n * C / (n * N) = C/N.
    # We want C/N = alpha * M => C = alpha * M * N.
    # Original (Theoretical): C = alpha * (n * N * M).
    # We adopt the User's definition for consistency with their constraints.
    ref_dim = dims[0]
    # dof = ref_dim * M # This effective 'dof' gives the user-expected scaling
    
    # PHYSICAL CONSTRAINT: alpha_max = N/M
    # When alpha > N/M, each node would need to connect to more edges than possible nodes.
    # This is physically impossible, so we cap alpha at this limit.
    alpha_max = ref_dim / M
    capped_alpha_values = []
    for alpha in alpha_values:
        if alpha > alpha_max:
            logger.warning(f"Alpha {alpha:.2f} exceeds physical limit N/M = {alpha_max:.2f}. Capping to {alpha_max:.2f}.")
            capped_alpha_values.append(alpha_max)
        else:
            capped_alpha_values.append(alpha)
    
    # Compute C for each alpha (using capped values)
    # C = alpha * M * N
    C_per_alpha = [int(alpha * M * ref_dim) for alpha in capped_alpha_values]
    C_max = max(1, max(C_per_alpha) if C_per_alpha else 1)
    
    # Generate indices for C_max edges (shared across alphas).  The legacy path
    # intentionally preserves the previous global RNG behavior.  The opt-in
    # partition-invariant path gives each (dimension, sample) its own stream so
    # the first C edges for one alpha do not depend on the batch C_max.
    if partition_invariant:
        indices = []
        for d in range(n):
            sample_indices = []
            for s in range(S):
                gen = torch.Generator(device=device).manual_seed(
                    stable_partition_seed(seed, "tensor_supergraph", "indices", d, s)
                )
                sample_indices.append(
                    torch.randint(0, dims[d], (C_max,), generator=gen, device=device)
                )
            indices.append(torch.stack(sample_indices, dim=0))
    else:
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
    
    # Precompute offset_indices for efficient gather/scatter
    # This eliminates repeated sample_offsets calculation in tensor_step_super
    # Similar to compute_offset_indices() in Bipartite spreading.py
    offset_indices = []
    for d in range(n):
        N_d = dims[d]
        idx_flat = indices[d].reshape(-1)  # (S*C_max,)
        # sample_offsets: each sample's factors are offset by s * N_d
        sample_offsets = torch.arange(S, device=device).unsqueeze(1) * N_d  # (S, 1)
        sample_offsets = sample_offsets.expand(S, C_max).reshape(-1)  # (S*C_max,)
        offset_indices.append(idx_flat + sample_offsets)
    
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
        offset_indices=offset_indices,
        device=device,
    )


def create_tensor_superdata(
    supergraph: TensorSuperGraph,
    teacher_factors: List[torch.Tensor],
    f_distribution: str = 'ising',
    seed: int = 12345,
    partition_invariant: bool = False,
) -> TensorSuperData:
    """
    Create TensorSuperData with F and Y tensors.
    
    Args:
        supergraph: TensorSuperGraph with index structure
        teacher_factors: n tensors of (N_d, M) teacher factors
        f_distribution: 'ising' or 'gaussian'
        seed: Random seed for F generation
        
    Returns:
        TensorSuperData with F_super, Y_super, and expanded alpha mask
    """
    S = supergraph.S
    C_max = supergraph.C_max
    M = supergraph.M
    n = supergraph.n
    device = supergraph.device
    f_distribution = normalize_f_distribution(f_distribution)
    
    # Generate F: (S, C_max, M).  The legacy path intentionally keeps the old
    # global RNG behavior.  The opt-in partition-invariant path gives each
    # sample its own stream so the first C edges do not depend on batch C_max.
    if partition_invariant:
        f_samples = []
        for s in range(S):
            gen = torch.Generator(device=device).manual_seed(
                stable_partition_seed(seed, "tensor_superdata", "F", s)
            )
            if f_distribution == F_DISTRIBUTION_ISING:
                F_s = (torch.randint(0, 2, (C_max, M), generator=gen, device=device, dtype=torch.int8) * 2 - 1)
            elif f_distribution == F_DISTRIBUTION_GAUSSIAN:
                F_s = torch.randn(C_max, M, generator=gen, device=device)
            f_samples.append(F_s)
        F_super = torch.stack(f_samples, dim=0)
    else:
        torch.manual_seed(seed)
        if f_distribution == F_DISTRIBUTION_ISING:
            F_super = (torch.randint(0, 2, (S, C_max, M), device=device, dtype=torch.int8) * 2 - 1)
        elif f_distribution == F_DISTRIBUTION_GAUSSIAN:
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
