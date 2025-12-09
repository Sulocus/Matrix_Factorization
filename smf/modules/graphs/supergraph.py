"""
Super-Graph data structure for coupled sampling parallelization.

The Super-Graph strategy enables parallel processing across alpha values
by treating smaller alpha graphs as subsets of larger ones.

Key insight: Instead of generating independent random graphs for each alpha,
we generate one large "super-graph" and use prefix masks to select
the appropriate number of edges for each alpha value.

This approach:
1. Reduces variance through coupled sampling
2. Enables GPU parallelization across all alpha values
3. Maintains statistical correctness (marginal distribution is correct)
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch
import numpy as np


@dataclass
class SuperGraphData:
    """
    Super-Graph data structure for coupled sampling.

    For S samples and A alpha values, we pre-compute:
    - Edge indices for each sample (at maximum alpha)
    - Alpha masks to select subsets of edges

    Attributes:
        i_idx: (S, C_max) Row indices of edges for each sample
        j_idx: (S, C_max) Column indices of edges for each sample
        C_per_alpha: (A,) Number of active edges for each alpha
        alpha_mask: (A, C_max) Boolean mask, mask[k, :C_k] = True
        N1, N2: Matrix dimensions
        C_max: Maximum number of edges (at alpha_max)
        seeds: (S,) Random seed for each sample
        alpha_values: (A,) Alpha values
    """
    i_idx: torch.Tensor       # (S, C_max)
    j_idx: torch.Tensor       # (S, C_max)
    C_per_alpha: torch.Tensor # (A,)
    alpha_mask: torch.Tensor  # (A, C_max)
    N1: int
    N2: int
    C_max: int
    seeds: torch.Tensor       # (S,)
    alpha_values: torch.Tensor  # (A,)

    def get_active_edges(self, alpha_idx: int) -> int:
        """Return number of active edges for given alpha index."""
        return int(self.C_per_alpha[alpha_idx].item())

    def get_sample_indices(self, sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (i_idx, j_idx) for a specific sample."""
        return self.i_idx[sample_idx], self.j_idx[sample_idx]

    def to(self, device: torch.device) -> 'SuperGraphData':
        """Move all tensors to specified device."""
        return SuperGraphData(
            i_idx=self.i_idx.to(device),
            j_idx=self.j_idx.to(device),
            C_per_alpha=self.C_per_alpha.to(device),
            alpha_mask=self.alpha_mask.to(device),
            N1=self.N1,
            N2=self.N2,
            C_max=self.C_max,
            seeds=self.seeds.to(device),
            alpha_values=self.alpha_values.to(device),
        )


def create_supergraph(
    N1: int,
    N2: int,
    M: int,
    alpha_values: List[float],
    S: int,
    base_seed: int,
    device: torch.device,
) -> SuperGraphData:
    """
    Create a SuperGraph for coupled sampling.

    The key idea: for each sample, we generate a random permutation of
    all possible edges (i, j). Then for each alpha, we take the first
    C_alpha = floor(alpha * M * N1) edges from this permutation.

    This ensures:
    1. Smaller alpha's edges are subsets of larger alpha's edges
    2. Each sample has independent random structure
    3. The marginal distribution of edges is uniform random

    Args:
        N1: Number of rows
        N2: Number of columns
        M: Latent dimension
        alpha_values: List of alpha values to sweep
        S: Number of samples
        base_seed: Base random seed
        device: Torch device

    Returns:
        SuperGraphData with pre-computed indices and masks
    """
    alpha_values = np.array(alpha_values)
    A = len(alpha_values)

    # Compute edge counts for each alpha
    # C = alpha * M * N1 (average degree per left node is alpha * M)
    C_per_alpha = np.floor(alpha_values * M * N1).astype(np.int64)
    C_max = int(C_per_alpha.max())

    # Handle edge case: if C_max is 0, set minimum
    if C_max == 0:
        C_max = 1
        C_per_alpha = np.maximum(C_per_alpha, 0)

    # Total possible edges
    total_edges = N1 * N2

    # Ensure C_max doesn't exceed total edges
    C_max = min(C_max, total_edges)
    C_per_alpha = np.minimum(C_per_alpha, total_edges)

    # Generate indices for each sample
    i_idx_all = torch.zeros((S, C_max), dtype=torch.long, device=device)
    j_idx_all = torch.zeros((S, C_max), dtype=torch.long, device=device)
    seeds = torch.zeros(S, dtype=torch.long, device=device)

    for s in range(S):
        seed = base_seed + s * 1000
        seeds[s] = seed

        # Generate random permutation of edge indices
        # Use GPU generator if device is CUDA for much faster generation
        if device.type == 'cuda':
            gen = torch.Generator(device=device).manual_seed(seed)
            perm = torch.randperm(total_edges, generator=gen, device=device)[:C_max]
            i_idx_all[s] = perm // N2
            j_idx_all[s] = perm % N2
        else:
            gen = torch.Generator(device='cpu').manual_seed(seed)
            perm = torch.randperm(total_edges, generator=gen)[:C_max]
            i_idx_all[s] = (perm // N2).to(device)
            j_idx_all[s] = (perm % N2).to(device)

    # Create alpha masks
    # mask[a, c] = True if c < C_per_alpha[a]
    alpha_mask = torch.zeros((A, C_max), dtype=torch.bool, device=device)
    for a in range(A):
        alpha_mask[a, :C_per_alpha[a]] = True

    return SuperGraphData(
        i_idx=i_idx_all,
        j_idx=j_idx_all,
        C_per_alpha=torch.tensor(C_per_alpha, dtype=torch.long, device=device),
        alpha_mask=alpha_mask,
        N1=N1,
        N2=N2,
        C_max=C_max,
        seeds=seeds,
        alpha_values=torch.tensor(alpha_values, dtype=torch.float32, device=device),
    )


def get_memory_estimate(N1: int, N2: int, M: int, alpha_max: float, S: int, A: int) -> dict:
    """
    Estimate memory usage for SuperGraph data.

    Args:
        N1, N2: Matrix dimensions
        M: Latent dimension
        alpha_max: Maximum alpha value
        S: Number of samples
        A: Number of alpha points

    Returns:
        Dictionary with memory estimates in MB
    """
    C_max = int(alpha_max * M * N1)

    # Tensor sizes in bytes
    i_idx_bytes = S * C_max * 8  # int64
    j_idx_bytes = S * C_max * 8
    C_per_alpha_bytes = A * 8
    alpha_mask_bytes = A * C_max * 1  # bool
    seeds_bytes = S * 8
    alpha_values_bytes = A * 4  # float32

    total_bytes = (i_idx_bytes + j_idx_bytes + C_per_alpha_bytes +
                   alpha_mask_bytes + seeds_bytes + alpha_values_bytes)

    return {
        'i_idx_MB': i_idx_bytes / 1e6,
        'j_idx_MB': j_idx_bytes / 1e6,
        'alpha_mask_MB': alpha_mask_bytes / 1e6,
        'total_MB': total_bytes / 1e6,
        'C_max': C_max,
    }
