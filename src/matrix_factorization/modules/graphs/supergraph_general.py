"""
General Graph (Super-Graph) data structure for coupled sampling parallelization.

This extends the bipartite Super-Graph model to allow intra-connection:
- W-W edges (between W vectors)
- W-X edges (original bipartite edges)  
- X-X edges (between X vectors)

The general graph enables triangular loops, breaking the bipartite constraint.
"""

from dataclasses import dataclass
from typing import Tuple, List
import torch
import numpy as np


# Edge type constants
EDGE_TYPE_WW = 0  # Edge between two W nodes
EDGE_TYPE_WX = 1  # Edge between W and X nodes (original bipartite)
EDGE_TYPE_XX = 2  # Edge between two X nodes


@dataclass
class SuperGraphDataGeneral:
    """
    Super-Graph data structure for general graphs (allowing W-W, W-X, X-X edges).

    For S samples and A alpha values, we pre-compute:
    - Unified node indices a_idx, b_idx where a < b
    - Edge types to distinguish W-W, W-X, X-X
    - Alpha masks to select subsets of edges

    Node encoding:
    - Nodes [0, N1): W nodes (W_0, W_1, ..., W_{N1-1})
    - Nodes [N1, N1+N2): X nodes (X_0, X_1, ..., X_{N2-1})

    Attributes:
        a_idx: (S, C_max) First node index (smaller)
        b_idx: (S, C_max) Second node index (larger), a < b guaranteed
        edge_type: (S, C_max) Edge type: 0=W-W, 1=W-X, 2=X-X
        C_per_alpha: (A,) Number of active edges for each alpha
        alpha_mask: (A, C_max) Boolean mask, mask[k, :C_k] = True
        N1, N2: Matrix dimensions
        C_max: Maximum number of edges (at alpha_max)
        seeds: (S,) Random seed for each sample
        alpha_values: (A,) Alpha values
    """
    a_idx: torch.Tensor        # (S, C_max) first node
    b_idx: torch.Tensor        # (S, C_max) second node
    edge_type: torch.Tensor    # (S, C_max) edge type
    C_per_alpha: torch.Tensor  # (A,)
    alpha_mask: torch.Tensor   # (A, C_max)
    N1: int
    N2: int
    C_max: int
    seeds: torch.Tensor        # (S,)
    alpha_values: torch.Tensor # (A,)

    @property
    def N_total(self) -> int:
        """Total number of nodes (W + X)."""
        return self.N1 + self.N2

    def get_active_edges(self, alpha_idx: int) -> int:
        """Return number of active edges for given alpha index."""
        return int(self.C_per_alpha[alpha_idx].item())

    def get_sample_indices(self, sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (a_idx, b_idx, edge_type) for a specific sample."""
        return self.a_idx[sample_idx], self.b_idx[sample_idx], self.edge_type[sample_idx]

    def to(self, device: torch.device) -> 'SuperGraphDataGeneral':
        """Move all tensors to specified device."""
        return SuperGraphDataGeneral(
            a_idx=self.a_idx.to(device),
            b_idx=self.b_idx.to(device),
            edge_type=self.edge_type.to(device),
            C_per_alpha=self.C_per_alpha.to(device),
            alpha_mask=self.alpha_mask.to(device),
            N1=self.N1,
            N2=self.N2,
            C_max=self.C_max,
            seeds=self.seeds.to(device),
            alpha_values=self.alpha_values.to(device),
        )


def decode_edge_id_general(edge_ids: torch.Tensor, N_total: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode linear edge IDs to (a, b) pairs for a general undirected graph.
    
    Edge ID encoding for undirected graph without self-loops:
    Edges are ordered as (0,1), (0,2), ..., (0,N-1), (1,2), ..., (N-2,N-1)
    Total edges = N*(N-1)/2
    
    For edge_id e, we find (a, b) where a < b:
    a = N - 2 - floor(sqrt((N-1)^2 - 2*e - 1/4) - 1/2)
    b = e - a*(2*N - a - 3)/2
    
    Args:
        edge_ids: (C,) tensor of edge IDs
        N_total: Total number of nodes
    
    Returns:
        a_idx: (C,) first node indices (smaller)
        b_idx: (C,) second node indices (larger)
    """
    # Use float64 for numerical stability in sqrt
    e = edge_ids.to(torch.float64)
    N = float(N_total)
    
    # Inverse formula for row index a
    # a = floor(N - 0.5 - sqrt((N-0.5)^2 - 2*e))
    # This gives the row where edge e belongs
    discriminant = (N - 0.5) ** 2 - 2 * e
    discriminant = torch.clamp(discriminant, min=0)  # Numerical safety
    a = torch.floor(N - 0.5 - torch.sqrt(discriminant)).to(torch.long)
    
    # Column index b
    # b = e - a*(2*N - a - 3)/2 + 1
    # Offset within row a
    row_start = (a * (2 * N - a - 3) / 2 + 1).to(torch.long)
    b = (edge_ids - row_start + a + 1).to(torch.long)
    
    # Ensure a < b
    a = torch.clamp(a, min=0, max=N_total - 2)
    b = torch.clamp(b, min=1, max=N_total - 1)
    
    return a, b


def compute_edge_type(a_idx: torch.Tensor, b_idx: torch.Tensor, N1: int) -> torch.Tensor:
    """
    Compute edge type based on node indices.
    
    Node encoding:
    - [0, N1): W nodes
    - [N1, N1+N2): X nodes
    
    Edge types:
    - 0: W-W (both a, b < N1)
    - 1: W-X (a < N1, b >= N1)
    - 2: X-X (both a, b >= N1)
    
    Args:
        a_idx: (C,) first node indices
        b_idx: (C,) second node indices
        N1: Number of W nodes
    
    Returns:
        edge_type: (C,) edge types
    """
    a_is_W = a_idx < N1
    b_is_W = b_idx < N1
    
    edge_type = torch.zeros_like(a_idx, dtype=torch.int8)
    edge_type[a_is_W & b_is_W] = EDGE_TYPE_WW
    edge_type[a_is_W & ~b_is_W] = EDGE_TYPE_WX
    edge_type[~a_is_W & ~b_is_W] = EDGE_TYPE_XX
    
    return edge_type


def create_supergraph_general(
    N1: int,
    N2: int,
    M: int,
    alpha_values: List[float],
    S: int,
    base_seed: int,
    device: torch.device,
    num_workers: int = 1,
    sample_offset: int = 0,
) -> SuperGraphDataGeneral:
    """
    Create a SuperGraph for general graphs (allowing W-W, W-X, X-X edges).

    The key idea: for each sample, we generate a random permutation of
    all possible edges in the complete graph. Then for each alpha, we take
    the first C_alpha = floor(alpha * M * N1) edges from this permutation.

    Node encoding:
    - Nodes [0, N1): W nodes
    - Nodes [N1, N1+N2): X nodes

    Args:
        N1: Number of W nodes
        N2: Number of X nodes
        M: Latent dimension (used for edge count calculation)
        alpha_values: List of alpha values to sweep
        S: Number of samples
        base_seed: Base random seed
        device: Torch device
        num_workers: Number of parallel workers (CPU only)

    Returns:
        SuperGraphDataGeneral with pre-computed indices and masks
    """
    alpha_values = np.array(alpha_values)
    A = len(alpha_values)
    N_total = N1 + N2

    # Compute edge counts for each alpha
    # C = alpha * M * N1 (same density definition as bipartite)
    C_per_alpha = np.floor(alpha_values * M * N1).astype(np.int64)
    C_max = int(C_per_alpha.max())

    # Handle edge case: if C_max is 0, set minimum
    if C_max == 0:
        C_max = 1
        C_per_alpha = np.maximum(C_per_alpha, 0)

    # Total possible edges (undirected, no self-loops)
    total_edges = N_total * (N_total - 1) // 2

    # Ensure C_max doesn't exceed total edges
    C_max = min(C_max, total_edges)
    C_per_alpha = np.minimum(C_per_alpha, total_edges)

    # Choose index dtype based on node count
    max_dim = N_total
    if max_dim < 32767:
        idx_dtype = torch.int16  # 75% memory savings vs int64
    else:
        idx_dtype = torch.int32  # 50% memory savings vs int64

    a_idx_all = torch.zeros((S, C_max), dtype=idx_dtype, device=device)
    b_idx_all = torch.zeros((S, C_max), dtype=idx_dtype, device=device)
    edge_type_all = torch.zeros((S, C_max), dtype=torch.int8, device=device)
    seeds = torch.zeros(S, dtype=torch.long, device=device)

    # Prepare seeds
    for s in range(S):
        seeds[s] = base_seed + (int(sample_offset) + s) * 1000

    # Generate edges for each sample
    if device.type == 'cuda':
        # GPU generation
        try:
            # Check memory requirements
            total_elements = S * total_edges
            needed_gb = total_elements * 2 / (1024**3)  # float16
            
            if needed_gb < 4.0:  # Only use batch mode if < 4GB VRAM
                noise = torch.empty((S, total_edges), device=device, dtype=torch.float16)
                
                for s in range(S):
                    seed = seeds[s].item()
                    gen = torch.Generator(device=device).manual_seed(seed)
                    noise[s].uniform_(0, 1, generator=gen)
                
                # Top-K selection
                _, flat_indices = torch.topk(noise, k=C_max, dim=1)
                
                # Decode to (a, b) pairs
                for s in range(S):
                    a_idx, b_idx = decode_edge_id_general(flat_indices[s], N_total)
                    a_idx_all[s] = a_idx.to(idx_dtype)
                    b_idx_all[s] = b_idx.to(idx_dtype)
                    edge_type_all[s] = compute_edge_type(a_idx, b_idx, N1)
                
                del noise
            else:
                raise MemoryError("Too large for batch init")
                
        except (MemoryError, RuntimeError):
            # Fallback to sequential GPU loop
            for s in range(S):
                seed = seeds[s].item()
                gen = torch.Generator(device=device).manual_seed(seed)
                perm = torch.randperm(total_edges, generator=gen, device=device)[:C_max]
                a_idx, b_idx = decode_edge_id_general(perm, N_total)
                a_idx_all[s] = a_idx.to(idx_dtype)
                b_idx_all[s] = b_idx.to(idx_dtype)
                edge_type_all[s] = compute_edge_type(a_idx, b_idx, N1)
    else:
        # CPU generation
        for s in range(S):
            seed = seeds[s].item()
            gen = torch.Generator(device='cpu').manual_seed(seed)
            perm = torch.randperm(total_edges, generator=gen)[:C_max]
            a_idx, b_idx = decode_edge_id_general(perm, N_total)
            a_idx_all[s] = a_idx.to(idx_dtype).to(device)
            b_idx_all[s] = b_idx.to(idx_dtype).to(device)
            edge_type_all[s] = compute_edge_type(a_idx, b_idx, N1).to(device)

    # Create alpha masks
    # mask[a, c] = True if c < C_per_alpha[a]
    alpha_mask = torch.zeros((A, C_max), dtype=torch.bool, device=device)
    for a in range(A):
        alpha_mask[a, :C_per_alpha[a]] = True

    return SuperGraphDataGeneral(
        a_idx=a_idx_all,
        b_idx=b_idx_all,
        edge_type=edge_type_all,
        C_per_alpha=torch.tensor(C_per_alpha, dtype=torch.long, device=device),
        alpha_mask=alpha_mask,
        N1=N1,
        N2=N2,
        C_max=C_max,
        seeds=seeds,
        alpha_values=torch.tensor(alpha_values, dtype=torch.float32, device=device),
    )


def get_memory_estimate_general(N1: int, N2: int, M: int, alpha_max: float, S: int, A: int) -> dict:
    """
    Estimate memory usage for SuperGraphDataGeneral.

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
    N_total = N1 + N2
    total_edges = N_total * (N_total - 1) // 2
    C_max = min(C_max, total_edges)

    # Tensor sizes in bytes
    a_idx_bytes = S * C_max * 4  # int32
    b_idx_bytes = S * C_max * 4
    edge_type_bytes = S * C_max * 1  # int8
    C_per_alpha_bytes = A * 8
    alpha_mask_bytes = A * C_max * 1  # bool
    seeds_bytes = S * 8
    alpha_values_bytes = A * 4  # float32

    total_bytes = (a_idx_bytes + b_idx_bytes + edge_type_bytes +
                   C_per_alpha_bytes + alpha_mask_bytes + 
                   seeds_bytes + alpha_values_bytes)

    return {
        'a_idx_MB': a_idx_bytes / 1e6,
        'b_idx_MB': b_idx_bytes / 1e6,
        'edge_type_MB': edge_type_bytes / 1e6,
        'alpha_mask_MB': alpha_mask_bytes / 1e6,
        'total_MB': total_bytes / 1e6,
        'C_max': C_max,
        'total_possible_edges': total_edges,
    }
