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


def _generate_single_sample(args):
    """Worker function for parallel graph generation."""
    seed, N1, N2, C_max, idx_dtype, total_edges = args
    
    # Independent RNG for each sample
    gen = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(total_edges, generator=gen)[:C_max]
    
    # Return directly as tensor
    i_idx = (perm // N2).to(idx_dtype)
    j_idx = (perm % N2).to(idx_dtype)
    return i_idx, j_idx


def create_supergraph(
    N1: int,
    N2: int,
    M: int,
    alpha_values: List[float],
    S: int,
    base_seed: int,
    device: torch.device,
    num_workers: int = 1,
) -> SuperGraphData:
    """
    Create a SuperGraph for coupled sampling.

    The key idea: for each sample, we generate a random permutation of
    all possible edges (i, j). Then for each alpha, we take the first
    C_alpha = floor(alpha * M * N1) edges from this permutation.

    Args:
        N1: Number of rows
        N2: Number of columns
        M: Latent dimension
        alpha_values: List of alpha values to sweep
        S: Number of samples
        base_seed: Base random seed
        device: Torch device
        num_workers: Number of parallel workers (CPU only)

    Returns:
        SuperGraphData with pre-computed indices and masks
    """
    import concurrent.futures
    import multiprocessing
    
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
    # Use int16 if dimensions fit (N < 32767), int32 otherwise - saves 75%/50% vs int64
    max_dim = max(N1, N2)
    if max_dim < 32767:
        idx_dtype = torch.int16  # 75% memory savings vs int64
    else:
        idx_dtype = torch.int32  # 50% memory savings vs int64
    
    i_idx_all = torch.zeros((S, C_max), dtype=idx_dtype, device=device)
    j_idx_all = torch.zeros((S, C_max), dtype=idx_dtype, device=device)
    seeds = torch.zeros(S, dtype=torch.long, device=device)

    # Prepare seeds
    for s in range(S):
        seeds[s] = base_seed + s * 1000

    # PARALLEL GENERATION (CPU ONLY)
    # If device is CUDA, generation inside the loop using CUDA generator is often faster
    # than MP overhead. MP is beneficial for CPU generation.
    run_parallel_cpu = (device.type == 'cpu' and num_workers > 1)
    
    if device.type == 'cuda':
        # GPU OPTIMIZED: Batched generation
        # Instead of S randperms, we generate a noise matrix and take top-k.
        # This vectorizes the sorting/selection.
        
        # 1. Allocate noise tensor (S, Total_Edges)
        # Use float16 to save memory (we don't need high precision for random order)
        try:
            # Check size to avoid OOM on huge matrices
            total_elements = S * total_edges
            needed_gb = total_elements * 2 / (1024**3) # float16
            
            if needed_gb < 4.0: # Only use batch mode if < 4GB VRAM
                noise = torch.empty((S, total_edges), device=device, dtype=torch.float16)
                
                # 2. Fill with seeded noise (Loop is fast for just random gen)
                for s in range(S):
                    seed = seeds[s].item()
                    gen = torch.Generator(device=device).manual_seed(seed)
                    noise[s].uniform_(0, 1, generator=gen)
                
                # 3. Top-K Selection (Batched)
                # This replaces the slow randperm loop
                _, flat_indices = torch.topk(noise, k=C_max, dim=1)
                
                # 4. Convert to (i, j)
                i_idx_all = (flat_indices // N2).to(idx_dtype)
                j_idx_all = (flat_indices % N2).to(idx_dtype)
                
                # Free noise memory immediately
                del noise
                
            else:
                # Fallback to Loop if too large
                raise MemoryError("Too large for batch init")
                
        except (MemoryError, RuntimeError):
            # Fallback to sequential GPU loop
            for s in range(S):
                seed = seeds[s].item()
                gen = torch.Generator(device=device).manual_seed(seed)
                perm = torch.randperm(total_edges, generator=gen, device=device)[:C_max]
                i_idx_all[s] = (perm // N2).to(idx_dtype)
                j_idx_all[s] = (perm % N2).to(idx_dtype)

    elif run_parallel_cpu:
        tasks = []
        for s in range(S):
            tasks.append((seeds[s].item(), N1, N2, C_max, idx_dtype, total_edges))
        
        # Use ProcessPool to bypass GIL
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Map returns results in order
            results = list(executor.map(_generate_single_sample, tasks))
            
        # Assemble results
        for s, (i_idx, j_idx) in enumerate(results):
            i_idx_all[s] = i_idx.to(device)
            j_idx_all[s] = j_idx.to(device)
            
    else:
        # Sequential CPU generation
        for s in range(S):
            seed = seeds[s].item()
            gen = torch.Generator(device='cpu').manual_seed(seed)
            perm = torch.randperm(total_edges, generator=gen)[:C_max]
            i_idx_all[s] = (perm // N2).to(idx_dtype).to(device)
            j_idx_all[s] = (perm % N2).to(idx_dtype).to(device)

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

    # Tensor sizes in bytes (i_idx, j_idx now int32)
    i_idx_bytes = S * C_max * 4  # int32
    j_idx_bytes = S * C_max * 4
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
