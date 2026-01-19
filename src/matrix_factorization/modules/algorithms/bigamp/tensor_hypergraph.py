"""
N-uniform hypergraph generation for tensor spreading.

This module provides functions to generate random n-uniform hypergraphs
for use in tensor CP decomposition with the spreading model.

A n-uniform hypergraph has hyperedges that each connect exactly n nodes,
one from each dimension of the tensor.
"""

import torch
from typing import Tuple
from .tensor_data import TensorHypergraph


def generate_tensor_hypergraph(
    dims: Tuple[int, ...],
    alpha: float,
    M: int,
    seed: int,
    device: torch.device,
) -> TensorHypergraph:
    """
    Generate random n-uniform hypergraph.
    
    The number of hyperedges C is determined by α (average degree):
    - When all dims equal N: C = α * N
    - When dims differ: C = α * min(dims)
    
    This ensures each node has approximately α connections on average
    (for the smallest dimension, larger dimensions have fewer connections).
    
    Args:
        dims: (N_1, ..., N_n) factor dimensions
        alpha: average node degree (C = alpha * min(N))
        M: latent dimension (unused but kept for interface consistency)
        seed: random seed for reproducibility
        device: torch device
        
    Returns:
        TensorHypergraph with random hyperedges
        
    Example:
        >>> dims = (50, 50, 50)
        >>> hg = generate_tensor_hypergraph(dims, alpha=2.0, M=20, seed=42, device='cpu')
        >>> hg.C  # = 2.0 * 50 = 100 hyperedges
        100
    """
    n = len(dims)
    
    # Compute number of hyperedges
    # Use min(dims) to ensure even the smallest dimension has α average degree
    N_min = min(dims)
    C = int(alpha * N_min)
    C = max(1, C)  # At least 1 edge
    
    # Generate random indices for each dimension
    gen = torch.Generator(device=device).manual_seed(seed)
    
    indices = []
    for d in range(n):
        idx = torch.randint(0, dims[d], (C,), generator=gen, device=device)
        indices.append(idx)
    
    return TensorHypergraph(order=n, indices=indices, dims=dims)


def generate_tensor_observations(
    teacher_factors: list,  # List of n tensors, each (N_d, M)
    hypergraph: TensorHypergraph,
    seed: int,
    device: torch.device,
    f_distribution: str = 'rademacher',
):
    """
    Generate F and Y for tensor spreading.
    
    Given teacher factors and a hypergraph structure, generates:
    - F: (C, M) spreading coefficients
    - Y: (C,) observed values
    
    Args:
        teacher_factors: List of n teacher factor matrices
        hypergraph: TensorHypergraph defining observation structure
        seed: random seed for F generation
        device: torch device
        f_distribution: 'rademacher' or 'gaussian'
        
    Returns:
        F: (C, M) spreading coefficients
        Y: (C,) observations
    """
    from .tensor_step import forward_pass_tensor
    
    C = hypergraph.C
    M = teacher_factors[0].shape[1]
    
    gen = torch.Generator(device=device).manual_seed(seed)
    
    if f_distribution == 'rademacher':
        F = (torch.randint(0, 2, (C, M), generator=gen, device=device) * 2 - 1).float()
    else:  # gaussian
        F = torch.randn(C, M, generator=gen, device=device)
    
    # Compute Y from teacher
    Y = forward_pass_tensor(teacher_factors, F, hypergraph.indices)
    
    return F, Y
