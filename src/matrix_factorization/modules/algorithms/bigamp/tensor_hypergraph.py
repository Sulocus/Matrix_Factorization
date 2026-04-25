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
    
    The number of hyperedges C is determined by α as a DoF-scaled density:
    - C = α * (sum(N_i) * M)
    
    This ensures the number of observations scales linearly with the number of 
    free parameters (Degrees of Freedom), preventing explosion for large tensors.
    
    Args:
        dims: (N_1, ..., N_n) factor dimensions
        alpha: average node degree (C = alpha * min(N))
        M: latent dimension (unused but kept for interface consistency)
        seed: random seed for reproducibility
        device: torch device
        
    Returns:
        TensorHypergraph with random hyperedges
    """
    n = len(dims)
    
    # Compute number of hyperedges C
    # Scaling law: C = alpha * DoF
    # DoF approx sum(N_i) * M
    # This ensures linear scaling with problem size, correcting the previous volume-based scaling
    dof = sum(dims) * M
    C = int(alpha * dof)
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


def generate_F_batch(
    S: int,
    C: int,
    M: int,
    base_seed: int,
    device: torch.device,
    distribution: str = 'rademacher',
) -> torch.Tensor:
    """
    Generate S independent F realizations for sample parallelization.
    
    Each sample gets an independent random F matrix, seeded deterministically
    from base_seed + sample_index * 1000.
    
    Args:
        S: Number of samples
        C: Number of hyperedges
        M: Latent dimension
        base_seed: Base random seed
        device: torch device
        distribution: 'rademacher' or 'gaussian'
        
    Returns:
        F: (S, C, M) spreading coefficients for all samples
    """
    F_list = []
    for s in range(S):
        gen = torch.Generator(device=device).manual_seed(base_seed + s * 1000)
        if distribution == 'rademacher':
            F_s = (torch.randint(0, 2, (C, M), generator=gen, device=device) * 2 - 1).float()
        else:  # gaussian
            F_s = torch.randn(C, M, generator=gen, device=device)
        F_list.append(F_s)
    
    return torch.stack(F_list)  # (S, C, M)


def generate_tensor_observations_batch(
    teacher_factors: list,  # List of n tensors, each (N_d, M)
    hypergraph: TensorHypergraph,
    S: int,
    base_seed: int,
    device: torch.device,
    f_distribution: str = 'rademacher',
):
    """
    Generate batched F and Y for sample parallelization.
    
    Creates S independent (F, Y) pairs where:
    - Each F[s] is an independent random realization
    - Y[s] = forward_pass(teacher_factors, F[s], indices)
    
    The hypergraph indices are SHARED across all samples.
    
    Args:
        teacher_factors: List of n teacher factor matrices, each (N_d, M)
        hypergraph: TensorHypergraph defining observation structure
        S: Number of samples
        base_seed: Base random seed
        device: torch device
        f_distribution: 'rademacher' or 'gaussian'
        
    Returns:
        F: (S, C, M) spreading coefficients
        Y: (S, C) observations
    """
    from .tensor_step import forward_pass_tensor
    
    C = hypergraph.C
    M = teacher_factors[0].shape[1]
    
    # Generate batched F
    F = generate_F_batch(S, C, M, base_seed, device, f_distribution)  # (S, C, M)
    
    # Compute Y for each sample using single-sample forward pass
    # (Could optimize with batch forward pass, but this is simple and correct)
    Y_list = []
    for s in range(S):
        Y_s = forward_pass_tensor(teacher_factors, F[s], hypergraph.indices)
        Y_list.append(Y_s)
    
    Y = torch.stack(Y_list)  # (S, C)
    
    return F, Y
