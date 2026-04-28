"""
F Generation strategies for BiG-AMP Spreading.
"""

from typing import Tuple, Callable, Dict
import math
import torch
from ....core.distributions import (
    F_DISTRIBUTION_GAUSSIAN,
    F_DISTRIBUTION_ISING,
    is_ising_f_distribution,
    normalize_f_distribution,
)
from ...graphs.supergraph import SuperGraphData
from ...graphs.supergraph_general import SuperGraphDataGeneral, EDGE_TYPE_WW, EDGE_TYPE_WX, EDGE_TYPE_XX


def generate_F_gaussian(
    C: int,
    M: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate F ~ N(0, 1) (Gaussian distribution).

    Args:
        C: Number of edges
        M: Hidden dimension
        seed: Random seed
        device: Target device

    Returns:
        F: (C, M) tensor with F ~ N(0, 1)
    """
    if C == 0:
        return torch.empty(0, M, device=device, dtype=torch.float32)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed ^ 0x5DEECE66D)
    return torch.randn(C, M, device=device, dtype=torch.float32, generator=gen)


def generate_F_ising(
    C: int,
    M: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate Ising spreading coefficients F, uniformly distributed on {-1, +1}.

    OPTIMIZATION: Uses int8 storage for 4x memory reduction.
    Values are stored as int8 and converted to float on demand.

    Properties:
        E[F] = 0
        Var[F] = 1
    Same first two moments as Gaussian.

    Args:
        C: Number of edges
        M: Hidden dimension
        seed: Random seed
        device: Target device

    Returns:
        F: (C, M) tensor with F ∈ {-1, +1} stored as int8
    """
    if C == 0:
        return torch.empty(0, M, device=device, dtype=torch.int8)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed ^ 0x5DEECE66D)

    # Generate 0 or 1, then map to -1 or +1, store as int8
    bits = torch.randint(0, 2, (C, M), device=device, dtype=torch.int8, generator=gen)
    return bits * 2 - 1  # {0, 1} -> {-1, +1} as int8


# Strategy dictionary
F_GENERATORS: Dict[str, Callable] = {
    F_DISTRIBUTION_GAUSSIAN: generate_F_gaussian,
    F_DISTRIBUTION_ISING: generate_F_ising,
}


def generate_F_super(
    supergraph: SuperGraphData,
    M: int,
    base_seed: int,
    device: torch.device,
    f_distribution: str = 'gaussian',
) -> torch.Tensor:
    """
    Generate F_super: (S, C_max, M) with quenched disorder.

    Each sample has independent F, but within a sample,
    different alphas share the same F (just different masks).

    OPTIMIZATION: For Ising F, stores as int8 (4x memory reduction).

    Args:
        supergraph: SuperGraphData with edge structure
        M: Hidden dimension
        base_seed: Base seed for F generation
        device: Target device
        f_distribution: 'gaussian' or 'ising'

    Returns:
        F_super: (S, C_max, M) tensor (float32 for gaussian, int8 for ising)
    """
    f_distribution = normalize_f_distribution(f_distribution)
    if f_distribution not in F_GENERATORS:
        raise ValueError(
            f"Invalid f_distribution='{f_distribution}'. "
            f"Available: {list(F_GENERATORS.keys())}"
        )

    generator = F_GENERATORS[f_distribution]
    S = supergraph.seeds.shape[0]
    C_max = supergraph.C_max

    # Determine dtype based on distribution
    dtype = torch.int8 if is_ising_f_distribution(f_distribution) else torch.float32
    F_super = torch.empty(S, C_max, M, device=device, dtype=dtype)

    for s in range(S):
        # Combine base_seed with sample seed for independence
        sample_seed = base_seed + int(supergraph.seeds[s].item())
        F_super[s] = generator(C_max, M, sample_seed, device)

    return F_super


def compute_Y_super(
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    supergraph: SuperGraphData,
    F_super: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Y values for all samples at all edge positions.

    Y[s, c] = (1/√M) Σ_μ F[s,c,μ] W[i[s,c],μ] X[μ, j[s,c]]

    Args:
        W_teacher: (N1, M) teacher W matrix
        X_teacher: (M, N2) teacher X matrix
        supergraph: SuperGraphData with edge indices
        F_super: (S, C_max, M) spreading coefficients (int8 or float32)

    Returns:
        Y_super: (S, C_max) Y values (always float32)
    """
    S, C_max, M = F_super.shape
    alpha_scale = 1.0 / math.sqrt(M)

    # Y is always float32 even if F is int8
    Y_super = torch.empty(S, C_max, device=F_super.device, dtype=torch.float32)

    for s in range(S):
        i_idx = supergraph.i_idx[s].long()  # (C_max,) - convert to long for indexing
        j_idx = supergraph.j_idx[s].long()  # (C_max,)

        W_sel = W_teacher[i_idx]     # (C_max, M)
        X_sel = X_teacher[:, j_idx].T  # (C_max, M)

        # Convert F to float for computation (handles int8 Ising F)
        F_s = F_super[s].float() if F_super.dtype == torch.int8 else F_super[s]
        
        # Y[c] = (1/√M) Σ_μ F[c,μ] W[i,μ] X[μ,j]
        Y_super[s] = alpha_scale * (F_s * W_sel * X_sel).sum(dim=1)

    return Y_super


def generate_F_super_general(
    supergraph: SuperGraphDataGeneral,
    M: int,
    base_seed: int,
    device: torch.device,
    f_distribution: str = F_DISTRIBUTION_ISING,
) -> torch.Tensor:
    """
    Generate spreading coefficients F for the general super-graph.
    Similar to generate_F_super but takes SuperGraphDataGeneral.
    """
    S = supergraph.seeds.shape[0]
    C_max = supergraph.C_max
    f_distribution = normalize_f_distribution(f_distribution)
    
    # Store as int8 for memory efficiency if Ising
    dtype = torch.int8 if is_ising_f_distribution(f_distribution) else torch.float32
    F_super = torch.empty(S, C_max, M, device=device, dtype=dtype)
    
    for s in range(S):
        # Use a unique seed for F generation distinct from graph content
        # Mix base_seed, sample index, and a magic number
        seed = base_seed + s * 777 + 100000
        gen = torch.Generator(device=device).manual_seed(seed)
        
        if f_distribution == F_DISTRIBUTION_ISING:
            # Generate 0/1 then map to -1/1
            bits = torch.randint(0, 2, (C_max, M), generator=gen, device=device, dtype=torch.int8)
            F_super[s] = bits * 2 - 1
        elif f_distribution == F_DISTRIBUTION_GAUSSIAN:
            F_super[s].normal_(0, 1, generator=gen)
            
    return F_super


def compute_Y_super_general(
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    supergraph: SuperGraphDataGeneral,
    F_super: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Y values for general graph (allowing W-W, W-X, X-X connections).
    
    Y[s, c] = (1/√M) Σ_μ F[s,c,μ] V[a,μ] V[b,μ]
    
    where V is the unified vector set (W and X).

    Args:
        W_teacher: (N1, M) teacher W matrix
        X_teacher: (M, N2) teacher X matrix
        supergraph: SuperGraphDataGeneral with unified indices and edge types
        F_super: (S, C_max, M) spreading coefficients
        
    Returns:
        Y_super: (S, C_max) Y values
    """
    S, C_max, M = F_super.shape
    alpha_scale = 1.0 / math.sqrt(M)
    N1 = W_teacher.shape[0]

    Y_super = torch.empty(S, C_max, device=F_super.device, dtype=torch.float32)

    for s in range(S):
        a_idx = supergraph.a_idx[s].long()
        b_idx = supergraph.b_idx[s].long()
        edge_type = supergraph.edge_type[s]
        
        # Gather vectors based on edge type
        # Ideally we construct V_unified = concat(W, X^T) but memory might be an issue.
        # So we gather selectively.
        
        # Prepare selectors
        is_W_W = (edge_type == EDGE_TYPE_WW)
        is_W_X = (edge_type == EDGE_TYPE_WX)
        is_X_X = (edge_type == EDGE_TYPE_XX)
        
        # Initialize V_a and V_b containers
        V_a = torch.empty(C_max, M, device=W_teacher.device)
        V_b = torch.empty(C_max, M, device=W_teacher.device)
        
        # Fill W-W edges
        if is_W_W.any():
            mask = is_W_W
            # For W-W: a < N1, b < N1. Both index into W.
            V_a[mask] = W_teacher[a_idx[mask]]
            V_b[mask] = W_teacher[b_idx[mask]]
            
        # Fill W-X edges
        if is_W_X.any():
            mask = is_W_X
            # For W-X: a < N1 (W), b >= N1 (X).
            # b_idx_corrected = b_idx - N1
            V_a[mask] = W_teacher[a_idx[mask]]
            V_b[mask] = X_teacher.T[b_idx[mask] - N1]
            
        # Fill X-X edges
        if is_X_X.any():
            mask = is_X_X
            # For X-X: Both >= N1. Both index into X.
            V_a[mask] = X_teacher.T[a_idx[mask] - N1]
            V_b[mask] = X_teacher.T[b_idx[mask] - N1]

        # Compute Y
        F_s = F_super[s].float() if F_super.dtype == torch.int8 else F_super[s]
        Y_super[s] = alpha_scale * (F_s * V_a * V_b).sum(dim=1)

    return Y_super
