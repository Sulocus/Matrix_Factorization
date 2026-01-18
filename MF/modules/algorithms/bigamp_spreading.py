"""
BiG-AMP with Random Spreading - Parallel Implementation.

This module implements BiG-AMP algorithm for the random spreading model
with Super-Graph parallelization across alpha values.

Key features:
1. Configurable F distribution: gaussian or rademacher
2. Super-Graph strategy: parallel processing of all alphas
3. Teacher type controlled by config.teacher_key (reuses existing system)

Physical model:
    Y_ij = (1/√M) Σ_μ F_ij,μ W_iμ X_μj

where F is quenched random disorder that breaks loop correlations.
"""

from typing import Tuple, Callable, Dict, Optional, List
import math
from pathlib import Path
import datetime
import torch

from ..registry import register_algorithm
from .base import AlgorithmBase
from ..graphs.supergraph import SuperGraphData, create_supergraph
from ..graphs.supergraph_general import SuperGraphDataGeneral, create_supergraph_general, EDGE_TYPE_WW, EDGE_TYPE_WX, EDGE_TYPE_XX
from ..teachers.random_spreading import SpreadingDataParallel


# ============================================================================
# Global GPU Optimizations (Phase 1)
# ============================================================================
# Enable TF32 for Tensor Core acceleration on RTX 30/40/50 (Ampere+)
# TF32 provides FP32-level precision for most workloads with ~8x throughput
# This is a global setting that affects all matmul operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




# ============================================================================
# F Generation Strategies
# ============================================================================

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


def generate_F_rademacher(
    C: int,
    M: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate F ~ Rademacher (uniform {-1, +1}).

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
    'gaussian': generate_F_gaussian,
    'rademacher': generate_F_rademacher,
}


# ============================================================================
# Super-Graph F Generation
# ============================================================================

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

    OPTIMIZATION: For Rademacher, stores as int8 (4x memory reduction).

    Args:
        supergraph: SuperGraphData with edge structure
        M: Hidden dimension
        base_seed: Base seed for F generation
        device: Target device
        f_distribution: 'gaussian' or 'rademacher'

    Returns:
        F_super: (S, C_max, M) tensor (float32 for gaussian, int8 for rademacher)
    """
    if f_distribution not in F_GENERATORS:
        raise ValueError(
            f"Invalid f_distribution='{f_distribution}'. "
            f"Available: {list(F_GENERATORS.keys())}"
        )

    generator = F_GENERATORS[f_distribution]
    S = supergraph.seeds.shape[0]
    C_max = supergraph.C_max

    # Determine dtype based on distribution
    dtype = torch.int8 if f_distribution == 'rademacher' else torch.float32
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

        # Convert F to float for computation (handles int8 Rademacher)
        F_s = F_super[s].float() if F_super.dtype == torch.int8 else F_super[s]
        
        # Y[c] = (1/√M) Σ_μ F[c,μ] W[i,μ] X[μ,j]
        Y_super[s] = alpha_scale * (F_s * W_sel * X_sel).sum(dim=1)

    return Y_super


def generate_F_super_general(
    supergraph: SuperGraphDataGeneral,
    M: int,
    base_seed: int,
    device: torch.device,
    f_distribution: str = 'rademacher',
) -> torch.Tensor:
    """
    Generate spreading coefficients F for the general super-graph.
    Similar to generate_F_super but takes SuperGraphDataGeneral.
    """
    S = supergraph.seeds.shape[0]
    C_max = supergraph.C_max
    
    # Store as int8 for memory efficiency if rademacher
    dtype = torch.int8 if f_distribution == 'rademacher' else torch.float32
    F_super = torch.empty(S, C_max, M, device=device, dtype=dtype)
    
    for s in range(S):
        # Use a unique seed for F generation distinct from graph content
        # Mix base_seed, sample index, and a magic number
        seed = base_seed + s * 777 + 100000
        gen = torch.Generator(device=device).manual_seed(seed)
        
        if f_distribution == 'rademacher':
            # Generate 0/1 then map to -1/1
            bits = torch.randint(0, 2, (C_max, M), generator=gen, device=device, dtype=torch.int8)
            F_super[s] = bits * 2 - 1
        elif f_distribution == 'gaussian':
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


# ============================================================================
# Parallel BiG-AMP Core Functions
# ============================================================================

def forward_pass_parallel(
    W_hat: torch.Tensor,
    X_hat: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    alpha_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel forward pass across all alphas.

    Z_hat[a, c] = (1/√M) Σ_μ F[c,μ] W_hat[a, i[c], μ] X_hat[a, μ, j[c]]
                  if mask[a, c] else 0

    Args:
        W_hat: (A, N1, M) student W estimates
        X_hat: (A, M, N2) student X estimates
        F: (C_max, M) spreading coefficients for this sample
        i_idx: (C_max,) row indices
        j_idx: (C_max,) column indices
        alpha_mask: (A, C_max) boolean mask

    Returns:
        Z_hat: (A, C_max) predicted Y values
    """
    A, N1, M = W_hat.shape
    C_max = F.shape[0]
    alpha_scale = 1.0 / math.sqrt(M)

    # Ensure indices are long
    i_idx = i_idx.long()
    j_idx = j_idx.long()

    # Gather: select W and X at edge positions
    # W_sel[a, c, μ] = W_hat[a, i_idx[c], μ]
    W_sel = W_hat[:, i_idx, :]  # (A, C_max, M)

    # X_sel[a, c, μ] = X_hat[a, μ, j_idx[c]]
    X_sel = X_hat[:, :, j_idx].transpose(1, 2)  # (A, C_max, M)

    # F is (C_max, M), broadcast to (1, C_max, M)
    F_expanded = F.unsqueeze(0)

    # Element-wise multiply and sum
    Z_raw = alpha_scale * (F_expanded * W_sel * X_sel).sum(dim=2)  # (A, C_max)

    # Apply mask
    Z_hat = Z_raw * alpha_mask.float()

    return Z_hat


def compute_variance_parallel(
    W_hat: torch.Tensor,
    X_hat: torch.Tensor,
    W_var: torch.Tensor,
    X_var: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    alpha_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute prediction variance at each edge.

    Uses E[F²] = 1 approximation (exact for both Gaussian and Rademacher).

    V[a, c] = (1/M) Σ_μ (W_var[a,i,μ] X²[a,μ,j] + W²[a,i,μ] X_var[a,μ,j])

    Args:
        W_hat, X_hat: (A, N, M) mean estimates
        W_var, X_var: (A, N, M) variance estimates
        F: (C_max, M) spreading coefficients
        i_idx, j_idx: (C_max,) edge indices
        alpha_mask: (A, C_max) mask

    Returns:
        V: (A, C_max) variance at each edge
    """
    A = W_hat.shape[0]
    M = W_hat.shape[2]
    alpha_scale_sq = 1.0 / M

    # Gather values
    W_sel = W_hat[:, i_idx, :]       # (A, C_max, M)
    X_sel = X_hat[:, :, j_idx].transpose(1, 2)  # (A, C_max, M)
    W_var_sel = W_var[:, i_idx, :]   # (A, C_max, M)
    X_var_sel = X_var[:, :, j_idx].transpose(1, 2)

    # F² - use actual F² values (critical for Gaussian spreading)
    F_sq = F.pow(2).unsqueeze(0)  # (1, C_max, M)
    
    # V = (1/M) Σ_μ F² * (W_var * X² + W² * X_var)
    V_raw = alpha_scale_sq * (
        F_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)
    ).sum(dim=2)  # (A, C_max)

    # Apply mask and add small epsilon for stability
    V = V_raw * alpha_mask.float() + 1e-10

    return V


def scatter_add_parallel(
    src: torch.Tensor,
    idx: torch.Tensor,
    target_size: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel scatter_add with masking.

    result[a, n, μ] = Σ_{c: idx[c]=n, mask[a,c]=1} src[a, c, μ]

    Args:
        src: (A, C_max, M) source values
        idx: (C_max,) target indices
        target_size: N (output dimension)
        mask: (A, C_max) boolean mask

    Returns:
        result: (A, N, M)
    """
    A, C_max, M = src.shape
    result = torch.zeros(A, target_size, M, device=src.device, dtype=src.dtype)

    # Apply mask
    src_masked = src * mask.unsqueeze(2).float()

    # Expand indices for scatter: (1, C_max, 1) -> (A, C_max, M)
    # Convert to int64 for scatter_add_ (required by PyTorch)
    idx_expanded = idx.long().view(1, C_max, 1).expand(A, C_max, M)

    # Scatter reduce (Phase 1 optimization)
    result.scatter_reduce_(1, idx_expanded, src_masked.contiguous(), reduce="sum", include_self=True)

    return result


def bigamp_spreading_step(
    W_hat: torch.Tensor,
    X_hat: torch.Tensor,
    W_var: torch.Tensor,
    X_var: torch.Tensor,
    Y_values: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    alpha_mask: torch.Tensor,
    damping: float,
    noise_var: float,
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single BiG-AMP step with parallel alpha processing.

    Args:
        W_hat: (A, N1, M) W mean estimates
        X_hat: (A, M, N2) X mean estimates
        W_var: (A, N1, M) W variance estimates
        X_var: (A, M, N2) X variance estimates
        Y_values: (C_max,) teacher Y values (shared across alphas)
        F: (C_max, M) spreading coefficients
        i_idx: (C_max,) row indices
        j_idx: (C_max,) column indices
        alpha_mask: (A, C_max) active edge mask
        damping: Damping factor
        noise_var: Observation noise variance
        prev_s: Previous s values for Onsager correction

    Returns:
        Updated (W_hat, X_hat, W_var, X_var, s_values)
    """
    A, N1, M = W_hat.shape
    _, _, N2 = X_hat.shape
    C_max = F.shape[0]
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # ===== Forward pass: compute predictions =====
    Z_hat = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)  # (A, C_max)

    # ===== Compute variance =====
    V = compute_variance_parallel(
        W_hat, X_hat, W_var, X_var, F, i_idx, j_idx, alpha_mask
    )  # (A, C_max)

    # ===== Compute residuals and beliefs =====
    # s = (Y - Z_hat) / (V + noise_var)
    Y_broadcast = Y_values.unsqueeze(0)  # (1, C_max)
    # Ensure denominator has minimum value for numerical stability
    denominator = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_broadcast - Z_hat) / denominator  # (A, C_max)

    # Clamp s_values to prevent explosion (critical for numerical stability)
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)

    # Apply mask
    s_values = s_values * alpha_mask.float()

    # ===== Onsager correction / Damping =====
    # REMOVED: s-damping (inconsistent with reference Wang/bigamp/train.py)
    # if prev_s is not None:
    #     s_values = damping * s_values + (1 - damping) * prev_s

    # ===== Update W =====
    # r_W[a,i,μ] = Σ_{c: i_idx[c]=i} F[c,μ] * X[a,μ,j_idx[c]] * s[a,c]
    X_sel = X_hat[:, :, j_idx].transpose(1, 2)  # (A, C_max, M)
    F_expanded = F.unsqueeze(0)  # (1, C_max, M)
    s_expanded = s_values.unsqueeze(2)  # (A, C_max, 1)

    r_W_contrib = alpha_scale * F_expanded * X_sel * s_expanded  # (A, C_max, M)
    r_W = scatter_add_parallel(r_W_contrib, i_idx, N1, alpha_mask)  # (A, N1, M)

    # tau_W = Σ_c (F²/V) * X²
    inv_V = (1.0 / denominator).unsqueeze(2)  # (A, C_max, 1)
    F_sq_expanded = F_expanded.pow(2)  # (1, C_max, M) - F² for correct variance weighting
    tau_W_contrib = alpha_scale_sq * F_sq_expanded * X_sel.pow(2) * inv_V  # (A, C_max, M)
    tau_W = scatter_add_parallel(tau_W_contrib, i_idx, N1, alpha_mask)  # (A, N1, M)
    tau_W = tau_W.clamp(min=1e-10)

    # W update with prior N(0, 1)
    W_var_new = 1.0 / (M + tau_W)  # CRITICAL FIX: M in denominator for numerical stability
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)  # Clamp r_W to prevent explosion
    W_hat_new = W_hat + W_var_new * r_W  # CRITICAL FIX: incremental update (was missing + W_hat)

    # ===== Update X =====
    # Similar logic for X
    W_sel = W_hat[:, i_idx, :]  # (A, C_max, M)

    r_X_contrib = alpha_scale * F_expanded * W_sel * s_expanded  # (A, C_max, M)
    # Need to transpose for X: aggregate by j_idx
    r_X_contrib_T = r_X_contrib.transpose(1, 2).contiguous()  # (A, M, C_max)

    # Scatter to (A, M, N2)
    r_X = torch.zeros(A, M, N2, device=W_hat.device, dtype=W_hat.dtype)
    j_idx_expanded = j_idx.long().view(1, 1, C_max).expand(A, M, C_max)
    mask_expanded_X = alpha_mask.unsqueeze(1).float()  # (A, 1, C_max)
    r_X.scatter_reduce_(2, j_idx_expanded, (r_X_contrib_T * mask_expanded_X).contiguous(), reduce="sum", include_self=True)

    tau_X_contrib = alpha_scale_sq * F_sq_expanded * W_sel.pow(2) * inv_V  # (A, C_max, M) - F² added
    tau_X_contrib_T = tau_X_contrib.transpose(1, 2).contiguous()  # (A, M, C_max)
    tau_X = torch.zeros(A, M, N2, device=W_hat.device, dtype=W_hat.dtype)
    tau_X.scatter_reduce_(2, j_idx_expanded, (tau_X_contrib_T * mask_expanded_X).contiguous(), reduce="sum", include_self=True)
    tau_X = tau_X.clamp(min=1e-10)

    X_var_new = 1.0 / (M + tau_X)  # CRITICAL FIX: M in denominator for numerical stability
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)  # Clamp r_X to prevent explosion
    X_hat_new = X_hat + X_var_new * r_X  # CRITICAL FIX: incremental update (was missing + X_hat)

    # ===== Damping =====
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var, min=1e-4, max=1.0)

    # Replace NaN with zero for numerical stability
    W_hat_out = torch.nan_to_num(W_hat_out, nan=0.0)
    X_hat_out = torch.nan_to_num(X_hat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)

    return W_hat_out, X_hat_out, W_var_out, X_var_out, s_values


# ============================================================================
# Disjoint Union Parallelization (All Samples Parallel)
# ============================================================================

def bigamp_step_disjoint_union(
    W_hat: torch.Tensor,      # (S, A, N1, M)
    X_hat: torch.Tensor,      # (S, A, M, N2)
    W_var: torch.Tensor,      # (S, A, N1, M)
    X_var: torch.Tensor,      # (S, A, M, N2)
    Y_super: torch.Tensor,    # (S, C_max)
    F_super: torch.Tensor,    # (S, C_max, M)
    i_offset: torch.Tensor,   # (S*C_max,) - precomputed offset indices
    j_offset: torch.Tensor,   # (S*C_max,) - precomputed offset indices
    alpha_mask: torch.Tensor, # (A, C_max)
    S: int,
    damping: float,
    noise_var: float,
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BiG-AMP step with Disjoint Union parallelization.

    All S samples are processed in parallel by:
    1. Flattening sample dimension into node indices (index offsetting)
    2. One large scatter_add for all S*C_max edges
    3. Reshape back to (S, A, N, M)

    This achieves true GPU parallelization across all samples.

    Args:
        W_hat: (S, A, N1, M) W estimates for all samples and alphas
        X_hat: (S, A, M, N2) X estimates
        W_var: (S, A, N1, M) W variance
        X_var: (S, A, M, N2) X variance
        Y_super: (S, C_max) Y values for all samples
        F_super: (S, C_max, M) F coefficients for all samples
        i_offset: (S*C_max,) row indices with sample offset (precomputed)
        j_offset: (S*C_max,) col indices with sample offset (precomputed)
        alpha_mask: (A, C_max) which edges are active for each alpha
        S: number of samples
        damping: damping factor
        noise_var: noise variance
        prev_s: previous s values for Onsager

    Returns:
        Updated (W_hat, X_hat, W_var, X_var, s_values)
    """
    _, A, N1, M = W_hat.shape
    N2 = X_hat.shape[3]
    C_max = F_super.shape[1]
    SC = S * C_max

    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # ===== 1. Flatten tensors for Disjoint Union =====
    # (S, A, N1, M) -> (A, S*N1, M)
    W_flat = W_hat.permute(1, 0, 2, 3).reshape(A, S * N1, M)
    X_flat = X_hat.permute(1, 0, 3, 2).reshape(A, S * N2, M)  # Note: (S,A,M,N2) -> (A,S*N2,M)
    W_var_flat = W_var.permute(1, 0, 2, 3).reshape(A, S * N1, M)
    X_var_flat = X_var.permute(1, 0, 3, 2).reshape(A, S * N2, M)

    # (S, C_max, M) -> (S*C_max, M)
    F_flat = F_super.reshape(SC, M)
    # (S, C_max) -> (S*C_max,)
    Y_flat = Y_super.reshape(SC)

    # alpha_mask: (A, C_max) -> (A, S*C_max) by repeating for each sample
    alpha_mask_exp = alpha_mask.unsqueeze(1).expand(A, S, C_max).reshape(A, SC)

    # ===== 2. Gather: one operation for all S*C_max edges =====
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    W_var_sel = W_var_flat[:, i_offset, :]  # (A, SC, M)
    X_var_sel = X_var_flat[:, j_offset, :]  # (A, SC, M)

    # ===== 3. Forward pass =====
    # Z_hat[a, sc] = (1/√M) Σ_μ F[sc,μ] W_sel[a,sc,μ] X_sel[a,sc,μ]
    Z_hat = alpha_scale * (F_flat.unsqueeze(0) * W_sel * X_sel).sum(dim=2)  # (A, SC)
    Z_hat = Z_hat * alpha_mask_exp.float()

    # ===== 4. Variance =====
    F_sq_flat = F_flat.pow(2).unsqueeze(0)  # (1, SC, M)
    # V = (1/M) Σ F² (...)
    V = alpha_scale_sq * (F_sq_flat * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V * alpha_mask_exp.float() + 1e-10

    # ===== 5. Residuals =====
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom  # (A, SC)
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    s_values = s_values * alpha_mask_exp.float()

    s_values = s_values * alpha_mask_exp.float()

    # Onsager correction
    # REMOVED: s-damping
    # if prev_s is not None:
    #     s_values = damping * s_values + (1 - damping) * prev_s

    # ===== 6. Scatter: one operation for all edges =====
    s_exp = s_values.unsqueeze(2)  # (A, SC, 1)
    mask_exp = alpha_mask_exp.unsqueeze(2).float()  # (A, SC, 1)
    F_exp = F_flat.unsqueeze(0)  # (1, SC, M)

    # W update
    r_W_contrib = alpha_scale * F_exp * X_sel * s_exp * mask_exp  # (A, SC, M)
    r_W = torch.zeros(A, S * N1, M, device=W_hat.device, dtype=W_hat.dtype)
    idx_W = i_offset.view(1, SC, 1).expand(A, SC, M)
    r_W.scatter_reduce_(1, idx_W, r_W_contrib, reduce="sum", include_self=True)

    inv_V = (1.0 / denom).unsqueeze(2)  # (A, SC, 1)
    F_sq_exp = F_exp.pow(2)  # (1, SC, M) - F² for correct variance weighting
    tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V * mask_exp
    tau_W = torch.zeros(A, S * N1, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_W.scatter_reduce_(1, idx_W, tau_W_contrib, reduce="sum", include_self=True)
    tau_W = tau_W.clamp(min=1e-10)

    W_var_new = 1.0 / (M + tau_W)  # CRITICAL FIX: M in denominator
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_flat + W_var_new * r_W  # CRITICAL FIX: incremental update (was missing + W_flat)

    # X update
    r_X_contrib = alpha_scale * F_exp * W_sel * s_exp * mask_exp  # (A, SC, M)
    r_X = torch.zeros(A, S * N2, M, device=W_hat.device, dtype=W_hat.dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_reduce_(1, idx_X, r_X_contrib, reduce="sum", include_self=True)

    tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V * mask_exp  # F² added
    tau_X = torch.zeros(A, S * N2, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_X.scatter_reduce_(1, idx_X, tau_X_contrib, reduce="sum", include_self=True)
    tau_X = tau_X.clamp(min=1e-10)

    X_var_new = 1.0 / (M + tau_X)  # CRITICAL FIX: M in denominator
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_flat + X_var_new * r_X  # CRITICAL FIX: incremental update (was missing + X_flat)

    # ===== 7. Reshape back: (A, S*N, M) -> (S, A, N, M) =====
    W_hat_new = W_hat_new.reshape(A, S, N1, M).permute(1, 0, 2, 3)
    W_var_new = W_var_new.reshape(A, S, N1, M).permute(1, 0, 2, 3)
    X_hat_new = X_hat_new.reshape(A, S, N2, M).permute(1, 0, 3, 2)  # -> (S, A, M, N2)
    X_var_new = X_var_new.reshape(A, S, N2, M).permute(1, 0, 3, 2)

    # ===== 8. Damping =====
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var, min=1e-4, max=1.0)

    # NaN protection
    W_hat_out = torch.nan_to_num(W_hat_out, nan=0.0)
    X_hat_out = torch.nan_to_num(X_hat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)

    return W_hat_out, X_hat_out, W_var_out, X_var_out, s_values


def compute_log_likelihood(Y_flat, Z_hat, V, noise_var):
    """
    Compute Log-Likelihood for each batch element (alpha).
    Y_flat: (SC,) - flattened observations (shared across batch in current implementation logic)
            Wait, Y_flat is (SC,) but Z_hat is (B, SC).
            We broadcast Y_flat to (B, SC) for per-batch calculation.
    Z_hat: (B, SC) - predicted means
    V: (B, SC) - predicted variances
    noise_var: scalar
    
    Returns:
        log_likelihood: (B,) tensor
    """
    # Y_flat (SC,) -> unsqueeze -> (1, SC) broadcasts to (B, SC)
    residual = Y_flat.unsqueeze(0) - Z_hat
    
    # Gaussian Log-Likelihood: -0.5 * log(2*pi*var) - 0.5 * (y-z)^2 / var
    # var = V + noise_var
    var = V + noise_var
    log_term = -0.5 * torch.log(2 * torch.pi * var)
    exp_term = -0.5 * (residual ** 2) / var
    
    # Sum over SC dimension (dim=1), keep Batch dimension (dim=0)
    return (log_term + exp_term).sum(dim=1)


def bigamp_step_disjoint_union_flat_adaptive(
    W_flat: torch.Tensor,      # (A, S*N1, M)
    X_flat: torch.Tensor,      # (A, S*N2, M)
    W_var_flat: torch.Tensor,  # (A, S*N1, M)
    X_var_flat: torch.Tensor,  # (A, S*N2, M)
    Y_flat: torch.Tensor,      # (S*C_max,)
    F_flat: torch.Tensor,      # (S*C_max, M)
    i_offset: torch.Tensor,    # (S*C_max,)
    j_offset: torch.Tensor,    # (S*C_max,)
    alpha_mask_exp: torch.Tensor,  # (A, S*C_max)
    S: int,
    N1: int,
    N2: int,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adaptive Step Function: Returns RAW updates + Z_hat + V for cost check.
    Does NOT apply damping.
    
    Returns:
        W_hat_new, X_hat_new, W_var_new, X_var_new, s_values, Z_hat, V
    """
    A = W_flat.shape[0]
    M = W_flat.shape[2]
    SC = F_flat.shape[0]
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Pre-cast mask
    mask_typed = alpha_mask_exp.to(W_flat.dtype)
    
    # ===== 1. Gather =====
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    W_var_sel = W_var_flat[:, i_offset, :]
    X_var_sel = X_var_flat[:, j_offset, :]
    
    # ===== 2. Forward pass =====
    F_compute = F_flat.to(W_flat.dtype)
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    
    Z_hat = alpha_scale * (F_exp * W_sel * X_sel).sum(dim=2)  # (A, SC)
    Z_hat = Z_hat * mask_typed
    
    # ===== 3. Variance =====
    if is_rademacher:
        V = alpha_scale_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel).sum(dim=2)
        F_sq_exp = None
    else:
        F_sq_exp = F_exp.pow(2)
        V = alpha_scale_sq * (F_sq_exp * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V * mask_typed + 1e-10

    # ===== 4. ONSAGER CORRECTION (Restored) =====
    # Save Z_raw (physical prediction) for Likelihood calculation before Onsager modification
    Z_raw = Z_hat.clone()  # Physical prediction: F*W*X
    if prev_s is not None:
        # Standard AMP Onsager: Cavity Z = Z_raw - V * prev_s
        # Cavity Z is used for computing residuals (s), not for Likelihood
        Z_hat = Z_hat - V * prev_s

    # ===== 5. Residuals =====
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    s_values = s_values * alpha_mask_exp.float()
    
    # ===== 6. Scatter =====
    s_exp = s_values.unsqueeze(2)
    inv_V = (1.0 / denom).unsqueeze(2)
    storage_dtype = W_flat.dtype
    mask_typed = mask_typed.unsqueeze(2)
    
    s_typed = s_exp.to(storage_dtype)
    inv_V_typed = inv_V.to(storage_dtype)
    
    # W update
    r_W_contrib = alpha_scale * F_exp * X_sel * s_typed * mask_typed
    r_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=storage_dtype)
    idx_W = i_offset.view(1, SC, 1).expand(A, SC, M)
    r_W.scatter_add_(1, idx_W, r_W_contrib)
    
    if is_rademacher:
        tau_W_contrib = alpha_scale_sq * X_sel.pow(2) * inv_V_typed * mask_typed
    else:
        tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V_typed * mask_typed
    
    tau_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=storage_dtype)
    tau_W.scatter_add_(1, idx_W, tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)
    
    W_var_new = 1.0 / (M + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_flat + W_var_new * r_W
    
    # X update
    r_X_contrib = alpha_scale * F_exp * W_sel * s_typed * mask_typed
    r_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=storage_dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_add_(1, idx_X, r_X_contrib)
    
    if is_rademacher:
        tau_X_contrib = alpha_scale_sq * W_sel.pow(2) * inv_V_typed * mask_typed
    else:
        tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V_typed * mask_typed
    
    tau_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=storage_dtype)
    tau_X.scatter_add_(1, idx_X, tau_X_contrib)
    tau_X = tau_X.clamp(min=1e-10)
    
    X_var_new = 1.0 / (M + tau_X)
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_flat + X_var_new * r_X
    
    return W_hat_new, X_hat_new, W_var_new, X_var_new, s_values, Z_raw, V


def bigamp_step_disjoint_union_flat(
    W_flat: torch.Tensor,      # (A, S*N1, M) - already flattened
    X_flat: torch.Tensor,      # (A, S*N2, M) - already flattened
    W_var_flat: torch.Tensor,  # (A, S*N1, M)
    X_var_flat: torch.Tensor,  # (A, S*N2, M)
    Y_flat: torch.Tensor,      # (S*C_max,) - flattened Y
    F_flat: torch.Tensor,      # (S*C_max, M) - flattened F
    i_offset: torch.Tensor,    # (S*C_max,) - precomputed offset indices
    j_offset: torch.Tensor,    # (S*C_max,) - precomputed offset indices
    alpha_mask_exp: torch.Tensor,  # (A, S*C_max) - expanded mask
    S: int,
    N1: int,
    N2: int,
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,  # Optimization: skip F² for Rademacher
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized BiG-AMP step operating on flat tensors.
    
    Key optimization: NO reshape/permute inside this function.
    All tensors remain in flat (A, S*N, M) format throughout.
    
    Args:
        W_flat: (A, S*N1, M) W estimates, already flattened
        X_flat: (A, S*N2, M) X estimates, already flattened  
        W_var_flat: (A, S*N1, M) W variance, flattened
        X_var_flat: (A, S*N2, M) X variance, flattened
        Y_flat: (S*C_max,) Y values, flattened
        F_flat: (S*C_max, M) F coefficients, flattened
        i_offset: (S*C_max,) row indices with sample offset
        j_offset: (S*C_max,) col indices with sample offset
        alpha_mask_exp: (A, S*C_max) expanded alpha mask
        S: number of samples
        N1, N2: original dimensions
        damping: damping factor
        noise_var: noise variance
        is_rademacher: if True, skip F² computation (F²=1)
        prev_s: previous s values (unused, kept for API compatibility)
    
    Returns:
        Updated (W_flat, X_flat, W_var_flat, X_var_flat, s_values)
        All in flat format (A, S*N, M)
    """
    A = W_flat.shape[0]
    M = W_flat.shape[2]
    SC = F_flat.shape[0]
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # Pre-cast mask once for reuse (avoid multiple casts in function)
    mask_typed = alpha_mask_exp.to(W_flat.dtype)
    
    # ===== 1. Gather: one operation for all S*C_max edges =====
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    W_var_sel = W_var_flat[:, i_offset, :]  # (A, SC, M)
    X_var_sel = X_var_flat[:, j_offset, :]  # (A, SC, M)
    
    # ===== 2. Forward pass =====
    # ALWAYS convert F to compute dtype to ensure scatter_add_ dtype consistency
    # (needed for both int8 Rademacher and float32 Gaussian when using bf16 storage)
    F_compute = F_flat.to(W_flat.dtype)
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    # Direct BF16 computation (RTX 5090 native support, 2.7x faster than .float())
    Z_hat = alpha_scale * (F_exp * W_sel * X_sel).sum(dim=2)  # (A, SC)
    Z_hat = Z_hat * mask_typed
    
    # ===== 3. Variance =====
    if is_rademacher:
        # F² = 1 for Rademacher, skip pow(2) computation
        V = alpha_scale_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel).sum(dim=2)
        F_sq_exp = None  # Not needed
    else:
        F_sq_exp = F_exp.pow(2)  # (1, SC, M)
        V = alpha_scale_sq * (F_sq_exp * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V * mask_typed + 1e-10

    # ===== 4. ONSAGER CORRECTION (Non-Adaptive Path) =====
    if prev_s is not None:
        # Standard AMP Onsager: Z_hat = Z_hat - V * prev_s
        Z_hat = Z_hat - V * prev_s
    
    # ===== 5. Residuals =====
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom  # (A, SC)
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    s_values = s_values * alpha_mask_exp.float()
    
    # ===== 5. Scatter: one operation for all edges =====
    s_exp = s_values.unsqueeze(2)  # (A, SC, 1)
    inv_V = (1.0 / denom).unsqueeze(2)  # (A, SC, 1)
    
    # Direct BF16 computation (no .float() conversion - 2.7x faster)
    storage_dtype = W_flat.dtype
    # Reuse pre-casted mask (A, SC) -> (A, SC, 1)
    mask_typed = mask_typed.unsqueeze(2)
    
    s_typed = s_exp.to(storage_dtype)
    inv_V_typed = inv_V.to(storage_dtype)
    
    # W update
    r_W_contrib = alpha_scale * F_exp * X_sel * s_typed * mask_typed  # (A, SC, M)
    r_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=storage_dtype)
    idx_W = i_offset.view(1, SC, 1).expand(A, SC, M)
    r_W.scatter_add_(1, idx_W, r_W_contrib)
    
    if is_rademacher:
        tau_W_contrib = alpha_scale_sq * X_sel.pow(2) * inv_V_typed * mask_typed
    else:
        tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V_typed * mask_typed
    tau_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=storage_dtype)
    tau_W.scatter_add_(1, idx_W, tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)
    
    W_var_new = 1.0 / (M + tau_W)  # CRITICAL FIX: M in denominator
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_flat + W_var_new * r_W
    
    # X update
    r_X_contrib = alpha_scale * F_exp * W_sel * s_typed * mask_typed  # (A, SC, M)
    r_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=storage_dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_add_(1, idx_X, r_X_contrib)
    
    if is_rademacher:
        tau_X_contrib = alpha_scale_sq * W_sel.pow(2) * inv_V_typed * mask_typed
    else:
        tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V_typed * mask_typed
    tau_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=storage_dtype)
    tau_X.scatter_add_(1, idx_X, tau_X_contrib)
    tau_X = tau_X.clamp(min=1e-10)
    
    X_var_new = 1.0 / (M + tau_X)  # CRITICAL FIX: M in denominator
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_flat + X_var_new * r_X
    
    # ===== 6. Damping =====
    W_flat_out = damping * W_hat_new + (1 - damping) * W_flat
    X_flat_out = damping * X_hat_new + (1 - damping) * X_flat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var_flat, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var_flat, min=1e-4, max=1.0)
    
    # NaN protection
    W_flat_out = torch.nan_to_num(W_flat_out, nan=0.0)
    X_flat_out = torch.nan_to_num(X_flat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)
    
    return W_flat_out, X_flat_out, W_var_out, X_var_out, s_values


# ============================================================================
# General Graph: Fused Vectorized Chunking (Memory Optimized)
# ============================================================================

def general_edge_kernel_chunk(
    V_a: torch.Tensor,       # (B, ChunkSize, M)
    V_b: torch.Tensor,       # (B, ChunkSize, M)
    V_a_var: torch.Tensor,   # (B, ChunkSize, M)
    V_b_var: torch.Tensor,   # (B, ChunkSize, M)
    F_chunk: torch.Tensor,   # (ChunkSize, M)
    Y_chunk: torch.Tensor,   # (ChunkSize,)
    alpha_mask_chunk: torch.Tensor,  # (B, ChunkSize)
    prev_s_chunk: Optional[torch.Tensor],  # (B, ChunkSize) or None
    alpha_scale: float,
    alpha_scale_sq: float,
    noise_var: float,
    is_rademacher: bool,
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused computational kernel for a single chunk of edges.
    Designed to be compiled by torch.compile for kernel fusion.
    
    Returns:
        s_chunk: (B, ChunkSize) - residual signals
        r_a_contrib: (B, ChunkSize, M) - message to node a
        r_b_contrib: (B, ChunkSize, M) - message to node b
        tau_a_contrib: (B, ChunkSize, M) - precision to node a
        tau_b_contrib: (B, ChunkSize, M) - precision to node b
        Z_hat: (B, ChunkSize) - predictions (for diagnostics)
    """
    # 1. Prepare Constants
    F_exp = F_chunk.to(compute_dtype).unsqueeze(0)  # (1, ChunkSize, M)
    mask_typed = alpha_mask_chunk.to(compute_dtype)  # (B, ChunkSize)
    
    # 2. Forward Pass (Z_hat)
    Z_hat = alpha_scale * (F_exp * V_a * V_b).sum(dim=2) * mask_typed
    
    # 3. Variance Calculation
    if is_rademacher:
        # F^2 = 1 for Rademacher
        V_val_raw = (V_a_var * V_b.pow(2) + V_a.pow(2) * V_b_var).sum(dim=2)
        F_sq_exp = None
    else:
        F_sq_exp = F_exp.pow(2)
        V_val_raw = (F_sq_exp * (V_a_var * V_b.pow(2) + V_a.pow(2) * V_b_var)).sum(dim=2)
        
    V_val = alpha_scale_sq * V_val_raw * mask_typed + 1e-10

    # 4. Onsager Correction
    if prev_s_chunk is not None:
        Z_hat = Z_hat - V_val * prev_s_chunk
    
    # 5. Residuals
    denom = V_val + noise_var
    denom = torch.clamp(denom, min=1e-6)
    s_chunk = (Y_chunk.unsqueeze(0) - Z_hat) / denom
    s_chunk = torch.clamp(s_chunk, min=-1e6, max=1e6) * alpha_mask_chunk.float()

    # 6. Compute Messages
    s_exp = s_chunk.to(compute_dtype).unsqueeze(2)     # (B, ChunkSize, 1)
    inv_V = (1.0 / denom).to(compute_dtype).unsqueeze(2)  # (B, ChunkSize, 1)
    mask_3d = mask_typed.unsqueeze(2)                   # (B, ChunkSize, 1)
    
    # Message to Node A: r_a = F * V_b * s
    r_a_contrib = alpha_scale * F_exp * V_b * s_exp * mask_3d
    
    # Message to Node B: r_b = F * V_a * s
    r_b_contrib = alpha_scale * F_exp * V_a * s_exp * mask_3d

    # Preconditioners (Tau)
    if is_rademacher:
        tau_common = alpha_scale_sq * inv_V * mask_3d
        tau_a_contrib = V_b.pow(2) * tau_common
        tau_b_contrib = V_a.pow(2) * tau_common
    else:
        tau_common = alpha_scale_sq * F_sq_exp * inv_V * mask_3d
        tau_a_contrib = V_b.pow(2) * tau_common
        tau_b_contrib = V_a.pow(2) * tau_common

    return s_chunk, r_a_contrib, r_b_contrib, tau_a_contrib, tau_b_contrib, Z_hat


# Compiled version will be cached at class level
_compiled_general_kernel = None


def bigamp_step_general_chunked(
    V_flat: torch.Tensor,      # (B, S*N_total, M)
    V_var_flat: torch.Tensor,  # (B, S*N_total, M)
    Y_flat: torch.Tensor,      # (S*C_max,)
    F_flat: torch.Tensor,      # (S*C_max, M)
    a_offset: torch.Tensor,    # (S*C_max,)
    b_offset: torch.Tensor,    # (S*C_max,)
    alpha_mask_exp: torch.Tensor,  # (B, S*C_max)
    S: int,
    N_total: int,
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
    chunk_size: int = 131072,
    use_compile: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-optimized General BiG-AMP step using chunked streaming.
    
    Key features:
    - Processes edges in fixed-size chunks to bound peak memory
    - Uses index_add_ for in-place accumulation (zero residual memory)
    - Supports torch.compile for kernel fusion within each chunk
    
    Args:
        V_flat: (B, S*N_total, M) unified node vectors
        V_var_flat: (B, S*N_total, M) unified node variances
        Y_flat: (SC,) flattened observations
        F_flat: (SC, M) flattened spreading coefficients
        a_offset: (SC,) first node indices
        b_offset: (SC,) second node indices
        alpha_mask_exp: (B, SC) alpha mask
        S: number of samples
        N_total: N1 + N2
        damping: damping factor
        noise_var: noise variance
        is_rademacher: True if F is Rademacher (F²=1)
        prev_s: (B, SC) previous s values for Onsager correction
        chunk_size: edges per chunk
        use_compile: whether to use compiled kernel
        
    Returns:
        V_flat_out, V_var_out, s_new_all, Z_hat (placeholder), V_val (placeholder)
    """
    global _compiled_general_kernel
    
    B, SN, M = V_flat.shape
    SC = F_flat.shape[0]
    compute_dtype = V_flat.dtype
    device = V_flat.device
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    # 1. Allocate Global Accumulators (Zero-initialized)
    # Size is only (B, Nodes, M), much smaller than Edges
    r_V = torch.zeros(B, SN, M, device=device, dtype=compute_dtype)
    tau_V = torch.zeros(B, SN, M, device=device, dtype=compute_dtype)
    
    # Container for s_values (needed for next step's Onsager)
    s_new_all = torch.empty(B, SC, device=device, dtype=torch.float32)
    
    # 2. Optional: Compile the kernel (cached at module level)
    kernel_fn = general_edge_kernel_chunk
    if use_compile and _compiled_general_kernel is None:
        try:
            torch._dynamo.reset()
            _compiled_general_kernel = torch.compile(
                general_edge_kernel_chunk,
                mode="default",  # Avoid CUDA Graphs issues
                fullgraph=False,
            )
        except Exception:
            _compiled_general_kernel = general_edge_kernel_chunk
    
    if use_compile and _compiled_general_kernel is not None:
        kernel_fn = _compiled_general_kernel
    
    # 3. Chunked Processing Loop
    num_chunks = (SC + chunk_size - 1) // chunk_size
    
    for i in range(0, SC, chunk_size):
        end = min(i + chunk_size, SC)
        chunk_len = end - i
        idx_slice = slice(i, end)
        
        # 3a. Gather Inputs for this chunk
        a_idx_chunk = a_offset[idx_slice]
        b_idx_chunk = b_offset[idx_slice]
        
        V_a = V_flat.index_select(1, a_idx_chunk)
        V_b = V_flat.index_select(1, b_idx_chunk)
        V_a_var = V_var_flat.index_select(1, a_idx_chunk)
        V_b_var = V_var_flat.index_select(1, b_idx_chunk)
        
        F_chunk = F_flat[idx_slice]
        Y_chunk = Y_flat[idx_slice]
        mask_chunk = alpha_mask_exp[:, idx_slice]
        prev_s_chunk = prev_s[:, idx_slice] if prev_s is not None else None

        # 3b. Execute Kernel
        s_chunk, r_a, r_b, tau_a, tau_b, _ = kernel_fn(
            V_a, V_b, V_a_var, V_b_var, F_chunk, Y_chunk, mask_chunk, prev_s_chunk,
            alpha_scale, alpha_scale_sq, noise_var, is_rademacher, compute_dtype
        )
        
        # 3c. Store s values
        s_new_all[:, idx_slice] = s_chunk
        
        # 3d. Scatter Aggregate (Use scatter_add_ with expanded 3D indices for better performance)
        # Expand indices for scatter_add_: (ChunkLen,) -> (B, ChunkLen, M)
        a_idx_exp = a_idx_chunk.view(1, -1, 1).expand(B, chunk_len, M)
        b_idx_exp = b_idx_chunk.view(1, -1, 1).expand(B, chunk_len, M)
        
        r_V.scatter_add_(1, a_idx_exp, r_a)
        r_V.scatter_add_(1, b_idx_exp, r_b)
        
        tau_V.scatter_add_(1, a_idx_exp, tau_a)
        tau_V.scatter_add_(1, b_idx_exp, tau_b)
        
        # Free expanded indices
        del a_idx_exp, b_idx_exp

    # 4. Final Node Updates
    tau_V = tau_V.clamp(min=1e-10)
    V_var_new = 1.0 / (M + tau_V)
    
    r_V = torch.clamp(r_V, min=-1e4, max=1e4)
    V_hat_new = V_flat + V_var_new * r_V
    
    # 5. Damping
    V_flat_out = damping * V_hat_new + (1 - damping) * V_flat
    V_var_out = torch.clamp(damping * V_var_new + (1 - damping) * V_var_flat, min=1e-4, max=1.0)
    
    # NaN protection
    V_flat_out = torch.nan_to_num(V_flat_out, nan=0.0)
    V_var_out = torch.nan_to_num(V_var_out, nan=1.0)
    
    # Return placeholders for Z_hat and V_val (not needed in training loop)
    return V_flat_out, V_var_out, s_new_all, None, None


def bigamp_step_disjoint_union_flat_general(
    V_flat: torch.Tensor,      # (A, S*N_total, M) - flattened unified vectors
    V_var_flat: torch.Tensor,  # (A, S*N_total, M)
    Y_flat: torch.Tensor,      # (S*C_max,)
    F_flat: torch.Tensor,      # (S*C_max, M)
    a_offset: torch.Tensor,    # (S*C_max,) - unified indices for first node
    b_offset: torch.Tensor,    # (S*C_max,) - unified indices for second node
    alpha_mask_exp: torch.Tensor,
    S: int,
    N_total: int,
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    General Graph optimized BiG-AMP step using Unified Vector V.
    
    V_flat contains both W and X nodes.
    a_offset and b_offset index into V_flat.
    """
    A = V_flat.shape[0]
    M = V_flat.shape[2]
    SC = F_flat.shape[0]
    SN = S * N_total
    storage_dtype = V_flat.dtype
    
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    mask_typed = alpha_mask_exp.to(V_flat.dtype)
    
    # ===== 1. Gather =====
    # ===== 1. Gather (Vectorized) =====
    # Use index_select for vectorized gathering
    # V_flat: (B, S*N_total, M)
    # a_offset: (S*C_max,)
    # Result: (B, SC, M)
    V_a = V_flat.index_select(1, a_offset)
    V_b = V_flat.index_select(1, b_offset)
    V_a_var = V_var_flat.index_select(1, a_offset)
    V_b_var = V_var_flat.index_select(1, b_offset)
    
    # ===== 2. Forward =====
    F_compute = F_flat.to(V_flat.dtype)
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    
    # Z_hat = (1/√M) Σ F * V_a * V_b
    Z_hat = alpha_scale * (F_exp * V_a * V_b).sum(dim=2) * mask_typed
    
    # ===== 3. Variance =====
    # MEMORY OPT: 预计算 F_sq_exp 用于后续 scatter update
    if is_rademacher:
        # F²=1
        V_val = alpha_scale_sq * (V_a_var * V_b.pow(2) + V_a.pow(2) * V_b_var).sum(dim=2)
        F_sq_exp = None  # Not needed for Rademacher
    else:
        F_sq_exp = F_exp.pow(2)  # 预计算，后续复用
        V_val = alpha_scale_sq * (F_sq_exp * (V_a_var * V_b.pow(2) + V_a.pow(2) * V_b_var)).sum(dim=2)
    
    # MEMORY OPT: 释放不再需要的 variance 张量
    del V_a_var, V_b_var
    
    V_val = V_val * mask_typed + 1e-10
    
    # ===== 4. Onsager =====
    if prev_s is not None:
        Z_hat = Z_hat - V_val * prev_s
        
    # ===== 5. Residuals =====
    denom = torch.clamp(V_val + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom
    s_values = torch.clamp(s_values, min=-1e6, max=1e6) * alpha_mask_exp.float()
    
    # ===== 6. Scatter Update =====
    s_exp = s_values.unsqueeze(2).to(V_flat.dtype)
    inv_V_typed = (1.0 / denom).unsqueeze(2).to(V_flat.dtype)
    mask_typed = mask_typed.unsqueeze(2)
    
    # ===== 6. Accumulation =====
    # 初始化累加器 (A, S*N_total, M)
    r_V = torch.zeros(A, SN, M, device=V_flat.device, dtype=storage_dtype)
    tau_V = torch.zeros(A, SN, M, device=V_flat.device, dtype=storage_dtype)
    
    # --- Phase A: 处理 'a' 节点更新 (使用 V_b) ---
    # OPTIMIZATION: 使用 index_add_ 替代 scatter_add_，避免构造 3D 扩展索引
    # PyTorch index_add_ 在 source 为 3D 时支持 1D index 广播
    term = alpha_scale * F_exp * V_b * s_exp * mask_typed
    r_V.index_add_(1, a_offset, term)
    del term
    
    if is_rademacher:
        term = alpha_scale_sq * V_b.pow(2) * inv_V_typed * mask_typed
    else:
        term = alpha_scale_sq * F_sq_exp * V_b.pow(2) * inv_V_typed * mask_typed
    tau_V.index_add_(1, a_offset, term)
    del term
    
    del V_b
    
    # --- Phase B: 处理 'b' 节点更新 (使用 V_a) ---
    term = alpha_scale * F_exp * V_a * s_exp * mask_typed
    r_V.index_add_(1, b_offset, term)
    del term
    
    if is_rademacher:
        term = alpha_scale_sq * V_a.pow(2) * inv_V_typed * mask_typed
    else:
        term = alpha_scale_sq * F_sq_exp * V_a.pow(2) * inv_V_typed * mask_typed
    tau_V.index_add_(1, b_offset, term)
    del term
    
    del V_a
    
    tau_V = tau_V.clamp(min=1e-10)
    
    # Final Update
    V_var_new = 1.0 / (M + tau_V)
    
    r_V = torch.clamp(r_V, min=-1e4, max=1e4)
    V_hat_new = V_flat + V_var_new * r_V
    
    # Damping
    V_flat_out = damping * V_hat_new + (1 - damping) * V_flat
    V_var_out = torch.clamp(damping * V_var_new + (1 - damping) * V_var_flat, min=1e-4, max=1.0)
    
    V_flat_out = torch.nan_to_num(V_flat_out, nan=0.0)
    V_var_out = torch.nan_to_num(V_var_out, nan=1.0)
    
    return V_flat_out, V_var_out, s_values, Z_hat, V_val


def compute_offset_indices(
    i_idx: torch.Tensor,  # (S, C_max)
    j_idx: torch.Tensor,  # (S, C_max)
    N1: int,
    N2: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute offset indices for Disjoint Union.

    Maps each sample's local indices to global indices:
    i_offset[s, c] = s * N1 + i_idx[s, c]
    j_offset[s, c] = s * N2 + j_idx[s, c]

    Args:
        i_idx: (S, C_max) row indices per sample
        j_idx: (S, C_max) col indices per sample
        N1: number of rows
        N2: number of columns

    Returns:
        i_offset: (S*C_max,) flattened offset row indices
        j_offset: (S*C_max,) flattened offset col indices
    """
    S = i_idx.shape[0]
    device = i_idx.device

    # Sample offsets: [0, N1, 2*N1, ...]
    offsets_N1 = torch.arange(S, device=device) * N1  # (S,)
    offsets_N2 = torch.arange(S, device=device) * N2  # (S,)

    # Add offsets and flatten
    i_offset = (i_idx + offsets_N1.unsqueeze(1)).reshape(-1)  # (S*C_max,)
    j_offset = (j_idx + offsets_N2.unsqueeze(1)).reshape(-1)  # (S*C_max,)

    return i_offset, j_offset


# ============================================================================
# Main Algorithm Class
# ============================================================================

@register_algorithm(
    key="bigamp_spreading",
    name="BiG-AMP Spreading",
    description="GPU parallel across all alphas - 30x faster for production",
    default_params={
        'damping': 0.5,
        'noise_var': 1e-10,
    },
)
class BiGAMPSpreading(AlgorithmBase):
    """
    BiG-AMP with random spreading, parallel across alpha values.

    Configurable options:
    - teacher_key: 'standard' (Gaussian) or 'orthogonal' - via config.teacher_key
    - f_distribution: 'gaussian' or 'rademacher' - via config.spreading.f_distribution

    Usage:
        config = Config(
            algorithm_key="bigamp_spreading",
            teacher_key="orthogonal",  # Controls W, X generation
            spreading=SpreadingConfig(f_distribution="rademacher"),
        )
    """

    # Class-level cache for compiled step function
    _compiled_step = None
    _compiled_step_adaptive = None
    _compiled_step_general = None

    @classmethod
    def clear_compile_cache(cls):
        """Clear compiled step function cache to release GPU memory.
        
        This is useful for OOM recovery when batch-to-batch execution
        accumulates torch.compile caches that cannot be freed by gc.collect()
        or torch.cuda.empty_cache().
        
        Call this before replanning execution after an OOM event.
        """
        cls._compiled_step = None
        cls._compiled_step_adaptive = None
        cls._compiled_step_general = None
        try:
            import torch._dynamo
            torch._dynamo.reset()  # Clear torch.compile internal caches
        except Exception:
            pass

    def __init__(self, config, device: torch.device):
        """
        Initialize parallel spreading algorithm.

        Args:
            config: Config object with algorithm parameters
            device: Target device
        """
        self.config = config
        self.device = device

        # Algorithm parameters
        self.damping = config.algorithm_params.damping
        self.noise_var = config.algorithm_params.noise_var
        self.max_steps = config.training.max_steps

        # Spreading configuration
        spreading_cfg = config.spreading
        if spreading_cfg is not None:
            self.f_distribution = spreading_cfg.f_distribution
            self.spreading_seed = spreading_cfg.seed
            self.onsager_correction = getattr(spreading_cfg, 'onsager_correction', True)
            self.allow_intra_connection = getattr(spreading_cfg, 'allow_intra_connection', False)
            # Default chunk_size to 0 (Unchunked) to utilize ParallelCoordinator's dynamic batching
            # instead of inefficient Python-level looping.
            self.chunk_size = getattr(spreading_cfg, 'chunk_size', 0) 
        else:
            # Default values
            self.f_distribution = 'gaussian'
            self.spreading_seed = 12345
            self.onsager_correction = True
            self.allow_intra_connection = False
            self.chunk_size = 0

        # Validate f_distribution
        if self.f_distribution not in F_GENERATORS:
            raise ValueError(
                f"Invalid f_distribution='{self.f_distribution}'. "
                f"Available: {list(F_GENERATORS.keys())}"
            )

        # torch.compile for kernel fusion (Phase 1 optimization - upgraded)
        # NOTE: max-autotune and reduce-overhead use CUDA Graphs which can cause issues
        # For large problems, we use 'default' mode (no CUDA Graphs, still has Triton kernels)
        self.use_compile = getattr(config.algorithm_params, 'use_compile', True)
        if self.use_compile and BiGAMPSpreading._compiled_step is None:
            # Determine if problem is "large" (needs memory-safe mode)
            N1 = config.matrix.N1
            N2 = config.matrix.N2
            M = config.matrix.M
            is_large_problem = (N1 * N2 * M > 50_000_000)  # ~50M elements
            
            if is_large_problem:
                # Large problem: use 'default' mode to avoid CUDA Graph issues
                compile_modes = ['default']
            else:
                # Normal size: try more aggressive modes first
                compile_modes = ['reduce-overhead', 'default']
            
            for mode in compile_modes:
                try:
                    BiGAMPSpreading._compiled_step = torch.compile(
                        bigamp_step_disjoint_union_flat,
                        mode=mode,
                        fullgraph=False,
                    )
                    break
                except Exception:
                    if mode == compile_modes[-1]:
                        self.use_compile = False
        
        # Also compile adaptive step function if needed
        if self.use_compile and BiGAMPSpreading._compiled_step_adaptive is None:
            try:
                BiGAMPSpreading._compiled_step_adaptive = torch.compile(
                    bigamp_step_disjoint_union_flat_adaptive,
                    mode='default',  # Use safe mode for adaptive (more intermediates)
                    fullgraph=False,
                )
            except Exception:
                pass  # Fall back to uncompiled if fails

        # Phase 3: BF16 mixed precision (auto-detect hardware support)
        self.use_bf16 = False
        self.storage_dtype = torch.float32
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.use_bf16 = True
            self.storage_dtype = torch.bfloat16

    @staticmethod
    def _initialize_near_teacher(
        target_shape: Tuple[int, int, int],  # (B, S*N, M)
        teacher_tensor: torch.Tensor,         # (N, M)
        m_init: float,
        S: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Generate initial estimate close to the teacher for Hysteresis Analysis.
        
        Formula: V_init = m_init * V_teacher + sqrt(1 - m_init^2) * noise
        
        This ensures:
        - Initial overlap ≈ m_init
        - Variance is preserved (proper normalization)
        
        Args:
            target_shape: (B, S*N, M) target flat shape
            teacher_tensor: (N, M) teacher tensor to initialize near
            m_init: Target initial overlap with teacher (0.9 - 0.99)
            S: Number of samples
            device: Target device
            dtype: Storage dtype (float32 or bfloat16)
            
        Returns:
            Tensor of shape (B, S*N, M) initialized near teacher
        """
        B, SN, M_dim = target_shape
        N = teacher_tensor.shape[0]
        
        # Broadcast: (N, M) -> (1, 1, N, M) -> (B, S, N, M) -> (B, S*N, M)
        teacher_expanded = teacher_tensor.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        teacher_flat = teacher_expanded.reshape(B, SN, M_dim).to(device, dtype=dtype)
        
        # Generate noise with same shape
        noise = torch.randn(target_shape, device=device, dtype=dtype)
        
        # Combine with variance preservation formula
        coeff_signal = m_init
        coeff_noise = math.sqrt(1 - m_init ** 2)
        
        return coeff_signal * teacher_flat + coeff_noise * noise

    def create_spreading_data(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        alpha_values: List[float],
        S: int,
        base_seed: int,
    ) -> SpreadingDataParallel:
        """
        Create SpreadingDataParallel for training.

        Args:
            W_teacher: (N1, M) teacher W matrix
            X_teacher: (M, N2) teacher X matrix
            alpha_values: List of alpha values
            S: Number of samples
            base_seed: Base random seed

        Returns:
            SpreadingDataParallel containing all data for parallel training
        """
        if getattr(self, 'allow_intra_connection', False):
            # General Graph Mode
            N1, M = W_teacher.shape
            _, N2 = X_teacher.shape

            supergraph = create_supergraph_general(
                N1=N1,
                N2=N2,
                M=M,
                alpha_values=alpha_values,
                S=S,
                base_seed=base_seed,
                device=self.device,
            )

            F_super = generate_F_super_general(
                supergraph=supergraph,
                M=M,
                base_seed=self.spreading_seed,
                device=self.device,
                f_distribution=self.f_distribution,
            )

            Y_super = compute_Y_super_general(
                W_teacher=W_teacher,
                X_teacher=X_teacher,
                supergraph=supergraph,
                F_super=F_super,
            )
        else:
            # Original Bipartite Mode
            N1, M = W_teacher.shape
            _, N2 = X_teacher.shape

            supergraph = create_supergraph(
                N1=N1,
                N2=N2,
                M=M,
                alpha_values=alpha_values,
                S=S,
                base_seed=base_seed,
                device=self.device,
            )

            F_super = generate_F_super(
                supergraph=supergraph,
                M=M,
                base_seed=self.spreading_seed,
                device=self.device,
                f_distribution=self.f_distribution,
            )

            Y_super = compute_Y_super(
                W_teacher=W_teacher,
                X_teacher=X_teacher,
                supergraph=supergraph,
                F_super=F_super,
            )

        return SpreadingDataParallel(
            supergraph=supergraph,
            F_super=F_super,
            Y_super=Y_super,
            M=M,
            alpha_values=torch.tensor(alpha_values, device=self.device),
            W_teacher=W_teacher,
            X_teacher=X_teacher,
        )

    def train_sample(
        self,
        spreading_data: SpreadingDataParallel,
        sample_idx: int,
        verbose: bool = False,
        step_callback=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train all alphas for a single sample.

        Args:
            spreading_data: SpreadingDataParallel
            sample_idx: Which sample to train
            verbose: Print progress
            step_callback: Optional callback(step, max_steps) for step-level progress

        Returns:
            W_students: (A, N1, M) trained W for all alphas
            X_students: (A, M, N2) trained X for all alphas
        """
        A = spreading_data.A
        N1 = spreading_data.supergraph.N1
        N2 = spreading_data.supergraph.N2
        M = spreading_data.M

        # Get sample-specific data
        F = spreading_data.get_F(sample_idx)  # (C_max, M)
        Y_values = spreading_data.Y_super[sample_idx]  # (C_max,)
        i_idx, j_idx = spreading_data.supergraph.get_sample_indices(sample_idx)
        # Ensure indices are long type for indexing
        i_idx = i_idx.long()
        j_idx = j_idx.long()
        alpha_mask = spreading_data.supergraph.alpha_mask  # (A, C_max)

        # Initialize student variables
        # Initialize student variables (Mean Field Scaling: N(0,1))
        # scale = 1.0 / math.sqrt(M)  # Removed for Mean Field
        W_hat = torch.randn(A, N1, M, device=self.device) * 0.1
        X_hat = torch.randn(A, M, N2, device=self.device) * 0.1
        W_var = torch.ones(A, N1, M, device=self.device)
        X_var = torch.ones(A, M, N2, device=self.device)

        prev_s = None

        # BiG-AMP iterations
        for step in range(self.max_steps):
            W_hat, X_hat, W_var, X_var, prev_s = bigamp_spreading_step(
                W_hat=W_hat,
                X_hat=X_hat,
                W_var=W_var,
                X_var=X_var,
                Y_values=Y_values,
                F=F,
                i_idx=i_idx,
                j_idx=j_idx,
                alpha_mask=alpha_mask,
                damping=self.damping,
                noise_var=self.noise_var,
                prev_s=prev_s,
            )

            if verbose and (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{self.max_steps}")

            # Step-level progress callback
            if step_callback:
                step_callback(step + 1, self.max_steps)

        return W_hat, X_hat

    def train_all_samples(
        self,
        spreading_data: SpreadingDataParallel,
        verbose: bool = True,
        step_callback=None,
        sample_callback=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train all samples (legacy sequential version).

        Args:
            spreading_data: SpreadingDataParallel
            verbose: Print progress
            step_callback: Optional callback(step, max_steps) for step-level progress
            sample_callback: Optional callback(sample, total_samples) for sample-level progress

        Returns:
            W_students: (S, A, N1, M)
            X_students: (S, A, M, N2)
        """
        S = spreading_data.S
        A = spreading_data.A
        N1 = spreading_data.supergraph.N1
        N2 = spreading_data.supergraph.N2
        M = spreading_data.M

        W_all = torch.zeros(S, A, N1, M, device=self.device)
        X_all = torch.zeros(S, A, M, N2, device=self.device)

        for s in range(S):
            if verbose:
                print(f"Training sample {s + 1}/{S}")

            # Pass step_callback to train_sample for step-level updates
            W_s, X_s = self.train_sample(spreading_data, s, verbose=False, step_callback=step_callback)
            W_all[s] = W_s
            X_all[s] = X_s

            # Update sample progress after each sample completes
            if sample_callback:
                sample_callback(s + 1, S)

        return W_all, X_all

    def train_full_parallel(
        self,
        spreading_data: SpreadingDataParallel,
        batch_alpha_indices: Optional[List[int]] = None,
        verbose: bool = False,
        step_callback=None,
        max_steps: Optional[int] = None,  # Allow override for step scanning
        batch_alpha_values: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train all samples in parallel using Disjoint Union with optimized flat tensors.

        OPTIMIZATIONS APPLIED:
        1. All tensors stored in flat format (A, S*N, M) - no per-iteration reshape
        2. Pre-flattened F, Y, alpha_mask computed once
        3. torch.compile for kernel fusion (if enabled)
        4. Rademacher F² optimization (F²=1 skips pow(2))

        Args:
            spreading_data: SpreadingDataParallel with F_super, Y_super, etc.
            batch_alpha_indices: Which alphas to train (None = all)
            verbose: Print progress
            step_callback: Optional callback(step, max_steps)
            max_steps: Optional override
            batch_alpha_values: Optional list of alpha values for debug recording

        Returns:
            W_students: (S, B, N1, M) where B = len(batch_alpha_indices) or A
            X_students: (S, B, M, N2)
        """
        S = spreading_data.S
        A = spreading_data.A
        N1 = spreading_data.supergraph.N1
        N2 = spreading_data.supergraph.N2
        M = spreading_data.M
        C_max = spreading_data.C_max
        SC = S * C_max

        # Check for General Graph Mode
        if getattr(self, 'allow_intra_connection', False):
            return self._train_full_parallel_general(
                spreading_data, batch_alpha_indices, verbose, step_callback, max_steps, batch_alpha_values
            )

        # Check for Adaptive Damping
        if hasattr(self.config.algorithm_params, 'adaptive_damping') and self.config.algorithm_params.adaptive_damping:
            return self._train_full_parallel_adaptive(
                spreading_data, batch_alpha_indices, verbose, step_callback, max_steps, batch_alpha_values
            )


        # Determine which alphas to train
        if batch_alpha_indices is None:
            batch_alpha_indices = list(range(A))
        B = len(batch_alpha_indices)

        # Get alpha mask for this batch
        full_alpha_mask = spreading_data.supergraph.alpha_mask  # (A, C_max)
        batch_alpha_mask = full_alpha_mask[batch_alpha_indices]  # (B, C_max)

        # Compute offset indices (once, reused for all steps)
        i_offset, j_offset = compute_offset_indices(
            spreading_data.supergraph.i_idx,  # (S, C_max)
            spreading_data.supergraph.j_idx,  # (S, C_max)
            N1, N2
        )

        # ===== OPTIMIZATION 1: Pre-flatten all data (once) =====
        F_flat = spreading_data.F_super.reshape(SC, M)  # (S*C_max, M)
        Y_flat = spreading_data.Y_super.reshape(SC)     # (S*C_max,)
        
        # Expand alpha mask: (B, C_max) -> (B, S*C_max)
        alpha_mask_exp = batch_alpha_mask.unsqueeze(1).expand(B, S, C_max).reshape(B, SC)

        # ===== INITIALIZATION (Teacher-Assisted or Random) =====
        init_mode = getattr(self.config.algorithm_params, 'init_mode', 'random')
        init_overlap = getattr(self.config.algorithm_params, 'init_overlap', 0.95)

        if init_mode == 'teacher':
            # Teacher-Assisted Initialization (Warm Start for Hysteresis Analysis)
            W_flat = self._initialize_near_teacher(
                (B, S * N1, M),
                spreading_data.W_teacher,  # (N1, M)
                init_overlap,
                S,
                self.device,
                self.storage_dtype
            )
            X_flat = self._initialize_near_teacher(
                (B, S * N2, M),
                spreading_data.X_teacher.T,  # (M, N2) -> (N2, M)
                init_overlap,
                S,
                self.device,
                self.storage_dtype
            )
            if verbose:
                print(f"  [Init] Teacher-Assisted (m={init_overlap})")
        else:
            # Random Initialization (Cold Start - default)
            W_flat = torch.randn(B, S * N1, M, device=self.device, dtype=self.storage_dtype) * 0.1
            X_flat = torch.randn(B, S * N2, M, device=self.device, dtype=self.storage_dtype) * 0.1

        W_var_flat = torch.ones(B, S * N1, M, device=self.device, dtype=self.storage_dtype)
        X_var_flat = torch.ones(B, S * N2, M, device=self.device, dtype=self.storage_dtype)


        prev_s = None
        is_rademacher = (self.f_distribution == 'rademacher')

        # ===== OPTIMIZATION 3: Use compiled step if available =====
        step_fn = BiGAMPSpreading._compiled_step if self.use_compile and BiGAMPSpreading._compiled_step is not None else bigamp_step_disjoint_union_flat

        # Use provided max_steps or fall back to config
        steps = max_steps if max_steps is not None else self.max_steps

        # BiG-AMP iterations with optimized flat function
        for step in range(steps):
            # CRITICAL FIX: Mark new CUDA Graph step to prevent "tensor overwritten" error
            if self.use_compile and BiGAMPSpreading._compiled_step is not None:
                torch.compiler.cudagraph_mark_step_begin()
            
            W_flat, X_flat, W_var_flat, X_var_flat, next_prev_s = step_fn(
                W_flat=W_flat,
                X_flat=X_flat,
                W_var_flat=W_var_flat,
                X_var_flat=X_var_flat,
                Y_flat=Y_flat,
                F_flat=F_flat,
                i_offset=i_offset,
                j_offset=j_offset,
                alpha_mask_exp=alpha_mask_exp,
                S=S,
                N1=N1,
                N2=N2,
                damping=self.damping,
                noise_var=self.noise_var,
                is_rademacher=is_rademacher,
                prev_s=prev_s,
            )
            
            # --- ONSAGER CONTROL FIX (Non-Adaptive) ---
            # Strictly respect config flag. If False, prev_s must be None.
            if self.onsager_correction:
                prev_s = next_prev_s
            else:
                prev_s = None
            
            # CLONE STRATEGY: Break CUDA Graph address dependency
            # When using torch.compile with reduce-overhead mode, CUDA Graphs captures
            # input tensor memory addresses during recording. The iterative pattern
            # `W_flat = step_fn(W_flat=W_flat)` causes outputs to overwrite input vars,
            # Graph thinks addresses are "polluted" and raises error:
            # "accessing tensor output of CUDAGraphs that has been overwritten"
            # Solution: Clone output tensors to allocate new memory, breaking the chain.
            if self.use_compile and BiGAMPSpreading._compiled_step is not None:
                W_flat = W_flat.clone()
                X_flat = X_flat.clone()
                W_var_flat = W_var_flat.clone()
                X_var_flat = X_var_flat.clone()
                if prev_s is not None:
                    prev_s = prev_s.clone()

            # [Memory Calibration] Check actual usage early in the run
            if (step + 1) == 10 and torch.cuda.is_available():
                 peak_bytes = torch.cuda.max_memory_allocated()
                 peak_gb = peak_bytes / (1024**3)
                 # Reset peak stats to track steady state separately if needed, but cumulative is safer
                 # print(f"[BiG-AMP Calibration] Step 10 Peak Memory: {peak_gb:.2f} GB") 
                 # We don't want to spam stdout if verbose=False, but it's important for calibration.
                 # We'll log it if verbose or if it's the first batch (we can't easily tell here).
                 # Let's just log it if verbose.
                 if verbose:
                     print(f"  [Memory Calibration] Peak VRAM: {peak_gb:.2f} GB")

            if verbose and (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{steps}")

            if step_callback:
                step_callback(step + 1, steps)

        # ===== Only reshape at the END for output =====
        # (B, S*N1, M) -> (B, S, N1, M) -> (S, B, N1, M)
        W_hat = W_flat.reshape(B, S, N1, M).permute(1, 0, 2, 3)
        # (B, S*N2, M) -> (B, S, N2, M) -> (S, B, M, N2)
        X_hat = X_flat.reshape(B, S, N2, M).permute(1, 0, 3, 2)

        return W_hat, X_hat

    def _train_full_parallel_adaptive(
        self,
        spreading_data: SpreadingDataParallel,
        batch_alpha_indices: List[int],
        verbose: bool,
        step_callback,
        max_steps: Optional[int],
        batch_alpha_values: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive Damping Training Loop with Backtracking.
        """
        S = spreading_data.S
        A = spreading_data.A
        N1 = spreading_data.supergraph.N1
        N2 = spreading_data.supergraph.N2
        M = spreading_data.M
        C_max = spreading_data.C_max
        SC = S * C_max
        
        # Handle None batch_alpha_indices
        if batch_alpha_indices is None:
            batch_alpha_indices = list(range(A))
        B = len(batch_alpha_indices)
        
        # Determine params
        params = self.config.algorithm_params
        step_min = getattr(params, 'step_min', 0.05)
        step_max = getattr(params, 'step_max', 1.0)
        step_incr = getattr(params, 'step_incr', 1.1)
        step_decr = getattr(params, 'step_decr', 0.5)
        max_bad = getattr(params, 'max_bad_steps', 10)
        
        # State: Current Damping
        # CRITICAL FIX: Damping must be per-alpha (B,) vector
        # Otherwise one diverging alpha drags everyone down
        damping = torch.full((B,), self.damping, device=self.device, dtype=self.storage_dtype)
        
        # Get masked data
        full_alpha_mask = spreading_data.supergraph.alpha_mask
        batch_alpha_mask = full_alpha_mask[batch_alpha_indices]
        i_offset, j_offset = compute_offset_indices(
            spreading_data.supergraph.i_idx,
            spreading_data.supergraph.j_idx,
            N1, N2
        )
        F_flat = spreading_data.F_super.reshape(SC, M)
        Y_flat = spreading_data.Y_super.reshape(SC)
        alpha_mask_exp = batch_alpha_mask.unsqueeze(1).expand(B, S, C_max).reshape(B, SC)
        
        # ===== INITIALIZATION (Teacher-Assisted or Random) =====
        init_mode = getattr(self.config.algorithm_params, 'init_mode', 'random')
        init_overlap = getattr(self.config.algorithm_params, 'init_overlap', 0.95)

        if init_mode == 'teacher':
            # Teacher-Assisted Initialization (Warm Start for Hysteresis Analysis)
            W_flat = self._initialize_near_teacher(
                (B, S * N1, M),
                spreading_data.W_teacher,
                init_overlap,
                S,
                self.device,
                self.storage_dtype
            )
            X_flat = self._initialize_near_teacher(
                (B, S * N2, M),
                spreading_data.X_teacher.T,
                init_overlap,
                S,
                self.device,
                self.storage_dtype
            )
            if verbose:
                print(f"  [Init] Teacher-Assisted Adaptive (m={init_overlap})")
        else:
            # Random Initialization (Cold Start - default)
            W_flat = torch.randn(B, S * N1, M, device=self.device, dtype=self.storage_dtype) * 0.1
            X_flat = torch.randn(B, S * N2, M, device=self.device, dtype=self.storage_dtype) * 0.1

        W_var_flat = torch.ones(B, S * N1, M, device=self.device, dtype=self.storage_dtype)
        X_var_flat = torch.ones(B, S * N2, M, device=self.device, dtype=self.storage_dtype)

        
        # "Safe" State (Last accepted) - ONLY clone at initialization
        # Subsequent saves will use reference swap to avoid memory explosion
        W_safe = W_flat.clone()
        X_safe = X_flat.clone()
        W_var_safe = W_var_flat.clone()
        X_var_safe = X_var_flat.clone()
        s_safe = None
        
        prev_s = None
        current_val = -float('inf')
        
        # Use compiled step function if available (like train_full_parallel L1337)
        if self.use_compile and BiGAMPSpreading._compiled_step_adaptive is not None:
            step_fn = BiGAMPSpreading._compiled_step_adaptive
        else:
            step_fn = bigamp_step_disjoint_union_flat_adaptive
        
        is_rademacher = (self.f_distribution == 'rademacher')
        steps = max_steps if max_steps is not None else self.max_steps
        
        # Warm Restart Parameters
        adaptive_restart = getattr(params, 'adaptive_restart', False)
        restart_patience = getattr(params, 'restart_patience', 50)
        restart_noise = getattr(params, 'restart_noise', 0.1)
        acceptance_tolerance = getattr(params, 'acceptance_tolerance', 0.0)
        stuck_counter = torch.zeros(B, dtype=torch.long, device=self.device)

        # Debug Recording
        damp_history = None
        if batch_alpha_values is not None:
             damp_history = torch.zeros(steps, B, dtype=torch.float32, device=self.device)

        for step in range(steps):
            # CUDA Graph compatibility mark (like train_full_parallel L1345-1346)
            if self.use_compile and BiGAMPSpreading._compiled_step is not None:
                torch.compiler.cudagraph_mark_step_begin()
            
            # Run step function to get raw updates
            W_raw, X_raw, W_var_raw, X_var_raw, s_vals, Z_hat, V = step_fn(
                W_flat, X_flat, W_var_flat, X_var_flat,
                Y_flat, F_flat, i_offset, j_offset, alpha_mask_exp,
                S, N1, N2, self.noise_var, is_rademacher, prev_s
            )
            
            new_val = compute_log_likelihood(Y_flat, Z_hat, V, self.noise_var)
            
            # Acceptance Logic (Vectorized)
            # pass_mask: (B,) boolean tensor
            if step == 0:
                pass_mask = torch.ones(B, dtype=torch.bool, device=self.device)
                current_val = new_val  # Initialize current_val
            else:
                # Accept if Likelihood improved or within tolerance
                # Tolerance allows "Metropolis-like" acceptance of slightly worse states (for Onsager)
                pass_mask = new_val >= (current_val - acceptance_tolerance)
            
            # Broadcast masks for shape (B, S*N, M)
            # W_flat: (B, S*N, M) -> mask needs (B, 1, 1)
            pass_mask_3d = pass_mask.view(B, 1, 1)
            pass_mask_2d = pass_mask.view(B, 1) # For prev_s (B, SC)
            
            # --- 1. Update Safe State (Commit Valid States) ---
            # If accepted: W_safe = W_flat (current position becomes the new safe base)
            # If rejected: W_safe remains unchanged (keeps the old safe base)
            # Note: We must update W_safe BEFORE changing W_flat
            
            # Initialize safely if first step
            if s_safe is None:
                 s_safe = torch.zeros_like(s_vals)
            
            W_safe = torch.where(pass_mask_3d, W_flat, W_safe)
            X_safe = torch.where(pass_mask_3d, X_flat, X_safe)
            W_var_safe = torch.where(pass_mask_3d, W_var_flat, W_var_safe)
            X_var_safe = torch.where(pass_mask_3d, X_var_flat, X_var_safe)
            if prev_s is not None and s_safe is not None:
                s_safe = torch.where(pass_mask_2d, prev_s, s_safe)
            
            # --- 2. Update Likelihood & Damping ---
            current_val = torch.where(pass_mask, new_val, current_val)
            
            damping = torch.where(pass_mask, 
                                  torch.clamp(damping * step_incr, max=step_max),
                                  torch.clamp(damping * step_decr, min=step_min))
            
            # --- 3. Compute Next State (Main Update) ---
            # If Accepted: New = Damping * Raw + (1-Damping) * Old
            # If Rejected: New = Safe (Backtrack)
            
            d_view = damping.view(B, 1, 1)
            
            # Candidate if accepted (Damped Update)
            W_accepted = d_view * W_raw + (1 - d_view) * W_flat
            X_accepted = d_view * X_raw + (1 - d_view) * X_flat
            W_var_accepted = d_view * W_var_raw + (1 - d_view) * W_var_flat
            X_var_accepted = d_view * X_var_raw + (1 - d_view) * X_var_flat
            
            # Candidate if rejected (Backtrack to Safe)
            # Since we just updated W_safe to be W_flat (on accept) or kept old W_safe (on reject),
            # W_safe NOW contains exactly what we want to backtrack to/start from.
            # Wait: If rejected, W_safe is the *old* point. We want to reset W_flat to that.
            # If accepted, W_safe is the *current* point. But we want W_flat to move forward.
            
            W_flat = torch.where(pass_mask_3d, W_accepted, W_safe)
            X_flat = torch.where(pass_mask_3d, X_accepted, X_safe)
            W_var_flat = torch.where(pass_mask_3d, W_var_accepted, W_var_safe)
            X_var_flat = torch.where(pass_mask_3d, X_var_accepted, X_var_safe)
            
            # Onsager Scaling (Vectorized)
            # prev_s logic:
            # If accepted: prev_s = s_vals * damping
            # If rejected: prev_s = s_safe (Backtrack)
            
            # --- ONSAGER CONTROL FIX (Adaptive) ---
            if self.onsager_correction:
                if prev_s is None:
                     prev_s = torch.zeros_like(s_vals)
                     
                prev_s_accepted = s_vals * damping.view(B, 1)
                prev_s_rejected = s_safe
                prev_s = torch.where(pass_mask_2d, prev_s_accepted, prev_s_rejected)
            else:
                prev_s = None


            # --- WARM RESTART LOGIC (Optimized) ---
            restart_msg = ""
            if adaptive_restart:
                # 1. Update counters
                is_stuck = (damping <= (step_min + 1e-6))
                
                # Masked update for stuck_counter (avoiding in-place boolean indexing if possible, but boolean mask index is fast)
                # stuck_counter[is_stuck] += 1
                # stuck_counter[~is_stuck] = 0
                stuck_counter = torch.where(is_stuck, stuck_counter + 1, torch.zeros_like(stuck_counter))
                
                # 2. Identify candidates
                restart_mask = (stuck_counter > restart_patience)
                # Avoid nonzero() sync unless necessary for logging or specific sparse ops
                # Here we can just use torch.where for the update
                
                # 3. Apply Warm Restart (Unconditional Masked Update - No Sync)
                
                # Reset damping
                damping = torch.where(restart_mask, torch.tensor(self.damping, device=self.device), damping)
                stuck_counter = torch.where(restart_mask, torch.zeros_like(stuck_counter), stuck_counter)
                current_val = torch.where(restart_mask, torch.tensor(-float('inf'), device=self.device), current_val)
                
                # Perturb State
                noise_W = torch.randn_like(W_flat) * restart_noise
                noise_X = torch.randn_like(X_flat) * restart_noise
                
                # Apply only where restart_mask
                restart_mask_3d = restart_mask.view(B, 1, 1)
                W_flat = torch.where(restart_mask_3d, W_flat + noise_W, W_flat)
                X_flat = torch.where(restart_mask_3d, X_flat + noise_X, X_flat)
                
                # Also update safe state (commit the jump)
                W_safe = torch.where(restart_mask_3d, W_flat, W_safe)
                X_safe = torch.where(restart_mask_3d, X_flat, X_safe)
                W_var_safe = torch.where(restart_mask_3d, W_var_flat, W_var_safe)
                X_var_safe = torch.where(restart_mask_3d, X_var_flat, X_var_safe)

            # Debug / Display
            if step % 500 == 0:
                if adaptive_restart:
                     n_rest = restart_mask.float().sum().item()
                     if n_rest > 0:
                         restart_msg = f" [Restarts: {int(n_rest)}]"

                mean_damp = damping.mean().item()
                min_damp = damping.min().item()
                pass_rate = pass_mask.float().mean().item() * 100
                import sys
                sys.stdout.write(f"\r[Adaptive] Damping: {mean_damp:.3f} (min {min_damp:.3f}) | Pass: {pass_rate:.0f}% | Step: {step}{restart_msg}   ")
                sys.stdout.flush()
                
            if step < 20:            
                 pass # print(f"DEBUG: Step {step}: damp_mean={damping.mean().item():.3f}, pass_cnt={pass_mask.sum().item()}/{B}")
            
            # [UI FIX] Restore Progress Bar Callback
            if step_callback:
                step_callback(step + 1, steps)

            # Record Damping History (GPU-side copy)
            if damp_history is not None:
                damp_history[step] = damping.detach().float()
        
        # Save Debug Data if recorded
        if damp_history is not None:
            debug_path = Path("MF/results/debug_damping.pt")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'alpha_values': batch_alpha_values,
                'steps': torch.arange(steps),
                'damping_history': damp_history.cpu(),
                'timestamp': datetime.datetime.now().isoformat()
            }, debug_path)
            # print(f"DEBUG: Saved damping history to {debug_path}")

        # Clear the damping display line after loop completes
        import sys
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        W_hat = W_flat.reshape(B, S, N1, M).permute(1, 0, 2, 3)
        X_hat = X_flat.reshape(B, S, N2, M).permute(1, 0, 3, 2)
        return W_hat, X_hat

    def supports_batch_training(self) -> bool:
        """Returns True - this algorithm supports parallel alpha training."""
        return True

    def train_batch_alphas(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: torch.Tensor,  # Not used - Super-Graph generates its own
        alpha_values: List[float],
        seed: int,
        max_steps: Optional[int] = None,  # Allow override for step scanning
        step_callback=None,  # Optional step-level callback
        sample_callback=None,  # Optional sample-level callback (now batch_callback)
        max_memory_gb: float = 24.0,  # Maximum GPU memory to use (default 24GB for safety)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train for multiple alpha values using Disjoint Union parallelization.

        PHASE 2 OPTIMIZATION (v3): Per-batch SuperGraph creation.
        - Each batch creates its own SuperGraph with its own C_max
        - C_max is determined by max(alpha) in that batch, not global alpha_max
        - Eliminates padding zero computation for small-alpha batches
        - Expected speedup: 50%+ for typical alpha sweeps (α=0~4)

        Architecture:
        - All S samples run in parallel (Disjoint Union)
        - Alphas are batched based on memory constraints (动态分组)
        - Each batch gets a fresh SuperGraph sized to its α_max

        Args:
            W_teacher: (N1, M) teacher W matrix
            X_teacher: (M, N2) teacher X matrix
            Y_teacher: (N1, N2) Y = W @ X (not used directly)
            masks: (num_alphas, N1, N2) observation masks (not used)
            alpha_values: List of alpha values to train
            seed: Random seed
            step_callback: Optional callback(step, max_steps) for step-level progress
            sample_callback: Optional callback(batch_idx, num_batches, batch_alphas) for batch progress
            max_memory_gb: Maximum GPU memory to use (default 24GB)

        Returns:
            W_students: (num_alphas, S, N1, M) trained W matrices
            X_students: (num_alphas, S, M, N2) trained X matrices
        """
        # Input alpha_values are already batched by ParallelCoordinator in runner.py
        # We process them as a single chunk here.
        S = self.config.training.samples_per_alpha
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        A = len(alpha_values)
        alpha_max = max(alpha_values) if alpha_values else 4.0

        num_batches = 1
        dynamic_batches = [(0, A, alpha_max)]

        # ===== Global SuperGraph Removed =====
        # Refactored to per-batch creation to prevent OOM on large problems.
        # See loop below.

        # Allocate result tensors
        W_result = torch.zeros(A, S, N1, M, device=self.device)
        X_result = torch.zeros(A, S, M, N2, device=self.device)

        # Train in batches using global SuperGraph
        for batch_idx, (alpha_start, alpha_end, _) in enumerate(dynamic_batches):
            batch_alpha_indices = list(range(alpha_start, alpha_end))
            batch_alpha_list = [alpha_values[i] for i in batch_alpha_indices]
            
            # Notify UI of current batch alpha range
            if sample_callback:
                sample_callback(batch_idx, num_batches, batch_alpha_list)

            # Create per-batch spreading_data to optimize memory (C_max tailored to batch max)
            # This ensures we don't allocate massive tensors for small alphas.
            # We offset seed by batch_idx to avoid identical random streams for different batches if safe
            # but usually base_seed is fine if alphas differ. We'll use base_seed + batch_idx for safety.
            batch_spreading_data = self.create_spreading_data(
                W_teacher, X_teacher, batch_alpha_list, S, seed + batch_idx
            )

            # Train this batch using LOCAL spreading_data
            # Note: batch_alpha_indices=None because spreading_data ONLY contains this batch's alphas
            W_batch, X_batch = self.train_full_parallel(
                batch_spreading_data,
                batch_alpha_indices=None,  # All alphas in this partial data
                verbose=False,
                step_callback=step_callback,
                max_steps=max_steps,
                batch_alpha_values=batch_alpha_list,
            )
            
            # free memory
            del batch_spreading_data
            # W_batch: (S, B, N1, M), X_batch: (S, B, M, N2)

            # Store results: transpose (S, B, ...) -> (B, S, ...)
            W_result[alpha_start:alpha_end] = W_batch.transpose(0, 1)
            X_result[alpha_start:alpha_end] = X_batch.transpose(0, 1)

            # Clear cache between batches
            if batch_idx < num_batches - 1:
                torch.cuda.empty_cache()

        return W_result, X_result


    def train_single_alpha(
        self,
        alpha: float,
        teacher_data,
        graph_data,
    ):
        """
        Required by AlgorithmBase but not used in parallel implementation.

        Use train_sample() or train_all_samples() instead for parallel training.
        """
        raise NotImplementedError(
            "BiGAMPSpreading uses train_sample() for parallel alpha training. "
            "Use train_all_samples() or run_spreading_parallel() instead."
        )


    def _train_full_parallel_general(
        self,
        spreading_data: SpreadingDataParallel,
        batch_alpha_indices: Optional[List[int]] = None,
        verbose: bool = False,
        step_callback=None,
        max_steps: Optional[int] = None,
        batch_alpha_values: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified Vector implementation for general graphs."""
        S = spreading_data.S
        A = spreading_data.A
        N1 = spreading_data.supergraph.N1
        N2 = spreading_data.supergraph.N2
        M = spreading_data.M
        # Use C_max from general graph
        C_max = spreading_data.supergraph.C_max
        SC = S * C_max
        N_total = N1 + N2

        if batch_alpha_indices is None:
            batch_alpha_indices = list(range(A))
        B = len(batch_alpha_indices)

        # DEBUG: Check Edge Distribution
        if verbose: # Print stats only if verbose requested
            s_idx_debug = 0
            if hasattr(spreading_data.supergraph, 'edge_type'):
                edge_types = spreading_data.supergraph.edge_type[s_idx_debug]
                # Filter valid edges (mask is per alpha, but edge_type is for max alpha)
                # Just show raw distribution of pre-generated edges
                n_ww = (edge_types == 0).sum().item()
                n_wx = (edge_types == 1).sum().item()
                n_xx = (edge_types == 2).sum().item()
                total = len(edge_types)
                
                print(f"\n[General Mode Statistics] Edge Distribution (Sample 0, C_max={total}):")
                print(f"  W-W (Type 0): {n_ww} ({n_ww/total*100:.1f}%)")
                print(f"  W-X (Type 1): {n_wx} ({n_wx/total*100:.1f}%)")
                print(f"  X-X (Type 2): {n_xx} ({n_xx/total*100:.1f}%)")
                if n_ww == 0 and n_xx == 0:
                    print("  [WARNING] No intra-connections found! Graph is effectively bipartite.")
            else:
                print("\n[General Mode Statistics] edge_type not found in supergraph!")

        # Get mask
        full_alpha_mask = spreading_data.supergraph.alpha_mask
        batch_alpha_mask = full_alpha_mask[batch_alpha_indices]

        # General Offset Calculation
        # Use unified indices from general supergraph
        i_offset, j_offset = compute_offset_indices(
            spreading_data.supergraph.a_idx,
            spreading_data.supergraph.b_idx,
            N_total, N_total # Use N_total for both
        )

        # Flat data
        F_flat = spreading_data.F_super.reshape(SC, M)
        Y_flat = spreading_data.Y_super.reshape(SC)
        alpha_mask_exp = batch_alpha_mask.unsqueeze(1).expand(B, S, C_max).reshape(B, SC)

        # === PHYSICAL SORTING for GPU Memory Coalescence ===
        # Sort edges by source node index to improve L2 cache hit rate
        # This transforms random memory access into sequential access
        sort_idx = torch.argsort(i_offset)
        i_offset = i_offset[sort_idx]
        j_offset = j_offset[sort_idx]
        F_flat = F_flat[sort_idx]
        Y_flat = Y_flat[sort_idx]
        alpha_mask_exp = alpha_mask_exp[:, sort_idx]
        if verbose:
            print("  [Optimization] Edges sorted for coalesced memory access")

        # ===== INITIALIZATION (Teacher-Assisted or Random) =====
        init_mode = getattr(self.config.algorithm_params, 'init_mode', 'random')
        init_overlap = getattr(self.config.algorithm_params, 'init_overlap', 0.95)

        if init_mode == 'teacher':
            # Teacher-Assisted Initialization for General Graph
            # Construct unified teacher vector: V = [W; X^T]
            W_teacher = spreading_data.W_teacher  # (N1, M)
            X_teacher_T = spreading_data.X_teacher.T  # (M, N2) -> (N2, M)
            V_teacher = torch.cat([W_teacher, X_teacher_T], dim=0)  # (N_total, M)
            
            V_flat = self._initialize_near_teacher(
                (B, S * N_total, M),
                V_teacher,
                init_overlap,
                S,
                self.device,
                self.storage_dtype
            )
            if verbose:
                print(f"  [Init] Teacher-Assisted General (m={init_overlap})")
        else:
            # Random Initialization (Cold Start - default)
            V_flat = torch.randn(B, S * N_total, M, device=self.device, dtype=self.storage_dtype) * 0.1

        V_var_flat = torch.ones(B, S * N_total, M, device=self.device, dtype=self.storage_dtype)

        prev_s = None
        is_rademacher = (self.f_distribution == 'rademacher')

        steps = max_steps if max_steps is not None else self.max_steps
        
        # Select step function based on chunk_size
        # chunk_size > 0: Use new chunked version (memory optimized)
        # chunk_size = 0: Use legacy version (for compatibility)
        use_chunked = (self.chunk_size > 0)
        
        if use_chunked:
            # NEW: Chunked streaming implementation
            step_fn = bigamp_step_general_chunked
            # No need for class-level caching - chunked version handles its own compilation
            if verbose:
                print(f"  [General Mode] Using chunked processing (chunk_size={self.chunk_size})")
        else:
            # LEGACY: Original implementation
            step_fn = bigamp_step_disjoint_union_flat_general
            if self.use_compile:
                if BiGAMPSpreading._compiled_step_general is None:
                    try:
                        torch._dynamo.reset()
                        BiGAMPSpreading._compiled_step_general = torch.compile(
                            bigamp_step_disjoint_union_flat_general, 
                            mode="default"
                        )
                    except Exception:
                        BiGAMPSpreading._compiled_step_general = bigamp_step_disjoint_union_flat_general
                step_fn = BiGAMPSpreading._compiled_step_general

        # Loop
        for step in range(steps):
            if use_chunked:
                # Chunked version: pass chunk_size and use_compile
                V_flat, V_var_flat, s_values, _, _ = step_fn(
                    V_flat, V_var_flat, Y_flat, F_flat, i_offset, j_offset,
                    alpha_mask_exp, S, N_total, self.damping, self.noise_var,
                    is_rademacher, prev_s, self.chunk_size, self.use_compile
                )
            else:
                # Legacy version
                if self.use_compile and BiGAMPSpreading._compiled_step_general is not None:
                    torch.compiler.cudagraph_mark_step_begin()
                
                V_flat, V_var_flat, s_values, _, _ = step_fn(
                    V_flat, V_var_flat, Y_flat, F_flat, i_offset, j_offset,
                    alpha_mask_exp, S, N_total, self.damping, self.noise_var,
                    is_rademacher, prev_s
                )

            # Onsager
            if self.onsager_correction:
                prev_s = s_values
            else:
                prev_s = None

            if verbose and (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{steps}")
            if step_callback:
                step_callback(step + 1, steps)

        # Unpack V to W and X
        # V: (B, S*N_total, M)
        # Reshape to (B, S, N_total, M)
        V_reshaped = V_flat.view(B, S, N_total, M)
        
        # Split
        W_out = V_reshaped[:, :, :N1, :] # (B, S, N1, M)
        X_out = V_reshaped[:, :, N1:, :] # (B, S, N2, M)
        
        # Original logic returns (B, S, N1, M).permute(1, 0, 2, 3) -> (S, B, N1, M)
        W_hat = W_out.permute(1, 0, 2, 3)  # (S, B, N1, M)
        
        # X_out is (B, S, N2, M), need to return (S, B, M, N2) for API compatibility
        X_hat = X_out.permute(1, 0, 3, 2)  # (S, B, M, N2)
        
        return W_hat, X_hat

# ============================================================================
# Convenience Functions
# ============================================================================

def run_spreading_parallel(
    config,
    verbose: bool = True,
    alpha_batch_size: int = 10,
    skip_metrics: bool = False,
) -> Dict:
    """
    Run complete spreading parallel experiment.
    
    Args:
        config: Experiment configuration
        verbose: Compute and print metrics during training
        alpha_batch_size: Number of alphas to process in one parallel batch.
                         Default is 10. Decrease for larger problems to avoid OOM.

    This is a standalone function that handles:
    1. Teacher creation (using config.teacher_key)
    2. SpreadingDataParallel creation
    3. Training all samples
    4. Metrics computation

    Returns:
        Dictionary with results for each alpha
    """
    import time
    from ..metrics.spreading import compute_all_metrics_spreading_parallel
    from ..registry import get_teacher
    from ...core.device import setup_device

    device, device_info = setup_device()

    # Get configuration
    m = config.matrix
    alpha_values = config.alpha.get_values()
    S = config.training.samples_per_alpha
    seed = config.training.seed

    if verbose:
        print("[Spreading Parallel] Running with:")
        print(f"  Matrix: {m.N1}x{m.N2}, M={m.M}")
        print(f"  Alpha: {alpha_values[0]:.2f} ~ {alpha_values[-1]:.2f} ({len(alpha_values)} points)")
        print(f"  Samples: {S}")
        print(f"  F distribution: {config.spreading.f_distribution if config.spreading else 'gaussian'}")

    start_time = time.time()

    # Create teacher using existing system
    teacher_cls = get_teacher(config.teacher_key).cls
    teacher = teacher_cls()
    W_teacher, X_teacher = teacher.create(m.N1, m.N2, m.M, device, seed)

    if verbose:
        print(f"  Teacher type: {config.teacher_key}")

    # Create algorithm instance
    algorithm = BiGAMPSpreading(config, device)

    # Create spreading data
    spreading_data = algorithm.create_spreading_data(
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        alpha_values=alpha_values,
        S=S,
        base_seed=seed,
    )

    # Train all samples (Parallel optimized with Alpha Batching)
    # Train all samples (Parallel optimized with Alpha Batching)
    # alpha_batch_size is passed as argument
    W_students = torch.zeros(S, len(alpha_values), m.N1, m.M, device=device)
    X_students = torch.zeros(S, len(alpha_values), m.M, m.N2, device=device)
    
    import math
    num_batches = math.ceil(len(alpha_values) / alpha_batch_size)
    
    for i in range(num_batches):
        start_idx = i * alpha_batch_size
        end_idx = min((i + 1) * alpha_batch_size, len(alpha_values))
        batch_indices = list(range(start_idx, end_idx))
        
        if verbose:
            print(f"  Training Alpha Batch {i+1}/{num_batches} (Alphas {start_idx}-{end_idx-1})")
        
        # Uses Disjoint Union to process all samples in parallel for this batch of alphas
        W_batch, X_batch = algorithm.train_full_parallel(
            spreading_data,
            batch_alpha_indices=batch_indices,
            verbose=verbose
        )
        
        # W_batch: (S, B, N1, M) -> assign to main storage
        W_students[:, start_idx:end_idx] = W_batch.detach()
        X_students[:, start_idx:end_idx] = X_batch.detach()
        
        # Clear cache between batches
        del W_batch, X_batch
        torch.cuda.empty_cache()

    # Compute metrics (skip for large problems to avoid OOM)
    if not skip_metrics:
        metrics = compute_all_metrics_spreading_parallel(
            W_students, X_students, spreading_data
        )
    else:
        metrics = None

    total_time = time.time() - start_time

    if verbose:
        print(f"\n[Spreading Parallel] Completed in {total_time:.1f}s")

    # Convert to standard result format
    results = {}
    if metrics is not None:
        for i, alpha in enumerate(alpha_values):
            results[float(alpha)] = {
                'Q_Y_mean': float(metrics['Q_Y_mean'][i]),
                'Q_Y_std': float(metrics['Q_Y_std'][i]),
                'Q_W_mean': float(metrics['Q_W_mean'][i]),
                'Q_W_std': float(metrics['Q_W_std'][i]),
                'Q_X_mean': float(metrics['Q_X_mean'][i]),
                'Q_X_std': float(metrics['Q_X_std'][i]),
            }
    # If skip_metrics, results will be empty and caller must compute manually

    return {
        'results': results,
        'config': config,
        'total_time': total_time,
        'spreading_data': spreading_data,
        'W_students': W_students,
        'X_students': X_students,
    }
