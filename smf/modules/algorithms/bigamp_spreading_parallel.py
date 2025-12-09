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
from dataclasses import dataclass
import math
import torch

from ..registry import register_algorithm
from .base import AlgorithmBase
from ..graphs.supergraph import SuperGraphData, create_supergraph
from ..teachers.random_spreading import SpreadingDataParallel


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
        i_idx = supergraph.i_idx[s]  # (C_max,)
        j_idx = supergraph.j_idx[s]  # (C_max,)

        W_sel = W_teacher[i_idx]     # (C_max, M)
        X_sel = X_teacher[:, j_idx].T  # (C_max, M)

        # Convert F to float for computation (handles int8 Rademacher)
        F_s = F_super[s].float() if F_super.dtype == torch.int8 else F_super[s]
        
        # Y[c] = (1/√M) Σ_μ F[c,μ] W[i,μ] X[μ,j]
        Y_super[s] = alpha_scale * (F_s * W_sel * X_sel).sum(dim=1)

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
    idx_expanded = idx.view(1, C_max, 1).expand(A, C_max, M)

    # Scatter add
    result.scatter_add_(1, idx_expanded, src_masked)

    return result


def bigamp_spreading_parallel_step(
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
    W_var_new = 1.0 / (1.0 + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)  # Clamp r_W to prevent explosion
    W_hat_new = W_hat + W_var_new * r_W  # CRITICAL FIX: incremental update (was missing + W_hat)

    # ===== Update X =====
    # Similar logic for X
    W_sel = W_hat[:, i_idx, :]  # (A, C_max, M)

    r_X_contrib = alpha_scale * F_expanded * W_sel * s_expanded  # (A, C_max, M)
    # Need to transpose for X: aggregate by j_idx
    r_X_contrib_T = r_X_contrib.transpose(1, 2)  # (A, M, C_max)

    # Scatter to (A, M, N2)
    r_X = torch.zeros(A, M, N2, device=W_hat.device, dtype=W_hat.dtype)
    j_idx_expanded = j_idx.view(1, 1, C_max).expand(A, M, C_max)
    mask_expanded_X = alpha_mask.unsqueeze(1).float()  # (A, 1, C_max)
    r_X.scatter_add_(2, j_idx_expanded, r_X_contrib_T * mask_expanded_X)

    tau_X_contrib = alpha_scale_sq * F_sq_expanded * W_sel.pow(2) * inv_V  # (A, C_max, M) - F² added
    tau_X_contrib_T = tau_X_contrib.transpose(1, 2)  # (A, M, C_max)
    tau_X = torch.zeros(A, M, N2, device=W_hat.device, dtype=W_hat.dtype)
    tau_X.scatter_add_(2, j_idx_expanded, tau_X_contrib_T * mask_expanded_X)
    tau_X = tau_X.clamp(min=1e-10)

    X_var_new = 1.0 / (1.0 + tau_X)
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)  # Clamp r_X to prevent explosion
    X_hat_new = X_hat + X_var_new * r_X  # CRITICAL FIX: incremental update (was missing + X_hat)

    # ===== Damping =====
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = damping * W_var_new + (1 - damping) * W_var
    X_var_out = damping * X_var_new + (1 - damping) * X_var

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
    r_W.scatter_add_(1, idx_W, r_W_contrib)

    inv_V = (1.0 / denom).unsqueeze(2)  # (A, SC, 1)
    F_sq_exp = F_exp.pow(2)  # (1, SC, M) - F² for correct variance weighting
    tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V * mask_exp
    tau_W = torch.zeros(A, S * N1, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_W.scatter_add_(1, idx_W, tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)

    W_var_new = 1.0 / (1.0 + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_flat + W_var_new * r_W  # CRITICAL FIX: incremental update (was missing + W_flat)

    # X update
    r_X_contrib = alpha_scale * F_exp * W_sel * s_exp * mask_exp  # (A, SC, M)
    r_X = torch.zeros(A, S * N2, M, device=W_hat.device, dtype=W_hat.dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_add_(1, idx_X, r_X_contrib)

    tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V * mask_exp  # F² added
    tau_X = torch.zeros(A, S * N2, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_X.scatter_add_(1, idx_X, tau_X_contrib)
    tau_X = tau_X.clamp(min=1e-10)

    X_var_new = 1.0 / (1.0 + tau_X)
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
    W_var_out = damping * W_var_new + (1 - damping) * W_var
    X_var_out = damping * X_var_new + (1 - damping) * X_var

    # NaN protection
    W_hat_out = torch.nan_to_num(W_hat_out, nan=0.0)
    X_hat_out = torch.nan_to_num(X_hat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)

    return W_hat_out, X_hat_out, W_var_out, X_var_out, s_values


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
    
    # ===== 1. Gather: one operation for all S*C_max edges =====
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    W_var_sel = W_var_flat[:, i_offset, :]  # (A, SC, M)
    X_var_sel = X_var_flat[:, j_offset, :]  # (A, SC, M)
    
    # ===== 2. Forward pass =====
    # Convert F to float if stored as int8 (Rademacher optimization)
    F_compute = F_flat.float() if F_flat.dtype == torch.int8 else F_flat
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    Z_hat = alpha_scale * (F_exp * W_sel * X_sel).sum(dim=2)  # (A, SC)
    Z_hat = Z_hat * alpha_mask_exp.float()
    
    # ===== 3. Variance =====
    if is_rademacher:
        # F² = 1 for Rademacher, skip pow(2) computation
        V = alpha_scale_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel).sum(dim=2)
        F_sq_exp = None  # Not needed
    else:
        F_sq_exp = F_exp.pow(2)  # (1, SC, M)
        V = alpha_scale_sq * (F_sq_exp * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V * alpha_mask_exp.float() + 1e-10
    
    # ===== 4. Residuals =====
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom  # (A, SC)
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    s_values = s_values * alpha_mask_exp.float()
    
    # ===== 5. Scatter: one operation for all edges =====
    s_exp = s_values.unsqueeze(2)  # (A, SC, 1)
    mask_exp = alpha_mask_exp.unsqueeze(2).float()  # (A, SC, 1)
    inv_V = (1.0 / denom).unsqueeze(2)  # (A, SC, 1)
    
    # W update
    r_W_contrib = alpha_scale * F_exp * X_sel * s_exp * mask_exp  # (A, SC, M)
    r_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=W_flat.dtype)
    idx_W = i_offset.view(1, SC, 1).expand(A, SC, M)
    r_W.scatter_add_(1, idx_W, r_W_contrib)
    
    if is_rademacher:
        tau_W_contrib = alpha_scale_sq * X_sel.pow(2) * inv_V * mask_exp
    else:
        tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V * mask_exp
    tau_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=W_flat.dtype)
    tau_W.scatter_add_(1, idx_W, tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)
    
    W_var_new = 1.0 / (1.0 + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_flat + W_var_new * r_W
    
    # X update
    r_X_contrib = alpha_scale * F_exp * W_sel * s_exp * mask_exp  # (A, SC, M)
    r_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=W_flat.dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_add_(1, idx_X, r_X_contrib)
    
    if is_rademacher:
        tau_X_contrib = alpha_scale_sq * W_sel.pow(2) * inv_V * mask_exp
    else:
        tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V * mask_exp
    tau_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=W_flat.dtype)
    tau_X.scatter_add_(1, idx_X, tau_X_contrib)
    tau_X = tau_X.clamp(min=1e-10)
    
    X_var_new = 1.0 / (1.0 + tau_X)
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_flat + X_var_new * r_X
    
    # ===== 6. Damping =====
    W_flat_out = damping * W_hat_new + (1 - damping) * W_flat
    X_flat_out = damping * X_hat_new + (1 - damping) * X_flat
    W_var_out = damping * W_var_new + (1 - damping) * W_var_flat
    X_var_out = damping * X_var_new + (1 - damping) * X_var_flat
    
    # NaN protection
    W_flat_out = torch.nan_to_num(W_flat_out, nan=0.0)
    X_flat_out = torch.nan_to_num(X_flat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)
    
    return W_flat_out, X_flat_out, W_var_out, X_var_out, s_values


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
    key="bigamp_spreading_parallel",
    name="BiG-AMP Spreading (Parallel)",
    description="GPU parallel across all alphas - 30x faster for production",
    default_params={
        'damping': 0.5,
        'noise_var': 1e-10,
    },
)
class BiGAMPSpreadingParallel(AlgorithmBase):
    """
    BiG-AMP with random spreading, parallel across alpha values.

    Configurable options:
    - teacher_key: 'standard' (Gaussian) or 'orthogonal' - via config.teacher_key
    - f_distribution: 'gaussian' or 'rademacher' - via config.spreading.f_distribution

    Usage:
        config = Config(
            algorithm_key="bigamp_spreading_parallel",
            teacher_key="orthogonal",  # Controls W, X generation
            spreading=SpreadingConfig(f_distribution="rademacher"),
        )
    """

    # Class-level cache for compiled step function
    _compiled_step = None

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
        self.damping = config.algorithm.damping
        self.noise_var = config.algorithm.noise_var
        self.max_steps = config.training.max_steps

        # Spreading configuration
        spreading_cfg = config.spreading
        if spreading_cfg is not None:
            self.f_distribution = spreading_cfg.f_distribution
            self.spreading_seed = spreading_cfg.seed
        else:
            # Default values
            self.f_distribution = 'gaussian'
            self.spreading_seed = 12345

        # Validate f_distribution
        if self.f_distribution not in F_GENERATORS:
            raise ValueError(
                f"Invalid f_distribution='{self.f_distribution}'. "
                f"Available: {list(F_GENERATORS.keys())}"
            )

        # torch.compile for kernel fusion (Phase 4 optimization)
        self.use_compile = getattr(config.algorithm, 'use_compile', True)
        if self.use_compile and BiGAMPSpreadingParallel._compiled_step is None:
            try:
                # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
                BiGAMPSpreadingParallel._compiled_step = torch.compile(
                    bigamp_step_disjoint_union_flat,
                    mode='default'
                )
                print(f"[BiG-AMP Spreading Parallel] torch.compile enabled")
            except Exception as e:
                print(f"[BiG-AMP Spreading Parallel] torch.compile failed: {e}")
                self.use_compile = False

        print(f"[BiG-AMP Spreading Parallel] F distribution: {self.f_distribution}")

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
        N1, M = W_teacher.shape
        _, N2 = X_teacher.shape

        # Create super-graph
        supergraph = create_supergraph(
            N1=N1,
            N2=N2,
            M=M,
            alpha_values=alpha_values,
            S=S,
            base_seed=base_seed,
            device=self.device,
        )

        # Generate F_super using selected distribution
        F_super = generate_F_super(
            supergraph=supergraph,
            M=M,
            base_seed=self.spreading_seed,
            device=self.device,
            f_distribution=self.f_distribution,
        )

        # Compute Y_super
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
            W_hat, X_hat, W_var, X_var, prev_s = bigamp_spreading_parallel_step(
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

        # ===== OPTIMIZATION 2: Initialize in FLAT format =====
        # Shape: (B, S*N, M) instead of (S, B, N, M)
        W_flat = torch.randn(B, S * N1, M, device=self.device) * 0.1
        X_flat = torch.randn(B, S * N2, M, device=self.device) * 0.1
        W_var_flat = torch.ones(B, S * N1, M, device=self.device)
        X_var_flat = torch.ones(B, S * N2, M, device=self.device)

        prev_s = None
        is_rademacher = (self.f_distribution == 'rademacher')

        # ===== OPTIMIZATION 3: Use compiled step if available =====
        step_fn = BiGAMPSpreadingParallel._compiled_step if self.use_compile and BiGAMPSpreadingParallel._compiled_step is not None else bigamp_step_disjoint_union_flat

        # BiG-AMP iterations with optimized flat function
        for step in range(self.max_steps):
            W_flat, X_flat, W_var_flat, X_var_flat, prev_s = step_fn(
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

            if verbose and (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{self.max_steps}")

            if step_callback:
                step_callback(step + 1, self.max_steps)

        # ===== Only reshape at the END for output =====
        # (B, S*N1, M) -> (B, S, N1, M) -> (S, B, N1, M)
        W_hat = W_flat.reshape(B, S, N1, M).permute(1, 0, 2, 3)
        # (B, S*N2, M) -> (B, S, N2, M) -> (S, B, M, N2)
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
        step_callback=None,  # Optional step-level callback
        sample_callback=None,  # Optional sample-level callback (now batch_callback)
        max_memory_gb: float = 24.0,  # Maximum GPU memory to use (default 24GB for safety)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train for multiple alpha values using Disjoint Union parallelization.

        New architecture (v2):
        - All S samples run in parallel (Disjoint Union)
        - Alphas are batched based on memory constraints

        Args:
            W_teacher: (N1, M) teacher W matrix
            X_teacher: (M, N2) teacher X matrix
            Y_teacher: (N1, N2) Y = W @ X (not used directly)
            masks: (num_alphas, N1, N2) observation masks (not used)
            alpha_values: List of alpha values to train
            seed: Random seed
            step_callback: Optional callback(step, max_steps) for step-level progress
            sample_callback: Optional callback(batch_idx, num_batches) for batch progress
            max_memory_gb: Maximum GPU memory to use (default 28GB)

        Returns:
            W_students: (num_alphas, S, N1, M) trained W matrices
            X_students: (num_alphas, S, M, N2) trained X matrices
        """
        from ...core.memory_manager import get_spreading_memory_strategy

        S = self.config.training.samples_per_alpha
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        A = len(alpha_values)
        alpha_max = max(alpha_values) if alpha_values else 4.0

        # Get memory strategy with dynamic batching
        strategy = get_spreading_memory_strategy(
            N1, N2, M, S, A, alpha_max,
            available_gb=max_memory_gb + 3.0,  # Add back the reserved 3GB
            verbose=True,
            alpha_values=alpha_values,  # Enable dynamic batching
        )
        
        # Use dynamic batches if available, otherwise fall back to fixed
        dynamic_batches = strategy.get('dynamic_batches')
        if dynamic_batches:
            num_batches = len(dynamic_batches)
        else:
            alphas_per_batch = strategy['alphas_per_batch']
            num_batches = strategy['num_batches']
            # Create fixed batch ranges
            dynamic_batches = [
                (i * alphas_per_batch, min((i + 1) * alphas_per_batch, A), alpha_max)
                for i in range(num_batches)
            ]

        # Create spreading data (generates its own masks via Super-Graph)
        spreading_data = self.create_spreading_data(
            W_teacher, X_teacher, alpha_values, S, seed
        )

        # Allocate result tensors
        W_result = torch.zeros(A, S, N1, M, device=self.device)
        X_result = torch.zeros(A, S, M, N2, device=self.device)

        # Train in batches (now with variable batch sizes)
        for batch_idx, (alpha_start, alpha_end, _) in enumerate(dynamic_batches):
            batch_alpha_indices = list(range(alpha_start, alpha_end))

            # Train this batch using Disjoint Union (all samples parallel)
            W_batch, X_batch = self.train_full_parallel(
                spreading_data,
                batch_alpha_indices=batch_alpha_indices,
                verbose=False,
                step_callback=step_callback,
            )
            # W_batch: (S, B, N1, M), X_batch: (S, B, M, N2)

            # Store results: transpose (S, B, ...) -> (B, S, ...)
            W_result[alpha_start:alpha_end] = W_batch.transpose(0, 1)
            X_result[alpha_start:alpha_end] = X_batch.transpose(0, 1)

            # Batch progress callback
            if sample_callback:
                sample_callback(batch_idx + 1, num_batches)

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
            "BiGAMPSpreadingParallel uses train_sample() for parallel alpha training. "
            "Use train_all_samples() or run_spreading_parallel() instead."
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def run_spreading_parallel(
    config,
    verbose: bool = True,
) -> Dict:
    """
    Run complete spreading parallel experiment.

    This is a standalone function that handles:
    1. Teacher creation (using config.teacher_key)
    2. SpreadingDataParallel creation
    3. Training all samples
    4. Metrics computation

    Args:
        config: Config object with all parameters
        verbose: Print progress

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
        print(f"[Spreading Parallel] Running with:")
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
    algorithm = BiGAMPSpreadingParallel(config, device)

    # Create spreading data
    spreading_data = algorithm.create_spreading_data(
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        alpha_values=alpha_values,
        S=S,
        base_seed=seed,
    )

    if verbose:
        print(f"  SuperGraph created: C_max={spreading_data.C_max}")

    # Train all samples
    W_students, X_students = algorithm.train_all_samples(
        spreading_data, verbose=verbose
    )

    # Compute metrics
    metrics = compute_all_metrics_spreading_parallel(
        W_students, X_students, spreading_data
    )

    total_time = time.time() - start_time

    if verbose:
        print(f"\n[Spreading Parallel] Completed in {total_time:.1f}s")

    # Convert to standard result format
    results = {}
    for i, alpha in enumerate(alpha_values):
        results[float(alpha)] = {
            'Q_Y_mean': float(metrics['Q_Y_mean'][i]),
            'Q_Y_std': float(metrics['Q_Y_std'][i]),
            'Q_W_mean': float(metrics['Q_W_mean'][i]),
            'Q_W_std': float(metrics['Q_W_std'][i]),
            'Q_X_mean': float(metrics['Q_X_mean'][i]),
            'Q_X_std': float(metrics['Q_X_std'][i]),
        }

    return {
        'results': results,
        'config': config,
        'total_time': total_time,
        'spreading_data': spreading_data,
        'W_students': W_students,
        'X_students': X_students,
    }
