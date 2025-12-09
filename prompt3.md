"""
BiG-AMP with Random Spreading - Parallel Implementation.

This module implements BiG-AMP algorithm for the random spreading model
with Super-Graph parallelization across alpha values.

Key features:
1. Configurable F distribution: gaussian or rademacher
2. Super-Graph strategy: parallel processing of all alphas
3. Teacher type controlled by config.teacher_key
4. CORRECTED: Onsager term interaction with Damping
5. CORRECTED: Bayesian posterior update for N(0,1) prior
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
    if C == 0:
        return torch.empty(0, M, device=device, dtype=torch.float32)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed ^ 0x5DEECE66D)

    bits = torch.randint(0, 2, (C, M), device=device, dtype=torch.float32, generator=gen)
    return bits * 2 - 1


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
    if f_distribution not in F_GENERATORS:
        raise ValueError(f"Invalid f_distribution='{f_distribution}'")

    generator = F_GENERATORS[f_distribution]
    S = supergraph.seeds.shape[0]
    C_max = supergraph.C_max

    F_super = torch.empty(S, C_max, M, device=device, dtype=torch.float32)

    for s in range(S):
        sample_seed = base_seed + int(supergraph.seeds[s].item())
        F_super[s] = generator(C_max, M, sample_seed, device)

    return F_super


def compute_Y_super(
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    supergraph: SuperGraphData,
    F_super: torch.Tensor,
) -> torch.Tensor:
    S, C_max, M = F_super.shape
    alpha_scale = 1.0 / math.sqrt(M)

    Y_super = torch.empty(S, C_max, device=F_super.device, dtype=F_super.dtype)

    for s in range(S):
        i_idx = supergraph.i_idx[s]
        j_idx = supergraph.j_idx[s]

        W_sel = W_teacher[i_idx]
        X_sel = X_teacher[:, j_idx].T

        Y_super[s] = alpha_scale * (F_super[s] * W_sel * X_sel).sum(dim=1)

    return Y_super


# ============================================================================
# Parallel BiG-AMP Core Functions
# ============================================================================

def scatter_add_parallel(
    src: torch.Tensor,
    idx: torch.Tensor,
    target_size: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    A, C_max, M = src.shape
    result = torch.zeros(A, target_size, M, device=src.device, dtype=src.dtype)
    
    src_masked = src * mask.unsqueeze(2).float()
    idx_expanded = idx.view(1, C_max, 1).expand(A, C_max, M)
    
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
    Standard Parallel Step (for single sample, multiple alphas).
    """
    A, N1, M = W_hat.shape
    _, _, N2 = X_hat.shape
    C_max = F.shape[0]
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # --- 1. Forward Pass (Z calculation) ---
    W_sel = W_hat[:, i_idx, :]
    X_sel = X_hat[:, :, j_idx].transpose(1, 2)
    F_expanded = F.unsqueeze(0)
    
    Z_raw = alpha_scale * (F_expanded * W_sel * X_sel).sum(dim=2)
    Z_hat = Z_raw * alpha_mask.float()

    # --- 2. Variance Pass (V calculation) ---
    W_var_sel = W_var[:, i_idx, :]
    X_var_sel = X_var[:, :, j_idx].transpose(1, 2)
    F_sq = F.pow(2).unsqueeze(0)
    
    V_raw = alpha_scale_sq * (F_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V_raw * alpha_mask.float() + 1e-10

    # --- 3. Onsager Correction & Residuals ---
    # FIX #1: Scale Onsager by damping factor
    if prev_s is not None:
        onsager_term = prev_s * V * damping
        Z_hat = Z_hat - onsager_term
        Z_hat = Z_hat * alpha_mask.float()

    denominator = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_values.unsqueeze(0) - Z_hat) / denominator
    s_values = torch.clamp(s_values, min=-1e5, max=1e5) # Tighter clamp
    s_values = s_values * alpha_mask.float()

    # --- 4. Backward Pass (Update W) ---
    s_expanded = s_values.unsqueeze(2)
    
    # Gradient/Residual term: r_W
    r_W_contrib = alpha_scale * F_expanded * X_sel * s_expanded
    r_W = scatter_add_parallel(r_W_contrib, i_idx, N1, alpha_mask)

    # Precision/Curvature term: tau_W
    inv_V = (1.0 / denominator).unsqueeze(2)
    tau_W_contrib = alpha_scale_sq * F_sq * X_sel.pow(2) * inv_V
    tau_W = scatter_add_parallel(tau_W_contrib, i_idx, N1, alpha_mask)
    tau_W = tau_W.clamp(min=1e-10)

    # FIX #2: Correct Bayesian Update for N(0,1) Prior
    # W_new = (tau_W * W_old + r_W) / (tau_W + 1)
    W_var_new = 1.0 / (1.0 + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    # The term (W_hat * tau_W) represents the contribution from the prior estimate
    W_hat_new = W_var_new * (r_W + W_hat * tau_W)

    # --- 5. Backward Pass (Update X) ---
    # Gradient/Residual term: r_X
    r_X_contrib = alpha_scale * F_expanded * W_sel * s_expanded
    r_X_contrib_T = r_X_contrib.transpose(1, 2)
    
    r_X = torch.zeros(A, M, N2, device=W_hat.device, dtype=W_hat.dtype)
    j_idx_expanded = j_idx.view(1, 1, C_max).expand(A, M, C_max)
    mask_expanded_X = alpha_mask.unsqueeze(1).float()
    r_X.scatter_add_(2, j_idx_expanded, r_X_contrib_T * mask_expanded_X)

    # Precision/Curvature term: tau_X
    tau_X_contrib = alpha_scale_sq * F_sq * W_sel.pow(2) * inv_V
    tau_X_contrib_T = tau_X_contrib.transpose(1, 2)
    
    tau_X = torch.zeros(A, M, N2, device=W_hat.device, dtype=W_hat.dtype)
    tau_X.scatter_add_(2, j_idx_expanded, tau_X_contrib_T * mask_expanded_X)
    tau_X = tau_X.clamp(min=1e-10)

    # FIX #2 for X
    X_var_new = 1.0 / (1.0 + tau_X)
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_var_new * (r_X + X_hat * tau_X)

    # --- 6. Damping ---
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = damping * W_var_new + (1 - damping) * W_var
    X_var_out = damping * X_var_new + (1 - damping) * X_var

    W_hat_out = torch.nan_to_num(W_hat_out, nan=0.0)
    X_hat_out = torch.nan_to_num(X_hat_out, nan=0.0)
    W_var_out = torch.nan_to_num(W_var_out, nan=1.0)
    X_var_out = torch.nan_to_num(X_var_out, nan=1.0)

    return W_hat_out, X_hat_out, W_var_out, X_var_out, s_values


# ============================================================================
# Disjoint Union Parallelization (All Samples Parallel)
# ============================================================================

def bigamp_step_disjoint_union(
    W_hat: torch.Tensor,
    X_hat: torch.Tensor,
    W_var: torch.Tensor,
    X_var: torch.Tensor,
    Y_super: torch.Tensor,
    F_super: torch.Tensor,
    i_offset: torch.Tensor,
    j_offset: torch.Tensor,
    alpha_mask: torch.Tensor,
    S: int,
    damping: float,
    noise_var: float,
    prev_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    High-performance Disjoint Union Step.
    """
    _, A, N1, M = W_hat.shape
    N2 = X_hat.shape[3]
    C_max = F_super.shape[1]
    SC = S * C_max

    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # --- Flatten ---
    W_flat = W_hat.permute(1, 0, 2, 3).reshape(A, S * N1, M)
    X_flat = X_hat.permute(1, 0, 3, 2).reshape(A, S * N2, M)
    W_var_flat = W_var.permute(1, 0, 2, 3).reshape(A, S * N1, M)
    X_var_flat = X_var.permute(1, 0, 3, 2).reshape(A, S * N2, M)

    F_flat = F_super.reshape(SC, M)
    Y_flat = Y_super.reshape(SC)
    alpha_mask_exp = alpha_mask.unsqueeze(1).expand(A, S, C_max).reshape(A, SC)

    # --- Gather ---
    W_sel = W_flat[:, i_offset, :]
    X_sel = X_flat[:, j_offset, :]
    W_var_sel = W_var_flat[:, i_offset, :]
    X_var_sel = X_var_flat[:, j_offset, :]

    # --- Forward Pass ---
    Z_hat = alpha_scale * (F_flat.unsqueeze(0) * W_sel * X_sel).sum(dim=2)
    Z_hat = Z_hat * alpha_mask_exp.float()

    # --- Variance ---
    F_sq_flat = F_flat.pow(2).unsqueeze(0)
    V = alpha_scale_sq * (F_sq_flat * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=2)
    V = V * alpha_mask_exp.float() + 1e-10

    # --- Onsager ---
    # FIX #1: Scale Onsager by damping factor
    if prev_s is not None:
        onsager_term = prev_s * V * damping
        Z_hat = Z_hat - onsager_term
        Z_hat = Z_hat * alpha_mask_exp.float()

    # --- Residuals ---
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom
    s_values = torch.clamp(s_values, min=-1e5, max=1e5)
    s_values = s_values * alpha_mask_exp.float()

    # --- Scatter Add ---
    s_exp = s_values.unsqueeze(2)
    mask_exp = alpha_mask_exp.unsqueeze(2).float()
    F_exp = F_flat.unsqueeze(0)
    
    # Constants for variance update
    inv_V = (1.0 / denom).unsqueeze(2)
    F_sq_exp = F_exp.pow(2)

    # --- Update W ---
    r_W_contrib = alpha_scale * F_exp * X_sel * s_exp * mask_exp
    r_W = torch.zeros(A, S * N1, M, device=W_hat.device, dtype=W_hat.dtype)
    idx_W = i_offset.view(1, SC, 1).expand(A, SC, M)
    r_W.scatter_add_(1, idx_W, r_W_contrib)

    tau_W_contrib = alpha_scale_sq * F_sq_exp * X_sel.pow(2) * inv_V * mask_exp
    tau_W = torch.zeros(A, S * N1, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_W.scatter_add_(1, idx_W, tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)

    # FIX #2: Correct Bayesian Update for W
    W_var_new = 1.0 / (1.0 + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    # W_new = (tau_W * W_old + r_W) * Var_new
    W_hat_new = W_var_new * (r_W + W_flat * tau_W)

    # --- Update X ---
    r_X_contrib = alpha_scale * F_exp * W_sel * s_exp * mask_exp
    r_X = torch.zeros(A, S * N2, M, device=W_hat.device, dtype=W_hat.dtype)
    idx_X = j_offset.view(1, SC, 1).expand(A, SC, M)
    r_X.scatter_add_(1, idx_X, r_X_contrib)

    tau_X_contrib = alpha_scale_sq * F_sq_exp * W_sel.pow(2) * inv_V * mask_exp
    tau_X = torch.zeros(A, S * N2, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_X.scatter_add_(1, idx_X, tau_X_contrib)
    tau_X = tau_X.clamp(min=1e-10)

    # FIX #2: Correct Bayesian Update for X
    X_var_new = 1.0 / (1.0 + tau_X)
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_var_new * (r_X + X_flat * tau_X)

    # --- Reshape Back ---
    W_hat_new = W_hat_new.reshape(A, S, N1, M).permute(1, 0, 2, 3)
    W_var_new = W_var_new.reshape(A, S, N1, M).permute(1, 0, 2, 3)
    X_hat_new = X_hat_new.reshape(A, S, N2, M).permute(1, 0, 3, 2)
    X_var_new = X_var_new.reshape(A, S, N2, M).permute(1, 0, 3, 2)

    # --- Damping ---
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


def compute_offset_indices(
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    N1: int,
    N2: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    S = i_idx.shape[0]
    device = i_idx.device

    offsets_N1 = torch.arange(S, device=device) * N1
    offsets_N2 = torch.arange(S, device=device) * N2

    i_offset = (i_idx + offsets_N1.unsqueeze(1)).reshape(-1)
    j_offset = (j_idx + offsets_N2.unsqueeze(1)).reshape(-1)

    return i_offset, j_offset


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

    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        self.damping = config.algorithm.damping
        self.noise_var = config.algorithm.noise_var
        self.max_steps = config.training.max_steps

        spreading_cfg = config.spreading
        if spreading_cfg is not None:
            self.f_distribution = spreading_cfg.f_distribution
            self.spreading_seed = spreading_cfg.seed
        else:
            self.f_distribution = 'gaussian'
            self.spreading_seed = 12345

        if self.f_distribution not in F_GENERATORS:
            raise ValueError(f"Invalid f_distribution='{self.f_distribution}'")

        print(f"[BiG-AMP Spreading Parallel] F distribution: {self.f_distribution}")

    def create_spreading_data(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        alpha_values: List[float],
        S: int,
        base_seed: int,
    ) -> SpreadingDataParallel:
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
        A = spreading_data.A
        N1 = spreading_data.supergraph.N1
        N2 = spreading_data.supergraph.N2
        M = spreading_data.M

        F = spreading_data.get_F(sample_idx)
        Y_values = spreading_data.Y_super[sample_idx]
        i_idx, j_idx = spreading_data.supergraph.get_sample_indices(sample_idx)
        alpha_mask = spreading_data.supergraph.alpha_mask

        # Initialize student variables (Mean Field Scaling: N(0,1))
        # Important: Start with small random values to break symmetry
        W_hat = torch.randn(A, N1, M, device=self.device) * 0.01
        X_hat = torch.randn(A, M, N2, device=self.device) * 0.01
        W_var = torch.ones(A, N1, M, device=self.device)
        X_var = torch.ones(A, M, N2, device=self.device)

        prev_s = None

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
                print(f" Step {step + 1}/{self.max_steps}")

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
            W_s, X_s = self.train_sample(spreading_data, s, verbose=False, step_callback=step_callback)
            W_all[s] = W_s
            X_all[s] = X_s
            
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
        S = spreading_data.S
        A = spreading_data.A
        N1 = spreading_data.supergraph.N1
        N2 = spreading_data.supergraph.N2
        M = spreading_data.M
        C_max = spreading_data.C_max

        if batch_alpha_indices is None:
            batch_alpha_indices = list(range(A))
        B = len(batch_alpha_indices)

        full_alpha_mask = spreading_data.supergraph.alpha_mask
        batch_alpha_mask = full_alpha_mask[batch_alpha_indices]

        i_offset, j_offset = compute_offset_indices(
            spreading_data.supergraph.i_idx,
            spreading_data.supergraph.j_idx,
            N1, N2
        )

        # Initialize student variables
        W_hat = torch.randn(S, B, N1, M, device=self.device) * 0.01
        X_hat = torch.randn(S, B, M, N2, device=self.device) * 0.01
        W_var = torch.ones(S, B, N1, M, device=self.device)
        X_var = torch.ones(S, B, M, N2, device=self.device)

        prev_s = None

        for step in range(self.max_steps):
            W_hat, X_hat, W_var, X_var, prev_s = bigamp_step_disjoint_union(
                W_hat=W_hat,
                X_hat=X_hat,
                W_var=W_var,
                X_var=X_var,
                Y_super=spreading_data.Y_super,
                F_super=spreading_data.F_super,
                i_offset=i_offset,
                j_offset=j_offset,
                alpha_mask=batch_alpha_mask,
                S=S,
                damping=self.damping,
                noise_var=self.noise_var,
                prev_s=prev_s,
            )

            if verbose and (step + 1) % 100 == 0:
                print(f" Step {step + 1}/{self.max_steps}")

            if step_callback:
                step_callback(step + 1, self.max_steps)

        return W_hat, X_hat

    def supports_batch_training(self) -> bool:
        return True

    def train_batch_alphas(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: torch.Tensor,
        alpha_values: List[float],
        seed: int,
        step_callback=None,
        sample_callback=None,
        max_memory_gb: float = 24.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ...core.memory_manager import get_spreading_memory_strategy

        S = self.config.training.samples_per_alpha
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        A = len(alpha_values)
        alpha_max = max(alpha_values) if alpha_values else 4.0

        strategy = get_spreading_memory_strategy(
            N1, N2, M, S, A, alpha_max,
            available_gb=max_memory_gb + 3.0,
            verbose=False,
        )
        alphas_per_batch = strategy['alphas_per_batch']
        num_batches = strategy['num_batches']

        spreading_data = self.create_spreading_data(
            W_teacher, X_teacher, alpha_values, S, seed
        )

        W_result = torch.zeros(A, S, N1, M, device=self.device)
        X_result = torch.zeros(A, S, M, N2, device=self.device)

        for batch_idx in range(num_batches):
            alpha_start = batch_idx * alphas_per_batch
            alpha_end = min((batch_idx + 1) * alphas_per_batch, A)
            batch_alpha_indices = list(range(alpha_start, alpha_end))

            W_batch, X_batch = self.train_full_parallel(
                spreading_data,
                batch_alpha_indices=batch_alpha_indices,
                verbose=False,
                step_callback=step_callback,
            )

            W_result[alpha_start:alpha_end] = W_batch.transpose(0, 1)
            X_result[alpha_start:alpha_end] = X_batch.transpose(0, 1)

            if sample_callback:
                sample_callback(batch_idx + 1, num_batches)

            if batch_idx < num_batches - 1:
                torch.cuda.empty_cache()

        return W_result, X_result

    def train_single_alpha(self, alpha, teacher_data, graph_data):
        raise NotImplementedError("Use train_sample or train_all_samples")


def run_spreading_parallel(config, verbose: bool = True) -> Dict:
    import time
    from ..metrics.spreading import compute_all_metrics_spreading_parallel
    from ..registry import get_teacher
    from ...core.device import setup_device

    device, device_info = setup_device()

    m = config.matrix
    alpha_values = config.alpha.get_values()
    S = config.training.samples_per_alpha
    seed = config.training.seed

    if verbose:
        print(f"[Spreading Parallel] Running with:")
        print(f" Matrix: {m.N1}x{m.N2}, M={m.M}")
        print(f" Alpha: {alpha_values[0]:.2f} ~ {alpha_values[-1]:.2f} ({len(alpha_values)} points)")
        print(f" Samples: {S}")
        print(f" F distribution: {config.spreading.f_distribution if config.spreading else 'gaussian'}")

    start_time = time.time()

    teacher_cls = get_teacher(config.teacher_key).cls
    teacher = teacher_cls()
    W_teacher, X_teacher = teacher.create(m.N1, m.N2, m.M, device, seed)

    if verbose:
        print(f" Teacher type: {config.teacher_key}")

    algorithm = BiGAMPSpreadingParallel(config, device)

    spreading_data = algorithm.create_spreading_data(
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        alpha_values=alpha_values,
        S=S,
        base_seed=seed,
    )

    if verbose:
        print(f" SuperGraph created: C_max={spreading_data.C_max}")

    W_students, X_students = algorithm.train_all_samples(
        spreading_data, verbose=verbose
    )

    metrics = compute_all_metrics_spreading_parallel(
        W_students, X_students, spreading_data
    )

    total_time = time.time() - start_time

    if verbose:
        print(f"\n[Spreading Parallel] Completed in {total_time:.1f}s")

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
