"""
BiG-AMP with random spreading for disordered matrix factorization.

Implements the message passing algorithm for the random spreading model:
    Y_ij = (1/√M) Σ_μ F_ij,μ W_iμ X_μj

Key differences from standard BiG-AMP:
1. Forward pass includes F weighting
2. Backward pass (residual propagation) includes F weighting
3. Uses sparse operations (scatter_add) for memory efficiency

Memory efficiency:
- Standard BiG-AMP: O(S × N1 × N2) intermediate tensors
- Spreading BiG-AMP: O(C × M) intermediate tensors, where C = α × M × N1
"""

from typing import Tuple, Optional, Callable
import torch

from ..registry import register_algorithm
from .base import AlgorithmBase
from ..teachers.random_spreading import SpreadingData, compute_sparse_Y_batched
from ...core.config import Config


def _scatter_add_2d(
    src: torch.Tensor,
    idx: torch.Tensor,
    dim_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """
    Scatter-add operation for 2D tensors.

    Aggregates values from src into output tensor at positions specified by idx.

    Args:
        src: (C, M) source tensor
        idx: (C,) indices for aggregation
        dim_size: Size of output dimension
        dim: Dimension to scatter (0 for rows, 1 for columns)

    Returns:
        If dim=0: (dim_size, M) tensor
        If dim=1: (M, dim_size) tensor

    Example:
        src = [[1, 2], [3, 4], [5, 6]]  # (3, 2)
        idx = [0, 1, 0]                  # aggregate to rows 0 and 1
        result[0] = [1+5, 2+6] = [6, 8]
        result[1] = [3, 4]
    """
    if dim == 0:
        out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
        idx_expanded = idx.unsqueeze(1).expand_as(src)
        out.scatter_add_(0, idx_expanded, src)
    else:
        # dim == 1: scatter along columns
        out = torch.zeros(src.shape[0], dim_size, device=src.device, dtype=src.dtype)
        idx_expanded = idx.unsqueeze(0).expand_as(src)
        out.scatter_add_(1, idx_expanded, src)
    return out


def _bigamp_spreading_step_single(
    w_hat: torch.Tensor,          # (N1, M)
    x_hat: torch.Tensor,          # (M, N2)
    w_var: torch.Tensor,          # (N1, M)
    x_var: torch.Tensor,          # (M, N2)
    spreading_data: SpreadingData,
    alpha_scale: float,
    damping: float,
    noise_var: float,
    M: int,
    N1: int,
    N2: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single BiG-AMP step with random spreading for one sample.

    Uses sparse operations to avoid N1 × N2 intermediate tensors.

    Math:
        Forward: z_hat[c] = (1/√M) Σ_μ F[c,μ] w_hat[i,μ] x_hat[μ,j]
        Residual: s[c] = (Y[c] - z_hat[c]) / V[c]
        W update: r_W[i,μ] = (1/√M) Σ_{j:(i,j)∈obs} F[ij,μ] s[ij] x_hat[μ,j]
        X update: r_X[μ,j] = (1/√M) Σ_{i:(i,j)∈obs} F[ij,μ] w_hat[i,μ] s[ij]
    """
    C = spreading_data.num_edges
    i_idx = spreading_data.i_idx
    j_idx = spreading_data.j_idx
    F = spreading_data.F              # (C, M)
    Y_values = spreading_data.Y_values  # (C,)

    F_sq = F ** 2  # (C, M) - precompute for efficiency

    # ==================== Forward pass ====================
    # z_hat[c] = (1/√M) Σ_μ F[c,μ] w[i,μ] x[μ,j]
    w_selected = w_hat[i_idx, :]       # (C, M)
    x_selected = x_hat[:, j_idx].T     # (C, M)

    z_hat_values = alpha_scale * (F * w_selected * x_selected).sum(dim=1)  # (C,)

    # ==================== Variance computation ====================
    # p_var[c] = (1/M) Σ_μ F²[c,μ] (w²[i,μ] x_var[μ,j] + w_var[i,μ] x²[μ,j])
    w_sq_sel = w_hat[i_idx, :] ** 2        # (C, M)
    x_sq_sel = x_hat[:, j_idx].T ** 2      # (C, M)
    w_var_sel = w_var[i_idx, :]            # (C, M)
    x_var_sel = x_var[:, j_idx].T          # (C, M)

    p_var_values = (alpha_scale ** 2) * (
        F_sq * (w_sq_sel * x_var_sel + w_var_sel * x_sq_sel)
    ).sum(dim=1)  # (C,)

    V_values = torch.clamp(p_var_values + noise_var, min=1e-8)

    # ==================== Residual ====================
    s_values = (Y_values - z_hat_values) / V_values  # (C,)

    # ==================== Update W ====================
    # τ_W[i,μ] = (1/M) Σ_{j:(i,j)∈obs} F²[ij,μ] × (1/V[ij]) × x²[μ,j]
    inv_V = 1.0 / V_values  # (C,)
    tau_W_contrib = F_sq * inv_V.unsqueeze(1) * x_sq_sel  # (C, M)
    tau_W = (alpha_scale ** 2) * _scatter_add_2d(tau_W_contrib, i_idx, N1, dim=0)
    tau_W = torch.clamp(tau_W, min=1e-8)

    w_var_new = 1.0 / (M + tau_W)  # (N1, M)

    # r_W[i,μ] = (1/√M) Σ_{j:(i,j)∈obs} F[ij,μ] × s[ij] × x[μ,j]
    r_W_contrib = F * s_values.unsqueeze(1) * x_selected  # (C, M)
    r_W = alpha_scale * _scatter_add_2d(r_W_contrib, i_idx, N1, dim=0)

    w_hat_new = w_hat + w_var_new * r_W

    # Apply damping
    w_hat = damping * w_hat + (1 - damping) * w_hat_new
    w_var = torch.clamp(
        damping * w_var + (1 - damping) * w_var_new,
        min=1e-8, max=1.0
    )

    # ==================== Update X (using updated W) ====================
    # Recompute with updated W
    w_selected2 = w_hat[i_idx, :]
    w_sq_sel2 = w_selected2 ** 2

    z_hat_values2 = alpha_scale * (F * w_selected2 * x_selected).sum(dim=1)

    p_var_values2 = (alpha_scale ** 2) * (
        F_sq * (w_sq_sel2 * x_var_sel + w_var[i_idx, :] * x_sq_sel)
    ).sum(dim=1)

    V_values2 = torch.clamp(p_var_values2 + noise_var, min=1e-8)
    s_values2 = (Y_values - z_hat_values2) / V_values2

    # τ_X[μ,j] = (1/M) Σ_{i:(i,j)∈obs} F²[ij,μ] × (1/V[ij]) × w²[i,μ]
    inv_V2 = 1.0 / V_values2
    tau_X_contrib = F_sq * inv_V2.unsqueeze(1) * w_sq_sel2  # (C, M)
    # Need to scatter to (M, N2), so transpose and scatter along dim 1
    tau_X = (alpha_scale ** 2) * _scatter_add_2d(tau_X_contrib.T, j_idx, N2, dim=1)
    tau_X = torch.clamp(tau_X, min=1e-8)

    x_var_new = 1.0 / (M + tau_X)  # (M, N2)

    # r_X[μ,j] = (1/√M) Σ_{i:(i,j)∈obs} F[ij,μ] × w[i,μ] × s[ij]
    r_X_contrib = F * s_values2.unsqueeze(1) * w_selected2  # (C, M)
    r_X = alpha_scale * _scatter_add_2d(r_X_contrib.T, j_idx, N2, dim=1)

    x_hat_new = x_hat + x_var_new * r_X

    # Apply damping
    x_hat = damping * x_hat + (1 - damping) * x_hat_new
    x_var = torch.clamp(
        damping * x_var + (1 - damping) * x_var_new,
        min=1e-8, max=1.0
    )

    return w_hat, x_hat, w_var, x_var


def _bigamp_spreading_step_batched(
    w_hat: torch.Tensor,          # (S, N1, M)
    x_hat: torch.Tensor,          # (S, M, N2)
    w_var: torch.Tensor,          # (S, N1, M)
    x_var: torch.Tensor,          # (S, M, N2)
    spreading_data: SpreadingData,
    alpha_scale: float,
    damping: float,
    noise_var: float,
    M: int,
    N1: int,
    N2: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BiG-AMP step with random spreading for batched samples.

    Processes S samples in parallel for better GPU utilization.
    """
    S = w_hat.shape[0]
    C = spreading_data.num_edges
    i_idx = spreading_data.i_idx
    j_idx = spreading_data.j_idx
    F = spreading_data.F              # (C, M)
    Y_values = spreading_data.Y_values  # (C,)

    F_sq = F ** 2  # (C, M)

    # Expand F for batch processing: (1, C, M)
    F_exp = F.unsqueeze(0)
    F_sq_exp = F_sq.unsqueeze(0)
    Y_exp = Y_values.unsqueeze(0)  # (1, C)

    # ==================== Forward pass ====================
    # w_selected: (S, C, M)
    w_selected = w_hat[:, i_idx, :]
    # x_selected: (S, C, M)
    x_selected = x_hat[:, :, j_idx].transpose(1, 2)

    # z_hat_values: (S, C)
    z_hat_values = alpha_scale * (F_exp * w_selected * x_selected).sum(dim=2)

    # ==================== Variance computation ====================
    w_sq_sel = w_selected ** 2        # (S, C, M)
    x_sq_sel = x_selected ** 2        # (S, C, M)
    w_var_sel = w_var[:, i_idx, :]    # (S, C, M)
    x_var_sel = x_var[:, :, j_idx].transpose(1, 2)  # (S, C, M)

    p_var_values = (alpha_scale ** 2) * (
        F_sq_exp * (w_sq_sel * x_var_sel + w_var_sel * x_sq_sel)
    ).sum(dim=2)  # (S, C)

    V_values = torch.clamp(p_var_values + noise_var, min=1e-8)

    # ==================== Residual ====================
    s_values = (Y_exp - z_hat_values) / V_values  # (S, C)

    # ==================== Update W (per sample) ====================
    inv_V = 1.0 / V_values  # (S, C)

    # Process each sample (scatter_add doesn't support batch dim)
    w_hat_new = torch.zeros_like(w_hat)
    w_var_new = torch.zeros_like(w_var)

    for s in range(S):
        tau_W_contrib = F_sq * inv_V[s].unsqueeze(1) * x_sq_sel[s]  # (C, M)
        tau_W = (alpha_scale ** 2) * _scatter_add_2d(tau_W_contrib, i_idx, N1, dim=0)
        tau_W = torch.clamp(tau_W, min=1e-8)
        w_var_new[s] = 1.0 / (M + tau_W)

        r_W_contrib = F * s_values[s].unsqueeze(1) * x_selected[s]  # (C, M)
        r_W = alpha_scale * _scatter_add_2d(r_W_contrib, i_idx, N1, dim=0)
        w_hat_new[s] = w_hat[s] + w_var_new[s] * r_W

    # Apply damping
    w_hat = damping * w_hat + (1 - damping) * w_hat_new
    w_var = torch.clamp(
        damping * w_var + (1 - damping) * w_var_new,
        min=1e-8, max=1.0
    )

    # ==================== Update X (using updated W) ====================
    w_selected2 = w_hat[:, i_idx, :]
    w_sq_sel2 = w_selected2 ** 2

    z_hat_values2 = alpha_scale * (F_exp * w_selected2 * x_selected).sum(dim=2)

    w_var_sel2 = w_var[:, i_idx, :]
    p_var_values2 = (alpha_scale ** 2) * (
        F_sq_exp * (w_sq_sel2 * x_var_sel + w_var_sel2 * x_sq_sel)
    ).sum(dim=2)

    V_values2 = torch.clamp(p_var_values2 + noise_var, min=1e-8)
    s_values2 = (Y_exp - z_hat_values2) / V_values2

    inv_V2 = 1.0 / V_values2

    x_hat_new = torch.zeros_like(x_hat)
    x_var_new = torch.zeros_like(x_var)

    for s in range(S):
        tau_X_contrib = F_sq * inv_V2[s].unsqueeze(1) * w_sq_sel2[s]  # (C, M)
        tau_X = (alpha_scale ** 2) * _scatter_add_2d(tau_X_contrib.T, j_idx, N2, dim=1)
        tau_X = torch.clamp(tau_X, min=1e-8)
        x_var_new[s] = 1.0 / (M + tau_X)

        r_X_contrib = F * s_values2[s].unsqueeze(1) * w_selected2[s]  # (C, M)
        r_X = alpha_scale * _scatter_add_2d(r_X_contrib.T, j_idx, N2, dim=1)
        x_hat_new[s] = x_hat[s] + x_var_new[s] * r_X

    # Apply damping
    x_hat = damping * x_hat + (1 - damping) * x_hat_new
    x_var = torch.clamp(
        damping * x_var + (1 - damping) * x_var_new,
        min=1e-8, max=1.0
    )

    return w_hat, x_hat, w_var, x_var


@register_algorithm(
    key="bigamp_spreading",
    name="BiG-AMP Random Spreading",
    description="BiG-AMP with random spreading F for disordered model",
    default_params={'damping': 0.5, 'noise_var': 1e-10},
)
class BiGAMPSpreadingAlgorithm(AlgorithmBase):
    """
    BiG-AMP algorithm for random spreading model.

    Uses sparse operations to handle spreading coefficients F
    without storing full N1 × N2 intermediate tensors.

    Memory complexity: O(C × M) instead of O(N1 × N2)
    where C = number of observed edges

    Usage:
        algorithm = BiGAMPSpreadingAlgorithm(config, device)

        # Create spreading data from teacher
        W_t, X_t, spreading_data = teacher.create_with_spreading(...)

        # Train
        W_s, X_s = algorithm.train_single_alpha_spreading(
            W_t, X_t, spreading_data, alpha, seed
        )
    """

    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, device)
        self.damping = config.algorithm.damping
        self.noise_var = config.algorithm.noise_var
        self.max_steps = config.training.max_steps
        self.S = config.training.samples_per_alpha

    def train_single_alpha_spreading(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        spreading_data: SpreadingData,
        alpha: float,
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train BiG-AMP with random spreading for single alpha.

        Args:
            W_teacher: (N1, M) Teacher W matrix
            X_teacher: (M, N2) Teacher X matrix
            spreading_data: SpreadingData containing F and Y_values
            alpha: Observation density (for logging/reference only)
            seed: Random seed for student initialization
            progress_callback: Optional callback(current_step, total_steps)

        Returns:
            W_student: (S, N1, M) Student W estimates
            X_student: (S, M, N2) Student X estimates
        """
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.S
        device = self.device

        alpha_scale = 1.0 / (M ** 0.5)
        scale = 1.0 / (M ** 0.5)

        # Ensure spreading_data is on correct device
        spreading_data = spreading_data.to(device)

        # Initialize student
        torch.manual_seed(seed)
        w_hat = torch.randn((S, N1, M), device=device) * scale
        x_hat = torch.randn((S, M, N2), device=device) * scale
        w_var = torch.ones((S, N1, M), device=device) * (1.0 / M)
        x_var = torch.ones((S, M, N2), device=device) * (1.0 / M)

        for step in range(self.max_steps):
            w_hat, x_hat, w_var, x_var = _bigamp_spreading_step_batched(
                w_hat, x_hat, w_var, x_var,
                spreading_data,
                alpha_scale, self.damping, self.noise_var, M, N1, N2
            )

            if progress_callback:
                progress_callback(step + 1, self.max_steps)

        return w_hat, x_hat

    def train_single_sample_spreading(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        spreading_data: SpreadingData,
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train single sample (S=1) for memory-constrained scenarios.

        Args:
            W_teacher: (N1, M) Teacher W
            X_teacher: (M, N2) Teacher X
            spreading_data: SpreadingData
            seed: Random seed
            progress_callback: Optional callback

        Returns:
            W_student: (N1, M) Single student W
            X_student: (M, N2) Single student X
        """
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        device = self.device

        alpha_scale = 1.0 / (M ** 0.5)
        scale = 1.0 / (M ** 0.5)

        spreading_data = spreading_data.to(device)

        # Initialize single sample
        torch.manual_seed(seed)
        w_hat = torch.randn((N1, M), device=device) * scale
        x_hat = torch.randn((M, N2), device=device) * scale
        w_var = torch.ones((N1, M), device=device) * (1.0 / M)
        x_var = torch.ones((M, N2), device=device) * (1.0 / M)

        for step in range(self.max_steps):
            w_hat, x_hat, w_var, x_var = _bigamp_spreading_step_single(
                w_hat, x_hat, w_var, x_var,
                spreading_data,
                alpha_scale, self.damping, self.noise_var, M, N1, N2
            )

            if progress_callback:
                progress_callback(step + 1, self.max_steps)

        return w_hat, x_hat

    def train_single_alpha(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        seed: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard interface - falls back to regular BiG-AMP.

        For random spreading, use train_single_alpha_spreading() instead.

        This method is provided for compatibility with the standard
        algorithm interface.
        """
        from .bigamp import BiGAMPAlgorithm
        regular = BiGAMPAlgorithm(self.config, self.device)
        return regular.train_single_alpha(
            W_teacher, X_teacher, Y_teacher, mask, alpha, seed, **kwargs
        )

    def train_batch_alphas(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: torch.Tensor,
        alpha_values: list[float],
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train BiG-AMP with random spreading for multiple alphas.

        For random spreading, each alpha requires its own SpreadingData
        (different observation positions → different F coefficients).
        This method loops over alphas sequentially.

        Args:
            W_teacher: (N1, M) Teacher W matrix
            X_teacher: (M, N2) Teacher X matrix
            Y_teacher: (N1, N2) Teacher Y matrix (used for mask positions only)
            masks: (num_alphas, N1, N2) Observation masks for each alpha
            alpha_values: List of alpha values
            seed: Base random seed
            progress_callback: Optional callback(current_step, total_steps)

        Returns:
            W_students: (num_alphas, S, N1, M) Student W estimates
            X_students: (num_alphas, S, M, N2) Student X estimates
        """
        from ..teachers.random_spreading import (
            generate_spreading_coefficients,
            compute_sparse_Y,
            SpreadingData,
        )

        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.S
        device = self.device
        num_alphas = len(alpha_values)

        # Storage for all results
        W_students = []
        X_students = []

        # Total steps for progress tracking
        total_steps = self.max_steps * num_alphas
        global_step = 0

        for alpha_idx, alpha in enumerate(alpha_values):
            mask = masks[alpha_idx]  # (N1, N2)

            # Get observed positions from mask
            i_idx, j_idx = torch.where(mask > 0)
            C = i_idx.shape[0]

            # Generate spreading coefficients for this alpha
            # Use alpha-specific seed for reproducibility
            alpha_seed = seed + int(alpha * 1000)
            F = generate_spreading_coefficients(
                i_idx, j_idx, M, alpha_seed, device
            )

            # Compute Y values at observed positions
            Y_values = compute_sparse_Y(W_teacher, X_teacher, i_idx, j_idx, F)

            # Create SpreadingData
            spreading_data = SpreadingData(
                i_idx=i_idx,
                j_idx=j_idx,
                F=F,
                Y_values=Y_values,
                seed=alpha_seed,
                M=M,
            )

            # Per-alpha progress callback
            def alpha_progress(step, max_steps):
                nonlocal global_step
                global_step = alpha_idx * self.max_steps + step
                if progress_callback:
                    progress_callback(global_step, total_steps)

            # Train for this alpha
            W_s, X_s = self.train_single_alpha_spreading(
                W_teacher, X_teacher, spreading_data, alpha,
                seed + alpha_idx * 10000,
                progress_callback=alpha_progress,
            )

            W_students.append(W_s)
            X_students.append(X_s)

        # Stack results: (num_alphas, S, N1, M) and (num_alphas, S, M, N2)
        W_students = torch.stack(W_students, dim=0)
        X_students = torch.stack(X_students, dim=0)

        return W_students, X_students

    def supports_batch_training(self) -> bool:
        """BiG-AMP spreading supports batch training."""
        return True

    def estimate_memory_per_alpha(self, N1: int, N2: int, M: int, S: int) -> float:
        """
        Estimate GPU memory needed.

        Unlike standard BiG-AMP, spreading version uses O(C × M) not O(N1 × N2).
        """
        # Student parameters: 4 tensors of shape (S, N1, M) or (S, M, N2)
        param_memory = 4 * S * M * (N1 + N2) * 4  # 4 bytes per float32

        # Spreading data: F is (C, M), Y_values is (C,), indices are (C,)
        # C ≈ alpha * M * N1 in typical usage
        # Estimate C as 2 * M * N1 (assuming alpha ≈ 2)
        C_estimate = 2 * M * N1
        spreading_memory = (C_estimate * M + C_estimate * 3) * 4

        # Intermediate tensors in step function: O(C × M)
        intermediate_memory = 6 * C_estimate * M * 4

        total_bytes = param_memory + spreading_memory + intermediate_memory
        return total_bytes / (1024 ** 3)  # Return in GB
