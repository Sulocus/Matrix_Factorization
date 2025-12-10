"""
BiG-AMP (Bilinear Generalized Approximate Message Passing) algorithm.

Supports torch.compile for kernel fusion acceleration (~2-3x speedup).
"""

from typing import Tuple, Optional, Callable
import torch

from ..registry import register_algorithm
from .base import AlgorithmBase
from ...core.config import Config


def _bigamp_step(
    w_hat: torch.Tensor,
    x_hat: torch.Tensor,
    w_var: torch.Tensor,
    x_var: torch.Tensor,
    Y: torch.Tensor,
    A: torch.Tensor,
    alpha_scale: float,
    damping: float,
    noise_var: float,
    M: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single BiG-AMP step. Can be compiled with torch.compile for fusion.

    Args:
        w_hat, x_hat: Student parameter estimates
        w_var, x_var: Variance estimates
        Y: Target matrix (Y_teacher)
        A: Observation mask
        alpha_scale: 1/sqrt(M) scaling factor
        damping: Damping coefficient
        noise_var: Noise variance
        M: Rank (hidden dimension)

    Returns:
        Updated (w_hat, x_hat, w_var, x_var)
    """
    # Forward pass
    z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
    w_sq = w_hat ** 2
    x_sq = x_hat ** 2
    p_var = (alpha_scale ** 2) * (
        torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq)
    )
    V = torch.clamp(p_var + noise_var, min=1e-8)
    residual = (Y - z_hat) * A
    s = residual / V

    # Update W
    tau_W = (alpha_scale ** 2) * torch.matmul(A / V, x_sq.transpose(-2, -1))
    tau_W = torch.clamp(tau_W, min=1e-8)
    w_var_new = 1.0 / (M + tau_W)
    r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
    w_hat_new = w_hat + w_var_new * r_W
    w_hat = damping * w_hat + (1 - damping) * w_hat_new
    w_var = torch.clamp(
        damping * w_var + (1 - damping) * w_var_new,
        min=1e-8, max=1.0
    )

    # Update X
    z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
    w_sq2 = w_hat ** 2
    p_var2 = (alpha_scale ** 2) * (
        torch.matmul(w_sq2, x_var) + torch.matmul(w_var, x_sq)
    )
    V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
    residual2 = (Y - z_hat2) * A
    s2 = residual2 / V2

    tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), A / V2)
    tau_X = torch.clamp(tau_X, min=1e-8)
    x_var_new = 1.0 / (M + tau_X)
    r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
    x_hat_new = x_hat + x_var_new * r_X
    x_hat = damping * x_hat + (1 - damping) * x_hat_new
    x_var = torch.clamp(
        damping * x_var + (1 - damping) * x_var_new,
        min=1e-8, max=1.0
    )

    return w_hat, x_hat, w_var, x_var


@register_algorithm(
    key="bigamp",
    name="BiG-AMP",
    description="Message passing algorithm, fast convergence (~200-5000 steps)",
    default_params={'damping': 0.5, 'noise_var': 1e-10},
)
class BiGAMPAlgorithm(AlgorithmBase):
    """
    BiG-AMP (Bilinear Generalized Approximate Message Passing) algorithm.

    Faster convergence than gradient descent (~200-5000 steps vs 20k+ epochs).
    Uses message passing to iteratively estimate W and X.

    Supports torch.compile for ~2-3x speedup on GPU.
    """

    # Class-level compiled function cache
    _compiled_step: Optional[Callable] = None

    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, device)
        self.damping = config.algorithm.damping
        self.noise_var = config.algorithm.noise_var
        self.max_steps = config.training.max_steps
        self.S = config.training.samples_per_alpha
        self.use_compile = getattr(config.algorithm, 'use_compile', True)

        # Initialize compiled step function if enabled
        if self.use_compile and device.type == 'cuda' and BiGAMPAlgorithm._compiled_step is None:
            try:
                # Use backend='inductor' with options to disable CUDA Graph
                # This avoids tensor overwrite conflicts in training loops
                BiGAMPAlgorithm._compiled_step = torch.compile(
                    _bigamp_step,
                    backend='inductor',
                    options={'triton.cudagraphs': False},
                )
            except Exception as e:
                print(f"[BiG-AMP] torch.compile failed: {e}, using eager mode")
                BiGAMPAlgorithm._compiled_step = _bigamp_step
        elif BiGAMPAlgorithm._compiled_step is None:
            BiGAMPAlgorithm._compiled_step = _bigamp_step

    def train_single_alpha(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        seed: int,
        use_fp16_storage: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train BiG-AMP for single alpha (sequential mode).

        Args:
            W_teacher, X_teacher, Y_teacher: Teacher model tensors
            mask: Observation mask
            alpha: Observation density
            seed: Random seed
            use_fp16_storage: If True, store parameters in FP16 to save memory
                             (extreme mode for very large matrices)
            progress_callback: Optional callback(current_step, total_steps) for progress updates

        Returns:
            Tuple of (W_student, X_student) tensors
        """
        # Warning: α=0 means no observations, student won't learn
        if alpha == 0 or mask.sum() == 0:
            import warnings
            warnings.warn(
                f"α={alpha}: No observations (mask all zeros). "
                "Student will maintain random initialization. "
                "Q_W/Q_X values will be random baseline (~0.1), not converged."
            )

        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.S
        device = self.device

        alpha_scale = 1.0 / (M ** 0.5)
        scale = 1.0 / (M ** 0.5)

        # Determine storage dtype
        storage_dtype = torch.float16 if use_fp16_storage else torch.float32
        compute_dtype = torch.float32  # Always compute in FP32

        # Ensure mask has batch dimension
        A = mask.unsqueeze(0) if mask.dim() == 2 else mask

        # Initialize student (stored in storage_dtype)
        torch.manual_seed(seed)
        w_hat = (torch.randn((S, N1, M), device=device) * scale).to(storage_dtype)
        x_hat = (torch.randn((S, M, N2), device=device) * scale).to(storage_dtype)
        w_var = (torch.ones((S, N1, M), device=device) * (1.0 / M)).to(storage_dtype)
        x_var = (torch.ones((S, M, N2), device=device) * (1.0 / M)).to(storage_dtype)

        # Get step function (compiled or eager)
        step_fn = BiGAMPAlgorithm._compiled_step if self.use_compile else _bigamp_step

        for step in range(self.max_steps):
            # Convert to compute dtype for numerical stability
            w_hat_c = w_hat.to(compute_dtype)
            x_hat_c = x_hat.to(compute_dtype)
            w_var_c = w_var.to(compute_dtype)
            x_var_c = x_var.to(compute_dtype)

            # Execute BiG-AMP step (possibly compiled)
            w_hat_c, x_hat_c, w_var_c, x_var_c = step_fn(
                w_hat_c, x_hat_c, w_var_c, x_var_c,
                Y_teacher, A, alpha_scale,
                self.damping, self.noise_var, M
            )

            # Convert back to storage dtype
            w_hat = w_hat_c.to(storage_dtype)
            x_hat = x_hat_c.to(storage_dtype)
            w_var = w_var_c.to(storage_dtype)
            x_var = x_var_c.to(storage_dtype)

            # Report progress
            if progress_callback:
                progress_callback(step + 1, self.max_steps)

        # Return in FP32 for evaluation
        return w_hat.to(torch.float32), x_hat.to(torch.float32)

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
        """Train BiG-AMP for multiple alphas in parallel."""
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.S
        device = self.device
        num_alphas = len(alpha_values)

        alpha_scale = 1.0 / (M ** 0.5)
        scale = 1.0 / (M ** 0.5)

        # masks: (num_alphas, N1, N2) -> (num_alphas, 1, N1, N2)
        A_all = masks.unsqueeze(1)

        # Initialize student - (num_alphas, S, N1, M)
        torch.manual_seed(seed)
        w_hat = torch.randn((num_alphas, S, N1, M), device=device) * scale
        x_hat = torch.randn((num_alphas, S, M, N2), device=device) * scale
        w_var = torch.ones_like(w_hat) * (1.0 / M)
        x_var = torch.ones_like(x_hat) * (1.0 / M)

        Y_exp = Y_teacher.unsqueeze(0).unsqueeze(0)  # (1, 1, N1, N2)

        # Get step function (compiled or eager)
        step_fn = BiGAMPAlgorithm._compiled_step if self.use_compile else _bigamp_step

        for step in range(self.max_steps):
            # Execute BiG-AMP step (possibly compiled)
            w_hat, x_hat, w_var, x_var = step_fn(
                w_hat, x_hat, w_var, x_var,
                Y_exp, A_all, alpha_scale,
                self.damping, self.noise_var, M
            )

            # Report progress
            if progress_callback:
                progress_callback(step + 1, self.max_steps)

        return w_hat, x_hat

    def supports_batch_training(self) -> bool:
        """BiG-AMP supports efficient batch training."""
        return True

    def estimate_memory_per_alpha(self, N1: int, N2: int, M: int, S: int) -> float:
        """
        Estimate GPU memory needed per alpha value.

        BiG-AMP creates intermediate tensors of shape (batch, S, N1, N2):
        - W update: z_hat, p_var, V, residual, s = 5 tensors
        - X update: z_hat2, p_var2, V2, residual2, s2 = 5 tensors

        With torch.compile: kernel fusion reduces peak to ~5 tensors.
        PyTorch overhead: ~50% for fragmentation and allocator pools.
        """
        # Use centralized memory estimation
        from smf.core.memory_manager import estimate_memory_per_alpha
        return estimate_memory_per_alpha(N1, N2, M, S, use_compile=self.use_compile)
