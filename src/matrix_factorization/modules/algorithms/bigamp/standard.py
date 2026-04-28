"""
BiG-AMP (Bilinear Generalized Approximate Message Passing) algorithm.

Supports torch.compile for kernel fusion acceleration (~2-3x speedup).
"""

import hashlib
import math
from typing import Tuple, Optional, Callable
import torch

from ...registry import register_algorithm
from ..base import AlgorithmBase
from ....core.config import Config
from ....core.contracts import AlgorithmStateView
from ....core.experiment.config import resolve_normalization_profile
from .conventions import blend_new_old, gaussian_posterior_update


def _stable_partition_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


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
    prior_precision_base: float,
    prior_variance: float,
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
        prior_precision_base: Gaussian prior precision, 1 for paper profile
        prior_variance: Gaussian prior variance, 1 for paper profile

    Returns:
        Updated (w_hat, x_hat, w_var, x_var)
    """
    # Forward pass
    z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
    w_sq = w_hat ** 2
    x_sq = x_hat ** 2
    z_var = (alpha_scale ** 2) * (
        torch.matmul(w_sq, x_var) + torch.matmul(w_var, x_sq)
    )
    cross_var = (alpha_scale ** 2) * torch.matmul(w_var, x_var)
    p_var = torch.clamp(z_var + cross_var, min=1e-10)
    V = torch.clamp(p_var + noise_var, min=1e-8)
    residual = (Y - z_hat) * A
    s = residual / V
    svar = A / V

    # Update W
    tau_W = (alpha_scale ** 2) * torch.matmul(svar, x_sq.transpose(-2, -1))
    gain_W = (alpha_scale ** 2) * torch.matmul(svar, x_var.transpose(-2, -1))
    tau_W = torch.clamp(tau_W, min=1e-8)
    r_W = alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
    w_hat_new, w_var_new = gaussian_posterior_update(
        w_hat, r_W, tau_W, prior_precision_base, prior_variance, gain_W
    )
    w_hat = blend_new_old(w_hat_new, w_hat, damping)
    w_var = torch.clamp(blend_new_old(w_var_new, w_var, damping), min=1e-8, max=prior_variance)

    # Update X
    z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
    w_sq2 = w_hat ** 2
    z_var2 = (alpha_scale ** 2) * (
        torch.matmul(w_sq2, x_var) + torch.matmul(w_var, x_sq)
    )
    cross_var2 = (alpha_scale ** 2) * torch.matmul(w_var, x_var)
    p_var2 = torch.clamp(z_var2 + cross_var2, min=1e-10)
    V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
    residual2 = (Y - z_hat2) * A
    s2 = residual2 / V2
    svar2 = A / V2

    tau_X = (alpha_scale ** 2) * torch.matmul(w_sq2.transpose(-2, -1), svar2)
    gain_X = (alpha_scale ** 2) * torch.matmul(w_var.transpose(-2, -1), svar2)
    tau_X = torch.clamp(tau_X, min=1e-8)
    r_X = alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
    x_hat_new, x_var_new = gaussian_posterior_update(
        x_hat, r_X, tau_X, prior_precision_base, prior_variance, gain_X
    )
    x_hat = blend_new_old(x_hat_new, x_hat, damping)
    x_var = torch.clamp(blend_new_old(x_var_new, x_var, damping), min=1e-8, max=prior_variance)

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
        algorithm_params = getattr(config, "algorithm_params", getattr(config, "algorithm", None))
        if algorithm_params is None:
            raise AttributeError("BiGAMPAlgorithm requires config.algorithm_params")
        self.algorithm_params = algorithm_params
        self.damping = algorithm_params.damping
        self.noise_var = algorithm_params.noise_var
        self.max_steps = config.training.max_steps
        self.S = config.training.samples_per_alpha
        self.normalization_profile = getattr(algorithm_params, "normalization_profile", "paper_sparse_sampling")
        self._norm = resolve_normalization_profile(self.normalization_profile, config.matrix.M)
        self.seed_partition_policy = getattr(algorithm_params, 'seed_partition_policy', 'legacy')
        if self.seed_partition_policy not in {'legacy', 'partition_invariant'}:
            raise ValueError(
                "algorithm_params.seed_partition_policy must be 'legacy' or 'partition_invariant', "
                f"got {self.seed_partition_policy!r}"
            )
        self.requested_use_tf32 = getattr(algorithm_params, 'use_tf32', True)
        torch.backends.cuda.matmul.allow_tf32 = bool(self.requested_use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(self.requested_use_tf32)
        self.precision_profile = getattr(algorithm_params, 'precision_profile', 'fast' if getattr(algorithm_params, 'use_bf16', False) else 'safe')
        if self.precision_profile == 'fast' and getattr(algorithm_params, 'use_bf16', True) is False:
            self.precision_profile = 'safe'
        self.precision_fallback_policy = getattr(algorithm_params, 'precision_fallback_policy', 'allow')
        if self.precision_fallback_policy == 'allow' and getattr(algorithm_params, 'dtype_fallback_policy', 'allow') == 'error':
            self.precision_fallback_policy = 'error'
        self.requested_use_bf16 = self.precision_profile in {'fast', 'aggressive'}
        bf16_supported = bool(device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        self.use_bf16 = self.requested_use_bf16 and bf16_supported
        if self.requested_use_bf16 and not self.use_bf16 and self.precision_fallback_policy == 'error':
            raise RuntimeError(
                "BF16 precision profile was requested but is unavailable and "
                "algorithm_params.precision_fallback_policy='error'"
            )
        self.storage_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        self.requested_use_compile = getattr(algorithm_params, 'use_compile', True)
        self.compile_fallback_policy = getattr(algorithm_params, 'compile_fallback_policy', 'allow')
        if self.compile_fallback_policy not in {'allow', 'error'}:
            raise ValueError(
                "algorithm_params.compile_fallback_policy must be 'allow' or 'error', "
                f"got {self.compile_fallback_policy!r}"
            )
        self.use_compile = bool(self.requested_use_compile)
        self.compile_attempts = []

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
                self._record_compile_attempt("bigamp_step", True, "")
            except Exception as e:
                self.use_compile = False
                self._record_compile_attempt("bigamp_step", False, type(e).__name__)
                self._handle_compile_failure("bigamp_step", e)
                BiGAMPAlgorithm._compiled_step = _bigamp_step
        elif BiGAMPAlgorithm._compiled_step is None:
            BiGAMPAlgorithm._compiled_step = _bigamp_step
            if self.use_compile and device.type != 'cuda':
                self.use_compile = False

        self._contract_execution_metadata = {
            "path": "bigamp_matrix",
            "normalization_schema_version": self._norm.schema_version,
            "normalization_profile": self.normalization_profile,
            "normalization_convention": self._norm.convention_label,
            "precision_profile": self.precision_profile,
            "precision_fallback_policy": self.precision_fallback_policy,
            "requested_use_bf16": bool(self.requested_use_bf16),
            "effective_use_bf16": bool(self.use_bf16),
            "storage_dtype": str(self.storage_dtype).replace("torch.", ""),
            "requested_use_compile": bool(self.requested_use_compile),
            "effective_use_compile": bool(
                self.use_compile and BiGAMPAlgorithm._compiled_step is not _bigamp_step
            ),
            "compile_fallback_policy": self.compile_fallback_policy,
            "compile_status": self._compile_status(),
            "compile_attempts": list(self.compile_attempts),
            "requested_use_tf32": bool(self.requested_use_tf32),
            "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
            "seed_partition_policy": self.seed_partition_policy,
            "metadata_only": True,
        }

    def _uses_partition_invariant_seed_policy(self) -> bool:
        return self.seed_partition_policy == "partition_invariant"

    def _randn_partitioned_matrix(
        self,
        *,
        alpha_values: list[float],
        sample_count: int,
        shape: Tuple[int, ...],
        seed: int,
        device: torch.device,
        scale: float,
        role: str,
        dtype: torch.dtype = torch.float32,
        mean: float = 0.0,
    ) -> torch.Tensor:
        alpha_blocks = []
        for alpha in alpha_values:
            sample_blocks = []
            alpha_token = f"{float(alpha):.12g}"
            for sample_idx in range(sample_count):
                gen = torch.Generator(device=device).manual_seed(
                    _stable_partition_seed(seed, "bigamp", role, alpha_token, sample_idx)
                )
                sample_blocks.append(
                    torch.randn(shape, generator=gen, device=device, dtype=dtype) * scale + mean
                )
            alpha_blocks.append(torch.stack(sample_blocks, dim=0))
        return torch.stack(alpha_blocks, dim=0)

    def _student_init_mean(self, scale: float) -> float:
        teacher_cfg = getattr(self.config, "teacher", None)
        if getattr(teacher_cfg, "init_distribution", "gaussian") != "biased_gaussian":
            return 0.0
        return float(getattr(teacher_cfg, "mean_scale", 0.0)) * float(scale)

    def _teacher_assisted_initialization(
        self,
        teacher: torch.Tensor,
        *,
        target_shape: Tuple[int, ...],
        init_overlap: float,
        seed: int,
        scale: float,
        role: str,
        dtype: torch.dtype = torch.float32,
        alpha_values: Optional[list[float]] = None,
    ) -> torch.Tensor:
        """Initialize student factors with a prescribed teacher projection."""

        rho = float(init_overlap)
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"algorithm_params.init_overlap must be in [0, 1], got {rho}")
        coeff_noise = math.sqrt(max(0.0, 1.0 - rho * rho))
        if self._uses_partition_invariant_seed_policy() and alpha_values is not None:
            has_alpha_axis = len(target_shape) == len(tuple(teacher.shape)) + 2
            sample_axis = 1 if has_alpha_axis else 0
            factor_shape = tuple(target_shape[sample_axis + 1:])
            noise = self._randn_partitioned_matrix(
                alpha_values=alpha_values,
                sample_count=int(target_shape[sample_axis]),
                shape=factor_shape,
                seed=seed,
                device=self.device,
                scale=scale,
                role=f"{role}_warm_start_noise",
                dtype=dtype,
            )
            if not has_alpha_axis:
                noise = noise[0]
        else:
            torch.manual_seed(seed)
            noise = torch.randn(target_shape, device=self.device, dtype=dtype) * scale
        teacher_view = teacher.to(device=self.device, dtype=dtype)
        while teacher_view.dim() < len(target_shape):
            teacher_view = teacher_view.unsqueeze(0)
        return rho * teacher_view.expand(target_shape) + coeff_noise * noise

    def _record_compile_attempt(self, target: str, success: bool, error: str) -> None:
        self.compile_attempts.append({
            "target": target,
            "success": bool(success),
            "error": error,
        })

    def _handle_compile_failure(self, target: str, exc: Exception) -> None:
        if self.compile_fallback_policy == 'error':
            raise RuntimeError(
                f"torch.compile failed for {target} and "
                "algorithm_params.compile_fallback_policy='error'"
            ) from exc

    def _compile_status(self) -> str:
        if not bool(getattr(self, "requested_use_compile", True)):
            return "disabled_by_config"
        if getattr(self, "device", None) is not None and self.device.type != 'cuda':
            return "fallback_to_eager_non_cuda"
        if bool(getattr(self, "use_compile", False)) and BiGAMPAlgorithm._compiled_step is not _bigamp_step:
            return "effective_for_bigamp_step"
        return "fallback_to_eager_bigamp_step"

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

        norm = resolve_normalization_profile(self.normalization_profile, M)
        alpha_scale = norm.interaction_scale
        scale = norm.student_init_std
        prior_precision_base = norm.prior_precision_base
        prior_variance = norm.prior_variance
        init_mean = self._student_init_mean(scale)

        # Determine storage dtype
        storage_dtype = torch.float16 if use_fp16_storage else torch.float32
        compute_dtype = torch.float32  # Always compute in FP32

        # Ensure mask has batch dimension
        A = mask.unsqueeze(0) if mask.dim() == 2 else mask

        # Handle missing Y_teacher
        if Y_teacher is None:
            w_t_c = W_teacher.to(compute_dtype)
            x_t_c = X_teacher.to(compute_dtype)
            
            # W: (N1, M) or (S, N1, M)
            # If teacher is shared (usual case), it lacks S dim.
            if w_t_c.dim() == 2:
                z_t = (w_t_c @ x_t_c) * alpha_scale
            else:
                 # Batch matmul if teacher varies per student (unlikely here but possible)
                z_t = torch.bmm(w_t_c, x_t_c) * alpha_scale
                
            # A is (S, N1, N2) or (1, N1, N2)
            # Broadcast Z to match A's potential batch dim if needed, usually just (N1, N2)
            if mask.dim() == 2:
                y_raw = z_t * mask
            else:
                y_raw = z_t * mask # Broadcasting should work
                
            Y_teacher = y_raw
            if A.dim() == 3 and Y_teacher.dim() == 2:
                 Y_teacher = Y_teacher.unsqueeze(0)

        # Initialize student (stored in storage_dtype)
        init_mode = getattr(self.algorithm_params, "init_mode", "random")
        init_overlap = getattr(self.algorithm_params, "init_overlap", 0.95)
        if init_mode == "teacher":
            w_hat = self._teacher_assisted_initialization(
                W_teacher,
                target_shape=(S, N1, M),
                init_overlap=init_overlap,
                seed=seed,
                scale=scale,
                role="W_student",
                dtype=storage_dtype,
                alpha_values=[alpha],
            )
            x_hat = self._teacher_assisted_initialization(
                X_teacher,
                target_shape=(S, M, N2),
                init_overlap=init_overlap,
                seed=seed + 1,
                scale=scale,
                role="X_student",
                dtype=storage_dtype,
                alpha_values=[alpha],
            )
        elif self._uses_partition_invariant_seed_policy():
            w_hat = self._randn_partitioned_matrix(
                alpha_values=[alpha],
                sample_count=S,
                shape=(N1, M),
                seed=seed,
                device=device,
                scale=scale,
                role="W_student",
                dtype=storage_dtype,
                mean=init_mean,
            )[0]
            x_hat = self._randn_partitioned_matrix(
                alpha_values=[alpha],
                sample_count=S,
                shape=(M, N2),
                seed=seed,
                device=device,
                scale=scale,
                role="X_student",
                dtype=storage_dtype,
                mean=init_mean,
            )[0]
        else:
            torch.manual_seed(seed)
            w_hat = (torch.randn((S, N1, M), device=device) * scale + init_mean).to(storage_dtype)
            x_hat = (torch.randn((S, M, N2), device=device) * scale + init_mean).to(storage_dtype)
        w_var = (torch.ones((S, N1, M), device=device) * prior_variance).to(storage_dtype)
        x_var = (torch.ones((S, M, N2), device=device) * prior_variance).to(storage_dtype)

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
                self.damping, self.noise_var, M,
                prior_precision_base, prior_variance
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
        max_steps: Optional[int] = None,  # Allow override for step scanning
        progress_callback: Optional[Callable[[int, int], None]] = None,
        initial_state: Optional[AlgorithmStateView] = None,
        return_continuation_state: bool = False,
        continuation_context: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train BiG-AMP for multiple alphas in parallel."""
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.S
        device = self.device
        num_alphas = len(alpha_values)
        
        # Use provided max_steps or fall back to config
        steps = max_steps if max_steps is not None else self.max_steps

        norm = resolve_normalization_profile(self.normalization_profile, M)
        alpha_scale = norm.interaction_scale
        scale = norm.student_init_std
        prior_precision_base = norm.prior_precision_base
        prior_variance = norm.prior_variance
        init_mean = self._student_init_mean(scale)

        # masks: (num_alphas, N1, N2) -> (num_alphas, 1, N1, N2)
        A_all = masks.unsqueeze(1)

        # Initialize student - (num_alphas, S, N1, M)
        init_mode = getattr(self.algorithm_params, "init_mode", "random")
        init_overlap = getattr(self.algorithm_params, "init_overlap", 0.95)
        if init_mode == "teacher":
            w_hat = self._teacher_assisted_initialization(
                W_teacher,
                target_shape=(num_alphas, S, N1, M),
                init_overlap=init_overlap,
                seed=seed,
                scale=scale,
                role="W_student",
                dtype=self.storage_dtype,
                alpha_values=alpha_values,
            )
            x_hat = self._teacher_assisted_initialization(
                X_teacher,
                target_shape=(num_alphas, S, M, N2),
                init_overlap=init_overlap,
                seed=seed + 1,
                scale=scale,
                role="X_student",
                dtype=self.storage_dtype,
                alpha_values=alpha_values,
            )
        elif self._uses_partition_invariant_seed_policy():
            w_hat = self._randn_partitioned_matrix(
                alpha_values=alpha_values,
                sample_count=S,
                shape=(N1, M),
                seed=seed,
                device=device,
                scale=scale,
                role="W_student",
                dtype=self.storage_dtype,
                mean=init_mean,
            )
            x_hat = self._randn_partitioned_matrix(
                alpha_values=alpha_values,
                sample_count=S,
                shape=(M, N2),
                seed=seed,
                device=device,
                scale=scale,
                role="X_student",
                dtype=self.storage_dtype,
                mean=init_mean,
            )
        else:
            torch.manual_seed(seed)
            w_hat = torch.randn((num_alphas, S, N1, M), device=device, dtype=self.storage_dtype) * scale + init_mean
            x_hat = torch.randn((num_alphas, S, M, N2), device=device, dtype=self.storage_dtype) * scale + init_mean
        w_var = torch.ones_like(w_hat) * prior_variance
        x_var = torch.ones_like(x_hat) * prior_variance

        if initial_state is not None:
            factors = initial_state.student_factors or {}
            variances = initial_state.factor_variances or {}
            if "W" in factors and "X" in factors:
                w_hat = self._coerce_continuation_factor(
                    factors["W"], (num_alphas, S, N1, M), device, self.storage_dtype
                )
                x_hat = self._coerce_continuation_factor(
                    factors["X"], (num_alphas, S, M, N2), device, self.storage_dtype
                )
                if "W_var" in variances:
                    w_var = self._coerce_continuation_factor(
                        variances["W_var"], (num_alphas, S, N1, M), device, self.storage_dtype
                    )
                if "X_var" in variances:
                    x_var = self._coerce_continuation_factor(
                        variances["X_var"], (num_alphas, S, M, N2), device, self.storage_dtype
                    )

        Y_exp = Y_teacher.unsqueeze(0).unsqueeze(0)  # (1, 1, N1, N2)

        # Get step function (compiled or eager)
        step_fn = BiGAMPAlgorithm._compiled_step if self.use_compile else _bigamp_step

        for step in range(steps):
            w_hat_c = w_hat.to(torch.float32)
            x_hat_c = x_hat.to(torch.float32)
            w_var_c = w_var.to(torch.float32)
            x_var_c = x_var.to(torch.float32)
            # Execute BiG-AMP step (possibly compiled)
            w_hat_c, x_hat_c, w_var_c, x_var_c = step_fn(
                w_hat_c, x_hat_c, w_var_c, x_var_c,
                Y_exp, A_all, alpha_scale,
                self.damping, self.noise_var, M,
                prior_precision_base, prior_variance
            )
            if self.use_bf16:
                w_hat = w_hat_c.to(self.storage_dtype)
                x_hat = x_hat_c.to(self.storage_dtype)
                w_var = w_var_c.to(self.storage_dtype)
                x_var = x_var_c.to(self.storage_dtype)
            else:
                w_hat, x_hat, w_var, x_var = w_hat_c, x_hat_c, w_var_c, x_var_c

            # Report progress
            if progress_callback:
                progress_callback(step + 1, steps)

        W_out = w_hat.to(torch.float32)
        X_out = x_hat.to(torch.float32)
        if return_continuation_state:
            W_var_out = w_var.to(torch.float32)
            X_var_out = x_var.to(torch.float32)
            self._last_continuation_state = AlgorithmStateView(
                student_factors={
                    "W": W_out[0].detach().clone() if W_out.dim() == 4 and W_out.shape[0] == 1 else W_out.detach().clone(),
                    "X": X_out[0].detach().clone() if X_out.dim() == 4 and X_out.shape[0] == 1 else X_out.detach().clone(),
                },
                factor_variances={
                    "W_var": W_var_out[0].detach().clone() if W_var_out.dim() == 4 and W_var_out.shape[0] == 1 else W_var_out.detach().clone(),
                    "X_var": X_var_out[0].detach().clone() if X_var_out.dim() == 4 and X_var_out.shape[0] == 1 else X_var_out.detach().clone(),
                },
                teacher_factors={"W": W_teacher.detach(), "X": X_teacher.detach()},
                step_index=steps,
                alpha=float(alpha_values[0]) if len(alpha_values) == 1 else None,
                metadata={
                    "algorithm_key": "bigamp",
                    "continuation_state": "student_factors+factor_variances",
                    "state_transfer": (continuation_context or {}).get("state_transfer", "full_algorithm_state"),
                },
            )
        else:
            self._last_continuation_state = None
        return W_out, X_out

    def train_batch_result(
        self,
        *,
        algorithm_key: str,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: Optional[torch.Tensor],
        alpha_values: list[float],
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        step_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ):
        """Return dense BiGAMP output as a native matrix AlgorithmResult."""
        call_kwargs = self._filter_train_batch_kwargs({
            "W_teacher": W_teacher,
            "X_teacher": X_teacher,
            "Y_teacher": Y_teacher,
            "masks": masks,
            "alpha_values": alpha_values,
            "seed": seed,
            "progress_callback": progress_callback,
            "step_callback": step_callback,
            **kwargs,
        })
        W_students, X_students = self.train_batch_alphas(**call_kwargs)
        result = self.coerce_native_matrix_result(
            algorithm_key=algorithm_key,
            W_students=W_students,
            X_students=X_students,
            result_source="native_bigamp_algorithm_result",
        )
        if kwargs.get("return_continuation_state", False):
            result.continuation_state = getattr(self, "_last_continuation_state", None)
        return result

    @staticmethod
    def _coerce_continuation_factor(
        value: torch.Tensor,
        target_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = value.detach().to(device=device, dtype=dtype)
        if tuple(tensor.shape) == target_shape:
            return tensor.clone()
        if len(target_shape) == 4 and tensor.dim() == 3 and target_shape[0] == 1 and tuple(tensor.shape) == target_shape[1:]:
            return tensor.unsqueeze(0).clone()
        raise ValueError(
            f"Continuation factor shape {tuple(tensor.shape)} cannot initialize target {target_shape}"
        )

    def supports_batch_training(self) -> bool:
        """BiG-AMP supports efficient batch training."""
        return True
