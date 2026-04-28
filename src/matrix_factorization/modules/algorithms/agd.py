"""
AGD (Alternating Gradient Descent) algorithm for matrix factorization.

Migrated from Wang/agd/train_parallel.py
"""

import hashlib
from typing import Tuple, Optional, Callable
import torch

from ..registry import register_algorithm
from .base import AlgorithmBase
from ...core.config import Config
from ...core.contracts import AlgorithmStateView
from ...core.experiment.config import resolve_normalization_profile


def _stable_partition_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


@register_algorithm(
    key="agd",
    name="Alternating Gradient Descent",
    description="Gradient descent, stable but slower (~20k epochs)",
    default_params={
        'learning_rate': 1e-2,
        'weight_decay': 0.0,
        'use_early_stop': False,
        'target_loss_threshold': 1e-8,
        'relative_change_threshold': 1e-7,
        'early_stop_check_interval': 100,
        'early_stop_patience': 5,
    },
)
class AGDAlgorithm(AlgorithmBase):
    """
    Alternating Gradient Descent for masked matrix factorization.

    Updates W and X alternately using gradient descent on masked MSE loss.
    Slower than BiG-AMP (~20k epochs vs ~1k steps) but more stable.
    """

    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, device)
        algorithm_params = getattr(config, "algorithm_params", getattr(config, "algorithm", None))
        if algorithm_params is None:
            raise AttributeError("AGDAlgorithm requires config.algorithm_params")
        self.algorithm_params = algorithm_params
        self.lr = algorithm_params.learning_rate
        self.max_epochs = config.training.max_epochs
        self.S = config.training.samples_per_alpha
        self.seed_partition_policy = getattr(algorithm_params, 'seed_partition_policy', 'legacy')
        if self.seed_partition_policy not in {'legacy', 'partition_invariant'}:
            raise ValueError(
                "algorithm_params.seed_partition_policy must be 'legacy' or 'partition_invariant', "
                f"got {self.seed_partition_policy!r}"
            )
        self.requested_use_tf32 = getattr(algorithm_params, 'use_tf32', True)
        torch.backends.cuda.matmul.allow_tf32 = bool(self.requested_use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(self.requested_use_tf32)

        # Early stop settings
        self.use_early_stop = getattr(algorithm_params, 'use_early_stop', False)
        self.target_loss = getattr(algorithm_params, 'target_loss_threshold', 1e-8)
        self.relative_threshold = getattr(algorithm_params, 'relative_change_threshold', 1e-7)
        self.check_interval = getattr(algorithm_params, 'early_stop_check_interval', 100)
        self.patience = getattr(algorithm_params, 'early_stop_patience', 5)

        # BF16 settings
        self.precision_profile = getattr(algorithm_params, 'precision_profile', 'fast' if getattr(algorithm_params, 'use_bf16', True) else 'safe')
        if self.precision_profile == 'fast' and getattr(algorithm_params, 'use_bf16', True) is False:
            self.precision_profile = 'safe'
        self.precision_fallback_policy = getattr(algorithm_params, 'precision_fallback_policy', 'allow')
        if self.precision_fallback_policy == 'allow' and getattr(algorithm_params, 'dtype_fallback_policy', 'allow') == 'error':
            self.precision_fallback_policy = 'error'
        self.requested_use_bf16 = self.precision_profile in {'fast', 'aggressive'}
        self.dtype_fallback_policy = self.precision_fallback_policy
        if self.dtype_fallback_policy not in {'allow', 'error'}:
            raise ValueError(
                "algorithm_params.dtype_fallback_policy must be 'allow' or 'error', "
                f"got {self.dtype_fallback_policy!r}"
            )
        bf16_supported = bool(device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        self.use_bf16 = bool(self.requested_use_bf16) and bf16_supported
        if self.use_bf16:
            self.dtype_status = "bf16_requested_and_effective_cuda_autocast"
        elif self.requested_use_bf16:
            self.dtype_status = "fallback_to_float32_bf16_unavailable"
            if self.dtype_fallback_policy == 'error':
                raise RuntimeError(
                    "BF16 was requested but AGD is running on a non-CUDA device and "
                    "algorithm_params.dtype_fallback_policy='error'"
                )
        else:
            self.dtype_status = "bf16_disabled_by_config"
        self.compute_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        self.normalization_profile = getattr(algorithm_params, "normalization_profile", "paper_sparse_sampling")
        self._norm = resolve_normalization_profile(self.normalization_profile, config.matrix.M)
        self._contract_execution_metadata = {
            "path": "agd_matrix_autocast",
            "normalization_schema_version": self._norm.schema_version,
            "normalization_profile": self.normalization_profile,
            "normalization_convention": self._norm.convention_label,
            "precision_profile": self.precision_profile,
            "precision_fallback_policy": self.precision_fallback_policy,
            "requested_use_bf16": bool(self.requested_use_bf16),
            "effective_use_bf16": bool(self.use_bf16),
            "dtype_fallback_policy": self.dtype_fallback_policy,
            "dtype_status": self.dtype_status,
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
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
        sample_offset: int = 0,
        shape: Tuple[int, ...],
        seed: int,
        device: torch.device,
        scale: float,
        role: str,
    ) -> torch.Tensor:
        alpha_blocks = []
        for alpha in alpha_values:
            sample_blocks = []
            alpha_token = f"{float(alpha):.12g}"
            for local_sample_idx in range(sample_count):
                sample_idx = int(sample_offset) + local_sample_idx
                gen = torch.Generator(device=device).manual_seed(
                    _stable_partition_seed(seed, "agd", role, alpha_token, sample_idx)
                )
                sample_blocks.append(torch.randn(shape, generator=gen, device=device) * scale)
            alpha_blocks.append(torch.stack(sample_blocks, dim=0))
        return torch.stack(alpha_blocks, dim=0)

    def train_single_alpha(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train AGD for single alpha value."""
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.S
        device = self.device
        lr = self.lr

        norm = resolve_normalization_profile(self.normalization_profile, M)
        alpha_scale = norm.interaction_scale
        scale = norm.student_init_std

        # Ensure mask has batch dimension
        A = mask.unsqueeze(0) if mask.dim() == 2 else mask  # (1, N1, N2)
        Y_teacher_b = Y_teacher.unsqueeze(0)  # (1, N1, N2)

        # Initialize student
        if self._uses_partition_invariant_seed_policy():
            W = self._randn_partitioned_matrix(
                alpha_values=[alpha],
                sample_count=S,
                shape=(N1, M),
                seed=seed,
                device=device,
                scale=scale,
                role="W_student",
            )[0].to(torch.float32)
            X = self._randn_partitioned_matrix(
                alpha_values=[alpha],
                sample_count=S,
                shape=(M, N2),
                seed=seed,
                device=device,
                scale=scale,
                role="X_student",
            )[0].to(torch.float32)
        else:
            torch.manual_seed(seed)
            W = torch.randn((S, N1, M), device=device, dtype=torch.float32) * scale
            X = torch.randn((S, M, N2), device=device, dtype=torch.float32) * scale

        # Early stop tracking
        from collections import deque
        loss_history = deque(maxlen=self.patience) if self.use_early_stop else None

        # Training loop
        for step in range(self.max_epochs):
            # Use autocast for BF16 acceleration
            with torch.autocast(device_type=device.type, dtype=self.compute_dtype,
                                enabled=self.use_bf16):
                # W update
                Y_student = alpha_scale * torch.matmul(W, X)
                Mres = (Y_teacher_b - Y_student) * A
                grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X.transpose(1, 2))

            W = W - lr * grad_W.float()

            with torch.autocast(device_type=device.type, dtype=self.compute_dtype,
                                enabled=self.use_bf16):
                # X update with updated W
                Y_student2 = alpha_scale * torch.matmul(W, X)
                Mres2 = (Y_teacher_b - Y_student2) * A
                grad_X = -2.0 * alpha_scale * torch.matmul(W.transpose(1, 2), Mres2)

            X = X - lr * grad_X.float()

            # Report progress
            if progress_callback:
                progress_callback(step + 1, self.max_epochs)

            # Early stop check
            if self.use_early_stop and (step + 1) % self.check_interval == 0:
                with torch.no_grad():
                    Y_check = alpha_scale * torch.matmul(W, X)
                    R_check = (Y_teacher_b - Y_check) * A
                    current_loss = float(torch.sum(R_check ** 2, dim=(1, 2)).mean().item())

                    if current_loss < self.target_loss:
                        break

                    if loss_history is not None:
                        loss_history.append(current_loss)
                        if len(loss_history) >= self.patience:
                            losses = list(loss_history)
                            max_loss, min_loss = max(losses), min(losses)
                            if max_loss > 1e-12:
                                relative_change = (max_loss - min_loss) / max_loss
                                if relative_change < self.relative_threshold:
                                    break

        return W.float(), X.float()

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
        step_callback: Optional[Callable[[int, int], None]] = None,
        sample_callback: Optional[Callable] = None,
        runtime_step_callback: Optional[Callable] = None,
        initial_state: Optional[AlgorithmStateView] = None,
        return_continuation_state: bool = False,
        continuation_context: Optional[dict] = None,
        sample_context: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train AGD for multiple alphas in parallel."""
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        S = self.S
        sample_offset = int((sample_context or {}).get("sample_start", 0))
        device = self.device
        lr = self.lr
        num_alphas = len(alpha_values)
        
        # Use provided max_steps or fall back to config
        steps = max_steps if max_steps is not None else self.max_epochs

        norm = resolve_normalization_profile(self.normalization_profile, M)
        alpha_scale = norm.interaction_scale
        scale = norm.student_init_std

        # masks: (num_alphas, N1, N2) -> (num_alphas, 1, N1, N2)
        A_all = masks.unsqueeze(1)
        Y_teacher_b = Y_teacher.unsqueeze(0).unsqueeze(0)  # (1, 1, N1, N2)

        # Initialize student - (num_alphas, S, N1, M)
        if self._uses_partition_invariant_seed_policy():
            W = self._randn_partitioned_matrix(
                alpha_values=alpha_values,
                sample_count=S,
                sample_offset=sample_offset,
                shape=(N1, M),
                seed=seed,
                device=device,
                scale=scale,
                role="W_student",
            ).to(torch.float32)
            X = self._randn_partitioned_matrix(
                alpha_values=alpha_values,
                sample_count=S,
                sample_offset=sample_offset,
                shape=(M, N2),
                seed=seed,
                device=device,
                scale=scale,
                role="X_student",
            ).to(torch.float32)
        else:
            torch.manual_seed(seed)
            W = torch.randn((num_alphas, S, N1, M), device=device, dtype=torch.float32) * scale
            X = torch.randn((num_alphas, S, M, N2), device=device, dtype=torch.float32) * scale

        if initial_state is not None:
            factors = initial_state.student_factors or {}
            if "W" in factors and "X" in factors:
                W = self._coerce_continuation_factor(
                    factors["W"], (num_alphas, S, N1, M), device, torch.float32
                )
                X = self._coerce_continuation_factor(
                    factors["X"], (num_alphas, S, M, N2), device, torch.float32
                )

        for step in range(steps):
            with torch.autocast(device_type=device.type, dtype=self.compute_dtype,
                                enabled=self.use_bf16):
                # W update
                Y_student = alpha_scale * torch.matmul(W, X)
                Mres = (Y_teacher_b - Y_student) * A_all
                grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X.transpose(-2, -1))

            W = W - lr * grad_W.float()

            with torch.autocast(device_type=device.type, dtype=self.compute_dtype,
                                enabled=self.use_bf16):
                # X update
                Y_student2 = alpha_scale * torch.matmul(W, X)
                Mres2 = (Y_teacher_b - Y_student2) * A_all
                grad_X = -2.0 * alpha_scale * torch.matmul(W.transpose(-2, -1), Mres2)

            X = X - lr * grad_X.float()

            # Report progress (step_callback is the new interface, progress_callback for backward compat)
            callback = step_callback or progress_callback
            if callback:
                callback(step + 1, steps)
            if runtime_step_callback:
                runtime_step_callback(
                    AlgorithmStateView(
                        student_factors={
                            "W": W.detach(),
                            "X": X.detach(),
                        },
                        teacher_factors={
                            "W": W_teacher.detach(),
                            "X": X_teacher.detach(),
                        },
                        step_index=step + 1,
                        metadata={
                            "algorithm_key": "agd",
                            "alpha_values": list(alpha_values),
                            "sample_count": S,
                            "loss": float((Mres2.detach().float() ** 2).mean().item()),
                            "metadata_only": True,
                        },
                    )
                )

        W_out = W.float()
        X_out = X.float()
        if return_continuation_state:
            self._last_continuation_state = AlgorithmStateView(
                student_factors={
                    "W": W_out[0].detach().clone() if W_out.dim() == 4 and W_out.shape[0] == 1 else W_out.detach().clone(),
                    "X": X_out[0].detach().clone() if X_out.dim() == 4 and X_out.shape[0] == 1 else X_out.detach().clone(),
                },
                teacher_factors={"W": W_teacher.detach(), "X": X_teacher.detach()},
                step_index=steps,
                alpha=float(alpha_values[0]) if len(alpha_values) == 1 else None,
                metadata={
                    "algorithm_key": "agd",
                    "continuation_state": "student_factors",
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
        """Return AGD output as a native matrix AlgorithmResult.

        The training loop is still the existing train_batch_alphas() path; this
        method only makes the active result contract explicit.
        """
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
            result_source="native_agd_algorithm_result",
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
        """AGD supports batch training."""
        return True

    def estimate_memory_per_alpha(self, N1: int, N2: int, M: int, S: int) -> float:
        """
        Estimate GPU memory needed per alpha value.

        AGD needs:
        - W, X student parameters: 2 * (S * N1 * M + S * M * N2)
        - Y_student, Mres, grad intermediates: ~6 * S * N1 * N2
        - Mask: S * N1 * N2
        Total: ~9 S*N1*N2 tensors + parameters
        """
        student_params = 2 * (S * N1 * M + S * M * N2)
        intermediate = 9 * S * N1 * N2
        total_elements = student_params + intermediate
        return total_elements * 4 / (1024**3)  # 4 bytes per float32
