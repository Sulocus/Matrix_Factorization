"""
BiG-AMP for N-dimensional tensor CP decomposition with sample parallelization.

This module provides a parallel version of BiGAMPTensorSpreading that
processes all S samples simultaneously, maximizing GPU utilization.

Key improvements over serial version:
- All S samples processed in a single GPU kernel call
- GPU utilization: ~25% -> ~100%
- VRAM usage: ~4G -> ~16G (trade-off for speed)
"""

import math
import torch
import logging
from typing import Any, List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from matrix_factorization.core.distributions import F_DISTRIBUTION_ISING, normalize_f_distribution

logger = logging.getLogger(__name__)

# Set to True for verbose debug output during development
DEBUG_VERBOSE = False

from .tensor_data import TensorHypergraph, TensorSpreadingData
from .tensor_contract import (
    build_tensor_alpha_batch_metadata,
    build_tensor_execution_metadata,
    build_tensor_result_metadata,
    pack_tensor_parallel_metrics,
    resolve_tensor_dims,
)
from .tensor_step_batch import tensor_step_batch, forward_pass_tensor_batch
from .tensor_hypergraph import (
    generate_tensor_hypergraph,
    generate_tensor_observations_batch,
)
from .tensor_supergraph import (
    TensorSuperGraph, TensorSuperData,
    create_tensor_supergraph, create_tensor_superdata, stable_partition_seed,
)
from .tensor_step_super import tensor_step_super, forward_pass_tensor_super

from matrix_factorization.modules.registry import register_algorithm
from matrix_factorization.modules.algorithms.base import AlgorithmBase
from matrix_factorization.core.contracts import AlgorithmStateView
from matrix_factorization.core.experiment.config import resolve_normalization_profile
from matrix_factorization.modules.metrics.tensor_metrics import (
    compute_factor_gram_overlap,
    compute_tensor_projection_abs,
    compute_tensor_factor_projection_overlaps,
)


# Enable TF32 for improved performance on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@register_algorithm(
    key="bigamp_tensor_parallel",
    name="BiG-AMP Tensor Parallel",
    description="N-dimensional tensor CP decomposition with sample parallelization",
    default_params={'damping': 0.5, 'noise_var': 1e-6},
)
class BiGAMPTensorSpreadingParallel(AlgorithmBase):
    """
    BiG-AMP for N-dimensional tensor CP decomposition with sample parallelization.

    This algorithm processes all S samples in parallel using batched tensor
    operations, dramatically improving GPU utilization.

    Architecture:
    - factors: (S, N_d, M) - batched factor matrices
    - F: (S, C, M) - batched spreading coefficients
    - Y: (S, C) - batched observations
    - indices: (C,) - SHARED hypergraph structure across samples

    Performance optimizations:
    - TF32: Enabled globally for Tensor Core acceleration
    - BF16: Auto-enabled on supported hardware (Ampere+)
    - torch.compile: Kernel fusion with Triton (if available)
    """

    # Class-level cache for compiled step function
    _compiled_step = None
    _compiled_step_super = None  # Phase 3: for tensor_step_super

    @classmethod
    def clear_compile_cache(cls):
        """Clear compiled step function cache to release GPU memory."""
        cls._compiled_step = None
        cls._compiled_step_super = None
        try:
            import torch._dynamo
            torch._dynamo.reset()
        except Exception:
            pass

    def __init__(self, config=None, device: torch.device = None, **kwargs):
        """
        Initialize from runner config or direct parameters.

        Same interface as BiGAMPTensorSpreading for drop-in replacement.
        """
        # === Legacy kwargs API ===
        if 'tensor_order' in kwargs or (config is None and device is None and len(kwargs) > 0):
            self.config = None
            self.order = kwargs.get('tensor_order', 3)
            self.dims = kwargs.get('dims', tuple([50] * self.order))
            self.M = kwargs.get('M', 20)
            self.max_steps = kwargs.get('max_steps', 200)
            self.S = kwargs.get('S', 1)
            self.damping = kwargs.get('damping', 0.5)
            self.noise_var = kwargs.get('noise_var', 1e-6)
            self.f_distribution = normalize_f_distribution(kwargs.get('f_distribution', F_DISTRIBUTION_ISING))
            self.onsager_correction = kwargs.get('onsager_correction', False)
            self.device = kwargs.get('device', device) or torch.device('cpu')
            self.precision_profile = kwargs.get(
                'precision_profile',
                'fast' if kwargs.get('use_bf16', True) else 'safe',
            )
            if self.precision_profile == 'fast' and kwargs.get('use_bf16', True) is False:
                self.precision_profile = 'safe'
            self.precision_fallback_policy = kwargs.get(
                'precision_fallback_policy',
                'allow',
            )
            if self.precision_fallback_policy == 'allow' and kwargs.get('dtype_fallback_policy', 'allow') == 'error':
                self.precision_fallback_policy = 'error'
            self.requested_use_bf16 = self.precision_profile in {'fast', 'aggressive'}
            self.dtype_fallback_policy = self.precision_fallback_policy
            self.requested_use_tf32 = kwargs.get('use_tf32', True)
            self.seed_partition_policy = kwargs.get('seed_partition_policy', 'legacy')
            self.requested_use_compile = kwargs.get('use_compile', True)
            self.compile_fallback_policy = kwargs.get('compile_fallback_policy', 'allow')
            self.normalization_profile = kwargs.get('normalization_profile', 'paper_sparse_sampling')

        # === From runner ===
        elif hasattr(config, 'matrix'):
            self.config = config
            self.device = device
            algorithm_params = getattr(config, 'algorithm_params', None)

            self.order = getattr(config.spreading, 'tensor_order', 3) if hasattr(config, 'spreading') else 3

            self.dims = resolve_tensor_dims(config.matrix, self.order)
            self.M = config.matrix.M
            self.max_steps = config.training.max_steps
            self.S = config.training.samples_per_alpha

            self.damping = config.algorithm_params.damping
            self.noise_var = config.algorithm_params.noise_var

            if hasattr(config, 'spreading') and config.spreading:
                self.f_distribution = normalize_f_distribution(config.spreading.f_distribution)
                self.onsager_correction = config.spreading.onsager_correction
            else:
                self.f_distribution = F_DISTRIBUTION_ISING
                self.onsager_correction = config.spreading.onsager_correction

            # Warm Start / Init Mode
            self.init_mode = getattr(algorithm_params, 'init_mode', 'spectral')
            # FIX: Correct parameter name matching config.py (init_overlap)
            self.warm_start_rho = getattr(algorithm_params, 'init_overlap', 0.9)
            self.debug_verbose = getattr(algorithm_params, 'debug_verbose', False)
            self.precision_profile = getattr(
                algorithm_params,
                'precision_profile',
                'fast' if getattr(algorithm_params, 'use_bf16', True) else 'safe',
            )
            if self.precision_profile == 'fast' and getattr(algorithm_params, 'use_bf16', True) is False:
                self.precision_profile = 'safe'
            self.precision_fallback_policy = getattr(
                algorithm_params,
                'precision_fallback_policy',
                'allow',
            )
            if self.precision_fallback_policy == 'allow' and getattr(algorithm_params, 'dtype_fallback_policy', 'allow') == 'error':
                self.precision_fallback_policy = 'error'
            self.requested_use_bf16 = self.precision_profile in {'fast', 'aggressive'}
            self.dtype_fallback_policy = self.precision_fallback_policy
            self.requested_use_tf32 = getattr(algorithm_params, 'use_tf32', True)
            self.seed_partition_policy = getattr(algorithm_params, 'seed_partition_policy', 'legacy')
            self.requested_use_compile = getattr(algorithm_params, 'use_compile', True)
            self.compile_fallback_policy = getattr(algorithm_params, 'compile_fallback_policy', 'allow')
            self.normalization_profile = getattr(algorithm_params, "normalization_profile", "paper_sparse_sampling")

            if DEBUG_VERBOSE:
                print(f"DEBUG: Configured Init Mode: {self.init_mode}, Init Overlap (rho): {self.warm_start_rho}", flush=True)

        elif isinstance(config, int):
            self.config = None
            self.device = device
            self.order = config
            self.dims = tuple([50] * self.order)
            self.M = 20
            self.max_steps = 200
            self.S = 1
            self.damping = 0.5
            self.noise_var = 1e-6
            self.f_distribution = F_DISTRIBUTION_ISING
            self.onsager_correction = False
            self.debug_verbose = False
            self.precision_profile = 'fast'
            self.precision_fallback_policy = 'allow'
            self.requested_use_bf16 = True
            self.dtype_fallback_policy = self.precision_fallback_policy
            self.requested_use_tf32 = True
            self.seed_partition_policy = 'legacy'
            self.requested_use_compile = True
            self.compile_fallback_policy = 'allow'
            self.normalization_profile = 'paper_sparse_sampling'
        else:
            raise ValueError(f"config must be a config object or int, got {type(config)}")

        if len(self.dims) != self.order:
            raise ValueError(f"dims length {len(self.dims)} must match tensor_order {self.order}")
        if self.precision_profile not in {'safe', 'fast', 'aggressive'}:
            raise ValueError(
                "algorithm_params.precision_profile must be 'safe', 'fast', or 'aggressive', "
                f"got {self.precision_profile!r}"
            )
        if self.compile_fallback_policy not in {'allow', 'error'}:
            raise ValueError(
                "algorithm_params.compile_fallback_policy must be 'allow' or 'error', "
                f"got {self.compile_fallback_policy!r}"
            )
        if self.dtype_fallback_policy not in {'allow', 'error'}:
            raise ValueError(
                "algorithm_params.dtype_fallback_policy must be 'allow' or 'error', "
                f"got {self.dtype_fallback_policy!r}"
            )
        if self.seed_partition_policy not in {'legacy', 'partition_invariant'}:
            raise ValueError(
                "algorithm_params.seed_partition_policy must be 'legacy' or 'partition_invariant', "
                f"got {self.seed_partition_policy!r}"
            )
        self._norm = resolve_normalization_profile(self.normalization_profile, self.M)
        torch.backends.cuda.matmul.allow_tf32 = bool(self.requested_use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(self.requested_use_tf32)

        # === Phase 1.5: BF16 Mixed Precision ===
        # Auto-detect hardware support for BF16 (Ampere+ GPUs)
        self.use_bf16 = False
        self.storage_dtype = torch.float32
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if self.requested_use_bf16 and bf16_supported:
            self.use_bf16 = True
            self.storage_dtype = torch.bfloat16
            self.dtype_status = "bf16_requested_and_effective"
        elif self.requested_use_bf16:
            self.dtype_status = "fallback_to_float32_bf16_unavailable"
            self._handle_dtype_fallback()
        else:
            self.dtype_status = "bf16_disabled_by_config"

        # === Phase 1.5: torch.compile Support ===
        self.use_compile = bool(self.requested_use_compile)
        self.compile_attempts = []
        if self.use_compile and BiGAMPTensorSpreadingParallel._compiled_step is None:
            try:
                # Use 'default' mode for safety (no CUDA Graph issues)
                BiGAMPTensorSpreadingParallel._compiled_step = torch.compile(
                    tensor_step_batch,
                    mode='default',
                    fullgraph=False,
                )
                self._record_compile_attempt("tensor_step_batch", True, "")
            except Exception as exc:
                self.use_compile = False
                self._record_compile_attempt("tensor_step_batch", False, type(exc).__name__)
                self._handle_compile_failure("tensor_step_batch", exc)

        # Phase 3: Compile tensor_step_super for Alpha + Sample parallelization
        if self.use_compile and BiGAMPTensorSpreadingParallel._compiled_step_super is None:
            try:
                BiGAMPTensorSpreadingParallel._compiled_step_super = torch.compile(
                    tensor_step_super,
                    mode='default',
                    fullgraph=False,
                )
                self._record_compile_attempt("tensor_step_super", True, "")
            except Exception as exc:
                # Fall back to non-compiled version
                self._record_compile_attempt("tensor_step_super", False, type(exc).__name__)
                self._handle_compile_failure("tensor_step_super", exc)

        # Batch metrics storage
        self._batch_metrics = {}

    # =========================================================================
    # AlgorithmBase Interface Implementation
    # =========================================================================

    def _record_compile_attempt(self, target: str, success: bool, error: str) -> None:
        self.compile_attempts.append({
            "target": target,
            "success": bool(success),
            "error": error,
        })

    def _handle_compile_failure(self, target: str, exc: Exception) -> None:
        if self.compile_fallback_policy == "error":
            raise RuntimeError(
                f"torch.compile failed for {target} and "
                "algorithm_params.compile_fallback_policy='error'"
            ) from exc

    def _handle_dtype_fallback(self) -> None:
        if self.dtype_fallback_policy == "error":
            raise RuntimeError(
                "BF16 was requested but is unavailable and "
                "algorithm_params.dtype_fallback_policy='error'"
            )

    def _apply_observation_precision(self, superdata: TensorSuperData) -> TensorSuperData:
        """Apply storage dtype policy to large observation tensors.

        Ising F remains int8.  Gaussian F and Y are allowed to use BF16
        storage only under the aggressive profile; all reductions still cast to
        FP32 inside metric/update code where needed.
        """
        if getattr(self, "precision_profile", "fast") == "aggressive" and getattr(self, "use_bf16", False):
            if torch.is_floating_point(superdata.F_super):
                superdata.F_super = superdata.F_super.to(self.storage_dtype)
            if torch.is_floating_point(superdata.Y_super):
                superdata.Y_super = superdata.Y_super.to(self.storage_dtype)
        return superdata

    def _compile_status_for_super_path(self) -> str:
        if not bool(getattr(self, "requested_use_compile", True)):
            return "disabled_by_config"
        if (
            bool(getattr(self, "use_compile", False))
            and BiGAMPTensorSpreadingParallel._compiled_step_super is not None
        ):
            return "effective_for_tensor_step_super"
        return "fallback_to_eager_tensor_step_super"

    def train_single_alpha(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """AlgorithmBase interface: Train for single alpha (delegates to batch)."""
        teacher_factors = self._create_teacher_factors(W_teacher, X_teacher)
        result = self._train_parallel(teacher_factors, alpha, seed, self.device)

        W_students = torch.zeros(self.S, self.dims[0], self.M, device=self.device)
        X_students = torch.zeros(self.S, self.M, self.dims[1] if self.order >= 2 else self.dims[0], device=self.device)

        self._last_result = result
        return W_students, X_students

    def train_batch_alphas(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: torch.Tensor,
        alpha_values: List[float],
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        step_callback: Optional[Callable[[int, int, Optional[Dict]], None]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AlgorithmBase interface: Train for multiple alpha values.

        Phase 3.1: Smart alpha batching based on memory constraints.
        Uses greedy algorithm to group alphas into batches that fit in GPU memory.
        """
        A = len(alpha_values)
        self._run_batch_metrics_only(
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            Y_teacher=Y_teacher,
            masks=masks,
            alpha_values=alpha_values,
            seed=seed,
            progress_callback=progress_callback,
            step_callback=step_callback,
            **kwargs,
        )

        # Legacy tuple API: tensor metrics-only callers should use
        # train_batch_result() to avoid allocating these placeholder factors.
        W_all = torch.zeros(A, self.S, self.dims[0], self.M, device=self.device)
        X_all = torch.zeros(A, self.S, self.M, self.dims[1] if self.order >= 2 else self.dims[0], device=self.device)
        return W_all, X_all

    def _run_batch_metrics_only(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: torch.Tensor,
        alpha_values: List[float],
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        step_callback: Optional[Callable[[int, int, Optional[Dict]], None]] = None,
        **kwargs,
    ) -> None:
        """Populate tensor batch metrics without allocating placeholder W/X."""
        teacher_factors = self._create_teacher_factors(W_teacher, X_teacher)

        self._batch_metrics = {}
        self._last_continuation_state = None

        # Smart alpha batching
        alpha_batches = self._compute_alpha_batches(alpha_values)
        self._last_alpha_values_original = [float(alpha) for alpha in alpha_values]
        self._last_alpha_values_execution_order = [
            float(alpha)
            for batch in alpha_batches
            for alpha in batch
        ]

        for batch_idx, batch_alphas in enumerate(alpha_batches):
            # Wrapper for step_callback to inject batch info
            current_callback = step_callback
            if step_callback is not None:
                def internal_step_callback(step: int, total: int, metrics: Optional[Dict] = None):
                    metrics = metrics or {}
                    metrics['batch_info'] = {
                        'batch_idx': batch_idx + 1,  # 1-based for UI
                        'total_batches': len(alpha_batches),
                        'batch_alphas': batch_alphas,
                    }
                    step_callback(step, total, metrics)
                current_callback = internal_step_callback

            # Process this batch
            try:
                batch_seed = self._seed_for_internal_alpha_batch(seed, batch_idx, batch_alphas)
                result = self._train_full_parallel(
                    teacher_factors,
                    batch_alphas,
                    batch_seed,
                    self.device,
                    current_callback,
                    initial_state=kwargs.get("initial_state") if len(batch_alphas) == 1 else None,
                    return_continuation_state=bool(kwargs.get("return_continuation_state", False)) and len(batch_alphas) == 1,
                    continuation_context=kwargs.get("continuation_context"),
                )
                if result.get("_continuation_state") is not None:
                    self._last_continuation_state = result["_continuation_state"]

                # Store metrics for each alpha in this batch
                for local_idx, alpha in enumerate(batch_alphas):
                    self._batch_metrics[alpha] = pack_tensor_parallel_metrics(result, local_idx)
            finally:
                # Clean up between batches. This is metadata/cleanup hardening
                # only; it does not retry or alter future batch partitioning.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def train_batch_result(
        self,
        *,
        algorithm_key: str,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: Optional[torch.Tensor],
        alpha_values: List[float],
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        step_callback: Optional[Callable[[int, int, Optional[Dict]], None]] = None,
        **kwargs: Any,
    ):
        """Return tensor metrics/artifacts through AlgorithmResult.

        This keeps the existing tensor-parallel numerical implementation in
        train_batch_alphas, but prevents the runner from treating placeholder
        W/X tensors as real matrix factors.
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
        self._run_batch_metrics_only(**call_kwargs)
        result = self._metrics_only_algorithm_result(algorithm_key)
        if kwargs.get("return_continuation_state", False):
            result.continuation_state = getattr(self, "_last_continuation_state", None)
        return result

    def _metrics_only_algorithm_result(self, algorithm_key: str):
        from matrix_factorization.core.contracts import AlgorithmResult
        from matrix_factorization.modules.registry import get_algorithm_spec

        spec = get_algorithm_spec(algorithm_key)
        metrics_by_alpha = {
            float(alpha): dict(metrics)
            for alpha, metrics in getattr(self, "_batch_metrics", {}).items()
        }
        artifacts = {}
        if any("overlap_matrix" in metrics for metrics in metrics_by_alpha.values()):
            artifacts["overlap_matrix"] = "per_alpha_metric_payload"
        norm = getattr(self, "_norm", None)
        normalization_profile = getattr(self, "normalization_profile", "paper_sparse_sampling")
        internal_alpha_batch_plan = getattr(self, "_last_internal_alpha_batch_plan", None)
        if internal_alpha_batch_plan is None:
            alpha_keys = sorted(metrics_by_alpha)
            internal_alpha_batch_plan = build_tensor_alpha_batch_metadata(
                planner="unknown_legacy_tensor_parallel_path",
                device=getattr(self, "device", None),
                alpha_values_input=alpha_keys,
                alpha_batches=[alpha_keys],
                sort_policy="metrics_key_order",
                probe_enabled=False,
                seed_partition_sensitive=True,
            )
        return AlgorithmResult.from_metrics_only(
            metrics_by_alpha=metrics_by_alpha,
            artifacts=artifacts,
            metadata=build_tensor_result_metadata(
                algorithm_key=algorithm_key,
                result_contract=spec.result_contract,
                result_source="tensor_algorithm_train_batch_result",
                dims=self.dims,
                tensor_order=self.order,
                graph_kind="tensor_supergraph",
                alpha_values_original=getattr(self, "_last_alpha_values_original", None),
                alpha_values_execution_order=getattr(self, "_last_alpha_values_execution_order", None),
                batching_source="algorithm_internal_probe_alpha_batches",
                extra={
                    "tensor_execution": build_tensor_execution_metadata(
                        path="parallel_tensor_supergraph",
                        device=self.device,
                        requested_use_bf16=bool(getattr(self, "requested_use_bf16", True)),
                        effective_use_bf16=bool(getattr(self, "use_bf16", False)),
                        dtype_fallback_policy=getattr(self, "dtype_fallback_policy", "allow"),
                        dtype_status=getattr(self, "dtype_status", ""),
                        storage_dtype=getattr(self, "storage_dtype", None),
                        precision_profile=getattr(self, "precision_profile", "fast"),
                        precision_fallback_policy=getattr(self, "precision_fallback_policy", "allow"),
                        normalization_profile=normalization_profile,
                        normalization_schema_version=getattr(norm, "schema_version", 5),
                        normalization_convention=getattr(norm, "convention_label", "paper_sparse_sampling_var1_latent_unit_prior"),
                        requested_use_compile=bool(getattr(self, "requested_use_compile", True)),
                        compile_fallback_policy=getattr(self, "compile_fallback_policy", "allow"),
                        effective_use_compile=bool(
                            getattr(self, "use_compile", False)
                            and BiGAMPTensorSpreadingParallel._compiled_step_super is not None
                        ),
                        compiled_step_available=BiGAMPTensorSpreadingParallel._compiled_step is not None,
                        compiled_super_step_available=BiGAMPTensorSpreadingParallel._compiled_step_super is not None,
                        compile_status=self._compile_status_for_super_path(),
                        compile_attempts=getattr(self, "compile_attempts", []),
                        requested_use_tf32=bool(getattr(self, "requested_use_tf32", True)),
                        tf32_matmul_enabled=torch.backends.cuda.matmul.allow_tf32,
                        tf32_cudnn_enabled=torch.backends.cudnn.allow_tf32,
                        notes="Execution metadata only; it does not change tensor update formulas.",
                    ),
                    "internal_alpha_batch_plan": internal_alpha_batch_plan,
                    "seed_partition_policy": {
                        "requested_policy": getattr(self, "seed_partition_policy", "legacy"),
                        "partition_invariant": self._uses_partition_invariant_seed_policy(),
                        "metadata_only": True,
                    },
                },
            ),
        )

    def _uses_partition_invariant_seed_policy(self) -> bool:
        return getattr(self, "seed_partition_policy", "legacy") == "partition_invariant"

    def _seed_for_internal_alpha_batch(
        self,
        base_seed: int,
        batch_idx: int,
        batch_alphas: List[float],
    ) -> int:
        if self._uses_partition_invariant_seed_policy():
            return int(base_seed)
        return int(base_seed) + int(batch_idx)

    def _randn_partitioned_factor(
        self,
        *,
        alpha_values: List[float],
        N_d: int,
        M: int,
        dim_index: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        scale: float,
        role: str,
    ) -> torch.Tensor:
        """Generate (A, S*N_d, M) noise independent of alpha batch partition."""
        alpha_blocks = []
        for alpha in alpha_values:
            sample_blocks = []
            alpha_token = f"{float(alpha):.12g}"
            for sample_idx in range(self.S):
                gen = torch.Generator(device=device).manual_seed(
                    stable_partition_seed(seed, role, alpha_token, dim_index, sample_idx)
                )
                sample_blocks.append(
                    torch.randn(N_d, M, generator=gen, device=device, dtype=dtype) * scale
                )
            alpha_blocks.append(torch.cat(sample_blocks, dim=0))
        return torch.stack(alpha_blocks, dim=0)

    def _compute_alpha_batches(self, alpha_values: List[float]) -> List[List[float]]:
        """
        Compute alpha batches using probing-based memory estimation.

        Uses probe_tensor_super_memory to measure actual memory for A=1,
        then calculates maximum alphas per batch based on available GPU memory.

        This is fully dynamic and adapts to any dims, M, S configuration.
        """
        if not torch.cuda.is_available():
            self._last_internal_alpha_batch_plan = build_tensor_alpha_batch_metadata(
                planner="cpu_no_internal_alpha_batching",
                device=self.device,
                alpha_values_input=alpha_values,
            alpha_batches=[alpha_values],
            sort_policy="preserve_input_order",
            probe_enabled=False,
            seed_partition_sensitive=not self._uses_partition_invariant_seed_policy(),
            empty_cache_between_batches=False,
        )
            return [alpha_values]  # No batching needed on CPU

        # Sort alphas (process smaller alphas first for better cache behavior)
        sorted_alphas = sorted(alpha_values)
        alpha_max = max(sorted_alphas) if sorted_alphas else 1.0

        # Use cached probe result if available
        cache_key = (tuple(self.dims), self.M, self.S, alpha_max)
        probe_cache_hit = False
        if hasattr(self, '_probe_cache') and cache_key in self._probe_cache:
            base_mem_gb = self._probe_cache[cache_key]
            probe_cache_hit = True
        else:
            # Probe memory for A=1
            from .tensor_memory import probe_tensor_super_memory
            base_mem_gb = probe_tensor_super_memory(
                self.dims, self.M, self.S, alpha_max, self.device, use_bf16=self.use_bf16
            )
            # Cache the result
            if not hasattr(self, '_probe_cache'):
                self._probe_cache = {}
            self._probe_cache[cache_key] = base_mem_gb
            logger.info(f"Phase 3 memory probe: A=1, alpha_max={alpha_max:.2f} -> {base_mem_gb:.2f} GB")

        # Get available GPU memory
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_gb = total_gb * 0.8  # Increased to 80% (from 50%) to maximize utilization

        # Calculate maximum alphas per batch
        fallback_reason = ""
        if base_mem_gb <= 0 or math.isinf(base_mem_gb):
            # Fallback if probe fails: attempt a reasonable batch size (e.g., 5)
            # This prevents fallback to serial execution (size=1) which is extremely slow
            max_alphas = min(5, len(sorted_alphas))
            fallback_reason = f"memory probe returned {base_mem_gb}"
            logger.warning(
                f"Memory probe failed (returned {base_mem_gb}). "
                f"Batching fallback: Defaulting to max_alphas={max_alphas}. "
                "If OOM occurs, reduce S or M."
            )
        else:
            max_alphas = max(1, int(available_gb / base_mem_gb))
        max_alphas = max(1, max_alphas)

        # Create batches
        batches = []
        n = len(sorted_alphas)
        for i in range(0, n, max_alphas):
            batch = sorted_alphas[i:i+max_alphas]
            batches.append(batch)

        if len(batches) > 1:
            logger.info(
                f"Split {n} alphas into {len(batches)} batches "
                f"(max {max_alphas} per batch, {base_mem_gb:.2f} GB each)"
            )

        self._last_internal_alpha_batch_plan = build_tensor_alpha_batch_metadata(
            planner="tensor_parallel_probe_alpha_batching",
            device=self.device,
            alpha_values_input=alpha_values,
            alpha_batches=batches,
            sort_policy="ascending_alpha_before_batching",
            probe_enabled=True,
            probe_method="probe_tensor_super_memory(A=1)",
            alpha_max=alpha_max,
            probe_result_gb=base_mem_gb,
            probe_cache_hit=probe_cache_hit,
            total_memory_gb=total_gb,
            target_memory_gb=available_gb,
            target_memory_fraction=0.8,
            max_alphas_per_batch=max_alphas,
            fallback_reason=fallback_reason,
            seed_partition_sensitive=not self._uses_partition_invariant_seed_policy(),
            empty_cache_between_batches=True,
        )
        return batches

    def supports_batch_training(self) -> bool:
        """Tensor parallel supports batch training."""
        return True

    # =========================================================================
    # Core Parallel Training
    # =========================================================================

    def _train_full_parallel(
        self,
        teacher_factors: List[torch.Tensor],
        alpha_values: List[float],
        seed: int,
        device: torch.device,
        step_callback: Optional[Callable] = None,
        initial_state: Optional[AlgorithmStateView] = None,
        return_continuation_state: bool = False,
        continuation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, any]:
        """
        Train all alphas AND all samples in parallel using TensorSuperGraph.

        Phase 3 optimization: Single GPU call processes A × S problems.

        Args:
            teacher_factors: n tensors of (N_d, M)
            alpha_values: List of alpha values
            seed: Random seed
            device: Target device
            step_callback: Optional progress callback

        Returns:
            Dict with Q_Y (full tensor), Q_Y_observed (observation-only) arrays
        """
        n = self.order
        S = self.S
        A = len(alpha_values)
        M = teacher_factors[0].shape[1]

        teacher_factors = [t.to(device) for t in teacher_factors]

        # Create TensorSuperGraph
        supergraph = create_tensor_supergraph(
            self.dims,
            alpha_values,
            M,
            S,
            seed,
            device,
            partition_invariant=self._uses_partition_invariant_seed_policy(),
        )

        # Create TensorSuperData
        superdata = create_tensor_superdata(
            supergraph,
            teacher_factors,
            self.f_distribution,
            seed + 1000,
            partition_invariant=self._uses_partition_invariant_seed_policy(),
        )
        superdata = self._apply_observation_precision(superdata)

        # Get flat tensors and PRECOMPUTED offset_indices
        F_flat, Y_flat = superdata.get_flat_tensors()
        offset_indices = supergraph.get_offset_indices()  # PRECOMPUTED!
        N_dims = list(self.dims)

        # Initialize student factors
        init_scale = self._norm.student_init_std

        if self.init_mode in ('warm_start', 'teacher'):
            # Warm Start: Initialize near Teacher
            # Student = Teacher * rho + Noise * sqrt(1 - rho^2)
            # Both teacher and noise follow the selected normalization profile.
            rho = self.warm_start_rho
            if DEBUG_VERBOSE:
                print("=" * 60, flush=True)
                print(f"INIT DEBUG: Using Warm Start, rho = {rho} (type: {type(rho).__name__})", flush=True)
                print("=" * 60, flush=True)

            factors = []
            for d in range(self.order):
                # Expand Teacher to (A, S*N_d, M)
                # Teacher is (N_d, M). Student requires (A, S*N_d, M).
                # 1. Expand to (A, S, N_d, M)
                t_expanded_temp = teacher_factors[d].unsqueeze(0).unsqueeze(0).expand(A, S, -1, -1)
                # 2. Flatten S and N_d -> (A, S*N_d, M)
                t_expanded = t_expanded_temp.reshape(A, S * self.dims[d], M)

                if self._uses_partition_invariant_seed_policy():
                    noise = self._randn_partitioned_factor(
                        alpha_values=alpha_values,
                        N_d=self.dims[d],
                        M=M,
                        dim_index=d,
                        seed=seed,
                        device=device,
                        dtype=t_expanded.dtype,
                        scale=self._norm.student_init_std,
                        role="warm_start_noise",
                    )
                else:
                    noise = torch.randn_like(t_expanded) * self._norm.student_init_std
                # Mix
                f_init = t_expanded * rho + noise * math.sqrt(1 - rho**2)
                factors.append(f_init.to(device))

                if DEBUG_VERBOSE:
                    print(f"DEBUG: Initialized Factor {d} with Warm Start. Mean={f_init.mean():.4f}, Std={f_init.std():.4f}", flush=True)

        elif self.init_mode == 'spectral':
             # Spectral Initialization: Use power method
             # First initialize factors randomly as starting point for power method
             init_scale = self._norm.student_init_std
             if self._uses_partition_invariant_seed_policy():
                 factors = [
                     self._randn_partitioned_factor(
                         alpha_values=alpha_values,
                         N_d=N_d,
                         M=M,
                         dim_index=d,
                         seed=seed,
                         device=device,
                         dtype=self.storage_dtype,
                         scale=init_scale,
                         role="spectral_init",
                     )
                     for d, N_d in enumerate(self.dims)
                 ]
             else:
                 factors = [
                     torch.randn(A, S * N_d, M, device=device, dtype=self.storage_dtype) * init_scale
                     for N_d in self.dims
                 ]
             # Then refine using spectral method
             factors = self._spectral_initialization(
                supergraph, Y_flat, F_flat, factors, offset_indices, iterations=30
             )

             # RESCALE metric: target follows the selected normalization profile.
             target_std = self._norm.student_init_std
             if DEBUG_VERBOSE:
                 print(f"DEBUG: Rescaling factors to canonical latent std={target_std}", flush=True)
             for d in range(len(factors)):
                if factors[d].std() > 0:
                    factors[d] = factors[d] * (target_std / factors[d].std())
                if DEBUG_VERBOSE:
                    print(f"DEBUG: Factor {d} init stats: Mean={factors[d].mean():.6e}, Std={factors[d].std():.6e}", flush=True)

        else:
             # Random Init (Cold Start)
             if DEBUG_VERBOSE:
                 print("=" * 60, flush=True)
                 print(f"INIT DEBUG: Using RANDOM INIT (Cold Start)", flush=True)
                 print("=" * 60, flush=True)

             init_scale = self._norm.student_init_std
             if self._uses_partition_invariant_seed_policy():
                 factors = [
                     self._randn_partitioned_factor(
                         alpha_values=alpha_values,
                         N_d=N_d,
                         M=M,
                         dim_index=d,
                         seed=seed,
                         device=device,
                         dtype=self.storage_dtype,
                         scale=init_scale,
                         role="random_init",
                     )
                     for d, N_d in enumerate(self.dims)
                 ]
             else:
                 factors = [
                     torch.randn(A, S * N_d, M, device=device, dtype=self.storage_dtype) * init_scale
                     for N_d in self.dims
                 ]
             if DEBUG_VERBOSE:
                 for d in range(self.order):
                     print(f"DEBUG: Factor {d} init stats: Mean={factors[d].mean():.6e}, Std={factors[d].std():.6e}", flush=True)

        factor_vars = [
            torch.ones(A, S * N_d, M, device=device, dtype=self.storage_dtype) * self._norm.prior_variance
            for N_d in self.dims
        ]

        if initial_state is not None:
            factors, factor_vars, prev_s, prev_svar = self._coerce_tensor_continuation_state(
                initial_state=initial_state,
                A=A,
                S=S,
                M=M,
                factor_vars=factor_vars,
                alpha_mask_exp=superdata.alpha_mask_exp,
            )
        else:
            prev_s = None
            prev_svar = None

        is_ising = (self.f_distribution == 'ising')

        # Use compiled step function if available (Phase 3 optimization)
        step_fn = (BiGAMPTensorSpreadingParallel._compiled_step_super
                   if self.use_compile and BiGAMPTensorSpreadingParallel._compiled_step_super is not None
                   else tensor_step_super)
        # Optional diagnostic: check whether initialization is random or teacher-like.
        if self.debug_verbose:
            with torch.no_grad():
                Z_hat_init = forward_pass_tensor_super(factors, F_flat, offset_indices, S, N_dims)
                mse_init = ((Y_flat - Z_hat_init)**2 * superdata.alpha_mask_exp.float()).sum() / superdata.alpha_mask_exp.sum()
                print("=" * 70, flush=True)
                print(f"STEP 0 DIAGNOSTIC: Initial MSE = {mse_init.item():.6f}", flush=True)
                print(f"  If MSE >> 0 (e.g. 1.5+): Initialization is RANDOM (correct)", flush=True)
                print(f"  If MSE ≈ 0: Initialization is TEACHER (BUG - data leakage!)", flush=True)
                print("=" * 70, flush=True)

        # BiG-AMP iterations
        for step in range(self.max_steps):
            # Noise annealing
            anneal_steps = int(0.2 * self.max_steps)
            current_noise_var = self.noise_var

            if step < anneal_steps:
                start_log = math.log(1.0)
                end_log = math.log(max(self.noise_var, 1e-4))
                progress = step / anneal_steps
                current_noise_var = math.exp(start_log + (end_log - start_log) * progress)

            # Debug: Save old factors to measure change
            if self.debug_verbose and step % 100 == 0:
                old_factors_debug = [f.clone() for f in factors]

            # Alpha + Sample parallel BiG-AMP step (using precomputed offset_indices)
            factors, factor_vars, next_prev_s, next_prev_svar = step_fn(
                factors, factor_vars, Y_flat, F_flat, offset_indices,
                S, N_dims, M,  # Added M parameter
                superdata.alpha_mask_exp,
                damping=self.damping,
                noise_var=current_noise_var,
                is_ising=is_ising,
                prev_s=prev_s if self.onsager_correction else None,
                prev_svar=prev_svar if self.onsager_correction else None,
                onsager_correction=self.onsager_correction,
                prior_precision_base=self._norm.prior_precision_base,
                prior_variance=self._norm.prior_variance,
            )
            if self.onsager_correction:
                prev_s = next_prev_s
                prev_svar = next_prev_svar
            else:
                prev_s = None
                prev_svar = None

            # Debug: Print progress
            if self.debug_verbose and step % 100 == 0:
                with torch.no_grad():
                    # Calculate factor change
                    change = (factors[0] - old_factors_debug[0]).abs().mean().item()

                    # Calculate current MSE (expensive, but necessary for debug)
                    Z_hat = forward_pass_tensor_super(factors, F_flat, offset_indices, S, N_dims)
                    mse = ((Y_flat - Z_hat)**2 * superdata.alpha_mask_exp.float()).sum() / superdata.alpha_mask_exp.sum()

                    if DEBUG_VERBOSE:
                        print(f"DEBUG Step {step}: Change={change:.6e}, MSE={mse:.6f}, Noise={current_noise_var:.6f}, FactorMean={factors[0].mean():.4f}, FactorStd={factors[0].std():.6f}", flush=True)

            # Progress callback - call every step (Rich auto-throttles to 10fps)
            if step_callback:
                step_callback(step + 1, self.max_steps)

        # Compute final metrics for all alphas and samples
        Z_hat = forward_pass_tensor_super(factors, F_flat, offset_indices, S, N_dims)

        # Reshape for per-alpha, per-sample metrics
        SC = supergraph.SC
        C_max = supergraph.C_max

        # Y_flat: (S*C_max,) -> (S, C_max)
        Y_reshaped = Y_flat.reshape(S, C_max)
        # Z_hat: (A, S*C_max) -> (A, S, C_max)
        Z_hat_reshaped = Z_hat.reshape(A, S, C_max)

        # Compute formal observed Q_Y per alpha/sample as FIT = 1 - NMSE
        # on the training hyperedge measurements.
        alpha_mask_reshaped = supergraph.alpha_mask.unsqueeze(1).expand(A, S, C_max)  # (A, S, C_max)
        mask_float = alpha_mask_reshaped.float()
        obs_dot = (Y_reshaped.unsqueeze(0) * Z_hat_reshaped * mask_float).sum(dim=2)
        obs_norm = ((Y_reshaped.unsqueeze(0) ** 2) * mask_float).sum(dim=2)
        obs_sse = (((Z_hat_reshaped - Y_reshaped.unsqueeze(0)) ** 2) * mask_float).sum(dim=2)
        Q_Y_per_alpha_sample = torch.where(
            obs_norm > 1e-12,
            1.0 - obs_sse / (obs_norm + 1e-12),
            torch.zeros_like(obs_norm),
        )
        NMSE_Y_per_alpha_sample = torch.where(
            obs_norm > 1e-12,
            obs_sse / (obs_norm + 1e-12),
            torch.ones_like(obs_norm),
        )
        Q_Y_observed_proj_abs_sample = torch.where(
            obs_norm > 1e-12,
            obs_dot.abs() / (obs_norm + 1e-12),
            torch.zeros_like(obs_norm),
        )

        # Deterministic heldout F-aware measurements.  These measurements are
        # generated after training and are not used by the AMP updates.
        heldout_supergraph = create_tensor_supergraph(
            self.dims,
            alpha_values,
            M,
            S,
            seed + 910_003,
            device,
            partition_invariant=self._uses_partition_invariant_seed_policy(),
        )
        heldout_superdata = create_tensor_superdata(
            heldout_supergraph,
            teacher_factors,
            self.f_distribution,
            seed + 911_021,
            partition_invariant=self._uses_partition_invariant_seed_policy(),
        )
        heldout_superdata = self._apply_observation_precision(heldout_superdata)
        F_holdout_flat, Y_holdout_flat = heldout_superdata.get_flat_tensors()
        Z_holdout = forward_pass_tensor_super(
            factors,
            F_holdout_flat,
            heldout_supergraph.get_offset_indices(),
            S,
            N_dims,
        )
        C_holdout_max = heldout_supergraph.C_max
        Y_holdout_reshaped = Y_holdout_flat.reshape(S, C_holdout_max)
        Z_holdout_reshaped = Z_holdout.reshape(A, S, C_holdout_max)
        holdout_mask = heldout_supergraph.alpha_mask.unsqueeze(1).expand(A, S, C_holdout_max).float()
        holdout_dot = (Y_holdout_reshaped.unsqueeze(0) * Z_holdout_reshaped * holdout_mask).sum(dim=2)
        holdout_norm = ((Y_holdout_reshaped.unsqueeze(0) ** 2) * holdout_mask).sum(dim=2)
        holdout_sse = (((Z_holdout_reshaped - Y_holdout_reshaped.unsqueeze(0)) ** 2) * holdout_mask).sum(dim=2)
        Q_Y_unobserved_sample = torch.where(
            holdout_norm > 1e-12,
            1.0 - holdout_sse / (holdout_norm + 1e-12),
            torch.zeros_like(holdout_norm),
        )
        NMSE_Y_unobserved_sample = torch.where(
            holdout_norm > 1e-12,
            holdout_sse / (holdout_norm + 1e-12),
            torch.ones_like(holdout_norm),
        )
        Q_Y_unobserved_proj_abs_sample = torch.where(
            holdout_norm > 1e-12,
            holdout_dot.abs() / (holdout_norm + 1e-12),
            torch.zeros_like(holdout_norm),
        )
        full_dot = obs_dot + holdout_dot
        full_norm = obs_norm + holdout_norm
        full_sse = obs_sse + holdout_sse
        Q_Y_full_sample = torch.where(
            full_norm > 1e-12,
            1.0 - full_sse / (full_norm + 1e-12),
            torch.zeros_like(full_norm),
        )
        NMSE_Y_full_sample = torch.where(
            full_norm > 1e-12,
            full_sse / (full_norm + 1e-12),
            torch.ones_like(full_norm),
        )
        Q_Y_full_proj_abs_sample = torch.where(
            full_norm > 1e-12,
            full_dot.abs() / (full_norm + 1e-12),
            torch.zeros_like(full_norm),
        )

        # Aggregate: mean and std over samples
        Q_Y_observed_per_alpha = Q_Y_per_alpha_sample.mean(dim=1).cpu().tolist()  # (A,)
        Q_Y_observed_std_per_alpha = Q_Y_per_alpha_sample.std(dim=1).cpu().tolist() if S > 1 else [0.0] * A
        NMSE_Y_observed_per_alpha = NMSE_Y_per_alpha_sample.mean(dim=1).cpu().tolist()
        NMSE_Y_observed_std_per_alpha = NMSE_Y_per_alpha_sample.std(dim=1).cpu().tolist() if S > 1 else [0.0] * A
        Q_Y_observed_proj_abs_per_alpha = Q_Y_observed_proj_abs_sample.mean(dim=1).cpu().tolist()
        Q_Y_observed_proj_abs_std_per_alpha = Q_Y_observed_proj_abs_sample.std(dim=1).cpu().tolist() if S > 1 else [0.0] * A
        Q_Y_unobserved_per_alpha = Q_Y_unobserved_sample.mean(dim=1).cpu().tolist()
        Q_Y_unobserved_std_per_alpha = Q_Y_unobserved_sample.std(dim=1).cpu().tolist() if S > 1 else [0.0] * A
        NMSE_Y_unobserved_per_alpha = NMSE_Y_unobserved_sample.mean(dim=1).cpu().tolist()
        NMSE_Y_unobserved_std_per_alpha = NMSE_Y_unobserved_sample.std(dim=1).cpu().tolist() if S > 1 else [0.0] * A
        Q_Y_unobserved_proj_abs_per_alpha = Q_Y_unobserved_proj_abs_sample.mean(dim=1).cpu().tolist()
        Q_Y_unobserved_proj_abs_std_per_alpha = Q_Y_unobserved_proj_abs_sample.std(dim=1).cpu().tolist() if S > 1 else [0.0] * A
        Q_Y_full_per_alpha = Q_Y_full_sample.mean(dim=1).cpu().tolist()
        Q_Y_full_std_per_alpha = Q_Y_full_sample.std(dim=1).cpu().tolist() if S > 1 else [0.0] * A
        NMSE_Y_full_per_alpha = NMSE_Y_full_sample.mean(dim=1).cpu().tolist()
        NMSE_Y_full_std_per_alpha = NMSE_Y_full_sample.std(dim=1).cpu().tolist() if S > 1 else [0.0] * A
        Q_Y_full_proj_abs_per_alpha = Q_Y_full_proj_abs_sample.mean(dim=1).cpu().tolist()
        Q_Y_full_proj_abs_std_per_alpha = Q_Y_full_proj_abs_sample.std(dim=1).cpu().tolist() if S > 1 else [0.0] * A

        # Reshape factors from (A, S*N_d, M) to (A, S, N_d, M)
        # We need access to individual samples for correct metrics.
        factors_all_samples = [] # List of Tensors (A, S, N_d, M)
        for d in range(self.order):
            N_d = self.dims[d]
            f_d = factors[d].view(A, S, N_d, -1)
            factors_all_samples.append(f_d)

        import numpy as np

        metric_choice = str(getattr(self, 'heatmap_metric', 'Q_Y') or 'Q_Y').upper()
        if metric_choice == "Q_W":
            heatmap_metric_fn = compute_factor_gram_overlap
            heatmap_metric_code = "Q_W"
        else:
            heatmap_metric_fn = compute_tensor_projection_abs
            heatmap_metric_code = "Q_Y"

        Q_N_modes = [[] for _ in range(self.order)]
        Q_N = []
        Q_N_std = []
        overlap_matrices = []  # One (Teacher + S replicas) matrix per alpha.

        for a_idx in range(A):
            # Calculate metrics per sample, then average. This avoids cancelling
            # incompatible replica states before measuring overlap.
            qn_samples = []
            sample_factors = []

            for s in range(S):
                student_factors_s = [f[a_idx, s, :, :] for f in factors_all_samples]
                sample_factors.append(student_factors_s)
                qn_modes_s = compute_tensor_factor_projection_overlaps(teacher_factors, student_factors_s)
                qn_samples.append(sum(qn_modes_s) / len(qn_modes_s))
                for d, value in enumerate(qn_modes_s):
                    Q_N_modes[d].append(value)

            qn_tensor = torch.tensor(qn_samples, dtype=torch.float32)
            Q_N.append(qn_tensor.mean().item())
            Q_N_std.append(qn_tensor.std().item() if S > 1 else 0.0)

            q_matrix = np.eye(S + 1, dtype=float)
            for s in range(S):
                val = heatmap_metric_fn(teacher_factors, sample_factors[s])
                q_matrix[0, s + 1] = val
                q_matrix[s + 1, 0] = val
            for i in range(S):
                for j in range(i + 1, S):
                    val = heatmap_metric_fn(sample_factors[i], sample_factors[j])
                    q_matrix[i + 1, j + 1] = val
                    q_matrix[j + 1, i + 1] = val
            overlap_matrices.append(q_matrix)

        result = {
            'Q_Y': Q_Y_full_per_alpha,
            'Q_Y_std': Q_Y_full_std_per_alpha,
            'NMSE_Y': NMSE_Y_full_per_alpha,
            'NMSE_Y_std': NMSE_Y_full_std_per_alpha,
            'Q_Y_PROJ_ABS': Q_Y_full_proj_abs_per_alpha,
            'Q_Y_PROJ_ABS_std': Q_Y_full_proj_abs_std_per_alpha,
            'Q_Y_observed': Q_Y_observed_per_alpha,
            'Q_Y_observed_std': Q_Y_observed_std_per_alpha,
            'NMSE_Y_observed': NMSE_Y_observed_per_alpha,
            'NMSE_Y_observed_std': NMSE_Y_observed_std_per_alpha,
            'Q_Y_observed_PROJ_ABS': Q_Y_observed_proj_abs_per_alpha,
            'Q_Y_observed_PROJ_ABS_std': Q_Y_observed_proj_abs_std_per_alpha,
            'Q_Y_unobserved': Q_Y_unobserved_per_alpha,
            'Q_Y_unobserved_std': Q_Y_unobserved_std_per_alpha,
            'NMSE_Y_unobserved': NMSE_Y_unobserved_per_alpha,
            'NMSE_Y_unobserved_std': NMSE_Y_unobserved_std_per_alpha,
            'Q_Y_unobserved_PROJ_ABS': Q_Y_unobserved_proj_abs_per_alpha,
            'Q_Y_unobserved_PROJ_ABS_std': Q_Y_unobserved_proj_abs_std_per_alpha,
            'Q_N': Q_N,
            'Q_N_std': Q_N_std,
            **{
                f'Q_N_mode{d}': [
                    float(torch.tensor(Q_N_modes[d][a_idx * S:(a_idx + 1) * S]).mean().item())
                    for a_idx in range(A)
                ]
                for d in range(self.order)
            },
            **{
                f'Q_N_mode{d}_std': [
                    float(torch.tensor(Q_N_modes[d][a_idx * S:(a_idx + 1) * S]).std().item()) if S > 1 else 0.0
                    for a_idx in range(A)
                ]
                for d in range(self.order)
            },
            'overlap_matrices': overlap_matrices,  # One (S+1)x(S+1) heatmap per alpha
            'overlap_matrix_metric': heatmap_metric_code,
            'A': A,
            'S': S,
            'C_max': C_max,
        }
        if return_continuation_state:
            result["_continuation_state"] = self._build_tensor_continuation_state(
                factors=factors,
                factor_vars=factor_vars,
                prev_s=prev_s,
                prev_svar=prev_svar,
                S=S,
                A=A,
                M=M,
                steps=self.max_steps,
                alpha_values=alpha_values,
                teacher_factors=teacher_factors,
            )
        return result

    def _coerce_tensor_continuation_state(
        self,
        *,
        initial_state: AlgorithmStateView,
        A: int,
        S: int,
        M: int,
        factor_vars: List[torch.Tensor],
        alpha_mask_exp: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        raw_factors = (initial_state.tensor_factors or {}).get("factors")
        if raw_factors is None:
            raw_factors = (initial_state.student_factors or {}).get("factors")
        if raw_factors is None:
            raise ValueError("tensor continuation requires tensor_factors['factors']")
        factors = [
            self._coerce_tensor_factor(raw_factors[d], (A, S, self.dims[d], M))
            .reshape(A, S * self.dims[d], M)
            for d in range(self.order)
        ]

        raw_vars = (initial_state.factor_variances or {}).get("factor_vars")
        if raw_vars is not None:
            factor_vars = [
                self._coerce_tensor_factor(raw_vars[d], (A, S, self.dims[d], M))
                .reshape(A, S * self.dims[d], M)
                for d in range(self.order)
            ]

        prev_s = None
        prev_svar = None
        residual = initial_state.onsager_residual or {}
        if self.onsager_correction and isinstance(residual, dict):
            if residual.get("prev_s") is not None:
                prev_s = self._coerce_tensor_residual(residual["prev_s"], alpha_mask_exp.shape)
                prev_s = prev_s * alpha_mask_exp.to(prev_s.dtype)
            if residual.get("prev_svar") is not None:
                prev_svar = self._coerce_tensor_residual(residual["prev_svar"], alpha_mask_exp.shape)
                prev_svar = prev_svar * alpha_mask_exp.to(prev_svar.dtype)
        return factors, factor_vars, prev_s, prev_svar

    def _build_tensor_continuation_state(
        self,
        *,
        factors: List[torch.Tensor],
        factor_vars: List[torch.Tensor],
        prev_s: Optional[torch.Tensor],
        prev_svar: Optional[torch.Tensor],
        S: int,
        A: int,
        M: int,
        steps: int,
        alpha_values: List[float],
        teacher_factors: List[torch.Tensor],
    ) -> AlgorithmStateView:
        factors_shaped = [
            factors[d].reshape(A, S, self.dims[d], M)
            for d in range(self.order)
        ]
        vars_shaped = [
            factor_vars[d].reshape(A, S, self.dims[d], M)
            for d in range(self.order)
        ]
        state_factors = [
            item[0].detach().clone() if A == 1 else item.detach().clone()
            for item in factors_shaped
        ]
        state_vars = [
            item[0].detach().clone() if A == 1 else item.detach().clone()
            for item in vars_shaped
        ]
        residual = None
        if self.onsager_correction and prev_s is not None:
            residual = {
                "prev_s": prev_s.detach().clone(),
                "prev_svar": prev_svar.detach().clone() if prev_svar is not None else None,
            }
        return AlgorithmStateView(
            tensor_factors={"factors": state_factors},
            factor_variances={"factor_vars": state_vars},
            onsager_residual=residual,
            teacher_factors={f"mode{idx}": value.detach() for idx, value in enumerate(teacher_factors)},
            step_index=steps,
            alpha=float(alpha_values[0]) if len(alpha_values) == 1 else None,
            metadata={
                "algorithm_key": "bigamp_tensor_parallel",
                "continuation_state": "tensor_factors+factor_variances"
                + ("+onsager_residual" if residual is not None else ""),
            },
        )

    def _coerce_tensor_factor(self, value: torch.Tensor, target_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        tensor = value.detach().to(device=self.device, dtype=self.storage_dtype)
        if tuple(tensor.shape) == target_shape:
            return tensor.clone()
        if tensor.dim() == 3 and target_shape[0] == 1 and tuple(tensor.shape) == target_shape[1:]:
            return tensor.unsqueeze(0).clone()
        raise ValueError(
            f"Continuation tensor factor shape {tuple(tensor.shape)} cannot initialize target {target_shape}"
        )

    def _coerce_tensor_residual(self, value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        tensor = value.detach().to(device=self.device, dtype=self.storage_dtype)
        if tuple(tensor.shape) == tuple(target_shape):
            return tensor.clone()
        if tensor.dim() == 1 and len(target_shape) == 2 and int(target_shape[0]) == 1 and int(tensor.shape[0]) == int(target_shape[1]):
            return tensor.unsqueeze(0).clone()
        raise ValueError(
            f"Continuation tensor residual shape {tuple(tensor.shape)} cannot initialize target {tuple(target_shape)}"
        )



    def _train_parallel(
        self,
        teacher_factors: List[torch.Tensor],
        alpha: float,
        seed: int,
        device: torch.device,
        step_callback: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Train all S samples in parallel for one alpha value.

        Key difference from serial: factors have shape (S, N_d, M) instead of (N_d, M).
        """
        n = self.order
        S = self.S
        M = teacher_factors[0].shape[1]

        teacher_factors = [t.to(device) for t in teacher_factors]

        # Generate SHARED hypergraph structure (indices are same for all samples)
        hg = generate_tensor_hypergraph(self.dims, alpha, M, seed, device)
        C = hg.C

        # Generate BATCHED F and Y
        F, Y = generate_tensor_observations_batch(
            teacher_factors, hg, S, seed + 1000, device, self.f_distribution
        )  # F: (S, C, M), Y: (S, C)
        if getattr(self, "precision_profile", "fast") == "aggressive" and getattr(self, "use_bf16", False):
            if torch.is_floating_point(F):
                F = F.to(self.storage_dtype)
            if torch.is_floating_point(Y):
                Y = Y.to(self.storage_dtype)

        # Initialize batched students (S, N_d, M) in the selected profile.
        init_scale = self._norm.student_init_std

        factors = [
            torch.randn(S, N_d, M, device=device, dtype=self.storage_dtype) * init_scale
            for N_d in self.dims
        ]
        factor_vars = [
            torch.ones(S, N_d, M, device=device, dtype=self.storage_dtype) * self._norm.prior_variance
            for N_d in self.dims
        ]

        prev_s = None
        prev_svar = None
        is_ising = (self.f_distribution == 'ising')

        # Select step function: compiled if available, else original
        step_fn = (BiGAMPTensorSpreadingParallel._compiled_step
                   if self.use_compile and BiGAMPTensorSpreadingParallel._compiled_step is not None
                   else tensor_step_batch)

        # BiG-AMP iterations (batched)
        for step in range(self.max_steps):
            # Mark CUDA Graph step for torch.compile compatibility
            if self.use_compile and BiGAMPTensorSpreadingParallel._compiled_step is not None:
                torch.compiler.cudagraph_mark_step_begin()

            # Noise annealing
            anneal_steps = int(0.2 * self.max_steps)
            current_noise_var = self.noise_var

            if step < anneal_steps:
                start_log = math.log(1.0)
                end_log = math.log(max(self.noise_var, 1e-4))
                progress = step / anneal_steps
                current_noise_var = math.exp(start_log + (end_log - start_log) * progress)

            # Batched BiG-AMP step (compiled or original)
            factors, factor_vars, next_prev_s, next_prev_svar = step_fn(
                factors, factor_vars, Y, F, hg.indices,
                damping=self.damping,
                noise_var=current_noise_var,
                is_ising=is_ising,
                prev_s=prev_s if self.onsager_correction else None,
                prev_svar=prev_svar if self.onsager_correction else None,
                onsager_correction=self.onsager_correction,
                prior_precision_base=self._norm.prior_precision_base,
                prior_variance=self._norm.prior_variance,
            )
            if self.onsager_correction:
                prev_s = next_prev_s
                prev_svar = next_prev_svar
            else:
                prev_s = None
                prev_svar = None

            # Progress callback (throttled)
            if step_callback and ((step + 1) % 20 == 0 or step == self.max_steps - 1):
                try:
                    step_callback(step + 1, self.max_steps)
                except Exception:
                    pass

        # Compute final metrics for all samples
        Y_student = forward_pass_tensor_batch(factors, F, hg.indices)  # (S, C)
        teacher_norm = (Y ** 2).sum(dim=1)
        projection_dot = (Y * Y_student).sum(dim=1).abs()
        Q_Y_per_sample = torch.where(
            teacher_norm > 1e-12,
            projection_dot / (teacher_norm + 1e-12),
            torch.zeros_like(teacher_norm),
        )

        Q_Y_mean = Q_Y_per_sample.mean().item()
        Q_Y_std = Q_Y_per_sample.std().item() if S > 1 else 0.0
        qn_samples = []
        for s in range(S):
            qn_modes = compute_tensor_factor_projection_overlaps(
                teacher_factors,
                [factors[d][s] for d in range(self.order)],
            )
            qn_samples.append(sum(qn_modes) / len(qn_modes))
        qn_tensor = torch.tensor(qn_samples, dtype=torch.float32)

        return {
            'Q_Y_mean': Q_Y_mean,
            'Q_Y_std': Q_Y_std,
            'Q_N_mean': qn_tensor.mean().item(),
            'Q_N_std': qn_tensor.std().item() if S > 1 else 0.0,
            'alpha': alpha,
            'C': C,
            'S': S,
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _create_teacher_factors(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Create n teacher factors from W and X matrices."""
        factors = []

        factors.append(W_teacher.to(self.device))

        if self.order >= 2:
            factors.append(X_teacher.T.to(self.device))

        for d in range(2, self.order):
            N_d = self.dims[d]

            # Additional tensor modes follow the same latent scale as W/X.
            scale = self._norm.latent_std

            torch.manual_seed(42 + d)
            factor_d = torch.randn(N_d, self.M, device=self.device) * scale
            factors.append(factor_d)

        return factors

    # =========================================================================
    # Legacy Interface
    # =========================================================================

    def train(
        self,
        teacher_factors: List[torch.Tensor],
        alpha_values: List[float],
        S: int,
        base_seed: int,
        device: torch.device,
        verbose: bool = False,
    ) -> List[Dict]:
        """Legacy train method for direct usage."""
        self.S = S  # Override instance S
        results = []

        for alpha_idx, alpha in enumerate(alpha_values):
            seed = base_seed + int(alpha * 100)

            if verbose:
                print(f"Alpha {alpha:.2f}, {S} samples (parallel)")

            result = self._train_parallel(teacher_factors, alpha, seed, device)

            if verbose:
                print(f"  Alpha {alpha:.2f}: Q_Y = {result['Q_Y_mean']:.4f} ± {result['Q_Y_std']:.4f}")

            results.append({
                'alpha': alpha,
                'alpha_idx': alpha_idx,
                'Q_Y': result['Q_Y_mean'],
                'Q_Y_std': result['Q_Y_std'],
                'Q_N': result['Q_N_mean'],
                'C': result['C'],
            })

        return results

    def create_teacher(
        self,
        device: torch.device,
        seed: int = 42,
        scale: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """Create random teacher factors."""
        torch.manual_seed(seed)
        if scale is None:
            scale = self._norm.latent_std
        return [
            torch.randn(N_d, self.M, device=device) * scale
            for N_d in self.dims
        ]

    def _spectral_initialization(
        self,
        supergraph,
        Y_flat: torch.Tensor,
        F_flat: torch.Tensor,
        factors: List[torch.Tensor],
        offset_indices: List[torch.Tensor],
        iterations: int = 30
    ) -> List[torch.Tensor]:
        """
        Spectral Initialization using Tensor Power Method with F correction.

        For Order 3+ tensors, random initialization + AMP cannot break symmetry.
        This method uses an iterative power method to find a good initial point.

        Key: Multiply by F to recover signal direction (since Y = F * prod(X)).
        """
        import math
        import logging
        logger = logging.getLogger(__name__)

        n = len(factors)
        A = factors[0].shape[0]
        M = factors[0].shape[2]
        S = supergraph.S
        C_max = supergraph.C_max
        SC = S * C_max

        # Expand Y and F for all alphas
        Y_exp = Y_flat.unsqueeze(0).expand(A, -1)  # (A, SC)
        F_exp = F_flat.unsqueeze(0)  # (1, SC, M)

        # Apply alpha mask
        mask = supergraph.alpha_mask.unsqueeze(1).expand(A, S, C_max).reshape(A, -1)
        Y_masked = Y_exp * mask.float()

        logger.info(f"Spectral Init: Running {iterations} iterations with F correction...")

        alpha_scale = 1.0 / math.sqrt(M)

        for it in range(iterations):
            # Sequential (Gauss-Seidel) update: update factors[d] in place
            for d in range(n):
                # Gather CURRENT factors (not from start of iteration)
                gathered = torch.stack([
                    factors[i][:, offset_indices[i].long()]
                    for i in range(n)
                ])  # (n, A, SC, M)

                # Product of other factors
                other_indices = [i for i in range(n) if i != d]
                other_prod = torch.stack([gathered[i] for i in other_indices]).prod(dim=0)  # (A, SC, M)

                # Gradient: Y * F * other_product (Key: multiply by F!)
                # Y = sum_m F_m * prod_d X_d[:,m], so gradient w.r.t. X_d is Y * F * prod_{d'!=d} X_d'
                grad = Y_masked.unsqueeze(2) * F_exp * other_prod * alpha_scale  # (A, SC, M)

                # Scatter Add to (A, SN, M)
                N_d = factors[d].shape[1]
                update = torch.zeros_like(factors[d])
                idx_exp = offset_indices[d].unsqueeze(0).unsqueeze(2).expand(A, -1, M)
                update.scatter_add_(1, idx_exp, grad.to(dtype=update.dtype))

                # Normalize to unit variance
                std = update.std(dim=[1, 2], keepdim=True) + 1e-10
                factors[d] = update / std

        return factors
