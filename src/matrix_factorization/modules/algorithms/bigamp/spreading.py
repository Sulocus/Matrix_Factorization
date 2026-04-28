"""
BiG-AMP with Random Spreading - Parallel Implementation.

This module implements BiG-AMP algorithm for the random spreading model
with Super-Graph parallelization across alpha values.

Key features:
1. Configurable F distribution: gaussian or ising
2. Super-Graph strategy: parallel processing of all alphas
3. Teacher type controlled by config.teacher_key (reuses existing system)

Physical model:
    Y_ij = (1/√M) Σ_μ F_ij,μ W_iμ X_μj

where F is quenched random disorder that breaks loop correlations.
"""

from typing import Any, Tuple, Callable, Dict, Optional, List
import math
import gc
from pathlib import Path
import datetime
import hashlib
import torch

from matrix_factorization.modules.registry import register_algorithm
from matrix_factorization.modules.algorithms.base import AlgorithmBase
from matrix_factorization.core.contracts import AlgorithmStateView
from matrix_factorization.core.distributions import (
    F_DISTRIBUTION_GAUSSIAN,
    F_DISTRIBUTION_ISING,
    normalize_f_distribution,
)
from matrix_factorization.core.experiment.config import resolve_normalization_profile
from matrix_factorization.modules.graphs.supergraph import SuperGraphData, create_supergraph
from matrix_factorization.modules.graphs.supergraph_general import SuperGraphDataGeneral, create_supergraph_general, EDGE_TYPE_WW, EDGE_TYPE_WX, EDGE_TYPE_XX
from matrix_factorization.modules.teachers.random_spreading import SpreadingDataParallel

from .f_gen import (
    generate_F_gaussian, generate_F_ising, F_GENERATORS,
    generate_F_super, compute_Y_super,
    generate_F_super_general, compute_Y_super_general
)
from .core import (
    forward_pass_parallel, compute_variance_parallel, scatter_add_parallel
)
from .step import (
    bigamp_spreading_step, bigamp_step_disjoint_union,
    compute_log_likelihood, bigamp_step_disjoint_union_flat_adaptive,
    bigamp_step_disjoint_union_flat, bigamp_step_disjoint_union_flat_legacy_fast,
    bigamp_step_general_chunked,
    bigamp_step_disjoint_union_flat_general, compute_offset_indices,
    forward_disjoint_union_flat_corrected,
    clear_step_cache
)


# ============================================================================
# Global GPU Optimizations (Phase 1)
# ============================================================================
# Enable TF32 for Tensor Core acceleration on RTX 30/40/50 (Ampere+)
# TF32 provides FP32-level precision for most workloads with ~8x throughput
# This is a global setting that affects all matmul operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _stable_partition_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)




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
    - f_distribution: 'gaussian' or 'ising' - via config.spreading.f_distribution

    Usage:
        config = Config(
            algorithm_key="bigamp_spreading",
            teacher_key="orthogonal",  # Controls W, X generation
            spreading=SpreadingConfig(f_distribution="ising"),
        )
    """

    # Class-level cache for compiled step function
    _compiled_step = None
    _compiled_step_corrected = None
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
        cls._compiled_step_corrected = None
        cls._compiled_step_adaptive = None
        cls._compiled_step_general = None

        # Clear external module caches
        clear_step_cache()

        try:
            import torch._dynamo
            torch._dynamo.reset()  # Clear torch.compile internal caches
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
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
        self.debug_verbose = getattr(config.algorithm_params, 'debug_verbose', False)
        self.normalization_profile = getattr(config.algorithm_params, "normalization_profile", "paper_sparse_sampling")
        self._norm = resolve_normalization_profile(self.normalization_profile, config.matrix.M)
        self.seed_partition_policy = getattr(config.algorithm_params, 'seed_partition_policy', 'legacy')
        if self.seed_partition_policy not in {'legacy', 'partition_invariant'}:
            raise ValueError(
                "algorithm_params.seed_partition_policy must be 'legacy' or 'partition_invariant', "
                f"got {self.seed_partition_policy!r}"
            )
        self.requested_use_tf32 = getattr(config.algorithm_params, 'use_tf32', True)
        torch.backends.cuda.matmul.allow_tf32 = bool(self.requested_use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(self.requested_use_tf32)

        # Spreading configuration
        spreading_cfg = config.spreading
        if spreading_cfg is not None:
            self.f_distribution = normalize_f_distribution(spreading_cfg.f_distribution)
            self.spreading_seed = spreading_cfg.seed
            self.onsager_correction = getattr(spreading_cfg, 'onsager_correction', False)
            self.allow_intra_connection = getattr(spreading_cfg, 'allow_intra_connection', False)
            # Default chunk_size to 0 (Unchunked) to utilize ParallelCoordinator's dynamic batching
            # instead of inefficient Python-level looping.
            self.chunk_size = getattr(spreading_cfg, 'chunk_size', 0) 
        else:
            # Default values
            self.f_distribution = F_DISTRIBUTION_GAUSSIAN
            self.spreading_seed = 12345
            self.onsager_correction = False
            self.allow_intra_connection = False
            self.chunk_size = 0

        # Log configuration for debugging
        import logging
        _logger = logging.getLogger(__name__)
        _logger.debug(f"BiGAMPSpreading init: onsager_correction={self.onsager_correction}, "
                      f"allow_intra_connection={self.allow_intra_connection}, "
                      f"f_distribution={self.f_distribution}")

        # Validate f_distribution
        if self.f_distribution not in F_GENERATORS:
            raise ValueError(
                f"Invalid f_distribution='{self.f_distribution}'. "
                f"Available: {list(F_GENERATORS.keys())}"
            )

        # torch.compile for kernel fusion.  The spreading flat routes keep the
        # legacy fast tensor algebra, but avoid reduce-overhead CUDA graph
        # capture because canonical scans switch between no-Onsager and
        # stateful Onsager groups in one process.
        self.requested_use_compile = getattr(config.algorithm_params, 'use_compile', True)
        self.compile_fallback_policy = getattr(config.algorithm_params, 'compile_fallback_policy', 'allow')
        if self.compile_fallback_policy not in {'allow', 'error'}:
            raise ValueError(
                "algorithm_params.compile_fallback_policy must be 'allow' or 'error', "
                f"got {self.compile_fallback_policy!r}"
            )
        self.adaptive_damping_requested = bool(getattr(config.algorithm_params, 'adaptive_damping', False))
        if not self.onsager_correction:
            self.onsager_update_route = "legacy_no_onsager"
        elif self.adaptive_damping_requested:
            self.onsager_update_route = "corrected_adaptive_onsager"
        else:
            self.onsager_update_route = "corrected_fixed_onsager"
        self.use_compile = bool(self.requested_use_compile)
        self.compile_attempts = []
        self.compile_disabled_reason = ""
        self.compile_cuda_graphs_enabled = False
        if self.use_compile and self.onsager_update_route in {
            "corrected_fixed_onsager",
            "corrected_adaptive_onsager",
        }:
            target = (
                "bigamp_step_disjoint_union_flat_adaptive_corrected"
                if self.onsager_update_route == "corrected_adaptive_onsager"
                else "bigamp_step_disjoint_union_flat_corrected"
            )
            reason = "corrected_onsager_compile_disabled_cuda_allocator_guard"
            self._record_compile_attempt(target, False, f"disabled:{reason}")
            if self.compile_fallback_policy == 'error':
                raise RuntimeError(
                    f"torch.compile disabled for {target} because {reason} and "
                    "algorithm_params.compile_fallback_policy='error'"
                )
            self.use_compile = False
            self.compile_disabled_reason = reason
        if (
            self.use_compile
            and not self.adaptive_damping_requested
            and self.onsager_update_route == "legacy_no_onsager"
            and BiGAMPSpreading._compiled_step is None
        ):
            try:
                BiGAMPSpreading._compiled_step = torch.compile(
                    bigamp_step_disjoint_union_flat_legacy_fast,
                    mode='default',
                    fullgraph=False,
                )
                self.compile_cuda_graphs_enabled = False
                self._record_compile_attempt("bigamp_step_disjoint_union_flat", True, "")
            except Exception as exc:
                self.use_compile = False
                self._record_compile_attempt(
                    "bigamp_step_disjoint_union_flat",
                    False,
                    type(exc).__name__,
                )
                self._handle_compile_failure("bigamp_step_disjoint_union_flat", exc)
        # Corrected fixed Onsager carries prev_s/prev_svar across steps.  Compile
        # it in default mode only; CUDA graph capture is not used for canonical
        # spreading scans.
        if (
            self.use_compile
            and not self.adaptive_damping_requested
            and self.onsager_update_route == "corrected_fixed_onsager"
            and BiGAMPSpreading._compiled_step_corrected is None
        ):
            try:
                BiGAMPSpreading._compiled_step_corrected = torch.compile(
                    bigamp_step_disjoint_union_flat,
                    mode='default',
                    fullgraph=False,
                )
                self.compile_cuda_graphs_enabled = False
                self._record_compile_attempt("bigamp_step_disjoint_union_flat_corrected", True, "")
            except Exception as exc:
                self.use_compile = False
                self._record_compile_attempt(
                    "bigamp_step_disjoint_union_flat_corrected",
                    False,
                    type(exc).__name__,
                )
                self._handle_compile_failure("bigamp_step_disjoint_union_flat_corrected", exc)
        
        # Adaptive Onsager has its own compiled corrected step.  Use default
        # mode to avoid reduce-overhead CUDA graph capture across changing
        # accepted/rejected damping states; fall back only on observed compile
        # or runtime failure.
        if (
            self.use_compile
            and self.adaptive_damping_requested
            and self.onsager_update_route == "corrected_adaptive_onsager"
            and BiGAMPSpreading._compiled_step_adaptive is None
        ):
            try:
                BiGAMPSpreading._compiled_step_adaptive = torch.compile(
                    bigamp_step_disjoint_union_flat_adaptive,
                    mode='default',
                    fullgraph=False,
                )
                self._record_compile_attempt(
                    "bigamp_step_disjoint_union_flat_adaptive_corrected",
                    True,
                    "",
                )
            except Exception as exc:
                self.use_compile = False
                self._record_compile_attempt(
                    "bigamp_step_disjoint_union_flat_adaptive_corrected",
                    False,
                    type(exc).__name__,
                )
                self._handle_compile_failure("bigamp_step_disjoint_union_flat_adaptive_corrected", exc)

        # Precision policy: profile is the source of truth; legacy use_bf16 is an alias.
        self.precision_profile = getattr(
            config.algorithm_params,
            'precision_profile',
            'fast' if getattr(config.algorithm_params, 'use_bf16', True) else 'safe',
        )
        if self.precision_profile == 'fast' and getattr(config.algorithm_params, 'use_bf16', True) is False:
            self.precision_profile = 'safe'
        self.precision_fallback_policy = getattr(
            config.algorithm_params,
            'precision_fallback_policy',
            'allow',
        )
        if self.precision_fallback_policy == 'allow' and getattr(config.algorithm_params, 'dtype_fallback_policy', 'allow') == 'error':
            self.precision_fallback_policy = 'error'
        self.requested_use_bf16 = self.precision_profile in {'fast', 'aggressive'}
        self.dtype_fallback_policy = self.precision_fallback_policy
        if self.dtype_fallback_policy not in {'allow', 'error'}:
            raise ValueError(
                "algorithm_params.dtype_fallback_policy must be 'allow' or 'error', "
                f"got {self.dtype_fallback_policy!r}"
            )
        self.use_bf16 = False
        self.storage_dtype = torch.float32
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if self.requested_use_bf16 and bf16_supported:
            self.use_bf16 = True
            self.storage_dtype = torch.bfloat16
            self.dtype_status = "bf16_requested_and_effective"
        elif self.requested_use_bf16:
            self.dtype_status = "fallback_to_float32_bf16_unavailable"
            if self.dtype_fallback_policy == 'error':
                raise RuntimeError(
                    "BF16 was requested but is unavailable and "
                    "algorithm_params.dtype_fallback_policy='error'"
                )
        else:
            self.dtype_status = "bf16_disabled_by_config"

        self._contract_execution_metadata = self._build_spreading_execution_metadata([])

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

    def _compile_status_for_spreading_path(self) -> str:
        if not bool(getattr(self, "requested_use_compile", True)):
            return "disabled_by_config"
        route = getattr(self, "onsager_update_route", "legacy_no_onsager")
        if route == "corrected_adaptive_onsager":
            if bool(getattr(self, "use_compile", False)) and BiGAMPSpreading._compiled_step_adaptive is not None:
                return "effective_for_corrected_adaptive_spreading_step"
            return "fallback_to_eager_corrected_adaptive_spreading_step"
        if route == "corrected_fixed_onsager":
            if bool(getattr(self, "use_compile", False)) and BiGAMPSpreading._compiled_step_corrected is not None:
                return "effective_for_corrected_spreading_step"
            return "fallback_to_eager_corrected_spreading_step"
        if bool(getattr(self, "use_compile", False)) and BiGAMPSpreading._compiled_step is not None:
            return "effective_for_legacy_no_onsager_spreading_step"
        return "fallback_to_eager_legacy_no_onsager_spreading_step"

    def _build_spreading_execution_metadata(
        self,
        alpha_values: List[float],
        dynamic_batches: Optional[List[Tuple[int, int, float]]] = None,
    ) -> Dict[str, Any]:
        chunk_size = int(getattr(self, "chunk_size", 0) or 0)
        norm = getattr(self, "_norm", None)
        normalization_profile = getattr(self, "normalization_profile", "paper_sparse_sampling")
        return {
            "path": "bigamp_spreading_general",
            "normalization_schema_version": getattr(norm, "schema_version", 5),
            "normalization_profile": normalization_profile,
            "normalization_convention": getattr(norm, "convention_label", "paper_sparse_sampling_var1_latent_unit_prior"),
            "onsager_update_route": getattr(self, "onsager_update_route", "legacy_no_onsager"),
            "precision_profile": getattr(self, "precision_profile", "fast"),
            "precision_fallback_policy": getattr(self, "precision_fallback_policy", "allow"),
            "chunk_size": chunk_size,
            "chunking_enabled": chunk_size > 0,
            "chunk_policy": "manual_config" if chunk_size > 0 else "disabled_legacy_unchunked",
            "requested_use_compile": bool(getattr(self, "requested_use_compile", False)),
            "effective_use_compile": bool(
                getattr(self, "use_compile", False)
                and (
                    BiGAMPSpreading._compiled_step_adaptive is not None
                    if getattr(self, "onsager_update_route", "legacy_no_onsager") == "corrected_adaptive_onsager"
                    else (
                        BiGAMPSpreading._compiled_step_corrected is not None
                        if getattr(self, "onsager_update_route", "legacy_no_onsager") == "corrected_fixed_onsager"
                        else BiGAMPSpreading._compiled_step is not None
                    )
                )
            ),
            "compile_fallback_policy": getattr(self, "compile_fallback_policy", "allow"),
            "compile_status": self._compile_status_for_spreading_path(),
            "compile_cuda_graphs_enabled": bool(getattr(self, "compile_cuda_graphs_enabled", False)),
            "compile_disabled_reason": getattr(self, "compile_disabled_reason", ""),
            "compile_attempts": list(getattr(self, "compile_attempts", [])),
            "requested_use_tf32": bool(getattr(self, "requested_use_tf32", True)),
            "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
            "requested_use_bf16": bool(getattr(self, "requested_use_bf16", True)),
            "effective_use_bf16": bool(getattr(self, "use_bf16", False)),
            "dtype_fallback_policy": getattr(self, "dtype_fallback_policy", "allow"),
            "dtype_status": getattr(self, "dtype_status", ""),
            "storage_dtype": str(getattr(self, "storage_dtype", torch.float32)).replace("torch.", ""),
            "alpha_values": [float(alpha) for alpha in alpha_values],
            "dynamic_batches": [
                {
                    "alpha_start": int(start),
                    "alpha_end": int(end),
                    "alpha_max": float(alpha_max),
                }
                for start, end, alpha_max in (dynamic_batches or [])
            ],
            "seed_partition_policy": getattr(self, "seed_partition_policy", "legacy"),
            "seed_partition": (
                "partition_invariant_alpha_sample_role_step"
                if self._uses_partition_invariant_seed_policy()
                else "seed + batch_idx"
            ),
            "metadata_only": True,
            "notes": "Spreading execution metadata only; chunk_size is manual config and no auto tuning is applied.",
        }

    def _uses_partition_invariant_seed_policy(self) -> bool:
        return getattr(self, "seed_partition_policy", "legacy") == "partition_invariant"

    def _spreading_batch_seed(self, seed: int, batch_idx: int) -> int:
        if self._uses_partition_invariant_seed_policy():
            return int(seed)
        return int(seed) + int(batch_idx)

    def _batch_alpha_values(
        self,
        spreading_data: SpreadingDataParallel,
        batch_alpha_indices: Optional[List[int]],
        batch_alpha_values: Optional[List[float]],
    ) -> List[float]:
        if batch_alpha_values is not None:
            return [float(alpha) for alpha in batch_alpha_values]
        if batch_alpha_indices is None:
            batch_alpha_indices = list(range(spreading_data.A))
        return [float(spreading_data.alpha_values[idx].item()) for idx in batch_alpha_indices]

    def _randn_partitioned_spreading_flat(
        self,
        *,
        alpha_values: List[float],
        sample_count: int,
        sample_offset: int = 0,
        node_count: int,
        latent_dim: int,
        seed: int,
        role: str,
        scale: float,
        mean: float = 0.0,
    ) -> torch.Tensor:
        alpha_blocks = []
        for alpha in alpha_values:
            sample_blocks = []
            alpha_token = f"{float(alpha):.12g}"
            for local_sample_idx in range(sample_count):
                sample_idx = int(sample_offset) + local_sample_idx
                gen = torch.Generator(device=self.device).manual_seed(
                    _stable_partition_seed(
                        seed,
                        "bigamp_spreading",
                        role,
                        alpha_token,
                        sample_idx,
                    )
                )
                sample_blocks.append(
                    torch.randn(
                        (node_count, latent_dim),
                        generator=gen,
                        device=self.device,
                        dtype=self.storage_dtype,
                    )
                    * scale
                    + mean
                )
            alpha_blocks.append(torch.stack(sample_blocks, dim=0).reshape(sample_count * node_count, latent_dim))
        return torch.stack(alpha_blocks, dim=0)

    def _student_init_mean(self) -> float:
        teacher_cfg = getattr(self.config, "teacher", None)
        if getattr(teacher_cfg, "init_distribution", "gaussian") != "biased_gaussian":
            return 0.0
        scale = self._norm.student_init_std
        return float(getattr(teacher_cfg, "mean_scale", 0.0)) * scale

    def _initialize_near_teacher_partitioned(
        self,
        *,
        alpha_values: List[float],
        teacher_tensor: torch.Tensor,
        init_overlap: float,
        sample_count: int,
        sample_offset: int = 0,
        seed: int,
        role: str,
    ) -> torch.Tensor:
        node_count, latent_dim = teacher_tensor.shape
        teacher_expanded = (
            teacher_tensor.unsqueeze(0)
            .unsqueeze(0)
            .expand(len(alpha_values), sample_count, -1, -1)
            .reshape(len(alpha_values), sample_count * node_count, latent_dim)
            .to(self.device, dtype=self.storage_dtype)
        )
        noise = self._randn_partitioned_spreading_flat(
            alpha_values=alpha_values,
            sample_count=sample_count,
            sample_offset=sample_offset,
            node_count=node_count,
            latent_dim=latent_dim,
            seed=seed,
            role=role,
            scale=1.0,
        )
        coeff_signal = init_overlap
        coeff_noise = math.sqrt(1 - init_overlap ** 2)
        return coeff_signal * teacher_expanded + coeff_noise * noise

    def _randn_partitioned_restart_noise(
        self,
        *,
        alpha_values: List[float],
        sample_count: int,
        sample_offset: int = 0,
        node_count: int,
        latent_dim: int,
        seed: int,
        role: str,
        step: int,
        scale: float,
    ) -> torch.Tensor:
        alpha_blocks = []
        for alpha in alpha_values:
            sample_blocks = []
            alpha_token = f"{float(alpha):.12g}"
            for local_sample_idx in range(sample_count):
                sample_idx = int(sample_offset) + local_sample_idx
                gen = torch.Generator(device=self.device).manual_seed(
                    _stable_partition_seed(
                        seed,
                        "bigamp_spreading",
                        "restart_noise",
                        role,
                        alpha_token,
                        sample_idx,
                        int(step),
                    )
                )
                sample_blocks.append(
                    torch.randn(
                        (node_count, latent_dim),
                        generator=gen,
                        device=self.device,
                        dtype=self.storage_dtype,
                    )
                    * scale
                )
            alpha_blocks.append(torch.stack(sample_blocks, dim=0).reshape(sample_count * node_count, latent_dim))
        return torch.stack(alpha_blocks, dim=0)


    @staticmethod
    def _initialize_near_teacher(
        target_shape: Tuple[int, int, int],  # (B, S*N, M)
        teacher_tensor: torch.Tensor,         # (N, M)
        m_init: float,
        S: int,
        device: torch.device,
        dtype: torch.dtype,
        noise_scale: float = 1.0,
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
        noise = torch.randn(target_shape, device=device, dtype=dtype) * noise_scale
        
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
        sample_offset: int = 0,
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
                sample_offset=sample_offset,
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
                sample_offset=sample_offset,
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

        if getattr(self, "precision_profile", "fast") == "aggressive" and getattr(self, "use_bf16", False):
            if torch.is_floating_point(F_super):
                F_super = F_super.to(self.storage_dtype)
            Y_super = Y_super.to(self.storage_dtype)

        return SpreadingDataParallel(
            supergraph=supergraph,
            F_super=F_super,
            Y_super=Y_super,
            M=M,
            alpha_values=torch.tensor(alpha_values, device=self.device),
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            f_distribution=self.f_distribution,
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

        # Initialize student variables in the selected normalization profile.
        norm = resolve_normalization_profile(self.normalization_profile, M)
        scale = norm.student_init_std
        W_hat = torch.randn(A, N1, M, device=self.device) * scale
        X_hat = torch.randn(A, M, N2, device=self.device) * scale
        W_var = torch.ones(A, N1, M, device=self.device) * norm.prior_variance
        X_var = torch.ones(A, M, N2, device=self.device) * norm.prior_variance

        prev_s = None
        prev_svar = None

        # BiG-AMP iterations
        for step in range(self.max_steps):
            W_hat, X_hat, W_var, X_var, next_prev_s, next_prev_svar = bigamp_spreading_step(
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
                prior_precision_base=norm.prior_precision_base,
                prior_variance=norm.prior_variance,
                prev_s=prev_s,
                prev_svar=prev_svar,
            )
            if self.onsager_correction:
                prev_s = next_prev_s
                prev_svar = next_prev_svar
            else:
                prev_s = None
                prev_svar = None

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
        base_seed: Optional[int] = None,
        sample_offset: int = 0,
        initial_state: Optional[AlgorithmStateView] = None,
        return_continuation_state: bool = False,
        continuation_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train all samples in parallel using Disjoint Union with optimized flat tensors.

        OPTIMIZATIONS APPLIED:
        1. All tensors stored in flat format (A, S*N, M) - no per-iteration reshape
        2. Pre-flattened F, Y, alpha_mask computed once
        3. torch.compile for kernel fusion (if enabled)
        4. Ising F² optimization (F²=1 skips pow(2))

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
                spreading_data,
                batch_alpha_indices,
                verbose,
                step_callback,
                max_steps,
                batch_alpha_values,
                base_seed,
                sample_offset=sample_offset,
                initial_state=initial_state,
                return_continuation_state=return_continuation_state,
                continuation_context=continuation_context,
            )

        # Check for Adaptive Damping
        if hasattr(self.config.algorithm_params, 'adaptive_damping') and self.config.algorithm_params.adaptive_damping:
            return self._train_full_parallel_adaptive(
                spreading_data,
                batch_alpha_indices,
                verbose,
                step_callback,
                max_steps,
                batch_alpha_values,
                base_seed,
                sample_offset=sample_offset,
                initial_state=initial_state,
                return_continuation_state=return_continuation_state,
                continuation_context=continuation_context,
            )


        # Determine which alphas to train
        if batch_alpha_indices is None:
            batch_alpha_indices = list(range(A))
        B = len(batch_alpha_indices)
        effective_alpha_values = self._batch_alpha_values(
            spreading_data, batch_alpha_indices, batch_alpha_values
        )
        if self._uses_partition_invariant_seed_policy() and base_seed is None:
            raise RuntimeError("bigamp_spreading partition_invariant initialization requires base_seed.")

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
        init_mean = self._student_init_mean()

        if init_mode == 'teacher' and self._uses_partition_invariant_seed_policy():
            W_flat = self._initialize_near_teacher_partitioned(
                alpha_values=effective_alpha_values,
                teacher_tensor=spreading_data.W_teacher,
                init_overlap=init_overlap,
                sample_count=S,
                sample_offset=sample_offset,
                seed=base_seed,
                role="W_student",
            )
            X_flat = self._initialize_near_teacher_partitioned(
                alpha_values=effective_alpha_values,
                teacher_tensor=spreading_data.X_teacher.T,
                init_overlap=init_overlap,
                sample_count=S,
                sample_offset=sample_offset,
                seed=base_seed,
                role="X_student",
            )
            if verbose:
                print(f"  [Init] Teacher-Assisted (m={init_overlap})")
        elif init_mode == 'teacher':
            # Teacher-Assisted Initialization (Warm Start for Hysteresis Analysis)
            W_flat = self._initialize_near_teacher(
                (B, S * N1, M),
                spreading_data.W_teacher,  # (N1, M)
                init_overlap,
                S,
                self.device,
                self.storage_dtype,
            )
            X_flat = self._initialize_near_teacher(
                (B, S * N2, M),
                spreading_data.X_teacher.T,  # (M, N2) -> (N2, M)
                init_overlap,
                S,
                self.device,
                self.storage_dtype,
            )
            if verbose:
                print(f"  [Init] Teacher-Assisted (m={init_overlap})")
        else:
            # Random Initialization (Cold Start - default)
            if self._uses_partition_invariant_seed_policy():
                W_flat = self._randn_partitioned_spreading_flat(
                    alpha_values=effective_alpha_values,
                    sample_count=S,
                    sample_offset=sample_offset,
                    node_count=N1,
                    latent_dim=M,
                    seed=base_seed,
                    role="W_student",
                    scale=0.1,
                    mean=init_mean,
                )
                X_flat = self._randn_partitioned_spreading_flat(
                    alpha_values=effective_alpha_values,
                    sample_count=S,
                    sample_offset=sample_offset,
                    node_count=N2,
                    latent_dim=M,
                    seed=base_seed,
                    role="X_student",
                    scale=0.1,
                    mean=init_mean,
                )
            else:
                W_flat = torch.randn(B, S * N1, M, device=self.device, dtype=self.storage_dtype) * 0.1 + init_mean
                X_flat = torch.randn(B, S * N2, M, device=self.device, dtype=self.storage_dtype) * 0.1 + init_mean

        W_var_flat = torch.ones(B, S * N1, M, device=self.device, dtype=self.storage_dtype)
        X_var_flat = torch.ones(B, S * N2, M, device=self.device, dtype=self.storage_dtype)


        prev_s = None
        prev_svar = None
        if initial_state is not None:
            W_flat, X_flat, W_var_flat, X_var_flat, prev_s, prev_svar = self._coerce_spreading_continuation_state(
                initial_state=initial_state,
                B=B,
                S=S,
                N1=N1,
                N2=N2,
                M=M,
                SC=SC,
                alpha_mask_exp=alpha_mask_exp,
            )
        is_ising = (self.f_distribution == 'ising')

        # ===== OPTIMIZATION 3: Route by Onsager convention =====
        # no_onsager keeps the legacy no-feedback fast route.  Once Onsager is
        # enabled, use the corrected BiG-AMP pvar/svar/gain convention.
        route = getattr(self, "onsager_update_route", "legacy_no_onsager")
        if route == "corrected_fixed_onsager":
            step_fn = (
                BiGAMPSpreading._compiled_step_corrected
                if self.use_compile and BiGAMPSpreading._compiled_step_corrected is not None
                else bigamp_step_disjoint_union_flat
            )
            compiled_step_active = step_fn is BiGAMPSpreading._compiled_step_corrected
        else:
            step_fn = (
                BiGAMPSpreading._compiled_step
                if self.use_compile and BiGAMPSpreading._compiled_step is not None
                else bigamp_step_disjoint_union_flat_legacy_fast
            )
            compiled_step_active = step_fn is BiGAMPSpreading._compiled_step
        compiled_step_uses_cuda_graph = (
            compiled_step_active
            and bool(getattr(self, "compile_cuda_graphs_enabled", False))
        )

        # Use provided max_steps or fall back to config
        steps = max_steps if max_steps is not None else self.max_steps

        # BiG-AMP iterations with optimized flat function
        for step in range(steps):
            while True:
                if compiled_step_uses_cuda_graph:
                    torch.compiler.cudagraph_mark_step_begin()

                call_prev_s = prev_s if route == "corrected_fixed_onsager" else None
                call_prev_svar = prev_svar if route == "corrected_fixed_onsager" else None
                try:
                    W_flat, X_flat, W_var_flat, X_var_flat, next_prev_s, next_prev_svar = step_fn(
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
                        prior_precision_base=self._norm.prior_precision_base,
                        prior_variance=self._norm.prior_variance,
                        is_ising=is_ising,
                        prev_s=call_prev_s,
                        prev_svar=call_prev_svar,
                    )

                    if route == "corrected_fixed_onsager":
                        prev_s = next_prev_s
                        prev_svar = next_prev_svar
                    else:
                        prev_s = None
                        prev_svar = None

                    if compiled_step_uses_cuda_graph:
                        W_flat = W_flat.clone()
                        X_flat = X_flat.clone()
                        W_var_flat = W_var_flat.clone()
                        X_var_flat = X_var_flat.clone()
                        if prev_s is not None:
                            prev_s = prev_s.clone()
                        if prev_svar is not None:
                            prev_svar = prev_svar.clone()
                    break
                except RuntimeError as exc:
                    if route != "corrected_fixed_onsager" or step_fn is not BiGAMPSpreading._compiled_step_corrected:
                        raise
                    BiGAMPSpreading._compiled_step_corrected = None
                    self.use_compile = False
                    self._record_compile_attempt(
                        "bigamp_step_disjoint_union_flat_corrected",
                        False,
                        f"runtime:{type(exc).__name__}",
                    )
                    self.compile_disabled_reason = "corrected_fixed_compile_runtime_fallback"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    step_fn = bigamp_step_disjoint_union_flat
                    compiled_step_active = False
                    compiled_step_uses_cuda_graph = False

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

        if return_continuation_state:
            self._last_continuation_state = self._build_spreading_continuation_state(
                W_hat,
                X_hat,
                W_var_flat,
                X_var_flat,
                prev_s,
                prev_svar,
                route=route,
                S=S,
                B=B,
                N1=N1,
                N2=N2,
                M=M,
                steps=steps,
                alpha_values=effective_alpha_values,
                W_teacher=spreading_data.W_teacher,
                X_teacher=spreading_data.X_teacher,
            )
        else:
            self._last_continuation_state = None

        return W_hat, X_hat

    def _train_full_parallel_adaptive(
        self,
        spreading_data: SpreadingDataParallel,
        batch_alpha_indices: List[int],
        verbose: bool,
        step_callback,
        max_steps: Optional[int],
        batch_alpha_values: Optional[List[float]] = None,
        base_seed: Optional[int] = None,
        sample_offset: int = 0,
        initial_state: Optional[AlgorithmStateView] = None,
        return_continuation_state: bool = False,
        continuation_context: Optional[Dict[str, Any]] = None,
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
        effective_alpha_values = self._batch_alpha_values(
            spreading_data, batch_alpha_indices, batch_alpha_values
        )
        if self._uses_partition_invariant_seed_policy() and base_seed is None:
            raise RuntimeError("bigamp_spreading partition_invariant initialization requires base_seed.")
        
        # Determine params
        params = self.config.algorithm_params
        step_min = getattr(params, 'step_min', 0.05)
        step_max = getattr(params, 'step_max', 0.5)
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
        init_mean = self._student_init_mean()

        if init_mode == 'teacher' and self._uses_partition_invariant_seed_policy():
            W_flat = self._initialize_near_teacher_partitioned(
                alpha_values=effective_alpha_values,
                teacher_tensor=spreading_data.W_teacher,
                init_overlap=init_overlap,
                sample_count=S,
                sample_offset=sample_offset,
                seed=base_seed,
                role="W_student",
            )
            X_flat = self._initialize_near_teacher_partitioned(
                alpha_values=effective_alpha_values,
                teacher_tensor=spreading_data.X_teacher.T,
                init_overlap=init_overlap,
                sample_count=S,
                sample_offset=sample_offset,
                seed=base_seed,
                role="X_student",
            )
            if verbose:
                print(f"  [Init] Teacher-Assisted Adaptive (m={init_overlap})")
        elif init_mode == 'teacher':
            # Teacher-Assisted Initialization (Warm Start for Hysteresis Analysis)
            W_flat = self._initialize_near_teacher(
                (B, S * N1, M),
                spreading_data.W_teacher,
                init_overlap,
                S,
                self.device,
                self.storage_dtype,
            )
            X_flat = self._initialize_near_teacher(
                (B, S * N2, M),
                spreading_data.X_teacher.T,
                init_overlap,
                S,
                self.device,
                self.storage_dtype,
            )
            if verbose:
                print(f"  [Init] Teacher-Assisted Adaptive (m={init_overlap})")
        else:
            # Random Initialization (Cold Start - default)
            if self._uses_partition_invariant_seed_policy():
                W_flat = self._randn_partitioned_spreading_flat(
                    alpha_values=effective_alpha_values,
                    sample_count=S,
                    sample_offset=sample_offset,
                    node_count=N1,
                    latent_dim=M,
                    seed=base_seed,
                    role="W_student",
                    scale=0.1,
                    mean=init_mean,
                )
                X_flat = self._randn_partitioned_spreading_flat(
                    alpha_values=effective_alpha_values,
                    sample_count=S,
                    sample_offset=sample_offset,
                    node_count=N2,
                    latent_dim=M,
                    seed=base_seed,
                    role="X_student",
                    scale=0.1,
                    mean=init_mean,
                )
            else:
                W_flat = torch.randn(B, S * N1, M, device=self.device, dtype=self.storage_dtype) * 0.1 + init_mean
                X_flat = torch.randn(B, S * N2, M, device=self.device, dtype=self.storage_dtype) * 0.1 + init_mean

        W_var_flat = torch.ones(B, S * N1, M, device=self.device, dtype=self.storage_dtype)
        X_var_flat = torch.ones(B, S * N2, M, device=self.device, dtype=self.storage_dtype)

        prev_s = None
        prev_svar = None
        if initial_state is not None:
            W_flat, X_flat, W_var_flat, X_var_flat, prev_s, prev_svar = self._coerce_spreading_continuation_state(
                initial_state=initial_state,
                B=B,
                S=S,
                N1=N1,
                N2=N2,
                M=M,
                SC=SC,
                alpha_mask_exp=alpha_mask_exp,
            )
        
        # "Safe" State (Last accepted) - ONLY clone at initialization
        # Subsequent saves will use reference swap to avoid memory explosion
        W_safe = W_flat.clone()
        X_safe = X_flat.clone()
        W_var_safe = W_var_flat.clone()
        X_var_safe = X_var_flat.clone()
        s_safe = None
        svar_safe = None
        
        current_val = -float('inf')
        
        step_fn = (
            BiGAMPSpreading._compiled_step_adaptive
            if self.use_compile and BiGAMPSpreading._compiled_step_adaptive is not None
            else bigamp_step_disjoint_union_flat_adaptive
        )
        
        is_ising = (self.f_distribution == 'ising')
        steps = max_steps if max_steps is not None else self.max_steps
        
        # Warm Restart Parameters
        adaptive_restart = getattr(params, 'adaptive_restart', False)
        restart_patience = getattr(params, 'restart_patience', 50)
        restart_noise = getattr(params, 'restart_noise', 0.1)
        acceptance_tolerance = getattr(params, 'acceptance_tolerance', 0.0)
        stuck_counter = torch.zeros(B, dtype=torch.long, device=self.device)

        # Debug Recording
        damp_history = None
        if self.debug_verbose and batch_alpha_values is not None:
            damp_history = torch.zeros(steps, B, dtype=torch.float32, device=self.device)

        for step in range(steps):
            beta_current = damping
            
            # Run step function to get raw updates
            try:
                W_raw, X_raw, W_var_raw, X_var_raw, s_vals, svar_vals, Z_hat, V = step_fn(
                    W_flat, X_flat, W_var_flat, X_var_flat,
                    Y_flat, F_flat, i_offset, j_offset, alpha_mask_exp,
                    S, N1, N2, self.noise_var,
                    self._norm.prior_precision_base,
                    self._norm.prior_variance,
                    is_ising, prev_s, prev_svar
                )
            except RuntimeError as exc:
                if step_fn is not BiGAMPSpreading._compiled_step_adaptive:
                    raise
                BiGAMPSpreading._compiled_step_adaptive = None
                self.use_compile = False
                self._record_compile_attempt(
                    "bigamp_step_disjoint_union_flat_adaptive_corrected",
                    False,
                    f"runtime:{type(exc).__name__}",
                )
                self.compile_disabled_reason = "adaptive_compile_runtime_fallback"
                torch.cuda.empty_cache()
                step_fn = bigamp_step_disjoint_union_flat_adaptive
                W_raw, X_raw, W_var_raw, X_var_raw, s_vals, svar_vals, Z_hat, V = step_fn(
                    W_flat, X_flat, W_var_flat, X_var_flat,
                    Y_flat, F_flat, i_offset, j_offset, alpha_mask_exp,
                    S, N1, N2, self.noise_var,
                    self._norm.prior_precision_base,
                    self._norm.prior_variance,
                    is_ising, prev_s, prev_svar
                )
            
            # Evaluate the actual damped candidate state.  The raw step output
            # is not the state we commit when beta < 1, so adaptive acceptance
            # must score the beta-mixed candidate rather than the pre-update or
            # beta=1 forward state.
            d_view = beta_current.view(B, 1, 1)
            W_accepted = d_view * W_raw + (1 - d_view) * W_flat
            X_accepted = d_view * X_raw + (1 - d_view) * X_flat
            W_var_accepted = d_view * W_var_raw + (1 - d_view) * W_var_flat
            X_var_accepted = d_view * X_var_raw + (1 - d_view) * X_var_flat
            Z_candidate, V_candidate = forward_disjoint_union_flat_corrected(
                W_accepted,
                X_accepted,
                W_var_accepted,
                X_var_accepted,
                F_flat,
                i_offset,
                j_offset,
                alpha_mask_exp,
                is_ising,
            )
            output_val = compute_log_likelihood(Y_flat, Z_candidate, V_candidate, self.noise_var)
            prior_penalty = 0.5 / float(self._norm.prior_variance) * (
                W_accepted.float().square().sum(dim=(1, 2))
                + X_accepted.float().square().sum(dim=(1, 2))
            )
            new_val = output_val - prior_penalty
            
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
            if svar_safe is None:
                 svar_safe = torch.zeros_like(svar_vals)
            
            W_safe = torch.where(pass_mask_3d, W_flat, W_safe)
            X_safe = torch.where(pass_mask_3d, X_flat, X_safe)
            W_var_safe = torch.where(pass_mask_3d, W_var_flat, W_var_safe)
            X_var_safe = torch.where(pass_mask_3d, X_var_flat, X_var_safe)
            if prev_s is not None and s_safe is not None:
                s_safe = torch.where(pass_mask_2d, prev_s, s_safe)
            if prev_svar is not None and svar_safe is not None:
                svar_safe = torch.where(pass_mask_2d, prev_svar, svar_safe)
            
            # --- 2. Update Likelihood & Damping ---
            current_val = torch.where(pass_mask, new_val, current_val)
            
            damping_next = torch.where(pass_mask,
                                       torch.clamp(damping * step_incr, max=step_max),
                                       torch.clamp(damping * step_decr, min=step_min))
            
            # --- 3. Compute Next State (Main Update) ---
            # If Accepted: New = Damping * Raw + (1-Damping) * Old
            # If Rejected: New = Safe (Backtrack)
            
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
            # If accepted: prev_s follows the same beta used for the accepted state.
            # If rejected: prev_s = s_safe (Backtrack)
            
            # --- ONSAGER CONTROL FIX (Adaptive) ---
            if self.onsager_correction:
                if prev_s is None:
                    prev_s_accepted = s_vals
                else:
                    prev_s_accepted = beta_current.view(B, 1) * s_vals + (1 - beta_current).view(B, 1) * prev_s
                if prev_svar is None:
                    prev_svar_accepted = svar_vals
                else:
                    prev_svar_accepted = beta_current.view(B, 1) * svar_vals + (1 - beta_current).view(B, 1) * prev_svar
                prev_s_rejected = s_safe
                prev_svar_rejected = svar_safe
                prev_s = torch.where(pass_mask_2d, prev_s_accepted, prev_s_rejected)
                prev_svar = torch.where(pass_mask_2d, prev_svar_accepted, prev_svar_rejected)
            else:
                prev_s = None
                prev_svar = None

            damping = damping_next

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
                
                # Perturb State. Legacy keeps using the current global RNG stream;
                # the opt-in policy makes restart noise independent of alpha batch partition.
                if self._uses_partition_invariant_seed_policy():
                    noise_W = self._randn_partitioned_restart_noise(
                        alpha_values=effective_alpha_values,
                        sample_count=S,
                        sample_offset=sample_offset,
                        node_count=N1,
                        latent_dim=M,
                        seed=base_seed,
                        role="W_student",
                        step=step,
                        scale=restart_noise,
                    )
                    noise_X = self._randn_partitioned_restart_noise(
                        alpha_values=effective_alpha_values,
                        sample_count=S,
                        sample_offset=sample_offset,
                        node_count=N2,
                        latent_dim=M,
                        seed=base_seed,
                        role="X_student",
                        step=step,
                        scale=restart_noise,
                    )
                else:
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
            debug_path = Path("results/debug_damping.pt")
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
        if return_continuation_state:
            self._last_continuation_state = self._build_spreading_continuation_state(
                W_hat,
                X_hat,
                W_var_flat,
                X_var_flat,
                prev_s,
                prev_svar,
                route="corrected_adaptive_onsager" if self.onsager_correction else "legacy_no_onsager",
                S=S,
                B=B,
                N1=N1,
                N2=N2,
                M=M,
                steps=steps,
                alpha_values=effective_alpha_values,
                W_teacher=spreading_data.W_teacher,
                X_teacher=spreading_data.X_teacher,
            )
        else:
            self._last_continuation_state = None
        return W_hat, X_hat

    def _coerce_spreading_continuation_state(
        self,
        *,
        initial_state: AlgorithmStateView,
        B: int,
        S: int,
        N1: int,
        N2: int,
        M: int,
        SC: int,
        alpha_mask_exp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        factors = initial_state.student_factors or {}
        variances = initial_state.factor_variances or {}
        if "W" not in factors or "X" not in factors:
            raise ValueError("bigamp_spreading continuation requires student_factors W and X")

        W = self._coerce_factor_tensor(factors["W"], (B, S, N1, M), self.storage_dtype)
        X = self._coerce_factor_tensor(factors["X"], (B, S, M, N2), self.storage_dtype)
        W_flat = W.reshape(B, S * N1, M)
        X_flat = X.permute(0, 1, 3, 2).reshape(B, S * N2, M)

        if "W_var" in variances:
            W_var = self._coerce_factor_tensor(variances["W_var"], (B, S, N1, M), self.storage_dtype)
            W_var_flat = W_var.reshape(B, S * N1, M)
        else:
            W_var_flat = torch.ones(B, S * N1, M, device=self.device, dtype=self.storage_dtype)
        if "X_var" in variances:
            X_var = self._coerce_factor_tensor(variances["X_var"], (B, S, M, N2), self.storage_dtype)
            X_var_flat = X_var.permute(0, 1, 3, 2).reshape(B, S * N2, M)
        else:
            X_var_flat = torch.ones(B, S * N2, M, device=self.device, dtype=self.storage_dtype)

        prev_s = None
        prev_svar = None
        residual = initial_state.onsager_residual or {}
        if self.onsager_correction and isinstance(residual, dict):
            if residual.get("prev_s") is not None:
                prev_s = self._coerce_residual_tensor(residual["prev_s"], (B, SC), self.storage_dtype)
                prev_s = prev_s * alpha_mask_exp.to(prev_s.dtype)
            if residual.get("prev_svar") is not None:
                prev_svar = self._coerce_residual_tensor(residual["prev_svar"], (B, SC), self.storage_dtype)
                prev_svar = prev_svar * alpha_mask_exp.to(prev_svar.dtype)
        return W_flat, X_flat, W_var_flat, X_var_flat, prev_s, prev_svar

    def _build_spreading_continuation_state(
        self,
        W_hat: torch.Tensor,
        X_hat: torch.Tensor,
        W_var_flat: torch.Tensor,
        X_var_flat: torch.Tensor,
        prev_s: Optional[torch.Tensor],
        prev_svar: Optional[torch.Tensor],
        *,
        route: str,
        S: int,
        B: int,
        N1: int,
        N2: int,
        M: int,
        steps: int,
        alpha_values: List[float],
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
    ) -> AlgorithmStateView:
        W_state = W_hat[:, 0].detach().clone() if B == 1 else W_hat.permute(1, 0, 2, 3).detach().clone()
        X_state = X_hat[:, 0].detach().clone() if B == 1 else X_hat.permute(1, 0, 2, 3).detach().clone()
        W_var = W_var_flat.reshape(B, S, N1, M)
        X_var = X_var_flat.reshape(B, S, N2, M).permute(0, 1, 3, 2)
        factor_variances = {
            "W_var": W_var[0].detach().clone() if B == 1 else W_var.detach().clone(),
            "X_var": X_var[0].detach().clone() if B == 1 else X_var.detach().clone(),
        }
        onsager_residual = None
        if route != "legacy_no_onsager" and prev_s is not None:
            onsager_residual = {
                "prev_s": prev_s.detach().clone(),
                "prev_svar": prev_svar.detach().clone() if prev_svar is not None else None,
            }
        return AlgorithmStateView(
            student_factors={"W": W_state, "X": X_state},
            factor_variances=factor_variances,
            onsager_residual=onsager_residual,
            teacher_factors={"W": W_teacher.detach(), "X": X_teacher.detach()},
            step_index=steps,
            alpha=float(alpha_values[0]) if len(alpha_values) == 1 else None,
            metadata={
                "algorithm_key": "bigamp_spreading",
                "continuation_state": "student_factors+factor_variances"
                + ("+onsager_residual" if onsager_residual is not None else ""),
                "onsager_update_route": route,
            },
        )

    def _coerce_spreading_general_continuation_state(
        self,
        *,
        initial_state: AlgorithmStateView,
        B: int,
        S: int,
        N1: int,
        N2: int,
        M: int,
        SC: int,
        alpha_mask_exp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        factors = initial_state.student_factors or {}
        variances = initial_state.factor_variances or {}
        if "W" not in factors or "X" not in factors:
            raise ValueError("bigamp_spreading general continuation requires student_factors W and X")

        W = self._coerce_factor_tensor(factors["W"], (B, S, N1, M), self.storage_dtype)
        X = self._coerce_factor_tensor(factors["X"], (B, S, M, N2), self.storage_dtype)
        W_flat = W.reshape(B, S * N1, M)
        X_flat = X.permute(0, 1, 3, 2).reshape(B, S * N2, M)
        V_flat = torch.cat([W_flat, X_flat], dim=1)

        if "W_var" in variances:
            W_var = self._coerce_factor_tensor(variances["W_var"], (B, S, N1, M), self.storage_dtype)
            W_var_flat = W_var.reshape(B, S * N1, M)
        else:
            W_var_flat = torch.ones(B, S * N1, M, device=self.device, dtype=self.storage_dtype)
        if "X_var" in variances:
            X_var = self._coerce_factor_tensor(variances["X_var"], (B, S, M, N2), self.storage_dtype)
            X_var_flat = X_var.permute(0, 1, 3, 2).reshape(B, S * N2, M)
        else:
            X_var_flat = torch.ones(B, S * N2, M, device=self.device, dtype=self.storage_dtype)
        V_var_flat = torch.cat([W_var_flat, X_var_flat], dim=1)

        prev_s = None
        prev_svar = None
        residual = initial_state.onsager_residual or {}
        if self.onsager_correction and isinstance(residual, dict):
            if residual.get("prev_s") is not None:
                prev_s = self._coerce_residual_tensor(residual["prev_s"], (B, SC), self.storage_dtype)
                prev_s = prev_s * alpha_mask_exp.to(prev_s.dtype)
            if residual.get("prev_svar") is not None:
                prev_svar = self._coerce_residual_tensor(residual["prev_svar"], (B, SC), self.storage_dtype)
                prev_svar = prev_svar * alpha_mask_exp.to(prev_svar.dtype)
        return V_flat, V_var_flat, prev_s, prev_svar

    def _build_spreading_general_continuation_state(
        self,
        W_hat: torch.Tensor,
        X_hat: torch.Tensor,
        V_var_flat: torch.Tensor,
        prev_s: Optional[torch.Tensor],
        prev_svar: Optional[torch.Tensor],
        *,
        route: str,
        S: int,
        B: int,
        N1: int,
        N2: int,
        M: int,
        steps: int,
        alpha_values: List[float],
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
    ) -> AlgorithmStateView:
        W_state = W_hat[:, 0].detach().clone() if B == 1 else W_hat.permute(1, 0, 2, 3).detach().clone()
        X_state = X_hat[:, 0].detach().clone() if B == 1 else X_hat.permute(1, 0, 2, 3).detach().clone()
        V_var = V_var_flat.reshape(B, S, N1 + N2, M)
        W_var = V_var[:, :, :N1, :]
        X_var = V_var[:, :, N1:, :].permute(0, 1, 3, 2)
        factor_variances = {
            "W_var": W_var[0].detach().clone() if B == 1 else W_var.detach().clone(),
            "X_var": X_var[0].detach().clone() if B == 1 else X_var.detach().clone(),
        }
        onsager_residual = None
        if route != "general_no_onsager" and prev_s is not None:
            onsager_residual = {
                "prev_s": prev_s.detach().clone(),
                "prev_svar": prev_svar.detach().clone() if prev_svar is not None else None,
            }
        return AlgorithmStateView(
            student_factors={"W": W_state, "X": X_state},
            factor_variances=factor_variances,
            onsager_residual=onsager_residual,
            teacher_factors={"W": W_teacher.detach(), "X": X_teacher.detach()},
            step_index=steps,
            alpha=float(alpha_values[0]) if len(alpha_values) == 1 else None,
            metadata={
                "algorithm_key": "bigamp_spreading",
                "continuation_state": "general_student_factors+factor_variances"
                + ("+onsager_residual" if onsager_residual is not None else ""),
                "onsager_update_route": route,
            },
        )

    def _coerce_factor_tensor(
        self,
        value: torch.Tensor,
        target_shape: Tuple[int, int, int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = value.detach().to(device=self.device, dtype=dtype)
        if tuple(tensor.shape) == target_shape:
            return tensor.clone()
        if tensor.dim() == 3 and target_shape[0] == 1 and tuple(tensor.shape) == target_shape[1:]:
            return tensor.unsqueeze(0).clone()
        raise ValueError(
            f"Continuation factor shape {tuple(tensor.shape)} cannot initialize target {target_shape}"
        )

    def _coerce_residual_tensor(
        self,
        value: torch.Tensor,
        target_shape: Tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = value.detach().to(device=self.device, dtype=dtype)
        if tuple(tensor.shape) == target_shape:
            return tensor.clone()
        if tensor.dim() == 1 and target_shape[0] == 1 and int(tensor.shape[0]) == target_shape[1]:
            return tensor.unsqueeze(0).clone()
        raise ValueError(
            f"Continuation residual shape {tuple(tensor.shape)} cannot initialize target {target_shape}"
        )

    def supports_batch_training(self) -> bool:
        """Returns True - this algorithm supports parallel alpha training."""
        return True

    def _estimate_flat_step_elements(self, *, alpha_count: int, alpha_max: float, sample_count: int) -> int:
        """Estimate (A, S*C_max, M) gathered elements for the flat spreading step."""
        n1 = int(self.config.matrix.N1)
        m = int(self.config.matrix.M)
        c_max = max(1, int(math.ceil(max(float(alpha_max), 0.0) * m * n1)))
        return int(alpha_count) * int(sample_count) * c_max * m

    def _compute_internal_spreading_alpha_batches(
        self,
        alpha_values: List[float],
        *,
        sample_count: int,
    ) -> List[Tuple[int, int, float]]:
        """Return internal alpha batches for the active spreading route.

        The legacy no-Onsager route is intentionally kept as one large compiled
        batch.  Corrected Onsager routes materialize several (A, S*C_max, M)
        tensors, so they need smaller alpha batches at the same S and alpha
        range.
        """
        if not alpha_values:
            return []

        route = getattr(self, "onsager_update_route", "legacy_no_onsager")
        if route == "legacy_no_onsager":
            return [(0, len(alpha_values), max(alpha_values))]

        target_elements = 160_000_000
        batches: List[Tuple[int, int, float]] = []
        start = 0
        while start < len(alpha_values):
            end = start + 1
            best_end = end
            best_alpha_max = float(alpha_values[start])
            while end <= len(alpha_values):
                alpha_max = max(float(alpha) for alpha in alpha_values[start:end])
                estimated = self._estimate_flat_step_elements(
                    alpha_count=end - start,
                    alpha_max=alpha_max,
                    sample_count=sample_count,
                )
                if end > start + 1 and estimated > target_elements:
                    break
                best_end = end
                best_alpha_max = alpha_max
                end += 1
            batches.append((start, best_end, best_alpha_max))
            start = best_end
        return batches

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
        spreading_data: Optional[SpreadingDataParallel] = None,
        sample_context: Optional[Dict[str, Any]] = None,
        initial_state: Optional[AlgorithmStateView] = None,
        return_continuation_state: bool = False,
        continuation_context: Optional[Dict[str, Any]] = None,
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
        sample_offset = int((sample_context or {}).get("sample_start", 0))
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        A = len(alpha_values)
        alpha_max = max(alpha_values) if alpha_values else 4.0

        if spreading_data is not None:
            continuation_active = (
                initial_state is not None
                or return_continuation_state
                or continuation_context is not None
            )
            if continuation_active and len(alpha_values) != 1:
                raise ValueError("provided spreading_data continuation path expects one alpha at a time")
            batch_alpha_indices = None
            if continuation_active:
                alpha_index = 0
                if isinstance(continuation_context, dict):
                    alpha_index = int(continuation_context.get("alpha_index", 0))
                batch_alpha_indices = [alpha_index]
                batch_seed = seed if self._uses_partition_invariant_seed_policy() else self._spreading_batch_seed(seed, 0)
                W_batch, X_batch = self.train_full_parallel(
                    spreading_data,
                    batch_alpha_indices=batch_alpha_indices,
                    verbose=False,
                    step_callback=step_callback,
                    max_steps=max_steps,
                    batch_alpha_values=alpha_values,
                    base_seed=batch_seed,
                    sample_offset=sample_offset,
                    initial_state=initial_state,
                    return_continuation_state=return_continuation_state,
                    continuation_context=continuation_context,
                )
                return W_batch.transpose(0, 1), X_batch.transpose(0, 1)

            dynamic_batches = self._compute_internal_spreading_alpha_batches(
                alpha_values,
                sample_count=S,
            )
            if not dynamic_batches and A:
                dynamic_batches = [(0, A, alpha_max)]
            self._contract_execution_metadata = self._build_spreading_execution_metadata(
                alpha_values,
                dynamic_batches,
            )

            W_result = torch.zeros(A, S, N1, M, device=self.device)
            X_result = torch.zeros(A, S, M, N2, device=self.device)
            for batch_idx, (alpha_start, alpha_end, _) in enumerate(dynamic_batches):
                batch_alpha_indices = list(range(alpha_start, alpha_end))
                batch_alpha_list = alpha_values[alpha_start:alpha_end]
                if sample_callback is not None:
                    sample_callback(batch_idx, len(dynamic_batches), batch_alpha_list)
                batch_seed = (
                    seed
                    if self._uses_partition_invariant_seed_policy()
                    else self._spreading_batch_seed(seed, batch_idx)
                )
                W_batch, X_batch = self.train_full_parallel(
                    spreading_data,
                    batch_alpha_indices=batch_alpha_indices,
                    verbose=False,
                    step_callback=step_callback,
                    max_steps=max_steps,
                    batch_alpha_values=batch_alpha_list,
                    base_seed=batch_seed,
                    sample_offset=sample_offset,
                )
                W_result[alpha_start:alpha_end] = W_batch.transpose(0, 1)
                X_result[alpha_start:alpha_end] = X_batch.transpose(0, 1)
                del W_batch, X_batch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            return W_result, X_result

        dynamic_batches = self._compute_internal_spreading_alpha_batches(
            alpha_values,
            sample_count=S,
        )
        if not dynamic_batches and A:
            dynamic_batches = [(0, A, alpha_max)]
        num_batches = len(dynamic_batches)
        self._contract_execution_metadata = self._build_spreading_execution_metadata(
            alpha_values,
            dynamic_batches,
        )

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
            # Legacy offsets seed by batch_idx; partition_invariant keeps base_seed stable.
            batch_seed = self._spreading_batch_seed(seed, batch_idx)
            batch_spreading_data = self.create_spreading_data(
                W_teacher, X_teacher, batch_alpha_list, S, batch_seed, sample_offset=sample_offset
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
                base_seed=batch_seed,
                sample_offset=sample_offset,
                initial_state=initial_state if len(batch_alpha_list) == 1 else None,
                return_continuation_state=return_continuation_state and len(batch_alpha_list) == 1,
                continuation_context=continuation_context,
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
        spreading_data: Optional[SpreadingDataParallel] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        step_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ):
        """Return spreading BiGAMP output as a native matrix AlgorithmResult."""
        import time

        call_kwargs = self._filter_train_batch_kwargs({
            "W_teacher": W_teacher,
            "X_teacher": X_teacher,
            "Y_teacher": Y_teacher,
            "masks": masks,
            "spreading_data": spreading_data,
            "alpha_values": alpha_values,
            "seed": seed,
            "progress_callback": progress_callback,
            "step_callback": step_callback,
            **kwargs,
        })
        train_start = time.perf_counter()
        W_students, X_students = self.train_batch_alphas(**call_kwargs)
        train_seconds = time.perf_counter() - train_start
        result = self.coerce_native_matrix_result(
            algorithm_key=algorithm_key,
            W_students=W_students,
            X_students=X_students,
            result_source="native_bigamp_spreading_algorithm_result",
            metadata={
                "batch_metric_path": (
                    "gpu_resident_batch_metric_payload"
                    if spreading_data is not None else "runner_metric_fallback"
                ),
            },
        )
        if spreading_data is not None:
            from ...metrics.spreading import BatchMetricPayload, compute_all_metrics_spreading_parallel

            W_for_metrics = W_students.transpose(0, 1) if W_students.shape[0] == len(alpha_values) else W_students
            X_for_metrics = X_students.transpose(0, 1) if X_students.shape[0] == len(alpha_values) else X_students
            metric_start = time.perf_counter()
            metric_tensors = compute_all_metrics_spreading_parallel(
                W_for_metrics,
                X_for_metrics,
                spreading_data,
                target_alpha_idx=(
                    int((kwargs.get("continuation_context") or {}).get("alpha_index"))
                    if isinstance(kwargs.get("continuation_context"), dict)
                    and "alpha_index" in kwargs.get("continuation_context")
                    and len(alpha_values) == 1
                    and len(spreading_data.alpha_values) > 1
                    else None
                ),
                edge_chunk_size=min(max(int(getattr(self, "chunk_size", 0) or 8192), 1024), 8192),
                sample_chunk_size=16,
            )
            if torch.cuda.is_available() and W_for_metrics.is_cuda:
                torch.cuda.synchronize(W_for_metrics.device)
            metric_seconds = time.perf_counter() - metric_start
            alpha_tensor = metric_tensors.get("alpha_values", spreading_data.alpha_values)
            payload = BatchMetricPayload(
                alpha_values=alpha_tensor,
                metrics={key: value for key, value in metric_tensors.items() if key != "alpha_values"},
                metadata={
                    "path": "bigamp_spreading_gpu_batch_metrics",
                    "materialization": "single_batch_cpu_transfer",
                },
            )
            materialize_start = time.perf_counter()
            result.metrics_by_alpha = payload.to_metrics_by_alpha()
            materialize_seconds = time.perf_counter() - materialize_start
            result.diagnostics["batch_metric_payload"] = dict(payload.metadata or {})
            result.diagnostics["batch_timing_seconds"] = {
                "train_gpu_wall": train_seconds,
                "metric_gpu_wall": metric_seconds,
                "metric_materialize_cpu": materialize_seconds,
            }
        else:
            result.diagnostics["batch_timing_seconds"] = {
                "train_gpu_wall": train_seconds,
                "metric_gpu_wall": 0.0,
                "metric_materialize_cpu": 0.0,
            }
        if kwargs.get("return_continuation_state", False):
            result.continuation_state = getattr(self, "_last_continuation_state", None)
        return result

    def train_single_alpha(
        self,
        alpha: float,
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
        base_seed: Optional[int] = None,
        sample_offset: int = 0,
        initial_state: Optional[AlgorithmStateView] = None,
        return_continuation_state: bool = False,
        continuation_context: Optional[Dict[str, Any]] = None,
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
        effective_alpha_values = self._batch_alpha_values(
            spreading_data, batch_alpha_indices, batch_alpha_values
        )
        if self._uses_partition_invariant_seed_policy() and base_seed is None:
            raise RuntimeError("bigamp_spreading partition_invariant initialization requires base_seed.")

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
        init_mean = self._student_init_mean()

        if init_mode == 'teacher':
            # Teacher-Assisted Initialization for General Graph
            # Construct unified teacher vector: V = [W; X^T]
            W_teacher = spreading_data.W_teacher  # (N1, M)
            X_teacher_T = spreading_data.X_teacher.T  # (M, N2) -> (N2, M)
            V_teacher = torch.cat([W_teacher, X_teacher_T], dim=0)  # (N_total, M)

            if self._uses_partition_invariant_seed_policy():
                V_flat = self._initialize_near_teacher_partitioned(
                    alpha_values=effective_alpha_values,
                    teacher_tensor=V_teacher,
                    init_overlap=init_overlap,
                    sample_count=S,
                    sample_offset=sample_offset,
                    seed=base_seed,
                    role="V_student",
                )
            else:
                V_flat = self._initialize_near_teacher(
                    (B, S * N_total, M),
                    V_teacher,
                    init_overlap,
                    S,
                    self.device,
                    self.storage_dtype,
                    noise_scale=self._norm.student_init_std,
                )
            if verbose:
                print(f"  [Init] Teacher-Assisted General (m={init_overlap})")
        else:
            # Random Initialization (Cold Start - default)
            if self._uses_partition_invariant_seed_policy():
                V_flat = self._randn_partitioned_spreading_flat(
                    alpha_values=effective_alpha_values,
                    sample_count=S,
                    sample_offset=sample_offset,
                    node_count=N_total,
                    latent_dim=M,
                    seed=base_seed,
                    role="V_student",
                    scale=self._norm.student_init_std,
                    mean=init_mean,
                )
            else:
                V_flat = torch.randn(B, S * N_total, M, device=self.device, dtype=self.storage_dtype) * self._norm.student_init_std + init_mean

        V_var_flat = torch.ones(B, S * N_total, M, device=self.device, dtype=self.storage_dtype) * self._norm.prior_variance

        prev_s = None
        prev_svar = None
        if initial_state is not None:
            V_flat, V_var_flat, prev_s, prev_svar = self._coerce_spreading_general_continuation_state(
                initial_state=initial_state,
                B=B,
                S=S,
                N1=N1,
                N2=N2,
                M=M,
                SC=SC,
                alpha_mask_exp=alpha_mask_exp,
            )
        is_ising = (self.f_distribution == 'ising')

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
                            backend='inductor',
                            fullgraph=False,
                            options={'triton.cudagraphs': False},
                        )
                    except Exception:
                        BiGAMPSpreading._compiled_step_general = bigamp_step_disjoint_union_flat_general
                step_fn = BiGAMPSpreading._compiled_step_general

        # Loop
        for step in range(steps):
            if use_chunked:
                # Chunked version: pass chunk_size and use_compile
                V_flat, V_var_flat, s_values, svar_values, _, _ = step_fn(
                    V_flat, V_var_flat, Y_flat, F_flat, i_offset, j_offset,
                    alpha_mask_exp, S, N_total, self.damping, self.noise_var,
                    self._norm.prior_precision_base,
                    self._norm.prior_variance,
                    is_ising, prev_s, prev_svar, self.chunk_size, self.use_compile
                )
            else:
                # Legacy version
                V_flat, V_var_flat, s_values, svar_values, _, _ = step_fn(
                    V_flat, V_var_flat, Y_flat, F_flat, i_offset, j_offset,
                    alpha_mask_exp, S, N_total, self.damping, self.noise_var,
                    self._norm.prior_precision_base,
                    self._norm.prior_variance,
                    is_ising, prev_s, prev_svar
                )

            # Onsager
            if self.onsager_correction:
                prev_s = s_values
                prev_svar = svar_values
            else:
                prev_s = None
                prev_svar = None

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

        if return_continuation_state:
            self._last_continuation_state = self._build_spreading_general_continuation_state(
                W_hat,
                X_hat,
                V_var_flat,
                prev_s,
                prev_svar,
                route="general_onsager" if self.onsager_correction else "general_no_onsager",
                S=S,
                B=B,
                N1=N1,
                N2=N2,
                M=M,
                steps=steps,
                alpha_values=effective_alpha_values,
                W_teacher=spreading_data.W_teacher,
                X_teacher=spreading_data.X_teacher,
            )
        else:
            self._last_continuation_state = None
        
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
    try:
        W_teacher, X_teacher = teacher.create(
            m.N1,
            m.N2,
            m.M,
            device,
            seed,
            normalization_profile=getattr(config.algorithm_params, "normalization_profile", "paper_sparse_sampling"),
        )
    except TypeError:
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
            verbose=verbose,
            base_seed=seed,
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
