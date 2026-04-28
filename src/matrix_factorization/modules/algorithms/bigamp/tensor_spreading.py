"""
BiG-AMP for N-dimensional tensor CP decomposition.

This module provides the main algorithm class for tensor spreading,
implementing BiG-AMP message passing for n-dimensional tensor 
factorization with random spreading coefficients.

Key features:
- Supports arbitrary tensor order n (n=2 reduces to matrix factorization)
- Simplified architecture: single-alpha, single-sample training
- Onsager correction optional (default OFF)
- Ising or Gaussian spreading coefficients
- Fully integrated with runner via AlgorithmBase interface
"""

import math
import torch
from typing import Any, List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from matrix_factorization.core.distributions import F_DISTRIBUTION_ISING, normalize_f_distribution
from .tensor_data import TensorHypergraph, TensorSpreadingData
from .tensor_contract import (
    build_tensor_execution_metadata,
    build_tensor_result_metadata,
    resolve_tensor_dims,
)
from .tensor_step import tensor_step, forward_pass_tensor
from .tensor_hypergraph import generate_tensor_hypergraph, generate_tensor_observations

from matrix_factorization.modules.registry import register_algorithm
from matrix_factorization.modules.algorithms.base import AlgorithmBase
from matrix_factorization.core.experiment.config import resolve_normalization_profile
from matrix_factorization.modules.metrics.tensor_metrics import compute_tensor_factor_projection_overlaps


@dataclass
class TensorSpreadingConfig:
    """Configuration for tensor spreading algorithm."""
    tensor_order: int = 3
    dims: Tuple[int, ...] = (50, 50, 50)
    M: int = 20
    max_steps: int = 200
    damping: float = 0.5
    noise_var: float = 1e-6
    f_distribution: str = 'ising'
    onsager_correction: bool = False  # Disabled by default for stability


@register_algorithm(
    key="bigamp_tensor",
    name="BiG-AMP Tensor Spreading",
    description="N-dimensional tensor CP decomposition with random spreading",
    default_params={'damping': 0.5, 'noise_var': 1e-6},
)
class BiGAMPTensorSpreading(AlgorithmBase):
    """
    BiG-AMP for N-dimensional tensor CP decomposition.
    
    This algorithm extends the matrix BiG-AMP spreading to n-dimensional
    tensors. Fully integrated with runner via AlgorithmBase interface.
    
    Supports two initialization modes:
    1. From runner: __init__(config, device) - config has matrix, training, etc.
    2. Direct use: Use class methods with explicit parameters
    """
    
    def __init__(self, config=None, device: torch.device = None, **kwargs):
        """
        Initialize from runner config or direct parameters.
        
        Args:
            config: Either a full config object (from runner) or tensor_order int (legacy)
            device: torch device
            **kwargs: Legacy kwargs API (tensor_order, dims, M, max_steps, damping, etc.)
        """
        # === Legacy kwargs API: BiGAMPTensorSpreading(tensor_order=3, dims=(...), ...) ===
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
                'fast' if kwargs.get('use_bf16', False) else 'safe',
            )
            if self.precision_profile == 'fast' and kwargs.get('use_bf16', True) is False:
                self.precision_profile = 'safe'
            self.precision_fallback_policy = kwargs.get(
                'precision_fallback_policy',
                'allow',
            )
            if self.precision_fallback_policy == 'allow' and kwargs.get('dtype_fallback_policy', 'allow') == 'error':
                self.precision_fallback_policy = 'error'
            self.normalization_profile = kwargs.get('normalization_profile', 'paper_sparse_sampling')
        
        # === From runner: BiGAMPTensorSpreading(config, device) ===
        elif hasattr(config, 'matrix'):
            self.config = config
            self.device = device
            
            # Get tensor_order from spreading config
            self.order = getattr(config.spreading, 'tensor_order', 3) if hasattr(config, 'spreading') else 3
            
            # N维张量：目前假设所有维度相等 (除了 matrix.M)
            if config.matrix.N1 != config.matrix.N2:
                # 如果 N1 != N2，目前 tensor 实现可能会有维度不匹配问题
                # (因为 factors[1] 是 X.T (N2, M)，但 dims 假设全是 N1)
                import logging
                logging.getLogger(__name__).warning(
                    f"BiGAMPTensorSpreading currently assumes isotropic dimensions (N1==N2). "
                    f"Got N1={config.matrix.N1}, N2={config.matrix.N2}. "
                    f"Will use N1 for all tensor dimensions."
                )
            
            self.dims = resolve_tensor_dims(config.matrix, self.order)
            self.M = config.matrix.M
            self.max_steps = config.training.max_steps
            self.S = config.training.samples_per_alpha
            
            # Algorithm params
            self.damping = config.algorithm_params.damping
            self.noise_var = config.algorithm_params.noise_var
            self.normalization_profile = getattr(config.algorithm_params, "normalization_profile", "paper_sparse_sampling")
            
            # Spreading params
            if hasattr(config, 'spreading') and config.spreading:
                self.f_distribution = normalize_f_distribution(config.spreading.f_distribution)
                # Respect user configuration for Onsager correction
                self.onsager_correction = config.spreading.onsager_correction
            else:
                self.f_distribution = F_DISTRIBUTION_ISING
                self.onsager_correction = False  # Default: OFF for consistency with SpreadingConfig
            algorithm_params = getattr(config, "algorithm_params", None)
            self.precision_profile = getattr(
                algorithm_params,
                'precision_profile',
                'fast' if getattr(algorithm_params, 'use_bf16', False) else 'safe',
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
                
        elif isinstance(config, int):
            # === Legacy direct call (tensor_order as first arg) ===
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
            self.precision_profile = 'safe'
            self.precision_fallback_policy = 'allow'
            self.normalization_profile = 'paper_sparse_sampling'
        else:
            raise ValueError(f"config must be a config object or int, got {type(config)}")
        
        # Validate dims matches order
        if len(self.dims) != self.order:
            raise ValueError(f"dims length {len(self.dims)} must match tensor_order {self.order}")
        if self.precision_profile not in {'safe', 'fast', 'aggressive'}:
            raise ValueError(
                "algorithm_params.precision_profile must be 'safe', 'fast', or 'aggressive', "
                f"got {self.precision_profile!r}"
            )
        if self.precision_fallback_policy not in {'allow', 'error'}:
            raise ValueError(
                "algorithm_params.precision_fallback_policy must be 'allow' or 'error', "
                f"got {self.precision_fallback_policy!r}"
            )
        self.requested_use_bf16 = self.precision_profile in {'fast', 'aggressive'}
        bf16_supported = bool(
            (self.device is not None)
            and getattr(self.device, "type", "cpu") == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        self.use_bf16 = self.requested_use_bf16 and bf16_supported
        if self.requested_use_bf16 and not self.use_bf16 and self.precision_fallback_policy == 'error':
            raise RuntimeError(
                "BF16 precision profile was requested but is unavailable and "
                "algorithm_params.precision_fallback_policy='error'"
            )
        self.storage_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        self.dtype_status = (
            "bf16_requested_and_effective"
            if self.use_bf16
            else ("fallback_to_float32_bf16_unavailable" if self.requested_use_bf16 else "bf16_disabled_by_config")
        )
        self._norm = resolve_normalization_profile(self.normalization_profile, self.M)

    # =========================================================================
    # AlgorithmBase Interface Implementation
    # =========================================================================
    
    def train_single_alpha(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AlgorithmBase interface: Train for single alpha.
        
        For tensor mode, W_teacher and X_teacher are used to construct
        n-dimensional teacher factors.
        """
        # Create teacher factors from W, X
        teacher_factors = self._create_teacher_factors(W_teacher, X_teacher)
        
        # Train single sample
        result = self._train_single_internal(teacher_factors, alpha, seed, self.device)
        
        # Return dummy W, X tensors (metrics computed in train_batch_alphas)
        W_students = torch.zeros(self.S, self.dims[0], self.M, device=self.device)
        X_students = torch.zeros(self.S, self.M, self.dims[1] if self.order >= 2 else self.dims[0], device=self.device)
        
        # Store Q_Y in instance for metrics retrieval
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
        
        This is the main entry point called by runner.
        """
        A = len(alpha_values)
        
        # Create teacher factors
        teacher_factors = self._create_teacher_factors(W_teacher, X_teacher)
        
        # Storage for results
        W_all = torch.zeros(A, self.S, self.dims[0], self.M, device=self.device)
        X_all = torch.zeros(A, self.S, self.M, self.dims[1] if self.order >= 2 else self.dims[0], device=self.device)
        
        # Also store metrics for runner
        self._batch_metrics = {}
        
        total_work = A * self.S * self.max_steps
        completed_steps = 0
        
        def wrapped_step_callback(step, total, metrics=None):
            # Pass batch-relative step (step, max_steps) instead of global cumulative
            # This allows UI to correctly display per-batch progress
            if step_callback:
                step_callback(step, total, metrics)
        
        for alpha_idx, alpha in enumerate(alpha_values):
            # Progress is tracked by runner, no print needed
            
            alpha_q_y_list = []
            alpha_nmse_y_list = []
            alpha_q_y_proj_abs_list = []
            alpha_qn_list = []
            alpha_qn_modes = [[] for _ in range(self.order)]
            
            # TODO: Consider reusing hypergraph for samples (resample_mask=False case)
            
            for s in range(self.S):
                sample_seed = seed + s * 1000 + int(alpha * 100)
                
                result = self._train_single_internal(
                    teacher_factors, alpha, sample_seed, self.device,
                    step_callback=wrapped_step_callback
                )
                
                alpha_q_y_list.append(result['Q_Y'])
                if 'NMSE_Y' in result:
                    alpha_nmse_y_list.append(result['NMSE_Y'])
                if 'Q_Y_PROJ_ABS' in result:
                    alpha_q_y_proj_abs_list.append(result['Q_Y_PROJ_ABS'])
                if 'Q_N' in result:
                    alpha_qn_list.append(result['Q_N'])
                for d in range(self.order):
                    key = f'Q_N_mode{d}'
                    if key in result:
                        alpha_qn_modes[d].append(result[key])
                completed_steps += self.max_steps
            
            # Store aggregated metrics per alpha
            mean_qy = sum(alpha_q_y_list) / len(alpha_q_y_list)
            self._batch_metrics[alpha] = {
                'Q_Y_mean': mean_qy,
                'Q_Y_observed_mean': mean_qy,
                'Q_Y_std': (sum((q - mean_qy)**2 for q in alpha_q_y_list) / max(1, len(alpha_q_y_list)-1)) ** 0.5 if len(alpha_q_y_list) > 1 else 0.0,
                'Q_Y_observed_std': (sum((q - mean_qy)**2 for q in alpha_q_y_list) / max(1, len(alpha_q_y_list)-1)) ** 0.5 if len(alpha_q_y_list) > 1 else 0.0,
            }
            if alpha_nmse_y_list:
                mean_nmse = sum(alpha_nmse_y_list) / len(alpha_nmse_y_list)
                nmse_std = (sum((q - mean_nmse)**2 for q in alpha_nmse_y_list) / max(1, len(alpha_nmse_y_list)-1)) ** 0.5 if len(alpha_nmse_y_list) > 1 else 0.0
                self._batch_metrics[alpha]['NMSE_Y_mean'] = mean_nmse
                self._batch_metrics[alpha]['NMSE_Y_observed_mean'] = mean_nmse
                self._batch_metrics[alpha]['NMSE_Y_std'] = nmse_std
                self._batch_metrics[alpha]['NMSE_Y_observed_std'] = nmse_std
            if alpha_q_y_proj_abs_list:
                mean_proj = sum(alpha_q_y_proj_abs_list) / len(alpha_q_y_proj_abs_list)
                proj_std = (sum((q - mean_proj)**2 for q in alpha_q_y_proj_abs_list) / max(1, len(alpha_q_y_proj_abs_list)-1)) ** 0.5 if len(alpha_q_y_proj_abs_list) > 1 else 0.0
                self._batch_metrics[alpha]['Q_Y_PROJ_ABS_mean'] = mean_proj
                self._batch_metrics[alpha]['Q_Y_observed_PROJ_ABS_mean'] = mean_proj
                self._batch_metrics[alpha]['Q_Y_PROJ_ABS_std'] = proj_std
                self._batch_metrics[alpha]['Q_Y_observed_PROJ_ABS_std'] = proj_std
            if alpha_qn_list:
                mean_qn = sum(alpha_qn_list) / len(alpha_qn_list)
                self._batch_metrics[alpha]['Q_N_mean'] = mean_qn
                self._batch_metrics[alpha]['Q_N_std'] = (sum((q - mean_qn)**2 for q in alpha_qn_list) / max(1, len(alpha_qn_list)-1)) ** 0.5 if len(alpha_qn_list) > 1 else 0.0
            for d, values in enumerate(alpha_qn_modes):
                if values:
                    mean_mode = sum(values) / len(values)
                    self._batch_metrics[alpha][f'Q_N_mode{d}_mean'] = mean_mode
                    self._batch_metrics[alpha][f'Q_N_mode{d}_std'] = (sum((q - mean_mode)**2 for q in values) / max(1, len(values)-1)) ** 0.5 if len(values) > 1 else 0.0

            # Metrics stored for runner retrieval
        
        return W_all, X_all

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
        """Return tensor metrics through the formal AlgorithmResult contract.

        The serial tensor path currently does not expose matrix-shaped W/X
        factors that are meaningful to the matrix result schema. Its formal
        result is therefore metrics-only; train_batch_alphas is still the
        numerical implementation.
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
        self.train_batch_alphas(**call_kwargs)
        return self._metrics_only_algorithm_result(algorithm_key)

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
        return AlgorithmResult.from_metrics_only(
            metrics_by_alpha=metrics_by_alpha,
            artifacts=artifacts,
            metadata=build_tensor_result_metadata(
                algorithm_key=algorithm_key,
                result_contract=spec.result_contract,
                result_source="tensor_algorithm_train_batch_result",
                dims=self.dims,
                tensor_order=self.order,
                graph_kind="tensor_hypergraph",
                alpha_values_original=sorted(metrics_by_alpha),
                alpha_values_execution_order=sorted(metrics_by_alpha),
                batching_source="serial_alpha_sample_loop",
                extra={
                    "tensor_execution": build_tensor_execution_metadata(
                        path="serial_tensor_hypergraph",
                        device=getattr(self, "device", None),
                        requested_use_bf16=bool(getattr(self, "requested_use_bf16", False)),
                        effective_use_bf16=bool(getattr(self, "use_bf16", False)),
                        dtype_fallback_policy=getattr(self, "precision_fallback_policy", "allow"),
                        dtype_status=getattr(self, "dtype_status", "float32_serial_path"),
                        storage_dtype=getattr(self, "storage_dtype", torch.float32),
                        precision_profile=getattr(self, "precision_profile", "safe"),
                        precision_fallback_policy=getattr(self, "precision_fallback_policy", "allow"),
                        normalization_profile=normalization_profile,
                        normalization_schema_version=getattr(norm, "schema_version", 5),
                        normalization_convention=getattr(norm, "convention_label", "paper_sparse_sampling_var1_latent_unit_prior"),
                        requested_use_compile=False,
                        compile_fallback_policy="not_applicable",
                        effective_use_compile=False,
                        compiled_step_available=False,
                        compiled_super_step_available=False,
                        compile_status="not_applicable_serial_tensor",
                        compile_attempts=[],
                        notes="Serial tensor metadata only; this path has no compile/dtype planner.",
                    ),
                    "internal_alpha_batch_plan": {
                        "planner": "serial_alpha_sample_loop",
                        "alpha_values_input": sorted(float(alpha) for alpha in metrics_by_alpha),
                        "alpha_values_execution_order": sorted(float(alpha) for alpha in metrics_by_alpha),
                        "alpha_batches": [[float(alpha)] for alpha in sorted(metrics_by_alpha)],
                        "num_batches": len(metrics_by_alpha),
                        "sort_policy": "legacy_sorted_metrics_keys",
                        "probe_enabled": False,
                        "seed_partition_sensitive": True,
                        "metadata_only": True,
                    },
                },
            ),
        )
    
    def supports_batch_training(self) -> bool:
        """Tensor spreading does support batch training across alphas."""
        return True
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _create_teacher_factors(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Create n teacher factors from W and X matrices.
        
        For n=2: [W, X.T]
        For n>=3: [W, X.T, ...additional random factors]
        """
        factors = []
        
        # Factor 0: W (N1, M)
        factors.append(W_teacher.to(self.device))
        
        if self.order >= 2:
            # Factor 1: X.T (N2, M) - note X is (M, N2), so we transpose
            factors.append(X_teacher.T.to(self.device))
        
        # For n > 2, create additional factors
        for d in range(2, self.order):
            N_d = self.dims[d]
            # Initialize from W's statistics
            scale = self._norm.latent_std
            torch.manual_seed(42 + d)
            factor_d = torch.randn(N_d, self.M, device=self.device) * scale
            factors.append(factor_d)
        
        return factors
    
    def _train_single_internal(
        self,
        teacher_factors: List[torch.Tensor],
        alpha: float,
        seed: int,
        device: torch.device,
        verbose: bool = False,
        step_callback: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Internal training for one alpha value.
        """
        n = self.order
        M = teacher_factors[0].shape[1]
        
        # Move teacher to device
        teacher_factors = [t.to(device) for t in teacher_factors]
        
        # Generate hypergraph
        hg = generate_tensor_hypergraph(self.dims, alpha, M, seed, device)
        
        # Generate F and Y
        F, Y = generate_tensor_observations(
            teacher_factors, hg, seed + 1000, device, self.f_distribution
        )
        if getattr(self, "precision_profile", "safe") == "aggressive" and getattr(self, "use_bf16", False):
            if torch.is_floating_point(F):
                F = F.to(self.storage_dtype)
            if torch.is_floating_point(Y):
                Y = Y.to(self.storage_dtype)

        # Initialize student in the selected normalization profile.
        init_scale = self._norm.student_init_std

        factors = [
            torch.randn(t.shape, device=device, dtype=self.storage_dtype) * init_scale
            for t in teacher_factors
        ]
        factor_vars = [
            torch.ones(t.shape, device=device, dtype=self.storage_dtype) * self._norm.prior_variance
            for t in teacher_factors
        ]
        
        prev_s = None
        prev_svar = None
        is_ising = (self.f_distribution == 'ising')
        
        # Precompute y_var for final metrics (fallback logic)
        y_var_scalar = 1.0 # default
        
        # BiG-AMP iterations
        for step in range(self.max_steps):
            # Noise Annealing
            # Decay from high noise to target noise_var over first 20% steps
            anneal_steps = int(0.2 * self.max_steps)
            current_noise_var = self.noise_var
            
            if step < anneal_steps:
                # Log-linear interpolation
                start_log = math.log(1.0) # Start with high variance (1.0)
                end_log = math.log(max(self.noise_var, 1e-4)) # Don't go too low too fast
                progress = step / anneal_steps
                current_noise_var = math.exp(start_log + (end_log - start_log) * progress)

            factors, factor_vars, next_prev_s, next_prev_svar = tensor_step(
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
            
            # Throttle callback to balance UI responsiveness vs CPU overhead
            # GPU monitor has 2s cache, so every 20 steps (10 updates per 200 steps) is sufficient
            if step_callback and ((step + 1) % 20 == 0 or step == self.max_steps - 1):
                try:
                    step_callback(step + 1, self.max_steps)
                except Exception as e:
                    pass  # Silently ignore callback errors
            
            # Print progress only (Standard algorithm behavior)
            if verbose and (step + 1) % 50 == 0:
                with torch.no_grad():
                    Y_pred = forward_pass_tensor(factors, F, hg.indices)
                    mse = ((Y - Y_pred) ** 2).mean().item()
                print(f"  Step {step + 1}/{self.max_steps}: MSE = {mse:.6f}")
        
        # Compute final FIT/NMSE metrics plus legacy projection diagnostics.
        Y_student = forward_pass_tensor(factors, F, hg.indices)
        norm_teacher_sq = (Y.flatten() ** 2).sum()
        if float(norm_teacher_sq.abs().item()) < 1e-12:
            Q_Y = 0.0
            NMSE_Y = 1.0
            Q_Y_PROJ_ABS = 0.0
        else:
            diff = Y_student.flatten() - Y.flatten()
            NMSE_Y = float((diff * diff).sum() / (norm_teacher_sq + 1e-12))
            Q_Y = 1.0 - NMSE_Y
            Q_Y_PROJ_ABS = float((Y_student.flatten() * Y.flatten()).sum().abs() / (norm_teacher_sq + 1e-12))
        qn_modes = compute_tensor_factor_projection_overlaps(teacher_factors, factors)
        Q_N = sum(qn_modes) / len(qn_modes)
        
        result = {
            'Q_Y': Q_Y,
            'NMSE_Y': NMSE_Y,
            'Q_Y_PROJ_ABS': Q_Y_PROJ_ABS,
            'Q_N': Q_N,
            'alpha': alpha,
            'C': hg.C,
        }
        for d, value in enumerate(qn_modes):
            result[f'Q_N_mode{d}'] = value
        return result
    
    # =========================================================================
    # Legacy Interface (for backward compatibility)
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
        """
        Legacy train method for direct usage.
        """
        results = []
        
        for alpha_idx, alpha in enumerate(alpha_values):
            alpha_results = []
            
            for s in range(S):
                seed = base_seed + s * 1000 + int(alpha * 100)
                
                if verbose:
                    print(f"Alpha {alpha:.2f}, Sample {s + 1}/{S}")
                
                result = self._train_single_internal(
                    teacher_factors, alpha, seed, device, verbose=False
                )
                result['sample'] = s
                result['alpha_idx'] = alpha_idx
                alpha_results.append(result)
            
            q_y_mean = sum(r['Q_Y'] for r in alpha_results) / S
            if verbose:
                print(f"  Alpha {alpha:.2f}: Q_Y = {q_y_mean:.4f}")
            
            results.extend(alpha_results)
        
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
