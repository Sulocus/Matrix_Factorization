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

from typing import Any, Tuple, Callable, Dict, Optional, List
import math
from pathlib import Path
import datetime
import torch

from matrix_factorization.modules.registry import register_algorithm
from matrix_factorization.modules.algorithms.base import AlgorithmBase
from matrix_factorization.modules.graphs.supergraph import SuperGraphData, create_supergraph
from matrix_factorization.modules.graphs.supergraph_general import SuperGraphDataGeneral, create_supergraph_general, EDGE_TYPE_WW, EDGE_TYPE_WX, EDGE_TYPE_XX
from matrix_factorization.modules.teachers.random_spreading import SpreadingDataParallel

from .f_gen import (
    generate_F_gaussian, generate_F_rademacher, F_GENERATORS,
    generate_F_super, compute_Y_super,
    generate_F_super_general, compute_Y_super_general
)
from .core import (
    forward_pass_parallel, compute_variance_parallel, scatter_add_parallel
)
from .step import (
    bigamp_spreading_step, bigamp_step_disjoint_union,
    compute_log_likelihood, bigamp_step_disjoint_union_flat_adaptive,
    bigamp_step_disjoint_union_flat, bigamp_step_general_chunked,
    bigamp_step_disjoint_union_flat_general, compute_offset_indices,
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

        # Clear external module caches
        clear_step_cache()

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
        self.debug_verbose = getattr(config.algorithm_params, 'debug_verbose', False)

        # Spreading configuration
        spreading_cfg = config.spreading
        if spreading_cfg is not None:
            self.f_distribution = spreading_cfg.f_distribution
            self.spreading_seed = spreading_cfg.seed
            self.onsager_correction = getattr(spreading_cfg, 'onsager_correction', False)
            self.allow_intra_connection = getattr(spreading_cfg, 'allow_intra_connection', False)
            # Default chunk_size to 0 (Unchunked) to utilize ParallelCoordinator's dynamic batching
            # instead of inefficient Python-level looping.
            self.chunk_size = getattr(spreading_cfg, 'chunk_size', 0) 
        else:
            # Default values
            self.f_distribution = 'gaussian'
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

        self._contract_execution_metadata = self._build_spreading_execution_metadata([])

    def _build_spreading_execution_metadata(
        self,
        alpha_values: List[float],
        dynamic_batches: Optional[List[Tuple[int, int, float]]] = None,
    ) -> Dict[str, Any]:
        chunk_size = int(getattr(self, "chunk_size", 0) or 0)
        return {
            "path": "bigamp_spreading_general",
            "chunk_size": chunk_size,
            "chunking_enabled": chunk_size > 0,
            "chunk_policy": "manual_config" if chunk_size > 0 else "disabled_legacy_unchunked",
            "requested_use_compile": bool(getattr(self, "use_compile", False)),
            "effective_use_bf16": bool(getattr(self, "use_bf16", False)),
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
            "seed_partition": "seed + batch_idx",
            "metadata_only": True,
            "notes": "Spreading execution metadata only; chunk_size is manual config and no auto tuning is applied.",
        }

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
        if self.debug_verbose and batch_alpha_values is not None:
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
