"""
AGD (Alternating Gradient Descent) with Random Spreading - Parallel Implementation.

This module implements AGD algorithm for the random spreading model
with Super-Graph parallelization across alpha values, mirroring BiGAMP Spreading structure.

Key features:
1. Configurable F distribution: gaussian or rademacher
2. Super-Graph strategy: parallel processing of all alphas
3. Simpler gradient descent (no variance estimation, no Onsager)

Physical model:
    Y_ij = (1/√M) Σ_μ F_ij,μ W_iμ X_μj

Algorithm:
    For each step:
        Z_hat = forward_pass(W, X, F)
        residual = Y - Z_hat
        grad_W = scatter_add(residual * F * X)
        grad_X = scatter_add(residual * F * W)
        W = W - lr * grad_W
        X = X - lr * grad_X
"""

from typing import Tuple, Callable, Dict, Optional, List
from dataclasses import dataclass
import math
from pathlib import Path
import datetime
import torch

from ..registry import register_algorithm
from .base import AlgorithmBase
from ..graphs.supergraph import SuperGraphData, create_supergraph
from ..teachers.random_spreading import SpreadingDataParallel

# Import reusable functions from BiGAMP Spreading
from .bigamp_spreading import (
    generate_F_super,
    compute_Y_super,
    compute_offset_indices,
    F_GENERATORS,
)


# ============================================================================
# Global GPU Optimizations
# ============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# AGD Core Functions
# ============================================================================

def agd_forward_pass_flat(
    W_flat: torch.Tensor,      # (A, S*N1, M)
    X_flat: torch.Tensor,      # (A, S*N2, M) 
    F_flat: torch.Tensor,      # (S*C_max, M)
    i_offset: torch.Tensor,    # (S*C_max,)
    j_offset: torch.Tensor,    # (S*C_max,)
    alpha_mask_exp: torch.Tensor,  # (A, S*C_max)
) -> torch.Tensor:
    """
    Forward pass: compute Z_hat = (1/√M) Σ F * W * X at observed edges.
    
    Args:
        W_flat: (A, S*N1, M) flattened W estimates
        X_flat: (A, S*N2, M) flattened X estimates
        F_flat: (S*C_max, M) spreading coefficients
        i_offset: (S*C_max,) row indices with sample offset
        j_offset: (S*C_max,) col indices with sample offset
        alpha_mask_exp: (A, S*C_max) which edges are active
    
    Returns:
        Z_hat: (A, S*C_max) predicted values
    """
    A = W_flat.shape[0]
    M = W_flat.shape[2]
    SC = F_flat.shape[0]
    alpha_scale = 1.0 / math.sqrt(M)
    
    # Gather W and X at edge positions
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    
    # Convert F to compute dtype if needed
    F_compute = F_flat.to(W_flat.dtype)
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    
    # Z_hat = (1/√M) Σ F * W * X
    Z_hat = alpha_scale * (F_exp * W_sel * X_sel).sum(dim=2)  # (A, SC)
    
    # Apply mask
    mask_typed = alpha_mask_exp.to(Z_hat.dtype)
    Z_hat = Z_hat * mask_typed
    
    return Z_hat


def agd_step_flat(
    W_flat: torch.Tensor,      # (A, S*N1, M)
    X_flat: torch.Tensor,      # (A, S*N2, M)
    Y_flat: torch.Tensor,      # (S*C_max,)
    F_flat: torch.Tensor,      # (S*C_max, M)
    i_offset: torch.Tensor,    # (S*C_max,)
    j_offset: torch.Tensor,    # (S*C_max,)
    alpha_mask_exp: torch.Tensor,  # (A, S*C_max)
    S: int,
    N1: int,
    N2: int,
    learning_rate: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single AGD step with parallel alpha processing.
    
    Gradient descent update:
        residual = Y - Z_hat
        grad_W = -2 * (1/√M) * scatter_add(residual * F * X)
        grad_X = -2 * (1/√M) * scatter_add(residual * F * W)
        W = W - lr * grad_W
        X = X - lr * grad_X
    
    Args:
        W_flat: (A, S*N1, M) W estimates
        X_flat: (A, S*N2, M) X estimates
        Y_flat: (S*C_max,) observations
        F_flat: (S*C_max, M) spreading coefficients
        i_offset, j_offset: (S*C_max,) precomputed offset indices
        alpha_mask_exp: (A, S*C_max) which edges are active
        S: number of samples
        N1, N2: matrix dimensions
        learning_rate: gradient descent step size
    
    Returns:
        Updated (W_flat, X_flat)
    """
    A = W_flat.shape[0]
    M = W_flat.shape[2]
    SC = F_flat.shape[0]
    alpha_scale = 1.0 / math.sqrt(M)
    storage_dtype = W_flat.dtype
    
    # Pre-cast mask
    mask_typed = alpha_mask_exp.to(storage_dtype)
    
    # ===== 1. Gather W and X at edge positions =====
    W_sel = W_flat[:, i_offset, :]  # (A, SC, M)
    X_sel = X_flat[:, j_offset, :]  # (A, SC, M)
    
    # ===== 2. Forward pass =====
    F_compute = F_flat.to(storage_dtype)
    F_exp = F_compute.unsqueeze(0)  # (1, SC, M)
    
    Z_hat = alpha_scale * (F_exp * W_sel * X_sel).sum(dim=2)  # (A, SC)
    Z_hat = Z_hat * mask_typed
    
    # ===== 3. Compute residual =====
    # residual = (Y - Z_hat), then apply mask
    residual = (Y_flat.unsqueeze(0) - Z_hat) * mask_typed  # (A, SC)
    residual = residual.unsqueeze(2)  # (A, SC, 1)
    
    # ===== 4. Compute gradients via scatter_add =====
    # grad_W[i] = -2 * (1/√M) * Σ_{c: i_offset[c]=i} residual[c] * F[c] * X_sel[c]
    # We flip sign later (subtract gradient for minimization)
    
    mask_exp = mask_typed.unsqueeze(2)  # (A, SC, 1)
    
    # W gradient contribution: residual * F * X_sel
    grad_W_contrib = alpha_scale * F_exp * X_sel * residual * mask_exp  # (A, SC, M)
    grad_W_contrib = grad_W_contrib.to(storage_dtype)  # Ensure dtype matches grad_W
    grad_W = torch.zeros(A, S * N1, M, device=W_flat.device, dtype=storage_dtype)
    idx_W = i_offset.long().view(1, SC, 1).expand(A, SC, M)  # Ensure int64
    grad_W.scatter_add_(1, idx_W, grad_W_contrib)
    
    # X gradient contribution: residual * F * W_sel
    grad_X_contrib = alpha_scale * F_exp * W_sel * residual * mask_exp  # (A, SC, M)
    grad_X_contrib = grad_X_contrib.to(storage_dtype)  # Ensure dtype matches grad_X
    grad_X = torch.zeros(A, S * N2, M, device=W_flat.device, dtype=storage_dtype)
    idx_X = j_offset.long().view(1, SC, 1).expand(A, SC, M)  # Ensure int64
    grad_X.scatter_add_(1, idx_X, grad_X_contrib)
    
    # ===== 5. Update W and X =====
    # Note: we computed residual = Y - Z_hat, and gradient is -2 * residual * ...
    # For gradient descent minimizing MSE: W_new = W - lr * (-2 * grad_W) = W + 2*lr*grad_W
    W_flat_new = W_flat + 2.0 * learning_rate * grad_W
    X_flat_new = X_flat + 2.0 * learning_rate * grad_X
    
    # NaN protection
    W_flat_new = torch.nan_to_num(W_flat_new, nan=0.0)
    X_flat_new = torch.nan_to_num(X_flat_new, nan=0.0)
    
    return W_flat_new, X_flat_new


# ============================================================================
# Main Algorithm Class  
# ============================================================================

@register_algorithm(
    key="agd_spreading",
    name="AGD Spreading",
    description="Gradient descent with random spreading - stable but slower",
    default_params={
        'learning_rate': 1e-2,
    },
)
class AGDSpreading(AlgorithmBase):
    """
    AGD with random spreading, parallel across alpha values.
    
    Configurable options:
    - f_distribution: 'gaussian' or 'rademacher' - via config.spreading.f_distribution
    
    Differences from BiGAMP Spreading:
    - Uses simple gradient descent instead of message passing
    - No variance estimation (W_var, X_var)
    - No Onsager correction
    - Slower convergence (~20x more steps)
    
    Usage:
        config = Config(
            algorithm_key="agd_spreading",
            spreading=SpreadingConfig(f_distribution="rademacher"),
        )
    """
    
    # Class-level compiled step function cache
    _compiled_step = None
    
    @classmethod
    def clear_compile_cache(cls):
        """Clear compiled step function cache to release GPU memory."""
        if cls._compiled_step is not None:
            cls._compiled_step = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def __init__(self, config, device: torch.device):
        """
        Initialize AGD Spreading algorithm.
        
        Args:
            config: Config object with algorithm parameters
            device: Target device
        """
        super().__init__(config, device)
        
        # Spreading configuration (match BiGAMP Spreading pattern)
        spreading_cfg = getattr(config, 'spreading', None)
        if spreading_cfg is not None:
            self.f_distribution = spreading_cfg.f_distribution
            self.spreading_seed = getattr(spreading_cfg, 'seed', 12345)
        else:
            # Default values
            self.f_distribution = 'rademacher'
            self.spreading_seed = 12345
        
        # Algorithm parameters
        self.max_epochs = config.training.max_epochs
        self.learning_rate = getattr(config.algorithm, 'learning_rate', 0.01)
        
        # torch.compile settings
        self.use_compile = getattr(config.algorithm, 'use_compile', True)
        if self.use_compile and AGDSpreading._compiled_step is None:
            try:
                AGDSpreading._compiled_step = torch.compile(
                    agd_step_flat,
                    mode='reduce-overhead',
                    fullgraph=True,
                )
            except Exception as e:
                print(f"[AGDSpreading] torch.compile failed: {e}, using eager mode")
                self.use_compile = False
        
        # BF16 mixed precision
        self.use_bf16 = False
        self.storage_dtype = torch.float32
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.use_bf16 = True
            self.storage_dtype = torch.bfloat16
    
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
        
        Reuses BiGAMP Spreading's data generation functions.
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
    
    def train_full_parallel(
        self,
        spreading_data: SpreadingDataParallel,
        batch_alpha_indices: Optional[List[int]] = None,
        verbose: bool = False,
        step_callback=None,
        max_steps: Optional[int] = None,
        batch_alpha_values: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train all samples in parallel using Disjoint Union with optimized flat tensors.
        
        Args:
            spreading_data: SpreadingDataParallel with F_super, Y_super, etc.
            batch_alpha_indices: Which alphas to train (None = all)
            verbose: Print progress
            step_callback: Optional callback(step, max_steps)
            max_steps: Optional override
            batch_alpha_values: Optional list of alpha values for debug
        
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
            spreading_data.supergraph.i_idx,
            spreading_data.supergraph.j_idx,
            N1, N2
        )
        
        # ===== Pre-flatten all data (once) =====
        F_flat = spreading_data.F_super.reshape(SC, M)  # (S*C_max, M)
        Y_flat = spreading_data.Y_super.reshape(SC)     # (S*C_max,)
        
        # Expand alpha mask: (B, C_max) -> (B, S*C_max)
        alpha_mask_exp = batch_alpha_mask.unsqueeze(1).expand(B, S, C_max).reshape(B, SC)
        
        # ===== Initialize in FLAT format =====
        # Shape: (B, S*N, M)
        W_flat = torch.randn(B, S * N1, M, device=self.device, dtype=self.storage_dtype) * 0.1
        X_flat = torch.randn(B, S * N2, M, device=self.device, dtype=self.storage_dtype) * 0.1
        
        # Use provided max_steps or fall back to config
        steps = max_steps if max_steps is not None else self.max_epochs
        
        # Choose step function (compiled or eager)
        step_fn = AGDSpreading._compiled_step if self.use_compile and AGDSpreading._compiled_step is not None else agd_step_flat
        
        # ===== Training loop =====
        for step in range(steps):
            # Mark CUDA Graph step for compiled mode
            if self.use_compile and AGDSpreading._compiled_step is not None:
                torch.compiler.cudagraph_mark_step_begin()
            
            W_flat, X_flat = step_fn(
                W_flat=W_flat,
                X_flat=X_flat,
                Y_flat=Y_flat,
                F_flat=F_flat,
                i_offset=i_offset,
                j_offset=j_offset,
                alpha_mask_exp=alpha_mask_exp,
                S=S,
                N1=N1,
                N2=N2,
                learning_rate=self.learning_rate,
            )
            
            # Clone for CUDA Graph compatibility
            if self.use_compile and AGDSpreading._compiled_step is not None:
                W_flat = W_flat.clone()
                X_flat = X_flat.clone()
            
            # Memory calibration logging
            if (step + 1) == 10 and torch.cuda.is_available() and verbose:
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_gb = peak_bytes / (1024**3)
                print(f"  [Memory Calibration] Peak VRAM: {peak_gb:.2f} GB")
            
            if verbose and (step + 1) % 1000 == 0:
                print(f"  Step {step + 1}/{steps}")
            
            if step_callback:
                step_callback(step + 1, steps)
        
        # ===== Reshape for output =====
        # (B, S*N1, M) -> (B, S, N1, M) -> (S, B, N1, M)
        W_hat = W_flat.reshape(B, S, N1, M).permute(1, 0, 2, 3)
        # (B, S*N2, M) -> (B, S, N2, M) -> (S, B, M, N2)
        X_hat = X_flat.reshape(B, S, N2, M).permute(1, 0, 3, 2)
        
        return W_hat, X_hat
    
    def supports_batch_training(self) -> bool:
        """Returns True - this algorithm supports parallel alpha training."""
        return True
    
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
        Required by AlgorithmBase but not used in parallel implementation.
        
        Use train_full_parallel() instead for parallel training.
        """
        raise NotImplementedError(
            "AGDSpreading uses train_full_parallel() for parallel alpha training. "
            "This method is not intended to be called directly."
        )
    
    def train_batch_alphas(
        self,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor,
        masks: torch.Tensor,  # Not used - SuperGraph generates its own
        alpha_values: List[float],
        seed: int,
        max_steps: Optional[int] = None,
        step_callback=None,
        sample_callback=None,
        max_memory_gb: float = 24.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train for multiple alpha values using Disjoint Union parallelization.
        
        Mirrors BiGAMPSpreading.train_batch_alphas but uses AGD step function.
        
        Args:
            W_teacher: (N1, M) teacher W matrix
            X_teacher: (M, N2) teacher X matrix
            Y_teacher: (N1, N2) Y = W @ X (not used directly)
            masks: (num_alphas, N1, N2) observation masks (not used)
            alpha_values: List of alpha values to train
            seed: Random seed
            step_callback: Optional callback(step, max_steps) for step-level progress
            sample_callback: Optional callback for batch progress
            max_memory_gb: Maximum GPU memory to use
        
        Returns:
            W_students: (num_alphas, S, N1, M) trained W matrices
            X_students: (num_alphas, S, M, N2) trained X matrices
        """
        S = self.config.training.samples_per_alpha
        N1, M = W_teacher.shape
        N2 = X_teacher.shape[1]
        A = len(alpha_values)
        
        # Create spreading data for this batch
        spreading_data = self.create_spreading_data(
            W_teacher, X_teacher, alpha_values, S, seed
        )
        
        # Notify UI if callback provided
        if sample_callback:
            sample_callback(0, 1, alpha_values)
        
        # Train using parallel implementation
        W_batch, X_batch = self.train_full_parallel(
            spreading_data,
            batch_alpha_indices=None,  # All alphas
            verbose=False,
            step_callback=step_callback,
            max_steps=max_steps,
            batch_alpha_values=alpha_values,
        )
        
        # Free memory
        del spreading_data
        
        # W_batch: (S, A, N1, M), X_batch: (S, A, M, N2)
        # Transpose to (A, S, N1, M) and (A, S, M, N2) for runner compatibility
        W_result = W_batch.transpose(0, 1).contiguous()
        X_result = X_batch.transpose(0, 1).contiguous()
        
        return W_result, X_result
    
    def estimate_memory_per_alpha(self, N1: int, N2: int, M: int, S: int) -> float:
        """
        Estimate GPU memory per alpha value (in GB).
        
        AGD Spreading uses less memory than BiGAMP Spreading:
        - No variance tensors (W_var, X_var)
        - Simpler intermediate tensors
        """
        # Student parameters: W_flat, X_flat = 2 * (S*N, M)
        student_params = 2 * (S * N1 * M + S * N2 * M)
        
        # Gradient buffers: grad_W, grad_X = 2 * (S*N, M)
        gradient_buffers = 2 * (S * N1 * M + S * N2 * M)
        
        # SuperGraph data: F_flat, Y_flat, indices
        import math
        alpha_max = 4.0  # Conservative estimate
        C_max = max(1, int(math.ceil(alpha_max * N1 * N2 / M)))
        SC = S * C_max
        f_bytes = 1 if self.f_distribution == 'rademacher' else 4
        supergraph_elements = SC * M * (f_bytes / 4) + SC + 2 * SC  # F + Y + indices
        
        # Gather tensors: W_sel, X_sel = 2 * (SC, M)
        gather_tensors = 2 * SC * M
        
        total_elements = student_params + gradient_buffers + supergraph_elements + gather_tensors
        
        # BF16 reduces memory by 2x for main tensors
        bytes_per_element = 2 if self.use_bf16 else 4
        return total_elements * bytes_per_element / (1024**3)


# ============================================================================
# Test Functions
# ============================================================================

def _test_forward_pass():
    """Unit test for forward pass correctness."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small test case
    A, S, N1, N2, M = 2, 3, 10, 10, 5
    C_max = 20
    SC = S * C_max
    
    # Create test tensors
    W_flat = torch.randn(A, S * N1, M, device=device)
    X_flat = torch.randn(A, S * N2, M, device=device)
    F_flat = torch.randn(SC, M, device=device)
    i_offset = torch.randint(0, S * N1, (SC,), device=device)
    j_offset = torch.randint(0, S * N2, (SC,), device=device)
    alpha_mask_exp = torch.ones(A, SC, device=device, dtype=torch.bool)
    
    # Run forward pass
    Z_hat = agd_forward_pass_flat(
        W_flat, X_flat, F_flat, i_offset, j_offset, alpha_mask_exp
    )
    
    # Check output shape
    assert Z_hat.shape == (A, SC), f"Expected {(A, SC)}, got {Z_hat.shape}"
    
    # Check no NaN
    assert not torch.isnan(Z_hat).any(), "Forward pass produced NaN"
    
    print("✓ Forward pass test passed")


def _test_gradient_step():
    """Unit test for gradient step."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small test case
    A, S, N1, N2, M = 2, 3, 10, 10, 5
    C_max = 20
    SC = S * C_max
    
    # Create test tensors
    W_flat = torch.randn(A, S * N1, M, device=device)
    X_flat = torch.randn(A, S * N2, M, device=device)
    Y_flat = torch.randn(SC, device=device)
    F_flat = torch.randn(SC, M, device=device)
    i_offset = torch.randint(0, S * N1, (SC,), device=device)
    j_offset = torch.randint(0, S * N2, (SC,), device=device)
    alpha_mask_exp = torch.ones(A, SC, device=device, dtype=torch.bool)
    
    # Run step
    W_new, X_new = agd_step_flat(
        W_flat, X_flat, Y_flat, F_flat, i_offset, j_offset,
        alpha_mask_exp, S, N1, N2, learning_rate=0.01
    )
    
    # Check output shapes
    assert W_new.shape == W_flat.shape, f"W shape mismatch"
    assert X_new.shape == X_flat.shape, f"X shape mismatch"
    
    # Check no NaN
    assert not torch.isnan(W_new).any(), "Gradient step produced NaN in W"
    assert not torch.isnan(X_new).any(), "Gradient step produced NaN in X"
    
    # Check that values changed
    assert not torch.allclose(W_new, W_flat), "W unchanged after gradient step"
    assert not torch.allclose(X_new, X_flat), "X unchanged after gradient step"
    
    print("✓ Gradient step test passed")
