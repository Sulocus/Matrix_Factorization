"""
Tensor Memory Estimation and Dynamic Batching.

Provides:
1. Memory probing for accurate VRAM estimation
2. Dynamic sample batching based on available GPU memory
3. OOM recovery with automatic batch size reduction

Design: Uses probing (trial run) instead of complex formulas.
The probe creates a small test run to measure actual memory usage.
"""

import math
import torch
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def probe_tensor_memory(
    dims: Tuple[int, ...],
    M: int,
    S: int,
    alpha: float,
    device: torch.device,
    use_bf16: bool = True,
) -> float:
    """
    Probe actual VRAM usage for tensor spreading with given parameters.
    
    Creates a small test run to measure real memory consumption,
    which is more accurate than formula estimation for complex computations.
    
    Args:
        dims: Tensor dimensions (N_1, N_2, ..., N_n)
        M: Latent dimension
        S: Number of samples
        alpha: Alpha value (affects hyperedge count)
        device: torch device
        use_bf16: Whether to use BF16 precision
        
    Returns:
        Estimated VRAM in GB for the full run
    """
    if not torch.cuda.is_available():
        return 0.0
    
    # Clean up before probing
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    n = len(dims)
    storage_dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    # Compute hyperedge count C
    dof = sum(dims) * M
    C = max(1, int(alpha * dof))
    
    try:
        # Create tensors matching actual training shapes
        # factors: (S, N_d, M) for each dimension
        factors = [
            torch.randn(S, N_d, M, device=device, dtype=storage_dtype)
            for N_d in dims
        ]
        factor_vars = [
            torch.ones(S, N_d, M, device=device, dtype=storage_dtype)
            for N_d in dims
        ]
        
        # F: (S, C, M), Y: (S, C)
        F = torch.randn(S, C, M, device=device, dtype=storage_dtype)
        Y = torch.randn(S, C, device=device, dtype=storage_dtype)
        
        # Indices (shared across samples)
        indices = [
            torch.randint(0, dims[d], (C,), device=device)
            for d in range(n)
        ]
        
        # Simulate one BiG-AMP step to capture compute memory
        # Gather
        gathered = torch.stack([
            factors[d][:, indices[d].long()]
            for d in range(n)
        ])  # (n, S, C, M)
        
        # Product
        product = gathered.prod(dim=0)  # (S, C, M)
        
        # Z_hat
        Z_hat = (F * product).sum(dim=2)  # (S, C)
        
        # Variance (simplified)
        V = torch.ones(S, C, device=device, dtype=storage_dtype)
        
        # Residual
        s_values = (Y - Z_hat) / (V + 1e-6)
        
        # Backward contribution (simplified)
        contrib = F * product * s_values.unsqueeze(2)
        
        # Scatter (simplified)
        for d in range(n):
            r_d = torch.zeros(S, dims[d], M, device=device, dtype=storage_dtype)
            idx_exp = indices[d].unsqueeze(0).unsqueeze(2).expand(S, -1, M)
            r_d.scatter_add_(1, idx_exp, contrib)
        
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_gb = peak_bytes / (1024**3)
        
        # Add safety margin (1.2x) for torch.compile overhead
        return peak_gb * 1.2
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"Probe OOM with S={S}")
            return float('inf')
        raise
    finally:
        # Cleanup
        del factors, factor_vars, F, Y, indices
        if 'gathered' in dir():
            del gathered, product, Z_hat, V, s_values, contrib
        torch.cuda.empty_cache()


def probe_tensor_super_memory(
    dims: Tuple[int, ...],
    M: int,
    S: int,
    alpha_max: float,
    device: torch.device,
    use_bf16: bool = True,
) -> float:
    """
    Probe actual VRAM usage for Phase 3 TensorSuperGraph format.
    
    Measures memory for A=1 alpha. Result can be linearly extrapolated
    to estimate memory for any A, since memory scales linearly with A.
    
    Args:
        dims: Tensor dimensions (N_1, N_2, ..., N_n)
        M: Latent dimension
        S: Number of samples  
        alpha_max: Maximum alpha value (determines C_max)
        device: torch device
        use_bf16: Whether to use BF16 precision
        
    Returns:
        Estimated VRAM in GB for A=1 alpha. Multiply by A for total.
    """
    if not torch.cuda.is_available():
        return 0.0
    
    # Clean up before probing
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    n = len(dims)
    storage_dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    # Compute C_max based on alpha_max
    dof = sum(dims) * M
    C_max = max(1, int(alpha_max * dof))
    SC = S * C_max
    A = 1  # Probe with single alpha
    
    try:
        # Create Phase 3 format tensors: (A, S*N_d, M)
        factors = [
            torch.randn(A, S * N_d, M, device=device, dtype=storage_dtype)
            for N_d in dims
        ]
        factor_vars = [
            torch.ones(A, S * N_d, M, device=device, dtype=storage_dtype)
            for N_d in dims
        ]
        
        # F_flat: (S*C_max, M), Y_flat: (S*C_max)
        F_flat = torch.randn(SC, M, device=device, dtype=storage_dtype)
        Y_flat = torch.randn(SC, device=device, dtype=storage_dtype)
        
        # Indices: (S*C_max,) for each dimension
        indices_flat = [
            torch.randint(0, dims[d], (SC,), device=device)
            for d in range(n)
        ]
        
        # Simulate forward_pass_tensor_super
        gathered_list = []
        for d in range(n):
            N_d = dims[d]
            sample_offsets = torch.arange(S, device=device).unsqueeze(1) * N_d
            sample_offsets = sample_offsets.expand(S, C_max).reshape(-1)
            offset_indices = indices_flat[d] + sample_offsets
            gathered = factors[d][:, offset_indices.long()]
            gathered_list.append(gathered)
        
        gathered = torch.stack(gathered_list)  # (n, A, S*C_max, M)
        product = gathered.prod(dim=0)  # (A, S*C_max, M)
        Z_hat = (F_flat.unsqueeze(0) * product).sum(dim=2)  # (A, S*C_max)
        
        # Simulate residual computation
        s_values = (Y_flat.unsqueeze(0) - Z_hat) / 1.0
        
        # Simulate backward contribution
        r_contrib = F_flat.unsqueeze(0) * product * s_values.unsqueeze(2)
        
        # Simulate scatter
        for d in range(n):
            N_d = dims[d]
            SN_d = S * N_d
            r_d = torch.zeros(A, SN_d, M, device=device, dtype=storage_dtype)
            sample_offsets = torch.arange(S, device=device).unsqueeze(1) * N_d
            sample_offsets = sample_offsets.expand(S, C_max).reshape(-1)
            offset_indices = indices_flat[d] + sample_offsets
            idx_exp = offset_indices.unsqueeze(0).unsqueeze(2).expand(A, -1, M)
            r_d.scatter_add_(1, idx_exp, r_contrib)
        
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_gb = peak_bytes / (1024**3)
        
        # Safety margin (1.3x) for torch.compile overhead and memory spikes
        return peak_gb * 1.3
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"Probe OOM for Phase 3 with alpha_max={alpha_max}")
            return float('inf')
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()


def estimate_max_samples(
    dims: Tuple[int, ...],
    M: int,
    alpha: float,
    device: torch.device,
    target_memory_ratio: float = 0.8,
    use_bf16: bool = True,
) -> int:
    """
    Estimate maximum number of samples that fit in GPU memory.
    
    Uses binary search with probing to find optimal S.
    
    Args:
        dims: Tensor dimensions
        M: Latent dimension
        alpha: Alpha value
        device: torch device
        target_memory_ratio: Target GPU memory usage ratio (0.8 = 80%)
        use_bf16: Whether to use BF16 precision
        
    Returns:
        Maximum S that fits in memory
    """
    if not torch.cuda.is_available():
        return 1
    
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    target_gb = total_gb * target_memory_ratio
    
    # Quick probe with S=1 to get baseline
    base_gb = probe_tensor_memory(dims, M, 1, alpha, device, use_bf16)
    
    if base_gb >= target_gb:
        logger.warning(f"Even S=1 uses {base_gb:.2f}GB > target {target_gb:.2f}GB")
        return 1
    
    # Binary search for max S
    lo, hi = 1, 64  # Max 64 samples
    best_s = 1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        estimated_gb = probe_tensor_memory(dims, M, mid, alpha, device, use_bf16)
        
        if estimated_gb <= target_gb:
            best_s = mid
            lo = mid + 1
        else:
            hi = mid - 1
    
    logger.info(f"Max samples for {dims}, M={M}, alpha={alpha}: S={best_s} (uses ~{probe_tensor_memory(dims, M, best_s, alpha, device, use_bf16):.2f}GB)")
    return best_s


class TensorBatchCoordinator:
    """
    Coordinates batch execution for Tensor Spreading Parallel.
    
    Automatically determines batch size based on available GPU memory
    and handles OOM recovery with reduced batch sizes.
    """
    
    def __init__(
        self,
        dims: Tuple[int, ...],
        M: int,
        device: torch.device,
        target_memory_ratio: float = 0.8,
        use_bf16: bool = True,
    ):
        """
        Initialize coordinator.
        
        Args:
            dims: Tensor dimensions
            M: Latent dimension
            device: torch device
            target_memory_ratio: Target GPU usage (0.8 = 80%)
            use_bf16: Whether to use BF16
        """
        self.dims = dims
        self.M = M
        self.device = device
        self.target_memory_ratio = target_memory_ratio
        self.use_bf16 = use_bf16
        self._last_max_s = None
    
    def get_max_samples(self, alpha: float) -> int:
        """Get maximum samples for given alpha."""
        if self._last_max_s is not None:
            return self._last_max_s
        
        self._last_max_s = estimate_max_samples(
            self.dims, self.M, alpha, self.device,
            self.target_memory_ratio, self.use_bf16
        )
        return self._last_max_s
    
    def create_batches(self, total_samples: int, alpha: float) -> list:
        """
        Create sample batches that fit in memory.
        
        Args:
            total_samples: Total S samples to process
            alpha: Alpha value
            
        Returns:
            List of sample counts per batch
        """
        max_s = self.get_max_samples(alpha)
        
        if total_samples <= max_s:
            return [total_samples]
        
        # Split into batches
        batches = []
        remaining = total_samples
        while remaining > 0:
            batch_size = min(remaining, max_s)
            batches.append(batch_size)
            remaining -= batch_size
        
        logger.info(f"Split {total_samples} samples into {len(batches)} batches: {batches}")
        return batches
    
    def handle_oom(self) -> None:
        """Reduce batch size after OOM."""
        if self._last_max_s is not None and self._last_max_s > 1:
            self._last_max_s = max(1, self._last_max_s // 2)
            logger.warning(f"OOM occurred, reducing max_samples to {self._last_max_s}")
    
    def cleanup(self) -> None:
        """Clean up GPU memory."""
        torch.cuda.empty_cache()
