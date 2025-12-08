"""
GPU Memory Management for SMF.

Provides intelligent memory mode selection based on matrix size and available GPU memory.
Migrated from Wang/bigamp/train.py with enhancements for modular integration.

Memory Modes:
- parallel: Batch multiple alphas together (fastest, for N <= 10000)
- optimized: Sequential alpha processing, on-demand mask generation (for N <= 20000)
- extreme: FP16 storage + sequential processing (for N <= 25000)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import torch


class MemoryMode(Enum):
    """GPU memory management mode."""
    PARALLEL = "parallel"      # Batch alphas, fastest for small matrices
    OPTIMIZED = "optimized"    # Sequential alpha, on-demand masks
    EXTREME = "extreme"        # FP16 storage + sequential


@dataclass
class MemoryStrategy:
    """Memory management strategy result."""
    mode: MemoryMode
    max_parallel_alphas: int
    use_fp16_storage: bool
    effective_available_gb: float
    per_alpha_memory_gb: float

    def __str__(self) -> str:
        return (f"MemoryStrategy(mode={self.mode.value}, "
                f"max_parallel={self.max_parallel_alphas}, "
                f"fp16={self.use_fp16_storage})")


def estimate_memory_per_alpha(
    N1: int, N2: int, M: int, S: int, dtype_bytes: int = 4,
    use_compile: bool = True
) -> float:
    """
    Estimate GPU memory needed per alpha value.

    BiG-AMP creates intermediate tensors of shape (batch, S, N1, N2):
    - W update: z_hat, p_var, V, residual, s = 5 tensors
    - X update: z_hat2, p_var2, V2, residual2, s2 = 5 tensors

    With torch.compile: kernel fusion allows W-phase tensors to be freed
    before X-phase, so peak is ~5 tensors.
    Without compile: ~8 tensors (some overlap due to Python GC).

    PyTorch memory overhead: ~50% for fragmentation, allocator pools,
    and temporary matmul intermediates.

    Args:
        N1: Number of rows
        N2: Number of columns
        M: Rank (hidden dimension)
        S: Samples per alpha
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)
        use_compile: Whether torch.compile is enabled (reduces memory)

    Returns:
        Estimated memory in GB
    """
    # Student parameters: w_hat, x_hat, w_var, x_var
    student_params = 4 * (S * N1 * M + S * M * N2)

    # Intermediate tensors: (S, N1, N2) per alpha
    # With torch.compile: peak ~5 tensors (W-phase freed before X-phase)
    # Without compile: ~8 tensors (some caching overlap)
    num_intermediate = 5 if use_compile else 8
    intermediate = num_intermediate * S * N1 * N2

    # Base memory
    base_elements = student_params + intermediate
    base_gb = base_elements * dtype_bytes / (1024**3)

    # PyTorch overhead: ~50% for fragmentation, allocator pools, temp tensors
    pytorch_overhead = 1.5

    return base_gb * pytorch_overhead


def get_available_gpu_memory() -> float:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0

    try:
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        available = total - reserved
        return available / (1024**3)
    except Exception:
        return 0.0


def select_memory_mode(
    N1: int,
    N2: int,
    M: int,
    S: int,
    num_alphas: int,
    available_gb: Optional[float] = None,
    mode_override: Optional[str] = None,
    verbose: bool = True,
) -> MemoryStrategy:
    """
    Select optimal memory mode based on matrix size and available GPU memory.

    Args:
        N1: Number of rows
        N2: Number of columns
        M: Rank (hidden dimension)
        S: Samples per alpha
        num_alphas: Total number of alpha values to process
        available_gb: Available GPU memory in GB (auto-detect if None)
        mode_override: Force specific mode ('parallel', 'optimized', 'extreme')
        verbose: Print selection info

    Returns:
        MemoryStrategy with selected mode and parameters
    """
    # Get available memory
    if available_gb is None:
        available_gb = get_available_gpu_memory()
        if available_gb == 0:
            available_gb = 8.0  # Default assumption for CPU

    # Cap at 32GB and reserve 3GB for system
    MAX_GPU_MEMORY_GB = min(available_gb, 32.0)
    RESERVED_MEMORY_GB = 3.0
    effective_available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

    # Memory estimates
    per_alpha_mem = estimate_memory_per_alpha(N1, N2, M, S)
    teacher_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)
    single_mask_mem = N1 * N2 * 4 / (1024**3)

    if verbose:
        print("\n[Memory Mode Selection]")
        print(f"  Matrix: {N1}×{N2}, M={M}, S={S}")
        print(f"  Available: {effective_available:.1f} GB")
        print(f"  Per-alpha training: {per_alpha_mem:.2f} GB")
        print(f"  Single mask: {single_mask_mem:.3f} GB")

    # Handle mode override
    if mode_override:
        mode = MemoryMode(mode_override)
        max_parallel = 1 if mode != MemoryMode.PARALLEL else calculate_smart_parallelism(
            N1, N2, M, S, num_alphas, effective_available
        )
        use_fp16 = mode == MemoryMode.EXTREME
        if verbose:
            print(f"  Mode override: {mode.value}")
        return MemoryStrategy(
            mode=mode,
            max_parallel_alphas=max_parallel,
            use_fp16_storage=use_fp16,
            effective_available_gb=effective_available,
            per_alpha_memory_gb=per_alpha_mem,
        )

    # Calculate how many alphas can fit in parallel mode
    # Use 90% of available memory for training (conservative but not overly so)
    usable_mem = effective_available * 0.90 - teacher_mem
    mem_per_batch_alpha = per_alpha_mem + single_mask_mem
    max_batch = max(1, int(usable_mem / mem_per_batch_alpha)) if mem_per_batch_alpha > 0 else num_alphas

    # Select mode
    if max_batch >= 2:
        mode = MemoryMode.PARALLEL
        max_parallel = min(max_batch, num_alphas)
        use_fp16 = False
        if verbose:
            print(f"  Selected: parallel (batch={max_parallel})")
    elif per_alpha_mem + single_mask_mem < effective_available * 0.8:
        mode = MemoryMode.OPTIMIZED
        max_parallel = 1
        use_fp16 = False
        if verbose:
            print("  Selected: optimized (sequential, on-demand masks)")
    else:
        mode = MemoryMode.EXTREME
        max_parallel = 1
        use_fp16 = True
        if verbose:
            print("  Selected: extreme (FP16 + sequential)")

    return MemoryStrategy(
        mode=mode,
        max_parallel_alphas=max_parallel,
        use_fp16_storage=use_fp16,
        effective_available_gb=effective_available,
        per_alpha_memory_gb=per_alpha_mem,
    )


def calculate_smart_parallelism(
    N1: int,
    N2: int,
    M: int,
    S: int,
    num_alphas: int,
    available_gb: Optional[float] = None,
) -> int:
    """
    Calculate optimal number of parallel alphas based on memory.

    Args:
        N1: Number of rows
        N2: Number of columns
        M: Rank (hidden dimension)
        S: Samples per alpha
        num_alphas: Total number of alpha values
        available_gb: Available GPU memory in GB

    Returns:
        Maximum number of alphas that can be processed in parallel
    """
    if available_gb is None:
        available_gb = get_available_gpu_memory()
        if available_gb == 0:
            return 1

    # Cap and reserve
    MAX_GPU_MEMORY_GB = min(available_gb, 32.0)
    RESERVED_MEMORY_GB = 3.0
    available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

    per_alpha_mem = estimate_memory_per_alpha(N1, N2, M, S)
    teacher_mem = (N1 * M + M * N2 + N1 * N2) * 4 / (1024**3)
    single_mask_mem = N1 * N2 * 4 / (1024**3)

    # Each batch alpha needs: training memory + mask storage
    mem_per_batch_alpha = per_alpha_mem + single_mask_mem
    usable_mem = available * 0.85 - teacher_mem

    if mem_per_batch_alpha <= 0:
        return num_alphas

    max_parallel = max(1, min(int(usable_mem / mem_per_batch_alpha), num_alphas))
    return max_parallel
