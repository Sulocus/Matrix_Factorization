"""
GPU Memory Management for SMF.

.. deprecated:: 2025.12
    This module is deprecated. Please use the new smart parallel module instead:
    
    from smf.core.parallel import (
        MemoryEstimator,
        ParallelCoordinator,
        AllocationPresets,
        EstimationParams,
    )
    
    See smf/core/parallel/ for the new unified memory management system.

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
    algorithm_key: Optional[str] = None,
    alpha_max: float = 4.0,
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

    # Dynamic memory reservation (90% usable, no hard cap)
    # Reserve 5% or 2GB (whichever is larger) for system overhead
    reserved_gb = max(2.0, available_gb * 0.05)
    effective_available = available_gb - reserved_gb
    
    # MAX_GPU_MEMORY_GB deprecated, using full available
    # effective_available = MAX_GPU_MEMORY_GB - RESERVED_MEMORY_GB

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

    # Use Spreading-specific estimator for Spreading algorithms
    if algorithm_key == "bigamp_spreading_parallel":
        spreading_strategy = get_spreading_memory_strategy(
            N1=N1, N2=N2, M=M, S=S,
            num_alphas=num_alphas,
            alpha_max=alpha_max,
            available_gb=available_gb,
            verbose=verbose,
        )
        # Convert to MemoryStrategy
        alphas_per_batch = spreading_strategy['alphas_per_batch']
        mode = MemoryMode.PARALLEL if alphas_per_batch >= 1 else MemoryMode.OPTIMIZED
        return MemoryStrategy(
            mode=mode,
            max_parallel_alphas=alphas_per_batch,
            use_fp16_storage=False,
            effective_available_gb=spreading_strategy['effective_available_gb'],
            per_alpha_memory_gb=spreading_strategy['estimated_memory_gb'] / alphas_per_batch if alphas_per_batch > 0 else 0,
        )

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
    # Dynamic: reserve 5% or 2GB
    reserved_gb = max(2.0, available_gb * 0.05)
    available = available_gb - reserved_gb

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


# ============================================================================
# Spreading Algorithm Memory Management (Disjoint Union Parallelization)
# ============================================================================

def estimate_memory_spreading_parallel(
    N1: int,
    N2: int,
    M: int,
    S: int,
    B: int,
    alpha_max: float = 4.0,
    dtype_bytes: int = 4,
    use_compile: bool = True,
) -> float:
    """
    BiG-AMP 显存估计公式 (校准版 2024-12-11 v2)
    
    经过实测校准: N=600, M=150, S=2, B=29, alpha_max=2.8
    估算应接近 26 GB (实测值)
    
    Memory components:
    1. CUDA overhead + PyTorch context: ~1.0 GB
    2. torch.compile overhead (Triton kernels, compilation cache): ~2.0 GB
    3. Persistent tensors: W_flat, X_flat, F, Y, indices  
    4. Clone overhead: 额外 4 × (B, S*N, M) 用于clone副本
    5. Gather 中间张量: 3.5 × (B, SC, M) 峰值 (W_sel, X_sel, F广播等)
    6. 其他中间计算

    Args:
        N1, N2: Matrix dimensions
        M: Hidden dimension (rank)
        S: Number of samples (all parallel)
        B: Number of alphas per batch
        alpha_max: Maximum alpha value in this batch
        dtype_bytes: Bytes per element (4 for float32, 2 for bfloat16)
        use_compile: Whether torch.compile is enabled

    Returns:
        Estimated memory in GB
    """
    C_max = int(alpha_max * M * N1)
    if C_max == 0:
        C_max = 1
    SC = S * C_max

    # === 1. 固定开销 (PyTorch Context + CUDA Kernels) ===
    cuda_overhead = 1.0 * (1024**3)

    # === 2. torch.compile 开销 (Triton 编译缓存、内核元数据) ===
    # 实测表明 torch.compile 即使在 mode=default 也需要额外 ~2 GB
    compile_overhead = 2.0 * (1024**3) if use_compile else 0

    # === 3. 持久输入/输出张量 (Persistent) ===
    storage_dtype_bytes = 2  # BF16 is now default
    # W_flat, X_flat, W_var_flat, X_var_flat: 4 × (B, S*N, M)
    params = 4 * B * S * (N1 * M + N2 * M) * storage_dtype_bytes
    
    # Indices: i_offset, j_offset (SC) int64
    indices = 2 * SC * 8
    
    # F Matrix (Rademacher 使用 int8)
    f_int8 = SC * M * 1
    
    # Y_super (SC) float32
    y_super = SC * 4

    persistent = params + indices + f_int8 + y_super

    # === 4. Clone 操作开销 (CUDA Graphs 修复) ===
    # Clone Strategy: 每次迭代后 clone 4 个主状态张量
    clone_overhead = 4 * B * S * (N1 * M + N2 * M) * storage_dtype_bytes

    # === 5. 中间计算张量峰值 (v3.1 校准) ===
    # 峰值发生在scatter阶段，同时存在:
    # - W_sel, X_sel, W_var_sel, X_var_sel: 4 × (B, SC, M) BF16
    # - r_W_contrib, tau_W_contrib, r_X_contrib, tau_X_contrib: 4 × (B, SC, M) BF16
    # 注意: 这些张量会被逐步释放，不会全部同时存在
    # 实测校准: 系数 5.5 在 400x100 和 200x50 之间取得较好平衡
    peak_compute_tensors = 5.5 * B * SC * M * 3  # 5.5 峰值张量, ~3 bytes混合精度

    # === 6. 其他中间计算 ===
    # Z_hat, V, s_values, residual 等: (B, SC) float32
    intermediate = B * SC * 4 * 6  # 6 intermediate (B, SC) tensors

    # === 7. 输出缓冲区 (r_W, r_X, tau_W, tau_X scatter 目标) ===
    output_buffers = 4 * B * S * (N1 * M + N2 * M) * storage_dtype_bytes

    total_bytes = (cuda_overhead + compile_overhead + persistent + clone_overhead 
                   + peak_compute_tensors + intermediate + output_buffers)
    
    # 添加 10% 安全裕度
    return total_bytes * 1.10 / (1024**3)






def calculate_spreading_batches(
    N1: int,
    N2: int,
    M: int,
    S: int,
    num_alphas: int,
    max_memory_gb: float = 24.0,
    alpha_max: float = 4.0,
) -> int:
    """
    Calculate optimal number of alphas per batch for Spreading algorithm.
    
    Note: This returns a FIXED batch size based on alpha_max.
    For dynamic batching, use compute_dynamic_batches() instead.

    Args:
        N1: Number of rows
        N2: Number of columns
        M: Hidden dimension
        S: Number of samples (all parallel)
        num_alphas: Total number of alpha values
        max_memory_gb: Maximum GPU memory to use
        alpha_max: Maximum alpha value (for C_max estimation)

    Returns:
        B: Number of alphas per batch (fixed)
    """
    # Binary search for maximum B
    low, high = 1, num_alphas

    while low < high:
        mid = (low + high + 1) // 2
        mem = estimate_memory_spreading_parallel(N1, N2, M, S, mid, alpha_max)
        if mem <= max_memory_gb:
            low = mid
        else:
            high = mid - 1

    return max(1, low)


def compute_dynamic_batches(
    N1: int,
    N2: int,
    M: int,
    S: int,
    alpha_values: list,
    max_memory_gb: float = 26.0,
) -> list:
    """
    Compute dynamic batch assignments based on per-batch memory requirements.
    
    Groups alphas greedily: start a new batch when adding another alpha
    would exceed memory limit. Uses the MAX alpha in each batch to estimate
    memory (since C_max is determined by the largest alpha).
    
    Args:
        N1, N2, M, S: Matrix dimensions
        alpha_values: List of alpha values (should be sorted ascending)
        max_memory_gb: Maximum GPU memory per batch
    
    Returns:
        List of tuples: [(start_idx, end_idx, batch_alpha_max), ...]
        Each tuple defines alphas[start_idx:end_idx] as one batch.
    """
    if not alpha_values:
        return []
    
    # Ensure sorted
    alpha_values = sorted(alpha_values)
    
    batches = []
    start_idx = 0
    n = len(alpha_values)
    
    while start_idx < n:
        # Try to fit as many alphas as possible starting from start_idx
        end_idx = start_idx + 1
        
        while end_idx <= n:
            # The max alpha in this potential batch
            batch_alpha_max = alpha_values[end_idx - 1]
            batch_size = end_idx - start_idx
            
            # Skip alpha=0 (no edges, negligible memory)
            if batch_alpha_max <= 0:
                end_idx += 1
                continue
            
            # Estimate memory for this batch
            mem = estimate_memory_spreading_parallel(
                N1, N2, M, S, batch_size, batch_alpha_max
            )
            
            if mem > max_memory_gb:
                # This alpha would exceed memory, stop here
                break
            
            end_idx += 1
        
        # end_idx is now one past the last alpha that fits
        actual_end = end_idx - 1
        if actual_end <= start_idx:
            actual_end = start_idx + 1  # At minimum, include one alpha
        
        batch_alpha_max = alpha_values[actual_end - 1] if actual_end > 0 else 0.1
        batches.append((start_idx, actual_end, batch_alpha_max))
        start_idx = actual_end
    
    return batches


def get_spreading_memory_strategy(
    N1: int,
    N2: int,
    M: int,
    S: int,
    num_alphas: int,
    alpha_max: float = 4.0,
    available_gb: Optional[float] = None,
    verbose: bool = True,
    alpha_values: Optional[list] = None,
) -> dict:
    """
    Get complete memory strategy for Spreading algorithm.

    If alpha_values is provided, uses DYNAMIC batching (different batch sizes
    for different alpha ranges). Otherwise falls back to fixed batching.

    Args:
        N1, N2, M, S: Matrix dimensions
        num_alphas: Total alpha values
        alpha_max: Maximum alpha (for C_max)
        available_gb: Available GPU memory (auto-detect if None)
        verbose: Print info
        alpha_values: Optional list of actual alpha values for dynamic batching

    Returns:
        dict with:
        - alphas_per_batch: Number of alphas per batch (for fixed mode)
        - num_batches: Total batches needed
        - estimated_memory_gb: Estimated peak memory per batch
        - samples_parallel: Always S (all samples parallel)
        - dynamic_batches: List of (start_idx, end_idx, alpha_max) if alpha_values provided
    """
    if available_gb is None:
        available_gb = get_available_gpu_memory()
        if available_gb == 0:
            available_gb = 8.0

    # Dynamic memory reservation (no hard 32GB cap)
    # Use 90% of available memory directly
    reserved_gb = max(2.0, available_gb * 0.05)
    effective_gb = (available_gb - reserved_gb) * 0.95  # Slightly safer for Spreading
    
    # effective_gb = (min(available_gb, 32.0) - 3.0) * 0.90

    # Dynamic batching if alpha_values provided
    if alpha_values is not None and len(alpha_values) > 0:
        dynamic_batches = compute_dynamic_batches(
            N1, N2, M, S, alpha_values, effective_gb
        )
        num_batches = len(dynamic_batches)
        
        # Estimate memory for the largest batch (worst case)
        max_batch_size = max(b[1] - b[0] for b in dynamic_batches) if dynamic_batches else 1
        max_batch_alpha = max(b[2] for b in dynamic_batches) if dynamic_batches else alpha_max
        estimated_mem = estimate_memory_spreading_parallel(
            N1, N2, M, S, max_batch_size, max_batch_alpha
        )
        
        if verbose:
            print("\n[Spreading Memory Strategy - Dynamic Batching]")
            print(f"  Matrix: {N1}×{N2}, M={M}")
            print(f"  Samples: {S} (all parallel via Disjoint Union)")
            print(f"  Alphas: {num_alphas} total, {num_batches} dynamic batches")
            print(f"  Batch breakdown:")
            for i, (start, end, batch_max) in enumerate(dynamic_batches):
                batch_size = end - start
                alpha_range = f"{alpha_values[start]:.2f}-{alpha_values[end-1]:.2f}"
                mem = estimate_memory_spreading_parallel(N1, N2, M, S, batch_size, batch_max)
                print(f"    Batch {i+1}: α {alpha_range}, {batch_size} alphas, ~{mem:.1f}GB")
            print(f"  Memory limit: {effective_gb:.1f} GB")
        
        return {
            'alphas_per_batch': max_batch_size,  # For compatibility
            'num_batches': num_batches,
            'estimated_memory_gb': estimated_mem,
            'samples_parallel': S,
            'effective_available_gb': effective_gb,
            'dynamic_batches': dynamic_batches,  # NEW: variable batch assignments
        }
    
    # Fallback: Fixed batch size based on alpha_max
    alphas_per_batch = calculate_spreading_batches(
        N1, N2, M, S, num_alphas, effective_gb, alpha_max
    )

    num_batches = (num_alphas + alphas_per_batch - 1) // alphas_per_batch
    estimated_mem = estimate_memory_spreading_parallel(
        N1, N2, M, S, alphas_per_batch, alpha_max
    )

    if verbose:
        print("\n[Spreading Memory Strategy]")
        print(f"  Matrix: {N1}×{N2}, M={M}")
        print(f"  Samples: {S} (all parallel via Disjoint Union)")
        print(f"  Alphas: {num_alphas} total, {alphas_per_batch}/batch, {num_batches} batches")
        print(f"  Memory: {estimated_mem:.2f} GB per batch (limit: {effective_gb:.1f} GB)")

    return {
        'alphas_per_batch': alphas_per_batch,
        'num_batches': num_batches,
        'estimated_memory_gb': estimated_mem,
        'samples_parallel': S,
        'effective_available_gb': effective_gb,
    }
