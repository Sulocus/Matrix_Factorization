"""
Memory Estimator for General Graph Mode (BiG-AMP Spreading).

This module provides memory estimation specifically for General mode,
which has different memory patterns than Bipartite mode:
- Unified V vector (N1 + N2) instead of separate W and X
- Three edge types: W-W, W-X, X-X
- Different scatter/gather patterns
"""
import math

# Import shared components from main estimator
from .memory_estimator import (
    MemoryEstimator, EstimationParams, MemoryBreakdown, MemoryComponent, TensorSpec, DType
)


def estimate_general_edge_count(alpha: float, N1: int, N2: int, M: int) -> int:
    """
    Estimate total edge count for General mode.
    
    This must match the logic in create_supergraph_general:
    C = alpha * M * N1
    
    We cap this at the total number of possible edges in the graph.
    """
    # Calculate expected edges based on density alpha
    # Use ceil to be slightly conservative for memory estimation
    estimated_edges = int(math.ceil(alpha * M * N1))
    
    # Cap at total possible edges in undirected graph (no self-loops)
    N_total = N1 + N2
    total_possible_edges = N_total * (N_total - 1) // 2
    
    return min(estimated_edges, total_possible_edges)


def get_general_spreading_breakdown(params: EstimationParams) -> MemoryBreakdown:
    """
    Get detailed memory breakdown for General mode BigAMP Spreading.
    
    Key differences from Bipartite:
    1. Unified V_flat: (B, S*N_total, M) instead of separate W_flat and X_flat
    2. Edge indices: a_offset, b_offset instead of i_offset, j_offset
    3. More edges due to W-W and X-X connections
    """
    N1, N2, M, S = params.N1, params.N2, params.M, params.S
    B = params.batch_size  # Alpha batch size
    alpha_max = params.alpha_max
    
    # Derived dimensions
    N_total = N1 + N2
    C_max = estimate_general_edge_count(alpha_max, N1, N2, M)
    SC = S * C_max  # Total edges across all samples
    SN = S * N_total  # Flattened node dimension
    
    # Dtype selection
    storage_dtype = DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32
    f_dtype = DType.INT8 if params.f_distribution == 'rademacher' else DType.FLOAT32
    
    breakdown = MemoryBreakdown(
        algorithm_key="bigamp_spreading_general",
        safety_margin=1.1  # 10% safety margin
    )
    
    # =========================================================================
    # COMPONENT 1: Student Parameters (Unified V)
    # V_flat, V_var_flat: 2 × (B, S*N_total, M)
    # =========================================================================
    student = MemoryComponent(name="Student Parameters (V)")
    student.add(TensorSpec(
        name="V_flat + V_var_flat",
        shape=(B, SN, M),
        shape_formula="2 × (B, S*N_total, M)",
        dtype=storage_dtype,
        count=2,
        notes="Unified node vector containing both W and X"
    ))
    breakdown.add_component(student)
    
    # =========================================================================
    # COMPONENT 2: SuperGraph Data
    # F_flat, Y_flat, a_offset, b_offset, alpha_mask
    # =========================================================================
    supergraph = MemoryComponent(name="SuperGraph Data")
    supergraph.add(TensorSpec(
        name="F_flat",
        shape=(SC, M),
        shape_formula="(S*C_max, M)",
        dtype=f_dtype,
    ))
    supergraph.add(TensorSpec(
        name="Y_flat",
        shape=(SC,),
        shape_formula="(S*C_max,)",
        dtype=DType.FLOAT32,
    ))
    supergraph.add(TensorSpec(
        name="a_offset + b_offset",
        shape=(SC,),
        shape_formula="(S*C_max,)",
        dtype=DType.INT64,
        count=2,
    ))
    supergraph.add(TensorSpec(
        name="alpha_mask_exp",
        shape=(B, SC),
        shape_formula="(B, S*C_max)",
        dtype=DType.BOOL,
    ))
    breakdown.add_component(supergraph)
    
    # =========================================================================
    # COMPONENT 3: Gather Tensors
    # V_a, V_b, V_a_var, V_b_var: 4 × (B, SC, M)
    # =========================================================================
    gather = MemoryComponent(name="Gather Tensors")
    gather.add(TensorSpec(
        name="V_a + V_b + V_a_var + V_b_var",
        shape=(B, SC, M),
        shape_formula="4 × (B, S*C_max, M)",
        dtype=storage_dtype,
        count=4,
        notes="Gathered node values for edge computation"
    ))
    breakdown.add_component(gather)
    
    # =========================================================================
    # COMPONENT 4: Forward Compute
    # Z_hat, V_val, s_values, denom + temp tensors
    # =========================================================================
    forward = MemoryComponent(name="Forward Compute")
    forward.add(TensorSpec(
        name="Z_hat + V_val + s_values + denom",
        shape=(B, SC),
        shape_formula="(B, S*C_max)",
        dtype=storage_dtype,
        count=4,
    ))
    # Temporary (B, SC, M) tensors during computation
    forward.add(TensorSpec(
        name="compute_temps",
        shape=(B, SC, M),
        shape_formula="(B, S*C_max, M)",
        dtype=storage_dtype,
        count=2,  # F*V_a*V_b, V_a_var*V_b^2, etc.
    ))
    breakdown.add_component(forward)
    
    # =========================================================================
    # COMPONENT 5: Scatter Buffers
    # r_V, tau_V: 2 × (B, S*N_total, M)
    # term (contribution): (B, SC, M)
    # =========================================================================
    scatter = MemoryComponent(name="Scatter Buffers")
    scatter.add(TensorSpec(
        name="r_V + tau_V",
        shape=(B, SN, M),
        shape_formula="2 × (B, S*N_total, M)",
        dtype=storage_dtype,
        count=2,
    ))
    scatter.add(TensorSpec(
        name="term (contribution)",
        shape=(B, SC, M),
        shape_formula="(B, S*C_max, M)",
        dtype=storage_dtype,
        count=1,
    ))
    breakdown.add_component(scatter)
    
    return breakdown


def estimate_general_spreading(params: EstimationParams) -> float:
    """
    Estimate memory for General mode BigAMP Spreading.
    
    Returns:
        Estimated memory in GB
    """
    breakdown = get_general_spreading_breakdown(params)
    return breakdown.total_gb


def compute_max_alpha_batch_general(
    N1: int, N2: int, M: int, S: int, 
    num_alphas: int, alpha_max: float,
    available_gb: float,
    use_bf16: bool = True,
    f_distribution: str = 'rademacher'
) -> int:
    """
    Compute maximum alpha batch size for General mode.
    
    Args:
        N1, N2, M, S: Problem dimensions
        num_alphas: Total number of alpha values to process
        alpha_max: Maximum alpha value (for edge count estimation)
        available_gb: Available GPU memory in GB
        use_bf16: Whether using BF16 storage
        f_distribution: 'rademacher' or 'gaussian'
    
    Returns:
        Maximum batch size (number of alphas to process in parallel)
    """
    # Binary search for maximum batch size
    low, high = 1, num_alphas
    best = 1
    
    while low <= high:
        mid = (low + high) // 2
        
        params = EstimationParams(
            N1=N1, N2=N2, M=M, S=S,
            alpha_values=[alpha_max] * mid,  # Worst case: all alphas at max
            algorithm_key="bigamp_spreading_general",
            use_bf16=use_bf16,
            use_compile=False,
            f_distribution=f_distribution,
        )
        
        estimated = estimate_general_spreading(params)
        
        if estimated <= available_gb * 0.9:  # 90% threshold
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return min(best, num_alphas)


# Register with main MemoryEstimator
@MemoryEstimator.register("bigamp_spreading_general")
def _estimate_general_wrapper(params: EstimationParams) -> float:
    return estimate_general_spreading(params)
