"""
Memory Estimator for Smart Parallel Module.

Provides unified memory estimation interface with algorithm-specific estimators
and calibration support for different GPU models.
"""
from typing import Callable, Dict, Optional, List
from pathlib import Path
import json
import logging
import torch

from .execution_modes import EstimationParams, MemoryEstimate

logger = logging.getLogger(__name__)


# =============================================================================
# Modular Memory Component System
# =============================================================================
# This modular system makes it easy to:
# 1. See which component uses how much memory
# 2. Change dtype for specific components (e.g., FP32 -> BF16 -> INT8)
# 3. Track where memory is allocated
#
# =============================================================================
# REFERENCE: PyTorch / CUDA Memory Behavior (Official Documentation)
# =============================================================================
# 
# [DATA TYPE SIZES] (from PyTorch torch.Tensor documentation)
# - torch.float32 (FP32): 4 bytes per element
# - torch.bfloat16 (BF16): 2 bytes per element (8 exponent, 7 mantissa bits)
# - torch.float16 (FP16): 2 bytes per element (5 exponent, 10 mantissa bits)
# - torch.int64: 8 bytes per element
# - torch.int32: 4 bytes per element
# - torch.int8: 1 byte per element
# - torch.bool: 1 byte per element (NOT 1 bit!)
#
# [MEMORY ALIGNMENT] (from NVIDIA CUDA Programming Guide)
# - cudaMalloc guarantees at least 256-byte alignment
# - Tensor Cores optimal: 128-bit (16-byte) alignment
# - For BF16/FP16: tensor dimensions should be multiples of 8 for Tensor Core
# - Alignment padding is handled automatically by PyTorch
# - This does NOT significantly affect total memory calculation (< 1% overhead)
#
# [CUDA CONTEXT OVERHEAD] (from PyTorch CUDA documentation)
# - First torch.cuda call: 600-1000 MB overhead for CUDA context
# - Loads cuDNN, cuBLAS, and other CUDA libraries
# - This is ONE-TIME overhead, NOT per-tensor
# - Subtract from available_memory, NOT add to per-batch estimate
#
# [CACHING ALLOCATOR] (from PyTorch torch.cuda documentation)
# - PyTorch uses caching allocator to speed up memory operations
# - nvidia-smi shows RESERVED memory (larger than actual usage)
# - torch.cuda.memory_allocated() shows ACTUAL tensor memory
# - torch.cuda.max_memory_allocated() shows PEAK tensor memory
# - torch.cuda.empty_cache() releases unused cached memory
# - Cached memory does NOT affect estimation (we estimate tensor bytes only)
#
# [TORCH.COMPILE BEHAVIOR] (from PyTorch 2.0 documentation)
# - mode='default': Standard compilation, no extra memory overhead
# - mode='reduce-overhead': Uses CUDA graphs, INCREASES memory due to caching
# - mode='max-autotune': Similar to reduce-overhead + extra autotuning
# - CUDA graphs freeze memory addresses, preventing reuse during graph
# - For mode='default', estimate as if no compilation (PyTorch optimizes)
#
# [TENSOR OPERATIONS MEMORY] (from PyTorch Internals)
# - a + b creates NEW tensor (not in-place)
# - a.add_(b) modifies a in-place (no new memory)
# - a * b * c: PyTorch creates ONE intermediate for (a * b), then result
# - .sum(), .mean() etc: result is smaller, intermediate may be released
# - PyTorch AGGRESSIVELY releases intermediates after reduction ops
#
# [SCATTER/GATHER OPERATIONS] (from PyTorch torch.Tensor.scatter_add_)
# - scatter_add_ is IN-PLACE, no extra memory for output
# - Index tensor (int64): 8 bytes per element
# - Source tensor must match dtype of target
#
# [PEAK MEMORY ESTIMATION STRATEGY]
# Peak = max over all timesteps of (sum of all live tensors at that time)
# - Student params: Always live (4 tensors)
# - SuperGraph data: Always live (F, Y, indices, mask)
# - Gather tensors: Live during forward, may be released during scatter
# - Compute temps: TRANSIENT - exist briefly during element-wise ops
# - Scatter buffers: Created sequentially (W update, then X update)
#
# For accurate estimation:
# 1. Count only tensors that exist SIMULTANEOUSLY
# 2. Use torch.cuda.max_memory_allocated() for validation
# 3. Do NOT include allocator cache (it's not real usage)
# =============================================================================

from dataclasses import dataclass, field as dataclass_field
from typing import Tuple
from enum import Enum


class DType(Enum):
    """Data type enumeration with byte sizes."""
    FLOAT32 = 4
    BFLOAT16 = 2
    FLOAT16 = 2
    INT64 = 8
    INT32 = 4
    INT8 = 1
    BOOL = 1
    
    @property
    def bytes(self) -> int:
        return self.value


@dataclass
class TensorSpec:
    """
    Specification for a single tensor's memory footprint.
    
    This modular representation allows easy dtype changes:
    - To change a tensor from FP32 to BF16, just change the dtype field
    - The memory calculation automatically updates
    
    Attributes:
        name: Human-readable name (e.g., "W_flat", "F_matrix")
        shape: Tuple of dimensions (e.g., (B, S*N, M))
        shape_formula: String formula for documentation (e.g., "(B, S*N, M)")
        dtype: Data type (affects bytes per element)
        count: Number of identical tensors (e.g., 4 for W, X, W_var, X_var)
        live_at_peak: Whether this tensor is live during peak memory usage
        notes: Optional notes about this tensor
    """
    name: str
    shape: Tuple[int, ...]
    shape_formula: str
    dtype: DType
    count: int = 1
    live_at_peak: bool = True
    notes: str = ""
    
    @property
    def elements(self) -> int:
        """Total number of elements across all tensor copies."""
        result = self.count
        for dim in self.shape:
            result *= dim
        return result
    
    @property
    def bytes(self) -> int:
        """Total bytes for this tensor (count × shape × dtype.bytes)."""
        return self.elements * self.dtype.bytes
    
    def with_dtype(self, new_dtype: DType) -> 'TensorSpec':
        """Return a copy with different dtype (for what-if analysis)."""
        return TensorSpec(
            name=self.name,
            shape=self.shape,
            shape_formula=self.shape_formula,
            dtype=new_dtype,
            count=self.count,
            live_at_peak=self.live_at_peak,
            notes=self.notes,
        )


@dataclass
class MemoryComponent:
    """
    A logical group of tensors (e.g., "Student Parameters", "SuperGraph Data").
    
    Components make it easy to:
    - See memory breakdown by category
    - Modify all tensors in a category at once
    """
    name: str
    tensors: list = dataclass_field(default_factory=list)
    notes: str = ""
    
    @property
    def bytes(self) -> int:
        """Total bytes for this component."""
        return sum(t.bytes for t in self.tensors if t.live_at_peak)
    
    @property
    def gb(self) -> float:
        """Total GB for this component."""
        return self.bytes / (1024**3)
    
    def add(self, tensor: TensorSpec) -> 'MemoryComponent':
        """Add a tensor to this component (fluent API)."""
        self.tensors.append(tensor)
        return self


@dataclass
class MemoryBreakdown:
    """
    Complete memory breakdown for an algorithm.
    
    Provides:
    - Total memory estimate
    - Per-component breakdown
    - Easy identification of which component to modify for dtype changes
    """
    algorithm_key: str
    components: list = dataclass_field(default_factory=list)
    safety_margin: float = 1.05
    override_total_gb: Optional[float] = None
    dominant_stage: str = ""
    stage_breakdown: Dict[str, float] = dataclass_field(default_factory=dict)
    
    def add_component(self, component: MemoryComponent) -> 'MemoryBreakdown':
        """Add a component (fluent API)."""
        self.components.append(component)
        return self
    
    @property
    def total_bytes(self) -> int:
        """Total bytes across all components."""
        if self.override_total_gb is not None:
            return int(float(self.override_total_gb) * (1024**3))
        return sum(c.bytes for c in self.components)
    
    @property
    def total_gb(self) -> float:
        """Total GB with safety margin."""
        base_gb = float(self.override_total_gb) if self.override_total_gb is not None else self.total_bytes / (1024**3)
        return base_gb * self.safety_margin
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [f"Memory Breakdown for {self.algorithm_key}:"]
        for c in self.components:
            pct = (c.bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0
            lines.append(f"  {c.name}: {c.gb:.3f} GB ({pct:.1f}%)")
        lines.append(f"  Total (with {self.safety_margin:.0%} margin): {self.total_gb:.3f} GB")
        return "\n".join(lines)


def _tensor_gb(shape: Tuple[int, ...], dtype: DType, count: int = 1) -> float:
    elements = int(count)
    for dim in shape:
        elements *= int(dim)
    return elements * dtype.bytes / (1024**3)


def _stage_override(
    breakdown: MemoryBreakdown,
    stages: Dict[str, float],
    *,
    safety_margin: float = 1.0,
) -> MemoryBreakdown:
    if stages:
        dominant_stage, dominant_gb = max(stages.items(), key=lambda item: item[1])
        breakdown.override_total_gb = float(dominant_gb)
        breakdown.dominant_stage = dominant_stage
        breakdown.stage_breakdown = {key: float(value) for key, value in stages.items()}
    breakdown.safety_margin = float(safety_margin)
    return breakdown




class MemoryEstimator:
    """
    Unified memory estimator with algorithm-specific estimation functions.
    
    Usage:
        estimator = MemoryEstimator()
        
        params = EstimationParams(
            N1=600, N2=600, M=150, S=20,
            alpha_values=[0.0, 0.5, 1.0, 1.5, 2.0],
            algorithm_key='bigamp_spreading'
        )
        
        estimate = estimator.estimate(params)
        print(f"Estimated memory: {estimate.total_gb:.2f} GB")
    """
    
    # Class-level registry of estimation functions
    _estimators: Dict[str, Callable[[EstimationParams], float]] = {}
    
    # Class-level calibration factors (loaded per GPU model)
    _calibration_data: Dict[str, Dict] = {}
    
    def __init__(self, calibration_dir: Optional[Path] = None, apply_calibration: bool = True):
        """
        Initialize memory estimator.
        
        Args:
            calibration_dir: Path to calibration data directory.
                           If None, uses default location.
        """
        if calibration_dir is None:
            self.calibration_dir = Path(__file__).parent / "calibration" / "data"
        else:
            self.calibration_dir = Path(calibration_dir)
        
        self.apply_calibration = bool(apply_calibration)
        self.gpu_model = self._get_gpu_model()
        self._load_calibration()
    
    @classmethod
    def register(cls, algo_key: str):
        """
        Decorator to register an algorithm-specific estimation function.
        
        Usage:
            @MemoryEstimator.register("my_algorithm")
            def estimate_my_algorithm(params: EstimationParams) -> float:
                # Return estimated memory in GB
                return computed_memory_gb
        """
        def decorator(fn: Callable[[EstimationParams], float]):
            cls._estimators[algo_key] = fn
            logger.debug(f"Registered estimator for algorithm: {algo_key}")
            return fn
        return decorator
    
    def estimate(
        self, 
        params: EstimationParams, 
        runtime_stats: Optional[Dict] = None
    ) -> MemoryEstimate:
        """
        Estimate precise memory requirements using dynamic system stats.
        
        Algorithm:
        1. Calculate pure mathematical tensor requirements (Static).
        2. Add Runtime Overheads (Dynamic):
           - Context: One-time CUDA init cost
           - Fragmentation: Inactive but reserved memory (cannot be freed)
           
        Args:
            params: Estimation parameters
            runtime_stats: Optional stats dict (from torch.cuda.memory_stats())
        """
        if params.algorithm_key not in self._estimators:
            raise ValueError(f"Unknown algorithm: {params.algorithm_key}")
        
        # 1. Pure Tensor Math, then optional local calibration from real runs.
        uncalibrated_tensor_gb = self._estimators[params.algorithm_key](params)
        calibration = self._get_calibration_adjustment(params.algorithm_key) if self.apply_calibration else {}
        calibration_factor = 1.0
        calibration_applied = 0.0
        calibration_source = "theory_unchecked"
        raw_tensor_gb = uncalibrated_tensor_gb
        if calibration:
            min_raw_value = calibration.get("min_raw_gb_for_apply", 1.0)
            min_raw_gb = 1.0 if min_raw_value is None else float(min_raw_value)
            if uncalibrated_tensor_gb >= min_raw_gb:
                calibration_factor = max(1.0, float(calibration.get("factor", 1.0) or 1.0))
                raw_tensor_gb = uncalibrated_tensor_gb * calibration_factor
                calibration_applied = 1.0
                calibration_source = str(calibration.get("source", "local_calibration"))
        
        # 2. Dynamic Overhead Calculation
        overhead_gb = 0.0
        fragmentation_gb = 0.0
        context_gb = 0.0
        
        if torch.cuda.is_available():
            try:
                # Use current stats if not provided
                stats = runtime_stats or torch.cuda.memory_stats()
                
                # Context + Driver: Difference between Reserved and Allocated (approximation)
                # But safer to assume a baseline min context if we are starting fresh
                current_reserved = torch.cuda.memory_reserved()
                current_allocated = torch.cuda.memory_allocated()
                
                # Fragmentation: inactive_split_bytes (cached but unusable for large blocks)
                # This is the precise measure of "wasted" memory by CachingAllocator
                frag_bytes = stats.get("inactive_split.all.current", 0)
                
                # Context: Initial usually ~500MB-800MB
                # If we are already running (reserved > 0), we can see how much is "hidden"
                # Hidden = (Reserved - Allocated - InactiveSplit)
                # This part is effectively the context/workspace
                if current_reserved > 0:
                    hidden_bytes = current_reserved - current_allocated - frag_bytes
                    context_gb = max(hidden_bytes, 0) / (1024**3)
                else:
                    context_gb = 0.6  # Default cold-start context assumption (600MB)
                
                fragmentation_gb = frag_bytes / (1024**3)
                
                # LOGIC:
                # We need enough space for:
                # (New Tensors) + (Existing Context) + (Existing Fragmentation)
                # Note: Some fragmentation might be reusable if blocks match, but 
                # strictly assuming it's overhead is the "Safe" precise way.
                
                overhead_gb = context_gb + fragmentation_gb
                
            except Exception as e:
                logger.warning(f"Failed to get dynamic stats: {e}")
                context_gb = 0.6 # Fallback
                overhead_gb = 0.6
        
        device_overhead_gb = max(
            overhead_gb,
            self._algorithm_device_overhead_gb(params, raw_tensor_gb),
        )
        total_gb = raw_tensor_gb + device_overhead_gb
        
        breakdown_obj = self._get_breakdown_object(params)
        breakdown = self._breakdown_to_dict(breakdown_obj)
        persistent_tensors = self._persistent_tensor_breakdown(breakdown_obj)
        transient_peak_tensors = dict(breakdown_obj.stage_breakdown) if breakdown_obj else {}
        if context_gb:
            breakdown["runtime.context_gb"] = context_gb
        if fragmentation_gb:
            breakdown["runtime.fragmentation_gb"] = fragmentation_gb
        breakdown["runtime.device_overhead_gb"] = device_overhead_gb
        breakdown["calibration.uncalibrated_tensor_gb"] = uncalibrated_tensor_gb
        breakdown["calibration.applied_factor"] = calibration_factor
        breakdown["calibration.applied"] = calibration_applied
        breakdown["calibration.calibrated_tensor_gb"] = raw_tensor_gb
        breakdown["estimator.raw_peak_allocated_gb"] = uncalibrated_tensor_gb
        breakdown["estimator.device_peak_gb"] = total_gb
        if breakdown_obj and breakdown_obj.dominant_stage:
            breakdown["estimator.dominant_stage_gb"] = float(
                breakdown_obj.stage_breakdown.get(breakdown_obj.dominant_stage, uncalibrated_tensor_gb)
            )
        breakdown_msg = (
            f"  Math Tensors: {raw_tensor_gb:.2f} GB\n"
            f"  + Context:    {context_gb:.2f} GB\n"
            f"  + Fragmentation:{fragmentation_gb:.2f} GB\n"
            f"  = Total Req:  {total_gb:.2f} GB"
        )
        # logger.debug(f"Precise Estimate:\n{breakdown_msg}")

        return MemoryEstimate(
            total_gb=total_gb,
            per_batch_gb=total_gb / max(1, params.batch_size),
            breakdown=breakdown,
            confidence=self._compute_confidence(params),
            raw_peak_allocated_gb=uncalibrated_tensor_gb,
            device_peak_gb=total_gb,
            dominant_stage=breakdown_obj.dominant_stage if breakdown_obj else "",
            stage_breakdown=dict(breakdown_obj.stage_breakdown) if breakdown_obj else {},
            persistent_tensors=persistent_tensors,
            transient_peak_tensors=transient_peak_tensors,
            calibration_source=calibration_source,
        )
    
    def estimate_raw(self, params: EstimationParams) -> float:
        """
        Get raw (uncalibrated) memory estimate.
        
        Args:
            params: Estimation parameters
            
        Returns:
            Raw estimated memory in GB
        """
        if params.algorithm_key not in self._estimators:
            raise ValueError(f"Unknown algorithm: {params.algorithm_key}")
        return self._estimators[params.algorithm_key](params)
    
    def record_actual(self, params: EstimationParams, actual_gb: float) -> None:
        """
        Record actual memory usage for future calibration.
        
        Args:
            params: The parameters used
            actual_gb: Actual peak memory in GB
        """
        estimated = self.estimate_raw(params)
        error = (estimated - actual_gb) / actual_gb if actual_gb > 0 else 0
        
        record = {
            "N1": params.N1,
            "N2": params.N2,
            "M": params.M,
            "S": params.S,
            "alpha_max": params.alpha_max,
            "batch_size": params.batch_size,
            "estimated_gb": estimated,
            "actual_gb": actual_gb,
            "error_pct": error * 100,
            "use_compile": params.use_compile,
            "use_bf16": params.use_bf16,
        }
        
        # Append to history file
        history_file = self.calibration_dir / f"{self.gpu_model}_history.jsonl"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        logger.info(
            f"Recorded memory usage: estimated={estimated:.2f}GB, "
            f"actual={actual_gb:.2f}GB, error={error:.1%}"
        )
    
    def record_oom_event(self, usage_ratio: float, params: Optional[EstimationParams] = None):
        """
        Record an OOM event for calibration feedback.
        
        Args:
            usage_ratio: Memory usage ratio when OOM occurred
            params: Optional parameters that caused OOM
        """
        record = {
            "event": "OOM",
            "usage_ratio": usage_ratio,
            "gpu_model": self.gpu_model,
        }
        if params:
            record.update({
                "N1": params.N1,
                "N2": params.N2,
                "M": params.M,
                "S": params.S,
                "alpha_max": params.alpha_max,
            })
        
        oom_file = self.calibration_dir / f"{self.gpu_model}_oom_events.jsonl"
        oom_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(oom_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        logger.warning(f"Recorded OOM event at {usage_ratio:.1%} usage")
    
    def get_available_memory(self) -> float:
        """
        Get available GPU memory in GB.
        
        Returns:
            Available memory in GB (0.0 if no GPU)
        """
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            available = total - reserved
            return available / (1024**3)
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
            return 0.0
    
    def get_total_memory(self) -> float:
        """
        Get total GPU memory in GB.
        
        Returns:
            Total memory in GB (0.0 if no GPU)
        """
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            return total / (1024**3)
        except Exception:
            return 0.0
    
    def _get_gpu_model(self) -> str:
        """Get GPU model name for calibration lookup."""
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                # Sanitize for filesystem
                return name.replace(" ", "_").replace("/", "-")
            except Exception:
                pass
        return "cpu"
    
    def _load_calibration(self) -> None:
        """Load calibration data for current GPU."""
        calib_file = self.calibration_dir / f"{self.gpu_model}.json"
        
        if calib_file.exists():
            try:
                with open(calib_file) as f:
                    self._calibration_data[self.gpu_model] = json.load(f)
                logger.info(f"Loaded calibration data for {self.gpu_model}")
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")
        local_file = self._local_calibration_file()
        if local_file.exists():
            try:
                with open(local_file, encoding="utf-8") as f:
                    local_payload = json.load(f)
                self._merge_calibration_payload(local_payload, source=str(local_file))
                logger.info(f"Loaded local calibration data from {local_file}")
            except Exception as e:
                logger.warning(f"Failed to load local calibration: {e}")

    def _merge_calibration_payload(self, payload: Dict, source: str) -> None:
        """Merge local runtime calibration coefficients into the active GPU data."""
        payload_gpu = str(payload.get("gpu_model", self.gpu_model))
        if payload_gpu not in {self.gpu_model, self.gpu_model.replace(" ", "_").replace("/", "-")}:
            return
        algorithms = payload.get("algorithms")
        if not isinstance(algorithms, dict):
            return
        gpu_data = self._calibration_data.setdefault(self.gpu_model, {})
        for algo_key, algo_payload in algorithms.items():
            if not isinstance(algo_payload, dict):
                continue
            merged = dict(gpu_data.get(algo_key, {}))
            merged.update(algo_payload)
            merged["source"] = source
            gpu_data[algo_key] = merged

    @staticmethod
    def _local_calibration_file() -> Path:
        return Path.cwd() / "runs" / "calibration" / "memory" / "latest_coefficients.json"
    
    def _get_calibration_factor(self, algo_key: str) -> float:
        """Get calibration factor for algorithm on current GPU."""
        adjustment = self._get_calibration_adjustment(algo_key)
        if adjustment:
            return float(adjustment.get("factor", 1.0) or 1.0)
        return 1.0  # No calibration, use raw estimate

    def _get_calibration_adjustment(self, algo_key: str) -> Dict:
        """Get full calibration adjustment payload for an algorithm."""
        if self.gpu_model in self._calibration_data:
            gpu_data = self._calibration_data[self.gpu_model]
            if algo_key in gpu_data:
                return gpu_data[algo_key]
        return {}
    
    def _compute_confidence(self, params: EstimationParams) -> float:
        """
        Compute confidence level for estimate.
        
        Confidence is higher when:
        - Calibration data is available
        - Parameters are within calibrated ranges
        - Algorithm is well-tested
        """
        base_confidence = 0.6  # Default without calibration
        
        # Boost if calibration data exists
        if self.gpu_model in self._calibration_data:
            if params.algorithm_key in self._calibration_data[self.gpu_model]:
                base_confidence = 0.85
        
        # Reduce for very large problems (less tested)
        if params.N1 * params.N2 > 1_000_000:  # > 1000x1000
            base_confidence *= 0.9
        
        # Reduce for edge cases
        if params.alpha_max > 4.0 or params.S > 50:
            base_confidence *= 0.9
        
        return min(1.0, base_confidence)
    
    def _get_breakdown_object(self, params: EstimationParams) -> Optional[MemoryBreakdown]:
        """Get memory breakdown object with component and stage metadata."""
        try:
            if params.algorithm_key == "bigamp":
                breakdown = get_bigamp_standard_breakdown(params)
            elif params.algorithm_key == "agd":
                breakdown = get_agd_breakdown(params)
            elif params.algorithm_key == "bigamp_spreading":
                if getattr(params, "allow_intra_connection", False):
                    from .memory_estimator_general import get_general_spreading_breakdown
                    breakdown = get_general_spreading_breakdown(params)
                else:
                    breakdown = get_spreading_parallel_breakdown(params)
            elif params.algorithm_key in {"bigamp_tensor", "bigamp_tensor_parallel"}:
                breakdown = get_tensor_spreading_breakdown(params)
            else:
                return None
        except Exception as exc:
            logger.warning(
                "Failed to build memory breakdown for %s: %s",
                params.algorithm_key,
                exc,
            )
            return None
        return breakdown

    def _breakdown_to_dict(self, breakdown: Optional[MemoryBreakdown]) -> Dict[str, float]:
        """Convert a breakdown object to numeric metadata for JSON outputs."""
        if breakdown is None:
            return {}
        result = {component.name: component.gb for component in breakdown.components}
        for stage, gb in breakdown.stage_breakdown.items():
            result[f"stage.{stage}"] = float(gb)
        if breakdown.override_total_gb is not None:
            result["stage.raw_peak_allocated_gb"] = breakdown.total_gb
        return result

    def _persistent_tensor_breakdown(self, breakdown: Optional[MemoryBreakdown]) -> Dict[str, float]:
        """Component-level resident tensor summary for scan/resource metadata."""
        if breakdown is None:
            return {}
        return {
            component.name: float(component.gb)
            for component in breakdown.components
        }

    def _algorithm_device_overhead_gb(self, params: EstimationParams, raw_tensor_gb: float) -> float:
        """Estimate device-used reserve overhead beyond max allocated tensors."""
        algorithm_key = params.algorithm_key
        if algorithm_key == "bigamp_spreading":
            return 0.27 * float(raw_tensor_gb)
        if algorithm_key == "agd":
            return 0.17 * float(raw_tensor_gb)
        if algorithm_key in {"bigamp", "bigamp_tensor", "bigamp_tensor_parallel"}:
            return 0.06 * float(raw_tensor_gb)
        return 0.6

    def _get_breakdown(self, params: EstimationParams) -> Dict[str, float]:
        """Get memory breakdown by component (algorithm-specific)."""
        return self._breakdown_to_dict(self._get_breakdown_object(params))


# =============================================================================
# Built-in Algorithm Estimators
# =============================================================================

@MemoryEstimator.register("bigamp_spreading")
def estimate_spreading_parallel(params: EstimationParams) -> float:
    """
    BiG-AMP Spreading Parallel VRAM estimation using modular tensor components.
    
    Automatically detects General vs Bipartite mode and uses appropriate estimation.
    
    MODULAR DESIGN:
    - Each tensor is defined with explicit dtype
    - Easy to change dtype (e.g., FP32 -> BF16 -> INT8) by modifying the dtype field
    - Memory breakdown shows which component uses how much
    
    See get_spreading_parallel_breakdown() for detailed component analysis.
    """
    # Check for General Graph mode
    if getattr(params, 'allow_intra_connection', False):
        from .memory_estimator_general import estimate_general_spreading
        return estimate_general_spreading(params)
    
    breakdown = get_spreading_parallel_breakdown(params)
    return breakdown.total_gb


def get_spreading_parallel_breakdown(params: EstimationParams) -> MemoryBreakdown:
    """
    Get detailed modular memory breakdown for bigamp_spreading.
    
    This function defines each tensor explicitly, making it easy to:
    1. See where memory is used
    2. Change dtype for specific tensors
    3. Validate against actual usage
    
    Returns:
        MemoryBreakdown with all components and their tensors
    """
    N1, N2, M, S = params.N1, params.N2, params.M, params.S
    B = params.batch_size  # Alpha batch size
    alpha_max = params.alpha_max
    
    # Compute derived dimensions.
    #
    # Must match modules/graphs/supergraph.py:create_supergraph exactly:
    #   C_alpha = floor(alpha * M * N1), capped by N1*N2.
    #
    # This is intentionally not alpha * N1 * N2 / M.  The spreading runner
    # constructs F/Y/indices from the supergraph formula above, and the step
    # kernel gathers tensors of shape (B, S*C_max, M).  Underestimating C_max
    # here lets the planner alpha-fold unsafe batches and can surface as a
    # low-level CUDA driver error during advanced indexing.
    import math
    C_max = int(math.floor(alpha_max * M * N1))
    C_max = max(1, min(C_max, N1 * N2))
    SC = S * C_max  # Total edges
    SN1 = S * N1  # Flattened row dim
    SN2 = S * N2  # Flattened col dim
    
    # Dtype selection based on params
    storage_dtype = DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32
    f_dtype = DType.INT8 if params.f_distribution == 'rademacher' else DType.FLOAT32
    
    # Adaptive mode needs slight margin due to extra intermediates (now accurate, was 1.8x when formula was wrong)
    adaptive_margin = 1.2 if params.adaptive_damping else 1.0
    breakdown = MemoryBreakdown(algorithm_key="bigamp_spreading", safety_margin=adaptive_margin)
    
    # =========================================================================
    # COMPONENT 1: Student Parameters
    # W_flat, X_flat, W_var_flat, X_var_flat: 4 × (B, S*N, M)
    # DTYPE: storage_dtype (BF16 or FP32)
    # =========================================================================
    student_params = MemoryComponent(
        name="Student Parameters",
        notes="W_flat, X_flat, W_var_flat, X_var_flat"
    )
    student_params.add(TensorSpec(
        name="W_flat + X_flat + W_var_flat + X_var_flat",
        shape=(B, SN1 + SN2, M),  # Combined for simplicity
        shape_formula="4 × (B, S*N, M)",
        dtype=storage_dtype,
        count=4,
        notes="Main student parameters in flat format"
    ))
    breakdown.add_component(student_params)
    
    # =========================================================================
    # COMPONENT 2: SuperGraph Data
    # F_flat, Y_flat, indices, alpha_mask
    # DTYPE: varies (F: int8/float32, Y: float32, indices: int64, mask: bool)
    # =========================================================================
    supergraph = MemoryComponent(
        name="SuperGraph Data",
        notes="Sparse structure: F, Y, indices, masks"
    )
    supergraph.add(TensorSpec(
        name="F_flat",
        shape=(SC, M),
        shape_formula="(S*C_max, M)",
        dtype=f_dtype,  # INT8 for Rademacher, FLOAT32 for Gaussian
        notes="Spreading coefficients - dtype depends on f_distribution"
    ))
    supergraph.add(TensorSpec(
        name="Y_flat",
        shape=(SC,),
        shape_formula="(S*C_max,)",
        dtype=DType.FLOAT32,
        notes="Observation values"
    ))
    supergraph.add(TensorSpec(
        name="i_offset + j_offset",
        shape=(SC,),
        shape_formula="(S*C_max,)",
        dtype=DType.INT64,
        count=2,
        notes="Row and column indices"
    ))
    supergraph.add(TensorSpec(
        name="alpha_mask_exp",
        shape=(B, SC),
        shape_formula="(B, S*C_max)",
        dtype=DType.BOOL,
        notes="Alpha batch mask"
    ))
    breakdown.add_component(supergraph)
    
    # =========================================================================
    # COMPONENT 3: Gather Tensors (Peak compute)
    # W_sel, X_sel: (A, SC, M) - advanced indexing creates copies
    # DTYPE: storage_dtype
    # NOTE: Only 2 live at peak (not all 4)
    # =========================================================================
    gather = MemoryComponent(
        name="Gather Tensors",
        notes="W_sel, X_sel via advanced indexing (copies)"
    )
    gather.add(TensorSpec(
        name="W_sel + X_sel + W_var_sel + X_var_sel",
        shape=(B, SC, M),
        shape_formula="(B, S*C_max, M)",
        dtype=storage_dtype,
        count=3,  # BALANCED: count=4 overest large M, count=2 underest small M
        notes="Gathered student values - ~3 live on average due to partial reuse"
    ))
    breakdown.add_component(gather)
    
    # =========================================================================
    # COMPONENT 4: Forward/Variance Compute
    # F_compute, Z_hat, V, s_values, denom
    # PLUS: Temporary tensors from element-wise operations!
    # CRITICAL: (F * W * X).sum() creates intermediate (B, SC, M) tensors
    # DTYPE: storage_dtype for compute, varies for intermediate
    # =========================================================================
    forward = MemoryComponent(
        name="Forward Compute",
        notes="Z_hat, V, s_values, denom + CRITICAL: temp (B,SC,M) tensors"
    )
    forward.add(TensorSpec(
        name="F_compute",
        shape=(SC, M),
        shape_formula="(S*C_max, M)",
        dtype=storage_dtype,
        notes="F converted to compute dtype"
    ))
    forward.add(TensorSpec(
        name="Z_hat + V + s_values + denom",
        shape=(B, SC),
        shape_formula="(B, S*C_max)",
        dtype=storage_dtype,
        count=2,  # ADJUSTED: PyTorch reuses buffers, not all 4 live simultaneously
        notes="Edge-level intermediate values (Z_hat, V reused for s, denom)"
    ))
    # CRITICAL ADDITION: Forward compute temporary tensors
    # Z_hat = (F * W * X).sum() creates: F*W (temp1), temp1*X (temp2)
    # V = (W_var * X^2 + W^2 * X_var).sum() creates: X^2, W^2, W_var*X^2, W^2*X_var, sum
    # At peak, at least 3 such (B, SC, M) tensors exist simultaneously
    forward.add(TensorSpec(
        name="compute_temps (1 at peak)",
        shape=(B, SC, M),
        shape_formula="(B, S*C_max, M)",
        dtype=storage_dtype,
        count=1,  # FIXED: PyTorch releases temps immediately after .sum()
        notes="Only 1 temp exists at peak - verified via actual VRAM measurement"
    ))
    breakdown.add_component(forward)
    
    # =========================================================================
    # COMPONENT 5: Scatter Buffers
    # r_W, tau_W / r_X, tau_X: sequential, so max not sum
    # DTYPE: storage_dtype
    # NOTE: W and X buffers not both live at peak
    # =========================================================================
    scatter = MemoryComponent(
        name="Scatter Buffers",
        notes="r_W/tau_W and r_X/tau_X (sequential)"
    )
    # Take max of W and X scatter (not both live)
    scatter_dim = max(SN1, SN2)
    scatter.add(TensorSpec(
        name="r + tau (simultaneous)",
        shape=(B, scatter_dim, M),
        shape_formula="max((B,S*N1,M), (B,S*N2,M))",
        dtype=storage_dtype,
        count=2,  # FIXED BACK: r_W and tau_W exist SIMULTANEOUSLY for W_hat_new calc
        notes="r and tau are both live during update: W_hat_new = W + W_var * r"
    ))
    scatter.add(TensorSpec(
        name="contrib_tensor",
        shape=(B, SC, M),
        shape_formula="(B, S*C_max, M)",
        dtype=storage_dtype,
        count=1,  # Only 1 live at a time
        notes="Contribution tensor for scatter_add"
    ))
    breakdown.add_component(scatter)
    
    # =========================================================================
    # COMPONENT 6: Adaptive Damping Backtracking
    # W_safe, X_safe, W_var_safe, X_var_safe
    # DTYPE: storage_dtype
    # =========================================================================
    if params.adaptive_damping:
        adaptive = MemoryComponent(
            name="Adaptive Backtracking",
            notes="Safe state copies + Raw updates: W_safe/raw, X_safe/raw, W_var_safe/raw, X_var_safe/raw"
        )
        # Safe state copies (4 tensors)
        adaptive.add(TensorSpec(
            name="Safe State Copies",
            shape=(B, SN1 + SN2, M),
            shape_formula="4 × (B, S*N, M)",
            dtype=storage_dtype,
            count=4,
            notes="W_safe, X_safe, W_var_safe, X_var_safe"
        ))
        # Raw update tensors returned by step function (4 tensors: W_raw, X_raw, W_var_raw, X_var_raw)
        adaptive.add(TensorSpec(
            name="Raw Update Tensors",
            shape=(B, SN1 + SN2, M),
            shape_formula="4 × (B, S*N, M)",
            dtype=storage_dtype,
            count=4,
            notes="W_raw, X_raw, W_var_raw, X_var_raw from step function"
        ))
        # Additional intermediate: Z_hat, V, s_vals (edge-level)
        adaptive.add(TensorSpec(
            name="Adaptive Intermediates",
            shape=(B, SC),
            shape_formula="3 × (B, S*C_max)",
            dtype=storage_dtype,
            count=3,
            notes="Z_hat, V, s_vals for log-likelihood computation"
        ))
        breakdown.add_component(adaptive)
    
    persistent_gb = student_params.gb + supergraph.gb
    edge_workspace_gb = _tensor_gb((B, SC, M), storage_dtype)
    edge_scalar_gb = _tensor_gb((B, SC), storage_dtype)
    f_compute_gb = _tensor_gb((SC, M), storage_dtype)
    scatter_state_gb = _tensor_gb((B, scatter_dim, M), storage_dtype, count=2)
    state_update_gb = _tensor_gb((B, S * N1, M), storage_dtype, count=8)
    if N2 != N1:
        state_update_gb = (
            _tensor_gb((B, S * N1, M), storage_dtype, count=4)
            + _tensor_gb((B, S * N2, M), storage_dtype, count=4)
        )

    # create_supergraph() can allocate a batched random-noise matrix on CUDA
    # before the algorithm step starts.  This peak is independent of alpha
    # batching but is part of the run-level torch.cuda.max_memory_allocated().
    total_edges = N1 * N2
    graph_noise_gb = 0.0
    graph_topk_partition_gb = 0.0
    graph_topk_selection_gb = 0.0
    if S * total_edges * DType.FLOAT16.bytes / (1024**3) < 4.0:
        graph_noise_gb = _tensor_gb((S, total_edges), DType.FLOAT16)
        # torch.topk(noise, k=C_max, dim=1) is not just the returned
        # (values, indices).  On CUDA it uses full-row partition work buffers.
        # Model those buffers explicitly by shape instead of hiding them in a
        # global calibration multiplier.
        graph_topk_partition_gb = _tensor_gb((S, total_edges), DType.FLOAT16)
        graph_topk_selection_gb = _tensor_gb((S, max(1, total_edges // 2)), DType.FLOAT16)
    graph_topk_gb = (
        _tensor_gb((S, C_max), DType.FLOAT16)  # topk values
        + _tensor_gb((S, C_max), DType.INT64)  # topk indices
    )

    # Explicit live-tensor model for bigamp_step_disjoint_union_flat().
    #
    # The compiled and eager paths have different live-buffer behavior.  With
    # torch.compile=True, Inductor fuses/reuses the elementwise chains feeding
    # reductions and scatter_add_; the unavoidable peak is dominated by the
    # four advanced-index gather copies.  In eager mode those chains can
    # materialize more edge-shaped temporaries, so keep the larger model there.
    gather_selected_gb = _tensor_gb((B, SC, M), storage_dtype, count=4)
    scalar_edge_workspace_gb = edge_scalar_gb * 8

    if params.use_compile:
        compiled_scalar_workspace_gb = edge_scalar_gb * 8
        compiled_fused_source_gb = _tensor_gb((B, SC, M), storage_dtype)
        compiled_scatter_peak_gb = (
            persistent_gb
            + _tensor_gb((B, SC, M), storage_dtype, count=2)  # selected factors used by the source expression
            + compiled_fused_source_gb
            + scatter_state_gb
            + compiled_scalar_workspace_gb
        )
        forward_peak_gb = (
            persistent_gb
            + gather_selected_gb
            + f_compute_gb
            + compiled_scalar_workspace_gb
        )
        variance_peak_gb = forward_peak_gb
        scatter_x_peak_gb = max(forward_peak_gb, compiled_scatter_peak_gb)
    else:
        forward_expression_gb = _tensor_gb((B, SC, M), storage_dtype, count=2)
        variance_expression_gb = _tensor_gb((B, SC, M), storage_dtype, count=4)
        retained_w_contrib_gb = _tensor_gb((B, SC, M), storage_dtype, count=2)
        current_x_contrib_gb = _tensor_gb((B, SC, M), storage_dtype, count=2)
        elementwise_chain_scratch_gb = _tensor_gb((B, SC, M), storage_dtype, count=2)
        scatter_index_workspace_gb = _tensor_gb((B, SC, M), DType.INT64)

        forward_peak_gb = (
            persistent_gb
            + gather_selected_gb
            + f_compute_gb
            + forward_expression_gb
            + scalar_edge_workspace_gb
        )
        variance_peak_gb = (
            persistent_gb
            + gather_selected_gb
            + f_compute_gb
            + variance_expression_gb
            + scalar_edge_workspace_gb
        )
        scatter_x_peak_gb = (
            persistent_gb
            + gather_selected_gb
            + f_compute_gb
            + retained_w_contrib_gb
            + current_x_contrib_gb
            + elementwise_chain_scratch_gb
            + scatter_index_workspace_gb
            + scalar_edge_workspace_gb
            + state_update_gb
        )

    stages = {
        "spreading_persistent_state": persistent_gb,
        "spreading_graph_generation_peak": (
            supergraph.gb
            + graph_noise_gb
            + graph_topk_partition_gb
            + graph_topk_selection_gb
            + graph_topk_gb
        ),
        "spreading_forward_peak": forward_peak_gb,
        "spreading_variance_peak": variance_peak_gb,
        "spreading_edge_update_peak": scatter_x_peak_gb,
        "spreading_scatter_peak": persistent_gb + edge_workspace_gb + scatter_state_gb,
    }
    if params.adaptive_damping:
        stages["spreading_adaptive_backtracking_peak"] = breakdown.total_gb

    return _stage_override(breakdown, stages, safety_margin=1.0)


@MemoryEstimator.register("bigamp")
def estimate_bigamp_standard(params: EstimationParams) -> float:
    """
    Standard BiG-AMP memory estimation using modular components.
    
    Uses dense matrix operations, simpler memory model than spreading version.
    """
    breakdown = get_bigamp_standard_breakdown(params)
    return breakdown.total_gb


def get_bigamp_standard_breakdown(params: EstimationParams) -> MemoryBreakdown:
    """
    Get modular memory breakdown for standard BiG-AMP.
    
    Dense algorithm with:
    - Student parameters: W, X, W_var, X_var → (B, S, N, M)
    - Dense intermediate: (B, S, N1, N2) operations (z_hat, V, residual, s, etc.)
    - Compute intermediate: (B, S, N, M) operations (w_sq, x_sq, tau, r, etc.)
    
    Note: B = batch_size = len(alpha_values) for parallel alpha processing.
    """
    N1, N2, M, S = params.N1, params.N2, params.M, params.S
    B = params.batch_size  # Alpha batch size (num_alphas for parallel processing)
    storage_dtype = DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32
    
    breakdown = MemoryBreakdown(algorithm_key="bigamp", safety_margin=1.15)
    
    # =========================================================================
    # COMPONENT 1: Student Parameters
    # w_hat, x_hat, w_var, x_var: 4 × (B, S, N, M)
    # DTYPE: storage_dtype
    # NOTE: B dimension for parallel alpha processing
    # =========================================================================
    student = MemoryComponent(name="Student Parameters")
    student.add(TensorSpec(
        name="w_hat + x_hat + w_var + x_var",
        shape=(B, S, N1 + N2, M),
        shape_formula="4 × (B, S, N, M)",
        dtype=storage_dtype,
        count=4,
    ))
    breakdown.add_component(student)
    
    # =========================================================================
    # COMPONENT 2: Dense Intermediate (N1 x N2)
    # z_hat, p_var, V, residual, s (per W and X update = 2x)
    # Shape: (B, S, N1, N2)
    # DTYPE: storage_dtype
    # Count: ~8-10 tensors (z_hat, p_var, V, residual, s for W update, 
    #        then z_hat2, p_var2, V2, residual2, s2 for X update)
    # =========================================================================
    intermediate_dense = MemoryComponent(name="Dense Intermediate (N1×N2)")
    num_dense = 8 if params.use_compile else 10  # torch.compile may fuse some
    intermediate_dense.add(TensorSpec(
        name="z_hat + p_var + V + residual + s (×2)",
        shape=(B, S, N1, N2),
        shape_formula="(B, S, N1, N2)",
        dtype=storage_dtype,
        count=num_dense,
        notes="Dense intermediate tensors for forward pass"
    ))
    breakdown.add_component(intermediate_dense)
    
    # =========================================================================
    # COMPONENT 3: Compute Intermediate (N × M)  *** PREVIOUSLY MISSING! ***
    # w_sq, x_sq, tau_W, tau_X, r_W, r_X, w_var_new, x_var_new, w_hat_new, x_hat_new
    # Shape: (B, S, N, M)
    # DTYPE: storage_dtype
    # Count: ~8-10 tensors live at peak
    # =========================================================================
    intermediate_compute = MemoryComponent(name="Compute Intermediate (N×M)")
    num_compute = 8 if params.use_compile else 10
    intermediate_compute.add(TensorSpec(
        name="w_sq + x_sq + tau + r + var_new + hat_new",
        shape=(B, S, N1 + N2, M),  # Combined W and X
        shape_formula="(B, S, N, M)",
        dtype=storage_dtype,
        count=num_compute,
        notes="Compute intermediate tensors: w_sq, x_sq, tau_W/X, r_W/X, *_new"
    ))
    breakdown.add_component(intermediate_compute)

    previous_total_gb = breakdown.total_gb
    stages = {
        "bigamp_student_state": student.gb,
        "bigamp_dense_update_peak": previous_total_gb,
        "bigamp_compute_update_peak": student.gb + intermediate_compute.gb,
    }
    return _stage_override(breakdown, stages, safety_margin=1.0)


@MemoryEstimator.register("agd")
def estimate_agd(params: EstimationParams) -> float:
    """
    AGD (Alternating Gradient Descent) memory estimation using modular components.
    
    Lighter memory footprint than BiG-AMP due to simpler update rules.
    """
    breakdown = get_agd_breakdown(params)
    return breakdown.total_gb


def get_agd_breakdown(params: EstimationParams) -> MemoryBreakdown:
    """
    Get modular memory breakdown for AGD algorithm.
    
    Simple gradient descent with:
    - Parameters: W, X → (B, S, N, M)
    - Gradients: grad_W, grad_X → (B, S, N, M)
    - Masks and predictions → (B, S, N1, N2)
    
    Note: B = batch_size = len(alpha_values) for parallel alpha processing.
    """
    N1, N2, M, S = params.N1, params.N2, params.M, params.S
    B = params.batch_size  # Alpha batch size (num_alphas for parallel processing)
    storage_dtype = DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32
    
    breakdown = MemoryBreakdown(algorithm_key="agd", safety_margin=1.0)
    
    # =========================================================================
    # COMPONENT 1: Parameters and Gradients
    # W, X, grad_W, grad_X: 4 × (B, S, N, M)
    # DTYPE: storage_dtype
    # NOTE: B dimension for parallel alpha processing
    # =========================================================================
    params_comp = MemoryComponent(name="Parameters & Gradients")
    params_comp.add(TensorSpec(
        name="W + X + grad_W + grad_X",
        shape=(B, S, N1 + N2, M),
        shape_formula="4 × (B, S, N, M)",
        dtype=storage_dtype,
        count=4,
    ))
    breakdown.add_component(params_comp)
    
    # =========================================================================
    # COMPONENT 2: Observation Masks
    # Binary masks: (B, N1, N2)
    # DTYPE: BOOL (1 byte)
    # NOTE: The implementation builds masks per alpha and broadcasts over S:
    # A_all = masks.unsqueeze(1), so S is not a resident mask dimension.
    # =========================================================================
    masks = MemoryComponent(name="Observation Masks")
    masks.add(TensorSpec(
        name="observation_mask",
        shape=(B, N1, N2),
        shape_formula="(B, N1, N2)",
        dtype=DType.BOOL,
        notes="Binary observation mask for parallel alpha"
    ))
    breakdown.add_component(masks)
    
    # =========================================================================
    # COMPONENT 3: Predictions and Residuals
    # predictions, residuals: (B, S, N1, N2)
    # DTYPE: storage_dtype
    # NOTE: B dimension for parallel alpha processing
    # =========================================================================
    pred_res = MemoryComponent(name="Predictions & Residuals")
    pred_res.add(TensorSpec(
        name="predictions + residuals",
        shape=(B, S, N1, N2),
        shape_formula="(B, S, N1, N2)",
        dtype=storage_dtype,
        count=2,
    ))
    breakdown.add_component(pred_res)

    # AGD peak is not component-sum. During train_batch_alphas(), W update and X
    # update keep dense (B,S,N1,N2) tensors live around matmul, residual,
    # masking, gradient construction, and float casts. Local GB calibration
    # showed that the effective dense workspace grows with the dense tensor
    # size: N=5240 needs ~3.16 dense copies, while the historical N=8176 profile
    # needs ~6.1.  This is modeled as a stage coefficient, not a post-hoc
    # calibration factor, so the target-size solver sees the corrected formula.
    param_gb = _tensor_gb((B, S, N1 + N2, M), storage_dtype, count=4)
    mask_gb = _tensor_gb((B, N1, N2), DType.BOOL)
    dense_one_gb = _tensor_gb((B, S, N1, N2), storage_dtype)
    if dense_one_gb < 1.6:
        dense_workspace_copies = 3.1
    elif 2.10 <= dense_one_gb <= 2.30:
        # cuBLAS/autocast workspace has a local cliff around the generated
        # N≈5900-6200 target profiles on RTX 5090. Model it as a discrete stage bin
        # so the planner does not treat it as a smooth low-memory region.
        dense_workspace_copies = 6.12
    elif dense_one_gb < 2.2442:
        dense_workspace_copies = 3.1641 + 2.3429 * (dense_one_gb - 1.6366)
    elif dense_one_gb < 3.50:
        dense_workspace_copies = 4.59
    elif dense_one_gb < 4.0:
        dense_workspace_copies = 4.59 + (6.12 - 4.59) * (dense_one_gb - 3.50) / 0.50
    else:
        dense_workspace_copies = 6.12
    dense_workspace_copies = min(6.2, max(3.1, dense_workspace_copies))
    stages = {
        "agd_persistent_state": param_gb + mask_gb,
        "agd_w_update_dense_peak": param_gb + mask_gb + dense_workspace_copies * dense_one_gb,
        "agd_x_update_dense_peak": param_gb + mask_gb + dense_workspace_copies * dense_one_gb,
        "agd_return_retention": param_gb + mask_gb + 2.0 * dense_one_gb,
    }
    return _stage_override(breakdown, stages, safety_margin=1.0)


@MemoryEstimator.register("bigamp_tensor")
@MemoryEstimator.register("bigamp_tensor_parallel")
def estimate_tensor_spreading(params: EstimationParams) -> float:
    """
    N-dimensional tensor spreading memory estimation.
    
    Works for both serial (bigamp_tensor) and parallel (bigamp_tensor_parallel) versions.
    Parallel version processes all S samples simultaneously, so memory scales with S.
    """
    breakdown = get_tensor_spreading_breakdown(params)
    return breakdown.total_gb


def get_tensor_spreading_breakdown(params: EstimationParams) -> MemoryBreakdown:
    """
    Get modular memory breakdown for tensor spreading.

    This mirrors ``estimate_tensor_spreading`` and only exposes the existing
    formula as component metadata. It does not change the estimator total.
    """
    N1, N2, M, S = params.N1, params.N2, params.M, params.S
    B = params.batch_size
    alpha_max = params.alpha_max

    tensor_order = getattr(params, 'tensor_order', 3)
    tensor_dims = getattr(params, 'tensor_dims', None) or tuple([N1] * tensor_order)
    
    # Estimate hyperedges: C = ceil(alpha * sum(N_d) * M) (degrees of freedom scaling)
    import math
    # Use DOF scaling: C = alpha * sum(N_d) * M
    dof = sum(tensor_dims) * M
    C = max(1, int(math.ceil(alpha_max * dof)))
    
    # Storage dtype
    storage_bytes = 2 if params.use_bf16 else 4  # BF16 or FP32
    f_bytes = 1 if params.f_distribution == 'rademacher' else 4
    
    # Factor tensors: n factors × (S, N, M) for parallel version
    factor_memory = S * sum(tensor_dims) * M * storage_bytes
    
    # F and Y: (S, C, M) and (S, C)
    fy_memory = S * C * M * f_bytes + S * C * storage_bytes
    
    # Intermediate: Var matrices × n factors
    var_memory = S * sum(tensor_dims) * M * storage_bytes
    
    # Scatter/gather temporaries: (S, C, M)
    temp_memory = 3 * S * C * M * storage_bytes

    breakdown = MemoryBreakdown(
        algorithm_key=params.algorithm_key,
        safety_margin=1.2,
    )
    factors = MemoryComponent(name="Tensor Factors")
    factors.add(TensorSpec(
        name="factor tensors",
        shape=(S, sum(tensor_dims), M),
        shape_formula="S × sum(N_d) × M",
        dtype=DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32,
    ))
    breakdown.add_component(factors)

    observations = MemoryComponent(name="Tensor Observations")
    observations.add(TensorSpec(
        name="F coefficients",
        shape=(S, C, M),
        shape_formula="S × C × M",
        dtype=DType.INT8 if params.f_distribution == 'rademacher' else DType.FLOAT32,
    ))
    observations.add(TensorSpec(
        name="Y values",
        shape=(S, C),
        shape_formula="S × C",
        dtype=DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32,
    ))
    breakdown.add_component(observations)

    variances = MemoryComponent(name="Tensor Variances")
    variances.add(TensorSpec(
        name="factor variances",
        shape=(S, sum(tensor_dims), M),
        shape_formula="S × sum(N_d) × M",
        dtype=DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32,
    ))
    breakdown.add_component(variances)

    temporaries = MemoryComponent(name="Tensor Scatter/Gather Temporaries")
    temporaries.add(TensorSpec(
        name="scatter/gather temporaries",
        shape=(S, C, M),
        shape_formula="3 × S × C × M",
        dtype=DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32,
        count=3,
    ))
    breakdown.add_component(temporaries)

    if params.algorithm_key == "bigamp_tensor_parallel":
        max_dim = max(tensor_dims)
        # TensorSuperGraph has allocator/probe workspace bins that are not
        # captured by the pure edge tensor count. Local profiles show a small
        # floor near N=760, a larger workspace around N=1960/3272, and no
        # visible extra term by the historical N=5120 profile.
        if max_dim <= 760:
            tensor_supergraph_workspace_gb = 0.74
        elif max_dim <= 1960:
            tensor_supergraph_workspace_gb = 0.74 + (1.92 - 0.74) * (max_dim - 760) / (1960 - 760)
        elif max_dim <= 3272:
            tensor_supergraph_workspace_gb = 1.92 + (3.24 - 1.92) * (max_dim - 1960) / (3272 - 1960)
        elif max_dim < 5120:
            tensor_supergraph_workspace_gb = 3.24 * (5120 - max_dim) / (5120 - 3272)
        else:
            tensor_supergraph_workspace_gb = 0.0
        factor_state_gb = _tensor_gb((B, S * sum(tensor_dims), M), DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32, count=2)
        observation_gb = (
            _tensor_gb((S * C, M), DType.INT8 if params.f_distribution == 'rademacher' else DType.FLOAT32)
            + _tensor_gb((S * C,), DType.FLOAT32)
            + _tensor_gb((S * C,), DType.INT64, count=tensor_order)
            + _tensor_gb((B, S * C), DType.BOOL)
        )
        edge_workspace_gb = _tensor_gb((B, S * C, M), DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32)
        scatter_state_gb = _tensor_gb((B, S * max(tensor_dims), M), DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32)
        stages = {
            "tensor_parallel_persistent_state": factor_state_gb + observation_gb,
            # Mirrors probe_tensor_super_memory(): gathered_list, stacked gathered,
            # product/Z/s/r_contrib, and per-dimension scatter buffers can overlap.
            "tensor_parallel_supergraph_peak": (
                factor_state_gb + observation_gb + 5.0 * edge_workspace_gb + tensor_supergraph_workspace_gb
            ),
            "tensor_parallel_scatter_peak": factor_state_gb + observation_gb + edge_workspace_gb + tensor_order * scatter_state_gb,
        }
        return _stage_override(breakdown, stages, safety_margin=1.0)

    factor_state_gb = _tensor_gb((S, sum(tensor_dims), M), DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32, count=2)
    observation_gb = _tensor_gb((S, C, M), DType.INT8 if params.f_distribution == 'rademacher' else DType.FLOAT32) + _tensor_gb((S, C), DType.FLOAT32)
    edge_workspace_gb = _tensor_gb((S, C, M), DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32)
    stages = {
        "tensor_serial_persistent_state": factor_state_gb + observation_gb,
        "tensor_serial_alpha_sample_peak": factor_state_gb + observation_gb + 4.2 * edge_workspace_gb,
        "tensor_serial_result_retention": factor_state_gb + observation_gb + 2.0 * edge_workspace_gb,
    }
    return _stage_override(breakdown, stages, safety_margin=1.0)


def get_bigamp_spreading_breakdown(params: EstimationParams) -> MemoryBreakdown:
    """
    Get modular memory breakdown for non-parallel spreading BiG-AMP.
    
    Sequential processing:
    - Processes one sample at a time
    - Memory ~O(C × M) where C = alpha × M × N1
    """
    N1, N2, M, S = params.N1, params.N2, params.M, params.S
    alpha_max = params.alpha_max
    
    # Edges per sample
    C = max(1, int(alpha_max * M * N1))
    
    storage_dtype = DType.BFLOAT16 if params.use_bf16 else DType.FLOAT32
    f_dtype = DType.INT8 if params.f_distribution == 'rademacher' else DType.FLOAT32
    
    breakdown = MemoryBreakdown(algorithm_key="bigamp_spreading", safety_margin=1.30)
    
    # =========================================================================
    # COMPONENT 1: Per-Sample Parameters
    # w_hat, x_hat, w_var, x_var: 4 × (N, M)
    # DTYPE: storage_dtype
    # NOTE: Only 1 sample processed at a time
    # =========================================================================
    sample_params = MemoryComponent(name="Per-Sample Parameters")
    sample_params.add(TensorSpec(
        name="w_hat + x_hat + w_var + x_var",
        shape=(N1 + N2, M),
        shape_formula="4 × (N, M)",
        dtype=storage_dtype,
        count=4,
        notes="One sample at a time"
    ))
    breakdown.add_component(sample_params)
    
    # =========================================================================
    # COMPONENT 2: Sparse Structure (per-sample)
    # F: (C, M), Y: (C,), indices
    # DTYPE: F depends on distribution, Y is float32
    # =========================================================================
    sparse = MemoryComponent(name="Sparse Structure")
    sparse.add(TensorSpec(
        name="F_matrix",
        shape=(C, M),
        shape_formula="(C, M)",
        dtype=f_dtype,
        notes="F depends on f_distribution"
    ))
    sparse.add(TensorSpec(
        name="Y_values",
        shape=(C,),
        shape_formula="(C,)",
        dtype=DType.FLOAT32,
    ))
    sparse.add(TensorSpec(
        name="indices",
        shape=(C,),
        shape_formula="(C,)",
        dtype=DType.INT64,
        count=2,
    ))
    breakdown.add_component(sparse)
    
    # =========================================================================
    # COMPONENT 3: Per-Sample Compute
    # W_sel, X_sel, intermediates: ~6 × (C, M)
    # DTYPE: storage_dtype
    # =========================================================================
    compute = MemoryComponent(name="Per-Sample Compute")
    compute.add(TensorSpec(
        name="W_sel + X_sel + intermediates",
        shape=(C, M),
        shape_formula="(C, M)",
        dtype=storage_dtype,
        count=6,
        notes="Edge-level intermediate tensors"
    ))
    breakdown.add_component(compute)
    
    return breakdown


# =============================================================================
# Utility Functions
# =============================================================================

def get_registered_algorithms() -> List[str]:
    """Get list of registered algorithm keys."""
    return list(MemoryEstimator._estimators.keys())


def is_algorithm_registered(algo_key: str) -> bool:
    """Check if an algorithm is registered."""
    return algo_key in MemoryEstimator._estimators
