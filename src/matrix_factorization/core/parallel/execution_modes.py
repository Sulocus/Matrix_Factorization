"""
Execution Modes and Core Data Structures for Smart Parallel Module.

This module defines the parallel execution modes, memory estimation parameters,
and execution planning structures.
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


class ParallelMode(Enum):
    """Parallel execution modes for algorithms."""
    LINEAR = auto()           # Fully sequential: S=1, A=1
    SAMPLE_PARALLEL = auto()  # Sample parallel: S parallel, A sequential
    ALPHA_PARALLEL = auto()   # Alpha parallel: S sequential, A parallel
    HYBRID = auto()           # Hybrid: S parallel, A batched
    FULL_PARALLEL = auto()    # Full parallel: S×A all parallel


@dataclass
class EstimationParams:
    """
    Parameters for memory estimation.
    
    Attributes:
        N1, N2: Matrix dimensions
        M: Hidden dimension (rank)
        S: Number of samples per alpha
        alpha_values: List of alpha values to process
        algorithm_key: Algorithm identifier (e.g., 'bigamp_spreading_parallel')
        use_compile: Whether torch.compile is enabled
        use_bf16: Whether using BF16 precision
        f_distribution: For spreading algorithms, 'gaussian' or 'rademacher'
                       Gaussian uses float32 (4 bytes), Rademacher uses int8 (1 byte)
    """
    N1: int
    N2: int
    M: int
    S: int
    alpha_values: List[float]
    algorithm_key: str
    use_compile: bool = True
    use_bf16: bool = False
    f_distribution: str = 'rademacher'  # 'gaussian' or 'rademacher'
    adaptive_damping: bool = False  # Whether using adaptive damping (doubles memory for backtracking)
    allow_intra_connection: bool = False  # General Graph mode (W-W, X-X connections)
    tensor_order: int = 2
    tensor_dims: Optional[Tuple[int, ...]] = None
    seed_partition_policy: str = "legacy"
    
    @property
    def alpha_max(self) -> float:
        """Maximum alpha value in the list."""
        return max(self.alpha_values) if self.alpha_values else 0.0
    
    @property
    def batch_size(self) -> int:
        """Number of alpha values (batch size)."""
        return len(self.alpha_values)
    
    @property
    def f_bytes_per_element(self) -> int:
        """Bytes per F element based on distribution type."""
        return 1 if self.f_distribution == 'rademacher' else 4
    
    @property
    def is_spreading_algorithm(self) -> bool:
        """Check if this is a spreading-type algorithm."""
        return 'spreading' in self.algorithm_key.lower()


@dataclass
class MemoryEstimate:
    """Memory estimation result."""
    total_gb: float
    per_batch_gb: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.8  # Estimation confidence (0.0-1.0)
    
    def __repr__(self) -> str:
        return (f"MemoryEstimate(total={self.total_gb:.2f}GB, "
                f"per_batch={self.per_batch_gb:.2f}GB, "
                f"confidence={self.confidence:.0%})")


@dataclass
class BatchConfig:
    """Configuration for a single execution batch."""
    sample_range: Tuple[int, int]  # (start, end)
    alpha_range: Tuple[int, int]   # (start, end)
    estimated_memory_gb: float
    alpha_values: List[float] = field(default_factory=list)
    memory_breakdown: Dict[str, float] = field(default_factory=dict)
    
    @property
    def num_samples(self) -> int:
        return self.sample_range[1] - self.sample_range[0]
    
    @property
    def num_alphas(self) -> int:
        return self.alpha_range[1] - self.alpha_range[0]


@dataclass
class AllocationConfig:
    """
    Memory allocation configuration.
    
    Threshold system:
    - warning_threshold (85%): Normal allocation upper limit
    - critical_threshold (95%): Trigger batch recovery
    
    This allows precise estimation while maintaining a buffer
    for torch.compile dynamic caches and unexpected memory spikes.
    """
    # Core allocation parameters
    allocation_ratio: float = 0.55      # Conservative allocation ratio
    max_allocation_gb: Optional[float] = None  # Optional hard limit
    
    # Runtime monitoring thresholds
    warning_threshold: float = 0.85     # Normal allocation limit
    critical_threshold: float = 0.95    # Batch recovery trigger
    
    # Estimation parameters
    apply_calibration: bool = True      # Apply calibration factors
    safety_margin: float = 1.1          # Safety margin multiplier


class AllocationPresets:
    """
    Pre-configured allocation settings.
    
    Note: max_allocation_gb is set to None by default, letting the system
    use the actual GPU memory. Set it explicitly only if you want a hard limit.
    """
    
    # Conservative: 75% allocation (optimized for 32GB GPUs)
    CONSERVATIVE = AllocationConfig(
        allocation_ratio=0.75,
        max_allocation_gb=None,
        warning_threshold=0.85,
        critical_threshold=0.95,
        safety_margin=1.1
    )
    
    # Balanced: Moderate buffer (85% of available)
    BALANCED = AllocationConfig(
        allocation_ratio=0.85,
        max_allocation_gb=None,
        warning_threshold=0.90,
        critical_threshold=0.95,
        safety_margin=1.1
    )
    
    # Aggressive: Minimal buffer (use with caution)
    AGGRESSIVE = AllocationConfig(
        allocation_ratio=0.85,
        max_allocation_gb=None,
        warning_threshold=0.90,
        critical_threshold=0.95,
        safety_margin=1.05
    )
    
    @classmethod
    def get(cls, name: str) -> AllocationConfig:
        """Get preset by name."""
        presets = {
            'conservative': cls.CONSERVATIVE,
            'balanced': cls.BALANCED,
            'aggressive': cls.AGGRESSIVE,
        }
        if name.lower() not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        return presets[name.lower()]
    
    @classmethod
    def with_limit(cls, preset_name: str, max_gb: float) -> AllocationConfig:
        """Get preset with custom memory limit."""
        from dataclasses import replace
        base = cls.get(preset_name)
        return replace(base, max_allocation_gb=max_gb)


@dataclass
class ExecutionPlan:
    """Complete execution plan for parallel algorithms."""
    mode: ParallelMode
    batches: List[BatchConfig]
    total_estimated_memory_gb: float
    allocation_config: AllocationConfig
    
    # Metadata
    algorithm_key: str = ""
    gpu_model: str = ""
    available_memory_gb: float = 0.0
    seed_partition_policy: str = "legacy"
    replan_policy_key: str = ""
    automatic_rebatch_allowed: bool = False
    replan_implemented: bool = False
    
    @property
    def num_batches(self) -> int:
        return len(self.batches)
    
    @property
    def utilization_ratio(self) -> float:
        """Expected memory utilization ratio."""
        if self.available_memory_gb <= 0:
            return 0.0
        return self.total_estimated_memory_gb / self.available_memory_gb
    
    def summary(self) -> str:
        """Human-readable summary of the plan."""
        lines = [
            f"ExecutionPlan: {self.mode.name}",
            f"  Algorithm: {self.algorithm_key}",
            f"  GPU: {self.gpu_model}",
            f"  Batches: {self.num_batches}",
            f"  Memory: {self.total_estimated_memory_gb:.2f}GB / "
            f"{self.available_memory_gb:.2f}GB ({self.utilization_ratio:.0%})",
            f"  Allocation ratio: {self.allocation_config.allocation_ratio:.0%}",
        ]
        return "\n".join(lines)
