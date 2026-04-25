"""
Experiment Configuration Data Structures.

All experiment parameters are defined here, enabling:
1. Complete reproducibility (all params saved)
2. Easy modification (one place to change)
3. Type safety (dataclasses with validation)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Any, Optional, Dict
from enum import Enum
import json
from pathlib import Path


class ScanDimension(Enum):
    """Supported scan dimensions."""
    ALPHA = "alpha"           # Observation density
    STEPS = "steps"           # Training steps (convergence curve)
    SAMPLES = "samples"       # Number of samples per alpha
    MATRIX_SIZE = "N"         # Matrix dimension scaling
    SEED = "seed"             # Replica analysis


@dataclass
class MatrixParams:
    """Matrix structure parameters."""
    N1: int  # Number of rows
    N2: int  # Number of columns  
    M: int   # Latent dimension (rank)
    
    def __post_init__(self):
        assert self.N1 > 0, "N1 must be positive"
        assert self.N2 > 0, "N2 must be positive"
        assert self.M > 0, "M must be positive"
    
    @property
    def is_square(self) -> bool:
        return self.N1 == self.N2
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingParams:
    """Training parameters."""
    samples_per_alpha: int = 20  # S: number of samples
    max_steps: int = 5000        # For BiGAMP algorithms
    max_epochs: int = 20000      # For AGD algorithm
    num_workers: int = 1         # Parallel workers for graph generation (CPU)
    
    def __post_init__(self):
        assert self.samples_per_alpha > 0, "samples_per_alpha must be positive"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.num_workers >= 1, "num_workers must be at least 1"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SeedConfig:
    """Random seed configuration for reproducibility."""
    base_seed: int = 42           # Graph structure seed
    teacher_seed: int = 12345     # Teacher matrix seed
    spreading_seed: int = 99999   # F matrix seed (spreading only)
    student_seed: int = 0         # Student initialization seed (0 = use base_seed)
    
    def get_student_seed(self) -> int:
        """Get effective student seed."""
        return self.student_seed if self.student_seed != 0 else self.base_seed
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ScanConfig:
    """Scan configuration for parameter sweeps."""
    dimension: str  # "alpha" / "steps" / "samples" / "N" / "seed"
    values: List[Any]  # Values to scan
    
    def __post_init__(self):
        # Validate dimension
        valid_dims = [d.value for d in ScanDimension]
        if self.dimension not in valid_dims:
            raise ValueError(
                f"Invalid scan dimension: {self.dimension}. "
                f"Valid options: {valid_dims}"
            )
        assert len(self.values) > 0, "values must not be empty"
    
    @property
    def num_points(self) -> int:
        """Number of scan points."""
        return len(self.values)
    
    @property
    def is_alpha_scan(self) -> bool:
        return self.dimension == ScanDimension.ALPHA.value
    
    @property
    def is_steps_scan(self) -> bool:
        return self.dimension == ScanDimension.STEPS.value
    
    def to_dict(self) -> Dict:
        return {"dimension": self.dimension, "values": self.values}


@dataclass
class SpreadingConfig:
    """Configuration specific to spreading algorithms."""
    f_distribution: str = "rademacher"  # "rademacher" or "gaussian"
    onsager_correction: bool = False  # Enable Onsager correction for Z update
    allow_intra_connection: bool = False  # Allow W-W and X-X connections (general graph)
    seed: int = 12345  # Spreading-specific random seed
    chunk_size: int = 131072  # Edges per chunk for General mode memory optimization (0 = disable)
    tensor_order: int = 2  # N-dimensional tensor order (2=matrix, 3+=tensor)
    
    def __post_init__(self):
        valid = ["rademacher", "gaussian"]
        if self.f_distribution not in valid:
            raise ValueError(
                f"Invalid f_distribution: {self.f_distribution}. "
                f"Valid options: {valid}"
            )
    
    @property
    def f_bytes_per_element(self) -> int:
        """Bytes per F element."""
        return 1 if self.f_distribution == "rademacher" else 4
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TeacherConfig:
    """Configuration specific to teacher model initialization."""
    init_distribution: str = "gaussian"  # "gaussian" or "rademacher"
    
    def __post_init__(self):
        valid = ["gaussian", "rademacher"]
        if self.init_distribution not in valid:
            raise ValueError(
                f"Invalid init_distribution: {self.init_distribution}. "
                f"Valid options: {valid}"
            )
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AlgorithmParams:
    """
    Algorithm-specific parameters.
    
    Different algorithms use different subsets:
    - AGD: learning_rate
    - BiGAMP: damping, noise_var
    - BiGAMPSpreading: damping, noise_var, f_distribution
    """
    # Common
    damping: float = 0.5
    noise_var: float = 1e-10
    
    # AGD specific
    learning_rate: float = 1e-2
    
    # Optimization
    use_compile: bool = True
    compile_fallback_policy: str = "allow"  # "allow" keeps eager fallback; "error" fails if compile fails
    use_bf16: bool = True
    dtype_fallback_policy: str = "allow"  # "allow" keeps FP32 fallback; "error" fails if BF16 is unavailable
    use_tf32: bool = True
    
    # Early stop (AGD)
    use_early_stop: bool = False
    target_loss_threshold: float = 1e-8
    
    # Step scanning: fixed alpha value when scanning steps
    default_alpha: float = 1.0

    # Adaptive Damping
    adaptive_damping: bool = False
    step_min: float = 0.05
    step_max: float = 0.5
    step_incr: float = 1.05
    step_decr: float = 0.5
    step_window: int = 5
    max_bad_steps: int = 10
    
    # Adaptive Warm Restart
    adaptive_restart: bool = False
    restart_patience: int = 50
    restart_noise: float = 0.1
    
    # Adaptive Tolerance
    acceptance_tolerance: float = 0.0  # Metropolis-like relaxation
    
    # Teacher-Assisted Initialization (Hysteresis Analysis)
    init_mode: str = "random"      # "random" = Cold Start, "teacher" = Warm Start
    init_overlap: float = 0.95     # Initial overlap with teacher (0.9 - 0.99)
    
    # Debugging
    debug_verbose: bool = False    # Enable expensive per-step metrics logging
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    All parameters in one place for:
    1. Full reproducibility
    2. Easy modification
    3. Unified saving/loading
    
    Usage:
        config = ExperimentConfig(
            matrix=MatrixParams(N1=600, N2=600, M=150),
            training=TrainingParams(samples_per_alpha=20, max_steps=5000),
            algorithm_key='bigamp_spreading',
            scan=ScanConfig(dimension='alpha', values=[0.0, 0.5, 1.0, 1.5, 2.0]),
        )
    """
    # Required
    matrix: MatrixParams
    training: TrainingParams
    algorithm_key: str  # "agd" / "bigamp" / "bigamp_spreading"
    scan: ScanConfig
    
    # Optional with defaults
    seeds: SeedConfig = field(default_factory=SeedConfig)
    algorithm_params: AlgorithmParams = field(default_factory=AlgorithmParams)
    spreading: Optional[SpreadingConfig] = None  # Only for spreading algorithms
    teacher: Optional[TeacherConfig] = None  # Teacher initialization config
    
    # Metadata
    experiment_name: str = "unnamed_experiment"
    teacher_key: str = "standard"  # "standard" or "orthogonal"
    notes: str = ""
    
    def __post_init__(self):
        # Validate algorithm key
        valid_algos = ["agd", "bigamp", "bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel"]
        if self.algorithm_key not in valid_algos:
            raise ValueError(
                f"Invalid algorithm_key: {self.algorithm_key}. "
                f"Valid options: {valid_algos}"
            )
        
        # Auto-create spreading config if needed
        if self.algorithm_key in ("bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel") and self.spreading is None:
            self.spreading = SpreadingConfig()
    
    @property
    def is_spreading_algorithm(self) -> bool:
        return "spreading" in self.algorithm_key
    
    @property
    def N1(self) -> int:
        return self.matrix.N1
    
    @property
    def N2(self) -> int:
        return self.matrix.N2
    
    @property
    def M(self) -> int:
        return self.matrix.M
    
    @property
    def S(self) -> int:
        return self.training.samples_per_alpha
    
    @property
    def alpha_values(self) -> List[float]:
        """Get alpha values if scanning alpha, else return default."""
        if self.scan.is_alpha_scan:
            return [float(v) for v in self.scan.values]
        else:
            # For non-alpha scans, use a single default alpha
            return [1.0]
    
    @property
    def max_alpha(self) -> float:
        """Maximum alpha value."""
        return max(self.alpha_values)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            "matrix": self.matrix.to_dict(),
            "training": self.training.to_dict(),
            "algorithm_key": self.algorithm_key,
            "scan": self.scan.to_dict(),
            "seeds": self.seeds.to_dict(),
            "algorithm_params": self.algorithm_params.to_dict(),
            "spreading": self.spreading.to_dict() if self.spreading else None,
            "teacher": self.teacher.to_dict() if self.teacher else None,
            "experiment_name": self.experiment_name,
            "teacher_key": self.teacher_key,
            "notes": self.notes,
        }
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            matrix=MatrixParams(**data["matrix"]),
            training=TrainingParams(**data["training"]),
            algorithm_key=data["algorithm_key"],
            scan=ScanConfig(**data["scan"]),
            seeds=SeedConfig(**data["seeds"]),
            algorithm_params=AlgorithmParams(**data["algorithm_params"]),
            spreading=SpreadingConfig(**data["spreading"]) if data.get("spreading") else None,
            teacher=TeacherConfig(**data["teacher"]) if data.get("teacher") else None,
            experiment_name=data.get("experiment_name", "unnamed"),
            teacher_key=data.get("teacher_key", "standard"),
            notes=data.get("notes", ""),
        )
    
    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(\n"
            f"  name='{self.experiment_name}',\n"
            f"  algorithm='{self.algorithm_key}',\n"
            f"  matrix={self.N1}x{self.N2}, M={self.M},\n"
            f"  scan={self.scan.dimension}: {len(self.scan.values)} points,\n"
            f")"
        )
