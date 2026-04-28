"""
Configuration management for experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any, List, Dict
from pathlib import Path
import yaml
import json
import hashlib


@dataclass
class MatrixConfig:
    """Matrix dimension configuration."""
    N1: int = 200
    N2: int = 200
    M: int = 50


@dataclass
class AlphaConfig:
    """Alpha sweep configuration."""
    start: float = 0.0
    stop: float = 4.0
    step: float = 0.1

    def get_values(self) -> list[float]:
        """Generate list of alpha values."""
        import numpy as np
        return list(np.arange(self.start, self.stop + self.step / 2, self.step))


@dataclass
class TrainingConfig:
    """Training parameters."""
    max_steps: int = 5000      # For BiG-AMP
    max_epochs: int = 20000    # For AGD
    samples_per_alpha: int = 1
    seed: int = 42
    resample_mask: bool = True


@dataclass
class SpreadingConfig:
    """Random spreading specific configuration."""
    f_distribution: str = "gaussian"  # gaussian | ising
    seed: int = 12345                 # Seed for F generation
    teacher_type: str = "standard"    # standard | orthogonal (for W/X generation)
    allow_intra_connection: bool = False  # Allow W-W and X-X connections (general graph)


@dataclass
class AlgorithmConfig:
    """Algorithm-specific parameters."""
    # BiG-AMP
    damping: float = 0.5
    noise_var: float = 1e-10
    # AGD
    learning_rate: float = 0.01
    # Common
    early_stop: bool = False
    convergence_threshold: float = 1e-6
    # Acceleration
    use_compile: bool = True  # Enable torch.compile for kernel fusion
    normalization_profile: str = "paper_sparse_sampling"  # paper_sparse_sampling | internal_normalized_legacy
    # Onsager correction (for BiGAMP Spreading)
    onsager_enabled: bool = True  # Enable Onsager correction for AMP de-correlation


@dataclass
class ExecutionConfig:
    """
    LLM-generated execution parameters.

    Controls which metrics to compute and which plots to generate.
    This enables dynamic execution based on user's natural language requests.
    """
    # Metrics to compute during evaluation
    metrics_to_compute: List[str] = field(
        default_factory=lambda: [
            'Q_Y',
            'FIT_Y',
            'Q_W',
            'Q_X',
            'Q_W_SIGN_GAUGE',
            'Q_X_SIGN_GAUGE',
            'Q_W_COS_ROOT',
            'Q_X_COS_ROOT',
        ]
    )

    # Plot configurations (each dict has: type, metrics, filename)
    plots: List[Dict[str, Any]] = field(default_factory=list)

    # Whether to generate the default summary plot
    include_summary_plot: bool = True

    # Whether to generate the default Q_Y-only plot
    include_qy_plot: bool = True

    # Matrix metric for replica analysis ("gram_overlap_normalized" | "projection_abs" | "cosine_similarity")
    matrix_metric: str = "gram_overlap_normalized"

    # Heatmap configuration
    enable_heatmap: bool = False
    rsb_ordering: bool = False
    heatmap_metric: str = "Q_Y" # "Q_W_SIGN_GAUGE" for factor heatmap or "Q_Y" for tensor fit


@dataclass
class Config:
    """Complete experiment configuration."""
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    # Module selections
    algorithm_key: str = "bigamp"
    graph_key: str = "random"
    teacher_key: str = "standard"

    # Algorithm-specific configurations
    spreading: Optional[SpreadingConfig] = None  # Only used when algorithm_key="bigamp_spreading*"

    # LLM-generated execution parameters
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Create config from dictionary."""
        # Handle execution config
        exec_data = data.get('execution', {})
        execution = ExecutionConfig(
            metrics_to_compute=exec_data.get('metrics_to_compute',
                ['Q_Y', 'FIT_Y', 'Q_W', 'Q_X', 'Q_W_SIGN_GAUGE', 'Q_X_SIGN_GAUGE', 'Q_W_COS_ROOT', 'Q_X_COS_ROOT']),
            plots=exec_data.get('plots', []),
            include_summary_plot=exec_data.get('include_summary_plot', True),
            include_qy_plot=exec_data.get('include_qy_plot', True),
            matrix_metric=exec_data.get('matrix_metric', 'gram_overlap_normalized'),
            enable_heatmap=exec_data.get('enable_heatmap', False),
            rsb_ordering=exec_data.get('rsb_ordering', False),
            heatmap_metric=exec_data.get('heatmap_metric', "Q_Y"),
        )

        # Handle spreading config (only for bigamp_spreading algorithms)
        spreading_data = data.get('spreading')
        spreading = SpreadingConfig(**spreading_data) if spreading_data else None

        return cls(
            matrix=MatrixConfig(**data.get('matrix', {})),
            alpha=AlphaConfig(**data.get('alpha', {})),
            training=TrainingConfig(**data.get('training', {})),
            algorithm=AlgorithmConfig(**data.get('algorithm', {})),
            algorithm_key=data.get('algorithm_key', 'bigamp'),
            graph_key=data.get('graph_key', 'random'),
            teacher_key=data.get('teacher_key', 'standard'),
            spreading=spreading,
            execution=execution,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Path) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_hash(self) -> str:
        """Generate a short hash of the config for unique identification."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_display_name(self) -> str:
        """Generate human-readable name for this config."""
        m = self.matrix
        return f"{m.N1}x{m.N2}_M{m.M}_{self.algorithm_key}_{self.graph_key}"
