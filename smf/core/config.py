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


@dataclass
class ExecutionConfig:
    """
    LLM-generated execution parameters.

    Controls which metrics to compute and which plots to generate.
    This enables dynamic execution based on user's natural language requests.
    """
    # Metrics to compute during evaluation
    metrics_to_compute: List[str] = field(
        default_factory=lambda: ['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error']
    )

    # Plot configurations (each dict has: type, metrics, filename)
    plots: List[Dict[str, Any]] = field(default_factory=list)

    # Whether to generate the default summary plot
    include_summary_plot: bool = True

    # Whether to generate the default Q_Y-only plot
    include_qy_plot: bool = True


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
                ['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error']),
            plots=exec_data.get('plots', []),
            include_summary_plot=exec_data.get('include_summary_plot', True),
            include_qy_plot=exec_data.get('include_qy_plot', True),
        )

        return cls(
            matrix=MatrixConfig(**data.get('matrix', {})),
            alpha=AlphaConfig(**data.get('alpha', {})),
            training=TrainingConfig(**data.get('training', {})),
            algorithm=AlgorithmConfig(**data.get('algorithm', {})),
            algorithm_key=data.get('algorithm_key', 'bigamp'),
            graph_key=data.get('graph_key', 'random'),
            teacher_key=data.get('teacher_key', 'standard'),
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
