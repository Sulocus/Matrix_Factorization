"""
Experiment Result Data Structures.

Unified result storage for all experiment types, enabling:
1. Consistent format across scan modes
2. Complete reproducibility (config saved with results)
3. Easy analysis and visualization
"""

from dataclasses import dataclass, field, asdict
from typing import List, Any, Optional, Dict, Union
from pathlib import Path
from datetime import datetime
import json
import torch


@dataclass
class Checkpoint:
    """
    Algorithm state checkpoint for resuming or continuing.
    
    Used for:
    1. Scanning steps: continue from previous checkpoint
    2. Error recovery: resume from last good state
    """
    step: int
    W_state: torch.Tensor  # (S, N1, M) or (A, S, N1, M)
    X_state: torch.Tensor  # (S, M, N2) or (A, S, M, N2)
    
    # Optional variance states (BiGAMP)
    W_var: Optional[torch.Tensor] = None
    X_var: Optional[torch.Tensor] = None
    
    # Additional algorithm state
    extra: Optional[Dict[str, Any]] = None
    
    def save(self, path: Path):
        """Save checkpoint to file."""
        path = Path(path)
        torch.save({
            'step': self.step,
            'W_state': self.W_state,
            'X_state': self.X_state,
            'W_var': self.W_var,
            'X_var': self.X_var,
            'extra': self.extra,
        }, path)
    
    @classmethod
    def load(cls, path: Path, device: torch.device = None) -> 'Checkpoint':
        """Load checkpoint from file."""
        data = torch.load(path, map_location=device)
        return cls(
            step=data['step'],
            W_state=data['W_state'],
            X_state=data['X_state'],
            W_var=data.get('W_var'),
            X_var=data.get('X_var'),
            extra=data.get('extra'),
        )


@dataclass
class SingleRunResult:
    """
    Result from a single algorithm run.
    
    Contains metrics and optionally the full student matrices.
    """
    # Scan point identifier
    scan_value: Any  # e.g., alpha=1.5 or steps=1000
    
    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    # Expected keys: Q_W_mean, Q_W_std, Q_X_mean, Q_X_std, Q_Y_mean, Q_Y_std
    
    # Optional: full student matrices (can be large)
    W_students: Optional[torch.Tensor] = None  # (S, N1, M)
    X_students: Optional[torch.Tensor] = None  # (S, M, N2)
    
    # Optional: convergence history
    history: Optional[List[Dict[str, float]]] = None
    
    # Timing
    duration_seconds: float = 0.0
    
    def to_dict(self, include_tensors: bool = False) -> Dict:
        """Convert to dictionary."""
        d = {
            'scan_value': self.scan_value,
            'metrics': self.metrics,
            'duration_seconds': self.duration_seconds,
        }
        if self.history:
            d['history'] = self.history
        if include_tensors and self.W_students is not None:
            d['has_tensors'] = True
        return d
    
    @property
    def Q_W(self) -> float:
        """Teacher-Student overlap for W."""
        return self.metrics.get('Q_W_mean', 0.0)
    
    @property
    def Q_X(self) -> float:
        """Teacher-Student overlap for X."""
        return self.metrics.get('Q_X_mean', 0.0)
    
    @property
    def Q_Y(self) -> float:
        """Prediction overlap."""
        return self.metrics.get('Q_Y_mean', 0.0)


@dataclass
class ExperimentMetadata:
    """Metadata about the experiment run."""
    timestamp: str = ""
    duration_total_seconds: float = 0.0
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    smf_version: str = "1.0.0"
    python_version: str = ""
    torch_version: str = ""
    
    @classmethod
    def create_now(cls) -> 'ExperimentMetadata':
        """Create metadata with current system info."""
        import sys
        
        gpu_name = ""
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return cls(
            timestamp=datetime.now().isoformat(),
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory,
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
        )
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """
    Complete experiment result with all metadata.
    
    This is the unified result format for all experiment types.
    
    Usage:
        result = ExperimentResult(
            experiment_id="exp_001",
            config=config,
            scan_dimension="alpha",
            scan_values=[0.0, 0.5, 1.0],
        )
        result.add_result(0.0, single_result)
        result.add_result(0.5, single_result)
        result.save('results/exp_001/')
    """
    # Identification
    experiment_id: str
    
    # Full configuration (for reproducibility)
    config: Any  # ExperimentConfig (Any to avoid circular import)
    
    # Scan information
    scan_dimension: str
    scan_values: List[Any]
    
    # Results indexed by scan value
    results: Dict[Any, SingleRunResult] = field(default_factory=dict)
    
    # Metadata
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    
    def add_result(self, scan_value: Any, result: SingleRunResult):
        """Add a single run result."""
        self.results[scan_value] = result
    
    @property
    def num_completed(self) -> int:
        """Number of completed scan points."""
        return len(self.results)
    
    @property
    def is_complete(self) -> bool:
        """Check if all scan points are completed."""
        return self.num_completed == len(self.scan_values)
    
    @property
    def completion_ratio(self) -> float:
        """Completion ratio (0.0 to 1.0)."""
        return self.num_completed / len(self.scan_values) if self.scan_values else 0.0
    
    def get_metric_curve(self, metric_name: str) -> Dict[Any, float]:
        """Get a metric across all scan values."""
        return {
            v: r.metrics.get(metric_name, 0.0)
            for v, r in sorted(self.results.items())
        }
    
    def save(self, path: Union[str, Path], save_tensors: bool = False):
        """
        Save result to directory.
        
        Creates:
        - config.json: Full experiment configuration
        - metadata.json: Run metadata
        - results.json: Metrics for all scan points
        - tensors/ (optional): Student matrices if save_tensors=True
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        if hasattr(self.config, 'save'):
            self.config.save(path / 'config.json')
        else:
            with open(path / 'config.json', 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
        
        # Save metadata
        self.metadata.duration_total_seconds = sum(
            r.duration_seconds for r in self.results.values()
        )
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save results summary
        results_summary = {
            'experiment_id': self.experiment_id,
            'scan_dimension': self.scan_dimension,
            'scan_values': [str(v) for v in self.scan_values],
            'results': {
                str(v): r.to_dict(include_tensors=save_tensors)
                for v, r in self.results.items()
            },
        }
        with open(path / 'results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save tensors if requested
        if save_tensors:
            tensors_dir = path / 'tensors'
            tensors_dir.mkdir(exist_ok=True)
            for v, r in self.results.items():
                if r.W_students is not None and r.X_students is not None:
                    torch.save({
                        'W_students': r.W_students,
                        'X_students': r.X_students,
                    }, tensors_dir / f'scan_{v}.pt')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentResult':
        """Load result from directory."""
        path = Path(path)
        
        # Load config
        from .config import ExperimentConfig
        config = ExperimentConfig.load(path / 'config.json')
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata_dict = json.load(f)
        metadata = ExperimentMetadata(**metadata_dict)
        
        # Load results
        with open(path / 'results.json', 'r') as f:
            results_data = json.load(f)
        
        # Parse scan values back to original types
        scan_values = []
        for v_str in results_data['scan_values']:
            try:
                scan_values.append(float(v_str))
            except ValueError:
                scan_values.append(v_str)
        
        result = cls(
            experiment_id=results_data['experiment_id'],
            config=config,
            scan_dimension=results_data['scan_dimension'],
            scan_values=scan_values,
            metadata=metadata,
        )
        
        # Load individual results
        for v_str, r_dict in results_data['results'].items():
            try:
                v = float(v_str)
            except ValueError:
                v = v_str
            
            single_result = SingleRunResult(
                scan_value=v,
                metrics=r_dict.get('metrics', {}),
                duration_seconds=r_dict.get('duration_seconds', 0.0),
                history=r_dict.get('history'),
            )
            
            # Load tensors if they exist
            tensor_file = path / 'tensors' / f'scan_{v}.pt'
            if tensor_file.exists():
                tensors = torch.load(tensor_file)
                single_result.W_students = tensors['W_students']
                single_result.X_students = tensors['X_students']
            
            result.results[v] = single_result
        
        return result
    
    def save_unified(self, path: Union[str, Path]):
        """
        Save ALL data to a SINGLE .pt file.
        
        This is the recommended method for production use:
        - Single file, no scattered small files
        - Complete reproducibility (all config/metadata saved)
        - Efficient loading (torch.save/load)
        
        File structure inside the .pt:
        {
            'experiment_id': str,
            'config': dict,
            'metadata': dict,
            'scan_dimension': str,
            'scan_values': list,
            'results': {
                scan_value: {
                    'metrics': dict,
                    'duration_seconds': float,
                    'history': list or None,
                    'W_students': tensor or None,
                    'X_students': tensor or None,
                }
            }
        }
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update total duration
        self.metadata.duration_total_seconds = sum(
            r.duration_seconds for r in self.results.values()
        )
        
        # Build unified data structure
        data = {
            'experiment_id': self.experiment_id,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config,
            'metadata': self.metadata.to_dict(),
            'scan_dimension': self.scan_dimension,
            'scan_values': self.scan_values,
            'results': {},
        }
        
        # Add all results (including tensors)
        for v, r in self.results.items():
            data['results'][v] = {
                'metrics': r.metrics,
                'duration_seconds': r.duration_seconds,
                'history': r.history,
                'W_students': r.W_students,  # Can be None
                'X_students': r.X_students,  # Can be None
            }
        
        # Save as single file
        torch.save(data, path)
        
        if hasattr(self, '_verbose') and self._verbose:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"Saved experiment to {path} ({size_mb:.1f} MB)")
    
    @classmethod
    def load_unified(cls, path: Union[str, Path], device: torch.device = None) -> 'ExperimentResult':
        """
        Load from a single .pt file.
        
        Args:
            path: Path to .pt file
            device: Target device for tensors (default: keep original)
            
        Returns:
            ExperimentResult with all data
        """
        path = Path(path)
        data = torch.load(path, map_location=device, weights_only=False)
        
        # Reconstruct config
        from .config import ExperimentConfig, MatrixParams, TrainingParams, ScanConfig, SeedConfig, AlgorithmParams, SpreadingConfig
        
        config_dict = data['config']
        config = ExperimentConfig(
            matrix=MatrixParams(**config_dict['matrix']),
            training=TrainingParams(**config_dict['training']),
            algorithm_key=config_dict['algorithm_key'],
            scan=ScanConfig(**config_dict['scan']),
            seeds=SeedConfig(**config_dict['seeds']),
            algorithm_params=AlgorithmParams(**config_dict['algorithm_params']),
            spreading=SpreadingConfig(**config_dict['spreading']) if config_dict.get('spreading') else None,
            experiment_name=config_dict.get('experiment_name', 'unnamed'),
            teacher_key=config_dict.get('teacher_key', 'standard'),
            notes=config_dict.get('notes', ''),
        )
        
        # Reconstruct metadata
        metadata = ExperimentMetadata(**data['metadata'])
        
        # Create result object
        result = cls(
            experiment_id=data['experiment_id'],
            config=config,
            scan_dimension=data['scan_dimension'],
            scan_values=data['scan_values'],
            metadata=metadata,
        )
        
        # Load individual results
        for v, r_dict in data['results'].items():
            single_result = SingleRunResult(
                scan_value=v,
                metrics=r_dict.get('metrics', {}),
                duration_seconds=r_dict.get('duration_seconds', 0.0),
                history=r_dict.get('history'),
                W_students=r_dict.get('W_students'),
                X_students=r_dict.get('X_students'),
            )
            result.results[v] = single_result
        
        return result
    
    def __repr__(self) -> str:
        return (
            f"ExperimentResult(\n"
            f"  id='{self.experiment_id}',\n"
            f"  scan={self.scan_dimension}: {len(self.scan_values)} points,\n"
            f"  completed={self.num_completed}/{len(self.scan_values)},\n"
            f")"
        )
