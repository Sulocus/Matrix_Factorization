"""
Complete Checkpoint Manager for SMF experiments.

Uses torch.save to store:
- Full config (for resume)
- Completed alphas list
- Results (metrics only, no tensors for performance)

Default path: runs/.checkpoint.pt
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Default fallback checkpoint path. CLI runs should pass a run-scoped path.
CHECKPOINT_PATH = Path("runs/.checkpoint.pt")


@dataclass
class CheckpointData:
    """Complete checkpoint data structure."""
    version: int = 2  # Version 2: added raw_yaml
    config_dict: Dict[str, Any] = field(default_factory=dict)
    completed_alphas: List[float] = field(default_factory=list)
    results: Dict[float, Dict[str, Any]] = field(default_factory=dict)  # alpha -> metrics
    timestamp: str = ""
    # Output options (rsb_ordering, save_tensors, uniform_colormap)
    output_options: Dict[str, Any] = field(default_factory=lambda: {
        'rsb_ordering': False,
        'save_tensors': True,
        'uniform_colormap': False,
    })
    # Raw YAML string for complete config restoration
    raw_yaml: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'version': self.version,
            'config_dict': self.config_dict,
            'completed_alphas': self.completed_alphas,
            'results': self.results,
            'timestamp': self.timestamp,
            'output_options': self.output_options,
            'raw_yaml': self.raw_yaml,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointData':
        return cls(
            version=data.get('version', 1),
            config_dict=data.get('config_dict', {}),
            completed_alphas=data.get('completed_alphas', []),
            results=data.get('results', {}),
            timestamp=data.get('timestamp', ''),
            output_options=data.get('output_options', {
                'rsb_ordering': False,
                'save_tensors': True,
                'uniform_colormap': False,
            }),
            raw_yaml=data.get('raw_yaml', ''),
        )


class CheckpointManager:
    """
    Complete checkpoint manager using torch.save.
    
    Usage:
        # Normal run: delete old checkpoint
        ckpt = CheckpointManager()
        ckpt.delete()
        
        # Save after each batch
        ckpt.save(config_dict, completed_alphas, results)
        
        # Resume: load and continue
        data = ckpt.load()
        if data:
            # Use data.config_dict, data.results, etc.
    """
    
    def __init__(self, path: Path = CHECKPOINT_PATH):
        self.path = Path(path)
        self._data: Optional[CheckpointData] = None
        # Async IO Worker
        # Max workers = 1 ensures sequential writes (no race on file access)
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def save(
        self,
        config_dict: Dict[str, Any],
        completed_alphas: List[float],
        results: Dict[float, Dict[str, Any]],
        output_options: Dict[str, Any] = None,
        raw_yaml: str = "",
    ) -> None: # Returns None because it's async
        """
        Async save checkpoint (non-blocking).
        
        Performs a shallow copy of mutable containers in the main thread,
        then serializes and writes to disk in a background thread.
        This prevents UI/Computation lag during large checkpoints.
        
        Args:
            config_dict: Experiment configuration
            completed_alphas: List of completed alpha values
            results: Dict of alpha -> metrics
            output_options: Output options (rsb_ordering, save_tensors, etc.)
            raw_yaml: Complete original YAML config string
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default output options
        if output_options is None:
            output_options = {
                'rsb_ordering': False,
                'save_tensors': True,
                'uniform_colormap': False,
            }
        
        # 1. Create Data Object
        # Note: We must COPY mutable containers (list, dict) to prevent
        # "dictionary changed size during iteration" or race conditions
        # while the background thread is pickling.
        payload = {
            'version': 2,  # Version 2: includes raw_yaml
            'config_dict': config_dict, # Assume config is immutable
            'completed_alphas': list(completed_alphas), # Copy list
            'results': results.copy(), # Shallow copy dict (inner metrics usually immutable)
            'timestamp': datetime.now().isoformat(),
            'output_options': output_options.copy(),  # Save output options!
            'raw_yaml': raw_yaml,  # Complete original YAML
        }
        
        # 2. Submit to background thread
        self._executor.submit(self._atomic_save, payload, self.path)
        
        # Returns immediately!
        logger.debug(f"Checkpoint save queued: {len(completed_alphas)} alphas")

    @staticmethod
    def _atomic_save(payload: Dict, path: Path):
        """
        Worker function: Atomic write (save to tmp -> rename).
        Running in background thread.
        """
        try:
            # Write to .tmp file first
            tmp_path = path.with_suffix('.pt.tmp')
            torch.save(payload, tmp_path)
            
            # Atomic rename (overwrites target if exists)
            # This ensures we never have a corrupted file at 'path'
            tmp_path.replace(path)
            
            # logger.debug(f"Async checkpoint complete: {path}")
        except Exception as e:
            logger.error(f"Async checkpoint failed: {e}")
    
    def load(self) -> Optional[CheckpointData]:
        """
        Load checkpoint if exists.
        
        Returns:
            CheckpointData or None
        """
        if not self.path.exists():
            return None
        
        try:
            data_dict = torch.load(self.path, weights_only=False)
            self._data = CheckpointData.from_dict(data_dict)
            logger.info(f"Checkpoint loaded: {len(self._data.completed_alphas)} alphas")
            return self._data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def delete(self):
        """Delete checkpoint file."""
        if self.path.exists():
            self.path.unlink()
            logger.info("Checkpoint deleted")
        self._data = None
    
    def exists(self) -> bool:
        """Check if checkpoint exists."""
        return self.path.exists()
    
    @property
    def data(self) -> Optional[CheckpointData]:
        """Get loaded checkpoint data."""
        return self._data


def config_to_dict(config) -> Dict[str, Any]:
    """
    Convert ExperimentConfig to serializable dict.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Dictionary with all config values
    """
    return {
        'matrix': {
            'N1': config.matrix.N1,
            'N2': config.matrix.N2,
            'M': config.matrix.M,
        },
        'training': {
            'samples_per_alpha': config.training.samples_per_alpha,
            'max_steps': config.training.max_steps,
        },
        'algorithm_key': config.algorithm_key,
        'teacher_key': getattr(config, 'teacher_key', 'standard'),
        'scan': {
            'dimension': config.scan.dimension,
            'values': [float(v) for v in config.scan.values],
        },
        'seeds': {
            'base_seed': config.seeds.base_seed,
        },
        'algorithm_params': {
            'damping': config.algorithm_params.damping,
            'noise_var': config.algorithm_params.noise_var,
            'use_compile': config.algorithm_params.use_compile,
        },
        'spreading': {
            'f_distribution': config.spreading.f_distribution if config.spreading else 'rademacher',
        } if config.spreading else None,
        'teacher': {
            'init_distribution': config.teacher.init_distribution,
        } if getattr(config, 'teacher', None) else None,
        'experiment_name': config.experiment_name,
    }


def dict_to_config(d: Dict[str, Any]):
    """
    Convert dict back to ExperimentConfig.
    
    Args:
        d: Dictionary from checkpoint
        
    Returns:
        ExperimentConfig object
    """
    from matrix_factorization.core.experiment.config import (
        ExperimentConfig, MatrixParams, TrainingParams, 
        ScanConfig, SeedConfig, AlgorithmParams, SpreadingConfig, TeacherConfig
    )
    
    matrix = MatrixParams(
        N1=d['matrix']['N1'],
        N2=d['matrix']['N2'],
        M=d['matrix']['M'],
    )
    
    training = TrainingParams(
        samples_per_alpha=d['training']['samples_per_alpha'],
        max_steps=d['training']['max_steps'],
    )
    
    scan = ScanConfig(
        dimension=d['scan']['dimension'],
        values=d['scan']['values'],
    )
    
    seeds = SeedConfig(
        base_seed=d['seeds']['base_seed'],
    )
    
    algo_params = AlgorithmParams(
        damping=d['algorithm_params']['damping'],
        noise_var=d['algorithm_params']['noise_var'],
        use_compile=d['algorithm_params']['use_compile'],
    )
    
    spreading = None
    if d.get('spreading'):
        spreading = SpreadingConfig(
            f_distribution=d['spreading']['f_distribution'],
        )
    
    # Teacher config (backward compatible)
    teacher = None
    if d.get('teacher'):
        teacher = TeacherConfig(
            init_distribution=d['teacher']['init_distribution'],
        )
    
    return ExperimentConfig(
        matrix=matrix,
        training=training,
        algorithm_key=d['algorithm_key'],
        scan=scan,
        seeds=seeds,
        algorithm_params=algo_params,
        spreading=spreading,
        teacher=teacher,
        experiment_name=d['experiment_name'],
        teacher_key=d.get('teacher_key', 'standard'),
    )
