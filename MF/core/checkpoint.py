"""
Checkpoint management for experiment resumption.

Allows long experiments to be interrupted and resumed from the last saved state.
Checkpoints are saved every N alphas and cleaned up after successful completion.
"""

import pickle
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List


@dataclass
class Checkpoint:
    """Checkpoint data structure."""
    config_hash: str               # Hash of config for consistency validation
    completed_alphas: List[float]  # Alpha values that have been processed
    results: Dict[str, Any]        # {alpha: metrics}
    teacher_seed: int              # Seed to recreate teacher model
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'config_hash': self.config_hash,
            'completed_alphas': self.completed_alphas,
            'results': self.results,
            'teacher_seed': self.teacher_seed,
            'timestamp': self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            config_hash=data['config_hash'],
            completed_alphas=data['completed_alphas'],
            results=data['results'],
            teacher_seed=data['teacher_seed'],
            timestamp=datetime.fromisoformat(data['timestamp']),
        )


class CheckpointManager:
    """
    Manages checkpoint creation, loading, and cleanup.

    Usage:
        mgr = CheckpointManager(output_dir, config)

        # Check for existing checkpoint
        if resume:
            remaining = mgr.get_remaining_alphas(all_alphas)
            checkpoint = mgr.load_latest()
            results = checkpoint.results.copy() if checkpoint else {}

        # During training
        for i, alpha in enumerate(remaining):
            results[alpha] = train(alpha)
            if (i + 1) % 10 == 0:
                mgr.save(list(results.keys()), results)

        # Cleanup after success
        mgr.cleanup()
    """

    def __init__(self, output_dir: Path, config: Any, save_interval: int = 10):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory for checkpoints
            config: Experiment configuration
            save_interval: Save checkpoint every N alphas
        """
        self.checkpoint_dir = Path(output_dir) / ".checkpoints"
        self.config = config
        self.config_hash = self._hash_config(config)
        self.save_interval = save_interval

    def _hash_config(self, config: Any) -> str:
        """Create a hash of the config for consistency checking."""
        # Extract key fields that affect results
        key_fields = {
            'N1': getattr(config.matrix, 'N1', None),
            'N2': getattr(config.matrix, 'N2', None),
            'M': getattr(config.matrix, 'M', None),
            'alpha_start': getattr(config.alpha, 'start', None),
            'alpha_stop': getattr(config.alpha, 'stop', None),
            'alpha_step': getattr(config.alpha, 'step', None),
            'max_steps': getattr(config.training, 'max_steps', None),
            'algorithm': getattr(config, 'algorithm_key', None),
            'graph': getattr(config, 'graph_key', None),
        }
        config_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def save(self, completed_alphas: List[float], results: Dict[str, Any]) -> Path:
        """
        Save a checkpoint.

        Args:
            completed_alphas: List of completed alpha values
            results: Dictionary of {alpha: metrics}

        Returns:
            Path to saved checkpoint
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = Checkpoint(
            config_hash=self.config_hash,
            completed_alphas=completed_alphas,
            results=results,
            teacher_seed=getattr(self.config.training, 'seed', 42),
        )

        # Use number of completed alphas in filename
        filename = f"checkpoint_{len(completed_alphas):04d}.pkl"
        path = self.checkpoint_dir / filename

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        return path

    def load_latest(self) -> Optional[Checkpoint]:
        """
        Load the most recent valid checkpoint.

        Returns:
            Checkpoint if found and valid, None otherwise
        """
        if not self.checkpoint_dir.exists():
            return None

        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return None

        # Try loading from newest to oldest
        for ckpt_path in reversed(checkpoints):
            try:
                with open(ckpt_path, 'rb') as f:
                    checkpoint = pickle.load(f)

                # Validate config hash
                if checkpoint.config_hash != self.config_hash:
                    print(f"[Checkpoint] Config mismatch, ignoring {ckpt_path.name}")
                    continue

                return checkpoint

            except Exception as e:
                print(f"[Checkpoint] Failed to load {ckpt_path.name}: {e}")
                continue

        return None

    def get_remaining_alphas(self, all_alphas: List[float]) -> List[float]:
        """
        Get alpha values that haven't been processed yet.

        Args:
            all_alphas: Complete list of alpha values

        Returns:
            List of remaining alpha values to process
        """
        checkpoint = self.load_latest()
        if not checkpoint:
            return list(all_alphas)

        completed = set(checkpoint.completed_alphas)
        return [a for a in all_alphas if float(a) not in completed]

    def should_save(self, num_completed: int) -> bool:
        """Check if it's time to save a checkpoint."""
        return num_completed > 0 and num_completed % self.save_interval == 0

    def cleanup(self):
        """Remove all checkpoints after successful completion."""
        if not self.checkpoint_dir.exists():
            return

        for f in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                f.unlink()
            except Exception:
                pass

        # Remove directory if empty
        try:
            self.checkpoint_dir.rmdir()
        except Exception:
            pass

    def get_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about existing checkpoint without loading full data."""
        checkpoint = self.load_latest()
        if not checkpoint:
            return None

        return {
            'num_completed': len(checkpoint.completed_alphas),
            'last_alpha': checkpoint.completed_alphas[-1] if checkpoint.completed_alphas else None,
            'timestamp': checkpoint.timestamp,
            'config_hash': checkpoint.config_hash,
        }
