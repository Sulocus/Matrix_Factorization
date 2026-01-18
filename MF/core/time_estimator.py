"""
Time estimation based on historical runs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

from ..modules.outputs.storage import list_results, ResultStorage


class TimeEstimator:
    """
    Estimates experiment runtime based on historical data.
    """

    # Default time per step per element (in seconds)
    # Calibrated from empirical data
    DEFAULT_TIME_PER_STEP_PER_ELEMENT = 1e-9  # very rough estimate

    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("smf/results")
        self._history_cache: List[Dict] = None

    def _load_history(self) -> List[Dict]:
        """Load historical run data."""
        if self._history_cache is not None:
            return self._history_cache

        history = []
        results = list_results(self.results_dir)

        for r in results:
            try:
                data = ResultStorage.load(r['path'])
                metadata = data.get('metadata', {})
                config = data.get('config')

                if not metadata.get('total_time') or not config:
                    continue

                # Handle both Config objects and dicts
                if hasattr(config, 'matrix'):
                    matrix = config.matrix
                    training = config.training
                    N1, N2, M = matrix.N1, matrix.N2, matrix.M
                    steps = training.max_steps
                    samples = training.samples_per_alpha
                else:
                    matrix = config.get('matrix', {})
                    training = config.get('training', {})
                    N1 = matrix.get('N1', 0)
                    N2 = matrix.get('N2', 0)
                    M = matrix.get('M', 0)
                    steps = training.get('max_steps', 0)
                    samples = training.get('samples_per_alpha', 1)

                if N1 and N2 and M and steps:
                    history.append({
                        'N1': N1,
                        'N2': N2,
                        'M': M,
                        'steps': steps,
                        'samples': samples,
                        'total_time': metadata['total_time'],
                        'elements': N1 * N2 * samples,
                    })
            except Exception:
                continue

        self._history_cache = history
        return history

    def estimate(
        self,
        N1: int,
        N2: int,
        M: int,
        steps: int,
        samples: int = 1,
        num_alphas: int = 41,
    ) -> Dict[str, Any]:
        """
        Estimate runtime for given parameters.

        Args:
            N1, N2: Matrix dimensions
            M: Latent dimension
            steps: Training steps
            samples: Samples per alpha
            num_alphas: Number of alpha values

        Returns:
            Dictionary with estimation details
        """
        history = self._load_history()

        # Target complexity
        target_elements = N1 * N2 * samples
        target_ops = target_elements * steps * num_alphas

        if history:
            # Find similar runs and extrapolate
            estimates = []
            for h in history:
                h_ops = h['elements'] * h['steps']
                scale = target_ops / (h_ops * 41)  # assume 41 alphas in history
                estimated_time = h['total_time'] * scale
                estimates.append(estimated_time)

            # Use median of estimates
            estimates.sort()
            median_idx = len(estimates) // 2
            estimated_seconds = estimates[median_idx]

            return {
                'estimated_seconds': estimated_seconds,
                'estimated_minutes': estimated_seconds / 60,
                'confidence': 'medium' if len(history) >= 3 else 'low',
                'based_on': len(history),
                'formatted': self._format_time(estimated_seconds),
            }
        else:
            # Fallback: rough estimate based on default rate
            estimated_seconds = target_ops * self.DEFAULT_TIME_PER_STEP_PER_ELEMENT
            # Add overhead
            estimated_seconds *= 2.0  # safety factor

            return {
                'estimated_seconds': estimated_seconds,
                'estimated_minutes': estimated_seconds / 60,
                'confidence': 'very_low',
                'based_on': 0,
                'formatted': self._format_time(estimated_seconds),
            }

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def print_estimate(
        self,
        N1: int,
        N2: int,
        M: int,
        steps: int,
        samples: int = 1,
        num_alphas: int = 41,
    ):
        """Print formatted time estimate."""
        est = self.estimate(N1, N2, M, steps, samples, num_alphas)

        print(f"\nEstimated runtime: {est['formatted']}")
        print(f"  Confidence: {est['confidence']}")
        if est['based_on'] > 0:
            print(f"  Based on {est['based_on']} historical runs")


# Global instance
_estimator: Optional[TimeEstimator] = None


def get_time_estimator() -> TimeEstimator:
    """Get or create global time estimator."""
    global _estimator
    if _estimator is None:
        _estimator = TimeEstimator()
    return _estimator


def estimate_time(N1: int, N2: int, M: int, steps: int, **kwargs) -> Dict[str, Any]:
    """Convenience function for time estimation."""
    return get_time_estimator().estimate(N1, N2, M, steps, **kwargs)
