"""
Specialized experiment modules.

Contains pre-configured experiment types for common research tasks.
"""

from .init_scale import InitScaleExperiment, run_init_scale
from .large_matrix_sweep import SizeScalingExperiment, run_size_scaling

# Backwards compatibility aliases
VarianceSweepExperiment = InitScaleExperiment
run_variance_sweep = run_init_scale
LargeMatrixSweepExperiment = SizeScalingExperiment
run_large_matrix_sweep = run_size_scaling

__all__ = [
    'InitScaleExperiment', 'run_init_scale',
    'SizeScalingExperiment', 'run_size_scaling',
    # Legacy aliases
    'VarianceSweepExperiment', 'run_variance_sweep',
    'LargeMatrixSweepExperiment', 'run_large_matrix_sweep',
]
