"""
Experiment Framework for SMF.

This module provides a unified framework for controlling experiments,
separating experiment logic from algorithm computation.

Key components:
- ExperimentConfig: All parameters in one place
- DataFactory: Create Teacher, Mask, SpreadingData
- ExperimentRunner: Scan strategies, batching, execution
- ExperimentResult: Unified result storage

Usage:
    from smf.core.experiment import ExperimentConfig, ExperimentRunner
    
    config = ExperimentConfig(
        matrix=MatrixParams(N1=600, N2=600, M=150),
        training=TrainingParams(samples_per_alpha=20, max_steps=5000),
        algorithm_key='bigamp_spreading',
        scan=ScanConfig(dimension='alpha', values=[0.0, 0.5, 1.0, 1.5, 2.0]),
    )
    
    runner = ExperimentRunner()
    result = runner.run(config)
    result.save('results/my_experiment/')
"""

from .config import (
    ExperimentConfig,
    MatrixParams,
    TrainingParams,
    SeedConfig,
    ScanConfig,
    SpreadingConfig,
    AlgorithmParams,
    TeacherConfig,
)

from .result import (
    ExperimentResult,
    SingleRunResult,
    Checkpoint,
)

from .data_factory import DataFactory

from .runner import ExperimentRunner

__all__ = [
    # Config
    'ExperimentConfig',
    'MatrixParams',
    'TrainingParams',
    'SeedConfig',
    'ScanConfig',
    'SpreadingConfig',
    'AlgorithmParams',
    'TeacherConfig',
    # Result
    'ExperimentResult',
    'SingleRunResult',
    'Checkpoint',
    # Factory
    'DataFactory',
    # Runner
    'ExperimentRunner',
]
