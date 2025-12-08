"""Core analysis modules for Phase Analyzer"""

from .bigamp_trainer import (
    BiGAMPTrainer,
    TrainingConfig,
    TrainingResult,
    EpochScanner,
    train_and_get_Q_Y
)
from .gradient_adaptive_sampler import (
    GradientAdaptiveSampler,
    AdaptiveSamplingResult,
    smart_redistribute,
    compare_sampling_quality
)
from .precise_phase_analyzer import (
    PrecisePhaseAnalyzer,
    PhaseTransitionResult,
    PreciseAnalysisResult,
    run_precise_analysis
)

__all__ = [
    # BiG-AMP Training
    'BiGAMPTrainer',
    'TrainingConfig',
    'TrainingResult',
    'EpochScanner',
    'train_and_get_Q_Y',
    # Gradient-Adaptive Sampling (Mode 2)
    'GradientAdaptiveSampler',
    'AdaptiveSamplingResult',
    'smart_redistribute',
    'compare_sampling_quality',
    # Precise Phase Analysis (Mode 3)
    'PrecisePhaseAnalyzer',
    'PhaseTransitionResult',
    'PreciseAnalysisResult',
    'run_precise_analysis',
]
