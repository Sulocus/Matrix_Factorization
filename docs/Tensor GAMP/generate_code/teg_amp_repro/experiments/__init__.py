"""
Experiments module for TeG-AMP reproduction.
Contains scripts to reproduce Figures 2, 3, and 9 from the paper.
"""

from .run_synthetic import run_synthetic_experiment
from .run_mnist import run_mnist_experiment
from .run_snr_test import run_snr_experiment

__all__ = [
    'run_synthetic_experiment',
    'run_mnist_experiment',
    'run_snr_experiment'
]
