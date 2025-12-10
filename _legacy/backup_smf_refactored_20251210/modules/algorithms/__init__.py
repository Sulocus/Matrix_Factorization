"""
Training algorithms module.

Available algorithms:
- bigamp: BiG-AMP (Bilinear Generalized Approximate Message Passing)
- bigamp_spreading: BiG-AMP with random spreading for disordered model
- bigamp_spreading_parallel: BiG-AMP spreading with Super-Graph parallelization
"""

from .base import AlgorithmBase
from .unified import BiGAMPUnified

__all__ = [
    'AlgorithmBase',
    'BiGAMPUnified',
]
