"""
Training algorithms module.

Available algorithms:
- bigamp: BiG-AMP (Bilinear Generalized Approximate Message Passing)
- bigamp_spreading: BiG-AMP with random spreading for disordered model
- agd: AGD (Alternating Gradient Descent)
- combined: Flexible algorithm selection
"""

from .base import AlgorithmBase
from .bigamp import BiGAMPAlgorithm
from .bigamp_spreading import BiGAMPSpreadingAlgorithm
from .agd import AGDAlgorithm
from .combined import CombinedAlgorithm

__all__ = [
    'AlgorithmBase',
    'BiGAMPAlgorithm',
    'BiGAMPSpreadingAlgorithm',
    'AGDAlgorithm',
    'CombinedAlgorithm',
]
