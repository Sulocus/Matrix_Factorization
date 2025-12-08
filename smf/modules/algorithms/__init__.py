"""
Training algorithms module.

Available algorithms:
- bigamp: BiG-AMP (Bilinear Generalized Approximate Message Passing)
- bigamp_spreading: BiG-AMP with random spreading for disordered model
- bigamp_spreading_parallel: BiG-AMP spreading with Super-Graph parallelization
- agd: AGD (Alternating Gradient Descent)
- combined: Flexible algorithm selection
"""

from .base import AlgorithmBase
from .bigamp import BiGAMPAlgorithm
from .bigamp_spreading import BiGAMPSpreadingAlgorithm
from .bigamp_spreading_parallel import BiGAMPSpreadingParallel
from .agd import AGDAlgorithm
from .combined import CombinedAlgorithm

__all__ = [
    'AlgorithmBase',
    'BiGAMPAlgorithm',
    'BiGAMPSpreadingAlgorithm',
    'BiGAMPSpreadingParallel',
    'AGDAlgorithm',
    'CombinedAlgorithm',
]
