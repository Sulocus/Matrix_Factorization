"""
Training algorithms module.

Available algorithms:
- bigamp: BiG-AMP (Bilinear Generalized Approximate Message Passing)
- bigamp_spreading: BiG-AMP with random spreading (Super-Graph parallelization)
- agd: AGD (Alternating Gradient Descent)
- combined: Flexible algorithm selection
"""

from .base import AlgorithmBase
from .bigamp import BiGAMPAlgorithm
from .bigamp_spreading import BiGAMPSpreading
from .bigamp_spreading_parallel_unit import BiGAMPSpreadingParallelUnit  # Unit scaling version
from .agd import AGDAlgorithm
from .combined import CombinedAlgorithm

__all__ = [
    'AlgorithmBase',
    'BiGAMPAlgorithm',
    'BiGAMPSpreading',
    'BiGAMPSpreadingParallelUnit',
    'AGDAlgorithm',
    'CombinedAlgorithm',
]
