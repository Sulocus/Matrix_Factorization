"""
Training algorithms module.

Available algorithms:
- bigamp: BiG-AMP (Bilinear Generalized Approximate Message Passing)
- bigamp_spreading: BiG-AMP with random spreading (Super-Graph parallelization)
- bigamp_tensor: BiG-AMP for N-dimensional tensor CP decomposition
- agd: AGD (Alternating Gradient Descent)
- combined: Flexible algorithm selection
"""

from .base import AlgorithmBase
from .bigamp import BiGAMPAlgorithm
from .bigamp.spreading import BiGAMPSpreading
from .bigamp.tensor_spreading import BiGAMPTensorSpreading
from .agd import AGDAlgorithm
from .agd_tensor import TensorAGD
from .combined import CombinedAlgorithm

__all__ = [
    'AlgorithmBase',
    'BiGAMPAlgorithm',
    'BiGAMPSpreading',
    'BiGAMPTensorSpreading',
    'AGDAlgorithm',
    'TensorAGD',
    'CombinedAlgorithm',
]

