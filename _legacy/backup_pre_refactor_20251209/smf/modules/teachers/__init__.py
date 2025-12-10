"""
Teacher model initialization module.

Available methods:
- standard: Standard Gaussian N(0, 1/sqrt(M))
- scaled_variance: Adjustable variance N(0, k/sqrt(M))
- orthogonal: QR-decomposed orthonormal columns/rows
- combined: Flexible combination of features
- random_spreading: Disordered model with quenched F ~ N(0,1)
"""

from .base import TeacherBase
from .standard import StandardTeacher
from .scaled_variance import ScaledVarianceTeacher
from .orthogonal import OrthogonalTeacher
from .combined import CombinedTeacher
from .random_spreading import (
    RandomSpreadingTeacher,
    SpreadingData,
    generate_spreading_coefficients,
    compute_sparse_Y,
    compute_sparse_Y_batched,
)

__all__ = [
    'TeacherBase',
    'StandardTeacher',
    'ScaledVarianceTeacher',
    'OrthogonalTeacher',
    'CombinedTeacher',
    'RandomSpreadingTeacher',
    'SpreadingData',
    'generate_spreading_coefficients',
    'compute_sparse_Y',
    'compute_sparse_Y_batched',
]
