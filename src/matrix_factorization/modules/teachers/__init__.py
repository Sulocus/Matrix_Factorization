"""
Teacher model initialization module.

Available methods:
- standard: Standard Gaussian using algorithm_params.normalization_profile
- scaled_variance: Legacy adjustable variance N(0, k/sqrt(M))
- orthogonal: QR-decomposed orthonormal columns/rows
- combined: Flexible combination of features
- random_spreading: Disordered model with quenched F ~ N(0,1)
"""

from .base import TeacherBase
from .standard import StandardTeacher
from .scaled_variance import ScaledVarianceTeacher
from .orthogonal import OrthogonalTeacher
from .orthogonal_unit import OrthogonalUnitTeacher  # O(1) scaling version
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
    'OrthogonalUnitTeacher',
    'CombinedTeacher',
    'RandomSpreadingTeacher',
    'SpreadingData',
    'generate_spreading_coefficients',
    'compute_sparse_Y',
    'compute_sparse_Y_batched',
]
