"""
Evaluation metrics module.

Available metrics:
- overlap: Q_Y, Q_W, Q_X overlap metrics
- gram: Gram matrix overlap (Q_W', Q_X')
- qy_unobserved: Q_Y computed only on unobserved positions
- spreading: Metrics for random spreading model
- combined: Flexible metric selection
"""

from .overlap import (
    compute_cosine_similarity,
    compute_physical_overlap,
    gram_overlap_normalized,
    compute_qy,
    compute_all_metrics,
    compute_generalization_error,
    compute_replica_overlap,
    aggregate_trial_metrics,
)

from .qy_unobserved import (
    compute_qy_unobserved,
    compute_qy_observed,
    compute_qy_split,
    compute_physical_overlap_unobserved,
    compute_physical_overlap_observed,
)

from .spreading import (
    compute_qy_spreading,
    compute_mse_spreading,
    compute_all_metrics_spreading,
    compute_physical_overlap_spreading,
    compute_qy_with_wrong_f,
)

from .combined import CombinedMetrics

# Replica overlap analysis (pairwise student-student overlap)
from .replica import (
    compute_replica_overlap as compute_replica_overlap_analysis,
    gram_overlap_cosine,
    gram_overlap_normalized as gram_overlap_baseline_corrected,
    analyze_replica_results,
)

__all__ = [
    'compute_cosine_similarity',
    'compute_physical_overlap',
    'gram_overlap_normalized',
    'compute_qy',
    'compute_all_metrics',
    'compute_generalization_error',
    'compute_replica_overlap',
    'aggregate_trial_metrics',
    'compute_qy_unobserved',
    'compute_qy_observed',
    'compute_qy_split',
    'compute_physical_overlap_unobserved',
    'compute_physical_overlap_observed',
    'compute_qy_spreading',
    'compute_mse_spreading',
    'compute_all_metrics_spreading',
    'compute_physical_overlap_spreading',
    'compute_qy_with_wrong_f',
    'CombinedMetrics',
]
