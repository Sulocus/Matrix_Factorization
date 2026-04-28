"""
Evaluation metrics module.

Available metrics:
- overlap: Q_Y, Q_W, Q_X overlap metrics
- gram: Cos-root diagnostics (Q_W_COS_ROOT, Q_X_COS_ROOT)
- qy_unobserved: Q_Y computed only on unobserved positions
- spreading: Metrics for random spreading model
- combined: Flexible metric selection
"""

from .overlap import (
    compute_cosine_similarity,
    compute_physical_overlap,
    physical_overlap_fixed,
    student_self_overlap_fixed,
    sign_gauge_overlap_fixed,
    scale_gauge_overlaps_fixed,
    normalized_mse_and_fit,
    PROJECTION_NORM_EPS,
    projection_abs,
    projection_abs_diagnostics,
    sign_aligned_projection_abs,
    sign_gauge_projection_abs,
    gram_overlap_normalized,
    cos_overlap_root,
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
from .spec_adapter import MetricPayloadCheck, MetricSpecAdapter
from .contract_compute import (
    compute_metric_payload,
    compute_matrix_metric_payload,
    compute_spreading_metric_payload,
)

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
    'physical_overlap_fixed',
    'student_self_overlap_fixed',
    'sign_gauge_overlap_fixed',
    'scale_gauge_overlaps_fixed',
    'normalized_mse_and_fit',
    'PROJECTION_NORM_EPS',
    'projection_abs',
    'projection_abs_diagnostics',
    'sign_aligned_projection_abs',
    'sign_gauge_projection_abs',
    'gram_overlap_normalized',
    'cos_overlap_root',
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
    'MetricPayloadCheck',
    'MetricSpecAdapter',
    'compute_metric_payload',
    'compute_matrix_metric_payload',
    'compute_spreading_metric_payload',
]
