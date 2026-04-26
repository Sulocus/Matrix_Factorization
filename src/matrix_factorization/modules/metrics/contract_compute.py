"""Metric computation adapter for the hard-interface runner path.

The formulas in this module are lifted from ExperimentRunner._compute_metrics
without changing numerical definitions. The purpose is structural: runner
selects the data path, while the metrics package owns metric payload creation.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def compute_metric_payload(
    *,
    W_students: torch.Tensor,
    X_students: torch.Tensor,
    data: Any,
) -> Dict[str, float]:
    """Compute the legacy flat metric payload for a runner scan point."""
    if data.spreading_data is not None:
        return compute_spreading_metric_payload(W_students, X_students, data)
    return compute_matrix_metric_payload(W_students, X_students, data)


def compute_spreading_metric_payload(
    W_students: torch.Tensor,
    X_students: torch.Tensor,
    data: Any,
) -> Dict[str, float]:
    """Compute spreading metrics through the existing spreading metric code."""
    from .spreading import compute_all_metrics_spreading_parallel

    # Reshape W, X for spreading metrics: (A, S, N, M) -> (S, A, N, M)
    if W_students.dim() == 4:
        # Already (A, S, N1, M) or (S, A, N1, M)
        # compute_all_metrics_spreading_parallel expects (S, A, N1, M)
        if W_students.shape[0] != data.spreading_data.S:
            W_for_metrics = W_students.transpose(0, 1)
            X_for_metrics = X_students.transpose(0, 1)
        else:
            W_for_metrics = W_students
            X_for_metrics = X_students
    else:
        W_for_metrics = W_students
        X_for_metrics = X_students

    # Check for single-alpha slice (when W has 1 alpha but spreading_data has many)
    target_alpha_idx = None
    _, A_in_W = W_for_metrics.shape[:2]
    A_spreading = len(data.spreading_data.alpha_values)

    if A_in_W == 1 and A_spreading > 1 and data.alpha_values is not None and len(data.alpha_values) == 1:
        current_alpha = data.alpha_values[0]
        diffs = [abs(a - current_alpha) for a in data.spreading_data.alpha_values]
        best_idx = diffs.index(min(diffs))
        target_alpha_idx = int(best_idx)

    metrics_tensor = compute_all_metrics_spreading_parallel(
        W_for_metrics,
        X_for_metrics,
        data.spreading_data,
        target_alpha_idx=target_alpha_idx,
    )

    result = {}
    for key, val in metrics_tensor.items():
        if key == "alpha_values":
            continue

        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                result[key] = float(val.item())
            elif val.dim() == 1 and len(val) > 0:
                result[key] = float(val[0])
            else:
                result[key] = float(val.mean().item())
        else:
            result[key] = float(val)
    return result


def compute_matrix_metric_payload(
    W_students: torch.Tensor,
    X_students: torch.Tensor,
    data: Any,
) -> Dict[str, float]:
    """Compute matrix metrics through the existing overlap definitions."""
    from .overlap import (
        compute_cosine_similarity,
        gram_overlap_normalized,
        gram_overlap_root,
        projection_abs,
        _compute_qy_masked,
    )
    import math
    import numpy as np

    if W_students.dim() == 4:
        W_for_metrics = W_students.mean(dim=0)
        X_for_metrics = X_students.mean(dim=0)
    else:
        W_for_metrics = W_students
        X_for_metrics = X_students

    S = W_for_metrics.shape[0]

    # Formal projection metrics
    Q_W_list = []
    Q_X_list = []
    # Gauge/rotation-insensitive diagnostics
    Q_W_gram_root_list = []
    Q_X_gram_root_list = []
    # Y metrics
    Q_Y_list = []
    Q_Y_observed_list = []
    Q_Y_unobserved_list = []

    # data.Y_teacher is scaled by 1/sqrt(M), so multiply back to match W@X convention.
    Y_teacher = data.Y_teacher * math.sqrt(data.M)

    # (S, N1, M) @ (S, M, N2) -> (S, N1, N2)
    Y_students = torch.bmm(W_for_metrics, X_for_metrics)

    for s in range(S):
        Q_W_list.append(projection_abs(W_for_metrics[s], data.W_teacher))
        Q_X_list.append(projection_abs(X_for_metrics[s], data.X_teacher))
        Q_W_gram_root_list.append(gram_overlap_root(W_for_metrics[s], data.W_teacher, use_left=True))
        Q_X_gram_root_list.append(gram_overlap_root(X_for_metrics[s], data.X_teacher, use_left=False))
        Q_Y_list.append(projection_abs(Y_students[s], Y_teacher))

        if data.masks is not None:
            if data.masks.dim() == 3:
                mask = data.masks[0]
            else:
                mask = data.masks

            Q_Y_observed_list.append(_compute_qy_masked(Y_students[s], Y_teacher, mask, observed=True))
            Q_Y_unobserved_list.append(_compute_qy_masked(Y_students[s], Y_teacher, mask, observed=False))

    # Replica metrics (Student-Student)
    Q_W_replica_list = []
    Q_X_replica_list = []
    Q_W_prime_replica_list = []
    Q_X_prime_replica_list = []

    if S >= 2:
        for i in range(S):
            for j in range(i + 1, S):
                Q_W_replica_list.append(compute_cosine_similarity(W_for_metrics[i], W_for_metrics[j], use_left=True))
                Q_X_replica_list.append(compute_cosine_similarity(X_for_metrics[i], X_for_metrics[j], use_left=False))
                Q_W_prime_replica_list.append(gram_overlap_normalized(W_for_metrics[i], W_for_metrics[j], use_left=True))
                Q_X_prime_replica_list.append(gram_overlap_normalized(X_for_metrics[i], X_for_metrics[j], use_left=False))

    result = {
        "Q_W_mean": float(np.mean(Q_W_list)),
        "Q_W_std": float(np.std(Q_W_list, ddof=1)) if len(Q_W_list) > 1 else 0.0,
        "Q_X_mean": float(np.mean(Q_X_list)),
        "Q_X_std": float(np.std(Q_X_list, ddof=1)) if len(Q_X_list) > 1 else 0.0,
        "Q_Y_mean": float(np.mean(Q_Y_list)),
        "Q_Y_std": float(np.std(Q_Y_list, ddof=1)) if len(Q_Y_list) > 1 else 0.0,
        "Q_W_GRAM_ROOT_mean": float(np.mean(Q_W_gram_root_list)),
        "Q_W_GRAM_ROOT_std": float(np.std(Q_W_gram_root_list, ddof=1)) if len(Q_W_gram_root_list) > 1 else 0.0,
        "Q_X_GRAM_ROOT_mean": float(np.mean(Q_X_gram_root_list)),
        "Q_X_GRAM_ROOT_std": float(np.std(Q_X_gram_root_list, ddof=1)) if len(Q_X_gram_root_list) > 1 else 0.0,
        # Replica
        "Q_W_replica_mean": float(np.mean(Q_W_replica_list)) if Q_W_replica_list else 0.0,
        "Q_X_replica_mean": float(np.mean(Q_X_replica_list)) if Q_X_replica_list else 0.0,
        "Q_W_prime_replica_mean": float(np.mean(Q_W_prime_replica_list)) if Q_W_prime_replica_list else 0.0,
        "Q_X_prime_replica_mean": float(np.mean(Q_X_prime_replica_list)) if Q_X_prime_replica_list else 0.0,
    }

    # C类: Observed/Unobserved
    if Q_Y_observed_list:
        result["Q_Y_observed_mean"] = float(np.mean(Q_Y_observed_list))
        result["Q_Y_observed_std"] = float(np.std(Q_Y_observed_list, ddof=1)) if len(Q_Y_observed_list) > 1 else 0.0
    if Q_Y_unobserved_list:
        result["Q_Y_unobserved_mean"] = float(np.mean(Q_Y_unobserved_list))
        result["Q_Y_unobserved_std"] = float(np.std(Q_Y_unobserved_list, ddof=1)) if len(Q_Y_unobserved_list) > 1 else 0.0

    return result
