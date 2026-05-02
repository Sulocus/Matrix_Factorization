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
    target_alpha_indices = None
    _, A_in_W = W_for_metrics.shape[:2]
    A_spreading = len(data.spreading_data.alpha_values)

    if A_in_W == 1 and A_spreading > 1 and data.alpha_values is not None and len(data.alpha_values) == 1:
        current_alpha = data.alpha_values[0]
        diffs = [abs(float(a.item()) - float(current_alpha)) for a in data.spreading_data.alpha_values]
        best_idx = diffs.index(min(diffs))
        target_alpha_idx = int(best_idx)
    elif (
        A_spreading > A_in_W
        and data.alpha_values is not None
        and len(data.alpha_values) == A_in_W
    ):
        target_alpha_indices = []
        used = set()
        data_alphas = [float(a.item()) for a in data.spreading_data.alpha_values]
        for current_alpha in data.alpha_values:
            diffs = [
                abs(alpha_value - float(current_alpha)) if idx not in used else float("inf")
                for idx, alpha_value in enumerate(data_alphas)
            ]
            best_idx = diffs.index(min(diffs))
            used.add(best_idx)
            target_alpha_indices.append(int(best_idx))

    metrics_tensor = compute_all_metrics_spreading_parallel(
        W_for_metrics,
        X_for_metrics,
        data.spreading_data,
        target_alpha_idx=target_alpha_idx,
        target_alpha_indices=target_alpha_indices,
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
        cos_overlap_root,
        normalized_mse_and_fit,
        physical_overlap_fixed,
        projection_abs,
        scale_gauge_overlaps_fixed,
        sign_gauge_overlap_fixed,
        student_self_overlap_fixed,
        _compute_qy_masked,
        _compute_nmse_fit_masked,
    )
    import math
    import numpy as np

    def signed_cosine(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-12) -> float:
        student_flat = student.flatten().float()
        teacher_flat = teacher.flatten().float()
        if student_flat.numel() == 0 or teacher_flat.numel() == 0:
            return 0.0
        denom = student_flat.norm() * teacher_flat.norm()
        if float(denom.abs().item()) < eps:
            return 0.0
        value = (student_flat * teacher_flat).sum() / (denom + eps)
        if not torch.isfinite(value):
            return 0.0
        return float(value)

    def signed_cosine_masked(
        student: torch.Tensor,
        teacher: torch.Tensor,
        mask: torch.Tensor,
        *,
        observed: bool,
    ) -> float:
        if mask.dim() == 3:
            mask = mask[0]
        selection_mask = mask > 0.5 if observed else mask < 0.5
        return signed_cosine(student[selection_mask], teacher[selection_mask])

    if W_students.dim() == 4:
        W_for_metrics = W_students.mean(dim=0)
        X_for_metrics = X_students.mean(dim=0)
    else:
        W_for_metrics = W_students
        X_for_metrics = X_students

    S = W_for_metrics.shape[0]

    # Formal physical-overlap metrics
    Q_W_list = []
    Q_X_list = []
    R_W_list = []
    R_X_list = []
    # Legacy projection diagnostics
    Q_W_proj_abs_list = []
    Q_X_proj_abs_list = []
    Q_Y_proj_abs_list = []
    # Per-channel sign-gauge diagnostics
    Q_W_sign_gauge_list = []
    Q_X_sign_gauge_list = []
    # Gauge/rotation-insensitive diagnostics
    Q_W_cos_root_list = []
    Q_X_cos_root_list = []
    # Y metrics
    Q_Y_list = []
    Q_Y_COS_list = []
    NMSE_Y_list = []
    FIT_Y_list = []
    Q_Y_observed_list = []
    Q_Y_unobserved_list = []
    Q_Y_observed_COS_list = []
    Q_Y_unobserved_COS_list = []
    NMSE_Y_observed_list = []
    NMSE_Y_unobserved_list = []
    FIT_Y_observed_list = []
    FIT_Y_unobserved_list = []
    Q_Y_observed_proj_abs_list = []
    Q_Y_unobserved_proj_abs_list = []
    Q_W_scale_gauge_list = []
    Q_X_scale_gauge_list = []
    Q_WX_scale_gauge_list = []
    median_abs_log_k_list = []

    # data.Y_teacher is scaled by 1/sqrt(M), so multiply back to match W@X convention.
    Y_teacher = data.Y_teacher * math.sqrt(data.M)

    # (S, N1, M) @ (S, M, N2) -> (S, N1, N2)
    Y_students = torch.bmm(W_for_metrics, X_for_metrics)

    for s in range(S):
        Q_W_list.append(physical_overlap_fixed(W_for_metrics[s], data.W_teacher))
        Q_X_list.append(physical_overlap_fixed(X_for_metrics[s], data.X_teacher))
        R_W_list.append(student_self_overlap_fixed(W_for_metrics[s]))
        R_X_list.append(student_self_overlap_fixed(X_for_metrics[s]))
        Q_W_proj_abs_list.append(projection_abs(W_for_metrics[s], data.W_teacher))
        Q_X_proj_abs_list.append(projection_abs(X_for_metrics[s], data.X_teacher))
        Q_W_sign_gauge_list.append(sign_gauge_overlap_fixed(W_for_metrics[s], data.W_teacher, latent_axis=-1))
        Q_X_sign_gauge_list.append(sign_gauge_overlap_fixed(X_for_metrics[s], data.X_teacher, latent_axis=0))
        Q_W_cos_root_list.append(cos_overlap_root(W_for_metrics[s], data.W_teacher, use_left=True))
        Q_X_cos_root_list.append(cos_overlap_root(X_for_metrics[s], data.X_teacher, use_left=False))
        scale_metrics = scale_gauge_overlaps_fixed(
            W_for_metrics[s],
            X_for_metrics[s],
            data.W_teacher,
            data.X_teacher,
        )
        Q_W_scale_gauge_list.append(scale_metrics["Q_W_SCALE_GAUGE"])
        Q_X_scale_gauge_list.append(scale_metrics["Q_X_SCALE_GAUGE"])
        Q_WX_scale_gauge_list.append(scale_metrics["Q_WX_SCALE_GAUGE"])
        median_abs_log_k_list.append(scale_metrics["median_abs_log_k"])
        nmse_y, fit_y = normalized_mse_and_fit(Y_students[s], Y_teacher)
        NMSE_Y_list.append(nmse_y)
        Q_Y_list.append(projection_abs(Y_students[s], Y_teacher))
        Q_Y_COS_list.append(signed_cosine(Y_students[s], Y_teacher))
        Q_Y_proj_abs_list.append(Q_Y_list[-1])
        FIT_Y_list.append(fit_y)

        if data.masks is not None:
            if data.masks.dim() == 3:
                mask = data.masks[0]
            else:
                mask = data.masks

            nmse_obs, fit_obs = _compute_nmse_fit_masked(Y_students[s], Y_teacher, mask, observed=True)
            nmse_unobs, fit_unobs = _compute_nmse_fit_masked(Y_students[s], Y_teacher, mask, observed=False)
            NMSE_Y_observed_list.append(nmse_obs)
            NMSE_Y_unobserved_list.append(nmse_unobs)
            FIT_Y_observed_list.append(fit_obs)
            FIT_Y_unobserved_list.append(fit_unobs)
            Q_Y_observed_proj_abs_list.append(_compute_qy_masked(Y_students[s], Y_teacher, mask, observed=True))
            Q_Y_unobserved_proj_abs_list.append(_compute_qy_masked(Y_students[s], Y_teacher, mask, observed=False))
            Q_Y_observed_list.append(Q_Y_observed_proj_abs_list[-1])
            Q_Y_unobserved_list.append(Q_Y_unobserved_proj_abs_list[-1])
            Q_Y_observed_COS_list.append(signed_cosine_masked(Y_students[s], Y_teacher, mask, observed=True))
            Q_Y_unobserved_COS_list.append(signed_cosine_masked(Y_students[s], Y_teacher, mask, observed=False))

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
        "R_W_mean": float(np.mean(R_W_list)),
        "R_W_std": float(np.std(R_W_list, ddof=1)) if len(R_W_list) > 1 else 0.0,
        "R_X_mean": float(np.mean(R_X_list)),
        "R_X_std": float(np.std(R_X_list, ddof=1)) if len(R_X_list) > 1 else 0.0,
        "Q_W_PROJ_ABS_mean": float(np.mean(Q_W_proj_abs_list)),
        "Q_W_PROJ_ABS_std": float(np.std(Q_W_proj_abs_list, ddof=1)) if len(Q_W_proj_abs_list) > 1 else 0.0,
        "Q_X_PROJ_ABS_mean": float(np.mean(Q_X_proj_abs_list)),
        "Q_X_PROJ_ABS_std": float(np.std(Q_X_proj_abs_list, ddof=1)) if len(Q_X_proj_abs_list) > 1 else 0.0,
        "Q_W_SIGN_GAUGE_mean": float(np.mean(Q_W_sign_gauge_list)),
        "Q_W_SIGN_GAUGE_std": float(np.std(Q_W_sign_gauge_list, ddof=1)) if len(Q_W_sign_gauge_list) > 1 else 0.0,
        "Q_X_SIGN_GAUGE_mean": float(np.mean(Q_X_sign_gauge_list)),
        "Q_X_SIGN_GAUGE_std": float(np.std(Q_X_sign_gauge_list, ddof=1)) if len(Q_X_sign_gauge_list) > 1 else 0.0,
        "Q_W_SIGN_ALIGNED_mean": float(np.mean(Q_W_sign_gauge_list)),
        "Q_W_SIGN_ALIGNED_std": float(np.std(Q_W_sign_gauge_list, ddof=1)) if len(Q_W_sign_gauge_list) > 1 else 0.0,
        "Q_X_SIGN_ALIGNED_mean": float(np.mean(Q_X_sign_gauge_list)),
        "Q_X_SIGN_ALIGNED_std": float(np.std(Q_X_sign_gauge_list, ddof=1)) if len(Q_X_sign_gauge_list) > 1 else 0.0,
        "Q_Y_mean": float(np.mean(Q_Y_list)),
        "Q_Y_std": float(np.std(Q_Y_list, ddof=1)) if len(Q_Y_list) > 1 else 0.0,
        "Q_Y_COS_mean": float(np.mean(Q_Y_COS_list)),
        "Q_Y_COS_std": float(np.std(Q_Y_COS_list, ddof=1)) if len(Q_Y_COS_list) > 1 else 0.0,
        "NMSE_Y_mean": float(np.mean(NMSE_Y_list)),
        "NMSE_Y_std": float(np.std(NMSE_Y_list, ddof=1)) if len(NMSE_Y_list) > 1 else 0.0,
        "FIT_Y_mean": float(np.mean(FIT_Y_list)) if FIT_Y_list else 0.0,
        "FIT_Y_std": float(np.std(FIT_Y_list, ddof=1)) if len(FIT_Y_list) > 1 else 0.0,
        "Q_Y_PROJ_ABS_mean": float(np.mean(Q_Y_proj_abs_list)),
        "Q_Y_PROJ_ABS_std": float(np.std(Q_Y_proj_abs_list, ddof=1)) if len(Q_Y_proj_abs_list) > 1 else 0.0,
        "Q_W_COS_ROOT_mean": float(np.mean(Q_W_cos_root_list)),
        "Q_W_COS_ROOT_std": float(np.std(Q_W_cos_root_list, ddof=1)) if len(Q_W_cos_root_list) > 1 else 0.0,
        "Q_X_COS_ROOT_mean": float(np.mean(Q_X_cos_root_list)),
        "Q_X_COS_ROOT_std": float(np.std(Q_X_cos_root_list, ddof=1)) if len(Q_X_cos_root_list) > 1 else 0.0,
        "Q_W_SCALE_GAUGE_mean": float(np.mean(Q_W_scale_gauge_list)),
        "Q_W_SCALE_GAUGE_std": float(np.std(Q_W_scale_gauge_list, ddof=1)) if len(Q_W_scale_gauge_list) > 1 else 0.0,
        "Q_X_SCALE_GAUGE_mean": float(np.mean(Q_X_scale_gauge_list)),
        "Q_X_SCALE_GAUGE_std": float(np.std(Q_X_scale_gauge_list, ddof=1)) if len(Q_X_scale_gauge_list) > 1 else 0.0,
        "Q_WX_SCALE_GAUGE_mean": float(np.mean(Q_WX_scale_gauge_list)),
        "Q_WX_SCALE_GAUGE_std": float(np.std(Q_WX_scale_gauge_list, ddof=1)) if len(Q_WX_scale_gauge_list) > 1 else 0.0,
        "median_abs_log_k_mean": float(np.mean(median_abs_log_k_list)),
        "median_abs_log_k_std": float(np.std(median_abs_log_k_list, ddof=1)) if len(median_abs_log_k_list) > 1 else 0.0,
        "median_abs_log_g_mean": float(np.mean(median_abs_log_k_list)),
        "median_abs_log_g_std": float(np.std(median_abs_log_k_list, ddof=1)) if len(median_abs_log_k_list) > 1 else 0.0,
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
        result["Q_Y_observed_COS_mean"] = float(np.mean(Q_Y_observed_COS_list))
        result["Q_Y_observed_COS_std"] = float(np.std(Q_Y_observed_COS_list, ddof=1)) if len(Q_Y_observed_COS_list) > 1 else 0.0
        result["NMSE_Y_observed_mean"] = float(np.mean(NMSE_Y_observed_list))
        result["NMSE_Y_observed_std"] = float(np.std(NMSE_Y_observed_list, ddof=1)) if len(NMSE_Y_observed_list) > 1 else 0.0
        result["FIT_Y_observed_mean"] = float(np.mean(FIT_Y_observed_list)) if FIT_Y_observed_list else 0.0
        result["FIT_Y_observed_std"] = float(np.std(FIT_Y_observed_list, ddof=1)) if len(FIT_Y_observed_list) > 1 else 0.0
        result["Q_Y_observed_PROJ_ABS_mean"] = float(np.mean(Q_Y_observed_proj_abs_list))
        result["Q_Y_observed_PROJ_ABS_std"] = float(np.std(Q_Y_observed_proj_abs_list, ddof=1)) if len(Q_Y_observed_proj_abs_list) > 1 else 0.0
    if Q_Y_unobserved_list:
        result["Q_Y_unobserved_mean"] = float(np.mean(Q_Y_unobserved_list))
        result["Q_Y_unobserved_std"] = float(np.std(Q_Y_unobserved_list, ddof=1)) if len(Q_Y_unobserved_list) > 1 else 0.0
        result["Q_Y_unobserved_COS_mean"] = float(np.mean(Q_Y_unobserved_COS_list))
        result["Q_Y_unobserved_COS_std"] = float(np.std(Q_Y_unobserved_COS_list, ddof=1)) if len(Q_Y_unobserved_COS_list) > 1 else 0.0
        result["NMSE_Y_unobserved_mean"] = float(np.mean(NMSE_Y_unobserved_list))
        result["NMSE_Y_unobserved_std"] = float(np.std(NMSE_Y_unobserved_list, ddof=1)) if len(NMSE_Y_unobserved_list) > 1 else 0.0
        result["FIT_Y_unobserved_mean"] = float(np.mean(FIT_Y_unobserved_list)) if FIT_Y_unobserved_list else 0.0
        result["FIT_Y_unobserved_std"] = float(np.std(FIT_Y_unobserved_list, ddof=1)) if len(FIT_Y_unobserved_list) > 1 else 0.0
        result["Q_Y_unobserved_PROJ_ABS_mean"] = float(np.mean(Q_Y_unobserved_proj_abs_list))
        result["Q_Y_unobserved_PROJ_ABS_std"] = float(np.std(Q_Y_unobserved_proj_abs_list, ddof=1)) if len(Q_Y_unobserved_proj_abs_list) > 1 else 0.0

    return result
