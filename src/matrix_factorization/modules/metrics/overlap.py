"""
Overlap metrics for evaluating student-teacher similarity.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch


PROJECTION_NORM_EPS = 1e-12


@torch.no_grad()
def physical_overlap_fixed(
    student: torch.Tensor,
    teacher: torch.Tensor,
    eps: float = PROJECTION_NORM_EPS,
) -> float:
    """
    Fixed-denominator latent physical overlap.

    Q = <student, teacher> / numel(teacher)

    This is the paper-style coordinate overlap for O(1) latent variables.  It
    deliberately does not divide by ||teacher||^2, so a finite Gaussian teacher
    has perfect-match overlap equal to its empirical second moment.
    """
    if student.shape != teacher.shape:
        raise ValueError(
            "physical_overlap_fixed requires student and teacher to have "
            f"the same shape, got {tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    denom = max(int(teacher.numel()), 1)
    dot = (student.flatten().float() * teacher.flatten().float()).sum()
    if not torch.isfinite(dot):
        return 0.0
    return float(dot / (float(denom) + eps))


@torch.no_grad()
def student_self_overlap_fixed(student: torch.Tensor, eps: float = PROJECTION_NORM_EPS) -> float:
    """Fixed-denominator student self-overlap R = <student, student> / numel."""
    denom = max(int(student.numel()), 1)
    val = (student.flatten().float() * student.flatten().float()).sum()
    if not torch.isfinite(val):
        return 0.0
    return float(val / (float(denom) + eps))


@torch.no_grad()
def sign_gauge_overlap_fixed(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    latent_axis: int,
    eps: float = PROJECTION_NORM_EPS,
) -> float:
    """
    Fixed-denominator overlap after quotienting per-channel sign gauge.

    Q = sum_k |<student_k, teacher_k>| / numel(teacher)
    """
    if student.shape != teacher.shape:
        raise ValueError(
            "sign_gauge_overlap_fixed requires student and teacher to have "
            f"the same shape, got {tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    if student.dim() == 0:
        return abs(physical_overlap_fixed(student, teacher, eps=eps))

    latent_axis = latent_axis % student.dim()
    perm = [axis for axis in range(student.dim()) if axis != latent_axis] + [latent_axis]
    student_by_channel = student.float().permute(perm).reshape(-1, student.shape[latent_axis])
    teacher_by_channel = teacher.float().permute(perm).reshape(-1, teacher.shape[latent_axis])
    per_channel_dot = (student_by_channel * teacher_by_channel).sum(dim=0).abs()
    denom = max(int(teacher.numel()), 1)
    return float(per_channel_dot.sum() / (float(denom) + eps))


@torch.no_grad()
def normalized_mse_and_fit(
    student: torch.Tensor,
    teacher: torch.Tensor,
    eps: float = PROJECTION_NORM_EPS,
) -> tuple[float, float]:
    """
    Return (NMSE, FIT) for an output/evaluation vector.

    NMSE = ||student - teacher||^2 / ||teacher||^2
    FIT = 1 - NMSE
    """
    student_flat = student.flatten().float()
    teacher_flat = teacher.flatten().float()
    if teacher_flat.numel() == 0:
        return 1.0, 0.0
    denom = (teacher_flat * teacher_flat).sum()
    if float(denom.abs().item()) < eps:
        return 1.0, 0.0
    diff = student_flat - teacher_flat
    nmse = (diff * diff).sum() / (denom + eps)
    if not torch.isfinite(nmse):
        return 1.0, 0.0
    fit = 1.0 - float(nmse)
    return float(nmse), fit


@torch.no_grad()
def projection_abs(student: torch.Tensor, teacher: torch.Tensor, eps: float = PROJECTION_NORM_EPS) -> float:
    """
    Absolute projection overlap used by the formal projection-first metrics.

    Q = |<student, teacher>| / <teacher, teacher>

    The value is not clipped.  Values above 1 are preserved because they carry
    scale information rather than being a numerical error.
    """
    student_flat = student.flatten()
    teacher_flat = teacher.flatten()
    norm_teacher_sq = (teacher_flat ** 2).sum()
    if float(norm_teacher_sq.abs().item()) < eps:
        return 0.0
    dot = (student_flat * teacher_flat).sum().abs()
    return float(dot / (norm_teacher_sq + eps))


@torch.no_grad()
def sign_aligned_projection_abs(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    latent_axis: int,
    eps: float = PROJECTION_NORM_EPS,
) -> float:
    """
    Projection overlap after quotienting the per-channel sign gauge.

    Q = sum_k |<student_k, teacher_k>| / sum_k <teacher_k, teacher_k>

    This diagnostic keeps the same projection-first normalization and
    no-clipping policy as Q_W/Q_X, but removes the paired sign ambiguity of
    matrix factors. It does not remove rotations or permutations.
    """
    if student.shape != teacher.shape:
        raise ValueError(
            "sign_aligned_projection_abs requires student and teacher to have "
            f"the same shape, got {tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    if student.dim() == 0:
        return projection_abs(student, teacher, eps=eps)

    latent_axis = latent_axis % student.dim()
    perm = [axis for axis in range(student.dim()) if axis != latent_axis] + [latent_axis]
    student_by_channel = student.permute(perm).reshape(-1, student.shape[latent_axis])
    teacher_by_channel = teacher.permute(perm).reshape(-1, teacher.shape[latent_axis])
    norm_teacher_sq = (teacher_by_channel * teacher_by_channel).sum()
    if float(norm_teacher_sq.abs().item()) < eps:
        return 0.0
    per_channel_dot = (student_by_channel * teacher_by_channel).sum(dim=0).abs()
    return float(per_channel_dot.sum() / (norm_teacher_sq + eps))


@torch.no_grad()
def sign_gauge_projection_abs(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    latent_axis: int,
    eps: float = PROJECTION_NORM_EPS,
) -> float:
    """Legacy alias for the old teacher-norm sign-aligned projection."""
    return sign_aligned_projection_abs(student, teacher, latent_axis=latent_axis, eps=eps)


def _best_scale_gauge_scalar_from_sums(a: float, b: float, c: float, d: float) -> float:
    """Solve one channel of the diagonal scale-gauge alignment objective."""
    if not np.all(np.isfinite([a, b, c, d])):
        return float("nan")
    if a <= 1e-18 or c <= 1e-18:
        return 1.0

    candidates: list[float] = []
    try:
        roots = np.roots([a, -b, 0.0, d, -c])
    except (FloatingPointError, ValueError):
        roots = []
    for root in roots:
        if abs(root.imag) < 1e-8 and abs(root.real) > 1e-12:
            candidates.append(float(root.real))
    scale = float(np.sqrt(c / a))
    candidates.extend([scale, -scale, 1.0, -1.0])

    def objective(k: float) -> float:
        return float(a * k * k - 2.0 * b * k + c / (k * k) - 2.0 * d / k)

    return min(candidates, key=objective)


@torch.no_grad()
def scale_gauge_overlaps_fixed(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    eps: float = PROJECTION_NORM_EPS,
) -> Dict[str, float]:
    """Joint diagonal scale-gauge aligned W/X overlaps with fixed denominators."""
    if W_student.shape != W_teacher.shape or X_student.shape != X_teacher.shape:
        raise ValueError("scale_gauge_overlaps_fixed requires matching W and X teacher/student shapes")
    if W_student.dim() != 2 or X_student.dim() != 2:
        raise ValueError("scale_gauge_overlaps_fixed expects W=(N1,M) and X=(M,N2)")
    if W_student.shape[1] != X_student.shape[0]:
        raise ValueError("W latent dimension and X latent dimension must match")

    W_s = W_student.detach().float().cpu().numpy().astype(np.float64)
    X_s = X_student.detach().float().cpu().numpy().astype(np.float64)
    W_t = W_teacher.detach().float().cpu().numpy().astype(np.float64)
    X_t = X_teacher.detach().float().cpu().numpy().astype(np.float64)
    M = int(W_s.shape[1])
    w_dot = 0.0
    x_dot = 0.0
    k_values = np.empty(M, dtype=np.float64)
    for mu in range(M):
        a = float(np.sum(W_s[:, mu] * W_s[:, mu]))
        b = float(np.sum(W_s[:, mu] * W_t[:, mu]))
        c = float(np.sum(X_s[mu, :] * X_s[mu, :]))
        d = float(np.sum(X_s[mu, :] * X_t[mu, :]))
        k = _best_scale_gauge_scalar_from_sums(a, b, c, d)
        if not np.isfinite(k) or abs(k) <= eps:
            return {
                "Q_W_SCALE_GAUGE": float("nan"),
                "Q_X_SCALE_GAUGE": float("nan"),
                "Q_WX_SCALE_GAUGE": float("nan"),
                "median_abs_log_k": float("nan"),
            }
        k_values[mu] = k
        w_dot += k * b
        x_dot += (1.0 / k) * d

    q_w = float(w_dot / (float(W_t.size) + eps))
    q_x = float(x_dot / (float(X_t.size) + eps))
    return {
        "Q_W_SCALE_GAUGE": q_w,
        "Q_X_SCALE_GAUGE": q_x,
        "Q_WX_SCALE_GAUGE": 0.5 * (q_w + q_x),
        "median_abs_log_k": float(np.median(np.abs(np.log(np.maximum(np.abs(k_values), eps))))),
    }


@torch.no_grad()
def projection_abs_diagnostics(
    student: torch.Tensor,
    teacher: torch.Tensor,
    eps: float = PROJECTION_NORM_EPS,
) -> Dict[str, Any]:
    """Return projection value plus the degenerate-teacher-norm flag."""
    teacher_flat = teacher.flatten()
    norm_teacher_sq = (teacher_flat ** 2).sum()
    degenerate = float(norm_teacher_sq.abs().item()) < eps
    return {
        "value": projection_abs(student, teacher, eps=eps),
        "teacher_norm_squared": float(norm_teacher_sq.item()),
        "teacher_norm_epsilon": float(eps),
        "degenerate_teacher_norm": bool(degenerate),
        "clipped": False,
        "formula": "absolute_projection_teacher_norm_squared",
    }


@torch.no_grad()
def compute_cosine_similarity(A: torch.Tensor, B: torch.Tensor, use_left: bool = True) -> float:
    """
    Compute Gram matrix overlap using cosine similarity.

    Args:
        A: First matrix
        B: Second matrix (same shape as A)
        use_left: If True, compute A @ A.T vs B @ B.T
                  If False, compute A.T @ A vs B.T @ B

    Returns:
        Cosine similarity between Gram matrices
    """
    if use_left:
        G_A = A @ A.T
        G_B = B @ B.T
    else:
        G_A = A.T @ A
        G_B = B.T @ B

    G_A_flat = G_A.flatten()
    G_B_flat = G_B.flatten()

    dot = (G_A_flat * G_B_flat).sum()
    norm_A = G_A_flat.norm()
    norm_B = G_B_flat.norm()

    return float(dot / (norm_A * norm_B + 1e-12))


@torch.no_grad()
def compute_physical_overlap(pred: torch.Tensor, true: torch.Tensor, absolute: bool = False) -> float:
    """
    Legacy projection helper kept for debug and old result interpretation.

    New formal schema v3 metrics should use projection_abs() and Q_Y/Q_W/Q_X
    names instead of physical_overlap_* flat keys.

    Compute physical overlap (Projection of Student on Teacher).
    Overlap = <Pred, True> / <True, True>

    Args:
        pred: Predicted tensor (Student)
        true: True tensor (Teacher)
        absolute: If True, take absolute value of dot product (for sign ambiguity)

    Returns:
        Projection coefficient (m or m^2 depending on variable)
    """
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    dot = (pred_flat * true_flat).sum()
    if absolute:
        dot = dot.abs()

    norm_true_sq = (true_flat ** 2).sum()

    return float(dot / (norm_true_sq + 1e-12))


@torch.no_grad()
def gram_overlap_normalized(A: torch.Tensor, B: torch.Tensor, use_left: bool = True) -> float:
    """
    Compute normalized Gram overlap in [0, 1] range with baseline correction.

    Uses baseline b = m/(m+n+1) which is the expected cosine for random matrices.
    This ensures random initialization gives Q' ≈ 0, and perfect match gives Q' = 1.

    Args:
        A: First matrix
        B: Second matrix
        use_left: If True, use left Gram matrix

    Returns:
        Normalized overlap in [0, 1]
    """
    q = compute_cosine_similarity(A, B, use_left)

    if use_left:
        n, m = A.shape
    else:
        n, m = A.shape[1], A.shape[0]

    # Baseline: expected cosine for random matrices
    b = m / (m + n + 1)
    qc = (q - b) / (1.0 - b + 1e-12)

    return float(max(0.0, min(1.0, qc)))


@torch.no_grad()
def cos_overlap_root(A: torch.Tensor, B: torch.Tensor, use_left: bool = True) -> float:
    """Square root of the baseline-corrected Gram-cosine diagnostic."""
    return float(np.sqrt(max(0.0, gram_overlap_normalized(A, B, use_left=use_left))))


@torch.no_grad()
def gram_overlap_root(A: torch.Tensor, B: torch.Tensor, use_left: bool = True) -> float:
    """Legacy alias for :func:`cos_overlap_root`."""
    return cos_overlap_root(A, B, use_left=use_left)


@torch.no_grad()
def compute_qy(Y_student: torch.Tensor, Y_teacher: torch.Tensor) -> float:
    """
    Legacy Y-space cosine similarity.

    Args:
        Y_student: Student's Y = W @ X
        Y_teacher: Teacher's Y = W_t @ X_t

    Returns:
        Cosine similarity between Y matrices
    """
    y_s = Y_student.flatten()
    y_t = Y_teacher.flatten()

    dot = (y_s * y_t).sum()
    norm_s = y_s.norm()
    norm_t = y_t.norm()

    return float(dot / (norm_s * norm_t + 1e-12))


@torch.no_grad()
def compute_generalization_error(Y_student: torch.Tensor, Y_teacher: torch.Tensor) -> float:
    """Legacy/internal debug MSE; not a formal schema v3 metric."""
    return float(torch.mean((Y_teacher - Y_student) ** 2))


@torch.no_grad()
def compute_all_metrics(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    Y_teacher: torch.Tensor = None,
    mask: Optional[torch.Tensor] = None,
    metrics_to_compute: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute overlap metrics dynamically based on user request.

    Args:
        W_student: Student W matrix (N1, M)
        X_student: Student X matrix (M, N2)
        W_teacher: Teacher W matrix (N1, M)
        X_teacher: Teacher X matrix (M, N2)
        Y_teacher: Pre-computed teacher Y (optional)
        mask: Observation mask (required for Q_Y_unobserved/Q_Y_observed)
        metrics_to_compute: List of metric names to compute.
            If None, computes all standard metrics.
            Valid names: Q_W, Q_X, Q_W_SIGN_GAUGE, Q_X_SIGN_GAUGE,
                        Q_W_COS_ROOT, Q_X_COS_ROOT, Q_Y,
                        Q_Y_unobserved, Q_Y_observed

    Returns:
        Dictionary with requested metrics
    """
    # Default: all standard metrics
    if metrics_to_compute is None:
        metrics_to_compute = [
            'Q_W',
            'Q_X',
            'Q_W_SIGN_GAUGE',
            'Q_X_SIGN_GAUGE',
            'Q_W_COS_ROOT',
            'Q_X_COS_ROOT',
            'Q_Y',
        ]
    else:
        legacy_metric_aliases = {
            'Q_W_GRAM_ROOT': 'Q_W_COS_ROOT',
            'Q_X_GRAM_ROOT': 'Q_X_COS_ROOT',
            'Q_W_SIGN_ALIGNED': 'Q_W_SIGN_GAUGE',
            'Q_X_SIGN_ALIGNED': 'Q_X_SIGN_GAUGE',
        }
        metrics_to_compute = [legacy_metric_aliases.get(metric, metric) for metric in metrics_to_compute]

    if Y_teacher is None:
        Y_teacher = W_teacher @ X_teacher

    Y_student = W_student @ X_student

    results = {}

    # Compute requested metrics dynamically
    if 'Q_W' in metrics_to_compute:
        results['Q_W'] = physical_overlap_fixed(W_student, W_teacher)

    if 'Q_X' in metrics_to_compute:
        results['Q_X'] = physical_overlap_fixed(X_student, X_teacher)

    if 'Q_W_SIGN_GAUGE' in metrics_to_compute:
        results['Q_W_SIGN_GAUGE'] = sign_gauge_overlap_fixed(W_student, W_teacher, latent_axis=-1)

    if 'Q_X_SIGN_GAUGE' in metrics_to_compute:
        results['Q_X_SIGN_GAUGE'] = sign_gauge_overlap_fixed(X_student, X_teacher, latent_axis=0)

    if {
        'Q_W_SCALE_GAUGE',
        'Q_X_SCALE_GAUGE',
        'Q_WX_SCALE_GAUGE',
        'median_abs_log_k',
    } & set(metrics_to_compute):
        scale_metrics = scale_gauge_overlaps_fixed(W_student, X_student, W_teacher, X_teacher)
        results.update(scale_metrics)
        results['median_abs_log_g'] = scale_metrics['median_abs_log_k']

    if 'Q_W_COS_ROOT' in metrics_to_compute:
        results['Q_W_COS_ROOT'] = cos_overlap_root(W_student, W_teacher, use_left=True)

    if 'Q_X_COS_ROOT' in metrics_to_compute:
        results['Q_X_COS_ROOT'] = cos_overlap_root(X_student, X_teacher, use_left=False)

    if 'Q_Y' in metrics_to_compute:
        nmse, fit = normalized_mse_and_fit(Y_student, Y_teacher)
        results['NMSE_Y'] = nmse
        results['FIT_Y'] = fit
        results['Q_Y'] = projection_abs(Y_student, Y_teacher)

    if 'Q_W_PROJ_ABS' in metrics_to_compute:
        results['Q_W_PROJ_ABS'] = projection_abs(W_student, W_teacher)

    if 'Q_X_PROJ_ABS' in metrics_to_compute:
        results['Q_X_PROJ_ABS'] = projection_abs(X_student, X_teacher)

    if 'Q_Y_PROJ_ABS' in metrics_to_compute:
        results['Q_Y_PROJ_ABS'] = projection_abs(Y_student, Y_teacher)

    # New Physical Overlap Metrics
    if 'physical_overlap_Y' in metrics_to_compute:
        results['physical_overlap_Y'] = projection_abs(Y_student, Y_teacher)

    if 'physical_overlap_W' in metrics_to_compute:
        # For W, absolute=True (sign ambiguity)
        results['physical_overlap_W'] = compute_physical_overlap(W_student, W_teacher, absolute=True)

    if 'physical_overlap_X' in metrics_to_compute:
        # For X, absolute=True
        results['physical_overlap_X'] = compute_physical_overlap(X_student, X_teacher, absolute=True)

    if 'Gen_Error' in metrics_to_compute:
        results['Gen_Error'] = compute_generalization_error(Y_student, Y_teacher)

    # Q_Y_unobserved and Q_Y_observed require mask
    if mask is not None:
        if 'Q_Y_unobserved' in metrics_to_compute:
            nmse, fit = _compute_nmse_fit_masked(Y_student, Y_teacher, mask, observed=False)
            results['NMSE_Y_unobserved'] = nmse
            results['FIT_Y_unobserved'] = fit
            results['Q_Y_unobserved'] = _compute_qy_masked(Y_student, Y_teacher, mask, observed=False)

        if 'Q_Y_observed' in metrics_to_compute:
            nmse, fit = _compute_nmse_fit_masked(Y_student, Y_teacher, mask, observed=True)
            results['NMSE_Y_observed'] = nmse
            results['FIT_Y_observed'] = fit
            results['Q_Y_observed'] = _compute_qy_masked(Y_student, Y_teacher, mask, observed=True)

        if 'Q_Y_observed_PROJ_ABS' in metrics_to_compute:
            results['Q_Y_observed_PROJ_ABS'] = _compute_qy_masked(Y_student, Y_teacher, mask, observed=True)

        if 'Q_Y_unobserved_PROJ_ABS' in metrics_to_compute:
            results['Q_Y_unobserved_PROJ_ABS'] = _compute_qy_masked(Y_student, Y_teacher, mask, observed=False)

        # Physical overlap on observed/unobserved positions
        if 'physical_overlap_Y_observed' in metrics_to_compute:
            results['physical_overlap_Y_observed'] = _compute_physical_overlap_masked(
                Y_student, Y_teacher, mask, observed=True
            )

        if 'physical_overlap_Y_unobserved' in metrics_to_compute:
            results['physical_overlap_Y_unobserved'] = _compute_physical_overlap_masked(
                Y_student, Y_teacher, mask, observed=False
            )

    return results


@torch.no_grad()
def _compute_qy_masked(
    Y_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    mask: torch.Tensor,
    observed: bool = False,
) -> float:
    """
    Compute projection-first Q_Y on observed or unobserved positions.

    Args:
        Y_student: Student reconstruction
        Y_teacher: Teacher Y matrix
        mask: Observation mask (1 = observed, 0 = unobserved)
        observed: If True, compute on observed positions; else unobserved

    Returns:
        Absolute projection overlap on the selected positions
    """
    # Handle batch dimension in mask
    if mask.dim() == 3:
        mask = mask[0]  # Take first sample's mask

    if observed:
        selection_mask = mask > 0.5
    else:
        selection_mask = mask < 0.5

    # Extract selected elements
    y_s = Y_student[selection_mask].flatten()
    y_t = Y_teacher[selection_mask].flatten()

    if y_s.numel() == 0:
        return 0.0

    return projection_abs(y_s, y_t)


@torch.no_grad()
def _compute_nmse_fit_masked(
    Y_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    mask: torch.Tensor,
    observed: bool = False,
) -> tuple[float, float]:
    """Compute (NMSE, FIT) on observed or unobserved positions."""
    if mask.dim() == 3:
        mask = mask[0]

    selection_mask = mask > 0.5 if observed else mask < 0.5
    y_s = Y_student[selection_mask].flatten()
    y_t = Y_teacher[selection_mask].flatten()
    return normalized_mse_and_fit(y_s, y_t)


@torch.no_grad()
def _compute_physical_overlap_masked(
    Y_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    mask: torch.Tensor,
    observed: bool = False,
) -> float:
    """
    Compute physical overlap on observed or unobserved positions.

    Args:
        Y_student: Student reconstruction
        Y_teacher: Teacher Y matrix
        mask: Observation mask (1 = observed, 0 = unobserved)
        observed: If True, compute on observed positions; else unobserved

    Returns:
        Physical overlap (projection) on selected positions
    """
    if mask.dim() == 3:
        mask = mask[0]

    if observed:
        selection_mask = mask > 0.5
    else:
        selection_mask = mask < 0.5

    y_s = Y_student[selection_mask].flatten()
    y_t = Y_teacher[selection_mask].flatten()

    if y_s.numel() == 0:
        return 0.0

    return projection_abs(y_s, y_t)


@torch.no_grad()
def compute_replica_overlap(W_all: torch.Tensor, X_all: torch.Tensor) -> Dict[str, float]:
    """
    Compute pairwise Gram overlap between S replicas.

    Args:
        W_all: (S, N1, M) - S replicas of W
        X_all: (S, M, N2) - S replicas of X

    Returns:
        Dictionary with replica overlap stats
    """
    S = W_all.shape[0]
    if S < 2:
        return {
            'Q_W_replica_mean': 0.0,
            'Q_W_replica_std': 0.0,
            'Q_X_replica_mean': 0.0,
            'Q_X_replica_std': 0.0,
        }

    Q_W_list, Q_X_list = [], []
    physical_W_list, physical_X_list = [], []

    for i in range(S):
        for j in range(i + 1, S):
            Q_W_list.append(compute_cosine_similarity(W_all[i], W_all[j], use_left=True))
            Q_X_list.append(compute_cosine_similarity(X_all[i], X_all[j], use_left=False))
            # Physical overlap between replicas (absolute=True for sign ambiguity)
            physical_W_list.append(compute_physical_overlap(W_all[i], W_all[j], absolute=True))
            physical_X_list.append(compute_physical_overlap(X_all[i], X_all[j], absolute=True))

    return {
        'Q_W_replica_mean': round(float(np.mean(Q_W_list)), 6),
        'Q_W_replica_std': round(float(np.std(Q_W_list, ddof=1)), 6) if len(Q_W_list) > 1 else 0.0,
        'Q_X_replica_mean': round(float(np.mean(Q_X_list)), 6),
        'Q_X_replica_std': round(float(np.std(Q_X_list, ddof=1)), 6) if len(Q_X_list) > 1 else 0.0,
        'physical_W_replica_mean': round(float(np.mean(physical_W_list)), 6),
        'physical_W_replica_std': round(float(np.std(physical_W_list, ddof=1)), 6) if len(physical_W_list) > 1 else 0.0,
        'physical_X_replica_mean': round(float(np.mean(physical_X_list)), 6),
        'physical_X_replica_std': round(float(np.std(physical_X_list, ddof=1)), 6) if len(physical_X_list) > 1 else 0.0,
    }


def aggregate_trial_metrics(trial_results: list[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple trials.

    Args:
        trial_results: List of metric dictionaries from each trial

    Returns:
        Dictionary with mean and std for each metric
    """
    if not trial_results:
        return {}

    aggregated = {}
    for key in trial_results[0].keys():
        vals = [r[key] for r in trial_results]
        # Round to 6 decimal places for cleaner output
        aggregated[f'{key}_mean'] = round(float(np.mean(vals)), 6)
        aggregated[f'{key}_std'] = round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else 0.0

    return aggregated


def build_interaction_matrix(
    samples_A: torch.Tensor,
    teacher_A: torch.Tensor,
    metric_fn: callable,
    samples_B: torch.Tensor = None,
    teacher_B: torch.Tensor = None,
    use_left: bool = True,
    absolute: bool = False,
) -> np.ndarray:
    """
    Build (S+1)x(S+1) interaction matrix between Teacher and S Replicas.

    Layout:
        Row/Col 0: Teacher
        Row/Col 1..S: Replicas 1..S

    Args:
        samples_A: (S, N, M) Student replicas
        teacher_A: (N, M) Teacher matrix
        metric_fn: Function to compute similarity (A, B) -> float
        samples_B: Optional second matrix argument (for asymmetric metrics)
        teacher_B: Optional second teacher argument
        use_left: Argument for metric_fn (if supported)
        absolute: Argument for metric_fn (if supported)

    Returns:
        (S+1, S+1) numpy array with metric values
    """
    S = samples_A.shape[0]
    matrix = np.zeros((S + 1, S + 1), dtype=np.float32)

    # Prepare list of (S+1) matrices [Teacher, Rep1, Rep2, ..., RepS]
    # Note: Clone to ensure no side effects
    all_A = [teacher_A] + [samples_A[i] for i in range(S)]
    
    if samples_B is not None and teacher_B is not None:
        all_B = [teacher_B] + [samples_B[i] for i in range(S)]
    else:
        all_B = all_A

    # Pre-compute metrics (Symmetric optimization possible depending on metric, 
    # but for safety we compute all N^2 or N(N+1)/2)
    # Using simple loop for clarity. Reliability > Micro-optimization here.
    
    import inspect
    sig = inspect.signature(metric_fn)
    has_use_left = 'use_left' in sig.parameters
    has_absolute = 'absolute' in sig.parameters

    kwargs = {}
    if has_use_left:
        kwargs['use_left'] = use_left
    if has_absolute:
        kwargs['absolute'] = absolute

    for i in range(S + 1):
        for j in range(S + 1):
            if i == j:
                # Self-overlap is usually 1.0 (normalized) or norm^2
                # We compute it explicitly to be safe
                val = metric_fn(all_A[i], all_B[i], **kwargs)
            else:
                val = metric_fn(all_A[i], all_B[j], **kwargs)
            
            matrix[i, j] = val

    return matrix


def get_metric_function(name: str) -> callable:
    """Get metric function by name."""
    name = name.lower()
    if 'gram' in name or 'normalized' in name:
        return gram_overlap_normalized
    elif 'cosine' in name:
        return compute_cosine_similarity
    elif 'physical' in name:
        return compute_physical_overlap
    elif 'qy' in name:
        return projection_abs
    else:
        raise ValueError(f"Unknown metric function: {name}")
