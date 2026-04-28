"""
Evaluation metrics for random spreading model.

Key difference from standard metrics:
- Q_Y uses the same F coefficients for both teacher and student.
- Formal Q_Y is FIT = 1 - NMSE on the configured measurement set.
- Q_W/Q_X are fixed-denominator physical overlaps; Cos-root metrics are diagnostics.
"""

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING
import numpy as np
import torch

from matrix_factorization.core.distributions import F_DISTRIBUTION_ISING

from ..teachers.random_spreading import SpreadingData, compute_sparse_Y
from .overlap import normalized_mse_and_fit, physical_overlap_fixed, projection_abs, sign_gauge_overlap_fixed

if TYPE_CHECKING:
    from ..teachers.random_spreading import SpreadingDataParallel

PROJECTION_NORM_EPS = 1e-12


@dataclass
class BatchMetricPayload:
    """GPU-resident metric vectors for one alpha batch.

    Values stay as tensors while the batch is being processed.  The only
    intended CPU materialization point is to_metrics_by_alpha(), which converts
    the compact per-alpha scalar vectors for JSON/result bookkeeping.
    """

    alpha_values: torch.Tensor
    metrics: Dict[str, torch.Tensor]
    metadata: Dict[str, object] | None = None

    def to_metrics_by_alpha(self) -> Dict[float, Dict[str, float]]:
        alpha_cpu = self.alpha_values.detach().float().cpu().tolist()
        metric_cpu = {
            key: value.detach().float().cpu().reshape(-1).tolist()
            for key, value in self.metrics.items()
            if key != "alpha_values"
        }
        result: Dict[float, Dict[str, float]] = {}
        for idx, alpha in enumerate(alpha_cpu):
            result[float(alpha)] = {
                key: float(values[idx])
                for key, values in metric_cpu.items()
                if idx < len(values)
            }
        return result


@torch.no_grad()
def _projection_abs_values(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    norm_teacher_sq = (teacher.flatten() ** 2).sum()
    if float(norm_teacher_sq.abs().item()) < PROJECTION_NORM_EPS:
        return torch.zeros((), device=student.device, dtype=torch.float32)
    return ((student.flatten() * teacher.flatten()).sum().abs() / (norm_teacher_sq + PROJECTION_NORM_EPS)).float()


@torch.no_grad()
def _projection_abs_batch(student: torch.Tensor, teacher: torch.Tensor, reduce_dims: tuple[int, ...]) -> torch.Tensor:
    norm_teacher_sq = (teacher * teacher).sum(dim=reduce_dims)
    dot = (student * teacher).sum(dim=reduce_dims).abs()
    return torch.where(
        norm_teacher_sq.abs() < PROJECTION_NORM_EPS,
        torch.zeros_like(dot, dtype=torch.float32),
        (dot / (norm_teacher_sq + PROJECTION_NORM_EPS)).float(),
    )


@torch.no_grad()
def _fixed_overlap_batch(student: torch.Tensor, teacher: torch.Tensor, reduce_dims: tuple[int, ...]) -> torch.Tensor:
    dot = (student.float() * teacher.float()).sum(dim=reduce_dims)
    denom = 1
    for dim in reduce_dims:
        denom *= int(student.shape[dim])
    return (dot / (float(max(denom, 1)) + PROJECTION_NORM_EPS)).float()


@torch.no_grad()
def _sign_aligned_projection_abs_batch(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    channel_sum_dim: int,
    channel_axis_after_sum: int = -1,
) -> torch.Tensor:
    """Batched per-channel sign-gauge-aligned projection.

    student is expected to carry leading sample/alpha axes followed by the
    factor axes.  teacher is broadcast to student shape by the caller.  The
    reduction first sums over the coordinate dimension within each latent
    channel, then takes abs per channel and sums channels.
    """
    norm_teacher_sq = (teacher * teacher).sum()
    if float(norm_teacher_sq.abs().item()) < PROJECTION_NORM_EPS:
        return torch.zeros(student.shape[:2], device=student.device, dtype=torch.float32)
    per_channel_dot = (student * teacher).sum(dim=channel_sum_dim).abs()
    if channel_axis_after_sum != -1:
        per_channel_dot = per_channel_dot.movedim(channel_axis_after_sum, -1)
    return (per_channel_dot.sum(dim=-1) / (norm_teacher_sq + PROJECTION_NORM_EPS)).float()


@torch.no_grad()
def _sign_gauge_overlap_fixed_batch(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    channel_sum_dim: int,
    channel_axis_after_sum: int = -1,
) -> torch.Tensor:
    per_channel_dot = (student.float() * teacher.float()).sum(dim=channel_sum_dim).abs()
    if channel_axis_after_sum != -1:
        per_channel_dot = per_channel_dot.movedim(channel_axis_after_sum, -1)
    denom = float(max(int(teacher.expand_as(student).shape[-2] * teacher.expand_as(student).shape[-1]), 1))
    return (per_channel_dot.sum(dim=-1) / (denom + PROJECTION_NORM_EPS)).float()


@torch.no_grad()
def _projection_abs_from_sums(dot: torch.Tensor, norm_teacher_sq: torch.Tensor) -> torch.Tensor:
    return torch.where(
        norm_teacher_sq.abs() < PROJECTION_NORM_EPS,
        torch.zeros_like(dot, dtype=torch.float32),
        (dot.abs() / (norm_teacher_sq + PROJECTION_NORM_EPS)).float(),
    )


@torch.no_grad()
def _fit_from_sums(sse: torch.Tensor, norm_teacher_sq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nmse = torch.where(
        norm_teacher_sq.abs() < PROJECTION_NORM_EPS,
        torch.ones_like(sse, dtype=torch.float32),
        (sse / (norm_teacher_sq + PROJECTION_NORM_EPS)).float(),
    )
    return 1.0 - nmse, nmse


@torch.no_grad()
def _observed_edge_projection_sums(
    W_a: torch.Tensor,
    X_a_t: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    F_values: torch.Tensor,
    Y_teacher: torch.Tensor,
    *,
    sqrt_m_inv: float,
    edge_chunk_size: int,
    sample_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate spreading measurement projection sums on GPU in edge chunks."""

    S, C = int(i_idx.shape[0]), int(i_idx.shape[1])
    dot = torch.zeros(S, device=W_a.device, dtype=torch.float32)
    norm_teacher_sq = torch.zeros(S, device=W_a.device, dtype=torch.float32)
    sse = torch.zeros(S, device=W_a.device, dtype=torch.float32)
    chunk = max(1, int(edge_chunk_size))
    sample_chunk = max(1, min(int(sample_chunk_size), S))
    for sample_start in range(0, S, sample_chunk):
        sample_stop = min(S, sample_start + sample_chunk)
        block = sample_stop - sample_start
        W_block = W_a[sample_start:sample_stop]
        X_block = X_a_t[sample_start:sample_stop]
        for start in range(0, C, chunk):
            stop = min(C, start + chunk)
            i_chunk = i_idx[sample_start:sample_stop, start:stop]
            j_chunk = j_idx[sample_start:sample_stop, start:stop]
            F_chunk = F_values[sample_start:sample_stop, start:stop]
            if not torch.is_floating_point(F_chunk):
                F_chunk = F_chunk.float()
            W_selected = W_block.gather(1, i_chunk.unsqueeze(-1).expand(block, stop - start, W_block.shape[-1]))
            X_selected = X_block.gather(1, j_chunk.unsqueeze(-1).expand(block, stop - start, X_block.shape[-1]))
            Y_student = sqrt_m_inv * (F_chunk * W_selected * X_selected).sum(dim=-1)
            Y_teacher_chunk = Y_teacher[sample_start:sample_stop, start:stop]
            dot[sample_start:sample_stop] += (Y_student * Y_teacher_chunk).sum(dim=-1).float()
            norm_teacher_sq[sample_start:sample_stop] += (Y_teacher_chunk * Y_teacher_chunk).sum(dim=-1).float()
            diff = Y_student - Y_teacher_chunk
            sse[sample_start:sample_stop] += (diff * diff).sum(dim=-1).float()
    return dot, norm_teacher_sq, sse


@torch.no_grad()
def _heldout_edge_projection_sums(
    W_a: torch.Tensor,
    X_a_t: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher_t: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    F_values: torch.Tensor,
    *,
    sqrt_m_inv: float,
    edge_chunk_size: int,
    sample_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate heldout spreading measurement projection sums on GPU."""

    S, C = int(i_idx.shape[0]), int(i_idx.shape[1])
    dot = torch.zeros(S, device=W_a.device, dtype=torch.float32)
    norm_teacher_sq = torch.zeros(S, device=W_a.device, dtype=torch.float32)
    sse = torch.zeros(S, device=W_a.device, dtype=torch.float32)
    chunk = max(1, int(edge_chunk_size))
    sample_chunk = max(1, min(int(sample_chunk_size), S))
    for sample_start in range(0, S, sample_chunk):
        sample_stop = min(S, sample_start + sample_chunk)
        block = sample_stop - sample_start
        W_block = W_a[sample_start:sample_stop]
        X_block = X_a_t[sample_start:sample_stop]
        for start in range(0, C, chunk):
            stop = min(C, start + chunk)
            i_chunk = i_idx[sample_start:sample_stop, start:stop]
            j_chunk = j_idx[sample_start:sample_stop, start:stop]
            F_chunk = F_values[sample_start:sample_stop, start:stop]
            if not torch.is_floating_point(F_chunk):
                F_chunk = F_chunk.float()

            W_teacher_chunk = W_teacher[i_chunk]
            X_teacher_chunk = X_teacher_t[j_chunk]
            Y_teacher = sqrt_m_inv * (F_chunk * W_teacher_chunk * X_teacher_chunk).sum(dim=-1)

            W_student_chunk = W_block.gather(1, i_chunk.unsqueeze(-1).expand(block, stop - start, W_block.shape[-1]))
            X_student_chunk = X_block.gather(1, j_chunk.unsqueeze(-1).expand(block, stop - start, X_block.shape[-1]))
            Y_student = sqrt_m_inv * (F_chunk * W_student_chunk * X_student_chunk).sum(dim=-1)

            dot[sample_start:sample_stop] += (Y_student * Y_teacher).sum(dim=-1).float()
            norm_teacher_sq[sample_start:sample_stop] += (Y_teacher * Y_teacher).sum(dim=-1).float()
            diff = Y_student - Y_teacher
            sse[sample_start:sample_stop] += (diff * diff).sum(dim=-1).float()
    return dot, norm_teacher_sq, sse


@torch.no_grad()
def _masked_projection_abs_batch(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask_f = mask.to(dtype=student.dtype)
    dot = (student * teacher * mask_f).sum(dim=-1).abs()
    norm_teacher_sq = ((teacher * teacher) * mask_f).sum(dim=-1)
    return torch.where(
        norm_teacher_sq.abs() < PROJECTION_NORM_EPS,
        torch.zeros_like(dot, dtype=torch.float32),
        (dot / (norm_teacher_sq + PROJECTION_NORM_EPS)).float(),
    )


@torch.no_grad()
def _mean_std(values: torch.Tensor, dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    return values.mean(dim=dim), values.std(dim=dim) if values.shape[dim] > 1 else torch.zeros_like(values.mean(dim=dim))


@torch.no_grad()
def _cos_root_and_replica_by_alpha(
    factors: torch.Tensor,
    teacher: torch.Tensor,
    *,
    use_left: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return teacher-student Cos-root and student-student Gram diagnostics.

    factors is (S, N, M) for W when use_left=True, or (S, M, N) for X when
    use_left=False.  The computation stays on the device and vectorizes the
    former S^2 Python pair loop.
    """
    if use_left:
        grams = torch.matmul(factors, factors.transpose(-1, -2))
        teacher_gram = teacher @ teacher.T
        n, m = int(factors.shape[-2]), int(factors.shape[-1])
    else:
        grams = torch.matmul(factors.transpose(-1, -2), factors)
        teacher_gram = teacher.T @ teacher
        n, m = int(factors.shape[-1]), int(factors.shape[-2])

    flat = grams.reshape(grams.shape[0], -1).float()
    teacher_flat = teacher_gram.reshape(-1).float()
    teacher_norm = teacher_flat.norm() + PROJECTION_NORM_EPS
    q = (flat * teacher_flat).sum(dim=1) / ((flat.norm(dim=1) * teacher_norm) + PROJECTION_NORM_EPS)
    baseline = float(m) / float(m + n + 1)
    corrected = ((q - baseline) / (1.0 - baseline + PROJECTION_NORM_EPS)).clamp(0.0, 1.0)
    cos_root = corrected.sqrt()

    if factors.shape[0] < 2:
        zero = torch.zeros((), device=factors.device, dtype=torch.float32)
        return cos_root.float(), zero, zero

    normalized = flat / (flat.norm(dim=1, keepdim=True) + PROJECTION_NORM_EPS)
    pair_cos = normalized @ normalized.T
    pair_corrected = ((pair_cos - baseline) / (1.0 - baseline + PROJECTION_NORM_EPS)).clamp(0.0, 1.0)
    upper = torch.triu_indices(factors.shape[0], factors.shape[0], offset=1, device=factors.device)
    return (
        cos_root.float(),
        pair_cos[upper[0], upper[1]].mean().float(),
        pair_corrected[upper[0], upper[1]].mean().float(),
    )


def _best_scale_gauge_scalar_from_sums(a: float, b: float, c: float, d: float) -> float:
    """Solve the per-channel W/X scale-gauge alignment problem.

    The minimized objective is
    ||k W_s[:,mu] - W_t[:,mu]||^2 + ||k^-1 X_s[mu,:] - X_t[mu,:]||^2.
    This mirrors scripts/analysis/posthoc_scale_gauge.py so runtime metrics
    and legacy posthoc diagnostics use the same convention.
    """
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

    def objective(g: float) -> float:
        return float(a * g * g - 2.0 * b * g + c / (g * g) - 2.0 * d / g)

    return min(candidates, key=objective)


@torch.no_grad()
def _scale_gauge_projection_batch(
    W_students: torch.Tensor,
    X_students: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute joint per-channel scale-gauge aligned factor diagnostics.

    Inputs are W: (S, A, N1, M), X: (S, A, M, N2).  The scalar gauge k_mu is
    chosen jointly for W channel k and X channel k, so this is a diagnostic
    for the diagonal scale gauge W_:mu -> k_mu W_:mu, X_mu: -> k_mu^-1 X_mu:.
    """
    device = W_students.device
    W_s = W_students.detach().float().cpu().numpy()
    X_s = X_students.detach().float().cpu().numpy()
    W_t = W_teacher.detach().float().cpu().numpy()
    X_t = X_teacher.detach().float().cpu().numpy()

    S, A, _, M = W_s.shape
    a_vals = np.sum(W_s.astype(np.float64) * W_s.astype(np.float64), axis=2)
    b_vals = np.sum(W_s.astype(np.float64) * W_t.astype(np.float64)[None, None, :, :], axis=2)
    c_vals = np.sum(X_s.astype(np.float64) * X_s.astype(np.float64), axis=3)
    d_vals = np.sum(X_s.astype(np.float64) * X_t.astype(np.float64)[None, None, :, :], axis=3)

    w_denominator = float(np.prod(W_t.shape)) + PROJECTION_NORM_EPS
    x_denominator = float(np.prod(X_t.shape)) + PROJECTION_NORM_EPS
    q_w = np.full((S, A), np.nan, dtype=np.float64)
    q_x = np.full((S, A), np.nan, dtype=np.float64)
    gauge_mag = np.full((S, A), np.nan, dtype=np.float64)

    for sample_idx in range(S):
        for alpha_idx in range(A):
            w_dot = 0.0
            x_dot = 0.0
            gauges = np.empty(M, dtype=np.float64)
            valid = True
            for channel_idx in range(M):
                k = _best_scale_gauge_scalar_from_sums(
                    float(a_vals[sample_idx, alpha_idx, channel_idx]),
                    float(b_vals[sample_idx, alpha_idx, channel_idx]),
                    float(c_vals[sample_idx, alpha_idx, channel_idx]),
                    float(d_vals[sample_idx, alpha_idx, channel_idx]),
                )
                if not np.isfinite(k) or abs(k) <= 1e-12:
                    valid = False
                    break
                gauges[channel_idx] = k
                w_dot += k * float(b_vals[sample_idx, alpha_idx, channel_idx])
                x_dot += (1.0 / k) * float(d_vals[sample_idx, alpha_idx, channel_idx])
            if not valid:
                continue
            q_w[sample_idx, alpha_idx] = w_dot / w_denominator
            q_x[sample_idx, alpha_idx] = x_dot / x_denominator
            gauge_mag[sample_idx, alpha_idx] = float(
                np.median(np.abs(np.log(np.maximum(np.abs(gauges), 1e-12))))
            )

    q_w_tensor = torch.as_tensor(q_w, dtype=torch.float32, device=device)
    q_x_tensor = torch.as_tensor(q_x, dtype=torch.float32, device=device)
    q_wx_tensor = 0.5 * (q_w_tensor + q_x_tensor)
    gauge_tensor = torch.as_tensor(gauge_mag, dtype=torch.float32, device=device)
    return q_w_tensor, q_x_tensor, q_wx_tensor, gauge_tensor


@torch.no_grad()
def _deterministic_heldout_edges(
    *,
    N1: int,
    N2: int,
    observed_i: torch.Tensor,
    observed_j: torch.Tensor,
    count: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample deterministic heldout matrix edges outside the observed set."""
    total = int(N1) * int(N2)
    if count <= 0 or total <= 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    observed_keys = (observed_i.long() * int(N2) + observed_j.long()).unique()
    available = max(0, total - int(observed_keys.numel()))
    target = min(int(count), available)
    if target <= 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    gen = torch.Generator(device=device).manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    selected_count = 0
    attempts = 0
    while selected_count < target and attempts < 32:
        attempts += 1
        draw = max(64, 3 * (target - selected_count))
        candidates = torch.randint(0, total, (draw,), generator=gen, device=device)
        mask = ~torch.isin(candidates, observed_keys)
        if selected:
            mask &= ~torch.isin(candidates, torch.cat(selected))
        unique_candidates = candidates[mask].unique()
        if unique_candidates.numel() == 0:
            continue
        take = unique_candidates[: target - selected_count]
        selected.append(take)
        selected_count += int(take.numel())

    if selected_count < target:
        # Deterministic fallback for dense regimes where rejection sampling stalls.
        all_keys = torch.arange(total, device=device)
        mask = ~torch.isin(all_keys, observed_keys)
        if selected:
            mask &= ~torch.isin(all_keys, torch.cat(selected))
        fallback = all_keys[mask][: target - selected_count]
        if fallback.numel() > 0:
            selected.append(fallback)
            selected_count += int(fallback.numel())

    if not selected:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    keys = torch.cat(selected)[:target].long()
    return keys // int(N2), keys % int(N2)


@torch.no_grad()
def compute_qy_spreading(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    spreading_data: SpreadingData,
) -> float:
    """
    Compute Q_Y for random spreading model.

    Both teacher Y and student Y are computed at observed positions
    using the SAME F coefficients. This ensures fair comparison.

    Q_Y = 1 - ||Y_student - Y_teacher||^2 / ||Y_teacher||^2

    Args:
        W_student: (N1, M) or (S, N1, M) student W matrix
        X_student: (M, N2) or (S, M, N2) student X matrix
        spreading_data: SpreadingData with F and teacher Y_values

    Returns:
        Output fit, not clipped.

    Note:
        If W_student has batch dimension, returns mean Q_Y across samples.
    """
    # Handle batch dimension
    if W_student.dim() == 3:
        # Batched: (S, N1, M), (S, M, N2)
        S = W_student.shape[0]
        qy_values = []
        for s in range(S):
            qy = compute_qy_spreading(
                W_student[s], X_student[s], spreading_data
            )
            qy_values.append(qy)
        return sum(qy_values) / len(qy_values)

    # Single sample: (N1, M), (M, N2)
    # Compute student Y at observed positions with same F
    Y_student_values = compute_sparse_Y(
        W_student, X_student,
        spreading_data.F,
        spreading_data.i_idx,
        spreading_data.j_idx,
    )

    Y_teacher_values = spreading_data.Y_values

    return float(_projection_abs_values(Y_student_values, Y_teacher_values))


@torch.no_grad()
def compute_physical_overlap_spreading(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    spreading_data: SpreadingData,
) -> float:
    """
    Legacy/debug name for spreading measurement projection.

    New formal schema v3 emits Q_Y/Q_Y_observed/Q_Y_unobserved instead of
    physical_overlap_* keys.

    Compute Physical Overlap for random spreading model.
    Overlap = <Y_s, Y_t> / <Y_t, Y_t>
    """
    if W_student.dim() == 3:
        S = W_student.shape[0]
        vals = []
        for s in range(S):
            vals.append(compute_physical_overlap_spreading(
                W_student[s], X_student[s], spreading_data
            ))
        return sum(vals) / len(vals)

    Y_student_values = compute_sparse_Y(
        W_student, X_student,
        spreading_data.F,
        spreading_data.i_idx,
        spreading_data.j_idx,
    )

    Y_teacher_values = spreading_data.Y_values

    return float(_projection_abs_values(Y_student_values, Y_teacher_values))


@torch.no_grad()
def compute_mse_spreading(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    spreading_data: SpreadingData,
) -> float:
    """
    Compute MSE (Mean Squared Error) for random spreading model.

    This is an internal/legacy debug loss helper.  It is not emitted as a
    formal schema v3 result metric.

    MSE = mean((Y_student - Y_teacher)^2) at observed positions.

    Args:
        W_student: (N1, M) student W matrix
        X_student: (M, N2) student X matrix
        spreading_data: SpreadingData with F and teacher Y_values

    Returns:
        Mean squared error (lower is better)
    """
    if W_student.dim() == 3:
        S = W_student.shape[0]
        mse_values = []
        for s in range(S):
            mse = compute_mse_spreading(
                W_student[s], X_student[s], spreading_data
            )
            mse_values.append(mse)
        return sum(mse_values) / len(mse_values)

    Y_student_values = compute_sparse_Y(
        W_student, X_student,
        spreading_data.F,
        spreading_data.i_idx,
        spreading_data.j_idx,
    )

    Y_teacher_values = spreading_data.Y_values

    mse = ((Y_student_values - Y_teacher_values) ** 2).mean()

    return float(mse)


@torch.no_grad()
def compute_all_metrics_spreading(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    spreading_data: SpreadingData,
) -> Dict[str, float]:
    """
    Compute all metrics for random spreading model.

    Includes:
    - Q_Y: F-aware projection overlap
    - Q_W, Q_X: coordinate projection overlap
    - Q_W_COS_ROOT, Q_X_COS_ROOT: Cos-root diagnostics

    Args:
        W_student: (N1, M) or (S, N1, M) student W
        X_student: (M, N2) or (S, M, N2) student X
        W_teacher: (N1, M) teacher W
        X_teacher: (M, N2) teacher X
        spreading_data: SpreadingData with F and Y_values

    Returns:
        Dictionary with all metrics
    """
    from .overlap import cos_overlap_root

    results = {}

    # Handle batch dimension for W_student, X_student
    if W_student.dim() == 3:
        W_s = W_student.mean(dim=0)  # Average over samples
        X_s = X_student.mean(dim=0)
    else:
        W_s = W_student
        X_s = X_student

    results['Q_W'] = physical_overlap_fixed(W_s, W_teacher)
    results['Q_X'] = physical_overlap_fixed(X_s, X_teacher)
    results['Q_W_PROJ_ABS'] = projection_abs(W_s, W_teacher)
    results['Q_X_PROJ_ABS'] = projection_abs(X_s, X_teacher)
    results['Q_W_SIGN_GAUGE'] = sign_gauge_overlap_fixed(W_s, W_teacher, latent_axis=-1)
    results['Q_X_SIGN_GAUGE'] = sign_gauge_overlap_fixed(X_s, X_teacher, latent_axis=0)
    results['Q_W_SIGN_ALIGNED'] = results['Q_W_SIGN_GAUGE']
    results['Q_X_SIGN_ALIGNED'] = results['Q_X_SIGN_GAUGE']
    results['Q_W_COS_ROOT'] = cos_overlap_root(W_s, W_teacher, use_left=True)
    results['Q_X_COS_ROOT'] = cos_overlap_root(X_s, X_teacher, use_left=False)

    # Spreading-aware Q_Y
    _, fit_y = normalized_mse_and_fit(
        compute_sparse_Y(W_s, X_s, spreading_data.F, spreading_data.i_idx, spreading_data.j_idx)
        if W_s.dim() == 2 else torch.zeros((), device=W_s.device),
        spreading_data.Y_values,
    ) if W_s.dim() == 2 else (1.0, 0.0)
    results['Q_Y'] = compute_qy_spreading(W_student, X_student, spreading_data)
    results['Q_Y_observed'] = results['Q_Y']
    results['Q_Y_PROJ_ABS'] = compute_physical_overlap_spreading(W_student, X_student, spreading_data)
    results['Q_Y_observed_PROJ_ABS'] = results['Q_Y_PROJ_ABS']
    results['FIT_Y'] = fit_y
    results['FIT_Y_observed'] = fit_y

    return results

@torch.no_grad()
def compute_all_metrics_spreading_parallel(
    W_students: torch.Tensor,
    X_students: torch.Tensor,
    spreading_data: 'SpreadingDataParallel',
    target_alpha_idx: int = None,
    edge_chunk_size: int = 32768,
    sample_chunk_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Compute all evaluation metrics for spreading model in parallel or for a single alpha.
    
    Args:
        W_students: (S, A, N1, M) or (S, 1, N1, M) if target_alpha_idx is used
        X_students: (S, A, M, N2) or (S, 1, M, N2) if target_alpha_idx is used
        spreading_data: SpreadingDataParallel containing F and Y
        target_alpha_idx: If set, only compute metrics for this alpha index from spreading_data.
                         W_students/X_students assumed to have size 1 on axis 1.

    Returns:
        Dictionary of metrics averaged across samples.
        If target_alpha_idx is set, tensors will have size (1,) instead of (A,).
    """
    if W_students.dim() != 4 or X_students.dim() != 4:
        raise ValueError("spreading batch metrics require 4D W/X tensors")
    if W_students.shape[0] != spreading_data.S and W_students.shape[1] == spreading_data.S:
        W_students = W_students.transpose(0, 1)
        X_students = X_students.transpose(0, 1)

    S = int(W_students.shape[0])
    W_teacher = spreading_data.W_teacher
    X_teacher = spreading_data.X_teacher
    device = spreading_data.device
    alpha_values = spreading_data.alpha_values
    A = len(alpha_values)

    if target_alpha_idx is not None:
        if target_alpha_idx < 0 or target_alpha_idx >= A:
            raise ValueError(f"target_alpha_idx {target_alpha_idx} out of range [0, {A})")
        actual_alpha_indices = [int(target_alpha_idx)]
        local_alpha_indices = [0]
    else:
        actual_alpha_indices = list(range(A))
        local_alpha_indices = list(range(len(actual_alpha_indices)))

    output_A = len(actual_alpha_indices)
    W_local = W_students[:, local_alpha_indices]
    X_local = X_students[:, local_alpha_indices]

    Q_W_all = _fixed_overlap_batch(
        W_local,
        W_teacher.unsqueeze(0).unsqueeze(0),
        reduce_dims=(-2, -1),
    )
    Q_X_all = _fixed_overlap_batch(
        X_local,
        X_teacher.unsqueeze(0).unsqueeze(0),
        reduce_dims=(-2, -1),
    )
    Q_W_proj_abs_all = _projection_abs_batch(
        W_local,
        W_teacher.unsqueeze(0).unsqueeze(0),
        reduce_dims=(-2, -1),
    )
    Q_X_proj_abs_all = _projection_abs_batch(
        X_local,
        X_teacher.unsqueeze(0).unsqueeze(0),
        reduce_dims=(-2, -1),
    )
    R_W_all = _fixed_overlap_batch(W_local, W_local, reduce_dims=(-2, -1))
    R_X_all = _fixed_overlap_batch(X_local, X_local, reduce_dims=(-2, -1))
    Q_W_sign_gauge_all = _sign_gauge_overlap_fixed_batch(
        W_local,
        W_teacher.unsqueeze(0).unsqueeze(0),
        channel_sum_dim=-2,
    )
    Q_X_sign_gauge_all = _sign_gauge_overlap_fixed_batch(
        X_local,
        X_teacher.unsqueeze(0).unsqueeze(0),
        channel_sum_dim=-1,
    )

    Q_W_cos_root_all = torch.zeros(S, output_A, device=device)
    Q_X_cos_root_all = torch.zeros(S, output_A, device=device)
    Q_Y_observed_all = torch.zeros(S, output_A, device=device)
    Q_Y_unobserved_all = torch.zeros(S, output_A, device=device)
    Q_Y_full_all = torch.zeros(S, output_A, device=device)
    FIT_Y_observed_all = torch.zeros(S, output_A, device=device)
    FIT_Y_unobserved_all = torch.zeros(S, output_A, device=device)
    FIT_Y_full_all = torch.zeros(S, output_A, device=device)
    NMSE_Y_observed_all = torch.ones(S, output_A, device=device)
    NMSE_Y_unobserved_all = torch.ones(S, output_A, device=device)
    NMSE_Y_full_all = torch.ones(S, output_A, device=device)
    Q_Y_observed_proj_abs_all = torch.zeros(S, output_A, device=device)
    Q_Y_unobserved_proj_abs_all = torch.zeros(S, output_A, device=device)
    Q_Y_full_proj_abs_all = torch.zeros(S, output_A, device=device)
    Q_W_replica_all = torch.zeros(output_A, device=device)
    Q_X_replica_all = torch.zeros(output_A, device=device)
    Q_W_prime_replica_all = torch.zeros(output_A, device=device)
    Q_X_prime_replica_all = torch.zeros(output_A, device=device)

    sqrt_m_inv = 1.0 / (float(spreading_data.M) ** 0.5)
    i_idx_all = spreading_data.supergraph.i_idx.long()
    j_idx_all = spreading_data.supergraph.j_idx.long()
    F_super = spreading_data.F_super

    for out_idx, (actual_alpha_idx, local_alpha_idx) in enumerate(zip(actual_alpha_indices, local_alpha_indices)):
        C_k = spreading_data.supergraph.get_active_edges(actual_alpha_idx)
        if C_k > 0:
            i_current = i_idx_all[:, :C_k]
            j_current = j_idx_all[:, :C_k]
            F_current = F_super[:, :C_k]
            W_a = W_students[:, local_alpha_idx]
            X_a_t = X_students[:, local_alpha_idx].transpose(1, 2)

            Y_teacher_obs = spreading_data.Y_super[:, :C_k]
            dot_obs, norm_obs, sse_obs = _observed_edge_projection_sums(
                W_a,
                X_a_t,
                i_current,
                j_current,
                F_current,
                Y_teacher_obs,
                sqrt_m_inv=sqrt_m_inv,
                edge_chunk_size=edge_chunk_size,
                sample_chunk_size=sample_chunk_size,
            )
            fit_obs, nmse_obs = _fit_from_sums(sse_obs, norm_obs)
            Q_Y_observed_all[:, out_idx] = _projection_abs_from_sums(dot_obs, norm_obs)
            FIT_Y_observed_all[:, out_idx] = fit_obs
            NMSE_Y_observed_all[:, out_idx] = nmse_obs
            Q_Y_observed_proj_abs_all[:, out_idx] = Q_Y_observed_all[:, out_idx]

            N1, N2 = int(spreading_data.supergraph.N1), int(spreading_data.supergraph.N2)
            h_i_all = torch.empty(S, C_k, dtype=torch.long, device=device)
            h_j_all = torch.empty(S, C_k, dtype=torch.long, device=device)
            f_dtype = torch.float32 if str(getattr(spreading_data, "f_distribution", F_DISTRIBUTION_ISING)) == "gaussian" else torch.int8
            F_holdout = torch.empty(S, C_k, spreading_data.M, dtype=f_dtype, device=device)
            valid_samples = torch.ones(S, dtype=torch.bool, device=device)
            for s in range(S):
                seed_base = int(spreading_data.supergraph.seeds[s].item())
                h_i, h_j = _deterministic_heldout_edges(
                    N1=N1,
                    N2=N2,
                    observed_i=i_current[s],
                    observed_j=j_current[s],
                    count=int(C_k),
                    seed=seed_base + 7919 * (int(actual_alpha_idx) + 1),
                    device=device,
                )
                if h_i.numel() < C_k:
                    valid_samples[s] = False
                    if h_i.numel() == 0:
                        h_i = torch.zeros(C_k, dtype=torch.long, device=device)
                        h_j = torch.zeros(C_k, dtype=torch.long, device=device)
                    else:
                        pad = C_k - h_i.numel()
                        h_i = torch.cat([h_i, h_i[-1:].expand(pad)])
                        h_j = torch.cat([h_j, h_j[-1:].expand(pad)])
                h_i_all[s] = h_i[:C_k]
                h_j_all[s] = h_j[:C_k]
                gen = torch.Generator(device=device).manual_seed(seed_base + 104729 * (int(actual_alpha_idx) + 1))
                if str(getattr(spreading_data, "f_distribution", F_DISTRIBUTION_ISING)) == "gaussian":
                    F_holdout[s].normal_(0, 1, generator=gen)
                else:
                    F_holdout[s] = torch.randint(
                        0,
                        2,
                        (C_k, spreading_data.M),
                        generator=gen,
                        device=device,
                        dtype=torch.int8,
                    ) * 2 - 1

            dot_unobs, norm_unobs, sse_unobs = _heldout_edge_projection_sums(
                W_a,
                X_a_t,
                W_teacher,
                X_teacher.T,
                h_i_all,
                h_j_all,
                F_holdout,
                sqrt_m_inv=sqrt_m_inv,
                edge_chunk_size=edge_chunk_size,
                sample_chunk_size=sample_chunk_size,
            )
            fit_unobs, nmse_unobs = _fit_from_sums(sse_unobs, norm_unobs)
            q_unobs_proj = _projection_abs_from_sums(dot_unobs, norm_unobs)
            Q_Y_unobserved_all[:, out_idx] = torch.where(valid_samples, q_unobs_proj, torch.zeros_like(q_unobs_proj))
            FIT_Y_unobserved_all[:, out_idx] = torch.where(valid_samples, fit_unobs, torch.zeros_like(fit_unobs))
            NMSE_Y_unobserved_all[:, out_idx] = torch.where(valid_samples, nmse_unobs, torch.ones_like(nmse_unobs))
            Q_Y_unobserved_proj_abs_all[:, out_idx] = torch.where(
                valid_samples,
                q_unobs_proj,
                torch.zeros_like(q_unobs_proj),
            )
            fit_full, nmse_full = _fit_from_sums(sse_obs + sse_unobs, norm_obs + norm_unobs)
            Q_Y_full_all[:, out_idx] = _projection_abs_from_sums(dot_obs + dot_unobs, norm_obs + norm_unobs)
            FIT_Y_full_all[:, out_idx] = fit_full
            NMSE_Y_full_all[:, out_idx] = nmse_full
            Q_Y_full_proj_abs_all[:, out_idx] = Q_Y_full_all[:, out_idx]

        w_root, w_replica, w_prime = _cos_root_and_replica_by_alpha(
            W_students[:, local_alpha_idx],
            W_teacher,
            use_left=True,
        )
        x_root, x_replica, x_prime = _cos_root_and_replica_by_alpha(
            X_students[:, local_alpha_idx],
            X_teacher,
            use_left=False,
        )
        Q_W_cos_root_all[:, out_idx] = w_root
        Q_X_cos_root_all[:, out_idx] = x_root
        Q_W_replica_all[out_idx] = w_replica
        Q_X_replica_all[out_idx] = x_replica
        Q_W_prime_replica_all[out_idx] = w_prime
        Q_X_prime_replica_all[out_idx] = x_prime

    qy_mean, qy_std = _mean_std(Q_Y_full_all, dim=0)
    qy_obs_mean, qy_obs_std = _mean_std(Q_Y_observed_all, dim=0)
    qy_unobs_mean, qy_unobs_std = _mean_std(Q_Y_unobserved_all, dim=0)
    nmse_y_mean, nmse_y_std = _mean_std(NMSE_Y_full_all, dim=0)
    nmse_y_obs_mean, nmse_y_obs_std = _mean_std(NMSE_Y_observed_all, dim=0)
    nmse_y_unobs_mean, nmse_y_unobs_std = _mean_std(NMSE_Y_unobserved_all, dim=0)
    fit_y_mean, fit_y_std = _mean_std(FIT_Y_full_all, dim=0)
    fit_y_obs_mean, fit_y_obs_std = _mean_std(FIT_Y_observed_all, dim=0)
    fit_y_unobs_mean, fit_y_unobs_std = _mean_std(FIT_Y_unobserved_all, dim=0)
    qy_proj_abs_mean, qy_proj_abs_std = _mean_std(Q_Y_full_proj_abs_all, dim=0)
    qy_obs_proj_abs_mean, qy_obs_proj_abs_std = _mean_std(Q_Y_observed_proj_abs_all, dim=0)
    qy_unobs_proj_abs_mean, qy_unobs_proj_abs_std = _mean_std(Q_Y_unobserved_proj_abs_all, dim=0)
    qw_mean, qw_std = _mean_std(Q_W_all, dim=0)
    qx_mean, qx_std = _mean_std(Q_X_all, dim=0)
    qw_proj_abs_mean, qw_proj_abs_std = _mean_std(Q_W_proj_abs_all, dim=0)
    qx_proj_abs_mean, qx_proj_abs_std = _mean_std(Q_X_proj_abs_all, dim=0)
    rw_mean, rw_std = _mean_std(R_W_all, dim=0)
    rx_mean, rx_std = _mean_std(R_X_all, dim=0)
    qw_sign_gauge_mean, qw_sign_gauge_std = _mean_std(Q_W_sign_gauge_all, dim=0)
    qx_sign_gauge_mean, qx_sign_gauge_std = _mean_std(Q_X_sign_gauge_all, dim=0)
    qw_cos_root_mean, qw_cos_root_std = _mean_std(Q_W_cos_root_all, dim=0)
    qx_cos_root_mean, qx_cos_root_std = _mean_std(Q_X_cos_root_all, dim=0)
    q_w_scale_gauge_all, q_x_scale_gauge_all, q_wx_scale_gauge_all, gauge_mag_all = _scale_gauge_projection_batch(
        W_local,
        X_local,
        W_teacher,
        X_teacher,
    )
    qw_scale_gauge_mean, qw_scale_gauge_std = _mean_std(q_w_scale_gauge_all, dim=0)
    qx_scale_gauge_mean, qx_scale_gauge_std = _mean_std(q_x_scale_gauge_all, dim=0)
    qwx_scale_gauge_mean, qwx_scale_gauge_std = _mean_std(q_wx_scale_gauge_all, dim=0)
    gauge_mag_mean, gauge_mag_std = _mean_std(gauge_mag_all, dim=0)

    return {
        'Q_Y_mean': qy_mean,
        'Q_Y_std': qy_std,
        'NMSE_Y_mean': nmse_y_mean,
        'NMSE_Y_std': nmse_y_std,
        'FIT_Y_mean': fit_y_mean,
        'FIT_Y_std': fit_y_std,
        'Q_Y_observed_mean': qy_obs_mean,
        'Q_Y_observed_std': qy_obs_std,
        'NMSE_Y_observed_mean': nmse_y_obs_mean,
        'NMSE_Y_observed_std': nmse_y_obs_std,
        'FIT_Y_observed_mean': fit_y_obs_mean,
        'FIT_Y_observed_std': fit_y_obs_std,
        'Q_Y_unobserved_mean': qy_unobs_mean,
        'Q_Y_unobserved_std': qy_unobs_std,
        'NMSE_Y_unobserved_mean': nmse_y_unobs_mean,
        'NMSE_Y_unobserved_std': nmse_y_unobs_std,
        'FIT_Y_unobserved_mean': fit_y_unobs_mean,
        'FIT_Y_unobserved_std': fit_y_unobs_std,
        'Q_Y_PROJ_ABS_mean': qy_proj_abs_mean,
        'Q_Y_PROJ_ABS_std': qy_proj_abs_std,
        'Q_Y_observed_PROJ_ABS_mean': qy_obs_proj_abs_mean,
        'Q_Y_observed_PROJ_ABS_std': qy_obs_proj_abs_std,
        'Q_Y_unobserved_PROJ_ABS_mean': qy_unobs_proj_abs_mean,
        'Q_Y_unobserved_PROJ_ABS_std': qy_unobs_proj_abs_std,
        'Q_W_mean': qw_mean,
        'Q_W_std': qw_std,
        'Q_X_mean': qx_mean,
        'Q_X_std': qx_std,
        'R_W_mean': rw_mean,
        'R_W_std': rw_std,
        'R_X_mean': rx_mean,
        'R_X_std': rx_std,
        'Q_W_PROJ_ABS_mean': qw_proj_abs_mean,
        'Q_W_PROJ_ABS_std': qw_proj_abs_std,
        'Q_X_PROJ_ABS_mean': qx_proj_abs_mean,
        'Q_X_PROJ_ABS_std': qx_proj_abs_std,
        'Q_W_SIGN_GAUGE_mean': qw_sign_gauge_mean,
        'Q_W_SIGN_GAUGE_std': qw_sign_gauge_std,
        'Q_X_SIGN_GAUGE_mean': qx_sign_gauge_mean,
        'Q_X_SIGN_GAUGE_std': qx_sign_gauge_std,
        'Q_W_SIGN_ALIGNED_mean': qw_sign_gauge_mean,
        'Q_W_SIGN_ALIGNED_std': qw_sign_gauge_std,
        'Q_X_SIGN_ALIGNED_mean': qx_sign_gauge_mean,
        'Q_X_SIGN_ALIGNED_std': qx_sign_gauge_std,
        'Q_W_COS_ROOT_mean': qw_cos_root_mean,
        'Q_W_COS_ROOT_std': qw_cos_root_std,
        'Q_X_COS_ROOT_mean': qx_cos_root_mean,
        'Q_X_COS_ROOT_std': qx_cos_root_std,
        'Q_W_SCALE_GAUGE_mean': qw_scale_gauge_mean,
        'Q_W_SCALE_GAUGE_std': qw_scale_gauge_std,
        'Q_X_SCALE_GAUGE_mean': qx_scale_gauge_mean,
        'Q_X_SCALE_GAUGE_std': qx_scale_gauge_std,
        'Q_WX_SCALE_GAUGE_mean': qwx_scale_gauge_mean,
        'Q_WX_SCALE_GAUGE_std': qwx_scale_gauge_std,
        'median_abs_log_k_mean': gauge_mag_mean,
        'median_abs_log_k_std': gauge_mag_std,
        'median_abs_log_g_mean': gauge_mag_mean,
        'median_abs_log_g_std': gauge_mag_std,
        'Q_W_replica_mean': Q_W_replica_all,
        'Q_X_replica_mean': Q_X_replica_all,
        'Q_W_prime_replica_mean': Q_W_prime_replica_all,
        'Q_X_prime_replica_mean': Q_X_prime_replica_all,
        'alpha_values': alpha_values[actual_alpha_indices],
    }


@torch.no_grad()
def compute_qy_with_wrong_f(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    spreading_data: SpreadingData,
    wrong_seed: int = 99999,
) -> float:
    """
    Compute Q_Y using WRONG F coefficients (for testing purposes).

    This demonstrates that using different F gives low Q_Y
    even when W_student = W_teacher and X_student = X_teacher.

    Args:
        W_student: Student W matrix
        X_student: Student X matrix
        spreading_data: Original SpreadingData (used for indices and teacher Y)
        wrong_seed: Seed for generating wrong F

    Returns:
        Q_Y computed with wrong F (should be much lower than with correct F)
    """
    from ..teachers.random_spreading import generate_spreading_coefficients

    # Generate different F
    wrong_F = generate_spreading_coefficients(
        spreading_data.i_idx,
        spreading_data.j_idx,
        spreading_data.M,
        wrong_seed,
        spreading_data.F.device,
    )

    # Compute Y_student with wrong F
    Y_student_wrong = compute_sparse_Y(
        W_student, X_student,
        wrong_F,  # Wrong F!
        spreading_data.i_idx,
        spreading_data.j_idx,
    )

    # Compare with teacher Y (computed with correct F)
    Y_teacher_values = spreading_data.Y_values

    # Cosine similarity
    dot = (Y_student_wrong * Y_teacher_values).sum()
    norm_s = Y_student_wrong.norm()
    norm_t = Y_teacher_values.norm()

    return float(dot / (norm_s * norm_t + 1e-12))
