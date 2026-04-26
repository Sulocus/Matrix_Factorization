"""
Evaluation metrics for random spreading model.

Key difference from standard metrics:
- Q_Y uses the same F coefficients for both teacher and student.
- Formal Q_Y is absolute projection, not cosine or reconstruction error.
- Q_W/Q_X are coordinate projection overlaps; Gram-root metrics are diagnostics.
"""

from typing import Dict, TYPE_CHECKING
import torch

from ..teachers.random_spreading import SpreadingData, compute_sparse_Y

if TYPE_CHECKING:
    from ..teachers.random_spreading import SpreadingDataParallel


@torch.no_grad()
def _projection_abs_values(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    norm_teacher_sq = (teacher.flatten() ** 2).sum()
    if float(norm_teacher_sq.abs().item()) < 1e-12:
        return torch.zeros((), device=student.device, dtype=torch.float32)
    return ((student.flatten() * teacher.flatten()).sum().abs() / (norm_teacher_sq + 1e-12)).float()


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

    Q_Y = |<Y_student, Y_teacher>| / <Y_teacher, Y_teacher>

    Args:
        W_student: (N1, M) or (S, N1, M) student W matrix
        X_student: (M, N2) or (S, M, N2) student X matrix
        spreading_data: SpreadingData with F and teacher Y_values

    Returns:
        Absolute projection overlap, not clipped.

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
    - Q_W_GRAM_ROOT, Q_X_GRAM_ROOT: Gram-root diagnostics

    Args:
        W_student: (N1, M) or (S, N1, M) student W
        X_student: (M, N2) or (S, M, N2) student X
        W_teacher: (N1, M) teacher W
        X_teacher: (M, N2) teacher X
        spreading_data: SpreadingData with F and Y_values

    Returns:
        Dictionary with all metrics
    """
    from .overlap import gram_overlap_root, projection_abs

    results = {}

    # Handle batch dimension for W_student, X_student
    if W_student.dim() == 3:
        W_s = W_student.mean(dim=0)  # Average over samples
        X_s = X_student.mean(dim=0)
    else:
        W_s = W_student
        X_s = X_student

    results['Q_W'] = projection_abs(W_s, W_teacher)
    results['Q_X'] = projection_abs(X_s, X_teacher)
    results['Q_W_GRAM_ROOT'] = gram_overlap_root(W_s, W_teacher, use_left=True)
    results['Q_X_GRAM_ROOT'] = gram_overlap_root(X_s, X_teacher, use_left=False)

    # Spreading-aware Q_Y
    results['Q_Y'] = compute_qy_spreading(W_student, X_student, spreading_data)
    results['Q_Y_observed'] = results['Q_Y']

    return results

@torch.no_grad()
def compute_all_metrics_spreading_parallel(
    W_students: torch.Tensor,
    X_students: torch.Tensor,
    spreading_data: 'SpreadingDataParallel',
    target_alpha_idx: int = None,
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
    from .overlap import compute_cosine_similarity, gram_overlap_normalized, gram_overlap_root, projection_abs
    from ..teachers.random_spreading import compute_sparse_Y

    S = W_students.shape[0]
    W_teacher = spreading_data.W_teacher
    X_teacher = spreading_data.X_teacher
    device = spreading_data.device

    alpha_values = spreading_data.alpha_values
    A = len(alpha_values)
    
    # Determine loop range and tensor access
    if target_alpha_idx is not None:
        if target_alpha_idx < 0 or target_alpha_idx >= A:
            raise ValueError(f"target_alpha_idx {target_alpha_idx} out of range [0, {A})")
        
        loop_indices = [target_alpha_idx]
        output_A = 1
        # If target_alpha_idx is used, we assume W_students is (S, 1, N, M) 
        # so we always access index 0.
        w_idx_map = {target_alpha_idx: 0}
    else:
        loop_indices = range(A)
        output_A = A
        # Normal case: W_students is (S, A, N, M), access index a
        w_idx_map = {a: a for a in range(A)}

    # Initialize result tensors
    Q_Y_all = torch.zeros(S, output_A, device=device)
    Q_W_all = torch.zeros(S, output_A, device=device)
    Q_X_all = torch.zeros(S, output_A, device=device)
    Q_W_gram_root_all = torch.zeros(S, output_A, device=device)
    Q_X_gram_root_all = torch.zeros(S, output_A, device=device)
    
    Q_Y_observed_all = torch.zeros(S, output_A, device=device)
    Q_Y_unobserved_all = torch.zeros(S, output_A, device=device)
    Q_Y_full_all = torch.zeros(S, output_A, device=device)

    for s in range(S):
        # Pre-calculate indices for this sample
        s_i_idx, s_j_idx = spreading_data.supergraph.get_sample_indices(s)
        F_sample = spreading_data.get_F(s)
        
        for out_idx, a in enumerate(loop_indices):
            # Access W/X using mapped index
            w_idx = w_idx_map[a]
            
            W_s = W_students[s, w_idx]
            X_s = X_students[s, w_idx]
            
            # --- 1. Compute Q_Y (observed / spreading) ---
            # Use same F as teacher for observed positions
            C_k = spreading_data.supergraph.get_active_edges(a)
            
            # Indices and F for active edges
            i_current = s_i_idx[:C_k]
            j_current = s_j_idx[:C_k]
            F_current = F_sample[:C_k]
            
            # Teacher Y at observed positions
            Y_teacher_obs = spreading_data.get_Y_masked(s, a)
            
            # Student Y at observed positions (with same F)
            Y_student_obs = compute_sparse_Y(
                W_s, X_s, F_current, i_current.long(), j_current.long()
            )
            
            qy_obs = _projection_abs_values(Y_student_obs, Y_teacher_obs)
            Q_Y_observed_all[s, out_idx] = qy_obs

            # --- 2. Overlap Metrics (Q_W, Q_X, etc) ---
            Q_W_all[s, out_idx] = projection_abs(W_s, W_teacher)
            Q_X_all[s, out_idx] = projection_abs(X_s, X_teacher)
            Q_W_gram_root_all[s, out_idx] = gram_overlap_root(W_s, W_teacher, use_left=True)
            Q_X_gram_root_all[s, out_idx] = gram_overlap_root(X_s, X_teacher, use_left=False)

            # --- 3. Heldout F-aware measurement projection ---
            N1, N2 = int(spreading_data.supergraph.N1), int(spreading_data.supergraph.N2)
            h_i, h_j = _deterministic_heldout_edges(
                N1=N1,
                N2=N2,
                observed_i=i_current.long(),
                observed_j=j_current.long(),
                count=int(C_k),
                seed=int(spreading_data.supergraph.seeds[s].item()) + 7919 * (int(a) + 1),
                device=device,
            )
            if h_i.numel() > 0:
                gen = torch.Generator(device=device).manual_seed(
                    int(spreading_data.supergraph.seeds[s].item()) + 104729 * (int(a) + 1)
                )
                if str(getattr(spreading_data, "f_distribution", "rademacher")) == "gaussian":
                    F_holdout = torch.randn(h_i.numel(), spreading_data.M, generator=gen, device=device)
                else:
                    F_holdout = (torch.randint(0, 2, (h_i.numel(), spreading_data.M), generator=gen, device=device, dtype=torch.int8) * 2 - 1).float()
                Y_teacher_holdout = compute_sparse_Y(W_teacher, X_teacher, F_holdout, h_i.long(), h_j.long())
                Y_student_holdout = compute_sparse_Y(W_s, X_s, F_holdout, h_i.long(), h_j.long())
                Q_Y_unobserved_all[s, out_idx] = _projection_abs_values(Y_student_holdout, Y_teacher_holdout)
                Q_Y_full_all[s, out_idx] = _projection_abs_values(
                    torch.cat([Y_student_obs.flatten(), Y_student_holdout.flatten()]),
                    torch.cat([Y_teacher_obs.flatten(), Y_teacher_holdout.flatten()]),
                )
            else:
                Q_Y_unobserved_all[s, out_idx] = 0.0
                Q_Y_full_all[s, out_idx] = qy_obs

    # ===== Replica metrics (student-student) =====
    Q_W_replica_all = torch.zeros(output_A, device=device)
    Q_X_replica_all = torch.zeros(output_A, device=device)
    Q_W_prime_replica_all = torch.zeros(output_A, device=device)
    Q_X_prime_replica_all = torch.zeros(output_A, device=device)
    
    if S >= 2:
        for out_idx, a in enumerate(loop_indices):
            w_idx = w_idx_map[a]
            w_pairs, x_pairs, wp_pairs, xp_pairs = [], [], [], []
            for i in range(S):
                for j in range(i+1, S):
                    ws_i, ws_j = W_students[i, w_idx], W_students[j, w_idx]
                    xs_i, xs_j = X_students[i, w_idx], X_students[j, w_idx]
                    
                    w_pairs.append(compute_cosine_similarity(ws_i, ws_j, use_left=True))
                    x_pairs.append(compute_cosine_similarity(xs_i, xs_j, use_left=False))
                    wp_pairs.append(gram_overlap_normalized(ws_i, ws_j, use_left=True))
                    xp_pairs.append(gram_overlap_normalized(xs_i, xs_j, use_left=False))
            
            Q_W_replica_all[out_idx] = sum(w_pairs) / len(w_pairs)
            Q_X_replica_all[out_idx] = sum(x_pairs) / len(x_pairs)
            Q_W_prime_replica_all[out_idx] = sum(wp_pairs) / len(wp_pairs)
            Q_X_prime_replica_all[out_idx] = sum(xp_pairs) / len(xp_pairs)

    # Aggregate across samples
    results = {
        'Q_Y_mean': Q_Y_full_all.mean(dim=0),
        'Q_Y_std': Q_Y_full_all.std(dim=0),
        'Q_Y_observed_mean': Q_Y_observed_all.mean(dim=0),
        'Q_Y_observed_std': Q_Y_observed_all.std(dim=0),
        'Q_Y_unobserved_mean': Q_Y_unobserved_all.mean(dim=0),
        'Q_Y_unobserved_std': Q_Y_unobserved_all.std(dim=0),
        'Q_W_mean': Q_W_all.mean(dim=0),
        'Q_W_std': Q_W_all.std(dim=0),
        'Q_X_mean': Q_X_all.mean(dim=0),
        'Q_X_std': Q_X_all.std(dim=0),
        'Q_W_GRAM_ROOT_mean': Q_W_gram_root_all.mean(dim=0),
        'Q_W_GRAM_ROOT_std': Q_W_gram_root_all.std(dim=0),
        'Q_X_GRAM_ROOT_mean': Q_X_gram_root_all.mean(dim=0),
        'Q_X_GRAM_ROOT_std': Q_X_gram_root_all.std(dim=0),
        'Q_W_replica_mean': Q_W_replica_all,
        'Q_X_replica_mean': Q_X_replica_all,
        'Q_W_prime_replica_mean': Q_W_prime_replica_all,
        'Q_X_prime_replica_mean': Q_X_prime_replica_all,
        'alpha_values': alpha_values[list(loop_indices)] if target_alpha_idx is not None else alpha_values,
    }

    return results


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
