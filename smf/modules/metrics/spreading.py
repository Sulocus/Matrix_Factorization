"""
Evaluation metrics for random spreading model.

Key difference from standard metrics:
- Q_Y must use the same F coefficients for both teacher and student
- Otherwise, even perfect W, X recovery gives Q_Y << 1

Q_W, Q_X, Q_W', Q_X' are unchanged (Gram matrix comparisons).
"""

from typing import Dict, TYPE_CHECKING
import torch

from ..teachers.random_spreading import SpreadingData, compute_sparse_Y

if TYPE_CHECKING:
    from ..teachers.random_spreading import SpreadingDataParallel


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

    Q_Y = cos(Y_student, Y_teacher) = dot / (||Y_s|| × ||Y_t||)

    Args:
        W_student: (N1, M) or (S, N1, M) student W matrix
        X_student: (M, N2) or (S, M, N2) student X matrix
        spreading_data: SpreadingData with F and teacher Y_values

    Returns:
        Cosine similarity in [0, 1] range (approximately)

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

    # Cosine similarity
    dot = (Y_student_values * Y_teacher_values).sum()
    norm_s = Y_student_values.norm()
    norm_t = Y_teacher_values.norm()

    return float(dot / (norm_s * norm_t + 1e-12))


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

    dot = (Y_student_values * Y_teacher_values).sum()
    norm_t_sq = (Y_teacher_values ** 2).sum()

    return float(dot / (norm_t_sq + 1e-12))


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
    - Q_Y: Y-space overlap (using spreading-aware computation)
    - Q_W, Q_X: Gram matrix cosine similarity
    - Q_W', Q_X': Normalized Gram overlap [0, 1]
    - MSE: Mean squared error at observed positions

    Args:
        W_student: (N1, M) or (S, N1, M) student W
        X_student: (M, N2) or (S, M, N2) student X
        W_teacher: (N1, M) teacher W
        X_teacher: (M, N2) teacher X
        spreading_data: SpreadingData with F and Y_values

    Returns:
        Dictionary with all metrics
    """
    from .overlap import compute_cosine_similarity, gram_overlap_normalized

    results = {}

    # Handle batch dimension for W_student, X_student
    if W_student.dim() == 3:
        W_s = W_student.mean(dim=0)  # Average over samples
        X_s = X_student.mean(dim=0)
    else:
        W_s = W_student
        X_s = X_student

    # Standard Gram overlaps (rotation-invariant)
    results['Q_W'] = compute_cosine_similarity(W_s, W_teacher, use_left=True)
    results['Q_X'] = compute_cosine_similarity(X_s, X_teacher, use_left=False)
    results['Q_W_prime'] = gram_overlap_normalized(W_s, W_teacher, use_left=True)
    results['Q_X_prime'] = gram_overlap_normalized(X_s, X_teacher, use_left=False)

    # Spreading-aware Q_Y
    results['Q_Y'] = compute_qy_spreading(W_student, X_student, spreading_data)
    results['physical_overlap_Y'] = compute_physical_overlap_spreading(W_student, X_student, spreading_data)

    # MSE
    results['MSE'] = compute_mse_spreading(W_student, X_student, spreading_data)

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
    from .overlap import compute_cosine_similarity, gram_overlap_normalized
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
    Physical_Y_all = torch.zeros(S, output_A, device=device)
    Q_W_all = torch.zeros(S, output_A, device=device)
    Q_X_all = torch.zeros(S, output_A, device=device)
    Q_W_prime_all = torch.zeros(S, output_A, device=device)
    Q_X_prime_all = torch.zeros(S, output_A, device=device)
    Physical_W_all = torch.zeros(S, output_A, device=device)
    Physical_X_all = torch.zeros(S, output_A, device=device)
    
    Q_Y_observed_all = torch.zeros(S, output_A, device=device)
    Q_Y_unobserved_all = torch.zeros(S, output_A, device=device)
    Q_Y_total_all = torch.zeros(S, output_A, device=device)
    Physical_Y_total_all = torch.zeros(S, output_A, device=device)
    MSE_all = torch.zeros(S, output_A, device=device)

    # 计算教师的完整 Y 矩阵（标准乘法，F=1）
    Y_teacher_full = W_teacher @ X_teacher  # (N1, N2)
    y_t_sq = Y_teacher_full ** 2 + 1e-12

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
            
            # Cosine Q_Y (Observed)
            dot = (Y_student_obs * Y_teacher_obs).sum()
            norm_s = Y_student_obs.norm()
            norm_t = Y_teacher_obs.norm()
            qy_obs = dot / (norm_s * norm_t + 1e-12)
            
            Q_Y_observed_all[s, out_idx] = qy_obs
            
            # Physical Overlap (Observed) - Not fully correct for spreading but kept for compat
            norm_t_sq_obs = (Y_teacher_obs ** 2).sum()
            Physical_Y_all[s, out_idx] = dot / (norm_t_sq_obs + 1e-12)

            # --- 2. Overlap Metrics (Q_W, Q_X, etc) ---
            Q_W_all[s, out_idx] = compute_cosine_similarity(W_s, W_teacher, use_left=True)
            Q_X_all[s, out_idx] = compute_cosine_similarity(X_s, X_teacher, use_left=False)
            Q_W_prime_all[s, out_idx] = gram_overlap_normalized(W_s, W_teacher, use_left=True)
            Q_X_prime_all[s, out_idx] = gram_overlap_normalized(X_s, X_teacher, use_left=False)
            
            # Physical Overlap (Projection coefficient: <A,B> / ||B||^2)
            w_dot = (W_s * W_teacher).sum()
            w_norm_sq = (W_teacher ** 2).sum() + 1e-12
            Physical_W_all[s, out_idx] = w_dot.abs() / w_norm_sq  # abs for sign ambiguity
            
            x_dot = (X_s * X_teacher).sum()
            x_norm_sq = (X_teacher ** 2).sum() + 1e-12
            Physical_X_all[s, out_idx] = x_dot.abs() / x_norm_sq

            # --- 3. Full Matrix Metrics (Q_Y_total, Physical_Total, Unobserved) ---
            # Student full Y
            Y_student_full = W_s @ X_s
            
            # a) Full Matrix Cosine (Q_Y_total)
            y_t_flat = Y_teacher_full.flatten()
            y_s_flat = Y_student_full.flatten()
            dot_full = (y_t_flat * y_s_flat).sum()
                
            Q_Y_total_all[s, out_idx] = dot_full / (y_t_flat.norm() * y_s_flat.norm() + 1e-12)
            
            # b) Physical Overlap Mean (average of point-wise overlaps)
            overlap_pointwise = (Y_student_full * Y_teacher_full) / y_t_sq
            Physical_Y_total_all[s, out_idx] = overlap_pointwise.mean()

            # c) MSE
            MSE_all[s, out_idx] = (y_t_flat - y_s_flat).pow(2).mean()
            
            # c) Unobserved
            # Create mask for observed
            N1, N2 = Y_teacher_full.shape
            observed_mask = torch.zeros(N1, N2, dtype=torch.bool, device=device)
            observed_mask[i_current.long(), j_current.long()] = True 
            
            y_t_unobs = Y_teacher_full[~observed_mask]
            y_s_unobs = Y_student_full[~observed_mask]
            
            if y_t_unobs.numel() > 0:
                dot_u = (y_t_unobs * y_s_unobs).sum()
                Q_Y_unobserved_all[s, out_idx] = dot_u / (y_t_unobs.norm() * y_s_unobs.norm() + 1e-12)
            else:
                Q_Y_unobserved_all[s, out_idx] = 1.0

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
        'Q_Y_mean': Q_Y_total_all.mean(dim=0),  # Full Cosine
        'Q_Y_std': Q_Y_total_all.std(dim=0),
        'Q_Y_observed_mean': Q_Y_observed_all.mean(dim=0),
        'Q_Y_observed_std': Q_Y_observed_all.std(dim=0),
        'Q_Y_unobserved_mean': Q_Y_unobserved_all.mean(dim=0),
        'Q_Y_unobserved_std': Q_Y_unobserved_all.std(dim=0),
        'physical_overlap_Y_mean': Physical_Y_total_all.mean(dim=0),
        'physical_overlap_Y_std': Physical_Y_total_all.std(dim=0),
        'Q_W_mean': Q_W_all.mean(dim=0),
        'Q_W_std': Q_W_all.std(dim=0),
        'Q_X_mean': Q_X_all.mean(dim=0),
        'Q_X_std': Q_X_all.std(dim=0),
        'Q_W_prime_mean': Q_W_prime_all.mean(dim=0),
        'Q_W_prime_std': Q_W_prime_all.std(dim=0),
        'Q_X_prime_mean': Q_X_prime_all.mean(dim=0),
        'Q_X_prime_std': Q_X_prime_all.std(dim=0),
        'physical_overlap_W_mean': Physical_W_all.mean(dim=0),
        'physical_overlap_W_std': Physical_W_all.std(dim=0),
        'physical_overlap_X_mean': Physical_X_all.mean(dim=0),
        'physical_overlap_X_std': Physical_X_all.std(dim=0),
        'MSE': MSE_all.mean(dim=0),
        'MSE_std': MSE_all.std(dim=0),
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
