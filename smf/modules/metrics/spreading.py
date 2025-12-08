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
    from .overlap import gram_overlap_cosine, gram_overlap_normalized

    results = {}

    # Handle batch dimension for W_student, X_student
    if W_student.dim() == 3:
        W_s = W_student.mean(dim=0)  # Average over samples
        X_s = X_student.mean(dim=0)
    else:
        W_s = W_student
        X_s = X_student

    # Standard Gram overlaps (rotation-invariant)
    results['Q_W'] = gram_overlap_cosine(W_s, W_teacher, use_left=True)
    results['Q_X'] = gram_overlap_cosine(X_s, X_teacher, use_left=False)
    results['Q_W_prime'] = gram_overlap_normalized(W_s, W_teacher, use_left=True)
    results['Q_X_prime'] = gram_overlap_normalized(X_s, X_teacher, use_left=False)

    # Spreading-aware Q_Y
    results['Q_Y'] = compute_qy_spreading(W_student, X_student, spreading_data)

    # MSE
    results['MSE'] = compute_mse_spreading(W_student, X_student, spreading_data)

    return results


@torch.no_grad()
def compute_qy_spreading_parallel(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    spreading_data: 'SpreadingDataParallel',
    sample_idx: int,
) -> torch.Tensor:
    """
    Compute Q_Y for all alphas of a single sample (parallel version).

    Args:
        W_student: (A, N1, M) student W for all alphas
        X_student: (A, M, N2) student X for all alphas
        spreading_data: SpreadingDataParallel
        sample_idx: Which sample index

    Returns:
        Q_Y: (A,) Q_Y for each alpha
    """
    from ..algorithms.bigamp_spreading_parallel import forward_pass_parallel

    A = W_student.shape[0]
    device = W_student.device

    # Get sample-specific data
    F = spreading_data.get_F(sample_idx)  # (C_max, M)
    Y_teacher = spreading_data.Y_super[sample_idx]  # (C_max,)
    i_idx, j_idx = spreading_data.supergraph.get_sample_indices(sample_idx)
    alpha_mask = spreading_data.supergraph.alpha_mask  # (A, C_max)

    # Compute student Y for all alphas
    Y_student = forward_pass_parallel(W_student, X_student, F, i_idx, j_idx, alpha_mask)  # (A, C_max)

    # Compute Q_Y for each alpha
    Q_Y = torch.zeros(A, device=device)

    for a in range(A):
        C_k = spreading_data.supergraph.get_active_edges(a)
        if C_k == 0:
            Q_Y[a] = 0.0
            continue

        y_t = Y_teacher[:C_k]
        y_s = Y_student[a, :C_k]

        dot = (y_t * y_s).sum()
        norm_t = y_t.norm()
        norm_s = y_s.norm()

        Q_Y[a] = dot / (norm_t * norm_s + 1e-12)

    return Q_Y


@torch.no_grad()
def compute_all_metrics_spreading_parallel(
    W_students: torch.Tensor,
    X_students: torch.Tensor,
    spreading_data: 'SpreadingDataParallel',
) -> Dict[str, torch.Tensor]:
    """
    Compute all metrics for parallel spreading model.

    Args:
        W_students: (S, A, N1, M) student W for all samples and alphas
        X_students: (S, A, M, N2) student X
        spreading_data: SpreadingDataParallel

    Returns:
        Dictionary with metrics, each value is (A,) tensor for each alpha:
        - Q_Y_mean, Q_Y_std
        - Q_W_mean, Q_W_std (based on teacher comparison)
        - Q_X_mean, Q_X_std
    """
    from .overlap import gram_overlap_cosine

    S, A = W_students.shape[:2]
    device = W_students.device
    W_teacher = spreading_data.W_teacher
    X_teacher = spreading_data.X_teacher

    # Collect Q_Y for each (sample, alpha)
    Q_Y_all = torch.zeros(S, A, device=device)
    Q_W_all = torch.zeros(S, A, device=device)
    Q_X_all = torch.zeros(S, A, device=device)

    for s in range(S):
        # Q_Y for this sample
        Q_Y_all[s] = compute_qy_spreading_parallel(
            W_students[s], X_students[s], spreading_data, s
        )

        # Q_W, Q_X for each alpha
        for a in range(A):
            Q_W_all[s, a] = gram_overlap_cosine(
                W_students[s, a], W_teacher, use_left=True
            )
            Q_X_all[s, a] = gram_overlap_cosine(
                X_students[s, a], X_teacher, use_left=False
            )

    # Aggregate across samples
    results = {
        'Q_Y_mean': Q_Y_all.mean(dim=0),  # (A,)
        'Q_Y_std': Q_Y_all.std(dim=0),
        'Q_W_mean': Q_W_all.mean(dim=0),
        'Q_W_std': Q_W_all.std(dim=0),
        'Q_X_mean': Q_X_all.mean(dim=0),
        'Q_X_std': Q_X_all.std(dim=0),
        'alpha_values': spreading_data.alpha_values,
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
