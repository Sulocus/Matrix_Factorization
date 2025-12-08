"""
Evaluation metrics for random spreading model.

Key difference from standard metrics:
- Q_Y must use the same F coefficients for both teacher and student
- Otherwise, even perfect W, X recovery gives Q_Y << 1

Q_W, Q_X, Q_W', Q_X' are unchanged (Gram matrix comparisons).
"""

from typing import Dict
import torch

from ..teachers.random_spreading import SpreadingData, compute_sparse_Y


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
