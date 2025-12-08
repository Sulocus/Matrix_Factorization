"""
Overlap metrics for evaluating student-teacher similarity.
"""

from typing import Dict, List, Optional
import numpy as np
import torch


@torch.no_grad()
def gram_overlap_cosine(A: torch.Tensor, B: torch.Tensor, use_left: bool = True) -> float:
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
    q = gram_overlap_cosine(A, B, use_left)

    if use_left:
        n, m = A.shape
    else:
        n, m = A.shape[1], A.shape[0]

    # Baseline: expected cosine for random matrices
    b = m / (m + n + 1)
    qc = (q - b) / (1.0 - b + 1e-12)

    return float(max(0.0, min(1.0, qc)))


@torch.no_grad()
def compute_qy(Y_student: torch.Tensor, Y_teacher: torch.Tensor) -> float:
    """
    Compute Y-space overlap (rotationally invariant).

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
    """Compute mean squared error between Y matrices."""
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
            Valid names: Q_W, Q_X, Q_W_prime, Q_X_prime, Q_Y, Gen_Error,
                        Q_Y_unobserved, Q_Y_observed

    Returns:
        Dictionary with requested metrics
    """
    # Default: all standard metrics
    if metrics_to_compute is None:
        metrics_to_compute = ['Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Q_Y', 'Gen_Error']

    if Y_teacher is None:
        Y_teacher = W_teacher @ X_teacher

    Y_student = W_student @ X_student

    results = {}

    # Compute requested metrics dynamically
    if 'Q_W' in metrics_to_compute:
        results['Q_W'] = gram_overlap_cosine(W_student, W_teacher, use_left=True)

    if 'Q_X' in metrics_to_compute:
        results['Q_X'] = gram_overlap_cosine(X_student, X_teacher, use_left=False)

    if 'Q_W_prime' in metrics_to_compute:
        results['Q_W_prime'] = gram_overlap_normalized(W_student, W_teacher, use_left=True)

    if 'Q_X_prime' in metrics_to_compute:
        results['Q_X_prime'] = gram_overlap_normalized(X_student, X_teacher, use_left=False)

    if 'Q_Y' in metrics_to_compute:
        results['Q_Y'] = compute_qy(Y_student, Y_teacher)

    if 'Gen_Error' in metrics_to_compute:
        results['Gen_Error'] = compute_generalization_error(Y_student, Y_teacher)

    # Q_Y_unobserved and Q_Y_observed require mask
    if mask is not None:
        if 'Q_Y_unobserved' in metrics_to_compute:
            results['Q_Y_unobserved'] = _compute_qy_masked(Y_student, Y_teacher, mask, observed=False)

        if 'Q_Y_observed' in metrics_to_compute:
            results['Q_Y_observed'] = _compute_qy_masked(Y_student, Y_teacher, mask, observed=True)

    return results


@torch.no_grad()
def _compute_qy_masked(
    Y_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    mask: torch.Tensor,
    observed: bool = False,
) -> float:
    """
    Compute Q_Y on observed or unobserved positions.

    Args:
        Y_student: Student reconstruction
        Y_teacher: Teacher Y matrix
        mask: Observation mask (1 = observed, 0 = unobserved)
        observed: If True, compute on observed positions; else unobserved

    Returns:
        Cosine similarity on the selected positions
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

    dot = (y_s * y_t).sum()
    norm_s = y_s.norm()
    norm_t = y_t.norm()

    return float(dot / (norm_s * norm_t + 1e-12))


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

    for i in range(S):
        for j in range(i + 1, S):
            Q_W_list.append(gram_overlap_cosine(W_all[i], W_all[j], use_left=True))
            Q_X_list.append(gram_overlap_cosine(X_all[i], X_all[j], use_left=False))

    return {
        'Q_W_replica_mean': float(np.mean(Q_W_list)),
        'Q_W_replica_std': float(np.std(Q_W_list, ddof=1)) if len(Q_W_list) > 1 else 0.0,
        'Q_X_replica_mean': float(np.mean(Q_X_list)),
        'Q_X_replica_std': float(np.std(Q_X_list, ddof=1)) if len(Q_X_list) > 1 else 0.0,
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
        aggregated[f'{key}_mean'] = float(np.mean(vals))
        aggregated[f'{key}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    return aggregated
