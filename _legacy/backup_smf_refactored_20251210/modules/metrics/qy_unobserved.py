"""
Q_Y metric computed only on unobserved positions.

This metric evaluates reconstruction quality on positions where
the student had no direct observation, providing a better measure
of true generalization ability.
"""

import torch


@torch.no_grad()
def compute_qy_unobserved(
    Y_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute Q_Y overlap only on unobserved positions (where mask=0).

    This metric measures how well the student reconstructs Y in regions
    it never directly observed during training, providing a true measure
    of generalization.

    Args:
        Y_student: Student's reconstructed Y = W @ X, shape (N1, N2)
        Y_teacher: Teacher's true Y = W_t @ X_t, shape (N1, N2)
        mask: Observation mask, shape (N1, N2). 1 = observed, 0 = unobserved

    Returns:
        Cosine similarity on unobserved positions, or 0.0 if no unobserved
        positions or norms are too small.

    Example:
        >>> mask = torch.randint(0, 2, (100, 100))  # Binary mask
        >>> Q_Y_unobs = compute_qy_unobserved(Y_student, Y_teacher, mask)
    """
    # Get unobserved mask (1 where mask=0)
    unobs_mask = 1.0 - mask.float()

    # Apply mask to both Y matrices
    Y_teacher_unobs = Y_teacher * unobs_mask
    Y_student_unobs = Y_student * unobs_mask

    # Compute norms
    norm_teacher = Y_teacher_unobs.norm()
    norm_student = Y_student_unobs.norm()

    # Check for valid norms
    if norm_teacher < 1e-12 or norm_student < 1e-12:
        return 0.0

    # Compute cosine similarity
    dot_product = (Y_teacher_unobs.flatten() * Y_student_unobs.flatten()).sum()
    cosine = dot_product / (norm_teacher * norm_student)

    return float(cosine)


@torch.no_grad()
def compute_qy_observed(
    Y_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute Q_Y overlap only on observed positions (where mask=1).

    This is the complement of Q_Y_unobserved, measuring reconstruction
    quality on the training data positions.

    Args:
        Y_student: Student's reconstructed Y = W @ X
        Y_teacher: Teacher's true Y
        mask: Observation mask (1 = observed, 0 = unobserved)

    Returns:
        Cosine similarity on observed positions.
    """
    obs_mask = mask.float()

    Y_teacher_obs = Y_teacher * obs_mask
    Y_student_obs = Y_student * obs_mask

    norm_teacher = Y_teacher_obs.norm()
    norm_student = Y_student_obs.norm()

    if norm_teacher < 1e-12 or norm_student < 1e-12:
        return 0.0

    dot_product = (Y_teacher_obs.flatten() * Y_student_obs.flatten()).sum()
    cosine = dot_product / (norm_teacher * norm_student)

    return float(cosine)


@torch.no_grad()
def compute_physical_overlap_unobserved(
    Y_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute Physical Overlap on unobserved positions.
    Overlap = <Y_s, Y_t> / <Y_t, Y_t>
    """
    unobs_mask = 1.0 - mask.float()
    Y_teacher_unobs = Y_teacher * unobs_mask
    Y_student_unobs = Y_student * unobs_mask

    norm_teacher_sq = (Y_teacher_unobs ** 2).sum()
    if norm_teacher_sq < 1e-12:
        return 0.0

    dot_product = (Y_teacher_unobs.flatten() * Y_student_unobs.flatten()).sum()
    return float(dot_product / norm_teacher_sq)


@torch.no_grad()
def compute_physical_overlap_observed(
    Y_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute Physical Overlap on observed positions.
    """
    obs_mask = mask.float()
    Y_teacher_obs = Y_teacher * obs_mask
    Y_student_obs = Y_student * obs_mask

    norm_teacher_sq = (Y_teacher_obs ** 2).sum()
    if norm_teacher_sq < 1e-12:
        return 0.0

    dot_product = (Y_teacher_obs.flatten() * Y_student_obs.flatten()).sum()
    return float(dot_product / norm_teacher_sq)


@torch.no_grad()
def compute_qy_split(
    Y_student: torch.Tensor,
    Y_teacher: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """
    Compute Q_Y metrics split by observed/unobserved positions.

    Provides a comprehensive view of reconstruction quality on both
    the training region and the held-out region.

    Args:
        Y_student: Student's reconstructed Y = W @ X
        Y_teacher: Teacher's true Y
        mask: Observation mask (1 = observed, 0 = unobserved)

    Returns:
        Dictionary with:
        - Q_Y_observed: overlap on observed positions
        - Q_Y_unobserved: overlap on unobserved positions
        - Q_Y_full: overlap on full matrix (for comparison)
        - observed_ratio: fraction of observed positions
    """
    y_s_flat = Y_student.flatten()
    y_t_flat = Y_teacher.flatten()

    # Full Q_Y
    Q_Y_full = float(
        (y_s_flat * y_t_flat).sum() /
        (y_s_flat.norm() * y_t_flat.norm() + 1e-12)
    )

    return {
        'Q_Y_observed': compute_qy_observed(Y_student, Y_teacher, mask),
        'Q_Y_unobserved': compute_qy_unobserved(Y_student, Y_teacher, mask),
        'Q_Y_full': Q_Y_full,
        'observed_ratio': float(mask.float().mean()),
    }
