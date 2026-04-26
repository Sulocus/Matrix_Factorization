"""
High-Dimensional Tensor Metrics Module.

This module provides efficient metric calculations for N-dimensional tensors
defined by CP factors, avoiding the construction of the full N^n tensor.

Key Metrics:
- Tensor Cosine Similarity (Overlap): <T, S> / (||T|| * ||S||)
  Computed via Factor Correlation in O(M^2 * N) instead of O(N^n).
"""

import math
import torch
from typing import List

def compute_tensor_cosine(
    factors_a: List[torch.Tensor],
    factors_b: List[torch.Tensor]
) -> float:
    """
    Compute Cosine Similarity between two CP-tensors A and B efficiently.

    Tensor A = sum_m A[0]_m * A[1]_m * ...
    Tensor B = sum_p B[0]_p * B[1]_p * ...

    Inner Product <A, B> = sum_{m,p} prod_d (factors_a[d][:,m] . factors_b[d][:,p])

    Args:
        factors_a: List of n factors for tensor A, each (N_d, M_a)
        factors_b: List of n factors for tensor B, each (N_d, M_b)

    Returns:
        Cosine similarity (float) in [-1, 1]
    """
    numerator = compute_tensor_inner_product(factors_a, factors_b)
    norm_a = math.sqrt(compute_tensor_inner_product(factors_a, factors_a))
    norm_b = math.sqrt(compute_tensor_inner_product(factors_b, factors_b))

    return numerator / (norm_a * norm_b + 1e-12)

def compute_factor_gram_overlap(
    factors_a: List[torch.Tensor],
    factors_b: List[torch.Tensor]
) -> float:
    """
    Compute a permutation-invariant factor Gram overlap.

    Each CP factor matrix is represented through its left Gram matrix
    ``F @ F.T`` before cosine comparison. This makes the metric invariant to
    latent-column sign flips and permutations, unlike raw flattened-factor
    cosine.

    Args:
        factors_a: List of n factors for tensor A
        factors_b: List of n factors for tensor B

    Returns:
        Average factor Gram overlap across tensor dimensions.
    """
    order = len(factors_a)
    if len(factors_b) != order:
        raise ValueError("Tensor orders must match")

    total_cosine = 0.0
    for d in range(order):
        fa = factors_a[d].float()
        fb = factors_b[d].float()

        gram_a = fa @ fa.T
        gram_b = fb @ fb.T
        inner = torch.sum(gram_a * gram_b)
        norm_a = torch.norm(gram_a)
        norm_b = torch.norm(gram_b)
        cosine = inner / (norm_a * norm_b + 1e-12)
        total_cosine += cosine.item()

    return total_cosine / order

def compute_tensor_inner_product(
    factors_a: List[torch.Tensor],
    factors_b: List[torch.Tensor]
) -> float:
    """
    Compute Inner Product <A, B> efficiently.
    """
    order = len(factors_a)
    if len(factors_b) != order:
        raise ValueError("Tensor orders must match")

    # Step 1: Compute Cross-Grams (Correlations) for each dimension
    # C_d[m, p] = (A_d[:, m]).T @ (B_d[:, p])
    cross_grams = []
    for d in range(order):
        # factors_a[d]: (N, Ma), factors_b[d]: (N, Mb)
        # Ensure float32 for metric calculation
        fa = factors_a[d].float()
        fb = factors_b[d].float()

        # result: (Ma, Mb)
        gram = torch.mm(fa.T, fb)
        cross_grams.append(gram)

    # Step 2: Hadamard product (element-wise) across all dimension grams
    # Res[m, p] = C_0[m,p] * C_1[m,p] * ...
    hadamard_res = cross_grams[0]
    for d in range(1, order):
        hadamard_res = hadamard_res * cross_grams[d]

    # Step 3: Sum all elements
    # <A, B> = sum_{m,p} (Res[m,p])
    return hadamard_res.sum().item()


def compute_tensor_reconstruction_q(
    teacher_factors: List[torch.Tensor],
    student_factors: List[torch.Tensor]
) -> float:
    """
    Compute Tensor Reconstruction Quality Q using normalized MSE.

    Q = 1 - ||T_teacher - T_student||² / ||T_teacher||²

    This is equivalent to Q_Y when the observation covers the full tensor.
    Uses Gram matrices for efficient O(M²N) computation instead of O(N^n).

    Formula:
        ||T - S||² = ||T||² + ||S||² - 2<T, S>
        Q = 1 - (||T||² + ||S||² - 2<T,S>) / ||T||²
          = 2<T,S>/||T||² - ||S||²/||T||²

    Args:
        teacher_factors: Ground truth factors, list of (N_d, M) tensors
        student_factors: Estimated factors, list of (N_d, M) tensors

    Returns:
        Q value in (-inf, 1], where 1 = perfect reconstruction
    """
    # Compute ||T||², ||S||², <T, S>
    norm_teacher_sq = compute_tensor_inner_product(teacher_factors, teacher_factors)
    norm_student_sq = compute_tensor_inner_product(student_factors, student_factors)
    inner_product = compute_tensor_inner_product(teacher_factors, student_factors)

    # MSE = ||T||² + ||S||² - 2<T, S>
    mse = norm_teacher_sq + norm_student_sq - 2 * inner_product

    # Q = 1 - MSE / ||T||²
    if norm_teacher_sq < 1e-12:
        return 0.0  # Degenerate case

    q = 1.0 - mse / norm_teacher_sq

    return q


def compute_tensor_physical_overlap(
    teacher_factors: List[torch.Tensor],
    student_factors: List[torch.Tensor],
    absolute: bool = False,
) -> float:
    """
    Compute Physical Overlap (Projection) for high-order tensors.

    Overlap = <S, T> / <T, T>

    This is the projection of Student onto Teacher, measuring how much
    of Teacher is captured by Student. Equivalent to the 2D version in
    overlap.py but for N-dimensional CP tensors.

    Args:
        teacher_factors: Ground truth CP factors, list of (N_d, M) tensors
        student_factors: Estimated CP factors, list of (N_d, M) tensors
        absolute: If True, take |<S, T>| for sign ambiguity

    Returns:
        Projection coefficient (can be outside [0, 1])
    """
    inner = compute_tensor_inner_product(student_factors, teacher_factors)
    norm_teacher_sq = compute_tensor_inner_product(teacher_factors, teacher_factors)

    if absolute:
        inner = abs(inner)

    if norm_teacher_sq < 1e-12:
        return 0.0

    return inner / norm_teacher_sq


def compute_tensor_projection_abs(
    teacher_factors: List[torch.Tensor],
    student_factors: List[torch.Tensor],
) -> float:
    """Formal tensor measurement projection: |<student, teacher>| / <teacher, teacher>."""
    return compute_tensor_physical_overlap(
        teacher_factors=teacher_factors,
        student_factors=student_factors,
        absolute=True,
    )


def compute_factor_projection_abs(student: torch.Tensor, teacher: torch.Tensor) -> float:
    """Absolute coordinate projection for one tensor factor/mode."""
    student_flat = student.float().flatten()
    teacher_flat = teacher.float().flatten()
    norm_teacher_sq = (teacher_flat ** 2).sum()
    if float(norm_teacher_sq.abs().item()) < 1e-12:
        return 0.0
    return float((student_flat * teacher_flat).sum().abs() / (norm_teacher_sq + 1e-12))


def compute_tensor_factor_projection_overlaps(
    teacher_factors: List[torch.Tensor],
    student_factors: List[torch.Tensor],
) -> List[float]:
    """Per-mode Q_N projection overlaps for tensor latent factors."""
    if len(teacher_factors) != len(student_factors):
        raise ValueError("Tensor orders must match")
    return [
        compute_factor_projection_abs(student_factors[d], teacher_factors[d])
        for d in range(len(teacher_factors))
    ]
