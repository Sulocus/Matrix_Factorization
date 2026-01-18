"""
Replica Overlap Analysis Module.

Computes pairwise overlaps between S independent replicas trained on the same mask.
This measures the uniqueness/consistency of solutions.

Key Metrics:
- Q_W_replica: W overlap between replica pairs
- Q_X_replica: X overlap between replica pairs
- Q_Y_replica: Y = W @ X overlap between replica pairs

Higher replica overlap indicates more consistent/unique solutions.
"""

from typing import Dict, Optional
import torch
import numpy as np


@torch.no_grad()
def gram_overlap_cosine(A: torch.Tensor, B: torch.Tensor, use_left: bool = True) -> float:
    """
    Compute Gram matrix overlap using cosine similarity.
    
    Args:
        A: First matrix (N, M)
        B: Second matrix (N, M)
        use_left: If True, use A @ A.T (left Gram), else A.T @ A (right Gram)
        
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
        A: First matrix (n, m)
        B: Second matrix (n, m)
        use_left: If True, use left Gram matrix
        
    Returns:
        Normalized overlap in [0, 1]
    """
    q = gram_overlap_cosine(A, B, use_left)
    if use_left:
        n, m = A.shape
    else:
        n, m = A.shape[1], A.shape[0]
    b = m / (m + n + 1)  # baseline: expected cosine for random matrices
    qc = (q - b) / (1.0 - b + 1e-12)  # baseline correction
    return float(max(0.0, min(1.0, qc)))


@torch.no_grad()
def compute_replica_overlap(
    W_students: torch.Tensor,
    X_students: torch.Tensor,
    W_teacher: Optional[torch.Tensor] = None,
    X_teacher: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute pairwise Gram overlap between S replicas.
    
    This measures the consistency of solutions: if all replicas converge to
    similar solutions, replica overlap will be high.
    
    Args:
        W_students: (S, N1, M) - S replicas of W
        X_students: (S, M, N2) - S replicas of X
        W_teacher: Optional teacher W for teacher-student overlap
        X_teacher: Optional teacher X for teacher-student overlap
        
    Returns:
        Dictionary with all replica metrics:
        - Q_W_replica_mean, Q_W_replica_std: Raw W overlap
        - Q_X_replica_mean, Q_X_replica_std: Raw X overlap
        - Q_Y_replica_mean, Q_Y_replica_std: Y = W @ X overlap
        - Q_W_replica_norm_mean, Q_W_replica_norm_std: Normalized W overlap
        - Q_X_replica_norm_mean, Q_X_replica_norm_std: Normalized X overlap
        - n_pairs: Number of replica pairs
    """
    S = W_students.shape[0]
    
    if S < 2:
        return {
            'Q_W_replica_mean': 0.0, 'Q_W_replica_std': 0.0,
            'Q_X_replica_mean': 0.0, 'Q_X_replica_std': 0.0,
            'Q_Y_replica_mean': 0.0, 'Q_Y_replica_std': 0.0,
            'Q_W_replica_norm_mean': 0.0, 'Q_W_replica_norm_std': 0.0,
            'Q_X_replica_norm_mean': 0.0, 'Q_X_replica_norm_std': 0.0,
            'n_pairs': 0,
        }
    
    # Compute all Y = W @ X
    Y_students = torch.bmm(W_students, X_students)  # (S, N1, N2)
    
    # Collect pairwise overlaps
    Q_W_list = []
    Q_X_list = []
    Q_Y_list = []
    Q_W_norm_list = []
    Q_X_norm_list = []
    
    for i in range(S):
        for j in range(i + 1, S):
            # W overlap (Gram matrix cosine - raw)
            Q_W_list.append(gram_overlap_cosine(W_students[i], W_students[j], use_left=True))
            
            # X overlap (Gram matrix cosine - raw)
            Q_X_list.append(gram_overlap_cosine(X_students[i], X_students[j], use_left=False))
            
            # W overlap with baseline correction
            Q_W_norm_list.append(gram_overlap_normalized(W_students[i], W_students[j], use_left=True))
            
            # X overlap with baseline correction
            Q_X_norm_list.append(gram_overlap_normalized(X_students[i], X_students[j], use_left=False))
            
            # Y overlap (direct cosine similarity)
            Y_i_flat = Y_students[i].flatten()
            Y_j_flat = Y_students[j].flatten()
            Q_Y = float((Y_i_flat * Y_j_flat).sum() /
                       (Y_i_flat.norm() * Y_j_flat.norm() + 1e-12))
            Q_Y_list.append(Q_Y)
    
    # Compute statistics
    metrics = {
        'Q_W_replica_mean': float(np.mean(Q_W_list)),
        'Q_X_replica_mean': float(np.mean(Q_X_list)),
        'Q_Y_replica_mean': float(np.mean(Q_Y_list)),
        'Q_W_replica_norm_mean': float(np.mean(Q_W_norm_list)),
        'Q_X_replica_norm_mean': float(np.mean(Q_X_norm_list)),
        'Q_W_replica_std': float(np.std(Q_W_list, ddof=1)) if len(Q_W_list) > 1 else 0.0,
        'Q_X_replica_std': float(np.std(Q_X_list, ddof=1)) if len(Q_X_list) > 1 else 0.0,
        'Q_Y_replica_std': float(np.std(Q_Y_list, ddof=1)) if len(Q_Y_list) > 1 else 0.0,
        'Q_W_replica_norm_std': float(np.std(Q_W_norm_list, ddof=1)) if len(Q_W_norm_list) > 1 else 0.0,
        'Q_X_replica_norm_std': float(np.std(Q_X_norm_list, ddof=1)) if len(Q_X_norm_list) > 1 else 0.0,
        'n_pairs': len(Q_W_list),
    }
    
    # Add teacher-student metrics if teacher provided
    if W_teacher is not None and X_teacher is not None:
        Y_teacher = W_teacher @ X_teacher
        
        Q_W_teacher_list = []
        Q_X_teacher_list = []
        Q_Y_teacher_list = []
        
        for s in range(S):
            # Q_W' (teacher-student, normalized)
            Q_W_teacher_list.append(gram_overlap_normalized(W_students[s], W_teacher, use_left=True))
            
            # Q_X' (teacher-student, normalized)
            Q_X_teacher_list.append(gram_overlap_normalized(X_students[s], X_teacher, use_left=False))
            
            # Q_Y (teacher-student)
            Yp = W_students[s] @ X_students[s]
            Q_Y = float((Y_teacher.flatten() * Yp.flatten()).sum() /
                       (Y_teacher.norm() * Yp.norm() + 1e-12))
            Q_Y_teacher_list.append(Q_Y)
        
        metrics['Q_W_teacher_mean'] = float(np.mean(Q_W_teacher_list))
        metrics['Q_W_teacher_std'] = float(np.std(Q_W_teacher_list, ddof=1)) if S > 1 else 0.0
        metrics['Q_X_teacher_mean'] = float(np.mean(Q_X_teacher_list))
        metrics['Q_X_teacher_std'] = float(np.std(Q_X_teacher_list, ddof=1)) if S > 1 else 0.0
        metrics['Q_Y_teacher_mean'] = float(np.mean(Q_Y_teacher_list))
        metrics['Q_Y_teacher_std'] = float(np.std(Q_Y_teacher_list, ddof=1)) if S > 1 else 0.0
    
    return metrics


def analyze_replica_results(
    results: Dict,
    W_teacher: Optional[torch.Tensor] = None,
    X_teacher: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze replica overlap from an ExperimentResult.
    
    Args:
        results: Dictionary of {scan_value: SingleRunResult}
        W_teacher: Optional teacher W
        X_teacher: Optional teacher X
        
    Returns:
        Dictionary of {scan_value: replica_metrics}
    """
    replica_results = {}
    
    for scan_value, single_result in results.items():
        if single_result.W_students is not None and single_result.X_students is not None:
            metrics = compute_replica_overlap(
                W_students=single_result.W_students,
                X_students=single_result.X_students,
                W_teacher=W_teacher,
                X_teacher=X_teacher,
            )
            replica_results[scan_value] = metrics
    
    return replica_results
