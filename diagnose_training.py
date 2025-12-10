"""
In-Training Diagnostic: Trace Q_Y evolution during training.

This script monitors Q_Y at each step to understand when/why it becomes inverted.
"""

import torch
import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

from smf.core.config import Config
from smf.modules.teachers import TeacherGenerator, SpreadingDataParallel
from smf.modules.graphs.supergraph import create_supergraph
from smf.modules.algorithms.bigamp_spreading_parallel import (
    generate_F_super, compute_Y_super, forward_pass_parallel
)
import math

def compute_qy_for_alpha(W_hat, X_hat, F, i_idx, j_idx, Y_teacher, C_k, alpha_idx):
    """Compute Q_Y for a specific alpha."""
    if C_k == 0:
        return 0.0
    
    # Create alpha_mask for this single alpha
    A = W_hat.shape[0]
    C_max = F.shape[0]
    alpha_mask = torch.zeros(A, C_max, device=W_hat.device, dtype=torch.bool)
    alpha_mask[alpha_idx, :C_k] = True
    
    Y_student = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, alpha_mask)
    
    y_t = Y_teacher[:C_k]
    y_s = Y_student[alpha_idx, :C_k]
    
    dot = (y_t * y_s).sum()
    norm_t = y_t.norm()
    norm_s = y_s.norm()
    
    return (dot / (norm_t * norm_s + 1e-12)).item()

def main():
    # Configuration
    cfg = Config()
    cfg.matrix.N1 = 100  # Smaller for fast testing
    cfg.matrix.N2 = 100
    cfg.matrix.M = 25
    cfg.alpha.start = 0.0
    cfg.alpha.stop = 4.0
    cfg.alpha.step = 1.0  # Fewer alphas for readability
    cfg.training.max_steps = 50  # Few steps for tracing
    cfg.training.samples_per_alpha = 1
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed = 42
    cfg.spreading.f_distribution = "rademacher"
    cfg.spreading.seed = 12345
    
    device = torch.device(cfg.training.device)
    
    print("=" * 70)
    print("IN-TRAINING DIAGNOSTIC: Q_Y EVOLUTION")
    print("=" * 70)
    print(f"N={cfg.matrix.N1}, M={cfg.matrix.M}")
    print(f"Alphas: {cfg.alpha_values}")
    print(f"Max steps: {cfg.training.max_steps}")
    print()
    
    N1, N2, M = cfg.matrix.N1, cfg.matrix.N2, cfg.matrix.M
    S = cfg.training.samples_per_alpha
    alpha_values = cfg.alpha_values
    seed = cfg.training.seed
    A = len(alpha_values)
    
    # Create Teacher
    teacher = TeacherGenerator(type="standard", spreading_seed=cfg.spreading.seed)
    W_true, X_true = teacher.create(N1, N2, M, device, seed)
    
    # Create SuperGraph
    supergraph = create_supergraph(N1, N2, M, alpha_values, S, seed, device)
    
    # Generate F
    F_super = generate_F_super(supergraph, M, cfg.spreading.seed, device, "rademacher")
    
    # Compute Y
    Y_super = compute_Y_super(W_true, X_true, supergraph, F_super)
    
    # Get data for sample 0
    F = F_super[0]  # (C_max, M)
    Y_teacher = Y_super[0]  # (C_max,)
    i_idx, j_idx = supergraph.get_sample_indices(0)
    
    # Convert F to float
    F = F.float()
    
    print("Edge counts per alpha:")
    for a, alpha in enumerate(alpha_values):
        C_k = supergraph.get_active_edges(a)
        print(f"  Alpha {alpha:.1f}: {C_k} edges")
    
    print("\n" + "-" * 70)
    print("TRAINING TRACE")
    print("-" * 70)
    
    # Initialize student (same as in algorithm)
    torch.manual_seed(9999)  # Different from teacher
    W_hat = torch.randn(A, N1, M, device=device) * 0.1
    X_hat = torch.randn(A, M, N2, device=device) * 0.1
    W_var = torch.ones(A, N1, M, device=device)
    X_var = torch.ones(A, M, N2, device=device)
    
    damping = cfg.algorithm.damping
    noise_var = cfg.algorithm.noise_var
    alpha_mask = supergraph.alpha_mask
    
    # Print initial Q_Y
    print("\nStep 0 (Initial):")
    for a, alpha in enumerate(alpha_values):
        C_k = supergraph.get_active_edges(a)
        qy = compute_qy_for_alpha(W_hat, X_hat, F, i_idx, j_idx, Y_teacher, C_k, a)
        print(f"  Alpha {alpha:.1f}: Q_Y = {qy:.4f}")
    
    # Manual training loop with detailed tracing
    from smf.modules.algorithms.bigamp_spreading_parallel import bigamp_spreading_parallel_step
    
    prev_s = None
    for step in range(cfg.training.max_steps):
        W_hat, X_hat, W_var, X_var, prev_s = bigamp_spreading_parallel_step(
            W_hat=W_hat,
            X_hat=X_hat,
            W_var=W_var,
            X_var=X_var,
            Y_values=Y_teacher,
            F=F,
            i_idx=i_idx,
            j_idx=j_idx,
            alpha_mask=alpha_mask,
            damping=damping,
            noise_var=noise_var,
            prev_s=prev_s,
        )
        
        # Print Q_Y every 10 steps
        if (step + 1) % 10 == 0 or step == 0:
            print(f"\nStep {step + 1}:")
            for a, alpha in enumerate(alpha_values):
                C_k = supergraph.get_active_edges(a)
                qy = compute_qy_for_alpha(W_hat, X_hat, F, i_idx, j_idx, Y_teacher, C_k, a)
                w_norm = W_hat[a].norm().item()
                x_norm = X_hat[a].norm().item()
                print(f"  Alpha {alpha:.1f}: Q_Y = {qy:.4f}, ||W|| = {w_norm:.2f}, ||X|| = {x_norm:.2f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Final check: Compute Y_student and compare norms
    print("\nFinal Y_student vs Y_teacher comparison:")
    full_mask = torch.ones(A, supergraph.C_max, device=device, dtype=torch.bool)
    Y_student_final = forward_pass_parallel(W_hat, X_hat, F, i_idx, j_idx, full_mask)
    
    for a, alpha in enumerate(alpha_values):
        C_k = supergraph.get_active_edges(a)
        if C_k == 0:
            print(f"  Alpha {alpha:.1f}: No edges")
            continue
        
        y_t = Y_teacher[:C_k]
        y_s = Y_student_final[a, :C_k]
        
        print(f"  Alpha {alpha:.1f}: ||Y_t|| = {y_t.norm():.2f}, ||Y_s|| = {y_s.norm():.2f}, "
              f"dot = {(y_t * y_s).sum():.2f}")

if __name__ == "__main__":
    main()
