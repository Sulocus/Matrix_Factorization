"""
Analyze why Q_Y_observed drops at high alpha.

Theoretical expectation:
- Q_Y_observed should ALWAYS be ~1.0 for all alphas (we're training on these edges!)
- Only Q_Y_unobserved (generalization) should show the phase transition

Current observation:
- Q_Y_observed at alpha=0.5: 1.0
- Q_Y_observed at alpha=4.0: 0.89 (WHY?)

This script investigates the root cause.
"""

import torch
import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

from smf.core.config import Config
from smf.modules.teachers import TeacherGenerator, SpreadingDataParallel
from smf.modules.graphs.supergraph import create_supergraph
from smf.modules.algorithms.bigamp_spreading_parallel import (
    generate_F_super, compute_Y_super, forward_pass_parallel, BiGAMPSpreadingParallel
)
import math

def main():
    # Configuration
    cfg = Config()
    cfg.matrix.N1 = 100
    cfg.matrix.N2 = 100
    cfg.matrix.M = 25
    cfg.alpha.start = 0.5
    cfg.alpha.stop = 4.0
    cfg.alpha.step = 0.5
    cfg.training.max_steps = 200  # More steps
    cfg.training.samples_per_alpha = 1
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed = 42
    cfg.spreading.f_distribution = "rademacher"
    cfg.spreading.seed = 12345
    
    device = torch.device(cfg.training.device)
    
    print("=" * 70)
    print("ANALYSIS: Why Q_Y_observed drops at high alpha?")
    print("=" * 70)
    
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
    F = F_super[0].float()
    Y_teacher = Y_super[0]
    i_idx, j_idx = supergraph.get_sample_indices(0)
    alpha_mask = supergraph.alpha_mask
    
    print(f"\nParameters: N={N1}, M={M}, max_steps={cfg.training.max_steps}")
    print(f"Alphas: {alpha_values}")
    
    print("\n" + "=" * 70)
    print("KEY ANALYSIS: Degrees of Freedom vs Constraints")
    print("=" * 70)
    
    for a, alpha in enumerate(alpha_values):
        C_k = supergraph.get_active_edges(a)
        # Degrees of freedom: N1*M + M*N2 (W and X parameters)
        dof = N1 * M + M * N2
        # Constraints: C_k equations
        constraints = C_k
        ratio = dof / constraints if constraints > 0 else float('inf')
        
        print(f"Alpha {alpha:.1f}: C_k={C_k}, DOF={dof}, Ratio={ratio:.2f}")
        if ratio > 1:
            print(f"         -> Underconstrained: easy to overfit!")
        else:
            print(f"         -> Overconstrained: harder to fit exactly")
    
    print("\n" + "=" * 70)
    print("TRAINING AND CONVERGENCE CHECK")
    print("=" * 70)
    
    # Train
    spreading_data = SpreadingDataParallel(
        supergraph=supergraph,
        F_super=F_super,
        Y_super=Y_super,
        M=M,
        alpha_values=torch.tensor(alpha_values, device=device),
        W_teacher=W_true,
        X_teacher=X_true,
    )
    
    algo = BiGAMPSpreadingParallel(cfg, device)
    W_hat, X_hat = algo.train_full_parallel(spreading_data, verbose=False)
    
    # W_hat is (S, A, N1, M), X_hat is (S, A, M, N2)
    W_hat = W_hat[0]  # (A, N1, M)
    X_hat = X_hat[0]  # (A, M, N2)
    
    print("\nFinal training result:")
    
    for a, alpha in enumerate(alpha_values):
        C_k = supergraph.get_active_edges(a)
        if C_k == 0:
            print(f"Alpha {alpha:.1f}: No edges")
            continue
        
        # Compute Y_student on observed edges
        Y_student_full = forward_pass_parallel(
            W_hat.unsqueeze(0), X_hat.unsqueeze(0), F, i_idx, j_idx, alpha_mask.unsqueeze(0)
        )[0]  # (A, C_max)
        
        y_t = Y_teacher[:C_k]
        y_s = Y_student_full[a, :C_k]
        
        # Residual
        residual = (y_t - y_s)
        mse = (residual ** 2).mean().item()
        
        # Q_Y
        dot = (y_t * y_s).sum()
        norm_t = y_t.norm()
        norm_s = y_s.norm()
        qy = (dot / (norm_t * norm_s + 1e-12)).item()
        
        print(f"Alpha {alpha:.1f}: Q_Y = {qy:.4f}, MSE = {mse:.6f}, ||Y_t|| = {norm_t:.2f}, ||Y_s|| = {norm_s:.2f}")
        
        if qy < 0.95:
            print(f"         -> PROBLEM: Q_Y should be ~1.0 for training set!")
            print(f"         -> Residual mean: {residual.mean():.4f}, std: {residual.std():.4f}")

    print("\n" + "=" * 70)
    print("HYPOTHESIS")
    print("=" * 70)
    print("""
At high alpha, we have more constraints than degrees of freedom.
The algorithm may not converge to a perfect solution because:
1. It's not possible to perfectly fit all equations
2. The noise_var parameter affects convergence
3. The max_steps may not be enough

But wait - in Spreading model with random F, perfect fitting IS theoretically possible
if W and X have enough capacity. Let me check the algorithm convergence...
""")

if __name__ == "__main__":
    main()
