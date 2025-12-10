"""
Test with many more iterations to see if Q_Y converges to 1.
"""

import torch
import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

from smf.core.config import Config
from smf.modules.teachers import TeacherGenerator, SpreadingDataParallel
from smf.modules.graphs.supergraph import create_supergraph
from smf.modules.algorithms.bigamp_spreading_parallel import (
    generate_F_super, compute_Y_super, BiGAMPSpreadingParallel
)
from smf.modules.metrics.spreading import compute_all_metrics_spreading_parallel

def main():
    cfg = Config()
    cfg.matrix.N1 = 100
    cfg.matrix.N2 = 100
    cfg.matrix.M = 25
    cfg.alpha.start = 3.0
    cfg.alpha.stop = 4.0
    cfg.alpha.step = 0.5
    cfg.training.max_steps = 10000  # Very many steps
    cfg.training.samples_per_alpha = 2
    cfg.training.device = "cuda"
    cfg.training.seed = 42
    cfg.algorithm.damping = 0.5
    cfg.algorithm.noise_var = 1e-10
    cfg.spreading.f_distribution = "rademacher"
    cfg.spreading.seed = 12345
    
    device = torch.device(cfg.training.device)
    N1, N2, M = cfg.matrix.N1, cfg.matrix.N2, cfg.matrix.M
    S = cfg.training.samples_per_alpha
    alpha_values = cfg.alpha_values
    seed = cfg.training.seed
    
    print("=" * 70)
    print("CONVERGENCE TEST WITH 10000 STEPS")
    print("=" * 70)
    print(f"N={N1}, M={M}, max_steps={cfg.training.max_steps}")
    print(f"Alphas: {alpha_values}")
    print()
    
    # Create Teacher
    teacher = TeacherGenerator(type="standard", spreading_seed=cfg.spreading.seed)
    W_true, X_true = teacher.create(N1, N2, M, device, seed)
    
    # Create SuperGraph
    supergraph = create_supergraph(N1, N2, M, alpha_values, S, seed, device)
    
    # Generate F
    F_super = generate_F_super(supergraph, M, cfg.spreading.seed, device, "rademacher")
    
    # Compute Y
    Y_super = compute_Y_super(W_true, X_true, supergraph, F_super)
    
    # Create spreading data
    spreading_data = SpreadingDataParallel(
        supergraph=supergraph,
        F_super=F_super,
        Y_super=Y_super,
        M=M,
        alpha_values=torch.tensor(alpha_values, device=device),
        W_teacher=W_true,
        X_teacher=X_true,
    )
    
    # Train
    algo = BiGAMPSpreadingParallel(cfg, device)
    W_students, X_students = algo.train_full_parallel(spreading_data, verbose=True)
    
    # Compute metrics
    metrics = compute_all_metrics_spreading_parallel(W_students, X_students, spreading_data)
    
    print("\n" + "=" * 70)
    print("RESULTS WITH 10000 STEPS")
    print("=" * 70)
    for a, alpha in enumerate(alpha_values):
        qy_obs = metrics["Q_Y_observed_mean"][a].item()
        qw = metrics["Q_W_mean"][a].item()
        qx = metrics["Q_X_mean"][a].item()
        print(f"Alpha {alpha:.1f}: Q_Y_obs = {qy_obs:.6f}, Q_W = {qw:.4f}, Q_X = {qx:.4f}")
        
        if qy_obs < 0.99:
            print(f"         ❌ Q_Y should be ~1.0 after convergence!")
        else:
            print(f"         ✅ Converged correctly")

if __name__ == "__main__":
    main()
