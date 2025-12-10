"""
Compare NEW code vs LEGACY code with identical parameters.
"""

import torch
import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')
sys.path.insert(0, '/home/sucia/Sparse-Matrix/_legacy/backup_pre_refactor_20251209')

def run_new_code():
    from smf.core.config import Config
    from smf.modules.teachers import TeacherGenerator, SpreadingDataParallel
    from smf.modules.graphs.supergraph import create_supergraph
    from smf.modules.algorithms.bigamp_spreading_parallel import (
        generate_F_super, compute_Y_super, BiGAMPSpreadingParallel
    )
    from smf.modules.metrics.spreading import compute_all_metrics_spreading_parallel
    
    cfg = Config()
    cfg.matrix.N1 = 100
    cfg.matrix.N2 = 100
    cfg.matrix.M = 25
    cfg.alpha.start = 3.0
    cfg.alpha.stop = 4.0
    cfg.alpha.step = 0.5
    cfg.training.max_steps = 5000
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
    
    print("\n--- NEW CODE RESULTS ---")
    for a, alpha in enumerate(alpha_values):
        qy = metrics["Q_Y_mean"][a].item()
        qw = metrics["Q_W_mean"][a].item()
        print(f"Alpha {alpha:.1f}: Q_Y = {qy:.4f}, Q_W = {qw:.4f}")
    
    return metrics

def main():
    print("=" * 70)
    print("COMPARING NEW CODE VS LEGACY CODE")
    print("=" * 70)
    
    print("\n[1] Running NEW code...")
    new_metrics = run_new_code()
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
