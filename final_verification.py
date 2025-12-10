"""
Final verification: With enough steps, does Q_W approach 1?
"""

import torch
import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

from smf.core.config import Config
from smf.core.runner import run_experiment

def main():
    cfg = Config()
    cfg.matrix.N1 = 100
    cfg.matrix.N2 = 100
    cfg.matrix.M = 25
    cfg.alpha.start = 2.0
    cfg.alpha.stop = 4.0
    cfg.alpha.step = 0.5
    cfg.training.max_steps = 2000  # More steps
    cfg.training.samples_per_alpha = 2
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed = 42
    cfg.algorithm.mode = "spreading_parallel"
    cfg.algorithm.damping = 0.5
    cfg.spreading.f_distribution = "rademacher"
    cfg.spreading.seed = 12345
    
    print("=" * 70)
    print("FINAL VERIFICATION: Convergence with more steps")
    print("=" * 70)
    print(f"N={cfg.matrix.N1}, M={cfg.matrix.M}, max_steps={cfg.training.max_steps}")
    print(f"Alphas: {cfg.alpha_values}")
    print()
    
    results = run_experiment(cfg)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Alpha':<8} {'Q_Y_obs':<10} {'Q_W':<10} {'Q_X':<10} {'Q_W_prime':<10}")
    print("-" * 70)
    
    for a, alpha in enumerate(results["alpha_values"]):
        qy_obs = results["Q_Y_observed"][a]
        qw = results["Q_W"][a]
        qx = results["Q_X"][a]
        qw_p = results["Q_W_prime"][a]
        
        print(f"{alpha:<8.1f} {qy_obs:<10.4f} {qw:<10.4f} {qx:<10.4f} {qw_p:<10.4f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    final_qw = results["Q_W"][-1]
    if final_qw > 0.9:
        print(f"✅ Q_W = {final_qw:.4f} at highest alpha - Algorithm is working correctly!")
        print("   The phase transition is happening as expected.")
    else:
        print(f"🔶 Q_W = {final_qw:.4f} - May need more steps or parameter tuning.")

if __name__ == "__main__":
    main()
