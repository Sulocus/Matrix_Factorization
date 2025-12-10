"""
Verify Q_Y_unobserved phase transition.

Expected behavior:
- Q_Y_observed: 1.0 for alpha < 2.0, then drops (overconstrained)
- Q_Y_unobserved: Low for small alpha (overfitting), high for large alpha (generalization)
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
    cfg.alpha.start = 0.5
    cfg.alpha.stop = 4.0
    cfg.alpha.step = 0.5
    cfg.training.max_steps = 500  # More steps for better convergence
    cfg.training.samples_per_alpha = 2
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed = 42
    cfg.algorithm.mode = "spreading_parallel"
    cfg.spreading.f_distribution = "rademacher"
    cfg.spreading.seed = 12345
    
    print("=" * 70)
    print("VERIFY Q_Y_unobserved PHASE TRANSITION")
    print("=" * 70)
    print(f"N={cfg.matrix.N1}, M={cfg.matrix.M}, max_steps={cfg.training.max_steps}")
    print(f"Alphas: {cfg.alpha_values}")
    print()
    
    # Run experiment
    results = run_experiment(cfg)
    
    print("\n" + "=" * 70)
    print("RESULTS: Q_Y_observed vs Q_Y_unobserved")
    print("=" * 70)
    print(f"\n{'Alpha':<8} {'Q_Y_obs':<10} {'Q_Y_unobs':<10} {'Q_W':<10} {'Interpretation'}")
    print("-" * 70)
    
    for a, alpha in enumerate(results["alpha_values"]):
        qy_obs = results["Q_Y_observed"][a]
        qy_unobs = results["Q_Y_unobserved"][a]
        qw = results["Q_W"][a]
        
        # Interpretation
        if qy_obs > 0.9 and qy_unobs < 0.3:
            interp = "❌ Overfitting (memorization)"
        elif qy_obs > 0.9 and qy_unobs > 0.7:
            interp = "✅ Generalization (true learning)"
        elif qy_obs < 0.9 and qy_unobs > 0.7:
            interp = "✅ Beyond overfit region"
        else:
            interp = "🔶 Transition region"
        
        print(f"{alpha:<8.1f} {qy_obs:<10.4f} {qy_unobs:<10.4f} {qw:<10.4f} {interp}")
    
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print("""
For N=100, M=25:
- DOF = 100*25 + 25*100 = 5000
- Critical alpha = DOF / (N * M) = 5000 / (100 * 25) = 2.0

At alpha < 2.0: Underconstrained, can memorize (Q_Y_obs=1, Q_Y_unobs=low)
At alpha ~ 2.0: Transition point
At alpha > 2.0: Overconstrained, must generalize (Q_Y_unobs should increase)
""")

if __name__ == "__main__":
    main()
