"""
Generate comprehensive Q_Y scan plot for Spreading BiG-AMP.

Parameters:
- N=200, M=50
- 5000 steps
- Rademacher F distribution
- Random graph
- Alpha: 0.0 to 4.0, step 0.1
"""

import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

import torch
import matplotlib.pyplot as plt
import numpy as np

from smf.core.config import Config
from smf.core.runner import run_experiment

def main():
    # Configuration
    cfg = Config()
    cfg.matrix.N1 = 200
    cfg.matrix.N2 = 200
    cfg.matrix.M = 50
    cfg.alpha.start = 0.0
    cfg.alpha.stop = 4.0
    cfg.alpha.step = 0.1
    cfg.training.max_steps = 5000
    cfg.training.samples_per_alpha = 2
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed = 42
    cfg.algorithm.mode = "spreading_parallel"
    cfg.algorithm.damping = 0.5
    cfg.algorithm.noise_var = 1e-10
    cfg.spreading.f_distribution = "rademacher"
    cfg.spreading.seed = 12345
    cfg.graph.type = "random"
    cfg.teacher.type = "standard"
    
    print("=" * 70)
    print("SPREADING BiG-AMP Q_Y COMPREHENSIVE SCAN")
    print("=" * 70)
    print(f"N = {cfg.matrix.N1}, M = {cfg.matrix.M}")
    print(f"Alpha: {cfg.alpha.start} -> {cfg.alpha.stop} (step {cfg.alpha.step})")
    print(f"Max steps: {cfg.training.max_steps}")
    print(f"F distribution: {cfg.spreading.f_distribution}")
    print(f"Graph type: {cfg.graph.type}")
    print()
    
    # Run experiment
    print("Running experiment (this may take a few minutes)...")
    results = run_experiment(cfg)
    
    # Extract data
    alphas = np.array(results["alpha_values"])
    Q_Y_observed = np.array(results["Q_Y_observed"])
    Q_Y_unobserved = np.array(results["Q_Y_unobserved"])
    Q_Y_total = np.array(results["Q_Y"])  # This is Q_Y_observed (total on training set)
    
    # For total Q_Y, we can compute as weighted average or just use Q_Y_observed
    # Since Q_Y is defined on observed edges in training, we'll plot Q_Y_observed as "Q_Y"
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Alpha':<8} {'Q_Y_obs':<12} {'Q_Y_unobs':<12}")
    print("-" * 40)
    for i, a in enumerate(alphas):
        print(f"{a:<8.1f} {Q_Y_observed[i]:<12.4f} {Q_Y_unobserved[i]:<12.4f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot curves
    ax.plot(alphas, Q_Y_observed, 'b-', linewidth=2, label='Q_Y (observed)', marker='o', markersize=4)
    ax.plot(alphas, Q_Y_unobserved, 'r--', linewidth=2, label='Q_Y (unobserved)', marker='s', markersize=4)
    
    # Styling
    ax.set_xlabel('α (observation density)', fontsize=14)
    ax.set_ylabel('Q_Y (overlap)', fontsize=14)
    ax.set_title(f'Spreading BiG-AMP Phase Transition\nN={cfg.matrix.N1}, M={cfg.matrix.M}, {cfg.training.max_steps} steps, {cfg.spreading.f_distribution} F', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 1.1)
    
    # Add phase transition annotation
    # Find approximate phase transition point (where Q_Y_unobserved starts increasing)
    for i in range(len(alphas)-1):
        if Q_Y_unobserved[i+1] > 0.5 and Q_Y_unobserved[i] < 0.5:
            ax.axvline(x=alphas[i], color='green', linestyle=':', linewidth=2, label=f'Phase transition ~α={alphas[i]:.1f}')
            ax.legend(fontsize=12)
            break
    
    # Save plot
    plt.tight_layout()
    plt.savefig('/home/sucia/Sparse-Matrix/spreading_qy_scan.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: /home/sucia/Sparse-Matrix/spreading_qy_scan.png")
    
    # Also save as PDF for better quality
    plt.savefig('/home/sucia/Sparse-Matrix/spreading_qy_scan.pdf', bbox_inches='tight')
    print(f"PDF saved to: /home/sucia/Sparse-Matrix/spreading_qy_scan.pdf")
    
    plt.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
