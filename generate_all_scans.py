"""
Generate comprehensive alpha sweep scans for all Teacher and Graph types.
N=200, M=50, 5000 steps, Rademacher F, Alpha 0-4 step 0.1
"""

import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from smf.core.config import Config
from smf.core.runner import run_experiment

def run_scan(teacher_type: str, graph_type: str, output_prefix: str):
    """Run a full alpha scan and save results."""
    
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
    cfg.teacher.type = teacher_type
    cfg.graph.type = graph_type
    
    print(f"\n{'='*60}")
    print(f"Scanning: Teacher={teacher_type}, Graph={graph_type}")
    print(f"{'='*60}")
    
    results = run_experiment(cfg)
    
    alphas = np.array(results["alpha_values"])
    Q_Y_obs = np.array(results["Q_Y_observed"])
    Q_Y_unobs = np.array(results["Q_Y_unobserved"])
    Q_W = np.array(results["Q_W"])
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Q_Y
    ax = axes[0]
    ax.plot(alphas, Q_Y_obs, 'b-', linewidth=2, marker='o', markersize=3, label='Q_Y (observed)')
    ax.plot(alphas, Q_Y_unobs, 'r--', linewidth=2, marker='s', markersize=3, label='Q_Y (unobserved)')
    ax.set_xlabel('α', fontsize=12)
    ax.set_ylabel('Q_Y', fontsize=12)
    ax.set_title(f'Q_Y Phase Transition\nTeacher={teacher_type}, Graph={graph_type}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 1.1)
    
    # Right: Q_W
    ax = axes[1]
    ax.plot(alphas, Q_W, 'g-', linewidth=2, marker='^', markersize=3, label='Q_W')
    ax.set_xlabel('α', fontsize=12)
    ax.set_ylabel('Q_W', fontsize=12)
    ax.set_title(f'Q_W Phase Transition\nTeacher={teacher_type}, Graph={graph_type}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'/home/sucia/Sparse-Matrix/{output_prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_prefix}.png")
    return alphas, Q_Y_obs, Q_Y_unobs, Q_W

def main():
    print("=" * 70)
    print("COMPREHENSIVE ALPHA SCAN FOR ALL TYPES")
    print("N=200, M=50, 5000 steps, Rademacher F")
    print("=" * 70)
    
    # Part 3: Teacher types (with random graph)
    print("\n### PART 3: TEACHER TYPES ###")
    teacher_types = ["standard", "orthogonal", "scaled_variance"]
    
    for t_type in teacher_types:
        run_scan(t_type, "random", f"scan_teacher_{t_type}")
    
    # Part 4: Graph types (with standard teacher)
    print("\n### PART 4: GRAPH TYPES ###")
    graph_types = ["random", "uniform", "low_loop"]
    
    for g_type in graph_types:
        run_scan("standard", g_type, f"scan_graph_{g_type}")
    
    print("\n" + "=" * 70)
    print("ALL SCANS COMPLETE!")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - scan_teacher_standard.png")
    print("  - scan_teacher_orthogonal.png")
    print("  - scan_teacher_scaled_variance.png")
    print("  - scan_graph_random.png")
    print("  - scan_graph_uniform.png")
    print("  - scan_graph_low_loop.png")

if __name__ == "__main__":
    main()
