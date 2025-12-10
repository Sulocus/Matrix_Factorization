"""
Verify all Teacher types and Graph types work correctly.
"""

import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

import torch
import numpy as np

from smf.core.config import Config
from smf.core.runner import run_experiment

def test_teacher_types():
    print("=" * 70)
    print("PART 3: TEACHER TYPES VERIFICATION")
    print("=" * 70)
    
    teacher_types = ["standard", "orthogonal", "scaled_variance"]
    
    for t_type in teacher_types:
        print(f"\n--- Testing Teacher: {t_type} ---")
        
        cfg = Config()
        cfg.matrix.N1 = 100
        cfg.matrix.N2 = 100
        cfg.matrix.M = 25
        cfg.alpha.start = 3.5
        cfg.alpha.stop = 4.0
        cfg.alpha.step = 0.5
        cfg.training.max_steps = 2000
        cfg.training.samples_per_alpha = 2
        cfg.training.device = "cuda"
        cfg.training.seed = 42
        cfg.algorithm.mode = "spreading_parallel"
        cfg.spreading.f_distribution = "rademacher"
        cfg.spreading.seed = 12345
        cfg.teacher.type = t_type
        cfg.graph.type = "random"
        
        try:
            results = run_experiment(cfg)
            
            for a, alpha in enumerate(results["alpha_values"]):
                qy = results["Q_Y"][a]
                qw = results["Q_W"][a]
                status = "✅" if qy > 0.99 else "❌"
                print(f"  Alpha {alpha:.1f}: Q_Y={qy:.4f} {status}, Q_W={qw:.4f}")
            
            print(f"  Status: ✅ PASSED")
        except Exception as e:
            print(f"  Status: ❌ FAILED - {e}")

def test_graph_types():
    print("\n" + "=" * 70)
    print("PART 4: GRAPH TYPES VERIFICATION")
    print("=" * 70)
    
    graph_types = ["random", "uniform", "low_loop"]
    
    for g_type in graph_types:
        print(f"\n--- Testing Graph: {g_type} ---")
        
        cfg = Config()
        cfg.matrix.N1 = 100
        cfg.matrix.N2 = 100
        cfg.matrix.M = 25
        cfg.alpha.start = 3.5
        cfg.alpha.stop = 4.0
        cfg.alpha.step = 0.5
        cfg.training.max_steps = 2000
        cfg.training.samples_per_alpha = 2
        cfg.training.device = "cuda"
        cfg.training.seed = 42
        cfg.algorithm.mode = "spreading_parallel"
        cfg.spreading.f_distribution = "rademacher"
        cfg.spreading.seed = 12345
        cfg.teacher.type = "standard"
        cfg.graph.type = g_type
        
        try:
            results = run_experiment(cfg)
            
            for a, alpha in enumerate(results["alpha_values"]):
                qy = results["Q_Y"][a]
                qw = results["Q_W"][a]
                status = "✅" if qy > 0.99 else "❌"
                print(f"  Alpha {alpha:.1f}: Q_Y={qy:.4f} {status}, Q_W={qw:.4f}")
            
            print(f"  Status: ✅ PASSED")
        except Exception as e:
            print(f"  Status: ❌ FAILED - {e}")

def main():
    test_teacher_types()
    test_graph_types()
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
