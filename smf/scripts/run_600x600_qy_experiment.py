#!/usr/bin/env python3
"""
Run 600x600 experiment with Q_Y and Q_Y_unobserved metrics.

Parameters:
- N = 600, M = 150, S = 20
- Steps = 4000
- Alpha: 0.0 ~ 4.0
- Supports: Standard/Orthogonal Teacher, BiG-AMP algorithm
"""

import sys
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import torch
import json
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smf.core.config import (
    Config, MatrixConfig, AlphaConfig, TrainingConfig, 
    AlgorithmConfig, ExecutionConfig
)
from smf.core.device import setup_device
from smf.modules.registry import get_algorithm, get_graph, get_teacher
from smf.modules.metrics.overlap import (
    build_interaction_matrix,
    gram_overlap_normalized,
    compute_qy,
    _compute_qy_masked,
)
from smf.modules.outputs.plotting import plot_replica_heatmap, create_gif

def run_600x600_experiment():
    """Run BiG-AMP N=600, M=150, S=20 with Q_Y metrics."""
    print("\n" + "="*80)
    print("600x600 Experiment with Q_Y and Q_Y_unobserved")
    print("="*80)
    
    N, M, S = 600, 150, 20
    steps = 4000
    
    device, _ = setup_device()
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path("smf/Replica_results") / f"bigamp_600x600_M{M}_S{S}_steps{steps}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.05),
        training=TrainingConfig(max_steps=steps, samples_per_alpha=S, seed=46),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key="bigamp",
        teacher_key="standard",
        execution=ExecutionConfig(matrix_metric="gram_overlap_normalized")
    )
    
    teacher_cls = get_teacher("standard").cls
    algorithm_cls = get_algorithm("bigamp").cls
    graph_cls = get_graph("random").cls
    
    teacher = teacher_cls()
    algorithm = algorithm_cls(config, device)
    graph = graph_cls()
    
    torch.manual_seed(config.training.seed)
    W_t, X_t, _ = teacher.create_with_Y(N, N, M, device, config.training.seed)
    Y_teacher = W_t @ X_t
    
    alpha_values = config.alpha.get_values()
    all_results = {}
    heatmap_paths_W = []
    
    start_time = time.time()
    
    print(f"  Matrix: {N}x{N}, M={M}, S={S}")
    print(f"  Steps: {steps}")
    print(f"  Alpha points: {len(alpha_values)}")
    print()
    
    for idx, alpha in enumerate(alpha_values):
        mask_seed = config.training.seed + int(alpha * 1000)
        mask, _ = graph.generate_mask(N, N, M, alpha, device, mask_seed)
        
        W_s, X_s = algorithm.train_single_alpha(
            W_t, X_t, None, mask, alpha, mask_seed + 10000,
        )
        
        elapsed = time.time() - start_time
        eta = elapsed / (idx + 1) * (len(alpha_values) - idx - 1)
        
        # Compute Y_student for each sample and metrics
        metric_fn = gram_overlap_normalized
        matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
        
        # Compute Q_Y metrics for each sample
        q_y_list = []
        q_y_obs_list = []
        q_y_unobs_list = []
        
        for s in range(S):
            Y_s = W_s[s] @ X_s[s]
            
            # Global Q_Y
            q_y = compute_qy(Y_s, Y_teacher)
            q_y_list.append(q_y)
            
            # Q_Y on observed positions
            if mask.sum() > 0:
                q_y_obs = _compute_qy_masked(Y_s, Y_teacher, mask, observed=True)
                q_y_obs_list.append(q_y_obs)
            else:
                q_y_obs_list.append(0.0)
            
            # Q_Y on unobserved positions
            if (1 - mask).sum() > 0:
                q_y_unobs = _compute_qy_masked(Y_s, Y_teacher, mask, observed=False)
                q_y_unobs_list.append(q_y_unobs)
            else:
                q_y_unobs_list.append(1.0)  # If fully observed, default to 1.0
        
        metrics = {
            'interaction_matrix_W': matrix_W.tolist(),
            'Q_Y_mean': float(np.mean(q_y_list)),
            'Q_Y_std': float(np.std(q_y_list)),
            'Q_Y_observed_mean': float(np.mean(q_y_obs_list)),
            'Q_Y_observed_std': float(np.std(q_y_obs_list)),
            'Q_Y_unobserved_mean': float(np.mean(q_y_unobs_list)),
            'Q_Y_unobserved_std': float(np.std(q_y_unobs_list)),
        }
        
        all_results[float(alpha)] = metrics
        
        print(f"\r  [{idx+1}/{len(alpha_values)}] α={alpha:.2f} | Q_Y={metrics['Q_Y_mean']:.3f} | Q_Y_unobs={metrics['Q_Y_unobserved_mean']:.3f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", end="", flush=True)
        
        # Save Raw
        raw_path = raw_dir / f"alpha_{alpha:.6f}.npz"
        np.savez_compressed(raw_path,
            W_student=W_s.detach().cpu().to(torch.float16).numpy(),
            W_teacher=W_t.detach().cpu().to(torch.float16).numpy(),
            interaction_matrix_W=matrix_W.astype(np.float16)
        )
        
        # Plot W
        path_W = plot_replica_heatmap(
            matrix_W, float(alpha), plots_dir,
            metric_name="Q_W", filename_prefix="heatmap_W"
        )
        heatmap_paths_W.append(path_W)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
    
    if heatmap_paths_W:
        create_gif(heatmap_paths_W, plots_dir / "animation_W.gif", duration=0.2)
    
    # Save results
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump({
            'config': config.to_dict(),
            'results': all_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        
    print(f"Saved results to {output_dir}")
    
    # Generate Q_Y plot
    import matplotlib.pyplot as plt
    
    alphas = sorted(all_results.keys())
    q_y_vals = [all_results[a]['Q_Y_mean'] for a in alphas]
    q_y_obs_vals = [all_results[a]['Q_Y_observed_mean'] for a in alphas]
    q_y_unobs_vals = [all_results[a]['Q_Y_unobserved_mean'] for a in alphas]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, q_y_vals, 'g-', label='Q_Y (Global)', linewidth=2, marker='o', markersize=3)
    ax.plot(alphas, q_y_obs_vals, 'b--', label='Q_Y_observed', linewidth=2, marker='s', markersize=3)
    ax.plot(alphas, q_y_unobs_vals, 'm-.', label='Q_Y_unobserved', linewidth=2, marker='^', markersize=3)
    
    ax.set_xlabel('Alpha (Measurement Ratio)')
    ax.set_ylabel('Q_Y Overlap')
    ax.set_title(f'Q_Y Evolution (N={N}, M={M}, S={S})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    plt.savefig(output_dir / "qy_evolution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: qy_evolution.png")
    
    # Call visualize script
    from smf.scripts.visualize_replica_results import process_experiment
    try:
        process_experiment(output_dir)
        print("Visualization script finished successfully.")
    except Exception as e:
        print(f"Visualization script failed: {e}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    run_600x600_experiment()
