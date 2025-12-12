#!/usr/bin/env python3
"""
Run 600x600 Spreading experiment with Q_Y and Q_Y_unobserved metrics.

Parameters:
- N = 600, M = 150, S = 20
- Steps = 4000
- Alpha: 0.0 ~ 4.0
- Orthogonal Teacher + Spreading Algorithm
"""

import sys
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import torch
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smf.core.config import (
    Config, MatrixConfig, AlphaConfig, TrainingConfig, 
    AlgorithmConfig, ExecutionConfig, SpreadingConfig
)
from smf.core.device import setup_device
from smf.modules.registry import get_teacher
from smf.modules.metrics.overlap import (
    build_interaction_matrix,
    gram_overlap_normalized,
    compute_qy,
    _compute_qy_masked,
)
from smf.modules.outputs.plotting import plot_replica_heatmap, create_gif
from smf.modules.algorithms.bigamp_spreading_parallel import run_spreading_parallel

def run_spreading_orthogonal_600():
    """Run Spreading N=600, M=150, S=20 with Orthogonal Teacher and Q_Y metrics."""
    print("\n" + "="*80)
    print("600x600 Spreading (Orthogonal) with Q_Y and Q_Y_unobserved")
    print("="*80)
    
    N, M, S = 600, 150, 20
    steps = 4000
    
    device, _ = setup_device()
    
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.05),
        training=TrainingConfig(max_steps=steps, samples_per_alpha=S, seed=47),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key="bigamp_spreading_parallel",
        teacher_key="orthogonal",
        spreading=SpreadingConfig(f_distribution="rademacher", seed=1047),
        execution=ExecutionConfig(matrix_metric="gram_overlap_normalized")
    )
    
    # Use alpha_batch_size=1 for 600x600 to be safe with memory
    # N=600 is 2.25x larger than N=400, memory ~5x higher
    # skip_metrics=True to avoid OOM in compute_all_metrics_spreading_parallel
    result = run_spreading_parallel(config, verbose=True, alpha_batch_size=1, skip_metrics=True)
    
    # Process results
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path("smf/Replica_results") / f"spreading_orthogonal_600x600_M{M}_S{S}_steps{steps}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    W_students = result['W_students']
    X_students = result['X_students']
    spreading_data = result['spreading_data']
    W_t = spreading_data.W_teacher
    X_t = spreading_data.X_teacher
    Y_teacher_full = W_t @ X_t  # For simple Q_Y comparison
    
    # Get supergraph for mask info
    supergraph = spreading_data.supergraph
    
    alpha_values = config.alpha.get_values()
    all_results = {}
    heatmap_paths_W = []
    
    metric_fn = gram_overlap_normalized
    
    print("Generating plots and computing Q_Y metrics...")
    for i, alpha in enumerate(alpha_values):
        W_s = W_students[:, i, :, :]  # (S, N1, M)
        X_s = X_students[:, i, :, :]  # (S, M, N2)
        
        matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
        
        # Compute Q_Y metrics for each sample using simple W @ X
        # For Spreading model, we compare the full matrix products
        q_y_list = []
        q_y_obs_list = []
        q_y_unobs_list = []
        
        # Number of active edges for this alpha
        C_k = supergraph.get_active_edges(i)
        
        for s in range(S):
            Y_s = W_s[s] @ X_s[s]  # Student reconstruction
            
            # Get sample-specific indices and create mask
            i_idx_s, j_idx_s = supergraph.get_sample_indices(s)
            
            # Create binary mask from sparse indices for this sample
            N = W_t.shape[0]
            mask = torch.zeros((N, N), device=W_t.device, dtype=torch.float32)
            if C_k > 0:
                mask[i_idx_s[:C_k].long(), j_idx_s[:C_k].long()] = 1.0
            
            # Global Q_Y
            q_y = compute_qy(Y_s, Y_teacher_full)
            q_y_list.append(q_y)
            
            # Q_Y on observed positions
            if mask.sum() > 0:
                q_y_obs = _compute_qy_masked(Y_s, Y_teacher_full, mask, observed=True)
                q_y_obs_list.append(q_y_obs)
            else:
                q_y_obs_list.append(0.0)
            
            # Q_Y on unobserved positions
            if (1 - mask).sum() > 0:
                q_y_unobs = _compute_qy_masked(Y_s, Y_teacher_full, mask, observed=False)
                q_y_unobs_list.append(q_y_unobs)
            else:
                q_y_unobs_list.append(1.0)
        
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
        
        if i % 10 == 0 or i == len(alpha_values) - 1:
            print(f"  [{i+1}/{len(alpha_values)}] α={alpha:.2f} | Q_Y={metrics['Q_Y_mean']:.3f} | Q_Y_unobs={metrics['Q_Y_unobserved_mean']:.3f}")
        
        # Save Raw
        raw_path = raw_dir / f"alpha_{alpha:.6f}.npz"
        np.savez_compressed(raw_path,
            W_student=W_s.detach().cpu().to(torch.float16).numpy(),
            W_teacher=W_t.detach().cpu().to(torch.float16).numpy(),
            interaction_matrix_W=matrix_W.astype(np.float16)
        )
        
        path_W = plot_replica_heatmap(
            matrix_W, float(alpha), plots_dir,
            metric_name="Q_W", filename_prefix="heatmap_W"
        )
        heatmap_paths_W.append(path_W)
        
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
    ax.set_title(f'Q_Y Evolution - Spreading Orthogonal (N={N}, M={M}, S={S})')
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
    run_spreading_orthogonal_600()
