#!/usr/bin/env python3
"""
Run BiG-AMP Experiment with Orthogonal Teacher.

Parameters:
- N = 200, M = 50
- S = 50 (replicas)
- Alpha: 0.0 ~ 4.0, step = 0.05
- Steps: 2000
- Teacher: Orthogonal
"""

import sys
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from smf.core.config import (
    Config, MatrixConfig, AlphaConfig, TrainingConfig, 
    AlgorithmConfig, ExecutionConfig, SpreadingConfig
)
from smf.core.device import setup_device
from smf.modules.registry import get_algorithm, get_graph, get_teacher
from smf.modules.metrics.overlap import (
    compute_all_metrics,
    build_interaction_matrix,
    get_metric_function,
    gram_overlap_normalized,
)
from smf.modules.outputs.plotting import plot_replica_heatmap, create_gif, plot_overlap_evolution

def run_orthogonal_experiment():
    # Parameters
    N = 200
    M = 50
    S = 50
    max_steps = 2000
    alpha_start = 0.0
    alpha_stop = 4.0
    alpha_step = 0.05
    teacher_key = "orthogonal"
    algorithm_key = "bigamp"
    metric_name = "gram_overlap_normalized"
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {algorithm_key.upper()} (Orthogonal) | Steps={max_steps}")
    print("=" * 70)
    
    device, device_info = setup_device()
    print(f"Device: {device_info.device_name}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path("smf/Replica_results") / f"{algorithm_key}_orthogonal_{N}x{N}_M{M}_S{S}_steps{max_steps}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    print(f"Output: {output_dir}")
    
    # Config
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
        training=TrainingConfig(max_steps=max_steps, samples_per_alpha=S, seed=42),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key=algorithm_key,
        graph_key="random",
        teacher_key=teacher_key,
        execution=ExecutionConfig(matrix_metric=metric_name),
    )
    
    # Load classes
    algorithm_cls = get_algorithm(config.algorithm_key).cls
    graph_cls = get_graph(config.graph_key).cls
    teacher_cls = get_teacher(config.teacher_key).cls
    
    # Create instances
    algorithm = algorithm_cls(config, device)
    graph = graph_cls()
    teacher = teacher_cls()
    
    # Create teacher model
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    W_t, X_t, Y_t = teacher.create_with_Y(N, N, M, device, config.training.seed)
    
    print(f"Teacher (Orthogonal): W={W_t.shape}, X={X_t.shape}")
    
    try:
        metric_fn = get_metric_function(metric_name)
    except ValueError:
        metric_fn = gram_overlap_normalized
    
    alpha_values = config.alpha.get_values()
    print(f"Alpha: {alpha_values[0]:.2f} ~ {alpha_values[-1]:.2f} ({len(alpha_values)} points)")
    
    all_results = {}
    heatmap_paths_W = []
    
    start_time = time.time()
    
    # Run loop
    for idx, alpha in enumerate(alpha_values):
        mask_seed = config.training.seed + int(alpha * 1000)
        mask, _ = graph.generate_mask(N, N, M, alpha, device, mask_seed)
        
        # Train
        W_s, X_s = algorithm.train_single_alpha(
            W_t, X_t, Y_t, mask, alpha, mask_seed + 10000,
        )
        
        # Progress
        elapsed = time.time() - start_time
        eta = elapsed / (idx + 1) * (len(alpha_values) - idx - 1)
        print(f"\r  [{idx+1}/{len(alpha_values)}] α={alpha:.2f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", end="", flush=True)
        
        # Compute metrics
        metrics = {}
        for s in range(S):
            trial_metrics = compute_all_metrics(W_s[s], X_s[s], W_t, X_t, Y_t, mask=mask)
            for k, v in trial_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
        
        # Aggregate
        for k in list(metrics.keys()):
            vals = metrics[k]
            metrics[f'{k}_mean'] = float(np.mean(vals))
            metrics[f'{k}_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0
            del metrics[k]
        
        # Interaction Matrix W
        matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
        metrics['interaction_matrix_W'] = matrix_W.tolist()
        
        # Save raw data
        all_results[float(alpha)] = metrics
        
        raw_path = raw_dir / f"alpha_{alpha:.6f}.npz"
        # X matrix is symmetric so we don't strictly need it for plot, but saving it is good practice
        # But for efficiency and user request, we can skip X plotting logic here.
        
        np.savez_compressed(raw_path,
            W_student=W_s.detach().cpu().to(torch.float16).numpy(),
            W_teacher=W_t.detach().cpu().to(torch.float16).numpy(),
            interaction_matrix_W=matrix_W.astype(np.float16)
        )
        
        # Plot Heatmap W
        path_W = plot_replica_heatmap(
            matrix_W, float(alpha), plots_dir,
            metric_name=f"Q_W ({metric_name})",
            filename_prefix="heatmap_W"
        )
        heatmap_paths_W.append(path_W)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
    
    # Generate GIF
    if heatmap_paths_W:
        create_gif(heatmap_paths_W, plots_dir / "animation_W.gif", duration=0.2)
        print("Generated animation_W.gif")
        
    # Generate Overlap Evolution (Root Dir)
    plot_overlap_evolution(all_results, output_dir, filename="overlap_evolution.png")
    print("Generated overlap_evolution.png")
    
    # Save Metrics
    import json
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump({
            'config': config.to_dict(),
            'results': all_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

if __name__ == "__main__":
    run_orthogonal_experiment()
