#!/usr/bin/env python3
"""
Replica Analysis Test Script.

This script tests the new replica analysis features:
1. Raw data storage (float16 npz)
2. Interaction matrix computation (S+1 x S+1)
3. Heatmap visualization (RdYlBu_r colormap)
4. GIF generation

Results are saved to: smf/Replica_results/{algorithm}_{params}_{timestamp}/
"""

import sys
from pathlib import Path
from datetime import datetime

# Add smf to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, AlgorithmConfig, ExecutionConfig
from smf.core.device import setup_device
from smf.modules.registry import get_algorithm, get_graph, get_teacher
from smf.modules.metrics.overlap import (
    compute_all_metrics, 
    aggregate_trial_metrics,
    build_interaction_matrix,
    get_metric_function,
    gram_overlap_normalized,
)
from smf.modules.outputs.plotting import plot_replica_heatmap, create_gif


def run_replica_test(
    N: int = 100,
    M: int = 25,
    S: int = 4,
    alpha_start: float = 1.0,
    alpha_stop: float = 3.0,
    alpha_step: float = 0.5,
    max_steps: int = 200,
    algorithm_key: str = "bigamp",
    metric_name: str = "gram_overlap_normalized",
):
    """
    Run a small replica analysis test.
    
    Args:
        N: Matrix dimension (N x N)
        M: Hidden dimension
        S: Number of replicas
        alpha_start/stop/step: Alpha sweep range
        max_steps: Training steps
        algorithm_key: Algorithm to use
        metric_name: Metric for interaction matrix
    """
    print("=" * 60)
    print("REPLICA ANALYSIS TEST")
    print("=" * 60)
    
    # Setup device
    device, device_info = setup_device()
    print(f"Device: {device_info.device_name}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path("smf/Replica_results") / f"{algorithm_key}_{N}x{N}_M{M}_S{S}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    print(f"Output: {output_dir}")
    
    # Create config
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
        training=TrainingConfig(max_steps=max_steps, samples_per_alpha=S, seed=42),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key=algorithm_key,
        graph_key="random",
        teacher_key="standard",
        execution=ExecutionConfig(matrix_metric=metric_name),
    )
    
    # Get components
    algorithm_cls = get_algorithm(config.algorithm_key).cls
    graph_cls = get_graph(config.graph_key).cls
    teacher_cls = get_teacher(config.teacher_key).cls
    
    # Create instances
    algorithm = algorithm_cls(config, device)
    graph = graph_cls()
    teacher = teacher_cls()
    
    # Create teacher model
    torch.manual_seed(config.training.seed)
    W_t, X_t, Y_t = teacher.create_with_Y(N, N, M, device, config.training.seed)
    
    print(f"\nTeacher created: W={W_t.shape}, X={X_t.shape}")
    
    # Get metric function
    try:
        metric_fn = get_metric_function(metric_name)
    except ValueError:
        print(f"Warning: Unknown metric {metric_name}, using gram_overlap_normalized")
        metric_fn = gram_overlap_normalized
    
    # Get alpha values
    alpha_values = config.alpha.get_values()
    print(f"Alpha values: {alpha_values}")
    
    # Storage for results and heatmap paths
    all_results = {}
    heatmap_paths_W = []
    heatmap_paths_X = []
    
    # Run training for each alpha
    for alpha in alpha_values:
        print(f"\n--- Alpha = {alpha:.2f} ---")
        
        # Generate mask
        mask_seed = config.training.seed + int(alpha * 1000)
        mask, _ = graph.generate_mask(N, N, M, alpha, device, mask_seed)
        
        # Train
        W_s, X_s = algorithm.train_single_alpha(
            W_t, X_t, Y_t, mask, alpha, mask_seed + 10000,
            progress_callback=lambda s, t: print(f"\r  Step {s}/{t}", end="") if s % 50 == 0 else None,
        )
        print()  # Newline after progress
        
        # W_s: (S, N, M), X_s: (S, M, N)
        print(f"  Students: W_s={W_s.shape}, X_s={X_s.shape}")
        
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
        
        print(f"  Q_Y_mean = {metrics.get('Q_Y_mean', 'N/A'):.4f}")
        
        # Compute interaction matrices
        matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
        matrix_X = build_interaction_matrix(X_s, X_t, metric_fn, use_left=False)
        
        print(f"  Interaction Matrix W: shape={matrix_W.shape}, diag[0]={matrix_W[0,0]:.4f}")
        print(f"  Interaction Matrix X: shape={matrix_X.shape}, diag[0]={matrix_X[0,0]:.4f}")
        
        # Store matrices in metrics
        metrics['interaction_matrix_W'] = matrix_W.tolist()
        metrics['interaction_matrix_X'] = matrix_X.tolist()
        
        all_results[float(alpha)] = metrics
        
        # Save raw data (float16)
        raw_path = raw_dir / f"alpha_{alpha:.6f}.npz"
        save_dict = {
            'W_student': W_s.detach().cpu().to(torch.float16).numpy(),
            'X_student': X_s.detach().cpu().to(torch.float16).numpy(),
            'W_teacher': W_t.detach().cpu().to(torch.float16).numpy(),
            'X_teacher': X_t.detach().cpu().to(torch.float16).numpy(),
            'mask': mask.detach().cpu().to(torch.float16).numpy(),
            'interaction_matrix_W': matrix_W.astype(np.float16),
            'interaction_matrix_X': matrix_X.astype(np.float16),
        }
        np.savez_compressed(raw_path, **save_dict)
        print(f"  Saved raw data: {raw_path.name}")
        
        # Generate heatmaps
        path_W = plot_replica_heatmap(
            matrix_W, float(alpha), plots_dir,
            metric_name=f"Q_W ({metric_name})",
            filename_prefix="heatmap_W"
        )
        path_X = plot_replica_heatmap(
            matrix_X, float(alpha), plots_dir,
            metric_name=f"Q_X ({metric_name})",
            filename_prefix="heatmap_X"
        )
        heatmap_paths_W.append(path_W)
        heatmap_paths_X.append(path_X)
        print(f"  Generated heatmaps: {path_W.name}, {path_X.name}")
    
    # Generate GIFs
    print("\n--- Generating GIFs ---")
    gif_W = create_gif(heatmap_paths_W, plots_dir / "animation_W.gif", duration=0.5)
    gif_X = create_gif(heatmap_paths_X, plots_dir / "animation_X.gif", duration=0.5)
    
    if gif_W:
        print(f"Created: {gif_W}")
    if gif_X:
        print(f"Created: {gif_X}")
    
    # Save config
    config.to_yaml(output_dir / "config.yaml")
    
    # Save metrics JSON
    import json
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump({
            'config': config.to_dict(),
            'results': all_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    return output_dir


if __name__ == "__main__":
    # Run test with small parameters for quick verification
    run_replica_test(
        N=100,
        M=25,
        S=4,
        alpha_start=1.0,
        alpha_stop=3.0,
        alpha_step=0.5,
        max_steps=200,
        algorithm_key="bigamp",
        metric_name="gram_overlap_normalized",
    )
