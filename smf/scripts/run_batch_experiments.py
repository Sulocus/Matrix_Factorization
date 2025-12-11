#!/usr/bin/env python3
"""
Batch Replica Analysis: 4 Experiments (2 algorithms × 2 step counts).

Experiments:
1. BiG-AMP + 1000 steps
2. BiG-AMP + 2000 steps
3. Spreading (Rademacher) + 1000 steps
4. Spreading (Rademacher) + 2000 steps

Parameters:
- N = 200, M = 50
- S = 50 (replicas)
- Alpha: 0.0 ~ 4.0, step = 0.05 (81 points)
- Metric: gram_overlap_normalized
"""

import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

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


def run_single_experiment(
    algorithm_key: str,
    max_steps: int,
    N: int = 200,
    M: int = 50,
    S: int = 50,
    alpha_start: float = 0.0,
    alpha_stop: float = 4.0,
    alpha_step: float = 0.05,
    f_distribution: str = "rademacher",
    metric_name: str = "gram_overlap_normalized",
    seed: int = 42,
):
    """Run a single experiment with given parameters."""
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {algorithm_key.upper()} | Steps={max_steps}")
    print("=" * 70)
    
    # Setup device
    device, device_info = setup_device()
    print(f"Device: {device_info.device_name}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path("smf/Replica_results") / f"{algorithm_key}_{N}x{N}_M{M}_S{S}_steps{max_steps}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    print(f"Output: {output_dir}")
    
    # Create config
    spreading_config = None
    if "spreading" in algorithm_key:
        spreading_config = SpreadingConfig(
            f_distribution=f_distribution,
            seed=seed + 1000,
        )
    
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
        training=TrainingConfig(max_steps=max_steps, samples_per_alpha=S, seed=seed),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key=algorithm_key,
        graph_key="random",
        teacher_key="standard",
        spreading=spreading_config,
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
    np.random.seed(config.training.seed)
    W_t, X_t, Y_t = teacher.create_with_Y(N, N, M, device, config.training.seed)
    
    print(f"Teacher: W={W_t.shape}, X={X_t.shape}")
    
    # Get metric function
    try:
        metric_fn = get_metric_function(metric_name)
    except ValueError:
        metric_fn = gram_overlap_normalized
    
    # Get alpha values
    alpha_values = config.alpha.get_values()
    print(f"Alpha: {alpha_values[0]:.2f} ~ {alpha_values[-1]:.2f} ({len(alpha_values)} points)")
    print(f"F distribution: {f_distribution if spreading_config else 'N/A'}")
    
    # Storage for results
    all_results = {}
    heatmap_paths_W = []
    heatmap_paths_X = []
    
    start_time = time.time()
    
    # Run training for each alpha
    for idx, alpha in enumerate(alpha_values):
        # Generate mask
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
        
        # Compute interaction matrices
        matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
        matrix_X = build_interaction_matrix(X_s, X_t, metric_fn, use_left=False)
        
        metrics['interaction_matrix_W'] = matrix_W.tolist()
        metrics['interaction_matrix_X'] = matrix_X.tolist()
        
        all_results[float(alpha)] = metrics
        
        # Save raw data (float16)
        raw_path = raw_dir / f"alpha_{alpha:.6f}.npz"
        save_dict = {
            'W_student': W_s.detach().cpu().to(torch.float16).numpy(),
            'X_student': X_s.detach().cpu().to(torch.float16).numpy(),
            'interaction_matrix_W': matrix_W.astype(np.float16),
            'interaction_matrix_X': matrix_X.astype(np.float16),
        }
        np.savez_compressed(raw_path, **save_dict)
        
        # Generate heatmaps
        path_W = plot_replica_heatmap(
            matrix_W, float(alpha), plots_dir,
            metric_name=f"Q_W ({metric_name})",
            filename_prefix="heatmap_W"
        )
        heatmap_paths_W.append(path_W)
        
        if config.matrix.N1 != config.matrix.N2:
            path_X = plot_replica_heatmap(
                matrix_X, float(alpha), plots_dir,
                metric_name=f"Q_X ({metric_name})",
                filename_prefix="heatmap_X"
            )
            heatmap_paths_X.append(path_X)
        
        # Clear cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s")
    
    # Generate GIFs
    print("\nGenerating GIFs...")
    if heatmap_paths_W:
        create_gif(heatmap_paths_W, plots_dir / "animation_W.gif", duration=0.2)
        print(f"  Created: animation_W.gif")
        
    if heatmap_paths_X:
        create_gif(heatmap_paths_X, plots_dir / "animation_X.gif", duration=0.2)
        print(f"  Created: animation_X.gif")
    
    # Generate Comparison Plot
    plot_overlap_evolution(all_results, plots_dir)
    print(f"  Created: overlap_evolution.png")
    
    # Save config
    config.to_yaml(output_dir / "config.yaml")
    
    # Save metrics JSON
    import json
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump({
            'config': config.to_dict(),
            'results': all_results,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    
    print(f"Results saved to: {output_dir}")
    
    return output_dir


def run_spreading_experiment(
    max_steps: int,
    N: int = 200,
    M: int = 50,
    S: int = 50,
    alpha_start: float = 0.0,
    alpha_stop: float = 4.0,
    alpha_step: float = 0.05,
    f_distribution: str = "rademacher",
    metric_name: str = "gram_overlap_normalized",
    seed: int = 42,
):
    """Run a Spreading experiment using the dedicated API."""
    from smf.modules.algorithms.bigamp_spreading_parallel import (
        BiGAMPSpreadingParallel, run_spreading_parallel
    )
    
    algorithm_key = "bigamp_spreading_parallel"
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {algorithm_key.upper()} | Steps={max_steps}")
    print("=" * 70)
    
    # Setup device
    device, device_info = setup_device()
    print(f"Device: {device_info.device_name}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path("smf/Replica_results") / f"{algorithm_key}_{N}x{N}_M{M}_S{S}_steps{max_steps}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    print(f"Output: {output_dir}")
    
    # Create config
    spreading_config = SpreadingConfig(
        f_distribution=f_distribution,
        seed=seed + 1000,
    )
    
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
        training=TrainingConfig(max_steps=max_steps, samples_per_alpha=S, seed=seed),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key=algorithm_key,
        graph_key="random",
        teacher_key="standard",
        spreading=spreading_config,
        execution=ExecutionConfig(matrix_metric=metric_name),
    )
    
    print(f"F distribution: {f_distribution}")
    
    # Get metric function
    try:
        metric_fn = get_metric_function(metric_name)
    except ValueError:
        metric_fn = gram_overlap_normalized
    
    start_time = time.time()
    
    # Run using the dedicated spreading API
    result = run_spreading_parallel(config, verbose=True)
    
    # Extract results
    W_students = result['W_students']  # (S, A, N1, M)
    X_students = result['X_students']  # (S, A, M, N2)
    spreading_data = result['spreading_data']
    
    alpha_values = config.alpha.get_values()
    A = len(alpha_values)
    
    # Get teacher from spreading_data
    W_t = spreading_data.W_teacher
    X_t = spreading_data.X_teacher
    
    print(f"\nProcessing {A} alpha values for heatmaps...")
    
    all_results = result['results']
    heatmap_paths_W = []
    heatmap_paths_X = []
    
    # Process each alpha for heatmaps
    for i, alpha in enumerate(alpha_values):
        # W_students: (S, A, N1, M) -> take [:, i, :, :] for alpha i
        W_s = W_students[:, i, :, :]  # (S, N1, M)
        X_s = X_students[:, i, :, :]  # (S, M, N2)
        
        # Compute interaction matrices
        matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
        matrix_X = build_interaction_matrix(X_s, X_t, metric_fn, use_left=False)
        
        # Update results with matrices
        all_results[float(alpha)]['interaction_matrix_W'] = matrix_W.tolist()
        all_results[float(alpha)]['interaction_matrix_X'] = matrix_X.tolist()
        
        # Save raw data (float16)
        raw_path = raw_dir / f"alpha_{alpha:.6f}.npz"
        save_dict = {
            'W_student': W_s.detach().cpu().to(torch.float16).numpy(),
            'X_student': X_s.detach().cpu().to(torch.float16).numpy(),
            'interaction_matrix_W': matrix_W.astype(np.float16),
            'interaction_matrix_X': matrix_X.astype(np.float16),
        }
        np.savez_compressed(raw_path, **save_dict)
        
        # Generate heatmaps
        path_W = plot_replica_heatmap(
            matrix_W, float(alpha), plots_dir,
            metric_name=f"Q_W ({metric_name})",
            filename_prefix="heatmap_W"
        )
        heatmap_paths_W.append(path_W)
        
        if config.matrix.N1 != config.matrix.N2:
            path_X = plot_replica_heatmap(
                matrix_X, float(alpha), plots_dir,
                metric_name=f"Q_X ({metric_name})",
                filename_prefix="heatmap_X"
            )
            heatmap_paths_X.append(path_X)
        
        print(f"\r  [{i+1}/{A}] Processed α={alpha:.2f}", end="", flush=True)
    
    print()  # Newline
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s")
    
    # Generate GIFs
    print("Generating GIFs...")
    gif_W = create_gif(heatmap_paths_W, plots_dir / "animation_W.gif", duration=0.3)
    if heatmap_paths_X:
        create_gif(heatmap_paths_X, plots_dir / "animation_X.gif", duration=0.2)
        print(f"  Created: animation_X.gif")
    
    # Generate Comparison Plot
    plot_overlap_evolution(all_results, plots_dir)
    print(f"  Created: overlap_evolution.png")
    
    # Save config
    config.to_yaml(output_dir / "config.yaml")
    
    # Save metrics JSON
    import json
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump({
            'config': config.to_dict(),
            'results': all_results,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    
    print(f"Results saved to: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    print("=" * 70)
    print("BATCH EXPERIMENT: 4 Runs (2 Algorithms × 2 Step Counts)")
    print("=" * 70)
    print("Parameters: N=200, M=50, S=50, Alpha=0.0~4.0 step=0.05")
    print("=" * 70)
    
    results = []
    
    # Experiment 1: BiG-AMP + 1000 steps
    r1 = run_single_experiment(
        algorithm_key="bigamp",
        max_steps=1000,
    )
    results.append(("bigamp_1000", r1))
    
    # Experiment 2: BiG-AMP + 2000 steps
    r2 = run_single_experiment(
        algorithm_key="bigamp",
        max_steps=2000,
    )
    results.append(("bigamp_2000", r2))
    
    # Experiment 3: Spreading + 1000 steps
    r3 = run_spreading_experiment(
        max_steps=1000,
        f_distribution="rademacher",
    )
    results.append(("spreading_1000", r3))
    
    # Experiment 4: Spreading + 2000 steps
    r4 = run_spreading_experiment(
        max_steps=2000,
        f_distribution="rademacher",
    )
    results.append(("spreading_2000", r4))
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    for name, path in results:
        print(f"  {name}: {path}")
    print("=" * 70)
