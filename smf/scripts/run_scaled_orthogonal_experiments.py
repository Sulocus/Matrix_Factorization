#!/usr/bin/env python3
"""
Run Scaled Orthogonal Experiments.

Experiments:
1. Spreading (Orthogonal): N=200, M=50, Steps=2000 (Re-run)
2. BiG-AMP (Orthogonal): N=400, M=100, Steps=4000 (Sequential)
3. Spreading (Orthogonal): N=400, M=100, Steps=4000 (Parallel, Low Batch Size)

Common:
- S = 50
- Alpha: 0.0 ~ 4.0
"""

import sys
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import torch
import gc

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
from smf.modules.algorithms.bigamp_spreading_parallel import run_spreading_parallel

def run_exp1_spreading_200():
    """Exp 1: Spreading N=200, Steps=2000 (Orthogonal)"""
    print("\n" + "="*80)
    print("EXP 1: Spreading (Orthogonal) N=200 Steps=2000")
    print("="*80)
    
    N, M, S = 200, 50, 50
    steps = 2000
    
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.05),
        training=TrainingConfig(max_steps=steps, samples_per_alpha=S, seed=42),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key="bigamp_spreading_parallel",
        teacher_key="orthogonal",
        spreading=SpreadingConfig(f_distribution="rademacher", seed=1042),
        execution=ExecutionConfig(matrix_metric="gram_overlap_normalized")
    )
    
    # Run
    # N=200 is small enough for batch_size=10
    result = run_spreading_parallel(config, verbose=True, alpha_batch_size=10)
    
    # Process & Plot
    process_parallel_results(result, "bigamp_spreading_parallel_orthogonal_200", config)


def run_exp2_bigamp_400():
    """Exp 2: BiG-AMP N=400, Steps=4000 (Orthogonal, Sequential)"""
    print("\n" + "="*80)
    print("EXP 2: BiG-AMP (Orthogonal) N=400 Steps=4000")
    print("="*80)
    
    N, M, S = 400, 100, 50
    steps = 4000
    
    device, _ = setup_device()
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path("smf/Replica_results") / f"bigamp_orthogonal_{N}x{N}_M{M}_S{S}_steps{steps}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.05),
        training=TrainingConfig(max_steps=steps, samples_per_alpha=S, seed=43),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key="bigamp",
        teacher_key="orthogonal",
        execution=ExecutionConfig(matrix_metric="gram_overlap_normalized")
    )
    
    teacher_cls = get_teacher("orthogonal").cls
    algorithm_cls = get_algorithm("bigamp").cls
    graph_cls = get_graph("random").cls
    
    teacher = teacher_cls()
    algorithm = algorithm_cls(config, device)
    graph = graph_cls()
    
    torch.manual_seed(config.training.seed)
    W_t, X_t, _ = teacher.create_with_Y(N, N, M, device, config.training.seed)
    
    alpha_values = config.alpha.get_values()
    all_results = {}
    heatmap_paths_W = []
    
    start_time = time.time()
    
    for idx, alpha in enumerate(alpha_values):
        mask_seed = config.training.seed + int(alpha * 1000)
        mask, _ = graph.generate_mask(N, N, M, alpha, device, mask_seed)
        
        W_s, X_s = algorithm.train_single_alpha(
            W_t, X_t, None, mask, alpha, mask_seed + 10000,
        )
        
        elapsed = time.time() - start_time
        eta = elapsed / (idx + 1) * (len(alpha_values) - idx - 1)
        print(f"\r  [{idx+1}/{len(alpha_values)}] α={alpha:.2f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", end="", flush=True)
        
        # Metrics
        metric_fn = gram_overlap_normalized
        matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
        
        metrics = {'interaction_matrix_W': matrix_W.tolist()}
        # Compute mean Q_W just for sanity if needed, but interaction matrix is enough for plots
        
        all_results[float(alpha)] = metrics
        
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
        
    save_final_results(output_dir, config, all_results)


def run_exp4_bigamp_400_standard():
    """Exp 4: BiG-AMP N=400, Steps=4000 (Standard Teacher, non-orthogonal)"""
    print("\n" + "="*80)
    print("EXP 4: BiG-AMP (Standard) N=400 Steps=4000")
    print("="*80)
    
    N, M, S = 400, 100, 50
    steps = 4000
    
    device, _ = setup_device()
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path("smf/Replica_results") / f"bigamp_standard_{N}x{N}_M{M}_S{S}_steps{steps}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.05),
        training=TrainingConfig(max_steps=steps, samples_per_alpha=S, seed=45),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key="bigamp",
        teacher_key="standard",  # <-- Non-Orthogonal
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
    
    alpha_values = config.alpha.get_values()
    all_results = {}
    heatmap_paths_W = []
    
    start_time = time.time()
    
    for idx, alpha in enumerate(alpha_values):
        mask_seed = config.training.seed + int(alpha * 1000)
        mask, _ = graph.generate_mask(N, N, M, alpha, device, mask_seed)
        
        W_s, X_s = algorithm.train_single_alpha(
            W_t, X_t, None, mask, alpha, mask_seed + 10000,
        )
        
        elapsed = time.time() - start_time
        eta = elapsed / (idx + 1) * (len(alpha_values) - idx - 1)
        print(f"\r  [{idx+1}/{len(alpha_values)}] α={alpha:.2f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", end="", flush=True)
        
        metric_fn = gram_overlap_normalized
        matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
        
        all_results[float(alpha)] = {'interaction_matrix_W': matrix_W.tolist()}
        
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
    
    if heatmap_paths_W:
        create_gif(heatmap_paths_W, plots_dir / "animation_W.gif", duration=0.2)
        
    save_final_results(output_dir, config, all_results)


def run_exp3_spreading_400():
    """Exp 3: Spreading N=400, Steps=4000 (Orthogonal)"""
    print("\n" + "="*80)
    print("EXP 3: Spreading (Orthogonal) N=400 Steps=4000")
    print("="*80)
    
    N, M, S = 400, 100, 50
    steps = 4000
    
    config = Config(
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.05),
        training=TrainingConfig(max_steps=steps, samples_per_alpha=S, seed=44),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        algorithm_key="bigamp_spreading_parallel",
        teacher_key="orthogonal",
        spreading=SpreadingConfig(f_distribution="rademacher", seed=1044),
        execution=ExecutionConfig(matrix_metric="gram_overlap_normalized")
    )
    
    # Run with reduced batch size to avoid OOM
    # N doubled -> Memory x4. 
    # Previous batch=10. Safe new batch = 10/4 ~ 2.
    result = run_spreading_parallel(config, verbose=True, alpha_batch_size=2)
    
    process_parallel_results(result, "bigamp_spreading_parallel_orthogonal_400", config)


def process_parallel_results(result, name_prefix, config):
    """Common processing for parallel results."""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    N = config.matrix.N1
    M = config.matrix.M
    S = config.training.samples_per_alpha
    max_steps = config.training.max_steps
    
    output_dir = Path("smf/Replica_results") / f"{name_prefix}_{N}x{N}_M{M}_S{S}_steps{max_steps}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    # Extract
    W_students = result['W_students']
    spreading_data = result['spreading_data']
    W_t = spreading_data.W_teacher
    
    alpha_values = config.alpha.get_values()
    all_results = {}
    heatmap_paths_W = []
    
    metric_fn = gram_overlap_normalized
    
    print("Generating plots and saving data...")
    for i, alpha in enumerate(alpha_values):
        W_s = W_students[:, i, :, :]
        
        matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
        
        all_results[float(alpha)] = {'interaction_matrix_W': matrix_W.tolist()}
        
        # Save Raw (Must include W_teacher for physical metric!)
        raw_path = raw_dir / f"alpha_{alpha:.6f}.npz"
        np.savez_compressed(raw_path,
            W_student=W_s.detach().cpu().to(torch.float16).numpy(),
            # Save teacher (shared) in every file for simplicity of visualizer
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
        
    save_final_results(output_dir, config, all_results)

def save_final_results(output_dir, config, all_results):
    import json
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump({
            'config': config.to_dict(),
            'results': all_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        
    print(f"Saved results to {output_dir}")
    
    # Call Visualize Script
    from smf.scripts.visualize_replica_results import process_experiment
    try:
        process_experiment(output_dir)
        print("Visualization script finished successfully.")
    except Exception as e:
        print(f"Visualization script failed: {e}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    try:
        # Exp 1 Completed Successfully
        # run_exp1_spreading_200()
        
        # time.sleep(5)
        # torch.cuda.empty_cache()
        # gc.collect()
        
        run_exp2_bigamp_400()
        time.sleep(5)
        torch.cuda.empty_cache()
        gc.collect()
        
        run_exp3_spreading_400()
        time.sleep(5)
        torch.cuda.empty_cache()
        gc.collect()
        
        run_exp4_bigamp_400_standard()
    except Exception as e:
        print(f"Comparison Suite Failed: {e}")
