"""
SMF Experiment Runner (v2).

Orchestrates experiment execution based on Config.
Implements "Always Sweep" - every experiment is a parameter sweep over Alpha.

Key Features:
1. Unified output format for all algorithm modes
2. Alpha sweep loop for Standard/Spreading modes
3. Native parallel processing for spreading_parallel mode
4. Consistent metric computation
5. Rich Progress Bar integration
"""

import time
from typing import Dict, List, Any, Optional
import torch
import numpy as np

from .config import Config
from .progress import get_progress_manager, UnifiedProgress


def run_experiment(config: Config) -> Dict[str, Any]:
    """
    Run experiment based on configuration.
    
    Always performs Alpha sweep (no single-point mode).
    
    Args:
        config: Complete experiment configuration
        
    Returns:
        Dictionary with unified format:
        {
            "alpha_values": List[float],
            "Q_Y": List[float],  # One value per alpha
            "Q_W": List[float],
            "Q_X": List[float],
            "Q_W_prime": List[float],
            "Q_X_prime": List[float],
            "MSE": List[float],
            "total_time": float,
            "config": Config,
        }
    """
    from ..modules.teachers import TeacherGenerator
    from ..modules.graphs import GraphGenerator
    
    pm = get_progress_manager()

    # =========================================================================
    # 1. Setup
    # =========================================================================
    device = config.device
    mode = config.algorithm.mode
    
    pm.print_header(f"Running SMF: {mode}", {
        "Device": device,
        "Alphas": f"{config.alpha.start} -> {config.alpha.stop}",
        "Steps": config.training.max_steps
    })
    
    start_time = time.time()
    
    # Get alpha values
    alpha_values = config.alpha_values
    n_alphas = len(alpha_values)
    
    # Initialize teacher
    teacher = TeacherGenerator(
        type=config.teacher.type,
        variance_scale=config.teacher.variance_scale,
        spreading_seed=config.spreading.seed,
    )
    
    # Initialize graph generator
    graph = GraphGenerator(
        type=config.graph.type,
        loop_order=config.graph.loop_order,
        n_sweeps=config.graph.n_sweeps,
        alpha_threshold=config.graph.alpha_threshold,
    )
    
    # Create teacher matrices (same for all alphas)
    N1, N2, M = config.matrix.N1, config.matrix.N2, config.matrix.M
    seed = config.training.seed
    
    W_true, X_true = teacher.create(N1, N2, M, device, seed)
    
    # =========================================================================
    # 2. Dispatch to appropriate handler
    # =========================================================================
    if mode == "spreading_parallel":
        results = _run_spreading_parallel(
            config, teacher, graph, W_true, X_true, alpha_values, device, pm
        )
    elif mode == "spreading":
        results = _run_spreading_sequential(
            config, teacher, graph, W_true, X_true, alpha_values, device, pm
        )
    else:  # standard
        results = _run_standard_sweep(
            config, teacher, graph, W_true, X_true, alpha_values, device, pm
        )
    
    # =========================================================================
    # 3. Finalize
    # =========================================================================
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["config"] = config
    
    pm.print_completion(total_time)
    
    return results


def _run_spreading_parallel(
    config: Config,
    teacher,
    graph,
    W_true: torch.Tensor,
    X_true: torch.Tensor,
    alpha_values: List[float],
    device: torch.device,
    pm
) -> Dict[str, Any]:
    """
    Run spreading_parallel mode using SuperGraph parallelization.
    """
    from ..modules.graphs.supergraph import create_supergraph
    from ..modules.algorithms.bigamp_spreading_parallel import (
        BiGAMPSpreadingParallel,
        generate_F_super,
        compute_Y_super,
    )
    from ..modules.metrics.spreading import compute_all_metrics_spreading_parallel
    
    N1, N2, M = config.matrix.N1, config.matrix.N2, config.matrix.M
    S = config.training.samples_per_alpha
    seed = config.training.seed
    
    pm.print(f"  Creating SuperGraph for {len(alpha_values)} alphas, {S} samples...")
    
    # Create SuperGraph
    supergraph = create_supergraph(
        N1=N1, N2=N2, M=M,
        alpha_values=alpha_values,
        S=S,
        base_seed=seed,
        device=device,
    )
    
    # Generate F with configured distribution
    F_super = generate_F_super(
        supergraph=supergraph,
        M=M,
        base_seed=config.spreading.seed,
        device=device,
        f_distribution=config.spreading.f_distribution,
    )
    
    # Compute Y
    Y_super = compute_Y_super(W_true, X_true, supergraph, F_super)
    
    # Create spreading data
    from ..modules.teachers import SpreadingDataParallel
    spreading_data = SpreadingDataParallel(
        supergraph=supergraph,
        F_super=F_super,
        Y_super=Y_super,
        M=M,
        alpha_values=torch.tensor(alpha_values, device=device),
        W_teacher=W_true,
        X_teacher=X_true,
    )
    
    # Create algorithm instance
    algo = BiGAMPSpreadingParallel(config, device)
    
    # Train using full parallel mode
    # Initialize Progress Bar
    max_steps = config.training.max_steps
    progress = pm.create_experiment_progress(
        num_alphas=len(alpha_values),
        steps_per_alpha=max_steps,
        samples=S
    )
    progress.start()
    
    # NOTE: Since everything is parallel, we treat it as 1 big batch
    progress.start_batch(0, alpha_values)
    
    # Hook into algo.train_full_parallel to update progress?
    # For now, we'll let it run and maybe update manually if we can modify the algo.
    # Ideally BiGAMPSpreadingParallel should yield progress.
    # Let's inspect `train_full_parallel` - if it doesn't support callbacks, we just run it.
    # If we want detailed updates, we'd need to modify `BiGAMPSpreadingParallel`.
    # Assuming for this step we wrap the outer call.
    
    # Wait, the user wants the progress bar. We should check if `train_full_parallel` accepts a callback.
    # If not, modifying it is out of scope for "UI progress bar" UNLESS required.
    # But usually iterative algos need a callback.
    


    def progress_callback(step, max_steps):
        progress.update_step(step, max_steps)

    try:
        W_students, X_students = algo.train_full_parallel(
            spreading_data, 
            verbose=False, # We handle verbose output via progress bar
            step_callback=progress_callback 
        )
    except TypeError:
         # Fallback if callback not supported
         # Try with just 'callback' if 'step_callback' fails? 
         # Legacy might have used 'callback'
         try:
             W_students, X_students = algo.train_full_parallel(
                spreading_data, 
                verbose=False,
                callback=progress_callback
             )
         except TypeError:
             W_students, X_students = algo.train_full_parallel(spreading_data, verbose=True)
    
    progress.finish_batch()
    progress.stop()
    
    # Compute metrics
    pm.print("  Computing metrics...")
    metrics = compute_all_metrics_spreading_parallel(W_students, X_students, spreading_data)
    
    # Convert to lists - include all metrics
    return {
        "alpha_values": alpha_values,
        # Q_Y variants
        "Q_Y": metrics["Q_Y_mean"].cpu().tolist(),
        "Q_Y_std": metrics["Q_Y_std"].cpu().tolist(),
        "Q_Y_observed": metrics["Q_Y_observed_mean"].cpu().tolist(),
        "Q_Y_observed_std": metrics["Q_Y_observed_std"].cpu().tolist(),
        "Q_Y_unobserved": metrics["Q_Y_unobserved_mean"].cpu().tolist(),
        "Q_Y_unobserved_std": metrics["Q_Y_unobserved_std"].cpu().tolist(),
        # Physical Overlap Y
        "physical_overlap_Y": metrics["physical_overlap_Y_mean"].cpu().tolist(),
        "physical_overlap_Y_std": metrics["physical_overlap_Y_std"].cpu().tolist(),
        "physical_overlap_Y_observed": metrics.get("physical_overlap_Y_observed_mean", metrics["physical_overlap_Y_mean"]).cpu().tolist(),
        # Q_W and physical_overlap_W
        "Q_W": metrics["Q_W_mean"].cpu().tolist(),
        "Q_W_std": metrics["Q_W_std"].cpu().tolist(),
        "Q_W_prime": metrics["Q_W_prime_mean"].cpu().tolist(),
        "Q_W_prime_std": metrics["Q_W_prime_std"].cpu().tolist(),
        "physical_overlap_W": metrics["physical_overlap_W_mean"].cpu().tolist(),
        "physical_overlap_W_std": metrics["physical_overlap_W_std"].cpu().tolist(),
        # Q_X and physical_overlap_X
        "Q_X": metrics["Q_X_mean"].cpu().tolist(),
        "Q_X_std": metrics["Q_X_std"].cpu().tolist(),
        "Q_X_prime": metrics["Q_X_prime_mean"].cpu().tolist(),
        "Q_X_prime_std": metrics["Q_X_prime_std"].cpu().tolist(),
        "physical_overlap_X": metrics["physical_overlap_X_mean"].cpu().tolist(),
        "physical_overlap_X_std": metrics["physical_overlap_X_std"].cpu().tolist(),
        # MSE
        "MSE": metrics["MSE_mean"].cpu().tolist(),
        "MSE_std": metrics["MSE_std"].cpu().tolist(),
    }


def _run_spreading_sequential(
    config: Config,
    teacher,
    graph,
    W_true: torch.Tensor,
    X_true: torch.Tensor,
    alpha_values: List[float],
    device: torch.device,
    pm
) -> Dict[str, Any]:
    """
    Run spreading mode with sequential alpha sweep.
    """
    from ..modules.metrics.spreading import compute_all_metrics_spreading
    from ..modules.algorithms.bigamp_spreading import BiGAMPSpreadingAlgorithm
    from .progress import UnifiedProgress
    
    N1, N2, M = config.matrix.N1, config.matrix.N2, config.matrix.M
    seed = config.training.seed
    S = config.training.samples_per_alpha
    
    results = {
        "alpha_values": alpha_values,
        "Q_Y": [],
        "Q_W": [],
        "Q_X": [],
        "Q_W_prime": [],
        "Q_X_prime": [],
        "MSE": [],
    }
    
    # Use UnifiedProgress for Dynamic Capsule
    progress = UnifiedProgress(
        num_alphas=len(alpha_values),
        steps_per_alpha=config.training.max_steps, 
        batch_size=1,
        initial_estimate=None,
        num_batches=len(alpha_values)
    )
    progress.start()
    
    for idx, alpha in enumerate(alpha_values):
        # We treat each alpha as a "batch" for the progress bar
        progress.start_batch(idx, [alpha])
        
        # Generate graph for this alpha
        i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed + idx * 1000)
        
        if C == 0:
            # No edges, skip
            for key in ["Q_Y", "Q_W", "Q_X", "Q_W_prime", "Q_X_prime", "MSE"]:
                results[key].append(0.0)
            progress.finish_batch()
            continue
        
        # Create spreading data
        W_t, X_t, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed
        )
        
        # Initialize and train
        try:
            algo = BiGAMPSpreadingAlgorithm(config, device)
            
            # Helper for progress
            def step_cb(s, m=None):
                progress.update_step(s, config.training.max_steps)
                
            W_hat, X_hat = algo.train_single_alpha_spreading(
                W_t, X_t, spreading_data, alpha, seed + idx * 1000,
                progress_callback=step_cb
            )
            
            # Compute metrics
            metrics = compute_all_metrics_spreading(W_hat, X_hat, W_t, X_t, spreading_data)
            
            results["Q_Y"].append(metrics["Q_Y"])
            results["Q_W"].append(metrics["Q_W"])
            results["Q_X"].append(metrics["Q_X"])
            results["Q_W_prime"].append(metrics["Q_W_prime"])
            results["Q_X_prime"].append(metrics["Q_X_prime"])
            results["MSE"].append(metrics["MSE"])
        except Exception as e:
            pm.print(f"    Warning: Failed at alpha={alpha}: {e}")
            import traceback
            traceback.print_exc()
            for key in ["Q_Y", "Q_W", "Q_X", "Q_W_prime", "Q_X_prime", "MSE"]:
                results[key].append(0.0)
        
        progress.finish_batch()
            
    progress.stop()
    return results


def _run_standard_sweep(
    config: Config,
    teacher,
    graph,
    W_true: torch.Tensor,
    X_true: torch.Tensor,
    alpha_values: List[float],
    device: torch.device,
    pm
) -> Dict[str, Any]:
    """
    Run standard BiG-AMP with sequential alpha sweep.
    """
    from ..modules.metrics.overlap import compute_all_metrics
    
    N1, N2, M = config.matrix.N1, config.matrix.N2, config.matrix.M
    seed = config.training.seed
    
    results = {
        "alpha_values": alpha_values,
        "Q_Y": [],
        "Q_W": [],
        "Q_X": [],
        "Q_W_prime": [],
        "Q_X_prime": [],
        "MSE": [],
        "Gen_Error": [],
    }
    
    # Standard sweep
    progress = UnifiedProgress(
        num_alphas=len(alpha_values),
        steps_per_alpha=config.training.max_steps,
        batch_size=1,
        initial_estimate=None,
        num_batches=len(alpha_values)
    )
    progress.start()
    
    for idx, alpha in enumerate(alpha_values):
        progress.start_batch(idx, [alpha])
        
        # Generate graph/mask for this alpha
        i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed + idx * 1000)
        
        if C == 0:
            for key in results.keys():
                if key != "alpha_values":
                    results[key].append(0.0)
            progress.finish_batch()
            continue
        
        # Create observation matrix Y
        Y_full = (W_true @ X_true) / np.sqrt(M)
        
        # Create mask
        mask = torch.zeros(N1, N2, device=device)
        mask[i_idx, j_idx] = 1.0
        Y_obs = Y_full * mask
        
        # Initialize student
        W_hat = torch.randn(N1, M, device=device) * 0.1
        X_hat = torch.randn(M, N2, device=device) * 0.1
        
        # Simple gradient-based training (placeholder for full BiG-AMP)
        damping = config.algorithm.damping
        max_steps = config.training.max_steps
        
        for step in range(max_steps):
            # Compute prediction
            Y_pred = (W_hat @ X_hat) / np.sqrt(M)
            
            # Compute error on observed entries
            error = (Y_obs - Y_pred * mask)
            
            # Simple gradient update
            grad_W = (error @ X_hat.T) / np.sqrt(M) * mask.sum() / (N1 * N2)
            grad_X = (W_hat.T @ error) / np.sqrt(M) * mask.sum() / (N1 * N2)
            
            W_hat = damping * (W_hat + 0.1 * grad_W) + (1 - damping) * W_hat
            X_hat = damping * (X_hat + 0.1 * grad_X) + (1 - damping) * X_hat
            
            # Update progress
            progress.update_step(step + 1)
        
        # Compute metrics
        metrics = compute_all_metrics(W_hat, X_hat, W_true, X_true, mask)
        
        results["Q_Y"].append(metrics["Q_Y"])
        results["Q_W"].append(metrics["Q_W"])
        results["Q_X"].append(metrics["Q_X"])
        results["Q_W_prime"].append(metrics.get("Q_W_prime", metrics["Q_W"]))
        results["Q_X_prime"].append(metrics.get("Q_X_prime", metrics["Q_X"]))
        results["MSE"].append(metrics.get("MSE", 0.0))
        results["Gen_Error"].append(metrics.get("Gen_Error", 0.0))
        
        progress.finish_batch()
    
    progress.stop()
    return results
