"""
SMF Experiment Runner (v2).

Orchestrates experiment execution based on Config.
Implements "Always Sweep" - every experiment is a parameter sweep over Alpha.

Key Features:
1. Unified output format for all algorithm modes
2. Alpha sweep loop for Standard/Spreading modes
3. Native parallel processing for spreading_parallel mode
4. Consistent metric computation
"""

import time
from typing import Dict, List, Any, Optional
import torch
import numpy as np

from .config import Config


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
    
    # =========================================================================
    # 1. Setup
    # =========================================================================
    device = config.device
    mode = config.algorithm.mode
    
    print(f"Running on {device} | Mode: {mode}")
    print(f"Alpha sweep: {config.alpha.start} -> {config.alpha.stop} (step={config.alpha.step})")
    
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
            config, teacher, graph, W_true, X_true, alpha_values, device
        )
    elif mode == "spreading":
        results = _run_spreading_sequential(
            config, teacher, graph, W_true, X_true, alpha_values, device
        )
    else:  # standard
        results = _run_standard_sweep(
            config, teacher, graph, W_true, X_true, alpha_values, device
        )
    
    # =========================================================================
    # 3. Finalize
    # =========================================================================
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["config"] = config
    
    print(f"Experiment completed in {total_time:.2f}s")
    
    return results


def _run_spreading_parallel(
    config: Config,
    teacher,
    graph,
    W_true: torch.Tensor,
    X_true: torch.Tensor,
    alpha_values: List[float],
    device: torch.device,
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
    
    print(f"  Creating SuperGraph for {len(alpha_values)} alphas, {S} samples...")
    
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
    print(f"  Training with {config.training.max_steps} steps...")
    W_students, X_students = algo.train_full_parallel(spreading_data, verbose=True)
    
    # Compute metrics
    print("  Computing metrics...")
    metrics = compute_all_metrics_spreading_parallel(W_students, X_students, spreading_data)
    
    # Convert to lists
    return {
        "alpha_values": alpha_values,
        "Q_Y": metrics["Q_Y_mean"].cpu().tolist(),
        "Q_Y_std": metrics["Q_Y_std"].cpu().tolist(),
        "Q_Y_observed": metrics["Q_Y_observed_mean"].cpu().tolist(),
        "Q_Y_unobserved": metrics["Q_Y_unobserved_mean"].cpu().tolist(),
        "Q_W": metrics["Q_W_mean"].cpu().tolist(),
        "Q_W_std": metrics["Q_W_std"].cpu().tolist(),
        "Q_X": metrics["Q_X_mean"].cpu().tolist(),
        "Q_X_std": metrics["Q_X_std"].cpu().tolist(),
        "Q_W_prime": metrics["Q_W_prime_mean"].cpu().tolist(),
        "Q_X_prime": metrics["Q_X_prime_mean"].cpu().tolist(),
        "MSE": metrics["MSE_mean"].cpu().tolist(),
        "physical_overlap_Y": metrics.get("physical_overlap_Y_mean", torch.zeros(len(alpha_values))).cpu().tolist(),
    }


def _run_spreading_sequential(
    config: Config,
    teacher,
    graph,
    W_true: torch.Tensor,
    X_true: torch.Tensor,
    alpha_values: List[float],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run spreading mode with sequential alpha sweep.
    """
    from ..modules.metrics.spreading import compute_all_metrics_spreading
    from ..modules.algorithms.bigamp_spreading import BiGAMPSpreading
    
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
    
    for idx, alpha in enumerate(alpha_values):
        print(f"  Alpha {idx+1}/{len(alpha_values)}: {alpha:.2f}")
        
        # Generate graph for this alpha
        i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed + idx * 1000)
        
        if C == 0:
            # No edges, skip
            for key in ["Q_Y", "Q_W", "Q_X", "Q_W_prime", "Q_X_prime", "MSE"]:
                results[key].append(0.0)
            continue
        
        # Create spreading data
        W_t, X_t, spreading_data = teacher.create_with_spreading(
            N1, N2, M, i_idx, j_idx, device, seed
        )
        
        # Initialize and train
        try:
            algo = BiGAMPSpreading(config, device)
            W_hat, X_hat = algo.train(spreading_data, W_t, X_t)
            
            # Compute metrics
            metrics = compute_all_metrics_spreading(W_hat, X_hat, W_t, X_t, spreading_data)
            
            results["Q_Y"].append(metrics["Q_Y"])
            results["Q_W"].append(metrics["Q_W"])
            results["Q_X"].append(metrics["Q_X"])
            results["Q_W_prime"].append(metrics["Q_W_prime"])
            results["Q_X_prime"].append(metrics["Q_X_prime"])
            results["MSE"].append(metrics["MSE"])
        except Exception as e:
            print(f"    Warning: Failed at alpha={alpha}: {e}")
            for key in ["Q_Y", "Q_W", "Q_X", "Q_W_prime", "Q_X_prime", "MSE"]:
                results[key].append(0.0)
    
    return results


def _run_standard_sweep(
    config: Config,
    teacher,
    graph,
    W_true: torch.Tensor,
    X_true: torch.Tensor,
    alpha_values: List[float],
    device: torch.device,
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
    
    for idx, alpha in enumerate(alpha_values):
        print(f"  Alpha {idx+1}/{len(alpha_values)}: {alpha:.2f}")
        
        # Generate graph/mask for this alpha
        i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed + idx * 1000)
        
        if C == 0:
            for key in results.keys():
                if key != "alpha_values":
                    results[key].append(0.0)
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
        
        # Compute metrics
        metrics = compute_all_metrics(W_hat, X_hat, W_true, X_true, mask)
        
        results["Q_Y"].append(metrics["Q_Y"])
        results["Q_W"].append(metrics["Q_W"])
        results["Q_X"].append(metrics["Q_X"])
        results["Q_W_prime"].append(metrics.get("Q_W_prime", metrics["Q_W"]))
        results["Q_X_prime"].append(metrics.get("Q_X_prime", metrics["Q_X"]))
        results["MSE"].append(metrics.get("MSE", 0.0))
        results["Gen_Error"].append(metrics.get("Gen_Error", 0.0))
    
    return results
