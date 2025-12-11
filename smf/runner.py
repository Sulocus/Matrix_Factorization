"""
Experiment runner - orchestrates training and evaluation.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import torch
import numpy as np

from .core.config import Config
from .core.device import setup_device, get_compute_dtype
from .core.progress import ProgressManager, get_progress_manager, UnifiedProgress
from .core.time_estimator import estimate_time
from .core.memory_manager import select_memory_mode, MemoryMode
from .core.checkpoint import CheckpointManager
# Import modules package to trigger registration
from . import modules
from .modules.registry import get_algorithm, get_graph, get_teacher
from .modules.metrics.overlap import (
    compute_all_metrics, 
    aggregate_trial_metrics, 
    compute_replica_overlap,
    build_interaction_matrix,
    get_metric_function,
    gram_overlap_normalized
)
from .modules.outputs.storage import ResultStorage
from smf.modules.outputs.plotting import plot_replica_heatmap, create_gif, plot_overlap_evolution


class ExperimentRunner:
    """
    Runs experiments with the configured modules.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device, self.device_info = setup_device()
        self.progress = get_progress_manager()

        # Get modules
        self.algorithm_cls = get_algorithm(config.algorithm_key).cls
        self.graph_cls = get_graph(config.graph_key).cls
        self.teacher_cls = get_teacher(config.teacher_key).cls

    def run(self, save_results: bool = True, resume: bool = False) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Args:
            save_results: Whether to save results to disk
            resume: If True, resume from last checkpoint

        Returns:
            Dictionary with all results
        """
        m = self.config.matrix
        alpha_values = self.config.alpha.get_values()
        S = self.config.training.samples_per_alpha
        seed = self.config.training.seed

        # Initialize checkpoint manager
        output_dir = Path(self.config.output.base_dir) if hasattr(self.config, 'output') else Path("results")
        checkpoint_mgr = CheckpointManager(output_dir, self.config)

        # Check for resume
        all_results = {}
        remaining_alphas = alpha_values

        if resume:
            checkpoint_info = checkpoint_mgr.get_checkpoint_info()
            if checkpoint_info:
                checkpoint = checkpoint_mgr.load_latest()
                if checkpoint:
                    all_results = checkpoint.results.copy()
                    remaining_alphas = checkpoint_mgr.get_remaining_alphas(alpha_values)
                    print(f"\n[Resume] Found checkpoint: {checkpoint_info['num_completed']}/{len(alpha_values)} completed")
                    print(f"[Resume] Continuing from α = {remaining_alphas[0]:.2f}" if remaining_alphas else "[Resume] All alphas completed!")

        if not remaining_alphas:
            # All alphas already completed
            return {
                'results': all_results,
                'config': self.config,
                'total_time': 0.0,
                'result_path': None,
                'resumed': True,
            }

        # Select memory strategy
        alpha_max = max(alpha_values) if alpha_values else 4.0
        strategy = select_memory_mode(
            N1=m.N1, N2=m.N2, M=m.M, S=S,
            num_alphas=len(remaining_alphas),
            available_gb=self.device_info.available_memory_gb if hasattr(self.device_info, 'available_memory_gb') else None,
            verbose=True,
            algorithm_key=self.config.algorithm_key,
            alpha_max=alpha_max,
        )

        # Print header
        resume_info = f" (resumed, {len(alpha_values) - len(remaining_alphas)} done)" if resume and len(remaining_alphas) < len(alpha_values) else ""
        self.progress.print_header(
            f"SMF Experiment - {self.config.algorithm_key.upper()}{resume_info}",
            {
                "Matrix": f"{m.N1}×{m.N2}, M={m.M}",
                "Device": f"{self.device_info.device_name}",
                "Memory Mode": f"{strategy.mode.value} (max_parallel={strategy.max_parallel_alphas})",
                "Alpha": f"{remaining_alphas[0]:.1f} ~ {remaining_alphas[-1]:.1f} ({len(remaining_alphas)} points)",
                "Steps": f"{self.config.training.max_steps}",
                "Samples": f"{S}",
            }
        )

        # Initialize components
        torch.manual_seed(seed)
        np.random.seed(seed)

        teacher = self.teacher_cls()
        graph = self.graph_cls()
        algorithm = self.algorithm_cls(self.config, self.device)

        # Create teacher model (same seed ensures reproducibility)
        W_t, X_t, Y_t = teacher.create_with_Y(
            m.N1, m.N2, m.M, self.device, seed
        )

        # Get time estimate
        time_est = estimate_time(
            N1=m.N1, N2=m.N2, M=m.M,
            steps=self.config.training.max_steps,
            samples=S,
            num_alphas=len(remaining_alphas),
        )
        initial_estimate = time_est.get('estimated_seconds')

        # Run training with unified progress display
        start_time = time.time()

        # Calculate number of batches based on memory strategy
        # For parallel mode: ceil(num_alphas / max_parallel_alphas)
        # For sequential mode: num_alphas (each alpha is one batch)
        batch_size = strategy.max_parallel_alphas if strategy.mode == MemoryMode.PARALLEL else 1
        num_batches = (len(remaining_alphas) + batch_size - 1) // batch_size

        # Create unified progress display
        unified_progress = UnifiedProgress(
            num_alphas=len(remaining_alphas),
            steps_per_alpha=self.config.training.max_steps,
            initial_estimate=initial_estimate,
            num_batches=num_batches,
        )
        unified_progress.start()

        try:
            if strategy.mode == MemoryMode.PARALLEL and algorithm.supports_batch_training():
                # Parallel mode: process alphas in batches
                batch_results = self._run_parallel_training(
                    algorithm, graph, W_t, X_t, Y_t,
                    remaining_alphas, strategy, seed, S, m, unified_progress
                )
                all_results.update(batch_results)
            else:
                # Sequential mode (OPTIMIZED or EXTREME)
                completed_count = len(all_results)

                for alpha in remaining_alphas:
                    unified_progress.start_alpha(alpha)

                    # Generate mask on-demand
                    mask_seed = seed + int(alpha * 1000)
                    mask, _ = graph.generate_mask(
                        m.N1, m.N2, m.M, alpha, self.device, mask_seed
                    )

                    # Train with progress callback
                    if strategy.use_fp16_storage:
                        W_s, X_s = algorithm.train_single_alpha(
                            W_t, X_t, Y_t, mask, alpha, mask_seed + 10000,
                            use_fp16_storage=True,
                            progress_callback=unified_progress.update_step,
                        )
                    else:
                        W_s, X_s = algorithm.train_single_alpha(
                            W_t, X_t, Y_t, mask, alpha, mask_seed + 10000,
                            progress_callback=unified_progress.update_step,
                        )

                    # Evaluate (pass mask for Q_Y_unobserved/Q_Y_observed)
                    metrics = self._evaluate(W_s, X_s, W_t, X_t, Y_t, S, mask=mask)
                    
                    # === NEW: Replica Analysis & Raw Data ===
                    # Get metric function
                    metric_name = self.config.execution.matrix_metric or 'gram_overlap_normalized'
                    try:
                        metric_fn = get_metric_function(metric_name)
                    except ValueError:
                        metric_fn = gram_overlap_normalized
                        
                    # Compute matrices
                    matrix_W = build_interaction_matrix(W_s, W_t, metric_fn, use_left=True)
                    matrix_X = build_interaction_matrix(X_s, X_t, metric_fn, use_left=False)
                    
                    # Add to metrics (optional, or rely on raw_data)
                    metrics['interaction_matrix_W'] = matrix_W.tolist()
                    metrics['interaction_matrix_X'] = matrix_X.tolist()
                    
                    all_results[float(alpha)] = metrics
                    
                    # Save Raw Data
                    storage = ResultStorage(self.config) # Re-init inside loop usually fine, or move out
                    raw_data = {
                        'W_student': W_s, 'X_student': X_s, 'mask': mask,
                        'W_teacher': W_t, 'X_teacher': X_t, 'Y_teacher': Y_t,
                        'interaction_matrix_W': matrix_W, 'interaction_matrix_X': matrix_X
                    }
                    storage.save_raw_batch(float(alpha), raw_data)
                    
                    # Generate Heatmap
                    plots_dir = storage.get_plots_dir()
                    path_W = plot_replica_heatmap(
                        matrix_W, float(alpha), plots_dir, 
                        metric_name=f"Q_W ({metric_name})", filename_prefix="heatmap_W"
                    )
                    if matrix_X is not None:
                        path_X = plot_replica_heatmap(
                            matrix_X, float(alpha), plots_dir, 
                            metric_name=f"Q_X ({metric_name})", filename_prefix="heatmap_X"
                        )
                    # Collect paths (need a list defined outside)
                    if not hasattr(self, '_seq_heatmap_paths_W'):
                        self._seq_heatmap_paths_W = []
                        self._seq_heatmap_paths_X = []
                    self._seq_heatmap_paths_W.append(path_W)
                    if matrix_X is not None:
                        self._seq_heatmap_paths_X.append(path_X)
                    
                    completed_count += 1
                    
                    unified_progress.finish_alpha(metrics)

                    # Save checkpoint periodically
                    if checkpoint_mgr.should_save(completed_count):
                        checkpoint_mgr.save(list(all_results.keys()), all_results)

                    # Clear cache
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
        finally:
            unified_progress.stop()

        total_time = time.time() - start_time

        # Save results
        result_path = None
        if save_results:
            result_path = self._save_results(all_results, total_time)
            
            # Generate GIFs
            # Gather paths from either parallel or sequential storage
            paths_W = getattr(self, '_last_heatmap_paths_W', []) or getattr(self, '_seq_heatmap_paths_W', [])
            paths_X = getattr(self, '_last_heatmap_paths_X', []) or getattr(self, '_seq_heatmap_paths_X', [])
            
            if paths_W:
                # Sort by alpha value extracted from filename or trust the order
                # Filename format: heatmap_W_alpha_1.234567.png
                # Python sort should handle lexicographical order of numbers correctly if fixed width, 
                # but these are floats.
                # Only if alphas are Monotonic. They usually are.
                # Let's rely on list order as we append sequentially.
                
                storage = ResultStorage(self.config)
                plots_dir = storage.get_plots_dir()
                
                gif_path_W = create_gif(paths_W, plots_dir / "animation_W.gif", duration=0.5)
                gif_path_X = create_gif(paths_X, plots_dir / "animation_X.gif", duration=0.5)
                
                if gif_path_W:
                    print(f"Generated GIF: {gif_path_W}")

        # Cleanup checkpoints after successful completion
        checkpoint_mgr.cleanup()

        self.progress.print_completion(total_time, str(result_path) if result_path else None)

        return {
            'results': all_results,
            'config': self.config,
            'total_time': total_time,
            'result_path': result_path,
        }

    def _evaluate(
        self,
        W_s: torch.Tensor,
        X_s: torch.Tensor,
        W_t: torch.Tensor,
        X_t: torch.Tensor,
        Y_t: torch.Tensor,
        S: int,
        mask: torch.Tensor = None,
    ) -> Dict[str, float]:
        """Evaluate trained model.

        Args:
            W_s: Student W matrices (S, N1, M)
            X_s: Student X matrices (S, M, N2)
            W_t: Teacher W matrix
            X_t: Teacher X matrix
            Y_t: Teacher Y matrix
            S: Number of samples
            mask: Observation mask (required for Q_Y_unobserved/Q_Y_observed)

        Returns:
            Aggregated metrics dictionary
        """
        trial_results = []

        # Get metrics to compute from execution config
        metrics_to_compute = self.config.execution.metrics_to_compute

        for s in range(S):
            metrics = compute_all_metrics(
                W_s[s], X_s[s], W_t, X_t, Y_t,
                mask=mask,
                metrics_to_compute=metrics_to_compute,
            )
            trial_results.append(metrics)

        aggregated = aggregate_trial_metrics(trial_results)

        # Add replica overlap if S > 1
        if S > 1:
            replica = compute_replica_overlap(W_s, X_s)
            aggregated.update(replica)
        else:
            aggregated.update({
                'Q_W_replica_mean': 0.0,
                'Q_W_replica_std': 0.0,
                'Q_X_replica_mean': 0.0,
                'Q_X_replica_std': 0.0,
            })

        return aggregated

    def _run_parallel_training(
        self,
        algorithm,
        graph,
        W_t: torch.Tensor,
        X_t: torch.Tensor,
        Y_t: torch.Tensor,
        alpha_values: list,
        strategy,
        seed: int,
        S: int,
        m,
        unified_progress: UnifiedProgress,
    ) -> Dict[str, Any]:
        """
        Run training in parallel batches for multiple alphas.
        
        IMPORTANT: Pass ALL alphas to train_batch_alphas at once.
        The algorithm handles its own internal smart batching (前密后疏).
        Do NOT pre-split alphas here - that causes double batching.

        Args:
            algorithm: Algorithm instance
            graph: Graph instance for mask generation
            W_t, X_t, Y_t: Teacher tensors
            alpha_values: List of alpha values to process
            strategy: MemoryStrategy with batch size info
            seed: Random seed
            S: Samples per alpha
            m: Matrix config
            unified_progress: UnifiedProgress instance

        Returns:
            Dictionary of {alpha: metrics}
        """
        all_results = {}
        
        # Define batch callback to update progress display with current batch info
        def on_batch_start(batch_idx, num_batches, batch_alphas):
            unified_progress.start_batch(batch_idx, batch_alphas, num_batches)
        
        # Train ALL alphas at once - algorithm does smart batching internally
        W_s_all, X_s_all = algorithm.train_batch_alphas(
            W_t, X_t, Y_t, 
            None,  # masks not used - algorithm generates via SuperGraph
            alpha_values, 
            seed,
            step_callback=unified_progress.update_step,
            sample_callback=on_batch_start,  # Receives batch alpha range updates
        )
        # W_s_all: (num_alphas, S, N1, M), X_s_all: (num_alphas, S, M, N2)
        
        # Arrays to store heatmap paths for GIF generation
        heatmap_paths_W = []
        heatmap_paths_X = []
        
        # Get metric function (default: gram_overlap_normalized)
        metric_name = self.config.execution.matrix_metric or 'gram_overlap_normalized'
        try:
            metric_fn = get_metric_function(metric_name)
        except ValueError:
            print(f"Warning: Unknown metric {metric_name}, using gram_overlap_normalized")
            metric_fn = gram_overlap_normalized

        # Initialize storage access
        storage = ResultStorage(self.config)

        # Evaluate each alpha
        for i, alpha in enumerate(alpha_values):
            W_s = W_s_all[i]
            X_s = X_s_all[i]
            
            # Generate mask for evaluation (needed for Q_Y_unobserved)
            mask_seed = seed + int(alpha * 1000)
            mask, _ = graph.generate_mask(
                m.N1, m.N2, m.M, alpha, self.device, mask_seed
            )
            
            # 1. Compute basic metrics
            metrics = self._evaluate(W_s, X_s, W_t, X_t, Y_t, S, mask=mask)
            
            # 2. Compute Full Interaction Matrices (Teacher + Replicas)
            # W Matrix
            matrix_W = build_interaction_matrix(
                W_s, W_t, metric_fn, use_left=True
            )
            metrics['interaction_matrix_W'] = matrix_W.tolist() # Save to specific field if needed, or separate
            
            # X Matrix
            matrix_X = build_interaction_matrix(
                X_s, X_t, metric_fn, use_left=False
            )
            metrics['interaction_matrix_X'] = matrix_X.tolist()

            all_results[float(alpha)] = metrics
            
            # 3. Save Raw Data (Incremental)
            raw_data = {
                'W_student': W_s,
                'X_student': X_s,
                'mask': mask,
                'W_teacher': W_t,
                'X_teacher': X_t,
                'Y_teacher': Y_t,
                'interaction_matrix_W': matrix_W,
                'interaction_matrix_X': matrix_X,
            }
            storage.save_raw_batch(float(alpha), raw_data)
            
            # 4. Generate Heatmaps
            plots_dir = storage.get_plots_dir()
            path_W = plot_replica_heatmap(
                matrix_W, float(alpha), plots_dir, 
                metric_name=f"Q_W ({metric_name})", 
                filename_prefix="heatmap_W"
            )
            heatmap_paths_W.append(path_W)
            
            if m.N1 != m.N2:
                path_X = plot_replica_heatmap(
                    matrix_X, float(alpha), plots_dir, 
                    metric_name=f"Q_X ({metric_name})", 
                    filename_prefix="heatmap_X"
                )
                heatmap_paths_X.append(path_X)
        
        unified_progress.finish_batch(metrics)
        
        if heatmap_paths_W:
            create_gif(heatmap_paths_W, storage.get_plots_dir() / "animation_W.gif", duration=0.2)
        
        if heatmap_paths_X:
            create_gif(heatmap_paths_X, storage.get_plots_dir() / "animation_X.gif", duration=0.2)
            
        # Generate Comparison Plot
        plot_overlap_evolution(all_results, storage.get_plots_dir())

        return all_results

    def _save_results(self, results: Dict, total_time: float) -> Path:
        """Save results to disk."""
        storage = ResultStorage(self.config)

        # Save data
        storage.save(
            results,
            metadata={
                'total_time': total_time,
                'device': self.device_info.device_name,
            }
        )

        # Create plots based on execution config
        plotter = ResultPlotter(self.config, storage.get_plots_dir())
        exec_cfg = self.config.execution

        # Generate summary plot only if requested
        if exec_cfg.include_summary_plot:
            plotter.plot_summary(results)

        # Generate Q_Y plot only if requested
        if exec_cfg.include_qy_plot:
            # Check if Q_Y_unobserved should also be plotted
            metrics_to_plot = ['Q_Y']
            if 'Q_Y_unobserved' in exec_cfg.metrics_to_compute:
                metrics_to_plot.append('Q_Y_unobserved')
            if 'Q_Y_observed' in exec_cfg.metrics_to_compute:
                metrics_to_plot.append('Q_Y_observed')

            if len(metrics_to_plot) == 1:
                # Standard Q_Y only plot
                plotter.plot_qy_only(results)
            else:
                # Multi-metric Q_Y plot (includes Q_Y_unobserved, etc.)
                plotter.plot_qy_comparison(results, metrics=metrics_to_plot)

        # Generate custom plots from LLM config
        for plot_config in exec_cfg.plots:
            self._generate_custom_plot(plotter, results, plot_config, storage.get_plots_dir())

        return storage.output_dir

    def _generate_custom_plot(
        self,
        plotter: ResultPlotter,
        results: Dict,
        plot_config: Dict,
        output_dir: Path,
    ):
        """Generate a custom plot based on LLM config.

        Args:
            plotter: ResultPlotter instance
            results: Results dictionary
            plot_config: Plot configuration from LLM, e.g.:
                {
                    "type": "comparison",
                    "metrics": ["Q_Y", "Q_Y_unobserved"],
                    "filename": "qy_comparison.png"
                }
            output_dir: Output directory
        """
        plot_type = plot_config.get('type', 'comparison')
        metrics = plot_config.get('metrics', ['Q_Y'])
        filename = plot_config.get('filename', 'custom_plot.png')

        if plot_type == 'comparison':
            plotter.plot_qy_comparison(results, metrics=metrics, filename=filename)
        # Can add more plot types here in the future


def run_experiment(config: Config, save: bool = True, resume: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run an experiment.

    Args:
        config: Experiment configuration
        save: Whether to save results
        resume: Whether to resume from checkpoint

    Returns:
        Results dictionary
    """
    runner = ExperimentRunner(config)
    return runner.run(save_results=save, resume=resume)
