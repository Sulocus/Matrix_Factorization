"""
Initialization scale sweep experiment.

Runs experiments with different spin initialization scales (k/√M) and compares results.
Tests how different normalization scales affect the phase transition.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, AlgorithmConfig
from ..core.device import setup_device
from ..core.progress import get_progress_manager
from ..modules.teachers.scaled_variance import ScaledVarianceTeacher
from ..modules.registry import get_graph, get_algorithm
from ..modules.metrics.overlap import compute_all_metrics, aggregate_trial_metrics
from ..modules.outputs.plotting import COLORS, STYLE


class InitScaleExperiment:
    """
    Run experiments with multiple initialization scale factors.

    Tests different spin normalization scales (k/√M) and compares Q_Y curves.
    """

    def __init__(
        self,
        config: Config,
        variance_scales: List[float],
        output_dir: Path = None,
    ):
        """
        Initialize init scale experiment.

        Args:
            config: Base configuration (matrix, alpha, training params)
            variance_scales: List of scale factors (k in k/√M) to test
                            e.g., [0.5, 1.0, 1.5, 2.0]
            output_dir: Output directory for results
        """
        self.config = config
        self.variance_scales = variance_scales  # keep internal name for compatibility
        self.device, self.device_info = setup_device()
        self.progress = get_progress_manager()

        # Setup output directory
        # Format: init_scale_{N1}x{N2}_M{M}_{MMDD_HHMM}
        if output_dir is None:
            time_suffix = datetime.now().strftime("%m%d_%H%M")
            m = config.matrix
            output_dir = Path(f"smf/results/init_scale_{m.N1}x{m.N2}_M{m.M}_{time_suffix}")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[float, Dict[float, Dict]]:
        """
        Run experiments for all variance scales.

        Returns:
            Dict mapping variance_scale -> {alpha -> metrics}
        """
        m = self.config.matrix
        alpha_values = self.config.alpha.get_values()
        S = self.config.training.samples_per_alpha
        seed = self.config.training.seed

        self.progress.print_header(
            "Init Scale Experiment",
            {
                "Matrix": f"{m.N1}×{m.N2}, M={m.M}",
                "Scale factors (k)": str(self.variance_scales),
                "Alpha": f"{alpha_values[0]:.2f} ~ {alpha_values[-1]:.2f}",
                "Steps": self.config.training.max_steps,
            }
        )

        all_results = {}
        total_start = time.time()

        # Get algorithm and graph
        algorithm_cls = get_algorithm(self.config.algorithm_key).cls
        graph_cls = get_graph(self.config.graph_key).cls
        graph = graph_cls()

        for var_scale in self.variance_scales:
            self.progress.print(f"\nRunning variance scale = {var_scale}")

            # Create teacher with this variance
            teacher = ScaledVarianceTeacher(variance_scale=var_scale)
            W_t, X_t, Y_t = teacher.create_with_Y(
                m.N1, m.N2, m.M, self.device, seed
            )

            # Create algorithm instance
            algorithm = algorithm_cls(self.config, self.device)

            # Run for all alpha values
            results = {}

            with self.progress.training_progress(len(alpha_values), self.config.training.max_steps) as updater:
                for alpha in alpha_values:
                    updater.start_alpha(alpha)

                    # Generate mask
                    mask_seed = seed + int(alpha * 1000)
                    mask, _ = graph.generate_mask(
                        m.N1, m.N2, m.M, alpha, self.device, mask_seed
                    )

                    # Train
                    W_s, X_s = algorithm.train_single_alpha(
                        W_t, X_t, Y_t, mask, alpha, mask_seed + 10000
                    )

                    # Evaluate
                    metrics = self._evaluate(W_s, X_s, W_t, X_t, Y_t, S)
                    results[float(alpha)] = metrics

                    updater.finish_alpha(metrics)

                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

            all_results[var_scale] = results

        total_time = time.time() - total_start

        # Save results and plot
        self._save_results(all_results, total_time)
        plot_path = self._plot_comparison(all_results)

        self.progress.print_completion(total_time, str(self.output_dir))
        self.progress.print(f"Comparison plot: {plot_path}")

        return all_results

    def _evaluate(self, W_s, X_s, W_t, X_t, Y_t, S) -> Dict[str, float]:
        """Evaluate trained model."""
        trial_results = []
        for s in range(S):
            metrics = compute_all_metrics(W_s[s], X_s[s], W_t, X_t, Y_t)
            trial_results.append(metrics)
        return aggregate_trial_metrics(trial_results)

    def _save_results(self, results: Dict, total_time: float):
        """Save results to JSON."""
        data = {
            "config": self.config.to_dict(),
            "variance_scales": self.variance_scales,
            "results": {str(k): v for k, v in results.items()},
            "total_time": total_time,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(data, f, indent=2)

    def _plot_comparison(self, results: Dict[float, Dict]) -> Path:
        """
        Plot Q_Y comparison for different variance scales.

        Args:
            results: {variance_scale: {alpha: metrics}}

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(self.variance_scales)))

        for i, var_scale in enumerate(self.variance_scales):
            if var_scale not in results:
                continue

            data = results[var_scale]
            alphas = sorted([float(a) for a in data.keys()])
            qy_mean = [data[a]['Q_Y_mean'] for a in alphas]
            qy_std = [data[a].get('Q_Y_std', 0) for a in alphas]

            label = f"k = {var_scale}" if var_scale != 1.0 else "k = 1 (standard)"

            ax.errorbar(
                alphas, qy_mean, yerr=qy_std,
                color=colors[i], label=label,
                linewidth=STYLE['linewidth'],
                marker=STYLE['marker'],
                markersize=STYLE['markersize'],
                capsize=STYLE['capsize'],
            )

        ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylabel('$Q_Y$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(
            fontsize=STYLE['fontsize']['legend'],
            loc='lower right',
            title='Init scale (k/√M)'
        )

        m = self.config.matrix
        ax.set_title(
            f"Init Scale Comparison: {m.N1}×{m.N2}, M={m.M}",
            fontsize=STYLE['fontsize']['title']
        )

        plt.tight_layout()
        plot_path = self.output_dir / "init_scale_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return plot_path


def run_init_scale(
    N1: int = 200,
    N2: int = 200,
    M: int = 50,
    scale_factors: List[float] = None,
    alpha_start: float = 0.0,
    alpha_stop: float = 4.0,
    alpha_step: float = 0.1,
    max_steps: int = 5000,
    samples: int = 1,
) -> Dict:
    """
    Convenience function to run init scale experiment.

    Args:
        N1, N2, M: Matrix dimensions
        scale_factors: List of scale factors k (in k/√M), default: [0.5, 1.0, 1.5, 2.0]
        alpha_start, alpha_stop, alpha_step: Alpha range
        max_steps: Training steps
        samples: Samples per alpha

    Returns:
        Results dictionary
    """
    if scale_factors is None:
        scale_factors = [0.5, 1.0, 1.5, 2.0]

    config = Config(
        matrix=MatrixConfig(N1=N1, N2=N2, M=M),
        alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
        training=TrainingConfig(max_steps=max_steps, samples_per_alpha=samples),
        algorithm_key="bigamp",
        graph_key="random",
    )

    experiment = InitScaleExperiment(config, scale_factors)
    return experiment.run()


# Backwards compatibility aliases
run_variance_sweep = run_init_scale
VarianceSweepExperiment = InitScaleExperiment
