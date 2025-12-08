"""
Size scaling experiment - Finite-size effect study.

Run experiments across multiple matrix sizes to observe finite-size effects.
Studies how phase transition behavior changes with system size N in the
NON-thermodynamic limit regime (where N is finite, not N → ∞).

By comparing Q_Y curves at different N values (with fixed M), we can observe:
- How the phase transition sharpens as N increases
- Finite-size corrections to the critical point α_c
- Scaling behavior in the non-thermodynamic regime

Example sizes: 1000x1000, 2000x2000, 3000x3000, 4000x4000, 5000x5000 (M=100)
Alpha range: 0 to 0.5, step 0.01
Steps: 10000
"""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig
from ..core.device import setup_device
from ..core.progress import get_progress_manager
from ..modules.registry import get_algorithm, get_graph
from ..modules.teachers.standard import StandardTeacher
from ..modules.metrics.overlap import compute_all_metrics, aggregate_trial_metrics
from ..modules.outputs.plotting import COLORS, STYLE


class SizeScalingExperiment:
    """
    Finite-size effect study across multiple matrix sizes.

    Studies how phase transition behavior changes with system size N
    in the NON-thermodynamic limit regime (N is finite).
    """

    def __init__(
        self,
        matrix_configs: List[tuple],  # [(N1, N2, M), ...]
        alpha_start: float = 0.0,
        alpha_stop: float = 0.5,
        alpha_step: float = 0.01,
        max_steps: int = 10000,
        samples: int = 1,
        output_dir: Path = None,
    ):
        """
        Initialize large matrix sweep experiment.

        Args:
            matrix_configs: List of (N1, N2, M) tuples
            alpha_start, alpha_stop, alpha_step: Alpha range
            max_steps: Training steps per configuration
            samples: Samples per alpha
            output_dir: Output directory
        """
        self.matrix_configs = matrix_configs
        self.alpha_start = alpha_start
        self.alpha_stop = alpha_stop
        self.alpha_step = alpha_step
        self.max_steps = max_steps
        self.samples = samples

        self.device, self.device_info = setup_device()
        self.progress = get_progress_manager()

        # Setup output directory
        # Format: size_scaling_{sizes_summary}_{MMDD_HHMM}
        if output_dir is None:
            time_suffix = datetime.now().strftime("%m%d_%H%M")
            # Summarize sizes: e.g., "1k-5k_M100" for 1000 to 5000
            if matrix_configs:
                min_n = min(c[0] for c in matrix_configs)
                max_n = max(c[0] for c in matrix_configs)
                M = matrix_configs[0][2]
                sizes_str = f"{min_n//1000}k-{max_n//1000}k_M{M}"
            else:
                sizes_str = "unknown"
            output_dir = Path(f"smf/results/size_scaling_{sizes_str}_{time_suffix}")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Dict]:
        """
        Run experiments for all matrix configurations.

        Returns:
            Dict mapping config_str -> {alpha -> metrics}
        """
        self.progress.print_header(
            "Large Matrix Sweep",
            {
                "Configurations": len(self.matrix_configs),
                "Alpha range": f"{self.alpha_start} ~ {self.alpha_stop}",
                "Alpha step": self.alpha_step,
                "Steps": self.max_steps,
                "Device": self.device_info.device_name,
            }
        )

        all_results = {}
        total_start = time.time()

        # Get algorithm and graph
        algorithm_cls = get_algorithm("bigamp").cls
        graph_cls = get_graph("random").cls
        graph = graph_cls()
        teacher = StandardTeacher()

        for idx, (N1, N2, M) in enumerate(self.matrix_configs):
            config_str = f"{N1}x{N2}_M{M}"
            self.progress.print(f"\n[{idx+1}/{len(self.matrix_configs)}] Running {config_str}")

            start_time = time.time()

            # Create config for this matrix size
            config = Config(
                matrix=MatrixConfig(N1=N1, N2=N2, M=M),
                alpha=AlphaConfig(
                    start=self.alpha_start,
                    stop=self.alpha_stop,
                    step=self.alpha_step
                ),
                training=TrainingConfig(
                    max_steps=self.max_steps,
                    samples_per_alpha=self.samples
                ),
                algorithm_key="bigamp",
                graph_key="random",
            )

            alpha_values = config.alpha.get_values()

            # Create teacher matrices
            W_t, X_t, Y_t = teacher.create_with_Y(N1, N2, M, self.device, seed=42)

            # Create algorithm instance
            algorithm = algorithm_cls(config, self.device)

            # Run for all alpha values
            results = {}

            with self.progress.training_progress(len(alpha_values), self.max_steps) as updater:
                for alpha in alpha_values:
                    updater.start_alpha(alpha)

                    # Generate mask
                    mask_seed = 42 + int(alpha * 1000)
                    mask, _ = graph.generate_mask(
                        N1, N2, M, alpha, self.device, mask_seed
                    )

                    # Train
                    W_s, X_s = algorithm.train_single_alpha(
                        W_t, X_t, Y_t, mask, alpha, mask_seed + 10000
                    )

                    # Evaluate
                    metrics = self._evaluate(W_s, X_s, W_t, X_t, Y_t, self.samples)
                    results[float(alpha)] = metrics

                    updater.finish_alpha(metrics)

                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

            elapsed = time.time() - start_time
            all_results[config_str] = {
                'results': results,
                'time': elapsed,
                'N1': N1, 'N2': N2, 'M': M
            }

            self.progress.print(f"  Completed in {elapsed:.1f}s")

            # Save intermediate results
            self._save_results(all_results, time.time() - total_start)

        total_time = time.time() - total_start

        # Final save and plot
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
            "matrix_configs": self.matrix_configs,
            "alpha_range": {
                "start": self.alpha_start,
                "stop": self.alpha_stop,
                "step": self.alpha_step
            },
            "max_steps": self.max_steps,
            "samples": self.samples,
            "results": results,
            "total_time": total_time,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(data, f, indent=2)

    def _plot_comparison(self, results: Dict) -> Path:
        """
        Plot Q_Y comparison for different matrix sizes.

        Args:
            results: {config_str: {'results': {alpha: metrics}, ...}}

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))

        for i, (config_str, data) in enumerate(results.items()):
            alphas = sorted([float(a) for a in data['results'].keys()])
            qy_mean = [data['results'][a]['Q_Y_mean'] for a in alphas]
            qy_std = [data['results'][a].get('Q_Y_std', 0) for a in alphas]

            N1, M = data['N1'], data['M']
            label = f"N={N1}, M={M}"

            ax.errorbar(
                alphas, qy_mean, yerr=qy_std,
                color=colors[i], label=label,
                linewidth=STYLE['linewidth'],
                marker=STYLE['marker'],
                markersize=STYLE['markersize'] - 1,
                capsize=STYLE['capsize'],
            )

        ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylabel('$Q_Y$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(
            fontsize=STYLE['fontsize']['legend'],
            loc='lower right',
            title='Matrix Size'
        )

        ax.set_title(
            f"Large Matrix Scaling: Alpha 0-0.5, {self.max_steps} steps",
            fontsize=STYLE['fontsize']['title']
        )

        plt.tight_layout()
        plot_path = self.output_dir / "size_scaling_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return plot_path


def run_size_scaling(
    sizes: List[tuple] = None,
    alpha_start: float = 0.0,
    alpha_stop: float = 0.5,
    alpha_step: float = 0.01,
    max_steps: int = 10000,
) -> Dict:
    """
    Convenience function to run size scaling experiment.

    Args:
        sizes: List of (N1, N2, M) tuples
               Default: [(1000,1000,100), (2000,2000,100), (3000,3000,100),
                        (4000,4000,100), (5000,5000,100)]
        alpha_start, alpha_stop, alpha_step: Alpha range
        max_steps: Training steps

    Returns:
        Results dictionary
    """
    if sizes is None:
        sizes = [
            (1000, 1000, 100),
            (2000, 2000, 100),
            (3000, 3000, 100),
            (4000, 4000, 100),
            (5000, 5000, 100),
        ]

    experiment = SizeScalingExperiment(
        matrix_configs=sizes,
        alpha_start=alpha_start,
        alpha_stop=alpha_stop,
        alpha_step=alpha_step,
        max_steps=max_steps,
    )
    return experiment.run()


# Backwards compatibility alias
run_large_matrix_sweep = run_size_scaling
LargeMatrixSweepExperiment = SizeScalingExperiment


if __name__ == "__main__":
    # Run the default size scaling experiment
    run_size_scaling()
