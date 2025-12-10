"""
Result comparison plotting module.

Compares multiple experiment results on the same plot.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .plotting import COLORS, STYLE


class ResultComparison:
    """
    Compare multiple experiment results.
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("smf/results/comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_qy_comparison(
        self,
        results_list: List[Dict[float, Dict]],
        labels: List[str],
        title: str = "Q_Y Comparison",
        filename: str = "qy_comparison.png",
        show_std: bool = True,
        metric: str = "Q_Y",
    ) -> Path:
        """
        Plot metric curves from multiple experiments on the same figure.

        Args:
            results_list: List of results dicts (alpha -> metrics)
            labels: Labels for each result set
            title: Plot title
            filename: Output filename
            show_std: Whether to show error bars
            metric: Metric to plot (default: Q_Y, can be Q_W, Q_X, Q_W_prime, etc.)

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use colormap for multiple curves
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

        # Determine metric keys
        metric_mean = f"{metric}_mean"
        metric_std = f"{metric}_std"

        for i, (results, label) in enumerate(zip(results_list, labels)):
            # Sort keys by float value but keep original keys for dict access
            alpha_keys = sorted(results.keys(), key=float)
            alphas = [float(a) for a in alpha_keys]
            values_mean = [results[a].get(metric_mean, results[a].get(metric, 0)) for a in alpha_keys]

            if show_std:
                values_std = [results[a].get(metric_std, 0) for a in alpha_keys]
                ax.errorbar(
                    alphas, values_mean, yerr=values_std,
                    color=colors[i], label=label,
                    linewidth=STYLE['linewidth'],
                    marker=STYLE['marker'],
                    markersize=STYLE['markersize'],
                    capsize=STYLE['capsize'],
                )
            else:
                ax.plot(
                    alphas, values_mean,
                    color=colors[i], label=label,
                    linewidth=STYLE['linewidth'],
                    marker=STYLE['marker'],
                    markersize=STYLE['markersize'],
                )

        # Format y-axis label
        metric_label = metric.replace('_prime', "'").replace('_', ' ')
        ylabel = f"${metric_label}$"
        ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylabel(ylabel, fontsize=STYLE['fontsize']['label'])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=STYLE['fontsize']['legend'], loc='best')
        ax.set_title(title, fontsize=STYLE['fontsize']['title'])

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_multi_metric_comparison(
        self,
        results_list: List[Dict[float, Dict]],
        labels: List[str],
        metrics: List[str] = None,
        filename: str = "multi_metric_comparison.png",
    ) -> Path:
        """
        Plot multiple metrics from multiple experiments.

        Args:
            results_list: List of results dicts
            labels: Labels for each result set
            metrics: Metrics to compare (default: Q_Y, Q_W', Q_X')
            filename: Output filename

        Returns:
            Path to saved plot
        """
        if metrics is None:
            metrics = ['Q_Y_mean', 'Q_W_prime_mean', 'Q_X_prime_mean']

        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))

        if num_metrics == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

        for ax, metric in zip(axes, metrics):
            for i, (results, label) in enumerate(zip(results_list, labels)):
                # Sort keys by float value but keep original keys for dict access
                alpha_keys = sorted(results.keys(), key=float)
                alphas = [float(a) for a in alpha_keys]
                values = [results[a].get(metric, 0) for a in alpha_keys]

                ax.plot(
                    alphas, values,
                    color=colors[i], label=label,
                    linewidth=STYLE['linewidth'],
                    marker=STYLE['marker'],
                    markersize=STYLE['markersize'],
                )

            metric_name = metric.replace('_mean', '').replace('_prime', "'")
            ax.set_xlabel(r'$\tilde{\alpha}$')
            ax.set_ylabel(metric_name)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_title(metric_name)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def compute_difference_summary(
        self,
        results1: Dict[float, Dict],
        results2: Dict[float, Dict],
        label1: str = "A",
        label2: str = "B",
    ) -> Dict[str, Any]:
        """
        Compute difference summary between two results.

        Returns:
            Dictionary with difference statistics
        """
        # Find common alpha values (compare as floats, but keep string keys)
        keys1 = {float(a): a for a in results1.keys()}  # float -> original key
        keys2 = {float(a): a for a in results2.keys()}
        common_floats = sorted(set(keys1.keys()) & set(keys2.keys()))

        if not common_floats:
            return {"error": "No common alpha values"}

        # Compute differences using original string keys
        qy_diffs = []
        for alpha_float in common_floats:
            qy1 = results1[keys1[alpha_float]]['Q_Y_mean']
            qy2 = results2[keys2[alpha_float]]['Q_Y_mean']
            qy_diffs.append(qy2 - qy1)

        return {
            "common_alphas": len(common_floats),
            "alpha_range": (min(common_floats), max(common_floats)),
            "Q_Y_diff_mean": float(np.mean(qy_diffs)),
            "Q_Y_diff_std": float(np.std(qy_diffs)),
            "Q_Y_diff_max": float(np.max(np.abs(qy_diffs))),
            "label1": label1,
            "label2": label2,
        }


def compare_results(
    results_list: List[Dict],
    labels: List[str],
    output_dir: Path = None,
) -> Path:
    """
    Convenience function to compare multiple results.

    Args:
        results_list: List of results dictionaries
        labels: Labels for each result
        output_dir: Output directory

    Returns:
        Path to comparison plot
    """
    comparison = ResultComparison(output_dir)
    return comparison.plot_qy_comparison(results_list, labels)
