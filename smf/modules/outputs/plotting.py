"""
Result plotting module with unified styles.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ..registry import register_output
from .base import OutputBase
from ...core.config import Config


# Unified color scheme
COLORS = {
    'Q_Y': '#d62728',           # Red
    'Q_Y_unobserved': '#17becf', # Cyan
    'Q_Y_observed': '#bcbd22',   # Yellow-green
    'Q_W_prime': '#9467bd',     # Purple
    'Q_X_prime': '#8c564b',     # Brown
    'Q_W': '#1f77b4',           # Blue
    'Q_X': '#ff7f0e',           # Orange
    'Gen_Error': '#2ca02c',     # Green
}

# Unified style settings
STYLE = {
    'linewidth': 1.5,
    'markersize': 4,
    'marker': 'o',
    'capsize': 2,
    'fontsize': {
        'title': 12,
        'label': 10,
        'tick': 9,
        'legend': 9,
    }
}


@register_output(
    key="plotting",
    name="Result Plotting",
    description="Unified style Q_Y, Q_W', Q_X' curve plots",
)
class ResultPlotter(OutputBase):
    """
    Unified result plotting with consistent styles.
    """

    def __init__(self, config: Config, output_dir: Path):
        super().__init__(config, output_dir)
        plt.style.use('default')
        plt.rcParams['font.size'] = STYLE['fontsize']['tick']

    def save(self, results: Dict[str, Any], **kwargs) -> Path:
        """Save standard result plot."""
        return self.plot_summary(results, **kwargs)

    def plot_summary(
        self,
        results: Dict[float, Dict[str, float]],
        title: str = None,
        filename: str = "summary.png",
        show_params: bool = True,
    ) -> Path:
        """
        Create summary plot with Q_Y, Q_W', Q_X' vs alpha.

        Args:
            results: Dict mapping alpha -> metrics dict
            title: Plot title (optional)
            filename: Output filename
            show_params: Whether to show parameter table

        Returns:
            Path to saved plot
        """
        # Extract data
        alphas = sorted([float(a) for a in results.keys()])
        qy_mean = [results[a]['Q_Y_mean'] for a in alphas]
        qy_std = [results[a].get('Q_Y_std', 0) for a in alphas]
        qw_prime_mean = [results[a]['Q_W_prime_mean'] for a in alphas]
        qw_prime_std = [results[a].get('Q_W_prime_std', 0) for a in alphas]
        qx_prime_mean = [results[a]['Q_X_prime_mean'] for a in alphas]
        qx_prime_std = [results[a].get('Q_X_prime_std', 0) for a in alphas]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot curves with unified style
        ax.errorbar(alphas, qy_mean, yerr=qy_std,
                    color=COLORS['Q_Y'], label='$Q_Y$',
                    linewidth=STYLE['linewidth'], marker=STYLE['marker'],
                    markersize=STYLE['markersize'], capsize=STYLE['capsize'])

        ax.errorbar(alphas, qw_prime_mean, yerr=qw_prime_std,
                    color=COLORS['Q_W_prime'], label="$Q'_W$",
                    linewidth=STYLE['linewidth'], marker=STYLE['marker'],
                    markersize=STYLE['markersize'], capsize=STYLE['capsize'])

        ax.errorbar(alphas, qx_prime_mean, yerr=qx_prime_std,
                    color=COLORS['Q_X_prime'], label="$Q'_X$",
                    linewidth=STYLE['linewidth'], marker=STYLE['marker'],
                    markersize=STYLE['markersize'], capsize=STYLE['capsize'])

        # Formatting
        ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylabel('Overlap', fontsize=STYLE['fontsize']['label'])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(min(alphas) - 0.1, max(alphas) + 0.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=STYLE['fontsize']['legend'], loc='lower right')

        if title:
            ax.set_title(title, fontsize=STYLE['fontsize']['title'])
        else:
            m = self.config.matrix
            ax.set_title(
                f"BiG-AMP Results: {m.N1}×{m.N2}, M={m.M}",
                fontsize=STYLE['fontsize']['title']
            )

        # Add parameter table if requested
        if show_params:
            self._add_param_table(ax)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _add_param_table(self, ax):
        """Add parameter table to plot."""
        m = self.config.matrix
        t = self.config.training
        alg = self.config.algorithm

        params = [
            f"N₁={m.N1}, N₂={m.N2}, M={m.M}",
            f"Steps={t.max_steps}, S={t.samples_per_alpha}",
            f"Graph: {self.config.graph_key}",
            f"Damping={alg.damping}",
        ]

        text = '\n'.join(params)

        # Position in top-left
        props = dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=STYLE['fontsize']['tick'] - 1,
                verticalalignment='top', bbox=props)

    def plot_qy_only(
        self,
        results: Dict[float, Dict[str, float]],
        filename: str = "qy_vs_alpha.png",
    ) -> Path:
        """Create Q_Y only plot."""
        alphas = sorted([float(a) for a in results.keys()])
        qy_mean = [results[a]['Q_Y_mean'] for a in alphas]
        qy_std = [results[a].get('Q_Y_std', 0) for a in alphas]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.errorbar(alphas, qy_mean, yerr=qy_std,
                    color=COLORS['Q_Y'], label='$Q_Y$',
                    linewidth=STYLE['linewidth'] * 1.2,
                    marker=STYLE['marker'], markersize=STYLE['markersize'] * 1.2,
                    capsize=STYLE['capsize'])

        ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylabel('$Q_Y$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        m = self.config.matrix
        ax.set_title(f"$Q_Y$ vs $\\alpha$: {m.N1}×{m.N2}, M={m.M}",
                     fontsize=STYLE['fontsize']['title'])

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_qy_comparison(
        self,
        results: Dict[float, Dict[str, float]],
        metrics: List[str] = None,
        filename: str = "qy_comparison.png",
    ) -> Path:
        """
        Create comparison plot for multiple Q_Y-related metrics.

        This is the dynamic version that plots whatever metrics the LLM requested.

        Args:
            results: Dict mapping alpha -> metrics dict
            metrics: List of metrics to plot, e.g., ['Q_Y', 'Q_Y_unobserved']
            filename: Output filename

        Returns:
            Path to saved plot
        """
        if metrics is None:
            metrics = ['Q_Y']

        alphas = sorted([float(a) for a in results.keys()])

        # Metric display names (LaTeX)
        METRIC_LABELS = {
            'Q_Y': '$Q_Y$',
            'Q_Y_unobserved': '$Q_Y$ (unobserved)',
            'Q_Y_observed': '$Q_Y$ (observed)',
            'Q_W': '$Q_W$',
            'Q_X': '$Q_X$',
            'Q_W_prime': "$Q'_W$",
            'Q_X_prime': "$Q'_X$",
            'Gen_Error': 'Gen Error',
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        for metric in metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'

            # Skip if metric not in results
            if mean_key not in results[alphas[0]]:
                continue

            values = [results[a].get(mean_key, 0) for a in alphas]
            stds = [results[a].get(std_key, 0) for a in alphas]

            color = COLORS.get(metric, '#333333')
            label = METRIC_LABELS.get(metric, metric)

            ax.errorbar(
                alphas, values, yerr=stds,
                color=color, label=label,
                linewidth=STYLE['linewidth'],
                marker=STYLE['marker'],
                markersize=STYLE['markersize'],
                capsize=STYLE['capsize'],
            )

        ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylabel('Overlap', fontsize=STYLE['fontsize']['label'])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=STYLE['fontsize']['legend'], loc='lower right')

        m = self.config.matrix
        ax.set_title(
            f"Q Metrics vs α: {m.N1}×{m.N2}, M={m.M}",
            fontsize=STYLE['fontsize']['title']
        )

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path


def plot_comparison(
    results_list: List[Dict[str, Any]],
    labels: List[str],
    output_path: Path,
    metric: str = 'Q_Y_mean',
    # New flexible options
    xlim: tuple = None,
    ylim: tuple = None,
    legend_loc: str = 'best',
    format: str = 'png',
    dpi: int = 150,
    error_style: str = 'bar',  # 'bar' or 'band'
    colormap: str = None,
) -> Path:
    """
    Plot comparison of multiple experiment results with flexible options.

    Args:
        results_list: List of results dictionaries
        labels: Labels for each result set
        output_path: Where to save the plot
        metric: Which metric to compare
        xlim: (xmin, xmax) or None for auto
        ylim: (ymin, ymax) or None for auto
        legend_loc: Legend position ('best', 'upper right', 'outside right', etc.)
        format: Output format ('png', 'pdf', 'svg')
        dpi: Resolution (150 for screen, 300 for publication)
        error_style: 'bar' for error bars, 'band' for fill_between
        colormap: Color scheme name (None for tab10)

    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Choose colormap
    if colormap:
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(results_list)))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    for i, (results, label) in enumerate(zip(results_list, labels)):
        alphas = sorted([float(a) for a in results.keys()])
        values = [results[a][metric] for a in alphas]
        stds = [results[a].get(metric.replace('_mean', '_std'), 0) for a in alphas]

        if error_style == 'band' and any(s > 0 for s in stds):
            # Fill between for error band
            ax.plot(alphas, values, color=colors[i], label=label,
                    linewidth=STYLE['linewidth'])
            ax.fill_between(alphas,
                           [v - s for v, s in zip(values, stds)],
                           [v + s for v, s in zip(values, stds)],
                           color=colors[i], alpha=0.2)
        else:
            ax.errorbar(alphas, values, yerr=stds if any(s > 0 for s in stds) else None,
                       color=colors[i], label=label,
                       linewidth=STYLE['linewidth'], marker=STYLE['marker'],
                       markersize=STYLE['markersize'], capsize=STYLE['capsize'])

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
    ax.set_ylabel(metric.replace('_', ' '), fontsize=STYLE['fontsize']['label'])
    ax.grid(True, alpha=0.3)

    # Flexible axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Flexible legend position
    if legend_loc == 'outside right':
        ax.legend(fontsize=STYLE['fontsize']['legend'],
                 bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(fontsize=STYLE['fontsize']['legend'], loc=legend_loc)

    plt.tight_layout()

    # Flexible output format
    if format != 'png':
        output_path = output_path.with_suffix(f'.{format}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=format)
    plt.close(fig)

    return output_path


def plot_with_inset(
    results: Dict[float, Dict[str, float]],
    output_path: Path,
    inset_xlim: tuple,
    inset_ylim: tuple = None,
    inset_position: str = 'upper right',
    metric: str = 'Q_Y_mean',
    dpi: int = 150,
    format: str = 'png',
) -> Path:
    """
    Create plot with inset magnification.

    Args:
        results: Dict mapping alpha -> metrics dict
        output_path: Where to save
        inset_xlim: (xmin, xmax) for inset
        inset_ylim: (ymin, ymax) for inset, or None for auto
        inset_position: 'upper right', 'upper left', etc.
        metric: Which metric to plot
        dpi: Resolution
        format: Output format

    Returns:
        Path to saved plot
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, ax = plt.subplots(figsize=(10, 6))

    alphas = sorted([float(a) for a in results.keys()])
    values = [results[a][metric] for a in alphas]

    ax.plot(alphas, values, color=COLORS['Q_Y'],
            linewidth=STYLE['linewidth'], marker=STYLE['marker'],
            markersize=STYLE['markersize'])

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
    ax.set_ylabel(metric.replace('_', ' '), fontsize=STYLE['fontsize']['label'])
    ax.grid(True, alpha=0.3)

    # Create inset
    pos_map = {
        'upper right': 1,
        'upper left': 2,
        'lower left': 3,
        'lower right': 4,
    }
    axins = inset_axes(ax, width="30%", height="30%", loc=pos_map.get(inset_position, 1))

    # Plot in inset
    axins.plot(alphas, values, color=COLORS['Q_Y'],
               linewidth=STYLE['linewidth'], marker=STYLE['marker'],
               markersize=STYLE['markersize'] * 0.7)

    axins.set_xlim(inset_xlim)
    if inset_ylim:
        axins.set_ylim(inset_ylim)
    axins.grid(True, alpha=0.3)

    plt.tight_layout()
    if format != 'png':
        output_path = output_path.with_suffix(f'.{format}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=format)
    plt.close(fig)

    return output_path


def plot_twin_axis(
    results: Dict[float, Dict[str, float]],
    output_path: Path,
    left_metric: str = 'Q_Y_mean',
    right_metric: str = 'slope',
    dpi: int = 150,
    format: str = 'png',
) -> Path:
    """
    Create plot with two Y axes.

    Args:
        results: Dict mapping alpha -> metrics dict
        output_path: Where to save
        left_metric: Metric for left Y axis
        right_metric: Metric for right Y axis
        dpi: Resolution
        format: Output format

    Returns:
        Path to saved plot
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    alphas = sorted([float(a) for a in results.keys()])
    left_values = [results[a].get(left_metric, 0) for a in alphas]
    right_values = [results[a].get(right_metric, 0) for a in alphas]

    line1, = ax1.plot(alphas, left_values, color=COLORS['Q_Y'], label=left_metric,
                      linewidth=STYLE['linewidth'], marker=STYLE['marker'],
                      markersize=STYLE['markersize'])
    line2, = ax2.plot(alphas, right_values, color='#17becf', label=right_metric,
                      linewidth=STYLE['linewidth'], marker='s',
                      markersize=STYLE['markersize'])

    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
    ax1.set_ylabel(left_metric.replace('_', ' '), fontsize=STYLE['fontsize']['label'],
                   color=COLORS['Q_Y'])
    ax2.set_ylabel(right_metric.replace('_', ' '), fontsize=STYLE['fontsize']['label'],
                   color='#17becf')

    ax1.tick_params(axis='y', labelcolor=COLORS['Q_Y'])
    ax2.tick_params(axis='y', labelcolor='#17becf')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=STYLE['fontsize']['legend'])

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if format != 'png':
        output_path = output_path.with_suffix(f'.{format}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=format)
    plt.close(fig)

    return output_path
