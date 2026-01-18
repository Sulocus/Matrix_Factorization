"""
Result plotting module with unified styles.

This module provides publication-quality plotting for SMF experiment results.
Supports Nature/Science journal formatting with proper fonts, sizing, and error bars.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ..registry import register_output
from .base import OutputBase
from ...core.config import Config

# Import publication style system
from .publication_style import (
    apply_publication_style,
    PUB_CONFIG,
    ERROR_CONFIG,
    StyleCycler,
    plot_with_error,
    auto_legend,
    get_figure_size,
    COLORBLIND_PALETTE,
)


# Unified color scheme (kept for backward compatibility and fixed metrics)
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

# Unified style settings - now uses publication config values
# Kept for backward compatibility with existing code
STYLE = {
    'linewidth': PUB_CONFIG.linewidth_plot,
    'markersize': PUB_CONFIG.markersize,
    'marker': 'o',
    'capsize': ERROR_CONFIG.capsize,
    'fontsize': {
        'title': PUB_CONFIG.font_size_title,
        'label': PUB_CONFIG.font_size_axis_label,
        'tick': PUB_CONFIG.font_size_tick,
        'legend': PUB_CONFIG.font_size_legend,
    }
}

# Default DPI for publication quality
DEFAULT_DPI = PUB_CONFIG.dpi


@register_output(
    key="plotting",
    name="Result Plotting",
    description="Unified style Q_Y, Q_W', Q_X' curve plots",
)
class ResultPlotter(OutputBase):
    """
    Unified result plotting with consistent styles.
    """

    def __init__(self, config: Config, output_dir: Path, use_publication_style: bool = True):
        super().__init__(config, output_dir)
        if use_publication_style:
            apply_publication_style()
        else:
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
        show_error_bar: bool = True,
        error_style: str = 'bar',
    ) -> Path:
        """
        Create summary plot with Q_Y, Q_W', Q_X' vs alpha.

        Args:
            results: Dict mapping alpha -> metrics dict
            title: Plot title (optional)
            filename: Output filename
            show_params: Whether to show parameter table
            show_error_bar: Whether to show error bars (default: True)
            error_style: Error bar style - 'bar' or 'band' (default: 'bar')

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

        # Determine if error bars should be shown for each metric
        def _get_yerr(std_list):
            if show_error_bar and any(s > 0 for s in std_list):
                return std_list
            return None
        
        # Common plot kwargs
        plot_kwargs = {
            'linewidth': STYLE['linewidth'],
            'marker': STYLE['marker'],
            'markersize': STYLE['markersize'],
            'capsize': STYLE['capsize'] if show_error_bar else 0,
        }
        
        # Plot curves with unified style and optional error bars
        if error_style == 'band' and show_error_bar:
            # Band style - use fill_between
            ax.plot(alphas, qy_mean, color=COLORS['Q_Y'], label='$Q_Y$', **plot_kwargs)
            if any(s > 0 for s in qy_std):
                ax.fill_between(alphas, 
                               [m - s for m, s in zip(qy_mean, qy_std)],
                               [m + s for m, s in zip(qy_mean, qy_std)],
                               color=COLORS['Q_Y'], alpha=ERROR_CONFIG.band_alpha)
            
            ax.plot(alphas, qw_prime_mean, color=COLORS['Q_W_prime'], label="$Q'_W$", **plot_kwargs)
            if any(s > 0 for s in qw_prime_std):
                ax.fill_between(alphas,
                               [m - s for m, s in zip(qw_prime_mean, qw_prime_std)],
                               [m + s for m, s in zip(qw_prime_mean, qw_prime_std)],
                               color=COLORS['Q_W_prime'], alpha=ERROR_CONFIG.band_alpha)
            
            ax.plot(alphas, qx_prime_mean, color=COLORS['Q_X_prime'], label="$Q'_X$", **plot_kwargs)
            if any(s > 0 for s in qx_prime_std):
                ax.fill_between(alphas,
                               [m - s for m, s in zip(qx_prime_mean, qx_prime_std)],
                               [m + s for m, s in zip(qx_prime_mean, qx_prime_std)],
                               color=COLORS['Q_X_prime'], alpha=ERROR_CONFIG.band_alpha)
        else:
            # Bar style - traditional error bars
            ax.errorbar(alphas, qy_mean, yerr=_get_yerr(qy_std),
                        color=COLORS['Q_Y'], label='$Q_Y$', **plot_kwargs)

            ax.errorbar(alphas, qw_prime_mean, yerr=_get_yerr(qw_prime_std),
                        color=COLORS['Q_W_prime'], label="$Q'_W$", **plot_kwargs)

            ax.errorbar(alphas, qx_prime_mean, yerr=_get_yerr(qx_prime_std),
                        color=COLORS['Q_X_prime'], label="$Q'_X$", **plot_kwargs)

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
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
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
        show_error_bar: bool = True,
        error_style: str = 'bar',
    ) -> Path:
        """
        Create Q_Y only plot.
        
        Args:
            results: Dict mapping alpha -> metrics dict
            filename: Output filename
            show_error_bar: Whether to show error bars (default: True)
            error_style: Error bar style - 'bar' or 'band' (default: 'bar')
            
        Returns:
            Path to saved plot
        """
        alphas = sorted([float(a) for a in results.keys()])
        qy_mean = [results[a]['Q_Y_mean'] for a in alphas]
        qy_std = [results[a].get('Q_Y_std', 0) for a in alphas]

        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Determine if error bars should be shown
        yerr = qy_std if (show_error_bar and any(s > 0 for s in qy_std)) else None
        
        plot_kwargs = {
            'color': COLORS['Q_Y'],
            'label': '$Q_Y$',
            'linewidth': STYLE['linewidth'] * 1.2,
            'marker': STYLE['marker'],
            'markersize': STYLE['markersize'] * 1.2,
        }
        
        if error_style == 'band' and yerr is not None:
            ax.plot(alphas, qy_mean, **plot_kwargs)
            ax.fill_between(alphas,
                           [m - s for m, s in zip(qy_mean, qy_std)],
                           [m + s for m, s in zip(qy_mean, qy_std)],
                           color=COLORS['Q_Y'], alpha=ERROR_CONFIG.band_alpha)
        else:
            ax.errorbar(alphas, qy_mean, yerr=yerr,
                        capsize=STYLE['capsize'] if yerr else 0,
                        **plot_kwargs)

        ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylabel('$Q_Y$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=PUB_CONFIG.grid_alpha)

        m = self.config.matrix
        ax.set_title(f"$Q_Y$ vs $\\alpha$: {m.N1}×{m.N2}, M={m.M}",
                     fontsize=STYLE['fontsize']['title'])

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def plot_qy_comparison(
        self,
        results: Dict[float, Dict[str, float]],
        metrics: List[str] = None,
        filename: str = "qy_comparison.png",
        show_error_bar: bool = True,
        error_style: str = 'bar',
    ) -> Path:
        """
        Create comparison plot for multiple Q_Y-related metrics.

        This is the dynamic version that plots whatever metrics the LLM requested.

        Args:
            results: Dict mapping alpha -> metrics dict
            metrics: List of metrics to plot, e.g., ['Q_Y', 'Q_Y_unobserved']
            filename: Output filename
            show_error_bar: Whether to show error bars (default: True)
            error_style: Error bar style - 'bar' or 'band' (default: 'bar')

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
            
            # Determine if error bars should be shown
            yerr = stds if (show_error_bar and any(s > 0 for s in stds)) else None
            
            if error_style == 'band' and yerr is not None:
                ax.plot(alphas, values, color=color, label=label,
                       linewidth=STYLE['linewidth'], marker=STYLE['marker'],
                       markersize=STYLE['markersize'])
                ax.fill_between(alphas,
                               [v - s for v, s in zip(values, stds)],
                               [v + s for v, s in zip(values, stds)],
                               color=color, alpha=ERROR_CONFIG.band_alpha)
            else:
                ax.errorbar(
                    alphas, values, yerr=yerr,
                    color=color, label=label,
                    linewidth=STYLE['linewidth'],
                    marker=STYLE['marker'],
                    markersize=STYLE['markersize'],
                    capsize=STYLE['capsize'] if yerr else 0,
                )

        ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
        ax.set_ylabel('Overlap', fontsize=STYLE['fontsize']['label'])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=PUB_CONFIG.grid_alpha)
        ax.legend(fontsize=STYLE['fontsize']['legend'], loc='lower right')

        m = self.config.matrix
        ax.set_title(
            f"Q Metrics vs α: {m.N1}×{m.N2}, M={m.M}",
            fontsize=STYLE['fontsize']['title']
        )

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
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
    dpi: int = None,  # None means use DEFAULT_DPI
    error_style: str = 'bar',  # 'bar' or 'band'
    colormap: str = None,
    palette: str = 'colorblind',  # For StyleCycler
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
        dpi: Resolution (None for DEFAULT_DPI, 150 for screen, 300 for publication)
        error_style: 'bar' for error bars, 'band' for fill_between
        colormap: Matplotlib colormap name (deprecated, use palette instead)
        palette: Color palette for StyleCycler ('colorblind', 'tab10', 'extended')

    Returns:
        Path to saved plot
    """
    # Use publication DPI by default
    if dpi is None:
        dpi = DEFAULT_DPI
        
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use StyleCycler for scalable color/linestyle handling
    n_curves = len(results_list)
    if colormap:
        # Legacy colormap support (deprecated)
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_curves))
        cycler = None
    else:
        cycler = StyleCycler(n_curves, palette=palette)
        colors = None

    for i, (results, label) in enumerate(zip(results_list, labels)):
        alphas = sorted([float(a) for a in results.keys()])
        values = [results[a][metric] for a in alphas]
        stds = [results[a].get(metric.replace('_mean', '_std'), 0) for a in alphas]
        
        # Get style from cycler or colormap
        if cycler:
            style = cycler.get_style(i)
            color = style['color']
            linestyle = style.get('linestyle', '-')
            marker = style.get('marker', 'o')
        else:
            color = colors[i]
            linestyle = '-'
            marker = STYLE['marker']

        if error_style == 'band' and any(s > 0 for s in stds):
            # Fill between for error band
            ax.plot(alphas, values, color=color, label=label,
                    linewidth=STYLE['linewidth'], linestyle=linestyle, marker=marker,
                    markersize=STYLE['markersize'])
            ax.fill_between(alphas,
                           [v - s for v, s in zip(values, stds)],
                           [v + s for v, s in zip(values, stds)],
                           color=color, alpha=ERROR_CONFIG.band_alpha)
        else:
            ax.errorbar(alphas, values, yerr=stds if any(s > 0 for s in stds) else None,
                       color=color, label=label, linestyle=linestyle, marker=marker,
                       linewidth=STYLE['linewidth'], markersize=STYLE['markersize'],
                       capsize=STYLE['capsize'])

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
    ax.set_ylabel(metric.replace('_', ' '), fontsize=STYLE['fontsize']['label'])
    ax.grid(True, alpha=PUB_CONFIG.grid_alpha)

    # Flexible axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Smart legend positioning based on number of curves
    if legend_loc == 'outside right':
        ax.legend(fontsize=STYLE['fontsize']['legend'],
                 bbox_to_anchor=(1.05, 1), loc='upper left')
    elif legend_loc == 'auto':
        auto_legend(ax, n_curves)
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
    dpi: int = None,
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
    ax.grid(True, alpha=PUB_CONFIG.grid_alpha)

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
    axins.grid(True, alpha=PUB_CONFIG.grid_alpha)

    plt.tight_layout()
    if format != 'png':
        output_path = output_path.with_suffix(f'.{format}')
    plt.savefig(output_path, dpi=dpi or DEFAULT_DPI, bbox_inches='tight', format=format)
    plt.close(fig)

    return output_path


def plot_twin_axis(
    results: Dict[float, Dict[str, float]],
    output_path: Path,
    left_metric: str = 'Q_Y_mean',
    right_metric: str = 'slope',
    dpi: int = None,
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

    ax1.grid(True, alpha=PUB_CONFIG.grid_alpha)

    plt.tight_layout()
    if format != 'png':
        output_path = output_path.with_suffix(f'.{format}')
    plt.savefig(output_path, dpi=dpi or DEFAULT_DPI, bbox_inches='tight', format=format)
    plt.close(fig)

    return output_path


class TwoSegmentNorm(plt.Normalize):
    """
    Custom normalization that enhances discrimination in high-value region.
    
    Maps:
        0.0 ~ breakpoint  ->  0.0 ~ color_breakpoint (compressed)
        breakpoint ~ 1.0  ->  color_breakpoint ~ 1.0 (expanded)
    
    This gives more color resolution to values near 1.0.
    """
    def __init__(self, breakpoint=0.9, color_breakpoint=0.6, vmin=0.0, vmax=1.0):
        super().__init__(vmin=vmin, vmax=vmax)
        self.breakpoint = breakpoint
        self.color_breakpoint = color_breakpoint
    
    def __call__(self, value, clip=None):
        # Normalize to 0-1 first
        x = np.asarray(value)
        result = np.zeros_like(x, dtype=float)
        
        # Low segment: 0 ~ breakpoint -> 0 ~ color_breakpoint
        low_mask = x <= self.breakpoint
        if self.breakpoint > 0:
            result[low_mask] = (x[low_mask] / self.breakpoint) * self.color_breakpoint
        
        # High segment: breakpoint ~ 1 -> color_breakpoint ~ 1
        high_mask = x > self.breakpoint
        if self.breakpoint < 1.0:
            result[high_mask] = self.color_breakpoint + \
                ((x[high_mask] - self.breakpoint) / (1.0 - self.breakpoint)) * (1.0 - self.color_breakpoint)
        
        return np.ma.masked_array(result)


def plot_replica_heatmap(
    matrix: np.ndarray,
    alpha: float,
    output_dir: Path,
    metric_name: str = "Overlap",
    filename_prefix: str = "heatmap",
    cmap: str = "RdYlBu_r",
    enhance_high_values: bool = True,
    breakpoint: float = 0.9,
    color_breakpoint: float = 0.6,
    rsb_ordering: bool = False,
) -> Path:
    """
    Plot (S+1)x(S+1) replica interaction heatmap.
    
    Args:
        matrix: (S+1, S+1) interaction matrix
        alpha: Current alpha value
        output_dir: Directory to save plot
        metric_name: Name of the metric (e.g., "$Q_W$")
        filename_prefix: Prefix for filename
        cmap: Colormap (default: RdYlBu_r where Red=1, Blue=0)
        enhance_high_values: If True, use non-linear norm to enhance 0.9-1.0 range
        breakpoint: Value where color mapping changes (default 0.9)
        color_breakpoint: Position in colormap at breakpoint (default 0.6)
        rsb_ordering: If True, reorder replicas using hierarchical clustering 
                      to reveal RSB structure (block diagonal pattern)
        
    Returns:
        Path to saved PNG
    """
    S_plus_1 = matrix.shape[0]
    S = S_plus_1 - 1
    
    # RSB ordering: use hierarchical clustering to reorder replicas
    if rsb_ordering and S > 2:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        
        # Extract replica-replica submatrix (exclude Teacher row/column 0)
        replica_matrix = matrix[1:, 1:]
        
        # Convert similarity to distance (1 - overlap)
        distance_matrix = 1 - replica_matrix
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0
        
        # Use condensed form for linkage
        condensed = squareform(distance_matrix, checks=False)
        
        # Hierarchical clustering
        Z = linkage(condensed, method='average')
        order = leaves_list(Z)
        
        # Reorder: Teacher stays at position 0, replicas reordered
        full_order = [0] + [i + 1 for i in order]
        matrix = matrix[np.ix_(full_order, full_order)]
    
    # Square plot
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Choose normalization
    if enhance_high_values:
        norm = TwoSegmentNorm(breakpoint=breakpoint, color_breakpoint=color_breakpoint)
    else:
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
    
    # Plot heatmap with custom norm
    im = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')
    
    # Add colorbar with explicit ticks showing actual values
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name, rotation=270, labelpad=15)
    
    # Set colorbar ticks to show actual data values (not normalized)
    # Key values: 0, 0.5, 0.9, 0.95, 1.0
    cbar_ticks = [0.0, 0.5, 0.9, 0.95, 1.0]
    cbar.set_ticks([norm(t) for t in cbar_ticks])
    cbar.set_ticklabels([f'{t:.2f}' for t in cbar_ticks])
    
    # Configure ticks
    # 0 is Teacher, 1..S are Replicas
    ticks = np.arange(S_plus_1)
    
    if S_plus_1 > 20:
        # Sparse ticks for large matrices
        step = max(1, S_plus_1 // 10)
        shown_indices = [0] + list(range(step, S_plus_1, step))
        if shown_indices[-1] != S_plus_1 - 1:
            shown_indices.append(S_plus_1 - 1)
            
        tick_labels = ['T' if i == 0 else str(i) for i in shown_indices]
        ax.set_xticks(shown_indices)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(shown_indices)
        ax.set_yticklabels(tick_labels)
    else:
        tick_labels = ['T'] + [str(i) for i in range(1, S_plus_1)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)
        
    ax.set_xlabel("Replicas (T=Teacher)")
    ax.set_ylabel("Replicas (T=Teacher)")
    ax.set_title(f"{metric_name} Map\n$\\alpha={alpha:.4f}$")
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    filename = f"{filename_prefix}_alpha_{alpha:.6f}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_gif(
    image_paths: List[Path],
    output_path: Path,
    duration: float = 0.5,
    loop: int = 0,
) -> Path:
    """
    Create a GIF from a list of images.
    
    Args:
        image_paths: List of paths to images (sorted)
        output_path: Path to save the GIF
        duration: Duration of each frame in seconds
        loop: Number of loops (0 = infinite)
        
    Returns:
        Path to saved GIF
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL/Pillow not installed. Skipping GIF generation.")
        return None
        
    if not image_paths:
        return None
        
    # Open images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"Failed to open image {path}: {e}")
            
    if not images:
        return None
        
    # Save GIF (duration in PIL is milliseconds)
    duration_ms = int(duration * 1000)
    
    try:
        image_paths[0].parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=duration * 1000,
            loop=0
        )
        return output_path
    except Exception as e:
        print(f"Failed to save GIF: {e}")
        return None


def plot_overlap_evolution(
    results: Dict[float, Dict[str, float]],
    output_dir: Path,
    filename: str = "overlap_evolution.png",
) -> Path:
    """
    Plot evolution of Teacher-Student and Student-Student overlaps.
    
    Args:
        results: Dictionary mapping alpha to metrics
        output_dir: Directory to save plot
        filename: Output filename
        
    Returns:
        Path to saved PNG
    """
    alphas = sorted([float(k) for k in results.keys()])
    q_ts = []  # Teacher-Student (Q_W)
    q_ss = []  # Student-Student (Q_W_replica)
    
    for alpha in alphas:
        metrics = results[alpha]
        # TS Overlap
        if 'Q_W_mean' in metrics:
            q_ts.append(metrics['Q_W_mean'])
        elif 'Q_W' in metrics:
             q_ts.append(metrics['Q_W'])
        else:
            q_ts.append(0.0)
            
        # SS Overlap
        if 'Q_W_replica_mean' in metrics:
            q_ss.append(metrics['Q_W_replica_mean'])
        elif 'Q_W_replica' in metrics:
            q_ss.append(metrics['Q_W_replica'])
        else:
            # Fallback for single replica case
            q_ss.append(1.0 if 'Q_W_mean' in metrics else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot curves
    ax.plot(alphas, q_ts, 'b-o', label='Teacher-Student ($Q_{TS}$)', linewidth=2, markersize=4)
    ax.plot(alphas, q_ss, 'r--s', label='Student-Student ($Q_{SS}$)', linewidth=2, markersize=4)
    
    ax.set_xlabel(r'$\alpha$ (Measurement Density)')
    ax.set_ylabel('Overlap (Normalized)')
    ax.set_title('Replica Overlap Evolution')
    ax.grid(True, alpha=PUB_CONFIG.grid_alpha)
    ax.legend(loc='lower right')
    ax.set_ylim(-0.1, 1.1)
    
    # Add phase transition marker if applicable
    # (Optional: could detect jump)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_replica_analysis(
    replica_results: Dict[float, Dict[str, float]],
    output_path: Path,
    title: str = None,
    N1: int = None,
    N2: int = None,
    M: int = None,
    S: int = None,
    show_teacher: bool = True,
    dpi: int = None,
) -> Path:
    """
    Generate publication-quality replica overlap analysis plot.
    
    Creates a 2x2 subplot figure:
    1. Q_Y: Replica vs Teacher
    2. Q_W and Q_X replica overlap
    3. Q_Y replica main result
    4. Standard deviation across pairs
    
    Uses publication_style for Nature/Science quality.
    
    Args:
        replica_results: Dict mapping alpha -> replica metrics
            Required keys: Q_Y_replica_mean, Q_W_replica_mean, etc.
        output_path: Where to save the plot
        title: Optional custom title
        N1, N2, M: Matrix dimensions for subtitle
        S: Number of replicas
        show_teacher: Whether to show teacher overlap comparison
        dpi: Output DPI (default: publication quality)
        
    Returns:
        Path to saved plot
    """
    # Apply publication style
    apply_publication_style()
    
    alphas = sorted([float(a) for a in replica_results.keys()])
    
    # Extract metrics
    qy_replica = [replica_results[a].get('Q_Y_replica_mean', 0) for a in alphas]
    qy_replica_std = [replica_results[a].get('Q_Y_replica_std', 0) for a in alphas]
    qw_replica = [replica_results[a].get('Q_W_replica_norm_mean', 
                  replica_results[a].get('Q_W_replica_mean', 0)) for a in alphas]
    qx_replica = [replica_results[a].get('Q_X_replica_norm_mean',
                  replica_results[a].get('Q_X_replica_mean', 0)) for a in alphas]
    
    # Teacher overlap if available
    qy_teacher = [replica_results[a].get('Q_Y_teacher_mean', 0) for a in alphas]
    qy_teacher_std = [replica_results[a].get('Q_Y_teacher_std', 0) for a in alphas]
    
    # Colors - using colorblind-safe palette
    REPLICA_COLOR = COLORBLIND_PALETTE[0]  # Blue
    TEACHER_COLOR = COLORBLIND_PALETTE[1]  # Orange
    W_COLOR = COLORBLIND_PALETTE[2]        # Green
    X_COLOR = COLORBLIND_PALETTE[3]        # Pink
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # ========== Plot 1: Q_Y Replica vs Teacher ==========
    ax1 = axes[0, 0]
    plot_with_error(ax1, np.array(alphas), np.array(qy_replica), 
                    yerr=np.array(qy_replica_std),
                    color=REPLICA_COLOR, label='$Q_Y$ (replica)',
                    marker='o', linewidth=STYLE['linewidth'])
    
    if show_teacher and any(q > 0 for q in qy_teacher):
        plot_with_error(ax1, np.array(alphas), np.array(qy_teacher),
                        yerr=np.array(qy_teacher_std),
                        color=TEACHER_COLOR, label='$Q_Y$ (teacher)',
                        marker='s', linestyle='--', linewidth=STYLE['linewidth'] * 0.8)
    
    ax1.set_xlabel(r'$\tilde{\alpha}$')
    ax1.set_ylabel('$Q_Y$')
    ax1.set_title('Y Overlap: Replica vs Teacher', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=STYLE['fontsize']['legend'])
    ax1.grid(True, alpha=PUB_CONFIG.grid_alpha)
    ax1.set_ylim(-0.05, 1.05)
    
    # ========== Plot 2: Q_W and Q_X Replica ==========
    ax2 = axes[0, 1]
    ax2.plot(alphas, qw_replica, 'o-', color=W_COLOR, 
             label="$Q'_W$ (replica)", linewidth=STYLE['linewidth'],
             markersize=STYLE['markersize'])
    ax2.plot(alphas, qx_replica, 's-', color=X_COLOR,
             label="$Q'_X$ (replica)", linewidth=STYLE['linewidth'],
             markersize=STYLE['markersize'])
    
    ax2.set_xlabel(r'$\tilde{\alpha}$')
    ax2.set_ylabel('Overlap (normalized)')
    ax2.set_title('W and X Replica Overlap', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=STYLE['fontsize']['legend'])
    ax2.grid(True, alpha=PUB_CONFIG.grid_alpha)
    ax2.set_ylim(-0.05, 1.05)
    
    # ========== Plot 3: Q_Y Replica Main Result ==========
    ax3 = axes[1, 0]
    plot_with_error(ax3, np.array(alphas), np.array(qy_replica),
                    yerr=np.array(qy_replica_std),
                    color=REPLICA_COLOR, 
                    marker='o', linewidth=STYLE['linewidth'] * 1.3,
                    markersize=STYLE['markersize'] * 1.2)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, 
                label='Perfect consistency')
    
    ax3.set_xlabel(r'$\tilde{\alpha}$')
    ax3.set_ylabel('$Q_Y$ (replica)')
    ax3.set_title('Replica Consistency (Main Result)', fontweight='bold')
    ax3.grid(True, alpha=PUB_CONFIG.grid_alpha)
    ax3.set_ylim(-0.05, 1.05)
    
    # ========== Plot 4: Standard Deviation ==========
    ax4 = axes[1, 1]
    ax4.plot(alphas, qy_replica_std, 'o-', color=REPLICA_COLOR,
             label='$Q_Y$ std', linewidth=STYLE['linewidth'],
             markersize=STYLE['markersize'])
    
    qw_std = [replica_results[a].get('Q_W_replica_std', 0) for a in alphas]
    qx_std = [replica_results[a].get('Q_X_replica_std', 0) for a in alphas]
    ax4.plot(alphas, qw_std, 's-', color=W_COLOR, alpha=0.7,
             label='$Q_W$ std', linewidth=STYLE['linewidth'] * 0.8)
    ax4.plot(alphas, qx_std, '^-', color=X_COLOR, alpha=0.7,
             label='$Q_X$ std', linewidth=STYLE['linewidth'] * 0.8)
    
    ax4.set_xlabel(r'$\tilde{\alpha}$')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_title('Overlap Variance Across Pairs', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=STYLE['fontsize']['legend'])
    ax4.grid(True, alpha=PUB_CONFIG.grid_alpha)
    
    # ========== Suptitle ==========
    if title:
        suptitle = title
    else:
        parts = []
        if N1 is not None and N2 is not None:
            parts.append(f'{N1}×{N2}')
        if M is not None:
            parts.append(f'M={M}')
        if S is not None:
            n_pairs = S * (S - 1) // 2
            parts.append(f'S={S} replicas ({n_pairs} pairs)')
        suptitle = 'Replica Overlap Analysis'
        if parts:
            suptitle += f': {", ".join(parts)}'
    
    plt.suptitle(suptitle, fontsize=STYLE['fontsize']['title'], 
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi or DEFAULT_DPI, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    
    return output_path


def plot_replica_comparison(
    replica_results: Dict[float, Dict[str, float]],
    output_path: Path,
    dpi: int = None,
) -> Path:
    """
    Generate Teacher-Student vs Replica-Replica comparison plot.
    
    Shows both overlap types on same axes for direct comparison.
    Uses normalized (baseline-corrected) overlap for fair comparison.
    
    Args:
        replica_results: Dict mapping alpha -> replica metrics
        output_path: Where to save
        dpi: Output DPI
        
    Returns:
        Path to saved plot
    """
    apply_publication_style()
    
    alphas = sorted([float(a) for a in replica_results.keys()])
    
    # Teacher-Student overlap (normalized)
    qw_teacher = [replica_results[a].get('Q_W_teacher_mean', 0) for a in alphas]
    qx_teacher = [replica_results[a].get('Q_X_teacher_mean', 0) for a in alphas]
    
    # Replica-Replica overlap (normalized)
    qw_replica = [replica_results[a].get('Q_W_replica_norm_mean', 
                  replica_results[a].get('Q_W_replica_mean', 0)) for a in alphas]
    qx_replica = [replica_results[a].get('Q_X_replica_norm_mean',
                  replica_results[a].get('Q_X_replica_mean', 0)) for a in alphas]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Teacher-Student curves
    ax.plot(alphas, qw_teacher, 'o-', color=COLORBLIND_PALETTE[4], 
            label="$Q'_W$ (teacher-student)", linewidth=STYLE['linewidth'],
            markersize=STYLE['markersize'], alpha=0.8)
    ax.plot(alphas, qx_teacher, 'v-', color=COLORBLIND_PALETTE[5],
            label="$Q'_X$ (teacher-student)", linewidth=STYLE['linewidth'],
            markersize=STYLE['markersize'], alpha=0.8)
    
    # Replica-Replica curves
    ax.plot(alphas, qw_replica, 's--', color=COLORBLIND_PALETTE[2],
            label="$Q'_W$ (replica-replica)", linewidth=STYLE['linewidth'],
            markersize=STYLE['markersize'], alpha=0.8)
    ax.plot(alphas, qx_replica, '^--', color=COLORBLIND_PALETTE[3],
            label="$Q'_X$ (replica-replica)", linewidth=STYLE['linewidth'],
            markersize=STYLE['markersize'], alpha=0.8)
    
    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
    ax.set_ylabel('Overlap (normalized)', fontsize=STYLE['fontsize']['label'])
    ax.set_title('Teacher-Student vs Replica-Replica Overlap\n(Both using baseline correction)',
                 fontsize=STYLE['fontsize']['title'], fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=PUB_CONFIG.grid_alpha)
    ax.legend(fontsize=STYLE['fontsize']['legend'], loc='lower right')
    
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi or DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_custom_curves(
    results: Dict[float, Dict[str, float]],
    curves: List[str],
    output_path: Path,
    title: str = None,
    show_error_bar: bool = True,
    error_style: str = 'bar',
    dpi: int = None,
) -> Path:
    """
    根据配置绘制自定义曲线组合。

    使用 plot_registry 中定义的格式解析曲线代码。
    
    Args:
        results: Dict mapping scan_value -> metrics dict
                 例如: {1.0: {'Q_Y_mean': 0.5, 'Q_Y_std': 0.01, ...}, ...}
        curves: 曲线代码列表，例如 ['A.y', 'B.w', 'A.y:R']
        output_path: 输出路径
        title: 图片标题 (可选)
        show_error_bar: 是否显示误差棒
        error_style: 误差样式 'bar' 或 'band'
        dpi: 分辨率

    Returns:
        Path to saved plot
    """
    from .plot_registry import parse_curve_code, CurveSpec
    
    # 解析曲线配置
    curve_specs: List[CurveSpec] = []
    for code in curves:
        try:
            spec = parse_curve_code(code)
            curve_specs.append(spec)
        except ValueError as e:
            print(f"Warning: Invalid curve code '{code}': {e}")
            continue
    
    if not curve_specs:
        print("No valid curves to plot")
        return None
    
    # 准备数据
    x_values = sorted([float(v) for v in results.keys()])
    
    # Metric 显示名称 (LaTeX)
    METRIC_LABELS = {
        'Q_Y': '$Q_Y$',
        'Q_W': '$Q_W$',
        'Q_X': '$Q_X$',
        'Q_W_prime': "$Q'_W$",
        'Q_X_prime': "$Q'_X$",
        'Q_Y_observed': '$Q_Y$ (obs)',
        'Q_Y_unobserved': '$Q_Y$ (unobs)',
        'physical_overlap_Y': 'Phys $Q_Y$',
        'physical_overlap_W': 'Phys $Q_W$',
        'physical_overlap_X': 'Phys $Q_X$',
        'Gen_Error': 'Gen Error',
    }
    
    # 扩展颜色映射 (支持 replica)
    EXTENDED_COLORS = {
        'Q_Y': '#d62728',
        'Q_Y_replica': '#ff9999',
        'Q_W': '#1f77b4',
        'Q_W_replica': '#a0c8e0',
        'Q_X': '#ff7f0e',
        'Q_X_replica': '#ffcc99',
        'Q_W_prime': '#9467bd',
        'Q_W_prime_replica': '#c9b3d6',
        'Q_X_prime': '#8c564b',
        'Q_X_prime_replica': '#c4a59e',
        'Q_Y_observed': '#bcbd22',
        'Q_Y_observed_replica': '#e0e088',
        'Q_Y_unobserved': '#17becf',
        'Q_Y_unobserved_replica': '#88daed',
        'physical_overlap_Y': '#d62728',
        'physical_overlap_Y_replica': '#ff9999',
        'physical_overlap_W': '#1f77b4',
        'physical_overlap_W_replica': '#a0c8e0',
        'physical_overlap_X': '#ff7f0e',
        'physical_overlap_X_replica': '#ffcc99',
        'Gen_Error': '#2ca02c',
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for spec in curve_specs:
        # 构建 metric key (加 _replica 后缀如果需要)
        if spec.is_replica:
            metric_key = spec.metric_name + '_replica'
        else:
            metric_key = spec.metric_name
        
        mean_key = f'{metric_key}_mean'
        std_key = f'{metric_key}_std'
        
        # 提取数据
        means = []
        stds = []
        for v in x_values:
            metrics = results.get(v, {})
            means.append(metrics.get(mean_key, 0))
            stds.append(metrics.get(std_key, 0))
        
        # 获取颜色和标签
        color = EXTENDED_COLORS.get(metric_key, '#333333')
        base_label = METRIC_LABELS.get(spec.metric_name, spec.metric_name)
        label = f"{base_label} (R)" if spec.is_replica else base_label
        
        # 绘制曲线
        yerr = stds if (show_error_bar and any(s > 0 for s in stds)) else None
        linestyle = '--' if spec.is_replica else '-'
        marker = 's' if spec.is_replica else 'o'
        
        if error_style == 'band' and yerr is not None:
            ax.plot(x_values, means, color=color, label=label,
                   linewidth=STYLE['linewidth'], linestyle=linestyle, marker=marker,
                   markersize=STYLE['markersize'])
            ax.fill_between(x_values,
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           color=color, alpha=ERROR_CONFIG.band_alpha)
        else:
            ax.errorbar(x_values, means, yerr=yerr,
                       color=color, label=label, linestyle=linestyle, marker=marker,
                       linewidth=STYLE['linewidth'], markersize=STYLE['markersize'],
                       capsize=STYLE['capsize'] if yerr else 0)
    
    # 格式化
    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
    ax.set_ylabel('Overlap', fontsize=STYLE['fontsize']['label'])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=PUB_CONFIG.grid_alpha)
    ax.legend(fontsize=STYLE['fontsize']['legend'], loc='best')
    
    if title:
        ax.set_title(title, fontsize=STYLE['fontsize']['title'])
    
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi or DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def plot_multi_metric_comparison(
    results_list: List[Dict[str, Any]],
    labels: List[str],
    output_path: Path,
    metrics: List[str],
    title: str = None,
    # Style options
    legend_loc: str = 'best',
    format: str = 'png',
    dpi: int = None,
) -> Path:
    """
    Plot comparison of multiple metrics across multiple experiments on a SINGLE plot.
    
    Styling Strategy:
    - Color: Distinguished by Metric (e.g. Q_Y=Red, Q_W=Blue)
    - LineStyle/Marker: Distinguished by Experiment (e.g. Cold=Solid, Warm=Dashed)
    
    Args:
        results_list: List of results dictionaries
        labels: Labels for each experiment (e.g. ['Cold Start', 'Warm Start'])
        output_path: Output file path
        metrics: List of metric keys to plot (e.g. ['Q_Y_mean', 'Q_W_prime_mean'])
        title: Plot title
        legend_loc: Legend location
        format: Output format
        dpi: Output DPI
        
    Returns:
        Path to saved plot
    """
    if dpi is None:
        dpi = DEFAULT_DPI
        
    fig, ax = plt.subplots(figsize=(10, 6))

    # Metric-to-Color Map (Extend global COLORS with fallback)
    # Using specific colors for standard metrics to maintain consistency
    METRIC_STYLE = {
        'Q_Y_mean': {'color': COLORS['Q_Y'], 'label': '$Q_Y$'},
        'Q_Y': {'color': COLORS['Q_Y'], 'label': '$Q_Y$'},
        
        'Q_W_prime_mean': {'color': COLORS['Q_W_prime'], 'label': "$Q'_W$"},
        'Q_W_prime': {'color': COLORS['Q_W_prime'], 'label': "$Q'_W$"},
        
        'Q_W_mean': {'color': COLORS['Q_W'], 'label': '$Q_W$'},
        'Q_W': {'color': COLORS['Q_W'], 'label': '$Q_W$'},
        
        'Q_X_prime_mean': {'color': COLORS['Q_X_prime'], 'label': "$Q'_X$"},
        'Q_X_prime': {'color': COLORS['Q_X_prime'], 'label': "$Q'_X$"},
        
        'MSE': {'color': '#2ca02c', 'label': 'MSE'},
        'D.y': {'color': '#2ca02c', 'label': 'MSE'},
        
        'physical_overlap_Y_mean': {'color': '#bcbd22', 'label': 'Phys-$Q_Y$'},
        'physical_overlap_W_mean': {'color': '#8c564b', 'label': 'Phys-$Q_W$'},
    }
    
    # Experiment-to-Style Map
    # Index 0 (Cold): Solid line, Circle
    # Index 1 (Warm): Dashed line, Triangle/Square
    EXP_STYLES = [
        {'linestyle': '-', 'marker': 'o'},     # Exp 1 (Cold)
        {'linestyle': '--', 'marker': 's'},    # Exp 2 (Warm)
        {'linestyle': ':', 'marker': '^'},     # Exp 3
        {'linestyle': '-.', 'marker': 'D'},    # Exp 4
    ]

    for metric in metrics:
        # Resolve Metric Style
        # Remove '_mean' suffix for checking fallback if needed
        base_key = metric.replace('_mean', '')
        
        if metric in METRIC_STYLE:
            m_style = METRIC_STYLE[metric]
        elif base_key in METRIC_STYLE:
             m_style = METRIC_STYLE[base_key]
        else:
             # Fallback color from palette if not defined
             import hashlib
             idx = int(hashlib.md5(metric.encode()).hexdigest(), 16) % len(COLORBLIND_PALETTE)
             m_style = {'color': COLORBLIND_PALETTE[idx], 'label': metric.replace('_', ' ')}
        
        metric_color = m_style['color']
        metric_label_base = m_style['label']
        
        # Iterate Experiments
        for i, (results, exp_label) in enumerate(zip(results_list, labels)):
            alphas = sorted([float(a) for a in results.keys()])
            
            # Allow metric lookup fallbacks (e.g. Q_Y vs Q_Y_mean)
            if metric in results[alphas[0]]:
                key = metric
            else:
                # Try simple variations
                alt_keys = [metric + '_mean', metric.replace('_mean', '')]
                found = False
                for k in alt_keys:
                    if k in results[alphas[0]]:
                        key = k
                        found = True
                        break
                if not found:
                    print(f"⚠️ Metric {metric} not found for {exp_label}")
                    continue
            
            values = [results[a][key] for a in alphas]
            std_key = key.replace('_mean', '_std')
            stds = [results[a].get(std_key, 0) for a in alphas]
            
            # Resolve Experiment Style
            exp_style = EXP_STYLES[i % len(EXP_STYLES)]
            
            combined_label = f"{exp_label} ({metric_label_base})"
            
            # Plot
            # Only show error bars if significant
            has_error = any(s > 0.001 for s in stds)
            
            ax.errorbar(
                alphas, values, yerr=stds if has_error else None,
                color=metric_color,
                linestyle=exp_style['linestyle'],
                marker=exp_style['marker'],
                label=combined_label,
                linewidth=STYLE['linewidth'],
                markersize=STYLE['markersize'],
                capsize=STYLE['capsize'] if has_error else 0,
                alpha=0.9
            )

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=STYLE['fontsize']['label'])
    ax.set_ylabel('Value', fontsize=STYLE['fontsize']['label'])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=PUB_CONFIG.grid_alpha)
    
    if legend_loc == 'outside right':
        ax.legend(fontsize=STYLE['fontsize']['legend'], bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(fontsize=STYLE['fontsize']['legend'], loc=legend_loc)
        
    if title:
        ax.set_title(title, fontsize=STYLE['fontsize']['title'])
        
    plt.tight_layout()
    
    if format != 'png':
        output_path = output_path.with_suffix(f'.{format}')
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=format)
    plt.close(fig)
    
    return output_path
