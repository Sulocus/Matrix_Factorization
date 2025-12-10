"""
Scientific plotting module for SMF.
Uses Matplotlib to generate publication-quality figures.

Style: Nature/Science journal formatting
- Legend on top (one row)
- Distinct colors per curve
- Clean axes
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import List, Dict, Any, Optional

# Nature/Science style
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.dpi': 150,
})

# Color palette - distinct and colorblind-friendly
COLORS = [
    '#1f77b4',  # Blue
    '#d62728',  # Red
    '#2ca02c',  # Green
    '#ff7f0e',  # Orange
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#17becf',  # Cyan
]

# Line styles
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

# Markers
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']


def plot_comparison(
    results_list: List[Dict[str, Any]], 
    metrics: List[str],
    title: Optional[str] = None,
    fig: Optional[plt.Figure] = None
) -> plt.Figure:
    """
    Generate a publication-quality comparison plot.
    
    Features:
    - Distinct colors for each (run, metric) combination
    - Legend on top in a single row
    - Scientific axis formatting
    
    Args:
        results_list: List of result dictionaries
        metrics: List of metrics to plot
        title: Optional figure title
        fig: Optional existing figure to plot on.
        
    Returns:
        Matplotlib Figure object
    """
    n_metrics = len(metrics)
    n_runs = len(results_list)
    
    if n_metrics == 0 or n_runs == 0:
        return None
    
    # Create or use figure
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        ax = fig.gca()
        ax.clear()
    
    # Plot each combination with unique color
    color_idx = 0
    for r_idx, result in enumerate(results_list):
        run_name = result.get('name', f"Run {r_idx+1}")
        alphas = result.get('alpha_values', [])
        
        for m_idx, metric in enumerate(metrics):
            values = result.get(metric, [])
            
            # Skip if no data
            min_len = min(len(alphas), len(values)) if alphas and values else 0
            if min_len == 0:
                continue
            
            # Unique color and style
            color = COLORS[color_idx % len(COLORS)]
            linestyle = '-'  # Solid for clarity
            marker = MARKERS[color_idx % len(MARKERS)]
            
            # Simplified label: just metric name for single run
            if n_runs == 1:
                label = metric
            else:
                label = f"{run_name}: {metric}"
            
            ax.plot(
                alphas[:min_len], 
                values[:min_len], 
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                marker=marker,
                markersize=4,
                markevery=max(1, min_len // 10),  # Don't overcrowd markers
                label=label,
                alpha=0.9
            )
            
            # Error bars if available
            std_key = f"{metric}_std"
            if std_key in result:
                stds = result[std_key]
                if len(stds) >= min_len:
                    ax.fill_between(
                        alphas[:min_len],
                        np.array(values[:min_len]) - np.array(stds[:min_len]),
                        np.array(values[:min_len]) + np.array(stds[:min_len]),
                        color=color,
                        alpha=0.15,
                        linewidth=0
                    )
            
            color_idx += 1
    
    # Axis styling
    ax.set_xlabel(r'$\alpha$ (Measurement Ratio)', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Y-axis limit for probability metrics
    if all(m.startswith('Q_') or 'overlap' in m.lower() for m in metrics):
        ax.set_ylim(-0.05, 1.05)
    
    # Legend on TOP, single row
    n_items = color_idx
    if n_items > 0:
        ncol = min(n_items, 4)  # Max 4 columns
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.18),
            ncol=ncol,
            fontsize=9,
            handlelength=1.5,
            columnspacing=1.0,
        )
    
    # Title
    if title:
        ax.set_title(title, fontsize=13, fontweight='medium', pad=35)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)  # Make room for legend
    
    return fig

# Specialized Wrappers
def plot_phase_transition(results: Dict, fig: Optional[plt.Figure] = None):
    """Plot Phase Transition metrics (Q_Y variants)."""
    return plot_comparison([results], ["Q_Y", "Q_Y_observed", "Q_Y_unobserved"], "Phase Transition: Q_Y vs α", fig=fig)

def plot_mse_evolution(results: Dict, fig: Optional[plt.Figure] = None):
    """Plot MSE Evolution."""
    return plot_comparison([results], ["MSE"], "MSE Evolution", fig=fig)

def plot_overlap_evolution(results: Dict, fig: Optional[plt.Figure] = None):
    """Plot Physical Overlap metrics."""
    return plot_comparison([results], ["physical_overlap_Y", "physical_overlap_X", "physical_overlap_W"], "Parameter Overlaps", fig=fig)
