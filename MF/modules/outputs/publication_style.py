"""
Publication-quality figure styling for Nature/Science journals.

This module provides a centralized configuration system for generating
publication-ready figures with proper fonts, sizes, and error bar handling.

Features:
    - Nature/Science compliant font and sizing
    - Smart error bar display (bar or band style)
    - Scalable multi-curve color/linestyle cycling
    - Automatic legend layout
    - Panel labeling (a, b, c...)

Usage:
    from MF.modules.outputs.publication_style import (
        apply_publication_style,
        StyleCycler,
        PUB_CONFIG,
    )
    
    # Apply global style at script start
    apply_publication_style()
    
    # For multi-curve plots
    cycler = StyleCycler(n_curves=10)
    for i, data in enumerate(datasets):
        style = cycler.get_style(i)
        ax.plot(x, y, **style)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Iterator
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# =============================================================================
# Publication Configuration
# =============================================================================

@dataclass
class PublicationConfig:
    """
    Configuration for publication-quality figures.
    
    Based on Nature/Science journal guidelines:
    - Single column width: 88mm (3.46 inches)
    - Double column width: 180mm (7.09 inches)
    - Minimum font size: 5-7pt
    - Sans-serif fonts (Arial/Helvetica)
    - Resolution: 300+ dpi
    """
    
    # Figure dimensions (inches)
    single_column_width: float = 3.46   # 88mm
    double_column_width: float = 7.09   # 180mm
    default_aspect_ratio: float = 0.75  # height/width
    
    # Resolution
    dpi: int = 300
    
    # Font configuration
    font_family: str = 'sans-serif'
    font_sans_serif: List[str] = field(
        default_factory=lambda: ['Arial', 'Helvetica', 'DejaVu Sans']
    )
    
    # Font sizes (pt) - designed to be readable after journal reduction
    font_size_base: int = 8
    font_size_axis_label: int = 8
    font_size_tick: int = 7
    font_size_legend: int = 7
    font_size_title: int = 9
    font_size_panel_label: int = 8  # For (a), (b), (c) labels
    
    # Line widths (pt)
    linewidth_plot: float = 1.0
    linewidth_axis: float = 0.5
    linewidth_error: float = 0.8
    linewidth_grid: float = 0.3
    
    # Tick configuration
    tick_direction: str = 'in'
    tick_length_major: float = 3.0
    tick_length_minor: float = 1.5
    tick_width: float = 0.5
    
    # Marker configuration
    markersize: float = 3.0
    markeredgewidth: float = 0.5
    
    # Grid
    grid_alpha: float = 0.3
    
    # Legend
    legend_framealpha: float = 0.9
    legend_edgecolor: str = 'none'
    

@dataclass
class ErrorBarConfig:
    """Configuration for error bar display."""
    
    show: bool = True
    style: str = 'bar'  # 'bar', 'band', 'auto'
    error_type: str = 'std'  # 'std', 'sem', 'ci95'
    
    # Bar style settings
    capsize: float = 2.0
    capthick: float = 0.8
    elinewidth: float = 0.8
    
    # Band style settings (fill_between)
    band_alpha: float = 0.2
    
    # Auto style threshold (use band if n_points > threshold)
    auto_threshold: int = 20


# Global configuration instances
PUB_CONFIG = PublicationConfig()
ERROR_CONFIG = ErrorBarConfig()


# =============================================================================
# Style Application
# =============================================================================

def apply_publication_style(config: Optional[PublicationConfig] = None) -> None:
    """
    Apply publication-quality matplotlib rcParams globally.
    
    Call this at the start of your script or in __init__ of plotting classes.
    
    Args:
        config: Optional custom configuration. Uses PUB_CONFIG if None.
    """
    cfg = config or PUB_CONFIG
    
    plt.rcParams.update({
        # Font
        'font.family': cfg.font_family,
        'font.sans-serif': cfg.font_sans_serif,
        'font.size': cfg.font_size_base,
        
        # Axes
        'axes.labelsize': cfg.font_size_axis_label,
        'axes.titlesize': cfg.font_size_title,
        'axes.linewidth': cfg.linewidth_axis,
        'axes.grid': False,  # We'll add grid manually with custom alpha
        
        # Ticks
        'xtick.labelsize': cfg.font_size_tick,
        'ytick.labelsize': cfg.font_size_tick,
        'xtick.direction': cfg.tick_direction,
        'ytick.direction': cfg.tick_direction,
        'xtick.major.size': cfg.tick_length_major,
        'ytick.major.size': cfg.tick_length_major,
        'xtick.minor.size': cfg.tick_length_minor,
        'ytick.minor.size': cfg.tick_length_minor,
        'xtick.major.width': cfg.tick_width,
        'ytick.major.width': cfg.tick_width,
        'xtick.top': True,
        'ytick.right': True,
        
        # Legend
        'legend.fontsize': cfg.font_size_legend,
        'legend.framealpha': cfg.legend_framealpha,
        'legend.edgecolor': cfg.legend_edgecolor,
        
        # Figure
        'figure.dpi': cfg.dpi,
        'savefig.dpi': cfg.dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Lines
        'lines.linewidth': cfg.linewidth_plot,
        'lines.markersize': cfg.markersize,
        'lines.markeredgewidth': cfg.markeredgewidth,
        
        # Error bars
        'errorbar.capsize': ERROR_CONFIG.capsize,
    })


def get_figure_size(
    columns: str = 'single',
    aspect_ratio: Optional[float] = None,
    config: Optional[PublicationConfig] = None,
) -> Tuple[float, float]:
    """
    Get figure size for publication.
    
    Args:
        columns: 'single' or 'double' column width
        aspect_ratio: height/width ratio. None uses default (0.75)
        config: Optional custom configuration
        
    Returns:
        (width, height) in inches
    """
    cfg = config or PUB_CONFIG
    ratio = aspect_ratio or cfg.default_aspect_ratio
    
    if columns == 'double':
        width = cfg.double_column_width
    else:
        width = cfg.single_column_width
    
    return (width, width * ratio)


# =============================================================================
# Color-Blind Safe Palettes
# =============================================================================

# Okabe-Ito colorblind-safe palette (widely recommended)
COLORBLIND_PALETTE = [
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion (orange-red)
    '#009E73',  # Bluish green
    '#E69F00',  # Orange
    '#56B4E9',  # Sky blue
    '#CC79A7',  # Reddish purple
    '#F0E442',  # Yellow
    '#000000',  # Black
]

# Extended palette for many curves (combines with linestyles)
EXTENDED_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


# =============================================================================
# Style Cycler for Multiple Curves
# =============================================================================

class StyleCycler:
    """
    Scalable style cycler for plots with many curves.
    
    Automatically combines colors, linestyles, and markers to ensure
    all curves are visually distinguishable.
    
    Example:
        cycler = StyleCycler(n_curves=15, palette='colorblind')
        for i, (x, y, label) in enumerate(datasets):
            style = cycler.get_style(i)
            ax.plot(x, y, label=label, **style)
    """
    
    # Linestyle cycle
    LINESTYLES = [
        '-',      # solid
        '--',     # dashed
        '-.',     # dash-dot
        ':',      # dotted
        (0, (5, 1)),       # densely dashed
        (0, (3, 1, 1, 1)), # densely dash-dotted
    ]
    
    # Marker cycle
    MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']
    
    def __init__(
        self,
        n_curves: int,
        palette: str = 'colorblind',
        use_linestyles: bool = True,
        use_markers: bool = True,
    ):
        """
        Initialize style cycler.
        
        Args:
            n_curves: Total number of curves to style
            palette: Color palette name ('colorblind', 'tab10', 'extended', 
                     or any matplotlib colormap name)
            use_linestyles: Whether to cycle through linestyles
            use_markers: Whether to cycle through markers
        """
        self.n = n_curves
        self.use_linestyles = use_linestyles
        self.use_markers = use_markers
        self.colors = self._get_palette(palette, n_curves)
    
    def _get_palette(self, name: str, n: int) -> List[str]:
        """Get color palette with enough colors for n curves."""
        if name == 'colorblind':
            base = COLORBLIND_PALETTE
        elif name == 'extended':
            base = EXTENDED_PALETTE
        elif name == 'tab10':
            base = [mpl.colors.rgb2hex(c) for c in plt.cm.tab10.colors]
        elif name == 'tab20':
            base = [mpl.colors.rgb2hex(c) for c in plt.cm.tab20.colors]
        else:
            # Use as colormap name
            try:
                cmap = plt.cm.get_cmap(name)
                return [mpl.colors.rgb2hex(cmap(i / max(n - 1, 1))) for i in range(n)]
            except ValueError:
                base = COLORBLIND_PALETTE
        
        # Cycle colors if needed
        return [base[i % len(base)] for i in range(n)]
    
    def get_style(self, index: int) -> Dict[str, Any]:
        """
        Get style dictionary for curve at given index.
        
        Args:
            index: Curve index (0-based)
            
        Returns:
            Dictionary with 'color', 'linestyle', 'marker' keys
        """
        n_colors = len(self.colors)
        color = self.colors[index % n_colors]
        
        style = {'color': color}
        
        if self.use_linestyles:
            # Change linestyle when cycling through colors
            ls_idx = (index // n_colors) % len(self.LINESTYLES)
            style['linestyle'] = self.LINESTYLES[ls_idx]
        
        if self.use_markers:
            style['marker'] = self.MARKERS[index % len(self.MARKERS)]
        
        return style
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all styles."""
        for i in range(self.n):
            yield self.get_style(i)
    
    def __len__(self) -> int:
        return self.n


# =============================================================================
# Smart Error Bar Plotting
# =============================================================================

def plot_with_error(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray] = None,
    style: str = 'auto',
    config: Optional[ErrorBarConfig] = None,
    **kwargs,
) -> None:
    """
    Plot data with smart error bar handling.
    
    Args:
        ax: Matplotlib axes
        x: X data
        y: Y data (mean values)
        yerr: Error values (std, sem, or CI half-width)
        style: Error style - 'bar', 'band', 'auto', or 'none'
        config: Error bar configuration
        **kwargs: Additional arguments passed to plot/errorbar
    """
    cfg = config or ERROR_CONFIG
    
    # Check if we should show error bars
    if yerr is None or not cfg.show:
        ax.plot(x, y, **kwargs)
        return
    
    yerr = np.asarray(yerr)
    if not np.any(yerr > 0):
        ax.plot(x, y, **kwargs)
        return
    
    # Determine style
    if style == 'auto':
        style = 'band' if len(x) > cfg.auto_threshold else 'bar'
    
    if style == 'band':
        # Fill between style
        color = kwargs.get('color', None)
        label = kwargs.pop('label', None)
        
        line, = ax.plot(x, y, label=label, **kwargs)
        if color is None:
            color = line.get_color()
        
        ax.fill_between(
            x,
            np.asarray(y) - yerr,
            np.asarray(y) + yerr,
            color=color,
            alpha=cfg.band_alpha,
            linewidth=0,
        )
    else:
        # Error bar style
        ax.errorbar(
            x, y,
            yerr=yerr,
            capsize=cfg.capsize,
            capthick=cfg.capthick,
            elinewidth=cfg.elinewidth,
            **kwargs,
        )


# =============================================================================
# Legend Utilities
# =============================================================================

def auto_legend(
    ax: Axes,
    n_items: Optional[int] = None,
    prefer_outside: bool = False,
    **kwargs,
) -> None:
    """
    Automatically position legend based on number of items.
    
    Args:
        ax: Matplotlib axes
        n_items: Number of legend items (auto-detected if None)
        prefer_outside: If True, prefer outside placement
        **kwargs: Additional legend arguments
    """
    if n_items is None:
        n_items = len(ax.get_legend_handles_labels()[0])
    
    if n_items == 0:
        return
    
    defaults = {
        'framealpha': PUB_CONFIG.legend_framealpha,
        'edgecolor': PUB_CONFIG.legend_edgecolor,
    }
    defaults.update(kwargs)
    
    if prefer_outside or n_items > 10:
        # Place below the plot
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(4, n_items),
            **defaults,
        )
    elif n_items > 6:
        # Two columns inside
        ax.legend(loc='best', ncol=2, **defaults)
    else:
        # Single column inside
        ax.legend(loc='best', **defaults)


# =============================================================================
# Panel Labeling
# =============================================================================

def add_panel_label(
    ax: Axes,
    label: str,
    fontsize: Optional[int] = None,
    fontweight: str = 'bold',
    offset: Tuple[float, float] = (0.02, 0.98),
) -> None:
    """
    Add panel label (a), (b), (c) etc. to axes.
    
    Args:
        ax: Matplotlib axes
        label: Label text (e.g., 'a', 'b', '(a)', 'A')
        loc: Location hint (used only if offset is None)
        fontsize: Font size (uses config default if None)
        fontweight: Font weight ('bold' recommended)
        offset: (x, y) position in axes coordinates
    """
    fs = fontsize or PUB_CONFIG.font_size_panel_label
    
    ax.text(
        offset[0], offset[1],
        label,
        transform=ax.transAxes,
        fontsize=fs,
        fontweight=fontweight,
        verticalalignment='top',
        horizontalalignment='left',
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_figure(
    columns: str = 'single',
    aspect_ratio: Optional[float] = None,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
) -> Tuple[Figure, Any]:
    """
    Create publication-quality figure with proper sizing.
    
    Args:
        columns: 'single' or 'double' column width
        aspect_ratio: height/width ratio
        nrows, ncols: Subplot grid dimensions
        **kwargs: Additional arguments to plt.subplots
        
    Returns:
        (fig, axes) tuple
    """
    width, height = get_figure_size(columns, aspect_ratio)
    
    # Adjust height for multiple rows
    if nrows > 1:
        height = height * nrows * 0.8  # Slightly compress
    
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def setup_axis(
    ax: Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    grid: bool = True,
) -> None:
    """
    Configure axis with publication-quality settings.
    
    Args:
        ax: Matplotlib axes
        xlabel, ylabel: Axis labels
        title: Axes title
        xlim, ylim: Axis limits
        grid: Whether to show grid
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True, alpha=PUB_CONFIG.grid_alpha, linewidth=PUB_CONFIG.linewidth_grid)


# =============================================================================
# Backward Compatibility - Updated STYLE dict
# =============================================================================

def get_legacy_style() -> Dict[str, Any]:
    """
    Get STYLE dict compatible with existing code but using publication values.
    
    This allows existing code using STYLE['linewidth'] etc. to work
    with the new publication-quality values.
    """
    cfg = PUB_CONFIG
    return {
        'linewidth': cfg.linewidth_plot,
        'markersize': cfg.markersize,
        'marker': 'o',
        'capsize': ERROR_CONFIG.capsize,
        'fontsize': {
            'title': cfg.font_size_title,
            'label': cfg.font_size_axis_label,
            'tick': cfg.font_size_tick,
            'legend': cfg.font_size_legend,
        }
    }


# Legacy STYLE for backward compatibility
STYLE = get_legacy_style()
