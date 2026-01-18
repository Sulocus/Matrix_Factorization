"""
Publication-quality figure generation.

Generates figures suitable for academic papers with:
- Correct fonts (serif)
- Appropriate sizes for single/double column
- Vector formats (PDF, EPS, SVG)
- LaTeX-style labels
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

try:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class FigureStyle(Enum):
    """Figure styles for different publication contexts."""
    SINGLE_COLUMN = "single"   # ~3.5 inch width (Nature, Science)
    DOUBLE_COLUMN = "double"   # ~7 inch width (full page width)
    PRESENTATION = "slide"     # Larger for slides/posters


# Publication-ready matplotlib settings
PUBLICATION_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'pdf.fonttype': 42,  # TrueType fonts in PDF
    'ps.fonttype': 42,
    'text.usetex': False,  # Set True if LaTeX available
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
}

FIGURE_SIZES = {
    FigureStyle.SINGLE_COLUMN: (3.5, 2.8),
    FigureStyle.DOUBLE_COLUMN: (7.0, 4.5),
    FigureStyle.PRESENTATION: (10.0, 7.0),
}

# Standard colors matching Wang/ color scheme
COLORS = {
    'Q_Y': '#d62728',       # Red - reconstruction quality
    'Q_W': '#ff7f0e',       # Orange - left factor raw
    'Q_X': '#2ca02c',       # Green - right factor raw
    'Q_W_prime': '#9467bd', # Purple - left factor normalized
    'Q_X_prime': '#8c564b', # Brown - right factor normalized
}


class PublicationFigure:
    """
    Generate publication-ready figures.

    Usage:
        pub = PublicationFigure(style=FigureStyle.SINGLE_COLUMN)
        pub.create_phase_diagram(data, "output.pdf")
    """

    def __init__(self, style: FigureStyle = FigureStyle.SINGLE_COLUMN):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for publication figures")

        self.style = style
        self._apply_style()

    def _apply_style(self):
        """Apply publication style settings."""
        rcParams.update(PUBLICATION_RC)

    def create_phase_diagram(
        self,
        data: Dict[str, Any],
        output: Path,
        title: str = None,
        show_metrics: List[str] = None,
    ) -> Path:
        """
        Create a standard phase transition diagram.

        Args:
            data: Results dictionary with {alpha: metrics}
            output: Output path
            title: Optional title
            show_metrics: List of metrics to show (default: ["Q_Y"])

        Returns:
            Path to saved figure
        """
        if show_metrics is None:
            show_metrics = ["Q_Y"]

        fig, ax = plt.subplots(figsize=FIGURE_SIZES[self.style])

        results = data.get('results', data)
        alphas = sorted([float(a) for a in results.keys()])

        for metric in show_metrics:
            values = [results[str(a)].get(f'{metric}_mean', 0) for a in alphas]
            color = COLORS.get(metric, '#1f77b4')

            # Use LaTeX-style labels
            label = rf'$Q_{{{metric.split("_")[1]}}}$' if '_' in metric else metric
            ax.plot(alphas, values, 'o-', color=color, label=label, markersize=3)

        ax.set_xlabel(r'$\alpha$ (observation density)')
        ax.set_ylabel('Overlap')
        if title:
            ax.set_title(title)
        ax.legend(frameon=False)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        fig.tight_layout()
        output = Path(output)
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)

        return output

    def create_size_scaling(
        self,
        datasets: List[Dict[str, Any]],
        labels: List[str],
        output: Path,
        metric: str = "Q_Y",
    ) -> Path:
        """
        Create figure showing size scaling (finite size effects).

        Args:
            datasets: List of result dictionaries
            labels: Labels for each dataset (e.g., "N=100", "N=500")
            output: Output path
            metric: Metric to plot

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZES[self.style])

        colors = plt.cm.viridis(np.linspace(0, 0.8, len(datasets)))

        for i, data in enumerate(datasets):
            results = data.get('results', data)
            alphas = sorted([float(a) for a in results.keys()])
            values = [results[str(a)].get(f'{metric}_mean', 0) for a in alphas]

            ax.plot(alphas, values, 'o-', color=colors[i],
                    label=labels[i], markersize=2, linewidth=1)

        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(rf'$Q_{{{metric.split("_")[1]}}}$' if '_' in metric else metric)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        fig.tight_layout()
        output = Path(output)
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)

        return output

    def create_algorithm_comparison(
        self,
        bigamp_data: Dict,
        agd_data: Dict,
        output: Path,
        metric: str = "Q_Y",
    ) -> Path:
        """
        Create figure comparing BiG-AMP vs AGD.

        Args:
            bigamp_data: BiG-AMP results
            agd_data: AGD results
            output: Output path
            metric: Metric to compare

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZES[self.style])

        for data, name, color, marker in [
            (bigamp_data, 'BiG-AMP', '#d62728', 'o'),
            (agd_data, 'AGD', '#1f77b4', 's'),
        ]:
            results = data.get('results', data)
            alphas = sorted([float(a) for a in results.keys()])
            values = [results[str(a)].get(f'{metric}_mean', 0) for a in alphas]

            ax.plot(alphas, values, marker=marker, color=color,
                    label=name, markersize=3, linewidth=1)

        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(rf'$Q_{{{metric.split("_")[1]}}}$')
        ax.legend(frameon=False)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        fig.tight_layout()
        output = Path(output)
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)

        return output


def set_publication_style():
    """Set matplotlib to publication style globally."""
    if HAS_MATPLOTLIB:
        rcParams.update(PUBLICATION_RC)


def export_for_publication(
    data: Dict,
    output_dir: Path,
    prefix: str = "fig",
    formats: List[str] = None,
) -> List[Path]:
    """
    Export figures in multiple formats for publication.

    Args:
        data: Results data
        output_dir: Output directory
        prefix: Filename prefix
        formats: List of formats (default: ['pdf', 'svg', 'png'])

    Returns:
        List of output paths
    """
    if formats is None:
        formats = ['pdf', 'svg', 'png']

    pub = PublicationFigure(FigureStyle.SINGLE_COLUMN)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for fmt in formats:
        output = output_dir / f"{prefix}_phase_diagram.{fmt}"
        pub.create_phase_diagram(data, output)
        outputs.append(output)

    return outputs
