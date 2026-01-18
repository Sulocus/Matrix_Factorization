"""
Multi-experiment comparison tools.

Compare multiple experiments on the same plot for:
- Algorithm comparison (BiG-AMP vs AGD)
- Size scaling studies
- Parameter sensitivity analysis
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..core.results_db import ResultsDatabase, ExperimentRecord


class ExperimentComparer:
    """
    Compare multiple experiments.

    Usage:
        comparer = ExperimentComparer()
        comparer.add_by_id("exp1")
        comparer.add_by_id("exp2")

        fig = comparer.plot_comparison(metric="Q_Y")
        comparer.save_plot("comparison.pdf")
    """

    def __init__(self):
        self.db = ResultsDatabase()
        self.experiments: List[ExperimentRecord] = []
        self.data: Dict[str, Dict] = {}  # exp_id -> loaded data

    def add_by_id(self, exp_id: str) -> bool:
        """
        Add an experiment by ID.

        Args:
            exp_id: Experiment ID

        Returns:
            True if added successfully
        """
        record = self.db.get(exp_id)
        if record:
            self.experiments.append(record)
            self._load_data(record)
            return True
        return False

    def add_by_path(self, result_path: str) -> bool:
        """
        Add an experiment by result path.

        Args:
            result_path: Path to results directory

        Returns:
            True if added successfully
        """
        path = Path(result_path)
        metrics_file = path / "metrics.json"

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)

            # Create a temporary record
            record = ExperimentRecord(
                id=path.name,
                timestamp="",
                algorithm=data.get('algorithm', 'unknown'),
                graph=data.get('graph', 'unknown'),
                N1=data.get('N1', 0),
                N2=data.get('N2', 0),
                M=data.get('M', 0),
                alpha_start=0.0,
                alpha_stop=0.0,
                max_steps=0,
                result_path=str(path),
            )

            self.experiments.append(record)
            self.data[record.id] = data
            return True
        return False

    def _load_data(self, record: ExperimentRecord):
        """Load experiment data from result path."""
        path = Path(record.result_path)
        metrics_file = path / "metrics.json"

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.data[record.id] = json.load(f)

    def get_labels(self) -> List[str]:
        """Generate labels for each experiment."""
        labels = []
        for exp in self.experiments:
            label = f"{exp.algorithm} N={exp.N1} M={exp.M}"
            labels.append(label)
        return labels

    def plot_comparison(
        self,
        metric: str = "Q_Y",
        output: Path = None,
        title: str = None,
        figsize: tuple = (10, 6),
    ):
        """
        Plot comparison of all experiments.

        Args:
            metric: Metric to compare (e.g., "Q_Y", "Q_W")
            output: Optional output path
            title: Optional title
            figsize: Figure size

        Returns:
            matplotlib figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting")

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h']

        for i, exp in enumerate(self.experiments):
            data = self.data.get(exp.id, {})
            results = data.get('results', {})

            if not results:
                continue

            # Extract alpha and metric values
            alphas = sorted([float(a) for a in results.keys()])
            values = [results[str(a)].get(f'{metric}_mean', 0) for a in alphas]

            label = f"{exp.algorithm} N={exp.N1} M={exp.M}"
            ax.plot(
                alphas, values,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=label,
                markersize=4,
                linewidth=1,
            )

        ax.set_xlabel(r'$\alpha$ (observation density)')
        ax.set_ylabel(metric)
        ax.set_title(title or f'{metric} Comparison')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)

        if output:
            fig.savefig(output, dpi=150, bbox_inches='tight')

        return fig

    def generate_summary_table(self) -> str:
        """
        Generate markdown comparison table.

        Returns:
            Markdown formatted table
        """
        lines = [
            "| ID | Algorithm | N | M | α range | max Q_Y |",
            "|-----|-----------|-----|-----|---------|---------|"
        ]

        for exp in self.experiments:
            data = self.data.get(exp.id, {})
            results = data.get('results', {})

            # Find max Q_Y
            max_qy = 0.0
            if results:
                for metrics in results.values():
                    qy = metrics.get('Q_Y_mean', 0)
                    if qy > max_qy:
                        max_qy = qy

            lines.append(
                f"| {exp.id[:8]} | {exp.algorithm} | {exp.N1} | {exp.M} | "
                f"{exp.alpha_start:.1f}-{exp.alpha_stop:.1f} | {max_qy:.4f} |"
            )

        return '\n'.join(lines)

    def find_phase_transitions(self) -> Dict[str, float]:
        """
        Estimate phase transition points for each experiment.

        Returns:
            Dictionary of {exp_id: alpha_c}
        """
        transitions = {}

        for exp in self.experiments:
            data = self.data.get(exp.id, {})
            results = data.get('results', {})

            if not results:
                continue

            # Find alpha where Q_Y first exceeds 0.9
            alphas = sorted([float(a) for a in results.keys()])
            for alpha in alphas:
                qy = results[str(alpha)].get('Q_Y_mean', 0)
                if qy > 0.9:
                    transitions[exp.id] = alpha
                    break

        return transitions

    def save_comparison_report(self, output: Path):
        """
        Save a full comparison report.

        Args:
            output: Output file path (markdown)
        """
        report = ["# Experiment Comparison Report\n"]
        report.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n")

        report.append("## Summary\n")
        report.append(self.generate_summary_table())
        report.append("\n")

        report.append("## Phase Transitions\n")
        transitions = self.find_phase_transitions()
        for exp_id, alpha_c in transitions.items():
            report.append(f"- {exp_id[:8]}: α_c ≈ {alpha_c:.2f}\n")

        with open(output, 'w') as f:
            f.write('\n'.join(report))
