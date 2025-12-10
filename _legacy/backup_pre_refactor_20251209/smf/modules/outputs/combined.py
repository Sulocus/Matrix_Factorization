"""
Combined output handler for flexible configuration.

Supports configuring both plotting and storage:
- plot_style: Color scheme, line styles, figure size
- storage_format: JSON, NPZ, or both
- metrics_to_plot: Which metrics to display

This module enables LLM-driven configuration to customize outputs
based on natural language descriptions.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import numpy as np

from ..registry import register_output
from .base import OutputBase
from .plotting import ResultPlotter, COLORS, STYLE
from .storage import ResultStorage
from ...core.config import Config


# Default metric groups
METRIC_GROUPS = {
    "standard": ["Q_Y", "Q_W_prime", "Q_X_prime"],
    "full": ["Q_Y", "Q_W", "Q_X", "Q_W_prime", "Q_X_prime", "Gen_Error"],
    "minimal": ["Q_Y"],
    "raw": ["Q_W", "Q_X"],
    "normalized": ["Q_W_prime", "Q_X_prime"],
}


@register_output(
    key="combined",
    name="Combined Output",
    description="Flexible plotting and storage configuration",
    default_params={
        "metrics_to_plot": ["Q_Y", "Q_W_prime", "Q_X_prime"],
        "storage_format": "json",
        "show_error_bars": True,
    },
)
class CombinedOutput(OutputBase):
    """
    Flexible output handler combining plotting and storage.

    Example configurations:
    - {"metrics_to_plot": "full"} -> Show all metrics
    - {"storage_format": "both"} -> Save as JSON and NPZ
    - {"show_error_bars": False} -> Clean plots without error bars

    This is the recommended approach for LLM-driven output configuration.
    """

    def __init__(
        self,
        config: Config,
        output_dir: Path,
        metrics_to_plot: Optional[List[str]] = None,
        storage_format: str = "json",
        show_error_bars: bool = True,
        figure_size: tuple = (10, 6),
        dpi: int = 150,
        title_suffix: str = "",
    ):
        """
        Initialize combined output handler.

        Args:
            config: Experiment configuration
            output_dir: Directory for output files
            metrics_to_plot: List of metrics to show in plots
            storage_format: "json", "npz", or "both"
            show_error_bars: Whether to show error bars
            figure_size: Figure size in inches
            dpi: DPI for saved figures
            title_suffix: Additional text for plot titles
        """
        super().__init__(config, output_dir)

        # Resolve metric group names
        if metrics_to_plot is None:
            metrics_to_plot = METRIC_GROUPS["standard"]
        elif isinstance(metrics_to_plot, str):
            metrics_to_plot = METRIC_GROUPS.get(metrics_to_plot, [metrics_to_plot])

        self.metrics_to_plot = list(metrics_to_plot)
        self.storage_format = storage_format.lower()
        self.show_error_bars = show_error_bars
        self.figure_size = figure_size
        self.dpi = dpi
        self.title_suffix = title_suffix

        # Create sub-handlers
        self._plotter = ResultPlotter(config, output_dir)
        self._storage = ResultStorage(config, output_dir)

    def save(self, results: Dict[str, Any], **kwargs) -> Dict[str, Path]:
        """
        Save results using configured format.

        Args:
            results: Experiment results dictionary
            **kwargs: Additional arguments passed to handlers

        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}

        # Save data files
        if self.storage_format in ["json", "both"]:
            json_path = self._storage.save_json(results)
            saved_files["json"] = json_path

        if self.storage_format in ["npz", "both"]:
            npz_path = self._storage.save_npz(results)
            saved_files["npz"] = npz_path

        # Save plot
        plot_path = self.plot_summary(results, **kwargs)
        saved_files["plot"] = plot_path

        return saved_files

    def plot_summary(
        self,
        results: Dict[str, Any],
        filename: str = None,
        **kwargs,
    ) -> Path:
        """
        Create summary plot with configured metrics.

        Args:
            results: Experiment results
            filename: Output filename (auto-generated if None)
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Extract alpha values and metrics
        alpha_values = []
        metric_data = {m: {"mean": [], "std": []} for m in self.metrics_to_plot}

        for alpha_str, data in sorted(results.items()):
            try:
                alpha = float(alpha_str)
            except (ValueError, TypeError):
                continue

            alpha_values.append(alpha)

            for metric in self.metrics_to_plot:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"

                if mean_key in data:
                    metric_data[metric]["mean"].append(data[mean_key])
                    metric_data[metric]["std"].append(data.get(std_key, 0))

        # Plot each metric
        for metric in self.metrics_to_plot:
            if not metric_data[metric]["mean"]:
                continue

            color = COLORS.get(metric, "#333333")
            means = metric_data[metric]["mean"]
            stds = metric_data[metric]["std"]

            if self.show_error_bars and any(s > 0 for s in stds):
                ax.errorbar(
                    alpha_values, means,
                    yerr=stds,
                    label=metric,
                    color=color,
                    linewidth=STYLE["linewidth"],
                    marker=STYLE["marker"],
                    markersize=STYLE["markersize"],
                    capsize=STYLE["capsize"],
                )
            else:
                ax.plot(
                    alpha_values, means,
                    label=metric,
                    color=color,
                    linewidth=STYLE["linewidth"],
                    marker=STYLE["marker"],
                    markersize=STYLE["markersize"],
                )

        # Styling
        ax.set_xlabel(r"$\tilde{\alpha}$", fontsize=STYLE["fontsize"]["label"])
        ax.set_ylabel("Overlap", fontsize=STYLE["fontsize"]["label"])

        title = f"Overlap vs Observation Density"
        if self.title_suffix:
            title += f" ({self.title_suffix})"
        ax.set_title(title, fontsize=STYLE["fontsize"]["title"])

        ax.legend(fontsize=STYLE["fontsize"]["legend"])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)

        # Save
        if filename is None:
            filename = f"summary_{self.config.experiment_name}.png"

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return output_path

    @classmethod
    def from_config(cls, config: Config, output_dir: Path, params: Dict[str, Any]) -> "CombinedOutput":
        """
        Create output handler from configuration dictionary.

        Args:
            config: Experiment configuration
            output_dir: Output directory
            params: Output parameters

        Returns:
            CombinedOutput instance
        """
        return cls(
            config=config,
            output_dir=output_dir,
            metrics_to_plot=params.get("metrics_to_plot"),
            storage_format=params.get("storage_format", "json"),
            show_error_bars=params.get("show_error_bars", True),
            figure_size=params.get("figure_size", (10, 6)),
            dpi=params.get("dpi", 150),
            title_suffix=params.get("title_suffix", ""),
        )

    @classmethod
    def from_natural_language(cls, description: str, config: Config, output_dir: Path) -> "CombinedOutput":
        """
        Parse natural language description to configure outputs.

        Supports:
        - "show all metrics"
        - "only Q_Y"
        - "save as npz"
        - "no error bars"
        - "high resolution plot"
        - Chinese: "显示所有指标", "保存为JSON"

        Args:
            description: Natural language description
            config: Experiment configuration
            output_dir: Output directory

        Returns:
            CombinedOutput instance
        """
        desc_lower = description.lower()
        metrics_to_plot = None
        storage_format = "json"
        show_error_bars = True
        dpi = 150

        # Metric selection
        if "all" in desc_lower or "所有" in description or "full" in desc_lower:
            metrics_to_plot = "full"
        elif "minimal" in desc_lower or "only q_y" in desc_lower or "只" in description:
            metrics_to_plot = "minimal"
        elif "raw" in desc_lower or "原始" in description:
            metrics_to_plot = "raw"
        elif "normalized" in desc_lower or "归一化" in description:
            metrics_to_plot = "normalized"

        # Storage format
        if "npz" in desc_lower:
            storage_format = "npz"
        if "both" in desc_lower or "两种" in description:
            storage_format = "both"

        # Error bars
        if "no error" in desc_lower or "clean" in desc_lower or "无误差" in description:
            show_error_bars = False

        # Resolution
        if "high res" in desc_lower or "高分辨率" in description:
            dpi = 300
        elif "low res" in desc_lower:
            dpi = 100

        return cls(
            config=config,
            output_dir=output_dir,
            metrics_to_plot=metrics_to_plot,
            storage_format=storage_format,
            show_error_bars=show_error_bars,
            dpi=dpi,
        )

    def __repr__(self) -> str:
        return f"CombinedOutput(metrics={self.metrics_to_plot}, format={self.storage_format})"
