"""
Combined metrics calculator for flexible configuration.

Supports selecting which metrics to compute:
- Q_Y: measurement FIT=1-NMSE
- Q_W, Q_X: fixed-denominator physical overlaps
- Q_W_SIGN_GAUGE, Q_X_SIGN_GAUGE: per-channel sign-gauge diagnostics
- Q_W_COS_ROOT, Q_X_COS_ROOT: Cos-root diagnostics
- Q_Y_unobserved: Q_Y on unobserved positions only
- Replica: Pairwise replica overlaps

This module enables LLM-driven configuration to select metrics
based on natural language descriptions.
"""

from typing import Dict, Any, List, Optional
import torch

from .overlap import (
    cos_overlap_root,
    normalized_mse_and_fit,
    physical_overlap_fixed,
    projection_abs,
    scale_gauge_overlaps_fixed,
    sign_gauge_overlap_fixed,
    _compute_nmse_fit_masked,
)


# Available metric keys
ALL_METRICS = {
    "Q_Y",           # measurement projection
    "Q_W",           # W coordinate projection
    "Q_X",           # X coordinate projection
    "Q_W_SIGN_GAUGE", # W sign-gauge diagnostic
    "Q_X_SIGN_GAUGE", # X sign-gauge diagnostic
    "Q_W_SIGN_ALIGNED", # legacy alias
    "Q_X_SIGN_ALIGNED", # legacy alias
    "Q_W_COS_ROOT", # W Cos-root diagnostic
    "Q_X_COS_ROOT", # X Cos-root diagnostic
    "Q_W_SCALE_GAUGE",
    "Q_X_SCALE_GAUGE",
    "Q_WX_SCALE_GAUGE",
    "Q_Y_unobserved", # Q_Y on unobserved positions
    "Q_Y_observed",  # Q_Y on observed positions
}

# Metric aliases for natural language parsing
METRIC_ALIASES = {
    "qy": "Q_Y",
    "q_y": "Q_Y",
    "reconstruction": "Q_Y",
    "qw": "Q_W",
    "q_w": "Q_W",
    "qx": "Q_X",
    "q_x": "Q_X",
    "qw_sign": "Q_W_SIGN_ALIGNED",
    "qx_sign": "Q_X_SIGN_ALIGNED",
    "qw_sign_gauge": "Q_W_SIGN_GAUGE",
    "qx_sign_gauge": "Q_X_SIGN_GAUGE",
    "sign_gauge": {"Q_W_SIGN_GAUGE", "Q_X_SIGN_GAUGE"},
    "sign_aligned": {"Q_W_SIGN_GAUGE", "Q_X_SIGN_GAUGE"},
    "qw_cos_root": "Q_W_COS_ROOT",
    "qx_cos_root": "Q_X_COS_ROOT",
    "qw_gram_root": "Q_W_COS_ROOT",
    "qx_gram_root": "Q_X_COS_ROOT",
    "cos_root": {"Q_W_COS_ROOT", "Q_X_COS_ROOT"},
    "gram_root": {"Q_W_COS_ROOT", "Q_X_COS_ROOT"},
    "scale_gauge": {"Q_W_SCALE_GAUGE", "Q_X_SCALE_GAUGE", "Q_WX_SCALE_GAUGE"},
    "qy_unobs": "Q_Y_unobserved",
    "unobserved": "Q_Y_unobserved",
}

LEGACY_METRIC_ALIASES = {
    "Q_W_GRAM_ROOT": "Q_W_COS_ROOT",
    "Q_X_GRAM_ROOT": "Q_X_COS_ROOT",
    "Q_W_SIGN_ALIGNED": "Q_W_SIGN_GAUGE",
    "Q_X_SIGN_ALIGNED": "Q_X_SIGN_GAUGE",
}


def _normalize_metric_name(metric: str) -> str:
    return LEGACY_METRIC_ALIASES.get(metric, metric)


class CombinedMetrics:
    """
    Flexible metrics calculator that computes selected metrics.

    Example configurations:
    - {"metrics": ["Q_Y", "Q_W_COS_ROOT", "Q_X_COS_ROOT"]} -> Standard metrics
    - {"metrics": ["Q_Y", "Q_Y_unobserved"]} -> Compare observed vs unobserved
    - {"metrics": "all"} -> All available metrics

    This is the recommended approach for LLM-driven metric selection.
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        include_unobserved: bool = False,
    ):
        """
        Initialize combined metrics calculator.

        Args:
            metrics: List of metric keys to compute. If None, computes standard set.
            include_unobserved: Whether to include Q_Y_unobserved (requires mask)
        """
        if metrics is None:
            # Default standard metrics
            self.metrics = {"Q_Y", "Q_W_COS_ROOT", "Q_X_COS_ROOT"}
        elif metrics == "all" or (isinstance(metrics, list) and "all" in metrics):
            self.metrics = ALL_METRICS.copy()
        else:
            self.metrics = {_normalize_metric_name(metric) for metric in metrics}

        if include_unobserved:
            self.metrics.add("Q_Y_unobserved")

    @torch.no_grad()
    def compute(
        self,
        W_student: torch.Tensor,
        X_student: torch.Tensor,
        W_teacher: torch.Tensor,
        X_teacher: torch.Tensor,
        Y_teacher: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Compute selected metrics.

        Args:
            W_student: Student W matrix (N1, M)
            X_student: Student X matrix (M, N2)
            W_teacher: Teacher W matrix (N1, M)
            X_teacher: Teacher X matrix (M, N2)
            Y_teacher: Pre-computed teacher Y (optional)
            mask: Observation mask for unobserved metrics (optional)

        Returns:
            Dictionary with computed metric values
        """
        if Y_teacher is None:
            Y_teacher = W_teacher @ X_teacher

        Y_student = W_student @ X_student

        results = {}

        # Compute requested metrics
        if "Q_Y" in self.metrics:
            nmse, fit = normalized_mse_and_fit(Y_student, Y_teacher)
            results["NMSE_Y"] = nmse
            results["Q_Y"] = fit

        if "Q_Y_PROJ_ABS" in self.metrics:
            results["Q_Y_PROJ_ABS"] = projection_abs(Y_student, Y_teacher)

        if "Q_W" in self.metrics:
            results["Q_W"] = physical_overlap_fixed(W_student, W_teacher)

        if "Q_X" in self.metrics:
            results["Q_X"] = physical_overlap_fixed(X_student, X_teacher)

        if "Q_W_PROJ_ABS" in self.metrics:
            results["Q_W_PROJ_ABS"] = projection_abs(W_student, W_teacher)

        if "Q_X_PROJ_ABS" in self.metrics:
            results["Q_X_PROJ_ABS"] = projection_abs(X_student, X_teacher)

        if "Q_W_SIGN_GAUGE" in self.metrics:
            results["Q_W_SIGN_GAUGE"] = sign_gauge_overlap_fixed(
                W_student,
                W_teacher,
                latent_axis=-1,
            )

        if "Q_X_SIGN_GAUGE" in self.metrics:
            results["Q_X_SIGN_GAUGE"] = sign_gauge_overlap_fixed(
                X_student,
                X_teacher,
                latent_axis=0,
            )

        if {"Q_W_SCALE_GAUGE", "Q_X_SCALE_GAUGE", "Q_WX_SCALE_GAUGE"} & self.metrics:
            results.update(scale_gauge_overlaps_fixed(W_student, X_student, W_teacher, X_teacher))

        if "Q_W_COS_ROOT" in self.metrics:
            results["Q_W_COS_ROOT"] = cos_overlap_root(W_student, W_teacher, use_left=True)

        if "Q_X_COS_ROOT" in self.metrics:
            results["Q_X_COS_ROOT"] = cos_overlap_root(X_student, X_teacher, use_left=False)

        # Unobserved metrics require mask
        if mask is not None:
            if "Q_Y_unobserved" in self.metrics:
                nmse, fit = _compute_nmse_fit_masked(Y_student, Y_teacher, mask, observed=False)
                results["NMSE_Y_unobserved"] = nmse
                results["Q_Y_unobserved"] = fit

            if "Q_Y_observed" in self.metrics:
                nmse, fit = _compute_nmse_fit_masked(Y_student, Y_teacher, mask, observed=True)
                results["NMSE_Y_observed"] = nmse
                results["Q_Y_observed"] = fit

        return results

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombinedMetrics":
        """
        Create metrics calculator from configuration dictionary.

        Args:
            config: Dictionary with keys:
                - metrics: List[str] or "all"
                - include_unobserved: bool

        Returns:
            CombinedMetrics instance
        """
        return cls(
            metrics=config.get("metrics"),
            include_unobserved=config.get("include_unobserved", False),
        )

    @classmethod
    def from_natural_language(cls, description: str) -> "CombinedMetrics":
        """
        Parse natural language description to select metrics.

        Supports:
        - "compute Q_Y and generalization error"
        - "show Q_W' and Q_X'"
        - "include unobserved Q_Y"
        - "all metrics"
        - Chinese: "计算重建质量", "显示未观测Q_Y"

        Args:
            description: Natural language description

        Returns:
            CombinedMetrics instance
        """
        desc_lower = description.lower()
        metrics = set()

        # Check for "all"
        if "all" in desc_lower or "所有" in description:
            return cls(metrics="all")

        # Parse individual metrics
        for alias, target in METRIC_ALIASES.items():
            if alias in desc_lower:
                if isinstance(target, set):
                    metrics.update(target)
                else:
                    metrics.add(target)

        # Direct metric name matching
        for metric in ALL_METRICS:
            if metric.lower() in desc_lower or metric in description:
                metrics.add(metric)

        # Check for unobserved
        include_unobserved = any(kw in desc_lower for kw in [
            "unobserved", "unobs", "未观测", "holdout", "held-out"
        ])

        if not metrics:
            # Default to standard metrics if nothing specified
            metrics = None

        return cls(metrics=list(metrics) if metrics else None,
                   include_unobserved=include_unobserved)

    def __repr__(self) -> str:
        return f"CombinedMetrics({sorted(self.metrics)})"
