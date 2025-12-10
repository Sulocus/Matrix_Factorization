"""
Combined metrics calculator for flexible configuration.

Supports selecting which metrics to compute:
- Q_Y: Y-space overlap (rotationally invariant)
- Q_W, Q_X: Factor overlaps (raw cosine)
- Q_W', Q_X': Normalized factor overlaps
- Q_Y_unobserved: Q_Y on unobserved positions only
- Gen_Error: Generalization error (MSE)
- Replica: Pairwise replica overlaps

This module enables LLM-driven configuration to select metrics
based on natural language descriptions.
"""

from typing import Dict, Any, List, Optional, Set
import torch

from .overlap import (
    compute_cosine_similarity,
    gram_overlap_normalized,
    compute_qy,
    compute_generalization_error,
    compute_all_metrics,
    compute_physical_overlap,
)
from .qy_unobserved import compute_qy_unobserved, compute_qy_split


# Available metric keys
ALL_METRICS = {
    "Q_Y",           # Y-space overlap
    "Q_W",           # W Gram cosine overlap
    "Q_X",           # X Gram cosine overlap
    "Q_W_prime",     # W Gram normalized overlap
    "Q_X_prime",     # X Gram normalized overlap
    "Q_Y_unobserved", # Q_Y on unobserved positions
    "Q_Y_observed",  # Q_Y on observed positions
    "Gen_Error",     # Generalization error (MSE)
    "physical_overlap_Y", # Physical overlap Y
    "physical_overlap_W", # Physical overlap W
    "physical_overlap_X", # Physical overlap X
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
    "qw_prime": "Q_W_prime",
    "qw'": "Q_W_prime",
    "qx_prime": "Q_X_prime",
    "qx'": "Q_X_prime",
    "normalized": {"Q_W_prime", "Q_X_prime"},
    "qy_unobs": "Q_Y_unobserved",
    "unobserved": "Q_Y_unobserved",
    "generalization": "Gen_Error",
    "mse": "Gen_Error",
    "error": "Gen_Error",
}


class CombinedMetrics:
    """
    Flexible metrics calculator that computes selected metrics.

    Example configurations:
    - {"metrics": ["Q_Y", "Q_W_prime", "Q_X_prime"]} -> Standard metrics
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
            self.metrics = {"Q_Y", "Q_W_prime", "Q_X_prime", "Gen_Error"}
        elif metrics == "all" or (isinstance(metrics, list) and "all" in metrics):
            self.metrics = ALL_METRICS.copy()
        else:
            self.metrics = set(metrics)

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
            results["Q_Y"] = compute_qy(Y_student, Y_teacher)

        if "Q_W" in self.metrics:
            results["Q_W"] = compute_cosine_similarity(W_student, W_teacher, use_left=True)

        if "Q_X" in self.metrics:
            results["Q_X"] = compute_cosine_similarity(X_student, X_teacher, use_left=False)

        if "Q_W_prime" in self.metrics:
            results["Q_W_prime"] = gram_overlap_normalized(W_student, W_teacher, use_left=True)

        if "Q_X_prime" in self.metrics:
            results["Q_X_prime"] = gram_overlap_normalized(X_student, X_teacher, use_left=False)

        if "Gen_Error" in self.metrics:
            results["Gen_Error"] = compute_generalization_error(Y_student, Y_teacher)

        # Unobserved metrics require mask
        if mask is not None:
            if "Q_Y_unobserved" in self.metrics:
                results["Q_Y_unobserved"] = compute_qy_unobserved(Y_student, Y_teacher, mask)

            if "Q_Y_observed" in self.metrics:
                split = compute_qy_split(Y_student, Y_teacher, mask)
                results["Q_Y_observed"] = split["Q_Y_observed"]

        # Physical overlaps
        if "physical_overlap_Y" in self.metrics:
            results["physical_overlap_Y"] = compute_physical_overlap(Y_student, Y_teacher, absolute=False)

        if "physical_overlap_W" in self.metrics:
            results["physical_overlap_W"] = compute_physical_overlap(W_student, W_teacher, absolute=True)

        if "physical_overlap_X" in self.metrics:
            results["physical_overlap_X"] = compute_physical_overlap(X_student, X_teacher, absolute=True)

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
