"""MetricSpec adapter for legacy metric payloads.

The formulas still live in the existing metric modules and runner code. This
adapter is the hard-interface layer that checks the legacy flat dict against
MetricSpec and AlgorithmSpec before the payload enters results/output code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from matrix_factorization.core.contracts import (
    get_algorithm_metric_keys,
    get_algorithm_metric_semantics,
    get_algorithm_specs,
    get_metric_specs,
)


@dataclass(frozen=True)
class MetricPayloadCheck:
    algorithm_key: str
    declared_keys: List[str] = field(default_factory=list)
    produced_artifacts: List[str] = field(default_factory=list)
    metric_specs: List[str] = field(default_factory=list)
    semantic_keys: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    source: str = "unknown"
    unexpected_keys: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.unexpected_keys

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_key": self.algorithm_key,
            "source": self.source,
            "is_valid": self.is_valid,
            "declared_keys": list(self.declared_keys),
            "produced_artifacts": list(self.produced_artifacts),
            "metric_specs": list(self.metric_specs),
            "semantic_keys": dict(self.semantic_keys),
            "unexpected_keys": list(self.unexpected_keys),
        }


class MetricSpecAdapter:
    """Validate legacy metric dicts against hard metric contracts."""

    @staticmethod
    def check_payload(
        algorithm_key: str,
        metrics: Dict[str, Any],
        source: str = "unknown",
    ) -> MetricPayloadCheck:
        algorithm_specs = get_algorithm_specs()
        metric_specs = get_metric_specs()
        declared = set(get_algorithm_metric_keys(algorithm_key))
        produced_artifacts = []
        matched_metric_specs = []
        if algorithm_key in algorithm_specs:
            algorithm_spec = algorithm_specs[algorithm_key]
            produced_artifacts = list(algorithm_specs[algorithm_key].produced_artifacts)
            declared.update(produced_artifacts)
            for metric_key in algorithm_spec.produced_metrics:
                spec = metric_specs.get(metric_key)
                if spec and set(spec.produces).intersection(metrics):
                    matched_metric_specs.append(metric_key)
        declared.update({"overlap_matrix_metric"})
        unexpected = sorted(set(metrics) - declared)
        semantics = get_algorithm_metric_semantics(algorithm_key)
        semantic_keys = {
            key: semantics[key]
            for key in sorted(set(metrics).intersection(semantics))
        }
        return MetricPayloadCheck(
            algorithm_key=algorithm_key,
            declared_keys=sorted(declared),
            produced_artifacts=sorted(produced_artifacts),
            metric_specs=sorted(set(matched_metric_specs)),
            semantic_keys=semantic_keys,
            source=source,
            unexpected_keys=unexpected,
        )

    @staticmethod
    def validate_payload(
        algorithm_key: str,
        metrics: Dict[str, Any],
        source: str = "unknown",
    ) -> MetricPayloadCheck:
        check = MetricSpecAdapter.check_payload(algorithm_key, metrics, source=source)
        if check.unexpected_keys:
            raise ValueError(
                f"algorithm '{algorithm_key}' produced undeclared metric keys: "
                f"{check.unexpected_keys}. Add or update MetricSpec/"
                "AlgorithmSpec before using these keys in the main path."
            )
        return check

    @staticmethod
    def describe_payload(
        algorithm_key: str,
        metrics: Dict[str, Any],
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """Return machine-readable MetricSpec coverage for a flat payload."""
        return MetricSpecAdapter.validate_payload(
            algorithm_key,
            metrics,
            source=source,
        ).to_dict()
