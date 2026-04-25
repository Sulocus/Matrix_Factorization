"""OutputSpec adapter for concrete result payloads.

The plotting/export code still lives in the existing output modules. This
adapter is the hard-interface layer that checks requested outputs against the
metrics and artifacts actually present in an ExperimentResult before plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matrix_factorization.core.experiment.result import ExperimentResult


@dataclass(frozen=True)
class OutputPayloadCheck:
    requested_specs: List[str] = field(default_factory=list)
    required_metrics: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)
    available_metrics: List[str] = field(default_factory=list)
    available_artifacts: List[str] = field(default_factory=list)
    missing_metrics: List[str] = field(default_factory=list)
    missing_artifacts: List[str] = field(default_factory=list)
    metric_semantics: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    plot_semantics: Dict[str, Any] = field(default_factory=dict)
    artifact_semantics: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.missing_metrics and not self.missing_artifacts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "requested_specs": list(self.requested_specs),
            "required_metrics": list(self.required_metrics),
            "required_artifacts": list(self.required_artifacts),
            "available_metrics": list(self.available_metrics),
            "available_artifacts": list(self.available_artifacts),
            "missing_metrics": list(self.missing_metrics),
            "missing_artifacts": list(self.missing_artifacts),
            "metric_semantics": dict(self.metric_semantics),
            "plot_semantics": dict(self.plot_semantics),
            "artifact_semantics": dict(self.artifact_semantics),
        }


class OutputSpecAdapter:
    """Validate concrete result payloads against OutputPlan requirements."""

    @staticmethod
    def check_result(
        result: "ExperimentResult",
        output_options: Optional[Dict[str, Any]] = None,
        run_dir: Optional[Path] = None,
    ) -> OutputPayloadCheck:
        output_options = dict(output_options or {})
        metric_keys = _result_metric_keys(result)
        artifacts = _result_artifacts(result, Path(run_dir) if run_dir is not None else None)
        output_plan = _contract_output_plan(result)

        required_metrics = set(output_plan.get("required_metrics") or [])
        required_artifacts = set(output_plan.get("required_artifacts") or [])
        requested_specs = list(output_plan.get("specs") or [])
        metric_semantics = output_plan.get("metric_semantics") or {}
        plot_semantics = output_plan.get("plot_semantics") or {}
        artifact_semantics = output_plan.get("artifact_semantics") or {}

        # Fallback for direct ExperimentResult.save() calls without preflight
        # metadata. This preserves legacy behavior while enforcing the same
        # concrete dependency checks.
        if output_options.get("plots"):
            for plot_config in output_options.get("plots") or []:
                for curve_code in plot_config.get("curves", []):
                    required_metrics.add(_curve_code_to_metric_key(curve_code))
            if "custom_curves" not in requested_specs:
                requested_specs.append("custom_curves")

        if output_options.get("enable_heatmap", True) and not output_plan:
            if "overlap_matrix" in metric_keys:
                required_artifacts.add("overlap_matrix")
            elif "matrix_factors" in artifacts:
                required_artifacts.add("matrix_factors")
            else:
                required_artifacts.add("overlap_matrix|matrix_factors")

        missing_metrics = sorted(required_metrics - metric_keys)
        missing_artifacts = sorted(
            artifact for artifact in required_artifacts
            if not _artifact_requirement_satisfied(artifact, artifacts)
        )
        return OutputPayloadCheck(
            requested_specs=sorted(set(requested_specs)),
            required_metrics=sorted(required_metrics),
            required_artifacts=sorted(required_artifacts),
            available_metrics=sorted(metric_keys),
            available_artifacts=sorted(artifacts),
            missing_metrics=missing_metrics,
            missing_artifacts=missing_artifacts,
            metric_semantics=metric_semantics,
            plot_semantics=plot_semantics,
            artifact_semantics=artifact_semantics,
        )

    @staticmethod
    def validate_result(
        result: "ExperimentResult",
        output_options: Optional[Dict[str, Any]] = None,
        run_dir: Optional[Path] = None,
    ) -> OutputPayloadCheck:
        check = OutputSpecAdapter.check_result(result, output_options, run_dir)
        if check.missing_metrics:
            available = ", ".join(check.available_metrics) or "none"
            raise ValueError(
                "Output contract missing required metric keys: "
                f"{check.missing_metrics}. Available: {available}"
            )
        if check.missing_artifacts:
            available = ", ".join(check.available_artifacts) or "none"
            heatmap_requested = bool((output_options or {}).get("enable_heatmap", True))
            prefix = "enable_heatmap=true but " if heatmap_requested else ""
            raise ValueError(
                f"{prefix}Output contract missing required artifacts: "
                f"{check.missing_artifacts}. Available: {available}"
            )
        return check


def _result_metric_keys(result: "ExperimentResult") -> set[str]:
    return {
        key
        for single_result in result.results.values()
        for key in (single_result.metrics or {})
    }


def _result_artifacts(result: "ExperimentResult", run_dir: Optional[Path]) -> set[str]:
    artifacts = set()
    if result.results:
        artifacts.add("metrics_by_alpha")
    if any(single_result.W_students is not None for single_result in result.results.values()):
        artifacts.add("matrix_factors")
    if "overlap_matrix" in _result_metric_keys(result):
        artifacts.add("overlap_matrix")
    if result.W_teacher is not None:
        artifacts.add("W_teacher")

    if run_dir is not None:
        for filename in ["config.json", "metadata.json", "metrics.json"]:
            if (run_dir / filename).exists():
                artifacts.add(filename)
    return artifacts


def _contract_output_plan(result: "ExperimentResult") -> Dict[str, Any]:
    contract = result.metadata.contract if result.metadata else {}
    if not isinstance(contract, dict):
        return {}
    output_plan = contract.get("output_plan")
    return output_plan if isinstance(output_plan, dict) else {}


def _artifact_requirement_satisfied(requirement: str, artifacts: set[str]) -> bool:
    if "|" in requirement:
        return any(part in artifacts for part in requirement.split("|"))
    return requirement in artifacts


def _curve_code_to_metric_key(curve_code: str) -> str:
    from matrix_factorization.modules.outputs.plot_registry import parse_curve_code

    spec = parse_curve_code(curve_code)
    metric_name = f"{spec.metric_name}_replica" if spec.is_replica else spec.metric_name
    return f"{metric_name}_mean"
