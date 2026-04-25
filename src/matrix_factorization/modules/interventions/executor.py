"""Runtime extension executor.

This executor wires InterventionSpec, ProbeSpec, and AnalyzerSpec into a
single runtime object without changing algorithm internals. It records hook
dispatch and analyzer summaries so vertical extensions are visible in run
metadata before behavior is migrated out of legacy algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from matrix_factorization.core.contracts import (
    AlgorithmStateView,
    AnalyzerSpec,
    ProbeSpec,
    get_analyzer_specs,
    get_probe_specs,
)

from .base import HookPoint, RuntimeHookContext, build_intervention, InterventionBase


@dataclass
class RuntimeExtensionReport:
    algorithm_key: str
    interventions: List[str] = field(default_factory=list)
    probes: List[str] = field(default_factory=list)
    analyzers: List[str] = field(default_factory=list)
    dispatched_hooks: List[Dict[str, Any]] = field(default_factory=list)
    probe_reports: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    analyzer_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_key": self.algorithm_key,
            "interventions": list(self.interventions),
            "probes": list(self.probes),
            "analyzers": list(self.analyzers),
            "dispatched_hooks": list(self.dispatched_hooks),
            "probe_reports": {key: list(value) for key, value in self.probe_reports.items()},
            "analyzer_reports": dict(self.analyzer_reports),
        }


class RuntimeExtensionExecutor:
    """No-formula-change executor for declared runtime extensions."""

    def __init__(
        self,
        *,
        algorithm_key: str,
        interventions: Optional[List[InterventionBase]] = None,
        probes: Optional[List[ProbeSpec]] = None,
        analyzers: Optional[List[AnalyzerSpec]] = None,
    ):
        self.algorithm_key = algorithm_key
        self.interventions = list(interventions or [])
        self.probes = list(probes or [])
        self.analyzers = list(analyzers or [])
        self.report = RuntimeExtensionReport(
            algorithm_key=algorithm_key,
            interventions=[item.spec.key for item in self.interventions],
            probes=[item.key for item in self.probes],
            analyzers=[item.key for item in self.analyzers],
        )

    @classmethod
    def from_contract(
        cls,
        contract: Optional[Dict[str, Any]],
        algorithm_key: Optional[str] = None,
    ) -> "RuntimeExtensionExecutor":
        contract = contract if isinstance(contract, dict) else {}
        resolved_algorithm = algorithm_key or contract.get("algorithm") or "<unknown>"
        probe_specs = get_probe_specs()
        analyzer_specs = get_analyzer_specs()

        interventions = [
            build_intervention(key)
            for key in contract.get("interventions", [])
        ]
        probes = [
            probe_specs[key]
            for key in contract.get("probes", [])
            if key in probe_specs
        ]
        analyzers = [
            analyzer_specs[key]
            for key in contract.get("analyzers", [])
            if key in analyzer_specs
        ]
        return cls(
            algorithm_key=resolved_algorithm,
            interventions=interventions,
            probes=probes,
            analyzers=analyzers,
        )

    def dispatch(
        self,
        hook: HookPoint | str,
        state: AlgorithmStateView,
        context: RuntimeHookContext,
    ) -> AlgorithmStateView:
        hook_point = HookPoint(hook)
        next_state = state
        triggered_interventions = []
        triggered_probes = []

        for intervention in self.interventions:
            if intervention.spec.trigger == hook_point.value:
                self._require_state(intervention.spec.key, intervention.spec.requires_state, next_state)
                next_state = intervention.apply(next_state, context)
                triggered_interventions.append(intervention.spec.key)

        for probe in self.probes:
            if probe.trigger == hook_point.value:
                self._require_state(probe.key, probe.requires_state, next_state)
                self._record_probe_report(probe, next_state, context)
                triggered_probes.append(probe.key)

        if triggered_interventions or triggered_probes:
            self.report.dispatched_hooks.append({
                "hook": hook_point.value,
                "interventions": triggered_interventions,
                "probes": triggered_probes,
                "state_capabilities": next_state.available_capabilities(),
                "step_index": context.step_index,
                "alpha": context.alpha,
            })
        return next_state

    def _record_probe_report(
        self,
        probe: ProbeSpec,
        state: AlgorithmStateView,
        context: RuntimeHookContext,
    ) -> None:
        if probe.key == "batch_summary":
            payload = {
                "hook": context.hook.value,
                "batch_idx": context.metadata.get("batch_idx", state.metadata.get("batch_idx")),
                "alpha_values": list(context.metadata.get("alpha_values", state.metadata.get("alpha_values", []))),
                "metric_keys": list(state.metadata.get("metric_keys", [])),
                "algorithm_result_outputs": list(state.metadata.get("algorithm_result_outputs", [])),
                "state_capabilities": state.available_capabilities(),
                "metadata_only": True,
            }
        else:
            payload = {
                "hook": context.hook.value,
                "state_capabilities": state.available_capabilities(),
                "status": "declared_no_payload_executor",
                "metadata_only": True,
            }
        self.report.probe_reports.setdefault(probe.key, []).append(payload)

    @staticmethod
    def _require_state(key: str, required: List[str], state: AlgorithmStateView) -> None:
        available = set(state.available_capabilities())
        missing = sorted(set(required) - available)
        if missing:
            raise RuntimeError(
                f"runtime extension '{key}' requires state {missing}, "
                f"but AlgorithmStateView only exposes {sorted(available)}"
            )

    def run_analyzers(self, result: Any) -> Dict[str, Dict[str, Any]]:
        """Run declared post-run analyzers using only lightweight result data."""
        reports: Dict[str, Dict[str, Any]] = {}
        for analyzer in self.analyzers:
            self._require_analyzer_inputs(analyzer, result)
            if analyzer.key == "replica_summary":
                reports[analyzer.key] = self._replica_summary(result)
            elif analyzer.key == "tensor_heatmap_summary":
                reports[analyzer.key] = self._tensor_heatmap_summary(result)
            else:
                reports[analyzer.key] = {"status": "declared_no_executor"}
        self.report.analyzer_reports.update(reports)
        return reports

    @classmethod
    def _require_analyzer_inputs(cls, analyzer: AnalyzerSpec, result: Any) -> None:
        available = cls._available_result_inputs(result)
        missing = sorted(set(analyzer.requires) - available)
        if missing:
            raise RuntimeError(
                f"analyzer '{analyzer.key}' requires {missing}, "
                f"but result only exposes {sorted(available)}"
            )

    @staticmethod
    def _available_result_inputs(result: Any) -> set[str]:
        results = getattr(result, "results", {}) or {}
        available = set()
        if results:
            available.add("metrics_by_alpha")
        for single_result in results.values():
            available.update((getattr(single_result, "metrics", None) or {}).keys())
            if getattr(single_result, "W_students", None) is not None or getattr(single_result, "X_students", None) is not None:
                available.add("matrix_factors")
        return available

    @staticmethod
    def _replica_summary(result: Any) -> Dict[str, Any]:
        metric_keys = {
            key
            for single_result in getattr(result, "results", {}).values()
            for key in (single_result.metrics or {})
        }
        return {
            "status": "ok",
            "scan_points": len(getattr(result, "results", {})),
            "replica_metric_keys": sorted(key for key in metric_keys if "replica" in key),
        }

    @staticmethod
    def _tensor_heatmap_summary(result: Any) -> Dict[str, Any]:
        overlap_count = sum(
            1
            for single_result in getattr(result, "results", {}).values()
            if "overlap_matrix" in (single_result.metrics or {})
        )
        return {
            "status": "ok" if overlap_count else "missing_overlap_matrix",
            "scan_points": len(getattr(result, "results", {})),
            "overlap_matrix_count": overlap_count,
        }
