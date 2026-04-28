"""Build and validate an effective experiment plan before execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import re
import yaml

from .contracts import (
    AlgorithmSpec,
    InterventionSpec,
    AnalyzerSpec,
    BatchingSpec,
    MetricSpec,
    MemoryModelSpec,
    NormalizationSpec,
    OutputSpec,
    ParameterSpec,
    PrecisionPolicySpec,
    ProbeSpec,
    ResourceSpec,
    SeedPolicySpec,
    TeacherSpec,
    get_algorithm_specs,
    get_algorithm_metric_keys,
    get_analyzer_specs,
    get_batching_specs,
    get_continuation_specs,
    get_effective_seed_policy_summary,
    get_intervention_specs,
    get_memory_model_specs,
    get_metric_specs,
    get_normalization_specs,
    get_output_specs,
    get_parameter_specs,
    get_precision_policy_specs,
    get_probe_specs,
    get_resource_specs,
    get_seed_policy_specs,
    get_teacher_specs,
    get_tensor_parity_report,
)
from .scan_planning import ScanPlan, build_scan_plan


@dataclass
class OutputPlan:
    specs: List[str] = field(default_factory=list)
    required_metrics: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    metric_semantics: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    plot_semantics: Dict[str, Any] = field(default_factory=dict)
    artifact_semantics: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specs": list(self.specs),
            "required_metrics": list(self.required_metrics),
            "required_artifacts": list(self.required_artifacts),
            "output_files": list(self.output_files),
            "metric_semantics": dict(self.metric_semantics),
            "plot_semantics": dict(self.plot_semantics),
            "artifact_semantics": dict(self.artifact_semantics),
        }


@dataclass
class ExperimentPlan:
    config: Any
    output_options: Dict[str, Any]
    raw_config: Dict[str, Any]
    config_path: Optional[Path] = None
    strict: bool = False
    algorithm_spec: Optional[AlgorithmSpec] = None
    teacher_spec: Optional[TeacherSpec] = None
    resource_spec: Optional[ResourceSpec] = None
    batching_spec: Optional[BatchingSpec] = None
    memory_model_spec: Optional[MemoryModelSpec] = None
    seed_policy_spec: Optional[SeedPolicySpec] = None
    normalization_spec: Optional[NormalizationSpec] = None
    precision_policy_spec: Optional[PrecisionPolicySpec] = None
    resource_plan: Dict[str, Any] = field(default_factory=dict)
    scan_plan: Optional[ScanPlan] = None
    metric_specs: List[MetricSpec] = field(default_factory=list)
    output_specs: List[OutputSpec] = field(default_factory=list)
    output_plan: OutputPlan = field(default_factory=OutputPlan)
    intervention_specs: List[InterventionSpec] = field(default_factory=list)
    probe_specs: List[ProbeSpec] = field(default_factory=list)
    analyzer_specs: List[AnalyzerSpec] = field(default_factory=list)
    effective_parameters: Dict[str, Any] = field(default_factory=dict)
    tensor_parity_report: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_path": str(self.config_path) if self.config_path else None,
            "strict": self.strict,
            "is_valid": self.is_valid,
            "algorithm": self.effective_parameters.get("algorithm_key"),
            "algorithm_spec": {
                "key": self.algorithm_spec.key,
                "status": self.algorithm_spec.status,
                "result_contract": self.algorithm_spec.result_contract,
                "capabilities": list(self.algorithm_spec.capabilities),
                "state_capabilities": list(self.algorithm_spec.state_capabilities),
                "data_requirements": list(self.algorithm_spec.data_requirements),
                "compatible_outputs": list(self.algorithm_spec.compatible_outputs),
            } if self.algorithm_spec else None,
            "teacher_spec": {
                "key": self.teacher_spec.key,
                "status": self.teacher_spec.status,
                "required_config_paths": list(self.teacher_spec.required_config_paths),
                "produced_artifacts": list(self.teacher_spec.produced_artifacts),
                "scale_convention": self.teacher_spec.scale_convention,
            } if self.teacher_spec else None,
            "resource_spec": {
                "algorithm_key": self.resource_spec.algorithm_key,
                "estimator_key": self.resource_spec.estimator_key,
                "device_support": list(self.resource_spec.device_support),
                "dtype_modes": list(self.resource_spec.dtype_modes),
                "compile_support": self.resource_spec.compile_support,
                "probe_support": self.resource_spec.probe_support,
                "empty_cache_policy": self.resource_spec.empty_cache_policy,
            } if self.resource_spec else None,
            "batching_spec": {
                "algorithm_key": self.batching_spec.algorithm_key,
                "planner_layers": list(self.batching_spec.planner_layers),
                "alpha_batching": self.batching_spec.alpha_batching,
                "sample_batching": self.batching_spec.sample_batching,
                "student_batching": self.batching_spec.student_batching,
                "scan_axis_batching": self.batching_spec.scan_axis_batching,
                "steps_reuse": self.batching_spec.steps_reuse,
                "chunking": self.batching_spec.chunking,
                "foldable_axes": list(self.batching_spec.foldable_axes),
                "random_sensitive_axes": list(self.batching_spec.random_sensitive_axes),
                "internal_batcher": self.batching_spec.internal_batcher,
                "seed_partition_sensitive": self.batching_spec.seed_partition_sensitive,
                "sample_range_honored": self.batching_spec.sample_range_honored,
                "metadata_only": self.batching_spec.metadata_only,
            } if self.batching_spec else None,
            "memory_model_spec": {
                "algorithm_key": self.memory_model_spec.algorithm_key,
                "estimator_entrypoint": self.memory_model_spec.estimator_entrypoint,
                "formula_basis": self.memory_model_spec.formula_basis,
                "tensor_components": list(self.memory_model_spec.tensor_components),
                "calibration_status": self.memory_model_spec.calibration_status,
                "probe_required": self.memory_model_spec.probe_required,
                "sample_range_policy": self.memory_model_spec.sample_range_policy,
                "drives_execution": self.memory_model_spec.drives_execution,
            } if self.memory_model_spec else None,
            "normalization_spec": {
                "algorithm_key": self.normalization_spec.algorithm_key,
                "schema_version": self.normalization_spec.schema_version,
                "profiles": list(self.normalization_spec.profiles),
                "default_profile": self.normalization_spec.default_profile,
                "latent_scale": self.normalization_spec.latent_scale,
                "teacher_init_variance": self.normalization_spec.teacher_init_variance,
                "student_init_variance": self.normalization_spec.student_init_variance,
                "prior_precision_base": self.normalization_spec.prior_precision_base,
                "interaction_scale": self.normalization_spec.interaction_scale,
                "alpha_edge_scale": self.normalization_spec.alpha_edge_scale,
                "metric_rescale_policy": self.normalization_spec.metric_rescale_policy,
                "status": self.normalization_spec.status,
            } if self.normalization_spec else None,
            "precision_policy_spec": {
                "algorithm_key": self.precision_policy_spec.algorithm_key,
                "profiles": list(self.precision_policy_spec.profiles),
                "default_profile": self.precision_policy_spec.default_profile,
                "requested_profile": getattr(getattr(self.config, "algorithm_params", None), "precision_profile", None),
                "role_dtypes": self.precision_policy_spec.role_dtype_map(
                    getattr(getattr(self.config, "algorithm_params", None), "precision_profile", self.precision_policy_spec.default_profile)
                ),
                "status": self.precision_policy_spec.status,
            } if self.precision_policy_spec else None,
            "seed_policy_spec": {
                "algorithm_key": self.seed_policy_spec.algorithm_key,
                "policy_key": self.seed_policy_spec.policy_key,
                "seed_inputs": list(self.seed_policy_spec.seed_inputs),
                "random_streams": list(self.seed_policy_spec.random_streams),
                "partition_invariant": self.seed_policy_spec.partition_invariant,
                "batch_partition_sensitive": self.seed_policy_spec.batch_partition_sensitive,
                "automatic_rebatch_allowed": self.seed_policy_spec.automatic_rebatch_allowed,
                "notes": self.seed_policy_spec.notes,
            } if self.seed_policy_spec else None,
            "resource_plan": dict(self.resource_plan),
            "scan_plan": self.scan_plan.to_dict() if self.scan_plan else None,
            "tensor_parity_report": dict(self.tensor_parity_report or {}),
            "metrics": [spec.key for spec in self.metric_specs],
            "available_metric_keys": _available_metric_keys(self),
            "outputs": [spec.key for spec in self.output_specs],
            "output_plan": self.output_plan.to_dict(),
            "interventions": [spec.key for spec in self.intervention_specs],
            "intervention_contracts": [
                {
                    "key": spec.key,
                    "trigger": spec.trigger,
                    "requires_state": list(spec.requires_state),
                    "modifies_state": list(spec.modifies_state),
                    "compatible_algorithms": list(spec.compatible_algorithms),
                    "physical_sensitive": spec.physical_sensitive,
                }
                for spec in self.intervention_specs
            ],
            "probes": [spec.key for spec in self.probe_specs],
            "probe_contracts": [
                {
                    "key": spec.key,
                    "trigger": spec.trigger,
                    "requires_state": list(spec.requires_state),
                    "produces": list(spec.produces),
                    "runtime_status": spec.runtime_status,
                }
                for spec in self.probe_specs
            ],
            "analyzers": [spec.key for spec in self.analyzer_specs],
            "effective_parameters": dict(self.effective_parameters),
            "parameter_chain": self.parameter_chain(),
            "parameter_consumption": self.parameter_chain(),
            "physical_sensitive_parameters": [
                item["path"]
                for item in self.parameter_chain()
                if item.get("physical_sensitive")
            ],
            "issues": _issue_records(self.errors, self.warnings),
            "error_codes": [_issue_code(item) for item in self.errors],
            "warning_codes": [_issue_code(item) for item in self.warnings],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    def format_validation(self) -> str:
        lines = ["配置校验结果"]
        lines.append(f"  config: {self.config_path or '<memory>'}")
        lines.append(f"  algorithm: {self.effective_parameters.get('algorithm_key', '<unknown>')}")
        if self.errors:
            lines.append("  errors:")
            lines.extend(f"    - {item}" for item in self.errors)
        else:
            lines.append("  errors: none")
        if self.warnings:
            lines.append("  warnings:")
            lines.extend(f"    - {item}" for item in self.warnings)
        else:
            lines.append("  warnings: none")
        return "\n".join(lines)

    def format_explain(self) -> str:
        lines = ["配置解释", ""]
        lines.append(f"文件: {self.config_path or '<memory>'}")
        lines.append(f"实际 algorithm: {self.effective_parameters.get('algorithm_key', '<unknown>')}")
        raw_algorithm = self.raw_config.get("algorithm") if isinstance(self.raw_config, dict) else None
        if raw_algorithm is not None:
            lines.append(f"YAML algorithm: {raw_algorithm}")
        if self.algorithm_spec:
            lines.append(f"algorithm status: {self.algorithm_spec.status}")
            lines.append(f"result contract: {self.algorithm_spec.result_contract}")
            lines.append(f"data requirements: {', '.join(self.algorithm_spec.data_requirements) or 'none'}")
            lines.append(f"capabilities: {', '.join(self.algorithm_spec.capabilities) or 'none'}")
            lines.append(f"compatible outputs: {', '.join(self.algorithm_spec.compatible_outputs) or 'none'}")
        if self.teacher_spec:
            lines.append(f"teacher status: {self.teacher_spec.status}")
            if self.teacher_spec.scale_convention:
                lines.append(f"teacher scale: {self.teacher_spec.scale_convention}")
        if self.normalization_spec:
            lines.append(
                "normalization: "
                f"{self.normalization_spec.latent_scale}, "
                f"prior={self.normalization_spec.prior_precision_base}, "
                f"schema=v{self.normalization_spec.schema_version}"
            )
        if self.precision_policy_spec:
            params = getattr(self.config, "algorithm_params", None)
            profile = getattr(params, "precision_profile", self.precision_policy_spec.default_profile)
            lines.append(f"precision profile: {profile}")
            role_map = self.precision_policy_spec.role_dtype_map(profile)
            for role in ["student_factors", "factor_variances", "observations_Y", "F_rademacher", "F_gaussian", "metric_reductions"]:
                if role in role_map:
                    dtypes = role_map[role]
                    lines.append(
                        f"  {role}: storage={dtypes.get('storage')}, "
                        f"compute={dtypes.get('compute')}, accumulator={dtypes.get('accumulator')}"
                    )
        if self.scan_plan:
            lines.append("")
            lines.append("scan plan:")
            lines.append(f"  kind: {self.scan_plan.scan_kind}")
            lines.append(f"  points: {self.scan_plan.num_points}")
            lines.append(
                "  axes: "
                f"{', '.join(axis.key for axis in self.scan_plan.axes) or 'none'}"
            )
            lines.append(
                "  grouping: "
                f"{', '.join(group.group_id for group in self.scan_plan.grouping) or 'none'}"
            )
        if self.resource_plan:
            lines.append("")
            lines.append("resource / batching contract:")
            lines.append(f"  estimator: {self.resource_plan.get('estimator_key')}")
            lines.append(f"  planner_layers: {', '.join(self.resource_plan.get('planner_layers') or []) or 'none'}")
            lines.append(f"  alpha_batching: {self.resource_plan.get('alpha_batching')}")
            lines.append(f"  sample_batching: {self.resource_plan.get('sample_batching')}")
            lines.append(f"  seed_partition_sensitive: {self.resource_plan.get('seed_partition_sensitive')}")
            lines.append(f"  metadata_only: {self.resource_plan.get('metadata_only')}")
            memory_model = self.resource_plan.get("memory_model") or {}
            if memory_model:
                lines.append(f"  memory_model: {memory_model.get('estimator_entrypoint')}")
                lines.append(f"  memory_calibration: {memory_model.get('calibration_status')}")
                lines.append(f"  memory_drives_execution: {memory_model.get('drives_execution')}")
            seed_policy = self.resource_plan.get("seed_policy") or {}
            if seed_policy:
                lines.append(f"  seed_policy: {seed_policy.get('policy_key')}")
                lines.append(f"  partition_invariant: {seed_policy.get('partition_invariant')}")
                lines.append(f"  automatic_rebatch_allowed: {seed_policy.get('automatic_rebatch_allowed')}")
            preview = self.resource_plan.get("resource_execution_plan_preview") or {}
            if preview:
                lines.append("  scan/resource execution plan:")
                lines.append(f"    groups: {preview.get('num_groups')}")
                lines.append(f"    batches: {preview.get('num_batches')}")
                lines.append(f"    work_items: {preview.get('num_work_items')}")
                lines.append(f"    max_allocated_estimate: {preview.get('max_estimated_allocated_gb')}")
                lines.append(f"    max_device_estimate: {preview.get('max_estimated_device_gb')}")
                lines.append(
                    "    calibration_sources: "
                    f"{', '.join(preview.get('calibration_sources') or []) or 'none'}"
                )
                if preview.get("preflight_errors"):
                    lines.append("    preflight_errors:")
                    lines.extend(f"      - {item}" for item in preview.get("preflight_errors") or [])
        if self.tensor_parity_report:
            lines.append("")
            lines.append("tensor serial/parallel parity:")
            lines.append(
                f"  same_result_contract: {self.tensor_parity_report.get('same_result_contract')}"
            )
            lines.append(
                "  shared_metrics: "
                f"{', '.join(self.tensor_parity_report.get('shared_metrics') or []) or 'none'}"
            )
            lines.append(
                "  serial_missing_parallel_metrics: "
                f"{', '.join(self.tensor_parity_report.get('serial_missing_parallel_metrics') or []) or 'none'}"
            )
            lines.append(
                "  parallel_missing_serial_metrics: "
                f"{', '.join(self.tensor_parity_report.get('parallel_missing_serial_metrics') or []) or 'none'}"
            )
        lines.append("")
        lines.append("有效参数摘要:")
        for key in sorted(self.effective_parameters):
            lines.append(f"  {key}: {self.effective_parameters[key]}")
        parameter_chain = self._format_parameter_chain()
        if parameter_chain:
            lines.append("")
            lines.append("参数链路:")
            lines.extend(parameter_chain)
        if self.output_specs:
            lines.append("")
            lines.append("将使用的 output contract:")
            for spec in self.output_specs:
                lines.append(f"  - {spec.key}: requires={spec.requires}")
        if self.output_plan.specs:
            lines.append("")
            lines.append("output plan:")
            lines.append(f"  required_metrics: {', '.join(self.output_plan.required_metrics) or 'none'}")
            lines.append(f"  required_artifacts: {', '.join(self.output_plan.required_artifacts) or 'none'}")
            lines.append(f"  output_files: {', '.join(self.output_plan.output_files) or 'none'}")
            if self.output_plan.plot_semantics:
                lines.append("  plot queries:")
                for key, semantics in self.output_plan.plot_semantics.items():
                    preview = semantics.get("selection_preview") or {}
                    query = semantics.get("query") or {}
                    lines.append(
                        f"    - {key}: x={query.get('x')}, "
                        f"series={preview.get('series_count', 0)}, "
                        f"points={preview.get('selected_point_count', 0)}"
                    )
                    for series in preview.get("series", [])[:5]:
                        lines.append(
                            f"      {series.get('label')}: "
                            f"{', '.join(series.get('point_ids') or [])}"
                        )
        available_metrics = _available_metric_keys(self)
        if available_metrics:
            lines.append("")
            lines.append("可用 flat metric keys:")
            lines.append(f"  {', '.join(available_metrics)}")
        if self.intervention_specs:
            lines.append("")
            lines.append("将使用的 intervention contract:")
            for spec in self.intervention_specs:
                lines.append(
                    f"  - {spec.key}: trigger={spec.trigger}, "
                    f"requires_state={spec.requires_state}, "
                    f"modifies_state={spec.modifies_state}, "
                    f"physical_sensitive={spec.physical_sensitive}"
                )
        if self.probe_specs:
            lines.append("")
            lines.append("将使用的 probe contract:")
            for spec in self.probe_specs:
                lines.append(f"  - {spec.key}: trigger={spec.trigger}, requires_state={spec.requires_state}")
        if self.analyzer_specs:
            lines.append("")
            lines.append("将使用的 analyzer contract:")
            for spec in self.analyzer_specs:
                lines.append(f"  - {spec.key}: requires={spec.requires}")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  - {item}" for item in self.warnings)
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            lines.extend(f"  - {item}" for item in self.errors)
        return "\n".join(lines)

    def _format_parameter_chain(self) -> List[str]:
        lines = []
        for item in self.parameter_chain():
            path = item["path"]
            if not item["registered"]:
                lines.append(f"  - {path}: 未注册")
                continue
            consumers = ", ".join(item["consumers"]) if item["consumers"] else "none"
            lines.append(
                f"  - {path}: owner={item['owner']}, status={item['status']}, "
                f"active={item['active_in_current_plan']}, "
                f"consumption={item['consumption_status']}, "
                f"effective={item['effective_value']}, "
                f"physical_sensitive={item['physical_sensitive']}, consumers={consumers}"
            )
        return lines

    def parameter_chain(self) -> List[Dict[str, Any]]:
        """Structured YAML path -> contract metadata trace for agents/tools."""
        parameter_specs = get_parameter_specs()
        chain = []
        emitted_paths = set()
        for path, value in sorted(_flatten_config_paths(self.raw_config), key=lambda item: item[0]):
            if _is_canonical_scan_internal_path(path):
                if path.startswith("scan.axes.") and "scan.axes" not in emitted_paths:
                    spec = parameter_specs["scan.axes"]
                    effective_value, effective_source, derived_effect = _parameter_effective_trace(self, "scan.axes")
                    active_in_current_plan = _path_active_in_current_plan(self, "scan.axes")
                    chain.append({
                        "path": "scan.axes",
                        "registered": True,
                        "value": _get_raw_path(self.raw_config, "scan.axes"),
                        "owner": spec.owner,
                        "status": spec.status,
                        "physical_sensitive": spec.physical_sensitive,
                        "consumers": list(spec.consumers),
                        "value_type": spec.value_type,
                        "active_in_current_plan": active_in_current_plan,
                        "effective_value": effective_value,
                        "effective_source": effective_source,
                        "derived_effect": derived_effect,
                        "consumption_status": _parameter_consumption_status(
                            spec=spec,
                            active_in_current_plan=active_in_current_plan,
                            effective_source=effective_source,
                            derived_effect=derived_effect,
                        ),
                    })
                    emitted_paths.add("scan.axes")
                continue
            spec = parameter_specs.get(path)
            if not spec:
                chain.append({
                    "path": path,
                    "registered": False,
                    "value": value,
                    "owner": None,
                    "status": "unregistered",
                    "physical_sensitive": False,
                    "consumers": [],
                    "value_type": None,
                    "active_in_current_plan": True,
                    "effective_value": None,
                    "effective_source": None,
                    "derived_effect": None,
                    "consumption_status": "unregistered",
                })
                emitted_paths.add(path)
                continue
            effective_value, effective_source, derived_effect = _parameter_effective_trace(self, path)
            active_in_current_plan = _path_active_in_current_plan(self, path)
            consumption_status = _parameter_consumption_status(
                spec=spec,
                active_in_current_plan=active_in_current_plan,
                effective_source=effective_source,
                derived_effect=derived_effect,
            )
            chain.append({
                "path": path,
                "registered": True,
                "value": value,
                "owner": spec.owner,
                "status": spec.status,
                "physical_sensitive": spec.physical_sensitive,
                "consumers": list(spec.consumers),
                "value_type": spec.value_type,
                "active_in_current_plan": active_in_current_plan,
                "effective_value": effective_value,
                "effective_source": effective_source,
                "derived_effect": derived_effect,
                "consumption_status": consumption_status,
            })
            emitted_paths.add(path)
        for path in [
            "algorithm_params.normalization_profile",
            "algorithm_params.precision_profile",
            "algorithm_params.precision_fallback_policy",
        ]:
            if path in emitted_paths or path not in parameter_specs:
                continue
            spec = parameter_specs[path]
            effective_value, effective_source, derived_effect = _parameter_effective_trace(self, path)
            if effective_source is None and derived_effect is None:
                continue
            active_in_current_plan = _path_active_in_current_plan(self, path)
            chain.append({
                "path": path,
                "registered": True,
                "value": None,
                "owner": spec.owner,
                "status": spec.status,
                "physical_sensitive": spec.physical_sensitive,
                "consumers": list(spec.consumers),
                "value_type": spec.value_type,
                "active_in_current_plan": active_in_current_plan,
                "effective_value": effective_value,
                "effective_source": effective_source,
                "derived_effect": derived_effect,
                "consumption_status": _parameter_consumption_status(
                    spec=spec,
                    active_in_current_plan=active_in_current_plan,
                    effective_source=effective_source,
                    derived_effect=derived_effect,
                ),
            })
        return chain


def build_experiment_plan(
    config: Any,
    output_options: Optional[Dict[str, Any]] = None,
    raw_yaml: str = "",
    config_path: Optional[Path] = None,
    strict: bool = False,
) -> ExperimentPlan:
    output_options = dict(output_options or {})
    raw_config = _load_raw_config(raw_yaml)
    plan = ExperimentPlan(
        config=config,
        output_options=output_options,
        raw_config=raw_config,
        config_path=Path(config_path) if config_path is not None else None,
        strict=strict,
    )

    parameter_specs = get_parameter_specs()
    algorithm_specs = get_algorithm_specs()
    metric_specs = get_metric_specs()
    output_specs = get_output_specs()
    intervention_specs = get_intervention_specs()
    probe_specs = get_probe_specs()
    analyzer_specs = get_analyzer_specs()
    teacher_specs = get_teacher_specs()
    resource_specs = get_resource_specs()
    batching_specs = get_batching_specs()
    memory_model_specs = get_memory_model_specs()
    seed_policy_specs = get_seed_policy_specs()
    normalization_specs = get_normalization_specs()
    precision_policy_specs = get_precision_policy_specs()

    _validate_raw_paths(plan, parameter_specs)
    _validate_scan_axis_paths(plan, parameter_specs)
    _validate_scan_execution_options(plan)
    _validate_parameter_values(plan, parameter_specs)

    algorithm_key = getattr(config, "algorithm_key", None)
    plan.effective_parameters.update(_effective_parameter_summary(config, output_options, raw_config))
    try:
        plan.scan_plan = build_scan_plan(config, raw_config)
        plan.effective_parameters["scan_plan.kind"] = plan.scan_plan.scan_kind
        plan.effective_parameters["scan_plan.num_points"] = plan.scan_plan.num_points
    except Exception as exc:
        plan.errors.append(f"ScanPlan 构造失败: {exc}")
    if algorithm_key in algorithm_specs:
        plan.algorithm_spec = algorithm_specs[algorithm_key]
    else:
        plan.errors.append(f"algorithm_key 未注册 AlgorithmSpec: {algorithm_key}")
        return plan
    if algorithm_key in {"bigamp_tensor", "bigamp_tensor_parallel"}:
        plan.tensor_parity_report = get_tensor_parity_report()
    plan.resource_spec = resource_specs.get(algorithm_key)
    plan.batching_spec = batching_specs.get(algorithm_key)
    plan.memory_model_spec = memory_model_specs.get(algorithm_key)
    plan.seed_policy_spec = seed_policy_specs.get(algorithm_key)
    plan.normalization_spec = normalization_specs.get(algorithm_key)
    plan.precision_policy_spec = precision_policy_specs.get(algorithm_key)
    _validate_scan_continuation_options(plan)
    _build_resource_plan(plan)

    teacher_key = getattr(config, "teacher_key", None)
    if teacher_key in teacher_specs:
        plan.teacher_spec = teacher_specs[teacher_key]
    else:
        plan.errors.append(f"teacher_key 未注册 TeacherSpec: {teacher_key}")
    if not plan.resource_spec:
        plan.errors.append(f"algorithm_key 未注册 ResourceSpec: {algorithm_key}")
    if not plan.batching_spec:
        plan.errors.append(f"algorithm_key 未注册 BatchingSpec: {algorithm_key}")
    if not plan.memory_model_spec:
        plan.errors.append(f"algorithm_key 未注册 MemoryModelSpec: {algorithm_key}")
    if not plan.seed_policy_spec:
        plan.errors.append(f"algorithm_key 未注册 SeedPolicySpec: {algorithm_key}")
    if not plan.normalization_spec:
        plan.errors.append(f"algorithm_key 未注册 NormalizationSpec: {algorithm_key}")
    if not plan.precision_policy_spec:
        plan.errors.append(f"algorithm_key 未注册 PrecisionPolicySpec: {algorithm_key}")

    _validate_required_config_paths(plan, parameter_specs)
    _validate_normalization_profile(plan)
    _validate_precision_policy(plan)

    if plan.algorithm_spec.status != "active":
        allow_experimental = bool(raw_config.get("allow_experimental", False)) if isinstance(raw_config, dict) else False
        if not allow_experimental:
            plan.errors.append(
                f"algorithm '{algorithm_key}' status={plan.algorithm_spec.status}; "
                "主链路默认只允许 active algorithm。"
            )

    plan.metric_specs = [
        metric_specs[key]
        for key in plan.algorithm_spec.produced_metrics
        if key in metric_specs
    ]
    _select_output_specs(plan, output_specs)
    _build_output_plan(plan)
    _select_intervention_specs(plan, intervention_specs)
    _select_probe_specs(plan, probe_specs)
    _select_analyzer_specs(plan, analyzer_specs)
    _validate_output_compatibility(plan)
    _validate_custom_plot_metrics(plan)
    _validate_runtime_extension_compatibility(plan)
    _validate_seed_policy_compatibility(plan)
    _warn_for_soft_parameters(plan, parameter_specs, strict=strict)
    _warn_for_parameter_consumption(plan, strict=strict)
    _warn_for_route_overrides(plan)
    _warn_for_parameterized_output_name(plan)
    return plan


def _warn_for_parameterized_output_name(plan: ExperimentPlan) -> None:
    output_name = str(plan.output_options.get("name") or "")
    if not output_name:
        return
    if re.search(r"(^|[_-])(?:S|N|M|steps?|alpha|a)\d+|(?:^|[_-])\d+x\d+", output_name, re.IGNORECASE):
        plan.warnings.append(
            "output.name 看起来包含尺寸/样本数/步数等参数事实；运行目录会使用实际 config 自动生成这些 token，"
            "建议 output.name 只写短标签。"
        )


def _validate_required_config_paths(
    plan: ExperimentPlan,
    parameter_specs: Dict[str, ParameterSpec],
) -> None:
    required_paths: List[Tuple[str, str]] = []
    if plan.algorithm_spec:
        required_paths.extend((path, f"AlgorithmSpec[{plan.algorithm_spec.key}]") for path in plan.algorithm_spec.required_config_paths)
    if plan.teacher_spec:
        required_paths.extend((path, f"TeacherSpec[{plan.teacher_spec.key}]") for path in plan.teacher_spec.required_config_paths)

    for path, owner in required_paths:
        if path not in parameter_specs:
            plan.errors.append(f"{owner} requires unknown ParameterSpec path: {path}")
            continue
        active = _path_active_in_current_plan(plan, path)
        value, source, derived_effect = _parameter_effective_trace(plan, path)
        if not active:
            plan.errors.append(f"{owner} requires {path}，但该字段在当前路由下不生效。")
            continue
        if source is None and derived_effect is None:
            plan.errors.append(f"{owner} requires {path}，但当前 plan 找不到 effective trace。")
            continue
        if value is None and derived_effect is None:
            plan.errors.append(f"{owner} requires {path}，但 effective value is None。")


def _validate_precision_policy(plan: ExperimentPlan) -> None:
    if not plan.precision_policy_spec:
        return
    algorithm_params = getattr(plan.config, "algorithm_params", None)
    profile = getattr(algorithm_params, "precision_profile", plan.precision_policy_spec.default_profile)
    if profile not in plan.precision_policy_spec.profiles:
        plan.errors.append(
            f"algorithm '{plan.algorithm_spec.key if plan.algorithm_spec else '<unknown>'}' "
            f"不支持 precision_profile={profile!r}; "
            f"允许值: {plan.precision_policy_spec.profiles}"
        )
    fallback = getattr(algorithm_params, "precision_fallback_policy", "allow")
    if fallback not in {"allow", "error"}:
        plan.errors.append(
            f"algorithm_params.precision_fallback_policy 必须是 'allow' 或 'error'，got {fallback!r}"
        )
    raw_algorithm_params = {}
    if isinstance(plan.raw_config, dict):
        raw_algorithm_params = plan.raw_config.get("algorithm_params") or {}
        if not isinstance(raw_algorithm_params, dict):
            raw_algorithm_params = {}
    if "precision_fallback_policy" in raw_algorithm_params and "dtype_fallback_policy" in raw_algorithm_params:
        message = "同时设置了 precision_fallback_policy 和 legacy dtype_fallback_policy；precision_fallback_policy 优先。"
        if plan.strict:
            plan.errors.append(message)
        else:
            plan.warnings.append(message)
    if "precision_profile" in raw_algorithm_params and "use_bf16" in raw_algorithm_params:
        expected_use_bf16 = profile in {"fast", "aggressive"}
        if bool(raw_algorithm_params.get("use_bf16")) != expected_use_bf16:
            message = (
                "同时设置了 precision_profile 和 legacy use_bf16，且语义冲突；"
                "precision_profile 优先。"
            )
            if plan.strict:
                plan.errors.append(message)
            else:
                plan.warnings.append(message)


def _validate_normalization_profile(plan: ExperimentPlan) -> None:
    if not plan.normalization_spec:
        return
    algorithm_params = getattr(plan.config, "algorithm_params", None)
    profile = getattr(algorithm_params, "normalization_profile", plan.normalization_spec.default_profile)
    if profile not in plan.normalization_spec.profiles:
        plan.errors.append(
            f"algorithm '{plan.algorithm_spec.key if plan.algorithm_spec else '<unknown>'}' "
            f"不支持 normalization_profile={profile!r}; "
            f"允许值: {plan.normalization_spec.profiles}"
        )


def _load_raw_config(raw_yaml: str) -> Dict[str, Any]:
    if not raw_yaml:
        return {}
    loaded = yaml.safe_load(raw_yaml)
    return loaded if isinstance(loaded, dict) else {}


def _flatten_config_paths(data: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(data, dict):
        if not data and prefix:
            yield prefix, data
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_config_paths(value, next_prefix)
    elif isinstance(data, list):
        if prefix:
            yield prefix, data
    else:
        if prefix:
            yield prefix, data


def _parameter_effective_trace(plan: ExperimentPlan, path: str) -> Tuple[Any, Optional[str], Optional[Dict[str, Any]]]:
    """Return raw path -> effective value/effect evidence for explain/JSON.

    This is a trace layer, not a second loader. It documents where the already
    parsed config exposes a YAML field after load_yaml_config has consumed it.
    """
    if path in plan.effective_parameters:
        return plan.effective_parameters[path], "effective_parameters", None

    config = plan.config
    raw_config = plan.raw_config if isinstance(plan.raw_config, dict) else {}

    direct_map = {
        "algorithm": ("algorithm_key", getattr(config, "algorithm_key", None)),
        "teacher": ("teacher_key", getattr(config, "teacher_key", None)),
        "scan.axes": ("scan_spec.axes", _get_raw_path(raw_config, "scan.axes")),
        "scan.execution": ("scan_spec.execution", _get_raw_path(raw_config, "scan.execution")),
        "scan.continuation.enabled": ("scan_spec.continuation.enabled", _get_raw_path(raw_config, "scan.continuation.enabled")),
        "scan.continuation.axis": ("scan_spec.continuation.axis", _get_raw_path(raw_config, "scan.continuation.axis")),
        "scan.continuation.order": ("scan_spec.continuation.order", _get_raw_path(raw_config, "scan.continuation.order")),
        "scan.continuation.state_transfer": ("scan_spec.continuation.state_transfer", _get_raw_path(raw_config, "scan.continuation.state_transfer")),
        "scan.continuation.observation_policy": ("scan_spec.continuation.observation_policy", _get_raw_path(raw_config, "scan.continuation.observation_policy")),
        "scan.continuation.strict_state": ("scan_spec.continuation.strict_state", _get_raw_path(raw_config, "scan.continuation.strict_state")),
        "scan.continuation.adaptive_controller_state": ("scan_spec.continuation.adaptive_controller_state", _get_raw_path(raw_config, "scan.continuation.adaptive_controller_state")),
        "tensor_order": ("spreading.tensor_order", getattr(getattr(config, "spreading", None), "tensor_order", None)),
        "training.seed": ("seeds.base_seed", getattr(getattr(config, "seeds", None), "base_seed", None)),
        "seeds.model": ("seeds.base_seed", getattr(getattr(config, "seeds", None), "base_seed", None)),
        "seeds.data": ("seeds.teacher_seed", getattr(getattr(config, "seeds", None), "teacher_seed", None)),
        "seeds.base_seed": ("seeds.base_seed", getattr(getattr(config, "seeds", None), "base_seed", None)),
        "seeds.teacher_seed": ("seeds.teacher_seed", getattr(getattr(config, "seeds", None), "teacher_seed", None)),
        "seeds.student_seed": ("seeds.student_seed", getattr(getattr(config, "seeds", None), "student_seed", None)),
        "seeds.spreading_seed": ("seeds.spreading_seed", getattr(getattr(config, "seeds", None), "spreading_seed", None)),
        "teacher_config.init_distribution": (
            "teacher.init_distribution",
            getattr(getattr(config, "teacher", None), "init_distribution", None),
        ),
        "teacher_config.mean_scale": (
            "teacher.mean_scale",
            getattr(getattr(config, "teacher", None), "mean_scale", None),
        ),
    }
    if path in direct_map:
        source, value = direct_map[path]
        return value, source, _derived_parameter_effect(plan, path)

    for prefix, attr_name in [
        ("matrix.", "matrix"),
        ("training.", "training"),
        ("seeds.", "seeds"),
        ("algorithm_params.", "algorithm_params"),
        ("spreading.", "spreading"),
    ]:
        if path.startswith(prefix):
            obj = getattr(config, attr_name, None)
            key = path.split(".", 1)[1]
            if obj is not None and hasattr(obj, key):
                return getattr(obj, key), f"{attr_name}.{key}", _derived_parameter_effect(plan, path)

    if path.startswith("output."):
        nested_value = _get_raw_path({"output": plan.output_options}, path)
        if nested_value is not None:
            return nested_value, f"output_options.{path.split('.', 1)[1]}", None
        key = path.split(".", 1)[1]
        if key in plan.output_options:
            return plan.output_options.get(key), f"output_options.{key}", None

    if path in {"probes", "analyzers"}:
        return raw_config.get(path), "raw_config.runtime_extensions", {
            "selected_specs": [spec.key for spec in (plan.probe_specs if path == "probes" else plan.analyzer_specs)]
        }

    return _get_raw_path(raw_config, path), None, _derived_parameter_effect(plan, path)


def _derived_parameter_effect(plan: ExperimentPlan, path: str) -> Optional[Dict[str, Any]]:
    scan_effect = _derived_scan_effect(plan, path)
    seed_effect = _derived_seed_effect(plan, path)
    if scan_effect and seed_effect:
        return {**scan_effect, **seed_effect}
    return scan_effect or seed_effect


def _derived_scan_effect(plan: ExperimentPlan, path: str) -> Optional[Dict[str, Any]]:
    scan = getattr(plan.config, "scan", None)
    if scan is None:
        return None
    values = list(getattr(scan, "values", []) or [])
    if path.startswith("scan."):
        preview = [_json_trace_scalar(value) for value in values[:3]]
        if len(values) > 3:
            preview = preview + ["..."]
        return {
            "scan_dimension": getattr(scan, "dimension", None),
            "scan_num_points": len(values),
            "scan_values_preview": preview,
        }
    return None


def _derived_seed_effect(plan: ExperimentPlan, path: str) -> Optional[Dict[str, Any]]:
    if path != "seeds.spreading_seed":
        return None
    spreading = getattr(plan.config, "spreading", None)
    if spreading is None:
        return {"actual_spreading_seed": None, "spreading_active": False}
    raw_spreading = plan.raw_config.get("spreading", {}) if isinstance(plan.raw_config, dict) else {}
    effect: Dict[str, Any] = {
        "actual_spreading_seed": getattr(spreading, "seed", None),
        "spreading_active": True,
    }
    if isinstance(raw_spreading, dict) and "seed" in raw_spreading:
        effect["overridden_by"] = "spreading.seed"
    return effect


def _json_trace_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value


def _get_raw_path(raw_config: Dict[str, Any], path: str) -> Any:
    current: Any = raw_config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _path_active_in_current_plan(plan: ExperimentPlan, path: str) -> bool:
    if path == "scan.axes":
        return True
    if path == "scan.execution":
        return _get_raw_path(plan.raw_config if isinstance(plan.raw_config, dict) else {}, path) is not None
    if path.startswith("scan.continuation."):
        raw_enabled = _get_raw_path(plan.raw_config if isinstance(plan.raw_config, dict) else {}, "scan.continuation.enabled")
        return bool(raw_enabled)
    algorithm_key = getattr(plan.config, "algorithm_key", None)
    if path.startswith("spreading."):
        return algorithm_key in {"bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel", "agd_spreading"}
    if path in {
        "algorithm_params.learning_rate",
        "algorithm_params.use_early_stop",
        "algorithm_params.target_loss_threshold",
    }:
        return algorithm_key in {"agd", "agd_tensor"}
    if path in {
        "algorithm_params.damping",
        "algorithm_params.noise_var",
        "algorithm_params.adaptive_damping",
        "algorithm_params.step_min",
        "algorithm_params.step_max",
        "algorithm_params.step_incr",
        "algorithm_params.step_decr",
        "algorithm_params.step_window",
        "algorithm_params.max_bad_steps",
    }:
        return algorithm_key in {"bigamp", "bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel"}
    if path in {
        "algorithm_params.use_compile",
    }:
        return algorithm_key in {"bigamp", "bigamp_spreading", "bigamp_tensor_parallel"}
    if path in {
        "algorithm_params.use_bf16",
        "algorithm_params.normalization_profile",
        "algorithm_params.precision_profile",
        "algorithm_params.precision_fallback_policy",
    }:
        return algorithm_key in {"agd", "bigamp", "bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel"}
    if path in {
        "algorithm_params.use_tf32",
    }:
        return algorithm_key in {"agd", "bigamp", "bigamp_spreading", "bigamp_tensor_parallel"}
    if path in {
        "algorithm_params.seed_partition_policy",
    }:
        return algorithm_key in {"agd", "bigamp", "bigamp_spreading", "bigamp_tensor_parallel"}
    if path in {
        "algorithm_params.compile_fallback_policy",
    }:
        return algorithm_key in {"bigamp", "bigamp_spreading", "bigamp_tensor_parallel"}
    if path in {
        "algorithm_params.dtype_fallback_policy",
    }:
        return algorithm_key in {"agd", "bigamp_spreading", "bigamp_tensor_parallel"}
    if path in {
        "algorithm_params.adaptive_restart",
        "algorithm_params.restart_patience",
        "algorithm_params.restart_noise",
        "algorithm_params.acceptance_tolerance",
    }:
        return algorithm_key in {"bigamp_spreading"}
    return True


def _parameter_consumption_status(
    *,
    spec: ParameterSpec,
    active_in_current_plan: bool,
    effective_source: Optional[str],
    derived_effect: Optional[Dict[str, Any]],
) -> str:
    if spec.status != "active":
        return spec.status
    if not active_in_current_plan:
        return "inactive_current_route"
    if derived_effect and derived_effect.get("overridden_by"):
        return "overridden_current_route"
    if effective_source or derived_effect:
        return "effective"
    if spec.consumers:
        return "declared_no_effective_trace"
    return "no_consumers"


def _validate_raw_paths(plan: ExperimentPlan, parameter_specs: Dict[str, ParameterSpec]) -> None:
    known_paths = set(parameter_specs)
    known_top_level = {path.split(".", 1)[0] for path in known_paths}
    legacy_scan_top_level = {"scan_mode", "alpha_scan", "steps_scan", "nested_scan", "hysteresis_scan"}
    for path, _ in _flatten_config_paths(plan.raw_config):
        if path.split(".", 1)[0] in legacy_scan_top_level:
            plan.errors.append(f"旧 scan 字段已移除，请改用 scan.axes: {path}")
            continue
        if _is_canonical_scan_internal_path(path):
            continue
        if path in known_paths:
            continue
        top = path.split(".", 1)[0]
        if top not in known_top_level:
            plan.errors.append(f"未知 YAML 字段: {path}")
        else:
            plan.errors.append(f"未注册参数字段: {path}")


def _is_canonical_scan_internal_path(path: str) -> bool:
    return path.startswith("scan.axes.") or path.startswith("scan.execution.")


def _validate_scan_axis_paths(plan: ExperimentPlan, parameter_specs: Dict[str, ParameterSpec]) -> None:
    raw_scan = plan.raw_config.get("scan", {}) if isinstance(plan.raw_config, dict) else {}
    axes = raw_scan.get("axes", {}) if isinstance(raw_scan, dict) else {}
    if not axes:
        plan.errors.append("canonical scan requires scan.axes")
        return
    for axis_key, axis_spec in axes.items():
        if not isinstance(axis_spec, dict):
            plan.errors.append(f"scan.axes.{axis_key} 必须是 mapping")
            continue
        kind = axis_spec.get("kind", "parameter")
        if kind == "composite":
            values = axis_spec.get("values", {})
            if not isinstance(values, dict):
                plan.errors.append(f"scan.axes.{axis_key}.values 必须是 mapping")
                continue
            for value_key, overrides in values.items():
                if not isinstance(overrides, dict):
                    plan.errors.append(f"scan.axes.{axis_key}.values.{value_key} 必须是 override mapping")
                    continue
                for override_path in overrides:
                    if override_path not in parameter_specs:
                        plan.errors.append(f"scan axis '{axis_key}' 引用了未注册参数路径: {override_path}")
            continue
        path = axis_spec.get("path", axis_key)
        if path not in {"alpha", "max_steps"} and path not in parameter_specs:
            plan.errors.append(f"scan axis '{axis_key}' 引用了未注册参数路径: {path}")


def _validate_scan_execution_options(plan: ExperimentPlan) -> None:
    raw_scan = plan.raw_config.get("scan", {}) if isinstance(plan.raw_config, dict) else {}
    raw_execution = raw_scan.get("execution", {}) if isinstance(raw_scan, dict) else {}
    if raw_execution in ({}, None):
        return
    if not isinstance(raw_execution, dict):
        plan.errors.append("scan.execution 必须是 mapping")
        return
    allowed_keys = {
        "max_allocated_gb",
        "target_utilization",
        "device_hard_stop_gb",
        "allowed_fold_axes",
        "auto_rebatch",
    }
    for key in raw_execution:
        if key not in allowed_keys:
            plan.errors.append(f"未注册 scan.execution 字段: scan.execution.{key}")
    numeric_keys = {"max_allocated_gb", "target_utilization", "device_hard_stop_gb"}
    for key in numeric_keys & set(raw_execution):
        value = raw_execution[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            plan.errors.append(f"scan.execution.{key} 类型无效: expected float, got {type(value).__name__}")
    if "allowed_fold_axes" in raw_execution and not isinstance(raw_execution["allowed_fold_axes"], list):
        plan.errors.append("scan.execution.allowed_fold_axes 类型无效: expected list")


def _validate_scan_continuation_options(plan: ExperimentPlan) -> None:
    raw_scan = plan.raw_config.get("scan", {}) if isinstance(plan.raw_config, dict) else {}
    if not raw_scan and isinstance(getattr(plan.config, "scan_spec", None), dict):
        raw_scan = getattr(plan.config, "scan_spec")
    raw_continuation = raw_scan.get("continuation", {}) if isinstance(raw_scan, dict) else {}
    if raw_continuation in ({}, None):
        return
    if not isinstance(raw_continuation, dict):
        plan.errors.append("scan.continuation 必须是 mapping")
        return
    allowed_keys = {
        "enabled",
        "axis",
        "order",
        "state_transfer",
        "observation_policy",
        "strict_state",
        "adaptive_controller_state",
    }
    for key in raw_continuation:
        if key not in allowed_keys:
            plan.errors.append(f"未注册 scan.continuation 字段: scan.continuation.{key}")
    if bool(raw_continuation.get("enabled", False)):
        axes = raw_scan.get("axes", {}) if isinstance(raw_scan, dict) else {}
        if "alpha" not in axes:
            plan.errors.append("scan.continuation.enabled=true 需要 scan.axes.alpha")
        algorithm_key = getattr(plan.config, "algorithm_key", "")
        continuation_spec = get_continuation_specs().get("alpha_descending_full_state")
        if continuation_spec is not None and algorithm_key not in continuation_spec.compatible_algorithms:
            plan.errors.append(
                "scan.continuation.enabled=true 目前不支持 algorithm="
                f"{algorithm_key!r}；支持: {continuation_spec.compatible_algorithms}"
            )
        capabilities = set(plan.algorithm_spec.capabilities) if plan.algorithm_spec is not None else set()
        if not ({"continuation_student_state", "continuation_tensor_state"} & capabilities):
            plan.errors.append(
                "scan.continuation.enabled=true 需要算法声明 continuation_student_state "
                "或 continuation_tensor_state capability"
            )
        spreading = getattr(plan.config, "spreading", None)
        if algorithm_key == "bigamp_spreading" and bool(getattr(spreading, "allow_intra_connection", False)):
            plan.errors.append(
                "scan.continuation.enabled=true 尚不支持 bigamp_spreading general graph "
                "(spreading.allow_intra_connection=true)：正式 spreading metrics 尚未接入 SuperGraphDataGeneral"
            )


def _validate_parameter_values(plan: ExperimentPlan, parameter_specs: Dict[str, ParameterSpec]) -> None:
    """Validate raw YAML values against ParameterSpec.type before runtime."""
    for path, value in _flatten_config_paths(plan.raw_config):
        if _is_canonical_scan_internal_path(path):
            continue
        spec = parameter_specs.get(path)
        if not spec:
            continue
        message = _parameter_type_error(path, value, spec.value_type)
        if message:
            plan.errors.append(message)


def _parameter_type_error(path: str, value: Any, type_spec: str) -> str:
    if type_spec.startswith("enum[") and type_spec.endswith("]"):
        allowed = [item.strip() for item in type_spec[5:-1].split(",") if item.strip()]
        if value not in allowed:
            return f"参数 {path} 取值无效: {value!r}；允许值: {allowed}"
        return ""
    if "|" in type_spec:
        errors = [
            _parameter_type_error(path, value, item.strip())
            for item in type_spec.split("|")
        ]
        return "" if any(not error for error in errors) else errors[0]
    if type_spec == "bool":
        return "" if isinstance(value, bool) else f"参数 {path} 类型无效: expected bool, got {type(value).__name__}"
    if type_spec == "int":
        return "" if isinstance(value, int) and not isinstance(value, bool) else f"参数 {path} 类型无效: expected int, got {type(value).__name__}"
    if type_spec == "float":
        return "" if isinstance(value, (int, float)) and not isinstance(value, bool) else f"参数 {path} 类型无效: expected float, got {type(value).__name__}"
    if type_spec == "str":
        return "" if isinstance(value, str) else f"参数 {path} 类型无效: expected str, got {type(value).__name__}"
    if type_spec.startswith("list"):
        return "" if isinstance(value, list) else f"参数 {path} 类型无效: expected list, got {type(value).__name__}"
    return ""


def _effective_parameter_summary(
    config: Any,
    output_options: Dict[str, Any],
    raw_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spreading = getattr(config, "spreading", None)
    teacher = getattr(config, "teacher", None)
    algorithm_params = getattr(config, "algorithm_params", None)
    training = getattr(config, "training", None)
    scan = getattr(config, "scan", None)
    raw_config = raw_config if isinstance(raw_config, dict) else {}
    return {
        "algorithm_key": getattr(config, "algorithm_key", None),
        "teacher_key": getattr(config, "teacher_key", None),
        "teacher.init_distribution": getattr(teacher, "init_distribution", None),
        "teacher.mean_scale": getattr(teacher, "mean_scale", None),
        "tensor_order": getattr(spreading, "tensor_order", None) if spreading else None,
        "spreading.f_distribution": getattr(spreading, "f_distribution", None) if spreading else None,
        "spreading.onsager_correction": getattr(spreading, "onsager_correction", None) if spreading else None,
        "spreading.seed": getattr(spreading, "seed", None) if spreading else None,
        "spreading.allow_intra_connection": getattr(spreading, "allow_intra_connection", None) if spreading else None,
        "training.samples_per_alpha": getattr(training, "samples_per_alpha", None),
        "training.max_steps": getattr(training, "max_steps", None),
        "training.num_workers": getattr(training, "num_workers", None),
        "scan.dimension": getattr(scan, "dimension", None),
        "scan.num_points": len(getattr(scan, "values", []) or []),
        "scan.continuation.enabled": _get_raw_path(raw_config, "scan.continuation.enabled"),
        "scan.continuation.axis": _get_raw_path(raw_config, "scan.continuation.axis"),
        "scan.continuation.order": _get_raw_path(raw_config, "scan.continuation.order"),
        "scan.continuation.state_transfer": _get_raw_path(raw_config, "scan.continuation.state_transfer"),
        "scan.continuation.observation_policy": _get_raw_path(raw_config, "scan.continuation.observation_policy"),
        "scan.continuation.strict_state": _get_raw_path(raw_config, "scan.continuation.strict_state"),
        "scan.continuation.adaptive_controller_state": _get_raw_path(raw_config, "scan.continuation.adaptive_controller_state"),
        "algorithm_params.damping": getattr(algorithm_params, "damping", None),
        "algorithm_params.noise_var": getattr(algorithm_params, "noise_var", None),
        "algorithm_params.use_compile": getattr(algorithm_params, "use_compile", None),
        "algorithm_params.compile_fallback_policy": getattr(algorithm_params, "compile_fallback_policy", None),
        "algorithm_params.normalization_profile": getattr(algorithm_params, "normalization_profile", None),
        "algorithm_params.precision_profile": getattr(algorithm_params, "precision_profile", None),
        "algorithm_params.precision_fallback_policy": getattr(algorithm_params, "precision_fallback_policy", None),
        "algorithm_params.use_bf16": getattr(algorithm_params, "use_bf16", None),
        "algorithm_params.dtype_fallback_policy": getattr(algorithm_params, "dtype_fallback_policy", None),
        "algorithm_params.use_tf32": getattr(algorithm_params, "use_tf32", None),
        "algorithm_params.seed_partition_policy": getattr(algorithm_params, "seed_partition_policy", None),
        "algorithm_params.init_mode": getattr(algorithm_params, "init_mode", None),
        "algorithm_params.init_overlap": getattr(algorithm_params, "init_overlap", None),
        "algorithm_params.adaptive_restart": getattr(algorithm_params, "adaptive_restart", None),
        "output.save_tensors": output_options.get("save_tensors"),
        "output.enable_heatmap": output_options.get("enable_heatmap"),
        "output.heatmap_metric": output_options.get("heatmap_metric"),
        "output.storage_mode": output_options.get("storage_mode"),
        "output.custom_plots": bool(output_options.get("plots")),
        "output.group_results.enabled": (output_options.get("group_results") or {}).get("enabled", "auto")
            if isinstance(output_options.get("group_results"), dict) else "auto",
        "output.group_results.save_tensors": (output_options.get("group_results") or {}).get("save_tensors", False)
            if isinstance(output_options.get("group_results"), dict) else False,
        "output.group_results.enable_heatmap": (output_options.get("group_results") or {}).get("enable_heatmap", False)
            if isinstance(output_options.get("group_results"), dict) else False,
        "output.group_results.write_plots": (output_options.get("group_results") or {}).get("write_plots", True)
            if isinstance(output_options.get("group_results"), dict) else True,
        "probes.count": len(_as_list(raw_config.get("probes"))),
        "analyzers.count": len(_as_list(raw_config.get("analyzers"))),
    }


def _build_resource_plan(plan: ExperimentPlan) -> None:
    if not plan.resource_spec or not plan.batching_spec:
        plan.resource_plan = {}
        return
    algorithm_params = getattr(plan.config, "algorithm_params", None)
    spreading = getattr(plan.config, "spreading", None)
    training = getattr(plan.config, "training", None)
    scan = getattr(plan.config, "scan", None)
    seed_policy_summary = _effective_seed_policy_summary(plan)
    plan.resource_plan = {
        "algorithm_key": plan.resource_spec.algorithm_key,
        "estimator_key": plan.resource_spec.estimator_key,
        "device_support": list(plan.resource_spec.device_support),
        "dtype_modes": list(plan.resource_spec.dtype_modes),
        "compile_support": plan.resource_spec.compile_support,
        "probe_support": plan.resource_spec.probe_support,
        "empty_cache_policy": plan.resource_spec.empty_cache_policy,
        "planner_layers": list(plan.batching_spec.planner_layers),
        "alpha_batching": plan.batching_spec.alpha_batching,
        "sample_batching": plan.batching_spec.sample_batching,
        "student_batching": plan.batching_spec.student_batching,
        "scan_axis_batching": plan.batching_spec.scan_axis_batching,
        "steps_reuse": plan.batching_spec.steps_reuse,
        "chunking": plan.batching_spec.chunking,
        "foldable_axes": list(plan.batching_spec.foldable_axes),
        "random_sensitive_axes": list(plan.batching_spec.random_sensitive_axes),
        "internal_batcher": plan.batching_spec.internal_batcher,
        "seed_partition_sensitive": plan.batching_spec.seed_partition_sensitive,
        "sample_range_honored": plan.batching_spec.sample_range_honored,
        "metadata_only": plan.batching_spec.metadata_only,
        "memory_model": {
            "estimator_entrypoint": plan.memory_model_spec.estimator_entrypoint if plan.memory_model_spec else None,
            "formula_basis": plan.memory_model_spec.formula_basis if plan.memory_model_spec else "",
            "tensor_components": list(plan.memory_model_spec.tensor_components) if plan.memory_model_spec else [],
            "calibration_status": plan.memory_model_spec.calibration_status if plan.memory_model_spec else "",
            "probe_required": plan.memory_model_spec.probe_required if plan.memory_model_spec else False,
            "sample_range_policy": plan.memory_model_spec.sample_range_policy if plan.memory_model_spec else "",
            "drives_execution": plan.memory_model_spec.drives_execution if plan.memory_model_spec else False,
        },
        "normalization": {
            "algorithm_key": plan.normalization_spec.algorithm_key if plan.normalization_spec else None,
            "schema_version": plan.normalization_spec.schema_version if plan.normalization_spec else None,
            "profiles": list(plan.normalization_spec.profiles) if plan.normalization_spec else [],
            "default_profile": plan.normalization_spec.default_profile if plan.normalization_spec else None,
            "requested_profile": getattr(algorithm_params, "normalization_profile", None),
            "latent_scale": plan.normalization_spec.latent_scale if plan.normalization_spec else "",
            "teacher_init_variance": plan.normalization_spec.teacher_init_variance if plan.normalization_spec else "",
            "student_init_variance": plan.normalization_spec.student_init_variance if plan.normalization_spec else "",
            "prior_precision_base": plan.normalization_spec.prior_precision_base if plan.normalization_spec else "",
            "interaction_scale": plan.normalization_spec.interaction_scale if plan.normalization_spec else "",
            "alpha_edge_scale": plan.normalization_spec.alpha_edge_scale if plan.normalization_spec else "",
            "metric_rescale_policy": plan.normalization_spec.metric_rescale_policy if plan.normalization_spec else "",
            "status": plan.normalization_spec.status if plan.normalization_spec else "",
        },
        "precision": {
            "algorithm_key": plan.precision_policy_spec.algorithm_key if plan.precision_policy_spec else None,
            "profiles": list(plan.precision_policy_spec.profiles) if plan.precision_policy_spec else [],
            "default_profile": plan.precision_policy_spec.default_profile if plan.precision_policy_spec else None,
            "requested_profile": getattr(algorithm_params, "precision_profile", None),
            "fallback_policy": getattr(algorithm_params, "precision_fallback_policy", None),
            "role_dtypes": (
                plan.precision_policy_spec.role_dtype_map(getattr(algorithm_params, "precision_profile", plan.precision_policy_spec.default_profile))
                if plan.precision_policy_spec else {}
            ),
        },
        "seed_policy": seed_policy_summary,
        "scan_plan": plan.scan_plan.to_dict() if plan.scan_plan else None,
        "scan_execution_constraints": (
            plan.scan_plan.execution_constraints.to_dict() if plan.scan_plan else {}
        ),
        "replan_safety": {
            "automatic_rebatch_allowed": seed_policy_summary["automatic_rebatch_allowed"],
            "replan_implemented": False,
            "notes": "Automatic OOM replan is not implemented; current behavior is checkpoint/resume.",
        },
        "config_effective": {
            "scan_num_points": len(getattr(scan, "values", []) or []),
            "samples_per_alpha": getattr(training, "samples_per_alpha", None),
            "max_steps": getattr(training, "max_steps", None),
            "use_compile": getattr(algorithm_params, "use_compile", None),
            "compile_fallback_policy": getattr(algorithm_params, "compile_fallback_policy", None),
            "normalization_profile": getattr(algorithm_params, "normalization_profile", None),
            "precision_profile": getattr(algorithm_params, "precision_profile", None),
            "precision_fallback_policy": getattr(algorithm_params, "precision_fallback_policy", None),
            "use_bf16": getattr(algorithm_params, "use_bf16", None),
            "dtype_fallback_policy": getattr(algorithm_params, "dtype_fallback_policy", None),
            "use_tf32": getattr(algorithm_params, "use_tf32", None),
            "seed_partition_policy": getattr(algorithm_params, "seed_partition_policy", None),
            "spreading.chunk_size": getattr(spreading, "chunk_size", None) if spreading else None,
            "spreading.tensor_order": getattr(spreading, "tensor_order", None) if spreading else None,
        },
        "notes": {
            "resource": plan.resource_spec.notes,
            "batching": plan.batching_spec.notes,
            "seed_policy": plan.seed_policy_spec.notes if plan.seed_policy_spec else "",
        },
    }
    _attach_resource_execution_preview(plan)


def _attach_resource_execution_preview(plan: ExperimentPlan) -> None:
    if plan.scan_plan is None:
        return
    if not plan.scan_plan.points:
        plan.resource_plan["resource_execution_plan_preview"] = {
            "num_groups": 0,
            "num_batches": 0,
            "num_work_items": 0,
            "preflight_errors": list(plan.scan_plan.errors),
        }
        return
    try:
        from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator
        from matrix_factorization.core.parallel.parallel_coordinator import ParallelCoordinator
        from matrix_factorization.core.parallel.resource_execution import build_scan_resource_execution_plan

        resource_execution_plan = build_scan_resource_execution_plan(
            scan_plan=plan.scan_plan,
            base_config=plan.config,
            coordinator=ParallelCoordinator(estimator=MemoryEstimator()),
            batching_spec=plan.batching_spec,
        )
    except Exception as exc:
        plan.warnings.append(f"ResourceExecutionPlan preview 构造失败: {exc}")
        return

    preview = resource_execution_plan.to_dict()
    plan.resource_plan["resource_execution_plan_preview"] = {
        "num_groups": len(preview.get("groups") or []),
        "num_batches": preview.get("num_batches", 0),
        "num_work_items": preview.get("num_work_items", 0),
        "max_estimated_allocated_gb": preview.get("max_estimated_allocated_gb", 0.0),
        "max_estimated_device_gb": preview.get("max_estimated_device_gb", 0.0),
        "dominant_stages": preview.get("dominant_stages", {}),
        "calibration_sources": preview.get("calibration_sources", []),
        "scan_execution": preview.get("scan_execution", {}),
        "preflight_errors": preview.get("preflight_errors", []),
        "fold_axis_rejections": preview.get("fold_axis_rejections", []),
    }
    for error in resource_execution_plan.preflight_errors:
        plan.errors.append(f"ResourceExecutionPlan preflight: {error}")


def _effective_seed_policy_summary(plan: ExperimentPlan) -> Dict[str, Any]:
    algorithm_params = getattr(plan.config, "algorithm_params", None)
    requested_policy = getattr(algorithm_params, "seed_partition_policy", "legacy")
    algorithm_key = getattr(plan.config, "algorithm_key", None)
    return get_effective_seed_policy_summary(algorithm_key, requested_policy)


def _select_output_specs(plan: ExperimentPlan, output_specs: Dict[str, OutputSpec]) -> None:
    selected = ["scalar_curves", "latest_display"]
    if plan.output_options.get("plots"):
        selected.append("custom_curves")
    if plan.output_options.get("enable_heatmap", True):
        algorithm_outputs = set(plan.algorithm_spec.compatible_outputs if plan.algorithm_spec else [])
        if "tensor_heatmap" in algorithm_outputs:
            selected.append("tensor_heatmap")
            selected.append("tensor_gif")
        elif "fallback_W_heatmap" in algorithm_outputs:
            selected.append("fallback_W_heatmap")
        else:
            selected.append("tensor_heatmap")
    plan.output_specs = [output_specs[key] for key in selected if key in output_specs]


def _build_output_plan(plan: ExperimentPlan) -> None:
    required_metrics = set()
    required_artifacts = set()
    output_files = []
    plot_semantics: Dict[str, Any] = {}
    for spec in plan.output_specs:
        output_files.extend(spec.output_files)
        for requirement in spec.requires:
            if requirement == "metrics_by_alpha":
                required_artifacts.add(requirement)
            elif requirement.endswith(".json") or requirement in {"overlap_matrix", "matrix_factors"}:
                required_artifacts.add(requirement)
            else:
                required_artifacts.add(requirement)
    for plot_config in plan.output_options.get("plots") or []:
        if "y" in plot_config:
            metric_key = str(plot_config["y"])
            preview, _ = _plot_query_selection_preview(plan, plot_config)
            required_metrics.add(metric_key)
            plot_semantics[metric_key] = {
                "metric_key": metric_key,
                "semantic_candidates": _semantic_candidates_for_flat_key(plan, metric_key),
                "query": {key: value for key, value in plot_config.items() if key in {"x", "y", "where", "compare", "series_by"}},
                "selection_preview": preview,
            }
            continue
        for curve_code in plot_config.get("curves", []):
            try:
                metric_key = _curve_code_to_metric_key(curve_code)
                required_metrics.add(metric_key)
                plot_semantics[curve_code] = {
                    "metric_key": metric_key,
                    "semantic_candidates": _semantic_candidates_for_flat_key(plan, metric_key),
                }
            except ValueError:
                pass
    metric_semantics = {
        metric_key: _semantic_candidates_for_flat_key(plan, metric_key)
        for metric_key in sorted(required_metrics)
    }
    artifact_semantics = _output_artifact_semantics(plan)
    plan.output_plan = OutputPlan(
        specs=[spec.key for spec in plan.output_specs],
        required_metrics=sorted(required_metrics),
        required_artifacts=sorted(required_artifacts),
        output_files=sorted(set(output_files)),
        metric_semantics=metric_semantics,
        plot_semantics=plot_semantics,
        artifact_semantics=artifact_semantics,
    )


def _select_intervention_specs(plan: ExperimentPlan, intervention_specs: Dict[str, InterventionSpec]) -> None:
    params = getattr(plan.config, "algorithm_params", None)
    selected = []
    init_mode = getattr(params, "init_mode", "random")
    selected.append("warm_start" if init_mode == "teacher" else "cold_start")
    if getattr(params, "adaptive_restart", False):
        selected.append("adaptive_restart")
    plan.intervention_specs = [intervention_specs[key] for key in selected if key in intervention_specs]


def _select_probe_specs(plan: ExperimentPlan, probe_specs: Dict[str, ProbeSpec]) -> None:
    requested = _requested_extension_keys(plan.raw_config.get("probes", []))
    for key in requested:
        if key not in probe_specs:
            plan.errors.append(f"未知 probe: {key}")
            continue
        plan.probe_specs.append(probe_specs[key])


def _select_analyzer_specs(plan: ExperimentPlan, analyzer_specs: Dict[str, AnalyzerSpec]) -> None:
    requested = _requested_extension_keys(plan.raw_config.get("analyzers", []))
    for key in requested:
        if key not in analyzer_specs:
            plan.errors.append(f"未知 analyzer: {key}")
            continue
        plan.analyzer_specs.append(analyzer_specs[key])


def _requested_extension_keys(value: Any) -> List[str]:
    keys = []
    for item in _as_list(value):
        if isinstance(item, str):
            keys.append(item)
        elif isinstance(item, dict):
            key = item.get("key") or item.get("type")
            if key:
                keys.append(str(key))
    return keys


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _validate_output_compatibility(plan: ExperimentPlan) -> None:
    if not plan.algorithm_spec:
        return
    compatible = set(plan.algorithm_spec.compatible_outputs)
    for spec in plan.output_specs:
        if spec.key in {"scalar_curves", "latest_display", "custom_curves"}:
            continue
        if spec.key not in compatible:
            plan.errors.append(
                f"output '{spec.key}' 需要 {spec.requires}，但 algorithm "
                f"'{plan.algorithm_spec.key}' 未声明兼容该 output。"
            )
    if plan.output_options.get("heatmap_metric") == "Q_W":
        has_factor_heatmap = "fallback_W_heatmap" in compatible or "tensor_heatmap" in compatible
        if not has_factor_heatmap:
            plan.errors.append("heatmap_metric=Q_W 但当前 algorithm 没有 factor heatmap contract。")


def _validate_custom_plot_metrics(plan: ExperimentPlan) -> None:
    plots = plan.output_options.get("plots") or []
    if not plots or not plan.algorithm_spec:
        return
    available = set(_available_metric_keys(plan))
    for plot_idx, plot_config in enumerate(plots, start=1):
        if "y" in plot_config:
            required_key = str(plot_config["y"])
            if required_key not in available:
                plan.errors.append(
                    f"output.plots[{plot_idx}] 请求 {required_key}，"
                    f"但 algorithm '{plan.algorithm_spec.key}' 的 MetricSpec 未声明该 flat key。"
                )
            _validate_plot_query_axes(plan, plot_idx, plot_config)
            _, preview_errors = _plot_query_selection_preview(plan, plot_config)
            for error in preview_errors:
                plan.errors.append(f"output.plots[{plot_idx}] {error}")
            continue
        for curve_code in plot_config.get("curves", []):
            try:
                required_key = _curve_code_to_metric_key(curve_code)
            except ValueError as exc:
                plan.errors.append(f"output.plots[{plot_idx}] 曲线代码无效: {exc}")
                continue
            if required_key not in available:
                plan.errors.append(
                    f"output.plots[{plot_idx}] 请求 {curve_code} -> {required_key}，"
                    f"但 algorithm '{plan.algorithm_spec.key}' 的 MetricSpec 未声明该 flat key。"
                )
            else:
                semantics = _semantic_candidates_for_flat_key(plan, required_key)
                if semantics:
                    plan.effective_parameters[f"output.plots[{plot_idx}].{curve_code}.canonical"] = (
                        semantics[0].get("canonical_key")
                    )


def _validate_plot_query_axes(plan: ExperimentPlan, plot_idx: int, plot_config: Dict[str, Any]) -> None:
    if not plan.scan_plan:
        return
    axis_keys = {axis.key for axis in plan.scan_plan.axes}
    x_axis = plot_config.get("x")
    if x_axis not in axis_keys:
        plan.errors.append(f"output.plots[{plot_idx}].x 引用了不存在的 scan axis: {x_axis}")
    for container_key in ["where"]:
        payload = plot_config.get(container_key) or {}
        if isinstance(payload, dict):
            for axis in payload:
                if axis not in axis_keys:
                    plan.errors.append(f"output.plots[{plot_idx}].{container_key} 引用了不存在的 scan axis: {axis}")
    for axis in plot_config.get("series_by") or []:
        if axis not in axis_keys:
            plan.errors.append(f"output.plots[{plot_idx}].series_by 引用了不存在的 scan axis: {axis}")
    for compare_idx, item in enumerate(plot_config.get("compare") or []):
        if not isinstance(item, dict):
            plan.errors.append(f"output.plots[{plot_idx}].compare[{compare_idx}] 必须是 mapping")
            continue
        for axis in item:
            if axis not in axis_keys:
                plan.errors.append(f"output.plots[{plot_idx}].compare[{compare_idx}] 引用了不存在的 scan axis: {axis}")


def _plot_query_selection_preview(plan: ExperimentPlan, plot_config: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    if not plan.scan_plan or "x" not in plot_config or "y" not in plot_config:
        return {}, []
    axis_keys = {axis.key for axis in plan.scan_plan.axes}
    x_axis = plot_config.get("x")
    metric_key = str(plot_config.get("y"))
    errors: List[str] = []
    if x_axis not in axis_keys:
        return {}, [f"PlotQuery x axis '{x_axis}' 不存在"]

    def validate_axes(payload: Any, context: str) -> None:
        axes = payload if isinstance(payload, list) else list((payload or {}).keys())
        for axis in axes:
            if axis not in axis_keys:
                errors.append(f"PlotQuery {context} 引用了不存在的 scan axis: {axis}")

    where = dict(plot_config.get("where") or {})
    validate_axes(where, "where")
    selected = [
        point for point in plan.scan_plan.points
        if all(point.coordinates.get(axis) == value for axis, value in where.items())
    ]
    if not selected:
        errors.append(f"PlotQuery where 没有选中任何 scan point: {where}")

    compare = list(plot_config.get("compare") or [])
    series_by = list(plot_config.get("series_by") or [])
    if compare:
        series_specs = []
        for compare_idx, item in enumerate(compare):
            if not isinstance(item, dict):
                errors.append(f"PlotQuery compare[{compare_idx}] 必须是 mapping")
                continue
            validate_axes(item, f"compare[{compare_idx}]")
            series_specs.append(dict(item))
    elif series_by:
        validate_axes(series_by, "series_by")
        seen = []
        for point in selected:
            key = tuple((axis, point.coordinates.get(axis)) for axis in series_by)
            if key not in seen:
                seen.append(key)
        series_specs = [{axis: value for axis, value in key} for key in seen]
    else:
        series_specs = [{}]

    series_preview = []
    for series_spec in series_specs:
        points = [
            point for point in selected
            if all(point.coordinates.get(axis) == value for axis, value in series_spec.items())
        ]
        if not points:
            errors.append(f"PlotQuery series {series_spec} 没有选中任何 scan point")
            continue
        point_ids = [point.point_id for point in points]
        x_values = [point.coordinates.get(x_axis) for point in points]
        seen_x: Dict[Any, str] = {}
        for point in points:
            x_value = point.coordinates.get(x_axis)
            if x_value in seen_x:
                errors.append(
                    f"PlotQuery series {series_spec} 在 x={x_value!r} 上选中了多个点: "
                    f"{seen_x[x_value]} 和 {point.point_id}；请收紧 where/compare 或增加 series_by"
                )
            seen_x[x_value] = point.point_id
        series_preview.append({
            "label": ", ".join(f"{key}={value}" for key, value in series_spec.items()) or metric_key,
            "coordinates": dict(series_spec),
            "point_ids": point_ids,
            "x_values": x_values,
        })

    return {
        "x": x_axis,
        "y": metric_key,
        "where": where,
        "selected_point_count": len(selected),
        "series_count": len(series_preview),
        "series": series_preview,
    }, errors


def _curve_code_to_metric_key(curve_code: str) -> str:
    from matrix_factorization.modules.outputs.plot_registry import parse_curve_code

    spec = parse_curve_code(curve_code)
    metric_name = f"{spec.metric_name}_replica" if spec.is_replica else spec.metric_name
    return f"{metric_name}_mean"


def _available_metric_keys(plan: ExperimentPlan) -> List[str]:
    if plan.algorithm_spec:
        keys = set(get_algorithm_metric_keys(plan.algorithm_spec.key))
    else:
        keys = set()
    for spec in plan.metric_specs:
        keys.update(spec.produces)
    return sorted(keys)


def _semantic_candidates_for_flat_key(plan: ExperimentPlan, metric_key: str) -> List[Dict[str, str]]:
    if not plan.algorithm_spec:
        return []
    from matrix_factorization.core.contracts import get_algorithm_metric_semantics

    return get_algorithm_metric_semantics(plan.algorithm_spec.key).get(metric_key, [])


def _output_artifact_semantics(plan: ExperimentPlan) -> Dict[str, Dict[str, str]]:
    specs = set(plan.output_plan.specs if plan.output_plan else [])
    # During construction plan.output_plan is not set yet, so fall back to
    # selected specs directly.
    if not specs:
        specs = {spec.key for spec in plan.output_specs}
    heatmap_metric = str(plan.output_options.get("heatmap_metric", "Q_Y") or "Q_Y").upper()
    semantics: Dict[str, Dict[str, str]] = {}
    if "tensor_heatmap" in specs or "tensor_gif" in specs:
        semantics["overlap_matrix"] = {
            "canonical_key": "replica.heatmap.teacher_and_students.matrix",
            "result_role": "diagnostic",
            "source": "algorithm overlap_matrix payload",
            "heatmap_metric": heatmap_metric,
        }
    if "fallback_W_heatmap" in specs:
        semantics["matrix_factors"] = {
            "canonical_key": "factor.W.teacher_student.gram_cosine",
            "result_role": "diagnostic",
            "source": "fallback W factor heatmap",
            "heatmap_metric": "Q_W",
        }
    return semantics


def _validate_runtime_extension_compatibility(plan: ExperimentPlan) -> None:
    if not plan.algorithm_spec:
        return
    algorithm_key = plan.algorithm_spec.key
    state_capabilities = set(plan.algorithm_spec.state_capabilities)
    artifacts_and_metrics = set(plan.algorithm_spec.produced_artifacts)
    artifacts_and_metrics.update({"metrics_by_alpha"})
    for spec in plan.intervention_specs:
        if algorithm_key not in spec.compatible_algorithms:
            plan.errors.append(f"intervention '{spec.key}' 未声明兼容 algorithm '{algorithm_key}'。")
        missing_state = sorted(set(spec.requires_state) - state_capabilities)
        if missing_state:
            plan.errors.append(
                f"intervention '{spec.key}' 需要 state {missing_state}，但 algorithm "
                f"'{algorithm_key}' 未声明这些 state_capabilities。"
            )
    for spec in plan.probe_specs:
        if algorithm_key not in spec.compatible_algorithms:
            plan.errors.append(f"probe '{spec.key}' 未声明兼容 algorithm '{algorithm_key}'。")
        if spec.runtime_status != "runtime_active":
            plan.errors.append(
                f"probe '{spec.key}' 当前是 {spec.runtime_status}，runner 尚未接入 "
                f"{spec.trigger} hook 的实际 payload。"
            )
        missing_state = sorted(set(spec.requires_state) - state_capabilities)
        if missing_state:
            plan.errors.append(
                f"probe '{spec.key}' 需要 state {missing_state}，但 algorithm "
                f"'{algorithm_key}' 未声明这些 state_capabilities。"
            )
    for spec in plan.analyzer_specs:
        if algorithm_key not in spec.compatible_algorithms:
            plan.errors.append(f"analyzer '{spec.key}' 未声明兼容 algorithm '{algorithm_key}'。")
        missing_inputs = sorted(set(spec.requires) - artifacts_and_metrics)
        if missing_inputs:
            plan.errors.append(
                f"analyzer '{spec.key}' 需要 {missing_inputs}，但当前 algorithm/result contract 未声明产出。"
            )


def _validate_seed_policy_compatibility(plan: ExperimentPlan) -> None:
    return


def _warn_for_soft_parameters(
    plan: ExperimentPlan,
    parameter_specs: Dict[str, ParameterSpec],
    strict: bool = False,
) -> None:
    raw_paths = {path for path, _ in _flatten_config_paths(plan.raw_config)}
    for path in sorted(raw_paths):
        spec = parameter_specs.get(path)
        if not spec:
            continue
        message = ""
        if spec.status == "parsed_only":
            message = f"{path} 当前是 parsed_only：会被读取或保留，但行为尚未完全硬化。"
        elif spec.status == "legacy":
            message = f"{path} 是 legacy 字段：当前主链路不应依赖它。"
        elif spec.status == "deprecated":
            message = f"{path} 已 deprecated。"
        if not message:
            continue
        if strict:
            plan.errors.append(f"strict mode: {message}")
        else:
            plan.warnings.append(message)


def _warn_for_parameter_consumption(plan: ExperimentPlan, strict: bool = False) -> None:
    for item in plan.parameter_chain():
        if not item.get("registered"):
            continue
        if item.get("status") != "active":
            continue
        status = item.get("consumption_status")
        path = item.get("path")
        if status == "inactive_current_route":
            message = f"{path} 在当前 algorithm/scan 路由下不会生效。"
        elif status == "declared_no_effective_trace":
            message = f"{path} 已注册并声明 consumer，但当前 plan 找不到 effective trace。"
        else:
            continue
        if strict:
            plan.errors.append(f"strict mode: {message}")
        else:
            plan.warnings.append(message)


def _warn_for_route_overrides(plan: ExperimentPlan) -> None:
    raw_tensor_order = plan.raw_config.get("tensor_order") if isinstance(plan.raw_config, dict) else None
    raw_algorithm = plan.raw_config.get("algorithm") if isinstance(plan.raw_config, dict) else None
    algorithm_key = getattr(plan.config, "algorithm_key", None)
    if raw_tensor_order is not None and raw_tensor_order >= 3 and algorithm_key == "bigamp_tensor_parallel":
        plan.warnings.append(
            f"tensor_order={raw_tensor_order} 会强制路由到 bigamp_tensor_parallel；"
            f"YAML algorithm={raw_algorithm} 不直接决定最终 tensor algorithm。"
        )


def _issue_records(errors: List[str], warnings: List[str]) -> List[Dict[str, str]]:
    return [
        {"severity": "error", "code": _issue_code(message), "message": message}
        for message in errors
    ] + [
        {"severity": "warning", "code": _issue_code(message), "message": message}
        for message in warnings
    ]


def _issue_code(message: str) -> str:
    if "未知 YAML 字段" in message:
        return "UNKNOWN_YAML_FIELD"
    if "未注册参数字段" in message:
        return "UNREGISTERED_PARAMETER_FIELD"
    if "取值无效" in message or "类型无效" in message:
        return "INVALID_PARAMETER_VALUE"
    if "requires unknown ParameterSpec path" in message:
        return "REQUIRED_PARAMETER_UNKNOWN"
    if "requires" in message and "当前 plan 找不到 effective trace" in message:
        return "REQUIRED_PARAMETER_UNTRACED"
    if "requires" in message and "当前路由下不生效" in message:
        return "REQUIRED_PARAMETER_INACTIVE"
    if "requires" in message and "effective value is None" in message:
        return "REQUIRED_PARAMETER_NONE"
    if "未注册 AlgorithmSpec" in message:
        return "UNKNOWN_ALGORITHM_SPEC"
    if "未注册 TeacherSpec" in message:
        return "UNKNOWN_TEACHER_SPEC"
    if "主链路默认只允许 active algorithm" in message:
        return "INACTIVE_ALGORITHM"
    if "Output contract missing required metric" in message or "请求" in message and "MetricSpec 未声明" in message:
        return "OUTPUT_METRIC_UNAVAILABLE"
    if "output '" in message and "未声明兼容" in message:
        return "OUTPUT_UNSUPPORTED"
    if "曲线代码无效" in message:
        return "INVALID_PLOT_CURVE_CODE"
    if "intervention '" in message and "未声明兼容" in message:
        return "INTERVENTION_UNSUPPORTED"
    if "intervention '" in message and "需要 state" in message:
        return "INTERVENTION_STATE_UNAVAILABLE"
    if "probe '" in message and "未声明兼容" in message:
        return "PROBE_UNSUPPORTED"
    if "probe '" in message and "declared_only" in message:
        return "PROBE_DECLARED_ONLY"
    if "probe '" in message and "需要 state" in message:
        return "PROBE_STATE_UNAVAILABLE"
    if "analyzer '" in message and "未声明兼容" in message:
        return "ANALYZER_UNSUPPORTED"
    if "analyzer '" in message and "需要" in message:
        return "ANALYZER_INPUT_UNAVAILABLE"
    if "strict mode:" in message and "parsed_only" in message:
        return "STRICT_PARSED_ONLY_FIELD"
    if "strict mode:" in message and "legacy" in message:
        return "STRICT_LEGACY_FIELD"
    if "strict mode:" in message and "当前 algorithm/scan 路由下不会生效" in message:
        return "STRICT_INACTIVE_PARAMETER_CURRENT_ROUTE"
    if "strict mode:" in message and "找不到 effective trace" in message:
        return "STRICT_UNTRACED_PARAMETER_EFFECT"
    if "当前 algorithm/scan 路由下不会生效" in message:
        return "INACTIVE_PARAMETER_CURRENT_ROUTE"
    if "找不到 effective trace" in message:
        return "UNTRACED_PARAMETER_EFFECT"
    if "parsed_only" in message:
        return "PARSED_ONLY_FIELD"
    if "legacy 字段" in message:
        return "LEGACY_FIELD"
    if "tensor_order=" in message and "强制路由" in message:
        return "ALGORITHM_ROUTE_OVERRIDE"
    return "VALIDATION_ISSUE"
