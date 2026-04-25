"""Build and validate an effective experiment plan before execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from .contracts import (
    AlgorithmSpec,
    InterventionSpec,
    AnalyzerSpec,
    MetricSpec,
    OutputSpec,
    ParameterSpec,
    ProbeSpec,
    TeacherSpec,
    get_algorithm_specs,
    get_algorithm_metric_keys,
    get_analyzer_specs,
    get_intervention_specs,
    get_metric_specs,
    get_output_specs,
    get_parameter_specs,
    get_probe_specs,
    get_teacher_specs,
)


@dataclass
class OutputPlan:
    specs: List[str] = field(default_factory=list)
    required_metrics: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specs": list(self.specs),
            "required_metrics": list(self.required_metrics),
            "required_artifacts": list(self.required_artifacts),
            "output_files": list(self.output_files),
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
    metric_specs: List[MetricSpec] = field(default_factory=list)
    output_specs: List[OutputSpec] = field(default_factory=list)
    output_plan: OutputPlan = field(default_factory=OutputPlan)
    intervention_specs: List[InterventionSpec] = field(default_factory=list)
    probe_specs: List[ProbeSpec] = field(default_factory=list)
    analyzer_specs: List[AnalyzerSpec] = field(default_factory=list)
    effective_parameters: Dict[str, Any] = field(default_factory=dict)
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
            "metrics": [spec.key for spec in self.metric_specs],
            "available_metric_keys": _available_metric_keys(self),
            "outputs": [spec.key for spec in self.output_specs],
            "output_plan": self.output_plan.to_dict(),
            "interventions": [spec.key for spec in self.intervention_specs],
            "probes": [spec.key for spec in self.probe_specs],
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
        available_metrics = _available_metric_keys(self)
        if available_metrics:
            lines.append("")
            lines.append("可用 flat metric keys:")
            lines.append(f"  {', '.join(available_metrics)}")
        if self.intervention_specs:
            lines.append("")
            lines.append("将使用的 intervention contract:")
            for spec in self.intervention_specs:
                lines.append(f"  - {spec.key}: trigger={spec.trigger}")
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
        for path, value in sorted(_flatten_config_paths(self.raw_config), key=lambda item: item[0]):
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

    _validate_raw_paths(plan, parameter_specs)

    algorithm_key = getattr(config, "algorithm_key", None)
    plan.effective_parameters.update(_effective_parameter_summary(config, output_options, raw_config))
    if algorithm_key in algorithm_specs:
        plan.algorithm_spec = algorithm_specs[algorithm_key]
    else:
        plan.errors.append(f"algorithm_key 未注册 AlgorithmSpec: {algorithm_key}")
        return plan

    teacher_key = getattr(config, "teacher_key", None)
    if teacher_key in teacher_specs:
        plan.teacher_spec = teacher_specs[teacher_key]
    else:
        plan.errors.append(f"teacher_key 未注册 TeacherSpec: {teacher_key}")

    _validate_required_config_paths(plan, parameter_specs)

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
    _warn_for_soft_parameters(plan, parameter_specs, strict=strict)
    _warn_for_parameter_consumption(plan, strict=strict)
    _warn_for_route_overrides(plan)
    return plan


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
        "scan_mode": ("scan.dimension", getattr(getattr(config, "scan", None), "dimension", None)),
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
        "steps_scan.alpha": (
            "algorithm_params.default_alpha",
            getattr(getattr(config, "algorithm_params", None), "default_alpha", None),
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
        key = path.split(".", 1)[1]
        if key in plan.output_options:
            return plan.output_options.get(key), f"output_options.{key}", None

    if path in {"probes", "analyzers"}:
        return raw_config.get(path), "raw_config.runtime_extensions", {
            "selected_specs": [spec.key for spec in (plan.probe_specs if path == "probes" else plan.analyzer_specs)]
        }

    if path.startswith(("alpha_scan.", "steps_scan.", "nested_scan.", "hysteresis_scan.")):
        return _get_raw_path(raw_config, path), "raw_scan_block", _derived_parameter_effect(plan, path)

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
    if path.startswith(("alpha_scan.", "steps_scan.", "nested_scan.", "hysteresis_scan.", "scan_mode")):
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
    raw_mode = _raw_scan_mode(plan.raw_config if isinstance(plan.raw_config, dict) else {})
    if path.startswith("alpha_scan."):
        return raw_mode == "alpha"
    if path.startswith("steps_scan."):
        return raw_mode == "steps"
    if path.startswith("nested_scan."):
        return raw_mode == "nested"
    if path.startswith("hysteresis_scan."):
        return raw_mode == "hysteresis"
    if path.startswith("scan."):
        return False
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
        "algorithm_params.adaptive_restart",
        "algorithm_params.restart_patience",
        "algorithm_params.restart_noise",
        "algorithm_params.acceptance_tolerance",
    }:
        return algorithm_key in {"bigamp_spreading"}
    return True


def _raw_scan_mode(raw_config: Dict[str, Any]) -> str:
    value = raw_config.get("scan_mode", 1)
    mapping = {
        1: "alpha",
        2: "steps",
        3: "nested",
        4: "hysteresis",
        "alpha": "alpha",
        "steps": "steps",
        "nested": "nested",
        "hysteresis": "hysteresis",
    }
    return mapping.get(value, str(value))


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
    for path, _ in _flatten_config_paths(plan.raw_config):
        if path in known_paths:
            continue
        top = path.split(".", 1)[0]
        if top not in known_top_level:
            plan.errors.append(f"未知 YAML 字段: {path}")
        else:
            plan.errors.append(f"未注册参数字段: {path}")


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
        "algorithm_params.damping": getattr(algorithm_params, "damping", None),
        "algorithm_params.noise_var": getattr(algorithm_params, "noise_var", None),
        "algorithm_params.use_compile": getattr(algorithm_params, "use_compile", None),
        "algorithm_params.use_bf16": getattr(algorithm_params, "use_bf16", None),
        "algorithm_params.init_mode": getattr(algorithm_params, "init_mode", None),
        "algorithm_params.init_overlap": getattr(algorithm_params, "init_overlap", None),
        "algorithm_params.adaptive_restart": getattr(algorithm_params, "adaptive_restart", None),
        "output.save_tensors": output_options.get("save_tensors"),
        "output.enable_heatmap": output_options.get("enable_heatmap"),
        "output.heatmap_metric": output_options.get("heatmap_metric"),
        "output.storage_mode": output_options.get("storage_mode"),
        "output.custom_plots": bool(output_options.get("plots")),
        "probes.count": len(_as_list(raw_config.get("probes"))),
        "analyzers.count": len(_as_list(raw_config.get("analyzers"))),
    }


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
        for curve_code in plot_config.get("curves", []):
            try:
                required_metrics.add(_curve_code_to_metric_key(curve_code))
            except ValueError:
                pass
    plan.output_plan = OutputPlan(
        specs=[spec.key for spec in plan.output_specs],
        required_metrics=sorted(required_metrics),
        required_artifacts=sorted(required_artifacts),
        output_files=sorted(set(output_files)),
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
