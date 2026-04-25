"""Research trial contracts and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from matrix_factorization.core.contracts import TrialSpec
from matrix_factorization.core.planning import ExperimentPlan, build_experiment_plan


VALID_TRIAL_STATUSES = {"draft", "active", "promoted", "archived", "abandoned"}
VALID_RUNTIME_CLASSES = {"quick", "medium", "gpu_heavy"}
IGNORED_ARTIFACT_PREFIXES = (
    Path("runs") / "trials",
    Path("artifacts") / "trials",
    Path("results") / "trials",
)


@dataclass
class TrialPlan:
    spec: TrialSpec
    manifest_path: Path
    config_path: Path
    output_root: Path
    experiment_plan: Optional[ExperimentPlan] = None
    registry_record: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors and (self.experiment_plan is None or self.experiment_plan.is_valid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.spec.key,
            "status": self.spec.status,
            "runtime_class": self.spec.runtime_class,
            "purpose": self.spec.purpose,
            "owner_area": self.spec.owner_area,
            "manifest_path": _repo_display_path(self.manifest_path),
            "config_path": _repo_display_path(self.config_path),
            "artifact_policy": self.spec.artifact_policy,
            "output_root": _repo_display_path(self.output_root),
            "expected_metrics": list(self.spec.expected_metrics),
            "promotion_target": list(self.spec.promotion_target),
            "algorithm": (
                self.experiment_plan.effective_parameters.get("algorithm_key")
                if self.experiment_plan else None
            ),
            "available_metric_keys": (
                self.experiment_plan.to_dict().get("available_metric_keys", [])
                if self.experiment_plan else []
            ),
            "experiment_plan": (
                self.experiment_plan.to_dict()
                if self.experiment_plan else None
            ),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "is_valid": self.is_valid,
        }

    def format_validation(self) -> str:
        lines = ["trial 校验结果"]
        lines.append(f"  key: {self.spec.key}")
        lines.append(f"  runtime_class: {self.spec.runtime_class}")
        lines.append(f"  config: {_repo_display_path(self.config_path)}")
        lines.append(f"  output_root: {_repo_display_path(self.output_root)}")
        if self.errors:
            lines.append("  errors:")
            lines.extend(f"    - {item}" for item in self.errors)
        elif self.experiment_plan and self.experiment_plan.errors:
            lines.append("  errors:")
            lines.extend(f"    - config: {item}" for item in self.experiment_plan.errors)
        else:
            lines.append("  errors: none")
        warnings = list(self.warnings)
        if self.experiment_plan:
            warnings.extend(self.experiment_plan.warnings)
        if warnings:
            lines.append("  warnings:")
            lines.extend(f"    - {item}" for item in warnings)
        else:
            lines.append("  warnings: none")
        return "\n".join(lines)

    def format_explain(self) -> str:
        lines = ["trial 配置解释", ""]
        lines.append(f"key: {self.spec.key}")
        lines.append(f"status: {self.spec.status}")
        lines.append(f"runtime_class: {self.spec.runtime_class}")
        lines.append(f"purpose: {self.spec.purpose or '<none>'}")
        lines.append(f"owner_area: {self.spec.owner_area or '<none>'}")
        lines.append(f"manifest: {_repo_display_path(self.manifest_path)}")
        lines.append(f"config: {_repo_display_path(self.config_path)}")
        lines.append(f"output_root: {_repo_display_path(self.output_root)}")
        if self.experiment_plan:
            payload = self.experiment_plan.to_dict()
            lines.append("")
            lines.append(f"实际 algorithm: {payload.get('algorithm')}")
            lines.append(f"scan points: {self.experiment_plan.effective_parameters.get('scan.num_points')}")
            lines.append(f"available metrics: {', '.join(payload.get('available_metric_keys', [])) or 'none'}")
            lines.append(f"expected metrics: {', '.join(self.spec.expected_metrics) or 'none'}")
            lines.append(f"output specs: {', '.join(payload.get('outputs', [])) or 'none'}")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  - {item}" for item in self.warnings)
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            lines.extend(f"  - {item}" for item in self.errors)
        return "\n".join(lines)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_trial_registry(root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    root = Path(root or repo_root())
    registry_path = root / "trials" / "registry.yaml"
    if not registry_path.exists():
        return {}
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    records = {}
    for item in data.get("trials", []):
        if isinstance(item, dict) and item.get("key"):
            records[str(item["key"])] = dict(item)
    return records


def list_trial_plans(root: Optional[Path] = None) -> List[TrialPlan]:
    root = Path(root or repo_root())
    plans = []
    for key in sorted(load_trial_registry(root)):
        plans.append(build_trial_plan(key, root=root))
    return plans


def load_trial_spec(ref: str, root: Optional[Path] = None) -> Tuple[TrialSpec, Path, Optional[Dict[str, Any]]]:
    root = Path(root or repo_root())
    manifest_path, registry_record = resolve_trial_manifest(ref, root=root)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    key = str(data.get("key") or (registry_record or {}).get("key") or manifest_path.parent.name)
    runtime_class = str(data.get("runtime_class", data.get("expected_runtime", "quick")))
    config_path = str(data.get("config", data.get("config_path", f"trials/active/{key}/config.yaml")))
    output_root = str(data.get("output_root", f"runs/trials/{key}"))
    spec = TrialSpec(
        key=key,
        status=str(data.get("status", "draft")),
        runtime_class=runtime_class,
        purpose=str(data.get("purpose", "")),
        owner_area=str(data.get("owner_area", "")),
        entrypoint=str(data.get("entrypoint", "")),
        config_path=config_path,
        output_root=output_root,
        artifact_policy=str(data.get("artifact_policy", "ignored_workspace")),
        promotion_target=_as_str_list(data.get("promotion_target", [])),
        allowed_outputs=_as_str_list(data.get("allowed_outputs", [])),
        artifact_roots=_as_str_list(data.get("artifact_roots", [])),
        expected_metrics=_as_str_list(data.get("expected_metrics", [])),
        notes=str(data.get("notes", "")),
    )
    return spec, manifest_path, registry_record


def resolve_trial_manifest(ref: str, root: Optional[Path] = None) -> Tuple[Path, Optional[Dict[str, Any]]]:
    root = Path(root or repo_root())
    candidate = Path(ref)
    if candidate.exists():
        return candidate.resolve(), None
    if not candidate.is_absolute() and (root / candidate).exists():
        return (root / candidate).resolve(), None

    registry = load_trial_registry(root)
    if ref not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise KeyError(f"Unknown trial '{ref}'. Available: {available}")
    record = registry[ref]
    manifest = root / str(record.get("path", f"trials/active/{ref}/trial.yaml"))
    if not manifest.exists():
        raise FileNotFoundError(f"Trial registry points to missing manifest: {manifest}")
    return manifest.resolve(), record


def build_trial_plan(ref: str, root: Optional[Path] = None) -> TrialPlan:
    root = Path(root or repo_root()).resolve()
    try:
        spec, manifest_path, registry_record = load_trial_spec(ref, root=root)
    except (FileNotFoundError, KeyError, OSError, yaml.YAMLError) as exc:
        placeholder = TrialSpec(key=str(ref), status="unknown")
        return TrialPlan(
            spec=placeholder,
            manifest_path=(root / "trials" / "missing.yaml"),
            config_path=(root / "trials" / "missing_config.yaml"),
            output_root=(root / "runs" / "trials" / str(ref)),
            errors=[str(exc)],
        )

    config_path = _resolve_repo_path(spec.config_path, root)
    output_root = _resolve_repo_path(spec.output_root or f"runs/trials/{spec.key}", root)
    plan = TrialPlan(
        spec=spec,
        manifest_path=manifest_path,
        config_path=config_path,
        output_root=output_root,
        registry_record=registry_record,
    )
    _validate_manifest_contract(plan, root)
    _attach_experiment_plan(plan)
    _validate_expected_metrics(plan)
    return plan


def trial_covered_paths(plan: TrialPlan) -> List[Path]:
    covered = {
        plan.manifest_path.resolve(),
        plan.config_path.resolve(),
        (plan.manifest_path.parent / "summary.md").resolve(),
    }
    if plan.spec.entrypoint:
        covered.add(_resolve_repo_path(plan.spec.entrypoint, repo_root()).resolve())
    return sorted(covered)


def _validate_manifest_contract(plan: TrialPlan, root: Path) -> None:
    spec = plan.spec
    if spec.status not in VALID_TRIAL_STATUSES:
        plan.errors.append(f"trial status 无效: {spec.status}")
    if spec.runtime_class not in VALID_RUNTIME_CLASSES:
        plan.errors.append(f"runtime_class 无效: {spec.runtime_class}")
    if spec.artifact_policy != "ignored_workspace":
        plan.errors.append("artifact_policy 必须是 ignored_workspace")
    if plan.registry_record:
        if plan.registry_record.get("status") and plan.registry_record["status"] != spec.status:
            plan.errors.append("registry.yaml 与 trial.yaml 的 status 不一致")
        if plan.registry_record.get("path"):
            expected = (root / str(plan.registry_record["path"])).resolve()
            if expected != plan.manifest_path.resolve():
                plan.errors.append("registry.yaml 与 trial manifest path 不一致")
    if spec.status == "active":
        expected_config = (root / "trials" / "active" / spec.key / "config.yaml").resolve()
        if plan.config_path.resolve() != expected_config:
            plan.errors.append("active trial 的 config 必须位于 trials/active/<trial_key>/config.yaml")
    if not plan.config_path.exists():
        plan.errors.append(f"trial config 不存在: {_repo_display_path(plan.config_path)}")
    if not _is_ignored_artifact_workspace(plan.output_root, root):
        plan.errors.append("output_root 必须位于 runs/trials、artifacts/trials 或 results/trials 下")
    for root_text in spec.artifact_roots:
        artifact_root = _resolve_repo_path(root_text, root)
        if not _is_ignored_artifact_workspace(artifact_root, root):
            plan.errors.append(f"artifact_roots 包含非 ignored workspace: {root_text}")


def _attach_experiment_plan(plan: TrialPlan) -> None:
    if plan.errors or not plan.config_path.exists():
        return
    from matrix_factorization.cli import load_yaml_config
    from matrix_factorization.core.experiment.config import ExperimentConfig

    config, output_options, raw_yaml = load_yaml_config(plan.config_path)
    if not isinstance(config, ExperimentConfig):
        plan.errors.append("trial v1 只支持直接 ExperimentConfig，不支持 nested/hysteresis trial")
        return
    plan.experiment_plan = build_experiment_plan(
        config,
        output_options=output_options,
        raw_yaml=raw_yaml,
        config_path=plan.config_path,
    )
    if plan.experiment_plan.errors:
        plan.errors.extend(f"config: {item}" for item in plan.experiment_plan.errors)


def _validate_expected_metrics(plan: TrialPlan) -> None:
    if not plan.experiment_plan:
        return
    available = set(plan.experiment_plan.to_dict().get("available_metric_keys", []))
    if plan.experiment_plan.algorithm_spec:
        available.update(plan.experiment_plan.algorithm_spec.produced_artifacts)
    missing = sorted(set(plan.spec.expected_metrics) - available)
    if missing:
        plan.errors.append(f"expected_metrics 未由当前 algorithm contract 声明: {missing}")


def _resolve_repo_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _is_ignored_artifact_workspace(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    parts = rel.parts
    return len(parts) >= 2 and Path(parts[0]) / parts[1] in IGNORED_ARTIFACT_PREFIXES


def _repo_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root().resolve()))
    except ValueError:
        return str(path)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
