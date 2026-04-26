"""Canonical parameter-space scan planning.

The scan system has one public shape: ``scan.axes``.  Former modes such as
alpha, steps, nested size sweeps, and hysteresis are now represented as
ordinary axes that override registered config paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
import json
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ScanAxis:
    key: str
    values: List[Any]
    path: Optional[str] = None
    kind: str = "parameter"
    physical_sensitive: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "path": self.path,
            "kind": self.kind,
            "values": list(self.values),
            "physical_sensitive": self.physical_sensitive,
            "description": self.description,
        }


@dataclass(frozen=True)
class ScanPoint:
    point_id: str
    coordinates: Dict[str, Any]
    overrides: Dict[str, Any]
    effective_config_hash: str
    physical_sensitive_paths: List[str] = field(default_factory=list)
    alpha: Optional[float] = None
    max_steps: Optional[int] = None
    matrix_size: Optional[Dict[str, int]] = None
    init_overlap: Optional[float] = None
    group_id: str = "default"
    output_group_id: str = "default"
    seed_scope: str = "scan_point"
    label: str = ""

    @property
    def axis_values(self) -> Dict[str, Any]:
        """Backward-compatible alias for resource planning metadata."""
        return dict(self.coordinates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "coordinates": dict(self.coordinates),
            "axis_values": dict(self.coordinates),
            "overrides": dict(self.overrides),
            "effective_config_hash": self.effective_config_hash,
            "physical_sensitive_paths": list(self.physical_sensitive_paths),
            "alpha": self.alpha,
            "max_steps": self.max_steps,
            "matrix_size": dict(self.matrix_size or {}),
            "init_overlap": self.init_overlap,
            "group_id": self.group_id,
            "output_group_id": self.output_group_id,
            "seed_scope": self.seed_scope,
            "label": self.label,
        }


@dataclass(frozen=True)
class ScanGrouping:
    group_id: str
    label: str
    point_ids: List[str] = field(default_factory=list)
    plot_axes: List[str] = field(default_factory=list)
    coordinates: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "label": self.label,
            "point_ids": list(self.point_ids),
            "plot_axes": list(self.plot_axes),
            "coordinates": dict(self.coordinates),
        }


@dataclass(frozen=True)
class ScanExecutionConstraints:
    alpha_folding: bool = True
    sample_folding: bool = False
    scan_axis_folding: bool = False
    student_folding: bool = False
    steps_reuse: bool = False
    foldable_axes: List[str] = field(default_factory=lambda: ["alpha"])
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_folding": self.alpha_folding,
            "sample_folding": self.sample_folding,
            "scan_axis_folding": self.scan_axis_folding,
            "student_folding": self.student_folding,
            "steps_reuse": self.steps_reuse,
            "foldable_axes": list(self.foldable_axes),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ScanPlan:
    scan_kind: str
    axes: List[ScanAxis]
    points: List[ScanPoint]
    grouping: List[ScanGrouping] = field(default_factory=list)
    execution_constraints: ScanExecutionConstraints = field(default_factory=ScanExecutionConstraints)
    source: str = "canonical_scan"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def num_points(self) -> int:
        return len(self.points)

    def point_ids(self) -> List[str]:
        return [point.point_id for point in self.points]

    def alpha_values(self) -> List[float]:
        values: List[float] = []
        for point in self.points:
            if point.alpha is not None and float(point.alpha) not in values:
                values.append(float(point.alpha))
        return values

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_kind": self.scan_kind,
            "source": self.source,
            "num_points": self.num_points,
            "axes": [axis.to_dict() for axis in self.axes],
            "points": [point.to_dict() for point in self.points],
            "grouping": [group.to_dict() for group in self.grouping],
            "execution_constraints": self.execution_constraints.to_dict(),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def build_scan_plan(config: Any, raw_config: Optional[Dict[str, Any]] = None) -> ScanPlan:
    """Build the canonical parameter-space ScanPlan."""
    scan_spec = _scan_spec_from_inputs(config, raw_config)
    if not isinstance(scan_spec, dict) or "axes" not in scan_spec:
        raise ValueError("canonical scan schema requires scan.axes")
    axes_payload = scan_spec.get("axes")
    if not isinstance(axes_payload, dict) or not axes_payload:
        raise ValueError("scan.axes must be a non-empty mapping")

    axes: List[ScanAxis] = []
    value_options: List[List[Any]] = []
    axis_specs: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    for axis_key, axis_payload in axes_payload.items():
        if not isinstance(axis_payload, dict):
            errors.append(f"scan axis '{axis_key}' must be a mapping")
            continue
        kind = str(axis_payload.get("kind", "parameter"))
        path = axis_payload.get("path")
        raw_values = axis_payload.get("values")
        values = _expand_axis_values(raw_values)
        if not values:
            errors.append(f"scan axis '{axis_key}' has no values")
        axis = ScanAxis(
            key=str(axis_key),
            path=str(path) if path is not None else None,
            kind=kind,
            values=values,
            physical_sensitive=bool(axis_payload.get("physical_sensitive", True)),
            description=str(axis_payload.get("description", "")),
        )
        axes.append(axis)
        value_options.append(values)
        axis_specs.append(axis_payload)

    if errors:
        return ScanPlan("parameter_space", axes, [], errors=errors, warnings=warnings)

    points: List[ScanPoint] = []
    for idx, combo in enumerate(product(*value_options)):
        coordinates = {axis.key: value for axis, value in zip(axes, combo)}
        overrides: Dict[str, Any] = {}
        physical_paths: List[str] = []
        for axis, axis_payload, value in zip(axes, axis_specs, combo):
            axis_overrides = _axis_value_overrides(axis, axis_payload, value)
            for path, override_value in axis_overrides.items():
                overrides[path] = override_value
                if axis.physical_sensitive:
                    physical_paths.append(path)
        alpha = _point_alpha(coordinates, overrides, config)
        max_steps = _point_max_steps(coordinates, overrides)
        matrix_size = _point_matrix_size(overrides)
        init_overlap = _point_init_overlap(overrides)
        group_coordinates = {
            key: value for key, value in coordinates.items()
            if key not in {"alpha", "max_steps"}
        }
        group_id = _group_id(group_coordinates)
        digest = _stable_hash({"coordinates": coordinates, "overrides": overrides})
        points.append(ScanPoint(
            point_id=f"p{idx:04d}",
            coordinates=coordinates,
            overrides=overrides,
            effective_config_hash=digest,
            physical_sensitive_paths=sorted(set(physical_paths)),
            alpha=alpha,
            max_steps=max_steps,
            matrix_size=matrix_size,
            init_overlap=init_overlap,
            group_id=group_id,
            output_group_id=group_id,
            seed_scope=f"{group_id}:alpha:{alpha}" if alpha is not None else group_id,
            label=", ".join(f"{key}={value}" for key, value in coordinates.items()),
        ))

    grouping = _build_grouping(points)
    steps_reuse = any(axis.key == "max_steps" or axis.path in {"max_steps", "training.max_steps"} for axis in axes)
    return ScanPlan(
        scan_kind="parameter_space",
        axes=axes,
        points=points,
        grouping=grouping,
        execution_constraints=ScanExecutionConstraints(
            alpha_folding=True,
            sample_folding=False,
            scan_axis_folding=False,
            student_folding=False,
            steps_reuse=steps_reuse,
            foldable_axes=["alpha"],
            notes="Only alpha may be folded inside identical non-alpha coordinates.",
        ),
        source="canonical_scan",
        warnings=warnings,
    )


def _scan_spec_from_inputs(config: Any, raw_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(raw_config, dict) and "scan" in raw_config:
        return raw_config["scan"]
    if isinstance(config, dict) and "scan" in config:
        return config["scan"]
    scan_spec = getattr(config, "scan_spec", None)
    if isinstance(scan_spec, dict):
        return scan_spec
    scan = getattr(config, "scan", None)
    if scan is not None:
        dimension = getattr(scan, "dimension", "alpha")
        values = list(getattr(scan, "values", []) or [])
        return {"axes": {dimension: {"path": dimension, "values": values}}}
    return {}


def _expand_axis_values(raw_values: Any) -> List[Any]:
    if isinstance(raw_values, dict) and {"start", "stop", "step"} <= set(raw_values):
        start = float(raw_values["start"])
        stop = float(raw_values["stop"])
        step = float(raw_values["step"])
        if step <= 0:
            raise ValueError("scan range step must be positive")
        values = []
        current = start
        while current <= stop + abs(step) * 1e-9:
            values.append(round(float(current), 12))
            current += step
        return values
    if isinstance(raw_values, dict):
        return list(raw_values.keys())
    if isinstance(raw_values, list):
        return list(raw_values)
    if raw_values is None:
        return []
    return [raw_values]


def _axis_value_overrides(axis: ScanAxis, axis_payload: Dict[str, Any], value: Any) -> Dict[str, Any]:
    if axis.kind == "composite":
        values = axis_payload.get("values", {})
        payload = values.get(value) if isinstance(values, dict) else None
        if not isinstance(payload, dict):
            raise ValueError(f"composite scan axis '{axis.key}' value '{value}' must map to overrides")
        return dict(payload)
    path = axis.path or axis.key
    if path == "alpha":
        return {}
    if path == "max_steps":
        return {"training.max_steps": int(value)}
    return {path: value}


def _point_alpha(coordinates: Dict[str, Any], overrides: Dict[str, Any], config: Any) -> Optional[float]:
    if "alpha" in coordinates:
        return float(coordinates["alpha"])
    default = overrides.get("algorithm_params.default_alpha")
    if default is None:
        default = getattr(getattr(config, "algorithm_params", None), "default_alpha", 1.0)
    return float(default)


def _point_max_steps(coordinates: Dict[str, Any], overrides: Dict[str, Any]) -> Optional[int]:
    if "max_steps" in coordinates:
        return int(coordinates["max_steps"])
    if "training.max_steps" in overrides:
        return int(overrides["training.max_steps"])
    return None


def _point_matrix_size(overrides: Dict[str, Any]) -> Optional[Dict[str, int]]:
    keys = {"matrix.N1", "matrix.N2", "matrix.M"}
    if not keys <= set(overrides):
        return None
    return {
        "N1": int(overrides["matrix.N1"]),
        "N2": int(overrides["matrix.N2"]),
        "M": int(overrides["matrix.M"]),
    }


def _point_init_overlap(overrides: Dict[str, Any]) -> Optional[float]:
    if "algorithm_params.init_overlap" in overrides:
        return float(overrides["algorithm_params.init_overlap"])
    if overrides.get("algorithm_params.init_mode") == "random":
        return 0.0
    return None


def _group_id(coordinates: Dict[str, Any]) -> str:
    if not coordinates:
        return "default"
    return "|".join(f"{key}={coordinates[key]}" for key in sorted(coordinates))


def _build_grouping(points: List[ScanPoint]) -> List[ScanGrouping]:
    grouped: Dict[str, List[ScanPoint]] = {}
    for point in points:
        grouped.setdefault(point.group_id, []).append(point)
    return [
        ScanGrouping(
            group_id=group_id,
            label=group_id,
            point_ids=[point.point_id for point in group_points],
            plot_axes=["alpha"] if any(point.alpha is not None for point in group_points) else [],
            coordinates={
                key: value
                for key, value in group_points[0].coordinates.items()
                if key not in {"alpha", "max_steps"}
            },
        )
        for group_id, group_points in grouped.items()
    ]


def _stable_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return sha256(text.encode("utf-8")).hexdigest()[:16]
