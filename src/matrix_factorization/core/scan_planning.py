"""Unified scan planning primitives.

This module turns the current alpha / steps / nested / hysteresis config
variants into one explicit, serializable scan plan.  It is deliberately a
planning layer: it does not change algorithm equations or numeric execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ScanAxis:
    key: str
    values: List[Any]
    physical_sensitive: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "values": list(self.values),
            "physical_sensitive": self.physical_sensitive,
            "description": self.description,
        }


@dataclass(frozen=True)
class ScanPoint:
    point_id: str
    axis_values: Dict[str, Any]
    alpha: Optional[float] = None
    max_steps: Optional[int] = None
    matrix_size: Optional[Dict[str, int]] = None
    init_overlap: Optional[float] = None
    output_group_id: str = "default"
    seed_scope: str = "scan_point"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "axis_values": dict(self.axis_values),
            "alpha": self.alpha,
            "max_steps": self.max_steps,
            "matrix_size": dict(self.matrix_size or {}),
            "init_overlap": self.init_overlap,
            "output_group_id": self.output_group_id,
            "seed_scope": self.seed_scope,
        }


@dataclass(frozen=True)
class ScanGrouping:
    group_id: str
    label: str
    point_ids: List[str] = field(default_factory=list)
    plot_axes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "label": self.label,
            "point_ids": list(self.point_ids),
            "plot_axes": list(self.plot_axes),
        }


@dataclass(frozen=True)
class ScanExecutionConstraints:
    alpha_folding: bool = True
    sample_folding: bool = True
    scan_axis_folding: bool = False
    student_folding: bool = False
    steps_reuse: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_folding": self.alpha_folding,
            "sample_folding": self.sample_folding,
            "scan_axis_folding": self.scan_axis_folding,
            "student_folding": self.student_folding,
            "steps_reuse": self.steps_reuse,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ScanPlan:
    scan_kind: str
    axes: List[ScanAxis]
    points: List[ScanPoint]
    grouping: List[ScanGrouping] = field(default_factory=list)
    execution_constraints: ScanExecutionConstraints = field(default_factory=ScanExecutionConstraints)
    source: str = "experiment_config"

    @property
    def num_points(self) -> int:
        return len(self.points)

    def point_ids(self) -> List[str]:
        return [point.point_id for point in self.points]

    def alpha_values(self) -> List[float]:
        values = []
        for point in self.points:
            if point.alpha is not None and point.alpha not in values:
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
        }


def build_scan_plan(config: Any, raw_config: Optional[Dict[str, Any]] = None) -> ScanPlan:
    """Build a ScanPlan from an ExperimentConfig or current legacy scan dict."""
    raw_config = raw_config if isinstance(raw_config, dict) else {}
    raw_mode = _raw_scan_mode(raw_config)
    if raw_mode == "nested" and "nested_scan" in raw_config:
        return _build_nested_scan_plan(_legacy_nested_config_from_raw(raw_config))
    if raw_mode == "hysteresis" and "hysteresis_scan" in raw_config:
        return _build_hysteresis_scan_plan(_legacy_hysteresis_config_from_raw(raw_config))
    if isinstance(config, dict):
        mode = config.get("mode")
        if mode == "nested":
            return _build_nested_scan_plan(config)
        if mode == "hysteresis":
            return _build_hysteresis_scan_plan(config)
        raise ValueError(f"Unsupported scan dict mode: {mode}")

    scan = getattr(config, "scan", None)
    if scan is None:
        raise ValueError("config has no scan attribute")
    dimension = getattr(scan, "dimension", "alpha")
    values = list(getattr(scan, "values", []) or [])
    if dimension == "steps":
        alpha = _steps_default_alpha(config, raw_config)
        return _build_steps_scan_plan(values, alpha)
    if dimension == "alpha":
        return _build_alpha_scan_plan(values)
    return _build_parameter_grid_scan_plan(dimension, values)


def _build_alpha_scan_plan(alpha_values: List[Any]) -> ScanPlan:
    points = [
        ScanPoint(
            point_id=f"alpha:{idx}",
            axis_values={"alpha": float(alpha)},
            alpha=float(alpha),
            output_group_id="alpha_curve",
            seed_scope=f"alpha:{float(alpha)}",
        )
        for idx, alpha in enumerate(alpha_values)
    ]
    return ScanPlan(
        scan_kind="alpha",
        axes=[ScanAxis("alpha", [float(v) for v in alpha_values], True, "observation density")],
        points=points,
        grouping=[ScanGrouping("alpha_curve", "alpha curve", [p.point_id for p in points], ["alpha"])],
        execution_constraints=ScanExecutionConstraints(
            alpha_folding=True,
            sample_folding=True,
            scan_axis_folding=False,
            notes="Standard alpha scan; alpha folding is allowed when algorithm seed policy permits it.",
        ),
    )


def _build_steps_scan_plan(step_values: List[Any], alpha: float) -> ScanPlan:
    points = [
        ScanPoint(
            point_id=f"steps:{idx}",
            axis_values={"max_steps": int(step), "alpha": float(alpha)},
            alpha=float(alpha),
            max_steps=int(step),
            output_group_id="steps_curve",
            seed_scope=f"steps:{int(step)}",
        )
        for idx, step in enumerate(step_values)
    ]
    return ScanPlan(
        scan_kind="steps",
        axes=[
            ScanAxis("max_steps", [int(v) for v in step_values], False, "checkpointed step budget"),
            ScanAxis("alpha", [float(alpha)], True, "fixed observation density for steps scan"),
        ],
        points=points,
        grouping=[ScanGrouping("steps_curve", "steps curve", [p.point_id for p in points], ["max_steps"])],
        execution_constraints=ScanExecutionConstraints(
            alpha_folding=False,
            sample_folding=True,
            scan_axis_folding=False,
            steps_reuse=True,
            notes="Steps scan reuses one trajectory through increasing max_steps.",
        ),
    )


def _build_nested_scan_plan(config: Dict[str, Any]) -> ScanPlan:
    sizes = list(config.get("sizes") or [])
    alpha_values = [float(value) for value in (config.get("alpha_values") or [])]
    points: List[ScanPoint] = []
    groups: List[ScanGrouping] = []
    for size_idx, size in enumerate(sizes):
        n1, n2, m = [int(v) for v in size]
        group_id = f"size:{n1}x{n2}_M{m}"
        group_points = []
        for alpha_idx, alpha in enumerate(alpha_values):
            point = ScanPoint(
                point_id=f"nested:{size_idx}:{alpha_idx}",
                axis_values={"N1": n1, "N2": n2, "M": m, "alpha": float(alpha)},
                alpha=float(alpha),
                matrix_size={"N1": n1, "N2": n2, "M": m},
                output_group_id=group_id,
                seed_scope=f"size:{n1}:{n2}:{m}:alpha:{float(alpha)}",
            )
            points.append(point)
            group_points.append(point.point_id)
        groups.append(ScanGrouping(group_id, f"N={n1}, M={m}", group_points, ["alpha"]))
    return ScanPlan(
        scan_kind="nested",
        axes=[
            ScanAxis("matrix_size", [list(size) for size in sizes], True, "outer matrix/rank sweep"),
            ScanAxis("alpha", alpha_values, True, "inner observation density"),
        ],
        points=points,
        grouping=groups,
        execution_constraints=ScanExecutionConstraints(
            alpha_folding=True,
            sample_folding=True,
            scan_axis_folding=False,
            notes="Nested scan keeps matrix-size axis separate unless a future batching spec proves it safe.",
        ),
        source="legacy_nested_config",
    )


def _build_hysteresis_scan_plan(config: Dict[str, Any]) -> ScanPlan:
    init_overlaps = [float(value) for value in (config.get("init_overlaps") or [])]
    alpha_values = [float(value) for value in (config.get("alpha_values") or [])]
    points: List[ScanPoint] = []
    groups: List[ScanGrouping] = []
    for overlap_idx, overlap in enumerate(init_overlaps):
        group_id = "cold_start" if overlap <= 1e-6 else f"warm_start:{overlap:g}"
        label = "Cold Start" if overlap <= 1e-6 else f"Warm Start (m={overlap:g})"
        group_points = []
        for alpha_idx, alpha in enumerate(alpha_values):
            point = ScanPoint(
                point_id=f"hysteresis:{overlap_idx}:{alpha_idx}",
                axis_values={"init_overlap": overlap, "alpha": float(alpha)},
                alpha=float(alpha),
                init_overlap=overlap,
                output_group_id=group_id,
                seed_scope=f"init_overlap:{overlap}:alpha:{float(alpha)}",
            )
            points.append(point)
            group_points.append(point.point_id)
        groups.append(ScanGrouping(group_id, label, group_points, ["alpha"]))
    return ScanPlan(
        scan_kind="hysteresis",
        axes=[
            ScanAxis("init_overlap", init_overlaps, True, "initialization / intervention axis"),
            ScanAxis("alpha", alpha_values, True, "observation density"),
        ],
        points=points,
        grouping=groups,
        execution_constraints=ScanExecutionConstraints(
            alpha_folding=True,
            sample_folding=True,
            scan_axis_folding=False,
            notes="Hysteresis scan keeps initialization axis separate; init_overlap is a scan axis, not a hidden CLI mutation.",
        ),
        source="legacy_hysteresis_config",
    )


def _build_parameter_grid_scan_plan(dimension: str, values: List[Any]) -> ScanPlan:
    points = [
        ScanPoint(
            point_id=f"{dimension}:{idx}",
            axis_values={dimension: value},
            output_group_id=f"{dimension}_curve",
            seed_scope=f"{dimension}:{value}",
        )
        for idx, value in enumerate(values)
    ]
    return ScanPlan(
        scan_kind="parameter_grid",
        axes=[ScanAxis(dimension, list(values), True, f"parameter grid over {dimension}")],
        points=points,
        grouping=[ScanGrouping(f"{dimension}_curve", f"{dimension} curve", [p.point_id for p in points], [dimension])],
        execution_constraints=ScanExecutionConstraints(
            alpha_folding=False,
            sample_folding=True,
            scan_axis_folding=False,
            notes="Generic parameter grid; no folding across the scanned axis by default.",
        ),
    )


def _steps_default_alpha(config: Any, raw_config: Dict[str, Any]) -> float:
    algorithm_params = getattr(config, "algorithm_params", None)
    value = getattr(algorithm_params, "default_alpha", None)
    if value is not None:
        return float(value)
    steps = raw_config.get("steps_scan", {}) if isinstance(raw_config, dict) else {}
    return float(steps.get("alpha", 1.0)) if isinstance(steps, dict) else 1.0


def _legacy_nested_config_from_raw(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    nested = raw_config.get("nested_scan", {}) if isinstance(raw_config, dict) else {}
    sizes = nested.get("sizes", [[200, 50], [400, 100]]) if isinstance(nested, dict) else []
    normalized_sizes = []
    for size in sizes:
        if len(size) == 2:
            normalized_sizes.append((int(size[0]), int(size[0]), int(size[1])))
        elif len(size) == 3:
            normalized_sizes.append((int(size[0]), int(size[1]), int(size[2])))
    alpha_cfg = nested.get("alpha", {}) if isinstance(nested, dict) else {}
    return {
        "mode": "nested",
        "sizes": normalized_sizes,
        "alpha_values": _range_values(alpha_cfg, 0.0, 4.0, 0.1),
    }


def _legacy_hysteresis_config_from_raw(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    hyst = raw_config.get("hysteresis_scan", {}) if isinstance(raw_config, dict) else {}
    alpha_cfg = hyst.get("alpha", raw_config.get("alpha_scan", {})) if isinstance(hyst, dict) else {}
    return {
        "mode": "hysteresis",
        "init_overlaps": hyst.get("init_overlaps", [0.0, 0.95]) if isinstance(hyst, dict) else [0.0, 0.95],
        "alpha_values": _range_values(alpha_cfg, 0.0, 4.0, 0.05),
    }


def _range_values(cfg: Dict[str, Any], default_start: float, default_stop: float, default_step: float) -> List[float]:
    cfg = cfg if isinstance(cfg, dict) else {}
    start = float(cfg.get("start", default_start))
    stop = float(cfg.get("stop", default_stop))
    step = float(cfg.get("step", default_step))
    values: List[float] = []
    current = start
    epsilon = abs(step) * 1e-9 + 1e-12
    if step <= 0:
        raise ValueError("scan step must be positive")
    while current <= stop + epsilon:
        values.append(float(current))
        current += step
    return values


def _raw_scan_mode(raw_config: Dict[str, Any]) -> str:
    value = raw_config.get("scan_mode", 1) if isinstance(raw_config, dict) else 1
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
