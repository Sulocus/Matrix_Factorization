"""Resource execution plan wrappers built from ScanPlan and ExecutionPlan.

The existing runner still executes algorithm batches through the stable
AlgorithmResult path.  This module makes the logical work items explicit so
batching, scan axes, and plotting groups can be validated before deeper
execution refactors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any, Dict, List, Optional, Tuple

from matrix_factorization.core.scan_planning import ScanPlan, ScanPoint
from matrix_factorization.core.distributions import F_DISTRIBUTION_ISING
from matrix_factorization.core.parallel.execution_modes import AllocationConfig, EstimationParams


@dataclass(frozen=True)
class WorkItem:
    scan_point_id: str
    axis_values: Dict[str, Any]
    alpha: Optional[float]
    sample_range: Tuple[int, int]
    student_range: Optional[Tuple[int, int]] = None
    algorithm_key: str = ""
    output_group_id: str = "default"
    seed_scope: str = "scan_point"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_point_id": self.scan_point_id,
            "axis_values": dict(self.axis_values),
            "alpha": self.alpha,
            "sample_range": [int(self.sample_range[0]), int(self.sample_range[1])],
            "student_range": (
                [int(self.student_range[0]), int(self.student_range[1])]
                if self.student_range is not None else None
            ),
            "algorithm_key": self.algorithm_key,
            "output_group_id": self.output_group_id,
            "seed_scope": self.seed_scope,
        }


@dataclass(frozen=True)
class ResourceBatch:
    batch_index: int
    work_items: List[WorkItem] = field(default_factory=list)
    batch_axes: List[str] = field(default_factory=list)
    group_id: str = "default"
    memory_estimate_gb: float = 0.0
    raw_peak_allocated_gb: float = 0.0
    device_peak_gb: float = 0.0
    dominant_stage: str = ""
    confidence: float = 0.0
    memory_breakdown: Dict[str, float] = field(default_factory=dict)
    persistent_tensors: Dict[str, float] = field(default_factory=dict)
    transient_peak_tensors: Dict[str, float] = field(default_factory=dict)
    calibration_source: str = "theory_unchecked"
    seed_partition_policy: str = "legacy"
    batching_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_index": int(self.batch_index),
            "work_items": [item.to_dict() for item in self.work_items],
            "batch_axes": list(self.batch_axes),
            "group_id": self.group_id,
            "memory_estimate_gb": float(self.memory_estimate_gb),
            "estimated_allocated_gb": float(self.raw_peak_allocated_gb or self.memory_estimate_gb),
            "estimated_device_gb": float(self.device_peak_gb or self.memory_estimate_gb),
            "dominant_stage": self.dominant_stage,
            "confidence": float(self.confidence),
            "memory_breakdown": dict(self.memory_breakdown),
            "persistent_tensors": dict(self.persistent_tensors),
            "transient_peak_tensors": dict(self.transient_peak_tensors),
            "calibration_source": self.calibration_source,
            "seed_partition_policy": self.seed_partition_policy,
            "batching_metadata": copy.deepcopy(self.batching_metadata),
        }


@dataclass(frozen=True)
class ResourceGroup:
    group_id: str
    coordinates: Dict[str, Any] = field(default_factory=dict)
    point_ids: List[str] = field(default_factory=list)
    effective_overrides: Dict[str, Any] = field(default_factory=dict)
    estimation_params: Dict[str, Any] = field(default_factory=dict)
    preflight_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "coordinates": dict(self.coordinates),
            "point_ids": list(self.point_ids),
            "effective_overrides": dict(self.effective_overrides),
            "estimation_params": dict(self.estimation_params),
            "preflight_errors": list(self.preflight_errors),
        }


@dataclass(frozen=True)
class ScanResourceOptions:
    max_allocated_gb: float = 24.0
    target_utilization: float = 0.75
    device_hard_stop_gb: float = 30.0
    allowed_fold_axes: List[str] = field(default_factory=lambda: ["alpha"])
    auto_rebatch: str = "preflight_only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_allocated_gb": float(self.max_allocated_gb),
            "target_utilization": float(self.target_utilization),
            "device_hard_stop_gb": float(self.device_hard_stop_gb),
            "allowed_fold_axes": list(self.allowed_fold_axes),
            "auto_rebatch": self.auto_rebatch,
        }


@dataclass(frozen=True)
class ResourceExecutionPlan:
    algorithm_key: str
    scan_kind: str
    groups: List[ResourceGroup] = field(default_factory=list)
    batches: List[ResourceBatch] = field(default_factory=list)
    plot_grouping: List[Dict[str, Any]] = field(default_factory=list)
    seed_partition_policy: str = "legacy"
    sample_range_honored: bool = False
    metadata_only: bool = True
    scan_execution: Dict[str, Any] = field(default_factory=dict)
    max_estimated_allocated_gb: float = 0.0
    max_estimated_device_gb: float = 0.0
    dominant_stages: Dict[str, int] = field(default_factory=dict)
    calibration_sources: List[str] = field(default_factory=list)
    preflight_errors: List[str] = field(default_factory=list)
    fold_axis_rejections: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def num_batches(self) -> int:
        return len(self.batches)

    @property
    def num_work_items(self) -> int:
        return sum(len(batch.work_items) for batch in self.batches)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_key": self.algorithm_key,
            "scan_kind": self.scan_kind,
            "groups": [group.to_dict() for group in self.groups],
            "num_batches": self.num_batches,
            "num_work_items": self.num_work_items,
            "batches": [batch.to_dict() for batch in self.batches],
            "plot_grouping": list(self.plot_grouping),
            "seed_partition_policy": self.seed_partition_policy,
            "sample_range_honored": self.sample_range_honored,
            "metadata_only": self.metadata_only,
            "scan_execution": dict(self.scan_execution),
            "max_estimated_allocated_gb": float(self.max_estimated_allocated_gb),
            "max_estimated_device_gb": float(self.max_estimated_device_gb),
            "dominant_stages": dict(self.dominant_stages),
            "calibration_sources": list(self.calibration_sources),
            "preflight_errors": list(self.preflight_errors),
            "fold_axis_rejections": list(self.fold_axis_rejections),
            "notes": self.notes,
        }


def build_scan_resource_execution_plan(
    *,
    scan_plan: ScanPlan,
    base_config: Any,
    coordinator: Any,
    batching_spec: Any = None,
) -> ResourceExecutionPlan:
    """Build a real scan-aware ResourceExecutionPlan from canonical ScanPlan.

    The planner uses one effective config per non-alpha group.  It only folds
    alpha values inside that group; max_steps points and all other axes stay
    isolated unless their contract later declares a real executable folding path.
    """
    options, option_errors = scan_execution_options_from_config(base_config)
    fold_errors = _validate_allowed_fold_axes(options, batching_spec)
    sample_range_honored = bool(getattr(batching_spec, "sample_range_honored", False))
    metadata_only = bool(getattr(batching_spec, "metadata_only", True))
    seed_policy = getattr(getattr(base_config, "algorithm_params", None), "seed_partition_policy", "legacy")
    planner = _coordinator_for_options(coordinator, options)
    points_by_id = {point.point_id: point for point in scan_plan.points}

    groups: List[ResourceGroup] = []
    batches: List[ResourceBatch] = []
    preflight_errors = list(option_errors) + list(fold_errors)
    batch_index = 0

    for group in scan_plan.grouping:
        group_points = [points_by_id[point_id] for point_id in group.point_ids]
        if not group_points:
            continue
        group_errors: List[str] = []
        try:
            step_values = {int(point.max_steps) for point in group_points if point.max_steps is not None}
            if len(step_values) > 1:
                for point in group_points:
                    point_config = effective_config_for_scan_points(base_config, [point])
                    params = estimation_params_from_config(point_config, [float(point.alpha or point_config.algorithm_params.default_alpha)])
                    execution_plan = planner.plan_execution(params)
                    template = execution_plan.batches[0] if execution_plan.batches else None
                    resource_batch = _resource_batch_from_template(
                        batch_index=batch_index,
                        group_id=group.group_id,
                        template=template,
                        work_items=[_work_item_from_point(point, (0, params.S), base_config.algorithm_key)],
                        seed_partition_policy=getattr(execution_plan, "seed_partition_policy", seed_policy),
                    )
                    batches.append(resource_batch)
                    batch_index += 1
                representative_params = estimation_params_from_config(
                    effective_config_for_scan_points(base_config, [group_points[0]]),
                    [float(group_points[0].alpha or base_config.algorithm_params.default_alpha)],
                )
            else:
                group_config = effective_config_for_scan_points(base_config, group_points)
                alpha_values = sorted({
                    float(point.alpha)
                    for point in group_points
                    if point.alpha is not None
                })
                if not alpha_values:
                    alpha_values = [float(group_config.algorithm_params.default_alpha)]
                if "alpha" not in options.allowed_fold_axes:
                    representative_params = estimation_params_from_config(
                        effective_config_for_scan_points(base_config, [group_points[0]]),
                        [float(group_points[0].alpha or base_config.algorithm_params.default_alpha)],
                    )
                    for point in group_points:
                        point_alpha = float(point.alpha or group_config.algorithm_params.default_alpha)
                        point_config = effective_config_for_scan_points(base_config, [point])
                        point_params = estimation_params_from_config(point_config, [point_alpha])
                        execution_plan = planner.plan_execution(point_params)
                        template = execution_plan.batches[0] if execution_plan.batches else None
                        resource_batch = _resource_batch_from_template(
                            batch_index=batch_index,
                            group_id=group.group_id,
                            template=template,
                            work_items=[
                                _work_item_from_point(
                                    point,
                                    tuple(getattr(template, "sample_range", (0, point_params.S))) if template else (0, point_params.S),
                                    base_config.algorithm_key,
                                )
                            ],
                            seed_partition_policy=getattr(execution_plan, "seed_partition_policy", seed_policy),
                        )
                        batches.append(resource_batch)
                        batch_index += 1
                else:
                    params = estimation_params_from_config(group_config, alpha_values)
                    execution_plan = planner.plan_execution(params)
                    for template in execution_plan.batches:
                        template_alphas = [float(value) for value in getattr(template, "alpha_values", []) or []]
                        work_items = [
                            _work_item_from_point(point, tuple(getattr(template, "sample_range", (0, params.S))), base_config.algorithm_key)
                            for point in group_points
                            if point.alpha is None or float(point.alpha) in template_alphas
                        ]
                        if not work_items:
                            continue
                        resource_batch = _resource_batch_from_template(
                            batch_index=batch_index,
                            group_id=group.group_id,
                            template=template,
                            work_items=work_items,
                            seed_partition_policy=getattr(execution_plan, "seed_partition_policy", seed_policy),
                        )
                        batches.append(resource_batch)
                        batch_index += 1
                    representative_params = params
        except MemoryError as exc:
            message = f"group {group.group_id}: {exc}"
            group_errors.append(message)
            preflight_errors.append(message)
            representative_params = estimation_params_from_config(
                effective_config_for_scan_points(base_config, [group_points[0]]),
                [float(group_points[0].alpha or base_config.algorithm_params.default_alpha)],
            )

        groups.append(ResourceGroup(
            group_id=group.group_id,
            coordinates=dict(group.coordinates),
            point_ids=list(group.point_ids),
            effective_overrides=dict(group_points[0].overrides),
            estimation_params=representative_params.to_dict(),
            preflight_errors=group_errors,
        ))

    device_errors = [
        (
            f"batch {batch.batch_index} estimated device peak {batch.device_peak_gb:.2f}GB "
            f"> hard stop {options.device_hard_stop_gb:.2f}GB"
        )
        for batch in batches
        if batch.device_peak_gb and batch.device_peak_gb > options.device_hard_stop_gb
    ]
    preflight_errors.extend(device_errors)

    return ResourceExecutionPlan(
        algorithm_key=base_config.algorithm_key,
        scan_kind=scan_plan.scan_kind,
        groups=groups,
        batches=batches,
        plot_grouping=[group.to_dict() for group in scan_plan.grouping],
        seed_partition_policy=seed_policy,
        sample_range_honored=sample_range_honored,
        metadata_only=metadata_only,
        scan_execution=options.to_dict(),
        max_estimated_allocated_gb=max((b.raw_peak_allocated_gb for b in batches), default=0.0),
        max_estimated_device_gb=max((b.device_peak_gb for b in batches), default=0.0),
        dominant_stages=_dominant_stage_counts(batches),
        calibration_sources=sorted({b.calibration_source for b in batches if b.calibration_source}),
        preflight_errors=preflight_errors,
        fold_axis_rejections=fold_errors,
        notes=(
            "Scan-aware resource plan. Only alpha is folded inside identical "
            "non-alpha coordinates; max_steps and arbitrary parameter axes are isolated."
        ),
    )


def build_resource_execution_plan(
    *,
    scan_plan: ScanPlan,
    execution_plan: Any,
    algorithm_key: str,
    samples_per_alpha: int,
    seed_partition_policy: str,
    sample_range_honored: bool,
    metadata_only: bool,
    calibration_source: str = "theory_unchecked",
) -> ResourceExecutionPlan:
    """Attach explicit WorkItems to a runner-level execution plan."""
    if scan_plan.scan_kind in {"steps", "parameter_grid"} or any(
        getattr(point, "max_steps", None) is not None for point in scan_plan.points
    ):
        return _single_point_resource_plan(
            scan_plan=scan_plan,
            execution_plan=execution_plan,
            algorithm_key=algorithm_key,
            samples_per_alpha=samples_per_alpha,
            seed_partition_policy=seed_partition_policy,
            sample_range_honored=sample_range_honored,
            metadata_only=metadata_only,
            calibration_source=calibration_source,
        )
    if scan_plan.scan_kind in {"nested", "hysteresis"} or len(getattr(scan_plan, "grouping", []) or []) > 1:
        return _grouped_alpha_resource_plan(
            scan_plan=scan_plan,
            execution_plan=execution_plan,
            algorithm_key=algorithm_key,
            samples_per_alpha=samples_per_alpha,
            seed_partition_policy=seed_partition_policy,
            sample_range_honored=sample_range_honored,
            metadata_only=metadata_only,
            calibration_source=calibration_source,
        )

    batches: List[ResourceBatch] = []
    scan_points_by_alpha = _points_by_alpha(scan_plan)
    used_point_ids = set()

    for batch_idx, batch in enumerate(getattr(execution_plan, "batches", []) or []):
        work_items: List[WorkItem] = []
        sample_range = tuple(getattr(batch, "sample_range", (0, samples_per_alpha)))
        alpha_values = [float(value) for value in getattr(batch, "alpha_values", []) or []]
        for alpha in alpha_values:
            candidates = scan_points_by_alpha.get(alpha, [])
            if not candidates:
                candidates = [
                    ScanPoint(
                        point_id=f"alpha:{alpha}",
                        coordinates={"alpha": alpha},
                        overrides={},
                        effective_config_hash=f"legacy-alpha-{alpha}",
                        alpha=alpha,
                        output_group_id="alpha_curve",
                        seed_scope=f"alpha:{alpha}",
                    )
                ]
            for point in candidates:
                used_point_ids.add(point.point_id)
                work_items.append(_work_item_from_point(point, sample_range, algorithm_key))

        batch_axes = _batch_axes(work_items)
        resource_batch = _resource_batch_from_template(
            batch_index=batch_idx,
            group_id="default",
            template=batch,
            work_items=work_items,
            seed_partition_policy=seed_partition_policy,
        )
        batches.append(resource_batch)
        setattr(batch, "work_items", work_items)
        setattr(batch, "batch_axes", batch_axes)
        setattr(batch, "calibration_source", calibration_source)

    missing_points = [point for point in scan_plan.points if point.point_id not in used_point_ids]
    if missing_points and scan_plan.scan_kind not in {"steps", "nested", "hysteresis"}:
        # For the current alpha runner this should not happen.  For multi-axis
        # scans, legacy handlers still own execution and the plan is a contract
        # map until the full runner migration lands.
        fallback_batch = ResourceBatch(
            batch_index=len(batches),
            work_items=[
                _work_item_from_point(point, (0, samples_per_alpha), algorithm_key)
                for point in missing_points
            ],
            batch_axes=["scan_axis"],
            seed_partition_policy=seed_partition_policy,
            calibration_source=calibration_source,
        )
        batches.append(fallback_batch)

    return ResourceExecutionPlan(
        algorithm_key=algorithm_key,
        scan_kind=scan_plan.scan_kind,
        batches=batches,
        plot_grouping=[group.to_dict() for group in scan_plan.grouping],
        seed_partition_policy=seed_partition_policy,
        sample_range_honored=sample_range_honored,
        metadata_only=metadata_only,
        max_estimated_allocated_gb=max((b.raw_peak_allocated_gb for b in batches), default=0.0),
        max_estimated_device_gb=max((b.device_peak_gb for b in batches), default=0.0),
        dominant_stages=_dominant_stage_counts(batches),
        calibration_sources=sorted({b.calibration_source for b in batches if b.calibration_source}),
        notes=(
            "WorkItem-level execution metadata. Current alpha runner consumes "
            "work_items for batch alpha selection; multi-axis legacy handlers "
            "still execute through their existing loops."
        ),
    )


def _grouped_alpha_resource_plan(
    *,
    scan_plan: ScanPlan,
    execution_plan: Any,
    algorithm_key: str,
    samples_per_alpha: int,
    seed_partition_policy: str,
    sample_range_honored: bool,
    metadata_only: bool,
    calibration_source: str,
) -> ResourceExecutionPlan:
    """Build batches that alpha-fold only inside each scan output group."""
    grouped_points: Dict[str, List[ScanPoint]] = {}
    for point in scan_plan.points:
        grouped_points.setdefault(point.output_group_id, []).append(point)

    batches: List[ResourceBatch] = []
    batch_index = 0
    for group_id in sorted(grouped_points):
        points_by_alpha: Dict[float, List[ScanPoint]] = {}
        for point in grouped_points[group_id]:
            if point.alpha is None:
                continue
            points_by_alpha.setdefault(float(point.alpha), []).append(point)
        for batch in getattr(execution_plan, "batches", []) or []:
            sample_range = tuple(getattr(batch, "sample_range", (0, samples_per_alpha)))
            work_items: List[WorkItem] = []
            for alpha in [float(value) for value in getattr(batch, "alpha_values", []) or []]:
                for point in points_by_alpha.get(alpha, []):
                    work_items.append(_work_item_from_point(point, sample_range, algorithm_key))
            if not work_items:
                continue
            batches.append(_resource_batch_from_template(
                batch_index=batch_index,
                group_id=group_id,
                template=batch,
                work_items=work_items,
                seed_partition_policy=seed_partition_policy,
            ))
            batch_index += 1

    return ResourceExecutionPlan(
        algorithm_key=algorithm_key,
        scan_kind=scan_plan.scan_kind,
        batches=batches,
        plot_grouping=[group.to_dict() for group in scan_plan.grouping],
        seed_partition_policy=seed_partition_policy,
        sample_range_honored=sample_range_honored,
        metadata_only=True if scan_plan.scan_kind in {"nested", "hysteresis"} else metadata_only,
        max_estimated_allocated_gb=max((b.raw_peak_allocated_gb for b in batches), default=0.0),
        max_estimated_device_gb=max((b.device_peak_gb for b in batches), default=0.0),
        dominant_stages=_dominant_stage_counts(batches),
        calibration_sources=sorted({b.calibration_source for b in batches if b.calibration_source}),
        notes=(
            "Grouped scan resource plan. Alpha folding is allowed only inside "
            "one output_group_id; matrix-size and initialization axes are never "
            "folded together."
        ),
    )


def _single_point_resource_plan(
    *,
    scan_plan: ScanPlan,
    execution_plan: Any,
    algorithm_key: str,
    samples_per_alpha: int,
    seed_partition_policy: str,
    sample_range_honored: bool,
    metadata_only: bool,
    calibration_source: str,
) -> ResourceExecutionPlan:
    """Build one WorkItem batch per scan point for non-alpha-foldable scans."""
    template_batches = list(getattr(execution_plan, "batches", []) or [])
    template = template_batches[0] if template_batches else None
    sample_range = tuple(getattr(template, "sample_range", (0, samples_per_alpha))) if template else (0, samples_per_alpha)
    batches = [
        _resource_batch_from_template(
            batch_index=idx,
            group_id="default",
            template=template,
            work_items=[_work_item_from_point(point, sample_range, algorithm_key)],
            seed_partition_policy=seed_partition_policy,
        )
        for idx, point in enumerate(scan_plan.points)
    ]
    return ResourceExecutionPlan(
        algorithm_key=algorithm_key,
        scan_kind=scan_plan.scan_kind,
        batches=batches,
        plot_grouping=[group.to_dict() for group in scan_plan.grouping],
        seed_partition_policy=seed_partition_policy,
        sample_range_honored=sample_range_honored,
        metadata_only=metadata_only,
        max_estimated_allocated_gb=max((b.raw_peak_allocated_gb for b in batches), default=0.0),
        max_estimated_device_gb=max((b.device_peak_gb for b in batches), default=0.0),
        dominant_stages=_dominant_stage_counts(batches),
        calibration_sources=sorted({b.calibration_source for b in batches if b.calibration_source}),
        notes="Non-alpha scan resource plan; each scan point is isolated.",
    )


def _points_by_alpha(scan_plan: ScanPlan) -> Dict[float, List[ScanPoint]]:
    mapping: Dict[float, List[ScanPoint]] = {}
    for point in scan_plan.points:
        if point.alpha is None:
            continue
        mapping.setdefault(float(point.alpha), []).append(point)
    return mapping


def _work_item_from_point(point: ScanPoint, sample_range: Tuple[int, int], algorithm_key: str) -> WorkItem:
    return WorkItem(
        scan_point_id=point.point_id,
        axis_values=dict(point.axis_values),
        alpha=float(point.alpha) if point.alpha is not None else None,
        sample_range=(int(sample_range[0]), int(sample_range[1])),
        algorithm_key=algorithm_key,
        output_group_id=point.output_group_id,
        seed_scope=point.seed_scope,
    )


def _batch_axes(work_items: List[WorkItem]) -> List[str]:
    if not work_items:
        return []
    axes = []
    alpha_values = {item.alpha for item in work_items}
    if len(alpha_values) > 1:
        axes.append("alpha")
    output_groups = {item.output_group_id for item in work_items}
    if len(output_groups) > 1:
        axes.append("scan_axis")
    sample_ranges = {item.sample_range for item in work_items}
    if len(sample_ranges) > 1:
        axes.append("sample")
    return axes or ["point"]


def scan_execution_options_from_config(config: Any) -> Tuple[ScanResourceOptions, List[str]]:
    scan_spec = getattr(config, "scan_spec", None)
    raw_execution = {}
    if isinstance(scan_spec, dict):
        raw_execution = scan_spec.get("execution") or {}
    errors: List[str] = []
    if raw_execution is None:
        raw_execution = {}
    if not isinstance(raw_execution, dict):
        return ScanResourceOptions(), ["scan.execution must be a mapping"]

    def _float_field(key: str, default: float) -> float:
        value = raw_execution.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            errors.append(f"scan.execution.{key} must be numeric")
            return default

    max_allocated_gb = _float_field("max_allocated_gb", 24.0)
    target_utilization = _float_field("target_utilization", 0.75)
    device_hard_stop_gb = _float_field("device_hard_stop_gb", 30.0)
    if not (0.0 < target_utilization <= 1.0):
        errors.append("scan.execution.target_utilization must be in (0, 1]")
        target_utilization = 0.75
    if max_allocated_gb <= 0:
        errors.append("scan.execution.max_allocated_gb must be positive")
        max_allocated_gb = 24.0
    if device_hard_stop_gb <= 0:
        errors.append("scan.execution.device_hard_stop_gb must be positive")
        device_hard_stop_gb = 30.0

    allowed = raw_execution.get("allowed_fold_axes", ["alpha"])
    if not isinstance(allowed, list):
        errors.append("scan.execution.allowed_fold_axes must be a list")
        allowed = ["alpha"]
    allowed_fold_axes = [str(axis) for axis in allowed]
    auto_rebatch = str(raw_execution.get("auto_rebatch", "preflight_only"))
    return ScanResourceOptions(
        max_allocated_gb=max_allocated_gb,
        target_utilization=target_utilization,
        device_hard_stop_gb=device_hard_stop_gb,
        allowed_fold_axes=allowed_fold_axes,
        auto_rebatch=auto_rebatch,
    ), errors


def effective_config_for_scan_points(base_config: Any, points: List[ScanPoint]) -> Any:
    from matrix_factorization.core.experiment.config import ScanConfig

    config = copy.deepcopy(base_config)
    if not points:
        return config
    group_overrides = dict(points[0].overrides)
    for path, value in group_overrides.items():
        if path == "alpha":
            continue
        # A batch that contains multiple max_steps values must not silently pick
        # one step budget. The scan-aware planner emits one batch per max_steps
        # point, and this guard keeps accidental grouped calls honest.
        if path == "training.max_steps" and len({p.max_steps for p in points}) > 1:
            continue
        _set_config_path(config, path, value)

    alpha_values = sorted({float(point.alpha) for point in points if point.alpha is not None})
    step_values = sorted({int(point.max_steps) for point in points if point.max_steps is not None})
    if len(step_values) > 1:
        config.scan = ScanConfig(dimension="steps", values=step_values)
        config.training.max_steps = max(step_values)
        if alpha_values:
            config.algorithm_params.default_alpha = float(alpha_values[0])
    elif step_values and alpha_values:
        config.training.max_steps = step_values[0]
        config.scan = ScanConfig(dimension="alpha", values=alpha_values)
    elif step_values:
        config.training.max_steps = step_values[0]
        config.scan = ScanConfig(dimension="steps", values=step_values)
    else:
        if not alpha_values:
            alpha_values = [float(config.algorithm_params.default_alpha)]
        config.scan = ScanConfig(dimension="alpha", values=alpha_values)

    if len(points) == 1:
        suffix = points[0].point_id
    else:
        suffix = points[0].group_id
    config.scan_spec = {"axes": {"alpha": {"path": "alpha", "values": list(config.scan.values)}}}
    config.experiment_name = f"{base_config.experiment_name}_{suffix}".replace("|", "_").replace("=", "-")
    return config


def estimation_params_from_config(config: Any, alpha_values: Optional[List[float]] = None) -> EstimationParams:
    spreading = config.spreading
    tensor_order = getattr(spreading, "tensor_order", 2) if spreading else 2
    tensor_dims = None
    if tensor_order >= 2:
        dims = [config.matrix.N1, config.matrix.N2]
        for _ in range(2, tensor_order):
            dims.append(config.matrix.N1)
        tensor_dims = tuple(dims)
    values = alpha_values if alpha_values is not None else [float(value) for value in config.scan.values]
    precision_profile = getattr(config.algorithm_params, "precision_profile", "fast" if getattr(config.algorithm_params, "use_bf16", False) else "safe")
    try:
        from matrix_factorization.core.contracts import get_precision_policy_specs
        precision_spec = get_precision_policy_specs().get(config.algorithm_key)
        role_dtype_map = precision_spec.role_dtype_map(precision_profile) if precision_spec else {}
    except Exception:
        role_dtype_map = {}
    return EstimationParams(
        N1=config.matrix.N1,
        N2=config.matrix.N2,
        M=config.matrix.M,
        S=config.training.samples_per_alpha,
        alpha_values=[float(value) for value in values],
        algorithm_key=config.algorithm_key,
        use_compile=config.algorithm_params.use_compile,
        use_bf16=config.algorithm_params.use_bf16,
        precision_profile=precision_profile,
        role_dtype_map=role_dtype_map,
        f_distribution=getattr(spreading, "f_distribution", F_DISTRIBUTION_ISING) if spreading else F_DISTRIBUTION_ISING,
        adaptive_damping=config.algorithm_params.adaptive_damping,
        allow_intra_connection=getattr(spreading, "allow_intra_connection", False) if spreading else False,
        tensor_order=tensor_order,
        tensor_dims=tensor_dims,
        seed_partition_policy=config.algorithm_params.seed_partition_policy,
        chunk_size=getattr(spreading, "chunk_size", None) if spreading else None,
        use_metric_plateau_stop=bool(
            getattr(config.algorithm_params, "use_metric_plateau_stop", False)
        ),
    )


def _coordinator_for_options(coordinator: Any, options: ScanResourceOptions) -> Any:
    ratio = max(float(options.target_utilization), 1e-6)
    cap_before_ratio = float(options.max_allocated_gb) / ratio
    allocation = AllocationConfig(
        allocation_ratio=ratio,
        max_allocation_gb=cap_before_ratio,
        warning_threshold=0.85,
        critical_threshold=0.95,
        apply_calibration=getattr(getattr(coordinator, "config", None), "apply_calibration", True),
        safety_margin=getattr(getattr(coordinator, "config", None), "safety_margin", 1.1),
    )
    return type(coordinator)(
        estimator=getattr(coordinator, "estimator", None),
        config=allocation,
    )


def _validate_allowed_fold_axes(options: ScanResourceOptions, batching_spec: Any) -> List[str]:
    declared = set(getattr(batching_spec, "foldable_axes", []) or [])
    errors = []
    for axis in options.allowed_fold_axes:
        if axis not in declared:
            errors.append(
                f"scan.execution.allowed_fold_axes contains unsupported axis '{axis}' "
                f"for algorithm {getattr(batching_spec, 'algorithm_key', '<unknown>')}"
            )
    if "sample" in options.allowed_fold_axes and not bool(getattr(batching_spec, "sample_range_honored", False)):
        errors.append("sample folding requested but sample_range_honored=false")
    return errors


def _resource_batch_from_template(
    *,
    batch_index: int,
    group_id: str,
    template: Any,
    work_items: List[WorkItem],
    seed_partition_policy: str,
) -> ResourceBatch:
    if template is None:
        sample_range = work_items[0].sample_range if work_items else (0, 0)
        template_breakdown = {}
    else:
        sample_range = tuple(getattr(template, "sample_range", (0, 0)))
        template_breakdown = dict(getattr(template, "memory_breakdown", {}) or {})
    if template is not None:
        for item in work_items:
            if item.sample_range != sample_range:
                break
    return ResourceBatch(
        batch_index=batch_index,
        work_items=work_items,
        batch_axes=_batch_axes(work_items),
        group_id=group_id,
        memory_estimate_gb=float(getattr(template, "estimated_memory_gb", 0.0) or 0.0),
        raw_peak_allocated_gb=float(getattr(template, "raw_peak_allocated_gb", 0.0) or 0.0),
        device_peak_gb=float(getattr(template, "device_peak_gb", 0.0) or 0.0),
        dominant_stage=str(getattr(template, "dominant_stage", "") or ""),
        confidence=float(getattr(template, "confidence", 0.0) or 0.0),
        memory_breakdown=template_breakdown,
        persistent_tensors=dict(getattr(template, "persistent_tensors", {}) or {}),
        transient_peak_tensors=dict(getattr(template, "transient_peak_tensors", {}) or {}),
        calibration_source=getattr(template, "calibration_source", "theory_unchecked"),
        seed_partition_policy=seed_partition_policy,
        batching_metadata=copy.deepcopy(getattr(template, "batching_metadata", {}) or {}),
    )


def _dominant_stage_counts(batches: List[ResourceBatch]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for batch in batches:
        if not batch.dominant_stage:
            continue
        counts[batch.dominant_stage] = counts.get(batch.dominant_stage, 0) + 1
    return counts


def _set_config_path(config: Any, path: str, value: Any) -> None:
    if path.startswith("teacher_config."):
        from matrix_factorization.core.experiment.config import TeacherConfig

        if getattr(config, "teacher", None) is None:
            config.teacher = TeacherConfig()
        setattr(config.teacher, path.split(".", 1)[1], value)
        return
    target: Any = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
