"""Resource execution plan wrappers built from ScanPlan and ExecutionPlan.

The existing runner still executes algorithm batches through the stable
AlgorithmResult path.  This module makes the logical work items explicit so
batching, scan axes, and plotting groups can be validated before deeper
execution refactors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from matrix_factorization.core.scan_planning import ScanPlan, ScanPoint


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
    memory_estimate_gb: float = 0.0
    memory_breakdown: Dict[str, float] = field(default_factory=dict)
    calibration_source: str = "theory_unchecked"
    seed_partition_policy: str = "legacy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_index": int(self.batch_index),
            "work_items": [item.to_dict() for item in self.work_items],
            "batch_axes": list(self.batch_axes),
            "memory_estimate_gb": float(self.memory_estimate_gb),
            "memory_breakdown": dict(self.memory_breakdown),
            "calibration_source": self.calibration_source,
            "seed_partition_policy": self.seed_partition_policy,
        }


@dataclass(frozen=True)
class ResourceExecutionPlan:
    algorithm_key: str
    scan_kind: str
    batches: List[ResourceBatch] = field(default_factory=list)
    plot_grouping: List[Dict[str, Any]] = field(default_factory=list)
    seed_partition_policy: str = "legacy"
    sample_range_honored: bool = False
    metadata_only: bool = True
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
            "num_batches": self.num_batches,
            "num_work_items": self.num_work_items,
            "batches": [batch.to_dict() for batch in self.batches],
            "plot_grouping": list(self.plot_grouping),
            "seed_partition_policy": self.seed_partition_policy,
            "sample_range_honored": self.sample_range_honored,
            "metadata_only": self.metadata_only,
            "notes": self.notes,
        }


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
                        axis_values={"alpha": alpha},
                        alpha=alpha,
                        output_group_id="alpha_curve",
                        seed_scope=f"alpha:{alpha}",
                    )
                ]
            for point in candidates:
                used_point_ids.add(point.point_id)
                work_items.append(_work_item_from_point(point, sample_range, algorithm_key))

        batch_axes = _batch_axes(work_items)
        resource_batch = ResourceBatch(
            batch_index=batch_idx,
            work_items=work_items,
            batch_axes=batch_axes,
            memory_estimate_gb=float(getattr(batch, "estimated_memory_gb", 0.0) or 0.0),
            memory_breakdown=dict(getattr(batch, "memory_breakdown", {}) or {}),
            calibration_source=calibration_source,
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
            batches.append(ResourceBatch(
                batch_index=batch_index,
                work_items=work_items,
                batch_axes=_batch_axes(work_items),
                memory_estimate_gb=float(getattr(batch, "estimated_memory_gb", 0.0) or 0.0),
                memory_breakdown=dict(getattr(batch, "memory_breakdown", {}) or {}),
                calibration_source=calibration_source,
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
        ResourceBatch(
            batch_index=idx,
            work_items=[_work_item_from_point(point, sample_range, algorithm_key)],
            batch_axes=["point"],
            memory_estimate_gb=float(getattr(template, "estimated_memory_gb", 0.0) or 0.0),
            memory_breakdown=dict(getattr(template, "memory_breakdown", {}) or {}),
            calibration_source=calibration_source,
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
