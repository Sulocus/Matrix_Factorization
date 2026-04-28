"""Stateful alpha continuation execution.

This module implements scan-level continuation: run high-alpha points first,
then use the final algorithm state as the initialization for the next lower
alpha.  It intentionally does not change algorithm formulas; algorithms opt in
by accepting ``initial_state`` and returning ``AlgorithmResult.continuation_state``.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from matrix_factorization.core.contracts import AlgorithmResult, AlgorithmStateView
from matrix_factorization.core.experiment.data_factory import ExperimentData
from matrix_factorization.core.experiment.result import (
    ExperimentMetadata,
    ExperimentResult,
    ResultCube,
    SingleRunResult,
)
from matrix_factorization.core.parallel.resource_execution import effective_config_for_scan_points
from matrix_factorization.core.scan_planning import ScanPlan, ScanPoint, build_scan_plan


@dataclass(frozen=True)
class ContinuationConfig:
    enabled: bool = False
    axis: str = "alpha"
    order: str = "descending"
    state_transfer: str = "full_algorithm_state"
    observation_policy: str = "nested_prefix"
    strict_state: bool = True
    adaptive_controller_state: str = "reset"

    @classmethod
    def from_scan_spec(cls, scan_spec: Optional[Dict[str, Any]]) -> "ContinuationConfig":
        if not isinstance(scan_spec, dict):
            return cls()
        raw = scan_spec.get("continuation") or {}
        if not isinstance(raw, dict):
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            axis=str(raw.get("axis", "alpha")),
            order=str(raw.get("order", "descending")),
            state_transfer=str(raw.get("state_transfer", "full_algorithm_state")),
            observation_policy=str(raw.get("observation_policy", "nested_prefix")),
            strict_state=bool(raw.get("strict_state", True)),
            adaptive_controller_state=str(raw.get("adaptive_controller_state", "reset")),
        )

    @property
    def spec_key(self) -> str:
        return "alpha_descending_full_state"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "axis": self.axis,
            "order": self.order,
            "state_transfer": self.state_transfer,
            "observation_policy": self.observation_policy,
            "strict_state": self.strict_state,
            "adaptive_controller_state": self.adaptive_controller_state,
            "spec_key": self.spec_key,
        }

    def to_scan_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "axis": self.axis,
            "order": self.order,
            "state_transfer": self.state_transfer,
            "observation_policy": self.observation_policy,
            "strict_state": self.strict_state,
            "adaptive_controller_state": self.adaptive_controller_state,
        }


def continuation_enabled(config: Any) -> bool:
    return ContinuationConfig.from_scan_spec(getattr(config, "scan_spec", None)).enabled


def run_alpha_descending_continuation(
    *,
    runner: Any,
    config: Any,
    observer: Optional[Callable[[Any], None]],
    output_options: Optional[Dict[str, Any]],
    raw_yaml: str,
) -> ExperimentResult:
    cont = ContinuationConfig.from_scan_spec(getattr(config, "scan_spec", None))
    _validate_runtime_continuation_config(cont, config)

    scan_plan = build_scan_plan(config)
    if scan_plan.errors:
        raise ValueError("invalid scan plan for continuation:\n" + "\n".join(scan_plan.errors))

    group_points = _points_by_group(scan_plan)
    total_points = sum(len(points) for points in group_points.values())
    from matrix_factorization.core.experiment.runner import ProgressEventType

    runner._emit(observer, ProgressEventType.EXECUTION_PLAN, {
        "batches": _execution_plan_batches(group_points),
        "total_batches": total_points,
        "total_points": total_points,
        "mode": "alpha_descending_continuation",
        "algorithm_key": config.algorithm_key,
        "scan_context": {
            "group_label": f"continuation scan: {len(group_points)} groups",
            "batch_label": f"{total_points} alpha states",
            "point_label": f"points 0/{total_points}",
            "point_progress": {"completed": 0, "total": total_points},
        },
    })

    aggregate = ExperimentResult(
        experiment_id=getattr(config, "experiment_name", "alpha_descending_continuation"),
        config=config,
        scan_dimension="scan",
        scan_values=scan_plan.point_ids(),
        metadata=ExperimentMetadata.create_now(),
    )
    if output_options and output_options.get("experiment_plan"):
        aggregate.metadata.contract = copy.deepcopy(output_options["experiment_plan"])
    aggregate.metadata.contract["scan_plan"] = scan_plan.to_dict()
    aggregate.metadata.contract["continuation"] = cont.to_dict()
    aggregate.result_cube = ResultCube(
        axes={axis.key: axis.to_dict() for axis in scan_plan.axes},
        metric_semantics=aggregate.metric_semantics(),
    )

    completed = 0
    previous_group_id: Optional[str] = None
    for group_id, points in group_points.items():
        if previous_group_id is not None and group_id != previous_group_id:
            runner._clear_algorithm_compile_cache(config.algorithm_key)
        previous_group_id = group_id
        _run_continuation_group(
            runner=runner,
            config=config,
            group_id=group_id,
            points=points,
            scan_plan=scan_plan,
            continuation=cont,
            aggregate=aggregate,
            observer=observer,
            output_options=output_options,
            completed_offset=completed,
            total_points=total_points,
        )
        completed += len(points)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    runner._emit(observer, ProgressEventType.EXPERIMENT_END, {"result": aggregate})
    return aggregate


def _run_continuation_group(
    *,
    runner: Any,
    config: Any,
    group_id: str,
    points: List[ScanPoint],
    scan_plan: ScanPlan,
    continuation: ContinuationConfig,
    aggregate: ExperimentResult,
    observer: Optional[Callable[[Any], None]],
    output_options: Optional[Dict[str, Any]],
    completed_offset: int,
    total_points: int,
) -> None:
    from matrix_factorization.core.experiment.runner import ProgressEventType

    ordered_points = sorted(points, key=lambda p: float(p.alpha), reverse=True)
    group_config = effective_config_for_scan_points(config, ordered_points)
    group_config.scan.values = [float(point.alpha) for point in ordered_points]
    group_config.scan_spec = {
        "axes": {"alpha": {"path": "alpha", "values": list(group_config.scan.values)}},
        "continuation": continuation.to_scan_dict(),
    }
    data = runner.data_factory.create(
        group_config,
        alpha_values=list(group_config.scan.values),
        observation_policy=continuation.observation_policy,
    )
    if aggregate.W_teacher is None:
        aggregate.W_teacher = runner._detach_to_cpu(data.W_teacher)
        aggregate.X_teacher = runner._detach_to_cpu(data.X_teacher)
        aggregate.Y_teacher = runner._detach_to_cpu(data.Y_teacher)

    algorithm = runner._get_algorithm(group_config)
    state: Optional[AlgorithmStateView] = None
    state_alpha: Optional[float] = None

    for local_idx, point in enumerate(ordered_points):
        alpha = float(point.alpha)
        runner._emit(observer, ProgressEventType.BATCH_START, {
            "batch_idx": completed_offset + local_idx,
            "total_batches": total_points,
            "alpha_values": [alpha],
            "steps_per_alpha": group_config.training.max_steps,
            "scan_context": {
                "canonical_scan": True,
                "continuation": True,
                "group_id": group_id,
                "coordinates": dict(point.coordinates),
                "point_progress": {"completed": completed_offset + local_idx, "total": total_points},
            },
        })
        runner._emit(observer, ProgressEventType.POINT_START, {
            "point_idx": completed_offset + local_idx + 1,
            "total_points": total_points,
            "value": point.point_id,
        })

        single_data = _single_alpha_data(data, alpha, local_idx)

        def step_callback(step: int, total: int, metrics: Optional[Dict[str, Any]] = None) -> None:
            runner._emit(observer, ProgressEventType.STEP_UPDATE, {
                "step": step,
                "total": total,
                "batch_idx": completed_offset + local_idx,
                "metrics": metrics or {},
                "scan_context": {
                    "continuation": True,
                    "source_alpha": state_alpha,
                    "target_alpha": alpha,
                    "group_id": group_id,
                },
            })

        start_time = time.time()
        algorithm_result = runner._run_algorithm_result(
            algorithm=algorithm,
            config=group_config,
            data=single_data,
            step_callback=step_callback,
            initial_state=state,
            return_continuation_state=True,
            continuation_context={
                "enabled": True,
                "source_alpha": state_alpha,
                "target_alpha": alpha,
                "alpha_index": local_idx,
                "alpha_values_full": list(group_config.scan.values),
                "observation_policy": continuation.observation_policy,
                "adaptive_controller_state": continuation.adaptive_controller_state,
            },
        )
        next_state = algorithm_result.continuation_state
        if next_state is None and continuation.strict_state:
            raise RuntimeError(
                f"algorithm '{group_config.algorithm_key}' did not return continuation_state "
                f"for alpha={alpha}; strict_state=true"
            )

        W_students, X_students = runner._matrix_factors_from_result(algorithm_result)
        W_single = runner._slice_alpha_factor(W_students, 0)
        X_single = runner._slice_alpha_factor(X_students, 0)
        metrics = runner._compute_metrics(
            W_students=W_single.unsqueeze(0) if W_single is not None and W_single.dim() == 3 else W_single,
            X_students=X_single.unsqueeze(0) if X_single is not None and X_single.dim() == 3 else X_single,
            data=single_data,
            algorithm=algorithm,
            algorithm_result=algorithm_result,
        )
        metrics = dict(metrics or {})
        point_continuation_metadata = _continuation_point_metadata(
            continuation,
            state,
            next_state,
            state_alpha,
            alpha,
        )
        aggregate.metadata.contract.setdefault("continuation_points", {})[point.point_id] = point_continuation_metadata
        metric_contract = runner._validate_metric_payload(
            group_config.algorithm_key,
            metrics,
            source=runner._metric_source(single_data, algorithm_result),
        ).to_dict()

        single_result = SingleRunResult(
            scan_value=point.point_id,
            metrics=metrics,
            W_students=runner._detach_to_cpu(W_single) if W_single is not None else None,
            X_students=runner._detach_to_cpu(X_single) if X_single is not None else None,
            duration_seconds=time.time() - start_time,
            metric_contract=metric_contract,
        )
        aggregate.add_result(point.point_id, single_result)
        aggregate.result_cube.add_point(
            point.point_id,
            point.coordinates,
            metrics,
            overrides=point.overrides,
            effective_config_hash=point.effective_config_hash,
            group_id=point.group_id,
            metric_contract=metric_contract,
            artifacts={"continuation": point_continuation_metadata},
        )
        runner._record_algorithm_result_summary(
            result=aggregate,
            batch_idx=completed_offset + local_idx,
            alpha_values=[alpha],
            algorithm_result=algorithm_result,
        )
        runner._emit(observer, ProgressEventType.POINT_COMPLETE, {
            "point_idx": completed_offset + local_idx + 1,
            "total_points": total_points,
            "value": point.point_id,
            "metrics": metrics,
        })
        runner._emit(observer, ProgressEventType.BATCH_END, {
            "batch_idx": completed_offset + local_idx,
            "duration": single_result.duration_seconds,
        })

        state = next_state
        state_alpha = alpha


def _single_alpha_data(data: ExperimentData, alpha: float, alpha_index: int) -> ExperimentData:
    masks = data.masks
    if masks is not None and getattr(masks, "dim", lambda: 0)() == 3:
        masks = masks[alpha_index:alpha_index + 1]
    return ExperimentData(
        W_teacher=data.W_teacher,
        X_teacher=data.X_teacher,
        Y_teacher=data.Y_teacher,
        masks=masks,
        spreading_data=data.spreading_data,
        alpha_values=[float(alpha)],
        device=data.device,
    )


def _points_by_group(scan_plan: ScanPlan) -> Dict[str, List[ScanPoint]]:
    points_by_id = {point.point_id: point for point in scan_plan.points}
    if scan_plan.grouping:
        return {
            group.group_id: [points_by_id[point_id] for point_id in group.point_ids]
            for group in scan_plan.grouping
        }
    groups: Dict[str, List[ScanPoint]] = {}
    for point in scan_plan.points:
        groups.setdefault(point.group_id, []).append(point)
    return groups


def _execution_plan_batches(group_points: Dict[str, List[ScanPoint]]) -> List[tuple[int, int, float]]:
    batches: List[tuple[int, int, float]] = []
    for points in group_points.values():
        for index, point in enumerate(sorted(points, key=lambda p: float(p.alpha), reverse=True)):
            batches.append((index, 1, float(point.alpha)))
    return batches


def _validate_runtime_continuation_config(cont: ContinuationConfig, config: Any) -> None:
    if not cont.enabled:
        return
    if cont.axis != "alpha":
        raise ValueError("scan.continuation.axis currently supports only 'alpha'")
    if cont.order != "descending":
        raise ValueError("scan.continuation.order currently supports only 'descending'")
    if cont.state_transfer != "full_algorithm_state":
        raise ValueError("scan.continuation.state_transfer currently supports only 'full_algorithm_state'")
    if cont.observation_policy not in {"nested_prefix", "independent_resample"}:
        raise ValueError("scan.continuation.observation_policy must be nested_prefix or independent_resample")
    if cont.adaptive_controller_state != "reset":
        raise ValueError("scan.continuation.adaptive_controller_state currently supports only 'reset'")
    scan_spec = getattr(config, "scan_spec", None)
    axes = scan_spec.get("axes", {}) if isinstance(scan_spec, dict) else {}
    if "alpha" not in axes:
        raise ValueError("scan.continuation requires scan.axes.alpha")
    spreading = getattr(config, "spreading", None)
    if getattr(config, "algorithm_key", "") == "bigamp_spreading" and bool(getattr(spreading, "allow_intra_connection", False)):
        raise ValueError(
            "scan.continuation does not yet support bigamp_spreading general graph "
            "(spreading.allow_intra_connection=true) because formal spreading metrics "
            "do not consume SuperGraphDataGeneral."
        )


def _continuation_point_metadata(
    continuation: ContinuationConfig,
    previous_state: Optional[AlgorithmStateView],
    next_state: Optional[AlgorithmStateView],
    source_alpha: Optional[float],
    target_alpha: float,
) -> Dict[str, Any]:
    return {
        "continuation.enabled": continuation.enabled,
        "continuation.order": continuation.order,
        "continuation.source_alpha": float(source_alpha) if source_alpha is not None else None,
        "continuation.target_alpha": float(target_alpha),
        "continuation.state_transfer": continuation.state_transfer,
        "continuation.state_keys": next_state.available_capabilities() if next_state is not None else [],
        "continuation.previous_state_keys": previous_state.available_capabilities() if previous_state is not None else [],
        "continuation.observation_policy": continuation.observation_policy,
        "continuation.physical_nested_observation": continuation.observation_policy == "nested_prefix",
    }
