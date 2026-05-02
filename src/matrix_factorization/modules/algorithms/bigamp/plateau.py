"""Teacher-assisted metric plateau stop for flat BiGAMP spreading.

This module is diagnostic glue: it observes committed student states and never
feeds back into AMP, Onsager residuals, damping, denoisers, or variance updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Sequence

import torch


PROJECTION_NORM_EPS = 1e-12
DEFAULT_PLATEAU_CHECK_INTERVAL = 100
DEFAULT_PLATEAU_WINDOW_STEPS = 500
DEFAULT_PLATEAU_PATIENCE = 3
DEFAULT_PLATEAU_ABS_TOL = 0.003
DEFAULT_PLATEAU_REL_TOL = 0.01
DEFAULT_PLATEAU_REL_FLOOR = 1e-2
DEFAULT_PLATEAU_TREND_RATIO_TOL = 0.95
DEFAULT_PLATEAU_SLOPE_ABS_TOL_RATIO = 0.05
DEFAULT_PLATEAU_SLOPE_ABS_TOL_CAP = 5e-4
DEFAULT_PLATEAU_NEAR_ZERO_SLOPE_ABS_TOL_RATIO = 0.02
DEFAULT_PLATEAU_NEAR_ZERO_SLOPE_ABS_TOL_CAP = 2e-4
DEFAULT_PLATEAU_TAIL_MIN_WINDOWS = 10
DEFAULT_PLATEAU_MIN_STEPS = 0
PLATEAU_SIGNAL_TEACHER_LATENT = "teacher_latent_overlap"
PLATEAU_MONITOR_QW_QX = "teacher_latent_overlap_qw_qx"
SIZE_AWARE_REFERENCE_STEP_CAPS = (
    (200, 3000),
    (500, 5000),
    (1000, 8000),
    (2000, 12000),
    (5000, 20000),
    (10000, 32000),
    (20000, 48000),
)


@dataclass(frozen=True)
class MetricPlateauStopConfig:
    enabled: bool = False
    check_interval: int = DEFAULT_PLATEAU_CHECK_INTERVAL
    window_steps: int = DEFAULT_PLATEAU_WINDOW_STEPS
    patience: int = DEFAULT_PLATEAU_PATIENCE
    abs_tol: float = DEFAULT_PLATEAU_ABS_TOL
    rel_tol: float = DEFAULT_PLATEAU_REL_TOL
    min_steps: int = DEFAULT_PLATEAU_MIN_STEPS
    signal: str = PLATEAU_SIGNAL_TEACHER_LATENT
    monitor: str = PLATEAU_MONITOR_QW_QX
    rel_floor: float = DEFAULT_PLATEAU_REL_FLOOR

    @classmethod
    def from_algorithm_params(cls, params: Any) -> "MetricPlateauStopConfig":
        return cls(
            enabled=bool(getattr(params, "use_metric_plateau_stop", False)),
            check_interval=int(getattr(params, "plateau_check_interval", DEFAULT_PLATEAU_CHECK_INTERVAL)),
            window_steps=int(getattr(params, "plateau_window_steps", DEFAULT_PLATEAU_WINDOW_STEPS)),
            patience=int(getattr(params, "plateau_patience", DEFAULT_PLATEAU_PATIENCE)),
            abs_tol=float(getattr(params, "plateau_abs_tol", DEFAULT_PLATEAU_ABS_TOL)),
            rel_tol=float(getattr(params, "plateau_rel_tol", DEFAULT_PLATEAU_REL_TOL)),
            min_steps=int(getattr(params, "plateau_min_steps", DEFAULT_PLATEAU_MIN_STEPS)),
            signal=str(getattr(params, "plateau_signal", PLATEAU_SIGNAL_TEACHER_LATENT)),
            monitor=str(getattr(params, "plateau_monitor", PLATEAU_MONITOR_QW_QX)),
        )

    def __post_init__(self) -> None:
        if self.check_interval <= 0:
            raise ValueError("plateau_check_interval must be positive")
        if self.window_steps < 0:
            raise ValueError("plateau_window_steps must be non-negative")
        if self.patience <= 0:
            raise ValueError("plateau_patience must be positive")
        if self.abs_tol < 0.0:
            raise ValueError("plateau_abs_tol must be non-negative")
        if self.rel_tol < 0.0:
            raise ValueError("plateau_rel_tol must be non-negative")
        if self.rel_floor <= 0.0:
            raise ValueError("plateau relative floor must be positive")
        if self.min_steps < 0:
            raise ValueError("plateau_min_steps must be non-negative")
        if self.signal != PLATEAU_SIGNAL_TEACHER_LATENT:
            raise ValueError("plateau_signal must be 'teacher_latent_overlap'")
        if self.monitor != PLATEAU_MONITOR_QW_QX:
            raise ValueError("plateau_monitor must be 'teacher_latent_overlap_qw_qx'")

    @property
    def effective_min_steps(self) -> int:
        """Deprecated min-step gate is intentionally ignored by self-convergence stop."""

        return 0

    @property
    def effective_window_steps(self) -> int:
        return int(self.window_steps) if int(self.window_steps) > 0 else int(self.check_interval)

    @property
    def effective_earliest_check_step(self) -> int:
        return int(self.effective_window_steps)

    @property
    def slope_abs_tol(self) -> float:
        return min(
            float(self.abs_tol) * float(DEFAULT_PLATEAU_SLOPE_ABS_TOL_RATIO),
            float(DEFAULT_PLATEAU_SLOPE_ABS_TOL_CAP),
        )

    @property
    def near_zero_slope_abs_tol(self) -> float:
        return min(
            float(self.abs_tol) * float(DEFAULT_PLATEAU_NEAR_ZERO_SLOPE_ABS_TOL_RATIO),
            float(DEFAULT_PLATEAU_NEAR_ZERO_SLOPE_ABS_TOL_CAP),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "check_interval": int(self.check_interval),
            "window_steps": int(self.window_steps),
            "effective_window_steps": int(self.effective_window_steps),
            "patience": int(self.patience),
            "abs_tol": float(self.abs_tol),
            "rel_tol": float(self.rel_tol),
            "rel_floor": float(self.rel_floor),
            "min_steps": int(self.min_steps),
            "min_steps_deprecated": True,
            "effective_min_steps": self.effective_min_steps,
            "effective_earliest_check_step": self.effective_earliest_check_step,
            "signal": self.signal,
            "monitor": self.monitor,
            "strategy": "self_convergence_window_trend_decay",
            "trend_window_policy": "non_overlapping_windows",
            "trend_ratio_tol": float(DEFAULT_PLATEAU_TREND_RATIO_TOL),
            "slope_abs_tol": float(self.slope_abs_tol),
            "slope_abs_tol_ratio": float(DEFAULT_PLATEAU_SLOPE_ABS_TOL_RATIO),
            "slope_abs_tol_cap": float(DEFAULT_PLATEAU_SLOPE_ABS_TOL_CAP),
            "near_zero_slope_abs_tol": float(self.near_zero_slope_abs_tol),
            "near_zero_slope_abs_tol_ratio": float(DEFAULT_PLATEAU_NEAR_ZERO_SLOPE_ABS_TOL_RATIO),
            "near_zero_slope_abs_tol_cap": float(DEFAULT_PLATEAU_NEAR_ZERO_SLOPE_ABS_TOL_CAP),
            "tail_min_windows": int(DEFAULT_PLATEAU_TAIL_MIN_WINDOWS),
            "convergence_semantics": (
                "direction-free self-history convergence over Q_W/Q_X; each monitored metric "
                "must satisfy absolute stability, relative stability, and either a near-zero "
                "slope fast path or a non-overlapping-window trend-decay gate"
            ),
        }


@dataclass(frozen=True)
class MetricPlateauProfile:
    check_interval: int
    window_steps: int
    patience: int = DEFAULT_PLATEAU_PATIENCE
    abs_tol: float = DEFAULT_PLATEAU_ABS_TOL
    rel_tol: float = DEFAULT_PLATEAU_REL_TOL

    def to_config(self, *, enabled: bool = True) -> MetricPlateauStopConfig:
        return MetricPlateauStopConfig(
            enabled=enabled,
            check_interval=self.check_interval,
            window_steps=self.window_steps,
            patience=self.patience,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            min_steps=0,
            signal=PLATEAU_SIGNAL_TEACHER_LATENT,
            monitor=PLATEAU_MONITOR_QW_QX,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_interval": int(self.check_interval),
            "window_steps": int(self.window_steps),
            "patience": int(self.patience),
            "abs_tol": float(self.abs_tol),
            "rel_tol": float(self.rel_tol),
        }


@dataclass(frozen=True)
class MetricPlateauSnapshot:
    step: int
    q_w: torch.Tensor
    q_x: torch.Tensor
    r_w: torch.Tensor
    r_x: torch.Tensor

    def vectors(self) -> Dict[str, torch.Tensor]:
        return {
            "Q_W": self.q_w,
            "Q_X": self.q_x,
            "R_W": self.r_w,
            "R_X": self.r_x,
        }


@dataclass(frozen=True)
class MetricPlateauCheckResult:
    step: int
    checked: bool
    stop: bool = False
    reason: str = ""
    delta_by_alpha: Optional[List[float]] = None
    bad_count_by_alpha: Optional[List[int]] = None
    stable_count_by_alpha: Optional[List[int]] = None


@torch.no_grad()
def compute_teacher_latent_snapshot(
    *,
    step: int,
    W_flat: torch.Tensor,
    X_flat: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    S: int,
    N1: int,
    N2: int,
    M: int,
) -> MetricPlateauSnapshot:
    """Compute schema-v5 fixed-denominator latent overlaps for each alpha.

    ``W_flat`` has shape ``(B, S*N1, M)`` and is compared to ``W_teacher``.
    ``X_flat`` has shape ``(B, S*N2, M)`` and is compared to ``X_teacher.T``.
    The returned vectors are mean-over-samples tensors on CPU.
    """

    B = int(W_flat.shape[0])
    W = W_flat.reshape(B, S, N1, M).float()
    X = X_flat.reshape(B, S, N2, M).float()
    W_ref = W_teacher.to(device=W_flat.device, dtype=torch.float32).reshape(1, 1, N1, M)
    X_ref = X_teacher.T.contiguous().to(device=X_flat.device, dtype=torch.float32).reshape(1, 1, N2, M)

    w_denom = float(max(N1 * M, 1)) + PROJECTION_NORM_EPS
    x_denom = float(max(N2 * M, 1)) + PROJECTION_NORM_EPS
    q_w = (W * W_ref).sum(dim=(-2, -1)) / w_denom
    q_x = (X * X_ref).sum(dim=(-2, -1)) / x_denom
    r_w = (W * W).sum(dim=(-2, -1)) / w_denom
    r_x = (X * X).sum(dim=(-2, -1)) / x_denom

    return MetricPlateauSnapshot(
        step=int(step),
        q_w=q_w.mean(dim=1).detach().cpu().float(),
        q_x=q_x.mean(dim=1).detach().cpu().float(),
        r_w=r_w.mean(dim=1).detach().cpu().float(),
        r_x=r_x.mean(dim=1).detach().cpu().float(),
    )


@dataclass
class MetricPlateauStopper:
    config: MetricPlateauStopConfig
    alpha_values: List[float]
    history: Dict[int, MetricPlateauSnapshot] = field(default_factory=dict)
    history_order: List[int] = field(default_factory=list)
    stable_counts: torch.Tensor = field(init=False)
    last_previous_values: Dict[str, torch.Tensor] = field(default_factory=dict)
    last_window_start_values: Dict[str, torch.Tensor] = field(default_factory=dict)
    last_values: Dict[str, torch.Tensor] = field(default_factory=dict)
    ready_steps: torch.Tensor = field(init=False)
    check_count: int = 0
    stop_step: Optional[int] = None
    stop_reason: str = ""
    last_delta: Optional[torch.Tensor] = None
    last_delta_components: Optional[Dict[str, torch.Tensor]] = None
    last_metric_diagnostics: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    last_metric_stable: Dict[str, torch.Tensor] = field(default_factory=dict)
    last_alpha_stable: Optional[torch.Tensor] = None
    last_check_step: Optional[int] = None

    def __post_init__(self) -> None:
        self.stable_counts = torch.zeros(len(self.alpha_values), dtype=torch.long)
        self.ready_steps = torch.full((len(self.alpha_values),), -1, dtype=torch.long)

    def observe(self, snapshot: MetricPlateauSnapshot) -> MetricPlateauCheckResult:
        self._validate_snapshot(snapshot)
        step = int(snapshot.step)
        self.history[step] = snapshot
        if step not in self.history_order:
            self.history_order.append(step)

        current_vectors = self._monitor_vectors(snapshot)
        self.last_values = {key: value.clone().float() for key, value in current_vectors.items()}
        if step <= 0:
            return MetricPlateauCheckResult(step=step, checked=False, reason="initial_snapshot")
        if step % self.config.check_interval != 0:
            return MetricPlateauCheckResult(step=step, checked=False, reason="not_check_interval")

        window = self._window_snapshots(step)
        if not window:
            return MetricPlateauCheckResult(step=step, checked=False, reason="waiting_for_window")
        previous_window = self._previous_window_snapshots(step, window[0].step)
        if len(window) < 2 or len(previous_window) < 2:
            return MetricPlateauCheckResult(step=step, checked=False, reason="waiting_for_window")

        self.check_count += 1
        self.last_check_step = step
        previous = self._previous_check_vectors(step, fallback_snapshot=window[0])
        window_start = self._monitor_vectors(window[0])

        diagnostics, stable_by_metric = self._metric_diagnostics_for_window(
            window=window,
            previous_window=previous_window,
            current=current_vectors,
            previous=previous,
            window_start=window_start,
        )
        alpha_stable = torch.ones(len(self.alpha_values), dtype=torch.bool)
        for metric_stable in stable_by_metric.values():
            alpha_stable &= metric_stable

        self.stable_counts = torch.where(
            alpha_stable,
            self.stable_counts + 1,
            torch.zeros_like(self.stable_counts),
        )
        newly_ready = (
            alpha_stable
            & (self.stable_counts >= int(self.config.patience))
            & (self.ready_steps < 0)
        )
        self.ready_steps = torch.where(
            alpha_stable,
            torch.where(newly_ready, torch.full_like(self.ready_steps, step), self.ready_steps),
            torch.full_like(self.ready_steps, -1),
        )

        delta_components = {
            key: torch.maximum(
                torch.maximum(values["window_abs_range"], values["endpoint_abs_delta"]),
                values["projected_abs_change"],
            ).float()
            for key, values in diagnostics.items()
        }
        delta = torch.stack(list(delta_components.values()), dim=0).amax(dim=0).float()
        if not torch.isfinite(delta).all():
            raise FloatingPointError(
                f"metric plateau stop saw non-finite Delta at step {step}"
            )
        self.last_delta = delta
        self.last_delta_components = delta_components
        self.last_metric_diagnostics = diagnostics
        self.last_metric_stable = stable_by_metric
        self.last_alpha_stable = alpha_stable
        self.last_previous_values = {key: value.clone().float() for key, value in previous.items()}
        self.last_window_start_values = {key: value.clone().float() for key, value in window_start.items()}

        stop = bool(
            self.stable_counts.numel() > 0
            and torch.all(self.stable_counts >= int(self.config.patience)).item()
        )
        if stop:
            self.stop_step = step
            self.stop_reason = "metric_self_convergence"
        return MetricPlateauCheckResult(
            step=step,
            checked=True,
            stop=stop,
            reason="metric_self_convergence" if stop else "checked",
            delta_by_alpha=[float(value) for value in delta.tolist()],
            bad_count_by_alpha=[int(value) for value in self.stable_counts.tolist()],
            stable_count_by_alpha=[int(value) for value in self.stable_counts.tolist()],
        )

    def summary(self, *, steps_run: int, configured_max_steps: int) -> Dict[str, Any]:
        last_delta_list = None
        last_delta_max = None
        if self.last_delta is not None:
            last_delta_list = [float(value) for value in self.last_delta.tolist()]
            last_delta_max = float(self.last_delta.max().item()) if self.last_delta.numel() else None
        last_delta_by_metric = None
        if self.last_delta_components is not None:
            last_delta_by_metric = {
                key: [float(value) for value in tensor.tolist()]
                for key, tensor in self.last_delta_components.items()
            }
        reason = self.stop_reason or "max_steps_reached"
        per_alpha_status = self._per_alpha_status()
        for item in per_alpha_status:
            item["actual_batch_stop_step"] = self.stop_step
        summary = {
            "enabled": bool(self.config.enabled),
            "teacher_assisted": bool(self.config.enabled),
            "config": self.config.to_dict(),
            "configured_max_steps": int(configured_max_steps),
            "steps_run": int(steps_run),
            "stop_reason": reason,
            "stop_step": self.stop_step,
            "last_check_step": self.last_check_step,
            "last_delta": last_delta_max,
            "last_delta_by_alpha": last_delta_list,
            "last_delta_by_metric": last_delta_by_metric,
            "per_alpha_status": per_alpha_status,
            "history_summary": self._history_summary(),
        }
        if _export_metric_plateau_history_enabled():
            summary["history_records"] = self._history_records()
        return summary

    @staticmethod
    def _monitor_vectors(snapshot: MetricPlateauSnapshot) -> Dict[str, torch.Tensor]:
        return {
            "Q_W": snapshot.q_w.float(),
            "Q_X": snapshot.q_x.float(),
        }

    def _validate_snapshot(self, snapshot: MetricPlateauSnapshot) -> None:
        expected = len(self.alpha_values)
        for key, tensor in snapshot.vectors().items():
            if tensor.shape != (expected,):
                raise ValueError(
                    f"metric plateau snapshot {key} has shape {tuple(tensor.shape)}, expected {(expected,)}"
                )
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(
                    f"metric plateau stop saw non-finite {key} at step {snapshot.step}"
                )

    def _window_snapshots(self, current_step: int) -> List[MetricPlateauSnapshot]:
        steps = sorted(set(self.history_order))
        lower = int(current_step) - int(self.config.effective_window_steps)
        start_candidates = [step for step in steps if step <= lower]
        if not start_candidates:
            return []
        start_step = start_candidates[-1]
        window_steps = [step for step in steps if start_step <= step <= int(current_step)]
        return [self.history[step] for step in window_steps]

    def _previous_window_snapshots(
        self,
        current_step: int,
        current_window_start_step: int,
    ) -> List[MetricPlateauSnapshot]:
        steps = sorted(set(self.history_order))
        lower = int(current_step) - 2 * int(self.config.effective_window_steps)
        start_candidates = [step for step in steps if step <= lower]
        if not start_candidates:
            return []
        start_step = start_candidates[-1]
        previous_end = int(current_window_start_step)
        previous_window_steps = [step for step in steps if start_step <= step <= previous_end]
        return [self.history[step] for step in previous_window_steps]

    def _previous_check_vectors(
        self,
        current_step: int,
        *,
        fallback_snapshot: MetricPlateauSnapshot,
    ) -> Dict[str, torch.Tensor]:
        previous_steps = [step for step in sorted(set(self.history_order)) if step < int(current_step)]
        if not previous_steps:
            return self._monitor_vectors(fallback_snapshot)
        return self._monitor_vectors(self.history[previous_steps[-1]])

    def _metric_diagnostics_for_window(
        self,
        *,
        window: Sequence[MetricPlateauSnapshot],
        previous_window: Sequence[MetricPlateauSnapshot],
        current: Dict[str, torch.Tensor],
        previous: Dict[str, torch.Tensor],
        window_start: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        step_tensor = torch.tensor([float(item.step) for item in window], dtype=torch.float32)
        previous_step_tensor = torch.tensor([float(item.step) for item in previous_window], dtype=torch.float32)
        diagnostics: Dict[str, Dict[str, torch.Tensor]] = {}
        stable_by_metric: Dict[str, torch.Tensor] = {}
        for key in ("Q_W", "Q_X"):
            values = torch.stack([self._monitor_vectors(item)[key] for item in window], dim=0).float()
            previous_values = torch.stack(
                [self._monitor_vectors(item)[key] for item in previous_window],
                dim=0,
            ).float()
            current_value = current[key].float()
            previous_value = previous[key].float()
            start_value = window_start[key].float()
            denom = torch.clamp(current_value.abs(), min=float(self.config.rel_floor))
            window_abs_range = values.max(dim=0).values - values.min(dim=0).values
            endpoint_abs_delta = (current_value - start_value).abs()
            recent_abs_delta = (current_value - previous_value).abs()
            slope_per_step = self._robust_slope_per_step(values, step_tensor)
            projected_abs_change = slope_per_step.abs() * float(self.config.effective_window_steps)
            previous_slope_per_step = self._robust_slope_per_step(previous_values, previous_step_tensor)
            previous_projected_abs_change = (
                previous_slope_per_step.abs() * float(self.config.effective_window_steps)
            )
            previous_window_abs_range = (
                previous_values.max(dim=0).values - previous_values.min(dim=0).values
            )
            previous_endpoint_abs_delta = (previous_values[-1] - previous_values[0]).abs()

            window_rel_range = window_abs_range / denom
            endpoint_rel_delta = endpoint_abs_delta / denom
            recent_rel_delta = recent_abs_delta / denom
            projected_rel_change = projected_abs_change / denom
            previous_projected_rel_change = previous_projected_abs_change / denom
            previous_window_rel_range = previous_window_abs_range / denom
            previous_endpoint_rel_delta = previous_endpoint_abs_delta / denom

            trend_abs_ratio = projected_abs_change / torch.clamp(
                previous_projected_abs_change,
                min=PROJECTION_NORM_EPS,
            )
            trend_rel_ratio = projected_rel_change / torch.clamp(
                previous_projected_rel_change,
                min=PROJECTION_NORM_EPS,
            )

            stable_abs = (
                (window_abs_range <= float(self.config.abs_tol))
                & (endpoint_abs_delta <= float(self.config.abs_tol))
                & (projected_abs_change <= float(self.config.abs_tol))
            )
            stable_rel = (
                (window_rel_range <= float(self.config.rel_tol))
                & (endpoint_rel_delta <= float(self.config.rel_tol))
                & (projected_rel_change <= float(self.config.rel_tol))
            )
            observed_windows = max(
                0,
                int((int(window[-1].step) - min(self.history_order)) // int(self.config.effective_window_steps)),
            )
            observed_windows_tensor = torch.full_like(
                projected_abs_change,
                float(observed_windows),
            ).float()
            slope_abs_tol = float(self.config.slope_abs_tol)
            near_zero_slope_abs_tol = float(self.config.near_zero_slope_abs_tol)
            trend_ratio_tol = float(DEFAULT_PLATEAU_TREND_RATIO_TOL)
            tail_min_windows = int(DEFAULT_PLATEAU_TAIL_MIN_WINDOWS)
            near_zero_fast_path = projected_abs_change <= near_zero_slope_abs_tol
            small_slope = projected_abs_change <= slope_abs_tol
            decayed_trend = trend_abs_ratio <= trend_ratio_tol
            enough_tail_history = observed_windows >= tail_min_windows
            tail_decay_path = small_slope & decayed_trend & enough_tail_history
            stable_trend = near_zero_fast_path | tail_decay_path
            stable = stable_abs & stable_rel & stable_trend
            diagnostics[key] = {
                "current_value": current_value,
                "window_start_value": start_value,
                "previous_check_value": previous_value,
                "window_abs_range": window_abs_range.float(),
                "window_rel_range": window_rel_range.float(),
                "endpoint_abs_delta": endpoint_abs_delta.float(),
                "endpoint_rel_delta": endpoint_rel_delta.float(),
                "recent_abs_delta": recent_abs_delta.float(),
                "recent_rel_delta": recent_rel_delta.float(),
                "slope_per_step": slope_per_step.float(),
                "projected_abs_change": projected_abs_change.float(),
                "projected_rel_change": projected_rel_change.float(),
                "previous_window_abs_range": previous_window_abs_range.float(),
                "previous_window_rel_range": previous_window_rel_range.float(),
                "previous_endpoint_abs_delta": previous_endpoint_abs_delta.float(),
                "previous_endpoint_rel_delta": previous_endpoint_rel_delta.float(),
                "previous_slope_per_step": previous_slope_per_step.float(),
                "previous_projected_abs_change": previous_projected_abs_change.float(),
                "previous_projected_rel_change": previous_projected_rel_change.float(),
                "slope_abs_tol": torch.full_like(projected_abs_change, slope_abs_tol).float(),
                "near_zero_slope_abs_tol": torch.full_like(
                    projected_abs_change,
                    near_zero_slope_abs_tol,
                ).float(),
                "trend_ratio_tol": torch.full_like(projected_abs_change, trend_ratio_tol).float(),
                "tail_min_windows": torch.full_like(projected_abs_change, float(tail_min_windows)).float(),
                "observed_windows": observed_windows_tensor,
                "trend_abs_ratio": trend_abs_ratio.float(),
                "trend_rel_ratio": trend_rel_ratio.float(),
                "near_zero_fast_path": near_zero_fast_path,
                "small_slope": small_slope,
                "decayed_trend": decayed_trend,
                "enough_tail_history": torch.full(
                    projected_abs_change.shape,
                    enough_tail_history,
                    dtype=torch.bool,
                ),
                "tail_decay_path": tail_decay_path,
                "stable_trend_abs": small_slope,
                "stable_trend_rel": decayed_trend,
                "stable_trend": stable_trend,
                "stable_abs": stable_abs,
                "stable_rel": stable_rel,
                "stable": stable,
            }
            for field_name, tensor in diagnostics[key].items():
                if tensor.dtype is torch.bool:
                    continue
                if not torch.isfinite(tensor.float()).all():
                    raise FloatingPointError(
                        f"metric plateau stop saw non-finite {key}.{field_name}"
                    )
            stable_by_metric[key] = stable
        return diagnostics, stable_by_metric

    @staticmethod
    def _robust_slope_per_step(values: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        if values.shape[0] < 2:
            return torch.zeros_like(values[-1])
        slopes = []
        for start_idx in range(values.shape[0] - 1):
            dt = (steps[start_idx + 1:] - steps[start_idx]).reshape(-1, 1)
            slopes.append((values[start_idx + 1:] - values[start_idx]) / dt)
        return torch.cat(slopes, dim=0).median(dim=0).values.float()

    def _per_alpha_status(self) -> List[Dict[str, Any]]:
        delta_values = [None] * len(self.alpha_values)
        if self.last_delta is not None:
            delta_values = [float(value) for value in self.last_delta.tolist()]
        component_values: Dict[str, List[Optional[float]]] = {}
        if self.last_delta_components is not None:
            component_values = {
                key: [float(value) for value in tensor.tolist()]
                for key, tensor in self.last_delta_components.items()
            }
        current_values = {
            key: [float(value) for value in tensor.tolist()]
            for key, tensor in self.last_values.items()
        }
        statuses = []
        for idx, alpha in enumerate(self.alpha_values):
            stable_count = int(self.stable_counts[idx].item()) if idx < self.stable_counts.numel() else 0
            ready_step = int(self.ready_steps[idx].item()) if idx < self.ready_steps.numel() else -1
            alpha_stable = (
                bool(self.last_alpha_stable[idx].item())
                if self.last_alpha_stable is not None and idx < self.last_alpha_stable.numel()
                else False
            )
            statuses.append({
                "alpha": float(alpha),
                "stable_count": stable_count,
                "patience_count": stable_count,
                "ready_step": ready_step if ready_step >= 0 else None,
                "plateau_satisfied": stable_count >= int(self.config.patience),
                "alpha_stable": alpha_stable,
                "last_delta": delta_values[idx] if idx < len(delta_values) else None,
                "last_delta_by_metric": {
                    key: values[idx] if idx < len(values) else None
                    for key, values in component_values.items()
                },
                "metrics": {
                    key: self._metric_status_for_alpha(key, idx, current_values)
                    for key in ("Q_W", "Q_X")
                },
            })
        return statuses

    def _metric_status_for_alpha(
        self,
        key: str,
        idx: int,
        current_values: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        diagnostics = self.last_metric_diagnostics.get(key, {})
        fields = (
            "current_value",
            "window_start_value",
            "previous_check_value",
            "window_abs_range",
            "window_rel_range",
            "endpoint_abs_delta",
            "endpoint_rel_delta",
            "recent_abs_delta",
            "recent_rel_delta",
            "slope_per_step",
            "projected_abs_change",
            "projected_rel_change",
            "previous_window_abs_range",
            "previous_window_rel_range",
            "previous_endpoint_abs_delta",
            "previous_endpoint_rel_delta",
            "previous_slope_per_step",
            "previous_projected_abs_change",
            "previous_projected_rel_change",
            "slope_abs_tol",
            "near_zero_slope_abs_tol",
            "trend_ratio_tol",
            "tail_min_windows",
            "observed_windows",
            "trend_abs_ratio",
            "trend_rel_ratio",
            "near_zero_fast_path",
            "small_slope",
            "decayed_trend",
            "enough_tail_history",
            "tail_decay_path",
            "stable_trend_abs",
            "stable_trend_rel",
            "stable_trend",
            "stable_abs",
            "stable_rel",
            "stable",
        )
        status: Dict[str, Any] = {
            "current_value": current_values.get(key, [None] * len(self.alpha_values))[idx]
            if key in current_values else None
        }
        for field_name in fields:
            value = diagnostics.get(field_name)
            if isinstance(value, torch.Tensor) and idx < value.numel():
                item = value[idx].item()
                status[field_name] = bool(item) if value.dtype is torch.bool else float(item)
            else:
                status.setdefault(field_name, None)
        return status

    def _history_summary(self, *, tail: int = 5) -> Dict[str, Any]:
        steps = list(dict.fromkeys(self.history_order))
        return {
            "num_snapshots": len(self.history),
            "num_checks": int(self.check_count),
            "first_step": min(steps) if steps else None,
            "last_step": max(steps) if steps else None,
            "stored_step_tail": steps[-tail:],
            "effective_window_steps": int(self.config.effective_window_steps),
        }

    def _history_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for step in sorted(set(self.history_order)):
            snapshot = self.history[step]
            records.append({
                "step": int(snapshot.step),
                "Q_W": [float(value) for value in snapshot.q_w.tolist()],
                "Q_X": [float(value) for value in snapshot.q_x.tolist()],
                "R_W": [float(value) for value in snapshot.r_w.tolist()],
                "R_X": [float(value) for value in snapshot.r_x.tolist()],
            })
        return records


def _export_metric_plateau_history_enabled() -> bool:
    value = os.environ.get("MF_METRIC_PLATEAU_EXPORT_HISTORY", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def metric_plateau_candidate_profiles_for_size(
    N: int,
    *,
    patience: int = DEFAULT_PLATEAU_PATIENCE,
    abs_tol: float = DEFAULT_PLATEAU_ABS_TOL,
) -> List[MetricPlateauProfile]:
    """Return size-aware self-convergence profiles for calibration replay.

    Profiles are candidates only. Formal runs should pick a profile by replaying
    teacher-student metric history against a full-step/reference result.
    """

    n_int = int(N)
    if n_int <= 500:
        intervals = (100, 200)
        windows = (500, 1000, 2000)
    elif n_int <= 2000:
        intervals = (250, 500)
        windows = (1000, 2000, 4000)
    else:
        intervals = (500, 1000, 2000)
        windows = (2000, 4000, 8000, 12000)

    return [
        MetricPlateauProfile(
            check_interval=interval,
            window_steps=window,
            patience=patience,
            abs_tol=abs_tol,
        )
        for interval in intervals
        for window in windows
    ]


def metric_plateau_reference_step_cap_for_size(N: int) -> int:
    """Feasible reference cap for self-convergence calibration by size."""

    n_int = int(N)
    for max_n, cap in SIZE_AWARE_REFERENCE_STEP_CAPS:
        if n_int <= max_n:
            return int(cap)
    return int(SIZE_AWARE_REFERENCE_STEP_CAPS[-1][1])


def replay_metric_plateau_profile(
    snapshots: Sequence[MetricPlateauSnapshot],
    *,
    alpha_values: Sequence[float],
    profile: MetricPlateauProfile,
    configured_max_steps: int,
) -> Dict[str, Any]:
    stopper = MetricPlateauStopper(
        config=profile.to_config(enabled=True),
        alpha_values=[float(alpha) for alpha in alpha_values],
    )
    stopped_snapshot: Optional[MetricPlateauSnapshot] = None
    ordered = sorted(snapshots, key=lambda item: int(item.step))
    for snapshot in ordered:
        result = stopper.observe(snapshot)
        if result.stop:
            stopped_snapshot = snapshot
            break
    steps_run = int(stopper.stop_step or (ordered[-1].step if ordered else 0))
    summary = stopper.summary(
        steps_run=steps_run,
        configured_max_steps=int(configured_max_steps),
    )
    summary["selected_profile"] = profile.to_dict()
    if stopped_snapshot is not None:
        summary["stop_q_w"] = [float(value) for value in stopped_snapshot.q_w.tolist()]
    return summary


def select_metric_plateau_profile(
    snapshots: Sequence[MetricPlateauSnapshot],
    *,
    alpha_values: Sequence[float],
    reference_q_w: Sequence[float],
    candidates: Sequence[MetricPlateauProfile],
    configured_max_steps: int,
    max_qw_error: float = 0.01,
) -> Dict[str, Any]:
    """Replay candidate profiles and select the earliest Q_W-safe stop.

    This helper does not run AMP. It only replays recorded diagnostics, which is
    the calibration path for deciding whether a self-convergence profile is
    safe for a given size/alpha region. ``max_qw_error`` is an absolute Q_W value
    tolerance, not a relative percentage.
    """

    reference = torch.tensor(list(reference_q_w), dtype=torch.float32)
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for profile in candidates:
        summary = replay_metric_plateau_profile(
            snapshots,
            alpha_values=alpha_values,
            profile=profile,
            configured_max_steps=configured_max_steps,
        )
        stop_q_w = summary.get("stop_q_w")
        if stop_q_w is None:
            summary["calibration_rejection"] = "profile_did_not_stop"
            rejected.append(summary)
            continue
        error = (torch.tensor(stop_q_w, dtype=torch.float32) - reference).abs().max().item()
        summary["absolute_q_w_error"] = float(error)
        summary["absolute_q_w_error_tolerance"] = float(max_qw_error)
        if error < float(max_qw_error):
            accepted.append(summary)
        else:
            summary["calibration_rejection"] = "q_w_reference_error_exceeds_tolerance"
            rejected.append(summary)

    if not accepted:
        return {
            "selected": None,
            "accepted": [],
            "rejected": rejected,
            "absolute_q_w_error_tolerance": float(max_qw_error),
            "selection_rule": "earliest_stop_with_q_w_error_below_tolerance",
        }

    selected = min(
        accepted,
        key=lambda item: (
            int(item.get("steps_run", configured_max_steps)),
            int(item["selected_profile"]["window_steps"]),
            int(item["selected_profile"]["check_interval"]),
        ),
    )
    return {
        "selected": selected,
        "accepted": accepted,
        "rejected": rejected,
        "absolute_q_w_error_tolerance": float(max_qw_error),
        "selection_rule": "earliest_stop_with_q_w_error_below_tolerance",
    }
