import math

import pytest
import torch

from matrix_factorization.modules.algorithms.bigamp.plateau import (
    MetricPlateauProfile,
    MetricPlateauSnapshot,
    MetricPlateauStopConfig,
    MetricPlateauStopper,
    compute_teacher_latent_snapshot,
    metric_plateau_candidate_profiles_for_size,
    metric_plateau_reference_step_cap_for_size,
    select_metric_plateau_profile,
)


def _config(**kwargs):
    values = {
        "enabled": True,
        "check_interval": 1,
        "window_steps": 2,
        "patience": 2,
        "abs_tol": 0.003,
        "rel_tol": 0.01,
        "min_steps": 0,
    }
    values.update(kwargs)
    return MetricPlateauStopConfig(**values)


def _snapshot(step, values, *, q_x=None):
    vector = torch.tensor(values, dtype=torch.float32)
    x_vector = torch.tensor(q_x, dtype=torch.float32) if q_x is not None else vector.clone()
    return MetricPlateauSnapshot(
        step=step,
        q_w=vector,
        q_x=x_vector,
        r_w=vector.clone(),
        r_x=x_vector.clone(),
    )


def test_flat_history_stops_after_first_full_window_and_patience():
    stopper = MetricPlateauStopper(_config(patience=2), alpha_values=[2.5])
    stopper.observe(_snapshot(0, [1.0]))

    stopped_at = None
    for step in [1, 2, 3, 4, 5, 6]:
        result = stopper.observe(_snapshot(step, [1.0]))
        if result.stop:
            stopped_at = result.step
            break

    assert stopped_at == 5
    summary = stopper.summary(steps_run=stopped_at, configured_max_steps=20)
    assert summary["stop_reason"] == "metric_self_convergence"
    assert summary["config"]["strategy"] == "self_convergence_window_trend_decay"
    assert summary["config"]["effective_min_steps"] == 0
    assert summary["config"]["effective_earliest_check_step"] == 2


def test_history_records_export_is_env_gated(monkeypatch):
    stopper = MetricPlateauStopper(_config(patience=2), alpha_values=[2.5])
    stopper.observe(_snapshot(0, [1.0]))
    stopper.observe(_snapshot(1, [0.9]))

    assert "history_records" not in stopper.summary(steps_run=1, configured_max_steps=10)

    monkeypatch.setenv("MF_METRIC_PLATEAU_EXPORT_HISTORY", "1")
    summary = stopper.summary(steps_run=1, configured_max_steps=10)

    assert summary["history_records"] == [
        {"step": 0, "Q_W": [1.0], "Q_X": [1.0], "R_W": [1.0], "R_X": [1.0]},
        {
            "step": 1,
            "Q_W": pytest.approx([0.9]),
            "Q_X": pytest.approx([0.9]),
            "R_W": pytest.approx([0.9]),
            "R_X": pytest.approx([0.9]),
        },
    ]


def test_deprecated_min_steps_does_not_gate_self_convergence():
    stopper = MetricPlateauStopper(
        _config(check_interval=1, window_steps=1, patience=1, min_steps=100),
        alpha_values=[2.5],
    )
    stopper.observe(_snapshot(0, [1.0]))
    stopper.observe(_snapshot(1, [1.0]))
    result = stopper.observe(_snapshot(2, [1.0]))

    assert result.stop is True
    assert result.step == 2
    summary = stopper.summary(steps_run=2, configured_max_steps=100)
    assert summary["config"]["min_steps"] == 100
    assert summary["config"]["min_steps_deprecated"] is True
    assert summary["config"]["effective_min_steps"] == 0


def test_monotone_decreasing_then_flat_can_stop_without_direction_assumption():
    stopper = MetricPlateauStopper(_config(abs_tol=0.001, rel_tol=0.0), alpha_values=[1.0])
    series = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5}

    stopped_at = None
    for step, value in series.items():
        result = stopper.observe(_snapshot(step, [value]))
        if result.stop:
            stopped_at = result.step
            break

    assert stopped_at == 5


def test_monotone_increasing_then_flat_can_stop_without_direction_assumption():
    stopper = MetricPlateauStopper(_config(abs_tol=0.001, rel_tol=0.0), alpha_values=[1.0])
    series = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5}

    stopped_at = None
    for step, value in series.items():
        result = stopper.observe(_snapshot(step, [value]))
        if result.stop:
            stopped_at = result.step
            break

    assert stopped_at == 5


def test_linear_slow_drift_cannot_stop_while_range_and_slope_exceed_tol():
    stopper = MetricPlateauStopper(
        _config(abs_tol=0.0005, rel_tol=0.001, patience=2),
        alpha_values=[0.0],
    )
    stopper.observe(_snapshot(0, [0.0]))

    for step in range(1, 8):
        result = stopper.observe(_snapshot(step, [0.001 * step]))
        assert not result.stop

    summary = stopper.summary(steps_run=7, configured_max_steps=20)
    assert summary["stop_reason"] == "max_steps_reached"
    assert summary["per_alpha_status"][0]["plateau_satisfied"] is False


def test_constant_slow_drift_cannot_stop_even_when_each_window_is_small():
    stopper = MetricPlateauStopper(
        _config(abs_tol=0.01, rel_tol=0.05, patience=1),
        alpha_values=[0.0],
    )

    for step in range(0, 12):
        result = stopper.observe(_snapshot(step, [1.0 - 0.003 * step]))
        assert not result.stop

    metrics = stopper.summary(steps_run=11, configured_max_steps=20)["per_alpha_status"][0]["metrics"]["Q_W"]
    assert metrics["stable_abs"] is True
    assert metrics["stable_rel"] is True
    assert metrics["stable_trend"] is False
    assert metrics["stable"] is False


def test_delayed_drift_does_not_stop_before_tail_history():
    stopper = MetricPlateauStopper(
        _config(check_interval=1, window_steps=1, abs_tol=0.01, rel_tol=0.05, patience=1),
        alpha_values=[1.05],
    )

    value = 1.0
    for step in range(13):
        if step > 0:
            value -= 0.0006 if step <= 5 else 0.003
        result = stopper.observe(_snapshot(step, [value]))
        assert result.stop is False

    metrics = stopper.summary(steps_run=12, configured_max_steps=20)["per_alpha_status"][0]["metrics"]["Q_W"]
    assert metrics["stable_abs"] is True
    assert metrics["stable_rel"] is True
    assert metrics["enough_tail_history"] is True
    assert metrics["tail_decay_path"] is False
    assert metrics["stable"] is False


def test_drift_can_stop_after_slope_decay():
    stopper = MetricPlateauStopper(
        _config(check_interval=1, window_steps=1, abs_tol=0.01, rel_tol=0.05, patience=1),
        alpha_values=[0.0],
    )
    values = {
        0: 1.0000,
        1: 0.9980,
        2: 0.9960,
        3: 0.9940,
        4: 0.9920,
        5: 0.9900,
        6: 0.9880,
        7: 0.9860,
        8: 0.9848,
        9: 0.9841,
        10: 0.98365,
    }

    stopped_at = None
    for step, value in values.items():
        result = stopper.observe(_snapshot(step, [value]))
        if result.stop:
            stopped_at = result.step
            break

    assert stopped_at == 10
    metrics = stopper.summary(steps_run=10, configured_max_steps=20)["per_alpha_status"][0]["metrics"]["Q_W"]
    assert metrics["stable_abs"] is True
    assert metrics["stable_rel"] is True
    assert metrics["stable_trend"] is True
    assert metrics["tail_decay_path"] is True
    assert metrics["stable"] is True


def test_n20000_like_slow_drift_is_rejected_by_slope_abs_gate():
    stopper = MetricPlateauStopper(
        _config(check_interval=1, window_steps=5, abs_tol=0.01, rel_tol=0.05, patience=1),
        alpha_values=[0.9],
    )
    value_at_break = 0.0857 - 0.0002094 * 8
    values = {
        step: (
            0.0857 - 0.0002094 * step
            if step <= 8
            else value_at_break - 0.0001968 * (step - 8)
        )
        for step in range(14)
    }

    result = None
    for step, value in values.items():
        result = stopper.observe(_snapshot(step, [value]))
        assert result.stop is False

    assert result is not None
    metrics = stopper.summary(steps_run=13, configured_max_steps=20)["per_alpha_status"][0]["metrics"]["Q_W"]
    assert metrics["projected_abs_change"] == pytest.approx(0.000984, rel=0.02)
    assert metrics["trend_abs_ratio"] == pytest.approx(0.94, rel=0.03)
    assert metrics["stable_abs"] is True
    assert metrics["stable_rel"] is True
    assert metrics["small_slope"] is False
    assert metrics["stable_trend"] is False
    assert metrics["stable"] is False


def test_q_x_drift_prevents_stop_when_q_w_is_stable():
    stopper = MetricPlateauStopper(
        _config(abs_tol=0.001, rel_tol=0.001, patience=1),
        alpha_values=[1.0],
    )

    for step in range(0, 7):
        result = stopper.observe(_snapshot(step, [0.5], q_x=[0.01 * step]))
        assert not result.stop

    summary = stopper.summary(steps_run=6, configured_max_steps=10)
    assert summary["per_alpha_status"][0]["metrics"]["Q_W"]["stable"] is True
    assert summary["per_alpha_status"][0]["metrics"]["Q_X"]["stable"] is False


def test_large_oscillation_does_not_stop_even_when_endpoint_delta_is_small():
    stopper = MetricPlateauStopper(
        _config(abs_tol=0.01, rel_tol=0.01, patience=1),
        alpha_values=[1.0],
    )
    for step, value in {0: 0.0, 1: 0.1, 2: 0.0, 3: 0.1, 4: 0.0}.items():
        result = stopper.observe(_snapshot(step, [value]))
        assert not result.stop

    metrics = stopper.summary(steps_run=4, configured_max_steps=10)["per_alpha_status"][0]["metrics"]
    assert metrics["Q_W"]["endpoint_abs_delta"] == pytest.approx(0.0)
    assert metrics["Q_W"]["window_abs_range"] > 0.01
    assert metrics["Q_W"]["stable"] is False


def test_relative_gate_alone_cannot_stop_when_absolute_gate_fails():
    stopper = MetricPlateauStopper(
        _config(abs_tol=1e-8, rel_tol=0.1, patience=1),
        alpha_values=[1.0],
    )
    result = None
    for step, value in enumerate([0.0001, 0.0002, 0.0003, 0.00031, 0.00032]):
        result = stopper.observe(_snapshot(step, [value]))

    assert result is not None
    assert result.stop is False
    metrics = stopper.summary(steps_run=4, configured_max_steps=5)["per_alpha_status"][0]["metrics"]
    assert metrics["Q_W"]["stable_abs"] is False
    assert metrics["Q_W"]["stable_rel"] is True
    assert metrics["Q_W"]["stable"] is False


def test_absolute_and_relative_gates_must_both_pass_to_stop():
    stopper = MetricPlateauStopper(
        _config(abs_tol=0.001, rel_tol=0.1, patience=1),
        alpha_values=[1.0],
    )
    result = None
    for step, value in enumerate([0.0001, 0.0002, 0.0003, 0.0003, 0.0003]):
        result = stopper.observe(_snapshot(step, [value]))

    assert result is not None
    assert result.stop is True
    metrics = stopper.summary(steps_run=4, configured_max_steps=5)["per_alpha_status"][0]["metrics"]
    assert metrics["Q_W"]["stable_abs"] is True
    assert metrics["Q_W"]["stable_rel"] is True
    assert metrics["Q_W"]["stable_trend"] is True
    assert metrics["Q_W"]["near_zero_fast_path"] is True
    assert metrics["Q_W"]["stable"] is True


def test_nonfinite_monitored_values_fail_never_count_as_plateau():
    stopper = MetricPlateauStopper(_config(), alpha_values=[1.5])

    with pytest.raises(FloatingPointError, match="Q_W"):
        stopper.observe(_snapshot(0, [math.nan]))


def test_mixed_alpha_batch_stops_only_after_all_alphas_are_stable():
    stopper = MetricPlateauStopper(
        _config(abs_tol=0.001, rel_tol=0.001, patience=1),
        alpha_values=[2.5, 0.75],
    )
    series = {
        0: [1.0, 0.0],
        1: [1.0, 0.01],
        2: [1.0, 0.02],
        3: [1.0, 0.03],
        4: [1.0, 0.04],
        5: [1.0, 0.04],
        6: [1.0, 0.04],
    }

    stopped_at = None
    for step, values in series.items():
        result = stopper.observe(_snapshot(step, values))
        if step == 4:
            assert stopper.stable_counts.tolist() == [1, 0]
            assert stopper.ready_steps.tolist() == [4, -1]
        if result.stop:
            stopped_at = result.step
            break

    assert stopped_at == 6
    statuses = stopper.summary(steps_run=6, configured_max_steps=20)["per_alpha_status"]
    assert [item["ready_step"] for item in statuses] == [4, 6]
    assert all(item["plateau_satisfied"] for item in statuses)


def test_teacher_latent_snapshot_uses_fixed_denominator_schema_v5():
    W_teacher = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    X_teacher = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    W_flat = W_teacher.reshape(1, 1 * 2, 2).clone()
    X_flat = X_teacher.T.reshape(1, 1 * 2, 2).clone()

    snapshot = compute_teacher_latent_snapshot(
        step=0,
        W_flat=W_flat,
        X_flat=X_flat,
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        S=1,
        N1=2,
        N2=2,
        M=2,
    )

    assert snapshot.q_w.item() == pytest.approx(float((W_teacher * W_teacher).mean()))
    assert snapshot.q_x.item() == pytest.approx(float((X_teacher * X_teacher).mean()))
    assert snapshot.r_w.item() == pytest.approx(float((W_teacher * W_teacher).mean()))
    assert snapshot.r_x.item() == pytest.approx(float((X_teacher * X_teacher).mean()))


def test_size_aware_candidate_profiles_are_valid_integer_windows():
    small = metric_plateau_candidate_profiles_for_size(200)
    medium = metric_plateau_candidate_profiles_for_size(2000)
    large = metric_plateau_candidate_profiles_for_size(20000)

    assert (small[0].check_interval, small[0].window_steps) == (100, 500)
    assert all(profile.window_steps > 0 for profile in small + medium + large)
    assert [profile.window_steps for profile in small[:3]] == [500, 1000, 2000]
    assert [profile.check_interval for profile in large[:4]] == [500, 500, 500, 500]
    assert large[3].window_steps == 12000
    assert metric_plateau_reference_step_cap_for_size(200) == 3000
    assert metric_plateau_reference_step_cap_for_size(500) == 5000
    assert metric_plateau_reference_step_cap_for_size(20000) == 48000


def test_profile_calibration_rejects_early_stop_with_large_reference_error():
    snapshots = [
        _snapshot(0, [0.0]),
        _snapshot(1, [0.0]),
        _snapshot(2, [0.0]),
        _snapshot(3, [0.050]),
    ]

    selection = select_metric_plateau_profile(
        snapshots,
        alpha_values=[0.0],
        reference_q_w=[0.050],
        candidates=[MetricPlateauProfile(check_interval=1, window_steps=1, patience=1, abs_tol=0.01, rel_tol=99.0)],
        configured_max_steps=3,
        max_qw_error=0.01,
    )

    assert selection["selected"] is None
    assert selection["rejected"][0]["calibration_rejection"] == "q_w_reference_error_exceeds_tolerance"
    assert selection["absolute_q_w_error_tolerance"] == 0.01
    assert selection["rejected"][0]["absolute_q_w_error"] >= 0.01
