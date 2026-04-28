import importlib.util
from pathlib import Path

import torch


def _load_posthoc_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "posthoc_scale_gauge.py"
    spec = importlib.util.spec_from_file_location("posthoc_scale_gauge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_posthoc_scale_gauge_resolves_group_alpha_payload_path(tmp_path):
    module = _load_posthoc_module()
    payload_path = tmp_path / "artifacts" / "points" / "0.0" / "results.pt"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text("placeholder", encoding="utf-8")

    resolved = module._resolve_point_payload_path(
        tmp_path,
        "p0000",
        {"coordinates": {"onsager_policy": "no_onsager", "alpha": 0.0}},
        {},
    )

    assert resolved == payload_path


def test_posthoc_scale_gauge_prefers_result_cube_artifact_pointer(tmp_path):
    module = _load_posthoc_module()
    payload_path = tmp_path / "custom" / "results.pt"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text("placeholder", encoding="utf-8")

    resolved = module._resolve_point_payload_path(
        tmp_path,
        "p0000",
        {"coordinates": {"alpha": 0.0}},
        {"p0000": {"point_results": "custom/results.pt"}},
    )

    assert resolved == payload_path


def test_posthoc_scale_gauge_returns_nan_for_nonfinite_factor():
    module = _load_posthoc_module()
    W_student = torch.ones(3, 2)
    X_student = torch.ones(2, 3)
    W_student[0, 0] = float("inf")
    W_teacher = torch.ones(3, 2)
    X_teacher = torch.ones(2, 3)

    q_w, q_x, gauge_mag = module._aligned_projection(
        W_student,
        X_student,
        W_teacher,
        X_teacher,
    )

    assert torch.isnan(torch.tensor(q_w))
    assert torch.isnan(torch.tensor(q_x))
    assert torch.isnan(torch.tensor(gauge_mag))
