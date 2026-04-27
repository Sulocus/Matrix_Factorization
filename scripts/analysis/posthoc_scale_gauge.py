"""Posthoc scale-gauge diagnostics for matrix spreading factor payloads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_gauge_scalar(w_s: np.ndarray, x_s: np.ndarray, w_t: np.ndarray, x_t: np.ndarray) -> float:
    a = float(np.dot(w_s, w_s))
    b = float(np.dot(w_s, w_t))
    c = float(np.dot(x_s, x_s))
    d = float(np.dot(x_s, x_t))
    if a <= 1e-18 or c <= 1e-18:
        return 1.0

    candidates = []
    roots = np.roots([a, -b, 0.0, d, -c])
    for root in roots:
        if abs(root.imag) < 1e-8 and abs(root.real) > 1e-12:
            candidates.append(float(root.real))
    scale = float(np.sqrt(c / a))
    candidates.extend([scale, -scale, 1.0, -1.0])

    def objective(g: float) -> float:
        return float(np.sum((g * w_s - w_t) ** 2) + np.sum((x_s / g - x_t) ** 2))

    return min(candidates, key=objective)


def _aligned_projection(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
) -> tuple[float, float, float]:
    W_s = W_student.detach().cpu().float().numpy()
    X_s = X_student.detach().cpu().float().numpy()
    W_t = W_teacher.detach().cpu().float().numpy()
    X_t = X_teacher.detach().cpu().float().numpy()
    m = W_t.shape[1]

    W_aligned = np.empty_like(W_s)
    X_aligned = np.empty_like(X_s)
    gauges = np.empty(m, dtype=np.float64)
    for k in range(m):
        g = _best_gauge_scalar(W_s[:, k], X_s[k, :], W_t[:, k], X_t[k, :])
        gauges[k] = g
        W_aligned[:, k] = g * W_s[:, k]
        X_aligned[k, :] = X_s[k, :] / g

    q_w = float(np.sum(W_aligned * W_t) / (np.sum(W_t * W_t) + 1e-12))
    q_x = float(np.sum(X_aligned * X_t) / (np.sum(X_t * X_t) + 1e-12))
    median_abs_log_g = float(np.median(np.abs(np.log(np.maximum(np.abs(gauges), 1e-12)))))
    return q_w, q_x, median_abs_log_g


def _teacher_from_artifacts(run_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the exact teacher factors saved by the run.

    Seed-based reconstruction is intentionally not used here: canonical child
    runs can alter seeding and batching, so a reconstructed teacher can be the
    right shape but the wrong physical object for gauge diagnostics.
    """
    candidate_paths = [
        run_dir / "artifacts" / "teacher_factors.pt",
        run_dir / "artifacts" / "results.pt",
        run_dir / "results.pt",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            continue
        W_teacher = payload.get("W_teacher")
        X_teacher = payload.get("X_teacher")
        if W_teacher is not None and X_teacher is not None:
            return W_teacher.detach().cpu().float(), X_teacher.detach().cpu().float()

    raise FileNotFoundError(
        f"{run_dir} does not contain artifacts/teacher_factors.pt with W_teacher/X_teacher. "
        "Rerun with output.save_tensors: true after the teacher-factor artifact patch."
    )


def _plot(rows: list[dict[str, Any]], key: str, ylabel: str, output_path: Path) -> None:
    policies = sorted({row["onsager_policy"] for row in rows})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in policies:
        sub = sorted((row for row in rows if row["onsager_policy"] == policy), key=lambda row: row["alpha"])
        ax.plot(
            [row["alpha"] for row in sub],
            [row[key] for row in sub],
            marker="o",
            linewidth=1.5,
            label=policy,
        )
    ax.set_xlabel("alpha")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-subdir", default="plots/posthoc_scale_gauge")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics = _load_json(run_dir / "metrics.json")
    W_teacher, X_teacher = _teacher_from_artifacts(run_dir)

    output_dir = run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for point_id, point in sorted(metrics["result_cube"]["points"].items()):
        payload_path = run_dir / "artifacts" / "points" / point_id / "results.pt"
        if not payload_path.exists():
            continue
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        W_students = payload["W_students"]
        X_students = payload["X_students"]
        q_w_values = []
        q_x_values = []
        gauge_values = []
        for sample_idx in range(W_students.shape[0]):
            q_w, q_x, gauge_mag = _aligned_projection(
                W_students[sample_idx],
                X_students[sample_idx],
                W_teacher,
                X_teacher,
            )
            q_w_values.append(q_w)
            q_x_values.append(q_x)
            gauge_values.append(gauge_mag)

        row = {
            "point_id": point_id,
            "onsager_policy": point["coordinates"]["onsager_policy"],
            "alpha": float(point["coordinates"]["alpha"]),
            "Q_W_SCALE_GAUGE_mean": float(np.mean(q_w_values)),
            "Q_W_SCALE_GAUGE_std": float(np.std(q_w_values, ddof=1)) if len(q_w_values) > 1 else 0.0,
            "Q_X_SCALE_GAUGE_mean": float(np.mean(q_x_values)),
            "Q_X_SCALE_GAUGE_std": float(np.std(q_x_values, ddof=1)) if len(q_x_values) > 1 else 0.0,
            "Q_WX_SCALE_GAUGE_mean": float(0.5 * (np.mean(q_w_values) + np.mean(q_x_values))),
            "median_abs_log_g_mean": float(np.mean(gauge_values)),
        }
        rows.append(row)

    (output_dir / "scale_gauge_metrics.json").write_text(
        json.dumps({"source_run": str(run_dir), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "scale_gauge_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["point_id"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        _plot(rows, "Q_W_SCALE_GAUGE_mean", "Q_W scale-gauge aligned", output_dir / "qw_scale_gauge.png")
        _plot(rows, "Q_X_SCALE_GAUGE_mean", "Q_X scale-gauge aligned", output_dir / "qx_scale_gauge.png")
        _plot(rows, "Q_WX_SCALE_GAUGE_mean", "mean(Q_W,Q_X) scale-gauge aligned", output_dir / "qwx_scale_gauge.png")
        _plot(rows, "median_abs_log_g_mean", "median |log |g||", output_dir / "gauge_magnitude.png")

    print(f"Wrote {len(rows)} point diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
