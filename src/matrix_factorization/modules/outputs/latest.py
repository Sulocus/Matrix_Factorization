"""Maintain a lightweight latest-results view for repository display."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_LATEST_COUNT = 3
PREFERRED_PLOTS = ("custom_plot_1.png", "overview.png", "summary.png")


def default_latest_dir(source_dir: Path) -> Path:
    """Return the display directory paired with an experiment output directory."""
    source_dir = Path(source_dir)
    if source_dir.name == "alpha_scan":
        return source_dir.parent / "latest"
    return source_dir / "latest"


def refresh_latest_results(
    source_dir: Path,
    latest_dir: Optional[Path] = None,
    count: int = DEFAULT_LATEST_COUNT,
) -> List[Dict[str, Any]]:
    """Refresh a small tracked view of the newest experiment runs.

    The output intentionally excludes tensor/checkpoint artifacts. It is safe to
    keep under git for README-facing summaries.
    """
    source_dir = Path(source_dir)
    latest_dir = Path(latest_dir) if latest_dir is not None else default_latest_dir(source_dir)

    runs = _newest_run_dirs(source_dir, count)
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for run_dir in runs:
        run_summary = _export_run(run_dir, latest_dir / run_dir.name)
        summaries.append(run_summary)

    index = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "latest_dir": str(latest_dir),
        "max_count": count,
        "count": len(summaries),
        "runs": summaries,
    }
    _write_json(latest_dir / "index.json", index)
    return summaries


def _newest_run_dirs(source_dir: Path, count: int) -> List[Path]:
    if not source_dir.exists():
        return []

    candidates = [
        path
        for path in source_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and path.name != "latest"
        and ((path / "config.json").exists() or (path / "metadata.json").exists())
    ]
    return sorted(candidates, key=_run_sort_key, reverse=True)[:count]


def _run_sort_key(path: Path) -> tuple:
    timestamp = _parse_run_timestamp(path.name)
    if timestamp is not None:
        return (timestamp, path.name)
    return (datetime.fromtimestamp(path.stat().st_mtime), path.name)


def _parse_run_timestamp(run_id: str) -> Optional[datetime]:
    try:
        return datetime.strptime(run_id[:13], "%Y%m%d_%H%M")
    except ValueError:
        return None


def _export_run(run_dir: Path, target_dir: Path) -> Dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(run_dir / "config.json")
    metadata = _load_json(run_dir / "metadata.json")

    for filename in ("config.json", "metadata.json"):
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)

    selected_plot = _copy_selected_plot(run_dir, target_dir)
    summary = _build_summary(run_dir, config, metadata, selected_plot)
    _write_json(target_dir / "summary.json", summary)
    return summary


def _build_summary(
    run_dir: Path,
    config: Dict[str, Any],
    metadata: Dict[str, Any],
    selected_plot: Optional[str],
) -> Dict[str, Any]:
    scan = config.get("scan", {})
    scan_values = scan.get("values") or []
    matrix = config.get("matrix", {})
    training = config.get("training", {})
    spreading = config.get("spreading", {})
    results_pt = run_dir / "results.pt"

    summary: Dict[str, Any] = {
        "run_id": run_dir.name,
        "source_run_dir": str(run_dir),
        "timestamp": metadata.get("timestamp"),
        "algorithm": config.get("algorithm_key") or config.get("algorithm"),
        "teacher": config.get("teacher_key") or config.get("teacher"),
        "tensor_order": spreading.get("tensor_order") or config.get("tensor_order"),
        "matrix": {
            "N1": matrix.get("N1"),
            "N2": matrix.get("N2"),
            "M": matrix.get("M"),
        },
        "training": {
            "samples_per_alpha": training.get("samples_per_alpha"),
            "max_steps": training.get("max_steps"),
        },
        "scan": {
            "dimension": scan.get("dimension"),
            "count": len(scan_values),
            "min": min(scan_values) if scan_values else None,
            "max": max(scan_values) if scan_values else None,
        },
        "hardware": {
            "gpu_name": metadata.get("gpu_name"),
            "gpu_memory_gb": metadata.get("gpu_memory_gb"),
        },
        "artifacts": {
            "results_pt_omitted": results_pt.exists(),
            "source_results_pt_bytes": results_pt.stat().st_size if results_pt.exists() else None,
            "selected_plot": selected_plot,
        },
    }
    return summary


def _copy_selected_plot(run_dir: Path, target_dir: Path) -> Optional[str]:
    source = _select_plot(run_dir / "plots")
    if source is None:
        return None

    plots_dir = target_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    target = plots_dir / "selected_overview.png"
    shutil.copy2(source, target)
    return str(Path("plots") / target.name)


def _select_plot(plots_dir: Path) -> Optional[Path]:
    if not plots_dir.exists():
        return None

    for filename in PREFERRED_PLOTS:
        candidate = plots_dir / filename
        if candidate.exists():
            return candidate

    pngs = sorted(_small_pngs(plots_dir))
    non_heatmaps = [path for path in pngs if not path.name.startswith("heatmap_")]
    if non_heatmaps:
        return non_heatmaps[0]
    return pngs[0] if pngs else None


def _small_pngs(plots_dir: Path, max_bytes: int = 2_000_000) -> Iterable[Path]:
    for path in plots_dir.glob("*.png"):
        if path.is_file() and path.stat().st_size <= max_bytes:
            yield path


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
