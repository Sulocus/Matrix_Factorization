"""
Experiment Result Data Structures.

Unified result storage for all experiment types, enabling:
1. Consistent format across scan modes
2. Complete reproducibility (config saved with results)
3. Easy analysis and visualization
"""

from dataclasses import dataclass, field, asdict
from typing import List, Any, Optional, Dict, Union
from pathlib import Path
from datetime import datetime
import json
import math
import torch


METRIC_DEFINITION_POLICY = {
    "metric_definition_profile": "projection_qy_physical_latent_v2",
    "Q_Y_formula": "absolute_projection_teacher_norm_squared",
    "Q_W_Q_X_normalization": "fixed_coordinate_count",
    "legacy_projection_suffix": "_PROJ_ABS",
    "teacher_norm_epsilon": 1e-12,
    "degenerate_teacher_norm": "return_zero",
    "clipped": False,
}

METRIC_KEY_ALIASES = {
    "Q_W_SIGN_ALIGNED": "Q_W_SIGN_GAUGE",
    "Q_X_SIGN_ALIGNED": "Q_X_SIGN_GAUGE",
    "Q_W_SIGN_ALIGNED_mean": "Q_W_SIGN_GAUGE_mean",
    "Q_W_SIGN_ALIGNED_std": "Q_W_SIGN_GAUGE_std",
    "Q_X_SIGN_ALIGNED_mean": "Q_X_SIGN_GAUGE_mean",
    "Q_X_SIGN_ALIGNED_std": "Q_X_SIGN_GAUGE_std",
    "Q_W_SIGN_GAUGE": "Q_W_SIGN_ALIGNED",
    "Q_X_SIGN_GAUGE": "Q_X_SIGN_ALIGNED",
    "Q_W_SIGN_GAUGE_mean": "Q_W_SIGN_ALIGNED_mean",
    "Q_W_SIGN_GAUGE_std": "Q_W_SIGN_ALIGNED_std",
    "Q_X_SIGN_GAUGE_mean": "Q_X_SIGN_ALIGNED_mean",
    "Q_X_SIGN_GAUGE_std": "Q_X_SIGN_ALIGNED_std",
    "median_abs_log_g": "median_abs_log_k",
    "median_abs_log_g_mean": "median_abs_log_k_mean",
    "median_abs_log_g_std": "median_abs_log_k_std",
    "Q_W_GRAM_ROOT": "Q_W_COS_ROOT",
    "Q_X_GRAM_ROOT": "Q_X_COS_ROOT",
    "Q_W_GRAM_ROOT_mean": "Q_W_COS_ROOT_mean",
    "Q_W_GRAM_ROOT_std": "Q_W_COS_ROOT_std",
    "Q_X_GRAM_ROOT_mean": "Q_X_COS_ROOT_mean",
    "Q_X_GRAM_ROOT_std": "Q_X_COS_ROOT_std",
    "Q_W_COS_ROOT": "Q_W_GRAM_ROOT",
    "Q_X_COS_ROOT": "Q_X_GRAM_ROOT",
    "Q_W_COS_ROOT_mean": "Q_W_GRAM_ROOT_mean",
    "Q_W_COS_ROOT_std": "Q_W_GRAM_ROOT_std",
    "Q_X_COS_ROOT_mean": "Q_X_GRAM_ROOT_mean",
    "Q_X_COS_ROOT_std": "Q_X_GRAM_ROOT_std",
}


def metric_key_candidates(metric_key: str) -> List[str]:
    """Return canonical/legacy lookup candidates for a metric key."""
    alias = METRIC_KEY_ALIASES.get(metric_key)
    return [metric_key, alias] if alias and alias != metric_key else [metric_key]


def _with_projection_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(contract or {})
    payload.setdefault("metric_definition_policy", dict(METRIC_DEFINITION_POLICY))
    return payload


@dataclass
class Checkpoint:
    """
    Algorithm state checkpoint for resuming or continuing.

    Used for:
    1. Scanning steps: continue from previous checkpoint
    2. Error recovery: resume from last good state
    """
    step: int
    W_state: Optional[torch.Tensor]  # (S, N1, M) or (A, S, N1, M)
    X_state: Optional[torch.Tensor]  # (S, M, N2) or (A, S, M, N2)

    # Optional variance states (BiGAMP)
    W_var: Optional[torch.Tensor] = None
    X_var: Optional[torch.Tensor] = None

    # Additional algorithm state
    extra: Optional[Dict[str, Any]] = None

    def save(self, path: Path):
        """Save checkpoint to file."""
        path = Path(path)
        torch.save({
            'step': self.step,
            'W_state': self.W_state,
            'X_state': self.X_state,
            'W_var': self.W_var,
            'X_var': self.X_var,
            'extra': self.extra,
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device = None) -> 'Checkpoint':
        """Load checkpoint from file."""
        data = torch.load(path, map_location=device)
        return cls(
            step=data['step'],
            W_state=data['W_state'],
            X_state=data['X_state'],
            W_var=data.get('W_var'),
            X_var=data.get('X_var'),
            extra=data.get('extra'),
        )


@dataclass
class SingleRunResult:
    """
    Result from a single algorithm run.

    Contains:
    - metrics: computed overlaps (Q_W, Q_X, Q_Y, etc.)
    - student matrices: trained W, X
    - raw data: mask, observation indices (for Q_Y_unobserved)
    """
    # Scan point identifier
    scan_value: Any  # e.g., alpha=1.5 or steps=1000

    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    # Expected keys: Q_W_mean, Q_W_std, Q_X_mean, Q_X_std, Q_Y_mean, Q_Y_std, Q_Y_unobserved

    # Student matrices (trained results)
    W_students: Optional[torch.Tensor] = None  # (S, N1, M)
    X_students: Optional[torch.Tensor] = None  # (S, M, N2)

    # ========== RAW DATA (for post-hoc analysis) ==========
    # Observation mask (for Q_Y_unobserved computation)
    mask: Optional[torch.Tensor] = None  # (N1, N2) or (S, N1, N2)

    # Spreading-specific: observation indices
    observation_indices: Optional[Dict[str, torch.Tensor]] = None
    # Keys: 'i_idx', 'j_idx', 'edge_counts' (per alpha)

    # Optional: convergence history
    history: Optional[List[Dict[str, float]]] = None

    # MetricSpec coverage for this flat metric payload
    metric_contract: Dict[str, Any] = field(default_factory=dict)

    # Timing
    duration_seconds: float = 0.0

    def to_dict(self, include_tensors: bool = False) -> Dict:
        """Convert to dictionary."""
        d = {
            'scan_value': self.scan_value,
            'metrics': self.metrics,
            'duration_seconds': self.duration_seconds,
        }
        if self.history:
            d['history'] = self.history
        if self.metric_contract:
            d['metric_contract'] = _with_projection_policy(self.metric_contract)
        if include_tensors and self.W_students is not None:
            d['has_tensors'] = True
        return d

    @property
    def Q_W(self) -> float:
        """Teacher-Student overlap for W."""
        return self.metrics.get('Q_W_mean', 0.0)

    @property
    def Q_X(self) -> float:
        """Teacher-Student overlap for X."""
        return self.metrics.get('Q_X_mean', 0.0)

    @property
    def Q_Y(self) -> float:
        """Prediction overlap."""
        return self.metrics.get('Q_Y_mean', 0.0)


@dataclass
class ResultCubePoint:
    point_id: str
    coordinates: Dict[str, Any]
    overrides: Dict[str, Any] = field(default_factory=dict)
    effective_config_hash: str = ""
    group_id: str = "default"
    metric_contract: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResultCube:
    axes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    points: Dict[str, ResultCubePoint] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metric_semantics: Dict[str, Any] = field(default_factory=dict)

    def add_point(
        self,
        point_id: str,
        coordinates: Dict[str, Any],
        metrics: Dict[str, Any],
        *,
        overrides: Optional[Dict[str, Any]] = None,
        effective_config_hash: str = "",
        group_id: str = "default",
        metric_contract: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.points[point_id] = ResultCubePoint(
            point_id=point_id,
            coordinates=dict(coordinates),
            overrides=dict(overrides or {}),
            effective_config_hash=effective_config_hash,
            group_id=group_id,
            metric_contract=dict(metric_contract or {}),
        )
        self.metrics[point_id] = dict(metrics or {})
        if artifacts:
            self.artifacts[point_id] = dict(artifacts)
        self.groups.setdefault(group_id, {"point_ids": []})
        if point_id not in self.groups[group_id]["point_ids"]:
            self.groups[group_id]["point_ids"].append(point_id)

    def is_empty(self) -> bool:
        return not self.points

    def is_alpha_only(self) -> bool:
        return set(self.axes.keys()) <= {"alpha"} if self.axes else True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "axes": dict(self.axes),
            "points": {key: point.to_dict() for key, point in self.points.items()},
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "groups": dict(self.groups),
            "metric_semantics": dict(self.metric_semantics),
        }

    def resolve_plot_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve a PlotQuery into concrete point ids and curve values.

        This is the hard query boundary used by plotting and tests. It fails
        when a query is ambiguous, references missing coordinates/metrics, or
        would silently merge multiple points with the same x value.
        """
        x_axis = query.get("x")
        metric_key = query.get("y")
        if not x_axis or not metric_key:
            raise ValueError("PlotQuery requires x and y")
        metric_candidates = metric_key_candidates(str(metric_key))
        if x_axis not in self.axes:
            raise ValueError(f"PlotQuery x axis '{x_axis}' is not in ResultCube axes")

        where = dict(query.get("where") or {})
        selected = self._select_points(where)
        compare = list(query.get("compare") or [])
        series_by = list(query.get("series_by") or [])

        if compare:
            series_specs = [dict(item) for item in compare]
        elif series_by:
            self._validate_axes(series_by, "series_by")
            seen = []
            for point in selected:
                key = tuple((axis, point.coordinates.get(axis)) for axis in series_by)
                if key not in seen:
                    seen.append(key)
            series_specs = [{axis: value for axis, value in key} for key in seen]
        else:
            series_specs = [{}]

        resolved = []
        for series_spec in series_specs:
            self._validate_axes(series_spec, "series")
            points = [
                point for point in selected
                if all(point.coordinates.get(key) == value for key, value in series_spec.items())
            ]
            points = sorted(points, key=lambda point: self._sort_key(point.coordinates.get(x_axis)))
            if not points:
                raise ValueError(f"PlotQuery selected no points for series {series_spec}")

            x_values = []
            y_values = []
            y_std_values = []
            point_ids = []
            seen_x = {}
            std_candidates = [
                key[:-5] + "_std"
                for key in metric_candidates
                if key and key.endswith("_mean")
            ]
            for point in points:
                if x_axis not in point.coordinates:
                    raise ValueError(f"PlotQuery x axis '{x_axis}' not found in point {point.point_id}")
                metrics = self.metrics.get(point.point_id, {})
                actual_metric_key = next((key for key in metric_candidates if key in metrics), None)
                if actual_metric_key is None:
                    raise ValueError(f"PlotQuery metric '{metric_key}' missing for point {point.point_id}")
                x_value = point.coordinates[x_axis]
                if x_value in seen_x:
                    raise ValueError(
                        f"PlotQuery series {series_spec} has duplicate x={x_value!r} "
                        f"for points {seen_x[x_value]!r} and {point.point_id!r}; "
                        "constrain where/compare or add series_by axes."
                    )
                seen_x[x_value] = point.point_id
                point_ids.append(point.point_id)
                x_values.append(float(x_value))
                actual_std_key = next((key for key in std_candidates if key in metrics), None)
                y_values.append(float(metrics[actual_metric_key]))
                y_std_values.append(float(metrics.get(actual_std_key, 0.0)) if actual_std_key else 0.0)

            label = ", ".join(f"{key}={value}" for key, value in series_spec.items()) or str(metric_key)
            resolved.append({
                "label": label,
                "coordinates": dict(series_spec),
                "point_ids": point_ids,
                "x_values": x_values,
                "y_values": y_values,
                "y_std_values": y_std_values,
            })
        return resolved

    def _select_points(self, where: Dict[str, Any]) -> List[ResultCubePoint]:
        self._validate_axes(where, "where")
        points = list(self.points.values())
        for axis, value in where.items():
            points = [point for point in points if point.coordinates.get(axis) == value]
        if not points:
            raise ValueError(f"PlotQuery where selected no points: {where}")
        return points

    def _validate_axes(self, payload: Any, context: str) -> None:
        axes = payload if isinstance(payload, list) else list((payload or {}).keys())
        for axis in axes:
            if axis not in self.axes:
                raise ValueError(f"PlotQuery {context} references unknown axis '{axis}'")

    @staticmethod
    def _sort_key(value: Any):
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))


@dataclass
class ExperimentMetadata:
    """Metadata about the experiment run."""
    timestamp: str = ""
    duration_total_seconds: float = 0.0
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    smf_version: str = "1.0.0"
    python_version: str = ""
    torch_version: str = ""
    contract: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_now(cls) -> 'ExperimentMetadata':
        """Create metadata with current system info."""
        import sys

        gpu_name = ""
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return cls(
            timestamp=datetime.now().isoformat(),
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory,
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """
    Complete experiment result with all metadata.

    This is the unified result format for all experiment types.

    Usage:
        result = ExperimentResult(
            experiment_id="exp_001",
            config=config,
            scan_dimension="alpha",
            scan_values=[0.0, 0.5, 1.0],
        )
        result.add_result(0.0, single_result)
        result.add_result(0.5, single_result)
        result.save('runs/exp_001/')
    """
    # Identification
    experiment_id: str

    # Full configuration (for reproducibility)
    config: Any  # ExperimentConfig (Any to avoid circular import)

    # Scan information
    scan_dimension: str
    scan_values: List[Any]

    # Results indexed by scan value
    results: Dict[Any, SingleRunResult] = field(default_factory=dict)

    # Metadata
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    # ========== RAW DATA (for post-hoc analysis) ==========
    # Teacher matrices (ground truth)
    W_teacher: Optional[torch.Tensor] = None  # (N1, M)
    X_teacher: Optional[torch.Tensor] = None  # (M, N2)
    Y_teacher: Optional[torch.Tensor] = None  # (N1, N2)

    # All masks (for multi-alpha experiments)
    all_masks: Optional[torch.Tensor] = None  # (num_alphas, N1, N2)

    # Spreading-specific: SuperGraph data
    supergraph_data: Optional[Dict[str, Any]] = None
    # Keys: 'i_idx', 'j_idx', 'edge_counts', 'F_super' (if needed)

    # Canonical multi-axis result index.
    result_cube: ResultCube = field(default_factory=ResultCube)

    def add_result(self, scan_value: Any, result: SingleRunResult):
        """Add a single run result."""
        self.results[scan_value] = result

    @property
    def num_completed(self) -> int:
        """Number of completed scan points."""
        return len(self.results)

    @property
    def is_complete(self) -> bool:
        """Check if all scan points are completed."""
        return self.num_completed == len(self.scan_values)

    @property
    def completion_ratio(self) -> float:
        """Completion ratio (0.0 to 1.0)."""
        return self.num_completed / len(self.scan_values) if self.scan_values else 0.0

    def get_metric_curve(self, metric_name: str) -> Dict[Any, float]:
        """Get a metric across all scan values."""
        candidates = metric_key_candidates(metric_name)
        return {
            v: next((r.metrics.get(key) for key in candidates if key in (r.metrics or {})), 0.0)
            for v, r in sorted(self.results.items())
        }

    @staticmethod
    def _sort_key(value: Any) -> Any:
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _json_default(obj: Any):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except (TypeError, ValueError):
                pass
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    def _sorted_result_items(self):
        return sorted(self.results.items(), key=lambda item: self._sort_key(item[0]))

    def to_metrics_dict(self) -> Dict[str, Dict[str, float]]:
        """Canonical JSON-friendly metric mapping: scan value -> metric dict."""
        return {
            str(scan_value): result.metrics
            for scan_value, result in self._sorted_result_items()
        }

    def to_results_dict(self) -> Dict[str, Dict[str, Any]]:
        """Canonical lightweight result mapping without tensor payloads."""
        return {
            str(scan_value): result.to_dict(include_tensors=False)
            for scan_value, result in self._sorted_result_items()
        }

    def metric_contracts(self) -> Dict[str, Dict[str, Any]]:
        """MetricSpec coverage for each scan value."""
        contracts = {}
        algorithm_key = getattr(self.config, "algorithm_key", None)
        for scan_value, result in self._sorted_result_items():
            if not algorithm_key:
                continue
            try:
                from matrix_factorization.modules.metrics.spec_adapter import MetricSpecAdapter
                validation = MetricSpecAdapter.validate_payload(
                    algorithm_key,
                    result.metrics,
                    source=(
                        result.metric_contract.get("source", "result_metric_contract")
                        if result.metric_contract else "result_schema_fallback"
                    ),
                )
            except Exception as exc:
                raise ValueError(
                    f"result metrics for scan value {scan_value!r} failed MetricSpec validation"
                ) from exc
            if result.metric_contract:
                contracts[str(scan_value)] = _with_projection_policy(result.metric_contract)
                continue
            contracts[str(scan_value)] = validation.to_dict()
        return contracts

    def metric_semantics(self) -> Dict[str, List[Dict[str, str]]]:
        """Semantic metadata for legacy flat metric keys in this result."""
        try:
            from matrix_factorization.core.contracts import get_algorithm_metric_semantics
        except ImportError:
            return {}
        algorithm_key = getattr(self.config, "algorithm_key", None)
        return get_algorithm_metric_semantics(algorithm_key) if algorithm_key else {}

    def metric_schema(self) -> Dict[str, Any]:
        """Structured semantic schema for this run's flat metric payload."""
        try:
            from matrix_factorization.core.contracts import get_metric_schema
        except ImportError:
            return {}
        algorithm_key = getattr(self.config, "algorithm_key", None)
        if not algorithm_key:
            return {}
        metric_keys = sorted(self._available_metric_keys())
        return get_metric_schema(algorithm_key, metric_keys=metric_keys)

    def factor_payload_contract(self) -> Dict[str, Any]:
        """Describe which factor payloads are real and serialized.

        Tensor metrics-only algorithms can legitimately have no W/X payload.
        This metadata keeps downstream tools from treating missing factors as
        an accidental empty tensor.
        """
        has_w = any(result.W_students is not None for result in self.results.values())
        has_x = any(result.X_students is not None for result in self.results.values())
        has_w_teacher = self.W_teacher is not None
        has_x_teacher = self.X_teacher is not None
        missing = []
        if not has_w:
            missing.append("W_students")
        if not has_x:
            missing.append("X_students")
        if not has_w_teacher:
            missing.append("W_teacher")
        if not has_x_teacher:
            missing.append("X_teacher")
        algorithm_key = getattr(self.config, "algorithm_key", None)
        return {
            "algorithm_key": algorithm_key,
            "matrix_factors": {
                "W_students": has_w,
                "X_students": has_x,
                "W_teacher": has_w_teacher,
                "X_teacher": has_x_teacher,
            },
            "tensor_factors": {
                "serialized": False,
                "reason": "not_exposed_by_current_result_schema",
            },
            "unavailable_fields": missing,
        }

    def to_artifact_manifest(self, run_dir: Path) -> Dict[str, Any]:
        """Build the web-friendly manifest for a saved run directory."""
        run_dir = Path(run_dir)
        artifacts = []
        for relative_path in [
            "config.json",
            "metadata.json",
            "metrics.json",
            "output_contract.json",
            "events.jsonl",
            "artifacts/results.pt",
            "artifacts/teacher_factors.pt",
            "artifacts/raw_data.npz",
            "checkpoints/latest.pt",
        ]:
            artifact_path = run_dir / relative_path
            if artifact_path.exists() or artifact_path.is_symlink():
                artifacts.append({
                    "path": relative_path,
                    "kind": artifact_path.suffix.lstrip(".") or "file",
                })

        plots_dir = run_dir / "plots"
        if plots_dir.exists():
            for plot_path in sorted(plots_dir.iterdir()):
                if plot_path.is_file():
                    artifacts.append({
                        "path": str(plot_path.relative_to(run_dir)),
                        "kind": plot_path.suffix.lstrip(".") or "plot",
                    })

        config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else {}
        return {
            "schema_version": 1,
            "run_id": self.experiment_id,
            "status": "complete" if self.is_complete else "partial",
            "created_at": self.metadata.timestamp,
            "updated_at": datetime.now().isoformat(),
            "scan": {
                "dimension": self.scan_dimension,
                "values": [str(v) for v in self.scan_values],
                "completed": self.num_completed,
                "total": len(self.scan_values),
            },
            "config_summary": {
                "algorithm_key": config_dict.get("algorithm_key"),
                "teacher_key": config_dict.get("teacher_key"),
                "matrix": config_dict.get("matrix"),
                "training": config_dict.get("training"),
            },
            "artifacts": artifacts,
        }

    def _should_disable_unavailable_heatmap(self, output_options: Dict[str, Any]) -> bool:
        if not output_options.get("enable_heatmap", True):
            return False
        has_overlap_matrix = any(
            isinstance(result.metrics, dict) and "overlap_matrix" in result.metrics
            for result in self.results.values()
        )
        has_matrix_factors = any(result.W_students is not None for result in self.results.values())
        if has_overlap_matrix or has_matrix_factors:
            return False
        return not self.result_cube.is_empty() and not self.result_cube.is_alpha_only()

    @classmethod
    def _write_json(cls, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=cls._json_default)

    def _teacher_factor_payload(self) -> Dict[str, Any]:
        """Build a shared teacher-factor artifact for post-hoc diagnostics."""
        payload: Dict[str, Any] = {}
        if self.W_teacher is not None:
            payload["W_teacher"] = self.W_teacher.detach().cpu().to(torch.float32)
        if self.X_teacher is not None:
            payload["X_teacher"] = self.X_teacher.detach().cpu().to(torch.float32)
        if payload:
            payload["payload_kind"] = "matrix_teacher_factors"
            payload["schema_version"] = 1
        return payload

    def save(self, path: Union[str, Path], save_tensors: bool = True, rsb_ordering: bool = False, uniform_colormap: bool = False, output_options: Optional[Dict[str, Any]] = None):
        """
        Save result to directory.

        Args:
            path: Output directory
            save_tensors: Whether to save raw tensors to results.pt
            rsb_ordering: Whether to use hierarchical clustering for RSB heatmap ordering
            uniform_colormap: If True, use linear colormap; if False, enhance 0.9-1.0 range
            output_options: Full output configuration dict with:
                - enable_heatmap: bool - Heatmap + GIF generation
                - plots: list - Custom curves config [{curves: [A.y, B.w]}, ...]
                - storage_mode: str - full/lightweight/plotting_only

        Creates:
        - config.json: Full experiment configuration
        - metadata.json: Run metadata
        - results.pt: Single unified tensor file (all alphas combined)
        - plots/: Evolution plots and heatmaps
        """
        import copy
        import matplotlib.pyplot as plt

        path = Path(path)
        output_options = copy.deepcopy(output_options or {})
        output_warnings: List[str] = []
        if self._should_disable_unavailable_heatmap(output_options):
            output_options["enable_heatmap"] = False
            output_options["_auto_disabled_unavailable_heatmap"] = True
            output_warnings.append(
                "enable_heatmap was disabled for this lightweight canonical "
                "ResultCube because matrix_factors/overlap_matrix artifacts "
                "are not available. Use output.group_results.enable_heatmap "
                "or save_tensors/full storage when factor heatmaps are needed."
            )
        if output_warnings:
            self.metadata.contract.setdefault("output_warnings", []).extend(output_warnings)
        path.mkdir(parents=True, exist_ok=True)
        plots_dir = path / 'plots'
        artifacts_dir = path / 'artifacts'
        checkpoints_dir = path / 'checkpoints'
        plots_dir.mkdir(exist_ok=True)
        artifacts_dir.mkdir(exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)

        # Save config
        if hasattr(self.config, 'save'):
            self.config.save(path / 'config.json')
        else:
            self._write_json(path / 'config.json', self.config)

        # Save metadata
        self.metadata.duration_total_seconds = sum(
            r.duration_seconds for r in self.results.values()
        )
        self._write_json(path / 'metadata.json', self.metadata.to_dict())

        # Canonical local artifacts for display and post-hoc analysis.
        metrics_dict = self.to_metrics_dict()
        sorted_items = self._sorted_result_items()
        sorted_values = [scan_value for scan_value, _ in sorted_items]
        self._write_json(path / 'metrics.json', {
            "schema_version": 5,
            "metric_definition_profile": "projection_qy_physical_latent_v2",
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
            "contract": self.metadata.contract,
            "factor_payload_contract": self.factor_payload_contract(),
            "scan_dimension": self.scan_dimension,
            "scan_values": [str(v) for v in self.scan_values],
            "result_cube": self.result_cube.to_dict(),
            "available_metric_keys": sorted(self._available_metric_keys()),
            "metric_schema": self.metric_schema(),
            "metric_semantics": self.metric_semantics(),
            "metric_contracts": self.metric_contracts(),
            "metrics": metrics_dict,
            "results": self.to_results_dict(),
            "generated_at": datetime.now().isoformat(),
        })
        with open(path / "events.jsonl", "a") as f:
            f.write(json.dumps({
                "type": "run_saved",
                "timestamp": datetime.now().isoformat(),
                "status": "complete" if self.is_complete else "partial",
                "completed": self.num_completed,
                "total": len(self.scan_values),
            }) + "\n")

        teacher_factors_relative_path: Optional[str] = None
        if save_tensors:
            teacher_payload = self._teacher_factor_payload()
            if teacher_payload:
                teacher_factors_relative_path = str(Path("artifacts") / "teacher_factors.pt")
                torch.save(teacher_payload, path / teacher_factors_relative_path)
                if not self.result_cube.is_empty():
                    self.result_cube.artifacts.setdefault("_shared", {})["teacher_factors"] = (
                        teacher_factors_relative_path
                    )

        # Collect all tensors into unified structure
        if save_tensors and not self.result_cube.is_empty() and not self.result_cube.is_alpha_only():
            points_dir = artifacts_dir / "points"
            for scan_value, r in sorted_items:
                point_dir = points_dir / str(scan_value)
                point_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "metrics": r.metrics,
                    "factor_payload_contract": self.factor_payload_contract(),
                }
                if teacher_factors_relative_path is not None:
                    payload["teacher_factors_path"] = teacher_factors_relative_path
                if r.W_students is not None:
                    payload["W_students"] = r.W_students.cpu().to(torch.float16)
                if r.X_students is not None:
                    payload["X_students"] = r.X_students.cpu().to(torch.float16)
                torch.save(payload, point_dir / "results.pt")
                if str(scan_value) in self.result_cube.points:
                    self.result_cube.artifacts.setdefault(str(scan_value), {})["point_results"] = str(
                        Path("artifacts") / "points" / str(scan_value) / "results.pt"
                    )
        elif save_tensors:
            # Stack all W_students and X_students across scan values
            all_W = []
            all_X = []
            for _, r in sorted_items:
                if r.W_students is not None:
                    all_W.append(r.W_students.cpu().to(torch.float16))
                if r.X_students is not None:
                    all_X.append(r.X_students.cpu().to(torch.float16))

            # Create unified results.pt
            results_data = {
                'alpha_values': [float(v) for v in sorted_values],
                'metrics': metrics_dict,
                'factor_payload_contract': self.factor_payload_contract(),
            }
            if teacher_factors_relative_path is not None:
                results_data['teacher_factors_path'] = teacher_factors_relative_path

            if all_W:
                # Stack: (num_alphas, S, N1, M)
                results_data['W_students'] = torch.stack(all_W, dim=0)
            if all_X:
                # Stack: (num_alphas, S, M, N2)
                results_data['X_students'] = torch.stack(all_X, dim=0)

            # Backward-compatible inline teacher copies for older local scripts.
            # New post-hoc tools should prefer artifacts/teacher_factors.pt.
            if self.W_teacher is not None:
                results_data['W_teacher'] = self.W_teacher.detach().cpu().to(torch.float16)
            if self.X_teacher is not None:
                results_data['X_teacher'] = self.X_teacher.detach().cpu().to(torch.float16)

            tensor_path = artifacts_dir / 'results.pt'
            torch.save(results_data, tensor_path)

            # Backward-compatible pointer for older local analysis scripts.
            legacy_tensor_path = path / 'results.pt'
            try:
                if legacy_tensor_path.exists() or legacy_tensor_path.is_symlink():
                    legacy_tensor_path.unlink()
                legacy_tensor_path.symlink_to(Path('artifacts') / 'results.pt')
            except OSError:
                pass

        if self.result_cube.artifacts:
            self._write_json(path / 'metrics.json', {
                "schema_version": 5,
                "metric_definition_profile": "projection_qy_physical_latent_v2",
                "experiment_id": self.experiment_id,
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                "contract": self.metadata.contract,
                "factor_payload_contract": self.factor_payload_contract(),
                "scan_dimension": self.scan_dimension,
                "scan_values": [str(v) for v in self.scan_values],
                "result_cube": self.result_cube.to_dict(),
                "available_metric_keys": sorted(self._available_metric_keys()),
                "metric_schema": self.metric_schema(),
                "metric_semantics": self.metric_semantics(),
                "metric_contracts": self.metric_contracts(),
                "metrics": metrics_dict,
                "results": self.to_results_dict(),
                "generated_at": datetime.now().isoformat(),
            })

        self._write_precision_comparison_report(path)

        # Generate evolution plots (only if user does NOT have custom plots configured)
        sorted_values = [scan_value for scan_value, _ in sorted_items]

        # Determine x-axis label based on scan dimension
        x_label = 'Alpha' if self.scan_dimension == 'alpha' else 'Steps'

        # Check if user has custom plots configured - if so, skip default plots
        has_custom_plots = output_options and output_options.get('plots')
        output_contract_check = self._validate_output_contracts_before_plots(output_options or {}, path)
        self._write_json(path / 'output_contract.json', output_contract_check.to_dict())

        has_plot_queries = bool(has_custom_plots and any("x" in plot and "y" in plot for plot in output_options.get("plots", [])))

        if not has_custom_plots and (self.result_cube.is_empty() or self.result_cube.is_alpha_only()):
            # Extract metrics for plotting
            x_values = [float(v) for v in sorted_values]
            metric_keys = self._available_metric_keys()
            q_y_key = self._first_available_metric(
                metric_keys,
                ["Q_Y_mean", "Q_Y_observed_mean", "Q_Y_unobserved_mean"],
            )
            if q_y_key is None:
                raise ValueError(
                    "scalar_curves output requires one of "
                    "['Q_Y_mean', 'Q_Y_observed_mean', 'Q_Y_unobserved_mean']; "
                    f"available metrics: {sorted(metric_keys)}"
                )
            q_w_key = self._first_available_metric(metric_keys, ["Q_W_mean", "Q_W_COS_ROOT_mean", "Q_N_mean"])
            q_y_means = self._metric_series(sorted_values, q_y_key)
            q_w_means = self._metric_series(sorted_values, q_w_key) if q_w_key else []

            # Plot Q_W and Q_Y evolution
            fig, ax = plt.subplots(figsize=(10, 6))
            if q_w_key:
                ax.plot(
                    x_values, q_w_means, 'r-',
                    label=self._plot_label_for_metric(q_w_key),
                    linewidth=2, marker='o', markersize=3,
                )
            ax.plot(
                x_values, q_y_means, 'g-',
                label=self._plot_label_for_metric(q_y_key),
                linewidth=2, marker='s', markersize=3,
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel('Metric value')
            ax.set_title(f'{self.experiment_id}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-0.1, 1.1)
            plt.savefig(plots_dir / 'qy_evolution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

            if q_w_key:
                # Plot Q_W only (convergence curve for steps scan)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x_values, q_w_means, 'r-', linewidth=2, marker='o', markersize=4)
                ax.set_xlabel(x_label)
                ax.set_ylabel(self._plot_label_for_metric(q_w_key))
                ax.set_title(f'{self._plot_label_for_metric(q_w_key)} Evolution - {self.experiment_id}')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.1, 1.1)
                plt.savefig(plots_dir / 'overlap_evolution.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

        # ===== 自定义曲线绘图 (来自 output_options['plots']) =====
        if has_plot_queries:
            self._plot_result_cube_queries(output_options.get("plots") or [], plots_dir)
        elif output_options and output_options.get('plots'):
            if not self._allows_legacy_alpha_curve_plots():
                with open(path / "events.jsonl", "a") as f:
                    f.write(json.dumps({
                        "type": "legacy_curve_plots_skipped",
                        "timestamp": datetime.now().isoformat(),
                        "reason": (
                            "legacy curves plots require alpha-only results; "
                            "use PlotQuery with x/y/where/series_by/compare "
                            "for final multi-axis canonical scan plots"
                        ),
                    }) + "\n")
                self.metadata.contract.setdefault("output_warnings", []).append(
                    "Legacy output.plots curves were skipped for final multi-axis "
                    "ResultCube. Use PlotQuery for cross-scan comparison plots."
                )
            else:
                from matrix_factorization.modules.outputs.plotting import plot_custom_curves

                # 构建 results 数据格式供 plot_custom_curves 使用
                plot_results = {
                    float(v): r.metrics for v, r in self.results.items()
                }

                for i, plot_config in enumerate(output_options['plots']):
                    curves = plot_config.get('curves', [])
                    if curves:
                        output_file = plots_dir / f'custom_plot_{i+1}.png'
                        plot_custom_curves(
                            plot_results,
                            curves,
                            output_file,
                            title=f"{self.experiment_id} - Plot {i+1}",
                        )
                        print(f"Generated: {output_file}")

        # ===== Heatmap 和 GIF (由 enable_heatmap 控制) =====
        enable_heatmap = True  # 默认开启
        if output_options:
            enable_heatmap = output_options.get('enable_heatmap', True)

        if enable_heatmap:
            try:
                from matrix_factorization.modules.outputs.plotting import plot_replica_heatmap, create_gif
                from matrix_factorization.modules.metrics.overlap import build_interaction_matrix, gram_overlap_normalized
                import numpy as np

                heatmap_paths = []
                heatmap_metric_codes = []
                W_teacher = self.W_teacher

                for v in sorted_values:
                    r = self.results[v]
                    heatmap_alpha = self._heatmap_alpha_for_scan_value(v)
                    matrix = r.metrics.get('overlap_matrix') if r.metrics else None
                    if matrix is not None:
                        metric_code = r.metrics.get(
                            'overlap_matrix_metric',
                            output_options.get('heatmap_metric', 'Q_Y') if output_options else 'Q_Y',
                        )
                        metric_code = str(metric_code).upper()
                        if metric_code == "Q_W":
                            metric_name = "Factor Metric ($Q_W$)"
                            filename_prefix = self._heatmap_filename_prefix("heatmap_W", v)
                        else:
                            metric_code = "Q_Y"
                            metric_name = "Tensor Fit ($Q_Y$)"
                            filename_prefix = self._heatmap_filename_prefix("heatmap_Y", v)

                        heatmap_path = plot_replica_heatmap(
                            np.asarray(matrix, dtype=float), heatmap_alpha, plots_dir,
                            metric_name=metric_name, filename_prefix=filename_prefix,
                            rsb_ordering=rsb_ordering,
                            enhance_high_values=not uniform_colormap,
                        )
                        if heatmap_path:
                            heatmap_paths.append(heatmap_path)
                            heatmap_metric_codes.append(metric_code)
                        continue

                    if r.W_students is not None and W_teacher is not None:
                        # Handle shape: W_students might be (1, S, N1, M) or (S, N1, M)
                        W_s = r.W_students
                        while W_s.dim() > 3:
                            W_s = W_s.squeeze(0)  # Remove leading dims until (S, N1, M)

                        heatmap_metric = str(
                            output_options.get('heatmap_metric', 'Q_W') if output_options else 'Q_W'
                        ).upper()
                        if heatmap_metric in {"Q_W_SIGN", "Q_W_SIGN_ALIGNED", "Q_W_SIGN_GAUGE", "D.W"}:
                            from matrix_factorization.modules.metrics.overlap import sign_gauge_overlap_fixed

                            def _w_sign_gauge(a, b):
                                return sign_gauge_overlap_fixed(a, b, latent_axis=-1)

                            matrix_W = build_interaction_matrix(
                                W_s, W_teacher, _w_sign_gauge, use_left=True
                            )
                            metric_name = "W Sign-Gauge Overlap ($Q_{W,sign}$)"
                            filename_prefix = self._heatmap_filename_prefix("heatmap_W_sign_gauge", v)
                            heatmap_code = "Q_W_SIGN_GAUGE"
                        else:
                            # Build interaction matrix
                            matrix_W = build_interaction_matrix(
                                W_s, W_teacher, gram_overlap_normalized, use_left=True
                            )
                            metric_name = "Q_W"
                            filename_prefix = self._heatmap_filename_prefix("heatmap_W", v)
                            heatmap_code = "Q_W"

                        # Save heatmap
                        heatmap_path = plot_replica_heatmap(
                            matrix_W, heatmap_alpha, plots_dir,
                            metric_name=metric_name, filename_prefix=filename_prefix,
                            rsb_ordering=rsb_ordering,
                            enhance_high_values=not uniform_colormap,  # uniform = no enhancement
                        )
                        if heatmap_path:
                            heatmap_paths.append(heatmap_path)
                            heatmap_metric_codes.append(heatmap_code)

                # Create GIF from heatmaps
                if heatmap_paths:
                    gif_suffix = heatmap_metric_codes[-1][-1] if heatmap_metric_codes else "W"
                    gif_path = create_gif(heatmap_paths, plots_dir / f"animation_{gif_suffix}.gif", duration=0.2)
                    if gif_path:
                        print(f"Generated GIF: {gif_path}")
            except Exception as e:
                raise RuntimeError(
                    "Heatmap/GIF output failed after OutputSpec preflight passed. "
                    "Treat this as an output integration error rather than a "
                    "non-fatal plotting warning."
                ) from e

        self._write_json(path / 'manifest.json', self.to_artifact_manifest(path))

    def save_partial_snapshot(
        self,
        path: Union[str, Path],
        *,
        output_options: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a lightweight canonical-scan snapshot during a long run.

        This intentionally avoids plots, heatmaps, and tensor payloads.  It is
        a recovery/inspection surface for long local scans where the final
        ExperimentResult is only saved after every scan point completes.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        partial_dir = path / "partial"
        partial_dir.mkdir(exist_ok=True)

        if hasattr(self.config, "save") and not (path / "config.json").exists():
            self.config.save(path / "config.json")

        self.metadata.duration_total_seconds = sum(
            r.duration_seconds for r in self.results.values()
        )
        metadata = self.metadata.to_dict()
        metadata["partial_snapshot"] = True
        metadata["partial_extra"] = dict(extra or {})
        self._write_json(partial_dir / "metadata_partial.json", metadata)

        completed_values = [str(value) for value, _ in self._sorted_result_items()]
        payload = {
            "schema_version": 5,
            "metric_definition_profile": "projection_qy_physical_latent_v2",
            "partial_snapshot": True,
            "snapshot_mode": "compact_progress",
            "experiment_id": self.experiment_id,
            "status": "complete" if self.is_complete else "partial",
            "completed": self.num_completed,
            "total": len(self.scan_values),
            "completion_ratio": self.completion_ratio,
            "scan_dimension": self.scan_dimension,
            "completed_scan_values": completed_values,
            "available_metric_keys": sorted(self._available_metric_keys()),
            "metric_schema": self.metric_schema(),
            "output_options": dict(output_options or {}),
            "extra": dict(extra or {}),
            "generated_at": datetime.now().isoformat(),
        }
        self._write_json(partial_dir / "metrics_partial.json", payload)
        self._write_json(path / "metrics.partial.json", payload)

        manifest = self.to_artifact_manifest(path)
        manifest["status"] = "complete" if self.is_complete else "partial"
        manifest["partial_snapshot"] = True
        manifest["partial"] = {
            "metrics": "partial/metrics_partial.json",
            "metadata": "partial/metadata_partial.json",
            "completed": self.num_completed,
            "total": len(self.scan_values),
            "extra": dict(extra or {}),
        }
        self._write_json(partial_dir / "manifest_partial.json", manifest)

        with open(path / "events.jsonl", "a") as f:
            f.write(json.dumps({
                "type": "partial_snapshot_saved",
                "timestamp": datetime.now().isoformat(),
                "completed": self.num_completed,
                "total": len(self.scan_values),
                "extra": dict(extra or {}),
            }, default=self._json_default) + "\n")

    def _plot_result_cube_queries(self, plot_queries: List[Dict[str, Any]], plots_dir: Path) -> None:
        """Render PlotQuery-style plots from ResultCube."""
        import matplotlib.pyplot as plt
        from matrix_factorization.modules.outputs.publication_style import (
            ERROR_CONFIG,
            PUB_CONFIG,
            StyleCycler,
            apply_publication_style,
        )

        if self.result_cube.is_empty():
            raise ValueError("PlotQuery requires result_cube data")
        apply_publication_style()
        for idx, query in enumerate(plot_queries):
            if "x" not in query or "y" not in query:
                continue
            x_axis = query["x"]
            metric_key = query["y"]
            filename = query.get("filename") or f"plot_query_{idx + 1}.png"
            resolved_series = self.result_cube.resolve_plot_query(query)

            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            style = StyleCycler(len(resolved_series), palette="colorblind")
            for series_idx, series in enumerate(resolved_series):
                curve_style = style.get_style(series_idx)
                yerr = series.get("y_std_values") or []
                yerr = yerr if any(float(value) > 0 for value in yerr) else None
                ax.errorbar(
                    series["x_values"],
                    series["y_values"],
                    yerr=yerr,
                    marker=curve_style.get("marker", "o"),
                    linestyle=curve_style.get("linestyle", "-"),
                    color=curve_style.get("color"),
                    linewidth=PUB_CONFIG.linewidth_plot,
                    markersize=PUB_CONFIG.markersize,
                    capsize=ERROR_CONFIG.capsize if yerr is not None else 0,
                    elinewidth=max(0.8, PUB_CONFIG.linewidth_plot * 0.65),
                    alpha=0.95,
                    label=series["label"],
                )
            ax.set_xlabel(str(x_axis))
            ax.set_ylabel(self._plot_label_for_metric(str(metric_key)))
            ax.grid(True, alpha=PUB_CONFIG.grid_alpha)
            ax.legend(frameon=False, fontsize=PUB_CONFIG.font_size_legend)
            ax.set_title(query.get("title") or f"{metric_key} vs {x_axis}")
            fig.tight_layout()
            fig.savefig(plots_dir / filename, dpi=PUB_CONFIG.dpi, bbox_inches="tight")
            plt.close(fig)

    def _write_precision_comparison_report(self, run_dir: Path) -> None:
        """Write a lightweight fast/aggressive comparison for precision scan runs."""
        if self.result_cube.is_empty():
            return
        precision_axis = None
        for axis_key, axis_payload in self.result_cube.axes.items():
            if axis_key == "precision" or (
                isinstance(axis_payload, dict)
                and axis_payload.get("path") == "algorithm_params.precision_profile"
            ):
                precision_axis = axis_key
                break
        if not precision_axis:
            return

        profiles = {
            str(point.coordinates.get(precision_axis))
            for point in self.result_cube.points.values()
            if precision_axis in point.coordinates
        }
        if not ({"fast", "aggressive"} <= profiles):
            return

        grouped: Dict[str, Dict[str, str]] = {}
        group_labels: Dict[str, Dict[str, Any]] = {}
        for point_id, point in self.result_cube.points.items():
            profile = str(point.coordinates.get(precision_axis))
            if profile not in {"fast", "aggressive"}:
                continue
            coords_without_precision = {
                key: value
                for key, value in point.coordinates.items()
                if key != precision_axis
            }
            group_key = json.dumps(coords_without_precision, sort_keys=True, default=str)
            grouped.setdefault(group_key, {})[profile] = point_id
            group_labels[group_key] = coords_without_precision

        rows = []
        nan_inf_rows = []
        for group_key, point_ids in grouped.items():
            fast_id = point_ids.get("fast")
            aggressive_id = point_ids.get("aggressive")
            if not fast_id or not aggressive_id:
                continue
            fast_metrics = self.result_cube.metrics.get(fast_id, {})
            aggressive_metrics = self.result_cube.metrics.get(aggressive_id, {})
            for metric_key in sorted(set(fast_metrics) & set(aggressive_metrics)):
                if not metric_key.startswith("Q_"):
                    continue
                try:
                    fast_value = float(fast_metrics[metric_key])
                    aggressive_value = float(aggressive_metrics[metric_key])
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(fast_value) and math.isfinite(aggressive_value)):
                    nan_inf_rows.append((group_labels[group_key], metric_key, fast_value, aggressive_value))
                    continue
                diff = aggressive_value - fast_value
                denom = max(abs(fast_value), 1e-12)
                rows.append({
                    "coords": group_labels[group_key],
                    "metric": metric_key,
                    "fast": fast_value,
                    "aggressive": aggressive_value,
                    "abs_diff": abs(diff),
                    "rel_diff": abs(diff) / denom,
                })

        rows.sort(key=lambda item: item["abs_diff"], reverse=True)
        runtime_plan = {}
        if isinstance(self.metadata.contract, dict):
            runtime_plan = self.metadata.contract.get("runtime_resource_plan") or {}

        lines = [
            "# Precision comparison",
            "",
            "This report compares `fast` and `aggressive` points in the same ResultCube coordinate group.",
            "",
            f"- precision axis: `{precision_axis}`",
            f"- compared groups: {sum(1 for ids in grouped.values() if {'fast', 'aggressive'} <= set(ids))}",
            f"- runtime groups: {runtime_plan.get('num_groups', 'unknown')}",
            f"- runtime batches: {runtime_plan.get('num_batches', 'unknown')}",
            "",
            "## Largest metric differences",
            "",
            "| coords | metric | fast | aggressive | abs diff | rel diff |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows[:50]:
            coord_label = ", ".join(f"{k}={v}" for k, v in sorted(row["coords"].items())) or "all"
            lines.append(
                f"| {coord_label} | {row['metric']} | {row['fast']:.6g} | "
                f"{row['aggressive']:.6g} | {row['abs_diff']:.6g} | {row['rel_diff']:.3%} |"
            )
        if not rows:
            lines.append("| no comparable finite Q metrics | - | - | - | - | - |")

        lines.extend(["", "## NaN / Inf checks", ""])
        if nan_inf_rows:
            for coords, metric_key, fast_value, aggressive_value in nan_inf_rows:
                coord_label = ", ".join(f"{k}={v}" for k, v in sorted(coords.items())) or "all"
                lines.append(f"- {coord_label}: `{metric_key}` fast={fast_value}, aggressive={aggressive_value}")
        else:
            lines.append("- No NaN/Inf values found in comparable Q metrics.")

        (run_dir / "precision_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _select_cube_points(self, where: Dict[str, Any]) -> List[ResultCubePoint]:
        points = list(self.result_cube.points.values())
        for axis, value in where.items():
            if axis not in self.result_cube.axes:
                raise ValueError(f"PlotQuery where references unknown axis '{axis}'")
            points = [point for point in points if point.coordinates.get(axis) == value]
        if not points:
            raise ValueError(f"PlotQuery where selected no points: {where}")
        return points

    def _validate_output_contracts_before_plots(
        self,
        output_options: Dict[str, Any],
        run_dir: Optional[Path] = None,
    ):
        """Check output dependencies against concrete result payloads."""
        from matrix_factorization.modules.outputs.spec_adapter import OutputSpecAdapter

        return OutputSpecAdapter.validate_result(self, output_options, run_dir)

    def _available_metric_keys(self) -> set[str]:
        return {
            key
            for result in self.results.values()
            for key in (result.metrics or {})
        }

    @staticmethod
    def _first_available_metric(metric_keys: set[str], candidates: List[str]) -> Optional[str]:
        for key in candidates:
            for candidate in metric_key_candidates(key):
                if candidate in metric_keys:
                    return candidate
        return None

    def _heatmap_alpha_for_scan_value(self, scan_value: Any) -> float:
        """Return the alpha coordinate for heatmap labels in alpha or canonical scans."""
        try:
            return float(scan_value)
        except (TypeError, ValueError):
            pass
        point = self.result_cube.points.get(str(scan_value))
        if point is not None and "alpha" in point.coordinates:
            try:
                return float(point.coordinates["alpha"])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _heatmap_filename_prefix(self, base: str, scan_value: Any) -> str:
        """Make heatmap filenames unique when scan values are ResultCube point ids."""
        if self.result_cube.is_empty() or self.result_cube.is_alpha_only():
            return base
        point_id = self._slugify_path_component(str(scan_value))
        point = self.result_cube.points.get(str(scan_value))
        if point is None:
            return f"{base}_{point_id}"
        group_id = self._slugify_path_component(str(point.group_id or "group"))
        return f"{base}_{group_id}_{point_id}"

    def _metric_series(self, sorted_values: List[Any], metric_key: str) -> List[float]:
        candidates = metric_key_candidates(metric_key)
        missing_values = [
            value for value in sorted_values
            if not any(candidate in (self.results[value].metrics or {}) for candidate in candidates)
        ]
        if missing_values:
            raise ValueError(
                f"scalar_curves output requires metric '{metric_key}' for all scan values; "
                f"missing values: {missing_values}"
            )
        values = []
        for value in sorted_values:
            metrics = self.results[value].metrics or {}
            actual_key = next(candidate for candidate in candidates if candidate in metrics)
            values.append(float(metrics[actual_key]))
        return values

    def _allows_legacy_alpha_curve_plots(self) -> bool:
        """Allow legacy curve plots for alpha curves with constant outer axes."""
        if self.result_cube.is_empty() or self.result_cube.is_alpha_only():
            return True
        if not self.results:
            return False
        try:
            [float(value) for value in self.results.keys()]
        except (TypeError, ValueError):
            return False
        non_alpha_values: Dict[str, set[str]] = {}
        for point in self.result_cube.points.values():
            for axis, value in (point.coordinates or {}).items():
                if axis == "alpha":
                    continue
                non_alpha_values.setdefault(axis, set()).add(str(value))
        return all(len(values) <= 1 for values in non_alpha_values.values())

    @staticmethod
    def _slugify_path_component(value: str, max_length: int = 80) -> str:
        text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
        text = "_".join(part for part in text.split("_") if part)
        return (text[:max_length] or "value").strip("._-") or "value"

    def _plot_label_for_metric(self, metric_key: str) -> str:
        metric_schema = self.metric_schema()
        flat_index = metric_schema.get("flat_key_index", {}) if isinstance(metric_schema, dict) else {}
        semantic_classes = metric_schema.get("semantic_classes", {}) if isinstance(metric_schema, dict) else {}
        candidates = flat_index.get(metric_key) or []
        if candidates:
            canonical_key = candidates[0].get("canonical_key")
            semantic_class = semantic_classes.get(canonical_key, {})
            display_name = semantic_class.get("display_name")
            if display_name:
                return display_name
        return {
            "Q_Y_mean": "Q_Y",
            "NMSE_Y_mean": "NMSE_Y",
            "Q_Y_observed_mean": "Q_Y observed",
            "Q_Y_unobserved_mean": "Q_Y unobserved",
            "Q_W_mean": "Q_W",
            "Q_X_mean": "Q_X",
            "Q_W_SIGN_GAUGE_mean": "Q_W sign gauge",
            "Q_W_COS_ROOT_mean": "Q_W Cos root",
            "Q_W_GRAM_ROOT_mean": "Q_W Cos root",
            "Q_W_SIGN_ALIGNED_mean": "Q_W sign-aligned",
            "Q_X_COS_ROOT_mean": "Q_X Cos root",
            "Q_X_GRAM_ROOT_mean": "Q_X Cos root",
            "Q_X_SIGN_GAUGE_mean": "Q_X sign gauge",
            "Q_X_SIGN_ALIGNED_mean": "Q_X sign-aligned",
            "Q_W_SCALE_GAUGE_mean": "Q_W scale gauge",
            "Q_X_SCALE_GAUGE_mean": "Q_X scale gauge",
            "Q_WX_SCALE_GAUGE_mean": "Q_WX scale gauge",
            "Q_N_mean": "Q_N",
        }.get(metric_key, metric_key)

    @staticmethod
    def _curve_code_to_metric_key(curve_code: str) -> str:
        from matrix_factorization.modules.outputs.plot_registry import parse_curve_code

        spec = parse_curve_code(curve_code)
        metric_name = f"{spec.metric_name}_replica" if spec.is_replica else spec.metric_name
        return f"{metric_name}_mean"

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentResult':
        """Load result from directory."""
        path = Path(path)

        # Load config
        from .config import ExperimentConfig
        config = ExperimentConfig.load(path / 'config.json')

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata_dict = json.load(f)
        metadata = ExperimentMetadata(**metadata_dict)

        # Load results. Prefer the Codex-web run schema; keep legacy fallback.
        metrics_path = path / 'metrics.json'
        legacy_results_path = path / 'results.json'
        metrics_payload = {}
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics_payload = json.load(f)
            results_data = {
                'experiment_id': metrics_payload.get('experiment_id', path.name),
                'scan_dimension': metrics_payload.get('scan_dimension', config.scan.dimension),
                'scan_values': metrics_payload.get('scan_values', []),
                'results': metrics_payload.get('results') or {
                    scan_value: {'metrics': metrics}
                    for scan_value, metrics in metrics_payload.get('metrics', {}).items()
                },
            }
        else:
            with open(legacy_results_path, 'r') as f:
                results_data = json.load(f)

        # Parse scan values back to original types
        scan_values = []
        for v_str in results_data['scan_values']:
            try:
                scan_values.append(float(v_str))
            except ValueError:
                scan_values.append(v_str)

        result = cls(
            experiment_id=results_data['experiment_id'],
            config=config,
            scan_dimension=results_data['scan_dimension'],
            scan_values=scan_values,
            metadata=metadata,
        )
        loaded_schema_version = int(metrics_payload.get("schema_version", 0) or 0) if metrics_payload else 0
        if loaded_schema_version < 3:
            result.metadata.contract.setdefault("metric_schema_compatibility", {
                "loaded_schema_version": loaded_schema_version,
                "q_y_mean_interpretation": "legacy_cosine_or_reconstruction_proxy",
                "new_schema_q_y_mean_interpretation": "absolute_projection",
                "new_old_q_y_mean_not_comparable": True,
            })
        elif loaded_schema_version < 4:
            result.metadata.contract.setdefault("metric_schema_compatibility", {
                "loaded_schema_version": loaded_schema_version,
                "q_y_mean_interpretation": "absolute_projection",
                "new_schema_q_y_mean_interpretation": "absolute_projection",
                "schema_v3_v4_q_y_mean_not_comparable": True,
            })
        elif loaded_schema_version < 5:
            result.metadata.contract.setdefault("metric_schema_compatibility", {
                "loaded_schema_version": loaded_schema_version,
                "q_y_mean_interpretation": "fit_1_minus_nmse",
                "new_schema_q_y_mean_interpretation": "absolute_projection",
                "schema_v4_v5_q_y_mean_not_comparable": True,
                "metric_definition_profile": metrics_payload.get("metric_definition_profile", "projection_qy_physical_latent_v2"),
            })
        else:
            result.metadata.contract.setdefault("metric_schema_compatibility", {
                "loaded_schema_version": loaded_schema_version,
                "q_y_mean_interpretation": "absolute_projection",
                "metric_definition_profile": metrics_payload.get("metric_definition_profile", "projection_qy_physical_latent_v2"),
            })
        cube_payload = metrics_payload.get("result_cube", {}) if isinstance(metrics_payload, dict) else {}
        if cube_payload:
            result.result_cube = ResultCube(
                axes=dict(cube_payload.get("axes", {}) or {}),
                metrics=dict(cube_payload.get("metrics", {}) or {}),
                artifacts=dict(cube_payload.get("artifacts", {}) or {}),
                groups=dict(cube_payload.get("groups", {}) or {}),
                metric_semantics=dict(cube_payload.get("metric_semantics", {}) or {}),
            )
            for point_id, point_payload in (cube_payload.get("points", {}) or {}).items():
                result.result_cube.points[point_id] = ResultCubePoint(
                    point_id=point_payload.get("point_id", point_id),
                    coordinates=dict(point_payload.get("coordinates", {}) or {}),
                    overrides=dict(point_payload.get("overrides", {}) or {}),
                    effective_config_hash=str(point_payload.get("effective_config_hash", "")),
                    group_id=str(point_payload.get("group_id", "default")),
                    metric_contract=dict(point_payload.get("metric_contract", {}) or {}),
                )

        tensor_path = path / 'artifacts' / 'results.pt'
        if not tensor_path.exists():
            tensor_path = path / 'results.pt'
        tensor_data = torch.load(tensor_path, map_location='cpu') if tensor_path.exists() else {}
        w_stack = tensor_data.get('W_students') if isinstance(tensor_data, dict) else None
        x_stack = tensor_data.get('X_students') if isinstance(tensor_data, dict) else None
        teacher_path = path / 'artifacts' / 'teacher_factors.pt'
        teacher_data = torch.load(teacher_path, map_location='cpu') if teacher_path.exists() else {}
        if isinstance(teacher_data, dict):
            result.W_teacher = teacher_data.get('W_teacher', result.W_teacher)
            result.X_teacher = teacher_data.get('X_teacher', result.X_teacher)
        if isinstance(tensor_data, dict):
            result.W_teacher = tensor_data.get('W_teacher', result.W_teacher)
            result.X_teacher = tensor_data.get('X_teacher', result.X_teacher)

        # Load individual results
        sorted_result_keys = sorted(results_data['results'].keys(), key=cls._sort_key)
        for v_str, r_dict in results_data['results'].items():
            try:
                v = float(v_str)
            except ValueError:
                v = v_str

            single_result = SingleRunResult(
                scan_value=v,
                metrics=r_dict.get('metrics', {}),
                duration_seconds=r_dict.get('duration_seconds', 0.0),
                history=r_dict.get('history'),
                metric_contract=r_dict.get('metric_contract', {}),
            )

            # Load tensors if they exist in the canonical artifact bundle.
            if w_stack is not None and v_str in sorted_result_keys:
                idx = sorted_result_keys.index(v_str)
                if idx < len(w_stack):
                    single_result.W_students = w_stack[idx]
            if x_stack is not None and v_str in sorted_result_keys:
                idx = sorted_result_keys.index(v_str)
                if idx < len(x_stack):
                    single_result.X_students = x_stack[idx]

            result.results[v] = single_result

        return result

    def save_unified(self, path: Union[str, Path]):
        """
        Save ALL data to a SINGLE .pt file.

        This is the recommended method for production use:
        - Single file, no scattered small files
        - Complete reproducibility (all config/metadata saved)
        - Efficient loading (torch.save/load)

        File structure inside the .pt:
        {
            'experiment_id': str,
            'config': dict,
            'metadata': dict,
            'scan_dimension': str,
            'scan_values': list,
            'raw_data': {
                'W_teacher': tensor,
                'X_teacher': tensor,
                'Y_teacher': tensor,
                'all_masks': tensor or None,
                'supergraph_data': dict or None,
            },
            'results': {
                scan_value: {
                    'metrics': dict,
                    'duration_seconds': float,
                    'history': list or None,
                    'W_students': tensor or None,
                    'X_students': tensor or None,
                    'mask': tensor or None,
                    'observation_indices': dict or None,
                }
            }
        }
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.pt')
        path.parent.mkdir(parents=True, exist_ok=True)

        # Update total duration
        self.metadata.duration_total_seconds = sum(
            r.duration_seconds for r in self.results.values()
        )

        # Build unified data structure
        data = {
            'experiment_id': self.experiment_id,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config,
            'metadata': self.metadata.to_dict(),
            'scan_dimension': self.scan_dimension,
            'scan_values': self.scan_values,
            # RAW DATA - for post-hoc analysis
            'raw_data': {
                'W_teacher': self.W_teacher,
                'X_teacher': self.X_teacher,
                'Y_teacher': self.Y_teacher,
                'all_masks': self.all_masks,
                'supergraph_data': self.supergraph_data,
            },
            'results': {},
        }

        # Add all results (including tensors and raw data)
        for v, r in self.results.items():
            data['results'][v] = {
                'metrics': r.metrics,
                'duration_seconds': r.duration_seconds,
                'history': r.history,
                'W_students': r.W_students,
                'X_students': r.X_students,
                'mask': r.mask,
                'observation_indices': r.observation_indices,
            }

        # Save as single file
        torch.save(data, path)

        if hasattr(self, '_verbose') and self._verbose:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"Saved experiment to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load_unified(cls, path: Union[str, Path], device: torch.device = None) -> 'ExperimentResult':
        """
        Load from a single .pt file.

        Args:
            path: Path to .pt file
            device: Target device for tensors (default: keep original)

        Returns:
            ExperimentResult with all data
        """
        path = Path(path)
        data = torch.load(path, map_location=device, weights_only=False)

        # Reconstruct config
        from .config import ExperimentConfig, MatrixParams, TrainingParams, ScanConfig, SeedConfig, AlgorithmParams, SpreadingConfig

        config_dict = data['config']
        config = ExperimentConfig(
            matrix=MatrixParams(**config_dict['matrix']),
            training=TrainingParams(**config_dict['training']),
            algorithm_key=config_dict['algorithm_key'],
            scan=ScanConfig(**config_dict['scan']),
            seeds=SeedConfig(**config_dict['seeds']),
            algorithm_params=AlgorithmParams(**config_dict['algorithm_params']),
            spreading=SpreadingConfig(**config_dict['spreading']) if config_dict.get('spreading') else None,
            experiment_name=config_dict.get('experiment_name', 'unnamed'),
            teacher_key=config_dict.get('teacher_key', 'standard'),
            notes=config_dict.get('notes', ''),
        )

        # Reconstruct metadata
        metadata = ExperimentMetadata(**data['metadata'])

        # Load raw data
        raw_data = data.get('raw_data', {})

        # Create result object
        result = cls(
            experiment_id=data['experiment_id'],
            config=config,
            scan_dimension=data['scan_dimension'],
            scan_values=data['scan_values'],
            metadata=metadata,
            # Raw data
            W_teacher=raw_data.get('W_teacher'),
            X_teacher=raw_data.get('X_teacher'),
            Y_teacher=raw_data.get('Y_teacher'),
            all_masks=raw_data.get('all_masks'),
            supergraph_data=raw_data.get('supergraph_data'),
        )

        # Load individual results
        for v, r_dict in data['results'].items():
            single_result = SingleRunResult(
                scan_value=v,
                metrics=r_dict.get('metrics', {}),
                duration_seconds=r_dict.get('duration_seconds', 0.0),
                history=r_dict.get('history'),
                W_students=r_dict.get('W_students'),
                X_students=r_dict.get('X_students'),
                mask=r_dict.get('mask'),
                observation_indices=r_dict.get('observation_indices'),
                metric_contract=r_dict.get('metric_contract', {}),
            )
            result.results[v] = single_result

        return result

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(\n"
            f"  id='{self.experiment_id}',\n"
            f"  scan={self.scan_dimension}: {len(self.scan_values)} points,\n"
            f"  completed={self.num_completed}/{len(self.scan_values)},\n"
            f")"
        )
