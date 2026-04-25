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
import torch


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
            d['metric_contract'] = self.metric_contract
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
        return {
            v: r.metrics.get(metric_name, 0.0)
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
            if result.metric_contract:
                contracts[str(scan_value)] = result.metric_contract
                continue
            if not algorithm_key:
                continue
            try:
                from matrix_factorization.modules.metrics.spec_adapter import MetricSpecAdapter
                contracts[str(scan_value)] = MetricSpecAdapter.describe_payload(
                    algorithm_key,
                    result.metrics,
                    source="result_schema_fallback",
                )
            except Exception:
                continue
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
        missing = []
        if not has_w:
            missing.append("W_students")
        if not has_x:
            missing.append("X_students")
        algorithm_key = getattr(self.config, "algorithm_key", None)
        return {
            "algorithm_key": algorithm_key,
            "matrix_factors": {
                "W_students": has_w,
                "X_students": has_x,
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

    @classmethod
    def _write_json(cls, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=cls._json_default)

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
        import matplotlib.pyplot as plt

        path = Path(path)
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

        # Canonical lightweight artifacts for Codex web and post-hoc analysis.
        metrics_dict = self.to_metrics_dict()
        sorted_items = self._sorted_result_items()
        sorted_values = [scan_value for scan_value, _ in sorted_items]
        self._write_json(path / 'metrics.json', {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
            "contract": self.metadata.contract,
            "factor_payload_contract": self.factor_payload_contract(),
            "scan_dimension": self.scan_dimension,
            "scan_values": [str(v) for v in self.scan_values],
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

        # Collect all tensors into unified structure
        if save_tensors:
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

            if all_W:
                # Stack: (num_alphas, S, N1, M)
                results_data['W_students'] = torch.stack(all_W, dim=0)
            if all_X:
                # Stack: (num_alphas, S, M, N2)
                results_data['X_students'] = torch.stack(all_X, dim=0)

            # Add teacher matrices
            if self.W_teacher is not None:
                results_data['W_teacher'] = self.W_teacher.cpu().to(torch.float16)
            if self.X_teacher is not None:
                results_data['X_teacher'] = self.X_teacher.cpu().to(torch.float16)

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

        # Generate evolution plots (only if user does NOT have custom plots configured)
        sorted_values = [scan_value for scan_value, _ in sorted_items]

        # Determine x-axis label based on scan dimension
        x_label = 'Alpha' if self.scan_dimension == 'alpha' else 'Steps'

        # Check if user has custom plots configured - if so, skip default plots
        has_custom_plots = output_options and output_options.get('plots')
        output_contract_check = self._validate_output_contracts_before_plots(output_options or {}, path)
        self._write_json(path / 'output_contract.json', output_contract_check.to_dict())

        if not has_custom_plots:
            # Extract metrics for plotting
            x_values = [float(v) for v in sorted_values]
            metric_keys = self._available_metric_keys()
            q_y_key = self._first_available_metric(
                metric_keys,
                ["Q_Y_mean", "Q_Y_observed_mean", "physical_overlap_Y_mean"],
            )
            if q_y_key is None:
                raise ValueError(
                    "scalar_curves output requires one of "
                    "['Q_Y_mean', 'Q_Y_observed_mean', 'physical_overlap_Y_mean']; "
                    f"available metrics: {sorted(metric_keys)}"
                )
            q_w_key = self._first_available_metric(metric_keys, ["Q_W_mean", "Q_W_prime_mean"])
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
            ax.set_ylabel('Overlap')
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
        if output_options and output_options.get('plots'):
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
                    matrix = r.metrics.get('overlap_matrix') if r.metrics else None
                    if matrix is not None:
                        metric_code = r.metrics.get(
                            'overlap_matrix_metric',
                            output_options.get('heatmap_metric', 'Q_Y') if output_options else 'Q_Y',
                        )
                        metric_code = str(metric_code).upper()
                        if metric_code == "Q_W":
                            metric_name = "Factor Gram Overlap ($Q_W$)"
                            filename_prefix = "heatmap_W"
                        else:
                            metric_code = "Q_Y"
                            metric_name = "Tensor Overlap ($Q_Y$)"
                            filename_prefix = "heatmap_Y"

                        heatmap_path = plot_replica_heatmap(
                            np.asarray(matrix, dtype=float), float(v), plots_dir,
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

                        # Build interaction matrix
                        matrix_W = build_interaction_matrix(
                            W_s, W_teacher, gram_overlap_normalized, use_left=True
                        )

                        # Save heatmap
                        heatmap_path = plot_replica_heatmap(
                            matrix_W, float(v), plots_dir,
                            metric_name="Q_W", filename_prefix="heatmap_W",
                            rsb_ordering=rsb_ordering,
                            enhance_high_values=not uniform_colormap,  # uniform = no enhancement
                        )
                        if heatmap_path:
                            heatmap_paths.append(heatmap_path)
                            heatmap_metric_codes.append("Q_W")

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
            if key in metric_keys:
                return key
        return None

    def _metric_series(self, sorted_values: List[Any], metric_key: str) -> List[float]:
        missing_values = [
            value for value in sorted_values
            if metric_key not in (self.results[value].metrics or {})
        ]
        if missing_values:
            raise ValueError(
                f"scalar_curves output requires metric '{metric_key}' for all scan values; "
                f"missing values: {missing_values}"
            )
        return [float(self.results[value].metrics[metric_key]) for value in sorted_values]

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
            "Q_Y_observed_mean": "Q_Y observed",
            "physical_overlap_Y_mean": "physical overlap Y",
            "Q_W_mean": "Q_W",
            "Q_W_prime_mean": "Q_W prime",
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

        tensor_path = path / 'artifacts' / 'results.pt'
        if not tensor_path.exists():
            tensor_path = path / 'results.pt'
        tensor_data = torch.load(tensor_path, map_location='cpu') if tensor_path.exists() else {}
        w_stack = tensor_data.get('W_students') if isinstance(tensor_data, dict) else None
        x_stack = tensor_data.get('X_students') if isinstance(tensor_data, dict) else None

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
