"""
Result storage module.
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json
import numpy as np
import torch

from ..registry import register_output
from .base import OutputBase
from ...core.config import Config


@register_output(
    key="storage",
    name="Data Storage",
    description="Save JSON metrics and NPZ raw data",
)
class ResultStorage(OutputBase):
    """
    Handles saving experiment results.

    Saves the Codex-web run layout:
    - manifest.json: artifact index and run status
    - config.yaml/config.json: configuration
    - metrics.json: JSON-first metrics
    - events.jsonl: lightweight run event stream
    - artifacts/raw_data.npz: optional raw arrays
    """

    def __init__(self, config: Config, output_dir: Path = None):
        if output_dir is None:
            # Generate default output directory
            # Format: {algorithm}_{graph}_{N1}x{N2}_M{M}_{MMDD_HHMM}
            # Example: bigamp_random_200x200_M50_1204_0539
            time_suffix = datetime.now().strftime("%m%d_%H%M")
            m = config.matrix
            dir_name = f"{config.algorithm_key}_{config.graph_key}_{m.N1}x{m.N2}_M{m.M}_{time_suffix}"
            output_dir = Path("runs") / dir_name

        super().__init__(config, output_dir)
        self.plots_dir = output_dir / "plots"
        self.artifacts_dir = output_dir / "artifacts"
        self.checkpoints_dir = output_dir / "checkpoints"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results: Dict[str, Any],
        raw_data: Dict[str, np.ndarray] = None,
        metadata: Dict[str, Any] = None,
    ) -> Path:
        """
        Save experiment results.

        Args:
            results: Dictionary of metrics (alpha -> metrics dict)
            raw_data: Optional raw numpy arrays to save
            metadata: Optional additional metadata

        Returns:
            Path to output directory
        """
        # ExperimentResult is the canonical producer; delegate to keep one schema.
        if hasattr(results, 'save') and hasattr(results, 'to_metrics_dict'):
            results.save(self.output_dir, save_tensors=metadata.get('save_tensors', True) if metadata else True)
            return self.output_dir

        # Save config
        if hasattr(self.config, 'to_yaml'):
            self.config.to_yaml(self.output_dir / "config.yaml")
        if hasattr(self.config, 'save'):
            self.config.save(self.output_dir / "config.json")

        # Prepare metrics for JSON
        # Handle ExperimentResult object
        if hasattr(results, 'to_metrics_dict'):
            metrics_dict = results.to_metrics_dict()
        elif hasattr(results, 'metrics'): # Fallback
             metrics_dict = results.metrics
        elif isinstance(results, dict):
            metrics_dict = results
        else:
            # Try to serialize whatever it is, or empty
            metrics_dict = {}

        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
        timestamp = datetime.now().isoformat()
        metrics_data = {
            "schema_version": 1,
            "config": config_dict,
            "metrics": metrics_dict,
            "results": metrics_dict,
            "metadata": metadata or {},
            "timestamp": timestamp,
        }

        # Save metrics JSON
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=self._json_serializer)

        # Save raw data if provided
        if raw_data:
            npz_path = self.artifacts_dir / "raw_data.npz"
            np.savez_compressed(npz_path, **raw_data)

        with open(self.output_dir / "events.jsonl", "a") as f:
            f.write(json.dumps({
                "type": "run_saved",
                "timestamp": timestamp,
                "status": "complete",
            }) + "\n")

        manifest = {
            "schema_version": 1,
            "run_id": self.output_dir.name,
            "status": "complete",
            "created_at": timestamp,
            "updated_at": timestamp,
            "config_summary": {
                "algorithm_key": config_dict.get("algorithm_key"),
                "graph_key": config_dict.get("graph_key"),
                "matrix": config_dict.get("matrix"),
            },
            "artifacts": [
                {"path": "config.yaml", "kind": "yaml"},
                {"path": "config.json", "kind": "json"},
                {"path": "metrics.json", "kind": "json"},
                {"path": "events.jsonl", "kind": "jsonl"},
            ],
        }
        if raw_data:
            manifest["artifacts"].append({"path": "artifacts/raw_data.npz", "kind": "npz"})
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=self._json_serializer)

        return self.output_dir

    def save_raw_batch(
        self,
        alpha: float,
        raw_data: Dict[str, torch.Tensor],
        metadata: Dict[str, Any] = None,
    ) -> Path:
        """
        Save raw data for a specific alpha batch (incremental saving).

        Optimizations:
        1. Auto-converts to float16 to save space.
        2. Compresses using npz.
        3. Saves to 'raw_data' subdirectory.

        Args:
            alpha: Current alpha value
            raw_data: Dictionary of torch tensors (W_s, X_s, etc.)
            metadata: Optional metadata dict

        Returns:
            Path to saved file
        """
        raw_dir = self.artifacts_dir / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for saving
        # Convert tensors to numpy float16
        save_dict = {}
        for k, v in raw_data.items():
            if isinstance(v, torch.Tensor):
                # Detach, move to cpu, cast to float16
                save_dict[k] = v.detach().cpu().to(torch.float16).numpy()
            elif isinstance(v, np.ndarray):
                save_dict[k] = v.astype(np.float16)
            else:
                save_dict[k] = v

        if metadata:
            save_dict['metadata'] = metadata

        # Format filename with fixed precision for sorting
        filename = f"alpha_{alpha:.6f}.npz"
        file_path = raw_dir / filename

        np.savez_compressed(file_path, **save_dict)
        return file_path

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    @classmethod
    def load(cls, result_dir: Path) -> Dict[str, Any]:
        """
        Load results from a saved directory.

        Args:
            result_dir: Path to result directory

        Returns:
            Dictionary with config, results, and raw_data (if exists)
        """
        result_dir = Path(result_dir)

        data = {}

        # Load config
        config_path = result_dir / "config.yaml"
        if config_path.exists():
            data['config'] = Config.from_yaml(config_path)

        # Load metrics
        metrics_path = result_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            data['results'] = metrics_data.get('results', {})
            data['metadata'] = metrics_data.get('metadata', {})
            data['timestamp'] = metrics_data.get('timestamp')

        # Load raw data if exists
        npz_path = result_dir / "artifacts" / "raw_data.npz"
        if not npz_path.exists():
            npz_path = result_dir / "raw_data.npz"
        if npz_path.exists():
            data['raw_data'] = dict(np.load(npz_path))

        return data

    def get_plots_dir(self) -> Path:
        """Get the plots subdirectory."""
        return self.plots_dir


def list_results(results_dir: Path = None) -> list[Dict[str, Any]]:
    """
    List all saved results from multiple directories.

    Scans:
    - runs/ (Codex-web run schema)
    - smf/results/ (legacy framework)
    - Result/ (legacy: standard baseline)
    - ResultNo4/ (legacy: loop-free experiments)
    - Result_compareNM/ (legacy: size scaling)
    - Result_replica/ (legacy: replica overlap)

    Args:
        results_dir: Root results directory (default: scans all known directories)

    Returns:
        List of result info dictionaries
    """
    # Define all result directories to scan with their experiment type hints
    # Format: (directory_path, type_hint_from_parent)
    result_dirs = []

    if results_dir is not None:
        # If specific directory given, only scan that
        result_dirs = [(Path(results_dir), None)]
    else:
        # Scan all known result directories
        base_path = Path(".")
        known_dirs = [
            ("runs", None),                  # Codex-web run schema
            ("smf/results", None),           # Legacy framework - infer from subdir name
            ("Result", "overlap_metrics"),    # Legacy baseline
            ("ResultNo4", "loop_free"),       # Legacy loop-free
            ("Result_compareNM", "size_scaling"),  # Legacy size scaling
            ("Result_replica", "replica"),    # Legacy replica overlap
        ]
        for dir_name, type_hint in known_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists():
                result_dirs.append((dir_path, type_hint))

    results = []

    for results_dir, parent_type_hint in result_dirs:
        if not results_dir.exists():
            continue

        for subdir in sorted(results_dir.iterdir(), reverse=True):
            if not subdir.is_dir():
                continue

            # Try multiple JSON file names
            data = None
            for json_name in ["metrics.json", "results.json", "multi_size_results*.json"]:
                if "*" in json_name:
                    # Glob pattern
                    matches = list(subdir.glob(json_name))
                    if matches:
                        try:
                            with open(matches[0], 'r') as f:
                                data = json.load(f)
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    json_path = subdir / json_name
                    if json_path.exists():
                        try:
                            with open(json_path, 'r') as f:
                                data = json.load(f)
                            break
                        except json.JSONDecodeError:
                            continue

            if data is None:
                continue

            try:
                config = data.get('config', {})
                matrix = config.get('matrix', {})

                # Determine experiment type - priority:
                # 1. Parent directory hint (for legacy directories)
                # 2. Subdirectory name pattern
                # 3. Default to overlap_metrics
                exp_type = parent_type_hint or "overlap_metrics"

                # Override based on subdirectory name patterns
                name_lower = subdir.name.lower()
                if "no4" in name_lower or "noloop" in name_lower or "c4" in name_lower or "loop" in name_lower:
                    exp_type = "loop_free"
                elif "replica" in name_lower:
                    exp_type = "replica"
                elif "large_matrix" in name_lower or "size_scaling" in name_lower or "multi_size" in name_lower:
                    exp_type = "size_scaling"
                elif "variance" in name_lower or "init_scale" in name_lower:
                    exp_type = "init_scale"

                # Handle different JSON structures
                N1 = matrix.get('N1')
                N2 = matrix.get('N2')
                M = matrix.get('M')

                # For large_matrix_sweep, extract from matrix_configs
                if N1 is None and 'matrix_configs' in data:
                    configs = data['matrix_configs']
                    if configs:
                        N1 = f"{configs[0][0]}-{configs[-1][0]}"
                        N2 = f"{configs[0][1]}-{configs[-1][1]}"
                        M = configs[0][2] if all(c[2] == configs[0][2] for c in configs) else "varies"

                # Extract from variance_scales if present (init_scale experiment)
                if 'variance_scales' in data:
                    exp_type = "init_scale"

                # Determine source for display
                source = "new" if any(name in str(results_dir) for name in ["smf/results", "runs"]) else "legacy"

                results.append({
                    'path': subdir,
                    'name': subdir.name,
                    'timestamp': data.get('timestamp'),
                    'N1': N1,
                    'N2': N2,
                    'M': M,
                    'algorithm': config.get('algorithm_key', 'bigamp'),
                    'graph': config.get('graph_key', 'random'),
                    'type': exp_type,
                    'source': source,
                })
            except (KeyError, TypeError):
                continue

    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x.get('timestamp') or '', reverse=True)

    return results
