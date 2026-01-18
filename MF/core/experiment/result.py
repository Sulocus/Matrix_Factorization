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
    W_state: torch.Tensor  # (S, N1, M) or (A, S, N1, M)
    X_state: torch.Tensor  # (S, M, N2) or (A, S, M, N2)
    
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
        result.save('results/exp_001/')
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
        import numpy as np
        import matplotlib.pyplot as plt
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        plots_dir = path / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Save config
        if hasattr(self.config, 'save'):
            self.config.save(path / 'config.json')
        else:
            with open(path / 'config.json', 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
        
        # Save metadata
        self.metadata.duration_total_seconds = sum(
            r.duration_seconds for r in self.results.values()
        )
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Prepare metrics dict for results.pt
        metrics_dict = {
            str(v): r.metrics for v, r in self.results.items()
        }
        
        # Collect all tensors into unified structure
        if save_tensors:
            sorted_values = sorted(self.results.keys(), key=lambda x: float(x) if isinstance(x, (int, float)) else x)
            
            # Stack all W_students and X_students across scan values
            all_W = []
            all_X = []
            for v in sorted_values:
                r = self.results[v]
                if r.W_students is not None:
                    all_W.append(r.W_students.cpu().to(torch.float16))
                if r.X_students is not None:
                    all_X.append(r.X_students.cpu().to(torch.float16))
            
            # Create unified results.pt
            results_data = {
                'alpha_values': [float(v) for v in sorted_values],
                'metrics': metrics_dict,
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
            
            torch.save(results_data, path / 'results.pt')
        
        # Generate evolution plots (only if user does NOT have custom plots configured)
        sorted_values = sorted(self.results.keys(), key=lambda x: float(x) if isinstance(x, (int, float)) else x)
        
        # Determine x-axis label based on scan dimension
        x_label = 'Alpha' if self.scan_dimension == 'alpha' else 'Steps'
        
        # Check if user has custom plots configured - if so, skip default plots
        has_custom_plots = output_options and output_options.get('plots')
        
        if not has_custom_plots:
            # Extract metrics for plotting
            x_values = [float(v) for v in sorted_values]
            q_w_means = [self.results[v].metrics.get('Q_W_mean', 0) for v in sorted_values]
            q_y_means = [self.results[v].metrics.get('Q_Y_mean', 0) for v in sorted_values]
            
            # Plot Q_W and Q_Y evolution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_values, q_w_means, 'r-', label='Q_W', linewidth=2, marker='o', markersize=3)
            ax.plot(x_values, q_y_means, 'g-', label='Q_Y', linewidth=2, marker='s', markersize=3)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Overlap')
            ax.set_title(f'{self.experiment_id}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-0.1, 1.1)
            plt.savefig(plots_dir / 'qy_evolution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Plot Q_W only (convergence curve for steps scan)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_values, q_w_means, 'r-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Q_W (Gram Overlap Normalized)')
            ax.set_title(f'Q_W Evolution - {self.experiment_id}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
            plt.savefig(plots_dir / 'overlap_evolution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # ===== 自定义曲线绘图 (来自 output_options['plots']) =====
        if output_options and output_options.get('plots'):
            try:
                from MF.modules.outputs.plotting import plot_custom_curves
                
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
            except Exception as e:
                import traceback
                print(f"Warning: Could not generate custom plots: {e}")
                traceback.print_exc()
        
        # ===== Heatmap 和 GIF (由 enable_heatmap 控制) =====
        enable_heatmap = True  # 默认开启
        if output_options:
            enable_heatmap = output_options.get('enable_heatmap', True)
        
        if enable_heatmap:
            try:
                from MF.modules.outputs.plotting import plot_replica_heatmap, create_gif
                from MF.modules.metrics.overlap import build_interaction_matrix, gram_overlap_normalized
                
                heatmap_paths = []
                W_teacher = self.W_teacher
                
                for v in sorted_values:
                    r = self.results[v]
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
                
                # Create GIF from heatmaps
                if heatmap_paths:
                    gif_path = create_gif(heatmap_paths, plots_dir / "animation_W.gif", duration=0.2)
                    if gif_path:
                        print(f"Generated GIF: {gif_path}")
            except Exception as e:
                import traceback
                print(f"Warning: Could not generate heatmaps/GIF: {e}")
                traceback.print_exc()
    
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
        
        # Load results
        with open(path / 'results.json', 'r') as f:
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
        
        # Load individual results
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
            )
            
            # Load tensors if they exist
            tensor_file = path / 'tensors' / f'scan_{v}.pt'
            if tensor_file.exists():
                tensors = torch.load(tensor_file)
                single_result.W_students = tensors['W_students']
                single_result.X_students = tensors['X_students']
            
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
