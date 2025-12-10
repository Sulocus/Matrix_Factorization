"""
SMF Configuration System (v2).

Defines the complete structure of the experiment configuration using Dataclasses.
This serves as the "Source of Truth" for the entire system.

Design Principles:
1. All parameters from legacy system are preserved
2. UI can be auto-generated from this schema
3. Unified output format for all algorithms
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union
import yaml
import os
import dataclasses

# ============================================================================
# Type Aliases
# ============================================================================

AlgorithmMode = Literal["standard", "spreading", "spreading_parallel"]
FDistribution = Literal["gaussian", "rademacher"]
TeacherType = Literal["standard", "orthogonal", "scaled_variance"]
GraphType = Literal["random", "uniform", "low_loop"]
Language = Literal["zh", "en"]

# ============================================================================
# Sub-Configs
# ============================================================================

@dataclass
class AlphaConfig:
    """Alpha sweep configuration (always sweep, no single-point mode)."""
    start: float = 0.0
    stop: float = 4.0
    step: float = 0.1
    
    def to_list(self) -> List[float]:
        """Generate list of alpha values."""
        import numpy as np
        return np.arange(self.start, self.stop + self.step / 2, self.step).tolist()


@dataclass
class SpreadingConfig:
    """Spreading-mode specific configuration."""
    f_distribution: FDistribution = "rademacher"
    seed: int = 12345


@dataclass
class AlgorithmConfig:
    """Algorithm hyperparameters."""
    mode: AlgorithmMode = "spreading_parallel"
    damping: float = 0.5
    noise_var: float = 1e-10
    tolerance: float = 1e-6
    early_stop: bool = False
    use_compile: bool = True
    onsager_enabled: bool = True  # Enable Onsager correction


@dataclass
class TeacherConfig:
    """Teacher model settings."""
    type: TeacherType = "standard"
    variance_scale: float = 1.0  # Only used when type="scaled_variance"
    seed: int = 42


@dataclass
class GraphConfig:
    """Graph topology settings."""
    type: GraphType = "random"
    # LowLoop-specific parameters
    loop_order: int = 2        # k for 2k-loops (2=4-loops, 3=6-loops)
    n_sweeps: int = 5          # MCMC sweeps for loop reduction
    alpha_threshold: float = 0.8  # Only run MCMC when alpha < threshold
    # Multi-sample settings
    randomize_per_sample: bool = True  # If False, use same graph structure for all S samples


@dataclass
class MatrixConfig:
    """Matrix dimensions."""
    N1: int = 200
    N2: int = 200
    M: int = 50


@dataclass
class TrainingConfig:
    """Training loop parameters."""
    max_steps: int = 5000
    samples_per_alpha: int = 4  # S (number of samples per alpha)
    seed: int = 42
    device: str = "cuda"


@dataclass
class ExecutionConfig:
    """Output and execution settings."""
    metrics_to_compute: List[str] = field(default_factory=lambda: [
        "Q_Y", "Q_W", "Q_X", "Q_W_prime", "Q_X_prime", "MSE"
    ])
    include_qy_plot: bool = True
    include_summary_plot: bool = True


@dataclass
class UIConfig:
    """UI-specific settings."""
    language: Language = "zh"  # Default Chinese


# ============================================================================
# Root Config
# ============================================================================

@dataclass
class Config:
    """
    Root configuration object.
    
    All parameters needed to run an experiment are defined here.
    The UI auto-generates widgets based on this schema.
    """
    # Sub-configs
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    spreading: SpreadingConfig = field(default_factory=SpreadingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Experiment metadata
    experiment_name: str = "experiment"
    output_dir: str = "./outputs"

    # ========================================================================
    # Convenience Properties
    # ========================================================================
    
    @property
    def N(self) -> int:
        """Shorthand for N1 (assumes square matrix)."""
        return self.matrix.N1
    
    @property
    def M(self) -> int:
        """Shorthand for latent dimension."""
        return self.matrix.M
    
    @property
    def alpha_values(self) -> List[float]:
        """Get list of alpha values for sweep."""
        return self.alpha.to_list()
    
    @property
    def device(self):
        """Get torch device."""
        import torch
        return torch.device(self.training.device)

    # ========================================================================
    # Serialization
    # ========================================================================

    def save(self, path: str):
        """Save config to YAML file."""
        def as_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: as_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
            
        data = as_dict(self)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        def from_dict(klass, d):
            if d is None:
                return klass()
            args = {}
            for f in dataclasses.fields(klass):
                if f.name in d:
                    val = d[f.name]
                    if dataclasses.is_dataclass(f.type):
                        args[f.name] = from_dict(f.type, val)
                    elif hasattr(f.type, '__origin__'):
                        # Handle typing generics like List[str]
                        args[f.name] = val
                    else:
                        args[f.name] = val
            return klass(**args)

        return from_dict(cls, data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        def as_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: as_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        return as_dict(self)


# ============================================================================
# Available Metrics (for UI multiselect)
# ============================================================================

ALL_METRICS = [
    # Y metrics (cosine)
    "Q_Y",                    # Cosine similarity over ALL edges
    "Q_Y_observed",           # Cosine over observed edges
    "Q_Y_unobserved",         # Cosine over unobserved edges (generalization)
    # Y metrics (physical overlap)
    "physical_overlap_Y",     # Physical overlap over ALL edges
    "physical_overlap_Y_observed",  # Physical overlap on observed
    # W metrics
    "Q_W",                    # Gram cosine similarity
    "Q_W_prime",              # Normalized Gram overlap [0,1]
    "physical_overlap_W",     # Physical overlap |<Ws,Wt>|/<Wt,Wt>
    # X metrics
    "Q_X",                    # Gram cosine similarity
    "Q_X_prime",              # Normalized Gram overlap [0,1]
    "physical_overlap_X",     # Physical overlap |<Xs,Xt>|/<Xt,Xt>
    # Error metrics
    "MSE",                    # Mean squared error
]

# Metric descriptions for UI (Chinese/English)
METRIC_DESCRIPTIONS = {
    "zh": {
        "Q_Y": "Y余弦相似度 cos(Y_s, Y_t)",
        "Q_W": "W Gram余弦 cos(W_s@W_s^T, W_t@W_t^T)",
        "Q_X": "X Gram余弦 cos(X_s^T@X_s, X_t^T@X_t)",
        "Q_W_prime": "W归一化Overlap [0,1]",
        "Q_X_prime": "X归一化Overlap [0,1]",
        "Q_Y_unobserved": "未观测边Y余弦 (泛化能力)",
        "Q_Y_observed": "观测边Y余弦 (拟合能力)",
        "physical_overlap_Y": "物理Overlap <Y_s,Y_t>/<Y_t,Y_t>",
        "physical_overlap_W": "物理Overlap |<W_s,W_t>|/<W_t,W_t>",
        "physical_overlap_X": "物理Overlap |<X_s,X_t>|/<X_t,X_t>",
        "MSE": "均方误差 ||Y_s-Y_t||²/N",
        "Gen_Error": "泛化误差",
    },
    "en": {
        "Q_Y": "Y Cosine Similarity cos(Y_s, Y_t)",
        "Q_W": "W Gram Cosine cos(W_sW_s^T, W_tW_t^T)",
        "Q_X": "X Gram Cosine cos(X_s^TX_s, X_t^TX_t)",
        "Q_W_prime": "W Normalized Overlap [0,1]",
        "Q_X_prime": "X Normalized Overlap [0,1]",
        "Q_Y_unobserved": "Unobserved Y Cosine (Generalization)",
        "Q_Y_observed": "Observed Y Cosine (Fitting)",
        "physical_overlap_Y": "Physical Overlap <Y_s,Y_t>/<Y_t,Y_t>",
        "physical_overlap_W": "Physical Overlap |<W_s,W_t>|/<W_t,W_t>",
        "physical_overlap_X": "Physical Overlap |<X_s,X_t>|/<X_t,X_t>",
        "MSE": "Mean Squared Error ||Y_s-Y_t||²/N",
        "Gen_Error": "Generalization Error",
    },
}

# UI Labels (Chinese/English)
UI_LABELS = {
    "zh": {
        # Tabs
        "tab_physics": "物理参数",
        "tab_algorithm": "算法配置",
        "tab_model": "模型配置",
        "tab_training": "训练与执行",
        # Matrix
        "N1": "矩阵行数 N1",
        "N2": "矩阵列数 N2",
        "M": "潜在维度 M",
        # Alpha
        "alpha_start": "Alpha 起始值",
        "alpha_stop": "Alpha 终止值",
        "alpha_step": "Alpha 步长",
        # Algorithm
        "mode": "算法模式",
        "damping": "阻尼系数",
        "noise_var": "噪声方差",
        "tolerance": "收敛阈值",
        "early_stop": "启用早停",
        "use_compile": "启用 torch.compile",
        "onsager_enabled": "启用 Onsager 校正",
        # Spreading
        "f_distribution": "F 分布类型",
        "spreading_seed": "F 生成种子",
        # Teacher
        "teacher_type": "教师模型类型",
        "variance_scale": "方差缩放因子",
        # Graph
        "graph_type": "图拓扑类型",
        "loop_order": "环阶数 (k for 2k-loops)",
        "n_sweeps": "MCMC 扫描次数",
        # Training
        "max_steps": "最大迭代步数",
        "samples_per_alpha": "每Alpha样本数 (S)",
        "device": "计算设备",
        "seed": "随机种子",
        # Execution
        "metrics_to_compute": "选择要计算的指标",
        # Actions
        "run_experiment": "🚀 运行实验",
        "download_config": "📥 下载配置",
        "language": "界面语言",
        # Results
        "results_title": "实验结果",
        "metrics_title": "指标曲线",
        "comparison_title": "历史对比",
    },
    "en": {
        # Tabs
        "tab_physics": "Physics",
        "tab_algorithm": "Algorithm",
        "tab_model": "Model",
        "tab_training": "Training & Execution",
        # Matrix
        "N1": "Matrix Rows N1",
        "N2": "Matrix Cols N2",
        "M": "Latent Dimension M",
        # Alpha
        "alpha_start": "Alpha Start",
        "alpha_stop": "Alpha Stop",
        "alpha_step": "Alpha Step",
        # Algorithm
        "mode": "Algorithm Mode",
        "damping": "Damping Factor",
        "noise_var": "Noise Variance",
        "tolerance": "Convergence Threshold",
        "early_stop": "Enable Early Stop",
        "use_compile": "Enable torch.compile",
        "onsager_enabled": "Enable Onsager Correction",
        # Spreading
        "f_distribution": "F Distribution",
        "spreading_seed": "F Generation Seed",
        # Teacher
        "teacher_type": "Teacher Model Type",
        "variance_scale": "Variance Scale Factor",
        # Graph
        "graph_type": "Graph Topology",
        "loop_order": "Loop Order (k for 2k-loops)",
        "n_sweeps": "MCMC Sweeps",
        # Training
        "max_steps": "Max Iterations",
        "samples_per_alpha": "Samples per Alpha (S)",
        "device": "Compute Device",
        "seed": "Random Seed",
        # Execution
        "metrics_to_compute": "Select Metrics to Compute",
        # Actions
        "run_experiment": "🚀 Run Experiment",
        "download_config": "📥 Download Config",
        "language": "UI Language",
        # Results
        "results_title": "Experiment Results",
        "metrics_title": "Metric Curves",
        "comparison_title": "Historical Comparison",
    },
}
