"""
Module knowledge base for LLM-assisted configuration.

Provides comprehensive information about all available modules,
their parameters, best use cases, and common combinations.

NOTE: This file provides DETAILED knowledge about modules for LLM understanding.
The canonical list of available modules comes from the registry dynamically.
If a module is registered but not in these dicts, basic info is auto-generated.
"""

from typing import Dict, List, Any


def _ensure_registry_sync():
    """
    Ensure all registered modules have at least basic knowledge entries.

    This syncs the knowledge dicts with the registry, adding placeholder
    entries for any modules that are registered but missing from knowledge.
    """
    try:
        from ..modules import list_algorithms, list_graphs, list_teachers
    except ImportError:
        return  # Registry not available (shouldn't happen)

    # Sync algorithms
    for info in list_algorithms():
        if info.key not in ALGORITHMS:
            ALGORITHMS[info.key] = {
                "name": info.name,
                "description": info.description or "No detailed description",
                "best_for": ["See module docstring"],
                "params": info.default_params,
                "note": "Auto-generated from registry",
            }

    # Sync graphs
    for info in list_graphs():
        if info.key not in GRAPHS:
            GRAPHS[info.key] = {
                "name": info.name,
                "description": info.description or "No detailed description",
                "best_for": ["See module docstring"],
                "params": info.default_params,
                "note": "Auto-generated from registry",
            }

    # Sync teachers
    for info in list_teachers():
        if info.key not in TEACHERS:
            TEACHERS[info.key] = {
                "name": info.name,
                "description": info.description or "No detailed description",
                "best_for": ["See module docstring"],
                "params": info.default_params,
                "note": "Auto-generated from registry",
            }


# ============================================================
# Algorithm Knowledge
# ============================================================

ALGORITHMS = {
    "bigamp": {
        "name": "BiG-AMP",
        "full_name": "Bilinear Generalized Approximate Message Passing",
        "description": "Message passing algorithm with fast convergence",
        "best_for": [
            "Large matrices (N > 1000)",
            "General phase transition study",
            "Speed-critical experiments",
        ],
        "params": {
            "damping": {
                "default": 0.5,
                "range": [0.1, 0.9],
                "description": "Message damping factor for stability",
            },
            "noise_var": {
                "default": 1e-10,
                "range": [1e-12, 1e-6],
                "description": "Noise variance for numerical stability",
            },
            "max_steps": {
                "default": 5000,
                "recommended_range": [200, 10000],
                "description": "Maximum iteration steps",
            },
        },
        "typical_convergence": "200-5000 steps",
        "memory_scaling": "O(N1 * N2 * M)",
    },
    "agd": {
        "name": "AGD",
        "full_name": "Alternating Gradient Descent",
        "description": "Gradient-based optimization with alternating updates",
        "best_for": [
            "Small to medium matrices (N < 500)",
            "Precise convergence needed",
            "When BiG-AMP diverges",
        ],
        "params": {
            "learning_rate": {
                "default": 0.01,
                "range": [0.001, 0.1],
                "description": "Step size for gradient updates",
            },
            "max_epochs": {
                "default": 20000,
                "recommended_range": [5000, 50000],
                "description": "Maximum training epochs",
            },
            "use_early_stop": {
                "default": True,
                "description": "Enable early stopping on convergence",
            },
        },
        "typical_convergence": "10k-30k epochs",
        "memory_scaling": "O(N1 * M + M * N2)",
    },
}


# ============================================================
# Graph (Mask) Knowledge
# ============================================================

GRAPHS = {
    "random": {
        "name": "Random Graph",
        "description": "Erdos-Renyi random graph with specified observation density",
        "best_for": [
            "Standard experiments",
            "Baseline comparisons",
            "General phase transition study",
        ],
        "params": {},
        "note": "Most common choice for phase transition experiments",
    },
    "uniform": {
        "name": "Uniform Graph",
        "description": "Uniform sampling graph",
        "best_for": [
            "Uniform observation patterns",
            "Structured sampling",
        ],
        "params": {},
        "note": "Alternative to random graph",
    },
    "dinic": {
        "name": "Bi-regular Graph (Dinic)",
        "description": "Strict bi-regular graph using Dinic max-flow algorithm",
        "best_for": [
            "Theoretical analysis requiring degree regularity",
            "Comparing with random graphs",
            "Studies where exact degree matters",
        ],
        "params": {},
        "note": "Each left node has exactly degree=round(alpha*M) edges",
    },
    "low_loop": {
        "name": "Low-Loop Graph (MCMC)",
        "description": "Graph with minimized short cycles using MCMC edge-switching",
        "best_for": [
            "Studying effect of loops on phase transition",
            "Comparing with random graph (has loops)",
            "AMP convergence analysis",
        ],
        "params": {
            "loop_order": {"default": 2, "description": "k for 2k-loops (2=4-loops, 3=6-loops)"},
            "n_sweeps": {"default": 5, "description": "MCMC sweeps (more=better)"},
        },
        "note": "C4-free possible for alpha<0.35 (Kovari-Sos-Turan theorem)",
    },
    "combined": {
        "name": "Combined Graph",
        "description": "Flexible graph combining multiple features",
        "best_for": [
            "LLM-driven configuration",
            "Mixed requirements",
        ],
        "params": {},
        "note": "Use from_natural_language() for easy configuration",
    },
}


# ============================================================
# Teacher Knowledge
# ============================================================

TEACHERS = {
    "standard": {
        "name": "Standard Teacher",
        "description": "Random Gaussian W and X matrices",
        "best_for": [
            "Standard experiments",
            "Phase transition study",
            "Most use cases",
        ],
        "params": {},
        "note": "W ~ N(0, 1/M), X ~ N(0, 1/M)",
    },
    "scaled_variance": {
        "name": "Scaled Variance Teacher",
        "description": "Teacher with scaled variance initialization",
        "best_for": [
            "Studying variance effects",
            "Initialization scale experiments",
        ],
        "params": {},
        "note": "Variance scaled by factor",
    },
    "orthogonal": {
        "name": "Orthogonal Teacher",
        "description": "QR-decomposed teacher with orthonormal W columns and X rows",
        "best_for": [
            "Eliminating finite-size fluctuations",
            "Simulating thermodynamic limit (N→∞)",
            "Precise phase transition studies",
        ],
        "params": {},
        "note": "W^T W = I_M, X X^T = I_M. Removes 2*alpha*M/N linear bias",
    },
    "combined": {
        "name": "Combined Teacher",
        "description": "Flexible teacher combining orthogonal and scaling features",
        "best_for": [
            "LLM-driven configuration",
            "Mixed requirements (e.g., orthogonal + scaled)",
        ],
        "params": {
            "orthogonal": {"default": False, "description": "Use QR orthogonalization"},
            "scale": {"default": 1.0, "description": "Variance scaling factor"},
        },
        "note": "Use from_natural_language() for easy configuration",
    },
}


# ============================================================
# Experiment Types
# ============================================================

EXPERIMENT_TYPES = {
    "standard": {
        "name": "Standard Experiment",
        "description": "Single (N, M) configuration, full α sweep",
        "output": "Q_Y vs α phase transition curve",
        "typical_config": {
            "N1": 200,
            "N2": 200,
            "M": 50,
            "alpha_range": [0.0, 4.0],
            "alpha_step": 0.1,
        },
    },
    "size_scaling": {
        "name": "Size Scaling (Finite-Size Effect)",
        "description": "Multiple N values to study finite-size effects",
        "output": "Multiple Q_Y curves on same plot",
        "typical_config": {
            "N_values": [1000, 2000, 3000, 4000, 5000],
            "M": 100,
            "alpha_range": [0.0, 0.5],
            "alpha_step": 0.01,
        },
        "note": "Study deviations from thermodynamic limit",
    },
    "init_scale": {
        "name": "Initialization Scale",
        "description": "Compare different k/√M initialization scales",
        "output": "Q_Y curves for different k values",
        "typical_config": {
            "k_values": [0.5, 1.0, 1.5, 2.0],
        },
    },
    "replica": {
        "name": "Replica Overlap",
        "description": "Measure solution uniqueness via replica overlap",
        "output": "Replica-replica vs teacher-student overlap",
        "typical_config": {
            "samples": 100,
        },
    },
}


# ============================================================
# Plotting Functions (绘图功能)
# ============================================================

PLOTTING_FUNCTIONS = {
    "ylim_adjust": {
        "name": "Y轴范围调整",
        "description": "调整图表Y轴范围，支持自适应或固定范围",
        "keywords": ["放大", "Y轴", "太矮", "看不清", "ylim", "纵坐标"],
        "params": {
            "ylim": {"type": "tuple", "default": "auto", "example": "[0, 0.3]"},
        },
        "note": "当前 SMF 绘图固定 Y 轴 [0, 1.05]，此功能需要修改 plotting.py",
    },
    "xlim_adjust": {
        "name": "X轴范围调整",
        "description": "裁剪alpha范围，只显示部分区域",
        "keywords": ["前面", "alpha小", "裁剪", "xlim", "横坐标", "范围"],
        "params": {
            "xlim": {"type": "tuple", "default": None, "example": "[0, 1.5]"},
        },
    },
    "multi_curve": {
        "name": "多曲线叠加",
        "description": "将多个实验结果叠加在同一张图上对比",
        "keywords": ["放一起", "叠加", "对比", "多个", "一起画"],
        "params": {
            "results": {"type": "list", "description": "多个结果文件路径"},
            "labels": {"type": "list", "description": "每条曲线的标签"},
        },
    },
    "error_band": {
        "name": "误差带",
        "description": "用色带而非误差线显示误差范围",
        "keywords": ["色带", "误差带", "fill", "填充", "不要误差线"],
        "params": {
            "error_style": {"type": "str", "options": ["bar", "band"], "default": "bar"},
            "alpha": {"type": "float", "default": 0.3, "description": "透明度"},
        },
    },
    "twin_axis": {
        "name": "双Y轴",
        "description": "左右两个Y轴显示不同指标",
        "keywords": ["双轴", "两个Y", "斜率", "一起画", "双Y轴"],
        "params": {
            "left_metric": {"type": "str", "default": "Q_Y"},
            "right_metric": {"type": "str", "default": "slope"},
        },
    },
    "inset": {
        "name": "嵌入放大图",
        "description": "在主图角落添加放大的子图",
        "keywords": ["放大图", "角落", "inset", "细节", "相变点附近"],
        "params": {
            "inset_xlim": {"type": "tuple", "description": "放大区域的X范围"},
            "position": {"type": "str", "options": ["upper left", "upper right", "lower left", "lower right"]},
        },
    },
    "colormap": {
        "name": "配色方案",
        "description": "更换图表配色",
        "keywords": ["颜色", "配色", "好看", "换色"],
        "params": {
            "colormap": {"type": "str", "options": ["default", "viridis", "plasma", "coolwarm"]},
        },
    },
    "legend_position": {
        "name": "图例位置",
        "description": "调整图例位置，避免遮挡曲线",
        "keywords": ["图例", "挡住", "移开", "外面", "legend"],
        "params": {
            "loc": {"type": "str", "options": ["best", "upper right", "outside right", "below"]},
        },
    },
    "save_format": {
        "name": "保存格式",
        "description": "指定输出图片格式",
        "keywords": ["PDF", "矢量", "PNG", "格式", "保存"],
        "params": {
            "format": {"type": "str", "options": ["png", "pdf", "svg", "eps"]},
        },
    },
    "dpi": {
        "name": "分辨率",
        "description": "设置输出图片DPI",
        "keywords": ["dpi", "分辨率", "出版", "高清", "300"],
        "params": {
            "dpi": {"type": "int", "default": 150, "recommended": 300},
        },
    },
}


# ============================================================
# Parameter Warnings (参数边界警告)
# ============================================================

PARAMETER_WARNINGS = {
    "alpha_stop": {
        "warning_threshold": 5.0,
        "warning_message": "α>5 不常见，相变通常在 α∈[0.5, 3]",
    },
    "N_large": {
        "warning_threshold": 5000,
        "warning_message": "N>5000 需要大量显存，约 N*N*M*4 bytes",
    },
    "N_small": {
        "warning_threshold": 50,
        "warning_message": "N<50 有限尺寸效应显著",
    },
    "max_steps_small": {
        "warning_threshold": 100,
        "warning_message": "BiG-AMP 步数<100 可能未收敛",
    },
    "max_epochs_small": {
        "warning_threshold": 1000,
        "warning_message": "AGD epochs<1000 可能未收敛",
    },
    "damping_zero": {
        "warning_condition": "value == 0",
        "warning_message": "damping=0 会导致震荡不收敛",
    },
    "M_vs_N": {
        "warning_condition": "M > N",
        "warning_message": "M > N 没有物理意义",
    },
}


# ============================================================
# Precision Options (精度选项)
# ============================================================

PRECISION_OPTIONS = {
    "bf16": {
        "name": "BF16 混合精度",
        "config_key": "use_bf16",
        "value": True,
        "best_for": ["大矩阵加速", "节省显存"],
        "note": "GPU 默认开启，精度足够",
    },
    "fp32": {
        "name": "FP32 全精度",
        "config_key": "use_bf16",
        "value": False,
        "best_for": ["精度对比", "数值稳定性分析"],
        "note": "CPU 默认使用，速度较慢",
    },
}


# ============================================================
# Noise Options (噪声配置)
# ============================================================

NOISE_OPTIONS = {
    "numerical_stability": {
        "param": "noise_var",
        "typical_values": [1e-12, 1e-10],
        "use_case": "默认设置，不影响物理结果",
    },
    "observation_noise": {
        "param": "noise_var",
        "typical_values": [0.01, 0.1, 0.5, 1.0],
        "use_case": "噪声实验，研究相变如何'变软'",
        "physical_effect": "增大噪声使相变更平缓",
    },
}


# ============================================================
# Common Combinations
# ============================================================

COMMON_COMBINATIONS = [
    {
        "name": "Quick Baseline",
        "description": "Fast baseline phase transition",
        "config": {
            "algorithm": "bigamp",
            "graph": "random",
            "teacher": "standard",
            "N1": 200,
            "N2": 200,
            "M": 50,
            "max_steps": 1000,
        },
    },
    {
        "name": "Large Matrix Study",
        "description": "Phase transition in large systems",
        "config": {
            "algorithm": "bigamp",
            "graph": "random",
            "teacher": "standard",
            "N1": 5000,
            "N2": 5000,
            "M": 500,
            "max_steps": 5000,
        },
    },
    {
        "name": "Finite-Size Effect",
        "description": "Study size scaling behavior",
        "experiment_type": "size_scaling",
        "config": {
            "algorithm": "bigamp",
            "graph": "random",
            "N_values": [1000, 2000, 3000, 4000, 5000],
            "M": 100,
            "alpha_range": [0.0, 0.5],
        },
    },
    {
        "name": "Graph Structure Study",
        "description": "Compare random vs loop-free graphs",
        "config": {
            "algorithm": "bigamp",
            "graphs": ["random", "loop_free"],
        },
    },
]


# ============================================================
# Formatting Functions
# ============================================================

def format_as_prompt_context() -> str:
    """
    Format the complete knowledge base as LLM prompt context.

    Returns:
        Formatted string for use as system context in LLM calls.
    """
    # First, sync with registry to ensure all modules are included
    _ensure_registry_sync()

    lines = []

    # Algorithms
    lines.append("=== Algorithms ===")
    for key, info in ALGORITHMS.items():
        lines.append(f"\n[{key}] {info['name']}")
        lines.append(f"  Description: {info['description']}")
        if 'best_for' in info:
            lines.append(f"  Best for: {', '.join(info['best_for'])}")
        if 'typical_convergence' in info:
            lines.append(f"  Convergence: {info['typical_convergence']}")

    # Graphs
    lines.append("\n\n=== Graph Types ===")
    for key, info in GRAPHS.items():
        lines.append(f"\n[{key}] {info['name']}")
        lines.append(f"  Description: {info['description']}")
        if 'best_for' in info:
            lines.append(f"  Best for: {', '.join(info['best_for'])}")

    # Teachers
    lines.append("\n\n=== Teacher Types ===")
    for key, info in TEACHERS.items():
        lines.append(f"\n[{key}] {info['name']}")
        lines.append(f"  Description: {info['description']}")
        if 'best_for' in info:
            lines.append(f"  Best for: {', '.join(info['best_for'])}")
        if 'note' in info:
            lines.append(f"  Note: {info['note']}")

    # Experiment Types
    lines.append("\n\n=== Experiment Types ===")
    for key, info in EXPERIMENT_TYPES.items():
        lines.append(f"\n[{key}] {info['name']}")
        lines.append(f"  Description: {info['description']}")
        lines.append(f"  Output: {info['output']}")

    # Plotting Functions (新增)
    lines.append("\n\n=== Plotting Functions (绘图功能) ===")
    lines.append("注意: 如果用户请求涉及绘图调整（Y轴、颜色、格式等），这是绘图任务，不是实验配置任务。")
    lines.append("应返回 experiment_type='plotting' 并在 config 中包含绘图参数。")
    for key, info in PLOTTING_FUNCTIONS.items():
        lines.append(f"\n[{key}] {info['name']}")
        lines.append(f"  Description: {info['description']}")
        lines.append(f"  Keywords: {', '.join(info['keywords'])}")

    # Common Combinations
    lines.append("\n\n=== Recommended Combinations ===")
    for combo in COMMON_COMBINATIONS:
        lines.append(f"\n• {combo['name']}: {combo['description']}")

    # Parameter Warnings
    lines.append("\n\n=== Parameter Warnings (参数边界警告) ===")
    lines.append("When user specifies values outside safe ranges, add to missing_important:")
    for key, info in PARAMETER_WARNINGS.items():
        if 'warning_threshold' in info:
            lines.append(f"  • {key}: threshold={info['warning_threshold']}, {info['warning_message']}")
        else:
            lines.append(f"  • {key}: {info['warning_message']}")

    # Precision Options
    lines.append("\n\n=== Precision Options (精度选项) ===")
    lines.append("关键词: 全精度、FP32、BF16、混合精度、精度对比")
    for key, info in PRECISION_OPTIONS.items():
        lines.append(f"  • {info['name']}: config.use_bf16={info['value']}, {info['note']}")

    # Noise Options
    lines.append("\n\n=== Noise Options (噪声配置) ===")
    lines.append("关键词: 噪声、noise、变软、平缓")
    for key, info in NOISE_OPTIONS.items():
        lines.append(f"  • {key}: {info['param']}={info['typical_values']}, {info['use_case']}")

    return '\n'.join(lines)


def get_algorithm_info(key: str) -> Dict[str, Any]:
    """Get information about a specific algorithm."""
    return ALGORITHMS.get(key, {})


def get_graph_info(key: str) -> Dict[str, Any]:
    """Get information about a specific graph type."""
    return GRAPHS.get(key, {})


def get_experiment_type_info(key: str) -> Dict[str, Any]:
    """Get information about a specific experiment type."""
    return EXPERIMENT_TYPES.get(key, {})


def list_all_options() -> Dict[str, List[str]]:
    """List all available options for each category."""
    return {
        "algorithms": list(ALGORITHMS.keys()),
        "graphs": list(GRAPHS.keys()),
        "teachers": list(TEACHERS.keys()),
        "experiment_types": list(EXPERIMENT_TYPES.keys()),
    }
