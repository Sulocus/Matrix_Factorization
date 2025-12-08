"""
Execution Plan - describes what will be executed.

This module provides a preview of what runner.py will do,
ensuring UI display matches actual execution.

Supports:
- Single-step execution (standard experiment)
- Multi-step execution (comparison experiments)
- Post-processing hooks (merge plots, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .config import Config


@dataclass
class ExecutionStep:
    """
    Single execution step for multi-step plans.

    Used when user requests comparisons (e.g., "compare random vs dinic").
    Each step runs independently and saves its own results.
    """
    config_dict: Dict[str, Any]   # Config as dict (for serialization)
    label: str                     # Display label: "Random Graph"
    result_path: Optional[Path] = None  # Filled after execution


@dataclass
class ModuleCall:
    """Single module call information."""
    key: str              # Module identifier: teacher, graph, algorithm, etc.
    module_type: str      # Module type from registry
    module_name: str      # Display name: e.g., "BiG-AMP"
    params: Dict[str, Any] = field(default_factory=dict)  # Key parameters


@dataclass
class ExecutionPlan:
    """
    Execution Plan - describes all modules to be executed.

    This is a preview layer that mirrors runner.py's initialization logic,
    but does not actually execute anything.
    """

    # Core modules (in execution order)
    teacher: ModuleCall
    graph: ModuleCall
    algorithm: ModuleCall

    # Parameter configuration
    matrix_info: str      # "1000×1000, M=100"
    alpha_info: str       # "0.0 ~ 3.0, step 0.1 (31 points)"
    alpha_count: int      # Number of alpha points

    # Evaluation configuration
    metrics: List[str]    # ["Q_Y", "Q_W", ...]
    plots: List[str]      # ["summary.png", "qy_vs_alpha.png", ...]

    # Algorithm-specific parameters (with defaults, must come after non-default fields)
    damping: float = 0.5          # BiG-AMP damping factor
    samples_per_alpha: int = 1    # Number of trials per alpha

    # Execution strategy (inferred from runner logic)
    execution_mode: str = "parallel"  # "parallel" | "sequential"
    estimated_time: Optional[str] = None

    # Multi-step support (for comparison experiments)
    steps: Optional[List[ExecutionStep]] = None  # None = single-step mode
    post_process: Optional[List[Dict[str, Any]]] = None  # Post-processing hooks
    # post_process format:
    # [{"type": "merge_plot",
    #   "sources": [0, 1],           # Step indices to merge
    #   "labels": ["Random", "Dinic"],
    #   "output": "comparison.png",
    #   "metric": "Q_Y"}]            # Which metric to plot

    def is_comparison(self) -> bool:
        """Check if this is a multi-step comparison plan."""
        return self.steps is not None and len(self.steps) > 1

    def get_step_count(self) -> int:
        """Get number of execution steps (1 for single-step plans)."""
        return len(self.steps) if self.steps else 1

    def to_display_list(self, lang: str = 'cn') -> List[Dict[str, Any]]:
        """
        Convert to UI display format.

        Returns a list of dicts with: key, label, value, is_special, edit_key
        """
        items = []

        # Labels
        labels = {
            'cn': {
                'teacher': '教师模型', 'graph': '图结构', 'algorithm': '算法',
                'algo_params': '算法参数',
                'matrix': '矩阵维度', 'alpha': 'Alpha范围',
                'metrics': '评估指标', 'outputs': '输出图表',
                'mode': '执行模式', 'steps': '执行步骤', 'post': '后处理'
            },
            'en': {
                'teacher': 'Teacher', 'graph': 'Graph', 'algorithm': 'Algorithm',
                'algo_params': 'Algo Params',
                'matrix': 'Matrix', 'alpha': 'Alpha',
                'metrics': 'Metrics', 'outputs': 'Plots',
                'mode': 'Mode', 'steps': 'Steps', 'post': 'Post-process'
            }
        }
        l = labels.get(lang, labels['cn'])

        # Multi-step comparison mode
        if self.is_comparison():
            mode_label = '对比实验' if lang == 'cn' else 'Comparison'
            step_labels = [s.label for s in self.steps]
            items.append({
                'key': 'mode',
                'label': l['mode'],
                'value': f"{mode_label} ({len(self.steps)} runs)",
                'is_special': True,
                'edit_key': str(len(items) + 1),
            })
            items.append({
                'key': 'steps',
                'label': l['steps'],
                'value': ' vs '.join(step_labels),
                'is_special': True,
                'edit_key': str(len(items) + 1),
            })
            if self.post_process:
                pp_desc = ', '.join(pp.get('output', 'merge') for pp in self.post_process)
                items.append({
                    'key': 'post',
                    'label': l['post'],
                    'value': pp_desc,
                    'is_special': False,
                    'edit_key': str(len(items) + 1),
                })

        # 1. Teacher
        items.append({
            'key': 'teacher',
            'label': l['teacher'],
            'value': self.teacher.module_name,
            'is_special': self.teacher.key != 'standard',
            'edit_key': str(len(items) + 1),
        })

        # 2. Graph
        items.append({
            'key': 'graph',
            'label': l['graph'],
            'value': self.graph.module_name,
            'is_special': self.graph.key != 'random',
            'edit_key': str(len(items) + 1),
        })

        # 3. Algorithm
        steps = self.algorithm.params.get('max_steps', 5000)
        step_label = '步' if lang == 'cn' else 'steps'
        items.append({
            'key': 'algorithm',
            'label': l['algorithm'],
            'value': f"{self.algorithm.module_name} ({steps} {step_label})",
            'is_special': False,
            'edit_key': str(len(items) + 1),
        })

        # 3.5. Algorithm parameters (damping, trials) - only show if non-default
        algo_params_parts = []
        if self.damping != 0.5:
            algo_params_parts.append(f"damping={self.damping}")
        if self.samples_per_alpha > 1:
            trials_label = '次采样' if lang == 'cn' else 'trials'
            algo_params_parts.append(f"{self.samples_per_alpha} {trials_label}")

        if algo_params_parts:
            items.append({
                'key': 'algo_params',
                'label': l['algo_params'],
                'value': ', '.join(algo_params_parts),
                'is_special': True,  # Non-default, so mark as special
                'edit_key': str(len(items) + 1),
            })

        # 4. Matrix
        items.append({
            'key': 'matrix',
            'label': l['matrix'],
            'value': self.matrix_info,
            'is_special': False,
            'edit_key': str(len(items) + 1),
        })

        # 5. Alpha
        items.append({
            'key': 'alpha',
            'label': l['alpha'],
            'value': self.alpha_info,
            'is_special': False,
            'edit_key': str(len(items) + 1),
        })

        # 6. Metrics
        items.append({
            'key': 'metrics',
            'label': l['metrics'],
            'value': ', '.join(self.metrics),
            'is_special': False,
            'edit_key': str(len(items) + 1),
        })

        # 7. Plots (only if any)
        if self.plots:
            items.append({
                'key': 'outputs',
                'label': l['outputs'],
                'value': ', '.join(self.plots),
                'is_special': False,
                'edit_key': str(len(items) + 1),
            })

        return items


def build_execution_plan(config: 'Config') -> ExecutionPlan:
    """
    Build ExecutionPlan from Config.

    This function mirrors runner.py's initialization logic,
    but does not actually execute - only generates a description
    of "what will be done".
    """
    from smf.modules.registry import get_algorithm, get_graph, get_teacher

    # Get module info (same as runner.__init__)
    algorithm_info = get_algorithm(config.algorithm_key)
    graph_info = get_graph(config.graph_key)
    teacher_info = get_teacher(config.teacher_key)

    # Build module call info
    teacher_call = ModuleCall(
        key=config.teacher_key,
        module_type='TeacherBase',
        module_name=teacher_info.name,
        params={}
    )

    graph_call = ModuleCall(
        key=config.graph_key,
        module_type='GraphBase',
        module_name=graph_info.name,
        params={}
    )

    algorithm_call = ModuleCall(
        key=config.algorithm_key,
        module_type='AlgorithmBase',
        module_name=algorithm_info.name,
        params={'max_steps': config.training.max_steps}
    )

    # Calculate alpha points
    alpha_values = config.alpha.get_values()
    alpha_count = len(alpha_values)
    alpha_info = f"{config.alpha.start} ~ {config.alpha.stop}, step {config.alpha.step} ({alpha_count} points)"

    # Infer execution mode (simplified, no actual GPU detection)
    # Check if algorithm class has supports_batch_training method
    algo_cls = algorithm_info.cls
    if hasattr(algo_cls, 'supports_batch_training'):
        try:
            # Try to call as class method or with dummy instance
            execution_mode = "parallel"  # Default to parallel for bigamp
        except Exception:
            execution_mode = "sequential"
    else:
        execution_mode = "sequential"

    # Collect metrics and plots
    metrics = list(config.execution.metrics_to_compute)

    plots = []
    if config.execution.include_summary_plot:
        plots.append("summary.png")
    if config.execution.include_qy_plot:
        plots.append("qy_vs_alpha.png")
    for p in config.execution.plots:
        plots.append(p.get('filename', 'custom.png'))

    # Matrix info
    matrix_info = f"{config.matrix.N1}×{config.matrix.N2}, M={config.matrix.M}"

    return ExecutionPlan(
        teacher=teacher_call,
        graph=graph_call,
        algorithm=algorithm_call,
        matrix_info=matrix_info,
        alpha_info=alpha_info,
        alpha_count=alpha_count,
        metrics=metrics,
        plots=plots,
        execution_mode=execution_mode,
    )


def build_execution_plan_from_dict(config_dict: Dict[str, Any],
                                    execution_params: Optional[Dict[str, Any]] = None,
                                    comparison_steps: Optional[List[Dict[str, Any]]] = None,
                                    post_process: Optional[List[Dict[str, Any]]] = None) -> ExecutionPlan:
    """
    Build ExecutionPlan from raw config dict (before Config object is created).

    This is useful in wizard.py where we only have AnalysisResult.config dict.

    Args:
        config_dict: Base configuration dictionary
        execution_params: Execution parameters (metrics, plots, etc.)
        comparison_steps: Optional list of comparison steps for multi-step experiments
        post_process: Optional list of post-processing hooks (e.g., merge_plot)
    """
    from smf.modules.registry import get_algorithm, get_graph, get_teacher

    # Extract keys with defaults
    algorithm_key = config_dict.get('algorithm_key', 'bigamp')
    graph_key = config_dict.get('graph_key', 'random')
    teacher_key = config_dict.get('teacher_key', 'standard')

    # Get module info
    try:
        algorithm_info = get_algorithm(algorithm_key)
    except KeyError:
        algorithm_info = type('MockInfo', (), {'name': algorithm_key, 'cls': None})()

    try:
        graph_info = get_graph(graph_key)
    except KeyError:
        graph_info = type('MockInfo', (), {'name': graph_key})()

    try:
        teacher_info = get_teacher(teacher_key)
    except KeyError:
        teacher_info = type('MockInfo', (), {'name': teacher_key})()

    # Build module calls
    teacher_call = ModuleCall(
        key=teacher_key,
        module_type='TeacherBase',
        module_name=teacher_info.name,
        params={}
    )

    graph_call = ModuleCall(
        key=graph_key,
        module_type='GraphBase',
        module_name=graph_info.name,
        params={}
    )

    max_steps = config_dict.get('max_steps', 5000)
    algorithm_call = ModuleCall(
        key=algorithm_key,
        module_type='AlgorithmBase',
        module_name=algorithm_info.name,
        params={'max_steps': max_steps}
    )

    # Alpha info
    alpha_start = config_dict.get('alpha_start', 0.0)
    alpha_stop = config_dict.get('alpha_stop', 4.0)
    alpha_step = config_dict.get('alpha_step', 0.1)

    # Calculate alpha count
    import numpy as np
    alpha_values = np.arange(alpha_start, alpha_stop + alpha_step / 2, alpha_step)
    alpha_count = len(alpha_values)
    alpha_info = f"{alpha_start} ~ {alpha_stop}, step {alpha_step} ({alpha_count} points)"

    # Matrix info
    N1 = config_dict.get('N1', 200)
    N2 = config_dict.get('N2', 200)
    M = config_dict.get('M', 50)
    matrix_info = f"{N1}×{N2}, M={M}"

    # Algorithm-specific parameters
    damping = config_dict.get('damping', 0.5)
    samples_per_alpha = config_dict.get('samples_per_alpha', 1)

    # Metrics and plots from execution_params
    exec_params = execution_params or {}
    metrics = exec_params.get('metrics_to_compute',
                              ['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error'])

    plots = []
    if exec_params.get('include_summary_plot', True):
        plots.append("summary.png")
    if exec_params.get('include_qy_plot', True):
        plots.append("qy_vs_alpha.png")
    for p in exec_params.get('plots', []):
        plots.append(p.get('filename', 'custom.png'))

    # Build ExecutionStep objects if comparison_steps provided
    steps = None
    if comparison_steps:
        steps = [
            ExecutionStep(
                config_dict=step.get('config', {}),
                label=step.get('label', f'Step {i+1}')
            )
            for i, step in enumerate(comparison_steps)
        ]

    return ExecutionPlan(
        teacher=teacher_call,
        graph=graph_call,
        algorithm=algorithm_call,
        matrix_info=matrix_info,
        alpha_info=alpha_info,
        alpha_count=alpha_count,
        damping=damping,
        samples_per_alpha=samples_per_alpha,
        metrics=metrics,
        plots=plots,
        execution_mode="parallel",  # Default
        steps=steps,
        post_process=post_process,
    )
