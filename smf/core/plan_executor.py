"""
Plan Executor - executes single or multi-step execution plans.

Supports:
- Single-step: Standard experiment (backward compatible)
- Multi-step: Comparison experiments (run N configs, then merge plots)

The executor is the bridge between ExecutionPlan and runner.run_experiment().
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .execution_plan import ExecutionPlan, ExecutionStep
from .config import Config
from .parameter_space import merge_config
from .llm_logger import get_logger


class PlanExecutor:
    """
    Execute single or multi-step plans.

    Usage:
        executor = PlanExecutor()
        result = executor.run(plan)
    """

    def __init__(self, progress_callback=None):
        """
        Initialize executor.

        Args:
            progress_callback: Optional callback(step_idx, total, label)
        """
        self.progress_callback = progress_callback

    def run(self, plan: ExecutionPlan, base_config: Optional[Config] = None) -> Dict[str, Any]:
        """
        Execute the plan.

        For single-step plans: runs the experiment directly.
        For multi-step plans: runs each step, then executes post-processing.

        Args:
            plan: ExecutionPlan to execute
            base_config: Base config (required for multi-step, optional for single-step)

        Returns:
            Dict with results, paths, and any generated plots
        """
        if plan.is_comparison():
            return self._run_multi_step(plan, base_config)
        else:
            return self._run_single_step(plan, base_config)

    def _run_single_step(self, plan: ExecutionPlan, base_config: Optional[Config]) -> Dict[str, Any]:
        """Run a single-step experiment (standard mode)."""
        from ..runner import run_experiment

        logger = get_logger()

        # If base_config provided, use it; otherwise build from plan
        if base_config is None:
            # Build config from plan's first step or from plan itself
            if plan.steps and len(plan.steps) == 1:
                base_config = self._build_config_from_dict(plan.steps[0].config_dict)
            else:
                # Single-step plan without explicit steps - use plan's module info
                logger.log_execution(
                    status="failed",
                    steps_completed=0,
                    steps_total=1,
                    error_message="Single-step plan requires base_config or plan.steps[0]"
                )
                raise ValueError("Single-step plan requires base_config or plan.steps[0]")

        try:
            result = run_experiment(base_config, save=True)
            result_path = result.get('result_path')
            logger.log_execution(
                status="success",
                steps_completed=1,
                steps_total=1,
                result_paths=[str(result_path)] if result_path else None
            )
            return result
        except Exception as e:
            logger.log_execution(
                status="failed",
                steps_completed=0,
                steps_total=1,
                error_message=str(e)
            )
            raise

    def _run_multi_step(self, plan: ExecutionPlan, base_config: Config) -> Dict[str, Any]:
        """
        Run a multi-step comparison experiment.

        1. Execute each step sequentially
        2. Save results for each step
        3. Run post-processing (merge plots, etc.)
        """
        from ..runner import run_experiment

        logger = get_logger()
        total_steps = len(plan.steps)

        if base_config is None:
            logger.log_execution(
                status="failed",
                steps_completed=0,
                steps_total=total_steps,
                error_message="Multi-step plans require a base_config"
            )
            raise ValueError("Multi-step plans require a base_config")

        all_results = {}
        step_paths = []
        completed_steps = 0

        # Create comparison output directory
        timestamp = datetime.now().strftime("%m%d_%H%M")
        comparison_dir = Path("smf/results") / f"comparison_{timestamp}"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Execute each step
        try:
            for idx, step in enumerate(plan.steps):
                if self.progress_callback:
                    self.progress_callback(idx, total_steps, step.label)

                # Merge step-specific config into base config
                step_config = merge_config(base_config, step.config_dict)

                # Run experiment
                result = run_experiment(step_config, save=True)

                # Store results
                step.result_path = result.get('result_path')
                all_results[step.label] = result.get('results', {})
                if step.result_path:
                    step_paths.append(str(step.result_path))

                completed_steps += 1

            # Execute post-processing
            post_results = self._run_post_process(plan, comparison_dir)

            # Log successful completion
            logger.log_execution(
                status="success",
                steps_completed=completed_steps,
                steps_total=total_steps,
                result_paths=step_paths + [str(comparison_dir)]
            )

            return {
                'type': 'comparison',
                'steps': total_steps,
                'step_results': all_results,
                'step_paths': step_paths,
                'comparison_dir': comparison_dir,
                'post_process_results': post_results,
            }

        except Exception as e:
            # Log partial completion
            logger.log_execution(
                status="partial" if completed_steps > 0 else "failed",
                steps_completed=completed_steps,
                steps_total=total_steps,
                error_message=str(e),
                result_paths=step_paths if step_paths else None
            )
            raise

    def _run_post_process(self, plan: ExecutionPlan, output_dir: Path) -> List[Dict[str, Any]]:
        """
        Execute post-processing hooks.

        Currently supports:
        - merge_plot: Merge results from multiple steps into one comparison plot
        """
        from ..modules.outputs.storage import ResultStorage
        from ..modules.outputs.comparison import ResultComparison

        results = []

        for pp in plan.post_process or []:
            pp_type = pp.get('type')

            if pp_type == 'merge_plot':
                # Load results from specified steps
                source_indices = pp.get('sources', list(range(len(plan.steps))))
                labels = pp.get('labels', [plan.steps[i].label for i in source_indices])
                output_file = pp.get('output', 'comparison.png')
                metric = pp.get('metric', 'Q_Y')

                # Load results
                results_list = []
                for idx in source_indices:
                    step = plan.steps[idx]
                    if step.result_path:
                        data = ResultStorage.load(step.result_path)
                        results_list.append(data.get('results', {}))

                if results_list:
                    # Generate comparison plot
                    comparison = ResultComparison(output_dir)
                    plot_path = comparison.plot_qy_comparison(
                        results_list,
                        labels,
                        title=f"{metric} Comparison",
                        filename=output_file,
                        metric=metric,
                    )
                    results.append({
                        'type': 'merge_plot',
                        'path': plot_path,
                        'labels': labels,
                        'metric': metric,
                    })

        return results

    def _build_config_from_dict(self, config_dict: Dict[str, Any]) -> Config:
        """Build a Config object from a dictionary."""
        return Config.from_dict(config_dict)


def run_plan(plan: ExecutionPlan, base_config: Optional[Config] = None) -> Dict[str, Any]:
    """
    Convenience function to execute a plan.

    Args:
        plan: ExecutionPlan to execute
        base_config: Base config (required for multi-step plans)

    Returns:
        Execution results
    """
    executor = PlanExecutor()
    return executor.run(plan, base_config)
