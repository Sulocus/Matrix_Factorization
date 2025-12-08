#!/usr/bin/env python3
"""
SMF - Sparse Matrix Factorization Framework

Main entry point for interactive experiment running.

Usage:
    python -m smf.run               # Interactive mode
    python smf/run.py               # Same as above
    python smf/run.py --quick       # Quick run with defaults
    python smf/run.py --init-scale  # Init scale sweep experiment
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smf.ui.menu import MainMenu
from smf.ui.wizard import ConfigWizard
from smf.ui.browser import ResultBrowser
from smf.runner import run_experiment
from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig
from smf.modules.outputs.storage import list_results


def main():
    """Main entry point."""
    # Check for command line modes
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            quick_run()
            return
        elif sys.argv[1] in ('--variance', '--init-scale'):
            run_init_scale_cli()
            return
        elif sys.argv[1] == '--browse':
            browser = ResultBrowser()
            browser.browse()
            return

    menu = MainMenu()

    while True:
        choice = menu.show()

        if choice == 'exit':
            menu.print_info("Goodbye!")
            break

        elif choice == 'run':
            run_new_experiment(menu)

        elif choice == 'browse':
            browser = ResultBrowser()
            browser.browse()


def run_new_experiment(menu: MainMenu, lang: str = 'cn'):
    """Run a new experiment with wizard."""
    wizard = ConfigWizard(lang=lang)
    wizard_result = wizard.run()

    if wizard_result is None:
        menu.print_info("Cancelled." if lang == 'en' else "已取消。")
        return

    exp_type = wizard_result.get('type', 'standard')
    msg = f"Starting {exp_type} experiment..." if lang == 'en' else f"正在启动 {exp_type} 实验..."
    menu.print_info(msg)

    try:
        if exp_type == 'standard':
            # Standard single-config experiment
            config = wizard_result['config']
            result = run_experiment(config)
            result_path = result['result_path']

        elif exp_type == 'size_scaling':
            # Size scaling (finite-size effect) experiment
            from smf.experiments.large_matrix_sweep import SizeScalingExperiment

            experiment = SizeScalingExperiment(
                matrix_configs=wizard_result['matrix_configs'],
                alpha_start=wizard_result['alpha_start'],
                alpha_stop=wizard_result['alpha_stop'],
                alpha_step=wizard_result['alpha_step'],
                max_steps=wizard_result['max_steps'],
                samples=wizard_result.get('samples', 1),
            )
            experiment.run()
            result_path = experiment.output_dir

        elif exp_type == 'init_scale':
            # Init scale sweep experiment
            from smf.experiments.init_scale import InitScaleExperiment

            config = wizard_result['config']
            scale_factors = wizard_result['scale_factors']

            experiment = InitScaleExperiment(config, scale_factors)
            experiment.run()
            result_path = experiment.output_dir

        elif exp_type == 'comparison':
            # Multi-step comparison experiment
            from smf.core.plan_executor import PlanExecutor
            from smf.core.execution_plan import build_execution_plan_from_dict

            # Build ExecutionPlan from config dict (properly creates ModuleCall objects)
            config = wizard_result['config']
            config_dict = {
                'algorithm_key': config.algorithm_key,
                'graph_key': config.graph_key,
                'teacher_key': config.teacher_key,
                'N1': config.matrix.N1,
                'N2': config.matrix.N2,
                'M': config.matrix.M,
                'alpha_start': config.alpha.start,
                'alpha_stop': config.alpha.stop,
                'alpha_step': config.alpha.step,
                'max_steps': config.training.max_steps,
                'samples_per_alpha': config.training.samples_per_alpha,
            }

            execution_params = {
                'metrics_to_compute': list(config.execution.metrics_to_compute),
                'plots': config.execution.plots,
                'include_summary_plot': config.execution.include_summary_plot,
                'include_qy_plot': config.execution.include_qy_plot,
            }

            plan = build_execution_plan_from_dict(
                config_dict,
                execution_params=execution_params,
                comparison_steps=wizard_result.get('comparison_steps', []),
                post_process=wizard_result.get('post_process', []),
            )

            executor = PlanExecutor()
            exp_result = executor.run(plan, config)
            result_path = exp_result.get('comparison_dir')

        else:
            menu.print_error(f"Unknown experiment type: {exp_type}")
            return

        menu.print_success(f"Experiment complete! Results saved to: {result_path}")

        # Offer to open result
        if menu.confirm("Open result image?", default=True):
            from smf.core.opener import open_image
            # Try common plot locations
            for plot_name in ["size_scaling_comparison.png", "summary.png", "init_scale_comparison.png"]:
                plot_path = result_path / plot_name
                if plot_path.exists():
                    open_image(plot_path)
                    break
            else:
                # Try plots subdirectory
                plots_dir = result_path / "plots"
                if plots_dir.exists():
                    summary = plots_dir / "summary.png"
                    if summary.exists():
                        open_image(summary)

    except Exception as e:
        menu.print_error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


def quick_run():
    """Quick run with default settings."""
    print("Quick run mode - using default parameters")

    config = Config(
        matrix=MatrixConfig(N1=200, N2=200, M=50),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.1),
        training=TrainingConfig(max_steps=500, samples_per_alpha=1, seed=42),
        algorithm_key="bigamp",
        graph_key="random",
        teacher_key="standard",
    )

    print(f"Config: {config.matrix.N1}×{config.matrix.N2}, M={config.matrix.M}")
    print(f"Steps: {config.training.max_steps} (quick test)")

    result = run_experiment(config)
    print(f"\nComplete! Results: {result['result_path']}")


def run_init_scale_cli():
    """Run init scale sweep experiment from command line."""
    from smf.experiments.init_scale import run_init_scale

    print("Init Scale Sweep Experiment")
    print("=" * 50)

    # Default parameters
    N1 = 200
    N2 = 200
    M = 50
    scale_factors = [0.5, 1.0, 1.5, 2.0]

    print(f"Matrix: {N1}×{N2}, M={M}")
    print(f"Scale factors (k in k/√M): {scale_factors}")

    run_init_scale(
        N1=N1, N2=N2, M=M,
        scale_factors=scale_factors,
        max_steps=1000,
    )


if __name__ == '__main__':
    main()
