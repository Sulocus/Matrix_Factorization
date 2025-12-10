#!/usr/bin/env python3
"""
SMF CLI - Command Line Interface

Usage:
    smf              # Interactive mode with natural language support
    smf run          # Run experiment with wizard
    smf run -c FILE  # Run from config file
    smf resume       # Resume from last checkpoint
    smf log          # View background run log
    smf vis          # Result browser
    smf test         # Quick test run with defaults
    smf plot merge   # Merge multiple results into comparison plot
"""

import argparse
import sys
import os
import glob
import subprocess
from pathlib import Path

# Global language setting (set once at startup)
_LANG = 'cn'


def _select_language() -> str:
    """Show language selection at startup."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel(
        "[bold]Language / 语言选择[/bold]\n\n"
        "[yellow][1][/yellow] 中文\n"
        "[yellow][2][/yellow] English",
        title="SMF",
        border_style="cyan"
    ))

    try:
        choice = input("Select / 选择 (默认: [1]): ").strip()
        if choice == '2':
            return 'en'
    except (KeyboardInterrupt, EOFError):
        pass
    return 'cn'


def main():
    """Main CLI entry point."""
    global _LANG

    # Language selection at startup (only once)
    _LANG = _select_language()

    parser = argparse.ArgumentParser(
        prog='smf',
        description='Sparse Matrix Factorization Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    smf              # Interactive menu
    smf run          # Run with wizard
    smf run --bg     # Run in background
    smf resume       # Resume from checkpoint
    smf log          # View latest background log
    smf vis          # Browse results
    smf test         # Quick test run
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # smf vis
    vis_parser = subparsers.add_parser('vis', help='Browse experiment results')
    vis_parser.add_argument('--filter', '-f', help='AI filter query')

    # smf test
    test_parser = subparsers.add_parser('test', help='Quick test run with defaults')
    test_parser.add_argument('--steps', '-s', type=int, default=500,
                             help='Max steps (default: 500)')

    # smf run
    run_parser = subparsers.add_parser('run', help='Run experiment')
    run_parser.add_argument('--config', '-c', type=Path,
                            help='Config YAML file')
    run_parser.add_argument('--estimate', '-e', action='store_true',
                            help='Show time estimate only, do not run')
    run_parser.add_argument('--bg', action='store_true',
                            help='Run in background (nohup)')

    # smf resume - Resume from checkpoint
    resume_parser = subparsers.add_parser('resume', help='Resume from last checkpoint')
    resume_parser.add_argument('--config', '-c', type=Path,
                               help='Config file (optional, auto-detect if not provided)')

    # smf log - View logs
    log_parser = subparsers.add_parser('log', help='View logs')
    log_subparsers = log_parser.add_subparsers(dest='log_command', help='Log commands')

    # smf log (default) - View background run log
    log_parser.add_argument('--follow', '-f', action='store_true',
                            help='Follow background log in real-time (tail -f)')

    # smf log llm - View LLM conversation logs
    llm_log_parser = log_subparsers.add_parser('llm', help='View LLM conversation logs')
    llm_log_parser.add_argument('session', nargs='?', default=None,
                                help='Session ID to view (default: list recent sessions)')
    llm_log_parser.add_argument('-n', '--limit', type=int, default=10,
                                help='Number of recent sessions to show (default: 10)')

    # smf plot - Plot utilities
    plot_parser = subparsers.add_parser('plot', help='Plot utilities')
    plot_subparsers = plot_parser.add_subparsers(dest='plot_command', help='Plot commands')

    # smf plot merge - Merge multiple experiment results into comparison plot
    merge_parser = plot_subparsers.add_parser('merge', help='Merge results into comparison plot')
    merge_parser.add_argument('paths', nargs='+', type=Path,
                              help='Result directories to merge')
    merge_parser.add_argument('--labels', '-l', nargs='+',
                              help='Labels for each result (default: directory names)')
    merge_parser.add_argument('--output', '-o', type=Path, default=Path('comparison.png'),
                              help='Output filename (default: comparison.png)')
    merge_parser.add_argument('--metric', '-m', default='Q_Y',
                              help='Metric to plot (default: Q_Y)')
    merge_parser.add_argument('--title', '-t',
                              help='Plot title (auto-generated if not specified)')

    args = parser.parse_args()

    if args.command is None:
        # No subcommand = interactive mode
        _run_interactive(_LANG)

    elif args.command == 'vis':
        _run_vis(args, _LANG)

    elif args.command == 'test':
        _run_test(args, _LANG)

    elif args.command == 'run':
        _run_experiment(args, _LANG)

    elif args.command == 'resume':
        _run_resume(args, _LANG)

    elif args.command == 'log':
        if args.log_command == 'llm':
            _view_llm_log(args)
        else:
            _view_log(args)

    elif args.command == 'plot':
        if args.plot_command == 'merge':
            _plot_merge(args, _LANG)
        else:
            plot_parser.print_help()


def _run_interactive(lang: str):
    """Interactive menu mode."""
    from smf.ui.menu import MainMenu
    from smf.run import run_new_experiment
    from smf.ui.browser import ResultBrowser

    menu = MainMenu(lang=lang)
    while True:
        choice = menu.show()
        if choice == 'exit':
            menu.print_info("Goodbye!" if lang == 'en' else "再见!")
            break
        elif choice == 'run':
            run_new_experiment(menu, lang=lang)
        elif choice == 'browse':
            browser = ResultBrowser(lang=lang)
            browser.browse()


def _run_vis(args, lang: str):
    """Result browser mode."""
    from smf.ui.browser import ResultBrowser

    browser = ResultBrowser(lang=lang)
    if args.filter:
        # Direct AI filter
        browser.refresh()
        from smf.core.llm_filter import filter_with_llm
        filtered, explanation = filter_with_llm(args.filter, browser.results)
        print(f"AI: {explanation}")
        print(f"{'Found' if lang == 'en' else '找到'} {len(filtered)} {'results' if lang == 'en' else '结果'}")
        browser._list_results(filtered)
        if filtered:
            browser._view_from_list(filtered)
    else:
        browser.browse()


def _run_test(args, lang: str):
    """Quick test run."""
    from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig
    from smf.runner import run_experiment

    if lang == 'en':
        print("Quick test mode - using default parameters")
        print(f"Matrix: 200×200, M=50")
        print(f"Steps: {args.steps}")
    else:
        print("快速测试模式 - 使用默认参数")
        print(f"矩阵: 200×200, M=50")
        print(f"步数: {args.steps}")
    print()

    config = Config(
        matrix=MatrixConfig(N1=200, N2=200, M=50),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.1),
        training=TrainingConfig(max_steps=args.steps, samples_per_alpha=1, seed=42),
        algorithm_key="bigamp",
        graph_key="random",
        teacher_key="standard",
    )

    result = run_experiment(config)
    if lang == 'en':
        print(f"\nComplete! Results: {result['result_path']}")
    else:
        print(f"\n完成！结果: {result['result_path']}")


def _run_experiment(args, lang: str):
    """Run experiment from config or wizard."""
    if args.config:
        from smf.core.config import Config
        from smf.runner import run_experiment

        config = Config.from_yaml(args.config)

        if args.estimate:
            _show_estimate(config, lang)
        elif args.bg:
            _run_in_background(config, lang)
        else:
            result = run_experiment(config)
            if result and result.get('result_path'):
                _ask_open_result(Path(result['result_path']), lang)
    else:
        from smf.ui.wizard import ConfigWizard
        from smf.runner import run_experiment
        from smf.experiments.large_matrix_sweep import SizeScalingExperiment
        from smf.experiments.init_scale import InitScaleExperiment

        # Pass lang to wizard (skip internal language selection)
        wizard = ConfigWizard(lang=lang)
        result = wizard.run()

        if result is None:
            print("Cancelled." if lang == 'en' else "已取消。")
            return

        exp_type = result.get('type', 'standard')

        if args.estimate and exp_type == 'standard':
            _show_estimate(result['config'], lang)
            return

        # Ask foreground or background
        run_bg = args.bg
        if not run_bg and exp_type == 'standard':
            prompt = "Run in background? (continues after disconnect) [y/N]: " if lang == 'en' else "后台运行？(断开连接后继续) [y/N]: "
            choice = input(prompt).strip().lower()
            run_bg = choice in ('y', 'yes', '是')

        if exp_type == 'standard':
            if run_bg:
                _run_in_background(result['config'], lang)
            else:
                exp_result = run_experiment(result['config'])
                if exp_result and exp_result.get('result_path'):
                    _ask_open_result(Path(exp_result['result_path']), lang)
        elif exp_type == 'comparison':
            # Multi-step comparison experiment
            from smf.core.plan_executor import PlanExecutor
            from smf.core.execution_plan import ExecutionPlan, ExecutionStep

            # Build ExecutionPlan from wizard result
            steps = [
                ExecutionStep(
                    config_dict=step['config'],
                    label=step['label']
                )
                for step in result.get('comparison_steps', [])
            ]

            plan = ExecutionPlan(
                teacher=result.get('teacher'),
                graph=result.get('graph'),
                algorithm=result.get('algorithm'),
                matrix_info=result.get('matrix_info', ''),
                alpha_info=result.get('alpha_info', ''),
                alpha_count=result.get('alpha_count', 0),
                metrics=result.get('metrics', ['Q_Y']),
                plots=[],
                steps=steps,
                post_process=result.get('post_process', []),
            )

            executor = PlanExecutor()
            exp_result = executor.run(plan, result['config'])

            # Show comparison results
            if exp_result and exp_result.get('comparison_dir'):
                _ask_open_result(Path(exp_result['comparison_dir']), lang)
        elif exp_type == 'size_scaling':
            experiment = SizeScalingExperiment(
                matrix_configs=result['matrix_configs'],
                alpha_start=result['alpha_start'],
                alpha_stop=result['alpha_stop'],
                alpha_step=result['alpha_step'],
                max_steps=result['max_steps'],
                samples=result.get('samples', 1),
            )
            exp_result = experiment.run()
            if exp_result and exp_result.get('result_path'):
                _ask_open_result(Path(exp_result['result_path']), lang)
        elif exp_type == 'init_scale':
            experiment = InitScaleExperiment(
                result['config'],
                result['scale_factors']
            )
            exp_result = experiment.run()
            if exp_result and exp_result.get('result_path'):
                _ask_open_result(Path(exp_result['result_path']), lang)


def _show_estimate(config, lang: str = 'cn'):
    """Show time estimate for config."""
    from smf.core.time_estimator import get_time_estimator

    estimator = get_time_estimator()
    alpha_values = config.alpha.get_values()

    estimator.print_estimate(
        N1=config.matrix.N1,
        N2=config.matrix.N2,
        M=config.matrix.M,
        steps=config.training.max_steps,
        samples=config.training.samples_per_alpha,
        num_alphas=len(alpha_values),
    )


def _view_llm_log(args):
    """View LLM conversation logs."""
    from smf.core.llm_logger import LLMLogger

    if args.session:
        # View specific session
        summary = LLMLogger.format_session_summary(args.session)
        print(summary)
    else:
        # List recent sessions
        sessions = LLMLogger.list_sessions(limit=args.limit)
        if not sessions:
            print("No LLM conversation logs found.")
            print("Logs are created when you use 'smf run' with AI configuration.")
            return

        print(f"Recent LLM sessions (last {len(sessions)}):")
        print("-" * 60)
        for log_file in sessions:
            session_id = log_file.stem.replace("session_", "")
            # Get brief stats
            entries = LLMLogger.load_session(session_id)
            request_count = sum(1 for e in entries if e.stage == "request")
            execution = next((e for e in reversed(entries) if e.stage == "execution"), None)

            status_icon = "✓" if execution and execution.execution_status == "success" else \
                          "⚠" if execution and execution.execution_status == "partial" else \
                          "✗" if execution and execution.execution_status == "failed" else "?"

            print(f"  {status_icon} {session_id}  ({request_count} request(s))")

        print("-" * 60)
        print("View details: smf log llm <session_id>")


def _view_log(args):
    """View background run log."""
    # Find the latest log file
    log_files = sorted(glob.glob("smf_run_*.log"), key=os.path.getmtime, reverse=True)

    if not log_files:
        print("No log files found. Run 'smf run --bg' to start a background experiment.")
        return

    latest_log = log_files[0]
    print(f"Log file: {latest_log}")
    print("-" * 50)

    if args.follow:
        # Real-time follow mode
        print("(Press Ctrl+C to stop)")
        try:
            subprocess.run(["tail", "-f", latest_log])
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        # Show last 50 lines
        subprocess.run(["tail", "-50", latest_log])
        print("-" * 50)
        print(f"Use 'smf log -f' to follow in real-time")


def _plot_merge(args, lang: str):
    """Merge multiple experiment results into a comparison plot."""
    from smf.modules.outputs.storage import ResultStorage
    from smf.modules.outputs.comparison import ResultComparison
    from rich.console import Console

    console = Console()

    # Validate paths
    valid_paths = []
    for p in args.paths:
        if not p.exists():
            console.print(f"[red]{'路径不存在' if lang == 'cn' else 'Path not found'}: {p}[/red]")
            continue
        # Check if it's a result directory (has results.json)
        result_file = p / "results.json"
        if not result_file.exists():
            console.print(f"[red]{'不是有效的结果目录' if lang == 'cn' else 'Not a valid result directory'}: {p}[/red]")
            continue
        valid_paths.append(p)

    if len(valid_paths) < 2:
        console.print(f"[red]{'至少需要2个有效的结果目录' if lang == 'cn' else 'Need at least 2 valid result directories'}[/red]")
        return

    # Generate labels if not provided
    labels = args.labels
    if not labels or len(labels) != len(valid_paths):
        labels = [p.name for p in valid_paths]

    # Load results
    console.print(f"{'正在加载结果...' if lang == 'cn' else 'Loading results...'}")
    results_list = []
    for p in valid_paths:
        try:
            data = ResultStorage.load(p)
            results_list.append(data.get('results', {}))
        except Exception as e:
            console.print(f"[red]{'加载失败' if lang == 'cn' else 'Failed to load'} {p}: {e}[/red]")
            return

    # Determine output path
    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    # Generate comparison plot
    console.print(f"{'正在生成对比图...' if lang == 'cn' else 'Generating comparison plot...'}")

    comparison = ResultComparison(output_path.parent)
    title = args.title or f"{args.metric} Comparison"

    try:
        plot_path = comparison.plot_qy_comparison(
            results_list,
            labels,
            title=title,
            filename=output_path.name,
            metric=args.metric,
        )
        console.print(f"[green]{'对比图已保存' if lang == 'cn' else 'Comparison plot saved'}: {plot_path}[/green]")

        # Ask to open
        from smf.core.opener import open_image
        from rich.prompt import Confirm
        prompt = f"{'打开图片？' if lang == 'cn' else 'Open image?'}"
        if Confirm.ask(prompt, default=True):
            open_image(plot_path)

    except Exception as e:
        console.print(f"[red]{'生成图表失败' if lang == 'cn' else 'Failed to generate plot'}: {e}[/red]")


def _run_resume(args, lang: str):
    """Resume from last checkpoint."""
    from smf.runner import run_experiment

    if args.config:
        from smf.core.config import Config
        config = Config.from_yaml(args.config)
    else:
        # Try to find config from checkpoint
        checkpoint_dirs = list(Path(".").glob("**/  .checkpoints"))
        if not checkpoint_dirs:
            msg = "No checkpoints found. Use 'smf run' to start a new experiment." if lang == 'en' else "未找到检查点。请使用 'smf run' 启动新实验。"
            print(msg)
            return

        # Find latest checkpoint
        latest_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
        config_file = latest_dir.parent / "config.yaml"

        if config_file.exists():
            from smf.core.config import Config
            config = Config.from_yaml(config_file)
        else:
            msg = "Cannot find config for checkpoint. Use 'smf resume -c CONFIG_FILE'" if lang == 'en' else "无法找到检查点配置。请使用 'smf resume -c 配置文件'"
            print(msg)
            return

    print("Resuming from checkpoint..." if lang == 'en' else "正在从检查点恢复...")
    run_experiment(config, resume=True)


def _run_in_background(config, wizard_lang='cn'):
    """Run experiment in background with nohup."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"smf_run_{timestamp}.log"
    config_file = f"/tmp/smf_config_{timestamp}.json"

    # Save config
    config.to_json(Path(config_file))

    # Build command
    python_cmd = sys.executable
    cmd = f'nohup {python_cmd} -c "from smf.runner import run_experiment; from smf.core.config import Config; from pathlib import Path; c = Config.from_json(Path(\\\"{config_file}\\\")); run_experiment(c)" > {log_file} 2>&1 &'

    subprocess.Popen(cmd, shell=True)

    print(f"\n{'后台运行已启动' if wizard_lang == 'cn' else 'Background run started'}!")
    print(f"{'日志文件' if wizard_lang == 'cn' else 'Log file'}: {log_file}")
    print(f"{'查看日志' if wizard_lang == 'cn' else 'View log'}: smf log")
    print(f"{'实时跟踪' if wizard_lang == 'cn' else 'Follow'}: smf log -f")


def _ask_open_result(result_path: Path, lang: str):
    """Ask user whether to open result image(s) after experiment completes."""
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from smf.core.opener import open_image

    console = Console()

    if not result_path or not result_path.exists():
        return

    # Find all PNG images
    plots = list(result_path.glob("*.png"))

    # Also check plots/ subdirectory
    plots_dir = result_path / "plots"
    if plots_dir.exists():
        plots.extend(plots_dir.glob("*.png"))

    if not plots:
        return

    console.print()

    if len(plots) == 1:
        # Single image: ask yes/no
        prompt = f"打开 {plots[0].name}？" if lang == 'cn' else f"Open {plots[0].name}?"
        if Confirm.ask(prompt, default=True):
            open_image(plots[0])
    else:
        # Multiple images: let user choose
        msg = f"生成了 {len(plots)} 张图片:" if lang == 'cn' else f"Generated {len(plots)} images:"
        console.print(msg)
        for i, p in enumerate(plots, 1):
            console.print(f"  [{i}] {p.name}")
        quit_label = "不打开" if lang == 'cn' else "Skip"
        console.print(f"  [q] {quit_label}")

        prompt = "打开哪张？(默认: 1)" if lang == 'cn' else "Open which? (default: 1)"
        choice = Prompt.ask(prompt, default="1", show_default=False)

        if choice.lower() == 'q':
            return
        elif choice.isdigit() and 1 <= int(choice) <= len(plots):
            open_image(plots[int(choice) - 1])


if __name__ == '__main__':
    main()
