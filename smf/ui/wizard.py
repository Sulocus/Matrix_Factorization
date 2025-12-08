"""
Configuration wizard for experiment setup.

Flow design:
1. Select language (EN/CN)
2. Choose EXPERIMENT TYPE (not algorithm!)
3. Configure based on the type:
   - Standard: single (N, M), full alpha range
   - Size Scaling: multiple N values, narrow alpha range
   - Init Scale: single (N, M), multiple k values
   - Custom: full manual configuration
4. Option to run in background (nohup)
"""

from typing import Optional, List, Tuple, Dict, TypedDict
from pathlib import Path
import sys
import os
import subprocess

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
    from rich.table import Table
    from rich.status import Status
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    box = None

from ..core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, AlgorithmConfig, ExecutionConfig
from ..modules.registry import list_algorithms, list_graphs, list_teachers
from ..core.llm_advisor import get_config_advisor, AnalysisResult
from ..core.execution_plan import build_execution_plan_from_dict, ExecutionPlan
from .theme import format_menu_item, THEME


# ============================================================
# Type Definitions (Defensive Principle 4: Use type annotations)
# ============================================================
class ConfigItem(TypedDict):
    """Configurable item structure for dynamic module display."""
    key: str           # Module identifier: teacher, graph, algorithm, etc.
    label: str         # Display label (bilingual)
    value: str         # Current config value
    is_special: bool   # Non-default option (needs ★ marker)
    edit_key: str      # Dynamic index ("1", "2", ...)


# ============================================================
# Default values (used for is_special detection)
# ============================================================
DEFAULTS = {
    'algorithm_key': 'bigamp',
    'graph_key': 'random',
    'teacher_key': 'standard',
}


# Bilingual text dictionary
TEXTS = {
    'en': {
        'select_language': 'Select Language',
        'step1_title': 'Step 1: Choose Experiment Type',
        'step1_desc': 'Different experiments have different configuration flows',
        'ai_config': 'AI Smart Config',
        'ai_config_desc': 'Describe your needs in natural language',
        'standard': 'Standard',
        'standard_desc': 'Single (N, M), Q_Y vs α curve (baseline)',
        'size_scaling': 'Size Scaling',
        'size_scaling_desc': 'Multiple N values, finite-size effect study',
        'init_scale': 'Init Scale',
        'init_scale_desc': 'Different k/√M initialization comparison',
        'custom': 'Custom',
        'custom_desc': 'Full manual configuration (advanced)',
        'select': 'Select (default: 0)',
        'confirm': 'Confirm?',
        'matrix_dim': 'Matrix Dimensions',
        'alpha_range': 'Alpha Scan Range',
        'training_params': 'Training Parameters',
        'steps': 'Max steps',
        'samples': 'Samples per alpha',
        'config_summary': 'Configuration Summary',
        'background_prompt': 'Run in background (continues after disconnect)?',
        'background_started': 'Started in background. Log file:',
        'ai_title': 'AI Smart Configuration',
        'ai_desc': 'Describe what experiment you want in natural language',
        'ai_examples': 'Examples:',
        'describe': 'Describe your experiment',
        'analyzing': 'Analyzing...',
        'ai_understanding': 'AI Understanding',
        'confirm_or_modify': 'Confirm or modify',
        'no_input': 'No input provided, using default',
    },
    'cn': {
        'select_language': '选择语言',
        'step1_title': '第一步：选择实验类型',
        'step1_desc': '不同的实验类型有不同的配置流程',
        'ai_config': 'AI 智能配置',
        'ai_config_desc': '用自然语言描述你的需求',
        'standard': '标准实验',
        'standard_desc': '单个 (N, M)，Q_Y vs α 相变曲线',
        'size_scaling': '尺寸缩放',
        'size_scaling_desc': '多个 N 值，有限尺寸效应研究',
        'init_scale': '初始化缩放',
        'init_scale_desc': '不同 k/√M 初始化比较',
        'custom': '自定义',
        'custom_desc': '完整手动配置（高级）',
        'select': '选择 (默认: 0)',
        'confirm': '确认？',
        'matrix_dim': '矩阵维度',
        'alpha_range': 'Alpha 扫描范围',
        'training_params': '训练参数',
        'steps': '最大步数',
        'samples': '每个 alpha 样本数',
        'config_summary': '配置概要',
        'background_prompt': '后台运行（断开连接后继续）？',
        'background_started': '已启动后台运行。日志文件：',
        'ai_title': 'AI 智能配置',
        'ai_desc': '用自然语言描述你想要的实验',
        'ai_examples': '示例：',
        'describe': '描述你的实验',
        'analyzing': '分析中...',
        'ai_understanding': 'AI 理解',
        'confirm_or_modify': '确认或修改',
        'no_input': '未提供输入，使用默认配置',
    }
}


class ConfigWizard:
    """Interactive configuration wizard with language support."""

    def __init__(self, lang: str = None):
        """
        Initialize wizard.

        Args:
            lang: Language code ('cn' or 'en'). If provided, skip language selection.
        """
        self.console = Console() if RICH_AVAILABLE else None
        self.lang = lang  # None means ask user, otherwise use provided lang

    def t(self, key: str) -> str:
        """Get translated text."""
        return TEXTS.get(self.lang, TEXTS['en']).get(key, key)

    def run(self) -> Optional[dict]:
        """
        Run the configuration wizard.

        Returns:
            dict with:
            - 'type': experiment type ('standard', 'size_scaling', 'init_scale')
            - 'config': Config object (for standard)
            - other type-specific parameters
            Or None if cancelled
        """
        if RICH_AVAILABLE:
            return self._run_rich()
        else:
            return self._run_simple()

    def _run_rich(self) -> Optional[dict]:
        """Rich-based wizard."""
        self.console.print()

        # Language already set from CLI, no need to ask again
        # Default to 'cn' if not set (shouldn't happen when called from CLI)
        if self.lang is None:
            self.lang = 'cn'

        # Step 1: Choose experiment TYPE first!
        self.console.print(Panel(
            f"[bold]{self.t('step1_title')}[/bold]\n"
            f"[dim]{self.t('step1_desc')}[/dim]",
            border_style="cyan"
        ))

        table = Table(show_header=False, box=None, padding=(0, 1))
        # AI option first
        table.add_row(
            f"[{THEME['option_number']}][0][/{THEME['option_number']}]",
            f"[{THEME['ai_title']}]{self.t('ai_config')}[/{THEME['ai_title']}]",
            f"[{THEME['option_desc']}]{self.t('ai_config_desc')}[/{THEME['option_desc']}]"
        )
        table.add_row(
            f"[{THEME['option_number']}][1][/{THEME['option_number']}]",
            f"[{THEME['option_title']}]{self.t('standard')}[/{THEME['option_title']}]",
            f"[{THEME['option_desc']}]{self.t('standard_desc')}[/{THEME['option_desc']}]"
        )
        table.add_row(
            f"[{THEME['option_number']}][2][/{THEME['option_number']}]",
            f"[{THEME['option_title']}]{self.t('size_scaling')}[/{THEME['option_title']}]",
            f"[{THEME['option_desc']}]{self.t('size_scaling_desc')}[/{THEME['option_desc']}]"
        )
        table.add_row(
            f"[{THEME['option_number']}][3][/{THEME['option_number']}]",
            f"[{THEME['option_title']}]{self.t('init_scale')}[/{THEME['option_title']}]",
            f"[{THEME['option_desc']}]{self.t('init_scale_desc')}[/{THEME['option_desc']}]"
        )
        table.add_row(
            f"[{THEME['option_number']}][4][/{THEME['option_number']}]",
            f"[{THEME['option_title']}]{self.t('custom')}[/{THEME['option_title']}]",
            f"[{THEME['option_desc']}]{self.t('custom_desc')}[/{THEME['option_desc']}]"
        )
        self.console.print(table)

        exp_type = Prompt.ask(self.t('select'), choices=["0", "1", "2", "3", "4"], default="0", show_choices=False, show_default=False)
        self.console.print()

        if exp_type == "0":
            return self._configure_with_ai()
        elif exp_type == "1":
            return self._configure_standard()
        elif exp_type == "2":
            return self._configure_size_scaling()
        elif exp_type == "3":
            return self._configure_init_scale()
        elif exp_type == "4":
            return self._configure_custom()

        return None

    def _configure_with_ai(self) -> Optional[dict]:
        """AI-powered configuration via natural language (unified UI)."""
        self.console.print(Panel(
            f"[{THEME['ai_title']}]AI Smart Configuration[/{THEME['ai_title']}]\n"
            "[dim]Describe what experiment you want in natural language[/dim]",
            border_style="magenta"
        ))

        self.console.print("[dim]Examples:[/dim]")
        self.console.print("  - 大矩阵的基准相变实验，N=5000")
        self.console.print("  - 对比不同尺寸的有限尺寸效应")
        self.console.print("  - 快速测试，200x200，500步")
        self.console.print()

        user_input = Prompt.ask("Describe your experiment")

        if not user_input.strip():
            self.console.print("[yellow]No input provided, using default[/yellow]")
            return self._configure_standard()

        # Analyze with AI - using dynamic spinner with API mode indicator
        self.console.print()
        advisor = get_config_advisor()

        from ..core.llm_filter import GeminiClient
        api_mode = GeminiClient.get_api_mode(self.lang)
        spinner_text = f'AI 正在理解你的需求 {api_mode}' if self.lang == 'cn' else f'AI analyzing {api_mode}'

        with Status(
            f"[bold cyan]{spinner_text}[/bold cyan]",
            spinner="dots",
            console=self.console
        ):
            result = advisor.analyze_request(user_input)

        # Show switch messages if any API/model switches occurred
        if result.switch_messages:
            import time
            for msg in result.switch_messages:
                self.console.print(f"[yellow]⟳ {msg}[/yellow]")
                time.sleep(0.3)  # Brief animation effect

        # Unified configuration loop
        while True:
            self.console.print()
            self._print_unified_config(result)

            # Dynamic prompt based on available items
            items = getattr(self, '_current_module_items', [])
            max_num = len(items) if items else 4

            prompt_text = '修改序号 / y / n / 或直接输入补充说明' if self.lang == 'cn' else 'Modify # / y / n / or enter clarification'
            choice = Prompt.ask(prompt_text, default="y")

            if choice.lower() == "y":
                return self._finalize_ai_config(result)
            elif choice.lower() == "n":
                cancel_msg = '已取消配置' if self.lang == 'cn' else 'Configuration cancelled'
                self.console.print(f"[yellow]{cancel_msg}[/yellow]")
                return None
            elif choice.isdigit():
                num = int(choice)
                if 1 <= num <= max_num:
                    result = self._modify_config_item(result, num)
                else:
                    msg = f"无效序号，有效范围: 1-{max_num}" if self.lang == 'cn' else f"Invalid number, valid range: 1-{max_num}"
                    self.console.print(f"[yellow]{msg}[/yellow]")
            else:
                # Any other input is treated as clarification text directly
                clarification = choice.strip()
                if not clarification:
                    continue
                api_mode = GeminiClient.get_api_mode(self.lang)
                spinner_text = f'AI 正在重新理解 {api_mode}' if self.lang == 'cn' else f'AI re-analyzing {api_mode}'
                with Status(
                    f"[bold cyan]{spinner_text}[/bold cyan]",
                    spinner="dots",
                    console=self.console
                ):
                    result = advisor.analyze_with_clarification(
                        user_input, clarification, result
                    )
                # Show switch messages if any API/model switches occurred
                if result.switch_messages:
                    import time
                    for msg in result.switch_messages:
                        self.console.print(f"[yellow]⟳ {msg}[/yellow]")
                        time.sleep(0.3)

    def _build_module_list(self, result: AnalysisResult) -> List[ConfigItem]:
        """
        Build dynamic module display list from ExecutionPlan.

        Uses ExecutionPlan to ensure UI display matches actual execution.
        This is pure data layer, returns ConfigItem list without any UI rendering logic.
        """
        # Build ExecutionPlan from config dict (including comparison data)
        plan = build_execution_plan_from_dict(
            result.config,
            result.execution_params,
            comparison_steps=result.comparison_steps,
            post_process=result.post_process,
        )

        # Convert to ConfigItem list using ExecutionPlan's method
        lang = self.lang or 'cn'
        plan_items = plan.to_display_list(lang)

        # Convert to ConfigItem TypedDict format
        items: List[ConfigItem] = []
        for item in plan_items:
            items.append(ConfigItem(
                key=item['key'],
                label=item['label'],
                value=item['value'],
                is_special=item['is_special'],
                edit_key=item['edit_key'],
            ))

        return items

    def _print_unified_config(self, result: AnalysisResult):
        """
        Display simplified configuration interface (Defensive Principle 2: Separate data from view).

        This is pure view layer, only renders, data from _build_module_list().
        TODO(Phase A2): Move to ConfigRenderer class
        """
        # 1. Get data (data layer)
        module_items = self._build_module_list(result)
        self._current_module_items = module_items  # Save for modification

        # 2. Render understanding text
        self.console.print(Panel(
            f"[bold green]{result.understanding}[/bold green]",
            border_style="green"
        ))

        # 3. Render module table
        table = Table(box=box.ROUNDED, show_header=False, expand=False)
        table.add_column("序号", style="yellow", width=4)
        table.add_column("模块", style="cyan", width=10)
        table.add_column("配置", style="white")

        for item in module_items:
            value_display = item['value']
            if item['is_special']:
                value_display = f"{item['value']} [bold magenta]★[/bold magenta]"

            table.add_row(f"[{item['edit_key']}]", item['label'], value_display)

        self.console.print(table)

        # 4. Render legend
        if any(item['is_special'] for item in module_items):
            legend = '★ = 非默认选项' if self.lang == 'cn' else '★ = non-default'
            self.console.print(f"[dim]{legend}[/dim]")

    def _modify_config_item(self, result: AnalysisResult, item_number: int) -> AnalysisResult:
        """
        Modify a specific config item by dynamic number.

        Uses _current_module_items to map number to actual module key.
        """
        items = getattr(self, '_current_module_items', [])
        if item_number < 1 or item_number > len(items):
            max_num = len(items) if items else 5
            msg = f"无效序号，有效范围: 1-{max_num}" if self.lang == 'cn' else f"Invalid number, valid range: 1-{max_num}"
            self.console.print(f"[yellow]{msg}[/yellow]")
            return result

        item = items[item_number - 1]
        c = result.config
        module_key = item['key']

        if module_key == 'algorithm':
            c['algorithm_key'] = self._select_algorithm()
            step_prompt = '步数' if self.lang == 'cn' else 'Steps'
            c['max_steps'] = IntPrompt.ask(step_prompt, default=c.get('max_steps', 5000))
        elif module_key == 'matrix':
            c['N1'] = IntPrompt.ask("N1", default=c.get('N1', 200))
            c['N2'] = IntPrompt.ask("N2", default=c.get('N2', c['N1']))
            c['M'] = IntPrompt.ask("M", default=c.get('M', 50))
        elif module_key == 'alpha':
            start_prompt = '起始' if self.lang == 'cn' else 'Start'
            stop_prompt = '终止' if self.lang == 'cn' else 'End'
            step_prompt = '步长' if self.lang == 'cn' else 'Step'
            c['alpha_start'] = FloatPrompt.ask(start_prompt, default=c.get('alpha_start', 0.0))
            c['alpha_stop'] = FloatPrompt.ask(stop_prompt, default=c.get('alpha_stop', 4.0))
            c['alpha_step'] = FloatPrompt.ask(step_prompt, default=c.get('alpha_step', 0.1))
        elif module_key == 'teacher':
            c['teacher_key'] = self._select_teacher()
        elif module_key == 'graph':
            c['graph_key'] = self._select_graph()
        elif module_key == 'algo_params':
            # Algorithm parameters: damping, samples_per_alpha
            damping_prompt = 'Damping (0.5~0.95)' if self.lang == 'cn' else 'Damping (0.5~0.95)'
            c['damping'] = FloatPrompt.ask(damping_prompt, default=c.get('damping', 0.5))
            trials_prompt = '每个α采样次数' if self.lang == 'cn' else 'Samples per alpha'
            c['samples_per_alpha'] = IntPrompt.ask(trials_prompt, default=c.get('samples_per_alpha', 1))
        elif module_key == 'metrics':
            new_metrics = self._select_metrics()
            c['metrics'] = new_metrics
            # Also update execution_params to keep in sync
            if result.execution_params is None:
                result.execution_params = {}
            result.execution_params['metrics_to_compute'] = new_metrics
        elif module_key == 'outputs':
            # TODO: Implement outputs modification UI
            msg = '输出配置修改功能暂未实现' if self.lang == 'cn' else 'Outputs modification not implemented yet'
            self.console.print(f"[yellow]{msg}[/yellow]")

        return result

    def _select_teacher(self) -> str:
        """
        Select teacher interactively.

        Defensive Principle 1: Get from registry dynamically.
        """
        teachers = list_teachers()
        if not teachers:
            return DEFAULTS['teacher_key']

        table = Table(show_header=False, box=None)
        for i, t in enumerate(teachers, 1):
            default_marker = ""
            if t.key == DEFAULTS['teacher_key']:
                default_marker = f" [dim]({'默认' if self.lang == 'cn' else 'default'})[/dim]"
            table.add_row(
                f"[yellow][{i}][/yellow]",
                t.name,
                f"[dim]{t.description}[/dim]{default_marker}"
            )
        self.console.print(table)

        prompt = '选择' if self.lang == 'cn' else 'Select'
        choice = IntPrompt.ask(prompt, default=1)
        choice = max(1, min(choice, len(teachers)))
        return teachers[choice - 1].key

    def _select_graph(self) -> str:
        """
        Select graph generation method interactively.

        Defensive Principle 1: Get from registry dynamically, not hardcoded list.
        """
        graphs = list_graphs()
        if not graphs:
            return DEFAULTS['graph_key']

        table = Table(show_header=False, box=None)
        for i, g in enumerate(graphs, 1):
            default_marker = ""
            if g.key == DEFAULTS['graph_key']:
                default_marker = f" [dim]({'默认' if self.lang == 'cn' else 'default'})[/dim]"
            table.add_row(
                f"[yellow][{i}][/yellow]",
                g.name,
                f"[dim]{g.description}[/dim]{default_marker}"
            )
        self.console.print(table)

        prompt = '选择' if self.lang == 'cn' else 'Select'
        choice = IntPrompt.ask(prompt, default=1)
        choice = max(1, min(choice, len(graphs)))
        return graphs[choice - 1].key

    def _select_metrics(self) -> List[str]:
        """Select metrics interactively."""
        self.console.print("[dim]可用指标: Q_Y, Q_W, Q_X, Q_Y_unobserved, Q_Y_observed[/dim]")
        metrics_str = Prompt.ask("输入指标 (逗号分隔)", default="Q_Y")
        return [m.strip() for m in metrics_str.split(",") if m.strip()]

    def _print_ai_config_summary(self, config: Dict):
        """Print AI-generated config summary."""
        self.console.print(Panel("[bold]Generated Configuration[/bold]", border_style="cyan"))
        table = Table(show_header=False, box=None)
        table.add_row("Algorithm", config.get('algorithm_key', 'bigamp'))
        table.add_row("Graph", config.get('graph_key', 'random'))
        table.add_row("Teacher", config.get('teacher_key', 'standard'))
        table.add_row("Matrix", f"{config.get('N1', 200)}×{config.get('N2', 200)}, M={config.get('M', 50)}")
        table.add_row("Alpha", f"{config.get('alpha_start', 0.0)} ~ {config.get('alpha_stop', 4.0)}, step {config.get('alpha_step', 0.1)}")
        table.add_row("Training", f"steps={config.get('max_steps', 5000)}, samples={config.get('samples_per_alpha', 1)}")
        # Show resample_mask with clear explanation
        resample = config.get('resample_mask', True)
        resample_label = "Resample Mask" if self.lang == 'en' else "每次重采样图"
        resample_value = f"{'✓' if resample else '✗'} ({'不同图' if resample else '相同图'})"
        table.add_row(resample_label, resample_value)
        self.console.print(table)

    def _finalize_ai_config(self, result: AnalysisResult) -> Optional[dict]:
        """Finalize AI-generated configuration into proper Config object."""
        c = result.config

        # Extract execution params from LLM output
        exec_params = result.execution_params or {}
        execution = ExecutionConfig(
            metrics_to_compute=exec_params.get('metrics_to_compute',
                ['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error']),
            plots=exec_params.get('plots', []),
            include_summary_plot=exec_params.get('include_summary_plot', True),
            include_qy_plot=exec_params.get('include_qy_plot', True),
        )

        config = Config(
            matrix=MatrixConfig(
                N1=c.get('N1', 200),
                N2=c.get('N2', 200),
                M=c.get('M', 50)
            ),
            alpha=AlphaConfig(
                start=c.get('alpha_start', 0.0),
                stop=c.get('alpha_stop', 4.0),
                step=c.get('alpha_step', 0.1)
            ),
            training=TrainingConfig(
                max_steps=c.get('max_steps', 5000),
                samples_per_alpha=c.get('samples_per_alpha', 1),
                resample_mask=c.get('resample_mask', True)
            ),
            algorithm_key=c.get('algorithm_key', 'bigamp'),
            graph_key=c.get('graph_key', 'random'),
            teacher_key=c.get('teacher_key', 'standard'),
            execution=execution,  # Pass LLM-generated execution params!
        )

        # Handle comparison experiments - pass through comparison data
        if result.experiment_type == 'comparison' and result.comparison_steps:
            return {
                'type': 'comparison',
                'config': config,  # Base config (shared params)
                'comparison_steps': result.comparison_steps,
                'post_process': result.post_process or [],
            }
        else:
            return {'type': 'standard', 'config': config}

    def _select_algorithm(self) -> str:
        """Select algorithm interactively."""
        algorithms = list_algorithms()
        if not algorithms:
            return "bigamp"

        table = Table(show_header=False, box=None)
        for i, alg in enumerate(algorithms, 1):
            table.add_row(f"[yellow][{i}][/yellow]", alg.name, f"[dim]{alg.description}[/dim]")
        self.console.print(table)

        choice = IntPrompt.ask("Select", default=1)
        choice = max(1, min(choice, len(algorithms)))
        return algorithms[choice - 1].key

    def _configure_standard(self) -> Optional[dict]:
        """Configure standard single-config experiment."""
        self.console.print(Panel(
            "[bold]Standard Experiment Configuration[/bold]\n"
            "[dim]Single (N, M) configuration, full α range[/dim]",
            border_style="blue"
        ))

        # Matrix dimensions
        self.console.print("[bold]Matrix Dimensions:[/bold]")
        N1 = IntPrompt.ask("  N1 (rows)", default=200)
        N2 = IntPrompt.ask("  N2 (columns)", default=N1)
        M = IntPrompt.ask("  M (latent dimension)", default=50)
        self.console.print()

        # Alpha range
        self.console.print("[bold]Alpha Scan Range:[/bold]")
        alpha_start = FloatPrompt.ask("  Start", default=0.0)
        alpha_stop = FloatPrompt.ask("  End", default=4.0)
        alpha_step = FloatPrompt.ask("  Step", default=0.1)
        self.console.print()

        # Training
        self.console.print("[bold]Training Parameters:[/bold]")
        max_steps = IntPrompt.ask("  Max steps", default=5000)
        samples = IntPrompt.ask("  Samples per alpha", default=1)
        self.console.print()

        config = Config(
            matrix=MatrixConfig(N1=N1, N2=N2, M=M),
            alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
            training=TrainingConfig(max_steps=max_steps, samples_per_alpha=samples),
            algorithm_key="bigamp",
            graph_key="random",
            teacher_key="standard",
        )

        # Summary
        self._print_standard_summary(config)

        if Confirm.ask("Confirm?", default=True):
            return {'type': 'standard', 'config': config}
        return None

    def _configure_size_scaling(self) -> Optional[dict]:
        """Configure size scaling (finite-size effect) experiment."""
        self.console.print(Panel(
            "[bold]Size Scaling Experiment[/bold]\n"
            "[dim]Multiple N values to study finite-size effects[/dim]",
            border_style="blue"
        ))

        # Multiple sizes
        self.console.print("[bold]Matrix Sizes:[/bold]")
        self.console.print("[dim]Enter multiple N values (all use same M)[/dim]")

        sizes_str = Prompt.ask(
            "  N values (comma separated)",
            default="1000, 2000, 3000, 4000, 5000"
        )
        M = IntPrompt.ask("  M (latent dimension, fixed)", default=100)

        # Parse sizes
        try:
            N_values = [int(x.strip()) for x in sizes_str.split(",")]
        except ValueError:
            self.console.print("[red]Invalid format. Using defaults.[/red]")
            N_values = [1000, 2000, 3000, 4000, 5000]

        matrix_configs = [(N, N, M) for N in N_values]
        self.console.print(f"  [green]→ {len(matrix_configs)} configurations[/green]")
        self.console.print()

        # Alpha range (typically narrow for size scaling)
        self.console.print("[bold]Alpha Scan Range:[/bold]")
        self.console.print("[dim]Typically narrow range (0~0.5) for size scaling[/dim]")
        alpha_start = FloatPrompt.ask("  Start", default=0.0)
        alpha_stop = FloatPrompt.ask("  End", default=0.5)
        alpha_step = FloatPrompt.ask("  Step", default=0.01)
        self.console.print()

        # Training
        self.console.print("[bold]Training Parameters:[/bold]")
        max_steps = IntPrompt.ask("  Max steps", default=10000)
        samples = IntPrompt.ask("  Samples per alpha", default=1)
        self.console.print()

        # Summary
        self.console.print(Panel("[bold]Configuration Summary[/bold]", border_style="green"))
        table = Table(show_header=False, box=None)
        table.add_row("Sizes", f"{N_values} (M={M})")
        table.add_row("Alpha", f"{alpha_start} ~ {alpha_stop}, step {alpha_step}")
        table.add_row("Training", f"steps={max_steps}, S={samples}")
        self.console.print(table)

        if Confirm.ask("Confirm?", default=True):
            return {
                'type': 'size_scaling',
                'matrix_configs': matrix_configs,
                'alpha_start': alpha_start,
                'alpha_stop': alpha_stop,
                'alpha_step': alpha_step,
                'max_steps': max_steps,
                'samples': samples,
            }
        return None

    def _configure_init_scale(self) -> Optional[dict]:
        """Configure init scale comparison experiment."""
        self.console.print(Panel(
            "[bold]Init Scale Experiment[/bold]\n"
            "[dim]Compare different k/√M initialization scales[/dim]",
            border_style="blue"
        ))

        # Matrix dimensions (single config)
        self.console.print("[bold]Matrix Dimensions:[/bold]")
        N1 = IntPrompt.ask("  N1", default=200)
        N2 = IntPrompt.ask("  N2", default=N1)
        M = IntPrompt.ask("  M", default=50)
        self.console.print()

        # Scale factors
        self.console.print("[bold]Initialization Scale Factors (k in k/√M):[/bold]")
        scales_str = Prompt.ask(
            "  k values (comma separated)",
            default="0.5, 1.0, 1.5, 2.0"
        )

        try:
            scale_factors = [float(x.strip()) for x in scales_str.split(",")]
        except ValueError:
            self.console.print("[red]Invalid format. Using defaults.[/red]")
            scale_factors = [0.5, 1.0, 1.5, 2.0]

        self.console.print(f"  [green]→ {len(scale_factors)} scale factors[/green]")
        self.console.print()

        # Alpha range
        self.console.print("[bold]Alpha Scan Range:[/bold]")
        alpha_start = FloatPrompt.ask("  Start", default=0.0)
        alpha_stop = FloatPrompt.ask("  End", default=4.0)
        alpha_step = FloatPrompt.ask("  Step", default=0.1)
        self.console.print()

        # Training
        self.console.print("[bold]Training Parameters:[/bold]")
        max_steps = IntPrompt.ask("  Max steps", default=1000)
        self.console.print()

        # Summary
        self.console.print(Panel("[bold]Configuration Summary[/bold]", border_style="green"))
        table = Table(show_header=False, box=None)
        table.add_row("Matrix", f"{N1}×{N2}, M={M}")
        table.add_row("Scale factors", str(scale_factors))
        table.add_row("Alpha", f"{alpha_start} ~ {alpha_stop}, step {alpha_step}")
        table.add_row("Training", f"steps={max_steps}")
        self.console.print(table)

        config = Config(
            matrix=MatrixConfig(N1=N1, N2=N2, M=M),
            alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
            training=TrainingConfig(max_steps=max_steps, samples_per_alpha=1),
            algorithm_key="bigamp",
            graph_key="random",
        )

        if Confirm.ask("Confirm?", default=True):
            return {
                'type': 'init_scale',
                'config': config,
                'scale_factors': scale_factors,
            }
        return None

    def _configure_custom(self) -> Optional[dict]:
        """Full manual configuration (original wizard)."""
        self.console.print(Panel(
            "[bold]Custom Configuration[/bold]\n"
            "[dim]Full manual control over all parameters[/dim]",
            border_style="blue"
        ))

        # Step 1: Algorithm
        self.console.print("[bold]Algorithm:[/bold]")
        algorithms = list_algorithms()
        if not algorithms:
            self.console.print("[yellow]Only BiG-AMP is currently available[/yellow]")
            algorithm_key = "bigamp"
        else:
            table = Table(show_header=False, box=None)
            for i, alg in enumerate(algorithms, 1):
                table.add_row(f"[yellow][{i}][/yellow]", alg.name, f"[dim]{alg.description}[/dim]")
            self.console.print(table)
            alg_choice = IntPrompt.ask("Select", default=1)
            alg_choice = max(1, min(alg_choice, len(algorithms)))
            algorithm_key = algorithms[alg_choice - 1].key
        self.console.print()

        # Step 2: Graph
        self.console.print("[bold]Graph Generation:[/bold]")
        graphs = list_graphs()
        table = Table(show_header=False, box=None)
        for i, g in enumerate(graphs, 1):
            table.add_row(f"[yellow][{i}][/yellow]", g.name, f"[dim]{g.description}[/dim]")
        self.console.print(table)
        graph_choice = IntPrompt.ask("Select", default=1)
        graph_choice = max(1, min(graph_choice, len(graphs)))
        graph_key = graphs[graph_choice - 1].key
        self.console.print()

        # Step 3: Matrix
        self.console.print("[bold]Matrix Dimensions:[/bold]")
        N1 = IntPrompt.ask("  N1", default=200)
        N2 = IntPrompt.ask("  N2", default=N1)
        M = IntPrompt.ask("  M", default=50)
        self.console.print()

        # Step 4: Alpha
        self.console.print("[bold]Alpha Range:[/bold]")
        alpha_start = FloatPrompt.ask("  Start", default=0.0)
        alpha_stop = FloatPrompt.ask("  End", default=4.0)
        alpha_step = FloatPrompt.ask("  Step", default=0.1)
        self.console.print()

        # Step 5: Training
        self.console.print("[bold]Training Parameters:[/bold]")
        max_steps = IntPrompt.ask("  Max steps", default=5000)
        samples = IntPrompt.ask("  Samples/alpha", default=1)
        seed = IntPrompt.ask("  Random seed", default=42)
        self.console.print()

        config = Config(
            matrix=MatrixConfig(N1=N1, N2=N2, M=M),
            alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
            training=TrainingConfig(max_steps=max_steps, samples_per_alpha=samples, seed=seed),
            algorithm=AlgorithmConfig(),
            algorithm_key=algorithm_key,
            graph_key=graph_key,
            teacher_key="standard",
        )

        self._print_standard_summary(config)

        if Confirm.ask("Confirm?", default=True):
            return {'type': 'standard', 'config': config}
        return None

    def _print_standard_summary(self, config: Config):
        """Print config summary."""
        self.console.print(Panel("[bold]Configuration Summary[/bold]", border_style="green"))
        table = Table(show_header=False, box=None)
        m = config.matrix
        a = config.alpha
        t = config.training
        table.add_row("Algorithm", config.algorithm_key)
        table.add_row("Graph", config.graph_key)
        table.add_row("Matrix", f"{m.N1}×{m.N2}, M={m.M}")
        table.add_row("Alpha", f"{a.start} ~ {a.stop}, step {a.step}")
        table.add_row("Training", f"steps={t.max_steps}, S={t.samples_per_alpha}")
        self.console.print(table)
        self.console.print()

    def _run_simple(self) -> Optional[dict]:
        """Simple text-based wizard (fallback)."""
        print("\n=== Experiment Configuration ===\n")

        print("Step 1: Choose Experiment Type")
        print("  [1] Standard - Single (N, M), Q_Y vs α curve")
        print("  [2] Size Scaling - Multiple N values, finite-size effect")
        print("  [3] Init Scale - Different k/√M initialization")
        print("  [4] Custom - Full manual configuration")

        exp_type = input("Select [1]: ").strip() or "1"

        if exp_type == "1":
            return self._configure_standard_simple()
        elif exp_type == "2":
            return self._configure_size_scaling_simple()
        elif exp_type == "3":
            return self._configure_init_scale_simple()
        elif exp_type == "4":
            return self._configure_custom_simple()

        return None

    def _configure_standard_simple(self) -> Optional[dict]:
        """Simple standard config."""
        print("\n--- Standard Experiment ---")

        N1 = int(input("N1 [200]: ") or "200")
        N2 = int(input(f"N2 [{N1}]: ") or str(N1))
        M = int(input("M [50]: ") or "50")

        alpha_start = float(input("Alpha start [0.0]: ") or "0.0")
        alpha_stop = float(input("Alpha end [4.0]: ") or "4.0")
        alpha_step = float(input("Alpha step [0.1]: ") or "0.1")

        max_steps = int(input("Steps [5000]: ") or "5000")
        samples = int(input("Samples [1]: ") or "1")

        config = Config(
            matrix=MatrixConfig(N1=N1, N2=N2, M=M),
            alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
            training=TrainingConfig(max_steps=max_steps, samples_per_alpha=samples),
            algorithm_key="bigamp",
            graph_key="random",
            teacher_key="standard",
        )

        confirm = input("\nConfirm? [Y/n]: ").strip().lower()
        if confirm in ('', 'y', 'yes'):
            return {'type': 'standard', 'config': config}
        return None

    def _configure_size_scaling_simple(self) -> Optional[dict]:
        """Simple size scaling config."""
        print("\n--- Size Scaling Experiment ---")

        sizes_str = input("N values (comma sep) [1000,2000,3000,4000,5000]: ") or "1000,2000,3000,4000,5000"
        M = int(input("M [100]: ") or "100")

        N_values = [int(x.strip()) for x in sizes_str.split(",")]
        matrix_configs = [(N, N, M) for N in N_values]

        alpha_start = float(input("Alpha start [0.0]: ") or "0.0")
        alpha_stop = float(input("Alpha end [0.5]: ") or "0.5")
        alpha_step = float(input("Alpha step [0.01]: ") or "0.01")

        max_steps = int(input("Steps [10000]: ") or "10000")

        confirm = input("\nConfirm? [Y/n]: ").strip().lower()
        if confirm in ('', 'y', 'yes'):
            return {
                'type': 'size_scaling',
                'matrix_configs': matrix_configs,
                'alpha_start': alpha_start,
                'alpha_stop': alpha_stop,
                'alpha_step': alpha_step,
                'max_steps': max_steps,
                'samples': 1,
            }
        return None

    def _configure_init_scale_simple(self) -> Optional[dict]:
        """Simple init scale config."""
        print("\n--- Init Scale Experiment ---")

        N1 = int(input("N1 [200]: ") or "200")
        N2 = int(input(f"N2 [{N1}]: ") or str(N1))
        M = int(input("M [50]: ") or "50")

        scales_str = input("k values (comma sep) [0.5,1.0,1.5,2.0]: ") or "0.5,1.0,1.5,2.0"
        scale_factors = [float(x.strip()) for x in scales_str.split(",")]

        alpha_start = float(input("Alpha start [0.0]: ") or "0.0")
        alpha_stop = float(input("Alpha end [4.0]: ") or "4.0")
        alpha_step = float(input("Alpha step [0.1]: ") or "0.1")

        max_steps = int(input("Steps [1000]: ") or "1000")

        config = Config(
            matrix=MatrixConfig(N1=N1, N2=N2, M=M),
            alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
            training=TrainingConfig(max_steps=max_steps, samples_per_alpha=1),
            algorithm_key="bigamp",
            graph_key="random",
        )

        confirm = input("\nConfirm? [Y/n]: ").strip().lower()
        if confirm in ('', 'y', 'yes'):
            return {
                'type': 'init_scale',
                'config': config,
                'scale_factors': scale_factors,
            }
        return None

    def _configure_custom_simple(self) -> Optional[dict]:
        """Simple custom config."""
        print("\n--- Custom Configuration ---")

        algorithms = list_algorithms()
        if algorithms:
            print("Algorithms:")
            for i, alg in enumerate(algorithms, 1):
                print(f"  [{i}] {alg.name}")
            alg_choice = int(input("Select [1]: ") or "1")
            algorithm_key = algorithms[min(max(alg_choice, 1), len(algorithms)) - 1].key
        else:
            print("Only BiG-AMP available")
            algorithm_key = "bigamp"

        graphs = list_graphs()
        print("Graphs:")
        for i, g in enumerate(graphs, 1):
            print(f"  [{i}] {g.name}")
        graph_choice = int(input("Select [1]: ") or "1")
        graph_key = graphs[min(max(graph_choice, 1), len(graphs)) - 1].key

        N1 = int(input("N1 [200]: ") or "200")
        N2 = int(input(f"N2 [{N1}]: ") or str(N1))
        M = int(input("M [50]: ") or "50")

        alpha_start = float(input("Alpha start [0.0]: ") or "0.0")
        alpha_stop = float(input("Alpha end [4.0]: ") or "4.0")
        alpha_step = float(input("Alpha step [0.1]: ") or "0.1")

        max_steps = int(input("Steps [5000]: ") or "5000")
        samples = int(input("Samples [1]: ") or "1")

        config = Config(
            matrix=MatrixConfig(N1=N1, N2=N2, M=M),
            alpha=AlphaConfig(start=alpha_start, stop=alpha_stop, step=alpha_step),
            training=TrainingConfig(max_steps=max_steps, samples_per_alpha=samples),
            algorithm_key=algorithm_key,
            graph_key=graph_key,
            teacher_key="standard",
        )

        confirm = input("\nConfirm? [Y/n]: ").strip().lower()
        if confirm in ('', 'y', 'yes'):
            return {'type': 'standard', 'config': config}
        return None


def run_in_background(config: Config, log_file: str = None) -> str:
    """
    Run experiment in background using nohup.

    Args:
        config: Experiment configuration
        log_file: Log file path (default: smf_run_TIMESTAMP.log)

    Returns:
        Log file path
    """
    from datetime import datetime
    import json as json_module

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"smf_run_{timestamp}.log"

    # Save config to temp file
    config_file = f"/tmp/smf_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    config.to_json(Path(config_file))

    # Build nohup command
    cmd = f'nohup python -c "from smf.runner import run_experiment; from smf.core.config import Config; c = Config.from_json(\\\"{config_file}\\\"); run_experiment(c)" > {log_file} 2>&1 &'

    subprocess.Popen(cmd, shell=True)

    return log_file


def ask_background_run(console, lang: str = 'cn') -> bool:
    """Ask user if they want to run in background."""
    if not RICH_AVAILABLE:
        return False

    prompt = TEXTS[lang]['background_prompt']
    return Confirm.ask(prompt, default=False)
