"""
Main menu interface using rich library.
"""

from typing import Optional, Callable
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .theme import THEME

# Menu texts
MENU_TEXTS = {
    'en': {
        'title': 'Sparse Matrix Factorization',
        'run': 'Run Experiment',
        'run_desc': 'Standard / Size Scaling / Init Scale',
        'browse': 'Browse Results',
        'browse_desc': 'View and analyze experiment history',
        'exit': 'Exit',
        'select': 'Select option',
    },
    'cn': {
        'title': '稀疏矩阵分解框架',
        'run': '运行实验',
        'run_desc': '标准 / 尺寸扫描 / 初始化缩放',
        'browse': '浏览结果',
        'browse_desc': '查看和分析历史实验',
        'exit': '退出',
        'select': '选择',
    }
}


class MainMenu:
    """Main menu interface for SMF."""

    def __init__(self, lang: str = 'cn'):
        self.console = Console() if RICH_AVAILABLE else None
        self.lang = lang

    def t(self, key: str) -> str:
        """Get translated text."""
        return MENU_TEXTS.get(self.lang, MENU_TEXTS['cn']).get(key, key)

    def show(self) -> str:
        """
        Display main menu and get user choice.

        Returns:
            Choice key: 'run', 'browse', 'preset', 'exit'
        """
        if RICH_AVAILABLE:
            return self._show_rich()
        else:
            return self._show_simple()

    def _show_rich(self) -> str:
        """Rich-based menu."""
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]{self.t('title')}[/bold cyan]",
            expand=False,
            border_style="cyan",
        ))
        self.console.print()

        table = Table(show_header=False, box=None, padding=(0, 1))

        table.add_row(
            f"[{THEME['option_number']}]\\[1][/{THEME['option_number']}]",
            f"[{THEME['option_title']}]{self.t('run')}[/{THEME['option_title']}]",
            f"[{THEME['option_desc']}]{self.t('run_desc')}[/{THEME['option_desc']}]"
        )
        table.add_row(
            f"[{THEME['option_number']}]\\[2][/{THEME['option_number']}]",
            f"[{THEME['option_title']}]{self.t('browse')}[/{THEME['option_title']}]",
            f"[{THEME['option_desc']}]{self.t('browse_desc')}[/{THEME['option_desc']}]"
        )
        table.add_row(
            f"[{THEME['option_number']}]\\[q][/{THEME['option_number']}]",
            f"[{THEME['option_title']}]{self.t('exit')}[/{THEME['option_title']}]",
            ""
        )

        self.console.print(table)
        self.console.print()

        prompt = f"{self.t('select')} (默认: [1])" if self.lang == 'cn' else f"{self.t('select')} (default: [1])"
        choice = Prompt.ask(
            prompt,
            choices=["1", "2", "q"],
            default="1",
            show_default=False,
            show_choices=False,
        )

        return {
            "1": "run",
            "2": "browse",
            "q": "exit",
        }.get(choice, "exit")

    def _show_simple(self) -> str:
        """Simple text-based menu."""
        print("\n" + "=" * 50)
        print(f"  {self.t('title')}")
        print("=" * 50)
        print()
        print(f"  [1] {self.t('run')} ({self.t('run_desc')})")
        print(f"  [2] {self.t('browse')}")
        print(f"  [q] {self.t('exit')}")
        print()

        choice = input(f"{self.t('select')} [1]: ").strip() or "1"

        return {
            "1": "run",
            "2": "browse",
            "q": "exit",
        }.get(choice, "exit")

    def confirm(self, message: str, default: bool = True) -> bool:
        """Ask for confirmation."""
        if RICH_AVAILABLE:
            from rich.prompt import Confirm
            return Confirm.ask(message, default=default)
        else:
            suffix = "[Y/n]" if default else "[y/N]"
            response = input(f"{message} {suffix}: ").strip().lower()
            if not response:
                return default
            return response in ('y', 'yes')

    def print_error(self, message: str):
        """Print error message."""
        if RICH_AVAILABLE:
            self.console.print(f"[bold red]Error:[/bold red] {message}")
        else:
            print(f"Error: {message}")

    def print_success(self, message: str):
        """Print success message."""
        if RICH_AVAILABLE:
            self.console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"✓ {message}")

    def print_info(self, message: str):
        """Print info message."""
        if RICH_AVAILABLE:
            self.console.print(f"[dim]{message}[/dim]")
        else:
            print(message)
