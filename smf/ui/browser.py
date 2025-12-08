"""
Result browser - interactive browsing and filtering of saved results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.status import Status
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    box = None

from ..modules.outputs.storage import list_results, ResultStorage
from ..core.opener import open_image, open_folder
from ..core.llm_filter import filter_with_llm
from .theme import THEME


class ResultBrowser:
    """Interactive browser for experiment results."""

    def __init__(self, lang: str = 'cn'):
        self.console = Console() if RICH_AVAILABLE else None
        self.results = []
        self.lang = lang

    def refresh(self):
        """Refresh the results list."""
        self.results = list_results()

    def browse(self):
        """Main browsing interface."""
        self.refresh()

        if not self.results:
            self._print("No saved results")
            return

        while True:
            action = self._show_main_menu()

            if action == 'list':
                self._list_results()
            elif action == 'ai_filter':
                self._ai_filter_results()
            elif action == 'filter':
                self._filter_results()
            elif action == 'view':
                self._view_result()
            elif action == 'exit':
                break

    def _show_main_menu(self) -> str:
        """Show main browser menu."""
        if RICH_AVAILABLE:
            self.console.print()
            title = "结果浏览器" if self.lang == 'cn' else "Result Browser"
            self.console.print(Panel(
                f"[bold cyan]{title}[/bold cyan]",
                border_style="cyan"
            ))

            table = Table(show_header=False, box=box.ROUNDED, padding=(0, 1))
            if self.lang == 'cn':
                table.add_row("[yellow][1][/yellow]", f"列出所有结果 ({len(self.results)} 项)")
                table.add_row("[yellow][2][/yellow]", f"[{THEME['ai_title']}]AI 智能筛选[/{THEME['ai_title']}]")
                table.add_row("[yellow][3][/yellow]", "手动筛选")
                table.add_row("[yellow][4][/yellow]", "查看详情")
                table.add_row("[yellow][q][/yellow]", "返回")
            else:
                table.add_row("[yellow][1][/yellow]", f"List all results ({len(self.results)} items)")
                table.add_row("[yellow][2][/yellow]", f"[{THEME['ai_title']}]AI Filter[/{THEME['ai_title']}]")
                table.add_row("[yellow][3][/yellow]", "Manual filter")
                table.add_row("[yellow][4][/yellow]", "View details")
                table.add_row("[yellow][q][/yellow]", "Back")
            self.console.print(table)

            prompt = "选择 (默认: 1)" if self.lang == 'cn' else "Select (default: 1)"
            choice = Prompt.ask(prompt, choices=["1", "2", "3", "4", "q"], default="1", show_default=False)
        else:
            print(f"\n=== Result Browser ({len(self.results)} results) ===")
            print("[1] List all results")
            print("[2] AI Filter (describe what you want)")
            print("[3] Manual filter by criteria")
            print("[4] View specific result")
            print("[q] Back")
            choice = input("Select (default: 1): ").strip() or "1"

        return {"1": "list", "2": "ai_filter", "3": "filter", "4": "view", "q": "exit"}.get(choice, "exit")

    def _list_results(self, results: List[Dict] = None, limit: int = 20):
        """List results in a table."""
        results = results or self.results

        if RICH_AVAILABLE:
            table = Table(title=f"Experiment Results (showing first {min(limit, len(results))})")
            table.add_column("#", style="dim")
            table.add_column("Type", style="magenta")
            table.add_column("Matrix", style="green")
            table.add_column("Time", style="cyan")
            table.add_column("Source", style="dim")

            # Type display names
            type_names = {
                "overlap_metrics": "baseline",
                "size_scaling": "size",
                "loop_free": "loop-free",
                "replica": "replica",
                "init_scale": "init",
            }

            for i, r in enumerate(results[:limit], 1):
                time_str = r.get('timestamp', '')[:16] if r.get('timestamp') else '?'
                matrix_str = f"{r.get('N1', '?')}×{r.get('N2', '?')}, M={r.get('M', '?')}"
                exp_type = type_names.get(r.get('type', ''), r.get('type', '?'))
                source = r.get('source', 'new')
                table.add_row(
                    str(i),
                    exp_type,
                    matrix_str,
                    time_str,
                    source
                )

            self.console.print(table)

            if len(results) > limit:
                self.console.print(f"[dim]... {len(results) - limit} more results[/dim]")
        else:
            print(f"\n--- Experiment Results (first {min(limit, len(results))}) ---")
            for i, r in enumerate(results[:limit], 1):
                print(f"  [{i}] {r.get('name', '?')}")
                print(f"      {r.get('N1', '?')}×{r.get('N2', '?')}, M={r.get('M', '?')}")
            if len(results) > limit:
                print(f"  ... {len(results) - limit} more results")

    def _ai_filter_results(self):
        """AI-powered natural language filtering."""
        if RICH_AVAILABLE:
            self.console.print()
            if self.lang == 'cn':
                self.console.print(Panel(
                    f"[{THEME['ai_title']}]AI 智能筛选[/{THEME['ai_title']}]\n"
                    "[dim]用自然语言描述你要找的实验[/dim]",
                    border_style="magenta"
                ))
                self.console.print("[dim]示例：'大矩阵', '正交教师', 'M=100', '最近的实验'[/dim]")
            else:
                self.console.print(Panel(
                    f"[{THEME['ai_title']}]AI Filter[/{THEME['ai_title']}]\n"
                    "[dim]Describe what you're looking for in natural language[/dim]",
                    border_style="magenta"
                ))
                self.console.print("[dim]Examples: 'large matrix', 'orthogonal teacher', 'M=100'[/dim]")
            self.console.print()
            prompt = "描述" if self.lang == 'cn' else "Describe"
            query = Prompt.ask(prompt, default="")
        else:
            print("\n=== AI Filter ===")
            print("Describe what you're looking for (e.g., '大矩阵实验', 'M=100')")
            query = input("Describe: ").strip()

        if not query:
            msg = "未输入查询" if self.lang == 'cn' else "No query provided"
            self._print(msg)
            return

        # Call LLM filter with spinner
        try:
            if RICH_AVAILABLE:
                spinner_text = "[bold cyan]AI 正在分析...[/bold cyan]" if self.lang == 'cn' else "[bold cyan]AI analyzing...[/bold cyan]"
                with Status(spinner_text, spinner="dots", console=self.console):
                    filtered, explanation = filter_with_llm(query, self.results)
            else:
                print("\nAnalyzing...")
                filtered, explanation = filter_with_llm(query, self.results)

            if RICH_AVAILABLE:
                self.console.print()
                self.console.print(f"[bold green]AI:[/bold green] {explanation}")
                found_msg = f"找到 {len(filtered)} 个结果" if self.lang == 'cn' else f"Found {len(filtered)} results"
                self.console.print(f"{found_msg}\n")
            else:
                print(f"\nAI: {explanation}")
                print(f"Found {len(filtered)} results\n")

            if filtered:
                self._list_results(filtered)
                self._view_from_list(filtered)
            else:
                msg = "没有匹配的结果" if self.lang == 'cn' else "No matching results"
                self._print(msg)

        except Exception as e:
            self._print(f"AI filter error: {e}")
            msg = "回退到手动筛选..." if self.lang == 'cn' else "Falling back to manual filter..."
            self._print(msg)
            self._filter_results()

    def _filter_results(self):
        """Filter results by criteria."""
        if RICH_AVAILABLE:
            self.console.print(Panel("[bold]Filter Criteria[/bold]", border_style="yellow"))

            # Get unique values
            algorithms = list(set(r.get('algorithm') for r in self.results if r.get('algorithm')))
            graphs = list(set(r.get('graph') for r in self.results if r.get('graph')))
            sizes = list(set(f"{r.get('N1')}x{r.get('N2')}" for r in self.results
                            if r.get('N1') and r.get('N2')))

            # Algorithm filter
            self.console.print(f"Available algorithms: {', '.join(algorithms) if algorithms else 'none'}")
            alg_filter = Prompt.ask("Algorithm (empty=all)", default="")

            # Graph filter
            self.console.print(f"Available graph modes: {', '.join(graphs) if graphs else 'none'}")
            graph_filter = Prompt.ask("Graph mode (empty=all)", default="")

            # Size filter
            n1_filter = Prompt.ask("N1 (empty=all)", default="")
            m_filter = Prompt.ask("M (empty=all)", default="")
        else:
            print("\n--- Filter Criteria ---")
            alg_filter = input("Algorithm (empty=all): ").strip()
            graph_filter = input("Graph mode (empty=all): ").strip()
            n1_filter = input("N1 (empty=all): ").strip()
            m_filter = input("M (empty=all): ").strip()

        # Apply filters (fuzzy match: case-insensitive substring)
        def fuzzy_match(value: str, pattern: str) -> bool:
            """Check if pattern is a substring of value (case-insensitive)."""
            if not value or not pattern:
                return False
            return pattern.lower() in value.lower()

        filtered = self.results
        if alg_filter:
            filtered = [r for r in filtered if fuzzy_match(r.get('algorithm', ''), alg_filter)]
        if graph_filter:
            filtered = [r for r in filtered if fuzzy_match(r.get('graph', ''), graph_filter)]
        if n1_filter:
            try:
                n1_val = int(n1_filter)
                filtered = [r for r in filtered if r.get('N1') == n1_val]
            except ValueError:
                pass
        if m_filter:
            try:
                m_val = int(m_filter)
                filtered = [r for r in filtered if r.get('M') == m_val]
            except ValueError:
                pass

        self._print(f"\nFilter results: {len(filtered)} items")
        self._list_results(filtered)

        # Allow viewing from filtered results
        if filtered:
            self._view_from_list(filtered)

    def _view_from_list(self, results_list: List[Dict]):
        """Allow user to select and view a result from a list."""
        if RICH_AVAILABLE:
            self.console.print()
            prompt = "输入序号查看，或 b 返回 (默认: b)" if self.lang == 'cn' else "Enter number to view (default: b)"
            choice = Prompt.ask(prompt, default="b", show_default=False)
        else:
            prompt = "输入序号查看，或 b 返回 (默认: b): " if self.lang == 'cn' else "Enter number to view (default: b): "
            choice = input(f"\n{prompt}").strip() or "b"

        if choice.lower() == 'b':
            return

        try:
            idx = int(choice)
            if 1 <= idx <= len(results_list):
                self._show_result_details(results_list[idx - 1])
            else:
                self._print("Invalid number")
        except ValueError:
            self._print("Invalid input")

    def _view_result(self):
        """View details of a specific result."""
        self._list_results(limit=20)
        self._view_from_list(self.results[:20])

    def _show_result_details(self, result: Dict):
        """Show detailed info for a result."""
        result_path = result.get('path')
        if not result_path:
            self._print("Cannot find result path")
            return

        # Load full result
        try:
            data = ResultStorage.load(result_path)
        except Exception as e:
            self._print(f"Load failed: {e}")
            return

        if RICH_AVAILABLE:
            self.console.print()
            self.console.print(Panel(f"[bold]{result['name']}[/bold]", border_style="green"))

            # Config info
            config = data.get('config')
            if config:
                title = "配置" if self.lang == 'cn' else "Config"
                table = Table(show_header=False, box=None, title=title)
                m = config.matrix if hasattr(config, 'matrix') else config.get('matrix', {})
                matrix_label = "矩阵" if self.lang == 'cn' else "Matrix"
                algo_label = "算法" if self.lang == 'cn' else "Algorithm"
                graph_label = "图模式" if self.lang == 'cn' else "Graph Mode"
                if isinstance(m, dict):
                    table.add_row(matrix_label, f"{m.get('N1')}×{m.get('N2')}, M={m.get('M')}")
                table.add_row(algo_label, str(config.algorithm_key if hasattr(config, 'algorithm_key')
                                          else config.get('algorithm_key', '?')))
                table.add_row(graph_label, str(config.graph_key if hasattr(config, 'graph_key')
                                            else config.get('graph_key', '?')))
                self.console.print(table)

            # Metadata
            metadata = data.get('metadata', {})
            if metadata:
                runtime_label = "运行时间" if self.lang == 'cn' else "Runtime"
                device_label = "设备" if self.lang == 'cn' else "Device"
                self.console.print(f"\n{runtime_label}: {metadata.get('total_time', '?'):.1f}s")
                self.console.print(f"{device_label}: {metadata.get('device', '?')}")

            # Actions
            self.console.print()
            if self.lang == 'cn':
                self.console.print("[o] 打开图片  [f] 打开文件夹  [b] 返回")
                prompt = "操作 (默认: o)"
            else:
                self.console.print("[o] Open image  [f] Open folder  [b] Back")
                prompt = "Action (default: o)"
            action = Prompt.ask(
                prompt,
                choices=["o", "f", "b"],
                default="o",
                show_default=False
            )
        else:
            print(f"\n--- {result['name']} ---")
            print(f"Path: {result_path}")
            if self.lang == 'cn':
                action = input("[o] 打开图片  [f] 打开文件夹  [b] 返回 (默认: o): ").strip().lower() or "o"
            else:
                action = input("[o] Open image  [f] Open folder  [b] Back (default: o): ").strip().lower() or "o"

        if action == 'o':
            result_dir = Path(result_path)
            image_found = False

            msg_opened = "图片已打开" if self.lang == 'cn' else "Image opened"
            msg_not_found = "未找到图片" if self.lang == 'cn' else "No image found"
            msg_folder = "文件夹已打开" if self.lang == 'cn' else "Folder opened"

            # Priority 1: plots/summary.png (standard experiments)
            plots_dir = result_dir / "plots"
            if plots_dir.exists():
                summary = plots_dir / "summary.png"
                if summary.exists():
                    open_image(summary)
                    self._print(msg_opened)
                    image_found = True
                else:
                    # Any image in plots/
                    images = list(plots_dir.glob("*.png"))
                    if images:
                        open_image(images[0])
                        self._print(msg_opened)
                        image_found = True

            # Priority 2: *_comparison.png in root (variance_sweep, large_matrix)
            if not image_found:
                comparison_images = list(result_dir.glob("*_comparison.png"))
                if comparison_images:
                    open_image(comparison_images[0])
                    self._print(msg_opened)
                    image_found = True

            # Priority 3: Any .png in root
            if not image_found:
                root_images = list(result_dir.glob("*.png"))
                if root_images:
                    open_image(root_images[0])
                    self._print(msg_opened)
                    image_found = True

            if not image_found:
                self._print(msg_not_found)
        elif action == 'f':
            open_folder(result_path)
            msg_folder = "文件夹已打开" if self.lang == 'cn' else "Folder opened"
            self._print(msg_folder)

    def _print(self, message: str):
        """Print message."""
        if RICH_AVAILABLE:
            self.console.print(message)
        else:
            print(message)


def browse_results():
    """Convenience function to start browser."""
    browser = ResultBrowser()
    browser.browse()
