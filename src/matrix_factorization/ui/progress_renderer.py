
from typing import Dict, List, Optional
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn
)

class ProgressRenderer:
    """
    Handles the visual rendering of progress bars using 'Dynamic Capsule' layout.
    Purely stateless renderer - receives state data and returns renderables.
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.live = None
        self._frame = 0

    def create_live_display(self, initial_renderable):
        """Creates and returns a Rich Live display context."""
        return Live(
            initial_renderable,
            refresh_per_second=10,
            console=self.console,
            transient=False,
        )

    def render_capsule_panel(self, state: Dict) -> Panel:
        """
        Render the Dynamic Capsule panel - Style A (Clean Cyan).
        
        Args:
            state: Dictionary containing current progress state:
                - step_progress: (current, total)
                - batch_progress: (current_idx, total_batches)
                - alphas: [current_alphas...]
                - timing: (elapsed, eta, batch_elapsed_str, batch_total_str)
                - throughput: it_per_sec
                - gpu: (power, memory)
        """
        self._frame += 1
        
        # Unpack state
        current_step, total_steps = state.get('step_progress', (0, 1))
        completed_batches, total_batches = state.get('batch_progress', (0, 1))
        completed_points, total_points = state.get('point_progress', (completed_batches, total_batches))
        current_alphas = state.get('alphas', [])
        context_lines = state.get('context_lines', [])
        elapsed_str, eta_str, batch_elapsed_str, batch_total_str = state.get('timing', ("--:--", "--:--", "--:--", "--:--"))
        it_per_sec = state.get('throughput', 0.0)
        power, memory = state.get('gpu', (0, 0))
        
        # Calculations
        pct_step = current_step / total_steps if total_steps > 0 else 0
        batch_progress_val = completed_batches + (pct_step if completed_batches < total_batches else 0)
        pct_batch = batch_progress_val / total_batches if total_batches > 0 else 0
        
        # Format strings
        if len(current_alphas) > 1:
            alpha_str = f"{current_alphas[0]:.2f}-{current_alphas[-1]:.2f}"
        else:
            alpha_str = f"{current_alphas[0]:.2f}" if current_alphas else "0.00"
            
        it_s_val = f"{it_per_sec:.1f}" if it_per_sec > 0 else "--"
        
        # Moon phase spinner
        moon_frames = "🌑🌒🌓🌔🌕🌖🌗🌘"
        spinner = moon_frames[self._frame // 3 % 8]

        # Build grid
        grid = Table.grid(padding=0)
        grid.add_column()

        for line in context_lines[:2]:
            if line:
                grid.add_row(Text.from_markup(f" [dim]{line}[/dim]"))

        # Row 1: Header
        grid.add_row(Text.from_markup(
            f" {spinner}  Step [cyan]{current_step}[/]/{total_steps}  "
            f"α [cyan]{alpha_str}[/]  [cyan]{it_s_val}[/]it/s  "
            f"[cyan]{batch_elapsed_str}[/]/{batch_total_str}"
        ))

        # Row 2: Step Bar
        row2 = Text(" Step  ")
        row2.append_text(self._make_bar_text(pct_step, 40, "━", "━", "cyan"))
        row2.append(f" {pct_step*100:5.1f}", style="cyan")
        row2.append("%")
        grid.add_row(row2)

        # Row 3: Total Bar
        row3 = Text(" Total ")
        row3.append_text(self._make_bar_text(pct_batch, 40, "━", "━", "cyan"))
        row3.append(f"  {completed_batches}", style="cyan")
        row3.append(f"/{total_batches}")
        if total_points and total_points != total_batches:
            row3.append(f"  pts {completed_points}/{total_points}", style="dim")
        grid.add_row(row3)

        # Row 4: Metrics
        grid.add_row(Text.from_markup(
            f" Power [cyan]{power:.0f}[/]W  VRAM [cyan]{memory:.1f}[/]G  "
            f"Elapsed [cyan]{elapsed_str}[/]  ETA [cyan]{eta_str}[/]"
        ))

        return Panel(
            grid,
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False
        )

    def _make_bar_text(self, pct, width, filled_char, empty_char, color):
        pct = max(0.0, min(1.0, pct))
        filled = int(width * pct)
        empty = width - filled
        text = Text()
        text.append(filled_char * filled, style=color)
        text.append(empty_char * empty, style="dim")
        return text
