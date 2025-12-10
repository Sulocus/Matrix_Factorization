"""
Progress display using rich library with GPU monitoring.

Modern UI with 'Dynamic Capsule' aesthetic.
"""

from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable, List
import time
import math
from datetime import timedelta

try:
    from rich.console import Console, Group
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn,
        MofNCompleteColumn
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.style import Style
    from rich.spinner import Spinner
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .gpu_monitor import GPUMonitor


class UnifiedProgress:
    """
    Modern progress display with 'Dynamic Capsule' layout - Style A (Clean Cyan).

    Structure for Disjoint Union parallel training:
    ╭────────────────────────────────────────────────────────╮
    │  ●  Batch  1/3     Alpha 0.0-1.5   Step  2500/5000     │
    │  Step  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  50.0% │
    │  Total ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0.0% │
    │  Power 485W   VRAM 13.5G   Elapsed   1:30   ETA   1:30 │
    ╰────────────────────────────────────────────────────────╯

    - All samples run in parallel (Disjoint Union architecture)
    - Alpha values are batched based on GPU memory
    - Total progress: completed batches / total batches (discrete, not interpolated)
    - ETA: based on avg_batch_time × remaining_batches
    """

    def __init__(
        self,
        num_alphas: int,
        steps_per_alpha: int,
        batch_size: int = 1,
        initial_estimate: float = None,
        num_batches: int = 1,  # Number of alpha batches
        batch_assignments: List[tuple] = None,  # Phase 4: [(start, end, alpha_max), ...] for physics-aware ETA
    ):
        self.num_alphas = num_alphas
        self.steps_per_alpha = steps_per_alpha
        self.batch_size = batch_size
        self.initial_estimate = initial_estimate
        self.gpu_monitor = GPUMonitor()

        # Batch tracking state
        self._total_batches = num_batches
        self._current_batch_idx = 0  # Current batch (0-indexed internally, displayed as 1-indexed)
        self._completed_batches = 0

        # Timing state
        self._total_start_time: float = None
        self._batch_start_time: float = None
        self._completed_alphas = 0
        self._alpha_times: List[float] = []
        self._batch_times: List[float] = []  # Time per batch for ETA
        self._current_alpha = 0.0
        self._current_batch_alphas: List[float] = []
        self._current_step = 0
        self._frame = 0

        # Phase 4: Physics-aware ETA estimator
        self._physics_eta = None
        if batch_assignments:
            try:
                from .physics_eta import PhysicsAwareETA
                self._physics_eta = PhysicsAwareETA(batch_assignments)
            except ImportError:
                pass  # Fall back to simple ETA

        if not RICH_AVAILABLE:
            self._live = None
            return

        self._console = Console()
        self._live = None


    def _render(self):
        """
        Render the Dynamic Capsule panel - Style A (Clean Cyan).

        Layout:
        - Row 1: Step count + Alpha range + it/s + batch countdown
        - Row 2: Step progress bar + percentage
        - Row 3: Total progress bar + batch count (e.g., 0/9)
        - Row 4: Power + VRAM + Elapsed + ETA (total)
        
        Color scheme:
        - Static labels = white (default)
        - Dynamic values = cyan
        """
        # Data Calculations
        pct_step = self._current_step / self.steps_per_alpha if self.steps_per_alpha > 0 else 0
        # Batch progress: completed batches + current step fraction
        batch_progress = self._completed_batches + (pct_step if self._completed_batches < self._total_batches else 0)
        pct_batch = batch_progress / self._total_batches if self._total_batches > 0 else 0

        # Timing
        elapsed = time.time() - self._total_start_time if self._total_start_time else 0
        eta = self._estimate_eta()

        # Calculate it/s (iterations per second for current batch)
        batch_elapsed = time.time() - self._batch_start_time if self._batch_start_time else 0
        if batch_elapsed > 0 and self._current_step > 0:
            it_per_sec = self._current_step / batch_elapsed
        else:
            it_per_sec = 0

        # Calculate batch countdown: elapsed / total_estimated
        # Left side: current elapsed time (starts at 0, increases)
        # Right side: estimated total batch time (stable)
        if batch_elapsed > 0 and self._current_step > 0:
            step_pct = self._current_step / self.steps_per_alpha
            if step_pct > 0.01:  # Avoid division issues at very start
                batch_total_estimated = batch_elapsed / step_pct
            else:
                batch_total_estimated = 0
        else:
            batch_total_estimated = 0

        def fmt_time(seconds):
            if seconds < 3600:
                return f"{int(seconds)//60}:{int(seconds)%60:02d}"
            h = int(seconds) // 3600
            m = (int(seconds) % 3600) // 60
            s = int(seconds) % 60
            return f"{h}:{m:02d}:{s:02d}"

        elapsed_str = fmt_time(elapsed)
        eta_str = fmt_time(eta) if eta >= 0 else "--:--"
        batch_elapsed_str = fmt_time(batch_elapsed)
        batch_total_str = fmt_time(batch_total_estimated)

        # GPU Stats
        gpu_status = self.gpu_monitor.get_status()
        if gpu_status:
            power = gpu_status['power_draw']
            memory = gpu_status['memory_used']
        else:
            power = 0
            memory = 0

        # Alpha range string
        if self._current_batch_alphas and len(self._current_batch_alphas) > 1:
            alpha_str = f"{self._current_batch_alphas[0]:.1f}-{self._current_batch_alphas[-1]:.1f}"
        else:
            alpha_str = f"{self._current_alpha:.2f}"

        # Moon phase spinner (8 frames, smooth animation)
        moon_frames = "🌑🌒🌓🌔🌕🌖🌗🌘"
        spinner = moon_frames[self._frame // 3 % 8]

        # Build progress bar as Text object for accurate width
        def make_bar_text(pct, width, filled_char, empty_char, color):
            pct = max(0.0, min(1.0, pct))
            filled = int(width * pct)
            empty = width - filled
            text = Text()
            text.append(filled_char * filled, style=color)
            text.append(empty_char * empty, style="dim")
            return text

        # Build grid
        grid = Table.grid(padding=0)
        grid.add_column()

        # Row 1: Step + α + it/s + batch countdown
        # Colors: labels=white, dynamic values=cyan
        it_s_val = f"{it_per_sec:.1f}" if it_per_sec > 0 else "--"
        grid.add_row(Text.from_markup(
            f" {spinner}  Step [cyan]{self._current_step}[/]/{self.steps_per_alpha}  "
            f"α [cyan]{alpha_str}[/]  [cyan]{it_s_val}[/]it/s  "
            f"[cyan]{batch_elapsed_str}[/]/{batch_total_str}"
        ))

        # Row 2: Step progress (within current batch)
        # Percentage: number=cyan, %=white
        row2 = Text(" Step  ")
        row2.append_text(make_bar_text(pct_step, 40, "━", "━", "cyan"))
        row2.append(f" {pct_step*100:5.1f}", style="cyan")
        row2.append("%")
        grid.add_row(row2)

        # Row 3: Total progress (batch count: completed=cyan, /total=white)
        row3 = Text(" Total ")
        row3.append_text(make_bar_text(pct_batch, 40, "━", "━", "cyan"))
        row3.append(f"  {self._completed_batches}", style="cyan")
        row3.append(f"/{self._total_batches}")
        grid.add_row(row3)

        # Row 4: GPU Metrics + Elapsed + ETA
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

    def _estimate_eta(self) -> float:
        """Estimate remaining time based on batch timing.
        
        Phase 4 Optimization: Uses PhysicsAwareETA if available, which models
        computational complexity as proportional to α_max in each batch.
        Falls back to simple avg_batch_time estimation otherwise.
        """
        if not self._total_start_time:
            return self.initial_estimate if self.initial_estimate else 0

        # Phase 4: Use physics-aware ETA if available
        if self._physics_eta and self._completed_batches > 0:
            # Physics ETA uses α_max-weighted workload for accurate estimation
            eta, _ = self._physics_eta.end_batch(self._completed_batches - 1)
            return eta

        total_elapsed = time.time() - self._total_start_time

        # If no batches completed yet, estimate from current batch progress
        if self._completed_batches == 0:
            if self._batch_start_time and self._current_step > 0:
                step_elapsed = time.time() - self._batch_start_time
                step_pct = self._current_step / self.steps_per_alpha
                if step_pct > 0.05:  # Wait for 5% progress before estimating
                    estimated_batch_time = step_elapsed / step_pct
                    remaining_in_current = estimated_batch_time * (1 - step_pct)
                    remaining_batches = self._total_batches - 1
                    return remaining_in_current + estimated_batch_time * remaining_batches
            return self.initial_estimate if self.initial_estimate else 0

        # Fallback: Use completed batch times for estimation
        if self._batch_times:
            avg_time_per_batch = sum(self._batch_times) / len(self._batch_times)
        else:
            avg_time_per_batch = total_elapsed / self._completed_batches

        remaining_batches = self._total_batches - self._completed_batches

        # Estimate remaining time in current batch
        remaining_in_current = 0
        if remaining_batches > 0 and self._batch_start_time and self._current_step > 0:
            step_elapsed = time.time() - self._batch_start_time
            step_pct = self._current_step / self.steps_per_alpha
            if step_pct > 0:
                estimated_batch_time = step_elapsed / step_pct
                remaining_in_current = estimated_batch_time * (1 - step_pct)
                remaining_batches -= 1  # Current batch partially counted

        return avg_time_per_batch * remaining_batches + remaining_in_current


    def start(self):
        """Start progress display."""
        if not RICH_AVAILABLE:
            return

        self._total_start_time = time.time()
        self._batch_start_time = time.time()
        self._current_step = 0
        self._frame = 0

        if self.initial_estimate:
             # Just a small log before we start
            est_str = str(timedelta(seconds=int(self.initial_estimate)))
            self._console.print(f"[dim]Estimated total time: ~{est_str}[/dim]")

        self._live = Live(
            self._render(),
            refresh_per_second=10,
            console=self._console,
            transient=False,
            # vertical_overflow="crop", # Removed to avoid cutting off the panel
        )
        self._live.start()

    def start_batch(self, batch_idx: int, batch_alphas: List[float], num_batches: int = None):
        """Start tracking a new batch.

        Args:
            batch_idx: 0-indexed batch number
            batch_alphas: List of alpha values in this batch
            num_batches: Optional total number of batches (updates _total_batches if provided)
        """
        if not RICH_AVAILABLE or not self._live:
            return

        # Update total batches if provided (dynamic batching)
        if num_batches is not None:
            self._total_batches = num_batches
        
        # Mark previous batch as complete if we're moving to a new batch
        if batch_idx > self._current_batch_idx:
            self._completed_batches = batch_idx
        
        self._current_batch_idx = batch_idx
        self._current_batch_alphas = batch_alphas
        self._current_alpha = batch_alphas[0] if batch_alphas else 0.0
        self._batch_start_time = time.time()
        self._current_step = 0
        self._frame += 1
        self._live.update(self._render())

    def start_alpha(self, alpha: float, batch_alphas: List[float] = None):
        """Start tracking a new alpha or batch (legacy interface).

        For backward compatibility. Prefer start_batch() for new code.
        """
        if not RICH_AVAILABLE or not self._live:
            return

        self._current_alpha = alpha
        self._current_batch_alphas = batch_alphas or [alpha]
        self._batch_start_time = time.time()
        self._current_step = 0
        self._frame += 1
        self._live.update(self._render())

    def update_step(self, current: int, total: int = None):
        """Update step progress within current batch."""
        if not RICH_AVAILABLE or not self._live:
            return

        self._current_step = current
        if total:
            self.steps_per_alpha = total
        self._frame += 1
        self._live.update(self._render())

    def finish_batch(self, metrics: Dict = None):
        """Finish tracking current batch and record timing.

        Call this after a batch completes to update batch progress.
        """
        if not RICH_AVAILABLE or not self._live:
            return

        if self._batch_start_time:
            batch_time = time.time() - self._batch_start_time
            self._batch_times.append(batch_time)

            # Also track per-alpha times for legacy compatibility
            num_in_batch = len(self._current_batch_alphas) if self._current_batch_alphas else 1
            time_per_alpha = batch_time / num_in_batch
            for _ in range(num_in_batch):
                self._alpha_times.append(time_per_alpha)

        self._completed_batches += 1
        self._completed_alphas += len(self._current_batch_alphas) if self._current_batch_alphas else 1
        # Step progress should show 100% when batch completes
        self._current_step = self.steps_per_alpha
        self._frame += 1
        self._live.update(self._render())

    def update_sample(self, current: int, total: int = None):
        """Update sample progress (legacy interface).

        Deprecated: Use finish_batch() for new code.
        Maps to batch completion for compatibility.
        """
        if not RICH_AVAILABLE or not self._live:
            return

        # Map sample to batch for backward compatibility
        self._completed_batches = current
        if total:
            self._total_batches = total
        # Step progress should show 100% when sample completes
        self._current_step = self.steps_per_alpha
        self._frame += 1
        self._live.update(self._render())

    def finish_alpha(self, metrics: Dict = None):
        """Finish tracking current batch (legacy interface).

        Deprecated: Use finish_batch() for new code.
        """
        self.finish_batch(metrics)

    def stop(self):
        """Stop progress display."""
        if self._live:
            self._live.stop()


class ExperimentProgress:
    """Legacy progress tracking - kept for compatibility."""

    def __init__(
        self,
        num_alphas: int,
        steps_per_alpha: int,
        samples: int,
        initial_estimate_seconds: float = None,
    ):
        self.num_alphas = num_alphas
        self.steps_per_alpha = steps_per_alpha
        self.samples = samples
        self.initial_estimate = initial_estimate_seconds

        self.gpu_monitor = GPUMonitor()
        self.console = Console() if RICH_AVAILABLE else None

        self.start_time: float = None
        self.alpha_start_time: float = None
        self.alpha_times: list = []
        self.completed_alphas = 0
        self.current_alpha: float = None

        self._progress: Progress = None
        self._alpha_task = None

    def start(self):
        """Start progress tracking."""
        self.start_time = time.time()

        if RICH_AVAILABLE:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=30),
                TaskProgressColumn(),
                TextColumn("[dim]{task.fields[status]}[/dim]"),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
            )
            self._alpha_task = self._progress.add_task(
                "Alpha Sweep",
                total=self.num_alphas,
                status=""
            )
            self._progress.start()

            if self.initial_estimate:
                 # Helper for time format
                est_str = str(timedelta(seconds=int(self.initial_estimate)))
                self.console.print(f"[dim]Estimated total time: ~{est_str}[/dim]")

    def start_alpha(self, alpha: float):
        """Start tracking a new alpha value."""
        self.current_alpha = alpha
        self.alpha_start_time = time.time()

        if self._progress:
            self._progress.update(self._alpha_task, status=f"α={alpha:.2f}")

    def finish_alpha(self, metrics: Dict[str, float] = None):
        """Finish tracking current alpha."""
        if self.alpha_start_time:
            alpha_duration = time.time() - self.alpha_start_time
            self.alpha_times.append(alpha_duration)

        self.completed_alphas += 1

        if self._progress:
            self._progress.update(self._alpha_task, advance=1)

    def finish(self) -> float:
        """Finish progress tracking and return total time."""
        total_time = time.time() - self.start_time if self.start_time else 0

        if self._progress:
            self._progress.stop()

        return total_time


class ProgressManager:
    """Manages progress display for experiments."""

    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None

    def print(self, *args, **kwargs):
        """Print message."""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args)

    def print_header(self, title: str, config_info: dict = None):
        """Print experiment header."""
        if self.use_rich:
            self.console.print()
            self.console.print(Panel(
                f"[bold cyan]{title}[/bold cyan]",
                expand=False,
                border_style="cyan"
            ))
            if config_info:
                table = Table(show_header=False, box=None, padding=(0, 2))
                for key, value in config_info.items():
                    table.add_row(f"[dim]{key}:[/dim]", str(value))
                self.console.print(table)
            self.console.print()
        else:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
            if config_info:
                for key, value in config_info.items():
                    print(f"  {key}: {value}")
            print()

    def print_completion(self, total_time: float, result_path: str = None):
        """Print completion message."""
        if self.use_rich:
            self.console.print()
            self.console.print(f"[bold green]✓ Complete![/bold green] Total time: {total_time:.1f}s")
            if result_path:
                self.console.print(f"  Results saved to: [cyan]{result_path}[/cyan]")
            self.console.print()
        else:
            print(f"\nComplete! Total time: {total_time:.1f}s")
            if result_path:
                print(f"Results saved to: {result_path}")
            print()

    @contextmanager
    def progress(self, description: str, total: int):
        """Context manager for progress bar."""
        if self.use_rich:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            ) as progress:
                task = progress.add_task(description, total=total)
                yield lambda n=1, **kw: progress.update(task, advance=n, **kw)
        else:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=total, desc=description, mininterval=1.0)
                yield lambda n=1, **kw: pbar.update(n)
                pbar.close()
            except ImportError:
                current = [0]
                def update(n=1, **kw):
                    current[0] += n
                    if current[0] % max(1, total // 10) == 0:
                        print(f"  {description}: {current[0]}/{total}")
                yield update

    @contextmanager
    def training_progress(
        self,
        num_alphas: int,
        steps_per_alpha: int,
        initial_estimate: float = None,
    ):
        """Context manager for training progress with GPU monitoring."""
        exp_progress = ExperimentProgress(
            num_alphas=num_alphas,
            steps_per_alpha=steps_per_alpha,
            samples=1,
            initial_estimate_seconds=initial_estimate,
        )

        try:
            exp_progress.start()
            yield exp_progress
        finally:
            exp_progress.finish()

    def create_experiment_progress(
        self,
        num_alphas: int,
        steps_per_alpha: int,
        samples: int = 1,
        initial_estimate: float = None,
    ) -> ExperimentProgress:
        """Create an ExperimentProgress instance."""
        return ExperimentProgress(
            num_alphas=num_alphas,
            steps_per_alpha=steps_per_alpha,
            samples=samples,
            initial_estimate_seconds=initial_estimate,
        )


# Global instance
_progress_manager: Optional[ProgressManager] = None


def get_progress_manager() -> ProgressManager:
    """Get or create the global progress manager."""
    global _progress_manager
    if _progress_manager is None:
        _progress_manager = ProgressManager()
    return _progress_manager
