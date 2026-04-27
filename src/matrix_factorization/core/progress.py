"""
Progress display using rich library with GPU monitoring.

Modern UI with 'Dynamic Capsule' aesthetic.
"""

from contextlib import contextmanager
from typing import Optional, Dict, List
import time
from datetime import timedelta
from collections import deque

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .gpu_monitor import GPUMonitor
from ..ui.progress_renderer import ProgressRenderer

class UnifiedProgress:
    """
    Modern progress display with 'Dynamic Capsule' layout - Style A (Clean Cyan).
    Logic layer that delegates rendering to ProgressRenderer.
    """

    def __init__(
        self,
        num_alphas: int,
        steps_per_alpha: int,
        batch_size: int = 1,
        initial_estimate: float = None,
        num_batches: int = 1,
        batch_assignments: List[tuple] = None,
    ):
        self.num_alphas = num_alphas
        self.steps_per_alpha = steps_per_alpha
        self.batch_size = batch_size
        self.initial_estimate = initial_estimate
        self.gpu_monitor = GPUMonitor()

        # Batch tracking state
        self._total_batches = num_batches
        self._current_batch_idx = 0
        self._completed_batches = 0

        # Timing state
        self._total_start_time: float = None
        self._batch_start_time: float = None
        self._completed_alphas = 0
        self._alpha_times: List[float] = []
        self._batch_times: List[float] = []
        self._current_alpha = 0.0
        self._current_batch_alphas: List[float] = []
        self._current_step = 0
        self._completed_points = 0
        self._total_points = num_batches
        self._context_lines: List[str] = []
        
        # Metrics storage
        self._current_metrics: Dict[str, float] = {}

        # Phase 4: Physics-aware ETA estimator
        self._physics_eta = None
        self._rate_history = deque(maxlen=50)
        self._last_history_update = 0
        self._current_it_per_sec = 0.0
        
        # Batch Total Time Throttling
        self._last_batch_total_update = 0
        self._cached_batch_total = -1.0
                
        # Global ETA Throttling
        self._last_eta_update = 0
        self._cached_eta = -1.0
                
        if batch_assignments:
            try:
                from .physics_eta import PhysicsAwareETA
                self._physics_eta = PhysicsAwareETA(batch_assignments)
            except ImportError:
                pass

        if not RICH_AVAILABLE:
            self._live = None
            return

        # Initialize Renderer
        self._renderer = ProgressRenderer(Console())
        self._live = None

    def set_plan(self, batch_assignments: List[tuple], total_batches: int = None, algorithm_key: str = None):
        """Update execution plan and initialize physics-aware ETA."""
        if total_batches:
            self._total_batches = total_batches
        self._total_points = max(self._total_points, self._total_batches)
            
        if batch_assignments:
            try:
                from .physics_eta import create_eta_estimator
                self._physics_eta = create_eta_estimator(
                    alpha_values=[],
                    dynamic_batches=batch_assignments,
                    algorithm_name=algorithm_key or "bigamp_spreading"
                )
            except ImportError:
                pass

    def set_context(self, context_lines: List[str] = None, completed_points: int = None, total_points: int = None):
        """Update scan context shown in the progress panel."""
        if context_lines is not None:
            self._context_lines = [str(line) for line in context_lines if line]
        if completed_points is not None:
            self._completed_points = max(0, int(completed_points))
        if total_points is not None:
            self._total_points = max(0, int(total_points))

    def _render(self):
        """Build state dict and delegate to renderer."""
        # 1. Gather Data
        now = time.time()
        elapsed = now - self._total_start_time if self._total_start_time else 0
        batch_elapsed = now - self._batch_start_time if self._batch_start_time else 0
        eta = self._estimate_eta()
        
        # Rate Limiting Logic and Calculations
        if now - self._last_history_update > 1.0:
            self._rate_history.append((now, self._current_step))
            self._last_history_update = now
            
        self._current_it_per_sec = 0.0
        if batch_elapsed >= 5.0 and len(self._rate_history) > 1:
            t_old, s_old = self._rate_history[0]
            dt = now - t_old
            ds = self._current_step - s_old
            if dt > 0.5:
                self._current_it_per_sec = ds / dt
                
        if self._current_it_per_sec < 1e-3 and batch_elapsed >= 5.0:
             self._current_it_per_sec = self._current_step / batch_elapsed
             
        it_per_sec = self._current_it_per_sec

        # Batch Prediction Logic
        batch_total_estimated = -1.0
        if elapsed < 10.0:
            batch_total_estimated = -1.0
        elif batch_elapsed > 0:
            if it_per_sec > 0.1:
                remaining_steps = max(0, self.steps_per_alpha - self._current_step)
                remaining_time = remaining_steps / it_per_sec
                batch_total_estimated = batch_elapsed + remaining_time
            elif self._batch_times:
                 batch_total_estimated = sum(self._batch_times) / len(self._batch_times)
            elif self._current_step > 0:
                 step_pct = max(1e-6, self._current_step / self.steps_per_alpha)
                 if step_pct < 0.01:
                     batch_total_estimated = -1.0
                 else:
                     batch_total_estimated = batch_elapsed / step_pct

        if now - self._last_batch_total_update > 1.0 or self._cached_batch_total <= 0:
            self._cached_batch_total = batch_total_estimated
            self._last_batch_total_update = now
        else:
            batch_total_estimated = self._cached_batch_total

        # Helpers
        def fmt_time(seconds):
            if seconds < 0: return "--:--"
            if seconds < 3600:
                return f"{int(seconds)//60}:{int(seconds)%60:02d}"
            h = int(seconds) // 3600
            m = (int(seconds) % 3600) // 60
            s = int(seconds) % 60
            return f"{h}:{m:02d}:{s:02d}"

        # GPU Stats
        gpu_status = self.gpu_monitor.get_status()
        power = gpu_status['power_draw'] if gpu_status else 0
        memory = gpu_status['memory_used'] if gpu_status else 0

        
        # [UI FIX] Check for internal algorithm batching info (e.g. from Tensor Parallel)
        # Prioritize algorithm's view of "Physical Batches" over Runner's "Logical Batches"
        display_alphas = self._current_batch_alphas or [self._current_alpha]
        display_completed_batches = self._completed_batches
        display_total_batches = self._total_batches
        
        if 'batch_info' in self._current_metrics:
            b_info = self._current_metrics['batch_info']
            # Algorithm reports 1-based index
            algo_batch_idx = int(b_info.get('batch_idx', 1))
            algo_total = int(b_info.get('total_batches', 1))
            algo_alphas = b_info.get('batch_alphas', [])
            
            # Override display state
            display_completed_batches = algo_batch_idx - 1
            display_total_batches = algo_total
            if algo_alphas:
                display_alphas = algo_alphas

        # Construct State
        state = {
            'step_progress': (self._current_step, self.steps_per_alpha),
            'batch_progress': (display_completed_batches, display_total_batches),
            'point_progress': (self._completed_points, self._total_points),
            'alphas': display_alphas,
            'context_lines': self._context_lines,
            'timing': (fmt_time(elapsed), fmt_time(eta), fmt_time(batch_elapsed), fmt_time(batch_total_estimated)),
            'throughput': it_per_sec,
            'gpu': (power, memory),
            'metrics': self._current_metrics
        }
        
        return self._renderer.render_capsule_panel(state)


    def _estimate_eta(self) -> float:
        """Estimate remaining time based on batch timing."""
        now = time.time()
        # Return cached value if within 1s window (smooth countdown)
        if now - self._last_eta_update < 1.0 and self._cached_eta >= 0:
            return self._cached_eta

        if not self._total_start_time:
            return self.initial_estimate if self.initial_estimate else -1
            
        elapsed = time.time() - self._total_start_time
        if elapsed < 10.0:
            return -1.0

        def _return_and_cache(val):
            self._cached_eta = val
            self._last_eta_update = now
            return val

        if self._physics_eta:
            step_pct = self._current_step / self.steps_per_alpha if self.steps_per_alpha > 0 else 0
            eta, _ = self._physics_eta.get_status(self._completed_batches, step_pct)
            return _return_and_cache(eta)

        if self._completed_batches == 0:
            if self._batch_start_time and self._current_step > 0:
                step_elapsed = time.time() - self._batch_start_time
                step_pct = self._current_step / self.steps_per_alpha
                if step_pct > 0.01:
                    estimated_batch_time = step_elapsed / step_pct
                    remaining_in_current = estimated_batch_time * (1 - step_pct)
                    remaining_batches = self._total_batches - 1
                    return _return_and_cache(remaining_in_current + estimated_batch_time * remaining_batches)
            
            return _return_and_cache(self.initial_estimate if self.initial_estimate else -1)

        # Fallback
        if self._batch_times:
            avg_time_per_batch = sum(self._batch_times) / len(self._batch_times)
        elif self._batch_start_time and self._current_step > 0:
            if self._current_it_per_sec > 0.01:
                avg_time_per_batch = max(1, self.steps_per_alpha) / self._current_it_per_sec
            else:
                 avg_time_per_batch = (self.initial_estimate or 0) / max(1, self._total_batches)
        else:
            avg_time_per_batch = (self.initial_estimate or 0) / max(1, self._total_batches)

        remaining_batches = self._total_batches - self._completed_batches
        remaining_in_current = 0
        if remaining_batches > 0 and self._batch_start_time and self._current_step > 0:
            step_elapsed = time.time() - self._batch_start_time
            step_pct = self._current_step / self.steps_per_alpha
            remaining_batches -= 1
            if step_pct > 0.01:
                estimated_batch_time = step_elapsed / step_pct
                remaining_in_current = estimated_batch_time * (1 - step_pct)
            else:
                remaining_in_current = avg_time_per_batch

        total_est = avg_time_per_batch * remaining_batches + remaining_in_current
        return _return_and_cache(total_est)


    def start(self):
        """Start progress display."""
        if not RICH_AVAILABLE:
            return

        self._total_start_time = time.time()
        self._batch_start_time = time.time()
        self._current_step = 0

        if self.initial_estimate:
            est_str = str(timedelta(seconds=int(self.initial_estimate)))
            self._renderer.console.print(f"[dim]Estimated total time: ~{est_str}[/dim]")

        self._live = Live(
            self._render(),
            refresh_per_second=10,
            console=self._renderer.console,
            transient=False,
        )
        self._live.start()

    def start_batch(self, batch_idx: int, batch_alphas: List[float], num_batches: int = None):
        """Start tracking a new batch."""
        if not RICH_AVAILABLE or not self._live:
            return

        if num_batches is not None:
            self._total_batches = num_batches
        
        if batch_idx > self._current_batch_idx:
            self._completed_batches = batch_idx
        
        self._current_batch_idx = batch_idx
        self._current_batch_alphas = batch_alphas
        self._current_alpha = batch_alphas[0] if batch_alphas else 0.0
        self._batch_start_time = time.time()
        self._current_step = 0
        
        if self._physics_eta:
            self._physics_eta.start_batch(batch_idx)
            
        self._live.update(self._render())

    def start_alpha(self, alpha: float, batch_alphas: List[float] = None):
        """Start tracking a new alpha or batch."""
        if not RICH_AVAILABLE or not self._live:
            return

        self._current_alpha = alpha
        self._current_batch_alphas = batch_alphas or [alpha]
        self._batch_start_time = time.time()
        self._current_step = 0
        self._live.update(self._render())

    def update_step(self, current: int, total: int = None, metrics: Dict = None):
        """Update step progress within current batch."""
        if not RICH_AVAILABLE or not self._live:
            return

        self._current_step = current
        if total:
            self.steps_per_alpha = total
            
        if metrics:
            self._current_metrics.update(metrics)
            
        self._live.update(self._render())

    def finish_batch(self, metrics: Dict = None):
        """Finish tracking current batch and record timing."""
        if not RICH_AVAILABLE or not self._live:
            return

        if self._batch_start_time:
            batch_time = time.time() - self._batch_start_time
            self._batch_times.append(batch_time)
            num_in_batch = len(self._current_batch_alphas) if self._current_batch_alphas else 1
            time_per_alpha = batch_time / num_in_batch
            for _ in range(num_in_batch):
                self._alpha_times.append(time_per_alpha)

        if self._physics_eta:
            self._physics_eta.end_batch(self._current_batch_idx)

        self._completed_batches += 1
        self._completed_alphas += len(self._current_batch_alphas) if self._current_batch_alphas else 1
        self._current_step = self.steps_per_alpha
        self._live.update(self._render())

    def update_sample(self, current: int, total: int = None):
        """Update sample progress (legacy interface)."""
        if not RICH_AVAILABLE or not self._live:
            return
        self._completed_batches = current
        if total:
            self._total_batches = total
        self._current_step = self.steps_per_alpha
        self._live.update(self._render())

    def finish_alpha(self, metrics: Dict = None):
        """Finish tracking current batch (legacy interface)."""
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


class ProgressBridge:
    """
    Bridge between ExperimentRunner events and UnifiedProgress UI.
    
    Transforms robust ProgressEvents into specific UI calls.
    """
    
    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        self.progress: Optional[UnifiedProgress] = None
        self.console = Console() if self.use_rich else None
        self._started = False
    
    def on_event(self, event):
        """Handle progress event (duck typing for ProgressEvent)."""
        if not self.use_rich:
            # Fallback for non-rich environments
            self._handle_legacy_event(event)
            return

        type_name = event.type.name if hasattr(event.type, 'name') else str(event.type)
        p = event.payload
        
        if type_name == 'EXPERIMENT_START':
            self._handle_start(p)
        elif type_name == 'BATCH_START':
            self._handle_batch_start(p)
        elif type_name == 'EXECUTION_PLAN':
            if self.progress:
                self.progress.set_plan(
                    batch_assignments=p.get('batches'),
                    total_batches=p.get('total_batches'),
                    algorithm_key=p.get('algorithm_key')
                )
                self._apply_context(p)
        elif type_name == 'STEP_UPDATE':
            if self.progress:
                self._apply_context(p)
                self.progress.update_step(p.get('step', 0), p.get('total'), metrics=p.get('metrics'))
        elif type_name == 'POINT_START':
            if self.progress:
                self._apply_context(p)
        elif type_name == 'POINT_COMPLETE':
            if self.progress:
                self._apply_context(p)
        elif type_name == 'BATCH_END':
            if self.progress:
                self._apply_context(p)
                self.progress.finish_batch()
        elif type_name == 'EXPERIMENT_END':
            if self.progress:
                self.progress.stop()
            self.progress = None
            self._started = False
            self._print_success(p.get('result'))
        elif type_name == 'ERROR':
            if self.progress:
                self.progress.stop()
            self.progress = None
            self._started = False
            if self.console:
                context = self._format_context_line(p.get("scan_context") or {})
                prefix = f" at {context}" if context else ""
                self.console.print(f"[bold red]Error{prefix}:[/bold red] {p.get('error')}")

    def _handle_start(self, p: Dict):
        """Initialize progress bar on start."""
        if self.progress is not None:
            self._apply_context(p)
            return
        if self.console:
            self.console.print(Panel(
                f"[bold cyan]Running {p.get('experiment_name', 'Experiment')}[/bold cyan]\n"
                f"[dim]{p.get('matrix')}, {p.get('scan')}[/dim]",
                border_style="cyan",
                expand=False
            ))
            
        self.progress = UnifiedProgress(
            num_alphas=1, 
            steps_per_alpha=1000,
            num_batches=1
        )
        self._apply_context(p)
        self.progress.start()
        self._started = True

    def _handle_batch_start(self, p: Dict):
        """Update UI for new batch."""
        if not self.progress:
            return
        self._apply_context(p)
            
        alphas = p.get('alpha_values', [])
        self.progress.start_batch(
            batch_idx=p.get('batch_idx', 0),
            batch_alphas=alphas,
            num_batches=p.get('total_batches', 1)
        )
        
        steps = p.get('steps_per_alpha')
        if steps:
            self.progress.steps_per_alpha = steps

    def _apply_context(self, p: Dict):
        if not self.progress:
            return
        context = p.get("scan_context") or {}
        lines = []
        if context:
            line = self._format_context_line(context)
            if line:
                lines.append(line)
            coordinates = context.get("coordinates")
            if isinstance(coordinates, dict) and coordinates:
                coord_line = ", ".join(f"{key}={value}" for key, value in coordinates.items())
                lines.append(coord_line)
        point_progress = p.get("point_progress") or context.get("point_progress") or {}
        self.progress.set_context(
            lines,
            completed_points=point_progress.get("completed") if isinstance(point_progress, dict) else None,
            total_points=point_progress.get("total") if isinstance(point_progress, dict) else p.get("total_points"),
        )

    @staticmethod
    def _format_context_line(context: Dict) -> str:
        if not isinstance(context, dict) or not context:
            return ""
        parts = []
        group_label = context.get("group_label")
        batch_label = context.get("batch_label")
        point_label = context.get("point_label")
        alpha_label = context.get("alpha_label")
        if group_label:
            parts.append(str(group_label))
        if batch_label:
            parts.append(str(batch_label))
        if point_label:
            parts.append(str(point_label))
        if alpha_label:
            parts.append(str(alpha_label))
        return " | ".join(parts)

    def _handle_legacy_event(self, event):
        """Simple print fallback."""
        type_name = str(event.type)
        p = event.payload
        
        if 'START' in type_name and 'BATCH' in type_name:
            n = p.get('batch_idx', 0) + 1
            total = p.get('total_batches', 1)
            print(f"Batch {n}/{total} started...")
        elif 'STEP' in type_name:
            step = p.get('step', 0)
            total = p.get('total', 100)
            if step % max(1, total // 10) == 0:
                print(f"  Step {step}/{total}")
        elif 'ERROR' in type_name:
             print(f"Error: {p.get('error')}")

    def _print_success(self, result):
        if self.console and result:
             path = getattr(result, 'result_path', 'memory')
             self.console.print("\n[bold green]Experiment Complete![/bold green]")
             self.console.print(f"Results saved to: [cyan]{path}[/cyan]\n")
