"""
Memory Guard for Smart Parallel Module.

Provides runtime GPU memory monitoring with warning and critical thresholds,
automatic OOM detection, and recovery mechanisms.
"""
from enum import Enum, auto
from dataclasses import dataclass
import threading
import time
import gc
import logging
from typing import Callable, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .parallel_coordinator import ParallelCoordinator

logger = logging.getLogger(__name__)


class MemoryEventType(Enum):
    """Types of memory events."""
    WARNING = auto()   # > 85% usage
    CRITICAL = auto()  # > 90% usage


class MemoryAbortException(Exception):
    """Raised when memory critical threshold is exceeded and abort is requested."""
    pass


@dataclass
class MemoryEvent:
    """Memory monitoring event."""
    type: MemoryEventType
    usage_ratio: float
    used_gb: float
    total_gb: float
    timestamp: float
    
    def __str__(self) -> str:
        return (
            f"MemoryEvent({self.type.name}: {self.usage_ratio:.1%} used, "
            f"{self.used_gb:.1f}/{self.total_gb:.1f} GB)"
        )


class MemoryGuard:
    """
    Runtime GPU memory monitoring guard.
    
    Monitors GPU memory usage in a background thread and triggers callbacks
    when usage exceeds warning or critical thresholds.
    
    Example:
        def on_warning(event):
            print(f"Warning: {event.usage_ratio:.0%} memory used")
        
        def on_critical(event):
            print(f"Critical: OOM imminent!")
            coordinator.abort_current_batch()
        
        guard = MemoryGuard(
            on_warning=on_warning,
            on_critical=on_critical,
        )
        
        guard.start()
        # ... run algorithm ...
        guard.stop()
    """
    
    # Default thresholds (can be overridden in constructor)
    DEFAULT_WARNING_THRESHOLD = 0.85  # Normal allocation upper limit
    DEFAULT_CRITICAL_THRESHOLD = 0.95  # Trigger batch recovery
    
    def __init__(
        self,
        on_warning: Optional[Callable[[MemoryEvent], None]] = None,
        on_critical: Optional[Callable[[MemoryEvent], None]] = None,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        check_interval: float = 0.5,
        warning_cooldown: float = 5.0,
    ):
        """
        Initialize memory guard.
        
        Args:
            on_warning: Callback for warning events (> warning_threshold)
            on_critical: Callback for critical events (> critical_threshold)
            warning_threshold: Ratio (0-1) for normal allocation limit
            critical_threshold: Ratio (0-1) for batch recovery trigger
            check_interval: Seconds between checks
            warning_cooldown: Seconds between repeated warnings
        """
        self.on_warning = on_warning
        self.on_critical = on_critical
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.warning_cooldown = warning_cooldown
        
        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # State
        self._last_warning_time = 0.0
        self._peak_usage = 0.0
        self._events: list = []
        self._is_running = False
        
        # ABORT MECHANISM: Set by on_critical, checked by algorithm
        self._abort_requested = threading.Event()
    
    def request_abort(self) -> None:
        """Request abort of current operation. Called from on_critical."""
        self._abort_requested.set()
        logger.warning("[MemoryGuard] ABORT REQUESTED - algorithm should check and stop")
    
    def check_abort(self) -> None:
        """Check if abort was requested. Raises MemoryAbortException if so.
        
        Call this in algorithm's step loop to enable OOM recovery.
        """
        if self._abort_requested.is_set():
            self._abort_requested.clear()  # Reset for next batch
            raise MemoryAbortException("Memory critical threshold exceeded, aborting batch")
    
    @property
    def abort_requested(self) -> bool:
        """Check if abort is pending without raising exception."""
        return self._abort_requested.is_set()
    
    def clear_abort(self) -> None:
        """Clear abort flag (call after successful recovery)."""
        self._abort_requested.clear()
    
    def start(self) -> None:
        """Start background monitoring thread."""
        if self._is_running:
            logger.warning("MemoryGuard already running")
            return
        
        self._stop_event.clear()
        self._peak_usage = 0.0
        self._events.clear()
        
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="MemoryGuard",
            daemon=True,
        )
        self._thread.start()
        self._is_running = True
        
        logger.debug(
            f"MemoryGuard started: warning={self.warning_threshold:.0%}, "
            f"critical={self.critical_threshold:.0%}"
        )
    
    def stop(self) -> dict:
        """
        Stop monitoring thread.
        
        Returns:
            Summary statistics
        """
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        self._is_running = False
        
        summary = {
            "peak_usage": self._peak_usage,
            "warning_count": sum(1 for e in self._events if e.type == MemoryEventType.WARNING),
            "critical_count": sum(1 for e in self._events if e.type == MemoryEventType.CRITICAL),
            "events": self._events.copy(),
        }
        
        logger.debug(f"MemoryGuard stopped: peak={self._peak_usage:.1%}")
        
        return summary
    
    @property
    def is_running(self) -> bool:
        """Check if guard is currently monitoring."""
        return self._is_running
    
    @property
    def peak_usage_ratio(self) -> float:
        """Get peak memory usage ratio observed."""
        return self._peak_usage
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        while not self._stop_event.is_set():
            try:
                status = self._get_memory_status()
                if status is None:
                    time.sleep(self.check_interval)
                    continue
                
                usage_ratio = status['used'] / status['total']
                self._peak_usage = max(self._peak_usage, usage_ratio)
                
                # Check critical threshold
                if usage_ratio >= self.critical_threshold:
                    event = MemoryEvent(
                        type=MemoryEventType.CRITICAL,
                        usage_ratio=usage_ratio,
                        used_gb=status['used'],
                        total_gb=status['total'],
                        timestamp=time.time(),
                    )
                    self._events.append(event)
                    logger.warning(f"CRITICAL memory event: {event}")
                    
                    if self.on_critical:
                        try:
                            self.on_critical(event)
                        except Exception as e:
                            logger.error(f"Critical callback error: {e}")
                
                # Check warning threshold (with cooldown)
                elif usage_ratio >= self.warning_threshold:
                    now = time.time()
                    if now - self._last_warning_time >= self.warning_cooldown:
                        event = MemoryEvent(
                            type=MemoryEventType.WARNING,
                            usage_ratio=usage_ratio,
                            used_gb=status['used'],
                            total_gb=status['total'],
                            timestamp=now,
                        )
                        self._events.append(event)
                        logger.warning(f"Memory warning: {event}")
                        
                        if self.on_warning:
                            try:
                                self.on_warning(event)
                            except Exception as e:
                                logger.error(f"Warning callback error: {e}")
                        
                        self._last_warning_time = now
                
            except Exception as e:
                # Monitoring should never crash the main program
                logger.debug(f"Monitor loop error (ignored): {e}")
            
            time.sleep(self.check_interval)
    
    def _get_memory_status(self) -> Optional[dict]:
        """Get current GPU memory status."""
        if not torch.cuda.is_available():
            return None
        
        # 1. Try pynvml (Most accurate & efficient)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_id = torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'used': info.used / (1024**3),
                'total': info.total / (1024**3),
            }
        except (ImportError, Exception):
            pass
            
        # 2. Try torch.cuda (Fastest, in-process)
        # Note: memory_reserved is what actually occupies VRAM from OS perspective
        try:
            device = torch.cuda.current_device()
            reserved = torch.cuda.memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
            return {
                'used': reserved / (1024**3),
                'total': total / (1024**3),
            }
        except Exception as e:
            logger.debug(f"torch.cuda memory check failed: {e}")
            
        # 3. Fallback to nvidia-smi (Slow subprocess - strictly last resort)
        # Only use if we really can't get data otherwise
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                 '--format=csv,noheader,nounits', '-i', str(torch.cuda.current_device())],
                capture_output=True, text=True, timeout=1.0
            )
            if result.returncode == 0:
                used_mb, total_mb = map(float, result.stdout.strip().split(','))
                return {
                    'used': used_mb / 1024,
                    'total': total_mb / 1024,
                }
        except Exception:
            pass
            
        return None


class OOMRecoveryHandler:
    """
    Automatic OOM recovery handler.
    
    Works with ParallelCoordinator to automatically recover from memory
    pressure by aborting current batch, cleaning up, and replanning.
    
    Example:
        coordinator = ParallelCoordinator(...)
        handler = OOMRecoveryHandler(coordinator)
        
        guard = MemoryGuard(on_critical=handler.handle_critical)
        coordinator.set_memory_guard(guard)
    """
    
    def __init__(
        self,
        coordinator: 'ParallelCoordinator',
        max_retries: int = 3,
        reduction_factor: float = 0.7,
    ):
        """
        Initialize recovery handler.
        
        Args:
            coordinator: The ParallelCoordinator to control
            max_retries: Maximum recovery attempts
            reduction_factor: Memory reduction factor per retry
        """
        self.coordinator = coordinator
        self.max_retries = max_retries
        self.reduction_factor = reduction_factor
        
        self.retry_count = 0
        self._lock = threading.Lock()
    
    def handle_critical(self, event: MemoryEvent) -> None:
        """
        Handle critical memory event with batch recovery.
        
        When memory exceeds 95%, trigger batch recovery:
        - Abort current batch
        - Cleanup GPU memory
        - Continue with next batch
        
        Args:
            event: The critical memory event
        """
        with self._lock:
            logger.error(
                f"[OOMRecoveryHandler] CRITICAL: {event.usage_ratio:.1%} used! "
                f"({event.used_gb:.1f}/{event.total_gb:.1f} GB)"
            )
            
            # Batch recovery mode
            print(f"\n⚠️ OOM PROTECTION: Memory at {event.usage_ratio:.1%}")
            print("🔄 Initiating batch recovery...")
            
            # 0. Request abort - algorithm should check this flag
            if self.coordinator._memory_guard:
                self.coordinator._memory_guard.request_abort()
            
            # 1. Abort current batch
            self.coordinator.abort_current_batch()
            
            # 2. Force cleanup
            self._force_cleanup()
            
            # 3. Record for calibration
            if self.coordinator.estimator:
                self.coordinator.estimator.record_oom_event(event.usage_ratio)
            
            # 4. Increment retry counter for tracking
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                logger.info(
                    f"Batch recovery initiated ({self.retry_count}/{self.max_retries})"
                )
                print(f"   Recovery attempt {self.retry_count}/{self.max_retries}")
            else:
                # Max retries exceeded - let the exception propagate
                raise RuntimeError(
                    f"OOM recovery failed after {self.max_retries} retries. "
                    f"Peak usage: {event.usage_ratio:.1%}. "
                    f"Consider reducing problem size or using 'smf resume' to continue."
                )
    
    def reset(self) -> None:
        """Reset retry counter."""
        self.retry_count = 0
    
    def _force_cleanup(self) -> None:
        """Force GPU memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
        # Wait for cleanup to take effect
        time.sleep(1.0)
        
        logger.debug("Forced memory cleanup complete")


def create_monitored_guard(
    coordinator: 'ParallelCoordinator',
    config: Optional['AllocationConfig'] = None,
) -> MemoryGuard:
    """
    Factory function to create a fully configured MemoryGuard.
    
    Args:
        coordinator: ParallelCoordinator instance
        config: AllocationConfig with thresholds (uses coordinator's if None)
        
    Returns:
        Configured MemoryGuard with OOM recovery
    """
    if config is None:
        config = coordinator.config
    
    # Create recovery handler
    recovery_handler = OOMRecoveryHandler(coordinator)
    
    # Create guard with thresholds from config
    guard = MemoryGuard(
        on_warning=lambda e: logger.warning(f"Memory warning: {e}"),
        on_critical=recovery_handler.handle_critical,
        warning_threshold=config.warning_threshold,
        critical_threshold=config.critical_threshold,
    )
    
    # Link to coordinator
    coordinator.set_memory_guard(guard)
    
    return guard
