"""NVIDIA GPU monitoring for real-time status display and batch diagnostics."""

import subprocess
import threading
import time
from typing import Optional, Dict, Any, List


class GPUMonitor:
    """NVIDIA GPU status monitor using nvidia-smi."""

    def __init__(self):
        self.available = self._check_nvidia()
        self._cache_timeout = 2.0  # 2 seconds cache
        self._last_status = None
        self._last_query_time = 0

    def _check_nvidia(self) -> bool:
        """Check if GPU monitoring is available (nvidia-smi or torch.cuda)."""
        try:
            import torch
            if torch.cuda.is_available():
               return True
        except ImportError:
            pass

        try:
            result = subprocess.run(
                ['nvidia-smi', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def get_status(self) -> Optional[Dict]:
        """
        Get current GPU status with caching.
        
        Uses 0.5 second cache to avoid excessive subprocess calls.

        Returns:
            Dict with power_draw, power_limit, memory_used, memory_total
            or None if not available
        """
        import time
        try:
            import torch
        except ImportError:
            return None
        
        if not self.available:
            return None
        
        # Return cached result if within timeout
        current_time = time.time()
        if self._last_status is not None and (current_time - self._last_query_time) < self._cache_timeout:
            return self._last_status

        if not torch.cuda.is_available():
             return self._last_status
        
        # 1. Try pynvml (Best)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            
            self._last_status = {
                'power_draw': power,
                'power_limit': limit,
                'memory_used': mem.used / (1024**3),
                'memory_total': mem.total / (1024**3),
                'gpu_name': pynvml.nvmlDeviceGetName(handle),
            }
            self._last_query_time = current_time
            return self._last_status
        except (ImportError, Exception):
            pass
            
        # 2. Try torch.cuda (Good for memory, no power data)
        # For power, we return 0 or keep last known value
        try:
            device = torch.cuda.current_device()
            reserved = torch.cuda.memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
            
            # If we have previous status, keep power/name, update memory
            last = self._last_status or {}
            self._last_status = {
                'power_draw': last.get('power_draw', 0.0),
                'power_limit': last.get('power_limit', 0.0),
                'memory_used': reserved / (1024**3),
                'memory_total': total / (1024**3),
                'gpu_name': last.get('gpu_name', 'GPU'),
                'partial': True # Flag indicating partial data
            }
            self._last_query_time = current_time
            return self._last_status
        except Exception:
            pass

        # 3. Last Limit Subprocess (Only if > 5s since last update)
        # Don't run every 2s if it's slow
        if current_time - self._last_query_time > 5.0:
            try:
                import subprocess
                result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=power.draw,power.limit,memory.used,memory.total,gpu_name',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=1.0) # Reduced timeout

                if result.returncode != 0:
                    return self._last_status

                # Parse: "245.00, 350.00, 8192, 24576, NVIDIA RTX 4090"
                line = result.stdout.strip().split('\n')[0]  # First GPU
                values = [v.strip() for v in line.split(',')]

                if len(values) < 4:
                    return self._last_status

                self._last_status = {
                    'power_draw': float(values[0]),           # W
                    'power_limit': float(values[1]),          # W
                    'memory_used': float(values[2]) / 1024,   # GB
                    'memory_total': float(values[3]) / 1024,  # GB
                    'gpu_name': values[4] if len(values) > 4 else 'GPU',
                }
                self._last_query_time = current_time
                return self._last_status
                
            except (subprocess.TimeoutExpired, ValueError, IndexError, Exception):
                return self._last_status  # Return cached on error

        return self._last_status


    def format_status(self, compact: bool = True) -> str:
        """
        Format GPU status for display.

        Args:
            compact: If True, use compact format with icons

        Returns:
            Formatted string or empty string if not available
        """
        status = self.get_status()
        if not status:
            return ""

        if compact:
            return (
                f"🔋 {status['power_draw']:.0f}W/{status['power_limit']:.0f}W  "
                f"💾 {status['memory_used']:.1f}GB/{status['memory_total']:.0f}GB"
            )
        else:
            return (
                f"GPU: {status['gpu_name']}\n"
                f"  Power: {status['power_draw']:.0f}W / {status['power_limit']:.0f}W\n"
                f"  Memory: {status['memory_used']:.1f}GB / {status['memory_total']:.0f}GB"
            )

    def format_rich(self) -> str:
        """Format for Rich console with colors."""
        status = self.get_status()
        if not status:
            return ""

        # Color based on utilization
        power_pct = status['power_draw'] / status['power_limit']
        mem_pct = status['memory_used'] / status['memory_total']

        power_color = "green" if power_pct < 0.7 else ("yellow" if power_pct < 0.9 else "red")
        mem_color = "green" if mem_pct < 0.7 else ("yellow" if mem_pct < 0.9 else "red")

        return (
            f"[{power_color}]🔋 {status['power_draw']:.0f}W/{status['power_limit']:.0f}W[/]  "
            f"[{mem_color}]💾 {status['memory_used']:.1f}GB/{status['memory_total']:.0f}GB[/]"
        )


# Global instance
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor() -> GPUMonitor:
    """Get or create global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor


def parse_nvidia_smi_compute_sample(line: str) -> Optional[Dict[str, float]]:
    """Parse one nvidia-smi utilization/power/memory CSV row."""
    if not line:
        return None
    values = [item.strip().replace("%", "") for item in line.split(",")]
    if len(values) < 3:
        return None
    try:
        return {
            "gpu_utilization": float(values[0]),
            "power_draw_w": float(values[1]),
            "memory_used_gb": float(values[2]) / 1024.0,
        }
    except ValueError:
        return None


class GPUComputeSampler:
    """Step-windowed GPU utilization sampler for planner calibration metadata."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        target_gpu_utilization: float = 0.88,
        warmup_steps: int = 20,
        measure_steps: int = 100,
        sample_interval_s: float = 0.5,
        min_samples: int = 5,
        min_measure_seconds: float = 3.0,
        device_index: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self.target_gpu_utilization = float(target_gpu_utilization)
        self.warmup_steps = int(warmup_steps)
        self.measure_steps = int(measure_steps)
        self.sample_interval_s = float(sample_interval_s)
        self.min_samples = int(min_samples)
        self.min_measure_seconds = float(min_measure_seconds)
        self.device_index = int(device_index)
        self._samples: List[Dict[str, float]] = []
        self._running = False
        self._measuring = False
        self._completed_window = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._started_at = 0.0
        self._stopped_at = 0.0
        self._measurement_started_at = 0.0
        self._measurement_stopped_at = 0.0
        self._partial_reason = ""

    def __enter__(self) -> "GPUComputeSampler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def start(self) -> None:
        self._started_at = time.time()
        self._stopped_at = 0.0
        if not self.enabled:
            return
        self._samples.clear()
        self._running = True
        self._measuring = self.warmup_steps <= 0
        self._completed_window = False
        self._measurement_started_at = self._started_at if self._measuring else 0.0
        self._measurement_stopped_at = 0.0
        self._partial_reason = ""
        self._reset_torch_peak_memory()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def mark_step(self, step: int) -> None:
        if not self.enabled:
            return
        step_int = int(step)
        if self._completed_window:
            return
        if step_int > self.warmup_steps + self.measure_steps and self._measurement_requirements_met():
            self._measuring = False
            self._completed_window = True
            if self._measurement_started_at and not self._measurement_stopped_at:
                self._measurement_stopped_at = time.time()
            return
        should_measure = step_int > self.warmup_steps
        if should_measure and not self._measuring:
            self._measurement_started_at = time.time()
            self._append_sample(self._sample_once())
        self._measuring = should_measure

    def stop(self, *, steps_run: Optional[int] = None, configured_max_steps: Optional[int] = None) -> Dict[str, Any]:
        if self.enabled:
            with self._lock:
                has_samples = bool(self._samples)
            if not has_samples:
                self._append_sample(self._sample_once())
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(self.sample_interval_s * 2.0, 0.1))
        self._stopped_at = time.time()
        if self._measurement_started_at and not self._measurement_stopped_at:
            self._measurement_stopped_at = self._stopped_at
        return self.report(steps_run=steps_run, configured_max_steps=configured_max_steps)

    def report(self, *, steps_run: Optional[int] = None, configured_max_steps: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
        util_values = [sample["gpu_utilization"] for sample in samples if "gpu_utilization" in sample]
        power_values = [sample["power_draw_w"] for sample in samples if "power_draw_w" in sample]
        memory_values = [sample["memory_used_gb"] for sample in samples if "memory_used_gb" in sample]
        peak_allocated_gb = self._torch_peak_allocated_gb()
        elapsed = (self._stopped_at or time.time()) - (self._started_at or time.time())
        measured_elapsed = 0.0
        if self._measurement_started_at:
            measured_elapsed = (self._measurement_stopped_at or time.time()) - self._measurement_started_at
        steps = int(steps_run) if steps_run is not None else None
        partial = not bool(util_values)
        too_few_samples = bool(samples) and len(samples) < self.min_samples
        too_short = bool(samples) and measured_elapsed < self.min_measure_seconds
        small_kernel_limited = bool(self.enabled and not partial and (too_few_samples or too_short))
        if partial:
            measurement_quality = "partial"
        elif small_kernel_limited:
            measurement_quality = "small_kernel_limited"
        else:
            measurement_quality = "complete"
        return {
            "enabled": bool(self.enabled),
            "available": bool(samples) or peak_allocated_gb is not None,
            "partial": partial,
            "partial_reason": "" if util_values else (self._partial_reason or "nvidia_smi_unavailable"),
            "target_gpu_utilization": self.target_gpu_utilization,
            "warmup_steps": self.warmup_steps,
            "measure_steps": self.measure_steps,
            "sample_interval_s": self.sample_interval_s,
            "min_samples": self.min_samples,
            "min_measure_seconds": self.min_measure_seconds,
            "num_samples": len(samples),
            "measured_seconds": measured_elapsed,
            "small_kernel_limited": small_kernel_limited,
            "measurement_quality": measurement_quality,
            "mean_gpu_utilization": (
                sum(util_values) / len(util_values) / 100.0 if util_values else None
            ),
            "max_gpu_utilization": (
                max(util_values) / 100.0 if util_values else None
            ),
            "mean_power_w": sum(power_values) / len(power_values) if power_values else None,
            "max_memory_used_gb": max(memory_values) if memory_values else None,
            "peak_allocated_gb": peak_allocated_gb,
            "elapsed_seconds": elapsed,
            "steps_run": steps,
            "configured_max_steps": int(configured_max_steps) if configured_max_steps is not None else None,
            "sec_per_step": (elapsed / steps) if steps and steps > 0 else None,
            "metadata_only": True,
        }

    def _sample_loop(self) -> None:
        while self._running:
            if self._measuring:
                self._append_sample(self._sample_once())
            time.sleep(max(self.sample_interval_s, 0.05))

    def _append_sample(self, sample: Optional[Dict[str, float]]) -> None:
        if sample is None:
            return
        with self._lock:
            self._samples.append(sample)

    def _measurement_requirements_met(self) -> bool:
        with self._lock:
            num_samples = len(self._samples)
        if num_samples < self.min_samples:
            return False
        if not self._measurement_started_at:
            return False
        return (time.time() - self._measurement_started_at) >= self.min_measure_seconds

    def _sample_once(self) -> Optional[Dict[str, float]]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    "--query-gpu=utilization.gpu,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=0.4,
            )
            if result.returncode == 0:
                line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
                parsed = parse_nvidia_smi_compute_sample(line)
                if parsed is not None:
                    return parsed
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, IndexError):
            self._partial_reason = "nvidia_smi_unavailable"
        except Exception:
            self._partial_reason = "nvidia_smi_error"
        torch_memory = self._torch_current_memory_gb()
        if torch_memory is not None:
            return {"memory_used_gb": torch_memory}
        return None

    @staticmethod
    def _reset_torch_peak_memory() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            return

    @staticmethod
    def _torch_current_memory_gb() -> Optional[float]:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return float(torch.cuda.memory_reserved(torch.cuda.current_device())) / (1024 ** 3)
        except Exception:
            return None

    @staticmethod
    def _torch_peak_allocated_gb() -> Optional[float]:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return float(torch.cuda.max_memory_allocated(torch.cuda.current_device())) / (1024 ** 3)
        except Exception:
            return None
