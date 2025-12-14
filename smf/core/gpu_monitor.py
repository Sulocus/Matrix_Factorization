"""
NVIDIA GPU monitoring for real-time status display.

Shows power draw and memory usage during experiments.
"""

import subprocess
from typing import Optional, Dict


class GPUMonitor:
    """NVIDIA GPU status monitor using nvidia-smi."""

    def __init__(self):
        self.available = self._check_nvidia()
        self._cache_timeout = 2.0  # 2 seconds cache
        self._last_status = None
        self._last_query_time = 0

    def _check_nvidia(self) -> bool:
        """Check if nvidia-smi is available."""
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
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
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
