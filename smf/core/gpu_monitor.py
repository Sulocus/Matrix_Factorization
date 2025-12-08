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
        self._cache_timeout = 0.5  # seconds
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
        Get current GPU status.

        Returns:
            Dict with power_draw, power_limit, memory_used, memory_total
            or None if not available
        """
        if not self.available:
            return None

        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=power.draw,power.limit,memory.used,memory.total,gpu_name',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return None

            # Parse: "245.00, 350.00, 8192, 24576, NVIDIA RTX 4090"
            line = result.stdout.strip().split('\n')[0]  # First GPU
            values = [v.strip() for v in line.split(',')]

            if len(values) < 4:
                return None

            return {
                'power_draw': float(values[0]),           # W
                'power_limit': float(values[1]),          # W
                'memory_used': float(values[2]) / 1024,   # GB
                'memory_total': float(values[3]) / 1024,  # GB
                'gpu_name': values[4] if len(values) > 4 else 'GPU',
            }
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return None

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
