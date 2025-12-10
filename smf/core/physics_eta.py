"""
Physics-aware ETA estimation for BiG-AMP training.

Complexity model: T ∝ S × C_max = S × α × M × N1
Since S, M, N1 are constants within a run, T ∝ α (max alpha in each batch).

This module provides accurate ETA estimation that accounts for the varying
computational load of different alpha batches, which is essential after
the Phase 2 optimization (Per-Batch SuperGraph).
"""

import time
from collections import deque
from typing import List, Tuple, Optional
import numpy as np


class PhysicsAwareETA:
    """
    ETA estimator based on computational complexity (α-weighted workload).
    
    Key insight: After Phase 2 optimization, computation time for each batch
    scales with max(α) in that batch, not with the number of alphas.
    
    Traditional ETA (avg_batch_time × remaining_batches) fails because:
    - Small α batch (α=0.5~1.0): 1.5s
    - Large α batch (α=3.0~4.0): 6.0s
    
    This estimator uses α_max as workload weight for accurate prediction.
    """
    
    def __init__(
        self,
        batch_assignments: List[Tuple[int, int, float]],  # (start, end, batch_alpha_max)
        window_size: int = 5,
    ):
        """
        Initialize the ETA estimator.
        
        Args:
            batch_assignments: List of (start_idx, end_idx, max_alpha_in_batch)
                from compute_dynamic_batches()
            window_size: Sliding window for rate smoothing
        """
        # Compute workload for each batch (proportional to max_alpha)
        # Workload unit = alpha_max (e.g., batch with α_max=4.0 has 4.0 units)
        self.batch_workloads = [alpha_max for (_, _, alpha_max) in batch_assignments]
        self.total_workload = sum(self.batch_workloads)
        self.num_batches = len(batch_assignments)
        
        self.processed_workload = 0.0
        self.start_time = time.time()
        self.rates = deque(maxlen=window_size)  # workload units / second
        self.last_check_time = self.start_time
        self.completed_batches = 0
    
    def start_batch(self, batch_idx: int):
        """Called when starting a new batch."""
        self.last_check_time = time.time()
    
    def end_batch(self, batch_idx: int) -> Tuple[float, float]:
        """
        Called when a batch completes.
        
        Args:
            batch_idx: 0-indexed batch number
        
        Returns:
            eta_seconds: Estimated remaining time in seconds
            progress: Current progress (0.0 - 1.0) based on workload
        """
        now = time.time()
        duration = now - self.last_check_time
        self.last_check_time = now
        
        if batch_idx < len(self.batch_workloads):
            batch_workload = self.batch_workloads[batch_idx]
            self.processed_workload += batch_workload
            self.completed_batches = batch_idx + 1
            
            # Record rate: workload units per second
            if duration > 0:
                current_rate = batch_workload / duration
                self.rates.append(current_rate)
        
        # Compute remaining workload
        remaining_workload = self.total_workload - self.processed_workload
        
        # Estimate ETA using exponential moving average of rates
        if self.rates:
            # Give more weight to recent batches
            avg_rate = sum(self.rates) / len(self.rates)
            eta = remaining_workload / avg_rate if avg_rate > 0 else 0
        else:
            eta = 0
        
        # Progress based on workload (not batch count)
        progress = self.processed_workload / self.total_workload if self.total_workload > 0 else 0
        
        return eta, progress
    
    def get_batch_progress(self) -> Tuple[int, int]:
        """Get completed batches and total batches."""
        return self.completed_batches, self.num_batches
    
    def get_workload_progress(self) -> Tuple[float, float]:
        """Get processed workload and total workload."""
        return self.processed_workload, self.total_workload
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 0:
            return "--:--"
        elif seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds) // 60
            secs = int(seconds) % 60
            return f"{minutes}:{secs:02d}"
        else:
            hours = int(seconds) // 3600
            minutes = (int(seconds) % 3600) // 60
            return f"{hours}:{minutes:02d}:00"


def create_eta_estimator(
    alpha_values: List[float],
    dynamic_batches: Optional[List[Tuple[int, int, float]]] = None,
) -> PhysicsAwareETA:
    """
    Factory function to create a PhysicsAwareETA estimator.
    
    Args:
        alpha_values: List of all alpha values
        dynamic_batches: Optional pre-computed batch assignments.
            If None, creates a single batch with max(alpha_values).
    
    Returns:
        PhysicsAwareETA instance
    """
    if dynamic_batches is None:
        # Default: single batch containing all alphas
        alpha_max = max(alpha_values) if alpha_values else 1.0
        dynamic_batches = [(0, len(alpha_values), alpha_max)]
    
    return PhysicsAwareETA(dynamic_batches)


# Backward compatibility: simple ETA estimator for non-dynamic batching
class SimpleETA:
    """Simple ETA estimator for legacy code (before Phase 2)."""
    
    def __init__(self, total_batches: int):
        self.total_batches = total_batches
        self.completed_batches = 0
        self.start_time = time.time()
        self.batch_times = deque(maxlen=5)
        self.last_batch_time = self.start_time
    
    def end_batch(self) -> float:
        """Record batch completion and return ETA."""
        now = time.time()
        batch_duration = now - self.last_batch_time
        self.last_batch_time = now
        self.batch_times.append(batch_duration)
        self.completed_batches += 1
        
        remaining = self.total_batches - self.completed_batches
        if self.batch_times:
            avg_time = sum(self.batch_times) / len(self.batch_times)
            return avg_time * remaining
        return 0.0
