"""
Adaptive ETA estimation for BiG-AMP training.

Uses dynamic calibration based on measured batch times to predict remaining time.
Initial prediction uses theoretical weights (α_max), then calibrates using
actual measurements as batches complete.

Key improvement over previous version:
- Records actual batch completion times
- Calculates calibration factor k = actual_time / α_max
- Uses k to predict remaining batch times more accurately
"""

import time
from collections import deque
from typing import List, Tuple, Optional


class PhysicsAwareETA:
    """
    Adaptive ETA estimator with dynamic calibration.
    
    Uses theoretical weights initially (batch time ∝ α_max), then
    calibrates predictions using actual measured batch times.
    
    This provides accurate ETA estimation regardless of:
    - Algorithm type (Spreading, AGD, BiGAMP)
    - Execution mode (Bipartite, General)
    - Hardware characteristics
    """
    
    def __init__(
        self,
        batch_assignments: List[Tuple[int, int, float]],  # (start, end, batch_alpha_max)
        algorithm_name: str = "bigamp_spreading",
        window_size: int = 20,
    ):
        """
        Initialize the adaptive ETA estimator.
        
        Args:
            batch_assignments: List of (start_idx, end_idx, max_alpha_in_batch)
            algorithm_name: 'bigamp_spreading', 'bigamp', 'agd', etc.
            window_size: Unused, kept for API compatibility
        """
        self.algorithm_name = algorithm_name.lower()
        self.num_batches = len(batch_assignments)
        
        # Store batch info for calibration
        self.batch_alpha_max = []
        for _, _, alpha_max in batch_assignments:
            self.batch_alpha_max.append(max(0.1, float(alpha_max)))
        
        # Theoretical weights (initial assumption: time ∝ α_max)
        self.batch_weights = self.batch_alpha_max.copy()
        self.total_workload = sum(self.batch_weights)
        
        # Calibration data: list of (alpha_max, actual_time) for completed batches
        self.completed_batches_data: List[Tuple[float, float]] = []
        self.calibration_factor: Optional[float] = None  # k = time / α
        
        # Runtime state
        self.session_start_time = time.time()
        self.start_time = self.session_start_time
        self.batch_start_time: Optional[float] = None
        self.current_batch_idx = -1
        
        # For warmup period
        self.warmup_duration = 10.0  # seconds
        
    def start_batch(self, batch_idx: int):
        """Called when starting a new batch."""
        self.current_batch_idx = batch_idx
        self.batch_start_time = time.time()
    
    def end_batch(self, batch_idx: int):
        """
        Called when a batch completes. Records actual time for calibration.
        
        This is the KEY improvement: we measure actual batch time and use
        it to calibrate future predictions.
        """
        if self.batch_start_time is None:
            return
        
        if 0 <= batch_idx < len(self.batch_alpha_max):
            actual_time = time.time() - self.batch_start_time
            alpha_max = self.batch_alpha_max[batch_idx]
            
            # Record for calibration
            self.completed_batches_data.append((alpha_max, actual_time))
            
            # Update calibration factor (average k = time / α)
            self._update_calibration()
    
    def _update_calibration(self):
        """Update calibration factor based on completed batch data."""
        if not self.completed_batches_data:
            return
        
        # Calculate k = Σ(time / α) / n
        # This gives us the average time per unit α
        total_k = sum(actual_time / alpha_max 
                      for alpha_max, actual_time in self.completed_batches_data)
        self.calibration_factor = total_k / len(self.completed_batches_data)
    
    def _predict_batch_time(self, batch_idx: int) -> float:
        """Predict time for a specific batch using calibration."""
        if batch_idx >= len(self.batch_alpha_max):
            return 0.0
        
        alpha_max = self.batch_alpha_max[batch_idx]
        
        if self.calibration_factor is not None:
            # Use calibrated prediction
            return self.calibration_factor * alpha_max
        else:
            # No calibration yet, use theoretical weight as relative value
            # Return -1 to signal unknown
            return -1.0

    def get_status(self, batch_idx: int, step_pct: float) -> Tuple[float, float]:
        """
        Get (ETA, Progress) using adaptive calibration.
        
        Strategy:
        1. For completed batches: use actual recorded times
        2. For current batch: use step progress within calibrated prediction
        3. For future batches: use calibrated predictions (k × α_max)
        """
        now = time.time()
        elapsed = now - self.session_start_time
        
        # Warmup period
        if elapsed < self.warmup_duration:
            return -1.0, 0.0
        
        # Calculate progress (batch-weighted)
        completed_weight = sum(self.batch_weights[:batch_idx])
        current_weight = self.batch_weights[batch_idx] if 0 <= batch_idx < len(self.batch_weights) else 0
        current_progress = current_weight * max(0.0, min(1.0, step_pct))
        total_progress_weight = completed_weight + current_progress
        progress = total_progress_weight / self.total_workload if self.total_workload > 0 else 0.0
        
        # ===== ETA Calculation =====
        
        # Method 1: If we have calibration data, use it
        if self.calibration_factor is not None and self.calibration_factor > 0:
            # Current batch remaining time
            if self.batch_start_time and 0 <= batch_idx < len(self.batch_alpha_max):
                current_batch_elapsed = now - self.batch_start_time
                current_batch_predicted = self.calibration_factor * self.batch_alpha_max[batch_idx]
                current_batch_remaining = max(0, current_batch_predicted - current_batch_elapsed)
            else:
                current_batch_remaining = 0
            
            # Future batches predicted time
            future_time = 0.0
            for j in range(batch_idx + 1, len(self.batch_alpha_max)):
                future_time += self.calibration_factor * self.batch_alpha_max[j]
            
            eta = current_batch_remaining + future_time
            return eta, progress
        
        # Method 2: No calibration yet, use simple elapsed-based estimation
        # This is less accurate but works for warmup period
        if progress > 0.01:
            total_time_estimate = elapsed / progress
            eta = max(0, total_time_estimate - elapsed)
            return eta, progress
        
        return -1.0, progress

    def update_progress(self, batch_idx: int, step_pct: float = 0.0):
        """Backward compatibility - no longer needed with new design."""
        pass


def create_eta_estimator(
    alpha_values: List[float],
    dynamic_batches: Optional[List[Tuple[int, int, float]]] = None,
    algorithm_name: str = "bigamp_spreading",
) -> PhysicsAwareETA:
    """Factory with algorithm awareness."""
    if dynamic_batches is None:
        alpha_max = max(alpha_values) if alpha_values else 1.0
        dynamic_batches = [(0, len(alpha_values), alpha_max)]
    
    return PhysicsAwareETA(dynamic_batches, algorithm_name=algorithm_name)


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
