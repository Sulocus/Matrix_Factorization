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
        algorithm_name: str = "bigamp_spreading",
        window_size: int = 20,
    ):
        """
        Initialize the ETA estimator with Relative Ratio Logic.
        
        Args:
            batch_assignments: List of (start_idx, end_idx, max_alpha_in_batch)
            algorithm_name: 'bigamp_spreading', 'bigamp', 'agd', etc.
            window_size: Sliding window for rate smoothing
        """
        self.algorithm_name = algorithm_name.lower()
        
        # 1. Calculate Complexity Weights based on Algorithm
        # User's model: "Batch 1 is 1.0, Batch 2 is 1.2..."
        self.batch_weights = []
        for _, _, alpha_max in batch_assignments:
            if 'spreading' in self.algorithm_name:
                # Spreading complexity ~ Alpha * M * N
                # Since M, N are constant, Weight ~ Alpha
                # We use max(0.1, alpha_max) to avoid zero weight
                weight = max(0.1, float(alpha_max))
            else:
                # Standard BiG-AMP / AGD: Complexity ~ N * M (Constant across alphas)
                weight = 1.0
            self.batch_weights.append(weight)
            
        # 2. Normalize to Relative Ratios (for clarity/logging)
        base_weight = self.batch_weights[0] if self.batch_weights else 1.0
        self.ratios = [w / base_weight for w in self.batch_weights]
        
        # 3. Total Workload
        self.total_workload = sum(self.batch_weights)
        self.num_batches = len(batch_assignments)
        
        # Runtime State
        self.processed_workload = 0.0 # Workload of COMPLETED batches
        
        # Session Tracking (for Resume Support)
        # Session = from when this estimator was created until now
        self.session_start_time = time.time()
        self.session_start_workload = None  # Set on first update
        self.start_time = self.session_start_time  # Alias for compatibility
        
        self.batch_start_time = None
        self.current_batch_idx = -1
                
        # Print the Plan for the user (Transparency)
        # We can't print easily here as it might break UI, but we can log
        # or expose it.
        
    def start_batch(self, batch_idx: int):
        """Called when starting a new batch."""
        self.current_batch_idx = batch_idx
        self.batch_start_time = time.time()
    
    def end_batch(self, batch_idx: int):
        """Called when a batch completes."""
        if 0 <= batch_idx < len(self.batch_weights):
            pass

    def update_progress(self, batch_idx: int, step_pct: float = 0.0):
        """
        Update rate estimation based on SESSION performance.
        
        Logic (User's Request - Simple and Correct):
        - Session Start Workload = Workload at moment Resume started
        - Session Work Done = Current Workload - Session Start Workload
        - Session Time = Now - Session Start Time
        - Rate = Session Work Done / Session Time
        """
        if self.current_batch_idx < 0:
            return

        now = time.time()
        session_elapsed = now - self.session_start_time
        
        # Skip the first 5 seconds (warmup for rate calculation)
        if session_elapsed < 5.0:
            return

        # Calculate Current Total Workload (absolute)
        current_workload = sum(self.batch_weights[:batch_idx])
        if 0 <= batch_idx < len(self.batch_weights):
             current_weight = self.batch_weights[batch_idx]
             pct = max(0.0, min(1.0, step_pct))
             current_workload += current_weight * pct

        # On first valid update (after warmup), record the baseline
        if self.session_start_workload is None:
            self.session_start_workload = current_workload
            self.session_start_time = now  # Reset session start to this moment
            return
        
        # Calculate Session-Based Rate (Simple Cumulative Average)
        session_work = current_workload - self.session_start_workload
        
        if session_work > 0 and session_elapsed > 0:
            self.current_session_rate = session_work / session_elapsed
        else:
            self.current_session_rate = 0.0

    def predict_eta(self) -> float:
        """
        Predict remaining time based on current rate.
        Stateless: does not modify internal counters.
        """
        # This method is effectively superseded by get_status for prediction.
        # If it were to be used, it would need to rely on a stored 'processed_workload'
        # or re-calculate it based on the last known state.
        # For now, we remove the 'pass' as per the user's implicit instruction
        # (by providing a new get_status but no new predict_eta).
        if not self.rates:
            return 0.0
            
        # Use average of recent rates
        avg_rate = sum(self.rates) / len(self.rates)
        
        # Calculate true remaining workload
        # We need to know current state. We assume update_progress was called recently.
        # But predict_eta doesn't take args. 
        # So we should rely on processed_workload updated by update_progress?
        # Actually, let's make predict_eta state-independent if possible, 
        # but rate depends on history.
        
        # Simpler: update_progress updates self.processed_workload snapshot
        # The original 'pass' is removed as per the user's instruction.
        return 0.0 # Placeholder, as get_status is the primary prediction method now.

    def get_status(self, batch_idx: int, step_pct: float) -> Tuple[float, float]:
        """
        Get (ETA, Progress) using Session-Based Rate.
        
        Rate = (Current Workload - Session Start Workload) / Session Elapsed Time
        ETA = Remaining Workload / Rate
        """
        # 1. Update internal state
        self.update_progress(batch_idx, step_pct)
        
        # 2. Check if we have a valid rate yet
        if not hasattr(self, 'current_session_rate') or self.current_session_rate <= 0:
             return -1.0, 0.0 # Signal unknown
        
        # 3. Calculate Current Workload and Remaining
        current_workload = sum(self.batch_weights[:batch_idx])
        if 0 <= batch_idx < len(self.batch_weights):
             current_workload += self.batch_weights[batch_idx] * max(0.0, min(1.0, step_pct))
              
        remaining_workload = max(0.0, self.total_workload - current_workload)
        
        # 4. Predict ETA
        eta = remaining_workload / self.current_session_rate
        progress = current_workload / self.total_workload if self.total_workload > 0 else 0.0
        
        return eta, progress

    # ... keep helpers ...

def create_eta_estimator(
    alpha_values: List[float],
    dynamic_batches: Optional[List[Tuple[int, int, float]]] = None,
    algorithm_name: str = "bigamp_spreading",
) -> PhysicsAwareETA:
    """Factory with algorithm awareness."""
    if dynamic_batches == None:
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
