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
        self.start_time = time.time()
        self.rates = deque(maxlen=window_size) # Workload units / second
        self.batch_start_time = None
        self.rates = deque(maxlen=window_size) # Workload units / second
        self.batch_start_time = None
        self.current_batch_idx = -1
        
        # Resume Support: Track initial workload to isolate session performance
        self.initial_workload_offset = None
                
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
            # Mark this batch as fully done in the accumulator
            # (In a real implementation, we might track continuous progress, 
            # but updating 'completed' blocks is safer).
            pass

    def update_progress(self, batch_idx: int, step_pct: float = 0.0):
        """
        Update rate estimation based on current progress using Relative Ratios.
        
        Logic:
        Total Done = Sum(Weights of prev batches) + (Current Batch Weight * step_pct)
        Global Rate = Total Done / Total Elapsed Time
        """
        if self.current_batch_idx < 0:
            return

        now = time.time()
        # Use initial start time to capture full history
        total_elapsed = now - self.start_time
        
        if total_elapsed < 1.0: # Ignore first second to allow warmup
            return

        # Calculate Done Workload
        done_workload = sum(self.batch_weights[:batch_idx])
        if 0 <= batch_idx < len(self.batch_weights):
             current_weight = self.batch_weights[batch_idx]
             # Clamp percentage
             pct = max(0.0, min(1.0, step_pct))
             done_workload += current_weight * pct

        # Initialize offset on first update (Resume capability)
        if self.initial_workload_offset is None:
            # We assume the session starts processing effectively from this point
            # To be safe against "start mid-batch", we set offset to the beginning of this batch
            # or simply use current done_workload if we assume we just started.
            # However, if we resume at batch 10, done_workload includes batches 0-9.
            # We must exclude 0-9 from rate calculation.
            
            # Better strategy: Set offset to workload of COMPLETED batches before this session.
            # But we don't know exactly which were completed offline.
            # SIMPLEST: Set offset = done_workload_at_start.
            # But update_progress is called continuously.
            # Let's assume the first call defines the baseline.
            # But wait, first call might have step_pct > 0.
            # For 100% safety: offset = sum(weights[:batch_idx]) (completed batches)
            # This ignores the partial progress of current batch in the offset, 
            # effectively treating current batch as "fresh work" for rate calc.
            self.initial_workload_offset = sum(self.batch_weights[:batch_idx])

        # Update Global Rate (Most stable metric) based on SESSION performance
        session_work = done_workload - self.initial_workload_offset
        
        if session_work > 0 and total_elapsed > 0:
             current_global_rate = session_work / total_elapsed
             self.rates.append(current_global_rate)

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
        Get (ETA, Progress) using the Relative Ratio Model.
        """
        # 1. Update internal rate state
        self.update_progress(batch_idx, step_pct)
        
        # 2. Get stable rate
        if not self.rates:
             return -1.0, 0.0 # Signal unknown
             
        avg_rate = sum(self.rates) / len(self.rates)
        
        # 3. Calculate Remaining Work
        done_workload = sum(self.batch_weights[:batch_idx])
        if 0 <= batch_idx < len(self.batch_weights):
             done_workload += self.batch_weights[batch_idx] * max(0.0, min(1.0, step_pct))
             
        remaining_workload = max(0.0, self.total_workload - done_workload)
        
        # 4. Predict
        eta = remaining_workload / avg_rate if avg_rate > 1e-9 else 0.0
        progress = done_workload / self.total_workload if self.total_workload > 0 else 0.0
        
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
