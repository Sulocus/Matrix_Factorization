import numpy as np
from abc import ABC, abstractmethod
import time
from teg_amp_repro.utils.tensor_math import nmse

class BaseAMPSolver(ABC):
    """
    Abstract base class for AMP-based solvers (TeG-AMP and TeS-AMP).
    Handles common iteration logic, logging, and convergence checks.
    """
    def __init__(self, max_iter=100, tol=1e-8, verbose=True):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.history = {
            'nmse': [],
            'time': [],
            'cost': [],
            'beta': []
        }
        self.beta = 1.0  # Initial damping factor

    @abstractmethod
    def _initialize(self, y, mask, **kwargs):
        """Initialize solver state (means, variances, etc.)"""
        pass

    @abstractmethod
    def _step(self, t):
        """
        Perform a single iteration of the AMP algorithm.
        Should return the cost J(t) for adaptive damping.
        """
        pass

    @abstractmethod
    def _get_reconstruction(self):
        """Return the current reconstructed tensor"""
        pass

    @abstractmethod
    def _backup_state(self):
        """Backup current state for damping rejection"""
        pass

    @abstractmethod
    def _restore_state(self):
        """Restore state from backup"""
        pass

    def solve(self, y, mask, ground_truth=None, damping_enabled=True, **kwargs):
        """
        Main solver loop with optional adaptive damping.
        
        Args:
            y: Observed tensor (with noise and missing values)
            mask: Binary mask (1 for observed, 0 for missing)
            ground_truth: Optional ground truth tensor for NMSE tracking
            damping_enabled: Whether to use adaptive damping logic
            **kwargs: Additional solver-specific parameters
        """
        start_time = time.time()
        self._initialize(y, mask, **kwargs)
        
        prev_cost = float('inf')
        prev_nmse = float('inf')
        self.beta = 1.0
        
        t = 0
        while t < self.max_iter:
            # Backup state in case we need to reject this step
            if damping_enabled:
                self._backup_state()
            
            # Perform one iteration step
            cost = self._step(t)
            
            # Adaptive Damping Logic
            if damping_enabled and t > 0:
                if cost > prev_cost + 1e-10: # Small epsilon for numerical stability
                    # Reject step
                    self._restore_state()
                    self.beta *= 0.5
                    if self.verbose:
                        print(f"Iteration {t}: Step rejected (Cost {cost:.4e} > {prev_cost:.4e}). New beta: {self.beta:.4f}")
                    
                    if self.beta < 1e-6:
                        if self.verbose:
                            print("Beta too small, stopping.")
                        break
                    # Re-run iteration t with smaller beta
                    continue
                else:
                    # Accept step
                    self.beta = min(1.0, self.beta * 1.1)
            
            # Track metrics
            reconstruction = self._get_reconstruction()
            current_nmse = 1.0
            if ground_truth is not None:
                current_nmse = nmse(ground_truth, reconstruction)
            
            self.history['nmse'].append(current_nmse)
            self.history['time'].append(time.time() - start_time)
            self.history['cost'].append(cost)
            self.history['beta'].append(self.beta)
            
            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t:3d}: NMSE = {current_nmse:.2e}, Cost = {cost:.2e}, Beta = {self.beta:.2f}")
            
            # Convergence check
            if abs(prev_nmse - current_nmse) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {t}")
                break
            
            prev_nmse = current_nmse
            prev_cost = cost
            t += 1
            
        return self._get_reconstruction(), self.history
