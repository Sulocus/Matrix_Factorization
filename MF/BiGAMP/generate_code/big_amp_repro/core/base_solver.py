import numpy as np
from abc import ABC, abstractmethod

class BaseBiGAMP(ABC):
    """
    Abstract base class for BiG-AMP solvers.
    Defines the interface and common utilities for bilinear inference.
    """
    def __init__(self, params=None):
        """
        Initialize the solver with parameters.
        
        Parameters:
            params (dict): Configuration parameters including:
                - nit: Maximum number of iterations (default: 100)
                - tol: Convergence tolerance (default: 1e-5)
                - verbose: Boolean for logging (default: False)
                - damping: Initial damping factor (default: 1.0)
        """
        self.params = params if params is not None else {}
        self.nit = self.params.get('nit', 100)
        self.tol = self.params.get('tol', 1e-5)
        self.verbose = self.params.get('verbose', False)
        self.damping = self.params.get('damping', 1.0)
        
    @abstractmethod
    def solve(self, Y, mask=None):
        """
        Solve the bilinear problem Y = AX + W.
        
        Parameters:
            Y (np.ndarray): Observation matrix (M x L)
            mask (np.ndarray): Binary mask for observed entries (M x L). 
                               If None, all entries are assumed observed.
            
        Returns:
            dict: Results containing A_hat, X_hat, and convergence history.
        """
        pass

    def _check_convergence(self, current_val, prev_val):
        """
        Check if the algorithm has converged based on relative change.
        """
        if prev_val is None:
            return False
        
        diff = np.linalg.norm(current_val - prev_val)
        norm = np.linalg.norm(current_val)
        
        if norm < 1e-12:
            return diff < self.tol
            
        return (diff / norm) < self.tol
