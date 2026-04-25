import numpy as np
from ..utils.tensor_math import tr_contract, nmse

class AltMinTR:
    """
    Alternating Least Squares (ALS) baseline for Tensor Ring (TR) completion.
    Updates each core by solving a linear least squares problem while fixing others.
    """
    def __init__(self, shape, ranks, max_iter=50, tol=1e-6):
        """
        Args:
            shape: Tuple of tensor dimensions (N1, ..., Nd).
            ranks: List of TR ranks (r1, ..., rd).
            max_iter: Maximum number of ALS iterations.
            tol: Convergence tolerance based on NMSE or reconstruction error.
        """
        self.shape = shape
        self.d = len(shape)
        self.ranks = list(ranks)
        self.max_iter = max_iter
        self.tol = tol
        self.cores = []

    def _initialize(self):
        """Initialize TR cores with small random Gaussian values."""
        self.cores = []
        for i in range(self.d):
            # Core i has shape (r_i, N_i, r_{i+1}) where r_{d} = r_0
            r_in = self.ranks[i]
            r_out = self.ranks[(i + 1) % self.d]
            core = np.random.normal(0, 0.1, (r_in, self.shape[i], r_out))
            self.cores.append(core)

    def solve(self, y, mask, ground_truth=None):
        """
        Solve the TR completion problem using ALS.
        
        Args:
            y: Observed tensor (with zeros at unobserved entries).
            mask: Binary mask of observed entries.
            ground_truth: Optional ground truth tensor for NMSE tracking.
            
        Returns:
            reconstructed_tensor: The full tensor after ALS.
            history: List of NMSE (if ground_truth provided) or reconstruction error.
        """
        self._initialize()
        
        obs_indices = np.argwhere(mask)
        obs_values = y[mask]
        
        history = []
        prev_error = float('inf')

        for it in range(self.max_iter):
            # Iterate through each core
            for k in range(self.d):
                # Update core k: Z_k is (r_k, N_k, r_{k+1})
                r_k = self.ranks[k]
                r_kp1 = self.ranks[(k + 1) % self.d]
                
                # For each slice n in 0...N_k-1
                for n in range(self.shape[k]):
                    # Find observations where index k is n
                    idx_in_obs = np.where(obs_indices[:, k] == n)[0]
                    if len(idx_in_obs) == 0:
                        continue
                    
                    target_y = obs_values[idx_in_obs]
                    
                    # Construct the design matrix A for the linear system A * vec(Z_k(n)) = target_y
                    # Each row of A corresponds to an observation x in Omega_{k,n}
                    A = np.zeros((len(idx_in_obs), r_k * r_kp1))
                    
                    for i, obs_idx_ptr in enumerate(idx_in_obs):
                        x = obs_indices[obs_idx_ptr]
                        
                        # Compute L = Z_0(x_0) ... Z_{k-1}(x_{k-1})
                        # L is (r_0, r_k)
                        L = np.eye(self.ranks[0])
                        for j in range(k):
                            # Z_j(x_j) is (r_j, r_{j+1})
                            L = L @ self.cores[j][:, x[j], :]
                        
                        # Compute R = Z_{k+1}(x_{k+1}) ... Z_{d-1}(x_{d-1})
                        # R is (r_{k+1}, r_0)
                        R = np.eye(self.ranks[(k + 1) % self.d])
                        for j in range(k + 1, self.d):
                            R = R @ self.cores[j][:, x[j], :]
                        
                        # M = R @ L is (r_{k+1}, r_k)
                        # Tr(L * Z_k(n) * R) = Tr(R * L * Z_k(n)) = sum_{a,b} M_{b,a} * Z_k(n)_{a,b}
                        M = R @ L
                        # The coefficient for Z_k(n)_{a,b} is M_{b,a}
                        # vec(Z_k(n)) is ordered such that (a,b) maps to a * r_{k+1} + b
                        A[i, :] = M.T.flatten()
                    
                    # Solve least squares: A * z = target_y
                    try:
                        z, _, _, _ = np.linalg.lstsq(A, target_y, rcond=None)
                        self.cores[k][:, n, :] = z.reshape(r_k, r_kp1)
                    except np.linalg.LinAlgError:
                        # Fallback if system is singular
                        pass
            
            # Evaluate current reconstruction
            # Note: tr_contract expects cores in (N, r_in, r_out) or similar?
            # Let's check tensor_math.py: it expects cores as list of (r_in, N, r_out)
            # Wait, my tensor_math.py tr_contract:
            # for i in range(d):
            #    core_labels = [f'r{i}', f'n{i}', f'r{(i+1)%d}']
            # Yes, it matches (r_in, N, r_out).
            
            current_tensor = tr_contract(self.cores)
            
            if ground_truth is not None:
                curr_err = nmse(ground_truth, current_tensor)
            else:
                recon_obs = current_tensor[mask]
                curr_err = np.mean((recon_obs - obs_values)**2)
            
            history.append(curr_err)
            
            if abs(prev_error - curr_err) < self.tol:
                break
            prev_error = curr_err
            
        return current_tensor, history
