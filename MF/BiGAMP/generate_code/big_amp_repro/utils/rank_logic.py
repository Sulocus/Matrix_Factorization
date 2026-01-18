import numpy as np
from scipy.linalg import svd

class RankManager:
    """
    Implements Rank Contraction and AICc for BiG-AMP as described in Section V-B.
    """

    @staticmethod
    def contract_rank(a_hat, x_hat, threshold=0.05):
        """
        Performs rank contraction based on SVD of X_hat.
        
        Parameters:
            a_hat (np.ndarray): Estimate of A (M x N)
            x_hat (np.ndarray): Estimate of X (N x L)
            threshold (float): Relative threshold for singular values.
            
        Returns:
            a_hat_new (np.ndarray): Truncated A
            x_hat_new (np.ndarray): Truncated X
            new_rank (int): The reduced rank
        """
        # SVD of X_hat (N x L)
        # Note: scipy.linalg.svd returns U (N x N), s (min(N,L)), Vh (min(N,L) x L)
        try:
            U, s, Vh = svd(x_hat, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback if SVD fails to converge
            return a_hat, x_hat, x_hat.shape[0]

        # Normalize singular values by the largest one to check relative significance
        if s[0] == 0:
            return a_hat, x_hat, x_hat.shape[0]
            
        s_norm = s / s[0]
        
        # Find how many singular values are above the threshold
        new_rank = int(np.sum(s_norm > threshold))
        
        # Ensure we don't drop to zero rank unless it's truly zero
        new_rank = max(new_rank, 1)
        
        if new_rank < x_hat.shape[0]:
            # Rotate A and X into the SVD basis of X and truncate
            # A_hat * X_hat \approx (A_hat * U[:, :new_rank]) * (diag(s[:new_rank]) * Vh[:new_rank, :])
            a_hat_new = a_hat @ U[:, :new_rank]
            x_hat_new = np.diag(s[:new_rank]) @ Vh[:new_rank, :]
            return a_hat_new, x_hat_new, new_rank
        
        return a_hat, x_hat, x_hat.shape[0]

    @staticmethod
    def compute_aicc(y, z_hat, mask, rank, m, l):
        """
        Computes the corrected Akaike Information Criterion (AICc).
        Used to compare models with different inner dimensions N.
        
        AICc = 2k - 2ln(L) + 2k(k+1)/(n-k-1)
        where:
            k = rank * (M + L - rank)  [Number of degrees of freedom]
            n = number of observations
            L = Likelihood
        """
        if mask is not None:
            n_obs = np.sum(mask)
            resid = (y - z_hat)[mask]
        else:
            n_obs = y.size
            resid = y - z_hat
            
        # Mean Squared Error
        mse = np.mean(resid**2)
        if mse == 0:
            return -np.inf
            
        # Log-likelihood for Gaussian noise (ignoring constant terms)
        # ln(L) \approx -n/2 * ln(mse)
        log_lik = -0.5 * n_obs * np.log(mse)
        
        # Number of parameters in the low-rank model
        k = rank * (m + l - rank)
        
        # Check for degrees of freedom
        if n_obs <= k + 1:
            return np.inf
            
        aic = 2 * k - 2 * log_lik
        aicc = aic + (2 * k * (k + 1)) / (n_obs - k - 1)
        
        return aicc

    @staticmethod
    def estimate_rank_aic(y, mask, max_rank, solver_factory):
        """
        Heuristic to find the optimal rank by sweeping and using AICc.
        solver_factory is a function that takes a rank and returns a fitted BiG-AMP result.
        """
        best_aicc = np.inf
        best_rank = 1
        results = {}

        # This is a simplified sweep. In practice, one might use a more efficient search.
        for r in range(1, max_rank + 1):
            res = solver_factory(r)
            aicc = RankManager.compute_aicc(y, res['z_hat'], mask, r, y.shape[0], y.shape[1])
            results[r] = aicc
            if aicc < best_aicc:
                best_aicc = aicc
                best_rank = r
        
        return best_rank, results
