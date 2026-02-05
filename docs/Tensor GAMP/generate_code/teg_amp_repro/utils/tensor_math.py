import numpy as np
import opt_einsum as oe

def tr_contract(cores):
    """
    Contract a list of TR cores into a full tensor.
    
    Args:
        cores: list of numpy arrays [Z1, Z2, ..., Zd]
               Zi has shape (r_i, n_i, r_{i+1}) where r_{d+1} = r_1
               
    Returns:
        Full tensor of shape (n1, n2, ..., nd)
    """
    num_cores = len(cores)
    if num_cores == 0:
        return None
    
    # Build einsum subscripts
    # Indices for bond dimensions (r): 0, 1, ..., d-1
    # Indices for spatial dimensions (n): d, d+1, ..., 2d-1
    # Z_1: (0, d, 1)
    # Z_2: (1, d+1, 2)
    # ...
    # Z_d: (d-1, 2d-1, 0)
    
    input_subscripts = []
    for i in range(num_cores):
        r_in = i
        r_out = (i + 1) % num_cores
        n_idx = num_cores + i
        input_subscripts.append((r_in, n_idx, r_out))
    
    output_subscript = tuple(range(num_cores, 2 * num_cores))
    
    # Convert to string format for opt_einsum or use the tuple format
    # opt_einsum supports the tuple format directly
    return oe.contract(*[val for pair in zip(cores, input_subscripts) for val in pair], output_subscript)

def cp_contract(factors):
    """
    Contract CP factors into a full tensor.
    
    Args:
        factors: list of numpy arrays [A1, A2, ..., Ad]
                 Ai has shape (n_i, r)
                 
    Returns:
        Full tensor of shape (n1, n2, ..., nd)
    """
    num_factors = len(factors)
    if num_factors == 0:
        return None
    
    # subscripts: 'ar,br,cr->abc'
    # Indices for spatial dimensions (n): 0, 1, ..., d-1
    # Index for rank (r): d
    
    input_subscripts = []
    for i in range(num_factors):
        input_subscripts.append((i, num_factors))
    
    output_subscript = tuple(range(num_factors))
    
    return oe.contract(*[val for pair in zip(factors, input_subscripts) for val in pair], output_subscript)

def tr_get_slice(cores, indices):
    """
    Get a specific element or slice from a TR tensor without forming the full tensor.
    indices: list of indices (x1, x2, ..., xd). If an index is None, it's a slice.
    Currently only supports full element retrieval (all indices specified).
    """
    # u_x = Tr(Z1[:, x1, :] * Z2[:, x2, :] * ... * Zd[:, xd, :])
    res = cores[0][:, indices[0], :]
    for i in range(1, len(cores)):
        res = res @ cores[i][:, indices[i], :]
    return np.trace(res)

def inner_product(A, B):
    """Compute inner product of two tensors."""
    return np.sum(A * B)

def norm_sq(A):
    """Compute squared Frobenius norm."""
    return np.sum(A**2)

def nmse(original, reconstructed):
    """Compute Normalized Mean Square Error."""
    return norm_sq(original - reconstructed) / norm_sq(original)
