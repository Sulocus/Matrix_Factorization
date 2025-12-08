"""
Bethe Free Energy and Mean Field Analysis for Ising Model on Bipartite Graphs

Implements GPU-accelerated Belief Propagation (BP) and Mean Field (MF) algorithms
to compute free energy approximations for Ising models on random bipartite graphs.

This analysis is relevant for understanding phase transitions in matrix factorization
problems through the lens of statistical physics.
"""

import torch
from typing import Tuple


# =============================================================================
# Configuration
# =============================================================================

# Matrix dimensions (must be equal: N1 = N2 = N)
N = 30000

# Latent dimension (rank of factorization, same as M in Main.py)
M = 100

# Sparsity parameter: alpha_tilde = C / (M * N), where C is the number of edges
# Same definition as in Main.py: alpha_tilde_left = C / (M * N1)
ALPHA = 1.0  # e.g., 1.0 means C = M * N edges

# Ising model parameters
BETA = 0.8            # Inverse temperature
J_COUPLING = 1.0      # Coupling constant

# Random seed (None for random)
SEED = 123


# =============================================================================
# Graph Generation
# =============================================================================

def sample_pairs_random_gpu(N1: int, N2: int, C: int,
                            device: torch.device, seed: int = None
                            ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Random edge sampling on GPU.

    Args:
        N1: Number of left nodes (rows)
        N2: Number of right nodes (columns)
        C: Number of edges to sample
        device: PyTorch device
        seed: Random seed

    Returns:
        Tuple of (row_indices, col_indices, edge_count)
    """
    if seed is not None:
        torch.manual_seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"Requested edge count C={C} exceeds matrix total size {N1}×{N2}={total}")

    if C == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    idx = torch.randperm(total, device=device)[:C]
    i_idx = idx // N2  # Row index on left side (0 to N1-1)
    j_idx = idx % N2   # Column index on right side (0 to N2-1)

    return i_idx, j_idx, C


def build_edge_tensors(N1: int, N2: int,
                       i_idx: torch.Tensor, j_idx: torch.Tensor,
                       device: torch.device) -> dict:
    """Build GPU edge data structure for parallel BP.

    Args:
        N1: Number of left nodes
        N2: Number of right nodes
        i_idx: Left node indices for each edge
        j_idx: Right node indices for each edge
        device: PyTorch device

    Returns:
        Dictionary containing edge data:
        - left_nodes: Left endpoints (shape: C)
        - right_nodes: Right endpoints (shape: C)
        - degree: Degree of each node (shape: total_N)
        - N1, N2, C, total_N: Graph dimensions
    """
    C = len(i_idx)

    # Edge endpoints (on graph with total_N = N1 + N2 nodes)
    # Left nodes: 0 to N1-1
    # Right nodes: N1 to N1+N2-1
    left_nodes = i_idx.clone()  # 0 to N1-1
    right_nodes = N1 + j_idx    # N1 to N1+N2-1

    # Compute degree for each node
    total_N = N1 + N2
    degree = torch.zeros(total_N, dtype=torch.long, device=device)
    degree.scatter_add_(0, left_nodes, torch.ones(C, dtype=torch.long, device=device))
    degree.scatter_add_(0, right_nodes, torch.ones(C, dtype=torch.long, device=device))

    return {
        'left_nodes': left_nodes,
        'right_nodes': right_nodes,
        'degree': degree,
        'N1': N1,
        'N2': N2,
        'C': C,
        'total_N': total_N,
        'device': device,
    }


# =============================================================================
# Belief Propagation
# =============================================================================

def run_bp_ising_gpu(edge_data: dict, beta: float, J: float,
                     max_iter: int = 2000, tol: float = 1e-8) -> torch.Tensor:
    """GPU-parallel Belief Propagation for Ising model.

    For bipartite graphs, messages only flow between left <-> right nodes.
    Each edge has two messages: h_{left->right} and h_{right->left}

    Update rule:
        h_{i->j} = sum_{k in N(i) \\ j} atanh(tanh(beta*J) * tanh(h_{k->i}))

    For bipartite graphs:
    - Left node neighbors are all on the right side
    - Right node neighbors are all on the left side

    Args:
        edge_data: Edge data dictionary from build_edge_tensors
        beta: Inverse temperature
        J: Coupling constant
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        h: Message tensor, shape (C, 2)
           h[:, 0] = h_{left->right}, h[:, 1] = h_{right->left}
    """
    device = edge_data['device']
    C = edge_data['C']
    total_N = edge_data['total_N']
    left_nodes = edge_data['left_nodes']
    right_nodes = edge_data['right_nodes']

    # Precompute tanh(beta * J)
    tanh_bJ = torch.tanh(torch.tensor(beta * J, device=device, dtype=torch.float64))

    # Initialize messages h = 0
    h = torch.zeros(C, 2, device=device, dtype=torch.float64)
    max_diff = float('inf')

    for it in range(max_iter):
        # Compute u = atanh(tanh(beta*J) * tanh(h))
        tanh_h = torch.tanh(h)
        x = tanh_bJ * tanh_h
        x = torch.clamp(x, -0.999999, 0.999999)
        u = 0.5 * torch.log((1 + x) / (1 - x))  # atanh(x)

        # For each left node i, collect all u_{right->left} from right neighbors
        # Then distribute to each edge: h_{left->right} = sum - current edge contribution

        # Method: First compute total u received by each node, then subtract current edge

        # Total u received by left nodes (from right neighbors)
        sum_u_to_left = torch.zeros(total_N, device=device, dtype=torch.float64)
        sum_u_to_left.scatter_add_(0, left_nodes, u[:, 1])  # Collect u_{right->left}

        # Total u received by right nodes (from left neighbors)
        sum_u_to_right = torch.zeros(total_N, device=device, dtype=torch.float64)
        sum_u_to_right.scatter_add_(0, right_nodes, u[:, 0])  # Collect u_{left->right}

        # New messages
        new_h = torch.zeros_like(h)
        # h_{left->right} = sum_u_to_left[left] - u_{right->left} (subtract current edge)
        new_h[:, 0] = sum_u_to_left[left_nodes] - u[:, 1]
        # h_{right->left} = sum_u_to_right[right] - u_{left->right}
        new_h[:, 1] = sum_u_to_right[right_nodes] - u[:, 0]

        # Check convergence
        max_diff = torch.max(torch.abs(new_h - h)).item()
        h = new_h

        if max_diff < tol:
            print(f"[BP] Converged in {it+1} iterations, max_diff={max_diff:.2e}")
            break
    else:
        print(f"[BP] Warning: not converged after {max_iter} iters, final max_diff={max_diff:.2e}")

    return h


# =============================================================================
# Bethe Free Energy
# =============================================================================

def bethe_free_energy_gpu(edge_data: dict, h: torch.Tensor,
                          beta: float, J: float) -> Tuple[float, dict]:
    """GPU-parallel computation of Bethe free energy.

    F_Bethe = U_Bethe - (1/beta) * S_Bethe

    Where:
        U_Bethe = sum_edges <-J * s_i * s_j>_{b_ij}
        S_Bethe = sum_edges H[b_ij] - sum_nodes (d_i - 1) * H[b_i]

    Args:
        edge_data: Edge data dictionary
        h: BP messages from run_bp_ising_gpu
        beta: Inverse temperature
        J: Coupling constant

    Returns:
        F_bethe: Bethe free energy (float)
        details: Dictionary with component values
    """
    device = edge_data['device']
    total_N = edge_data['total_N']
    left_nodes = edge_data['left_nodes']
    right_nodes = edge_data['right_nodes']
    degree = edge_data['degree']

    eps = 1e-12

    h_L2R = h[:, 0]  # h_{left->right}, shape (C,)
    h_R2L = h[:, 1]  # h_{right->left}, shape (C,)

    # -------------------------------------------------------------------------
    # Edge contributions
    # -------------------------------------------------------------------------
    # Compute edge beliefs b_ij(s_i, s_j) for s_i, s_j in {+1, -1}
    # b_ij(s_i, s_j) = exp(beta*J*s_i*s_j) * m_{L->R}(s_i) * m_{R->L}(s_j) / Z_ij
    # m_{L->R}(s) = exp(h_L2R * s) / (2 * cosh(h_L2R))

    Z_msg_L = 2.0 * torch.cosh(h_L2R)  # (C,)
    Z_msg_R = 2.0 * torch.cosh(h_R2L)  # (C,)

    # m_L(+1), m_L(-1), m_R(+1), m_R(-1)
    m_L_plus = torch.exp(h_L2R) / Z_msg_L
    m_L_minus = torch.exp(-h_L2R) / Z_msg_L
    m_R_plus = torch.exp(h_R2L) / Z_msg_R
    m_R_minus = torch.exp(-h_R2L) / Z_msg_R

    # exp(beta*J*s_i*s_j)
    exp_bJ = torch.exp(torch.tensor(beta * J, device=device, dtype=torch.float64))
    exp_neg_bJ = torch.exp(torch.tensor(-beta * J, device=device, dtype=torch.float64))

    # Unnormalized beliefs
    b_pp = exp_bJ * m_L_plus * m_R_plus       # s_i=+1, s_j=+1
    b_pm = exp_neg_bJ * m_L_plus * m_R_minus  # s_i=+1, s_j=-1
    b_mp = exp_neg_bJ * m_L_minus * m_R_plus  # s_i=-1, s_j=+1
    b_mm = exp_bJ * m_L_minus * m_R_minus     # s_i=-1, s_j=-1

    Z_ij = b_pp + b_pm + b_mp + b_mm  # (C,)

    # Normalize
    b_pp = b_pp / Z_ij
    b_pm = b_pm / Z_ij
    b_mp = b_mp / Z_ij
    b_mm = b_mm / Z_ij

    # Edge energy: U_ij = sum_{s_i, s_j} b(s_i, s_j) * (-J * s_i * s_j)
    # s_i*s_j: ++ -> +1, +- -> -1, -+ -> -1, -- -> +1
    U_edges = torch.sum(-J * (b_pp * 1 + b_pm * (-1) + b_mp * (-1) + b_mm * 1))

    # Edge entropy: H_ij = -sum b * log(b)
    def safe_entropy(b):
        return -torch.sum(torch.where(b > eps, b * torch.log(b + eps), torch.zeros_like(b)))

    H_edges = safe_entropy(b_pp) + safe_entropy(b_pm) + safe_entropy(b_mp) + safe_entropy(b_mm)

    # -------------------------------------------------------------------------
    # Node contributions
    # -------------------------------------------------------------------------
    # For each node i, compute b_i(s) = prod_{k in N(i)} m_{k->i}(s) / Z_i
    # Then compute H_i = -sum_s b_i(s) * log(b_i(s))
    #
    # For left nodes: neighbors are on right, messages are h_{right->left}
    # For right nodes: neighbors are on left, messages are h_{left->right}
    #
    # Use log-space accumulation to avoid numerical issues:
    # log(b_i(+1)) = sum_{k} log(m_{k->i}(+1)) = sum_{k} [h_{k->i} - log(2*cosh(h_{k->i}))]
    # log(b_i(-1)) = sum_{k} [-h_{k->i} - log(2*cosh(h_{k->i}))]

    log_Z_msg_L = torch.log(Z_msg_L)  # log(2*cosh(h_L2R))
    log_Z_msg_R = torch.log(Z_msg_R)  # log(2*cosh(h_R2L))

    log_b_plus = torch.zeros(total_N, device=device, dtype=torch.float64)
    log_b_minus = torch.zeros(total_N, device=device, dtype=torch.float64)

    # Left nodes (indices 0 to N1-1) receive messages h_R2L
    log_b_plus.scatter_add_(0, left_nodes, h_R2L - log_Z_msg_R)
    log_b_minus.scatter_add_(0, left_nodes, -h_R2L - log_Z_msg_R)

    # Right nodes (indices N1 to N1+N2-1) receive messages h_L2R
    log_b_plus.scatter_add_(0, right_nodes, h_L2R - log_Z_msg_L)
    log_b_minus.scatter_add_(0, right_nodes, -h_L2R - log_Z_msg_L)

    # Normalize using log-sum-exp trick
    log_max = torch.maximum(log_b_plus, log_b_minus)
    # For degree-0 nodes: log_b_plus = log_b_minus = 0, so Z_i = 2
    Z_i = torch.exp(log_b_plus - log_max) + torch.exp(log_b_minus - log_max)
    log_Z_i = log_max + torch.log(Z_i)

    # Normalized beliefs
    b_i_plus = torch.exp(log_b_plus - log_Z_i)
    b_i_minus = torch.exp(log_b_minus - log_Z_i)

    # Node entropy
    H_i = -torch.where(b_i_plus > eps, b_i_plus * torch.log(b_i_plus + eps), torch.zeros_like(b_i_plus)) \
          -torch.where(b_i_minus > eps, b_i_minus * torch.log(b_i_minus + eps), torch.zeros_like(b_i_minus))

    # Bethe entropy: S = H_edges - sum_i (d_i - 1) * H_i
    H_nodes = torch.sum((degree - 1).float() * H_i)

    S_bethe = H_edges - H_nodes
    F_bethe = U_edges - (1.0 / beta) * S_bethe

    details = {
        'U_edges': U_edges.item(),
        'H_edges': H_edges.item(),
        'H_nodes': H_nodes.item(),
        'S_bethe': S_bethe.item(),
    }

    return F_bethe.item(), details


# =============================================================================
# Mean Field Approximation
# =============================================================================

def mean_field_ising_gpu(edge_data: dict, beta: float, J: float,
                         max_iter: int = 5000, tol: float = 1e-8) -> Tuple[torch.Tensor, float, dict]:
    """GPU-parallel Mean Field iteration for Ising model.

    Update rule:
        m_i = tanh(beta * sum_j J_ij * m_j)

    For bipartite graphs, uses sparse matrix multiplication.

    Args:
        edge_data: Edge data dictionary
        beta: Inverse temperature
        J: Coupling constant
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        m: Magnetization vector (shape: total_N)
        F_mf: Mean field free energy
        details: Dictionary with energy and entropy components
    """
    device = edge_data['device']
    total_N = edge_data['total_N']
    C = edge_data['C']
    left_nodes = edge_data['left_nodes']
    right_nodes = edge_data['right_nodes']

    # Initialize m = 0
    m = torch.zeros(total_N, device=device, dtype=torch.float64)
    diff = float('inf')

    # Build sparse adjacency matrix edge indices
    # Both edges (left, right) and (right, left) are needed
    edge_index = torch.stack([
        torch.cat([left_nodes, right_nodes]),
        torch.cat([right_nodes, left_nodes])
    ], dim=0)  # shape: (2, 2*C)

    edge_weight = torch.full((2 * C,), J, device=device, dtype=torch.float64)

    for it in range(max_iter):
        # Compute sum_j J_ij * m_j using scatter_add for sparse matrix-vector product
        Jm = torch.zeros(total_N, device=device, dtype=torch.float64)
        Jm.scatter_add_(0, edge_index[0], edge_weight * m[edge_index[1]])

        new_m = torch.tanh(beta * Jm)
        diff = torch.max(torch.abs(new_m - m)).item()
        m = new_m

        if diff < tol:
            print(f"[MF] Converged in {it+1} iterations, max_diff={diff:.2e}")
            break
    else:
        print(f"[MF] Warning: not converged after {max_iter} iters, final diff={diff:.2e}")

    # Compute free energy
    # Energy: E = -sum_{edges} J * m_i * m_j
    m_left = m[left_nodes]
    m_right = m[right_nodes]
    energy = -J * torch.sum(m_left * m_right)

    # Entropy: S = -sum_i [p_+ * log(p_+) + p_- * log(p_-)]
    # where p_+ = (1 + m) / 2, p_- = (1 - m) / 2
    eps = 1e-12
    p_plus = (1 + m) / 2
    p_minus = (1 - m) / 2

    # Negative entropy
    neg_entropy = torch.sum(
        torch.where(p_plus > eps, p_plus * torch.log(p_plus + eps), torch.zeros_like(p_plus)) +
        torch.where(p_minus > eps, p_minus * torch.log(p_minus + eps), torch.zeros_like(p_minus))
    )

    F_mf = energy + (1.0 / beta) * neg_entropy
    true_entropy = -neg_entropy

    details = {
        'energy': energy.item(),
        'entropy': true_entropy.item(),
    }

    return m, F_mf.item(), details


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run Bethe and Mean Field analysis on random bipartite graph."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Force N1 = N2 = N
    N1 = N
    N2 = N

    # Compute edge count C = alpha * M * N
    C = int(ALPHA * M * N)

    print(f"Configuration: N={N}, M={M}, alpha={ALPHA}")
    print(f"  -> Edge count C = alpha * M * N = {C}")
    print(f"  -> beta={BETA}, J={J_COUPLING}")
    print()

    # 1. Generate random mask (bipartite edges)
    i_idx, j_idx, C_real = sample_pairs_random_gpu(N1, N2, C, device, seed=SEED)
    print(f"Generated {C_real} edges")

    # 2. Build edge data structure
    edge_data = build_edge_tensors(N1, N2, i_idx, j_idx, device)
    print(f"Built edge tensors, total nodes: {edge_data['total_N']}")

    # 3. Run Belief Propagation (GPU)
    print("\nRunning Belief Propagation...")
    h = run_bp_ising_gpu(edge_data, beta=BETA, J=J_COUPLING)

    # 4. Compute Bethe free energy (GPU)
    F_bethe, bethe_details = bethe_free_energy_gpu(edge_data, h, beta=BETA, J=J_COUPLING)
    print(f"\nBethe free energy: F_Bethe = {F_bethe:.6f}")
    print(f"  U_edges = {bethe_details['U_edges']:.6f}")
    print(f"  H_edges = {bethe_details['H_edges']:.6f}")
    print(f"  H_nodes = {bethe_details['H_nodes']:.6f}")
    print(f"  S_bethe = {bethe_details['S_bethe']:.6f}")

    # 5. Run Mean Field (GPU)
    print("\nRunning Mean Field...")
    m_mf, F_mf, mf_details = mean_field_ising_gpu(edge_data, beta=BETA, J=J_COUPLING)
    print(f"\nMean-field free energy: F_MF = {F_mf:.6f}")
    print(f"  Energy = {mf_details['energy']:.6f}")
    print(f"  Entropy = {mf_details['entropy']:.6f}")

    # 6. Summary comparison
    print(f"\n{'='*50}")
    print(f"Difference F_MF - F_Bethe = {F_mf - F_bethe:.6f}")
    print(f"\nPer-edge free energy:")
    print(f"  F_Bethe / C = {F_bethe / C:.6f}")
    print(f"  F_MF / C    = {F_mf / C:.6f}")
    print(f"\nPer-node free energy:")
    print(f"  F_Bethe / N = {F_bethe / (2*N):.6f}")
    print(f"  F_MF / N    = {F_mf / (2*N):.6f}")


if __name__ == "__main__":
    main()
