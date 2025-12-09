#!/usr/bin/env python3
"""Precise memory breakdown analysis."""

def analyze_memory(N1, N2, M, S, B, alpha):
    """Break down memory usage by component."""
    C = max(1, int(alpha * M * N1))
    SC = S * C
    
    # All sizes in GB
    def gb(bytes): return bytes / (1024**3)
    
    # ===== Persistent tensors (always allocated) =====
    # Student params: W_hat, X_hat, W_var, X_var
    student_W = S * B * N1 * M * 4  # (S,B,N1,M) float32
    student_X = S * B * M * N2 * 4  # (S,B,M,N2) float32
    student = 4 * (student_W + student_X)
    
    # F_super, Y_super
    f_super = S * C * M * 4
    y_super = S * C * 4
    
    # Indices
    indices = 2 * SC * 8  # int64
    
    persistent = student + f_super + y_super + indices
    
    # ===== Gather tensors (during step) =====
    # W_sel, X_sel, W_var_sel, X_var_sel: (B, SC, M) each
    gather_each = B * SC * M * 4
    gather_total = 4 * gather_each
    
    # ===== Forward pass intermediates =====
    # Z_hat: (B, SC)
    z_hat = B * SC * 4
    # V: (B, SC)
    v = B * SC * 4
    # denom: (B, SC)
    denom = B * SC * 4
    # s_values: (B, SC)
    s_values = B * SC * 4
    # F_flat: (SC, M) - reshaped, no new memory
    # F_sq: (1, SC, M) - F_flat.pow(2) creates copy
    f_sq = SC * M * 4
    
    forward = z_hat + v + denom + s_values + f_sq
    
    # ===== Scatter intermediates =====
    # r_W, tau_W: (B, S*N1, M)
    scatter_W = 2 * B * S * N1 * M * 4
    # r_X, tau_X: (B, S*N2, M)
    scatter_X = 2 * B * S * N2 * M * 4
    # s_exp: (B, SC, 1)
    s_exp = B * SC * 4
    # idx_W, idx_X: (B, SC, M) - expanded indices
    idx_expand = 2 * B * SC * M * 8  # int64!
    # r_W_contrib: (B, SC, M)
    contrib = B * SC * M * 4
    
    scatter = scatter_W + scatter_X + s_exp + idx_expand + contrib
    
    # Total theoretical
    total_theory = persistent + gather_total + forward + scatter
    
    return {
        'C': C,
        'SC': SC,
        'persistent': gb(persistent),
        'gather': gb(gather_total),
        'forward': gb(forward),
        'scatter': gb(scatter),
        'total_theory': gb(total_theory),
    }


# New benchmark data
data = [
    (500, 500, 100, 1, 5, 1.0, 0.75),
    (500, 500, 100, 1, 5, 2.0, 1.49),
    (500, 500, 100, 1, 5, 4.0, 2.98),
    (1000, 1000, 250, 1, 1, 1.0, 2.26),
    (1000, 1000, 250, 1, 1, 2.0, 4.52),
    (1000, 1000, 250, 1, 1, 4.0, 9.03),
    (1000, 1000, 250, 1, 3, 2.0, 11.53),
    (1000, 1000, 250, 1, 5, 2.0, 18.54),
    (1000, 1000, 250, 1, 5, 4.0, 37.06),
    (1000, 1000, 250, 1, 10, 2.0, 36.07),
]

print("Memory component breakdown:")
print("=" * 120)
print(f"{'N1':>6} {'M':>4} {'B':>3} {'α':>4} | {'Persist':>8} | {'Gather':>8} | {'Forward':>8} | {'Scatter':>8} | {'Theory':>8} | {'Actual':>8} | {'Ratio':>6}")
print("=" * 120)

for N1, N2, M, S, B, alpha, actual in data:
    r = analyze_memory(N1, N2, M, S, B, alpha)
    ratio = actual / r['total_theory'] if r['total_theory'] > 0 else 0
    print(f"{N1:>6} {M:>4} {B:>3} {alpha:>4.1f} | {r['persistent']:>7.2f}G | {r['gather']:>7.2f}G | {r['forward']:>7.2f}G | {r['scatter']:>7.2f}G | {r['total_theory']:>7.2f}G | {actual:>7.2f}G | {ratio:>5.2f}x")

print()
print("Analysis:")
print("- If ratio < 1: Our theory overestimates (some tensors freed early)")
print("- If ratio > 1: Missing some tensors in calculation")
print("- Ideal: ratio ≈ 1.0")
