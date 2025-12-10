"""
SMF System Comprehensive Diagnostic Script.

This script performs Layer 1-5 diagnostics to identify issues in the SMF system.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_section(title):
    print(f"\n--- {title} ---")

def main():
    from smf.core.config import Config
    from smf.modules.teachers import TeacherGenerator
    from smf.modules.graphs.supergraph import create_supergraph
    from smf.modules.algorithms.bigamp_spreading_parallel import (
        generate_F_super, compute_Y_super
    )
    
    # =========================================================================
    # Configuration (User's actual settings)
    # =========================================================================
    print_header("LAYER 1: DATA FLOW & PARAMETER VERIFICATION")
    
    cfg = Config()
    cfg.matrix.N1 = 200
    cfg.matrix.N2 = 200
    cfg.matrix.M = 50
    cfg.alpha.start = 0.0
    cfg.alpha.stop = 4.0
    cfg.alpha.step = 0.5
    cfg.training.max_steps = 100  # Reduced for diagnostic
    cfg.training.samples_per_alpha = 2
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed = 42
    cfg.algorithm.mode = "spreading_parallel"
    cfg.spreading.f_distribution = "rademacher"
    cfg.spreading.seed = 12345
    cfg.teacher.type = "standard"
    cfg.graph.type = "random"
    
    device = torch.device(cfg.training.device)
    
    print_section("Config Parameters")
    print(f"  N1 = {cfg.matrix.N1}")
    print(f"  N2 = {cfg.matrix.N2}")
    print(f"  M = {cfg.matrix.M}")
    print(f"  Alpha: {cfg.alpha.start} -> {cfg.alpha.stop} (step {cfg.alpha.step})")
    print(f"  Alpha values: {cfg.alpha_values}")
    print(f"  Max steps: {cfg.training.max_steps}")
    print(f"  Samples per alpha: {cfg.training.samples_per_alpha}")
    print(f"  Device: {cfg.training.device}")
    print(f"  Seed: {cfg.training.seed}")
    print(f"  Algorithm mode: {cfg.algorithm.mode}")
    print(f"  F distribution: {cfg.spreading.f_distribution}")
    print(f"  Teacher type: {cfg.teacher.type}")
    print(f"  Graph type: {cfg.graph.type}")
    
    N1, N2, M = cfg.matrix.N1, cfg.matrix.N2, cfg.matrix.M
    S = cfg.training.samples_per_alpha
    alpha_values = cfg.alpha_values
    seed = cfg.training.seed
    
    # =========================================================================
    # Create Teacher
    # =========================================================================
    print_section("Teacher Creation")
    
    teacher = TeacherGenerator(
        type=cfg.teacher.type,
        variance_scale=cfg.teacher.variance_scale,
        spreading_seed=cfg.spreading.seed,
    )
    W_true, X_true = teacher.create(N1, N2, M, device, seed)
    
    print(f"  W_true shape: {W_true.shape}")
    print(f"  X_true shape: {X_true.shape}")
    print(f"  W_true var: {W_true.var().item():.4f}")
    print(f"  X_true var: {X_true.var().item():.4f}")
    
    # =========================================================================
    # LAYER 2: Physical Model Verification
    # =========================================================================
    print_header("LAYER 2: PHYSICAL MODEL VERIFICATION")
    
    print_section("SuperGraph Creation")
    
    supergraph = create_supergraph(
        N1=N1, N2=N2, M=M,
        alpha_values=alpha_values,
        S=S,
        base_seed=seed,
        device=device,
    )
    
    print(f"  SuperGraph shape: i_idx={supergraph.i_idx.shape}, j_idx={supergraph.j_idx.shape}")
    print(f"  C_max = {supergraph.C_max}")
    print(f"  alpha_mask shape: {supergraph.alpha_mask.shape}")
    
    print_section("Alpha Mask Verification (Critical!)")
    print("  Expected: Low alpha -> few edges, High alpha -> many edges")
    
    for a, alpha in enumerate(alpha_values):
        C_k = supergraph.get_active_edges(a)
        expected_deg = int(round(alpha * M))
        expected_C = N1 * expected_deg
        print(f"  Alpha {alpha:.1f}: {C_k} edges (expected ~{expected_C})")
    
    print_section("F Distribution Verification")
    
    F_super = generate_F_super(
        supergraph=supergraph,
        M=M,
        base_seed=cfg.spreading.seed,
        device=device,
        f_distribution=cfg.spreading.f_distribution,
    )
    
    print(f"  F_super shape: {F_super.shape}")
    print(f"  F_super dtype: {F_super.dtype}")
    print(f"  F_super[0,0,:10]: {F_super[0, 0, :10].tolist()}")
    
    # For Rademacher, values should be {-1, +1}
    if cfg.spreading.f_distribution == "rademacher":
        F_float = F_super[0].float()
        unique_vals = F_float.unique()
        print(f"  Unique values in F: {unique_vals.tolist()}")
        if set(unique_vals.tolist()) == {-1, 1}:
            print("  ✅ F distribution correct (Rademacher {-1, +1})")
        else:
            print("  ❌ F distribution INCORRECT!")
    
    print_section("Y Computation Verification")
    
    Y_super = compute_Y_super(W_true, X_true, supergraph, F_super)
    
    print(f"  Y_super shape: {Y_super.shape}")
    print(f"  Y_super[0,:10]: {Y_super[0, :10].tolist()}")
    print(f"  Y_super mean: {Y_super.mean().item():.4f}")
    print(f"  Y_super std: {Y_super.std().item():.4f}")
    
    # =========================================================================
    # LAYER 3: Algorithm Execution Verification
    # =========================================================================
    print_header("LAYER 3: ALGORITHM EXECUTION VERIFICATION")
    
    from smf.modules.teachers import SpreadingDataParallel
    from smf.modules.algorithms.bigamp_spreading_parallel import BiGAMPSpreadingParallel
    
    spreading_data = SpreadingDataParallel(
        supergraph=supergraph,
        F_super=F_super,
        Y_super=Y_super,
        M=M,
        alpha_values=torch.tensor(alpha_values, device=device),
        W_teacher=W_true,
        X_teacher=X_true,
    )
    
    print_section("Initial Q_Y Check (Before Training)")
    
    from smf.modules.metrics.spreading import compute_qy_spreading_parallel
    
    A = len(alpha_values)
    # Create random initial student
    torch.manual_seed(9999)  # Different from teacher
    W_init = torch.randn(A, N1, M, device=device) * 0.1
    X_init = torch.randn(A, M, N2, device=device) * 0.1
    
    initial_Q_Y = compute_qy_spreading_parallel(W_init, X_init, spreading_data, 0)
    print(f"  Initial Q_Y (should be ~0): {initial_Q_Y.tolist()}")
    
    if initial_Q_Y.mean().item() > 0.3:
        print("  ❌ WARNING: Initial Q_Y too high! Possible initialization issue.")
    else:
        print("  ✅ Initial Q_Y is low as expected")
    
    print_section("Training Execution")
    
    algo = BiGAMPSpreadingParallel(cfg, device)
    
    # Train
    print(f"  Training with {cfg.training.max_steps} steps...")
    W_students, X_students = algo.train_full_parallel(spreading_data, verbose=False)
    
    print(f"  W_students shape: {W_students.shape}")
    print(f"  X_students shape: {X_students.shape}")
    
    if W_students.shape[0] != S:
        print(f"  ❌ WARNING: W_students S dimension mismatch: {W_students.shape[0]} vs {S}")
    
    # =========================================================================
    # LAYER 4: Metric Computation Audit
    # =========================================================================
    print_header("LAYER 4: METRIC COMPUTATION AUDIT")
    
    from smf.modules.metrics.spreading import compute_all_metrics_spreading_parallel
    
    metrics = compute_all_metrics_spreading_parallel(W_students, X_students, spreading_data)
    
    print_section("Computed Metrics")
    print(f"  Available keys: {list(metrics.keys())}")
    
    print_section("Q_Y vs Alpha (The Critical Check)")
    print("  Expected: Low alpha (~0-1) -> Q_Y near 0")
    print("            High alpha (~3-4) -> Q_Y near 1")
    
    Q_Y = metrics["Q_Y_mean"].cpu().tolist()
    for a, (alpha, qy) in enumerate(zip(alpha_values, Q_Y)):
        status = ""
        if alpha < 1.0 and qy > 0.5:
            status = "❌ TOO HIGH!"
        elif alpha > 3.0 and qy < 0.5:
            status = "❌ TOO LOW!"
        else:
            status = "✅"
        print(f"  Alpha {alpha:.1f}: Q_Y = {qy:.4f} {status}")
    
    print_section("Q_W, Q_X (Factor Overlaps)")
    Q_W = metrics["Q_W_mean"].cpu().tolist()
    Q_X = metrics["Q_X_mean"].cpu().tolist()
    for a, (alpha, qw, qx) in enumerate(zip(alpha_values, Q_W, Q_X)):
        print(f"  Alpha {alpha:.1f}: Q_W = {qw:.4f}, Q_X = {qx:.4f}")
    
    # =========================================================================
    # LAYER 5: Missing Metrics Check
    # =========================================================================
    print_header("LAYER 5: MISSING METRICS CHECK")
    
    expected_metrics = [
        "Q_Y", "Q_W", "Q_X", "Q_W_prime", "Q_X_prime",
        "Q_Y_unobserved", "Q_Y_observed", "MSE", "Gen_Error",
        "physical_overlap_Y", "physical_overlap_W", "physical_overlap_X"
    ]
    
    returned_keys = [k.replace("_mean", "").replace("_std", "") for k in metrics.keys()]
    returned_set = set(returned_keys)
    
    for metric in expected_metrics:
        if metric in returned_set or f"{metric}_mean" in metrics:
            print(f"  ✅ {metric}")
        else:
            print(f"  ❌ {metric} - MISSING!")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("DIAGNOSTIC SUMMARY")
    
    issues = []
    
    # Check Q_Y curve shape
    if Q_Y[0] > 0.3:
        issues.append("Q_Y at low alpha is too high (should be ~0)")
    if Q_Y[-1] < 0.7:
        issues.append("Q_Y at high alpha is too low (should be ~1)")
    
    # Check if Q_Y increases with alpha
    if Q_Y[0] > Q_Y[-1]:
        issues.append("Q_Y does not increase with alpha!")
    
    # Check missing metrics
    if "Q_Y_unobserved" not in returned_set:
        issues.append("Q_Y_unobserved not implemented")
    if "MSE" not in returned_set and "MSE_mean" not in metrics:
        issues.append("MSE not returned")
    
    if issues:
        print("\n  ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("\n  ✅ No major issues detected!")
    
    print("\n" + "=" * 70)
    print("  END OF DIAGNOSTIC")
    print("=" * 70)

if __name__ == "__main__":
    main()
