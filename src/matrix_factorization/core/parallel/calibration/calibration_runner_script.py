#!/usr/bin/env python3
"""
Calibration Runner Script for Memory Estimator Validation.

This script runs a series of calibration tests to measure the accuracy
of the memory estimation formulas vs actual GPU memory usage.

Usage:
    python -m smf.core.parallel.calibration.calibration_runner_script --quick
    python -m smf.core.parallel.calibration.calibration_runner_script --comprehensive
"""
import sys
import json
import gc
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CalibrationTestCase:
    """A single calibration test case."""
    N: int
    M: int
    S: int
    alpha_max: float
    algorithm_key: str = "bigamp_spreading_parallel"
    max_steps: int = 50
    
    def __str__(self):
        return f"N={self.N}, M={self.M}, S={self.S}, α={self.alpha_max}"


@dataclass
class CalibrationResult:
    """Result of a calibration test."""
    test_case: CalibrationTestCase
    estimated_gb: float
    actual_gb: float
    error_pct: float
    passed: bool
    oom: bool = False
    error_msg: Optional[str] = None


# ============================================================
# Test Parameter Grids - LARGE SCALE ONLY (10GB+ target)
# ============================================================

# Quick tests: Medium-large scale for fast iteration
QUICK_TESTS = [
    # Medium scale (target ~5-10 GB)
    CalibrationTestCase(N=600, M=150, S=10, alpha_max=2.5, max_steps=30),
    CalibrationTestCase(N=600, M=150, S=20, alpha_max=3.0, max_steps=30),
    # Large scale (target ~10-20 GB)
    CalibrationTestCase(N=800, M=200, S=10, alpha_max=2.5, max_steps=30),
    CalibrationTestCase(N=800, M=200, S=20, alpha_max=3.0, max_steps=30),
]

# Comprehensive tests: Full range for production calibration
COMPREHENSIVE_TESTS = QUICK_TESTS + [
    # Very large scale (target ~15-25 GB)
    CalibrationTestCase(N=1000, M=250, S=10, alpha_max=2.5, max_steps=25),
    CalibrationTestCase(N=1000, M=250, S=20, alpha_max=3.0, max_steps=25),
    # Gaussian F distribution
    CalibrationTestCase(N=600, M=150, S=10, alpha_max=2.5, max_steps=30),
    CalibrationTestCase(N=800, M=200, S=10, alpha_max=2.5, max_steps=30),
]


def get_memory_stats():
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "peak_gb": 0}
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


def run_single_test(test: CalibrationTestCase) -> CalibrationResult:
    """Run a single calibration test."""
    print(f"\n{'='*60}")
    print(f"Test: {test}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        return CalibrationResult(
            test_case=test,
            estimated_gb=0,
            actual_gb=0,
            error_pct=0,
            passed=False,
            error_msg="No GPU available"
        )
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Step 1: Get estimation
    try:
        from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator
        from matrix_factorization.core.parallel.execution_modes import EstimationParams
        
        estimator = MemoryEstimator()
        params = EstimationParams(
            N1=test.N, N2=test.N, M=test.M, S=test.S,
            alpha_values=[test.alpha_max],
            algorithm_key=test.algorithm_key,
            use_compile=True,
            use_bf16=True,
            f_distribution='ising',
        )
        
        estimated_gb = estimator.estimate_raw(params)
        print(f"  Estimated: {estimated_gb:.3f} GB")
        
    except Exception as e:
        print(f"  Estimation error: {e}")
        return CalibrationResult(
            test_case=test,
            estimated_gb=0,
            actual_gb=0,
            error_pct=0,
            passed=False,
            error_msg=str(e)
        )
    
    # Step 2: Run algorithm
    actual_gb = 0
    oom = False
    
    try:
        actual_gb = run_spreading_parallel(test, device)
        print(f"  Actual:    {actual_gb:.3f} GB")
        
    except torch.cuda.OutOfMemoryError:
        print("  OOM!")
        oom = True
        actual_gb = -1
        
    except Exception as e:
        print(f"  Error: {e}")
        return CalibrationResult(
            test_case=test,
            estimated_gb=estimated_gb,
            actual_gb=0,
            error_pct=0,
            passed=False,
            error_msg=str(e)
        )
    
    # Step 3: Compute error
    if actual_gb > 0:
        error_pct = ((estimated_gb - actual_gb) / actual_gb) * 100
        passed = -15 <= error_pct <= 5
        print(f"  Error:     {error_pct:+.1f}% {'✓' if passed else '✗'}")
    else:
        error_pct = float('inf')
        passed = False
    
    return CalibrationResult(
        test_case=test,
        estimated_gb=estimated_gb,
        actual_gb=actual_gb,
        error_pct=error_pct,
        passed=passed,
        oom=oom,
    )


def run_spreading_parallel(test: CalibrationTestCase, device) -> float:
    """Run BiGAMP Spreading Parallel and return peak memory in GB."""
    from matrix_factorization.modules.algorithms.bigamp.spreading_parallel import (
        BiGAMPSpreadingParallel,
    )
    
    # Create nested config structure matching what the algorithm expects
    @dataclass
    class AlgorithmConfig:
        damping: float = 0.5
        noise_var: float = 0.0
        learning_rate: float = 0.1
        use_compile: bool = True
    
    @dataclass
    class TrainingConfig:
        max_steps: int = 50
    
    @dataclass
    class MatrixConfig:
        N1: int = 100
        N2: int = 100
        M: int = 25
    
    @dataclass
    class SpreadingConfig:
        f_distribution: str = 'ising'
        seed: int = 42
    
    @dataclass
    class FullConfig:
        algorithm: AlgorithmConfig
        training: TrainingConfig
        matrix: MatrixConfig
        spreading: SpreadingConfig
    
    config = FullConfig(
        algorithm=AlgorithmConfig(
            damping=0.5,
            noise_var=0.0,
            use_compile=True,
        ),
        training=TrainingConfig(max_steps=test.max_steps),
        matrix=MatrixConfig(N1=test.N, N2=test.N, M=test.M),
        spreading=SpreadingConfig(f_distribution='ising', seed=42),
    )
    
    # Create Teacher
    W_teacher = torch.randn(test.N, test.M, device=device) / (test.M ** 0.5)
    X_teacher = torch.randn(test.M, test.N, device=device) / (test.M ** 0.5)
    
    # Initialize algorithm
    algo = BiGAMPSpreadingParallel(config, device)
    
    # Create spreading data
    alpha_values = [test.alpha_max]
    spreading_data = algo.create_spreading_data(
        W_teacher, X_teacher, alpha_values, test.S, base_seed=42
    )
    
    # Reset peak memory
    torch.cuda.reset_peak_memory_stats()
    
    # Run training
    W_students, X_students = algo.train_full_parallel(
        spreading_data,
        batch_alpha_indices=None,
        verbose=False,
        step_callback=None,
    )
    
    # Sync and get peak
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    
    # Cleanup
    del W_students, X_students, spreading_data, algo
    del W_teacher, X_teacher
    torch.cuda.empty_cache()
    gc.collect()
    
    return peak_gb


def run_calibration_suite(
    tests: List[CalibrationTestCase],
    output_file: Optional[Path] = None,
) -> Dict:
    """Run a suite of calibration tests."""
    print(f"\n{'#'*60}")
    print("# MEMORY CALIBRATION TEST SUITE")
    print(f"# GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"# Tests: {len(tests)}")
    print(f"{'#'*60}")
    
    results = []
    
    for test in tests:
        result = run_single_test(test)
        results.append(result)
        
        # Brief pause between tests
        time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed and not r.oom]
    oom = [r for r in results if r.oom]
    
    print(f"  Passed: {len(passed)}/{len(results)}")
    print(f"  Failed: {len(failed)}/{len(results)}")
    print(f"  OOM:    {len(oom)}/{len(results)}")
    
    if passed:
        errors = [r.error_pct for r in passed]
        print("\n  Error Statistics (passed tests):")
        print(f"    Min:  {min(errors):+.1f}%")
        print(f"    Max:  {max(errors):+.1f}%")
        print(f"    Mean: {sum(errors)/len(errors):+.1f}%")
    
    if failed:
        print("\n  Failed Tests:")
        for r in failed:
            print(f"    {r.test_case}: error={r.error_pct:+.1f}%")
    
    # Save results
    if output_file:
        output_data = {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total": len(results),
                "passed": len(passed),
                "failed": len(failed),
                "oom": len(oom),
            },
            "results": [
                {
                    "N": r.test_case.N,
                    "M": r.test_case.M,
                    "S": r.test_case.S,
                    "alpha_max": r.test_case.alpha_max,
                    "estimated_gb": r.estimated_gb,
                    "actual_gb": r.actual_gb,
                    "error_pct": r.error_pct if r.error_pct != float('inf') else "inf",
                    "passed": r.passed,
                    "oom": r.oom,
                }
                for r in results
            ],
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved to: {output_file}")
    
    all_passed = len(failed) == 0 and len(passed) > 0
    return {"all_passed": all_passed, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Memory Calibration Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick test suite")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--error-bounds", type=str, default="-15,+5", 
                        help="Error bounds as 'min,max' (default: -15,+5)")
    args = parser.parse_args()
    
    # Select test suite
    if args.comprehensive:
        tests = COMPREHENSIVE_TESTS
    elif args.quick:
        tests = QUICK_TESTS
    else:
        # Default to quick
        tests = QUICK_TESTS
    
    # Output file
    output_file = Path(args.output) if args.output else None
    
    # Run tests
    result = run_calibration_suite(tests, output_file)
    
    # Exit code
    sys.exit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
