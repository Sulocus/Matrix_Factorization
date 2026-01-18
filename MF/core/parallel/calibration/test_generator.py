"""
Calibration Test Generator for Smart Parallel Module.

Generates test configurations and scripts for memory estimation calibration
across different parameter combinations.
"""
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import itertools


@dataclass
class TestConfig:
    """Single calibration test configuration."""
    N: int
    M: int
    S: int
    alpha_max: float
    duration_seconds: int  # Test duration
    test_dynamic_cache: bool  # Whether to test torch.compile dynamic behavior
    algorithm_key: str = "bigamp_spreading_parallel"
    
    @property
    def rough_memory_estimate(self) -> float:
        """Quick memory estimate for filtering."""
        # Very rough: N^2 * M * S * alpha / 1e9
        return (self.N * self.N * self.M * self.S * self.alpha_max) / 1e9
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "N": self.N,
            "M": self.M,
            "S": self.S,
            "alpha_max": self.alpha_max,
            "duration_seconds": self.duration_seconds,
            "test_dynamic_cache": self.test_dynamic_cache,
            "algorithm_key": self.algorithm_key,
        }


def generate_test_configs(
    algo_key: str,
    available_memory_gb: float,
    comprehensive: bool = False,
) -> List[TestConfig]:
    """
    Generate test parameter combinations for calibration.
    
    Args:
        algo_key: Algorithm key to calibrate
        available_memory_gb: Available GPU memory
        comprehensive: If True, generate more test points
        
    Returns:
        List of TestConfig objects
    """
    # Parameter grids
    if comprehensive:
        N_values = [100, 200, 300, 400, 500, 600, 800, 1000]
        M_ratios = [0.1, 0.2, 0.25, 0.33, 0.5]
        S_values = [1, 2, 5, 10, 20, 50]
        alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    else:
        # Quick calibration
        N_values = [200, 400, 600]
        M_ratios = [0.25, 0.5]
        S_values = [2, 10, 20]
        alpha_values = [1.0, 2.0, 3.0, 4.0]
    
    configs = []
    
    for N, M_ratio, S, alpha_max in itertools.product(
        N_values, M_ratios, S_values, alpha_values
    ):
        M = max(1, int(N * M_ratio))
        
        # Quick estimate to filter impossible combinations
        rough_estimate = (N * N * M * S * alpha_max) / 1e9
        
        # Skip combinations that definitely won't fit
        if rough_estimate > available_memory_gb * 0.95:
            continue
        
        # Skip very small combinations (not useful for calibration)
        if rough_estimate < 0.1:
            continue
        
        # Determine test duration based on size
        if rough_estimate > 20:
            duration = 60  # Large: 60 seconds
        elif rough_estimate > 10:
            duration = 30  # Medium: 30 seconds
        elif rough_estimate > 5:
            duration = 15  # Small: 15 seconds
        else:
            duration = 10  # Tiny: 10 seconds
        
        # Test dynamic cache for larger problems
        test_dynamic = rough_estimate > 15
        
        configs.append(TestConfig(
            N=N,
            M=M,
            S=S,
            alpha_max=alpha_max,
            duration_seconds=duration,
            test_dynamic_cache=test_dynamic,
            algorithm_key=algo_key,
        ))
    
    return configs


def generate_test_script(
    config: TestConfig,
    output_path: Path,
    include_warmup: bool = True,
    max_steps: int = 50,
) -> Path:
    """
    Generate a standalone calibration test script that ACTUALLY runs the algorithm.
    
    Key improvement: This script now executes the full algorithm to measure
    real peak memory, not just import and idle.
    
    Args:
        config: Test configuration
        output_path: Where to save the script
        include_warmup: Include warmup iterations
        max_steps: Number of algorithm steps to run (50 is enough to reach peak)
        
    Returns:
        Path to generated script
    """
    script = f'''#!/usr/bin/env python3
"""
Auto-generated calibration test for {config.algorithm_key}
Parameters: N={config.N}, M={config.M}, S={config.S}, alpha_max={config.alpha_max}

This script ACTUALLY runs the algorithm to measure real peak memory usage.
"""
import sys
import gc
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_memory_stats():
    """Get current memory statistics."""
    if not torch.cuda.is_available():
        return {{"allocated": 0, "reserved": 0, "peak": 0}}
    
    return {{
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }}


def run_calibration_test():
    """Run the calibration test with ACTUAL algorithm execution."""
    # Parameters
    N = {config.N}
    M = {config.M}
    S = {config.S}
    alpha_max = {config.alpha_max}
    algo_key = "{config.algorithm_key}"
    max_steps = {max_steps}
    
    print(f"="*60)
    print(f"Calibration Test: {{algo_key}}")
    print(f"  N={{N}}, M={{M}}, S={{S}}, alpha_max={{alpha_max}}")
    print(f"  Steps: {{max_steps}}")
    print(f"="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        return None
    
    # Clear GPU memory completely
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Record baseline
    baseline = get_memory_stats()
    print(f"Baseline memory: {{baseline['allocated_gb']:.3f}} GB")
    
    # ========================================
    # Step 1: Get memory estimation
    # ========================================
    try:
        from MF.core.parallel.memory_estimator import MemoryEstimator
        from MF.core.parallel.execution_modes import EstimationParams
        
        estimator = MemoryEstimator()
        
        params = EstimationParams(
            N1=N, N2=N, M=M, S=S,
            alpha_values=[alpha_max],
            algorithm_key=algo_key,
            use_compile=True,
            use_bf16=True,
            f_distribution='rademacher',
        )
        
        estimated_gb = estimator.estimate_raw(params)
        print(f"Estimated memory: {{estimated_gb:.3f}} GB")
        
    except Exception as e:
        print(f"Estimation error: {{e}}")
        estimated_gb = 0
    
    # ========================================
    # Step 2: Run ACTUAL algorithm
    # ========================================
    actual_peak_gb = 0
    
    try:
        if algo_key == "bigamp_spreading_parallel":
            actual_peak_gb = _run_spreading_parallel(
                N, M, S, alpha_max, max_steps, device
            )
        elif algo_key == "bigamp":
            actual_peak_gb = _run_bigamp_standard(
                N, M, S, alpha_max, max_steps, device
            )
        elif algo_key == "agd":
            actual_peak_gb = _run_agd(
                N, M, S, alpha_max, max_steps, device
            )
        else:
            print(f"Unknown algorithm: {{algo_key}}")
            return None
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM Error: {{e}}")
        actual_peak_gb = -1  # Indicate OOM
    except Exception as e:
        print(f"Algorithm error: {{e}}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================
    # Step 3: Compute results
    # ========================================
    if actual_peak_gb > 0:
        error_pct = ((estimated_gb - actual_peak_gb) / actual_peak_gb) * 100
    else:
        error_pct = float('inf')
    
    result = {{
        "N": N,
        "M": M,
        "S": S,
        "alpha_max": alpha_max,
        "algo_key": algo_key,
        "max_steps": max_steps,
        "baseline_gb": baseline["allocated_gb"],
        "estimated_gb": estimated_gb,
        "peak_gb": actual_peak_gb,
        "error_pct": error_pct,
    }}
    
    print(f"\\n" + "="*60)
    print("CALIBRATION_RESULT:")
    print(json.dumps(result, indent=2))
    print("="*60)
    
    # Status summary
    if actual_peak_gb < 0:
        print("\\n*** OOM - Parameters too large for this GPU ***")
    elif -15 <= error_pct <= 5:
        print(f"\\n✓ PASS: Error {{error_pct:.1f}}% is within [-15%, +5%]")
    else:
        print(f"\\n✗ FAIL: Error {{error_pct:.1f}}% is outside [-15%, +5%]")
    
    return result


def _run_spreading_parallel(N, M, S, alpha_max, max_steps, device):
    """Run BiGAMP Spreading Parallel and return peak memory."""
    from MF.modules.algorithms.bigamp_spreading_parallel import (
        BiGAMPSpreadingParallel,
        SuperGraphData,
        generate_F_super,
        compute_Y_super,
    )
    
    print(f"\\nRunning BiGAMP Spreading Parallel...")
    
    # Create a minimal config object
    @dataclass
    class MinimalConfig:
        N1: int = N
        N2: int = N
        M: int = M
        S: int = S
        max_steps: int = max_steps
        damping: float = 0.5
        noise_var: float = 0.0
        learning_rate: float = 0.1
        use_bf16: bool = True
        teacher_key: str = 'standard'
        f_distribution: str = 'rademacher'
    
    config = MinimalConfig()
    
    # Create Teacher matrices
    print("  Creating Teacher matrices...")
    W_teacher = torch.randn(N, M, device=device) / (M ** 0.5)
    X_teacher = torch.randn(M, N, device=device) / (M ** 0.5)
    
    # Create SuperGraph
    print(f"  Creating SuperGraph for alpha={{alpha_max}}...")
    alpha_values = [alpha_max]
    supergraph = SuperGraphData.create(
        N1=N, N2=N, M=M, S=S,
        alpha_values=alpha_values,
        base_seed=42,
        device=device,
    )
    
    # Generate F and Y
    print("  Generating F and Y...")
    F_super = generate_F_super(
        supergraph, M, base_seed=42, device=device,
        f_distribution='rademacher'
    )
    Y_super = compute_Y_super(W_teacher, X_teacher, supergraph, F_super)
    
    # Initialize algorithm
    print("  Initializing algorithm...")
    algo = BiGAMPSpreadingParallel(config, device)
    
    # Create spreading data
    spreading_data = algo.create_spreading_data(
        W_teacher, X_teacher, alpha_values, S, base_seed=42
    )
    
    # Reset peak memory before training
    torch.cuda.reset_peak_memory_stats()
    
    # Run training
    print(f"  Running {{max_steps}} steps...")
    W_students, X_students = algo.train_full_parallel(
        spreading_data,
        batch_alpha_indices=None,
        verbose=False,
        step_callback=None,
    )
    
    # Synchronize and get peak memory
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"  Peak memory: {{peak_gb:.3f}} GB")
    
    # Cleanup
    del W_students, X_students, spreading_data, algo
    del F_super, Y_super, supergraph, W_teacher, X_teacher
    torch.cuda.empty_cache()
    gc.collect()
    
    return peak_gb


def _run_bigamp_standard(N, M, S, alpha_max, max_steps, device):
    """Run standard BiGAMP and return peak memory."""
    # Placeholder - implement if needed
    print("Standard BiGAMP calibration not yet implemented")
    return 0


def _run_agd(N, M, S, alpha_max, max_steps, device):
    """Run AGD and return peak memory."""
    # Placeholder - implement if needed
    print("AGD calibration not yet implemented")
    return 0


if __name__ == "__main__":
    result = run_calibration_test()
    if result:
        sys.exit(0 if -15 <= result.get("error_pct", 100) <= 5 else 1)
    else:
        sys.exit(1)
'''
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script)
    output_path.chmod(0o755)  # Make executable
    
    return output_path


def generate_batch_test_script(
    configs: List[TestConfig],
    output_dir: Path,
) -> Path:
    """
    Generate a batch script that runs all calibration tests.
    
    Args:
        configs: List of test configurations
        output_dir: Directory for generated scripts
        
    Returns:
        Path to batch runner script
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual test scripts
    script_paths = []
    for i, config in enumerate(configs):
        script_name = f"test_{config.N}_{config.M}_{config.S}_{config.alpha_max}.py"
        script_path = output_dir / script_name
        generate_test_script(config, script_path)
        script_paths.append(script_path)
    
    # Generate batch runner
    batch_script = output_dir / "run_all_tests.py"
    
    script_lines = [
        "#!/usr/bin/env python3",
        '"""Batch calibration test runner."""',
        "import subprocess",
        "import sys",
        "import json",
        "from pathlib import Path",
        "",
        "def main():",
        "    results = []",
        f"    test_dir = Path(__file__).parent",
        "",
    ]
    
    for i, config in enumerate(configs):
        script_name = f"test_{config.N}_{config.M}_{config.S}_{config.alpha_max}.py"
        script_lines.extend([
            f'    print(f"Running test {i+1}/{len(configs)}: {script_name}")',
            f'    result = subprocess.run(',
            f'        [sys.executable, test_dir / "{script_name}"],',
            f'        capture_output=True, text=True',
            f'    )',
            f'    print(result.stdout)',
            f'    if result.returncode != 0:',
            f'        print(f"Error: {{result.stderr}}")',
            "",
        ])
    
    script_lines.extend([
        '    print("\\nAll tests complete!")',
        "",
        'if __name__ == "__main__":',
        "    main()",
    ])
    
    batch_script.write_text("\n".join(script_lines))
    batch_script.chmod(0o755)
    
    return batch_script
