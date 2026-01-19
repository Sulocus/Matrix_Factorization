"""
Calibration Runner for Smart Parallel Module.

Runs calibration tests, collects results, computes calibration factors,
and validates estimation accuracy.
"""
from dataclasses import dataclass
from typing import List, Optional, Any
from pathlib import Path
import subprocess
import json
import sys
import re
import logging
import statistics

import torch

from .test_generator import TestConfig, generate_test_configs, generate_test_script

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """Raised when calibration fails validation."""
    pass


@dataclass
class CalibrationResult:
    """Result of a single calibration test."""
    config: TestConfig
    estimated_gb: float
    actual_gb: float
    error_pct: float
    success: bool
    error_message: Optional[str] = None
    
    @property
    def is_overestimate(self) -> bool:
        """True if estimation was higher than actual."""
        return self.estimated_gb > self.actual_gb
    
    @property
    def is_underestimate(self) -> bool:
        """True if estimation was lower than actual (risky)."""
        return self.estimated_gb < self.actual_gb


@dataclass
class CalibrationSummary:
    """Summary of calibration run."""
    algorithm_key: str
    gpu_model: str
    results: List[CalibrationResult]
    calibration_factor: float
    mean_error_pct: float
    max_error_pct: float
    min_error_pct: float
    validation_passed: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "algorithm_key": self.algorithm_key,
            "gpu_model": self.gpu_model,
            "calibration_factor": self.calibration_factor,
            "mean_error_pct": self.mean_error_pct,
            "max_error_pct": self.max_error_pct,
            "min_error_pct": self.min_error_pct,
            "validation_passed": self.validation_passed,
            "num_tests": len(self.results),
            "num_passed": sum(1 for r in self.results if r.success),
        }


class CalibrationRunner:
    """
    Calibration test runner.
    
    Runs calibration tests, analyzes results, and produces calibration factors
    for memory estimation.
    
    Example:
        from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator
        
        runner = CalibrationRunner(MemoryEstimator())
        summary = runner.run_calibration("bigamp_spreading_parallel")
        
        if summary.validation_passed:
            print(f"Calibration factor: {summary.calibration_factor}")
    """
    
    # Error bounds for validation
    # Negative = underestimate (dangerous), Positive = overestimate (safe)
    MIN_ERROR_BOUND = -0.15  # -15% (underestimate limit)
    MAX_ERROR_BOUND = 0.10   # +10% (overestimate limit, relaxed for safety)
    
    def __init__(
        self,
        estimator: Any,
        temp_dir: Optional[Path] = None,
    ):
        """
        Initialize calibration runner.
        
        Args:
            estimator: MemoryEstimator instance
            temp_dir: Directory for temporary test scripts
        """
        self.estimator = estimator
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            import tempfile
            self.temp_dir = Path(tempfile.gettempdir()) / "mf_calibration"
        
        # Get GPU info
        if torch.cuda.is_available():
            self.gpu_model = torch.cuda.get_device_name(0).replace(" ", "_")
            self.available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            self.gpu_model = "cpu"
            self.available_memory = 8.0
    
    def run_calibration(
        self,
        algo_key: str,
        comprehensive: bool = False,
        save_results: bool = True,
    ) -> CalibrationSummary:
        """
        Run full calibration for an algorithm.
        
        Args:
            algo_key: Algorithm key to calibrate
            comprehensive: Use comprehensive parameter grid
            save_results: Save results to calibration data directory
            
        Returns:
            CalibrationSummary with results and factor
        """
        logger.info(f"Starting calibration for {algo_key} on {self.gpu_model}")
        
        # Generate test configurations
        configs = generate_test_configs(
            algo_key,
            self.available_memory,
            comprehensive=comprehensive,
        )
        
        logger.info(f"Generated {len(configs)} test configurations")
        
        # Run tests
        results = []
        for i, config in enumerate(configs):
            logger.info(f"Running test {i+1}/{len(configs)}: N={config.N}, M={config.M}")
            
            result = self._run_single_test(config)
            results.append(result)
            
            if not result.success:
                logger.warning(f"Test failed: {result.error_message}")
        
        # Compute calibration factor
        factor = self._compute_calibration_factor(results)
        
        # Validate with factor applied
        validation_passed = self._validate_with_factor(results, factor)
        
        # Compute statistics
        successful_results = [r for r in results if r.success]
        if successful_results:
            errors = [r.error_pct for r in successful_results]
            mean_error = statistics.mean(errors)
            max_error = max(errors)
            min_error = min(errors)
        else:
            mean_error = max_error = min_error = 0.0
        
        summary = CalibrationSummary(
            algorithm_key=algo_key,
            gpu_model=self.gpu_model,
            results=results,
            calibration_factor=factor,
            mean_error_pct=mean_error,
            max_error_pct=max_error,
            min_error_pct=min_error,
            validation_passed=validation_passed,
        )
        
        # Save results
        if save_results:
            self._save_calibration(summary)
        
        logger.info(
            f"Calibration complete: factor={factor:.3f}, "
            f"mean_error={mean_error:.1f}%, "
            f"validation={'PASSED' if validation_passed else 'FAILED'}"
        )
        
        return summary
    
    def run_quick_validation(
        self,
        algo_key: str,
        num_tests: int = 5,
    ) -> bool:
        """
        Run quick validation to check if current calibration is still valid.
        
        Args:
            algo_key: Algorithm key
            num_tests: Number of test points
            
        Returns:
            True if validation passes
        """
        # Generate small set of diverse test configs
        all_configs = generate_test_configs(
            algo_key,
            self.available_memory,
            comprehensive=False,
        )
        
        # Sample evenly across the range
        step = max(1, len(all_configs) // num_tests)
        configs = all_configs[::step][:num_tests]
        
        results = []
        for config in configs:
            result = self._run_single_test(config)
            results.append(result)
        
        # Check if all results are within bounds
        for result in results:
            if result.success:
                if result.error_pct < self.MIN_ERROR_BOUND * 100:
                    logger.warning(f"Underestimate detected: {result.error_pct:.1f}%")
                    return False
                if result.error_pct > self.MAX_ERROR_BOUND * 100:
                    logger.warning(f"Overestimate detected: {result.error_pct:.1f}%")
                    return False
        
        return True
    
    def _run_single_test(self, config: TestConfig) -> CalibrationResult:
        """Run a single calibration test."""
        # Generate test script
        script_path = self.temp_dir / f"test_{config.N}_{config.M}.py"
        generate_test_script(config, script_path)
        
        try:
            # Run script in subprocess
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=config.duration_seconds + 30,  # Extra buffer
            )
            
            if result.returncode != 0:
                return CalibrationResult(
                    config=config,
                    estimated_gb=0,
                    actual_gb=0,
                    error_pct=0,
                    success=False,
                    error_message=f"Script failed: {result.stderr[:200]}",
                )
            
            # Parse output for CALIBRATION_RESULT
            output = result.stdout
            match = re.search(r'CALIBRATION_RESULT:\s*(\{[^}]+\})', output, re.DOTALL)
            
            if not match:
                # Try to parse JSON directly after the marker
                try:
                    json_start = output.find('CALIBRATION_RESULT:')
                    if json_start >= 0:
                        json_str = output[json_start + len('CALIBRATION_RESULT:'):]
                        # Find the JSON object
                        brace_start = json_str.find('{')
                        if brace_start >= 0:
                            # Simple JSON extraction
                            depth = 0
                            end = brace_start
                            for i, c in enumerate(json_str[brace_start:], brace_start):
                                if c == '{':
                                    depth += 1
                                elif c == '}':
                                    depth -= 1
                                    if depth == 0:
                                        end = i + 1
                                        break
                            json_str = json_str[brace_start:end]
                            data = json.loads(json_str)
                        else:
                            raise ValueError("No JSON object found")
                    else:
                        raise ValueError("No CALIBRATION_RESULT marker")
                except Exception as e:
                    return CalibrationResult(
                        config=config,
                        estimated_gb=0,
                        actual_gb=0,
                        error_pct=0,
                        success=False,
                        error_message=f"Could not parse output: {e}",
                    )
            else:
                data = json.loads(match.group(1))
            
            estimated = data.get("estimated_gb", 0)
            actual = data.get("peak_gb", 0)
            
            if actual <= 0:
                return CalibrationResult(
                    config=config,
                    estimated_gb=estimated,
                    actual_gb=0,
                    error_pct=0,
                    success=False,
                    error_message="No peak memory recorded",
                )
            
            error_pct = ((estimated - actual) / actual) * 100
            
            return CalibrationResult(
                config=config,
                estimated_gb=estimated,
                actual_gb=actual,
                error_pct=error_pct,
                success=True,
            )
            
        except subprocess.TimeoutExpired:
            return CalibrationResult(
                config=config,
                estimated_gb=0,
                actual_gb=0,
                error_pct=0,
                success=False,
                error_message="Test timed out",
            )
        except Exception as e:
            return CalibrationResult(
                config=config,
                estimated_gb=0,
                actual_gb=0,
                error_pct=0,
                success=False,
                error_message=str(e),
            )
    
    def _compute_calibration_factor(
        self,
        results: List[CalibrationResult],
    ) -> float:
        """
        Compute calibration factor from test results.
        
        Uses median of (actual / estimated) ratios for robustness.
        """
        successful = [r for r in results if r.success and r.actual_gb > 0 and r.estimated_gb > 0]
        
        if not successful:
            logger.warning("No successful tests, using factor 1.0")
            return 1.0
        
        # Compute ratio for each result
        ratios = [r.actual_gb / r.estimated_gb for r in successful]
        
        # Use median for robustness against outliers
        factor = statistics.median(ratios)
        
        # Sanity check: factor should be between 0.5 and 2.0
        factor = max(0.5, min(2.0, factor))
        
        return factor
    
    def _validate_with_factor(
        self,
        results: List[CalibrationResult],
        factor: float,
    ) -> bool:
        """
        Validate that applying factor brings errors within bounds.
        """
        successful = [r for r in results if r.success]
        
        for result in successful:
            calibrated_estimate = result.estimated_gb * factor
            error = (calibrated_estimate - result.actual_gb) / result.actual_gb
            
            if error < self.MIN_ERROR_BOUND:
                logger.warning(
                    f"Validation failed: {result.config.N}x{result.config.M} "
                    f"underestimates by {error:.1%}"
                )
                return False
            
            if error > self.MAX_ERROR_BOUND:
                logger.warning(
                    f"Validation failed: {result.config.N}x{result.config.M} "
                    f"overestimates by {error:.1%}"
                )
                return False
        
        return True
    
    def _save_calibration(self, summary: CalibrationSummary) -> None:
        """Save calibration results to data directory."""
        data_dir = Path(self.estimator.calibration_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data or create new
        calib_file = data_dir / f"{self.gpu_model}.json"
        
        if calib_file.exists():
            with open(calib_file) as f:
                data = json.load(f)
        else:
            data = {}
        
        # Update with new calibration
        data[summary.algorithm_key] = {
            "factor": summary.calibration_factor,
            "mean_error_pct": summary.mean_error_pct,
            "num_tests": len(summary.results),
            "validation_passed": summary.validation_passed,
        }
        
        with open(calib_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved calibration to {calib_file}")
