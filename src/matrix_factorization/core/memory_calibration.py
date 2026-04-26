"""Local GPU memory calibration profiles and CLI helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import torch

from matrix_factorization.core.experiment import (
    AlgorithmParams,
    ExperimentConfig,
    ExperimentRunner,
    MatrixParams,
    ScanConfig,
    SeedConfig,
    SpreadingConfig,
    TeacherConfig,
    TrainingParams,
)
from matrix_factorization.core.parallel.execution_modes import EstimationParams
from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator


@dataclass(frozen=True)
class MemoryCalibrationProfile:
    key: str
    algorithm_key: str
    matrix: Dict[str, int]
    alpha_values: List[float]
    samples_per_alpha: int = 1
    max_steps: int = 2
    max_epochs: int = 2
    tensor_order: int = 2
    f_distribution: str = "rademacher"
    use_compile: bool = False
    use_bf16: bool = False
    use_tf32: bool = True
    purpose: str = ""
    runtime_class: str = "quick"
    output_root: str = "runs/calibration/memory"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "algorithm_key": self.algorithm_key,
            "matrix": dict(self.matrix),
            "alpha_values": list(self.alpha_values),
            "samples_per_alpha": self.samples_per_alpha,
            "max_steps": self.max_steps,
            "max_epochs": self.max_epochs,
            "tensor_order": self.tensor_order,
            "f_distribution": self.f_distribution,
            "use_compile": self.use_compile,
            "use_bf16": self.use_bf16,
            "use_tf32": self.use_tf32,
            "purpose": self.purpose,
            "runtime_class": self.runtime_class,
            "output_root": self.output_root,
            "notes": self.notes,
        }


def get_memory_calibration_profiles() -> Dict[str, MemoryCalibrationProfile]:
    return {
        "matrix_bigamp_small": MemoryCalibrationProfile(
            key="matrix_bigamp_small",
            algorithm_key="bigamp",
            matrix={"N1": 8, "N2": 8, "M": 2},
            alpha_values=[0.0, 0.1],
            samples_per_alpha=1,
            max_steps=2,
            purpose="校准 dense matrix BiGAMP 的小尺寸峰值显存记录链路。",
        ),
        "matrix_agd_small": MemoryCalibrationProfile(
            key="matrix_agd_small",
            algorithm_key="agd",
            matrix={"N1": 8, "N2": 8, "M": 2},
            alpha_values=[0.0, 0.1],
            samples_per_alpha=1,
            max_steps=2,
            max_epochs=2,
            purpose="校准 AGD 小尺寸峰值显存记录链路。",
        ),
        "spreading_bigamp_small": MemoryCalibrationProfile(
            key="spreading_bigamp_small",
            algorithm_key="bigamp_spreading",
            matrix={"N1": 8, "N2": 8, "M": 2},
            alpha_values=[0.0, 0.1],
            samples_per_alpha=1,
            max_steps=2,
            tensor_order=2,
            purpose="校准 matrix spreading BiGAMP 的小尺寸 graph/F 路径。",
        ),
        "tensor_parallel_small": MemoryCalibrationProfile(
            key="tensor_parallel_small",
            algorithm_key="bigamp_tensor_parallel",
            matrix={"N1": 4, "N2": 4, "M": 2},
            alpha_values=[0.0, 0.1],
            samples_per_alpha=1,
            max_steps=2,
            tensor_order=3,
            purpose="校准 tensor parallel 小尺寸 TensorSuperGraph 路径。",
        ),
    }


def build_calibration_config(profile: MemoryCalibrationProfile) -> ExperimentConfig:
    spreading = None
    if "spreading" in profile.algorithm_key or "tensor" in profile.algorithm_key:
        spreading = SpreadingConfig(
            f_distribution=profile.f_distribution,
            onsager_correction=False,
            allow_intra_connection=False,
            seed=321,
            chunk_size=0,
            tensor_order=profile.tensor_order,
        )
    return ExperimentConfig(
        matrix=MatrixParams(**profile.matrix),
        training=TrainingParams(
            samples_per_alpha=profile.samples_per_alpha,
            max_steps=profile.max_steps,
            max_epochs=profile.max_epochs,
            num_workers=1,
        ),
        algorithm_key=profile.algorithm_key,
        scan=ScanConfig(dimension="alpha", values=list(profile.alpha_values)),
        seeds=SeedConfig(base_seed=42, teacher_seed=12345, spreading_seed=321, student_seed=7),
        algorithm_params=AlgorithmParams(
            damping=0.5,
            noise_var=1.0e-5,
            use_compile=profile.use_compile,
            use_bf16=profile.use_bf16,
            use_tf32=profile.use_tf32,
            seed_partition_policy="partition_invariant",
        ),
        spreading=spreading,
        teacher=TeacherConfig(init_distribution="gaussian"),
        experiment_name=f"calibration_{profile.key}",
        teacher_key="standard",
    )


def explain_memory_profile(profile_key: str) -> str:
    profile = _get_profile(profile_key)
    config = build_calibration_config(profile)
    params = estimation_params_from_config(config)
    estimator = MemoryEstimator()
    estimate = estimator.estimate(params)
    raw_estimate_gb = estimator.estimate_raw(params)
    lines = ["memory calibration profile", ""]
    lines.append(f"key: {profile.key}")
    lines.append(f"algorithm: {profile.algorithm_key}")
    lines.append(f"runtime_class: {profile.runtime_class}")
    lines.append(f"matrix: {profile.matrix}")
    lines.append(f"alpha_values: {profile.alpha_values}")
    lines.append(f"samples_per_alpha: {profile.samples_per_alpha}")
    lines.append(f"max_steps: {profile.max_steps}")
    lines.append(f"dtype: {'bf16' if profile.use_bf16 else 'fp32'}")
    lines.append(f"compile: {profile.use_compile}")
    lines.append(f"tensor_order: {profile.tensor_order}")
    lines.append(f"theoretical_tensor_estimate_gb: {raw_estimate_gb:.6f}")
    lines.append(f"estimated_total_with_runtime_gb: {estimate.total_gb:.6f}")
    lines.append(f"gpu_model: {estimator.gpu_model}")
    lines.append(f"purpose: {profile.purpose}")
    lines.append(f"output_root: {profile.output_root}/{profile.key}")
    return "\n".join(lines)


def run_memory_calibration(profile_key: str, output_root: Optional[Path] = None) -> Dict[str, Any]:
    profile = _get_profile(profile_key)
    config = build_calibration_config(profile)
    estimator = MemoryEstimator()
    params = estimation_params_from_config(config)
    estimate = estimator.estimate(params)
    raw_estimate_gb = estimator.estimate_raw(params)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    runner = ExperimentRunner(device=device, verbose=False)
    started_at = datetime.now()
    runner.run(
        config,
        output_options={
            "save_tensors": False,
            "enable_heatmap": False,
            "checkpoint_path": str(_profile_output_dir(profile, output_root, started_at) / "checkpoints" / "latest.pt"),
        },
        raw_yaml="",
    )

    cuda_available = torch.cuda.is_available()
    actual_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if cuda_available else 0.0
    reserved_peak_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if cuda_available else 0.0
    allocated_after_gb = torch.cuda.memory_allocated() / (1024 ** 3) if cuda_available else 0.0
    reserved_after_gb = torch.cuda.memory_reserved() / (1024 ** 3) if cuda_available else 0.0
    raw_error_ratio = (
        (raw_estimate_gb - actual_peak_gb) / actual_peak_gb
        if actual_peak_gb > 0 else None
    )
    total_error_ratio = (
        (estimate.total_gb - actual_peak_gb) / actual_peak_gb
        if actual_peak_gb > 0 else None
    )

    output_dir = _profile_output_dir(profile, output_root, started_at)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "profile": profile.to_dict(),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "gpu": {
            "cuda_available": cuda_available,
            "name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
            "total_memory_gb": (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if cuda_available else 0.0
            ),
            "free_memory_gb_at_estimate": estimator.get_available_memory(),
        },
        "estimation_params": params.to_dict(),
        "theoretical_estimate_gb": raw_estimate_gb,
        "estimated_total_with_runtime_gb": estimate.total_gb,
        "actual_peak_memory_gb": actual_peak_gb,
        "reserved_peak_memory_gb": reserved_peak_gb,
        "allocated_after_gb": allocated_after_gb,
        "reserved_after_gb": reserved_after_gb,
        "error_ratio": raw_error_ratio,
        "error_pct": raw_error_ratio * 100 if raw_error_ratio is not None else None,
        "total_error_ratio": total_error_ratio,
        "total_error_pct": total_error_ratio * 100 if total_error_ratio is not None else None,
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def estimation_params_from_config(config: ExperimentConfig) -> EstimationParams:
    spreading = config.spreading
    tensor_order = getattr(spreading, "tensor_order", 2) if spreading else 2
    tensor_dims = None
    if tensor_order >= 2:
        dims = [config.matrix.N1, config.matrix.N2]
        for _ in range(2, tensor_order):
            dims.append(config.matrix.N1)
        tensor_dims = tuple(dims)
    return EstimationParams(
        N1=config.matrix.N1,
        N2=config.matrix.N2,
        M=config.matrix.M,
        S=config.training.samples_per_alpha,
        alpha_values=[float(value) for value in config.scan.values],
        algorithm_key=config.algorithm_key,
        use_compile=config.algorithm_params.use_compile,
        use_bf16=config.algorithm_params.use_bf16,
        f_distribution=getattr(spreading, "f_distribution", "rademacher") if spreading else "rademacher",
        adaptive_damping=config.algorithm_params.adaptive_damping,
        allow_intra_connection=getattr(spreading, "allow_intra_connection", False) if spreading else False,
        tensor_order=tensor_order,
        tensor_dims=tensor_dims,
        seed_partition_policy=config.algorithm_params.seed_partition_policy,
    )


def _get_profile(profile_key: str) -> MemoryCalibrationProfile:
    profiles = get_memory_calibration_profiles()
    if profile_key not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"unknown memory calibration profile '{profile_key}'. Available: {available}")
    return profiles[profile_key]


def _profile_output_dir(
    profile: MemoryCalibrationProfile,
    output_root: Optional[Path],
    timestamp: datetime,
) -> Path:
    root = Path(output_root) if output_root is not None else Path(profile.output_root)
    return root / profile.key / timestamp.strftime("%Y%m%d_%H%M%S")
