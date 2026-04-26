"""Local GPU memory calibration profiles and CLI helpers.

The small profiles validate that the calibration command path works.  The
target-memory profiles are the ones that are useful for smart batching: they
choose problem sizes whose theoretical tensor footprint is in the GB range, so
the fixed CUDA/PyTorch context cost no longer dominates the calibration signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import gc
import os
import threading
import time

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
    target_tensor_gb: Optional[float] = None
    calibration_kind: str = "small"
    sampler_interval_s: float = 0.1

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
            "target_tensor_gb": self.target_tensor_gb,
            "calibration_kind": self.calibration_kind,
            "sampler_interval_s": self.sampler_interval_s,
        }


class MemoryTimelineSampler:
    """Background sampler for the full VRAM curve during a calibration run."""

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = max(float(interval_s), 0.02)
        self.samples: List[Dict[str, Any]] = []
        self._started_at: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._stop_event.clear()
        self._append_sample(stage="start")
        self._thread = threading.Thread(target=self._run, name="mf-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        self._append_sample(stage="stop")

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for sample in self.samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"sample_count": 0}
        baseline = self.samples[0]
        peak_allocated = max(float(s.get("torch_allocated_gb", 0.0)) for s in self.samples)
        peak_reserved = max(float(s.get("torch_reserved_gb", 0.0)) for s in self.samples)
        peak_max_allocated = max(float(s.get("torch_max_allocated_gb", 0.0)) for s in self.samples)
        peak_max_reserved = max(float(s.get("torch_max_reserved_gb", 0.0)) for s in self.samples)
        peak_cuda_used = max(float(s.get("cuda_device_used_gb", 0.0)) for s in self.samples)
        baseline_cuda_used = float(baseline.get("cuda_device_used_gb", 0.0))
        baseline_allocated = float(baseline.get("torch_allocated_gb", 0.0))
        baseline_reserved = float(baseline.get("torch_reserved_gb", 0.0))
        return {
            "sample_count": len(self.samples),
            "duration_s": float(self.samples[-1].get("time_s", 0.0)),
            "baseline_torch_allocated_gb": baseline_allocated,
            "baseline_torch_reserved_gb": baseline_reserved,
            "baseline_cuda_device_used_gb": baseline_cuda_used,
            "peak_torch_allocated_gb": peak_allocated,
            "peak_torch_reserved_gb": peak_reserved,
            "peak_torch_max_allocated_gb": peak_max_allocated,
            "peak_torch_max_reserved_gb": peak_max_reserved,
            "peak_cuda_device_used_gb": peak_cuda_used,
            "delta_peak_torch_allocated_gb": max(0.0, peak_allocated - baseline_allocated),
            "delta_peak_torch_reserved_gb": max(0.0, peak_reserved - baseline_reserved),
            "delta_peak_torch_max_allocated_gb": max(0.0, peak_max_allocated - baseline_allocated),
            "delta_peak_torch_max_reserved_gb": max(0.0, peak_max_reserved - baseline_reserved),
            "delta_peak_cuda_device_used_gb": max(0.0, peak_cuda_used - baseline_cuda_used),
        }

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._append_sample(stage="running")

    def _append_sample(self, stage: str) -> None:
        if self._started_at is None:
            elapsed = 0.0
        else:
            elapsed = time.perf_counter() - self._started_at
        sample: Dict[str, Any] = {
            "time_s": elapsed,
            "stage": stage,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                used_bytes = total_bytes - free_bytes
                sample.update({
                    "torch_allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                    "torch_reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
                    "torch_max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                    "torch_max_reserved_gb": torch.cuda.max_memory_reserved() / (1024 ** 3),
                    "cuda_device_free_gb": free_bytes / (1024 ** 3),
                    "cuda_device_total_gb": total_bytes / (1024 ** 3),
                    "cuda_device_used_gb": used_bytes / (1024 ** 3),
                })
            except Exception as exc:
                sample["sample_error"] = str(exc)
        self.samples.append(sample)


def get_memory_calibration_profiles() -> Dict[str, MemoryCalibrationProfile]:
    profiles = {
        "matrix_bigamp_small": MemoryCalibrationProfile(
            key="matrix_bigamp_small",
            algorithm_key="bigamp",
            matrix={"N1": 8, "N2": 8, "M": 2},
            alpha_values=[0.0, 0.1],
            samples_per_alpha=1,
            max_steps=2,
            purpose="校准 dense matrix BiGAMP 的小尺寸峰值显存记录链路。",
        ),
        "matrix_agd_target_10gb": MemoryCalibrationProfile(
            key="matrix_agd_target_10gb",
            algorithm_key="agd",
            matrix={"N1": 5936, "N2": 5936, "M": 32},
            alpha_values=[0.0, 0.1, 0.2, 0.3],
            samples_per_alpha=4,
            max_steps=2,
            max_epochs=2,
            tensor_order=2,
            use_compile=False,
            use_bf16=False,
            target_tensor_gb=10.0,
            purpose="10GB 级 AGD 校准；覆盖参数、梯度、prediction/residual 路径。",
            runtime_class="quick",
            calibration_kind="target_memory",
            sampler_interval_s=0.05,
            notes=(
                "固定在 N=5936 的 13GB 级实际 workspace bin；AGD 在 10GB "
                "target 附近存在 CUDA workspace 跳变，自动 solver 会贴到不稳定边界。"
            ),
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
    profiles.update(_build_target_memory_profiles())
    return profiles


def _build_target_memory_profiles() -> Dict[str, MemoryCalibrationProfile]:
    """Construct GB-scale profiles by solving N from the estimator itself."""
    specs = [
        {
            "key": "matrix_bigamp_target_10gb",
            "algorithm_key": "bigamp",
            "target_tensor_gb": 10.0,
            "samples_per_alpha": 4,
            "alpha_values": [0.0, 0.1, 0.2, 0.3],
            "M": 32,
            "tensor_order": 2,
            "purpose": "10GB 级 dense matrix BiGAMP 校准；用于估计 alpha/sample 并行真实显存系数。",
        },
        {
            "key": "matrix_bigamp_target_16gb",
            "algorithm_key": "bigamp",
            "target_tensor_gb": 16.0,
            "samples_per_alpha": 4,
            "alpha_values": [0.0, 0.1, 0.2, 0.3],
            "M": 32,
            "tensor_order": 2,
            "purpose": "16GB 级 dense matrix BiGAMP 校准候选；10GB profile 正常后再运行。",
        },
        {
            "key": "spreading_bigamp_target_10gb",
            "algorithm_key": "bigamp_spreading",
            "target_tensor_gb": 10.0,
            "samples_per_alpha": 4,
            "alpha_values": [0.1, 0.2, 0.3],
            "M": 32,
            "tensor_order": 2,
            "purpose": "10GB 级 matrix spreading BiGAMP 校准；覆盖 graph/F/gather/scatter 显存。",
        },
        {
            "key": "spreading_bigamp_target_16gb",
            "algorithm_key": "bigamp_spreading",
            "target_tensor_gb": 16.0,
            "samples_per_alpha": 4,
            "alpha_values": [0.1, 0.2, 0.3],
            "M": 32,
            "tensor_order": 2,
            "purpose": "16GB 级 matrix spreading BiGAMP 校准；验证 stage peak 公式的外推误差。",
        },
        {
            "key": "tensor_serial_target_6gb",
            "algorithm_key": "bigamp_tensor",
            "target_tensor_gb": 6.0,
            "samples_per_alpha": 4,
            "alpha_values": [0.05, 0.10],
            "M": 256,
            "tensor_order": 3,
            "purpose": "6GB raw serial tensor reference path 校准；覆盖 legacy serial alpha/sample loop。",
        },
        {
            "key": "tensor_serial_target_10gb",
            "algorithm_key": "bigamp_tensor",
            "target_tensor_gb": 10.0,
            "samples_per_alpha": 4,
            "alpha_values": [0.05, 0.10],
            "M": 256,
            "tensor_order": 3,
            "purpose": "10GB raw serial tensor reference path 校准；验证 serial tensor stage 公式外推。",
        },
        {
            "key": "tensor_parallel_target_6gb",
            "algorithm_key": "bigamp_tensor_parallel",
            "target_tensor_gb": 6.0,
            "samples_per_alpha": 4,
            "alpha_values": [0.05, 0.10],
            "M": 256,
            "tensor_order": 3,
            "purpose": "6GB raw / 10GB 级实际占用的 tensor parallel 爬坡校准；10GB raw profile 失败时先跑它。",
        },
        {
            "key": "tensor_parallel_target_10gb",
            "algorithm_key": "bigamp_tensor_parallel",
            "target_tensor_gb": 10.0,
            "samples_per_alpha": 4,
            "alpha_values": [0.05, 0.10],
            "M": 256,
            "tensor_order": 3,
            "purpose": "10GB 级 tensor parallel 校准；覆盖 TensorSuperGraph 与内部 alpha batching。",
        },
        {
            "key": "matrix_agd_target_16gb",
            "algorithm_key": "agd",
            "target_tensor_gb": 16.0,
            "samples_per_alpha": 4,
            "alpha_values": [0.0, 0.1, 0.2, 0.3],
            "M": 32,
            "tensor_order": 2,
            "purpose": "16GB 级 AGD 校准；验证 dense workspace peak 公式的外推误差。",
        },
    ]
    return {spec["key"]: _make_target_profile(**spec) for spec in specs}


def _make_target_profile(
    *,
    key: str,
    algorithm_key: str,
    target_tensor_gb: float,
    samples_per_alpha: int,
    alpha_values: List[float],
    M: int,
    tensor_order: int,
    purpose: str,
) -> MemoryCalibrationProfile:
    matrix_size = _solve_square_matrix_size_for_target(
        algorithm_key=algorithm_key,
        target_tensor_gb=target_tensor_gb,
        samples_per_alpha=samples_per_alpha,
        alpha_values=alpha_values,
        M=M,
        tensor_order=tensor_order,
    )
    return MemoryCalibrationProfile(
        key=key,
        algorithm_key=algorithm_key,
        matrix={"N1": matrix_size, "N2": matrix_size, "M": M},
        alpha_values=alpha_values,
        samples_per_alpha=samples_per_alpha,
        max_steps=2,
        max_epochs=2,
        tensor_order=tensor_order,
        purpose=purpose,
        runtime_class="quick",
        target_tensor_gb=target_tensor_gb,
        calibration_kind="target_memory",
        sampler_interval_s=0.05,
        notes=(
            "自动按 target_tensor_gb 反推 N；用于实测校准，不用于评价物理曲线。"
        ),
    )


def _solve_square_matrix_size_for_target(
    *,
    algorithm_key: str,
    target_tensor_gb: float,
    samples_per_alpha: int,
    alpha_values: List[float],
    M: int,
    tensor_order: int,
) -> int:
    """Pick N so raw estimator is close to a requested tensor footprint."""
    estimator = MemoryEstimator(apply_calibration=False)

    def raw_for_n(n: int) -> float:
        profile = MemoryCalibrationProfile(
            key="_target_solver",
            algorithm_key=algorithm_key,
            matrix={"N1": n, "N2": n, "M": M},
            alpha_values=alpha_values,
            samples_per_alpha=samples_per_alpha,
            max_steps=2,
            max_epochs=2,
            tensor_order=tensor_order,
            use_compile=False,
            use_bf16=False,
            use_tf32=True,
        )
        params = estimation_params_from_config(build_calibration_config(profile))
        return estimator.estimate_raw(params)

    low = 8
    high = 512
    while raw_for_n(high) < target_tensor_gb and high < 65536:
        high *= 2
    if high >= 65536 and raw_for_n(high) < target_tensor_gb:
        return high

    best_n = high
    best_error = abs(raw_for_n(high) - target_tensor_gb)
    while low <= high:
        mid = (low + high) // 2
        raw = raw_for_n(mid)
        error = abs(raw - target_tensor_gb)
        if error < best_error:
            best_n = mid
            best_error = error
        if raw < target_tensor_gb:
            low = mid + 1
        else:
            high = mid - 1
    # Some stage models intentionally contain discrete workspace bins. A pure
    # binary search can land on the wrong side of a discontinuity, so do a local
    # aligned scan around the binary candidate before finalizing the profile.
    local_low = max(8, int(best_n) - 2048)
    local_high = min(65536, int(best_n) + 2048)
    for n in range(int(round(local_low / 8) * 8), local_high + 1, 8):
        raw = raw_for_n(n)
        error = abs(raw - target_tensor_gb)
        if error < best_error:
            best_n = n
            best_error = error
    # Tensor cores and allocator bins behave better on aligned dimensions.
    return max(8, int(round(best_n / 8) * 8))


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
    estimator = MemoryEstimator(apply_calibration=False)
    estimate = estimator.estimate(params)
    raw_estimate_gb = estimator.estimate_raw(params)
    lines = ["memory calibration profile", ""]
    lines.append(f"key: {profile.key}")
    lines.append(f"algorithm: {profile.algorithm_key}")
    lines.append(f"runtime_class: {profile.runtime_class}")
    lines.append(f"calibration_kind: {profile.calibration_kind}")
    if profile.target_tensor_gb is not None:
        lines.append(f"target_tensor_gb: {profile.target_tensor_gb:.3f}")
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
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        lines.append(f"gpu_total_gb: {total_bytes / (1024 ** 3):.3f}")
        lines.append(f"gpu_free_now_gb: {free_bytes / (1024 ** 3):.3f}")
    lines.append(f"purpose: {profile.purpose}")
    lines.append(f"output_root: {profile.output_root}/{profile.key}")
    return "\n".join(lines)


def run_memory_calibration(profile_key: str, output_root: Optional[Path] = None) -> Dict[str, Any]:
    profile = _get_profile(profile_key)
    return _run_memory_calibration_profile(profile, output_root=output_root)


def tune_memory_calibration(
    algorithm_key: str,
    *,
    target_allocated_gb: float,
    max_device_gb: float = 24.0,
) -> Dict[str, Any]:
    """Build and run a local calibration candidate from a target raw footprint."""
    defaults = {
        "bigamp": {"samples_per_alpha": 4, "alpha_values": [0.0, 0.1, 0.2, 0.3], "M": 32, "tensor_order": 2},
        "agd": {"samples_per_alpha": 4, "alpha_values": [0.0, 0.1, 0.2, 0.3], "M": 32, "tensor_order": 2},
        "bigamp_spreading": {"samples_per_alpha": 4, "alpha_values": [0.1, 0.2, 0.3], "M": 32, "tensor_order": 2},
        "bigamp_tensor": {"samples_per_alpha": 4, "alpha_values": [0.05, 0.10], "M": 256, "tensor_order": 3},
        "bigamp_tensor_parallel": {"samples_per_alpha": 4, "alpha_values": [0.05, 0.10], "M": 256, "tensor_order": 3},
    }
    if algorithm_key not in defaults:
        raise KeyError(f"unknown calibration algorithm '{algorithm_key}'")
    spec = defaults[algorithm_key]
    profile = _make_target_profile(
        key=f"tune_{algorithm_key}_{float(target_allocated_gb):.1f}gb".replace(".", "p"),
        algorithm_key=algorithm_key,
        target_tensor_gb=float(target_allocated_gb),
        samples_per_alpha=spec["samples_per_alpha"],
        alpha_values=spec["alpha_values"],
        M=spec["M"],
        tensor_order=spec["tensor_order"],
        purpose=(
            f"Auto-tuned local memory calibration for {algorithm_key}; "
            f"target raw allocated {float(target_allocated_gb):.2f}GB."
        ),
    )
    config = build_calibration_config(profile)
    estimator = MemoryEstimator(apply_calibration=False)
    estimate = estimator.estimate(estimation_params_from_config(config))
    if estimate.device_peak_gb > float(max_device_gb):
        started_at = datetime.now()
        output_dir = _profile_output_dir(profile, None, started_at)
        output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "profile": profile.to_dict(),
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "status": "aborted_by_memory_guard",
            "reason": (
                f"estimated device peak {estimate.device_peak_gb:.3f}GB exceeds "
                f"max_device_gb {float(max_device_gb):.3f}GB"
            ),
            "theoretical_estimate_gb": estimator.estimate_raw(estimation_params_from_config(config)),
            "estimated_total_with_runtime_gb": estimate.total_gb,
            "output_dir": str(output_dir),
        }
        _write_json_with_retry(output_dir / "manifest.json", record)
        _write_json_with_retry(output_dir / "config.json", config.to_dict())
        return record
    return _run_memory_calibration_profile(profile, output_root=None)


def _run_memory_calibration_profile(
    profile: MemoryCalibrationProfile,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    config = build_calibration_config(profile)
    estimator = MemoryEstimator(apply_calibration=False)
    params = estimation_params_from_config(config)
    estimate = estimator.estimate(params)
    raw_estimate_gb = estimator.estimate_raw(params)
    fd_limit = _raise_open_file_limit()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        _ensure_profile_is_safe_to_run(estimate.total_gb)
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    runner = ExperimentRunner(device=device, verbose=False)
    started_at = datetime.now()
    output_dir = _profile_output_dir(profile, output_root, started_at)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = MemoryTimelineSampler(interval_s=profile.sampler_interval_s)
    run_status = "success"
    run_error = None
    run_exception: Optional[Exception] = None
    timeline_write_error = None
    sampler.start()
    try:
        runner.run(
            config,
            output_options={
                "save_tensors": False,
                "enable_heatmap": False,
                "checkpoint_path": str(output_dir / "checkpoints" / "latest.pt"),
            },
            raw_yaml="",
        )
    except Exception as exc:
        run_status = "failed"
        run_error = repr(exc)
        run_exception = exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sampler.stop()
        gc.collect()
        try:
            sampler.write_jsonl(output_dir / "memory_timeline.jsonl")
        except OSError as exc:
            timeline_write_error = repr(exc)

    cuda_available = torch.cuda.is_available()
    actual_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if cuda_available else 0.0
    reserved_peak_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if cuda_available else 0.0
    allocated_after_gb = torch.cuda.memory_allocated() / (1024 ** 3) if cuda_available else 0.0
    reserved_after_gb = torch.cuda.memory_reserved() / (1024 ** 3) if cuda_available else 0.0
    sampler_summary = sampler.summary()
    actual_delta_peak_gb = float(
        sampler_summary.get("delta_peak_torch_max_allocated_gb", actual_peak_gb)
    )
    cuda_delta_peak_gb = float(
        sampler_summary.get("delta_peak_cuda_device_used_gb", 0.0)
    )
    error_status = _memory_error_statuses(
        raw_estimate_gb=raw_estimate_gb,
        estimated_total_gb=estimate.total_gb,
        actual_tensor_allocated_gb=actual_delta_peak_gb,
        actual_device_used_gb=cuda_delta_peak_gb,
    )

    record = {
        "profile": profile.to_dict(),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "status": run_status,
        "error": run_error,
        "gpu": {
            "cuda_available": cuda_available,
            "name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
            "total_memory_gb": (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if cuda_available else 0.0
            ),
            "free_memory_gb_at_estimate": estimator.get_available_memory(),
        },
        "file_descriptor_limit": fd_limit,
        "estimation_params": params.to_dict(),
        "theoretical_estimate_gb": raw_estimate_gb,
        "estimated_total_with_runtime_gb": estimate.total_gb,
        "actual_peak_memory_gb": actual_peak_gb,
        "actual_delta_peak_tensor_allocated_gb": actual_delta_peak_gb,
        "actual_delta_peak_cuda_device_used_gb": cuda_delta_peak_gb,
        "reserved_peak_memory_gb": reserved_peak_gb,
        "allocated_after_gb": allocated_after_gb,
        "reserved_after_gb": reserved_after_gb,
        "memory_timeline": {
            "path": str(output_dir / "memory_timeline.jsonl"),
            "summary": sampler_summary,
            "write_error": timeline_write_error,
        },
        "error_ratio": error_status["formula_error_ratio"],
        "error_pct": error_status["formula_error_pct"],
        "formula_abs_error_pct": error_status["formula_abs_error_pct"],
        "formula_status": error_status["formula_status"],
        "device_error_ratio": error_status["device_error_ratio"],
        "device_error_pct": error_status["device_error_pct"],
        "device_abs_error_pct": error_status["device_abs_error_pct"],
        "device_status": error_status["device_status"],
        "calibration_status": error_status["calibration_status"],
        # Backward-compatible aliases. Older manifests used total_error_* for a
        # mixed raw-vs-device comparison; new records make it the actual device
        # estimate error.
        "total_error_ratio": error_status["device_error_ratio"],
        "total_error_pct": error_status["device_error_pct"],
        "output_dir": str(output_dir),
    }
    _write_json_with_retry(output_dir / "manifest.json", record)
    _write_json_with_retry(output_dir / "config.json", config.to_dict())
    _update_local_calibration_coefficients(record)
    if run_exception is not None:
        raise run_exception
    return record


def _memory_error_statuses(
    *,
    raw_estimate_gb: float,
    estimated_total_gb: float,
    actual_tensor_allocated_gb: float,
    actual_device_used_gb: float,
) -> Dict[str, Any]:
    """Compute the two calibration criteria used by memory profiles."""
    formula_error_ratio = (
        (raw_estimate_gb - actual_tensor_allocated_gb) / actual_tensor_allocated_gb
        if actual_tensor_allocated_gb > 0 else None
    )
    formula_abs_error_pct = (
        abs(formula_error_ratio) * 100 if formula_error_ratio is not None else None
    )
    formula_status = (
        "within_tolerance"
        if formula_abs_error_pct is not None and formula_abs_error_pct <= 10.0
        else "formula_mismatch"
    )

    device_error_ratio = (
        (estimated_total_gb - actual_device_used_gb) / actual_device_used_gb
        if actual_device_used_gb > 0 else None
    )
    device_abs_error_pct = (
        abs(device_error_ratio) * 100 if device_error_ratio is not None else None
    )
    device_status = (
        "within_tolerance"
        if device_abs_error_pct is not None and device_abs_error_pct <= 15.0
        else "device_mismatch"
    )

    return {
        "formula_error_ratio": formula_error_ratio,
        "formula_error_pct": (
            formula_error_ratio * 100 if formula_error_ratio is not None else None
        ),
        "formula_abs_error_pct": formula_abs_error_pct,
        "formula_status": formula_status,
        "device_error_ratio": device_error_ratio,
        "device_error_pct": (
            device_error_ratio * 100 if device_error_ratio is not None else None
        ),
        "device_abs_error_pct": device_abs_error_pct,
        "device_status": device_status,
        "calibration_status": (
            "within_tolerance"
            if formula_status == "within_tolerance" and device_status == "within_tolerance"
            else "calibration_mismatch"
        ),
    }


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
        chunk_size=getattr(spreading, "chunk_size", None) if spreading else None,
    )


def _ensure_profile_is_safe_to_run(estimated_total_gb: float, max_fraction_of_free: float = 0.82) -> None:
    """Refuse a calibration profile that is too close to current free VRAM."""
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    allowed_gb = free_gb * max_fraction_of_free
    if estimated_total_gb > allowed_gb:
        raise MemoryError(
            "Calibration profile is too large for current free VRAM: "
            f"estimated={estimated_total_gb:.2f}GB, allowed={allowed_gb:.2f}GB "
            f"({max_fraction_of_free:.0%} of free {free_gb:.2f}GB, total {total_gb:.2f}GB). "
            "Free GPU memory or choose a smaller calibration profile."
        )


def _raise_open_file_limit(min_soft_limit: int = 4096) -> Dict[str, Any]:
    """Raise RLIMIT_NOFILE for large local calibration runs when possible."""
    info: Dict[str, Any] = {
        "platform": os.name,
        "requested_soft_limit": min_soft_limit,
        "changed": False,
    }
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(int(soft), min_soft_limit), int(hard))
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            info["changed"] = True
        new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        info.update({
            "before_soft": int(soft),
            "before_hard": int(hard),
            "after_soft": int(new_soft),
            "after_hard": int(new_hard),
        })
    except Exception as exc:
        info["error"] = repr(exc)
    return info


def _write_json_with_retry(path: Path, payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    last_error: Optional[OSError] = None
    for _ in range(3):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except OSError as exc:
            last_error = exc
            gc.collect()
            time.sleep(0.2)
    if last_error is not None:
        raise last_error


def _update_local_calibration_coefficients(record: Dict[str, Any]) -> None:
    """Persist a conservative local calibration factor for future planning."""
    profile = record.get("profile", {})
    algorithm_key = str(profile.get("algorithm_key", ""))
    raw_gb = float(record.get("theoretical_estimate_gb", 0.0) or 0.0)
    actual_tensor_delta_gb = float(record.get("actual_delta_peak_tensor_allocated_gb", 0.0) or 0.0)
    output_dir = str(record.get("output_dir", ""))
    gpu_name = str(record.get("gpu", {}).get("name", "cpu"))
    gpu_model_key = gpu_name.replace(" ", "_").replace("/", "-")
    now = datetime.now().isoformat(timespec="seconds")

    path = _local_coefficients_path()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    payload.setdefault("format", "mf_memory_calibration_v1")
    payload["updated_at"] = now
    payload["gpu_model"] = gpu_model_key
    payload["source"] = "runs/calibration/memory/latest_coefficients.json"
    algorithms = payload.setdefault("algorithms", {})

    status_success = record.get("status") == "success"
    status_failed_lower_bound = (
        record.get("status") == "failed"
        and raw_gb >= 1.0
        and actual_tensor_delta_gb >= raw_gb * 1.05
    )
    valid_for_planner = (
        (status_success or status_failed_lower_bound)
        and raw_gb >= 1.0
        and actual_tensor_delta_gb >= 1.0
        and bool(algorithm_key)
        and record.get("formula_status") == "within_tolerance"
        and record.get("device_status") == "within_tolerance"
    )
    ratio = actual_tensor_delta_gb / raw_gb if raw_gb > 0 else None
    failure_margin = 1.25 if status_failed_lower_bound else 1.10
    conservative_factor = max(1.0, float(ratio or 1.0) * failure_margin) if valid_for_planner else 1.0

    existing = algorithms.get(algorithm_key, {})
    previous_factor = float(existing.get("factor", 1.0) or 1.0)
    factor = conservative_factor if valid_for_planner else previous_factor
    if record.get("formula_status") == "formula_mismatch":
        status = "formula_mismatch"
    elif record.get("device_status") == "device_mismatch":
        status = "device_mismatch"
    elif status_failed_lower_bound and valid_for_planner:
        status = "failed_lower_bound_active"
    elif valid_for_planner:
        status = "active"
    else:
        status = "ignored_insufficient_scale"

    algorithms[algorithm_key] = {
        "factor": factor,
        "latest_measured_factor": ratio,
        "latest_conservative_factor": conservative_factor,
        "min_raw_gb_for_apply": 1.0,
        "min_actual_gb_for_apply": 1.0,
        "latest_raw_gb": raw_gb,
        "latest_actual_delta_tensor_gb": actual_tensor_delta_gb,
        "latest_cuda_device_delta_gb": record.get("actual_delta_peak_cuda_device_used_gb"),
        "latest_profile_key": profile.get("key"),
        "latest_output_dir": output_dir,
        "latest_status": status,
        "updated_at": now,
    }

    history = payload.setdefault("history", [])
    history.append({
        "algorithm_key": algorithm_key,
        "profile_key": profile.get("key"),
        "raw_gb": raw_gb,
        "actual_delta_tensor_gb": actual_tensor_delta_gb,
        "measured_factor": ratio,
        "conservative_factor": conservative_factor,
        "planner_factor_after_update": factor,
        "status": status,
        "output_dir": output_dir,
        "updated_at": now,
    })
    payload["history"] = history[-100:]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _local_coefficients_path() -> Path:
    return Path("runs") / "calibration" / "memory" / "latest_coefficients.json"


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
