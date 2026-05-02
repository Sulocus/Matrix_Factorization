import subprocess

import pytest

from matrix_factorization.core.gpu_monitor import (
    GPUComputeSampler,
    parse_nvidia_smi_compute_sample,
)


def test_parse_nvidia_smi_compute_sample():
    parsed = parse_nvidia_smi_compute_sample("91, 248.5, 12288")

    assert parsed == {
        "gpu_utilization": 91.0,
        "power_draw_w": 248.5,
        "memory_used_gb": 12.0,
    }


def test_gpu_compute_sampler_reads_nvidia_smi(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="89, 250.0, 10240\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampler = GPUComputeSampler(enabled=True)

    sample = sampler._sample_once()

    assert sample["gpu_utilization"] == 89.0
    assert sample["power_draw_w"] == 250.0
    assert sample["memory_used_gb"] == pytest.approx(10.0)


def test_gpu_compute_sampler_samples_when_measurement_window_opens(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args[0], 0, stdout="88, 240.0, 8192\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampler = GPUComputeSampler(enabled=True, warmup_steps=2, measure_steps=100)

    sampler.mark_step(1)
    sampler.mark_step(2)
    assert sampler.report()["num_samples"] == 0
    sampler.mark_step(3)

    report = sampler.report(steps_run=3, configured_max_steps=10)
    assert report["num_samples"] == 1
    assert report["mean_gpu_utilization"] == pytest.approx(0.88)
    assert report["small_kernel_limited"] is True
    assert report["measurement_quality"] == "small_kernel_limited"


def test_gpu_compute_sampler_falls_back_to_torch_memory(monkeypatch):
    def missing_smi(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing_smi)
    monkeypatch.setattr(
        GPUComputeSampler,
        "_torch_current_memory_gb",
        staticmethod(lambda: 1.25),
    )
    monkeypatch.setattr(
        GPUComputeSampler,
        "_torch_peak_allocated_gb",
        staticmethod(lambda: 1.5),
    )
    sampler = GPUComputeSampler(enabled=True)

    sample = sampler._sample_once()
    sampler._samples.append(sample)
    report = sampler.report(steps_run=10, configured_max_steps=20)

    assert sample == {"memory_used_gb": 1.25}
    assert report["partial"] is True
    assert report["partial_reason"] == "nvidia_smi_unavailable"
    assert report["min_samples"] == 5
    assert report["min_measure_seconds"] == 3.0
    assert report["measurement_quality"] == "partial"
    assert report["max_memory_used_gb"] == 1.25
    assert report["peak_allocated_gb"] == 1.5
