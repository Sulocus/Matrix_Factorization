from dataclasses import replace

import pytest
import torch

from matrix_factorization.core.contracts import AlgorithmResult, get_batching_specs
from matrix_factorization.core.experiment.config import (
    AlgorithmParams,
    ExperimentConfig,
    MatrixParams,
    SampleShardingConfig,
    ScanConfig,
    TrainingParams,
)
from matrix_factorization.core.experiment.runner import ExperimentRunner


class _SampleAwareFakeAlgorithm:
    def __init__(self, config, calls):
        self.config = config
        self.calls = calls

    def train_batch_result(self, *, W_teacher, X_teacher, alpha_values, sample_context=None, **_kwargs):
        self.calls.append(dict(sample_context or {}))
        sample_start = int((sample_context or {}).get("sample_start", 0))
        sample_count = int(self.config.training.samples_per_alpha)
        scales = torch.arange(
            sample_start + 1,
            sample_start + sample_count + 1,
            dtype=W_teacher.dtype,
            device=W_teacher.device,
        )
        W_samples = torch.stack([scale * W_teacher for scale in scales], dim=0)
        X_samples = X_teacher.unsqueeze(0).expand(sample_count, -1, -1).clone()
        W_students = torch.stack([W_samples for _ in alpha_values], dim=0)
        X_students = torch.stack([X_samples for _ in alpha_values], dim=0)
        return AlgorithmResult(
            matrix_factors={"W_students": W_students, "X_students": X_students},
            metadata={"result_source": "sample_aware_fake"},
        )


def _config(*, sharded: bool) -> ExperimentConfig:
    return ExperimentConfig(
        matrix=MatrixParams(N1=3, N2=3, M=1),
        training=TrainingParams(samples_per_alpha=4, max_steps=1),
        algorithm_key="bigamp",
        scan=ScanConfig(dimension="alpha", values=[0.5]),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=False,
            seed_partition_policy="partition_invariant",
        ),
        sample_sharding=SampleShardingConfig(
            enabled=True if sharded else False,
            max_samples_per_shard=2,
        ),
        teacher_key="standard",
    )


def _metrics_from_result(runner, config, result):
    data = runner.data_factory.create(config, alpha_values=[0.5])
    W_students, X_students = runner._matrix_factors_from_result(result)
    return runner._compute_metrics(
        W_students=W_students,
        X_students=X_students,
        data=data,
        algorithm=None,
        algorithm_result=result,
    )


def test_sample_sharding_merges_per_sample_metrics_without_changing_full_s_semantics(monkeypatch):
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    calls = []
    monkeypatch.setattr(
        runner,
        "_get_algorithm",
        lambda config: _SampleAwareFakeAlgorithm(config, calls),
    )

    full_config = _config(sharded=False)
    full_result = runner._run_algorithm_result_sample_sharded(
        config=full_config,
        alpha_values=[0.5],
        step_callback=None,
    )
    full_metrics = _metrics_from_result(runner, full_config, full_result)

    calls.clear()
    sharded_config = _config(sharded=True)
    sharded_result = runner._run_algorithm_result_sample_sharded(
        config=sharded_config,
        alpha_values=[0.5],
        step_callback=None,
    )
    sharded_metrics = sharded_result.metrics_by_alpha[0.5]

    assert [call["sample_start"] for call in calls] == [0, 2]
    assert [call["sample_end"] for call in calls] == [2, 4]
    assert sharded_result.metadata["sample_sharding"]["shard_count"] == 2
    for key in [
        "Q_Y_mean",
        "Q_Y_std",
        "FIT_Y_mean",
        "FIT_Y_std",
        "Q_W_mean",
        "Q_W_std",
        "Q_W_PROJ_ABS_mean",
        "Q_W_PROJ_ABS_std",
    ]:
        assert sharded_metrics[key] == pytest.approx(full_metrics[key])


def test_sample_sharding_requires_partition_invariant_seed_policy(monkeypatch):
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    monkeypatch.setattr(
        runner,
        "_get_algorithm",
        lambda config: _SampleAwareFakeAlgorithm(config, []),
    )
    config = replace(
        _config(sharded=True),
        algorithm_params=AlgorithmParams(
            use_compile=False,
            use_bf16=False,
            seed_partition_policy="legacy",
        ),
    )

    with pytest.raises(RuntimeError, match="seed_partition_policy='partition_invariant'"):
        runner._run_algorithm_result_sample_sharded(
            config=config,
            alpha_values=[0.5],
            step_callback=None,
        )


def test_formal_algorithm_contracts_advertise_sample_range_honoring():
    specs = get_batching_specs()

    for algorithm_key in [
        "agd",
        "bigamp",
        "bigamp_spreading",
        "bigamp_tensor",
        "bigamp_tensor_parallel",
        "agd_tensor",
    ]:
        assert specs[algorithm_key].sample_range_honored is True
