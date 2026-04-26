import pytest

from matrix_factorization.core.parallel.execution_modes import EstimationParams
from matrix_factorization.core.parallel.memory_estimator import MemoryEstimator


@pytest.mark.parametrize(
    "algorithm_key",
    ["bigamp", "agd", "bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel"],
)
def test_active_algorithm_estimates_expose_stage_peak_metadata(algorithm_key):
    tensor_order = 3 if "tensor" in algorithm_key else 2
    params = EstimationParams(
        N1=16,
        N2=16,
        M=4,
        S=2,
        alpha_values=[0.1, 0.2],
        algorithm_key=algorithm_key,
        use_compile=False,
        use_bf16=False,
        tensor_order=tensor_order,
        tensor_dims=(16, 16, 16) if tensor_order == 3 else None,
    )
    estimate = MemoryEstimator(apply_calibration=False).estimate(params)

    assert estimate.raw_peak_allocated_gb > 0
    assert estimate.device_peak_gb >= estimate.raw_peak_allocated_gb
    assert estimate.dominant_stage
    assert estimate.stage_breakdown
    assert estimate.persistent_tensors
    assert estimate.transient_peak_tensors
    assert "estimator.raw_peak_allocated_gb" in estimate.breakdown


@pytest.mark.parametrize(
    ("name", "params", "recorded_peak_gb"),
    [
        (
            "agd_dense_peak_profile",
            EstimationParams(
                N1=8176,
                N2=8176,
                M=32,
                S=4,
                alpha_values=[0.0, 0.1, 0.2, 0.3],
                algorithm_key="agd",
                use_compile=False,
            ),
            24.728,
        ),
        (
            "tensor_parallel_stage_peak_profile",
            EstimationParams(
                N1=5120,
                N2=5120,
                M=256,
                S=4,
                alpha_values=[0.05, 0.10],
                algorithm_key="bigamp_tensor_parallel",
                use_compile=False,
                tensor_order=3,
                tensor_dims=(5120, 5120, 5120),
            ),
            15.480,
        ),
    ],
)
def test_underestimated_historical_profiles_are_stage_model_regressions(name, params, recorded_peak_gb):
    raw = MemoryEstimator(apply_calibration=False).estimate_raw(params)

    assert raw == pytest.approx(recorded_peak_gb, rel=0.10), name
