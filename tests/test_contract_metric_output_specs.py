from pathlib import Path

import pytest

import matrix_factorization.modules.outputs  # noqa: F401
from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.contracts import (
    get_algorithm_specs,
    get_algorithm_metric_keys,
    get_algorithm_metric_semantics,
    get_metric_schema,
    get_metric_semantic_classes,
    get_metric_specs,
    get_output_specs,
)
from matrix_factorization.core.experiment.data_factory import ExperimentData
from matrix_factorization.core.experiment.runner import ExperimentRunner
from matrix_factorization.core.planning import build_experiment_plan
from matrix_factorization.modules.metrics.contract_compute import compute_matrix_metric_payload
from matrix_factorization.modules.metrics.spec_adapter import MetricSpecAdapter
from matrix_factorization.modules.outputs.spec_adapter import OutputPayloadCheck
from matrix_factorization.modules.registry import (
    get_output,
    get_output_spec,
    register_metric,
    register_output,
    validate_metric_registry_contracts,
    validate_output_registry_contracts,
)


def test_metric_specs_have_semantic_metadata():
    specs = get_metric_specs()

    for key in ["matrix.full.Q_Y", "spreading.observed.Q_Y", "tensor.full.Q_Y"]:
        spec = specs[key]
        assert spec.produces
        assert spec.space
        assert spec.scope
        assert spec.relation
        assert spec.normalization
        assert spec.physical_meaning
        assert spec.canonical_key
        assert spec.equivalence_class
        assert spec.result_role
        assert spec.order_parameter_status


def test_metric_semantic_classes_cover_active_metric_specs():
    classes = get_metric_semantic_classes()
    specs = get_metric_specs()

    for spec in specs.values():
        assert spec.canonical_key in classes
        assert spec.key in get_algorithm_specs()[spec.compatible_algorithms[0]].produced_metrics
        assert set(spec.produces).intersection(classes[spec.canonical_key].legacy_aliases)


def test_metric_semantic_docs_cover_all_canonical_classes():
    classes = get_metric_semantic_classes()
    semantics_doc = Path("docs/metrics_semantics.md").read_text(encoding="utf-8")
    naming_doc = Path("docs/metric_naming_decisions.md").read_text(encoding="utf-8")

    for canonical_key in classes:
        assert canonical_key in semantics_doc
        assert canonical_key in naming_doc


def test_metric_specs_and_algorithm_specs_are_bidirectionally_consistent():
    algorithm_specs = get_algorithm_specs()
    metric_specs = get_metric_specs()

    for algorithm_key, algorithm_spec in algorithm_specs.items():
        for metric_key in algorithm_spec.produced_metrics:
            assert algorithm_key in metric_specs[metric_key].compatible_algorithms

    for metric_key, metric_spec in metric_specs.items():
        for algorithm_key in metric_spec.compatible_algorithms:
            assert algorithm_key in algorithm_specs
            assert metric_key in algorithm_specs[algorithm_key].produced_metrics


def test_output_specs_declare_requirements():
    specs = get_output_specs()

    assert "overlap_matrix" in specs["tensor_heatmap"].requires
    assert "metrics_by_alpha" in specs["scalar_curves"].requires
    assert "config.json" in specs["latest_display"].requires


def test_algorithm_compatible_outputs_have_declared_artifact_supply():
    algorithm_specs = get_algorithm_specs()
    output_specs = get_output_specs()
    generic_requirements = {
        "metrics_by_alpha",
        "config.json",
        "metadata.json",
    }

    for algorithm_key, algorithm_spec in algorithm_specs.items():
        supply = set(algorithm_spec.produced_artifacts)
        if "matrix_factors" in algorithm_spec.capabilities:
            supply.add("matrix_factors")
        if "tensor_factors" in algorithm_spec.capabilities:
            supply.add("tensor_factors")
        for output_key in algorithm_spec.compatible_outputs:
            output_spec = output_specs[output_key]
            missing = [
                requirement
                for requirement in output_spec.requires
                if requirement not in supply and requirement not in generic_requirements
            ]
            assert missing == [], f"{algorithm_key} declares {output_key} but cannot supply {missing}"


def test_registered_outputs_have_output_specs():
    assert validate_output_registry_contracts() == []
    assert get_output("plotting").output_spec is get_output_spec("plotting")
    assert get_output_spec("storage").key == "storage"


def test_register_output_requires_existing_spec():
    with pytest.raises(ValueError, match="without an OutputSpec"):
        register_output(key="missing_output_contract", name="Missing Output")


def test_register_metric_requires_existing_spec():
    assert validate_metric_registry_contracts() == []
    with pytest.raises(ValueError, match="without a MetricSpec"):
        register_metric(key="missing_metric_contract", name="Missing Metric")


def test_output_payload_check_is_hard_contract_report():
    check = OutputPayloadCheck(
        requested_specs=["custom_curves"],
        required_metrics=["Q_W_mean"],
        available_metrics=["Q_Y_mean"],
        missing_metrics=["Q_W_mean"],
    )

    assert not check.is_valid
    assert check.to_dict()["missing_metrics"] == ["Q_W_mean"]


def test_algorithm_metric_keys_include_legacy_flat_outputs():
    keys = get_algorithm_metric_keys("bigamp_spreading")

    assert "Q_Y_mean" in keys
    assert "Q_Y_observed_mean" in keys
    assert "Q_Y_unobserved_mean" in keys
    assert "Q_W_COS_ROOT_mean" in keys
    assert "Q_W_prime_replica_mean" in keys
    assert "MSE" not in keys


def test_flat_metric_semantics_keep_qy_algorithm_context():
    dense_semantics = get_algorithm_metric_semantics("bigamp")["Q_Y_mean"][0]
    tensor_semantics = get_algorithm_metric_semantics("bigamp_tensor_parallel")["Q_Y_mean"][0]

    assert dense_semantics["space"] == "measurement"
    assert tensor_semantics["space"] == "measurement"
    assert dense_semantics["metric_spec"] != tensor_semantics["metric_spec"]
    assert dense_semantics["canonical_key"] == "measurement.full.teacher_student.Q_Y_projection"
    assert tensor_semantics["canonical_key"] == "measurement.full.teacher_student.Q_Y_projection"


def test_metric_schema_preserves_flat_keys_but_indexes_semantic_classes():
    dense_schema = get_metric_schema(
        "bigamp",
        metric_keys=["Q_Y_mean", "Q_W_COS_ROOT_mean", "Q_W_SIGN_ALIGNED_mean"],
    )
    tensor_schema = get_metric_schema("bigamp_tensor_parallel", metric_keys=["Q_Y_mean"])

    assert dense_schema["schema_version"] == 3
    assert dense_schema["compatibility"]["legacy_flat_keys_preserved"] is True
    assert dense_schema["compatibility"]["projection_metric_migration"] is True
    assert dense_schema["flat_key_index"]["Q_Y_mean"][0]["canonical_key"] == "measurement.full.teacher_student.Q_Y_projection"
    assert dense_schema["flat_key_index"]["Q_W_COS_ROOT_mean"][0]["canonical_key"] == "latent.W.teacher_student.Q_W_COS_ROOT"
    assert dense_schema["flat_key_index"]["Q_W_SIGN_ALIGNED_mean"][0]["canonical_key"] == "latent.W.teacher_student.Q_W_SIGN_ALIGNED"
    assert tensor_schema["flat_key_index"]["Q_Y_mean"][0]["canonical_key"] == "measurement.full.teacher_student.Q_Y_projection"
    assert (
        dense_schema["flat_key_index"]["Q_Y_mean"][0]["metric_spec"]
        != tensor_schema["flat_key_index"]["Q_Y_mean"][0]["metric_spec"]
    )


def test_active_algorithm_metric_keys_are_all_indexed_by_metric_schema():
    for algorithm_key, algorithm_spec in get_algorithm_specs().items():
        if algorithm_spec.status != "active":
            continue
        keys = set(get_algorithm_metric_keys(algorithm_key))
        schema = get_metric_schema(algorithm_key, metric_keys=sorted(keys))

        assert keys <= set(schema["flat_key_index"]), algorithm_key


def test_metric_spec_adapter_validates_legacy_flat_payloads():
    check = MetricSpecAdapter.check_payload(
        "bigamp",
        {"Q_Y_mean": 0.1, "Q_W_mean": 0.2, "Q_W_COS_ROOT_mean": 0.3},
        source="runner_matrix_metrics",
    )

    assert check.is_valid
    assert "Q_Y_mean" in check.declared_keys
    assert check.source == "runner_matrix_metrics"
    assert "matrix.full.Q_Y" in check.metric_specs
    assert "matrix.factor.Q_W" in check.metric_specs
    assert check.semantic_keys["Q_Y_mean"][0]["space"] == "measurement"


def test_metric_spec_adapter_rejects_undeclared_flat_payloads():
    check = MetricSpecAdapter.check_payload(
        "bigamp",
        {"Q_Y_mean": 0.1, "unregistered_metric": 0.2},
    )

    assert not check.is_valid
    assert check.unexpected_keys == ["unregistered_metric"]


def test_matrix_metric_compute_adapter_produces_declared_flat_keys():
    import torch

    data = ExperimentData(
        W_teacher=torch.eye(2, 1),
        X_teacher=torch.ones(1, 2),
        Y_teacher=torch.eye(2, 1) @ torch.ones(1, 2),
        masks=torch.ones(1, 2, 2),
        alpha_values=[0.5],
    )
    metrics = compute_matrix_metric_payload(
        W_students=torch.eye(2, 1).unsqueeze(0),
        X_students=torch.ones(1, 2).unsqueeze(0),
        data=data,
    )
    check = MetricSpecAdapter.validate_payload(
        "bigamp",
        metrics,
        source="contract_compute_matrix_metrics",
    )

    assert metrics["Q_Y_mean"] == pytest.approx(1.0)
    assert "MSE" not in metrics
    assert metrics["Q_W_mean"] == pytest.approx(1.0)
    assert metrics["Q_W_SIGN_ALIGNED_mean"] == pytest.approx(1.0)
    assert metrics["Q_X_SIGN_ALIGNED_mean"] == pytest.approx(1.0)
    assert metrics["Q_W_COS_ROOT_mean"] == pytest.approx(1.0)
    assert "matrix.full.Q_Y" in check.metric_specs
    assert "matrix.factor.Q_W" in check.metric_specs


def test_runner_delegates_fallback_metric_computation_to_contract_compute(monkeypatch):
    import torch
    from matrix_factorization.modules.metrics import contract_compute

    data = ExperimentData(
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        masks=torch.ones(1, 2, 2),
        alpha_values=[0.5],
    )
    captured = {}

    def fake_compute_metric_payload(*, W_students, X_students, data):
        captured["shape"] = tuple(W_students.shape)
        captured["alpha"] = data.alpha_values[0]
        return {"Q_Y_mean": 0.4}

    monkeypatch.setattr(contract_compute, "compute_metric_payload", fake_compute_metric_payload)
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)
    metrics = runner._compute_metrics(
        W_students=torch.zeros(1, 2, 1),
        X_students=torch.zeros(1, 1, 2),
        data=data,
    )

    assert metrics == {"Q_Y_mean": 0.4}
    assert captured == {"shape": (1, 2, 1), "alpha": 0.5}


def test_runner_metric_adapter_import_failure_is_hard_error(monkeypatch):
    import torch
    from matrix_factorization.modules.metrics import contract_compute

    data = ExperimentData(
        W_teacher=torch.zeros(2, 1),
        X_teacher=torch.zeros(1, 2),
        Y_teacher=torch.zeros(2, 2),
        masks=torch.ones(1, 2, 2),
        alpha_values=[0.5],
    )

    def fail_compute_metric_payload(*, W_students, X_students, data):
        raise ImportError("missing metric backend")

    monkeypatch.setattr(contract_compute, "compute_metric_payload", fail_compute_metric_payload)
    runner = ExperimentRunner(device=torch.device("cpu"), verbose=False)

    with pytest.raises(RuntimeError, match="zero-valued placeholder metrics"):
        runner._compute_metrics(
            W_students=torch.zeros(1, 2, 1),
            X_students=torch.zeros(1, 1, 2),
            data=data,
        )


def test_custom_plot_metric_without_contract_is_validation_error(tmp_path):
    config_path = tmp_path / "bad_plot.yaml"
    config_path.write_text(
        """
tensor_order: 3
algorithm: 4
teacher: 2
matrix:
  N1: 4
  N2: 4
  M: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
training:
  samples_per_alpha: 1
  max_steps: 2
algorithm_params:
  damping: 0.5
  noise_var: 1.0e-5
  use_compile: false
  use_bf16: false
spreading:
  f_distribution: 1
output:
  enable_heatmap: false
  plots:
    - curves: [A.w]
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("A.w -> Q_W_mean" in error for error in plan.errors)


def test_custom_plot_invalid_curve_code_is_validation_error(tmp_path):
    config_path = tmp_path / "bad_curve.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
training:
  samples_per_alpha: 1
  max_steps: 2
output:
  enable_heatmap: false
  plots:
    - curves: [Z.z]
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("曲线代码无效" in error for error in plan.errors)


def test_output_plan_records_custom_plot_metric_requirements(tmp_path):
    config_path = tmp_path / "plot_plan.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
training:
  samples_per_alpha: 1
  max_steps: 2
output:
  enable_heatmap: false
  plots:
    - curves: [A.y, A.w, D.w]
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert plan.errors == []
    assert plan.output_plan.required_metrics == ["Q_W_SIGN_ALIGNED_mean", "Q_W_mean", "Q_Y_mean"]
    assert "custom_curves" in plan.output_plan.specs
    assert plan.output_plan.plot_semantics["A.y"]["metric_key"] == "Q_Y_mean"
    assert plan.output_plan.plot_semantics["D.w"]["metric_key"] == "Q_W_SIGN_ALIGNED_mean"
    assert (
        plan.output_plan.plot_semantics["A.y"]["semantic_candidates"][0]["canonical_key"]
        == "measurement.full.teacher_student.Q_Y_projection"
    )
    assert plan.output_plan.metric_semantics["Q_W_mean"][0]["canonical_key"] == "latent.W.teacher_student.Q_W_projection"
    assert (
        plan.output_plan.metric_semantics["Q_W_SIGN_ALIGNED_mean"][0]["canonical_key"]
        == "latent.W.teacher_student.Q_W_SIGN_ALIGNED"
    )


def test_flat_key_metric_semantics_split_prime_projection_and_replica_variants():
    semantics = get_algorithm_metric_semantics("bigamp")

    assert semantics["Q_W_mean"][0]["canonical_key"] == "latent.W.teacher_student.Q_W_projection"
    assert semantics["Q_W_COS_ROOT_mean"][0]["canonical_key"] == "latent.W.teacher_student.Q_W_COS_ROOT"
    assert semantics["Q_W_SIGN_ALIGNED_mean"][0]["canonical_key"] == "latent.W.teacher_student.Q_W_SIGN_ALIGNED"
    assert semantics["Q_X_mean"][0]["canonical_key"] == "latent.X.teacher_student.Q_X_projection"
    assert semantics["Q_X_COS_ROOT_mean"][0]["canonical_key"] == "latent.X.teacher_student.Q_X_COS_ROOT"
    assert semantics["Q_X_SIGN_ALIGNED_mean"][0]["canonical_key"] == "latent.X.teacher_student.Q_X_SIGN_ALIGNED"
    assert "physical_overlap_W_mean" not in semantics
    assert "MSE" not in semantics
    assert semantics["Q_W_replica_mean"][0]["canonical_key"] == "factor.W.replica.student_student.gram_cosine"
    assert semantics["Q_W_prime_replica_mean"][0]["canonical_key"] == "factor.W.replica.student_student.baseline_corrected_gram_cosine"


def test_tensor_heatmap_output_plan_records_diagnostic_artifact_semantics(tmp_path):
    config_path = tmp_path / "tensor_heatmap_plan.yaml"
    config_path.write_text(
        """
tensor_order: 3
algorithm: 4
teacher: 2
matrix:
  N1: 4
  N2: 4
  M: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
training:
  samples_per_alpha: 1
  max_steps: 2
algorithm_params:
  damping: 0.5
  noise_var: 1.0e-5
  use_compile: false
  use_bf16: false
spreading:
  f_distribution: 1
output:
  enable_heatmap: true
  heatmap_metric: Q_W
""",
        encoding="utf-8",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert plan.errors == []
    assert plan.output_plan.artifact_semantics["overlap_matrix"]["canonical_key"] == "replica.heatmap.teacher_and_students.matrix"
    assert plan.output_plan.artifact_semantics["overlap_matrix"]["result_role"] == "diagnostic"
    assert plan.output_plan.artifact_semantics["overlap_matrix"]["heatmap_metric"] == "Q_W"
