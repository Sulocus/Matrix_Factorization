from pathlib import Path
from dataclasses import replace

import matrix_factorization.modules.algorithms  # noqa: F401
from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.contracts import (
    get_algorithm_source_inventory,
    get_algorithm_specs,
    get_batching_specs,
    get_memory_model_specs,
    get_metric_specs,
    get_output_specs,
    get_parameter_specs,
    get_resource_specs,
    get_seed_policy_specs,
)
from matrix_factorization.core.planning import build_experiment_plan
import matrix_factorization.core.planning as planning_module
from matrix_factorization.modules.registry import (
    get_algorithm,
    get_algorithm_spec,
    list_algorithm_specs,
    register_algorithm,
    validate_algorithm_registry_contracts,
)


def test_active_algorithm_specs_cover_primary_algorithms():
    specs = get_algorithm_specs()

    for key in [
        "agd",
        "bigamp",
        "bigamp_spreading",
        "bigamp_tensor",
        "bigamp_tensor_parallel",
    ]:
        spec = specs[key]
        assert spec.status == "active"
        assert spec.result_contract
        assert spec.data_requirements
        assert spec.compatible_outputs
        assert spec.state_capabilities


def test_algorithm_integration_doc_covers_all_algorithm_specs():
    text = Path("docs/algorithm_integration_contract.md").read_text(encoding="utf-8")

    for key in get_algorithm_specs():
        assert key in text


def test_non_main_algorithms_are_not_active():
    specs = get_algorithm_specs()

    assert specs["agd_tensor"].status == "experimental_unintegrated"
    assert specs["combined"].status == "non_trainable_helper"


def test_registry_exposes_algorithm_specs():
    spec = get_algorithm_spec("bigamp_tensor_parallel")

    assert spec.key == "bigamp_tensor_parallel"
    assert spec.status == "active"
    assert "overlap_matrix" in spec.produced_artifacts
    assert any(item.key == "bigamp" for item in list_algorithm_specs())
    assert get_algorithm("bigamp_tensor_parallel").algorithm_spec is spec


def test_registered_algorithms_have_contracts():
    assert validate_algorithm_registry_contracts() == []


def test_register_algorithm_requires_existing_spec():
    try:
        register_algorithm(key="missing_contract", name="Missing Contract")
    except ValueError as exc:
        assert "without an AlgorithmSpec" in str(exc)
    else:
        raise AssertionError("register_algorithm accepted a missing AlgorithmSpec")


def test_duplicate_algorithm_registry_key_is_rejected():
    try:
        decorator = register_algorithm(key="bigamp", name="Duplicate BiGAMP")
        decorator(type("DuplicateBiGAMP", (), {}))
    except KeyError as exc:
        assert "Duplicate registry key" in str(exc)
    else:
        raise AssertionError("duplicate registry key was accepted")


def test_algorithm_source_files_are_classified():
    root = Path("src/matrix_factorization/modules/algorithms")
    discovered = {
        str(path)
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    inventory = set(get_algorithm_source_inventory())

    assert discovered - inventory == set()


def test_algorithm_entry_sources_have_specs():
    inventory = get_algorithm_source_inventory()
    specs = get_algorithm_specs()

    for item in inventory.values():
        if item.role == "algorithm_entry":
            assert item.registry_key in specs


def test_algorithm_specs_reference_existing_contracts():
    algorithm_specs = get_algorithm_specs()
    parameter_specs = get_parameter_specs()
    metric_specs = get_metric_specs()
    output_specs = get_output_specs()

    for algorithm_key, spec in algorithm_specs.items():
        for path in spec.required_config_paths:
            assert path in parameter_specs, f"{algorithm_key} requires unknown parameter {path}"
        for metric_key in spec.produced_metrics:
            assert metric_key in metric_specs, f"{algorithm_key} produces unknown metric {metric_key}"
        for output_key in spec.compatible_outputs:
            assert output_key in output_specs, f"{algorithm_key} references unknown output {output_key}"


def test_active_algorithms_have_resource_and_batching_specs():
    resource_specs = get_resource_specs()
    batching_specs = get_batching_specs()
    memory_model_specs = get_memory_model_specs()
    seed_policy_specs = get_seed_policy_specs()

    for algorithm_key, spec in get_algorithm_specs().items():
        assert algorithm_key in resource_specs
        assert algorithm_key in batching_specs
        assert algorithm_key in memory_model_specs
        assert algorithm_key in seed_policy_specs
        if spec.status == "active":
            assert resource_specs[algorithm_key].estimator_key
            assert batching_specs[algorithm_key].planner_layers
            assert memory_model_specs[algorithm_key].calibration_status
            assert seed_policy_specs[algorithm_key].policy_key


def test_algorithm_required_config_paths_are_preflight_checked(tmp_path, monkeypatch):
    config_path = tmp_path / "required_param.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan_mode: 1
alpha_scan:
  start: 0.0
  stop: 0.0
  step: 1.0
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )
    specs = dict(get_algorithm_specs())
    specs["bigamp"] = replace(
        specs["bigamp"],
        required_config_paths=list(specs["bigamp"].required_config_paths) + ["spreading.seed"],
    )
    monkeypatch.setattr(planning_module, "get_algorithm_specs", lambda: specs)

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("AlgorithmSpec[bigamp] requires spreading.seed" in error for error in plan.errors)
    assert "REQUIRED_PARAMETER_INACTIVE" in plan.to_dict()["error_codes"]
