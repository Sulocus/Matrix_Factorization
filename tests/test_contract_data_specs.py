import matrix_factorization.modules.graphs  # noqa: F401
import matrix_factorization.modules.teachers  # noqa: F401

from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.contracts import get_graph_specs, get_parameter_specs, get_teacher_specs
from matrix_factorization.core.planning import build_experiment_plan
from matrix_factorization.modules.registry import (
    get_graph,
    get_graph_spec,
    get_teacher,
    get_teacher_spec,
    register_graph,
    register_teacher,
    validate_graph_registry_contracts,
    validate_teacher_registry_contracts,
)


def test_teacher_specs_cover_active_yaml_teachers():
    specs = get_teacher_specs()

    assert specs["standard"].status == "active"
    assert specs["orthogonal"].status == "active"
    assert "teacher_config.init_distribution" in specs["standard"].required_config_paths
    assert "Y_teacher" in specs["standard"].produced_artifacts
    assert specs["random_spreading"].status == "legacy_reference"


def test_graph_specs_classify_registered_and_active_path_graphs():
    specs = get_graph_specs()

    assert specs["random"].status == "legacy_reference"
    assert specs["uniform"].status == "legacy_reference"
    assert specs["supergraph"].status == "active_path"
    assert "spreading_graph" in specs["supergraph"].produced_artifacts


def test_registered_teachers_and_graphs_have_specs():
    assert validate_teacher_registry_contracts() == []
    assert validate_graph_registry_contracts() == []
    assert get_teacher("standard").teacher_spec is get_teacher_spec("standard")
    assert get_graph("random").graph_spec is get_graph_spec("random")


def test_register_teacher_and_graph_require_existing_specs():
    try:
        register_teacher(key="missing_teacher_contract", name="Missing Teacher")
    except ValueError as exc:
        assert "without a TeacherSpec" in str(exc)
    else:
        raise AssertionError("register_teacher accepted a missing TeacherSpec")

    try:
        register_graph(key="missing_graph_contract", name="Missing Graph")
    except ValueError as exc:
        assert "without a GraphSpec" in str(exc)
    else:
        raise AssertionError("register_graph accepted a missing GraphSpec")


def test_data_specs_reference_known_parameters():
    parameter_specs = get_parameter_specs()
    for spec in get_teacher_specs().values():
        for path in spec.required_config_paths:
            assert path in parameter_specs
    for spec in get_graph_specs().values():
        for path in spec.required_config_paths:
            assert path in parameter_specs


def test_experiment_plan_includes_teacher_spec(tmp_path):
    config_path = tmp_path / "teacher_plan.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
teacher: 2
teacher_config:
  init_distribution: 2
matrix:
  N1: 4
  N2: 4
  M: 2
training:
  samples_per_alpha: 1
  max_steps: 2
scan:
  axes:
    alpha:
      path: alpha
      values:
        start: 0.0
        stop: 0.0
        step: 1.0
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    payload = plan.to_dict()

    assert not plan.errors
    assert payload["teacher_spec"]["key"] == "standard"
    assert "normalization_profile" in payload["teacher_spec"]["scale_convention"]
