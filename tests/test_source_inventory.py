from pathlib import Path

from matrix_factorization.core.contracts import (
    get_auxiliary_source_inventory,
    get_config_source_inventory,
    get_graph_source_inventory,
    get_legacy_pytest_inventory,
    get_metric_source_inventory,
    get_output_source_inventory,
    get_repository_surface_inventory,
    get_runtime_extension_source_inventory,
    get_teacher_source_inventory,
)


def _discover_python_sources(*roots):
    discovered = set()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            discovered.add(str(path))
    return discovered


def test_auxiliary_debug_and_experiment_sources_are_classified():
    discovered = _discover_python_sources(
        "experiments",
        "scripts",
        "src/matrix_factorization/export",
        "tests/debug",
        "tests/verification",
    )
    inventory = set(get_auxiliary_source_inventory())

    assert discovered - inventory == set()


def test_auxiliary_source_inventory_uses_explicit_non_active_statuses():
    inventory = get_auxiliary_source_inventory()

    assert inventory["experiments/teacher_matrix_analysis/main.py"].status == "local_experiment"
    assert inventory["scripts/debug/debug_tensor_tau.py"].status == "debug_only"
    assert inventory["scripts/experiments/memory_benchmark.py"].status == "local_experiment"
    assert inventory["scripts/verification/verify_scaling.py"].status == "local_gpu_check"
    assert inventory["src/matrix_factorization/export/bundler.py"].status == "support_module"


def test_repository_surface_files_are_classified():
    controlled_suffixes = {".py", ".md", ".yaml", ".yml", ".toml", ".png", ".pt"}
    discovered = {
        str(path)
        for path in Path(".").glob("*")
        if path.is_file()
        and path.suffix in controlled_suffixes
        and not path.name.startswith(".")
    }
    inventory = set(get_repository_surface_inventory())

    assert discovered - inventory == set()


def test_config_entrypoint_files_are_classified():
    discovered = {
        str(path)
        for root in [Path("configs"), Path("src/matrix_factorization")]
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".yaml", ".yml", ".md"}
        and "__pycache__" not in path.parts
        and (str(path).startswith("configs/") or path.name == "config.yaml")
    }
    inventory = set(get_config_source_inventory())

    assert discovered - inventory == set()


def test_runtime_extension_sources_are_classified():
    discovered = _discover_python_sources("src/matrix_factorization/modules/interventions")
    inventory = set(get_runtime_extension_source_inventory())

    assert discovered - inventory == set()


def test_output_sources_are_classified():
    discovered = _discover_python_sources("src/matrix_factorization/modules/outputs")
    inventory = set(get_output_source_inventory())

    assert discovered - inventory == set()


def test_output_source_inventory_marks_active_contract_adapter():
    inventory = get_output_source_inventory()

    assert inventory["src/matrix_factorization/modules/outputs/spec_adapter.py"].status == "active_path"
    assert inventory["src/matrix_factorization/modules/outputs/latest.py"].role == "latest_display_exporter"


def test_metric_sources_are_classified():
    discovered = _discover_python_sources("src/matrix_factorization/modules/metrics")
    inventory = set(get_metric_source_inventory())

    assert discovered - inventory == set()


def test_metric_source_inventory_marks_contract_compute_active():
    inventory = get_metric_source_inventory()

    assert inventory["src/matrix_factorization/modules/metrics/contract_compute.py"].status == "active_path"
    assert inventory["src/matrix_factorization/modules/metrics/spec_adapter.py"].role == "metric_contract_validator"


def test_teacher_sources_are_classified():
    discovered = _discover_python_sources("src/matrix_factorization/modules/teachers")
    inventory = set(get_teacher_source_inventory())

    assert discovered - inventory == set()


def test_graph_sources_are_classified():
    discovered = _discover_python_sources("src/matrix_factorization/modules/graphs")
    inventory = set(get_graph_source_inventory())

    assert discovered - inventory == set()


def test_legacy_pytest_modules_are_classified_and_module_skipped():
    inventory = get_legacy_pytest_inventory()

    for path, spec in inventory.items():
        source = Path(path)
        assert source.exists(), path
        assert spec.status == "legacy_skipped"
        text = source.read_text(encoding="utf-8")
        assert "pytest.skip(" in text
        assert "allow_module_level=True" in text
