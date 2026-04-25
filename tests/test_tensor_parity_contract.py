from pathlib import Path

from matrix_factorization.core.contracts import (
    get_algorithm_specs,
    get_tensor_parity_report,
    get_tensor_parity_specs,
)


def test_tensor_parity_contract_doc_exists():
    path = Path("docs/tensor_parity_contract.md")

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "teacher scale" in text
    assert "alpha normalization" in text
    assert "Q_Y / Q_Y_observed semantics" in text
    for key in get_tensor_parity_specs():
        assert key in text


def test_tensor_serial_and_parallel_have_specs():
    specs = get_algorithm_specs()

    serial = specs["bigamp_tensor"]
    parallel = specs["bigamp_tensor_parallel"]
    assert serial.status == "active"
    assert parallel.status == "active"
    assert "tensor_metrics" in serial.capabilities
    assert "tensor_metrics" in parallel.capabilities


def test_tensor_parity_specs_are_machine_readable():
    specs = get_tensor_parity_specs()

    for key in [
        "teacher_scale",
        "alpha_normalization",
        "graph_definition",
        "f_distribution",
        "damping_semantics",
        "onsager_handling",
        "qy_semantics",
        "result_schema",
        "initialization_semantics",
        "seed_partition",
        "dtype_compile_semantics",
        "batching_semantics",
    ]:
        assert key in specs
        assert specs[key].required
        assert specs[key].risk in {"medium", "high"}
        assert specs[key].area
        assert specs[key].review_required


def test_tensor_parity_report_records_current_contract_gaps():
    report = get_tensor_parity_report()

    assert report["same_result_contract"] is True
    assert report["result_contracts"]["serial"] == "legacy_tensor_metrics_only"
    assert report["shared_metrics"] == []
    assert "tensor.serial_observed.Q_Y" in report["parallel_missing_serial_metrics"]
    assert "tensor.full.Q_Y" in report["serial_missing_parallel_metrics"]
    assert "tensor.observed.Q_Y" in report["serial_missing_parallel_metrics"]
    assert "tensor_supergraph" in report["parallel_only_data_requirements"]
    assert report["parity_item_details"]["seed_partition"]["risk"] == "high"
    assert report["parity_item_details"]["batching_semantics"]["area"] == "resource"
