from matrix_factorization.cli import load_yaml_config
from matrix_factorization.core.contracts import AlgorithmStateView, get_analyzer_specs, get_probe_specs
from matrix_factorization.core.experiment.result import ExperimentResult, SingleRunResult
from matrix_factorization.core.experiment.runner import ExperimentRunner
from matrix_factorization.core.planning import build_experiment_plan
from matrix_factorization.modules.interventions import (
    HookPoint,
    RuntimeExtensionExecutor,
    RuntimeHookContext,
)


def _write_config(path, extra: str):
    path.write_text(
        f"""
tensor_order: 3
algorithm: 4
teacher: 2
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
algorithm_params:
  damping: 0.5
  noise_var: 1.0e-5
  use_compile: false
spreading:
  f_distribution: 1
output:
  enable_heatmap: true
{extra}
""",
        encoding="utf-8",
    )


def test_probe_and_analyzer_specs_exist():
    assert "tensor_state_slice" in get_probe_specs()
    assert "batch_summary" in get_probe_specs()
    assert "tensor_heatmap_summary" in get_analyzer_specs()


def test_compatible_tensor_probe_passes_preflight(tmp_path):
    config_path = tmp_path / "with_probe.yaml"
    _write_config(
        config_path,
        """
probes:
  - key: tensor_state_slice
analyzers:
  - key: tensor_heatmap_summary
""",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert not plan.errors
    assert [spec.key for spec in plan.probe_specs] == ["tensor_state_slice"]
    assert [spec.key for spec in plan.analyzer_specs] == ["tensor_heatmap_summary"]


def test_incompatible_probe_fails_preflight(tmp_path):
    config_path = tmp_path / "bad_probe.yaml"
    _write_config(
        config_path,
        """
probes:
  - key: variance_slice
""",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("probe 'variance_slice' 未声明兼容 algorithm" in error for error in plan.errors)


def test_incompatible_intervention_fails_preflight(tmp_path):
    config_path = tmp_path / "bad_intervention.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 1
teacher: 2
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
algorithm_params:
  init_mode: teacher
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)

    assert any("intervention 'warm_start' 未声明兼容 algorithm 'bigamp'" in error for error in plan.errors)


def test_warm_start_runtime_state_exposes_teacher_factors(tmp_path):
    config_path = tmp_path / "warm_start_spreading.yaml"
    config_path.write_text(
        """
tensor_order: 2
algorithm: 2
teacher: 2
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
algorithm_params:
  init_mode: teacher
spreading:
  f_distribution: 1
output:
  enable_heatmap: false
""",
        encoding="utf-8",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    executor = RuntimeExtensionExecutor.from_contract(plan.to_dict(), config.algorithm_key)
    runner = ExperimentRunner(device=None, verbose=False)
    state = runner._initial_runtime_state(
        config=config,
        W_teacher="W_teacher",
        X_teacher="X_teacher",
        Y_teacher="Y_teacher",
    )

    assert not plan.errors
    returned = executor.dispatch(
        HookPoint.BEFORE_INITIALIZE,
        state,
        RuntimeHookContext(
            algorithm_key=config.algorithm_key,
            hook=HookPoint.BEFORE_INITIALIZE,
        ),
    )

    assert returned is state
    assert "teacher_factors" in state.available_capabilities()
    assert executor.report.dispatched_hooks[0]["interventions"] == ["warm_start"]


def test_runtime_extension_executor_dispatches_declared_probe_and_analyzer(tmp_path):
    config_path = tmp_path / "with_runtime_extensions.yaml"
    _write_config(
        config_path,
        """
probes:
  - key: tensor_state_slice
analyzers:
  - key: tensor_heatmap_summary
""",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    executor = RuntimeExtensionExecutor.from_contract(plan.to_dict(), config.algorithm_key)
    state = AlgorithmStateView(tensor_factors={"mode0": "factor"}, step_index=1)

    returned_state = executor.dispatch(
        HookPoint.AFTER_STEP,
        state,
        RuntimeHookContext(
            algorithm_key=config.algorithm_key,
            hook=HookPoint.AFTER_STEP,
            step_index=1,
        ),
    )

    assert returned_state is state
    assert executor.report.dispatched_hooks[0]["probes"] == ["tensor_state_slice"]

    result = ExperimentResult(
        experiment_id="runtime_extension_contract",
        config=config,
        scan_dimension=config.scan.dimension,
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(
            scan_value=0.0,
            metrics={"Q_Y_mean": 0.5, "overlap_matrix": [[1.0]]},
        ),
    )

    reports = executor.run_analyzers(result)

    assert reports["tensor_heatmap_summary"]["overlap_matrix_count"] == 1
    assert executor.report.to_dict()["analyzer_reports"]["tensor_heatmap_summary"]["status"] == "ok"


def test_analyzer_requires_actual_result_inputs(tmp_path):
    config_path = tmp_path / "with_runtime_extensions.yaml"
    _write_config(
        config_path,
        """
analyzers:
  - key: tensor_heatmap_summary
""",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    executor = RuntimeExtensionExecutor.from_contract(plan.to_dict(), config.algorithm_key)
    result = ExperimentResult(
        experiment_id="missing_analyzer_input",
        config=config,
        scan_dimension=config.scan.dimension,
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(
            scan_value=0.0,
            metrics={"Q_Y_mean": 0.5},
        ),
    )

    try:
        executor.run_analyzers(result)
    except RuntimeError as exc:
        assert "analyzer 'tensor_heatmap_summary' requires ['overlap_matrix']" in str(exc)
    else:
        raise AssertionError("analyzer accepted a result without required inputs")


def test_runtime_extension_dispatch_requires_declared_state(tmp_path):
    config_path = tmp_path / "with_probe.yaml"
    _write_config(
        config_path,
        """
probes:
  - key: tensor_state_slice
""",
    )

    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    executor = RuntimeExtensionExecutor.from_contract(plan.to_dict(), config.algorithm_key)

    try:
        executor.dispatch(
            HookPoint.AFTER_STEP,
            AlgorithmStateView(step_index=1),
            RuntimeHookContext(
                algorithm_key=config.algorithm_key,
                hook=HookPoint.AFTER_STEP,
                step_index=1,
            ),
        )
    except RuntimeError as exc:
        assert "requires state ['tensor_factors']" in str(exc)
    else:
        raise AssertionError("runtime extension dispatch accepted missing state")


def test_runner_after_batch_probe_records_batch_summary(tmp_path):
    config_path = tmp_path / "with_batch_probe.yaml"
    _write_config(
        config_path,
        """
probes:
  - key: batch_summary
""",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    executor = RuntimeExtensionExecutor.from_contract(plan.to_dict(), config.algorithm_key)
    runner = ExperimentRunner(device=None, verbose=False)
    runner._runtime_extensions = executor
    result = ExperimentResult(
        experiment_id="after_batch_contract",
        config=config,
        scan_dimension=config.scan.dimension,
        scan_values=[0.0],
    )
    result.add_result(
        0.0,
        SingleRunResult(
            scan_value=0.0,
            metrics={"Q_Y_mean": 0.5, "overlap_matrix": [[1.0]]},
        ),
    )

    runner._dispatch_after_batch_runtime_extensions(
        config=config,
        batch_idx=0,
        alpha_values=[0.0],
        algorithm_result=None,
        result=result,
    )

    report = executor.report.to_dict()
    assert report["probes"] == ["batch_summary"]
    assert report["dispatched_hooks"][0]["hook"] == "after_batch"
    assert report["dispatched_hooks"][0]["probes"] == ["batch_summary"]
    assert "batch_idx" in report["dispatched_hooks"][0]["state_capabilities"]


def test_runner_records_runtime_extension_report_without_running_hooks_inside_algorithm(tmp_path, monkeypatch):
    config_path = tmp_path / "with_runtime_report.yaml"
    _write_config(
        config_path,
        """
analyzers:
  - key: tensor_heatmap_summary
""",
    )
    config, output_options, raw_yaml = load_yaml_config(config_path)
    plan = build_experiment_plan(config, output_options, raw_yaml, config_path)
    runner = ExperimentRunner(device=None, verbose=False)

    monkeypatch.setattr(runner, "_get_algorithm", lambda _config: object())

    def fake_standard_scan(config, algorithm, result, observer, resume_results=None, output_options=None, raw_yaml=""):
        result.add_result(
            0.0,
            SingleRunResult(
                scan_value=0.0,
                metrics={"Q_Y_mean": 0.5, "overlap_matrix": [[1.0]]},
            ),
        )

    monkeypatch.setattr(runner, "_run_standard_scan", fake_standard_scan)
    result = runner.run(
        config,
        output_options={"experiment_plan": plan.to_dict()},
        raw_yaml=raw_yaml,
    )

    report = result.metadata.contract["runtime_extension_report"]
    assert report["analyzers"] == ["tensor_heatmap_summary"]
    assert report["analyzer_reports"]["tensor_heatmap_summary"]["overlap_matrix_count"] == 1
