from matrix_factorization.core.trials import build_trial_plan, list_trial_plans, load_trial_registry


def test_trial_registry_and_manifest_are_consistent():
    registry = load_trial_registry()
    plans = {plan.spec.key: plan for plan in list_trial_plans()}

    assert "matrix_bigamp_quick" in registry
    assert "matrix_bigamp_quick" in plans
    assert plans["matrix_bigamp_quick"].spec.status == registry["matrix_bigamp_quick"]["status"]


def test_matrix_bigamp_quick_trial_validates_against_experiment_plan():
    plan = build_trial_plan("matrix_bigamp_quick")

    assert plan.is_valid
    assert plan.spec.runtime_class == "quick"
    assert plan.experiment_plan is not None
    assert plan.experiment_plan.effective_parameters["algorithm_key"] == "bigamp"
    assert 3 <= plan.experiment_plan.effective_parameters["scan.num_points"] <= 5
    assert "Q_Y_mean" in plan.to_dict()["available_metric_keys"]
    assert "MSE" not in plan.to_dict()["available_metric_keys"]
    assert "Q_W_COS_ROOT_mean" in plan.to_dict()["available_metric_keys"]


def test_trial_expected_metrics_must_be_declared(tmp_path):
    trial_dir = tmp_path / "trials" / "active" / "bad_metric"
    trial_dir.mkdir(parents=True)
    (tmp_path / "trials" / "registry.yaml").write_text(
        """
trials:
  - key: bad_metric
    status: active
    path: trials/active/bad_metric/trial.yaml
""",
        encoding="utf-8",
    )
    (trial_dir / "trial.yaml").write_text(
        """
key: bad_metric
status: active
runtime_class: quick
config: trials/active/bad_metric/config.yaml
output_root: artifacts/trials/bad_metric
artifact_policy: ignored_workspace
expected_metrics:
  - not_a_metric
""",
        encoding="utf-8",
    )
    (trial_dir / "config.yaml").write_text(
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

    plan = build_trial_plan("bad_metric", root=tmp_path)

    assert not plan.is_valid
    assert any("expected_metrics" in error for error in plan.errors)


def test_active_trial_config_must_live_inside_trial_directory(tmp_path):
    trial_dir = tmp_path / "trials" / "active" / "bad_config_path"
    trial_dir.mkdir(parents=True)
    (tmp_path / "trials" / "registry.yaml").write_text(
        """
trials:
  - key: bad_config_path
    status: active
    path: trials/active/bad_config_path/trial.yaml
""",
        encoding="utf-8",
    )
    (trial_dir / "trial.yaml").write_text(
        """
key: bad_config_path
status: active
runtime_class: quick
config: configs/bad_trial.yaml
output_root: artifacts/trials/bad_config_path
artifact_policy: ignored_workspace
""",
        encoding="utf-8",
    )

    plan = build_trial_plan("bad_config_path", root=tmp_path)

    assert any("active trial 的 config 必须位于" in error for error in plan.errors)
