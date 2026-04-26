from matrix_factorization.core.trials import build_trial_plan


def test_canonical_scan_quick_trials_validate_with_resource_preview():
    for key in [
        "scan_alpha_quick",
        "scan_steps_quick",
        "scan_size_quick",
        "scan_init_quick",
        "scan_mixed_axes_quick",
    ]:
        trial_plan = build_trial_plan(key)

        assert trial_plan.is_valid, (key, trial_plan.errors)
        preview = trial_plan.experiment_plan.resource_plan.get("resource_execution_plan_preview")
        assert preview
        assert preview["num_work_items"] >= 1
        assert preview["preflight_errors"] == []
