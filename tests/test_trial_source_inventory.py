from pathlib import Path
import subprocess

from matrix_factorization.core.contracts import get_trial_source_inventory
from matrix_factorization.core.trials import build_trial_plan, list_trial_plans, trial_covered_paths


CONTROLLED_SUFFIXES = {".py", ".md", ".yaml", ".yml"}


def test_trial_source_inventory_covers_tracked_trial_files():
    discovered = {
        str(path)
        for path in Path("trials").rglob("*")
        if path.is_file() and path.suffix in CONTROLLED_SUFFIXES
    }
    inventory = set(get_trial_source_inventory())

    assert discovered - inventory == set()


def test_active_trial_files_are_covered_by_manifest_or_registry():
    covered = {Path("trials/README.md").resolve(), Path("trials/registry.yaml").resolve()}
    for plan in list_trial_plans():
        covered.update(trial_covered_paths(plan))

    discovered = {
        path.resolve()
        for path in Path("trials/active").rglob("*")
        if path.is_file() and path.suffix in CONTROLLED_SUFFIXES
    }

    assert discovered - covered == set()


def test_trial_output_root_is_ignored_workspace():
    plan = build_trial_plan("matrix_bigamp_quick")
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(plan.output_root)],
        check=False,
    )

    assert plan.is_valid
    assert str(plan.output_root).endswith("runs/trials/matrix_bigamp_quick")
    assert result.returncode == 0


def test_trial_manifest_does_not_point_to_repo_tracked_large_artifacts():
    plan = build_trial_plan("matrix_bigamp_quick")
    tracked = subprocess.run(
        ["git", "ls-files", str(plan.output_root)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert tracked.stdout.strip() == ""
