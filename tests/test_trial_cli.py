from pathlib import Path
import json
import subprocess
import sys


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "matrix_factorization.cli", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_trial_list_command_shows_builtin_trial():
    result = _run_cli("trial", "list")

    assert result.returncode == 0
    assert "registered trials" in result.stdout
    assert "matrix_bigamp_quick" in result.stdout


def test_trial_explain_and_validate_commands_show_contract():
    explain = _run_cli("trial", "explain", "matrix_bigamp_quick")
    validate = _run_cli("trial", "validate", "matrix_bigamp_quick")

    assert explain.returncode == 0
    assert "trial 配置解释" in explain.stdout
    assert "实际 algorithm: bigamp" in explain.stdout
    assert "trials/active/matrix_bigamp_quick/config.yaml" in explain.stdout
    assert validate.returncode == 0
    assert "trial 校验结果" in validate.stdout
    assert "errors: none" in validate.stdout


def test_trial_run_command_generates_run_directory():
    result = _run_cli("trial", "run", "matrix_bigamp_quick")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "trial run complete" in result.stdout
    output_line = next(line for line in result.stdout.splitlines() if line.strip().startswith("output:"))
    output_path = Path(output_line.split("output:", 1)[1].strip())

    assert output_path.exists()
    assert "artifacts/trials/matrix_bigamp_quick" in str(output_path)
    for filename in ["config.json", "metadata.json", "metrics.json", "manifest.json"]:
        assert (output_path / filename).exists()
    metadata = json.loads((output_path / "metadata.json").read_text(encoding="utf-8"))
    runtime_plan = metadata["contract"]["runtime_resource_plan"]
    assert runtime_plan["algorithm_key"] == "bigamp"
    assert runtime_plan["metadata_only"] is True
    assert runtime_plan["num_batches"] >= 1
    assert runtime_plan["batches"][0]["sample_range_honored_by_runner"] is False


def test_trial_run_rejects_medium_or_gpu_heavy(tmp_path):
    manifest = tmp_path / "medium_trial.yaml"
    manifest.write_text(
        """
key: medium_trial
status: active
runtime_class: medium
config: trials/active/medium_trial/config.yaml
output_root: artifacts/trials/medium_trial
artifact_policy: ignored_workspace
""",
        encoding="utf-8",
    )

    result = _run_cli("trial", "run", str(manifest))

    assert result.returncode == 1
    assert "runtime_class=medium" in result.stdout
    assert "only runs quick trials" in result.stdout
