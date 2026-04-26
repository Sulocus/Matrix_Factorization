# Matrix Factorization

Teacher-student matrix and tensor factorization experiments with PyTorch
implementations of AGD, BiG-AMP, matrix spreading, and tensor spreading variants.

## Setup

```bash
pip install -e ".[dev]"
```

## Local Validation

```bash
python -m pytest -q tests/test_tensor_metrics.py tests/test_plotting_compat.py tests/test_registry_imports.py
```

## Config Preflight

```bash
mf validate src/matrix_factorization/config.yaml
mf explain-config src/matrix_factorization/config.yaml
```

These commands do not run an experiment. They explain the effective algorithm
route, parameter consumption status, output contracts, and known warnings.

## Usage

```bash
# Local GPU research preset
mf configs/local_gpu/tensor_local_gpu.yaml --output-dir runs
```

`mf` writes new runs to ignored `runs/{run_id}/` directories by default.

## Project Structure

```text
Matrix_Factorization/
  src/matrix_factorization/   # Installable package and CLI
  configs/                    # Tracked local research configs
  tests/                      # Unit, compatibility, verification, and research tests
  scripts/                    # Experiments, analysis, debug, verification
  docs/                       # Theory, reports, figures, reference code
  runs/                       # Ignored runtime output
  artifacts/                  # Ignored optional large data
```

## Artifact Policy

Large generated tensors, checkpoints, heatmap batches, and historical experiment
directories are not source code. Keep them in ignored `runs/`, `results/`,
`artifacts/`, or `src/matrix_factorization/Replica_results/`, and record
externally stored data in `docs/artifacts_manifest.md` when needed.

Use this repository for code review, contract tests, quick trials, and versioned
changes. High-memory scientific validation should run as local GPU validation.

## More

See `docs/local_workflow.md` for the local workflow and run artifact schema.
