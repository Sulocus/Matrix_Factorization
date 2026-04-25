# Matrix Factorization

Teacher-student matrix and tensor factorization experiments with PyTorch
implementations of AGD, BiG-AMP, matrix spreading, and tensor spreading variants.

## Setup

```bash
pip install -e ".[dev]"
```

## Fast Validation

```bash
python -m pytest -q tests/test_tensor_metrics.py tests/test_plotting_compat.py tests/smoke/test_registry_imports.py
```

## Usage

```bash
# Small Codex/web-safe smoke config
mf configs/smoke/matrix_spreading_smoke.yaml

# Local GPU research preset
mf configs/local_gpu/tensor_local_gpu.yaml --output-dir runs
```

`mf` writes new runs to ignored `runs/{run_id}/` directories by default.

## Project Structure

```text
Matrix_Factorization/
  src/matrix_factorization/   # Installable package and CLI
  configs/                    # Tracked smoke and local-GPU configs
  tests/                      # Unit, smoke, verification, and research tests
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

Codex web should be used for code review, lightweight tests, and versioned
changes. High-memory scientific validation should run on the local GPU.

## More

See `docs/CODEX_WEB.md` for the Codex web workflow and run artifact schema.
