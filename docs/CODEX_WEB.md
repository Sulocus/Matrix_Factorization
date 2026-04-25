# Codex Web Workflow

Use the `dev` branch as the Codex web baseline. The web workspace should contain
source code, tests, docs, and small smoke configs only.

## Setup

```bash
pip install -e ".[dev]"
```

## Fast Checks

```bash
python -m pytest -q tests/test_tensor_metrics.py tests/test_plotting_compat.py tests/smoke/test_registry_imports.py
```

## Run Artifacts

Runtime output belongs under ignored local directories:

- `runs/`
- `results/`
- `artifacts/`
- `src/matrix_factorization/Replica_results/`

The canonical run layout is:

```text
runs/{run_id}/
  manifest.json
  config.json
  metrics.json
  events.jsonl
  artifacts/
    results.pt
    raw_data.npz
  plots/
  checkpoints/
    latest.pt
```

Keep large `.pt` tensors and historical experiments local, or store them in an
external artifact store with a manifest. Codex web should run smoke tests and
code review; local GPU runs remain the source of high-memory validation.
