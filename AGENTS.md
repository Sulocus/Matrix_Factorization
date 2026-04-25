# Agent Guidance

Work from the `dev` branch. The active Python package is
`src/matrix_factorization`, not the older `MF/` or `Wang/` layout mentioned in
historical notes.

Use these commands for Codex web checks:

```bash
pip install -e ".[dev]"
python -m pytest -q tests/test_tensor_metrics.py tests/test_plotting_compat.py tests/smoke/test_registry_imports.py
```

Do not commit generated experiment data from `runs/`, `results/`,
`artifacts/`, or `src/matrix_factorization/Replica_results/`. Large GPU
experiments should run locally and record artifacts through an external manifest.

When changing algorithms, keep physical conventions explicit: latent scaling,
alpha normalization, damping semantics, Onsager handling, and the exact meaning
of each `Q_Y` variant must be documented with the code change.
