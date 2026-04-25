# Agent Guidance

Work from the `dev` branch. The active Python package is
`src/matrix_factorization`, not the older `MF/` or `Wang/` layout mentioned in
historical notes.

Use these commands for lightweight local checks:

```bash
pip install -e ".[dev]"
python -m pytest -q tests/test_tensor_metrics.py tests/test_plotting_compat.py tests/test_registry_imports.py
```

Do not commit generated experiment data from `runs/`, `results/`,
`artifacts/`, or `src/matrix_factorization/Replica_results/`. Large GPU
experiments should run locally and record artifacts through an external manifest.

When changing algorithms, keep physical conventions explicit: latent scaling,
alpha normalization, damping semantics, Onsager handling, and the exact meaning
of each `Q_Y` variant must be documented with the code change.

Hard-interface rule: any new algorithm, teacher, graph, metric, output, probe,
analyzer, or intervention must be declared in
`src/matrix_factorization/core/contracts.py` before it is wired into the
runtime. Registries are spec-gated: `@register_algorithm(...)`,
`@register_teacher(...)`, `@register_graph(...)`, `@register_metric(...)`, and
`@register_output(...)` without a matching spec must fail during import. New
source files in classified areas must also be added to the source inventory.
After adding extension points, run `mf validate <config>` plus the contract
tests covering registry/spec/planning behavior.

Research trial rule: if the user asks for a small trial run, quick debug,
parameter poke, or "see if it runs", use `trials/active/<trial_key>/config.yaml`
and the `mf trial ...` commands. Do not edit
`src/matrix_factorization/config.yaml` for trial/debug runs unless the user
explicitly asks to change the formal default config. Trial outputs must stay in
`runs/trials/`, `artifacts/trials/`, or `results/trials/`.
