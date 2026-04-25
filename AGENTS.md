# Agent Guidance

Work from the `dev` branch. The active Python package is
`src/matrix_factorization`, not the older `MF/` or `Wang/` layout mentioned in
historical notes.

Use these commands for lightweight local checks:

```bash
pip install -e ".[dev]"
python -m pytest -q tests/test_contract_parameter_specs.py tests/test_config_contract.py tests/test_parallel_memory_contract.py
python -m pytest -q tests/test_trial_contract.py tests/test_trial_cli.py
```

Before committing structural hard-interface work, run the default lightweight
suite:

```bash
python -m pytest -q
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

Parameter rule: every new YAML-facing knob must have a `ParameterSpec`, type or
enum validation, an effective parameter trace, and metadata/result visibility if
it changes execution. A parsed field that is not consumed by the active route
must show up as an inactive-route warning or strict-mode error.

Research trial rule: if the user asks for a small trial run, quick debug,
parameter poke, or "see if it runs", use `trials/active/<trial_key>/config.yaml`
and the `mf trial ...` commands. Do not edit
`src/matrix_factorization/config.yaml` for trial/debug runs unless the user
explicitly asks to change the formal default config. Trial outputs must stay in
`runs/trials/`, `artifacts/trials/`, or `results/trials/`.
