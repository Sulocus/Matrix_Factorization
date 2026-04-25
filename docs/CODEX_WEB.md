# Development Workflow

Use the `dev` branch as the development baseline. The repository should contain
source code, tests, docs, and shareable research configs only.

## Setup

```bash
pip install -e ".[dev]"
```

## Fast Checks

```bash
python -m pytest -q tests/test_contract_parameter_specs.py tests/test_config_contract.py tests/test_parallel_memory_contract.py
python -m pytest -q tests/test_trial_contract.py tests/test_trial_cli.py
```

For hard-interface or workflow changes, run the relevant contract tests or the
default suite:

```bash
python -m pytest -q
```

## Config Validation

Before running an experiment from YAML, use the preflight commands:

```bash
mf validate path/to/config.yaml
mf explain-config path/to/config.yaml
mf validate --json path/to/config.yaml
```

New algorithms, teachers, graphs, metrics, outputs, probes, analyzers, and
interventions must be declared in `src/matrix_factorization/core/contracts.py`
before they are wired into registries or runtime paths.

Every YAML-facing parameter must have a `ParameterSpec`, basic type/enum
validation, an effective parameter trace, and metadata/result visibility when it
changes execution. `mf validate --strict` should fail if an active-route
parameter is only half-wired.

## Research Trials

Small parameter/debug runs should use the trial workflow, not the formal
default config:

```bash
mf trial list
mf trial explain matrix_bigamp_quick
mf trial validate matrix_bigamp_quick
mf trial run matrix_bigamp_quick
```

Trial parameters live in `trials/active/<trial_key>/config.yaml`. Do not edit
`src/matrix_factorization/config.yaml` for quick trial/debug runs unless the
user explicitly asks to change the formal default.

## Run Artifacts

Runtime output belongs under ignored local directories:

- `runs/`
- `results/`
- `artifacts/`
- `runs/trials/`
- `artifacts/trials/`
- `results/trials/`
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
external artifact store with a manifest. Local GPU runs remain the source of
high-memory validation.
