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

Full dev map: [`docs/project_structure_algorithms_metrics.md`](docs/project_structure_algorithms_metrics.md).

```text
Matrix_Factorization/
├─ src/matrix_factorization/
│  ├─ cli.py
│  ├─ config.yaml
│  ├─ core/
│  │  ├─ contracts.py
│  │  ├─ planning.py
│  │  ├─ scan_planning.py
│  │  ├─ experiment/
│  │  └─ parallel/
│  ├─ modules/
│  │  ├─ algorithms/
│  │  ├─ graphs/
│  │  ├─ metrics/
│  │  ├─ outputs/
│  │  ├─ teachers/
│  │  └─ interventions/
│  ├─ presets/
│  ├─ export/
│  └─ ui/
├─ configs/
├─ trials/
├─ tests/
├─ scripts/
├─ docs/
├─ experiments/
├─ _legacy/
├─ runs/
├─ results/
└─ artifacts/
```

```text
Algorithms
├─ AGD family
│  ├─ agd
│  ├─ agd_tensor
│  └─ agd_spreading
├─ AMP / BiGAMP family
│  ├─ bigamp
│  ├─ bigamp_spreading
│  │  ├─ F distribution
│  │  ├─ graph route
│  │  ├─ Onsager route
│  │  └─ stop route
│  ├─ bigamp_tensor
│  └─ bigamp_tensor_parallel
└─ Helper
   └─ combined
```

```text
Metrics
├─ Output / measurement metrics
│  ├─ Q_Y
│  ├─ Q_Y_COS
│  └─ FIT_Y / NMSE_Y
├─ Matrix latent metrics
│  ├─ Q_W / Q_X
│  ├─ R_W / R_X
│  ├─ Q_W_SIGN_GAUGE / Q_X_SIGN_GAUGE
│  ├─ Q_W_SCALE_GAUGE / Q_X_SCALE_GAUGE / Q_WX_SCALE_GAUGE
│  └─ Q_W_COS_ROOT / Q_X_COS_ROOT
├─ Tensor latent metrics
│  └─ Q_N and Q_N_mode*
└─ Replica diagnostics
   └─ Q_W_replica, Q_X_replica, Q_W_prime_replica, Q_X_prime_replica
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
