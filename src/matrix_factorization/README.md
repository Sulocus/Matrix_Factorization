# Matrix Factorization Package

This package contains the installable experiment framework used by the root
`mf` command.

Current entry points:

```bash
mf configs/smoke/matrix_spreading_smoke.yaml
mf configs/local_gpu/tensor_local_gpu.yaml --output-dir runs
```

Generated runs use the repository-level `runs/{run_id}/` schema:

```text
manifest.json
config.json
metrics.json
events.jsonl
artifacts/
plots/
checkpoints/latest.pt
```

Do not store generated experiment outputs inside this package. Historical local
data under `src/matrix_factorization/Replica_results/` is ignored and should not
be committed again.
