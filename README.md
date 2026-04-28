# Matrix_Factorization

This branch is the lightweight display branch for the research project. The full
codebase, experiment runners, tests, theory notes, and local GPU workflow live
on the `dev` branch.

## Display Contents

```text
Matrix_Factorization/
├── README.md
├── parameters/
│   └── current_local_gpu.yaml
├── results/
│   └── latest/
│       ├── index.json
│       ├── 20260206_0523_bigamp_tensor_parallel_standard_200x200_M50_tensor_n3/
│       ├── 20260206_0505_bigamp_tensor_parallel_standard_200x200_M50_tensor_n3/
│       └── 20260206_0458_bigamp_tensor_parallel_standard_200x200_M50_tensor_n3/
└── showcase_results/
    └── .gitkeep
```

`results/latest/` contains tracked lightweight summaries only. Large tensors,
checkpoints, caches, full experiment artifacts, and local run directories are
intentionally excluded from `main`.

## Current Dev Snapshot

The active research configuration on `dev` has moved back to the spreading
matrix-factorization path used for the sparse-sampling paper audit:

- tensor order: `2`
- algorithm: `bigamp_spreading`
- teacher: `standard` Gaussian factors
- matrix size: `N1=200`, `N2=200`, `M=50`
- scan: `onsager_policy` groups crossed with alpha from `0.0` to `4.0` in
  increments of `0.1`
- samples per alpha: `20`
- max steps: `2000`
- spreading coefficients: Rademacher / `+-1`
- normalization profile: `paper_sparse_sampling`
- precision profile: `aggressive`, with fallback allowed
- default initialization: cold start
- heatmap metric: `Q_W_SIGN_ALIGNED`

The tracked parameter snapshot is stored in
`parameters/current_local_gpu.yaml`. It is a local-GPU research preset, not a
Codex web smoke test.

## Recent Dev Updates

The latest `dev` work since the previous display refresh is mostly research
infrastructure and convention hardening:

- Sparse-sampling paper convention map: latent variables use unit variance,
  interactions use the paper-style `1/sqrt(M)` scale, and the old internal
  `1/M` latent convention is kept only as an explicit legacy profile.
- Formal metric naming: `Q_Y` is an absolute projection, `Q_W` and `Q_X` are
  coordinate projections, and old cosine-style `Q_Y`, MSE, and generalization
  error are not formal metrics.
- Gauge diagnostics: sign-aligned `Q_W/Q_X`, Gram-root diagnostics, and
  posthoc scale-gauge aligned factor diagnostics are integrated into plotting
  and result handling.
- Canonical scan output: group-level plots are generated for individual scan
  groups, while cross-group comparison plots remain part of the full scan
  output.
- Onsager routing: no-Onsager spreading uses the legacy fast no-feedback route;
  fixed/adaptive Onsager use the corrected BiG-AMP message convention.
- Alpha continuation: `scan.continuation` can run alpha in descending order and
  pass full algorithm state from a high-alpha fixed point to the next lower
  alpha.
- Precision profile audit: `safe`, `fast`, and `aggressive` are exposed through
  the contract, but memory and speed gains still depend on actual runtime dtype
  roles and the dominant buffers in each algorithm route.

## Full Project Structure on `dev`

```text
dev
├── src/matrix_factorization/
│   ├── cli.py
│   ├── config.yaml
│   ├── core/
│   │   ├── contracts.py       # hard-interface specs and registries
│   │   ├── experiment/        # config, runner, result schema, continuation
│   │   └── parallel/          # scan resource planning and memory execution
│   ├── modules/
│   │   ├── algorithms/
│   │   │   ├── agd.py
│   │   │   └── bigamp/
│   │   │       ├── standard.py
│   │   │       ├── spreading.py
│   │   │       ├── step.py
│   │   │       └── tensor_spreading_parallel.py
│   │   ├── graphs/
│   │   ├── metrics/
│   │   └── outputs/
│   └── ui/
├── configs/
│   ├── manual_onsager_groups/
│   ├── smoke/
│   └── local_gpu/
├── scripts/
├── tests/
├── docs/
│   ├── METRICS_GUIDE.md
│   ├── scan_system_contract.md
│   └── theory/
│       ├── alpha_descending_continuation.md
│       ├── bigamp_onsager_damping_audit.md
│       └── code_vs_sparse_sampling_paper.md
├── trials/
└── runs/                     # ignored local experiment outputs
```

## Algorithm Map

- `AGD`: alternating-gradient matrix baseline.
- `BiG-AMP Standard`: dense bipartite matrix-factorization AMP baseline.
- `BiG-AMP Spreading`: active sparse-sampling matrix path with random spreading
  coefficients and optional Onsager correction.
- `BiG-AMP Tensor Parallel`: active tensor implementation for tensor-order
  experiments and memory-planning work.
- `legacy` and experimental tensor variants: kept on `dev` for reference and
  parity audits, not treated as the primary display path.

## Branch Roles

- `main`: compact English display branch with lightweight results and the
  current parameter snapshot.
- `dev`: complete research branch for source code, tests, documentation,
  Codex work, local trials, and GPU experiments.
