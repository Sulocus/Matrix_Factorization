# Matrix_Factorization

This branch is the lightweight display branch for the research project. The full
codebase, experiment runners, tests, and Codex Web development workflow live on
the `dev` branch.

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

`results/latest/` tracks the newest three lightweight result summaries shared
with `dev`. Large tensors, checkpoints, caches, and full experiment artifacts
are intentionally excluded.

## Current Parameters

`parameters/current_local_gpu.yaml` is the current local-GPU research preset:

- tensor order: 3
- algorithm: `bigamp_tensor`
- teacher: `standard`
- matrix: `N1=200`, `N2=200`, `M=50`
- scan: alpha from `0.0` to `4.0` with step `0.1`
- samples per alpha: 20
- max steps: 2000
- precision/performance: `torch.compile` and BF16 enabled
- saved tensors: disabled for lightweight output

## Full Project Structure on `dev`

```text
dev
├── src/matrix_factorization/
│   ├── cli.py
│   ├── runner.py
│   ├── config.yaml
│   ├── core/
│   │   ├── experiment/        # config, runner, result schema
│   │   ├── checkpoint.py      # resumable local experiments
│   │   └── parallel/          # batch checkpoint helpers
│   ├── modules/
│   │   ├── algorithms/
│   │   │   ├── agd.py         # alternating gradient baseline
│   │   │   ├── agd_tensor.py  # tensor AGD baseline
│   │   │   └── bigamp/
│   │   │       ├── standard.py
│   │   │       ├── spreading.py
│   │   │       ├── tensor_step.py
│   │   │       ├── tensor_spreading_parallel.py
│   │   │       └── tensor_step_super.py
│   │   ├── graphs/            # uniform, random, low-loop, supergraph builders
│   │   ├── teachers/          # standard, orthogonal, scaled, random-spreading
│   │   ├── metrics/           # overlaps, replica, tensor and unobserved metrics
│   │   └── outputs/           # storage, plotting, latest-result export
│   └── ui/                    # local result browsing tools
├── configs/
│   ├── smoke/                 # lightweight Codex/Web checks
│   └── local_gpu/             # real local-GPU research presets
├── scripts/
│   ├── analysis/
│   ├── experiments/
│   ├── maintenance/
│   └── verification/
├── tests/
├── docs/
├── results/alpha_scan/        # ignored local full experiment outputs
└── results/latest/            # tracked lightweight latest summaries
```

## Algorithm Map

- `AGD`: alternating-gradient baseline for matrix factorization; slower but
  useful as an optimization reference.
- `AGD Tensor`: tensor-order extension of AGD for comparison against AMP-style
  tensor methods.
- `BiG-AMP Standard`: bipartite matrix-factorization AMP baseline.
- `BiG-AMP Spreading`: BiG-AMP with random spreading factors and optional
  Onsager-related corrections.
- `BiG-AMP Tensor`: tensor-order AMP path used when `tensor_order >= 3`.
- `BiG-AMP Tensor Parallel`: current main tensor implementation for local GPU
  experiments.
- `Tensor Super/Batch` variants: active research branches for memory and
  parallel execution behavior.
- `legacy`: historical algorithm variants kept on `dev` for reference only.

## Branch Roles

- `main`: compact display branch with latest lightweight results and the current
  parameter snapshot.
- `dev`: complete research branch for code, tests, scripts, docs, Codex Web
  review, and local GPU experiments.
