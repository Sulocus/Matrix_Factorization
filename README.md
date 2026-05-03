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

当前开发分支是 `dev`，active package 是 `src/matrix_factorization`。
历史笔记里出现的 `MF/`、`Wang/` 或旧脚本目录不是当前主入口。更完整的
目录、算法分支和 metric 说明见
[`docs/project_structure_algorithms_metrics.md`](docs/project_structure_algorithms_metrics.md)。

```text
Matrix_Factorization/
  src/matrix_factorization/   # 当前 Python package、CLI、runner、algorithm、metric
  configs/                    # 版本管理内的本地研究配置和 GPU preset
  trials/                     # quick/debug trial 配置；输出仍写 ignored run 区
  tests/                      # contract、runtime、compatibility、verification 测试
  scripts/                    # analysis、experiment、debug、maintenance 工具
  docs/                       # 项目地图、metric 语义、理论审计、handoff、报告和图
  experiments/                # 分析性小工具或独立研究脚本
  _legacy/                    # 已整理的历史文件；不作为 active path
  runs/                       # ignored runtime output，不提交
  results/                    # ignored/local 展示或分析输出，不提交大结果
  artifacts/                  # ignored optional large data，不提交
```

`src/matrix_factorization/` 内部主要分层如下：

```text
core/                         # config、contract、plan、runner、parallel resource plan
modules/algorithms/           # agd、bigamp、bigamp_spreading、tensor variants
modules/graphs/               # dense mask、spreading supergraph、tensor supergraph
modules/metrics/              # Q_Y/Q_W/Q_X/Q_N、sign/scale/cos/replica diagnostics
modules/outputs/              # plotting、storage、latest display、publication style
modules/teachers/             # standard、orthogonal、random spreading teachers
presets/                      # package 内置 preset
ui/                           # CLI UI/progress/browser helpers
```

当前主要 algorithm key：

| Key | 用途 | 主要区别 |
| --- | --- | --- |
| `agd` | dense matrix AGD baseline | 梯度下降；dense mask；主要用于和 AMP 结果对照。 |
| `bigamp` | dense matrix BiGAMP | dense observation route；有 damping/noise variance；无 random spreading F。 |
| `bigamp_spreading` | matrix random-spreading BiGAMP | 使用 F-aware spreading supergraph；支持 Onsager/no Onsager/adaptive damping 分支；当前主要物理扫描路线。 |
| `bigamp_tensor` | tensor spreading serial/reference | tensor 参考实现；不是大规模并行主路线。 |
| `bigamp_tensor_parallel` | tensor spreading parallel | tensor supergraph 并行路线；支持 alpha batch 和 tensor metrics。 |
| `agd_tensor` | experimental tensor AGD | contract 中登记为 experimental，当前不是主实验路线。 |
| `agd_spreading` | legacy spreading AGD | legacy/broken reference，active package 默认不把它当主路线。 |

当前 metric schema 为 `schema_version = 6`，
`metric_definition_profile = projection_qy_supergraph_full_cos_v3`。最常用
metric family：

| Family | 主要 flat keys | 语义 |
| --- | --- | --- |
| `Q_Y` | `Q_Y_mean`, `Q_Y_observed_mean`, `Q_Y_unobserved_mean` | output absolute projection；spreading full 使用同一套 full supergraph F-aware measurement。 |
| `Q_Y_COS` | `Q_Y_COS_mean`, `Q_Y_observed_COS_mean`, `Q_Y_unobserved_COS_mean` | output signed cosine；当前共享展示图优先使用这个来比较 Y。 |
| `FIT_Y/NMSE_Y` | `FIT_Y_mean`, `NMSE_Y_mean` 等 | output reconstruction fit，`FIT_Y = 1 - NMSE_Y`。 |
| `Q_W/Q_X` | `Q_W_mean`, `Q_X_mean`, `R_W_mean`, `R_X_mean` | fixed-denominator physical latent overlap；`R_*` 是 student self-overlap。 |
| sign gauge | `Q_W_SIGN_GAUGE_mean`, `Q_X_SIGN_GAUGE_mean` | 每个 latent channel 允许独立正负号对齐；`SIGN_ALIGNED` 是 legacy alias。 |
| scale gauge | `Q_W_SCALE_GAUGE_mean`, `Q_X_SCALE_GAUGE_mean`, `Q_WX_SCALE_GAUGE_mean` | posthoc diagonal scale 对齐 diagnostic，不改变训练。 |
| cos-root | `Q_W_COS_ROOT_mean`, `Q_X_COS_ROOT_mean` | latent Gram/cos-root diagnostic，不是 physical overlap。 |
| tensor | `Q_N_mean`, `Q_N_mode*_mean` | tensor factor fixed-denominator overlap。 |

不要把旧 schema v3/v4/v5 的同名 `Q_Y/Q_W/Q_X` flat keys 和当前 schema v6
静默比较；必须同时检查 `metric_schema.schema_version` 和
`metric_definition_profile`。

## Artifact Policy

Large generated tensors, checkpoints, heatmap batches, and historical experiment
directories are not source code. Keep them in ignored `runs/`, `results/`,
`artifacts/`, or `src/matrix_factorization/Replica_results/`, and record
externally stored data in `docs/artifacts_manifest.md` when needed.

Use this repository for code review, contract tests, quick trials, and versioned
changes. High-memory scientific validation should run as local GPU validation.

## More

See `docs/local_workflow.md` for the local workflow and run artifact schema.
