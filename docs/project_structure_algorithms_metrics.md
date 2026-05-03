# dev 分支项目结构、算法分支与 Metric 地图

这份文档面向接手项目或共享给同学时的代码阅读。它描述的是 `dev`
分支当前 active path；历史 handoff、旧 notebook、`MF/`、`Wang/` 里出现的名称
可能已经不是当前运行入口。

## 1. 基本入口和约定

- 当前开发分支：`dev`。
- 当前 Python package：`src/matrix_factorization`。
- CLI 入口：`mf ...`，对应 `pyproject.toml` 中的
  `matrix_factorization.cli:main`。
- 默认配置：`src/matrix_factorization/config.yaml`。
- quick/debug/trial 配置：`trials/active/<trial_key>/config.yaml`。
- 生成结果目录：`runs/`、`results/`、`artifacts/`，这些目录不作为源码提交。
- 当前 metric schema：`schema_version = 6`。
- 当前 metric profile：
  `metric_definition_profile = projection_qy_supergraph_full_cos_v3`。

最重要的硬接口规则：

- 新 YAML 参数必须先在 `core/contracts.py` 里有 `ParameterSpec`。
- 新 algorithm 必须先有 `AlgorithmSpec`，再注册到 registry。
- 新 metric/output/analyzer/probe/intervention 必须先有对应 spec。
- 新 source file 如果落在 classified source area，需要更新 source inventory。
- 旧 schema 的同名 flat metric key 不能和 schema v6 静默比较。

## 2. 顶层目录

```text
Matrix_Factorization/
├─ README.md
│  └─ 项目共享入口；只放短说明、关键图和跳转。
├─ AGENTS.md
│  └─ Codex/agent 工作规则：dev 分支、active package、测试命令、artifact policy。
├─ pyproject.toml
│  └─ Python package、依赖、CLI entry point。
├─ src/matrix_factorization/
│  └─ 当前唯一 active Python package。
├─ configs/
│  └─ 版本管理内的研究配置和本地 GPU preset。
├─ trials/
│  ├─ active/
│  │  └─ quick/debug/trial 配置；不直接改正式默认配置。
│  └─ archive/
│     └─ 已归档 trial 配置。
├─ tests/
│  └─ contract、runtime、compatibility、verification、debug/unit 测试。
├─ scripts/
│  ├─ analysis/
│  ├─ debug/
│  ├─ experiments/
│  ├─ maintenance/
│  └─ verification/
│     └─ 独立脚本；脚本能复用 package，但不是 package 入口本身。
├─ docs/
│  ├─ METRICS_GUIDE.md
│  ├─ metrics_semantics.md
│  ├─ current_project_map.md
│  ├─ theory/
│  ├─ reports/
│  ├─ figures/
│  ├─ reference_code/
│  └─ archive/
│     └─ metric 语义、理论审计、handoff、报告和图。
├─ experiments/
│  └─ 分析性小工具或一次性研究脚本。
├─ _legacy/
│  └─ 整理后的历史文件，不作为 active path。
├─ smf/
│  └─ 历史/生成物区域，不是当前 active package。
├─ runs/
│  └─ ignored runtime output；正式或 trial run 结果都不提交。
├─ results/
│  └─ ignored/local 分析或展示结果；只提交明确挑选的小图或文档引用。
└─ artifacts/
   └─ ignored optional large data、tensor、checkpoint、trial artifact。
```

本地工具目录：

- `.agent/`：本地 agent 计划和技能缓存。
- `.vscode/`：本地编辑器配置。
- `.pytest_cache/`、`__pycache__/`：测试/解释器缓存，不是源码。

## 3. `src/matrix_factorization` 包内结构

```text
src/matrix_factorization/
├─ cli.py
│  └─ `mf` CLI：读 YAML、validate/explain/trial、启动 runner。
├─ __main__.py
│  └─ `python -m matrix_factorization` 入口。
├─ config.yaml
│  └─ 当前默认 YAML；不要为 quick trial 临时改它。
├─ core/
│  ├─ contracts.py
│  │  └─ ParameterSpec、AlgorithmSpec、MetricSpec、OutputSpec、ProbeSpec 等硬接口。
│  ├─ planning.py
│  │  └─ 从 YAML 构造 ExperimentPlan；检查参数是否被当前 route 消费。
│  ├─ scan_planning.py
│  │  └─ canonical `scan.axes` 到 ScanPlan/ScanPoint/group 的展开。
│  ├─ experiment/
│  │  ├─ config.py
│  │  │  └─ ExperimentConfig 和 dataclass 配置对象。
│  │  ├─ data_factory.py
│  │  │  └─ teacher、mask、spreading graph、tensor graph、F/Y 构造。
│  │  ├─ runner.py
│  │  │  └─ 主运行器；调 algorithm、收 metrics、写 result。
│  │  ├─ result.py
│  │  │  └─ SingleRunResult/ExperimentResult；canonical run schema 和绘图入口。
│  │  └─ continuation.py
│  │     └─ warm-start/continuation state。
│  ├─ parallel/
│  │  ├─ parallel_coordinator.py
│  │  │  └─ alpha/sample batch 规划。
│  │  ├─ resource_execution.py
│  │  │  └─ WorkItem/ResourceExecutionPlan，记录每个 batch 的调度元数据。
│  │  ├─ memory_estimator.py
│  │  ├─ memory_estimator_general.py
│  │  ├─ memory_guard.py
│  │  ├─ batch_checkpoint.py
│  │  └─ execution_modes.py
│  ├─ precision.py
│  ├─ physics_eta.py
│  ├─ gpu_monitor.py
│  ├─ progress.py
│  ├─ trials.py
│  └─ time_estimator.py
├─ modules/
│  ├─ registry.py
│  │  └─ algorithm/teacher/graph/metric/output registry；注册必须被 spec gate。
│  ├─ algorithms/
│  │  └─ AGD、BiGAMP、spreading、tensor 和 legacy 算法。
│  ├─ graphs/
│  │  └─ dense random mask、uniform/low-loop graph、spreading supergraph。
│  ├─ metrics/
│  │  └─ overlap、Q_Y、spreading、tensor、replica、contract adapter。
│  ├─ outputs/
│  │  └─ storage、plotting、publication style、latest display。
│  ├─ teachers/
│  │  └─ standard、orthogonal、random spreading teacher。
│  └─ interventions/
│     └─ runtime hook/intervention 骨架；不能绕过 AlgorithmStateView。
├─ presets/
│  ├─ builtin/
│  └─ test/
├─ export/
│  └─ standalone bundler/export 路径。
└─ ui/
   └─ CLI progress、browser、menu、theme、wizard helper。
```

## 4. 主运行链路

```text
YAML
  -> cli.load_yaml_config()
    -> core.planning.ExperimentPlan
      -> preflight validation / explain-config
    -> core.experiment.config.ExperimentConfig
      -> core.experiment.runner.ExperimentRunner.run()
        -> core.scan_planning.ScanPlan
        -> core.parallel.ParallelCoordinator.plan_execution()
        -> core.parallel.ResourceExecutionPlan / WorkItem
        -> core.experiment.data_factory.DataFactory.create()
          -> teacher / graph / F / Y / mask
        -> modules.registry.get_algorithm()
          -> AlgorithmSpec-gated algorithm
          -> algorithm.train_batch_alphas()
        -> ExperimentRunner._compute_metrics()
          -> MetricSpec-gated flat metric payload
        -> core.experiment.result.ExperimentResult.save()
          -> config.json / metadata.json / metrics.json / results.pt / plots
```

关键点：

- `scan.axes` 是 canonical scan UI；旧 `alpha_scan`、`steps_scan`、
  `nested_scan` 只作为历史概念保留。
- child runner 在 canonical scan 里不能重启顶层 progress UI，必须带 scan
  coordinate 和 resource batch 元数据。
- `output.name` 只是短标签；run directory 由 effective config、size、sample、
  steps、axis 和 hash 派生。

## 5. Algorithm 分支总览

| Algorithm key | 状态 | 主要文件 | 数据路线 | 主要用途 |
| --- | --- | --- | --- | --- |
| `agd` | active | `modules/algorithms/agd.py` | dense matrix teacher + dense mask | 梯度下降 baseline；用于和 AMP/BiGAMP 曲线对照。 |
| `bigamp` | active | `modules/algorithms/bigamp/standard.py` | dense matrix teacher + dense mask | dense BiGAMP；无 random spreading F；有 damping/noise variance。 |
| `bigamp_spreading` | active | `modules/algorithms/bigamp/spreading.py` | matrix teacher + spreading graph + `F_super` | 当前主要 random-spreading matrix route；支持 Onsager/no Onsager/adaptive damping 和 teacher-assisted diagnostic stop。 |
| `bigamp_tensor` | active/reference | `modules/algorithms/bigamp/tensor_spreading.py` | tensor teacher factors + tensor hypergraph + `F_tensor` | serial/reference tensor route；不是大规模并行主路线。 |
| `bigamp_tensor_parallel` | active | `modules/algorithms/bigamp/tensor_spreading_parallel.py` | tensor teacher factors + tensor supergraph + `F_tensor` | tensor spreading parallel route；支持 alpha batch、tensor metrics、overlap matrix artifact。 |
| `agd_tensor` | experimental | `modules/algorithms/agd_tensor.py` | internal tensor teacher | 实验性 tensor AGD；不是当前主实验路线。 |
| `agd_spreading` | legacy_broken | `modules/algorithms/legacy/agd_spreading.py` | matrix teacher + spreading graph + `F_super` | 历史 spreading AGD reference；active package 不把它作为正式 route。 |
| `combined` | helper | `modules/algorithms/combined.py` | none | selector/helper，不是 trainable algorithm。 |

### 5.1 `agd`

- 模型：dense matrix factorization。
- 预测：`Y_student = (1/sqrt(M)) * W_student @ X_student`。
- 优化：gradient descent，主要参数是 `algorithm_params.learning_rate`。
- 数据：dense teacher matrix 和 observed/unobserved dense mask。
- 输出：matrix full/observed/unobserved `Q_Y`、`Q_Y_COS`，以及
  `Q_W/Q_X`、sign gauge、scale gauge、replica diagnostic。
- 作用：慢但直观，适合作为 sanity baseline 和论文图中的比较对象。

### 5.2 `bigamp`

- 模型：dense matrix BiGAMP。
- 预测同 dense matrix route。
- 更新：AMP/BiGAMP posterior update；主要参数是 `damping`、`noise_var`。
- 数据：dense teacher matrix 和 dense mask。
- 区别于 `agd`：不是梯度下降，而是 message passing；通常步数少很多。
- 区别于 `bigamp_spreading`：没有 random spreading 的 per-edge `F`，也不使用
  spreading supergraph。

### 5.3 `bigamp_spreading`

- 模型：

```text
Y_ij = (1/sqrt(M)) * sum_mu F_ij,mu * W_i,mu * X_mu,j
```

- `F` 是 quenched random disorder，可以是 `ising` 或 `gaussian`。
- 数据对象：`spreading_graph`、`F_super`、`Y_super`。
- full/observed/unobserved 的 `Q_Y` 都必须使用同一套 F-aware supergraph：
  full 是 `0:C_max`，observed 是 `0:C_k`，unobserved 是 `C_k:C_max`。
- 这个路线有几个重要子分支：

| 子分支 | 配置/触发 | 含义 |
| --- | --- | --- |
| no Onsager | `spreading.onsager_correction: false` | 不使用 Onsager residual；常用于对照 random F 本身的贡献。 |
| fixed Onsager | `spreading.onsager_correction: true`, `adaptive_damping: false` | 使用修正后的固定 Onsager update；当前物理扫描常用。 |
| adaptive Onsager | `spreading.onsager_correction: true`, `adaptive_damping: true` | 使用 adaptive acceptance/damping；适合诊断不稳定区域。 |
| bipartite/flat | `allow_intra_connection: false` | W-X 二部结构；大部分 matrix spreading scan 使用这个。 |
| general graph | `allow_intra_connection: true` | 允许 WW/WX/XX edge 类型；使用 general supergraph kernel。 |
| `F=ising` | `spreading.f_distribution: ising` | F 取离散符号；当前多数对比图使用这个。 |
| `F=gaussian` | `spreading.f_distribution: gaussian` | F 为 Gaussian；用于分布敏感性检查。 |
| metric plateau stop | `use_metric_plateau_stop: true` | teacher-assisted convergence diagnostic，只读 `Q_W/Q_X` 等监控量，不改变 AMP 公式。 |

注意：early/convergence stop 是实验加速和诊断工具，不是无监督算法本身的一部分；
metadata 必须标出 teacher-assisted。

### 5.4 Tensor routes

- `bigamp_tensor`：serial/reference tensor route，保存 metrics-only result。
- `bigamp_tensor_parallel`：parallel tensor supergraph route，支持 alpha batch、
  tensor full/observed/unobserved `Q_Y` 和 `Q_N`。
- `agd_tensor`：实验性 route，contract 中登记但不作为主路线。

Tensor route 的 factor metric 是 `Q_N`，不是 `Q_W/Q_X`。如果要和 matrix route
比较，必须先明确 teacher scaling、alpha 定义、graph object、damping 语义和
metric scope。

## 6. Teacher、Graph、F 和初始化

当前 teacher spec 中的核心约定：

- `standard` teacher：按 `teacher_config.init_distribution` 生成 W/X。
- `orthogonal` teacher：SVD/orthogonal 风格，按 normalization profile 缩放。
- `random_spreading` teacher：spreading route 用的随机 F/Y 构造辅助。
- `normalization_profile = paper_sparse_sampling` 时，teacher/student latent entry
  variance 按论文式 sparse sampling 约定；输出里显式使用
  `Y = (1/sqrt(M)) * W @ X` 或 spreading 版本
  `Y_ij = (1/sqrt(M)) * sum_mu F_ij,mu W_i,mu X_mu,j`。
- `normalization_profile = internal_normalized_legacy` 是历史归一化路径；旧结果不能
  和当前 paper profile 静默比较。

Graph/F 相关目录：

- `modules/graphs/random.py`、`uniform.py`、`low_loop.py`：dense 或常规图构造。
- `modules/graphs/supergraph.py`：flat/bipartite spreading supergraph。
- `modules/graphs/supergraph_general.py`：general graph supergraph。
- `modules/algorithms/bigamp/f_gen.py`：`F` 和 `Y_super` 的生成及 F-aware output。
- `modules/algorithms/bigamp/tensor_supergraph.py`、`tensor_hypergraph.py`：tensor
  supergraph/hypergraph。

## 7. Metric schema v6 总览

当前 `metrics.json.metric_schema` 应包含：

```text
schema_version = 6
metric_definition_profile = projection_qy_supergraph_full_cos_v3
metric_policy.Q_Y_formula = absolute_projection_teacher_norm_squared
metric_policy.Q_W_Q_X_normalization = fixed_coordinate_count
metric_policy.spreading_Q_Y_full_scope = full_supergraph_F_aware_measurements_0_Cmax
metric_policy.spreading_Q_Y_cosine_suffix = _COS
metric_policy.legacy_projection_suffix = _PROJ_ABS
metric_policy.clipped = false
```

读取结果时必须先看 schema/profile，再解释 flat key。尤其是：

- schema v3/v4 的 `Q_Y_mean` 不是 schema v6 的完整语义。
- schema v5 的 spreading full `Q_Y` 是旧 equal-count/full 语义，不能当作
  schema v6 的 full supergraph F-aware `Q_Y`。
- `Q_Y_PROJ_ABS`、`Q_W_PROJ_ABS`、`Q_X_PROJ_ABS` 是 legacy diagnostic，不是当前
  formal metric 的主解释。

## 8. `Q_Y` family

### 8.1 Formal `Q_Y`

`Q_Y` 是 output absolute projection：

```text
Q_Y = abs(sum_e Y_student[e] * Y_teacher[e]) / sum_e Y_teacher[e]^2
```

对应 reconstruction fit：

```text
NMSE_Y = sum_e (Y_student[e] - Y_teacher[e])^2 / sum_e Y_teacher[e]^2
FIT_Y = 1 - NMSE_Y
```

特点：

- `Q_Y` 不裁切；如果 scale 太大，可以大于 `1`。
- `FIT_Y` 不裁切；坏结果可以小于 `0`。
- 完美重构时 `Q_Y = 1` 且 `FIT_Y = 1`。
- 零输出通常给 `Q_Y = 0`、`FIT_Y = 0`。
- 如果 evaluation set 为空或 teacher norm 过小，按 contract 返回安全默认值。

### 8.2 `Q_Y_COS`

`Q_Y_COS` 是同一 measurement scope 上的 signed cosine：

```text
Q_Y_COS = sum_e Y_student[e] * Y_teacher[e]
          / (sqrt(sum_e Y_student[e]^2) * sqrt(sum_e Y_teacher[e]^2))
```

它不吸收 scale 错误。比如 `Y_student = 2 * Y_teacher` 时：

```text
Q_Y = 2
Q_Y_COS = 1
```

因此共享展示图里用 `Q_Y_COS` 更适合说明输出方向是否对齐；`Q_Y/FIT_Y/NMSE_Y`
仍然保留，用于检查 scale 和 reconstruction error。

### 8.3 Dense matrix `Q_Y` scopes

| MetricSpec | Flat keys | Scope |
| --- | --- | --- |
| `matrix.full.Q_Y` | `Q_Y_mean`, `FIT_Y_mean`, `NMSE_Y_mean` | 所有 dense matrix entries。 |
| `matrix.observed.Q_Y` | `Q_Y_observed_mean`, `FIT_Y_observed_mean`, `NMSE_Y_observed_mean` | observed/training dense mask。 |
| `matrix.unobserved.Q_Y` | `Q_Y_unobserved_mean`, `FIT_Y_unobserved_mean`, `NMSE_Y_unobserved_mean` | dense mask 的 heldout/unobserved entries。 |
| `matrix.full.Q_Y_COS` | `Q_Y_COS_mean` | full output signed cosine。 |
| `matrix.observed.Q_Y_COS` | `Q_Y_observed_COS_mean` | observed output signed cosine。 |
| `matrix.unobserved.Q_Y_COS` | `Q_Y_unobserved_COS_mean` | unobserved output signed cosine。 |

### 8.4 Spreading `Q_Y` scopes

| MetricSpec | Flat keys | Scope |
| --- | --- | --- |
| `spreading.full.Q_Y` | `Q_Y_mean`, `FIT_Y_mean`, `NMSE_Y_mean` | full supergraph F-aware measurement，`0:C_max`。 |
| `spreading.observed.Q_Y` | `Q_Y_observed_mean`, `FIT_Y_observed_mean`, `NMSE_Y_observed_mean` | observed prefix，`0:C_k`。 |
| `spreading.unobserved.Q_Y` | `Q_Y_unobserved_mean`, `FIT_Y_unobserved_mean`, `NMSE_Y_unobserved_mean` | unobserved suffix，`C_k:C_max`。 |
| `spreading.full.Q_Y_COS` | `Q_Y_COS_mean` | full supergraph F-aware signed cosine。 |
| `spreading.observed.Q_Y_COS` | `Q_Y_observed_COS_mean` | observed prefix signed cosine。 |
| `spreading.unobserved.Q_Y_COS` | `Q_Y_unobserved_COS_mean` | unobserved suffix signed cosine。 |

这里的 full/observed/unobserved 必须来自同一组 `F_super/Y_super`。不能为了每个
alpha 或每个 batch 重新定义一套不同的 complete graph；否则不同 alpha 点之间的
`Q_Y` 就不是同一个统计对象。

### 8.5 Tensor `Q_Y` scopes

| MetricSpec | Flat keys | Scope |
| --- | --- | --- |
| `tensor.full.Q_Y` | `Q_Y_mean`, `FIT_Y_mean`, `NMSE_Y_mean` | observed + deterministic heldout tensor hyperedges。 |
| `tensor.serial_observed.Q_Y` | `Q_Y_mean`, `Q_Y_observed_mean` | serial tensor route 的 observed hyperedges；兼容旧 flat key。 |
| `tensor.observed.Q_Y` | `Q_Y_observed_mean`, `FIT_Y_observed_mean`, `NMSE_Y_observed_mean` | parallel tensor observed hyperedges。 |
| `tensor.unobserved.Q_Y` | `Q_Y_unobserved_mean`, `FIT_Y_unobserved_mean`, `NMSE_Y_unobserved_mean` | deterministic heldout tensor hyperedges。 |

## 9. `Q_W/Q_X` latent factor family

`Q_W` 和 `Q_X` 是 fixed-denominator physical latent overlap：

```text
Q_W = sum_i,mu W_student[i,mu] * W_teacher[i,mu] / (N1 * M)
Q_X = sum_mu,j X_student[mu,j] * X_teacher[mu,j] / (M * N2)
```

这不是 teacher-norm projection。Gaussian teacher 在有限尺寸下完美恢复时，
`Q_W/Q_X` 接近 teacher empirical second moment，但不被强制等于 `1`。

Self-overlap：

```text
R_W = sum_i,mu W_student[i,mu]^2 / (N1 * M)
R_X = sum_mu,j X_student[mu,j]^2 / (M * N2)
```

`R_W/R_X` 用来判断 student norm collapse 或 blow-up。

主要 flat keys：

| MetricSpec | Flat keys | 说明 |
| --- | --- | --- |
| `matrix.factor.Q_W` | `Q_W_mean`, `Q_W_std` | W physical overlap。 |
| `matrix.factor.Q_W` | `R_W_mean`, `R_W_std` | W student self-overlap。 |
| `matrix.factor.Q_X` | `Q_X_mean`, `Q_X_std` | X physical overlap。 |
| `matrix.factor.Q_X` | `R_X_mean`, `R_X_std` | X student self-overlap。 |

## 10. Gauge 和 latent diagnostic

这里的 `gauge` 不是 gate。它表示 factorization 里的不唯一性：例如同一个输出
可以由不同 sign 或 scale 的 latent factor 表示。Gauge diagnostic 是 posthoc
读数，不改变训练轨迹。

### 10.1 Sign gauge

每个 latent channel 允许独立正负号对齐：

```text
Q_W_SIGN_GAUGE = sum_mu abs(sum_i W_student[i,mu] * W_teacher[i,mu]) / (N1 * M)
Q_X_SIGN_GAUGE = sum_mu abs(sum_j X_student[mu,j] * X_teacher[mu,j]) / (M * N2)
```

Flat keys：

- `Q_W_SIGN_GAUGE_mean/std`
- `Q_X_SIGN_GAUGE_mean/std`
- `Q_W_SIGN_ALIGNED_mean/std`
- `Q_X_SIGN_ALIGNED_mean/std`

`SIGN_ALIGNED` 是 legacy alias；当前正式名是 `SIGN_GAUGE`。

### 10.2 Scale gauge

每个 latent channel 求一个 diagonal scale `k_mu`，同时对齐 W 和 X：

```text
k_mu = argmin_k [
  sum_i (k * W_student[i,mu] - W_teacher[i,mu])^2
  + sum_j ((1/k) * X_student[mu,j] - X_teacher[mu,j])^2
]
```

然后用 fixed-denominator convention 重新计算 aligned overlap。

Flat keys：

- `Q_W_SCALE_GAUGE_mean/std`
- `Q_X_SCALE_GAUGE_mean/std`
- `Q_WX_SCALE_GAUGE_mean/std`
- `median_abs_log_k_mean/std`
- `median_abs_log_g_mean/std`，legacy alias。

### 10.3 Cos-root

Cos-root diagnostic 用 baseline-corrected Gram overlap：

```text
Q_W_COS_ROOT = sqrt(max(baseline_corrected_gram_overlap(W), 0))
Q_X_COS_ROOT = sqrt(max(baseline_corrected_gram_overlap(X), 0))
```

Flat keys：

- `Q_W_COS_ROOT_mean/std`
- `Q_X_COS_ROOT_mean/std`

它不是 physical overlap，不应该和 `Q_W_mean/Q_X_mean` 互换解释。

## 11. Tensor factor metric `Q_N`

Tensor latent factor 使用同一 fixed-denominator convention：

```text
Q_N_mode_d = sum_entries N_student_mode_d * N_teacher_mode_d
             / numel(N_teacher_mode_d)
Q_N = mean_d Q_N_mode_d
```

Flat keys：

- `Q_N_mean/std`
- `Q_N_mode0_mean/std`
- `Q_N_mode1_mean/std`
- `Q_N_mode2_mean/std`
- `Q_N_mode3_mean/std`

## 12. Replica diagnostic

`replica.factor` 是 student-student diagnostic，用来观察不同 replica 之间的
相似性，不是 teacher-student physical metric。

Flat keys：

- `Q_W_replica_mean`
- `Q_X_replica_mean`
- `Q_W_prime_replica_mean`
- `Q_X_prime_replica_mean`

## 13. Output 和 result schema

标准 run 目录：

```text
run_dir/
├─ config.json
├─ metadata.json
├─ metrics.json
├─ events.jsonl
├─ artifacts/
│  └─ results.pt
├─ results.pt -> artifacts/results.pt
├─ plots/
└─ manifest.json
```

常见输出组件：

- `modules/outputs/storage.py`：保存 result payload。
- `modules/outputs/plotting.py`：常规曲线图。
- `modules/outputs/publication_style.py`：论文风格绘图 helper。
- `modules/outputs/latest.py`：轻量 latest display。
- `core/experiment/result.py`：canonical run save、metric schema、plot labels。

共享图片可以提交到 `docs/figures/` 或 README 引用的位置；完整 tensor、`.pt`、
run directory 不提交。

## 14. 测试地图

常用 quick checks：

```bash
python -m pytest -q tests/test_contract_parameter_specs.py tests/test_config_contract.py
python -m pytest -q tests/test_trial_contract.py tests/test_trial_cli.py
mf validate src/matrix_factorization/config.yaml
```

结构或硬接口改动后：

```bash
python -m pytest -q
```

重点测试类别：

- contract/spec：`tests/test_contract_*.py`、`tests/test_config_contract.py`
- resource plan：`tests/test_scan_resource_*.py`
- trial CLI：`tests/test_trial_*.py`
- metric/plot compatibility：`tests/test_tensor_metrics.py`、`tests/test_plotting_compat.py`
- algorithm convention：`tests/test_bigamp_onsager_convention.py`、
  `tests/test_spreading_batch_metrics.py`

## 15. 改动入口速查

| 想改什么 | 先看哪里 | 必须同步什么 |
| --- | --- | --- |
| 新 YAML 参数 | `core/contracts.py` 的 `ParameterSpec` | config dataclass、parser、effective trace、route consumer、warning/strict behavior。 |
| 新 algorithm | `AlgorithmSpec`、`modules/registry.py` | algorithm 注册、runner result contract、metric spec、tests。 |
| 新 metric | `MetricSpec`、`modules/metrics/spec_adapter.py` | metric 计算、flat key、semantic metadata、docs/METRICS_GUIDE.md。 |
| 新 output/plot | `OutputSpec`、`modules/outputs/` | dependency preflight、plot label、result schema compatibility。 |
| 新 graph/teacher | `GraphSpec`/`TeacherSpec`、`DataFactory` | seed/scaling convention、metadata、tests。 |
| 新 diagnostic stop/probe | `ProbeSpec` 或 algorithm params | teacher-assisted 标记、metadata、inactive route handling。 |
| 新 source file | 对应 classified inventory | `tests/test_source_inventory.py`。 |

## 16. 不要混用的历史语义

- 不要把 schema v3/v4/v5 的 `Q_Y_mean/Q_W_mean/Q_X_mean` 当作 schema v6。
- 不要把旧 spreading equal-count full `Q_Y` 当作现在的 full supergraph F-aware
  `Q_Y`。
- 不要把 `Q_Y_PROJ_ABS` 当作 `Q_Y_COS`。
- 不要把 `SIGN_ALIGNED` 当成新的独立 metric；它只是 `SIGN_GAUGE` legacy alias。
- 不要把 `_legacy/`、`smf/`、历史 `MF/`/`Wang/` 路径当作 active package。
- 不要把 `runs/`、`results/`、`artifacts/` 或 `.pt` 结果提交进源码。
