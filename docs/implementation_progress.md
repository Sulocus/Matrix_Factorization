# Implementation Progress

这份文档记录当前硬接口化施工状态。它只写已经进入代码、测试和提交的内容；没有完成的部分明确标成 limitation 或 next work。

最新接手说明见：

- `docs/project_handoff_2026-04-28.md`：当前长对话的最新交接，包含论文 convention、归一化风险和下一步任务。
- `docs/project_handoff_2026-04-27.md`：hard-interface / scan / trial / ResultCube / probe 的上一阶段交接。

## 当前 checkpoint

- 分支：`dev`
- 当前未提交施工线：normalization schema v4 + precision profile migration。
  - active algorithms 已新增 `NormalizationSpec` / `PrecisionPolicySpec`。
  - teacher/student/tensor init 正在统一到 `1/sqrt(M)` latent scale 与 `1/M` variance。
  - `precision_profile: safe / fast / aggressive` 已接入 config、plan、memory estimator 和主要 algorithm metadata。
  - 这会改变 spreading/tensor 数值行为，属于 breaking semantic migration；旧 result 不应和 schema v4 result 静默比较。
- 最近阶段提交：
  - `e0e8b31 Complete projection metric contract fixtures`
  - `256ce69 Harden AlgorithmResult active paths`
  - `7733425 Calibrate scan-aware memory estimator`
  - `8da8fb5 Wire first step-level runtime probe`
  - `d78e43c Complete ResultCube plot query flow`
- 最近全量测试：`python -m pytest -q` -> `295 passed, 9 skipped`
- 最近正式配置校验：`mf validate src/matrix_factorization/config.yaml` -> no errors，只有 parsed-only / inactive-route warnings。

## Completed

### Hard-interface baseline

- `ParameterSpec / AlgorithmSpec / TeacherSpec / GraphSpec / MetricSpec / OutputSpec / InterventionSpec / ProbeSpec / AnalyzerSpec / SourceInventorySpec / TrialSpec / ResourceSpec / BatchingSpec / MemoryModelSpec / SeedPolicySpec` 已进入 `src/matrix_factorization/core/contracts.py`。
- registry 已 spec-gated：新增 algorithm/teacher/graph/metric/output 没有 matching spec 会 import/test fail。
- `ExperimentPlan`、`mf validate`、`mf explain-config`、`--json`、`--strict` 已接入。
- `parameter_chain` 会记录 YAML path、effective value、consumer、当前 route 是否 active、物理敏感性。

### Trial workflow

- `trials/active/<trial_key>/config.yaml` 是 quick/debug trial 的唯一受控参数入口。
- `mf trial list/explain/validate/run` 已实现。
- `mf trial run` 只自动运行 `runtime_class: quick`，输出进入 ignored `runs/trials/`，不刷新 `results/latest`。
- 已有 quick trials：`matrix_bigamp_quick`、`scan_alpha_quick`、`scan_steps_quick`、`scan_size_quick`、`scan_init_quick`、`scan_mixed_axes_quick`。

### Projection metric schema v3

- Formal metric 已迁移到 projection-first：
  - `Q_Y`：absolute projection，teacher norm squared normalization，不 clip。
  - `Q_W/Q_X`：matrix latent coordinate projection。
  - `Q_N`：tensor latent node/spin/factor projection。
  - `Q_W_COS_ROOT/Q_X_COS_ROOT`：Gram diagnostic，不是 coordinate projection。
- Formal schema 不再把 `MSE / Gen_Error / Q_Y_COS / physical_overlap_* / Q_W_prime / Q_X_prime` 当 active metric。
- 旧 result schema `<3` 的 `Q_Y_mean` 会标记为 legacy cosine/proxy；schema `3` 的 `Q_Y_mean` 标记为 absolute projection。
- `projection_policy` 记录 degenerate teacher norm 约定：teacher norm 过小时返回 0。
- 手算 fixture 已覆盖 projection helper、matrix observed/unobserved/full、Cos-root 分离、spreading observed perfect teacher、tensor observed/Q_N。

### AlgorithmResult active path

- runner active path 优先消费 `AlgorithmResult.metrics_by_alpha`。
- metrics-only tensor route 不再保存 dummy zero `W/X` 当真实 factor。
- active tensor route 缺 metric 时 runner 不再从私有 `_batch_metrics` 补救。
- `agd / bigamp / bigamp_spreading` 已有原生 `train_batch_result()`，active path 直接返回 native matrix `AlgorithmResult`。
- legacy `train_batch_alphas()` 仍保留给旧调用方；base adapter 仍作为非 migrated algorithm 的兼容层。
- run metadata 的 `contract.algorithm_result_batches` 会记录每个 batch 的 `result_source/result_kind/result_contract/available_outputs`，因此不用开启 probe 也能回看 active path 的 result 来源。

### Canonical scan + ResultCube + PlotQuery

- 旧 `scan_mode / alpha_scan / steps_scan / nested_scan / hysteresis_scan` 已被 canonical `scan.axes` 主入口替代。
- `ScanPlan` 展开 `ScanPoint`，每个点有 coordinates、overrides、effective_config_hash、group_id。
- canonical scan 结果写入 `ResultCube`。
- `ResultCube.resolve_plot_query()` 已成为查询硬边界：
  - `where / series_by / compare / x / y` 引用不存在坐标会 error。
  - 缺 metric 会 error。
  - 同一条曲线选中重复 x 值会 error，要求收紧 `where/compare` 或增加 `series_by`。
- `mf explain-config` 会预览 PlotQuery 选择的 series 和 point_id。
- `scan_mixed_axes_quick` 已真实生成 `plots/qy_mixed_axes.png`。

### Scan-aware memory / resource planning

- `ResourceExecutionPlan` 已按 canonical scan 的 non-alpha group 分组。
- 当前真实自动 folding 维度只有 `alpha`；`sample/student` folding 仍被 contract 禁用。
- `size/init/damping/onsager/max_steps` 默认不跨 group folding。
- planner 会把 memory estimate、dominant stage、calibration source、seed policy 写入 plan/runtime metadata。
- GB 级本地 calibration 已闭环，当前 profile raw formula vs allocated peak 都在目标内：
  - `matrix_bigamp_target_10gb / 16gb`
  - `matrix_agd_target_10gb / 16gb`
  - `spreading_bigamp_target_10gb / 16gb`
  - `tensor_serial_target_6gb / 10gb`
  - `tensor_parallel_target_6gb / 10gb`
- spreading 大尺寸 graph fallback 已改成 sparse unique edge sampling，避免 `randperm(N1*N2)` 的巨大临时 allocation。

### Runtime extensions

- `batch_summary` probe 是 runner-level `after_batch` active probe。
- `state_slice` 已成为 AGD 的真实 `after_step` active probe：
  - AGD 每步暴露 `AlgorithmStateView`。
  - executor 只记录轻量 JSON summary：factor shape/dtype/device/mean/norm、step_index、loss。
  - 不复制大 tensor，不修改 W/X，不改变训练数值。
- `tensor_state_slice` 和 `variance_slice` 仍是 `declared_only`；请求它们会 preflight error。
- warm/cold/adaptive restart 仍主要是 metadata/spec 映射；未迁移成真正独立 intervention executor。

## Active but limited

- Matrix algorithms 已原生返回 `AlgorithmResult`，但 metric compute 仍在 runner/metric adapter 层，不是 algorithm 内部产出。
- OOM 后同进程自动 retry 还未实现；runner 仍 checkpoint/flush/exit。
- `partition_invariant` seed policy 已打开 planner 层安全 rebatch 可能性，但 runner 默认不会自动改 batch 并继续跑。
- `spreading.chunk_size` 只记录 metadata，不做 auto tuning。
- `MetricSpec` 已接入 semantic/validation，但 metric compute 仍包住现有公式实现，不是每个 MetricSpec 都有独立 compute class。
- PlotQuery 已有硬查询和绘图，但没有交互式 result browser/UI。

## Not started / future work

- tensor serial/parallel 物理合并。
- tensor serial/parallel 数值 parity 长跑验证。
- sample/student folding 的真实执行接入：
  - runner 传 `sample_range/sample_offset`
  - algorithm 使用 global sample index
  - metrics 合并 sample batch
- 真正会改变训练轨迹的 intervention，例如 Metropolis-like kick。
- medium/gpu-heavy trial 的自动执行策略和 promotion CLI。

## Verification commands

常规修改后至少运行：

```bash
python -m pytest -q
mf validate src/matrix_factorization/config.yaml
mf trial validate matrix_bigamp_quick
mf trial run matrix_bigamp_quick
mf trial run scan_mixed_axes_quick
git diff --check
```

显存公式或 planner 修改后还需要重新跑 `docs/parallel_memory_contract.md` 中列出的 GB calibration profiles。
