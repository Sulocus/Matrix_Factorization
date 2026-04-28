# Project Handoff 2026-04-27

这份文档给下一轮 Codex 对话接手使用。它记录本轮从旧项目重新接管到 hard-interface、trial、canonical scan、Projection Metric、智能并行、runtime probe 和 PlotQuery 的真实完成状态。

当前分支：`dev`。

## 0. 当前 checkpoint

最近已经 push 到 `origin/dev` 的阶段性提交：

- `e0e8b31 Complete projection metric contract fixtures`
- `256ce69 Harden AlgorithmResult active paths`
- `7733425 Calibrate scan-aware memory estimator`
- `8da8fb5 Wire first step-level runtime probe`
- `d78e43c Complete ResultCube plot query flow`

最近验证：

```bash
python -m pytest -q
# 295 passed, 9 skipped

mf validate src/matrix_factorization/config.yaml
# errors: none
# warnings: parsed_only / inactive-current-route / tensor route override

mf trial run scan_mixed_axes_quick
# 生成 ignored runs/trials/.../plots/qy_mixed_axes.png
```

如果新对话接手，应先运行：

```bash
git status --short --branch
python -m pytest -q
mf validate src/matrix_factorization/config.yaml
```

## 1. 总体设计现状

项目已经从“目录分层 + 软约定”升级到 contract-driven research framework：

```text
YAML
  -> load_yaml_config()
  -> ExperimentPlan
  -> ScanPlan
  -> ResourceExecutionPlan
  -> ExperimentRunner
  -> AlgorithmResult
  -> ResultCube / metrics.json / plots
```

核心规则：

- 新 YAML-facing 参数必须有 `ParameterSpec`。
- 新 algorithm/teacher/graph/metric/output/probe/analyzer/intervention/trial 必须有对应 spec。
- registry 是 spec-gated，缺 spec 会 import/test fail。
- active route 参数没有实际消费会 warning；strict mode 会 error。
- output/plot/probe/analyzer 的依赖缺失会 preflight 或 save 阶段 hard error。
- trial/debug run 默认只改 `trials/active/<key>/config.yaml`，不改正式默认 config。

## 2. 已完成内容

### Hard interface

位置：

- `src/matrix_factorization/core/contracts.py`
- `src/matrix_factorization/core/planning.py`
- `src/matrix_factorization/cli.py`

已完成：

- `ParameterSpec / AlgorithmSpec / MetricSpec / OutputSpec / InterventionSpec / ProbeSpec / AnalyzerSpec / TrialSpec / ResourceSpec / BatchingSpec / MemoryModelSpec / SeedPolicySpec`。
- `mf validate / mf explain-config / --json / --strict`。
- `parameter_chain`：YAML path -> effective value -> consumer -> current-route active/inactive。
- source inventory：新增源码、trial 文件、debug/experiment 文件未登记会测试失败。

### Projection metric v3

位置：

- `src/matrix_factorization/modules/metrics/contract_compute.py`
- `src/matrix_factorization/modules/metrics/tensor.py`
- `src/matrix_factorization/modules/metrics/spreading.py`
- `src/matrix_factorization/core/experiment/result.py`
- `docs/METRICS_GUIDE.md`

已完成：

- `Q_Y` = absolute projection，normalization 是 teacher norm squared，不 clip。
- `Q_W/Q_X` = matrix latent coordinate projection。
- `Q_N` = tensor latent node/spin/factor projection。
- `Q_W_COS_ROOT/Q_X_COS_ROOT` = Cos-root diagnostic。
- `MSE / Gen_Error / Q_Y_COS / physical_overlap_* / Q_W_prime / Q_X_prime` 不再是 formal metric。
- schema `<3` 的旧 `Q_Y_mean` 标记为 legacy cosine/proxy；schema `3` 的 `Q_Y_mean` 标记为 absolute projection。
- degenerate teacher norm policy 写入 metric schema/contract。

边界：

- 旧 result 仍能 load，但不能和 schema v3 的 `Q_Y_mean` 静默比较。
- internal loss/debug helper 可保留，但文档/metadata 必须标记 legacy/debug/internal。

### AlgorithmResult

位置：

- `src/matrix_factorization/core/contracts.py`
- `src/matrix_factorization/core/experiment/runner.py`
- `src/matrix_factorization/modules/algorithms/base.py`
- tensor algorithms under `src/matrix_factorization/modules/algorithms/bigamp/`

已完成：

- runner active path 优先消费 `AlgorithmResult.metrics_by_alpha`。
- tensor serial/parallel metrics-only route 不再保存 dummy zero `W/X`。
- active tensor route 缺 metric 时不从私有 `_batch_metrics` 补救。
- `agd / bigamp / bigamp_spreading` 已有原生 `train_batch_result()`，active path 直接返回 native matrix `AlgorithmResult`。
- legacy `train_batch_alphas()` 仍保留给旧脚本/兼容调用方；base legacy adapter 仍存在，但不应是 active matrix algorithms 的主路径。
- `metadata.contract.algorithm_result_batches` 会记录 batch-level result summary，包括 `result_source/result_kind/result_contract/available_outputs` 和轻量 execution metadata。

边界：

- Matrix algorithms 的 metric compute 仍由 runner/metric adapter 负责；原生 `AlgorithmResult` 目前主要声明真实 matrix factors 和 execution metadata。

### Research trial workflow

位置：

- `trials/`
- `src/matrix_factorization/core/trials.py`
- `src/matrix_factorization/cli.py`

已完成：

- `mf trial list/explain/validate/run`。
- quick trial 真实运行并写入 ignored `runs/trials/`。
- `mf trial run` 不刷新 `results/latest`。
- active quick trials:
  - `matrix_bigamp_quick`
  - `scan_alpha_quick`
  - `scan_steps_quick`
  - `scan_size_quick`
  - `scan_init_quick`
  - `scan_mixed_axes_quick`

边界：

- `medium/gpu_heavy` 可以登记和 validate/explain，但 v1 不自动 run。
- trial promotion lifecycle 仍主要靠 manifest/summary，不是完整 CLI。

### Canonical scan

位置：

- `src/matrix_factorization/core/scan_planning.py`
- `src/matrix_factorization/core/parallel/resource_execution.py`
- `src/matrix_factorization/core/experiment/runner.py`
- `docs/scan_system_contract.md`

已完成：

- 主入口是 `scan.axes`。
- 旧 `scan_mode / alpha_scan / steps_scan / nested_scan / hysteresis_scan` 不再是主链路支持对象。
- `alpha/max_steps/size/init/damping/onsager` 都是 axis。
- composite axis 支持 size、cold/warm start 等多参数联动。
- `ScanPlan` 展开 `ScanPoint`，并记录 coordinates、overrides、effective hash、group_id。
- canonical multi-axis result 写入 `ResultCube`。

边界：

- `steps` axis 当前可以运行，但 checkpoint reuse 语义仍需要后续单独设计。

### ResultCube / PlotQuery

位置：

- `src/matrix_factorization/core/experiment/result.py`
- `src/matrix_factorization/core/planning.py`
- `tests/test_plot_query.py`

已完成：

- `ResultCube.resolve_plot_query()` 是正式查询硬边界。
- `where / series_by / compare / x / y` 坐标引用错误会 hard error。
- metric 缺失会 hard error。
- 同一 series 选中重复 x 值会 hard error，避免悄悄把不同参数点混成一条曲线。
- `mf explain-config` 会输出 PlotQuery 预览：series 数、point 数、每条 series 的 point_id。
- `scan_mixed_axes_quick` 已验证能按 `series_by: [damping, onsager]` 生成图。

### Scan-aware memory / smart parallel

位置：

- `src/matrix_factorization/core/parallel/memory_estimator.py`
- `src/matrix_factorization/core/parallel/resource_execution.py`
- `src/matrix_factorization/core/memory_calibration.py`
- `docs/parallel_memory_contract.md`
- `docs/parallel_memory_review_queue.md`

已完成：

- planner 按 canonical scan 的 non-alpha group 分组。
- 当前真实 folding 只允许 `alpha`。
- `sample/student` folding 被 gate 住，直到 runner/algorithm 真正消费 `sample_range/sample_offset`。
- `size/init/damping/onsager/max_steps` 默认不跨 group folding。
- 每个 active algorithm 的 memory estimate 输出 stage breakdown、dominant stage、persistent/transient peak、calibration source、confidence。
- GB 级本地 calibration 已闭环，当前 profile raw formula vs allocated peak 均在目标误差内。
- spreading 大尺寸 fallback 不再用完整 `randperm(N1*N2)`，改成 sparse unique edge sampling。

已验证 profile：

- `matrix_bigamp_target_10gb / 16gb`
- `matrix_agd_target_10gb / 16gb`
- `spreading_bigamp_target_10gb / 16gb`
- `tensor_serial_target_6gb / 10gb`
- `tensor_parallel_target_6gb / 10gb`

边界：

- 这不是“所有算法所有参数组合都已经最优并行”。
- 换 GPU、dtype、compile、tensor order、internal batching 后需要重新 calibration。
- OOM 后同进程自动 retry 未实现。

### Runtime probe / intervention

位置：

- `src/matrix_factorization/modules/interventions/`
- `src/matrix_factorization/modules/algorithms/agd.py`
- `tests/test_contract_runtime_extensions.py`

已完成：

- `batch_summary` 是 runner-level `after_batch` active probe。
- `state_slice` 是 AGD `after_step` active probe：
  - 暴露 `AlgorithmStateView`。
  - 记录轻量 JSON summary。
  - 不复制大 tensor。
  - 不修改训练 state。
  - 开启/关闭 probe 的 `Q_Y_mean` fixture 一致。

边界：

- `tensor_state_slice` 和 `variance_slice` 仍是 `declared_only`。
- warm/cold/adaptive restart 仍是 spec/metadata/no-op executor 层，尚未完全迁移出 algorithm。
- Metropolis-like kick 未开始实现。

## 3. 还没完成的大方向

优先级建议：

1. 如果想继续偏工程、低物理风险，优先把 OutputPlan executor 从 `ExperimentResult.save()` 中拆出，让 output/plot/export 也有独立硬执行层。
2. 如果想继续 runtime extension，可扩展 step-level state view 到 dense BigAMP/spreading/tensor。
3. 如果目标转向物理，优先做 tensor serial/parallel parity review，不要直接合并。
4. 如果目标转向性能，优先做 sample/student folding 的真实 execution contract，而不是只改 estimator。
5. 如果目标转向研究产出体验，继续做 ResultCube browser / query CLI / compare-runs 工具。

具体 backlog：

- MetricSpec compute adapter 完全接管 runner metric 计算。
- OutputPlan executor 从 `ExperimentResult.save()` 中拆出。
- sample/student folding：
  - runner 传 `sample_range/sample_offset`
  - algorithm 用 global sample index
  - metric aggregator 合并多个 sample batch
- OOM auto retry：
  - 只允许 partition-invariant seed policy
  - 记录 parent plan、failed batch、new plan
  - 保持 checkpoint/flush 语义
- spreading chunk auto tuning。
- tensor serial/parallel parity 和可能合并。
- real intervention：adaptive restart migration、Metropolis-like kick。
- trial promotion CLI。

推荐给下一轮 agent 的工程分支 prompt 已写在：

- `docs/next_agent_prompt_output_plan_executor.md`

这个 prompt 的任务是：把 output/plot/export 执行从 `ExperimentResult.save()` 中拆到独立 OutputPlan executor，不改变算法和 metric 数值。

## 4. 不应提交的东西

不要提交：

- `runs/`
- `results/`
- `artifacts/`
- `src/matrix_factorization/Replica_results/`
- calibration raw timeline、checkpoint、`.pt`、完整 heatmap、大批量实验数据。

可以提交：

- contract/spec/docs/tests。
- 轻量 calibration summary 或 coefficient 文件，前提是没有 raw GPU artifact。

## 5. 推荐接手命令

```bash
git status --short --branch
python -m pytest -q
mf validate src/matrix_factorization/config.yaml
mf trial validate matrix_bigamp_quick
mf trial run matrix_bigamp_quick
mf trial run scan_mixed_axes_quick
git diff --check
```

如果改并行/显存公式，额外跑：

```bash
mf calibrate memory run matrix_bigamp_target_10gb
mf calibrate memory run matrix_bigamp_target_16gb
mf calibrate memory run matrix_agd_target_10gb
mf calibrate memory run matrix_agd_target_16gb
mf calibrate memory run spreading_bigamp_target_10gb
mf calibrate memory run spreading_bigamp_target_16gb
mf calibrate memory run tensor_serial_target_6gb
mf calibrate memory run tensor_serial_target_10gb
mf calibrate memory run tensor_parallel_target_6gb
mf calibrate memory run tensor_parallel_target_10gb
```
