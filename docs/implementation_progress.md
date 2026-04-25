# Implementation Progress

这份文档记录硬接口化之后的连续施工状态，避免把阶段性 commit/push 误认为整个长期任务完成。

## 已完成并 push

- Hard interface baseline：parameter / algorithm / metric / output / intervention / probe / analyzer / source inventory contract。
- Trial workflow v1：`mf trial list/explain/validate/run`，quick trial 真实运行，输出隔离到 ignored workspace。
- Metric semantic class v1：`Q_Y_mean` 等旧 flat key 已能通过 algorithm context 映射到 canonical semantic class。
- Result schema metadata：`metrics.json` 写入 `metric_schema`，保留旧 flat keys。
- Output semantic metadata：custom plots 和 heatmap artifact 写入 semantic metadata。
- Tensor parity contract：记录 teacher scale、alpha normalization、seed partition、dtype/compile、batching 等风险。
- Resource/Batching contract：`ResourceSpec`、`BatchingSpec`、`mf explain-config` resource summary、run metadata `runtime_resource_plan`。
- Parallel memory docs：记录当前并行/显存 contract 和高风险 review queue。
- Tensor execution metadata：tensor serial/parallel result metadata 记录 dtype/compile、internal alpha batch plan、probe 状态。
- Tensor parallel metrics-only 主路径：`train_batch_result()` 不再分配 legacy placeholder `W_all/X_all`，旧 tuple API 仍保留。
- Parallel memory contract tests：`tests/test_parallel_memory_contract.py` 强制检查 metadata-only、seed partition review、tensor batch metadata schema。
- Memory model contract：`MemoryModelSpec` 记录每个 algorithm 的 estimator 入口、公式依据、component、probe/calibration 状态；进入 `ExperimentPlan.resource_plan.memory_model`。
- Seed policy contract：`SeedPolicySpec` 记录每个 algorithm 当前随机流、是否 partition invariant、是否允许自动重分批；进入 `mf explain-config`、`mf validate --json` 和 run metadata。
- Tensor memory estimator 参数链路：`EstimationParams` 现在显式携带 `tensor_order/tensor_dims`，tensor memory estimator 不再隐式默认三阶张量。
- Pytest collection boundary：默认 `python -m pytest` 只收集 `tests/`，legacy/local-GPU/debug 测试模块显式 skip，当前全量默认测试为 `170 passed, 9 skipped`。
- Legacy pytest inventory：`get_legacy_pytest_inventory()` 和 `tests/test_source_inventory.py` 强制这些 legacy/local-GPU 测试必须登记并模块级 skip。
- Source inventory expansion：根目录表面文件、`experiments/`、`scripts/analysis|debug|experiments|maintenance|verification/`、`tests/debug|verification/`、`trials/`、config 入口、teacher/graph/metric/output/runtime-extension 源文件都已进入机器清点测试。
- Runtime probe report：runner-level `batch_summary` probe 现在写入轻量 `probe_reports` payload，记录 batch/alpha/metric/output metadata，不进入算法 step。
- Probe wiring hardening：未接入实际 runtime hook 的 `state_slice/tensor_state_slice/variance_slice` 标记为 `declared_only`，YAML 请求会 preflight error，而不是静默无产物。
- Algorithm config trace：runner algorithm cache 已按 effective config signature 分区，run metadata 写入 `algorithm_config_trace`，防止同 key 不同参数复用旧 algorithm 实例。
- Tensor parallel dead-path cleanup：删除 `bigamp_tensor_parallel` 中 `_compute_alpha_batches()` 返回后的不可达 legacy memory-estimate 残片；当前显存估计入口以 `MemoryModelSpec`、runner estimator 和 tensor probe metadata 为准。
- Runtime batch timing metadata：runner 的 `BATCH_END` 事件记录真实 elapsed duration，不再写固定 `0.0` 占位值。
- OOM replan gate：`ParallelCoordinator.replan_with_safety()` 不再返回当前 plan 伪装成缩 batch，而是根据 `SeedPolicySpec` 明确拒绝未实现/不安全的自动重分批；`MemoryGuard` 文案改为 abort/checkpoint handoff。
- Compile status metadata：tensor parallel 的 `tensor_execution` 区分 `requested_use_compile` 和实际 super-step compile 是否生效，并记录 `compile_status/compile_attempts`。
- Spreading chunk metadata：`bigamp_spreading` 的 `AlgorithmResult.metadata.execution_metadata` 记录 `chunk_size/chunk_policy/dynamic_batches`，说明当前是手动 chunk 配置，不做 auto tuning。

## 本轮继续推进

- Flat-key 级 metric semantic 拆分（已完成并 push）：
  - `Q_W_mean` 与 `Q_W_prime_mean` 拆开。
  - `Q_X_mean` 与 `Q_X_prime_mean` 拆开。
  - `physical_overlap_W/X/Y` 拆开。
  - replica raw / prime 拆开。
- 并行/显存 hardening：
  - execution metadata 已接入 tensor result。
  - internal alpha batch plan 已接入 tensor result。
  - metrics-only tensor path 已避免无用 placeholder factor 分配。
  - tensor parallel 不再保留不可达的旧 `_estimate_batch_memory` 残片。
  - batch end progress event 已记录真实 elapsed duration。
  - seed policy 已机器可读化；当前 partition-sensitive 算法禁止把自动重分批当成等价行为。
  - OOM 自动 replan 已 hard-gate；当前策略是 checkpoint/resume，不自动改变 batch partition。
  - tensor parallel compile fallback 已进入 metadata；`effective_use_compile` 不再把 “super step fallback eager” 误写成生效。
  - spreading chunk_size 已进入 algorithm result metadata；当前仍是手动配置而不是自动调参。
  - Resource/Batching 与 tensor parity 的显存约束已加入测试。

## 尚未完成

- 真正的 seed partition invariant 改造。
- OOM retry 自动缩 batch 的真实实现（当前已 hard-gate，不会伪装成已实现）。
- compile/dtype fallback 的用户策略选择（当前已记录 metadata，但未新增“失败即报错/允许 fallback”的可配策略）。
- spreading chunk size auto tuning 的真实实现（当前已记录执行 metadata，但不自动调参）。
- tensor serial/parallel 训练 loop 合并。
- tensor serial/parallel teacher scale、alpha graph、damping 语义统一。
- per-algorithm memory formula 的数值校准与真实 probe 对照。
- runtime probe/intervention 真正接入算法内部 step state（当前未接线 probe 已 hard error）。
- metric/review queue 的最终命名选择需要人工确认。

## 暂不自动改的高风险项

这些项可能改变数值路径或物理含义，只能先建测试、metadata、review queue：

- teacher/student scale。
- alpha normalization。
- damping update direction。
- Onsager / `prev_s`。
- seed 与 batch partition 的关系。
- BF16/TF32/torch.compile 自动切换。
- OOM 后自动重试并继续跑。
