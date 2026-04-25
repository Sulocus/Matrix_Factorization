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
- Pytest collection boundary：默认 `python -m pytest` 只收集 `tests/`，legacy/local-GPU/debug 测试模块显式 skip，当前全量默认测试为 `181 passed, 9 skipped`。
- Legacy pytest inventory：`get_legacy_pytest_inventory()` 和 `tests/test_source_inventory.py` 强制这些 legacy/local-GPU 测试必须登记并模块级 skip。
- Source inventory expansion：根目录表面文件、`experiments/`、`scripts/analysis|debug|experiments|maintenance|verification/`、`tests/debug|verification/`、`trials/`、config 入口、teacher/graph/metric/output/runtime-extension 源文件都已进入机器清点测试。
- Runtime probe report：runner-level `batch_summary` probe 现在写入轻量 `probe_reports` payload，记录 batch/alpha/metric/output metadata，不进入算法 step。
- Runtime probe metadata keys：`batch_summary` 还会记录 `AlgorithmResult.metadata`、`execution_metadata`、`tensor_execution` 的 key 列表，只暴露轻量索引，不复制 factor/tensor payload。
- Metric schema coverage gate：active algorithm 声明的每个 legacy flat metric key 都必须被 `get_metric_schema()` 索引；新增 metric 如果没有 semantic class / flat-key 映射，contract 测试会失败。
- Result-save metric gate：`ExperimentResult.save()` 现在会重新校验每个 scan point 的 metric payload；手工塞入未声明 metric/artifact 也会在落盘前失败。
- Probe wiring hardening：未接入实际 runtime hook 的 `state_slice/tensor_state_slice/variance_slice` 标记为 `declared_only`，YAML 请求会 preflight error，而不是静默无产物。
- Algorithm config trace：runner algorithm cache 已按 effective config signature 分区，run metadata 写入 `algorithm_config_trace`，防止同 key 不同参数复用旧 algorithm 实例。
- Tensor parallel dead-path cleanup：删除 `bigamp_tensor_parallel` 中 `_compute_alpha_batches()` 返回后的不可达 legacy memory-estimate 残片；当前显存估计入口以 `MemoryModelSpec`、runner estimator 和 tensor probe metadata 为准。
- Runtime batch timing metadata：runner 的 `BATCH_END` 事件记录真实 elapsed duration，不再写固定 `0.0` 占位值。
- OOM replan gate：`ParallelCoordinator.replan_with_safety()` 不再返回当前 plan 伪装成缩 batch，而是根据 `SeedPolicySpec` 明确拒绝未实现/不安全的自动重分批；`MemoryGuard` 文案改为 abort/checkpoint handoff。
- Compile status metadata：tensor parallel 的 `tensor_execution` 区分 `requested_use_compile` 和实际 super-step compile 是否生效，并记录 `compile_status/compile_attempts`。
- Spreading chunk metadata：`bigamp_spreading` 的 `AlgorithmResult.metadata.execution_metadata` 记录 `chunk_size/chunk_policy/dynamic_batches`，说明当前是手动 chunk 配置，不做 auto tuning。
- Memory breakdown reporting：`MemoryEstimator.estimate()` 现在会返回按组件拆分的 `breakdown`，覆盖 AGD、dense BiGAMP、spreading BiGAMP 和 tensor spreading；每个 runner batch 的 `runtime_resource_plan.batches[*].memory_breakdown` 会保存这份 metadata。这只暴露已有估计公式，不改变训练或 batching 行为。
- Tensor parity report in plan：tensor serial/parallel gap report 已接入 `ExperimentPlan`、`mf explain-config` 和 `mf validate --json`，tensor 配置会直接显示 shared/missing metrics 与 result contract 差异。
- Intervention contract detail：`ExperimentPlan.to_dict()` 现在输出 `intervention_contracts`，记录 trigger、requires_state、modifies_state 和 physical_sensitive；后续新增 intervention 不会只在名字层面接入。
- Trial regression：`mf trial validate/run matrix_bigamp_quick` 已在 result-save hard gate 后重新验证，quick trial 输出仍隔离在 ignored `runs/trials/`，并写出 metric schema 与 batch memory breakdown。
- Compile fallback policy：新增 `algorithm_params.compile_fallback_policy`，默认 `allow` 保持旧 eager fallback；设为 `error` 时 dense BigAMP/spreading/tensor parallel 的 `torch.compile` 失败会在初始化阶段报错。该字段进入 `ParameterSpec`、parameter chain、resource plan、algorithm config trace 和 execution metadata。
- DType fallback policy：新增 `algorithm_params.dtype_fallback_policy`，默认 `allow` 保持 BF16 不可用时回到 FP32 的旧行为；设为 `error` 时 AGD/spreading/tensor parallel 请求 BF16 但不可用会初始化失败。该字段进入 `ParameterSpec`、parameter chain、resource plan、algorithm config trace 和 execution metadata。
- TF32 policy：新增 `algorithm_params.use_tf32`，默认 `true` 保持旧的全局 TF32 开启行为；AGD、dense BigAMP、spreading 和 tensor parallel 初始化时会按该字段设置 torch backend，并写入 execution metadata。
- Resource parameter route narrowing：`algorithm_params.use_compile/use_bf16` 的 `ParameterSpec` 和 active-route 判断已收窄到真实消费它们的 algorithm；例如 dense `bigamp` 下写 `use_bf16` 会显示 inactive，AGD 下写 `use_compile` 会显示 inactive。
- Seed partition policy v1：`algorithm_params.seed_partition_policy` 默认 `legacy` 保持旧随机流；`partition_invariant` 已覆盖 tensor parallel、spreading、AGD 和 dense BigAMP。tensor parallel 按 alpha/sample/dimension/role 分流 graph/F/student initialization；spreading 按 alpha/sample/role 分流 student initialization，并去掉 opt-in 路径中的 internal `batch_idx` seed 偏移；AGD/dense BigAMP 按 alpha/sample/role 分流 student initialization。resource plan 与 runtime metadata 会标记 `automatic_rebatch_allowed=true`。
- Runtime seed policy metadata：run metadata 的 `runtime_resource_plan.seed_policy` 现在使用 effective seed policy；显式 `partition_invariant` 不会出现 plan 与 run metadata 一个说 invariant、一个说 legacy 的断裂。
- Parameter value validation：`ExperimentPlan` 现在会按 `ParameterSpec.type` 检查 raw YAML 中的 enum/bool/int/float/list 基础类型；非法枚举值会在 validate 阶段报 `INVALID_PARAMETER_VALUE`，不再等到 algorithm 初始化。
- Metric naming decisions：新增 `docs/metric_naming_decisions.md`，把所有 `MetricSemanticClass.canonical_key` 按 equivalent class 列成命名决策表。contract test 会检查 `metrics_semantics.md` 和命名表覆盖所有 canonical metric class，避免后续新增 metric 只改代码不改语义文档。
- Result schema map：`docs/result_schema_contract.md` 新增 run directory/result/latest 的层级地图，明确 `config.json`、`metadata.json`、`metrics.json`、`output_contract.json`、`events.jsonl`、`manifest.json`、`artifacts/results.pt`、`plots/` 和 `results/latest` 的角色。测试会检查文档覆盖 canonical result files。
- Algorithm integration map：新增 `docs/algorithm_integration_contract.md`，把 `agd/bigamp/bigamp_spreading/bigamp_tensor/bigamp_tensor_parallel/agd_tensor/agd_spreading/combined` 的主链路状态、result contract 和去重策略列成硬文档。测试会检查所有 `AlgorithmSpec.key` 都被文档覆盖。
- Tensor parity seed status：tensor parity contract 现在区分 parallel 默认 legacy batch-sensitive seed 与 opt-in `partition_invariant`，避免把“parallel 内部分批稳定”误读成“serial/parallel 物理 parity 已完成”。

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
  - tensor parallel 新增 opt-in `seed_partition_policy=partition_invariant`；默认 legacy 不变，开启后 tensor supergraph index 前缀和 student init 随机流不依赖 internal alpha batch 的 C_max。
  - AGD 与 dense BigAMP 新增 opt-in `seed_partition_policy=partition_invariant`；默认 legacy 不变，开启后同一个 alpha/sample 的 student initialization 不依赖 alpha batch 分组。
  - spreading 新增 opt-in `seed_partition_policy=partition_invariant`；默认 legacy 不变，开启后 graph/F 使用稳定 base seed，cold/warm start 初始化按 alpha/sample/role 分流。`adaptive_restart=true` 暂时会 preflight error，因为 restart noise 还没纳入随机流 contract。
  - OOM 自动 replan 已 hard-gate；当前策略是 checkpoint/resume，不自动改变 batch partition。
  - tensor parallel compile fallback 已进入 metadata；`effective_use_compile` 不再把 “super step fallback eager” 误写成生效。
  - dense `bigamp` compile fallback 已进入 metadata；compile 失败时不再只靠 console print 暴露。
  - `bigamp_spreading` compile fallback 已进入 metadata；`requested_use_compile/effective_use_compile` 不再混用同一个字段。
  - spreading chunk_size 已进入 algorithm result metadata；当前仍是手动配置而不是自动调参。
  - `agd` 已消费 `algorithm_params.use_bf16=false`，不会再只根据 CUDA device 自动打开 autocast。
  - `bigamp_spreading` 已消费 `algorithm_params.use_bf16=false`，不会再在用户显式关闭 BF16 时根据硬件自动打开。
  - `algorithm_params.use_tf32` 已接入 AGD、dense BigAMP、spreading 和 tensor parallel，默认保持旧行为，显式关闭会写入 metadata。
  - Resource/Batching 与 tensor parity 的显存约束已加入测试。

## 尚未完成

- spreading adaptive restart 的 partition-invariant restart noise 改造（普通 spreading、tensor parallel、dense BigAMP、AGD 已有 opt-in v1，默认仍是 legacy）。
- OOM retry 自动缩 batch 的真实实现（当前已 hard-gate，不会伪装成已实现）。
- TF32 OOM fallback 的用户策略选择（`use_tf32` 已可显式控制，但 OOM 后不自动切换）。
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
- BF16/TF32/torch.compile OOM 后自动切换。
- OOM 后自动重试并继续跑。
