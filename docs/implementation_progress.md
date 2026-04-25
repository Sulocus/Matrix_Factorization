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
- Pytest collection boundary：默认 `python -m pytest` 只收集 `tests/`，legacy/local-GPU/debug 测试模块显式 skip，当前全量默认测试为 `169 passed, 9 skipped`。
- Legacy pytest inventory：`get_legacy_pytest_inventory()` 和 `tests/test_source_inventory.py` 强制这些 legacy/local-GPU 测试必须登记并模块级 skip。
- Runtime probe report：runner-level `batch_summary` probe 现在写入轻量 `probe_reports` payload，记录 batch/alpha/metric/output metadata，不进入算法 step。
- Algorithm config trace：runner algorithm cache 已按 effective config signature 分区，run metadata 写入 `algorithm_config_trace`，防止同 key 不同参数复用旧 algorithm 实例。

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
  - Resource/Batching 与 tensor parity 的显存约束已加入测试。

## 尚未完成

- 真正的 seed partition invariant 改造。
- OOM retry 自动缩 batch。
- compile/dtype fallback 策略。
- spreading chunk size auto tuning。
- tensor serial/parallel 训练 loop 合并。
- tensor serial/parallel teacher scale、alpha graph、damping 语义统一。
- per-algorithm memory formula 与 probe 统一。
- runtime probe/intervention 真正接入算法内部 step state。
- source inventory 扩展到 scripts/debug/verification 与 trial 之外的所有探索文件。
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
