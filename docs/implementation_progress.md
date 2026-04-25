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

## 本轮继续推进

- Flat-key 级 metric semantic 拆分：
  - `Q_W_mean` 与 `Q_W_prime_mean` 拆开。
  - `Q_X_mean` 与 `Q_X_prime_mean` 拆开。
  - `physical_overlap_W/X/Y` 拆开。
  - replica raw / prime 拆开。

## 尚未完成

- 真正的 seed partition invariant 改造。
- OOM retry 自动缩 batch。
- compile/dtype fallback 策略。
- spreading chunk size auto tuning。
- tensor serial/parallel 训练 loop 合并。
- tensor serial/parallel teacher scale、alpha graph、damping 语义统一。
- per-algorithm memory formula 与 probe 统一。
- runtime probe/intervention 真正接入算法内部 step state。

## 暂不自动改的高风险项

这些项可能改变数值路径或物理含义，只能先建测试、metadata、review queue：

- teacher/student scale。
- alpha normalization。
- damping update direction。
- Onsager / `prev_s`。
- seed 与 batch partition 的关系。
- BF16/TF32/torch.compile 自动切换。
- OOM 后自动重试并继续跑。
