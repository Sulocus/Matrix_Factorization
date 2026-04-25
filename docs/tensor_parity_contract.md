# Tensor Serial / Parallel Parity Contract

长期目标：`bigamp_tensor` 和 `bigamp_tensor_parallel` 应该是同一个 tensor-spreading 物理算法的 serial / parallel 实现，而不是两个物理定义不同的算法。

这份 contract 先记录合并前必须对齐的内容。当前不直接合并代码，也不改变现有数值行为。

## 必须对齐的项目

```text
teacher scale
  serial 和 parallel 必须使用同一套 teacher factor 尺度。

alpha normalization
  同一个 alpha 必须对应同一类观测密度定义。

graph / hypergraph definition
  serial hypergraph 和 parallel TensorSuperGraph 必须描述同一个统计对象。

F distribution
  rademacher / gaussian 的含义和归一化必须一致。

damping semantics
  damping=0 和 damping=1 的含义必须在两个实现中一致。

Onsager handling
  是否使用 Onsager correction，以及 prev_s 的含义必须一致。

Q_Y / Q_Y_observed semantics
  full tensor cosine、observed reconstruction quality、physical overlap 不能共用模糊名称。

result schema
  两者都必须返回同一种 AlgorithmResult，不允许一个走 placeholder W/X，另一个走私有 _batch_metrics。
```

## 当前已知不一致

- `bigamp_tensor` 当前更像 serial/reference 路径，已返回正式 metrics-only `AlgorithmResult`，但数值实现内部仍暂用 `_batch_metrics` 缓冲。
- `bigamp_tensor_parallel` 是当前 tensor 主运行路径，支持 `overlap_matrix` heatmap。
- 两者 alpha edge count 约定不完全一致。
- 两者当前都通过正式 `AlgorithmResult` 返回 metrics-only payload，但底层数值实现仍保留 legacy `_batch_metrics` 缓冲；后续 parity 阶段应继续把 tensor factors/artifacts 原生暴露出来。
- `Q_Y_mean` 在两条路径中的语义不应默认视为相同。

## 合并前验收条件

- 两者都有 `AlgorithmSpec`。
- 两者声明相同或明确可比较的 data requirements。
- 两者的 `MetricSpec` 能区分 full/observed tensor metrics。
- 小尺寸 parity test 至少验证输出 shape、metric key、result contract 一致。
- 物理数值 parity 需要单独本地 GPU 或长跑验证，不应由轻量测试承担。

## 机器可读 contract

`src/matrix_factorization/core/contracts.py` 中的 `get_tensor_parity_specs()` 是这份文档的机器可读版本。它列出：

- `teacher_scale`
- `alpha_normalization`
- `graph_definition`
- `f_distribution`
- `damping_semantics`
- `onsager_handling`
- `qy_semantics`
- `result_schema`

`get_tensor_parity_report()` 会从 `AlgorithmSpec` 自动生成当前 gap report。当前已知 contract-level gap 包括：

- serial 缺少 `tensor.full.Q_Y` 和 `tensor.physical_overlap_Y`。
- parallel 使用 `tensor_supergraph`，serial 使用 `tensor_hypergraph`。
- 两者虽然都已经返回正式 metrics-only `AlgorithmResult`，但 result contract 仍标记为 `legacy_tensor_metrics_only`，表示它们还没有统一暴露 tensor factors、variance state 和完整 artifact schema。
