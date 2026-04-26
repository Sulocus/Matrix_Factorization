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

initialization semantics
  serial 和 parallel 的初始化方式必须声明清楚。parallel 当前有 random / spectral / warm_start，serial 更像内部随机初始化。

seed partition
  batch 切分、alpha 排序、sample 顺序不应静默改变随机图、F 或 student init。当前这是高风险项。

dtype / compile semantics
  TF32、BF16、torch.compile 是实际执行路径的一部分，必须进入 metadata。

batching semantics
  serial 的 alpha/sample loop 与 parallel 的 TensorSuperGraph + probe batching 不是单纯的工程替换。
```

## 当前已知不一致

- `bigamp_tensor` 当前更像 serial/reference 路径，已返回正式 metrics-only `AlgorithmResult`，但数值实现内部仍暂用 `_batch_metrics` 缓冲。
- `bigamp_tensor_parallel` 是当前 tensor 主运行路径，支持 `overlap_matrix` heatmap。
- 两者 alpha edge count 约定不完全一致。
- 两者当前都通过正式 `AlgorithmResult` 返回 metrics-only payload，但底层数值实现仍保留 legacy `_batch_metrics` 缓冲；后续 parity 阶段应继续把 tensor factors/artifacts 原生暴露出来。
- `Q_Y_mean` 在两条路径中的语义不应默认视为相同。
- `seed` 路径默认仍对 batch partition 敏感：parallel legacy 路径内部分批和 alpha 排序可能改变 graph/F/student 随机流。`algorithm_params.seed_partition_policy=partition_invariant` 已提供 opt-in v1，但这只解决 parallel 内部分批稳定性，不证明 serial/parallel 数值 parity。
- serial 与 parallel damping 更新式存在方向一致性风险，必须单独审查。
- teacher factor scale、alpha edge count、graph sharing scope、initialization mode 都需要进入 parity checklist。

## 合并前验收条件

- 两者都有 `AlgorithmSpec`。
- 两者声明相同或明确可比较的 data requirements。
- 两者的 `MetricSpec` 能区分 full/observed tensor metrics。
- 两者的 result metadata 记录 graph kind、tensor order、dims、batching source、alpha execution order。
- 小尺寸 parity test 至少验证输出 shape、metric key、result contract 一致。
- 物理数值 parity 需要单独本地 GPU 或长跑验证，不应由普通 contract 测试承担。

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
- `initialization_semantics`
- `seed_partition`
- `dtype_compile_semantics`
- `batching_semantics`

`get_tensor_parity_report()` 会从 `AlgorithmSpec` 自动生成当前 gap report。当前已知 contract-level gap 包括：

- serial 缺少 `tensor.full.Q_Y` 和 `tensor.physical_overlap_Y`。
- parallel 使用 `tensor_supergraph`，serial 使用 `tensor_hypergraph`。
- 两者虽然都已经返回正式 metrics-only `AlgorithmResult`，但 result contract 仍标记为 `legacy_tensor_metrics_only`，表示它们还没有统一暴露 tensor factors、variance state 和完整 artifact schema。
- parallel 默认 seed policy 仍是 legacy batch-sensitive；`partition_invariant` 是 opt-in 执行策略，不代表 serial/parallel 随机流已经统一。
- `tensor_contract.py` 只抽取 shared dims/result metadata/metric payload packing；它不改变 serial 或 parallel 的训练 loop。
