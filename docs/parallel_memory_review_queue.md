# Parallel / Memory Review Queue

这份文档记录并行与显存优化中不能直接自动修改的高风险项。它们可能改变随机样本、数值路径、dtype 行为或物理解释，因此需要单独审查。

## Seed Partition Invariant

当前风险：

- tensor serial 使用 `seed + s * 1000 + int(alpha * 100)`。
- tensor parallel 内部分批使用 `seed + batch_idx`，并会排序 alpha。
- spreading path 也存在 per-batch graph creation 和 seed offset。
- 当前 `SeedPolicySpec` 已把这些策略机器可读化；`bigamp_tensor_parallel` 标记为 `partition_invariant=false`、`automatic_rebatch_allowed=false`。
- tensor parallel、spreading、dense BigAMP 和 AGD 已有 opt-in `algorithm_params.seed_partition_policy=partition_invariant`；默认 legacy 不变。
- spreading 的 opt-in policy 已升级到 v2，包含 adaptive restart noise 的 alpha/sample/step/role 分流。

需要确认：

- spreading adaptive restart noise 已进入 opt-in `partition_invariant` seed policy：`seed = hash(base_seed, alpha, sample, step, role)`。后续只需用本地实验确认该路径的物理结果是否满足预期。
- tensor parallel v1 是否需要本地 GPU 长跑确认曲线是否和 legacy 可比。

未确认前禁止：

- 对未启用 partition-invariant policy 的路径，自动缩 batch 后继续跑并声称结果等价。
- 合并 serial/parallel 随机流。

## OOM Retry / Auto Shrink Batch

当前风险：

- OOM 后缩小 batch 会改变 tensor parallel 的 `batch_idx`，进而改变 seed。
- compile cache、CUDA allocator cache、BF16/TF32 状态也会影响实际执行路径。

需要确认：

- OOM retry 是否只允许在 seed partition invariant 完成后启用。
- retry metadata 是否必须记录原 batch plan、失败原因和新 batch plan。planner 层已经记录 `plan_id`、原始 `EstimationParams` 和 replan provenance；runner 仍未在同一进程自动 retry。

## Compile / DType Fallback

当前风险：

- `use_compile`、BF16 storage、TF32 matmul 不是纯性能参数，可能带来数值差异。
- tensor parallel 的实际 dtype/compile 状态已进入 `tensor_execution` metadata；spreading 的 dtype/compile 状态已进入 `execution_metadata`。
- `algorithm_params.use_bf16=false` 已经会阻止 AGD/spreading/tensor parallel 自动启用 BF16；这是参数生效链路修正，不是 fallback 策略。
- `algorithm_params.use_tf32` 已经可以显式控制 AGD、dense BigAMP、spreading 和 tensor parallel 的 torch TF32 backend；默认 `true` 保持旧行为。
- `algorithm_params.compile_fallback_policy` 已经把 dense BigAMP/spreading/tensor parallel 的普通 `torch.compile` 初始化失败分成 `allow/error` 两种策略；默认 `allow` 保持旧 eager fallback，`error` 用于严格调试。
- `algorithm_params.dtype_fallback_policy` 已经把 BF16 不可用分成 `allow/error` 两种策略；默认 `allow` 保持旧 FP32 fallback，`error` 用于严格调试。

需要确认：

- 是否允许 OOM 时自动关闭 compile。
- 是否允许 OOM 时自动切 BF16 或 FP32。
- plot/result metadata 中是否应把 dtype/compile/TF32 作为 physical-sensitive execution parameter。

## Chunk Size Auto Tuning

当前风险：

- spreading `chunk_size` 影响内存峰值和计算路径。
- 过小 chunk 可能改变性能甚至数值误差积累顺序。

需要确认：

- 是否允许自动调 `spreading.chunk_size`。
- 自动调参是否必须写入 effective parameters 和 metadata。

## Tensor Serial / Parallel Memory Planner Merge

当前风险：

- serial 和 parallel 不是同一物理路径的简单工程实现。
- teacher scale、alpha graph definition、damping direction、init mode、seed partition 仍存在 parity gap。

需要确认：

- 合并 memory planner 前是否先要求 tensor parity checklist 全部通过。
- serial 是否保留为 reference implementation。

## Per-Algorithm Memory Formula

当前风险：

- matrix、spreading、tensor 的 live tensors 不同。
- tensor order、C formula、F dtype、indices dtype、compile cache 都影响实际峰值。

需要确认：

- 是否要为每个 active algorithm 建独立 MemoryModel。
- 公式估计和 probe 估计冲突时以哪个为准。

## 当前允许继续做的低风险工作

- 记录 ResourceSpec / BatchingSpec。
- 在 `mf explain-config` 显示 resource summary。
- 在 result metadata 中保存 runner-level batch plan。
- 在 quick trial 中验证 metadata 是否生成。
- 文档化高风险点。
