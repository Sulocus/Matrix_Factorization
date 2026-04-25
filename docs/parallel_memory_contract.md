# Parallel / Memory Contract

这份文档记录当前并行、分批和显存策略。它不改变执行行为，只解释当前程序会怎样计划 batch、估计 memory、选择 dtype/compile 路径。

机器可读版本在 `src/matrix_factorization/core/contracts.py`：

- `ResourceSpec`：algorithm 的设备、dtype、compile、probe、empty-cache 能力声明。
- `BatchingSpec`：algorithm 的 alpha batching、sample batching、chunking、seed partition 风险声明。
- `MemoryModelSpec`：algorithm 的 memory estimator 入口、公式依据、主要 live tensor component、calibration/probe 状态。
- `ExperimentPlan.resource_plan`：`mf explain-config` / `mf validate --json` 中的静态 resource summary。
- `runtime_resource_plan`：run metadata 中的 runner-level execution plan。

## 当前策略

### AGD

```text
planner: runner.ParallelCoordinator
alpha batching: runner alpha batches
sample batching: samples inside algorithm tensor
resource estimator: runner.MemoryEstimator
dtype: float32 / cuda bf16 autocast, controlled by algorithm_params.use_bf16
dtype fallback: algorithm_params.dtype_fallback_policy
```

当前 contract 标记 `sample_range_honored=false`，因为 runner 的 sample-range plan 没有作为正式 algorithm 输入传入。

### Matrix BiGAMP

```text
planner: runner.ParallelCoordinator
alpha batching: runner alpha batches
sample batching: W/X tensor batch
compile: optional torch.compile
```

这是 matrix dense path。当前显存 metadata 只记录 runner-level plan，不驱动新的 batch 行为。

### BiGAMP Spreading

```text
planner layers:
  - runner.ParallelCoordinator
  - algorithm per-batch supergraph
alpha batching: runner alpha batches
sample batching: disjoint-union sample parallel
chunking: spreading.chunk_size edge streaming
dtype: float32 / bf16 storage, controlled by algorithm_params.use_bf16
dtype fallback: algorithm_params.dtype_fallback_policy
compile fallback: algorithm_params.compile_fallback_policy
seed sensitive: true
chunk policy: manual_config if spreading.chunk_size > 0, disabled_legacy_unchunked if 0
```

风险：per-batch graph creation、batch seed offset、chunking 都可能和随机路径或执行路径绑定，因此自动优化 batch size 前必须先做 seed partition 审查。

### Serial Tensor

```text
planner: algorithm loop
alpha batching: serial alpha loop
sample batching: serial sample loop
resource estimator: none
```

它更像 reference / serial path。当前不把它当成 parallel tensor 的纯慢速实现。

### Tensor Parallel

```text
planner layers:
  - runner.ParallelCoordinator
  - algorithm internal probe batches
alpha batching: probe-based internal alpha batches
sample batching: TensorSuperGraph sample parallel
probe: A=1 tensor supergraph probe
dtype: float32 / bf16 storage / tf32 matmul
compile: torch.compile default optional
seed sensitive: true
seed policy: legacy_tensor_parallel_batch_idx_seed
automatic rebatch allowed: false
estimation params: tensor_order and tensor_dims are passed explicitly to MemoryEstimator
```

风险：algorithm 内部会排序 alpha，并使用 `seed + batch_idx`。改变 batch partition 会改变 graph/F/student random stream。当前 `SeedPolicySpec` 明确标记 `partition_invariant=false`、`automatic_rebatch_allowed=false`；所以 OOM retry / 自动缩 batch 只能先做 preflight 或 metadata，不能静默改变执行分批后继续声称等价。

`ParallelCoordinator.replan_with_safety()` 当前会明确报错，而不是返回旧 plan 伪装成缩 batch。`MemoryGuard` 只负责 abort/checkpoint handoff：触发 critical memory 后由 runner 保存 checkpoint 并退出，用户再用 clean process resume。

## Metadata 写入位置

`mf explain-config` 会显示：

```text
resource / batching contract
```

`mf validate --json` 会包含：

```text
resource_spec
batching_spec
resource_plan
  memory_model
    estimator_entrypoint
    formula_basis
    tensor_components
    calibration_status
    probe_required
    sample_range_policy
    drives_execution
  seed_policy
    policy_key
    seed_inputs
    random_streams
    partition_invariant
    batch_partition_sensitive
    automatic_rebatch_allowed
```

真实 run 的 `metadata.json` 会包含：

```text
contract.runtime_resource_plan
  algorithm_key
  device
  gpu_model
  available_memory_gb
  total_estimated_memory_gb
  mode
  num_batches
  allocation
  config_effective
  seed_policy
  batches
  metadata_only
```

tensor serial/parallel 的 `AlgorithmResult.metadata` 还会包含：

```text
tensor_execution
  path
  device
  requested_use_bf16
  effective_use_bf16
  dtype_fallback_policy
  dtype_status
  storage_dtype
  requested_use_compile
  compile_fallback_policy
  effective_use_compile
  compiled_step_available
  compiled_super_step_available
  compile_status
  compile_attempts
  tf32_matmul_enabled
  tf32_cudnn_enabled
  metadata_only

internal_alpha_batch_plan
  planner
  alpha_values_input
  alpha_values_execution_order
  alpha_batches
  sort_policy
  probe_enabled
  probe_method
  probe_result_gb
  target_memory_gb
  max_alphas_per_batch
  seed_partition_sensitive
  metadata_only
```

这两块 metadata 只记录实际执行选择和内部分批；不做自动 retry，不自动改变 batch partition。

`algorithm_params.compile_fallback_policy` 只控制 `torch.compile` 初始化失败时的行为，目前接入 `bigamp_spreading` 和 `bigamp_tensor_parallel`：

- `allow`：默认值，保持旧行为；compile 失败会记录到 `compile_attempts`，然后继续 eager path。
- `error`：严格调试模式；compile 失败会立即抛错，避免用户以为 compile 已经生效。

这个策略不处理 OOM retry，也不自动切换 BF16/TF32。

`algorithm_params.dtype_fallback_policy` 只控制 BF16 请求不能满足时的行为，目前接入 `agd`、`bigamp_spreading` 和 `bigamp_tensor_parallel`：

- `allow`：默认值，保持旧行为；BF16 不可用时使用 FP32，并在 `dtype_status` 里记录 fallback。
- `error`：严格调试模式；如果用户请求 BF16 但设备不可用或不支持 BF16，初始化直接失败。

这个策略不自动打开 BF16，也不处理 TF32 或 OOM 后 dtype retry。

`bigamp_spreading` 的 matrix-factor `AlgorithmResult.metadata.execution_metadata` 还会记录：

```text
path
chunk_size
chunking_enabled
chunk_policy
requested_use_compile
effective_use_compile
compile_fallback_policy
compile_status
compile_attempts
requested_use_bf16
effective_use_bf16
dtype_fallback_policy
dtype_status
storage_dtype
alpha_values
dynamic_batches
seed_partition
metadata_only
```

这块 metadata 只记录 `spreading.chunk_size` 是否实际打开 chunked path；v1 不做 chunk size auto tuning。

## Metrics-only tensor path

`bigamp_tensor_parallel.train_batch_result()` 是主 runner 使用的正式路径。它现在直接走 metrics-only 执行，不再分配 legacy `W_all/X_all` placeholder。旧 `train_batch_alphas()` 仍保留 tuple API，并在执行完 metrics 后返回 placeholder tensor，以兼容旧调用方。

## 不在本阶段做的事

- 不根据 ResourceSpec 自动改 batch size。
- 不在 OOM 后自动切换 dtype；普通 BF16 不可用只按 `dtype_fallback_policy` 处理。
- 不在 OOM 后自动关闭 compile；普通 `torch.compile` 初始化失败只按 `compile_fallback_policy` 处理。
- 不在 OOM 后自动重试。
- 不改变 seed 与 batch partition 的关系。
- 不把 tensor serial/parallel 合并。

这些都进入 `docs/parallel_memory_review_queue.md`。
