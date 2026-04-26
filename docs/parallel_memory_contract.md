# Parallel / Memory Contract

这份文档记录当前并行、分批和显存策略。它不改变执行行为，只解释当前程序会怎样计划 batch、估计 memory、选择 dtype/compile 路径。

机器可读版本在 `src/matrix_factorization/core/contracts.py`：

- `ResourceSpec`：algorithm 的设备、dtype、compile、probe、empty-cache 能力声明。
- `BatchingSpec`：algorithm 的 alpha batching、sample batching、chunking、seed partition 风险声明。
- `MemoryModelSpec`：algorithm 的 memory estimator 入口、公式依据、主要 live tensor component、calibration/probe 状态。
- `ExperimentPlan.resource_plan`：`mf explain-config` / `mf validate --json` 中的静态 resource summary。
- `runtime_resource_plan`：run metadata 中的 runner-level execution plan。
- `ScanPlan`：把 `alpha/steps/nested/hysteresis` 展开为统一 scan point 和 plot grouping。
- `ResourceExecutionPlan` / `WorkItem`：把 runner batch 映射到显式 work items，记录 alpha/sample/scan-axis 折叠情况。

## 当前策略

### AGD

```text
planner: runner.ParallelCoordinator
alpha batching: runner alpha batches
sample batching: samples inside algorithm tensor
resource estimator: runner.MemoryEstimator
dtype: float32 / cuda bf16 autocast, controlled by algorithm_params.use_bf16
dtype fallback: algorithm_params.dtype_fallback_policy
tf32: controlled by algorithm_params.use_tf32
```

当前 contract 标记 `sample_range_honored=false`，因为 runner 的 sample-range plan 没有作为正式 algorithm 输入传入。

### Matrix BiGAMP

```text
planner: runner.ParallelCoordinator
alpha batching: runner alpha batches
sample batching: W/X tensor batch
compile: optional torch.compile
compile fallback: algorithm_params.compile_fallback_policy
tf32: controlled by algorithm_params.use_tf32
```

这是 matrix dense path。当前 `ResourceExecutionPlan` 会把 runner alpha batch 映射成 `WorkItem`；sample_range 仍未作为正式 algorithm 输入传入。

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
tf32: controlled by algorithm_params.use_tf32
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
tf32: controlled by algorithm_params.use_tf32
compile: torch.compile default optional
seed sensitive: true
seed policy: legacy_tensor_parallel_batch_idx_seed
automatic rebatch allowed: false
estimation params: tensor_order and tensor_dims are passed explicitly to MemoryEstimator
```

风险：algorithm 内部会排序 alpha，并使用 `seed + batch_idx`。改变 batch partition 会改变 graph/F/student random stream。当前 `SeedPolicySpec` 明确标记 `partition_invariant=false`、`automatic_rebatch_allowed=false`；所以 OOM retry / 自动缩 batch 只能先做 preflight 或 metadata，不能静默改变执行分批后继续声称等价。

如果显式设置：

```yaml
algorithm_params:
  seed_partition_policy: partition_invariant
```

tensor parallel 会启用 opt-in v1 seed policy：

- tensor supergraph indices 按 `(base_seed, dimension, sample)` 分流，单个 alpha 的前缀不依赖同 batch 的 `C_max`。
- student initialization 按 `(base_seed, alpha, sample, dimension, role)` 分流。
- internal batch seed 不再使用 `seed + batch_idx`。
- `resource_plan.seed_policy` 会标记 `partition_invariant=true`、`batch_partition_sensitive=false`、`automatic_rebatch_allowed=true`。
- run metadata 的 `runtime_resource_plan.seed_policy` 也会记录同一套 effective policy。

spreading、AGD 与 dense BigAMP 也可以显式设置同一个字段。AGD/dense BigAMP 的 opt-in v1 只覆盖 student initialization：同一个 `(base_seed, alpha, sample, role)` 会得到同一个初始 factor，不依赖当前 alpha batch 怎么分组。spreading 的 opt-in v2 会保留 graph/F 的 base seed，不再混入 internal `batch_idx`，让 cold/warm start 初始化按 alpha/sample/role 分流，并让 adaptive restart noise 按 alpha/sample/step/role 分流。默认 `legacy` 不变。

`ParallelCoordinator.replan_with_safety()` 当前只在 effective seed policy 允许 rebatch 且 `ExecutionPlan` 带有原始 `EstimationParams` 快照时，生成一个更保守的新 plan；否则明确报错，而不是返回旧 plan 伪装成缩 batch。`MemoryGuard` 仍只负责 abort/checkpoint handoff：触发 critical memory 后由 runner 保存 checkpoint、等待 checkpoint flush 完成并退出，用户再用 clean process resume。

effective seed policy 现在是 replan safety 的唯一来源：

```text
get_effective_seed_policy_summary(algorithm_key, requested_policy)
  -> policy_key
  -> partition_invariant
  -> batch_partition_sensitive
  -> automatic_rebatch_allowed
```

`ExecutionPlan` 会记录：

```text
plan_id
parent_plan_id
replan_attempt
seed_partition_policy
replan_policy_key
automatic_rebatch_allowed
replan_implemented=false
estimation_params
replan_provenance
```

这几个字段说明当前随机流 contract 和 planner 是否允许 rebatch。当前真实行为是：

- `automatic_rebatch_allowed=false`：`replan_with_safety()` 抛 `RuntimeError`，说明当前 seed policy 不允许静默重分批。
- `automatic_rebatch_allowed=true` 且 plan 有 `estimation_params`：`replan_with_safety()` 可生成新的更保守 `ExecutionPlan`，并写入 `parent_plan_id`、`replan_attempt` 和 `replan_provenance`。

也就是说，`partition_invariant` 现在打开的是 planner 层安全 rebatch；runner 的 OOM 路径仍然 checkpoint/exit，不会在同一进程里自动 retry。

checkpoint 语义：

- `CheckpointManager.save()` 仍是异步排队，避免正常 batch 间阻塞。
- `CheckpointManager.flush()` 会等待所有已排队写入，并把后台写入异常重新抛到主线程。
- OOM abort 退出前 runner 会调用 `flush()`；成功完成并删除 checkpoint 前也会先 `flush()`，避免后台 save 覆盖 cleanup。
- cooperative `MemoryAbortException` 和直接抛出的 `torch.cuda.OutOfMemoryError` 都走同一条 checkpoint/flush/exit 路径。
- resume 时如果一个 planned batch 里只有部分 alpha 已完成，runner 只会把未完成的 alpha 交给 algorithm；checkpoint 清理以 `completed_alphas` 是否覆盖全 scan 为准。

## Metadata 写入位置

`mf explain-config` 会显示：

```text
resource / batching contract
```

`mf validate --json` 会包含：

```text
scan_plan
  scan_kind
  axes
  points
  grouping
  execution_constraints
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
  replan_safety
    automatic_rebatch_allowed
    replan_implemented
    notes
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
  replan_safety
    plan_id
    parent_plan_id
    replan_attempt
    seed_partition_policy
    replan_policy_key
    automatic_rebatch_allowed
    replan_implemented
    estimation_params
    replan_provenance
	  batches
	    batch_axes
	    calibration_source
	    work_items
	  resource_execution_plan
	    batches
	    plot_grouping
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
  requested_use_tf32
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

这两块 metadata 记录实际执行选择和内部分批；runner OOM 路径仍默认 checkpoint/exit，不在 seed-sensitive 路径里自动改变 batch partition。

## 本地 GPU 校准

`mf calibrate memory` 是本地 GPU 显存校准入口：

```bash
mf calibrate memory list
mf calibrate memory explain matrix_bigamp_target_10gb
mf calibrate memory run matrix_bigamp_target_10gb
```

当前内置两类 profiles：

- `matrix_bigamp_small`
- `matrix_agd_small`
- `spreading_bigamp_small`
- `tensor_parallel_small`
- `matrix_agd_target_10gb`
- `matrix_bigamp_target_10gb`
- `matrix_bigamp_target_16gb`
- `spreading_bigamp_target_10gb`
- `tensor_serial_target_6gb`
- `tensor_parallel_target_6gb`
- `tensor_parallel_target_10gb`

`*_small` 只验证记录链路；它们的显存误差会被 CUDA/PyTorch 固定开销支配，不会写成 planner 可用的校准系数。`*_target_10gb` / `*_target_16gb` 才用于正式并行估计校准。

校准 raw 记录写入被 ignore 的 `runs/calibration/memory/<profile>/<timestamp>/manifest.json`。记录同时保存：

- `theoretical_estimate_gb`：纯 tensor 公式估计，不含 CUDA context。
- `estimated_total_with_runtime_gb`：`MemoryEstimator.estimate()` 的运行时总估计，含 context/fragmentation 估算。
- `actual_peak_memory_gb`：`torch.cuda.max_memory_allocated()` 记录的实际 peak allocated。
- `actual_delta_peak_tensor_allocated_gb`：扣除 sampler baseline 后的 tensor allocated 峰值，用于拟合 planner 系数。
- `actual_delta_peak_cuda_device_used_gb`：扣除 sampler baseline 后的 device used 峰值，用于观察 CUDA context、allocator reserve、外部进程等整体影响。
- `reserved_peak_memory_gb`、`allocated_after_gb`、`reserved_after_gb`。
- `memory_timeline.path`：完整 VRAM 时间线，默认 `memory_timeline.jsonl`。

当 raw 估计和实际 delta 都超过 1GB 时，校准命令会更新本地 `runs/calibration/memory/latest_coefficients.json`。`MemoryEstimator` 会在后续 planner 中读取这个文件，并只做保守校正：如果实测比公式大，放大估算；如果实测比公式小，默认不把系数降到 1 以下。

### 2026-04-26 本地 RTX 5090 校准记录

这组记录是实际本地运行，不是 smoke，也不是只看代码估算。raw artifact 在 ignored 的 `runs/calibration/memory/`，下面只记录轻量摘要：

| algorithm | profile | raw estimate GB | actual max allocated GB | device used delta GB | planner factor |
| --- | --- | ---: | ---: | ---: | ---: |
| `bigamp` | `matrix_bigamp_target_16gb` | 15.982 | 15.399 | 16.805 | 1.060 |
| `bigamp_spreading` | `spreading_bigamp_target_10gb` | 9.996 | 6.718 | 8.527 | 1.000 |
| `bigamp_tensor` | `tensor_serial_target_6gb` | 5.998 | 6.806 | 7.703 | 1.248 |
| `bigamp_tensor_parallel` | `tensor_parallel_target_6gb` | 5.998 | 15.480 | 15.629 | 2.839 |
| `agd` | `matrix_agd_target_10gb` | 9.999 | 24.728 | 29.005 | 2.720 |

结论：

- dense `bigamp` 公式在 10-16GB 级别基本可信，保守系数约 1.06。
- `bigamp_spreading` 当前公式偏保守，仍保持 factor=1，不让 planner 变激进。
- `bigamp_tensor_parallel` 的 internal probe 路径显著超出公式估计，必须使用实测 factor；原 `tensor_parallel_target_10gb` 在未校准时实际冲到 18.55GB 并失败，校准后同一 profile 会被运行前保护拒绝。
- `agd` 公式严重低估，10GB raw 实际接近 25GB allocated / 29GB device delta，后续 AGD batch 必须用实测 factor。

### 当前真实支持的 batching 维度

当前 runner 真实接通的是 alpha batching：同一 batch 可以包含多个 `alpha_values`，算法实际收到这组 alpha 并并行或内部调度。

`sample_range` / student folding 目前只允许作为 metadata 出现在 `ResourceExecutionPlan`，不能用于降低真实执行显存。原因是 runner 尚未把 sample offset 传入算法，算法初始化随机流也没有统一暴露 `sample_offset` contract。上一版 planner 曾尝试在内存不够时把 `S` 降到 1 做估算，但 runner 实际仍按完整 `samples_per_alpha` 执行，这会低估显存；现在已经禁用。后续要恢复 sample/student folding，必须先完成：

- `AlgorithmSpec/BatchingSpec` 声明 `sample_range_honored=true`。
- runner 把 `sample_range` 和 `sample_offset` 传入 algorithm。
- algorithm 的 partition-invariant seed 使用全局 sample index，而不是 batch-local index。
- metrics 聚合能把多个 sample batch 合并回同一个 alpha 的统计量。

在这些条件满足前，planner 的最小真实 batch 是“单 alpha + 完整 S”，不是 “S=1”。

linear fallback 的安全检查必须检查每一个 single-alpha/full-S batch；只要最大 alpha batch 超过当前安全分配，planner 就报错。不能只看最小 alpha batch，否则会把低 alpha 能跑误判成整组能跑。

`algorithm_params.compile_fallback_policy` 只控制 `torch.compile` 初始化失败时的行为，目前接入 `bigamp`、`bigamp_spreading` 和 `bigamp_tensor_parallel`：

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
requested_use_tf32
tf32_matmul_enabled
tf32_cudnn_enabled
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

## 仍未完成的事

- 不用中大型校准 profile 自动回写 memory coefficient。
- 不在 OOM 后自动切换 dtype；普通 BF16 不可用只按 `dtype_fallback_policy` 处理。
- 不在 OOM 后自动关闭 compile；普通 `torch.compile` 初始化失败只按 `compile_fallback_policy` 处理。
- seed-sensitive 路径不在 OOM 后自动重分批。
- `nested/hysteresis` 仍由 legacy handler 执行；`ScanPlan` 先作为硬 contract 和 plot grouping 来源。
- 不把 tensor serial/parallel 合并。

这些都进入 `docs/parallel_memory_review_queue.md`。
