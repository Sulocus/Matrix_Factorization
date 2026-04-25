# Parallel / Memory Contract

这份文档记录当前并行、分批和显存策略。它不改变执行行为，只解释当前程序会怎样计划 batch、估计 memory、选择 dtype/compile 路径。

机器可读版本在 `src/matrix_factorization/core/contracts.py`：

- `ResourceSpec`：algorithm 的设备、dtype、compile、probe、empty-cache 能力声明。
- `BatchingSpec`：algorithm 的 alpha batching、sample batching、chunking、seed partition 风险声明。
- `ExperimentPlan.resource_plan`：`mf explain-config` / `mf validate --json` 中的静态 resource summary。
- `runtime_resource_plan`：run metadata 中的 runner-level execution plan。

## 当前策略

### AGD

```text
planner: runner.ParallelCoordinator
alpha batching: runner alpha batches
sample batching: samples inside algorithm tensor
resource estimator: runner.MemoryEstimator
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
seed sensitive: true
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
```

风险：algorithm 内部会排序 alpha，并使用 `seed + batch_idx`。改变 batch partition 会改变 graph/F/student random stream。当前所有 ResourceSpec/BatchingSpec 都是 metadata-only，不能直接用来自动调 batch。

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
  batches
  metadata_only
```

## 不在本阶段做的事

- 不根据 ResourceSpec 自动改 batch size。
- 不自动切换 dtype。
- 不自动关闭 compile。
- 不在 OOM 后自动重试。
- 不改变 seed 与 batch partition 的关系。
- 不把 tensor serial/parallel 合并。

这些都进入 `docs/parallel_memory_review_queue.md`。
