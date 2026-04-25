# Metrics 语义说明

这份文档记录当前 metric 名称及其含义。它不修改任何代码或 result key。当前目标是避免把同一个名字误读成同一个物理量。

## 命名原则

理解每个 metric 时，至少要同时看四个属性：

- `space`：factor、matrix、tensor，或 graph-observed space。
- `scope`：full、observed、unobserved，或 replica。
- `relation`：teacher-student，或 student-student。
- `normalization`：cosine、baseline-corrected Gram overlap、projection、MSE，或 reconstruction-quality transform。

为了兼容旧结果，flat key 暂时保留。未来 schema 应该显式携带这些属性。

## Matrix Metrics

```text
Q_W_mean / Q_W_std
Q_X_mean / Q_X_std
```

Factor Gram cosine。它们不是 raw coordinate-wise factor cosine。它们对 latent permutation 和 sign flip 比较稳定，但不能覆盖所有非正交 gauge 或 scale redistribution。

```text
Q_W_prime_mean / Q_W_prime_std
Q_X_prime_mean / Q_X_prime_std
```

Baseline-corrected Gram overlap。它们比 raw factor projection 更接近 factor-level order parameter，但仍然需要明确 gauge convention。

```text
Q_Y_mean / Q_Y_std
```

当由 standard runner 为 `agd` 或 `bigamp` 生成时，它表示 dense matrix 的 full-output cosine。

```text
Q_Y_observed_mean / Q_Y_observed_std
Q_Y_unobserved_mean / Q_Y_unobserved_std
```

按 dense mask 划分的 matrix cosine。它们是 scope-specific diagnostic，不应和 full `Q_Y_mean` 静默混用。

```text
MSE / MSE_std
Gen_Error
loss
```

这些名称目前没有完全统一。主 runner 中的 `MSE` 是 full matrix reconstruction error。`Gen_Error` 出现在旧 metric helper 中。算法内部的 `loss` 通常是 training diagnostic 或 early-stop criterion，不是 canonical result metric。

```text
physical_overlap_W_mean
physical_overlap_X_mean
physical_overlap_Y_mean
```

Projection-style diagnostic。Factor-level physical overlap 仍然对 permutation、sign 和 gauge 选择敏感。只有当 `physical_overlap_Y_mean` 是 output space 中的 global projection 时，才适合当作 physical order parameter。

## Spreading Metrics

Matrix spreading 的观测是 F-aware graph measurement。因此 `Y` 至少有两个不同空间：

- full dense matrix reconstruction space；
- 使用同一份 quenched `F` 的 observed spreading graph space。

当前 key：

```text
Q_Y_mean
Q_Y_observed_mean
Q_Y_unobserved_mean
```

在 parallel spreading metrics 路径中，`Q_Y_mean` 当前用于 full matrix cosine。`Q_Y_observed_mean` 是 F-aware observed graph metric。没有显式 scope label 时，不能把它们当作同一个量。

Spreading 的 physical overlap key 也需要同样谨慎。有些路径计算 global projection，有些路径计算 pointwise ratio 后再平均。除非明确具体实现，否则应先视作 diagnostic。

## Tensor Metrics

```text
Q_Y_mean / Q_Y_std
```

对 `bigamp_tensor_parallel` 来说，这是 full CP tensor cosine。对 serial `bigamp_tensor` 路径来说，当前更接近 observed-edge reconstruction-quality diagnostic。两者不能直接互换。

机器 contract 中这两者已经拆开：`bigamp_tensor_parallel` 的 `Q_Y_mean` 对应 `tensor.full.Q_Y`；serial `bigamp_tensor` 的 legacy `Q_Y_mean` 对应 `tensor.serial_observed.Q_Y`。

```text
Q_Y_observed_mean / Q_Y_observed_std
```

Observation-only tensor diagnostic。在 tensor parallel 中，它来自 observed-edge reconstruction error 和 target variance，不是 full tensor cosine。

```text
physical_overlap_Y_mean
```

Tensor-space projection。如果 teacher 和 student tensor 使用同一套 scale convention，它可以作为候选 physical order parameter。

```text
overlap_matrix
overlap_matrix_metric
```

Teacher 加 replicas 的 heatmap payload。这是 matrix-valued diagnostic，用于 replica/RSB 可视化，不是 scalar order parameter。

```text
compute_factor_gram_overlap()
```

Tensor helper 使用的 factor Gram overlap。它对 CP column permutation 和 sign 比较稳定，但不能覆盖所有跨 mode 的 CP scale redistribution。

## Replica Metrics

Replica metrics 比较的是 student replicas 之间的 overlap，而不是 student 和 teacher。它们用于诊断 multiplicity、stability 和可能的 RSB-like behavior。

常见名称：

```text
Q_W_replica_mean
Q_X_replica_mean
Q_W_prime_replica_mean
Q_X_prime_replica_mean
```

未来 schema 应显式区分 teacher-student relation 和 student-student relation。

## 未来建议的 Canonical 名称

为了兼容旧结果，现有 flat key 不应直接删除或重命名。可以先增加 structured alias 或 metadata：

```text
metrics:
  y:
    full:
      cosine_mean
      mse_mean
      physical_projection_mean
    observed:
      cosine_mean
      reconstruction_quality_mean
    unobserved:
      cosine_mean
  factors:
    W:
      gram_cosine_mean
      gram_cosine_baseline_corrected_mean
      projection_mean
    X:
      gram_cosine_mean
      gram_cosine_baseline_corrected_mean
      projection_mean
  replica:
    W:
      gram_cosine_mean
      gram_cosine_baseline_corrected_mean
  heatmap:
    matrix
    metric
```

在兼容层存在之前，不直接改当前 key。
