# Metrics 语义地图

这份文档描述当前程序实际使用的 metric 语义。它不修改任何公式，也不重命名旧结果 key。目标是把旧的 flat key 放进明确的 equivalent class，避免把同名字段误读成同一个物理量。

机器可读版本在 `src/matrix_factorization/core/contracts.py`：

- `MetricSpec`：metric 的输入、输出、适用 algorithm。
- `MetricSemanticClass`：metric 的 canonical 语义类。
- `get_metric_schema()`：生成写入 `metrics.json` 的 result schema metadata。

## 读法

每个 metric 至少由这些属性共同决定：

```text
space: matrix / tensor / factor / graph / matrix-valued
scope: full / observed / unobserved / replica
relation: teacher-student / student-student
normalization: cosine / masked cosine / F-aware cosine / projection / MSE / reconstruction quality
result_role: formal / diagnostic / plotting_helper
order_parameter_status: candidate / diagnostic / not_order_parameter / review
```

旧 key 继续保留，例如 `Q_Y_mean`。但解释旧 key 时必须同时看 algorithm context 和 `metric_schema.flat_key_index`。

## Equivalent Classes

### output_similarity.full

```text
matrix.full.teacher_student.output_cosine
  legacy aliases: Q_Y_mean, Q_Y_std
  appears in: agd, bigamp, bigamp_spreading
  meaning: dense matrix full-output cosine
  status: candidate order parameter
  risk: medium

tensor.full.teacher_student.cp_tensor_cosine
  legacy aliases: Q_Y_mean, Q_Y_std
  appears in: bigamp_tensor_parallel
  meaning: full CP tensor cosine
  status: candidate order parameter
  risk: high
```

注意：这两个 class 都使用 `Q_Y_mean`，但不是同一个语义。matrix 和 tensor 的 `Q_Y_mean` 必须依靠 `metric_schema` 区分。

### output_similarity.observed

```text
matrix.observed.teacher_student.output_cosine
  legacy aliases: Q_Y_observed_mean, Q_Y_observed_std
  appears in: agd, bigamp
  meaning: dense matrix mask-observed cosine
  status: diagnostic
  risk: medium

spreading.observed.teacher_student.F_aware_output_cosine
  legacy aliases: Q_Y_observed_mean, Q_Y_observed_std
  appears in: bigamp_spreading
  meaning: observed graph metric using the same quenched F
  status: candidate order parameter
  risk: high

tensor.observed.teacher_student.serial_reconstruction_quality
  legacy aliases: Q_Y_mean, Q_Y_std
  appears in: bigamp_tensor
  meaning: serial tensor observed-edge reconstruction quality
  status: diagnostic
  risk: high

tensor.observed.teacher_student.reconstruction_quality
  legacy aliases: Q_Y_observed_mean, Q_Y_observed_std
  appears in: bigamp_tensor_parallel, agd_tensor
  meaning: tensor observed-edge reconstruction diagnostic
  status: diagnostic
  risk: high
```

注意：`observed` 在 matrix、spreading、tensor 中不是同一个观测空间。spreading observed 是 F-aware graph measurement；tensor observed 当前更接近 reconstruction-quality diagnostic。

### output_similarity.unobserved

```text
matrix.unobserved.teacher_student.output_cosine
  legacy aliases: Q_Y_unobserved_mean, Q_Y_unobserved_std
  appears in: agd, bigamp
  meaning: dense matrix unobserved-entry cosine
  status: candidate order parameter
  risk: medium

spreading.unobserved.teacher_student.dense_output_cosine
  legacy aliases: Q_Y_unobserved_mean, Q_Y_unobserved_std
  appears in: bigamp_spreading
  meaning: dense-output diagnostic induced by spreading mask
  status: diagnostic
  risk: high
```

### factor_overlap.teacher_student

```text
factor.W.teacher_student.gram_cosine
  legacy aliases: Q_W_mean, Q_W_std
  appears in: agd, bigamp, bigamp_spreading, agd_spreading
  meaning: W factor Gram cosine overlap
  status: candidate / diagnostic boundary requires review
  risk: medium

factor.W.teacher_student.baseline_corrected_gram_cosine
  legacy aliases: Q_W_prime_mean, Q_W_prime_std
  appears in: agd, bigamp, bigamp_spreading, agd_spreading
  meaning: W factor baseline-corrected Gram cosine overlap
  status: candidate / diagnostic boundary requires review
  risk: medium

factor.X.teacher_student.gram_cosine
  legacy aliases: Q_X_mean, Q_X_std
  appears in: agd, bigamp, bigamp_spreading
  meaning: X factor Gram cosine overlap
  status: candidate / diagnostic boundary requires review
  risk: medium

factor.X.teacher_student.baseline_corrected_gram_cosine
  legacy aliases: Q_X_prime_mean, Q_X_prime_std
  appears in: agd, bigamp, bigamp_spreading
  meaning: X factor baseline-corrected Gram cosine overlap
  status: candidate / diagnostic boundary requires review
  risk: medium
```

这些量比 coordinate-wise projection 更稳定，但仍需要明确 gauge、permutation、sign convention。

### physical_projection

```text
factor.W.teacher_student.coordinate_projection_abs
  legacy aliases: physical_overlap_W_mean, physical_overlap_W_std
  appears in: agd, bigamp, bigamp_spreading
  meaning: absolute coordinate projection of W onto teacher W
  status: diagnostic
  risk: high

factor.X.teacher_student.coordinate_projection_abs
  legacy aliases: physical_overlap_X_mean, physical_overlap_X_std
  appears in: agd, bigamp, bigamp_spreading
  meaning: absolute coordinate projection of X onto teacher X
  status: diagnostic
  risk: high

matrix.full.teacher_student.output_projection
  legacy aliases: physical_overlap_Y_mean, physical_overlap_Y_std
  appears in: agd, bigamp, bigamp_spreading
  meaning: global output-space projection
  status: review
  risk: high

tensor.full.teacher_student.projection_overlap
  legacy aliases: physical_overlap_Y_mean
  appears in: bigamp_tensor_parallel
  meaning: tensor-space teacher/student projection
  status: candidate, but scale convention must be reviewed
  risk: high
```

factor-level projection overlap is sign/gauge sensitive. `physical_overlap_Y_mean` is a better physical-order-parameter candidate only when output-space scale convention is explicit.

### reconstruction_error

```text
matrix.full.teacher_student.reconstruction_mse
  legacy aliases: MSE, MSE_std, Gen_Error
  appears in: agd, bigamp, bigamp_spreading; Gen_Error appears in legacy helpers
  meaning: dense full reconstruction mean squared error
  status: formal metric, not an order parameter
  risk: low
```

`loss` is not included here by default. Algorithm loss is a training diagnostic or early-stop criterion unless explicitly registered as a formal metric.

### replica_overlap

```text
factor.W.replica.student_student.gram_cosine
  legacy aliases: Q_W_replica_mean
  appears in: agd, bigamp, bigamp_spreading
  meaning: student-student W replica Gram cosine
  status: diagnostic
  risk: medium

factor.X.replica.student_student.gram_cosine
  legacy aliases: Q_X_replica_mean
  appears in: agd, bigamp, bigamp_spreading
  meaning: student-student X replica Gram cosine
  status: diagnostic
  risk: medium

factor.W.replica.student_student.baseline_corrected_gram_cosine
  legacy aliases: Q_W_prime_replica_mean
  appears in: agd, bigamp, bigamp_spreading
  meaning: student-student W replica baseline-corrected Gram cosine
  status: diagnostic
  risk: medium

factor.X.replica.student_student.baseline_corrected_gram_cosine
  legacy aliases: Q_X_prime_replica_mean
  appears in: agd, bigamp, bigamp_spreading
  meaning: student-student X replica baseline-corrected Gram cosine
  status: diagnostic
  risk: medium
```

Replica overlap is not teacher-student recovery. It diagnoses solution multiplicity, initialization sensitivity, and possible RSB-like behavior.

### replica_heatmap

```text
replica.heatmap.teacher_and_students.matrix
  legacy aliases: overlap_matrix, overlap_matrix_metric
  appears in: bigamp_tensor_parallel heatmap path
  meaning: teacher plus student replicas matrix-valued visualization payload
  status: diagnostic, not scalar order parameter
  risk: medium
```

Heatmap 的每个 entry 使用 `overlap_matrix_metric` 指定的 metric，例如 `Q_Y` 或 `Q_W`。它不应被当成 scalar timeseries。

## 同名不同义风险

- `Q_Y_mean`：
  - matrix full dense output cosine；
  - tensor full CP tensor cosine；
  - serial tensor observed reconstruction-quality diagnostic。
- `Q_Y_observed_mean`：
  - dense matrix masked observed cosine；
  - spreading F-aware graph observed metric；
  - tensor observed reconstruction-quality diagnostic。
- `physical_overlap_Y_mean`：
  - matrix output projection；
  - tensor output projection。

这些 key 在 `metrics.json` 中必须通过 `metric_schema.flat_key_index` 解释。

## 同义不同名风险

- `MSE` 和 legacy `Gen_Error` 都指向 reconstruction error class，但主链路正式 key 是 `MSE`。
- `Q_W_prime / Q_X_prime` 是 factor Gram overlap 的 baseline-corrected variant，不应和 raw `Q_W / Q_X` 静默合并。
- `overlap_matrix_metric=Q_Y` 指的是 heatmap cell 的选择，不等价于 scalar `Q_Y_mean`。

## 暂定命名

当前 canonical 名称是机器语义名，不是最终论文图例名。需要用户最终选择显示名称的内容集中记录在 `docs/semantic_review_queue.md`。
