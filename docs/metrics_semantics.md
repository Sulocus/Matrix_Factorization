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

### latent_factor_scale_gauge_projection

```text
latent.W.teacher_student.Q_W_SCALE_GAUGE
  legacy aliases: Q_W_SCALE_GAUGE_mean, Q_W_SCALE_GAUGE_std
  appears in: bigamp_spreading
  meaning: W projection after jointly aligning each W/X latent channel under diagonal scale gauge
  status: diagnostic
  risk: medium

latent.X.teacher_student.Q_X_SCALE_GAUGE
  legacy aliases: Q_X_SCALE_GAUGE_mean, Q_X_SCALE_GAUGE_std
  appears in: bigamp_spreading
  meaning: X projection after jointly aligning each W/X latent channel under diagonal scale gauge
  status: diagnostic
  risk: medium

latent.WX.teacher_student.Q_WX_SCALE_GAUGE
  legacy aliases: Q_WX_SCALE_GAUGE_mean, Q_WX_SCALE_GAUGE_std
  appears in: bigamp_spreading
  meaning: mean of Q_W and Q_X after joint diagonal scale-gauge alignment
  status: diagnostic
  risk: medium

latent.WX.teacher_student.scale_gauge_magnitude
  legacy aliases: median_abs_log_g_mean, median_abs_log_g_std
  appears in: bigamp_spreading
  meaning: median channel magnitude |log |g_k|| of the fitted diagonal scale gauge
  status: diagnostic
  risk: medium
```

这些量是 diagnostic，不改变训练轨迹。它们用于判断 raw `Q_W/Q_X` 低是否主要来自 diagonal scale gauge 未对齐。

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

- schema v3 之后，active `Q_Y_mean` 统一解释为 measurement absolute projection。
- 旧 schema 的 `Q_Y_mean` 仍可能是 cosine 或 reconstruction-quality diagnostic，不能和 schema v3 的 `Q_Y_mean` 混合比较。
- `metrics.json.metric_schema.compatibility.legacy_q_y_cosine_not_comparable=true` 用来提醒这一点。

## 同义不同名风险

- `physical_overlap_Y/W/X` 是旧 projection 名，schema v3 的正式名是 `Q_Y/Q_W/Q_X`。
- `Q_W_prime / Q_X_prime` 是旧 baseline-corrected Gram 名，schema v3 的正式 diagnostic 是 `Q_W_GRAM_ROOT / Q_X_GRAM_ROOT`。
- `MSE` 和 legacy `Gen_Error` 不再是 formal result metric；只能作为 algorithm 内部 loss/debug 或 legacy result 解释。
- `overlap_matrix_metric=Q_Y` 指的是 heatmap cell 的选择，不等价于 scalar `Q_Y_mean`。

## 暂定命名

当前 canonical 名称是机器语义名，不是最终论文图例名。需要用户最终选择显示名称的内容集中记录在 `docs/semantic_review_queue.md`。

## Projection Metric Migration v3

从 schema v3 开始，active formal metrics 使用 projection-first 定义。旧 cosine / reconstruction / physical_overlap key 只作为 legacy 解释存在，不能和新 run 的同名 flat key 直接比较。

```text
measurement.full.teacher_student.Q_Y_projection
  aliases: Q_Y_mean, Q_Y_std
  formula: abs(<Y_s, Y_t>) / <Y_t, Y_t>
  normalization: teacher_norm_squared, clipped=false
  scope: full configured measurement set

measurement.observed.teacher_student.Q_Y_projection
  aliases: Q_Y_observed_mean, Q_Y_observed_std
  formula: abs(<Y_s, Y_t>) / <Y_t, Y_t>
  scope: observed/training measurements

measurement.unobserved.teacher_student.Q_Y_projection
  aliases: Q_Y_unobserved_mean, Q_Y_unobserved_std
  formula: abs(<Y_s, Y_t>) / <Y_t, Y_t>
  scope: heldout or unobserved measurements

latent.W.teacher_student.Q_W_projection
  aliases: Q_W_mean, Q_W_std
  formula: abs(<W_s, W_t>) / <W_t, W_t>

latent.X.teacher_student.Q_X_projection
  aliases: Q_X_mean, Q_X_std
  formula: abs(<X_s, X_t>) / <X_t, X_t>

latent.W.teacher_student.Q_W_GRAM_ROOT
  aliases: Q_W_GRAM_ROOT_mean, Q_W_GRAM_ROOT_std
  formula: sqrt(max(baseline_corrected_gram_overlap, 0))
  role: gauge/rotation-insensitive diagnostic

latent.W.teacher_student.Q_W_SIGN_ALIGNED
  aliases: Q_W_SIGN_ALIGNED_mean, Q_W_SIGN_ALIGNED_std
  formula: sum_k abs(<W_s[:,k], W_t[:,k]>) / sum_k ||W_t[:,k]||^2
  role: per-channel sign-gauge diagnostic, still rotation/permutation sensitive

latent.X.teacher_student.Q_X_GRAM_ROOT
  aliases: Q_X_GRAM_ROOT_mean, Q_X_GRAM_ROOT_std
  formula: sqrt(max(baseline_corrected_gram_overlap, 0))
  role: gauge/rotation-insensitive diagnostic

latent.X.teacher_student.Q_X_SIGN_ALIGNED
  aliases: Q_X_SIGN_ALIGNED_mean, Q_X_SIGN_ALIGNED_std
  formula: sum_k abs(<X_s[k,:], X_t[k,:]>) / sum_k ||X_t[k,:]||^2
  role: per-channel sign-gauge diagnostic, still rotation/permutation sensitive

latent.N.teacher_student.Q_N_projection
  aliases: Q_N_mean, Q_N_std, Q_N_mode*_mean, Q_N_mode*_std
  formula: mean over tensor modes of abs(<N_s^(d), N_t^(d)>)/<N_t^(d), N_t^(d)>
```
