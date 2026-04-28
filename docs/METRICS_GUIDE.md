# Metric Guide

本文档记录当前 active path 的正式指标。schema v5 的 profile 是
`projection_qy_physical_latent_v2`：`Q_W/Q_X/Q_N` 是固定分母 latent
overlap，`Q_Y` 是 output absolute projection，`FIT_Y` 是输出重构 fit。
旧 schema 的同名字段必须靠 `metric_schema.schema_version` 解释。

## Formal Metrics

### `Q_Y`

`Q_Y` 是 output absolute projection：

```text
Q_Y = |<Y_student, Y_teacher>| / <Y_teacher, Y_teacher>
```

`FIT_Y` 是 output reconstruction fit：

```text
NMSE_Y = sum_e (Y_student[e] - Y_teacher[e])^2 / sum_e Y_teacher[e]^2
FIT_Y = 1 - NMSE_Y
```

规则：

- `Q_Y` 不裁切；scale 错误会反映为大于 `1`。
- `FIT_Y` 不裁切；坏结果可以小于 `0`。
- 完美重构时 `Q_Y = 1` 且 `FIT_Y = 1`，零输出通常给 `Q_Y = 0`、`FIT_Y = 0`。
- 若 evaluation set 为空或 teacher norm 小于 `1e-12`，`Q_Y=0`、`NMSE_Y=1`、`FIT_Y=0`。
- `Q_Y_observed` 表示 observed/training measurement set。
- `Q_Y_unobserved` 表示 heldout 或 unobserved measurement set。
- `Q_Y_PROJ_ABS` 是旧 absolute projection diagnostic，不是正式 `Q_Y`。

### `Q_W` / `Q_X`

`Q_W` 和 `Q_X` 是 paper-style fixed-denominator overlap：

```text
Q_W = sum_i,mu W_student[i,mu] W_teacher[i,mu] / (N1 M)
Q_X = sum_mu,j X_student[mu,j] X_teacher[mu,j] / (M N2)
```

这不是 teacher-norm projection。Gaussian teacher 完美恢复时，有限尺寸下
`Q_W/Q_X` 等于 teacher empirical second moment，通常接近但不强制等于 `1`。
`R_W/R_X` 是 student self-overlap，用来监控 norm collapse 或 blow-up。

### `Q_N`

tensor latent factor 使用同一 fixed-denominator convention：

```text
Q_N_mode_d = sum_entries N_student^(d) N_teacher^(d) / numel(N_teacher^(d))
Q_N = mean_d Q_N_mode_d
```

## Diagnostics

### `Q_W_COS_ROOT` / `Q_X_COS_ROOT`

```text
Q_W_COS_ROOT = sqrt(max(baseline_corrected_gram_overlap(W), 0))
Q_X_COS_ROOT = sqrt(max(baseline_corrected_gram_overlap(X), 0))
```

这是 Cos-root diagnostic，不是 physical overlap。

### `Q_W_SIGN_GAUGE` / `Q_X_SIGN_GAUGE`

```text
Q_W_SIGN_GAUGE = sum_mu abs(sum_i W_s[i,mu] W_t[i,mu]) / (N1 M)
Q_X_SIGN_GAUGE = sum_mu abs(sum_j X_s[mu,j] X_t[mu,j]) / (M N2)
```

它只修正每个 latent channel 的 sign gauge，不处理 permutation、rotation 或
continuous scale。`Q_W_SIGN_ALIGNED/Q_X_SIGN_ALIGNED` 是 legacy aliases。

### `Q_W_SCALE_GAUGE` / `Q_X_SCALE_GAUGE`

对每个 latent channel 求：

```text
k_mu* = argmin_{k != 0} [
  ||k W_s[:,mu] - W_t[:,mu]||^2
  + ||k^-1 X_s[mu,:] - X_t[mu,:]||^2
]
```

然后用同一个 fixed-denominator convention 计算 aligned overlap。
`Q_WX_SCALE_GAUGE` 是 W/X 两侧的平均。`median_abs_log_k` 记录 fitted
scale gauge 的大小。这个 diagnostic 不改变训练轨迹。

## Legacy Metrics

- `Q_Y_PROJ_ABS / Q_W_PROJ_ABS / Q_X_PROJ_ABS`：schema v3 absolute projection。
- `Q_W_GRAM_ROOT / Q_X_GRAM_ROOT`：旧名，alias 到 `Q_W_COS_ROOT/Q_X_COS_ROOT`。
- `median_abs_log_g`：旧名，alias 到 `median_abs_log_k`。
- `MSE / Gen_Error / physical_overlap_* / Q_Y_COS`：legacy/debug，不进入正式 metric surface。

## Result Schema

新 run 的 `metrics.json.metric_schema.schema_version` 为 `5`，并包含：

```text
metric_definition_profile = projection_qy_physical_latent_v2
compatibility.physical_overlap_metric_migration = true
metric_policy.Q_Y_formula = absolute_projection_teacher_norm_squared
metric_policy.Q_W_Q_X_normalization = fixed_coordinate_count
metric_policy.legacy_projection_suffix = _PROJ_ABS
metric_policy.clipped = false
```

`FIT_Y_mean` 另外保存 `1 - NMSE_Y`。因此 schema v3/v4 的
`Q_Y_mean/Q_W_mean/Q_X_mean` 不能重解释成 schema v5 的同名字段；必须同时查看
`metric_schema.schema_version` 和 `metric_definition_profile`。
