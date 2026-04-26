# Metric Guide

本文档记录当前 active path 的正式指标。旧结果中的 cosine `Q_Y`、`physical_overlap_*`、`MSE`、`Gen_Error` 和 `Q_*_prime` 只作为 legacy/debug 解释保留，不能和新 schema v3 的同名或近似名字直接比较。

## Formal Projection Metrics

### `Q_Y`

公式：

```text
Q_Y = abs(<Y_student, Y_teacher>) / <Y_teacher, Y_teacher>
```

规则：

- 不做 cosine normalization。
- 不裁切，大于 1 的值保留。
- 若 teacher norm 小于 `1e-12`，projection 返回 `0.0`，并通过 metric schema / metric contract 的 `projection_policy.degenerate_teacher_norm = return_zero` 记录该约定。
- `Q_Y_mean / Q_Y_std` 的 `mean/std` 只是 sample 或 replica 统计后缀。
- `Q_Y_observed` 表示 observed/training measurement set。
- `Q_Y_unobserved` 表示 heldout 或 unobserved measurement set。
- matrix、spreading、tensor 使用同一个物理概念，只是 measurement set 的生成方式不同。

### `Q_W` / `Q_X`

公式：

```text
Q_W = abs(<W_student, W_teacher>) / <W_teacher, W_teacher>
Q_X = abs(<X_student, X_teacher>) / <X_teacher, X_teacher>
```

这是 matrix latent factor 的 coordinate projection overlap。它对 rotation/gauge/permutation 敏感，因此需要同时看 Gram-root diagnostic。

### `Q_N`

公式：

```text
Q_N_mode_d = abs(<N_student^(d), N_teacher^(d)>) / <N_teacher^(d), N_teacher^(d)>
Q_N = mean_d Q_N_mode_d
```

`Q_N` 是 tensor latent node/spin/factor overlap，对齐 matrix 的 `Q_W/Q_X`。

## Formal Diagnostics

### `Q_W_GRAM_ROOT` / `Q_X_GRAM_ROOT`

公式：

```text
Q_W_GRAM_ROOT = sqrt(max(baseline_corrected_gram_overlap(W), 0))
Q_X_GRAM_ROOT = sqrt(max(baseline_corrected_gram_overlap(X), 0))
```

它们不是 coordinate projection，而是解决 matrix factor rotation/gauge 后更稳定的 learning diagnostic。

## Removed From Formal Metrics

- `MSE`：只能作为 algorithm 内部 loss/debug，不进入 formal result metric。
- `Gen_Error`：legacy alias，不进入 formal result metric。
- `physical_overlap_Y/W/X`：旧 projection 名，已迁移到 `Q_Y/Q_W/Q_X`。
- `Q_W_prime/Q_X_prime`：旧 baseline-corrected Gram 名，已迁移到 `Q_W_GRAM_ROOT/Q_X_GRAM_ROOT`。
- `Q_Y_COS` 或旧 cosine `Q_Y`：legacy result 解释，不进入新 run formal schema。

## Result Schema

新 run 的 `metrics.json.metric_schema.schema_version` 为 `3`，并包含：

```text
compatibility.projection_metric_migration = true
compatibility.legacy_q_y_cosine_not_comparable = true
projection_policy.formula = absolute_projection
projection_policy.normalization = teacher_norm_squared
projection_policy.clipped = false
projection_policy.degenerate_teacher_norm = return_zero
```

因此旧 schema 中的 `Q_Y_mean` 不应被重解释成新 projection `Q_Y_mean`。

## Current Implementation Status

- Projection helper、matrix observed/unobserved/full fixture、spreading observed fixture、tensor observed/Q_N fixture 已进入测试。
- `MetricSpec` 和 `metric_schema` 使用 projection-first semantic metadata。
- `ExperimentResult.load()` 会给旧 schema `<3` 的 `Q_Y_mean` 添加 legacy interpretation metadata。
- 新 run 不应在 formal metric surface 中产出 `MSE / Gen_Error / Q_Y_COS / physical_overlap_* / Q_W_prime / Q_X_prime`；这些只能作为 legacy/debug/internal 解释存在。
