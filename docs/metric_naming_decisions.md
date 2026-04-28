# Metric 命名决策表

这份表只管理“以后显示、文档和对话里怎么叫这些量”。它不修改任何公式，也不改变旧结果里的 flat key。机器定义仍以 `src/matrix_factorization/core/contracts.py` 的 `MetricSemanticClass` 为准。

规则：

- `canonical_key` 是程序内部稳定语义名，不建议随便改。
- `legacy aliases` 是当前 `metrics.json` 里保留的旧 flat key。
- `current display` 是现在文档里的临时显示名。
- `preferred display` 先留空或写候选，等你确认后再固化。
- `decision status=pending_user` 表示需要你之后选名字；这不阻塞代码结构硬化。

## Equivalent Class: output_similarity.full

| canonical_key | legacy aliases | current display | preferred display | decision status | 备注 |
| --- | --- | --- | --- | --- | --- |
| `matrix.full.teacher_student.output_cosine` | `Q_Y_mean`, `Q_Y_std` | matrix full output cosine | pending | pending_user | dense matrix full-output cosine，matrix 主 order parameter 候选。 |
| `tensor.full.teacher_student.cp_tensor_cosine` | `Q_Y_mean`, `Q_Y_std` | tensor full CP tensor cosine | pending | pending_user | tensor full CP tensor cosine；和 matrix 的 `Q_Y_mean` 同名但不同义。 |

## Equivalent Class: output_similarity.observed

| canonical_key | legacy aliases | current display | preferred display | decision status | 备注 |
| --- | --- | --- | --- | --- | --- |
| `matrix.observed.teacher_student.output_cosine` | `Q_Y_observed_mean`, `Q_Y_observed_std` | matrix observed output cosine | pending | pending_user | dense matrix mask-observed cosine，偏 diagnostic。 |
| `spreading.observed.teacher_student.F_aware_output_cosine` | `Q_Y_observed_mean`, `Q_Y_observed_std` | spreading observed F-aware output cosine | pending | pending_user | F-aware graph measurement；名字最好体现 F/graph observed。 |
| `tensor.observed.teacher_student.serial_reconstruction_quality` | `Q_Y_mean`, `Q_Y_std` | serial tensor observed reconstruction quality | pending | pending_user | serial tensor legacy diagnostic；不应直接和 tensor full `Q_Y_mean` 混用。 |
| `tensor.observed.teacher_student.reconstruction_quality` | `Q_Y_observed_mean`, `Q_Y_observed_std` | tensor observed reconstruction quality | pending | pending_user | tensor observed-edge reconstruction diagnostic。 |

## Equivalent Class: output_similarity.unobserved

| canonical_key | legacy aliases | current display | preferred display | decision status | 备注 |
| --- | --- | --- | --- | --- | --- |
| `matrix.unobserved.teacher_student.output_cosine` | `Q_Y_unobserved_mean`, `Q_Y_unobserved_std` | matrix unobserved output cosine | pending | pending_user | dense unobserved-entry cosine，generalization 指标候选。 |
| `spreading.unobserved.teacher_student.dense_output_cosine` | `Q_Y_unobserved_mean`, `Q_Y_unobserved_std` | spreading induced unobserved dense output cosine | pending | pending_user | spreading mask induced dense-output diagnostic，不是 F-observed graph metric。 |

## Equivalent Class: factor_overlap.teacher_student

| canonical_key | legacy aliases | current display | preferred display | decision status | 备注 |
| --- | --- | --- | --- | --- | --- |
| `factor.W.teacher_student.gram_cosine` | `Q_W_mean`, `Q_W_std` | W teacher-student Gram cosine | pending | pending_user | W Gram overlap；比 coordinate projection 更 gauge-friendly，但仍需确认 rank/gauge convention。 |
| `factor.W.teacher_student.baseline_corrected_gram_cosine` | `Q_W_prime_mean`, `Q_W_prime_std` | W teacher-student baseline-corrected Gram cosine | pending | pending_user | W baseline-corrected Gram overlap。 |
| `factor.X.teacher_student.gram_cosine` | `Q_X_mean`, `Q_X_std` | X teacher-student Gram cosine | pending | pending_user | X Gram overlap。 |
| `factor.X.teacher_student.baseline_corrected_gram_cosine` | `Q_X_prime_mean`, `Q_X_prime_std` | X teacher-student baseline-corrected Gram cosine | pending | pending_user | X baseline-corrected Gram overlap。 |

## Equivalent Class: physical_projection

| canonical_key | legacy aliases | current display | preferred display | decision status | 备注 |
| --- | --- | --- | --- | --- | --- |
| `factor.W.teacher_student.coordinate_projection_abs` | `physical_overlap_W_mean`, `physical_overlap_W_std` | W coordinate projection absolute overlap | pending | pending_user | sign/permutation/gauge sensitive，当前只应叫 diagnostic。 |
| `factor.X.teacher_student.coordinate_projection_abs` | `physical_overlap_X_mean`, `physical_overlap_X_std` | X coordinate projection absolute overlap | pending | pending_user | sign/permutation/gauge sensitive，当前只应叫 diagnostic。 |
| `matrix.full.teacher_student.output_projection` | `physical_overlap_Y_mean`, `physical_overlap_Y_std` | matrix Y output projection overlap | pending | pending_user | output-space projection；是否叫 physical order parameter 需要确认 scale convention。 |
| `tensor.full.teacher_student.projection_overlap` | `physical_overlap_Y_mean` | tensor Y projection overlap | pending | pending_user | tensor-space projection；需要和 tensor serial/parallel parity 一起确认。 |

## Equivalent Class: reconstruction_error

| canonical_key | legacy aliases | current display | preferred display | decision status | 备注 |
| --- | --- | --- | --- | --- | --- |
| `matrix.full.teacher_student.reconstruction_mse` | `MSE`, `MSE_std`, `Gen_Error` | matrix full reconstruction MSE | pending | pending_user | formal error metric，不是 order parameter；`Gen_Error` 是 legacy alias。 |

## Equivalent Class: replica_overlap

| canonical_key | legacy aliases | current display | preferred display | decision status | 备注 |
| --- | --- | --- | --- | --- | --- |
| `factor.W.replica.student_student.gram_cosine` | `Q_W_replica_mean` | W replica Gram cosine | pending | pending_user | student-student W replica diagnostic。 |
| `factor.X.replica.student_student.gram_cosine` | `Q_X_replica_mean` | X replica Gram cosine | pending | pending_user | student-student X replica diagnostic。 |
| `factor.W.replica.student_student.baseline_corrected_gram_cosine` | `Q_W_prime_replica_mean` | W replica baseline-corrected Gram cosine | pending | pending_user | baseline-corrected W replica diagnostic。 |
| `factor.X.replica.student_student.baseline_corrected_gram_cosine` | `Q_X_prime_replica_mean` | X replica baseline-corrected Gram cosine | pending | pending_user | baseline-corrected X replica diagnostic。 |

## Equivalent Class: replica_heatmap

| canonical_key | legacy aliases | current display | preferred display | decision status | 备注 |
| --- | --- | --- | --- | --- | --- |
| `replica.heatmap.teacher_and_students.matrix` | `overlap_matrix`, `overlap_matrix_metric` | teacher plus replicas overlap matrix | pending | pending_user | matrix-valued diagnostic artifact，不是 scalar timeseries。 |

## 暂不纳入 formal metric 的名字

| name | 当前处理 | 原因 |
| --- | --- | --- |
| `loss` | training diagnostic / early-stop criterion | 不同 algorithm 的 loss 不是同一个物理量；未来如需保存，需要单独 `MetricSpec`。 |
| `MSE_mean` | tensor single-alpha diagnostic | 当前 tensor parallel 的主 schema 使用 `Q_Y_mean` / `physical_overlap_Y_mean` / heatmap artifact；`MSE_mean` 需要后续 tensor result schema 统一时再决定。 |
| `factor_cosine` | descriptive helper name | 当前正式 flat key 是 `Q_W_mean/Q_X_mean` 及 prime variants。 |

## Projection Metric Migration v3: approved active names

| canonical_key | active aliases | display | decision status | 备注 |
| --- | --- | --- | --- | --- |
| `measurement.full.teacher_student.Q_Y_projection` | `Q_Y_mean`, `Q_Y_std` | Q_Y | approved | measurement/output absolute projection，full scope。 |
| `measurement.observed.teacher_student.Q_Y_projection` | `Q_Y_observed_mean`, `Q_Y_observed_std` | Q_Y observed | approved | observed/training measurement projection。 |
| `measurement.unobserved.teacher_student.Q_Y_projection` | `Q_Y_unobserved_mean`, `Q_Y_unobserved_std` | Q_Y unobserved | approved | heldout/unobserved measurement projection。 |
| `latent.W.teacher_student.Q_W_projection` | `Q_W_mean`, `Q_W_std` | Q_W | approved | matrix W latent coordinate projection。 |
| `latent.X.teacher_student.Q_X_projection` | `Q_X_mean`, `Q_X_std` | Q_X | approved | matrix X latent coordinate projection。 |
| `latent.W.teacher_student.Q_W_COS_ROOT` | `Q_W_COS_ROOT_mean`, `Q_W_COS_ROOT_std` | Q_W Cos root | approved | sqrt baseline-corrected Cos-root diagnostic。 |
| `latent.X.teacher_student.Q_X_COS_ROOT` | `Q_X_COS_ROOT_mean`, `Q_X_COS_ROOT_std` | Q_X Cos root | approved | sqrt baseline-corrected Cos-root diagnostic。 |
| `latent.W.teacher_student.Q_W_SIGN_ALIGNED` | `Q_W_SIGN_ALIGNED_mean`, `Q_W_SIGN_ALIGNED_std` | Q_W sign-aligned | approved | 逐 channel 取绝对值的 sign-gauge diagnostic；不处理 permutation/rotation。 |
| `latent.X.teacher_student.Q_X_SIGN_ALIGNED` | `Q_X_SIGN_ALIGNED_mean`, `Q_X_SIGN_ALIGNED_std` | Q_X sign-aligned | approved | 逐 channel 取绝对值的 sign-gauge diagnostic；不处理 permutation/rotation。 |
| `latent.W.teacher_student.Q_W_SCALE_GAUGE` | `Q_W_SCALE_GAUGE_mean`, `Q_W_SCALE_GAUGE_std` | Q_W scale-gauge | approved | 联合 W/X diagonal scale-gauge 对齐后的 W diagnostic；不改变训练轨迹。 |
| `latent.X.teacher_student.Q_X_SCALE_GAUGE` | `Q_X_SCALE_GAUGE_mean`, `Q_X_SCALE_GAUGE_std` | Q_X scale-gauge | approved | 联合 W/X diagonal scale-gauge 对齐后的 X diagnostic；不改变训练轨迹。 |
| `latent.WX.teacher_student.Q_WX_SCALE_GAUGE` | `Q_WX_SCALE_GAUGE_mean`, `Q_WX_SCALE_GAUGE_std` | Q_WX scale-gauge | approved | `Q_W_SCALE_GAUGE` 与 `Q_X_SCALE_GAUGE` 的平均 diagnostic。 |
| `latent.WX.teacher_student.scale_gauge_magnitude` | `median_abs_log_g_mean`, `median_abs_log_g_std` | scale-gauge magnitude | approved | 拟合出的 channel scale gauge 大小，公式是 median `|log |g_k||`。 |
| `latent.N.teacher_student.Q_N_projection` | `Q_N_mean`, `Q_N_std`, `Q_N_mode*_mean`, `Q_N_mode*_std` | Q_N | approved | tensor latent node/spin/factor projection。 |
