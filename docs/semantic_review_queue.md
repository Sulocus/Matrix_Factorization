# Semantic Review Queue

这份文档记录需要用户之后确认的命名和物理解释问题。这里的项目不会阻塞低风险软件结构工作；它们只在涉及显示名、物理 order parameter、或者公式解释时需要人工确认。

完整的逐项命名决策表见 `docs/metric_naming_decisions.md`。那份表按 `MetricSemanticClass.canonical_key` 覆盖所有当前 metric 等价类；新增 canonical metric 如果没有写进命名表，contract test 会失败。

## Metric 命名偏好

- `matrix.full.teacher_student.output_cosine`
  - 当前 legacy key: `Q_Y_mean`
  - 待确认：图例是否继续显示为 `Q_Y`，还是显示为 `matrix full Q_Y`。

- `tensor.full.teacher_student.cp_tensor_cosine`
  - 当前 legacy key: `Q_Y_mean`
  - 待确认：是否允许图例也显示为 `Q_Y`，还是必须显示为 `tensor full Q_Y`。

- `spreading.observed.teacher_student.F_aware_output_cosine`
  - 当前 legacy key: `Q_Y_observed_mean`
  - 待确认：是否应命名为 `Q_Y_observed`、`Q_Y_F_observed`，或其他能体现 F-aware graph measurement 的名字。

- `tensor.observed.teacher_student.reconstruction_quality`
  - 当前 legacy key: `Q_Y_observed_mean`
  - 待确认：它是否应继续叫 `Q_Y_observed`，还是改显示名为 `observed reconstruction quality`。

## Physical Order Parameter 判断

- `matrix.full.teacher_student.output_cosine`
  - 候选 order parameter。
  - 待确认：是否作为 matrix 主物理 order parameter。

- `spreading.observed.teacher_student.F_aware_output_cosine`
  - 候选 order parameter。
  - 待确认：是否比 dense full `Q_Y_mean` 更适合作为 spreading 主物理 order parameter。

- `tensor.full.teacher_student.cp_tensor_cosine`
  - 候选 order parameter。
  - 待确认：必须先确认 teacher/student tensor scale convention。

- `matrix.full.teacher_student.output_projection`
  - 当前标为 review。
  - 风险：是否能作为 physical order parameter 取决于 output-space scale convention。

- `factor.W.teacher_student.coordinate_projection_abs`
  - 当前标为 diagnostic。
  - 风险：factor-level projection 对 sign、permutation、gauge 敏感。

- `factor.X.teacher_student.coordinate_projection_abs`
  - 当前标为 diagnostic。
  - 风险：factor-level projection 对 sign、permutation、gauge 敏感。

- `tensor.full.teacher_student.projection_overlap`
  - 当前标为 candidate。
  - 风险：不同 tensor implementation 的 scale convention 需要 parity 审查。

## Legacy Key 保留问题

- `Q_Y_mean`
  - 当前短期保留。
  - 待确认：未来是否增加 structured alias 后逐步从 plot label 中淡化这个 flat key。

- `Gen_Error`
  - 当前只作为 legacy alias 记录。
  - 待确认：是否还有正式脚本依赖它。

- `loss`
  - 当前不纳入 formal metric。
  - 待确认：哪些 algorithm 的 loss 需要进入 result schema，哪些只保留在 debug/history。

## Tensor Serial / Parallel 命名问题

- `bigamp_tensor` serial path 的 `Q_Y_mean`
  - 当前语义：observed reconstruction-quality diagnostic。
  - 待确认：是否应长期保留这个 legacy key，或只作为 serial/reference diagnostic。

- `bigamp_tensor_parallel` 的 `Q_Y_mean`
  - 当前语义：full CP tensor cosine。
  - 待确认：与 serial path 合并前是否要求 serial 也产生同名 full tensor cosine。
