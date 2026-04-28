# Alpha Descending Continuation 设计记录

本文档记录 `scan.continuation` 的物理语义和程序 contract。

## 物理语义

`teacher warmstart` 和 `alpha continuation` 不是同一件事。

- `algorithm_params.init_mode: teacher` 只设置第 0 步 student 与 teacher 的初始 overlap。若 prior 是 zero-mean Gaussian，低 alpha 下 posterior mean 仍然可以被 prior shrink 拉回 0。
- `scan.continuation.enabled: true` 表示先在高 alpha 跑到最终状态，再把这个完整 algorithm state 作为下一个更低 alpha 的初始状态。这是在找 descending branch / hysteresis branch，不是改变 prior 或 Onsager 公式。

默认 continuation 约定：

```yaml
scan:
  continuation:
    enabled: true
    axis: alpha
    order: descending
    state_transfer: full_algorithm_state
    observation_policy: nested_prefix
    strict_state: true
    adaptive_controller_state: reset
```

`nested_prefix` 是物理敏感约定。dense mask 使用同一个 random permutation 的 prefix；spreading 和 tensor supergraph 使用已有 prefix `alpha_mask`。这样低 alpha 的观测集是高 alpha 的子集。

## State Contract

算法通过 `AlgorithmResult.continuation_state` 返回 runtime-only state。这个 state 不默认写入正式 artifact；它只在同一个 continuation chain 内传给下一个 alpha。

当前 state 语义：

- `agd`: `student_factors = W/X`
- `bigamp`: `student_factors = W/X`，`factor_variances = W_var/X_var`
- `bigamp_spreading` no-Onsager: `student_factors = W/X`，不传 `prev_s`
- `bigamp_spreading` fixed/adaptive Onsager: `student_factors + factor_variances + onsager_residual(prev_s/prev_svar)`
- `bigamp_tensor_parallel`: `tensor_factors + factor_variances`，若 Onsager 开启则附带 residual

当前正式支持范围是 `agd`、`bigamp`、`bigamp_spreading` bipartite graph、`bigamp_tensor_parallel`。`bigamp_spreading` general graph 的 state 可以表示，但正式 spreading metrics 尚未接入 `SuperGraphDataGeneral`，因此 continuation 会在 plan/runtime 阶段拒绝。`bigamp_tensor` serial reference path 和 `agd_tensor` experimental path 没有完整 state capture/inject；`strict_state=true` 下不能声明为 continuation-compatible。

降 alpha 时，spreading/tensor 的 residual 会乘当前 alpha mask，避免高 alpha 中已经移除的边继续进入 Onsager cavity。

## Execution Contract

Continuation 是 scan executor strategy：

- 每个非 alpha scan group 单独形成 chain。
- alpha 执行顺序强制 descending。
- 禁用 alpha folding；每个 alpha 点顺序执行。
- 保留 sample parallelism、precision profile、compile policy。
- adaptive damping 的 controller state 默认每个 alpha 重置；物理 state 继续传递。

每个 result point 会在 metadata 中记录：

- `continuation.enabled`
- `continuation.order`
- `continuation.source_alpha`
- `continuation.target_alpha`
- `continuation.state_transfer`
- `continuation.state_keys`
- `continuation.observation_policy`
- `continuation.physical_nested_observation`

正式 metric key 不因为 continuation 增加；raw/sign/gauge/Gram plots 继续复用现有 result cube 和 plotting 系统。
