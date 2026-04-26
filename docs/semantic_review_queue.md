# Semantic Review Queue

这份文档记录仍需要之后确认的 metric / physics 解释问题。Projection metric v3 已经把 active formal names 固定下来；这里保留的是后续论文图例、legacy 结果解释、tensor parity 的审查项。

## 已决定的 active formal names

- `Q_Y`：measurement/output absolute projection。
- `Q_Y_observed`：observed/training measurement projection。
- `Q_Y_unobserved`：heldout/unobserved measurement projection。
- `Q_W` / `Q_X`：matrix latent factor coordinate projection。
- `Q_N`：tensor latent node/spin/factor coordinate projection。
- `Q_W_GRAM_ROOT` / `Q_X_GRAM_ROOT`：sqrt baseline-corrected Gram diagnostic。

## 仍需人工确认

- `Q_Y` 的论文图例是否只显示 `Q_Y`，还是在多空间图中显示 `Q_Y^{obs}`、`Q_Y^{unobs}`、`Q_Y^{full}`。
- `Q_W/Q_X` 在 gauge/rotation 敏感时是否默认放进主图，还是默认只画 `Q_W_GRAM_ROOT/Q_X_GRAM_ROOT`。
- `Q_N` 是否未来也需要 Gram-root 或 permutation-invariant diagnostic。
- spreading/tensor heldout measurement set 的数量默认等于 observed edge count 是否符合之后的物理比较需求。
- tensor serial 和 tensor parallel 在 teacher scale、alpha normalization、F distribution、seed partition 上通过 parity 审查前，不能把两条路径的曲线当成同一数值实现。

## Legacy 解释

- 旧 schema 的 `Q_Y_mean` 可能是 cosine 或 reconstruction-quality diagnostic；不能和 schema v3 的 projection `Q_Y_mean` 直接比较。
- `MSE` / `Gen_Error` 只作为 legacy/debug/loss 解释，不再是 formal result metric。
- `physical_overlap_*` 是旧 projection 名，active schema 已迁移到 `Q_Y/Q_W/Q_X`。
- `Q_W_prime/Q_X_prime` 是旧 baseline-corrected Gram 名，active schema 已迁移到 `Q_W_GRAM_ROOT/Q_X_GRAM_ROOT`。
