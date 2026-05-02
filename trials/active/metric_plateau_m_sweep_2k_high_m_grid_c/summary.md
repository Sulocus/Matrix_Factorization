# metric_plateau_m_sweep_2k_high_m_grid_c

在已有 `N1=N2=2000`、`M=20,50,80,120,150` 的 M-scan 基础上，补跑高 rank：

- `M=200,300,400`
- alpha grid C
- `S=1`
- fixed Onsager warm start
- `paper_sparse_sampling`
- `precision_profile=safe`
- 启用 `self_convergence_window_trend_decay` metric plateau stop

目标是确认高 rank 是否能放进当前 GPU，并为后续合并 M-scan 图补充右侧点。

`M=400` 在默认 75% allocation 下会被 preflight 拦下；估算最大 single-alpha full-S
batch 约 `22.07 GB`，物理可用显存约 `31.34 GB`。本 trial 因此显式使用
`scan.execution.target_utilization=0.95`、`max_allocated_gb=30.0` 做 tight-memory
feasibility。

`M=500` 已做配置 preflight：当前 `safe` 精度下最大 single-alpha full-S batch 约需
`34.44 GB`，超过当前 GPU 可用显存，因此不进入本轮正式 scan。
