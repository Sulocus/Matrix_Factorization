# metric_plateau_m50_no_onsager_s10_grid_c

这个 trial 用于给 `N1=N2=2000` 的 M-scan 图增加一条 no-Onsager 对照曲线：

- `M=50`
- `S=2`
- alpha grid C
- `paper_sparse_sampling`
- `precision_profile=safe`
- `spreading.onsager_correction=false`
- 启用同一套 `self_convergence_window_trend_decay` metric plateau stop

对照曲线会单独标记为 `M50_no_onsager_S2`，不和已有 `M=50, S=1,
fixed Onsager` 曲线混淆。
