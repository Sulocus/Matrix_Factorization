# metric_plateau_m_sweep_2k_grid_c

固定 `N1=N2=2000`，扫描 rank `M`：

- `M=20,50,80,120,150`
- alpha grid C
- `S=1`
- fixed Onsager warm start
- `paper_sparse_sampling`
- `precision_profile=safe`
- 启用 `self_convergence_window_trend_decay` metric plateau stop

输出保留 full tensors，关闭 heatmap/GIF，并生成 `Q_W_mean` vs `alpha` by `rank_M`。
