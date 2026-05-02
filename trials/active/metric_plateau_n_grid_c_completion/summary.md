# metric_plateau_n_grid_c_completion

补齐 square N-scan 的 grid C 数据：

- 新跑：`N=200,500,1000,2000,5000,10000`
- 复用：已有 `N=20000` grid C serial run
- 固定 `M=50`, `S=1`
- 使用 `bigamp_spreading`、fixed Onsager warm start、`paper_sparse_sampling`、
  `precision_profile=safe`
- 启用 `self_convergence_window_trend_decay` metric plateau stop

输出保留 full tensors，关闭 heatmap/GIF，并生成 `Q_W_mean` vs `alpha` by `size_N`。
