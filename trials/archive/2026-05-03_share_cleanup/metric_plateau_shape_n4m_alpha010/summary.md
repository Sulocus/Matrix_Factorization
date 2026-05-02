# metric_plateau_shape_n4m_alpha010

对比固定面积 `N1*N2=4e6` 下的矩阵长宽比例：

- `rect_4000x1000`
- `square_2000x2000`

配置使用 `bigamp_spreading`、fixed Onsager warm start `init_overlap=0.5`、
`paper_sparse_sampling`、`precision_profile=safe`，并启用
`self_convergence_window_trend_decay` metric plateau stop。

输出保留 full tensors，关闭 heatmap/GIF，并生成同图 `Q_W_mean` vs `alpha`
曲线。
