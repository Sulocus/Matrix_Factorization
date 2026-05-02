# metric_plateau_shape_40k10k_alpha015

`40000x10000, M=50, S=1` 的矩形大尺寸 shape probe。

使用同一套 `bigamp_spreading`、fixed Onsager warm start `init_overlap=0.5`、
`paper_sparse_sampling`、`precision_profile=safe` 和
`self_convergence_window_trend_decay` metric plateau stop。

alpha 网格为 `0.0, 0.1, ..., 1.5, 2.5`。结果用于和已有
`20000x20000` grid-C run 叠图比较。

alpha folding 关闭：第一次 batched run 在 metrics 计算阶段触发 CUDA driver
error，因此正式重跑采用每个 alpha 独立执行，优先保证结果可靠和 stop step 清晰。
