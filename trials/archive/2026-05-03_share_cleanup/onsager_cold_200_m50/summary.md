# onsager_cold_200_m50

用途：在 `paper_sparse_sampling` normalization 下，对比 matrix spreading 的
no-Onsager baseline、corrected Onsager fixed beta、corrected Onsager adaptive
beta。

固定条件：

- Gaussian teacher。
- Rademacher `F`。
- `N1=N2=200, M=50`。
- `samples_per_alpha=10`。
- 完全 cold start：`algorithm_params.init_mode=random`。

当前 alpha 点是稀疏 smoke/diagnostic 版本：

```text
[0.5, 1.0, 1.5, 2.0, 3.0]
```

本 trial 不声称复现论文曲线；它用于检查 corrected Onsager path 是否能在目标
尺寸上稳定运行，并和 no-Onsager optimizer-like baseline 做第一轮对照。

输出设置为 `storage_mode: full`、`save_tensors: true`，因为后续 posthoc
scale-gauge aligned diagnostic 需要 teacher/student factor tensors。生成的
`.pt` artifact 位于 ignored trial run 目录，不能提交。

## 2026-04-28 首次运行

已完成一次 150-step diagnostic run：

```text
runs/trials/onsager_cold_200_m50/20260428_023821_bgs_N200_M50_onsa3-a5_1bd4ad
```

该 run 生成了 7 张 plot，并为 15 个 scan points 保存了
`artifacts/points/p*/results.pt`，可用于后续 posthoc gauge 诊断。

核心 `Q_Y_mean` 结果：

| policy | alpha=0.5 | alpha=1.0 | alpha=1.5 | alpha=2.0 | alpha=3.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_onsager | 0.0027 | 0.0205 | 0.1044 | 0.4211 | 0.9846 |
| onsager_fixed_beta005 | 0.0031 | 0.0121 | 0.0456 | 0.1771 | 0.6894 |
| onsager_adaptive_beta005 | 0.0043 | 0.0164 | 0.1669 | 0.1985 | 1.0005 |

初步读法：

- no-Onsager baseline 仍然在该 150-step 设置下更强，尤其 alpha=2.0。
- corrected Onsager fixed beta 已经能稳定运行，但同样步数下偏慢。
- adaptive beta 不是完整 reference backtracking；alpha=1.5 有改善，
  alpha=2.0 反而弱于 baseline，需要后续加入 beta/cost/zvar/pvar diagnostics。
- 该 run 不是论文曲线复现：graph 仍是 average-degree，不是 exact
  `c=alpha M`；步数也只是 diagnostic 级别。

注意：该首次 run 没有保存真实 `W_teacher/X_teacher` 共享 artifact，因此不能
可靠计算 posthoc scale-gauge aligned diagnostic。从 config/seed 重建 teacher
在 canonical child run 下可能得到错误 teacher。

## 2026-04-28 teacher artifact 修正后重跑

保存层新增 `artifacts/teacher_factors.pt` 后重跑：

```text
runs/trials/onsager_cold_200_m50/20260428_024747_bgs_N200_M50_onsa3-a5_1bd4ad
```

新增输出：

```text
artifacts/teacher_factors.pt
plots/posthoc_scale_gauge/scale_gauge_metrics.csv
plots/posthoc_scale_gauge/qw_scale_gauge.png
plots/posthoc_scale_gauge/qx_scale_gauge.png
plots/posthoc_scale_gauge/qwx_scale_gauge.png
plots/posthoc_scale_gauge/gauge_magnitude.png
```

核心 `Q_Y_mean`：

| policy | alpha=0.5 | alpha=1.0 | alpha=1.5 | alpha=2.0 | alpha=3.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_onsager | 0.0027 | 0.0204 | 0.1046 | 0.4398 | 0.9848 |
| onsager_fixed_beta005 | 0.0031 | 0.0121 | 0.0456 | 0.1771 | 0.6915 |
| onsager_adaptive_beta005 | 0.0043 | 0.0168 | 0.1669 | 0.1967 | 1.0005 |

posthoc scale-gauge aligned `Q_WX`：

| policy | alpha=0.5 | alpha=1.0 | alpha=1.5 | alpha=2.0 | alpha=3.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_onsager | 0.0297 | 0.1164 | 0.3078 | 0.6514 | 0.9931 |
| onsager_fixed_beta005 | 0.0274 | 0.0807 | 0.1763 | 0.3701 | 0.8282 |
| onsager_adaptive_beta005 | 0.0370 | 0.0985 | 0.3773 | 0.3972 | 1.0001 |

scale-gauge aligned 指标与 sign-aligned 指标接近，median `|log |g||` 大约
`0.01-0.06`。这说明该小 trial 中 raw `Q_W/Q_X` 低主要来自 sign/coordinate
gauge，而不是明显连续 scale gauge 漂移。
