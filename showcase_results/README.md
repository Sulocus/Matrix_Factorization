# Showcase Results

这里放的是可以直接打开查看的轻量展示图。完整 trial 输出和大 tensor 不在 `main` 分支。

## 当前主展示

- `01_qy_cosine_algorithm_comparison/`：AGD、dense BiGAMP、spreading BiGAMP no-Onsager 的 \(Q_Y^{\cos}\) 对比。数据来自 2026-05-03 重新运行的 schema v6 metrics。
- `02_n_sweep_qy_cosine/`：固定 \(M=50\) 的 \(N\) scan 趋势图。原始 scan 是 schema v5，因此图中的 \(Q_Y^{\cos}\) 是由 \(Q_Y\)+`FIT_Y` 派生的 proxy。
- `03_m_sweep_qy_cosine/`：固定 \(N_1=N_2=2000\) 的 \(M\) scan 趋势图。原始 scan 是 schema v5，因此图中的 \(Q_Y^{\cos}\) 是由 \(Q_Y\)+`NMSE_Y` 派生的 proxy。
- `04_spreading_ablation/`：临时去掉 Gaussian posterior shrinkage 假设的 spreading no-Onsager 消融诊断图；主程序源码没有保留该改动。

## Metric Notes

当前主图优先看：

$$
Q_Y^{\cos}
= \frac{\langle \hat Y,Y^\star\rangle}
{\|\hat Y\|_2\|Y^\star\|_2}.
$$

旧 scan proxy 使用：

$$
Q_{Y,\mathrm{proxy}}^{\cos}
\approx
\frac{Q_Y}{\sqrt{\mathrm{NMSE}_Y-1+2Q_Y}}.
$$

这个 proxy 用于展示旧 scan 的曲线形状，不替代 schema v6 的原生 `Q_Y_COS_mean`。
