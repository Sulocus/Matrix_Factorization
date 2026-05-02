# Spreading No-Gaussian-Posterior Ablation

![QY cosine ablation](qy_cosine_with_no_gaussian_posterior_temp.png)

这个目录保留一次临时 in-process 消融：在 no-Onsager spreading 的更新里去掉 Gaussian posterior shrinkage 假设，并保持其它配置不变。运行结束后 monkey patch 已恢复，主程序没有保留该公式改动。

目录中两张图分别是 \(Q_Y\) projection 和 \(Q_Y^{\cos}\)。它们用于定位 spreading no-Onsager 路线的 metric/公式敏感性，不作为正式算法改动结论。
