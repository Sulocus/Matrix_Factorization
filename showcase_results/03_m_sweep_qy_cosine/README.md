# M Sweep Output Cosine

![M sweep QY cosine proxy](m_sweep_qy_cosine_proxy.png)

这张图固定 \(N_1=N_2=2000\)，比较不同 latent rank \(M\) 的 spreading BiGAMP 结果，并加入 `M=50, no Onsager, S=10` 作为诊断对比。

原始 scan 是 schema v5，因此 \(Q_Y^{\cos}\) 是由 \(Q_Y\)+`NMSE_Y` 派生的 proxy。`M=200` 当时只留下了 \(Q_W\) 诊断 summary，没有完整 \(Q_Y\) final metrics，所以没有放入这张图。
