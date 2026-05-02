# AGD / BiGAMP Output Cosine Comparison

![QY cosine comparison](qy_cos_agd_bigamp_spreading.png)

这张图使用 schema v6 的原生 `Q_Y_COS_mean`：

$$
Q_Y^{\cos}
= \frac{\langle \hat Y,Y^\star\rangle}
{\|\hat Y\|_2\|Y^\star\|_2}.
$$

配置为 \(N_1=N_2=200\)、\(M=50\)、\(S=5\)、Gaussian teacher。AGD 运行 200000 step，dense BiGAMP 和 spreading BiGAMP no-Onsager 各运行 5000 step。CSV 中保留了同一 run 的 `Q_Y_mean`、`NMSE_Y_mean`、`Q_W_mean`、`Q_X_mean` 以便复查。
