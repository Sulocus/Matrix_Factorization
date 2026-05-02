# AGD / BiGAMP Output Cosine Comparison

![QY cosine comparison](qy_cos_agd_bigamp_spreading.png)

This figure uses the native output cosine metric:

```math
Q_Y^{\cos}
= \frac{\langle \hat Y,Y^\star\rangle}
{\|\hat Y\|_2\|Y^\star\|_2}.
```

Configuration: `N1=N2=200`, `M=50`, `S=5`, Gaussian teacher. AGD runs for
200000 steps. Dense BiGAMP and no-Onsager spreading BiGAMP each run for
5000 steps. The CSV also keeps `Q_Y_mean`, `NMSE_Y_mean`, `Q_W_mean`, and
`Q_X_mean` from the same runs.
