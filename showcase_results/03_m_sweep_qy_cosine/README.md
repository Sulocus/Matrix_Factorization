# M Sweep Output Cosine

![M sweep QY cosine](m_sweep_qy_cosine.png)

This figure fixes `N1=N2=2000` and compares spreading BiGAMP results across
latent ranks `M`. It also includes `M=50, no Onsager, S=10` as a diagnostic
curve.

The plotted column is `Q_Y_COS`:

```math
Q_Y^{\cos}
= \frac{\langle \hat Y,Y^\star\rangle}
{\|\hat Y\|_2\|Y^\star\|_2}.
```

The CSV also includes `Q_Y_mean`, `FIT_Y_mean`, `NMSE_Y_mean`, `Q_W_mean`, and
`Q_X_mean` for cross-checks.
