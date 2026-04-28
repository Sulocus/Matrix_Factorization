# Showcase Results

Selected lightweight plots from local `dev` runs. Large tensors and checkpoints
are not included.

## Folders

- `01_warm_start`: warm-start comparison.
- `02_fixed_onsager_vs_no_onsager`: fixed Onsager vs no Onsager.
- `03_warm_start_onsager`: fixed Onsager with warm-start overlap `0.2`.

## Metrics

For each scan point and each sample, the code first computes an absolute
teacher projection, then the plotted curve uses the sample mean:

$$
q_Y^{(s)}=\frac{|\langle Y_s^{(s)},Y_t^{(s)}\rangle|}
{\langle Y_t^{(s)},Y_t^{(s)}\rangle},
\qquad
Q_Y=\frac{1}{S}\sum_{s=1}^S q_Y^{(s)}.
$$

The same convention is used for `Q_W`, with `W_student` projected onto
`W_teacher`. Projection values are not clipped, so values above `1` can reflect
scale mismatch.

## Gauge Views

The factorization has a per-channel scale freedom
`W[:,m] -> k_m W[:,m]`, `X[m,:] -> k_m^{-1} X[m,:]`.

- `qw_sign`: ignores the magnitude of `k_m` and aligns only the sign of each
  W-channel inner product.
- `qwqx_gauge`: fits one scalar per channel by the objective

$$
g_m^\star=\arg\min_{g\ne0}
\left[
\|gW_{s,:m}-W_{t,:m}\|^2+
\|g^{-1}X_{s,m:}-X_{t,m:}\|^2
\right].
$$

These gauge views are diagnostics only. They do not prove that early training is
physically correct; they only remove later sign/scale mismatch between teacher
and student factors.
