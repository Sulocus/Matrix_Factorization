# Showcase Results

Selected lightweight plots from local `dev` runs.

## Folders

- `01_warm_start`: warm start.
- `02_fixed_onsager_vs_no_onsager`: fixed Onsager vs no Onsager.
- `03_warm_start_onsager`: warm start Onsager, overlap `0.2`.

## Metric Definitions

The basic overlaps are absolute projections onto the teacher, normalized by the
teacher norm squared:

$$
Q_Y =
\frac{|\langle Y_s,Y_t\rangle|}{\langle Y_t,Y_t\rangle},
\qquad
Q_W =
\frac{|\langle W_s,W_t\rangle|}{\langle W_t,W_t\rangle}.
$$

These projection values are not clipped, so values above `1` are possible when
the student has a different scale.

## Sign And Gauge

Matrix factorization has an $M$-dimensional gauge vector
$\boldsymbol{k} = (k_1,\ldots,k_M)$:

$$
W_{:m} \mapsto k_m W_{:m},
\qquad
X_{m:} \mapsto k_m^{-1} X_{m:}.
$$

The sign view only uses the sign of each $k_m$ and ignores its magnitude. For W
this is equivalent to choosing the sign from each teacher/student inner product:

$$
Q_W^{\mathrm{sign}} =
\frac{\sum_{m=1}^M |\langle W_{s,:m}, W_{t,:m}\rangle|}
{\sum_{m=1}^M \|W_{t,:m}\|^2}.
$$

The scale-gauge view fits a continuous scalar for each channel:

$$
g_m^\star =
\arg\min_{g\ne 0}
\left(
\|g W_{s,:m}-W_{t,:m}\|^2
+ \|g^{-1} X_{s,m:}-X_{t,m:}\|^2
\right).
$$

With
$a_m=\|W_{s,:m}\|^2$,
$b_m=\langle W_{s,:m},W_{t,:m}\rangle$,
$c_m=\|X_{s,m:}\|^2$, and
$d_m=\langle X_{s,m:},X_{t,m:}\rangle$, the optimized part is

$$
a_m g^2 - 2b_m g + \frac{c_m}{g^2} - \frac{2d_m}{g}.
$$

The implementation checks real nonzero roots of

$$
a_m g^4 - b_m g^3 + d_m g - c_m = 0
$$

plus fallback candidates $\pm\sqrt{c_m/a_m}$ and $\pm 1$, then selects the
candidate with the smallest objective. This does not force $|g_m|$ toward `1`;
the reported gauge magnitude uses median $\left|\log |g_m|\right|$ only as a
side measurement.

After alignment,

$$
Q_W^{\mathrm{gauge}} =
\frac{\sum_m g_m^\star \langle W_{s,:m},W_{t,:m}\rangle}
{\sum_m \|W_{t,:m}\|^2},
\qquad
Q_X^{\mathrm{gauge}} =
\frac{\sum_m (g_m^\star)^{-1} \langle X_{s,m:},X_{t,m:}\rangle}
{\sum_m \|X_{t,m:}\|^2}.
$$

The `qwqx_gauge` plot shows
$\frac{1}{2}(Q_W^{\mathrm{gauge}}+Q_X^{\mathrm{gauge}})$.

These views do not guarantee early-training physical correctness. They only
remove later symmetry breaking and teacher/student gauge mismatch.
