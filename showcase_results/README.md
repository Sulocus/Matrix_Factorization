# Showcase Results

Lightweight plots selected from local `dev` runs.

## Result Folders

- `01_warmstart_hysteresis`: warmstart hysteresis.
- `02_onsager_fixed_vs_none`: fixed Onsager vs no Onsager.
- `03_hot_start_onsager`: hot-start Onsager.

## Sign And Gauge

The matrix factorization has an $M$-dimensional gauge vector
$\boldsymbol{k} = (k_1,\ldots,k_M)$:

$$
W_{:m} \mapsto k_m W_{:m},
\qquad
X_{m:} \mapsto k_m^{-1} X_{m:}.
$$

- sign: restrict $|k_m|=1$ and align the remaining $\pm 1$ sign by the teacher/student inner product.
- gauge: use the full $\boldsymbol{k}$ and choose each $k_m$ by an argmin over the W/X mismatch.

These views do not guarantee early-training physical correctness. They only remove later symmetry breaking and teacher/student gauge mismatch.
