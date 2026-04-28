# Showcase Results

Selected lightweight plots from local `dev` runs.

## Folders

- `01_warm_start_hysteresis`: warm start hysteresis.
- `02_fixed_onsager_vs_no_onsager`: fixed Onsager vs no Onsager.
- `03_warm_start_onsager`: warm start Onsager, overlap `0.2`.

## Sign And Gauge

Matrix factorization has an $M$-dimensional gauge vector
$\boldsymbol{k} = (k_1,\ldots,k_M)$:

$$
W_{:m} \mapsto k_m W_{:m},
\qquad
X_{m:} \mapsto k_m^{-1} X_{m:}.
$$

- sign: restrict $|k_m|=1$ and align the remaining $\pm 1$ sign by the teacher/student inner product.
- gauge: use the full $\boldsymbol{k}$ and choose each $k_m$ by an argmin over the W/X mismatch.

These views do not guarantee early-training physical correctness. They only remove later symmetry breaking and teacher/student gauge mismatch.
