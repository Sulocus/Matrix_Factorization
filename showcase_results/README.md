# Showcase Results

This folder contains the lightweight figures shown in the root README. The
plotted CSV files are kept beside each figure so the curves can be inspected or
redrawn without loading large tensors.

## Included Figures

- `01_qy_cosine_algorithm_comparison/`: output cosine comparison for `AGD`,
  `BiGAMP (F = 1)`, `BiGAMP (F = Ising, no Onsager)`, and
  `BiGAMP (F = Ising, Onsager)`.
- `02_n_sweep_qw/`: `Q_W` as the square size changes at fixed `M = 50`.
- `03_m_sweep_qw/`: `Q_W` as the latent rank changes at fixed
  `N = 2000 x 2000`.

## Metrics

```math
Q_Y^{\cos}
=
\frac{
\sum_a Y_a Y^{*}_a
}{
\sqrt{
\left(\sum_a Y_a^2\right)
\left(\sum_a (Y^{*}_a)^2\right)
}
}
```

```math
Q_W
=
\frac{1}{N_1 M}
\sum_{i=1}^{N_1}
\sum_{\mu=1}^{M}
W_{i\mu} W^{*}_{i\mu}
```

```math
Q_X
=
\frac{1}{M N_2}
\sum_{\mu=1}^{M}
\sum_{j=1}^{N_2}
X_{\mu j} X^{*}_{\mu j}
```
