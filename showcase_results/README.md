# Showcase Results

This folder contains the lightweight figures shown in the root README. The
plotted CSV files are kept beside each figure so the curves can be inspected or
redrawn without loading large tensors.

## Included Figures

- `01_qy_cosine_algorithm_comparison/`: output cosine comparison for AGD, dense
  BiGAMP, and no-Onsager spreading BiGAMP.
- `02_n_sweep_qy_cosine/`: output cosine trend as the square size `N` changes
  at fixed `M=50`.
- `03_m_sweep_qy_cosine/`: output cosine trend as the latent rank `M` changes
  at fixed `N1=N2=2000`, with one no-Onsager spreading diagnostic curve.
- `04_spreading_ablation/`: no-Gaussian-posterior spreading ablation used to
  diagnose the no-Onsager update.

## Metric

The main displayed metric is:

```math
Q_Y^{\cos}
= \frac{\langle \hat Y,Y^\star\rangle}
{\|\hat Y\|_2\|Y^\star\|_2}.
```

For the N and M sweep plots, the CSV column `Q_Y_COS` stores the exact value
used by the figure. It is computed from the saved output projection and
output-error summaries.
