# N=20000 grid C serial early-stop scan

Full N=20000, M=50, S=1 scan using the trend-gate-v2 metric self-convergence
stopper.

- Alpha grid C: 0.00, 0.30, 0.60, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05,
  1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90,
  2.00, 2.10, 2.30, 2.50.
- Alpha folding is disabled to avoid batch members dragging each other after
  their own early-stop point.
- Output stores full tensors and disables heatmap/GIF.
