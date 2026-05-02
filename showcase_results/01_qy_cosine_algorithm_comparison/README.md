# Algorithm Comparison

Output and sign-aligned latent overlap for four algorithms on the same
cold-start problem.

- `N = 200 x 200`, `M = 50`
- `initialization = cold start`
- Metric: `Q_Y^cos`
- Diagnostic metric: `Q_W_SIGN_GAUGE`
- Spreading curves use the scan-wide supergraph/F domain for `Q_Y^cos`, so
  alpha folding does not change the metric definition.

Files:

- `qy_cos_algorithm_comparison.png`
- `qy_cos_algorithm_comparison.csv`
- `qw_algorithm_comparison.png`
- `qw_algorithm_comparison.csv`
- `qw_sign_gauge_algorithm_comparison.png`
- `qw_sign_gauge_algorithm_comparison.csv`
