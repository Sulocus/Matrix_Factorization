# N=20000 alpha 0.75/0.90 trend-decay stop probe

Diagnostic trial for the current self-convergence stopper.

- N = 20000, M = 50, S = 1.
- alpha = 0.75, 0.90.
- Stopper uses `self_convergence_window_trend_decay`.
- Probe profile: interval 100, window 500, patience 2, abs tolerance 0.01, relative tolerance 0.05.
- Reference steps are generated only after observing the actual stop step.
