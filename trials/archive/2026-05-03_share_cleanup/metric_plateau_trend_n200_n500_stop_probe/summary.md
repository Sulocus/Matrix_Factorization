# N=200/N=500 trend-decay stop probe

Diagnostic trial for the updated self-convergence stopper.

- N = 200, 500; M = 50; S = 1.
- alpha = 0.0, 0.75, 0.9, 1.0, 1.05, 1.2, 1.5, 2.1, 2.5.
- Stopper uses `self_convergence_window_trend_decay`.
- Probe profile: interval 100, window 500, patience 2, abs tolerance 0.01, relative tolerance 0.05.
- Reference trials are generated only after observing the actual stop steps.
