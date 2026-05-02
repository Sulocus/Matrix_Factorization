# N=200 trend-decay reference

Fixed-step reference for the N=200 portion of the trend-decay stop probe.

- N = 200, M = 50, S = 1.
- alpha = 0.0, 0.75, 0.9, 1.0, 1.05, 1.2, 1.5, 2.1, 2.5.
- Reference steps = 12000, chosen from `ceil(1.5 * 8000 / 100) * 100`.
- Early stop is disabled.
