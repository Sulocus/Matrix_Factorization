# N=20000 alpha 0.75/0.90 relaxed reference

Fixed-step reference for the paired relaxed self-convergence stop probe.

- N = 20000, M = 50, S = 1.
- alpha = 0.75, 0.90.
- Reference steps = 13700, chosen from `ceil(1.5 * 9100 / 100) * 100`.
- Early stop is disabled; all other teacher, spreading, seed, precision, and initialization settings match the probe.
