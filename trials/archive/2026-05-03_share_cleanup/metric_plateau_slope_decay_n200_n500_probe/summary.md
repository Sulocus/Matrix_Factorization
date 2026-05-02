# Metric plateau slope-decay diagnostic

Small-N diagnostic trial for fitting how teacher-student latent overlap changes
decay over AMP iterations.

- Sizes: N = 200, 500; M = 50; S = 1.
- Alpha points: 0.30, 0.75, 0.90, 1.05, 1.50, 2.50.
- Fixed cap: 15000 steps.
- The metric plateau controller records Q_W/Q_X/R_W/R_X every 100 steps.
- `plateau_patience` is intentionally huge so this trial records history rather
  than stopping early.
