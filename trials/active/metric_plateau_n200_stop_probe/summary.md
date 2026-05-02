# metric_plateau_n200_stop_probe

用途：诊断 self-convergence stop 在 `N=200, M=50, S=1` 下是否会在合理步数停止。

该 trial 使用临时高上限 `max_steps=20000`，不是正式 sweep，也不进入正式汇总。
