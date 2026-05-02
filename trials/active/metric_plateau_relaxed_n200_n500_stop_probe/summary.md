# metric_plateau_relaxed_n200_n500_stop_probe

用途：诊断 relaxed self-convergence stop 在 `N=200/500, M=50, S=1` 下的停止步数和参考误差。

该 trial 使用临时高上限 `max_steps=20000`，不是正式 sweep，也不进入正式汇总。
