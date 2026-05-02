# 2026-04-30 Metric Plateau N/M Sweep 计划

## 目标

用当前 `bigamp_spreading` 的 `self_convergence_window_trend_decay`
teacher-assisted early stop，补齐两类 exploratory plot 数据：

1. `M=50` 固定时的 square N-scan：`N=200,500,1000,2000,5000,10000,20000`。
2. `N1=N2=2000` 固定时的 M-scan：`M=20,50,80,120,150`。

这两个实验都沿用当前 schema v5 metric 语义，只比较当前正式字段，不和旧
schema v3/v4 的同名字段混用。

## 复用规则

- 已有完整 `N=20000, M=50` grid C 数据，直接复用：
  `artifacts/trials/metric_plateau_trend_n20000_grid_c_serial/20260429_220711_grid_c_serial_bgs_N20000_M50_a24_S1_steps60000_1fc77e`
- `N=200/500/1000/2000/5000/10000` 没有同一 grid C、同一 early-stop 策略的完整
  square N-scan 数据，因此统一补跑。
- 旧的 probe/reference/partial run 只作为诊断记录，不进入本次正式汇总。

## 公共实验条件

- `algorithm = bigamp_spreading`
- Gaussian teacher, Ising `F`
- `paper_sparse_sampling`
- `precision_profile = safe`
- fixed Onsager warm start：`onsager_correction=true`, `damping=0.05`,
  `init_mode=teacher`, `init_overlap=0.5`
- `S=1`
- output 使用 `storage_mode=full`, `save_tensors=true`, `enable_heatmap=false`

## Early Stop 参数

沿用当前已验证的 exploratory profile：

- `use_metric_plateau_stop=true`
- `plateau_check_interval=100`
- `plateau_window_steps=500`
- `plateau_patience=2`
- `plateau_abs_tol=0.01`
- `plateau_rel_tol=0.05`
- `plateau_min_steps=0`
- `plateau_monitor=teacher_latent_overlap_qw_qx`

实际策略为 `self_convergence_window_trend_decay`。它是 teacher-assisted convergence
diagnostic，不改变 AMP 更新、Onsager、damping、denoiser 或 variance 公式。

## N-scan

新增 trial：`metric_plateau_n_grid_c_completion`

- 扫描 `N=200,500,1000,2000,5000,10000`
- 复用已有 `N=20000`
- `M=50`
- alpha grid C：
  `0.00,0.30,0.60,0.75,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00,2.10,2.30,2.50`
- `max_steps=60000` 作为宽松 guard；若频繁撞 cap，说明 stopper 仍需复查，而不是把
  cap 当成有效收敛。
- 生成：
  - 新 run 自带 `Q_W_mean` vs alpha by `size_N`
  - 手动汇总图：加入已有 `N=20000` 的同图对比
  - CSV：每个 N/alpha 的 `Q_W/Q_X/steps_run/stop_reason`

## M-scan

新增 trial：`metric_plateau_m_sweep_2k_grid_c`

- 固定 `N1=N2=2000`
- 扫描 `M=20,50,80,120,150`
- 使用同一 alpha grid C
- `max_steps=60000`
- 生成：
  - `Q_W_mean` vs alpha by `rank_M`
  - CSV：每个 M/alpha 的 `Q_W/Q_X/steps_run/stop_reason`

## 执行顺序

1. 写入两个 trial 配置、manifest、summary，并登记 `trials/registry.yaml` 和 source inventory。
2. 运行：
   - `mf validate trials/active/metric_plateau_n_grid_c_completion/config.yaml`
   - `mf validate trials/active/metric_plateau_m_sweep_2k_grid_c/config.yaml`
   - `python -m pytest -q tests/test_trial_contract.py tests/test_source_inventory.py`
3. 先跑 N-scan completion。
4. 汇总 N-scan，叠加已有 `N=20000`。
5. 再跑 M-scan。
6. 汇总 M-scan。
7. 最后中文汇报，并用 Discord 发送两行以内短通知。
