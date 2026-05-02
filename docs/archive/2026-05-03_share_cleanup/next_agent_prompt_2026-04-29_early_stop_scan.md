# Next Agent Prompt: BiGAMP Early Stop + Infinite-Size Scan

把下面这段完整发给新 agent。

```text
你接手的是 /home/sucia/Matrix_Factorization，请在 dev 分支工作。

先不要改代码，也不要直接开大实验。你需要先自己只读扫描 repo，然后再读
handoff，把自己的理解和 handoff 合并。

必须先读：

1. /home/sucia/Matrix_Factorization/AGENTS.md
2. /home/sucia/Matrix_Factorization/docs/project_handoff_2026-04-29.md
3. /home/sucia/Matrix_Factorization/docs/project_handoff_2026-04-28.md
4. /home/sucia/Matrix_Factorization/docs/project_handoff_2026-04-27.md
5. /home/sucia/Matrix_Factorization/docs/METRICS_GUIDE.md
6. /home/sucia/Matrix_Factorization/docs/scan_system_contract.md
7. /home/sucia/Matrix_Factorization/docs/theory/bigamp_onsager_damping_audit.md

第一步检查：

```bash
cd /home/sucia/Matrix_Factorization
git status --short --branch
mf validate src/matrix_factorization/config.yaml
python -m pytest -q tests/test_contract_parameter_specs.py tests/test_config_contract.py
```

当前最重要任务：

- 为 `bigamp_spreading` 设计并实现 teacher-student metric plateau early stop。
- 用小尺寸验证它不会误停。
- 验证后用高精度 alpha grid C 继续 fixed Onsager warm-start 大尺寸 scan。

如果你可以调用 sub-agent，请并行分四路：

1. 代码接入审计：
   - 检查 `src/matrix_factorization/modules/algorithms/bigamp/spreading.py`
   - 确认 formal batch loop 每步可拿到哪些状态：
     `W_flat/X_flat/W_var_flat/X_var_flat/prev_s/prev_svar`
   - 找到最低成本计算 `Q_W/Q_X/R_W/R_X` 的位置。

2. 理论/文献审计：
   - 读 BiGAMP damping 相关材料。
   - 参考 arXiv:1310.2632 和 arXiv:1412.2005。
   - 明确 early stop 是 convergence diagnostic，不改变 AMP/Onsager 公式。

3. 测试设计：
   - 设计 synthetic history unit tests。
   - 设计小尺寸 runtime tests。
   - 验证低 alpha 慢漂移不会被短窗口误判。

4. Scan 参数设计：
   - 使用当前 handoff 里的 calibration 结果。
   - 设计 high-resolution alpha grid C。
   - 避免再把 `N=20000` 的 `64000` 步机械套给所有小 N。

如果 sub-agent 不可用，就按同样四个视角串行完成，不要跳过。

当前 metric schema：

```text
schema_version = 5
metric_definition_profile = projection_qy_physical_latent_v2
```

当前正式 metric 语义：

```text
Q_W/Q_X = fixed-denominator physical latent overlap
Q_Y = output absolute projection
FIT_Y = 1 - NMSE_Y
Q_W_SIGN_GAUGE/Q_X_SIGN_GAUGE = sign-gauge diagnostic
Q_W_SCALE_GAUGE/Q_X_SCALE_GAUGE/Q_WX_SCALE_GAUGE = scale-gauge diagnostic
Q_W_COS_ROOT/Q_X_COS_ROOT = cos-root diagnostic
```

不要把旧 schema v3/v4 的同名字段和当前 schema v5 静默比较。

当前已经完成的 calibration：

```text
N=30000,M=50,steps=20 smoke passed
N=10000,M=100,steps=20 smoke passed

N=2000,M=50:
  steps 5000/10000/20000/40000
  40000 vs 20000 max Q_W delta ~= 0.007457
  considered converged under 0.01 criterion

N=20000,M=50:
  steps 8000/16000/32000/64000
  32000 vs 16000 max Q_W delta ~= 0.012804
  64000 vs 32000 max Q_W delta ~= 0.0100105
  near-threshold finite-time, not strictly plateau
```

被中断的 run：

```text
artifacts/trials/infinite_n_sweep_qw_64000/
  20260429_121048_n_sweep_qw_64k_finite_time_bgs_N2000_M50_expe1-size5-a15_S1_steps64000_b93781
```

它只完整完成了 `N=2000` 和 `N=5000`，`N=10000` 中途停止。不能把这次
partial 和后续新 grid 混成正式 N sweep。

Early stop 推荐接口：

```yaml
algorithm_params:
  use_metric_plateau_stop: true
  plateau_check_interval: 2000
  plateau_window_steps: 16000
  plateau_patience: 2
  plateau_abs_tol: 0.003
  plateau_min_steps: 16000
  plateau_signal: teacher_latent_overlap
```

推荐停止判据：

```text
Delta(t) = max(
  |Q_W(t) - Q_W(t - W)|,
  |Q_X(t) - Q_X(t - W)|,
  |R_W(t) - R_W(t - W)|,
  |R_X(t) - R_X(t - W)|
)
```

其中 `W = plateau_window_steps`。连续 `plateau_patience` 次满足
`Delta(t) < plateau_abs_tol` 才停。

注意：

- 不要只比较相邻 check interval；低 alpha 会慢漂移。
- early-stop check 只算 latent metrics，不算 `Q_Y`、sign gauge、scale gauge、
  heatmap。
- 这是 teacher-student 实验加速/诊断工具，不是真实无监督算法的一部分。
- metadata 必须写 `teacher_assisted: true` 和实际 stop reason/stop step。

Hard contract 要求：

- 新 YAML 参数必须有 `ParameterSpec`。
- 必须进入 config parsing、effective config trace、metadata/result visibility。
- 不消费该参数的 route 必须 warning 或 strict-mode error。
- 新 metric/output/analyzer 必须先有 spec 再注册。
- 任何新增源文件如果落在 classified area，要更新 source inventory。
- 不要绕过 `mf validate` 和 contract tests。

小尺寸测试计划：

```text
N = 200, 500, 1000, 2000
M = 50
S = 1
alpha = 0.0, 0.75, 0.9, 1.0, 1.05, 1.2, 1.5, 2.1, 2.5
fixed_onsager_warm_start_050
Gaussian teacher
Ising F
paper_sparse_sampling
precision_profile = safe
```

验收：

- early stop run 与 fixed full-step run 的主要 sentinel：
  `|Q_W_early - Q_W_full| < 0.01`。
- 高 alpha 应明显提前停。
- transition 区不能过早停。
- 低 alpha 慢漂移不能被短窗口误判。
- 无 NaN/Inf。

大尺寸 scan 只在小测试通过后运行。推荐 alpha grid C：

```text
0.00, 0.30, 0.60, 0.75,
0.85, 0.90, 0.95,
1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30,
1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10,
2.30, 2.50
```

大尺寸输出建议：

```yaml
output:
  storage_mode: full
  save_tensors: true
  enable_heatmap: false
```

这样后续可以补算 `Q_X`、sign gauge、scale gauge、`Q_Y/FIT_Y`。不要保存
per-step trajectory，不要开 heatmap/GIF。

实验数据规则：

- quick/debug/trial 只能用 `trials/active/<trial_key>/config.yaml`。
- trial 输出进 `artifacts/trials/`、`runs/trials/` 或 `results/trials/`。
- 正式 run 只有用户明确要求时才写 `runs/<run_name>/`。
- 不提交 `runs/`、`results/`、`artifacts/` 或 `.pt`。

最低验收命令：

```bash
python -m pytest -q tests/test_contract_parameter_specs.py tests/test_config_contract.py
python -m pytest -q tests/test_spreading_batch_metrics.py tests/test_bigamp_onsager_convention.py
mf validate src/matrix_factorization/config.yaml
```

如果改了 hard-interface 或 cross-cutting runner/contract，最后跑：

```bash
python -m pytest -q
```
```

