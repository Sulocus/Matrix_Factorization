# Project Handoff 2026-04-29

本文件是 2026-04-29 当前对话窗口的交接文档。它不替代旧文档，而是按日期
追加最新状态。新的 agent 可以先读本文件，再按需向前追溯。

推荐阅读顺序：

1. 快速接手：先读 `docs/project_handoff_2026-04-29.md`。
2. 追溯背景：再读 `docs/project_handoff_2026-04-28.md` 和
   `docs/project_handoff_2026-04-27.md`。
3. 当前任务 prompt：读
   `docs/next_agent_prompt_2026-04-29_early_stop_scan.md`。
4. 固定规范：始终遵守 `AGENTS.md`、`docs/scan_system_contract.md`、
   `docs/METRICS_GUIDE.md`、`docs/theory/bigamp_onsager_damping_audit.md`。

当前开发分支：`dev`。主开发包：`src/matrix_factorization`。

## 1. 当前工作树状态

最近一次检查：

```text
## dev...origin/dev
?? trials/active/fixed_onsager_warm_start_m200_probe/
?? trials/active/fixed_onsager_warm_start_m_sweep_qw/
?? trials/active/fixed_onsager_warm_start_n_sweep_qw/
?? trials/active/fixed_onsager_warm_start_step_probe_qw/
?? trials/active/infinite_m100_smoke/
?? trials/active/infinite_n30000_smoke/
?? trials/active/infinite_n_calib_2000/
?? trials/active/infinite_n_calib_20000/
?? trials/active/infinite_n_calib_20000_64000/
?? trials/active/infinite_n_sweep_qw_64000/
?? trials/active/max_size_fixed_onsager_warm_start_050/
```

含义：

- 当前 tracked code/docs 在 `dev` 上看起来与 `origin/dev` 对齐，但有多个未跟踪
  trial config 目录。
- 这些 trial config 是当前 scan/calibration 工作留下的输入配置，不是大实验
  artifact；是否提交需要用户明确决定。
- `runs/`、`results/`、`artifacts/`、`*.pt` 仍然不能提交。
- 新 agent 接手后先跑：

```bash
cd /home/sucia/Matrix_Factorization
git status --short --branch
mf validate src/matrix_factorization/config.yaml
python -m pytest -q tests/test_contract_parameter_specs.py tests/test_config_contract.py
```

若要改 hard-interface 相关代码，最后再跑：

```bash
python -m pytest -q
```

## 2. Metric schema 当前事实

旧 handoff 中关于 schema v3/v4 的部分已经过时。当前代码里：

```text
metric_schema.schema_version = 5
metric_definition_profile = projection_qy_physical_latent_v2
```

当前正式语义：

- `Q_W_mean/std`：fixed-denominator physical overlap。
- `Q_X_mean/std`：fixed-denominator physical overlap。
- `Q_Y_mean/std`：output absolute projection。
- `FIT_Y_mean/std`：`1 - NMSE_Y`，只作为 reconstruction diagnostic。
- `NMSE_Y_mean/std`：输出 reconstruction error diagnostic。
- `Q_W_PROJ_ABS` / `Q_X_PROJ_ABS` / `Q_Y_PROJ_ABS`：旧 projection 诊断。
- `Q_W_SIGN_GAUGE` / `Q_X_SIGN_GAUGE`：正式 sign-gauge diagnostic。
- `Q_W_SIGN_ALIGNED` / `Q_X_SIGN_ALIGNED`：legacy alias。
- `Q_W_SCALE_GAUGE` / `Q_X_SCALE_GAUGE` / `Q_WX_SCALE_GAUGE`：
  diagonal scale-gauge diagnostic。
- `Q_W_COS_ROOT` / `Q_X_COS_ROOT`：cos-root diagnostic。
- `Q_W_GRAM_ROOT` / `Q_X_GRAM_ROOT`：legacy alias。

对每个 sample `r` 和每个 `alpha`：

```text
Q_W^(r)(alpha) =
  1 / (N1 * M) * sum_i sum_mu W_s^(r)[i, mu](alpha) * W_t[i, mu]

Q_X^(r)(alpha) =
  1 / (M * N2) * sum_mu sum_j X_s^(r)[mu, j](alpha) * X_t[mu, j]
```

最终 `Q_W_mean/std`、`Q_X_mean/std` 是先逐 sample 算，再对 sample 聚合。
这不是 cosine，也不是逐 channel normalize。

`Q_Y` 是 output projection：

```text
Q_Y = abs(<Y_student, Y_teacher>) / <Y_teacher, Y_teacher>
```

`FIT_Y` 是：

```text
NMSE_Y = ||Y_student - Y_teacher||^2 / ||Y_teacher||^2
FIT_Y = 1 - NMSE_Y
```

新旧结果不能只看同名 flat key 混画。必须先检查 `schema_version` 和
`metric_definition_profile`。

## 3. 当前 physical setup

最近用户最关心的线是 `bigamp_spreading`、fixed Onsager、warm start 的大尺寸
趋势：

```text
algorithm: bigamp_spreading
teacher: Gaussian / standard
F distribution: Ising / Rademacher
normalization_profile: paper_sparse_sampling
precision_profile: safe
onsager_correction: true
adaptive_damping: false
damping: 0.05
init_mode: teacher
init_overlap: 0.5
S: 1
M: 50 for N sweep
```

注意：这里的 `fixed_onsager_warm_start_050` 是当前看起来最有物理意义的分支。
不要把它和以前发散的 adaptive/fixed cold-start 结果混为一谈。

## 4. 已完成的 scan / calibration 结果

### 4.1 Smoke runs

`N=30000,M=50,alpha=2.1,steps=20,S=1` smoke 通过：

```text
artifacts/trials/infinite_n30000_smoke/
  20260429_073336_bgs_N30000_M50_expe1-a1_S1_steps20_40e491
```

命令实测时间约 `3.98s`。这只证明显存和 basic route 可行，不证明收敛。

`N=10000,M=100,alpha=2.1,steps=20,S=1` smoke 通过：

```text
artifacts/trials/infinite_m100_smoke/
  20260429_073350_bgs_N10000_M100_expe1-a1_S1_steps20_b2ac1e
```

命令实测时间约 `3.53s`。这只证明 M sweep 目标尺寸可放入显存，不证明收敛。

### 4.2 N=2000 calibration

路径：

```text
artifacts/trials/infinite_n_calib_2000/
  20260429_073409_bgs_N2000_M50_expe1-st4-a6_S1_steps5000_a7770b
```

配置：

```text
N1=N2=2000, M=50, S=1
steps = 5000, 10000, 20000, 40000
alpha = 0.90, 1.05, 1.50, 1.80, 1.95, 2.10
```

实测命令时间约 `938.11s`。

`Q_W_mean`：

```text
alpha   5000      10000     20000     40000
0.90    0.125623  0.106202  0.086213  0.078958
1.05    0.192555  0.178199  0.168629  0.161172
1.50    0.496054  0.495872  0.495866  0.495866
1.80    0.801193  0.801195  0.801195  0.801195
1.95    0.947586  0.948705  0.948737  0.948738
2.10    0.997528  0.997493  0.997492  0.997490
```

结论：

- `40000` vs `20000` 的最大差约 `0.007457`，满足之前的
  `max |Q_W(T)-Q_W(T/2)| < 0.01` 判据。
- 低 alpha 仍然是最慢区域；高 alpha 很早平台。

### 4.3 N=20000 calibration

第一段路径：

```text
artifacts/trials/infinite_n_calib_20000/
  20260429_075013_bgs_N20000_M50_expe1-st3-a6_S1_steps8000_c8220b
```

配置：

```text
N1=N2=20000, M=50, S=1
steps = 8000, 16000, 32000
alpha = 0.90, 1.05, 1.50, 1.80, 1.95, 2.10
```

实测命令时间约 `7227.95s`。

第二段路径：

```text
artifacts/trials/infinite_n_calib_20000_64000/
  20260429_095151_bgs_N20000_M50_expe1-a6_S1_steps64000_7ad19b
```

配置：

```text
N1=N2=20000, M=50, S=1
steps = 64000
alpha = 0.90, 1.05, 1.50, 1.80, 1.95, 2.10
```

实测命令时间约 `8247.24s`。

`Q_W_mean`：

```text
alpha   8000      16000     32000     64000
0.90    0.099285  0.083325  0.070666  0.061127
1.05    0.165938  0.149680  0.136876  0.126866
1.50    0.498668  0.498653  0.498653  0.498653
1.80    0.800735  0.800735  0.800735  0.800735
1.95    0.950287  0.950447  0.950448  0.950448
2.10    0.998985  0.998987  0.998988  0.998990
```

结论：

- `32000` vs `16000` 最大差约 `0.012804`，没有过 `0.01`。
- `64000` vs `32000` 最大差约 `0.0100105`，只比阈值多约 `1e-5`，
  卡在 `alpha=1.05`。
- 严格说 `64000` 仍是 near-threshold finite-time trajectory；不能写成已经
  严格平台。
- 物理上最慢的仍是低 alpha / transition 前侧，不是高 alpha 平台。

### 4.4 被中断的 uniform 64000 full N sweep

路径：

```text
artifacts/trials/infinite_n_sweep_qw_64000/
  20260429_121048_n_sweep_qw_64k_finite_time_bgs_N2000_M50_expe1-size5-a15_S1_steps64000_b93781
```

配置：

```text
N = 2000, 5000, 10000, 20000, 30000
M = 50
S = 1
steps = 64000 for every N and every alpha
alpha = 0.0, 0.15, ..., 2.10
```

实际状态：

```text
completed = 30 / 75
completed groups = N2000, N5000
last_group_id = experiment_policy=fixed_onsager_warm_start_050|size_N=N5000
```

命令被用户指出不合理后中断，耗时约 `9763.59s`。`N=10000` 正在中途，
不能作为完整曲线使用。

结论：

- 这次 run 只能把 `N=2000,N=5000` 当作 partial 可参考结果。
- 不要把它和后续新 alpha grid 的 `N=10000,N=20000` 混画成正式 N sweep。
- 错误在于把 `N=20000` 的 `64000` 步机械套给所有 N，导致总任务变成
  十几到几十小时量级，M sweep 被无限后推。

## 5. 当前 scan 策略结论

用户希望的是逼近大系统/无穷大系统行为，不是快速粗略图。因此不能回到
`steps=2000` 那种不足步数的扫描。

但 calibration 也表明：

- 高 alpha 平台段不需要 `64000` 步。
- 低 alpha 和 transition 前侧可能慢慢洗掉 warm-start overlap，不能用短窗口
  简单早停。
- 与其手工给每个 alpha 猜 steps，更合理的是给 `bigamp_spreading` 加
  teacher-student metric plateau early stop。

建议使用高精度 alpha grid C：

```text
0.00, 0.30, 0.60, 0.75,
0.85, 0.90, 0.95,
1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30,
1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10,
2.30, 2.50
```

解释：

- `0.00-0.75`：低 alpha 平台，点稀疏。
- `0.85-1.30`：transition 前后重点区，点密。
- `1.40-2.10`：线性/高 overlap 区，点适中。
- `2.30,2.50`：右侧平台补点，避免曲线刚到平台就结束。

## 6. Early stop 当前事实与推荐设计

当前事实：

- `algorithm_params.use_early_stop` 主要接在 `AGDAlgorithm`。
- `bigamp_spreading` formal batch loop 当前没有可靠 metric plateau early stop。
- 不能只在 YAML 里打开 `use_early_stop` 就期望 spreading 自动停。

推荐新增 `bigamp_spreading` 专用 teacher-student plateau stop。首版只用于
teacher-student 实验加速/诊断，不作为真实无监督算法默认行为。

推荐 YAML-facing 参数：

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

推荐判据：

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

不要只比较相邻 `2000` 步，因为低 alpha 可能每 `2000` 步变化很小但
`16000` 步窗口仍有明显慢漂移。

每次检查只算 latent metrics：

- `Q_W`
- `Q_X`
- `R_W`
- `R_X`

不要在 early-stop check 中算：

- edge-level `Q_Y`
- sign gauge
- scale gauge
- heatmap
- per-step saved tensors

否则检查本身会拖慢大 run。

metadata 必须记录：

```text
early_stop.enabled
early_stop.signal
early_stop.teacher_assisted
early_stop.check_interval
early_stop.window_steps
early_stop.patience
early_stop.abs_tol
early_stop.min_steps
early_stop.stop_step
early_stop.reason
early_stop.last_window_delta
early_stop.history_summary
```

## 7. Hard contract / project rules

这个项目有硬接口规则，新 agent 不要绕过。

新增 YAML-facing 参数必须：

- 在 `src/matrix_factorization/core/contracts.py` 增加 `ParameterSpec`。
- 进入 config dataclass / parsing / effective config trace。
- 在 metadata 或 diagnostics 里可见。
- 如果当前 route 不消费该参数，必须给 inactive-route warning 或 strict-mode error。
- 增加 contract tests。

新增 metric/output/analyzer/probe 必须：

- 先声明对应 spec。
- 注册项与 spec 对齐。
- 更新 source inventory（若该区域有 inventory 检查）。
- 增加测试覆盖 import/spec/validation 行为。

修改算法时必须：

- 明确物理 convention：latent scale、alpha normalization、F distribution、
  damping semantics、Onsager state、metric 定义。
- 不要为了跑通 silently 关闭 compile、改 precision、改 normalization。
- 若 fallback 是必要的，metadata 必须写原因。

实验数据规则：

- quick/debug/trial 使用 `trials/active/<trial_key>/config.yaml`。
- trial 输出进 `artifacts/trials/`、`runs/trials/` 或 `results/trials/`。
- 用户明确要求正式 run 时，才写 `runs/<run_name>/`。
- 不提交 `runs/`、`results/`、`artifacts/`、`.pt`。

## 8. 推荐给新 agent 的下一步

1. 只读扫描 repo 和当前分支，确认本 handoff 是否仍然准确。
2. 阅读 `docs/next_agent_prompt_2026-04-29_early_stop_scan.md`。
3. 设计并实现 `bigamp_spreading` metric plateau early stop。
4. 小尺寸测试：

```text
N = 200, 500, 1000, 2000
M = 50
S = 1
alpha = 0.0, 0.75, 0.9, 1.0, 1.05, 1.2, 1.5, 2.1, 2.5
```

5. 对比 fixed-step full run 与 early-stop run：

```text
|Q_W_early - Q_W_full| < 0.01
```

6. 若小测试通过，再跑 high-resolution alpha grid C 的大尺寸 N sweep。

不要在 early stop 尚未验证前继续开十几个小时的大 scan。

