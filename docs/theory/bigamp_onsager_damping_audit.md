# BiGAMP Onsager 与 Damping 设计审计

状态：审计与交接文档；2026-04-28 已完成第一轮公式/接口修正和小尺寸
diagnostic run，但未声称已经复现论文相图。

2026-04-28 后续核验：当前 `bigamp_spreading` formal batch route 已经把
no-Onsager 路径切回 `bigamp_step_disjoint_union_flat_legacy_fast`。因此
`spreading.onsager_correction: false` 时不会再进入 corrected step 里额外计算
`pvar/cross_var/gain/svar` 后又丢弃 Onsager state。更精确地说，旧问题不是
“每一步都实际减去 Onsager 项”，因为 `prev_s=None` 时 cavity subtract 不发生；
问题是 no-Onsager route 仍使用了 corrected step 的额外 BiGAMP state 计算与
数值路径。当前版本已避免这部分浪费。

同一次核验也确认：formal batch 的 `onsager_fixed_beta005` 和
`onsager_adaptive_beta005` 现在同样走 legacy-fast step 加 `prev_s` memory。
这可作为 2026-04-27 07:30 风格 baseline/速度回归对照，但不应写成
“corrected Onsager 已验证”。Corrected `zvar/pvar/svar/gain` functions 仍在
`step.py` 中保留，并由单函数/部分路径测试覆盖；若要继续论文级 Onsager 对照，
需要显式选择并重新验证 corrected route，而不是从当前 formal scan 自动推断。

本文只讨论当前代码里的 BiGAMP / spreading / tensor AMP 风格更新和
Onsager correction。目标是回答两个问题：

1. 当前 Onsager 项打开后结果变差，是否只是 damping 没调好。
2. 后续在 Gaussian teacher-student、Ising F、paper normalization、
   cold start 下，应该怎样设计可验证的修正和实验。

结论先行：当前问题不应首先归因于某个固定 damping 值没选对。更核心的
风险是当前实现把完整 BiGAMP 的多个 state convention 压成了一个
optimizer-like update：`V/pvar`、`prev_s/shat`、Gaussian prior denoiser、
`rGain/qGain`、以及 damping state 没有和论文及 Matlab reference 同步。
不开 Onsager 时，这个算法更像 diagonal-preconditioned residual fitting；
打开 Onsager 后，Onsager 所依赖的 cavity state 不再自洽，因此容易坏。

## 参考对象

目标 BiGAMP 参考材料：

- `/home/sucia/DeepCode/deepcode_lab/papers/BiGAMP/1.md`
- `/home/sucia/DeepCode/deepcode_lab/papers/BiGAMP/generate_code/big_amp_repro/`
- `/home/sucia/Matrix_Factorization/docs/reference_code/gamp_matlab/BiGAMP/`
- `/home/sucia/Matrix_Factorization/docs/reference_code/gamp_matlab/main/gampEstBasic.m`

当前实现重点路径：

- `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/step.py`
- `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/spreading.py`
- `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/standard.py`
- `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/tensor_step.py`
- `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/tensor_step_batch.py`
- `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/tensor_step_super.py`
- `/home/sucia/Matrix_Factorization/src/matrix_factorization/core/contracts.py`
- `/home/sucia/Matrix_Factorization/src/matrix_factorization/config.yaml`

## Reference BiGAMP 的最低必要 convention

BiGAMP 的 bilinear 模型是 `Z = A X`。对 spreading matrix，可以把每个观测
写成带权 rank sum：

```text
y_ij = sum_mu (F_ijmu / sqrt(M)) W_i,mu X_mu,j + noise
```

这和 BiGAMP 的 `sum_n a_mn x_nl` 同型，只是每个 rank component 多了已知
coefficient `F/sqrt(M)`。

论文 Table III 的关键结构是：

```text
p_ml = sum_n a_mn x_nl
nu_p_linear = sum_n |a_mn|^2 nu_x + nu_a |x_nl|^2
nu_p_output = nu_p_linear + sum_n nu_a nu_x
phat_ml = p_ml - shat_ml(t-1) * nu_p_like_term
```

OCR 文本中这些公式在
`/home/sucia/DeepCode/deepcode_lab/papers/BiGAMP/1.md:1281` 附近。
Matlab reference 更清楚地把两个 variance 分成：

- `zvar = Avar*xhat2 + Ahat2*xvar`
- `pvar = zvar + Avar*xvar`

见
`/home/sucia/Matrix_Factorization/docs/reference_code/gamp_matlab/BiGAMP/BiGAMP.m:378`
到 `:389`。

AWGN output step 对 observed entries 等价于：

```text
shat = (y - phat) / (pvar + noise_var)
svar = 1 / (pvar + noise_var)
```

论文 OCR 在 `/home/sucia/DeepCode/deepcode_lab/papers/BiGAMP/1.md:1533`
附近有 PIAWGN 段落，但 OCR 条件符号有错位；以 Matlab reference 和 Table
III 的结构为准。

Input linear step 不是单纯 `x += preconditioned_gradient`，而是先构造
`rhat/qhat`：

```text
rvar = 1 / sum_m |ahat_mn|^2 svar_ml
rhat = xhat * (1 - rvar * sum_m avar_mn * svar_ml)
       + rvar * sum_m ahat_mn^* shat_ml
```

Matlab reference 对应
`/home/sucia/Matrix_Factorization/docs/reference_code/gamp_matlab/BiGAMP/BiGAMP.m:686`
到 `:724`，`qhat` 对应 `:752` 到 `:815`。

Gaussian prior denoiser 对 `N(0, prior_var)` 是：

```text
x_new = prior_var / (prior_var + rvar) * rhat
xvar_new = prior_var * rvar / (prior_var + rvar)
```

论文 OCR 在 `/home/sucia/DeepCode/deepcode_lab/papers/BiGAMP/1.md:1661`
附近，Matlab reference 通过 `gX.estim(rhat, rvar)` 调用，见
`/home/sucia/Matrix_Factorization/docs/reference_code/gamp_matlab/BiGAMP/BiGAMP.m:819`
到 `:825`。

## Damping 的 reference 语义

BiGAMP 论文 Section IV 说 damping factor `beta(t) in (0,1]` 用来减慢
若干状态的演化。关键语义是：

- `beta = 1`：无 damping，完全接受新状态。
- `beta = 0`：冻结。

论文 OCR 在 `/home/sucia/DeepCode/deepcode_lab/papers/BiGAMP/1.md:1763`
附近明确提到 damping 应作用于 `nu_p`、`nu_s`、`shat`、`xhat`、`ahat`
等变量。

Matlab reference 的默认 adaptive step 也采用这个语义：

- `step = 0.05`
- `adaptStep = true`
- `stepMin = 0.05`
- `stepMax = 0.5`
- `stepIncr = 1.1`
- `stepDecr = 0.5`
- `stepWindow = 1`
- `pvarStep = true`

见
`/home/sucia/Matrix_Factorization/docs/reference_code/gamp_matlab/BiGAMP/BiGAMPOpt.m:73`
到 `:124`。

Reference 的 damping 不是只 damp factor mean。`BiGAMP.m` 中：

- `pvar/zvar` 按 step 混合，见
  `/home/sucia/Matrix_Factorization/docs/reference_code/gamp_matlab/BiGAMP/BiGAMP.m:405`
  到 `:409`。
- output step 先得到 `shatNew/svarNew`，见 `BiGAMP.m:497` 到 `:510`。
- 然后用 `shat=(1-step)*shatOpt + step*shatNew`，并同步 damp
  `svar/xhatBar/AhatBar`，见 `BiGAMP.m:664` 到 `:674`。

因此，如果代码只 damp `W/X`，但下一轮 Onsager 用 raw `s`，就不再是
reference BiGAMP 的 state convention。

## 当前代码的 confirmed 差异

### 1. Damping 语义在算法之间不一致

论文和 Matlab reference 语义是 `out = beta * new + (1 - beta) * old`。
当前 matrix spreading flat path 采用这个语义：

- `bigamp_step_disjoint_union_flat`：
  `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/step.py:565`
  到 `:569`

但 dense `bigamp` 使用相反语义：

- `_bigamp_step`：
  `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/standard.py:73`
  到 `:97`

tensor serial/batch 也使用旧语义：

- `tensor_step`：
  `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/tensor_step.py:218`
  到 `:220`
- `tensor_step_batch`：
  `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/tensor_step_batch.py:217`
  到 `:219`

tensor super path 又回到 `beta * new + (1-beta) * old`：

- `tensor_step_super`：
  `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/tensor_step_super.py:312`
  到 `:314`

这意味着 `algorithm_params.damping=0.5` 因为对称看不出问题；一旦设成
`0.05`、`0.1`、`0.8`，不同算法里含义相反。这个必须在任何 damping
sweep 前修正或显式标记。

### 2. Matrix spreading 的 `prev_s` 不是 reference 的 damped `shat`

matrix spreading flat step 中，如果 `prev_s` 非空，会做：

```text
Z_hat = Z_hat - V * prev_s
```

见
`/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/step.py:506`
到 `:509`。

随后代码只 damp `W/X/W_var/X_var`：

```text
W_flat_out = damping * W_hat_new + (1 - damping) * W_flat
X_flat_out = damping * X_hat_new + (1 - damping) * X_flat
```

见 `step.py:565` 到 `:569`。

runner 侧在 non-adaptive path 中直接保存 raw `s_values`：

```text
if self.onsager_correction:
    prev_s = next_prev_s
```

见
`/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/spreading.py:952`
到 `:957`。

这和 reference 不一致：reference 下一轮 Onsager 用的是 damp 后的
`shat` state，而不是 raw residual。`beta` 越小，这个错配越大，因为
`W/X` 走得很慢，`prev_s` 却跳到 raw candidate。

adaptive path 试图处理 per-alpha damping，但也不是 reference 的
`shat=(1-beta)*shatOpt+beta*shatNew`。当前代码使用：

```text
prev_s_accepted = s_vals * damping
prev_s = where(pass, prev_s_accepted, s_safe)
```

见 `spreading.py:1242` 到 `:1254`。这缺少 old `shat` 的 convex blend。

### 3. 当前 matrix/dense variance 缺少 bilinear `var * var` 对 output variance 的贡献

当前 matrix spreading 的 `V` 是：

```text
V = 1/M * sum_mu F^2 * (W_var * X_hat^2 + W_hat^2 * X_var)
```

见
`/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/step.py:497`
到 `:504`，以及 core helper
`/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/core.py:62`
到 `:109`。

Dense `bigamp` 也同样只算这两项，见
`/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/standard.py:60`
到 `:63`。

Reference BiGAMP 分 `zvar` 与 `pvar`：

```text
zvar = mean^2 * var + var * mean^2
pvar = zvar + var * var
```

见 `BiGAMP.m:378` 到 `:389`。这对 cold start 很关键：当 means 很小而
variances 约等于 prior variance 时，`var * var` 可能是 output uncertainty
的主项。当前 denominator 使用 `V + noise_var`，会低估 output variance，
在 near-noiseless limit 下使 `s=(Y-Z)/(V+noise)` 过度自信。

更精确地说：Matlab reference 的 Onsager cavity 用 `zvar`，output denoiser
用 `pvar`。当前代码把两者折叠成一个 `V`，即便 Onsager correction 的
scale 接近 `zvar`，AWGN residual denominator 也仍少了 cross variance。

### 4. Gaussian denoiser/input step 不完整

当前 matrix spreading update 是：

```text
W_var_new = 1 / (prior_precision_base + tau_W)
W_hat_new = W_flat + W_var_new * r_W
```

见 `step.py:543` 到 `:545`；`X` 同理见 `step.py:561` 到 `:563`。

这更像 diagonal-preconditioned residual update。若按简化 Gaussian posterior
写成 incremental 形式，应至少有：

```text
new = old + new_var * (r - prior_precision * old)
```

如果进一步贴近 reference，还要加 `rGain/qGain` 对变量方差导致的
self-reaction 做修正：

```text
new = new_var * ((tau - tau_var_correction) * old + r)
```

当前 matrix path 缺少 prior shrink 和 `rGain/qGain`。这会解释一个现象：
不开 Onsager 时它能像优化器一样工作；一旦加 Onsager，就要求 state
满足 AMP cavity 假设，而当前 input denoiser 并不满足完整 BiGAMP。

各 tensor path 也不统一：

- `tensor_step.py` / `tensor_step_batch.py` 用
  `new_var * (tau * factor + r)`，这是接近“无 rGain 的 Gaussian posterior”
  的形式，但 damping 语义是旧的。
- `tensor_step_super.py` 改成
  `factor + new_var * (r - tau * factor)`，见
  `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/tensor_step_super.py:303`
  到 `:314`。这里的 shrink coefficient 使用 `tau`，而简化 Gaussian
  posterior incremental form 中应至少是 `prior_precision`；若包含
  reference gain，还应是 `prior_precision + tau_var_correction`。这需要
  单步 parity test 重新验证。

### 5. Adaptive damping 当前是 heuristic，不是 reference backtracking

当前 adaptive matrix spreading 有 per-alpha `damping` vector：

- 初始化见 `spreading.py:1042`
- pass/reject 更新见 `spreading.py:1212` 到 `:1217`

这是方向正确的，因为 alpha-parallel 时不同 alpha 的稳定步长确实可能不同。
但它不是 reference BiGAMP adaptive damping：

1. cost 使用 `compute_log_likelihood`，只包含 Gaussian output likelihood，
   见 `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/bigamp/step.py:276`
   到 `:299`。论文 Eq. (108) 还包含 input posterior 相对 prior 的 KL 项。
2. reference 失败时会减小 step 并重试/backtrack 当前候选；当前代码主要是
   在循环入口 state 上评估 likelihood，然后构造下一步 damped candidate，
   没有完整复刻 `shatOpt/shatNew/svarOpt/svarNew/pvarOpt/zvarOpt` 的状态机。
3. `step_window`、`max_bad_steps` 的 reference 逻辑没有完整落地。
4. 当前默认 config 对 Onsager 调试偏激进：
   `/home/sucia/Matrix_Factorization/src/matrix_factorization/config.yaml:112`
   到 `:127` 中 `step_max=1`、`step_decr=0.9`、`acceptance_tolerance=50`。
   Matlab reference 是 `stepMax=0.5`、`stepDecr=0.5`、`stepWindow=1`。

因此，当前 adaptive damping 可以作为 heuristic 记录，但不能把它当成
“论文 adaptive damping 已实现”。

### 6. Config/spec 已声明风险，但没有强制公式一致

`core/contracts.py` 已有参数声明：

- `algorithm_params.damping`：`core/contracts.py:420`
- `algorithm_params.adaptive_damping` 和 step 参数：`core/contracts.py:435`
  到 `:441`
- `spreading.onsager_correction`：`core/contracts.py:450`

也已有 parity 风险条目：

- `damping_semantics`：`core/contracts.py:1710` 到 `:1715`
- `onsager_handling`：`core/contracts.py:1718` 到 `:1723`

这些 spec 能提醒路由和 metadata，但目前没有单步公式 parity test 去强制：

- `beta` endpoint 语义一致；
- `zvar/pvar` 分离；
- `prev_s` 是 damped `shat`；
- Gaussian posterior mean 与 variance 公式正确；
- adaptive backtracking 是否真的评价并回滚 candidate state。

## No-Onsager 版本到底退化成什么

当前 no-Onsager BigAMP 不是 AGD，但也不是严格 Bayes AMP。它更接近：

```text
Z = forward(theta)
V = local variance / curvature estimate
s = (Y - Z) / (V + noise)
r = J^T s
tau = diag(J^T diag(1/(V + noise)) J)
theta <- theta + (prior_precision + tau)^(-1) r
```

这可以理解为 diagonal-preconditioned residual fitting、IRLS、或
Gauss-Newton-like optimizer。它和 AGD 的主要区别是：

- AGD 使用固定 learning rate，见
  `/home/sucia/Matrix_Factorization/src/matrix_factorization/modules/algorithms/agd.py:206`
  到 `:219`。
- no-Onsager BigAMP 每个坐标都有自适应步长 `1/(prior_precision+tau)`。
- no-Onsager BigAMP 用 `V+noise` 缩放 residual，等价于对不同观测置信度做
  reweighting。

所以它可能比 AGD 快很多，也可能在 warm start 或高 alpha 时表现很好。
但这个好结果不证明它满足 BiGAMP state evolution；它更像一个稳定的
preconditioned optimizer。Onsager correction 需要严格的 cavity/state
匹配，因此打开后会暴露上述不一致。

## 为什么不建议先做手写 alpha-dependent damping array

用户提出 alpha 并行时如果不同 alpha 需要不同 damping，可能要用数组。
这个判断方向是对的：临界区、低 alpha、高 alpha 的稳定步长通常不同。
但在当前公式未自洽前，手写 `alpha -> damping` 表很可能只是在补偿实现缺陷。

合理顺序应是：

1. 先修 `zvar/pvar`、Gaussian denoiser、`prev_s/shat`、damping 语义。
2. 用单步 parity test 确认公式。
3. 再启用 per-alpha adaptive `beta`。
4. 最后才考虑可选的 static `damping_by_alpha`，用于复现实验，而不是默认策略。

建议默认使用 per-alpha adaptive policy，而不是固定数组。原因：

- alpha 越接近临界点，局部 Jacobian 更容易接近不稳定边界。
- high alpha 观测更多，`tau` 更大，posterior step 本身更小，通常可以逐步增大
  beta。
- low alpha 信号弱且 residual 噪声大，固定大 beta 更容易震荡。
- alpha-parallel batch 中用一个 scalar beta，会迫使所有 alpha 跟随最不稳定点。

建议 reference 起点：

```yaml
algorithm_params:
  adaptive_damping: true
  damping: 0.05          # initial beta, new-state weight
  step_min: 0.05
  step_max: 0.5
  step_incr: 1.1
  step_decr: 0.5
  step_window: 1
  acceptance_tolerance: 0
```

若需要更保守的 quick check，可固定 `beta=0.05` 或 `beta=0.1`，但这只用于
确认 corrected Onsager 不爆炸，不应作为最终论文对照。

## 建议的修正路线

### Phase 1：单步公式 parity，不跑大实验

新增小型 deterministic test，构造几条 edge 和固定小矩阵，直接比较公式：

1. `zvar = sum F^2/M * (W_var * X_hat^2 + W_hat^2 * X_var)`
2. `pvar = zvar + sum F^2/M * W_var * X_var`
3. `phat = zhat - zvar * shat_prev` 或按最终选定 reference convention；
   注意要在文档中解释为何用 `zvar` 而不是 `pvar`。
4. AWGN output：
   `shat_new=(Y-phat)/(pvar+noise)`，
   `svar_new=1/(pvar+noise)`。
5. Gaussian input：
   `rvar=1/tau`，
   `rhat=old*rGain+rvar*r`，
   `new=prior_var/(prior_var+rvar)*rhat`。
6. Damping：
   `state = beta*new + (1-beta)*old`，并验证 `beta=1` 完全接受、
   `beta=0` 冻结。

这个 test 先覆盖 matrix spreading flat path。tensor path 可以在 matrix path
稳定后再对齐。

### Phase 2：统一 runtime semantics

建议引入一个小 helper，避免各文件重复写相反公式：

```python
def blend_new_old(new, old, beta):
    return beta * new + (1.0 - beta) * old
```

然后把 dense `bigamp`、tensor serial、tensor batch、tensor super、matrix
spreading 都迁移到同一 endpoint 语义。为避免旧结果误读，文档和 result
metadata 中要明确：

```text
damping_semantics = "beta_new_weight"
beta=1 means full new-state acceptance.
beta=0 means frozen state.
```

### Phase 3：拆分 `zvar/pvar/shat/svar`

对 matrix spreading step，至少返回并保存：

- `shat`：damped Onsager residual state；
- `svar`：damped output precision state，或者至少用于 input tau 的
  `1/(pvar+noise)`；
- `zvar`：Onsager cavity factor；
- `pvar`：AWGN denominator；
- optional diagnostics：`zvar_mean/pvar_mean/cross_var_mean/s_norm`。

当前 `prev_s` 字段可以保留为 legacy name，但文档中应改称 `shat_state`。
更清楚的代码接口是新增 state dataclass，而不是继续传一个裸 tensor。

### Phase 4：修正 Gaussian input update

Matrix path 应从：

```text
W_new = W + new_var * r
```

改到至少：

```text
W_new = W + new_var * (r - prior_precision * W)
```

更贴近 reference 的版本还要包含 variance-gain correction：

```text
W_new = new_var * ((tau - tau_var_correction) * W + r)
```

这里的 `tau_var_correction` 对应 reference `rGain/qGain` 中的
`sum var_other * svar`。这部分要通过单步 test 固定。

### Phase 5：per-alpha adaptive damping

修完公式后，再实现或重构 adaptive damping：

- per-alpha `beta` vector，shape `(num_alpha_in_batch,)`；
- 记录 `beta_history`、`pass_mask_history`、`cost_history`；
- 默认 reference 参数；
- fail 时回滚到 last accepted state；
- acceptance 先用 conservative output likelihood + prior KL；如果 prior KL
  实现复杂，必须在 metadata 里标成 heuristic。

不要把 `acceptance_tolerance=50` 作为 corrected Onsager 默认。它会让坏 step
太容易通过。

## 建议的验证实验设计

该实验是未来步骤，不是本文已执行内容。

目标：Gaussian teacher-student，Ising F，paper_sparse_sampling
normalization，矩阵 spreading，完全 cold start。

建议尺寸：

```yaml
matrix:
  N1: 200
  N2: 200
  M: 50
spreading:
  f_distribution: ising
  onsager_correction: false/true
algorithm_params:
  normalization_profile: paper_sparse_sampling
  init_mode: random
  adaptive_damping: false/true
```

alpha 点先稀疏：

```text
[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
```

samples per alpha：`10`。

对照组：

1. no-Onsager legacy/preconditioned baseline。
2. corrected Onsager + fixed beta quick check，例如 `beta=0.05` 或 `0.1`。
3. corrected Onsager + adaptive per-alpha beta。

必要 plots：

- `Q_Y`
- `Q_W` 和 `Q_X`
- `Q_W_COS_ROOT` 和 `Q_X_COS_ROOT`
- `Q_W_SIGN_ALIGNED` 和 `Q_X_SIGN_ALIGNED`
- posthoc scale-gauge aligned `Q_W/Q_X`，或合并成一个 gauge diagnostic plot
- damping diagnostics：`beta_history(alpha, step)`、pass rate、`zvar/pvar`
  mean、`cross_var/pvar` ratio、`||shat||`

必要 artifacts：

- per-point student/teacher factors，或者至少保存轻量 gauge payload；
- beta/pass/cost history；
- formula diagnostics summary。

注意：之前 7:30 Gaussian run 如果没有保存 factor tensors，不能 posthoc 计算
同样的 gauge-aligned metric。要验证 Gaussian teacher，必须重跑并保存必要
payload。

## 当前可回答的问题

### Onsager 符号是不是错的？

`Z_hat = Z_hat - V * prev_s` 的符号和 AMP/BiGAMP cavity 方向大体一致。
主要问题不是符号，而是 `V`、`prev_s`、damping state 和 Gaussian denoiser
不匹配。

### 为什么 no-Onsager 反而好？

因为 no-Onsager 当前更像 diagonal-preconditioned optimizer。它不需要严格
cavity memory，所以少受 state mismatch 影响。加 Onsager 后，算法开始依赖
上一轮 residual 与当前 damped variables 的精确关系，已有不一致会被放大。

### damping 是否应该随 alpha 变化？

最终应该允许 per-alpha adaptive beta。不要先手工硬编码数组。先修公式，
再让 adaptive policy 记录每个 alpha 的 beta 轨迹。若之后发现 adaptive
收敛到稳定的 alpha-dependent pattern，再把它提炼成可复现的 static schedule。

### 当前能不能直接跑 200x200x50 对照？

可以作为 legacy diagnostic，但不建议作为论文结论。当前 Onsager path 的
公式问题会污染实验解释。建议先完成单步 parity 和 small smoke，再跑该对照。

## 交接结论

当前 Onsager path 的失败不是一个单独 damping 参数能解释的问题。最重要的
修正顺序是：

1. 统一 damping endpoint 语义。
2. 分离 `zvar` 与 `pvar`，补上 `var * var` output variance。
3. 用 damped `shat` state 做下一轮 Onsager，而不是 raw `s_values`。
4. 修正 Gaussian prior denoiser/input update，至少补 prior shrink。
5. 用单步 parity test 锁住公式。
6. 再做 per-alpha adaptive damping 和 200x200x50 cold-start 对照。

在这之前，大量扫 damping 或 alpha-dependent damping array 很可能只是把实现
缺陷调成某个局部现象，不能回答论文物理问题。

## 2026-04-28 实现状态更新

本节记录本审计之后已经落到代码里的部分。这里不是说 200x200x50
物理实验已经完成，只说明公式 convention 和接口层已经做了第一轮收口。

### 已修改

1. 新增共享 convention helper：
   `src/matrix_factorization/modules/algorithms/bigamp/conventions.py`。

   其中 `DAMPING_SEMANTICS = "beta_new_weight"`，并提供
   `blend_new_old(new, old, beta)`：

   ```text
   beta = 1  -> 完全采用 new
   beta = 0  -> 保留 old
   ```

2. Matrix spreading 路径：
   `src/matrix_factorization/modules/algorithms/bigamp/step.py`。

   已把 flat / adaptive / general chunked 路径统一为：

   ```text
   zvar = sum F^2/M * (var_a * mean_b^2 + mean_a^2 * var_b)
   pvar = zvar + sum F^2/M * var_a * var_b
   phat = zhat - zvar * shat_prev
   shat = (Y - phat) / (pvar + noise)
   svar = 1 / (pvar + noise)
   ```

   Gaussian denoiser 统一走
   `gaussian_posterior_update(...)`，包含 zero-mean Gaussian prior shrink 和
   first-order gain/self-reaction correction。

3. Spreading 训练入口：
   `src/matrix_factorization/modules/algorithms/bigamp/spreading.py`。

   已让 fixed damping 和 adaptive damping 都携带 damped `shat` 与 damped
   `svar` 状态。`onsager_correction=false` 时不会再把上一轮 residual
   偷偷传入下一步。

   adaptive damping 仍是 heuristic backtracking，不是完整 Matlab reference
   cost/state machine；但它现在使用当前 step 的 beta 更新当前 accepted state，
   再把增减后的 beta 留给下一步。

4. Dense matrix BigAMP：
   `src/matrix_factorization/modules/algorithms/bigamp/standard.py`。

   已补 full `pvar = zvar + var*var`，damping 语义改为 beta-new-weight，并使用
   Gaussian posterior update。该路径仍没有显式 Onsager memory state，因此不要把它
   当成 corrected Onsager 的主要验证路径。

5. Tensor step 路径：
   `src/matrix_factorization/modules/algorithms/bigamp/tensor_step.py`，
   `src/matrix_factorization/modules/algorithms/bigamp/tensor_step_batch.py`，
   `src/matrix_factorization/modules/algorithms/bigamp/tensor_step_super.py`。

   已把 output variance 改成

   ```text
   pvar = sum F^2/M * (prod_d E[x_d^2] - prod_d E[x_d]^2)
   ```

   并把 Onsager `zvar` 改成显式 exclude-dimension product：

   ```text
   zvar = sum_d var_d * prod_{e != d} mean_e^2
   ```

   这避免 cold start 中某个 mean 接近 0 时，除法形式把应该存在的一阶 variance
   项误消掉。tensor serial / batch / super 也开始返回并传递 damped `svar`。

6. 参数和 source inventory：
   `src/matrix_factorization/core/experiment/config.py`，
   `src/matrix_factorization/core/contracts.py`，
   `src/matrix_factorization/config.yaml`，
   `docs/config_user_template.yaml`。

   已加入 damping/adaptive damping 的范围校验，`step_incr` 默认改为 `1.1`，
   `step_window` 默认改为 `1`，默认 `step_max=0.5`、`step_decr=0.5`、
   `acceptance_tolerance=0.0`。新 helper file 已加入 algorithm source
   inventory。

7. Tensor AGD：
   `src/matrix_factorization/modules/algorithms/agd_tensor.py`。

   `cp_contract_factors(...)` 已改为乘以 `normalization_profile` 解析出的
   `interaction_scale = 1/sqrt(M)`，避免 dense tensor AGD 继续使用未缩放的
   CP contraction。

### 已验证

已运行：

```bash
python -m compileall -q src/matrix_factorization/modules/algorithms src/matrix_factorization/core
python -m pytest -q tests/test_bigamp_onsager_convention.py tests/test_tensor_step.py tests/test_config_contract.py tests/test_contract_parameter_specs.py tests/test_normalization_precision_contract.py
python -m pytest -q tests/test_config_contract.py tests/test_contract_parameter_specs.py tests/test_normalization_precision_contract.py tests/test_contract_algorithm_specs.py
mf validate src/matrix_factorization/config.yaml
```

新增测试：
`tests/test_bigamp_onsager_convention.py`，覆盖：

- damping endpoint；
- Gaussian posterior update；
- fixed damping `beta=0` 是否冻结 factor 和 Onsager state；
- flat spreading cold-start pvar 是否包含 `var*var` cross variance；
- tensor cold-start pvar 是否包含 full product variance。

### 2026-04-28 diagnostic run 与 posthoc gauge

已跑一次用户指定尺寸的 200x200x50 cold-start Gaussian teacher /
Ising F diagnostic trial，但这只是 150-step smoke，不是最终论文曲线。
最初 run 是：

```text
runs/trials/onsager_cold_200_m50/20260428_023821_bgs_N200_M50_onsa3-a5_1bd4ad
```

该 run 保存了 per-point `results.pt`，但没有保存真实 `W_teacher/X_teacher`
artifact，因此不能可靠做 posthoc scale-gauge 诊断；从 config/seed 重建 teacher
会得到形状正确但物理对象不匹配的结果。

因此保存层新增了共享 artifact：

- `src/matrix_factorization/core/experiment/result.py`
- `artifacts/teacher_factors.pt`
- point payload 中写入 `teacher_factors_path`

并新增 posthoc 脚本：

- `scripts/analysis/posthoc_scale_gauge.py`

该脚本只读取 run 保存的真实 teacher artifact；如果缺失会失败，不再从 config
反推 teacher。

修正保存层后重跑了同一 trial：

```text
runs/trials/onsager_cold_200_m50/20260428_024747_bgs_N200_M50_onsa3-a5_1bd4ad
```

生成的 posthoc scale-gauge 输出位于：

```text
runs/trials/onsager_cold_200_m50/20260428_024747_bgs_N200_M50_onsa3-a5_1bd4ad/plots/posthoc_scale_gauge
```

核心 `Q_Y_mean` 仍显示：no-Onsager baseline 在 150 steps 下 alpha=2.0 更强；
fixed Onsager 稳定但慢；adaptive Onsager 在 alpha=1.5 和 alpha=3.0 表现更好，
但 alpha=2.0 仍弱于 baseline。

scale-gauge aligned `Q_WX` 与 sign-aligned 指标接近，median `|log |g||`
约在 `0.01-0.06`，说明这次 cold-start 结果的主要 gauge 问题更像 sign gauge，
不是明显的连续 scale gauge 漂移。

### 尚未完成

1. adaptive damping 仍没有完整复刻 Matlab BiGAMP 的 `pvarStep`、
   cost window、prior KL / value history state machine。
2. 还没有记录 `beta_history`、`pass_mask_history`、`zvar/pvar/cross_var`
   diagnostics 到正式 artifact。
3. posthoc scale-gauge aligned metric 仍是 diagnostic 方向，尚未提升为
   hard-interface formal metric。
4. 当前 trial 仍是 average-degree graph、150-step sparse alpha diagnostic；
   不能当作论文 exact-degree `c=alpha M` 相图复现。

## 2026-04-28 05:30 追加核验：formal-size scan / compile / damping

本轮目标是用户指定的 Gaussian teacher-student、Ising `F`、
`N1=N2=200, M=50, S=100`、cold start。核心结论：

- `no_onsager` 正式 2000-step baseline 可用；在
  `runs/20260428_043906_bgs_N200_M50_ons3-a41_S100_steps2000_a4258e`
  中，`alpha=4.0` 时 `Q_Y_mean≈0.996`，sign-aligned 和 Cos-root
  latent diagnostics 也接近 `1`。
- `onsager_fixed_beta005` 正式组不是可用物理结果；`Q_Y_mean` 到
  `10^6-10^7` 量级，posthoc scale-gauge 也出现 non-finite 或巨大值。
- 更小 fixed beta 不是充分修复。在
  `runs/trials/onsager_damping_probe/20260428_052930_onsager_damping_probe_bgs_N200_M50_ons5-a6_S100_steps500_f770ef`
  中，`beta=0.01` 和 `beta=0.005` 仍在 500 steps 内爆炸；
  `beta=0.0005` 也仍给出 `Q_Y_mean≈10^5` 量级。
- adaptive Onsager 不能写成已解决。`onsager_adaptive_beta005` standalone
  full-alpha 100-step route 可以用 compiled adaptive step 跑完，但低 alpha
  出现 `Q_Y>1` 和 sign-aligned latent projection `>1` 的 over-scale/overfit
  现象；`onsager_adaptive_cap005` standalone 500-step 也不物理。
- mixed multi-policy scan 中，fixed 组已经发散后再进入 adaptive 组，会在
  candidate forward variance 处触发 CUDA driver error。清理跨 group
  compile cache 不足以修复这个现象；adaptive 目前应单独 trial 隔离诊断，
  不应和已知发散的 fixed Onsager 组放在同一长进程里解释。

本轮代码修正：

- `src/matrix_factorization/modules/algorithms/bigamp/spreading.py`
  不再无条件关闭 adaptive 的 `torch.compile`。现在 adaptive 使用
  `bigamp_step_disjoint_union_flat_adaptive_legacy_fast` 的 compiled default-mode
  路径；只有 compile/runtime 抛错时才 fallback，并把状态写入
  `execution_metadata.compile_status`。
- `src/matrix_factorization/modules/algorithms/bigamp/step.py`
  新增 `forward_disjoint_union_flat_legacy_fast`。adaptive acceptance 现在评估
  beta 混合后的候选状态，而不是旧状态或 beta=1 raw state。这修正了一个
  明确的 damping/acceptance 语义 bug，但并没有让当前 Onsager 物理上成功。
- `src/matrix_factorization/core/experiment/runner.py`
  在 canonical scan 的 group 切换处清理算法 compile cache，并把 child
  algorithm result 的 `execution_metadata` 汇总到 aggregate metadata。

因此，当前工程判断是：`no_onsager` baseline 可继续用于论文 convention /
normalization / metric 对比；fixed/adaptive Onsager 都仍属于未验证施工线。
下一步不应继续盲跑 2000-step adaptive，而应回到 BiGAMP reference 的
`pvar/zvar/svar/shat`、value/cost、damping state machine 和 Gaussian denoiser
逐项对齐。
