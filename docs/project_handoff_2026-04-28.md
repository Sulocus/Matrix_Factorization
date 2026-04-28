# Project Handoff 2026-04-28

这份文档用于把本轮长对话交接给新的 Codex agent。它记录：

- 已经完成或已经动过手的工程改造。
- 当前工作树里还没有完全闭环的修改线。
- 最近围绕 projection metric、gauge、论文归一化、alpha 定义的关键结论。
- 下一轮应该从哪里继续，避免新 agent 重复扫描或误把未完成内容当成完成。

当前开发分支：`dev`。主开发包：`src/matrix_factorization`。

## 0. 当前工作树状态

最近一次检查显示工作树不是干净状态：

```text
## dev...origin/dev
 M docs/METRICS_GUIDE.md
 M docs/config_user_template.yaml
 M docs/hard_interface_framework.md
 M docs/implementation_progress.md
 M docs/metric_naming_decisions.md
 M docs/metrics_semantics.md
 M docs/parallel_memory_contract.md
 M src/matrix_factorization/cli.py
 M src/matrix_factorization/config.yaml
 M src/matrix_factorization/core/config.py
 M src/matrix_factorization/core/contracts.py
 M src/matrix_factorization/core/experiment/config.py
 M src/matrix_factorization/core/experiment/data_factory.py
 M src/matrix_factorization/core/experiment/result.py
 M src/matrix_factorization/core/experiment/runner.py
 M src/matrix_factorization/core/memory_calibration.py
 M src/matrix_factorization/core/parallel/batch_checkpoint.py
 M src/matrix_factorization/core/parallel/execution_modes.py
 M src/matrix_factorization/core/parallel/memory_estimator.py
 M src/matrix_factorization/core/parallel/memory_estimator_general.py
 M src/matrix_factorization/core/parallel/resource_execution.py
 M src/matrix_factorization/core/planning.py
 M src/matrix_factorization/modules/algorithms/agd.py
 M src/matrix_factorization/modules/algorithms/bigamp/spreading.py
 M src/matrix_factorization/modules/algorithms/bigamp/standard.py
 M src/matrix_factorization/modules/algorithms/bigamp/step.py
 M src/matrix_factorization/modules/algorithms/bigamp/tensor_contract.py
 M src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading.py
 M src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading_parallel.py
 M src/matrix_factorization/modules/algorithms/bigamp/tensor_step.py
 M src/matrix_factorization/modules/algorithms/bigamp/tensor_step_batch.py
 M src/matrix_factorization/modules/algorithms/bigamp/tensor_step_super.py
 M src/matrix_factorization/modules/metrics/__init__.py
 M src/matrix_factorization/modules/metrics/combined.py
 M src/matrix_factorization/modules/metrics/contract_compute.py
 M src/matrix_factorization/modules/metrics/overlap.py
 M src/matrix_factorization/modules/metrics/spreading.py
 M src/matrix_factorization/modules/outputs/plot_registry.py
 M src/matrix_factorization/modules/outputs/plotting.py
 M tests/test_config_contract.py
 M tests/test_contract_metric_output_specs.py
 M tests/test_contract_parameter_specs.py
 M tests/test_contract_runtime_extensions.py
 M tests/test_plot_query.py
 M tests/test_projection_metric_contract.py
 M tests/test_result_cube.py
 M tests/test_scan_resource_plan_contract.py
 M tests/test_scan_runner.py
?? "Tensor Factorization.pdf"
?? tests/test_normalization_precision_contract.py
?? tests/test_spreading_batch_metrics.py
```

含义：

- 不能假设当前修改已经提交或全部通过测试。
- `Tensor Factorization.pdf` 是用户提供或放在项目根目录的论文文件，不是生成结果。
- `runs/`、`results/`、`artifacts/`、`src/matrix_factorization/Replica_results/` 仍然不能提交。
- 新 agent 接手第一步必须重新运行：

```bash
git status --short --branch
python -m pytest -q
mf validate src/matrix_factorization/config.yaml
```

## 1. 已完成并应保留的工程方向

旧 handoff 见 `docs/project_handoff_2026-04-27.md`。这里压缩总结，不重复所有细节。

### 1.1 Hard interface baseline

核心文件：

- `src/matrix_factorization/core/contracts.py`
- `src/matrix_factorization/core/planning.py`
- `src/matrix_factorization/cli.py`

已经建立：

- `ParameterSpec`
- `AlgorithmSpec`
- `TeacherSpec`
- `GraphSpec`
- `MetricSpec`
- `OutputSpec`
- `InterventionSpec`
- `ProbeSpec`
- `AnalyzerSpec`
- `TrialSpec`
- `SourceInventorySpec`
- `BatchingSpec`
- `SeedPolicySpec`
- `MemoryModelSpec`
- `ResourceSpec`

目标是：新增参数、algorithm、metric、output、probe、intervention 时，必须先声明 spec，否则 import、validate 或 contract tests 失败。

已经有：

- `mf validate`
- `mf explain-config`
- `--json`
- `--strict`
- 参数链路：YAML path -> effective value -> consumer -> current-route status。

### 1.2 Research trial workflow

核心目录/文件：

- `trials/`
- `trials/active/*/trial.yaml`
- `trials/active/*/config.yaml`

命令：

```bash
mf trial list
mf trial explain <trial_key>
mf trial validate <trial_key>
mf trial run <trial_key>
```

规则：

- quick/debug trial 只改 `trials/active/<trial_key>/config.yaml`。
- 正式默认配置是 `src/matrix_factorization/config.yaml`，除非用户明确说要改正式配置。
- trial 输出进 ignored `runs/trials/`，不刷新 `results/latest`。

### 1.3 Canonical scan / ResultCube / PlotQuery

核心文件：

- `src/matrix_factorization/core/scan_planning.py`
- `src/matrix_factorization/core/parallel/resource_execution.py`
- `src/matrix_factorization/core/experiment/runner.py`
- `docs/scan_system_contract.md`

当前主入口：

```yaml
scan:
  axes:
    alpha:
      path: alpha
      values: {start: 0.0, stop: 4.0, step: 0.1}
```

旧入口已经不应再作为主链路：

- `scan_mode`
- `alpha_scan`
- `steps_scan`
- `nested_scan`
- `hysteresis_scan`

当前概念：

- 所有 scan 都是 axis 组合。
- `alpha/max_steps/size/init/damping/onsager/precision_profile` 都可以表达成 axis。
- `ScanPlan` 展开为 `ScanPoint`。
- canonical multi-axis result 写入 `ResultCube`。
- `PlotQuery` 使用 `where / compare / series_by / x / y` 从 ResultCube 查询曲线。

### 1.4 Projection metric schema v3

核心文件：

- `src/matrix_factorization/modules/metrics/contract_compute.py`
- `src/matrix_factorization/modules/metrics/overlap.py`
- `src/matrix_factorization/modules/metrics/spreading.py`
- `src/matrix_factorization/modules/metrics/combined.py`
- `src/matrix_factorization/core/experiment/result.py`
- `docs/METRICS_GUIDE.md`
- `docs/metric_naming_decisions.md`

用户已经确认的正式语义：

```text
Q_Y = abs(<Y_student, Y_teacher>) / <Y_teacher, Y_teacher>
Q_W = abs(<W_student, W_teacher>) / <W_teacher, W_teacher>
Q_X = abs(<X_student, X_teacher>) / <X_teacher, X_teacher>
Q_N = tensor latent node/spin/factor projection
```

规则：

- projection 不做 cosine normalization。
- projection 不 clip，大于 1 保留。
- `mean/std` 只是 sample/replica 统计后缀。
- `MSE / Gen_Error / Q_Y_COS / physical_overlap_* / Q_W_prime / Q_X_prime` 不再是 formal metric。

正式 diagnostic：

```text
Q_W_COS_ROOT / Q_X_COS_ROOT
Q_W_SIGN_ALIGNED / Q_X_SIGN_ALIGNED
```

`Q_W_SIGN_ALIGNED` 和 `Q_X_SIGN_ALIGNED` 是逐 latent channel 的 sign-gauge diagnostic：

```text
Q_W_SIGN_ALIGNED = sum_k abs(<W_s[:, k], W_t[:, k]>) / sum_k ||W_t[:, k]||^2
Q_X_SIGN_ALIGNED = sum_k abs(<X_s[k, :], X_t[k, :]>) / sum_k ||X_t[k, :]||^2
```

它不替代 `Q_W/Q_X`，只用于诊断“学到了 channel 但 sign sector 没对齐”。

### 1.5 AlgorithmResult active path

核心文件：

- `src/matrix_factorization/core/contracts.py`
- `src/matrix_factorization/core/experiment/runner.py`
- `src/matrix_factorization/modules/algorithms/agd.py`
- `src/matrix_factorization/modules/algorithms/bigamp/standard.py`
- `src/matrix_factorization/modules/algorithms/bigamp/spreading.py`
- tensor algorithms under `src/matrix_factorization/modules/algorithms/bigamp/`

已经建立：

- active path 优先消费 `AlgorithmResult.metrics_by_alpha`。
- tensor metrics-only route 不再保存 dummy zero `W/X` 当真实 factor。
- active tensor route 缺 metric 时 runner 不应从私有 `_batch_metrics` 补救。

仍需注意：

- `train_batch_alphas()` legacy tuple 还存在，旧脚本可能调用。
- 新 agent 不要把 legacy tuple 当 active path 的正式结果对象。

### 1.6 Scan-aware memory / resource planning

核心文件：

- `src/matrix_factorization/core/parallel/memory_estimator.py`
- `src/matrix_factorization/core/parallel/memory_estimator_general.py`
- `src/matrix_factorization/core/parallel/resource_execution.py`
- `src/matrix_factorization/core/memory_calibration.py`
- `docs/parallel_memory_contract.md`
- `docs/parallel_memory_review_queue.md`

当前规则：

- canonical scan 按 non-alpha group 分组。
- 当前真实自动 folding 维度只有 `alpha`。
- `sample/student` folding 仍被 contract 禁用，直到 runner/algorithm 真正消费 `sample_range/sample_offset`。
- `size/init/damping/onsager/max_steps` 默认不跨 group folding。

重要限制：

- 不要说“所有组合的智能并行都完成了”。
- 已有 calibration 只覆盖当前本地 GPU、当前 dtype/compile/profile。
- 换 GPU、dtype、compile、tensor order、internal batching 后必须重新 calibration。
- OOM 后同进程自动 retry 仍未完成。

### 1.7 Runtime probe / intervention

核心文件：

- `src/matrix_factorization/modules/interventions/`
- `src/matrix_factorization/modules/algorithms/agd.py`
- `tests/test_contract_runtime_extensions.py`

已完成：

- `batch_summary` 是 runner-level `after_batch` active probe。
- `state_slice` 是 AGD `after_step` active probe。
- probe 只读、轻量 JSON summary，不复制大 tensor，不改变训练数值。

未完成：

- `tensor_state_slice` 和 `variance_slice` 仍是 `declared_only`。
- warm/cold/adaptive restart 仍主要是 spec/metadata 映射，没有完全迁移成独立 intervention executor。
- Metropolis-like kick 没开始。

## 2. 最近实验和物理诊断

### 2.1 重要 run 目录

这些是最近对话中反复引用的结果目录。

```text
runs/20260427_193005_bgs_N1000_M50_init5-a41_cbd228
```

- `N=1000, M=50, S=10, max_steps=5000`
- `init_overlap = 0.0,0.2,0.4,0.6,0.8`
- `alpha = 0..4 step 0.1`
- `bigamp_spreading`
- Gaussian teacher
- Ising F
- 轻量结果，没有保存 W/X tensor，因此不能事后计算新的 gauge metric。

```text
runs/20260427_214246_bgs_N1000_M50_init5-a41_f08cc9
```

- 同样是 `N=1000, M=50, S=10, max_steps=5000`
- `init_overlap = 0.0,0.2,0.4,0.6,0.8`
- `alpha = 0..4 step 0.1`
- `bigamp_spreading`
- Ising/Rademacher teacher
- Ising F
- 保存了 per-point tensors，适合 posthoc factor diagnostic。

Posthoc gauge plot 输出在：

```text
runs/20260427_214246_bgs_N1000_M50_init5-a41_f08cc9/plots/posthoc_gauge_aligned
```

重要文件：

- `compare_G_w_by_warmstart.png`
- `compare_G_x_by_warmstart.png`
- `compare_G_wx_mean_by_warmstart.png`
- `overlay_W_gauge_init_0.png`
- `gauge_aligned_metrics.csv`
- `gauge_aligned_metrics.json`
- `README.md`

### 2.2 Sign / gauge 诊断为什么重要

这部分比 OCR 查找更重要。它来自用户对 spreading cold start 现象的追问：

- random `F` 应该破坏普通 matrix factorization 的 rotation symmetry。
- 但实验上，cold start 的 raw `A.w` 很低，而 warm start 只要有一点重合度就能学到很高结果。
- 用户怀疑：是不是算法其实学到了输出或 latent channel，但 `W/X` 的符号或尺度 gauge 没对齐，导致 raw projection 看起来低。

最后确认的数学结构：

普通 matrix factorization：

```math
Y = W X
```

存在一般 gauge：

```math
W\to W R,\qquad X\to R^{-1}X
```

spreading matrix：

```math
Y_{ij}
=
\sum_k F_{ijk}W_{ik}X_{kj}
```

随机 `F_{ijk}` 破坏一般 rotation/mixing，即通常不能再用任意矩阵 `R` 混合 latent components。但它仍保留每个 latent component 的 diagonal gauge：

```math
W_{:k}\to a_k W_{:k},
\qquad
X_{k:}\to a_k^{-1}X_{k:}
```

其中 sign gauge 是特例：

```math
a_k=\pm1
```

这就是为什么只翻转同一个 latent index `k` 的整列 W 和整行 X 后，所有包含这个 `k` 的项都不变：

```math
(-W_{ik})(-X_{kj})=W_{ik}X_{kj}
```

用户一开始误以为这是“翻转某个 observation 相关的一行或一列”，后来澄清为：不是按 observation 翻转，而是按 latent component 翻转。

### 2.3 已加入或讨论过的 sign diagnostic

为了验证 cold start 是否只是 sign sector 没对齐，新增/讨论了：

```text
Q_W_SIGN_ALIGNED_mean/std
Q_X_SIGN_ALIGNED_mean/std
```

定义：

```math
Q_W^{sign}
=
\frac{\sum_k |\langle W_s[:,k], W_t[:,k]\rangle|}
{\sum_k \|W_t[:,k]\|^2}
```

```math
Q_X^{sign}
=
\frac{\sum_k |\langle X_s[k,:], X_t[k,:]\rangle|}
{\sum_k \|X_t[k,:]\|^2}
```

语义：

- 这是 sign-gauge diagnostic。
- 不替代 raw `Q_W/Q_X`。
- 不处理 permutation。
- 不处理 continuous scale gauge。
- 不处理 general rotation。
- 用来判断“没学到 factor”还是“学到了 component，但每个 component 的 sign sector 不同”。

plot shorthand：

```text
D.w = Q_W_SIGN_ALIGNED
D.x = Q_X_SIGN_ALIGNED
```

这部分应该保留在 metric system 中，即使它没有最终解释所有现象；因为它是判断 gauge 问题的必要 diagnostic。

### 2.4 Posthoc scale gauge 诊断

用户随后指出：除了符号，还有 scale 自由度。

spreading matrix 仍有：

```math
W_{:k}\to a_k W_{:k},
\qquad
X_{k:}\to a_k^{-1}X_{k:}
```

其中 `a_k` 可以是任意非零实数，不只是 `±1`。

提出的 posthoc diagnostic 是为每个 latent component 找一个 signed scale `g_k`：

```math
g_k
=
\arg\min_{g\ne0}
\left[
\|g W_s[:,k]-W_t[:,k]\|^2
+
\|g^{-1}X_s[k,:]-X_t[k,:]\|^2
\right]
```

然后计算 gauge-aligned 的 `Q_W/Q_X` 曲线。

注意：

- 这是 posthoc analysis，不改变训练。
- 它不应该进入默认 formal metric，除非之后加 `MetricSpec` 并明确叫 diagnostic。
- 它特别适合分析“训练中途是否已经学到 latent structure，但 gauge/scale 尚未对齐”。
- 用户关心的不是最终 `Y=WX` 是否已经拟合，而是 phase transition 前后 latent factors 是否已出现可恢复结构。

### 2.5 Gauge 诊断结论

对 21:42 Ising/Rademacher run，修正了 teacher RNG 重建问题后，posthoc 结果显示：

```text
init=0.0, alpha=4:
  Q_Y ~= 0.997
  raw A.w ~= 0.079
  sign/gauge aligned W/X ~= 0.998-0.999

init=0.2:
  raw A.w ~= 0.588
  sign/gauge aligned W/X ~= 0.998-0.999

init=0.4,0.6,0.8:
  raw and aligned metrics all high or near high.
```

解释：

- 对 Ising/Rademacher 这组数据，cold start 的 raw `A.w` 低，很大程度可以被 sign/scale gauge 解释。
- 但是这不等于所有 Gaussian prior / spreading run 的现象都被解释。
- 7:30 Gaussian run 没保存 factor tensors，不能事后算同样的 gauge diagnostic。
- 如果要验证 Gaussian teacher 下的 sign/scale gauge，需要重跑并开启保存 factor tensors 或专门保存轻量 gauge payload。

### 2.6 Precision profile 的状态和实测疑点

对话中还启动过 `precision_profile` 方向：

```text
safe
fast
aggressive
```

目标：

- `safe`：FP32 为主，用作基准。
- `fast`：student state / workspace 尽量 BF16，metrics/reductions 保持 FP32。
- `aggressive`：更多 large observation/workspace BF16，FP32 accumulation 保留。

用户后来要求用 `aggressive` 跑 spreading，并观察显存/速度是否改善。实际对话中的反馈是：

- 看起来 aggressive 相比旧版本并没有明显降低显存或显著提速。
- 这可能意味着：
  - precision profile 只是进入了 config/spec/metadata，但没有真正覆盖关键 large tensors；
  - 或者当前瓶颈不在 dtype，而在 graph/index/scatter/gather、CPU/GPU 同步、plot/save、compile/cache；
  - 或者 BF16 只影响少量 state，主显存被 indices/F/Y/intermediate buffers 占用；
  - 或者 memory estimator/metadata 声称 dtype 改了，但 runtime tensor dtype 没真正改。

因此下一轮不能说 precision migration 已完成。必须审计：

- `PrecisionPolicySpec` 是否只是声明。
- `algorithm_params.precision_profile` 是否真正进入 AGD / dense BigAMP / spreading / tensor serial / tensor parallel。
- `F/Y/student state/variance/workspace/reductions/metrics/artifacts` 各自真实 dtype。
- `MemoryEstimator` 是否按 role dtype 计算 bytes。
- `mf explain-config` 和 run metadata 是否展示真实 dtype，而不是预期 dtype。

建议新增一个专门文档：

```text
docs/precision_profile_audit.md
```

并用小 run 输出每类 tensor 的 dtype summary，不要只看 config。

## 3. 论文文件和 OCR/理论定位

用户最初给了一个不相关 OCR：

```text
/home/sucia/DeepCode/deepcode_lab/papers/1/1.md
```

这个不是目标论文。

在 MF 项目本地找到的目标论文是：

```text
/home/sucia/Matrix_Factorization/Tensor Factorization.pdf
```

标题：

```text
Graphical model for tensor factorization by sparse sampling
```

作者：

```text
Angelo Giorgio Cavaliere
Riki Nagasawa
Shuta Yokoi
Tomoyuki Obuchi
Hajime Yoshino
```

arXiv:

```text
2510.17886
```

注意：

```text
/home/sucia/Matrix_Factorization/docs/theory/tensor_gamp/1.md
```

不是这篇论文，而是另一篇：

```text
Tensor Generalized Approximate Message Passing
```

不要把 TeG-AMP 那篇当成当前 sparse sampling / graphical tensor factorization 的理论依据。

## 4. 目标论文中的关键定义

从 `Tensor Factorization.pdf` 抽出的关键公式：

### 4.1 变量

论文考虑 `N` 个 `M` 维向量：

```math
x_i=(x_{i1},x_{i2},\ldots,x_{iM})^\top\in\mathbb{R}^M,
\qquad i=1,\ldots,N
```

每个 observation 是一个 `p`-plet。

### 4.2 观测图和 alpha

每个变量节点被观测：

```math
c=\alpha M
```

次。

edge set size：

```math
N_\square=|E|=\frac{Nc}{p}=\frac{\alpha}{p}NM
```

论文还定义：

```math
\gamma=\frac{\alpha}{p}
```

所以如果论文图用的是 `gamma` 横轴，而代码用的是 `alpha` 横轴，需要乘以 `p` 做映射。

### 4.3 Noiseless observation

无噪声观测：

```math
\pi_\square
=
\frac{\lambda}{\sqrt{M}}
\sum_{\mu=1}^M
F_{\square,\mu}
\prod_{i\in\partial\square}x_{i\mu}
```

无噪声极限下：

```math
y_\square=\pi_\square
```

### 4.4 Prior 和 F

论文技术假设：

- prior zero mean。
- `F` 有两种情况：
  - deterministic: `F=1`
  - random spreading: zero mean, unit variance

论文说明 deterministic 和 random F 在 dense limit 下给出相同宏观结果，但 random F 能显著改善 message passing 收敛，尤其是 `p=2`。

这和用户当前偏好的设置一致：

- 不要随手把 F 改成 Gaussian。
- 默认保留 Rademacher/Ising `F = ±1`，除非用户明确要求。

## 5. 当前代码和论文定义的关键不一致风险

下一轮必须重点核对这一节，不能直接继续跑长实验。

### 5.1 Bipartite matrix spreading alpha 映射

当前 matrix spreading graph 代码：

```text
src/matrix_factorization/modules/graphs/supergraph.py
```

核心逻辑：

```text
C = floor(alpha * M * N1)
```

若把 matrix spreading 理解成 `p=2`，且平衡 `N1=N2=N`，论文总节点数是：

```math
N_{\mathrm{paper}}=N_1+N_2=2N
```

论文 edge count：

```math
|E|
=\frac{\alpha_{\mathrm{paper}}}{2}(2N)M
=\alpha_{\mathrm{paper}}NM
```

代码：

```math
|E|_{\mathrm{code}}=\alpha_{\mathrm{code}}M N_1
```

平衡情况下：

```math
\alpha_{\mathrm{code}}=\alpha_{\mathrm{paper}}
```

但如果论文图横轴是：

```math
\gamma=\alpha/p
```

那么 `p=2` 时：

```math
\alpha_{\mathrm{code}}=2\gamma_{\mathrm{paper}}
```

这点需要在所有论文复现实验图中明确标注。

### 5.2 General graph / tensor hypergraph alpha 不一致

当前 general graph：

```text
src/matrix_factorization/modules/graphs/supergraph_general.py
```

也使用：

```text
C = alpha * M * N1
```

这在 `N1 != N2` 或把总节点数解释成 `N1+N2` 时，需要重新映射。

当前 tensor hypergraph：

```text
src/matrix_factorization/modules/algorithms/bigamp/tensor_hypergraph.py
```

使用：

```text
C = alpha * sum(dims) * M
```

而论文若有 `p` 阶、总节点数 `N_total=sum(dims)`，则：

```math
|E|_{\mathrm{paper}}
=
\frac{\alpha_{\mathrm{paper}}}{p}N_{\mathrm{total}}M
```

因此这条路径下：

```math
\alpha_{\mathrm{code}}
=
\alpha_{\mathrm{paper}}/p
=
\gamma_{\mathrm{paper}}
```

但另一个 tensor supergraph 路径：

```text
src/matrix_factorization/modules/algorithms/bigamp/tensor_supergraph.py
```

在 equal dims 情况下使用：

```text
C = alpha * M * ref_dim
```

若 `dims=(N,N,...,N)` 且阶数 `p=n`，论文：

```math
|E|_{\mathrm{paper}}
=
\frac{\alpha_{\mathrm{paper}}}{p}(pN)M
=
\alpha_{\mathrm{paper}}NM
```

此时：

```math
\alpha_{\mathrm{code}}=\alpha_{\mathrm{paper}}
```

所以 tensor serial / tensor parallel / generic hypergraph 的 alpha convention 可能不一致。必须先审计，再跑论文对比。

### 5.3 Graph degree ensemble 不一致

论文假设每个变量节点精确 degree：

```math
|\partial i|=c=\alpha M
```

当前 matrix spreading `supergraph.py` 是对所有 edge 做 random permutation / top-k / sparse sampling，通常只保证平均 degree，而不是严格每个变量节点 exactly degree。

这可能影响 phase transition 和有限尺寸曲线，尤其是在 `N=1000, M=50` 这类不是极限的实验中。

后续任务：

- 查是否已有 regular / low-loop / degree-constrained graph generator。
- 如果没有，新增一个不改变旧路径的 `graph_profile: paper_regular` 或类似选项。
- 不要直接替换 legacy graph，否则旧结果不可比。

### 5.4 Normalization convention 可能不一致

论文公式把尺度放在 interaction/output 前面：

```math
\pi_\square
=
\frac{\lambda}{\sqrt{M}}
\sum_{\mu=1}^M
F_{\square,\mu}
\prod_{i\in\partial\square}x_{i\mu}
```

论文 prior 例子：

```math
P_{\mathrm{pri}}(x)=\frac12\delta(x-1)+\frac12\delta(x+1)
```

或

```math
P_{\mathrm{pri}}(x)=\mathcal{N}(0,1)
```

即 latent `x` 是 `O(1)` 尺度，不是 `O(1/sqrt(M))`。

因此如果代码当前同时做了：

```text
teacher/student latent scale = 1/sqrt(M)
interaction scale = 1/sqrt(M)
```

那就很可能不是论文 convention。对 `p=2` 粗略看，若 `x~O(1/sqrt(M))`，每项乘积约 `O(1/M)`，随机求和 std 约 `O(1/sqrt(M))`，再乘 `1/sqrt(M)` 后输出 std 约 `O(1/M)`，明显小于论文想要的 `O(1)` 量级。

下一轮必须做：

- 审计当前 `NormalizationSpec` 是否已经把 active algorithms 改成 `1/sqrt(M)` latent scale。
- 审计 AMP update 中 prior precision / posterior variance 是否同步迁移。
- 决定是否引入新的 profile：

```text
normalization_profile: paper_sparse_sampling
```

可能定义：

```text
latent prior variance = 1
student init variance = 1
prior precision base = 1
interaction scale = lambda / sqrt(M)
F variance = 1
```

同时保留当前 internal normalized profile 给旧实验，不要静默覆盖。

## 6. 用户已经明确的偏好/约束

### 6.1 运行环境

- 已经没有外部托管运行语义。
- 所有运行都是本地。
- `quick` 只表示小规模快速试跑，不表示远端执行环境。
- 大结果、checkpoint、`.pt`、完整 heatmap、大图、GPU calibration raw artifact 不提交。

### 6.2 配置入口

- 正式运行入口是：

```bash
mf src/matrix_factorization/config.yaml
```

- debug/trial 使用：

```bash
mf trial ...
```

- 用户希望正式配置模板有清晰注释和数字选项，不要要求手填长字符串。

### 6.3 F distribution

除非用户明确要求，不要把 F 改成 Gaussian。

当前默认物理偏好：

```text
F = Rademacher / ±1 / random spreading factor
```

teacher prior 可以在 Gaussian / Ising 之间切换，但 F 不要乱动。

### 6.4 Metrics

正式物理量：

- `Q_Y`
- `Q_W`
- `Q_X`
- `Q_N`

正式 diagnostics：

- `Q_W_COS_ROOT`
- `Q_X_COS_ROOT`
- `Q_W_SIGN_ALIGNED`
- `Q_X_SIGN_ALIGNED`

不要把 old cosine `Q_Y`、MSE、Gen_Error 重新当 formal metric。

### 6.5 Plot

用户希望：

- 科研风格 plot。
- 组间对比图带 error bar。
- 能从 ResultCube 按参数坐标挑曲线。
- 如果中途 scan point 完成，最好能保存可查看的中间结果；但如果 plot 需要跨组对比，可以最后统一生成。

这部分还有工程 backlog，不要说已完全完成。

## 7. 当前最重要的下一步

建议新对话不要立刻继续跑大实验，而是先做“论文 convention 对齐审计”。

### Step 1: 固化论文引用资料

建议新增：

```text
docs/theory/graphical_tensor_sparse_sampling.md
```

内容：

- paper title / arXiv / path。
- 关键公式：`c=alpha M`、`|E|=(alpha/p)NM`、`gamma=alpha/p`、`pi=lambda/sqrt(M)*sum(...)`。
- prior examples：Ising / Gaussian。
- F choices：deterministic / random zero mean unit variance。
- dense limit assumptions。

可以从 PDF 直接用 PyPDF2 抽文本，不必等待 OCR。

### Step 2: 写 code-vs-paper convention map

建议新增：

```text
docs/theory/code_vs_sparse_sampling_paper.md
```

必须逐项比较：

- alpha mapping：
  - bipartite matrix spreading
  - general graph
  - tensor_hypergraph
  - tensor_supergraph / tensor_parallel
- edge count `C`
- degree constraint：exact `c=alpha M` vs average degree sampling
- prior scale：paper `Var(x)=1` vs current code
- interaction scale：paper `lambda/sqrt(M)` vs current code
- F distribution and variance
- noise convention / noiseless limit
- metric convention：paper order parameter vs project metric schema v3

### Step 3: 审计当前 normalization / precision migration

当前 `docs/implementation_progress.md` 记录了未提交施工线：

```text
normalization schema v4 + precision profile migration
```

但用户后续质疑了这条线是否符合论文 convention。下一轮必须：

1. 查 `src/matrix_factorization/core/contracts.py` 中 `NormalizationSpec / PrecisionPolicySpec` 当前真实定义。
2. 查 teacher/student init：
   - `src/matrix_factorization/core/experiment/data_factory.py`
   - teacher modules
   - spreading/tensor algorithms
3. 查 AMP update prior precision / posterior variance 是否同步。
4. 查 `precision_profile: safe/fast/aggressive` 是否真正改变 dtype 路径和 memory estimator。
5. 运行 targeted tests：

```bash
python -m pytest -q tests/test_normalization_precision_contract.py
python -m pytest -q tests/test_config_contract.py tests/test_contract_parameter_specs.py
mf validate src/matrix_factorization/config.yaml
```

如果这条线没完成，不能写成 completed。

### Step 4: 决定实现方式

推荐不要直接把所有 active algorithms 强行改成单一 convention。更稳妥：

```yaml
algorithm_params:
  normalization_profile: paper_sparse_sampling
```

并保留：

```yaml
algorithm_params:
  normalization_profile: internal_normalized_legacy
```

或类似名字。

原因：

- 旧 result 不可静默比较。
- 当前代码里 matrix / spreading / tensor 可能已经混用不同 convention。
- 直接替换会让所有历史曲线失去可解释性。

### Step 5: 小实验验证

在完成 convention map 和 profile 实现后，先跑 small/quick：

```bash
mf trial run matrix_bigamp_quick
mf trial run scan_alpha_quick
mf trial run scan_mixed_axes_quick
```

然后再考虑正式 `N=1000, M=50, S=10/20` 的 spreading run。

## 8. 明确不要做的事

- 不要把 `docs/theory/tensor_gamp/1.md` 当成当前论文。
- 不要默认把 F 改成 Gaussian。
- 不要在没有 convention map 的情况下继续大规模比较论文曲线。
- 不要把 normalization profile 的未完成施工写成 completed。
- 不要提交 `runs/`、`results/`、`artifacts/` 或 `.pt`。
- 不要为了让图好看而改 metric 定义。
- 不要在 active path 偷读 `_batch_metrics` 或保存 tensor dummy W/X。

## 9. 推荐给下一轮的启动命令

```bash
cd /home/sucia/Matrix_Factorization
git status --short --branch
python -m pytest -q tests/test_config_contract.py tests/test_contract_parameter_specs.py tests/test_normalization_precision_contract.py
mf validate src/matrix_factorization/config.yaml
```

如果只是做论文审计，不需要跑 GPU。

如果要确认 PDF 文本：

```bash
python - <<'PY'
from PyPDF2 import PdfReader
p = "/home/sucia/Matrix_Factorization/Tensor Factorization.pdf"
r = PdfReader(p)
for i in range(3, 7):
    print(f"===== page {i+1} =====")
    print((r.pages[i].extract_text() or "")[:4000])
PY
```

## 10. 推荐接手优先级

1. `docs/theory/graphical_tensor_sparse_sampling.md`
2. `docs/theory/code_vs_sparse_sampling_paper.md`
3. Audit `NormalizationSpec / PrecisionPolicySpec` 当前代码是否真实完成。
4. 如果不符合论文，设计 `normalization_profile: paper_sparse_sampling`，不要直接覆盖旧 convention。
5. 增加 normalization fixture：
   - paper prior variance = 1
   - output variance scale with `lambda/sqrt(M)`
   - alpha edge count mapping
   - graph degree exact/average distinction
6. 更新 `mf explain-config` 显示 normalization convention、paper alpha/gamma mapping。
7. 通过 quick trials 后再跑正式 spreading experiment。

## 11. 2026-04-28 05:00 scan / plotting / gauge / adaptive 状态

本节记录 2026-04-28 凌晨正式 `onsager_policy × alpha` run 暴露出来的问题。这里的内容比 OCR 查找更重要，下一轮 agent 应优先阅读。

### 11.1 当前正式 run 状态

正式 run 根目录：

```text
/home/sucia/Matrix_Factorization/runs/20260428_043906_bgs_N200_M50_ons3-a41_S100_steps2000_a4258e
```

配置意图：

- `algorithm = bigamp_spreading`
- `N1=N2=200, M=50`
- `samples_per_alpha = 100`
- `max_steps = 2000`
- `F = ising`
- `teacher = gaussian`
- `precision_profile = aggressive`
- `scan axes = onsager_policy × alpha`
- `onsager_policy` 三组：
  - `no_onsager`
  - `onsager_fixed_beta005`
  - `onsager_adaptive_beta005`

完成情况：

- `groups/000_onsager_policy-no_onsager` 已完成。
- `groups/001_onsager_policy-onsager_fixed_beta005` 已完成。
- 第三组 `onsager_adaptive_beta005` 在第一个 batch 失败。

失败位置：

```text
src/matrix_factorization/modules/algorithms/bigamp/spreading.py
  _train_full_parallel_adaptive()
    -> bigamp_step_disjoint_union_flat_adaptive_legacy_fast()
      -> bigamp_step_disjoint_union_flat_legacy_fast()

src/matrix_factorization/modules/algorithms/bigamp/step.py
  line around tau_X_contrib / X variance update
```

报错：

```text
RuntimeError: CUDA driver error: unknown error
canonical scan failed at group 3/3 |
onsager_policy=onsager_adaptive_beta005 |
batch 9/13 | alpha 0.00-1.40 | points 82/123
```

重新只跑 adaptive 组后，同样在第一批 `alpha 0.00-1.40` 失败。说明这不是主 run 聚合状态造成的单次偶发。

### 11.2 已做但尚未 commit 的代码修改

当前工作树包含未提交修改。不要直接回滚；需要下一轮 agent 审核后继续。

相关文件：

```text
src/matrix_factorization/modules/algorithms/bigamp/spreading.py
src/matrix_factorization/core/experiment/runner.py
src/matrix_factorization/core/experiment/result.py
src/matrix_factorization/config.yaml
```

修改意图：

1. 恢复 spreading 的 legacy fast no-Onsager / fixed-Onsager 快路径。
2. `adaptive_damping=true` 时禁用 torch.compile，避免 inductor/CUDA graph driver error。
3. canonical group 中间结果保存时，允许把 fixed `series_by` 轴剥离后生成组内 PlotQuery 图。
4. heatmap 支持 `Q_W_SIGN_ALIGNED`，使 heatmap 至少可以画 sign-aligned W，而不是只能画旧 Gram/raw fallback。
5. 默认配置里 `output.heatmap_metric` 改成 `Q_W_SIGN_ALIGNED`。

已通过的 targeted tests：

```bash
python -m pytest -q tests/test_result_cube.py tests/test_plot_query.py tests/test_scan_runner.py
python -m pytest -q tests/test_bigamp_onsager_convention.py tests/test_spreading_batch_metrics.py tests/test_config_contract.py
```

未完成验证：

- 这些修改尚未跑完整 `python -m pytest -q`。
- `Q_W_SIGN_ALIGNED` heatmap runtime 还需要用小 group result 验证。
- adaptive full-size 仍失败，因此不能把 adaptive 写成完成。

### 11.3 Adaptive Onsager 的实际状态

不要把 adaptive compile 问题理解成“随便开关 compile”。当前事实是：

1. `no_onsager` 和 `onsager_fixed_beta005` 的 compiled fast path 可以跑，并且 GPU 利用率正常。
2. `onsager_adaptive_beta005` 的 compiled path 在正式尺寸触发 CUDA driver error。
3. 禁用 compile 后，极小尺寸 debug 通过：

```text
S=2, steps=2, alpha=0.0..0.1
precision=safe      -> pass
precision=aggressive -> pass
```

4. 禁用 compile 后，正式尺寸 `S=100, steps=2000, alpha batch 0.00..1.40` 仍失败。

因此 adaptive 的问题不是单纯 compile 开关；更可能与以下因素之一有关：

- adaptive path 在大 batch 下的 `scatter_add_` / variance update / BF16 workspace；
- alpha batch 太宽，导致某个 CUDA kernel 参数或临时张量布局触发 driver bug；
- adaptive 的 safe-state / acceptance logic 持有额外大张量，和 legacy fast step 的临时张量叠加；
- `max_steps` UI 显示为 `1000` 但 config 是 `2000`，说明 step display 或 effective max_steps 还有显示/传递不一致，需要审计。

下一步建议：

1. 不要直接跑 41 点 adaptive 大 scan。
2. 先用 `CUDA_LAUNCH_BLOCKING=1` 跑单 alpha / 小 alpha batch：

```yaml
scan:
  axes:
    onsager_policy:
      kind: composite
      values:
        onsager_adaptive_beta005: ...
    alpha:
      path: alpha
      values: [0.0]
```

3. 分别测试：

```text
S=100, steps=2
S=100, steps=50
S=100, steps=2000
precision=safe / fast / aggressive
alpha batch size = 1 / 2 / 4 / 8
```

4. 只有确认 adaptive 大尺寸稳定后，才恢复第三组正式 scan。

### 11.4 Group-level plotting 问题

用户要求：多组 scan 时，每一组如果可以独立解释，就在该组完成后立即生成完整普通图，而不是只生成 heatmap，最终跨组对比图仍在总 run 完成后生成。

当前问题：

- group-level 中间保存已写：
  - `metrics.json`
  - `metadata.json`
  - `manifest.json`
  - heatmap/GIF
  - `artifacts/points/*/results.pt`
- 但普通 PlotQuery 曲线没有生成。

原因：

```text
src/matrix_factorization/core/experiment/runner.py
  _group_independent_plot_configs()
```

原逻辑看到 `series_by: [onsager_policy]`，而 group 本身已经固定了 `onsager_policy`，于是把这些 plot 全部跳过。

已做修改：

- 如果 `series_by` 的轴已经被 group 固定，则从 `series_by` 中剥离该轴，而不是跳过该 plot。
- 文件名自动加 suffix，例如：

```text
qy_by_onsager_policy_onsager_policy-no_onsager.png
```

这部分需要下一轮 agent 用小 canonical scan 验证 runtime。

### 11.5 普通图不够科研风格的问题

用户指出应急补图“不像科研绘图”。这个判断是对的。

应急补图路径：

```text
groups/000_onsager_policy-no_onsager/plots/curves_QY_projection.png
groups/000_onsager_policy-no_onsager/plots/curves_WX_projection.png
groups/000_onsager_policy-no_onsager/plots/curves_WX_diagnostics.png
```

这些图是临时脚本直接画的，只用于即时查看，不应作为正式输出风格。

正式图应该走：

```text
src/matrix_factorization/core/experiment/result.py
  _plot_result_cube_queries()

src/matrix_factorization/modules/outputs/publication_style.py
```

下一步应该：

- 删除或不再依赖手工补图逻辑。
- 确保 group-level save 调用 `ExperimentResult.save(... output_options=group_output_options)` 时真正触发 `_plot_result_cube_queries()`。
- PlotQuery 图必须带 errorbar，使用 `*_std`。
- legend 来自 scan coordinate，不靠目录名猜。

### 11.6 Sign 与 scale-gauge 图

当前已有 formal sign-aligned metric：

```text
Q_W_SIGN_ALIGNED_mean/std
Q_X_SIGN_ALIGNED_mean/std
```

简写：

```text
D.w = Q_W_SIGN_ALIGNED
D.x = Q_X_SIGN_ALIGNED
```

相关文件：

```text
src/matrix_factorization/modules/metrics/overlap.py
src/matrix_factorization/modules/metrics/spreading.py
src/matrix_factorization/modules/outputs/plot_registry.py
src/matrix_factorization/core/contracts.py
```

用户提到的 `Q_w gauged` 更准确地说是 posthoc scale-gauge aligned diagnostic，不是当前 formal MetricSpec。

现有脚本：

```text
scripts/analysis/posthoc_scale_gauge.py
```

它计算：

```text
Q_W_SCALE_GAUGE_mean/std
Q_X_SCALE_GAUGE_mean/std
Q_WX_SCALE_GAUGE_mean
median_abs_log_g_mean
```

当前问题：

- 该脚本默认按 ResultCube `point_id` 找：

```text
artifacts/points/<point_id>/results.pt
```

- group result 实际保存路径是：

```text
artifacts/points/<alpha>/results.pt
```

因此脚本对 group 目录会写出 0 rows。需要修脚本，让它支持：

1. `point_id` 路径；
2. alpha 路径；
3. 或从 `result_cube.points[point_id].coordinates.alpha` 反查路径。

是否要把 scale-gauge 升级为 formal metric：

- 短期：建议保留为 posthoc diagnostic，因为它需要保存 W/X tensor，计算成本高，不适合默认每步/每 batch 都算。
- 中期：可以新增 `GaugeMetricSpec` 或 `AnalyzerSpec`，把它纳入 result post-analysis，而不是训练主 metric。
- 如果用户希望默认绘图包含它，可以在 `output.plots` 支持 posthoc plot source，但这需要 OutputSpec 声明依赖 `matrix_factors`。

### 11.7 Heatmap 应该画 sign 或 gauge

当前 heatmap 旧逻辑：

- 如果 metrics 里有 `overlap_matrix`，用它；
- 否则用 `W_students` 和 `W_teacher` 生成 `Q_W` Gram heatmap。

用户希望至少切成 sign heatmap，或者 gauge heatmap。

已做修改：

```text
src/matrix_factorization/core/experiment/result.py
```

支持：

```yaml
output:
  heatmap_metric: Q_W_SIGN_ALIGNED
```

它会用 `sign_aligned_projection_abs(... latent_axis=-1)` 生成 W sign-aligned replica/teacher heatmap，并保存成：

```text
heatmap_W_sign_*_alpha_*.png
animation_D.gif 或类似 suffix
```

注意：

- gauge heatmap 尚未实现。
- scale-gauge heatmap 需要同时用 W 和 X，并求每对 replica 的最佳 diagonal scale；这比 sign heatmap 更贵，建议作为 posthoc analyzer，不建议默认每个 alpha 都开。

### 11.8 第一组 / 第二组当前趋势提示

第一组 `no_onsager` 已读出：

- `Q_Y`、`Q_W_COS_ROOT`、`Q_X_COS_ROOT`、`Q_W_SIGN_ALIGNED`、`Q_X_SIGN_ALIGNED` 在 `alpha ≈ 3.5` 附近跳升。
- `alpha ≈ 3.6` 后很多指标超过 `0.8`。
- `alpha = 4.0` 时 `Q_Y` 和 Gram/sign 指标接近 `1`。
- 原始 `Q_W/Q_X` 仍只有约 `0.12`，说明 coordinate projection 被 sign/scale/gauge 敏感性压低。

第二组 `onsager_fixed_beta005` 完成，但快速检查看到：

```text
Q_Y_mean(alpha=4.0) ≈ 1.36e6
```

这不是合理 overlap 数值；说明 fixed Onsager 在当前 aggressive / normalization / damping 组合下可能数值爆炸，或者 metric 输入尺度爆了。不要把第二组当成成功物理结果。

### 11.9 新 agent 不应做的事

- 不要把第三组 adaptive 写成“已完成”。
- 不要继续盲跑 `onsager_adaptive_beta005` 大 scan。
- 不要把手工补图当成正式 plotting 修复。
- 不要把 scale-gauge 直接塞进每步训练 metric，除非先声明依赖和计算成本。
- 不要把 heatmap 默认为 raw `Q_W`，用户希望至少是 sign-aligned。
- 不要提交 `runs/` 或 `.pt`。
- 不要直接 revert 当前未提交代码；先审查这些 patch 是否合理。

### 11.10 2026-04-28 05:30 后续执行结果

已完成的正式 run 仍是：

```text
runs/20260428_043906_bgs_N200_M50_ons3-a41_S100_steps2000_a4258e
```

该 run 的 `no_onsager` 组是当前可用 baseline：

- `alpha=4.0`：`Q_Y_mean≈0.9964`
- `Q_W_SIGN_ALIGNED_mean≈1.0049`
- `Q_X_SIGN_ALIGNED_mean≈0.9907`
- `Q_W_COS_ROOT_mean≈0.9956`
- raw coordinate `Q_W/Q_X≈0.118`，仍然是 gauge-sensitive 读数。

`onsager_fixed_beta005` 组已确认发散：

- `Q_Y_mean` 多个 alpha 在 `10^5-10^7` 量级；
- posthoc scale-gauge 对 alpha=1/2 出现 non-finite，对 alpha=3/4 仍是
  `10^2-10^3` 量级；
- 这不是绘图或 gauge 对齐问题，而是 factor/output scale 已经爆掉。

已补的输出：

```text
runs/20260428_043906_bgs_N200_M50_ons3-a41_S100_steps2000_a4258e/groups/000_onsager_policy-no_onsager/plots/*_onsager_policy-no_onsager.png
runs/20260428_043906_bgs_N200_M50_ons3-a41_S100_steps2000_a4258e/groups/001_onsager_policy-onsager_fixed_beta005/plots/*_onsager_policy-onsager_fixed_beta005.png
runs/20260428_043906_bgs_N200_M50_ons3-a41_S100_steps2000_a4258e/plots/posthoc_no_vs_fixed/
```

posthoc gauge 脚本已修：

- 支持 group result 的 `artifacts/points/<alpha>/results.pt` 路径；
- 遇到 non-finite factor 时输出 `NaN` diagnostic，不再崩溃；
- 新增测试 `tests/test_posthoc_scale_gauge.py`。

新增/修正的 runtime 行为：

- adaptive Onsager 不再被构造期硬关 compile；现在尝试 compiled adaptive
  default-mode step，失败才 fallback。
- canonical scan 会在 group 切换时清理算法 compile cache，并保留 child
  `execution_metadata`，方便确认 effective compile/dtype。
- `scan.execution.allowed_fold_axes: []` 现在真正能关闭 alpha folding，按单
  alpha batch 执行；这用于后续隔离 adaptive CUDA 问题。

追加 trial 结论：

```text
runs/trials/adaptive_route_probe/20260428_052543_bgs_N200_M50_ons1-a41_S100_steps100_eebe31
runs/trials/adaptive_route_probe/20260428_053508_adaptive_cap005_probe_bgs_N200_M50_ons1-a6_S100_steps500_57f6fe
runs/trials/onsager_damping_probe/20260428_052930_onsager_damping_probe_bgs_N200_M50_ons5-a6_S100_steps500_f770ef
```

- adaptive standalone 可以进入 compiled route，metadata 显示
  `compile_status=effective_for_adaptive_spreading_step`、`storage_dtype=bfloat16`。
- 但 adaptive 指标不物理：低 alpha 出现 `Q_Y>1` 和 sign-aligned latent
  projection `>1`，说明 over-scale/overfit。
- fixed beta 从 `0.05` 降到 `0.0005` 仍不能避免 500-step 爆炸。
- mixed scan 中 fixed 组发散后再进入 adaptive 组仍可能触发 CUDA driver error；
  不要把这个解释为单纯显存不够，也不要靠永久关闭 compile 掩盖。

当前建议：

- 继续使用 `no_onsager` 作为正式 baseline。
- Onsager 线先停止大跑，回到 reference BiGAMP 的 `pvar/zvar/svar/shat`、
  value/cost 和 damping state machine 做逐项修正。
- 若必须跑 adaptive，只能先用单独 trial/单独进程隔离；不要接在已知发散的
  fixed Onsager group 后面。

## 2026-04-28 21:30 metric schema v4 迁移状态

当前 active metric profile 已迁移为 `physical_overlap_v1`：

- `Q_Y_mean/std`：不再是 absolute projection；现在是 `1 - NMSE_Y`。
- `NMSE_Y_mean/std`：同步输出，用来解释 `Q_Y`。
- `Q_W_mean/std`、`Q_X_mean/std`：不再除以 teacher norm squared；现在是固定分母 physical overlap。
- `Q_N_mean/std`：tensor latent factor 同步改为固定分母 overlap。
- 旧 projection 指标保留为 `Q_Y_PROJ_ABS`、`Q_W_PROJ_ABS`、`Q_X_PROJ_ABS`。
- `Q_W_SIGN_GAUGE` / `Q_X_SIGN_GAUGE` 是正式 sign-gauge 名；`Q_W_SIGN_ALIGNED` / `Q_X_SIGN_ALIGNED` 只作为 legacy alias 保留。
- `median_abs_log_k` 是正式 scale-gauge magnitude 名；`median_abs_log_g` 只作为 legacy alias 保留。
- `metrics.json.metric_schema.schema_version = 4`，并写入 `metric_definition_profile = physical_overlap_v1`。

不要把 schema v3 的 `Q_Y_mean/Q_W_mean/Q_X_mean` 和 schema v4 的同名字段直接比较。
正式定义见 `docs/METRICS_GUIDE.md`。
