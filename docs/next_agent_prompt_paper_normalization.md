# Next Agent Prompt: Paper Convention / Normalization Handoff

把下面这段直接贴给新的 Codex agent。

```text
你接手的是 /home/sucia/Matrix_Factorization，必须在 dev 分支工作。先阅读：

- AGENTS.md
- docs/project_handoff_2026-04-28.md
- docs/project_handoff_2026-04-27.md
- docs/METRICS_GUIDE.md
- docs/metric_naming_decisions.md
- docs/scan_system_contract.md
- docs/parallel_memory_review_queue.md

当前任务不是立刻跑大实验，也不是继续随意改物理公式。请先做“Graphical model for tensor factorization by sparse sampling”论文 convention 与当前代码 convention 的审计和交接。

关键背景：

1. 目标论文在：
   /home/sucia/Matrix_Factorization/Tensor Factorization.pdf

   标题是：
   Graphical model for tensor factorization by sparse sampling

   不是：
   /home/sucia/Matrix_Factorization/docs/theory/tensor_gamp/1.md

   后者是另一篇 TeG-AMP 论文，不要混用。

2. 论文关键定义：

   - N 个 M 维向量 x_i。
   - 每个变量节点 degree c = alpha M。
   - edge count:
     |E| = Nc/p = (alpha/p) N M
   - gamma = alpha/p。
   - noiseless observation:
     pi_square = lambda / sqrt(M) * sum_mu F_square,mu * product_{i in partial square} x_i_mu
   - Ising prior 是 ±1，Gaussian prior 是 N(0,1)，即 latent x 是 O(1) convention。
   - F 可以是 deterministic F=1 或 random zero-mean unit-variance；用户当前偏好默认 F=Rademacher/±1，不要随手改成 Gaussian。

3. 当前代码可能存在 convention 不一致：

   - matrix spreading graph:
     src/matrix_factorization/modules/graphs/supergraph.py
     uses C = alpha * M * N1

   - general graph:
     src/matrix_factorization/modules/graphs/supergraph_general.py
     uses C = alpha * M * N1

   - tensor_hypergraph:
     src/matrix_factorization/modules/algorithms/bigamp/tensor_hypergraph.py
     uses C = alpha * sum(dims) * M

   - tensor_supergraph / tensor parallel:
     src/matrix_factorization/modules/algorithms/bigamp/tensor_supergraph.py
     uses C = alpha * M * ref_dim

   这些路径和论文 alpha/gamma 的映射可能不同。请先写 code-vs-paper convention map，不要直接改。

4. 最近曾经引入 NormalizationSpec / PrecisionPolicySpec / precision_profile: safe/fast/aggressive，但这条施工线在当前工作树里未确认完全闭环。用户后续质疑它是否符合论文 convention。不要把它写成 completed。必须先审计：

   - src/matrix_factorization/core/contracts.py
   - src/matrix_factorization/core/experiment/data_factory.py
   - src/matrix_factorization/modules/algorithms/bigamp/spreading.py
   - src/matrix_factorization/modules/algorithms/bigamp/tensor_*.py
   - src/matrix_factorization/core/parallel/memory_estimator.py
   - tests/test_normalization_precision_contract.py

5. 用户已经确认的 metrics：

   - Q_Y = absolute projection, not cosine.
   - Q_W/Q_X = coordinate projection.
   - Q_N = tensor latent node/spin/factor projection.
   - Q_W_COS_ROOT / Q_X_COS_ROOT = Cos-root diagnostic.
   - Q_W_SIGN_ALIGNED / Q_X_SIGN_ALIGNED = sign-gauge diagnostic.
   - MSE / Gen_Error / old cosine Q_Y 不再是 formal metrics.

6. Sign / gauge 诊断是当前交接重点之一，不要只关注 OCR。

   spreading matrix 形式：

   Y_ij = sum_k F_ijk W_ik X_kj

   random F 破坏一般 rotation/mixing，但仍保留 diagonal gauge：

   W_:k -> a_k W_:k
   X_k: -> a_k^{-1} X_k:

   sign gauge 是 a_k = ±1。

   已新增或讨论过：

   - Q_W_SIGN_ALIGNED / Q_X_SIGN_ALIGNED
   - plot shorthand: D.w / D.x
   - posthoc scale-gauge aligned diagnostic:
     choose g_k to minimize ||g W_s[:,k]-W_t[:,k]||^2 + ||g^{-1} X_s[k,:]-X_t[k,:]||^2

   这部分是 diagnostic，不应直接改变训练轨迹。它用于判断 cold start 下 raw Q_W/Q_X 低是不是因为 sign/scale gauge 没对齐，而不是没学到 latent structure。

   重要 run：

   /home/sucia/Matrix_Factorization/runs/20260427_214246_bgs_N1000_M50_init5-a41_f08cc9

   这个 run 保存了 per-point tensors，可以做 posthoc gauge 诊断。已生成：

   /home/sucia/Matrix_Factorization/runs/20260427_214246_bgs_N1000_M50_init5-a41_f08cc9/plots/posthoc_gauge_aligned

   7:30 Gaussian run 没保存 factor tensors，不能事后算同样 gauge metric；如需验证 Gaussian teacher，必须重跑并保存必要 tensor 或轻量 gauge payload。

7. Precision profile 是另一条已动手但未闭环的线。

   曾引入：

   - precision_profile: safe
   - precision_profile: fast
   - precision_profile: aggressive

   用户观察到 aggressive 对显存/速度似乎没有明显改善，所以不能把 precision migration 写成完成。必须审计它是真正改变了 runtime tensor dtype，还是只进入了 config/spec/metadata。

   需要查：

   - PrecisionPolicySpec
   - algorithm_params.precision_profile
   - AGD / dense BigAMP / spreading / tensor serial / tensor parallel 的实际 dtype
   - F/Y/student state/variance/workspace/reductions/metrics/artifacts 的 role dtype
   - MemoryEstimator 是否按真实 role dtype 计算

   建议新增 docs/precision_profile_audit.md，并用小 run 输出实际 dtype summary。

8. 先不要提交 runs/results/artifacts/.pt。大 GPU 实验不是本任务第一步。

请执行：

Step A:
  git status --short --branch
  python -m pytest -q tests/test_config_contract.py tests/test_contract_parameter_specs.py tests/test_normalization_precision_contract.py
  mf validate src/matrix_factorization/config.yaml

Step B:
  新增或更新 docs/theory/graphical_tensor_sparse_sampling.md，记录目标论文的关键公式、路径、alpha/gamma、normalization、F/prior。

Step C:
  新增 docs/theory/code_vs_sparse_sampling_paper.md，逐项比较当前代码与论文：
  - alpha mapping
  - edge count C
  - exact degree c=alpha M vs random average degree
  - prior scale
  - interaction scale lambda/sqrt(M)
  - F distribution
  - noiseless output
  - metrics convention

Step D:
  审计 NormalizationSpec / PrecisionPolicySpec 当前实现。给出明确结论：
  - 哪些已经真实接入
  - 哪些只是 spec/metadata
  - 哪些会改变数值
  - 哪些测试覆盖不足

  Precision 审计里必须回答：为什么 aggressive 没有明显减少显存/提速？是没有真正生效，还是瓶颈在 indices/F/Y/intermediate buffers/CPU-GPU sync/save/plot？

Step E:
  如果需要代码修改，优先设计 profile，而不是静默覆盖旧 convention：

  algorithm_params:
    normalization_profile: paper_sparse_sampling

  可能定义：
  - latent prior variance = 1
  - student init variance = 1
  - prior precision base = 1
  - interaction scale = lambda/sqrt(M)
  - F variance = 1

  旧 convention 应保留为 legacy/internal profile，防止旧结果失去解释。

先完成文档和审计，再决定是否实现 profile。所有结论必须引用具体文件路径和函数/类。不要把未验证内容写成已完成。
```
