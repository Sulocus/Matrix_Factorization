# Code vs Sparse Sampling Paper Convention Map

审计日期：2026-04-28。

论文来源：`Tensor Factorization.pdf`，标题
`Graphical model for tensor factorization by sparse sampling`。

这是一份 audit map。2026-04-28 的后续实现已经新增
`algorithm_params.normalization_profile`，并把默认 profile 对齐到论文的
`paper_sparse_sampling` latent/prior scale。旧的内部 convention 保留为
`internal_normalized_legacy`。

## 总结

论文使用：

```text
x = O(1)
Var(x) = 1
exact node degree c = alpha M
edge count |E| = (alpha / p) N M
interaction/output scale = lambda / sqrt(M)
```

当前默认 active code 使用：

```text
normalization_profile = paper_sparse_sampling
latent entry variance = 1
student variance = 1
prior precision base = 1
interaction/output scale = 1/sqrt(M)
```

旧 convention 仍可显式选择：

```text
normalization_profile = internal_normalized_legacy
latent entry variance = 1/M
student variance = 1/M
prior precision base = M
interaction/output scale = 1/sqrt(M)
```

也就是说，归一化已经可以和论文对齐；但 alpha/edge count 和 exact-degree
ensemble 仍没有完全对齐论文。

## Alpha 与 Edge Count

| Route | 当前代码位置 | 代码里的 edge count | 与论文的 alpha/gamma 映射 |
| --- | --- | --- | --- |
| Matrix spreading bipartite | `src/matrix_factorization/modules/graphs/supergraph.py:create_supergraph` | `C = floor(alpha_code * M * N1)` | 若 `N1=N2=N`，把它解释为 `p=2` 且论文总 node 数是 `2N`，论文给出 `|E| = alpha_paper N M`，因此 `alpha_code = alpha_paper`。若和论文 `gamma=alpha/p` 比较，则 `alpha_code = 2 gamma_paper`。 |
| General graph | `src/matrix_factorization/modules/graphs/supergraph_general.py:create_supergraph_general` | 在 `N1+N2` 个 nodes 的 complete graph 上取 `C = floor(alpha_code * M * N1)` | 论文 `p=2` 且总 node 数 `N_total=N1+N2` 时，`|E| = alpha_paper N_total M / 2`。因此 `alpha_code = alpha_paper * (N1+N2)/(2N1)`。平衡维度时退化成 `alpha_code = alpha_paper`。 |
| Tensor serial hypergraph | `src/matrix_factorization/modules/algorithms/bigamp/tensor_hypergraph.py:generate_tensor_hypergraph` | `C = int(alpha_code * sum(dims) * M)` | 如果 `sum(dims)` 对应论文总 node 数，tensor order 是 `p`，论文给出 `|E| = alpha_paper sum(dims) M / p`。因此这里 `alpha_code = alpha_paper / p = gamma_paper`。 |
| Tensor supergraph / parallel | `src/matrix_factorization/modules/algorithms/bigamp/tensor_supergraph.py:create_tensor_supergraph` | `C = int(alpha_code * M * ref_dim)`，其中 `ref_dim=dims[0]` | 若各 mode 等维 `(N,...,N)` 且 order 是 `p`，论文给出 `|E| = alpha_paper N M`，因此 `alpha_code = alpha_paper`。若 dims 不等，这条路径不是 `sum(dims)/p` convention，而是锚定在 `dims[0]`。 |
| Tensor memory estimator | `src/matrix_factorization/core/parallel/memory_estimator.py:get_tensor_spreading_breakdown` | `C = ceil(alpha * sum(tensor_dims) * M)` | 这更接近 serial hypergraph 的 DoF convention，不是 tensor supergraph convention。等维 order `p` 时，它是 tensor-supergraph edge count 的 `p` 倍。 |

结论：同一个 YAML 字段名 `alpha` 在不同 route 里不总是同一个论文量。
matrix spreading 的平衡二部图可以自然解释为论文的 `alpha`；serial tensor
hypergraph 当前更像论文的 `gamma=alpha/p`；tensor supergraph/parallel 在等维
时又更像论文的 `alpha`。

## Degree Ensemble

论文要求：

```text
每个 variable node 的 degree 精确等于 c = alpha M
```

当前代码是：

- `supergraph.py:create_supergraph` 从所有 bipartite edges 的随机顺序中取
  prefix，或用随机 score 做 top-k。它固定总边数，但每个 node 的 degree 是
  random average-degree，不是精确 `alpha M`。
- `supergraph_general.py:create_supergraph_general` 从 `N1+N2` 个 nodes 的
  complete graph 中随机取 unordered pairs。它同样是 average-degree，并且会
  混合 `W-W`、`W-X`、`X-X` edge types。
- `tensor_hypergraph.py:generate_tensor_hypergraph` 对每个 mode 用
  `torch.randint` 抽 index，所以 degree 是随机平均意义上的，且可能出现
  duplicate hyperedges。
- `tensor_supergraph.py:create_tensor_supergraph` 也是每个 mode 用
  `torch.randint` 抽 index，因此不是 exact-degree generator。

结论：当前没有 active route 是论文定义的 exact `c=alpha M`
paper-regular graph generator。

## Prior Scale

论文：

```text
Ising x in {-1,+1}
Gaussian x ~ N(0,1)
latent variance = 1
```

当前代码：

- `src/matrix_factorization/core/experiment/data_factory.py:DataFactory.create_teacher`
  通过 `algorithm_params.normalization_profile` 决定 teacher scale。
  `paper_sparse_sampling` 下 Gaussian/Rademacher teacher entry variance 是 `1`；
  `internal_normalized_legacy` 下是 `1/M`。
- `src/matrix_factorization/core/contracts.py:get_normalization_specs` 声明
  默认 `paper_sparse_sampling`，即 `latent_scale="O(1)"`、
  `teacher_init_variance="1"`、`student_init_variance="1"`、
  `prior_precision_base="1"`，并保留 `internal_normalized_legacy`。
- `src/matrix_factorization/modules/algorithms/bigamp/spreading.py`、
  `standard.py`、`agd.py`、`tensor_spreading.py` 和
  `tensor_spreading_parallel.py` 的 active routes 都读取同一个 profile helper。
  paper profile 下 student factors 以 std `1` 初始化，factor variance 上限为
  `1`，AMP prior precision base 为 `1`；legacy profile 下保持旧的
  `1/sqrt(M)`、`1/M` 和 `M`。

结论：默认 profile 已经使用论文的 prior scale；旧结果需要显式解释为
`internal_normalized_legacy`。

## Interaction Scale

论文：

```text
pi_square = lambda / sqrt(M) * sum_mu F_square,mu prod_i x_i,mu
```

当前代码：

- Matrix teacher：
  `DataFactory.create_teacher()` 计算 `Y_teacher = (1/sqrt(M))*W_teacher@X_teacher`。
- Spreading teacher：
  `src/matrix_factorization/modules/algorithms/bigamp/f_gen.py:compute_Y_super`
  计算 `Y = (1/sqrt(M))*sum_mu F W X`。
- Tensor teacher：
  `tensor_step.py:forward_pass_tensor`、
  `tensor_step_batch.py:forward_pass_tensor_batch` 和
  `tensor_step_super.py:forward_pass_tensor_super` 使用 `1/sqrt(M)`。
- 当前被审计的 active forward paths 里没有独立的 `lambda` 参数；等效是
  `lambda=1`。

因此，`paper_sparse_sampling` 下 `1/sqrt(M)` prefactor 和 latent scale 都与论文
一致。以 `p=2` 且 random `F` 为例：

```text
paper: x = O(1)
sum_mu F_mu W_mu X_mu has std O(sqrt(M))
(1/sqrt(M))*sum has std O(1)
```

只有在显式选择 legacy profile 时，才回到旧的不同 normalization：

```text
legacy: W_mu, X_mu = O(1/sqrt(M))
sum_mu F_mu W_mu X_mu has std O(1/sqrt(M))
(1/sqrt(M))*sum has std O(1/M)
```

注意：当前仍没有独立 `lambda` YAML 参数，active forward paths 等效
`lambda=1`。

## F Distribution

论文：

```text
deterministic F=1, or random F with E[F]=0 and E[F^2]=1
```

当前代码：

- `src/matrix_factorization/core/experiment/config.py:SpreadingConfig`
  允许 `"ising"` 或 `"gaussian"`，旧 `"rademacher"` 作为兼容别名。
- `src/matrix_factorization/modules/algorithms/bigamp/f_gen.py:generate_F_ising`
  生成 `{-1,+1}`，在 spreading 路径中以 `int8` 存储。
- `src/matrix_factorization/modules/algorithms/bigamp/f_gen.py:generate_F_gaussian`
  生成 `N(0,1)`。
- `src/matrix_factorization/config.yaml` 当前的 `spreading.f_distribution: 1`
  映射到 Ising。

Ising 满足论文 random zero-mean unit-variance 条件，也符合当前项目偏好。

## Noiseless Output 与 Noise

当前 spreading 和 tensor 路径的 teacher generation 是 noiseless：`Y` 直接由
teacher factors 和 `F` 通过 forward model 算出。

`algorithm_params.noise_var` 会进入 AMP update equations，例如
`src/matrix_factorization/modules/algorithms/bigamp/step.py:bigamp_step_disjoint_union_flat`
和 tensor step functions。它在这次审计的路径里不是作为 sampled additive
noise 注入 teacher observation tensor。

## Metrics

论文的 order parameters 是 latent `m` 和 `q`，并且 MSE 公式依赖
`Var(x)=1` 的 prior scale。

当前项目 formal metrics 是 schema v3 projection diagnostics：

- `Q_Y`：absolute projection，denominator 是 teacher norm squared。
- `Q_W` 和 `Q_X`：coordinate latent projections。
- `Q_N`：tensor latent node/spin/factor projection。
- `Q_W_COS_ROOT`、`Q_X_COS_ROOT`：Cos-root diagnostics。
- `Q_W_SIGN_ALIGNED`、`Q_X_SIGN_ALIGNED`：sign-gauge diagnostics。

这些 metrics 可以用于诊断当前程序，但即使 normalization profile 已经对齐，
它们仍不应直接标注成论文的 `m`、`q`、MSE 或 generalization error；需要另写
order-parameter 映射。

## Sign 与 Scale Gauge

对 spreading matrix observations：

```text
Y_ij = sum_k F_ijk W_ik X_kj
```

random `F` 会破坏一般 rotation/mixing，但仍保留 diagonal gauge：

```text
W_:k -> a_k W_:k
X_k: -> a_k^{-1} X_k:
```

其中 sign gauge `a_k = +/-1` 是特例。现有
`Q_W_SIGN_ALIGNED` 和 `Q_X_SIGN_ALIGNED` 只是 diagnostics。Posthoc scale
alignment 也只是 diagnostic，不应改变 training trajectory。

## Normalization Profile 状态

已新增显式 YAML-facing profile：

```yaml
algorithm_params:
  normalization_profile: paper_sparse_sampling
```

当前实现语义：

```text
latent prior variance = 1
student init variance = 1
prior precision base = 1
interaction scale = lambda / sqrt(M)
F variance = 1
```

旧 convention 保留为：

```text
internal_normalized_legacy:
  latent prior variance = 1/M
  student init variance = 1/M
  prior precision base = M
  interaction scale = 1/sqrt(M)
```

这样旧结果不会失去解释，新结果也能明确声明是否采用 paper convention。尚未
对齐的主要部分是 alpha/edge count 的 route 差异、exact `c=alpha M` graph
generator、以及独立 `lambda` 参数。
