# Graphical Tensor Sparse Sampling Convention

审计日期：2026-04-28。

这份笔记只记录目标论文 `Graphical model for tensor factorization by
sparse sampling` 的 convention。不要把
`docs/theory/tensor_gamp/1.md` 当成本文献来源；那个文件对应另一篇
TeG-AMP 论文。

## 文献来源

- 本地论文路径：`Tensor Factorization.pdf`
- 标题：`Graphical model for tensor factorization by sparse sampling`
- arXiv：`2510.17886`
- 作者：Angelo Giorgio Cavaliere, Riki Nagasawa, Shuta Yokoi, Tomoyuki
  Obuchi, Hajime Yoshino

## 变量与图

论文要估计的是 `N` 个 `M` 维向量：

```text
x_i = (x_i1, ..., x_iM)^T in R^M,  i = 1, ..., N.
```

每个 observation 连接一个 `p`-plet edge。被采样的 edge set 记为 `E`。
对某条 edge `square`，`partial square` 表示它连接的 `p` 个 variable
nodes。

论文的 sparse sampling graph 不是只固定总边数的 Erdos-Renyi 平均度图，
而是要求每个 variable node 的 degree 精确等于：

```text
c = alpha M
```

因此 observed `p`-plets 的数量是：

```text
|E| = N_square = N c / p = (alpha / p) N M.
```

论文还定义：

```text
gamma = alpha / p.
```

所以当论文图或公式使用 `gamma` 时，代码里的参数如果叫 `alpha`，需要先
确认它到底对应论文的 `alpha`，还是对应论文的 `gamma`。

## Noiseless Observation

对一条 edge `square`，论文的 noiseless observation 是：

```text
pi_square =
  lambda / sqrt(M) *
  sum_{mu=1}^M F_{square,mu} prod_{i in partial square} x_{i,mu}.
```

在 noiseless additive-output 情况下：

```text
y_square = pi_square.
```

这里 `lambda` 是 signal-strength prefactor。论文的 `1/sqrt(M)` 放在
interaction/output 里，而不是把 latent prior 本身缩成 `1/sqrt(M)`。

## Prior Scale

论文的技术假设要求 prior 零均值。文中使用的两个具体 prior 是：

```text
Ising:    P_pri(x) = 1/2 delta(x - 1) + 1/2 delta(x + 1)
Gaussian: P_pri(x) = N(0, 1)
```

因此 latent variable 是 `O(1)` convention，方差是 `1`。这不是
`O(1/sqrt(M))` 的 latent convention。

## Spreading Factor F

论文允许两类 `F`：

```text
deterministic: F_{square,mu} = 1
random:        F_{square,mu} iid, E[F] = 0, E[F^2] = 1
```

Rademacher `F in {-1, +1}` 满足 random spreading 的零均值、单位方差条件。
当前项目偏好是：除非 run 明确请求 Gaussian `F`，否则 random spreading
默认保留 Rademacher。

## Output Models

论文讨论一般的 `P_out(y | pi)`，包括 additive noise 和 sign output。本仓库
这次 convention 审计先聚焦 noiseless additive case `y = pi`，因为当前被
审计的 teacher generation 路径直接从 noiseless forward model 生成观测，而
`algorithm_params.noise_var` 主要作为 AMP likelihood/noise 参数进入更新式。

## Order Parameters 与项目 Metrics

论文的主要 latent order parameters 是：

```text
m = E_y[ (1/(N M)) sum_i sum_mu x^*_{i,mu} <x_{i,mu}> ]
q = E_y[ (1/(N M)) sum_i sum_mu <x_{i,mu}>^2 ]
```

Bayes-optimal 情况下有 `m = q`。论文还在 `E[(x^*)^2] = 1` 的 prior
scale 下把它们和 input/output MSE 联系起来。

项目当前 formal metrics 不是这些 order parameters 的直接替代。当前已确认
的项目 metrics 是 projection diagnostics：

```text
Q_Y = abs(<Y_student, Y_teacher>) / <Y_teacher, Y_teacher>
Q_W, Q_X = coordinate latent projections
Q_N = tensor latent node/spin/factor projection
```

项目还保留 Gram-root 和 sign-aligned diagnostics。旧 cosine `Q_Y`、`MSE`
和 `Gen_Error` 不再作为新结果的 formal metrics。

## Dense Limit

论文的 dense limit 是：

```text
N, M -> infinity, N >> M, alpha = O(1), c = alpha M.
```

也就是说，在 `O(N^p)` 个可能 tensor components 中，只观察 `O(N M)` 个。
这个 degree-constrained sparse graph convention 是论文定义的一部分。做论文
复现时，不能静默替换成只固定总边数的 average-degree graph。
