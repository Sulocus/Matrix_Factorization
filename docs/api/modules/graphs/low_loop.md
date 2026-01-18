# G4: 低循环图生成

通过 MCMC 边交换最小化短循环（4-loop, 6-loop 等）的图生成方法。

**模块 ID**: G4
**SMF 路径**: `modules/graphs/low_loop.py`

---

## 🌐 宏观视角

### 系统定位

```
观测掩码生成层
├── G1: random.py      ← 纯随机
├── G2: dinic.py       ← 最大流
├── G3: uniform.py     ← 双正则图
└── G4: low_loop.py    ← 本模块（AMP 理论最精确）
```

### 引入动机

**问题**：AMP (Approximate Message Passing) 算法假设因子图是**树状**结构。实际的随机图存在大量**短循环**：

```
4-loop (C4):     6-loop (C6):
  i ─── j          i ─── j
  │  ×  │          │     │
  i'─── j'         i'─── j'─── j''
                   │           │
                   i''─────────┘
```

短循环越多，AMP 的均场近似越不准确，导致：
- 相转移点偏移
- 收敛性变差
- 与理论预测不符

### 相对优势

| 方法 | 4-loop 数量 | AMP 精度 | 速度 |
|------|------------|---------|------|
| G1 随机图 | O(C²/N) 很多 | 较差 | ⚡ 最快 |
| G3 双正则图 | O(C²/N) 很多 | 较差 | 🐢 慢 |
| **G4 低循环图** | **接近 0** | **最好** | 🐌 最慢 |

### 物理图景 🌟

**为什么短循环影响 AMP？**

AMP 通过消息传递更新每个节点的"信念"：
```
消息 i→j: 基于 i 的所有邻居（除了 j）的信息
消息 j→i: 基于 j 的所有邻居（除了 i）的信息
```

**树结构**：消息独立，无循环依赖
```
    ○ → ○ → ○
    ↑       ↓
    ○ ← ○ ← ○
    (消息不会"绕圈"回来)
```

**有 4-loop**：消息会在 2 步内"回来"
```
    i ──→ j
    ↑  ↙  ↓
    i' ←─ j'

    i 发给 j 的消息，经过 j→j'→i'→i 回到 i
    导致信息重复计算，估计偏差
```

**MCMC 边交换**：通过随机交换边来消除短循环
```
原始:     交换后:
i ─ j     i   j
    ×   →   ×
i'─ j'    i'  j'

(i,j) + (i',j') → (i,j') + (i',j)
```

### Kővári–Sós–Turán 定理

对于 N1×N2 的二分图，完全消除 4-loop 的最大边数为：
```
C_max ≈ (1/2) × N2 × √(N1) + N1/2

当 N1 = N2 = N 时: C_max ≈ (1/2) × N^1.5
```

这意味着：
- **α < 0.35** 时可以完全消除 4-loop
- **α > 0.35** 时只能减少 20-30%

### 使用场景

**适用**：
- 验证 AMP 理论预测
- 研究图结构对收敛性的影响
- 低密度区域（α < 0.35）的精确实验

**不适用**：
- 高密度区域（α > 0.8，效果有限）
- 速度优先的大规模实验

**配置方式**：
```python
FORBID_4_CYCLES = True     # 启用 4-loop 最小化
MCMC_SWEEPS = 20           # MCMC 扫描次数
LOOP_ORDER = 2             # k=2 (4-loop), k=3 (6-loop), k=4 (8-loop)
```

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/low_loop_graph.py | `sample_pairs_no_c4` | 586-656 |
| bigamp/low_loop_graph.py | `mcmc_minimize_2k_loops_gpu` | 353-467 |
| bigamp/low_loop_graph.py | `count_2k_loops_gpu` | 290-351 |

### 数学定义

**4-loop 计数**（使用邻接矩阵）：
```
A: N1×N2 邻接矩阵
B = A @ A^T: N1×N1 共同邻居矩阵
B[i,i'] = i 和 i' 的共同右邻居数

4-loop 数量 = (1/4) × Σᵢⱼ A[i,j] × (B² @ A)[i,j]
            = (1/4) × trace(A × (B² @ A)^T)
```

**2k-loop 计数**（推广）：
```
2k-loop 数量 ∝ trace(A × (B^(k-1) @ A)^T)
```

### 输入/输出

```python
def sample_pairs_no_c4(N1, N2, C, device, seed=None,
                       lambda_penalty=None, n_sweeps=None, loop_order=None):
    """
    Args:
        N1: int - 矩阵行数
        N2: int - 矩阵列数
        C: int - 边数
        device: torch.device - 目标设备
        seed: int, optional - 随机种子
        lambda_penalty: float - MCMC 惩罚系数（默认 5.0）
        n_sweeps: int - MCMC 扫描次数（默认 20）
        loop_order: int - 循环阶数 k（默认 2，即 4-loop）

    Returns:
        i_idx: Tensor[C] - 行索引
        j_idx: Tensor[C] - 列索引
        C: int - 边数
    """
```

### 标准实现（核心部分）

```python
def sample_pairs_no_c4(N1, N2, C, device, seed=None, lambda_penalty=None,
                       n_sweeps=None, loop_order=None):
    """Generate graph with minimized 2k-loops using MCMC edge-switching."""

    # 默认参数
    if lambda_penalty is None:
        lambda_penalty = MCMC_LAMBDA
    if n_sweeps is None:
        n_sweeps = MCMC_SWEEPS
    if loop_order is None:
        loop_order = LOOP_ORDER

    # 1. 生成初始随机图
    if seed is not None:
        torch.manual_seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"C={C} > N1*N2={total}")

    idx = torch.randperm(total, device=device)[:C]
    i_idx = idx // N2
    j_idx = idx % N2
    edges = list(zip(i_idx.cpu().tolist(), j_idx.cpu().tolist()))

    # 2. 运行 MCMC 最小化循环
    edges, accept_rate, n_initial, n_final = mcmc_minimize_2k_loops_gpu(
        edges, N1, N2, device, k=loop_order,
        lambda_penalty=lambda_penalty, n_sweeps=n_sweeps, seed=seed
    )

    # 3. 转换为张量输出
    i_idx = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    j_idx = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)

    return i_idx, j_idx, C
```

### MCMC 边交换算法

```python
def mcmc_minimize_2k_loops_gpu(edges, N1, N2, device, k=2,
                                lambda_penalty=5.0, n_sweeps=5, seed=None):
    """
    通过边交换最小化 2k-loop

    算法:
    1. 构建邻接矩阵 A
    2. 计算 score_matrix = B^(k-1) @ A，其中 B = A @ A^T
    3. score_matrix[i,j] 表示边 (i,j) 参与的 2k-loop 数量
    4. 优先交换高分边与低分边
    5. 只接受减少循环数的交换
    """

    for sweep in range(n_sweeps):
        # 计算每条边的循环分数
        B = A.float() @ A.T.float()
        if k == 2:
            score_matrix = B @ A.float()
        elif k == 3:
            score_matrix = B @ B @ A.float()
        # ...

        edge_scores = score_matrix[edges_i, edges_j]
        sorted_idx = edge_scores.argsort(descending=True)

        # 尝试交换高分边和低分边
        for kk in range(n_top):
            idx1 = sorted_idx[kk]           # 高分边
            idx2 = sorted_idx[C//2 + rand]  # 低分边

            i1, j1 = edges_i[idx1], edges_j[idx1]
            i2, j2 = edges_i[idx2], edges_j[idx2]

            # 检查交换有效性
            if i1 == i2 or j1 == j2: continue
            if A[i1, j2] > 0 or A[i2, j1] > 0: continue

            # 尝试交换
            old_count = count_2k_loops_gpu(A, k)
            A[i1, j1], A[i2, j2] = 0, 0
            A[i1, j2], A[i2, j1] = 1, 1
            new_count = count_2k_loops_gpu(A, k)

            if new_count < old_count:
                # 接受交换
                edges_i[idx1], edges_j[idx1] = i1, j2
                edges_i[idx2], edges_j[idx2] = i2, j1
            else:
                # 撤销交换
                A[i1, j2], A[i2, j1] = 0, 0
                A[i1, j1], A[i2, j2] = 1, 1
```

### 实现细节

1. **GPU 加速**：矩阵运算在 GPU 上执行
2. **贪婪策略**：只接受减少循环数的交换（非 Metropolis）
3. **智能选择**：优先交换高循环分数的边
4. **度数保持**：边交换保持所有节点的度数不变

### 复杂度

- 时间：O(n_sweeps × C × (N1² + N2²))
- 空间：O(N1×N2)
- 对于大矩阵，这是最慢的方法

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Generate graphs with minimized short loops (4-loop, 6-loop) via MCMC edge-switching",
    "when_to_use_en": "AMP theory verification, graph structure studies, low-density experiments",
    "limitations_en": "Slow, limited effect at high density (α > 0.35), C4-free impossible above KST bound",
    "tags_en": ["low loop", "C4-free", "MCMC", "edge switching", "girth", "AMP", "tree-like"],

    # 中文
    "purpose_zh": "通过 MCMC 边交换生成短循环最小化的图",
    "when_to_use_zh": "AMP 理论验证、图结构研究、低密度实验",
    "limitations_zh": "较慢，高密度时效果有限，α > 0.35 时无法完全消除 4-loop",
    "tags_zh": ["低循环", "C4-free", "MCMC", "边交换", "围长", "AMP"],

    # 日文
    "purpose_ja": "MCMC辺交換により短ループを最小化したグラフを生成",
    "when_to_use_ja": "AMP理論検証、グラフ構造研究、低密度実験",
    "tags_ja": ["低ループ", "C4-free", "MCMC", "辺交換", "AMP"],

    # 技术参数
    "inputs": ["N1", "N2", "C", "device", "seed?", "lambda_penalty?", "n_sweeps?", "loop_order?"],
    "outputs": ["i_idx: Tensor[C]", "j_idx: Tensor[C]", "C: int"],
    "compute_cost": "O(n_sweeps × C × N²)",
    "gpu_friendly": True,
    "config_options": ["FORBID_4_CYCLES", "MCMC_SWEEPS", "LOOP_ORDER"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/graphs/low_loop.py`

---

*最后更新：2025年12月*
