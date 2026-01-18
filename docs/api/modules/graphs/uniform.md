# G3: 双正则图生成

使用 Dinic 最大流算法生成度分布均匀的双正则二分图。

**模块 ID**: G3
**SMF 路径**: `modules/graphs/uniform.py`

---

## 🌐 宏观视角

### 系统定位

```
观测掩码生成层
├── G1: random.py      ← 纯随机
├── G2: dinic.py       ← 最大流（本模块依赖）
├── G3: uniform.py     ← 本模块（推荐用于理论验证）
└── G4: low_loop.py    ← 低循环图
```

### 引入动机

**问题**：随机图的度分布不均匀（泊松分布），导致：
1. 有限尺寸效应较大
2. 与理论预测（假设均匀度分布）存在偏差

**解决方案**：生成**双正则图**（bi-regular graph）
- 每个左节点度数相同：`deg_left = α̃ × M`
- 右节点度数尽可能均匀

### 相对优势

| 特性 | G1 随机图 | G3 双正则图 |
|------|----------|------------|
| 生成速度 | ⚡ 快 | 🐢 慢 |
| 左节点度分布 | 泊松分布 | **完全均匀** |
| 右节点度分布 | 泊松分布 | **近似均匀** |
| 有限尺寸效应 | 较大 | 较小 |
| 适用于理论验证 | ❌ | ✅ |

### 物理图景 🌟

**随机图 vs 双正则图**：

```
随机图（度分布不均）：
    Row 0: ○───●───●───○───○   (度=2)
    Row 1: ○───○───●───○───●   (度=2)
    Row 2: ●───●───●───●───○   (度=4) ← 过多！
    Row 3: ○───○───○───●───○   (度=1) ← 过少！

双正则图（度分布均匀）：
    Row 0: ●───●───○───○───○   (度=2)
    Row 1: ○───●───●───○───○   (度=2)
    Row 2: ○───○───●───●───○   (度=2)
    Row 3: ●───○───○───○───●   (度=2)
```

**为什么均匀度分布重要？**

理论分析（如 replica theory）通常假设：
- 每个节点"看到"的局部结构相同
- 度分布均匀时，这个假设更准确
- 减少有限尺寸效应，实验结果更接近 N→∞ 极限

### 使用场景

**适用**：
- 验证理论预测
- 研究有限尺寸效应
- 需要精确相转移点的实验

**不适用**：
- 速度优先的大规模实验
- 快速原型验证

**配置方式**：
```python
USE_BIREGULAR_GRAPH = True  # 启用双正则图
```

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/train.py | `sample_pairs_biregular_exact` | 160-282 |
| bigamp/compare_sizes.py | `sample_pairs_biregular_exact` | 176-... |
| agd/train_sequential.py | `sample_pairs_biregular_exact` | 164-... |
| agd/train_parallel.py | `sample_pairs_biregular_exact` | 171-... |

### 数学定义

给定参数：
- N1: 左节点数（矩阵行数）
- N2: 右节点数（矩阵列数）
- M: 秩
- α̃: 观测密度参数

构建双正则图：
```
deg_left = round(α̃ × M)           # 每个左节点的度数
total_edges = N1 × deg_left       # 总边数
deg_right_base = total_edges // N2  # 右节点基准度数
remainder = total_edges % N2       # 余数
```

右节点度分布：
- `N2 - remainder` 个节点度数为 `deg_right_base`
- `remainder` 个节点度数为 `deg_right_base + 1`

### 输入/输出

```python
def sample_pairs_biregular_exact(N1, N2, M, alpha_tilde_left, device, seed=None):
    """
    Args:
        N1: int - 矩阵行数（左节点数）
        N2: int - 矩阵列数（右节点数）
        M: int - 秩（隐维度）
        alpha_tilde_left: float - 观测密度参数 α̃
        device: torch.device - 目标设备
        seed: int, optional - 随机种子

    Returns:
        i_idx: Tensor[C] - 行索引
        j_idx: Tensor[C] - 列索引
        C: int - 总边数
    """
```

### 标准实现（核心部分）

```python
def sample_pairs_biregular_exact(N1, N2, M, alpha_tilde_left, device, seed=None):
    deg_left = int(round(alpha_tilde_left * M))
    deg_left = max(0, min(deg_left, N2))
    total_edges = N1 * deg_left

    if not USE_BIREGULAR_GRAPH:
        return sample_pairs_random_gpu(N1, N2, total_edges, device, seed)

    # 边界检查
    if deg_left == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    # 计算右节点目标度数
    base = total_edges // N2
    rem = total_edges % N2
    right_target = np.full(N2, base, dtype=int)
    if rem > 0:
        idx = np.arange(N2)
        rng.shuffle(idx)
        right_target[idx[:rem]] += 1

    # 构建流网络并求解
    S, L_off, R_off = 0, 1, 1 + N1
    T = R_off + N2
    din = Dinic(T + 1)

    # S → 左节点，容量 = deg_left
    for i in range(N1):
        din.add_edge(S, L_off + i, deg_left)

    # 左节点 → 右节点，容量 = 1（随机顺序）
    all_pairs = list(itertools.product(range(N1), range(N2)))
    rng.shuffle(all_pairs)
    for i, j in all_pairs:
        din.add_edge(L_off + i, R_off + j, 1)

    # 右节点 → T，容量 = right_target[j]
    for j in range(N2):
        din.add_edge(R_off + j, T, int(right_target[j]))

    # 求最大流
    f = din.max_flow(S, T)
    if f != total_edges:
        raise RuntimeError(f"maxflow only got {f}/{total_edges}")

    # 从流网络中提取边
    i_list, j_list = [], []
    for i in range(N1):
        u = L_off + i
        for v, cap, rev in din.g[u]:
            if R_off <= v < R_off + N2:
                if din.g[v][rev][1] > 0:  # 反向边有流量 = 正向边被使用
                    j = v - R_off
                    i_list.append(i)
                    j_list.append(j)

    return (torch.tensor(i_list, dtype=torch.long, device=device),
            torch.tensor(j_list, dtype=torch.long, device=device),
            len(i_list))
```

### 实现细节

1. **随机打乱边顺序**：避免确定性模式
2. **度数约束**：`deg_left ≤ N2`，否则不可行
3. **余数处理**：随机选择部分右节点增加一度
4. **可行性检查**：最大流必须等于 `total_edges`

### 复杂度

- 时间：O(N1 × N2 × √(N1 + N2))（Dinic 在二分图上的复杂度）
- 空间：O(N1 × N2)（存储所有可能的边）

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Generate bi-regular bipartite graph with uniform degree distribution",
    "when_to_use_en": "Theory verification, finite-size effect study, precise phase transition",
    "limitations_en": "Slower than random graph, O(N1×N2) space complexity",
    "tags_en": ["biregular", "uniform", "degree", "graph", "Dinic", "max flow", "theory"],

    # 中文
    "purpose_zh": "生成度分布均匀的双正则二分图",
    "when_to_use_zh": "理论验证、有限尺寸效应研究、精确相转移实验",
    "limitations_zh": "比随机图慢，空间复杂度 O(N1×N2)",
    "tags_zh": ["双正则", "均匀", "度分布", "图", "Dinic", "最大流", "理论"],

    # 日文
    "purpose_ja": "一様な次数分布を持つ双正則二部グラフを生成",
    "when_to_use_ja": "理論検証、有限サイズ効果研究、精密相転移実験",
    "tags_ja": ["双正則", "一様", "次数分布", "グラフ", "最大流"],

    # 技术参数
    "inputs": ["N1: int", "N2: int", "M: int", "alpha_tilde_left: float", "device", "seed?"],
    "outputs": ["i_idx: Tensor[C]", "j_idx: Tensor[C]", "C: int"],
    "compute_cost": "O(N1 × N2 × √(N1+N2))",
    "gpu_friendly": False,  # Dinic 在 CPU 上运行
    "config_option": "USE_BIREGULAR_GRAPH",
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/graphs/uniform.py`

---

*最后更新：2025年12月*
