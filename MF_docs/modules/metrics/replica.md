# M5: Replica Overlap

测量多个独立求解结果（replica）之间的重叠度，用于验证解的唯一性。

**模块 ID**: M5
**SMF 路径**: `modules/metrics/replica.py`

---

## 🌐 宏观视角

### 系统定位

```
评估指标层
├── M1-M4: 学生-教师比较
├── M5: replica.py  ← 本模块（replica 间比较）
├── M6: qy_unobserved.py
└── M7: aggregators.py
```

### 引入动机

**问题**：优化找到的解是唯一的吗？还是存在多个等价的局部最优？

**验证方法**：从不同随机初始化运行算法多次，比较结果：
```
Run 1: (W₁, X₁) ─┐
Run 2: (W₂, X₂) ─┼─→ 计算两两 overlap
Run 3: (W₃, X₃) ─┘
```

### 相对优势

| 指标 | 比较对象 | 用途 |
|------|---------|------|
| Q_Y | 学生 vs 教师 | 恢复质量 |
| **Q_inter** | replica vs replica | 解的唯一性 |
| Q_self | replica vs 教师 | 各 replica 的质量 |

### 物理图景 🌟

**能量景观与 replica overlap**：

```
唯一解（单峰）:              多个解（多峰）:
    E                            E
    │    ╲   ╱                   │  ╲  ╱   ╲  ╱
    │     ╲ ╱                    │   ╲╱     ╲╱
    │      ●                     │   ● ● ●
    └─────────                   └─────────
        W                            W

所有 replica → 同一解           replica → 不同解
Q_inter ≈ 1                    Q_inter < Q_self
```

**replica theory 预测**：
- α < αc: 多个等价解（旋转对称性未破缺）
- α > αc: 唯一解（旋转对称性破缺）

**验证方式**：
```
Q_inter / Q_self ≈ 1: 所有 replica 收敛到同一解
Q_inter / Q_self < 1: 存在多个不同的解
```

### 使用场景

**适用**：
- 验证解的唯一性
- 研究能量景观结构
- 验证 replica symmetry breaking

**不适用**：
- 只关心恢复质量时
- 计算资源有限时（需要多次运行）

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/replica_overlap.py | `compute_replica_overlaps` | 主函数 |
| bigamp/train.py | `compute_replica_overlap` | ~492-520 |

### 数学定义

给定 K 个 replica: {(W₁, X₁), ..., (W_K, X_K)}

**Self overlap**（与教师）：
```
Q_self[k] = gram_overlap(W_k, W_teacher)
```

**Inter overlap**（replica 间）：
```
Q_inter[k,l] = gram_overlap(W_k, W_l), k ≠ l
```

**聚合指标**：
```
Q_self_mean = (1/K) × Σ_k Q_self[k]
Q_inter_mean = (2/(K×(K-1))) × Σ_{k<l} Q_inter[k,l]
```

### 输入/输出

```python
def compute_replica_overlaps(replicas, W_teacher, X_teacher):
    """
    Args:
        replicas: List[(W_k, X_k)] - K 个 replica 的解
        W_teacher: Tensor[N1, M] - 教师左因子
        X_teacher: Tensor[M, N2] - 教师右因子

    Returns:
        Q_self_W: List[float] - 每个 replica 的 Q_W
        Q_self_X: List[float] - 每个 replica 的 Q_X
        Q_inter_W: float - replica 间平均 Q_W
        Q_inter_X: float - replica 间平均 Q_X
    """
```

### 标准实现

```python
@torch.no_grad()
def compute_replica_overlaps(replicas, W_teacher, X_teacher):
    """Compute self and inter-replica overlaps."""
    K = len(replicas)

    # Self overlaps (each replica vs teacher)
    Q_self_W = []
    Q_self_X = []
    for W_k, X_k in replicas:
        Q_self_W.append(gram_overlap_cosine(W_k, W_teacher, use_left=True))
        Q_self_X.append(gram_overlap_cosine(X_k, X_teacher, use_left=False))

    # Inter overlaps (replica vs replica)
    Q_inter_W_list = []
    Q_inter_X_list = []
    for i in range(K):
        for j in range(i + 1, K):
            W_i, X_i = replicas[i]
            W_j, X_j = replicas[j]
            Q_inter_W_list.append(gram_overlap_cosine(W_i, W_j, use_left=True))
            Q_inter_X_list.append(gram_overlap_cosine(X_i, X_j, use_left=False))

    Q_inter_W = sum(Q_inter_W_list) / len(Q_inter_W_list) if Q_inter_W_list else 0
    Q_inter_X = sum(Q_inter_X_list) / len(Q_inter_X_list) if Q_inter_X_list else 0

    return Q_self_W, Q_self_X, Q_inter_W, Q_inter_X
```

### 实现细节

1. **复杂度**：O(K² × N × M²) 用于计算所有 pair
2. **典型 K 值**：3-10 个 replica
3. **独立性**：每个 replica 需要不同的随机种子

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Measure overlap between multiple independent solutions (replicas)",
    "when_to_use_en": "Verify solution uniqueness, study energy landscape, replica symmetry",
    "tags_en": ["replica", "overlap", "uniqueness", "multimodal", "energy landscape", "RSB"],

    # 中文
    "purpose_zh": "测量多个独立解（replica）之间的重叠度",
    "when_to_use_zh": "验证解的唯一性，研究能量景观，replica 对称性",
    "tags_zh": ["副本", "重叠度", "唯一性", "多峰", "能量景观", "RSB"],

    # 日文
    "purpose_ja": "複数の独立解（レプリカ）間のオーバーラップを測定",
    "when_to_use_ja": "解の唯一性検証、エネルギー地形研究、レプリカ対称性",
    "tags_ja": ["レプリカ", "オーバーラップ", "唯一性", "多峰性"],

    # 技术参数
    "inputs": ["replicas: List[(W, X)]", "W_teacher", "X_teacher"],
    "outputs": ["Q_self_W", "Q_self_X", "Q_inter_W", "Q_inter_X"],
    "compute_cost": "O(K² × N × M²)",
    "gpu_friendly": True,
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/metrics/replica.py`

---

*最后更新：2025年12月*
