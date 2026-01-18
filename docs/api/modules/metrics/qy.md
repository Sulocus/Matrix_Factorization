# M3: Q_Y 重构质量

测量学生矩阵乘积 Y_student = W @ X 与教师 Y_teacher = W* @ X* 的相似度。

**模块 ID**: M3
**SMF 路径**: `modules/metrics/qy.py`

---

## 🌐 宏观视角

### 系统定位

```
评估指标层
├── M1: gram_cosine.py     ← Gram overlap (cosine)
├── M2: gram_normalized.py ← Gram overlap (normalized)
├── M3: qy.py              ← 本模块（最重要的指标）
├── M4: generalization.py  ← 泛化误差
├── M5: replica.py         ← Replica overlap
├── M6: qy_unobserved.py   ← 未观测位置的 Q_Y
└── M7: aggregators.py     ← 统计聚合
```

### 引入动机

**核心问题**：学生是否成功恢复了教师的矩阵分解？

最直接的答案是比较矩阵乘积：
```
Y_student = W_student @ X_student
Y_teacher = W_teacher @ X_teacher

Q_Y = similarity(Y_student, Y_teacher)
```

### 相对优势

| 指标 | 含义 | 旋转不变 | 范围 |
|------|------|---------|------|
| **Q_Y** | Y 的相似度 | ✅ | [0, 1] |
| Q_W | W 的相似度 | ✅ | [-1, 1] |
| Q_X | X 的相似度 | ✅ | [-1, 1] |
| MSE | 均方误差 | ✅ | [0, ∞) |

**Q_Y 是最重要的指标**，因为：
1. 直接测量最终目标（矩阵乘积的恢复）
2. 对 W 和 X 的旋转/置换不变
3. 相转移点由 Q_Y 急剧上升定义

### 物理图景 🌟

**为什么 Q_Y 对旋转不变？**

矩阵分解存在内在的模糊性：
```
Y = W @ X = (W @ R) @ (R^{-1} @ X) = W' @ X'
```
对于任意可逆矩阵 R，W' 和 X' 产生相同的 Y。

**Gram 矩阵消除模糊性**：
```
Y @ Y^T = W @ X @ X^T @ W^T

如果 Y_student ≈ Y_teacher，
则 Y_s @ Y_s^T ≈ Y_t @ Y_t^T

Q_Y = cosine(flatten(Y_s @ Y_s^T), flatten(Y_t @ Y_t^T))
```

**相转移的含义**：
```
Q_Y
 1 │            ╱
   │           ╱
   │          ╱  ← Q_Y 急剧上升 = 相转移
   │        ╱
 0 │───────╱
   └────────────── α
           αc

α < αc: 信息不足，Q_Y ≈ 0（无法恢复）
α > αc: 信息足够，Q_Y → 1（完美恢复）
```

### 使用场景

**适用**：
- 判断是否成功恢复矩阵分解
- 确定相转移点位置
- 所有实验的主要评估指标

**不适用**：
- 需要单独评估 W 或 X 时（使用 Q_W, Q_X）
- 需要误差的绝对值时（使用 MSE）

---

## 🔬 微观视角

### 代码位置

Q_Y 通常通过 `gram_overlap_cosine(Y_student, Y_teacher, use_left=True)` 计算，
在各程序中直接调用 M1。

| 程序 | 调用方式 | 行号 |
|------|---------|------|
| bigamp/train.py | `gram_overlap_cosine(Y_student, Y_teacher)` | ~700 |
| agd/train_*.py | `gram_overlap_cosine(Y_student, Y_teacher)` | 评估函数中 |

### 数学定义

```
Y_student = W_student @ X_student ∈ ℝ^(N1×N2)
Y_teacher = W_teacher @ X_teacher ∈ ℝ^(N1×N2)

G_s = Y_student @ Y_student^T ∈ ℝ^(N1×N1)
G_t = Y_teacher @ Y_teacher^T ∈ ℝ^(N1×N1)

Q_Y = (vec(G_s) · vec(G_t)) / (||vec(G_s)|| × ||vec(G_t)||)
    = trace(G_s × G_t) / (||G_s||_F × ||G_t||_F)
```

**范围**：Q_Y ∈ [-1, 1]，实际中通常 Q_Y ∈ [0, 1]

### 输入/输出

```python
def compute_qy(W_student, X_student, W_teacher, X_teacher):
    """
    Args:
        W_student: Tensor[N1, M] - 学生左因子
        X_student: Tensor[M, N2] - 学生右因子
        W_teacher: Tensor[N1, M] - 教师左因子
        X_teacher: Tensor[M, N2] - 教师右因子

    Returns:
        Q_Y: float in [0, 1] - 重构质量
    """
```

### 标准实现

```python
@torch.no_grad()
def compute_qy(W_student, X_student, W_teacher, X_teacher):
    """Compute reconstruction quality Q_Y."""
    Y_student = W_student @ X_student
    Y_teacher = W_teacher @ X_teacher
    return gram_overlap_cosine(Y_student, Y_teacher, use_left=True)
```

或直接使用 M1：

```python
Q_Y = gram_overlap_cosine(W @ X, W_teacher @ X_teacher, use_left=True)
```

### 实现细节

1. **数值稳定性**：分母添加 1e-12 防止除零
2. **内存**：需要存储完整的 N1×N2 矩阵乘积
3. **GPU 友好**：所有操作可在 GPU 上执行

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Measure reconstruction quality of Y = W @ X vs Y* = W* @ X*",
    "when_to_use_en": "Primary metric for all experiments, phase transition detection",
    "tags_en": ["Q_Y", "reconstruction", "overlap", "gram", "phase transition", "primary"],

    # 中文
    "purpose_zh": "测量 Y = W @ X 与 Y* = W* @ X* 的重构质量",
    "when_to_use_zh": "所有实验的主要指标，相转移检测",
    "tags_zh": ["Q_Y", "重构", "重叠度", "相转移", "主要指标"],

    # 日文
    "purpose_ja": "Y = W @ X と Y* = W* @ X* の再構成品質を測定",
    "when_to_use_ja": "全ての実験の主要指標、相転移検出",
    "tags_ja": ["Q_Y", "再構成", "オーバーラップ", "相転移"],

    # 技术参数
    "inputs": ["W_student", "X_student", "W_teacher", "X_teacher"],
    "outputs": ["Q_Y: float in [0, 1]"],
    "compute_cost": "O(N1 × M × N2 + N1² × N2)",
    "gpu_friendly": True,
    "rotation_invariant": True,
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/metrics/qy.py`

---

*最后更新：2025年12月*
