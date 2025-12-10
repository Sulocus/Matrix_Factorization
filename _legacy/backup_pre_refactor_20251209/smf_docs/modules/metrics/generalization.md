# M4: 泛化误差

计算学生模型在未观测位置的预测误差（MSE）。

**模块 ID**: M4
**SMF 路径**: `modules/metrics/generalization.py`

---

## 🌐 宏观视角

### 系统定位

```
评估指标层
├── M1-M2: Gram overlap
├── M3: qy.py              ← 观测+未观测的 Q_Y
├── M4: generalization.py  ← 本模块（未观测位置的 MSE）
├── M5: replica.py
├── M6: qy_unobserved.py   ← 未观测位置的 Q_Y
└── M7: aggregators.py
```

### 引入动机

训练时学生只看到部分观测：
```
Y_observed[i,j] = Y_teacher[i,j]  for (i,j) ∈ Ω
```

**关键问题**：学生在**未观测位置** (i,j) ∉ Ω 的表现如何？

泛化误差测量学生对未见数据的预测能力。

### 相对优势

| 指标 | 测量内容 | 用途 |
|------|---------|------|
| Training Loss | 观测位置误差 | 优化目标 |
| **Gen_Error** | 未观测位置误差 | 真实性能 |
| Q_Y | 整体相似度 | 相转移检测 |

### 物理图景 🌟

```
观测掩码:           训练/泛化分离:
┌───────────┐      ┌───────────┐
│ ○ ● ○ ● ○ │      │   T   G   │
│ ● ○ ● ○ ● │      │ T   G   T │
│ ○ ● ○ ○ ● │      │   T   G   │
│ ● ○ ○ ● ○ │      │ G   T   G │
└───────────┘      └───────────┘
● = 观测位置         T = Training loss
○ = 未观测位置       G = Generalization error

泛化误差 = (1/|Ω̄|) × Σ_{(i,j)∉Ω} (Y_student[i,j] - Y_teacher[i,j])²
```

**相转移时的行为**：
```
Gen_Error
   │
   │\ ← α < αc: 高泛化误差（无法推断未观测）
   │ \
   │  \
   │   \____ ← α > αc: 低泛化误差（成功泛化）
   │
   └────────── α
          αc
```

### 使用场景

**适用**：
- 评估模型的泛化能力
- 与 Q_Y 交叉验证
- 机器学习视角的分析

**不适用**：
- 只关心相转移点时（Q_Y 更直接）

---

## 🔬 微观视角

### 代码位置

泛化误差在评估函数中计算，通常与其他指标一起。

### 数学定义

```
Ω: 观测位置集合
Ω̄ = {1..N1} × {1..N2} \ Ω: 未观测位置集合

Gen_Error = (1/|Ω̄|) × Σ_{(i,j)∈Ω̄} (Y_student[i,j] - Y_teacher[i,j])²
```

### 输入/输出

```python
def compute_generalization_error(Y_student, Y_teacher, mask):
    """
    Args:
        Y_student: Tensor[N1, N2] - 学生预测
        Y_teacher: Tensor[N1, N2] - 教师真值
        mask: Tensor[N1, N2] - 观测掩码 (1=观测, 0=未观测)

    Returns:
        gen_error: float - 未观测位置的 MSE
    """
```

### 标准实现

```python
@torch.no_grad()
def compute_generalization_error(Y_student, Y_teacher, mask):
    """Compute generalization error on unobserved entries."""
    unobserved_mask = 1.0 - mask
    n_unobserved = unobserved_mask.sum()

    if n_unobserved == 0:
        return 0.0

    diff_sq = ((Y_student - Y_teacher) ** 2) * unobserved_mask
    gen_error = diff_sq.sum() / n_unobserved

    return float(gen_error)
```

### 实现细节

1. **掩码处理**：用 `1 - mask` 选择未观测位置
2. **归一化**：除以未观测位置数量
3. **边界情况**：全观测时返回 0

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Compute MSE on unobserved matrix entries",
    "when_to_use_en": "Evaluate generalization ability, cross-validate with Q_Y",
    "tags_en": ["generalization", "MSE", "unobserved", "prediction", "error"],

    # 中文
    "purpose_zh": "计算未观测矩阵元素的均方误差",
    "when_to_use_zh": "评估泛化能力，与 Q_Y 交叉验证",
    "tags_zh": ["泛化", "MSE", "未观测", "预测", "误差"],

    # 日文
    "purpose_ja": "未観測行列要素のMSEを計算",
    "when_to_use_ja": "汎化能力の評価、Q_Yとのクロス検証",
    "tags_ja": ["汎化", "MSE", "未観測", "予測", "誤差"],

    # 技术参数
    "inputs": ["Y_student: Tensor[N1, N2]", "Y_teacher: Tensor[N1, N2]", "mask: Tensor[N1, N2]"],
    "outputs": ["gen_error: float >= 0"],
    "compute_cost": "O(N1 × N2)",
    "gpu_friendly": True,
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/metrics/generalization.py`

---

*最后更新：2025年12月*
