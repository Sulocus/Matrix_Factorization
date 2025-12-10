# M6: Q_Y (Unobserved)

仅在未观测位置计算的 Q_Y，用于评估泛化的相似度。

**模块 ID**: M6
**SMF 路径**: `modules/metrics/qy_unobserved.py`

---

## 🌐 宏观视角

### 系统定位

```
评估指标层
├── M3: qy.py              ← 全矩阵 Q_Y
├── M4: generalization.py  ← 未观测 MSE
├── M6: qy_unobserved.py   ← 本模块（未观测 Q_Y）
```

### 引入动机

M3 的 Q_Y 计算整个矩阵的相似度，包括观测和未观测位置。

M6 仅在**未观测位置**计算 Q_Y：
- 更直接反映泛化能力
- 排除训练数据的影响

### 物理图景 🌟

```
         观测位置              未观测位置
    ┌─────────────┐       ┌─────────────┐
    │ 训练数据    │       │ 测试数据    │
    │ (可能过拟合)│       │ (真实泛化)  │
    └─────────────┘       └─────────────┘
         ↓                     ↓
    Q_Y (full)            Q_Y (unobserved)
```

### 使用场景

**适用**：
- 严格评估泛化能力
- 与 M4 (MSE) 交叉验证

**不适用**：
- 相转移检测（用 M3 全矩阵 Q_Y）

---

## 🔬 微观视角

### 数学定义

```
Ω: 观测位置
Ω̄: 未观测位置

Y_s_masked = Y_student ⊙ (1 - mask)
Y_t_masked = Y_teacher ⊙ (1 - mask)

Q_Y_unobs = gram_overlap(Y_s_masked, Y_t_masked)
```

### 标准实现

```python
@torch.no_grad()
def compute_qy_unobserved(Y_student, Y_teacher, mask):
    """Compute Q_Y only on unobserved entries."""
    unobserved_mask = 1.0 - mask

    Y_s_masked = Y_student * unobserved_mask
    Y_t_masked = Y_teacher * unobserved_mask

    return gram_overlap_cosine(Y_s_masked, Y_t_masked, use_left=True)
```

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "Compute Q_Y only on unobserved entries for generalization evaluation",
    "when_to_use_en": "Strict generalization assessment, cross-validate with MSE",
    "tags_en": ["Q_Y", "unobserved", "generalization", "masked"],
    "tags_zh": ["Q_Y", "未观测", "泛化", "掩码"],
}
```

---

*最后更新：2025年12月*
