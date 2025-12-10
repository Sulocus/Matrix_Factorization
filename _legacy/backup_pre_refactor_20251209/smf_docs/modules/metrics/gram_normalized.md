# M2: Gram Overlap Normalized

归一化的 Gram 重叠度，输出范围 [0, 1]

**模块 ID**: M2
**SMF 路径**: `sparse_matrix_factorization/modules/metrics/gram_normalized.py`

---

## 目的

计算带基线校正的归一化 Gram 重叠度。使得随机初始化时 Q' ≈ 0，完美匹配时 Q' = 1。

---

## 数学定义

1. 先计算原始余弦相似度:
```
q = gram_overlap_cosine(A, B, use_left)
```

2. 计算随机矩阵的期望余弦值（基线）:
```
b = m / (m + n + 1)
```
其中 n 是行数，m 是列数（对于 left Gram）

3. 基线校正:
```
q' = (q - b) / (1 - b)
```

4. 裁剪到 [0, 1]:
```
return clamp(q', 0, 1)
```

---

## 代码位置

| 程序 | 行号 | 函数名 | 状态 |
|------|------|--------|------|
| bigamp/train.py | 156-172 | gram_overlap_zero_to_one | ✅ 一致 |
| bigamp/compare_sizes.py | 369-381 | gram_overlap_zero_to_one | ✅ 一致 |
| bigamp/orthogonal_teacher.py | 345-355 | gram_overlap_zero_to_one | ✅ 一致 |

---

## 标准实现

```python
@torch.no_grad()
def gram_overlap_zero_to_one(A, B, use_left=True):
    """Normalized Gram overlap in [0, 1] range with baseline correction.

    Uses baseline b = m/(m+n+1) which is the expected cosine for random matrices.
    This ensures random initialization gives Q' ≈ 0, and perfect match gives Q' = 1.

    Args:
        A: Student matrix
        B: Teacher matrix
        use_left: If True, compute for W (left Gram); else for X (right Gram)

    Returns:
        Normalized overlap in [0, 1]
    """
    q = gram_overlap_cosine(A, B, use_left)
    if use_left:
        n, m = A.shape
    else:
        n, m = A.shape[1], A.shape[0]
    b = m / (m + n + 1)  # baseline: expected cosine for random matrices
    qc = (q - b) / (1.0 - b + 1e-12)  # baseline correction
    return float(max(0.0, min(1.0, qc)))
```

---

## 输入/输出

**输入**:
- `A`: torch.Tensor, 学生矩阵
- `B`: torch.Tensor, 教师矩阵
- `use_left`: bool, 默认 True

**输出**:
- `float`: 归一化重叠度，范围 [0, 1]

---

## 与 M1 的关系

| 指标 | M1 (gram_cosine) | M2 (gram_normalized) |
|------|------------------|----------------------|
| 输出范围 | [-1, 1] | [0, 1] |
| 随机基线 | m/(m+n+1) | 0 |
| 完美匹配 | 1 | 1 |
| 用途 | 原始值 | 归一化显示 |

---

## 使用场景

```python
# 计算归一化的 W overlap (Q_W')
Q_W_prime = gram_overlap_zero_to_one(W_student, W_teacher, use_left=True)

# 计算归一化的 X overlap (Q_X')
Q_X_prime = gram_overlap_zero_to_one(X_student, X_teacher, use_left=False)
```

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Normalized Gram overlap in [0, 1] range with baseline correction",
    "when_to_use_en": "When you need overlap metric normalized to [0, 1] for visualization",
    "limitations_en": "Assumes Gaussian random matrix baseline",
    "inputs": ["A: (N1, M) or (M, N2)", "B: same shape as A", "use_left: bool"],
    "outputs": ["overlap: float in [0, 1]"],
    "formula": "(q - baseline) / (1 - baseline), baseline = m/(m+n+1)",
    "compute_cost": "O(N^2 * M) + O(1) for baseline correction",
    "related": ["gram_cosine (M1)"],
    "tags_en": ["overlap", "gram", "normalized", "baseline", "Q_W_prime", "Q_X_prime", "zero-to-one"],

    # 中文
    "purpose_zh": "归一化 Gram 重叠度，输出范围 [0, 1]，带基线校正",
    "when_to_use_zh": "需要 [0, 1] 范围的归一化重叠度指标用于可视化",
    "limitations_zh": "假设高斯随机矩阵基线",
    "tags_zh": ["重叠度", "归一化", "基线校正", "Q_W'", "Q_X'", "零到一"],

    # 日文
    "purpose_ja": "ベースライン補正付きの正規化Gramオーバーラップ [0, 1]",
    "when_to_use_ja": "[0, 1]範囲の正規化オーバーラップが必要な場合（可視化用）",
    "limitations_ja": "ガウスランダム行列のベースラインを仮定",
    "tags_ja": ["オーバーラップ", "正規化", "ベースライン補正", "Q_W'", "Q_X'"],
}
```

---

*最后更新：2024年12月*
