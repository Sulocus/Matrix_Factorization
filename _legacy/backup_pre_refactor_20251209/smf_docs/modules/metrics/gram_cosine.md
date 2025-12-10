# M1: Gram Overlap Cosine

通过 Gram 矩阵的余弦相似度测量矩阵相似性

**模块 ID**: M1
**SMF 路径**: `sparse_matrix_factorization/modules/metrics/gram_cosine.py`

---

## 目的

计算学生矩阵与教师矩阵的 Gram 矩阵余弦相似度，作为重叠度（overlap）指标。

---

## 数学定义

对于矩阵 A 和 B：

**Left Gram (use_left=True)**:
```
G_A = A @ A^T
G_B = B @ B^T
```

**Right Gram (use_left=False)**:
```
G_A = A^T @ A
G_B = B^T @ B
```

**余弦相似度**:
```
overlap = trace(G_A * G_B) / (||G_A||_F * ||G_B||_F)
        = dot(flatten(G_A), flatten(G_B)) / (norm(G_A) * norm(G_B))
```

---

## 代码位置

| 程序 | 行号 | 状态 |
|------|------|------|
| bigamp/train.py | 137-154 | ✅ 一致 |
| bigamp/compare_sizes.py | 349-366 | ✅ 一致 |
| bigamp/orthogonal_teacher.py | 325-342 | ✅ 一致 |

---

## 标准实现

```python
@torch.no_grad()
def gram_overlap_cosine(A, B, use_left=True):
    """Compute Gram matrix overlap using cosine similarity.

    Args:
        A: Student matrix, shape (N1, M) or (M, N2)
        B: Teacher matrix, same shape as A
        use_left: If True, compute A@A^T vs B@B^T; else A^T@A vs B^T@B

    Returns:
        Cosine similarity in [-1, 1]
    """
    if use_left:
        G_A = A @ A.T
        G_B = B @ B.T
    else:
        G_A = A.T @ A
        G_B = B.T @ B

    G_A_flat = G_A.flatten()
    G_B_flat = G_B.flatten()

    dot = (G_A_flat * G_B_flat).sum()
    norm_A = G_A_flat.norm()
    norm_B = G_B_flat.norm()

    return float(dot / (norm_A * norm_B + 1e-12))
```

---

## 输入/输出

**输入**:
- `A`: torch.Tensor, 学生矩阵, shape (N1, M) 或 (M, N2)
- `B`: torch.Tensor, 教师矩阵, 与 A 相同 shape
- `use_left`: bool, 默认 True
  - True: 计算 W 的 Gram overlap (A@A^T vs B@B^T)
  - False: 计算 X 的 Gram overlap (A^T@A vs B^T@B)

**输出**:
- `float`: 余弦相似度，范围 [-1, 1]

---

## 使用场景

```python
# 计算 W 的 overlap (Q_W)
Q_W = gram_overlap_cosine(W_student, W_teacher, use_left=True)

# 计算 X 的 overlap (Q_X)
Q_X = gram_overlap_cosine(X_student, X_teacher, use_left=False)
```

---

## 注意事项

1. **尺度敏感**: 输出范围是 [-1, 1]，随机矩阵的期望值不是 0
2. **归一化版本**: 如需 [0, 1] 范围，使用 `gram_overlap_zero_to_one` (M2)
3. **数值稳定性**: 使用 1e-12 防止除零

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Measure similarity between student and teacher via Gram matrix cosine",
    "when_to_use_en": "Standard overlap metric for W or X matrices, raw cosine value",
    "limitations_en": "Output in [-1, 1], not normalized; random baseline is not 0",
    "inputs": ["A: (N1, M) or (M, N2)", "B: same shape as A", "use_left: bool"],
    "outputs": ["overlap: float in [-1, 1]"],
    "formula": "dot(G_A, G_B) / (norm(G_A) * norm(G_B))",
    "compute_cost": "O(N^2 * M) for N×M matrix",
    "related": ["gram_normalized (M2)", "qy (M3)"],
    "tags_en": ["overlap", "gram", "cosine", "W", "X", "Q_W", "Q_X", "evaluation", "metric", "similarity"],

    # 中文
    "purpose_zh": "通过 Gram 矩阵余弦相似度测量学生与教师的相似性",
    "when_to_use_zh": "W 或 X 矩阵的标准重叠度指标，原始余弦值",
    "limitations_zh": "输出范围 [-1, 1]，未归一化；随机基线不是 0",
    "tags_zh": ["重叠度", "格拉姆矩阵", "余弦相似度", "W矩阵", "X矩阵", "Q_W", "Q_X", "评估", "相似度"],

    # 日文
    "purpose_ja": "Gram行列のコサイン類似度で学生と教師の類似性を測定",
    "when_to_use_ja": "W/X行列の標準オーバーラップ指標、生のコサイン値",
    "limitations_ja": "出力範囲 [-1, 1]、正規化なし；ランダム基準は0ではない",
    "tags_ja": ["オーバーラップ", "グラム行列", "コサイン類似度", "W行列", "X行列", "評価", "類似度"],
}
```

---

*最后更新：2024年12月*
