# SMF 指标定义文档

本文档定义了 SMF 系统中使用的所有指标及其物理意义。

---

## 余弦相似度指标 (Cosine Similarity)

### Q_Y - Y 余弦相似度
**公式**: `cos(Y_s, Y_t) = <Y_s, Y_t> / (||Y_s|| * ||Y_t||)`

**意义**: 测量学生矩阵 Y_student 与教师矩阵 Y_teacher 的方向相似度。
- Q_Y = 1: 完美重建
- Q_Y ≈ 0: 随机/正交
- Q_Y = -1: 完全反向

### Q_W - W Gram 余弦相似度
**公式**: `cos(W_s @ W_s^T, W_t @ W_t^T)`

**意义**: 比较 W 的 Gram 矩阵（行-行协方差结构）。
消除了 W 和 X 的内在置换对称性（gauge symmetry）。

### Q_X - X Gram 余弦相似度
**公式**: `cos(X_s^T @ X_s, X_t^T @ X_t)`

**意义**: 比较 X 的 Gram 矩阵（列-列协方差结构）。

---

## 归一化重叠 (Normalized Overlap)

### Q_W_prime, Q_X_prime
**公式**: 在 [0, 1] 范围内的归一化版本

**意义**: 与 Q_W/Q_X 相同，但保证结果在 [0, 1] 区间。

---

## 观测/未观测边指标 (Observed/Unobserved)

### Q_Y_observed - 观测边余弦
**公式**: 只在观测到的边位置计算 cos(Y_s, Y_t)

**意义**: 测量**拟合能力**
- 高 Q_Y_observed 但低 Q_Y_unobserved = 过拟合
- 两者都高 = 真正学习

### Q_Y_unobserved - 未观测边余弦
**公式**: 只在未观测的边位置计算 cos(Y_s, Y_t)

**意义**: 测量**泛化能力/预测能力**
- 这是判断是否发生相变的关键指标
- 相变后 Q_Y_unobserved → 1

---

## 物理重叠 (Physical Overlap)

### physical_overlap_Y
**公式**: `<Y_s, Y_t> / <Y_t, Y_t>`

**意义**: Y_student 在 Y_teacher 方向上的投影系数（不归一化 Y_s）。

### physical_overlap_W, physical_overlap_X
**公式**: `|<W_s, W_t>| / <W_t, W_t>` （取绝对值处理符号对称性）

**意义**: 对应变量的投影系数。

---

## 误差指标 (Error Metrics)

### MSE - 均方误差
**公式**: `||Y_s - Y_t||² / N`

**意义**: 平均每个元素的平方误差。

### Gen_Error - 泛化误差
**意义**: 在未观测边上的预测误差。

---

## 相变行为总结

| 区域 | Q_Y_observed | Q_Y_unobserved | Q_W | 解释 |
|------|--------------|----------------|-----|------|
| α < α_c | 1.0 | ~0 | 低 | 过拟合 |
| α ≈ α_c | ~1 | 增长中 | 增长中 | 临界区 |
| α > α_c | 1.0 | 1.0 | ~1 | 相变成功 |

其中 α_c 是相变临界点（对于 N=200, M=50，大约在 α ≈ 3.4）。
