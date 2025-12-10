# SMF 诊断结果分析与详细修复计划

## 1. 诊断发现

### 🔴 严重问题

**问题 1：Q_Y 曲线完全倒置！**

```
Alpha 0.0: Q_Y = 0.0000 ✅ (正确，因为 0 边)
Alpha 0.5: Q_Y = 1.0000 ❌ 应该接近 0
Alpha 1.0: Q_Y = 1.0000 ❌ 应该 < 0.5
Alpha 1.5: Q_Y = 0.9997 ❌
Alpha 2.0: Q_Y = 0.9935 ❌
Alpha 2.5: Q_Y = 0.9712 ❌
Alpha 3.0: Q_Y = 0.9342 ❌ 这里才应该开始接近 1
Alpha 3.5: Q_Y = 0.8997 ❌
Alpha 4.0: Q_Y = 0.8911 ❌ 应该最接近 1
```

**物理意义**：
- 低 alpha = 少量观测 → 信息不足，无法恢复 → Q_Y 应该 **低**
- 高 alpha = 大量观测 → 信息充足，可以恢复 → Q_Y 应该 **高**

**实际表现**：
- 低 alpha → Q_Y ≈ 1.0 (完美恢复？？？)
- 高 alpha → Q_Y 反而**下降**到 0.89

这**完全违背物理直觉**！

**问题 2：Q_W/Q_X 也是类似倒置趋势**

```
Alpha 0.5: Q_W = 0.2046 (低)
Alpha 4.0: Q_W = 0.4638 (高)
```

Q_W/Q_X 的趋势是对的（随 alpha 增加），但数值很低（理论上高 alpha 应该接近 1）。

### 🟡 次要问题

**问题 3：多个指标缺失**
- Q_W_prime, Q_X_prime
- Q_Y_unobserved, Q_Y_observed
- MSE, Gen_Error
- physical_overlap_W, physical_overlap_X

### ✅ 正常的部分

1. **SuperGraph alpha_mask 正确**：边数随 alpha 线性增加
2. **F 分布正确**：Rademacher {-1, +1}
3. **初始 Q_Y 正确**：训练前 Q_Y ≈ 0

---

## 2. 根因分析

### 假设 A：Q_Y 计算逻辑错误

Q_Y 是在**观测边**上计算的余弦相似度。

如果 Q_Y 在低 alpha 就接近 1，可能是：

1. **训练前 Q_Y 已经很高**：❌ 诊断已排除（初始 Q_Y ≈ 0）
2. **算法在少量边上过拟合**：低 alpha 时边少，算法更容易"记住"所有观测值，但这不代表真正学到了！
3. **Q_Y 计算使用了错误的 Y_teacher**：可能使用了全部边而非 masked 边

**最可能原因**：**观测边过拟合 + 缺少 Q_Y_unobserved 来区分**

### 假设 B：算法在低 alpha 时"记住"了答案

当边数很少（如 alpha=0.5 时仅 5000 边）时：
- 变量数：200×50 + 50×200 = 20,000 个
- 方程数：5000 个
- 自由度过多，算法可以轻松拟合观测值

**但这不意味着它真正学到了 W 和 X！**

证据：Q_W = 0.2 (低)，说明 W 矩阵本身并没有被恢复。

### 假设 C：compute_qy_spreading_parallel 计算方式有问题

需要检查：
1. 是否使用了正确的 F 系数
2. 是否只在活跃边（alpha_mask）上计算

---

## 3. 详细修复计划

### Phase 3.1：验证 Q_Y 计算逻辑

**目标**：确认 Q_Y 是否正确计算

**步骤**：
1. 在 `compute_qy_spreading_parallel` 中添加调试打印
2. 检查 Y_student 和 Y_teacher 的值
3. 验证是否只在 C_k 个边上计算

---

### Phase 3.2：实现 Q_Y_unobserved

**目标**：区分过拟合和真正学习

**实现方案**：
```python
def compute_qy_unobserved_spreading_parallel(...):
    """
    计算**未观测**边上的 Q_Y。
    
    如果这个值也高，说明真正学到了。
    如果这个值低而 observed 高，说明只是过拟合。
    """
    # 使用全部边（C_max）而非活跃边（C_k）
    # 对比学生和教师在未观测位置的预测
```

---

### Phase 3.3：修复 Q_W/Q_X 的低值问题

**可能原因**：
1. 计算公式错误
2. 缺少适当的归一化

**步骤**：
1. 对比旧代码的 `compute_cosine_similarity`
2. 验证 Gram 矩阵计算

---

### Phase 3.4：补全缺失指标

| 指标 | 实现位置 |
|------|----------|
| Q_W_prime, Q_X_prime | `compute_all_metrics_spreading_parallel` |
| Q_Y_unobserved | 新函数 |
| MSE | `compute_mse_spreading_parallel` |

---

### Phase 3.5：与旧代码对比验证

**目标**：使用相同参数，对比新旧代码的输出

**步骤**：
1. 创建 `compare_with_legacy.py` 脚本
2. 使用相同 seed (42, 12345)
3. 对比 Q_Y, Q_W, Q_X 曲线

---

## 4. 执行顺序

1. **Phase 3.1**：验证 Q_Y 计算逻辑（找出根因）
2. **Phase 3.2**：实现 Q_Y_unobserved（关键诊断指标）
3. **Phase 3.3**：修复 Q_W/Q_X
4. **Phase 3.4**：补全所有缺失指标
5. **Phase 3.5**：与旧代码对比

---

**状态**：准备开始执行
