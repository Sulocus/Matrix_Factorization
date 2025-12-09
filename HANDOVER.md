# SMF 任务交接文档

**最后更新**: 2025-12-09 (第二十三次更新)
**上次会话**: Spreading Parallel 致命 Bug 修复 - `+ W_hat` 增量更新

---

## 当前任务状态

### 已完成（本次会话）

- [x] **`+ W_hat` 致命 Bug 修复（最关键）**
  - 问题：Update Rule 缺少增量更新，每步从零开始估计
  - 修复 4 处：
    - L408: `W_hat_new = W_hat + W_var_new * r_W`
    - L432: `X_hat_new = X_hat + X_var_new * r_X`
    - L569: `W_hat_new = W_flat + W_var_new * r_W`
    - L584: `X_hat_new = X_flat + X_var_new * r_X`
  - 验证结果：Q_W 在 α=4-5 时达到 0.93-0.94 ✅

- [x] **prompt3.md 更新**
  - 追加了 Claude Code 独立分析报告（Section 4-8）
  - 记录了修复位置和验证结果

- [x] **O(1) 框架确认**
  - 用户确认 `1/(1 + tau_W)` 是有意设计（非 Mean Field Scaling）
  - 初始方差 `1.0` 是有意设计
  - 这些**不是 bug**，无需修改

### 待验证

- [ ] F² Bug 修复后算法结果是否正确（上次会话遗留）
- [ ] F 分布选项是否正确弹出（上次会话遗留）

### 待开始

- [ ] 恢复误删文件：results_db.py, queue_manager.py, analysis/compare.py

---

## 本次会话关键变更

### 修改文件

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `smf/modules/algorithms/bigamp_spreading_parallel.py` | **修复** | `+ W_hat` 增量更新 4 处 |
| `prompt3.md` | 更新 | 追加 Claude Code 分析报告 |

---

## `+ W_hat` Bug 详解

### 问题根因

更新公式缺少增量项，导致算法每步"忘记"之前学习的内容：

```python
# 错误实现（缺少 + W_hat）
W_hat_new = W_var_new * r_W

# 正确实现（增量更新）
W_hat_new = W_hat + W_var_new * r_W
```

### 数学公式

BiG-AMP 后验均值更新：

$$\hat{w}_{new} = \sigma^2_{new} \left( \frac{\hat{w}_{old}}{\sigma^2_{old}} + r_W \right) \approx \hat{w}_{old} + \sigma^2_{new} \cdot r_W$$

**`r_W` 是增量信息（梯度方向），不是绝对估计。**

### 修复位置

| 文件 | 行号 | 函数 |
|------|------|------|
| bigamp_spreading_parallel.py | ~408 | `bigamp_spreading_step_all_alphas` - W_hat |
| bigamp_spreading_parallel.py | ~432 | `bigamp_spreading_step_all_alphas` - X_hat |
| bigamp_spreading_parallel.py | ~569 | `bigamp_step_disjoint_union` - W_hat |
| bigamp_spreading_parallel.py | ~584 | `bigamp_step_disjoint_union` - X_hat |

### 验证结果

| 配置 | 步数 | α 范围 | Q_W 结果 |
|------|------|--------|----------|
| 200×200, M=50 | 1000 | 0-3 | 0.53 ❌ |
| 200×200, M=50 | 3000 | 0-5 | 0.93-0.94 ✅ |

---

## O(1) 框架 vs Mean Field Scaling

| 项目 | Mean Field (Wang) | O(1) 框架 (当前) |
|------|-------------------|-----------------|
| 先验方差 | $1/M$ | $1$ |
| W_var 公式 | `1/(M + tau_W)` | `1/(1 + tau_W)` |
| 物理意义 | 强先验，小权重 | 弱先验，允许大波动 |
| 相变点 | 较早 (α ≈ 1) | 较晚 (α ≈ 3-4) |

用户**有意**选择 O(1) 框架，因此方差公式不需要修改。

---

## 恢复命令

下一个对话使用:
```
/rem
```
即可恢复上下文。

---

*本文档由 Claude Code 在 2025-12-09 自动更新（第二十三次）*
