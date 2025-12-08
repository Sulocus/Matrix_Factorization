# Prompt 3: 深度程序分析与优化报告

## 1. 核心问题诊断 (Critical Diagnosis)

经过与参考代码 (`Wang`) 的逐行比对和物理图景分析，发现以下关键差异和潜在问题：

### A. 更新公式的致命差异 (Update Rule Discrepancy)

**这是目前最可疑的 Bug。**

*   **参考代码 (`Wang/bigamp/train.py`, line 59)**:
    ```python
    w_hat_new = w_hat + w_var_new * r_W
    ```
    这是一个**增量更新**。新值 = 旧值 + (方差 * 梯度/残差)。这保留了之前的学习成果，类似于梯度下降步。

*   **当前代码 (`smf/.../bigamp_spreading_parallel.py`, line 569)**:
    ```python
    W_hat_new = W_var_new * r_W
    ```
    这是一个**替换更新**。**缺少了 `+ W_hat`**！
    这意味着每一步 `W_hat_new` 仅由当前的残差 `r_W` 决定。这导致算法“失忆”，每一步都像是在原点 (0) 附近主要基于当前梯度进行一次跳跃，而不是在现有估计基础上微调。这极大地解释了为什么算法难以收敛或性能极差。

### B. Q_Y 指标定义 (Metric Definition)

*   目前 `smf` 中的 `Q_Y` 是**余弦相似度 (Cosine Similarity)**：
    $$ Q_Y = \frac{Y_{pred} \cdot Y_{true}}{\|Y_{pred}\| \|Y_{true}\|} $$
*   物理上的序参量 $m$ (Magnetization) 通常指投影：
    $$ m = \frac{Y_{pred} \cdot Y_{true}}{\|Y_{true}\|^2} $$
*   **差异**: 由于收缩效应 (Shrinkage)，$\|Y_{pred}\|$ 通常小于 $\|Y_{true}\|$。余弦相似度会归一化掉这个模长差异，导致数值虚高 (看起来接近 1.0，但实际信号强度 $m$ 可能很小)。建议同时监控这两个指标。

### C. 物理尺度 (Physical Scaling) - 已修正

*   我们已经成功将 Teacher 和 Student 统一到了 **Mean Field Scaling** ($N(0,1)$)。
*   这解决了初始方差不匹配的问题，为算法提供了正确的物理起点。

### D. Onsager 校正

*   目前我们移除了 `Z` 计算中的显式 Onsager 校正 (`- s * V`) 和 `s` 的 Damping。
*   这与参考代码 (`Wang`) 保持了一致，是去除不稳定因素的正确举措。

---

## 2. 修正与优化方案 (Action Plan)

### 第一步：修复更新公式 (Top Priority)

必须立即在 `bigamp_spreading_parallel.py` 中找回丢失的 `+ W_hat` 和 `+ X_hat`。

**修改目标**:
```python
# W update
r_W = ...
W_hat_new = W_hat + W_var_new * r_W  # 加上 W_hat

# X update
r_X = ...
X_hat_new = X_hat + X_var_new * r_X  # 加上 X_hat
```
*(注意：需要确认 `Wang` 代码中 `r_W` 的确切数学含义。如果 `smf` 中的 `r_W` 定义与 `Wang` 完全一致，那么这个加法就是必须的。)*

### 第二步：完善指标监控

建议在 `smf/modules/metrics/spreading.py` 中增加 `overlap_projected` 指标：
```python
overlap = dot / (norm_t * norm_t)  # 除以真值的模长平方
```
直接对应物理序参量 $m$ (或 $m^2$)。

### 第三步：重新验证

在修复更新公式后，重新运行实验。结合 Mean Field Scaling 的修正，预期 Q_Y 应该能观察到明显的相变行为。

## 3. 待查证细节

*   **`r_W` 的定义**: 再次确认 `smf` 中的 `r_W` 计算逻辑（`scatter_add` 部分）是否在数学上等价于 `Wang` 中的 `torch.matmul(s, x.T)`。从代码看结构是一致的，都是 $\sum_{\mu} s_{i\mu} x_{\mu j}$。

---

**总结**: 强烈建议优先修复 **"丢失的 W_hat"** 问题。这很可能是导致算法表现不如旧代码的根本原因。

---

# Claude Code 独立分析报告 (2025-12-09)

## 4. 多 Agent 深度分析结果

通过 3 个并行 Explore agent 对 spreading 相关代码进行了全面分析：

### 分析方法
1. **Agent 1**: 算法实现对比（bigamp.py vs bigamp_spreading.py vs bigamp_spreading_parallel.py）
2. **Agent 2**: Teacher/Graph 模块分析（random_spreading.py, supergraph.py）
3. **Agent 3**: 指标和结果分析（spreading.py, 实验结果文件）

### 关键发现汇总

| 问题 | prompt3.md | Claude 分析 | 一致性 |
|------|-----------|-------------|--------|
| **缺少 `+ W_hat`** | ✅ 指出 | ✅ 确认 | **完全一致** |
| 方差公式差异 `1/(1+τ)` vs `1/(M+τ)` | ❌ 未提及 | ✅ 发现 | Claude 补充 |
| 初始化缩放 0.1 vs 1/√M | 提到已修正 | ✅ 发现仍存在 | Claude 补充 |
| Q_Y 指标定义 | ✅ 建议监控 | ❌ 未深入 | prompt3 更全面 |

---

## 5. O(1) 框架下的自洽性验证

### 背景说明

用户确认：**故意采用 O(1) 量级框架**，而非传统的 Mean Field Scaling (1/M 方差)。

因此以下设计是**有意的**，不是 bug：
- `1/(1 + tau_W)` 方差公式 → 对应先验 W ~ N(0, I)
- 初始方差 `1.0` → O(1) 量级
- 初始缩放 `0.1` 或其他 O(1) 值

### Sequential vs Parallel 对比验证

| 版本 | 方差公式 | 更新公式 | 框架 | 状态 |
|------|----------|----------|------|------|
| **Wang (参考)** | `1/(M + τ)` | `w + var*r` | Mean Field | ✅ 基准 |
| **Sequential spreading** | `1/(M + τ)` | `w + var*r` | Mean Field | ✅ 正确 |
| **Parallel spreading** | `1/(1 + τ)` | `var*r` | O(1) | ❌ 缺少 `+W` |

**关键证据**：
```python
# Sequential spreading (bigamp_spreading.py L133) - 正确
w_hat_new = w_hat + w_var_new * r_W  # ✅ 有 + w_hat

# Parallel spreading (bigamp_spreading_parallel.py L408) - 错误
W_hat_new = W_var_new * r_W  # ❌ 缺少 + W_hat
```

**结论**：无论使用 Mean Field 还是 O(1) 框架，`+ W_hat` 增量更新结构都是必需的。这是消息传递算法的基本结构，与先验选择无关。

---

## 6. 数学证明

BiG-AMP 后验均值更新（无论先验）：

$$\hat{w}_{new} = \sigma^2_{new} \left( \frac{\hat{w}_{old}}{\sigma^2_{old}} + r_W \right)$$

简化近似：

$$\hat{w}_{new} \approx \hat{w}_{old} + \sigma^2_{new} \cdot r_W$$

**`r_W` 代表增量信息（梯度方向），不是绝对估计。**

当前公式 `W_hat_new = W_var_new * r_W` 丢失了 `W_hat_old`，导致：
- 每一步都从零开始估计
- 之前学习的信息被丢弃
- 算法无法积累信息收敛

---

## 7. 修复实施记录

### 修复位置

**文件**: `smf/modules/algorithms/bigamp_spreading_parallel.py`

| 函数 | 行号 | 修改内容 |
|------|------|----------|
| `bigamp_spreading_step_all_alphas()` | L408 | `W_hat_new = W_hat + W_var_new * r_W` |
| `bigamp_spreading_step_all_alphas()` | L432 | `X_hat_new = X_hat + X_var_new * r_X` |
| `bigamp_spreading_step_disjoint()` | L569 | 同上（需处理形状） |
| `bigamp_spreading_step_disjoint()` | L584 | 同上（需处理形状） |

### 修复后验证

**测试 1** (N1=N2=100, M=25, steps=1000):
```
α=3.5: Q_W=0.53  ← 相变中，未完全收敛
```

**测试 2** (N1=N2=100, M=25, steps=3000, 更高 α):
```
α=2.0: Q_W=0.21
α=3.0: Q_W=0.27  ← 相变开始
α=4.0: Q_W=0.93  ← 相变完成！
α=5.0: Q_W=0.94
```

**结论**:
- 相变点 α_c ≈ 3.5
- α > 4 时 Q_W → 0.93+ (接近 1.0)
- 需要足够的迭代步数 (≥3000) 才能完全收敛

✅ **修复成功！Q_W 可达 0.93+！**

---

## 8. 结论

**prompt3.md 的分析完全正确**：`+ W_hat` 缺失是导致算法失效的根本原因。

**Claude 补充发现**：
1. 方差公式 `1/(1+τ)` vs `1/(M+τ)` 是 O(1) 框架的设计选择，**不需要修改**
2. 初始化相关参数在 O(1) 框架下是合理的，**不需要修改**
3. 唯一需要修复的是 `+ W_hat` / `+ X_hat` 增量更新
