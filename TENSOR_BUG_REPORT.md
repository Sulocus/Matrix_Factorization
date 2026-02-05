# Tensor GAMP 算法严重 Bug 诊断报告

> **日期**: 2026-02-04  
> **状态**: 🔴 未解决 - 存在根本性算法错误  
> **紧急程度**: 高 - 阻塞所有 n≥3 张量实验

---

## 1. 问题摘要

`bigamp_tensor_parallel` 算法在 n=3 张量分解任务中产生**物理上不可能的结果**：

- **Cold Start (Random Init) + Alpha=0.1 (10% 观测)** → Q ≈ 0.999+ (近乎完美重构)
- **信息论极限**: 在只有 10% 观测的情况下，从纯随机初始化不可能收敛到完美解

这意味着算法代码中存在**数据泄漏**或**根本性逻辑错误**。

---

## 2. 证据汇总

### 2.1 诊断脚本结果 (隔离测试)

运行 `debug_tensor_leak.py` 的结果：

```
=== Test 2: Z_hat with Random Factors (Cold Start) ===
  MSE per alpha: [1.69, 2.02, 2.02, 1.99, 1.97]
  Q per alpha (should be << 1): [-0.71, -1.04, -1.05, -1.01, -0.99]
```

**结论**: 隔离测试中，Random Init 产生 Q < 0（正确的物理行为）。

### 2.2 完整算法运行结果

用户配置：
```yaml
init_mode: random   # Cold Start
init_overlap: 0     # 无 Teacher 信息
alpha_scan: 0.1-4.0
```

运行结果：
```
INIT DEBUG: Using RANDOM INIT (Cold Start)
DEBUG: Factor 0 init stats: Mean=2.59e-03, Std=1.00
DEBUG: Factor 1 init stats: Mean=-7.17e-04, Std=1.01

DEBUG Step 1500: MSE=0.011, FactorStd=1.17
...
DEBUG Q_Y raw (first 5): [0.9996, 0.9996, 0.9995, 0.9995, 0.9995]
```

**矛盾**: 隔离测试 Q < 0，完整算法 Q ≈ 1.0。

### 2.3 物理不可能性

对于 n=3 张量 CP 分解：
- 未知变量: 3 × N × M = 3 × 50 × 10 = 1500
- Alpha=0.1 时观测数: C = 0.1 × M × N = 50
- **自由度比**: 50 / 1500 = 0.033 (3.3%)

在只有 3.3% 约束的情况下，系统是严重欠定的，不可能从随机初始化收敛。

---

## 3. 已排查的可能原因

### 3.1 ✅ 配置加载问题 (已解决)

**问题**: `init_mode` 和 `init_overlap` 未被正确读取。

**根因**: 
1. 代码使用 `getattr(..., 'warm_start_rho')` 但 YAML 键名是 `init_overlap`
2. 代码检查 `if init_mode == 'warm_start'` 但 YAML 值是 `'teacher'`

**修复**: 已修正参数绑定和字符串匹配。

### 3.2 ✅ Warm Start 维度错误 (已解决)

**问题**: Teacher Factors 扩展时维度不匹配，导致 `RuntimeError`。

**根因**: 使用 `expand(A, S, -1, -1)` 后未 `reshape` 到 `(A, S*N_d, M)`。

**修复**: 添加了正确的 `reshape` 操作。

### 3.3 ✅ `else` 分支缺少初始化 (已解决)

**问题**: `init_mode` 不是 `warm_start/teacher/spectral` 时，`factors` 未初始化。

**修复**: 添加了 Random Init 逻辑。

### 3.4 ❌ Alpha Mask 应用 (未发现问题)

检查了 `tensor_step_super.py`：
- L206: `s_values = s_values * alpha_mask.float()` ✓
- L248: `tau_contrib = ... * mask_exp` ✓

Mask 看起来正确应用于残差和精度更新。

### 3.5 ❌ offset_indices 范围 (未发现问题)

诊断脚本验证：
```
Dim 0: offset_indices range: [0, 199], valid range: [0, 199] ✓
Dim 1: offset_indices range: [0, 199], valid range: [0, 199] ✓
Dim 2: offset_indices range: [0, 199], valid range: [0, 199] ✓
```

### 3.6 ❌ Alpha 独立性 (未发现问题)

诊断脚本验证不同 Alpha 的 Factors 相互独立：
```
Correlation between Alpha 0.1 and Alpha 4.0 factors: ≈ 0.01 ✓
```

---

## 4. 未解决的核心问题

### 4.1 隔离测试 vs 完整算法的差异

**关键矛盾**:
- `debug_tensor_leak.py` 中单步 AMP (`tensor_step_super`) 行为正确
- 完整 `_train_full_parallel` 循环中 5000 步后 Q ≈ 1.0

**可能原因**:
1. AMP 循环中某处有隐藏的数据泄漏
2. `torch.compile` 优化引入了错误
3. `prev_s` (Onsager 修正) 传递有问题
4. 循环中 `factors` 被意外重写

### 4.2 Factor Std 发散但 Q=1

**观察**:
- `init_mode: teacher, rho=0` 时: FactorStd = 3.15 (发散), Q = 1.0
- `init_mode: random` 时: FactorStd = 1.17 (稳定), Q = 0.999

**这意味着算法找到了"等价解"**：不同的 Factors 但乘积相同？

### 4.3 Step 0 诊断未打印

我添加的 `STEP 0 DIAGNOSTIC` 代码没有出现在用户 Log 中，可能：
1. 代码语法错误导致未执行
2. 代码位置不正确

需要检查 `tensor_spreading_parallel.py` 中该代码块的实际状态。

---

## 5. 关键文件清单

| 文件 | 作用 | 怀疑程度 |
|------|------|----------|
| `tensor_spreading_parallel.py` | 主训练循环 | 🔴 高 |
| `tensor_step_super.py` | 单步 AMP 更新 | 🟡 中 |
| `tensor_supergraph.py` | 图结构与 Y 计算 | 🟡 中 |
| `tensor_metrics.py` | Q 值计算 | 🟢 低 |

---

## 6. 建议的下一步调查

### 6.1 对比隔离测试与完整循环

在完整循环中添加**逐步诊断**：
```python
for step in range(max_steps):
    if step == 0:
        print(f"BEFORE STEP 0: Z_hat-Y MSE = {compute_mse()}")
    
    factors, factor_vars, prev_s = step_fn(...)
    
    if step == 0:
        print(f"AFTER STEP 0: Z_hat-Y MSE = {compute_mse()}")
```

### 6.2 禁用 torch.compile

在 `config.yaml` 中设置 `use_compile: false`，排除编译器优化问题。

### 6.3 单步调试

使用 `debug_tensor_leak.py` 进行多步迭代，观察 MSE 变化：
```python
for step in range(100):
    factors, vars, s = tensor_step_super(...)
    Z_hat = forward_pass(...)
    mse = compute_mse(Z_hat, Y)
    print(f"Step {step}: MSE = {mse}")
```

### 6.4 检查 Y 是否被意外修改

在循环前后打印 `Y_flat.sum()`，确认 Y 未被修改。

### 6.5 深度代码审计

使用 `systematic-debugging` skill 对以下函数进行逐行审计：
1. `_train_full_parallel` (主循环)
2. `tensor_step_super` (更新公式)
3. `forward_pass_tensor_super` (Z_hat 计算)

---

## 7. 为新会话的指导

### 7.1 阅读此文档

```bash
cat /home/sucia/Matrix_Factorization/TENSOR_BUG_REPORT.md
```

### 7.2 使用 Sequential Thinking

```
请使用 sequential-thinking MCP 工具分析此问题：
1. 阅读 TENSOR_BUG_REPORT.md 了解背景
2. 运行 debug_tensor_leak.py 复现隔离测试
3. 逐步对比隔离测试与完整算法的差异
4. 使用 systematic-debugging skill 进行根因分析
```

### 7.3 使用相关 Skills

- `systematic-debugging`: 结构化 Bug 分析
- `code-quality`: 检查代码复杂度和潜在问题
- `scientific-unit-tests`: 验证数学公式正确性

### 7.4 关键检查点

1. [ ] `Y_flat` 在循环中是否不变？
2. [ ] `factors` 初始化后是否真正是随机？
3. [ ] `torch.compile` 是否引入问题？
4. [ ] Loop 中是否有全局状态污染？

---

## 8. 附录：诊断脚本

诊断脚本位于：`/home/sucia/Matrix_Factorization/debug_tensor_leak.py`

运行方式：
```bash
cd /home/sucia/Matrix_Factorization
python debug_tensor_leak.py
```

---

*此文档由 Antigravity Agent 生成，用于问题交接和后续调试。*
