# SMF 系统全面诊断与修复计划 (完整版)

## 0. 问题现象

| 问题 | 描述 | 严重性 |
|------|------|--------|
| **Q_Y 曲线异常** | Alpha=1 时 Q_Y ≈ 1.0（应在 Alpha~3 才相变）| 🔴 严重 |
| **部分指标无法绘制** | Q_Y_unobserved 等选择后不显示 | 🟡 中等 |
| **运行速度异常快** | 可能算法未正确执行 | 🔴 待验证 |

**用户配置**：N=200, M=50, Alpha 0.0-4.0

---

## 1. 诊断层次体系

### Layer 1: 数据流与参数传递
检查从 UI → Config → Runner → Algorithm 的完整数据流。

| 检查点 | 位置 | 检查方法 |
|--------|------|----------|
| Config 是否正确读取 | `runner.py` 入口 | 打印 `config.matrix.N1/N2/M`, `config.alpha.start/stop/step` |
| Teacher 是否正确创建 | `runner.py` L67 | 打印 `W_true.shape`, `X_true.shape` |
| Alpha 列表是否正确 | `config.alpha_values` | 打印完整列表 |
| SuperGraph 边数是否正确 | `supergraph.py` | 打印每个 alpha 的边数 |

### Layer 2: 物理模型正确性
验证 Spreading 物理模型的实现是否正确。

| 检查点 | 预期 | 检查方法 |
|--------|------|----------|
| F 分布 | Rademacher: {-1, +1} 均匀分布 | 打印 `F_super[0,0,:10]` 值和 dtype |
| Y 计算公式 | Y = (1/√M) Σ F·W·X | 验证 `compute_Y_super` 结果 |
| alpha_mask | 低 alpha 边数 << 高 alpha 边数 | 打印每个 alpha 的 `C_k` |

### Layer 3: 算法执行验证
确认 BiG-AMP 迭代是否真正执行。

| 检查点 | 预期 | 检查方法 |
|--------|------|----------|
| 迭代次数 | 实际执行 max_steps 次 | 添加计数器 |
| W_hat 变化 | 每步 W_hat 范数应变化 | 打印每 500 步的 `W_flat.norm()` |
| 残差变化 | s_values 应随迭代减小 | 打印 `s_values.norm()` |

### Layer 4: 指标计算审计
验证 Q_Y 等指标的计算逻辑。

| 检查点 | 预期 | 检查方法 |
|--------|------|----------|
| 初始 Q_Y | 学生随机初始化 → Q_Y ≈ 0 | 训练前计算 Q_Y |
| 最终 Q_Y | 低 alpha 接近 0，高 alpha 接近 1 | 打印完整曲线 |
| Q_Y_unobserved | **未实现** | 需要补充 |

### Layer 5: UI 绘图链路
验证结果能否正确显示。

| 检查点 | 预期 | 检查方法 |
|--------|------|----------|
| Runner 返回 keys | 应包含所有选中指标 | 打印 `results.keys()` |
| UI 绘图数据 | 应包含所有选中指标 | 打印 `plot_data.keys()` |

---

## 2. 根因分析

### 2.1 Q_Y 虚高的可能原因

**假设 A：alpha_mask 扩展错误**
在 `train_full_parallel` (Line 1100-1101):
```python
alpha_mask_exp = batch_alpha_mask.unsqueeze(1).expand(B, S, C_max).reshape(B, SC)
```
如果 `batch_alpha_mask` 全为 True，则所有 alpha 都使用全部边。

**验证**：打印每个 alpha 的活跃边数占比。

**假设 B：F 没有起到"打乱"作用**
如果 F 全为 1（而非 ±1），则 Y = WX/√M，变成简单矩阵分解，低 alpha 也能恢复。

**验证**：检查 `F_super` 的实际值和 dtype。

**假设 C：过拟合 (Observed vs Unobserved)**
当前 Q_Y 仅在**观测点**计算。如果算法只是"记住"了观测值而非真正学习，则：
- Observed Q_Y = 1.0 (高)
- Unobserved Q_Y = 0.0 (低)

**这是区分"死记硬背"和"真正学习"的唯一标准！**

**验证**：实现并计算 Q_Y_unobserved。

**假设 D：学生初始化太接近教师**

**验证**：训练前计算初始 Q_Y。

### 2.2 指标缺失的根因

**直接原因**：
Runner `_run_spreading_parallel` 返回的 keys：
```python
{
    "Q_Y": ...,
    "Q_W": ...,
    "Q_X": ...,
    "physical_overlap_Y": ...,
}
```

**缺失**：
- `Q_W_prime`, `Q_X_prime` - `compute_all_metrics_spreading_parallel` 没有计算
- `Q_Y_unobserved`, `Q_Y_observed` - 完全未实现
- `MSE`, `Gen_Error` - 未返回

### 2.3 运行速度快的原因

**可能是正常的**：Disjoint Union 并行化 + torch.compile 确实能带来巨大加速。

**需要验证**：算法是否真的执行了完整的迭代。

---

## 3. 验证测试计划

### Test 1：检验 F 分布
```python
# 验证 F_super 的分布是否正确
print(f"F_super dtype: {F_super.dtype}")
print(f"F_super[0,0,:10]: {F_super[0,0,:10]}")
# Rademacher 应该是 {-1, +1}，dtype=int8
# Gaussian 应该是连续值，dtype=float32
```

### Test 2：检验 alpha_mask
```python
# 验证 alpha_mask 是否正确
# 低 alpha（如 0.5）的边数应该远少于高 alpha（如 3.0）
print(f"alpha_mask shape: {supergraph.alpha_mask.shape}")
for a in range(len(alpha_values)):
    edges_a = supergraph.get_active_edges(a)
    print(f"Alpha {alpha_values[a]:.2f}: {edges_a} edges")
```

### Test 3：检验初始 Q_Y
```python
# 在训练前计算 Q_Y（应该接近 0）
# 因为学生是随机初始化的
initial_W = torch.randn(A, N1, M) * 0.1
initial_X = torch.randn(A, M, N2) * 0.1
initial_Q_Y = compute_qy_spreading_parallel(initial_W, initial_X, spreading_data, 0)
print(f"Initial Q_Y: {initial_Q_Y}")  # 应该接近 0
```

### Test 4：检验训练后 Q_Y 随 alpha 变化
```python
# 打印完整的 Q_Y vs Alpha 曲线
for a, alpha in enumerate(alpha_values):
    print(f"Alpha {alpha:.2f}: Q_Y = {results['Q_Y'][a]:.4f}")
# 应该看到 alpha 从 0 到 4 时，Q_Y 从 0 逐渐增加到 1
```

---

## 4. 关键代码检查清单

| 文件 | 行号 | 检查内容 |
|------|------|----------|
| `runner.py` | ~140 | `create_supergraph` 参数是否正确 |
| `runner.py` | ~148 | `generate_F_super` 是否使用正确的 f_distribution |
| `supergraph.py` | 全文件 | alpha_mask 的生成逻辑 |
| `bigamp_spreading_parallel.py` | ~1100 | alpha_mask_exp 扩展逻辑 |
| `metrics/spreading.py` | ~304 | `compute_all_metrics_spreading_parallel` 返回值 |

---

## 5. 修复计划

### Phase A: 完善指标返回 (优先级：高)

**文件**：`smf/core/runner.py`

**修改内容**：
1. 在 `_run_spreading_parallel` 中增加缺失指标的计算和返回：
   - `Q_W_prime`, `Q_X_prime`
   - `MSE`
2. 为不支持的指标返回 `None` 或空列表，而非不返回

---

### Phase B: 验证 SuperGraph 和 alpha_mask (优先级：极高)

**文件**：`smf/modules/graphs/supergraph.py`

**检查点**：
1. `create_supergraph` 是否正确计算每个 alpha 的边数
2. `alpha_mask` 是否正确反映不同 alpha 的边数

**测试方法**：添加 debug 打印，对比旧代码

---

### Phase C: 验证 Q_Y 计算 (优先级：极高)

**文件**：`smf/modules/metrics/spreading.py`

**检查点**：
1. `compute_qy_spreading_parallel` 是否正确计算学生 Y
2. 是否正确使用了相同的 F 系数
3. 是否正确截取了 Y_teacher[:C_k]

---

### Phase D: 实现 Q_Y_unobserved (优先级：极高)

**意义**：区分"过拟合/死记硬背"和"真正泛化学习"的唯一标准。

**实现方案**：
1. 在训练前划分 10% 边作为测试集
2. 训练只在 90% 观测边上进行
3. 评估时分别计算 Observed Q_Y 和 Unobserved Q_Y

---

### Phase E: 验证算法迭代 (优先级：高)

**文件**：`smf/modules/algorithms/bigamp_spreading_parallel.py`

**检查点**：
1. `train_full_parallel` 是否真正执行了 `max_steps` 次迭代
2. 每次迭代后 W_hat, X_hat 是否在变化
3. damping, noise_var 是否正确传递

---

### Phase F: UI 指标映射修复 (优先级：中)

**文件**：`smf/ui/app.py`

**修改内容**：
1. 修复指标 key 映射（处理 `_mean` 后缀）
2. 对于不支持的指标显示提示信息

---

### Phase G: 完善 Standard 模式 (优先级：中)

**文件**：`smf/core/runner.py` → `_run_standard_sweep`

目前 Standard 模式的实现是一个**简化占位符**，需要替换为正确的 BiG-AMP 算法。

---

## 6. 与旧代码的关键差异

| 位置 | 旧代码 | 新代码 | 潜在问题 |
|------|--------|--------|----------|
| Runner 入口 | 直接调用算法类 | 通过 Config 间接调用 | Config 字段映射可能错误 |
| SpreadingDataParallel 创建 | 在算法内部 | 在 Runner 中 | 数据格式可能不匹配 |
| F 生成种子 | 从 config.spreading.seed | 从 config.spreading.seed | ✅ 相同 |

---

## 7. 执行顺序

1. **先验证，后修复**：执行 Test 1-4 确认问题根因
2. **Phase B + C**：优先修复 SuperGraph 和 Q_Y 计算（核心物理问题）
3. **Phase D**：实现 Q_Y_unobserved（关键诊断指标）
4. **Phase A + F**：修复指标返回和 UI 显示
5. **Phase E**：验证算法迭代
6. **Phase G**：完善 Standard 模式

---

## 8. 待确认项

1. 是否先执行诊断脚本（添加打印 → 分析输出 → 定位问题）？
2. 还是直接按计划逐步修复？
3. Standard 模式是否需要完整实现，还是暂时只关注 Spreading Parallel？

---

**状态**：等待用户审批
