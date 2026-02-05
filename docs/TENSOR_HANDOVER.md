# Tensor 并行化 - 任务交接文档

## 给新会话的说明

> **复制以下内容给新会话即可开始工作**

---

### 项目背景

我在 `Matrix_Factorization` 项目中开发 N 维张量分解的 BiG-AMP 算法（`tensor_order: 3`）。之前完成了基础的 Alpha + Sample 并行化架构，但存在严重性能问题。

### 上一个会话完成的工作

**性能优化（已完成）**：
1. **offset_indices 预计算** - 在 `tensor_supergraph.py` 中预计算索引，消除每步 12 次重复计算
2. **torch.compile 集成** - 对 `tensor_step_super` 进行编译优化
3. **variance 公式修复** - 将 `1/tau` 改为 `1/(M+tau)`
4. **step_callback 优化** - 改为每步调用（Rich 自动限流）

**修改的文件**：
- `src/matrix_factorization/modules/algorithms/bigamp/tensor_supergraph.py`
- `src/matrix_factorization/modules/algorithms/bigamp/tensor_step_super.py`
- `src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading_parallel.py`

**验证结果**：计算速度提升，step 更新连续，GPU 功耗接近 ~480W

---

### 待解决的问题（需要你处理）

#### 问题 1：显存占用异常大

**现象**：VRAM 只用了 4.3G，但感觉应该更小或问题出在其他地方

**调查方向**：
- F 矩阵（Rademacher ±1）是否用了 `int8` 存储？还是 `float32`？
- 中间张量是否有不必要的 `float32`？应该用 `bfloat16`
- 对于 n=3 的情况，F 是 (S, C_max, M) 的形状，检查存储格式

**相关文件**：
- `tensor_supergraph.py` 的 `create_tensor_superdata()` - F 生成
- `tensor_step_super.py` - 中间计算精度

#### 问题 2：算法物理上完全错误

**现象**：Q_Y 在零附近停止，完全没有收敛。这说明算法的物理背景有根本性错误。

**调查方向**：
- 对比 Bipartite 版本 (`spreading.py`, `step.py`) 的正确实现
- 检查 n 维张量的 BiG-AMP 更新公式是否正确
- 验证 forward_pass、variance、backward pass 的数学正确性
- 可能需要从物理/数学角度重新推导 n 维情况

**相关文件**：
- `tensor_step_super.py` - 核心算法（需要与 `step.py` 对比）
- `spreading.py` 的 `bigamp_step_disjoint_union_flat` - 正确参考

#### 问题 3：批次参数未正确传递到 UI

**现象**：
- Alpha 分批处理（如 0-2.0, 2.6-4.0），但 UI 显示"1/1 批次"
- Alpha 范围显示不正确（总是显示 0-4.0）
- 用户无法从界面了解当前在处理哪个批次

**调查方向**：
- `_train_full_parallel()` 中的 `step_callback` 参数传递
- `_compute_alpha_batches()` 的批次信息如何传递到 UI
- `progress.py` 的 `ProgressBridge` 如何接收和显示批次信息

**相关文件**：
- `tensor_spreading_parallel.py` - 批次生成和回调
- `core/progress.py` - UI 显示逻辑
- `core/experiment/runner.py` - 事件传递

---

### 文件结构

```
src/matrix_factorization/modules/algorithms/bigamp/
├── tensor_spreading_parallel.py  # 主算法类
├── tensor_step_super.py          # BiG-AMP 步函数（已优化）
├── tensor_supergraph.py          # 数据结构（已添加 offset_indices）
├── spreading.py                  # Bipartite 版本（正确参考）
└── step.py                       # Bipartite step 函数（正确参考）
```

---

### 运行测试

```bash
# 设置 tensor_order: 3 运行 Tensor 模式
mf

# 设置 tensor_order: 2 运行 Bipartite 模式（用于对比）
mf
```

---

## 已完成工作的详细记录

### Phase 1: offset_indices 预计算

**文件**: `tensor_supergraph.py`

添加了 `offset_indices` 字段到 `TensorSuperGraph`，在 `create_tensor_supergraph()` 中预计算：

```python
offset_indices = []
for d in range(n):
    N_d = dims[d]
    idx_flat = indices[d].reshape(-1)
    sample_offsets = torch.arange(S, device=device).unsqueeze(1) * N_d
    sample_offsets = sample_offsets.expand(S, C_max).reshape(-1)
    offset_indices.append(idx_flat + sample_offsets)
```

### Phase 2: tensor_step_super 重构

**文件**: `tensor_step_super.py`

- 修改参数：`indices_flat` → `offset_indices`
- 添加参数：`M: int`
- 删除所有 `sample_offsets` 重复计算
- 修复 variance：`1.0 / tau_d` → `1.0 / (M + tau_d)`

### Phase 3: torch.compile + 调用链

**文件**: `tensor_spreading_parallel.py`

- 添加 `_compiled_step_super` 类变量
- 在 `__init__` 中编译 `tensor_step_super`
- 使用 `offset_indices` 替代 `indices_flat`
- `step_callback` 改为每步调用

---

## 问题优先级建议

1. **问题 2（物理错误）** - 最关键，算法不收敛等于完全无法使用
2. **问题 1（显存）** - 次要，但影响大规模问题
3. **问题 3（UI）** - 最低优先级，不影响功能
