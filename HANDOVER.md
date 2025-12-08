# SMF 任务交接文档

**最后更新**: 2025-12-09 (第二十二次更新)
**上次会话**: Spreading Parallel 功能完善 - F² Bug 修复 + UI/UX 改进

---

## 当前任务状态

### 已完成（本次会话）

- [x] **F² Algorithm Bug 修复（最关键）**
  - τ_W 和 τ_X 计算缺少 F² 项，导致方差衰减错误
  - 修复 4 处：`bigamp_spreading_parallel_step()` 2处 + `bigamp_step_disjoint_union()` 2处
  - 正确公式: `τ_W[i,μ] = (1/M) Σ_c F²[c,μ] × (1/V[c]) × x²[c,μ]`
  - 同时修复 V 计算也需要 F²

- [x] **F 分布选项弹出修复**
  - 修复 `_configure_custom()` 流程（用户使用的流程）
  - 修复 `_finalize_ai_config()` 流程（AI 推荐流程）
  - 选择 spreading 算法后自动弹出 F 分布 + Teacher 类型选项

- [x] **正交/非正交选项添加**
  - `_select_spreading()` 新增 teacher_type 选择
  - Standard: 标准高斯 W,X ~ N(0,1)
  - Orthogonal: QR 分解正交化 W,X
  - SpreadingConfig 新增 teacher_type 字段

- [x] **进度条 UI 修改**
  - Row 3: "Batch" → "Total"
  - 离散完成：只在 batch 完成时更新（不插值）
  - 不显示数字（因为左上角已有 Batch X/Y）

- [x] **显存限制调整**
  - 默认上限从 28GB 改为 24GB
  - `calculate_spreading_batches()` 和 `train_batch_alphas()` 均更新

### 待验证

- [ ] F 分布选项是否正确弹出（用户报告 "F还是没有选项" 后已修复）
- [ ] F² 修复后算法结果是否正确

### 待开始

- [ ] 恢复误删文件：results_db.py, queue_manager.py, analysis/compare.py

---

## 本次会话关键变更

### 修改文件

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `smf/modules/algorithms/bigamp_spreading_parallel.py` | **修复** | F² bug 4处修复 + 24GB 限制 |
| `smf/ui/wizard.py` | 修改 | `_configure_custom()` + `_finalize_ai_config()` spreading 检测 |
| `smf/core/config.py` | 修改 | SpreadingConfig 新增 teacher_type 字段 |
| `smf/core/progress.py` | 修改 | Batch → Total，离散完成 |
| `smf/core/memory_manager.py` | 修改 | 默认 24GB |

---

## F² Bug 详解

### 问题根因

τ_W 和 τ_X 的计算中**缺少 F²**：

```python
# 错误实现（缺少 F²）
tau_W_contrib = alpha_scale_sq * X_sel.pow(2) * inv_V

# 正确实现（包含 F²）
F_sq_expanded = F_expanded.pow(2)  # (1, C_max, M)
tau_W_contrib = alpha_scale_sq * F_sq_expanded * X_sel.pow(2) * inv_V
```

### 数学公式

正确：`τ_W[i,μ] = (1/M) Σ_{c: i_idx[c]=i} F²[c,μ] × (1/V[c]) × x²[c,μ]`

### 修复位置

| 文件 | 行号 | 函数 |
|------|------|------|
| bigamp_spreading_parallel.py | ~399-400 | `bigamp_spreading_parallel_step` - τ_W |
| bigamp_spreading_parallel.py | ~422 | `bigamp_spreading_parallel_step` - τ_X |
| bigamp_spreading_parallel.py | ~559-560 | `bigamp_step_disjoint_union` - τ_W |
| bigamp_spreading_parallel.py | ~575 | `bigamp_step_disjoint_union` - τ_X |

同时 V 的计算也需要 F²（~行 275-280）。

---

## 进度条显示

### 修改后
```
╭────────────────────────────────────────────────────────╮
│  ●  Batch  1/3     Alpha 0.0-1.5   Step  2500/5000     │
│  Step  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  50.0% │
│  Total ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0.0% │
│  Power 485W   VRAM 13.5G   Elapsed   1:30   ETA   1:30 │
╰────────────────────────────────────────────────────────╯
```

- Total 进度按 batch 离散更新（完成 1 batch 后跳到 33.3%）
- 不显示 "1/3" 数字（左上角已有 Batch X/Y）

---

## Spreading 配置界面

```
Spreading 算法需要配置 F 分布和教师类型:

F distribution:
 [1]  Gaussian    Standard normal distribution N(0,1)
 [2]  Rademacher  Random ±1 values
选择 (默认: [1]):

W/X 教师类型:
 [1]  Standard    标准高斯分布 W,X ~ N(0,1)
 [2]  Orthogonal  QR 分解正交化 W,X
选择 (默认: [1]):

F 生成种子: 12345
```

---

## 恢复命令

下一个对话使用:
```
/rem
```
即可恢复上下文。

---

*本文档由 Claude Code 在 2025-12-09 自动更新（第二十二次）*
