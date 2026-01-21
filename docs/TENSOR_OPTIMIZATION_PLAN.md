# Tensor 优化执行计划

> **本文档是持续维护的执行计划，每完成一项打勾。新对话请先阅读此文档。**
>
> **最后更新**: 2026-01-21 15:02 JST

## 核心原则

1. **不修改旧代码**：旧代码作为参考，新功能作为新模块
2. **分步执行**：每步先设计、实现、测试，再进入下一步
3. **使用 Sequential Thinking**：每个大项用 MCP 分析
4. **详细记录**：每步完成打勾，记录问题和解决方案

---

## Phase 1: Sample 并行化

**目标**：将 S 个 samples 同时处理，GPU 利用率 ~25% → ~100%

### 1.1 创建批量 tensor_step
- [x] 创建 `tensor_step_batch.py`（新模块，不修改 `tensor_step.py`）
- [x] `forward_pass_tensor_batch(factors, F, indices)` → `(S, C)`
- [x] `compute_variance_tensor_batch(...)` → `(S, C)`
- [x] `tensor_step_batch(...)` → 批量 BiG-AMP 迭代

### 1.2 创建批量 hypergraph
- [x] 在 `tensor_hypergraph.py` 添加 `generate_F_batch(S, C, M, ...)`
- [x] 添加 `generate_tensor_observations_batch` 返回 `(S, C, M)` 和 `(S, C)`

### 1.3 创建并行 spreading
- [x] 创建 `tensor_spreading_parallel.py`（新模块）
- [x] 创建 `BiGAMPTensorSpreadingParallel` 类
- [x] 修改 `_train_parallel` 处理所有 S 个样本并行

### 1.4 验证
- [x] 对比并行版与串行版的 `Q_Y` 结果（误差 < 0.1）✓
- [x] 性能基准测试：**2.96x 加速**，峰值显存 0.21GB

---

## Phase 1.5: 性能优化

**目标**：套用 Bipartite 的优化技术

### 1.5.1 TF32 / BF16
- [x] 添加 TF32 全局启用（参考 `spreading.py:53-54`）
- [x] 添加 BF16 混合精度支持（参考 `spreading.py:205-210`）
- [x] `storage_dtype = torch.bfloat16`

### 1.5.2 torch.compile
- [x] 添加 `torch.compile` 支持（参考 `spreading.py:168-192`）
- [x] 使用 `default` 模式（安全模式）
- [N/A] 小问题用 `reduce-overhead` 模式（跳过，避免 CUDA Graph 问题）
- [x] 添加 CUDA Graph mark（`cudagraph_mark_step_begin`）
- [N/A] Clone 策略（Tensor 模式暂不需要）

### 1.5.3 Pre-computation
- [ ] Pre-flatten F, Y 避免每步 reshape
- [ ] Rademacher F²=1 优化

### 1.5.4 互联网搜索
- [ ] 搜索 torch.compile 对高维张量的支持
- [ ] 搜索 einsum 优化技巧
- [ ] 搜索 scatter_add 高效实现
- [ ] 记录发现的新优化方案

---

## Phase 2: 显存管理

**目标**：动态批处理 + OOM 恢复

### 2.1 显存估算
- [x] 评估公式估算 vs 试探法 → 选择试探法
- [x] 实现选定的方案 (`tensor_memory.py`)
- [x] `probe_tensor_memory(dims, M, S, alpha)` 创建小试验图测量实际显存

### 2.2 动态批处理
- [x] 创建 `TensorBatchCoordinator`（参考 Bipartite 的 `ParallelCoordinator`）
- [x] 根据显存自动分批 (`create_batches`)

### 2.3 OOM 恢复
- [x] 添加 OOM 捕获和重试逻辑 (`handle_oom`)

---

## Phase 3: TensorSuperGraph（可选）

**目标**：完整重构，支持 Alpha + Sample 并行

### 3.1 架构设计
- [ ] 设计 `TensorSuperGraph` 类
- [ ] 定义 Disjoint Union 数据结构

### 3.2 实现
- [ ] 创建 `tensor_supergraph.py`
- [ ] 创建 `tensor_step_super.py`

### 3.3 测试
- [ ] 完整功能测试
- [ ] 性能对比测试

---

## 参考：Bipartite 优化技术清单

| 技术 | 文件位置 | 应用状态 |
|------|----------|----------|
| TF32 全局启用 | `spreading.py:53-54` | [x] 已应用 |
| BF16 混合精度 | `spreading.py:205-210` | [x] 已应用 |
| torch.compile 分级 | `spreading.py:168-192` | [x] 已应用 (default) |
| CUDA Graph mark | `spreading.py:579-581` | [x] 已应用 |
| Clone 防地址污染 | `spreading.py:616-622` | [N/A] Tensor 模式不需要 |
| Pre-flatten 数据 | `spreading.py:529-530` | [ ] 待应用 |
| Rademacher F²=1 | `step.py` | [ ] 待应用 |

---

## 问题记录

| 日期 | 问题 | 解决方案 |
|------|------|----------|
| | | |

---

## 变更日志

| 日期 | 变更内容 |
|------|----------|
| 2026-01-21 15:02 | 初始版本 |
| 2026-01-21 15:20 | Phase 1 完成，验证通过 |
| 2026-01-21 16:17 | Phase 1.5 完成 (BF16 + torch.compile) |
