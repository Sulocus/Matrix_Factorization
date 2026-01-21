# Tensor 并行化技术架构

> 本文档维护 Tensor 模式的并行化优化设计，逐步更新。

## 1. 核心概念

### 1.1 SuperGraph / Disjoint Union

**什么是 SuperGraph？**

SuperGraph 是一种将多个独立的计算问题"合并"成一个大问题的技术。

```
独立问题:           SuperGraph:
┌─────┐ ┌─────┐     ┌─────────────┐
│ S1  │ │ S2  │ ==> │ S1 | S2 | S3│  ← 并排放置
└─────┘ └─────┘     └─────────────┘
   ↓       ↓              ↓
GPU 调用1  调用2        单次调用      ← 减少 kernel launch 开销
```

**Disjoint Union 的数学含义**：
- 多个图结构并排放置，节点不重叠
- 邻接矩阵变成块对角矩阵
- 一次矩阵运算处理所有子图

**为什么有效？**
- GPU 擅长大规模并行，小问题串行时大量核心闲置
- 合并后的大问题可以充分利用 GPU 带宽
- 减少 Python→CUDA 的调用开销

### 1.2 当前 Bipartite 实现

`spreading.py` 的 `train_full_parallel` 使用以下张量格式：

```python
W_student: (A, S*N1, M)   # A 个 alpha，S 个 sample，N1 节点
X_student: (A, S*N2, M)   # Disjoint Union 格式
F: (A, S*C, M)            # 合并后的 spreading 矩阵
Y: (A, S*C)               # 合并后的观测值
```

这允许一次 GPU 调用处理所有 alphas 和 samples。

### 1.3 当前 Tensor 实现的问题

`tensor_spreading.py` 是串行的：

```python
for alpha in alpha_values:          # 串行
    for sample in range(S):         # 串行
        result = _train_single_internal(...)  # 每次只处理 1 个问题
```

导致 GPU 利用率仅 ~25%（功耗 117W / 满载 ~400W）。

---

## 2. 并行化层次

| 层次 | 描述 | 当前状态 | 优化方向 |
|------|------|----------|----------|
| **Sample 并行** | 同一 alpha 下 S 个采样同时运行 | ✅ 已完成 | Phase 1 |
| **Alpha 并行** | 多个 alpha 值同时运行 | ❌ 串行 | ~~Phase 2~~ → Phase 3 |
| **张量运算** | 单个 BiG-AMP step 内的计算 | ✅ 已优化 | - |

---

## 3. 优化路线图

### Phase 1: Sample 并行化（✅ 已完成）

**实现文件**：
- `tensor_step_batch.py`：批量 BiG-AMP 步骤函数
- `tensor_hypergraph.py`：`generate_F_batch`, `generate_tensor_observations_batch`
- `tensor_spreading_parallel.py`：`BiGAMPTensorSpreadingParallel` 类

**实际效果**：
- GPU 利用率：~25% → ~100%
- 加速：2.96x
- 峰值显存：0.33GB (BF16+torch.compile)

---

### Phase 1.5: 性能优化（✅ 已完成）

**已应用优化**：
- TF32 全局启用
- BF16 混合精度（Ampere+ 自动检测）
- torch.compile `default` 模式
- CUDA Graph 标记

---

### ~~Phase 2: Alpha 并行（已跳过）~~

> [!WARNING]
> **决定跳过此阶段**，原因：
> 1. 需要修改 `MemoryEstimator` 支持 n 维张量显存估算
> 2. 需要修改 `ParallelCoordinator` 支持可变维度
> 3. Bipartite 的显存估算已经调试了很久，Tensor 更复杂
> 4. Phase 3 可以从头设计，避免这些问题

---

### Phase 3: TensorSuperGraph 完整重构（可选）

**核心思想**：创建 Tensor 专用的 SuperGraph 架构，而非复用 Bipartite 的基础设施。

**新增组件**：

```
src/matrix_factorization/modules/algorithms/bigamp/
├── tensor_supergraph.py   # TensorSuperGraph 类
├── tensor_step_batch.py   # 批量 BiG-AMP step
└── tensor_memory.py       # Tensor 专用显存估算
```

**TensorSuperGraph 设计**：

```python
class TensorSuperGraph:
    """
    管理 n 维张量的 Disjoint Union 结构。
    
    支持：
    - 任意 n 维
    - Alpha + Sample 并行
    - 动态批处理
    """
    
    def __init__(self, dims: Tuple[int, ...], n_alphas: int, n_samples: int):
        self.n = len(dims)
        self.dims = dims
        self.A = n_alphas
        self.S = n_samples
        
    def create_union(self, alpha_values: List[float]) -> TensorUnionData:
        """创建 Disjoint Union 数据结构"""
        # factors: (A, S, N_d, M) 或 (A, S*N_d, M)
        # F: (A, S, C, M) 或 (A, S*C, M)
        # indices: (A, S, n, C)
        ...
```

**优势**：
- 专为 Tensor 设计，不受 Bipartite 遗留代码限制
- 显存估算可以针对 n 维特点优化
- 支持未来扩展到 10+ 维

---

## 4. 显存估算公式

### Tensor 模式显存估算

```
VRAM ≈ factors + variances + forward/backward buffers

factors:     n × S × N × M × 4 bytes (float32)
variances:   n × S × N × M × 4 bytes
buffers:     ~2 × S × C × M × 4 bytes

其中 C = α × (Σ N_d) × M
```

**示例**（n=3, N=200, M=50, S=4, α=0.5）：
- C = 0.5 × 600 × 50 = 15000
- factors: 3 × 4 × 200 × 50 × 4 = 4.8 MB
- buffers: 2 × 4 × 15000 × 50 × 4 = 24 MB
- 总计: ~30 MB × overhead ≈ 100-200 MB per sample
- S=4 并行: ~1-2 GB

实际测量 ~4G，说明还有其他开销（torch.compile 缓存等）。

---

## 5. 维护记录

| 日期 | 更新内容 |
|------|----------|
| 2026-01-21 | 初始版本，Phase 1-3 设计 |
| 2026-01-21 | Phase 1 完成：Sample 并行化，2.96x 加速 |
| 2026-01-21 | Phase 1.5 完成：BF16 + torch.compile |
