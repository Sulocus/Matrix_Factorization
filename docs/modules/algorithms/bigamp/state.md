# A2: BiG-AMP 状态管理

BiG-AMP 算法的状态变量初始化和管理。

**模块 ID**: A2
**SMF 路径**: `modules/algorithms/bigamp/state.py`

---

## 🌐 宏观视角

### 系统定位

```
算法层/bigamp/
├── A1: core.py      ← 核心消息传递
├── A2: state.py     ← 本模块（状态管理）
└── A3: damping.py   ← 阻尼机制
```

### 引入动机

BiG-AMP 需要跟踪多个状态变量：
- `w_hat`, `x_hat`: 均值估计
- `w_var`, `x_var`: 方差估计

这些变量需要：
1. 正确初始化（影响收敛性）
2. 正确的形状（支持批量处理）
3. 适当的数值范围（防止溢出/下溢）

### 物理图景 🌟

**AMP 的贝叶斯视角**：

```
先验:
  W[i,k] ~ N(0, 1)    ← 均值=0, 方差=1
  X[k,j] ~ N(0, 1)

初始化:
  w_hat[i,k] = 0      ← 先验均值
  w_var[i,k] = 1      ← 先验方差

迭代后:
  w_hat → 后验均值（接近真值）
  w_var → 后验方差（置信度）
```

**随机初始化 vs 零初始化**：

```
零初始化（理论推荐）:
  w_hat = 0, x_hat = 0
  ↓
  第一次更新基于纯噪声
  ↓
  对称性自然打破

随机初始化（实践中有时使用）:
  w_hat ~ N(0, 0.01)
  ↓
  可能加速初期收敛
  ↓
  但可能引入偏差
```

### 使用场景

**适用**：
- BiG-AMP 算法启动
- 状态重置
- 多样本并行初始化

---

## 🔬 微观视角

### 代码位置

| 程序 | 位置 | 行号 |
|------|------|------|
| bigamp/train.py | `train_bigamp_parallel` 内部 | 444-451 |
| bigamp/train.py | `train_bigamp_single` 内部 | 507-514 |

### 数学定义

```
初始化:
  w_hat[batch, sample, i, k] = 0
  w_var[batch, sample, i, k] = 1
  x_hat[batch, sample, k, j] = 0
  x_var[batch, sample, k, j] = 1

形状说明:
  batch = num_alpha_values（α 值数量）
  sample = S（每个 α 的样本数）
  i = 0..N1-1（行索引）
  k = 0..M-1（隐维度索引）
  j = 0..N2-1（列索引）
```

### 输入/输出

```python
def initialize_bigamp_state(N1, N2, M, batch_size, sample_size, device, dtype):
    """
    Args:
        N1: int - 矩阵行数
        N2: int - 矩阵列数
        M: int - 秩（隐维度）
        batch_size: int - α 值数量
        sample_size: int - 每个 α 的样本数
        device: torch.device - 目标设备
        dtype: torch.dtype - 数据类型

    Returns:
        w_hat: Tensor[batch, sample, N1, M] - W 均值估计
        w_var: Tensor[batch, sample, N1, M] - W 方差估计
        x_hat: Tensor[batch, sample, M, N2] - X 均值估计
        x_var: Tensor[batch, sample, M, N2] - X 方差估计
    """
```

### 标准实现

```python
def initialize_bigamp_state(N1, N2, M, batch_size, sample_size, device, dtype):
    """Initialize BiG-AMP state variables."""
    # W 的状态
    w_hat = torch.zeros(batch_size, sample_size, N1, M,
                        device=device, dtype=dtype)
    w_var = torch.ones(batch_size, sample_size, N1, M,
                       device=device, dtype=dtype)

    # X 的状态
    x_hat = torch.zeros(batch_size, sample_size, M, N2,
                        device=device, dtype=dtype)
    x_var = torch.ones(batch_size, sample_size, M, N2,
                       device=device, dtype=dtype)

    return w_hat, w_var, x_hat, x_var
```

### 实现细节

1. **零初始化均值**：符合高斯先验 N(0,1)
2. **单位初始化方差**：表示最大不确定性
3. **批量维度**：支持多 α 并行计算
4. **样本维度**：支持多次重复实验

### 内存估算

```
每个状态变量:
  w_hat: batch × sample × N1 × M × 4 bytes (float32)
  w_var: batch × sample × N1 × M × 4 bytes
  x_hat: batch × sample × M × N2 × 4 bytes
  x_var: batch × sample × M × N2 × 4 bytes

总计: 4 × batch × sample × M × (N1 + N2) × 4 bytes

示例 (N1=N2=1000, M=50, batch=40, sample=5):
  ≈ 4 × 40 × 5 × 50 × 2000 × 4 = 320 MB
```

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Initialize and manage BiG-AMP state variables",
    "when_to_use_en": "Algorithm startup, state reset, batch initialization",
    "tags_en": ["state", "initialization", "mean", "variance", "prior"],

    # 中文
    "purpose_zh": "初始化和管理 BiG-AMP 状态变量",
    "when_to_use_zh": "算法启动、状态重置、批量初始化",
    "tags_zh": ["状态", "初始化", "均值", "方差", "先验"],

    # 技术参数
    "inputs": ["N1", "N2", "M", "batch_size", "sample_size", "device", "dtype"],
    "outputs": ["w_hat", "w_var", "x_hat", "x_var"],
    "memory_cost": "O(batch × sample × M × (N1 + N2))",
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/algorithms/bigamp/state.py`

---

*最后更新：2025年12月*
