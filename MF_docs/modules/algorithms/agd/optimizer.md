# A5: Adam 优化器配置

AGD 中使用的 Adam 优化器封装和配置。

**模块 ID**: A5
**SMF 路径**: `modules/algorithms/agd/optimizer.py`

---

## 🌐 宏观视角

### 系统定位

```
算法层/agd/
├── A4: core.py      ← 核心梯度更新
├── A5: optimizer.py ← 本模块（Adam 优化器）
├── A6: scheduler.py ← 学习率调度
└── A7: convergence.py ← 收敛检测
```

### 引入动机

**为什么用 Adam 而不是 SGD？**

```
SGD 问题:
  - 学习率敏感
  - 不同参数可能需要不同学习率
  - 容易震荡

Adam 优势:
  - 自适应学习率
  - 动量加速收敛
  - 对超参数不敏感
```

### 物理图景 🌟

**Adam 的自适应机制**：

```
SGD: 固定步长
     ──●──────●──────●──────●→
       同样大的步子

Adam: 自适应步长
     ──●────●──●─●●●●●●●→
       大步    小步（接近最优时）

原理:
  - 一阶矩 m: 梯度的移动平均（方向）
  - 二阶矩 v: 梯度平方的移动平均（步长调节）

  更新 = m / (√v + ε)

  梯度大 → v 大 → 步长小（防止震荡）
  梯度小 → v 小 → 步长大（加速收敛）
```

### 使用场景

**适用**：
- AGD 训练
- 需要稳定收敛
- 不想仔细调学习率

**不适用**：
- BiG-AMP（不基于梯度下降）

---

## 🔬 微观视角

### 代码位置

| 程序 | 位置 | 行号 |
|------|------|------|
| agd/train_parallel.py | 优化器创建 | 205-206 |
| agd/train_sequential.py | 优化器创建 | 180-181 |

### 数学定义

**Adam 更新规则**：

```
# 输入: 梯度 g_t, 之前的 m_{t-1}, v_{t-1}
# 超参数: α (学习率), β1=0.9, β2=0.999, ε=1e-8

m_t = β1 × m_{t-1} + (1 - β1) × g_t      # 一阶矩估计
v_t = β2 × v_{t-1} + (1 - β2) × g_t²     # 二阶矩估计

# 偏差修正
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)

# 参数更新
θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
```

### 输入/输出

```python
def create_optimizer(params, lr, betas=(0.9, 0.999), eps=1e-8):
    """
    Args:
        params: 需要优化的参数（W 或 X）
        lr: float - 学习率
        betas: Tuple[float, float] - 动量系数 (β1, β2)
        eps: float - 数值稳定项

    Returns:
        optimizer: torch.optim.Adam - 优化器实例
    """
```

### 标准实现

```python
def create_optimizer(params, lr, betas=(0.9, 0.999), eps=1e-8):
    """Create Adam optimizer with standard configuration."""
    return torch.optim.Adam(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=0  # 通常不使用权重衰减
    )


def create_optimizers_for_agd(W, X, lr_W=0.01, lr_X=0.01):
    """Create separate optimizers for W and X."""
    optimizer_W = create_optimizer([W], lr=lr_W)
    optimizer_X = create_optimizer([X], lr=lr_X)
    return optimizer_W, optimizer_X
```

### 实现细节

1. **分离优化器**：W 和 X 使用独立的优化器
   - 动量状态分开累积
   - 可以设置不同学习率

2. **默认超参数**：
   - `lr = 0.01`: 比较保守，稳定
   - `β1 = 0.9`: 动量系数
   - `β2 = 0.999`: 二阶矩系数
   - `ε = 1e-8`: 防止除零

3. **无权重衰减**：矩阵分解问题通常不需要正则化

### 学习率选择指南

| 学习率 | 适用场景 | 注意事项 |
|--------|---------|---------|
| 0.001 | 非常保守 | 收敛慢但稳定 |
| **0.01** | 默认推荐 | 平衡速度和稳定性 |
| 0.1 | 激进 | 可能震荡 |

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Configure Adam optimizer for AGD training",
    "when_to_use_en": "AGD training setup",
    "tags_en": ["Adam", "optimizer", "learning rate", "momentum", "adaptive"],

    # 中文
    "purpose_zh": "为 AGD 训练配置 Adam 优化器",
    "when_to_use_zh": "AGD 训练设置",
    "tags_zh": ["Adam", "优化器", "学习率", "动量", "自适应"],

    # 技术参数
    "inputs": ["params", "lr", "betas", "eps"],
    "outputs": ["optimizer: torch.optim.Adam"],
    "hyperparameters": ["lr=0.01", "betas=(0.9, 0.999)", "eps=1e-8"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/algorithms/agd/optimizer.py`

---

*最后更新：2025年12月*
