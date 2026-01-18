# A4: AGD 核心梯度更新

交替梯度下降（Alternating Gradient Descent）的核心算法实现。

**模块 ID**: A4
**SMF 路径**: `modules/algorithms/agd/core.py`

---

## 🌐 宏观视角

### 系统定位

```
算法层
├── bigamp/
│   ├── A1: core.py      ← BiG-AMP 消息传递
│   ├── A2: state.py     ← 状态管理
│   └── A3: damping.py   ← 阻尼机制
└── agd/
    ├── A4: core.py      ← 本模块（核心梯度更新）
    ├── A5: optimizer.py ← Adam 优化器
    ├── A6: scheduler.py ← 学习率调度
    └── A7: convergence.py ← 收敛检测
```

### 引入动机

**最直观的矩阵分解方法**：

给定观测 Y = W* @ X* + noise（部分观测），
通过梯度下降最小化重建误差：

```
Loss = ||mask ⊙ (Y - W @ X)||²_F

梯度:
  ∂L/∂W = -2 × (mask ⊙ (Y - W @ X)) @ X^T
  ∂L/∂X = -2 × W^T @ (mask ⊙ (Y - W @ X))
```

### 相对优势

| 特性 | AGD | BiG-AMP |
|------|-----|---------|
| 实现难度 | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| 收敛速度 | ~20k epochs | ~200-5000 steps |
| 理论基础 | 梯度下降 | 统计物理 |
| 灵活性 | ✅ 高 | ❌ 低 |
| 早停 | ✅ 支持 | ❌ 固定步数 |

### 物理图景 🌟

**AGD 的优化景观**：

```
Loss 曲面:
                        ╭───╮
                       ╱     ╲
        ╭────────────╱       ╲────────────╮
       ╱           ╱           ╲           ╲
      ╱    local  ╱    global   ╲  local    ╲
     ╱    min    ╱     min       ╲  min      ╲
    ╱     ↓     ╱       ↓         ╲   ↓       ╲
───╯      ●    ╱        ★          ╲  ●        ╰───

AGD 容易陷入局部最小
BiG-AMP 理论上趋向全局最优
```

**交替更新的必要性**：

```
同时更新 W 和 X:
  → 梯度相互耦合
  → 更新方向不稳定
  → 收敛困难

交替更新:
  固定 X，更新 W → 子问题是凸的
  固定 W，更新 X → 子问题是凸的
  → 稳定收敛
```

### 使用场景

**适用**：
- 小到中等规模矩阵（N < 1000）
- 需要早停机制
- 需要灵活的损失函数修改
- 教学和原型验证

**不适用**：
- 大规模矩阵（N > 1000，用 BiG-AMP）
- 需要快速收敛

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| agd/train_parallel.py | 训练循环 | 215-280 |
| agd/train_sequential.py | 训练循环 | 189-250 |

### 数学定义

**损失函数**：
```
L(W, X) = (1/2C) × ||A ⊙ (Y - (1/√M) × W @ X)||²_F

其中:
  A: 观测掩码（0/1 矩阵）
  C: 观测点数量
  1/√M: 缩放因子
```

**梯度**：
```
∂L/∂W = -(1/(C√M)) × (A ⊙ residual) @ X^T
∂L/∂X = -(1/(C√M)) × W^T @ (A ⊙ residual)

其中 residual = Y - (1/√M) × W @ X
```

### 输入/输出

```python
def train_agd(Wt, Xt, Y_teacher, mask, epochs, lr, device):
    """
    Args:
        Wt: Tensor[N1, M] - 教师 W（用于评估）
        Xt: Tensor[M, N2] - 教师 X（用于评估）
        Y_teacher: Tensor[N1, N2] - 观测矩阵
        mask: Tensor[N1, N2] - 观测掩码
        epochs: int - 训练轮数
        lr: float - 学习率
        device: torch.device - 目标设备

    Returns:
        W: Tensor[N1, M] - 学生 W
        X: Tensor[M, N2] - 学生 X
        history: Dict - 训练历史（loss, Q_Y, 等）
    """
```

### 标准实现

```python
def train_agd_step(W, X, Y_target, mask, M, optimizer_W, optimizer_X):
    """Single AGD training step."""
    # Forward pass
    scale = 1.0 / math.sqrt(M)
    Y_pred = scale * torch.matmul(W, X)

    # Compute masked loss
    residual = Y_target - Y_pred
    masked_residual = mask * residual
    loss = 0.5 * (masked_residual ** 2).sum() / mask.sum()

    # Backward pass
    optimizer_W.zero_grad()
    optimizer_X.zero_grad()
    loss.backward()

    # Update
    optimizer_W.step()
    optimizer_X.step()

    return loss.item()


def train_agd(Wt, Xt, Y_teacher, mask, epochs, lr, device):
    """Full AGD training loop."""
    N1, M = Wt.shape
    M, N2 = Xt.shape

    # Initialize student
    W = torch.randn(N1, M, device=device, requires_grad=True)
    X = torch.randn(M, N2, device=device, requires_grad=True)

    # Optimizers
    optimizer_W = torch.optim.Adam([W], lr=lr)
    optimizer_X = torch.optim.Adam([X], lr=lr)

    history = {'loss': [], 'Q_Y': []}

    for epoch in range(epochs):
        loss = train_agd_step(W, X, Y_teacher, mask, M,
                              optimizer_W, optimizer_X)
        history['loss'].append(loss)

        # Evaluate periodically
        if epoch % 100 == 0:
            Q_Y = compute_Q_Y(W, X, Wt, Xt, M)
            history['Q_Y'].append(Q_Y)

    return W.detach(), X.detach(), history
```

### 实现细节

1. **缩放因子**：`1/√M` 保证正确的信噪比
2. **损失归一化**：除以观测点数 C
3. **Adam 优化器**：比 SGD 更稳定
4. **学习率**：典型值 0.01-0.1

### 收敛行为

```
典型收敛曲线:

Loss
 │╲
 │ ╲
 │  ╲
 │   ╲__________  ← plateau (可能陷入局部最小)
 │
 └────────────── epoch
```

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Alternating Gradient Descent for matrix factorization",
    "when_to_use_en": "Small-medium matrices (N<1000), need early stopping, flexible loss",
    "limitations_en": "Slow convergence (~20k epochs), may stuck in local minima",
    "tags_en": ["AGD", "gradient descent", "alternating", "optimization", "training"],

    # 中文
    "purpose_zh": "用于矩阵分解的交替梯度下降",
    "when_to_use_zh": "小到中等矩阵 (N<1000)、需要早停、灵活损失函数",
    "limitations_zh": "收敛慢 (~20k epochs)、可能陷入局部最小",
    "tags_zh": ["AGD", "梯度下降", "交替", "优化", "训练"],

    # 技术参数
    "inputs": ["Wt", "Xt", "Y_teacher", "mask", "epochs", "lr", "device"],
    "outputs": ["W", "X", "history"],
    "compute_cost": "O(epochs × N1 × N2 × M)",
    "hyperparameters": ["lr=0.01", "epochs=20000"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/algorithms/agd/core.py`

---

*最后更新：2025年12月*
