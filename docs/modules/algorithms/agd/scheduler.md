# A6: 学习率调度器

AGD 训练过程中的学习率动态调整策略。

**模块 ID**: A6
**SMF 路径**: `modules/algorithms/agd/scheduler.py`

---

## 🌐 宏观视角

### 系统定位

```
算法层/agd/
├── A4: core.py      ← 核心梯度更新
├── A5: optimizer.py ← Adam 优化器
├── A6: scheduler.py ← 本模块（学习率调度）
└── A7: convergence.py ← 收敛检测
```

### 引入动机

**固定学习率的问题**：

```
训练初期:
  - 距离最优解远
  - 需要大学习率快速接近

训练后期:
  - 接近最优解
  - 大学习率会震荡
  - 需要小学习率精细调整

解决: 学习率调度 - 随训练进度调整
```

### 物理图景 🌟

**学习率衰减的效果**：

```
固定学习率:
     ╭───╮
    ╱     ╲     ╭───╮
───╱       ╲───╱     ╲─── 持续震荡

学习率衰减:
     ╭───╮
    ╱     ╲
───╱       ╲_____________ 平滑收敛
```

### 常用调度策略

| 策略 | 公式 | 适用场景 |
|------|------|---------|
| StepLR | lr × γ^(epoch/step) | 分段衰减 |
| ExponentialLR | lr × γ^epoch | 平滑衰减 |
| CosineAnnealing | lr × (1 + cos(πt/T))/2 | 周期性训练 |
| ReduceOnPlateau | 根据 loss 自动调整 | 自适应 |

### 使用场景

**适用**：
- 长时间 AGD 训练（>5000 epochs）
- 需要精细收敛

**不适用**：
- BiG-AMP（固定步数）
- 短期训练（<1000 epochs）

---

## 🔬 微观视角

### 代码位置

| 程序 | 位置 | 行号 |
|------|------|------|
| agd/train_parallel.py | scheduler 创建 | 207-208 |
| agd/train_sequential.py | scheduler 创建 | 182-183 |

**注意**：当前 Wang/ 代码主要使用固定学习率，scheduler 是可选增强。

### 数学定义

**StepLR**:
```
lr_t = lr_0 × γ^⌊t/step_size⌋

示例 (lr_0=0.01, γ=0.1, step_size=5000):
  epoch 0-4999:    lr = 0.01
  epoch 5000-9999: lr = 0.001
  epoch 10000+:    lr = 0.0001
```

**CosineAnnealing**:
```
lr_t = lr_min + (lr_max - lr_min) × (1 + cos(πt/T)) / 2

特点: 平滑变化，周期结束时 lr = lr_min
```

### 输入/输出

```python
def create_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Args:
        optimizer: torch.optim.Optimizer - 优化器
        scheduler_type: str - 调度器类型
        **kwargs: 调度器特定参数

    Returns:
        scheduler: torch.optim.lr_scheduler - 调度器实例
    """
```

### 标准实现

```python
def create_scheduler(optimizer, scheduler_type='step', **kwargs):
    """Create learning rate scheduler."""
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 5000),
            gamma=kwargs.get('gamma', 0.5)
        )
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.9999)
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 10000),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 1000)
        )
    else:
        return None  # 不使用调度器


def scheduler_step(scheduler, loss=None):
    """Step the scheduler (handles different scheduler types)."""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(loss)
    else:
        scheduler.step()
```

### 训练循环中的使用

```python
# 创建
optimizer_W = create_optimizer([W], lr=0.01)
optimizer_X = create_optimizer([X], lr=0.01)
scheduler_W = create_scheduler(optimizer_W, 'step', step_size=5000)
scheduler_X = create_scheduler(optimizer_X, 'step', step_size=5000)

# 训练循环
for epoch in range(epochs):
    loss = train_step(...)

    # 更新学习率
    scheduler_step(scheduler_W, loss)
    scheduler_step(scheduler_X, loss)
```

### 实现细节

1. **分离调度**：W 和 X 可以有不同的调度策略
2. **ReduceLROnPlateau 特殊处理**：需要传入 loss
3. **与 Adam 的配合**：Adam 本身有自适应性，调度器效果可能有限

### 推荐配置

| 训练时长 | 推荐调度器 | 参数 |
|----------|-----------|------|
| < 5000 epochs | 无 | - |
| 5000-20000 epochs | StepLR | step=5000, γ=0.5 |
| > 20000 epochs | CosineAnnealing | T_max=epochs |

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Dynamically adjust learning rate during AGD training",
    "when_to_use_en": "Long AGD training (>5000 epochs), need fine convergence",
    "tags_en": ["scheduler", "learning rate", "decay", "cosine", "step"],

    # 中文
    "purpose_zh": "AGD 训练过程中动态调整学习率",
    "when_to_use_zh": "长时间 AGD 训练 (>5000 epochs)、需要精细收敛",
    "tags_zh": ["调度器", "学习率", "衰减", "余弦", "阶梯"],

    # 技术参数
    "inputs": ["optimizer", "scheduler_type", "kwargs"],
    "outputs": ["scheduler"],
    "scheduler_types": ["step", "exponential", "cosine", "plateau"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/algorithms/agd/scheduler.py`

---

*最后更新：2025年12月*
