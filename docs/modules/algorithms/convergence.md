# A7: 收敛检测

检测训练是否收敛的早停机制。

**模块 ID**: A7
**SMF 路径**: `modules/algorithms/convergence.py`

---

## 🌐 宏观视角

### 系统定位

```
算法层
├── bigamp/
│   └── (无早停，固定步数)
└── agd/
    ├── A4: core.py
    ├── A5: optimizer.py
    ├── A6: scheduler.py
    └── A7: convergence.py  ← 本模块（收敛检测）
```

### 引入动机

**问题**：AGD 训练 20k epochs 很慢，但实际可能早已收敛

```
典型训练曲线:
Loss
 │╲
 │ ╲
 │  ╲___________________
 │        ↑
 │    epoch 5000 已收敛
 │    后面 15000 epochs 都是浪费
 └──────────────────────── epoch
```

**解决方案**：早停（early stopping）
- 监控 loss 或 Q_Y
- 如果不再改善，提前终止

### 物理图景 🌟

**收敛判据**：

```
1. 绝对收敛:
   |loss_new - loss_old| < ε
   → loss 变化小于阈值

2. 相对收敛:
   |loss_new - loss_old| / loss_old < ε
   → loss 相对变化小于阈值

3. Patience 机制:
   连续 P 个 epoch 没有改善
   → 避免因噪声误判

4. Q_Y 收敛:
   Q_Y > 0.99
   → 重建质量足够好
```

### 使用场景

**适用**：
- AGD 训练
- 需要节省计算资源
- 超参数搜索（需要快速评估）

**不适用**：
- BiG-AMP（固定步数更可靠）
- 需要完整训练曲线的实验

---

## 🔬 微观视角

### 代码位置

| 程序 | 位置 | 说明 |
|------|------|------|
| agd/train_sequential.py | 训练循环内 | 可选早停检查 |
| agd/train_parallel.py | 训练循环内 | 可选早停检查 |

**注意**：当前 Wang/ 代码中早停是可选功能，默认运行固定 epochs。

### 数学定义

**基于 loss 的收敛检测**：

```
给定:
  loss_history: 最近 window 个 epoch 的 loss
  patience: 允许不改善的最大 epoch 数
  min_delta: 最小改善量

收敛条件:
  best_loss = min(loss_history[:-patience])
  current_loss = loss_history[-1]

  如果 current_loss > best_loss - min_delta
     且持续 patience 个 epoch
  → 判定为收敛
```

**基于 Q_Y 的收敛检测**：

```
如果 Q_Y > threshold (如 0.99)
→ 判定为收敛（重建成功）
```

### 输入/输出

```python
def check_convergence(history, patience=500, min_delta=1e-6, metric='loss'):
    """
    Args:
        history: List[float] - 历史指标值
        patience: int - 允许不改善的最大步数
        min_delta: float - 最小改善量
        metric: str - 监控指标 ('loss' 或 'Q_Y')

    Returns:
        converged: bool - 是否收敛
        best_value: float - 最佳指标值
    """
```

### 标准实现

```python
class EarlyStopping:
    """Early stopping to stop training when metric stops improving."""

    def __init__(self, patience=500, min_delta=1e-6, mode='min'):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for Q_Y (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self._is_improvement(current_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def _is_improvement(self, current_value):
        if self.mode == 'min':
            return current_value < self.best_value - self.min_delta
        else:  # mode == 'max'
            return current_value > self.best_value + self.min_delta


def check_convergence_simple(history, patience=500, min_delta=1e-6):
    """Simple convergence check without class."""
    if len(history) < patience:
        return False, min(history)

    recent = history[-patience:]
    best_before = min(history[:-patience]) if len(history) > patience else history[0]
    best_recent = min(recent)

    converged = best_recent > best_before - min_delta
    return converged, min(history)
```

### 训练循环中的使用

```python
# 使用类
early_stopping = EarlyStopping(patience=500, min_delta=1e-6, mode='min')

for epoch in range(max_epochs):
    loss = train_step(...)

    if early_stopping(loss):
        print(f"Early stopping at epoch {epoch}")
        break

# 使用函数
loss_history = []
for epoch in range(max_epochs):
    loss = train_step(...)
    loss_history.append(loss)

    if epoch % 100 == 0:
        converged, best = check_convergence_simple(loss_history, patience=500)
        if converged:
            print(f"Converged at epoch {epoch}, best loss: {best}")
            break
```

### 实现细节

1. **监控窗口**：不是每个 epoch 都检查，可以每 100 epochs 检查一次
2. **Patience 选择**：
   - 太小：可能因噪声误判
   - 太大：失去早停的意义
   - 推荐：总 epochs 的 5-10%

3. **Q_Y 阈值选择**：
   - 0.99：高质量重建
   - 0.95：一般质量
   - 0.90：低质量但可能足够

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Detect training convergence for early stopping",
    "when_to_use_en": "AGD training, save computation, hyperparameter search",
    "tags_en": ["convergence", "early stopping", "patience", "monitoring"],

    # 中文
    "purpose_zh": "检测训练收敛以实现早停",
    "when_to_use_zh": "AGD 训练、节省计算资源、超参数搜索",
    "tags_zh": ["收敛", "早停", "耐心值", "监控"],

    # 技术参数
    "inputs": ["history", "patience", "min_delta", "metric"],
    "outputs": ["converged: bool", "best_value: float"],
    "hyperparameters": ["patience=500", "min_delta=1e-6"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/algorithms/convergence.py`

---

*最后更新：2025年12月*
