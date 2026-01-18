# A3: BiG-AMP 阻尼机制

防止 BiG-AMP 迭代震荡的阻尼更新策略。

**模块 ID**: A3
**SMF 路径**: `modules/algorithms/bigamp/damping.py`

---

## 🌐 宏观视角

### 系统定位

```
算法层/bigamp/
├── A1: core.py      ← 核心消息传递
├── A2: state.py     ← 状态管理
└── A3: damping.py   ← 本模块（阻尼机制）
```

### 引入动机

**问题**：标准 AMP 更新可能震荡或发散

```
无阻尼:
  w_hat_new = f(x_hat, ...)
  x_hat_new = g(w_hat_new, ...)

问题: 更新幅度过大 → 来回震荡 → 不收敛
```

**解决方案**：阻尼（damping）= 新旧值的加权平均

```
有阻尼:
  w_hat = d × w_hat_old + (1-d) × w_hat_new

d=0.5 时: 每次只更新一半
→ 更新变得平滑
→ 收敛更稳定
```

### 物理图景 🌟

**阻尼的物理类比**：

想象一个弹簧系统：
```
无阻尼（欠阻尼）:
    ╭─╮   ╭─╮   ╭─╮
    │ │   │ │   │ │
────╯ ╰───╯ ╰───╯ ╰──── 持续震荡

有阻尼:
    ╭─╮
    │ ╰╮
────╯  ╰────────────── 平滑收敛

过阻尼:
    ╭
    │
────╯─────────────────── 收敛太慢
```

**在 AMP 中**：
- d = 0: 无阻尼，最快但可能震荡
- d = 0.5: 标准阻尼，平衡速度和稳定性
- d → 1: 过阻尼，非常稳定但收敛慢

### 阻尼系数选择

| 场景 | 推荐 d | 原因 |
|------|--------|------|
| 一般情况 | 0.5 | 平衡稳定性和速度 |
| 大矩阵 (N>1000) | 0.3-0.5 | 更稳定 |
| 小矩阵 (N<200) | 0.5-0.7 | 可以更激进 |
| 不收敛时 | 增大到 0.7-0.9 | 强制稳定 |

### 使用场景

**适用**：
- BiG-AMP 每步更新
- 需要平衡收敛速度和稳定性

**不适用**：
- AGD（使用动量代替）
- 单次计算（无迭代）

---

## 🔬 微观视角

### 代码位置

| 程序 | 位置 | 行号 |
|------|------|------|
| bigamp/train.py | W 更新阻尼 | 469-471 |
| bigamp/train.py | X 更新阻尼 | 483-485 |

### 数学定义

```
阻尼更新公式:
  θ_new = d × θ_old + (1 - d) × θ_computed

其中:
  θ ∈ {w_hat, w_var, x_hat, x_var}
  d ∈ [0, 1] 为阻尼系数

方差额外截断:
  var_new = clamp(var_new, min=1e-8, max=1.0)
```

### 输入/输出

```python
def apply_damping(old_value, new_value, damping):
    """
    Args:
        old_value: Tensor - 上一步的值
        new_value: Tensor - 本步计算的新值
        damping: float - 阻尼系数 d ∈ [0, 1]

    Returns:
        damped_value: Tensor - 阻尼后的值
    """
```

### 标准实现

```python
def apply_damping(old_value, new_value, damping):
    """Apply damping to prevent oscillation."""
    return damping * old_value + (1 - damping) * new_value


def apply_variance_damping(old_var, new_var, damping, min_var=1e-8, max_var=1.0):
    """Apply damping to variance with clamping."""
    damped = damping * old_var + (1 - damping) * new_var
    return torch.clamp(damped, min=min_var, max=max_var)
```

### 在核心循环中的使用

```python
for step in range(steps):
    # 计算新值
    w_hat_new, w_var_new = compute_w_update(...)

    # 应用阻尼
    w_hat = apply_damping(w_hat, w_hat_new, damping)
    w_var = apply_variance_damping(w_var, w_var_new, damping)

    # 类似处理 X
    x_hat_new, x_var_new = compute_x_update(...)
    x_hat = apply_damping(x_hat, x_hat_new, damping)
    x_var = apply_variance_damping(x_var, x_var_new, damping)
```

### 实现细节

1. **方差截断**：防止数值问题
   - `min=1e-8`: 防止除零
   - `max=1.0`: 防止方差爆炸（先验方差为 1）

2. **均值无截断**：均值理论上无界

3. **原地更新**：为节省内存，通常直接覆盖旧值

### 收敛诊断

```python
# 监控阻尼效果
change = (new_value - old_value).abs().mean()
if change > threshold:
    print(f"Warning: Large update {change:.4f}, consider increasing damping")
```

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Prevent oscillation in BiG-AMP via damped updates",
    "when_to_use_en": "Every BiG-AMP iteration step",
    "tags_en": ["damping", "stability", "oscillation", "convergence", "relaxation"],

    # 中文
    "purpose_zh": "通过阻尼更新防止 BiG-AMP 震荡",
    "when_to_use_zh": "每个 BiG-AMP 迭代步",
    "tags_zh": ["阻尼", "稳定性", "震荡", "收敛", "松弛"],

    # 技术参数
    "inputs": ["old_value", "new_value", "damping"],
    "outputs": ["damped_value"],
    "hyperparameters": ["damping=0.5", "min_var=1e-8", "max_var=1.0"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/algorithms/bigamp/damping.py`

---

*最后更新：2025年12月*
