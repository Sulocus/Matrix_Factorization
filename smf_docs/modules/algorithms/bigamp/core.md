# A1: BiG-AMP 消息传递

双线性广义近似消息传递（Bilinear Generalized AMP）的核心算法实现。

**模块 ID**: A1
**SMF 路径**: `modules/algorithms/bigamp/core.py`

---

## 🌐 宏观视角

### 系统定位

```
算法层
├── bigamp/
│   ├── A1: core.py      ← 本模块（核心消息传递）
│   ├── A2: state.py     ← 状态管理
│   └── A3: damping.py   ← 阻尼机制
└── agd/
    ├── A4: core.py      ← 梯度更新
    ├── A5: optimizer.py ← Adam 优化器
    ├── A6: scheduler.py ← 学习率调度
    └── A7: convergence.py ← 收敛检测
```

### 引入动机

**AGD 的问题**：
- 收敛慢（~20k epochs）
- 对初始化敏感
- 不保证全局最优

**BiG-AMP 的优势**：
- 收敛快（~200-5000 steps）
- 基于统计物理的原理性方法
- 在大系统极限下理论最优

### 相对优势

| 算法 | 收敛步数 | 理论基础 | 适用规模 |
|------|---------|---------|---------|
| AGD | ~20,000 | 梯度下降 | 小-中 |
| **BiG-AMP** | ~200-5000 | 消息传递/均场 | 中-大 |

### 物理图景 🌟

**AMP 的核心思想**：

把矩阵分解问题建模为因子图上的推断：

```
因子图:
    W[i,:]      X[:,j]
       ↘       ↙
         Y[i,j]      ← 观测节点
       ↗       ↖
    W[i',:]    X[:,j']

消息传递:
- 每个变量节点 W[i,:] 收集来自相邻因子的"信念"
- 每个因子节点 Y[i,j] 根据观测更新消息
- 迭代直到收敛
```

**AMP 的简化**：

标准 belief propagation 在有循环的图上不收敛。
AMP 通过以下近似使其可行：

1. **高斯假设**：消息用均值+方差参数化
2. **Onsager 修正**：减去"回声"项，防止信息重复计算
3. **大系统极限**：N→∞ 时渐进精确

**BiG-AMP 的特殊性**：

标准 AMP 用于线性问题 Y = AX。
BiG-AMP 扩展到双线性问题 Y = W @ X：

```
W 和 X 都是未知的
=> 需要交替更新 W 和 X 的估计
=> 复杂度从 O(N²) 变为 O(N²M)
```

### 使用场景

**适用**：
- 大规模矩阵（N > 1000）
- 需要快速收敛
- 理论研究

**不适用**：
- 需要早停机制时（BiG-AMP 固定步数）
- 需要灵活损失函数时

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/train.py | `train_bigamp_parallel` | 434-487 |
| bigamp/train.py | `train_bigamp_single` | 493-559 |

### 数学定义

**观测模型**：
```
Y = (1/√M) × W @ X + noise
Y[i,j] 只在 (i,j) ∈ Ω 处被观测
```

**变量**：
- `w_hat[i,k]`: W[i,k] 的估计均值
- `w_var[i,k]`: W[i,k] 的估计方差
- `x_hat[k,j]`: X[k,j] 的估计均值
- `x_var[k,j]`: X[k,j] 的估计方差

**更新方程**（简化）：

```
# W 更新
z_hat = (1/√M) × w_hat @ x_hat           # 预测
residual = (Y - z_hat) × mask            # 残差
s = residual / (p_var + noise_var)       # 缩放残差
tau_W = (1/M) × (mask/V) @ x_hat²        # 精度
w_var_new = 1 / (M + tau_W)              # 后验方差
w_hat_new = w_hat + w_var_new × s @ x_hat.T  # 后验均值

# X 更新（对称）
tau_X = (1/M) × w_hat².T @ (mask/V)
x_var_new = 1 / (M + tau_X)
x_hat_new = x_hat + x_var_new × w_hat.T @ s
```

### 输入/输出

```python
def train_bigamp_parallel(Wt, Xt, Y_teacher, A_all, alpha_values, steps, S,
                          damping=0.5, noise_var=1e-6):
    """
    Args:
        Wt: Tensor[N1, M] - 教师 W（用于评估）
        Xt: Tensor[M, N2] - 教师 X（用于评估）
        Y_teacher: Tensor[N1, N2] - 观测矩阵
        A_all: Tensor[num_alphas, S, N1, N2] - 观测掩码
        alpha_values: List[float] - α 值列表
        steps: int - 迭代步数
        S: int - 每个 α 的样本数
        damping: float - 阻尼系数 (0-1)
        noise_var: float - 假设的噪声方差

    Returns:
        w_hat: Tensor[num_alphas, S, N1, M] - W 的估计
        x_hat: Tensor[num_alphas, S, M, N2] - X 的估计
    """
```

### 标准实现（核心循环）

```python
for step in range(steps):
    # ===== W 更新 =====
    z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
    w_sq = w_hat ** 2
    x_sq = x_hat ** 2
    p_var = (alpha_scale ** 2) * (torch.matmul(w_sq, x_var) +
                                   torch.matmul(w_var, x_sq))
    V = torch.clamp(p_var + noise_var, min=1e-8)
    residual = (Y_teacher - z_hat) * A  # mask
    s = residual / V

    tau_W = (alpha_scale ** 2) * torch.matmul(A / V, x_sq.T)
    tau_W = torch.clamp(tau_W, min=1e-8)
    w_var_new = 1.0 / (M + tau_W)
    r_W = alpha_scale * torch.matmul(s, x_hat.T)
    w_hat_new = w_hat + w_var_new * r_W

    # 阻尼
    w_hat = damping * w_hat + (1 - damping) * w_hat_new
    w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new,
                        min=1e-8, max=1.0)

    # ===== X 更新 =====
    # 类似，对称交换 W 和 X
```

### 实现细节

1. **阻尼**：防止震荡，`new = d*old + (1-d)*update`
2. **方差截断**：`clamp(var, 1e-8, 1.0)` 防止数值问题
3. **缩放因子**：`1/√M` 保证正确的信噪比
4. **内存模式**：parallel（GPU 内存充足）vs single（内存受限）

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Bilinear Generalized AMP for fast matrix factorization",
    "when_to_use_en": "Large matrices (N>1000), fast convergence, theory research",
    "limitations_en": "Fixed steps (no early stop), assumes Gaussian prior",
    "tags_en": ["BiG-AMP", "AMP", "message passing", "belief propagation", "mean field"],

    # 中文
    "purpose_zh": "用于快速矩阵分解的双线性广义 AMP",
    "when_to_use_zh": "大矩阵 (N>1000)、快速收敛、理论研究",
    "limitations_zh": "固定步数（无早停）、假设高斯先验",
    "tags_zh": ["BiG-AMP", "AMP", "消息传递", "置信传播", "均场"],

    # 日文
    "purpose_ja": "高速行列分解のための双線形一般化AMP",
    "when_to_use_ja": "大行列 (N>1000)、高速収束、理論研究",
    "tags_ja": ["BiG-AMP", "AMP", "メッセージパッシング", "平均場"],

    # 技术参数
    "inputs": ["Wt", "Xt", "Y_teacher", "A_all", "alpha_values", "steps", "S", "damping", "noise_var"],
    "outputs": ["w_hat: Tensor", "x_hat: Tensor"],
    "compute_cost": "O(steps × N1 × N2 × M)",
    "gpu_friendly": True,
    "hyperparameters": ["damping=0.5", "noise_var=1e-6", "steps=1000"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/algorithms/bigamp/core.py`

---

*最后更新：2025年12月*
