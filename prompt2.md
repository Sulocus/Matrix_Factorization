# Onsager 反作用项问题分析

## 问题背景

在 `bigamp_spreading_parallel.py` 中加入 Onsager 反作用项后出现问题。参考的正常工作版本是 `bigamp.py`。

---

## 发现的问题

### 问题 1: `bigamp_step_disjoint_union` 中 Onsager 项实现错误

**位置**: 第 536-541 行

```python
# Onsager correction: Z_hat -= s_prev * V
if prev_s is not None:
    Z_hat = Z_hat - prev_s * V  # ← 问题所在
    Z_hat = Z_hat * alpha_mask_exp.float()
```

**错误原因**:
1. **使用了当前步的 V 而非前一步的 V** - 标准 AMP 理论中 Onsager 项应使用前一迭代的 variance
2. **形式可能不正确** - `prev_s * V` 不是标准的 Onsager correction 形式

**标准形式参考**:
在 AMP 理论中，Onsager correction 的目的是消除消息传递中的"回声"效应，标准形式为：
```
residual = Y - Z_hat - (Onsager term)
```
其中 Onsager term 应该是 `prev_s * (∂Z_hat/∂prev_s)`，具体形式取决于模型结构。

---

### 问题 2: 两个函数实现不一致

| 函数 | Onsager 实现 |
|------|--------------|
| `bigamp_spreading_parallel_step` (第323-446行) | ❌ 没有实现（代码被注释） |
| `bigamp_step_disjoint_union` (第453-611行) | ✅ 有实现（但可能错误） |

这导致：
- `train_sample()` 使用 `bigamp_spreading_parallel_step` → **无 Onsager**
- `train_full_parallel()` 使用 `bigamp_step_disjoint_union` → **有 Onsager**

两种训练方式行为不一致。

---

### 问题 3: 冗余代码

**位置**: 第 547 和 549 行

```python
s_values = s_values * alpha_mask_exp.float()  # 第547行

s_values = s_values * alpha_mask_exp.float()  # 第549行 - 重复！
```

`alpha_mask_exp.float()` 被重复应用了两次。

---

## 对比 bigamp.py

`bigamp.py` 中的简单实现**没有使用 Onsager correction**:

```python
# bigamp.py 第51-52行
residual = (Y - z_hat) * A
s = residual / V
```

这个版本工作正常，说明 Onsager 项的引入方式有问题。

---

## 建议修复方案

### 方案 A: 暂时移除 Onsager correction

最简单的方案，保持和 `bigamp.py` 一致。

```python
# 注释掉第 536-541 行
# if prev_s is not None:
#     Z_hat = Z_hat - prev_s * V
#     Z_hat = Z_hat * alpha_mask_exp.float()
```

### 方案 B: 正确实现 Onsager correction

如果需要 Onsager correction，需要：

1. **保存前一步的 V** (或相关量)
2. **使用正确的形式**，例如：
   ```python
   # 需要根据理论推导确定正确形式
   if prev_s is not None and prev_V is not None:
       onsager_term = prev_s * prev_V / (prev_V + noise_var)
       Z_hat = Z_hat - onsager_term
   ```
3. **保持两个函数一致** - 同时修改 `bigamp_spreading_parallel_step` 和 `bigamp_step_disjoint_union`

### 方案 C: 同时修复冗余代码

删除第 549 行的重复代码：
```python
# 删除这行
s_values = s_values * alpha_mask_exp.float()  # 第549行
```

---

## 需要确认

1. 你想要实现的 Onsager correction 的理论形式是什么？
2. 是否有相关论文或推导可以参考？
3. 是否需要同时保持两种训练方式 (`train_sample` 和 `train_full_parallel`) 的行为一致？

BiG-AMP 算法 Onsager 项失效深度分析
1. 核心诊断：阻尼策略的不一致性 (Damping Inconsistency)
你的代码在引入 Onsager 校正项后计算失效，其根本原因在于变量更新的阻尼（Damping）与残差传递之间存在逻辑断层。
简而言之：
你对权重矩阵 $W$ 和 $X$ 施加了阻尼（只走了一小步），却把基于“完整一步”假设计算出的残差 $s$ 直接传递给了下一轮迭代作为 prev_s。这导致 Onsager 校正项在下一轮中进行了过度的“回溯校正”，引发了数值震荡。
2. 错误机理深度解析
2.1 Onsager 项的作用
在 AMP 类算法中，Onsager 项（$- s^{t-1} V^t$）的作用是消除迭代过程中产生的自相关性（Self-interaction）。它假设当前的估计值 $Z^t$ 包含了来自上一步残差 $s^{t-1}$ 的“回声”，因此需要减去这个回声，使得当前的噪声近似于高斯白噪声。
2.2 阻尼（Damping）的作用
阻尼用于防止算法在复杂的非凸景观中跳跃过大。公式通常为：

$$\theta_{new} = \beta \cdot \theta_{calc} + (1-\beta) \cdot \theta_{old}$$

其中 $\beta$ 是阻尼系数（例如 0.5）。这意味着实际的参数只更新了计算量的一半。
2.3 冲突点：当阻尼遇见 Onsager
在你的代码中发生了以下冲突：
计算阶段：你根据当前的 $W^t, X^t$ 计算出了预测值 $Z^t$ 和方差 $V^t$。
残差计算：你计算了 $s^t_{raw} = (Y - Z^t) / (V^t + \Delta)$。这个 $s^t_{raw}$ 代表了如果没有任何阻尼，系统“想要”迈出的步伐方向和大小。
保存历史：你将未处理的 $s^t_{raw}$ 直接赋值给 prev_s，准备在 $t+1$ 轮使用。
实际更新：在代码最后，你对 $W$ 和 $X$ 进行了阻尼：
$$W^{t+1} = 0.5 W_{new} + 0.5 W^t$$

这意味着系统实际上并没有移动到 $s^t_{raw}$ 所暗示的位置，而是停在了半路上。
后果（在 $t+1$ 轮）：
当进入下一轮时，Onsager 项公式为 $Z^{t+1}_{corr} = Z^{t+1} - s^{t} V^{t+1}$。
这里的 $s^t$ 是未阻尼的（过大的），而 $Z^{t+1}$ 是基于阻尼后的 $W^{t+1}$ 计算的（变化较小的）。
结果：Onsager 项减去了一个“并不存在于当前 $Z$ 中的强回声”，导致严重的过矫正（Over-correction）。
3. 代码层面的具体证据
请看 bigamp_step_disjoint_union 函数中的逻辑漏洞：
# ... [前向传播计算 Z_hat 和 V] ...

# 1. 这里的 Onsager 校正是基于 prev_s 的
if prev_s is not None:
    Z_hat = Z_hat - prev_s * V 

# ... [计算 s_values] ...
s_values = (Y_flat.unsqueeze(0) - Z_hat) / denom

# ... [关键错误点] ...
# 原代码注释掉了 s 的阻尼！
# REMOVED: s-damping
# if prev_s is not None:
#     s_values = damping * s_values + (1 - damping) * prev_s

# ... [函数末尾] ...
# 2. 变量 W 和 X 被阻尼了
W_hat_out = damping * W_hat_new + (1 - damping) * W_hat

# 3. 返回的是未阻尼的 s_values，它将在下一轮变成 prev_s
return W_hat_out, ..., s_values


分析：
$W$ 的状态滞后（因为被 Damping 拖住了）。
$s$ 的状态超前（因为没有 Damping，直接冲到了目标位置）。
下一轮的 Z_hat - prev_s * V 出现了时间步上的错位。
4. 修复方案原理
为了使 Onsager 项正常工作，残差 $s$ 的更新速率必须与变量 $W, X$ 的更新速率保持同步。
如果变量只更新了 50%（damping=0.5），那么传递给 Onsager 项的“历史残差”也必须反映这 50% 的更新。
修正后的数学表达
我们需要维护一个“有效残差” $s_{eff}$：
$$ s_{eff}^t = \beta \cdot s_{raw}^t + (1-\beta) \cdot s_{eff}^{t-1} $$
这样，$s_{eff}^t$ 就真实地反映了系统在 $t$ 时刻的实际状态变化量，与 $W^{t+1}$ 的状态是匹配的。
修正代码块
在计算出 s_values 后，必须执行以下操作：
# 必须恢复这段逻辑
if prev_s is not None:
    # 这里的 damping 必须与 W_hat/X_hat 使用的 damping 数值完全一致
    s_values = damping * s_values + (1 - damping) * prev_s
else:
    # 第一步通常也要阻尼，防止初始残差过大导致第一轮 Onsager 爆炸
    s_values = damping * s_values


5. 总结
本来好好的程序加了这一项就崩了，不是因为 Onsager 项本身写错了，而是因为在存在阻尼（Damping）的 AMP 迭代中，所有状态变量（包括辅助变量 $s$）必须以相同的速率演进。
只要加上对 s_values 的阻尼，Onsager 项就能起到它应有的“稳定剂”作用，而不是变成“破坏者”。

