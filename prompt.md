**角色：** 统计物理推断算法（G-AMP）与数值计算专家

**任务：**
请基于我现有的**标准高斯教师（Standard Non-orthogonal Teacher）**代码框架，修改程序以实现**“随机扩频/无序模型（Random Spreading / Disordered Model）”**。目的是引入淬火随机性 $F$ 来消除稠密极限下的有限尺寸回路效应。

**物理模型变更：**
*   **原模型：** 观测值 $Y_{ij} = \sum_{\mu=1}^M W_{i\mu} X_{j\mu}$ （隐含 $F=1$）。
*   **目标模型：** 观测值变更为加权内积：
    $$ Y_{ij} = \frac{1}{\sqrt{M}} \sum_{\mu=1}^M F_{ij,\mu} W_{i\mu} X_{j\mu} $$
    其中 $F_{ij,\mu} \sim \mathcal{N}(0, 1)$ 是针对每条观测边 $(i,j)$ 固定的随机系数。

**具体代码修改要求（按模块）：**

**1. Teacher 模块（数据生成）：**
*   在 `generate_data` 函数中，保持 $W, X$ 的生成逻辑不变（标准高斯分布）。
*   **新增逻辑：** 计算 $Y$ 时，不要直接做矩阵乘法。
    *   对于被 Mask 选中的每个观测点 $(i, j)$，生成对应的随机向量 $\vec{F}_{ij}$。
    *   执行逐元素乘法求和：$Y_{ij} = \text{sum}(F_{ij} \odot W_i \odot X_j) / \sqrt{M}$。
*   **内存优化关键：** 请勿在内存中存储完整的 $F$ 张量（形状为 $N_{obs} \times M$）。请实现一个**即时生成（On-the-fly）**机制：利用 `(global_seed, i, j)` 作为哈希种子，在计算时动态生成 $F_{ij}$。

**2. G-AMP 模块（核心算法）：**
你需要重写所有涉及矩阵乘法（消息传递）的部分，将其改为**带 $F$ 权重的逐元素聚合**。
*   **前向传播（估计量 $\hat{Z}$ 和方差 $V$）：**
    *   均值更新：$\hat{Z}_{ij} = \frac{1}{\sqrt{M}} \sum_\mu F_{ij,\mu} \hat{w}_{i\mu} \hat{x}_{j\mu}$
    *   方差更新：$V_{ij}^p = \frac{1}{M} \sum_\mu F_{ij,\mu}^2 (\dots)$。（注：为了性能，你可以将 $F^2$ 近似为期望值 1，或者实现精确的 $F^2$ 加权，请自行权衡）。
*   **反向传播（残差回传）：**
    *   在计算对 $W$ 的更新场（如 `r_W` 或 `Theta_W`）时，原逻辑是 $S \cdot \hat{X}$，现在必须改为：
        $$ \text{Field}_{W, i\mu} = \frac{1}{\sqrt{M}} \sum_{j \in \partial i} F_{ij,\mu} \cdot S_{ij} \cdot \hat{x}_{j\mu} $$
    *   对 $X$ 的更新同理。确保 Onsager 反作用项的系数也适配新的 $\frac{1}{M}$ 缩放。

**3. Evaluation 模块（评估指标）：**
*   修改计算 **$Q_Y$（预测重叠度/MSE）** 的函数。
*   计算学生模型的预测值 $\hat{Y}_{pred}$ 时，必须使用与 Teacher **完全相同**的 $F$ 生成逻辑（相同的 Seed），否则预测值将无法与真实 $Y$ 对齐，导致评估结果错误。
*   注：$Q_W$ 和 $Q_X$ 的计算逻辑保持不变。

**实现提示：**
*   请假设使用 PyTorch。
*   尽量使用 `broadcasting` 或 `torch.func.vmap` 等向量化操作来避免 Python 循环，确保 $F$ 的注入不会显著拖慢运行速度。
*   请直接提供修改后的核心代码片段或重构后的类方法。