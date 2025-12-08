**Role:** 你是一位精通统计物理推断（Statistical Inference）和数值计算的算法专家。

**Task:** 我们需要对现有的评估代码（Evaluation Metrics）进行一次关键的修正和升级，以便更准确地验证 G-AMP 算法是否符合论文中的理论预测（State Evolution）。

**Context & Reasoning:**
我们之前使用的 `gram_overlap_cosine` 本质上是在计算余弦相似度（Cosine Similarity）。虽然它在工程上能反映相关性，但在 G-AMP（贝叶斯最优 MMSE 估计）的语境下，它存在一个物理定义上的偏差：
*   MMSE 估计具有**收缩效应（Shrinkage）**，即预测向量的模长 $\|\hat{x}\|$ 会随着信号强度 $m$ 变短。
*   余弦相似度的分母中包含 `norm(pred)`，这会人为地把变短的预测向量拉长，导致算出来的数值虚高（算出的是 $\sqrt{m}$ 或 $m$ 而不是理论上的 $m^2$ 或 $m$）。
*   为了与论文中的序参量 $m$（Magnetization/Overlap）进行严格的物理对齐，我们需要一个基于**线性投影（Linear Projection）**的新指标。

**Action Items:**

请按以下步骤修改代码（主要是在评估模块）：

1.  **重命名旧指标：**
    *   请把原有的 `gram_overlap_cosine` 函数（以及相关的调用逻辑）重命名为 `compute_cosine_similarity`。
    *   保留它作为参考指标，但不再作为物理验证的核心依据。

2.  **实现新的物理重合度函数：**
    *   请创建一个新的通用函数，命名为 `compute_physical_overlap`。
    *   **数学定义：** Projection of Student on Teacher。
        $$ \text{Overlap} = \frac{\langle \text{Pred}, \text{True} \rangle}{\langle \text{True}, \text{True} \rangle} $$
    *   **实现细节：**
        *   输入：两个张量 `pred` 和 `true`。
        *   计算：先将它们展平（flatten），然后计算点积（dot product）。
        *   分母：**只除以 `true` 的点积（模长平方）**。千万不要除以 `pred` 的模长。
        *   绝对值处理：增加一个可选参数 `absolute` (默认 `False`)。
            *   当计算 $W$ 或 $X$ 时（存在符号模糊性），设为 `True`（取绝对值）。
            *   当计算 $Y$ 时（符号已确定），设为 `False`。

3.  **在评估循环中应用新指标：**
    *   **针对 $Y$ (Observation Matrix):**
        *   计算 `overlap_Y = compute_physical_overlap(Y_pred, Y_true, absolute=False)`。
        *   **关键注释：** 请在代码注释中注明：对于 $p=2$ 矩阵分解，理论上这个值对应 $m^2$。如果想和论文的 $m$ 曲线对比，应该对它**开根号**。
    *   **针对 $W$ 和 $X$ (Factor Matrices):**
        *   计算 `overlap_W = compute_physical_overlap(W_pred, W_true, absolute=True)`。
        *   **关键注释：** 请注明：除非先解决了旋转对称性（如 Procrustes 对齐），否则这个值在 $p=2$ 时可能很小，不具备直接参考价值，仅作调试用。

4.  **输出与日志：**
    *   请确保在 JSON 结果和日志中，清晰地区分 `cosine_sim_Y` 和 `physical_overlap_Y`。我们后续画图主要用后者。

请基于以上逻辑，帮我修改评估部分的代码。代码风格请保持现有的模块化结构。
