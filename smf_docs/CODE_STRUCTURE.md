# 代码结构详解

本文档详细记录每个代码模块的理解、优化记录和 AI 关键词。
**语言**: 中文（技术术语用英语）
**不上传**: 此文件仅供本地开发参考

---

## 模块索引

### 设备与工具
- [D1: 设备检测](#d1-设备检测)
- [D2: 进度显示](#d2-进度显示)

### 图生成
- [G1: 随机图生成](#g1-随机图生成)
- [G2: Dinic 最大流](#g2-dinic-最大流)
- [G3: 正则图生成](#g3-正则图生成)
- [G4: 低循环图](#g4-低循环图)

### 教师矩阵
- [T1: 标准教师](#t1-标准教师)
- [T2: 缩放方差教师](#t2-缩放方差教师)
- [T3: 正交教师](#t3-正交教师)

### 评估指标
- [M1: Gram Overlap Cosine](#m1-gram-overlap-cosine)
- [M2: Gram Overlap Normalized](#m2-gram-overlap-normalized)
- [M3: Q_Y 计算](#m3-qy-计算)
- [M4: 泛化误差](#m4-泛化误差)
- [M5: Replica Overlap](#m5-replica-overlap)
- [M6: Q_Y Unobserved](#m6-qy-unobserved)
- [M7: 指标聚合](#m7-指标聚合)

### 算法
- [A1: BiG-AMP 消息传递](#a1-bigamp-消息传递)
- [A2: BiG-AMP 状态管理](#a2-bigamp-状态管理)
- [A3: BiG-AMP 阻尼](#a3-bigamp-阻尼)
- [A4: AGD 梯度更新](#a4-agd-梯度更新)
- [A5: AGD Adam 优化器](#a5-agd-adam-优化器)
- [A6: 学习率调度](#a6-学习率调度)
- [A7: 收敛检测](#a7-收敛检测)

### 输出
- [P1: 颜色常量](#p1-颜色常量)
- [P2: 绘图样式](#p2-绘图样式)
- [P3: 曲线绘制](#p3-曲线绘制)
- [P4: 对比图](#p4-对比图)
- [S1: JSON 序列化](#s1-json-序列化)
- [S2: 结果扫描](#s2-结果扫描)

### 配置
- [C1: 配置数据类](#c1-配置数据类)
- [C2: 配置验证](#c2-配置验证)
- [C3: YAML 序列化](#c3-yaml-序列化)

---

## 程序分析记录

### bigamp/train.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
BiG-AMP 优化版训练程序，支持三种内存模式（parallel/optimized/extreme）

#### 代码结构
```
1-28:    文档字符串和模块说明
29-62:   导入和默认参数
49-62:   设备设置 (D1)
64-98:   DeviceInfo 数据类和设备信息获取
100-131: 工具函数 (set_seed, create_teacher, sample_mask)
133-172: 评估指标 (M1: gram_overlap_cosine, M2: gram_overlap_zero_to_one)
174-255: 内存管理函数
257-314: BiG-AMP 并行训练 (A1)
316-387: BiG-AMP 单 alpha 训练
389-520: 评估函数 (evaluate_batch, evaluate_single, compute_replica_overlap)
522-620: 主训练函数 (run_parallel_mode, run_sequential_mode)
622-753: 可视化函数 (plot_results, plot_replica_comparison)
755-841: main 函数和 CLI
```

#### 可切分点识别
| ID | 行号 | 切分点 | 说明 |
|----|------|--------|------|
| D1 | 49-62 | 设备设置 | DEVICE, USE_BF16, COMPUTE_DTYPE |
| D2 | 279, 342 | 进度显示 | tqdm 使用 |
| G1 | 119-131 | 随机图生成 | sample_mask 函数 |
| T1 | 110-116 | 标准教师 | create_teacher 函数 |
| M1 | 137-154 | Gram Overlap Cosine | gram_overlap_cosine 函数 |
| M2 | 156-172 | Gram Overlap Normalized | gram_overlap_zero_to_one 函数 |
| M5 | 492-520 | Replica Overlap | compute_replica_overlap 函数 |
| A1 | 261-314 | BiG-AMP 并行训练 | train_bigamp_parallel 函数 |
| A2 | 320-387 | BiG-AMP 单 alpha 训练 | train_bigamp_single 函数 |
| P3 | 626-699 | 曲线绘制 | plot_results 函数 |
| S1 | 801-818 | JSON 序列化 | 结果保存 |

#### 优化空间分析

**已识别的直接优化（Step 2 完成）：**

1. **代码统一 - 图生成模块**
   - 问题：train.py 使用简化版 `sample_mask`，缺少 bi-regular graph 支持
   - 优化：添加 `USE_BIREGULAR_GRAPH` 选项（默认 False）
   - 优化：添加 `RESAMPLE_MASK_EACH_TRIAL` 选项（默认 True）
   - 优化：统一使用完整版 `sample_pairs_biregular_exact` 和 Dinic 算法

2. **Bug 修复 - train_bigamp_single**
   - 问题：`w_var` 和 `x_var` 未转换为 `storage_dtype`（极端模式下可能导致精度问题）
   - 修复：添加 `.to(storage_dtype)` 转换

3. **代码简洁 - select_memory_mode**
   - 问题：打印信息过多
   - 保持：便于调试，不修改

**保持不变的功能差异：**
- `use_fp16` 参数（train.py 独有，用于极端内存模式）
- `compute_replica_overlap`（train.py 独有，用于 replica 分析）
- 这些是功能性差异，其他程序如需要可从 train.py 复制

#### AI 关键词
*待完成*

---

## D1: 设备检测

### 目的
自动检测可用的计算设备（CUDA/MPS/CPU），配置精度设置

### 代码位置
- `bigamp/train.py`: 49-62
- `bigamp/compare_sizes.py`: TBD
- `agd/train_parallel.py`: TBD

### 详细理解
1. 检测 CUDA GPU 可用性
2. 检测 MPS (Apple Silicon) 可用性
3. 设置 BF16 混合精度（仅 CUDA）
4. 启用 TF32 加速（仅 CUDA）

### 当前实现（bigamp/train.py）
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
USE_BF16 = DEVICE.type == 'cuda'
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### 标准版本
```python
# =============================================================================
# Device Setup
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
USE_BF16 = DEVICE.type == 'cuda'
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
STORAGE_DTYPE = torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### 优化记录
- 添加 MPS 支持（Apple Silicon）
- 添加 STORAGE_DTYPE 常量（部分程序需要）

### AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Auto-detect computing device and configure precision settings",
    "when_to_use_en": "At program startup to initialize GPU/CPU configuration",
    "tags_en": ["device", "cuda", "mps", "bf16", "tf32", "precision", "gpu"],

    # 中文
    "purpose_zh": "自动检测计算设备并配置精度设置",
    "when_to_use_zh": "程序启动时初始化 GPU/CPU 配置",
    "tags_zh": ["设备", "CUDA", "MPS", "混合精度", "GPU", "精度设置"],

    # 日文
    "purpose_ja": "計算デバイスを自動検出し、精度設定を構成",
    "when_to_use_ja": "プログラム起動時にGPU/CPU設定を初期化",
    "tags_ja": ["デバイス", "CUDA", "MPS", "混合精度", "GPU"],
}
```

### SMF 对应模块
`sparse_matrix_factorization/core/device.py`

---

## M1: Gram Overlap Cosine

### 目的
通过 Gram 矩阵的余弦相似度测量学生矩阵与教师矩阵的相似性

### 代码位置
- `bigamp/train.py`: 137-154

### 详细理解
计算公式：
```
如果 use_left=True:  G_A = A @ A^T,  G_B = B @ B^T
如果 use_left=False: G_A = A^T @ A,  G_B = B^T @ B

overlap = trace(G_A * G_B) / (||G_A||_F * ||G_B||_F)
```

### 当前实现
```python
@torch.no_grad()
def gram_overlap_cosine(A, B, use_left=True):
    """Compute Gram matrix overlap using cosine similarity"""
    if use_left:
        G_A = A @ A.T
        G_B = B @ B.T
    else:
        G_A = A.T @ A
        G_B = B.T @ B

    G_A_flat = G_A.flatten()
    G_B_flat = G_B.flatten()

    dot = (G_A_flat * G_B_flat).sum()
    norm_A = G_A_flat.norm()
    norm_B = G_B_flat.norm()

    return float(dot / (norm_A * norm_B + 1e-12))
```

### 优化记录
- 无需优化，实现已经简洁高效

### AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Measure similarity between student and teacher via Gram matrix cosine",
    "when_to_use_en": "Standard overlap metric, works for any W/X pair",
    "limitations_en": "Sensitive to scale, use normalized version for [0,1] range",
    "inputs": ["A: (N1, M) or (M, N2)", "B: (N1, M) or (M, N2)", "use_left: bool"],
    "outputs": ["overlap: float in [-1, 1]"],
    "formula": "trace(G_A * G_B) / (||G_A||_F * ||G_B||_F)",
    "tags_en": ["overlap", "gram", "cosine", "W", "X", "evaluation", "metric"],

    # 中文
    "purpose_zh": "通过 Gram 矩阵余弦相似度测量学生与教师的相似性",
    "when_to_use_zh": "标准重叠度量，适用于任意 W/X 矩阵对",
    "limitations_zh": "对尺度敏感，需要 [0,1] 范围时使用归一化版本",
    "tags_zh": ["重叠度", "格拉姆矩阵", "余弦相似度", "W矩阵", "X矩阵", "评估"],

    # 日文
    "purpose_ja": "Gram行列のコサイン類似度で学生と教師の類似性を測定",
    "when_to_use_ja": "標準的なオーバーラップ指標、任意のW/Xペアに適用可能",
    "tags_ja": ["オーバーラップ", "グラム行列", "コサイン", "評価"],
}
```

### SMF 对应模块
`sparse_matrix_factorization/modules/metrics/gram_cosine.py`

---

## M2: Gram Overlap Normalized

### 目的
归一化的 Gram 重叠度，输出范围 [0, 1]，带基线校正

### 代码位置
- `bigamp/train.py`: 156-172

### 详细理解
使用基线校正，使得随机初始化时 Q' ≈ 0，完美匹配时 Q' = 1

计算公式：
```
q = gram_overlap_cosine(A, B, use_left)
b = m / (m + n + 1)  # 随机矩阵的期望余弦值
q' = (q - b) / (1 - b)
return clamp(q', 0, 1)
```

### 当前实现
```python
@torch.no_grad()
def gram_overlap_zero_to_one(A, B, use_left=True):
    q = gram_overlap_cosine(A, B, use_left)
    if use_left:
        n, m = A.shape
    else:
        n, m = A.shape[1], A.shape[0]
    b = m / (m + n + 1)  # baseline
    qc = (q - b) / (1.0 - b + 1e-12)
    return float(max(0.0, min(1.0, qc)))
```

### 优化记录
- 无需优化

### AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Normalized Gram overlap in [0, 1] range with baseline correction",
    "when_to_use_en": "When you need overlap metric normalized to [0, 1]",
    "inputs": ["A: (N1, M) or (M, N2)", "B: (N1, M) or (M, N2)", "use_left: bool"],
    "outputs": ["overlap: float in [0, 1]"],
    "formula": "(q - baseline) / (1 - baseline), baseline = m/(m+n+1)",
    "tags_en": ["overlap", "gram", "normalized", "baseline", "Q_W_prime", "Q_X_prime"],

    # 中文
    "purpose_zh": "归一化 Gram 重叠度，输出范围 [0, 1]，带基线校正",
    "when_to_use_zh": "需要 [0, 1] 范围的归一化重叠度指标时",
    "tags_zh": ["重叠度", "归一化", "基线校正", "Q_W'", "Q_X'"],

    # 日文
    "purpose_ja": "ベースライン補正付きの正規化Gramオーバーラップ [0, 1]",
    "when_to_use_ja": "[0, 1]範囲の正規化オーバーラップが必要な場合",
    "tags_ja": ["オーバーラップ", "正規化", "ベースライン補正"],
}
```

### SMF 对应模块
`sparse_matrix_factorization/modules/metrics/gram_normalized.py`

---

---

### bigamp/compare_sizes.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
有限尺寸效应分析工具，比较不同 N, M 配置下的相转移曲线

#### 主要功能
- 多配置同时运行（不同 N, M 组合）
- 生成尺寸依赖性对比图
- 验证 finite-size scaling 理论

#### AI 关键词
```python
ai_metadata = {
    "purpose_en": "Analyze finite size effects by comparing phase transitions across different N, M",
    "when_to_use_en": "Study size dependence, verify scaling theory",
    "tags_en": ["finite size", "scaling", "comparison", "N/M ratio", "phase transition"],
    "purpose_zh": "通过比较不同 N, M 的相转移分析有限尺寸效应",
    "tags_zh": ["有限尺寸", "尺寸效应", "对比", "相转移"],
}
```

---

### bigamp/orthogonal_teacher.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
正交教师矩阵实验，通过 QR 分解生成正交 W* 和 X*

#### 物理图景
标准教师矩阵（随机高斯）存在有限尺寸涨落。正交教师消除这些涨落：
```
W*: N1×M，列正交 (W*^T @ W* = I_M)
X*: M×N2，行正交 (X* @ X*^T = I_M)
```
这使得实验结果更接近理论预测（N→∞ 极限）

#### 优化记录
- 添加 STORAGE_DTYPE 常量（与标准版统一）

#### AI 关键词
```python
ai_metadata = {
    "purpose_en": "Compare standard vs orthogonal teacher to verify finite-size effects",
    "when_to_use_en": "Verify theory, eliminate finite-size fluctuations",
    "tags_en": ["orthogonal", "QR", "finite size", "teacher", "theory verification"],
    "purpose_zh": "对比标准教师与正交教师，验证有限尺寸效应",
    "tags_zh": ["正交", "QR分解", "有限尺寸", "教师矩阵", "理论验证"],
}
```

---

### bigamp/low_loop_graph.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
低循环（C4-free）图实验，研究图结构对 BiG-AMP 收敛性的影响

#### 物理图景
AMP 算法假设图是树状结构（无短循环）。标准随机图包含许多 4-loop：
```
随机图:     ○ - ○
           / × /   ← 4-loop (C4) 普遍存在
          ○ - ○

C4-free:   ○ - ○
           |   |   ← 无 C4，更接近树结构
           ○   ○
```
C4-free 图使 AMP 的均场近似更准确

#### 核心算法
MCMC edge switching（马尔科夫链蒙特卡洛边交换）消除 4-loop

#### 优化记录
- 添加 STORAGE_DTYPE 常量

#### AI 关键词
```python
ai_metadata = {
    "purpose_en": "Study short loop effects on BiG-AMP via C4-free graphs",
    "when_to_use_en": "Investigate graph structure impact on AMP convergence",
    "tags_en": ["low loop", "C4-free", "girth", "MCMC", "edge switching", "graph structure"],
    "purpose_zh": "通过 C4-free 图研究短循环对 BiG-AMP 的影响",
    "tags_zh": ["低循环", "C4-free", "图结构", "MCMC", "边交换"],
}
```

---

### bigamp/replica_overlap.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
Replica overlap 分析，研究多个独立解之间的重叠度

#### 物理图景
从不同随机初始化运行算法得到多个"replica"解：
```
Replica 1: (W₁, X₁) ─┐
Replica 2: (W₂, X₂) ─┼─→ 计算两两之间的 Gram overlap
Replica 3: (W₃, X₃) ─┘

如果所有 replica 收敛到相同解: Q_inter ≈ 1
如果存在多个局部最优: Q_inter < Q_self
```
高 replica overlap 表明解的唯一性；低 replica overlap 暗示能量景观多峰

#### 优化记录
- 添加 STORAGE_DTYPE 常量

#### AI 关键词
```python
ai_metadata = {
    "purpose_en": "Analyze solution uniqueness via pairwise replica overlap",
    "when_to_use_en": "Verify solution uniqueness, study energy landscape",
    "tags_en": ["replica", "overlap", "uniqueness", "multimodal", "energy landscape"],
    "purpose_zh": "通过 replica 成对重叠分析解的唯一性",
    "tags_zh": ["副本", "重叠度", "唯一性", "多峰", "能量景观"],
}
```

---

### agd/train_sequential.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
AGD 顺序 α 扫描训练，支持早停

#### 物理图景
交替梯度下降的直觉：在 W-X 空间的山谷中寻找最低点
```
固定 X → 沿 W 方向下降
固定 W → 沿 X 方向下降
重复直到收敛
```
每步都单调下降（学习率合适时），比同时更新更稳定

#### 设计特点
- 使用 COMPUTE_DTYPE 模式（无需 STORAGE_DTYPE）
- 支持早停机制
- 逐个 alpha 处理，内存友好

#### AI 关键词
```python
ai_metadata = {
    "purpose_en": "AGD training with sequential alpha processing and early stopping support",
    "when_to_use_en": "Small-medium matrices, debugging, need early stop, large matrices",
    "tags_en": ["AGD", "alternating", "gradient descent", "sequential", "early stop"],
    "purpose_zh": "AGD 顺序 α 训练，支持早停",
    "tags_zh": ["AGD", "交替梯度下降", "顺序训练", "早停"],
}
```

---

### agd/train_parallel.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
AGD 并行 α 训练，所有 alpha 同时处理

#### 物理图景
```
Sequential:  Track 1 → Track 2 → Track 3 → ...  总时间: 21×T
Parallel:    Track 1 ─┐
             Track 2 ─┼─→ 同时处理  总时间: ~T
             Track 3 ─┘
```
减少 Python 循环开销 ~21 倍

#### 设计特点
- 使用 COMPUTE_DTYPE 模式（无需 STORAGE_DTYPE）
- BF16 混合精度（CUDA）
- 无早停（所有 α 训练相同步数）
- 显存使用增加 ~A 倍

#### AI 关键词
```python
ai_metadata = {
    "purpose_en": "AGD parallel alpha training for maximum GPU utilization",
    "when_to_use_en": "Small matrices, ample GPU memory, speed priority",
    "tags_en": ["AGD", "parallel", "batch", "GPU utilization", "BF16", "TF32"],
    "purpose_zh": "AGD 并行 α 训练，最大化 GPU 利用率",
    "tags_zh": ["AGD", "并行训练", "批量", "GPU 利用率"],
}
```

---

### analysis/slope_ratio.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
分析相转移前 Q_Y 线性区域斜率与 N/M 的关系

#### 物理图景
```
在低 α 区域（信息不足），学生无法恢复教师：
Q_Y_baseline ≈ E[cosine(random, teacher)] ≈ 2 × α × M/N

斜率 k = d(Q_Y)/d(α) ≈ 2 × M/N
```
验证 slope vs M/N 的线性关系

#### AI 关键词
```python
ai_metadata = {
    "purpose_en": "Analyze pre-transition slope and its relationship with N/M ratio",
    "when_to_use_en": "Verify finite-size effect theory, study Q_Y linear region",
    "tags_en": ["slope", "analysis", "N/M ratio", "finite size", "linear region"],
    "purpose_zh": "分析相转移前斜率与 N/M 的关系",
    "tags_zh": ["斜率", "分析", "N/M 比值", "有限尺寸"],
}
```

---

### analysis/compare_algorithms.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
AGD vs BiG-AMP 算法对比工具

#### 物理图景
```
假设: 在低 α 区域观察到 Q_Y 的线性偏移
问题: 这是算法 bug，还是真实的物理现象？

验证方法:
    如果 Q_Y_AGD ≈ Q_Y_BiGAMP → 物理现象
    如果 Q_Y_AGD ≠ Q_Y_BiGAMP → 可能是算法问题
```

#### AI 关键词
```python
ai_metadata = {
    "purpose_en": "Verify physical phenomena by comparing AGD vs BiG-AMP results",
    "when_to_use_en": "Verify observations are not algorithm-specific artifacts",
    "tags_en": ["comparison", "AGD", "BiG-AMP", "verification", "algorithm"],
    "purpose_zh": "通过对比 AGD 和 BiG-AMP 结果验证物理现象",
    "tags_zh": ["对比", "AGD", "BiG-AMP", "验证", "算法"],
}
```

---

### analysis/degree_distribution.py

**处理状态**: ✅ 已完成
**完成日期**: 2024年12月

#### 概述
随机图度分布分析工具，验证图生成的统计性质

#### 物理图景
```
观测 mask A 可以看作一个二分图:
    左节点度分布: d_i = Σ_j A[i,j] = 行 i 的观测数
    理论期望: E[d] = C / N1 = α × M
```
验证图生成代码的正确性

#### AI 关键词
```python
ai_metadata = {
    "purpose_en": "Verify graph generation correctness via degree distribution analysis",
    "when_to_use_en": "Debug graph generation, verify statistical properties",
    "tags_en": ["degree", "distribution", "graph", "verification", "statistics"],
    "purpose_zh": "通过度分布分析验证图生成的正确性",
    "tags_zh": ["度分布", "图", "验证", "统计"],
}
```

---

## 处理进度总结

| 程序 | Step1 | Step2 | Step3 | Step4 | Step5 | Step6 | Step7 | 状态 |
|------|-------|-------|-------|-------|-------|-------|-------|------|
| bigamp/train.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| bigamp/compare_sizes.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| bigamp/orthogonal_teacher.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| bigamp/low_loop_graph.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| bigamp/replica_overlap.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| agd/train_sequential.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| agd/train_parallel.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| analysis/slope_ratio.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| analysis/compare_algorithms.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| analysis/degree_distribution.py | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |

**所有程序处理完成！**

---

*最后更新：2024年12月*
