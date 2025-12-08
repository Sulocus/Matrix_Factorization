# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 全局记忆

**语言偏好设置：**
- 默认与用户沟通语言：**中文**
- CLAUDE.md 文件语言：**中文**
- 代码注释、变量名、commit message 等：可以使用英文（不强制中文）

---

## 项目概述

这是一个基于 PyTorch 的 Teacher-Student 掩码矩阵分解研究实现，支持 GPU 加速。代码模拟稀疏矩阵分解的学习过程，其中学生模型尝试从部分观测中恢复教师模型的矩阵分解 (Y = W × X)。

实现了两种主要算法：
1. **交替梯度下降 (AGD)** - 迭代优化方法
2. **BiG-AMP** - 双线性广义近似消息传递（收敛更快）

## 运行代码

### 主入口（推荐）

```bash
python smf/run.py
```

交互式菜单驱动界面，支持：
- 使用模块组合运行新实验
- 浏览/搜索历史结果
- 使用预设配置快速运行

### 旧版脚本（在 `_legacy/` 目录）

直接执行脚本请使用 `_legacy/` 目录中的文件：

#### BiG-AMP 训练（大矩阵推荐）

```bash
python _legacy/bigamp_optimized.py                          # 默认 400x400
python _legacy/bigamp_optimized.py --n1 20000 --m 141       # 大矩阵
python _legacy/bigamp_optimized.py --memory-mode extreme    # 内存受限模式
python _legacy/bigamp_multi_size.py                         # 多尺寸对比
python _legacy/bigamp_no4loop.py --compare                  # 4-loop 最小化 vs 随机
```

#### AGD 训练

```bash
python _legacy/Main.py              # 顺序 alpha 训练
python _legacy/Main_multi_alpha.py  # 并行 alpha 训练
```

#### 精确相变分析

```bash
python -m precise_phase_analysis.run_mode2              # 梯度自适应采样
python -m precise_phase_analysis.run_mode3 --N1 2000 --M 100  # 精确 α_c 确定
```

### 关键配置参数

所有参数通过每个脚本顶部的全局变量配置：

**矩阵维度：**
- `N1`, `N2`：矩阵维度（教师矩阵为 N1×N2）
- `M`：隐维度（分解的秩）

**训练配置：**
- `ALPHA_TILDE_START`, `ALPHA_TILDE_STOP`, `ALPHA_TILDE_STEP`：alpha 值范围（稀疏度级别）
- `EPOCHS_PER_ALPHA`：每个 alpha 值的训练步数
- `LEARNING_RATE`：学生模型学习率
- `SAMPLES_PER_ALPHA`：每个 alpha 的独立试验次数

**图生成：**
- `USE_BIREGULAR_GRAPH`：选择图生成方法
  - `True`：使用 Dinic 最大流算法生成双正则图（均匀度分布）
  - `False`：纯随机 GPU 生成（更快，支持任意 N1≠N2）
- `RESAMPLE_MASK_EACH_TRIAL`：是否为每次试验生成不同的掩码

**早停（仅 Main.py）：**
- `USE_EARLY_STOP`：启用/禁用早停
- `TARGET_LOSS_THRESHOLD`：停止的绝对损失阈值
- `RELATIVE_CHANGE_THRESHOLD`：收敛的相对变化阈值
- `EARLY_STOP_CHECK_INTERVAL`：收敛检查间隔步数
- `EARLY_STOP_PATIENCE`：无变化后停止前的检查次数

**设备与性能：**
- `DEVICE`：自动选择 MPS (Apple Silicon)、CUDA 或 CPU
- `USE_BF16`：启用 BF16 混合精度（仅 CUDA）
- TF32 加速在 CUDA 设备上自动启用

## 代码架构

### 核心算法流程

1. **教师模型创建** (`create_teacher` / `create_teacher_dense`)：
   - 生成真值 W_true (N1×M) 和 X_true (M×N2)
   - 始终使用 FP32 确保精度
   - 模块化版本：`smf/modules/teachers/`

2. **图/掩码生成**（两种方法可用）：
   - **双正则**：使用 Dinic 最大流算法实现均匀度分布
   - **随机**：使用 `torch.randperm` 的纯 GPU 随机采样（更快）
   - 模块化版本：`smf/modules/graphs/`

3. **学生模型训练**：
   - **BiG-AMP**（推荐）：`_legacy/bigamp_optimized.py` 或 `smf/modules/algorithms/bigamp.py`
   - **AGD 顺序**：`_legacy/Main.py`
   - **AGD 并行**：`_legacy/Main_multi_alpha.py`

4. **评估**：
   - Gram 重叠指标 (Q_W, Q_X) - 衡量学生恢复教师子空间的程度
   - Y 重叠 (Q_Y) - 全矩阵的旋转不变度量
   - **Q_Y_unobserved** - 仅在未观测位置的重叠（泛化能力指标）
   - 泛化误差 - 整个矩阵的 MSE
   - m² 指标用于理论验证

### 性能优化

代码实现了多项 GPU 优化：

1. **核融合**（仅 AGD）：
   - 每训练步的 GPU 核启动从 ~18 减少到 ~6
   - 兼容 MPS 和 CUDA
   - 可在 PyTorch 2.0+ 上使用 `torch.compile` 编译

2. **并行 Alpha 训练** (`_legacy/Main_multi_alpha.py`)：
   - 将所有 alpha 值批处理：形状变为 (num_alphas, S, N1, M/N2)
   - CPU 开销减少 N_alpha 倍

3. **混合精度**（仅 CUDA）：
   - 前向/反向传播使用 BF16（2倍加速，50% 内存减少）
   - 参数存储和最终评估使用 FP32

4. **TF32 加速**（仅 CUDA）：
   - 矩阵乘法自动启用
   - matmul 操作约 8 倍加速，精度影响极小

### 训练循环实现

核心训练实现交替梯度下降：
- **步骤 1**：使用当前 (W, X) 计算 W 的梯度，更新 W
- **步骤 2**：使用更新后的 W 重新计算，计算 X 的梯度，更新 X

关键实现细节：
- 教师参数始终为 FP32 以确保真值准确
- 学生参数以 FP32 存储，训练时转换为 COMPUTE_DTYPE
- 使用 `torch.autocast` 支持混合精度
- 收集最终结果前进行设备同步

### 输出格式

结果保存到 `Result/{N1}_{N2}_{M}/`，文件名格式：
```
{GraphType}_{Resample}_{EarlyStop}_{KeyParam}_batch{S}.png
```

其中：
- GraphType：`BiReg` 或 `Rand`
- Resample：`Resample` 或 `NoResample`
- EarlyStop：`ET`（启用）或 `EF`（禁用）
- KeyParam：如启用早停则为 `Loss{threshold}`，否则为 `Epoch{count}`

绘图包含：
- Q_Y 指标（旋转不变重叠）
- Q_W' 和 Q_X' 指标（归一化到 0-1 的 Gram 重叠）
- 配置参数表

## 文件组织

**仓库结构：**

```
Sparse-Matrix/
├── smf/                                 # 新模块化框架（主要）
│   ├── run.py                           # 交互式入口
│   ├── runner.py                        # 实验运行器
│   ├── core/                            # 核心工具
│   │   ├── device.py                    # 设备检测
│   │   ├── config.py                    # 配置管理
│   │   ├── progress.py                  # 进度显示
│   │   ├── opener.py                    # 跨平台文件打开
│   │   └── time_estimator.py            # 时间估计
│   ├── modules/                         # 可插拔模块
│   │   ├── algorithms/bigamp.py         # BiG-AMP 算法
│   │   ├── graphs/{random,uniform}.py   # 图生成器
│   │   ├── teachers/{standard,scaled_variance}.py  # 教师模型
│   │   ├── metrics/overlap.py           # 重叠指标
│   │   └── outputs/{plotting,storage}.py  # 输出处理
│   ├── ui/                              # 用户界面
│   │   ├── menu.py, wizard.py           # 交互式菜单
│   │   └── browser.py                   # 结果浏览器
│   └── experiments/                     # 实验模板
│       └── large_matrix_sweep.py        # 大矩阵实验
│
├── _legacy/                             # 原始脚本（仍可用）
│   ├── bigamp_optimized.py              # BiG-AMP 生产版本
│   ├── bigamp_multi_size.py             # 多尺寸实验
│   ├── bigamp_no4loop.py                # 4-loop 最小化图
│   ├── Main.py, Main_multi_alpha.py     # AGD 训练
│   └── analyze_*.py, Bethe.py           # 分析工具
│
├── precise_phase_analysis/              # 相变分析模块
│   ├── core/                            # 核心分析器
│   └── run_mode{2,3}.py                 # 入口点
│
├── Result/                              # 单尺寸结果
├── ResultNo4/                           # 4-loop 最小化结果
├── Result_compareNM/                    # 多尺寸对比
└── optimization_tests/                  # 实验性代码
```

**文件组织规则：**

1. **新实验** → 使用 `smf/run.py` 或在 `smf/modules/` 添加模块
2. **旧版脚本** → `_legacy/` 目录（仍可用）
3. **结果分离**：
   - 新结果：`smf/results/`（已索引）
   - 旧结果：`Result/`、`ResultNo4/`、`Result_compareNM/`

## 算法配置

### BiG-AMP 参数 (bigamp_optimized.py)

```python
DAMPING = 0.5        # 消息阻尼因子（稳定性 vs 速度）
NOISE_VAR = 1e-10    # 假设噪声方差
MAX_STEPS = 5000     # 最大迭代步数
```

BiG-AMP 在 ~200-5000 步内收敛，而 AGD 需要 20k+ epochs。

### AGD + Adam + 余弦退火学习率

AGD 优化中，手动 Adam 保持交替梯度下降：

```python
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
LEARNING_RATE = 1e-2            # 初始学习率
LR_SCHEDULER_ETA_MIN = 1e-6     # 最小学习率
```

**为什么使用手动 Adam？**
- PyTorch 的 `torch.optim.Adam` 期望在一步中更新所有参数
- 我们的算法需要交替更新（先 W 后 X）
- 手动实现为 W 和 X 维护独立的动量/速度状态

## 开发工作流

修改参数时：
1. 编辑脚本顶部的全局变量
2. 快速测试时减小 `N1`、`N2`、`M` 和 `EPOCHS_PER_ALPHA`
3. 运行脚本验证更改生效

调试性能时：
- 检查是否启用了 `torch.compile`（需要 PyTorch 2.0+）
- 验证 CUDA 上是否使用 BF16（检查控制台输出）
- 确保收集结果前进行设备同步
- MPS 不使用 BF16（使用 FP32）

添加新指标时：
- 在评估阶段（训练循环后）使用 FP32 计算
- 在顺序和并行版本的结果字典中添加
- 相应更新 `display_results` 和 `plot_results` 函数

运行测试时：
- 始终使用 CUDA 提速（避免 CPU 测试）
- **测试时增加 `SAMPLES_PER_ALPHA`**（如 20-50 而非 5）
  - 更大的批次给出更准确的统计
  - 对验证和对比测试至关重要
- 测试脚本放在 `optimization_tests/` 目录
- 使用描述性名称（测试脚本用 `test_*.py`）

## 算法对比

| 算法 | 收敛速度 | 最适用于 | 入口点 |
|------|---------|---------|--------|
| AGD (SGD) | ~20k epochs | 小矩阵，基线 | `Main.py`, `Main_multi_alpha.py` |
| AGD (Adam+LR) | ~2k epochs | 更快收敛的 AGD | `optimization_tests/step1/` |
| **BiG-AMP** | ~200-5000 步 | **大矩阵 (N>1000)** | `bigamp_optimized.py` |

**推荐**：生产实验使用 BiG-AMP（更快，扩展性更好）。

## 相变分析模式

`precise_phase_analysis/` 模块提供三种分析模式：

### 模式 1：基础后处理
分析现有结果以检测相变：
```python
from optimization_tests.phase_transition_analyzer import PhaseTransitionAnalyzer
analyzer = PhaseTransitionAnalyzer(alphas, metrics)
phase = analyzer.detect_phase_transition_enhanced()
```

### 模式 2：梯度自适应采样
基于梯度轮廓的智能 alpha 重分布：
```bash
python -m precise_phase_analysis.run_mode2
```
- 总点数与均匀采样相同
- 相变区域分辨率更高
- 使用 `GradientAdaptiveSampler` 类

### 模式 3：精确相变分析
在热力学极限下确定 α_c：
```bash
python -m precise_phase_analysis.run_mode3 --N1 2000 --M 100
```
- 随步数增加逐步细化
- 外推至 steps → ∞
- 使用 `PrecisePhaseAnalyzer` 类

## 关键设计决策

以下决策经过充分讨论，未经仔细考虑**不得更改**：

### 1. 并行 Alpha 训练架构
**设计**：将所有 alpha 批处理：形状为 `(num_alphas, S, N1, M)`

**原因**：减少 Python 循环开销 N_alpha 倍（如 31 个 alpha → 减少 31 倍迭代）

### 2. AGD 的手动 Adam
**原因**：PyTorch 的 `torch.optim.Adam` 同时更新所有参数，但 AGD 需要**交替更新**（W 和 X 分别更新）。

**实现**：为 W 和 X 维护独立的动量/速度状态。

### 3. 分析器不训练
**设计**：相变分析器接收训练结果，不包含训练逻辑

**原因**：关注点分离 - 分析逻辑应独立于训练实现

### 4. 结果目录分离
- **生产结果**：`Result/` 和 `Result_compareNM/`
- **测试结果**：`optimization_tests/Result/`

**原因**：保留基线数据用于验证，防止数据损坏

---

## 模块化架构

代码库于 2024 年 12 月重构为模块化插件架构：

- **插件系统**：通过装饰器注册模块（`@register_algorithm`、`@register_graph` 等）
- **模块类别**：algorithms、graphs、teachers、metrics、outputs
- **添加新模块**：创建继承基类的类，添加装饰器，在 `smf/modules/__init__.py` 中导入

添加新模块示例：
```python
# smf/modules/graphs/my_new_graph.py
from ..registry import register_graph

@register_graph(key="my_graph", name="自定义图", description="自定义图结构")
class MyGraphGenerator(GraphGeneratorBase):
    def generate(self, N1, N2, alpha, **kwargs):
        # 实现
        ...
```

**重构中新增的指标：**
- **Q_Y_unobserved**：仅在未观测位置计算的重叠（泛化能力指标）
  - 已添加到：`bigamp_optimized.py`、`bigamp_multi_size.py`、`bigamp_no4loop.py`
  - 生成额外对比图：`*_qy_unobserved_*.png`
