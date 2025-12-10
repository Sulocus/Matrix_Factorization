# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 必须遵守的规则

### 语言规范
- **沟通语言**：中文
- **代码/注释/commit**：英文

### Wang/ 程序规范（重要）
1. **程序独立性**：每个程序完整独立，不使用共享模块导入
2. **代码一致性**：不同程序中相同代码块必须**完全一致**（复制粘贴级别）
3. **文档语言**：Wang/README.md 用日语，技术术语用英语

### SMF 文档同步规则（重要）
- **更新 smf/ 模块时，必须同步更新 `smf_docs/` 中对应的文档**
- 新功能不明白时，先查阅 `smf_docs/README.md`

---

## 常用命令

```bash
# 安装 SMF 框架（开发模式）
pip install -e .

# 测试
pytest tests/                           # 运行所有测试
pytest tests/test_file.py -v            # 运行单个测试，详细输出

# SMF CLI
smf                # 交互模式（自然语言配置）
smf run            # 实验向导
smf run --bg       # 后台运行
smf resume         # 检查点恢复
smf log            # 查看日志
smf log -f         # 实时跟踪日志
smf vis            # 结果浏览器
smf test           # 快速测试

# 生产训练 (Wang/) - 独立程序，不依赖 smf
python Wang/bigamp/train.py           # BiG-AMP（推荐）
python Wang/agd/train_parallel.py     # AGD 并行版
```

---

## 项目架构

### 双轨系统

| 目录 | Git | 用途 |
|------|-----|------|
| `Wang/` | main 分支 | 生产代码，与日本同学共享 |
| `smf/` | dev 分支 | 模块化框架，本地开发 |
| `smf_docs/` | dev 分支 | SMF 模块文档 |
| `_legacy/` | - | 归档的旧代码 |

### Wang/ 目录
```
Wang/
├── bigamp/              # BiG-AMP 算法（推荐）
│   ├── train.py         # 标准训练
│   ├── compare_sizes.py # 尺寸对比实验
│   ├── orthogonal_teacher.py
│   ├── low_loop_graph.py
│   └── replica_overlap.py
├── agd/                 # 交替梯度下降
│   ├── train_parallel.py
│   └── train_sequential.py
├── analysis/            # 分析工具
└── results/             # 实验结果
```

### smf/ 框架
```
smf/
├── cli.py              # 命令行入口
├── core/               # 核心功能
│   ├── config.py       # 配置系统
│   ├── device.py       # GPU/CPU 检测
│   ├── experiment.py   # 实验运行器
│   ├── checkpoint.py   # 检查点
│   ├── llm_advisor.py  # 自然语言配置
│   └── progress.py     # 进度显示
└── modules/            # 可插拔模块
    ├── algorithms/     # bigamp, agd
    ├── graphs/         # random, dinic, low_loop
    ├── teachers/       # standard, orthogonal
    ├── metrics/        # Q_Y, overlap
    └── outputs/        # plotting, storage
```

---

## 研究背景

### 问题定义
Teacher-Student 掩码矩阵分解：给定部分观测 `Y_obs = mask(W₀ × X₀)`，恢复 `W, X` 使得 `Y ≈ Y₀`

### 核心研究：相变现象
- **观测密度** `α̃ = (观测数) / (N₁ × N₂)`
- 当 `α̃ > α̃_c` 时，重建质量 Q_Y 急剧接近 1
- 临界值 `α̃_c` 依赖于矩阵维度比例

### 算法选择
| 算法 | 收敛 | 适用 |
|------|------|------|
| **BiG-AMP** | 200-5000 步 | 大矩阵 (N≥500)，推荐 |
| **AGD** | ~20k epochs | 调试，小矩阵 |

---

## 核心指标

| 指标 | 范围 | 含义 |
|------|------|------|
| `Q_Y` | [0, 1] | 重建质量（旋转不变） |
| `Q_Y_unobserved` | [0, 1] | 未观测位置重合度（泛化） |
| `Q_W`, `Q_X` | [-1, 1] | 因子余弦相似度 |

---

## 关键配置参数

```python
# 矩阵维度
N1, N2, M = 200, 200, 50        # 行数, 列数, 秩

# Alpha 扫描
ALPHA_TILDE_START = 0.0
ALPHA_TILDE_STOP = 4.0
ALPHA_TILDE_STEP = 0.1

# BiG-AMP
MAX_STEPS = 1000                # 迭代数
DAMPING = 0.5                   # 阻尼因子

# 图结构
USE_BIREGULAR_GRAPH = False    # True=Dinic图, False=随机图
```

---

## Git 分支

| 分支 | 用途 |
|------|------|
| `main` | 生产代码，推送到远程 |
| `dev` | 本地开发，包含 smf/ |

```bash
git checkout main             # 切换到生产分支
git checkout dev              # 切换到开发分支
git push origin main dev      # 推送两个分支
```

**远程仓库**: `https://github.com/Sulocus/Sparse-Matrix-Factorization.git`

---

## 跨会话任务接力

| 命令 | 用途 |
|------|------|
| `/pass` | 结束对话前，保存任务状态到 HANDOVER.md |
| `/rem` | 新对话开始时，恢复上下文 |

---

## 依赖

```bash
# 核心依赖
pip install torch numpy scipy matplotlib rich tqdm pyyaml

# GPU 加速 (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
