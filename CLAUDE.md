# CLAUDE.md

Claude Code 项目指南 - 精简索引版

---

## 🔑 必须遵守的规则

### 语言规范
- **沟通语言**：中文
- **代码/注释/commit**：英文

### Wang/ 程序规范（重要）
1. **程序独立性**：每个程序完整独立，不使用共享模块导入
2. **代码一致性**：不同程序中相同代码块必须**完全一致**（复制粘贴级别）
3. **文档语言**：Wang/README.md 用日语，技术术语用英语

### SMF 文档同步规则（重要）
- **更新程序模块时，必须同步更新 `docs/` 中对应的文档**
- 每个 `smf/core/` 模块应在 `docs/core/` 有对应文档
- 每个 `smf/modules/` 模块应在 `docs/modules/` 有对应文档
- 新功能不明白时，先查阅 `docs/` 目录

---

## 📂 项目结构

| 目录 | Git | 说明 | 详细文档 |
|------|-----|------|----------|
| `Wang/` | ✅ | 生产代码，与日本同学共享 | `Wang/README.md` |
| `smf/` | ❌ | 模块化开发框架 | `docs/README.md` |
| `docs/` | ❌ | SMF 模块详细文档 | `docs/CODE_STRUCTURE.md` |
| `results/` | ❌ | 实验结果 | - |
| `scripts/` | ❌ | 独立脚本工具 | - |

### 📚 docs/ 文档索引
| 类别 | 路径 | 内容 |
|------|------|------|
| 核心模块 | `docs/core/` | device, config 等 |
| 算法 | `docs/modules/algorithms/` | BiG-AMP, AGD |
| 图结构 | `docs/modules/graphs/` | random, dinic, low_loop |
| 教师模型 | `docs/modules/teachers/` | standard, orthogonal |
| 指标 | `docs/modules/metrics/` | Q_Y, overlap 等 |
| 输出 | `docs/modules/outputs/` | 绘图, 存储 |

---

## 📖 项目概述

基于 PyTorch 的 Teacher-Student 掩码矩阵分解研究 (Y = W × X)。

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **BiG-AMP** | ~200-5000 步 | 推荐，适合大矩阵 (N>1000) |
| **AGD** | ~20k epochs | 适合小矩阵 |

---

## 🚀 快速入口

```bash
# 生产训练 (Wang/)
python Wang/bigamp/train.py           # BiG-AMP（推荐）
python Wang/agd/train_parallel.py     # AGD 并行版

# SMF 框架
smf                # 交互模式（自然语言配置）
smf run            # 实验向导
smf run --bg       # 后台运行
smf resume         # 检查点恢复
smf log            # 查看日志

# 实验追踪
aim up             # Aim Web UI
```

---

## 📊 核心指标

| Metric | 范围 | 含义 |
|--------|------|------|
| `Q_Y` | [0, 1] | Reconstruction quality（回转不变） |
| `Q_W`, `Q_X` | [-1, 1] | Factor cosine overlap |
| `Q_W'`, `Q_X'` | [0, 1] | Normalized versions |

**相转移点 (α_c)**：Q_Y 急剧接近 1 的临界观测密度

---

## ⚙️ 关键配置参数

```python
N1, N2, M = 200, 200, 50        # 矩阵维度
ALPHA_TILDE_START/STOP/STEP    # α 扫描范围
MAX_STEPS = 1000                # BiG-AMP 迭代数
USE_BIREGULAR_GRAPH = False    # True=Dinic图, False=随机图
DAMPING = 0.5                   # BiG-AMP 阻尼因子
```

---

## 🛠️ SMF 框架功能

### 核心模块
| 模块 | 文件 | 状态 |
|------|------|------|
| 配置系统 | `core/config.py` | ✅ |
| 实验运行 | `core/runner.py` | ✅ |
| 检查点 | `core/checkpoint.py` | ✅ |
| LLM 配置 | `core/llm_advisor.py` | ✅ |
| 执行计划 | `core/execution_plan.py` | ✅ |
| 计划执行器 | `core/plan_executor.py` | ✅ |
| LLM 日志 | `core/llm_logger.py` | ✅ |

### 待恢复模块（误删）
| 模块 | 文件 | 说明 |
|------|------|------|
| 结果数据库 | `core/results_db.py` | TODO: 需重写 |
| 批量队列 | `core/queue_manager.py` | TODO: 需重写 |
| 实验对比 | `analysis/compare.py` | TODO: 需重写 |

---

## 📋 跨会话任务接力

| 命令 | 用途 |
|------|------|
| `/pass` | 当前对话结束前，保存任务状态到 HANDOVER.md |
| `/rem` | 新对话开始时，恢复上下文 |

---

## 📦 依赖

```
torch>=2.0.0  numpy>=1.21.0  scipy>=1.7.0  matplotlib>=3.5.0
```

GPU 加速：`pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

*最后更新：2025年12月*
