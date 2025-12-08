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
- **更新程序模块时，必须同步更新 `smf_docs/` 中对应的文档**
- 每个 `smf/core/` 模块应在 `smf_docs/core/` 有对应文档
- 每个 `smf/modules/` 模块应在 `smf_docs/modules/` 有对应文档
- 新功能不明白时，先查阅 `smf_docs/` 目录

---

## 常用命令

```bash
# 安装 SMF 框架（开发模式）
pip install -e .

# 运行测试
pytest tests/                           # 运行所有测试
pytest tests/test_e2e_comprehensive.py  # 运行单个测试文件

# SMF CLI
smf                # 交互模式（自然语言配置）
smf run            # 实验向导
smf run --bg       # 后台运行
smf resume         # 检查点恢复
smf log            # 查看日志
smf vis            # 结果浏览器
smf test           # 快速测试

# 生产训练 (Wang/)
python Wang/bigamp/train.py           # BiG-AMP（推荐）
python Wang/agd/train_parallel.py     # AGD 并行版
```

---

## 项目结构

**双轨架构**：
- `Wang/` - 生产代码，提交到 Git，与日本同学共享（README 用日语）
- `smf/` - 本地模块化框架，不提交（通过 `pip install -e .` 安装使用）

| 目录 | Git | 说明 |
|------|-----|------|
| `Wang/` | ✅ | 生产代码：`agd/`, `bigamp/`, `analysis/` |
| `smf/` | - | 模块化框架：`core/`, `modules/`, `ui/`, `scripts/` |
| `smf_docs/` | - | SMF 模块文档，查阅 `smf_docs/README.md` |
| `tests/` | - | 临时测试目录 |
| `_legacy/` | - | 归档的旧代码 |

### smf_docs/ 文档索引
| 类别 | 路径 | 内容 |
|------|------|------|
| 核心模块 | `smf_docs/core/` | device, config 等 |
| 算法 | `smf_docs/modules/algorithms/` | BiG-AMP, AGD |
| 图结构 | `smf_docs/modules/graphs/` | random, dinic, low_loop |
| 教师模型 | `smf_docs/modules/teachers/` | standard, orthogonal |
| 指标 | `smf_docs/modules/metrics/` | Q_Y, overlap 等 |
| 输出 | `smf_docs/modules/outputs/` | 绘图, 存储 |

---

## 项目概述

基于 PyTorch 的 Teacher-Student 掩码矩阵分解研究：`Y = W × X`

研究 **相转移现象 (phase transition)**：在稀疏观测下，当观测密度 α 超过临界值 α_c 时，Q_Y 急剧接近 1。

| 算法 | 收敛速度 | 适用场景 |
|------|----------|----------|
| **BiG-AMP** | ~200-5000 步 | 推荐，大矩阵 (N>1000) |
| **AGD** | ~20k epochs | 小矩阵，调试 |

---

## 核心指标

| Metric | 范围 | 含义 |
|--------|------|------|
| `Q_Y` | [0, 1] | Reconstruction quality（回转不变） |
| `Q_W`, `Q_X` | [-1, 1] | Factor cosine overlap |
| `Q_W'`, `Q_X'` | [0, 1] | Normalized versions |

---

## 关键配置参数

```python
N1, N2, M = 200, 200, 50        # 矩阵维度
ALPHA_TILDE_START/STOP/STEP    # α 扫描范围
MAX_STEPS = 1000                # BiG-AMP 迭代数
USE_BIREGULAR_GRAPH = False    # True=Dinic图, False=随机图
DAMPING = 0.5                   # BiG-AMP 阻尼因子
```

---

## SMF 架构

### 核心模块 (smf/core/)
| 模块 | 说明 |
|------|------|
| `config.py` | 配置系统 |
| `device.py` | GPU/CPU 设备检测 |
| `experiment.py` | 实验运行器 |
| `checkpoint.py` | 检查点保存/恢复 |
| `execution_plan.py` | 执行计划生成 |
| `plan_executor.py` | 计划执行器 |
| `llm_advisor.py` | LLM 自然语言配置 |
| `llm_logger.py` | LLM 日志记录 |
| `memory_manager.py` | GPU 内存管理 |

### 模块系统 (smf/modules/)
| 类别 | 模块 |
|------|------|
| algorithms | `bigamp.py`, `agd.py` |
| graphs | `random.py`, `dinic.py`, `low_loop.py`, `uniform.py` |
| teachers | `standard.py`, `orthogonal.py`, `scaled_variance.py` |
| metrics | `overlap.py`, `qy_unobserved.py`, `spreading.py` |
| outputs | `plotting.py`, `storage.py`, `comparison.py` |

---

## 跨会话任务接力

| 命令 | 用途 |
|------|------|
| `/pass` | 当前对话结束前，保存任务状态到 HANDOVER.md |
| `/rem` | 新对话开始时，恢复上下文 |

---

## 依赖

核心依赖：`torch>=2.0.0`, `numpy`, `scipy`, `matplotlib`, `rich`

```bash
# GPU 加速
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
