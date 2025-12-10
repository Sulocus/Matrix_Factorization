# 代码结构文档

本目录包含代码模块的详细中文文档，与 SMF 模块化架构一一对应。

**语言**: 中文（技术术语用英语）
**不上传**: 此目录仅供本地开发参考

---

## 目录结构

```
smf_docs/
├── README.md                 ← 本文件（总览）
├── core/                     ← 核心工具
│   └── device.md             # D1: 设备检测
│
└── modules/                  ← 模块文档
    ├── graphs/               ← 观测掩码生成
    │   ├── random.md         # G1: 随机图
    │   ├── dinic.md          # G2: Dinic 最大流
    │   ├── uniform.md        # G3: 双正则图
    │   └── low_loop.md       # G4: 低循环图
    │
    ├── teachers/             ← 教师矩阵生成
    │   ├── standard.md       # T1: 标准高斯
    │   ├── scaled.md         # T2: 缩放方差
    │   └── orthogonal.md     # T3: 正交教师
    │
    ├── metrics/              ← 评估指标
    │   ├── gram_cosine.md    # M1: Gram Cosine
    │   ├── gram_normalized.md# M2: Gram Normalized
    │   ├── qy.md             # M3: Q_Y 重建质量
    │   ├── generalization.md # M4: 泛化误差
    │   ├── replica.md        # M5: Replica Overlap
    │   ├── qy_unobserved.md  # M6: Q_Y 未观测
    │   └── aggregators.md    # M7: 指标聚合
    │
    ├── algorithms/           ← 算法实现
    │   ├── bigamp/
    │   │   ├── core.md       # A1: BiG-AMP 消息传递
    │   │   ├── state.md      # A2: 状态管理
    │   │   └── damping.md    # A3: 阻尼机制
    │   ├── agd/
    │   │   ├── core.md       # A4: AGD 梯度更新
    │   │   ├── optimizer.md  # A5: Adam 优化器
    │   │   └── scheduler.md  # A6: 学习率调度
    │   └── convergence.md    # A7: 收敛检测
    │
    └── outputs/              ← 输出处理
        ├── plotting/
        │   ├── colors.md     # P1: 颜色配置
        │   ├── styles.md     # P2: 样式配置
        │   ├── curves.md     # P3: 曲线绘制
        │   └── comparison.md # P4: 对比图
        └── storage/
            ├── json_io.md    # S1: JSON 读写
            └── scanner.md    # S2: 结果扫描
```

---

## 模块 ID 索引

### 核心工具 (D)
| ID | 模块 | 文档 | 说明 |
|----|------|------|------|
| D1 | device | [device.md](core/device.md) | GPU/CPU 检测和配置 |

### 图生成 (G)
| ID | 模块 | 文档 | 说明 |
|----|------|------|------|
| G1 | random | [random.md](modules/graphs/random.md) | 纯随机掩码，最快 |
| G2 | dinic | [dinic.md](modules/graphs/dinic.md) | 最大流算法，G3 的依赖 |
| G3 | uniform | [uniform.md](modules/graphs/uniform.md) | 双正则图，理论验证推荐 |
| G4 | low_loop | [low_loop.md](modules/graphs/low_loop.md) | 无 C4 图，AMP 最优 |

### 教师矩阵 (T)
| ID | 模块 | 文档 | 说明 |
|----|------|------|------|
| T1 | standard | [standard.md](modules/teachers/standard.md) | 标准高斯，默认 |
| T2 | scaled | [scaled.md](modules/teachers/scaled.md) | 自定义方差 |
| T3 | orthogonal | [orthogonal.md](modules/teachers/orthogonal.md) | 正交化，消除有限尺寸涨落 |

### 评估指标 (M)
| ID | 模块 | 文档 | 说明 |
|----|------|------|------|
| M1 | gram_cosine | [gram_cosine.md](modules/metrics/gram_cosine.md) | Gram 矩阵余弦相似度 |
| M2 | gram_normalized | [gram_normalized.md](modules/metrics/gram_normalized.md) | 归一化 [0,1] 范围 |
| M3 | qy | [qy.md](modules/metrics/qy.md) | 最重要！重建质量 |
| M4 | generalization | [generalization.md](modules/metrics/generalization.md) | 未观测位置 MSE |
| M5 | replica | [replica.md](modules/metrics/replica.md) | 解唯一性分析 |
| M6 | qy_unobserved | [qy_unobserved.md](modules/metrics/qy_unobserved.md) | 泛化能力测试 |
| M7 | aggregators | [aggregators.md](modules/metrics/aggregators.md) | 统计聚合 |

### 算法 (A)
| ID | 模块 | 文档 | 说明 |
|----|------|------|------|
| A1 | bigamp/core | [core.md](modules/algorithms/bigamp/core.md) | BiG-AMP 核心 |
| A2 | bigamp/state | [state.md](modules/algorithms/bigamp/state.md) | 状态初始化 |
| A3 | bigamp/damping | [damping.md](modules/algorithms/bigamp/damping.md) | 阻尼更新 |
| A4 | agd/core | [core.md](modules/algorithms/agd/core.md) | AGD 核心 |
| A5 | agd/optimizer | [optimizer.md](modules/algorithms/agd/optimizer.md) | Adam 配置 |
| A6 | agd/scheduler | [scheduler.md](modules/algorithms/agd/scheduler.md) | 学习率调度 |
| A7 | convergence | [convergence.md](modules/algorithms/convergence.md) | 早停检测 |

### 输出 (P, S)
| ID | 模块 | 文档 | 说明 |
|----|------|------|------|
| P1 | colors | [colors.md](modules/outputs/plotting/colors.md) | 标准颜色 |
| P2 | styles | [styles.md](modules/outputs/plotting/styles.md) | 绘图样式 |
| P3 | curves | [curves.md](modules/outputs/plotting/curves.md) | 曲线绘制 |
| P4 | comparison | [comparison.md](modules/outputs/plotting/comparison.md) | 对比图 |
| S1 | json_io | [json_io.md](modules/outputs/storage/json_io.md) | JSON 读写 |
| S2 | scanner | [scanner.md](modules/outputs/storage/scanner.md) | 结果扫描 |

---

## Wang/ 程序与模块对应

每个 Wang/ 程序是多个模块的组合：

| 程序 | 使用的模块 |
|------|-----------|
| `bigamp/train.py` | D1, G1-G3, T1, M1-M5, A1-A3, P1-P3, S1 |
| `bigamp/compare_sizes.py` | D1, G1-G3, T1, M1-M6, A1-A3, P1-P4, S1-S2 |
| `bigamp/orthogonal_teacher.py` | D1, G1-G3, T3, M1-M6, A1-A3, P1-P3, S1 |
| `bigamp/low_loop_graph.py` | D1, G4, T1, M1-M5, A1-A3, P1-P3, S1 |
| `bigamp/replica_overlap.py` | D1, G1-G3, T1, M5, A1-A3, P1-P3, S1 |
| `agd/train_parallel.py` | D1, G1-G3, T1, M1-M3, A4-A7, P1-P3, S1 |
| `agd/train_sequential.py` | D1, G1, T1, M1-M3, A4-A7, P1-P3, S1 |
| `analysis/compare_algorithms.py` | S1-S2, P4 |
| `analysis/slope_ratio.py` | S1-S2, P3 |
| `analysis/degree_distribution.py` | G1-G4, P3 |

---

## 算法选择指南

```
需要矩阵分解？
    │
    ├── N < 500 且需要调试
    │       └── 使用 AGD (A4-A7)
    │
    └── N ≥ 500 或需要快速收敛
            └── 使用 BiG-AMP (A1-A3)

需要理论验证？
    │
    ├── 研究有限尺寸效应
    │       └── 使用正交教师 (T3) + 双正则图 (G3)
    │
    └── 最精确的 AMP 验证
            └── 使用低循环图 (G4)
```

---

## 文档模板

每个模块文档遵循**双视角**结构：

```markdown
# 模块名称

**模块 ID**: Xn
**SMF 路径**: `modules/...`

---

## 🌐 宏观视角
- 系统定位
- 引入动机
- 相对优势
- 物理图景 🌟
- 使用场景

---

## 🔬 微观视角
- 代码位置
- 数学定义
- 输入/输出
- 标准实现
- 实现细节

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "...",
    "when_to_use_en": "...",
    "tags_en": [...],
    "tags_zh": [...],
}
```
```

---

*最后更新：2025年12月*
