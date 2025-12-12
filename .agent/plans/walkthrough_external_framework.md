# 外部控制框架重构 - 工作记录

## 概述

**目标**：将算法变成纯净的计算单元——只负责计算，不负责数据创建、保存、调度。

**核心思想**：
- 外部框架负责：实验逻辑、参数控制、数据创建、并行调度、结果保存
- 算法只负责：接收数据，执行计算，返回结果

---

## 已完成工作

### 阶段 1：创建外部实验框架 ✅

**创建文件**：
- [config.py](file:///home/sucia/Sparse-Matrix/smf/core/experiment/config.py) - 所有配置数据结构
- [result.py](file:///home/sucia/Sparse-Matrix/smf/core/experiment/result.py) - 统一结果保存
- [data_factory.py](file:///home/sucia/Sparse-Matrix/smf/core/experiment/data_factory.py) - 数据创建工厂
- [runner.py](file:///home/sucia/Sparse-Matrix/smf/core/experiment/runner.py) - 实验执行器

**测试结果**：✅ 所有导入和保存/加载测试通过

---

### 阶段 2：简化算法接口 ✅

**修改文件**：
- [base.py](file:///home/sucia/Sparse-Matrix/smf/modules/algorithms/base.py) - 添加 `run_single()` 统一接口

**新接口**：
```python
# 算法只做计算，不知道实验目的
W, X, checkpoint = algorithm.run_single(data, max_steps=50)
```

**测试结果**：✅ 所有算法正确继承 AlgorithmBase

---

### 阶段 3：集成测试 ✅

**完整流程测试**：
```
ExperimentConfig → DataFactory → Algorithm.run_single()
```

**测试输出**：
```
✓ 配置: 100x100, M=25
✓ Runner
✓ 算法: BiGAMPSpreading
✓ 数据: W=torch.Size([100, 25])
✓ 计算: W=torch.Size([2, 4, 100, 25])
集成测试通过！
```

---

## 支持的功能

### 扫描维度
- `alpha` - 观测密度扫描 ✅
- `steps` - 步数扫描（收敛曲线）✅
- `N` / `M` - 矩阵规模扫描（外层循环）✅

### 嵌套扫描
```python
runner.run_scaling_sweep(
    base_config=config,
    matrix_sizes=[(200, 200, 50), (400, 400, 100), (600, 600, 150)],
)
```

---

## 架构图

```
┌─────────────────────────────────────────┐
│  ExperimentConfig (参数表)              │
│  - matrix: N1, N2, M                    │
│  - training: S, max_steps               │
│  - scan: dimension, values              │
│  - seeds: base_seed, spreading_seed     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  ExperimentRunner (执行控制)            │
│  - 路由扫描模式                         │
│  - 内存管理                             │
│  - 收集结果                             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  DataFactory (数据准备)                 │
│  - 创建 Teacher                         │
│  - 创建 Mask / SpreadingData            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Algorithm.run_single() (纯计算)        │
│  - 接收准备好的数据                     │
│  - 执行计算                             │
│  - 返回结果                             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  ExperimentResult (统一保存)            │
│  - config.json                          │
│  - results.json                         │
│  - tensors/ (可选)                      │
└─────────────────────────────────────────┘
```

---

## 文档位置

- 详细设计方案：[.agent/plans/external_control_framework.md](file:///home/sucia/Sparse-Matrix/.agent/plans/external_control_framework.md)
- 本工作记录：[.agent/plans/walkthrough_external_framework.md](file:///home/sucia/Sparse-Matrix/.agent/plans/walkthrough_external_framework.md)
