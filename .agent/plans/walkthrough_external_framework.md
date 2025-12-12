# 外部控制框架重构 - 工作记录

## 概述

**目标**：将算法变成纯净的计算单元——只负责计算，不负责数据创建、保存、调度。

---

## 已完成工作

### 阶段 1-3：核心框架 ✅

- [config.py](file:///home/sucia/Sparse-Matrix/smf/core/experiment/config.py) - 配置数据结构
- [result.py](file:///home/sucia/Sparse-Matrix/smf/core/experiment/result.py) - 统一结果保存
- [data_factory.py](file:///home/sucia/Sparse-Matrix/smf/core/experiment/data_factory.py) - 数据创建工厂
- [runner.py](file:///home/sucia/Sparse-Matrix/smf/core/experiment/runner.py) - 实验执行器
- [base.py](file:///home/sucia/Sparse-Matrix/smf/modules/algorithms/base.py) - `run_single()` 接口

### 数据保存扩展 ✅

**问题**：之前保存不完整，无法计算 Q_Y_unobserved（缺少 mask）

**解决**：扩展 `.pt` 文件包含所有原始数据：

```python
{
    # 配置（完整可重现）
    'config': {...},
    'metadata': {...},
    
    # 原始数据（用于后续分析）
    'raw_data': {
        'W_teacher': tensor,    # 教师 W
        'X_teacher': tensor,    # 教师 X
        'Y_teacher': tensor,    # 教师 Y = W @ X
        'all_masks': tensor,    # 所有 alpha 的 mask
        'supergraph_data': dict, # Spreading 图结构
    },
    
    # 每个扫描点的结果
    'results': {
        0.5: {
            'metrics': {'Q_W_mean': 0.9, 'Q_Y_unobserved': 0.7},
            'W_students': tensor,
            'X_students': tensor,
            'mask': tensor,            # 这个 alpha 的 mask
            'observation_indices': {   # Spreading 观测索引
                'i_idx': tensor,
                'j_idx': tensor,
                'edge_counts': tensor,
            },
        },
        1.0: {...},
    }
}
```

**测试结果**：
```
✓ W_teacher: torch.Size([50, 10])
✓ mask: torch.Size([50, 50])
✓ observation_indices: keys=['i_idx', 'j_idx']
```

---

## 使用示例

```python
from smf.core.experiment import ExperimentConfig, ExperimentRunner, ExperimentResult

# 1. 配置
config = ExperimentConfig(
    matrix=MatrixParams(N1=600, N2=600, M=150),
    scan=ScanConfig(dimension='alpha', values=[0.5, 1.0, 1.5, 2.0]),
    ...
)

# 2. 运行
runner = ExperimentRunner()
result = runner.run(config)

# 3. 保存（单文件！）
result.save_unified('results/my_exp.pt')

# 4. 加载和后续分析
loaded = ExperimentResult.load_unified('results/my_exp.pt')

# 现在可以计算 Q_Y_unobserved：
mask = loaded.results[0.5].mask
W_teacher = loaded.W_teacher
# ... 计算
```

---

## 文档位置

- 详细设计：[.agent/plans/external_control_framework.md](file:///home/sucia/Sparse-Matrix/.agent/plans/external_control_framework.md)
