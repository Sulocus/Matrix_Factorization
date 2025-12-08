# BiG-AMP Spreading Parallel

GPU 并行化的 BiG-AMP Random Spreading 算法实现。

## 概述

`bigamp_spreading_parallel` 是 `bigamp_spreading` 的并行优化版本，通过 **Super-Graph 策略** 实现跨所有 alpha 值的 GPU 并行化。

### 核心洞察

传统的 spreading 算法面临"锯齿张量"问题：不同 alpha 值有不同数量的观测边，无法直接批量处理。

**Super-Graph 策略**解决了这个问题：
- 小 alpha 的图是大 alpha 图的**子集**（Coupled Sampling）
- 预计算最大 alpha 的完整边集
- 使用 mask 选择每个 alpha 的活跃边

## 配置

### SpreadingConfig

```python
@dataclass
class SpreadingConfig:
    f_distribution: str = "gaussian"  # gaussian | rademacher
    seed: int = 12345                 # F 生成种子
```

### 完整配置示例

```python
from smf.core.config import Config, SpreadingConfig

config = Config(
    algorithm_key='bigamp_spreading_parallel',
    spreading=SpreadingConfig(
        f_distribution='gaussian',  # 或 'rademacher'
        seed=12345,
    ),
    # ... 其他配置
)
```

## F 分布类型

| 类型 | 分布 | E[F] | Var[F] | 特点 |
|------|------|------|--------|------|
| `gaussian` | F ~ N(0, 1) | 0 | 1 | 连续值，默认选择 |
| `rademacher` | F ~ {-1, +1} | 0 | 1 | 离散值，计算更稳定 |

两种分布都满足 E[F²] = 1，保证算法数学等价性。

## 数据结构

### SuperGraphData

```python
@dataclass
class SuperGraphData:
    i_idx: torch.Tensor       # (S, C_max) 边的行索引
    j_idx: torch.Tensor       # (S, C_max) 边的列索引
    C_per_alpha: torch.Tensor # (A,) 每个 alpha 的边数
    alpha_mask: torch.Tensor  # (A, C_max) 活跃边掩码
    N1, N2: int               # 矩阵维度
    C_max: int                # 最大边数
    seeds: torch.Tensor       # (S,) 每个样本的种子
    alpha_values: torch.Tensor # (A,) alpha 值列表
```

### SpreadingDataParallel

```python
@dataclass
class SpreadingDataParallel:
    supergraph: SuperGraphData
    F_super: torch.Tensor    # (S, C_max, M) quenched disorder
    Y_super: torch.Tensor    # (S, C_max) 观测值
    M: int
    alpha_values: torch.Tensor
    W_teacher: torch.Tensor  # (N1, M)
    X_teacher: torch.Tensor  # (M, N2)
```

## 使用方法

### 方法 1: 通过 Config + Runner

```python
from smf.core.config import Config, SpreadingConfig
from smf.runner import run_experiment

config = Config(
    algorithm_key='bigamp_spreading_parallel',
    spreading=SpreadingConfig(f_distribution='rademacher'),
    # ...
)
results = run_experiment(config)
```

### 方法 2: 直接调用

```python
from smf.modules.algorithms.bigamp_spreading_parallel import run_spreading_parallel

results = run_spreading_parallel(
    N1=200, N2=200, M=50,
    alpha_values=[0.5, 1.0, 1.5, 2.0],
    S=10,  # 样本数
    max_steps=1000,
    f_distribution='gaussian',
    seed=42,
    device=torch.device('cuda'),
)

# 结果包含:
# - W_teacher, X_teacher: 教师矩阵
# - W_hat, X_hat: (S, A, N, M) 学生估计
# - F_super: (S, C_max, M) F 系数
# - i_idx, j_idx, alpha_mask: 图结构
```

## 显存估算

| 参数 | 示例值 |
|------|--------|
| N1=N2 | 200 |
| M | 50 |
| A (alpha 数) | 61 |
| S (样本数) | 20 |
| C_max | 120,000 |

| 变量 | 形状 | 大小 |
|------|------|------|
| F_super | (20, 120k, 50) | 480 MB |
| Y_super | (20, 120k) | 9.6 MB |
| W_hat | (20, 61, 200, 50) | 49 MB |
| X_hat | (20, 61, 50, 200) | 49 MB |
| 中间变量 | ~2× | ~1.4 GB |

**总计**: ~2.1 GB（RTX 3090/4090 24GB 充裕）

## 与顺序版本对比

| 特性 | bigamp_spreading | bigamp_spreading_parallel |
|------|-----------------|---------------------------|
| 执行模式 | 顺序 for 循环 | GPU 并行 |
| 样本关联 | 独立 | Coupled Sampling |
| 方差 | 较高 | 较低（共享图结构） |
| GPU 利用率 | 低 | 高 |
| 适用规模 | 小规模调试 | 生产训练 |

## 相关文件

- 算法实现: [bigamp_spreading_parallel.py](smf/modules/algorithms/bigamp_spreading_parallel.py)
- SuperGraph: [supergraph.py](smf/modules/graphs/supergraph.py)
- 配置: [config.py](smf/core/config.py) - `SpreadingConfig`
- 测试: [test_supergraph.py](tests/test_supergraph.py)
