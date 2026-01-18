# 外部控制框架重构方案

## 核心思想

**算法变成 "Jupyter Cell"**：只负责计算，不知道实验目的、不知道并行、不知道保存。

```
┌─────────────────────────────────────────────────────────────┐
│  ExperimentConfig (参数表 - 第一个 Cell)                    │
│  所有可控参数集中定义，完整记录以便重现                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  ExperimentRunner (执行控制)                                │
│  - 根据扫描模式决定数据复用策略                             │
│  - 根据内存估算决定分批                                     │
│  - 调用算法执行计算                                         │
│  - 收集和保存结果                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Algorithm.run_single() (纯计算 - 第二个 Cell)              │
│  - 接收准备好的数据                                         │
│  - 执行计算，返回结果                                       │
│  - 不知道扫描什么，不知道并行                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 三个算法的参数提取

### 参数对比表

| 参数类别 | AGD | BiGAMP | BiGAMPSpreading |
|---------|-----|--------|-----------------|
| **矩阵结构** | N1, N2, M | N1, N2, M | N1, N2, M |
| **训练参数** | S, max_epochs | S, max_steps | S, max_steps |
| **算法参数** | learning_rate | damping, noise_var | damping, noise_var |
| **随机种子** | seed | seed | base_seed, spreading_seed |
| **扫描维度** | alpha_values | alpha_values | alpha_values |
| **特有参数** | early_stop | use_compile | f_distribution |
| **数据依赖** | mask | mask | SpreadingData (F, Y, Graph) |

### 统一配置结构

```python
@dataclass
class ExperimentConfig:
    """完整实验配置 - 所有参数集中管理"""
    
    # 矩阵结构
    matrix: MatrixParams  # N1, N2, M
    
    # 训练参数
    training: TrainingParams  # S, max_steps, max_epochs
    
    # 算法选择
    algorithm_key: str  # "agd" / "bigamp" / "bigamp_spreading"
    algorithm_params: Dict[str, Any]  # damping, lr, noise_var, etc.
    
    # 随机性控制
    seeds: SeedConfig  # base_seed, spreading_seed, teacher_seed
    
    # 扫描配置
    scan: ScanConfig  # dimension, values
    
    # Spreading 特有（可选）
    spreading: Optional[SpreadingConfig]  # f_distribution
    
    # 元信息
    experiment_name: str
    notes: str = ""

@dataclass
class MatrixParams:
    N1: int
    N2: int
    M: int

@dataclass
class TrainingParams:
    samples_per_alpha: int  # S
    max_steps: int = 5000
    max_epochs: int = 20000  # For AGD

@dataclass
class SeedConfig:
    base_seed: int = 42
    teacher_seed: int = 12345
    spreading_seed: int = 99999  # Only for spreading

@dataclass
class ScanConfig:
    dimension: str  # "alpha" / "steps" / "samples" / "seed"
    values: List[Any]
```

---

## 扫描模式支持

### 扫描维度对数据复用的影响

| 扫描维度 | Teacher 复用 | Mask/Graph 复用 | Student 初始化 |
|---------|-------------|----------------|---------------|
| α (alpha) | ✓ 复用 | ✗ 重建 | ✗ 重建 |
| steps | ✓ 复用 | ✓ 复用 | ✓ Checkpoint 继续 |
| S (samples) | ✓ 复用 | ✗ 重建 | ✗ 重建 |
| N (matrix size) | ✗ 重建 | ✗ 重建 | ✗ 重建 |
| seed (replica) | ✓ 复用 | ✗ 重建 | ✗ 重建 |

### 扫描 Steps 的特殊处理

```python
# 需要支持 Checkpoint 机制
result_100 = algorithm.run_single(data, max_steps=100)
result_500 = algorithm.run_single(data, max_steps=500, 
                                   checkpoint=result_100.state)
result_1000 = algorithm.run_single(data, max_steps=1000, 
                                    checkpoint=result_500.state)
```

---

## 新文件结构

```
MF/core/experiment/          # 新模块
├── __init__.py
├── config.py                 # ExperimentConfig 等数据结构
├── data_factory.py           # 创建 Teacher, Mask, SpreadingData
├── runner.py                 # ExperimentRunner 主执行器
├── result.py                 # ExperimentResult 统一保存
├── scan_modes.py             # 扫描模式定义和策略
└── memory_planner.py         # 内存估算和分批策略

MF/modules/algorithms/       # 修改现有
├── base.py                   # 简化接口，新增 run_single()
├── bigamp.py                 # 移除内部分批
├── agd.py                    # 移除内部分批
└── bigamp_spreading.py       # 移除内部数据创建
```

---

## 算法接口变化

### 旧接口（将废弃，保留兼容）

```python
# 算法内部知道太多
result = algorithm.train_batch_alphas(
    W_teacher, X_teacher, Y_teacher, 
    masks, alpha_values, seed
)
```

### 新接口（纯计算）

```python
# 外部准备数据
data = data_factory.create(config, scan_value)

# 算法只做计算
result = algorithm.run_single(
    data=data,
    params=training_params,
    checkpoint=None,  # 可选：从上一步继续
)
```

### Algorithm 基类变化

```python
class AlgorithmBase:
    """算法基类 - 只负责计算"""
    
    def run_single(
        self,
        data: ExperimentData,
        params: TrainingParams,
        checkpoint: Optional[Checkpoint] = None,
    ) -> RunResult:
        """
        执行单次计算
        
        Args:
            data: 准备好的实验数据
            params: 训练参数
            checkpoint: 可选的检查点（用于扫描 steps）
            
        Returns:
            RunResult with metrics and optional state
        """
        raise NotImplementedError
    
    # 兼容层（调用新接口）
    def train_batch_alphas(self, ...):
        """旧接口，内部调用 run_single()"""
        ...
```

---

## 统一数据保存

### ExperimentResult 结构

```python
@dataclass
class ExperimentResult:
    """统一的实验结果结构"""
    
    # 标识
    experiment_id: str
    timestamp: str
    
    # 配置（完整，可重现）
    config: ExperimentConfig
    
    # 扫描信息
    scan_dimension: str
    scan_values: List[Any]
    
    # 结果
    results: Dict[Any, SingleRunResult]
    
    # 元信息
    metadata: Dict  # gpu_name, duration, smf_version, etc.
    
    def save(self, path: Path):
        """保存到目录（config.json + results.pt）"""
        ...
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentResult':
        """从目录加载"""
        ...

@dataclass
class SingleRunResult:
    """单次运行结果"""
    scan_value: Any
    metrics: Dict[str, float]  # Q_W, Q_X, Q_Y, etc.
    W_students: Optional[torch.Tensor] = None
    X_students: Optional[torch.Tensor] = None
    history: Optional[List[Dict]] = None  # 收敛曲线
```

---

## 实施步骤

### 阶段 1：创建外部框架 (4h)

1. 创建 `MF/core/experiment/` 目录
2. 实现 `config.py` - 所有配置数据结构
3. 实现 `data_factory.py` - 数据创建工厂
4. 实现 `result.py` - 结果保存/加载

### 阶段 2：简化算法接口 (3h)

1. 在 `base.py` 添加 `run_single()` 抽象方法
2. 修改 `agd.py` - 实现 `run_single()`
3. 修改 `bigamp.py` - 实现 `run_single()`
4. 修改 `bigamp_spreading.py` - 移除内部数据创建

### 阶段 3：实现 Runner (3h)

1. 实现 `runner.py` - ExperimentRunner
2. 实现 `scan_modes.py` - 扫描策略
3. 实现 `memory_planner.py` - 分批策略

### 阶段 4：测试验证 (2h)

1. 单元测试：各组件
2. 集成测试：完整扫描 α 流程
3. 集成测试：扫描 steps 流程

**总计：~12-14 小时 / 2 天**

---

## 需要确认的问题

1. **兼容性**：是否保留旧的 `train_batch_alphas` 接口作为兼容层？
2. **图结构持久化**：Spreading 的 SuperGraph 是否需要独立保存（用于跨实验复用同一图结构）？
3. **扫描优先级**：除了 α 和 steps，哪些扫描维度需要优先支持？(S / N / seed)

