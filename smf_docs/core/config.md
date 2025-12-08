# Config: 实验配置系统

配置管理模块，使用 dataclass 定义实验参数

**模块 ID**: C1
**SMF 路径**: `smf/core/config.py`

---

## 目的

提供结构化的实验配置管理，支持 YAML/JSON 序列化和配置哈希生成。

---

## 功能说明

1. **分层配置结构**: MatrixConfig, AlphaConfig, TrainingConfig, AlgorithmConfig, ExecutionConfig
2. **序列化支持**: YAML 和 JSON 格式的读写
3. **配置哈希**: 生成唯一标识符用于实验追踪
4. **LLM 执行参数**: ExecutionConfig 支持动态指标和绘图配置

---

## 主要类

### MatrixConfig
```python
@dataclass
class MatrixConfig:
    N1: int = 200       # 矩阵行数
    N2: int = 200       # 矩阵列数
    M: int = 50         # 秩 (rank)
```

### AlphaConfig
```python
@dataclass
class AlphaConfig:
    start: float = 0.0  # Alpha 扫描起点
    stop: float = 4.0   # Alpha 扫描终点
    step: float = 0.1   # Alpha 步长

    def get_values(self) -> list[float]:
        """生成 alpha 值列表"""
```

### TrainingConfig
```python
@dataclass
class TrainingConfig:
    max_steps: int = 5000       # BiG-AMP 迭代数
    max_epochs: int = 20000     # AGD epochs
    samples_per_alpha: int = 1  # 每个 alpha 的采样次数
    seed: int = 42              # 随机种子
    resample_mask: bool = True  # 是否每次重采样 mask
```

### AlgorithmConfig
```python
@dataclass
class AlgorithmConfig:
    damping: float = 0.5              # BiG-AMP 阻尼因子
    noise_var: float = 1e-10          # 噪声方差
    learning_rate: float = 0.01       # AGD 学习率
    early_stop: bool = False          # 提前终止
    convergence_threshold: float = 1e-6
    use_compile: bool = True          # torch.compile 加速
```

### ExecutionConfig
```python
@dataclass
class ExecutionConfig:
    metrics_to_compute: List[str]     # 要计算的指标列表
    plots: List[Dict[str, Any]]       # 绘图配置
    include_summary_plot: bool = True
    include_qy_plot: bool = True
```

### Config (主配置类)
```python
@dataclass
class Config:
    matrix: MatrixConfig
    alpha: AlphaConfig
    training: TrainingConfig
    algorithm: AlgorithmConfig
    algorithm_key: str = "bigamp"
    graph_key: str = "random"
    teacher_key: str = "standard"
    execution: ExecutionConfig

    def to_dict(self) -> dict
    def to_yaml(self, path: Path) -> None
    def to_json(self, path: Path) -> None
    @classmethod
    def from_dict(cls, data: dict) -> 'Config'
    @classmethod
    def from_yaml(cls, path: Path) -> 'Config'
    @classmethod
    def from_json(cls, path: Path) -> 'Config'
    def get_hash(self) -> str
    def get_display_name(self) -> str
```

---

## 使用示例

```python
from smf.core.config import Config, MatrixConfig, AlphaConfig

# 创建默认配置
config = Config()

# 自定义配置
config = Config(
    matrix=MatrixConfig(N1=1000, N2=1000, M=100),
    alpha=AlphaConfig(start=0.0, stop=3.0, step=0.05),
    algorithm_key="bigamp",
    graph_key="dinic",
)

# 保存/加载
config.to_yaml(Path("config.yaml"))
loaded = Config.from_yaml(Path("config.yaml"))

# 获取配置哈希
hash_id = config.get_hash()  # e.g., "a1b2c3d4"
```

---

## 输入/输出

**输入**:
- 配置参数（直接构造或从文件加载）

**输出**:
- `Config` 对象
- YAML/JSON 文件
- 配置哈希字符串

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "Structured experiment configuration management",
    "when_to_use_en": "When setting up experiments with specific parameters",
    "tags_en": ["config", "dataclass", "yaml", "json", "serialization"],

    "purpose_zh": "结构化实验配置管理",
    "when_to_use_zh": "设置实验参数时使用",
    "tags_zh": ["配置", "数据类", "序列化", "YAML", "JSON"],
}
```

---

*最后更新：2025年12月*
