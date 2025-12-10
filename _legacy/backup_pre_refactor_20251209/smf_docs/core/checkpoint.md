# Checkpoint: 检查点管理

实验断点续传系统，支持长时间实验的中断恢复

**模块 ID**: C2
**SMF 路径**: `smf/core/checkpoint.py`

---

## 目的

允许长时间运行的实验在中断后从最后保存的状态恢复，避免重复计算已完成的 alpha 点。

---

## 功能说明

1. **定期保存**: 每 N 个 alpha 点自动保存检查点
2. **配置验证**: 通过哈希校验确保恢复的配置一致
3. **自动清理**: 实验成功完成后删除检查点文件
4. **增量恢复**: 只运行未完成的 alpha 点

---

## 主要类

### Checkpoint
```python
@dataclass
class Checkpoint:
    config_hash: str               # 配置哈希，用于一致性验证
    completed_alphas: List[float]  # 已完成的 alpha 值列表
    results: Dict[str, Any]        # {alpha: metrics} 结果字典
    teacher_seed: int              # 教师模型随机种子
    timestamp: datetime            # 保存时间戳

    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint"
```

### CheckpointManager
```python
class CheckpointManager:
    """检查点管理器"""

    def __init__(self, output_dir: Path, config: Any, save_interval: int = 10):
        """
        Args:
            output_dir: 检查点保存目录
            config: 实验配置对象
            save_interval: 每 N 个 alpha 保存一次
        """

    def save(self, completed_alphas: List[float], results: Dict) -> Path:
        """保存检查点"""

    def load_latest(self) -> Optional[Checkpoint]:
        """加载最新的有效检查点"""

    def get_remaining_alphas(self, all_alphas: List[float]) -> List[float]:
        """获取未完成的 alpha 值列表"""

    def should_save(self, num_completed: int) -> bool:
        """判断是否应该保存检查点"""

    def cleanup(self):
        """清理所有检查点文件"""

    def get_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """获取检查点信息（不加载完整数据）"""
```

---

## 使用示例

```python
from smf.core.checkpoint import CheckpointManager

# 初始化
mgr = CheckpointManager(output_dir, config, save_interval=10)

# 检查是否有可恢复的检查点
if resume:
    remaining = mgr.get_remaining_alphas(all_alphas)
    checkpoint = mgr.load_latest()
    results = checkpoint.results.copy() if checkpoint else {}
else:
    remaining = all_alphas
    results = {}

# 训练循环
for i, alpha in enumerate(remaining):
    results[alpha] = train(alpha)

    # 定期保存
    if mgr.should_save(i + 1):
        mgr.save(list(results.keys()), results)

# 成功完成后清理
mgr.cleanup()
```

---

## 文件结构

```
output_dir/
└── .checkpoints/
    ├── checkpoint_0010.pkl  # 完成 10 个 alpha
    ├── checkpoint_0020.pkl  # 完成 20 个 alpha
    └── checkpoint_0030.pkl  # 完成 30 个 alpha
```

---

## 输入/输出

**输入**:
- `output_dir`: 输出目录
- `config`: 实验配置
- `save_interval`: 保存间隔

**输出**:
- `.pkl` 格式的检查点文件
- `Checkpoint` 对象（加载时）

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "Experiment checkpoint management for interruption recovery",
    "when_to_use_en": "For long-running experiments that may be interrupted",
    "tags_en": ["checkpoint", "resume", "recovery", "pickle", "serialization"],

    "purpose_zh": "实验检查点管理，支持中断恢复",
    "when_to_use_zh": "用于可能中断的长时间实验",
    "tags_zh": ["检查点", "断点续传", "恢复", "序列化"],
}
```

---

*最后更新：2025年12月*
