# ExecutionPlan: 执行计划

描述实验执行内容的预览层，支持单步和多步执行

**模块 ID**: C3
**SMF 路径**: `smf/core/execution_plan.py`

---

## 目的

在实际执行前生成实验计划预览，用于 UI 显示和用户确认。支持：
- 单步执行（标准实验）
- 多步执行（对比实验）
- 后处理钩子（合并绘图等）

---

## 功能说明

1. **模块调用信息**: 记录 teacher, graph, algorithm 等模块配置
2. **参数预览**: 矩阵维度、alpha 范围、指标列表
3. **多步支持**: 对比实验的多配置管理
4. **后处理**: merge_plot 等后处理操作

---

## 主要类

### ModuleCall
```python
@dataclass
class ModuleCall:
    key: str              # 模块标识符
    module_type: str      # 模块类型（TeacherBase, GraphBase, etc.）
    module_name: str      # 显示名称
    params: Dict[str, Any]  # 关键参数
```

### ExecutionStep
```python
@dataclass
class ExecutionStep:
    """多步计划中的单个步骤"""
    config_dict: Dict[str, Any]   # 步骤配置（覆盖基础配置）
    label: str                     # 显示标签: "Random Graph"
    result_path: Optional[Path]    # 执行后填充
```

### ExecutionPlan
```python
@dataclass
class ExecutionPlan:
    # 核心模块
    teacher: ModuleCall
    graph: ModuleCall
    algorithm: ModuleCall

    # 参数信息
    matrix_info: str      # "1000×1000, M=100"
    alpha_info: str       # "0.0 ~ 3.0, step 0.1 (31 points)"
    alpha_count: int

    # 评估配置
    metrics: List[str]    # ["Q_Y", "Q_W", ...]
    plots: List[str]      # ["summary.png", "qy_vs_alpha.png"]

    # 算法参数
    damping: float = 0.5
    samples_per_alpha: int = 1

    # 执行策略
    execution_mode: str = "parallel"
    estimated_time: Optional[str] = None

    # 多步支持
    steps: Optional[List[ExecutionStep]] = None
    post_process: Optional[List[Dict[str, Any]]] = None

    def is_comparison(self) -> bool:
        """是否为多步对比实验"""

    def get_step_count(self) -> int:
        """获取执行步骤数"""

    def to_display_list(self, lang: str = 'cn') -> List[Dict[str, Any]]:
        """转换为 UI 显示格式"""
```

---

## 构建函数

### build_execution_plan
```python
def build_execution_plan(config: Config) -> ExecutionPlan:
    """从 Config 对象构建执行计划"""
```

### build_execution_plan_from_dict
```python
def build_execution_plan_from_dict(
    config_dict: Dict[str, Any],
    execution_params: Optional[Dict[str, Any]] = None,
    comparison_steps: Optional[List[Dict[str, Any]]] = None,
    post_process: Optional[List[Dict[str, Any]]] = None
) -> ExecutionPlan:
    """从原始字典构建执行计划（用于 wizard.py）"""
```

---

## 使用示例

```python
from smf.core.execution_plan import build_execution_plan, ExecutionPlan

# 从配置构建
plan = build_execution_plan(config)

# 检查是否为对比实验
if plan.is_comparison():
    print(f"对比实验: {plan.get_step_count()} 步")

# 获取 UI 显示列表
display_items = plan.to_display_list(lang='cn')
for item in display_items:
    print(f"{item['label']}: {item['value']}")

# 多步计划示例
comparison_steps = [
    {"config": {"graph_key": "random"}, "label": "Random Graph"},
    {"config": {"graph_key": "dinic"}, "label": "Dinic Graph"},
]
post_process = [
    {"type": "merge_plot", "sources": [0, 1],
     "labels": ["Random", "Dinic"], "output": "comparison.png"}
]
plan = build_execution_plan_from_dict(
    config_dict, comparison_steps=comparison_steps, post_process=post_process
)
```

---

## 后处理格式

```python
post_process = [
    {
        "type": "merge_plot",
        "sources": [0, 1],           # 要合并的步骤索引
        "labels": ["Random", "Dinic"],
        "output": "comparison.png",
        "metric": "Q_Y"              # 要绘制的指标
    }
]
```

---

## 输入/输出

**输入**:
- `Config` 对象或配置字典
- 对比步骤和后处理配置（可选）

**输出**:
- `ExecutionPlan` 对象
- UI 显示列表

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "Execution plan preview for experiments",
    "when_to_use_en": "Before executing experiments to show what will run",
    "tags_en": ["plan", "preview", "execution", "comparison", "multi-step"],

    "purpose_zh": "实验执行计划预览",
    "when_to_use_zh": "执行前预览实验内容",
    "tags_zh": ["计划", "预览", "执行", "对比实验", "多步"],
}
```

---

*最后更新：2025年12月*
