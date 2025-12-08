# PlanExecutor: 计划执行器

执行 ExecutionPlan 的核心模块，支持单步和多步实验

**模块 ID**: C4
**SMF 路径**: `smf/core/plan_executor.py`

---

## 目的

作为 ExecutionPlan 和 runner.run_experiment() 之间的桥梁，处理：
- 单步执行（标准实验）
- 多步执行（对比实验，顺序运行多个配置）
- 后处理（合并绘图等）

---

## 功能说明

1. **单步执行**: 直接调用 run_experiment()
2. **多步执行**: 顺序执行每个步骤，保存各自结果
3. **后处理**: 执行 merge_plot 等后处理操作
4. **进度回调**: 支持自定义进度更新函数
5. **日志集成**: 自动记录执行状态到 LLM 日志

---

## 主要类

### PlanExecutor
```python
class PlanExecutor:
    def __init__(self, progress_callback=None):
        """
        Args:
            progress_callback: 可选回调函数 callback(step_idx, total, label)
        """

    def run(self, plan: ExecutionPlan, base_config: Optional[Config] = None) -> Dict[str, Any]:
        """
        执行计划

        对于单步计划：直接运行实验
        对于多步计划：顺序运行每步，然后执行后处理

        Args:
            plan: ExecutionPlan 对象
            base_config: 基础配置（多步计划必需）

        Returns:
            包含结果、路径和生成图表的字典
        """
```

---

## 返回值格式

### 单步执行
```python
{
    'result_path': Path,      # 结果文件路径
    'results': Dict,          # 实验结果
    'plots': List[Path],      # 生成的图表
}
```

### 多步执行
```python
{
    'type': 'comparison',
    'steps': int,                      # 步骤数
    'step_results': Dict[str, Dict],   # {label: results}
    'step_paths': List[str],           # 各步骤结果路径
    'comparison_dir': Path,            # 对比输出目录
    'post_process_results': [          # 后处理结果
        {
            'type': 'merge_plot',
            'path': Path,
            'labels': List[str],
            'metric': str,
        }
    ],
}
```

---

## 使用示例

```python
from smf.core.plan_executor import PlanExecutor, run_plan
from smf.core.execution_plan import build_execution_plan

# 方式 1: 使用类
executor = PlanExecutor(
    progress_callback=lambda idx, total, label: print(f"Step {idx+1}/{total}: {label}")
)
result = executor.run(plan, base_config)

# 方式 2: 使用便捷函数
result = run_plan(plan, base_config)

# 处理结果
if result.get('type') == 'comparison':
    print(f"对比实验完成: {result['steps']} 步")
    print(f"对比图: {result['comparison_dir']}")
else:
    print(f"结果: {result['result_path']}")
```

---

## 后处理支持

当前支持的后处理类型：

### merge_plot
合并多个步骤的结果到一张对比图：

```python
post_process = [
    {
        "type": "merge_plot",
        "sources": [0, 1],      # 步骤索引
        "labels": ["A", "B"],   # 图例标签
        "output": "comparison.png",
        "metric": "Q_Y"
    }
]
```

---

## 错误处理

执行器会自动记录执行状态：
- **success**: 所有步骤成功完成
- **partial**: 部分步骤完成后失败
- **failed**: 执行失败

日志通过 `llm_logger` 记录，可用 `smf log llm` 查看。

---

## 输入/输出

**输入**:
- `ExecutionPlan` 对象
- `Config` 对象（多步计划必需）
- 进度回调函数（可选）

**输出**:
- 执行结果字典
- 结果文件（JSON）
- 图表文件（PNG）

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "Execute single or multi-step experiment plans",
    "when_to_use_en": "When running experiments from ExecutionPlan",
    "tags_en": ["executor", "runner", "comparison", "multi-step", "post-process"],

    "purpose_zh": "执行单步或多步实验计划",
    "when_to_use_zh": "从 ExecutionPlan 运行实验时使用",
    "tags_zh": ["执行器", "运行", "对比实验", "多步", "后处理"],
}
```

---

*最后更新：2025年12月*
