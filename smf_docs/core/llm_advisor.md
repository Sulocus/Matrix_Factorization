# LLMAdvisor: LLM 配置建议

使用 Gemini 模型分析用户需求并生成实验配置

**模块 ID**: C6
**SMF 路径**: `smf/core/llm_advisor.py`

---

## 目的

将用户的自然语言描述转换为结构化的实验配置：
- 分析用户意图
- 检测可能遗漏的重要选项
- 生成完整配置
- 提供优化建议

---

## 功能说明

1. **智能模型选择**: Flash（快速）vs Pro（复杂推理）
2. **参数安全验证**: 检测危险参数（damping=0, N>30000）
3. **对比实验支持**: 自动生成 comparison_steps 和 post_process
4. **多轮对话**: 支持用户追问和配置修改
5. **回退解析**: LLM 失败时使用正则表达式提取参数

---

## 主要类

### AnalysisResult
```python
@dataclass
class AnalysisResult:
    understanding: str              # 中文理解摘要
    experiment_type: str            # standard|comparison|size_scaling|...
    specified: Dict[str, Any]       # 用户明确指定的参数
    inferred: Dict[str, Any]        # 推断的参数
    missing_important: List[Dict]   # 重要但缺失的选项
    config: Dict[str, Any]          # 完整配置字典
    plotting_config: Optional[Dict] # 绘图配置（混合任务）
    confidence: str                 # high|medium|low
    requirement_summary: Optional[Dict]  # 非参数化需求摘要
    execution_params: Optional[Dict]     # 执行参数（指标、绘图）
    comparison_steps: Optional[List]     # 对比实验步骤
    post_process: Optional[List]         # 后处理钩子
    switch_messages: Optional[List[str]] # API 切换消息
```

### ConfigAdvisor
```python
class ConfigAdvisor:
    def __init__(self):
        self.client = GeminiClient()
        self.knowledge = format_as_prompt_context()

    def analyze_request(self, user_input: str) -> AnalysisResult:
        """
        分析用户的自然语言请求

        智能模型选择:
        - Flash: 简单明确的请求
        - Pro: 复杂模糊的请求
        """

    def analyze_with_clarification(
        self,
        original_input: str,
        clarification: str,
        previous_result: Optional[AnalysisResult] = None
    ) -> AnalysisResult:
        """
        带用户追问的重新分析
        """

    def suggest_improvements(self, config: Dict) -> List[Dict]:
        """建议配置优化"""

    def explain_config(self, config: Dict) -> str:
        """生成配置的中文说明"""
```

---

## 智能模型选择

### 使用 Flash（快速）的场景
- 明确的算法指定：`bigamp`, `agd`
- 明确的参数：`N=1000`, `M=50`
- 标准实验类型：`标准`, `基准`, `默认`

### 使用 Pro（推理）的场景
- 模糊描述：`可能`, `也许`, `看情况`
- 条件逻辑：`如果`, `当...时`
- 对比分析：`对比`, `比较`, `为什么`
- 高级功能：`replica`, `loop`, `有限尺寸`

---

## 使用示例

```python
from smf.core.llm_advisor import get_config_advisor, analyze_user_request

# 方式 1: 便捷函数
result = analyze_user_request("运行 N=1000 的 bigamp 实验，对比 random 和 dinic 图")

# 方式 2: 使用类
advisor = get_config_advisor()
result = advisor.analyze_request("正交教师，alpha 扫到 3.0")

# 访问结果
print(result.understanding)  # "使用正交教师模型，alpha 范围 0-3.0"
print(result.config)         # {"N1": 200, "teacher_key": "orthogonal", ...}
print(result.experiment_type)  # "standard" 或 "comparison"

# 对比实验
if result.comparison_steps:
    for step in result.comparison_steps:
        print(f"步骤: {step['label']}, 配置: {step['config']}")

# 用户追问
result2 = advisor.analyze_with_clarification(
    original_input="运行实验",
    clarification="把 damping 设为 0.8",
    previous_result=result
)
```

---

## 参数安全边界

只对真正危险的参数发出警告：

| 参数 | 危险值 | 警告 |
|------|--------|------|
| `damping` | 0 | BiG-AMP 必定发散 |
| `N1/N2` | >30000 | 几乎必定 OOM |

**注意**: 不警告 N>5000 或 alpha>5，用户被视为专家。

---

## 回退解析

当 LLM API 失败时，使用正则表达式提取：
- 算法：`bigamp`, `agd`, `梯度`
- 图结构：`dinic`, `双正则`, `low_loop`
- 教师：`orthogonal`, `正交`
- 尺寸：`N=xxx`, `Nxxx`, `xxxxx`
- Alpha：`alpha 到 x.x`, `步长 0.1`

---

## 输入/输出

**输入**:
- 用户自然语言描述
- 追问/修改请求

**输出**:
- `AnalysisResult` 对象
- 完整配置字典
- 对比实验步骤（可选）

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "LLM-powered experiment configuration from natural language",
    "when_to_use_en": "When user describes experiment in natural language",
    "tags_en": ["llm", "gemini", "config", "nlp", "advisor", "comparison"],

    "purpose_zh": "使用 LLM 从自然语言生成实验配置",
    "when_to_use_zh": "用户用自然语言描述实验时",
    "tags_zh": ["LLM", "Gemini", "配置", "自然语言", "建议", "对比"],
}
```

---

*最后更新：2025年12月*
