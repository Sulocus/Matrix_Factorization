# LLMFilter: LLM 结果筛选

使用 Gemini API 进行自然语言结果筛选

**模块 ID**: C7
**SMF 路径**: `smf/core/llm_filter.py`

---

## 目的

提供基于 LLM 的实验结果筛选功能：
- 双模型支持（Flash/Pro）
- API Key 轮换和付费降级
- 本地关键词匹配回退

---

## 功能说明

1. **GeminiClient**: API 调用客户端，支持自动 Key 轮换
2. **GeminiFilter**: 自然语言查询解析和结果筛选
3. **智能回退**: API 失败时使用本地关键词匹配
4. **实验类型知识库**: 内置实验类型和关键词定义

---

## 主要类

### GeminiModel
```python
class GeminiModel(Enum):
    FLASH = "gemini-2.5-flash"  # 快速任务
    PRO = "gemini-2.5-pro"      # 复杂推理
```

### GeminiClient
```python
class GeminiClient:
    # API Key 轮换
    FREE_API_KEYS = [...]   # 免费 API Keys
    PAID_API_KEY = "..."    # 付费 API Key（降级）

    def __init__(self, api_key: str = None)

    def call(self, prompt: str, model: GeminiModel = GeminiModel.FLASH,
             system_context: str = None, temperature: float = 0.1,
             max_tokens: int = 2048) -> str:
        """调用 Gemini API"""

    def call_flash(self, prompt: str, **kwargs) -> str:
        """快速任务"""

    def call_pro(self, prompt: str, **kwargs) -> str:
        """复杂推理"""

    def call_with_fallback(self, prompt: str, ...) -> tuple:
        """带模型降级的调用
        返回: (response, model_used, switch_messages)
        """

    @classmethod
    def rotate_key(cls):
        """轮换到下一个 API Key"""

    @classmethod
    def get_api_mode(cls, lang: str = 'cn') -> str:
        """获取当前 API 模式标签（免费/付费）"""
```

### GeminiFilter
```python
class GeminiFilter:
    def __init__(self, api_key: str = None)

    def parse_query(self, user_query: str, available_results: List[Dict]) -> Dict:
        """
        将自然语言查询解析为结构化筛选条件

        返回:
            {
                "algorithm": "bigamp" | None,
                "graph": "random" | None,
                "n1_min": int | None,
                "n1_max": int | None,
                "m_min": int | None,
                "m_max": int | None,
                "type": "overlap_metrics" | "size_scaling" | ...,
                "explanation": "筛选说明"
            }
        """

    def apply_filter(self, results: List[Dict], criteria: Dict) -> List[Dict]:
        """应用筛选条件到结果列表"""
```

---

## 实验类型知识库

```python
EXPERIMENT_KNOWLEDGE = {
    "overlap_metrics": {
        "description": "标准 BiG-AMP 相变实验",
        "keywords_cn": ["基准", "标准", "相变", "Q_Y", ...],
        "keywords_en": ["baseline", "standard", "phase transition", ...],
    },
    "size_scaling": {
        "description": "有限尺寸效应实验",
        "keywords_cn": ["多尺寸", "有限尺寸", "scaling", ...],
    },
    "loop_free": {
        "description": "无短环实验",
        "keywords_cn": ["环", "C4-free", "低循环", ...],
    },
    "replica": {
        "description": "副本一致性实验",
        "keywords_cn": ["副本", "一致性", "唯一解", ...],
    },
    "init_scale": {
        "description": "初始化尺度实验",
        "keywords_cn": ["初始化", "k/√M", "方差", ...],
    },
}
```

---

## 使用示例

```python
from smf.core.llm_filter import (
    get_gemini_client, get_gemini_filter, filter_with_llm
)

# 方式 1: 便捷函数
filtered, explanation = filter_with_llm("大矩阵 bigamp 实验", results)
print(explanation)  # "尺寸筛选: N≥1000; 算法: bigamp"

# 方式 2: 使用类
gemini_filter = get_gemini_filter()
criteria = gemini_filter.parse_query("N=1000 的 dinic 图实验", results)
filtered = gemini_filter.apply_filter(results, criteria)

# 直接调用 Gemini
client = get_gemini_client()
response = client.call_flash("解释这个配置...")

# 带降级的调用
response, model_used, messages = client.call_with_fallback(
    prompt,
    start_model=GeminiModel.PRO
)
for msg in messages:
    print(msg)  # "Pro(免费) 返回空响应", "切换到 Flash 模型..."
```

---

## API Key 轮换策略

1. 从第一个免费 Key 开始
2. 遇到 429 (Rate Limit) 时轮换到下一个
3. 所有免费 Key 用尽后切换到付费 Key
4. UI 显示当前模式：`(免费)` 或 `(付费)`

---

## 本地回退

当 API 不可用时，使用 `smart_keyword_match()`:

```python
def smart_keyword_match(query: str) -> Dict[str, Any]:
    """
    基于关键词的本地匹配

    返回:
        {"type": "size_scaling", "n1_min": 1000, ...}
    """
```

---

## 输入/输出

**输入**:
- 自然语言查询
- 结果元数据列表

**输出**:
- 筛选条件字典
- 筛选后的结果列表
- 筛选说明

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "LLM-based natural language result filtering",
    "when_to_use_en": "When filtering experiment results with natural language",
    "tags_en": ["llm", "gemini", "filter", "nlp", "api", "fallback"],

    "purpose_zh": "基于 LLM 的自然语言结果筛选",
    "when_to_use_zh": "用自然语言筛选实验结果时",
    "tags_zh": ["LLM", "Gemini", "筛选", "自然语言", "API", "回退"],
}
```

---

*最后更新：2025年12月*
