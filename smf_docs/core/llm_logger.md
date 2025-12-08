# LLMLogger: LLM 对话日志

记录 LLM 对话和实验执行状态的日志系统

**模块 ID**: C5
**SMF 路径**: `smf/core/llm_logger.py`

---

## 目的

追踪 LLM 交互的完整流程：
- 用户输入
- LLM 原始响应
- 解析后的配置
- 执行状态（成功/部分/失败）

---

## 功能说明

1. **分阶段记录**: request → response → parsed → execution
2. **JSONL 格式**: 每行一条日志，便于追加和解析
3. **会话管理**: 按会话 ID 组织日志文件
4. **格式化输出**: 支持人类可读的会话摘要

---

## 主要类

### LLMLogEntry
```python
@dataclass
class LLMLogEntry:
    timestamp: str
    session_id: str
    stage: str  # "request" | "response" | "parsed" | "execution"

    # Request 阶段
    user_input: Optional[str] = None

    # Response 阶段
    raw_response: Optional[str] = None
    parse_success: Optional[bool] = None
    parse_error: Optional[str] = None

    # Parsed 阶段
    experiment_type: Optional[str] = None
    config_summary: Optional[Dict[str, Any]] = None
    comparison_steps: Optional[List[Dict]] = None

    # Execution 阶段
    execution_status: Optional[str] = None  # "success" | "partial" | "failed"
    steps_completed: Optional[int] = None
    steps_total: Optional[int] = None
    error_message: Optional[str] = None
    result_paths: Optional[List[str]] = None
```

### LLMLogger
```python
class LLMLogger:
    LOG_DIR = Path("smf/logs/llm")

    def __init__(self, session_id: str = None):
        """session_id 自动生成如不提供"""

    def log_request(self, user_input: str):
        """记录用户请求"""

    def log_response(self, raw_response: str, parse_success: bool,
                     parse_error: str = None):
        """记录 LLM 响应"""

    def log_parsed(self, experiment_type: str, config_summary: Dict,
                   comparison_steps: List[Dict] = None):
        """记录解析后的配置"""

    def log_execution(self, status: str, steps_completed: int = 0,
                      steps_total: int = 1, error_message: str = None,
                      result_paths: List[str] = None):
        """记录执行状态"""

    @classmethod
    def list_sessions(cls, limit: int = 10) -> List[Path]:
        """列出最近的会话日志文件"""

    @classmethod
    def load_session(cls, session_id: str) -> List[LLMLogEntry]:
        """加载会话日志条目"""

    @classmethod
    def format_session_summary(cls, session_id: str) -> str:
        """格式化会话摘要用于显示"""
```

---

## 全局函数

```python
def get_logger(session_id: str = None) -> LLMLogger:
    """获取或创建当前会话的 logger"""

def reset_logger():
    """重置全局 logger（用于测试）"""
```

---

## 使用示例

```python
from smf.core.llm_logger import get_logger, LLMLogger

# 获取 logger
logger = get_logger()

# 记录用户请求
logger.log_request("运行 N=1000 的 bigamp 实验")

# 记录 LLM 响应
logger.log_response(raw_response, parse_success=True)

# 记录解析结果
logger.log_parsed(
    experiment_type="standard",
    config_summary={"N1": 1000, "algorithm": "bigamp"}
)

# 记录执行状态
logger.log_execution(
    status="success",
    steps_completed=1,
    steps_total=1,
    result_paths=["/path/to/result.json"]
)

# 查看会话历史
for session_file in LLMLogger.list_sessions(limit=5):
    session_id = session_file.stem.replace("session_", "")
    print(LLMLogger.format_session_summary(session_id))
```

---

## 文件结构

```
smf/logs/llm/
├── session_20251208_143022.jsonl
├── session_20251208_150512.jsonl
└── session_20251208_161234.jsonl
```

每个 `.jsonl` 文件包含多行 JSON，每行一个 `LLMLogEntry`。

---

## CLI 命令

```bash
smf log llm           # 查看最近的 LLM 会话日志
smf log llm --last 5  # 查看最近 5 个会话
```

---

## 输入/输出

**输入**:
- 用户输入文本
- LLM 响应
- 配置摘要
- 执行状态

**输出**:
- `.jsonl` 日志文件
- 格式化的会话摘要

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "Log LLM conversations and experiment execution status",
    "when_to_use_en": "Automatically used during smf wizard interactions",
    "tags_en": ["log", "llm", "session", "jsonl", "tracking"],

    "purpose_zh": "记录 LLM 对话和实验执行状态",
    "when_to_use_zh": "smf wizard 交互时自动使用",
    "tags_zh": ["日志", "LLM", "会话", "追踪"],
}
```

---

*最后更新：2025年12月*
