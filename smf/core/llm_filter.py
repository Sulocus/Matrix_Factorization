"""
LLM-based natural language result filtering using Gemini API.

Supports dual model selection:
- Flash (gemini-2.5-flash): Quick tasks - keyword filtering, simple validation
- Pro (gemini-2.5-pro): Complex tasks - user intent analysis, missing option detection
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Set, Literal
from enum import Enum


class GeminiModel(Enum):
    """Available Gemini models."""
    FLASH = "gemini-2.5-flash"
    PRO = "gemini-2.5-pro"


# ============================================================
# Experiment Characteristics Knowledge Base
# ============================================================

SIZE_KEYWORDS = {
    # Large matrix keywords → N >= 1000
    "大矩阵": {"n1_min": 1000},
    "大尺寸": {"n1_min": 1000},
    "large matrix": {"n1_min": 1000},
    "large": {"n1_min": 1000},
    "big matrix": {"n1_min": 1000},

    # Small matrix keywords → N <= 500
    "小矩阵": {"n1_max": 500},
    "小尺寸": {"n1_max": 500},
    "small matrix": {"n1_max": 500},
    "small": {"n1_max": 500},

    # Medium matrix keywords → 500 < N < 1000
    "中等": {"n1_min": 500, "n1_max": 1000},
    "medium": {"n1_min": 500, "n1_max": 1000},
}

EXPERIMENT_KNOWLEDGE = {
    "overlap_metrics": {
        "description": "标准 BiG-AMP 相变实验，绘制 Q_Y vs α 曲线",
        "keywords_cn": [
            "基准", "标准", "基础", "普通", "正常", "默认", "一般",
            "相变", "相变曲线", "相变点", "临界点", "α_c",
            "Q_Y", "重叠度", "overlap", "收敛",
            "单个", "单一配置", "一个尺寸",
        ],
        "keywords_en": [
            "baseline", "standard", "basic", "normal", "default", "regular",
            "phase transition", "transition", "critical", "alpha_c",
            "overlap", "convergence", "Q_Y",
            "single", "one size", "single configuration",
        ],
        "not_keywords": ["对比", "多个", "不同尺寸", "replica", "loop", "环"],
    },
    "size_scaling": {
        "description": "有限尺寸效应实验，对比不同 N 值下的 Q_Y 曲线（观察非热力学极限下的行为）",
        "keywords_cn": [
            "多尺寸", "不同尺寸", "尺寸对比", "尺寸比较", "N对比",
            "多条曲线", "同一图", "对比曲线", "曲线对比",
            "有限尺寸", "有限N", "finite size", "尺寸效应",
            "非热力学", "偏离极限",
            "scaling", "缩放行为", "尺度行为",
            "compareNM", "multi_size",
        ],
        "keywords_en": [
            "multi-size", "multiple sizes", "different sizes", "size comparison",
            "compare N", "varying N", "N comparison",
            "finite size", "finite-size effect", "size effect",
            "non-thermodynamic", "scaling behavior",
            "same plot", "multiple curves",
        ],
        "not_keywords": ["初始化", "variance", "replica", "loop", "环"],
    },
    "loop_free": {
        "description": "无短环实验，对比随机图与 C4-free 图的相变行为",
        "keywords_cn": [
            "环", "loop", "4-loop", "4环", "四环", "短环",
            "C4", "C4-free", "无环", "低环", "去环",
            "环最小化", "环消除", "环减少",
            "MCMC", "马尔可夫链",
            "图结构", "图论", "图的结构",
            "随机图对比", "结构对比",
            "no4loop", "no4", "noloop",
        ],
        "keywords_en": [
            "loop", "4-loop", "4loop", "short loop", "cycle",
            "C4", "C4-free", "loop-free", "low loop",
            "loop minimization", "loop reduction",
            "MCMC", "Markov chain",
            "graph structure", "structure comparison",
        ],
        "not_keywords": ["尺寸", "size", "replica", "初始化"],
    },
    "replica": {
        "description": "副本一致性实验，测量多个独立副本间的重叠度",
        "keywords_cn": [
            "副本", "replica", "复制品",
            "一致性", "唯一性", "唯一解",
            "多个解", "局部最优", "local optima",
            "副本重叠", "副本间", "replica overlap",
            "同一mask", "共享mask",
            "解的稳定性", "解的分布",
            "S=100", "100个", "多次训练",
        ],
        "keywords_en": [
            "replica", "replicate", "copy",
            "consistency", "uniqueness", "unique solution",
            "multiple solutions", "local optima", "local minimum",
            "replica overlap", "between replicas",
            "same mask", "shared mask",
            "solution stability", "solution distribution",
        ],
        "not_keywords": ["尺寸", "size", "loop", "环", "初始化"],
    },
    "init_scale": {
        "description": "初始化尺度实验，测试不同 k/√M 初始化对收敛的影响",
        "keywords_cn": [
            "初始化", "初始", "初值", "起始",
            "scale", "尺度", "缩放因子",
            "k/√M", "k", "方差", "variance",
            "spin初始化", "自旋初始化",
            "不同k", "k值对比",
            "归一化", "normalization",
        ],
        "keywords_en": [
            "initialization", "initial", "init", "starting",
            "scale", "scale factor", "scaling factor",
            "k/sqrt(M)", "variance",
            "spin initialization",
            "different k", "k values",
            "normalization",
        ],
        "not_keywords": ["尺寸", "size", "replica", "loop", "环", "大矩阵"],
    },
}


def _match_keywords(query: str, keywords: List[str]) -> int:
    """Count how many keywords match in the query (case-insensitive)."""
    query_lower = query.lower()
    count = 0
    for kw in keywords:
        if kw.lower() in query_lower:
            count += 1
    return count


def _has_negative_keywords(query: str, not_keywords: List[str]) -> bool:
    """Check if query contains negative keywords that exclude this type."""
    query_lower = query.lower()
    for kw in not_keywords:
        if kw.lower() in query_lower:
            return True
    return False


def _extract_size_keywords(query: str) -> Dict[str, int]:
    """Extract size filters from query using SIZE_KEYWORDS."""
    query_lower = query.lower()
    size_filters = {}

    for keyword, filters in SIZE_KEYWORDS.items():
        if keyword.lower() in query_lower:
            size_filters.update(filters)

    return size_filters


def smart_keyword_match(query: str) -> Dict[str, Any]:
    """
    Use keyword matching to determine filters from natural language.
    This is a fallback when LLM API is unavailable.
    """
    result = {}

    # 1. Extract SIZE filters first
    size_filters = _extract_size_keywords(query)
    result.update(size_filters)

    # 2. Match experiment TYPE
    best_type = None
    best_score = 0

    for exp_type, info in EXPERIMENT_KNOWLEDGE.items():
        if _has_negative_keywords(query, info.get("not_keywords", [])):
            continue

        score_cn = _match_keywords(query, info["keywords_cn"])
        score_en = _match_keywords(query, info["keywords_en"])
        total_score = score_cn + score_en

        if total_score > best_score:
            best_score = total_score
            best_type = exp_type

    if best_score >= 1:
        result["type"] = best_type

    return result


class GeminiClient:
    """
    Gemini API client with dual model support and API key rotation.

    Model Selection:
    - Flash: Quick keyword filtering, simple validation
    - Pro: User intent analysis, missing option detection, config generation

    Fallback Strategy:
    - Try free keys first (with rotation on rate limit)
    - Auto-switch to paid API when all free keys are rate limited
    """

    # API keys for rotation (free tier limits)
    FREE_API_KEYS = [
        "AIzaSyDQbCDTzWhVgU1DpCPIDms46yQsms-9WSE",  # Default free 1
        "AIzaSyBgR0L5kyX3VCXYPs7jcRcHWk6BBbzi2VM",  # Free 2
        "AIzaSyAlyXtVamnrQkX5q4n6j4dSl_wPA6rjWz8",  # Free 3
        "AIzaSyA7e40HymYKxDiaqkl2FgfvokeLf5Cgoz4",  # Free 4
    ]

    # Paid API key (fallback when free keys are rate limited)
    PAID_API_KEY = "AIzaSyCFXEXyqc5V1HEaBv-EQ2WzwqyfPwLp45s"

    # Keep backward compatibility
    API_KEYS = FREE_API_KEYS

    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
    _current_key_index = 0
    _using_paid = False  # Track if we've switched to paid API
    _rate_limited_keys = set()  # Track which free keys are rate limited

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", self.FREE_API_KEYS[0])

    @classmethod
    def rotate_key(cls):
        """Rotate to next API key, switch to paid if all free keys are rate limited."""
        if cls._using_paid:
            # Already using paid, just return it
            return cls.PAID_API_KEY

        # Mark current free key as rate limited
        current_key = cls.FREE_API_KEYS[cls._current_key_index]
        cls._rate_limited_keys.add(current_key)

        # Check if all free keys are rate limited
        if len(cls._rate_limited_keys) >= len(cls.FREE_API_KEYS):
            cls._using_paid = True
            return cls.PAID_API_KEY

        # Find next non-rate-limited free key
        for _ in range(len(cls.FREE_API_KEYS)):
            cls._current_key_index = (cls._current_key_index + 1) % len(cls.FREE_API_KEYS)
            next_key = cls.FREE_API_KEYS[cls._current_key_index]
            if next_key not in cls._rate_limited_keys:
                return next_key

        # All free keys tried, switch to paid
        cls._using_paid = True
        return cls.PAID_API_KEY

    @classmethod
    def get_current_key(cls):
        """Get current API key."""
        if cls._using_paid:
            return cls.PAID_API_KEY
        return cls.FREE_API_KEYS[cls._current_key_index]

    @classmethod
    def reset_rate_limits(cls):
        """Reset rate limit tracking (e.g., after some time has passed)."""
        cls._rate_limited_keys.clear()
        cls._using_paid = False
        cls._current_key_index = 0

    @classmethod
    def get_api_mode(cls, lang: str = 'cn') -> str:
        """Get current API mode label for display in spinner."""
        if cls._using_paid:
            return "(付费)" if lang == 'cn' else "(paid)"
        return "(免费)" if lang == 'cn' else "(free)"

    def call(
        self,
        prompt: str,
        model: GeminiModel = GeminiModel.FLASH,
        system_context: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Call Gemini API with the specified model.

        Args:
            prompt: The prompt to send
            model: FLASH for quick tasks, PRO for complex reasoning
            system_context: Optional system context (knowledge base, etc.)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens

        Returns:
            Generated text response
        """
        import urllib.request
        import urllib.error

        url = f"{self.API_BASE}/{model.value}:generateContent?key={self.api_key}"

        # Build content parts
        parts = []
        if system_context:
            parts.append({"text": f"=== Background Context ===\n{system_context}\n\n"})
        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        data = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        import time
        # Allow enough retries for all free keys + paid fallback
        max_retries = len(GeminiClient.FREE_API_KEYS) + 2  # 3 free + 2 paid attempts
        last_error = None

        for attempt in range(max_retries):
            # Use current key (may have been rotated)
            current_key = self.api_key if attempt == 0 else GeminiClient.get_current_key()
            current_url = f"{self.API_BASE}/{model.value}:generateContent?key={current_key}"
            current_request = urllib.request.Request(
                current_url,
                data=data,
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(current_request, timeout=30) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    candidates = result.get('candidates', [])
                    if not candidates:
                        raise Exception("No candidates in response")
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if not parts:
                        raise Exception("No parts in response")
                    return parts[0].get('text', '')
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8')
                last_error = f"API error {e.code}: {error_body}"
                if e.code == 429:  # Rate limit - rotate key immediately
                    GeminiClient.rotate_key()
                    # No wait - just try next key right away
                    continue
                raise Exception(last_error)
            except urllib.error.URLError as e:
                last_error = f"Network error: {e.reason}"
                time.sleep(3)
                continue
            except Exception as e:
                last_error = str(e)
                GeminiClient.rotate_key()  # Try different key
                time.sleep(3)
                continue

        raise Exception(f"Failed after {max_retries} retries: {last_error}")

    def call_flash(self, prompt: str, **kwargs) -> str:
        """Convenience method for Flash model (quick tasks)."""
        return self.call(prompt, model=GeminiModel.FLASH, **kwargs)

    def call_pro(self, prompt: str, **kwargs) -> str:
        """Convenience method for Pro model (complex reasoning)."""
        return self.call(prompt, model=GeminiModel.PRO, **kwargs)

    def call_with_fallback(
        self,
        prompt: str,
        system_context: str = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        on_switch: callable = None,
        start_model: GeminiModel = GeminiModel.FLASH,
    ) -> tuple:
        """
        Call API with automatic model fallback.

        Fallback order: Flash → Pro (each model tries all keys via call())
        Validates JSON completeness before returning.

        Args:
            prompt: The prompt to send
            system_context: Optional system context
            temperature: Sampling temperature
            max_tokens: Maximum output tokens (default 4096 for complex configs)
            on_switch: Optional callback(message: str) when switching models/keys
            start_model: Which model to start with (FLASH or PRO)

        Returns:
            Tuple of (response_text, model_used, switch_messages)
        """
        import time
        switch_messages = []

        # Define model order based on start_model
        if start_model == GeminiModel.PRO:
            models_to_try = [(GeminiModel.PRO, "Pro"), (GeminiModel.FLASH, "Flash")]
        else:
            models_to_try = [(GeminiModel.FLASH, "Flash"), (GeminiModel.PRO, "Pro")]

        last_error = None

        for idx, (model, model_name) in enumerate(models_to_try):
            # Reset rate limit tracking before each model attempt
            # so the new model can try all keys fresh
            GeminiClient.reset_rate_limits()

            try:
                is_paid = GeminiClient._using_paid
                key_type = "付费" if is_paid else "免费"

                result = self.call(
                    prompt,
                    model=model,
                    system_context=system_context,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Check for empty response
                if not result or not result.strip():
                    msg = f"{model_name}({key_type}) 返回空响应"
                    switch_messages.append(msg)
                    if on_switch:
                        on_switch(msg)
                    # Add switch message if there's another model to try
                    if idx < len(models_to_try) - 1:
                        next_model = models_to_try[idx + 1][1]
                        msg = f"切换到 {next_model} 模型..."
                        switch_messages.append(msg)
                        if on_switch:
                            on_switch(msg)
                    continue  # Try next model

                # Check for truncated JSON (incomplete response)
                stripped = result.strip()
                if stripped.startswith('{') and not stripped.endswith('}'):
                    msg = f"{model_name} 响应被截断"
                    switch_messages.append(msg)
                    if on_switch:
                        on_switch(msg)
                    # Add switch message if there's another model to try
                    if idx < len(models_to_try) - 1:
                        next_model = models_to_try[idx + 1][1]
                        msg = f"切换到 {next_model} 模型..."
                        switch_messages.append(msg)
                        if on_switch:
                            on_switch(msg)
                    continue  # Try next model

                # Valid response
                return result, model_name, switch_messages

            except Exception as e:
                error_str = str(e)
                last_error = e
                msg = f"{model_name} 错误: {error_str[:60]}"
                switch_messages.append(msg)
                if on_switch:
                    on_switch(msg)
                # Add switch message if there's another model to try
                if idx < len(models_to_try) - 1:
                    next_model = models_to_try[idx + 1][1]
                    msg = f"切换到 {next_model} 模型..."
                    switch_messages.append(msg)
                    if on_switch:
                        on_switch(msg)
                # Continue to next model
                continue

        # All attempts failed
        raise Exception(f"所有API尝试失败: {last_error}")


class GeminiFilter:
    """
    Natural language filter for experiment results using Gemini API.
    Uses Flash model for quick filtering tasks.
    """

    def __init__(self, api_key: str = None):
        self.client = GeminiClient(api_key)

    def parse_query(self, user_query: str, available_results: List[Dict]) -> Dict[str, Any]:
        """Parse natural language query into structured filter criteria."""
        # First try local keyword matching
        local_match = smart_keyword_match(user_query)
        local_type = local_match.get("type")

        # Extract numeric patterns locally
        local_criteria = self._extract_numeric_patterns(user_query)

        # Merge size filters
        for key in ['n1_min', 'n1_max', 'm_min', 'm_max']:
            if key in local_match and key not in local_criteria:
                local_criteria[key] = local_match[key]

        # Build context
        result_summary = self._summarize_results(available_results)
        knowledge_context = self._build_knowledge_context()

        prompt = f"""You are a filter parser for sparse matrix factorization experiment results.
Parse the user's natural language query into structured filter criteria.

=== Experiment Types Knowledge ===
{knowledge_context}

=== Available Results ===
{result_summary}

=== User Query ===
"{user_query}"

=== Task ===
Return a JSON object with these fields (use null for unspecified):
- algorithm: "bigamp" or "agd" or null
- graph: "random" or "uniform" or null
- n1_min: integer or null
- n1_max: integer or null
- m_min: integer or null
- m_max: integer or null
- mn_ratio: float or null
- type: one of ["overlap_metrics", "size_scaling", "loop_free", "replica", "init_scale"] or null
- explanation: brief Chinese explanation

=== CRITICAL ===
"大矩阵" is a SIZE filter (N>=1000), NOT a TYPE filter!
"size_scaling" means finite-size effect study, NOT large matrices!

Return ONLY the JSON object."""

        try:
            # Use Flash for quick filtering
            response = self.client.call_flash(prompt)
            criteria = self._parse_response(response)

            # Merge with local extraction
            if local_type and not criteria.get('type'):
                criteria['type'] = local_type
            for key in ['n1_min', 'n1_max', 'm_min', 'm_max', 'mn_ratio']:
                if local_criteria.get(key) and not criteria.get(key):
                    criteria[key] = local_criteria[key]

            return criteria

        except Exception as e:
            # Fallback to local matching
            return {
                "algorithm": None,
                "graph": None,
                "n1_min": local_criteria.get('n1_min'),
                "n1_max": local_criteria.get('n1_max'),
                "m_min": local_criteria.get('m_min'),
                "m_max": local_criteria.get('m_max'),
                "mn_ratio": local_criteria.get('mn_ratio'),
                "type": local_type,
                "explanation": self._generate_local_explanation(local_type, local_criteria),
                "fallback": True
            }

    def _extract_numeric_patterns(self, query: str) -> Dict[str, Any]:
        """Extract numeric constraints from query using regex."""
        criteria = {}

        # N >= xxx
        match = re.search(r'[Nn]\s*[>≥]=?\s*(\d+)', query)
        if match:
            criteria['n1_min'] = int(match.group(1))

        # N <= xxx
        match = re.search(r'[Nn]\s*[<≤]=?\s*(\d+)', query)
        if match:
            criteria['n1_max'] = int(match.group(1))

        # N = xxx
        match = re.search(r'[Nn]\s*=\s*(\d+)', query)
        if match:
            val = int(match.group(1))
            criteria['n1_min'] = val
            criteria['n1_max'] = val

        # 200x200 pattern
        match = re.search(r'(\d+)\s*[x×]\s*(\d+)', query)
        if match:
            val = int(match.group(1))
            criteria['n1_min'] = val
            criteria['n1_max'] = val

        # M = xxx
        match = re.search(r'[Mm]\s*=\s*(\d+)', query)
        if match:
            val = int(match.group(1))
            criteria['m_min'] = val
            criteria['m_max'] = val

        # M/N ratio
        match = re.search(r'[Mm]/[Nn]\s*=\s*([\d.]+)', query)
        if match:
            criteria['mn_ratio'] = float(match.group(1))

        return criteria

    def _build_knowledge_context(self) -> str:
        """Build knowledge context string for LLM prompt."""
        lines = []
        for exp_type, info in EXPERIMENT_KNOWLEDGE.items():
            lines.append(f"- {exp_type}: {info['description']}")
            sample_keywords = info['keywords_cn'][:5]
            lines.append(f"  关键词: {', '.join(sample_keywords)}")
        return '\n'.join(lines)

    def _generate_local_explanation(self, exp_type: Optional[str], criteria: Dict) -> str:
        """Generate explanation when using local fallback."""
        parts = []

        size_parts = []
        if criteria.get('n1_min'):
            size_parts.append(f"N≥{criteria['n1_min']}")
        if criteria.get('n1_max') and criteria.get('n1_max') != criteria.get('n1_min'):
            size_parts.append(f"N≤{criteria['n1_max']}")
        if criteria.get('m_min'):
            size_parts.append(f"M={criteria['m_min']}")

        if size_parts:
            parts.append(f"尺寸筛选: {', '.join(size_parts)}")

        if exp_type:
            info = EXPERIMENT_KNOWLEDGE.get(exp_type, {})
            parts.append(f"类型筛选: {info.get('description', exp_type)}")

        return '; '.join(parts) if parts else "未能解析查询"

    def _summarize_results(self, results: List[Dict]) -> str:
        """Create a summary of available results for context."""
        if not results:
            return "No results available."

        types = set()
        sizes = []
        m_values = set()

        for r in results:
            if r.get('type'):
                types.add(r['type'])

            n1 = r.get('N1')
            m = r.get('M')
            try:
                n1_val = int(str(n1).split('-')[0]) if n1 else None
                m_val = int(str(m).split('-')[0]) if m else None
                if n1_val:
                    sizes.append(n1_val)
                if m_val:
                    m_values.add(m_val)
            except ValueError:
                pass

        summary_parts = [
            f"Total: {len(results)} results",
            f"Types: {', '.join(types) if types else 'overlap_metrics'}",
        ]
        if sizes:
            summary_parts.append(f"N1 range: {min(sizes)} ~ {max(sizes)}")
        if m_values:
            summary_parts.append(f"M values: {', '.join(str(m) for m in sorted(m_values))}")

        return '\n'.join(summary_parts)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        response = response.strip()
        if response.startswith("```"):
            lines = response.split('\n')
            end_idx = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            response = '\n'.join(lines[1:end_idx])

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise Exception("Failed to parse JSON from response")

    def apply_filter(self, results: List[Dict], criteria: Dict[str, Any]) -> List[Dict]:
        """Apply parsed filter criteria to results."""
        if criteria.get('error'):
            return results

        filtered = results

        # Algorithm filter
        if criteria.get('algorithm'):
            alg = criteria['algorithm'].lower()
            filtered = [r for r in filtered
                       if alg in str(r.get('algorithm', '')).lower()]

        # Graph filter
        if criteria.get('graph'):
            g = criteria['graph'].lower()
            filtered = [r for r in filtered
                       if g in str(r.get('graph', '')).lower()]

        # Type filter
        if criteria.get('type'):
            t = criteria['type'].lower()
            type_aliases = {
                'loop_free': ['loop_free', 'no4loop', 'noloop', 'c4free'],
                'size_scaling': ['size_scaling', 'multi_size', 'finite_size'],
                'init_scale': ['init_scale', 'variance', 'variance_sweep'],
                'replica': ['replica', 'replica_overlap'],
                'overlap_metrics': ['overlap_metrics', 'standard', 'baseline'],
            }

            matching_types = set()
            for canonical, aliases in type_aliases.items():
                if t in aliases or t == canonical:
                    matching_types.update(aliases)
                    matching_types.add(canonical)

            if matching_types:
                filtered = [r for r in filtered
                           if any(alias in str(r.get('type', '')).lower()
                                  or alias in str(r.get('name', '')).lower()
                                  for alias in matching_types)]
            else:
                filtered = [r for r in filtered
                           if t in str(r.get('type', '')).lower()
                           or t in str(r.get('name', '')).lower()]

        # Size filters
        def get_n1(r):
            n1 = r.get('N1')
            if n1 is None:
                return None
            try:
                return int(str(n1).split('-')[0])
            except ValueError:
                return None

        def get_m(r):
            m = r.get('M')
            if m is None:
                return None
            try:
                return int(str(m).split('-')[0])
            except ValueError:
                return None

        if criteria.get('n1_min'):
            filtered = [r for r in filtered
                       if get_n1(r) is not None and get_n1(r) >= criteria['n1_min']]

        if criteria.get('n1_max'):
            filtered = [r for r in filtered
                       if get_n1(r) is not None and get_n1(r) <= criteria['n1_max']]

        if criteria.get('m_min'):
            filtered = [r for r in filtered
                       if get_m(r) is not None and get_m(r) >= criteria['m_min']]

        if criteria.get('m_max'):
            filtered = [r for r in filtered
                       if get_m(r) is not None and get_m(r) <= criteria['m_max']]

        # M/N ratio filter
        if criteria.get('mn_ratio'):
            target_ratio = criteria['mn_ratio']
            tolerance = 0.05

            def matches_ratio(r):
                n1 = get_n1(r)
                m = get_m(r)
                if n1 is None or m is None or n1 == 0:
                    return False
                actual_ratio = m / n1
                return abs(actual_ratio - target_ratio) / target_ratio <= tolerance

            filtered = [r for r in filtered if matches_ratio(r)]

        return filtered


# Global instances
_gemini_client: Optional[GeminiClient] = None
_gemini_filter: Optional[GeminiFilter] = None


def get_gemini_client() -> GeminiClient:
    """Get or create global Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def get_gemini_filter() -> GeminiFilter:
    """Get or create global Gemini filter instance."""
    global _gemini_filter
    if _gemini_filter is None:
        _gemini_filter = GeminiFilter()
    return _gemini_filter


def filter_with_llm(query: str, results: List[Dict]) -> tuple[List[Dict], str]:
    """
    Convenience function to filter results with natural language.

    Args:
        query: Natural language query
        results: List of result metadata

    Returns:
        Tuple of (filtered_results, explanation)
    """
    gemini = get_gemini_filter()
    criteria = gemini.parse_query(query, results)
    filtered = gemini.apply_filter(results, criteria)
    explanation = criteria.get('explanation', '')

    if criteria.get('fallback'):
        explanation = f"[本地匹配] {explanation}"

    return filtered, explanation
