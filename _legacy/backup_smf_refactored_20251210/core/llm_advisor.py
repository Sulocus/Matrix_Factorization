"""
LLM-powered configuration advisor.

Uses Gemini Pro model for complex reasoning:
- Analyze user requirements
- Detect potentially missing options
- Generate complete configurations
- Provide recommendations
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .llm_filter import GeminiClient, GeminiModel
from .module_knowledge import format_as_prompt_context, ALGORITHMS, GRAPHS, EXPERIMENT_TYPES
from .llm_logger import get_logger


# ============================================================
# Smart Model Selection (智能模型选择)
# ============================================================

# Keywords that indicate a SIMPLE request (use Flash)
SIMPLE_KEYWORDS = {
    # Clear algorithm specification
    'bigamp', 'amp', 'big-amp', 'agd', '梯度',
    # Clear parameter specification
    'n=', 'n1=', 'n2=', 'm=', 'alpha', 'α',
    # Standard experiment types
    '标准', '基准', '默认', '普通', '正常',
    # Clear numeric patterns
    'n10', 'n20', 'n50', 'n100', 'n200', 'n500', 'n1000', 'n2000', 'n5000', 'n10000',
    # Common compact patterns
    ',m', ' m', '步',
    # Graph types
    'random', 'dinic', 'low_loop', 'low-loop', '随机', '双正则', '低循环',
    # Teacher types
    'orthogonal', '正交', 'standard', 'scaled',
}

# Keywords that indicate a COMPLEX request (use Pro)
COMPLEX_KEYWORDS = {
    # Ambiguous descriptions
    '可能', '也许', '大概', '估计', '不确定', '看情况',
    '或者', '或', '要么...要么', '还是',
    # Conditional logic
    '如果', '假如', '若', '当...时',
    # Comparison/analysis requests
    '对比', '比较', '分析', '为什么', '怎么',
    # Multi-task requests
    '然后', '之后', '接着', '再',
    # Advanced features
    'replica', '副本', 'loop', '环', 'scaling', '有限尺寸',
    # Plotting with complex requirements
    '色带', '误差带', '图例位置', '双Y轴', '子图',
}


def _should_use_pro(user_input: str) -> Tuple[bool, str]:
    """
    Determine if Pro model is needed based on request complexity.

    Returns:
        Tuple of (use_pro: bool, reason: str)
    """
    text = user_input.lower()

    # Count keyword matches
    simple_count = sum(1 for kw in SIMPLE_KEYWORDS if kw in text)
    complex_count = sum(1 for kw in COMPLEX_KEYWORDS if kw in text)

    # Check for clear numeric parameters
    has_clear_n = bool(re.search(r'n\s*[=:]\s*\d+|^n\d+|[,\s]n\d+', text, re.IGNORECASE))
    has_clear_m = bool(re.search(r'm\s*[=:]\s*\d+|[,\s]m\d+', text, re.IGNORECASE))
    has_clear_params = has_clear_n and has_clear_m

    # Decision logic:
    # 1. Many complex keywords → Pro
    if complex_count >= 2:
        return True, f"复杂请求 (complex_kw={complex_count})"

    # 2. Ambiguous with few simple keywords → Pro
    if simple_count < 2 and not has_clear_params:
        return True, f"参数不明确 (simple_kw={simple_count}, params={has_clear_params})"

    # 3. Has complex keywords but also clear params → Flash
    if complex_count == 1 and has_clear_params:
        return False, f"有复杂词但参数清晰 (params={has_clear_params})"

    # 4. Simple request with clear keywords → Flash
    if simple_count >= 2 or has_clear_params:
        return False, f"简单请求 (simple_kw={simple_count}, params={has_clear_params})"

    # 5. Default: use Flash for efficiency
    return False, "默认使用 Flash"


# ============================================================
# Parameter Safety Boundaries (参数安全边界)
# ============================================================

PARAMETER_BOUNDS = {
    # Only warn for truly critical issues, not "soft" suggestions
    # N size warnings removed - user knows what they're doing

    # Only warn for damping=0 which will definitely crash
    'damping': {
        'critical_min': 0.0,
        'critical_warning': 'damping=0 无阻尼，BiG-AMP 必定发散'
    },

    # Only warn for OOM-level sizes (N > 30000 with typical M)
    # 30000 x 30000 x 100 x 4 bytes ≈ 36GB - truly dangerous
    'N1': {
        'hard_max': 30000,
        'warning_max': 'N1 > 30000 几乎必定爆显存 (需要 >36GB)'
    },
    'N2': {
        'hard_max': 30000,
        'warning_max': 'N2 > 30000 几乎必定爆显存 (需要 >36GB)'
    },
}


@dataclass
class AnalysisResult:
    """Result of user request analysis."""
    understanding: str
    experiment_type: str
    specified: Dict[str, Any]
    inferred: Dict[str, Any]
    missing_important: List[Dict[str, str]]
    config: Dict[str, Any]
    plotting_config: Optional[Dict[str, Any]] = None  # For mixed tasks
    confidence: str = 'medium'  # 'high', 'medium', 'low'
    requirement_summary: Optional[Dict[str, Any]] = None  # Non-parametric requirements
    execution_params: Optional[Dict[str, Any]] = None  # LLM-generated execution parameters
    # Multi-step comparison support
    comparison_steps: Optional[List[Dict[str, Any]]] = None  # For comparison experiments
    # comparison_steps format: [{"config": {...}, "label": "Random Graph"}, ...]
    post_process: Optional[List[Dict[str, Any]]] = None  # Post-processing hooks
    # post_process format: [{"type": "merge_plot", "sources": [0,1], "labels": [...], "output": "..."}]
    # API switch tracking
    switch_messages: Optional[List[str]] = None  # Messages when API/model was switched


class ConfigAdvisor:
    """
    AI-powered configuration advisor.

    Uses Pro model for:
    - Understanding user intent
    - Detecting missing options that might be important
    - Generating complete configurations
    """

    def __init__(self):
        self.client = GeminiClient()
        self.knowledge = format_as_prompt_context()

    def analyze_request(self, user_input: str) -> AnalysisResult:
        """
        Analyze user's natural language request.

        Smart model selection:
        - Flash: Simple, clear requests with explicit parameters
        - Pro: Complex, ambiguous requests needing reasoning

        Args:
            user_input: User's description of what they want to do

        Returns:
            AnalysisResult with understanding, config, and suggestions
        """
        # Smart model selection
        use_pro, selection_reason = _should_use_pro(user_input)

        prompt = f"""You are an expert assistant for sparse matrix factorization experiments.
Analyze the user's request and generate a complete configuration.

=== Task ===
1. First determine if this is an EXPERIMENT task or a PLOTTING task:
   - PLOTTING: mentions 图、画、Y轴、颜色、放大、格式、保存、PDF、dpi、放一起、叠加、对比、误差、色带、图例 等
   - EXPERIMENT: mentions 跑、运行、实验、算法、N、M、alpha、扫描、训练 等
   注意: "把...放一起"、"...对比" 通常是绘图任务（多曲线叠加），不是运行新实验
2. Identify what they explicitly specified
3. Infer reasonable defaults for unspecified options
4. Detect important options they might have overlooked
5. Generate a complete configuration

=== User Request ===
"{user_input}"

=== Response Format ===
Return a JSON object:
{{
    "understanding": "Brief Chinese summary of what user wants",
    "requirement_summary": {{
        "purpose": "一句话总结实验目标（用户想验证/观察什么）",
        "comparison_focus": ["要对比的指标，使用精确模块名如 Q_Y, Q_Y_unobserved"],
        "analysis_angle": "关注哪些物理或数学性质？使用精确术语如 orthogonal, bigamp",
        "expected_outcome": "期望从实验看到什么结果？"
    }},
    "execution_params": {{
        "metrics_to_compute": ["Q_Y", "Q_W", "Q_X", ...],
        "plots": [
            {{
                "type": "comparison",
                "metrics": ["Q_Y", "Q_Y_unobserved"],
                "filename": "qy_comparison.png",
                "title": "Q_Y vs Q_Y_unobserved"
            }}
        ],
        "include_summary_plot": true,
        "include_qy_plot": true
    }},
    "experiment_type": "standard|comparison|size_scaling|init_scale|replica|plotting|mixed",
    "specified": {{
        "items user explicitly mentioned": "values"
    }},
    "inferred": {{
        "items inferred with defaults": "values"
    }},
    "missing_important": [
        {{
            "option": "option name",
            "reason": "Chinese explanation of why this might matter",
            "suggestion": "suggested value"
        }}
    ],
    "config": {{
        "algorithm_key": "bigamp",
        "graph_key": "random",
        "teacher_key": "standard",
        "N1": 200,
        "N2": 200,
        "M": 50,
        "alpha_start": 0.0,
        "alpha_stop": 4.0,
        "alpha_step": 0.1,
        "max_steps": 5000,
        "samples_per_alpha": 1,
        "resample_mask": true
    }},
    "comparison_steps": null,
    "post_process": null,
    "confidence": "high|medium|low"
}}

=== CRITICAL INFERENCE RULES (推断规则) ===
1. If user says "N=xxx" or "Nxxx" WITHOUT specifying N1/N2 separately → N1=N2=xxx (方阵)
2. NEVER ask for confirmation of N1/N2 if user only gave one N value
3. NEVER add warnings for N>5000 - user knows what they're doing
4. ONLY add warnings for truly FATAL issues:
   - damping=0 (BiG-AMP will crash)
   - N>30000 (will definitely OOM)
5. resample_mask controls whether each trial uses a DIFFERENT random mask:
   - true (default): Each trial resamples the observation mask (不同图/mask)
   - false: All trials share the SAME mask (相同图/mask)
   - If user says "不同的图"/"每次随机生成"/"resample" → resample_mask=true
   - If user says "同一张图"/"fixed mask"/"相同mask" → resample_mask=false
   - ALWAYS show this parameter in the config display

=== Module Terminology (模块标准术语) ===
When describing metrics, algorithms, graphs, or teachers in requirement_summary,
ALWAYS use the exact module names instead of user's colloquial terms:

Metrics (指标):
- Q_Y: 完整矩阵重合度 (用户可能说: "std QY", "标准QY", "完整重合度")
- Q_Y_unobserved: 未观测位置重合度 (用户可能说: "unobserved QY", "泛化重合度")
- Q_Y_observed: 已观测位置重合度
- Q_W: 左因子重合度 (用户可能说: "W的重合度")
- Q_X: 右因子重合度
- Q_W_prime: 归一化左因子重合度
- Q_X_prime: 归一化右因子重合度

Teachers (教师模型):
- standard: 标准高斯 (用户可能说: "普通", "默认")
- orthogonal: 正交初始化 (用户可能说: "正交教师", "正交W")
- scaled_variance: 缩放方差

Algorithms (算法):
- bigamp: BiG-AMP 消息传递
- agd: 交替梯度下降

Graphs (图结构):
- random: 随机图
- dinic: 双正则图 (Dinic算法生成)
- low_loop: 低循环图

=== execution_params Rules (执行参数规则) ===
The execution_params field controls WHAT gets computed and plotted. This is CRITICAL:

1. metrics_to_compute: List ALL metrics that should be calculated
   - Default: ["Q_Y", "Q_W", "Q_X", "Q_W_prime", "Q_X_prime", "Gen_Error"]
   - If user asks for "Q_Y_unobserved", ADD it to the list
   - If user says "只要QY", use ["Q_Y"] only

2. plots: Define CUSTOM plots beyond the defaults
   - Use when user wants specific comparisons (e.g., "把QY和QY_unobserved画一起")
   - Each plot has: type, metrics (list), filename, title

3. include_summary_plot: Whether to generate default summary.png
   - Set to FALSE if user says "不要summary" or only wants specific plots

4. include_qy_plot: Whether to generate default qy_vs_alpha.png
   - Set to FALSE if user provides custom plot config with Q_Y

Example: User says "画QY和QY_unobserved对比图"
→ execution_params = {{
    "metrics_to_compute": ["Q_Y", "Q_Y_unobserved"],
    "plots": [{{
        "type": "comparison",
        "metrics": ["Q_Y", "Q_Y_unobserved"],
        "filename": "qy_comparison.png",
        "title": "Q_Y vs Q_Y_unobserved"
    }}],
    "include_summary_plot": false,
    "include_qy_plot": false
}}

=== COMPARISON EXPERIMENT RULES (对比实验规则) ===
When user says "对比", "比较", "vs", "和...比", "几种图/教师/算法":
→ Set experiment_type = "comparison"
→ Generate comparison_steps with MULTIPLE configs
→ Generate post_process with merge_plot

Example: User says "对比 random 和 dinic 图，N=1000，正交教师"
→ Response:
{{
    "experiment_type": "comparison",
    "config": {{...共享配置: N1=1000, N2=1000, teacher_key="orthogonal"...}},
    "comparison_steps": [
        {{"config": {{"graph_key": "random"}}, "label": "Random Graph"}},
        {{"config": {{"graph_key": "dinic"}}, "label": "Dinic Graph"}}
    ],
    "post_process": [
        {{"type": "merge_plot", "sources": [0, 1],
          "labels": ["Random Graph", "Dinic Graph"],
          "output": "graph_comparison.png"}}
    ]
}}

Example: User says "比较标准教师和正交教师"
→ Response:
{{
    "experiment_type": "comparison",
    "comparison_steps": [
        {{"config": {{"teacher_key": "standard"}}, "label": "Standard Teacher"}},
        {{"config": {{"teacher_key": "orthogonal"}}, "label": "Orthogonal Teacher"}}
    ],
    "post_process": [
        {{"type": "merge_plot", "sources": [0, 1],
          "labels": ["Standard Teacher", "Orthogonal Teacher"],
          "output": "teacher_comparison.png"}}
    ]
}}

IMPORTANT for comparison:
- comparison_steps contains ONLY the differing parameters (overrides)
- config contains the SHARED base configuration
- post_process MUST include merge_plot with correct sources and labels
- If NOT a comparison request, set comparison_steps=null and post_process=null

=== Important ===
- FIRST check if this is a plotting request (绘图) vs experiment request (实验)
- If plotting: set experiment_type="plotting" and use plotting config format
- If COMPARISON (user wants to compare different configs): set experiment_type="comparison"
- If MIXED task (run experiment THEN plot): set experiment_type="mixed"
- missing_important should be EMPTY for normal requests
- Only add to missing_important for FATAL issues (damping=0, N>30000)
- User is an expert, don't question their parameter choices
- Use Chinese for "understanding" explanations
- In requirement_summary, ALWAYS use exact module names (e.g., "Q_Y" not "std QY")
- execution_params controls ACTUAL execution - LLM must fill this correctly!

Return ONLY the JSON object, no markdown code blocks."""

        # Log user request
        logger = get_logger()
        logger.log_request(user_input)

        # Track switch messages for UI
        switch_messages = []

        try:
            # Use call_with_fallback for automatic model/key rotation
            from .llm_filter import GeminiModel
            start_model = GeminiModel.PRO if use_pro else GeminiModel.FLASH

            response, model_used, switch_messages = self.client.call_with_fallback(
                prompt,
                system_context=self.knowledge,
                temperature=0.2,
                max_tokens=4096,  # Always use 4096 for complex config responses
                start_model=start_model,
            )

            # Log raw response and parse result
            try:
                result = self._parse_response(response)
                logger.log_response(response, parse_success=True)
            except Exception as parse_err:
                logger.log_response(response, parse_success=False, parse_error=str(parse_err))
                raise

            # Post-process: validate and fix config keys
            config = result.get('config', self._default_config())
            config = self._validate_config_keys(config)
            post_warnings = self._post_process_validation(config)

            # Filter out LLM's non-critical warnings (N size, alpha range, etc)
            # Only keep truly fatal warnings
            llm_warnings = result.get('missing_important', [])
            critical_keywords = ['damping', 'crash', 'OOM', '爆显存', '发散', '30000']
            filtered_llm_warnings = [
                w for w in llm_warnings
                if any(kw in str(w.get('option', '')) + str(w.get('reason', ''))
                       for kw in critical_keywords)
            ]
            all_warnings = filtered_llm_warnings + post_warnings

            # Handle mixed tasks (experiment + plotting)
            plotting_config = result.get('plotting_config', None)
            experiment_type = result.get('experiment_type', 'standard')
            requirement_summary = result.get('requirement_summary', None)
            execution_params = result.get('execution_params', None)

            # Handle comparison experiments
            comparison_steps = result.get('comparison_steps', None)
            post_process = result.get('post_process', None)

            # Ensure execution_params has defaults if not provided
            if execution_params is None:
                execution_params = {
                    'metrics_to_compute': ['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error'],
                    'plots': [],
                    'include_summary_plot': True,
                    'include_qy_plot': True,
                }

            # Log parsed configuration
            config_summary = {
                'algorithm': config.get('algorithm_key', 'bigamp'),
                'graph': config.get('graph_key', 'random'),
                'teacher': config.get('teacher_key', 'standard'),
                'matrix': f"{config.get('N1', 200)}x{config.get('N2', 200)}, M={config.get('M', 50)}",
                'alpha': f"{config.get('alpha_start', 0)}-{config.get('alpha_stop', 4)}, step={config.get('alpha_step', 0.1)}",
                'steps': config.get('max_steps', 5000),
                'samples': config.get('samples_per_alpha', 1),
            }
            logger.log_parsed(
                experiment_type=experiment_type,
                config_summary=config_summary,
                comparison_steps=comparison_steps,
            )

            return AnalysisResult(
                understanding=result.get('understanding', ''),
                experiment_type=experiment_type,
                specified=result.get('specified', {}),
                inferred=result.get('inferred', {}),
                missing_important=all_warnings,
                config=config,
                plotting_config=plotting_config,
                confidence=result.get('confidence', 'medium'),
                requirement_summary=requirement_summary,
                execution_params=execution_params,
                comparison_steps=comparison_steps,
                post_process=post_process,
                switch_messages=switch_messages if switch_messages else None,
            )

        except Exception as e:
            # Fallback to regex-based parsing
            fallback_config = self._fallback_parse(user_input)
            post_warnings = self._post_process_validation(fallback_config)

            # Provide more helpful error message
            error_str = str(e)
            if "所有API尝试失败" in error_str:
                # All API attempts failed - include switch history
                error_msg = "所有API尝试失败，使用正则提取"
            elif "column 1 (char 0)" in error_str or "empty" in error_str.lower():
                # Empty response - likely rate limit
                error_msg = "API 返回空响应（可能是速率限制），使用正则提取"
            elif "429" in error_str or "rate" in error_str.lower():
                error_msg = "API 请求过于频繁，使用正则提取"
            elif "timeout" in error_str.lower():
                error_msg = "API 请求超时，使用正则提取"
            else:
                error_msg = f"LLM解析失败: {error_str[:80]}，使用正则提取"

            return AnalysisResult(
                understanding=error_msg,
                experiment_type='standard',
                specified={},
                inferred={},
                missing_important=post_warnings,
                config=fallback_config,
                confidence='low',
                execution_params={
                    'metrics_to_compute': ['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error'],
                    'plots': [],
                    'include_summary_plot': True,
                    'include_qy_plot': True,
                },
                switch_messages=switch_messages if switch_messages else None,
            )

    def analyze_with_clarification(
        self,
        original_input: str,
        clarification: str,
        previous_result: Optional[AnalysisResult] = None
    ) -> AnalysisResult:
        """
        Re-analyze request with user clarification.

        Uses Pro model for multi-turn conversation.

        Args:
            original_input: User's original description
            clarification: User's clarification/follow-up
            previous_result: Previous analysis result for context

        Returns:
            Updated AnalysisResult
        """
        context = ""
        prev_execution_params = None
        if previous_result:
            prev_execution_params = previous_result.execution_params
            context = f"""
=== Previous Understanding ===
{previous_result.understanding}

=== Previous Requirement Summary ===
{json.dumps(previous_result.requirement_summary, ensure_ascii=False, indent=2) if previous_result.requirement_summary else 'None'}

=== Previous Config ===
{json.dumps(previous_result.config, indent=2)}

=== Previous Execution Params (IMPORTANT - preserve unless user explicitly changes) ===
{json.dumps(previous_result.execution_params, ensure_ascii=False, indent=2) if previous_result.execution_params else 'None'}
"""

        prompt = f"""You are an expert assistant for sparse matrix factorization experiments.
The user provided additional clarification or asked a question. Re-analyze and update the configuration.

=== Original Request ===
"{original_input}"

=== User Clarification/Question ===
"{clarification}"
{context}
=== CONVERSATION STYLE (对话风格 - 非常重要!) ===
当用户提问时（如"damping是多少？推荐多少？"），你必须在 understanding 中：
1. **先直接回答问题**：给出当前值和推荐值
2. **再解释为什么**：简短说明原因
3. **如果用户要求修改**（如"那就设为0.8"），明确确认修改

示例:
- 用户问: "damping是多少？推荐多少？"
- 正确回答: "当前 damping=0.5（默认值）。为稳定收敛，推荐 0.7~0.9，建议使用 0.8。"
- 错误回答: "用户询问了damping参数，经过分析后决定..."（太啰嗦，没有对话感）

- 用户说: "那就设为0.8"
- 正确回答: "已将 damping 设为 0.8。实验配置已更新。"
- 错误回答: "用户确认将damping参数设置为0.8..."（没有对话感）

=== Task ===
1. **First, directly answer any question** the user asked
2. Update your understanding with CONVERSATIONAL style (not report style!)
3. Focus on what the user REALLY wants (non-parametric requirements)
4. Update the requirement_summary to reflect the clarified intent
5. Keep config parameters unless user explicitly changes them
6. Use exact module names in requirement_summary (see terminology below)
7. **CRITICAL**: PRESERVE the previous execution_params (especially metrics_to_compute) unless user explicitly asks to change metrics!

=== Module Terminology (模块标准术语) ===
ALWAYS use exact module names instead of user's colloquial terms:

Metrics: Q_Y, Q_Y_unobserved, Q_Y_observed, Q_W, Q_X, Q_W_prime, Q_X_prime
Teachers: standard, orthogonal, scaled_variance
Algorithms: bigamp, agd
Graphs: random, dinic, low_loop

=== Response Format ===
Return a JSON object with the same format as before:
{{
    "understanding": "Updated Chinese summary based on clarification",
    "requirement_summary": {{
        "purpose": "更新后的实验目标",
        "comparison_focus": ["使用精确模块名，如 Q_Y, Q_Y_unobserved"],
        "analysis_angle": "更新后的分析角度",
        "expected_outcome": "更新后的期望结果"
    }},
    "experiment_type": "standard|size_scaling|init_scale|replica|plotting|mixed",
    "specified": {{ ... }},
    "inferred": {{ ... }},
    "missing_important": [],
    "config": {{ ... }},
    "execution_params": {{
        "metrics_to_compute": ["PRESERVE from previous unless user changes"],
        "plots": [...],
        "include_summary_plot": false,
        "include_qy_plot": true
    }},
    "confidence": "high|medium|low"
}}

Return ONLY the JSON object, no markdown code blocks."""

        # Track switch messages for UI
        switch_messages = []

        try:
            # Use call_with_fallback for automatic model/key rotation (start with Pro for clarification)
            from .llm_filter import GeminiModel
            response, model_used, switch_messages = self.client.call_with_fallback(
                prompt,
                system_context=self.knowledge,
                temperature=0.2,
                max_tokens=4096,
                start_model=GeminiModel.PRO,
            )
            result = self._parse_response(response)

            config = result.get('config', self._default_config())
            config = self._validate_config_keys(config)
            post_warnings = self._post_process_validation(config)

            # Preserve previous execution_params if LLM didn't return new ones
            default_execution = {
                'metrics_to_compute': ['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error'],
                'plots': [],
                'include_summary_plot': True,
                'include_qy_plot': True,
            }
            if prev_execution_params:
                # Use previous execution_params as default (preserve user's original metric choices)
                default_execution = prev_execution_params
            execution_params = result.get('execution_params', default_execution)

            return AnalysisResult(
                understanding=result.get('understanding', ''),
                experiment_type=result.get('experiment_type', 'standard'),
                specified=result.get('specified', {}),
                inferred=result.get('inferred', {}),
                missing_important=post_warnings,
                config=config,
                plotting_config=result.get('plotting_config', None),
                confidence=result.get('confidence', 'medium'),
                requirement_summary=result.get('requirement_summary', None),
                execution_params=execution_params,
                switch_messages=switch_messages if switch_messages else None,
            )

        except Exception as e:
            # On failure, return previous result with error note
            if previous_result:
                return AnalysisResult(
                    understanding=f"{previous_result.understanding} (追问解析失败: {str(e)[:50]})",
                    experiment_type=previous_result.experiment_type,
                    specified=previous_result.specified,
                    inferred=previous_result.inferred,
                    missing_important=previous_result.missing_important,
                    config=previous_result.config,
                    plotting_config=previous_result.plotting_config,
                    confidence='low',
                    requirement_summary=previous_result.requirement_summary,
                    execution_params=previous_result.execution_params,
                    switch_messages=switch_messages if switch_messages else None,
                )
            # Fallback to regex
            fallback_config = self._fallback_parse(original_input + " " + clarification)
            return AnalysisResult(
                understanding=f"追问解析失败: {str(e)[:100]}",
                experiment_type='standard',
                specified={},
                inferred={},
                missing_important=[],
                config=fallback_config,
                confidence='low',
                execution_params={
                    'metrics_to_compute': ['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error'],
                    'plots': [],
                    'include_summary_plot': True,
                    'include_qy_plot': True,
                },
                switch_messages=switch_messages if switch_messages else None,
            )

    def suggest_improvements(self, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Suggest potential improvements to a configuration.

        Args:
            config: Current configuration

        Returns:
            List of suggestions with explanations
        """
        prompt = f"""Review this experiment configuration and suggest improvements.

=== Current Configuration ===
{json.dumps(config, indent=2)}

=== Task ===
Identify potential issues or improvements. Focus on:
1. Parameter combinations that might cause problems
2. Settings that seem unusual for the experiment type
3. Optimizations for speed or accuracy

Return a JSON array:
[
    {{
        "issue": "Brief description of issue",
        "suggestion": "Suggested change",
        "priority": "high|medium|low"
    }}
]

Only include meaningful suggestions. Return empty array [] if config looks good.
Return ONLY the JSON array."""

        try:
            response = self.client.call_flash(
                prompt,
                system_context=self.knowledge,
            )
            return self._parse_array_response(response)
        except Exception:
            return []

    def explain_config(self, config: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of a configuration.

        Args:
            config: Configuration to explain

        Returns:
            Chinese explanation of what this config will do
        """
        prompt = f"""Explain this experiment configuration in simple Chinese.

=== Configuration ===
{json.dumps(config, indent=2)}

=== Task ===
Write a brief (2-3 sentences) Chinese explanation of:
1. What kind of experiment this is
2. What the key parameters mean
3. What to expect from the results

Keep it concise and user-friendly."""

        try:
            return self.client.call_flash(prompt, system_context=self.knowledge)
        except Exception as e:
            return f"无法生成说明: {str(e)}"

    def _get_valid_keys(self):
        """
        Get valid keys dynamically from the module registry.

        This eliminates hardcoded lists - keys are read from registered modules.
        """
        try:
            from ..modules import (
                get_valid_algorithm_keys,
                get_valid_graph_keys,
                get_valid_teacher_keys,
            )
            return {
                'algorithms': get_valid_algorithm_keys(),
                'graphs': get_valid_graph_keys(),
                'teachers': get_valid_teacher_keys(),
            }
        except ImportError:
            # Fallback if modules not loaded (should not happen)
            return {
                'algorithms': {'bigamp', 'agd'},
                'graphs': {'random', 'uniform'},
                'teachers': {'standard'},
            }

    def _validate_config_keys(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix config keys - replace invalid keys with defaults.

        Keys are validated against the registry dynamically.
        """
        valid = self._get_valid_keys()

        # Fix algorithm_key
        alg = config.get('algorithm_key', 'bigamp')
        if alg not in valid['algorithms']:
            print(f"[LLM] Invalid algorithm '{alg}', using 'bigamp'")
            config['algorithm_key'] = 'bigamp'

        # Fix graph_key
        graph = config.get('graph_key', 'random')
        if graph not in valid['graphs']:
            print(f"[LLM] Invalid graph '{graph}', using 'random'")
            config['graph_key'] = 'random'

        # Fix teacher_key
        teacher = config.get('teacher_key', 'standard')
        if teacher not in valid['teachers']:
            print(f"[LLM] Invalid teacher '{teacher}', using 'standard'")
            config['teacher_key'] = 'standard'

        return config

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "algorithm_key": "bigamp",
            "graph_key": "random",
            "teacher_key": "standard",
            "N1": 200,
            "N2": 200,
            "M": 50,
            "alpha_start": 0.0,
            "alpha_stop": 4.0,
            "alpha_step": 0.1,
            "max_steps": 5000,
            "samples_per_alpha": 1,
        }

    def _fallback_parse(self, user_input: str) -> Dict[str, Any]:
        """
        Simple regex-based parsing when LLM fails.
        Extracts basic parameters from user input.
        """
        import re

        config = self._default_config()
        text = user_input.lower()

        # Algorithm detection
        if 'agd' in text or '梯度' in text:
            config['algorithm_key'] = 'agd'
            if config.get('max_steps') == 5000:  # Default for BiG-AMP
                config['max_epochs'] = 20000
                del config['max_steps']

        # Graph detection
        if 'dinic' in text or '双正则' in text or 'bi-regular' in text or 'biregular' in text:
            config['graph_key'] = 'dinic'
        elif 'low_loop' in text or 'low-loop' in text or '低循环' in text or 'no 4-cycle' in text or 'c4-free' in text:
            config['graph_key'] = 'low_loop'

        # Teacher detection
        if 'orthogonal' in text or '正交' in text:
            config['teacher_key'] = 'orthogonal'
        elif 'scaled' in text or '缩放' in text:
            config['teacher_key'] = 'scaled_variance'

        # N (matrix size) extraction - handle N10000, N=10000, N:10000
        n_patterns = [
            r'^n(\d+)',                 # N10000 at very start
            r'[,，\s]n(\d+)',           # N10000 after comma/space
            r'n\s*[=:]\s*(\d+)',        # N=10000 or N:10000
            r'n1?\s*[=:]\s*(\d+)',      # N1=10000
            r'n\s*是?\s*(\d+)',         # N是10000
            r'(\d+)\s*[xX×]\s*\d+',     # 100x100
        ]
        for pattern in n_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                n_val = int(match.group(1))
                config['N1'] = n_val
                config['N2'] = n_val
                break

        # M (rank) extraction - handle M100, M=100, M:100, M 100
        m_patterns = [
            r'[,，\s]m(\d+)',           # M100 or ,M100
            r'm\s*[=:]\s*(\d+)',        # M=100 or M:100
            r'm\s+(\d+)',               # M 100
            r'秩\s*[=:为是]?\s*(\d+)',
            r'rank\s*[=:]\s*(\d+)',
        ]
        for pattern in m_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                config['M'] = int(match.group(1))
                break

        # Alpha extraction
        # alpha_stop
        alpha_stop_patterns = [
            r'alpha\s*(?:扫)?到\s*(\d+(?:\.\d+)?)',
            r'alpha.*?到\s*(\d+(?:\.\d+)?)',
            r'到\s*(\d+(?:\.\d+)?)\s*$',
        ]
        for pattern in alpha_stop_patterns:
            match = re.search(pattern, user_input)
            if match:
                config['alpha_stop'] = float(match.group(1))
                break

        # alpha_start
        alpha_start_patterns = [
            r'alpha\s*从\s*(\d+(?:\.\d+)?)',
            r'从\s*(\d+(?:\.\d+)?)\s*到',
        ]
        for pattern in alpha_start_patterns:
            match = re.search(pattern, user_input)
            if match:
                config['alpha_start'] = float(match.group(1))
                break

        # alpha_step
        step_patterns = [
            r'步长\s*[=:为是]?\s*(\d+(?:\.\d+)?)',
            r'step\s*[=:]\s*(\d+(?:\.\d+)?)',
        ]
        for pattern in step_patterns:
            match = re.search(pattern, user_input)
            if match:
                config['alpha_step'] = float(match.group(1))
                break

        # max_steps / max_epochs
        steps_patterns = [
            r'(\d+)\s*步',
            r'steps?\s*[=:]\s*(\d+)',
            r'max_steps\s*[=:]\s*(\d+)',
        ]
        for pattern in steps_patterns:
            match = re.search(pattern, user_input)
            if match:
                val = int(match.group(1))
                if config['algorithm_key'] == 'agd':
                    config['max_epochs'] = val
                else:
                    config['max_steps'] = val
                break

        # damping
        damping_pattern = r'damping\s*[=:]\s*(\d+(?:\.\d+)?)'
        match = re.search(damping_pattern, user_input)
        if match:
            config['damping'] = float(match.group(1))

        return config

    def _post_process_validation(self, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Post-process validation: ONLY warn for truly critical issues.

        Only warns for:
        - damping=0 (will definitely crash)
        - N > 30000 (will definitely OOM on most GPUs)

        Does NOT warn for:
        - N > 5000 (user knows what they're doing)
        - alpha > 5 (user knows what they're doing)
        - max_steps < 100 (user knows what they're doing)
        """
        warnings = []

        for param, bounds in PARAMETER_BOUNDS.items():
            value = config.get(param)
            if value is None:
                continue

            # Critical value check (e.g., damping=0) - WILL crash
            if 'critical_min' in bounds and value == bounds['critical_min']:
                warnings.append({
                    'type': 'critical_warning',
                    'option': param,
                    'reason': bounds.get('critical_warning', ''),
                    'suggestion': '0.5',
                })
                continue

            # Hard upper limit check (e.g., N > 30000) - WILL OOM
            if 'hard_max' in bounds and value > bounds['hard_max']:
                warning_msg = bounds.get('warning_max', '')
                warnings.append({
                    'type': 'critical_warning',
                    'option': param,
                    'reason': warning_msg,
                    'suggestion': str(bounds['hard_max']),
                })

        return warnings

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with robust extraction."""
        if not response:
            raise ValueError("API returned empty response (rate limit?)")

        response = response.strip()
        if not response:
            raise ValueError("API returned empty response (rate limit?)")

        # Remove markdown code blocks
        if response.startswith("```"):
            lines = response.split('\n')
            # Find start line (skip ```json or ```)
            start_idx = 1
            # Find end ```
            end_idx = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            response = '\n'.join(lines[start_idx:end_idx])

        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Find balanced braces for proper JSON extraction
        try:
            start = response.find('{')
            if start == -1:
                raise ValueError("No JSON object found")

            depth = 0
            in_string = False
            escape_next = False
            end = start

            for i, c in enumerate(response[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if c == '\\':
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            json_str = response[start:end]
            return json.loads(json_str)

        except (json.JSONDecodeError, ValueError) as e:
            # Last resort: return default with error info
            raise ValueError(f"Cannot parse JSON: {str(e)[:50]}")

    def _parse_array_response(self, response: str) -> List[Dict[str, str]]:
        """Parse JSON array response from LLM."""
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
            result = json.loads(response)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []


# Global instance
_advisor: Optional[ConfigAdvisor] = None


def get_config_advisor() -> ConfigAdvisor:
    """Get or create global config advisor instance."""
    global _advisor
    if _advisor is None:
        _advisor = ConfigAdvisor()
    return _advisor


def analyze_user_request(user_input: str) -> AnalysisResult:
    """Convenience function to analyze user request."""
    return get_config_advisor().analyze_request(user_input)
