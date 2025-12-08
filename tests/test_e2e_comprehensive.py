#!/usr/bin/env python3
"""
End-to-End Comprehensive Test Suite for SMF.

This test suite:
1. Runs tests ONE BY ONE in foreground
2. Actually executes the experiment (small scale)
3. Catches ALL errors: LLM parsing, config validation, runtime
4. Provides detailed analysis and fixes

30 test cases covering real user scenarios.
"""

import sys
import os
import time
import json
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPhase(Enum):
    LLM_PARSE = "LLM解析"
    CONFIG_VALID = "配置验证"
    RUNTIME = "实际运行"
    RESULT = "结果检查"


class TestStatus(Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


@dataclass
class E2ETestCase:
    """End-to-end test case."""
    id: str
    query: str
    description: str
    # Expected values (flexible matching)
    expect_config: Dict[str, Any] = field(default_factory=dict)
    expect_warnings: List[str] = field(default_factory=list)
    # Should this actually run? (small scale tests only)
    should_run: bool = True
    # Scale override for actual run (keep small!)
    run_override: Dict[str, Any] = field(default_factory=dict)


@dataclass
class E2ETestResult:
    """Comprehensive test result."""
    test_id: str
    status: TestStatus = TestStatus.FAIL  # Default to FAIL, update on success
    # Phase results
    llm_ok: bool = False
    config_ok: bool = False
    runtime_ok: bool = False
    result_ok: bool = False
    # Details
    understanding: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    # Timing
    llm_time: float = 0.0
    run_time: float = 0.0


# ============================================================
# 30 Test Cases - Real User Scenarios
# ============================================================

TEST_CASES = [
    # ========== Group 1: Basic Functionality (5) ==========
    E2ETestCase(
        id="E01",
        query="跑一个基础实验",
        description="最简请求，测试默认值",
        expect_config={"algorithm_key": "bigamp"},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 2.0, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E02",
        query="N=100, M=20, alpha从0到2",
        description="基础参数指定",
        expect_config={"N1": 100, "M": 20, "alpha_stop": 2.0},
        should_run=True,
        run_override={"max_steps": 100, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E03",
        query="用BiG-AMP算法，500步",
        description="算法和步数指定",
        expect_config={"algorithm_key": "bigamp", "max_steps": 500},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E04",
        query="用AGD，N=80，M=15",
        description="AGD算法测试",
        expect_config={"algorithm_key": "agd", "N1": 80, "M": 15},
        should_run=True,
        run_override={"max_epochs": 500,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E05",
        query="标准相变实验，N=60, M=12, alpha 0到3步长0.2",
        description="完整参数指定",
        expect_config={"N1": 60, "M": 12, "alpha_stop": 3.0, "alpha_step": 0.2},
        should_run=True,
        run_override={"max_steps": 100},
    ),

    # ========== Group 2: Boundary Tests (5) ==========
    E2ETestCase(
        id="E06",
        query="N=40的小矩阵测试",
        description="N低于边界（50），应有警告",
        expect_config={"N1": 40},
        expect_warnings=["小", "有限尺寸"],
        should_run=True,
        run_override={"M": 8, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E07",
        query="alpha扫到6看看",
        description="alpha超过边界（5），应有警告",
        expect_config={"alpha_stop": 6.0},
        expect_warnings=["alpha", "超", "范围"],
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100,
                      "alpha_start": 5.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E08",
        query="只跑50步快速测试",
        description="步数低于边界（100），应有警告",
        expect_config={"max_steps": 50},
        expect_warnings=["步", "收敛"],
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E09",
        query="M=100, N=80测试",
        description="M>N，应有警告",
        expect_config={"M": 100, "N1": 80},
        expect_warnings=["M", "N", "大于"],
        should_run=False,  # Invalid config, don't run
    ),
    E2ETestCase(
        id="E10",
        query="damping=0试试",
        description="damping=0，应有严重警告",
        expect_config={"damping": 0},
        expect_warnings=["damping", "发散"],
        should_run=False,  # Dangerous config
    ),

    # ========== Group 3: Implicit Expression (5) ==========
    E2ETestCase(
        id="E11",
        query="快速baseline测试",
        description="隐晦表达：快速→小参数",
        expect_config={},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 2.0, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E12",
        query="中等规模矩阵，看相变曲线",
        description="隐晦表达：中等规模",
        expect_config={},
        should_run=True,
        run_override={"N1": 100, "N2": 100, "M": 20, "max_steps": 200,
                      "alpha_start": 0.5, "alpha_stop": 2.0, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E13",
        query="精细扫描alpha，步长0.05",
        description="步长参数提取",
        expect_config={"alpha_step": 0.05},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 1.2},
    ),
    E2ETestCase(
        id="E14",
        query="研究有限尺寸效应",
        description="实验类型识别：size_scaling",
        expect_config={},  # Should recognize as size_scaling
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E15",
        query="对比两种算法的收敛速度",
        description="对比实验识别",
        expect_config={},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),

    # ========== Group 4: Parameter Combinations (5) ==========
    E2ETestCase(
        id="E16",
        query="N=100, M=25, BiG-AMP 1000步, alpha 0.5到2.5",
        description="多参数组合",
        expect_config={"N1": 100, "M": 25, "max_steps": 1000,
                       "alpha_start": 0.5, "alpha_stop": 2.5},
        should_run=True,
        run_override={"max_steps": 100, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E17",
        query="AGD算法，学习率0.01，10000个epoch",
        description="AGD特有参数",
        expect_config={"algorithm_key": "agd", "max_epochs": 10000},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_epochs": 200,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E18",
        query="使用uniform图，N=80",
        description="图类型指定",
        expect_config={"graph_key": "uniform", "N1": 80},
        should_run=True,
        run_override={"M": 15, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E19",
        query="3次trial取平均",
        description="多次采样",
        expect_config={"samples_per_alpha": 3},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 50,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E20",
        query="damping=0.3更保守一点",
        description="damping调整",
        expect_config={"damping": 0.3},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),

    # ========== Group 5: Error Handling (5) ==========
    E2ETestCase(
        id="E21",
        query="",
        description="空输入处理",
        expect_config={},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 50,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E22",
        query="N=abc, M=xyz",
        description="无效参数处理",
        expect_config={},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 50,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E23",
        query="用XXX算法跑",
        description="未知算法处理",
        expect_config={"algorithm_key": "bigamp"},  # Should fallback
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 50,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E24",
        query="alpha从5到1",
        description="反向范围处理",
        expect_config={},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 50,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E25",
        query="N=-100",
        description="负数参数处理",
        expect_config={},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 50,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),

    # ========== Group 6: Scientific Scenarios (5) ==========
    E2ETestCase(
        id="E26",
        query="相变点附近精细扫描，alpha 1.0到1.5，步长0.02",
        description="精细扫描科学场景",
        expect_config={"alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.02},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100},
    ),
    E2ETestCase(
        id="E27",
        query="小矩阵快速验证，N=30，500步",
        description="玩具模型验证",
        expect_config={"N1": 30, "max_steps": 500},
        expect_warnings=["小", "有限"],
        should_run=True,
        run_override={"M": 6, "max_steps": 50,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E28",
        query="noise_var=0.1加噪声实验",
        description="噪声参数",
        expect_config={"noise_var": 0.1},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E29",
        query="scaled_variance教师测试",
        description="教师类型指定",
        expect_config={"teacher_key": "scaled_variance"},
        should_run=True,
        run_override={"N1": 50, "N2": 50, "M": 10, "max_steps": 100,
                      "alpha_start": 1.0, "alpha_stop": 1.5, "alpha_step": 0.5},
    ),
    E2ETestCase(
        id="E30",
        query="完整实验：N=80, M=16, BiG-AMP 200步, damping=0.5, alpha 0.5到2步长0.1",
        description="完整复杂配置",
        expect_config={"N1": 80, "M": 16, "max_steps": 200, "damping": 0.5,
                       "alpha_start": 0.5, "alpha_stop": 2.0, "alpha_step": 0.1},
        should_run=True,
        run_override={"max_steps": 50, "alpha_step": 0.5},
    ),
]


def run_llm_phase(query: str) -> Tuple[bool, Dict, List, str, float]:
    """Run LLM parsing phase."""
    from smf.core.llm_advisor import analyze_user_request

    start = time.time()
    try:
        result = analyze_user_request(query)
        elapsed = time.time() - start

        return (
            True,
            result.config,
            result.missing_important,
            result.understanding,
            elapsed
        )
    except Exception as e:
        elapsed = time.time() - start
        return False, {}, [], f"LLM Error: {str(e)}", elapsed


def validate_config(config: Dict, run_override: Dict) -> Tuple[bool, Dict, List[str]]:
    """Validate and prepare config for running."""
    from smf.modules.registry import get_algorithm, get_graph, get_teacher

    errors = []

    # Merge with override for actual run
    run_config = {**config, **run_override}

    # Validate algorithm
    alg_key = run_config.get('algorithm_key', 'bigamp')
    try:
        get_algorithm(alg_key)
    except KeyError as e:
        errors.append(f"Invalid algorithm: {alg_key}")
        run_config['algorithm_key'] = 'bigamp'

    # Validate graph
    graph_key = run_config.get('graph_key', 'random')
    try:
        get_graph(graph_key)
    except KeyError as e:
        errors.append(f"Invalid graph: {graph_key}")
        run_config['graph_key'] = 'random'

    # Validate teacher
    teacher_key = run_config.get('teacher_key', 'standard')
    try:
        get_teacher(teacher_key)
    except KeyError as e:
        errors.append(f"Invalid teacher: {teacher_key}")
        run_config['teacher_key'] = 'standard'

    # Ensure required fields
    defaults = {
        'N1': 50, 'N2': 50, 'M': 10,
        'alpha_start': 0.0, 'alpha_stop': 2.0, 'alpha_step': 0.5,
        'max_steps': 100, 'samples_per_alpha': 1,
        'damping': 0.5,
    }
    for key, val in defaults.items():
        if key not in run_config or run_config[key] is None:
            run_config[key] = val

    return len(errors) == 0, run_config, errors


def run_experiment_phase(flat_config: Dict) -> Tuple[bool, float, List[str]]:
    """Actually run a small experiment."""
    from smf.runner import run_experiment
    from smf.core.config import Config

    errors = []
    start = time.time()

    try:
        # Convert flat dict to nested structure for Config
        nested_config = {
            'matrix': {
                'N1': flat_config.get('N1', 50),
                'N2': flat_config.get('N2', flat_config.get('N1', 50)),
                'M': flat_config.get('M', 10),
            },
            'alpha': {
                'start': flat_config.get('alpha_start', 0.0),
                'stop': flat_config.get('alpha_stop', 2.0),
                'step': flat_config.get('alpha_step', 0.5),
            },
            'training': {
                'max_steps': flat_config.get('max_steps', 100),
                'samples_per_alpha': flat_config.get('samples_per_alpha', 1),
                'seed': flat_config.get('seed', 42),
            },
            'algorithm': {
                'damping': flat_config.get('damping', 0.5),
                'learning_rate': flat_config.get('learning_rate', 0.01),
                'use_compile': False,  # Disable for testing to avoid CUDAGraphs issues
            },
            'algorithm_key': flat_config.get('algorithm_key', 'bigamp'),
            'graph_key': flat_config.get('graph_key', 'random'),
            'teacher_key': flat_config.get('teacher_key', 'standard'),
        }

        # Handle AGD specific config
        if nested_config['algorithm_key'] == 'agd':
            if 'max_epochs' in flat_config:
                nested_config['training']['max_steps'] = flat_config['max_epochs']

        config = Config.from_dict(nested_config)

        # Run with small scale
        results = run_experiment(config, save=False)
        elapsed = time.time() - start

        # Check results
        if results is None:
            errors.append("No results returned")
            return False, elapsed, errors

        return True, elapsed, []

    except Exception as e:
        elapsed = time.time() - start
        errors.append(f"Runtime error: {str(e)}")
        errors.append(traceback.format_exc()[-500:])  # Last 500 chars of traceback
        return False, elapsed, errors


def check_config_match(actual: Dict, expected: Dict) -> List[str]:
    """Check if actual config matches expected."""
    issues = []
    for key, exp_val in expected.items():
        act_val = actual.get(key)
        if act_val is None:
            issues.append(f"Missing: {key}")
        elif isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
            if abs(act_val - exp_val) > max(0.01, abs(exp_val) * 0.1):
                issues.append(f"{key}: got {act_val}, expected {exp_val}")
        elif act_val != exp_val:
            issues.append(f"{key}: got {act_val}, expected {exp_val}")
    return issues


def check_warnings(warnings: List[Dict], expected: List[str]) -> List[str]:
    """Check if expected warning keywords are present."""
    issues = []
    warning_text = " ".join([str(w) for w in warnings])

    for keyword in expected:
        if keyword not in warning_text:
            issues.append(f"Missing warning: '{keyword}'")

    return issues


def run_single_test(test: E2ETestCase) -> E2ETestResult:
    """Run a single end-to-end test."""
    result = E2ETestResult(test_id=test.id)
    all_errors = []

    print(f"\n{'='*60}")
    print(f"Test {test.id}: {test.description}")
    print(f"Query: \"{test.query[:50]}{'...' if len(test.query) > 50 else ''}\"")
    print(f"{'='*60}")

    # Phase 1: LLM Parsing
    print(f"\n[Phase 1] LLM解析...")
    llm_ok, config, warnings, understanding, llm_time = run_llm_phase(test.query)
    result.llm_time = llm_time
    result.understanding = understanding
    result.config = config
    result.warnings = warnings

    if llm_ok:
        result.llm_ok = True
        print(f"  ✓ 理解: {understanding[:60]}...")
        print(f"  ✓ 配置: N={config.get('N1')}, M={config.get('M')}, "
              f"algo={config.get('algorithm_key')}")
        print(f"  ✓ 警告数: {len(warnings)}")

        # Check config match
        config_issues = check_config_match(config, test.expect_config)
        if config_issues:
            print(f"  ! 配置偏差: {config_issues}")
            all_errors.extend(config_issues)

        # Check warnings
        warning_issues = check_warnings(warnings, test.expect_warnings)
        if warning_issues:
            print(f"  ! 警告缺失: {warning_issues}")
            all_errors.extend(warning_issues)
    else:
        print(f"  ✗ LLM解析失败: {understanding}")
        all_errors.append(f"LLM failed: {understanding}")

    # Phase 2: Config Validation
    print(f"\n[Phase 2] 配置验证...")
    config_ok, run_config, config_errors = validate_config(config, test.run_override)
    result.config_ok = config_ok

    if config_ok:
        print(f"  ✓ 配置有效")
    else:
        print(f"  ! 配置问题: {config_errors}")
        all_errors.extend(config_errors)
        # Still continue with fixed config

    # Phase 3: Actual Run (if enabled)
    if test.should_run:
        print(f"\n[Phase 3] 实际运行 (小规模)...")
        print(f"  运行配置: N={run_config.get('N1')}, M={run_config.get('M')}, "
              f"steps={run_config.get('max_steps', run_config.get('max_epochs'))}")

        runtime_ok, run_time, runtime_errors = run_experiment_phase(run_config)
        result.run_time = run_time
        result.runtime_ok = runtime_ok

        if runtime_ok:
            print(f"  ✓ 运行成功 ({run_time:.1f}s)")
            result.result_ok = True
        else:
            print(f"  ✗ 运行失败:")
            for err in runtime_errors[:2]:
                print(f"    {err[:100]}")
            all_errors.extend(runtime_errors)
    else:
        print(f"\n[Phase 3] 跳过运行 (配置不安全)")
        result.runtime_ok = True  # Skip is OK
        result.result_ok = True

    # Determine final status
    result.errors = all_errors
    if len(all_errors) == 0:
        result.status = TestStatus.PASS
    elif len(all_errors) <= 2 and result.runtime_ok:
        result.status = TestStatus.PARTIAL
    else:
        result.status = TestStatus.FAIL

    status_icon = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}[result.status.value]
    print(f"\n[结果] {status_icon} {result.status.value}")

    return result


def run_all_tests(start_from: int = 1, delay: float = 2.0) -> List[E2ETestResult]:
    """Run all tests sequentially."""
    results = []

    print(f"\n{'#'*60}")
    print(f"# SMF End-to-End Comprehensive Test")
    print(f"# Total: {len(TEST_CASES)} tests")
    print(f"# Starting from: {start_from}")
    print(f"{'#'*60}")

    for i, test in enumerate(TEST_CASES):
        if i + 1 < start_from:
            continue

        result = run_single_test(test)
        results.append(result)

        time.sleep(delay)

    return results


def print_summary(results: List[E2ETestResult]):
    """Print test summary."""
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    total = len(results)
    passed = sum(1 for r in results if r.status == TestStatus.PASS)
    partial = sum(1 for r in results if r.status == TestStatus.PARTIAL)
    failed = sum(1 for r in results if r.status == TestStatus.FAIL)

    print(f"Total:   {total}")
    print(f"Passed:  {passed} ({100*passed/total:.1f}%)")
    print(f"Partial: {partial} ({100*partial/total:.1f}%)")
    print(f"Failed:  {failed} ({100*failed/total:.1f}%)")

    if failed > 0:
        print(f"\nFailed Tests:")
        for r in results:
            if r.status == TestStatus.FAIL:
                print(f"  {r.test_id}: {r.errors[0][:60] if r.errors else 'Unknown'}")


def save_results(results: List[E2ETestResult], filename: str):
    """Save results to JSON."""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "test_id": r.test_id,
                "status": r.status.value,
                "llm_ok": r.llm_ok,
                "config_ok": r.config_ok,
                "runtime_ok": r.runtime_ok,
                "understanding": r.understanding,
                "config": r.config,
                "errors": r.errors,
            }
            for r in results
        ]
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMF E2E Comprehensive Test")
    parser.add_argument("--start", type=int, default=1, help="Start from test number")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between tests")
    parser.add_argument("--output", type=str, default="tests/e2e_results.json")
    args = parser.parse_args()

    results = run_all_tests(start_from=args.start, delay=args.delay)
    print_summary(results)
    save_results(results, args.output)
