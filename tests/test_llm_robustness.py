#!/usr/bin/env python3
"""
LLM Robustness Test Suite for SMF ConfigAdvisor.

30 comprehensive test cases covering:
- A: Plotting functions (5 cases)
- B: Physical scenarios (5 cases)
- C: LLM edge cases (5 cases)
- D: Boundary warnings (8 cases) - Focus area after fix
- E: Mixed tasks (4 cases) - Focus area after fix
- F: Smart follow-up (3 cases)

Each test validates:
1. Layer 1: LLM understanding (did it parse correctly?)
2. Layer 2: Feature mapping (did it choose right features?)
3. Layer 3: Boundary checking (did it generate warnings?)
"""

import sys
import os
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smf.core.llm_advisor import analyze_user_request, AnalysisResult


class TestStatus(Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


@dataclass
class TestCase:
    """A single test case."""
    id: str
    category: str
    query: str
    expected_type: str  # experiment_type
    check_config: Dict[str, Any]  # Keys to check in config
    check_warnings: List[str]  # Expected warning keywords
    description: str


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    status: TestStatus
    understanding: str
    experiment_type: str
    config: Dict[str, Any]
    warnings: List[Dict[str, str]]
    issues: List[str]


# ============================================================
# Test Cases (30 total)
# ============================================================

TEST_CASES = [
    # A: Plotting Functions (5 cases)
    TestCase(
        id="A1", category="Plotting",
        query="把Y轴放大到0.3看看细节",
        expected_type="plotting",
        check_config={"plotting_function": "ylim"},
        check_warnings=[],
        description="Y-axis zoom"
    ),
    TestCase(
        id="A2", category="Plotting",
        query="保存成PDF格式，300dpi",
        expected_type="plotting",
        check_config={"format": "pdf"},
        check_warnings=[],
        description="Save format"
    ),
    TestCase(
        id="A3", category="Plotting",
        query="把两个实验结果放一起对比",
        expected_type="plotting",
        check_config={"plotting_function": "multi_curve"},
        check_warnings=[],
        description="Multi-curve overlay"
    ),
    TestCase(
        id="A4", category="Plotting",
        query="用色带代替误差线",
        expected_type="plotting",
        check_config={"error_style": "band"},
        check_warnings=[],
        description="Error band style"
    ),
    TestCase(
        id="A5", category="Plotting",
        query="图例挡住曲线了，移到外面",
        expected_type="plotting",
        check_config={},
        check_warnings=[],
        description="Legend position"
    ),

    # B: Physical Scenarios (5 cases)
    TestCase(
        id="B1", category="Physics",
        query="研究有限尺寸效应，N从1000扫到5000",
        expected_type="size_scaling",
        check_config={"N_values": True},
        check_warnings=[],
        description="Finite size effect"
    ),
    TestCase(
        id="B2", category="Physics",
        query="用正交教师矩阵跑一下",
        expected_type="standard",
        check_config={"teacher_key": "orthogonal"},
        check_warnings=[],
        description="Orthogonal teacher"
    ),
    TestCase(
        id="B3", category="Physics",
        query="加点噪声看相变怎么变软",
        expected_type="standard",
        check_config={"noise_var": True},
        check_warnings=[],
        description="Noise effect on phase transition"
    ),
    TestCase(
        id="B4", category="Physics",
        query="用全精度FP32跑一遍对比",
        expected_type="standard",
        check_config={"use_bf16": False},
        check_warnings=[],
        description="FP32 precision"
    ),
    TestCase(
        id="B5", category="Physics",
        query="比较随机图和双正则图的相变",
        expected_type="standard",
        check_config={},
        check_warnings=[],
        description="Graph type comparison"
    ),

    # C: LLM Edge Cases (5 cases)
    TestCase(
        id="C1", category="LLM Edge",
        query="跑一下",
        expected_type="standard",
        check_config={},
        check_warnings=[],
        description="Minimal input"
    ),
    TestCase(
        id="C2", category="LLM Edge",
        query="N=500, M=100, alpha从0到3步长0.05，用BiG-AMP跑5000步",
        expected_type="standard",
        check_config={"N1": 500, "M": 100, "alpha_stop": 3, "max_steps": 5000},
        check_warnings=[],
        description="Fully specified"
    ),
    TestCase(
        id="C3", category="LLM Edge",
        query="快速baseline",
        expected_type="standard",
        check_config={},
        check_warnings=[],
        description="Shorthand request"
    ),
    TestCase(
        id="C4", category="LLM Edge",
        query="",
        expected_type="standard",
        check_config={},
        check_warnings=[],
        description="Empty input"
    ),
    TestCase(
        id="C5", category="LLM Edge",
        query="alpha=2.5附近做精细扫描，步长0.01",
        expected_type="standard",
        check_config={"alpha_step": 0.01},
        check_warnings=[],
        description="Fine-grained scan"
    ),

    # D: Boundary Warnings (8 cases) - FOCUS AREA
    TestCase(
        id="D1", category="Boundary",
        query="alpha扫到15",
        expected_type="standard",
        check_config={"alpha_stop": 15},
        check_warnings=["alpha", "范围", "超"],
        description="Alpha too large"
    ),
    TestCase(
        id="D2", category="Boundary",
        query="N=10000跑一下",
        expected_type="standard",
        check_config={"N1": 10000},
        check_warnings=["显存", "时间", "N"],
        description="N too large"
    ),
    TestCase(
        id="D3", category="Boundary",
        query="就跑1步看看",
        expected_type="standard",
        check_config={"max_steps": 1},
        check_warnings=["步数", "收敛"],
        description="Steps too few"
    ),
    TestCase(
        id="D4", category="Boundary",
        query="damping设成0试试",
        expected_type="standard",
        check_config={"damping": 0},
        check_warnings=["damping", "发散", "震荡"],
        description="Damping zero"
    ),
    TestCase(
        id="D5", category="Boundary",
        query="M=200, N=100",
        expected_type="standard",
        check_config={"M": 200, "N1": 100},
        check_warnings=["M", "N", "秩", "大于"],
        description="M > N"
    ),
    TestCase(
        id="D6", category="Boundary",
        query="N=20的小矩阵测试",
        expected_type="standard",
        check_config={"N1": 20},
        check_warnings=["有限尺寸", "小", "N"],
        description="N too small"
    ),
    TestCase(
        id="D7", category="Boundary",
        query="用AGD跑100个epoch",
        expected_type="standard",
        check_config={"max_epochs": 100},
        check_warnings=["epoch", "收敛", "AGD"],
        description="AGD epochs too few"
    ),
    TestCase(
        id="D8", category="Boundary",
        query="damping=0.95试试看",
        expected_type="standard",
        check_config={"damping": 0.95},
        check_warnings=["damping", "不稳定"],
        description="Damping too high"
    ),

    # E: Mixed Tasks (4 cases) - FOCUS AREA
    TestCase(
        id="E1", category="Mixed",
        query="跑完实验后画图分析",
        expected_type="mixed",
        check_config={},
        check_warnings=[],
        description="Experiment then plot"
    ),
    TestCase(
        id="E2", category="Mixed",
        query="N=500跑一个实验，完了保存PDF",
        expected_type="mixed",
        check_config={"N1": 500},
        check_warnings=[],
        description="Run and save PDF"
    ),
    TestCase(
        id="E3", category="Mixed",
        query="对比两个算法然后画对比图",
        expected_type="mixed",
        check_config={},
        check_warnings=[],
        description="Compare and plot"
    ),
    TestCase(
        id="E4", category="Mixed",
        query="做finite size实验然后叠加到同一张图",
        expected_type="mixed",
        check_config={},
        check_warnings=[],
        description="Size scaling then overlay"
    ),

    # F: Smart Follow-up (3 cases)
    TestCase(
        id="F1", category="Follow-up",
        query="刚才那个再跑一遍，但是alpha细一点",
        expected_type="standard",
        check_config={},
        check_warnings=[],
        description="Refine previous"
    ),
    TestCase(
        id="F2", category="Follow-up",
        query="换成AGD试试",
        expected_type="standard",
        check_config={"algorithm_key": "agd"},
        check_warnings=[],
        description="Switch algorithm"
    ),
    TestCase(
        id="F3", category="Follow-up",
        query="把步长改成0.02重跑",
        expected_type="standard",
        check_config={"alpha_step": 0.02},
        check_warnings=[],
        description="Adjust step size"
    ),
]


def run_single_test(test: TestCase, delay: float = 2.0) -> TestResult:
    """Run a single test case."""
    issues = []

    try:
        result = analyze_user_request(test.query)

        # Check experiment type
        if test.expected_type != "any":
            if result.experiment_type != test.expected_type:
                # Allow some flexibility
                if not (test.expected_type == "mixed" and result.experiment_type in ["standard", "plotting"]):
                    issues.append(f"Type mismatch: expected={test.expected_type}, got={result.experiment_type}")

        # Check config values
        for key, expected in test.check_config.items():
            actual = result.config.get(key)
            if expected is True:
                # Just check existence
                if actual is None:
                    issues.append(f"Missing config key: {key}")
            elif actual != expected:
                # Check if close enough for numeric values
                if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                    if abs(actual - expected) > 0.01:
                        issues.append(f"Config mismatch: {key}={actual}, expected={expected}")
                else:
                    issues.append(f"Config mismatch: {key}={actual}, expected={expected}")

        # Check warnings (most important for D category)
        if test.check_warnings:
            warning_texts = " ".join([
                w.get('reason', '') + w.get('option', '')
                for w in result.missing_important
            ])
            for keyword in test.check_warnings:
                if keyword not in warning_texts:
                    issues.append(f"Missing warning keyword: '{keyword}'")

        # Determine status
        if not issues:
            status = TestStatus.PASS
        elif len(issues) <= 2:
            status = TestStatus.PARTIAL
        else:
            status = TestStatus.FAIL

        return TestResult(
            test_id=test.id,
            status=status,
            understanding=result.understanding,
            experiment_type=result.experiment_type,
            config=result.config,
            warnings=result.missing_important,
            issues=issues,
        )

    except Exception as e:
        return TestResult(
            test_id=test.id,
            status=TestStatus.FAIL,
            understanding=f"Error: {str(e)}",
            experiment_type="error",
            config={},
            warnings=[],
            issues=[f"Exception: {str(e)}"],
        )
    finally:
        time.sleep(delay)


def run_all_tests(delay: float = 2.0) -> List[TestResult]:
    """Run all test cases."""
    results = []
    total = len(TEST_CASES)

    print(f"\n{'='*60}")
    print(f"SMF LLM Robustness Test Suite - 30 Cases")
    print(f"{'='*60}\n")

    for i, test in enumerate(TEST_CASES):
        print(f"[{i+1}/{total}] {test.id}: {test.description}...")
        result = run_single_test(test, delay)
        results.append(result)

        status_icon = {
            TestStatus.PASS: "✓",
            TestStatus.PARTIAL: "~",
            TestStatus.FAIL: "✗",
        }[result.status]

        print(f"  {status_icon} {result.status.value}")
        if result.issues:
            for issue in result.issues[:2]:  # Show max 2 issues
                print(f"    - {issue}")

    return results


def analyze_results(results: List[TestResult]) -> Dict[str, Any]:
    """Analyze test results by category."""
    categories = {}

    for result in results:
        test = next(t for t in TEST_CASES if t.id == result.test_id)
        cat = test.category

        if cat not in categories:
            categories[cat] = {"pass": 0, "partial": 0, "fail": 0, "total": 0}

        categories[cat]["total"] += 1
        if result.status == TestStatus.PASS:
            categories[cat]["pass"] += 1
        elif result.status == TestStatus.PARTIAL:
            categories[cat]["partial"] += 1
        else:
            categories[cat]["fail"] += 1

    # Calculate rates
    for cat, stats in categories.items():
        stats["pass_rate"] = (stats["pass"] + 0.5 * stats["partial"]) / stats["total"] * 100

    # Overall
    total_pass = sum(c["pass"] for c in categories.values())
    total_partial = sum(c["partial"] for c in categories.values())
    total_fail = sum(c["fail"] for c in categories.values())
    total = len(results)

    overall_rate = (total_pass + 0.5 * total_partial) / total * 100

    return {
        "categories": categories,
        "overall": {
            "pass": total_pass,
            "partial": total_partial,
            "fail": total_fail,
            "total": total,
            "pass_rate": overall_rate,
        }
    }


def print_summary(analysis: Dict[str, Any], results: List[TestResult]):
    """Print test summary."""
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}\n")

    # By category
    print("By Category:")
    print("-" * 50)
    print(f"{'Category':<15} {'Pass':<6} {'Partial':<8} {'Fail':<6} {'Rate':<8}")
    print("-" * 50)

    for cat, stats in analysis["categories"].items():
        print(f"{cat:<15} {stats['pass']:<6} {stats['partial']:<8} {stats['fail']:<6} {stats['pass_rate']:.1f}%")

    print("-" * 50)
    overall = analysis["overall"]
    print(f"{'OVERALL':<15} {overall['pass']:<6} {overall['partial']:<8} {overall['fail']:<6} {overall['pass_rate']:.1f}%")

    # Failed tests
    failed = [r for r in results if r.status == TestStatus.FAIL]
    if failed:
        print(f"\n\nFailed Tests ({len(failed)}):")
        print("-" * 50)
        for r in failed:
            test = next(t for t in TEST_CASES if t.id == r.test_id)
            print(f"  {r.test_id}: {test.description}")
            for issue in r.issues:
                print(f"    - {issue}")

    # Partial tests
    partial = [r for r in results if r.status == TestStatus.PARTIAL]
    if partial:
        print(f"\n\nPartial Tests ({len(partial)}):")
        print("-" * 50)
        for r in partial:
            test = next(t for t in TEST_CASES if t.id == r.test_id)
            print(f"  {r.test_id}: {test.description}")
            for issue in r.issues[:1]:  # Just first issue
                print(f"    - {issue}")


def save_results(results: List[TestResult], analysis: Dict[str, Any], filename: str):
    """Save results to file."""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis": analysis,
        "results": [
            {
                "test_id": r.test_id,
                "status": r.status.value,
                "understanding": r.understanding,
                "experiment_type": r.experiment_type,
                "config": r.config,
                "warnings": r.warnings,
                "issues": r.issues,
            }
            for r in results
        ],
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMF LLM Robustness Test")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls (seconds)")
    parser.add_argument("--output", type=str, default="test_results.json", help="Output file")
    args = parser.parse_args()

    results = run_all_tests(delay=args.delay)
    analysis = analyze_results(results)
    print_summary(analysis, results)
    save_results(results, analysis, args.output)
