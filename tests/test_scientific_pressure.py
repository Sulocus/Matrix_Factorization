#!/usr/bin/env python3
"""
Scientific Pressure Test Suite for SMF ConfigAdvisor.

20 comprehensive test cases designed with scientific rigor:
- G: Comprehensive boundary tests (6 cases) - Multiple params at boundary
- H: Implicit expression tests (4 cases) - Descriptive language
- I: Scientific scenario tests (6 cases) - Real research use cases
- J: Edge case tests (4 cases) - Semantic contradictions

Key improvements over previous tests:
1. "Just hit boundary" vs "clearly exceed" distinction
2. Multi-parameter combinations
3. LLM vs post-processing separation
4. Scientific validity for all edge cases
"""

import sys
import os
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smf.core.llm_advisor import analyze_user_request, AnalysisResult


class TestStatus(Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


@dataclass
class ScientificTestCase:
    """A scientific pressure test case."""
    id: str
    category: str
    query: str
    # Expected configuration checks
    check_config: Dict[str, Any]
    # Expected warnings (keywords that should appear)
    expect_warnings: List[str] = field(default_factory=list)
    # Warnings that should NOT appear (for "safe boundary" tests)
    reject_warnings: List[str] = field(default_factory=list)
    # Scientific purpose
    purpose: str = ""
    # Difficulty rating (1-5)
    difficulty: int = 3


@dataclass
class ScientificTestResult:
    """Result with LLM vs post-processing separation."""
    test_id: str
    status: TestStatus
    understanding: str
    config: Dict[str, Any]
    # Separated warnings
    llm_warnings: List[Dict]  # From LLM (no 'type' field)
    post_warnings: List[Dict]  # From post-processing (has 'type' field)
    issues: List[str]
    # Analysis
    llm_contributed: bool  # Did LLM generate useful warnings?
    post_saved: bool  # Did post-processing save the test?


# ============================================================
# G Class: Comprehensive Boundary Tests (6)
# ============================================================

G_TESTS = [
    ScientificTestCase(
        id="G1", category="Boundary",
        query="N=5000的大矩阵，alpha扫到5.0",
        check_config={"N1": 5000, "alpha_stop": 5.0},
        expect_warnings=["5000", "5.0"],  # Both should trigger
        purpose="双边界刚好命中",
        difficulty=4,
    ),
    ScientificTestCase(
        id="G2", category="Boundary",
        query="用N=4800的矩阵，alpha扫到4.8",
        check_config={"N1": 4800, "alpha_stop": 4.8},
        expect_warnings=[],  # Should NOT warn
        reject_warnings=["警告", "超", "大"],  # Should not have these
        purpose="边界附近但安全（测试过度敏感）",
        difficulty=4,
    ),
    ScientificTestCase(
        id="G3", category="Boundary",
        query="N=30的小矩阵快速测试，只跑50步",
        check_config={"N1": 30, "max_steps": 50},
        expect_warnings=["30", "50", "小", "步"],
        purpose="双下限违反",
        difficulty=3,
    ),
    ScientificTestCase(
        id="G4", category="Boundary",
        query="研究极端情况：N=10000，alpha一直扫到8",
        check_config={"N1": 10000, "alpha_stop": 8.0},
        expect_warnings=["10000", "8", "显存", "超"],
        purpose="极限压力测试",
        difficulty=4,
    ),
    ScientificTestCase(
        id="G5", category="Boundary",
        query="M=180, N=200，研究满秩极限",
        check_config={"M": 180, "N1": 200},
        expect_warnings=[],  # M/N=0.9 is valid, should not error
        purpose="M接近N的满秩研究",
        difficulty=3,
    ),
    ScientificTestCase(
        id="G6", category="Boundary",
        query="用M=300, N=200做过完备字典学习",
        check_config={"M": 300, "N1": 200},
        expect_warnings=["M", "N", "大于"],  # Should warn but recognize validity
        purpose="M>N的过完备研究",
        difficulty=4,
    ),
]


# ============================================================
# H Class: Implicit Expression Tests (4)
# ============================================================

H_TESTS = [
    ScientificTestCase(
        id="H1", category="Implicit",
        query="用最大的矩阵跑一遍，看看GPU极限",
        check_config={},  # Should infer large N
        expect_warnings=["显存", "大"],
        purpose="描述性语言推断N",
        difficulty=3,
    ),
    ScientificTestCase(
        id="H2", category="Implicit",
        query="看看加噪声后相变怎么变软",
        check_config={"noise_var": True},  # Should have noise_var
        expect_warnings=[],
        purpose="科学术语理解（变软→噪声）",
        difficulty=3,
    ),
    ScientificTestCase(
        id="H3", category="Implicit",
        query="我需要快速收敛，矩阵不大，N=300",
        check_config={"N1": 300, "algorithm_key": "bigamp"},
        expect_warnings=[],
        purpose="需求反推算法",
        difficulty=3,
    ),
    ScientificTestCase(
        id="H4", category="Implicit",
        query="把alpha范围扩大5倍，从0到20",
        check_config={"alpha_stop": 20},
        expect_warnings=["20", "超", "范围"],
        purpose="相对描述理解",
        difficulty=4,
    ),
]


# ============================================================
# I Class: Scientific Scenario Tests (6)
# ============================================================

I_TESTS = [
    ScientificTestCase(
        id="I1", category="Scientific",
        query="在alpha=1.5附近做超精细扫描，步长0.005",
        check_config={"alpha_step": 0.005},
        expect_warnings=[],  # Fine step is valid
        purpose="相变临界区精细扫描",
        difficulty=2,
    ),
    ScientificTestCase(
        id="I2", category="Scientific",
        query="做finite-size scaling，N从100到2000",
        check_config={},  # Should recognize size_scaling
        expect_warnings=[],
        purpose="Finite-size scaling序列",
        difficulty=3,
    ),
    ScientificTestCase(
        id="I3", category="Scientific",
        query="对比随机图和双正则图的相变差异",
        check_config={},  # Should recognize comparison
        expect_warnings=[],
        purpose="图结构对比研究",
        difficulty=3,
    ),
    ScientificTestCase(
        id="I4", category="Scientific",
        query="测量replica overlap，看解的唯一性",
        check_config={},  # Should recognize replica experiment
        expect_warnings=[],
        purpose="Replica overlap分析",
        difficulty=4,
    ),
    ScientificTestCase(
        id="I5", category="Scientific",
        query="noise_var=0.5测试强噪声下的恢复能力",
        check_config={"noise_var": 0.5},
        expect_warnings=[],  # 0.5 is valid noise level
        purpose="强噪声鲁棒性",
        difficulty=2,
    ),
    ScientificTestCase(
        id="I6", category="Scientific",
        query="alpha从0.01开始扫，找最小可恢复密度",
        check_config={"alpha_start": 0.01},
        expect_warnings=[],  # Low alpha is valid for info theory
        purpose="低观测极限研究",
        difficulty=3,
    ),
]


# ============================================================
# J Class: Edge Case Tests (4)
# ============================================================

J_TESTS = [
    ScientificTestCase(
        id="J1", category="Edge",
        query="用AGD跑N=3000的大矩阵",
        check_config={"algorithm_key": "agd", "N1": 3000},
        expect_warnings=["AGD", "大", "BiG-AMP"],  # Should suggest BiG-AMP
        purpose="算法与尺度不匹配",
        difficulty=4,
    ),
    ScientificTestCase(
        id="J2", category="Edge",
        query="测试damping=0.05的稳定性边界",
        check_config={"damping": 0.05},
        expect_warnings=["damping", "不稳定", "低"],
        purpose="damping边界值研究",
        difficulty=4,
    ),
    ScientificTestCase(
        id="J3", category="Edge",
        query="要精确收敛，但只跑10步",
        check_config={"max_steps": 10},
        expect_warnings=["10", "步", "收敛", "少"],
        purpose="精度与速度矛盾",
        difficulty=5,
    ),
    ScientificTestCase(
        id="J4", category="Edge",
        query="研究热力学极限，用N=50的小矩阵",
        check_config={"N1": 50},
        expect_warnings=["50", "热力学", "小", "有限"],
        purpose="概念矛盾检测",
        difficulty=5,
    ),
]

ALL_TESTS = G_TESTS + H_TESTS + I_TESTS + J_TESTS


def separate_warnings(warnings: List[Dict]) -> tuple:
    """Separate LLM warnings from post-processing warnings."""
    llm_warnings = [w for w in warnings if 'type' not in w]
    post_warnings = [w for w in warnings if 'type' in w]
    return llm_warnings, post_warnings


def check_warnings(
    all_warnings: List[Dict],
    expect: List[str],
    reject: List[str]
) -> tuple:
    """Check if warnings match expectations."""
    issues = []
    warning_text = " ".join([
        str(w.get('reason', '')) + str(w.get('option', ''))
        for w in all_warnings
    ])

    # Check expected keywords
    for keyword in expect:
        if keyword not in warning_text:
            issues.append(f"Missing expected warning: '{keyword}'")

    # Check rejected keywords
    for keyword in reject:
        if keyword in warning_text:
            issues.append(f"Unwanted warning contains: '{keyword}'")

    return issues


def run_single_test(test: ScientificTestCase, delay: float = 3.0) -> ScientificTestResult:
    """Run a single scientific test case."""
    issues = []

    try:
        result = analyze_user_request(test.query)

        # Separate LLM and post-processing warnings
        llm_warnings, post_warnings = separate_warnings(result.missing_important)

        # Check config values
        for key, expected in test.check_config.items():
            actual = result.config.get(key)
            if expected is True:
                if actual is None:
                    issues.append(f"Missing config: {key}")
            elif expected is not None and actual != expected:
                # Allow some tolerance for numbers
                if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                    if abs(actual - expected) > max(1, expected * 0.1):
                        issues.append(f"Config mismatch: {key}={actual}, expected={expected}")
                else:
                    issues.append(f"Config mismatch: {key}={actual}, expected={expected}")

        # Check warnings
        warning_issues = check_warnings(
            result.missing_important,
            test.expect_warnings,
            test.reject_warnings
        )
        issues.extend(warning_issues)

        # Determine status
        if not issues:
            status = TestStatus.PASS
        elif len(issues) <= 2:
            status = TestStatus.PARTIAL
        else:
            status = TestStatus.FAIL

        # Analyze contributions
        llm_contributed = len(llm_warnings) > 0
        post_saved = len(post_warnings) > 0 and len(llm_warnings) == 0

        return ScientificTestResult(
            test_id=test.id,
            status=status,
            understanding=result.understanding,
            config=result.config,
            llm_warnings=llm_warnings,
            post_warnings=post_warnings,
            issues=issues,
            llm_contributed=llm_contributed,
            post_saved=post_saved,
        )

    except Exception as e:
        return ScientificTestResult(
            test_id=test.id,
            status=TestStatus.FAIL,
            understanding=f"Error: {str(e)}",
            config={},
            llm_warnings=[],
            post_warnings=[],
            issues=[f"Exception: {str(e)}"],
            llm_contributed=False,
            post_saved=False,
        )
    finally:
        time.sleep(delay)


def run_all_tests(delay: float = 3.0) -> List[ScientificTestResult]:
    """Run all scientific pressure tests."""
    results = []
    total = len(ALL_TESTS)

    print(f"\n{'='*60}")
    print(f"Scientific Pressure Test Suite - {total} Cases")
    print(f"{'='*60}\n")

    for i, test in enumerate(ALL_TESTS):
        print(f"[{i+1}/{total}] {test.id}: {test.purpose}...")
        result = run_single_test(test, delay)
        results.append(result)

        status_icon = {
            TestStatus.PASS: "✓",
            TestStatus.PARTIAL: "~",
            TestStatus.FAIL: "✗",
        }[result.status]

        contrib = []
        if result.llm_contributed:
            contrib.append("LLM")
        if result.post_saved:
            contrib.append("POST")
        contrib_str = f" [{'+'.join(contrib)}]" if contrib else ""

        print(f"  {status_icon} {result.status.value}{contrib_str}")
        if result.issues:
            for issue in result.issues[:2]:
                print(f"    - {issue}")

    return results


def analyze_results(results: List[ScientificTestResult]) -> Dict[str, Any]:
    """Analyze results with LLM vs post-processing breakdown."""
    categories = {}

    for result in results:
        test = next(t for t in ALL_TESTS if t.id == result.test_id)
        cat = test.category

        if cat not in categories:
            categories[cat] = {
                "pass": 0, "partial": 0, "fail": 0, "total": 0,
                "llm_contributed": 0, "post_saved": 0
            }

        categories[cat]["total"] += 1
        if result.status == TestStatus.PASS:
            categories[cat]["pass"] += 1
        elif result.status == TestStatus.PARTIAL:
            categories[cat]["partial"] += 1
        else:
            categories[cat]["fail"] += 1

        if result.llm_contributed:
            categories[cat]["llm_contributed"] += 1
        if result.post_saved:
            categories[cat]["post_saved"] += 1

    # Calculate rates
    for cat, stats in categories.items():
        stats["pass_rate"] = (stats["pass"] + 0.5 * stats["partial"]) / stats["total"] * 100
        stats["llm_rate"] = stats["llm_contributed"] / stats["total"] * 100

    # Overall
    total = len(results)
    total_pass = sum(c["pass"] for c in categories.values())
    total_partial = sum(c["partial"] for c in categories.values())
    total_fail = sum(c["fail"] for c in categories.values())
    total_llm = sum(c["llm_contributed"] for c in categories.values())
    total_post = sum(c["post_saved"] for c in categories.values())

    return {
        "categories": categories,
        "overall": {
            "pass": total_pass,
            "partial": total_partial,
            "fail": total_fail,
            "total": total,
            "pass_rate": (total_pass + 0.5 * total_partial) / total * 100,
            "llm_contributed": total_llm,
            "post_saved": total_post,
            "llm_rate": total_llm / total * 100,
        }
    }


def print_summary(analysis: Dict[str, Any], results: List[ScientificTestResult]):
    """Print comprehensive test summary."""
    print(f"\n{'='*60}")
    print("SCIENTIFIC PRESSURE TEST SUMMARY")
    print(f"{'='*60}\n")

    # By category
    print("By Category:")
    print("-" * 70)
    print(f"{'Category':<12} {'Pass':<5} {'Part':<5} {'Fail':<5} {'Rate':<8} {'LLM%':<8}")
    print("-" * 70)

    for cat, stats in analysis["categories"].items():
        print(f"{cat:<12} {stats['pass']:<5} {stats['partial']:<5} {stats['fail']:<5} "
              f"{stats['pass_rate']:.1f}%   {stats['llm_rate']:.1f}%")

    print("-" * 70)
    overall = analysis["overall"]
    print(f"{'OVERALL':<12} {overall['pass']:<5} {overall['partial']:<5} {overall['fail']:<5} "
          f"{overall['pass_rate']:.1f}%   {overall['llm_rate']:.1f}%")

    # LLM vs Post-processing breakdown
    print(f"\n\nLayer Analysis:")
    print("-" * 50)
    print(f"LLM contributed warnings: {overall['llm_contributed']}/{overall['total']} ({overall['llm_rate']:.1f}%)")
    print(f"Post-processing saved:    {overall['post_saved']}/{overall['total']}")

    # Failed tests
    failed = [r for r in results if r.status == TestStatus.FAIL]
    if failed:
        print(f"\n\nFailed Tests ({len(failed)}):")
        print("-" * 50)
        for r in failed:
            test = next(t for t in ALL_TESTS if t.id == r.test_id)
            print(f"  {r.test_id} [{test.category}]: {test.purpose}")
            for issue in r.issues[:2]:
                print(f"    - {issue}")


def save_results(results: List[ScientificTestResult], analysis: Dict, filename: str):
    """Save results to JSON."""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_type": "scientific_pressure",
        "analysis": analysis,
        "results": [
            {
                "test_id": r.test_id,
                "status": r.status.value,
                "understanding": r.understanding,
                "config": r.config,
                "llm_warnings": r.llm_warnings,
                "post_warnings": r.post_warnings,
                "issues": r.issues,
                "llm_contributed": r.llm_contributed,
                "post_saved": r.post_saved,
            }
            for r in results
        ],
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMF Scientific Pressure Test")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between API calls")
    parser.add_argument("--output", type=str, default="tests/scientific_pressure_results.json")
    args = parser.parse_args()

    results = run_all_tests(delay=args.delay)
    analysis = analyze_results(results)
    print_summary(analysis, results)
    save_results(results, analysis, args.output)
