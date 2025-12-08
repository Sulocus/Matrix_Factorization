"""
相变前斜率分析工具

分析不同(N, M)配置下Q_Y在相变前线性增长区域的斜率，
并拟合斜率与N/M的关系。

使用方法：
    1. 修改下方参数配置
    2. python analyze_slope_vs_ratio.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple

# ============================================================
# 参数配置 - 在这里修改
# ============================================================

# JSON结果文件路径（相对路径）
# 标准数据示例: "_legacy/Result_compareNM/500x50_.../multi_size_results_steps10000.json"
# 正交教师数据示例: "_legacy/Result_ortho_compare/.../ortho_results_steps5000.json"
JSON_PATH = "_legacy/Result_ortho_compare/1000x100_2000x100_3000x100_4000x100/ortho_results_steps5000.json"

# 线性拟合的alpha范围
# 设为 None 则自动检测，设为具体值则手动指定
ALPHA_START = 1  # 例如: 0.5
ALPHA_END = 1.5   # 例如: 1.5

# 输出目录（None = 与输入文件同目录）
OUTPUT_DIR = None

# 分析模式:
#   "both" = 同时分析Q_Y和Q_Y_unobserved
#   "Q_Y" = 只分析Q_Y
#   "Q_Y_unobserved" = 只分析Q_Y_unobserved
#   "all" = 分析所有4个指标（包括正交教师的 Q_Y_ortho, Q_Y_ortho_unobserved）
#   "ortho" = 只分析正交教师的2个指标
ANALYSIS_MODE = "both"

# 是否为正交教师比较数据（自动检测）
IS_ORTHO_COMPARISON = None  # 设为 True/False 强制指定，None 则自动检测

# ============================================================


def load_results(json_path: str) -> dict:
    """加载JSON结果文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def parse_config_key(key: str) -> Tuple[int, int]:
    """解析配置键，如 '200x50' -> (200, 50)"""
    parts = key.split('x')
    return int(parts[0]), int(parts[1])


def extract_qy_data(results: dict, config_key: str, alpha_values: List[float],
                    metric: str = "Q_Y") -> np.ndarray:
    """提取指定配置的Q_Y或Q_Y_unobserved数据

    Args:
        metric: "Q_Y" 或 "Q_Y_unobserved"
    """
    config_results = results[config_key]
    metric_key = f"{metric}_mean"
    qy = []
    for alpha in alpha_values:
        alpha_str = str(float(alpha))
        if alpha_str in config_results:
            # 优先使用指定的metric，如果不存在则回退到Q_Y_mean
            val = config_results[alpha_str].get(metric_key, config_results[alpha_str].get('Q_Y_mean', 0))
            qy.append(val)
        else:
            # 尝试其他格式
            for k in config_results.keys():
                if abs(float(k) - alpha) < 1e-6:
                    val = config_results[k].get(metric_key, config_results[k].get('Q_Y_mean', 0))
                    qy.append(val)
                    break
    return np.array(qy)


def detect_transition_point(alpha_values: np.ndarray, qy_values: np.ndarray) -> Tuple[float, dict]:
    """
    智能检测相变点（梯度峰值法）

    方法：
    1. 计算Q_Y的一阶导数（梯度）
    2. 找梯度最大的位置 = 相变点
    3. 同时返回相变区域的特征

    返回：(相变点alpha, 详细信息dict)
    """
    # 平滑处理（简单移动平均）
    window = 3
    if len(qy_values) > window * 2:
        qy_smooth = np.convolve(qy_values, np.ones(window)/window, mode='valid')
        alpha_smooth = alpha_values[window//2:-(window//2)]
    else:
        qy_smooth = qy_values
        alpha_smooth = alpha_values

    # 计算梯度
    gradient = np.gradient(qy_smooth, alpha_smooth)

    # 找梯度最大点 = 相变点
    max_grad_idx = np.argmax(gradient)
    transition_alpha = alpha_smooth[max_grad_idx]
    max_gradient = gradient[max_grad_idx]

    # 计算相变宽度（梯度超过最大值一半的区域）
    half_max = max_gradient / 2
    above_half = gradient > half_max
    if np.any(above_half):
        first_idx = np.argmax(above_half)
        last_idx = len(above_half) - 1 - np.argmax(above_half[::-1])
        transition_width = alpha_smooth[last_idx] - alpha_smooth[first_idx]
        transition_start = alpha_smooth[first_idx]
        transition_end = alpha_smooth[last_idx]
    else:
        transition_width = 0.2
        transition_start = transition_alpha - 0.1
        transition_end = transition_alpha + 0.1

    info = {
        'transition_alpha': transition_alpha,
        'max_gradient': max_gradient,
        'transition_width': transition_width,
        'transition_start': transition_start,
        'transition_end': transition_end,
        'gradient_curve': (alpha_smooth, gradient)
    }

    return transition_alpha, info


def detect_linear_region(alpha_values: np.ndarray, qy_values: np.ndarray,
                         min_alpha: float = 0.3) -> Tuple[float, float, dict]:
    """
    自动检测线性增长区域

    策略：
    1. 用梯度峰值法找相变点
    2. 起点：Q_Y > 0.02 且 alpha >= min_alpha 的第一个点
    3. 终点：相变开始点前0.1个alpha单位

    返回：(alpha_start, alpha_end, phase_info)
    """
    # 智能检测相变点
    transition_alpha, phase_info = detect_transition_point(alpha_values, qy_values)

    # 找起点：Q_Y首次明显大于0
    start_idx = 0
    for i, qy in enumerate(qy_values):
        if qy > 0.02 and alpha_values[i] >= min_alpha:
            start_idx = i
            break

    # 终点设为相变开始点前0.1个alpha单位
    end_alpha = phase_info['transition_start'] - 0.1

    # 找对应的索引
    end_idx = len(alpha_values) - 1
    for i, alpha in enumerate(alpha_values):
        if alpha > end_alpha:
            end_idx = max(start_idx + 2, i - 1)  # 至少保证3个点
            break

    return alpha_values[start_idx], alpha_values[end_idx], phase_info


def fit_linear_region(alpha_values: np.ndarray, qy_values: np.ndarray,
                      alpha_start: float, alpha_end: float) -> Tuple[float, float, float, float]:
    """
    对指定区间进行线性拟合

    返回：(斜率, 截距, R², 斜率标准误差)
    """
    # 筛选指定区间的数据
    mask = (alpha_values >= alpha_start) & (alpha_values <= alpha_end)
    x = alpha_values[mask]
    y = qy_values[mask]

    if len(x) < 3:
        print(f"  警告：数据点太少 ({len(x)}个)，无法可靠拟合")
        return 0.0, 0.0, 0.0, 0.0

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return slope, intercept, r_value**2, std_err


def fit_slope_vs_ratio(ratios_NM: np.ndarray, slopes: np.ndarray) -> Dict:
    """
    拟合斜率与M/N的关系

    输入的ratios_NM是N/M，函数内部转换为M/N进行分析

    主要模型：
    1. 过原点线性: slope = a * (M/N)  【物理上最合理】
    2. 带截距线性: slope = a * (M/N) + b
    3. 幂律: slope = a * (M/N)^b
    """
    results = {}

    # 转换为 M/N
    ratios = 1.0 / ratios_NM  # M/N

    # 1. 过原点线性拟合: slope = a * (M/N) 【最重要】
    # 使用最小二乘: a = sum(x*y) / sum(x^2)
    try:
        a_origin = np.sum(ratios * slopes) / np.sum(ratios ** 2)
        y_pred = a_origin * ratios
        ss_res = np.sum((slopes - y_pred) ** 2)
        ss_tot = np.sum(slopes ** 2)  # 过原点时，R²的计算方式不同
        r2_origin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['linear_origin'] = {
            'a': a_origin, 'R2': r2_origin,
            'formula': f'slope = {a_origin:.6f} * (M/N)'
        }
    except:
        results['linear_origin'] = None

    # 2. 带截距线性拟合: slope = a * (M/N) + b
    try:
        slope_lin, intercept_lin, r_lin, _, _ = stats.linregress(ratios, slopes)
        results['linear'] = {
            'a': slope_lin, 'b': intercept_lin, 'R2': r_lin**2,
            'formula': f'slope = {slope_lin:.6f} * (M/N) + {intercept_lin:.6f}'
        }
    except:
        results['linear'] = None

    # 3. 幂律拟合: slope = a * (M/N)^b
    try:
        def power_law(x, a, b):
            return a * np.power(x, b)
        popt, _ = curve_fit(power_law, ratios, slopes, p0=[1.0, 1.0], maxfev=5000)
        y_pred = power_law(ratios, *popt)
        ss_res = np.sum((slopes - y_pred)**2)
        ss_tot = np.sum((slopes - np.mean(slopes))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['power_law'] = {
            'a': popt[0], 'b': popt[1], 'R2': r2,
            'formula': f'slope = {popt[0]:.6f} * (M/N)^{popt[1]:.4f}'
        }
    except Exception:
        results['power_law'] = None

    # 4. 二次多项式: slope = a*(M/N)² + b*(M/N) + c
    try:
        coeffs = np.polyfit(ratios, slopes, 2)
        y_pred = np.polyval(coeffs, ratios)
        ss_res = np.sum((slopes - y_pred)**2)
        ss_tot = np.sum((slopes - np.mean(slopes))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['quadratic'] = {
            'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2], 'R2': r2,
            'formula': f'slope = {coeffs[0]:.4f}×(M/N)² + {coeffs[1]:.4f}×(M/N) + {coeffs[2]:.4f}'
        }
    except Exception:
        results['quadratic'] = None

    # 5. 过原点二次: slope = a*(M/N)² + b*(M/N)
    try:
        # 最小二乘: [a, b] = (X^T X)^{-1} X^T y
        X = np.vstack([ratios**2, ratios]).T
        coeffs, _, _, _ = np.linalg.lstsq(X, slopes, rcond=None)
        y_pred = coeffs[0]*ratios**2 + coeffs[1]*ratios
        ss_res = np.sum((slopes - y_pred)**2)
        ss_tot = np.sum(slopes**2)  # 过原点时用 sum(y²)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['quadratic_origin'] = {
            'a': coeffs[0], 'b': coeffs[1], 'R2': r2,
            'formula': f'slope = {coeffs[0]:.4f}×(M/N)² + {coeffs[1]:.4f}×(M/N)'
        }
    except Exception:
        results['quadratic_origin'] = None

    # 6. 平方根（过原点）: slope = a*√(M/N)
    try:
        sqrt_ratios = np.sqrt(ratios)
        a_sqrt = np.sum(sqrt_ratios * slopes) / np.sum(sqrt_ratios**2)
        y_pred = a_sqrt * sqrt_ratios
        ss_res = np.sum((slopes - y_pred)**2)
        ss_tot = np.sum(slopes**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['sqrt_origin'] = {
            'a': a_sqrt, 'R2': r2,
            'formula': f'slope = {a_sqrt:.4f}×√(M/N)'
        }
    except Exception:
        results['sqrt_origin'] = None

    return results


def get_metric_display_info(metric: str) -> Tuple[str, str, str]:
    """根据 metric 名称返回显示标签、标题、文件后缀"""
    metric_info = {
        "Q_Y": (r'$Q_Y$', 'Q_Y (Standard)', ''),
        "Q_Y_unobserved": (r'$Q_Y^{unobs}$', 'Q_Y_unobserved (Standard)', '_unobserved'),
        "Q_Y_ortho": (r'$Q_Y^{ortho}$', 'Q_Y (Orthogonal)', '_ortho'),
        "Q_Y_ortho_unobserved": (r'$Q_Y^{ortho,unobs}$', 'Q_Y_unobserved (Orthogonal)', '_ortho_unobserved'),
    }
    return metric_info.get(metric, (metric, metric, f'_{metric}'))


def plot_analysis(alpha_values: np.ndarray, all_qy: Dict[str, np.ndarray],
                  linear_fits: Dict[str, Dict], slope_analysis: Dict,
                  save_dir: Path, metric: str = "Q_Y"):
    """生成分析图表（简化版，只生成2张图）"""
    metric_label, metric_title, file_suffix = get_metric_display_info(metric)

    # 颜色调色板
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#a65628', '#f781bf', '#999999']

    # =========================================================
    # 图1: Q_Y曲线 + 线性拟合
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    for i, (config_key, qy) in enumerate(all_qy.items()):
        N, M = parse_config_key(config_key)
        ratio = N / M
        color = colors[i % len(colors)]

        # 原始数据
        ax1.plot(alpha_values, qy, 'o-', color=color, markersize=4,
                 linewidth=1.5, label=f'N={N}, M={M} (N/M={ratio:.0f})')

        # 线性拟合线
        if config_key in linear_fits:
            fit = linear_fits[config_key]
            x_fit = np.linspace(fit['alpha_start'], fit['alpha_end'], 50)
            y_fit = fit['slope'] * x_fit + fit['intercept']
            ax1.plot(x_fit, y_fit, '--', color=color, linewidth=2, alpha=0.7)

    ax1.set_xlabel(r'$\tilde{\alpha}$', fontsize=14)
    ax1.set_ylabel(metric_label, fontsize=14)
    ax1.set_title(f'{metric_title} vs Alpha (with Linear Fits)', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    save_path1 = save_dir / f'qy{file_suffix}_with_linear_fits.png'
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"  曲线图已保存: {save_path1}")
    plt.close(fig1)

    # =========================================================
    # 图2: 斜率 vs M/N
    # =========================================================
    ratios_MN = []  # M/N
    slopes_list = []
    slope_errs = []
    labels = []

    for config_key, fit in linear_fits.items():
        N, M = parse_config_key(config_key)
        ratios_MN.append(M / N)
        slopes_list.append(fit['slope'])
        slope_errs.append(fit['slope_err'])
        labels.append(f'N={N}')

    ratios_MN = np.array(ratios_MN)
    slopes_arr = np.array(slopes_list)
    slope_errs = np.array(slope_errs)

    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # 数据点
    ax2.errorbar(ratios_MN, slopes_arr, yerr=slope_errs, fmt='o', markersize=10,
                 capsize=5, color='#2563eb', label='Measured slopes')

    # 添加标签
    for r, s, lbl in zip(ratios_MN, slopes_arr, labels):
        ax2.annotate(lbl, (r, s), textcoords="offset points", xytext=(5, 5), fontsize=9)

    # 拟合曲线（显示所有模型）
    x_fit = np.linspace(0.001, max(ratios_MN) * 1.1, 100)
    fit_styles = [
        ('linear_origin', '#dc2626', '-', 2.5),      # 红色实线
        ('quadratic_origin', '#16a34a', '--', 2.0),  # 绿色虚线
        ('power_law', '#9333ea', '-.', 2.0),         # 紫色点划线
        ('sqrt_origin', '#ea580c', ':', 2.0),        # 橙色点线
    ]

    for model_key, color, linestyle, lw in fit_styles:
        if slope_analysis and slope_analysis.get(model_key):
            m = slope_analysis[model_key]
            if model_key == 'linear_origin':
                y_fit = m['a'] * x_fit
            elif model_key == 'quadratic_origin':
                y_fit = m['a'] * x_fit**2 + m['b'] * x_fit
            elif model_key == 'power_law':
                y_fit = m['a'] * np.power(x_fit, m['b'])
            elif model_key == 'sqrt_origin':
                y_fit = m['a'] * np.sqrt(x_fit)
            else:
                continue
            ax2.plot(x_fit, y_fit, linestyle=linestyle, color=color, linewidth=lw,
                     label=f"{m['formula']}, R²={m['R2']:.4f}")

    ax2.set_xlabel('M/N', fontsize=14)
    ax2.set_ylabel(f'Slope (d{metric_title}/dα)', fontsize=14)
    ax2.set_title(f'Pre-Transition Slope vs M/N ({metric_title})', fontsize=14, fontweight='bold')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    save_path2 = save_dir / f'slope_vs_ratio{file_suffix}.png'
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"  斜率图已保存: {save_path2}")
    plt.close(fig2)


def generate_report(configs, linear_fits, slope_analysis, best_model, metric: str = "Q_Y") -> str:
    """生成自然语言分析报告"""
    _, metric_title, _ = get_metric_display_info(metric)
    is_ortho = "ortho" in metric.lower()

    lines = []
    lines.append(f"# 相变前斜率分析报告 - {metric_title}\n")

    # 数据概述
    lines.append("## 数据概述\n")
    lines.append(f"本次分析包含 **{len(configs)}** 组不同的 (N, M) 配置：\n")
    for config_key, N, M, ratio in configs:
        fit = linear_fits[config_key]
        lines.append(f"- N={N}, M={M} (N/M={ratio:.1f}): 斜率 = {fit['slope']:.4f}, 相变点 α_c ≈ {fit['transition_alpha']:.2f}")
    lines.append("")

    # 核心发现
    lines.append("## 核心发现\n")

    # 模型对比表格
    lines.append("### 拟合模型对比\n")
    lines.append("| 模型 | 公式 | R² |")
    lines.append("|------|------|-----|")

    model_order = ['linear_origin', 'quadratic_origin', 'power_law', 'sqrt_origin', 'linear', 'quadratic']
    model_names = {
        'linear_origin': '过原点线性',
        'quadratic_origin': '过原点二次',
        'power_law': '幂律',
        'sqrt_origin': '平方根',
        'linear': '带截距线性',
        'quadratic': '二次多项式',
    }

    best_r2 = 0
    best_model_key = None
    for model_key in model_order:
        if slope_analysis.get(model_key):
            m = slope_analysis[model_key]
            lines.append(f"| {model_names[model_key]} | `{m['formula']}` | {m['R2']:.4f} |")
            if m['R2'] > best_r2:
                best_r2 = m['R2']
                best_model_key = model_key

    lines.append("")
    if best_model_key:
        lines.append(f"**最佳拟合模型**: {model_names[best_model_key]} (R² = {best_r2:.4f})\n")

    # 分析过原点线性拟合
    if slope_analysis.get('linear_origin'):
        lo = slope_analysis['linear_origin']
        lines.append("### 过原点线性拟合\n")
        lines.append(f"**公式**: `slope = {lo['a']:.4f} × (M/N)`\n")
        lines.append(f"**R² = {lo['R2']:.4f}**\n")

        if lo['R2'] > 0.95:
            lines.append("拟合效果优秀，数据强烈支持 **斜率与 M/N 成正比** 的假设。")
        elif lo['R2'] > 0.85:
            lines.append("拟合效果良好，数据基本支持斜率与 M/N 成正比的假设。")
        else:
            lines.append("拟合效果一般，可能存在其他影响因素。")
        lines.append("")

    # 物理解释
    lines.append("## 物理意义\n")

    if is_ortho:
        lines.append("**正交教师模型**通过 QR 分解强制 W^T W = I_M，消除了有限尺寸涨落。\n")
        lines.append("- 标准教师的 Gram 矩阵 G = (M/N) W^T W = I + Δ，其中 Δ 是 O(1/√N) 的随机涨落")
        lines.append("- 正交教师强制 Δ = 0，因此低 α 区域的线性偏移应该更小")
        lines.append("")
        lines.append("如果正交教师的斜率系数显著小于标准教师，说明 **有限尺寸涨落是导致低 α 区域 Q_Y 偏高的主要原因**。")
    else:
        lines.append("相变前线性增长区域的斜率反映了系统在低观测密度下的学习效率：\n")
        lines.append("- **斜率 ∝ M/N** 意味着：当隐变量维度 M 相对于观测维度 N 越大时，")
        lines.append("  每增加一个单位的观测密度 α，Q_Y 增长越快")
        lines.append("- 这与直觉一致：M/N 越大，信息冗余度越高，更容易从部分观测中恢复信号")
    lines.append("")

    # 结论
    lines.append("## 结论\n")

    if slope_analysis.get('linear_origin') and slope_analysis['linear_origin']['R2'] > 0.9:
        a = slope_analysis['linear_origin']['a']
        metric_short = "Q_Y" if "unobs" not in metric else "Q_Y_unobs"
        lines.append("数据支持以下经验公式：\n")
        lines.append("```")
        lines.append(f"d{metric_short}/dα ≈ {a:.2f} × (M/N)    (相变前线性区)")
        lines.append("```\n")
        lines.append(f"即相变前 {metric_title} 的增长速率与 M/N 成正比，比例系数约为 **{a:.2f}**。")
    else:
        lines.append(f"最佳拟合模型为 **{best_model}**，但数据可能需要更多样本点来确认关系。")

    return "\n".join(lines)


def run_single_analysis(data: dict, configs: list, alpha_values: np.ndarray,
                        output_dir: Path, metric: str = "Q_Y") -> dict:
    """对单个指标进行完整分析

    Args:
        metric: "Q_Y", "Q_Y_unobserved", "Q_Y_ortho", 或 "Q_Y_ortho_unobserved"

    Returns:
        分析结果字典
    """
    _, metric_title, file_suffix = get_metric_display_info(metric)

    results = data['results']

    print(f"\n{'─' * 60}")
    print(f"  分析指标: {metric_title}")
    print(f"{'─' * 60}")

    # 存储所有数据和拟合结果
    all_qy = {}
    linear_fits = {}

    for config_key, N, M, ratio in configs:
        print(f"\n[N={N}, M={M}, N/M={ratio:.1f}]")

        # 提取数据
        qy = extract_qy_data(results, config_key, alpha_values, metric=metric)
        all_qy[config_key] = qy

        # 检测线性区域和相变点
        if ALPHA_START is not None and ALPHA_END is not None:
            alpha_start, alpha_end = ALPHA_START, ALPHA_END
            _, phase_info = detect_transition_point(alpha_values, qy)
            print(f"  使用指定线性区域: [{alpha_start:.2f}, {alpha_end:.2f}]")
        else:
            alpha_start, alpha_end, phase_info = detect_linear_region(alpha_values, qy)
            print(f"  自动检测线性区域: [{alpha_start:.2f}, {alpha_end:.2f}]")

        # 输出相变点信息
        print(f"  检测到相变点: α_c = {phase_info['transition_alpha']:.3f}")
        print(f"  相变区域: [{phase_info['transition_start']:.2f}, {phase_info['transition_end']:.2f}]")
        print(f"  最大梯度: {phase_info['max_gradient']:.4f}")

        # 线性拟合
        slope, intercept, r2, slope_err = fit_linear_region(
            alpha_values, qy, alpha_start, alpha_end
        )

        print(f"  斜率: {slope:.6f} ± {slope_err:.6f}")
        print(f"  截距: {intercept:.6f}")
        print(f"  R²: {r2:.6f}")

        linear_fits[config_key] = {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'slope_err': slope_err,
            'alpha_start': alpha_start,
            'alpha_end': alpha_end,
            'transition_alpha': phase_info['transition_alpha'],
            'transition_start': phase_info['transition_start'],
            'transition_end': phase_info['transition_end'],
            'max_gradient': phase_info['max_gradient']
        }

    # 分析斜率与N/M的关系
    print(f"\n  斜率 vs M/N 关系拟合:")

    ratios = np.array([c[3] for c in configs])
    slopes = np.array([linear_fits[c[0]]['slope'] for c in configs])

    slope_analysis = fit_slope_vs_ratio(ratios, slopes)

    # 只显示过原点线性拟合结果
    if slope_analysis.get('linear_origin'):
        result = slope_analysis['linear_origin']
        print(f"    slope = {result['a']:.4f} × (M/N), R² = {result['R2']:.4f}")

    # 找最佳模型
    best_model = 'linear_origin'
    best_r2 = slope_analysis.get('linear_origin', {}).get('R2', 0)

    # 生成图表

    plot_analysis(alpha_values, all_qy, linear_fits, slope_analysis, output_dir, metric=metric)

    # 保存分析结果
    linear_fits_clean = {}
    for k, v in linear_fits.items():
        linear_fits_clean[k] = {kk: vv for kk, vv in v.items() if kk != 'gradient_curve'}

    analysis_results = {
        'metric': metric,
        'configs': [{'key': c[0], 'N': c[1], 'M': c[2], 'ratio': c[3]} for c in configs],
        'linear_fits': linear_fits_clean,
        'slope_vs_ratio': {k: v for k, v in slope_analysis.items() if v is not None},
        'best_model': best_model
    }

    analysis_path = output_dir / f'slope_analysis_results{file_suffix}.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n分析结果已保存: {analysis_path}")

    # 生成自然语言报告
    report = generate_report(configs, linear_fits, slope_analysis, best_model, metric=metric)
    report_path = output_dir / f'analysis_report{file_suffix}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"分析报告已保存: {report_path}")

    return analysis_results


def detect_ortho_data(data: dict) -> bool:
    """检测数据是否包含正交教师指标"""
    results = data.get('results', {})
    if not results:
        return False
    first_config = list(results.values())[0]
    first_alpha = list(first_config.values())[0]
    return 'Q_Y_ortho_mean' in first_alpha


def generate_summary_report(all_results: dict, output_dir: Path, is_ortho: bool) -> str:
    """生成综合对比报告"""
    lines = []
    lines.append("# 斜率分析综合报告\n")
    lines.append(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 汇总表格
    lines.append("## 斜率系数汇总\n")
    lines.append("| 指标 | 斜率系数 a | R² | 物理含义 |")
    lines.append("|------|-----------|-----|---------|")

    for metric, res in all_results.items():
        lo = res.get('slope_vs_ratio', {}).get('linear_origin', {})
        a = lo.get('a', 0)
        r2 = lo.get('R2', 0)
        _, title, _ = get_metric_display_info(metric)

        if "ortho" in metric:
            meaning = "正交教师（无有限尺寸涨落）"
        elif "unobs" in metric:
            meaning = "未观测位置（泛化能力）"
        else:
            meaning = "全部位置（含训练数据）"

        lines.append(f"| {title} | {a:.4f} | {r2:.4f} | {meaning} |")

    lines.append("")

    # 多模型拟合对比（取第一个指标的详细对比）
    first_metric = list(all_results.keys())[0] if all_results else None
    if first_metric:
        slope_vs_ratio = all_results[first_metric].get('slope_vs_ratio', {})
        if slope_vs_ratio:
            lines.append("## 拟合模型对比（以第一个指标为例）\n")
            lines.append("| 模型 | 公式 | R² |")
            lines.append("|------|------|-----|")

            model_order = ['linear_origin', 'quadratic_origin', 'power_law', 'sqrt_origin']
            model_names = {
                'linear_origin': '过原点线性',
                'quadratic_origin': '过原点二次',
                'power_law': '幂律',
                'sqrt_origin': '平方根',
            }

            for model_key in model_order:
                if slope_vs_ratio.get(model_key):
                    m = slope_vs_ratio[model_key]
                    lines.append(f"| {model_names[model_key]} | `{m['formula']}` | {m['R2']:.4f} |")

            lines.append("")

    # 对比分析
    if is_ortho:
        lines.append("## 标准教师 vs 正交教师对比\n")

        std_qy = all_results.get('Q_Y', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)
        ortho_qy = all_results.get('Q_Y_ortho', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)

        if std_qy > 0 and ortho_qy > 0:
            reduction = (std_qy - ortho_qy) / std_qy * 100
            lines.append(f"- 标准教师 Q_Y 斜率系数: **{std_qy:.4f}**")
            lines.append(f"- 正交教师 Q_Y 斜率系数: **{ortho_qy:.4f}**")
            lines.append(f"- 斜率降低: **{reduction:.1f}%**")
            lines.append("")

            if reduction > 50:
                lines.append("**结论**: 正交教师显著降低了相变前斜率，说明 **有限尺寸涨落是导致低 α 区域 Q_Y 偏高的主要原因**。")
            elif reduction > 20:
                lines.append("**结论**: 正交教师降低了相变前斜率，有限尺寸涨落对低 α 区域有一定影响。")
            else:
                lines.append("**结论**: 正交教师对斜率影响不大，低 α 区域的 Q_Y 偏移可能有其他原因。")
        lines.append("")

        # unobserved 对比
        std_unobs = all_results.get('Q_Y_unobserved', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)
        ortho_unobs = all_results.get('Q_Y_ortho_unobserved', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)

        if std_unobs > 0 and ortho_unobs > 0:
            lines.append("### 泛化能力 (unobserved) 对比\n")
            reduction_unobs = (std_unobs - ortho_unobs) / std_unobs * 100
            lines.append(f"- 标准教师 Q_Y_unobserved 斜率系数: **{std_unobs:.4f}**")
            lines.append(f"- 正交教师 Q_Y_ortho_unobserved 斜率系数: **{ortho_unobs:.4f}**")
            lines.append(f"- 斜率降低: **{reduction_unobs:.1f}%**")
    else:
        # 非正交数据的对比
        qy = all_results.get('Q_Y', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)
        qy_unobs = all_results.get('Q_Y_unobserved', {}).get('slope_vs_ratio', {}).get('linear_origin', {}).get('a', 0)

        if qy > 0 and qy_unobs > 0:
            lines.append("## Q_Y vs Q_Y_unobserved 对比\n")
            ratio = qy_unobs / qy
            lines.append(f"- Q_Y 斜率系数: **{qy:.4f}**")
            lines.append(f"- Q_Y_unobserved 斜率系数: **{qy_unobs:.4f}**")
            lines.append(f"- 比值: **{ratio:.4f}**")

    return "\n".join(lines)


def main():
    # 使用顶部配置的参数
    json_path = Path(JSON_PATH)

    if not json_path.exists():
        print(f"错误：文件不存在 {json_path}")
        print("请检查 JSON_PATH 配置是否正确")
        return

    print("=" * 70)
    print("  相变前斜率分析")
    print("=" * 70)
    print(f"输入文件: {json_path.name}")

    data = load_results(json_path)

    # 自动检测是否为正交教师数据
    is_ortho = IS_ORTHO_COMPARISON if IS_ORTHO_COMPARISON is not None else detect_ortho_data(data)
    print(f"数据类型: {'正交教师对比数据' if is_ortho else '标准数据'}")

    # 提取配置
    alpha_values = np.array(data['alpha_values'])
    results = data['results']
    config_keys = list(results.keys())

    print(f"Alpha 范围: {alpha_values[0]:.2f} - {alpha_values[-1]:.2f} ({len(alpha_values)} 点)")
    print(f"配置数量: {len(config_keys)}")

    # 解析所有配置
    configs = []
    for key in config_keys:
        N, M = parse_config_key(key)
        configs.append((key, N, M, N/M))

    # 按N/M排序
    configs.sort(key=lambda x: x[3])

    for config_key, N, M, ratio in configs:
        print(f"  - N={N}, M={M} (N/M={ratio:.0f})")

    # 输出目录
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
    else:
        output_dir = json_path.parent / 'slope_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 根据 ANALYSIS_MODE 和数据类型决定分析哪些指标
    if ANALYSIS_MODE == "all" and is_ortho:
        metrics_to_analyze = ["Q_Y", "Q_Y_unobserved", "Q_Y_ortho", "Q_Y_ortho_unobserved"]
    elif ANALYSIS_MODE == "ortho" and is_ortho:
        metrics_to_analyze = ["Q_Y_ortho", "Q_Y_ortho_unobserved"]
    elif ANALYSIS_MODE == "both":
        if is_ortho:
            metrics_to_analyze = ["Q_Y", "Q_Y_unobserved", "Q_Y_ortho", "Q_Y_ortho_unobserved"]
        else:
            metrics_to_analyze = ["Q_Y", "Q_Y_unobserved"]
    elif ANALYSIS_MODE == "Q_Y":
        metrics_to_analyze = ["Q_Y"]
    elif ANALYSIS_MODE == "Q_Y_unobserved":
        metrics_to_analyze = ["Q_Y_unobserved"]
    else:
        print(f"未知的分析模式: {ANALYSIS_MODE}")
        return

    print(f"分析指标: {', '.join(metrics_to_analyze)}")

    all_results = {}
    for metric in metrics_to_analyze:
        all_results[metric] = run_single_analysis(data, configs, alpha_values, output_dir, metric=metric)

    # 生成综合报告
    print(f"\n{'=' * 70}")
    print("  生成综合报告")
    print(f"{'=' * 70}")

    summary = generate_summary_report(all_results, output_dir, is_ortho)
    summary_path = output_dir / 'summary_report.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"综合报告已保存: {summary_path}")

    # 打印摘要
    print("\n" + "─" * 60)
    print("  斜率系数 (过原点线性: slope = a × M/N)")
    print("─" * 60)
    for metric, res in all_results.items():
        lo = res.get('slope_vs_ratio', {}).get('linear_origin', {})
        _, title, _ = get_metric_display_info(metric)
        print(f"  {title:30s}: a = {lo.get('a', 0):.4f}, R² = {lo.get('R2', 0):.4f}")

    # 打印多模型对比（取第一个指标）
    first_metric = list(all_results.keys())[0] if all_results else None
    if first_metric:
        slope_vs_ratio = all_results[first_metric].get('slope_vs_ratio', {})
        if slope_vs_ratio:
            print("\n" + "─" * 60)
            print(f"  多模型拟合对比（{first_metric}）")
            print("─" * 60)
            model_names = {
                'linear_origin': '过原点线性',
                'quadratic_origin': '过原点二次',
                'power_law': '幂律',
                'sqrt_origin': '平方根',
            }
            for model_key in ['linear_origin', 'quadratic_origin', 'power_law', 'sqrt_origin']:
                if slope_vs_ratio.get(model_key):
                    m = slope_vs_ratio[model_key]
                    print(f"  {model_names[model_key]:12s}: R² = {m['R2']:.4f}  {m['formula']}")

    print(f"\n{'=' * 70}")
    print(f"  分析完成! 结果目录: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
