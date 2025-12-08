"""
Phase Transition Analyzer - 基于梯度分析和多指标一致性的相变检测系统

核心功能：
1. 梯度分析：检测相变点（梯度峰值）
2. 异常检测：识别梯度异常区域
3. 三指标一致性：Q_Y, Q_W', Q_X' 在相变点的一致性分析
4. 自适应采样：根据区域特征建议采样策略
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time


class PhaseTransitionAnalyzer:
    """增强版梯度分析器"""

    def __init__(self, alphas: List[float], metrics: Dict,
                 decel_threshold: float = 2.0,
                 volatility_threshold: float = 0.5,
                 severity_threshold: float = 2.0):
        """
        参数:
            alphas: alpha值列表
            metrics: 字典 {alpha: {metric_name: value, ...}}
            decel_threshold: 梯度减速阈值（二阶导数，默认2.0，只检测剧烈减速）
            volatility_threshold: 梯度波动阈值（默认0.5，只检测剧烈波动）
            severity_threshold: 最小严重性阈值（默认2.0，只对高严重性异常密集采样）
        """
        self.alphas = np.array(alphas)
        self.metrics = metrics

        # 提取三个关键指标
        self.Q_Y = np.array([metrics[str(a)]['Q_Y_mean'] for a in alphas])
        self.Q_W_prime = np.array([metrics[str(a)]['Q_W_prime_mean'] for a in alphas])
        self.Q_X_prime = np.array([metrics[str(a)]['Q_X_prime_mean'] for a in alphas])

        # 异常检测阈值
        self.decel_threshold = decel_threshold
        self.volatility_threshold = volatility_threshold
        self.severity_threshold = severity_threshold

        # 缓存计算结果
        self._gradients = {}
        self._consistency = None

    # ========================================
    # 1. 基础梯度分析
    # ========================================

    def compute_gradient(self, metric: str = 'Q_Y') -> np.ndarray:
        """
        计算指定指标的数值导数 dQ/dα (中心差分法)

        参数:
            metric: 'Q_Y', 'Q_W_prime', 'Q_X_prime'
        """
        if metric in self._gradients:
            return self._gradients[metric]

        if metric == 'Q_Y':
            values = self.Q_Y
        elif metric == 'Q_W_prime':
            values = self.Q_W_prime
        elif metric == 'Q_X_prime':
            values = self.Q_X_prime
        else:
            raise ValueError(f"Unknown metric: {metric}")

        gradient = np.gradient(values, self.alphas)
        self._gradients[metric] = gradient
        return gradient

    def find_max_gradient_point(self, metric: str = 'Q_Y') -> Dict:
        """
        找到梯度最大的点（相变候选点）

        返回:
            {
                'alpha_c': 相变中心alpha值,
                'gradient_max': 最大梯度值,
                'index': 索引位置,
                'Q_before': 前一个点的Q值,
                'Q_after': 后一个点的Q值
            }
        """
        gradient = self.compute_gradient(metric)
        max_idx = np.argmax(gradient)

        if metric == 'Q_Y':
            values = self.Q_Y
        elif metric == 'Q_W_prime':
            values = self.Q_W_prime
        else:
            values = self.Q_X_prime

        return {
            'alpha_c': self.alphas[max_idx],
            'gradient_max': gradient[max_idx],
            'index': max_idx,
            'Q_before': values[max_idx-1] if max_idx > 0 else None,
            'Q_after': values[max_idx+1] if max_idx < len(values)-1 else None,
            'Q_at_transition': values[max_idx]
        }

    # ========================================
    # 2. 三指标一致性分析
    # ========================================

    def compute_metric_consistency(self) -> np.ndarray:
        """
        计算 Q_Y, Q_W', Q_X' 的梯度一致性

        物理意义：N1≠N2时，三个指标在相变附近应该表现统一

        返回:
            一致性数组，范围 [0, 1]，1表示完全一致
        """
        if self._consistency is not None:
            return self._consistency

        # 三个指标的梯度
        grad_QY = self.compute_gradient('Q_Y')
        grad_QW = self.compute_gradient('Q_W_prime')
        grad_QX = self.compute_gradient('Q_X_prime')

        # 计算三者的平均梯度和标准差
        grad_stack = np.stack([grad_QY, grad_QW, grad_QX], axis=0)
        grad_mean = np.mean(grad_stack, axis=0)
        grad_std = np.std(grad_stack, axis=0)

        # 一致性度量：变异系数的倒数
        # CV = std / mean, consistency = 1 / (1 + CV)
        # 当三者梯度接近时，CV小，consistency接近1
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = grad_std / (np.abs(grad_mean) + 1e-8)
            consistency = 1.0 / (1.0 + cv)
            consistency = np.nan_to_num(consistency, nan=0.0, posinf=1.0, neginf=0.0)

        self._consistency = consistency
        return consistency

    def detect_phase_transition_enhanced(self) -> Dict:
        """
        增强相变检测：梯度峰值 + 三指标一致性

        返回:
            {
                'alpha_c': 相变中心,
                'gradient': Q_Y梯度值,
                'consistency': 相变点的一致性,
                'confidence': 综合置信度 = gradient * consistency,
                'all_gradients': {Q_Y, Q_W', Q_X'的梯度}
            }
        """
        grad_QY = self.compute_gradient('Q_Y')
        grad_QW = self.compute_gradient('Q_W_prime')
        grad_QX = self.compute_gradient('Q_X_prime')
        consistency = self.compute_metric_consistency()

        # 相变得分 = Q_Y梯度 × 一致性
        # 物理意义：既要Q_Y快速增长，又要三个指标同步变化
        phase_score = grad_QY * consistency
        alpha_c_idx = np.argmax(phase_score)

        return {
            'alpha_c': self.alphas[alpha_c_idx],
            'index': alpha_c_idx,
            'gradient': grad_QY[alpha_c_idx],
            'consistency': consistency[alpha_c_idx],
            'confidence': phase_score[alpha_c_idx],
            'all_gradients': {
                'Q_Y': grad_QY[alpha_c_idx],
                'Q_W_prime': grad_QW[alpha_c_idx],
                'Q_X_prime': grad_QX[alpha_c_idx]
            },
            'Q_Y_before': self.Q_Y[alpha_c_idx-1] if alpha_c_idx > 0 else None,
            'Q_Y_after': self.Q_Y[alpha_c_idx+1] if alpha_c_idx < len(self.Q_Y)-1 else None,
            'Q_Y_at_transition': self.Q_Y[alpha_c_idx]
        }

    # ========================================
    # 3. 异常区域检测
    # ========================================

    def detect_anomalies(self, metric: str = 'Q_Y') -> List[Dict]:
        """
        检测梯度异常区域

        异常类型:
        1. deceleration: 梯度突然减小（线性增长后突然减速）
        2. decline: 梯度变负（指标下降）
        3. volatility: 梯度剧烈波动

        返回:
            异常列表，每个异常包含 {type, alpha, severity, index}
        """
        grad = self.compute_gradient(metric)
        grad_change = np.gradient(grad, self.alphas)  # 二阶导数

        anomalies = []

        for i in range(1, len(grad)-1):
            # 类型1：梯度突然减小（二阶导数大幅为负）
            if grad[i] > 0.01 and grad_change[i] < -self.decel_threshold:
                anomalies.append({
                    'type': 'deceleration',
                    'alpha': self.alphas[i],
                    'index': i,
                    'severity': abs(grad_change[i]),
                    'description': f'梯度从 {grad[i-1]:.3f} 减速到 {grad[i]:.3f}'
                })

            # 类型2：梯度变负（指标开始下降）
            if grad[i-1] > 0.01 and grad[i] < -0.01:
                anomalies.append({
                    'type': 'decline',
                    'alpha': self.alphas[i],
                    'index': i,
                    'severity': abs(grad[i]),
                    'description': f'指标从增长转为下降，梯度={grad[i]:.3f}'
                })

            # 类型3：梯度剧烈波动
            if abs(grad[i] - grad[i-1]) > self.volatility_threshold:
                anomalies.append({
                    'type': 'volatility',
                    'alpha': self.alphas[i],
                    'index': i,
                    'severity': abs(grad[i] - grad[i-1]),
                    'description': f'梯度剧烈变化: {grad[i-1]:.3f} → {grad[i]:.3f}'
                })

        return anomalies

    # ========================================
    # 4. 区域分类
    # ========================================

    def classify_regions(self,
                        phase_margin: float = 0.5,
                        anomaly_margin: float = 0.3,
                        exclude_phase_overlap: bool = True) -> Dict:
        """
        将alpha范围分为：相变区、异常区、平稳区

        参数:
            phase_margin: 相变点周围的margin范围
            anomaly_margin: 异常点周围的margin范围
            exclude_phase_overlap: 是否排除与相变区重叠的异常区（避免过度采样）

        返回:
            {
                'phase_transition': (start, end),
                'anomalies': [(start1, end1), ...],
                'stable': [(start1, end1), ...]
            }
        """
        phase = self.detect_phase_transition_enhanced()
        anomalies = self.detect_anomalies()

        # 相变区域
        alpha_c = phase['alpha_c']
        phase_region = (
            max(self.alphas[0], alpha_c - phase_margin),
            min(self.alphas[-1], alpha_c + phase_margin)
        )

        # 异常区域（只保留高严重性异常）
        anomaly_regions = []
        for anom in anomalies:
            # 只考虑严重性超过阈值的异常
            if anom['severity'] < self.severity_threshold:
                continue

            a = anom['alpha']
            region = (
                max(self.alphas[0], a - anomaly_margin),
                min(self.alphas[-1], a + anomaly_margin)
            )

            # 排除与相变区重叠的异常（可选）
            if exclude_phase_overlap:
                # 检查是否与相变区重叠
                if not (region[1] < phase_region[0] or region[0] > phase_region[1]):
                    # 有重叠，跳过此异常
                    continue

            anomaly_regions.append(region)

        # 合并重叠的异常区域
        anomaly_regions = self._merge_overlapping_regions(anomaly_regions)

        # 平稳区域（通过排除法得到）
        stable_regions = self._compute_stable_regions(phase_region, anomaly_regions)

        return {
            'phase_transition': phase_region,
            'anomalies': anomaly_regions,
            'stable': stable_regions
        }

    def _merge_overlapping_regions(self, regions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """合并重叠的区间"""
        if not regions:
            return []

        sorted_regions = sorted(regions, key=lambda x: x[0])
        merged = [sorted_regions[0]]

        for current in sorted_regions[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # 有重叠
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)

        return merged

    def _compute_stable_regions(self,
                               phase_region: Tuple[float, float],
                               anomaly_regions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """计算平稳区域（排除相变和异常后的剩余区域）"""
        # 收集所有特殊区域
        special_regions = [phase_region] + anomaly_regions
        special_regions = self._merge_overlapping_regions(special_regions)
        special_regions = sorted(special_regions, key=lambda x: x[0])

        # 找到间隙
        stable = []
        current = self.alphas[0]

        for start, end in special_regions:
            if current < start:
                stable.append((current, start))
            current = max(current, end)

        # 最后一段
        if current < self.alphas[-1]:
            stable.append((current, self.alphas[-1]))

        return stable

    # ========================================
    # 5. 自适应采样建议
    # ========================================

    def suggest_adaptive_sampling(self,
                                 phase_step: float = 0.01,
                                 anomaly_step: float = 0.02,
                                 stable_step: float = 0.5) -> List[Dict]:
        """
        根据区域分类建议采样策略

        参数:
            phase_step: 相变区域步长
            anomaly_step: 异常区域步长
            stable_step: 平稳区域步长

        返回:
            采样计划列表 [{range, step, reason, num_points}, ...]
        """
        regions = self.classify_regions()
        sampling_plan = []

        # 1. 相变区域：密集采样
        start, end = regions['phase_transition']
        num_points = int((end - start) / phase_step) + 1
        sampling_plan.append({
            'range': (start, end),
            'step': phase_step,
            'reason': 'phase_transition',
            'num_points': num_points,
            'alphas': np.arange(start, end + phase_step/2, phase_step).tolist()
        })

        # 2. 异常区域：中等密集
        for start, end in regions['anomalies']:
            num_points = int((end - start) / anomaly_step) + 1
            sampling_plan.append({
                'range': (start, end),
                'step': anomaly_step,
                'reason': 'anomaly',
                'num_points': num_points,
                'alphas': np.arange(start, end + anomaly_step/2, anomaly_step).tolist()
            })

        # 3. 平稳区域：稀疏采样
        for start, end in regions['stable']:
            num_points = int((end - start) / stable_step) + 1
            sampling_plan.append({
                'range': (start, end),
                'step': stable_step,
                'reason': 'stable',
                'num_points': num_points,
                'alphas': np.arange(start, end + stable_step/2, stable_step).tolist()
            })

        return sampling_plan

    def get_all_adaptive_alphas(self, **kwargs) -> np.ndarray:
        """
        获取完整的自适应alpha列表（去重排序）

        参数:
            **kwargs: 传递给 suggest_adaptive_sampling 的参数
        """
        plan = self.suggest_adaptive_sampling(**kwargs)
        all_alphas = []
        for item in plan:
            all_alphas.extend(item['alphas'])

        # 去重并排序
        all_alphas = sorted(set(all_alphas))
        return np.array(all_alphas)

    # ========================================
    # 6. 报告生成
    # ========================================

    def generate_report(self) -> str:
        """生成文本分析报告"""
        phase = self.detect_phase_transition_enhanced()
        anomalies = self.detect_anomalies()
        regions = self.classify_regions()
        sampling = self.suggest_adaptive_sampling()

        report = []
        report.append("=" * 80)
        report.append("相变分析报告 - Phase Transition Analysis Report")
        report.append("=" * 80)
        report.append("")

        # 1. 相变检测结果
        report.append("【1. 相变检测结果】")
        report.append(f"  相变中心 α_c = {phase['alpha_c']:.3f}")
        report.append(f"  Q_Y 梯度 = {phase['gradient']:.4f}")
        report.append(f"  三指标一致性 = {phase['consistency']:.4f}")
        report.append(f"  综合置信度 = {phase['confidence']:.4f}")
        report.append("")
        report.append(f"  相变点处Q_Y值: {phase['Q_Y_at_transition']:.4f}")
        if phase['Q_Y_before'] is not None:
            report.append(f"  相变前Q_Y: {phase['Q_Y_before']:.4f}")
        if phase['Q_Y_after'] is not None:
            report.append(f"  相变后Q_Y: {phase['Q_Y_after']:.4f}")
        report.append("")

        report.append(f"  各指标梯度:")
        report.append(f"    dQ_Y/dα      = {phase['all_gradients']['Q_Y']:.4f}")
        report.append(f"    dQ_W'/dα     = {phase['all_gradients']['Q_W_prime']:.4f}")
        report.append(f"    dQ_X'/dα     = {phase['all_gradients']['Q_X_prime']:.4f}")
        report.append("")

        # 2. 异常检测结果
        report.append("【2. 异常检测结果】")
        if anomalies:
            report.append(f"  检测到 {len(anomalies)} 个异常区域:")
            for i, anom in enumerate(anomalies, 1):
                report.append(f"  {i}. α={anom['alpha']:.3f}, 类型={anom['type']}, "
                            f"严重性={anom['severity']:.4f}")
                report.append(f"     {anom['description']}")
        else:
            report.append("  未检测到异常区域")
        report.append("")

        # 3. 区域分类
        report.append("【3. 区域分类】")
        report.append(f"  相变区域: [{regions['phase_transition'][0]:.2f}, "
                     f"{regions['phase_transition'][1]:.2f}]")

        if regions['anomalies']:
            report.append(f"  异常区域 ({len(regions['anomalies'])} 个):")
            for i, (start, end) in enumerate(regions['anomalies'], 1):
                report.append(f"    {i}. [{start:.2f}, {end:.2f}]")
        else:
            report.append(f"  异常区域: 无")

        if regions['stable']:
            report.append(f"  平稳区域 ({len(regions['stable'])} 个):")
            for i, (start, end) in enumerate(regions['stable'], 1):
                report.append(f"    {i}. [{start:.2f}, {end:.2f}]")
        report.append("")

        # 4. 自适应采样建议
        report.append("【4. 自适应采样建议】")
        total_points = sum(item['num_points'] for item in sampling)
        report.append(f"  总采样点数: {total_points}")
        report.append("")
        for i, item in enumerate(sampling, 1):
            report.append(f"  {i}. {item['reason']} 区域:")
            report.append(f"     范围: [{item['range'][0]:.2f}, {item['range'][1]:.2f}]")
            report.append(f"     步长: Δα = {item['step']}")
            report.append(f"     采样点数: {item['num_points']}")
        report.append("")

        # 5. 对比均匀采样
        uniform_points_01 = int((3.0 - 0.0) / 0.01) + 1
        uniform_points_02 = int((3.0 - 0.0) / 0.02) + 1
        uniform_points_05 = int((3.0 - 0.0) / 0.05) + 1

        report.append("【5. 计算量对比】")
        report.append(f"  自适应采样: {total_points} 个点")
        report.append(f"  均匀采样 Δα=0.05: {uniform_points_05} 个点 "
                     f"(节省 {(1-total_points/uniform_points_05)*100:.1f}%)")
        report.append(f"  均匀采样 Δα=0.02: {uniform_points_02} 个点 "
                     f"(节省 {(1-total_points/uniform_points_02)*100:.1f}%)")
        report.append(f"  均匀采样 Δα=0.01: {uniform_points_01} 个点 "
                     f"(节省 {(1-total_points/uniform_points_01)*100:.1f}%)")
        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    # ========================================
    # 7. 可视化
    # ========================================

    def plot_full_analysis(self, save_path: Optional[Path] = None, show: bool = True):
        """
        完整分析可视化（4子图）

        1. Q_Y, Q_W', Q_X' 曲线
        2. 三者梯度曲线 + 一致性
        3. 一致性热图 + 相变标记
        4. 区域分类标记
        """
        phase = self.detect_phase_transition_enhanced()
        anomalies = self.detect_anomalies()
        regions = self.classify_regions()
        consistency = self.compute_metric_consistency()

        grad_QY = self.compute_gradient('Q_Y')
        grad_QW = self.compute_gradient('Q_W_prime')
        grad_QX = self.compute_gradient('Q_X_prime')

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # ========================================
        # 子图1: Q_Y, Q_W', Q_X' 曲线
        # ========================================
        ax1 = axes[0, 0]
        ax1.plot(self.alphas, self.Q_Y, 'b-', linewidth=2, label='Q_Y')
        ax1.plot(self.alphas, self.Q_W_prime, 'r--', linewidth=1.5, label="Q_W'")
        ax1.plot(self.alphas, self.Q_X_prime, 'g--', linewidth=1.5, label="Q_X'")

        # 标记相变点
        ax1.axvline(phase['alpha_c'], color='purple', linestyle=':', linewidth=2,
                   label=f"相变点 α_c={phase['alpha_c']:.2f}")

        ax1.set_xlabel('Alpha (α)', fontsize=12)
        ax1.set_ylabel('Overlap Metrics', fontsize=12)
        ax1.set_title('Overlap Metrics vs Alpha', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # ========================================
        # 子图2: 梯度曲线
        # ========================================
        ax2 = axes[0, 1]
        ax2.plot(self.alphas, grad_QY, 'b-', linewidth=2, label='dQ_Y/dα')
        ax2.plot(self.alphas, grad_QW, 'r--', linewidth=1.5, label="dQ_W'/dα")
        ax2.plot(self.alphas, grad_QX, 'g--', linewidth=1.5, label="dQ_X'/dα")

        # 标记相变点
        ax2.axvline(phase['alpha_c'], color='purple', linestyle=':', linewidth=2)
        ax2.scatter([phase['alpha_c']], [phase['gradient']],
                   color='purple', s=200, marker='*', zorder=5,
                   label=f"最大梯度={phase['gradient']:.3f}")

        # 标记异常点
        for anom in anomalies:
            ax2.axvline(anom['alpha'], color='orange', linestyle='--', alpha=0.5, linewidth=1)

        ax2.set_xlabel('Alpha (α)', fontsize=12)
        ax2.set_ylabel('Gradient (dQ/dα)', fontsize=12)
        ax2.set_title('Gradient Analysis', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # ========================================
        # 子图3: 一致性分析
        # ========================================
        ax3 = axes[1, 0]

        # 一致性曲线
        ax3_twin = ax3.twinx()

        line1 = ax3.plot(self.alphas, consistency, 'purple', linewidth=2.5,
                        label='三指标一致性')
        ax3.fill_between(self.alphas, 0, consistency, alpha=0.3, color='purple')

        # Q_Y曲线（作为参考）
        line2 = ax3_twin.plot(self.alphas, self.Q_Y, 'b--', linewidth=1.5,
                             alpha=0.6, label='Q_Y (参考)')

        # 标记相变点
        ax3.axvline(phase['alpha_c'], color='red', linestyle=':', linewidth=2)
        ax3.scatter([phase['alpha_c']], [phase['consistency']],
                   color='red', s=200, marker='*', zorder=5,
                   label=f"相变点一致性={phase['consistency']:.3f}")

        ax3.set_xlabel('Alpha (α)', fontsize=12)
        ax3.set_ylabel('Consistency (三指标一致性)', fontsize=12, color='purple')
        ax3_twin.set_ylabel('Q_Y', fontsize=12, color='b')
        ax3.set_title('Metric Consistency Analysis', fontsize=14, fontweight='bold')

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='best')

        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)

        # ========================================
        # 子图4: 区域分类
        # ========================================
        ax4 = axes[1, 1]

        # Q_Y曲线作为背景
        ax4.plot(self.alphas, self.Q_Y, 'k-', linewidth=2, alpha=0.3, label='Q_Y')

        # 用不同颜色标记不同区域
        # 相变区域
        start, end = regions['phase_transition']
        ax4.axvspan(start, end, alpha=0.3, color='red', label='相变区域')

        # 异常区域
        for i, (start, end) in enumerate(regions['anomalies']):
            label = '异常区域' if i == 0 else None
            ax4.axvspan(start, end, alpha=0.3, color='orange', label=label)

        # 平稳区域
        for i, (start, end) in enumerate(regions['stable']):
            label = '平稳区域' if i == 0 else None
            ax4.axvspan(start, end, alpha=0.2, color='green', label=label)

        ax4.set_xlabel('Alpha (α)', fontsize=12)
        ax4.set_ylabel('Q_Y', fontsize=12)
        ax4.set_title('Region Classification', fontsize=14, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # ========================================
        # 整体布局
        # ========================================
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")

        if show:
            plt.show()

        return fig

    def save_analysis_json(self, save_path: Path):
        """保存分析结果为JSON"""
        phase = self.detect_phase_transition_enhanced()
        anomalies = self.detect_anomalies()
        regions = self.classify_regions()
        sampling = self.suggest_adaptive_sampling()

        result = {
            'phase_transition': {
                'alpha_c': float(phase['alpha_c']),
                'gradient': float(phase['gradient']),
                'consistency': float(phase['consistency']),
                'confidence': float(phase['confidence']),
                'all_gradients': {k: float(v) for k, v in phase['all_gradients'].items()}
            },
            'anomalies': [
                {
                    'type': a['type'],
                    'alpha': float(a['alpha']),
                    'severity': float(a['severity']),
                    'description': a['description']
                }
                for a in anomalies
            ],
            'regions': {
                'phase_transition': [float(regions['phase_transition'][0]),
                                    float(regions['phase_transition'][1])],
                'anomalies': [[float(s), float(e)] for s, e in regions['anomalies']],
                'stable': [[float(s), float(e)] for s, e in regions['stable']]
            },
            'adaptive_sampling': sampling
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"分析结果已保存: {save_path}")

    # ========================================
    # 8. 功能2: 简单自适应分析（智能采样加速）
    # ========================================

    @staticmethod
    def simple_adaptive_analysis(
        train_callback,
        alpha_range: Tuple[float, float] = (0.0, 3.0),
        coarse_epochs: int = 2000,
        coarse_step: float = 0.1,
        phase_step: float = 0.02,
        stable_step: float = 0.2,
        fine_epochs: int = 20000,
        sensitivity: str = 'medium',
        save_smart_alphas: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict:
        """
        简单自适应分析：两阶段训练（粗扫 + 智能采样）

        工作流程：
        1. 粗扫：用均匀稀疏alpha（Δα=coarse_step）进行快速训练
        2. 分析：检测相变和异常区域
        3. 生成智能alpha：相变区密集、平稳区稀疏
        4. 精细训练：用智能alpha进行高质量训练

        参数:
            train_callback: 训练函数 callback(alphas_array, epochs) -> metrics_dict
            alpha_range: alpha范围 (start, stop)
            coarse_epochs: 粗扫训练轮数
            coarse_step: 粗扫步长
            phase_step: 相变区域精细步长
            stable_step: 平稳区域稀疏步长
            fine_epochs: 精细训练轮数
            sensitivity: 异常检测灵敏度 ('strict', 'medium', 'loose')
            save_smart_alphas: 保存智能alpha列表的路径（用于对拍）
            verbose: 是否打印详细信息

        返回:
            {
                'coarse_results': 粗扫结果,
                'coarse_alphas': 粗扫alpha列表,
                'fine_results': 精细训练结果,
                'smart_alphas': 智能alpha列表,
                'phase_analysis': 相变分析结果,
                'adaptive_plan': 自适应采样计划,
                'speedup': 加速倍率（vs均匀密集采样）
            }
        """
        if verbose:
            print("=" * 80)
            print("简单自适应分析 - Simple Adaptive Analysis")
            print("=" * 80)
            print(f"Alpha范围: [{alpha_range[0]:.1f}, {alpha_range[1]:.1f}]")
            print(f"粗扫: Δα={coarse_step}, epochs={coarse_epochs}")
            print(f"精细: 相变区Δα={phase_step}, 平稳区Δα={stable_step}, epochs={fine_epochs}")
            print(f"异常检测灵敏度: {sensitivity}")
            print("=" * 80)

        # ========================================
        # 阶段1：粗扫
        # ========================================
        if verbose:
            print("\n[阶段 1/3] 粗扫训练...")

        coarse_alphas = np.arange(alpha_range[0], alpha_range[1] + coarse_step/2, coarse_step)

        if verbose:
            print(f"  粗扫alpha数量: {len(coarse_alphas)}")
            print(f"  开始粗扫训练（epochs={coarse_epochs}）...")

        coarse_results = train_callback(coarse_alphas, coarse_epochs)

        if verbose:
            print(f"  ✓ 粗扫完成")

        # ========================================
        # 阶段2：分析并生成智能alpha
        # ========================================
        if verbose:
            print("\n[阶段 2/3] 分析相变并生成智能alpha列表...")

        # 设置异常检测阈值
        if sensitivity == 'strict':
            severity_threshold = 3.0
        elif sensitivity == 'loose':
            severity_threshold = 0.5
        else:  # medium
            severity_threshold = 2.0

        # 创建分析器
        analyzer = PhaseTransitionAnalyzer(
            coarse_alphas,
            coarse_results,
            severity_threshold=severity_threshold
        )

        # 检测相变
        phase = analyzer.detect_phase_transition_enhanced()

        if verbose:
            print(f"  检测到相变点: α_c = {phase['alpha_c']:.3f}")
            print(f"  相变梯度: {phase['gradient']:.4f}")
            print(f"  三指标一致性: {phase['consistency']:.4f}")

        # 生成自适应采样计划
        adaptive_plan = analyzer.suggest_adaptive_sampling(
            phase_step=phase_step,
            anomaly_step=phase_step,  # 异常区域使用相同密度
            stable_step=stable_step
        )

        # 获取完整智能alpha列表
        smart_alphas = analyzer.get_all_adaptive_alphas(
            phase_step=phase_step,
            anomaly_step=phase_step,
            stable_step=stable_step
        )

        total_smart_points = len(smart_alphas)
        uniform_dense_points = int((alpha_range[1] - alpha_range[0]) / phase_step) + 1
        speedup = uniform_dense_points / total_smart_points

        if verbose:
            print(f"\n  自适应采样计划:")
            for i, plan in enumerate(adaptive_plan, 1):
                print(f"    {i}. {plan['reason']:20s} [{plan['range'][0]:.2f}, {plan['range'][1]:.2f}]  "
                      f"Δα={plan['step']:5.2f}  {plan['num_points']:3d}点")
            print(f"\n  智能alpha总数: {total_smart_points}")
            print(f"  均匀密集采样 (Δα={phase_step}): {uniform_dense_points}点")
            print(f"  加速倍率: {speedup:.2f}x")

        # 保存智能alpha列表（用于对拍）
        if save_smart_alphas:
            np.save(save_smart_alphas, smart_alphas)
            if verbose:
                print(f"  ✓ 智能alpha列表已保存: {save_smart_alphas}")

        # ========================================
        # 阶段3：精细训练
        # ========================================
        if verbose:
            print(f"\n[阶段 3/3] 使用智能alpha进行精细训练...")
            print(f"  智能alpha数量: {len(smart_alphas)}")
            print(f"  开始精细训练（epochs={fine_epochs}）...")

        fine_results = train_callback(smart_alphas, fine_epochs)

        if verbose:
            print(f"  ✓ 精细训练完成")

        # ========================================
        # 返回结果
        # ========================================
        if verbose:
            print("\n" + "=" * 80)
            print("✓ 简单自适应分析完成")
            print("=" * 80)

        return {
            'coarse_results': coarse_results,
            'coarse_alphas': coarse_alphas.tolist(),
            'fine_results': fine_results,
            'smart_alphas': smart_alphas.tolist(),
            'phase_analysis': {
                'alpha_c': float(phase['alpha_c']),
                'gradient': float(phase['gradient']),
                'consistency': float(phase['consistency']),
                'confidence': float(phase['confidence'])
            },
            'adaptive_plan': adaptive_plan,
            'speedup': {
                'smart_points': total_smart_points,
                'uniform_dense_points': uniform_dense_points,
                'speedup_factor': float(speedup)
            }
        }

    # ========================================
    # 9. 功能3: 精密相变分析（热力学极限）
    # ========================================

    @staticmethod
    def precise_phase_analysis(
        train_callback,
        initial_alpha_c: float,
        N1: int,
        N2: int,
        M: int,
        initial_window: float = 0.5,
        initial_step: float = 0.05,
        initial_epochs: int = 20000,
        epoch_multiplier: float = 2.5,
        max_rounds: int = 10,
        alpha_c_tolerance: float = 0.001,
        gradient_tolerance: float = 0.01,
        convergence_patience: int = 3,
        result_dir: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict:
        """
        精密相变分析：通过逐步增加epochs和收窄范围，确定热力学极限下的相变点

        核心思路：
        - 只训练相变点附近小范围（α_c ± window）
        - 每轮增加epochs → 提高精度
        - 逐步收窄window → 聚焦相变点
        - 检测收敛 → 3轮稳定即停止
        - 外推到热力学极限（epochs → ∞）

        参数:
            train_callback: 训练函数 callback(alphas_array, epochs) -> metrics_dict
            initial_alpha_c: 初始相变点估计（从粗扫或Mode 2获得）
            N1, N2, M: 矩阵维度（用于结果目录命名）
            initial_window: 初始搜索窗口 (α_c ± window)
            initial_step: 初始alpha步长
            initial_epochs: 初始训练epochs
            epoch_multiplier: 每轮epochs倍增系数
            max_rounds: 最大迭代轮数
            alpha_c_tolerance: α_c收敛判断阈值
            gradient_tolerance: 梯度相对变化阈值
            convergence_patience: 连续稳定轮数（默认3轮）
            result_dir: 结果保存目录
            verbose: 是否打印详细信息

        返回:
            {
                'alpha_c_thermodynamic': 热力学极限下的α_c（外推值）,
                'alpha_c_final': 最后一轮的α_c,
                'history': 各轮训练历史,
                'converged': 是否收敛,
                'total_rounds': 实际迭代轮数,
                'confidence_interval': α_c的置信区间
            }
        """
        if verbose:
            print("=" * 80)
            print("精密相变分析 - Precise Phase Transition Analysis")
            print("=" * 80)
            print(f"矩阵尺寸: {N1}×{N2}×{M}")
            print(f"初始α_c估计: {initial_alpha_c:.3f}")
            print(f"初始window: ±{initial_window}")
            print(f"初始epochs: {initial_epochs}")
            print(f"Epochs倍增: {epoch_multiplier}x per round")
            print(f"收敛条件: |Δα_c| < {alpha_c_tolerance}, {convergence_patience}轮稳定")
            print("=" * 80)

        # 准备结果目录
        if result_dir is None:
            result_dir = Path(f"precise_phase_analysis/Result/{N1}_{N2}_{M}")
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        # 历史记录
        history = []
        alpha_c_list = []
        gradient_list = []
        epochs_list = []

        # 迭代参数
        current_alpha_c = initial_alpha_c
        current_window = initial_window
        current_step = initial_step
        current_epochs = initial_epochs

        converged = False
        stable_count = 0  # 连续稳定计数

        for round_num in range(1, max_rounds + 1):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Round {round_num}/{max_rounds}")
                print(f"{'='*80}")
                print(f"  α_c中心: {current_alpha_c:.4f}")
                print(f"  搜索范围: [{current_alpha_c - current_window:.4f}, "
                      f"{current_alpha_c + current_window:.4f}]")
                print(f"  步长: Δα = {current_step:.4f}")
                print(f"  Epochs: {current_epochs}")

            # 生成alpha范围
            alpha_start = max(0.0, current_alpha_c - current_window)
            alpha_end = current_alpha_c + current_window
            alphas = np.arange(alpha_start, alpha_end + current_step/2, current_step)

            if verbose:
                print(f"  Alpha点数: {len(alphas)}")
                print(f"  开始训练...")

            # 训练
            round_start_time = time.time()
            results = train_callback(alphas, current_epochs)
            round_duration = time.time() - round_start_time

            if verbose:
                print(f"  ✓ 训练完成 ({round_duration/60:.1f}分钟)")

            # 相变分析
            analyzer = PhaseTransitionAnalyzer(alphas, results)
            phase = analyzer.detect_phase_transition_enhanced()

            detected_alpha_c = phase['alpha_c']
            detected_gradient = phase['gradient']
            consistency = phase['consistency']

            if verbose:
                print(f"\n  [分析结果]")
                print(f"    检测到α_c: {detected_alpha_c:.4f}")
                print(f"    梯度: {detected_gradient:.4f}")
                print(f"    一致性: {consistency:.4f}")

            # 保存本轮结果
            round_result = {
                'round': round_num,
                'epochs': current_epochs,
                'alpha_c': float(detected_alpha_c),
                'gradient': float(detected_gradient),
                'consistency': float(consistency),
                'window': current_window,
                'step': current_step,
                'num_alphas': len(alphas),
                'duration_minutes': round_duration / 60,
                'alphas': alphas.tolist()
            }
            history.append(round_result)
            alpha_c_list.append(detected_alpha_c)
            gradient_list.append(detected_gradient)
            epochs_list.append(current_epochs)

            # 保存JSON
            round_json_path = result_dir / f"round{round_num}_epoch{current_epochs}.json"
            with open(round_json_path, 'w') as f:
                json.dump(results, f, indent=2)
            if verbose:
                print(f"  ✓ 结果已保存: {round_json_path.name}")

            # 收敛检测
            if round_num >= 2:
                alpha_c_change = abs(alpha_c_list[-1] - alpha_c_list[-2])
                gradient_change_rel = abs(gradient_list[-1] - gradient_list[-2]) / (abs(gradient_list[-2]) + 1e-8)

                if verbose:
                    print(f"\n  [收敛检测]")
                    print(f"    |Δα_c| = {alpha_c_change:.5f} (阈值: {alpha_c_tolerance})")
                    print(f"    |Δgradient|/|gradient| = {gradient_change_rel:.5f} (阈值: {gradient_tolerance})")

                # 检查是否满足收敛条件
                if alpha_c_change < alpha_c_tolerance and gradient_change_rel < gradient_tolerance:
                    stable_count += 1
                    if verbose:
                        print(f"    ✓ 稳定 ({stable_count}/{convergence_patience})")

                    if stable_count >= convergence_patience:
                        converged = True
                        if verbose:
                            print(f"\n  ✓✓✓ 收敛！连续{convergence_patience}轮稳定")
                        break
                else:
                    stable_count = 0
                    if verbose:
                        print(f"    ✗ 未稳定，重置计数器")

            # 准备下一轮
            if round_num < max_rounds:
                # 更新α_c中心
                current_alpha_c = detected_alpha_c

                # 收窄window (每轮缩小到60%)
                current_window = current_window * 0.6

                # 细化步长 (每轮缩小到70%)
                current_step = current_step * 0.7

                # 增加epochs
                current_epochs = int(current_epochs * epoch_multiplier)

                if verbose:
                    print(f"\n  [下一轮参数]")
                    print(f"    新α_c中心: {current_alpha_c:.4f}")
                    print(f"    新window: ±{current_window:.4f}")
                    print(f"    新步长: Δα = {current_step:.5f}")
                    print(f"    新epochs: {current_epochs}")

        # 热力学极限外推
        if verbose:
            print(f"\n{'='*80}")
            print("热力学极限外推 (Thermodynamic Limit Extrapolation)")
            print(f"{'='*80}")

        # 使用 1/epochs 作为x轴，α_c作为y轴进行线性拟合
        # α_c(epochs) ≈ α_c(∞) + A/epochs
        # 即 α_c vs 1/epochs 应该是线性的

        if len(alpha_c_list) >= 3:
            inv_epochs = np.array([1.0/e for e in epochs_list])
            alpha_c_array = np.array(alpha_c_list)

            # 线性拟合: α_c = intercept + slope * (1/epochs)
            # intercept 就是 α_c(epochs→∞)
            coeffs = np.polyfit(inv_epochs, alpha_c_array, deg=1)
            slope, intercept = coeffs[0], coeffs[1]

            alpha_c_thermodynamic = intercept

            # 计算R²和置信区间
            alpha_c_pred = np.polyval(coeffs, inv_epochs)
            ss_res = np.sum((alpha_c_array - alpha_c_pred) ** 2)
            ss_tot = np.sum((alpha_c_array - np.mean(alpha_c_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # 简单置信区间估计（基于残差标准差）
            residuals_std = np.sqrt(ss_res / (len(alpha_c_list) - 2))
            confidence_interval = (
                alpha_c_thermodynamic - 2 * residuals_std,
                alpha_c_thermodynamic + 2 * residuals_std
            )

            if verbose:
                print(f"  拟合方程: α_c(epochs) = {intercept:.5f} + {slope:.5f} / epochs")
                print(f"  R² = {r_squared:.6f}")
                print(f"\n  ✓ 热力学极限 (epochs→∞):")
                print(f"    α_c(∞) = {alpha_c_thermodynamic:.5f}")
                print(f"    95%置信区间: [{confidence_interval[0]:.5f}, {confidence_interval[1]:.5f}]")
        else:
            # 数据点太少，无法外推
            alpha_c_thermodynamic = alpha_c_list[-1]
            confidence_interval = None
            r_squared = None
            if verbose:
                print(f"  数据点不足（{len(alpha_c_list)} < 3），无法外推")
                print(f"  使用最后一轮结果: α_c = {alpha_c_thermodynamic:.5f}")

        # 保存收敛历史
        convergence_history = {
            'converged': converged,
            'total_rounds': len(history),
            'alpha_c_thermodynamic': float(alpha_c_thermodynamic) if alpha_c_thermodynamic is not None else None,
            'alpha_c_final': float(alpha_c_list[-1]),
            'confidence_interval': [float(confidence_interval[0]), float(confidence_interval[1])] if confidence_interval else None,
            'r_squared': float(r_squared) if r_squared is not None else None,
            'history': history
        }

        history_path = result_dir / "convergence_history.json"
        with open(history_path, 'w') as f:
            json.dump(convergence_history, f, indent=2)

        if verbose:
            print(f"\n✓ 收敛历史已保存: {history_path}")

        # 生成收敛曲线图
        if len(alpha_c_list) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 子图1: α_c vs Round
            ax1 = axes[0, 0]
            ax1.plot(range(1, len(alpha_c_list)+1), alpha_c_list, 'bo-', linewidth=2, markersize=8)
            ax1.axhline(alpha_c_thermodynamic, color='r', linestyle='--', label=f'α_c(∞) = {alpha_c_thermodynamic:.5f}')
            ax1.set_xlabel('Round', fontsize=12)
            ax1.set_ylabel('α_c', fontsize=12)
            ax1.set_title('Phase Transition Point Convergence', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 子图2: α_c vs 1/epochs (热力学极限外推)
            ax2 = axes[0, 1]
            if len(alpha_c_list) >= 3:
                inv_epochs = np.array([1.0/e for e in epochs_list])
                ax2.plot(inv_epochs, alpha_c_list, 'go', markersize=10, label='Training results')

                # 拟合线
                x_fit = np.linspace(0, max(inv_epochs), 100)
                y_fit = slope * x_fit + intercept
                ax2.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Fit: {intercept:.5f} + {slope:.5f}/epochs')
                ax2.plot([0], [alpha_c_thermodynamic], 'r*', markersize=20, label=f'α_c(∞) = {alpha_c_thermodynamic:.5f}')

                ax2.set_xlabel('1 / epochs', fontsize=12)
                ax2.set_ylabel('α_c', fontsize=12)
                ax2.set_title(f'Thermodynamic Limit Extrapolation (R²={r_squared:.4f})', fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # 子图3: Gradient vs Round
            ax3 = axes[1, 0]
            ax3.plot(range(1, len(gradient_list)+1), gradient_list, 'mo-', linewidth=2, markersize=8)
            ax3.set_xlabel('Round', fontsize=12)
            ax3.set_ylabel('Gradient (dQ_Y/dα)', fontsize=12)
            ax3.set_title('Gradient Convergence', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # 子图4: 收敛指标
            ax4 = axes[1, 1]
            if len(alpha_c_list) >= 2:
                alpha_c_changes = [abs(alpha_c_list[i] - alpha_c_list[i-1]) for i in range(1, len(alpha_c_list))]
                ax4.semilogy(range(2, len(alpha_c_list)+1), alpha_c_changes, 'co-', linewidth=2, markersize=8, label='|Δα_c|')
                ax4.axhline(alpha_c_tolerance, color='r', linestyle='--', label=f'Threshold = {alpha_c_tolerance}')
                ax4.set_xlabel('Round', fontsize=12)
                ax4.set_ylabel('|Δα_c|', fontsize=12)
                ax4.set_title('Convergence Criteria', fontsize=14, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3, which='both')

            plt.tight_layout()
            plot_path = result_dir / "thermodynamic_limit_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            if verbose:
                print(f"✓ 分析图表已保存: {plot_path}")

        if verbose:
            print(f"\n{'='*80}")
            print("✓ 精密相变分析完成")
            print(f"{'='*80}")

        return convergence_history


if __name__ == "__main__":
    # 简单测试
    print("PhaseTransitionAnalyzer V2 类已定义（含Mode 3: 精密相变分析）")
    print("使用 run_precise_analysis.py 进行完整测试")
