"""
梯度自适应采样器

核心思想:
- **保持总采样点数不变**
- 根据梯度重新分布采样点
- 相变区（高梯度）→ 高密度采样
- 稳定区（低梯度）→ 低密度采样
- 最终获得比均匀采样更好的曲线质量

设计原则:
- 总计算量不变，只是重新分配
- 数据驱动：基于实际梯度分布
- 曲线质量优先：在关键区域获得更精细的分辨率
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from scipy import interpolate


@dataclass
class AdaptiveSamplingResult:
    """自适应采样结果"""
    original_alphas: np.ndarray     # 原始均匀采样点
    adaptive_alphas: np.ndarray     # 自适应采样点
    density_function: np.ndarray    # 采样密度函数
    gradient_profile: np.ndarray    # 梯度轮廓
    phase_zone: Tuple[float, float] # 检测到的相变区
    quality_improvement: float      # 预估质量提升


class GradientAdaptiveSampler:
    """
    梯度自适应采样器

    基于粗扫描的梯度信息，在**保持总点数不变**的情况下
    重新分布采样点，使得相变区获得更高的采样密度
    """

    def __init__(self,
                 alphas: np.ndarray,
                 Q_Y: np.ndarray,
                 smooth_sigma: float = 1.0):
        """
        参数:
            alphas: 粗扫描的 alpha 值
            Q_Y: 对应的 Q_Y 值
            smooth_sigma: 梯度平滑参数
        """
        # 排序
        sort_idx = np.argsort(alphas)
        self.alphas = np.asarray(alphas)[sort_idx]
        self.Q_Y = np.asarray(Q_Y)[sort_idx]
        self.smooth_sigma = smooth_sigma

        # 预计算梯度
        self._compute_gradient()

    def _compute_gradient(self):
        """计算平滑后的梯度"""
        # 原始梯度
        raw_grad = np.gradient(self.Q_Y, self.alphas)

        # 高斯平滑
        if self.smooth_sigma > 0 and len(raw_grad) > 3:
            from scipy.ndimage import gaussian_filter1d
            self.gradient = gaussian_filter1d(np.abs(raw_grad), self.smooth_sigma)
        else:
            self.gradient = np.abs(raw_grad)

        # 归一化
        self.gradient = self.gradient / (self.gradient.max() + 1e-10)

    def _find_phase_zone(self, fwhm_ratio: float = 0.7) -> Tuple[float, float]:
        """
        找到相变区 (梯度显著高于平均的区域)

        改进的 FWHM 方法:
        1. 排除边界效应（前25%后10%），避免 alpha=0 处的虚假高梯度
        2. 使用 fwhm_ratio * max_gradient 作为阈值
        3. 从峰值向两侧找到阈值交叉点

        参数:
            fwhm_ratio: 半高宽比例 (默认 0.5 = 半高宽)
        """
        n = len(self.gradient)
        if n < 5:
            return (float(self.alphas[0]), float(self.alphas[-1]))

        # 非常激进地排除边界：
        # 左边排除 35%（避免 alpha=0 附近的虚假梯度 - Q_Y从0跳到正值）
        # 右边排除 10%（饱和区梯度本来就低）
        left_margin = max(4, int(n * 0.35))
        right_margin = max(2, n // 10)

        # 确保有足够的内部区域
        if left_margin + right_margin >= n - 3:
            left_margin = max(2, n // 5)
            right_margin = max(1, n // 10)

        interior_gradient = self.gradient[left_margin:-right_margin]

        if len(interior_gradient) < 3:
            interior_gradient = self.gradient[2:-2]
            left_margin = 2

        # 在内部区域找最大梯度
        max_interior_idx = np.argmax(interior_gradient)
        max_idx = max_interior_idx + left_margin  # 转换回原始索引
        max_grad = self.gradient[max_idx]

        # FWHM 阈值
        threshold = fwhm_ratio * max_grad

        # 从最大梯度点向左找边界，但不要越过安全边界
        # 安全边界：至少跳过前20%的点（避免边界假象）
        safe_left_limit = max(2, n // 5)
        left_idx = max_idx
        while left_idx > safe_left_limit and self.gradient[left_idx] > threshold:
            left_idx -= 1

        # 从最大梯度点向右找边界
        right_idx = max_idx
        while right_idx < n - 1 and self.gradient[right_idx] > threshold:
            right_idx += 1

        # 添加小余量 (约 10% 的相变区宽度)
        width = right_idx - left_idx
        margin = max(1, width // 10)
        left_idx = max(0, left_idx - margin)
        right_idx = min(n - 1, right_idx + margin)

        return (float(self.alphas[left_idx]), float(self.alphas[right_idx]))

    def compute_density_function(self,
                                  base_density: float = 0.1,
                                  gradient_weight: float = 0.9,
                                  power: float = 3.0) -> np.ndarray:
        """
        计算采样密度函数 (使用幂次增强对比度)

        采样密度 ∝ base_density + gradient_weight * gradient^power

        参数:
            base_density: 基础密度 (稳定区最小密度)
            gradient_weight: 梯度权重 (相变区额外密度)
            power: 梯度幂次 (增大可以压制低梯度区)

        返回:
            密度函数 (归一化到积分为 1)
        """
        # 使用幂次压制低梯度区域，增强高梯度区域
        # gradient^3 将 0.3 -> 0.027, 但 1.0 -> 1.0
        enhanced_gradient = np.power(self.gradient, power)

        # 密度 = 基础 + 梯度加权
        density = base_density + gradient_weight * enhanced_gradient

        # 归一化使得积分为 1
        integral = np.trapz(density, self.alphas)
        density = density / integral

        return density

    def generate_adaptive_alphas(self,
                                  n_points: int,
                                  base_density: float = 0.1,
                                  gradient_weight: float = 0.9,
                                  power: float = 3.0) -> np.ndarray:
        """
        生成自适应采样点

        核心算法: 逆CDF采样
        1. 计算密度函数 p(α)
        2. 计算累积分布函数 F(α) = ∫p(α)dα
        3. 在 [0,1] 均匀采样 u_i
        4. 求 α_i = F^{-1}(u_i)

        参数:
            n_points: 目标采样点数 (与原始相同)
            base_density: 基础密度
            gradient_weight: 梯度权重
            power: 梯度幂次 (控制对比度)

        返回:
            自适应采样的 alpha 数组
        """
        # 1. 计算密度函数
        density = self.compute_density_function(base_density, gradient_weight, power)

        # 2. 计算累积分布函数 (CDF)
        # 使用累积梯形积分
        cdf = np.zeros_like(density)
        for i in range(1, len(density)):
            cdf[i] = cdf[i-1] + 0.5 * (density[i] + density[i-1]) * (self.alphas[i] - self.alphas[i-1])

        # 归一化 CDF 到 [0, 1]
        cdf = cdf / cdf[-1]

        # 3. 在 [0, 1] 均匀采样
        # 使用 linspace 确保边界被包含
        u = np.linspace(0, 1, n_points)

        # 4. 逆 CDF 采样
        # 创建插值函数 F^{-1}
        # 处理 CDF 中可能的重复值
        unique_mask = np.diff(cdf, prepend=-1) > 1e-10
        cdf_unique = cdf[unique_mask]
        alphas_unique = self.alphas[unique_mask]

        # 确保端点被包含
        if cdf_unique[0] > 0:
            cdf_unique = np.concatenate([[0], cdf_unique])
            alphas_unique = np.concatenate([[self.alphas[0]], alphas_unique])
        if cdf_unique[-1] < 1:
            cdf_unique = np.concatenate([cdf_unique, [1]])
            alphas_unique = np.concatenate([alphas_unique, [self.alphas[-1]]])

        inv_cdf = interpolate.interp1d(cdf_unique, alphas_unique,
                                       kind='linear', bounds_error=False,
                                       fill_value=(self.alphas[0], self.alphas[-1]))

        adaptive_alphas = inv_cdf(u)

        # 确保边界
        adaptive_alphas = np.clip(adaptive_alphas, self.alphas[0], self.alphas[-1])

        # 去重并排序
        adaptive_alphas = np.unique(np.round(adaptive_alphas, decimals=6))

        return adaptive_alphas

    def redistribute_zone_based(self,
                                 n_points: Optional[int] = None,
                                 phase_fraction: float = 0.6) -> AdaptiveSamplingResult:
        """
        基于区域的显式分配采样 (视觉效果更明显)

        策略：
        - 相变区获得 phase_fraction 比例的点
        - 其余点均匀分配给相变区外

        参数:
            n_points: 目标点数
            phase_fraction: 相变区获得的点数比例 (默认0.6 = 60%)
        """
        if n_points is None:
            n_points = len(self.alphas)

        # 找相变区
        phase_zone = self._find_phase_zone()
        alpha_min, alpha_max = self.alphas[0], self.alphas[-1]

        # 计算各区域点数
        n_phase = int(n_points * phase_fraction)
        n_outside = n_points - n_phase

        # 计算各区域宽度
        below_width = phase_zone[0] - alpha_min
        above_width = alpha_max - phase_zone[1]
        total_outside_width = below_width + above_width

        # 按宽度比例分配外部点
        if total_outside_width > 0:
            n_below = max(2, int(n_outside * below_width / total_outside_width))
            n_above = max(2, n_outside - n_below)
        else:
            n_below = n_outside // 2
            n_above = n_outside - n_below

        # 生成各区域的点
        points = []

        # 相变区前
        if n_below > 0 and below_width > 0:
            below_pts = np.linspace(alpha_min, phase_zone[0], n_below, endpoint=False)
            points.extend(below_pts)

        # 相变区 (密集采样)
        phase_pts = np.linspace(phase_zone[0], phase_zone[1], n_phase)
        points.extend(phase_pts)

        # 相变区后
        if n_above > 0 and above_width > 0:
            above_pts = np.linspace(phase_zone[1], alpha_max, n_above + 1)[1:]  # 跳过第一个避免重复
            points.extend(above_pts)

        adaptive_alphas = np.unique(np.round(np.array(points), decimals=6))

        # 计算密度函数（用于可视化）
        density = self.compute_density_function(0.05, 0.95, 3.0)

        # 计算质量提升
        phase_mask = (adaptive_alphas >= phase_zone[0]) & (adaptive_alphas <= phase_zone[1])
        uniform_density = len(self.alphas) / (alpha_max - alpha_min)
        phase_width = phase_zone[1] - phase_zone[0]
        adaptive_phase_density = phase_mask.sum() / phase_width if phase_width > 0 else 1

        quality_improvement = adaptive_phase_density / uniform_density

        return AdaptiveSamplingResult(
            original_alphas=self.alphas,
            adaptive_alphas=adaptive_alphas,
            density_function=density,
            gradient_profile=self.gradient,
            phase_zone=phase_zone,
            quality_improvement=quality_improvement,
        )

    def redistribute(self,
                     n_points: Optional[int] = None,
                     min_density_ratio: float = 0.05,
                     max_density_ratio: float = 15.0,
                     power: float = 3.0) -> AdaptiveSamplingResult:
        """
        重新分布采样点

        参数:
            n_points: 目标点数 (默认与原始相同)
            min_density_ratio: 最小密度比 (相对于均匀采样)
            max_density_ratio: 最大密度比 (相对于均匀采样)
            power: 梯度幂次 (控制对比度，越大越集中在相变区)

        返回:
            AdaptiveSamplingResult 对象
        """
        if n_points is None:
            n_points = len(self.alphas)

        # 计算密度函数参数
        # 使用更激进的参数：稳定区密度极低，相变区密度极高
        base_density = min_density_ratio
        gradient_weight = 1.0 - base_density

        # 生成自适应采样点
        adaptive_alphas = self.generate_adaptive_alphas(
            n_points,
            base_density=base_density,
            gradient_weight=gradient_weight,
            power=power
        )

        # 计算密度函数用于分析
        density = self.compute_density_function(base_density, gradient_weight, power)

        # 找相变区
        phase_zone = self._find_phase_zone()

        # 估计质量提升
        # 基于相变区采样密度的提升
        phase_mask = (adaptive_alphas >= phase_zone[0]) & (adaptive_alphas <= phase_zone[1])
        uniform_density_in_phase = len(self.alphas) / (self.alphas[-1] - self.alphas[0])
        adaptive_density_in_phase = phase_mask.sum() / (phase_zone[1] - phase_zone[0]) if phase_zone[1] > phase_zone[0] else 1

        quality_improvement = adaptive_density_in_phase / uniform_density_in_phase

        return AdaptiveSamplingResult(
            original_alphas=self.alphas,
            adaptive_alphas=adaptive_alphas,
            density_function=density,
            gradient_profile=self.gradient,
            phase_zone=phase_zone,
            quality_improvement=quality_improvement,
        )

    def analyze_distribution(self, adaptive_alphas: np.ndarray) -> Dict:
        """
        分析采样分布质量

        返回各区域的采样密度统计
        """
        alpha_range = self.alphas[-1] - self.alphas[0]
        phase_zone = self._find_phase_zone()

        # 分区统计
        in_phase = (adaptive_alphas >= phase_zone[0]) & (adaptive_alphas <= phase_zone[1])
        below_phase = adaptive_alphas < phase_zone[0]
        above_phase = adaptive_alphas > phase_zone[1]

        phase_span = phase_zone[1] - phase_zone[0]
        below_span = phase_zone[0] - self.alphas[0]
        above_span = self.alphas[-1] - phase_zone[1]

        def safe_density(count, span):
            return count / span if span > 0 else 0

        # 均匀采样的对照密度
        uniform_density = len(self.alphas) / alpha_range

        return {
            'total_points': len(adaptive_alphas),
            'phase_zone': phase_zone,
            'phase_zone_span': phase_span,
            'regions': {
                'below_phase': {
                    'count': int(below_phase.sum()),
                    'span': below_span,
                    'density': safe_density(below_phase.sum(), below_span),
                    'relative_density': safe_density(below_phase.sum(), below_span) / uniform_density if uniform_density > 0 else 0,
                },
                'phase': {
                    'count': int(in_phase.sum()),
                    'span': phase_span,
                    'density': safe_density(in_phase.sum(), phase_span),
                    'relative_density': safe_density(in_phase.sum(), phase_span) / uniform_density if uniform_density > 0 else 0,
                },
                'above_phase': {
                    'count': int(above_phase.sum()),
                    'span': above_span,
                    'density': safe_density(above_phase.sum(), above_span),
                    'relative_density': safe_density(above_phase.sum(), above_span) / uniform_density if uniform_density > 0 else 0,
                },
            },
            'uniform_density': uniform_density,
        }


def smart_redistribute(alphas: np.ndarray,
                       Q_Y: np.ndarray,
                       n_points: Optional[int] = None,
                       smooth_sigma: float = 1.0,
                       power: float = 3.0) -> Tuple[np.ndarray, Dict]:
    """
    便捷函数: 智能重新分布采样点

    在保持总点数不变的情况下，根据梯度重新分布
    相变区获得更高密度，稳定区获得更低密度

    参数:
        alphas: 粗扫描的 alpha 值
        Q_Y: 对应的 Q_Y 值
        n_points: 目标点数 (默认与原始相同)
        smooth_sigma: 梯度平滑参数
        power: 梯度幂次 (越大越集中，默认3.0)

    返回:
        (adaptive_alphas, info_dict)
    """
    sampler = GradientAdaptiveSampler(alphas, Q_Y, smooth_sigma=smooth_sigma)
    result = sampler.redistribute(n_points=n_points, power=power)

    info = sampler.analyze_distribution(result.adaptive_alphas)
    info['quality_improvement'] = result.quality_improvement
    info['phase_zone'] = result.phase_zone

    return result.adaptive_alphas, info


def compare_sampling_quality(uniform_alphas: np.ndarray,
                             adaptive_alphas: np.ndarray,
                             true_Q_Y_func: callable,
                             metric: str = 'mse') -> Dict:
    """
    比较均匀采样和自适应采样的质量

    参数:
        uniform_alphas: 均匀采样点
        adaptive_alphas: 自适应采样点
        true_Q_Y_func: 真实 Q_Y 函数 (用于评估)
        metric: 评估指标 ('mse', 'max_error', 'integral')

    返回:
        比较结果字典
    """
    # 在细密网格上评估
    fine_alphas = np.linspace(
        min(uniform_alphas.min(), adaptive_alphas.min()),
        max(uniform_alphas.max(), adaptive_alphas.max()),
        1000
    )
    true_values = np.array([true_Q_Y_func(a) for a in fine_alphas])

    # 用各自的采样点插值
    uniform_Q_Y = np.array([true_Q_Y_func(a) for a in uniform_alphas])
    adaptive_Q_Y = np.array([true_Q_Y_func(a) for a in adaptive_alphas])

    uniform_interp = interpolate.interp1d(uniform_alphas, uniform_Q_Y, kind='linear',
                                          bounds_error=False, fill_value='extrapolate')
    adaptive_interp = interpolate.interp1d(adaptive_alphas, adaptive_Q_Y, kind='linear',
                                           bounds_error=False, fill_value='extrapolate')

    uniform_pred = uniform_interp(fine_alphas)
    adaptive_pred = adaptive_interp(fine_alphas)

    # 计算误差
    uniform_error = np.abs(uniform_pred - true_values)
    adaptive_error = np.abs(adaptive_pred - true_values)

    return {
        'uniform': {
            'mse': float(np.mean(uniform_error ** 2)),
            'max_error': float(np.max(uniform_error)),
            'mean_error': float(np.mean(uniform_error)),
        },
        'adaptive': {
            'mse': float(np.mean(adaptive_error ** 2)),
            'max_error': float(np.max(adaptive_error)),
            'mean_error': float(np.mean(adaptive_error)),
        },
        'improvement': {
            'mse_ratio': float(np.mean(uniform_error ** 2) / (np.mean(adaptive_error ** 2) + 1e-10)),
            'max_error_ratio': float(np.max(uniform_error) / (np.max(adaptive_error) + 1e-10)),
        },
    }
