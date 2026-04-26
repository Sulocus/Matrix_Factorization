"""
Plot Registry - 绘图指标映射表

指标格式: 类别.指标[:R]
  - :R = replica (学生-学生)
  - 不加 :R = 教师-学生

类别:
  A: Projection metrics (Q_Y, Q_W, Q_X)
  B: Gram-root diagnostics (Q_W_GRAM_ROOT, Q_X_GRAM_ROOT)
  C: Observed Split (Q_Y_observed, Q_Y_unobserved)

用法:
    plots:
      - curves: [A.y, A.w, A.y:R]
      - curves: [C.o, C.u]
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# 指标映射表: (类别, 指标代码) -> 内部 metric 名称
METRIC_MAP: Dict[Tuple[str, str], str] = {
    # A: Projection metrics
    ('A', 'y'): 'Q_Y',
    ('A', 'w'): 'Q_W',
    ('A', 'x'): 'Q_X',
    
    # B: Gram-root diagnostics
    ('B', 'w'): 'Q_W_GRAM_ROOT',
    ('B', 'x'): 'Q_X_GRAM_ROOT',
    
    # C: Observed Split
    ('C', 'o'): 'Q_Y_observed',
    ('C', 'u'): 'Q_Y_unobserved',
    
    # N: Tensor latent factor projection
    ('N', 'n'): 'Q_N',
}

# 支持 replica 的指标
REPLICA_SUPPORTED = {
    'Q_Y', 'Q_W', 'Q_X',
    'Q_W_GRAM_ROOT', 'Q_X_GRAM_ROOT',
    'Q_Y_observed', 'Q_Y_unobserved',
}


@dataclass
class CurveSpec:
    """曲线规格"""
    metric_name: str      # 内部 metric 名称
    is_replica: bool      # 是否是 replica (学生-学生)
    display_name: str     # 显示名称
    code: str             # 原始代码 (如 A.y:R)


def parse_curve_code(code: str) -> CurveSpec:
    """
    解析曲线代码
    
    Args:
        code: 曲线代码，如 'A.y', 'C.o:R'
        
    Returns:
        CurveSpec 对象
        
    Raises:
        ValueError: 无效的代码格式
    """
    # 检查 :R 后缀
    is_replica = code.endswith(':R')
    clean_code = code[:-2] if is_replica else code
    
    # 解析 类别.指标
    if '.' not in clean_code:
        raise ValueError(f"Invalid curve code format: {code}. Expected 'X.y' or 'X.y:R'")
    
    parts = clean_code.split('.')
    if len(parts) != 2:
        raise ValueError(f"Invalid curve code format: {code}. Expected 'X.y' or 'X.y:R'")
    
    category, metric = parts[0].upper(), parts[1].lower()
    
    # 查找映射
    key = (category, metric)
    if key not in METRIC_MAP:
        valid_codes = [f"{k[0]}.{k[1]}" for k in METRIC_MAP.keys()]
        raise ValueError(f"Unknown metric code: {code}. Valid codes: {valid_codes}")
    
    metric_name = METRIC_MAP[key]
    
    # 检查 replica 支持
    if is_replica and metric_name not in REPLICA_SUPPORTED:
        raise ValueError(f"Metric {metric_name} does not support replica mode (:R)")
    
    # 构建显示名称
    display_name = metric_name
    if is_replica:
        display_name += " (Replica)"
    
    return CurveSpec(
        metric_name=metric_name,
        is_replica=is_replica,
        display_name=display_name,
        code=code,
    )


def parse_plot_config(plot_config: Dict) -> List[CurveSpec]:
    """
    解析单个 plot 配置
    
    Args:
        plot_config: {'curves': ['A.y', 'A.w:R', ...]}
        
    Returns:
        CurveSpec 列表
    """
    curves = plot_config.get('curves', [])
    return [parse_curve_code(code) for code in curves]


def get_required_metrics(plot_configs: List[Dict]) -> set:
    """
    从绘图配置中提取所有需要的 metrics
    
    Args:
        plot_configs: 绘图配置列表
        
    Returns:
        需要计算的 metric 名称集合 (包括 replica 后缀)
    """
    required = set()
    
    for config in plot_configs:
        for curve in parse_plot_config(config):
            if curve.is_replica:
                # Replica 使用 _replica 后缀
                required.add(curve.metric_name + '_replica')
            else:
                required.add(curve.metric_name)
    
    return required


def print_metric_table():
    """打印完整的指标表"""
    print("=" * 60)
    print("绘图指标格式: 类别.指标[:R]")
    print("  :R = replica (学生-学生)")
    print("=" * 60)
    
    current_category = None
    for (cat, metric), name in sorted(METRIC_MAP.items()):
        if cat != current_category:
            current_category = cat
            cat_names = {'A': 'Cosine', 'B': 'Norm', 'C': 'Split', 'D': 'Phys', 'E': 'Other'}
            print(f"\n{cat} ({cat_names.get(cat, cat)}):")
        
        replica = '✓' if name in REPLICA_SUPPORTED else '✗'
        print(f"  {cat}.{metric}  →  {name:<20}  [:R {replica}]")
    
    print("=" * 60)


if __name__ == "__main__":
    print_metric_table()
    
    # 测试解析
    print("\n测试解析:")
    test_codes = ['A.y', 'A.w:R', 'C.o', 'C.u:R', 'E.e']
    for code in test_codes:
        try:
            spec = parse_curve_code(code)
            print(f"  {code} → {spec.display_name}")
        except ValueError as e:
            print(f"  {code} → ERROR: {e}")
