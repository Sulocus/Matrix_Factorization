#!/usr/bin/env python3
"""
400x100 单独实验 - 减少S避免OOM
S=30, max_steps=2000
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/sucia/Sparse-Matrix')

import numpy as np
import matplotlib.pyplot as plt

from MF.core.config import (
    Config, MatrixConfig, AlphaConfig, TrainingConfig, 
    AlgorithmConfig, SpreadingConfig, ExecutionConfig
)
from MF.runner import run_experiment

ALL_METRICS = [
    'Q_Y', 'Q_Y_observed', 'Q_Y_unobserved',
    'physical_overlap_Y', 'physical_overlap_Y_observed', 'physical_overlap_Y_unobserved',
    'Q_W', 'Q_W_prime', 'physical_overlap_W',
    'Q_X', 'Q_X_prime', 'physical_overlap_X',
    'Gen_Error',
]


def create_config(N: int, M: int, S: int, max_steps: int) -> Config:
    return Config(
        algorithm_key="bigamp_spreading_parallel_unit",
        teacher_key="orthogonal",
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.1),
        training=TrainingConfig(max_steps=max_steps, samples_per_alpha=S, seed=42),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        spreading=SpreadingConfig(f_distribution="rademacher", seed=12345),
        execution=ExecutionConfig(
            metrics_to_compute=ALL_METRICS,
            include_summary_plot=True,
            include_qy_plot=True,
        ),
    )


def verify_phase_transition(results: dict, key: str) -> bool:
    """验证相变点"""
    alphas = []
    q_y_values = []
    for alpha, data in sorted(results.items(), key=lambda x: float(x[0])):
        alphas.append(float(alpha))
        q_y_values.append(data.get('Q_Y_mean', 0))
    
    alphas = np.array(alphas)
    values = np.array(q_y_values)
    
    low_region = values[alphas < 3.0]
    high_region = values[alphas > 3.5]
    
    if len(low_region) > 0 and len(high_region) > 0:
        low_mean = np.mean(low_region)
        high_mean = np.mean(high_region)
        print(f"   [验证] {key}: α<3.0 平均 Q_Y={low_mean:.3f}, α>3.5 平均 Q_Y={high_mean:.3f}")
        
        if low_mean > 0.5:
            print(f"   ⚠️ 警告: 低 alpha 区域 Q_Y 偏高！")
            return False
        if high_mean < 0.8:
            print(f"   ⚠️ 警告: 高 alpha 区域 Q_Y 偏低！")
            return False
        print(f"   ✓ 相变特征正常")
    return True


def main():
    # 使用已有的输出目录
    output_dir = Path("/home/sucia/Sparse-Matrix/smf/results/full_metrics_1211_2032")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 400x100 配置，S=10 避免 OOM (显存估算严重低估，实际使用约1.5倍)
    N, M, S = 400, 100, 10
    max_steps = 2000
    
    print(f"\n{'='*60}")
    print(f"运行实验: N={N}, M={M}, S={S}, max_steps={max_steps}")
    print(f"(减少 S 从 50 到 30 以避免 OOM)")
    print('='*60)
    
    config = create_config(N, M, S, max_steps)
    result = run_experiment(config, save=True)
    
    key = f"{N}x{M}"
    
    # 验证
    verify_phase_transition(result.get('results', {}), key)
    
    # 保存
    result_file = output_dir / f"result_{key}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'config': f"N={N}, M={M}, S={S}, max_steps={max_steps}",
            'results': result.get('results', {}), 
            'total_time': result.get('total_time', 0)
        }, f, indent=2)
    print(f"\n保存结果: {result_file}")
    
    # 显示样本指标
    sample_alpha = "3.5"
    if sample_alpha in result.get('results', {}):
        sample = result['results'][sample_alpha]
        print(f"\n样本指标 (α={sample_alpha}):")
        for k, v in sorted(sample.items())[:10]:
            print(f"  {k}: {v}")
    
    print(f"\n✅ 完成！")


if __name__ == '__main__':
    main()
