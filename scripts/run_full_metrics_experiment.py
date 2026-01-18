#!/usr/bin/env python3
"""
完整指标实验脚本 - 包含所有可计算指标
S=50 样本, max_steps=2000
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/sucia/Sparse-Matrix')

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from MF.core.config import (
    Config, MatrixConfig, AlphaConfig, TrainingConfig, 
    AlgorithmConfig, SpreadingConfig, ExecutionConfig
)
from MF.runner import run_experiment


# 所有可计算指标
ALL_METRICS = [
    # Y 相关
    'Q_Y', 'Q_Y_observed', 'Q_Y_unobserved',
    'physical_overlap_Y', 'physical_overlap_Y_observed', 'physical_overlap_Y_unobserved',
    # W 相关
    'Q_W', 'Q_W_prime', 'physical_overlap_W',
    # X 相关
    'Q_X', 'Q_X_prime', 'physical_overlap_X',
    # 误差
    'Gen_Error',
]


def create_config(N: int, M: int, S: int = 50, max_steps: int = 2000) -> Config:
    """创建包含所有指标的配置"""
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


def extract_all_metrics(result: dict) -> dict:
    """提取所有指标数据"""
    results = result.get('results', result)
    data = {}
    
    for alpha, metrics in sorted(results.items(), key=lambda x: float(x[0])):
        alpha_f = float(alpha)
        for key, value in metrics.items():
            if key not in data:
                data[key] = {'alpha': [], 'value': []}
            data[key]['alpha'].append(alpha_f)
            data[key]['value'].append(value)
    
    # Convert to numpy
    for key in data:
        data[key]['alpha'] = np.array(data[key]['alpha'])
        data[key]['value'] = np.array(data[key]['value'])
    
    return data


def plot_all_metrics(results_dict: dict, output_dir: Path):
    """生成所有对比图"""
    configs = list(results_dict.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(configs)]
    
    # 1. Q_Y 全系列对比
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    for metric, ax, title in [
        ('Q_Y_mean', axes1[0], 'Q_Y (Cosine)'),
        ('Q_Y_observed_mean', axes1[1], 'Q_Y Observed'),
        ('Q_Y_unobserved_mean', axes1[2], 'Q_Y Unobserved'),
    ]:
        for i, key in enumerate(configs):
            data = extract_all_metrics(results_dict[key])
            if metric in data:
                ax.plot(data[metric]['alpha'], data[metric]['value'], 
                       color=colors[i], label=key, linewidth=2)
        ax.set_xlabel('α')
        ax.set_ylabel('Q_Y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.05)
    fig1.tight_layout()
    fig1.savefig(output_dir / 'q_y_all_comparison.png', dpi=150)
    print(f"保存: {output_dir / 'q_y_all_comparison.png'}")
    
    # 2. Physical Overlap Y 对比
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for metric, ax, title in [
        ('physical_overlap_Y_mean', axes2[0], 'Physical Y (全局)'),
        ('physical_overlap_Y_observed_mean', axes2[1], 'Physical Y Observed'),
        ('physical_overlap_Y_unobserved_mean', axes2[2], 'Physical Y Unobserved'),
    ]:
        for i, key in enumerate(configs):
            data = extract_all_metrics(results_dict[key])
            if metric in data:
                ax.plot(data[metric]['alpha'], data[metric]['value'], 
                       color=colors[i], label=key, linewidth=2)
        ax.set_xlabel('α')
        ax.set_ylabel('Physical Overlap')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.05)
    fig2.tight_layout()
    fig2.savefig(output_dir / 'physical_y_comparison.png', dpi=150)
    print(f"保存: {output_dir / 'physical_y_comparison.png'}")
    
    # 3. W 和 X 指标对比
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
    w_metrics = [('Q_W_mean', 'Q_W'), ('Q_W_prime_mean', 'Q_W\''), ('physical_overlap_W_mean', 'Physical W')]
    x_metrics = [('Q_X_mean', 'Q_X'), ('Q_X_prime_mean', 'Q_X\''), ('physical_overlap_X_mean', 'Physical X')]
    
    for row, metrics in enumerate([w_metrics, x_metrics]):
        for col, (metric, title) in enumerate(metrics):
            ax = axes3[row, col]
            for i, key in enumerate(configs):
                data = extract_all_metrics(results_dict[key])
                if metric in data:
                    ax.plot(data[metric]['alpha'], data[metric]['value'], 
                           color=colors[i], label=key, linewidth=2)
            ax.set_xlabel('α')
            ax.set_ylabel('Overlap')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 1.05)
    fig3.tight_layout()
    fig3.savefig(output_dir / 'w_x_comparison.png', dpi=150)
    print(f"保存: {output_dir / 'w_x_comparison.png'}")
    
    # 4. Replica 指标对比 (如果有的话)
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 8))
    replica_metrics = [
        ('Q_W_replica_mean', axes4[0, 0], 'Q_W Replica'),
        ('Q_X_replica_mean', axes4[0, 1], 'Q_X Replica'),
        ('physical_W_replica_mean', axes4[1, 0], 'Physical W Replica'),
        ('physical_X_replica_mean', axes4[1, 1], 'Physical X Replica'),
    ]
    for metric, ax, title in replica_metrics:
        for i, key in enumerate(configs):
            data = extract_all_metrics(results_dict[key])
            if metric in data:
                ax.plot(data[metric]['alpha'], data[metric]['value'], 
                       color=colors[i], label=key, linewidth=2)
        ax.set_xlabel('α')
        ax.set_ylabel('Replica Overlap')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 4)
    fig4.tight_layout()
    fig4.savefig(output_dir / 'replica_comparison.png', dpi=150)
    print(f"保存: {output_dir / 'replica_comparison.png'}")
    
    plt.close('all')


def verify_phase_transition(data: dict, key: str) -> bool:
    """验证相变点是否合理 (Q_Y 应该在 alpha~3.0-3.5 时从低跳到高)"""
    if 'Q_Y_mean' not in data:
        return True  # 无法验证
    
    alphas = data['Q_Y_mean']['alpha']
    values = data['Q_Y_mean']['value']
    
    # 检查 alpha < 3 时 Q_Y < 0.5，alpha > 3.5 时 Q_Y > 0.9
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
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path(f"/home/sucia/Sparse-Matrix/smf/results/full_metrics_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"实验输出目录: {output_dir}")
    
    # 配置: N=200,M=50 和 N=400,M=100, S=50, max_steps=2000
    configs = [(200, 50), (400, 100)]
    results_dict = {}
    
    for N, M in configs:
        print(f"\n{'='*60}")
        print(f"运行实验: N={N}, M={M}, S=50, max_steps=2000")
        print('='*60)
        
        config = create_config(N, M, S=50, max_steps=2000)
        result = run_experiment(config, save=True)
        
        key = f"{N}x{M}"
        results_dict[key] = result
        
        # 验证相变特征
        data = extract_all_metrics(result)
        verify_phase_transition(data, key)
        
        # 保存结果 (6位小数已在计算时处理)
        result_file = output_dir / f"result_{key}.json"
        with open(result_file, 'w') as f:
            json.dump({
                'config': f"N={N}, M={M}, S=50, max_steps=2000",
                'results': result.get('results', {}), 
                'total_time': result.get('total_time', 0)
            }, f, indent=2)
        print(f"保存结果: {result_file}")
        
        # 显示样本指标
        sample_alpha = "3.5"
        if sample_alpha in result.get('results', {}):
            sample = result['results'][sample_alpha]
            print(f"\n   样本指标 (α={sample_alpha}):")
            for k, v in sorted(sample.items())[:10]:
                print(f"     {k}: {v}")
    
    # 生成对比图
    print(f"\n{'='*60}")
    print("生成对比图...")
    print('='*60)
    plot_all_metrics(results_dict, output_dir)
    
    print(f"\n✅ 完成！结果在: {output_dir}")
    print(f"   - 包含 {len(ALL_METRICS)} 个核心指标")
    print(f"   - 包含 8 个 replica 指标")


if __name__ == '__main__':
    main()
