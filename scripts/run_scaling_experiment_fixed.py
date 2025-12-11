#!/usr/bin/env python3
"""
修复版实验脚本 - 包含 Q_Y_unobserved 指标
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

from smf.core.config import (
    Config, MatrixConfig, AlphaConfig, TrainingConfig, 
    AlgorithmConfig, SpreadingConfig, ExecutionConfig
)
from smf.runner import run_experiment


def create_config(N: int, M: int) -> Config:
    """创建包含 Q_Y_unobserved 指标的配置"""
    return Config(
        algorithm_key="bigamp_spreading_parallel_unit",
        teacher_key="orthogonal",
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.1),
        training=TrainingConfig(max_steps=5000, samples_per_alpha=2, seed=42),
        algorithm=AlgorithmConfig(damping=0.5, noise_var=1e-10),
        spreading=SpreadingConfig(f_distribution="rademacher", seed=12345),
        # 关键修复: 添加 Q_Y_unobserved 到 metrics_to_compute
        execution=ExecutionConfig(
            metrics_to_compute=['Q_Y', 'Q_W', 'Q_X', 'Q_W_prime', 'Q_X_prime', 'Gen_Error', 'Q_Y_unobserved'],
            include_summary_plot=True,
            include_qy_plot=True,
        ),
    )


def extract_metrics(result: dict) -> tuple:
    alpha_values, q_y_values, q_y_unobs_values = [], [], []
    results = result.get('results', result)  # 兼容两种格式
    for alpha, data in sorted(results.items(), key=lambda x: float(x[0])):
        alpha_values.append(float(alpha))
        q_y_values.append(data.get('Q_Y_mean', data.get('Q_Y', 0)))
        q_y_unobs_values.append(data.get('Q_Y_unobserved_mean', data.get('Q_Y_unobserved', 0)))
    return np.array(alpha_values), np.array(q_y_values), np.array(q_y_unobs_values)


def plot_comparison(results_dict: dict, output_dir: Path):
    configs = [(200, 50), (400, 100), (600, 150)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = [f'N={N}, M={M}' for N, M in configs]
    
    # 图1: Q_Y 对比
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i, (N, M) in enumerate(configs):
        key = f"{N}x{M}"
        if key in results_dict:
            alphas, q_y, _ = extract_metrics(results_dict[key])
            ax1.plot(alphas, q_y, color=colors[i], label=labels[i], linewidth=2)
    ax1.set_xlabel('α', fontsize=12)
    ax1.set_ylabel('Q_Y (observed)', fontsize=12)
    ax1.set_title('Q_Y Phase Transition - Scaling Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 1.05)
    fig1.tight_layout()
    fig1.savefig(output_dir / 'q_y_comparison.png', dpi=150)
    print(f"保存: {output_dir / 'q_y_comparison.png'}")
    
    # 图2: Q_Y Unobserved 对比
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    has_unobs_data = False
    for i, (N, M) in enumerate(configs):
        key = f"{N}x{M}"
        if key in results_dict:
            alphas, _, q_y_unobs = extract_metrics(results_dict[key])
            if np.any(q_y_unobs != 0):  # 检查是否有有效数据
                has_unobs_data = True
            ax2.plot(alphas, q_y_unobs, color=colors[i], label=labels[i], linewidth=2)
    
    if not has_unobs_data:
        ax2.text(0.5, 0.5, 'No Q_Y_unobserved data available.\nRe-run experiment with Q_Y_unobserved metric.',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='red')
    
    ax2.set_xlabel('α', fontsize=12)
    ax2.set_ylabel('Q_Y (unobserved)', fontsize=12)
    ax2.set_title('Q_Y Unobserved Phase Transition', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 1.05)
    fig2.tight_layout()
    fig2.savefig(output_dir / 'q_y_unobserved_comparison.png', dpi=150)
    print(f"保存: {output_dir / 'q_y_unobserved_comparison.png'}")
    
    # 图3: 600x150 全局 vs Unobserved
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    key = "600x150"
    if key in results_dict:
        alphas, q_y, q_y_unobs = extract_metrics(results_dict[key])
        ax3.plot(alphas, q_y, color='#1f77b4', label='Q_Y (observed)', linewidth=2)
        ax3.plot(alphas, q_y_unobs, color='#ff7f0e', label='Q_Y (unobserved)', linewidth=2, linestyle='--')
    ax3.set_xlabel('α', fontsize=12)
    ax3.set_ylabel('Q_Y', fontsize=12)
    ax3.set_title('N=600, M=150: Observed vs Unobserved', fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 1.05)
    fig3.tight_layout()
    fig3.savefig(output_dir / 'q_y_600x150_comparison.png', dpi=150)
    print(f"保存: {output_dir / 'q_y_600x150_comparison.png'}")
    plt.close('all')


def main():
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path(f"/home/sucia/Sparse-Matrix/smf/results/scaling_experiment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"实验输出目录: {output_dir}")
    
    configs = [(200, 50), (400, 100), (600, 150)]
    results_dict = {}
    
    for N, M in configs:
        print(f"\n{'='*60}")
        print(f"运行实验: N={N}, M={M}")
        print('='*60)
        
        config = create_config(N, M)
        result = run_experiment(config, save=True)
        
        key = f"{N}x{M}"
        results_dict[key] = result
        
        # 保存结果
        with open(output_dir / f"result_{key}.json", 'w') as f:
            json.dump({
                'results': result.get('results', {}), 
                'total_time': result.get('total_time', 0)
            }, f, indent=2)
        print(f"保存结果: {output_dir / f'result_{key}.json'}")
    
    # 生成对比图
    print(f"\n{'='*60}")
    print("生成对比图...")
    print('='*60)
    plot_comparison(results_dict, output_dir)
    print(f"\n完成！结果在: {output_dir}")


if __name__ == '__main__':
    main()
