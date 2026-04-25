#!/usr/bin/env python3
"""
等比例放大实验脚本
运行三组配置 (200x50, 400x100, 600x150)，N:M = 4:1
使用 Spreading 模型，Rademacher F 分布

生成三张对比图：
1. Q_Y 全局对比
2. Q_Y Unobserved 对比
3. 600x150 的全局 vs Unobserved
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/home/sucia/Matrix_Factorization/src')

import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from matrix_factorization.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, AlgorithmConfig, SpreadingConfig
from matrix_factorization.runner import run_experiment


def create_config(N: int, M: int, algorithm_key: str = "bigamp_spreading_parallel_unit") -> Config:
    """创建实验配置"""
    return Config(
        algorithm_key=algorithm_key,
        teacher_key="orthogonal",  # 正交教师
        matrix=MatrixConfig(N1=N, N2=N, M=M),
        alpha=AlphaConfig(start=0.0, stop=4.0, step=0.1),
        training=TrainingConfig(
            max_steps=5000,
            samples_per_alpha=2,  # S=2
            seed=42,
        ),
        algorithm=AlgorithmConfig(
            damping=0.5,
            noise_var=1e-10,
        ),
        spreading=SpreadingConfig(
            f_distribution="rademacher",
            seed=12345,
        ),
    )


def run_single_experiment(N: int, M: int, algorithm_key: str) -> dict:
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"开始实验: N={N}, M={M}, Algorithm={algorithm_key}")
    print(f"{'='*60}")
    
    config = create_config(N, M, algorithm_key)
    
    start_time = time.time()
    result = run_experiment(config, save=True)
    elapsed = time.time() - start_time
    
    print(f"\n实验完成: N={N}, M={M}, 耗时: {elapsed:.1f}s")
    
    return result


def extract_metrics(result: dict) -> tuple:
    """从结果中提取 Q_Y 和 Q_Y_unobserved"""
    alpha_values = []
    q_y_values = []
    q_y_unobs_values = []
    
    for alpha, data in sorted(result['results'].items()):
        alpha_values.append(float(alpha))
        q_y_values.append(data.get('Q_Y_mean', data.get('Q_Y', 0)))
        # Q_Y_unobserved 可能在不同的键下
        q_y_unobs = data.get('Q_Y_unobserved_mean', data.get('Q_Y_unobserved', 0))
        q_y_unobs_values.append(q_y_unobs)
    
    return np.array(alpha_values), np.array(q_y_values), np.array(q_y_unobs_values)


def plot_comparison(results_dict: dict, output_dir: Path):
    """生成对比图"""
    
    # 提取所有结果
    configs = [(200, 50), (400, 100), (600, 150)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = [f'N={N}, M={M}' for N, M in configs]
    
    # ===== 图1: Q_Y 全局对比 =====
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for i, (N, M) in enumerate(configs):
        key = f"{N}x{M}"
        if key in results_dict:
            alphas, q_y, _ = extract_metrics(results_dict[key])
            ax1.plot(alphas, q_y, color=colors[i], label=labels[i], linewidth=2)
    
    ax1.set_xlabel('α (observation ratio)', fontsize=12)
    ax1.set_ylabel('Q_Y (observed)', fontsize=12)
    ax1.set_title('Q_Y Phase Transition - Scaling Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 1.05)
    
    fig1.tight_layout()
    fig1.savefig(output_dir / 'q_y_comparison.png', dpi=150)
    print(f"保存: {output_dir / 'q_y_comparison.png'}")
    
    # ===== 图2: Q_Y Unobserved 对比 =====
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for i, (N, M) in enumerate(configs):
        key = f"{N}x{M}"
        if key in results_dict:
            alphas, _, q_y_unobs = extract_metrics(results_dict[key])
            ax2.plot(alphas, q_y_unobs, color=colors[i], label=labels[i], linewidth=2)
    
    ax2.set_xlabel('α (observation ratio)', fontsize=12)
    ax2.set_ylabel('Q_Y (unobserved)', fontsize=12)
    ax2.set_title('Q_Y Unobserved Phase Transition - Scaling Comparison', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 1.05)
    
    fig2.tight_layout()
    fig2.savefig(output_dir / 'q_y_unobserved_comparison.png', dpi=150)
    print(f"保存: {output_dir / 'q_y_unobserved_comparison.png'}")
    
    # ===== 图3: 600x150 全局 vs Unobserved =====
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    key = "600x150"
    if key in results_dict:
        alphas, q_y, q_y_unobs = extract_metrics(results_dict[key])
        ax3.plot(alphas, q_y, color='#1f77b4', label='Q_Y (observed)', linewidth=2)
        ax3.plot(alphas, q_y_unobs, color='#ff7f0e', label='Q_Y (unobserved)', linewidth=2, linestyle='--')
    
    ax3.set_xlabel('α (observation ratio)', fontsize=12)
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
    # 输出目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path(f"/home/sucia/Matrix_Factorization/src/smf/results/scaling_experiment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"实验输出目录: {output_dir}")
    
    # 实验配置
    configs = [
        (200, 50),
        (400, 100),
        (600, 150),
    ]
    
    # 使用 unit scaling 版本（与用户测试一致）
    algorithm_key = "bigamp_spreading_parallel_unit"
    
    results_dict = {}
    
    for N, M in configs:
        key = f"{N}x{M}"
        try:
            result = run_single_experiment(N, M, algorithm_key)
            results_dict[key] = result
            
            # 保存单个结果
            result_path = output_dir / f"result_{key}.json"
            # 转换为可序列化格式
            serializable = {
                'config': str(result.get('config', '')),
                'results': result.get('results', {}),
                'total_time': result.get('total_time', 0),
            }
            with open(result_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"保存结果: {result_path}")
            
        except Exception as e:
            print(f"实验失败 {key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比图
    if results_dict:
        print(f"\n{'='*60}")
        print("生成对比图...")
        print(f"{'='*60}")
        plot_comparison(results_dict, output_dir)
        print(f"\n所有实验完成！结果保存在: {output_dir}")
    else:
        print("没有成功的实验结果，无法生成图表")


if __name__ == '__main__':
    main()
