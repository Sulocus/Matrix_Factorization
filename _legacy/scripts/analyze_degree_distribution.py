"""
分析随机图生成的节点连接度分布

复制自 Main.py 的随机图生成方法，统计左侧节点的连接度分布
重点：均值和方差
"""

import numpy as np
import torch
from collections import Counter

# ============================================================
# 参数配置
# ============================================================
N1 = 200
N2 = 200
M = 50
ALPHA = 2.0  # 观测密度

NUM_SAMPLES = 50  # 重复采样次数
SEED = 42

# ============================================================
# 随机图生成（复制自 Main.py）
# ============================================================
def sample_pairs_random_gpu(N1, N2, C, device, seed=None):
    """
    Pure random mask generation (entirely on GPU)
    """
    if seed is not None:
        torch.manual_seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"Requested edge count C={C} exceeds matrix total size {N1}×{N2}={total}")

    if C == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    # 在 GPU 上随机打乱所有位置索引
    perm = torch.randperm(total, device=device)
    selected = perm[:C]

    # 将 1D 索引恢复为 (i, j) 坐标
    i_idx = selected // N2
    j_idx = selected % N2

    return i_idx, j_idx, C


def analyze_degree_distribution():
    """分析左侧节点的连接度分布 - 重点：均值和方差"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 计算边数
    C = int(ALPHA * M * N1)
    expected_degree = C / N1  # 理论期望度数 = α × M

    print(f"\n{'='*60}")
    print("参数设置")
    print(f"{'='*60}")
    print(f"N1 = {N1}, N2 = {N2}, M = {M}")
    print(f"Alpha = {ALPHA}")
    print(f"边数 C = α × M × N1 = {C}")
    print(f"理论期望度数 E[d] = C / N1 = α × M = {expected_degree:.2f}")

    # 理论方差（超几何分布）
    # 每个左节点有 N2 个可能位置，总共 N1*N2 个位置，选 C 个
    # 单个节点的度数服从超几何分布 Hypergeometric(N1*N2, N2, C)
    N_total = N1 * N2
    K = N2  # 每个左节点对应的位置数
    n = C   # 总采样数
    # E[X] = n * K / N = C * N2 / (N1*N2) = C/N1 ✓
    # Var[X] = n * K/N * (1 - K/N) * (N-n)/(N-1)
    theoretical_var = n * (K/N_total) * (1 - K/N_total) * (N_total - n) / (N_total - 1)
    theoretical_std = np.sqrt(theoretical_var)

    print(f"\n理论预测（超几何分布）:")
    print(f"  E[d] = {expected_degree:.4f}")
    print(f"  Var[d] = {theoretical_var:.4f}")
    print(f"  Std[d] = {theoretical_std:.4f}")

    # 多次采样统计
    print(f"\n{'='*60}")
    print(f"实测统计 ({NUM_SAMPLES} 次采样)")
    print(f"{'='*60}")

    all_means = []
    all_vars = []

    for sample_idx in range(NUM_SAMPLES):
        seed = SEED + sample_idx
        i_idx, _, _ = sample_pairs_random_gpu(N1, N2, C, device, seed=seed)

        # 统计每个左侧节点的度数
        i_idx_cpu = i_idx.cpu().numpy()
        degree_counts = Counter(i_idx_cpu)
        degrees = np.array([degree_counts.get(i, 0) for i in range(N1)])

        all_means.append(np.mean(degrees))
        all_vars.append(np.var(degrees))

    print(f"\n单次采样内部统计（每次 {N1} 个节点的度数）:")
    print(f"  平均度数的均值: {np.mean(all_means):.4f} (理论: {expected_degree:.4f})")
    print(f"  平均度数的标准差: {np.std(all_means):.4f}")
    print(f"  方差的均值: {np.mean(all_vars):.4f} (理论: {theoretical_var:.4f})")
    print(f"  方差的标准差: {np.std(all_vars):.4f}")

    # 单次采样详细展示
    print(f"\n{'='*60}")
    print("单次采样详细结果 (seed=42)")
    print(f"{'='*60}")

    i_idx, _, _ = sample_pairs_random_gpu(N1, N2, C, device, seed=SEED)
    i_idx_cpu = i_idx.cpu().numpy()
    degree_counts = Counter(i_idx_cpu)
    degrees = np.array([degree_counts.get(i, 0) for i in range(N1)])

    print(f"  均值: {np.mean(degrees):.4f}")
    print(f"  方差: {np.var(degrees):.4f}")
    print(f"  标准差: {np.std(degrees):.4f}")
    print(f"  最小度数: {np.min(degrees)}")
    print(f"  最大度数: {np.max(degrees)}")
    print(f"  度数范围: [{np.min(degrees)}, {np.max(degrees)}]")

    # 结论
    print(f"\n{'='*60}")
    print("结论")
    print(f"{'='*60}")
    print(f"1. 均值: E[d] = α × M = {expected_degree:.1f} (精确)")
    print(f"2. 方差: Var[d] ≈ {theoretical_var:.2f} (超几何分布)")
    print(f"3. 标准差: Std[d] ≈ {theoretical_std:.2f}")
    print(f"4. 度数波动范围约 [{expected_degree - 3*theoretical_std:.0f}, {expected_degree + 3*theoretical_std:.0f}] (3σ)")
    print(f"\n对比 BiRegular 图: 每个节点度数恒为 α × M = {expected_degree:.0f}，方差为 0")


if __name__ == "__main__":
    analyze_degree_distribution()
