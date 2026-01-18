#!/usr/bin/env python3
"""
Teacher Matrix Analysis - Marchenko-Pastur Distribution Verification

This program generates various teacher matrices (Standard, Orthogonal, Spreading)
and compares their eigenvalue distributions with the Marchenko-Pastur law.

Usage:
    em                          # Use default configuration
    em --N 5000 --M 100        # Custom parameters
    em --samples 10            # 10 samples for ensemble average
"""

# =============================================================================
# ★★★ 配置参数 - 在这里修改 N 和 M ★★★
# =============================================================================
N = 10000           # 矩阵大小 (N1 = N2 = N)
M = 100         # 隐藏维度
SEED = 42          # 随机种子
DEVICE = 'cuda'    # 计算设备: 'cuda' 或 'cpu'

# 要生成的矩阵类型 (可删除不需要的类型)
MATRIX_TYPES = [
    'standard',              # 标准高斯
    # 'orthogonal',            # QR正交化
    'spreading_rademacher',  # Spreading (F ∈ {-1, +1})
    'spreading_gaussian',    # Spreading (F ~ N(0, 1))
]

# 输出设置
OUTPUT_FILENAME = 'marchenko_pastur_comparison.png'
HISTOGRAM_BINS = 100
N_SAMPLES = 1         # 随机采样次数 (Ensemble Average，多次取平均使直方图更平滑)
# =============================================================================

import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Literal, Optional
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TeacherMatrices:
    """Container for teacher matrix W, X, and their product Y."""
    W: torch.Tensor  # (N1, M)
    X: torch.Tensor  # (M, N2)
    Y: torch.Tensor  # (N1, N2) = (1/√M) W @ X
    matrix_type: str
    
    @property
    def N1(self) -> int:
        return self.W.shape[0]
    
    @property
    def N2(self) -> int:
        return self.X.shape[1]
    
    @property
    def M(self) -> int:
        return self.W.shape[1]


# =============================================================================
# Memory Estimation
# =============================================================================

def estimate_memory_gb(N: int, M: int, matrix_type: str) -> dict:
    """
    Estimate GPU memory requirements for generating a teacher matrix.
    
    Args:
        N: Matrix size (N1 = N2 = N)
        M: Hidden dimension
        matrix_type: 'standard', 'orthogonal', 'spreading_*'
        
    Returns:
        Dict with memory breakdown in GB
    """
    bytes_per_float = 2  # float16
    gb = 1024**3
    
    # Core matrices (always needed)
    W_memory = N * M * bytes_per_float / gb  # (N, M)
    X_memory = M * N * bytes_per_float / gb  # (M, N)
    
    # Y matrix
    Y_memory = N * N * bytes_per_float / gb  # (N, N)
    
    # Eigenvalue computation workspace
    if 'spreading' in matrix_type:
        # Full-rank: need Y^T @ Y (N×N) + eigvalsh workspace (~3N²)
        eig_workspace = 3 * N * N * bytes_per_float / gb
    else:
        # Low-rank: only need X @ X^T (M×M) + eigvalsh workspace (~3M²)
        eig_workspace = 3 * M * M * bytes_per_float / gb
    
    total = W_memory + X_memory + Y_memory + eig_workspace
    
    return {
        'W': W_memory,
        'X': X_memory,
        'Y': Y_memory,
        'eig_workspace': eig_workspace,
        'total': total,
    }


def check_memory_and_warn(N: int, M: int, matrix_types: list, gpu_memory_gb: float = 32.0) -> dict:
    """
    Check if matrix generation will exceed GPU memory and print warnings.
    
    Returns:
        Dict with flags for each matrix type indicating if it should be skipped
    """
    skip_flags = {}
    
    print(f"显存估算 (GPU: {gpu_memory_gb:.1f} GB):")
    print("-" * 40)
    
    for matrix_type in matrix_types:
        mem = estimate_memory_gb(N, M, matrix_type)
        
        if mem['total'] > gpu_memory_gb * 0.9:  # 90% threshold
            print(f"  {matrix_type}: {mem['total']:.2f} GB [⚠️ 跳过 - 超出显存]")
            skip_flags[matrix_type] = True
        elif mem['total'] > gpu_memory_gb * 0.7:  # 70% threshold
            print(f"  {matrix_type}: {mem['total']:.2f} GB [⚠️ 警告 - 接近上限]")
            skip_flags[matrix_type] = False
        else:
            print(f"  {matrix_type}: {mem['total']:.2f} GB [✓]")
            skip_flags[matrix_type] = False
    
    print("-" * 40)
    return skip_flags


# =============================================================================
# Teacher Matrix Generators
# =============================================================================

def generate_standard_teacher(
    N1: int,
    N2: int,
    M: int,
    device: torch.device,
    seed: int = 42,
) -> TeacherMatrices:
    """
    Generate standard Gaussian teacher matrices.
    
    W, X ~ N(0, 1), standard i.i.d. Gaussian entries.
    Y = (1/√M) W @ X for proper scaling.
    """
    torch.manual_seed(seed)
    
    # Standard Gaussian entries with unit variance (float16 to save memory)
    W = torch.randn(N1, M, device=device, dtype=torch.float16)
    X = torch.randn(M, N2, device=device, dtype=torch.float16)
    
    # Y = (1/√M) W @ X
    scale = 1.0 / math.sqrt(M)
    Y = scale * (W @ X)
    
    return TeacherMatrices(W=W, X=X, Y=Y, matrix_type='standard')


def generate_orthogonal_teacher(
    N1: int,
    N2: int,
    M: int,
    device: torch.device,
    seed: int = 42,
) -> TeacherMatrices:
    """
    Generate orthogonal teacher matrices using QR decomposition.
    
    Mathematical properties:
    - W^T @ W = N1 * I_M (scaled orthonormal columns)
    - X @ X^T = N2 * I_M (scaled orthonormal rows)
    - Eigenvalues of (1/N) X @ X^T = 1 (deterministic, at MP center)
    """
    torch.manual_seed(seed)
    
    # Generate random matrices (float32 for QR decomposition)
    W_raw = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_raw = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # QR decomposition for W (thin QR)
    W_ortho, _ = torch.linalg.qr(W_raw, mode='reduced')
    
    # QR decomposition for X^T, then transpose
    X_ortho_T, _ = torch.linalg.qr(X_raw.T, mode='reduced')
    X_ortho = X_ortho_T.T
    
    # Scale to match expected Frobenius norm of standard teacher
    # For ENERGY MATCHING with standard Gaussian:
    # Standard: E[||X_row||^2] = N (each row of X has expected length sqrt(N))
    # Orthogonal: ||X_ortho_row|| = 1 (unit vectors)
    # Scale factor: sqrt(N) to match energy
    #
    # This gives: X @ X^T = N * I_M
    # And: (1/N) X @ X^T = I_M (eigenvalues = 1, matching MP center)
    W = (W_ortho * math.sqrt(N1)).half()  # Convert to float16
    X = (X_ortho * math.sqrt(N2)).half()
    
    # Y = (1/√M) W @ X
    scale = 1.0 / math.sqrt(M)
    Y = scale * (W @ X)
    
    # Clean up float32 intermediate tensors
    del W_raw, X_raw, W_ortho, X_ortho
    
    return TeacherMatrices(W=W, X=X, Y=Y, matrix_type='orthogonal')


def generate_spreading_teacher(
    N1: int,
    N2: int,
    M: int,
    device: torch.device,
    seed: int = 42,
    f_distribution: Literal['rademacher', 'gaussian'] = 'rademacher',
) -> TeacherMatrices:
    """
    Generate teacher matrices with full F-matrix spreading.
    
    For each (i, j), the Y value is computed as:
        Y_ij = (1/√M) Σ_μ F_ijμ × W_iμ × X_μj
    
    where F_ijμ is an independent random variable for each (i, j, μ).
    
    This breaks all loop correlations and produces "maximally disordered" Y.
    
    Memory optimization: Uses chunked computation to avoid OOM for large N.
    """
    torch.manual_seed(seed)
    
    # Generate W, X with standard Gaussian entries (float16 to save memory)
    W = torch.randn(N1, M, device=device, dtype=torch.float16)
    X = torch.randn(M, N2, device=device, dtype=torch.float16)
    
    # Memory estimate for full F tensor: N1 * N2 * M * 2 bytes (float16)
    estimated_memory_gb = N1 * N2 * M * 2 / (1024**3)
    chunk_threshold_gb = 8.0  # Use chunking if F would exceed 8GB
    
    scale = 1.0 / math.sqrt(M)
    
    if estimated_memory_gb > chunk_threshold_gb:
        # Chunked computation: generate F row-by-row to save memory
        print(f"  [使用分块计算: F 张量估计 {estimated_memory_gb:.1f} GB > {chunk_threshold_gb} GB 阈值]")
        
        Y = torch.zeros(N1, N2, device=device, dtype=torch.float16)
        chunk_size = max(1, int(chunk_threshold_gb * (1024**3) / (N2 * M * 2)))
        chunk_size = min(chunk_size, N1)
        
        torch.manual_seed(seed + 12345)  # Different seed for F
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 12345)
        
        for i_start in range(0, N1, chunk_size):
            i_end = min(i_start + chunk_size, N1)
            chunk_N1 = i_end - i_start
            
            # Generate F chunk: (chunk_N1, N2, M)
            if f_distribution == 'rademacher':
                F_chunk = torch.randint(0, 2, (chunk_N1, N2, M), device=device, 
                                        dtype=torch.float16, generator=gen) * 2 - 1
            else:  # gaussian
                F_chunk = torch.randn(chunk_N1, N2, M, device=device, 
                                      dtype=torch.float16, generator=gen)
            
            # W[i_start:i_end, :] → (chunk_N1, 1, M)
            W_chunk = W[i_start:i_end, :].unsqueeze(1)
            # X.T → (1, N2, M)  
            X_expanded = X.T.unsqueeze(0)
            
            # Y[i_start:i_end, :] = (1/√M) Σ_μ F × W × X
            Y[i_start:i_end, :] = scale * (F_chunk * W_chunk * X_expanded).sum(dim=2)
            
            del F_chunk, W_chunk
            torch.cuda.empty_cache()
    else:
        # Standard full-tensor computation
        torch.manual_seed(seed + 12345)  # Different seed for F
        
        if f_distribution == 'rademacher':
            # F ∈ {-1, +1} with equal probability, use float16 to save memory
            F = torch.randint(0, 2, (N1, N2, M), device=device, dtype=torch.float16) * 2 - 1
        else:  # gaussian
            F = torch.randn(N1, N2, M, device=device, dtype=torch.float16)
        
        W_expanded = W.unsqueeze(1)           # (N1, 1, M)
        X_expanded = X.T.unsqueeze(0)         # (1, N2, M)
        
        # Y_ij = (1/√M) Σ_μ F_ijμ × W_iμ × X_μj
        Y = scale * (F * W_expanded * X_expanded).sum(dim=2)  # (N1, N2)
        
        del F, W_expanded, X_expanded
        torch.cuda.empty_cache()
    
    matrix_type = f'spreading_{f_distribution}'
    return TeacherMatrices(W=W, X=X, Y=Y, matrix_type=matrix_type)


# =============================================================================
# Eigenvalue Analysis
# =============================================================================

def marchenko_pastur_pdf(x: np.ndarray, gamma: float, sigma: float = 1.0) -> np.ndarray:
    """
    Marchenko-Pastur probability density function.
    
    For a M×N random matrix X with i.i.d. entries of variance σ², 
    the eigenvalue density of (1/N) X^T @ X converges to this distribution.
    
    Args:
        x: Eigenvalue points to evaluate
        gamma: Aspect ratio M/N (0 < γ ≤ 1)
        sigma: Standard deviation of matrix entries
        
    Returns:
        PDF values at each point x
    """
    lambda_minus = sigma**2 * (1 - np.sqrt(gamma))**2
    lambda_plus = sigma**2 * (1 + np.sqrt(gamma))**2
    
    pdf = np.zeros_like(x, dtype=np.float64)
    
    # Find valid range
    valid = (x >= lambda_minus) & (x <= lambda_plus) & (x > 0)
    
    if np.any(valid):
        x_valid = x[valid]
        # Avoid division by zero
        denominator = 2 * np.pi * gamma * sigma**2 * x_valid
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        sqrt_term = (lambda_plus - x_valid) * (x_valid - lambda_minus)
        sqrt_term = np.maximum(sqrt_term, 0)  # Numerical safety
        
        pdf[valid] = np.sqrt(sqrt_term) / denominator
        
        # Handle any NaN or Inf
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    
    return pdf


def compute_gram_eigenvalues(Y: torch.Tensor, normalize: bool = True) -> np.ndarray:
    """
    Compute eigenvalues of the Gram matrix Y^T @ Y.
    
    Args:
        Y: Input matrix (N1, N2)
        normalize: If True, compute eigenvalues of (1/N1) Y^T @ Y
        
    Returns:
        Sorted eigenvalues as numpy array
    """
    if normalize:
        gram = (1.0 / Y.shape[0]) * (Y.T @ Y)
    else:
        gram = Y.T @ Y
    
    eigenvalues = torch.linalg.eigvalsh(gram.float())  # float32 for eigvalsh
    return eigenvalues.cpu().numpy()


# =============================================================================
# Visualization
# =============================================================================

def plot_ensemble_average(
    merged_eigenvalues: dict,
    N: int, M: int, n_samples: int,
    output_path: Optional[Path] = None,
    bins: int = 100,
) -> Figure:
    """
    Plot eigenvalue distributions from ensemble averaging.
    
    CRITICAL PHYSICS DISTINCTIONS:
    
    1. Standard Teacher (Low-rank, MP applies to X):
       - Y = (1/√M) W @ X has rank M
       - W, X ~ N(0, 1) are i.i.d. Gaussian
       - Analyze: (1/N) X @ X^T (M×M matrix), γ = M/N
       - MP law applies!
       
    2. Orthogonal Teacher (Low-rank, DETERMINISTIC):
       - Y = (1/√M) W @ X has rank M
       - BUT W, X are QR-orthogonalized → NOT random Gaussian!
       - X @ X^T = N * I_M → eigenvalues all equal to 1 (after normalization)
       - MP law does NOT apply! (Deterministic spectrum)
       
    3. Spreading Teacher (Full-rank, MP applies):
       - Y_ij = (1/√M) Σ F_ijμ W_iμ X_μj → i.i.d. Gaussian by CLT
       - Y is full-rank N×N random matrix
       - Analyze: (1/N) Y^T @ Y, γ = 1
       - MP law applies!
    """
    n_types = len(merged_eigenvalues)
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5))
    
    if n_types == 1:
        axes = [axes]
    
    for ax, (matrix_type, eigenvalues) in zip(axes, merged_eigenvalues.items()):
        is_spreading = 'spreading' in matrix_type
        is_orthogonal = 'orthogonal' in matrix_type
        
        # Filter positive eigenvalues
        eig_positive = eigenvalues[eigenvalues > 1e-10]
        
        # Compute theoretical parameters
        if is_spreading:
            gamma = 1.0
            sigma = 1.0  # Y ~ N(0, 1) by CLT
        else:
            gamma = M / N
            sigma = 1.0
        
        lambda_minus = sigma**2 * (1 - math.sqrt(gamma))**2
        lambda_plus = sigma**2 * (1 + math.sqrt(gamma))**2
        
        # Plot histogram
        if len(eig_positive) > 1:
            eig_min, eig_max = eig_positive.min(), eig_positive.max()
            if is_orthogonal:
                # Orthogonal: tight range around 1.0
                bin_edges = np.linspace(0.9, 1.1, bins + 1)
            else:
                bin_edges = np.linspace(max(0, lambda_minus * 0.5), lambda_plus * 1.5, bins + 1)
            
            ax.hist(eig_positive, bins=bin_edges, density=True, alpha=0.7,
                   color='steelblue', edgecolor='white', linewidth=0.5,
                   label=f'Empirical ({n_samples} samples)')
        
        # Plot theoretical curve
        if not is_orthogonal:
            x_range = np.linspace(max(0, lambda_minus * 0.5), lambda_plus * 1.5, 500)
            y_theory = marchenko_pastur_pdf(x_range, gamma, sigma)
            ax.plot(x_range, y_theory, 'r-', linewidth=2, label='Marchenko-Pastur')
            
            ax.axvline(lambda_minus, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axvline(lambda_plus, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.set_xlim(max(0, lambda_minus * 0.5), lambda_plus * 1.5)
        else:
            ax.axvline(1.0, color='red', linestyle='-', linewidth=2, label='Theory: λ = 1.0')
            ax.set_xlim(0.9, 1.1)
        
        # Styling
        if is_spreading:
            title_extra = "Full-rank, γ=1"
        elif is_orthogonal:
            title_extra = "Orthogonal (deterministic)"
        else:
            title_extra = f"Low-rank, γ={gamma:.4f}"
        
        ax.set_xlabel('Eigenvalue λ', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{matrix_type}\n{title_extra}', fontsize=10)
        ax.legend(fontsize=9)
        
        # Info text
        info_text = f'N={N}, M={M}\n{n_samples} samples\n{len(eig_positive)} eigenvalues'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    return fig


# =============================================================================
# Statistics
# =============================================================================

def print_matrix_statistics(matrices: TeacherMatrices) -> None:
    """Print detailed statistics about the teacher matrices."""
    W, X, Y = matrices.W, matrices.X, matrices.Y
    
    print(f"\n{'='*60}")
    print(f"Matrix Type: {matrices.matrix_type}")
    print(f"{'='*60}")
    print(f"Dimensions: N1={matrices.N1}, N2={matrices.N2}, M={matrices.M}")
    print()
    
    # W statistics
    print("W matrix:")
    print(f"  Shape: {tuple(W.shape)}")
    print(f"  Mean: {W.float().mean().item():.6f}")
    print(f"  Std:  {W.float().std().item():.6f}")
    print(f"  ||W||_F^2: {(W.float()**2).sum().item():.2f}")  # float32 to avoid overflow
    print(f"  Expected ||W||_F^2 (std): {matrices.N1 * matrices.M:.2f}")
    
    # Check orthogonality (only M×M matrix, memory safe)
    WtW = (W.T @ W).float() / matrices.N1
    off_diag = WtW - torch.eye(matrices.M, device=W.device)
    print(f"  W^T @ W / N1 off-diagonal Frobenius: {torch.sqrt((off_diag**2).sum()).item():.6f}")
    del WtW, off_diag
    
    print()
    
    # X statistics
    print("X matrix:")
    print(f"  Shape: {tuple(X.shape)}")
    print(f"  Mean: {X.float().mean().item():.6f}")
    print(f"  Std:  {X.float().std().item():.6f}")
    print(f"  ||X||_F^2: {(X.float()**2).sum().item():.2f}")  # float32 to avoid overflow
    print(f"  Expected ||X||_F^2 (std): {matrices.M * matrices.N2:.2f}")
    
    print()
    
    # Y statistics (use float32 for accurate stats)
    print("Y matrix:")
    print(f"  Shape: {tuple(Y.shape)}")
    print(f"  Mean: {Y.float().mean().item():.6f}")
    print(f"  Std:  {Y.float().std().item():.6f}")
    print(f"  Min:  {Y.min().item():.6f}")
    print(f"  Max:  {Y.max().item():.6f}")
    
    # Eigenvalue statistics
    is_spreading = 'spreading' in matrices.matrix_type
    
    if is_spreading:
        # Spreading 矩阵是满秩的，必须用 Y^T @ Y
        gram = (1.0 / matrices.N1) * (Y.T @ Y)
        eigenvalues = torch.linalg.eigvalsh(gram.float()).cpu().numpy()  # float32 for eigvalsh
        print()
        print("Gram matrix (1/N) Y^T @ Y eigenvalues (full-rank):")
        del gram
    else:
        # 低秩矩阵: use core M×M matrix
        core_gram = (1.0 / matrices.N2) * (X @ X.T)  # M×M
        eigenvalues = torch.linalg.eigvalsh(core_gram.float()).cpu().numpy()  # float32 for eigvalsh
        print()
        print("Core matrix (1/N) X @ X^T eigenvalues (M×M):")
        del core_gram
    
    print(f"  Min eigenvalue: {eigenvalues.min():.6f}")
    print(f"  Max eigenvalue: {eigenvalues.max():.6f}")
    print(f"  Mean eigenvalue: {eigenvalues.mean():.6f}")
    
    gamma = matrices.M / matrices.N1
    sigma_y = float(Y.float().std())
    lambda_minus = sigma_y**2 * (1 - math.sqrt(gamma))**2
    lambda_plus = sigma_y**2 * (1 + math.sqrt(gamma))**2
    print(f"  Theoretical λ_min (MP): {lambda_minus:.6f}")
    print(f"  Theoretical λ_max (MP): {lambda_plus:.6f}")
    
    torch.cuda.empty_cache()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Teacher Matrix Analysis and Marchenko-Pastur Verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
直接运行 (使用文件顶部的配置):
  em

使用命令行参数覆盖配置:
  em --N 500 --M 100
  em --samples 20  # 20次随机采样取平均
        """
    )
    
    # 使用文件顶部的配置作为默认值
    parser.add_argument('--N', type=int, default=N,
                       help=f'矩阵大小 (默认: {N})')
    parser.add_argument('--M', type=int, default=M,
                       help=f'隐藏维度 (默认: {M})')
    parser.add_argument('--seed', type=int, default=SEED,
                       help=f'随机种子 (默认: {SEED})')
    parser.add_argument('--device', type=str, default=DEVICE,
                       choices=['cuda', 'cpu'],
                       help=f'计算设备 (默认: {DEVICE})')
    parser.add_argument('--types', nargs='+', 
                       default=MATRIX_TYPES,
                       help='要生成的矩阵类型')
    parser.add_argument('--output', type=str, default=OUTPUT_FILENAME,
                       help=f'输出文件名 (默认: {OUTPUT_FILENAME})')
    parser.add_argument('--no-plot', action='store_true',
                       help='跳过生成图像')
    parser.add_argument('--bins', type=int, default=HISTOGRAM_BINS,
                       help=f'直方图箱数 (默认: {HISTOGRAM_BINS})')
    parser.add_argument('--samples', type=int, default=N_SAMPLES,
                       help=f'随机采样次数 (默认: {N_SAMPLES})')
    
    args = parser.parse_args()
    
    # 设备设置
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print(f"参数: N={args.N}, M={args.M}, seed={args.seed}, samples={args.samples}")
    print()
    
    # 获取 GPU 显存大小
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        gpu_memory_gb = 64.0  # Assume 64GB for CPU
    
    # 显存估算和检查
    skip_flags = check_memory_and_warn(args.N, args.M, args.types, gpu_memory_gb)
    
    # 过滤掉会 OOM 的类型
    valid_types = [t for t in args.types if not skip_flags.get(t, False)]
    if len(valid_types) < len(args.types):
        skipped = set(args.types) - set(valid_types)
        print(f"跳过的矩阵类型: {skipped}")
    
    # 收集多次采样的特征值 (Ensemble Average)
    all_eigenvalues = {matrix_type: [] for matrix_type in valid_types}
    
    for sample_idx in range(args.samples):
        seed = args.seed + sample_idx * 1000  # 每次用不同的种子
        
        if args.samples > 1:
            print(f"\n--- Sample {sample_idx + 1}/{args.samples} (seed={seed}) ---")
        
        for matrix_type in valid_types:  # 使用过滤后的 valid_types
            if args.samples == 1:
                print(f"生成 {matrix_type} 教师矩阵...")
            
            if matrix_type == 'standard':
                matrices = generate_standard_teacher(args.N, args.N, args.M, device, seed)
            elif matrix_type == 'orthogonal':
                matrices = generate_orthogonal_teacher(args.N, args.N, args.M, device, seed)
            elif matrix_type == 'spreading_rademacher':
                matrices = generate_spreading_teacher(args.N, args.N, args.M, device, seed, 'rademacher')
            elif matrix_type == 'spreading_gaussian':
                matrices = generate_spreading_teacher(args.N, args.N, args.M, device, seed, 'gaussian')
            else:
                continue
            
            # 收集特征值 (统一使用 Y^T @ Y 分析)
            # 
            # 物理解释：
            # - Standard/Orthogonal: Y = (1/√M) W @ X，所以 Y 的秩只有 M，
            #   Y^T Y 会有 N-M 个零特征值。
            # - Spreading: Y 是满秩的 N×N 矩阵。
            #
            # 缩放问题：
            # - 对于 Standard，W^T W ≈ N * I，所以 Y^T Y 的非零特征值
            #   会比原来的 X @ X^T 大 N/M 倍。
            # - 为了让图贴合 MP 理论曲线，需要乘以 scaling_factor = M/N。
            
            Y = matrices.Y
            is_spreading = 'spreading' in matrix_type
            
            # 统一计算 N×N 的 Gram 矩阵
            gram = (1.0 / args.N) * (Y.T @ Y)
            all_eigs = torch.linalg.eigvalsh(gram.float()).cpu().numpy()
            del gram
            
            if is_spreading:
                # Spreading 矩阵是满秩的，直接使用全部特征值
                eigenvalues = all_eigs
            else:
                # 低秩矩阵 (standard/orthogonal)：
                # 1. 过滤零特征值 - 只保留最大的 M 个（即非零部分）
                eigs = np.sort(all_eigs)[-args.M:]
                
                # 2. 重新缩放 - 因为 Y^T Y 把能量集中在 M 个方向上，
                #    导致特征值比直接算 X @ X^T 大了 N/M 倍。
                #    乘以 M/N 让它回到 [λ_min, λ_max] 理论区间。
                scaling_factor = args.M / args.N
                eigenvalues = eigs * scaling_factor
            
            all_eigenvalues[matrix_type].append(eigenvalues)
            
            # 只在第一个 sample 打印统计信息
            if sample_idx == 0 and args.samples == 1:
                print_matrix_statistics(matrices)
            
            # 清理内存
            del matrices
            torch.cuda.empty_cache()
    
    # 合并所有特征值
    print(f"\n合并 {args.samples} 次采样的特征值...")
    merged_eigenvalues = {}
    for matrix_type in valid_types:
        if all_eigenvalues[matrix_type]:
            merged_eigenvalues[matrix_type] = np.concatenate(all_eigenvalues[matrix_type])
            print(f"  {matrix_type}: {len(merged_eigenvalues[matrix_type])} 个特征值")
    
    # 生成 Ensemble Average 图像
    if not args.no_plot and merged_eigenvalues:
        print(f"\n生成 Marchenko-Pastur 分布对比图 (Ensemble Average)...")
        output_path = Path(__file__).parent / args.output
        plot_ensemble_average(
            merged_eigenvalues, 
            args.N, args.M, args.samples,
            output_path, args.bins
        )
    
    print("\n完成!")


if __name__ == '__main__':
    main()
