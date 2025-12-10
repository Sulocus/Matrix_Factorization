# P3: 曲线绘制

Overlap 曲线和相转移图的标准绘制函数。

**模块 ID**: P3
**SMF 路径**: `modules/outputs/plotting/curves.py`

---

## 🌐 宏观视角

### 系统定位

```
输出层/plotting/
├── P1: colors.py     ← 颜色配置
├── P2: styles.py     ← 样式配置
├── P3: curves.py     ← 本模块（曲线绘制）
└── P4: comparison.py ← 对比图
```

### 引入动机

绘制 overlap vs α 曲线是最常见的可视化需求：

```
标准相转移图:

Q_Y
 1 │                    ╭────
   │                   ╱
   │                  ╱
   │                 ╱
   │               ╱
   │            ╭─╯
 0 │────────────╯
   └──────────────────────── α̃
              α_c
```

### 物理图景 🌟

**相转移图的解读**：

```
α < α_c:
  → Q_Y ≈ 0
  → 观测不足，无法重建
  → "无信息相"

α ≈ α_c:
  → Q_Y 急剧上升
  → 相转移点

α > α_c:
  → Q_Y → 1
  → 可以完美重建
  → "有信息相"
```

### 使用场景

**适用**：
- 实验结果可视化
- 相转移分析
- 算法比较

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/train.py | `plot_results` | 560-620 |
| agd/train_parallel.py | `plot_results` | 350-400 |

### 输入/输出

```python
def plot_overlap_curves(alpha_values, results, title, output_path,
                        metrics=['Q_Y', 'Q_W', 'Q_X']):
    """
    Args:
        alpha_values: List[float] - α 值列表
        results: Dict - 聚合后的结果
            {metric_mean: List, metric_std: List, ...}
        title: str - 图表标题
        output_path: str - 输出文件路径
        metrics: List[str] - 要绘制的指标

    Returns:
        None (保存图片到文件)
    """
```

### 标准实现

```python
def plot_overlap_curves(alpha_values, results, title, output_path,
                        metrics=['Q_Y', 'Q_W', 'Q_X']):
    """Plot overlap curves with error bands."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for metric in metrics:
        if f'{metric}_mean' not in results:
            continue

        mean = results[f'{metric}_mean']
        std = results.get(f'{metric}_std', None)
        color = COLORS.get(metric, '#333333')

        # Main curve
        ax.plot(alpha_values, mean,
                color=color,
                linewidth=2.0,
                label=metric,
                marker='o',
                markersize=4)

        # Error band
        if std is not None:
            ax.fill_between(alpha_values,
                            [m - s for m, s in zip(mean, std)],
                            [m + s for m, s in zip(mean, std)],
                            color=color,
                            alpha=0.2)

    # Styling
    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax.set_ylabel('Overlap', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
```

### 带理论曲线的版本

```python
def plot_with_theory(alpha_values, results, theory_curve, title, output_path):
    """Plot experimental results with theoretical prediction."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Experimental data
    mean = results['Q_Y_mean']
    std = results['Q_Y_std']
    ax.errorbar(alpha_values, mean, yerr=std,
                color=COLORS['Q_Y'],
                fmt='o',
                markersize=6,
                capsize=3,
                label='Experiment')

    # Theory curve
    ax.plot(theory_curve['alpha'], theory_curve['Q_Y'],
            color='black',
            linestyle='--',
            linewidth=1.5,
            label='Theory')

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax.set_ylabel(r'$Q_Y$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

### 文件命名规范

```python
def generate_filename(N1, N2, M, algorithm, epochs_or_steps, extra_info=''):
    """Generate standard filename for plots."""
    # 格式: Algorithm_N1xN2_M{M}_Epoch{E}[_extra].png
    base = f"{algorithm}_{N1}x{N2}_M{M}"
    if 'bigamp' in algorithm.lower():
        base += f"_Steps{epochs_or_steps}"
    else:
        base += f"_Epoch{epochs_or_steps}"
    if extra_info:
        base += f"_{extra_info}"
    return base + ".png"
```

### 实现细节

1. **α 轴标签**：使用 LaTeX 格式 `r'$\tilde{\alpha}$'`
2. **Y 轴范围**：固定 [-0.05, 1.05] 便于比较
3. **误差带**：`fill_between` 更美观，`errorbar` 更精确
4. **关闭图形**：`plt.close()` 防止内存泄漏

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Plot overlap curves and phase transition diagrams",
    "when_to_use_en": "Result visualization, phase transition analysis",
    "tags_en": ["plot", "curve", "overlap", "phase transition", "visualization"],

    # 中文
    "purpose_zh": "绘制 overlap 曲线和相转移图",
    "when_to_use_zh": "结果可视化、相转移分析",
    "tags_zh": ["绘图", "曲线", "overlap", "相转移", "可视化"],

    # 技术参数
    "inputs": ["alpha_values", "results", "title", "output_path", "metrics"],
    "outputs": ["PNG file"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/outputs/plotting/curves.py`

---

*最后更新：2025年12月*
