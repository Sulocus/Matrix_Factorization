# P4: 算法对比图

比较不同算法、不同尺寸、不同图类型的实验结果。

**模块 ID**: P4
**SMF 路径**: `modules/outputs/plotting/comparison.py`

---

## 🌐 宏观视角

### 系统定位

```
输出层/plotting/
├── P1: colors.py     ← 颜色配置
├── P2: styles.py     ← 样式配置
├── P3: curves.py     ← 单次实验曲线
└── P4: comparison.py ← 本模块（多实验对比）
```

### 引入动机

常见对比需求：

1. **算法对比**：AGD vs BiG-AMP
2. **尺寸效应**：N=200 vs N=500 vs N=1000
3. **图类型对比**：随机图 vs 双正则图 vs 低循环图
4. **参数敏感性**：不同 damping、不同学习率

### 物理图景 🌟

**有限尺寸效应对比**：

```
Q_Y
 1 │                    ╭── N=1000 (接近理论)
   │                  ╱╭── N=500
   │                ╱╱╭── N=200
   │              ╱╱╱
   │            ╱╱╱
   │         ╭╯╱╱
   │       ╭╯╱╱
 0 │──────╯╱╱
   └──────────────────────── α̃

N 越大，相转移越陡峭，越接近理论预测
```

**算法对比**：

```
Q_Y
 1 │              BiG-AMP ─╮╭── AGD
   │                       ╰╯
   │
   │    理论: 两者应该收敛到相同结果
   │    实际: BiG-AMP 更快，AGD 可能陷入局部最优
   │
 0 │
   └──────────────────────── α̃
```

### 使用场景

**适用**：
- 论文结果对比图
- 方法验证
- 参数选择分析

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/compare_sizes.py | `plot_comparison` | 650-720 |
| analysis/compare_algorithms.py | `plot_comparison` | 200-280 |

### 输入/输出

```python
def plot_comparison(experiments, title, output_path, metric='Q_Y'):
    """
    Args:
        experiments: List[Dict] - 实验列表
            [{
                'name': str,           # 图例名称
                'alpha': List[float],  # α 值
                'mean': List[float],   # 均值
                'std': List[float],    # 标准差（可选）
                'color': str,          # 颜色（可选）
                'linestyle': str,      # 线型（可选）
            }, ...]
        title: str - 图表标题
        output_path: str - 输出路径
        metric: str - 指标名称（用于 Y 轴标签）

    Returns:
        None (保存图片)
    """
```

### 标准实现

```python
def plot_comparison(experiments, title, output_path, metric='Q_Y'):
    """Compare multiple experiments in one plot."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Color palette for automatic assignment
    default_colors = plt.cm.tab10.colors

    for i, exp in enumerate(experiments):
        color = exp.get('color', default_colors[i % 10])
        linestyle = exp.get('linestyle', '-')
        marker = exp.get('marker', 'o')

        # Main curve
        ax.plot(exp['alpha'], exp['mean'],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                linewidth=2.0,
                label=exp['name'])

        # Error band if available
        if 'std' in exp and exp['std'] is not None:
            mean = np.array(exp['mean'])
            std = np.array(exp['std'])
            ax.fill_between(exp['alpha'],
                            mean - std, mean + std,
                            color=color,
                            alpha=0.15)

    ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison: {output_path}")
```

### 尺寸效应对比

```python
def plot_finite_size_effect(sizes, results_by_size, output_path):
    """Plot finite-size scaling comparison."""
    experiments = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, N in enumerate(sizes):
        experiments.append({
            'name': f'N={N}',
            'alpha': results_by_size[N]['alpha'],
            'mean': results_by_size[N]['Q_Y_mean'],
            'std': results_by_size[N]['Q_Y_std'],
            'color': colors[i % len(colors)],
        })

    plot_comparison(
        experiments,
        title='Finite-Size Effect',
        output_path=output_path,
        metric=r'$Q_Y$'
    )
```

### 算法对比

```python
def plot_algorithm_comparison(agd_results, bigamp_results, alpha_values, output_path):
    """Compare AGD and BiG-AMP."""
    experiments = [
        {
            'name': 'AGD (20k epochs)',
            'alpha': alpha_values,
            'mean': agd_results['Q_Y_mean'],
            'std': agd_results['Q_Y_std'],
            'color': '#d62728',
            'linestyle': '-',
            'marker': 'o',
        },
        {
            'name': 'BiG-AMP (1k steps)',
            'alpha': alpha_values,
            'mean': bigamp_results['Q_Y_mean'],
            'std': bigamp_results['Q_Y_std'],
            'color': '#1f77b4',
            'linestyle': '--',
            'marker': 's',
        },
    ]

    plot_comparison(
        experiments,
        title='AGD vs BiG-AMP Comparison',
        output_path=output_path,
        metric=r'$Q_Y$'
    )
```

### 实现细节

1. **自动颜色分配**：使用 `plt.cm.tab10` 色板
2. **线型区分**：不同算法使用不同线型
3. **标记区分**：`'o'`, `'s'`, `'^'`, `'v'` 等
4. **透明度层叠**：多个误差带重叠时使用低透明度

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Create comparison plots for multiple experiments",
    "when_to_use_en": "Algorithm comparison, finite-size effects, parameter sensitivity",
    "tags_en": ["comparison", "plot", "algorithm", "finite-size", "benchmark"],

    # 中文
    "purpose_zh": "创建多实验对比图",
    "when_to_use_zh": "算法对比、有限尺寸效应、参数敏感性分析",
    "tags_zh": ["对比", "绘图", "算法", "有限尺寸", "基准测试"],

    # 技术参数
    "inputs": ["experiments: List[Dict]", "title", "output_path", "metric"],
    "outputs": ["PNG file"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/outputs/plotting/comparison.py`

---

*最后更新：2025年12月*
