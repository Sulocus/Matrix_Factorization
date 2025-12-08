# P2: 绘图样式配置

Matplotlib 绘图的标准样式设置（字体、线宽、图例等）。

**模块 ID**: P2
**SMF 路径**: `modules/outputs/plotting/styles.py`

---

## 🌐 宏观视角

### 系统定位

```
输出层/plotting/
├── P1: colors.py     ← 颜色配置
├── P2: styles.py     ← 本模块（样式配置）
├── P3: curves.py     ← 曲线绘制
└── P4: comparison.py ← 对比图
```

### 引入动机

**问题**：默认 matplotlib 样式不适合论文

```
默认样式:
- 字体太小
- 线条太细
- 刻度不清晰
- 图例位置不佳
```

**解决方案**：预定义适合论文的样式

### 物理图景 🌟

**好的科学绘图标准**：

```
1. 可读性:
   - 字体足够大（打印后仍清晰）
   - 高对比度（颜色区分明显）

2. 一致性:
   - 同一论文/项目风格统一
   - 符合期刊要求

3. 专业性:
   - 无多余装饰
   - 网格适度
   - 轴标签清晰
```

### 使用场景

**适用**：
- 所有绘图代码
- 论文/报告图表

---

## 🔬 微观视角

### 代码位置

| 程序 | 位置 | 行号 |
|------|------|------|
| bigamp/train.py | 绘图函数 | 560-620 |
| agd/train_parallel.py | 绘图函数 | 350-400 |

### 标准样式定义

```python
# =============================================================================
# Standard Plot Styles (MUST be identical across all programs)
# =============================================================================
PLOT_STYLES = {
    # Figure
    'figsize': (10, 6),
    'dpi': 150,

    # Lines
    'linewidth': 2.0,
    'markersize': 6,

    # Font
    'fontsize_title': 14,
    'fontsize_label': 12,
    'fontsize_tick': 10,
    'fontsize_legend': 10,

    # Grid
    'grid_alpha': 0.3,
    'grid_linestyle': '--',

    # Legend
    'legend_loc': 'best',
    'legend_framealpha': 0.8,

    # Error bands
    'fill_alpha': 0.2,
}
```

### 使用示例

```python
import matplotlib.pyplot as plt

def setup_plot_style():
    """Apply standard plot style settings."""
    plt.rcParams.update({
        'figure.figsize': PLOT_STYLES['figsize'],
        'figure.dpi': PLOT_STYLES['dpi'],
        'lines.linewidth': PLOT_STYLES['linewidth'],
        'lines.markersize': PLOT_STYLES['markersize'],
        'axes.titlesize': PLOT_STYLES['fontsize_title'],
        'axes.labelsize': PLOT_STYLES['fontsize_label'],
        'xtick.labelsize': PLOT_STYLES['fontsize_tick'],
        'ytick.labelsize': PLOT_STYLES['fontsize_tick'],
        'legend.fontsize': PLOT_STYLES['fontsize_legend'],
        'grid.alpha': PLOT_STYLES['grid_alpha'],
        'grid.linestyle': PLOT_STYLES['grid_linestyle'],
    })


def create_figure(title, xlabel, ylabel):
    """Create a figure with standard style."""
    fig, ax = plt.subplots(figsize=PLOT_STYLES['figsize'],
                           dpi=PLOT_STYLES['dpi'])
    ax.set_title(title, fontsize=PLOT_STYLES['fontsize_title'])
    ax.set_xlabel(xlabel, fontsize=PLOT_STYLES['fontsize_label'])
    ax.set_ylabel(ylabel, fontsize=PLOT_STYLES['fontsize_label'])
    ax.grid(True, alpha=PLOT_STYLES['grid_alpha'],
            linestyle=PLOT_STYLES['grid_linestyle'])
    return fig, ax
```

### 完整绘图模板

```python
def plot_overlap_curves(alpha_values, results, title, output_path):
    """Standard overlap curve plot."""
    fig, ax = create_figure(
        title=title,
        xlabel=r'$\tilde{\alpha}$',
        ylabel='Overlap'
    )

    # Plot each metric
    for metric, color in [('Q_Y', COLORS['Q_Y']),
                          ('Q_W', COLORS['Q_W']),
                          ('Q_X', COLORS['Q_X'])]:
        mean = results[f'{metric}_mean']
        std = results[f'{metric}_std']

        ax.plot(alpha_values, mean, color=color,
                linewidth=PLOT_STYLES['linewidth'],
                label=metric)
        ax.fill_between(alpha_values,
                        mean - std, mean + std,
                        color=color,
                        alpha=PLOT_STYLES['fill_alpha'])

    ax.legend(loc=PLOT_STYLES['legend_loc'],
              framealpha=PLOT_STYLES['legend_framealpha'])
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_STYLES['dpi'])
    plt.close()
```

### 实现细节

1. **rcParams vs 显式参数**：建议显式传参，避免全局状态污染
2. **DPI 设置**：
   - 屏幕显示：100-150
   - 论文打印：300+
3. **tight_layout**：自动调整边距，防止标签被截断

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Define standard matplotlib styles for publication-quality plots",
    "when_to_use_en": "All plotting code, paper/report figures",
    "tags_en": ["style", "matplotlib", "font", "linewidth", "publication"],

    # 中文
    "purpose_zh": "定义适合发表的 matplotlib 标准样式",
    "when_to_use_zh": "所有绘图代码、论文/报告图表",
    "tags_zh": ["样式", "matplotlib", "字体", "线宽", "发表"],

    # 技术参数
    "outputs": ["PLOT_STYLES: Dict"],
    "settings": ["figsize", "dpi", "linewidth", "fontsize", "grid", "legend"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/outputs/plotting/styles.py`

---

*最后更新：2025年12月*
