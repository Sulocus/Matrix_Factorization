# P1: 标准颜色配置

定义各指标的标准颜色，确保不同程序绘图风格一致。

**模块 ID**: P1
**SMF 路径**: `modules/outputs/plotting/colors.py`

---

## 🌐 宏观视角

### 系统定位

```
输出层
├── plotting/
│   ├── P1: colors.py     ← 本模块（颜色配置）
│   ├── P2: styles.py     ← 绘图样式
│   ├── P3: curves.py     ← 曲线绘制
│   └── P4: comparison.py ← 对比图
└── storage/
    ├── S1: json_io.py    ← JSON 读写
    └── S2: scanner.py    ← 结果扫描
```

### 引入动机

**问题**：不同程序使用不同颜色，结果难以比较

```
程序 A:          程序 B:
Q_Y = 红色       Q_Y = 蓝色
Q_W = 蓝色       Q_W = 红色
                 ↓
            混乱！
```

**解决方案**：统一颜色定义

```python
COLORS = {
    'Q_Y': '#d62728',       # 红色 - 重建质量
    'Q_W': '#ff7f0e',       # 橙色 - 左因子
    'Q_X': '#2ca02c',       # 绿色 - 右因子
    ...
}
```

### 物理图景 🌟

**颜色选择的逻辑**：

```
Q_Y (红色 #d62728):
  → 最重要的指标
  → 红色醒目

Q_W/Q_X (橙色/绿色):
  → 次要指标
  → 暖色/冷色区分左右因子

Q_W'/Q_X' (紫色/棕色):
  → 归一化版本
  → 与原始版本色系相近但可区分
```

### 使用场景

**适用**：
- 所有绘图代码
- 需要多指标对比时

---

## 🔬 微观视角

### 代码位置

| 程序 | 位置 | 行号 |
|------|------|------|
| bigamp/train.py | COLORS 定义 | 48-54 |
| agd/train_parallel.py | COLORS 定义 | 48-54 |
| 所有绘图程序 | COLORS 定义 | 同上 |

### 标准定义

```python
# =============================================================================
# Standard Colors (MUST be identical across all programs)
# =============================================================================
COLORS = {
    'Q_Y': '#d62728',       # Red - reconstruction quality
    'Q_W': '#ff7f0e',       # Orange - left factor raw
    'Q_X': '#2ca02c',       # Green - right factor raw
    'Q_W_prime': '#9467bd', # Purple - left factor normalized
    'Q_X_prime': '#8c564b', # Brown - right factor normalized
    'Gen_Error': '#1f77b4', # Blue - generalization error
    'Replica': '#17becf',   # Cyan - replica overlap
}
```

### 颜色规格

| 指标 | 颜色名 | HEX | RGB |
|------|--------|-----|-----|
| Q_Y | 红色 | #d62728 | (214, 39, 40) |
| Q_W | 橙色 | #ff7f0e | (255, 127, 14) |
| Q_X | 绿色 | #2ca02c | (44, 160, 44) |
| Q_W' | 紫色 | #9467bd | (148, 103, 189) |
| Q_X' | 棕色 | #8c564b | (140, 86, 75) |
| Gen_Error | 蓝色 | #1f77b4 | (31, 119, 180) |
| Replica | 青色 | #17becf | (23, 190, 207) |

### 使用示例

```python
import matplotlib.pyplot as plt

# 使用标准颜色
plt.plot(alpha_values, Q_Y_mean, color=COLORS['Q_Y'], label='Q_Y')
plt.plot(alpha_values, Q_W_mean, color=COLORS['Q_W'], label='Q_W')
plt.plot(alpha_values, Q_X_mean, color=COLORS['Q_X'], label='Q_X')

# 填充区域（误差带）
plt.fill_between(
    alpha_values,
    Q_Y_mean - Q_Y_std,
    Q_Y_mean + Q_Y_std,
    color=COLORS['Q_Y'],
    alpha=0.2
)
```

### 实现细节

1. **复制粘贴一致性**：COLORS 字典必须在所有程序中完全相同
2. **不导入共享模块**：Wang/ 程序独立，颜色定义内联
3. **matplotlib 兼容**：HEX 格式直接被 matplotlib 支持

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Define standard colors for metrics to ensure consistent plotting",
    "when_to_use_en": "All plotting code, multi-metric comparison",
    "tags_en": ["colors", "plotting", "visualization", "style", "consistency"],

    # 中文
    "purpose_zh": "定义指标的标准颜色以确保绘图一致性",
    "when_to_use_zh": "所有绘图代码、多指标对比",
    "tags_zh": ["颜色", "绘图", "可视化", "样式", "一致性"],

    # 技术参数
    "outputs": ["COLORS: Dict[str, str]"],
    "colors_defined": ["Q_Y", "Q_W", "Q_X", "Q_W_prime", "Q_X_prime", "Gen_Error", "Replica"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/outputs/plotting/colors.py`

---

*最后更新：2025年12月*
