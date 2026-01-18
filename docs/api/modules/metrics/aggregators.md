# M7: 指标聚合

统计聚合多次试验的评估指标（mean, std 等）。

**模块 ID**: M7
**SMF 路径**: `modules/metrics/aggregators.py`

---

## 🌐 宏观视角

### 系统定位

```
评估指标层
├── M1-M6: 单次试验指标
└── M7: aggregators.py  ← 本模块（多次试验聚合）
```

### 引入动机

实验通常运行多次（SAMPLES_PER_ALPHA 次），需要：
1. 计算均值作为代表值
2. 计算标准差估计不确定性
3. 可选：中位数、四分位数等

### 物理图景 🌟

```
单次试验:        聚合后:
Q_Y[1] = 0.82
Q_Y[2] = 0.85    →  Q_Y_mean = 0.84
Q_Y[3] = 0.84       Q_Y_std = 0.015
Q_Y[4] = 0.86
Q_Y[5] = 0.83

绘图时:
  ──○── ← mean
  ▒▒▒▒▒ ← ±std 填充
```

### 使用场景

**适用**：
- 结果汇总和可视化
- 误差估计
- 结果保存

---

## 🔬 微观视角

### 标准实现

```python
def aggregate_metrics(values):
    """
    Args:
        values: List[float] - 多次试验的指标值

    Returns:
        mean: float - 均值
        std: float - 标准差
    """
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def aggregate_all_metrics(results_per_alpha):
    """
    Args:
        results_per_alpha: Dict[alpha, List[Dict]] - 每个 alpha 的多次结果

    Returns:
        aggregated: Dict[alpha, Dict] - 聚合后的结果
            {alpha: {'Q_Y_mean': ..., 'Q_Y_std': ..., ...}}
    """
    aggregated = {}
    for alpha, trials in results_per_alpha.items():
        metrics = {}
        for key in trials[0].keys():
            values = [t[key] for t in trials]
            mean, std = aggregate_metrics(values)
            metrics[f'{key}_mean'] = mean
            metrics[f'{key}_std'] = std
        aggregated[alpha] = metrics
    return aggregated
```

### 输出格式

```python
{
    0.0: {'Q_Y_mean': 0.02, 'Q_Y_std': 0.01, 'Q_W_mean': ..., ...},
    0.1: {'Q_Y_mean': 0.05, 'Q_Y_std': 0.02, ...},
    ...
    2.0: {'Q_Y_mean': 0.99, 'Q_Y_std': 0.001, ...},
}
```

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "Aggregate metrics across multiple trials (mean, std)",
    "when_to_use_en": "Result summarization, error estimation, visualization",
    "tags_en": ["aggregate", "mean", "std", "statistics", "trials"],
    "tags_zh": ["聚合", "均值", "标准差", "统计", "试验"],
}
```

---

*最后更新：2025年12月*
