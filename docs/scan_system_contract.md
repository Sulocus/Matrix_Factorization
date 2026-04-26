# Scan System Contract

当前主链路只接受 canonical `scan.axes`。旧的 `scan_mode`、`alpha_scan`、`steps_scan`、`nested_scan`、`hysteresis_scan` 不再作为运行入口；如果出现在 YAML 中，loader/preflight 会报错。

## Schema

```yaml
scan:
  axes:
    alpha:
      path: alpha
      values: {start: 0.0, stop: 2.0, step: 0.1}

    damping:
      path: algorithm_params.damping
      values: [0.2, 0.5, 0.8]

    init:
      kind: composite
      values:
        cold:
          algorithm_params.init_mode: random
        warm_095:
          algorithm_params.init_mode: teacher
          algorithm_params.init_overlap: 0.95
```

规则：

- 普通 axis 的 `path` 必须是 `ParameterSpec` 路径，或特殊路径 `alpha` / `max_steps`。
- `kind: composite` 用于一个 axis 同时覆盖多个参数，例如 matrix size 或 cold/warm start。
- `max_steps` 是普通 axis，会映射到 `training.max_steps`；配合一个固定 `alpha` 表达旧 steps scan。
- `size`、`init`、`damping`、`onsager` 等都只是坐标，不再有专用 handler。

## Execution

`ScanPlan` 会展开所有 axis 组合为 `ScanPoint`：

```text
ScanPoint
├─ point_id
├─ coordinates
├─ overrides
├─ effective_config_hash
├─ physical_sensitive_paths
└─ group_id
```

执行层只允许同一个 non-alpha 坐标组内做 alpha folding。其他 axis 默认不跨组折叠：

- `alpha`：同组内可折叠。
- `max_steps`：走 steps runtime scan，不与 alpha batch 混用。
- `size`：改变 shape，必须隔离成不同 group。
- `init`：改变 initialization 语义，必须隔离成不同 group。
- `damping/onsager`：改变算法参数，必须隔离成不同 group。

## ResultCube

所有 canonical scan 结果写入 `metrics.json.result_cube`：

```text
result_cube
├─ axes
├─ points
│  └─ point_id -> coordinates / overrides / group_id
├─ metrics
│  └─ point_id -> metric dict
├─ artifacts
├─ groups
└─ metric_semantics
```

alpha-only 结果会继续导出旧 `metrics` / `results` flat schema；multi-axis 结果以 `ResultCube` 为正式索引。

## PlotQuery

新绘图只查询 `ResultCube` 坐标和 metric：

```yaml
output:
  plots:
    - x: alpha
      y: Q_Y_mean
      where:
        size: N200_M50
      series_by: [onsager, damping, init]
      filename: qy_compare.png
```

约束：

- `x`、`where`、`series_by`、`compare` 必须引用存在的 scan axis。
- `y` 必须是当前 algorithm 的 `MetricSpec` 声明 key。
- 查询不到点、缺 metric、坐标不存在都会在 preflight 或 save 阶段报错。
