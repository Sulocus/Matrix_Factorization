# Result Schema Contract

这份文档说明当前主链路落盘结果的结构。它不改变任何 metric 公式，只规定“结果文件里每一层代表什么”，避免后续 agent 把调试 payload、plot helper 或 tensor placeholder 当成正式物理结果。

## 主 run 目录

`ExperimentResult.save()` 生成的 run 目录是当前 canonical result schema：

```text
run_dir/
├─ config.json
├─ metadata.json
├─ metrics.json
├─ output_contract.json
├─ events.jsonl
├─ manifest.json
├─ artifacts/
│  └─ results.pt
├─ results.pt -> artifacts/results.pt
├─ checkpoints/
└─ plots/
```

`results/latest/` 只是展示快照，不是正式分析输入。它只复制轻量 summary 和选中的 plot。

## metrics.json

`metrics.json` 是轻量后处理优先读取的文件：

```text
metrics.json
├─ schema_version
├─ experiment_id
├─ config
├─ contract
│  ├─ experiment_plan
│  ├─ output_plan
│  ├─ runtime_resource_plan
│  ├─ algorithm_config_trace
│  └─ runtime_extension_report
├─ factor_payload_contract
├─ scan_dimension
├─ scan_values
├─ available_metric_keys
├─ metric_schema
├─ metric_semantics
├─ metric_contracts
├─ metrics
└─ results
```

关键含义：

- `available_metric_keys`：本次 run 实际出现的 flat metric key。
- `metric_schema`：把本次实际 flat key 映射到 canonical semantic class。
- `metric_semantics`：当前 algorithm 可能产生的 metric 语义表。
- `metric_contracts`：每个 scan point 的 metric payload 是由哪些 `MetricSpec` 覆盖。
- `factor_payload_contract`：声明 `W_students/X_students` 是否是真实 matrix factor。tensor metrics-only 路径这里会明确标记 unavailable。
- `metrics`：旧兼容 flat dict，形式是 `scan_value -> metric dict`。
- `results`：轻量 `SingleRunResult` 展开，不包含 tensor payload。

保存阶段会重新跑 `MetricSpec` 校验。即使某个测试或后处理手工构造了 `SingleRunResult`，只要塞入未声明 metric/artifact，`ExperimentResult.save()` 就会失败。

## metadata.json

`metadata.json` 保存 run-level 元数据：

```text
metadata.json
├─ experiment_id / start_time / duration
├─ status
├─ device
└─ contract
   ├─ experiment_plan
   ├─ output_plan
   ├─ runtime_resource_plan
   ├─ algorithm_config_trace
   └─ runtime_extension_report
```

`contract.runtime_resource_plan` 是 metadata-only 资源计划。它记录 batch plan、memory model、seed policy 和 dtype/compile 摘要，但不表示自动重分批已经实现。

## output_contract.json

`output_contract.json` 是保存 plot/export 之前的依赖检查结果：

```text
output_contract.json
├─ requested_specs
├─ required_metrics
├─ required_artifacts
├─ available_metrics
├─ available_artifacts
├─ missing_metrics
├─ missing_artifacts
└─ is_valid
```

如果 custom plot 要的 metric 不存在，或者 heatmap 要的 `overlap_matrix` 不存在，保存阶段会报错，不再静默画 0 或跳过图。

## artifacts/results.pt

`artifacts/results.pt` 是可选 tensor payload。当前规则：

- matrix algorithm 有真实 `W_students/X_students` 时才保存这些 factor。
- tensor metrics-only algorithm 不再保存 dummy zero `W/X`。
- `factor_payload_contract` 必须同步写入 `.pt`，让本地分析脚本知道哪些字段不可用。

## 正式结果、诊断和展示

```text
formal metric
  进入 MetricSpec，并在 metrics.json 中有 semantic metadata。

diagnostic
  也可以进入 MetricSpec，但 result_role/order_parameter_status 会标明不能直接当物理 order parameter。

plotting helper
  只用于画图或选择 artifact；必须声明，不允许和 formal metric 混淆。

display snapshot
  results/latest/ 的轻量展示结果，不作为分析输入。
```

## 当前仍保留的兼容路径

- `ResultStorage.save()` 仍支持旧 dict output schema。
- `export/bundler.py` 仍支持 standalone export workflow。
- 旧 flat metric key 继续保留；新的语义解释依赖 `metric_schema` 和 algorithm context。

