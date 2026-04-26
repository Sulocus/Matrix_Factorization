# Next Agent Prompt: OutputPlan Executor Extraction

下面这段可以直接复制给新 Codex 对话。它选择的是一个偏程序/工程、低物理风险的大分支：把 output/plot/export 执行从 `ExperimentResult.save()` 中拆出，形成独立 `OutputPlanExecutor`。

```text
你现在接手 Matrix_Factorization，工作分支是 dev。请先阅读：

- AGENTS.md
- docs/project_handoff_2026-04-27.md
- docs/implementation_progress.md
- docs/hard_interface_framework.md
- docs/result_schema_contract.md
- docs/scan_system_contract.md

当前项目已经完成 hard-interface、Projection metric v3、canonical scan、ResultCube/PlotQuery、scan-aware memory calibration、AGD step-level state_slice probe，以及 active matrix algorithms 的 native AlgorithmResult。请不要重新做这些已经完成的工作。

本任务目标：

把 output/plot/export 执行从 `ExperimentResult.save()` 中拆成独立、可测试的 `OutputPlanExecutor`，继续硬化 output 层。不要改变任何算法公式、metric 数值定义、normalization、damping、Onsager、seed policy 或 scan 语义。

背景：

- `OutputSpec` 和 `OutputPlan` 已经能声明 output 需要哪些 metric/artifact。
- `ExperimentResult.save()` 已经会做 output contract preflight，并能 hard error。
- 但实际保存、default scalar plots、PlotQuery plots、heatmap/GIF、manifest 写出等逻辑仍集中在 `ExperimentResult.save()` 中。
- 这导致 output 层职责偏重，新增 output backend 时仍容易把 validation、render、artifact policy 混在一起。

必须遵守：

1. 不改 AGD / BigAMP / spreading / tensor 的训练逻辑。
2. 不改 Projection metric v3 的任何公式或 key 语义。
3. 不改 ResultCube 的存储 schema，除非只增加 backward-compatible metadata。
4. 不提交 `runs/ / results/ / artifacts/ / .pt / checkpoint / 大图`。
5. 新 output handler 或 source file 必须更新 `OutputSpec` 和 source inventory。
6. 缺 metric/artifact 必须继续 hard error，不能回到 silent fallback 或画 0。

建议实施步骤：

1. Baseline
   - 运行：
     - `git status --short --branch`
     - `python -m pytest -q tests/test_result_schema.py tests/test_plot_query.py tests/test_contract_metric_output_specs.py`
     - `mf validate src/matrix_factorization/config.yaml`
   - 阅读：
     - `src/matrix_factorization/core/experiment/result.py`
     - `src/matrix_factorization/modules/outputs/spec_adapter.py`
     - `src/matrix_factorization/core/planning.py`
     - `tests/test_result_schema.py`
     - `tests/test_plot_query.py`

2. 新增 executor 模块
   - 建议新增：
     - `src/matrix_factorization/modules/outputs/executor.py`
   - 定义：
     - `OutputExecutionContext`
     - `OutputExecutionReport`
     - `OutputPlanExecutor`
   - executor 输入：
     - `ExperimentResult`
     - `output_options`
     - `run_dir`
     - 已经 preflight 过的 output contract check
   - executor 输出：
     - generated files list
     - skipped outputs with reasons
     - errors
     - lightweight metadata/report

3. 从 `ExperimentResult.save()` 中迁移职责
   - 保留 `ExperimentResult.save()` 作为 public entrypoint。
   - 让它负责：
     - 写基础 JSON / tensor artifact。
     - 调用 output preflight。
     - 调用 `OutputPlanExecutor`.
     - 写 `output_execution_report.json` 或把 report 加进 manifest。
   - 把以下逻辑迁到 executor 私有方法：
     - default scalar curves。
     - PlotQuery rendering。
     - legacy custom curves。
     - heatmap/GIF。
   - 不要改变现有输出文件名，除非测试和 docs 一起更新。

4. Output dependency hardening
   - executor 不能自己猜 metric。
   - default scalar plot 仍只能画真实存在的 metric。
   - PlotQuery 必须继续走 `ResultCube.resolve_plot_query()`。
   - heatmap 只有 `overlap_matrix` 或 real matrix factors 可用时才生成。
   - plotting backend 失败仍必须 hard error。

5. Tests
   - 更新/新增：
     - `tests/test_result_schema.py`
     - `tests/test_plot_query.py`
     - `tests/test_contract_metric_output_specs.py`
     - 可新增 `tests/test_output_plan_executor.py`
   - 覆盖：
     - save 调用 executor 并写 report。
     - default scalar curves 文件名保持不变。
     - PlotQuery 图仍生成。
     - missing metric hard error。
     - heatmap backend failure hard error。
     - output_execution_report 记录 generated files。

6. Trials / validation
   - 必须运行：
     - `python -m pytest -q tests/test_result_schema.py tests/test_plot_query.py tests/test_contract_metric_output_specs.py`
     - `mf validate src/matrix_factorization/config.yaml`
     - `mf trial run matrix_bigamp_quick`
     - `mf trial run scan_mixed_axes_quick`
   - 最后运行：
     - `python -m pytest -q`
     - `git diff --check`

7. Docs
   - 更新：
     - `docs/implementation_progress.md`
     - `docs/project_handoff_2026-04-27.md`
     - `docs/result_schema_contract.md`（如果新增 `output_execution_report.json`）
     - `docs/hard_interface_framework.md`
   - 明确写：
     - output execution 已从 `ExperimentResult.save()` 拆出。
     - 文件 schema 和旧 plots 文件名保持兼容。
     - 这不改变算法/metric 物理定义。

验收标准：

- `ExperimentResult.save()` 不再直接承载大段 plotting/rendering 逻辑。
- OutputPlanExecutor 有独立 tests。
- 现有 run directory schema 兼容。
- `matrix_bigamp_quick` 和 `scan_mixed_axes_quick` 仍能真实运行并生成图。
- 全量 pytest 通过。
- 工作树只包含源码、测试、文档改动，不包含 generated artifact。
- 完成后 commit/push 到 `dev`。

非目标：

- 不做 MetricSpec compute adapter 全迁移。
- 不做 sample/student folding。
- 不做 OOM 同进程 retry。
- 不做 tensor serial/parallel 合并。
- 不做 Metropolis-like intervention。

如果你发现某一步必须改变 metric 数值或 algorithm 行为才能继续，停止该子项，只记录 evidence 和 limitation，不要自行改物理。
```

## 用户需要做什么

这个任务偏工程，通常不需要用户先做物理判断。用户只需要确认是否同意下一轮优先做 output executor，而不是 runtime hooks、sample folding 或 tensor parity。
