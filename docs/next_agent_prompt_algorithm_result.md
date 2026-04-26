# Next Agent Prompt: Native Matrix AlgorithmResult

下面这段可以直接复制给新 Codex 对话。它选择的是一个偏程序/工程、低物理风险的大分支：把 matrix algorithms 从 legacy tuple adapter 迁移到原生 `AlgorithmResult`，继续收紧 active path。

```text
你现在接手 Matrix_Factorization，工作分支是 dev。请先阅读：

- AGENTS.md
- docs/project_handoff_2026-04-27.md
- docs/implementation_progress.md
- docs/hard_interface_framework.md
- docs/result_schema_contract.md
- docs/algorithm_integration_contract.md

当前项目已经完成 hard-interface、Projection metric v3、canonical scan、ResultCube/PlotQuery、scan-aware memory calibration、AGD step-level state_slice probe。请不要重新做这些已经完成的工作。

本任务目标：

把 active matrix algorithms 从 “legacy train_batch_alphas tuple + base adapter” 迁移到原生 `AlgorithmResult`，但不改变任何 AGD / dense BiGAMP / spreading BiGAMP 的数值公式、训练步骤、metric 定义、normalization、seed 语义、damping 或 Onsager 行为。

背景：

- Tensor serial/parallel 已经显式返回 metrics-only `AlgorithmResult`。
- Matrix algorithms 目前仍主要依赖 `AlgorithmBase.train_batch_result()` 默认实现，把 legacy `(W_students, X_students)` tuple 包装成 `AlgorithmResult`。
- 这已经比旧 path 硬，但 active result metadata 仍会出现 `legacy_matrix_adapter` 或 `legacy_matrix_runner_adapter`，说明 matrix algorithms 还不是原生 result path。
- 本任务只把 result container 原生化，不改 metric 公式，不把 metric compute 移进 algorithm。

必须遵守：

1. 不改变 AGD / BiGAMP / spreading 的训练数学。
2. 不改变 projection metric v3 的定义。
3. 不删除 legacy `train_batch_alphas()`，它仍作为旧调用兼容入口。
4. 不重新引入 tensor dummy `W/X`。
5. 不从 private `_batch_metrics` 给 active matrix route 补 metric。
6. 任何新增参数、metric、output、probe、source 文件必须先有 contract/spec/source inventory。
7. 不提交 runs/results/artifacts/checkpoint/.pt/大图。

建议实施步骤：

1. Baseline 检查
   - 运行：
     - `git status --short --branch`
     - `python -m pytest -q tests/test_result_schema.py tests/test_scan_runner.py`
     - `mf validate src/matrix_factorization/config.yaml`
   - 阅读：
     - `src/matrix_factorization/modules/algorithms/base.py`
     - `src/matrix_factorization/core/experiment/runner.py`
     - `src/matrix_factorization/modules/algorithms/agd.py`
     - `src/matrix_factorization/modules/algorithms/bigamp/standard.py`
     - `src/matrix_factorization/modules/algorithms/bigamp/spreading.py`

2. 为 AGD 添加原生 `train_batch_result()`
   - 在 `AGDAlgorithm` 中覆写 `train_batch_result()`。
   - 内部仍调用现有 `train_batch_alphas()`，不改 loop。
   - 返回 `AlgorithmResult`：
     - `matrix_factors={"W_students": W_students, "X_students": X_students}`
     - `metrics_by_alpha={}`，让 runner 继续用现有 metric compute path。
     - metadata 至少包含：
       - `algorithm_key`
       - `result_kind="matrix_factors"`
       - `result_contract`
       - `result_source="native_matrix_algorithm_result"`
       - `matrix_factors_available=True`
       - dtype/compile/tf32/seed policy 的轻量 execution metadata，如果该 algorithm 已有对应字段。
   - 保留 legacy `train_batch_alphas()` 给旧调用方。

3. 为 dense BiGAMP 添加原生 `train_batch_result()`
   - 路径通常是 `src/matrix_factorization/modules/algorithms/bigamp/standard.py`。
   - 不改 `train_batch_alphas()` 数值逻辑。
   - metadata 写清楚 compile fallback、TF32、seed policy、result source。
   - `metrics_by_alpha` 可以为空，由 runner 统一计算 matrix metrics。

4. 为 BiGAMP spreading 添加原生 `train_batch_result()`
   - 路径通常是 `src/matrix_factorization/modules/algorithms/bigamp/spreading.py`。
   - 不改 graph/F/Y 构造、chunking、adaptive damping、Onsager 或 restart 数值逻辑。
   - metadata 必须保留现有 spreading execution metadata：
     - chunk_size/chunk_policy/dynamic_batches
     - dtype/compile/tf32 status
     - seed_partition policy
   - `metrics_by_alpha` 若算法已有正式 per-alpha spreading metrics 可以保留；否则由 runner 计算。

5. 收紧 runner active path 测试
   - active registered matrix algorithms 通过 `runner._run_algorithm_result()` 后，metadata 不应再是：
     - `legacy_matrix_adapter`
     - `legacy_matrix_runner_adapter`
   - legacy fallback 可以保留，但只用于测试 fixture 或非 migrated algorithm。
   - runner 仍应支持旧算法 object 没有 native override 的兼容路径，但 active `agd/bigamp/bigamp_spreading` 不走它。

6. 测试要求
   - 新增/更新 tests：
     - `tests/test_result_schema.py`
     - `tests/test_scan_runner.py`
     - 如有必要新增 `tests/test_algorithm_result_native_matrix.py`
   - 覆盖：
     - AGD native result returns `AlgorithmResult` with real matrix_factors。
     - dense BiGAMP native result returns `AlgorithmResult` with real matrix_factors。
     - spreading native result returns `AlgorithmResult` with real matrix_factors and execution metadata。
     - active runner path for these algorithms reports `result_source="native_matrix_algorithm_result"` 或更具体的 native source。
     - metrics 数值与旧 `train_batch_alphas()` + runner metric compute path 一致。
     - legacy tuple API 仍可调用。

7. Trial / validation
   - 必须运行：
     - `python -m pytest -q tests/test_result_schema.py tests/test_scan_runner.py`
     - `mf validate src/matrix_factorization/config.yaml`
     - `mf trial run matrix_bigamp_quick`
     - `mf trial run scan_alpha_quick`
   - 最后运行：
     - `python -m pytest -q`
     - `git diff --check`

8. 文档
   - 更新：
     - `docs/implementation_progress.md`
     - `docs/project_handoff_2026-04-27.md`
     - `docs/algorithm_integration_contract.md`（如果该文档列出了 result path 状态）
   - 明确写：
     - matrix algorithms 已从 legacy adapter 迁移为 native AlgorithmResult。
     - legacy `train_batch_alphas()` 保留为兼容 API。
     - 本任务没有改变训练公式和 metric 定义。

验收标准：

- `agd/bigamp/bigamp_spreading` active path 不再报告 legacy matrix adapter result source。
- Tensor metrics-only path 不回退、不受影响。
- Projection metric v3 测试保持通过。
- Quick trials 能真实运行。
- 全量 pytest 通过。
- 工作树只包含源码、测试、文档改动，不包含 runs/results/artifacts。
- 完成后 commit/push 到 `dev`。

非目标：

- 不做 MetricSpec compute adapter 全迁移。
- 不做 OutputPlan executor 拆分。
- 不做 sample/student folding。
- 不做 OOM 同进程 retry。
- 不做 tensor serial/parallel 合并。
- 不做 Metropolis-like intervention。

如果你发现某一步必须改变物理公式或 metric 定义才能继续，停止该子项，只记录证据和 limitation，不要自行改物理。
```

## 用户需要做什么

这个任务本身偏工程，通常不需要用户先做物理判断。用户只需要确认：

- 是否同意下一轮优先做这个分支，而不是 sample/student folding、OutputPlan executor 或 tensor parity。
- 如果新 agent 发现某个 algorithm 的 native result metadata 应该用更具体名字，例如 `native_agd_algorithm_result`、`native_bigamp_algorithm_result`，是否接受这种更细命名。默认可以接受。

不建议用户在这个任务中手动改正式 `src/matrix_factorization/config.yaml`。需要试跑时继续使用 `mf trial ...`。
