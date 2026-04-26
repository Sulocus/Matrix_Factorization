# Project Handoff 2026-04-27

这份文档是给下一轮 Codex 对话的接手说明。它记录本轮从项目重新接管开始，到 hard-interface、trial、canonical scan、resource planning、projection metric 迁移为止，已经做了什么、没有做什么、哪些只是建了框架、哪些已经进入主链路，以及下一轮应该从哪里开始。

当前工作分支：`dev`。

当前重要状态：

- 最近已提交内容覆盖 hard-interface、trial、canonical scan、resource/memory planning 等结构性改造。
- 当前工作树仍有一组未提交修改，核心是 `Projection Metric 体系重构 v2`。
- 当前未提交修改已经通过：
  - `python -m pytest -q` -> `279 passed, 9 skipped`
  - metric/schema 定向测试 -> `73 passed`
  - `mf validate src/matrix_factorization/config.yaml` -> no errors，只有 parsed-only / inactive-route warnings
  - `mf trial run matrix_bigamp_quick` -> 成功生成 schema v3 trial result
- 当前未提交 quick trial 最新检查显示：
  - `metrics.json.schema_version = 3`
  - `metrics.json.metric_schema.schema_version = 3`
  - formal metrics 不再包含 `MSE`
  - formal metrics 包含 `Q_Y_*`、`Q_W/Q_X`、`Q_W_GRAM_ROOT/Q_X_GRAM_ROOT`

## 1. 接管目标和基本原则

项目最初的问题不是单个公式或单个 bug，而是旧研究代码逐步叠加后出现的软接口问题：

- YAML 写了参数，但不确定是否传到 algorithm。
- Algorithm 注册了 class，但不一定声明自己需要什么 data、能产出什么 metric。
- Plot/output 可能假设某个 metric 存在，实际没有时静默跳过或画错。
- Tensor / matrix / spreading / replica 的 metric 名称混杂，不同地方可能同名异义。
- Scan 模式有多个历史入口，`alpha_scan / steps_scan / nested_scan / hysteresis_scan` 彼此绕开。
- 试跑、debug、临时脚本和结果容易散落在根目录或 tests 里。
- 并行和显存估算以前更像辅助注释，不是真正由主链路消费的计划对象。

本轮核心目标是把项目改成 hard-interface 研究框架：

- 新参数、新 algorithm、新 metric、新 output、新 intervention、新 trial 必须先声明 contract。
- 配置如果只接了一半，应该在 import、validate、preflight 或 test 阶段失败。
- 不改变现有物理公式时，只做结构、metadata、validation、schema、测试。
- 涉及物理含义的改动必须显式记录，并配套 schema/version/doc/test。

## 2. 已完成的全局扫描和文档化

早期先做了项目扫描，明确了这些目录角色：

- `src/matrix_factorization/`：当前 active Python package。
- `src/matrix_factorization/core/`：config、planning、contracts、runner、result、resource planning 等主链路。
- `src/matrix_factorization/modules/algorithms/`：AGD、BiGAMP、spreading、tensor、replica/combined 等算法实现。
- `src/matrix_factorization/modules/metrics/`：overlap、spreading、tensor、contract metric payload。
- `src/matrix_factorization/modules/outputs/`：plot、comparison、combined output。
- `src/matrix_factorization/modules/teachers/`：teacher/data generation。
- `trials/`：受控 research trial，不是 pytest，也不是正式实验。
- `runs/ / results/ / artifacts/`：ignored artifact workspace，不提交大结果。
- `docs/`：当前 hard-interface、scan、metric、result schema、legacy、local workflow 等说明文档。

相关文档：

- `docs/current_project_map.md`
- `docs/hard_interface_framework.md`
- `docs/result_schema_contract.md`
- `docs/algorithm_integration_contract.md`
- `docs/legacy_debug_inventory.md`
- `docs/local_workflow.md`

## 3. Hard-interface 框架已完成的内容

核心文件：

- `src/matrix_factorization/core/contracts.py`
- `src/matrix_factorization/core/planning.py`
- `src/matrix_factorization/core/validation.py`
- `src/matrix_factorization/core/config.py`
- `src/matrix_factorization/cli.py`

已经建立的 contract 类型：

- `ParameterSpec`
- `AlgorithmSpec`
- `TeacherSpec`
- `GraphSpec`
- `MetricSpec`
- `OutputSpec`
- `InterventionSpec`
- `ProbeSpec`
- `AnalyzerSpec`
- `SourceInventorySpec`
- `TrialSpec`
- `ResourceSpec`
- `BatchingSpec`
- `MemoryModelSpec`
- `SeedPolicySpec`

已经落地的硬规则：

- 新 algorithm 必须有 `AlgorithmSpec`，否则注册/import 失败。
- 新 teacher 必须有 `TeacherSpec`。
- 新 graph 必须有 `GraphSpec`。
- 新 metric handler 必须有 `MetricSpec`。
- 新 output handler 必须有 `OutputSpec`。
- 新 YAML-facing 参数必须有 `ParameterSpec`。
- unknown YAML 字段在 validate 阶段报错。
- active 参数在当前路由不生效时，普通 validate warning，strict mode error。
- `mf validate --json` 输出机器可读 issue code。
- `parameter_chain` 记录 YAML path、effective value、consumer、active/inactive route、derived effect。
- `ExperimentPlan` 记录实际 algorithm route、warnings/errors、resource plan、metric/output/intervention/probe/analyzer contract。

主要命令：

```bash
mf validate path/to/config.yaml
mf validate --strict path/to/config.yaml
mf validate --json path/to/config.yaml
mf explain-config path/to/config.yaml
```

当前仍存在的已知 warning：

- `output.name` 是 `parsed_only`。
- `output.storage_mode` 是 `parsed_only`。
- 默认 config 中一些 AGD/spreading 参数在当前 tensor route 下 inactive。
- `tensor_order=3` 会路由到 `bigamp_tensor_parallel`，不是单纯由 YAML `algorithm` 决定。

这些 warning 不是当前失败，而是 hard-interface 正确暴露的“字段存在但当前路由不消费”。

## 4. AlgorithmResult 和 result schema 已完成的内容

核心文件：

- `src/matrix_factorization/core/contracts.py`
- `src/matrix_factorization/core/experiment/runner.py`
- `src/matrix_factorization/core/experiment/result.py`
- `src/matrix_factorization/modules/algorithms/bigamp/tensor_contract.py`

已经完成：

- `AlgorithmResult` 成为 runner 内部标准算法输出对象。
- Matrix algorithm 仍可通过 legacy adapter 包装 `(W_all, X_all)`。
- Tensor metrics-only path 不再保存 dummy zero `W/X` 当真实 factor。
- `ExperimentResult.save()` 写出：
  - `config.json`
  - `metadata.json`
  - `metrics.json`
  - `output_contract.json`
  - `events.jsonl`
  - `manifest.json`
  - optional plots/artifacts
- `metrics.json` 写入 metric schema、metric contract、result cube、flat legacy compatibility keys。
- `results/latest` 明确只是 display schema，不作为正式分析输入。

仍未完成：

- Matrix algorithms 还没有完全改成原生返回 `AlgorithmResult`；目前仍有 adapter。
- Runner 内部仍有部分 legacy fallback 代码路径，但 active path 已被 contract/test 覆盖。
- Tensor serial/parallel 的 result contract 已拉齐一部分，但训练 loop 和物理 parity 还没有合并。

## 5. Research Trial 工作流已完成的内容

核心目录：

- `trials/`
- `trials/registry.yaml`
- `trials/active/*/trial.yaml`
- `trials/active/*/config.yaml`

核心命令：

```bash
mf trial list
mf trial explain <trial_key>
mf trial validate <trial_key>
mf trial run <trial_key>
```

已经完成：

- Trial 是受控试跑入口，不是旧式轻量预检，也不是 pytest。
- Trial config 固定放在 `trials/active/<trial_key>/config.yaml`。
- 用户说“试跑 / 调试参数 / quick trial / 看一下能不能跑”时，agent 应改 trial config，不改正式默认 config。
- `mf trial run` 当前只自动运行 `runtime_class: quick`。
- Trial 输出写入 ignored workspace，例如 `runs/trials/<trial_key>/...`。
- `mf trial run` 不刷新 `results/latest`，避免污染正式展示结果。
- Trial manifest 检查 config、expected metrics、artifact policy、registry consistency。

当前 active quick trials：

- `matrix_bigamp_quick`
- `scan_alpha_quick`
- `scan_steps_quick`
- `scan_size_quick`
- `scan_init_quick`
- `scan_mixed_axes_quick`

仍未完成：

- Trial 生命周期可以登记 `draft/active/promoted/archived/abandoned`，但 promotion 流程还主要靠文档，不是完整 CLI 自动化。
- Medium/gpu-heavy trial 只建议登记和 explain/validate；自动 run 策略还没做。
- Trial summary 自动汇总和 artifact manifest 还可以继续强化。

## 6. Canonical Scan 系统已完成的内容

核心文件：

- `src/matrix_factorization/core/scan.py`
- `src/matrix_factorization/core/scan_schema.py`
- `src/matrix_factorization/core/scan_plotting.py`
- `src/matrix_factorization/core/experiment/runner.py`
- `docs/scan_system_contract.md`

新 scan 唯一入口：

```yaml
scan:
  axes:
    alpha:
      path: alpha
      values: {start: 0.0, stop: 1.0, step: 0.1}
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

已经完成：

- 主链路不再接受旧 `scan_mode / alpha_scan / steps_scan / nested_scan / hysteresis_scan`。
- `alpha / max_steps / size / init / damping / onsager` 都是普通 axis，不再是四套特殊模式。
- Composite axis 用于 size、cold/warm start 等多参数联动。
- `ScanPlan` 展开 `ScanPoint`，每个点有 coordinates、overrides、effective config hash。
- `ResultCube` 成为 multi-axis scan 正式 schema。
- alpha-only 仍导出旧 flat metrics 兼容字段。
- PlotQuery 可以基于 `ResultCube` 的 coordinate 做 where、series_by、compare。

已经验证：

- alpha-only scan quick trial。
- steps scan quick trial。
- size scan quick trial。
- init scan quick trial。
- mixed axes quick trial。

仍未完成：

- PlotQuery 的复杂研究图还没有做到完全“任意选两组曲线都能自动画”的成熟 UI/CLI 体验。
- 多 size / 多 init / 多 damping 的大规模 result browsing 还需要更多真实实验结果验证。
- 旧 scan 配置的 migration 命令如果存在，也只是辅助；主链路不兼容旧字段。

## 7. Resource planning / 并行 / 显存估算已完成的内容

核心文件：

- `src/matrix_factorization/core/parallel.py`
- `src/matrix_factorization/core/resource_planning.py`
- `src/matrix_factorization/core/memory.py`
- `src/matrix_factorization/core/calibration.py`
- `docs/parallel_memory_contract.md`
- `docs/parallel_memory_review_queue.md`

已经完成：

- `ResourceExecutionPlan` 接入 `ExperimentPlan` 和 run metadata。
- Resource plan 会记录 groups、batches、work items、batch axes、memory estimate、seed policy。
- Canonical scan 下，planner 按 non-alpha group 分组。
- 当前真实自动折叠维度只承认 `alpha`。
- `size / init / damping / onsager / max_steps / arbitrary parameter axis` 默认不跨组折叠。
- `sample/student folding` 继续禁用，直到 runner 和 algorithm 真正消费 `sample_range/sample_offset`。
- `SeedPolicySpec` 记录 legacy vs partition-invariant。
- `partition_invariant` 是 opt-in，不改变默认 legacy 数值。
- OOM replan gate 已存在：只有 partition-invariant 且有 provenance 的 plan 可以生成更保守计划。
- Runner 当前 OOM 策略仍是 checkpoint/flush/exit，不自动同进程 retry。
- Memory estimator 已有 stage/breakdown metadata，并接入 tests。
- Calibration workflow 已有 CLI 和 profile。

已经完成的本地 calibration 提交：

- matrix bigamp target profiles
- matrix AGD target profiles
- spreading bigamp target profiles
- tensor serial target profiles
- tensor parallel target profiles

重要澄清：

- 当前 calibration 已经跑过 GB 级 profile，但之前报告里“误差在 10% 内”的描述曾过度乐观。
- 后续不能只看 safety factor 或 conservative estimate；如果要求公式本身误差 <= 10%，需要继续修 estimator formula 并重复 calibration。
- 当前 smart parallel 的执行层还不是“所有算法所有 scan 组合都已最优并行”。它是 contract-driven resource planner + conservative alpha folding + metadata/calibration foundation。

仍未完成：

- per-algorithm memory formula 真正修到 `raw formula vs torch.cuda.max_memory_allocated <= 10%`。
- 对所有 active algorithm 的 10GB/16GB 或 6GB/10GB profile 形成稳定系数文件。
- Tensor parallel 的 stage peak 模型仍需要继续修正。
- AGD 的 live tensor / autograd / retention 模型仍需要继续修正。
- Spreading 的 graph/F/Y/gather/scatter/chunk stage peak 仍需要继续修正。
- Device used delta 与 allocator reserved/fragmentation 还需要更系统记录。
- 自动 replan 还没有进入同进程 retry。

## 8. Projection Metric 体系重构已完成的内容

这是当前工作树中尚未提交的一大组修改。

核心目标：

- 把 `Q_Y cosine / physical_overlap / MSE / Gram` 混杂体系改成 projection-first 物理指标体系。
- `Q_Y`：measurement/output 的 absolute projection overlap。
- `Q_W/Q_X`：matrix latent factor 的 coordinate projection overlap。
- `Q_N`：tensor latent node/spin/factor 的 coordinate projection overlap，对齐 matrix 的 `Q_W/Q_X`。
- `Q_W_GRAM_ROOT/Q_X_GRAM_ROOT`：baseline-corrected Gram overlap 开根号后的 diagnostic，用来解决 rotation/gauge 导致 projection 不可读的问题。
- `MSE / Gen_Error / Q_Y_COS` 不再作为 formal metric。

统一公式：

```text
projection_abs(student, teacher)
  = abs(sum student * teacher) / sum teacher^2
```

规则：

- 不做 cosine normalization。
- 不 clip，大于 1 保留。
- `mean/std` 只是 sample/replica statistics suffix，不是物理量本身。
- teacher norm 过小时返回 0，并在后续 metadata 中应继续标记 degenerate teacher norm。

已改核心文件：

- `src/matrix_factorization/modules/metrics/overlap.py`
- `src/matrix_factorization/modules/metrics/contract_compute.py`
- `src/matrix_factorization/modules/metrics/spreading.py`
- `src/matrix_factorization/modules/metrics/tensor_metrics.py`
- `src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading.py`
- `src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading_parallel.py`
- `src/matrix_factorization/modules/algorithms/bigamp/tensor_contract.py`
- `src/matrix_factorization/core/contracts.py`
- `src/matrix_factorization/core/experiment/result.py`
- `src/matrix_factorization/modules/outputs/plot_registry.py`
- `src/matrix_factorization/modules/outputs/plotting.py`
- `src/matrix_factorization/modules/outputs/comparison.py`
- `src/matrix_factorization/modules/outputs/combined.py`
- `src/matrix_factorization/modules/metrics/combined.py`
- `src/matrix_factorization/core/config.py`
- `src/matrix_factorization/config.yaml`
- `src/matrix_factorization/ui/wizard.py`

Matrix 新 formal metrics：

- `Q_Y_mean / Q_Y_std`
- `Q_Y_observed_mean / Q_Y_observed_std`
- `Q_Y_unobserved_mean / Q_Y_unobserved_std`
- `Q_W_mean / Q_W_std`
- `Q_X_mean / Q_X_std`
- `Q_W_GRAM_ROOT_mean / Q_W_GRAM_ROOT_std`
- `Q_X_GRAM_ROOT_mean / Q_X_GRAM_ROOT_std`

Spreading 新 formal metrics：

- `Q_Y_mean / Q_Y_std`
- `Q_Y_observed_mean / Q_Y_observed_std`
- `Q_Y_unobserved_mean / Q_Y_unobserved_std`
- `Q_W_mean / Q_W_std`
- `Q_X_mean / Q_X_std`
- `Q_W_GRAM_ROOT_mean / Q_W_GRAM_ROOT_std`
- `Q_X_GRAM_ROOT_mean / Q_X_GRAM_ROOT_std`

Tensor 新 formal metrics：

- `Q_Y_mean / Q_Y_std`
- `Q_Y_observed_mean / Q_Y_observed_std`
- `Q_Y_unobserved_mean / Q_Y_unobserved_std`
- `Q_N_mean / Q_N_std`
- `Q_N_mode0_mean / Q_N_mode0_std`
- `Q_N_mode1_mean / Q_N_mode1_std`
- 后续 mode 数随 tensor order 扩展

Schema/version：

- `metrics.json.schema_version = 3`
- `metric_schema.schema_version = 3`
- 新旧 run 的 `Q_Y_mean` 不能直接比较。
- 旧 schema 中的 `Q_Y_mean` 应解释为 legacy cosine 或 legacy reconstruction proxy。
- 新 schema 中的 `Q_Y_mean` 是 absolute projection。

已改文档：

- `docs/METRICS_GUIDE.md`
- `docs/metrics_semantics.md`
- `docs/metric_naming_decisions.md`
- `docs/semantic_review_queue.md`
- `docs/tensor_parity_contract.md`

已改 tests/trials：

- `tests/test_contract_metric_output_specs.py`
- `tests/test_result_schema.py`
- `tests/test_result_cube.py`
- `tests/test_parallel_memory_contract.py`
- `tests/test_cli_plan_commands.py`
- `tests/test_tensor_parity_contract.py`
- `tests/test_trial_contract.py`
- `trials/active/*/trial.yaml`

已验证：

```bash
python -m pytest -q
# 279 passed, 9 skipped

python -m pytest -q tests/test_contract_metric_output_specs.py tests/test_result_schema.py tests/test_result_cube.py tests/test_trial_contract.py tests/test_tensor_parity_contract.py
# 73 passed

mf validate src/matrix_factorization/config.yaml
# errors: none

mf trial validate matrix_bigamp_quick
# errors: none

mf trial run matrix_bigamp_quick
# success
```

已确认最新 quick trial：

- formal metric keys 不包含 `MSE`。
- formal metric keys 包含 `Q_Y_*`。
- formal metric keys 包含 `Q_W_GRAM_ROOT/Q_X_GRAM_ROOT`。

仍需注意：

- `compute_mse_*` 或 internal MSE/loss helper 仍可保留为 debug/internal loss，不是 formal metric。
- Experimental/unintegrated algorithm 如 `agd_tensor.py` 可能仍有 legacy `MSE` result，需要单独决定是否清理。
- Legacy docs 中仍会出现 `MSE / physical_overlap / Q_W_prime` 等词，用于解释旧结果，不代表新 formal schema。

## 9. 目前明确没有完成或不能声称完成的内容

下面这些不能在新对话里误认为已经完成：

### 9.1 物理正确性没有被“自动证明”

Hard-interface 只能保证：

- 参数有没有被声明。
- 参数有没有进入 effective plan。
- algorithm 声明了什么输入输出。
- metric/output 是否声明并通过 preflight。
- result 是否按 schema 写出。

它不能自动证明：

- AMP 方程正确。
- Onsager 项正确。
- damping 物理语义正确。
- teacher/student scale 正确。
- tensor serial/parallel 物理等价。
- projection metric 一定是最终物理选择。

这些仍需要本地实验和理论检查。

### 9.2 Tensor serial/parallel 尚未合并

已完成：

- 两者有 AlgorithmSpec。
- parity contract 文档存在。
- metrics/result schema 在 projection 方向上靠近。
- tensor parallel 已有 QN/QY projection formal metric。

未完成：

- 训练 loop 合并。
- teacher scale 和 alpha normalization 完全对齐。
- Graph/hypergraph/F distribution 完全对齐。
- Onsager handling 完全对齐。
- Serial/parallel 数值 parity。

### 9.3 Intelligent parallel 还不是最终智能调度系统

已完成：

- scan-aware resource planning foundation。
- alpha folding conservative plan。
- seed policy gate。
- calibration CLI/profile。
- metadata/provenance。

未完成：

- 公式级显存估算修到 <= 10%。
- 所有算法/scan 组合的 GB 级 exhaustive validation。
- sample/student folding。
- 同进程 OOM retry/replan。
- chunk size auto tuning。

### 9.4 Runtime probe/intervention 还只是框架和部分 hook

已完成：

- `ProbeSpec / AnalyzerSpec / InterventionSpec`。
- 高层 runner-level `batch_summary` probe。
- declared-only probe 如果被请求会 preflight error。

未完成：

- Algorithm step 内部 `AlgorithmStateView` 全面暴露。
- `before_step/after_step/on_plateau` 等 hook 真正进入所有 algorithm loop。
- Metropolis-like kick、adaptive restart、warm/cold start 完全迁移成 intervention executor。

### 9.5 Result browsing / plotting 还不是最终研究 UI

已完成：

- ResultCube。
- PlotQuery schema。
- output contract。
- default plots 更新为 projection-first。

未完成：

- 面向研究使用的高级 plot query CLI/UI。
- 任意两组参数曲线对比的完整用户体验。
- 多 run/multi trial result browser。

## 10. 如果新开对话，推荐接手顺序

### 第一步：确认当前未提交 projection metric 修改

建议新对话第一件事运行：

```bash
git status --short
python -m pytest -q
mf validate src/matrix_factorization/config.yaml
mf trial run matrix_bigamp_quick
```

如果仍通过，建议 commit：

```bash
git add docs src tests trials
git commit -m "Migrate formal metrics to projection schema"
git push
```

### 第二步：决定是否继续清理 legacy metric surface

可做的低风险清理：

- 把 remaining formal output 中的 legacy metric name 全部降级为 debug/internal。
- 确认 `MSE` 不再出现在 active formal MetricSpec。
- 确认 `Q_W_prime/Q_X_prime` 只作为 legacy/raw debug，不进入 default plot。
- 给 `compute_mse_*` 加 legacy/internal 注释。

不要直接删：

- internal loss。
- old result loader compatibility。
- legacy docs。
- experimental algorithm 的历史输出，除非单独审查。

### 第三步：补 projection metric fixture

建议新增或扩展测试：

- projection helper perfect/sign flip/scale > 1/no clip。
- matrix 手算 full/observed/unobserved QY。
- matrix 手算 QW/QX projection。
- Gram-root 与 projection 分离。
- spreading perfect teacher observed F graph QY = 1。
- tensor perfect observed hyperedge QY = 1。
- tensor QN mode/aggregate 手算。

### 第四步：继续 intelligent parallel 真正公式修正

如果下一轮目标是并行/显存，建议不要只改 docs 或 metadata，而是按算法逐个做：

1. `bigamp`：作为 baseline，确认 estimator formula。
2. `agd`：补 live tensors、autograd temps、old+new W/X、retention。
3. `bigamp_spreading`：拆 graph/F/Y、gather/scatter/chunk/backtracking stage。
4. `bigamp_tensor`：serial 单 alpha/sample peak。
5. `bigamp_tensor_parallel`：TensorSuperGraph path stage peak。

每个算法至少需要：

- 理论 stage breakdown。
- GB 级 calibration。
- `raw formula vs max_memory_allocated` error。
- 如果误差 > 10%，记录在 `docs/parallel_memory_review_queue.md`，不要声称完成。

### 第五步：继续 runtime intervention/probe

如果下一轮目标是垂直扩展，例如中途切片分析、Metropolis kick、restart：

- 先定义 `AlgorithmStateView` 里需要暴露的 state。
- 再给 algorithm loop 插 hook。
- Probe 只读，不改变 state。
- Intervention 可改 state，但必须声明 `modifies_state` 和 physical sensitivity。
- 每个 intervention 必须写入 metadata。

## 11. 下一轮可直接使用的测试命令

普通全量：

```bash
python -m pytest -q
```

Hard-interface 相关：

```bash
python -m pytest -q tests/test_contract_parameter_specs.py tests/test_config_contract.py tests/test_parallel_memory_contract.py
```

Trial 相关：

```bash
python -m pytest -q tests/test_trial_contract.py tests/test_trial_cli.py
```

Metric/schema 相关：

```bash
python -m pytest -q tests/test_contract_metric_output_specs.py tests/test_result_schema.py tests/test_result_cube.py tests/test_tensor_parity_contract.py
```

Config/trial runtime：

```bash
mf validate src/matrix_factorization/config.yaml
mf trial validate matrix_bigamp_quick
mf trial run matrix_bigamp_quick
mf trial run scan_alpha_quick
mf trial run scan_mixed_axes_quick
```

注意：

- `runs/ / results/ / artifacts/` 不应提交。
- 大 GB calibration raw artifact 不应提交。
- 如果要跑 calibration，先确认本地 GPU 资源和 hard stop。

## 12. 当前代码审查重点

新对话如果要 review 当前未提交 projection metric 迁移，应重点看：

- `projection_abs()` 是否正是用户认可的公式。
- `Q_Y` 是否不再调用 cosine normalization。
- `Q_W/Q_X` 是否是 coordinate projection。
- `Q_W_GRAM_ROOT/Q_X_GRAM_ROOT` 是否是 sqrt 后 diagnostic，不替代 projection。
- Tensor `Q_N` 是否对齐 matrix latent `Q_W/Q_X`。
- Spreading/tensor `Q_Y_unobserved` 是否来自 heldout F-aware measurement，而不是 dense mask complement。
- `metrics.json` schema v3 是否明确标记新旧 `Q_Y_mean` 不可直接比较。
- Default plots 是否不再默认画 `MSE/Gen_Error/Q_W_prime/Q_X_prime/physical_overlap`。

## 13. 当前 open questions

需要用户未来确认的问题：

- `Q_N` 是否需要类似 Gram-root 的 gauge/permutation diagnostic。
- Tensor heldout edge 数量是否默认等于 observed edge 数，还是应暴露为 YAML 参数。
- `Q_Y_unobserved` 在 tensor/spreading 中是否应始终强制生成，还是允许 unavailable。
- Legacy result loader 是否需要自动标注旧 run 的 `Q_Y_mean` 为 `legacy_cosine`。
- Default plot 上是否优先显示 `Q_W/Q_X projection`，还是优先显示 `Q_W_GRAM_ROOT/Q_X_GRAM_ROOT`。
- Projection metric 的 degenerate teacher norm 是否只返回 0，还是应该带 warning/error。

需要本地 GPU 验证的问题：

- Projection metric 新 schema 下各算法曲线是否符合物理直觉。
- Tensor parallel QN/QY projection 是否稳定。
- Spreading heldout QY 是否和观测 QY 呈合理差异。
- Intelligent parallel estimator 是否需要按新 projection result retention 重新校准。

## 14. 建议下一步任务拆分

### Task A：提交当前 projection metric 迁移

允许改代码：否，只做最后检查和 commit。

需要 GPU：否。

内容：

- 跑 `python -m pytest -q`。
- 跑 `mf validate` 和代表性 quick trial。
- commit/push 当前未提交 projection metric 迁移。

### Task B：补 projection metric 手算 fixture

允许改代码：是。

需要 GPU：否。

内容：

- 增加 matrix/spreading/tensor 小 fixture。
- 验证 projection > 1 不 clip。
- 验证 sign flip absolute。
- 验证 observed/unobserved/full set。

### Task C：清理 legacy formal metric surface

允许改代码：是。

需要 GPU：否。

内容：

- 确认 active formal schema 没有 `MSE/Q_Y_COS/Gen_Error/physical_overlap/Q_prime`。
- 保留 internal loss/debug helper。
- 对 legacy helper 添加注释和 docs。

### Task D：Projection schema 旧结果兼容

允许改代码：是。

需要 GPU：否。

内容：

- Loader 识别旧 schema。
- 旧 `Q_Y_mean` 标记为 legacy cosine/proxy。
- 新旧 run 对比时 warning。

### Task E：智能显存估算公式修正

允许改代码：是。

需要 GPU：是，本地 GB 级 calibration。

内容：

- 按算法重做 stage peak formula。
- 目标 raw allocated error <= 10%。
- 不通过就记录 review queue，不 push “已完成”结论。

### Task F：Scan/PlotQuery 研究图体验

允许改代码：是。

需要 GPU：否，先用 quick trial。

内容：

- 增强 `mf explain-config` 对 PlotQuery 的解释。
- 增加 CLI 或 helper 查询 ResultCube。
- 支持用户挑选两组 coordinates 画同一 metric。

### Task G：Runtime probe/intervention 接入 algorithm loop

允许改代码：是。

需要 GPU：否，先 small fixture。

内容：

- 先选一个 matrix algorithm 暴露 step-level state。
- 接入只读 probe。
- 再考虑 intervention。

### Task H：Tensor serial/parallel parity

允许改代码：谨慎。

需要 GPU：是。

内容：

- 先只做 parity tests/metadata。
- 再逐项核对 teacher scale、alpha normalization、F distribution、Onsager、damping。
- 不要直接合并训练 loop。

## 15. 给新对话的最短接手提示

如果新开 Codex 对话，可以直接贴：

```text
请阅读 docs/project_handoff_2026-04-27.md。
当前分支 dev，src/matrix_factorization 是 active package。
先不要改物理公式。先确认当前未提交 projection metric 迁移是否仍通过 tests/validate/trial。
如果通过，先提交；然后从文档 Task B 或 Task E 继续。
```
