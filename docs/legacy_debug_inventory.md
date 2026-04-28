# Legacy 和 Debug 清单

这份清单记录哪些文件有助于理解项目，但不应在未经审查时当作 active runtime path 使用。

## Active Runtime Path

当前主链路由这些文件定义：

- `src/matrix_factorization/cli.py`
- `src/matrix_factorization/core/experiment/config.py`
- `src/matrix_factorization/core/experiment/runner.py`
- `src/matrix_factorization/core/experiment/data_factory.py`
- `src/matrix_factorization/core/experiment/result.py`
- `src/matrix_factorization/modules/registry.py`
- `src/matrix_factorization/modules/algorithms/agd.py`
- `src/matrix_factorization/modules/algorithms/bigamp/standard.py`
- `src/matrix_factorization/modules/algorithms/bigamp/spreading.py`
- `src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading_parallel.py`
- `src/matrix_factorization/modules/metrics/`
- `src/matrix_factorization/modules/outputs/`

## 已注册但不是主链路的算法

- `src/matrix_factorization/modules/algorithms/agd_tensor.py`
  - 原因：注册名是 `agd_tensor`，但 `ExperimentConfig` 不接受该 key，CLI mapping 也没有暴露它。
  - 风险：它会在内部重新生成 tensor teacher data，而不是使用 runner 传入的 teacher/data。

- `src/matrix_factorization/modules/algorithms/combined.py`
  - 原因：注册名是 `combined`，但它更像 selector/helper，不是 runner-compatible training algorithm。
  - 风险：constructor 形状和 `ExperimentRunner._get_algorithm()` 不匹配。

- `src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading.py`
  - 原因：注册名是 `bigamp_tensor`，但 CLI 在 `tensor_order >= 3` 时会路由到 `bigamp_tensor_parallel`。
  - 状态：在 parity plan 写清之前，保留为 serial/reference tensor 路径。

## Legacy 或 Broken Path

- `src/matrix_factorization/modules/algorithms/legacy/agd_spreading.py`
  - 原因：历史 spreading AGD 实现；active algorithm package 默认不导入。直接 import 时仍指向过时的相对 registry 路径。

- import `matrix_factorization.core.runner` 的 scripts 或 tests
  - 原因：当前 active runner 是 `matrix_factorization.core.experiment.runner`。

- import `matrix_factorization.modules.algorithms.bigamp.spreading_parallel` 的 scripts 或 tests
  - 原因：active tree 中已没有该 module。

- 使用 algorithm key `bigamp_spreading_parallel_unit` 的 scripts
  - 原因：active registry 当前没有这个 key。

- 比较或引用 `src/smf/` 的 scripts
  - 原因：active package 是 `src/matrix_factorization`，不是 `smf`。

## Debug 和 Experiment Scripts

除非明确提升为测试，否则这些目录先视作本地调查工具：

```text
scripts/debug/
scripts/experiments/
scripts/verification/
tests/debug/
tests/verification/
```

这些区域常见问题：

- 直接实例化 algorithm，而不是走 `ExperimentRunner`；
- standalone teacher/data generation；
- 绕开 `ExperimentResult.save()` 的输出 schema；
- hard-coded 的旧 module name 或 registry key；
- 运行时间长或依赖 GPU，不适合放入常规 test execution。

## Legacy / Local-GPU pytest modules

这些测试文件保留为历史诊断或本地 GPU 调查，但已经在模块级 `pytest.skip(..., allow_module_level=True)`，不属于默认 `python -m pytest`：

- `tests/test_clamp.py`（旧 `spreading_parallel` variance clamp 调查）
- `tests/test_damping.py`（旧 damping sweep 调查）
- `tests/test_independent_alpha.py`（旧 independent alpha 调查）
- `tests/test_llm_robustness.py`（已移除 ConfigAdvisor / `llm_advisor`）
- `tests/test_scientific_pressure.py`（已移除 ConfigAdvisor / `llm_advisor`）
- `tests/test_random_spreading.py`（legacy random_spreading verification，依赖旧 helper export）
- `tests/test_supergraph.py`（旧 supergraph parallel suite，依赖已移除 `spreading_parallel`）
- `tests/test_onsager_impact.py`（大尺寸 tensor Onsager 本地 GPU 诊断）
- `tests/test_tensor_parallel_verification.py`（旧 serial/parallel parity 长跑诊断，使用 `src.*` 导入）

默认 pytest 边界在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 中声明：只收集 `tests/`，不收集 `docs/`、`runs/`、`results/`、`artifacts/`。

机器可读清点表在 `src/matrix_factorization/core/contracts.py`：

- `get_algorithm_source_inventory()` 覆盖 `modules/algorithms/`。
- `get_auxiliary_source_inventory()` 覆盖 `experiments/`、`scripts/`、`src/matrix_factorization/export/`、`tests/debug/`、`tests/verification/`。
- `get_runtime_extension_source_inventory()` 覆盖 `modules/interventions/`。
- `get_metric_source_inventory()` 覆盖 `modules/metrics/`。
- `get_output_source_inventory()` 覆盖 `modules/outputs/`。
- `get_teacher_source_inventory()` 覆盖 `modules/teachers/`。
- `get_graph_source_inventory()` 覆盖 `modules/graphs/`。
- `get_config_source_inventory()` 覆盖 versioned config 入口。
- `get_repository_surface_inventory()` 控制根目录可出现的 `.py/.md/.yaml/.toml/.png/.pt` 文件。
- `get_trial_source_inventory()` 覆盖 `trials/` 中登记的 research trial 文件。
- `get_legacy_pytest_inventory()` 覆盖默认 pytest 中显式跳过的 legacy/local-GPU 诊断测试。

新增这些区域的 `.py` 文件如果没有进入清点表，source inventory test 会失败。这个机制的目的不是删除实验脚本，而是防止仓库再次出现“没人知道用途”的隐藏入口。

## Research Trial 区域

`trials/` 是调试试跑入口，不是正式 pytest，也不是大实验 artifact 区域。

- trial 参数文件放在 `trials/active/<trial_key>/config.yaml`。
- 用户要求“试跑、调试参数、quick trial、看一下能不能跑”时，应优先改 trial config，而不是正式 `src/matrix_factorization/config.yaml`。
- `mf trial run` v1 只运行 `runtime_class: quick`。
- `medium` 和 `gpu_heavy` 只允许登记、解释和校验，不能被默认自动运行。
- trial 结果写入 ignored workspace：`artifacts/trials/`。

## Artifact 区域

不要把这些目录当作源码依赖：

```text
runs/
results/alpha_scan/
artifacts/
src/matrix_factorization/Replica_results/
smf/test_output/
```

`results/latest/` 只是 versioned lightweight display snapshot，不是 canonical analysis input。

## 安全清理顺序

1. 删除或移动旧脚本前，先在清单里记录。
2. 先做文档级分类：active、debug、local experiment、reference、broken legacy。
3. 只有确认脚本仍要运行时，才替换旧 import path。
4. 不要静默把旧实验脚本改成新的物理 convention。
5. 只有小型、确定性检查才提升到 `tests/`。
6. 任何改变 tensor、AMP、spreading 或 memory-planning 行为的改动，都需要本地 GPU 验证。

## 候选后续任务

- 增加 CLI config contract tests。
- 增加 result schema contract tests。
- 增加 latest exporter contract tests。
- 给 registry duplicate key 增加 warning。
- 写 tensor serial/parallel parity design。
- 决定 `agd_tensor` 是提升、重写还是归档。
- 决定 `combined` 是否应该留在 algorithm registry。
- 替换仍有价值脚本中的 obsolete import。
- 标注或归档纯历史 reference 脚本。
