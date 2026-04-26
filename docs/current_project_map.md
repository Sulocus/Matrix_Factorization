# 当前项目地图

这份文档记录 `dev` 分支当前状态，用来在新增物理内容或算法内容之前，先恢复对项目结构的控制。

## 基本约定

- 当前 active package 是 `src/matrix_factorization`。
- 当前开发分支是 `dev`；`main` 不是开发目标。
- 生成的实验输出不进入源码管理，应保留在被 ignore 的 `runs/`、`results/`、`artifacts/` 或 `src/matrix_factorization/Replica_results/`。
- 快速检查属于普通本地测试；小规模真实试跑使用 quick trial。高显存实验以本地 GPU 验证为准。
- 任何算法改动都必须显式记录 latent scaling、alpha normalization、damping 语义、Onsager handling，以及每个 `Q_Y` variant 的准确含义。

## 顶层结构

```text
Matrix_Factorization/
├─ src/matrix_factorization/        当前 Python package 和 CLI
├─ configs/                         本地研究配置
├─ scripts/                         analysis、debug、experiment、maintenance 工具
├─ tests/                           unit、compatibility、verification 测试
├─ docs/                            项目地图、理论、报告、reference code
├─ results/latest/                  轻量展示快照
├─ results/alpha_scan/              被 ignore 的本地 run artifact 区域
├─ src/matrix_factorization/Replica_results/
│                                     被 ignore 的历史 GPU artifact 区域
└─ smf/                              生成物或 legacy 输出区，不是 active package
```

## Active Package 结构

```text
src/matrix_factorization/
├─ cli.py                            YAML/CLI 入口
├─ __main__.py                       python -m matrix_factorization 入口
├─ config.yaml                       当前默认 YAML，仍含部分历史字段
├─ core/
│  ├─ contracts.py                  参数、algorithm、metric、output、intervention 的硬接口声明
│  ├─ planning.py                   从 config 构造 ExperimentPlan 并做 preflight validation
│  ├─ config.py                      旧 Config schema
│  ├─ experiment/
│  │  ├─ config.py                   ExperimentConfig 和 dataclass
│  │  ├─ data_factory.py             teacher/data/mask/supergraph 构造
│  │  ├─ result.py                   SingleRunResult 和 ExperimentResult
│  │  └─ runner.py                   主 ExperimentRunner
│  └─ parallel/
│     ├─ parallel_coordinator.py     alpha batch 规划
│     ├─ resource_execution.py       WorkItem / ResourceExecutionPlan 显式调度地图
│     ├─ memory_estimator.py         显存估计
│     ├─ memory_guard.py             运行时显存监控
│     ├─ batch_checkpoint.py         batch checkpoint
│     └─ calibration/                校准工具，部分仍是历史路径
├─ modules/
│  ├─ registry.py                    algorithm/graph/teacher/metric/output registry
│  ├─ algorithms/                    AGD、BiGAMP、tensor 和 legacy 算法
│  ├─ graphs/                        random mask 和 spreading supergraph
│  ├─ metrics/                       overlap、spreading、tensor、replica metrics
│  │  ├─ contract_compute.py          runner fallback metric payload adapter
│  │  ├─ spec_adapter.py              MetricSpec flat-key 校验和语义报告
│  │  └─ overlap/spreading/tensor/... 现有公式定义；不在硬化阶段改公式
│  ├─ outputs/                       plotting、storage、latest exporter
│  ├─ interventions/                 runtime hook/intervention 骨架；当前不改变算法行为
│  └─ teachers/                      teacher 实现和 registry hook
├─ export/                           standalone bundling/export 路径
├─ presets/                          package 内部 preset
└─ ui/                               UI helper
```

## 主运行链路

```text
YAML
  -> matrix_factorization.cli.load_yaml_config()
    -> core.planning.ExperimentPlan
      -> preflight validation / explain-config
    -> core.experiment.config.ExperimentConfig
      -> core.experiment.runner.ExperimentRunner.run()
        -> core.scan_planning.ScanPlan
        -> core.parallel.ParallelCoordinator.plan_execution()
        -> core.parallel.ResourceExecutionPlan / WorkItem metadata
        -> core.experiment.data_factory.DataFactory.create()
          -> create_teacher()
          -> create_masks() or create_spreading_data()
        -> modules.registry.get_algorithm()
          -> registry-gated AlgorithmSpec
          -> algorithm.train_batch_alphas()
          -> ExperimentRunner._coerce_algorithm_result()
        -> ExperimentRunner._compute_metrics()
          -> MetricSpec semantic metadata / flat key compatibility
          -> core.experiment.result.SingleRunResult
            -> core.experiment.result.ExperimentResult
              -> ExperimentResult.save()
                -> output dependency preflight
                -> config.json
                -> metadata.json
                -> metrics.json
                -> events.jsonl
                -> artifacts/results.pt
                -> plots/
                -> manifest.json
              -> modules.outputs.latest.refresh_latest_results()
```

关键位置：

- CLI 入口：`pyproject.toml` -> `matrix_factorization.cli:main`。
- YAML 读取和 `output_options` 解析：`src/matrix_factorization/cli.py`。
- 硬接口声明：`core/contracts.py`。
- 运行前计划和 validation：`core/planning.py`。
- 实验配置 dataclass：`core/experiment/config.py`。
- 数据构造：`core/experiment/data_factory.py`。
- Algorithm registry：`modules/registry.py`；algorithm 注册现在必须绑定 `AlgorithmSpec`。
- Runner 执行和 metrics 分发：`core/experiment/runner.py`。
- Canonical result schema：`core/experiment/result.py`。
- 轻量 latest 展示导出：`modules/outputs/latest.py`。
- Runtime hook 骨架：`modules/interventions/`。

## Algorithm 家族

```text
modules/algorithms/
├─ agd.py
│  └─ 注册名 "agd"；当前 matrix optimization baseline
├─ agd_tensor.py
│  └─ 注册名 "agd_tensor"；ExperimentConfig 当前不接受该 key
├─ combined.py
│  └─ 注册名 "combined"；selector/helper，不是 runner-compatible trainer
├─ legacy/agd_spreading.py
│  └─ 历史 spreading AGD 路径；active registry 默认不导入
└─ bigamp/
   ├─ standard.py
   │  └─ 注册名 "bigamp"；当前 dense matrix BiGAMP
   ├─ spreading.py
   │  └─ 注册名 "bigamp_spreading"；当前 matrix/general spreading BiGAMP
   ├─ tensor_spreading.py
   │  └─ 注册名 "bigamp_tensor"；serial/reference tensor 路径
   ├─ tensor_spreading_parallel.py
   │  └─ 注册名 "bigamp_tensor_parallel"；当前 tensor 运行路径
   ├─ step.py, core.py, f_gen.py
   │  └─ matrix spreading AMP kernel 和 F/Y 生成
   └─ tensor_step*.py, tensor_hypergraph.py, tensor_supergraph.py, tensor_memory.py
      └─ tensor kernel、graph 生成、memory/probing helper
```

长期目标上，`bigamp_tensor` 和 `bigamp_tensor_parallel` 应该描述同一个 tensor-spreading 物理算法。当前它们仍是两个实现，batching、alpha 约定和 metrics 生产路径不同。在写清 parity plan 之前，先把 `bigamp_tensor` 视作 serial/reference 路径。

## Result Schema

`ExperimentResult.save()` 生成的 canonical run 目录：

```text
run_dir/
├─ config.json
├─ metadata.json
├─ metrics.json
├─ events.jsonl
├─ artifacts/
│  └─ results.pt
├─ results.pt -> artifacts/results.pt
├─ plots/
└─ manifest.json
```

`results/latest/` 是轻量展示快照，不是完整分析 schema。它会刻意省略 tensor 和 checkpoint。

当前仍有多条 schema 路径：

- `ExperimentResult.save()` 是主 schema。
- `ResultStorage.save()` 对普通 dict 有 fallback schema。
- `export/bundler.py` 会为部分旧 workflow 写 standalone result JSON。

详细 schema contract 见 `docs/result_schema_contract.md`。当前保存阶段会重新校验每个 scan point 的 metric payload；未声明 key 不能落盘。

## 已知控制风险

- `ScanPlan` 已能把 `alpha/steps/nested/hysteresis` 表达成统一点集；`nested_scan` 和 `hysteresis_scan` 的实际执行仍暂时走 legacy handler，后续需要迁移到统一 runner。
- `teacher.init_distribution` 已进入 `DataFactory.create_teacher()` 和 saved teacher 逻辑；后续仍需要本地结果确认不同 teacher 分布下的历史曲线可比性。
- `ResourceExecutionPlan` 已把 alpha batch 映射成 `WorkItem` metadata；sample splitting 仍受 `BatchingSpec.sample_range_honored` 约束，未声明支持的 algorithm 不能假装支持。
- Tensor parallel 在主 coordinator 之外还有内部 alpha batching。
- Teacher 和 graph registry 存在，但主 `DataFactory` 只使用了其中一部分。
- `Q_Y_mean` 在 dense、spreading、serial tensor、parallel tensor 中含义不同。
- Tensor serial/parallel algorithm class 已经显式返回 metrics-only `AlgorithmResult`；底层数值实现仍暂用 `_batch_metrics` 作为内部缓冲。保存时不再把 tensor placeholder `W/X` 标成真实 matrix factors。

## 当前低风险工作队列

1. 主要结构变化后同步更新这份地图。
2. 新参数必须先进入 `ParameterSpec`，再进入 YAML/loader/consumer。
3. 新 algorithm 必须先进入 `AlgorithmSpec`，再进入 registry 和 runner。
4. 新运行中诊断必须先进入 `ProbeSpec`，并声明需要的 `state_capabilities`。
5. 新 intervention 必须先进入 `InterventionSpec`，并只通过 `AlgorithmStateView` 声明读写 state。
6. 新 run 后分析必须先进入 `AnalyzerSpec`，并声明需要的 metric/artifact。
7. 任何 metrics 生产或命名变化后，同步更新 `docs/metrics_semantics.md` 和 `MetricSpec`。
8. 删除或移动历史脚本前，同步更新 `docs/legacy_debug_inventory.md`。
9. 在合并 tensor serial 和 tensor parallel 前，先写 parity contract：teacher scale、alpha 定义、graph object、damping 语义、Onsager 设置和 metric 语义必须一致。
