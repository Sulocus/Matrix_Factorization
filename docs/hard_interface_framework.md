# 硬接口框架

这份文档描述当前已经落地的硬接口框架。它的目标不是改变物理算法，而是在运行前暴露“参数有没有接上、algorithm 能产出什么、output 能不能消费”的信息。

## 当前已落地部分

```text
YAML
  -> load_yaml_config()
  -> ExperimentPlan
  -> preflight validation
  -> ExperimentRunner
```

新增核心对象：

- `ParameterSpec`：声明 YAML 参数路径、owner、consumer、是否物理敏感、字段状态。
- `AlgorithmSpec`：声明 algorithm 的 data requirements、capabilities、produced metrics/artifacts、compatible outputs。
- `TeacherSpec`：声明 teacher initializer 的状态、需要的配置、产出 artifact 和尺度约定。
- `GraphSpec`：声明 graph/mask generator 的状态、需要的配置、产出 artifact 和 ensemble 语义。
- `AlgorithmResult`：runner 内部的正式算法输出容器；matrix algorithm 包装真实 `W/X`，tensor metrics-only 路径不再把 dummy zero `W/X` 当成真实 factor。
- `AlgorithmStateView`：probe/intervention/analyzer 面向的状态视图，只暴露 capability 名称，不要求外部插件知道 algorithm 内部变量名。
- `MetricSpec`：声明 metric 的输入、输出、space/scope/relation/normalization。
- `OutputSpec`：声明 plot/export 需要哪些 metric 或 artifact。
- `InterventionSpec`：声明训练过程 intervention 的 hook、state requirement 和兼容 algorithm。
- `ProbeSpec`：声明运行中只读诊断，例如按 step 抽取 state slice，不修改训练状态。
- `AnalyzerSpec`：声明 run 后分析，例如 replica summary 或 tensor heatmap summary。
- `ExperimentPlan`：把当前 config 解析成“实际会跑什么”的计划，并收集 warnings/errors。
- `OutputPlan`：从 `OutputSpec` 和 `output.plots` 展开本次 run 需要的 metric、artifact 和 output file。

## 新命令

```bash
mf validate path/to/config.yaml
mf explain-config path/to/config.yaml
mf validate --strict path/to/config.yaml
mf validate --json path/to/config.yaml
mf trial list
mf trial explain matrix_bigamp_quick
mf trial validate matrix_bigamp_quick
mf trial run matrix_bigamp_quick
```

`validate` 只判断配置是否能通过 hard contract。它不运行算法。

`explain-config` 输出实际生效的 algorithm、teacher、spreading、output、intervention 和 warnings。它用于回答“我在 YAML 里写的东西到底有没有生效”。

`--strict` 会把 `parsed_only`、`legacy`、`deprecated` 字段从 warning 升级成 error，也会把“当前 algorithm/scan 路由下不生效”的 active 字段升级成 error，用来检查新主链路是否还依赖软字段或半接入字段。`--json` 输出机器可读的 plan，方便后续 agent 或脚本按 error 修。JSON 中包含 `parameter_chain`，记录 YAML path、owner、status、consumer、是否物理敏感；后续 agent 可以直接按这张链路表检查“写了但没接上”的字段。

`mf trial ...` 是研究试跑入口。它使用 `trials/active/<trial_key>/config.yaml`，不会修改正式默认配置。v1 只允许 `runtime_class: quick` 被 `mf trial run` 自动执行；`medium` 和 `gpu_heavy` 只能登记、解释和校验，后续必须显式放行才运行。

## 字段状态

```text
active
  当前主链路中被正式消费。

parsed_only
  会被读取或保留，但行为尚未完全硬化。validate 不失败，但会 warning。

legacy
  历史字段。validate 不失败，但会 warning。

deprecated
  不推荐继续使用。
```

## 当前策略

- 未注册 YAML 字段直接 error。
- `@register_teacher(...)` / `@register_graph(...)` 注册时必须能绑定同名 `TeacherSpec` / `GraphSpec`；新增 teacher 或 graph 生成器不能只注册代码而没有 contract。
- active 参数如果在当前 algorithm/scan 路由下不生效，普通 `validate` 会 warning，`validate --strict` 会 error。
- `AlgorithmSpec.required_config_paths` 和 `TeacherSpec.required_config_paths` 会被 preflight 实际检查；如果 spec 声明需要某个字段但当前 plan 找不到 effective trace，或该字段在当前路由下不生效，validate 会 error。
- `@register_algorithm(...)` 注册时必须能绑定同名 `AlgorithmSpec`；缺 spec 或重复 key 会在 import/test 阶段失败。
- `@register_metric(...)` 注册时必须能绑定同名 `MetricSpec`；未来新增 metric handler 时不会只注册代码而漏掉语义声明。
- `@register_output(...)` 注册时必须能绑定同名 `OutputSpec`；旧的 `plotting/storage/combined` 已作为 legacy/backend output spec 明确分类。
- active algorithm 必须有 `AlgorithmSpec`，且 active spec 必须出现在 registry 中。
- output 必须和 algorithm 的 `compatible_outputs` 匹配。
- `output.plots` 的曲线代码必须能映射到当前 algorithm 的 `MetricSpec.produces`；缺失时 `mf validate` 失败。
- intervention 必须和 algorithm 的 `state_capabilities` 匹配；例如 tensor algorithm 不能静默接受只为 spreading 声明的 restart 行为。
- probe 必须和 algorithm 的 `state_capabilities` 匹配。
- analyzer 必须和 algorithm/result contract 的产出匹配。
- tensor 路径的 `bigamp_tensor_parallel` 仍保持现有数值行为，但 `use_compile` 不再被硬编码覆盖。
- runner 会优先调用 algorithm 的 `train_batch_result()`；`bigamp_tensor` 和 `bigamp_tensor_parallel` 已经显式返回 metrics-only `AlgorithmResult`。active path 在拿到 `AlgorithmResult` 后不会再从私有 `_batch_metrics` 补缺失指标；旧 `_batch_metrics` 只保留为 tensor 数值实现内部的临时缓冲和未迁移 legacy fallback。
- tensor metrics-only result 的 `SingleRunResult.W_students/X_students` 保存为 `None`，避免把 placeholder 当成真实 factor。
- runner 的 algorithm cache 已按 effective config signature 分区，不再只按 algorithm key 复用。每个 algorithm 实例会带 `_contract_config_trace`，run metadata 中写入 `algorithm_config_trace`，用于追踪参数是否实际传入 algorithm 构造。
- `metrics.json` 保留旧 flat keys，同时新增 `metric_semantics` 和 `factor_payload_contract`，用于区分同名 key 在 matrix/tensor/spreading 中的语义，并声明当前 run 是否真的有 `W_students/X_students`。
- runner 中的 matrix/spreading metric fallback 已移到 `modules/metrics/contract_compute.py`；runner 只负责选择 metric 来源，metrics package 负责生成 payload 并接受 `MetricSpecAdapter` 校验。
- `modules/metrics/` 已纳入 source inventory。新增 metric 源文件如果没有分类，source inventory 测试会失败。
- source inventory 已扩展到 `experiments/`、根目录表面文件、config 入口、`modules/teachers/` 和 `modules/graphs/`。新的根目录 `.py/.md/.yaml/.toml/.png/.pt`、新的 `experiments/*.py`、新的 config 文件或新的 teacher/graph 源文件如果没有分类，source inventory 测试会失败。
- heatmap/GIF 已从“失败只打印 warning”改成 hard error：如果 OutputSpec preflight 认为可以生成图，但 plotting backend 失败，`ExperimentResult.save()` 会直接报错。
- metric fallback 失败不再返回全 0 placeholder。`contract_compute` import 或执行失败会直接报错，避免把“metric 计算失败”伪装成“overlap 为 0”。
- `mf validate --json` 会输出 `issues/error_codes/warning_codes`。例如 `UNKNOWN_YAML_FIELD`、`INTERVENTION_UNSUPPORTED`、`OUTPUT_METRIC_UNAVAILABLE`，便于后续 agent 按错误码补齐 contract。
- `ExperimentPlan` 会包含当前 teacher 的 `TeacherSpec`，用于在解释配置时显示 teacher status、产出和尺度约定；这不改变 `DataFactory` 的 teacher 构造逻辑。
- `parameter_chain` 已扩展为参数消费 trace：每个 YAML path 会记录 `active_in_current_plan`、`effective_value`、`effective_source`、`derived_effect` 和 `consumption_status`。这不是证明物理行为正确，而是让“写了参数但没进入有效计划”的问题更容易在 preflight 中定位。
- `parameter_chain` 会处理 seed alias 和覆盖关系：`seeds.model` 追踪到 `seeds.base_seed`，`seeds.data` 追踪到 `seeds.teacher_seed`；如果同时写了 `seeds.spreading_seed` 和更具体的 `spreading.seed`，前者会标成 `overridden_current_route` 并显示实际 spreading seed。
- `parameter_chain` 会按当前 algorithm/scan 路由标记 inactive 字段。例如 AGD 路由下写入 `algorithm_params.damping` 或 `spreading.seed`，会显示 `inactive_current_route`，避免把“被 dataclass 接住”误读成“算法实际消费”。
- runtime hooks 已先接入低风险 runner-level `after_batch`：`batch_summary` probe 会在 `runtime_extension_report.probe_reports.batch_summary` 里记录 batch index、alpha values、metric keys 和 AlgorithmResult 可用输出。它不进入算法 step loop，因此不改变训练状态、随机数或数值行为。
- `state_slice`、`tensor_state_slice`、`variance_slice` 目前标记为 `declared_only`。如果用户在 YAML 中请求这些尚未接入 algorithm step hook 的 probe，`mf validate` 会报 `PROBE_DECLARED_ONLY`，避免“配置看似生效但运行时没有产物”。
- runtime extension 触发时会检查实际 `AlgorithmStateView` 是否包含 spec 声明的 `requires_state`；如果 algorithm spec 声明了能力但运行时没有传出对应 state，会直接报错。
- analyzer 在 run 后执行前会检查实际 `ExperimentResult` 是否包含 `AnalyzerSpec.requires` 里的输入；如果理论 contract 说能产出、实际结果缺失，会直接报错。
- 每个 scan point 的 result 会保存 `metric_contract`，`metrics.json` 顶层也会保存 `metric_contracts`，用于追踪 flat metrics 对应哪些 `MetricSpec` 以及来源是 `algorithm_result`、runner matrix 计算还是 runner spreading 计算。
- runner 会检查 algorithm 实际返回的 metric keys 是否已被 `MetricSpec` / `AlgorithmSpec.produced_artifacts` 声明；新 metric 没注册会在主链路失败。
- `ExperimentResult.save()` 在绘图前检查 custom plot 和 heatmap 的实际依赖；缺 metric/artifact 会抛错，不再静默画 0 或跳过。
- 保存阶段会写出 `output_contract.json`，记录 output plan 实际需要的 metric/artifact、当前 result 实际提供了什么、是否缺失。
- 默认 scalar plot 只画真实存在的 metric；如果没有 `Q_W_mean`，不会再补 0 曲线；如果没有任何可用的 `Q_Y` 类 metric，会直接报错。
- run metadata 会保存 `experiment_plan` 摘要，方便回看本次 run 的 effective parameters 和 warnings。
- `metrics.json` 同样写入 `contract` 摘要，方便轻量后处理只读一个 JSON 就能知道本次结果的 contract 背景。
- Research Trial v1 已接入：`matrix_bigamp_quick` 是内置 quick trial，真实运行时写入 `runs/trials/matrix_bigamp_quick/`，不刷新 `results/latest`。
- nested scaling sweep 会透传 `teacher` 配置和 `experiment_plan`，避免外层 YAML 生效但内层尺寸 run 丢失配置。
- `results/latest` 只接受同时具有 `config.json` 和 `metadata.json` 的 run；半成品目录不会进入 display schema。
- runner 会构造 `RuntimeExtensionExecutor`，在高层 `before_initialize` / `after_run` hook 记录 runtime extension report；当前 executor 不进入算法 step，也不改变 warm start / restart 行为。

当前默认 `src/matrix_factorization/config.yaml` 会触发一个有意保留的 hard-interface error：`tensor_order=3` 路由到 `bigamp_tensor_parallel`，但 YAML 中 `algorithm_params.adaptive_restart: true` 只声明兼容 `bigamp_spreading`。这不是本轮自动修复对象，而是 validator 正确暴露出的“参数写了但当前算法未声明消费”的断点。

## 垂直扩展入口

结构骨架在 `src/matrix_factorization/modules/interventions/`。当前只定义 hook interface，不迁移现有算法行为。

运行中只读分析应优先做成 `ProbeSpec`，不要直接改 algorithm 内部变量。例如：

```yaml
probes:
  - key: tensor_state_slice
```

运行后分析应优先做成 `AnalyzerSpec`：

```yaml
analyzers:
  - key: tensor_heatmap_summary
```

如果 probe 需要的 state 没有被当前 algorithm 声明为 `state_capabilities`，`mf validate` 会失败。若后续代码真正触发 hook 但没有把对应 state 放进 `AlgorithmStateView`，runtime executor 也会失败。当前 runtime executor 会记录 probe/analyzer 的调度报告；真正的 per-step state slice 还需要后续 algorithm 暴露 `AlgorithmStateView` 后接入。

## 仍未完成的后续阶段

- `AlgorithmResult` 已接入 runner 的标准 alpha scan 和 steps/checkpoint scan。matrix algorithms 仍通过 base adapter 包装 legacy `W/X` tuple；tensor serial/parallel 已经在 algorithm class 内显式返回 metrics-only `AlgorithmResult`。后续若要继续硬化，应把 matrix algorithms 也逐步改成原生返回 `AlgorithmResult`。
- `MetricSpec` 已接入 plan validation、result semantic metadata、per-point metric contract 和 registry gate；runner 仍复用旧公式实现，后续可把公式计算进一步拆到 MetricSpec compute adapter 中。
- `OutputSpec` 已用于 preflight、保存前依赖检查和 `output_contract.json`，但具体绘图函数仍在 `ExperimentResult.save()` 中调度，后续可继续拆成独立 OutputPlan executor。
- `InterventionSpec` 已定义并有 no-op executor；现有 warm start / adaptive restart 还没有完全迁移成 intervention module。
- `ProbeSpec` / `AnalyzerSpec` 已定义并可 preflight，高层 runtime executor 已接入；per-step probe 和完整 analyzer executor 仍需后续 algorithm state 暴露。
- `bigamp_tensor` 与 `bigamp_tensor_parallel` 已有机器可读 parity contract 和 gap report；数值 parity、teacher scale、alpha normalization 等物理一致性仍需本地验证后才能合并。
