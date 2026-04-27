# Normalization 与 Precision Profile Audit

审计日期：2026-04-28。

审计范围：

- `src/matrix_factorization/core/contracts.py`
- `src/matrix_factorization/core/experiment/data_factory.py`
- `src/matrix_factorization/modules/algorithms/agd.py`
- `src/matrix_factorization/modules/algorithms/bigamp/standard.py`
- `src/matrix_factorization/modules/algorithms/bigamp/spreading.py`
- `src/matrix_factorization/modules/algorithms/bigamp/tensor_*.py`
- `src/matrix_factorization/core/parallel/memory_estimator.py`
- `tests/test_normalization_precision_contract.py`

这份审计不把 precision migration 标记为 completed。Normalization profile
已接入 runtime，但仍记录 graph/lambda/runtime probe 等剩余缺口。

## Verification Run

已在 `/home/sucia/Matrix_Factorization` 的 `dev` 分支运行：

```text
git status --short --branch
python -m pytest -q tests/test_config_contract.py tests/test_contract_parameter_specs.py tests/test_normalization_precision_contract.py
mf validate src/matrix_factorization/config.yaml
```

结果：

```text
branch: dev
tests: 52 passed
mf validate: errors none
warnings: legacy use_bf16, parsed_only output.storage_mode, inactive learning_rate/target_loss_threshold/use_early_stop
```

审计开始前，worktree 已经存在多处 unrelated modified/untracked files。

Normalization profile 实装后又运行：

```text
python -m pytest -q tests/test_config_contract.py tests/test_contract_parameter_specs.py tests/test_normalization_precision_contract.py
python -m pytest -q tests/test_standard_bigamp.py tests/test_random_spreading.py tests/test_tensor_e2e.py tests/test_tensor_parity_contract.py tests/test_contract_data_specs.py
python -m pytest -q
mf validate src/matrix_factorization/config.yaml
```

结果：

```text
contract/normalization tests: 60 passed
algorithm path checks: 11 passed, 1 skipped
full local suite: 335 passed, 9 skipped
mf validate: errors none
warnings unchanged: legacy use_bf16, parsed_only output.storage_mode, inactive learning_rate/target_loss_threshold/use_early_stop
```

2026-04-28 后续核验又运行：

```text
python -m pytest -q tests/test_bigamp_onsager_convention.py tests/test_spreading_batch_metrics.py tests/test_normalization_precision_contract.py tests/test_config_contract.py tests/test_contract_parameter_specs.py tests/test_scan_resource_planner_effective_config.py
python -m pytest -q
mf validate src/matrix_factorization/config.yaml
```

结果：

```text
targeted tests: 77 passed
full local suite: 348 passed, 9 skipped
mf validate: errors none
warnings unchanged: legacy use_bf16, parsed_only output.storage_mode, inactive learning_rate/target_loss_threshold/use_early_stop
```

## NormalizationSpec 状态

`src/matrix_factorization/core/contracts.py:NormalizationSpec` 当前声明默认 profile：

```text
default_profile = paper_sparse_sampling
profiles = [paper_sparse_sampling, internal_normalized_legacy]
latent_scale = O(1)
teacher_init_variance = 1
student_init_variance = 1
prior_precision_base = 1
interaction_scale = inverse_sqrt_M
schema_version = 5
```

`get_normalization_specs()` 把这个 active convention 应用于 `agd`、`bigamp`、
`bigamp_spreading`、`bigamp_tensor` 和 `bigamp_tensor_parallel`。

已经真实接入 runtime 的部分：

- `AlgorithmParams.normalization_profile` 已新增，默认
  `paper_sparse_sampling`，允许 `internal_normalized_legacy`。
- `DataFactory.create_teacher()` 根据 profile 生成 Gaussian/Rademacher 和
  orthogonal teacher factors，并继续计算 `Y=(1/sqrt(M))*W@X`。
- `AGDAlgorithm` 根据 profile 初始化 student；interaction scale 保持
  `1/sqrt(M)`。
- Dense `BiGAMPAlgorithm` 根据 profile 设置 student init、factor variance、
  prior precision base 和 variance clamp。
- `BiGAMPSpreading` 的 matrix/general/adaptive step 路径根据 profile 设置
  student init、factor variance、prior precision base 和 variance clamp。
- Serial 和 parallel tensor paths 根据 profile 设置 teacher extra modes、
  student init、factor variance、prior precision base 和 variance clamp。
- `agd_tensor` 是 experimental/unintegrated，但其 teacher/student init 也读取
  同一个 profile helper。

尚未接入的部分：

- 被审计的 active forward paths 里没有独立 `lambda` signal-strength 参数；
  当前等效 `lambda=1`。
- 没有 exact-degree paper graph profile。

结论：normalization profile 现在不是纯 metadata，`paper_sparse_sampling`
已经真实改变 teacher、student、variance 和 AMP update 的尺度，并成为默认
profile。旧 `1/M` convention 仍可用，但必须显式选择
`internal_normalized_legacy`。

## PrecisionPolicySpec 状态

`src/matrix_factorization/core/contracts.py:PrecisionPolicySpec` 声明了
`safe`、`fast`、`aggressive` 三个 profiles。`ParameterSpec` 声明
`algorithm_params.precision_profile`；`algorithm_params.use_bf16` 仍是 legacy
alias。`cli.load_yaml_config()` 在两个字段同时存在时以 `precision_profile`
为 source of truth，planning 也会验证 profile/fallback。

已经真实接入 metadata/config 的部分：

- `build_experiment_plan()` 暴露 requested profile 和 role dtype map。
- `estimation_params_from_config()` 把 `precision_profile`、`use_bf16` 和
  `role_dtype_map` 传给 memory estimator。
- AGD、dense BigAMP、spreading、serial tensor 和 tensor parallel 的 runtime
  metadata 会记录 requested/effective BF16、storage dtype、TF32、compile 和
  fallback 状态。

已经真实影响 runtime dtype 的部分：

- AGD：`precision_profile` 控制 CUDA autocast 里的 matmul/gradient blocks。
  但 parameter tensors 仍是 FP32。因此它会改变数值路径和 workspace dtype，
  但不会把 W/X state 存成 BF16。
- Dense BigAMP：CUDA BF16 可用时，fast/aggressive 会在 step 之间使用 BF16
  storage；但 `train_batch_alphas()` 在每次 `_bigamp_step()` 前把 state 和
  variance tensors cast 到 FP32，step 后再 cast 回去。因此 compute 大多仍是
  FP32。
- Matrix spreading：fast/aggressive 会把 student factors 和 variances 存为
  BF16。step kernels 使用 `W_flat.dtype`，计算时把 `F` cast 到同一 dtype，
  scatter/gather buffers 也按该 dtype 分配。aggressive 还会在
  `BiGAMPSpreading.create_spreading_data()` 中转换 floating `F_super` 和
  `Y_super`。Rademacher `F` 保持 `int8`。
- Serial tensor：storage dtype 会影响 student factors 和 variances。
  aggressive 会在训练前转换 floating `F` 和 `Y`。注意 serial Rademacher
  generator 当前生成 floating Rademacher `F`，不同于 spreading/tensor-supergraph
  的 `int8` Rademacher 路径。
- Tensor parallel：storage dtype 会影响 factors 和 variances。aggressive 会把
  floating `Y_super` 转成 BF16；Rademacher `F_super` 保持 `int8`。

比 spec 更弱或仍需验证的部分：

- Role contract 说 metric reductions 应保持 FP32，但 tensor/spreading 的
  metric/update code 没有系统性地把每个 reduction 都 cast 到 FP32。
- Dense BigAMP 的 memory estimation 把 dense intermediates 当作 storage
  dtype 估算，但 runtime 会在 `_bigamp_step()` 前把 step inputs cast 到 FP32。
- AGD 的 `PrecisionPolicySpec` 里写了 BF16 gradients，但 memory estimator 用
  FP32 student state dtype 估整个 AGD breakdown，没有单独建模 gradient/workspace
  role dtypes。
- Spreading memory estimation 把 `Y_flat` resident component 写死为 FP32，
  但 aggressive training 会把 `Y_super` 转成 BF16。
- Tensor parallel 的 algorithm-internal memory probe 用
  `use_bf16=self.use_bf16` 调用，但还不是完整 role-by-role dtype contract。

## Small Runtime DType Probe

已跑一个不写 artifacts 的 CUDA tiny probe：

```text
algorithm = bigamp_spreading
N1=N2=8, M=4, S=1
one alpha, one step
use_compile=false
F = Rademacher
```

观测结果：

```text
safe:
  cuda_bf16_supported = true
  algorithm_storage_dtype = torch.float32
  effective_use_bf16 = false
  F_super_dtype = torch.int8
  Y_super_dtype = torch.float32
  returned_W_dtype = torch.float32
  returned_X_dtype = torch.float32

aggressive:
  cuda_bf16_supported = true
  algorithm_storage_dtype = torch.bfloat16
  effective_use_bf16 = true
  F_super_dtype = torch.int8
  Y_super_dtype = torch.bfloat16
  returned_W_dtype = torch.bfloat16
  returned_X_dtype = torch.bfloat16
```

结论：在这台 CUDA 机器的 spreading training kernel 上，aggressive 不是纯
metadata，它确实改变了 student/Y 的 runtime dtype。

## 为什么 aggressive 没有明显省显存或提速

对当前默认 Rademacher spreading config，aggressive 本来就很可能和 fast
非常接近：

- Rademacher `F` 已经是 `int8`，aggressive 不能再把它减半。
- 主峰值来自 `O(B*S*C*M)` 的 gather/scatter/workspace buffers。fast 已经把
  student/workspace path 推到 BF16。
- aggressive 主要额外改变 `Y`，但 `Y` 是 `O(S*C)`，在 `M=50` 时远小于
  `O(S*C*M)` edge workspaces。
- active `train_batch_alphas()` 会在返回前分配 FP32 的 `W_result` 和
  `X_result`，final result retention/copies 可能抵消部分可见 savings。
- step 很可能被 memory/index/scatter 限制。Advanced indexing、`scatter_add_`、
  int64 offset tensors、masks、graph generation/top-k、compile cache、metric
  materialization、CPU-GPU sync、saving 和 plotting 都可能支配 wall time。
  BF16 不会自动加速这些成本。

memory estimator 对当前 formal config
`src/matrix_factorization/config.yaml` 的 Rademacher 路径也给出：

```text
safe       raw_peak ~= 64.728782 GB
fast       raw_peak ~= 32.467768 GB
aggressive raw_peak ~= 32.467768 GB
```

也就是说，当前 estimator 预测 `safe -> fast` 有明显下降，但
`fast -> aggressive` 没有额外下降。这支持“aggressive 没明显改善”并不一定是
完全没生效，而是默认 Rademacher route 下 fast 已经吃掉了主要 dtype savings。

## Test Coverage Gaps

当前 `tests/test_normalization_precision_contract.py` 覆盖：

- active algorithms 都有 `NormalizationSpec` 和 `PrecisionPolicySpec`；
- default `paper_sparse_sampling` teacher initialization variance 是 `1`；
- `internal_normalized_legacy` teacher initialization variance 是 `1/M`；
- direct tensor teacher default 是 `1`，legacy profile 是 `1/M`；
- invalid `normalization_profile` 会被拒绝；
- invalid `precision_profile` 会被拒绝；
- strict mode 会拒绝冲突的 `precision_profile` 和 legacy `use_bf16`；
- memory-estimation role dtype map 在 aggressive 下会变化。

它没有覆盖：

- 每个 algorithm path 的完整 runtime tensor scale probe；
- AGD、dense BigAMP、spreading、serial tensor、tensor parallel 的 runtime dtype
  assertions；
- `F`、`Y`、state、variances、workspace、reductions、metrics、artifacts 的
  role-by-role dtype summary；
- Rademacher 与 Gaussian `F` 下 fast/aggressive 的差异；
- metric reductions 是否实际保持 FP32；
- memory estimator role dtypes 是否匹配真实 runtime tensors；
- exact graph degree `c=alpha M`。

## Recommendation

当前已经新增显式 YAML-facing profile：

```yaml
algorithm_params:
  normalization_profile: paper_sparse_sampling
```

当前语义：

```text
paper_sparse_sampling:
  latent prior variance = 1
  student init variance = 1
  prior precision base = 1
  interaction scale = lambda / sqrt(M)
  F variance = 1

internal_normalized_legacy:
  latent prior variance = 1/M
  student init variance = 1/M
  prior precision base = M
  interaction scale = 1/sqrt(M)
```

已补上 `ParameterSpec`、effective-parameter trace、metadata visibility 和
基础 scale contract tests。后续仍建议增加轻量 runtime scale/dtype summary
probe，把 scale 和 dtype 写进 run metadata，避免之后只从 config 猜测 profile
是否真实生效。
