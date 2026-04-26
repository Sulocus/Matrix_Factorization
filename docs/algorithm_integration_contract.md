# Algorithm 集成状态地图

这份文档回答一个问题：仓库里每个 registered algorithm 到底处于什么状态。它不删除、不移动、不合并任何代码，只把主链路、reference path、legacy path 和 helper path 的边界写硬。

机器定义在 `src/matrix_factorization/core/contracts.py` 的 `AlgorithmSpec`。这份文档是人读版本；测试会检查每个 `AlgorithmSpec.key` 都出现在这里。

## active 主链路

```text
agd
  status: active
  result_contract: legacy_matrix_result
  active_result_path: native_agd_algorithm_result
  CLI/config: algorithm=3, tensor_order=2 matrix path
  role: matrix AGD optimizer
  merge/delete: 不删除；可继续硬化 runtime hooks 和 metric compute adapters

bigamp
  status: active
  result_contract: legacy_matrix_result
  active_result_path: native_bigamp_algorithm_result
  CLI/config: algorithm=1, tensor_order=2 matrix path
  role: dense matrix BiGAMP
  merge/delete: 不删除；可继续硬化 metrics 和 memory contract

bigamp_spreading
  status: active
  result_contract: legacy_spreading_result
  active_result_path: native_bigamp_spreading_algorithm_result
  CLI/config: algorithm=2 spreading path
  role: random spreading matrix BiGAMP
  merge/delete: 不删除；adaptive restart/intervention 和 spreading seed contract 仍需后续硬化

bigamp_tensor
  status: active
  result_contract: legacy_tensor_metrics_only
  CLI/config: serial/reference tensor path；当前普通 tensor_order>=3 主路由不是它
  role: tensor serial/reference implementation
  merge/delete: 不删除；长期目标是和 bigamp_tensor_parallel 做 parity 后统一为 serial/parallel backend

bigamp_tensor_parallel
  status: active
  result_contract: legacy_tensor_metrics_only
  CLI/config: tensor_order>=3 当前主 tensor 路径
  role: tensor spreading parallel implementation, heatmap/overlap_matrix producer
  merge/delete: 不删除；长期目标是和 bigamp_tensor 统一物理 contract
```

## 非主链路但已注册

```text
agd_tensor
  status: experimental_unintegrated
  result_contract: experimental_private_result
  CLI/config: 当前主 CLI 不应默认路由到它
  role: experimental tensor AGD
  merge/delete: 先保留；要进入主链路必须补 AlgorithmSpec/MetricSpec/OutputSpec parity 和 result schema

agd_spreading
  status: legacy_broken
  result_contract: legacy_private_result
  CLI/config: legacy reference file under algorithms/legacy
  role: broken/old spreading AGD reference
  merge/delete: 先保留为 legacy/reference；不作为 active runner contract

combined
  status: non_trainable_helper
  result_contract: not_a_training_algorithm
  CLI/config: 不应作为 trainable algorithm 进入 runner
  role: selector/helper
  merge/delete: 先保留；未来可考虑移出 algorithm registry，但这会影响 registry contract，需单独做
```

## 去重原则

- `active` 不等于“物理正确已确认”，只表示它能被主链路识别并有 contract。
- `legacy_broken` 和 `experimental_unintegrated` 可以注册，但默认 runner 不应静默执行。
- `combined` 这种 helper 不应伪装成 trainable algorithm；如果以后保留，应考虑拆到 selector/helper 层。
- `bigamp_tensor` 和 `bigamp_tensor_parallel` 不应直接删除任何一个；应先完成 `docs/tensor_parity_contract.md` 中的 parity checklist。
- 新增 algorithm 文件时，必须同时更新 `AlgorithmSpec`、source inventory 和本文件，否则测试失败。
