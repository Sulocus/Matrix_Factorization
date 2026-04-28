# Local Workflow

本项目现在只维护本地开发和本地 GPU 验证流程；`quick` 只表示小规模快速试跑，不表示远端或容器环境。

## Checks

```bash
python -m pytest -q
mf validate src/matrix_factorization/config.yaml
mf explain-config src/matrix_factorization/config.yaml
```

## Quick Trials

临时调参和一分钟级真实试跑使用 trial 工作流：

```bash
mf trial list
mf trial validate matrix_bigamp_quick
mf trial run matrix_bigamp_quick
```

trial 参数只改 `trials/active/<trial_key>/config.yaml`，输出写入被 ignore 的 `artifacts/trials/`。

## Local GPU Calibration

显存估计需要本地 GPU 实测校准：

```bash
mf calibrate memory list
mf calibrate memory explain matrix_bigamp_target_10gb
mf calibrate memory run matrix_bigamp_target_10gb
```

校准 raw artifact 写入 `runs/calibration/memory/`。`*_small` profile 只检查记录链路；真正用于并行 planner 的 profile 从 10GB 级别开始，运行时会保存 `memory_timeline.jsonl`，并用扣除 baseline 后的 peak allocated 更新 `runs/calibration/memory/latest_coefficients.json`。这个系数文件是本地状态，不提交。

常用本地校准顺序：

```bash
mf calibrate memory run matrix_bigamp_target_10gb
mf calibrate memory run matrix_bigamp_target_16gb
mf calibrate memory run spreading_bigamp_target_10gb
mf calibrate memory run tensor_serial_target_6gb
mf calibrate memory run tensor_parallel_target_6gb
mf calibrate memory run matrix_agd_target_10gb
```

如果某个 profile 在实测后被判定超过当前 free VRAM 的安全比例，CLI 会在运行前拒绝，而不是硬跑到 OOM。

## Run Artifacts

canonical run 目录：

```text
runs/{run_id}/
  manifest.json
  config.json
  metrics.json
  events.jsonl
  artifacts/
    results.pt
  plots/
  checkpoints/
    latest.pt
```

`results/latest/` 只作为展示快照，不作为正式分析输入。
