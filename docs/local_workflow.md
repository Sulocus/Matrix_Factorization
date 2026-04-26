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

trial 参数只改 `trials/active/<trial_key>/config.yaml`，输出写入被 ignore 的 `runs/trials/`。

## Local GPU Calibration

显存估计需要本地 GPU 实测校准：

```bash
mf calibrate memory list
mf calibrate memory explain matrix_bigamp_small
mf calibrate memory run matrix_bigamp_small
```

校准 raw artifact 写入 `runs/calibration/memory/`。若之后需要把校准系数纳入版本管理，只提交轻量 summary 或系数文件，不提交完整运行结果。

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
