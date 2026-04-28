# Next Agent Prompt: Scan Plot / Gauge / Adaptive Onsager 收尾

把下面这段完整发给新 agent：

```text
你接手的是 /home/sucia/Matrix_Factorization，请在 dev 分支工作。

先阅读这些文件，不要直接开始乱改：

1. /home/sucia/Matrix_Factorization/AGENTS.md
2. /home/sucia/Matrix_Factorization/docs/project_handoff_2026-04-28.md
   特别是 “11. 2026-04-28 05:00 scan / plotting / gauge / adaptive 状态”
3. /home/sucia/Matrix_Factorization/docs/METRICS_GUIDE.md
4. /home/sucia/Matrix_Factorization/docs/scan_system_contract.md
5. /home/sucia/Matrix_Factorization/docs/theory/bigamp_onsager_damping_audit.md

当前用户最关心的是：scan 输出、绘图、gauge diagnostic、adaptive Onsager 稳定性。

背景：

- 正式 run 目录：
  /home/sucia/Matrix_Factorization/runs/20260428_043906_bgs_N200_M50_ons3-a41_S100_steps2000_a4258e
- no_onsager 和 onsager_fixed_beta005 两组已完成。
- onsager_adaptive_beta005 组在正式尺寸失败。
- no_onsager 的普通图是临时补图，不是正式 publication-style plotting。
- 用户希望：
  - 每个独立 scan group 完成后都生成完整普通图；
  - 最终跨组对比图仍在总 run 完成后生成；
  - sign-aligned 图默认能画；
  - `Q_w gauged` 即 scale-gauge aligned diagnostic 能被结果体系识别；
  - heatmap 至少能画 sign-aligned W，而不是旧 raw/Gram fallback；
  - adaptive compile/driver 问题要解释清楚，不要为了跑通随意关 compile。

当前未提交代码已经做过一些尝试，不要直接 revert：

- src/matrix_factorization/modules/algorithms/bigamp/spreading.py
  - adaptive_damping=true 时禁用 torch.compile；
  - fixed/no-Onsager 保留 legacy fast compiled path。
- src/matrix_factorization/core/experiment/runner.py
  - group-level PlotQuery 不再因为 series_by 轴被 group 固定而跳过。
- src/matrix_factorization/core/experiment/result.py
  - heatmap_metric 支持 Q_W_SIGN_ALIGNED。
- src/matrix_factorization/config.yaml
  - output.heatmap_metric 改为 Q_W_SIGN_ALIGNED。

你需要先做这些检查：

```bash
cd /home/sucia/Matrix_Factorization
git status --short
python -m pytest -q tests/test_result_cube.py tests/test_plot_query.py tests/test_scan_runner.py
python -m pytest -q tests/test_bigamp_onsager_convention.py tests/test_spreading_batch_metrics.py tests/test_config_contract.py
mf validate src/matrix_factorization/config.yaml
```
任务 A：正式修复 group-level plotting

- 确认 group-level `ExperimentResult.save()` 能触发 `_plot_result_cube_queries()`。
- 对 fixed `series_by` 轴，例如 group 已固定 `onsager_policy=no_onsager`，不要跳过 plot，而是剥离该 series axis，生成单组曲线。
- 图必须走 publication-style PlotQuery，不要使用临时 matplotlib 手写脚本。
- 每个 group 完成后应生成：
  - Q_Y plot
  - Q_W raw projection plot
  - Q_X raw projection plot
  - Q_W_COS_ROOT plot
  - Q_X_COS_ROOT plot
  - Q_W_SIGN_ALIGNED plot
  - Q_X_SIGN_ALIGNED plot
- 图要带 errorbar，使用 `*_std`。

任务 B：把 sign heatmap 做成正式选项

- `output.heatmap_metric: Q_W_SIGN_ALIGNED` 应生成 sign-aligned W heatmap。
- 文件名应能区分旧 Gram heatmap 和 sign heatmap，例如：
  - `heatmap_W_sign_...png`
- heatmap title / colorbar 应清楚标注 `Q_W_SIGN_ALIGNED`。
- 添加测试覆盖该选项，不要只靠手动运行。

任务 C：整理 `Q_w gauged`

- 现有脚本：
  /home/sucia/Matrix_Factorization/scripts/analysis/posthoc_scale_gauge.py
- 当前问题：脚本只按 `artifacts/points/<point_id>/results.pt` 找文件，但 group result 里路径是 `artifacts/points/<alpha>/results.pt`。
- 先修脚本，让它同时支持：
  - ResultCube point_id 路径；
  - alpha 路径；
  - 从 `result_cube.points[point_id].coordinates.alpha` 反查 alpha 路径。
- 输出：
  - `scale_gauge_metrics.csv`
  - `scale_gauge_metrics.json`
  - `qw_scale_gauge.png`
  - `qx_scale_gauge.png`
  - `qwx_scale_gauge.png`
  - `gauge_magnitude.png`
- 短期不要把 scale-gauge 直接塞进训练主 metric；它依赖 saved W/X tensor，成本较高。更合理的是作为 posthoc Analyzer 或 OutputSpec 依赖 `matrix_factors`。
- 如果要提升为正式功能，要加 contract/spec，不允许软接。

任务 D：Adaptive Onsager 稳定性

- 不要盲跑完整 adaptive 大 scan。
- 先复现并定位：
  - full config: S=100, steps=2000, alpha batch 0.00-1.40 会 driver error。
  - tiny debug: S=2, steps=2, alpha=0.0..0.1 safe/aggressive 都通过。
- 逐步测试：
  - S=100, steps=2, alpha=[0.0]
  - S=100, steps=50, alpha=[0.0]
  - S=100, steps=2000, alpha=[0.0]
  - alpha batch size 1/2/4/8
  - precision safe/fast/aggressive
- 用 `CUDA_LAUNCH_BLOCKING=1` 获取准确报错。
- 如果单 alpha 稳定，说明是 alpha batch 过宽或 planner 对 adaptive 的 batch 约束不够硬；把 adaptive 的 BatchingSpec 收紧。
- 如果单 alpha 也失败，说明 adaptive implementation 本身有大尺寸问题，优先审计 `_train_full_parallel_adaptive()` 的 live tensors、acceptance state、`prev_s/prev_svar`、BF16 dtype。
- 不要把 adaptive 写成 completed，除非正式尺寸通过。

任务 E：文档状态

- 更新 /home/sucia/Matrix_Factorization/docs/project_handoff_2026-04-28.md
- 明确哪些是 completed，哪些只是 attempted，哪些仍 failing。
- 不要提交 runs/results/artifacts 或 `.pt`。

验收最低要求：

```bash
python -m pytest -q tests/test_result_cube.py tests/test_plot_query.py tests/test_scan_runner.py
python -m pytest -q tests/test_bigamp_onsager_convention.py tests/test_spreading_batch_metrics.py tests/test_config_contract.py
mf validate src/matrix_factorization/config.yaml
```

如果要继续跑 GPU，只先跑 quick/small validation，不要直接开完整三组大 scan。
```
