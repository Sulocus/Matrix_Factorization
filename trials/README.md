# Research Trials

`trials/` 存放研究试跑和调试试跑的轻量 manifest、专用参数文件和摘要。

这里不是正式 pytest，也不是大实验结果目录。目标是让 agent 做“看一下能不能跑”“调一下小参数”时有固定入口：

- 只改 `trials/active/<trial_key>/config.yaml`。
- 不改 `src/matrix_factorization/config.yaml`，除非用户明确要求修改正式默认配置。
- 结果只写到 `artifacts/trials/`。
- `runtime_class: quick` 可以用 `mf trial run <key>` 运行。
- `runtime_class: medium` 和 `gpu_heavy` 可以登记、解释和校验，但默认不能由 `mf trial run` 自动启动。

常用命令：

```bash
mf trial list
mf trial explain matrix_bigamp_quick
mf trial validate matrix_bigamp_quick
mf trial run matrix_bigamp_quick
```
