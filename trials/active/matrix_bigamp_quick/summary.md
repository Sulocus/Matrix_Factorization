# matrix_bigamp_quick

用途：验证 Research Trial 的最小运行链路。

这个 trial 使用小尺寸 matrix BiGAMP，少量 alpha 点和 2 个 AMP step。它只用于确认：

- trial 专用 config 被读取。
- `mf trial validate` 能通过。
- `mf trial run` 能生成 `config.json`、`metadata.json`、`metrics.json`、`manifest.json`。
- 输出只进入 `runs/trials/matrix_bigamp_quick/`。
