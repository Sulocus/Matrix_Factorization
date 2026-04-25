# Artifact Policy

Large experiment data is intentionally excluded from normal source control.

Recommended practice:

- Commit source, small configs, tests, documentation, and selected report figures.
- Keep generated tensors, checkpoints, and full historical result directories in
  ignored local paths.
- Record externally stored artifacts with run id, config path, storage URI, hash,
  tensor shape summary, and the commit that produced them.

Suggested manifest fields:

```json
{
  "run_id": "20260101_1200_bigamp_tensor_parallel",
  "commit": "<git-sha>",
  "config": "configs/local_gpu/tensor_local_gpu.yaml",
  "artifacts": [
    {
      "path": "results.pt",
      "storage": "local-or-object-store-uri",
      "sha256": "<hash>",
      "notes": "Large tensor bundle, not committed"
    }
  ]
}
```
