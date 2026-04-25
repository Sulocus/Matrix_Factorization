# Experiment Configs

This directory contains repo-tracked YAML configs that are safe to share with
Codex web.

- `smoke/`: small CPU or small-GPU checks for imports, output schema, and CLI wiring.
- `local_gpu/`: larger research runs that should be launched on the local machine.

Large generated tensors, checkpoints, and historical result directories stay in
ignored `runs/`, `results/`, `artifacts/`, or local external storage.
