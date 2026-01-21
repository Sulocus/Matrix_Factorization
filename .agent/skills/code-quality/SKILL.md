---
name: code-quality
description: Python 代码质量分析与优化。使用 vulture 检测死代码、ruff 检测代码风格问题、radon 分析复杂度。帮助清理冗余代码、未使用的导入、过度复杂的函数。
---

# Code Quality (代码质量分析与优化)

## 何时使用此 Skill

- 清理未使用的导入和变量
- 检测死代码（从未调用的函数/类）
- 识别过度复杂的函数（需要重构）
- 代码格式化和风格统一
- 代码审查前的自动检查

## 工具概览

### 1. Vulture - 死代码检测

检测未使用的代码（函数、类、变量、导入）。

```bash
# 基本使用
vulture <directory> --min-confidence 80

# 输出示例
# file.py:10: unused function 'old_func' (100% confidence)
# file.py:25: unused import 'os' (90% confidence)
```

**置信度说明**:
- 100%: 绝对确定是死代码
- 60-90%: 可能是死代码，需人工确认

### 2. Ruff - 超快 Linter + Formatter

替代 flake8, black, isort, pyupgrade 等多个工具。

```bash
# 检查问题
ruff check <directory>

# 按规则统计
ruff check <directory> --statistics

# 自动修复安全问题
ruff check <directory> --fix

# 格式化代码
ruff format <directory>
```

**常见规则**:
- `F401`: 未使用的导入
- `F841`: 未使用的变量
- `E501`: 行太长
- `I001`: 导入顺序错误

### 3. Radon - 复杂度分析

分析代码复杂度，识别需要重构的函数。

```bash
# Cyclomatic Complexity (圈复杂度)
radon cc <directory> -a -s

# 只显示高复杂度 (>= C 级别，即 >= 11)
radon cc <directory> -a -s --min C

# Maintainability Index (可维护性指数)
radon mi <directory> -s
```

**复杂度等级**:
- A (1-5): 简单，低风险
- B (6-10): 稍复杂，低风险
- C (11-20): 较复杂，中风险 ⚠️
- D (21-30): 复杂，高风险 🔴
- E (31-40): 非常复杂 🔴
- F (41+): 极度复杂，需立即重构 🔴

## 工作流程

### 快速检查

```bash
# 1. 检测死代码
vulture <directory> --min-confidence 80

# 2. 检测风格问题
ruff check <directory> --statistics

# 3. 检测复杂度
radon cc <directory> -a -s --min C
```

### 自动修复

```bash
# 自动修复 Ruff 可修复的问题
ruff check <directory> --fix

# 格式化
ruff format <directory>
```

### 手动优化

对于 Vulture 和 Radon 发现的问题，需要人工处理：

1. **死代码**: 确认后删除
2. **高复杂度函数**: 拆分为多个小函数
3. **未使用导入**: 删除或确认是否为动态导入

## 配置文件

在 `pyproject.toml` 中配置：

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]  # 忽略行长度

[tool.vulture]
min_confidence = 80
paths = ["MF"]
```

## 与 CI/CD 集成

```yaml
# .github/workflows/code-quality.yml
name: Code Quality
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install ruff vulture radon
      - run: ruff check . --exit-non-zero-on-fix
      - run: vulture . --min-confidence 100
```

## 最佳实践

1. **定期运行**: 每次 PR 前运行完整检查
2. **渐进式修复**: 不要一次修复所有问题
3. **保留假阳性白名单**: 对于动态调用的代码，创建 whitelist
4. **关注复杂度趋势**: 新代码复杂度应 <= B 级

## 参考资源

- [Ruff Docs](https://docs.astral.sh/ruff/)
- [Vulture PyPI](https://pypi.org/project/vulture/)
- [Radon Docs](https://radon.readthedocs.io/)
