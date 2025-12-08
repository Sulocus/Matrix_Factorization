# SMF 任务交接文档

**最后更新**: 2025-12-08 (第十六次更新)
**上次会话**: 项目文件系统重组 + CLAUDE.md 重构

---

## 当前任务状态

### 已完成（本次会话）

- [x] **CLAUDE.md 重构**
  - 添加标准 Claude Code 头部
  - 添加常用命令（pip install, pytest, smf CLI）
  - 更新项目结构表格
  - 添加 Git 分支策略说明

- [x] **项目文件系统重组**
  - `docs/` → `smf_docs/`（重命名）
  - `scripts/` → `smf/scripts/`（移动）
  - `phase_analysis/` → `_legacy/phase_analysis/`（归档）
  - `New module.md` → `prompt.md`（重命名）

- [x] **冗余文件清理**（共清理 ~3MB）
  - 删除 `results/`（空数据库）
  - 删除 `tests/*.json`（中间测试结果）
  - 删除 `test_progress*.md`（过时日志）
  - 删除所有 `__pycache__/`（22个目录）

- [x] **Git 分支配置**
  - main 分支：`README.md` → `README_Japanese.md`
  - dev 分支：删除 README.md
  - 已推送两个分支到远程

### 待开始

- [ ] 恢复误删文件：results_db.py, queue_manager.py, analysis/compare.py
- [ ] 实际运行 `smf run` 测试完整流程
- [ ] 创建随机扩频模块文档 (`smf_docs/modules/teachers/random_spreading.md`)

---

## 本次会话关键变更

### 文件移动/重命名

| 原路径 | 新路径 |
|--------|--------|
| `docs/` | `smf_docs/` |
| `scripts/` | `smf/scripts/` |
| `phase_analysis/` | `_legacy/phase_analysis/` |
| `New module.md` | `prompt.md` |
| `README.md` (main) | `README_Japanese.md` (main only) |

### 删除的文件

| 路径 | 原因 |
|------|------|
| `results/experiments.db` | 空数据库，相关代码已删 |
| `tests/*.json` | 中间测试结果 |
| `test_progress*.md` | 过时开发日志 |
| `__pycache__/` (22个) | 编译缓存 |

### Git 提交记录

```
main: 607703d - Rename README.md to README_Japanese.md
dev:  7123671 - Major project reorganization and cleanup
```

---

## 当前项目结构

```
Sparse-Matrix/           (dev 分支)
├── Wang/                 # 生产代码 (Git tracked)
├── smf/                  # 模块化框架
│   ├── core/
│   ├── modules/
│   ├── scripts/          # ← 从根目录移入
│   └── ui/
├── smf_docs/             # ← 原 docs/
├── _legacy/
│   └── phase_analysis/   # ← 从根目录归档
├── tests/
├── share/
├── CLAUDE.md
├── HANDOVER.md
├── prompt.md             # ← 原 New module.md
└── pyproject.toml
```

---

## 远程仓库信息

**仓库地址**: `https://github.com/Sulocus/Sparse-Matrix-Factorization.git`

（注意：仓库已从 Sparse-Matrix 迁移到 Sparse-Matrix-Factorization）

---

## 恢复命令

下一个对话使用:
```
/rem
```
即可恢复上下文。

---

*本文档由 Claude Code 在 2025-12-08 自动更新（第十六次）*
