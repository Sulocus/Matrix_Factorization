# SMF 任务交接文档

**最后更新**: 2025-12-08 (第十五次更新)
**上次会话**: 随机扩频模块 (Random Spreading Module) 完整实现

---

## 当前任务状态

### 已完成

- [x] **随机扩频模块完整实现**（本次会话）
  - `smf/modules/teachers/random_spreading.py` - Teacher 模块
  - `smf/modules/algorithms/bigamp_spreading.py` - BiG-AMP 算法
  - `smf/modules/metrics/spreading.py` - 评估指标
  - `tests/test_random_spreading.py` - 31 个物理正确性验证测试
- [x] **docs/core/ 文档补全**（上次会话）
- [x] **CLAUDE.md 精简重构**
- [x] **/pass /rem 命令增强**

### 待开始

- [ ] 恢复误删文件：results_db.py, queue_manager.py, analysis/compare.py
- [ ] 实际运行 `smf run` 测试完整流程
- [ ] 创建随机扩频模块文档 (`docs/modules/teachers/random_spreading.md`)

---

## 本次会话关键变更

### 新增文件

| 文件 | 说明 |
|------|------|
| `smf/modules/teachers/random_spreading.py` | 随机扩频 Teacher，包含 SpreadingData, F 生成, Y 计算 |
| `smf/modules/algorithms/bigamp_spreading.py` | BiG-AMP 稀疏消息传递算法 |
| `smf/modules/metrics/spreading.py` | compute_qy_spreading, compute_all_metrics_spreading |
| `tests/test_random_spreading.py` | 31 个测试，覆盖 F 确定性、Y 计算、消息传递、Q_Y 评估 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `smf/modules/teachers/__init__.py` | 导出 RandomSpreadingTeacher, SpreadingData |
| `smf/modules/algorithms/__init__.py` | 导出 BiGAMPSpreadingAlgorithm |
| `smf/modules/metrics/__init__.py` | 导出 spreading 指标函数 |

### 物理模型

**原模型**：
```
Y_ij = Σ_μ W_iμ X_μj
```

**随机扩频模型**：
```
Y_ij = (1/√M) Σ_μ F_ij,μ W_iμ X_μj
```
其中 F_ij,μ ~ N(0,1) 是淬火随机系数。

---

## 测试验证

### 31 个测试全部通过

| 测试类别 | 数量 | 验证内容 |
|----------|------|----------|
| F 确定性 | 4 | 同种子→同F，统计分布 N(0,1) |
| Y 计算 | 4 | 小规模手动验证 + 向量化 vs 循环 |
| Teacher 类 | 4 | create, create_with_spreading, 可复现性 |
| SpreadingData | 2 | to(), clone() |
| scatter_add | 2 | dim=0, dim=1 聚合 |
| BiG-AMP 消息传递 | 4 | r_W, r_X, tau_W, z_hat 计算 |
| BiG-AMP 集成 | 2 | 单步运行，完美初始化收敛 |
| Q_Y 评估 | 6 | 完美恢复→1，错误F→低，批量 |
| 渐进恢复 | 1 | 噪声↓→Q_Y↑ |

---

## 关键文件位置

| 文件 | 用途 |
|------|------|
| `smf/modules/teachers/random_spreading.py` | 核心 Teacher 模块 |
| `smf/modules/algorithms/bigamp_spreading.py` | BiG-AMP 算法 |
| `smf/modules/metrics/spreading.py` | 评估指标 |
| `tests/test_random_spreading.py` | 测试文件 |
| `/home/sucia/.claude/plans/expressive-sniffing-abelson.md` | 原始计划文件 |

---

## 使用示例

```python
from smf.modules.teachers import RandomSpreadingTeacher
from smf.modules.algorithms import BiGAMPSpreadingAlgorithm
from smf.modules.metrics import compute_qy_spreading

# 创建 Teacher
teacher = RandomSpreadingTeacher(spreading_seed=12345)
i_idx, j_idx = ...  # 观测边索引
W, X, spreading_data = teacher.create_with_spreading(
    N1, N2, M, i_idx, j_idx, device, seed=42
)

# 训练
algorithm = BiGAMPSpreadingAlgorithm(config, device)
W_s, X_s = algorithm.train_single_alpha_spreading(
    W, X, spreading_data, alpha, seed
)

# 评估
Q_Y = compute_qy_spreading(W_s, X_s, spreading_data)
```

---

## 恢复命令

下一个对话使用:
```
/rem
```
即可恢复上下文。

---

*本文档由 Claude Code 在 2025-12-08 自动更新（第十五次）*
