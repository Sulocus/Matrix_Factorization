# SMF 系统全面修复计划 (完整版)

## 第一部分：当前诊断问题修复

### 问题 1.1：Q_Y 曲线倒置
- [ ] 验证 `compute_qy_spreading_parallel` 计算逻辑
- [ ] 检查 Y_student vs Y_teacher 对比逻辑
- [ ] 修复倒置问题
- [ ] 测试验证

### 问题 1.2：Q_W/Q_X 数值偏低
- [ ] 检查 `compute_cosine_similarity` 实现
- [ ] 对比旧代码
- [ ] 修复后测试

### 问题 1.3：缺失指标
- [ ] 实现 Q_Y_unobserved (关键！区分过拟合)
- [ ] 实现 Q_Y_observed
- [ ] 补全 Q_W_prime, Q_X_prime
- [ ] 补全 MSE, Gen_Error
- [ ] 补全 physical_overlap_W, physical_overlap_X
- [ ] 更新 Runner 返回所有指标
- [ ] 测试验证

---

## 第二部分：所有算法模式完整验证与修复

### 2.1 spreading_parallel 模式
- [ ] 修复 Q_Y 计算
- [ ] 验证边数/alpha_mask
- [ ] 验证 F 分布正确性
- [ ] 验证 BiG-AMP 迭代正确性
- [ ] 与旧代码对比测试

### 2.2 spreading (sequential) 模式
- [ ] 创建诊断测试
- [ ] 验证算法实现
- [ ] 对比 spreading_parallel 结果应一致
- [ ] 修复发现的问题

### 2.3 standard 模式
- [ ] 创建诊断测试
- [ ] 验证当前实现是否完整（注：目前是简化占位符）
- [ ] 如需，实现完整的 Standard BiG-AMP
- [ ] 测试验证

---

## 第三部分：所有 Teacher 类型验证与修复

### 3.1 standard Teacher
- [ ] 验证 Gaussian N(0,1) 初始化
- [ ] 验证 W, X 方差正确
- [ ] 测试

### 3.2 orthogonal Teacher
- [ ] 验证 QR 分解正确性
- [ ] 验证正交性 W^T @ W ≈ I
- [ ] 验证缩放因子
- [ ] 与旧代码对比

### 3.3 scaled_variance Teacher
- [ ] 验证方差缩放 N(0, k/√M)
- [ ] 测试不同 variance_scale 值
- [ ] 验证与 spreading 模式兼容

---

## 第四部分：所有 Graph 类型验证与修复

### 4.1 random Graph
- [ ] 验证边数 = N * round(alpha * M)
- [ ] 验证随机性
- [ ] 测试

### 4.2 uniform Graph (Dinic)
- [ ] 验证双正则性 (每个节点度数相等)
- [ ] 测试边数正确性
- [ ] 与旧代码对比

### 4.3 low_loop Graph (MCMC)
- [ ] 验证 MCMC 边交换逻辑
- [ ] 验证 4-loop 计数减少
- [ ] 测试

---

## 第五部分：UI 完整性验证与修复

### 5.1 参数传递
- [ ] 验证所有 Config 参数正确传递到算法
- [ ] 验证 UI 修改参数后生效

### 5.2 指标显示
- [ ] 验证所有指标能够绘制
- [ ] 验证多指标选择功能
- [ ] 修复指标缺失问题

### 5.3 语言切换
- [ ] 验证中英文切换正常
- [ ] 测试所有标签

---

## 第六部分：与旧代码对比验证

### 6.1 创建对比测试脚本
- [ ] 使用相同 seed (42, 12345)
- [ ] 使用相同参数 (N=200, M=50, alpha 0-4)
- [ ] 运行旧代码和新代码

### 6.2 对比指标
- [ ] Q_Y 曲线对比
- [ ] Q_W, Q_X 曲线对比
- [ ] 运行时间对比
- [ ] 内存使用对比

### 6.3 验收标准
- [ ] 相同参数下结果差异 < 5%
- [ ] 新增功能正常工作

---

## 第七部分：新增功能专项测试

### 7.1 orthogonal Teacher (新增)
- [ ] 单独测试 orthogonal 模式
- [ ] 验证相变曲线形状

### 7.2 scaled_variance Teacher (新增)
- [ ] 不同 variance_scale 的影响
- [ ] 验证物理预期

### 7.3 low_loop Graph (新增)
- [ ] 验证 4-loop 减少效果
- [ ] 对比 random 图的相变

### 7.4 Q_Y_unobserved (新增)
- [ ] 验证能区分过拟合和真学习
- [ ] 验证曲线形状

---

## 执行顺序

1. **第一部分** (当前问题修复) - 优先级最高
2. **第二部分** (算法模式) - 先修复 spreading_parallel
3. **第六部分** (与旧代码对比) - 确认修复有效
4. **第三部分 + 第四部分** (Teacher + Graph)
5. **第五部分** (UI)
6. **第七部分** (新增功能)

---

## 每步测试原则

1. **修改前**：记录当前行为
2. **修改**：最小化改动
3. **测试**：运行相关测试脚本
4. **验证**：确认问题解决且无新问题
5. **记录**：更新 task.md 检查项

---

**状态**：等待确认后开始执行第一部分
