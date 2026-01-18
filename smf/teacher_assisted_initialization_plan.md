# Teacher-Assisted Initialization Implementation Plan
## 亚稳态生存测试 (Metastability Check) 功能集成

> **目标**: 在现有 BiG-AMP Spreading 算法中实现 "从 Teacher 附近初始化" (Warm Start) 功能，用于验证相变区域是否存在物理解。

---

## 第一部分：项目结构分析

### 1.1 程序流程总览

```mermaid
flowchart TD
    A[config.yaml] --> B[cli.py main()]
    B --> C[load_yaml_config()]
    C --> D[ExperimentConfig 数据类]
    D --> E[ExperimentRunner.run()]
    E --> F[DataFactory.create()]
    F --> G[create_teacher W, X, Y]
    F --> H[create_spreading_data]
    H --> I[SpreadingDataParallel]
    I --> J[BiGAMPSpreading.train_full_parallel]
    J --> K[W_hat, X_hat 输出]
    K --> L[_compute_metrics 计算指标]
    L --> M[ExperimentResult]
    M --> N[plotting.py 绘图]
```

### 1.2 核心文件清单

| 文件 | 职责 |
|------|------|
| `smf/config.yaml` | 唯一配置入口，定义所有实验参数 |
| `smf/cli.py` | CLI 入口，解析 YAML 并构建 `ExperimentConfig` |
| `smf/core/experiment/config.py` | 定义 `AlgorithmParams`, `SpreadingConfig` 等数据类 |
| `smf/core/experiment/runner.py` | 实验执行引擎，调用算法并收集结果 |
| `smf/core/experiment/data_factory.py` | 创建 Teacher 矩阵和 SpreadingData |
| `smf/modules/algorithms/bigamp_spreading.py` | 核心算法实现，包含三个训练循环 |
| `smf/modules/teachers/random_spreading.py` | 定义 `SpreadingDataParallel` 数据结构 |
| `smf/modules/outputs/plotting.py` | 结果绘图模块 |

### 1.3 初始化代码位置

当前的随机初始化代码位于 `bigamp_spreading.py` 的三个位置：

1. **`train_full_parallel`** (Line 1868-1871) - Bipartite 模式
   ```python
   W_flat = torch.randn(B, S * N1, M, device=...) * 0.1
   X_flat = torch.randn(B, S * N2, M, device=...) * 0.1
   ```

2. **`_train_full_parallel_adaptive`** (Line 2007-2010) - Adaptive Damping 模式
   ```python
   W_flat = torch.randn(B, S * N1, M, device=...) * 0.1
   X_flat = torch.randn(B, S * N2, M, device=...) * 0.1
   ```

3. **`_train_full_parallel_general`** (Line 2421-2422) - General Graph 模式
   ```python
   V_flat = torch.randn(B, S * N_total, M, device=...) * 0.1
   ```

### 1.4 关键数据结构

**SpreadingDataParallel** (定义于 `random_spreading.py:81-144`):
```python
@dataclass
class SpreadingDataParallel:
    supergraph: SuperGraphData
    F_super: torch.Tensor       # (S, C_max, M)
    Y_super: torch.Tensor       # (S, C_max)
    M: int
    alpha_values: torch.Tensor  # (A,)
    W_teacher: torch.Tensor     # (N1, M) ★ 已保存 Teacher
    X_teacher: torch.Tensor     # (M, N2) ★ 已保存 Teacher
```

> **关键发现**: `SpreadingDataParallel` 已经保存了 `W_teacher` 和 `X_teacher`，无需额外传递！

---

## 第二部分：理论背景

### 2.1 问题描述

在一阶相变 (First-order Phase Transition) 中，能量景观存在两个极小值：
- **坑 A (m=0)**: 随机猜测状态，无信息
- **坑 B (m>0)**: 接近 Ground Truth 的正确状态

**当前问题**: 从坑 A 出发（随机初始化），算法可能无法翻越能垒到达坑 B，导致在 α=2 附近误判为 "无解"。

### 2.2 解决方案：迟滞分析 (Hysteresis Analysis)

1. **Cold Start (现有)**: 随机初始化 → 扫描增加 α
2. **Warm Start (新增)**: 从 Teacher 附近初始化 → 扫描减小 α

**预期结果**:
- 如果 α=2 时 Warm Start 能保持高 overlap，说明解存在（只是 Cold Start 找不到）
- 真正的物理相变点在两条曲线分离的地方

### 2.3 初始化公式

为保持方差归一化：
```
W_init = m_init × W_teacher + √(1 - m_init²) × noise
```
其中 `m_init ∈ [0.9, 0.99]` 是目标初始重叠度。

---

## 第三部分：实施步骤

### Step 1. 扩展 AlgorithmParams 配置

**文件**: `smf/core/experiment/config.py`

在 `AlgorithmParams` 数据类中添加两个新字段：

```python
@dataclass
class AlgorithmParams:
    # ... 现有字段 ...
    
    # Teacher-Assisted Initialization (NEW)
    init_mode: str = "random"      # "random" 或 "teacher"
    init_overlap: float = 0.95     # 初始重叠度 m_init ∈ (0, 1)
```

---

### Step 2. 修改 YAML 配置加载

**文件**: `smf/cli.py`

在 `load_yaml_config()` 函数中添加对新字段的解析：

```python
# 在 algorithm_params 解析部分 (约 Line 120-150)
init_mode = algo.get('init_mode', 'random')
init_overlap = algo.get('init_overlap', 0.95)
```

---

### Step 3. 实现初始化辅助函数

**文件**: `smf/modules/algorithms/bigamp_spreading.py`

在 `BiGAMPSpreading` 类中添加静态方法：

```python
@staticmethod
def _initialize_near_teacher(
    target_shape: Tuple[int, int, int],  # (B, S*N, M)
    teacher_tensor: torch.Tensor,         # (N, M) or (M, N)
    m_init: float,
    device: torch.device,
    dtype: torch.dtype,
    is_X: bool = False,  # True if X (M, N2), else W (N1, M)
) -> torch.Tensor:
    """
    生成接近 Teacher 的初始估计。
    
    公式: W_init = m_init * W_teacher + sqrt(1 - m_init^2) * noise
    确保初始重叠度约为 m_init，且方差保持不变。
    """
    B, SN, M_dim = target_shape
    
    if is_X:
        # X_teacher: (M, N2) -> (N2, M) for broadcasting
        N = teacher_tensor.shape[1]
        teacher_flat = teacher_tensor.T  # (N2, M)
    else:
        # W_teacher: (N1, M)
        N = teacher_tensor.shape[0]
        teacher_flat = teacher_tensor  # (N1, M)
    
    S = SN // N
    
    # Broadcast: (1, 1, N, M) -> (B, S, N, M) -> (B, S*N, M)
    teacher_expanded = teacher_flat.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
    teacher_flat = teacher_expanded.reshape(B, SN, M_dim).to(device, dtype=dtype)
    
    # Generate noise
    noise = torch.randn(target_shape, device=device, dtype=dtype)
    
    # Combine with variance preservation
    coeff_signal = m_init
    coeff_noise = math.sqrt(1 - m_init ** 2)
    
    return coeff_signal * teacher_flat + coeff_noise * noise
```

---

### Step 4. 修改 train_full_parallel (Bipartite)

**文件**: `smf/modules/algorithms/bigamp_spreading.py`
**位置**: Line 1865-1871

替换初始化代码：

```python
# ===== TEACHER-ASSISTED INITIALIZATION =====
init_mode = getattr(self.config.algorithm_params, 'init_mode', 'random')
init_overlap = getattr(self.config.algorithm_params, 'init_overlap', 0.95)

if init_mode == 'teacher':
    W_flat = self._initialize_near_teacher(
        (B, S * N1, M),
        spreading_data.W_teacher,
        init_overlap,
        self.device,
        self.storage_dtype,
        is_X=False
    )
    X_flat = self._initialize_near_teacher(
        (B, S * N2, M),
        spreading_data.X_teacher.T,  # (M, N2) -> (N2, M) handled inside
        init_overlap,
        self.device,
        self.storage_dtype,
        is_X=True
    )
    if verbose:
        print(f"  [Init] Teacher-Assisted (m={init_overlap})")
else:
    W_flat = torch.randn(B, S * N1, M, device=self.device, dtype=self.storage_dtype) * 0.1
    X_flat = torch.randn(B, S * N2, M, device=self.device, dtype=self.storage_dtype) * 0.1
```

---

### Step 5. 修改 _train_full_parallel_adaptive

**文件**: `smf/modules/algorithms/bigamp_spreading.py`
**位置**: Line 2006-2010

应用与 Step 4 相同的模式。

---

### Step 6. 修改 _train_full_parallel_general

**文件**: `smf/modules/algorithms/bigamp_spreading.py`
**位置**: Line 2419-2422

General 模式使用统一向量 V，需要特殊处理：

```python
# ===== TEACHER-ASSISTED INITIALIZATION (General Mode) =====
init_mode = getattr(self.config.algorithm_params, 'init_mode', 'random')
init_overlap = getattr(self.config.algorithm_params, 'init_overlap', 0.95)

if init_mode == 'teacher':
    # 构造统一 Teacher 向量
    W_teacher = spreading_data.W_teacher  # (N1, M)
    X_teacher = spreading_data.X_teacher.T  # (N2, M)
    V_teacher = torch.cat([W_teacher, X_teacher], dim=0)  # (N_total, M)
    
    V_flat = self._initialize_near_teacher(
        (B, S * N_total, M),
        V_teacher,
        init_overlap,
        self.device,
        self.storage_dtype,
        is_X=False
    )
    if verbose:
        print(f"  [Init] Teacher-Assisted General (m={init_overlap})")
else:
    V_flat = torch.randn(B, S * N_total, M, device=self.device, dtype=self.storage_dtype) * 0.1
```

---

### Step 7. 更新 config.yaml 示例

在 `algorithm_params` 部分添加新配置项：

```yaml
algorithm_params:
  # ... 现有配置 ...
  
  # Teacher-Assisted Initialization (Hysteresis Analysis)
  init_mode: random       # "random" = Cold Start, "teacher" = Warm Start
  init_overlap: 0.95      # Initial overlap with teacher (0.9 - 0.99)
```

---

## 第四部分：验证计划

### 4.1 单元测试

创建新测试文件 `smf/tests/test_teacher_init.py`：

```python
def test_initialize_near_teacher_overlap():
    """验证初始化后的重叠度接近目标值"""
    # 1. 创建 Teacher
    # 2. 调用 _initialize_near_teacher
    # 3. 计算 cosine overlap
    # 4. 断言 overlap ≈ m_init ± 0.05
```

**运行命令**:
```bash
cd /home/sucia/Sparse-Matrix
python -m pytest smf/tests/test_teacher_init.py -v
```

### 4.2 功能测试

**Cold Start vs Warm Start 对比实验**:

1. 创建配置文件 `smf/presets/hysteresis_cold.yaml`:
   ```yaml
   algorithm_params:
     init_mode: random
   alpha_scan:
     start: 0
     stop: 4
     step: 0.2
   ```

2. 创建配置文件 `smf/presets/hysteresis_warm.yaml`:
   ```yaml
   algorithm_params:
     init_mode: teacher
     init_overlap: 0.99
   alpha_scan:
     start: 0
     stop: 4
     step: 0.2
   ```

3. 运行对比：
   ```bash
   smf smf/presets/hysteresis_cold.yaml
   smf smf/presets/hysteresis_warm.yaml
   ```

4. 比较两次实验的 Q_Y 曲线：
   - Cold Start: 在 α < 3 应该 overlap ≈ 0
   - Warm Start: 在 α > 1.5 应该 overlap ≈ 1（如果解存在）

### 4.3 手动验证步骤

1. **验证配置加载**:
   ```bash
   smf smf/config.yaml --dry-run  # (如果支持) 打印解析后的配置
   ```

2. **验证初始化消息**:
   - 运行时应看到 `[Init] Teacher-Assisted (m=0.95)` 输出

3. **验证数值正确性**:
   - 在第一个 step 后检查 overlap 是否接近 `init_overlap`

---

## 第五部分：实施顺序清单

- [ ] **Step 1**: 修改 `config.py` 添加 `init_mode` 和 `init_overlap` 字段
- [ ] **Step 2**: 修改 `cli.py` 解析新配置字段
- [ ] **Step 3**: 在 `bigamp_spreading.py` 添加 `_initialize_near_teacher` 方法
- [ ] **Step 4**: 修改 `train_full_parallel` 初始化逻辑
- [ ] **Step 5**: 修改 `_train_full_parallel_adaptive` 初始化逻辑
- [ ] **Step 6**: 修改 `_train_full_parallel_general` 初始化逻辑
- [ ] **Step 7**: 更新 `config.yaml` 添加示例配置
- [ ] **Step 8**: 创建单元测试 `test_teacher_init.py`
- [ ] **Step 9**: 创建对比实验配置文件
- [ ] **Step 10**: 运行验证实验

---

## 第六部分：风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 张量形状不匹配 | 中 | 添加详细的 shape 断言和错误消息 |
| BF16 精度问题 | 低 | 初始化公式中保持 float32 计算，最后转换 |
| 内存增加 | 低 | Teacher-Assisted 和 Random 初始化内存消耗相同 |
| 破坏现有测试 | 低 | 默认 `init_mode="random"` 保持后向兼容 |

---

## 附录：配置字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `init_mode` | str | `"random"` | `"random"` = 随机初始化 (Cold Start), `"teacher"` = Teacher 附近初始化 (Warm Start) |
| `init_overlap` | float | `0.95` | Warm Start 时的目标初始重叠度 m_init ∈ (0, 1)，建议 0.9 - 0.99 |
