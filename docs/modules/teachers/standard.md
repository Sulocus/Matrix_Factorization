# T1: 标准教师矩阵

使用高斯分布初始化的标准教师矩阵 W* 和 X*。

**模块 ID**: T1
**SMF 路径**: `modules/teachers/standard.py`

---

## 🌐 宏观视角

### 系统定位

```
教师矩阵生成层
├── T1: standard.py    ← 本模块（默认方法）
├── T2: scaled.py      ← 缩放方差版本
└── T3: orthogonal.py  ← 正交教师（消除有限尺寸涨落）
```

### 引入动机

Teacher-Student 框架中，**教师**持有真实矩阵分解：
```
Y* = W* @ X*
```
学生只能观测到 Y* 的部分元素，目标是恢复 W* 和 X*。

T1 是最基础的教师生成方法：独立同分布高斯随机矩阵。

### 相对优势

| 方法 | 有限尺寸涨落 | 理论符合度 | 适用场景 |
|------|------------|-----------|---------|
| **T1 标准** | 存在 | 一般 | 大多数实验 |
| T3 正交 | 消除 | 更高 | 理论验证 |

### 物理图景 🌟

**为什么使用高斯分布？**

1. **中心极限定理**：许多实际数据可近似为高斯
2. **数学便利**：高斯矩阵的谱性质已被充分研究
3. **理论假设**：replica theory 通常假设高斯先验

**缩放因子 1/√M 的含义**：

```
W*[i,k] ~ N(0, 1/M)
X*[k,j] ~ N(0, 1/M)

则 Y*[i,j] = Σ_k W*[i,k] × X*[k,j]
E[Y*²] = M × (1/M) × (1/M) × M = 1
```

这保证了 Y* 的元素具有 O(1) 的方差，与 M 无关。

### 使用场景

**适用**：
- 标准实验和基准测试
- 不需要精确理论比较的场景
- 快速原型验证

**不适用**：
- 需要消除有限尺寸效应时（使用 T3）
- 研究方差缩放影响时（使用 T2）

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/train.py | `create_teacher` | 121-127 |
| bigamp/compare_sizes.py | `create_teacher` | 127-... |
| bigamp/orthogonal_teacher.py | `create_teacher` | 128-134 |
| agd/train_sequential.py | `create_teacher_dense` | 109-... |
| agd/train_parallel.py | `create_teacher_dense` | 116-... |

### 数学定义

```
W* ∈ ℝ^(N1×M), W*[i,k] ~ N(0, 1/M)
X* ∈ ℝ^(M×N2), X*[k,j] ~ N(0, 1/M)
Y* = W* @ X* ∈ ℝ^(N1×N2)
```

**性质**：
- E[||W*||²_F] = N1
- E[||X*||²_F] = N2
- E[||Y*||²_F] = N1 × N2 / M

### 输入/输出

```python
def create_teacher(N1, N2, M, device, seed=42):
    """
    Args:
        N1: int - 矩阵行数
        N2: int - 矩阵列数
        M: int - 秩（隐维度）
        device: torch.device - 目标设备
        seed: int - 随机种子

    Returns:
        W: Tensor[N1, M] - 教师左因子
        X: Tensor[M, N2] - 教师右因子
    """
```

### 标准实现

```python
def create_teacher(N1, N2, M, device, seed=42):
    """Create standard Gaussian teacher model W_true and X_true"""
    torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale
    return W, X
```

### 实现细节

1. **随机种子**：确保可复现性
2. **缩放因子**：`1/√M` 保证 Y* 元素的方差为 O(1)
3. **数据类型**：使用 float32 保证数值精度

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Generate standard Gaussian teacher matrices W* and X*",
    "when_to_use_en": "Default teacher for most experiments, benchmark",
    "tags_en": ["teacher", "Gaussian", "random", "initialization", "standard"],

    # 中文
    "purpose_zh": "生成标准高斯教师矩阵 W* 和 X*",
    "when_to_use_zh": "大多数实验的默认教师，基准测试",
    "tags_zh": ["教师", "高斯", "随机", "初始化", "标准"],

    # 日文
    "purpose_ja": "標準ガウス教師行列 W* と X* を生成",
    "when_to_use_ja": "ほとんどの実験のデフォルト教師、ベンチマーク",
    "tags_ja": ["教師", "ガウス", "ランダム", "初期化"],

    # 技术参数
    "inputs": ["N1: int", "N2: int", "M: int", "device", "seed: int"],
    "outputs": ["W: Tensor[N1, M]", "X: Tensor[M, N2]"],
    "compute_cost": "O(N1×M + M×N2)",
    "gpu_friendly": True,
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/teachers/standard.py`

---

*最后更新：2025年12月*
