# T3: 正交教师矩阵

使用 QR 分解生成正交教师矩阵，消除有限尺寸涨落。

**模块 ID**: T3
**SMF 路径**: `modules/teachers/orthogonal.py`

---

## 🌐 宏观视角

### 系统定位

```
教师矩阵生成层
├── T1: standard.py    ← 标准高斯
├── T2: scaled.py      ← 缩放方差
└── T3: orthogonal.py  ← 本模块（理论验证最佳）
```

### 引入动机

**问题**：标准高斯教师矩阵存在**有限尺寸涨落**：

```
对于随机高斯 W*:
  实际: W*^T @ W* ≈ I_M + O(1/√N1) 的涨落
  理论: W*^T @ W* = I_M (N→∞ 极限)
```

这导致：
1. 低 α 区域 Q_Y 出现线性偏移（2αM/N）
2. 实验结果与理论预测存在系统性偏差
3. 相转移点位置微小偏移

**解决方案**：使用 QR 分解强制正交性
```
W*^T @ W* = I_M (精确)
X* @ X*^T = I_M (精确)
```

### 相对优势

| 特性 | T1 标准 | T3 正交 |
|------|--------|--------|
| W*^T @ W* | ≈ I + 涨落 | = I (精确) |
| 低 α 偏移 | 2αM/N | 0 |
| 理论符合度 | 一般 | **很高** |
| 适用于精确验证 | ❌ | ✅ |

### 物理图景 🌟

**有限尺寸涨落的来源**：

对于 N1×M 的随机高斯矩阵 W*：
```
Gram 矩阵: G = (1/N1) × W*^T @ W*

N1 → ∞ 时: G → I_M (Marchenko-Pastur 定律)
N1 有限时: G = I_M + δG, ||δG|| ~ O(1/√N1)
```

**这如何影响 overlap？**

在低 α 区域，学生矩阵接近随机初始化：
```
Q_Y = cosine(Y_student, Y_teacher)

如果 W_student 随机，W_teacher 随机：
  Q_Y ≈ <W_s^T W_t> / (||W_s|| ||W_t||)
      ≈ 2αM/N  (非零！)

如果 W_teacher 正交：
  Q_Y ≈ 0  (理想行为)
```

**正交化的效果**：

```
标准教师:                    正交教师:
Q_Y                          Q_Y
 │    ╱                       │        ╱
 │   ╱                        │       ╱
 │  ╱                         │      ╱
 │ ╱ ← 线性偏移               │     ╱
 │╱                           │────╱
 └────── α                    └────── α
      αc                           αc

低 α 区域有 2αM/N 偏移      低 α 区域 Q_Y ≈ 0
```

### 使用场景

**适用**：
- 精确验证理论预测
- 研究有限尺寸效应
- 需要与 N→∞ 极限比较时

**不适用**：
- 一般性实验（过度拟合理论条件）
- 研究随机矩阵本身的性质时

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/orthogonal_teacher.py | `create_orthogonal_teacher` | 137-174 |

**注意**：此函数仅在 `orthogonal_teacher.py` 中定义。

### 数学定义

```
生成过程:
1. W_raw ~ N(0, I)_{N1×M}
2. W_ortho, R = QR(W_raw)  # thin QR
3. W* = W_ortho × √(N1/M)  # 缩放

正交性:
  W*^T @ W* = (N1/M) × I_M

类似地对 X:
  X* @ X*^T = (N2/M) × I_M
```

**缩放因子的选择**：

```
标准教师: E[||W*||²_F] = N1
正交教师: ||W_ortho||²_F = M (正交矩阵)
缩放后:   ||W*||²_F = M × (N1/M) = N1 ✓
```

### 输入/输出

```python
def create_orthogonal_teacher(N1, N2, M, device, seed=42):
    """
    Args:
        N1: int - 矩阵行数
        N2: int - 矩阵列数
        M: int - 秩（隐维度）
        device: torch.device - 目标设备
        seed: int - 随机种子

    Returns:
        W: Tensor[N1, M] - 正交教师左因子
        X: Tensor[M, N2] - 正交教师右因子

    Properties:
        W^T @ W = (N1/M) × I_M
        X @ X^T = (N2/M) × I_M
    """
```

### 标准实现

```python
def create_orthogonal_teacher(N1, N2, M, device, seed=42):
    """Create orthogonal teacher model using QR decomposition."""
    torch.manual_seed(seed)

    # Generate random matrices
    W_raw = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_raw = torch.randn(M, N2, device=device, dtype=torch.float32)

    # QR decomposition for W (thin QR)
    # After QR: W_ortho^T @ W_ortho = I_M
    W_ortho, _ = torch.linalg.qr(W_raw, mode='reduced')

    # QR decomposition for X^T, then transpose back
    # After QR: X_ortho @ X_ortho^T = I_M
    X_ortho_T, _ = torch.linalg.qr(X_raw.T, mode='reduced')
    X_ortho = X_ortho_T.T

    # Scale to match expected Frobenius norm of standard teacher
    W_true = W_ortho * (N1 / M) ** 0.5
    X_true = X_ortho * (N2 / M) ** 0.5

    return W_true, X_true
```

### 实现细节

1. **Thin QR**：`mode='reduced'` 产生 N1×M 的 Q 矩阵
2. **X 的处理**：对 X^T 做 QR 再转置，保证行正交
3. **缩放匹配**：√(N/M) 使 Frobenius 范数与标准教师一致

### 验证正交性

```python
# 验证代码
W, X = create_orthogonal_teacher(N1, N2, M, device)
WTW = W.T @ W
XXT = X @ X.T
print(f"W^T W 与 (N1/M)*I 的偏差: {(WTW - (N1/M)*torch.eye(M)).abs().max()}")
print(f"X X^T 与 (N2/M)*I 的偏差: {(XXT - (N2/M)*torch.eye(M)).abs().max()}")
# 输出应接近 0 (浮点精度内)
```

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Generate orthogonal teacher matrices to eliminate finite-size fluctuations",
    "when_to_use_en": "Theory verification, precise phase transition study, comparison with N→∞ limit",
    "limitations_en": "Over-idealized conditions, not representative of general random matrices",
    "tags_en": ["orthogonal", "QR", "finite-size", "teacher", "theory", "thermodynamic limit"],

    # 中文
    "purpose_zh": "生成正交教师矩阵以消除有限尺寸涨落",
    "when_to_use_zh": "理论验证、精确相转移研究、与 N→∞ 极限比较",
    "limitations_zh": "过度理想化条件，不代表一般随机矩阵",
    "tags_zh": ["正交", "QR分解", "有限尺寸", "教师", "理论", "热力学极限"],

    # 日文
    "purpose_ja": "有限サイズ揺らぎを除去するため直交教師行列を生成",
    "when_to_use_ja": "理論検証、精密相転移研究、N→∞極限との比較",
    "tags_ja": ["直交", "QR分解", "有限サイズ", "教師", "理論"],

    # 技术参数
    "inputs": ["N1: int", "N2: int", "M: int", "device", "seed: int"],
    "outputs": ["W: Tensor[N1, M]", "X: Tensor[M, N2]"],
    "properties": ["W^T @ W = (N1/M) × I_M", "X @ X^T = (N2/M) × I_M"],
    "compute_cost": "O(N1×M² + N2×M²)",  # QR 复杂度
    "gpu_friendly": True,
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/teachers/orthogonal.py`

---

*最后更新：2025年12月*
