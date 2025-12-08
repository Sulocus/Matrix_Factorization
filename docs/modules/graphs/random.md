# G1: 随机图生成

GPU 上的纯随机掩码生成，支持任意 N1≠N2 的矩阵维度。

**模块 ID**: G1
**SMF 路径**: `modules/graphs/random.py`

---

## 🌐 宏观视角

### 系统定位

```
观测掩码生成层
├── G1: random.py      ← 本模块（最简单、最快）
├── G2: dinic.py       ← 最大流算法（G3的依赖）
├── G3: uniform.py     ← 双正则图（更均匀的度分布）
└── G4: low_loop.py    ← 低循环图（AMP理论更准确）
```

### 引入动机

在矩阵分解问题中，我们只能观测到部分矩阵元素。需要生成一个二值掩码 `mask[i,j] ∈ {0,1}` 来指定哪些位置被观测。

**最简单的方法**：从 N1×N2 个位置中随机选择 C 个，这就是 G1 的实现。

### 相对优势

| 方法 | 速度 | 度分布均匀性 | 适用场景 |
|------|------|------------|---------|
| **G1 随机图** | ⚡ 最快 | ❌ 不均匀（泊松分布） | 快速测试、大规模实验 |
| G3 双正则图 | 🐢 较慢 | ✅ 均匀 | 理论验证、有限尺寸效应研究 |
| G4 低循环图 | 🐌 最慢 | ✅ 均匀 | AMP 理论精确验证 |

### 物理图景 🌟

将观测矩阵看作一个二分图：
```
左节点 (行 i=0..N1-1)     右节点 (列 j=0..N2-1)
    ○ ─────────────────── ○
    ○ ────┐          ┌─── ○
    ○ ───┐│          │┌── ○
         ││    ...   ││
    ○ ───┘│          │└── ○
    ○ ────┘          └─── ○

每条边 (i,j) 表示位置 [i,j] 被观测
```

**随机图的特点**：每条边独立随机选取
- 左节点度数（每行被观测的列数）服从**泊松分布**
- 大多数节点度数接近均值，但存在较大方差
- 简单高效，但可能产生度数极端的节点

### 使用场景

**适用**：
- 快速原型验证
- 大规模实验（速度优先）
- 不关心有限尺寸效应的场景

**不适用**：
- 需要精确验证理论预测时（应使用 G3）
- 研究图结构对算法收敛性影响时（应使用 G4）

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/train.py | `sample_pairs_random_gpu` | 130-157 |
| bigamp/compare_sizes.py | `sample_pairs_random_gpu` | 136-... |
| bigamp/orthogonal_teacher.py | `sample_pairs_random_gpu` | 177-... |
| agd/train_sequential.py | `sample_pairs_random_gpu` | 125-... |
| agd/train_parallel.py | `sample_pairs_random_gpu` | 132-... |

### 数学定义

给定 N1×N2 矩阵，需要选择 C 个观测点：

```
从 {0, 1, ..., N1×N2-1} 中无放回抽取 C 个索引
将线性索引转换为二维坐标：
    i = idx // N2
    j = idx % N2
```

### 输入/输出

```python
def sample_pairs_random_gpu(N1, N2, C, device, seed=None):
    """
    Args:
        N1: int - 矩阵行数
        N2: int - 矩阵列数
        C: int - 需要的边数（观测点数）
        device: torch.device - 目标设备
        seed: int, optional - 随机种子

    Returns:
        i_idx: Tensor[C] - 行索引
        j_idx: Tensor[C] - 列索引
        C: int - 边数
    """
```

### 标准实现

```python
def sample_pairs_random_gpu(N1, N2, C, device, seed=None):
    """Pure random mask generation (entirely on GPU, supports any N1≠N2)"""
    if seed is not None:
        torch.manual_seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"Requested edge count C={C} exceeds matrix total size {N1}×{N2}={total}")

    if C == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    idx = torch.randperm(total, device=device)[:C]
    i_idx = idx // N2
    j_idx = idx % N2

    return i_idx, j_idx, C
```

### 实现细节

1. **GPU 加速**: `torch.randperm` 在 GPU 上执行
2. **无放回抽样**: 保证每个位置最多被选一次
3. **边界检查**: C > N1×N2 时抛出异常
4. **空图处理**: C=0 时返回空张量

### 复杂度

- 时间: O(N1×N2) 用于生成排列
- 空间: O(N1×N2) 用于存储排列
- 对于大矩阵，这是最快的方法

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Generate random observation mask on GPU via uniform sampling",
    "when_to_use_en": "Fast prototyping, large-scale experiments, speed priority",
    "limitations_en": "Non-uniform degree distribution (Poisson), not for theory verification",
    "tags_en": ["random", "mask", "graph", "GPU", "sampling", "observation", "sparse"],

    # 中文
    "purpose_zh": "在 GPU 上通过均匀采样生成随机观测掩码",
    "when_to_use_zh": "快速原型验证、大规模实验、速度优先场景",
    "limitations_zh": "度分布不均匀（泊松分布），不适合理论验证",
    "tags_zh": ["随机", "掩码", "图", "GPU", "采样", "观测", "稀疏"],

    # 日文
    "purpose_ja": "GPU上で一様サンプリングによりランダム観測マスクを生成",
    "when_to_use_ja": "高速プロトタイピング、大規模実験、速度優先",
    "tags_ja": ["ランダム", "マスク", "グラフ", "GPU", "サンプリング"],

    # 技术参数
    "inputs": ["N1: int", "N2: int", "C: int", "device: torch.device", "seed: int?"],
    "outputs": ["i_idx: Tensor[C]", "j_idx: Tensor[C]", "C: int"],
    "compute_cost": "O(N1×N2)",
    "gpu_friendly": True,
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/graphs/random.py`

---

*最后更新：2025年12月*
