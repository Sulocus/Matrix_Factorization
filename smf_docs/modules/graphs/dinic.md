# G2: Dinic 最大流算法

用于生成双正则二分图的最大流算法实现。

**模块 ID**: G2
**SMF 路径**: `modules/graphs/dinic.py`

---

## 🌐 宏观视角

### 系统定位

```
观测掩码生成层
├── G1: random.py      ← 纯随机（独立模块）
├── G2: dinic.py       ← 本模块（G3 的核心依赖）
├── G3: uniform.py     ← 双正则图（调用 G2）
└── G4: low_loop.py    ← 低循环图（可选依赖 G2）
```

G2 是 G3 的**内部实现**，通常不直接调用。

### 引入动机

**问题**：如何生成一个二分图，使得：
- 每个左节点恰好有 `deg_left` 条边
- 每个右节点的度数尽可能均匀

**解法**：将其建模为**最大流问题**：
```
源点 S → 左节点 → 右节点 → 汇点 T
```
Dinic 算法是解决此问题的高效方法。

### 相对优势

| 算法 | 时间复杂度 | 特点 |
|------|-----------|------|
| **Dinic** | O(V²E) | 实现简单，适合稀疏图 |
| Ford-Fulkerson | O(E × maxflow) | 可能很慢 |
| Push-Relabel | O(V³) | 密集图更优 |

对于我们的场景（N1, N2 ~ 数百到数千），Dinic 已经足够高效。

### 物理图景 🌟

**最大流的直觉**：想象水从源点流向汇点

```
        ┌─→ L0 ─┬─→ R0 ─┐
        │       ×       │
   S ───┼─→ L1 ─┼─→ R1 ─┼──→ T
        │       ×       │
        └─→ L2 ─┴─→ R2 ─┘

S→Li 容量 = deg_left（每个左节点需要的边数）
Li→Rj 容量 = 1（每对最多一条边）
Rj→T 容量 = right_target[j]（每个右节点允许的边数）

最大流 = 总边数 = N1 × deg_left
```

**Dinic 的工作方式**：
1. BFS 建立层次图（按距离分层）
2. DFS 在层次图中找增广路径
3. 重复直到无法增广

### 使用场景

**直接调用 G2**：
- 需要自定义流网络时
- 调试或验证图生成正确性时

**通过 G3 间接使用**：
- 生成双正则图时（推荐）

---

## 🔬 微观视角

### 代码位置

| 程序 | 类名 | 行号 |
|------|------|------|
| bigamp/train.py | `class Dinic` | 205-245 |
| bigamp/compare_sizes.py | `class Dinic` | 236-... |
| agd/train_sequential.py | `class Dinic` | 233-... |
| agd/train_parallel.py | `class Dinic` | 240-... |

**注意**：Dinic 类定义在 `sample_pairs_biregular_exact` 函数内部。

### 算法步骤

1. **构建流网络**：
   - 节点：S(源), L0..L(N1-1)(左), R0..R(N2-1)(右), T(汇)
   - 边：S→Li, Li→Rj, Rj→T

2. **BFS 分层**：计算每个节点到源的最短距离

3. **DFS 增广**：沿着层次递增的路径推送流量

4. **重复**：直到无法从 S 到达 T

### 输入/输出

```python
class Dinic:
    def __init__(self, n):
        """
        Args:
            n: int - 节点总数（包括 S 和 T）
        """

    def add_edge(self, u, v, cap):
        """
        Args:
            u, v: int - 边的起点和终点
            cap: int - 边的容量
        """

    def max_flow(self, s, t):
        """
        Args:
            s, t: int - 源点和汇点

        Returns:
            flow: int - 最大流值
        """
```

### 标准实现

```python
class Dinic:
    __slots__ = ("n", "g", "lvl", "it")

    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        self.g[u].append([v, cap, len(self.g[v])])
        self.g[v].append([u, 0, len(self.g[u]) - 1])

    def bfs(self, s, t):
        self.lvl = [-1] * self.n
        q = deque([s])
        self.lvl[s] = 0
        while q:
            u = q.popleft()
            for v, cap, _ in self.g[u]:
                if cap > 0 and self.lvl[v] < 0:
                    self.lvl[v] = self.lvl[u] + 1
                    q.append(v)
        return self.lvl[t] >= 0

    def dfs(self, u, t, f):
        if u == t: return f
        for i in range(self.it[u], len(self.g[u])):
            self.it[u] = i
            v, cap, rev = self.g[u][i]
            if cap > 0 and self.lvl[u] + 1 == self.lvl[v]:
                d = self.dfs(v, t, min(f, cap))
                if d > 0:
                    self.g[u][i][1] -= d
                    self.g[v][rev][1] += d
                    return d
        return 0

    def max_flow(self, s, t):
        flow = 0
        INF = 10**9
        while self.bfs(s, t):
            self.it = [0] * self.n
            while True:
                f = self.dfs(s, t, INF)
                if f == 0: break
                flow += f
        return flow
```

### 实现细节

1. **邻接表存储**：`g[u]` 存储 `[v, cap, rev]` 三元组
2. **反向边**：`rev` 记录反向边在 `g[v]` 中的索引，用于流量回退
3. **层次剪枝**：只沿层次递增的边搜索
4. **当前弧优化**：`self.it` 避免重复访问已饱和的边

### 复杂度

- 时间：O(V²E)，对于二分图可达 O(E√V)
- 空间：O(V + E)

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Maximum flow algorithm for generating bi-regular bipartite graphs",
    "when_to_use_en": "Internal implementation for uniform degree graph generation",
    "tags_en": ["Dinic", "max flow", "bipartite", "graph", "algorithm", "network flow"],

    # 中文
    "purpose_zh": "用于生成双正则二分图的最大流算法",
    "when_to_use_zh": "G3 的内部实现，通常不直接调用",
    "tags_zh": ["Dinic", "最大流", "二分图", "图算法", "网络流"],

    # 日文
    "purpose_ja": "双正則二部グラフ生成のための最大流アルゴリズム",
    "when_to_use_ja": "G3の内部実装、通常は直接呼び出さない",
    "tags_ja": ["Dinic", "最大流", "二部グラフ", "グラフアルゴリズム"],

    # 技术参数
    "compute_cost": "O(V²E)",
    "gpu_friendly": False,  # 纯 Python/NumPy 实现
    "dependencies": ["collections.deque"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/graphs/dinic.py`

---

*最后更新：2025年12月*
