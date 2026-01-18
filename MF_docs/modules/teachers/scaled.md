# T2: 缩放方差教师矩阵

支持自定义方差缩放的教师矩阵生成。

**模块 ID**: T2
**SMF 路径**: `modules/teachers/scaled.py`
**状态**: ⬜ 待实现

---

## 🌐 宏观视角

### 系统定位

```
教师矩阵生成层
├── T1: standard.py    ← 标准高斯 (σ² = 1/M)
├── T2: scaled.py      ← 本模块（自定义方差）
└── T3: orthogonal.py  ← 正交教师
```

### 引入动机

T1 使用固定的方差 `1/M`。T2 允许自定义方差缩放，用于：
1. 研究方差对相转移的影响
2. 模拟不同先验假设
3. 数值稳定性测试

### 物理图景 🌟

```
标准: W*[i,k] ~ N(0, 1/M)
缩放: W*[i,k] ~ N(0, σ²/M)

σ > 1: 信号更强，相转移点可能更低
σ < 1: 信号更弱，相转移点可能更高
```

---

## 🔬 微观视角

### 当前状态

此模块在计划中定义，但尚未在 Wang/ 代码中单独实现。
标准实现（T1）可通过修改 `scale` 参数来实现类似功能：

```python
def create_teacher_scaled(N1, N2, M, device, variance_scale=1.0, seed=42):
    """Create teacher with custom variance scaling."""
    torch.manual_seed(seed)
    scale = (variance_scale / M) ** 0.5
    W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale
    return W, X
```

### 待办事项

- [ ] 确定是否需要单独实现
- [ ] 如果需要，添加到 Wang/ 程序中
- [ ] 同步到 SMF 框架

---

## AI 关键词

```python
ai_metadata = {
    "purpose_en": "Generate teacher matrices with custom variance scaling",
    "when_to_use_en": "Study variance effects, custom prior assumptions",
    "status": "planned",
    "tags_en": ["teacher", "variance", "scaling", "prior"],
    "tags_zh": ["教师", "方差", "缩放", "先验"],
}
```

---

*最后更新：2025年12月*
