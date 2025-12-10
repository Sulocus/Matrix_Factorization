# S2: 结果扫描器

扫描指定目录，自动发现和加载实验结果文件。

**模块 ID**: S2
**SMF 路径**: `modules/outputs/storage/scanner.py`

---

## 🌐 宏观视角

### 系统定位

```
输出层/storage/
├── S1: json_io.py  ← JSON 读写（单文件）
└── S2: scanner.py  ← 本模块（批量发现）
```

### 引入动机

**问题**：实验结果文件散落在各处

```
Result/
├── 200_200_50/
│   ├── results_bigamp_200x200_M50_S1000.json
│   ├── results_agd_200x200_M50_E20000.json
│   └── old_results.json
├── 500_500_100/
│   └── results_bigamp_500x500_M100_S2000.json
└── ...
```

**需求**：自动发现、筛选、聚合结果

### 物理图景 🌟

**结果管理工作流**：

```
扫描目录
   ↓
匹配文件名模式
   ↓
解析参数（从文件名或内容）
   ↓
筛选（按尺寸、算法等）
   ↓
加载 & 聚合
   ↓
对比分析
```

### 使用场景

**适用**：
- 批量加载实验结果
- 生成对比图
- 结果汇总报告

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| analysis/compare_algorithms.py | `scan_results` | 50-100 |
| bigamp/compare_sizes.py | `load_all_results` | 700-750 |

### 输入/输出

```python
def scan_results(base_dir, pattern='*.json', recursive=True):
    """
    Args:
        base_dir: str - 基础目录
        pattern: str - 文件名模式（glob）
        recursive: bool - 是否递归搜索子目录

    Returns:
        files: List[Dict] - 发现的文件列表
            [{
                'path': str,        # 完整路径
                'filename': str,    # 文件名
                'parameters': Dict, # 解析出的参数
            }, ...]
    """


def filter_results(files, algorithm=None, N1=None, M=None):
    """
    Args:
        files: List[Dict] - scan_results 的输出
        algorithm: str - 筛选算法（'bigamp' 或 'agd'）
        N1: int - 筛选矩阵尺寸
        M: int - 筛选秩

    Returns:
        filtered: List[Dict] - 筛选后的文件列表
    """
```

### 标准实现

```python
import os
import glob
import re
from typing import List, Dict, Optional


def scan_results(base_dir: str, pattern: str = '*.json',
                 recursive: bool = True) -> List[Dict]:
    """Scan directory for result files."""
    if recursive:
        search_pattern = os.path.join(base_dir, '**', pattern)
        paths = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(base_dir, pattern)
        paths = glob.glob(search_pattern)

    files = []
    for path in paths:
        filename = os.path.basename(path)
        params = parse_filename(filename)
        files.append({
            'path': path,
            'filename': filename,
            'parameters': params,
        })

    return files


def parse_filename(filename: str) -> Dict:
    """Extract parameters from filename."""
    params = {}

    # Algorithm
    if 'bigamp' in filename.lower():
        params['algorithm'] = 'BiG-AMP'
    elif 'agd' in filename.lower():
        params['algorithm'] = 'AGD'

    # Matrix size: NxN or N1xN2
    size_match = re.search(r'(\d+)x(\d+)', filename)
    if size_match:
        params['N1'] = int(size_match.group(1))
        params['N2'] = int(size_match.group(2))

    # Rank M
    m_match = re.search(r'M(\d+)', filename)
    if m_match:
        params['M'] = int(m_match.group(1))

    # Steps or Epochs
    steps_match = re.search(r'S(\d+)', filename)
    if steps_match:
        params['steps'] = int(steps_match.group(1))

    epochs_match = re.search(r'E(\d+)', filename)
    if epochs_match:
        params['epochs'] = int(epochs_match.group(1))

    return params


def filter_results(files: List[Dict],
                   algorithm: Optional[str] = None,
                   N1: Optional[int] = None,
                   M: Optional[int] = None) -> List[Dict]:
    """Filter result files by parameters."""
    filtered = []
    for f in files:
        params = f['parameters']

        if algorithm and params.get('algorithm', '').lower() != algorithm.lower():
            continue
        if N1 and params.get('N1') != N1:
            continue
        if M and params.get('M') != M:
            continue

        filtered.append(f)

    return filtered
```

### 批量加载

```python
def load_all_results(files: List[Dict]) -> List[Dict]:
    """Load all result files."""
    from json_io import load_results

    loaded = []
    for f in files:
        try:
            data = load_results(f['path'])
            data['_file_info'] = f  # 附加文件信息
            loaded.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {f['path']}: {e}")

    return loaded


def group_by_size(loaded_results: List[Dict]) -> Dict[tuple, List]:
    """Group results by matrix size."""
    groups = {}
    for result in loaded_results:
        params = result['parameters']
        key = (params.get('N1'), params.get('N2'), params.get('M'))
        if key not in groups:
            groups[key] = []
        groups[key].append(result)
    return groups
```

### 使用示例

```python
# 扫描所有结果
files = scan_results('Result/', pattern='results_*.json')
print(f"Found {len(files)} result files")

# 筛选 BiG-AMP, N=200
bigamp_200 = filter_results(files, algorithm='bigamp', N1=200)
print(f"BiG-AMP N=200: {len(bigamp_200)} files")

# 加载并按尺寸分组
loaded = load_all_results(bigamp_200)
groups = group_by_size(loaded)

# 生成对比图
for size, results in groups.items():
    print(f"Size {size}: {len(results)} experiments")
```

### 实现细节

1. **glob 递归**：`**` 匹配任意深度子目录
2. **正则解析**：从文件名提取参数
3. **错误处理**：跳过损坏的 JSON 文件
4. **元信息保留**：`_file_info` 保留文件路径等

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Scan directories to discover and load experiment results",
    "when_to_use_en": "Batch loading, comparison plots, result aggregation",
    "tags_en": ["scan", "discover", "batch", "filter", "aggregate"],

    # 中文
    "purpose_zh": "扫描目录以发现和加载实验结果",
    "when_to_use_zh": "批量加载、生成对比图、结果聚合",
    "tags_zh": ["扫描", "发现", "批量", "筛选", "聚合"],

    # 技术参数
    "inputs": ["base_dir", "pattern", "recursive", "filters"],
    "outputs": ["files: List[Dict]", "loaded_results: List[Dict]"],
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/outputs/storage/scanner.py`

---

*最后更新：2025年12月*
