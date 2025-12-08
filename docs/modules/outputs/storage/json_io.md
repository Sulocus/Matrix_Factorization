# S1: JSON 读写

实验结果的 JSON 格式保存和加载。

**模块 ID**: S1
**SMF 路径**: `modules/outputs/storage/json_io.py`

---

## 🌐 宏观视角

### 系统定位

```
输出层
├── plotting/
│   └── (P1-P4 绘图模块)
└── storage/
    ├── S1: json_io.py  ← 本模块（JSON 读写）
    └── S2: scanner.py  ← 结果扫描
```

### 引入动机

**为什么用 JSON？**

1. **人类可读**：可以直接查看结果
2. **跨平台**：Python, JavaScript, MATLAB 都支持
3. **版本控制友好**：文本格式易于 diff
4. **无二进制依赖**：不需要特定版本的 pickle

### 物理图景 🌟

**实验数据流**：

```
训练 → 结果字典 → JSON 文件 → 分析/绘图
         ↑
       {
         'alpha_values': [...],
         'Q_Y_mean': [...],
         'Q_Y_std': [...],
         'parameters': {...}
       }
```

### 使用场景

**适用**：
- 保存实验结果
- 跨程序数据交换
- 长期存档

---

## 🔬 微观视角

### 代码位置

| 程序 | 函数 | 行号 |
|------|------|------|
| bigamp/train.py | `save_results` | 620-650 |
| agd/train_parallel.py | `save_results` | 400-430 |

### 标准 JSON 格式

```json
{
    "metadata": {
        "algorithm": "BiG-AMP",
        "timestamp": "2025-12-05T10:30:00",
        "version": "1.0"
    },
    "parameters": {
        "N1": 200,
        "N2": 200,
        "M": 50,
        "alpha_start": 0.0,
        "alpha_stop": 4.0,
        "alpha_step": 0.1,
        "steps": 1000,
        "samples_per_alpha": 5,
        "damping": 0.5,
        "noise_var": 1e-6
    },
    "results": {
        "alpha_values": [0.0, 0.1, 0.2, ...],
        "Q_Y_mean": [0.02, 0.05, 0.08, ...],
        "Q_Y_std": [0.01, 0.02, 0.03, ...],
        "Q_W_mean": [...],
        "Q_W_std": [...],
        "Q_X_mean": [...],
        "Q_X_std": [...]
    }
}
```

### 输入/输出

```python
def save_results(results, parameters, output_path, algorithm='BiG-AMP'):
    """
    Args:
        results: Dict - 实验结果
            {'alpha_values': [...], 'Q_Y_mean': [...], ...}
        parameters: Dict - 实验参数
            {'N1': 200, 'M': 50, ...}
        output_path: str - 输出文件路径
        algorithm: str - 算法名称

    Returns:
        None (保存到文件)
    """


def load_results(input_path):
    """
    Args:
        input_path: str - JSON 文件路径

    Returns:
        data: Dict - 包含 metadata, parameters, results
    """
```

### 标准实现

```python
import json
from datetime import datetime


def save_results(results, parameters, output_path, algorithm='BiG-AMP'):
    """Save experiment results to JSON."""
    data = {
        'metadata': {
            'algorithm': algorithm,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
        },
        'parameters': parameters,
        'results': results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def load_results(input_path):
    """Load experiment results from JSON."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def results_to_dict(alpha_values, Q_Y_list, Q_W_list, Q_X_list):
    """Convert result lists to dictionary format."""
    # Aggregate across samples
    import numpy as np

    results = {
        'alpha_values': list(alpha_values),
        'Q_Y_mean': [float(np.mean(q)) for q in Q_Y_list],
        'Q_Y_std': [float(np.std(q)) for q in Q_Y_list],
        'Q_W_mean': [float(np.mean(q)) for q in Q_W_list],
        'Q_W_std': [float(np.std(q)) for q in Q_W_list],
        'Q_X_mean': [float(np.mean(q)) for q in Q_X_list],
        'Q_X_std': [float(np.std(q)) for q in Q_X_list],
    }
    return results
```

### Numpy/Tensor 兼容

```python
def numpy_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_json_serializable(v) for v in obj]
    else:
        return obj


def save_results_safe(results, parameters, output_path, algorithm='BiG-AMP'):
    """Save with automatic numpy conversion."""
    results_clean = numpy_to_json_serializable(results)
    parameters_clean = numpy_to_json_serializable(parameters)
    save_results(results_clean, parameters_clean, output_path, algorithm)
```

### 文件命名规范

```python
def generate_result_filename(N1, N2, M, algorithm, epochs_or_steps):
    """Generate standard filename for results."""
    # 格式: results_{algorithm}_{N1}x{N2}_M{M}_E{epochs}.json
    if 'bigamp' in algorithm.lower():
        return f"results_bigamp_{N1}x{N2}_M{M}_S{epochs_or_steps}.json"
    else:
        return f"results_agd_{N1}x{N2}_M{M}_E{epochs_or_steps}.json"
```

### 实现细节

1. **UTF-8 编码**：支持中文注释
2. **indent=2**：格式化输出，便于阅读
3. **类型转换**：numpy 类型转 Python 原生类型
4. **版本字段**：便于未来格式升级

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Save and load experiment results in JSON format",
    "when_to_use_en": "Result persistence, cross-program data exchange, archiving",
    "tags_en": ["JSON", "save", "load", "serialization", "results"],

    # 中文
    "purpose_zh": "以 JSON 格式保存和加载实验结果",
    "when_to_use_zh": "结果持久化、跨程序数据交换、存档",
    "tags_zh": ["JSON", "保存", "加载", "序列化", "结果"],

    # 技术参数
    "inputs": ["results: Dict", "parameters: Dict", "output_path: str"],
    "outputs": ["JSON file"],
    "format": "metadata + parameters + results",
}
```

### SMF 对应模块

`sparse_matrix_factorization/modules/outputs/storage/json_io.py`

---

*最后更新：2025年12月*
