# D1: 设备检测

自动检测计算设备并配置精度设置

**模块 ID**: D1
**SMF 路径**: `sparse_matrix_factorization/core/device.py`

---

## 目的

自动检测可用的计算设备（CUDA/MPS/CPU），配置混合精度和 TF32 加速设置。

---

## 功能说明

1. **设备检测顺序**: CUDA → MPS → CPU
2. **BF16 混合精度**: 仅 CUDA 启用
3. **TF32 加速**: 仅 CUDA 启用（matmul 和 cuDNN）

---

## 代码位置

| 程序 | 行号 | 状态 |
|------|------|------|
| bigamp/train.py | 49-62 | ✅ 一致 |
| bigamp/compare_sizes.py | 68-79 | ✅ 一致 |
| bigamp/orthogonal_teacher.py | 69-80 | ✅ 一致 |
| agd/train_parallel.py | TBD | 待检查 |

---

## 标准实现

```python
# =============================================================================
# Device Setup
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
USE_BF16 = DEVICE.type == 'cuda'
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
STORAGE_DTYPE = torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

---

## 输入/输出

**输入**: 无（从系统检测）

**输出**:
- `DEVICE`: torch.device 对象
- `USE_BF16`: bool，是否使用 BF16 精度
- `COMPUTE_DTYPE`: torch.dtype，计算精度
- `STORAGE_DTYPE`: torch.dtype，存储精度

---

## AI 关键词

```python
ai_metadata = {
    # 英文
    "purpose_en": "Auto-detect computing device and configure precision settings",
    "when_to_use_en": "At program startup to initialize GPU/CPU configuration",
    "limitations_en": "MPS support is experimental, BF16 only on CUDA",
    "inputs": [],
    "outputs": ["DEVICE: torch.device", "USE_BF16: bool", "COMPUTE_DTYPE: torch.dtype", "STORAGE_DTYPE: torch.dtype"],
    "tags_en": ["device", "cuda", "mps", "cpu", "bf16", "tf32", "precision", "gpu", "initialization"],

    # 中文
    "purpose_zh": "自动检测计算设备并配置精度设置",
    "when_to_use_zh": "程序启动时初始化 GPU/CPU 配置",
    "limitations_zh": "MPS 支持为实验性，BF16 仅支持 CUDA",
    "tags_zh": ["设备检测", "CUDA", "MPS", "CPU", "混合精度", "TF32", "GPU", "初始化"],

    # 日文
    "purpose_ja": "計算デバイスを自動検出し、精度設定を構成",
    "when_to_use_ja": "プログラム起動時にGPU/CPU設定を初期化",
    "limitations_ja": "MPSサポートは実験的、BF16はCUDAのみ",
    "tags_ja": ["デバイス検出", "CUDA", "MPS", "混合精度", "GPU", "初期化"],
}
```

---

## 优化记录

| 日期 | 优化内容 | 影响 |
|------|---------|------|
| - | 添加 MPS 支持 | Apple Silicon 兼容 |
| - | 添加 STORAGE_DTYPE 常量 | 统一存储精度控制 |

---

*最后更新：2024年12月*
