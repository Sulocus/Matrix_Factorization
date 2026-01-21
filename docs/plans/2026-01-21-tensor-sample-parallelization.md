# Tensor Sample 并行化实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 Tensor 模式的 S 个 samples 从串行处理改为并行处理，提升 GPU 利用率从 ~25% 到 ~100%。

**Architecture:** 创建新模块 `tensor_step_batch.py` 和 `tensor_spreading_parallel.py`，不修改现有代码。所有张量添加 batch 维度 `S`，一次 GPU 调用处理所有 samples。

**Tech Stack:** PyTorch, einsum, scatter_add

---

## Scope

- **In:**
  - Sample 并行（同一 alpha 下 S 个样本同时处理）
  - BF16 混合精度
  - torch.compile 支持
  
- **Out:**
  - Alpha 并行（Phase 3）
  - 显存动态估算（Phase 2）
  - 修改现有 `tensor_step.py`、`tensor_spreading.py`

---

## Task 1: 创建 tensor_step_batch.py

**Files:**
- Create: `src/matrix_factorization/modules/algorithms/bigamp/tensor_step_batch.py`
- Reference: `src/matrix_factorization/modules/algorithms/bigamp/tensor_step.py`

**Step 1: Create file with forward_pass_tensor_batch**

```python
"""
Batched n-dimensional tensor BiG-AMP step functions.

Supports batch dimension S for parallel sample processing.
"""

import math
import torch
from typing import List, Optional, Tuple


def forward_pass_tensor_batch(
    factors: List[torch.Tensor],  # n tensors of (S, N_d, M)
    F: torch.Tensor,              # (S, C, M)
    indices: List[torch.Tensor],  # n tensors of (C,) - shared across samples
) -> torch.Tensor:
    """
    Batched forward pass for n-dimensional tensor.
    
    Returns:
        Z_hat: (S, C) predicted values
    """
    S, C, M = F.shape
    alpha_scale = 1.0 / math.sqrt(M)
    
    # Gather factors at edge positions: (n, S, C, M)
    gathered = torch.stack([
        factors[d][:, indices[d].long()]  # (S, C, M)
        for d in range(len(factors))
    ])  # (n, S, C, M)
    
    # Product across factors
    product = gathered.prod(dim=0)  # (S, C, M)
    
    # Z_hat = (1/√M) Σ_μ F * product
    Z_hat = alpha_scale * (F * product).sum(dim=2)  # (S, C)
    
    return Z_hat
```

**Step 2: Add compute_variance_tensor_batch**

```python
def compute_variance_tensor_batch(
    factors: List[torch.Tensor],      # n tensors of (S, N_d, M)
    factor_vars: List[torch.Tensor],  # n tensors of (S, N_d, M)
    F: torch.Tensor,                  # (S, C, M)
    indices: List[torch.Tensor],      # n tensors of (C,)
    is_rademacher: bool = False,
) -> torch.Tensor:
    """
    Batched variance computation.
    
    Returns:
        V: (S, C) variance at each hyperedge
    """
    S, C, M = F.shape
    n = len(factors)
    alpha_scale_sq = 1.0 / M
    
    # Gather factors and variances
    gathered = torch.stack([
        factors[d][:, indices[d].long()] for d in range(n)
    ])  # (n, S, C, M)
    gathered_var = torch.stack([
        factor_vars[d][:, indices[d].long()] for d in range(n)
    ])  # (n, S, C, M)
    
    F_sq = torch.ones_like(F) if is_rademacher else F.pow(2)
    all_sq = gathered.pow(2)
    product_all_sq = all_sq.prod(dim=0)  # (S, C, M)
    
    V_sum = torch.zeros(S, C, M, device=F.device, dtype=F.dtype)
    for d in range(n):
        other_product = product_all_sq / (all_sq[d] + 1e-10)
        V_sum += gathered_var[d] * other_product
    
    V = alpha_scale_sq * (F_sq * V_sum).sum(dim=2) + 1e-10
    return V
```

**Step 3: Add tensor_step_batch (full iteration)**

```python
def tensor_step_batch(
    factors: List[torch.Tensor],
    factor_vars: List[torch.Tensor],
    Y: torch.Tensor,                  # (S, C)
    F: torch.Tensor,                  # (S, C, M)
    indices: List[torch.Tensor],      # n tensors of (C,)
    damping: float,
    noise_var: float,
    is_rademacher: bool = False,
    prev_s: Optional[torch.Tensor] = None,
    onsager_correction: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Batched BiG-AMP step for n-dimensional tensor.
    
    Returns:
        new_factors, new_factor_vars, s_values
    """
    n = len(factors)
    S, C, M = F.shape
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M
    
    Z_hat = forward_pass_tensor_batch(factors, F, indices)
    V = compute_variance_tensor_batch(factors, factor_vars, F, indices, is_rademacher)
    
    if onsager_correction and prev_s is not None:
        correction = V * prev_s
        correction = torch.clamp(correction, min=-0.5, max=0.5)
        Z_hat = Z_hat - correction
    
    denom = torch.clamp(V + noise_var, min=1e-6)
    s_values = (Y - Z_hat) / denom
    s_values = torch.clamp(s_values, min=-1e6, max=1e6)
    
    gathered = torch.stack([
        factors[d][:, indices[d].long()] for d in range(n)
    ])
    F_sq = torch.ones_like(F) if is_rademacher else F.pow(2)
    
    new_factors = []
    new_vars = []
    
    for d in range(n):
        N_d = factors[d].shape[1]
        
        if n > 1:
            other_indices = [dd for dd in range(n) if dd != d]
            other_gathered = torch.stack([gathered[dd] for dd in other_indices])
            other_product = other_gathered.prod(dim=0)
        else:
            other_product = torch.ones(S, C, M, device=F.device, dtype=F.dtype)
        
        r_contrib = alpha_scale * F * other_product * s_values.unsqueeze(2)
        
        if n > 1:
            other_sq = torch.stack([gathered[dd].pow(2) for dd in other_indices])
            other_sq_product = other_sq.prod(dim=0)
        else:
            other_sq_product = torch.ones(S, C, M, device=F.device, dtype=F.dtype)
        tau_contrib = alpha_scale_sq * F_sq * other_sq_product / denom.unsqueeze(2)
        
        # Scatter add with batch dimension
        r_d = torch.zeros(S, N_d, M, device=F.device, dtype=F.dtype)
        tau_d = torch.zeros(S, N_d, M, device=F.device, dtype=F.dtype)
        idx_exp = indices[d].long().unsqueeze(0).unsqueeze(2).expand(S, -1, M)
        r_d.scatter_add_(1, idx_exp, r_contrib)
        tau_d.scatter_add_(1, idx_exp, tau_contrib)
        tau_d = tau_d.clamp(min=1e-10)
        
        new_var_d = 1.0 / tau_d
        new_var_d = new_var_d.clamp(max=1.0)
        new_factor_d = new_var_d * (tau_d * factors[d] + r_d)
        new_factor_d = torch.clamp(new_factor_d, min=-10.0, max=10.0)
        
        new_factor_d = damping * factors[d] + (1 - damping) * new_factor_d
        new_var_d = damping * factor_vars[d] + (1 - damping) * new_var_d
        
        new_factors.append(new_factor_d)
        new_vars.append(new_var_d)
    
    return new_factors, new_vars, s_values
```

**Step 4: Run verification test**

Run: `python -c "from matrix_factorization.modules.algorithms.bigamp.tensor_step_batch import *; print('Import OK')"`

Expected: `Import OK`

**Step 5: Commit**

```bash
git add src/matrix_factorization/modules/algorithms/bigamp/tensor_step_batch.py
git commit -m "feat(tensor): add batched tensor step functions for sample parallelization"
```

---

## Task 2: 添加 generate_F_batch 到 tensor_hypergraph.py

**Files:**
- Modify: `src/matrix_factorization/modules/algorithms/bigamp/tensor_hypergraph.py`

**Step 1: Add generate_F_batch function**

在文件末尾添加：

```python
def generate_F_batch(
    S: int,
    C: int,
    M: int,
    base_seed: int,
    device: torch.device,
    distribution: str = 'rademacher',
) -> torch.Tensor:
    """
    Generate S independent F realizations.
    
    Args:
        S: Number of samples
        C: Number of hyperedges
        M: Latent dimension
        base_seed: Base random seed
        device: torch device
        distribution: 'rademacher' or 'gaussian'
        
    Returns:
        F: (S, C, M) spreading coefficients
    """
    F_list = []
    for s in range(S):
        gen = torch.Generator(device=device).manual_seed(base_seed + s * 1000)
        if distribution == 'rademacher':
            F_s = (torch.randint(0, 2, (C, M), generator=gen, device=device) * 2 - 1).float()
        else:
            F_s = torch.randn(C, M, generator=gen, device=device)
        F_list.append(F_s)
    
    return torch.stack(F_list)  # (S, C, M)
```

**Step 2: Commit**

```bash
git add src/matrix_factorization/modules/algorithms/bigamp/tensor_hypergraph.py
git commit -m "feat(tensor): add generate_F_batch for sample parallelization"
```

---

## Task 3: 创建 tensor_spreading_parallel.py

**Files:**
- Create: `src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading_parallel.py`
- Reference: `src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading.py`

**Step 1: Create file with class skeleton**

复制 `BiGAMPTensorSpreading` 类结构，重命名为 `BiGAMPTensorSpreadingParallel`。

**Step 2: Modify `_train_batch_samples` to process all S samples in parallel**

（详细代码见实施阶段）

**Step 3: Register new algorithm**

```python
@register_algorithm(
    key="bigamp_tensor_parallel",
    name="BiG-AMP Tensor Parallel",
    description="Tensor CP decomposition with sample parallelization",
)
```

**Step 4: Test import**

Run: `python -c "from matrix_factorization.modules.algorithms.bigamp.tensor_spreading_parallel import *; print('Import OK')"`

**Step 5: Commit**

```bash
git add src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading_parallel.py
git commit -m "feat(tensor): add parallel tensor spreading algorithm"
```

---

## Task 4: 功能验证

**Step 1: Create test comparing serial vs parallel**

**Step 2: Run comparison**

预期：`Q_Y` 差异 < 1e-5

**Step 3: Commit test**

---

## Task 5: 性能优化 - BF16

**Files:**
- Modify: `tensor_spreading_parallel.py`

**Step 1: Add BF16 support**

```python
self.use_bf16 = False
self.storage_dtype = torch.float32
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    self.use_bf16 = True
    self.storage_dtype = torch.bfloat16
```

**Step 2: Commit**

---

## Task 6: 性能优化 - torch.compile

**Step 1: Add torch.compile with mode selection**

**Step 2: Add CUDA Graph mark**

**Step 3: Commit**

---

## Open Questions

1. 显存估算使用公式还是试探法？（建议：试探法更可靠）
2. Phase 3 (TensorSuperGraph) 的优先级？（建议：Phase 1+1.5 完成后评估）
