# Random Spreading Teacher Module

## Overview

The random spreading model introduces quenched randomness through spreading coefficients `F`, creating a disordered system that is theoretically tractable via replica analysis.

## Physical Model

$$Y_{ij} = \frac{1}{\sqrt{M}} \sum_{\mu=1}^{M} F_{ij,\mu} W_{i\mu} X_{\mu j}$$

where:
- `F_{ij,μ}` ~ N(0,1) are **quenched random** spreading coefficients
- Each observed position (i,j) has its own random F vector of length M
- The disorder is "annealed" in the sense that F is fixed during optimization

## Key Components

### SpreadingData

```python
@dataclass
class SpreadingData:
    i_idx: torch.Tensor      # (C,) row indices of observed positions
    j_idx: torch.Tensor      # (C,) column indices of observed positions
    F: torch.Tensor          # (C, M) spreading coefficients
    Y_values: torch.Tensor   # (C,) observed Y values at these positions
    seed: int                # Seed for reproducibility
    M: int                   # Latent dimension
```

### Functions

#### `generate_spreading_coefficients(i_idx, j_idx, M, seed, device)`
Generates deterministic F coefficients for given observation positions.

**Important**: Same (seed, i, j) always produces the same F vector, enabling:
- Reproducibility across runs
- Memory-efficient sparse storage (only store observed positions)

#### `compute_sparse_Y(W, X, F, i_idx, j_idx)`
Computes Y values at observed positions using the spreading model.

**Parameter Order**: `(W, X, F, i_idx, j_idx)` - F comes before indices!

## Algorithm Integration

### BiGAMPSpreadingAlgorithm

Located in `smf/modules/algorithms/bigamp_spreading.py`

**Key Difference from Standard BiG-AMP**:
- Each alpha has different observation positions → different F coefficients
- Cannot use true batch parallelization across alphas
- `supports_batch_training()` returns `False`

### Memory Efficiency

Standard model: O(N1 × N2) for full Y matrix
Spreading model: O(C × M) for F coefficients, where C = number of observations

For sparse observations (small α), this is much more efficient.

## Usage Example

```python
from smf.modules.teachers.random_spreading import (
    SpreadingData,
    generate_spreading_coefficients,
    compute_sparse_Y
)

# Given a mask
i_idx, j_idx = torch.where(mask > 0)
C = i_idx.shape[0]

# Generate F coefficients
F = generate_spreading_coefficients(i_idx, j_idx, M, seed=42, device='cuda')

# Compute Y values
Y_values = compute_sparse_Y(W_teacher, X_teacher, F, i_idx, j_idx)

# Create SpreadingData
spreading_data = SpreadingData(
    i_idx=i_idx,
    j_idx=j_idx,
    F=F,
    Y_values=Y_values,
    seed=42,
    M=M
)
```

## SMF CLI Usage

```bash
smf
# Choose: [4] Custom Configuration
# Algorithm: [2] BiG-AMP Random Spreading
# Graph: [1] Random Graph (GPU)
# Configure other parameters as needed
```

## Related Files

- `smf/modules/teachers/random_spreading.py` - Core implementation
- `smf/modules/algorithms/bigamp_spreading.py` - BiG-AMP algorithm for spreading
- `smf/modules/metrics/spreading.py` - Q_Y evaluation for spreading
- `tests/test_random_spreading.py` - Unit tests (31 tests)
