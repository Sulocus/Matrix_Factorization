# Matrix Factorization (MF)

A Teacher-Student masked matrix factorization framework with GPU acceleration.

## Features

- **BiG-AMP Spreading Algorithm**: Advanced message passing algorithm for matrix factorization
- **GPU Acceleration**: Full CUDA support with PyTorch backend
- **Parallel Execution**: Automatic batching and parallelization across alpha values
- **Multiple Teacher Types**: Standard, Orthogonal, Scaled Variance, Random Spreading

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Run with default config
mf run

# Run with custom config
mf run --config MF/config.yaml

# Show available presets
mf presets
```

## Project Structure

```
Matrix_Factorization/
├── MF/                     # Main package
│   ├── core/               # Core execution engine
│   ├── modules/            # Algorithms, teachers, metrics, outputs
│   ├── presets/            # Configuration presets
│   └── ui/                 # User interface components
├── MF_docs/                # Documentation
├── tests/                  # Test suite
└── scripts/                # Utility scripts
```

## License

MIT
