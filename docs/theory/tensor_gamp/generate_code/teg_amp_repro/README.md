# Tensor Generalized Approximate Message Passing (TeG-AMP) Reproduction

This repository contains a Python implementation of the **TeG-AMP** and **TeS-AMP** algorithms for low-rank tensor inference, as described in the paper *"Tensor Generalized Approximate Message Passing"*.

## Project Overview

TeG-AMP is a message-passing framework designed for recovering low-rank tensors from noisy, incomplete observations. It supports both **Tensor Ring (TR)** and **Canonical Polyadic (CP)** decomposition models. Key features include:
- **Adaptive Damping**: Ensures convergence in finite-dimensional settings using a KL-divergence-based cost function.
- **Taylor-Series Variance Updates**: Efficiently approximates message variances for high-order TR contractions.
- **Multiple Solvers**: Includes TeG-AMP (TR), TeS-AMP (CP), and AltMin (Alternating Least Squares) baselines.

## File Structure

```text
teg_amp_repro/
├── solvers/            # TeG-AMP, TeS-AMP, and AltMin implementations
├── models/             # TR and CP contraction logic
├── utils/              # Damping, Priors, Channels, and Tensor Math
├── data/               # Synthetic generators and MNIST loader
├── experiments/        # Scripts to reproduce paper figures
└── main.py             # Unified entry point
```

## Installation

### Requirements
- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- opt_einsum >= 3.3.0 (Critical for optimized contractions)
- Matplotlib >= 3.4.0

### Setup
```bash
pip install -r requirements.txt
```

## Running Experiments

The `main.py` script provides a CLI to run the reproduction experiments. Results (plots and metrics) are saved to the `results/` directory.

### 1. Synthetic Phase Transition (Figure 2)
Reproduces the recovery performance vs. sampling rate for TR-rank [2,2,2] and [2,3,3] tensors.
```bash
python main.py --experiment synthetic --trials 10
```

### 2. MNIST Tensor Completion (Figure 3)
Reproduces visual digit reconstruction from 40% sampled pixels using a $28 \times 28 \times 6$ tensor.
```bash
python main.py --experiment mnist --sampling_rate 0.4
```

### 3. SNR Robustness Comparison
Compares TeG-AMP, TeS-AMP, and AltMin across different noise levels.
```bash
python main.py --experiment snr --trials 5
```

## Implementation Details

- **TeG-AMP (Algorithm 1)**: Implements the subset-sum variance update (Eq 25) and partial contractions for core updates.
- **TeS-AMP (Algorithm 2)**: Optimized for CP-rank tensors with exact variance calculations.
- **Adaptive Damping**: Implements the $J(t)$ cost criterion (Appendix G) to dynamically adjust the damping factor $\beta$.
- **Priors & Channels**: Supports Gaussian priors and AWGN channels with adaptive noise variance estimation.

## License
This project is for reproduction and research purposes.
