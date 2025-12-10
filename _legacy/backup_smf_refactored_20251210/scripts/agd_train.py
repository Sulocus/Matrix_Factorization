"""
AGD Training Script

Alternating Gradient Descent for masked matrix factorization.

Auto-generated from SMF framework.

SMF Git: 1cffaf8f (main)
Generated: 2025-12-05 23:00:06

This script is standalone - no external smf imports required.
You can copy it directly to Wang/ for sharing.
"""

from pathlib import Path
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import deque
import itertools

# ============================================================
# Default Parameters
# ============================================================
N1 = 200
N2 = 200
M = 50

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 3
ALPHA_TILDE_STEP = 0.1

LEARNING_RATE = 1e-2
MAX_EPOCHS = 20000

SAMPLES_PER_ALPHA = 5
RESAMPLE_MASK_EACH_TRIAL = True
SEED = 42

USE_BIREGULAR_GRAPH = False
USE_EARLY_STOP = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))

USE_BF16 = DEVICE.type == 'cuda'
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

RESULT_DIR = Path(__file__).parent.parent / "results/standard" / f"{N1}_{N2}_{M}"


# ============================================================
# Utility Functions
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_teacher(N1, N2, M, device, seed=42):
    torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale
    return W, X


def sample_pairs_random_gpu(N1, N2, C, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    total = N1 * N2
    if C == 0 or C > total:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0 if C == 0 else -1)
    idx = torch.randperm(total, device=device)[:C]
    return idx // N2, idx % N2, C


def sample_mask(N1, N2, M, alpha, device, seed=None):
    deg = int(round(alpha * M))
    C = N1 * max(0, min(deg, N2))
    i_idx, j_idx, _ = sample_pairs_random_gpu(N1, N2, C, device, seed)
    mask = torch.zeros((N1, N2), device=device, dtype=torch.float32)
    if i_idx.numel() > 0:
        mask[i_idx, j_idx] = 1.0
    return mask


@torch.no_grad()
def compute_cosine_similarity(A, B, use_left=True):
    G_A = A @ A.T if use_left else A.T @ A
    G_B = B @ B.T if use_left else B.T @ B
    return float((G_A.flatten() * G_B.flatten()).sum() /
                 (G_A.flatten().norm() * G_B.flatten().norm() + 1e-12))


@torch.no_grad()
def compute_qy(Y_student, Y_teacher):
    y_s, y_t = Y_student.flatten(), Y_teacher.flatten()
    return float((y_s * y_t).sum() / (y_s.norm() * y_t.norm() + 1e-12))


# ============================================================
# AGD Training
# ============================================================
def train_agd(W_teacher, X_teacher, Y_teacher, mask, alpha, seed,
              lr=1e-2, max_epochs=20000, S=1):
    """AGD training for single alpha"""
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    device = W_teacher.device

    alpha_scale = scale = 1.0 / (M ** 0.5)
    A = mask.unsqueeze(0)
    Y_b = Y_teacher.unsqueeze(0)

    torch.manual_seed(seed)
    W = torch.randn((S, N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((S, M, N2), device=device, dtype=torch.float32) * scale

    for _ in tqdm(range(max_epochs), desc=f"AGD α={alpha:.2f}", leave=False, mininterval=1.0):
        with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
            Y_s = alpha_scale * torch.matmul(W, X)
            Mres = (Y_b - Y_s) * A
            grad_W = -2.0 * alpha_scale * torch.matmul(Mres, X.transpose(1, 2))
        W = W - lr * grad_W.float()

        with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE, enabled=USE_BF16):
            Y_s2 = alpha_scale * torch.matmul(W, X)
            Mres2 = (Y_b - Y_s2) * A
            grad_X = -2.0 * alpha_scale * torch.matmul(W.transpose(1, 2), Mres2)
        X = X - lr * grad_X.float()

    return W.float(), X.float()


# ============================================================
# Main
# ============================================================
def main():
    print(f"[Device] {DEVICE}")
    print(f"[Size] {N1}x{N2}, M={M}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    W_t, X_t = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_t = W_t @ X_t

    alphas = np.arange(ALPHA_TILDE_START, ALPHA_TILDE_STOP + 1e-9, ALPHA_TILDE_STEP)
    results = []

    for alpha in tqdm(alphas, desc="Alpha sweep"):
        mask = sample_mask(N1, N2, M, alpha, DEVICE, seed=SEED if not RESAMPLE_MASK_EACH_TRIAL else None)
        W_s, X_s = train_agd(W_t, X_t, Y_t, mask, alpha, SEED,
                             lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, S=SAMPLES_PER_ALPHA)

        Y_s = W_s[0] @ X_s[0]
        q_y = compute_qy(Y_s, Y_t)
        q_w = compute_cosine_similarity(W_s[0], W_t)
        q_x = compute_cosine_similarity(X_s[0], X_t, use_left=False)

        results.append({"alpha": float(alpha), "Q_Y": q_y, "Q_W": q_w, "Q_X": q_x})
        print(f"α={alpha:.2f}: Q_Y={q_y:.4f}")

    with open(RESULT_DIR / "results_agd.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULT_DIR / 'results_agd.json'}")


if __name__ == "__main__":
    main()
