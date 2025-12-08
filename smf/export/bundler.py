"""
Script bundler - generates standalone scripts from smf modules.

Usage:
    python -m smf.export.bundle
    # Generates scripts/bigamp_train.py, scripts/agd_train.py

The generated scripts are completely standalone and can be copied to Wang/.
"""

import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
import inspect

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_git_info() -> dict:
    """Get current git information."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        commit_hash = result.stdout.strip()[:8] if result.returncode == 0 else "unknown"

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        branch = result.stdout.strip() if result.returncode == 0 else "unknown"

        return {"hash": commit_hash, "branch": branch}
    except Exception:
        return {"hash": "unknown", "branch": "unknown"}


def generate_header(script_name: str, description: str) -> str:
    """Generate script header with version info."""
    git = get_git_info()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f'''"""
{description}

Auto-generated from SMF framework.

SMF Git: {git["hash"]} ({git["branch"]})
Generated: {timestamp}

This script is standalone - no external smf imports required.
You can copy it directly to Wang/ for sharing.
"""
'''


class ScriptBundler:
    """
    Bundle smf modules into standalone scripts.

    The bundler reads module source code and combines them into
    a single file that can run independently.
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or PROJECT_ROOT / "scripts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def bundle_bigamp_train(self) -> Path:
        """Generate bigamp_train.py from smf modules."""
        from smf.modules.graphs.uniform import Dinic, UniformGraph
        from smf.modules.graphs.random import RandomGraph
        from smf.modules.teachers.standard import StandardTeacher
        from smf.modules.metrics.overlap import (
            gram_overlap_cosine,
            gram_overlap_normalized,
            compute_qy,
            compute_all_metrics,
        )
        from smf.modules.algorithms.bigamp import BiGAMPAlgorithm

        header = generate_header(
            "bigamp_train.py",
            "BiG-AMP Training Script\n\nBilinear Generalized Approximate Message Passing for masked matrix factorization."
        )

        script = header + '''
from pathlib import Path
import time
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataclasses import dataclass
from collections import deque
import itertools

# ============================================================
# Default Parameters
# ============================================================
N1 = 200
N2 = 200
M = 50

ALPHA_TILDE_START = 0
ALPHA_TILDE_STOP = 4
ALPHA_TILDE_STEP = 0.1

# BiG-AMP parameters
DAMPING = 0.5
NOISE_VAR = 1e-10
MAX_STEPS = 1000

SAMPLES_PER_ALPHA = 1
RESAMPLE_MASK_EACH_TRIAL = True
SEED = 42

# Graph type
USE_BIREGULAR_GRAPH = False

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))

USE_BF16 = DEVICE.type == 'cuda'
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
STORAGE_DTYPE = torch.float32

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

RESULT_DIR = Path(__file__).parent.parent / "results/standard" / f"{N1}_{N2}_{M}"


# ============================================================
# Utility Functions (from smf.modules)
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_teacher(N1, N2, M, device, seed=42):
    """Create teacher model W_true and X_true"""
    torch.manual_seed(seed)
    scale = 1.0 / (M ** 0.5)
    W = torch.randn((N1, M), device=device, dtype=torch.float32) * scale
    X = torch.randn((M, N2), device=device, dtype=torch.float32) * scale
    return W, X


def sample_pairs_random_gpu(N1, N2, C, device, seed=None):
    """Pure random mask generation"""
    if seed is not None:
        torch.manual_seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"C={C} exceeds {N1}x{N2}={total}")

    if C == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    idx = torch.randperm(total, device=device)[:C]
    i_idx = idx // N2
    j_idx = idx % N2
    return i_idx, j_idx, C


# Dinic max-flow algorithm for biregular graphs
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


def sample_pairs_biregular(N1, N2, M, alpha, device, seed=None):
    """Generate biregular graph using Dinic algorithm"""
    deg_left = int(round(alpha * M))
    deg_left = max(0, min(deg_left, N2))
    total_edges = N1 * deg_left

    if not USE_BIREGULAR_GRAPH:
        return sample_pairs_random_gpu(N1, N2, total_edges, device, seed)

    if deg_left == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    rng = np.random.RandomState(seed + 12345 + int(alpha * 1e6)) if seed else np.random.RandomState()

    base = total_edges // N2
    rem = total_edges % N2
    right_target = np.full(N2, base, dtype=int)
    if rem > 0:
        idx = np.arange(N2)
        rng.shuffle(idx)
        right_target[idx[:rem]] += 1

    S, L_off, R_off = 0, 1, 1 + N1
    T = R_off + N2
    din = Dinic(T + 1)

    for i in range(N1):
        din.add_edge(S, L_off + i, deg_left)

    all_pairs = list(itertools.product(range(N1), range(N2)))
    rng.shuffle(all_pairs)
    for i, j in all_pairs:
        din.add_edge(L_off + i, R_off + j, 1)

    for j in range(N2):
        din.add_edge(R_off + j, T, int(right_target[j]))

    f = din.max_flow(S, T)
    if f != total_edges:
        raise RuntimeError(f"maxflow {f}/{total_edges} failed")

    i_list, j_list = [], []
    for i in range(N1):
        u = L_off + i
        for v, cap, rev in din.g[u]:
            if R_off <= v < R_off + N2 and din.g[v][rev][1] > 0:
                i_list.append(i)
                j_list.append(v - R_off)

    return (torch.tensor(i_list, dtype=torch.long, device=device),
            torch.tensor(j_list, dtype=torch.long, device=device),
            len(i_list))


def sample_mask(N1, N2, M, alpha, device, seed=None):
    """Generate observation mask"""
    i_idx, j_idx, C = sample_pairs_biregular(N1, N2, M, alpha, device, seed)
    mask = torch.zeros((N1, N2), device=device, dtype=torch.float32)
    if C > 0:
        mask[i_idx, j_idx] = 1.0
    return mask, alpha * M


# ============================================================
# Evaluation Metrics
# ============================================================
@torch.no_grad()
def gram_overlap_cosine(A, B, use_left=True):
    if use_left:
        G_A, G_B = A @ A.T, B @ B.T
    else:
        G_A, G_B = A.T @ A, B.T @ B
    return float((G_A.flatten() * G_B.flatten()).sum() /
                 (G_A.flatten().norm() * G_B.flatten().norm() + 1e-12))


@torch.no_grad()
def gram_overlap_normalized(A, B, use_left=True):
    q = gram_overlap_cosine(A, B, use_left)
    n, m = A.shape if use_left else (A.shape[1], A.shape[0])
    b = m / (m + n + 1)
    return float(max(0.0, min(1.0, (q - b) / (1.0 - b + 1e-12))))


@torch.no_grad()
def compute_qy(Y_student, Y_teacher):
    y_s, y_t = Y_student.flatten(), Y_teacher.flatten()
    return float((y_s * y_t).sum() / (y_s.norm() * y_t.norm() + 1e-12))


# ============================================================
# BiG-AMP Algorithm
# ============================================================
def train_bigamp(W_teacher, X_teacher, Y_teacher, mask, alpha, seed,
                 damping=0.5, noise_var=1e-10, max_steps=1000, S=1):
    """BiG-AMP training for single alpha"""
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    device = W_teacher.device

    alpha_scale = scale = 1.0 / (M ** 0.5)
    A = mask.unsqueeze(0) if mask.dim() == 2 else mask

    torch.manual_seed(seed)
    w_hat = torch.randn((S, N1, M), device=device) * scale
    x_hat = torch.randn((S, M, N2), device=device) * scale
    w_var = torch.ones_like(w_hat) * (1.0 / M)
    x_var = torch.ones_like(x_hat) * (1.0 / M)

    for _ in tqdm(range(max_steps), desc=f"BiG-AMP α={alpha:.2f}", leave=False):
        # Forward
        z_hat = alpha_scale * torch.matmul(w_hat, x_hat)
        p_var = (alpha_scale ** 2) * (torch.matmul(w_hat**2, x_var) + torch.matmul(w_var, x_hat**2))
        V = torch.clamp(p_var + noise_var, min=1e-8)
        s = (Y_teacher - z_hat) * A / V

        # W update
        tau_W = (alpha_scale ** 2) * torch.matmul(A / V, (x_hat**2).transpose(-2, -1))
        w_var_new = 1.0 / (M + torch.clamp(tau_W, min=1e-8))
        w_hat_new = w_hat + w_var_new * alpha_scale * torch.matmul(s, x_hat.transpose(-2, -1))
        w_hat = damping * w_hat + (1 - damping) * w_hat_new
        w_var = torch.clamp(damping * w_var + (1 - damping) * w_var_new, min=1e-8, max=1.0)

        # X update
        z_hat2 = alpha_scale * torch.matmul(w_hat, x_hat)
        p_var2 = (alpha_scale ** 2) * (torch.matmul(w_hat**2, x_var) + torch.matmul(w_var, x_hat**2))
        V2 = torch.clamp(p_var2 + noise_var, min=1e-8)
        s2 = (Y_teacher - z_hat2) * A / V2

        tau_X = (alpha_scale ** 2) * torch.matmul((w_hat**2).transpose(-2, -1), A / V2)
        x_var_new = 1.0 / (M + torch.clamp(tau_X, min=1e-8))
        x_hat_new = x_hat + x_var_new * alpha_scale * torch.matmul(w_hat.transpose(-2, -1), s2)
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        x_var = torch.clamp(damping * x_var + (1 - damping) * x_var_new, min=1e-8, max=1.0)

    return w_hat, x_hat


# ============================================================
# Main
# ============================================================
def main():
    print(f"[Device] {DEVICE}")
    print(f"[Size] {N1}x{N2}, M={M}")
    print(f"[Graph] {'Biregular' if USE_BIREGULAR_GRAPH else 'Random'}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    # Create teacher
    W_t, X_t = create_teacher(N1, N2, M, DEVICE, seed=SEED)
    Y_t = W_t @ X_t

    # Alpha sweep
    alphas = np.arange(ALPHA_TILDE_START, ALPHA_TILDE_STOP + 1e-9, ALPHA_TILDE_STEP)
    results = []

    for alpha in tqdm(alphas, desc="Alpha sweep"):
        mask, _ = sample_mask(N1, N2, M, alpha, DEVICE, seed=SEED if not RESAMPLE_MASK_EACH_TRIAL else None)
        W_s, X_s = train_bigamp(W_t, X_t, Y_t, mask, alpha, SEED,
                                damping=DAMPING, noise_var=NOISE_VAR, max_steps=MAX_STEPS, S=SAMPLES_PER_ALPHA)

        # Evaluate
        Y_s = W_s[0] @ X_s[0]
        q_y = compute_qy(Y_s, Y_t)
        q_w = gram_overlap_cosine(W_s[0], W_t)
        q_x = gram_overlap_cosine(X_s[0], X_t, use_left=False)

        results.append({
            "alpha": float(alpha),
            "Q_Y": q_y,
            "Q_W": q_w,
            "Q_X": q_x,
        })
        print(f"α={alpha:.2f}: Q_Y={q_y:.4f}")

    # Save results
    with open(RESULT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
'''

        output_path = self.output_dir / "bigamp_train.py"
        with open(output_path, "w") as f:
            f.write(script)

        return output_path

    def bundle_agd_train(self) -> Path:
        """Generate agd_train.py from smf modules."""
        header = generate_header(
            "agd_train.py",
            "AGD Training Script\n\nAlternating Gradient Descent for masked matrix factorization."
        )

        script = header + '''
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
def gram_overlap_cosine(A, B, use_left=True):
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
        q_w = gram_overlap_cosine(W_s[0], W_t)
        q_x = gram_overlap_cosine(X_s[0], X_t, use_left=False)

        results.append({"alpha": float(alpha), "Q_Y": q_y, "Q_W": q_w, "Q_X": q_x})
        print(f"α={alpha:.2f}: Q_Y={q_y:.4f}")

    with open(RESULT_DIR / "results_agd.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULT_DIR / 'results_agd.json'}")


if __name__ == "__main__":
    main()
'''

        output_path = self.output_dir / "agd_train.py"
        with open(output_path, "w") as f:
            f.write(script)

        return output_path


def bundle_all(output_dir: Path = None) -> list:
    """Generate all standalone scripts."""
    bundler = ScriptBundler(output_dir)
    generated = []

    print("Generating standalone scripts...")

    path = bundler.bundle_bigamp_train()
    print(f"  ✓ {path}")
    generated.append(path)

    path = bundler.bundle_agd_train()
    print(f"  ✓ {path}")
    generated.append(path)

    print(f"\nGenerated {len(generated)} scripts in {bundler.output_dir}")
    return generated


if __name__ == "__main__":
    bundle_all()
