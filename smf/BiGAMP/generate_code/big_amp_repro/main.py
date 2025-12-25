import argparse
import sys
import numpy as np
from big_amp_repro.experiments.run_synthetic_ptc import run_synthetic_ptc
from big_amp_repro.experiments.run_movielens import run_movielens_experiment
from big_amp_repro.experiments.run_video_rpca import run_video_rpca_experiment
from big_amp_repro.apps.dict_learning import DictLearning
from big_amp_repro.utils.metrics import compute_nmse

def run_dict_learning_demo():
    """
    A simple demonstration of Dictionary Learning (Section VIII).
    """
    print("Running Dictionary Learning Demo...")
    M, L, N = 100, 100, 50
    sparsity = 0.1
    
    # Generate synthetic DL data
    A_true = np.random.randn(M, N)
    X_true = (np.random.rand(N, L) < sparsity) * np.random.randn(N, L)
    Y_true = A_true @ X_true
    noise_var = 1e-6
    Y = Y_true + np.sqrt(noise_var) * np.random.randn(M, L)
    
    dl = DictLearning(rank=N, nit=100, use_em=True, verbose=True)
    dl.fit(Y)
    
    metrics = dl.get_metrics(A_true, X_true, Y_true)
    print(f"Dictionary Learning Results:")
    print(f"  NMSE (Y): {metrics['nmse_y']:.2f} dB")
    print(f"  A Correlation: {metrics['a_corr']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Bilinear Generalized Approximate Message Passing (BiG-AMP) Reproduction")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # PTC Experiment
    subparsers.add_parser("ptc", help="Run Synthetic Phase Transition Curve experiment (Fig 2)")

    # MovieLens Experiment
    subparsers.add_parser("movielens", help="Run MovieLens 100k Matrix Completion experiment (Fig 7)")

    # Video RPCA Experiment
    subparsers.add_parser("video", help="Run Video Robust PCA experiment (Fig 11)")

    # Dictionary Learning Demo
    subparsers.add_parser("dict", help="Run Dictionary Learning demonstration (Section VIII)")

    args = parser.parse_args()

    if args.command == "ptc":
        print("Starting Phase Transition Curve experiment...")
        run_synthetic_ptc()
    elif args.command == "movielens":
        print("Starting MovieLens 100k experiment...")
        run_movielens_experiment()
    elif args.command == "video":
        print("Starting Video RPCA experiment...")
        run_video_rpca_experiment()
    elif args.command == "dict":
        run_dict_learning_demo()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
