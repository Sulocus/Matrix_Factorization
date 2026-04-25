import argparse
import os
import sys
from teg_amp_repro.experiments import (
    run_synthetic_experiment,
    run_mnist_experiment,
    run_snr_experiment
)

def main():
    parser = argparse.ArgumentParser(description="TeG-AMP Reproduction Framework")
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        choices=["synthetic", "mnist", "snr", "all"],
        default="synthetic",
        help="The experiment to run (default: synthetic)"
    )
    
    parser.add_argument(
        "--trials", 
        type=int, 
        default=10,
        help="Number of Monte Carlo trials for synthetic/snr experiments (default: 10)"
    )
    
    parser.add_argument(
        "--sampling_rate", 
        type=float, 
        default=0.4,
        help="Sampling rate for MNIST experiment (default: 0.4)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Directory to save plots and results (default: results)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"=== TeG-AMP Reproduction: Running {args.experiment} experiment ===")

    if args.experiment == "synthetic" or args.experiment == "all":
        print("\n--- Running Synthetic Phase Transition (Figure 2) ---")
        run_synthetic_experiment(trials=args.trials, save_path=args.output_dir)

    if args.experiment == "mnist" or args.experiment == "all":
        print("\n--- Running MNIST Tensor Completion (Figure 3) ---")
        run_mnist_experiment(sampling_rate=args.sampling_rate, save_path=args.output_dir)

    if args.experiment == "snr" or args.experiment == "all":
        print("\n--- Running SNR Robustness Comparison ---")
        run_snr_experiment(trials=args.trials, save_path=args.output_dir)

    print(f"\nExperiments completed. Results saved to '{args.output_dir}/'")

if __name__ == "__main__":
    main()
