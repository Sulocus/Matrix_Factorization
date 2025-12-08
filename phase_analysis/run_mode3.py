#!/usr/bin/env python3
"""
Mode 3: Precise Phase Transition Analysis

Determines the exact phase transition point alpha_c through:
1. Progressive refinement with increasing epochs
2. Convergence detection
3. Extrapolation to thermodynamic limit (steps -> infinity)

Usage:
    python run_mode3.py                          # Default 200x200 M=50
    python run_mode3.py --N1 2000 --M 100        # Custom dimensions
    python run_mode3.py --steps 200,400,800,1600 # Custom step levels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse

from core.precise_phase_analyzer import PrecisePhaseAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Mode 3: Precise Phase Analysis")
    parser.add_argument("--N1", type=int, default=200, help="Matrix dimension N1")
    parser.add_argument("--N2", type=int, default=None, help="Matrix dimension N2 (default: same as N1)")
    parser.add_argument("--M", type=int, default=50, help="Latent dimension M")
    parser.add_argument("--samples", type=int, default=5, help="Samples per alpha")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated list of step levels (auto-scaled if not specified)")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Convergence threshold for alpha_c")
    parser.add_argument("--patience", type=int, default=2,
                        help="Convergence patience (stable rounds needed)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    N2 = args.N2 if args.N2 is not None else args.N1

    # Parse step_levels or use auto-scaling
    if args.steps is not None:
        step_levels = [int(s.strip()) for s in args.steps.split(",")]
    else:
        step_levels = None  # Will be auto-scaled based on matrix size

    # Output directory
    result_dir = Path(__file__).parent.parent / "Result" / f"{args.N1}_{N2}_{args.M}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analyzer = PrecisePhaseAnalyzer(
        N1=args.N1,
        N2=N2,
        M=args.M,
        samples_per_alpha=args.samples
    )

    result = analyzer.analyze(
        step_levels=step_levels,
        convergence_threshold=args.threshold,
        convergence_patience=args.patience,
        verbose=not args.quiet
    )

    # Save results
    analyzer.save_results(result, result_dir)

    # Plot convergence
    plot_path = result_dir / f"mode3_convergence_{args.N1}x{N2}_M{args.M}.png"
    analyzer.plot_convergence(result, plot_path)

    print(f"\nOutput files:")
    print(f"  JSON: {result_dir}/mode3_precise_{args.N1}x{N2}_M{args.M}.json")
    print(f"  PNG:  {plot_path}")


if __name__ == "__main__":
    main()
