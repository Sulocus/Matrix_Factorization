#!/usr/bin/env python3
"""
Run large matrix sweep experiment.

User request:
- Sizes: 1000x100, 2000x100, 3000x100, 4000x100, 5000x100
- Alpha: 0 to 0.5, step 0.01
- Steps: 10000
"""

import sys
sys.path.insert(0, '/home/sucia/Sparse-Matrix')

from smf.experiments.large_matrix_sweep import LargeMatrixSweepExperiment

# Matrix configurations as requested (N1, N2, M)
# Note: N1=N2=N for square matrices, M=100
MATRIX_CONFIGS = [
    (1000, 1000, 100),
    (2000, 2000, 100),
    (3000, 3000, 100),
    (4000, 4000, 100),
    (5000, 5000, 100),
]

def main():
    print("=" * 60)
    print("Large Matrix Sweep Experiment")
    print("=" * 60)
    print(f"Configurations: {len(MATRIX_CONFIGS)}")
    for N1, N2, M in MATRIX_CONFIGS:
        print(f"  - {N1}x{N2}, M={M}")
    print(f"Alpha range: 0.0 to 0.5, step 0.01 (51 points)")
    print(f"Steps: 10000")
    print("=" * 60)
    print()

    experiment = LargeMatrixSweepExperiment(
        matrix_configs=MATRIX_CONFIGS,
        alpha_start=0.0,
        alpha_stop=0.5,
        alpha_step=0.01,
        max_steps=10000,
        samples=1,
    )

    results = experiment.run()

    print("\n" + "=" * 60)
    print("Summary of Results")
    print("=" * 60)

    for config_str, data in results.items():
        print(f"\n{config_str}:")
        print(f"  Time: {data['time']:.1f}s")

        # Find phase transition point (Q_Y crosses 0.5)
        alphas = sorted([float(a) for a in data['results'].keys()])
        for i, alpha in enumerate(alphas):
            qy = data['results'][alpha]['Q_Y_mean']
            if qy > 0.5 and i > 0:
                prev_qy = data['results'][alphas[i-1]]['Q_Y_mean']
                alpha_c = alphas[i-1] + (0.5 - prev_qy) / (qy - prev_qy) * (alpha - alphas[i-1])
                print(f"  Phase transition (Q_Y=0.5): alpha_c ~ {alpha_c:.3f}")
                break
        else:
            print(f"  Phase transition not detected in range")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print(f"Results saved to: {experiment.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
