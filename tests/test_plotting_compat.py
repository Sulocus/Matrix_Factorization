
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.getcwd())

from matrix_factorization.core.experiment.config import ExperimentConfig, MatrixParams, TrainingParams, SeedConfig, ScanConfig, AlgorithmParams
from matrix_factorization.modules.outputs.plotting import ResultPlotter

# 模拟结果数据结构
# 假设我们有两个结果集: Cold 和 Warm
# 每个结果集是 map: alpha -> metrics_dict

alphas = [0.0, 1.0, 2.0, 3.0, 4.0]

results_cold = {
    a: {
        'Q_Y_mean': 0.1 if a < 3.5 else 0.99,
        'Q_Y_std': 0.01,
        'Q_W_mean': 0.0,
    } for a in alphas
}

results_warm = {
    a: {
        'Q_Y_mean': 0.15 if a < 2.5 else 0.99,
        'Q_Y_std': 0.02,
        'Q_W_mean': 0.8,
    } for a in alphas
}

# 创建一个假的 Config 对象用于初始化 Plotter
config = ExperimentConfig(
    matrix=MatrixParams(N1=200, N2=200, M=50),
    training=TrainingParams(samples_per_alpha=5, max_steps=100),
    algorithm_key='bigamp_spreading',
    scan=ScanConfig(dimension='alpha', values=alphas),
    seeds=SeedConfig(),
    algorithm_params=AlgorithmParams(),
    experiment_name='test_plot',
    teacher_key='standard'
)

# 初始化 Plotter
output_dir = Path("smf/test_output")
output_dir.mkdir(exist_ok=True, parents=True)
plotter = ResultPlotter(config, output_dir)

# 尝试调用 plot_comparison
try:
    print("Testing plot_comparison...")
    res_list = [results_cold, results_warm]
    labels = ["Cold Start", "Warm Start"]
    
    # 注意: plot_comparison 是模块级函数还是类方法？
    # 查看 plotting.py，plot_comparison 是一个独立函数，不在 ResultPlotter 类里
    from matrix_factorization.modules.outputs.plotting import plot_comparison
    
    output_path = output_dir / "comparison_test.png"
    plot_comparison(
        results_list=res_list,
        labels=labels,
        output_path=output_path,
        metric='Q_Y_mean',
        legend_loc='upper left'
    )
    print(f"Success! Plot saved to {output_path}")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
