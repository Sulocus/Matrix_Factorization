"""端到端测试：tensor_order=3 通过 runner 运行"""
import torch

def test_tensor_runner_e2e():
    """测试完整的 tensor runner 流程"""
    from matrix_factorization.core.experiment import ExperimentConfig, ExperimentRunner
    from matrix_factorization.core.experiment.config import (
        MatrixParams, TrainingParams, SeedConfig, ScanConfig, 
        AlgorithmParams, SpreadingConfig
    )
    
    config = ExperimentConfig(
        matrix=MatrixParams(N1=20, N2=20, M=5),
        training=TrainingParams(samples_per_alpha=2, max_steps=30),
        algorithm_key='bigamp_tensor',
        scan=ScanConfig(dimension='alpha', values=[1.0, 2.0]),
        seeds=SeedConfig(base_seed=42),
        algorithm_params=AlgorithmParams(damping=0.5, noise_var=1e-6),
        spreading=SpreadingConfig(tensor_order=3),
        experiment_name='test_tensor_e2e',
    )
    
    print(f"Config: {config}")
    print(f"spreading.tensor_order: {config.spreading.tensor_order}")
    
    runner = ExperimentRunner(device=torch.device('cpu'), verbose=True)
    result = runner.run(config)
    
    print(f"\nResults: {len(result.results)} alpha values")
    for alpha, single_result in result.results.items():
        print(f"  alpha={alpha}: {single_result.metrics}")
        assert 'Q_Y_mean' in single_result.metrics or len(single_result.metrics) > 0, f"Missing metrics for alpha={alpha}"
    
    print("\n✅ 端到端测试通过")

if __name__ == '__main__':
    test_tensor_runner_e2e()
