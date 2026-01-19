
import logging
import torch
import traceback
from matrix_factorization.core.experiment import ExperimentConfig, ExperimentRunner
from matrix_factorization.core.experiment.config import (
    MatrixParams, TrainingParams, SeedConfig, ScanConfig, 
    AlgorithmParams, SpreadingConfig
)

# 配置日志
logging.basicConfig(level=logging.INFO)

def reproduce():
    try:
        # 构造用户配置
        config = ExperimentConfig(
            experiment_name='reproduce_crash',
            matrix=MatrixParams(N1=200, N2=200, M=50),
            training=TrainingParams(samples_per_alpha=10, max_steps=100), 
            algorithm_key='bigamp_tensor',
            scan=ScanConfig(
                dimension='alpha',
                values=[0.5, 1.0], # 直接指定 values
            ),
            seeds=SeedConfig(base_seed=42),
            algorithm_params=AlgorithmParams(damping=0.5, noise_var=1e-5),
            spreading=SpreadingConfig(tensor_order=3),
        )
        
        print("Starting reproduction runner...")
        runner = ExperimentRunner(device=torch.device('cpu'), verbose=True)
        result = runner.run(config)
        print("Run complete successfully.")
        
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    reproduce()
