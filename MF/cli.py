#!/usr/bin/env python3
"""
SMF Experiment CLI - 简洁版

使用方法:
    # 使用 YAML 配置
    python smf/cli.py config.yaml
    
    # 快速运行 (默认参数)
    python smf/cli.py --quick
    
    # 指定参数
    python smf/cli.py --N 200 --M 50 --steps 2000 --algorithm bigamp
    
    # 从 checkpoint 恢复中断的实验
    smf resume [output_dir]
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Configure CUDA Memory Pool for better performance (reduce fragmentation/stuttering)
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, str(Path(__file__).parent.parent))

from MF.core.experiment import (
    ExperimentConfig, ExperimentRunner,
    MatrixParams, TrainingParams, SeedConfig, ScanConfig, 
    AlgorithmParams, SpreadingConfig, TeacherConfig
)
from MF.core.progress import ProgressBridge


def load_yaml_config(yaml_path: Path):
    """从 YAML 文件加载配置，支持所有扫描模式。返回 (config, output_options, raw_yaml)"""
    import yaml
    with open(yaml_path, 'r') as f:
        raw_yaml = f.read()  # 保存原始 YAML 字符串用于 checkpoint
    cfg = yaml.safe_load(raw_yaml)
    
    # 数字选项映射
    ALGORITHM_MAP = {1: 'bigamp', 2: 'bigamp_spreading', 3: 'agd', 'bigamp': 'bigamp', 'bigamp_spreading': 'bigamp_spreading', 'agd': 'agd'}
    TEACHER_MAP = {1: 'orthogonal', 2: 'standard', 'orthogonal': 'orthogonal', 'standard': 'standard'}
    SCAN_MODE_MAP = {1: 'alpha', 2: 'steps', 3: 'nested', 4: 'hysteresis', 'alpha': 'alpha', 'steps': 'steps', 'nested': 'nested', 'hysteresis': 'hysteresis'}
    F_DIST_MAP = {1: 'rademacher', 2: 'gaussian', 'rademacher': 'rademacher', 'gaussian': 'gaussian'}
    
    algorithm_key = ALGORITHM_MAP.get(cfg.get('algorithm', 1), 'bigamp')
    teacher_key = TEACHER_MAP.get(cfg.get('teacher', 1), 'orthogonal')
    scan_mode = SCAN_MODE_MAP.get(cfg.get('scan_mode', 1), 'alpha')
    
    # 矩阵
    m = cfg.get('matrix', {})
    matrix = MatrixParams(N1=m.get('N1', 200), N2=m.get('N2', 200), M=m.get('M', 50))
    
    # 训练
    t = cfg.get('training', {})
    training = TrainingParams(
        samples_per_alpha=t.get('samples_per_alpha', t.get('samples', 50)),
        max_steps=t.get('max_steps', 2000)
    )
    
    # 算法参数
    a = cfg.get('algorithm_params', {})
    algo_params = AlgorithmParams(**a)
    
    # Spreading
    spreading = None
    # allow_intra_connection is now properly inside spreading section
    if 'spreading' in algorithm_key:
        s = cfg.get('spreading', {})
        f_dist = F_DIST_MAP.get(s.get('f_distribution', 1), 'rademacher')
        onsager = s.get('onsager_correction', False)
        # allow_intra_connection: 优先顶层，回退到 spreading 节点
        allow_intra = cfg.get('allow_intra_connection', s.get('allow_intra_connection', False))
        chunk_size = s.get('chunk_size', 131072)  # Edges per chunk for General mode
        spreading = SpreadingConfig(f_distribution=f_dist, onsager_correction=onsager, allow_intra_connection=allow_intra, chunk_size=chunk_size)
    
    INIT_DIST_MAP = {1: 'gaussian', 2: 'rademacher', 'gaussian': 'gaussian', 'rademacher': 'rademacher'}
    
    # Teacher Config (Fixed Bug: was ignored previously)
    teacher_cfg_dict = cfg.get('teacher_config', {})
    init_dist = INIT_DIST_MAP.get(teacher_cfg_dict.get('init_distribution', 1), 'gaussian')
    teacher_config = TeacherConfig(init_distribution=init_dist)

    # 输出选项 (完整解析)
    output_cfg = cfg.get('output', {})
    output_options = {
        'rsb_ordering': output_cfg.get('rsb_ordering', False),
        'save_tensors': output_cfg.get('save_tensors', True),
        'uniform_colormap': output_cfg.get('uniform_colormap', False),
        'storage_mode': output_cfg.get('storage_mode', 'full'),
        'enable_heatmap': output_cfg.get('enable_heatmap', True),  # Heatmap + GIF 开关
        'plots': output_cfg.get('plots', []),  # 新格式: [{curves: [A.y, B.w]}, ...]
    }
    
    # 根据扫描模式创建不同的 ScanConfig
    if scan_mode == 'alpha':
        alpha_cfg = cfg.get('alpha_scan', {})
        alpha_values = list(np.arange(
            alpha_cfg.get('start', 0.0),
            alpha_cfg.get('stop', 4.0) + 0.01,
            alpha_cfg.get('step', 0.05)
        ))
        scan = ScanConfig(dimension='alpha', values=alpha_values)
        graph_mode = 'general' if allow_intra else 'bipartite'
        name = f"{algorithm_key}_{teacher_key}_{matrix.N1}x{matrix.N2}_M{matrix.M}_{graph_mode}"
        
        return ExperimentConfig(
            matrix=matrix,
            training=training,
            algorithm_key=algorithm_key,
            scan=scan,
            seeds=SeedConfig(base_seed=t.get('seed', 42)),
            algorithm_params=algo_params,
            spreading=spreading,
            experiment_name=name,
            teacher_key=teacher_key,
            teacher=teacher_config, # Pass the parsed config!
        ), output_options, raw_yaml
    
    elif scan_mode == 'steps':
        steps_cfg = cfg.get('steps_scan', {})
        multiplier = steps_cfg.get('multiplier', 1)
        fixed_alpha = steps_cfg.get('alpha', 1.5)
        
        # 支持两种生成方式：log_space (对数均匀) 或手动 step_values
        log_space_cfg = steps_cfg.get('log_space')
        if log_space_cfg:
            # 对数均匀分布: np.geomspace(start, stop, num)
            start = log_space_cfg.get('start', 100)
            stop = log_space_cfg.get('stop', 10000)
            num = log_space_cfg.get('num', 10)
            step_values = [int(v) for v in np.geomspace(start, stop, num)]
            # 去重并排序 (因为取整可能导致重复)
            step_values = sorted(set(step_values))
            print(f"  Log-space steps: {step_values}")
        else:
            step_values = steps_cfg.get('step_values', [100, 500, 1000, 2000, 5000])
        
        step_values = [int(v * multiplier) for v in step_values]  # Apply multiplier
        scan = ScanConfig(dimension='steps', values=step_values)
        graph_mode = 'general' if allow_intra else 'bipartite'
        name = f"{algorithm_key}_{matrix.N1}x{matrix.N2}_M{matrix.M}_steps_alpha{fixed_alpha}_{graph_mode}"
        
        # 把 fixed_alpha 存到 algorithm_params 里
        algo_params.default_alpha = fixed_alpha
        
        return ExperimentConfig(
            matrix=matrix,
            training=training,
            algorithm_key=algorithm_key,
            scan=scan,
            seeds=SeedConfig(base_seed=t.get('seed', 42)),
            algorithm_params=algo_params,
            spreading=spreading,
            experiment_name=name,
            teacher_key=teacher_key,
            teacher=teacher_config, # Pass parsed config
        ), output_options, raw_yaml
    
    elif scan_mode == 'nested':
        nested_cfg = cfg.get('nested_scan', {})
        sizes = nested_cfg.get('sizes', [[200, 50], [400, 100]])
        alpha_cfg = nested_cfg.get('alpha', {})
        alpha_values = list(np.arange(
            alpha_cfg.get('start', 0.0),
            alpha_cfg.get('stop', 4.0) + 0.01,
            alpha_cfg.get('step', 0.1)
        ))
        
        # 返回嵌套扫描配置（特殊格式）
        return {
            'mode': 'nested',
            'sizes': [(s[0], s[0], s[1]) for s in sizes],  # (N1, N2, M)
            'alpha_values': alpha_values,
            'training': training,
            'algorithm_key': algorithm_key,
            'teacher_key': teacher_key,
            'teacher': teacher_config, # Pass parsed config
            'seeds': SeedConfig(base_seed=t.get('seed', 42)),
            'algorithm_params': algo_params,
            'spreading': spreading,
        }, output_options, raw_yaml

    elif scan_mode == 'hysteresis':
        hyst_cfg = cfg.get('hysteresis_scan', {})
        # 默认对比: [Cold Start (0.0), Warm Start (0.95)]
        init_overlaps = hyst_cfg.get('init_overlaps', [0.0, 0.95])
        
        # Alpha 扫描范围
        alpha_cfg = hyst_cfg.get('alpha', cfg.get('alpha_scan', {}))
        alpha_values = list(np.arange(
            alpha_cfg.get('start', 0.0),
            alpha_cfg.get('stop', 4.0) + 0.01,
            alpha_cfg.get('step', 0.05)
        ))
        
        return {
            'mode': 'hysteresis',
            'init_overlaps': init_overlaps,
            'alpha_values': alpha_values,
            'base_config': ExperimentConfig(
                matrix=matrix,
                training=training,
                algorithm_key=algorithm_key,
                scan=ScanConfig(dimension='alpha', values=alpha_values), # Placeholder
                seeds=SeedConfig(base_seed=t.get('seed', 42)),
                algorithm_params=algo_params,
                spreading=spreading,
                experiment_name='hysteresis_placeholder',
                teacher_key=teacher_key,
                teacher=teacher_config,
            ),
            'output_plots': output_cfg.get('plots', []) # Pass output plots config
        }, output_options, raw_yaml
    
    else:
        raise ValueError(f"Unknown scan_mode: {scan_mode}")


def build_config(args) -> ExperimentConfig:
    """从命令行参数创建配置"""
    alpha_values = list(np.arange(args.alpha_start, args.alpha_stop + 0.01, args.alpha_step))
    
    spreading = None
    if args.algorithm == 'bigamp_spreading':
        spreading = SpreadingConfig(f_distribution=args.f_dist)
    
    name = f"{args.algorithm}_{args.teacher}_{args.N}x{args.N}_M{args.M}_S{args.S}_steps{args.steps}"
    
    return ExperimentConfig(
        matrix=MatrixParams(N1=args.N, N2=args.N, M=args.M),
        training=TrainingParams(samples_per_alpha=args.S, max_steps=args.steps),
        algorithm_key=args.algorithm,
        scan=ScanConfig(dimension='alpha', values=alpha_values),
        seeds=SeedConfig(base_seed=args.seed),
        algorithm_params=AlgorithmParams(damping=args.damping, noise_var=1e-10, use_compile=not args.no_compile),
        spreading=spreading,
        experiment_name=name,
        teacher_key=args.teacher,
    )


def handle_resume(resume_dir: str = None):
    """
    处理 smf resume 命令，从 checkpoint 恢复中断的实验。
    使用 checkpoint 中保存的配置继续运行。
    """
    import yaml
    from MF.core.parallel.batch_checkpoint import CheckpointManager, dict_to_config
    
    print()
    print("=" * 60)
    print("🔄 SMF Resume - 从 Checkpoint 恢复")
    print("=" * 60)
    
    # 加载 checkpoint (固定路径 smf/.checkpoint.pt)
    ckpt_mgr = CheckpointManager()
    ckpt_data = ckpt_mgr.load()
    
    if not ckpt_data:
        print("❌ 未找到可恢复的 checkpoint")
        print()
        print("提示:")
        print("  - checkpoint 位置: smf/.checkpoint.pt")
        print("  - 确保之前的实验因中断而保存了 checkpoint")
        return
    
    # 显示 checkpoint 信息
    completed_alphas = ckpt_data.completed_alphas
    config_dict = ckpt_data.config_dict
    results = ckpt_data.results
    raw_yaml = ckpt_data.raw_yaml  # 读取原始 YAML！
    
    # 从 raw_yaml 恢复完整配置（包括绘图选项）
    if raw_yaml:
        # 有原始 YAML，可以完整恢复
        cfg = yaml.safe_load(raw_yaml)
        output_cfg = cfg.get('output', {})
        output_options = {
            'rsb_ordering': output_cfg.get('rsb_ordering', False),
            'save_tensors': output_cfg.get('save_tensors', True),
            'uniform_colormap': output_cfg.get('uniform_colormap', False),
            'storage_mode': output_cfg.get('storage_mode', 'full'),
            'enable_heatmap': output_cfg.get('enable_heatmap', True),
            'plots': output_cfg.get('plots', []),
        }
        print("   配置来源: 原始 YAML (完整)")
    else:
        # 兼容旧版 checkpoint，使用 output_options 字段
        output_options = ckpt_data.output_options
        print("   配置来源: 旧版 checkpoint")
    
    print(f"📂 找到 checkpoint: smf/.checkpoint.pt")
    print(f"   已完成 alpha: {len(completed_alphas)}")
    if completed_alphas:
        print(f"   Alpha 范围: {min(completed_alphas):.2f} - {max(completed_alphas):.2f}")
    print(f"   时间戳: {ckpt_data.timestamp}")
    print(f"   输出选项: rsb_ordering={output_options.get('rsb_ordering', False)}")
    print()
    
    # 从 checkpoint 恢复配置
    try:
        config = dict_to_config(config_dict)
    except Exception as e:
        print(f"❌ 无法恢复配置: {e}")
        return
    
    print(f"🚀 恢复实验: {config.experiment_name}")
    print(f"   矩阵: {config.matrix.N1}x{config.matrix.N2}, M={config.matrix.M}")
    print(f"   步数: {config.training.max_steps}")
    print(f"   剩余 alpha: {len(config.scan.values) - len(completed_alphas)}")
    print("=" * 60)
    print()
    
    # 运行实验（runner 会跳过已完成的 alpha）
    from MF.core.experiment.runner import ExperimentRunner
    from MF.core.progress import ProgressBridge
    
    runner = ExperimentRunner(verbose=True)
    bridge = ProgressBridge()
    
    # 传递已完成的 results 和 raw_yaml 给 runner
    result = runner.run(
        config, 
        observer=bridge.on_event, 
        resume_results=results,
        output_options=output_options,
        raw_yaml=raw_yaml,
    )
    
    # 保存结果 - 使用恢复的输出选项
    output_dir = Path(f"smf/Replica_results/alpha_scan/{config.experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(
        output_dir,
        save_tensors=output_options.get('save_tensors', True),
        rsb_ordering=output_options.get('rsb_ordering', False),
        uniform_colormap=output_options.get('uniform_colormap', False),
        output_options=output_options,
    )
    
    # 删除 checkpoint（成功完成）
    ckpt_mgr.delete()
    
    print()
    print("=" * 60)
    print(f"✅ Resume 完成! 保存到: {output_dir}")
    print("=" * 60)



def main():
    # 特殊处理: smf resume 子命令 (在argparse之前)
    if len(sys.argv) >= 2 and sys.argv[1] == 'resume':
        handle_resume(sys.argv[2] if len(sys.argv) > 2 else None)
        return
    
    # 特殊处理: smf conf 子命令 (在argparse之前)
    if len(sys.argv) >= 2 and sys.argv[1] == 'conf':
        import subprocess
        import shutil
        config_path = Path(__file__).parent / 'config.yaml'
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            sys.exit(1)
        editor = os.environ.get('EDITOR')
        if not editor:
            for cmd in ['code', 'vim', 'nano', 'vi']:
                if shutil.which(cmd):
                    editor = cmd
                    break
        if editor:
            print(f"📝 编辑配置: {config_path}")
            subprocess.run([editor, str(config_path)])
        else:
            print(f"配置文件路径: {config_path}")
        return
    
    parser = argparse.ArgumentParser(
        description='SMF Experiment CLI\n\n用法:\n  smf          运行实验 (使用 smf/config.yaml)\n  smf conf     编辑配置文件',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 配置文件
    parser.add_argument('config', nargs='?', help='YAML 配置文件 (默认: smf/config.yaml)')
    parser.add_argument('--quick', action='store_true', help='快速运行 (默认参数)')
    parser.add_argument('-c', '--conf', action='store_true', help='编辑配置文件')
    
    # 参数
    parser.add_argument('--N', type=int, default=200, help='矩阵尺寸')
    parser.add_argument('--M', type=int, default=50, help='隐藏维度')
    parser.add_argument('--S', type=int, default=50, help='样本数')
    parser.add_argument('--steps', type=int, default=2000, help='训练步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--algorithm', choices=['bigamp', 'bigamp_spreading', 'agd'], default='bigamp')
    parser.add_argument('--teacher', choices=['standard', 'orthogonal'], default='orthogonal')
    parser.add_argument('--alpha-start', type=float, default=0.0)
    parser.add_argument('--alpha-stop', type=float, default=4.0)
    parser.add_argument('--alpha-step', type=float, default=0.05)
    parser.add_argument('--damping', type=float, default=0.5)
    parser.add_argument('--f-dist', choices=['rademacher', 'gaussian'], default='rademacher')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--output-dir', default='smf/Replica_results')
    
    args = parser.parse_args()
    
    # 无参数时默认使用 config.yaml 运行
    if args.config is None and not args.quick and not args.conf:
        default_config = Path(__file__).parent / 'config.yaml'
        if default_config.exists():
            args.config = str(default_config)
            print(f"📄 使用默认配置: {default_config}")
        else:
            parser.print_help()
            return
    
    # 加载配置
    output_options = {'rsb_ordering': False, 'save_tensors': True}  # defaults
    raw_yaml = ""  # 原始 YAML 字符串 (用于 checkpoint)
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Error: Config file not found: {args.config}")
            print(f"   Tip: Use full path like 'smf/config_template.yaml'")
            sys.exit(1)
        print(f"📄 Loading: {args.config}")
        config, output_options, raw_yaml = load_yaml_config(config_path)
    else:
        config = build_config(args)
    
    runner = ExperimentRunner(verbose=False)  # 由ProgressBridge处理输出
    bridge = ProgressBridge(use_rich=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')  # YYYYMMDD_HHMM for sort-friendly naming
    
    # 检查是否为嵌套扫描模式
    if isinstance(config, dict) and config.get('mode') == 'nested':
        print()
        print("=" * 60)
        print("🚀 Nested Scaling Sweep")
        print("=" * 60)
        print(f"  Algorithm:  {config['algorithm_key']}")
        print(f"  Teacher:    {config['teacher_key']}")
        print(f"  Sizes:      {config['sizes']}")
        print(f"  Alpha:      {min(config['alpha_values']):.2f} → {max(config['alpha_values']):.2f} ({len(config['alpha_values'])} points)")
        print()
        
        # 创建基础配置
        base_config = ExperimentConfig(
            matrix=MatrixParams(N1=200, N2=200, M=50),  # 会被 sizes 覆盖
            training=config['training'],
            algorithm_key=config['algorithm_key'],
            scan=ScanConfig(dimension='alpha', values=config['alpha_values']),
            seeds=config['seeds'],
            algorithm_params=config['algorithm_params'],
            spreading=config['spreading'],
            experiment_name='nested_scaling_sweep',
            teacher_key=config['teacher_key'],
        )
        
        # 运行嵌套扫描
        # For nested scan, we use a single runner instance
        runner = ExperimentRunner(verbose=False)
        output_path = Path(args.output_dir) / "nested_scan" / f"{timestamp}_nested_sweep"
        output_path.mkdir(parents=True, exist_ok=True)
        results = runner.run_scaling_sweep(
            base_config=base_config,
            matrix_sizes=config['sizes'],
            output_dir=output_path,
            observer=bridge.on_event,
        )
        
        print()
        print("=" * 60)
        print(f"✅ Done! {len(results)} sizes completed")
        print(f"   Saved to: {output_path}")
        print("=" * 60)

    elif isinstance(config, dict) and config.get('mode') == 'hysteresis':
        print()
        print("=" * 60)
        print("🚀 Hysteresis Analysis (Cold vs Warm)")
        print("=" * 60)
        
        base_config = config['base_config']
        init_overlaps = config['init_overlaps']
        output_plots = config.get('output_plots', [])
        
        METRIC_MAP = {
            'A.y': 'Q_Y_mean', 'A.w': 'Q_W_mean', 'A.x': 'Q_X_mean',
            'B.w': 'Q_W_prime_mean', 'B.x': 'Q_X_prime_mean',
            'D.y': 'physical_overlap_Y_mean', 
            'D.w': 'physical_overlap_W_mean', 'D.x': 'physical_overlap_X_mean',
            'E.e': 'MSE',
            'Physical.y': 'physical_overlap_Y_mean',
            'Physical.w': 'physical_overlap_W_mean',
        }
        
        # 结果容器
        results_list = []
        labels = []
        
        # 创建输出目录
        output_path = Path(args.output_dir) / "hysteresis_scan" / f"{timestamp}_{base_config.experiment_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        for overlap in init_overlaps:
            # 修改配置
            import copy
            run_config = copy.deepcopy(base_config)
            
            if overlap <= 1e-6:
                mode_name = "Cold Start"
                init_mode_val = "random"
            else:
                mode_name = f"Warm Start (m={overlap})"
                init_mode_val = "teacher"
            
            # --------------------------------------------------------------------------------
            # CRITICAL FIX: Ensure parameter propagation
            # ExperimentRunner might use run_config.algorithm OR run_config.algorithm_params
            # We must update BOTH to ensure init_mode is correctly passed.
            # --------------------------------------------------------------------------------
            
            # 1. Update algorithm_params (if exists)
            if hasattr(run_config, 'algorithm_params'):
                run_config.algorithm_params.init_mode = init_mode_val
                if init_mode_val == 'teacher':
                    run_config.algorithm_params.init_overlap = overlap

            # 2. Update algorithm (if exists - this is often the one used by internal logic)
            if hasattr(run_config, 'algorithm'):
                try:
                    # Config objects might be nested or frozen, try direct attribute set
                    run_config.algorithm.init_mode = init_mode_val
                    if init_mode_val == 'teacher':
                        run_config.algorithm.init_overlap = overlap
                except Exception as e:
                    print(f"Warning: Could not update run_config.algorithm: {e}")

            # Debug Log to confirm what we set
            def get_val(obj, attr, default):
                return getattr(obj, attr, default)

            curr_mode = get_val(getattr(run_config, 'algorithm_params', None), 'init_mode', 'N/A')
            curr_overlap = get_val(getattr(run_config, 'algorithm_params', None), 'init_overlap', 'N/A')
            algo_mode = get_val(getattr(run_config, 'algorithm', None), 'init_mode', 'N/A')
            
            print(f"▶ Running: {mode_name}")
            
            # --------------------------------------------------------------------------------
            # CRITICAL FIX: Instantiate fresh Runner for each iteration
            # ExperimentRunner caches algorithm instances based on algorithm_key.
            # Since both Cold and Warm start use the same key ('bigamp_spreading'),
            # the Runner would reuse the Cold Start instance (random init) for Warm Start
            # and ignore the config change.
            # We MUST create a new runner to bypass this cache.
            # --------------------------------------------------------------------------------
            loop_runner = ExperimentRunner(verbose=False)

            # 运行实验
            result = loop_runner.run(
                config=run_config,
                observer=bridge.on_event,
                output_options=output_options,
            )
            
            # Clean up
            del loop_runner
            import gc
            gc.collect()

            # 存储结果 (Metrics are in result.metrics)
            # ExperimentResult.results 是按 scan_value (alpha) 索引的 metrics 字典
            # 我们需要把它展平，方便绘图
            # plot_comparison 需要 list of dict: alpha -> metrics
            results_dict = {}
            for res_item in result.results.values():
                results_dict[res_item.scan_value] = res_item.metrics
            
            results_list.append(results_dict)
            labels.append(mode_name)
            
            # 单独保存这一轮的结果
            sub_output_path = output_path / mode_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            result.save(
                sub_output_path,
                save_tensors=output_options.get('save_tensors', True),
                rsb_ordering=output_options.get('rsb_ordering', False),
                uniform_colormap=output_options.get('uniform_colormap', False),
                output_options=output_options,
            )
        
        print(f"\n📊 Generating Hysteresis Plots...")
        
        # 绘制对比图
        from MF.modules.outputs.plotting import plot_multi_metric_comparison
        
        # 1. 解析要绘制的图表组
        plot_groups = []
        if not output_plots:
            # 默认 fallback
            plot_groups.append({'metrics': ['Q_Y_mean'], 'filename': 'hysteresis_comparison_Q_Y_mean.png'})
        else:
            for idx, plot_cfg in enumerate(output_plots):
                curves = plot_cfg.get('curves', [])
                group_metrics = []
                for curve in curves:
                    # 查找映射
                    metric_key = METRIC_MAP.get(curve, curve)
                    # 处理后缀 (e.g. :R) - 目前简单去除，未来可支持 Replica 曲线
                    if ':' in metric_key:
                        metric_key = metric_key.split(':')[0]
                    group_metrics.append(metric_key)
                
                if group_metrics:
                    # 生成文件名
                    if len(group_metrics) == 1:
                        fname = f"hysteresis_comparison_{group_metrics[0]}.png"
                    else:
                        # 对于组合图，使用 custom_plot_N 命名，或尝试拼接
                        fname = f"custom_plot_{idx+1}.png"
                    plot_groups.append({'metrics': group_metrics, 'filename': fname})

        # 2. 循环绘制每个组
        for grp in plot_groups:
            metrics = grp['metrics']
            filename = grp['filename']
            plot_path = output_path / filename
            
            # 检查 metrics 是否存在 (至少一个)
            first_results = results_list[0]
            first_data = first_results[next(iter(first_results))]
            valid_metrics = [m for m in metrics if m in first_data or m.replace('_mean', '') in first_data]
            
            if not valid_metrics:
                 print(f"⚠️  No valid metrics found for plot {filename}, skipping.")
                 print(f"    Requested: {metrics}")
                 print(f"    Available: {list(first_data.keys())}")
                 continue

            print(f"   Writing {filename} ({', '.join(valid_metrics)})...")
            
            try:
                plot_multi_metric_comparison(
                    results_list=results_list,
                    labels=labels,
                    output_path=plot_path,
                    metrics=valid_metrics,
                    legend_loc='best',
                    dpi=300
                )
            except Exception as e:
                print(f"   ❌ Failed to plot {filename}: {e}")
        
        # 链接到最新结果 (Symbolic Link or Copy)
        # ...
        
        print()
        print("=" * 60)
        print("✅ Hysteresis Analysis Complete!")
        print(f"   Results saved to: {output_path}")
        print("=" * 60)

    else:

        # 单一扫描模式 (alpha 或 steps)
        print()
        print("=" * 60)
        print(f"🚀 {config.experiment_name}")
        print("=" * 60)
        print(f"  Algorithm:  {config.algorithm_key}")
        print(f"  Teacher:    {config.teacher_key}")
        print(f"  Matrix:     {config.N1}x{config.N2}, M={config.M}")
        print(f"  Scan:       {config.scan.dimension} ({len(config.scan.values)} points)")
        print(f"  Samples:    {config.S}")
        print(f"  Steps:      {config.training.max_steps}")
        print()
        
        # 运行 - 传递 output_options 和 raw_yaml 以便 checkpoint 系统保存
        result = runner.run(config, observer=bridge.on_event, output_options=output_options, raw_yaml=raw_yaml)
        
        # 保存 - 按扫描类型分类
        scan_type_dir = "steps_scan" if config.scan.dimension == 'steps' else "alpha_scan"
        output_path = Path(args.output_dir) / scan_type_dir / f"{timestamp}_{config.experiment_name}"
        result.save(
            output_path,
            save_tensors=output_options.get('save_tensors', True),
            rsb_ordering=output_options.get('rsb_ordering', False),
            uniform_colormap=output_options.get('uniform_colormap', False),
            output_options=output_options,
        )
        
        # Update _latest symlink for easy access to newest result
        scan_dir = Path(args.output_dir) / scan_type_dir
        latest_link = scan_dir / "_latest"
        try:
            if latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(output_path.name)
        except OSError:
            pass  # Symlink creation may fail on some filesystems
        
        print()
        print("=" * 60)
        print(f"✅ Done! Saved to: {output_path}")
        print(f"   _latest -> {output_path.name}")
        print("=" * 60)


if __name__ == '__main__':
    main()

