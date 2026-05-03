# Development Structure, Algorithm, and Metric Map

Scope: `dev` branch only. Active package: `src/matrix_factorization`.

## Repository Tree

```text
Matrix_Factorization/
├─ README.md
├─ AGENTS.md
├─ pyproject.toml
├─ src/
│  └─ matrix_factorization/
│     ├─ cli.py
│     ├─ __main__.py
│     ├─ config.yaml
│     ├─ core/
│     │  ├─ contracts.py
│     │  ├─ planning.py
│     │  ├─ scan_planning.py
│     │  ├─ experiment/
│     │  │  ├─ config.py
│     │  │  ├─ continuation.py
│     │  │  ├─ data_factory.py
│     │  │  ├─ result.py
│     │  │  └─ runner.py
│     │  └─ parallel/
│     │     ├─ algorithm_adapters.py
│     │     ├─ batch_checkpoint.py
│     │     ├─ execution_modes.py
│     │     ├─ memory_estimator.py
│     │     ├─ memory_estimator_general.py
│     │     ├─ memory_guard.py
│     │     ├─ parallel_coordinator.py
│     │     └─ resource_execution.py
│     ├─ modules/
│     │  ├─ registry.py
│     │  ├─ algorithms/
│     │  ├─ graphs/
│     │  ├─ metrics/
│     │  ├─ outputs/
│     │  ├─ teachers/
│     │  └─ interventions/
│     ├─ presets/
│     ├─ export/
│     └─ ui/
├─ configs/
├─ trials/
│  ├─ active/
│  └─ archive/
├─ tests/
│  ├─ unit/
│  ├─ verification/
│  └─ debug/
├─ scripts/
│  ├─ analysis/
│  ├─ debug/
│  ├─ experiments/
│  ├─ maintenance/
│  └─ verification/
├─ docs/
│  ├─ METRICS_GUIDE.md
│  ├─ metrics_semantics.md
│  ├─ current_project_map.md
│  ├─ metric_naming_decisions.md
│  ├─ theory/
│  ├─ reports/
│  ├─ figures/
│  ├─ reference_code/
│  ├─ plans/
│  └─ archive/
├─ experiments/
├─ _legacy/
├─ smf/
├─ runs/
├─ results/
└─ artifacts/
```

## Runtime Tree

```text
Runtime
├─ Input
│  ├─ config.yaml
│  ├─ configs/*.yaml
│  └─ trials/active/*/config.yaml
├─ CLI
│  ├─ mf validate
│  ├─ mf explain-config
│  ├─ mf trial
│  └─ mf <config>
├─ Planning
│  ├─ ParameterSpec
│  ├─ AlgorithmSpec
│  ├─ MetricSpec
│  ├─ OutputSpec
│  ├─ ExperimentPlan
│  └─ ScanPlan
├─ Data
│  ├─ teacher factors
│  ├─ dense mask
│  ├─ spreading graph
│  ├─ F_super / Y_super
│  ├─ tensor graph
│  └─ F_tensor
├─ Algorithms
│  ├─ AGD family
│  └─ AMP / BiGAMP family
├─ Metrics
│  ├─ output metrics
│  ├─ latent metrics
│  ├─ gauge diagnostics
│  ├─ tensor metrics
│  └─ replica diagnostics
└─ Output
   ├─ config.json
   ├─ metadata.json
   ├─ metrics.json
   ├─ events.jsonl
   ├─ artifacts/results.pt
   ├─ plots/
   └─ manifest.json
```

## Algorithm Tree

```text
Algorithms
├─ AGD family
│  ├─ agd
│  │  ├─ file: modules/algorithms/agd.py
│  │  ├─ model: dense matrix
│  │  ├─ data: dense teacher + dense mask
│  │  └─ status: active
│  ├─ agd_tensor
│  │  ├─ file: modules/algorithms/agd_tensor.py
│  │  ├─ model: tensor
│  │  └─ status: experimental_unintegrated
│  └─ agd_spreading
│     ├─ file: modules/algorithms/legacy/agd_spreading.py
│     ├─ model: matrix spreading
│     └─ status: legacy_broken
├─ AMP / BiGAMP family
│  ├─ bigamp
│  │  ├─ file: modules/algorithms/bigamp/standard.py
│  │  ├─ model: dense matrix
│  │  ├─ data: dense teacher + dense mask
│  │  ├─ parameters: damping, noise_var
│  │  └─ status: active
│  ├─ bigamp_spreading
│  │  ├─ file: modules/algorithms/bigamp/spreading.py
│  │  ├─ model: matrix random spreading
│  │  ├─ data: matrix teacher + spreading_graph + F_super
│  │  ├─ parameters: damping, noise_var, spreading config
│  │  ├─ status: active
│  │  ├─ F distribution
│  │  │  ├─ ising
│  │  │  └─ gaussian
│  │  ├─ graph route
│  │  │  ├─ bipartite / flat
│  │  │  └─ general graph
│  │  ├─ Onsager route
│  │  │  ├─ no Onsager
│  │  │  ├─ fixed Onsager
│  │  │  └─ adaptive Onsager
│  │  └─ stop route
│  │     ├─ fixed max_steps
│  │     └─ teacher-assisted metric convergence stop
│  ├─ bigamp_tensor
│  │  ├─ file: modules/algorithms/bigamp/tensor_spreading.py
│  │  ├─ model: tensor spreading
│  │  ├─ route: serial/reference
│  │  └─ status: active
│  └─ bigamp_tensor_parallel
│     ├─ file: modules/algorithms/bigamp/tensor_spreading_parallel.py
│     ├─ model: tensor spreading
│     ├─ route: parallel
│     └─ status: active
└─ Helper
   └─ combined
      ├─ file: modules/algorithms/combined.py
      └─ status: non_trainable_helper
```

## Algorithm Support Files

```text
modules/algorithms/
├─ base.py
├─ agd.py
├─ agd_tensor.py
├─ combined.py
├─ legacy/
│  └─ agd_spreading.py
└─ bigamp/
   ├─ standard.py
   ├─ spreading.py
   ├─ plateau.py
   ├─ conventions.py
   ├─ core.py
   ├─ f_gen.py
   ├─ step.py
   ├─ tensor_contract.py
   ├─ tensor_data.py
   ├─ tensor_hypergraph.py
   ├─ tensor_memory.py
   ├─ tensor_spreading.py
   ├─ tensor_spreading_parallel.py
   ├─ tensor_step.py
   ├─ tensor_step_batch.py
   ├─ tensor_step_super.py
   └─ tensor_supergraph.py
```

## Metric Tree

```text
Metrics
├─ Output / Y metrics
│  ├─ Q_Y
│  │  ├─ matrix.full.Q_Y
│  │  ├─ matrix.observed.Q_Y
│  │  ├─ matrix.unobserved.Q_Y
│  │  ├─ spreading.full.Q_Y
│  │  ├─ spreading.observed.Q_Y
│  │  ├─ spreading.unobserved.Q_Y
│  │  ├─ tensor.full.Q_Y
│  │  ├─ tensor.serial_observed.Q_Y
│  │  ├─ tensor.observed.Q_Y
│  │  └─ tensor.unobserved.Q_Y
│  ├─ Q_Y_COS
│  │  ├─ matrix.full.Q_Y_COS
│  │  ├─ matrix.observed.Q_Y_COS
│  │  ├─ matrix.unobserved.Q_Y_COS
│  │  ├─ spreading.full.Q_Y_COS
│  │  ├─ spreading.observed.Q_Y_COS
│  │  └─ spreading.unobserved.Q_Y_COS
│  └─ FIT_Y / NMSE_Y
│     ├─ full
│     ├─ observed
│     └─ unobserved
├─ Matrix latent metrics
│  ├─ Q_W
│  │  ├─ Q_W_mean
│  │  ├─ Q_W_std
│  │  ├─ R_W_mean
│  │  └─ R_W_std
│  ├─ Q_X
│  │  ├─ Q_X_mean
│  │  ├─ Q_X_std
│  │  ├─ R_X_mean
│  │  └─ R_X_std
│  ├─ sign gauge
│  │  ├─ Q_W_SIGN_GAUGE_mean
│  │  ├─ Q_X_SIGN_GAUGE_mean
│  │  ├─ Q_W_SIGN_ALIGNED_mean
│  │  └─ Q_X_SIGN_ALIGNED_mean
│  ├─ scale gauge
│  │  ├─ Q_W_SCALE_GAUGE_mean
│  │  ├─ Q_X_SCALE_GAUGE_mean
│  │  ├─ Q_WX_SCALE_GAUGE_mean
│  │  ├─ median_abs_log_k_mean
│  │  └─ median_abs_log_g_mean
│  └─ cos-root
│     ├─ Q_W_COS_ROOT_mean
│     └─ Q_X_COS_ROOT_mean
├─ Tensor latent metrics
│  └─ Q_N
│     ├─ Q_N_mean
│     ├─ Q_N_mode0_mean
│     ├─ Q_N_mode1_mean
│     ├─ Q_N_mode2_mean
│     └─ Q_N_mode3_mean
└─ Replica diagnostics
   ├─ Q_W_replica_mean
   ├─ Q_X_replica_mean
   ├─ Q_W_prime_replica_mean
   └─ Q_X_prime_replica_mean
```

## Metric Formula Index

```text
schema_version = 6
metric_definition_profile = projection_qy_supergraph_full_cos_v3
```

```text
Q_Y = abs(sum_e Y_student[e] * Y_teacher[e]) / sum_e Y_teacher[e]^2
Q_Y_COS = sum_e Y_student[e] * Y_teacher[e]
          / (sqrt(sum_e Y_student[e]^2) * sqrt(sum_e Y_teacher[e]^2))
NMSE_Y = sum_e (Y_student[e] - Y_teacher[e])^2 / sum_e Y_teacher[e]^2
FIT_Y = 1 - NMSE_Y
```

```text
Q_W = sum_i,mu W_student[i,mu] * W_teacher[i,mu] / (N1 * M)
Q_X = sum_mu,j X_student[mu,j] * X_teacher[mu,j] / (M * N2)
R_W = sum_i,mu W_student[i,mu]^2 / (N1 * M)
R_X = sum_mu,j X_student[mu,j]^2 / (M * N2)
```

```text
Q_W_SIGN_GAUGE =
  sum_mu abs(sum_i W_student[i,mu] * W_teacher[i,mu]) / (N1 * M)

Q_X_SIGN_GAUGE =
  sum_mu abs(sum_j X_student[mu,j] * X_teacher[mu,j]) / (M * N2)
```

```text
Q_N_mode_d =
  sum_entries N_student_mode_d * N_teacher_mode_d / numel(N_teacher_mode_d)

Q_N = mean_d Q_N_mode_d
```

## Metric Scope Tree

```text
Scope
├─ matrix
│  ├─ full
│  ├─ observed
│  └─ unobserved
├─ spreading
│  ├─ full: 0:C_max on the shared F-aware supergraph
│  ├─ observed: 0:C_k on the shared F-aware supergraph
│  └─ unobserved: C_k:C_max on the shared F-aware supergraph
└─ tensor
   ├─ full
   ├─ serial_observed
   ├─ observed
   └─ unobserved
```

## Contract Tree

```text
Hard interface
├─ YAML parameter
│  └─ ParameterSpec
├─ algorithm
│  ├─ AlgorithmSpec
│  └─ @register_algorithm
├─ metric
│  ├─ MetricSpec
│  └─ @register_metric
├─ output
│  ├─ OutputSpec
│  └─ @register_output
├─ teacher
│  ├─ TeacherSpec
│  └─ @register_teacher
├─ graph
│  ├─ GraphSpec
│  └─ @register_graph
├─ probe
│  └─ ProbeSpec
├─ analyzer
│  └─ AnalyzerSpec
└─ intervention
   └─ InterventionSpec
```

## Generated Output Tree

```text
run_dir/
├─ config.json
├─ metadata.json
├─ metrics.json
├─ events.jsonl
├─ artifacts/
│  └─ results.pt
├─ results.pt -> artifacts/results.pt
├─ plots/
└─ manifest.json
```
