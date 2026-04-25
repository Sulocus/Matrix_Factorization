"""
Hard interface contracts for experiments.

These dataclasses describe what parameters, algorithms, metrics, outputs, and
interventions are allowed to do. They are intentionally lightweight: Phase 1
uses them for validation and explanation without changing algorithm physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    value_type: str
    default: Any = None
    owner: str = "config"
    consumers: List[str] = field(default_factory=list)
    physical_sensitive: bool = False
    status: str = "active"  # active / legacy / parsed_only / deprecated
    description: str = ""


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    status: str
    required_config_paths: List[str] = field(default_factory=list)
    data_requirements: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    state_capabilities: List[str] = field(default_factory=list)
    supported_interventions: List[str] = field(default_factory=list)
    produced_metrics: List[str] = field(default_factory=list)
    produced_artifacts: List[str] = field(default_factory=list)
    compatible_outputs: List[str] = field(default_factory=list)
    result_contract: str = "legacy_matrix_result"
    notes: str = ""


@dataclass(frozen=True)
class TeacherSpec:
    key: str
    status: str
    required_config_paths: List[str] = field(default_factory=list)
    produced_artifacts: List[str] = field(default_factory=list)
    compatible_algorithms: List[str] = field(default_factory=list)
    scale_convention: str = ""
    notes: str = ""


@dataclass(frozen=True)
class GraphSpec:
    key: str
    status: str
    required_config_paths: List[str] = field(default_factory=list)
    produced_artifacts: List[str] = field(default_factory=list)
    compatible_algorithms: List[str] = field(default_factory=list)
    ensemble: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ResourceSpec:
    algorithm_key: str
    estimator_key: str = "none"
    device_support: List[str] = field(default_factory=list)
    dtype_modes: List[str] = field(default_factory=list)
    compile_support: str = "none"
    probe_support: str = "none"
    empty_cache_policy: str = "none"
    notes: str = ""


@dataclass(frozen=True)
class BatchingSpec:
    algorithm_key: str
    planner_layers: List[str] = field(default_factory=list)
    alpha_batching: str = "none"
    sample_batching: str = "none"
    chunking: str = "none"
    seed_partition_sensitive: bool = False
    sample_range_honored: bool = True
    metadata_only: bool = True
    notes: str = ""


@dataclass(frozen=True)
class MemoryModelSpec:
    algorithm_key: str
    estimator_entrypoint: str = "none"
    formula_basis: str = ""
    tensor_components: List[str] = field(default_factory=list)
    calibration_status: str = "uncalibrated"
    probe_required: bool = False
    sample_range_policy: str = ""
    drives_execution: bool = False
    notes: str = ""


@dataclass(frozen=True)
class SeedPolicySpec:
    algorithm_key: str
    policy_key: str
    seed_inputs: List[str] = field(default_factory=list)
    random_streams: List[str] = field(default_factory=list)
    partition_invariant: bool = False
    batch_partition_sensitive: bool = True
    automatic_rebatch_allowed: bool = False
    notes: str = ""


@dataclass
class AlgorithmStateView:
    """Capability-based state exposed to probes/interventions/analyzers.

    Runtime extensions should depend on these stable capability names instead
    of reaching into algorithm-private attributes.
    """

    student_factors: Optional[Dict[str, Any]] = None
    tensor_factors: Optional[Dict[str, Any]] = None
    factor_variances: Optional[Dict[str, Any]] = None
    onsager_residual: Optional[Any] = None
    graph: Optional[Any] = None
    teacher_factors: Optional[Dict[str, Any]] = None
    teacher: Optional[Dict[str, Any]] = None
    step_index: Optional[int] = None
    alpha: Optional[float] = None
    sample_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def available_capabilities(self) -> List[str]:
        keys = []
        for key in [
            "student_factors",
            "tensor_factors",
            "factor_variances",
            "onsager_residual",
            "graph",
            "teacher_factors",
            "teacher",
            "step_index",
            "alpha",
            "sample_index",
        ]:
            if getattr(self, key) is not None:
                keys.append(key)
        keys.extend(sorted(self.metadata))
        return keys


@dataclass
class AlgorithmResult:
    """Formal algorithm output container for the hard-interface migration."""

    metrics_by_alpha: Dict[float, Dict[str, Any]] = field(default_factory=dict)
    matrix_factors: Optional[Dict[str, Any]] = None
    tensor_factors: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_matrix_factors(
        cls,
        *,
        alpha: float,
        metrics: Dict[str, Any],
        W_students: Any,
        X_students: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AlgorithmResult":
        return cls(
            metrics_by_alpha={float(alpha): dict(metrics)},
            matrix_factors={"W_students": W_students, "X_students": X_students},
            metadata=dict(metadata or {}, result_kind="matrix_factors"),
        )

    @classmethod
    def from_metrics_only(
        cls,
        *,
        metrics_by_alpha: Dict[float, Dict[str, Any]],
        artifacts: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AlgorithmResult":
        return cls(
            metrics_by_alpha={float(alpha): dict(metrics) for alpha, metrics in metrics_by_alpha.items()},
            matrix_factors=None,
            tensor_factors=None,
            artifacts=dict(artifacts or {}),
            metadata=dict(metadata or {}, result_kind="metrics_only"),
        )

    def available_outputs(self) -> List[str]:
        outputs = []
        if self.metrics_by_alpha:
            outputs.append("metrics_by_alpha")
        if self.matrix_factors and any(value is not None for value in self.matrix_factors.values()):
            outputs.append("matrix_factors")
        if self.tensor_factors and any(value is not None for value in self.tensor_factors.values()):
            outputs.append("tensor_factors")
        outputs.extend(sorted(self.artifacts))
        outputs.extend(f"diagnostic:{key}" for key in sorted(self.diagnostics))
        return outputs


@dataclass(frozen=True)
class MetricSpec:
    key: str
    requires: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    space: str = ""
    scope: str = ""
    relation: str = ""
    normalization: str = ""
    physical_meaning: str = ""
    compatible_algorithms: List[str] = field(default_factory=list)
    canonical_key: str = ""
    equivalence_class: str = ""
    legacy_aliases: List[str] = field(default_factory=list)
    result_role: str = "formal"  # formal / diagnostic / plotting_helper
    order_parameter_status: str = "candidate"  # candidate / diagnostic / not_order_parameter / review
    review_status: str = "provisional"
    risk_level: str = "low"


@dataclass(frozen=True)
class MetricSemanticClass:
    canonical_key: str
    equivalence_class: str
    display_name: str
    legacy_aliases: List[str] = field(default_factory=list)
    space: str = ""
    scope: str = ""
    relation: str = ""
    normalization: str = ""
    result_role: str = "formal"
    order_parameter_status: str = "candidate"
    risk_level: str = "low"
    review_status: str = "provisional"
    description: str = ""


@dataclass(frozen=True)
class OutputSpec:
    key: str
    requires: List[str] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    description: str = ""


@dataclass(frozen=True)
class InterventionSpec:
    key: str
    trigger: str
    requires_state: List[str] = field(default_factory=list)
    modifies_state: List[str] = field(default_factory=list)
    compatible_algorithms: List[str] = field(default_factory=list)
    physical_sensitive: bool = True
    description: str = ""


@dataclass(frozen=True)
class ProbeSpec:
    key: str
    trigger: str
    requires_state: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    compatible_algorithms: List[str] = field(default_factory=list)
    runtime_status: str = "declared_only"
    description: str = ""


@dataclass(frozen=True)
class AnalyzerSpec:
    key: str
    requires: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    compatible_algorithms: List[str] = field(default_factory=list)
    description: str = ""


@dataclass(frozen=True)
class SourceInventorySpec:
    path: str
    role: str  # package / base / algorithm_entry / support_module / legacy
    registry_key: Optional[str] = None
    status: str = "active"
    description: str = ""


@dataclass(frozen=True)
class TrialSpec:
    key: str
    status: str
    runtime_class: str = "quick"
    purpose: str = ""
    owner_area: str = ""
    entrypoint: str = ""
    config_path: str = ""
    output_root: str = ""
    artifact_policy: str = "ignored_workspace"
    promotion_target: List[str] = field(default_factory=list)
    allowed_outputs: List[str] = field(default_factory=list)
    artifact_roots: List[str] = field(default_factory=list)
    expected_metrics: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class TensorParitySpec:
    key: str
    required: bool = True
    area: str = ""
    serial_status: str = "unknown"
    parallel_status: str = "unknown"
    risk: str = ""
    numeric_sensitive: bool = True
    review_required: bool = True
    notes: str = ""


def get_parameter_specs() -> Dict[str, ParameterSpec]:
    specs = [
        ParameterSpec("tensor_order", "int", 2, "model", ["cli.load_yaml_config"], True),
        ParameterSpec("algorithm", "int|str", 2, "algorithm", ["cli.load_yaml_config"], True),
        ParameterSpec("teacher", "int|str", 1, "teacher", ["cli.load_yaml_config"], True),
        ParameterSpec("teacher_config.init_distribution", "int|str", 1, "teacher", ["DataFactory.create"], True),
        ParameterSpec("matrix.N1", "int", 200, "model", ["MatrixParams"], True),
        ParameterSpec("matrix.N2", "int", 200, "model", ["MatrixParams"], True),
        ParameterSpec("matrix.M", "int", 50, "model", ["MatrixParams"], True),
        ParameterSpec("scan_mode", "int|str", 1, "scan", ["cli.load_yaml_config"], False),
        ParameterSpec("alpha_scan.start", "float", 0.0, "scan", ["ScanConfig"], True),
        ParameterSpec("alpha_scan.stop", "float", 4.0, "scan", ["ScanConfig"], True),
        ParameterSpec("alpha_scan.step", "float", 0.05, "scan", ["ScanConfig"], True),
        ParameterSpec("steps_scan.alpha", "float", 1.5, "scan", ["AlgorithmParams.default_alpha"], True),
        ParameterSpec("steps_scan.step_values", "list[int]", None, "scan", ["ScanConfig"], False),
        ParameterSpec("steps_scan.multiplier", "int", 1, "scan", ["cli.load_yaml_config"], False),
        ParameterSpec("steps_scan.log_space.start", "int", 100, "scan", ["cli.load_yaml_config"], False),
        ParameterSpec("steps_scan.log_space.stop", "int", 10000, "scan", ["cli.load_yaml_config"], False),
        ParameterSpec("steps_scan.log_space.num", "int", 10, "scan", ["cli.load_yaml_config"], False),
        ParameterSpec("scan.axis", "int", None, "scan", [], False, "legacy"),
        ParameterSpec("scan.values", "list", None, "scan", [], False, "legacy"),
        ParameterSpec("nested_scan.sizes", "list", None, "scan", ["cli.load_yaml_config"], True),
        ParameterSpec("nested_scan.alpha.start", "float", 0.0, "scan", ["cli.load_yaml_config"], True),
        ParameterSpec("nested_scan.alpha.stop", "float", 4.0, "scan", ["cli.load_yaml_config"], True),
        ParameterSpec("nested_scan.alpha.step", "float", 0.1, "scan", ["cli.load_yaml_config"], True),
        ParameterSpec("hysteresis_scan.init_overlaps", "list[float]", None, "intervention", ["cli.load_yaml_config"], True),
        ParameterSpec("hysteresis_scan.alpha.start", "float", 0.0, "scan", ["cli.load_yaml_config"], True),
        ParameterSpec("hysteresis_scan.alpha.stop", "float", 4.0, "scan", ["cli.load_yaml_config"], True),
        ParameterSpec("hysteresis_scan.alpha.step", "float", 0.05, "scan", ["cli.load_yaml_config"], True),
        ParameterSpec("training.samples_per_alpha", "int", 20, "training", ["TrainingParams"], True),
        ParameterSpec("training.samples", "int", None, "training", ["TrainingParams.samples_per_alpha"], True, "legacy"),
        ParameterSpec("training.max_steps", "int", 5000, "training", ["TrainingParams"], True),
        ParameterSpec("training.max_epochs", "int", 20000, "training", ["TrainingParams"], True),
        ParameterSpec("training.num_workers", "int", 1, "training", ["TrainingParams"], False),
        ParameterSpec("training.seed", "int", 42, "seed", ["SeedConfig.base_seed"], True),
        ParameterSpec("seeds.model", "int", 42, "seed", ["SeedConfig.base_seed"], True),
        ParameterSpec("seeds.data", "int", 42, "seed", ["SeedConfig.teacher_seed"], True),
        ParameterSpec("seeds.base_seed", "int", 42, "seed", ["SeedConfig.base_seed"], True),
        ParameterSpec("seeds.teacher_seed", "int", 12345, "seed", ["SeedConfig.teacher_seed"], True),
        ParameterSpec("seeds.spreading_seed", "int", 99999, "seed", ["SeedConfig.spreading_seed"], True),
        ParameterSpec("seeds.student_seed", "int", 0, "seed", ["SeedConfig.student_seed"], True),
        ParameterSpec("algorithm_params.damping", "float", 0.5, "algorithm", ["algorithms"], True),
        ParameterSpec("algorithm_params.noise_var", "float", 1e-10, "algorithm", ["algorithms"], True),
        ParameterSpec("algorithm_params.learning_rate", "float", 1e-2, "algorithm", ["AGDAlgorithm"], True),
        ParameterSpec("algorithm_params.use_compile", "bool", True, "algorithm", ["bigamp", "bigamp_spreading", "bigamp_tensor_parallel"], False),
        ParameterSpec("algorithm_params.compile_fallback_policy", "enum[allow,error]", "allow", "algorithm", ["bigamp", "bigamp_spreading", "bigamp_tensor_parallel"], False),
        ParameterSpec("algorithm_params.use_bf16", "bool", True, "algorithm", ["agd", "bigamp_spreading", "bigamp_tensor_parallel"], False),
        ParameterSpec("algorithm_params.dtype_fallback_policy", "enum[allow,error]", "allow", "algorithm", ["agd", "bigamp_spreading", "bigamp_tensor_parallel"], False),
        ParameterSpec("algorithm_params.use_tf32", "bool", True, "algorithm", ["agd", "bigamp", "bigamp_spreading", "bigamp_tensor_parallel"], False),
        ParameterSpec("algorithm_params.seed_partition_policy", "enum[legacy,partition_invariant]", "legacy", "algorithm", ["agd", "bigamp", "bigamp_spreading", "bigamp_tensor_parallel"], True),
        ParameterSpec("algorithm_params.use_early_stop", "bool", False, "algorithm", ["AGDAlgorithm"], False),
        ParameterSpec("algorithm_params.target_loss_threshold", "float", 1e-8, "algorithm", ["AGDAlgorithm"], True),
        ParameterSpec("algorithm_params.default_alpha", "float", 1.0, "algorithm", ["steps_scan"], True),
        ParameterSpec("algorithm_params.adaptive_damping", "bool", False, "algorithm", ["BiGAMPSpreading"], True),
        ParameterSpec("algorithm_params.step_min", "float", 0.05, "algorithm", ["BiGAMPSpreading"], True),
        ParameterSpec("algorithm_params.step_max", "float", 0.5, "algorithm", ["BiGAMPSpreading"], True),
        ParameterSpec("algorithm_params.step_incr", "float", 1.05, "algorithm", ["BiGAMPSpreading"], True),
        ParameterSpec("algorithm_params.step_decr", "float", 0.5, "algorithm", ["BiGAMPSpreading"], True),
        ParameterSpec("algorithm_params.step_window", "int", 5, "algorithm", ["BiGAMPSpreading"], True),
        ParameterSpec("algorithm_params.max_bad_steps", "int", 10, "algorithm", ["BiGAMPSpreading"], True),
        ParameterSpec("algorithm_params.adaptive_restart", "bool", False, "intervention", ["adaptive_restart"], True),
        ParameterSpec("algorithm_params.restart_patience", "int", 50, "intervention", ["adaptive_restart"], True),
        ParameterSpec("algorithm_params.restart_noise", "float", 0.1, "intervention", ["adaptive_restart"], True),
        ParameterSpec("algorithm_params.acceptance_tolerance", "float", 0.0, "intervention", ["adaptive_restart"], True),
        ParameterSpec("algorithm_params.init_mode", "str", "random", "intervention", ["warm_start", "cold_start"], True),
        ParameterSpec("algorithm_params.init_overlap", "float", 0.95, "intervention", ["warm_start"], True),
        ParameterSpec("algorithm_params.debug_verbose", "bool", False, "debug", ["algorithms"], False),
        ParameterSpec("spreading.f_distribution", "int|str", 1, "data", ["SpreadingConfig", "algorithms"], True),
        ParameterSpec("spreading.onsager_correction", "bool", False, "algorithm", ["SpreadingConfig", "algorithms"], True),
        ParameterSpec("spreading.allow_intra_connection", "bool", False, "data", ["SpreadingConfig"], True, "parsed_only"),
        ParameterSpec("spreading.seed", "int", 12345, "seed", ["SpreadingConfig"], True),
        ParameterSpec("spreading.chunk_size", "int", 131072, "algorithm", ["BiGAMPSpreading"], False),
        ParameterSpec("spreading.tensor_order", "int", 2, "model", ["SpreadingConfig"], True, "parsed_only"),
        ParameterSpec("output.name", "str", "", "output", [], False, "parsed_only"),
        ParameterSpec("output.storage_mode", "str", "full", "output", ["ExperimentResult.save"], False, "parsed_only"),
        ParameterSpec("output.save_tensors", "bool", True, "output", ["ExperimentResult.save"], False),
        ParameterSpec("output.enable_heatmap", "bool", True, "output", ["heatmap"], False),
        ParameterSpec("output.rsb_ordering", "bool", False, "output", ["heatmap"], False),
        ParameterSpec("output.heatmap_metric", "str", "Q_Y", "output", ["heatmap"], False),
        ParameterSpec("output.uniform_colormap", "bool", False, "output", ["heatmap"], False),
        ParameterSpec("output.plots", "list", [], "output", ["custom_curves"], False),
        ParameterSpec("probes", "list", [], "probe", ["ProbeSpec"], False),
        ParameterSpec("analyzers", "list", [], "analyzer", ["AnalyzerSpec"], False),
    ]
    return {spec.path: spec for spec in specs}


def get_algorithm_specs() -> Dict[str, AlgorithmSpec]:
    return {
        "agd": AlgorithmSpec(
            key="agd",
            status="active",
            required_config_paths=["matrix.N1", "matrix.N2", "matrix.M", "algorithm_params.learning_rate"],
            data_requirements=["matrix_teacher", "dense_mask"],
            capabilities=["batch_alpha", "gpu_optional", "matrix_factors"],
            state_capabilities=["student_factors", "teacher_factors", "step_index", "loss"],
            produced_metrics=[
                "matrix.full.Q_Y",
                "matrix.observed.Q_Y",
                "matrix.unobserved.Q_Y",
                "matrix.factor.Q_W",
                "matrix.factor.Q_X",
                "matrix.physical_overlap",
                "matrix.error.MSE",
                "replica.factor",
            ],
            produced_artifacts=["matrix_factors"],
            compatible_outputs=["scalar_curves", "fallback_W_heatmap", "latest_display"],
            result_contract="legacy_matrix_result",
        ),
        "bigamp": AlgorithmSpec(
            key="bigamp",
            status="active",
            required_config_paths=["matrix.N1", "matrix.N2", "matrix.M", "algorithm_params.damping", "algorithm_params.noise_var"],
            data_requirements=["matrix_teacher", "dense_mask"],
            capabilities=["batch_alpha", "gpu_optional", "matrix_factors"],
            state_capabilities=["student_factors", "teacher_factors", "factor_variances", "step_index"],
            produced_metrics=[
                "matrix.full.Q_Y",
                "matrix.observed.Q_Y",
                "matrix.unobserved.Q_Y",
                "matrix.factor.Q_W",
                "matrix.factor.Q_X",
                "matrix.physical_overlap",
                "matrix.error.MSE",
                "replica.factor",
            ],
            produced_artifacts=["matrix_factors"],
            compatible_outputs=["scalar_curves", "fallback_W_heatmap", "latest_display"],
            result_contract="legacy_matrix_result",
        ),
        "bigamp_spreading": AlgorithmSpec(
            key="bigamp_spreading",
            status="active",
            required_config_paths=["spreading.f_distribution", "algorithm_params.damping", "algorithm_params.noise_var"],
            data_requirements=["matrix_teacher", "spreading_graph", "F_super"],
            capabilities=["batch_alpha", "gpu_optional", "matrix_factors", "spreading_metrics"],
            state_capabilities=["student_factors", "teacher_factors", "factor_variances", "onsager_residual", "spreading_graph", "step_index"],
            produced_metrics=[
                "matrix.full.Q_Y",
                "spreading.observed.Q_Y",
                "spreading.unobserved.Q_Y",
                "matrix.factor.Q_W",
                "matrix.factor.Q_X",
                "matrix.physical_overlap",
                "matrix.error.MSE",
                "replica.factor",
            ],
            produced_artifacts=["matrix_factors"],
            compatible_outputs=["scalar_curves", "fallback_W_heatmap", "latest_display"],
            result_contract="legacy_spreading_result",
        ),
        "bigamp_tensor": AlgorithmSpec(
            key="bigamp_tensor",
            status="active",
            required_config_paths=["spreading.tensor_order", "spreading.f_distribution"],
            data_requirements=["tensor_teacher_factors", "tensor_hypergraph", "F_tensor"],
            capabilities=["tensor_metrics", "serial_tensor_reference"],
            state_capabilities=["tensor_factors", "teacher_factors", "step_index"],
            produced_metrics=["tensor.serial_observed.Q_Y"],
            produced_artifacts=[],
            compatible_outputs=["scalar_curves", "latest_display"],
            result_contract="legacy_tensor_metrics_only",
            notes="Serial/reference path; CLI tensor_order>=3 currently routes to bigamp_tensor_parallel.",
        ),
        "bigamp_tensor_parallel": AlgorithmSpec(
            key="bigamp_tensor_parallel",
            status="active",
            required_config_paths=["spreading.tensor_order", "spreading.f_distribution", "algorithm_params.damping", "algorithm_params.noise_var"],
            data_requirements=["tensor_teacher_factors", "tensor_supergraph", "F_tensor"],
            capabilities=["batch_alpha", "gpu_optional", "tensor_metrics", "overlap_matrix"],
            state_capabilities=["tensor_factors", "teacher_factors", "tensor_factor_variances", "tensor_supergraph", "onsager_residual", "step_index"],
            produced_metrics=["tensor.full.Q_Y", "tensor.observed.Q_Y", "tensor.physical_overlap_Y"],
            produced_artifacts=["overlap_matrix"],
            compatible_outputs=["scalar_curves", "tensor_heatmap", "tensor_gif", "latest_display"],
            result_contract="legacy_tensor_metrics_only",
        ),
        "agd_tensor": AlgorithmSpec(
            key="agd_tensor",
            status="experimental_unintegrated",
            data_requirements=["internal_tensor_teacher"],
            capabilities=["tensor_agd_experimental"],
            state_capabilities=["tensor_factors", "teacher_factors", "step_index", "loss"],
            produced_metrics=["tensor.observed.Q_Y"],
            compatible_outputs=["scalar_curves"],
            result_contract="experimental_private_result",
        ),
        "agd_spreading": AlgorithmSpec(
            key="agd_spreading",
            status="legacy_broken",
            data_requirements=["matrix_teacher", "spreading_graph", "F_super"],
            capabilities=["legacy_spreading_agd"],
            state_capabilities=["student_factors", "step_index", "loss"],
            produced_metrics=["spreading.observed.Q_Y", "matrix.factor.Q_W"],
            compatible_outputs=["scalar_curves"],
            result_contract="legacy_private_result",
            notes="Legacy reference file under algorithms/legacy; not imported by the active package.",
        ),
        "combined": AlgorithmSpec(
            key="combined",
            status="non_trainable_helper",
            result_contract="not_a_training_algorithm",
        ),
    }


def get_teacher_specs() -> Dict[str, TeacherSpec]:
    specs = [
        TeacherSpec(
            "standard",
            "active",
            ["teacher_config.init_distribution", "matrix.N1", "matrix.N2", "matrix.M"],
            ["W_teacher", "X_teacher", "Y_teacher"],
            ["agd", "bigamp", "bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel"],
            "DataFactory scale: W/X drawn at 1/sqrt(M), Y=(1/sqrt(M))*W@X.",
        ),
        TeacherSpec(
            "orthogonal",
            "active",
            ["matrix.N1", "matrix.N2", "matrix.M"],
            ["W_teacher", "X_teacher", "Y_teacher"],
            ["agd", "bigamp", "bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel"],
            "DataFactory orthogonal teacher: SVD factors scaled by sqrt(M), Y=(1/sqrt(M))*W@X.",
        ),
        TeacherSpec("orthogonal_unit", "legacy_reference", notes="Registered legacy teacher with O(1) scaling."),
        TeacherSpec("scaled_variance", "legacy_reference", notes="Registered exploratory variance-scaling teacher."),
        TeacherSpec("random_spreading", "legacy_reference", notes="Older random spreading teacher/data helper."),
        TeacherSpec("combined", "legacy_reference", notes="Legacy teacher composition wrapper."),
    ]
    return {spec.key: spec for spec in specs}


def get_graph_specs() -> Dict[str, GraphSpec]:
    specs = [
        GraphSpec("random", "legacy_reference", ["matrix.N1", "matrix.N2", "matrix.M"], ["mask"], ["agd", "bigamp"], "Dense random observation mask."),
        GraphSpec("uniform", "legacy_reference", ["matrix.N1", "matrix.N2", "matrix.M"], ["mask"], ["agd", "bigamp"], "Bi-regular graph reference implementation."),
        GraphSpec("low_loop", "legacy_reference", ["matrix.N1", "matrix.N2", "matrix.M"], ["mask"], ["agd", "bigamp"], "MCMC low-short-cycle graph reference implementation."),
        GraphSpec("combined", "legacy_reference", notes="Legacy graph selection wrapper."),
        GraphSpec("supergraph", "active_path", ["spreading.seed", "spreading.f_distribution"], ["spreading_graph", "F_super"], ["bigamp_spreading"], "Spreading supergraph used by DataFactory."),
        GraphSpec("supergraph_general", "active_path", ["spreading.seed", "spreading.f_distribution"], ["spreading_graph", "F_super"], ["bigamp_spreading"], "General spreading supergraph used by tensor_order=1 route."),
    ]
    return {spec.key: spec for spec in specs}


def get_resource_specs() -> Dict[str, ResourceSpec]:
    specs = [
        ResourceSpec("agd", "runner.MemoryEstimator", ["cpu", "cuda"], ["float32"], "none", "none", "none", "Matrix AGD path; no algorithm-local memory probe."),
        ResourceSpec("bigamp", "runner.MemoryEstimator", ["cpu", "cuda"], ["float32"], "torch.compile optional", "none", "none", "Dense matrix BiG-AMP uses batched alpha tensors."),
        ResourceSpec("bigamp_spreading", "runner.MemoryEstimator + spreading chunk_size", ["cpu", "cuda"], ["float32", "bf16 storage"], "torch.compile optional", "none", "between_batches", "Spreading path has chunked edge streaming and per-batch graph creation."),
        ResourceSpec("bigamp_tensor", "none", ["cpu", "cuda"], ["float32"], "none", "none", "none", "Serial tensor/reference path; alpha/sample loops are serial."),
        ResourceSpec("bigamp_tensor_parallel", "runner.MemoryEstimator + tensor_memory.probe_tensor_super_memory", ["cpu", "cuda"], ["float32", "bf16 storage", "tf32 matmul"], "torch.compile default optional", "A=1 tensor supergraph probe", "between_batches", "TensorSuperGraph path uses algorithm-internal probe-based alpha batching."),
        ResourceSpec("agd_tensor", "none", ["cpu", "cuda"], ["float32"], "none", "none", "none", "Experimental tensor AGD path."),
        ResourceSpec("agd_spreading", "none", ["cpu", "cuda"], ["float32"], "none", "none", "none", "Legacy broken spreading AGD path."),
        ResourceSpec("combined", "none", [], [], "none", "none", "none", "Non-trainable helper."),
    ]
    return {spec.algorithm_key: spec for spec in specs}


def get_batching_specs() -> Dict[str, BatchingSpec]:
    specs = [
        BatchingSpec("agd", ["runner.ParallelCoordinator"], "runner alpha batches", "samples inside algorithm tensor", "none", False, False, notes="Runner plans batches; sample_range is not a formal algorithm input."),
        BatchingSpec("bigamp", ["runner.ParallelCoordinator"], "runner alpha batches", "samples parallel in W/X tensors", "none", False, False),
        BatchingSpec("bigamp_spreading", ["runner.ParallelCoordinator", "algorithm per-batch supergraph"], "runner alpha batches", "disjoint-union sample parallel", "spreading.chunk_size edge streaming", True, False, notes="Batch seed offsets and per-batch graph creation must be treated as random-path sensitive."),
        BatchingSpec("bigamp_tensor", ["algorithm loop"], "serial alpha loop", "serial sample loop", "none", True, True, notes="sample_seed depends on sample index and alpha value."),
        BatchingSpec("bigamp_tensor_parallel", ["runner.ParallelCoordinator", "algorithm internal probe batches"], "probe-based internal alpha batches", "TensorSuperGraph sample parallel", "none", True, False, notes="Algorithm sorts alphas internally and uses seed + batch_idx."),
        BatchingSpec("agd_tensor", ["algorithm private"], "experimental", "experimental", "none", True, True),
        BatchingSpec("agd_spreading", ["legacy private"], "legacy", "legacy", "none", True, True),
        BatchingSpec("combined", [], "none", "none", "none", False, True),
    ]
    return {spec.algorithm_key: spec for spec in specs}


def get_memory_model_specs() -> Dict[str, MemoryModelSpec]:
    specs = [
        MemoryModelSpec(
            "agd",
            "MemoryEstimator.register('agd')",
            "matrix factor tensors plus optimizer/loss buffers",
            ["student_factors", "gradients", "masks", "predictions"],
            "formula_only",
            False,
            "runner plans sample_range but active AGD path does not consume it",
            False,
        ),
        MemoryModelSpec(
            "bigamp",
            "MemoryEstimator.register('bigamp')",
            "dense matrix BiGAMP W/X variances, masks, predictions, residuals",
            ["student_factors", "factor_variances", "masks", "predictions", "residuals"],
            "formula_only",
            False,
            "runner plans sample_range but active dense BiGAMP path does not consume it",
            False,
        ),
        MemoryModelSpec(
            "bigamp_spreading",
            "MemoryEstimator.register('bigamp_spreading')",
            "spreading graph edges C, F distribution, chunk_size edge streaming",
            ["student_factors", "factor_variances", "supergraph", "F_super", "Y_super", "chunk_temporaries"],
            "formula_only_with_chunking_metadata",
            False,
            "runner plans sample_range but spreading algorithm constructs per-batch graph internally",
            False,
        ),
        MemoryModelSpec(
            "bigamp_tensor",
            "MemoryEstimator.register('bigamp_tensor')",
            "serial tensor alpha/sample loop with DoF-scaled hyperedges",
            ["tensor_factors", "tensor_observations", "serial_temporaries"],
            "formula_only_reference",
            False,
            "serial sample loop; no runner sample_range split",
            False,
        ),
        MemoryModelSpec(
            "bigamp_tensor_parallel",
            "MemoryEstimator.register('bigamp_tensor_parallel') + probe_tensor_super_memory(A=1)",
            "TensorSuperGraph DoF-scaled edges, batched factors, F/Y, gather/scatter temporaries",
            ["tensor_factors", "tensor_factor_variances", "tensor_supergraph", "F_tensor", "Y_tensor", "gather_temporaries"],
            "formula_plus_algorithm_probe_metadata",
            True,
            "runner sample_range metadata is not consumed; algorithm performs internal alpha batching",
            False,
            "Probe and formula are observable metadata only until seed partition invariant is solved.",
        ),
        MemoryModelSpec("agd_tensor", "none", "experimental private path", [], "untracked_experimental", False, "experimental", False),
        MemoryModelSpec("agd_spreading", "none", "legacy broken path", [], "legacy_untracked", False, "legacy", False),
        MemoryModelSpec("combined", "none", "non-trainable helper", [], "not_applicable", False, "none", False),
    ]
    return {spec.algorithm_key: spec for spec in specs}


def get_seed_policy_specs() -> Dict[str, SeedPolicySpec]:
    specs = [
        SeedPolicySpec(
            "agd",
            "legacy_vectorized_batch_manual_seed",
            ["seeds.base_seed", "runner batch alpha order"],
            ["student_initialization"],
            partition_invariant=False,
            batch_partition_sensitive=True,
            automatic_rebatch_allowed=False,
            notes="AGD vectorized batch calls torch.manual_seed(seed) once per runner batch; changing alpha batch partition can change/repeat student streams.",
        ),
        SeedPolicySpec(
            "bigamp",
            "legacy_vectorized_batch_manual_seed",
            ["seeds.base_seed", "runner batch alpha order"],
            ["student_initialization"],
            partition_invariant=False,
            batch_partition_sensitive=True,
            automatic_rebatch_allowed=False,
            notes="Dense BiGAMP vectorized batch calls torch.manual_seed(seed) once per runner batch; changing alpha batch partition can change/repeat student streams.",
        ),
        SeedPolicySpec(
            "bigamp_spreading",
            "legacy_spreading_batch_offset",
            ["seeds.base_seed", "spreading.seed", "algorithm internal batch_idx"],
            ["student_initialization", "spreading_graph", "F_super"],
            partition_invariant=False,
            batch_partition_sensitive=True,
            automatic_rebatch_allowed=False,
            notes="Spreading path offsets seed by internal batch index for some training batches; auto-rebatch must not be silent.",
        ),
        SeedPolicySpec(
            "bigamp_tensor",
            "legacy_tensor_serial_alpha_sample_seed",
            ["seeds.base_seed", "alpha", "sample_index"],
            ["student_initialization", "tensor_hypergraph", "F_tensor"],
            partition_invariant=True,
            batch_partition_sensitive=False,
            automatic_rebatch_allowed=True,
            notes="Serial tensor path uses seed + sample_index * 1000 + int(alpha * 100), so it is independent of runner batch partition.",
        ),
        SeedPolicySpec(
            "bigamp_tensor_parallel",
            "legacy_tensor_parallel_batch_idx_seed",
            ["seeds.base_seed", "sorted internal alpha batches", "algorithm internal batch_idx"],
            ["student_initialization", "tensor_supergraph", "F_tensor"],
            partition_invariant=False,
            batch_partition_sensitive=True,
            automatic_rebatch_allowed=False,
            notes="Tensor parallel sorts alpha values and calls _train_full_parallel(..., seed + batch_idx); changing internal batch partition changes random streams.",
        ),
        SeedPolicySpec("agd_tensor", "experimental_untracked", [], [], False, True, False, "Experimental tensor AGD path; seed semantics are not active-path contract."),
        SeedPolicySpec("agd_spreading", "legacy_untracked", [], [], False, True, False, "Broken legacy spreading AGD path; not an active runner contract."),
        SeedPolicySpec("combined", "not_applicable", [], [], True, False, False, "Non-trainable helper."),
    ]
    return {spec.algorithm_key: spec for spec in specs}


def get_metric_specs() -> Dict[str, MetricSpec]:
    specs = [
        MetricSpec("matrix.full.Q_Y", ["W_students", "X_students", "W_teacher", "X_teacher"], ["Q_Y_mean", "Q_Y_std"], "matrix", "full", "teacher-student", "cosine", "Dense matrix full-output cosine.", ["agd", "bigamp", "bigamp_spreading"]),
        MetricSpec("matrix.observed.Q_Y", ["W_students", "X_students", "W_teacher", "X_teacher", "mask"], ["Q_Y_observed_mean", "Q_Y_observed_std"], "matrix", "observed", "teacher-student", "masked cosine", "Dense observed-entry output cosine.", ["agd", "bigamp"]),
        MetricSpec("matrix.unobserved.Q_Y", ["W_students", "X_students", "W_teacher", "X_teacher", "mask"], ["Q_Y_unobserved_mean", "Q_Y_unobserved_std"], "matrix", "unobserved", "teacher-student", "masked cosine", "Dense unobserved-entry output cosine.", ["agd", "bigamp"]),
        MetricSpec("spreading.observed.Q_Y", ["W_students", "X_students", "spreading_graph", "F_super", "Y_super"], ["Q_Y_observed_mean", "Q_Y_observed_std"], "graph", "observed", "teacher-student", "F-aware cosine", "Observed spreading graph overlap using the same F.", ["bigamp_spreading", "agd_spreading"]),
        MetricSpec("spreading.unobserved.Q_Y", ["W_students", "X_students", "spreading_graph", "Y_teacher"], ["Q_Y_unobserved_mean", "Q_Y_unobserved_std"], "matrix", "unobserved", "teacher-student", "cosine", "Unobserved dense-output cosine induced by the spreading mask.", ["bigamp_spreading"]),
        MetricSpec("tensor.full.Q_Y", ["tensor_student_factors", "tensor_teacher_factors"], ["Q_Y_mean", "Q_Y_std"], "tensor", "full", "teacher-student", "cosine", "Full CP tensor cosine.", ["bigamp_tensor_parallel"]),
        MetricSpec("tensor.serial_observed.Q_Y", ["tensor_observations"], ["Q_Y_mean", "Q_Y_std"], "tensor", "observed", "teacher-student", "reconstruction quality", "Legacy serial tensor observed-edge diagnostic stored under Q_Y_mean/Q_Y_std.", ["bigamp_tensor"]),
        MetricSpec("tensor.observed.Q_Y", ["tensor_observations"], ["Q_Y_observed_mean", "Q_Y_observed_std"], "tensor", "observed", "teacher-student", "reconstruction quality", "Observed-edge tensor diagnostic.", ["bigamp_tensor_parallel", "agd_tensor"]),
        MetricSpec("matrix.factor.Q_W", ["W_students", "W_teacher"], ["Q_W_mean", "Q_W_std", "Q_W_prime_mean", "Q_W_prime_std"], "factor", "full", "teacher-student", "Gram cosine", "Factor Gram overlap.", ["agd", "bigamp", "bigamp_spreading", "agd_spreading"]),
        MetricSpec("matrix.factor.Q_X", ["X_students", "X_teacher"], ["Q_X_mean", "Q_X_std", "Q_X_prime_mean", "Q_X_prime_std"], "factor", "full", "teacher-student", "Gram cosine", "Right-factor Gram overlap.", ["agd", "bigamp", "bigamp_spreading"]),
        MetricSpec("matrix.physical_overlap", ["W_students", "X_students", "W_teacher", "X_teacher"], ["physical_overlap_W_mean", "physical_overlap_W_std", "physical_overlap_X_mean", "physical_overlap_X_std", "physical_overlap_Y_mean", "physical_overlap_Y_std"], "matrix", "full", "teacher-student", "projection", "Projection-style physical overlap diagnostics.", ["agd", "bigamp", "bigamp_spreading"]),
        MetricSpec("matrix.error.MSE", ["W_students", "X_students", "Y_teacher"], ["MSE", "MSE_std"], "matrix", "full", "teacher-student", "mean squared error", "Dense reconstruction mean squared error.", ["agd", "bigamp", "bigamp_spreading"]),
        MetricSpec("replica.factor", ["student_replicas"], ["Q_W_replica_mean", "Q_X_replica_mean", "Q_W_prime_replica_mean", "Q_X_prime_replica_mean"], "factor", "replica", "student-student", "Gram cosine", "Replica overlap diagnostic.", ["agd", "bigamp", "bigamp_spreading"]),
        MetricSpec("tensor.physical_overlap_Y", ["tensor_student_factors", "tensor_teacher_factors"], ["physical_overlap_Y_mean"], "tensor", "full", "teacher-student", "projection", "Tensor-space physical projection.", ["bigamp_tensor_parallel"]),
    ]
    specs = _attach_metric_semantic_metadata(specs)
    return {spec.key: spec for spec in specs}


def get_metric_semantic_classes() -> Dict[str, MetricSemanticClass]:
    """Canonical semantic classes for legacy flat metric keys.

    These classes describe what a metric means. They are deliberately separate
    from flat result names such as ``Q_Y_mean`` because old keys are reused
    across matrix, spreading, tensor, and replica paths.
    """
    classes = [
        MetricSemanticClass(
            "matrix.full.teacher_student.output_cosine",
            "output_similarity.full",
            "matrix full output cosine",
            ["Q_Y_mean", "Q_Y_std"],
            "matrix",
            "full",
            "teacher-student",
            "cosine",
            "formal",
            "candidate",
            "medium",
            description="Dense matrix output-space cosine between student reconstruction and teacher output.",
        ),
        MetricSemanticClass(
            "matrix.observed.teacher_student.output_cosine",
            "output_similarity.observed",
            "matrix observed output cosine",
            ["Q_Y_observed_mean", "Q_Y_observed_std"],
            "matrix",
            "observed",
            "teacher-student",
            "masked cosine",
            "diagnostic",
            "diagnostic",
            "medium",
            description="Dense matrix cosine restricted to observed mask entries.",
        ),
        MetricSemanticClass(
            "matrix.unobserved.teacher_student.output_cosine",
            "output_similarity.unobserved",
            "matrix unobserved output cosine",
            ["Q_Y_unobserved_mean", "Q_Y_unobserved_std"],
            "matrix",
            "unobserved",
            "teacher-student",
            "masked cosine",
            "formal",
            "candidate",
            "medium",
            description="Dense matrix cosine restricted to unobserved entries.",
        ),
        MetricSemanticClass(
            "spreading.observed.teacher_student.F_aware_output_cosine",
            "output_similarity.observed",
            "spreading observed F-aware output cosine",
            ["Q_Y_observed_mean", "Q_Y_observed_std"],
            "graph",
            "observed",
            "teacher-student",
            "F-aware cosine",
            "formal",
            "candidate",
            "high",
            description="Observed graph metric using the quenched spreading coefficients F.",
        ),
        MetricSemanticClass(
            "spreading.unobserved.teacher_student.dense_output_cosine",
            "output_similarity.unobserved",
            "spreading induced unobserved dense output cosine",
            ["Q_Y_unobserved_mean", "Q_Y_unobserved_std"],
            "matrix",
            "unobserved",
            "teacher-student",
            "cosine",
            "diagnostic",
            "diagnostic",
            "high",
            description="Dense-output diagnostic induced by the spreading mask, not an F-observed graph metric.",
        ),
        MetricSemanticClass(
            "tensor.full.teacher_student.cp_tensor_cosine",
            "output_similarity.full",
            "tensor full CP tensor cosine",
            ["Q_Y_mean", "Q_Y_std"],
            "tensor",
            "full",
            "teacher-student",
            "cosine",
            "formal",
            "candidate",
            "high",
            description="Full tensor cosine computed from CP teacher/student factors.",
        ),
        MetricSemanticClass(
            "tensor.observed.teacher_student.serial_reconstruction_quality",
            "output_similarity.observed",
            "serial tensor observed reconstruction quality",
            ["Q_Y_mean", "Q_Y_std"],
            "tensor",
            "observed",
            "teacher-student",
            "reconstruction quality",
            "diagnostic",
            "diagnostic",
            "high",
            description="Legacy serial tensor observed-edge reconstruction quality stored under Q_Y flat keys.",
        ),
        MetricSemanticClass(
            "tensor.observed.teacher_student.reconstruction_quality",
            "output_similarity.observed",
            "tensor observed reconstruction quality",
            ["Q_Y_observed_mean", "Q_Y_observed_std"],
            "tensor",
            "observed",
            "teacher-student",
            "reconstruction quality",
            "diagnostic",
            "diagnostic",
            "high",
            description="Tensor observed-edge reconstruction diagnostic.",
        ),
        MetricSemanticClass(
            "factor.W.teacher_student.gram_cosine",
            "factor_overlap.teacher_student",
            "W teacher-student Gram cosine",
            ["Q_W_mean", "Q_W_std"],
            "factor",
            "full",
            "teacher-student",
            "Gram cosine",
            "formal",
            "candidate",
            "medium",
            description="Left-factor Gram cosine overlap diagnostic.",
        ),
        MetricSemanticClass(
            "factor.W.teacher_student.baseline_corrected_gram_cosine",
            "factor_overlap.teacher_student",
            "W teacher-student baseline-corrected Gram cosine",
            ["Q_W_prime_mean", "Q_W_prime_std"],
            "factor",
            "full",
            "teacher-student",
            "baseline-corrected Gram cosine",
            "formal",
            "candidate",
            "medium",
            description="Left-factor baseline-corrected Gram overlap diagnostic.",
        ),
        MetricSemanticClass(
            "factor.X.teacher_student.gram_cosine",
            "factor_overlap.teacher_student",
            "X teacher-student Gram cosine",
            ["Q_X_mean", "Q_X_std"],
            "factor",
            "full",
            "teacher-student",
            "Gram cosine",
            "formal",
            "candidate",
            "medium",
            description="Right-factor Gram cosine overlap diagnostic.",
        ),
        MetricSemanticClass(
            "factor.X.teacher_student.baseline_corrected_gram_cosine",
            "factor_overlap.teacher_student",
            "X teacher-student baseline-corrected Gram cosine",
            ["Q_X_prime_mean", "Q_X_prime_std"],
            "factor",
            "full",
            "teacher-student",
            "baseline-corrected Gram cosine",
            "formal",
            "candidate",
            "medium",
            description="Right-factor baseline-corrected Gram overlap diagnostic.",
        ),
        MetricSemanticClass(
            "factor.W.teacher_student.coordinate_projection_abs",
            "physical_projection",
            "W coordinate projection absolute overlap",
            ["physical_overlap_W_mean", "physical_overlap_W_std"],
            "factor",
            "full",
            "teacher-student",
            "absolute coordinate projection",
            "diagnostic",
            "diagnostic",
            "high",
            description="Coordinate projection diagnostic for W; gauge/sign/permutation sensitive.",
        ),
        MetricSemanticClass(
            "factor.X.teacher_student.coordinate_projection_abs",
            "physical_projection",
            "X coordinate projection absolute overlap",
            ["physical_overlap_X_mean", "physical_overlap_X_std"],
            "factor",
            "full",
            "teacher-student",
            "absolute coordinate projection",
            "diagnostic",
            "diagnostic",
            "high",
            description="Coordinate projection diagnostic for X; gauge/sign/permutation sensitive.",
        ),
        MetricSemanticClass(
            "matrix.full.teacher_student.output_projection",
            "physical_projection",
            "matrix Y output projection overlap",
            ["physical_overlap_Y_mean", "physical_overlap_Y_std"],
            "matrix",
            "full",
            "teacher-student",
            "projection",
            "formal",
            "review",
            "high",
            description="Global output-space projection diagnostic; candidate order parameter after scale convention review.",
        ),
        MetricSemanticClass(
            "matrix.full.teacher_student.reconstruction_mse",
            "reconstruction_error",
            "matrix full reconstruction MSE",
            ["MSE", "MSE_std", "Gen_Error"],
            "matrix",
            "full",
            "teacher-student",
            "mean squared error",
            "formal",
            "not_order_parameter",
            "low",
            description="Dense reconstruction mean squared error.",
        ),
        MetricSemanticClass(
            "factor.W.replica.student_student.gram_cosine",
            "replica_overlap",
            "W replica Gram cosine",
            ["Q_W_replica_mean"],
            "factor",
            "replica",
            "student-student",
            "Gram cosine",
            "diagnostic",
            "diagnostic",
            "medium",
            description="Student-student W replica Gram overlap diagnostic.",
        ),
        MetricSemanticClass(
            "factor.X.replica.student_student.gram_cosine",
            "replica_overlap",
            "X replica Gram cosine",
            ["Q_X_replica_mean"],
            "factor",
            "replica",
            "student-student",
            "Gram cosine",
            "diagnostic",
            "diagnostic",
            "medium",
            description="Student-student X replica Gram overlap diagnostic.",
        ),
        MetricSemanticClass(
            "factor.W.replica.student_student.baseline_corrected_gram_cosine",
            "replica_overlap",
            "W replica baseline-corrected Gram cosine",
            ["Q_W_prime_replica_mean"],
            "factor",
            "replica",
            "student-student",
            "baseline-corrected Gram cosine",
            "diagnostic",
            "diagnostic",
            "medium",
            description="Student-student W replica baseline-corrected overlap diagnostic.",
        ),
        MetricSemanticClass(
            "factor.X.replica.student_student.baseline_corrected_gram_cosine",
            "replica_overlap",
            "X replica baseline-corrected Gram cosine",
            ["Q_X_prime_replica_mean"],
            "factor",
            "replica",
            "student-student",
            "baseline-corrected Gram cosine",
            "diagnostic",
            "diagnostic",
            "medium",
            description="Student-student X replica baseline-corrected overlap diagnostic.",
        ),
        MetricSemanticClass(
            "tensor.full.teacher_student.projection_overlap",
            "physical_projection",
            "tensor Y projection overlap",
            ["physical_overlap_Y_mean"],
            "tensor",
            "full",
            "teacher-student",
            "projection",
            "formal",
            "candidate",
            "high",
            description="Tensor-space teacher/student projection; scale convention must be reviewed before comparing with matrix variants.",
        ),
        MetricSemanticClass(
            "replica.heatmap.teacher_and_students.matrix",
            "replica_heatmap",
            "teacher plus replicas overlap matrix",
            ["overlap_matrix", "overlap_matrix_metric"],
            "matrix-valued",
            "replica",
            "teacher-student and student-student",
            "selected heatmap metric",
            "diagnostic",
            "not_order_parameter",
            "medium",
            description="Matrix-valued heatmap payload for RSB/replica visualization.",
        ),
    ]
    return {item.canonical_key: item for item in classes}


def _attach_metric_semantic_metadata(specs: List[MetricSpec]) -> List[MetricSpec]:
    mapping = {
        "matrix.full.Q_Y": "matrix.full.teacher_student.output_cosine",
        "matrix.observed.Q_Y": "matrix.observed.teacher_student.output_cosine",
        "matrix.unobserved.Q_Y": "matrix.unobserved.teacher_student.output_cosine",
        "spreading.observed.Q_Y": "spreading.observed.teacher_student.F_aware_output_cosine",
        "spreading.unobserved.Q_Y": "spreading.unobserved.teacher_student.dense_output_cosine",
        "tensor.full.Q_Y": "tensor.full.teacher_student.cp_tensor_cosine",
        "tensor.serial_observed.Q_Y": "tensor.observed.teacher_student.serial_reconstruction_quality",
        "tensor.observed.Q_Y": "tensor.observed.teacher_student.reconstruction_quality",
        "matrix.factor.Q_W": "factor.W.teacher_student.gram_cosine",
        "matrix.factor.Q_X": "factor.X.teacher_student.gram_cosine",
        "matrix.physical_overlap": "matrix.full.teacher_student.output_projection",
        "matrix.error.MSE": "matrix.full.teacher_student.reconstruction_mse",
        "replica.factor": "factor.W.replica.student_student.gram_cosine",
        "tensor.physical_overlap_Y": "tensor.full.teacher_student.projection_overlap",
    }
    classes = get_metric_semantic_classes()
    enriched = []
    for spec in specs:
        canonical_key = mapping.get(spec.key, "")
        semantic_class = classes.get(canonical_key)
        if not semantic_class:
            enriched.append(spec)
            continue
        enriched.append(replace(
            spec,
            canonical_key=semantic_class.canonical_key,
            equivalence_class=semantic_class.equivalence_class,
            legacy_aliases=list(semantic_class.legacy_aliases),
            result_role=semantic_class.result_role,
            order_parameter_status=semantic_class.order_parameter_status,
            review_status=semantic_class.review_status,
            risk_level=semantic_class.risk_level,
        ))
    return enriched


def _metric_flat_key_canonical_key(metric_spec_key: str, flat_key: str, default: str) -> str:
    overrides = {
        ("matrix.factor.Q_W", "Q_W_prime_mean"): "factor.W.teacher_student.baseline_corrected_gram_cosine",
        ("matrix.factor.Q_W", "Q_W_prime_std"): "factor.W.teacher_student.baseline_corrected_gram_cosine",
        ("matrix.factor.Q_X", "Q_X_prime_mean"): "factor.X.teacher_student.baseline_corrected_gram_cosine",
        ("matrix.factor.Q_X", "Q_X_prime_std"): "factor.X.teacher_student.baseline_corrected_gram_cosine",
        ("matrix.physical_overlap", "physical_overlap_W_mean"): "factor.W.teacher_student.coordinate_projection_abs",
        ("matrix.physical_overlap", "physical_overlap_W_std"): "factor.W.teacher_student.coordinate_projection_abs",
        ("matrix.physical_overlap", "physical_overlap_X_mean"): "factor.X.teacher_student.coordinate_projection_abs",
        ("matrix.physical_overlap", "physical_overlap_X_std"): "factor.X.teacher_student.coordinate_projection_abs",
        ("matrix.physical_overlap", "physical_overlap_Y_mean"): "matrix.full.teacher_student.output_projection",
        ("matrix.physical_overlap", "physical_overlap_Y_std"): "matrix.full.teacher_student.output_projection",
        ("replica.factor", "Q_W_replica_mean"): "factor.W.replica.student_student.gram_cosine",
        ("replica.factor", "Q_X_replica_mean"): "factor.X.replica.student_student.gram_cosine",
        ("replica.factor", "Q_W_prime_replica_mean"): "factor.W.replica.student_student.baseline_corrected_gram_cosine",
        ("replica.factor", "Q_X_prime_replica_mean"): "factor.X.replica.student_student.baseline_corrected_gram_cosine",
    }
    return overrides.get((metric_spec_key, flat_key), default)


def get_algorithm_metric_keys(algorithm_key: str) -> List[str]:
    """Flat metric keys declared for an algorithm through MetricSpec."""
    algorithm_specs = get_algorithm_specs()
    metric_specs = get_metric_specs()
    if algorithm_key not in algorithm_specs:
        return []
    keys = []
    for metric_key in algorithm_specs[algorithm_key].produced_metrics:
        spec = metric_specs.get(metric_key)
        if not spec:
            continue
        keys.extend(spec.produces)
    return sorted(set(keys))


def get_algorithm_metric_semantics(algorithm_key: str) -> Dict[str, List[Dict[str, str]]]:
    """Map legacy flat metric keys to semantic MetricSpec metadata."""
    algorithm_specs = get_algorithm_specs()
    metric_specs = get_metric_specs()
    if algorithm_key not in algorithm_specs:
        return {}
    semantics: Dict[str, List[Dict[str, str]]] = {}
    for metric_key in algorithm_specs[algorithm_key].produced_metrics:
        spec = metric_specs.get(metric_key)
        if not spec:
            continue
        for flat_key in spec.produces:
            canonical_key = _metric_flat_key_canonical_key(spec.key, flat_key, spec.canonical_key)
            semantic_class = get_metric_semantic_classes().get(canonical_key)
            if semantic_class:
                payload = {
                    "metric_spec": spec.key,
                    "canonical_key": semantic_class.canonical_key,
                    "equivalence_class": semantic_class.equivalence_class,
                    "space": semantic_class.space,
                    "scope": semantic_class.scope,
                    "relation": semantic_class.relation,
                    "normalization": semantic_class.normalization,
                    "physical_meaning": semantic_class.description,
                    "result_role": semantic_class.result_role,
                    "order_parameter_status": semantic_class.order_parameter_status,
                    "review_status": semantic_class.review_status,
                    "risk_level": semantic_class.risk_level,
                }
            else:
                payload = {
                    "metric_spec": spec.key,
                    "canonical_key": spec.canonical_key,
                    "equivalence_class": spec.equivalence_class,
                    "space": spec.space,
                    "scope": spec.scope,
                    "relation": spec.relation,
                    "normalization": spec.normalization,
                    "physical_meaning": spec.physical_meaning,
                    "result_role": spec.result_role,
                    "order_parameter_status": spec.order_parameter_status,
                    "review_status": spec.review_status,
                    "risk_level": spec.risk_level,
                }
            semantics.setdefault(flat_key, []).append(dict(payload))
    return semantics


def get_metric_schema(algorithm_key: str, metric_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Result-schema metadata for legacy flat metric keys under one algorithm."""
    semantics = get_algorithm_metric_semantics(algorithm_key)
    requested = set(metric_keys or semantics.keys())
    flat_key_index = {
        flat_key: payload
        for flat_key, payload in semantics.items()
        if flat_key in requested
    }
    canonical_keys = {
        item.get("canonical_key")
        for payloads in flat_key_index.values()
        for item in payloads
        if item.get("canonical_key")
    }
    classes = get_metric_semantic_classes()
    semantic_classes = {
        key: {
            "canonical_key": classes[key].canonical_key,
            "equivalence_class": classes[key].equivalence_class,
            "display_name": classes[key].display_name,
            "legacy_aliases": list(classes[key].legacy_aliases),
            "space": classes[key].space,
            "scope": classes[key].scope,
            "relation": classes[key].relation,
            "normalization": classes[key].normalization,
            "result_role": classes[key].result_role,
            "order_parameter_status": classes[key].order_parameter_status,
            "risk_level": classes[key].risk_level,
            "review_status": classes[key].review_status,
            "description": classes[key].description,
        }
        for key in sorted(canonical_keys)
        if key in classes
    }
    review_required = sorted(
        key for key, item in semantic_classes.items()
        if item["review_status"] != "approved"
        or item["risk_level"] in {"medium", "high"}
        or item["order_parameter_status"] == "review"
    )
    return {
        "schema_version": 2,
        "algorithm_key": algorithm_key,
        "compatibility": {
            "legacy_flat_keys_preserved": True,
            "flat_keys_require_algorithm_context": True,
        },
        "semantic_classes": semantic_classes,
        "flat_key_index": flat_key_index,
        "review_required": review_required,
    }


def get_tensor_parity_specs() -> Dict[str, TensorParitySpec]:
    """Machine-readable parity checklist for tensor serial/parallel paths."""
    specs = [
        TensorParitySpec(
            "teacher_scale",
            area="teacher",
            serial_status="implicit_in_serial_teacher_factors",
            parallel_status="normalizes_W_X_in_create_teacher_factors",
            risk="high",
            notes="Must verify serial and parallel use the same teacher factor scale before numeric parity.",
        ),
        TensorParitySpec(
            "alpha_normalization",
            area="data",
            serial_status="serial_hypergraph_edge_count",
            parallel_status="tensor_supergraph_edge_count",
            risk="high",
            notes="Same alpha must correspond to the same observation density definition.",
        ),
        TensorParitySpec(
            "graph_definition",
            area="data",
            serial_status="tensor_hypergraph",
            parallel_status="tensor_supergraph",
            risk="high",
            notes="Both graph objects must represent the same statistical ensemble.",
        ),
        TensorParitySpec(
            "f_distribution",
            area="data",
            serial_status="spreading.f_distribution",
            parallel_status="spreading.f_distribution",
            risk="medium",
            notes="Rademacher/Gaussian meaning and normalization must match.",
        ),
        TensorParitySpec(
            "damping_semantics",
            area="algorithm",
            serial_status="algorithm_params.damping",
            parallel_status="algorithm_params.damping",
            risk="medium",
            notes="The meaning of damping endpoints must be identical.",
        ),
        TensorParitySpec(
            "onsager_handling",
            area="algorithm",
            serial_status="not_fully_declared",
            parallel_status="onsager_residual_state_capability",
            risk="high",
            notes="Onsager correction and prev_s semantics must be reviewed before merging.",
        ),
        TensorParitySpec(
            "qy_semantics",
            area="metric",
            serial_status="tensor.observed.Q_Y_only",
            parallel_status="tensor.full.Q_Y_and_tensor.observed.Q_Y",
            risk="high",
            notes="Flat Q_Y_mean cannot be compared unless scope is explicit.",
        ),
        TensorParitySpec(
            "result_schema",
            area="result",
            serial_status="legacy_tensor_metrics_only",
            parallel_status="legacy_tensor_metrics_only",
            risk="medium",
            notes="Both paths must eventually return native AlgorithmResult, not private _batch_metrics.",
        ),
        TensorParitySpec(
            "initialization_semantics",
            area="algorithm",
            serial_status="random internal initialization",
            parallel_status="init_mode random/spectral/warm_start",
            risk="high",
            notes="Serial and parallel initialization modes are not contract-equivalent.",
        ),
        TensorParitySpec(
            "seed_partition",
            area="randomness",
            serial_status="seed + s*1000 + int(alpha*100)",
            parallel_status="legacy seed + batch_idx; opt-in partition_invariant available",
            risk="high",
            notes="Default legacy path is still batch-partition sensitive; opt-in partition_invariant reduces this risk but does not prove numeric parity with serial.",
        ),
        TensorParitySpec(
            "dtype_compile_semantics",
            area="resource",
            serial_status="float32 eager path",
            parallel_status="TF32/BF16/torch.compile optional path",
            risk="medium",
            notes="Parallel path resource settings are part of the numerical execution path.",
        ),
        TensorParitySpec(
            "batching_semantics",
            area="resource",
            serial_status="alpha and sample loops are serial",
            parallel_status="algorithm-internal probe-based alpha batches plus sample parallelism",
            risk="high",
            notes="Parallel batching is not a pure metadata change until seed partition is invariant.",
        ),
    ]
    return {spec.key: spec for spec in specs}


def get_tensor_parity_report() -> Dict[str, Any]:
    """Current contract-level gap report for tensor serial/parallel paths."""
    algorithm_specs = get_algorithm_specs()
    serial = algorithm_specs["bigamp_tensor"]
    parallel = algorithm_specs["bigamp_tensor_parallel"]
    serial_metrics = set(serial.produced_metrics)
    parallel_metrics = set(parallel.produced_metrics)
    serial_data = set(serial.data_requirements)
    parallel_data = set(parallel.data_requirements)
    return {
        "serial_key": serial.key,
        "parallel_key": parallel.key,
        "same_result_contract": serial.result_contract == parallel.result_contract,
        "result_contracts": {
            "serial": serial.result_contract,
            "parallel": parallel.result_contract,
        },
        "shared_metrics": sorted(serial_metrics & parallel_metrics),
        "serial_missing_parallel_metrics": sorted(parallel_metrics - serial_metrics),
        "parallel_missing_serial_metrics": sorted(serial_metrics - parallel_metrics),
        "shared_data_requirements": sorted(serial_data & parallel_data),
        "serial_only_data_requirements": sorted(serial_data - parallel_data),
        "parallel_only_data_requirements": sorted(parallel_data - serial_data),
        "parity_items": [item.key for item in get_tensor_parity_specs().values()],
        "parity_item_details": {
            item.key: {
                "area": item.area,
                "serial_status": item.serial_status,
                "parallel_status": item.parallel_status,
                "risk": item.risk,
                "numeric_sensitive": item.numeric_sensitive,
                "review_required": item.review_required,
                "notes": item.notes,
            }
            for item in get_tensor_parity_specs().values()
        },
    }


def get_output_specs() -> Dict[str, OutputSpec]:
    specs = [
        OutputSpec("scalar_curves", ["metrics_by_alpha"], ["Q_Y_mean", "Q_W_mean"], ["plots/qy_evolution.png", "plots/overlap_evolution.png"], "Scalar metric curves."),
        OutputSpec("custom_curves", ["metrics_by_alpha"], ["output.plots"], ["plots/custom_plot_*.png"], "User-requested custom curves."),
        OutputSpec("fallback_W_heatmap", ["matrix_factors"], ["W_students", "W_teacher"], ["plots/heatmap_W_alpha_*.png"], "Fallback W Gram heatmap."),
        OutputSpec("tensor_heatmap", ["overlap_matrix"], ["overlap_matrix", "overlap_matrix_metric"], ["plots/heatmap_Y_alpha_*.png", "plots/heatmap_W_alpha_*.png"], "Tensor replica heatmap."),
        OutputSpec("tensor_gif", ["overlap_matrix"], ["heatmap sequence"], ["plots/animation_Y.gif", "plots/animation_W.gif"], "Heatmap GIF sequence."),
        OutputSpec("latest_display", ["config.json", "metadata.json"], ["selected small png"], ["latest/index.json"], "Lightweight latest display snapshot."),
        OutputSpec("plotting", ["metrics_by_alpha"], ["legacy ResultPlotter"], ["plots/*.png"], "Registered plotting backend used by legacy output registry."),
        OutputSpec("storage", ["metrics_by_alpha"], ["legacy ResultStorage"], ["metrics.json", "manifest.json"], "Registered storage backend used by legacy output registry."),
        OutputSpec("combined", ["metrics_by_alpha"], ["legacy CombinedOutput"], ["plots/*.png", "metrics.json"], "Legacy combined output backend."),
    ]
    return {spec.key: spec for spec in specs}


def get_intervention_specs() -> Dict[str, InterventionSpec]:
    specs = [
        InterventionSpec("cold_start", "before_initialize", [], ["student_factors"], ["agd", "bigamp", "bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel"], True, "Random initialization."),
        InterventionSpec("warm_start", "before_initialize", ["teacher_factors"], ["student_factors"], ["bigamp_spreading", "bigamp_tensor_parallel"], True, "Teacher-assisted initialization."),
        InterventionSpec("adaptive_restart", "on_plateau", ["student_factors"], ["student_factors"], ["bigamp_spreading"], True, "Existing adaptive restart behavior."),
        InterventionSpec("metropolis_kick", "after_step", ["current_student_state", "current_metric_or_energy"], ["student_factors"], [], True, "Reserved framework slot; not active by default."),
    ]
    return {spec.key: spec for spec in specs}


def get_probe_specs() -> Dict[str, ProbeSpec]:
    specs = [
        ProbeSpec(
            "state_slice",
            "after_step",
            ["student_factors", "step_index"],
            ["slice_artifact"],
            ["agd", "bigamp", "bigamp_spreading"],
            "declared_only",
            "Read-only runtime state snapshot for matrix factor algorithms.",
        ),
        ProbeSpec(
            "tensor_state_slice",
            "after_step",
            ["tensor_factors", "step_index"],
            ["tensor_slice_artifact"],
            ["bigamp_tensor", "bigamp_tensor_parallel", "agd_tensor"],
            "declared_only",
            "Read-only runtime state snapshot for tensor factor algorithms.",
        ),
        ProbeSpec(
            "variance_slice",
            "after_step",
            ["factor_variances", "step_index"],
            ["variance_slice_artifact"],
            ["bigamp", "bigamp_spreading"],
            "declared_only",
            "Read-only variance diagnostic slice.",
        ),
        ProbeSpec(
            "batch_summary",
            "after_batch",
            [],
            ["batch_summary"],
            ["agd", "bigamp", "bigamp_spreading", "bigamp_tensor", "bigamp_tensor_parallel"],
            "runtime_active",
            "Read-only runner-level batch summary without touching algorithm internals.",
        ),
    ]
    return {spec.key: spec for spec in specs}


def get_analyzer_specs() -> Dict[str, AnalyzerSpec]:
    specs = [
        AnalyzerSpec(
            "replica_summary",
            ["metrics_by_alpha"],
            ["replica_summary"],
            ["agd", "bigamp", "bigamp_spreading", "bigamp_tensor_parallel"],
            "Post-run replica summary from produced metrics.",
        ),
        AnalyzerSpec(
            "tensor_heatmap_summary",
            ["overlap_matrix"],
            ["tensor_heatmap_summary"],
            ["bigamp_tensor_parallel"],
            "Post-run tensor heatmap summary.",
        ),
    ]
    return {spec.key: spec for spec in specs}


def get_algorithm_source_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for algorithm source files.

    This prevents new algorithm/support files from silently appearing without a
    classification. Paths are repo-relative.
    """
    specs = [
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/__init__.py", "package"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/base.py", "base"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/agd.py", "algorithm_entry", "agd"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/agd_tensor.py", "algorithm_entry", "agd_tensor", "experimental_unintegrated"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/combined.py", "algorithm_entry", "combined", "non_trainable_helper"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/legacy/agd_spreading.py", "legacy", "agd_spreading", "legacy_broken"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/__init__.py", "package"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/core.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/f_gen.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/standard.py", "algorithm_entry", "bigamp"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/spreading.py", "algorithm_entry", "bigamp_spreading"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading.py", "algorithm_entry", "bigamp_tensor"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_spreading_parallel.py", "algorithm_entry", "bigamp_tensor_parallel"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_contract.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/step.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_data.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_hypergraph.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_memory.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_step.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_step_batch.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_step_super.py", "support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/algorithms/bigamp/tensor_supergraph.py", "support_module"),
    ]
    return {spec.path: spec for spec in specs}


def get_auxiliary_source_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for non-package experiment/debug sources."""
    specs = [
        SourceInventorySpec("experiments/teacher_matrix_analysis/__init__.py", "local_analysis_package", status="local_experiment"),
        SourceInventorySpec("experiments/teacher_matrix_analysis/main.py", "local_analysis", status="local_experiment"),
        SourceInventorySpec("scripts/analysis/compare_algorithms.py", "local_analysis", status="local_experiment"),
        SourceInventorySpec("scripts/analysis/compare_runs.py", "local_analysis", status="local_experiment"),
        SourceInventorySpec("scripts/analysis/generate_final_plot.py", "local_analysis", status="local_experiment"),
        SourceInventorySpec("scripts/analysis/generate_qy_scan.py", "local_analysis", status="local_experiment"),
        SourceInventorySpec("scripts/analysis/inspect_results.py", "local_analysis", status="local_experiment"),
        SourceInventorySpec("scripts/debug/debug_edge_count.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/debug/debug_graph.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/debug/debug_tensor_agd_autograd.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/debug/debug_tensor_agd_baseline.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/debug/debug_tensor_amp_stepwise.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/debug/debug_tensor_leak.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/debug/debug_tensor_multistep.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/debug/debug_tensor_q_compare.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/debug/debug_tensor_tau.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/debug/plot_debug_damping.py", "debug_only", status="debug_only"),
        SourceInventorySpec("scripts/experiments/compare_scaling.py", "local_experiment", status="local_experiment"),
        SourceInventorySpec("scripts/experiments/compare_scaling_fast.py", "local_experiment", status="local_experiment"),
        SourceInventorySpec("scripts/experiments/generate_all_scans.py", "local_experiment", status="local_experiment"),
        SourceInventorySpec("scripts/experiments/memory_benchmark.py", "local_experiment", status="local_experiment"),
        SourceInventorySpec("scripts/experiments/run_400x100_only.py", "local_experiment", status="local_experiment"),
        SourceInventorySpec("scripts/experiments/run_full_metrics_experiment.py", "local_experiment", status="local_experiment"),
        SourceInventorySpec("scripts/experiments/run_scaling_experiment.py", "local_experiment", status="local_experiment"),
        SourceInventorySpec("scripts/experiments/run_scaling_experiment_fixed.py", "local_experiment", status="local_experiment"),
        SourceInventorySpec("scripts/experiments/run_third_only.py", "local_experiment", status="local_experiment"),
        SourceInventorySpec("scripts/maintenance/update_latest_results.py", "maintenance", status="support_module"),
        SourceInventorySpec("scripts/verification/verify_bipartite_scaling.py", "verification_reference", status="local_gpu_check"),
        SourceInventorySpec("scripts/verification/verify_est.py", "verification_reference", status="local_gpu_check"),
        SourceInventorySpec("scripts/verification/verify_scaling.py", "verification_reference", status="local_gpu_check"),
        SourceInventorySpec("src/matrix_factorization/export/__init__.py", "export_support", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/export/__main__.py", "export_support", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/export/bundler.py", "export_support", status="support_module"),
        SourceInventorySpec("tests/debug/analyze_qy_drop.py", "debug_only", status="debug_only"),
        SourceInventorySpec("tests/debug/debug_alpha3.py", "debug_only", status="debug_only"),
        SourceInventorySpec("tests/debug/debug_variance.py", "debug_only", status="debug_only"),
        SourceInventorySpec("tests/debug/diagnose_system.py", "debug_only", status="debug_only"),
        SourceInventorySpec("tests/debug/diagnose_training.py", "debug_only", status="debug_only"),
        SourceInventorySpec("tests/verification/final_verification.py", "verification_reference", status="local_gpu_check"),
        SourceInventorySpec("tests/verification/verify_all_types.py", "verification_reference", status="local_gpu_check"),
        SourceInventorySpec("tests/verification/verify_formula.py", "verification_reference", status="local_gpu_check"),
        SourceInventorySpec("tests/verification/verify_qy_unobs.py", "verification_reference", status="local_gpu_check"),
    ]
    return {spec.path: spec for spec in specs}


def get_repository_surface_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for files that are allowed at repo surface.

    This is deliberately small. New temporary Python/Markdown/YAML files at the
    repository root should either move into a classified area or be added here
    with an explicit status.
    """
    specs = [
        SourceInventorySpec("AGENTS.md", "agent_guidance", status="active_path"),
        SourceInventorySpec("AGENT.md", "legacy_agent_guidance", status="legacy_reference"),
        SourceInventorySpec("README.md", "project_readme", status="active_path"),
        SourceInventorySpec("TENSOR_BUG_REPORT.md", "historical_bug_report", status="legacy_reference"),
        SourceInventorySpec("pyproject.toml", "packaging_config", status="active_path"),
    ]
    return {spec.path: spec for spec in specs}


def get_config_source_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for tracked config entrypoints."""
    specs = [
        SourceInventorySpec("configs/README.md", "config_readme", status="active_path"),
        SourceInventorySpec("configs/local_gpu/tensor_local_gpu.yaml", "local_gpu_config", status="local_gpu_check"),
        SourceInventorySpec("src/matrix_factorization/config.yaml", "default_config", status="active_path"),
    ]
    return {spec.path: spec for spec in specs}


def get_runtime_extension_source_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for hook/probe/intervention source files."""
    specs = [
        SourceInventorySpec("src/matrix_factorization/modules/interventions/__init__.py", "runtime_extension_package", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/interventions/base.py", "intervention_base", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/interventions/executor.py", "runtime_extension_executor", status="active_path"),
    ]
    return {spec.path: spec for spec in specs}


def get_output_source_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for output/plot/export source files."""
    specs = [
        SourceInventorySpec("src/matrix_factorization/modules/outputs/__init__.py", "output_package", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/outputs/base.py", "output_base", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/outputs/combined.py", "registered_output", "combined", "legacy_reference"),
        SourceInventorySpec("src/matrix_factorization/modules/outputs/comparison.py", "plotting_helper", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/outputs/latest.py", "latest_display_exporter", status="active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/outputs/plot_registry.py", "plot_registry", status="active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/outputs/plotting.py", "plotting_backend", "plotting", "active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/outputs/publication_style.py", "plotting_style", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/outputs/spec_adapter.py", "output_contract_adapter", status="active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/outputs/storage.py", "registered_output", "storage", "support_module"),
    ]
    return {spec.path: spec for spec in specs}


def get_metric_source_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for metric source files."""
    specs = [
        SourceInventorySpec("src/matrix_factorization/modules/metrics/__init__.py", "metric_package", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/metrics/combined.py", "metric_composer", status="legacy_reference"),
        SourceInventorySpec("src/matrix_factorization/modules/metrics/contract_compute.py", "metric_contract_adapter", status="active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/metrics/overlap.py", "matrix_metric_definitions", status="active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/metrics/qy_unobserved.py", "matrix_metric_helper", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/metrics/replica.py", "replica_metric_helper", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/metrics/spec_adapter.py", "metric_contract_validator", status="active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/metrics/spreading.py", "spreading_metric_definitions", status="active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/metrics/tensor_metrics.py", "tensor_metric_definitions", status="active_path"),
    ]
    return {spec.path: spec for spec in specs}


def get_teacher_source_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for teacher source files."""
    specs = [
        SourceInventorySpec("src/matrix_factorization/modules/teachers/__init__.py", "teacher_package", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/teachers/base.py", "teacher_base", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/teachers/combined.py", "registered_teacher", "combined", "legacy_reference"),
        SourceInventorySpec("src/matrix_factorization/modules/teachers/orthogonal.py", "registered_teacher", "orthogonal", "active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/teachers/orthogonal_unit.py", "registered_teacher", "orthogonal_unit", "legacy_reference"),
        SourceInventorySpec("src/matrix_factorization/modules/teachers/random_spreading.py", "registered_teacher", "random_spreading", "legacy_reference"),
        SourceInventorySpec("src/matrix_factorization/modules/teachers/scaled_variance.py", "registered_teacher", "scaled_variance", "legacy_reference"),
        SourceInventorySpec("src/matrix_factorization/modules/teachers/standard.py", "registered_teacher", "standard", "active_path"),
    ]
    return {spec.path: spec for spec in specs}


def get_graph_source_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for graph source files."""
    specs = [
        SourceInventorySpec("src/matrix_factorization/modules/graphs/__init__.py", "graph_package", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/graphs/base.py", "graph_base", status="support_module"),
        SourceInventorySpec("src/matrix_factorization/modules/graphs/combined.py", "registered_graph", "combined", "legacy_reference"),
        SourceInventorySpec("src/matrix_factorization/modules/graphs/low_loop.py", "registered_graph", "low_loop", "legacy_reference"),
        SourceInventorySpec("src/matrix_factorization/modules/graphs/random.py", "registered_graph", "random", "legacy_reference"),
        SourceInventorySpec("src/matrix_factorization/modules/graphs/supergraph.py", "spreading_supergraph", status="active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/graphs/supergraph_general.py", "general_spreading_supergraph", status="active_path"),
        SourceInventorySpec("src/matrix_factorization/modules/graphs/uniform.py", "registered_graph", "uniform", "legacy_reference"),
    ]
    return {spec.path: spec for spec in specs}


def get_trial_source_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for tracked research trial files."""
    specs = [
        SourceInventorySpec("trials/README.md", "trial_readme", status="active_path"),
        SourceInventorySpec("trials/registry.yaml", "trial_registry", status="active_path"),
        SourceInventorySpec("trials/active/matrix_bigamp_quick/trial.yaml", "trial_manifest", "matrix_bigamp_quick", "active_path"),
        SourceInventorySpec("trials/active/matrix_bigamp_quick/config.yaml", "trial_config", "matrix_bigamp_quick", "active_path"),
        SourceInventorySpec("trials/active/matrix_bigamp_quick/summary.md", "trial_summary", "matrix_bigamp_quick", "active_path"),
    ]
    return {spec.path: spec for spec in specs}


def get_legacy_pytest_inventory() -> Dict[str, SourceInventorySpec]:
    """Machine-readable inventory for skipped legacy/local diagnostic tests."""
    specs = [
        SourceInventorySpec("tests/test_clamp.py", "legacy_local_gpu_pytest", status="legacy_skipped"),
        SourceInventorySpec("tests/test_damping.py", "legacy_local_gpu_pytest", status="legacy_skipped"),
        SourceInventorySpec("tests/test_independent_alpha.py", "legacy_local_gpu_pytest", status="legacy_skipped"),
        SourceInventorySpec("tests/test_llm_robustness.py", "legacy_config_advisor_pytest", status="legacy_skipped"),
        SourceInventorySpec("tests/test_scientific_pressure.py", "legacy_config_advisor_pytest", status="legacy_skipped"),
        SourceInventorySpec("tests/test_random_spreading.py", "legacy_random_spreading_pytest", status="legacy_skipped"),
        SourceInventorySpec("tests/test_supergraph.py", "legacy_spreading_parallel_pytest", status="legacy_skipped"),
        SourceInventorySpec("tests/test_onsager_impact.py", "local_gpu_tensor_diagnostic", status="legacy_skipped"),
        SourceInventorySpec("tests/test_tensor_parallel_verification.py", "local_gpu_tensor_parity_diagnostic", status="legacy_skipped"),
    ]
    return {spec.path: spec for spec in specs}
