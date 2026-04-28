"""Shared tensor algorithm contract helpers.

These helpers only package metadata and duplicate-free structural logic. They
must not change tensor algorithm formulas, random streams, graph construction,
or update rules.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def resolve_tensor_dims(matrix: Any, tensor_order: int) -> Tuple[int, ...]:
    """Return the existing tensor dims convention used by serial/parallel paths."""
    dims = [matrix.N1]
    if tensor_order >= 2:
        dims.append(matrix.N2)
    for _ in range(2, tensor_order):
        dims.append(matrix.N1)
    return tuple(dims)


def pack_tensor_parallel_metrics(result: Dict[str, Any], local_idx: int) -> Dict[str, Any]:
    """Pack the legacy tensor-parallel result arrays into flat metric keys."""
    metrics = {
        "Q_Y_mean": result["Q_Y"][local_idx],
        "Q_Y_std": result["Q_Y_std"][local_idx],
    }
    if "NMSE_Y" in result:
        metrics["NMSE_Y_mean"] = result["NMSE_Y"][local_idx]
    if "NMSE_Y_std" in result:
        metrics["NMSE_Y_std"] = result["NMSE_Y_std"][local_idx]
    if "FIT_Y" in result:
        metrics["FIT_Y_mean"] = result["FIT_Y"][local_idx]
    if "FIT_Y_std" in result:
        metrics["FIT_Y_std"] = result["FIT_Y_std"][local_idx]
    if "Q_Y_PROJ_ABS" in result:
        metrics["Q_Y_PROJ_ABS_mean"] = result["Q_Y_PROJ_ABS"][local_idx]
    if "Q_Y_PROJ_ABS_std" in result:
        metrics["Q_Y_PROJ_ABS_std"] = result["Q_Y_PROJ_ABS_std"][local_idx]
    if "Q_Y_observed" in result:
        metrics["Q_Y_observed_mean"] = result["Q_Y_observed"][local_idx]
    if "Q_Y_observed_std" in result:
        metrics["Q_Y_observed_std"] = result["Q_Y_observed_std"][local_idx]
    if "NMSE_Y_observed" in result:
        metrics["NMSE_Y_observed_mean"] = result["NMSE_Y_observed"][local_idx]
    if "NMSE_Y_observed_std" in result:
        metrics["NMSE_Y_observed_std"] = result["NMSE_Y_observed_std"][local_idx]
    if "FIT_Y_observed" in result:
        metrics["FIT_Y_observed_mean"] = result["FIT_Y_observed"][local_idx]
    if "FIT_Y_observed_std" in result:
        metrics["FIT_Y_observed_std"] = result["FIT_Y_observed_std"][local_idx]
    if "Q_Y_observed_PROJ_ABS" in result:
        metrics["Q_Y_observed_PROJ_ABS_mean"] = result["Q_Y_observed_PROJ_ABS"][local_idx]
    if "Q_Y_observed_PROJ_ABS_std" in result:
        metrics["Q_Y_observed_PROJ_ABS_std"] = result["Q_Y_observed_PROJ_ABS_std"][local_idx]
    if "Q_Y_unobserved" in result:
        metrics["Q_Y_unobserved_mean"] = result["Q_Y_unobserved"][local_idx]
    if "Q_Y_unobserved_std" in result:
        metrics["Q_Y_unobserved_std"] = result["Q_Y_unobserved_std"][local_idx]
    if "NMSE_Y_unobserved" in result:
        metrics["NMSE_Y_unobserved_mean"] = result["NMSE_Y_unobserved"][local_idx]
    if "NMSE_Y_unobserved_std" in result:
        metrics["NMSE_Y_unobserved_std"] = result["NMSE_Y_unobserved_std"][local_idx]
    if "FIT_Y_unobserved" in result:
        metrics["FIT_Y_unobserved_mean"] = result["FIT_Y_unobserved"][local_idx]
    if "FIT_Y_unobserved_std" in result:
        metrics["FIT_Y_unobserved_std"] = result["FIT_Y_unobserved_std"][local_idx]
    if "Q_Y_unobserved_PROJ_ABS" in result:
        metrics["Q_Y_unobserved_PROJ_ABS_mean"] = result["Q_Y_unobserved_PROJ_ABS"][local_idx]
    if "Q_Y_unobserved_PROJ_ABS_std" in result:
        metrics["Q_Y_unobserved_PROJ_ABS_std"] = result["Q_Y_unobserved_PROJ_ABS_std"][local_idx]
    if "Q_N" in result:
        metrics["Q_N_mean"] = result["Q_N"][local_idx]
    if "Q_N_std" in result:
        metrics["Q_N_std"] = result["Q_N_std"][local_idx]
    for key, values in result.items():
        if key.startswith("Q_N_mode") and not key.endswith("_std"):
            metrics[f"{key}_mean"] = values[local_idx]
        elif key.startswith("Q_N_mode") and key.endswith("_std"):
            metrics[key] = values[local_idx]
    if "overlap_matrices" in result:
        matrix = result["overlap_matrices"][local_idx]
        metrics["overlap_matrix"] = matrix.tolist() if hasattr(matrix, "tolist") else matrix
        metrics["overlap_matrix_metric"] = result.get("overlap_matrix_metric", "Q_Y")
    return metrics


def _json_scalar(value: Any) -> Any:
    """Return a JSON-friendly scalar without importing torch in this helper."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value).replace("torch.", "")


def build_tensor_execution_metadata(
    *,
    path: str,
    device: Any = None,
    requested_use_bf16: Optional[bool] = None,
    effective_use_bf16: Optional[bool] = None,
    dtype_fallback_policy: Optional[str] = None,
    dtype_status: str = "",
    storage_dtype: Any = None,
    precision_profile: Optional[str] = None,
    precision_fallback_policy: Optional[str] = None,
    normalization_profile: Optional[str] = None,
    normalization_schema_version: Optional[int] = None,
    normalization_convention: Optional[str] = None,
    requested_use_compile: Optional[bool] = None,
    compile_fallback_policy: Optional[str] = None,
    effective_use_compile: Optional[bool] = None,
    compiled_step_available: Optional[bool] = None,
    compiled_super_step_available: Optional[bool] = None,
    compile_status: str = "",
    compile_attempts: Optional[List[Dict[str, Any]]] = None,
    requested_use_tf32: Optional[bool] = None,
    tf32_matmul_enabled: Optional[bool] = None,
    tf32_cudnn_enabled: Optional[bool] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Build metadata for tensor execution choices without changing execution."""
    return {
        "path": path,
        "device": str(device) if device is not None else None,
        "requested_use_bf16": requested_use_bf16,
        "effective_use_bf16": effective_use_bf16,
        "dtype_fallback_policy": dtype_fallback_policy,
        "dtype_status": dtype_status,
        "storage_dtype": _json_scalar(storage_dtype),
        "precision_profile": precision_profile,
        "precision_fallback_policy": precision_fallback_policy,
        "normalization_profile": normalization_profile,
        "normalization_schema_version": normalization_schema_version,
        "normalization_convention": normalization_convention,
        "requested_use_compile": requested_use_compile,
        "compile_fallback_policy": compile_fallback_policy,
        "effective_use_compile": effective_use_compile,
        "compiled_step_available": compiled_step_available,
        "compiled_super_step_available": compiled_super_step_available,
        "compile_status": compile_status,
        "compile_attempts": list(compile_attempts or []),
        "requested_use_tf32": requested_use_tf32,
        "tf32_matmul_enabled": tf32_matmul_enabled,
        "tf32_cudnn_enabled": tf32_cudnn_enabled,
        "metadata_only": True,
        "notes": notes,
    }


def build_tensor_alpha_batch_metadata(
    *,
    planner: str,
    device: Any,
    alpha_values_input: Iterable[float],
    alpha_batches: Iterable[Iterable[float]],
    sort_policy: str,
    probe_enabled: bool,
    seed_partition_sensitive: bool,
    probe_method: str = "",
    alpha_max: Optional[float] = None,
    probe_result_gb: Optional[float] = None,
    probe_cache_hit: Optional[bool] = None,
    total_memory_gb: Optional[float] = None,
    target_memory_gb: Optional[float] = None,
    target_memory_fraction: Optional[float] = None,
    max_alphas_per_batch: Optional[int] = None,
    fallback_reason: str = "",
    empty_cache_between_batches: bool = False,
) -> Dict[str, Any]:
    """Build metadata for tensor alpha batching without driving batch choices."""
    batches = [[float(alpha) for alpha in batch] for batch in alpha_batches]
    return {
        "planner": planner,
        "device": str(device),
        "alpha_values_input": [float(alpha) for alpha in alpha_values_input],
        "alpha_values_execution_order": [float(alpha) for batch in batches for alpha in batch],
        "alpha_batches": batches,
        "num_batches": len(batches),
        "sort_policy": sort_policy,
        "probe_enabled": probe_enabled,
        "probe_method": probe_method,
        "alpha_max": None if alpha_max is None else float(alpha_max),
        "probe_result_gb": None if probe_result_gb is None else float(probe_result_gb),
        "probe_cache_hit": probe_cache_hit,
        "total_memory_gb": None if total_memory_gb is None else float(total_memory_gb),
        "target_memory_gb": None if target_memory_gb is None else float(target_memory_gb),
        "target_memory_fraction": target_memory_fraction,
        "max_alphas_per_batch": max_alphas_per_batch,
        "fallback_reason": fallback_reason,
        "seed_partition_sensitive": seed_partition_sensitive,
        "empty_cache_between_batches": empty_cache_between_batches,
        "metadata_only": True,
    }


def build_tensor_result_metadata(
    *,
    algorithm_key: str,
    result_contract: str,
    result_source: str,
    dims: Iterable[int],
    tensor_order: int,
    graph_kind: str,
    alpha_values_original: Optional[List[float]] = None,
    alpha_values_execution_order: Optional[List[float]] = None,
    batching_source: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build common metrics-only AlgorithmResult metadata for tensor paths."""
    metadata = {
        "algorithm_key": algorithm_key,
        "result_contract": result_contract,
        "result_source": result_source,
        "matrix_factors_available": False,
        "tensor_order": int(tensor_order),
        "dims": [int(value) for value in dims],
        "graph_kind": graph_kind,
        "batching_source": batching_source,
    }
    if alpha_values_original is not None:
        metadata["alpha_values_original"] = [float(value) for value in alpha_values_original]
    if alpha_values_execution_order is not None:
        metadata["alpha_values_execution_order"] = [float(value) for value in alpha_values_execution_order]
    if extra:
        metadata.update(extra)
    return metadata
