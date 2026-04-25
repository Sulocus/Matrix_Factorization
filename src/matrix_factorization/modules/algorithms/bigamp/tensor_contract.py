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
    if "Q_Y_observed" in result:
        metrics["Q_Y_observed_mean"] = result["Q_Y_observed"][local_idx]
    if "Q_Y_observed_std" in result:
        metrics["Q_Y_observed_std"] = result["Q_Y_observed_std"][local_idx]
    if "physical_overlap" in result:
        metrics["physical_overlap_Y_mean"] = result["physical_overlap"][local_idx]
    if "overlap_matrices" in result:
        matrix = result["overlap_matrices"][local_idx]
        metrics["overlap_matrix"] = matrix.tolist() if hasattr(matrix, "tolist") else matrix
        metrics["overlap_matrix_metric"] = result.get("overlap_matrix_metric", "Q_Y")
    return metrics


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
