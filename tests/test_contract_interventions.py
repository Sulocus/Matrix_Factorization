import pytest

from matrix_factorization.core.contracts import AlgorithmStateView, get_algorithm_specs, get_intervention_specs
from matrix_factorization.modules.interventions import (
    HookPoint,
    RuntimeHookContext,
    build_intervention,
)


def test_intervention_specs_declare_state_contracts():
    specs = get_intervention_specs()

    warm_start = specs["warm_start"]
    assert warm_start.trigger == "before_initialize"
    assert "teacher_factors" in warm_start.requires_state
    assert "student_factors" in warm_start.modifies_state
    assert warm_start.physical_sensitive

    metropolis = specs["metropolis_kick"]
    assert metropolis.trigger == "after_step"
    assert "current_student_state" in metropolis.requires_state


def test_intervention_implementation_validates_against_algorithm_spec():
    intervention = build_intervention("warm_start")
    algorithm_spec = get_algorithm_specs()["bigamp_spreading"]

    intervention.validate_compatible(algorithm_spec)


def test_incompatible_intervention_implementation_fails_validation():
    intervention = build_intervention("metropolis_kick")
    algorithm_spec = get_algorithm_specs()["bigamp_spreading"]

    with pytest.raises(ValueError, match="not compatible"):
        intervention.validate_compatible(algorithm_spec)


def test_intervention_base_is_noop_until_behavior_is_migrated():
    intervention = build_intervention("cold_start")
    state = AlgorithmStateView(student_factors={"W": "old"})
    context = RuntimeHookContext(
        algorithm_key="bigamp",
        hook=HookPoint.BEFORE_INITIALIZE,
    )

    assert intervention.apply(state, context) is state
