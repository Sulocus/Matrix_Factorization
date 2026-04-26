from matrix_factorization.core.scan_planning import build_scan_plan


def test_mixed_axis_scan_expands_to_full_parameter_space():
    plan = build_scan_plan({
        "scan": {
            "axes": {
                "onsager": {"path": "spreading.onsager_correction", "values": [False, True]},
                "damping": {"path": "algorithm_params.damping", "values": [0.2, 0.5, 0.8]},
                "init": {
                    "kind": "composite",
                    "values": {
                        "cold": {"algorithm_params.init_mode": "random"},
                        "warm_095": {
                            "algorithm_params.init_mode": "teacher",
                            "algorithm_params.init_overlap": 0.95,
                        },
                    },
                },
                "size": {
                    "kind": "composite",
                    "values": {
                        "N4_M2": {"matrix.N1": 4, "matrix.N2": 4, "matrix.M": 2},
                        "N6_M3": {"matrix.N1": 6, "matrix.N2": 6, "matrix.M": 3},
                    },
                },
                "alpha": {"path": "alpha", "values": [0.0, 0.1]},
            }
        }
    })

    assert plan.scan_kind == "parameter_space"
    assert plan.num_points == 48
    assert len({point.effective_config_hash for point in plan.points}) == 48
    assert all(point.alpha in {0.0, 0.1} for point in plan.points)
    assert {point.init_overlap for point in plan.points} == {0.0, 0.95}


def test_non_foldable_axes_define_execution_groups():
    plan = build_scan_plan({
        "scan": {
            "axes": {
                "damping": {"path": "algorithm_params.damping", "values": [0.2, 0.5]},
                "alpha": {"path": "alpha", "values": [0.0, 0.1, 0.2]},
            }
        }
    })

    assert plan.execution_constraints.foldable_axes == ["alpha"]
    assert {group.group_id for group in plan.grouping} == {"damping=0.2", "damping=0.5"}
    assert all(len(group.point_ids) == 3 for group in plan.grouping)
