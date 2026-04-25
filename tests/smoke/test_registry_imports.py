def test_primary_algorithms_are_registered():
    import matrix_factorization.modules.algorithms  # noqa: F401
    from matrix_factorization.modules.registry import get_algorithm

    for key in [
        "agd",
        "bigamp",
        "bigamp_spreading",
        "bigamp_tensor",
        "bigamp_tensor_parallel",
    ]:
        assert get_algorithm(key).cls is not None
