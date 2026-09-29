import numpy as np

from fhm_dapp_tim.fhm_dapp.uji_v9_postlock_audit import bootstrap_mean_interval


def test_bootstrap_mean_interval_is_deterministic_and_contains_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    first = bootstrap_mean_interval(values, repetitions=200, seed=7)
    second = bootstrap_mean_interval(values, repetitions=200, seed=7)
    assert first == second
    assert first["lower"] < first["mean"] < first["upper"]
    assert first["coordinate_groups"] == 4


def test_bootstrap_mean_interval_rejects_nonfinite_values():
    try:
        bootstrap_mean_interval(np.array([1.0, np.nan]))
    except ValueError as error:
        assert "finite vector" in str(error)
    else:
        raise AssertionError("non-finite input must be rejected")
