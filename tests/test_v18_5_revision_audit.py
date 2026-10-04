from fhm_dapp_tim.fhm_dapp.v18_5_revision_audit import (
    _synthetic_math_checks,
)


def test_revision_audit_synthetic_math_contracts():
    result = _synthetic_math_checks()
    assert result["epsilon_zero_identity"] is True
    assert result["beta_greater_than_one"] > 1.0
    assert result["minimum_energy_tie_limit"] is True
    assert result["energy_affine_invariance"] is True
    assert result["synthetic_calibration_cells"] == 192
    assert result["synthetic_parsimony_choice"] == {
        "temperature": 1.0,
        "epsilon_kl": 0.0,
        "scale_multiplier": 1.0,
    }
