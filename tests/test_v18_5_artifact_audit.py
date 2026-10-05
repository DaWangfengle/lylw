import copy

import pytest

from fhm_dapp_tim.fhm_dapp.v18_5_artifact_audit import (
    validate_calibration_surface,
)
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG


def _surface():
    cells = []
    for temperature in map(float, V185_CONFIG["temperature_grid"]):
        for epsilon in map(float, V185_CONFIG["epsilon_kl_grid"]):
            for scale in map(float, V185_CONFIG["scale_grid"]):
                cells.append(
                    {
                        "cell_id": (
                            f"T={temperature:.2f}|E={epsilon:.3f}|"
                            f"S={scale:.2f}"
                        ),
                        "temperature": temperature,
                        "epsilon_kl": epsilon,
                        "scale_multiplier": scale,
                        "energy_score_m": 5.0,
                        "energy_score_se_m": 0.1,
                        "nll": 6.0,
                        "nll_se": 0.1,
                        "posterior_mean_mae_m": 7.0,
                        "posterior_mean_mae_se_m": 0.2,
                        "mean_entropy_nats": 2.0,
                        "mean_effective_anchors": 7.4,
                        "mean_top1_probability": 0.4,
                        "mean_radio_compatibility": 0.5,
                        "mean_realized_kl": (
                            0.0 if epsilon == 0.0 else 0.5 * epsilon
                        ),
                        "mean_effective_epsilon_kl": (
                            0.0 if epsilon == 0.0 else 0.5 * epsilon
                        ),
                        "max_kl_budget_excess": 0.0,
                        "dual_beta_infinite_fraction": 0.0,
                        "mean_finite_dual_beta": (
                            0.0 if epsilon == 0.0 else 1.0
                        ),
                    }
                )
    return cells


def test_artifact_surface_validator_accepts_exact_registered_grid():
    summary = validate_calibration_surface(_surface())
    assert summary["cells"] == 192
    assert summary["identity_epsilon_cells"] == 48
    assert summary["positive_epsilon_cells"] == 144


def test_artifact_surface_validator_rejects_missing_cell():
    cells = _surface()
    cells.pop()
    with pytest.raises(ValueError, match="192"):
        validate_calibration_surface(cells)


def test_artifact_surface_validator_rejects_duplicate_cell():
    cells = _surface()
    cells[-1] = copy.deepcopy(cells[0])
    with pytest.raises(ValueError, match="duplicate|grid"):
        validate_calibration_surface(cells)


def test_artifact_surface_validator_rejects_nonidentity_epsilon_zero():
    cells = _surface()
    zero = next(cell for cell in cells if cell["epsilon_kl"] == 0.0)
    zero["mean_realized_kl"] = 1e-3
    with pytest.raises(ValueError, match="epsilon=0"):
        validate_calibration_surface(cells)


def test_artifact_surface_validator_rejects_kl_budget_excess():
    cells = _surface()
    positive = next(cell for cell in cells if cell["epsilon_kl"] > 0.0)
    positive["max_kl_budget_excess"] = 1e-3
    with pytest.raises(ValueError, match="KL budget"):
        validate_calibration_surface(cells)
