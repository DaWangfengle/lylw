import copy

import pytest

from fhm_dapp_tim.fhm_dapp.v18_5_artifact_audit import (
    validate_calibration_surface,
    validate_joint_selection,
    validate_radio_diagnostics,
)
from fhm_dapp_tim.fhm_dapp.v18_5_core import select_joint_calibration
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



def _radio_diagnostics():
    records = []
    quant = {"p05": 0.1, "p25": 0.2, "p50": 0.3, "p75": 0.4, "p95": 0.5}
    unit_quant = {"p05": 1.0, "p25": 1.0, "p50": 1.0, "p75": 1.0, "p95": 1.0}
    zero_quant = {"p05": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    for temperature in map(float, V185_CONFIG["temperature_grid"]):
        eps_records = []
        for epsilon in map(float, V185_CONFIG["epsilon_kl_grid"]):
            value = 0.0 if epsilon == 0.0 else epsilon * 0.5
            eps_records.append(
                {
                    "epsilon_kl": epsilon,
                    "mean_effective_budget": value,
                    "effective_budget_quantiles": (
                        zero_quant if epsilon == 0.0 else {
                            key: value for key in zero_quant
                        }
                    ),
                    "mean_realized_kl": value,
                    "realized_kl_quantiles": (
                        zero_quant if epsilon == 0.0 else {
                            key: value for key in zero_quant
                        }
                    ),
                    "max_kl_budget_excess": 0.0,
                    "dual_beta_infinite_fraction": 0.0,
                    "finite_dual_beta_quantiles": (
                        zero_quant if epsilon == 0.0 else quant
                    ),
                    "maximum_finite_dual_beta": (
                        0.0 if epsilon == 0.0 else 0.6
                    ),
                }
            )
        records.append(
            {
                "temperature": temperature,
                "rows": 3,
                "training_reference_rows": 5,
                "training_reference_objective_quantiles": quant,
                "calibration_profiled_objective_quantiles": quant,
                "compatibility_mean": 0.5,
                "compatibility_quantiles": quant,
                "mean_abs_style_offset": 0.1,
                "abs_style_offset_quantiles": quant,
                "style_scale_mean": 1.0,
                "style_scale_quantiles": unit_quant,
                "q_temperature_entropy_mean": 2.0,
                "q_temperature_top1_mean": 0.3,
                "epsilon_diagnostics": eps_records,
            }
        )
    return records


def test_radio_diagnostic_validator_accepts_exact_8x4_grid():
    summary = validate_radio_diagnostics(
        _radio_diagnostics(),
        calibration_rows=3,
        selection_refit_rows=5,
    )
    assert summary["temperatures"] == 8
    assert summary["epsilon_records"] == 32


def test_radio_diagnostic_validator_rejects_missing_temperature():
    records = _radio_diagnostics()
    records.pop()
    with pytest.raises(ValueError, match="eight"):
        validate_radio_diagnostics(
            records,
            calibration_rows=3,
            selection_refit_rows=5,
        )


def test_radio_diagnostic_validator_rejects_epsilon_zero_nonidentity():
    records = _radio_diagnostics()
    zero = records[0]["epsilon_diagnostics"][0]
    zero["mean_realized_kl"] = 1e-3
    with pytest.raises(ValueError, match="epsilon=0"):
        validate_radio_diagnostics(
            records,
            calibration_rows=3,
            selection_refit_rows=5,
        )



def _selection_payload():
    surface = []
    selector_cells = []
    for temperature in map(float, V185_CONFIG["temperature_grid"]):
        for epsilon in map(float, V185_CONFIG["epsilon_kl_grid"]):
            for scale in map(float, V185_CONFIG["scale_grid"]):
                cell_id = f"T={temperature:.2f}|E={epsilon:.3f}|S={scale:.2f}"
                summary = {
                    "cell_id": cell_id,
                    "temperature": temperature,
                    "epsilon_kl": epsilon,
                    "scale_multiplier": scale,
                    "nll": 1.0,
                    "energy_score_m": 2.0,
                    "posterior_mean_mae_m": 3.0,
                }
                surface.append(summary)
                selector_cells.append({
                    **summary,
                    "group_nll": [1.0, 1.0, 1.0],
                    "group_energy_score": [2.0, 2.0, 2.0],
                })
    return {
        "surface": surface,
        "selection": select_joint_calibration(selector_cells),
    }


def test_joint_selection_artifact_validator_accepts_registered_parsimony():
    payload = _selection_payload()
    summary = validate_joint_selection(payload)
    assert summary["nll_one_se_size"] == 192
    assert summary["es_one_se_size"] == 192
    assert summary["selected_parameters"] == {
        "temperature": 1.0,
        "epsilon_kl": 0.0,
        "scale_multiplier": 1.0,
    }


def test_joint_selection_artifact_validator_rejects_retained_flag_tamper():
    payload = _selection_payload()
    gate = payload["selection"]["nll_one_se"]
    key = next(iter(gate["records"]))
    gate["records"][key]["retained"] = False
    with pytest.raises(ValueError, match="retained"):
        validate_joint_selection(payload)


def test_joint_selection_artifact_validator_rejects_final_choice_tamper():
    payload = _selection_payload()
    wrong = next(
        cell for cell in payload["surface"]
        if cell["temperature"] == 3.0
        and cell["epsilon_kl"] == 0.1
        and cell["scale_multiplier"] == 2.0
    )
    payload["selection"]["selected_cell_id"] = wrong["cell_id"]
    payload["selection"]["selected_cell"] = dict(wrong)
    with pytest.raises(ValueError, match="parsimony"):
        validate_joint_selection(payload)


def test_joint_selection_artifact_validator_rejects_surface_best_tamper():
    payload = _selection_payload()
    # Make one non-selected cell strictly better in NLL without changing
    # the stored one-SE gate.  The audit must re-derive the raw best.
    target = next(
        cell for cell in payload["surface"]
        if cell["temperature"] == 3.0
        and cell["epsilon_kl"] == 0.1
        and cell["scale_multiplier"] == 2.0
    )
    target["nll"] = 0.5
    with pytest.raises(ValueError, match="surface minimum|mean loss"):
        validate_joint_selection(payload)


def test_joint_selection_artifact_validator_rejects_mean_difference_tamper():
    payload = _selection_payload()
    gate = payload["selection"]["nll_one_se"]
    key = next(
        key for key in gate["records"]
        if key != gate["best"]
    )
    gate["records"][key]["mean_difference_vs_best"] = 0.25
    with pytest.raises(ValueError, match="mean difference"):
        validate_joint_selection(payload)
