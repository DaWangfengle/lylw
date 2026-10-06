from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fhm_dapp_tim.fhm_dapp.v18_5_core import select_joint_calibration
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG
from fhm_dapp_tim.fhm_dapp import v18_5_external_artifact_audit as audit


def _calibration_fixture():
    selector_cells = []
    surface = []
    selection_vectors = {}
    group_order = [[0, 0, 0.0, 0.0], [0, 0, 1.0, 0.0], [0, 0, 2.0, 0.0]]
    for temperature in map(float, V185_CONFIG["temperature_grid"]):
        for epsilon in map(float, V185_CONFIG["epsilon_kl_grid"]):
            for scale in map(float, V185_CONFIG["scale_grid"]):
                cell_id = f"T={temperature:.2f}|E={epsilon:.3f}|S={scale:.2f}"
                group_nll = [1.0, 1.0, 1.0]
                group_es = [2.0, 2.0, 2.0]
                summary = {
                    "cell_id": cell_id,
                    "temperature": temperature,
                    "epsilon_kl": epsilon,
                    "scale_multiplier": scale,
                    "nll": 1.0,
                    "nll_se": 0.0,
                    "energy_score_m": 2.0,
                    "energy_score_se_m": 0.0,
                    "posterior_mean_mae_m": 3.0,
                }
                surface.append(summary)
                selection_vectors[cell_id] = {
                    "group_nll": group_nll,
                    "group_energy_score": group_es,
                }
                selector_cells.append({
                    **summary,
                    "group_nll": group_nll,
                    "group_energy_score": group_es,
                })
    return {
        "surface": surface,
        "coordinate_groups": len(group_order),
        "selection_group_order": group_order,
        "selection_vectors": selection_vectors,
        "selection": select_joint_calibration(selector_cells),
    }


def test_external_selection_audit_rederives_one_se_and_parsimony():
    calibration = _calibration_fixture()
    result = audit.validate_selection(calibration)
    assert result["nll_one_se_size"] == 192
    assert result["es_one_se_size"] == 192
    assert result["selected_parameters"] == {
        "temperature": 1.0,
        "epsilon_kl": 0.0,
        "scale_multiplier": 1.0,
    }


def test_external_selection_audit_rejects_tampered_final_choice():
    calibration = _calibration_fixture()
    tampered = copy.deepcopy(calibration)
    wrong = next(
        cell for cell in tampered["surface"]
        if cell["epsilon_kl"] == 0.1 and cell["temperature"] == 3.0
    )
    tampered["selection"]["selected_cell_id"] = wrong["cell_id"]
    tampered["selection"]["selected_cell"] = dict(wrong)
    with pytest.raises(ValueError, match="tie-break"):
        audit.validate_selection(tampered)


def test_external_selection_audit_rejects_tampered_one_se_flag():
    calibration = _calibration_fixture()
    tampered = copy.deepcopy(calibration)
    gate = tampered["selection"]["nll_one_se"]
    key = next(iter(gate["records"]))
    gate["records"][key]["retained"] = False
    with pytest.raises(ValueError, match="retained list"):
        audit.validate_selection(tampered)


def test_result_metric_audit_accepts_identity_projection():
    metrics = {
        "rows": 10,
        "coordinate_groups": 5,
        "posterior_mean_coordinate_macro_mae_m": 4.0,
        "posterior_mean_coordinate_macro_mae_se_m": 0.2,
        "row_mae_m": 4.1,
        "row_median_error_m": 3.5,
        "row_p75_error_m": 5.0,
        "row_p90_error_m": 7.0,
        "energy_score_coordinate_macro_m": 3.0,
        "energy_score_coordinate_macro_se_m": 0.1,
        "mixture_nll_coordinate_macro": 5.0,
        "mixture_nll_coordinate_macro_se": 0.2,
        "mean_entropy_nats": 1.5,
        "mean_effective_anchors": 4.5,
        "mean_top1_probability": 0.4,
        "mean_support_size": 50.0,
        "natural_truncation_fraction": 0.0,
        "coverage": {
            level: {
                "row_coverage": 0.5,
                "coordinate_macro_coverage": 0.5,
                "mean_radius_m": 10.0,
                "median_radius_m": 9.0,
            }
            for level in ("50", "80", "90", "95")
        },
        "radio_projection": {
            "active": False,
            "mean_radio_compatibility": None,
            "mean_realized_kl": 0.0,
            "mean_effective_epsilon_kl": 0.0,
            "max_kl_budget_excess": 0.0,
            "mean_finite_dual_beta": 0.0,
            "dual_beta_infinite_fraction": 0.0,
            "mean_abs_style_offset": None,
            "mean_style_scale": None,
        },
    }
    result = audit._validate_test_metrics(metrics, epsilon_kl=0.0)
    assert result["rows"] == 10
    assert result["mae_m"] == 4.0


def test_result_metric_audit_rejects_kl_budget_violation():
    metrics = {
        "rows": 1,
        "coordinate_groups": 1,
        "posterior_mean_coordinate_macro_mae_m": 1.0,
        "posterior_mean_coordinate_macro_mae_se_m": 0.0,
        "row_mae_m": 1.0,
        "row_median_error_m": 1.0,
        "row_p75_error_m": 1.0,
        "row_p90_error_m": 1.0,
        "energy_score_coordinate_macro_m": 1.0,
        "energy_score_coordinate_macro_se_m": 0.0,
        "mixture_nll_coordinate_macro": 1.0,
        "mixture_nll_coordinate_macro_se": 0.0,
        "mean_entropy_nats": 1.0,
        "mean_effective_anchors": 2.0,
        "mean_top1_probability": 0.5,
        "mean_support_size": 2.0,
        "natural_truncation_fraction": 0.0,
        "coverage": {
            level: {
                "row_coverage": 1.0,
                "coordinate_macro_coverage": 1.0,
                "mean_radius_m": 1.0,
                "median_radius_m": 1.0,
            }
            for level in ("50", "80", "90", "95")
        },
        "radio_projection": {
            "active": True,
            "mean_radio_compatibility": 0.5,
            "mean_realized_kl": 0.02,
            "mean_effective_epsilon_kl": 0.01,
            "max_kl_budget_excess": 0.01,
            "mean_finite_dual_beta": 1.0,
            "dual_beta_infinite_fraction": 0.0,
            "mean_abs_style_offset": 0.0,
            "mean_style_scale": 1.0,
        },
    }
    with pytest.raises(ValueError, match="KL budget"):
        audit._validate_test_metrics(metrics, epsilon_kl=0.025)
