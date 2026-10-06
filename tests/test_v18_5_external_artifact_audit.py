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


def _identity_result_metrics():
    return {
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


def _write_result_chain(tmp_path: Path):
    audit.external._set_seed(audit.external.MODEL_SEED)
    environment = audit.external.environment_fingerprint(
        model_seed=audit.external.MODEL_SEED,
        scoring_seed=audit.external.SCORING_SEED,
        predictive_draws=audit.external.PREDICTIVE_DRAWS,
    )
    environment_sha = audit.external._canonical_sha(environment)
    freeze = {
        "dataset": "UTSIndoorLoc",
        "heldout_device": None,
        "training_freeze_sha256": "train-freeze",
        "uji_method_freeze_sha256": "uji-freeze",
        "checkpoint_sha256": "checkpoint",
        "lambda_spatial": 0.25,
        "selected_epoch": 12,
        "posterior_temperature": 1.0,
        "epsilon_kl": 0.0,
        "scale_multiplier": 1.0,
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

    lock = {
        "protocol_version": audit.external.PROTOCOL_VERSION,
        "slug": "uts",
        "dataset": "UTSIndoorLoc",
        "heldout_device": None,
        "training_freeze_sha256": "train-freeze",
        "uji_method_freeze_sha256": "uji-freeze",
        "checkpoint_sha256": "checkpoint",
        "evaluation_device": "cpu",
        "evaluation_environment": environment,
        "evaluation_environment_sha256": environment_sha,
        "formal_test_performance_evaluation_started": True,
        "rerun_after_lock_forbidden": True,
    }
    lock["lock_sha256"] = audit.external._canonical_sha(lock)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = {
        "protocol_version": audit.external.PROTOCOL_VERSION,
        "slug": "uts",
        "dataset": "UTSIndoorLoc",
        "heldout_device": None,
        "training_freeze_sha256": "train-freeze",
        "uji_method_freeze_sha256": "uji-freeze",
        "test_consumption_lock_sha256": lock["lock_sha256"],
        "checkpoint_sha256": "checkpoint",
        "evaluation_device": "cpu",
        "evaluation_environment_sha256": environment_sha,
        "test_files": [],
        "metrics": _identity_result_metrics(),
        "method": {
            "lambda_spatial": 0.25,
            "selected_epoch": 12,
            "posterior_temperature": 1.0,
            "epsilon_kl": 0.0,
            "scale_multiplier": 1.0,
            "decoder": "posterior_mean_of_component_locations",
            "correction": "radio_only_exact_KL_projection",
        },
        "formal_test_performance_evaluated": True,
        "no_post_test_retuning_permitted": True,
    }
    result["result_sha256"] = audit.external._canonical_sha(result)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return freeze_path, lock_path, result_path, result


def test_external_result_chain_binds_evaluation_environment(tmp_path: Path):
    freeze_path, lock_path, result_path, _ = _write_result_chain(tmp_path)
    with patch.object(
        audit,
        "validate_external_training_chain",
        return_value={"status": "PASS"},
    ):
        summary = audit.validate_external_result_chain(
            "uts",
            training_freeze_path=freeze_path,
            lock_path=lock_path,
            result_path=result_path,
        )
    assert summary["status"] == "PASS"
    assert len(summary["evaluation_environment_sha256"]) == 64


def test_external_result_chain_rejects_environment_link_tamper(tmp_path: Path):
    freeze_path, lock_path, result_path, result = _write_result_chain(tmp_path)
    result["evaluation_environment_sha256"] = "0" * 64
    result["result_sha256"] = audit.external._canonical_sha(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with patch.object(
        audit,
        "validate_external_training_chain",
        return_value={"status": "PASS"},
    ):
        with pytest.raises(ValueError, match="evaluation_environment_sha256"):
            audit.validate_external_result_chain(
                "uts",
                training_freeze_path=freeze_path,
                lock_path=lock_path,
                result_path=result_path,
            )
