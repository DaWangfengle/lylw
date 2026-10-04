from pathlib import Path

from fhm_dapp_tim.fhm_dapp.uji_v18_5_development import _select_global_lambda
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG


def _report(lam, es, nll, mae):
    return {
        "lambda_spatial": lam,
        "selected_epoch": 12,
        "selected_base_metrics": {
            "energy_score_m": es,
            "nll": nll,
            "posterior_mean_mae_m": mae,
        },
    }


def test_global_lambda_selector_uses_es_shortlist_then_nll_mean_and_smaller_lambda():
    reports = [
        _report(0.0, 5.00, 6.3, 7.0),
        _report(0.1, 5.03, 6.1, 7.1),
        _report(0.25, 5.04, 6.1, 6.9),
        _report(0.5, 5.20, 5.0, 5.0),
    ]
    selected = _select_global_lambda(reports)
    assert selected["lambda_spatial"] == 0.25


def test_joint_surface_is_exactly_192_cells_and_contains_identity():
    assert (
        len(V185_CONFIG["temperature_grid"])
        * len(V185_CONFIG["epsilon_kl_grid"])
        * len(V185_CONFIG["scale_grid"])
        == 192
    )
    assert 0.0 in V185_CONFIG["epsilon_kl_grid"]


def test_v185_driver_has_no_gm_or_geometry_correction_selection_path():
    root = Path(__file__).resolve().parents[1]
    driver = (
        root / "fhm_dapp_tim" / "fhm_dapp" / "uji_v18_5_development.py"
    ).read_text(encoding="utf-8")
    model = (
        root / "fhm_dapp_tim" / "fhm_dapp" / "v18_5_model.py"
    ).read_text(encoding="utf-8")
    assert "weighted_component_geometric_median" not in driver
    assert "_masked_geometry_energy" not in model
    assert "geometry_reliability" not in model
    assert "posterior_mean" in driver


def test_development_does_not_select_epsilon():
    root = Path(__file__).resolve().parents[1]
    driver = (
        root / "fhm_dapp_tim" / "fhm_dapp" / "uji_v18_5_development.py"
    ).read_text(encoding="utf-8")
    select_block = driver.split("def _select_global_lambda", 1)[1].split(
        "def develop", 1
    )[0]
    assert "epsilon" not in select_block
