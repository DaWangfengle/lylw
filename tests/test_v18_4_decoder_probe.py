from __future__ import annotations

import torch

from fhm_dapp_tim.fhm_dapp.v18_4_decoder_probe import (
    DECODERS,
    _meets_mechanism_rule,
    decode_all,
    map_component,
    mechanism_decision,
    posterior_mean,
    topk_mean,
    weighted_geometric_median,
)


def test_decoder_probe_has_exactly_four_preregistered_decoders() -> None:
    assert DECODERS == (
        "posterior_mean",
        "map_component",
        "top8_mean",
        "geometric_median",
    )


def test_posterior_mean_and_map_component() -> None:
    weights = torch.tensor([[0.2, 0.7, 0.1]], dtype=torch.float32)
    locations = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]]],
        dtype=torch.float32,
    )
    mean = posterior_mean(weights, locations)
    map_point = map_component(weights, locations)
    assert torch.allclose(mean, torch.tensor([[2.4, 0.0]]), atol=1e-6)
    assert torch.allclose(map_point, torch.tensor([[2.0, 0.0]]), atol=1e-6)


def test_topk_mean_renormalizes_selected_mass() -> None:
    weights = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32)
    locations = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [100.0, 0.0]]],
        dtype=torch.float32,
    )
    point = topk_mean(weights, locations, k=2)
    expected = torch.tensor([[2.0 / 3.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(point, expected, atol=1e-6)


def test_geometric_median_resists_low_mass_far_outlier() -> None:
    weights = torch.tensor([[0.45, 0.45, 0.10]], dtype=torch.float32)
    locations = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]]],
        dtype=torch.float32,
    )
    mean = posterior_mean(weights, locations)
    median = weighted_geometric_median(weights, locations)
    assert float(mean[0, 0]) > 10.0
    assert abs(float(median[0, 0]) - 1.0) < 1e-3


def test_geometric_median_returns_symmetric_center() -> None:
    weights = torch.full((1, 4), 0.25)
    locations = torch.tensor(
        [[[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    median = weighted_geometric_median(weights, locations)
    assert torch.allclose(median, torch.zeros_like(median), atol=1e-5)


def test_decode_all_outputs_finite_points() -> None:
    weights = torch.tensor(
        [[0.40, 0.30, 0.20, 0.10]],
        dtype=torch.float32,
    )
    locations = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [5.0, 1.0], [20.0, 0.0]]],
        dtype=torch.float32,
    )
    decoded = decode_all(weights, locations)
    assert set(decoded) == set(DECODERS)
    assert all(bool(torch.isfinite(value).all()) for value in decoded.values())


def _role(mean_mae: float, gm_mae: float, top8_mae: float) -> dict:
    return {
        "decoder_metrics": {
            "posterior_mean": {"coordinate_macro_mae_m": mean_mae},
            "map_component": {"coordinate_macro_mae_m": mean_mae},
            "top8_mean": {"coordinate_macro_mae_m": top8_mae},
            "geometric_median": {"coordinate_macro_mae_m": gm_mae},
        }
    }


def test_mechanism_rule_requires_improvement_on_both_roles() -> None:
    results = {
        "early_stop": _role(10.0, 8.5, 10.0),
        "calibration": _role(10.0, 9.5, 10.0),
    }
    assert not _meets_mechanism_rule(results, "geometric_median")


def test_mechanism_decision_promotes_geometric_median_only_by_fixed_rule() -> None:
    results = {
        "early_stop": _role(10.0, 8.5, 8.8),
        "calibration": _role(12.0, 10.0, 10.7),
    }
    decision = mechanism_decision(results)
    assert decision["geometric_median_supported"] is True
    assert decision["conclusion"] == "loss_aligned_geometric_median_supported"


def test_mechanism_decision_can_identify_tail_dilution_without_gm() -> None:
    results = {
        "early_stop": _role(10.0, 9.4, 8.5),
        "calibration": _role(12.0, 11.0, 10.5),
    }
    decision = mechanism_decision(results)
    assert decision["geometric_median_supported"] is False
    assert decision["top8_tail_contraction_supported"] is True
    assert decision["conclusion"] == "posterior_tail_dilution_supported"


def test_mechanism_decision_falls_back_to_posterior_learning() -> None:
    results = {
        "early_stop": _role(10.0, 9.5, 9.4),
        "calibration": _role(12.0, 11.5, 11.0),
    }
    decision = mechanism_decision(results)
    assert decision["geometric_median_supported"] is False
    assert decision["top8_tail_contraction_supported"] is False
    assert decision["conclusion"] == "posterior_learning_dominates"
