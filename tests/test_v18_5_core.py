import math

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.v18_5_core import (
    exact_kl_energy_projection,
    paired_one_se_set,
    posterior_mean,
    select_joint_calibration,
)


def test_zero_budget_preserves_normalized_input_bitwise():
    q = torch.tensor([[0.55, 0.25, 0.20, 0.0]], dtype=torch.float32)
    e = torch.tensor([[4.0, 1.0, 0.0, 9.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True, False]])
    result = exact_kl_energy_projection(q, e, epsilon_kl=0.0, mask=mask)
    assert torch.equal(result["posterior"], q)
    assert float(result["realized_kl"][0]) == 0.0


def test_constant_energy_is_identity_even_with_positive_budget():
    q = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    e = torch.ones_like(q) * 7.3
    result = exact_kl_energy_projection(q, e, epsilon_kl=0.4)
    assert torch.allclose(result["posterior"], q, atol=0, rtol=0)
    assert float(result["realized_kl"][0]) == 0.0


def test_large_budget_returns_exact_unique_minimum_limit():
    q = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
    e = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    result = exact_kl_energy_projection(q, e, epsilon_kl=3.0)
    expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    assert torch.equal(result["posterior"], expected)
    assert bool(result["dual_beta_infinite"][0])
    assert math.isclose(
        float(result["limiting_kl"][0]), -math.log(0.1), rel_tol=1e-12
    )


def test_large_budget_preserves_q_ratios_on_tied_minima():
    q = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
    e = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    result = exact_kl_energy_projection(q, e, epsilon_kl=1.0)
    expected = torch.tensor([[0.625, 0.375, 0.0]], dtype=torch.float64)
    assert torch.allclose(result["posterior"], expected, atol=1e-14, rtol=0)
    assert math.isclose(float(result["minimum_energy_mass"][0]), 0.8)


def test_active_budget_can_require_beta_greater_than_one():
    q = torch.tensor([[0.9, 0.1]], dtype=torch.float64)
    e = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    result = exact_kl_energy_projection(q, e, epsilon_kl=0.20)
    assert float(result["dual_beta"][0]) > 1.0
    assert abs(float(result["realized_kl"][0]) - 0.20) < 1e-9
    assert float(result["expected_energy_after"][0]) < 0.9


def test_energy_shift_and_positive_scale_invariance():
    q = torch.tensor([[0.55, 0.25, 0.12, 0.08]], dtype=torch.float64)
    e = torch.tensor([[0.0, 1.0, 3.0, 7.0]], dtype=torch.float64)
    first = exact_kl_energy_projection(q, e, epsilon_kl=0.08)
    shifted = exact_kl_energy_projection(q, e + 123.0, epsilon_kl=0.08)
    scaled = exact_kl_energy_projection(q, 5.0 * e, epsilon_kl=0.08)
    assert torch.allclose(first["posterior"], shifted["posterior"], atol=1e-12)
    assert torch.allclose(first["posterior"], scaled["posterior"], atol=1e-12)
    assert math.isclose(
        float(first["dual_beta"][0]) / 5.0,
        float(scaled["dual_beta"][0]),
        rel_tol=1e-9,
    )


def test_budget_monotonicity_and_expected_energy():
    q = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64)
    e = torch.tensor([[3.0, 2.0, 1.0, 0.0]], dtype=torch.float64)
    budgets = [0.01, 0.05, 0.15]
    results = [
        exact_kl_energy_projection(q, e, epsilon_kl=value)
        for value in budgets
    ]
    kls = [float(value["realized_kl"][0]) for value in results]
    energies = [float(value["expected_energy_after"][0]) for value in results]
    assert kls == sorted(kls)
    assert energies == sorted(energies, reverse=True)


def test_padding_and_q_zero_cannot_gain_mass():
    q = torch.tensor([[0.7, 0.3, 0.0, 0.0]], dtype=torch.float64)
    e = torch.tensor([[2.0, 1.0, -100.0, -1000.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False]])
    result = exact_kl_energy_projection(q, e, epsilon_kl=0.1, mask=mask)
    assert float(result["posterior"][0, 2]) == 0.0
    assert float(result["posterior"][0, 3]) == 0.0


def test_posterior_mean_exact():
    p = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    loc = torch.tensor([[[0.0, 0.0], [4.0, 8.0]]], dtype=torch.float64)
    mask = torch.ones_like(p, dtype=torch.bool)
    result = posterior_mean(p, loc, mask)
    assert torch.allclose(result, torch.tensor([[3.0, 6.0]], dtype=torch.float64))


def test_paired_one_se_gate_keeps_statistically_near_cell():
    losses = {
        "a": np.array([1.00, 1.10, 0.90, 1.00]),
        "b": np.array([1.01, 1.09, 0.91, 1.01]),
        "c": np.array([1.30, 1.30, 1.30, 1.30]),
    }
    result = paired_one_se_set(losses)
    assert result["best"] == "a"
    assert "a" in result["retained"]
    assert "b" in result["retained"]
    assert "c" not in result["retained"]


def test_joint_selector_uses_parsimony_after_two_one_se_gates():
    groups = [1.0, 1.0, 1.0, 1.0]
    cells = [
        {
            "cell_id": "complex",
            "group_nll": groups,
            "group_energy_score": groups,
            "posterior_mean_mae_m": 5.0,
            "epsilon_kl": 0.1,
            "temperature": 1.5,
            "scale_multiplier": 1.25,
        },
        {
            "cell_id": "simple",
            "group_nll": groups,
            "group_energy_score": groups,
            "posterior_mean_mae_m": 5.0,
            "epsilon_kl": 0.0,
            "temperature": 1.0,
            "scale_multiplier": 1.0,
        },
    ]
    result = select_joint_calibration(cells)
    assert result["selected_cell_id"] == "simple"
