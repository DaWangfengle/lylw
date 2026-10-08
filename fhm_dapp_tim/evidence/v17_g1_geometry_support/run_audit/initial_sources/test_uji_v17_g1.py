"""Pure and contract-level regression checks for V17-G1. No dataset/GPU required."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1
from fhm_dapp_tim.fhm_dapp import uji_v17_g1_experiment as run


def keys_fixture():
    return {
        (building, floor, float(x), float(x * 3))
        for building in (0, 1)
        for floor in (0, 1)
        for x in range(11)
    }


def test_folds_deterministic_stratified_disjoint():
    all_keys = keys_fixture()
    first = g1.split_development_keys(all_keys)
    second = g1.split_development_keys(set(reversed(sorted(all_keys))))
    assert first == second
    a, b = first
    assert a and b and not a.intersection(b)
    assert a | b == all_keys
    for building in (0, 1):
        for floor in (0, 1):
            assert any(k[:2] == (building, floor) for k in a)
            assert any(k[:2] == (building, floor) for k in b)


def test_fold_changes_do_not_depend_on_order():
    all_keys = keys_fixture()
    a, b = g1.split_development_keys(all_keys)
    shuffled = list(all_keys)
    np.random.default_rng(41).shuffle(shuffled)
    assert g1.split_development_keys(set(shuffled)) == (a, b)


@pytest.mark.parametrize("bad", [set(), {(0, 0, 0.0)}, {(0, 0, 0.0, 0.0)}])
def test_invalid_development_keys_fail_closed(bad):
    with pytest.raises(ValueError):
        g1.split_development_keys(bad)


def test_density_excludes_self_and_separates_strata():
    xy = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [2., 0.]])
    building = torch.tensor([0, 0, 1, 0])
    floor = torch.tensor([0, 0, 0, 1])
    result = g1.support_log_density(xy, building, floor, radius_m=1.1)
    assert torch.allclose(result, torch.tensor([math.log(2), math.log(2), 0, 0],
                                               dtype=torch.float64))


def test_density_identical_points_in_different_slots_count():
    xy = torch.tensor([[0., 0.], [0., 0.]])
    result = g1.support_log_density(xy, torch.zeros(2), torch.zeros(2))
    assert torch.allclose(result, torch.full((2,), math.log(2), dtype=torch.float64))


def test_density_invalid_inputs():
    with pytest.raises(ValueError):
        g1.support_log_density(torch.zeros(3, 3), torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError):
        g1.support_log_density(torch.zeros(2, 2), torch.zeros(2), torch.zeros(2), radius_m=0)


def test_gamma_zero_is_exact_historical_atomic_prior_identity():
    p = torch.tensor([[[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]]], dtype=torch.float32)
    d = torch.tensor([1., 2., 4.])
    assert g1.corrected_probability(p, d, 0.0) is p
    assert torch.equal(g1.corrected_probability(p, d, 0.0), p)


def test_inverse_density_is_proper_and_penalizes_dense_support():
    prior = torch.tensor([[[0.5, 0.5], [0.25, 0.75]]], dtype=torch.float32)
    corrected = g1.corrected_probability(prior, torch.tensor([math.log(8), 0.]), 1.0)
    assert torch.allclose(corrected.sum(dim=-1), torch.ones((1, 2)), atol=1e-7)
    assert corrected[0, 0, 0] < prior[0, 0, 0]
    assert corrected[0, 0, 1] > prior[0, 0, 1]


@pytest.mark.parametrize("gamma", [-1.0, float("nan"), float("inf")])
def test_invalid_gamma_fails_closed(gamma):
    with pytest.raises(ValueError):
        g1.corrected_probability(torch.ones((2, 2)), torch.ones(2), gamma)


def test_bad_posterior_fails_closed():
    with pytest.raises(ValueError):
        g1.corrected_probability(torch.tensor([[1., -1.]]), torch.zeros(2), 0.3)
    with pytest.raises(ValueError):
        g1.corrected_probability(torch.tensor([[1., 1.]]), torch.zeros(3), 0.3)


def test_placebo_permutation_preserves_density_distribution_within_strata():
    d = torch.tensor([0., 2., 4., 6., 1., 3., 5., 7.], dtype=torch.float64)
    b = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    f = torch.zeros(8)
    shuffled = g1.placebo_log_density(d, b, f)
    assert torch.equal(shuffled, g1.placebo_log_density(d, b, f))
    for bb in (0, 1):
        assert sorted(shuffled[b == bb].tolist()) == sorted(d[b == bb].tolist())


def test_calibration_choice_uses_only_registered_metrics_and_energy_guard():
    values = {
        "gamma_0": {"mae_2d_m": 6.0, "energy_score_2d_m": 5.0},
        "gamma_0p35": {"mae_2d_m": 5.5, "energy_score_2d_m": 5.03},
        "gamma_0p7": {"mae_2d_m": 5.7, "energy_score_2d_m": 5.01},
    }
    assert g1.choose_gamma(values) == 0.7
    values["gamma_0p35"]["energy_score_2d_m"] = 5.0
    assert g1.choose_gamma(values) == 0.35


def test_gate_conjunctive_and_placebo_matters():
    success = g1.confirm_gate(-0.20, -0.02, -0.1, -0.001, -0.05)
    assert success["passed"]
    assert not g1.confirm_gate(-0.20, -0.02, -0.1, 0.01, -0.05)["passed"]
    assert not g1.confirm_gate(-0.20, -0.02, -0.1, -0.001, -0.40)["passed"]


def test_immutable_seeds_epochs_and_r3_contract():
    assert len(g1.SEEDS) == len(g1.EPOCHS) == 25
    assert g1.SEEDS == tuple(range(17301, 17326))
    assert sum(g1.EPOCHS) > 1000
    assert g1.GAMMAS[0] == 0
    assert run.TRAIN_DATA.name == "trainingData.csv"


def test_no_cross_stage_predictive_shortcuts():
    training = inspect.getsource(run.train_member)
    evaluation = inspect.getsource(run._evaluation_rows)
    assert 'rows["selection_refit"]' in training
    assert 'rows["outer_refit"]' not in training
    assert 'rows["outer_test"]' not in training
    assert 'roles["outer_test"]' not in evaluation
    assert "validationData.csv" not in inspect.getsource(run)


def test_calibration_and_confirmation_are_separate_commands():
    calibration = inspect.getsource(run.calibrate)
    confirmation = inspect.getsource(run.confirm)
    assert '_evaluate("calibration"' in calibration
    assert '_evaluate("confirmation"' in confirmation
    assert "G1Error" in calibration and "G1Error" in confirmation
