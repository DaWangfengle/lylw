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


@pytest.fixture(scope="session", autouse=True)
def _isolated_legacy_r3_design_fixture(tmp_path_factory):
    """Supply only the missing legacy test document within the G1 temp area.

    This is an explicitly labelled test fixture, not recovered preregistration.
    The frozen legacy source is included as the authority for the estimand.
    """
    if run.r3p.DEFAULT_DESIGN.is_file():
        yield run.r3p.DEFAULT_DESIGN
        return
    fixture = (
        tmp_path_factory.mktemp("g1_legacy_r3_test_document")
        / "R3_EXPECTED_LOSS_TEST_FIXTURE.md"
    )
    fixture.write_text(
        "# G1-owned isolated fixture for the missing legacy R3 test document\n\n"
        "This is a test dependency, not the original R3 design or evidence.\n"
        "Losses, not predictive distributions, are averaged across the fixed "
        "25 overlapping balanced E16 subsets. The production scoring source "
        "below remains unchanged.\n\n"
        + inspect.getsource(run.r3._subset_run_records)
        + "\n",
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run.r3p, "DEFAULT_DESIGN", fixture)
        yield fixture


def test_legacy_test_fixture_retains_original_subset_loss_estimand(
    _isolated_legacy_r3_design_fixture,
):
    # Average forecast would have zero error here, whereas expected loss is 2m.
    means = torch.tensor([[[-2.0, 0.0]], [[2.0, 0.0]]], dtype=torch.float64)
    target = torch.zeros((1, 2), dtype=torch.float64)
    per_subset_mae = torch.linalg.vector_norm(means - target, dim=-1)
    energy = per_subset_mae * 0.5
    subsets = ((0,), (1,))
    records = run.r3._subset_run_records(
        energy, per_subset_mae, subsets, torch.tensor([0]), 1
    )
    assert torch.linalg.vector_norm(means.mean(dim=0) - target).item() == 0.0
    assert [r["exact_mixture_mean_mae_coordinate_macro_m"] for r in records] == [2.0, 2.0]
    assert "Losses, not predictive distributions" in (
        _isolated_legacy_r3_design_fixture.read_text(encoding="utf-8")
    )


def test_new_selection_refit_checkpoint_roundtrip_and_validated_resume(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    source_rows = np.array([2, 7, 11], dtype=np.int64)
    selection_keys = {(0, 0, float(i), float(i + 1)) for i in range(595)}
    registry = {"registry_sha256": "fixture_registry", "roles": {}}
    checkpoint_path = tmp_path / "g1_member_01.pt"
    refit_csv = tmp_path / "selection_refit.csv"
    refit_csv.write_text("G1 synthetic serialization fixture\n", encoding="utf-8")
    calls = []
    fake_model = SimpleNamespace(
        n_anchors=595,
        state_dict=lambda: {"weight": torch.tensor([1.25], dtype=torch.float32)},
    )

    def fake_refit(data, rows, *, seed, epochs, device, log_every):
        assert np.array_equal(rows, source_rows)
        assert seed == 17301 and epochs == 65 and device.type == "cpu"
        calls.append((seed, epochs))
        return (
            fake_model,
            {"n_aps": 1, "array_stat": np.array([1.0], dtype=np.float32)},
            np.array([7.0, 13.0], dtype=np.float32),
            {5: 0},
            [{"epoch": epoch} for epoch in range(1, epochs + 1)],
        )

    monkeypatch.setattr(run, "prepare", lambda device_name: registry)
    monkeypatch.setattr(run, "member_path", lambda index: checkpoint_path)
    monkeypatch.setattr(run, "REFIT_CSV", refit_csv)
    monkeypatch.setattr(run, "load_uji_v16", lambda path: {"fixture": True})
    monkeypatch.setattr(run, "_read_hashed", lambda path, field: registry)
    monkeypatch.setattr(
        run, "_roles",
        lambda data, historical: (
            {"selection_refit": selection_keys},
            {"selection_refit": source_rows},
        ),
    )
    monkeypatch.setattr(
        run, "r0", SimpleNamespace(_fit_refit=fake_refit, PATIENCE=run.r0.PATIENCE)
    )
    first = run.train_member(1, "cpu")
    assert first["resumed"] is False
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["phase"] == "g1_selection_refit"
    assert saved["epoch"] == len(saved["refit_history"]) == 65
    assert torch.equal(saved["coord_origin"], torch.tensor([7.0, 13.0]))
    assert len(saved["anchor_coordinate_keys"]) == 595
    assert all(value is False for key, value in saved["split_contract"].items()
               if key != "training_roles")
    second = run.train_member(1, "cpu")
    assert second["resumed"] is True
    assert second["sha256"] == first["sha256"]
    assert calls == [(17301, 65)]
