from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import pytest

from fhm_dapp_tim.fhm_dapp.uji_v18_4_development import (
    CALIBRATION_SCALE_GRID,
    CALIBRATION_TEMPERATURE_GRID,
    ENERGY_SHORTLIST_MARGIN_M,
    _canonical_sha,
    _assert_source_revision,
    _cache_corrected_distribution,
    _calibration_nll,
    _coordinate_macro,
    _git_source_state,
    _pad_concat_support,
    load_registry,
    prepare,
    select_method_record,
)


@pytest.fixture
def clean_source_fixture():
    # Registry fixtures exercise role/hash serialization without requiring a
    # clean developer checkout. Production dirty/source guards are tested
    # separately; no production guard is disabled by this fixture.
    with patch('fhm_dapp_tim.fhm_dapp.uji_v18_4_development._git_source_state',
               return_value={'source_commit_sha': 'f' * 40, 'tracked_source_dirty': False}):
        yield


def test_v184_source_state_checks_tracked_tree_against_head() -> None:
    with patch(
        "fhm_dapp_tim.fhm_dapp.uji_v18_4_development.subprocess.check_output",
        return_value="a" * 40,
    ), patch(
        "fhm_dapp_tim.fhm_dapp.uji_v18_4_development.subprocess.run",
        return_value=SimpleNamespace(returncode=1),
    ) as run:
        try:
            _git_source_state()
        except RuntimeError as exc:
            assert "tracked working-tree modifications" in str(exc)
        else:
            raise AssertionError("tracked source difference must fail closed")
        assert run.call_args.args[0] == [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
        ]


def test_v184_prepare_freezes_fresh_uji_roles(tmp_path: Path, clean_source_fixture) -> None:
    registry_path = tmp_path / "registry.json"
    registry = prepare(registry_path=registry_path)

    assert registry["uji_official_validation_preexposed"] is True
    assert registry["uji_official_validation_used_for_v184_selection"] is False
    assert (
        registry["external_provided_tests_preexposed_for_adapter_validation"]
        is True
    )
    assert registry["external_test_performance_used_for_v184_selection"] is False
    assert len(registry["source_commit_sha"]) == 40
    assert registry["tracked_source_dirty"] is False
    assert Path(registry["train_path"]).as_posix() == "data/UJIndoorLoc/trainingData.csv"
    assert not Path(registry["train_path"]).is_absolute()
    assert registry["training_rows"] == 19937
    assert registry["training_coordinate_groups"] == 933
    assert registry["aps"] == 520
    assert registry["roles"]["fit"]["row_count"] == 14126
    assert registry["roles"]["development"]["row_count"] == 2936
    assert registry["roles"]["selection_refit"]["row_count"] == 17062
    assert registry["roles"]["calibration"]["row_count"] == 2875
    assert registry["roles"]["fit"]["coordinate_count"] == 655
    assert registry["roles"]["development"]["coordinate_count"] == 139
    assert registry["roles"]["calibration"]["coordinate_count"] == 139

    loaded = load_registry(registry_path)
    assert loaded["registry_sha256"] == registry["registry_sha256"]


def test_v184_registry_hash_fails_closed(tmp_path: Path, clean_source_fixture) -> None:
    registry_path = tmp_path / "registry.json"
    registry = prepare(registry_path=registry_path)
    registry["aps"] = 519
    registry_path.write_text(
        json.dumps(registry, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        load_registry(registry_path)
    except ValueError as exc:
        assert "self-hash" in str(exc)
    else:
        raise AssertionError("tampered registry must fail closed")



def test_v184_variable_width_support_padding_is_inert() -> None:
    first = torch.tensor(
        [[0.7, 0.3], [1.0, 0.0]],
        dtype=torch.float32,
    )
    second = torch.tensor(
        [[0.5, 0.3, 0.2, 0.0]],
        dtype=torch.float32,
    )
    probability = _pad_concat_support(
        [first, second],
        fill_value=0.0,
    )
    assert probability.shape == (3, 4)
    assert torch.allclose(
        probability.sum(dim=1),
        torch.ones(3),
    )
    assert torch.all(probability[:2, 2:] == 0.0)

    mask = _pad_concat_support(
        [
            torch.tensor([[True, True], [True, False]]),
            torch.tensor([[True, True, True, False]]),
        ],
        fill_value=False,
    )
    assert mask.shape == (3, 4)
    assert not bool(mask[0, 2:].any())
    assert not bool(mask[1, 1:].any())

    location = _pad_concat_support(
        [
            torch.zeros(2, 2, 2),
            torch.ones(1, 4, 2),
        ],
        fill_value=0.0,
    )
    assert location.shape == (3, 4, 2)
    assert torch.all(location[:2, 2:] == 0.0)


def test_v184_method_selection_obeys_energy_shortlist_then_gm() -> None:
    assert ENERGY_SHORTLIST_MARGIN_M == 0.05
    records = [
        {
            "lambda_spatial": 0.10,
            "epsilon_kl": 0.025,
            "energy_score_m": 5.000,
            "gm_mae_m": 6.0,
            "nll": 6.0,
        },
        {
            "lambda_spatial": 0.25,
            "epsilon_kl": 0.050,
            "energy_score_m": 5.040,
            "gm_mae_m": 5.0,
            "nll": 6.2,
        },
        {
            # Best GM but outside the 0.05 m Energy shortlist.
            "lambda_spatial": 0.50,
            "epsilon_kl": 0.100,
            "energy_score_m": 5.051,
            "gm_mae_m": 1.0,
            "nll": 1.0,
        },
    ]
    selected = select_method_record(records)
    assert selected["lambda_spatial"] == 0.25
    assert selected["epsilon_kl"] == 0.050


def test_v184_method_selection_ties_prefer_smaller_registered_values() -> None:
    records = [
        {
            "lambda_spatial": 0.25,
            "epsilon_kl": 0.050,
            "energy_score_m": 5.0,
            "gm_mae_m": 5.0,
            "nll": 5.0,
        },
        {
            "lambda_spatial": 0.10,
            "epsilon_kl": 0.100,
            "energy_score_m": 5.0,
            "gm_mae_m": 5.0,
            "nll": 5.0,
        },
        {
            "lambda_spatial": 0.10,
            "epsilon_kl": 0.025,
            "energy_score_m": 5.0,
            "gm_mae_m": 5.0,
            "nll": 5.0,
        },
    ]
    selected = select_method_record(records)
    assert selected["lambda_spatial"] == 0.10
    assert selected["epsilon_kl"] == 0.025


def test_coordinate_macro_weights_physical_groups_equally() -> None:
    values = torch.tensor([1.0, 1.0, 9.0], dtype=torch.float32)
    keys = [
        (0, 0, 0.0, 0.0),
        (0, 0, 0.0, 0.0),
        (0, 0, 1.0, 0.0),
    ]
    # Group means are 1 and 9, so macro average is 5, not row mean 11/3.
    assert abs(_coordinate_macro(values, keys) - 5.0) < 1e-12


def test_v184_calibration_grid_is_frozen_and_symmetric_around_one() -> None:
    assert CALIBRATION_TEMPERATURE_GRID == (
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
        2.50,
        3.00,
    )
    assert CALIBRATION_SCALE_GRID == (
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
    )


def test_v184_prepared_source_revision_mismatch_still_fails_closed() -> None:
    with patch('fhm_dapp_tim.fhm_dapp.uji_v18_4_development._git_source_state',
               return_value={'source_commit_sha': 'a' * 40, 'tracked_source_dirty': False}):
        with pytest.raises(RuntimeError, match='differs'):
            _assert_source_revision({'source_commit_sha': 'b' * 40})


def test_v184_calibration_cache_retains_final_corrected_probability() -> None:
    from fhm_dapp_tim.fhm_dapp.v18_4_core import temperature_base_posterior, reliability_bounded_posterior_correction
    q = torch.tensor([[.70, .25, .05]], dtype=torch.float64).expand(3, -1)
    mask = torch.ones_like(q, dtype=torch.bool)
    q_T = temperature_base_posterior(q, mask, temperature=.75)
    corrected = reliability_bounded_posterior_correction(q_T, torch.tensor([[5., 1., 0.]]).expand(3, -1),
        torch.tensor([[5., 1., 0.]]).expand(3, -1), radio_reliability=torch.ones(3),
        geometry_reliability=torch.ones(3), epsilon_kl=.05, mask=mask)
    fake_distribution = {
        'candidate_weights': corrected['posterior'], 'calibrated_base_probability': q_T,
        'correction_realized_kl': corrected['realized_kl'],
        'correction_epsilon_kl_effective': corrected['epsilon_kl_effective'], 'candidate_mask': mask,
        'component_locations': torch.zeros(3, 3, 2, dtype=q.dtype),
        'component_scales': torch.ones(3, 3, 2, dtype=q.dtype),
        'natural_pool_size_before_ceiling': torch.full((3,), 3), 'natural_truncated': torch.zeros(3, dtype=torch.bool),
    }
    model = SimpleNamespace(eval=lambda: None,
        student_log_prob=lambda target, location, scale: torch.tensor([[0., -2., -5.]], dtype=q.dtype).expand(3, -1))
    with patch('fhm_dapp_tim.fhm_dapp.uji_v18_4_development.v184_distribution', return_value=fake_distribution) as distribution:
        cache = _cache_corrected_distribution(model=model, reference=None, epsilon_kl=.05,
            rss_all=torch.zeros(3, 2), row_indices=np.arange(3), device=torch.device('cpu'), final_temperature=.75)
    assert distribution.call_args.kwargs['final_temperature'] == .75
    assert distribution.call_args.kwargs['scale_multiplier'] == 1.
    assert torch.equal(cache['probability'], corrected['posterior'])
    assert cache['posterior_temperature'] == .75
    target = torch.zeros(3, 2, dtype=q.dtype)
    keys = [(0, 0, 0., 0.)] * 3
    actual = _calibration_nll(cache=cache, target=target, keys=keys, model=model,
             temperature=.75, scale_multiplier=1., device=torch.device('cpu'))
    expected = -torch.logsumexp(corrected['posterior'].log() + torch.tensor([[0., -2., -5.]], dtype=q.dtype), dim=1).mean()
    assert abs(actual - float(expected)) < 1e-12
    with pytest.raises(ValueError, match='recomputed'):
        _calibration_nll(cache=cache, target=target, keys=keys, model=model,
                        temperature=1.5, scale_multiplier=1., device=torch.device('cpu'))
