from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.uji_v18_4_development import (
    CALIBRATION_SCALE_GRID,
    CALIBRATION_TEMPERATURE_GRID,
    ENERGY_SHORTLIST_MARGIN_M,
    _canonical_sha,
    _coordinate_macro,
    load_registry,
    prepare,
    select_method_record,
)


def test_v184_prepare_freezes_fresh_uji_roles(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry = prepare(registry_path=registry_path)

    assert registry["official_validation_opened"] is False
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


def test_v184_registry_hash_fails_closed(tmp_path: Path) -> None:
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


def test_v184_method_selection_obeys_energy_shortlist_then_gm() -> None:
    assert ENERGY_SHORTLIST_MARGIN_M == 0.05
    records = [
        {
            "lambda_ot": 0.10,
            "epsilon_kl": 0.025,
            "energy_score_m": 5.000,
            "gm_mae_m": 6.0,
            "nll": 6.0,
        },
        {
            "lambda_ot": 0.25,
            "epsilon_kl": 0.050,
            "energy_score_m": 5.040,
            "gm_mae_m": 5.0,
            "nll": 6.2,
        },
        {
            # Best GM but outside the 0.05 m Energy shortlist.
            "lambda_ot": 0.50,
            "epsilon_kl": 0.100,
            "energy_score_m": 5.051,
            "gm_mae_m": 1.0,
            "nll": 1.0,
        },
    ]
    selected = select_method_record(records)
    assert selected["lambda_ot"] == 0.25
    assert selected["epsilon_kl"] == 0.050


def test_v184_method_selection_ties_prefer_smaller_registered_values() -> None:
    records = [
        {
            "lambda_ot": 0.25,
            "epsilon_kl": 0.050,
            "energy_score_m": 5.0,
            "gm_mae_m": 5.0,
            "nll": 5.0,
        },
        {
            "lambda_ot": 0.10,
            "epsilon_kl": 0.100,
            "energy_score_m": 5.0,
            "gm_mae_m": 5.0,
            "nll": 5.0,
        },
        {
            "lambda_ot": 0.10,
            "epsilon_kl": 0.025,
            "energy_score_m": 5.0,
            "gm_mae_m": 5.0,
            "nll": 5.0,
        },
    ]
    selected = select_method_record(records)
    assert selected["lambda_ot"] == 0.10
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
