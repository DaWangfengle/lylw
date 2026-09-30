from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v18_1_experiment as exp
from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.v18_1_single import (
    V181_CONFIG,
    build_v181_model,
    explicit_radio_correction,
    validate_v181_config,
    v181_training_loss,
)


def _synthetic_grid() -> set[tuple[int, int, float, float]]:
    keys = set()
    for building in range(2):
        for floor in range(2):
            for x in range(7):
                for y in range(7):
                    keys.add((building, floor, float(x), float(y)))
    return keys


def _synthetic_memory() -> tuple[dict, np.ndarray]:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [4.0, 0.0],
            [4.0, 2.0],
        ],
        dtype=np.float32,
    )
    building = np.zeros(len(coords), dtype=np.int64)
    floor = np.zeros(len(coords), dtype=np.int64)
    generator = torch.Generator().manual_seed(7)
    rss = -0.80 + 0.60 * torch.rand(
        len(coords),
        12,
        generator=generator,
    )
    rss[0, 4:] = -1.0
    rss[1, 8:] = -1.0
    memory, sample_anchor = build_anchor_memory(
        rss,
        coords,
        building,
        floor,
        np.arange(len(coords)),
    )
    return memory, sample_anchor


def test_v181_contract_is_strictly_single_member() -> None:
    validate_v181_config(V181_CONFIG)
    assert V181_CONFIG["residual_experts"] == 1
    assert V181_CONFIG["scan_feature_mode"] == "bounded_maximal_invariant"
    assert V181_CONFIG["prototype_metric"] == "none"
    assert V181_CONFIG["candidate_reranker"] is True

    bad = dict(V181_CONFIG)
    bad["residual_experts"] = 2
    with pytest.raises(ValueError, match="exactly one residual expert"):
        validate_v181_config(bad)


def test_v181_training_is_leave_self_out_and_finite() -> None:
    memory, sample_anchor = _synthetic_memory()
    model, _geometry, neighbours, weights = build_v181_model(
        memory=memory,
        n_aps=12,
        n_buildings=1,
        n_devices=1,
        device=torch.device("cpu"),
    )
    anchor_id = torch.arange(model.n_anchors).unsqueeze(1)
    assert not bool(
        ((neighbours.cpu() == anchor_id) & (weights.cpu() > 0.0)).any()
    )

    rss = memory["prototype_rss"].clone()
    target = memory["coords"].clone()
    terms = v181_training_loss(
        model,
        rss,
        target,
        torch.from_numpy(sample_anchor).long(),
        torch.zeros(len(rss), dtype=torch.long),
        neighbours,
        weights,
    )
    assert all(
        bool(torch.isfinite(value))
        for value in terms.values()
        if value.ndim == 0
    )
    terms["loss"].backward()
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_v181_explicit_radio_correction_is_finite_and_changes_logits() -> None:
    memory, _sample_anchor = _synthetic_memory()
    model, _geometry, _neighbours, _weights = build_v181_model(
        memory=memory,
        n_aps=12,
        n_buildings=1,
        n_devices=1,
        device=torch.device("cpu"),
    )
    rss = memory["prototype_rss"][:2]
    with torch.no_grad():
        logits, embedding = model.retrieval_logits(rss)
        coarse, indices = torch.topk(logits, 4, dim=1)
        learned = model.rerank_candidate_logits(
            rss,
            embedding,
            indices,
            coarse,
        )
    corrected, energy, style = explicit_radio_correction(
        model,
        rss,
        indices,
        learned,
    )
    assert corrected.shape == learned.shape
    assert energy.shape == learned.shape
    assert bool(torch.isfinite(corrected).all())
    assert bool(torch.isfinite(energy).all())
    assert bool(torch.isfinite(style["style_offset"]).all())
    assert bool(torch.isfinite(style["style_scale"]).all())
    assert torch.allclose(
        corrected,
        learned - energy / float(V181_CONFIG["radio_lambda"]),
        atol=1e-6,
    )


def test_v181_roles_are_disjoint_and_complete() -> None:
    keys = _synthetic_grid()
    roles, _details = exp.construct_coordinate_roles(keys)
    audit = exp.audit_coordinate_roles(roles, keys)

    assert roles["member_fit"].isdisjoint(roles["early_stop"])
    assert roles["member_fit"] | roles["early_stop"] == roles["selection_refit"]
    assert roles["selection_refit"].isdisjoint(roles["calibration"])
    assert roles["selection_refit"] | roles["calibration"] == keys
    assert roles["full_train"] == keys
    assert audit["exact_union_identities"] is True
    assert all(value == 0 for value in audit["hull_checks"].values())


def test_v181_epoch_selection_ignores_ineligible_epoch_one() -> None:
    history = []
    for epoch in range(1, 15):
        energy = 3.0
        mae = 3.0
        if epoch == 1:
            energy, mae = 0.5, 0.5
        elif epoch == 12:
            energy, mae = 1.00, 2.00
        elif epoch == 13:
            energy, mae = 0.98, 2.20
        elif epoch == 14:
            energy, mae = 1.01, 1.70
        history.append(
            {
                "epoch": epoch,
                "early_stop": {
                    "coordinate_macro_energy_score_m": energy,
                    "coordinate_macro_mae_m": mae,
                },
            }
        )
    assert exp.select_epoch_from_history(history) == 14


def test_v181_calibration_grid_returns_finite_registered_choice() -> None:
    cache = {
        "logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        "location": torch.tensor(
            [
                [[0.0, 0.0], [5.0, 0.0]],
                [[0.0, 0.0], [5.0, 0.0]],
            ]
        ),
        "scale": torch.ones(2, 2, 2),
        "target": torch.tensor([[0.2, 0.0], [4.8, 0.0]]),
    }
    keys = [
        (0, 0, 0.2, 0.0),
        (0, 0, 4.8, 0.0),
    ]
    result = exp.fit_calibration_grid(cache, keys)
    assert result["posterior_temperature"] in exp.CALIBRATION_TEMPERATURE_GRID
    assert result["scale_multiplier"] in exp.CALIBRATION_SCALE_GRID
    assert np.isfinite(result["best_coordinate_macro_mixture_nll"])
    assert len(result["surface"]) == (
        len(exp.CALIBRATION_TEMPERATURE_GRID)
        * len(exp.CALIBRATION_SCALE_GRID)
    )


def test_v181_pre_external_stages_do_not_reference_external_path() -> None:
    for function in (
        exp.prepare_protocol,
        exp.train_selection,
        exp.calibrate_and_freeze,
        exp.train_final,
    ):
        source = inspect.getsource(function)
        assert "DEFAULT_EXTERNAL" not in source
        assert "validationData.csv" not in source


def test_v181_external_lock_is_irreversible(tmp_path: Path) -> None:
    path = tmp_path / "consumed.lock"
    first = exp.create_external_consumption_lock(
        path,
        registry_sha256="registry",
        freeze_sha256="freeze",
        final_checkpoint_sha256="checkpoint",
    )
    assert path.exists()
    assert first["external_consumed"] is True
    with pytest.raises(PermissionError, match="already been consumed"):
        exp.create_external_consumption_lock(
            path,
            registry_sha256="registry",
            freeze_sha256="freeze",
            final_checkpoint_sha256="checkpoint",
        )
