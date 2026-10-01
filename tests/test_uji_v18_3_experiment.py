from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp import uji_v18_3_experiment as exp
from fhm_dapp_tim.fhm_dapp.v18_3_single import (
    EVIDENCE_FAMILIES,
    EVIDENCE_SOURCES,
    V183_CONFIG,
    _apply_family_dropout,
    _bounded_family_weights,
    _bounded_pair_consensus,
    _family_disagreement,
    build_v183_model,
    candidate_union,
    prototype_retrieval_score,
    structured_evidence_opinions,
    validate_v183_config,
    v183_predict,
    v183_training_loss,
)


def _synthetic_memory() -> tuple[dict, np.ndarray]:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [6.0, 0.0],
            [6.0, 2.0],
        ],
        dtype=np.float32,
    )
    building = np.zeros(len(coords), dtype=np.int64)
    floor = np.zeros(len(coords), dtype=np.int64)
    g = torch.Generator().manual_seed(183)
    rss = -0.82 + 0.62 * torch.rand(len(coords), 16, generator=g)
    rss[0, 8:] = -1.0
    rss[1, 10:] = -1.0
    rss[2, 12:] = -1.0
    memory, sample_anchor = build_anchor_memory(
        rss,
        coords,
        building,
        floor,
        np.arange(len(coords)),
    )
    return memory, sample_anchor


def _model():
    memory, sample_anchor = _synthetic_memory()
    model, geometry, neighbours, weights = build_v183_model(
        memory=memory,
        n_aps=16,
        n_buildings=1,
        n_devices=1,
        device=torch.device("cpu"),
    )
    return model, memory, sample_anchor, geometry, neighbours, weights


def _synthetic_grid() -> set[tuple[int, int, float, float]]:
    keys = set()
    for building in range(2):
        for floor in range(2):
            for x in range(7):
                for y in range(7):
                    keys.add((building, floor, float(x), float(y)))
    return keys


def test_v183_contract_is_single_member_without_trainable_gate() -> None:
    validate_v183_config(V183_CONFIG)
    assert V183_CONFIG["residual_experts"] == 1
    assert V183_CONFIG["candidate_union_size"] == 96
    assert V183_CONFIG["posterior_top_k"] == 64
    assert V183_CONFIG["family_min_weight"] == pytest.approx(0.15)
    assert V183_CONFIG["within_family_min_source_weight"] == pytest.approx(0.30)
    assert len(EVIDENCE_FAMILIES) == 3
    assert len(EVIDENCE_SOURCES) == 6

    model, *_ = _model()
    assert not hasattr(model, "evidence_gate")
    parameter_names = [name for name, _ in model.named_parameters()]
    assert not any("gate" in name for name in parameter_names)

    source = inspect.getsource(
        __import__(
            "fhm_dapp_tim.fhm_dapp.v18_3_single",
            fromlist=["dummy"],
        )
    ).lower()
    assert "projectedensemble" not in source
    assert "knowledge_distillation" not in source
    assert "teacher checkpoint" not in source

    bad = dict(V183_CONFIG)
    bad["residual_experts"] = 2
    with pytest.raises(ValueError, match="exactly one residual expert"):
        validate_v183_config(bad)


def test_v183_within_family_consensus_has_hard_source_bounds() -> None:
    first = torch.tensor([[0.99, 0.01], [0.50, 0.50]], dtype=torch.float32)
    second = torch.tensor([[0.50, 0.50], [0.99, 0.01]], dtype=torch.float32)
    family, weights = _bounded_pair_consensus(
        first,
        second,
        min_weight=float(V183_CONFIG["within_family_min_source_weight"]),
        reliability_temperature=float(
            V183_CONFIG["within_family_reliability_temperature"]
        ),
    )
    assert torch.allclose(family.sum(dim=1), torch.ones(2), atol=1e-6)
    assert bool((weights >= 0.30 - 1e-7).all())
    assert bool((weights <= 0.70 + 1e-7).all())
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_v183_family_weights_have_positive_floor_and_radio_mismatch_penalty() -> None:
    family_probability = torch.tensor(
        [
            [
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
            ],
            [
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
            ],
        ],
        dtype=torch.float32,
    )
    reliability = _bounded_family_weights(
        family_probability,
        torch.tensor([0.0, 1.0]),
        config=V183_CONFIG,
    )
    weights = reliability["weights"]
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert bool(
        (
            weights
            >= float(V183_CONFIG["family_min_weight"]) - 1e-7
        ).all()
    )
    assert weights[1, 1] < weights[0, 1]


def test_v183_family_dropout_never_drops_more_than_one_family() -> None:
    weights = torch.full((128, 3), 1.0 / 3.0)
    torch.manual_seed(183)
    dropped, keep = _apply_family_dropout(
        weights,
        probability=1.0,
        training=True,
    )
    assert bool((keep.sum(dim=1) == 2).all())
    assert torch.allclose(dropped.sum(dim=1), torch.ones(128), atol=1e-6)
    assert bool((dropped >= 0.0).all())

    inference, inference_keep = _apply_family_dropout(
        weights,
        probability=1.0,
        training=False,
    )
    assert torch.allclose(inference, weights)
    assert bool((inference_keep == 1.0).all())


def test_v183_disagreement_is_zero_for_identical_and_positive_for_conflict() -> None:
    identical = torch.tensor(
        [[[0.8, 0.1, 0.1]] * 3],
        dtype=torch.float32,
    )
    per_identical, global_identical = _family_disagreement(identical)
    assert torch.allclose(per_identical, torch.zeros_like(per_identical), atol=1e-7)
    assert torch.allclose(
        global_identical,
        torch.zeros_like(global_identical),
        atol=1e-7,
    )

    conflict = torch.tensor(
        [
            [
                [0.98, 0.01, 0.01],
                [0.01, 0.98, 0.01],
                [0.01, 0.01, 0.98],
            ]
        ],
        dtype=torch.float32,
    )
    _per_conflict, global_conflict = _family_disagreement(conflict)
    assert float(global_conflict) > 0.5


def test_v183_structured_opinions_respect_inference_family_floor() -> None:
    model, memory, _sample_anchor, _geometry, _neighbours, _weights = _model()
    rss = memory["prototype_rss"][:2]
    with torch.no_grad():
        invariant, embedding = model.retrieval_logits(rss)
        prototype = prototype_retrieval_score(model, rss)
        union = candidate_union(
            model,
            invariant,
            prototype,
            config=V183_CONFIG,
        )
    evidence = structured_evidence_opinions(
        model,
        rss,
        embedding,
        invariant,
        prototype,
        union,
        training=False,
    )
    assert evidence["source_probability"].shape[1] == 6
    assert evidence["family_probability"].shape[1] == 3
    assert torch.allclose(
        evidence["consensus_probability"].sum(dim=1),
        torch.ones(len(rss)),
        atol=1e-6,
    )
    assert bool(
        (
            evidence["inference_family_weights"]
            >= float(V183_CONFIG["family_min_weight"]) - 1e-7
        ).all()
    )
    assert bool((evidence["disagreement_temperature"] >= 1.0).all())
    assert bool(
        (
            evidence["disagreement_temperature"]
            <= float(V183_CONFIG["disagreement_temperature_max"]) + 1e-7
        ).all()
    )


def test_v183_training_loss_is_finite_and_backpropagates_without_gate() -> None:
    model, memory, sample_anchor, _geometry, neighbours, weights = _model()
    terms = v183_training_loss(
        model,
        memory["prototype_rss"].clone(),
        memory["coords"].clone(),
        torch.from_numpy(sample_anchor).long(),
        torch.zeros(len(sample_anchor), dtype=torch.long),
        neighbours,
        weights,
    )
    for name, value in terms.items():
        if value.ndim == 0:
            assert bool(torch.isfinite(value)), name
    terms["loss"].backward()
    grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert grads
    assert all(bool(torch.isfinite(grad).all()) for grad in grads)


def test_v183_predict_is_finite_and_calibration_cannot_sharpen() -> None:
    model, memory, _sample_anchor, _geometry, _neighbours, _weights = _model()
    result = v183_predict(
        model,
        memory["prototype_rss"][:2],
        posterior_temperature=1.0,
        scale_multiplier=1.0,
        target_coords=memory["coords"][:2],
    )
    assert result["family_weights"].shape == (2, 3)
    assert result["within_family_weights"].shape == (2, 3, 2)
    assert torch.allclose(
        result["candidate_weights"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    assert bool(torch.isfinite(result["nll"]).all())

    with pytest.raises(ValueError, match="never sharpen"):
        v183_predict(
            model,
            memory["prototype_rss"][:1],
            posterior_temperature=0.75,
            scale_multiplier=1.0,
        )


def test_v183_temperature_grid_is_non_sharpening() -> None:
    assert min(exp.CALIBRATION_TEMPERATURE_GRID) >= 1.0
    assert 0.75 not in exp.CALIBRATION_TEMPERATURE_GRID


def test_v183_nested_roles_are_disjoint_complete_and_new() -> None:
    keys = _synthetic_grid()
    roles, _details = exp.construct_coordinate_roles(keys)
    audit = exp.audit_coordinate_roles(roles, keys)
    assert roles["member_fit"].isdisjoint(roles["early_stop"])
    assert roles["member_fit"] | roles["early_stop"] == roles["selection_refit"]
    assert roles["selection_refit"].isdisjoint(roles["calibration"])
    assert roles["selection_refit"] | roles["calibration"] == roles["outer_refit"]
    assert roles["outer_refit"].isdisjoint(roles["confirmation"])
    assert roles["outer_refit"] | roles["confirmation"] == keys
    assert audit["exact_union_identities"] is True
    assert all(value == 0 for value in audit["hull_checks"].values())
    assert exp.OUTER_GEOMETRY_SEED != 18211


def test_v183_epoch_selection_still_rejects_early_epochs() -> None:
    history = []
    for epoch in range(1, 15):
        energy = 5.0
        mae = 5.0
        if epoch == 1:
            energy, mae = 0.1, 0.1
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


def test_v183_confirmation_lock_is_irreversible(tmp_path: Path) -> None:
    path = tmp_path / "confirmation.lock"
    first = exp._create_lock(
        path,
        registry_sha256="registry",
        freeze_sha256="freeze",
        checkpoint_sha256="checkpoint",
    )
    assert first["confirmation_consumed"] is True
    with pytest.raises(PermissionError, match="already been consumed"):
        exp._create_lock(
            path,
            registry_sha256="registry",
            freeze_sha256="freeze",
            checkpoint_sha256="checkpoint",
        )


def test_v183_benchmark_is_explicitly_preexposed() -> None:
    source = inspect.getsource(exp.evaluate_preexposed_benchmark)
    assert "pre_exposed" in source
    assert "blind_confirmation_claim" in source
    registry_source = inspect.getsource(exp.prepare_protocol)
    assert "DEFAULT_BENCHMARK" not in registry_source
    assert "validationData.csv" not in registry_source
