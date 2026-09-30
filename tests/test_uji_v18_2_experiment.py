from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp import uji_v18_2_experiment as exp
from fhm_dapp_tim.fhm_dapp.v18_2_single import (
    EVIDENCE_SOURCES,
    V182_CONFIG,
    build_v182_model,
    candidate_union,
    evidence_opinions,
    prototype_retrieval_score,
    validate_v182_config,
    v182_predict,
    v182_training_loss,
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
    g = torch.Generator().manual_seed(182)
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
    config = dict(V182_CONFIG)
    # Synthetic dictionary has only 8 anchors; support sizes are clipped by
    # implementation, but the registered L>K relation remains unchanged.
    model, geometry, neighbours, weights = build_v182_model(
        memory=memory,
        n_aps=16,
        n_buildings=1,
        n_devices=1,
        device=torch.device("cpu"),
        config=config,
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


def test_v182_contract_is_strictly_single_member_and_not_teacher_based() -> None:
    validate_v182_config(V182_CONFIG)
    assert V182_CONFIG["residual_experts"] == 1
    assert V182_CONFIG["evidence_alpha"] == 1.0
    assert V182_CONFIG["candidate_union_size"] > V182_CONFIG["posterior_top_k"]
    assert V182_CONFIG["scan_feature_mode"] == "bounded_maximal_invariant"
    assert V182_CONFIG["prototype_metric"] == "none"

    source = inspect.getsource(
        __import__(
            "fhm_dapp_tim.fhm_dapp.v18_2_single",
            fromlist=["dummy"],
        )
    ).lower()
    assert "projected_ensemble" not in source
    assert "knowledge_distillation" not in source
    assert "teacher checkpoint" not in source

    bad = dict(V182_CONFIG)
    bad["residual_experts"] = 2
    with pytest.raises(ValueError, match="exactly one residual expert"):
        validate_v182_config(bad)


def test_v182_candidate_union_contains_forced_geometry_targets() -> None:
    model, memory, _sample_anchor, _geometry, neighbours, _weights = _model()
    rss = memory["prototype_rss"][:3]
    with torch.no_grad():
        invariant, _embedding = model.retrieval_logits(rss)
        prototype = prototype_retrieval_score(model, rss)
    forced = neighbours[:3]
    union = candidate_union(
        model,
        invariant,
        prototype,
        force_indices=forced,
    )
    indices = union["candidate_indices"]
    for row in range(len(indices)):
        for target in forced[row].tolist():
            assert int(target) in indices[row].tolist()


def test_v182_six_opinions_form_exact_adaptive_arithmetic_pool() -> None:
    model, memory, _sample_anchor, _geometry, _neighbours, _weights = _model()
    rss = memory["prototype_rss"][:2]
    with torch.no_grad():
        invariant, embedding = model.retrieval_logits(rss)
        prototype = prototype_retrieval_score(model, rss)
        union = candidate_union(model, invariant, prototype)
    evidence = evidence_opinions(
        model,
        rss,
        embedding,
        invariant,
        prototype,
        union,
    )
    assert evidence["source_probability"].shape[1] == len(EVIDENCE_SOURCES)
    assert torch.allclose(
        evidence["source_probability"].sum(dim=2),
        torch.ones(len(rss), len(EVIDENCE_SOURCES)),
        atol=1e-5,
    )
    assert torch.allclose(
        evidence["gate_weights"].sum(dim=1),
        torch.ones(len(rss)),
        atol=1e-6,
    )
    expected = torch.sum(
        evidence["gate_weights"].unsqueeze(2)
        * evidence["source_probability"],
        dim=1,
    )
    expected = expected / expected.sum(dim=1, keepdim=True)
    assert torch.allclose(
        evidence["consensus_probability"],
        expected,
        atol=1e-6,
    )


def test_v182_training_loss_is_finite_and_backpropagates() -> None:
    model, memory, sample_anchor, _geometry, neighbours, weights = _model()
    rss = memory["prototype_rss"].clone()
    target = memory["coords"].clone()
    terms = v182_training_loss(
        model,
        rss,
        target,
        torch.from_numpy(sample_anchor).long(),
        torch.zeros(len(rss), dtype=torch.long),
        neighbours,
        weights,
    )
    for name, value in terms.items():
        if value.ndim == 0:
            assert bool(torch.isfinite(value)), name
    terms["loss"].backward()
    gate_grad = model.evidence_gate[-1].weight.grad
    assert gate_grad is not None
    assert bool(torch.isfinite(gate_grad).all())


def test_v182_predict_has_one_posterior_and_evidence_diagnostics() -> None:
    model, memory, _sample_anchor, _geometry, _neighbours, _weights = _model()
    result = v182_predict(
        model,
        memory["prototype_rss"][:2],
        posterior_temperature=1.0,
        scale_multiplier=1.0,
        target_coords=memory["coords"][:2],
    )
    assert result["candidate_weights"].ndim == 2
    assert torch.allclose(
        result["candidate_weights"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    assert result["gate_weights"].shape == (2, len(EVIDENCE_SOURCES))
    assert bool(torch.isfinite(result["nll"]).all())
    assert result["union_candidate_indices"].shape[1] >= (
        result["candidate_indices"].shape[1]
    )


def test_v182_nested_roles_are_new_disjoint_and_complete() -> None:
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


def test_v182_epoch_selection_rejects_early_epochs() -> None:
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


def test_v182_confirmation_lock_is_irreversible(tmp_path: Path) -> None:
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


def test_v182_benchmark_is_explicitly_not_blind() -> None:
    source = inspect.getsource(exp.evaluate_preexposed_benchmark)
    assert "pre_exposed" in source
    assert "blind_confirmation_claim" in source
    registry_source = inspect.getsource(exp.prepare_protocol)
    assert "DEFAULT_BENCHMARK" not in registry_source
    assert "validationData.csv" not in registry_source
