from __future__ import annotations

import copy

import torch

from fhm_dapp_tim.fhm_dapp.v18_4_model import (
    CorrectionReference,
    V184_CONFIG,
    _masked_geometry_energy,
    apply_correction,
    base_posterior,
    build_v184_model,
    candidate_support,
    fit_correction_reference,
    prototype_retrieval_score,
    raw_correction_energies,
    v184_predict,
    v184_training_loss,
    validate_v184_config,
)


def _memory(rows: int = 16, aps: int = 20) -> dict[str, torch.Tensor]:
    torch.manual_seed(1844)
    side = int(rows ** 0.5)
    assert side * side == rows
    coords = torch.tensor(
        [
            [float(x) * 5.0, float(y) * 5.0]
            for y in range(side)
            for x in range(side)
        ],
        dtype=torch.float32,
    )
    prototype = -0.85 + 0.55 * torch.rand(rows, aps)
    missing = torch.rand(rows, aps) < 0.20
    prototype = torch.where(
        missing,
        torch.full_like(prototype, -1.0),
        prototype,
    )
    detected = prototype > -0.999
    detection = torch.where(
        detected,
        torch.full_like(prototype, 0.80),
        torch.full_like(prototype, 0.15),
    )
    center = torch.where(
        detected,
        prototype,
        torch.full_like(prototype, -0.85),
    )
    scale = torch.full_like(prototype, 0.08)
    return {
        "prototype_rss": prototype,
        "detection_prob": detection,
        "rss_center": center,
        "rss_scale": scale,
        "coords": coords,
        "building": torch.zeros(rows, dtype=torch.long),
        "floor": torch.zeros(rows, dtype=torch.long),
        "counts": torch.full((rows,), 4, dtype=torch.long),
    }


def _model() -> tuple:
    memory = _memory()
    model, geometry, neighbours, weights = build_v184_model(
        memory=memory,
        n_aps=memory["prototype_rss"].shape[1],
        n_buildings=1,
        n_devices=2,
        spatial_scale_m=5.0,
        device=torch.device("cpu"),
    )
    return model, memory, geometry, neighbours, weights


def test_v184_config_is_single_member_without_voting_gate() -> None:
    validate_v184_config()
    model, *_ = _model()
    assert model.residual_experts == 1
    assert model.use_candidate_reranker is True
    assert not hasattr(model, "evidence_gate")
    assert V184_CONFIG["prototype_metric"] == "none"
    assert V184_CONFIG["support_ceiling"] == 128


def test_v184_dynamic_support_uses_mask_and_preserves_forced_positive() -> None:
    model, memory, *_ = _model()
    rss = memory["prototype_rss"][:2]
    with torch.no_grad():
        invariant, _embedding = model.retrieval_logits(rss)
        prototype = prototype_retrieval_score(model, rss)

    small = copy.deepcopy(V184_CONFIG)
    small.update(
        {
            "invariant_retrieval_k": 2,
            "prototype_retrieval_k": 2,
            "floor_retrieval_k": 2,
            "geometry_seed_k": 1,
            "support_ceiling": 12,
        }
    )
    force_indices = torch.tensor(
        [[0, 3, 7, 15], [1, 4, 8, 14]],
        dtype=torch.long,
    )
    force_weights = torch.tensor(
        [[0.5, 0.3, 0.2, 0.0], [0.4, 0.4, 0.2, 0.0]],
        dtype=torch.float32,
    )
    support = candidate_support(
        model,
        invariant,
        prototype,
        force_indices=force_indices,
        force_weights=force_weights,
        config=small,
    )
    assert support["candidate_indices"].shape[0] == 2
    assert support["candidate_indices"].shape[1] <= 12
    assert support["candidate_indices"].shape[1] == int(
        support["candidate_mask"].sum(dim=1).max()
    )
    assert support["candidate_mask"].dtype == torch.bool
    for row in range(2):
        selected = set(
            support["candidate_indices"][row][
                support["candidate_mask"][row]
            ].tolist()
        )
        required = {
            int(index)
            for index, weight in zip(
                force_indices[row].tolist(),
                force_weights[row].tolist(),
            )
            if weight > 0.0
        }
        assert required <= selected

    # A zero-weight padding target is never mandatory by itself.
    assert torch.all(support["candidate_mask"].sum(dim=1) <= 12)


def test_v184_training_loss_is_finite_and_backpropagates() -> None:
    model, memory, _geometry, neighbours, weights = _model()
    model.train()
    target_anchor = torch.tensor([0, 5, 10, 15], dtype=torch.long)
    rss = memory["prototype_rss"][target_anchor].clone()
    target = memory["coords"][target_anchor].clone()
    device_id = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    result = v184_training_loss(
        model,
        rss,
        target,
        target_anchor,
        device_id,
        neighbours,
        weights,
        lambda_ot=0.25,
    )
    assert bool(torch.isfinite(result["loss"]))
    assert float(result["natural_recall_mass"]) >= 0.0
    assert float(result["natural_recall_mass"]) <= 1.0
    assert float(result["forced_support_size"]) >= float(
        result["natural_support_size"]
    ) - 16.0

    result["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
    reranker_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.evidence_reranker.parameters()
        if parameter.grad is not None
    )
    assert reranker_grad > 0.0



def test_v184_geometry_energy_ignores_other_building_floor_coordinates() -> None:
    probability = torch.tensor(
        [[0.45, 0.45, 0.10]],
        dtype=torch.float32,
    )
    mask = torch.ones_like(probability, dtype=torch.bool)
    building = torch.tensor([[0, 0, 0]], dtype=torch.long)
    floor = torch.tensor([[0, 0, 1]], dtype=torch.long)
    locations_near = torch.tensor(
        [[[0.0, 0.0], [10.0, 0.0], [0.1, 0.0]]],
        dtype=torch.float32,
    )
    locations_far = locations_near.clone()
    locations_far[0, 2] = torch.tensor([1000.0, 1000.0])

    near = _masked_geometry_energy(
        probability,
        locations_near,
        mask,
        building,
        floor,
        spatial_scale_m=5.0,
        sigma_scale=1.5,
        margin_scale=0.5,
    )
    far = _masked_geometry_energy(
        probability,
        locations_far,
        mask,
        building,
        floor,
        spatial_scale_m=5.0,
        sigma_scale=1.5,
        margin_scale=0.5,
    )
    assert torch.allclose(
        near[:, :2],
        far[:, :2],
        atol=1e-7,
        rtol=1e-7,
    )


def test_v184_correction_profiler_does_not_create_model_parameter_gradients() -> None:
    model, memory, *_ = _model()
    model.eval()
    rss = memory["prototype_rss"][:4].clone()
    with torch.no_grad():
        invariant, embedding = model.retrieval_logits(rss)
        prototype = prototype_retrieval_score(model, rss)
        support = candidate_support(model, invariant, prototype)
        base = base_posterior(
            model,
            rss,
            embedding,
            invariant,
            support,
        )

    model.zero_grad(set_to_none=True)
    quick = copy.deepcopy(V184_CONFIG)
    quick["radio_profile_steps"] = 1
    raw = raw_correction_energies(
        model,
        rss,
        base["probability"],
        support,
        config=quick,
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert bool(torch.isfinite(raw["radio_energy_raw"]).all())
    assert bool(torch.isfinite(raw["geometry_energy_raw"]).all())


def test_v184_reference_and_trust_region_prediction_are_well_formed() -> None:
    model, memory, *_ = _model()
    model.eval()
    rss = memory["prototype_rss"][:4].clone()
    quick = copy.deepcopy(V184_CONFIG)
    quick["radio_profile_steps"] = 1

    with torch.no_grad():
        invariant, embedding = model.retrieval_logits(rss)
        prototype = prototype_retrieval_score(model, rss)
        support = candidate_support(model, invariant, prototype)
        base = base_posterior(
            model,
            rss,
            embedding,
            invariant,
            support,
        )
    raw = raw_correction_energies(
        model,
        rss,
        base["probability"],
        support,
        config=quick,
    )
    reference = fit_correction_reference(
        radio_energy_raw=raw["radio_energy_raw"],
        geometry_energy_raw=raw["geometry_energy_raw"],
        base_probability=base["probability"],
        candidate_mask=base["candidate_mask"],
        style_offset=raw["style_offset"],
        style_scale=raw["style_scale"],
        config=quick,
    )
    assert isinstance(reference, CorrectionReference)

    corrected = apply_correction(
        base_probability=base["probability"],
        candidate_mask=base["candidate_mask"],
        raw=raw,
        reference=reference,
        epsilon_kl=0.05,
        config=quick,
    )
    assert torch.allclose(
        corrected["posterior"].sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )
    assert bool((corrected["realized_kl"] <= 0.050001).all())
    assert bool((corrected["radio_reliability"] >= 0.0).all())
    assert bool((corrected["radio_reliability"] <= 1.0).all())
    assert bool((corrected["geometry_reliability"] >= 0.0).all())
    assert bool((corrected["geometry_reliability"] <= 1.0).all())

    prediction = v184_predict(
        model,
        rss[:2],
        correction_reference=reference,
        epsilon_kl=0.05,
        config=quick,
    )
    assert prediction["coords"].shape == (2, 2)
    assert prediction["component_locations"].shape[2] == 2
    assert torch.allclose(
        prediction["candidate_weights"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    assert bool(
        (
            prediction["correction_realized_kl"]
            <= 0.050001
        ).all()
    )



def test_v184_final_temperature_does_not_change_trust_region_correction() -> None:
    model, memory, *_ = _model()
    model.eval()
    rss = memory["prototype_rss"][:4].clone()
    quick = copy.deepcopy(V184_CONFIG)
    quick["radio_profile_steps"] = 1

    with torch.no_grad():
        invariant, embedding = model.retrieval_logits(rss)
        prototype = prototype_retrieval_score(model, rss)
        support = candidate_support(model, invariant, prototype)
        base = base_posterior(model, rss, embedding, invariant, support)
    raw = raw_correction_energies(
        model,
        rss,
        base["probability"],
        support,
        config=quick,
    )
    reference = fit_correction_reference(
        radio_energy_raw=raw["radio_energy_raw"],
        geometry_energy_raw=raw["geometry_energy_raw"],
        base_probability=base["probability"],
        candidate_mask=base["candidate_mask"],
        style_offset=raw["style_offset"],
        style_scale=raw["style_scale"],
        config=quick,
    )
    from fhm_dapp_tim.fhm_dapp.v18_4_model import v184_distribution

    cold = v184_distribution(
        model,
        rss,
        correction_reference=reference,
        epsilon_kl=0.05,
        final_temperature=0.75,
        config=quick,
    )
    warm = v184_distribution(
        model,
        rss,
        correction_reference=reference,
        epsilon_kl=0.05,
        final_temperature=1.50,
        config=quick,
    )
    assert torch.allclose(
        cold["corrected_probability"],
        warm["corrected_probability"],
        atol=1e-7,
        rtol=1e-7,
    )
    assert torch.allclose(
        cold["correction_realized_kl"],
        warm["correction_realized_kl"],
        atol=1e-7,
        rtol=1e-7,
    )
    assert not torch.allclose(
        cold["candidate_weights"],
        warm["candidate_weights"],
    )


def test_v184_rejects_unregistered_method_hyperparameters() -> None:
    model, memory, _geometry, neighbours, weights = _model()
    target_anchor = torch.tensor([0, 1], dtype=torch.long)
    try:
        v184_training_loss(
            model,
            memory["prototype_rss"][:2],
            memory["coords"][:2],
            target_anchor,
            torch.tensor([0, 1]),
            neighbours,
            weights,
            lambda_ot=0.333,
        )
    except ValueError as exc:
        assert "registered" in str(exc)
    else:
        raise AssertionError("unregistered lambda_ot must fail closed")

    dummy = CorrectionReference(
        radio_energy_scale=torch.tensor(1.0),
        geometry_energy_scale=torch.tensor(1.0),
        radio_mismatch_reference=torch.tensor([0.1, 0.2]),
        geometry_mismatch_reference=torch.tensor([0.1, 0.2]),
    )
    base = torch.tensor([[0.5, 0.5]])
    mask = torch.ones_like(base, dtype=torch.bool)
    raw = {
        "radio_energy_raw": torch.tensor([[0.0, 1.0]]),
        "geometry_energy_raw": torch.tensor([[0.0, 1.0]]),
        "style_offset": torch.tensor([0.0]),
        "style_scale": torch.tensor([1.0]),
    }
    try:
        apply_correction(
            base_probability=base,
            candidate_mask=mask,
            raw=raw,
            reference=dummy,
            epsilon_kl=0.077,
        )
    except ValueError as exc:
        assert "registered" in str(exc)
    else:
        raise AssertionError("unregistered epsilon_kl must fail closed")
