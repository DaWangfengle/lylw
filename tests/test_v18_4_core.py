from __future__ import annotations

import torch

from fhm_dapp_tim.fhm_dapp.v18_4_core import (
    empirical_survival_reliability,
    fit_empirical_reference,
    kl_p_to_q,
    masked_softmax,
    normalize_energy,
    robust_energy_scale,
    spatial_sinkhorn_divergence,
    tail_mass_diagnostics,
    trust_region_posterior_correction,
    weighted_component_geometric_median,
)


def test_masked_softmax_has_exact_zero_padding() -> None:
    logits = torch.tensor(
        [[2.0, 1.0, 100.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [[True, True, False], [True, False, True]],
    )
    probability = masked_softmax(logits, mask)
    assert torch.allclose(
        probability.sum(dim=1),
        torch.ones(2, dtype=probability.dtype),
    )
    assert probability[0, 2] == 0.0
    assert probability[1, 1] == 0.0


def test_energy_normalization_is_additive_shift_invariant() -> None:
    energy = torch.tensor(
        [[3.0, 4.0, 8.0], [10.0, 12.0, 0.0]],
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [[True, True, True], [True, True, False]],
    )
    scale = robust_energy_scale(energy, mask)
    first = normalize_energy(energy, mask, scale)
    shifted = normalize_energy(
        energy + torch.tensor([[100.0], [-37.0]]),
        mask,
        scale,
    )
    assert torch.allclose(first, shifted, atol=1e-12)
    assert bool((first[mask] >= 0.0).all())


def test_empirical_survival_reliability_is_monotone() -> None:
    reference = fit_empirical_reference(
        torch.tensor([0.1, 0.2, 0.4, 0.8, 1.6])
    )
    values = torch.tensor([0.0, 0.2, 0.5, 2.0])
    reliability = empirical_survival_reliability(values, reference)
    assert bool((reliability[:-1] >= reliability[1:]).all())
    assert bool((reliability > 0.0).all())
    assert bool((reliability < 1.0).all())


def test_trust_region_correction_respects_kl_budget() -> None:
    q = torch.tensor(
        [[0.55, 0.25, 0.15, 0.05]],
        dtype=torch.float64,
    )
    energy = torch.tensor(
        [[2.0, 1.0, 0.1, 0.0]],
        dtype=torch.float64,
    )
    mask = torch.ones_like(q, dtype=torch.bool)
    result = trust_region_posterior_correction(
        q,
        energy,
        epsilon_kl=0.05,
        mask=mask,
        bisection_iterations=60,
    )
    p = result["posterior"]
    assert float(result["realized_kl"][0]) <= 0.0500001
    assert 0.0 <= float(result["rho"][0]) <= 1.0
    assert p[0, 3] > q[0, 3]
    assert torch.allclose(p.sum(dim=1), torch.ones(1, dtype=p.dtype))


def test_trust_region_zero_budget_is_identity() -> None:
    q = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
    energy = torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float64)
    mask = torch.ones_like(q, dtype=torch.bool)
    result = trust_region_posterior_correction(
        q,
        energy,
        epsilon_kl=0.0,
        mask=mask,
    )
    assert torch.allclose(result["posterior"], q, atol=1e-12)
    assert torch.all(result["rho"] == 0.0)


def test_trust_region_large_budget_applies_full_correction() -> None:
    q = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
    energy = torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float64)
    mask = torch.ones_like(q, dtype=torch.bool)
    result = trust_region_posterior_correction(
        q,
        energy,
        epsilon_kl=10.0,
        mask=mask,
    )
    assert torch.allclose(result["rho"], torch.ones(1, dtype=q.dtype))
    assert torch.allclose(
        result["realized_kl"],
        kl_p_to_q(result["posterior"], q, mask),
    )


def test_trust_region_is_invariant_to_energy_translation() -> None:
    q = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1]],
        dtype=torch.float64,
    )
    energy = torch.tensor(
        [[0.2, 0.8, 1.5, 4.0]],
        dtype=torch.float64,
    )
    mask = torch.ones_like(q, dtype=torch.bool)
    first = trust_region_posterior_correction(
        q,
        energy,
        epsilon_kl=0.05,
        mask=mask,
        bisection_iterations=60,
    )
    second = trust_region_posterior_correction(
        q,
        energy + 137.0,
        epsilon_kl=0.05,
        mask=mask,
        bisection_iterations=60,
    )
    assert torch.allclose(
        first["posterior"],
        second["posterior"],
        atol=1e-10,
    )
    assert torch.allclose(first["rho"], second["rho"], atol=1e-10)


def test_sinkhorn_divergence_zero_for_identical_distribution() -> None:
    probability = torch.tensor(
        [[0.5, 0.3, 0.2]],
        dtype=torch.float64,
    )
    coords = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]],
        dtype=torch.float64,
    )
    mask = torch.ones_like(probability, dtype=torch.bool)
    divergence = spatial_sinkhorn_divergence(
        probability,
        coords,
        probability,
        coords,
        spatial_scale=1.0,
        predicted_mask=mask,
        target_mask=mask,
        epsilon_ot=0.1,
        iterations=80,
    )
    assert float(divergence[0]) < 1e-8


def test_sinkhorn_penalizes_farther_spatial_mass_more() -> None:
    predicted = torch.tensor(
        [[0.9, 0.1]],
        dtype=torch.float64,
    )
    pred_coords = torch.tensor(
        [[[0.0, 0.0], [10.0, 0.0]]],
        dtype=torch.float64,
    )
    target = torch.tensor([[1.0]], dtype=torch.float64)
    near_target = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    far_target = torch.tensor([[[9.0, 0.0]]], dtype=torch.float64)
    predicted_mask = torch.ones_like(predicted, dtype=torch.bool)
    target_mask = torch.ones_like(target, dtype=torch.bool)

    near = spatial_sinkhorn_divergence(
        predicted,
        pred_coords,
        target,
        near_target,
        spatial_scale=1.0,
        predicted_mask=predicted_mask,
        target_mask=target_mask,
        epsilon_ot=0.1,
        iterations=80,
    )
    far = spatial_sinkhorn_divergence(
        predicted,
        pred_coords,
        target,
        far_target,
        spatial_scale=1.0,
        predicted_mask=predicted_mask,
        target_mask=target_mask,
        epsilon_ot=0.1,
        iterations=80,
    )
    assert float(far[0]) > float(near[0])


def test_sinkhorn_has_finite_gradient_to_predicted_probability() -> None:
    logits = torch.tensor(
        [[1.0, 0.0, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    probability = torch.softmax(logits, dim=1)
    coords = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [8.0, 0.0]]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.7, 0.3]],
        dtype=torch.float64,
    )
    target_coords = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0]]],
        dtype=torch.float64,
    )
    loss = spatial_sinkhorn_divergence(
        probability,
        coords,
        target,
        target_coords,
        spatial_scale=2.0,
        epsilon_ot=0.1,
        iterations=60,
    ).mean()
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().sum()) > 0.0



def test_sinkhorn_padding_has_finite_gradient() -> None:
    logits = torch.tensor(
        [[1.0, 0.0, -1.0, -2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    predicted_mask = torch.tensor([[True, True, True, False]])
    probability = torch.softmax(
        logits.masked_fill(~predicted_mask, float("-inf")),
        dim=1,
    )
    coords = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [8.0, 0.0], [999.0, 999.0]]],
        dtype=torch.float64,
    )
    target = torch.tensor([[0.7, 0.3, 0.0]], dtype=torch.float64)
    target_coords = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [999.0, 999.0]]],
        dtype=torch.float64,
    )
    target_mask = torch.tensor([[True, True, False]])
    loss = spatial_sinkhorn_divergence(
        probability,
        coords,
        target,
        target_coords,
        spatial_scale=2.0,
        predicted_mask=predicted_mask,
        target_mask=target_mask,
        epsilon_ot=0.1,
        iterations=60,
    ).mean()
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad[:, :3].abs().sum()) > 0.0


def test_component_geometric_median_resists_far_low_mass() -> None:
    probability = torch.tensor(
        [[0.45, 0.45, 0.10]],
        dtype=torch.float64,
    )
    locations = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]]],
        dtype=torch.float64,
    )
    mean = torch.sum(probability.unsqueeze(2) * locations, dim=1)
    median = weighted_component_geometric_median(
        probability,
        locations,
    )
    assert float(mean[0, 0]) > 10.0
    assert abs(float(median[0, 0]) - 1.0) < 1e-3


def test_tail_mass_diagnostics_detect_far_leverage() -> None:
    probability = torch.tensor(
        [[0.85, 0.10, 0.05]],
        dtype=torch.float64,
    )
    locations = torch.tensor(
        [[[0.0, 0.0], [5.0, 0.0], [100.0, 0.0]]],
        dtype=torch.float64,
    )
    target = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    result = tail_mass_diagnostics(
        probability,
        locations,
        target_coords=target,
    )
    assert torch.allclose(
        result["truth_far_mass_20m"],
        torch.tensor([0.05], dtype=torch.float64),
    )
    assert float(result["gm_spatial_leverage_m"][0]) > 5.0
