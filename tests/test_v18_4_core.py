from __future__ import annotations

import torch
import pytest

from fhm_dapp_tim.fhm_dapp.v18_4_core import (
    empirical_survival_reliability,
    cap_anisotropic_residual_transform,
    fit_empirical_reference,
    kl_p_to_q,
    masked_softmax,
    normalize_energy,
    robust_energy_scale,
    spatial_energy_distance,
    tail_mass_diagnostics,
    trust_region_posterior_correction,
    reliability_bounded_posterior_correction,
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


def test_spatial_energy_divergence_zero_for_identical_distribution() -> None:
    probability = torch.tensor(
        [[0.5, 0.3, 0.2]],
        dtype=torch.float64,
    )
    coords = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]],
        dtype=torch.float64,
    )
    mask = torch.ones_like(probability, dtype=torch.bool)
    divergence = spatial_energy_distance(
        probability,
        coords,
        probability,
        coords,
        spatial_scale=1.0,
        predicted_mask=mask,
        target_mask=mask,
    )
    assert float(divergence[0]) < 1e-8


def test_spatial_energy_penalizes_farther_spatial_mass_more() -> None:
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

    near = spatial_energy_distance(
        predicted,
        pred_coords,
        target,
        near_target,
        spatial_scale=1.0,
        predicted_mask=predicted_mask,
        target_mask=target_mask,
    )
    far = spatial_energy_distance(
        predicted,
        pred_coords,
        target,
        far_target,
        spatial_scale=1.0,
        predicted_mask=predicted_mask,
        target_mask=target_mask,
    )
    assert float(far[0]) > float(near[0])


def test_spatial_energy_has_finite_gradient_to_predicted_probability() -> None:
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
    loss = spatial_energy_distance(
        probability,
        coords,
        target,
        target_coords,
        spatial_scale=2.0,
    ).mean()
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().sum()) > 0.0



def test_spatial_energy_padding_has_finite_gradient() -> None:
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
    loss = spatial_energy_distance(
        probability,
        coords,
        target,
        target_coords,
        spatial_scale=2.0,
        predicted_mask=predicted_mask,
        target_mask=target_mask,
    ).mean()
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad[:, :3].abs().sum()) > 0.0
    assert float(logits.grad[:, 3].abs().sum()) == 0.0


def test_spatial_energy_batch_matches_analytic_discrete_half_distance() -> None:
    # First row: half-energy distance to a point = 0.25 * (2/2).
    # Second row has coincident support points and must be exactly zero.
    q = torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float64)
    x = torch.tensor([[[0., 0.], [2., 0.]], [[7., 1.], [7., 1.]]], dtype=torch.float64)
    t = torch.ones(2, 1, dtype=torch.float64)
    y = torch.tensor([[[0., 0.]], [[7., 1.]]], dtype=torch.float64)
    actual = spatial_energy_distance(q, x, t, y, spatial_scale=2.0)
    torch.testing.assert_close(actual, torch.tensor([0.25, 0.0], dtype=torch.float64))
    for row in range(2):
        single = spatial_energy_distance(q[row:row+1], x[row:row+1], t[row:row+1], y[row:row+1], spatial_scale=2.0)
        torch.testing.assert_close(actual[row:row+1], single)


def test_spatial_energy_padding_is_inert_even_for_nonfinite_garbage() -> None:
    q = torch.tensor([[0.3, 0.7]], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[[0., 0.], [4., 1.]]], dtype=torch.float64)
    t = torch.tensor([[1.]], dtype=torch.float64)
    y = torch.tensor([[[2., 0.]]], dtype=torch.float64)
    plain = spatial_energy_distance(q, x, t, y, spatial_scale=3.0)
    qp = torch.cat([q, torch.tensor([[float('nan')]], dtype=q.dtype)], dim=1)
    xp = torch.cat([x, torch.full((1, 1, 2), float('inf'), dtype=x.dtype)], dim=1)
    padded = spatial_energy_distance(qp, xp, torch.tensor([[1., -99.]], dtype=t.dtype),
        torch.cat([y, torch.full((1, 1, 2), float('nan'), dtype=y.dtype)], dim=1),
        spatial_scale=3.0, predicted_mask=torch.tensor([[True, True, False]]),
        target_mask=torch.tensor([[True, False]]))
    assert torch.equal(plain, padded)
    first_grad = torch.autograd.grad(plain.sum(), q, retain_graph=True)[0]
    padded_grad = torch.autograd.grad(padded.sum(), q)[0]
    assert torch.equal(first_grad, padded_grad)


def test_spatial_energy_is_nonnegative_on_broad_discrete_support() -> None:
    generator = torch.Generator().manual_seed(184400)
    q = torch.softmax(torch.randn(6, 128, generator=generator, dtype=torch.float64), dim=1)
    t = torch.softmax(torch.randn(6, 18, generator=generator, dtype=torch.float64), dim=1)
    x = torch.randn(6, 128, 2, generator=generator, dtype=torch.float64) * 500.0
    y = torch.randn(6, 18, 2, generator=generator, dtype=torch.float64) * 500.0
    loss = spatial_energy_distance(q, x, t, y, spatial_scale=5.0)
    assert bool((loss >= -1e-10).all())
    assert bool(torch.isfinite(loss).all())


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_trust_region_row_budgets_and_padding(dtype) -> None:
    q = torch.tensor([[0.55, 0.25, 0.20, 0.0]] * 4, dtype=dtype)
    energy = torch.tensor([[8.0, 2.0, 0.0, float('nan')]] * 4, dtype=dtype)
    mask = torch.tensor([[True, True, True, False]] * 4)
    budgets = torch.tensor([0., 0.001, 0.025, 0.10], dtype=torch.float64)
    result = trust_region_posterior_correction(q, energy, epsilon_kl=budgets, mask=mask)
    assert torch.equal(result['posterior'][0], q[0])
    assert torch.equal(result['posterior'][:, 3], torch.zeros(4, dtype=dtype))
    actual = kl_p_to_q(result['posterior'].double(), q.double(), mask)
    assert bool((actual <= budgets + 1e-7).all())
    torch.testing.assert_close(result['realized_kl'], actual, atol=1e-12, rtol=1e-6)
    assert bool((actual[1:] > 0.0).all())
    scalar = trust_region_posterior_correction(q, energy, epsilon_kl=0.025, mask=mask)
    repeated = trust_region_posterior_correction(q, energy, epsilon_kl=torch.full((4,), .025, dtype=torch.float64), mask=mask)
    assert torch.equal(scalar['posterior'], repeated['posterior'])


@pytest.mark.parametrize('invalid', [torch.tensor([-0.1, .1]), torch.tensor([float('nan'), .1]), torch.zeros(2, 2)])
def test_trust_region_invalid_row_budget_fails_closed(invalid) -> None:
    with pytest.raises(ValueError):
        trust_region_posterior_correction(torch.full((2, 2), .5), torch.zeros(2, 2), epsilon_kl=invalid)


def test_reliability_cannot_be_cancelled_by_rho() -> None:
    q = torch.tensor([[.55, .25, .20, .0]] * 5, dtype=torch.float64)
    radio = torch.tensor([[8., 2., 0., 999.]] * 5, dtype=q.dtype)
    geometry = 2.0 * radio
    rr = torch.tensor([1., .5, .1, .01, 0.], dtype=q.dtype)
    result = reliability_bounded_posterior_correction(q, radio, geometry,
        radio_reliability=rr, geometry_reliability=.5 * rr, epsilon_kl=.1,
        mask=torch.tensor([[True, True, True, False]] * 5))
    assert bool((result['epsilon_kl_effective'][:-1] >= result['epsilon_kl_effective'][1:]).all())
    assert bool((result['realized_kl'][:-1] >= result['realized_kl'][1:] - 1e-10).all())
    assert float(result['realized_kl'][0]) > 5 * float(result['realized_kl'][2])
    assert torch.equal(result['posterior'][-1], q[-1])
    torch.testing.assert_close(result['combined_energy'][:4], result['combined_energy'][0].expand(4, -1), atol=1e-14, rtol=1e-14)
    assert bool((result['epsilon_kl_effective'] <= .1).all())
    assert bool((result['realized_kl'] <= result['epsilon_kl_effective'] + 1e-10).all())
    assert bool((result['posterior'][:, -1] == 0.).all())


def test_reliable_source_dominates_direction_without_probability_floor() -> None:
    q = torch.tensor([[.5, .5], [.5, .5]], dtype=torch.float64)
    radio = torch.tensor([[0., 5.], [0., 5.]], dtype=q.dtype)
    geometry = radio.flip(1)
    result = reliability_bounded_posterior_correction(q, radio, geometry,
        radio_reliability=torch.tensor([1., 0.], dtype=q.dtype),
        geometry_reliability=torch.tensor([.01, 1.], dtype=q.dtype), epsilon_kl=.1)
    assert float(result['radio_direction_weight'][0]) > .99
    assert torch.equal(result['combined_energy'][1], geometry[1])
    assert result['posterior'][0, 0] > .5
    assert result['posterior'][1, 1] > .5


def test_anisotropic_cap_preserves_axes_and_ratio() -> None:
    angle = .37
    rotation = torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
                             [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]], dtype=torch.float64)
    oversized = rotation @ torch.diag(torch.tensor([300., 90.], dtype=torch.float64)) @ rotation.T
    normal = torch.tensor([[9., 1.], [1., 3.]], dtype=torch.float64)
    transforms = torch.stack([oversized, normal, torch.zeros_like(normal)])
    capped = cap_anisotropic_residual_transform(transforms, max_residual_m=30.)
    torch.testing.assert_close(capped[0], oversized / 10., rtol=1e-12, atol=1e-12)
    assert torch.equal(capped[1:], transforms[1:])
    assert float(torch.linalg.svdvals(capped).max()) <= 30. + 1e-10


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
