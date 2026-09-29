from __future__ import annotations

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    build_delaunay_geometry,
    common_student_t_randomness,
    continuous_energy_score_mc,
    marked_kl_update,
    maximal_invariant_scan,
    select_information_saturated_E,
    topk_prior_projection,
    uniform_member_anchor_prior,
)


def test_maximal_invariant_is_affine_invariant_and_preserves_missing_event() -> None:
    rss = torch.tensor(
        [[-0.90, -0.70, -0.30, -1.0], [-0.55, -0.55, -1.0, -1.0]],
        dtype=torch.float32,
    )
    visible = rss > -0.999
    # The positive-affine action is conditional on the fixed visibility event;
    # choose an equivalent representative that remains above the file sentinel.
    transformed = torch.where(visible, 0.8 * rss + 0.15, rss)
    first = maximal_invariant_scan(rss)
    second = maximal_invariant_scan(transformed)
    assert torch.equal(first == -1.0, ~visible)
    assert torch.equal(second == -1.0, ~visible)
    assert torch.allclose(first[visible], second[visible], atol=1e-6)


def test_geometry_targets_are_normalized_and_do_not_cross_strata() -> None:
    coords = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [20.0, 0.0], [22.0, 0.0]],
        dtype=torch.float32,
    )
    buildings = torch.tensor([0, 0, 0, 0, 1, 1])
    floors = torch.zeros(6, dtype=torch.long)
    geometry = build_delaunay_geometry(coords, buildings, floors)
    indices = geometry["neighbor_indices"]
    weights = geometry["neighbor_weights"]
    transform = geometry["residual_transform"]
    assert torch.allclose(weights.sum(dim=1), torch.ones(6), atol=1e-6)
    for anchor in range(6):
        active = indices[anchor][weights[anchor] > 0]
        assert bool((buildings[active] == buildings[anchor]).all())
        assert bool((floors[active] == floors[anchor]).all())
    assert torch.allclose(transform, transform.transpose(1, 2), atol=1e-6)
    eigenvalues = torch.linalg.eigvalsh(transform)
    assert bool((eigenvalues >= -1e-6).all())
    assert int(torch.linalg.matrix_rank(transform[4]).item()) == 1


def test_uniform_joint_and_topk_are_exact_reverse_kl_projection() -> None:
    logits = torch.tensor(
        [
            [[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
            [[2.0, 0.0, 1.0], [1.0, 3.0, 0.0]],
        ]
    )
    joint = uniform_member_anchor_prior(logits)
    assert joint.shape == (2, 2, 3)
    assert torch.allclose(joint.sum(dim=(1, 2)), torch.ones(2))
    assert torch.allclose(joint.sum(dim=2), torch.full((2, 2), 0.5))
    projected = topk_prior_projection(joint, top_k=2)
    retained = projected["retained_mass"]
    assert torch.allclose(projected["joint_probability"].sum(dim=(1, 2)), torch.ones(2))
    assert torch.allclose(projected["reverse_kl"], -torch.log(retained))


def test_marked_update_is_normalized_and_shared_across_members() -> None:
    prior = torch.tensor([[[0.30, 0.20], [0.10, 0.40]]])
    energy = torch.tensor([[0.0, 1.0]])
    posterior = marked_kl_update(prior, energy, lambda_value=0.5)
    assert torch.allclose(posterior.sum(dim=(1, 2)), torch.ones(1))
    prior_member_given_anchor = prior / prior.sum(dim=1, keepdim=True)
    post_member_given_anchor = posterior / posterior.sum(dim=1, keepdim=True)
    assert torch.allclose(prior_member_given_anchor, post_member_given_anchor)


def test_continuous_energy_uses_component_scale_with_common_randomness() -> None:
    randomness = common_student_t_randomness(3, draws=256, seed=7)
    probability = torch.ones(3, 1)
    location = torch.zeros(3, 1, 2)
    target = torch.zeros(3, 2)
    small, _, _ = continuous_energy_score_mc(
        probability,
        location,
        torch.full_like(location, 0.1),
        target,
        randomness,
    )
    large, _, _ = continuous_energy_score_mc(
        probability,
        location,
        torch.full_like(location, 2.0),
        target,
        randomness,
    )
    assert float(large.mean()) > float(small.mean())


def test_continuous_common_random_number_score_matches_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        return
    randomness = common_student_t_randomness(4, draws=32, seed=17)
    probability = torch.softmax(torch.arange(12, dtype=torch.float32).reshape(4, 3), dim=1)
    location = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2) / 5.0
    scale = torch.full_like(location, 0.7)
    target = torch.ones(4, 2)
    cpu, _, _ = continuous_energy_score_mc(
        probability, location, scale, target, randomness
    )
    gpu, _, _ = continuous_energy_score_mc(
        probability.cuda(),
        location.cuda(),
        scale.cuda(),
        target.cuda(),
        {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in randomness.items()},
    )
    assert torch.allclose(cpu, gpu.cpu(), atol=2e-5, rtol=2e-5)


def test_saturation_uses_paired_differences_and_two_later_values() -> None:
    base = np.linspace(-0.02, 0.02, 40)
    losses = {
        1: 1.20 + base,
        2: 1.00 + base,
        3: 0.99 + base,
        4: 0.99 + base,
    }
    selected = select_information_saturated_E(
        losses,
        tolerance=0.03,
        bootstrap_repetitions=500,
        seed=11,
    )
    assert selected["saturated"] is True
    assert selected["selected_E"] == 2
