import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp.msep_v17_spatial_operators_v1 import (
    build_doubly_stochastic_floor_graph,
    graph_diffuse_logits,
    layer_mass_preserving_softmax,
    combine_retrieval_and_g13_logits,
    full_logit_graph_diffusion,
    log_partition_gap,
    interpolate_g13_midpoint_memory,
    g13_continuous_field_features,
    interpolate_midpoint_centers,
    interpolate_midpoint_scales,
    cell_lse_expansion,
    joint_bf_xy_nll,
    component_center_transport_risk,
    replace_unmasked_nll_with_joint_bf_xy,
    add_transport_risk_increment,
)
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import energy_score_mc
from fhm_dapp_tim.fhm_dapp.uji_v17_g13_multiscan_observation import (
    observation_features as canonical_g13_observation_features,
)


def test_floor_graph_is_symmetric_doubly_stochastic_and_never_crosses_bf():
    coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.0, 0.0], [1.0, 1.0],
        [8.0, 8.0],
    ])
    building = np.array([0, 0, 0, 0, 0, 1])
    floor = np.array([0, 0, 0, 1, 1, 0])
    graph, edges = build_doubly_stochastic_floor_graph(
        coords, building, floor, count=2, temperature_m=2.0
    )
    dense = graph.to_dense().numpy()
    assert np.allclose(dense, dense.T, atol=2e-6)
    assert np.allclose(dense.sum(axis=0), 1.0, atol=2e-6)
    assert np.allclose(dense.sum(axis=1), 1.0, atol=2e-6)
    for i, j in edges.tolist():
        assert building[i] == building[j]
        assert floor[i] == floor[j]
    nz_i, nz_j = np.nonzero(dense > 0)
    assert np.all(building[nz_i] == building[nz_j])
    assert np.all(floor[nz_i] == floor[nz_j])


def test_graph_diffusion_identity_and_jensen_partition_bound():
    coords = np.array([[0., 0.], [1., 0.], [0., 1.]])
    graph, _ = build_doubly_stochastic_floor_graph(
        coords, np.zeros(3), np.zeros(3), count=2, temperature_m=2.0
    )
    logits = torch.tensor([[2.0, -1.0, 0.5], [-0.3, 0.2, 1.1]])
    identity = graph_diffuse_logits(logits, graph, 0.0)
    assert torch.equal(identity, logits)
    diffused = graph_diffuse_logits(logits, graph, 0.37)
    assert torch.all(log_partition_gap(logits, diffused) <= 2e-6)


def test_complete_retrieval_g13_logit_matches_locked_formula():
    scores = torch.tensor([[1.0, -2.0]])
    evidence = torch.tensor([[[0.2, -0.4], [0.5, 0.7]]])
    theta = torch.tensor([0.3, 0.8])
    actual = combine_retrieval_and_g13_logits(scores, evidence, 2.0, theta)
    expected = scores / 2.0 + 0.3 * evidence[:, :, 0] + 0.8 * evidence[:, :, 1]
    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_equal_alpha_dual_channel_is_full_logit_diffusion():
    coords = np.array([[0., 0.], [1., 0.], [0., 1.]])
    graph, _ = build_doubly_stochastic_floor_graph(
        coords, np.zeros(3), np.zeros(3), count=2, temperature_m=2.0
    )
    scores = torch.tensor([[0.2, 1.0, -0.5]])
    evidence = torch.tensor([[[0.1, 0.3], [0.7, -0.2], [-0.1, 0.4]]])
    theta = torch.tensor([0.4, 0.6])
    base = combine_retrieval_and_g13_logits(scores, evidence, 0.8, theta)
    base2, dual = full_logit_graph_diffusion(scores, evidence, theta, 0.8, graph, 0.25, 0.25)
    direct = graph_diffuse_logits(base, graph, 0.25)
    assert torch.allclose(base, base2, atol=1e-7, rtol=1e-7)
    assert torch.allclose(dual, direct, atol=1e-6, rtol=1e-6)
    assert torch.all(log_partition_gap(base, dual) <= 2e-6)


def test_layer_mass_preservation_uses_raw_unscaled_bf_prior():
    raw_scores = torch.tensor([[4.0, 0.0, -1.0, -2.0]])
    within_logits = raw_scores / 3.0  # a high-temperature posterior must not set BF mass
    classes = torch.tensor([0, 0, 1, 1])
    raw_q = torch.softmax(raw_scores, dim=-1)
    q = layer_mass_preserving_softmax(raw_scores, within_logits, classes)
    for cls in (0, 1):
        mask = classes == cls
        assert torch.allclose(q[:, mask].sum(-1), raw_q[:, mask].sum(-1), atol=1e-7)
    assert torch.allclose(q.sum(-1), torch.ones(1), atol=1e-7)
    assert not torch.allclose(
        q[:, classes == 0].sum(-1),
        torch.softmax(within_logits, dim=-1)[:, classes == 0].sum(-1),
    )


def test_g13_midpoint_interpolation_restores_gamma_space_variance():
    pi = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    mu = torch.tensor([[-0.5, 0.1], [0.3, -0.2], [0.6, 0.5]])
    var = torch.tensor([[0.04, 0.09], [0.16, 0.25], [0.36, 0.49]])
    edges = torch.tensor([[0, 1], [1, 2]])
    a = interpolate_g13_midpoint_memory(pi, mu, var, edges, gamma_space=0.0)
    b = interpolate_g13_midpoint_memory(pi, mu, var, edges, gamma_space=0.4)
    assert torch.allclose(a["detection_probability"], 0.5 * (pi[edges[:, 0]] + pi[edges[:, 1]]))
    assert torch.allclose(a["conditional_mean"], 0.5 * (mu[edges[:, 0]] + mu[edges[:, 1]]))
    assert torch.allclose(a["conditional_variance"], 0.5 * (var[edges[:, 0]] + var[edges[:, 1]]))
    assert torch.allclose(
        b["conditional_variance"] - a["conditional_variance"],
        torch.full_like(a["conditional_variance"], 0.2),
    )
    with pytest.raises(ValueError, match="nonnegative"):
        interpolate_g13_midpoint_memory(pi, mu, var, edges, gamma_space=-0.1)


def test_virtual_g13_features_use_parent_scaling_and_do_not_change_parent_features():
    query = torch.tensor([
        [-0.2, -1.0, -0.6],
        [-1.0, -1.0, -1.0],
    ])
    seen = torch.tensor([True, False, True])
    pi = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.4, 0.7]])
    mu = torch.tensor([[-0.5, 0.1], [0.3, -0.2], [0.0, 0.4]])
    var = torch.tensor([[0.04, 0.09], [0.16, 0.25], [0.08, 0.12]])
    actual = g13_continuous_field_features(query, seen, pi, mu, var, n_physical=2)
    physical_memory = {
        "detection_probability": pi[:2],
        "conditional_mean": mu[:2],
        "conditional_variance": var[:2],
        "model_ap_count": torch.tensor(2),
    }
    ids = torch.arange(2)[None, :].expand(len(query), -1)
    expected_parent = canonical_g13_observation_features(query, ids, seen, physical_memory)
    assert torch.allclose(actual[:, :2], expected_parent, atol=1e-7, rtol=1e-7)
    assert torch.isfinite(actual).all()
    # Appending a different virtual candidate cannot rescale the physical logits.
    changed_pi = torch.cat((pi[:2], torch.tensor([[0.9, 0.1]])), dim=0)
    changed_mu = torch.cat((mu[:2], torch.tensor([[0.8, -0.8]])), dim=0)
    changed_var = torch.cat((var[:2], torch.tensor([[0.2, 0.3]])), dim=0)
    changed = g13_continuous_field_features(
        query, seen, changed_pi, changed_mu, changed_var, n_physical=2
    )
    assert torch.equal(changed[:, :2], actual[:, :2])


def test_midpoint_continuous_centers_and_scales_use_linear_parent_interpolation():
    anchor_xy = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 1.0]])
    parent_mu = torch.tensor([
        [[0.2, 0.4], [2.4, -0.2], [4.8, 1.2]],
        [[-0.1, 0.1], [1.8, 0.5], [4.1, 0.7]],
    ])
    parent_sigma = torch.tensor([
        [[0.5, 0.8], [0.7, 0.6], [0.9, 1.0]],
        [[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]],
    ])
    edges = torch.tensor([[0, 1], [1, 2]])
    result = interpolate_midpoint_centers(anchor_xy, parent_mu, edges)
    expected = 0.5 * (parent_mu[:, edges[:, 0], :] + parent_mu[:, edges[:, 1], :])
    assert torch.allclose(result["residual_interpolated_centers"], expected, atol=1e-7)
    assert result["geometric_centers"].shape == expected.shape
    assert torch.allclose(
        interpolate_midpoint_scales(parent_sigma, edges),
        0.5 * (parent_sigma[:, edges[:, 0], :] + parent_sigma[:, edges[:, 1], :]),
    )


def _cell_inputs(parent_logits, virtual_logits, beta, *, preserve=False, raw_scores=None):
    batch, n = parent_logits.shape
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    parent_centers = torch.arange(batch * n * 2, dtype=torch.float32).reshape(batch, n, 2) / 10
    parent_scales = torch.full_like(parent_centers, 0.8)
    virtual_centers = torch.tensor([[[0.5, 0.0], [1.5, 0.0]]]).expand(batch, -1, -1).clone()
    virtual_scales = torch.full_like(virtual_centers, 0.9)
    return cell_lse_expansion(
        parent_logits, virtual_logits, edges, beta,
        parent_centers, parent_scales, virtual_centers, virtual_scales,
        torch.zeros(n, dtype=torch.long),
        preserve_class_mass=preserve,
        raw_base_scores=raw_scores,
    )


def test_cell_lse_is_measure_preserving_for_constant_complete_logit_field():
    beta = 0.4
    parent_logits = torch.full((2, 3), 0.7)
    virtual_logits = torch.full((2, 2), 0.7)
    result = _cell_inputs(parent_logits, virtual_logits, beta)
    assert torch.allclose(result["cell_logits"], parent_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(result["cell_probabilities"].sum(-1), torch.ones(2), atol=1e-7)
    assert len(result["cell_internal_weights"]) == 3
    assert torch.allclose(result["expanded_weights"].sum(-1), torch.ones(2), atol=1e-7)
    assert result["expanded_weights"].shape == (2, 5)
    assert result["expanded_centers"].shape == (2, 5, 2)
    assert result["expanded_scales"].shape == (2, 5, 2)
    assert torch.all(result["expanded_scales"] > 0)


def test_cell_lse_rejects_cross_floor_or_duplicate_midpoint_edges():
    parent_logits = torch.zeros((1, 2))
    centers = torch.zeros((1, 2, 2))
    scales = torch.ones_like(centers)
    virtual_centers = torch.zeros((1, 1, 2))
    virtual_scales = torch.ones_like(virtual_centers)
    with pytest.raises(ValueError, match="crosses building-floor"):
        cell_lse_expansion(
            parent_logits, torch.zeros((1, 1)), torch.tensor([[0, 1]]), 0.3,
            centers, scales, virtual_centers, virtual_scales,
            torch.tensor([0, 1]),
        )
    with pytest.raises(ValueError, match="exactly once"):
        cell_lse_expansion(
            torch.zeros((1, 2)), torch.zeros((1, 2)), torch.tensor([[0, 1], [1, 0]]), 0.3,
            centers, scales, virtual_centers.expand(1, 2, 2), torch.ones((1, 2, 2)),
            torch.tensor([0, 0]),
        )


def test_beta_zero_compact_mixture_nll_and_energy_score_equal_parent_baseline():
    parent_logits = torch.tensor([[1.0, -0.5, 0.3], [-0.2, 0.4, 1.1]])
    parent_q = torch.softmax(parent_logits, dim=-1)
    parent_centers = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],
        [[0.2, 0.1], [1.2, 0.1], [9.8, 0.2]],
    ])
    parent_scales = torch.full_like(parent_centers, 0.8)
    edges = torch.tensor([[0, 1]])
    virtual_logits = torch.tensor([[3.0], [-2.0]])
    virtual_centers = 0.5 * (parent_centers[:, 0:1, :] + parent_centers[:, 1:2, :])
    virtual_scales = 0.5 * (parent_scales[:, 0:1, :] + parent_scales[:, 1:2, :])
    result = cell_lse_expansion(
        parent_logits, virtual_logits, edges, 0.0,
        parent_centers, parent_scales, virtual_centers, virtual_scales,
        torch.tensor([0, 0, 1]),
    )
    assert result["expanded_weights"].shape == (2, 4)  # exactly N+E, no padded slots
    assert torch.equal(result["cell_logits"], parent_logits)
    assert torch.equal(result["expanded_weights"][:, :3], parent_q)
    assert torch.equal(result["expanded_weights"][:, 3], torch.zeros(2))
    assert torch.equal(result["expanded_centers"][:, :3], parent_centers)
    assert torch.equal(result["cell_centers"], parent_centers)
    assert torch.equal(result["expanded_scales"][:, :3], parent_scales)
    assert torch.all(result["expanded_scales"] > 0)

    target = torch.tensor([[0.4, 0.0], [9.4, 0.0]])
    target_building = torch.tensor([0, 1])
    target_floor = torch.tensor([0, 0])
    anchor_building = torch.tensor([0, 0, 1])
    anchor_floor = torch.tensor([0, 0, 0])
    parent_nll = joint_bf_xy_nll(
        parent_q, torch.arange(3).expand(2, -1), parent_centers, parent_scales, target,
        target_building, target_floor, anchor_building, anchor_floor,
    )
    expanded_nll = joint_bf_xy_nll(
        result["expanded_weights"], result["expanded_ids"],
        result["expanded_centers"], result["expanded_scales"], target,
        target_building, target_floor, anchor_building, anchor_floor,
    )
    assert torch.allclose(expanded_nll, parent_nll)

    parent_es = energy_score_mc(parent_q, parent_centers, parent_scales, target, draws=64, seed=17301)
    expanded_es = energy_score_mc(
        result["expanded_weights"], result["expanded_centers"], result["expanded_scales"],
        target, draws=64, seed=17301,
    )
    assert torch.allclose(expanded_es, parent_es)


def test_joint_nll_masks_wrong_bf_even_when_wrong_component_density_is_huge():
    q = torch.tensor([[0.5, 0.5]])
    ids = torch.tensor([[0, 1]])
    scales = torch.ones((1, 2, 2))
    target = torch.tensor([[0.0, 0.0]])
    target_building = torch.tensor([0])
    target_floor = torch.tensor([0])
    anchor_building = torch.tensor([0, 1])
    anchor_floor = torch.tensor([0, 0])
    # The wrong-BF component is centered exactly on the target in the first arm.
    centers_a = torch.tensor([[[10.0, 0.0], [0.0, 0.0]]])
    centers_b = torch.tensor([[[10.0, 0.0], [100.0, 100.0]]])
    nll_a = joint_bf_xy_nll(
        q, ids, centers_a, scales, target, target_building, target_floor,
        anchor_building, anchor_floor,
    )
    nll_b = joint_bf_xy_nll(
        q, ids, centers_b, scales, target, target_building, target_floor,
        anchor_building, anchor_floor,
    )
    assert torch.equal(nll_a, nll_b)


def test_joint_nll_strictly_ignores_zero_weight_components():
    target = torch.tensor([[0.1, 0.0]])
    target_building = torch.tensor([0])
    target_floor = torch.tensor([0])
    anchor_building = torch.tensor([0, 1, 0])
    anchor_floor = torch.tensor([0, 0, 0])
    q2 = torch.tensor([[0.5, 0.5]])
    ids2 = torch.tensor([[0, 1]])
    centers2 = torch.tensor([[[0.0, 0.0], [4.0, 0.0]]])
    scales2 = torch.ones_like(centers2)
    expected = joint_bf_xy_nll(
        q2, ids2, centers2, scales2, target, target_building, target_floor,
        anchor_building, anchor_floor,
    )
    q3 = torch.tensor([[0.5, 0.5, 0.0]])
    ids3 = torch.tensor([[0, 1, 2]])
    centers3 = torch.tensor([[[0.0, 0.0], [4.0, 0.0], [0.1, 0.0]]])
    scales3 = torch.ones_like(centers3)
    actual = joint_bf_xy_nll(
        q3, ids3, centers3, scales3, target, target_building, target_floor,
        anchor_building, anchor_floor,
    )
    assert torch.equal(actual, expected)


def test_cell_lse_beta_zero_preserves_raw_bf_mass_when_requested():
    parent_logits = torch.tensor([[0.3, -0.1, 1.0, -0.5]])
    raw_scores = torch.tensor([[4.0, 0.0, -1.0, -2.0]])
    edges = torch.tensor([[0, 1]])
    centers = torch.zeros((1, 4, 2))
    scales = torch.ones_like(centers)
    virtual_centers = torch.zeros((1, 1, 2))
    result = cell_lse_expansion(
        parent_logits, torch.tensor([[1.2]]), edges, 0.0,
        centers, scales, virtual_centers, torch.ones_like(virtual_centers),
        torch.tensor([0, 0, 1, 1]), preserve_class_mass=True, raw_base_scores=raw_scores,
    )
    expected = layer_mass_preserving_softmax(raw_scores, parent_logits, torch.tensor([0, 0, 1, 1]))
    assert torch.allclose(result["cell_probabilities"], expected, atol=0.0, rtol=0.0)
    for cls in (0, 1):
        mask = torch.tensor([0, 0, 1, 1]) == cls
        assert torch.allclose(
            result["cell_probabilities"][:, mask].sum(-1),
            torch.softmax(raw_scores, dim=-1)[:, mask].sum(-1),
            atol=1e-7,
        )


def test_beta_zero_bf_mass_preserving_expansion_preserves_joint_metrics():
    parent_logits = torch.tensor([[1.2, 0.2, -0.5, 0.1]])
    raw_scores = torch.tensor([[4.0, 0.0, -1.0, -2.0]])
    classes = torch.tensor([0, 0, 1, 1])
    parent_centers = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]]])
    parent_scales = torch.full_like(parent_centers, 0.8)
    edges = torch.tensor([[0, 1]])
    virtual_centers = 0.5 * (parent_centers[:, 0:1, :] + parent_centers[:, 1:2, :])
    virtual_scales = 0.5 * (parent_scales[:, 0:1, :] + parent_scales[:, 1:2, :])
    result = cell_lse_expansion(
        parent_logits, torch.tensor([[1.5]]), edges, 0.0,
        parent_centers, parent_scales, virtual_centers, virtual_scales,
        classes, preserve_class_mass=True, raw_base_scores=raw_scores,
    )
    parent_q = layer_mass_preserving_softmax(raw_scores, parent_logits, classes)
    target = torch.tensor([[0.3, 0.0]])
    target_building = torch.tensor([0])
    target_floor = torch.tensor([0])
    anchor_building = torch.tensor([0, 0, 1, 1])
    anchor_floor = torch.tensor([0, 0, 0, 0])
    parent_nll = joint_bf_xy_nll(
        parent_q, torch.arange(4)[None, :], parent_centers, parent_scales, target,
        target_building, target_floor, anchor_building, anchor_floor,
    )
    expanded_nll = joint_bf_xy_nll(
        result["expanded_weights"], result["expanded_ids"], result["expanded_centers"],
        result["expanded_scales"], target, target_building, target_floor,
        anchor_building, anchor_floor,
    )
    assert torch.allclose(expanded_nll, parent_nll)
    parent_es = energy_score_mc(parent_q, parent_centers, parent_scales, target, draws=64, seed=17302)
    expanded_es = energy_score_mc(
        result["expanded_weights"], result["expanded_centers"], result["expanded_scales"],
        target, draws=64, seed=17302,
    )
    assert torch.allclose(expanded_es, parent_es)


def test_joint_nll_and_transport_risk_have_finite_gradients():
    logits = torch.tensor([[0.2, 0.8, -0.4]], requires_grad=True)
    q = torch.softmax(logits, dim=-1)
    centers = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]], requires_grad=True)
    scales = torch.ones_like(centers)
    target = torch.tensor([[0.4, 0.0]])
    anchor_building = torch.tensor([0, 0, 1])
    anchor_floor = torch.tensor([0, 0, 0])
    nll = joint_bf_xy_nll(
        q, torch.tensor([[0, 1, 2]]), centers, scales, target,
        torch.tensor([0]), torch.tensor([0]), anchor_building, anchor_floor,
    ).mean()
    risk = component_center_transport_risk(q, centers, target).mean()
    (nll + 0.3 * risk).backward()
    assert torch.isfinite(nll)
    assert torch.isfinite(risk)
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(centers.grad).all()


def test_training_loss_chain_replaces_nll_and_adds_transport_without_replacing_coord_term():
    original_nll = torch.tensor(2.0, requires_grad=True)
    other_terms = torch.tensor(3.0, requires_grad=True)  # includes retrieval/hierarchy/0.25*coord
    joint_nll = torch.tensor(5.0, requires_grad=True)
    loss_a = original_nll + other_terms
    loss_c = replace_unmasked_nll_with_joint_bf_xy(loss_a, original_nll, joint_nll)
    assert torch.allclose(loss_c, other_terms + joint_nll)
    grad_old, grad_other, grad_joint = torch.autograd.grad(
        loss_c, (original_nll, other_terms, joint_nll), retain_graph=True
    )
    assert torch.equal(grad_old, torch.tensor(0.0))
    assert torch.equal(grad_other, torch.tensor(1.0))
    assert torch.equal(grad_joint, torch.tensor(1.0))

    risk = torch.tensor([1.0, 3.0], requires_grad=True)
    loss_d = add_transport_risk_increment(loss_c, risk)
    assert torch.allclose(loss_d, loss_c + 0.30 * risk.mean())
    grad_risk = torch.autograd.grad(loss_d, risk)[0]
    assert torch.allclose(grad_risk, torch.full_like(risk, 0.15))
