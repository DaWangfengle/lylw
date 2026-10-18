import math

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp.msep_v17_spatial_operators_v1 import (
    build_doubly_stochastic_floor_graph,
    graph_diffuse_logits,
    layer_mass_preserving_softmax,
    full_logit_graph_diffusion,
    log_partition_gap,
    cell_lse_expansion,
    joint_bf_xy_nll,
    component_center_transport_risk,
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


def test_equal_alpha_dual_channel_is_full_logit_diffusion():
    coords = np.array([[0., 0.], [1., 0.], [0., 1.]])
    graph, _ = build_doubly_stochastic_floor_graph(
        coords, np.zeros(3), np.zeros(3), count=2, temperature_m=2.0
    )
    scores = torch.tensor([[0.2, 1.0, -0.5]])
    evidence = torch.tensor([[[0.1, 0.3], [0.7, -0.2], [-0.1, 0.4]]])
    theta = torch.tensor([0.4, 0.6])
    base, dual = full_logit_graph_diffusion(
        scores, evidence, theta, 0.8, graph, 0.25, 0.25
    )
    direct = graph_diffuse_logits(base, graph, 0.25)
    assert torch.allclose(dual, direct, atol=1e-6, rtol=1e-6)
    assert torch.all(log_partition_gap(base, dual) <= 2e-6)


def test_layer_mass_preserving_softmax_preserves_each_bf_mass():
    base = torch.tensor([[1.2, 0.2, -0.5, 0.1, 0.8]])
    diffused = torch.tensor([[0.1, 1.0, 0.9, -0.2, 0.5]])
    classes = torch.tensor([0, 0, 1, 1, 1])
    q0 = torch.softmax(base, dim=-1)
    q1 = layer_mass_preserving_softmax(base, diffused, classes)
    for cls in (0, 1):
        mask = classes == cls
        assert torch.allclose(q0[:, mask].sum(-1), q1[:, mask].sum(-1), atol=1e-7)
    assert torch.allclose(q1.sum(-1), torch.ones(1), atol=1e-7)
    identity = layer_mass_preserving_softmax(base, base, classes)
    assert torch.allclose(identity, q0, atol=1e-7, rtol=1e-7)


def _cell_inputs(parent_logits, virtual_logits, beta):
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
    )


def test_cell_lse_is_measure_preserving_for_homogeneous_field():
    beta = 0.4
    parent_logits = torch.full((2, 3), 0.7)
    virtual_logits = torch.full((2, 2), 0.7)
    result = _cell_inputs(parent_logits, virtual_logits, beta)
    assert torch.allclose(result["cell_logits"], parent_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(result["cell_probabilities"].sum(-1), torch.ones(2), atol=1e-7)
    assert torch.allclose(result["internal_weights"].sum(-1), torch.ones((2, 3)), atol=1e-7)
    assert torch.allclose(result["expanded_weights"].sum(-1), torch.ones(2), atol=1e-7)


def test_cell_lse_beta_zero_exactly_returns_parent_components():
    parent_logits = torch.tensor([[1.0, -0.5, 0.3]])
    virtual_logits = torch.tensor([[3.0, 4.0]])
    result = _cell_inputs(parent_logits, virtual_logits, 0.0)
    assert torch.equal(result["cell_logits"], parent_logits)
    assert torch.allclose(result["cell_probabilities"], torch.softmax(parent_logits, dim=-1))
    weights = result["expanded_weights"].reshape(1, 3, -1)
    assert torch.allclose(weights[:, :, 0], result["cell_probabilities"], atol=1e-7)
    assert torch.allclose(weights[:, :, 1:], torch.zeros_like(weights[:, :, 1:]), atol=1e-7)


def test_cell_lse_rejects_cross_floor_midpoint_edges():
    parent_logits = torch.zeros((1, 2))
    edge = torch.tensor([[0, 1]])
    centers = torch.zeros((1, 2, 2))
    scales = torch.ones_like(centers)
    virtual_centers = torch.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="crosses building-floor"):
        cell_lse_expansion(
            parent_logits, torch.zeros((1, 1)), edge, 0.3,
            centers, scales, virtual_centers, virtual_centers,
            torch.tensor([0, 1]),
        )


def test_joint_bf_xy_nll_and_center_transport_risk_have_finite_gradients():
    logits = torch.tensor([[0.2, 0.8, -0.4]], requires_grad=True)
    q = torch.softmax(logits, dim=-1)
    centers = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]], requires_grad=True)
    scales = torch.ones_like(centers)
    target = torch.tensor([[0.4, 0.0]])
    anchor_building = torch.tensor([0, 0, 1])
    anchor_floor = torch.tensor([0, 0, 0])
    nll = joint_bf_xy_nll(
        q, torch.tensor([[0, 1, 2]]), centers, scales, target,
        torch.tensor([0]), torch.tensor([0]), anchor_building, anchor_floor
    ).mean()
    risk = component_center_transport_risk(q, centers, target).mean()
    (nll + 0.3 * risk).backward()
    assert torch.isfinite(nll)
    assert torch.isfinite(risk)
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(centers.grad).all()
