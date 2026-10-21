import math

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import msep_v17_dataset_adaptive_joint_fit_v1 as joint


def test_cutoff_graph_is_symmetric_bistochastic_and_keeps_isolated_self_loops():
    xy = np.asarray([[0., 0.], [0.5, 0.], [3., 0.], [0., 5.]])
    building = np.asarray([0, 0, 0, 0])
    floor = np.asarray([0, 0, 0, 1])
    edges = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    graph, kept, diag = joint.build_cutoff_graph(xy, building, floor, edges, 1.0, 2.0)
    dense = graph.to_dense().numpy()
    assert np.array_equal(kept.numpy(), np.asarray([[0, 1]]))
    assert np.allclose(dense, dense.T, atol=1e-7)
    assert np.allclose(dense.sum(0), 1., atol=2e-6)
    assert np.allclose(dense.sum(1), 1., atol=2e-6)
    assert dense[2, 2] == pytest.approx(1.0)
    assert dense[3, 3] == pytest.approx(1.0)
    assert diag["zero_neighbor_nodes"] == 2
    assert diag["connected_components"] == 3


def test_cutoff_graph_rejects_cross_floor_or_building_edges():
    xy = np.asarray([[0., 0.], [0.5, 0.]])
    with pytest.raises(ValueError, match="cross-BF"):
        joint.build_cutoff_graph(xy, np.asarray([0, 0]), np.asarray([0, 1]),
                                 np.asarray([[0, 1]]), 2.0, 1.0)


def test_virtual_endpoint_weights_conserve_each_parent_measure_and_infinite_tau_is_degree_prior():
    xy = np.asarray([[0., 0.], [1., 0.], [0., 2.], [10., 10.]])
    edges = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    finite = joint.normalized_virtual_endpoint_weights(edges, xy, 1.5)
    for node in range(3):
        total = sum(finite["weight_i"][k] if i == node else finite["weight_j"][k]
                    for k, (i, j) in enumerate(finite["edges"]))
        assert total == pytest.approx(1.0)
    assert finite["degree"][3] == 0
    infinite = joint.normalized_virtual_endpoint_weights(edges, xy, math.inf)
    degree = infinite["degree"]
    for k, (i, j) in enumerate(infinite["edges"]):
        assert infinite["weight_i"][k] == pytest.approx(1.0 / degree[i])
        assert infinite["weight_j"][k] == pytest.approx(1.0 / degree[j])
    assert np.allclose(infinite["kernel"], 1.0)


def test_virtual_distance_kernel_is_monotone_and_edges_are_unique():
    xy = np.asarray([[0., 0.], [1., 0.], [3., 0.]])
    edges = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    weights = joint.normalized_virtual_endpoint_weights(edges, xy, 2.0)
    assert weights["lengths_m"][0] < weights["lengths_m"][1]
    assert weights["kernel"][0] > weights["kernel"][1]
    with pytest.raises(ValueError, match="unique"):
        joint.normalized_virtual_endpoint_weights(np.asarray([[0, 1], [1, 0]]), xy, 2.0)


def test_weighted_cell_mixture_beta_zero_is_exact_clean_softmax():
    parent_logits = torch.tensor([[0.2, -0.4, 1.1]], dtype=torch.float32)
    virtual_logits = torch.tensor([[0.3]], dtype=torch.float32)
    edges = torch.tensor([[0, 1]], dtype=torch.long)
    endpoint = {"weight_i": np.asarray([1.]), "weight_j": np.asarray([1.]),
                "degree": np.asarray([1, 1, 0])}
    pc = torch.zeros((1, 3, 2))
    ps = torch.ones((1, 3, 2))
    vc = torch.zeros((1, 1, 2))
    vs = torch.ones((1, 1, 2))
    result = joint.weighted_cell_mixture(parent_logits, virtual_logits, edges, endpoint, 0.0,
                                         pc, ps, vc, vs)
    assert torch.equal(result["q"], torch.softmax(parent_logits, dim=1))
    assert result["q"].shape == (1, 3)
    result2 = joint.weighted_cell_mixture(parent_logits, virtual_logits, edges, endpoint, 0.35,
                                          pc, ps, vc, vs)
    assert torch.isfinite(result2["q"]).all()
    assert torch.allclose(result2["q"].sum(1), torch.ones(1), atol=1e-7)
    assert result2["q"].shape == (1, 4)


def test_joint_grid_contains_clean_s1_s2_boundaries_and_locks_gamma_zero():
    candidates = joint._calibration_candidates()
    assert {x["beta"] for x in candidates} >= {0., .05, .10, .15, .20, .35, .50, .65, .80}
    assert all("gamma" not in x for x in candidates)
    assert any(x["alpha_prior"] == 0 and x["alpha_obs"] == 0 and x["beta"] == 0 for x in candidates)
    assert any(x["beta"] == 0 and (x["alpha_prior"] > 0 or x["alpha_obs"] > 0) for x in candidates)
    assert any(x["alpha_prior"] == 0 and x["alpha_obs"] == 0 and x["beta"] > 0 for x in candidates)
