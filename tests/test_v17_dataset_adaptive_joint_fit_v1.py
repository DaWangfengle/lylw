import hashlib
import json
import math
from pathlib import Path

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


def test_cutoff_graph_scales_with_edges_instead_of_dense_anchor_square():
    # A sparse graph with many isolated anchors must remain sparse internally;
    # its stored operator has exactly one diagonal per anchor and two entries
    # per retained undirected edge.
    n = 12000
    xy = np.column_stack((np.arange(n, dtype=np.float64), np.zeros(n)))
    building = np.zeros(n, dtype=np.int64)
    floor = np.zeros(n, dtype=np.int64)
    edges = np.asarray([[0, 1], [10, 11], [100, 101], [n - 2, n - 1]], dtype=np.int64)
    graph, kept, diag = joint.build_cutoff_graph(xy, building, floor, edges, 1.01, 2.0)
    assert graph.layout == torch.sparse_coo
    assert graph._nnz() == n + 2 * len(kept)
    assert len(kept) == len(edges)
    assert diag["zero_neighbor_nodes"] == n - 2 * len(edges)
    assert diag["sinkhorn_row_error"] <= 2e-6
    assert diag["sinkhorn_col_error"] <= 2e-6


def test_chunked_fit_neighbor_diagnostics_match_exact_reference():
    xy = np.asarray([[0., 0.], [1., 0.], [0., 0.], [0., 3.], [4., 0.], [10., 0.]])
    building = np.asarray([0, 0, 0, 0, 0, 1])
    floor = np.asarray([0, 0, 0, 0, 0, 0])
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [4, 3]], dtype=np.int64)
    diag = joint._pairwise_fit_geometry(xy, building, floor, edges)
    ids = np.asarray([0, 1, 2, 3, 4])
    block = xy[ids]
    distances = np.linalg.norm(block[:, None, :] - block[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    sorted_distances = np.sort(distances, axis=1)
    assert diag["zero_distance_pairs_within_building_floor"] == 1
    for k in (1, 2, 4):
        expected = sorted_distances[:, k - 1]
        actual = diag["kth_neighbor_distance_m"][str(k)]
        assert actual["count"] == len(expected)
        assert actual["p50"] == pytest.approx(float(np.quantile(expected, .5)))
        for threshold in (5, 10, 15, 20, 30, 40):
            expected_count = int((expected > threshold).sum())
            assert diag["kth_neighbor_threshold_counts"][str(k)][f"gt_{threshold}m"]["count"] == expected_count


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
        total = sum((finite["weight_i"][k] if i == node else finite["weight_j"][k])
                    for k, (i, j) in enumerate(finite["edges"]) if i == node or j == node)
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
    assert all("gamma" not in x and "gamma_space" not in x for x in candidates)
    assert any(x["alpha_prior"] == 0 and x["alpha_obs"] == 0 and x["beta"] == 0 for x in candidates)
    assert any(x["beta"] == 0 and (x["alpha_prior"] > 0 or x["alpha_obs"] > 0) for x in candidates)
    assert any(x["alpha_prior"] == 0 and x["alpha_obs"] == 0 and x["beta"] > 0 for x in candidates)


def test_cutoff_graph_is_deterministic_and_rejects_nonfinite_coordinates():
    xy = np.asarray([[2., 0.], [0., 0.], [1., 0.]])
    b = np.asarray([0, 0, 0])
    f = np.asarray([0, 0, 0])
    edges = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    g1, e1, _ = joint.build_cutoff_graph(xy, b, f, edges, 1.1, 2.0)
    g2, e2, _ = joint.build_cutoff_graph(xy, b, f, edges, 1.1, 2.0)
    assert torch.equal(e1, e2)
    assert torch.equal(g1.to_dense(), g2.to_dense())
    bad = xy.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        joint.build_cutoff_graph(bad, b, f, edges, 1.1, 2.0)


def test_virtual_edge_weights_reject_invalid_indices_and_nonfinite_coordinates():
    with pytest.raises(ValueError, match="endpoint outside"):
        joint.normalized_virtual_endpoint_weights(np.asarray([[0, 9]]), np.zeros((2, 2)), 2.0)
    bad = np.asarray([[0., 0.], [np.inf, 1.]])
    with pytest.raises(ValueError, match="finite"):
        joint.normalized_virtual_endpoint_weights(np.asarray([[0, 1]]), bad, 2.0)


def test_near_optimal_sensitivity_uses_fixed_tolerance_and_excludes_infeasible_rows():
    rows = []
    for candidate_id, geometry_id, objective, feasible, radius in (
        ("best", "g0", 1.000, True, 2.0),
        ("near", "g1", 1.005, True, 4.0),
        ("outside", "g2", 1.020, True, 8.0),
        ("infeasible", "g3", 0.900, False, 16.0),
    ):
        rows.append({
            "candidate_id": candidate_id, "geometry_id": geometry_id, "feasible": feasible,
            "selection_objective_lambda_0.02": objective,
            "selection_objective_lambda_0.05": objective,
            "normalized_bf_cross_entropy": 1.0,
            "coordinate_mae_m": 5.0, "transport_w1_m": 1.0,
            "bf_accuracy": 0.9, "bf_cross_entropy": 1.0, "joint_nll": 2.0,
            "site_p95_mae_m": 8.0, "site_p99_mae_m": 9.0, "worst_site_mae_m": 10.0,
            "r_max_s1_m": radius, "tau_graph_m": 2.0,
            "r_max_s2_m": radius, "tau_virt_m": 2.0,
            "temperature": 1.0, "theta_v": 0.75, "theta_s": 0.75,
            "alpha_prior": 0.0, "alpha_obs": 0.0, "beta": 0.0,
        })
    summary = joint._near_optimal_sensitivity(rows)
    for lambda_key in ("lambda_0.02", "lambda_0.05"):
        entry = summary[lambda_key]
        assert entry["objective_tolerance_absolute"] == pytest.approx(0.01)
        assert entry["near_optimal_candidate_count"] == 2
        assert entry["near_optimal_geometry_count"] == 2
        assert set(entry["top_candidate_ids"]) == {"best", "near"}
        assert entry["parameter_ranges"]["r_max_s1_m"]["min"] == pytest.approx(2.0)
        assert entry["parameter_ranges"]["r_max_s1_m"]["max"] == pytest.approx(4.0)


def test_test_row_guard_rejects_test_indices_without_reading_labels():
    ctx = {"cal_fit": np.asarray([0, 1, 3]), "cal_eval": np.asarray([2]),
           "roles": {"test": np.asarray([3, 4])}}
    assert joint._safe_rows(ctx, [0, 1], "CAL-fit").tolist() == [0, 1]
    assert joint._safe_rows(ctx, [2], "CAL-eval").tolist() == [2]
    with pytest.raises(RuntimeError, match="TEST row guard"):
        joint._safe_rows(ctx, [3], "CAL-fit")


def test_negative_infinity_scale_and_nonfinite_mixture_logits_are_rejected():
    xy = np.asarray([[0., 0.], [1., 0.]])
    with pytest.raises(ValueError, match="R_max"):
        joint.build_cutoff_graph(xy, np.asarray([0, 0]), np.asarray([0, 0]),
                                 np.asarray([[0, 1]]), -math.inf, 1.0)
    endpoint = {"weight_i": np.asarray([1.]), "weight_j": np.asarray([1.]),
                "degree": np.asarray([1, 1])}
    bad_logits = torch.tensor([[float("nan"), 0.0]])
    with pytest.raises(ValueError, match="nonfinite"):
        joint.weighted_cell_mixture(bad_logits, torch.zeros((1, 1)), torch.tensor([[0, 1]]),
            endpoint, 0.2, torch.zeros((1, 2, 2)), torch.ones((1, 2, 2)),
            torch.zeros((1, 1, 2)), torch.ones((1, 1, 2)))


def test_geometry_pool_uses_fit_neighbor_scales_for_k1_k2_k4_k8():
    diag = {"kth_neighbor_distance_m": {
        str(k): {"p50": float(2 * k), "p90": float(3 * k),
                 "p75": float(2.5 * k), "p95": float(3.5 * k), "p25": float(1.5 * k)}
        for k in (1, 2, 4, 8)
    }}
    pool = joint._geometry_value_pool(diag)
    assert len(pool) <= 8
    assert {2.0, 4.0, 8.0, 16.0}.issubset(set(pool))
    assert all(math.isfinite(x) and x > 0 for x in pool)


def test_historical_clean_and_old_combo_must_match_frozen_checkpoint(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "fixture_seed7" / "D"
    run_dir.mkdir(parents=True)
    selection_path = run_dir / "CAL_FIT_SELECTION.json"
    selection_path.write_text(json.dumps({
        "checkpoint_sha256": "old-checkpoint",
        "joint_calibrated_base": {"temperature": 0.5, "theta": [1.0, 1.0]},
        "structure1": {"1D_global": {"alpha_prior": 0.1, "alpha_obs": 0.2}},
        "combined": {"1D_global": {"2A": {"beta": 0.35}}},
    }), encoding="utf-8")
    monkeypatch.setattr(joint.comp, "DEFAULT_OUTPUT", tmp_path)
    monkeypatch.setattr(joint.s3, "grid_calibrate",
                        lambda pack, ctx: ({"temperature": 1.25, "theta": [0.25, 0.75]}, [], 0.0))

    clean = joint._legacy_clean_config("fixture", 7, "D", {}, {}, "current-checkpoint")
    assert clean["baseline_source"] == "recomputed_current_checkpoint_CAL_fit_only"
    assert clean["temperature"] == pytest.approx(1.25)
    assert joint._old_combo("fixture", 7, "D", "current-checkpoint") is None

def test_weighted_cell_mixture_rejects_positive_beta_without_retained_edges():
    parent_logits = torch.zeros((1, 2))
    virtual_logits = torch.empty((1, 0))
    edges = torch.empty((0, 2), dtype=torch.long)
    endpoint = {"weight_i": np.empty(0), "weight_j": np.empty(0),
                "degree": np.zeros(2, dtype=np.int64)}
    parent_centers = torch.zeros((1, 2, 2))
    parent_scales = torch.ones_like(parent_centers)
    virtual_centers = torch.empty((1, 0, 2))
    virtual_scales = torch.empty_like(virtual_centers)
    with pytest.raises(ValueError, match="positive beta requires"):
        joint.weighted_cell_mixture(
            parent_logits, virtual_logits, edges, endpoint, 0.2,
            parent_centers, parent_scales, virtual_centers, virtual_scales,
        )


def test_dataset_selector_supports_three_isolated_sequential_experiments():
    assert joint._selected_datasets("all") == tuple(joint.s3.DATASETS)
    for dataset in joint.s3.DATASETS:
        assert joint._selected_datasets(dataset) == (dataset,)
    with pytest.raises(ValueError, match="unknown dataset"):
        joint._selected_datasets("not-a-dataset")

