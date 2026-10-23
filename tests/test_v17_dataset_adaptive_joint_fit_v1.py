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


def test_joint_grid_is_baseline_centered_and_contains_clean_s1_s2_boundaries():
    baseline = {"temperature": 1.0, "theta": [0.5, 1.0], "baseline_source": "fixture"}
    candidates = joint._calibration_candidates(baseline)
    assert {x["beta"] for x in candidates} == {0., .05, .10, .15, .20, .35, .50, .65, .80}
    assert {x["alpha_prior"] for x in candidates} == {0., .05, .10, .20, .35, .65}
    assert {x["alpha_obs"] for x in candidates} == {0., .10, .25}
    assert all("gamma" not in x and "gamma_space" not in x for x in candidates)
    assert any(x["temperature"] == 1.0 and x["theta_v"] == 0.5 and x["theta_s"] == 1.0
               and x["alpha_prior"] == 0 and x["alpha_obs"] == 0 and x["beta"] == 0
               for x in candidates)
    spec = joint._calibration_grid_spec(baseline)
    assert spec["temperature_values"] == pytest.approx([0.75, 1.0, 1.25])
    assert spec["theta_v_values"] == pytest.approx([0.375, 0.5, 0.625])
    assert spec["theta_s_values"] == pytest.approx([0.75, 1.0, 1.25])
    assert len(candidates) == joint._calibration_candidate_count() == 4374

    other = joint._calibration_grid_spec({"temperature": 2.0, "theta": [0.25, 0.5]})
    assert other["temperature_values"] != spec["temperature_values"]
    assert other["theta_v_values"] != spec["theta_v_values"]


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


def test_geometry_pool_separates_cutoff_radii_from_kernel_temperatures():
    edge_quantiles = {"p50": 2.0, "p75": 3.0, "p90": 4.0, "p95": 5.0, "p100": 6.0}
    diag = {
        "base_union_knn_edges": 100,
        "base_union_knn_edge_distance_m": {**edge_quantiles, "p99": 5.5, "count": 100},
        "kth_neighbor_distance_m": {
            str(k): {"p50": float(2 * k), "p90": float(3 * k),
                     "p75": float(2.5 * k), "p95": float(3.5 * k), "p25": float(1.5 * k)}
            for k in (1, 2, 4, 8)
        },
    }
    pools = joint._geometry_candidate_pools(diag)
    assert pools["r_max_s1_m"] == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0])
    assert pools["r_max_s2_m"] == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0])
    expected_kernel = sorted({2.0, 4.0, 8.0, 16.0, 3.0})
    assert pools["tau_graph_m"] == pytest.approx(expected_kernel)
    assert pools["tau_virt_m"] == pytest.approx(expected_kernel)
    assert 0.5 not in pools["r_max_s1_m"]
    assert max(pools["r_max_s1_m"]) == pytest.approx(edge_quantiles["p100"])

    geoms, metadata = joint.geometry_candidates(diag)
    assert len(geoms) == 12
    assert all(math.isfinite(g["r_max_s1_m"]) and math.isfinite(g["tau_graph_m"])
               and math.isfinite(g["r_max_s2_m"]) for g in geoms)
    assert all(g["r_max_s2_m"] <= g["r_max_s1_m"] for g in geoms)
    assert any(math.isinf(g["tau_virt_m"]) for g in geoms)
    assert all(g["r_max_s1_m"] in pools["r_max_s1_m"] for g in geoms)
    assert all(g["r_max_s2_m"] in pools["r_max_s2_m"] for g in geoms)
    assert all(g["tau_graph_m"] in pools["tau_graph_m"] for g in geoms)
    assert all(math.isinf(g["tau_virt_m"]) or g["tau_virt_m"] in pools["tau_virt_m"] for g in geoms)
    assert all(g["r_max_s1_quantile"] in ("p75", "p90", "p95", "p100") for g in geoms)
    assert all(g["r_max_s2_quantile"] in ("p50", "p75", "p90", "p95", "p100") for g in geoms)
    assert metadata["r_max_s2_le_r_max_s1"] is True
    assert joint._validate_kernel_scales(pools, geoms) is True
    invalid_tau = [dict(g) for g in geoms]
    invalid_tau[0]["tau_graph_m"] = 0.5
    with pytest.raises(RuntimeError, match="tau_graph is not a current FIT"):
        joint._validate_kernel_scales(pools, invalid_tau)


def test_geometry_support_validation_rejects_sparse_or_empty_virtual_graphs(monkeypatch):
    coords = np.asarray([[0., 0.], [1., 0.], [3., 0.], [6., 0.], [10., 0.],
                         [15., 0.], [21., 0.], [28., 0.], [36., 0.], [45., 0.]])
    edges = np.asarray([[i, i + 1] for i in range(len(coords) - 1)], dtype=np.int64)
    buildings = np.zeros(len(coords), dtype=np.int64)
    floors = np.zeros(len(coords), dtype=np.int64)
    diag = joint._pairwise_fit_geometry(coords, buildings, floors, edges)
    geoms, _ = joint.geometry_candidates(diag)
    support = joint._validate_geometry_support(coords, edges, geoms)
    assert len(support) == len(geoms) == 12
    assert all(v["retained_s1_edges"] > 0 for v in support.values())
    assert all(v["retained_virtual_edges"] >= v["minimum_required_virtual_edges"] for v in support.values())
    assert all(v["retained_virtual_fraction"] >= 0.45 for v in support.values())

    invalid = [dict(g) for g in geoms]
    invalid[0]["r_max_s2_m"] = 0.5
    with pytest.raises(RuntimeError, match="R_max_S2 is not FIT edge"):
        joint._validate_geometry_support(coords, edges, invalid)

    # Separately exercise the hard support threshold even if a malformed/stale
    # diagnostic claimed a sparse radius was a P50 quantile.
    original_quantiles = joint._quantiles
    fake_edge_quantiles = {"p50": 1.0, "p75": 2.0, "p90": 3.0, "p95": 4.0, "p100": 9.0}
    def quantiles_with_sparse_p50(values):
        if np.asarray(values).size == len(edges):
            return fake_edge_quantiles
        return original_quantiles(values)
    monkeypatch.setattr(joint, "_quantiles", quantiles_with_sparse_p50)
    sparse = [dict(g) for g in geoms]
    sparse[0]["r_max_s1_m"] = 9.0
    sparse[0]["r_max_s2_m"] = 1.0
    with pytest.raises(RuntimeError, match="requires at least"):
        joint._validate_geometry_support(coords, edges, sparse)


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

def test_joint_grid_scores_virtual_candidates_with_model_dtype_and_mass_diagnostics(monkeypatch):
    coords = np.asarray([[0., 0.], [1., 0.], [2., 0.]])
    building = np.asarray([0, 1, 1])
    floor = np.asarray([0, 0, 0])
    edges = torch.tensor([[1, 2]], dtype=torch.long)
    graph, _, _ = joint.build_cutoff_graph(coords, building, floor, edges, 10.0, 2.0)
    endpoint = joint.normalized_virtual_endpoint_weights(edges, coords, 2.0)
    ctx = {
        "memory": {
            "coords": torch.as_tensor(coords, dtype=torch.float32),
            "building": torch.as_tensor(building),
            "floor": torch.as_tensor(floor),
        },
        "edges": edges,
        "classes": torch.tensor([0, 1, 1]),
        "class_pairs": torch.tensor([[0, 0], [1, 0]]),
    }
    xy = torch.as_tensor(coords, dtype=torch.float32)[None].repeat(2, 1, 1)
    pack = {
        "scores": torch.tensor([[3., 0., -1.], [0., 2., 1.]]),
        "z": torch.zeros((2, 3, 2)),
        "virtual_scores": torch.zeros((2, 1)),
        "virtual_z": torch.zeros((2, 1, 2)),
        "xy": xy,
        "sigma": torch.ones_like(xy),
        "virtual_xy": torch.tensor([[[1.5, 0.]], [[1.5, 0.]]]),
        "virtual_sigma": torch.ones((2, 1, 2)),
        "target": torch.tensor([[0.1, 0.1], [1.2, 0.1]]),
        "building": torch.tensor([0, 1]),
        "floor": torch.tensor([0, 0]),
        "weight": torch.tensor([0.5, 0.5]),
        "keys": ["site_a", "site_b"],
    }
    candidate = {
        "candidate_id": "g__cal_test", "geometry_id": "g",
        "r_max_s1_m": 10., "tau_graph_m": 2.,
        "r_max_s2_m": 10., "tau_virt_m": 2.,
        "temperature": 1., "theta_v": 0.75, "theta_s": 0.75,
        "alpha_prior": 0.25, "alpha_obs": 0.25, "beta": 0.2,
    }
    candidates = [dict(candidate, candidate_id=f"g__cal_beta_{beta:.2f}", beta=beta)
                  for beta in joint.BETAS]
    sparse_mm_calls = {"count": 0}
    original_sparse_mm = torch.sparse.mm
    def counted_sparse_mm(*args, **kwargs):
        sparse_mm_calls["count"] += 1
        return original_sparse_mm(*args, **kwargs)
    monkeypatch.setattr(torch.sparse, "mm", counted_sparse_mm)
    rows = joint.evaluate_calibration_grid(
        pack, ctx, graph, edges.numpy(), {"tau_virt_m": 2.}, candidates,
        endpoint_cache=endpoint,
    )
    assert len(rows) == len(joint.BETAS)
    # Parent and observation diffusion are independent of beta: exactly two
    # sparse multiplications, not two per beta value.
    assert sparse_mm_calls["count"] == 2
    for row in rows:
        for metric in ("coordinate_mae_m", "transport_w1_m", "bf_accuracy",
                       "bf_cross_entropy", "joint_nll", "site_p95_mae_m",
                       "site_p99_mae_m", "worst_site_mae_m"):
            assert math.isfinite(row[metric])
    assert {row["beta"] for row in rows} == set(joint.BETAS)
    selected = joint._attach_virtual_measure(
        {"geometry_id": "g", "beta": 0.2},
        {"g": {"S2": {
            "retained_virtual_edges": len(endpoint["edges"]),
            "effective_virtual_mass_by_parent": endpoint["effective_virtual_mass_by_parent"],
        }}},
    )
    assert selected["pre_normalization_total_measure"] == pytest.approx(3.0)

def test_geometry_scales_change_proportionally_with_dataset_fit_quantiles():
    def make_diag(scale):
        return {
            "kth_neighbor_distance_m": {
                str(k): {
                    "p25": 1.0 * k * scale, "p50": 2.0 * k * scale,
                    "p75": 2.5 * k * scale, "p90": 3.0 * k * scale,
                    "p95": 3.5 * k * scale, "p99": 4.0 * k * scale,
                } for k in (1, 2, 4, 8)
            },
            "base_union_knn_edge_distance_m": {
                "p50": 2.0 * scale, "p75": 3.0 * scale,
                "p90": 4.0 * scale, "p95": 5.0 * scale, "p99": 6.0 * scale,
                "p100": 7.0 * scale, "count": 100,
            },
            "base_union_knn_edges": 100,
        }
    a = joint._geometry_candidate_pools(make_diag(1.0))
    b = joint._geometry_candidate_pools(make_diag(10.0))
    for name in a:
        assert b[name] == pytest.approx([10.0 * x for x in a[name]])
    # No cross-dataset fixed meter list can survive this scale-covariance check.


def test_baseline_centered_grid_handles_zero_theta_and_parameter_bounds():
    baseline = {"temperature": 0.25, "theta": [0.0, 0.0]}
    spec = joint._calibration_grid_spec(baseline)
    assert spec["temperature_values"][0] == pytest.approx(0.25)
    assert 0.25 in spec["temperature_values"]
    assert spec["theta_v_values"][0] == 0.0
    assert spec["theta_s_values"][0] == 0.0
    assert len(spec["temperature_values"]) == len(spec["theta_v_values"]) == len(spec["theta_s_values"]) == 3
    assert all(0.25 <= x <= 3.0 for x in spec["temperature_values"])
    assert all(0.0 <= x <= 1.5 for x in spec["theta_v_values"] + spec["theta_s_values"])


def test_bf_cross_entropy_guard_uses_floor_and_absolute_tolerance():
    base = {
        "coordinate_mae_m": 2.0, "transport_w1_m": 1.0,
        "bf_accuracy": 1.0, "bf_cross_entropy": 1e-4, "joint_nll": 3.0,
    }
    candidate = {
        "candidate_id": "small_ce_perturbation", "geometry_id": "g0",
        "coordinate_mae_m": 2.0, "transport_w1_m": 1.0,
        "bf_accuracy": 1.0, "bf_cross_entropy": 0.004,
        "joint_nll": 3.0,
    }
    checked = joint._add_objectives([candidate], base)[0]
    assert checked["normalized_bf_cross_entropy"] == pytest.approx(0.004 / 0.05)
    assert checked["normalized_bf_cross_entropy_delta"] == pytest.approx((0.004 - 1e-4) / 0.05)
    assert checked["normalized_bf_cross_entropy_penalty"] == pytest.approx((0.004 - 1e-4) / 0.05)
    assert checked["normalized_primary"] == pytest.approx(1.0)
    assert checked["normalized_w1_delta"] == pytest.approx(0.0)
    assert checked["mae_non_degradation"] is True
    assert checked["bf_cross_entropy_guard"] is True
    assert checked["feasible"] is True

    worse = dict(candidate, candidate_id="outside_absolute_tolerance", bf_cross_entropy=0.006)
    checked_worse = joint._add_objectives([worse], base)[0]
    assert checked_worse["bf_cross_entropy_guard"] is False
    assert checked_worse["feasible"] is False


def test_mae_guard_blocks_tradeoffs_even_when_bf_ce_and_nll_improve():
    base = {
        "coordinate_mae_m": 2.0, "transport_w1_m": 1.0,
        "bf_accuracy": 1.0, "bf_cross_entropy": 0.015, "joint_nll": 10.0,
    }
    candidate = {
        "candidate_id": "lower_ce_nll_but_worse_mae", "geometry_id": "g0",
        "coordinate_mae_m": 2.05, "transport_w1_m": 0.8,
        "bf_accuracy": 1.0, "bf_cross_entropy": 0.0065, "joint_nll": 5.5,
    }
    checked = joint._add_objectives([candidate], base)[0]
    assert checked["normalized_bf_cross_entropy_penalty"] == pytest.approx(0.0)
    assert checked["normalized_bf_cross_entropy_delta"] < 0.0
    assert checked["normalized_primary"] == pytest.approx(2.05 / 2.0)
    assert checked["normalized_w1_delta"] == pytest.approx(-0.2)
    expected02 = 2.05 / 2.0 + 0.10 * (-0.2) + 0.02 * ((5.5 - 10.0) / 10.0)
    expected05 = 2.05 / 2.0 + 0.10 * (-0.2) + 0.05 * ((5.5 - 10.0) / 10.0)
    assert checked["selection_objective_lambda_0.02"] == pytest.approx(expected02)
    assert checked["selection_objective_lambda_0.05"] == pytest.approx(expected05)
    assert expected05 < 1.0  # scalar score alone would prefer this candidate
    assert checked["mae_non_degradation"] is False
    assert checked["feasible"] is False


def test_exact_clean_anchor_guarantees_nonempty_feasible_set():
    base = {
        "candidate_id": "clean_baseline_reference", "coordinate_mae_m": 2.0,
        "transport_w1_m": 1.0, "bf_accuracy": 1.0, "bf_cross_entropy": 0.015,
        "joint_nll": 4.0, "site_p95_mae_m": 3.0, "site_p99_mae_m": 4.0,
        "worst_site_mae_m": 5.0,
    }
    baseline_config = {"temperature": 1.0, "theta": [0.75, 0.5]}
    anchors = []
    for i in range(3):
        anchors.append({
            "candidate_id": f"g{i}__exact_clean_anchor", "geometry_id": f"g{i}",
            "temperature": 1.0, "theta_v": 0.75, "theta_s": 0.5,
            "alpha_prior": 0.0, "alpha_obs": 0.0, "beta": 0.0,
            "coordinate_mae_m": 2.0, "transport_w1_m": 1.0,
            "bf_accuracy": 1.0, "bf_cross_entropy": 0.015, "joint_nll": 4.0,
            "site_p95_mae_m": 3.0, "site_p99_mae_m": 4.0, "worst_site_mae_m": 5.0,
        })
    rows = joint._add_objectives(anchors + [base], base)
    joint._assert_clean_anchor_feasible(rows, baseline_config, expected_geometry_count=3)
    assert all(r["feasible"] for r in rows[:3])

def test_legacy_clean_baseline_is_comparison_only_not_a_search_candidate():
    common = {
        "feasible": True,
        "normalized_bf_cross_entropy": 1.0,
        "coordinate_mae_m": 2.0, "transport_w1_m": 1.0,
        "bf_accuracy": 1.0, "bf_cross_entropy": 0.0, "joint_nll": 3.0,
        "site_p95_mae_m": 3.0, "site_p99_mae_m": 4.0, "worst_site_mae_m": 5.0,
        "temperature": 1.0, "theta_v": 0.5, "theta_s": 0.5,
        "alpha_prior": 0.0, "alpha_obs": 0.0, "beta": 0.0,
    }
    baseline = dict(common, candidate_id="clean_baseline_reference",
                    geometry_id="legacy_unbounded_8m")
    baseline["selection_objective_lambda_0.02"] = -100.0
    candidate = dict(common, candidate_id="fit_quantile_candidate",
                     geometry_id="fit_quantile_geometry_00")
    candidate["selection_objective_lambda_0.02"] = 1.0
    chosen = joint._choose([baseline, candidate], 0.02)
    assert chosen["candidate_id"] == "fit_quantile_candidate"


def test_calibration_row_chunks_preserve_site_aggregation_for_all_betas():
    coords = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
    edges = torch.tensor([[0, 1], [1, 2]])
    building = np.zeros(3, dtype=np.int64)
    floor = np.zeros(3, dtype=np.int64)
    graph, _, _ = joint.build_cutoff_graph(
        coords.numpy(), building, floor, edges, 2.0, 1.0,
    )
    ctx = {
        "memory": {"coords": coords, "building": torch.zeros(3, dtype=torch.long),
                   "floor": torch.zeros(3, dtype=torch.long)},
        "edges": edges, "classes": torch.zeros(3, dtype=torch.long),
        "class_pairs": torch.tensor([[0, 0]]),
    }
    targets = torch.tensor([[0.2, 0.1]] * 3 + [[1.7, 0.3]] * 2)
    pack = {
        "scores": torch.tensor([[2., 0., -1.], [1., 0., -1.], [2., 1., 0.],
                                [-1., 1., 2.], [-1., 0., 1.]]),
        "z": torch.zeros((5, 3, 2)), "virtual_scores": torch.zeros((5, 2)),
        "virtual_z": torch.zeros((5, 2, 2)),
        "xy": coords[None].repeat(5, 1, 1),
        "sigma": torch.ones((5, 3, 2)),
        "virtual_xy": torch.tensor([[[0.5, 0.], [1.5, 0.]]]).repeat(5, 1, 1),
        "virtual_sigma": torch.ones((5, 2, 2)), "target": targets,
        "building": torch.zeros(5, dtype=torch.long),
        "floor": torch.zeros(5, dtype=torch.long),
        "weight": torch.tensor([1. / 6.] * 3 + [1. / 4.] * 2),
        "keys": ["site_a"] * 3 + ["site_b"] * 2,
    }
    candidates = [
        {"candidate_id": f"beta_{beta}", "temperature": 1., "theta_v": .75,
         "theta_s": .75, "alpha_prior": .1, "alpha_obs": .25, "beta": beta}
        for beta in joint.BETAS
    ]
    full = joint.evaluate_calibration_grid(
        pack, ctx, graph, edges, {"tau_virt_m": 1.}, candidates, row_chunk_size=5,
    )
    chunked = joint.evaluate_calibration_grid(
        pack, ctx, graph, edges, {"tau_virt_m": 1.}, candidates, row_chunk_size=2,
    )
    assert len(full) == len(chunked) == 9
    for reference, actual in zip(full, chunked):
        assert actual["n_fit_rows"] == 5
        assert actual["n_fit_sites"] == 2
        for metric in ("coordinate_mae_m", "transport_w1_m", "bf_accuracy",
                       "bf_cross_entropy", "joint_nll", "site_p95_mae_m",
                       "site_p99_mae_m", "worst_site_mae_m"):
            assert math.isfinite(actual[metric])
            assert actual[metric] == pytest.approx(reference[metric], abs=1e-6, rel=1e-6)


@pytest.mark.parametrize("device,counts", [
    ("cpu", [3] * 10),
    pytest.param("cuda", [7] * 30 + [6] * 39,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@pytest.mark.parametrize("one_wrong", [False, True])
def test_accuracy_uses_integer_site_counts_without_relaxing_the_guard(device, counts, one_wrong):
    n_rows = sum(counts)
    coords = torch.tensor([[0., 0.], [1., 0.], [10., 0.]], device=device)
    edges = torch.tensor([[0, 1]], device=device)
    graph, _, _ = joint.build_cutoff_graph(
        coords.cpu().numpy(), np.array([0, 0, 1]), np.zeros(3, dtype=np.int64), edges, 2., 1.,
    )
    buildings = torch.zeros(n_rows, dtype=torch.long, device=device)
    if one_wrong:
        buildings[-1] = 1
    ctx = {"memory": {"coords": coords, "building": torch.tensor([0, 0, 1], device=device),
                      "floor": torch.zeros(3, dtype=torch.long, device=device)},
           "edges": edges, "classes": torch.tensor([0, 0, 1], device=device),
           "class_pairs": torch.tensor([[0, 0], [1, 0]], device=device)}
    keys = [f"site_{i}" for i, count in enumerate(counts) for _ in range(count)]
    pack = {"scores": torch.tensor([[3., -1., -5.]], device=device).repeat(n_rows, 1),
            "z": torch.zeros((n_rows, 3, 2), device=device),
            "virtual_scores": torch.zeros((n_rows, 1), device=device),
            "virtual_z": torch.zeros((n_rows, 1, 2), device=device),
            "xy": coords[None].repeat(n_rows, 1, 1),
            "sigma": torch.ones((n_rows, 3, 2), device=device),
            "virtual_xy": torch.tensor([[[.5, 0.]]], device=device).repeat(n_rows, 1, 1),
            "virtual_sigma": torch.ones((n_rows, 1, 2), device=device),
            "target": torch.zeros((n_rows, 2), device=device),
            "building": buildings, "floor": torch.zeros_like(buildings),
            "weight": torch.tensor([1. / (len(counts) * count) for count in counts for _ in range(count)], device=device),
            "keys": keys}
    candidates = [{"candidate_id": f"beta_{beta}", "temperature": 1., "theta_v": .75,
                   "theta_s": .75, "alpha_prior": .1, "alpha_obs": .25, "beta": beta}
                  for beta in joint.BETAS]
    rows = joint.evaluate_calibration_grid(pack, ctx, graph, edges, {"tau_virt_m": 1.}, candidates)
    exact = ((len(counts) - 1) + (counts[-1] - 1) / counts[-1]) / len(counts) if one_wrong else 1.
    assert len(rows) == 9
    for row in rows:
        assert math.isfinite(row["bf_accuracy"])
        assert row["bf_accuracy"] == pytest.approx(exact, abs=1e-15, rel=0.)
        if not one_wrong:
            base = dict(row, bf_accuracy=1.)
            checked = joint._add_objectives([dict(row)], base)[0]
            assert checked["bf_accuracy_non_degradation"] is True


import argparse


def boundary_context():
    xy = np.asarray([[0., 0.], [1., 1.],
                     [np.nextafter(np.float32(1.), np.float32(math.inf)), 1.]], dtype=np.float32)
    edges = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    exact = np.linalg.norm(xy.astype(np.float64)[edges[:, 0]] - xy.astype(np.float64)[edges[:, 1]], axis=1)
    radius = float(np.quantile(exact, .5))
    expected = edges[exact <= radius]
    rounded = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    assert len(expected) == 1
    assert int((rounded <= radius).sum()) == 2  # Demonstrate the old boundary ambiguity.
    ctx = {"memory": {"coords": torch.from_numpy(xy), "building": torch.zeros(3, dtype=torch.long),
                       "floor": torch.zeros(3, dtype=torch.long)}, "edges": torch.from_numpy(edges),
           "cal_fit": np.asarray([0]), "cal_eval": np.asarray([1]), "roles": {"test": np.asarray([2])}}
    return ctx, radius, expected, exact


def test_evaluation_cutoff_matches_float64_preflight_without_changing_components():
    ctx, radius, expected, exact = boundary_context()
    xy = ctx["memory"]["coords"]
    centers = (xy[ctx["edges"][:, 0]] + xy[ctx["edges"][:, 1]]) / 2
    pack = {"scores": torch.tensor([[.4, .2, -.1]]), "z": torch.zeros(1, 3, 2),
            "xy": xy[None], "sigma": torch.ones(1, 3, 2),
            "virtual_scores": torch.tensor([[.1, .3]]), "virtual_z": torch.zeros(1, 2, 2),
            "virtual_xy": centers[None], "virtual_sigma": torch.ones(1, 2, 2),
            "geometric_xy": centers[None]}
    config = {"R_max_S1_m": float(exact.max()), "tau_graph_m": 1.,
              "R_max_S2_m": radius, "tau_virt_m": math.inf,
              "theta": [0., 0.], "temperature": 1., "alpha_prior": 0., "alpha_obs": 0., "beta": .5}
    mix, sub, _ = joint._make_probability(pack, ctx, config)
    assert sub["virtual_scores"].shape[1] == len(expected)
    assert torch.equal(sub["virtual_xy"], pack["virtual_xy"][:, :1])
    assert torch.equal(mix["xy"][:, :3], pack["xy"])
    assert torch.equal(mix["sigma"][:, :3], pack["sigma"])
    assert mix["q"].shape[1] == 3 + len(expected)


def test_run_cutoff_matches_preflight_on_float32_anchor_storage(monkeypatch, tmp_path):
    upper = np.nextafter(np.float32(1.), np.float32(math.inf))
    xy = np.asarray([[0., 0.], [1., 1.], [upper, 1.], [10., 0.],
                     [20., 0.], [30., 0.], [40., 0.]], dtype=np.float32)
    edges = np.asarray([[0, 1], [0, 2], [1, 2], [3, 4]], dtype=np.int64)
    ctx = {"memory": {"coords": torch.from_numpy(xy), "building": torch.zeros(7, dtype=torch.long),
                       "floor": torch.zeros(7, dtype=torch.long)}, "edges": torch.from_numpy(edges),
           "cal_fit": np.asarray([0]), "cal_eval": np.asarray([1]), "roles": {"test": np.asarray([2])}}
    diag = joint._pairwise_fit_geometry(xy, np.zeros(7), np.zeros(7), edges)
    geoms, _ = joint.geometry_candidates(diag)
    radius = geoms[0]["r_max_s2_m"]
    xy64 = xy.astype(np.float64)
    expected = edges[np.linalg.norm(xy64[edges[:, 0]] - xy64[edges[:, 1]], axis=1) <= radius]
    assert len(expected) == 2
    assert int((np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1) <= radius).sum()) == 3
    support = joint._validate_geometry_support(xy, ctx["edges"], geoms)
    assert support[geoms[0]["geometry_id"]]["retained_virtual_edges"] == len(expected)
    checkpoint = tmp_path / "synthetic_identity.pt"
    checkpoint.write_bytes(b"not loaded; synthetic identity only")
    monkeypatch.setattr(joint.s3, "SEEDS", (17301,))
    monkeypatch.setattr(joint, "ARMS", ("C",))
    monkeypatch.setattr(joint.s3, "reconstruct", lambda *args: ctx)
    monkeypatch.setattr(joint, "_read_json", lambda *args: {"outer_geometry_candidates": {"tuji_seed17301": {"geometry_candidates": geoms}}})
    monkeypatch.setattr(joint.comp, "_checkpoint_path", lambda *args: checkpoint)
    monkeypatch.setattr(joint.comp, "_load_model", lambda *args: (None, {}, 0.))
    monkeypatch.setattr(joint.s3, "make_pack", lambda *args, **kwargs: {})
    monkeypatch.setattr(joint, "_legacy_clean_config", lambda *args: {})
    monkeypatch.setattr(joint, "_calibration_candidates", lambda *args: [{"candidate_id": "synthetic"}])
    monkeypatch.setattr(joint, "_calibration_grid_spec", lambda *args: {})
    class CanonicalSubsetReached(Exception):
        pass
    def stop_before_scoring(pack, runtime_ctx, graph, s2, geom, candidates, **kwargs):
        assert np.array_equal(s2, expected)
        raise CanonicalSubsetReached
    monkeypatch.setattr(joint, "evaluate_calibration_grid", stop_before_scoring)
    with pytest.raises(CanonicalSubsetReached):
        joint.run(argparse.Namespace(output=str(tmp_path), input_root=str(tmp_path), checkpoint_root=str(tmp_path),
                                     dataset="tuji", device="cpu"))


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@pytest.mark.parametrize("rows", [413, 444])
def test_clean_grid_mean_matches_reference_point_reduction_at_realistic_support_size(device, rows):
    # No real data/checkpoint: zero spatial coefficients must retain the same
    # point decision even with 2500 virtual columns and uneven scan counts.
    generator = torch.Generator().manual_seed(131)
    anchors, virtual_count = 1052, 2500
    coords = (torch.randn(anchors, 2, generator=generator) * 20 + 50).to(device)
    pairs = [(i, i + gap) for gap in (1, 2, 3) for i in range(anchors - gap)]
    edges = torch.tensor(pairs[:virtual_count], dtype=torch.long, device=device)
    xy = coords[None].expand(rows, -1, -1).contiguous()
    virtual_xy = (xy[:, edges[:, 0]] + xy[:, edges[:, 1]]) / 2
    def normal(*shape):
        return torch.randn(*shape, generator=generator).to(device)
    pack = {"scores": normal(rows, anchors), "z": normal(rows, anchors, 2),
            "virtual_scores": normal(rows, virtual_count), "virtual_z": normal(rows, virtual_count, 2),
            "xy": xy, "sigma": torch.ones_like(xy), "virtual_xy": virtual_xy,
            "virtual_sigma": torch.ones_like(virtual_xy), "target": normal(rows, 2) * 5 + 45,
            "building": torch.zeros(rows, dtype=torch.long, device=device),
            "floor": torch.zeros(rows, dtype=torch.long, device=device),
            "keys": [(0, 0, i // 5, 0) for i in range(rows)]}
    counts = {key: pack["keys"].count(key) for key in set(pack["keys"])}
    pack["weight"] = torch.tensor([1 / (len(counts) * counts[key]) for key in pack["keys"]], device=device)
    ctx = {"memory": {"coords": coords, "building": torch.zeros(anchors, dtype=torch.long, device=device),
                       "floor": torch.zeros(anchors, dtype=torch.long, device=device)},
           "edges": edges, "classes": torch.zeros(anchors, dtype=torch.long, device=device),
           "class_pairs": torch.tensor([[0, 0]], device=device)}
    graph = torch.eye(anchors, device=device).to_sparse().coalesce()
    endpoint = joint.normalized_virtual_endpoint_weights(edges.cpu().numpy(), coords.cpu().numpy(), math.inf)
    candidates = [{"candidate_id": f"synthetic_{group}_{beta}", "temperature": .9805950440880841,
                   "theta_v": 1.5, "theta_s": 1.5, "alpha_prior": 0., "alpha_obs": group * .1, "beta": beta}
                  for group in range(2) for beta in joint.BETAS]
    actual = joint.evaluate_calibration_grid(pack, ctx, graph, edges.cpu().numpy(),
        {"tau_virt_m": math.inf}, candidates, endpoint_cache=endpoint)
    logits = joint.op.combine_retrieval_and_g13_logits(pack["scores"], pack["z"], candidates[0]["temperature"],
                                                     torch.tensor([1.5, 1.5], device=device))
    q = torch.softmax(logits, dim=-1)
    reference, _ = joint.s3.spatial_metrics(q, torch.arange(anchors, device=device)[None].expand(rows, -1),
        xy, pack["target"], pack["building"], pack["floor"], ctx["classes"], ctx["class_pairs"], keys=pack["keys"])
    assert len(actual) == 2 * len(joint.BETAS)
    assert abs(actual[0]["coordinate_mae_m"] - reference["macro_mae_full_m"]) <= joint.MAE_NON_DEGRADATION_TOLERANCE_M


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_reference_bf_macro_matches_integer_correct_counts(monkeypatch, device):
    # Ten sites, three scans each; exactly one scan is assigned the wrong BF.
    scans = 30
    q = torch.tensor([[.9, .1]] * scans, device=device)
    q[0] = torch.tensor([.1, .9], device=device)
    centers = torch.tensor([[0., 0.], [10., 0.]], device=device)[None].repeat(scans, 1, 1)
    ids = torch.arange(2, device=device)[None].expand(scans, -1)
    pack = {"xy": centers, "sigma": torch.ones_like(centers), "ids": ids,
            "scores": torch.zeros(scans, 2, device=device), "target": torch.zeros(scans, 2, device=device),
            "building": torch.zeros(scans, dtype=torch.long, device=device),
            "floor": torch.zeros(scans, dtype=torch.long, device=device),
            "keys": [(0, 0, i // 3, 0) for i in range(scans)],
            "weight": torch.full((scans,), 1 / scans, device=device)}
    ctx = {"memory": {"building": torch.tensor([0, 1], device=device),
                       "floor": torch.tensor([0, 0], device=device)},
           "classes": torch.tensor([0, 1], device=device),
           "class_pairs": torch.tensor([[0, 0], [1, 0]], device=device)}
    mix = {"q": q, "xy": centers, "sigma": pack["sigma"], "ids": ids}
    monkeypatch.setattr(joint, "_make_probability", lambda *args: (mix, pack, {}))
    scored = joint._eval_config(pack, ctx, {}, 131)
    expected = (9. + 2. / 3.) / 10.
    assert scored["metrics"]["macro_bf_accuracy"] == pytest.approx(expected, abs=1e-15, rel=0.)
