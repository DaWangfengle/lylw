import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_operators_v1 as op
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_composition_v1 as composition


def _toy_graph():
    coords = np.asarray([[0., 0.], [1., 0.], [2., 0.], [0., 5.]], dtype=np.float32)
    building = np.asarray([0, 0, 0, 1], dtype=np.int64)
    floor = np.asarray([0, 0, 0, 0], dtype=np.int64)
    graph, edges = op.build_doubly_stochastic_floor_graph(
        coords, building, floor, count=2, temperature_m=2.0
    )
    return coords, building, floor, graph, edges


def test_composed_parent_logits_apply_prior_and_observation_diffusion_asymmetrically():
    _, _, _, graph, _ = _toy_graph()
    scores = torch.tensor([[0.2, 1.4, -0.3, 0.5]], dtype=torch.float32)
    evidence = torch.tensor(
        [[[0.3, -0.4], [0.8, 0.2], [-0.5, 0.6], [0.1, -0.2]]],
        dtype=torch.float32,
    )
    pack = {"scores": scores, "z": evidence}
    ctx = {"graph": graph}
    clean = {"temperature": 1.7, "theta": [0.4, 0.6]}
    selected = {"alpha_prior": 0.5, "alpha_obs": 0.25}

    prior = scores / clean["temperature"]
    obs = (evidence * torch.tensor(clean["theta"])).sum(-1)
    expected_1a = op.graph_diffuse_logits(prior, graph, 0.5) + obs
    expected_1d = (
        op.graph_diffuse_logits(prior, graph, 0.5)
        + op.graph_diffuse_logits(obs, graph, 0.25)
    )
    actual_1a = composition.compose_parent_logits(
        pack, ctx, clean, s1_arm="1A_global", s1_params=selected
    )
    actual_1d = composition.compose_parent_logits(
        pack, ctx, clean, s1_arm="1D_global", s1_params=selected
    )
    assert torch.allclose(actual_1a, expected_1a, atol=1e-7, rtol=1e-7)
    assert torch.allclose(actual_1d, expected_1d, atol=1e-7, rtol=1e-7)
    assert not torch.allclose(actual_1a, actual_1d)


def test_composed_cell_lse_uses_given_s1_parent_logits_and_compact_n_plus_e_order():
    coords, building, floor, graph, edges = _toy_graph()
    device = torch.device("cpu")
    edges = edges.to(device)
    batch, n, e = 2, len(coords), len(edges)
    parent_logits = torch.tensor(
        [[0.1, 0.7, -0.4, 0.2], [0.8, -0.2, 0.3, 0.1]], dtype=torch.float32
    )
    virtual_scores = torch.linspace(-0.5, 0.5, batch * e).reshape(batch, e)
    virtual_z = torch.zeros((batch, e, 2), dtype=torch.float32)
    centers = torch.as_tensor(coords)[None].expand(batch, -1, -1).clone()
    scales = torch.ones((batch, n, 2), dtype=torch.float32)
    virtual_centers = torch.zeros((batch, e, 2), dtype=torch.float32)
    virtual_scales = torch.ones((batch, e, 2), dtype=torch.float32)
    pack = {
        "virtual_scores": virtual_scores,
        "virtual_z": virtual_z,
        "xy": centers,
        "sigma": scales,
        "virtual_xy": virtual_centers,
        "virtual_sigma": virtual_scales,
    }
    ctx = {"edges": edges}
    clean = {"temperature": 1.0, "theta": [0.0, 0.0]}

    identity = composition.compose_cell_mixture(pack, ctx, clean, parent_logits, 0.0)
    assert identity["q"].shape == (batch, n + e)
    assert torch.equal(identity["q"][:, :n], torch.softmax(parent_logits, dim=1))
    assert torch.count_nonzero(identity["q"][:, n:]) == 0
    assert torch.equal(identity["xy"][:, :n], centers)
    assert torch.equal(identity["sigma"][:, :n], scales)

    expanded = composition.compose_cell_mixture(pack, ctx, clean, parent_logits, 0.35)
    assert expanded["q"].shape == (batch, n + e)
    assert expanded["xy"].shape == (batch, n + e, 2)
    assert expanded["sigma"].shape == (batch, n + e, 2)
    assert expanded["ids"].shape == (batch, n + e)
    assert torch.isfinite(expanded["q"]).all()
    assert torch.all(expanded["q"] >= 0)
    assert torch.allclose(expanded["q"].sum(1), torch.ones(batch), atol=2e-6, rtol=2e-6)


def test_expected_composition_methods_are_complete_and_do_not_include_training_arms():
    assert len(composition.METHOD_NAMES) == 13
    assert len(set(composition.METHOD_NAMES)) == 13
    assert "raw_backbone" in composition.METHOD_NAMES
    assert "joint_calibrated_base" in composition.METHOD_NAMES
    assert "1A_global" in composition.METHOD_NAMES
    assert "1D_global" in composition.METHOD_NAMES
    assert "combo_1A_2B" in composition.METHOD_NAMES
    assert "combo_1D_2B" in composition.METHOD_NAMES
    assert all(not name.startswith(("O_", "A_", "C_", "D_", "E_")) for name in composition.METHOD_NAMES)
    pairs = set(composition._comparisons(set(composition.METHOD_NAMES)))
    assert ("raw_backbone", "joint_calibrated_base") in pairs
    assert all(
        ("raw_backbone", candidate) in pairs
        for candidate in composition.METHOD_NAMES if candidate != "raw_backbone"
    )
    assert composition.TEST_COUNTS == {
        "internal_test_rows_scored": 0,
        "official_test_rows_scored": 0,
    }


def test_fast_cell_mixture_matches_public_cell_lse_operator():
    coords, building, floor, graph, edges = _toy_graph()
    batch, n, e = 3, len(coords), len(edges)
    parent_logits = torch.tensor(
        [[0.1, 0.7, -0.4, 0.2], [0.8, -0.2, 0.3, 0.1], [-0.5, 0.2, 0.4, -0.1]],
        dtype=torch.float32,
    )
    virtual_logits = torch.linspace(-0.6, 0.7, batch * e).reshape(batch, e)
    parent_centers = torch.as_tensor(coords)[None].expand(batch, -1, -1).clone()
    parent_scales = torch.ones((batch, n, 2), dtype=torch.float32)
    virtual_centers = torch.zeros((batch, e, 2), dtype=torch.float32)
    virtual_scales = torch.full((batch, e, 2), 1.5, dtype=torch.float32)
    class_ids = torch.as_tensor(building * 2 + floor, dtype=torch.long)
    for beta in (0.0, 0.2, 0.65):
        fast = composition.s3.fast_cell_mixture(
            parent_logits, virtual_logits, edges, beta,
            parent_centers, parent_scales, virtual_centers, virtual_scales,
        )
        public = op.cell_lse_expansion(
            parent_logits, virtual_logits, edges, beta,
            parent_centers, parent_scales, virtual_centers, virtual_scales,
            class_ids,
        )
        assert torch.allclose(fast["q"], public["expanded_weights"], atol=2e-6, rtol=2e-6)
        assert torch.allclose(fast["cell_logits"], public["cell_logits"], atol=2e-6, rtol=2e-6)
        assert torch.allclose(fast["xy"], public["expanded_centers"], atol=1e-7, rtol=1e-7)
        assert torch.allclose(fast["sigma"], public["expanded_scales"], atol=1e-7, rtol=1e-7)
        assert torch.equal(fast["ids"], public["expanded_ids"])
        assert torch.allclose(fast["q"].sum(1), torch.ones(batch), atol=2e-6, rtol=2e-6)


def test_s1_delta_propagates_to_virtual_edges_and_preserves_s2_residual():
    coords, building, floor, _, edges = _toy_graph()
    batch, n, e = 2, len(coords), len(edges)
    clean_parent = torch.tensor(
        [[0.2, -0.1, 0.5, 0.3], [-0.4, 0.7, 0.1, -0.2]], dtype=torch.float32
    )
    delta = torch.tensor(
        [[-0.7, 0.2, -0.4, 0.6], [0.4, -0.6, 0.7, -0.2]], dtype=torch.float32
    )
    diffused_parent = clean_parent + delta
    pack = {
        "virtual_scores": torch.linspace(-0.8, 0.9, batch * e).reshape(batch, e),
        "virtual_z": torch.linspace(-0.5, 0.6, batch * e * 2).reshape(batch, e, 2),
        "xy": torch.as_tensor(coords)[None].expand(batch, -1, -1).clone(),
        "sigma": torch.full((batch, n, 2), 1.1, dtype=torch.float32),
        "virtual_xy": torch.zeros((batch, e, 2), dtype=torch.float32),
        "virtual_sigma": torch.full((batch, e, 2), 1.7, dtype=torch.float32),
    }
    ctx = {"edges": edges}
    clean_params = {"temperature": 1.7, "theta": [0.35, 0.65]}
    beta = 0.65

    raw_virtual = op.combine_retrieval_and_g13_logits(
        pack["virtual_scores"], pack["virtual_z"], clean_params["temperature"],
        torch.tensor(clean_params["theta"], dtype=torch.float32),
    )
    expected_virtual = composition.propagate_s1_delta_to_virtual_logits(
        raw_virtual, diffused_parent, clean_parent, edges
    )
    edge_index = torch.as_tensor(edges, dtype=torch.long)
    clean_endpoint_mean = 0.5 * (
        clean_parent[:, edge_index[:, 0]] + clean_parent[:, edge_index[:, 1]]
    )
    diffused_endpoint_mean = 0.5 * (
        diffused_parent[:, edge_index[:, 0]] + diffused_parent[:, edge_index[:, 1]]
    )
    # The virtual logit's residual relative to the endpoint mean is exactly unchanged.
    assert torch.allclose(
        expected_virtual - diffused_endpoint_mean,
        raw_virtual - clean_endpoint_mean,
        atol=1e-7, rtol=1e-7,
    )

    actual = composition.compose_cell_mixture(
        pack, ctx, clean_params, diffused_parent, beta,
        clean_parent_logits=clean_parent,
    )
    expected = composition.s3.fast_cell_mixture(
        diffused_parent, expected_virtual, edges, beta,
        pack["xy"], pack["sigma"], pack["virtual_xy"], pack["virtual_sigma"],
    )
    for key in ("q", "cell_logits", "xy", "sigma"):
        assert torch.allclose(actual[key], expected[key], atol=2e-6, rtol=2e-6)
    assert torch.equal(actual["ids"], expected["ids"])

    unpropagated = composition.compose_cell_mixture(
        pack, ctx, clean_params, diffused_parent, beta
    )
    assert not torch.allclose(actual["q"], unpropagated["q"], atol=1e-6, rtol=1e-6)

    # A zero S1 displacement must reduce exactly to the independent S2 field.
    zero_delta = composition.compose_cell_mixture(
        pack, ctx, clean_params, clean_parent, beta, clean_parent_logits=clean_parent
    )
    standalone = composition.compose_cell_mixture(pack, ctx, clean_params, clean_parent, beta)
    assert torch.allclose(zero_delta["q"], standalone["q"], atol=1e-7, rtol=1e-7)


def test_preflight_emitter_binding_and_entrypoint(tmp_path, monkeypatch, capsys):
    assert callable(composition.emit)
    assert composition.emit is composition.s3.emit

    input_root = tmp_path / "input"
    checkpoint_root = tmp_path / "checkpoints"
    output_root = tmp_path / "output"
    input_root.mkdir()
    checkpoint_root.mkdir()
    (input_root / "INTEGRITY_REVIEW.json").write_text(
        composition.json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    (checkpoint_root / "EXECUTION_MANIFEST.json").write_text("{}", encoding="utf-8")
    (checkpoint_root / "PHASE_T_INTEGRITY_REVIEW.json").write_text(
        composition.json.dumps({"status": "PASS", "completed": 45, "artifact_sha256": {}}),
        encoding="utf-8",
    )

    toy_ctx = {
        "memory": {"coords": torch.tensor([[0.0, 0.0], [3.0, 0.0]])},
        "edges": torch.tensor([[0, 1]], dtype=torch.long),
        "keys": [("site-a",), ("site-b",)],
        "cal_fit": [0],
        "cal_eval": [1],
    }
    monkeypatch.setattr(composition.s3, "DATASETS", ("TOY",))
    monkeypatch.setattr(composition.s3, "SEEDS", (17,))
    monkeypatch.setattr(composition, "_source_hashes", lambda: {"toy_source.py": "toy-sha"})
    monkeypatch.setattr(
        composition.s3, "audit_results",
        lambda *args, **kwargs: {"artifact_sha256": {}},
    )
    monkeypatch.setattr(
        composition.s3, "reconstruct",
        lambda dataset, seed, root, device: toy_ctx,
    )
    monkeypatch.setattr(
        composition, "_verify_checkpoint",
        lambda root, dataset, seed, arm, hashes: {
            "path": f"{dataset}_seed{seed}/{arm}.pt", "sha256": f"sha-{arm}"
        },
    )
    args = composition.argparse.Namespace(
        input_root=str(input_root), checkpoint_root=str(checkpoint_root),
        output=str(output_root), device="cpu",
    )
    composition.preflight(args)

    manifest = composition.s3.read_json(output_root / "EXECUTION_MANIFEST.json")
    assert manifest["status"] == "PASS"
    assert manifest["checkpoint_sha256"] == {
        f"TOY_seed17/{arm}.pt": f"sha-{arm}" for arm in composition.ARMS
    }
    assert manifest["splits"]["TOY_seed17"]["physical_anchors"] == 2
    assert manifest["splits"]["TOY_seed17"]["unique_midpoint_edges"] == 1
    emitted = capsys.readouterr().out
    assert "preflight_complete" in emitted
