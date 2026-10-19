"""Synthetic regression tests for the new runner, without research data."""
import copy
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fhm_dapp_tim.fhm_dapp.model import AnchorStudentTModel
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_structure_v1 as study
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_operators_v1 as op
from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import same_floor_neighbors, class_structure
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import energy_score_mc


def toy_model():
    torch.manual_seed(19)
    coords = torch.tensor([[0., 0.], [2., 0.], [4., 1.], [0., 0.], [3., 2.]])
    b, f = torch.zeros(5, dtype=torch.long), torch.tensor([0, 0, 0, 1, 1])
    prototypes = torch.tensor([[-.4, -.2, -1., -.7], [-.6, -.3, -.8, -1.],
        [-.8, -.4, -.3, -.5], [-.2, -.6, -1., -.4], [-.5, -.4, -.3, -.2]])
    model = AnchorStudentTModel(n_aps=4, anchor_coords=coords, anchor_buildings=b, anchor_floors=f,
        seen_ap_mask=torch.ones(4, dtype=torch.bool), prototype_rss=prototypes,
        d_model=16, num_heads=2, num_layers=1, max_visible=4, dropout=.1)
    model.initialize_anchor_keys(prototypes)
    graph, edges = op.build_doubly_stochastic_floor_graph(coords, b, f, count=2)
    ni, nw = same_floor_neighbors(coords, b, f, count=2)
    return model, graph, edges, ni, nw, prototypes[:3], coords[:3], torch.arange(3)


def kwargs():
    return dict(retrieval_weight=.5, hierarchy_weight=1., self_target_weight=.5,
                coordinate_weight=.25, rss_offset_aug_db=4., rss_scale_aug=.08,
                train_candidates=4, ap_dropout=.1)


def test_opt_in_disabled_preserves_default_losses_gradients_predictions_exactly():
    base, graph, edges, ni, nw, x, y, anchor = toy_model()
    wrapped = study.enable_training_arm(base, "A", graph)
    base.train(); wrapped.train()
    torch.manual_seed(117)
    expected = base.training_loss(x, y, anchor, ni, nw, **kwargs())
    torch.manual_seed(117)
    actual = wrapped.training_loss(x, y, anchor, ni, nw, **kwargs())
    assert expected.keys() == actual.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key
    expected["loss"].backward(); actual["loss"].backward()
    for (name, p), (name2, q) in zip(base.named_parameters(), wrapped.named_parameters()):
        assert name == name2
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad), name
    base.eval(); wrapped.eval()
    with torch.no_grad():
        s1, z1 = base.retrieval_logits(x)
        s2, z2 = wrapped.retrieval_logits(x)
        assert torch.equal(s1, s2) and torch.equal(z1, z2)
        ids = torch.arange(base.n_anchors)[None].expand(len(x), -1)
        for a, b in zip(base._components(z1, ids), wrapped._components(z2, ids)):
            assert torch.equal(a, b)


@pytest.mark.parametrize("arm", ["C", "D", "E"])
def test_training_new_losses_reuse_one_encoder_forward_and_have_finite_gradients(arm):
    base, graph, edges, ni, nw, x, y, anchor = toy_model()
    model = study.enable_training_arm(base, arm, graph)
    model.train()
    with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encoder_forward:
        terms = model.training_loss(x, y, anchor, ni, nw, **kwargs())
    assert encoder_forward.call_count == 1
    assert model._spatial_forward_count == 1
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
    if arm == "E":
        assert float(model.raw_alpha.sigmoid().detach()) == pytest.approx(.1)
        assert model.raw_alpha.grad is not None and torch.isfinite(model.raw_alpha.grad)
        assert abs(float(model.raw_alpha.grad)) > 1e-8
        model.eval(); base.eval()
        with torch.no_grad():
            raw, _ = base.retrieval_logits(x)
            diffused, _ = model.retrieval_logits(x)
            assert torch.allclose(diffused, op.graph_diffuse_logits(raw, graph, model.raw_alpha.sigmoid()))
        restored = study.enable_training_arm(base, "E", graph)
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        restored.eval()
        assert torch.equal(restored.retrieval_logits(x)[0], model.retrieval_logits(x)[0])


@pytest.mark.parametrize("beta", [0., .1, .35, .8])
def test_vectorized_compact_cell_matches_public_operator_including_singleton(beta):
    torch.manual_seed(87)
    parent = torch.randn(3, 5)
    edges = torch.tensor([[0, 1], [1, 2], [0, 2]])
    classes = torch.tensor([0, 0, 0, 1, 2])
    virtual = torch.randn(3, 3)
    centers, scales = torch.randn(3, 5, 2), torch.rand(3, 5, 2) + .2
    vc, vs = torch.randn(3, 3, 2), torch.rand(3, 3, 2) + .2
    expected = op.cell_lse_expansion(parent, virtual, edges, beta, centers, scales, vc, vs, classes)
    actual = study.fast_cell_mixture(parent, virtual, edges, beta, centers, scales, vc, vs)
    assert actual["q"].shape == (3, 8)
    assert torch.allclose(actual["q"], expected["expanded_weights"], atol=2e-7, rtol=2e-6)
    assert torch.allclose(actual["cell_logits"], expected["cell_logits"], atol=3e-7, rtol=2e-6)
    assert torch.equal(actual["ids"], expected["expanded_ids"])
    assert torch.equal(actual["xy"], expected["expanded_centers"])
    assert torch.equal(actual["sigma"], expected["expanded_scales"])
    if beta == 0.:
        assert torch.equal(actual["q"][:, :5], torch.softmax(parent, dim=1))
        assert torch.count_nonzero(actual["q"][:, 5:]) == 0
    target = torch.zeros(3, 2)
    b, f = torch.zeros(5, dtype=torch.long), classes
    n1 = op.joint_bf_xy_nll(actual["q"], actual["ids"], actual["xy"], actual["sigma"], target,
                          torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long), b, f)
    n2 = op.joint_bf_xy_nll(expected["expanded_weights"], expected["expanded_ids"], expected["expanded_centers"],
                          expected["expanded_scales"], target, torch.zeros(3, dtype=torch.long),
                          torch.zeros(3, dtype=torch.long), b, f)
    assert torch.allclose(n1, n2)


def test_virtual_retrieval_interpolates_stored_key_and_prototype_without_zero_imputation():
    model, graph, edges, ni, nw, x, y, anchor = toy_model()
    model.eval()
    with torch.no_grad():
        _, embedding = model.retrieval_logits(x)
        actual = study.virtual_retrieval_scores(model, embedding, x, edges)
        i, j = edges.T
        keys = F.normalize((model.anchor_keys[i] + model.anchor_keys[j]) / 2, dim=-1)
        prototypes = (model.prototype_rss[i] + model.prototype_rss[j]) / 2
        distance = torch.cdist(x, prototypes) / math.sqrt(x.shape[1])
        expected = study.HISTORICAL_NEURAL_SCALE * model.logit_scale.exp().clamp(max=100.) * (embedding @ keys.T) - study.HISTORICAL_PROTOTYPE_WEIGHT * distance
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_cached_objective_equals_masked_joint_nll_plus_transport_site_equal():
    base, graph, edges, ni, nw, x, y, anchor = toy_model()
    q = torch.tensor([[.2, .3, .1, .2, .2], [.1, .4, .1, .1, .3], [.3, .2, .1, .2, .2]])
    pack = {"xy": base.anchor_coords[None].expand(3, -1, -1), "sigma": torch.ones(3, 5, 2),
            "ids": torch.arange(5)[None].expand(3, -1), "target": y,
            "building": torch.zeros(3, dtype=torch.long), "floor": torch.zeros(3, dtype=torch.long),
            "weight": torch.tensor([.25, .25, .5])}
    ctx = {"memory": {"building": base.anchor_buildings, "floor": base.anchor_floors}}
    actual = study.objective_from_q(q, study.objective_cache(pack, ctx))
    nll = op.joint_bf_xy_nll(q, pack["ids"], pack["xy"], pack["sigma"], y,
        pack["building"], pack["floor"], base.anchor_buildings, base.anchor_floors)
    risk = op.component_center_transport_risk(q, pack["xy"], y)
    assert torch.allclose(actual, ((nll + .30 * risk) * pack["weight"]).sum())


def test_bf_mass_runner_path_uses_raw_scores_despite_high_temperature():
    base, graph, edges, ni, nw, x, y, anchor = toy_model()
    classes = class_structure(base.anchor_buildings, base.anchor_floors)[1]
    pack = {"scores": torch.tensor([[5., 2., 1., -1., -2.]]), "z": torch.randn(1, 5, 2)}
    ctx = {"graph": graph, "classes": classes}
    _, _, q = study.s1_probability(pack, ctx, {"temperature": 3., "theta": [1., .5]}, .35, .8, "bf_mass")
    raw_q = torch.softmax(pack["scores"], 1)
    for cls in classes.unique():
        mask = classes == cls
        assert torch.allclose(q[:, mask].sum(1), raw_q[:, mask].sum(1), atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA shape-dependent reduction regression")
def test_cuda_appending_virtual_field_keeps_parent_features_bitwise_equal():
    rng = np.random.default_rng(791)
    device = "cuda"
    n, e, aps, batch = 646, 2946, 120, 8
    x = torch.tensor(rng.uniform(-.9, .4, (batch, aps)), dtype=torch.float32, device=device)
    x[:, ::3] = -1.
    seen = torch.ones(aps, dtype=torch.bool, device=device)
    pi = torch.tensor(rng.uniform(.05, .95, (n + e, aps)), dtype=torch.float32, device=device)
    mu = torch.tensor(rng.uniform(-.8, .4, (n + e, aps)), dtype=torch.float32, device=device)
    var = torch.tensor(rng.uniform(.0036, .5, (n + e, aps)), dtype=torch.float32, device=device)
    parents = op.g13_continuous_field_features(x, seen, pi[:n], mu[:n], var[:n], n_physical=n)
    expanded = op.g13_continuous_field_features(x, seen, pi, mu, var, n_physical=n)
    assert torch.equal(expanded[:, :n], parents)
    assert torch.isfinite(expanded).all()
