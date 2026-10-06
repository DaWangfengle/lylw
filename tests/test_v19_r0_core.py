from __future__ import annotations

import math

import torch

from fhm_dapp_tim.fhm_dapp.v19_r0_core import (
    V19_R0_CONFIG,
    consensus_joint_distribution,
    dense_member_distribution,
    member_seeds,
    member_top_l,
    student_t_mixture_nll,
    validate_v19_r0_config,
)


class _FakeDenseModel:
    n_anchors = 5

    def retrieval_logits_v185(self, rss, observed_mask):
        logits = torch.tensor(
            [[0.2, 1.1, -0.4, 0.7, 0.0]],
            dtype=rss.dtype,
            device=rss.device,
        ).expand(len(rss), -1)
        embedding = torch.ones(
            len(rss), 2, dtype=rss.dtype, device=rss.device
        )
        return logits, embedding

    def rerank_candidate_logits_v185(
        self, rss, observed_mask, embedding, candidate_indices, coarse_logits
    ):
        return coarse_logits + candidate_indices.to(coarse_logits) * 0.03

    def _components(self, embedding, candidate_indices):
        value = candidate_indices.to(embedding.dtype)
        location = torch.stack([value, -value], dim=2)
        scale = torch.ones_like(location) * 0.5
        return location, scale


def test_v19_r0_registered_constants_and_member_seeds():
    validate_v19_r0_config()
    assert V19_R0_CONFIG["member_prefixes"] == (1, 2, 4, 8, 16)
    assert V19_R0_CONFIG["member_top_l"] == 192
    assert V19_R0_CONFIG["consensus_top_k"] == 192
    assert V19_R0_CONFIG["consensus_alpha"] == 1.0
    assert V19_R0_CONFIG["fixed_lambda_spatial"] == 0.0
    assert V19_R0_CONFIG["fixed_epochs"] == 27
    seeds = member_seeds()
    assert len(seeds) == 16
    assert seeds[0] == 185511
    assert seeds[1:] == tuple(range(19301, 19316))


def test_dense_member_distribution_is_anchor_chunk_invariant():
    model = _FakeDenseModel()
    rss = torch.zeros(3, 4)
    observed = torch.ones_like(rss, dtype=torch.bool)
    first = dense_member_distribution(
        model, rss, observed, anchor_chunk=1
    )
    second = dense_member_distribution(
        model, rss, observed, anchor_chunk=3
    )
    for key in (
        "probability",
        "component_location",
        "component_scale",
        "dense_logits",
    ):
        assert torch.allclose(first[key], second[key], atol=0, rtol=0)
    assert torch.allclose(
        first["probability"].sum(dim=1),
        torch.ones(3),
        atol=2e-6,
        rtol=0,
    )
    expected = torch.softmax(
        torch.tensor([[0.2, 1.1, -0.4, 0.7, 0.0]]), dim=1
    ).expand(3, -1)
    assert torch.allclose(
        first["probability"], expected, atol=0, rtol=0
    )
    assert first["candidate_reranker_applied"] is False


def test_member_top_l_matches_v17_radial_second_moment():
    probability = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
    location = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.float64
    )
    scale = torch.ones_like(location)
    result = member_top_l(
        probability, location, scale, top_l=2, student_df=4.0
    )
    assert torch.allclose(
        result["weights"], probability, atol=1e-12, rtol=0
    )
    assert math.isclose(
        float(result["radial_scale"][0]),
        math.sqrt(4.96),
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_e1_full_support_consensus_is_identity():
    weights = torch.tensor(
        [[[0.6, 0.3, 0.1]]], dtype=torch.float64
    )
    indices = torch.tensor([[[0, 1, 2]]], dtype=torch.long)
    radial = torch.tensor([[2.0]], dtype=torch.float64)
    locations = torch.tensor(
        [[[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]],
        dtype=torch.float64,
    )
    scales = torch.ones_like(locations)
    result = consensus_joint_distribution(
        weights,
        indices,
        radial,
        locations,
        scales,
        members=1,
        top_k=3,
        alpha=1.0,
        epsilon=1e-8,
    )
    assert torch.allclose(
        result["probability"], weights[0], atol=1e-10, rtol=0
    )
    assert torch.equal(
        result["consensus_anchor_indices"],
        torch.tensor([[0, 1, 2]]),
    )


def test_joint_consensus_preserves_member_specific_components():
    weights = torch.tensor(
        [
            [[0.8, 0.2]],
            [[0.8, 0.2]],
        ],
        dtype=torch.float64,
    )
    indices = torch.tensor(
        [
            [[0, 1]],
            [[0, 1]],
        ],
        dtype=torch.long,
    )
    radial = torch.ones(2, 1, dtype=torch.float64)
    locations = torch.tensor(
        [
            [[[0.0, 0.0], [1.0, 0.0]]],
            [[[10.0, 0.0], [11.0, 0.0]]],
        ],
        dtype=torch.float64,
    )
    scales = torch.ones_like(locations)
    result = consensus_joint_distribution(
        weights,
        indices,
        radial,
        locations,
        scales,
        members=2,
        top_k=2,
        alpha=1.0,
    )
    assert result["probability"].shape == (1, 4)
    assert result["component_location"].shape == (1, 4, 2)
    x = set(result["component_location"][0, :, 0].tolist())
    assert x == {0.0, 1.0, 10.0, 11.0}
    assert torch.allclose(
        result["member_weight"],
        torch.full((2, 1), 0.5, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_student_t_mixture_nll_matches_single_component_formula():
    probability = torch.ones(1, 1, dtype=torch.float64)
    location = torch.zeros(1, 1, 2, dtype=torch.float64)
    scale = torch.ones_like(location)
    target = torch.zeros(1, 2, dtype=torch.float64)
    nll = student_t_mixture_nll(
        probability,
        location,
        scale,
        target,
        degrees_freedom=4.0,
    )
    normalizer = (
        math.lgamma(2.5)
        - math.lgamma(2.0)
        - 0.5 * math.log(4.0 * math.pi)
    )
    assert math.isclose(
        float(nll[0]), -2.0 * normalizer, rel_tol=0, abs_tol=1e-12
    )
