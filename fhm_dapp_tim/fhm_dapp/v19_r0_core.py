"""Core mathematics for V19-R0 accuracy recovery.

V19-R0 is intentionally narrow. It restores two capabilities that existed in
the strongest pre-V18 systems while retaining the V18.5 explicit-mask model:

1. independent-member epistemic diversity with a wide common anchor consensus;
2. the Bayes spatial-median action for Euclidean localization error.

Radio projection, temperature calibration and spatial-energy regularization are
not part of R0. They are held out so the accuracy contribution of member
diversity and the point-decision rule can be measured cleanly.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    continuous_energy_score_mc,
    spatial_median_from_draws,
)
from fhm_dapp_tim.fhm_dapp.consensus import alpha_consensus_prior
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185AnchorModel


V19_R0_CONFIG: dict[str, object] = {
    "member_count": 16,
    "member_prefixes": (1, 2, 4, 8, 16),
    "member_top_l": 192,
    "consensus_top_k": 192,
    "consensus_alpha": 1.0,
    "consensus_epsilon": 1e-6,
    "fixed_lambda_spatial": 0.0,
    "fixed_epochs": 27,
    "student_df": 4.0,
    "predictive_draws": 64,
    "member_seed_start": 19301,
    "scoring_seed": 19931,
    "reranker_anchor_chunk": 96,
}


def validate_v19_r0_config(
    config: Mapping[str, object] = V19_R0_CONFIG,
) -> None:
    if int(config["member_count"]) != 16:
        raise ValueError("V19-R0 registers exactly 16 members")
    if tuple(map(int, config["member_prefixes"])) != (1, 2, 4, 8, 16):
        raise ValueError("V19-R0 member prefixes changed")
    if int(config["member_top_l"]) != 192:
        raise ValueError("V19-R0 member support L must be 192")
    if int(config["consensus_top_k"]) != 192:
        raise ValueError("V19-R0 consensus support K must be 192")
    if float(config["consensus_alpha"]) != 1.0:
        raise ValueError("V19-R0 fixes alpha=1")
    if not (0.0 < float(config["consensus_epsilon"]) < 1e-3):
        raise ValueError("invalid V19-R0 consensus epsilon")
    if float(config["fixed_lambda_spatial"]) != 0.0:
        raise ValueError("V19-R0 fixes lambda_spatial=0")
    if int(config["fixed_epochs"]) != 27:
        raise ValueError("V19-R0 fixes the inherited V18.5 epoch count at 27")
    if float(config["student_df"]) != 4.0:
        raise ValueError("V19-R0 retains Student-t df=4")
    if int(config["predictive_draws"]) != 64:
        raise ValueError("V19-R0 uses 64 predictive draws per stream")
    if int(config["reranker_anchor_chunk"]) <= 0:
        raise ValueError("reranker anchor chunk must be positive")


def member_seeds(
    config: Mapping[str, object] = V19_R0_CONFIG,
) -> tuple[int, ...]:
    validate_v19_r0_config(config)
    start = int(config["member_seed_start"])
    return tuple(range(start, start + int(config["member_count"])))


@torch.no_grad()
def dense_member_distribution(
    model: V185AnchorModel,
    rss: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    anchor_chunk: int | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate one V18.5 member on the complete anchor dictionary.

    V18.5 used a <=128 proposal set before its learned candidate reranker.
    V19-R0 deliberately removes that inference bottleneck for the accuracy
    recovery experiment. The already-trained reranker is evaluated on every
    anchor identity in deterministic chunks; the mathematical posterior is
    therefore independent of the chunk size.
    """

    if rss.ndim != 2 or observed_mask.shape != rss.shape:
        raise ValueError("rss/observed_mask must share [B,A] shape")
    if observed_mask.dtype != torch.bool:
        raise ValueError("observed_mask must be bool")
    if len(rss) == 0:
        raise ValueError("empty batch")

    invariant_logits, embedding = model.retrieval_logits_v185(
        rss, observed_mask
    )
    anchors = int(model.n_anchors)
    width = (
        int(V19_R0_CONFIG["reranker_anchor_chunk"])
        if anchor_chunk is None
        else int(anchor_chunk)
    )
    if width <= 0:
        raise ValueError("anchor_chunk must be positive")

    logits_parts: list[torch.Tensor] = []
    location_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    for start in range(0, anchors, width):
        stop = min(start + width, anchors)
        index = torch.arange(
            start, stop, device=rss.device, dtype=torch.long
        ).unsqueeze(0).expand(len(rss), -1)
        coarse = invariant_logits[:, start:stop]
        logits_parts.append(
            model.rerank_candidate_logits_v185(
                rss, observed_mask, embedding, index, coarse
            )
        )
        location, scale = model._components(embedding, index)
        location_parts.append(location)
        scale_parts.append(scale)

    dense_logits = torch.cat(logits_parts, dim=1)
    location = torch.cat(location_parts, dim=1)
    scale = torch.cat(scale_parts, dim=1)
    probability = torch.softmax(dense_logits, dim=1)

    if dense_logits.shape != (len(rss), anchors):
        raise RuntimeError("dense member logits have wrong shape")
    if location.shape != (len(rss), anchors, 2):
        raise RuntimeError("dense member locations have wrong shape")
    if scale.shape != location.shape:
        raise RuntimeError("dense member scales have wrong shape")
    if not bool(
        torch.isfinite(probability).all()
        and torch.isfinite(location).all()
        and torch.isfinite(scale).all()
    ):
        raise RuntimeError("nonfinite dense member distribution")
    if bool((scale <= 0).any()):
        raise RuntimeError("nonpositive dense member scale")
    if not torch.allclose(
        probability.sum(dim=1),
        torch.ones(len(rss), device=rss.device, dtype=probability.dtype),
        atol=2e-6,
        rtol=0,
    ):
        raise RuntimeError("dense member posterior is not normalized")

    return {
        "probability": probability,
        "component_location": location,
        "component_scale": scale,
        "embedding": embedding,
        "dense_logits": dense_logits,
    }


def member_top_l(
    probability: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    *,
    top_l: int,
    student_df: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Return V17-style Top-L member support and radial predictive scale."""

    if probability.ndim != 2:
        raise ValueError("probability must be [B,A]")
    if location.shape != (*probability.shape, 2) or scale.shape != location.shape:
        raise ValueError("location/scale must be [B,A,2]")
    width = int(top_l)
    if not 1 <= width <= probability.shape[1]:
        raise ValueError("top_l outside anchor dictionary")
    df = float(student_df)
    if not math.isfinite(df) or df <= 2.0:
        raise ValueError("student_df must exceed 2 for radial variance")

    values, indices = torch.topk(
        probability, width, dim=1, largest=True, sorted=True
    )
    retained = values.sum(dim=1)
    weights = values / retained.unsqueeze(1).clamp_min(1e-30)
    gather = indices.unsqueeze(2).expand(-1, -1, 2)
    selected_location = torch.gather(location, 1, gather)
    selected_scale = torch.gather(scale, 1, gather)
    mean = torch.sum(weights.unsqueeze(2) * selected_location, dim=1)
    component_variance = selected_scale.square() * df / (df - 2.0)
    radial_second = torch.sum(
        weights
        * (
            (selected_location - mean.unsqueeze(1)).square().sum(dim=2)
            + component_variance.sum(dim=2)
        ),
        dim=1,
    )
    radial = radial_second.clamp_min(1e-8).sqrt()
    return {
        "weights": weights,
        "indices": indices,
        "radial_scale": radial,
        "retained_mass": retained,
    }


def consensus_joint_distribution(
    member_top_weights: torch.Tensor,
    member_top_indices: torch.Tensor,
    member_radial_scale: torch.Tensor,
    member_location: torch.Tensor,
    member_scale: torch.Tensor,
    *,
    members: int,
    top_k: int,
    alpha: float = 1.0,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Build the V19-R0 member-anchor joint mixture."""

    if member_top_weights.ndim != 3:
        raise ValueError("member_top_weights must be [M,B,L]")
    if member_top_indices.shape != member_top_weights.shape:
        raise ValueError("member top weights/indices must align")
    if member_radial_scale.shape != member_top_weights.shape[:2]:
        raise ValueError("member radial scale must be [M,B]")
    if member_location.ndim != 4 or member_location.shape[-1] != 2:
        raise ValueError("member_location must be [M,B,A,2]")
    if member_scale.shape != member_location.shape:
        raise ValueError("member_scale must align with member_location")
    if member_location.shape[:2] != member_top_weights.shape[:2]:
        raise ValueError("member rows do not align")
    count = int(members)
    if not 1 <= count <= member_top_weights.shape[0]:
        raise ValueError("members outside available prefix")
    anchors = int(member_location.shape[2])
    width = int(top_k)
    if not 1 <= width <= anchors:
        raise ValueError("top_k outside anchor dictionary")

    priors = member_top_weights[:count]
    indices = member_top_indices[:count]
    radial = member_radial_scale[:count]
    reliability = radial.square().clamp_min(1e-8).reciprocal()
    reliability = reliability / reliability.sum(
        dim=0, keepdim=True
    ).clamp_min(1e-30)

    consensus_indices, consensus_probability = alpha_consensus_prior(
        priors,
        indices,
        reliability,
        anchors,
        width,
        epsilon=float(epsilon),
        alpha=float(alpha),
    )

    locations = member_location[:count]
    scales = member_scale[:count]
    gather = consensus_indices.unsqueeze(0).unsqueeze(3).expand(
        count, -1, -1, 2
    )
    selected_location = torch.gather(locations, 2, gather)
    selected_scale = torch.gather(scales, 2, gather)
    joint = (
        reliability.transpose(0, 1).unsqueeze(2)
        * consensus_probability.unsqueeze(1)
    )
    probability = joint.reshape(len(consensus_probability), -1)
    location = selected_location.permute(1, 0, 2, 3).reshape(
        len(consensus_probability), -1, 2
    )
    scale = selected_scale.permute(1, 0, 2, 3).reshape(
        len(consensus_probability), -1, 2
    )

    if not torch.allclose(
        probability.sum(dim=1),
        torch.ones(
            len(probability),
            dtype=probability.dtype,
            device=probability.device,
        ),
        atol=2e-6,
        rtol=0,
    ):
        raise RuntimeError("V19 joint posterior is not normalized")
    if not bool(
        torch.isfinite(probability).all()
        and torch.isfinite(location).all()
        and torch.isfinite(scale).all()
    ):
        raise RuntimeError("nonfinite V19 joint distribution")
    return {
        "probability": probability,
        "component_location": location,
        "component_scale": scale,
        "member_weight": reliability,
        "consensus_anchor_indices": consensus_indices,
        "consensus_anchor_probability": consensus_probability,
    }


def student_t_mixture_nll(
    probability: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    *,
    degrees_freedom: float = 4.0,
) -> torch.Tensor:
    """Independent-axis Student-t mixture NLL."""

    if probability.ndim != 2:
        raise ValueError("probability must be [B,C]")
    if location.shape != (*probability.shape, 2) or scale.shape != location.shape:
        raise ValueError("component tensors must be [B,C,2]")
    if target.shape != (len(probability), 2):
        raise ValueError("target must be [B,2]")
    df = float(degrees_freedom)
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("invalid degrees_freedom")
    z = (target.unsqueeze(1) - location) / scale
    normalizer = (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
    )
    component = (
        normalizer
        - torch.log(scale)
        - 0.5 * (df + 1.0) * torch.log1p(z.square() / df)
    ).sum(dim=2)
    return -torch.logsumexp(
        probability.clamp_min(1e-30).log() + component, dim=1
    )


def predictive_actions_and_scores(
    probability: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    randomness: Mapping[str, torch.Tensor | float | int],
    *,
    degrees_freedom: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Score one mixture with mean and predictive spatial-median actions."""

    energy_score, first, second = continuous_energy_score_mc(
        probability, location, scale, target, randomness
    )
    mean = torch.sum(probability.unsqueeze(2) * location, dim=1)
    spatial = spatial_median_from_draws(first, second)
    mean_error = torch.linalg.vector_norm(mean - target, dim=1)
    spatial_error = torch.linalg.vector_norm(spatial - target, dim=1)
    nll = student_t_mixture_nll(
        probability,
        location,
        scale,
        target,
        degrees_freedom=degrees_freedom,
    )
    return {
        "posterior_mean": mean,
        "predictive_spatial_median": spatial,
        "posterior_mean_error": mean_error,
        "predictive_spatial_median_error": spatial_error,
        "energy_score": energy_score,
        "nll": nll,
        "first_draws": first,
        "second_draws": second,
    }


__all__ = [
    "V19_R0_CONFIG",
    "consensus_joint_distribution",
    "dense_member_distribution",
    "member_seeds",
    "member_top_l",
    "predictive_actions_and_scores",
    "student_t_mixture_nll",
    "validate_v19_r0_config",
]
