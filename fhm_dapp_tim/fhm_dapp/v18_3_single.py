"""V18.3 single-member hierarchical robust evidence consensus.

V18.3 keeps the successful V18.2 multi-source candidate union and removes the
failure mode observed in V18.2: an unconstrained learned six-way gate collapsed
onto the learned reranker.

The model forms the same six interpretable evidence opinions, but organizes
them into three semantic families:

Representation = invariant neural + learned reranker
Radio          = raw prototype + robust marked radio
Structure      = soft floor + geometry manifold

There is no trainable evidence gate.  Within-family and between-family weights
are deterministic reliability functions of entropy, disagreement, and
source-specific mismatch.  Every family receives a positive inference weight
floor.  During training, evidence-family dropout removes at most one family per
scan, preventing the learned representation family from becoming a shortcut.

A disagreement-aware temperature softens the posterior when the three families
contradict one another.  The deployed model remains one checkpoint with one
Student-t residual head: no teacher, no distillation, no member axis, and no
ensemble.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    build_delaunay_geometry,
    profile_shared_affine_marked_energy,
)
from fhm_dapp_tim.fhm_dapp.dapp import (
    anchor_ap_reliability,
    initial_support_manifold_energy,
)
from fhm_dapp_tim.fhm_dapp.model import (
    AnchorStudentTModel,
    gradient_reverse,
)
from fhm_dapp_tim.fhm_dapp.train_anchor import _ap_building_owner
from fhm_dapp_tim.fhm_dapp.v18_single import delaunay_neighbor_targets
from fhm_dapp_tim.fhm_dapp.v18_2_single import (
    EVIDENCE_SOURCES,
    _candidate_floor_probability,
    _hierarchy_prediction,
    _standardize,
    _support_recall_mass,
    _weighted_target_mass,
    candidate_union,
    prototype_retrieval_score,
)


VISIBLE_THRESHOLD = -0.999

EVIDENCE_FAMILIES = (
    "representation",
    "radio",
    "structure",
)
FAMILY_SOURCE_INDICES = (
    (0, 2),  # invariant, learned
    (1, 3),  # prototype, marked_radio
    (4, 5),  # floor, geometry
)

V183_CONFIG = {
    # Backbone.
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "max_visible": 64,
    "dropout": 0.10,
    # Predictive family.
    "student_df": 4.0,
    "min_scale_m": 0.25,
    "max_residual_m": 30.0,
    "residual_experts": 1,
    # Neural representation and learned candidate evidence.
    "scan_feature_mode": "bounded_maximal_invariant",
    "prototype_metric": "none",
    "candidate_reranker": True,
    "candidate_reranker_mode": "summary",
    "reranker_anchor_decoder": False,
    # Keep the successful V18.2 wide candidate union.
    "invariant_retrieval_k": 32,
    "prototype_retrieval_k": 32,
    "floor_retrieval_k": 16,
    "geometry_seed_k": 8,
    "candidate_union_size": 96,
    "posterior_top_k": 64,
    # Alpha=1 remains the opinion-pool law, but now hierarchically structured.
    "evidence_alpha": 1.0,
    "geometry_manifold_sigma_m": 6.0,
    "geometry_manifold_margin_m": 2.0,
    # Bounded within-family reliability.  For two members, each source always
    # receives at least 30% of its family's mass and at most 70%.
    "within_family_min_source_weight": 0.30,
    "within_family_reliability_temperature": 0.50,
    # Between-family reliability.  Every family receives >=15% at inference.
    "family_min_weight": 0.15,
    "family_reliability_temperature": 0.50,
    "family_entropy_penalty": 1.0,
    "family_disagreement_penalty": 1.0,
    "radio_mismatch_penalty": 1.0,
    # Train-time regularization against evidence-family shortcuts.
    "family_dropout_probability": 0.30,
    # Model-internal adaptive softening from family disagreement.
    "disagreement_temperature_strength": 1.0,
    "disagreement_temperature_max": 2.0,
    # Robust marked-radio opinion.
    "radio_lambda": 1.0,
    "radio_profile_steps": 6,
    "radio_profile_learning_rate": 0.20,
    "radio_profile_offset_max": 0.20,
    "radio_profile_log_scale_max": 0.25,
    "radio_profile_offset_weight": 0.10,
    "radio_profile_log_scale_weight": 0.10,
    # Optimization.
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "maximum_epochs": 60,
    "minimum_epochs": 12,
    "patience": 8,
    "minimum_improvement_energy_m": 0.01,
    "energy_tie_margin_m": 0.05,
    # Robustness.
    "rss_offset_aug_db": 4.0,
    "rss_scale_aug": 0.08,
    "ap_dropout": 0.10,
    "domain_adversarial_weight": 0.05,
    # Auxiliary losses.  Ranking is deliberately weaker than V18.2 because
    # V18.2 showed that sharp ranking without robust consensus harms transfer.
    "invariant_aux_weight": 0.25,
    "reranker_aux_weight": 0.25,
    "hierarchy_weight": 0.50,
    "coordinate_weight": 0.05,
    "ranking_weight": 0.05,
    "ranking_margin": 0.25,
    # Inference.
    "condition_building": False,
    "deterministic_building": True,
}


def validate_v183_config(
    config: Mapping[str, object] = V183_CONFIG,
) -> None:
    """Fail closed when the registered V18.3 model law drifts."""

    if int(config["residual_experts"]) != 1:
        raise ValueError("V18.3 requires exactly one residual expert")
    if str(config["scan_feature_mode"]) != "bounded_maximal_invariant":
        raise ValueError("V18.3 requires the bounded maximal-invariant encoder")
    if str(config["prototype_metric"]) != "none":
        raise ValueError(
            "V18.3 keeps raw RSS as an explicit evidence source, not a "
            "hidden neural retrieval metric"
        )
    if not bool(config["candidate_reranker"]):
        raise ValueError("V18.3 requires the learned representation opinion")
    if bool(config["reranker_anchor_decoder"]):
        raise ValueError("V18.3 requires one shared Student-t residual head")
    if float(config["evidence_alpha"]) != 1.0:
        raise ValueError("V18.3 registers hierarchical alpha=1 consensus")

    union = int(config["candidate_union_size"])
    posterior = int(config["posterior_top_k"])
    if union <= posterior:
        raise ValueError("candidate_union_size must exceed posterior_top_k")
    primary_quota = (
        int(config["invariant_retrieval_k"])
        + int(config["prototype_retrieval_k"])
        + int(config["floor_retrieval_k"])
    )
    if primary_quota > union:
        raise ValueError(
            "candidate_union_size must contain all registered primary "
            "source quotas"
        )

    source_floor = float(config["within_family_min_source_weight"])
    if not 0.0 < source_floor < 0.5:
        raise ValueError("within-family source weight floor must lie in (0,0.5)")
    family_floor = float(config["family_min_weight"])
    if not 0.0 < family_floor < 1.0 / len(EVIDENCE_FAMILIES):
        raise ValueError("family weight floor must lie in (0,1/3)")
    if float(config["within_family_reliability_temperature"]) <= 0.0:
        raise ValueError("within-family reliability temperature must be positive")
    if float(config["family_reliability_temperature"]) <= 0.0:
        raise ValueError("family reliability temperature must be positive")
    if not 0.0 <= float(config["family_dropout_probability"]) < 1.0:
        raise ValueError("family dropout probability must lie in [0,1)")
    if float(config["disagreement_temperature_strength"]) < 0.0:
        raise ValueError("disagreement temperature strength cannot be negative")
    if float(config["disagreement_temperature_max"]) < 1.0:
        raise ValueError("disagreement temperature maximum must be >= 1")
    if float(config["radio_lambda"]) <= 0.0:
        raise ValueError("radio_lambda must be positive")
    if int(config["radio_profile_steps"]) < 1:
        raise ValueError("radio_profile_steps must be positive")
    if float(config["ranking_weight"]) < 0.0:
        raise ValueError("ranking_weight cannot be negative")


class StructuredEvidenceAnchorModel(AnchorStudentTModel):
    """One checkpoint with deterministic structured evidence reliability."""


def build_v183_model(
    *,
    memory: Mapping[str, torch.Tensor],
    n_aps: int,
    n_buildings: int,
    n_devices: int,
    device: torch.device,
    config: Mapping[str, object] = V183_CONFIG,
) -> tuple[
    StructuredEvidenceAnchorModel,
    dict[str, object],
    torch.Tensor,
    torch.Tensor,
]:
    validate_v183_config(config)
    required = {
        "prototype_rss",
        "detection_prob",
        "rss_center",
        "rss_scale",
        "coords",
        "building",
        "floor",
        "counts",
    }
    missing = required - set(memory)
    if missing:
        raise KeyError(f"anchor memory misses V18.3 fields: {sorted(missing)}")

    geometry = build_delaunay_geometry(
        memory["coords"],
        memory["building"],
        memory["floor"],
    )
    neighbour_indices, neighbour_weights = delaunay_neighbor_targets(geometry)
    if neighbour_indices.shape[1] > int(config["candidate_union_size"]):
        raise ValueError(
            "candidate_union_size cannot contain the widest Delaunay target"
        )
    anchor_id = torch.arange(
        len(neighbour_indices),
        dtype=torch.long,
    ).unsqueeze(1)
    if bool(
        (
            (neighbour_indices == anchor_id)
            & (neighbour_weights > 0.0)
        ).any()
    ):
        raise ValueError(
            "V18.3 leave-self-out geometry requires at least two anchors "
            "per building-floor stratum"
        )

    seen = (memory["prototype_rss"] > VISIBLE_THRESHOLD).any(dim=0)
    owner = _ap_building_owner(
        memory["prototype_rss"],
        memory["building"],
        int(n_buildings),
    )
    model = StructuredEvidenceAnchorModel(
        n_aps=int(n_aps),
        anchor_coords=memory["coords"],
        anchor_buildings=memory["building"],
        anchor_floors=memory["floor"],
        seen_ap_mask=seen,
        prototype_rss=memory["prototype_rss"],
        ap_building_owner=owner,
        n_devices=int(n_devices),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        max_visible=int(config["max_visible"]),
        dropout=float(config["dropout"]),
        visible_threshold=VISIBLE_THRESHOLD,
        student_df=float(config["student_df"]),
        max_residual_m=float(config["max_residual_m"]),
        min_scale_m=float(config["min_scale_m"]),
        residual_experts=int(config["residual_experts"]),
        scan_feature_mode=str(config["scan_feature_mode"]),
        prototype_metric=str(config["prototype_metric"]),
        anchor_detection_prob=memory["detection_prob"],
        anchor_rss_center=memory["rss_center"],
        anchor_rss_scale=memory["rss_scale"],
        anchor_sample_count=memory["counts"].float(),
        candidate_reranker=bool(config["candidate_reranker"]),
        candidate_reranker_mode=str(config["candidate_reranker_mode"]),
        reranker_anchor_decoder=bool(config["reranker_anchor_decoder"]),
        local_residual_transform=geometry["residual_transform"],
    ).to(device)
    model.initialize_anchor_keys(memory["prototype_rss"].to(device))

    reliability = anchor_ap_reliability(
        memory["detection_prob"].float(),
        memory["rss_center"].float(),
        memory["rss_scale"].float(),
    )
    model.register_buffer(
        "evidence_ap_reliability",
        reliability.to(device),
        persistent=False,
    )
    model.register_buffer(
        "evidence_geometry_neighbors",
        neighbour_indices.to(device),
        persistent=False,
    )
    return (
        model,
        geometry,
        neighbour_indices.to(device),
        neighbour_weights.to(device),
    )


def _normalized_entropy(probability: torch.Tensor) -> torch.Tensor:
    entropy = -torch.sum(
        probability * probability.clamp(min=1e-12).log(),
        dim=1,
    )
    return entropy / math.log(max(probability.shape[1], 2))


def _js_divergence(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Jensen-Shannon divergence normalized to [0,1]."""

    first = first.clamp(min=1e-12)
    second = second.clamp(min=1e-12)
    midpoint = 0.5 * (first + second)
    kl_first = torch.sum(first * (first.log() - midpoint.log()), dim=1)
    kl_second = torch.sum(second * (second.log() - midpoint.log()), dim=1)
    return (0.5 * (kl_first + kl_second) / math.log(2.0)).clamp(
        min=0.0,
        max=1.0,
    )


def _bounded_pair_consensus(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    min_weight: float,
    reliability_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two opinions while guaranteeing both a nonzero role."""

    entropy = torch.stack(
        [_normalized_entropy(first), _normalized_entropy(second)],
        dim=1,
    )
    confidence = 1.0 - entropy
    raw = F.softmax(
        confidence / float(reliability_temperature),
        dim=1,
    )
    weights = float(min_weight) + (
        1.0 - 2.0 * float(min_weight)
    ) * raw
    family = (
        weights[:, 0:1] * first
        + weights[:, 1:2] * second
    )
    family = family / family.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return family, weights


def _family_disagreement(
    family_probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-family and global pairwise JS disagreement."""

    pair01 = _js_divergence(
        family_probability[:, 0, :],
        family_probability[:, 1, :],
    )
    pair02 = _js_divergence(
        family_probability[:, 0, :],
        family_probability[:, 2, :],
    )
    pair12 = _js_divergence(
        family_probability[:, 1, :],
        family_probability[:, 2, :],
    )
    per_family = torch.stack(
        [
            0.5 * (pair01 + pair02),
            0.5 * (pair01 + pair12),
            0.5 * (pair02 + pair12),
        ],
        dim=1,
    )
    global_disagreement = (pair01 + pair02 + pair12) / 3.0
    return per_family, global_disagreement


def _radio_mismatch(
    posterior_radio_energy: torch.Tensor,
    style_offset: torch.Tensor,
    style_scale: torch.Tensor,
    *,
    config: Mapping[str, object],
) -> torch.Tensor:
    """Dimensionless radio mismatch from likelihood and affine correction."""

    energy_term = 1.0 - torch.exp(
        -posterior_radio_energy.clamp(min=0.0)
    )
    offset_term = (
        style_offset.abs()
        / float(config["radio_profile_offset_max"])
    ).clamp(max=1.0)
    gain_term = (
        torch.log(style_scale.clamp(min=1e-6)).abs()
        / float(config["radio_profile_log_scale_max"])
    ).clamp(max=1.0)
    style_term = 0.5 * (offset_term + gain_term)
    return 0.5 * (energy_term + style_term)


def _bounded_family_weights(
    family_probability: torch.Tensor,
    radio_mismatch: torch.Tensor,
    *,
    config: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Compute deterministic reliability weights with a positive family floor."""

    family_entropy = torch.stack(
        [
            _normalized_entropy(family_probability[:, family_index, :])
            for family_index in range(len(EVIDENCE_FAMILIES))
        ],
        dim=1,
    )
    family_disagreement, global_disagreement = _family_disagreement(
        family_probability
    )
    score = (
        -float(config["family_entropy_penalty"]) * family_entropy
        -float(config["family_disagreement_penalty"])
        * family_disagreement
    )
    score = score.clone()
    score[:, 1] = (
        score[:, 1]
        - float(config["radio_mismatch_penalty"]) * radio_mismatch
    )

    raw = F.softmax(
        score / float(config["family_reliability_temperature"]),
        dim=1,
    )
    floor = float(config["family_min_weight"])
    weights = floor + (
        1.0 - len(EVIDENCE_FAMILIES) * floor
    ) * raw
    return {
        "weights": weights,
        "entropy": family_entropy,
        "disagreement": family_disagreement,
        "global_disagreement": global_disagreement,
        "reliability_score": score,
    }


def _apply_family_dropout(
    weights: torch.Tensor,
    *,
    probability: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop at most one family per scan during training."""

    keep = torch.ones_like(weights)
    if training and float(probability) > 0.0:
        batch = len(weights)
        drop_event = torch.rand(
            batch,
            device=weights.device,
        ) < float(probability)
        drop_family = torch.randint(
            low=0,
            high=weights.shape[1],
            size=(batch,),
            device=weights.device,
        )
        rows = torch.arange(batch, device=weights.device)[drop_event]
        keep[rows, drop_family[drop_event]] = 0.0

    dropped = weights * keep
    dropped = dropped / dropped.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return dropped, keep


def _temperature_soften(
    probability: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    logp = probability.clamp(min=1e-12).log()
    softened = F.softmax(
        logp / temperature.unsqueeze(1),
        dim=1,
    )
    return softened


def structured_evidence_opinions(
    model: StructuredEvidenceAnchorModel,
    rss: torch.Tensor,
    embedding: torch.Tensor,
    invariant_logits: torch.Tensor,
    prototype_score: torch.Tensor,
    union: Mapping[str, torch.Tensor],
    *,
    training: bool,
    config: Mapping[str, object] = V183_CONFIG,
) -> dict[str, torch.Tensor]:
    """Build six source opinions and robust three-family consensus."""

    validate_v183_config(config)
    candidate_indices = union["candidate_indices"]
    inv_score = torch.gather(
        invariant_logits,
        1,
        candidate_indices,
    )
    proto_score = torch.gather(
        prototype_score,
        1,
        candidate_indices,
    )
    learned_score = model.rerank_candidate_logits(
        rss,
        embedding,
        candidate_indices,
        inv_score,
    )

    preliminary = (
        F.softmax(inv_score.detach(), dim=1)
        + F.softmax(_standardize(proto_score.detach()), dim=1)
        + F.softmax(learned_score.detach(), dim=1)
    ) / 3.0

    with torch.enable_grad():
        radio_energy, style = profile_shared_affine_marked_energy(
            rss.detach(),
            candidate_indices.detach(),
            {
                "detection_probability": model.anchor_detection_prob.detach(),
                "rss_center": model.anchor_rss_center.detach(),
                "rss_scale": model.anchor_rss_scale.detach(),
            },
            preliminary,
            degrees_freedom=float(config["student_df"]),
            reliability=model.evidence_ap_reliability.detach(),
            steps=int(config["radio_profile_steps"]),
            learning_rate=float(config["radio_profile_learning_rate"]),
            offset_max=float(config["radio_profile_offset_max"]),
            log_scale_max=float(config["radio_profile_log_scale_max"]),
            offset_weight=float(config["radio_profile_offset_weight"]),
            log_scale_weight=float(config["radio_profile_log_scale_weight"]),
        )
    radio_energy = radio_energy.to(inv_score)
    style_offset = style["style_offset"].to(inv_score)
    style_scale = style["style_scale"].to(inv_score)

    floor_probability, _floor_logp = _candidate_floor_probability(
        model,
        invariant_logits,
        candidate_indices,
    )
    candidate_locations = model.anchor_coords[candidate_indices]
    geometry_energy = initial_support_manifold_energy(
        preliminary,
        candidate_locations,
        sigma=float(config["geometry_manifold_sigma_m"]),
        margin=float(config["geometry_manifold_margin_m"]),
    )

    source_probability = torch.stack(
        [
            F.softmax(inv_score, dim=1),
            F.softmax(_standardize(proto_score), dim=1),
            F.softmax(learned_score, dim=1),
            F.softmax(
                -radio_energy / float(config["radio_lambda"]),
                dim=1,
            ),
            floor_probability,
            F.softmax(-geometry_energy, dim=1),
        ],
        dim=1,
    )
    source_entropy = torch.stack(
        [
            _normalized_entropy(source_probability[:, source_index, :])
            for source_index in range(len(EVIDENCE_SOURCES))
        ],
        dim=1,
    )

    family_parts = []
    within_weights = []
    for first_index, second_index in FAMILY_SOURCE_INDICES:
        family, weights = _bounded_pair_consensus(
            source_probability[:, first_index, :],
            source_probability[:, second_index, :],
            min_weight=float(config["within_family_min_source_weight"]),
            reliability_temperature=float(
                config["within_family_reliability_temperature"]
            ),
        )
        family_parts.append(family)
        within_weights.append(weights)
    family_probability = torch.stack(family_parts, dim=1)
    within_family_weights = torch.stack(within_weights, dim=1)

    posterior_radio_energy = torch.sum(
        preliminary * radio_energy,
        dim=1,
    )
    radio_mismatch = _radio_mismatch(
        posterior_radio_energy,
        style_offset,
        style_scale,
        config=config,
    )
    reliability = _bounded_family_weights(
        family_probability,
        radio_mismatch,
        config=config,
    )
    inference_family_weights = reliability["weights"]
    active_family_weights, family_keep_mask = _apply_family_dropout(
        inference_family_weights,
        probability=float(config["family_dropout_probability"]),
        training=training,
    )

    base_consensus = torch.sum(
        active_family_weights.unsqueeze(2) * family_probability,
        dim=1,
    )
    base_consensus = base_consensus / base_consensus.sum(
        dim=1,
        keepdim=True,
    ).clamp(min=1e-12)

    disagreement_temperature = (
        1.0
        + float(config["disagreement_temperature_strength"])
        * reliability["global_disagreement"]
    ).clamp(
        min=1.0,
        max=float(config["disagreement_temperature_max"]),
    )
    consensus = _temperature_soften(
        base_consensus,
        disagreement_temperature,
    )

    return {
        "candidate_indices": candidate_indices,
        "candidate_locations": candidate_locations,
        "invariant_score": inv_score,
        "prototype_score": proto_score,
        "learned_score": learned_score,
        "radio_energy": radio_energy,
        "geometry_energy": geometry_energy,
        "source_probability": source_probability,
        "source_entropy": source_entropy,
        "family_probability": family_probability,
        "within_family_weights": within_family_weights,
        "family_weights": active_family_weights,
        "inference_family_weights": inference_family_weights,
        "family_keep_mask": family_keep_mask,
        "family_entropy": reliability["entropy"],
        "family_disagreement": reliability["disagreement"],
        "family_reliability_score": reliability["reliability_score"],
        "global_disagreement": reliability["global_disagreement"],
        "disagreement_temperature": disagreement_temperature,
        "radio_mismatch": radio_mismatch,
        "posterior_radio_energy": posterior_radio_energy,
        "base_consensus_probability": base_consensus,
        "consensus_probability": consensus,
        "style_offset": style_offset,
        "style_scale": style_scale,
    }


def v183_training_loss(
    model: StructuredEvidenceAnchorModel,
    rss: torch.Tensor,
    target_coords: torch.Tensor,
    target_anchor: torch.Tensor,
    device_ids: torch.Tensor,
    neighbour_indices: torch.Tensor,
    neighbour_weights: torch.Tensor,
    *,
    config: Mapping[str, object] = V183_CONFIG,
) -> dict[str, torch.Tensor]:
    """Train one structured-consensus posterior under leave-anchor-out geometry."""

    validate_v183_config(config)
    model_rss = model.augment_scan_style(
        rss,
        offset_std_db=float(config["rss_offset_aug_db"]),
        scale_std=float(config["rss_scale_aug"]),
    )
    invariant_logits, embedding = model.retrieval_logits(
        model_rss,
        ap_dropout=float(config["ap_dropout"]),
    )
    prototype_score = prototype_retrieval_score(model, model_rss)

    target_indices = neighbour_indices[target_anchor]
    target_weights = neighbour_weights[target_anchor].to(invariant_logits)
    if target_indices.shape != target_weights.shape:
        raise ValueError("Delaunay target indices and weights disagree")
    if not torch.allclose(
        target_weights.sum(dim=1),
        torch.ones(len(target_weights), device=target_weights.device),
        atol=1e-5,
    ):
        raise ValueError("V18.3 geometry targets are not normalized")

    natural_union = candidate_union(
        model,
        invariant_logits.detach(),
        prototype_score.detach(),
        force_indices=None,
        config=config,
    )
    forced_union = candidate_union(
        model,
        invariant_logits,
        prototype_score,
        force_indices=target_indices,
        config=config,
    )
    evidence = structured_evidence_opinions(
        model,
        model_rss,
        embedding,
        invariant_logits,
        prototype_score,
        forced_union,
        training=True,
        config=config,
    )

    candidate_indices = evidence["candidate_indices"]
    candidate_target = _weighted_target_mass(
        candidate_indices,
        target_indices,
        target_weights,
    )
    if bool((candidate_target.sum(dim=1) <= 0.999).any()):
        raise RuntimeError("V18.3 forced union lost geometry target mass")

    consensus = evidence["consensus_probability"]
    row_anchor = -torch.sum(
        candidate_target * consensus.clamp(min=1e-12).log(),
        dim=1,
    )

    geometry_location, geometry_scale = model._components(
        embedding,
        target_indices,
    )
    log_coordinate = model.student_log_prob(
        target_coords,
        geometry_location,
        geometry_scale,
    )
    row_coordinate = -torch.sum(
        target_weights * log_coordinate,
        dim=1,
    )
    joint_geometry_loss = (row_anchor + row_coordinate).mean()

    invariant_logp = F.log_softmax(invariant_logits, dim=1)
    invariant_aux = -torch.sum(
        target_weights
        * torch.gather(invariant_logp, 1, target_indices),
        dim=1,
    ).mean()
    reranker_aux = -torch.sum(
        candidate_target
        * F.log_softmax(evidence["learned_score"], dim=1),
        dim=1,
    ).mean()

    target_building = model.anchor_buildings[target_anchor]
    target_floor = model.anchor_floors[target_anchor]
    hierarchy_loss = (
        0.25
        * F.cross_entropy(
            model._marginal_logits(
                invariant_logits,
                model.anchor_buildings,
            ),
            target_building,
        )
        + F.cross_entropy(
            model._marginal_logits(
                invariant_logits,
                model.anchor_floors,
            ),
            target_floor,
        )
    )

    candidate_location, _candidate_scale = model._components(
        embedding,
        candidate_indices,
    )
    point_prediction = torch.sum(
        consensus.unsqueeze(2) * candidate_location,
        dim=1,
    )
    coordinate_loss = torch.linalg.vector_norm(
        point_prediction - target_coords,
        dim=1,
    ).mean()

    log_consensus = consensus.clamp(min=1e-12).log()
    positive = torch.sum(candidate_target * log_consensus, dim=1)
    hard_negative = log_consensus.masked_fill(
        candidate_target > 0.0,
        float("-inf"),
    ).max(dim=1).values
    ranking_loss = F.relu(
        float(config["ranking_margin"]) - positive + hard_negative
    ).mean()

    domain_loss = joint_geometry_loss.new_zeros(())
    if (
        model.device_head is not None
        and float(config["domain_adversarial_weight"]) > 0.0
    ):
        domain_loss = F.cross_entropy(
            model.device_head(gradient_reverse(embedding)),
            device_ids,
        )

    loss = (
        joint_geometry_loss
        + float(config["invariant_aux_weight"]) * invariant_aux
        + float(config["reranker_aux_weight"]) * reranker_aux
        + float(config["hierarchy_weight"]) * hierarchy_loss
        + float(config["coordinate_weight"]) * coordinate_loss
        + float(config["ranking_weight"]) * ranking_loss
        + float(config["domain_adversarial_weight"]) * domain_loss
    )

    natural_candidate = natural_union["candidate_indices"]
    return {
        "loss": loss,
        "joint_geometry_loss": joint_geometry_loss.detach(),
        "fused_anchor_nll": row_anchor.mean().detach(),
        "coordinate_nll": row_coordinate.mean().detach(),
        "invariant_aux_loss": invariant_aux.detach(),
        "reranker_aux_loss": reranker_aux.detach(),
        "hierarchy_loss": hierarchy_loss.detach(),
        "coordinate_loss": coordinate_loss.detach(),
        "ranking_loss": ranking_loss.detach(),
        "domain_loss": domain_loss.detach(),
        "invariant_recall_mass": _support_recall_mass(
            natural_union["invariant_indices"],
            target_indices,
            target_weights,
        ).mean().detach(),
        "prototype_recall_mass": _support_recall_mass(
            natural_union["prototype_indices"],
            target_indices,
            target_weights,
        ).mean().detach(),
        "floor_recall_mass": _support_recall_mass(
            natural_union["floor_indices"],
            target_indices,
            target_weights,
        ).mean().detach(),
        "union_recall_mass": _support_recall_mass(
            natural_candidate,
            target_indices,
            target_weights,
        ).mean().detach(),
        **{
            f"family_weight_{name}": evidence["inference_family_weights"][
                :, family_index
            ].mean().detach()
            for family_index, name in enumerate(EVIDENCE_FAMILIES)
        },
        "mean_global_disagreement": evidence[
            "global_disagreement"
        ].mean().detach(),
        "mean_disagreement_temperature": evidence[
            "disagreement_temperature"
        ].mean().detach(),
        "mean_radio_mismatch": evidence["radio_mismatch"].mean().detach(),
        "family_dropout_fraction": (
            1.0 - evidence["family_keep_mask"]
        ).sum(dim=1).gt(0).float().mean().detach(),
    }


def v183_predict(
    model: StructuredEvidenceAnchorModel,
    rss: torch.Tensor,
    *,
    posterior_temperature: float,
    scale_multiplier: float,
    target_coords: torch.Tensor | None = None,
    config: Mapping[str, object] = V183_CONFIG,
) -> dict[str, torch.Tensor]:
    """Predict with one checkpoint and structured evidence consensus."""

    validate_v183_config(config)
    if (
        not math.isfinite(float(posterior_temperature))
        or float(posterior_temperature) < 1.0
    ):
        raise ValueError(
            "V18.3 posterior_temperature must be finite and >= 1; "
            "calibration may soften but never sharpen"
        )
    if (
        not math.isfinite(float(scale_multiplier))
        or float(scale_multiplier) <= 0.0
    ):
        raise ValueError("scale_multiplier must be positive and finite")

    model.eval()
    with torch.no_grad():
        invariant_logits, embedding = model.retrieval_logits(rss)
        prototype_score = prototype_retrieval_score(model, rss)
        union = candidate_union(
            model,
            invariant_logits,
            prototype_score,
            force_indices=None,
            config=config,
        )

    evidence = structured_evidence_opinions(
        model,
        rss,
        embedding,
        invariant_logits,
        prototype_score,
        union,
        training=False,
        config=config,
    )

    with torch.no_grad():
        union_probability = evidence["consensus_probability"]
        posterior_k = min(
            int(config["posterior_top_k"]),
            union_probability.shape[1],
        )
        top_probability, local_index = torch.topk(
            union_probability,
            posterior_k,
            dim=1,
        )
        candidate_indices = torch.gather(
            evidence["candidate_indices"],
            1,
            local_index,
        )
        # The model-internal disagreement temperature has already been applied.
        # Calibration is restricted to non-sharpening temperature >=1.
        probability = top_probability.clamp(min=1e-12).pow(
            1.0 / float(posterior_temperature)
        )
        probability = probability / probability.sum(
            dim=1,
            keepdim=True,
        ).clamp(min=1e-12)

        location, base_scale = model._candidate_components(
            embedding,
            candidate_indices,
        )
        scale = base_scale * float(scale_multiplier)

        pred_building, pred_floor = _hierarchy_prediction(
            model,
            rss,
            invariant_logits,
            deterministic_building=bool(config["deterministic_building"]),
        )
        coord_probability = probability
        if bool(config["condition_building"]):
            same_building = (
                model.anchor_buildings[candidate_indices]
                == pred_building.unsqueeze(1)
            )
            conditioned = probability * same_building
            valid = conditioned.sum(dim=1, keepdim=True) > 0.0
            conditioned = conditioned / conditioned.sum(
                dim=1,
                keepdim=True,
            ).clamp(min=1e-12)
            coord_probability = torch.where(
                valid,
                conditioned,
                probability,
            )

        mean = torch.sum(
            coord_probability.unsqueeze(2) * location,
            dim=1,
        )
        component_variance = (
            scale.square()
            * float(config["student_df"])
            / (float(config["student_df"]) - 2.0)
        )
        within_second = torch.sum(
            coord_probability * component_variance.sum(dim=2),
            dim=1,
        )
        between_second = torch.sum(
            coord_probability
            * (location - mean.unsqueeze(1)).square().sum(dim=2),
            dim=1,
        )
        entropy = -torch.sum(
            probability * probability.clamp(min=1e-12).log(),
            dim=1,
        )

        source_probability_final = torch.stack(
            [
                torch.gather(
                    evidence["source_probability"][:, source_index, :],
                    1,
                    local_index,
                )
                for source_index in range(len(EVIDENCE_SOURCES))
            ],
            dim=1,
        )
        source_probability_final = source_probability_final / (
            source_probability_final.sum(dim=2, keepdim=True).clamp(min=1e-12)
        )
        family_probability_final = torch.stack(
            [
                torch.gather(
                    evidence["family_probability"][:, family_index, :],
                    1,
                    local_index,
                )
                for family_index in range(len(EVIDENCE_FAMILIES))
            ],
            dim=1,
        )
        family_probability_final = family_probability_final / (
            family_probability_final.sum(dim=2, keepdim=True).clamp(min=1e-12)
        )

        result = {
            "coords": mean,
            "building": pred_building,
            "floor": pred_floor,
            "radial_scale": (
                within_second + between_second
            ).clamp(min=1e-8).sqrt(),
            "within_second_moment": within_second,
            "between_second_moment": between_second,
            "candidate_indices": candidate_indices,
            "candidate_weights": probability,
            "component_locations": location,
            "component_scales": scale,
            "component_scales_uncalibrated": base_scale,
            "anchor_entropy": entropy,
            "effective_anchor_count": torch.exp(entropy),
            "top1_probability": probability.max(dim=1).values,
            "source_probability": source_probability_final,
            "source_entropy_union": evidence["source_entropy"],
            "family_probability": family_probability_final,
            "within_family_weights": evidence["within_family_weights"],
            "family_weights": evidence["inference_family_weights"],
            "family_entropy": evidence["family_entropy"],
            "family_disagreement": evidence["family_disagreement"],
            "global_disagreement": evidence["global_disagreement"],
            "disagreement_temperature": evidence["disagreement_temperature"],
            "radio_mismatch": evidence["radio_mismatch"],
            "radio_energy_union": evidence["radio_energy"],
            "style_offset": evidence["style_offset"],
            "style_scale": evidence["style_scale"],
            "union_candidate_indices": evidence["candidate_indices"],
            "union_probability": union_probability,
            "retrieval_invariant_indices": union["invariant_indices"],
            "retrieval_prototype_indices": union["prototype_indices"],
            "retrieval_floor_indices": union["floor_indices"],
            "retrieval_geometry_indices": union["geometry_indices"],
        }
        if target_coords is not None:
            component_lp = model.student_log_prob(
                target_coords,
                location,
                scale,
            )
            result["nll"] = -torch.logsumexp(
                probability.clamp(min=1e-12).log() + component_lp,
                dim=1,
            )
        return result
