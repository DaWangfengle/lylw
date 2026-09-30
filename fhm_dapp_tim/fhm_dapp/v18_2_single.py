"""V18.2 single-member adaptive multi-evidence localization.

V18.2 keeps the V18.1 leave-anchor-out geometry law, but replaces the
"neural Top-K then radio correction" bottleneck with a single-model,
multi-evidence posterior.

One checkpoint produces six interpretable anchor opinions:

1. maximal-invariant neural retrieval;
2. V17-style raw-RSS prototype retrieval;
3. learned candidate radio-evidence reranking;
4. V9 robust marked-radio likelihood with shared affine profiling;
5. soft floor evidence;
6. geometry-manifold evidence.

The opinions are combined by a per-scan learned reliability gate using the
V17-selected alpha=1 arithmetic barycenter principle.  There is no member
axis, no teacher, no distillation, and no ensemble at training or inference.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
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


VISIBLE_THRESHOLD = -0.999

EVIDENCE_SOURCES = (
    "invariant",
    "prototype",
    "learned",
    "marked_radio",
    "floor",
    "geometry",
)

V182_CONFIG = {
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
    # Multi-source candidate discovery. L is deliberately wider than K.
    "invariant_retrieval_k": 32,
    "prototype_retrieval_k": 32,
    "floor_retrieval_k": 16,
    "geometry_seed_k": 8,
    "candidate_union_size": 96,
    "posterior_top_k": 64,
    # Evidence consensus.
    "evidence_alpha": 1.0,
    "evidence_gate_hidden": 32,
    "geometry_manifold_sigma_m": 6.0,
    "geometry_manifold_margin_m": 2.0,
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
    # Auxiliary losses. The fused geometry joint score has coefficient one.
    "invariant_aux_weight": 0.25,
    "reranker_aux_weight": 0.25,
    "hierarchy_weight": 0.50,
    "coordinate_weight": 0.05,
    "ranking_weight": 0.10,
    "ranking_margin": 0.25,
    # Inference.
    "condition_building": False,
    "deterministic_building": True,
}


def validate_v182_config(
    config: Mapping[str, object] = V182_CONFIG,
) -> None:
    """Fail closed when the V18.2 scientific model is changed accidentally."""

    if int(config["residual_experts"]) != 1:
        raise ValueError("V18.2 requires exactly one residual expert")
    if str(config["scan_feature_mode"]) != "bounded_maximal_invariant":
        raise ValueError("V18.2 requires the bounded maximal-invariant encoder")
    if str(config["prototype_metric"]) != "none":
        raise ValueError(
            "prototype RSS is a separate evidence opinion, not a hidden "
            "neural retrieval term"
        )
    if not bool(config["candidate_reranker"]):
        raise ValueError("V18.2 requires the learned evidence opinion")
    if bool(config["reranker_anchor_decoder"]):
        raise ValueError("V18.2 requires the common Student-t residual head")
    if float(config["evidence_alpha"]) != 1.0:
        raise ValueError(
            "V18.2 registers the V17-selected alpha=1 evidence barycenter"
        )
    union = int(config["candidate_union_size"])
    posterior = int(config["posterior_top_k"])
    if union <= posterior:
        raise ValueError("candidate_union_size must exceed posterior_top_k")
    for name in (
        "invariant_retrieval_k",
        "prototype_retrieval_k",
        "floor_retrieval_k",
        "geometry_seed_k",
        "posterior_top_k",
    ):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(config["minimum_epochs"]) < 1:
        raise ValueError("minimum_epochs must be positive")
    if int(config["maximum_epochs"]) < int(config["minimum_epochs"]):
        raise ValueError("maximum_epochs must be at least minimum_epochs")
    if int(config["patience"]) < 1:
        raise ValueError("patience must be positive")
    if float(config["radio_lambda"]) <= 0.0:
        raise ValueError("radio_lambda must be positive")
    if int(config["radio_profile_steps"]) < 1:
        raise ValueError("radio_profile_steps must be positive")
    if float(config["ranking_margin"]) < 0.0:
        raise ValueError("ranking_margin cannot be negative")


class AdaptiveEvidenceAnchorModel(AnchorStudentTModel):
    """One probabilistic localizer with an adaptive evidence-consensus gate."""

    def __init__(
        self,
        *args,
        evidence_gate_hidden: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Gate features:
        # visible fraction; six source entropies; invariant/prototype support
        # overlap; profiled absolute offset; profiled absolute log-gain.
        self.evidence_gate_feature_dim = 10
        self.evidence_gate = nn.Sequential(
            nn.Linear(self.evidence_gate_feature_dim, int(evidence_gate_hidden)),
            nn.GELU(),
            nn.Linear(int(evidence_gate_hidden), len(EVIDENCE_SOURCES)),
        )
        # Equal source weights at initialization.  The gate must earn any
        # preference from the leave-anchor-out training objective.
        nn.init.zeros_(self.evidence_gate[-1].weight)
        nn.init.zeros_(self.evidence_gate[-1].bias)


def build_v182_model(
    *,
    memory: Mapping[str, torch.Tensor],
    n_aps: int,
    n_buildings: int,
    n_devices: int,
    device: torch.device,
    config: Mapping[str, object] = V182_CONFIG,
) -> tuple[
    AdaptiveEvidenceAnchorModel,
    dict[str, object],
    torch.Tensor,
    torch.Tensor,
]:
    """Build the single V18.2 model and deterministic anchor evidence memory."""

    validate_v182_config(config)
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
        raise KeyError(f"anchor memory misses V18.2 fields: {sorted(missing)}")

    geometry = build_delaunay_geometry(
        memory["coords"],
        memory["building"],
        memory["floor"],
    )
    neighbour_indices, neighbour_weights = delaunay_neighbor_targets(geometry)
    anchor_id = torch.arange(len(neighbour_indices), dtype=torch.long).unsqueeze(1)
    if bool(
        (
            (neighbour_indices == anchor_id)
            & (neighbour_weights > 0.0)
        ).any()
    ):
        raise ValueError(
            "V18.2 leave-self-out geometry requires two or more anchors "
            "in every building-floor stratum"
        )

    seen = (memory["prototype_rss"] > VISIBLE_THRESHOLD).any(dim=0)
    owner = _ap_building_owner(
        memory["prototype_rss"],
        memory["building"],
        int(n_buildings),
    )
    model = AdaptiveEvidenceAnchorModel(
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
        evidence_gate_hidden=int(config["evidence_gate_hidden"]),
    ).to(device)
    model.initialize_anchor_keys(memory["prototype_rss"].to(device))

    # Deterministic, reconstructible buffers are excluded from checkpoints.
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


def _standardize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1, keepdim=True).sqrt()
    return centered / scale.clamp(min=1e-4)


def prototype_retrieval_score(
    model: AdaptiveEvidenceAnchorModel,
    rss: torch.Tensor,
) -> torch.Tensor:
    """Return the V17-style raw-RSS prototype opinion over all anchors."""

    seen = model.encoder.seen_ap_mask
    query = rss[:, seen].float()
    prototypes = model.prototype_rss[:, seen].float()
    query_norm = query.square().mean(dim=1, keepdim=True)
    prototype_norm = prototypes.square().mean(dim=1).unsqueeze(0)
    cross = query @ prototypes.t() / max(1, int(seen.sum()))
    distance = (
        query_norm + prototype_norm - 2.0 * cross
    ).clamp(min=0.0).sqrt()
    return -distance


def _floor_log_probability(
    model: AdaptiveEvidenceAnchorModel,
    invariant_logits: torch.Tensor,
) -> torch.Tensor:
    floor_logits = model._marginal_logits(
        invariant_logits,
        model.anchor_floors,
    )
    return F.log_softmax(floor_logits, dim=1)


def _scatter_candidate_mask(
    mask: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    mask.scatter_(1, indices, True)


def candidate_union(
    model: AdaptiveEvidenceAnchorModel,
    invariant_logits: torch.Tensor,
    prototype_score: torch.Tensor,
    *,
    force_indices: torch.Tensor | None = None,
    config: Mapping[str, object] = V182_CONFIG,
) -> dict[str, torch.Tensor]:
    """Build one deterministic union from neural, radio, floor, and geometry.

    The union is a support-construction operation, not an ensemble.  Every
    source comes from the same checkpoint or deterministic anchor memory.
    """

    validate_v182_config(config)
    batch, anchors = invariant_logits.shape
    if anchors != model.n_anchors:
        raise ValueError("invariant logits do not match the anchor dictionary")

    inv_k = min(int(config["invariant_retrieval_k"]), anchors)
    proto_k = min(int(config["prototype_retrieval_k"]), anchors)
    floor_k = min(int(config["floor_retrieval_k"]), anchors)
    seed_k = min(int(config["geometry_seed_k"]), anchors)
    union_k = min(int(config["candidate_union_size"]), anchors)

    inv_indices = torch.topk(
        invariant_logits,
        inv_k,
        dim=1,
    ).indices
    proto_indices = torch.topk(
        prototype_score,
        proto_k,
        dim=1,
    ).indices

    floor_logp = _floor_log_probability(model, invariant_logits)
    floor_anchor_score = torch.gather(
        floor_logp,
        1,
        model.anchor_floors.unsqueeze(0).expand(batch, -1),
    )
    floor_aware_score = _standardize(invariant_logits) + floor_anchor_score
    floor_indices = torch.topk(
        floor_aware_score,
        floor_k,
        dim=1,
    ).indices

    inv_seed = inv_indices[:, :seed_k]
    proto_seed = proto_indices[:, :seed_k]
    seed = torch.cat([inv_seed, proto_seed], dim=1)
    geometry_indices = model.evidence_geometry_neighbors[seed].reshape(batch, -1)

    source_mask = torch.zeros(
        batch,
        anchors,
        dtype=torch.bool,
        device=invariant_logits.device,
    )
    _scatter_candidate_mask(source_mask, inv_indices)
    _scatter_candidate_mask(source_mask, proto_indices)
    _scatter_candidate_mask(source_mask, floor_indices)
    _scatter_candidate_mask(source_mask, geometry_indices)

    # Source-normalized priority prevents the numeric scale of raw RSS distance
    # from overwhelming learned neural logits.
    priority = torch.maximum(
        _standardize(invariant_logits),
        _standardize(prototype_score),
    )
    priority = torch.maximum(
        priority,
        _standardize(floor_aware_score),
    )

    # Guarantee at least union_k candidates even when all sources agree.
    fallback = torch.topk(
        (
            _standardize(invariant_logits)
            + _standardize(prototype_score)
            + 0.5 * floor_anchor_score
        ),
        union_k,
        dim=1,
    ).indices
    _scatter_candidate_mask(source_mask, fallback)

    if force_indices is not None:
        if force_indices.ndim != 2 or len(force_indices) != batch:
            raise ValueError("force_indices must be [batch,width]")
        _scatter_candidate_mask(source_mask, force_indices)

    masked_priority = priority.masked_fill(~source_mask, float("-inf"))
    if force_indices is not None:
        # Outcome labels are used only in training.  Mandatory non-self
        # Delaunay targets outrank natural candidates so the supervised joint
        # score is always defined.
        forced_priority = (
            masked_priority.max(dim=1, keepdim=True).values
            + 10.0
            + torch.arange(
                force_indices.shape[1],
                dtype=masked_priority.dtype,
                device=masked_priority.device,
            ).unsqueeze(0)
            * 1e-3
        )
        masked_priority = masked_priority.scatter(
            1,
            force_indices,
            forced_priority,
        )

    candidate_indices = torch.topk(
        masked_priority,
        union_k,
        dim=1,
    ).indices

    def on_union(indices: torch.Tensor) -> torch.Tensor:
        membership = (
            candidate_indices.unsqueeze(2) == indices.unsqueeze(1)
        ).any(dim=2)
        return membership

    return {
        "candidate_indices": candidate_indices,
        "invariant_indices": inv_indices,
        "prototype_indices": proto_indices,
        "floor_indices": floor_indices,
        "geometry_indices": geometry_indices,
        "candidate_from_invariant": on_union(inv_indices),
        "candidate_from_prototype": on_union(proto_indices),
        "candidate_from_floor": on_union(floor_indices),
        "candidate_from_geometry": on_union(geometry_indices),
    }


def _normalized_entropy(probability: torch.Tensor) -> torch.Tensor:
    entropy = -torch.sum(
        probability * probability.clamp(min=1e-12).log(),
        dim=1,
    )
    return entropy / math.log(max(probability.shape[1], 2))


def _source_overlap(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    intersection = (
        first.unsqueeze(2) == second.unsqueeze(1)
    ).any(dim=2).float().sum(dim=1)
    union = first.shape[1] + second.shape[1] - intersection
    return intersection / union.clamp(min=1.0)


def _candidate_floor_probability(
    model: AdaptiveEvidenceAnchorModel,
    invariant_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distribute each floor's posterior mass uniformly over its candidates.

    This prevents a floor from gaining probability merely because the current
    candidate union contains more anchors from that floor.
    """

    floor_logp = _floor_log_probability(model, invariant_logits)
    floor_probability = floor_logp.exp()
    candidate_floor = model.anchor_floors[candidate_indices]
    floor_count = floor_probability.shape[1]
    one_hot = F.one_hot(
        candidate_floor,
        num_classes=floor_count,
    ).to(floor_probability.dtype)
    count = one_hot.sum(dim=1).clamp(min=1.0)
    candidate_floor_probability = torch.gather(
        floor_probability,
        1,
        candidate_floor,
    )
    candidate_floor_count = torch.gather(
        count,
        1,
        candidate_floor,
    )
    candidate_floor_probability = (
        candidate_floor_probability / candidate_floor_count
    )
    candidate_floor_probability = (
        candidate_floor_probability
        / candidate_floor_probability.sum(dim=1, keepdim=True).clamp(min=1e-12)
    )
    return candidate_floor_probability, floor_logp


def evidence_opinions(
    model: AdaptiveEvidenceAnchorModel,
    rss: torch.Tensor,
    embedding: torch.Tensor,
    invariant_logits: torch.Tensor,
    prototype_score: torch.Tensor,
    union: Mapping[str, torch.Tensor],
    *,
    config: Mapping[str, object] = V182_CONFIG,
) -> dict[str, torch.Tensor]:
    """Construct six normalized opinions and their adaptive alpha=1 pool."""

    validate_v182_config(config)
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

    floor_probability, floor_logp = _candidate_floor_probability(
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
    visible_fraction = (
        (
            (rss > model.encoder.visible_threshold)
            & model.encoder.seen_ap_mask.unsqueeze(0)
        )
        .float()
        .sum(dim=1)
        / max(1, int(model.encoder.seen_ap_mask.sum()))
    )
    overlap = _source_overlap(
        union["invariant_indices"],
        union["prototype_indices"],
    )
    style_offset = style["style_offset"].to(inv_score)
    style_scale = style["style_scale"].to(inv_score)
    gate_features = torch.cat(
        [
            visible_fraction.unsqueeze(1),
            source_entropy,
            overlap.unsqueeze(1),
            style_offset.abs().unsqueeze(1),
            torch.log(style_scale.clamp(min=1e-6)).abs().unsqueeze(1),
        ],
        dim=1,
    )
    if gate_features.shape[1] != model.evidence_gate_feature_dim:
        raise RuntimeError("V18.2 evidence-gate feature width drifted")

    gate_logits = model.evidence_gate(gate_features)
    gate_weights = F.softmax(gate_logits, dim=1)

    # alpha=1: normalized weighted arithmetic opinion pool.  This is the
    # evidence-source analogue of the final V17 arithmetic consensus, but all
    # opinions belong to one model.
    consensus = torch.sum(
        gate_weights.unsqueeze(2) * source_probability,
        dim=1,
    )
    consensus = consensus / consensus.sum(dim=1, keepdim=True).clamp(min=1e-12)

    return {
        "candidate_indices": candidate_indices,
        "candidate_locations": candidate_locations,
        "invariant_score": inv_score,
        "prototype_score": proto_score,
        "learned_score": learned_score,
        "radio_energy": radio_energy,
        "geometry_energy": geometry_energy,
        "floor_log_probability": torch.log(
            floor_probability.clamp(min=1e-12)
        ),
        "source_probability": source_probability,
        "source_entropy": source_entropy,
        "gate_features": gate_features,
        "gate_weights": gate_weights,
        "consensus_probability": consensus,
        "style_offset": style_offset,
        "style_scale": style_scale,
    }


def _weighted_target_mass(
    candidate_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(
        (
            candidate_indices.unsqueeze(2)
            == target_indices.unsqueeze(1)
        ).to(target_weights.dtype)
        * target_weights.unsqueeze(1),
        dim=2,
    )


def _support_recall_mass(
    candidate_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    matched = (
        candidate_indices.unsqueeze(2)
        == target_indices.unsqueeze(1)
    ).any(dim=1)
    return torch.sum(
        matched.to(target_weights.dtype) * target_weights,
        dim=1,
    )


def v182_training_loss(
    model: AdaptiveEvidenceAnchorModel,
    rss: torch.Tensor,
    target_coords: torch.Tensor,
    target_anchor: torch.Tensor,
    device_ids: torch.Tensor,
    neighbour_indices: torch.Tensor,
    neighbour_weights: torch.Tensor,
    *,
    config: Mapping[str, object] = V182_CONFIG,
) -> dict[str, torch.Tensor]:
    """Train one adaptive evidence posterior under leave-anchor-out geometry."""

    validate_v182_config(config)
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
        raise ValueError("V18.2 geometry targets are not normalized")

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
    evidence = evidence_opinions(
        model,
        model_rss,
        embedding,
        invariant_logits,
        prototype_score,
        forced_union,
        config=config,
    )

    candidate_indices = evidence["candidate_indices"]
    candidate_target = _weighted_target_mass(
        candidate_indices,
        target_indices,
        target_weights,
    )
    if bool((candidate_target.sum(dim=1) <= 0.999).any()):
        raise RuntimeError("V18.2 forced union lost geometry target mass")

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
            f"gate_weight_{name}": evidence["gate_weights"][
                :, source_index
            ].mean().detach()
            for source_index, name in enumerate(EVIDENCE_SOURCES)
        },
    }


def _hierarchy_prediction(
    model: AdaptiveEvidenceAnchorModel,
    rss: torch.Tensor,
    invariant_logits: torch.Tensor,
    *,
    deterministic_building: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    building_logits = model._marginal_logits(
        invariant_logits,
        model.anchor_buildings,
    )
    floor_logits = model._marginal_logits(
        invariant_logits,
        model.anchor_floors,
    )
    pred_building = building_logits.argmax(dim=1)
    pred_floor = floor_logits.argmax(dim=1)
    if deterministic_building:
        visible = (
            (rss > model.encoder.visible_threshold)
            & model.encoder.seen_ap_mask.unsqueeze(0)
        )
        signal = (rss + 1.0).clamp(min=0.0) * visible
        building_scores = torch.stack(
            [
                signal[:, model.ap_building_owner == class_id].sum(dim=1)
                for class_id in range(building_logits.shape[1])
            ],
            dim=1,
        )
        has_signal = building_scores.sum(dim=1) > 0.0
        pred_building = torch.where(
            has_signal,
            building_scores.argmax(dim=1),
            pred_building,
        )
    return pred_building, pred_floor


def v182_predict(
    model: AdaptiveEvidenceAnchorModel,
    rss: torch.Tensor,
    *,
    posterior_temperature: float,
    scale_multiplier: float,
    target_coords: torch.Tensor | None = None,
    config: Mapping[str, object] = V182_CONFIG,
) -> dict[str, torch.Tensor]:
    """Predict with one checkpoint and adaptive multi-evidence consensus."""

    validate_v182_config(config)
    if (
        not math.isfinite(float(posterior_temperature))
        or float(posterior_temperature) <= 0.0
    ):
        raise ValueError("posterior_temperature must be positive and finite")
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

    evidence = evidence_opinions(
        model,
        rss,
        embedding,
        invariant_logits,
        prototype_score,
        union,
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
        probability = (
            top_probability.clamp(min=1e-12).pow(
                1.0 / float(posterior_temperature)
            )
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

        # Gather final-support source opinions for diagnostics only.
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
            "gate_weights": evidence["gate_weights"],
            "source_entropy_union": evidence["source_entropy"],
            "source_probability": source_probability_final,
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
