"""V18.4 single-checkpoint spatial posterior model.

V18.4 deliberately separates three mechanisms:

1. proposal support construction;
2. learned discriminative base posterior q_theta and Student-t residual model;
3. deterministic radio/geometry trust-region correction at inference.

The structured correction never participates in backbone gradient training.
There is no evidence-voting gate, member axis, teacher, distillation or
multi-checkpoint inference path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    build_delaunay_geometry,
    profile_shared_affine_marked_energy,
)
from fhm_dapp_tim.fhm_dapp.dapp import anchor_ap_reliability
from fhm_dapp_tim.fhm_dapp.model import (
    AnchorStudentTModel,
    gradient_reverse,
)
from fhm_dapp_tim.fhm_dapp.train_anchor import _ap_building_owner
from fhm_dapp_tim.fhm_dapp.v18_single import delaunay_neighbor_targets
from fhm_dapp_tim.fhm_dapp.v18_4_core import (
    empirical_survival_reliability,
    fit_empirical_reference,
    masked_softmax,
    normalize_energy,
    robust_energy_scale,
    spatial_sinkhorn_divergence,
    trust_region_posterior_correction,
    weighted_component_geometric_median,
)


VISIBLE_THRESHOLD = -0.999

V184_CONFIG: dict[str, object] = {
    # Single learned backbone.
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
    # Invariant representation + one learned candidate posterior.
    "scan_feature_mode": "bounded_maximal_invariant",
    "prototype_metric": "none",
    "candidate_reranker": True,
    "candidate_reranker_mode": "summary",
    "reranker_anchor_decoder": False,
    # Dynamic natural support.
    "invariant_retrieval_k": 32,
    "prototype_retrieval_k": 32,
    "floor_retrieval_k": 16,
    "geometry_seed_k": 8,
    "support_ceiling": 128,
    # Spatial posterior supervision.
    "epsilon_ot": 0.10,
    "lambda_ot_grid": (0.10, 0.25, 0.50),
    # Structured correction.
    "epsilon_kl_grid": (0.025, 0.050, 0.100),
    "geometry_sigma_scale": 1.50,
    "geometry_margin_scale": 0.50,
    "radio_profile_steps": 6,
    "radio_profile_learning_rate": 0.20,
    "radio_profile_offset_max": 0.20,
    "radio_profile_log_scale_max": 0.25,
    "radio_profile_offset_weight": 0.10,
    "radio_profile_log_scale_weight": 0.10,
    # Training.
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "maximum_epochs": 60,
    "minimum_epochs": 12,
    "patience": 8,
    "energy_tie_margin_m": 0.05,
    "rss_offset_aug_db": 4.0,
    "rss_scale_aug": 0.08,
    "ap_dropout": 0.10,
    # Objective coefficients.  The learned candidate posterior itself is
    # supervised; there is no separate reranker auxiliary objective.
    "anchor_weight": 1.0,
    "coordinate_weight": 1.0,
    "invariant_aux_weight": 0.25,
    "hierarchy_weight": 0.50,
    "domain_adversarial_weight": 0.05,
}


@dataclass(frozen=True)
class CorrectionReference:
    """Training-fitted, label-free reference for deterministic correction."""

    radio_energy_scale: torch.Tensor
    geometry_energy_scale: torch.Tensor
    radio_mismatch_reference: torch.Tensor
    geometry_mismatch_reference: torch.Tensor

    def to(self, device: torch.device) -> "CorrectionReference":
        return CorrectionReference(
            radio_energy_scale=self.radio_energy_scale.to(device),
            geometry_energy_scale=self.geometry_energy_scale.to(device),
            radio_mismatch_reference=self.radio_mismatch_reference.to(device),
            geometry_mismatch_reference=self.geometry_mismatch_reference.to(device),
        )


class TrustRegionAnchorModel(AnchorStudentTModel):
    """One checkpoint; no additional trainable routing or evidence gate."""


def validate_v184_config(
    config: Mapping[str, object] = V184_CONFIG,
) -> None:
    if int(config["residual_experts"]) != 1:
        raise ValueError("V18.4 requires exactly one residual expert")
    if str(config["scan_feature_mode"]) != "bounded_maximal_invariant":
        raise ValueError("V18.4 requires bounded maximal-invariant encoding")
    if str(config["prototype_metric"]) != "none":
        raise ValueError(
            "raw RSS prototype retrieval is an explicit proposal source, not "
            "a hidden neural retrieval metric"
        )
    if not bool(config["candidate_reranker"]):
        raise ValueError("V18.4 requires one learned candidate posterior scorer")
    if bool(config["reranker_anchor_decoder"]):
        raise ValueError("V18.4 requires the shared Student-t residual head")

    primary = (
        int(config["invariant_retrieval_k"])
        + int(config["prototype_retrieval_k"])
        + int(config["floor_retrieval_k"])
    )
    if int(config["support_ceiling"]) < primary:
        raise ValueError("support ceiling cannot hold registered primary quotas")
    for name in (
        "invariant_retrieval_k",
        "prototype_retrieval_k",
        "floor_retrieval_k",
        "geometry_seed_k",
        "support_ceiling",
        "radio_profile_steps",
    ):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(config["epsilon_ot"]) <= 0.0:
        raise ValueError("epsilon_ot must be positive")
    if float(config["geometry_sigma_scale"]) <= 0.0:
        raise ValueError("geometry sigma scale must be positive")
    if float(config["geometry_margin_scale"]) < 0.0:
        raise ValueError("geometry margin scale cannot be negative")
    if any(float(value) <= 0.0 for value in config["lambda_ot_grid"]):
        raise ValueError("lambda_ot grid values must be positive")
    if any(float(value) < 0.0 for value in config["epsilon_kl_grid"]):
        raise ValueError("epsilon_kl grid values cannot be negative")
    if int(config["minimum_epochs"]) < 1:
        raise ValueError("minimum_epochs must be positive")
    if int(config["maximum_epochs"]) < int(config["minimum_epochs"]):
        raise ValueError("maximum_epochs must be >= minimum_epochs")


def _positive_target_width(weights: torch.Tensor) -> int:
    if weights.ndim != 2:
        raise ValueError("target weights must be a matrix")
    return int((weights > 0.0).sum(dim=1).max().item())


def training_spatial_scale_m(
    memory: Mapping[str, torch.Tensor],
) -> float:
    """Median positive same-stratum Delaunay edge length from training anchors."""

    geometry = build_delaunay_geometry(
        memory["coords"],
        memory["building"],
        memory["floor"],
    )
    indices, weights = delaunay_neighbor_targets(geometry)
    anchor = torch.arange(len(indices), dtype=torch.long).unsqueeze(1)
    active = (weights > 0.0) & (indices != anchor)
    source = memory["coords"].unsqueeze(1).expand(-1, indices.shape[1], -1)
    target = memory["coords"][indices]
    distance = torch.linalg.vector_norm(source - target, dim=2)
    positive = distance[active & (distance > 0.0)]
    if positive.numel() == 0:
        raise ValueError("training geometry has no positive non-self edge")
    value = float(torch.median(positive).item())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("training spatial scale is invalid")
    return value


def build_v184_model(
    *,
    memory: Mapping[str, torch.Tensor],
    n_aps: int,
    n_buildings: int,
    n_devices: int,
    spatial_scale_m: float,
    device: torch.device,
    config: Mapping[str, object] = V184_CONFIG,
) -> tuple[
    TrustRegionAnchorModel,
    dict[str, object],
    torch.Tensor,
    torch.Tensor,
]:
    """Build the one-checkpoint V18.4 model and leave-self-out geometry."""

    validate_v184_config(config)
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
        raise KeyError(f"anchor memory misses V18.4 fields: {sorted(missing)}")
    if not math.isfinite(float(spatial_scale_m)) or float(spatial_scale_m) <= 0.0:
        raise ValueError("spatial_scale_m must be positive and finite")

    geometry = build_delaunay_geometry(
        memory["coords"],
        memory["building"],
        memory["floor"],
    )
    neighbour_indices, neighbour_weights = delaunay_neighbor_targets(geometry)

    primary = (
        min(int(config["invariant_retrieval_k"]), len(memory["coords"]))
        + min(int(config["prototype_retrieval_k"]), len(memory["coords"]))
        + min(int(config["floor_retrieval_k"]), len(memory["coords"]))
    )
    # The conservative sum ignores source deduplication.  If this cannot fit,
    # the formal support contract is not guaranteed.
    required_capacity = primary + _positive_target_width(neighbour_weights)
    if required_capacity > int(config["support_ceiling"]):
        raise ValueError(
            "support ceiling cannot simultaneously preserve registered "
            f"primary proposal quotas and all positive targets: "
            f"{required_capacity}>{int(config['support_ceiling'])}"
        )

    anchors = torch.arange(len(neighbour_indices), dtype=torch.long).unsqueeze(1)
    if bool(
        ((neighbour_indices == anchors) & (neighbour_weights > 0.0)).any()
    ):
        raise ValueError("V18.4 leave-self-out target contains positive self mass")

    seen = (memory["prototype_rss"] > VISIBLE_THRESHOLD).any(dim=0)
    owner = _ap_building_owner(
        memory["prototype_rss"],
        memory["building"],
        int(n_buildings),
    )
    model = TrustRegionAnchorModel(
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
        "v184_ap_reliability",
        reliability.to(device),
        persistent=False,
    )
    model.register_buffer(
        "v184_geometry_neighbors",
        neighbour_indices.to(device),
        persistent=False,
    )
    model.register_buffer(
        "v184_geometry_neighbor_weights",
        neighbour_weights.to(device),
        persistent=False,
    )
    model.register_buffer(
        "v184_spatial_scale_m",
        torch.tensor(float(spatial_scale_m), dtype=torch.float32, device=device),
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
    model: TrustRegionAnchorModel,
    rss: torch.Tensor,
) -> torch.Tensor:
    """V17-style raw-RSS prototype proposal score over all anchors."""

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


def _floor_anchor_score(
    model: TrustRegionAnchorModel,
    invariant_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    floor_logits = model._marginal_logits(
        invariant_logits,
        model.anchor_floors,
    )
    floor_logp = F.log_softmax(floor_logits, dim=1)
    anchor_score = torch.gather(
        floor_logp,
        1,
        model.anchor_floors.unsqueeze(0).expand(len(invariant_logits), -1),
    )
    return _standardize(invariant_logits) + anchor_score, floor_logp


def _scatter_membership(
    indices: torch.Tensor,
    active: torch.Tensor,
    *,
    anchors: int,
) -> torch.Tensor:
    """Return [B,A] membership, robust to repeated padded indices."""

    if indices.shape != active.shape:
        raise ValueError("membership indices/active shapes differ")
    counts = torch.zeros(
        len(indices),
        int(anchors),
        dtype=torch.int32,
        device=indices.device,
    )
    counts.scatter_add_(
        1,
        indices,
        active.to(dtype=counts.dtype),
    )
    return counts > 0


def _select_support(
    pool_mask: torch.Tensor,
    mandatory_mask: torch.Tensor,
    priority: torch.Tensor,
    *,
    ceiling: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorized support selection preserving every mandatory member."""

    if not (
        pool_mask.shape == mandatory_mask.shape == priority.shape
    ):
        raise ValueError("support masks and priority must have equal shape")
    mandatory_count = mandatory_mask.sum(dim=1)
    if bool((mandatory_count > int(ceiling)).any()):
        raise RuntimeError(
            "mandatory support exceeds registered computational ceiling"
        )

    # Strict bonus guarantees all mandatory entries outrank optional geometry.
    finite_priority = torch.where(
        pool_mask,
        priority,
        torch.zeros_like(priority),
    )
    span = (
        finite_priority.max(dim=1, keepdim=True).values
        - finite_priority.min(dim=1, keepdim=True).values
        + 10.0
    )
    ranked = priority + mandatory_mask.to(priority.dtype) * span
    ranked = ranked.masked_fill(~pool_mask, float("-inf"))

    width = min(int(ceiling), priority.shape[1])
    selected = torch.topk(ranked, width, dim=1).indices
    valid = torch.gather(pool_mask, 1, selected)
    return selected, valid


def candidate_support(
    model: TrustRegionAnchorModel,
    invariant_logits: torch.Tensor,
    prototype_score: torch.Tensor,
    *,
    force_indices: torch.Tensor | None = None,
    force_weights: torch.Tensor | None = None,
    config: Mapping[str, object] = V184_CONFIG,
) -> dict[str, torch.Tensor]:
    """Build dynamic masked support without host/device synchronization.

    The three primary proposal lists are mandatory. Positive training targets
    are additionally mandatory when supplied. Geometry-neighbour expansion is
    optional and is truncated only after those contracts are satisfied.
    """

    validate_v184_config(config)
    if invariant_logits.shape != prototype_score.shape:
        raise ValueError("proposal score matrices must have equal shape")
    batch, anchors = invariant_logits.shape
    if anchors != model.n_anchors:
        raise ValueError("proposal scores do not match anchor dictionary")
    if (force_indices is None) != (force_weights is None):
        raise ValueError("force_indices and force_weights must be supplied together")
    if force_indices is not None and force_indices.shape != force_weights.shape:
        raise ValueError("forced indices/weights shape mismatch")
    if force_indices is not None and len(force_indices) != batch:
        raise ValueError("forced targets must have one row per scan")

    inv_k = min(int(config["invariant_retrieval_k"]), anchors)
    proto_k = min(int(config["prototype_retrieval_k"]), anchors)
    floor_k = min(int(config["floor_retrieval_k"]), anchors)
    seed_k = min(int(config["geometry_seed_k"]), anchors)
    ceiling = min(int(config["support_ceiling"]), anchors)

    inv_indices = torch.topk(invariant_logits, inv_k, dim=1).indices
    proto_indices = torch.topk(prototype_score, proto_k, dim=1).indices
    floor_score, floor_logp = _floor_anchor_score(model, invariant_logits)
    floor_indices = torch.topk(floor_score, floor_k, dim=1).indices

    primary_mask = torch.zeros(
        batch,
        anchors,
        dtype=torch.bool,
        device=invariant_logits.device,
    )
    primary_mask.scatter_(1, inv_indices, True)
    primary_mask.scatter_(1, proto_indices, True)
    primary_mask.scatter_(1, floor_indices, True)

    seed = torch.cat(
        [
            inv_indices[:, :seed_k],
            proto_indices[:, :seed_k],
        ],
        dim=1,
    )
    geometry_indices = model.v184_geometry_neighbors[seed].reshape(batch, -1)
    geometry_active = (
        model.v184_geometry_neighbor_weights[seed] > 0.0
    ).reshape(batch, -1)
    geometry_mask = _scatter_membership(
        geometry_indices,
        geometry_active,
        anchors=anchors,
    )

    natural_pool = primary_mask | geometry_mask

    # Source-normalized priority is used only to choose among optional geometry
    # candidates when the natural union is wider than the compute ceiling.
    priority = torch.maximum(
        _standardize(invariant_logits),
        _standardize(prototype_score),
    )
    priority = torch.maximum(priority, _standardize(floor_score))
    # Stable anchor-id epsilon breaks exact score ties without changing source
    # ordering at meaningful scale.
    anchor_tie = torch.arange(
        anchors,
        device=priority.device,
        dtype=priority.dtype,
    ).unsqueeze(0) * 1e-7
    priority = priority + anchor_tie

    natural_indices, natural_valid = _select_support(
        natural_pool,
        primary_mask,
        priority,
        ceiling=ceiling,
    )
    natural_size = natural_valid.sum(dim=1)

    if force_indices is None:
        final_pool = natural_pool
        mandatory = primary_mask
    else:
        force_mask = _scatter_membership(
            force_indices,
            force_weights > 0.0,
            anchors=anchors,
        )
        final_pool = natural_pool | force_mask
        mandatory = primary_mask | force_mask

    candidate_indices, candidate_mask = _select_support(
        final_pool,
        mandatory,
        priority,
        ceiling=ceiling,
    )
    natural_membership = torch.gather(
        natural_pool,
        1,
        candidate_indices,
    ) & candidate_mask

    return {
        "candidate_indices": candidate_indices,
        "candidate_mask": candidate_mask,
        "natural_membership_mask": natural_membership,
        "natural_size": natural_size,
        "natural_candidate_indices": natural_indices,
        "natural_candidate_mask": natural_valid,
        "invariant_indices": inv_indices,
        "prototype_indices": proto_indices,
        "floor_indices": floor_indices,
        "floor_log_probability": floor_logp,
    }

def base_posterior(
    model: TrustRegionAnchorModel,
    rss: torch.Tensor,
    embedding: torch.Tensor,
    invariant_logits: torch.Tensor,
    support: Mapping[str, torch.Tensor],
    *,
    posterior_temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Score one learned discriminative posterior on masked support."""

    if float(posterior_temperature) <= 0.0:
        raise ValueError("posterior temperature must be positive")
    indices = support["candidate_indices"]
    mask = support["candidate_mask"].bool()
    coarse = torch.gather(invariant_logits, 1, indices)
    logits = model.rerank_candidate_logits(
        rss,
        embedding,
        indices,
        coarse,
    )
    probability = masked_softmax(
        logits / float(posterior_temperature),
        mask,
    )
    return {
        "candidate_indices": indices,
        "candidate_mask": mask,
        "candidate_logits": logits,
        "probability": probability,
    }


def _weighted_target_mass(
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    match = (
        candidate_indices.unsqueeze(2)
        == target_indices.unsqueeze(1)
    ).to(target_weights.dtype)
    mass = torch.sum(match * target_weights.unsqueeze(1), dim=2)
    return mass * candidate_mask.to(mass.dtype)


def _support_recall_mass(
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
    target_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    match = (
        support_indices.unsqueeze(2)
        == target_indices.unsqueeze(1)
    ) & support_mask.unsqueeze(2)
    covered = match.any(dim=1)
    return torch.sum(
        covered.to(target_weights.dtype) * target_weights,
        dim=1,
    )


def _source_recall_mass(
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    covered = (
        source_indices.unsqueeze(2) == target_indices.unsqueeze(1)
    ).any(dim=1)
    return torch.sum(
        covered.to(target_weights.dtype) * target_weights,
        dim=1,
    )


def _hierarchy_loss(
    model: TrustRegionAnchorModel,
    invariant_logits: torch.Tensor,
    target_anchor: torch.Tensor,
) -> torch.Tensor:
    target_building = model.anchor_buildings[target_anchor]
    target_floor = model.anchor_floors[target_anchor]
    building_logits = model._marginal_logits(
        invariant_logits,
        model.anchor_buildings,
    )
    floor_logits = model._marginal_logits(
        invariant_logits,
        model.anchor_floors,
    )
    return (
        0.25 * F.cross_entropy(building_logits, target_building)
        + F.cross_entropy(floor_logits, target_floor)
    )


def v184_training_loss(
    model: TrustRegionAnchorModel,
    rss: torch.Tensor,
    target_coords: torch.Tensor,
    target_anchor: torch.Tensor,
    device_ids: torch.Tensor,
    neighbour_indices: torch.Tensor,
    neighbour_weights: torch.Tensor,
    *,
    lambda_ot: float,
    config: Mapping[str, object] = V184_CONFIG,
) -> dict[str, torch.Tensor]:
    """Train q_theta and Student-t residuals; correction is not in gradients."""

    validate_v184_config(config)
    if float(lambda_ot) not in {
        float(value) for value in config["lambda_ot_grid"]
    }:
        raise ValueError("lambda_ot is outside the registered V18.4 grid")

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
    if not torch.allclose(
        target_weights.sum(dim=1),
        torch.ones(len(target_weights), device=target_weights.device),
        atol=1e-5,
    ):
        raise RuntimeError("leave-self-out geometry targets are not normalized")

    natural = candidate_support(
        model,
        invariant_logits.detach(),
        prototype_score.detach(),
        config=config,
    )
    forced = candidate_support(
        model,
        invariant_logits.detach(),
        prototype_score.detach(),
        force_indices=target_indices,
        force_weights=target_weights,
        config=config,
    )
    posterior = base_posterior(
        model,
        model_rss,
        embedding,
        invariant_logits,
        forced,
    )
    candidate_indices = posterior["candidate_indices"]
    candidate_mask = posterior["candidate_mask"]
    q_theta = posterior["probability"]

    target_mass = _weighted_target_mass(
        candidate_indices,
        candidate_mask,
        target_indices,
        target_weights,
    )
    retained = target_mass.sum(dim=1)
    if bool((retained < 1.0 - 1e-5).any()):
        raise RuntimeError("forced support lost positive Delaunay target mass")
    target_mass = target_mass / retained.unsqueeze(1).clamp(min=1e-12)

    anchor_nll = -torch.sum(
        target_mass * q_theta.clamp(min=1e-12).log(),
        dim=1,
    ).mean()

    # Local continuous likelihood is trained directly on the registered
    # non-self Delaunay target, independent of proposal padding.
    target_location, target_scale = model._components(
        embedding,
        target_indices,
    )
    log_coordinate = model.student_log_prob(
        target_coords,
        target_location,
        target_scale,
    )
    coordinate_nll = -torch.sum(
        target_weights * log_coordinate,
        dim=1,
    ).mean()

    candidate_coords = model.anchor_coords[candidate_indices]
    target_coords_anchor = model.anchor_coords[target_indices]
    ot_loss = spatial_sinkhorn_divergence(
        q_theta,
        candidate_coords,
        target_weights,
        target_coords_anchor,
        spatial_scale=model.v184_spatial_scale_m,
        predicted_mask=candidate_mask,
        target_mask=target_weights > 0.0,
        epsilon_ot=float(config["epsilon_ot"]),
    ).mean()

    invariant_logp = F.log_softmax(invariant_logits, dim=1)
    invariant_aux = -torch.sum(
        target_weights
        * torch.gather(invariant_logp, 1, target_indices),
        dim=1,
    ).mean()
    hierarchy_loss = _hierarchy_loss(
        model,
        invariant_logits,
        target_anchor,
    )

    domain_loss = anchor_nll.new_zeros(())
    valid_device = device_ids >= 0
    if (
        model.device_head is not None
        and bool(valid_device.any())
        and float(config["domain_adversarial_weight"]) > 0.0
    ):
        domain_loss = F.cross_entropy(
            model.device_head(gradient_reverse(embedding[valid_device])),
            device_ids[valid_device],
        )

    loss = (
        float(config["anchor_weight"]) * anchor_nll
        + float(config["coordinate_weight"]) * coordinate_nll
        + float(lambda_ot) * ot_loss
        + float(config["invariant_aux_weight"]) * invariant_aux
        + float(config["hierarchy_weight"]) * hierarchy_loss
        + float(config["domain_adversarial_weight"]) * domain_loss
    )

    natural_indices = natural["candidate_indices"]
    natural_mask = natural["candidate_mask"]
    return {
        "loss": loss,
        "anchor_nll": anchor_nll.detach(),
        "coordinate_nll": coordinate_nll.detach(),
        "spatial_ot_loss": ot_loss.detach(),
        "invariant_aux_loss": invariant_aux.detach(),
        "hierarchy_loss": hierarchy_loss.detach(),
        "domain_loss": domain_loss.detach(),
        "natural_recall_mass": _support_recall_mass(
            natural_indices,
            natural_mask,
            target_indices,
            target_weights,
        ).mean().detach(),
        "invariant_recall_mass": _source_recall_mass(
            natural["invariant_indices"],
            target_indices,
            target_weights,
        ).mean().detach(),
        "prototype_recall_mass": _source_recall_mass(
            natural["prototype_indices"],
            target_indices,
            target_weights,
        ).mean().detach(),
        "floor_recall_mass": _source_recall_mass(
            natural["floor_indices"],
            target_indices,
            target_weights,
        ).mean().detach(),
        "natural_support_size": natural["natural_size"].float().mean().detach(),
        "forced_support_size": forced["candidate_mask"].sum(dim=1).float().mean().detach(),
    }


def _masked_geometry_energy(
    probability: torch.Tensor,
    locations: torch.Tensor,
    mask: torch.Tensor,
    *,
    spatial_scale_m: torch.Tensor | float,
    sigma_scale: float,
    margin_scale: float,
) -> torch.Tensor:
    """Distance-to-base-posterior-manifold energy with exact padding mask."""

    q = probability * mask.to(probability.dtype)
    q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-12)
    spatial_scale = torch.as_tensor(
        spatial_scale_m,
        dtype=locations.dtype,
        device=locations.device,
    ).clamp(min=1e-8)
    sigma = spatial_scale * float(sigma_scale)
    margin = spatial_scale * float(margin_scale)
    dist2 = torch.cdist(locations, locations).square()
    log_q = torch.where(
        mask,
        q.clamp(min=1e-12).log(),
        torch.full_like(q, float("-inf")),
    )
    log_support = (
        log_q.unsqueeze(1)
        - 0.5 * dist2 / sigma.square().clamp(min=1e-12)
    )
    # Invalid query candidates receive zero placeholder energy and stay masked.
    soft_distance2 = (
        -2.0
        * sigma.square()
        * torch.logsumexp(log_support, dim=2)
    ).clamp(min=0.0)
    soft_distance = soft_distance2.sqrt()
    excess = torch.relu(soft_distance - margin)
    energy = (excess / spatial_scale).square()
    return torch.where(mask, energy, torch.zeros_like(energy))


def raw_correction_energies(
    model: TrustRegionAnchorModel,
    rss: torch.Tensor,
    base_probability: torch.Tensor,
    support: Mapping[str, torch.Tensor],
    *,
    config: Mapping[str, object] = V184_CONFIG,
) -> dict[str, torch.Tensor]:
    """Compute detached radio/geometry energies for one frozen base posterior."""

    indices = support["candidate_indices"]
    mask = support["candidate_mask"].bool()
    prior = base_probability.detach() * mask.to(base_probability.dtype)
    prior = prior / prior.sum(dim=1, keepdim=True).clamp(min=1e-12)

    fields = {
        "detection_probability": model.anchor_detection_prob.detach(),
        "rss_center": model.anchor_rss_center.detach(),
        "rss_scale": model.anchor_rss_scale.detach(),
    }
    # The profiler optimizes only two temporary scan-level nuisance tensors.
    # Detaching query/prior and all anchor fields prevents model-parameter
    # gradients while still allowing those nuisance tensors to use autograd.
    with torch.enable_grad():
        radio_energy, style = profile_shared_affine_marked_energy(
            rss.detach(),
            indices.detach(),
            fields,
            prior.detach(),
            degrees_freedom=float(config["student_df"]),
            reliability=model.v184_ap_reliability.detach(),
            steps=int(config["radio_profile_steps"]),
            learning_rate=float(config["radio_profile_learning_rate"]),
            offset_max=float(config["radio_profile_offset_max"]),
            log_scale_max=float(config["radio_profile_log_scale_max"]),
            offset_weight=float(config["radio_profile_offset_weight"]),
            log_scale_weight=float(config["radio_profile_log_scale_weight"]),
        )
    radio_energy = torch.where(
        mask,
        radio_energy.to(base_probability),
        torch.zeros_like(base_probability),
    )
    locations = model.anchor_coords[indices]
    geometry_energy = _masked_geometry_energy(
        prior,
        locations,
        mask,
        spatial_scale_m=model.v184_spatial_scale_m,
        sigma_scale=float(config["geometry_sigma_scale"]),
        margin_scale=float(config["geometry_margin_scale"]),
    ).detach()
    return {
        "radio_energy_raw": radio_energy.detach(),
        "geometry_energy_raw": geometry_energy,
        "style_offset": style["style_offset"].to(base_probability).detach(),
        "style_scale": style["style_scale"].to(base_probability).detach(),
    }


def _mismatch_values(
    base_probability: torch.Tensor,
    mask: torch.Tensor,
    radio_energy: torch.Tensor,
    geometry_energy: torch.Tensor,
    style_offset: torch.Tensor,
    style_scale: torch.Tensor,
    *,
    config: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    q = base_probability * mask.to(base_probability.dtype)
    q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-12)
    radio_expectation = torch.sum(q * radio_energy, dim=1)
    geometry_expectation = torch.sum(q * geometry_energy, dim=1)

    offset_norm = (
        style_offset.abs()
        / max(float(config["radio_profile_offset_max"]), 1e-8)
    ).clamp(max=1.0)
    log_gain_norm = (
        torch.log(style_scale.clamp(min=1e-8)).abs()
        / max(float(config["radio_profile_log_scale_max"]), 1e-8)
    ).clamp(max=1.0)
    radio_mismatch = (
        0.50 * (1.0 - torch.exp(-radio_expectation.clamp(min=0.0)))
        + 0.25 * offset_norm
        + 0.25 * log_gain_norm
    )
    return radio_mismatch, geometry_expectation


def fit_correction_reference(
    *,
    radio_energy_raw: torch.Tensor,
    geometry_energy_raw: torch.Tensor,
    base_probability: torch.Tensor,
    candidate_mask: torch.Tensor,
    style_offset: torch.Tensor,
    style_scale: torch.Tensor,
    config: Mapping[str, object] = V184_CONFIG,
) -> CorrectionReference:
    """Fit label-free correction scales/reliability CDFs on fitting scans."""

    radio_scale = robust_energy_scale(
        radio_energy_raw,
        candidate_mask,
    )
    geometry_scale = robust_energy_scale(
        geometry_energy_raw,
        candidate_mask,
    )
    radio = normalize_energy(
        radio_energy_raw,
        candidate_mask,
        radio_scale,
    )
    geometry = normalize_energy(
        geometry_energy_raw,
        candidate_mask,
        geometry_scale,
    )
    radio_mismatch, geometry_mismatch = _mismatch_values(
        base_probability,
        candidate_mask.bool(),
        radio,
        geometry,
        style_offset,
        style_scale,
        config=config,
    )
    return CorrectionReference(
        radio_energy_scale=radio_scale.detach().cpu(),
        geometry_energy_scale=geometry_scale.detach().cpu(),
        radio_mismatch_reference=fit_empirical_reference(
            radio_mismatch
        ).detach().cpu(),
        geometry_mismatch_reference=fit_empirical_reference(
            geometry_mismatch
        ).detach().cpu(),
    )


def apply_correction(
    *,
    base_probability: torch.Tensor,
    candidate_mask: torch.Tensor,
    raw: Mapping[str, torch.Tensor],
    reference: CorrectionReference,
    epsilon_kl: float,
    config: Mapping[str, object] = V184_CONFIG,
) -> dict[str, torch.Tensor]:
    """Apply reliability-weighted correction inside the KL trust region."""

    if float(epsilon_kl) not in {
        float(value) for value in config["epsilon_kl_grid"]
    }:
        raise ValueError("epsilon_kl is outside the registered V18.4 grid")
    local_reference = reference.to(base_probability.device)
    radio = normalize_energy(
        raw["radio_energy_raw"],
        candidate_mask,
        local_reference.radio_energy_scale,
    )
    geometry = normalize_energy(
        raw["geometry_energy_raw"],
        candidate_mask,
        local_reference.geometry_energy_scale,
    )
    radio_mismatch, geometry_mismatch = _mismatch_values(
        base_probability,
        candidate_mask.bool(),
        radio,
        geometry,
        raw["style_offset"],
        raw["style_scale"],
        config=config,
    )
    radio_reliability = empirical_survival_reliability(
        radio_mismatch,
        local_reference.radio_mismatch_reference,
    )
    geometry_reliability = empirical_survival_reliability(
        geometry_mismatch,
        local_reference.geometry_mismatch_reference,
    )
    combined = (
        radio_reliability.unsqueeze(1) * radio
        + geometry_reliability.unsqueeze(1) * geometry
    )
    corrected = trust_region_posterior_correction(
        base_probability,
        combined,
        epsilon_kl=float(epsilon_kl),
        mask=candidate_mask,
    )
    return {
        **corrected,
        "radio_energy": radio,
        "geometry_energy": geometry,
        "combined_energy": combined,
        "radio_mismatch": radio_mismatch,
        "geometry_mismatch": geometry_mismatch,
        "radio_reliability": radio_reliability,
        "geometry_reliability": geometry_reliability,
    }


def v184_predict(
    model: TrustRegionAnchorModel,
    rss: torch.Tensor,
    *,
    correction_reference: CorrectionReference | None = None,
    epsilon_kl: float | None = None,
    posterior_temperature: float = 1.0,
    scale_multiplier: float = 1.0,
    config: Mapping[str, object] = V184_CONFIG,
) -> dict[str, torch.Tensor]:
    """Predict from natural support; optionally apply frozen structured correction."""

    validate_v184_config(config)
    if float(scale_multiplier) <= 0.0:
        raise ValueError("scale_multiplier must be positive")

    invariant_logits, embedding = model.retrieval_logits(rss)
    prototype_score = prototype_retrieval_score(model, rss)
    support = candidate_support(
        model,
        invariant_logits.detach(),
        prototype_score.detach(),
        config=config,
    )
    base = base_posterior(
        model,
        rss,
        embedding,
        invariant_logits,
        support,
        posterior_temperature=posterior_temperature,
    )
    base_probability = base["probability"]
    final_probability = base_probability
    correction: dict[str, torch.Tensor] | None = None

    if correction_reference is not None:
        if epsilon_kl is None:
            raise ValueError("epsilon_kl is required with correction reference")
        raw = raw_correction_energies(
            model,
            rss,
            base_probability,
            support,
            config=config,
        )
        correction = apply_correction(
            base_probability=base_probability,
            candidate_mask=base["candidate_mask"],
            raw=raw,
            reference=correction_reference,
            epsilon_kl=float(epsilon_kl),
            config=config,
        )
        final_probability = correction["posterior"]
    elif epsilon_kl is not None:
        raise ValueError("epsilon_kl without correction reference is undefined")

    component_location, component_scale = model._candidate_components(
        embedding,
        base["candidate_indices"],
    )
    component_scale = component_scale * float(scale_multiplier)
    # Point decisions are inference outputs, not training objectives.
    with torch.no_grad():
        gm = weighted_component_geometric_median(
            final_probability.detach(),
            component_location.detach(),
            mask=base["candidate_mask"],
        )
        mean = torch.sum(
            final_probability.detach().unsqueeze(2)
            * component_location.detach(),
            dim=1,
        )

    result = {
        "coords": gm,
        "geometric_median": gm,
        "posterior_mean": mean,
        "candidate_indices": base["candidate_indices"],
        "candidate_mask": base["candidate_mask"],
        "natural_support_size": support["natural_size"],
        "base_probability": base_probability,
        "candidate_weights": final_probability,
        "component_locations": component_location,
        "component_scales": component_scale,
        "invariant_logits": invariant_logits,
        "prototype_score": prototype_score,
    }
    if correction is not None:
        result.update(
            {
                f"correction_{key}": value
                for key, value in correction.items()
            }
        )
    return result
