"""V18.1 single-member geometry-radio probabilistic localizer.

V18.1 fixes the main failure mode observed in V18.0:

1. training never uses a self-anchor mixture shortcut;
2. the primary objective is a leave-self-out V9-style geometry joint log score;
3. the learned radio reranker is trained on leave-anchor-out candidates;
4. inference applies an explicit robust marked-radio posterior correction;
5. posterior temperature and Student-t scale are calibrated separately.

The deployed model still uses exactly one checkpoint and one residual expert.
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
from fhm_dapp_tim.fhm_dapp.model import (
    AnchorStudentTModel,
    gradient_reverse,
)
from fhm_dapp_tim.fhm_dapp.train_anchor import _ap_building_owner
from fhm_dapp_tim.fhm_dapp.v18_single import delaunay_neighbor_targets


VISIBLE_THRESHOLD = -0.999

V181_CONFIG = {
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
    # Invariant neural retrieval + learned radio evidence.
    "scan_feature_mode": "bounded_maximal_invariant",
    "prototype_metric": "none",
    "candidate_reranker": True,
    "candidate_reranker_mode": "summary",
    "reranker_anchor_decoder": False,
    # Optimization.
    "batch_size": 96,
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
    # V18.1 objective. The geometry joint loss has unit coefficient.
    "hierarchy_weight": 0.50,
    "coordinate_weight": 0.05,
    "reranker_weight": 0.50,
    "train_candidates": 64,
    # Explicit radio posterior.
    "radio_lambda": 1.0,
    "radio_profile_steps": 6,
    "radio_profile_learning_rate": 0.20,
    "radio_profile_offset_max": 0.20,
    "radio_profile_log_scale_max": 0.25,
    "radio_profile_offset_weight": 0.10,
    "radio_profile_log_scale_weight": 0.10,
    # Single-model inference. Temperature and scale multiplier are calibrated
    # after training and therefore are not hard-coded here.
    "inference_top_k": 64,
    "condition_building": False,
    "deterministic_building": True,
}


def validate_v181_config(
    config: Mapping[str, object] = V181_CONFIG,
) -> None:
    """Fail closed if a configuration breaks the V18.1 scientific contract."""

    if int(config["residual_experts"]) != 1:
        raise ValueError("V18.1 requires exactly one residual expert")
    if str(config["scan_feature_mode"]) != "bounded_maximal_invariant":
        raise ValueError("V18.1 requires the bounded maximal-invariant encoder")
    if str(config["prototype_metric"]) != "none":
        raise ValueError(
            "V18.1 keeps raw RSS out of neural prototype-distance retrieval"
        )
    if not bool(config["candidate_reranker"]):
        raise ValueError("V18.1 requires the learned radio-evidence reranker")
    if bool(config["reranker_anchor_decoder"]):
        raise ValueError("V18.1 requires the learned local Student-t residual head")
    if int(config["train_candidates"]) <= 0:
        raise ValueError("train_candidates must be positive")
    if int(config["inference_top_k"]) <= 0:
        raise ValueError("inference_top_k must be positive")
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


def build_v181_model(
    *,
    memory: Mapping[str, torch.Tensor],
    n_aps: int,
    n_buildings: int,
    n_devices: int,
    device: torch.device,
    config: Mapping[str, object] = V181_CONFIG,
) -> tuple[AnchorStudentTModel, dict[str, object], torch.Tensor, torch.Tensor]:
    """Build one V18.1 checkpoint and deterministic Delaunay geometry."""

    validate_v181_config(config)
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
        raise KeyError(f"anchor memory misses V18.1 fields: {sorted(missing)}")
    if len(memory["coords"]) < 2:
        raise ValueError("V18.1 requires at least two anchors")

    geometry = build_delaunay_geometry(
        memory["coords"],
        memory["building"],
        memory["floor"],
    )
    neighbour_indices, neighbour_weights = delaunay_neighbor_targets(geometry)
    anchors = torch.arange(len(neighbour_indices), dtype=torch.long).unsqueeze(1)
    positive_self = (
        (neighbour_indices == anchors) & (neighbour_weights > 0.0)
    )
    if bool(positive_self.any()):
        raise ValueError(
            "V18.1 leave-self-out geometry requires at least two anchors "
            "in every building-floor stratum"
        )
    seen = (memory["prototype_rss"] > VISIBLE_THRESHOLD).any(dim=0)
    owner = _ap_building_owner(
        memory["prototype_rss"],
        memory["building"],
        int(n_buildings),
    )
    model = AnchorStudentTModel(
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
    return (
        model,
        geometry,
        neighbour_indices.to(device),
        neighbour_weights.to(device),
    )


def _weighted_support_recall(
    candidate_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    """Return Delaunay target probability mass naturally retrieved per row."""

    matched = (
        candidate_indices.unsqueeze(2) == target_indices.unsqueeze(1)
    ).any(dim=1)
    return torch.sum(
        matched.to(target_weights.dtype) * target_weights,
        dim=1,
    )


def v181_training_loss(
    model: AnchorStudentTModel,
    rss: torch.Tensor,
    target_coords: torch.Tensor,
    target_anchor: torch.Tensor,
    device_ids: torch.Tensor,
    neighbour_indices: torch.Tensor,
    neighbour_weights: torch.Tensor,
    *,
    config: Mapping[str, object] = V181_CONFIG,
) -> dict[str, torch.Tensor]:
    """V18.1 leave-anchor-out geometry-joint training objective.

    The true anchor is not a target component. Geometry targets are the
    non-self Delaunay neighbours. This prevents the training mixture from
    winning by collapsing probability and scale onto the observed anchor.
    """

    validate_v181_config(config)
    model_rss = model.augment_scan_style(
        rss,
        offset_std_db=float(config["rss_offset_aug_db"]),
        scale_std=float(config["rss_scale_aug"]),
    )
    raw_logits, embedding = model.retrieval_logits(
        model_rss,
        ap_dropout=float(config["ap_dropout"]),
    )
    target_indices = neighbour_indices[target_anchor]
    target_weights = neighbour_weights[target_anchor].to(raw_logits)
    if target_indices.shape != target_weights.shape:
        raise ValueError("Delaunay target indices and weights disagree")
    if bool((target_weights < 0.0).any()) or not torch.allclose(
        target_weights.sum(dim=1),
        torch.ones(len(target_weights), device=target_weights.device),
        atol=1e-5,
    ):
        raise ValueError("Delaunay target rows must be normalized")

    # V9-style joint anchor/coordinate log score, now leave-self-out.
    log_anchor = torch.gather(
        F.log_softmax(raw_logits, dim=1),
        1,
        target_indices,
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
    row_retrieval = -(target_weights * log_anchor).sum(dim=1)
    row_coordinate = -(target_weights * log_coordinate).sum(dim=1)
    joint_geometry_loss = (row_retrieval + row_coordinate).mean()

    target_building = model.anchor_buildings[target_anchor]
    target_floor = model.anchor_floors[target_anchor]
    building_logits = model._marginal_logits(
        raw_logits,
        model.anchor_buildings,
    )
    floor_logits = model._marginal_logits(
        raw_logits,
        model.anchor_floors,
    )
    hierarchy_loss = (
        0.25 * F.cross_entropy(building_logits, target_building)
        + F.cross_entropy(floor_logits, target_floor)
    )

    # Natural retrieval diagnostics are measured before target forcing.
    natural_scores = raw_logits.detach().clone()
    natural_scores.scatter_(
        1,
        target_anchor.unsqueeze(1),
        float("-inf"),
    )
    natural_count = min(
        int(config["inference_top_k"]),
        model.n_anchors - 1,
    )
    natural_indices = torch.topk(
        natural_scores,
        natural_count,
        dim=1,
    ).indices
    natural_recall_mass = _weighted_support_recall(
        natural_indices,
        target_indices,
        target_weights,
    ).mean()

    # The learned reranker is trained leave-anchor-out. Delaunay neighbours are
    # forced into the training candidate set only for supervised coverage; the
    # natural recall above reveals whether inference would have found them.
    candidate_count = min(
        max(
            int(config["train_candidates"]),
            target_indices.shape[1],
        ),
        model.n_anchors - 1,
    )
    selection_scores = natural_scores.clone()
    forced = (
        selection_scores.max(dim=1, keepdim=True).values
        + 1.0
        + torch.arange(
            target_indices.shape[1],
            device=rss.device,
            dtype=raw_logits.dtype,
        ).unsqueeze(0)
        * 1e-3
    )
    selection_scores.scatter_(1, target_indices, forced)
    candidate_indices = torch.topk(
        selection_scores,
        candidate_count,
        dim=1,
    ).indices

    coarse_candidate_logits = torch.gather(
        raw_logits,
        1,
        candidate_indices,
    )
    candidate_logits = model.rerank_candidate_logits(
        model_rss,
        embedding,
        candidate_indices,
        coarse_candidate_logits,
    )
    candidate_target_weights = torch.sum(
        (
            candidate_indices.unsqueeze(2)
            == target_indices.unsqueeze(1)
        ).to(raw_logits.dtype)
        * target_weights.unsqueeze(1),
        dim=2,
    )
    target_mass = candidate_target_weights.sum(dim=1, keepdim=True)
    if bool((target_mass <= 0.0).any()):
        raise RuntimeError("forced V18.1 candidates lost all geometry target mass")
    candidate_target_weights = candidate_target_weights / target_mass
    reranker_loss = -torch.sum(
        candidate_target_weights
        * F.log_softmax(candidate_logits, dim=1),
        dim=1,
    ).mean()

    candidate_weights = F.softmax(candidate_logits, dim=1)
    candidate_location, _candidate_scale = model._components(
        embedding,
        candidate_indices,
    )
    point_prediction = torch.sum(
        candidate_weights.unsqueeze(-1) * candidate_location,
        dim=1,
    )
    coordinate_loss = torch.linalg.vector_norm(
        point_prediction - target_coords,
        dim=1,
    ).mean()

    domain_loss = joint_geometry_loss.new_zeros(())
    if (
        model.device_head is not None
        and float(config["domain_adversarial_weight"]) > 0.0
    ):
        device_logits = model.device_head(gradient_reverse(embedding))
        domain_loss = F.cross_entropy(device_logits, device_ids)

    loss = (
        joint_geometry_loss
        + float(config["hierarchy_weight"]) * hierarchy_loss
        + float(config["coordinate_weight"]) * coordinate_loss
        + float(config["reranker_weight"]) * reranker_loss
        + float(config["domain_adversarial_weight"]) * domain_loss
    )
    return {
        "loss": loss,
        "joint_geometry_loss": joint_geometry_loss.detach(),
        "retrieval_nll": row_retrieval.mean().detach(),
        "coordinate_nll": row_coordinate.mean().detach(),
        "hierarchy_loss": hierarchy_loss.detach(),
        "coordinate_loss": coordinate_loss.detach(),
        "reranker_loss": reranker_loss.detach(),
        "domain_loss": domain_loss.detach(),
        "natural_geometry_recall_mass": natural_recall_mass.detach(),
    }


def _radio_fields(model: AnchorStudentTModel) -> dict[str, torch.Tensor]:
    return {
        "detection_probability": model.anchor_detection_prob,
        "rss_center": model.anchor_rss_center,
        "rss_scale": model.anchor_rss_scale,
    }


def explicit_radio_correction(
    model: AnchorStudentTModel,
    rss: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    config: Mapping[str, object] = V181_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the deterministic robust marked-radio posterior correction.

    The nuisance offset/gain profiling is optimized only with respect to two
    per-scan nuisance variables. Model parameters and candidate prior are
    detached, so this helper is an inference correction rather than a hidden
    second learner.
    """

    validate_v181_config(config)
    prior = F.softmax(candidate_logits.detach(), dim=1)
    with torch.enable_grad():
        energy, style = profile_shared_affine_marked_energy(
            rss.detach(),
            candidate_indices.detach(),
            {name: value.detach() for name, value in _radio_fields(model).items()},
            prior,
            degrees_freedom=float(config["student_df"]),
            reliability=None,
            steps=int(config["radio_profile_steps"]),
            learning_rate=float(config["radio_profile_learning_rate"]),
            offset_max=float(config["radio_profile_offset_max"]),
            log_scale_max=float(config["radio_profile_log_scale_max"]),
            offset_weight=float(config["radio_profile_offset_weight"]),
            log_scale_weight=float(config["radio_profile_log_scale_weight"]),
        )
    energy = energy.to(
        device=candidate_logits.device,
        dtype=candidate_logits.dtype,
    )
    corrected = (
        candidate_logits
        - energy / float(config["radio_lambda"])
    )
    return corrected, energy, {
        name: value.to(candidate_logits.device)
        for name, value in style.items()
    }


def _hierarchy_prediction(
    model: AnchorStudentTModel,
    rss: torch.Tensor,
    hierarchy_logits: torch.Tensor,
    *,
    deterministic_building: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    building_logits = model._marginal_logits(
        hierarchy_logits,
        model.anchor_buildings,
    )
    floor_logits = model._marginal_logits(
        hierarchy_logits,
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
        voted = building_scores.argmax(dim=1)
        pred_building = torch.where(
            has_signal,
            voted,
            pred_building,
        )
    return pred_building, pred_floor


def v181_predict(
    model: AnchorStudentTModel,
    rss: torch.Tensor,
    *,
    top_k: int,
    posterior_temperature: float,
    scale_multiplier: float,
    target_coords: torch.Tensor | None = None,
    config: Mapping[str, object] = V181_CONFIG,
) -> dict[str, torch.Tensor]:
    """Single-checkpoint V18.1 prediction with explicit radio correction."""

    validate_v181_config(config)
    if not 1 <= int(top_k) <= model.n_anchors:
        raise ValueError("top_k is outside the anchor dictionary")
    if not math.isfinite(float(posterior_temperature)) or float(
        posterior_temperature
    ) <= 0.0:
        raise ValueError("posterior_temperature must be positive and finite")
    if not math.isfinite(float(scale_multiplier)) or float(
        scale_multiplier
    ) <= 0.0:
        raise ValueError("scale_multiplier must be positive and finite")

    model.eval()
    with torch.no_grad():
        logits, embedding = model.retrieval_logits(rss)
        coarse_logits, candidate_indices = torch.topk(
            logits,
            int(top_k),
            dim=1,
        )
        learned_candidate_logits = model.rerank_candidate_logits(
            rss,
            embedding,
            candidate_indices,
            coarse_logits,
        )
        location, base_scale = model._candidate_components(
            embedding,
            candidate_indices,
        )

    corrected_logits, radio_energy, style = explicit_radio_correction(
        model,
        rss,
        candidate_indices,
        learned_candidate_logits,
        config=config,
    )

    with torch.no_grad():
        weights = F.softmax(
            corrected_logits / float(posterior_temperature),
            dim=1,
        )
        scale = base_scale * float(scale_multiplier)
        pred_building, pred_floor = _hierarchy_prediction(
            model,
            rss,
            logits,
            deterministic_building=bool(config["deterministic_building"]),
        )

        coord_weights = weights
        if bool(config["condition_building"]):
            same_building = (
                model.anchor_buildings[candidate_indices]
                == pred_building.unsqueeze(1)
            )
            conditioned = weights * same_building
            has_candidate = conditioned.sum(dim=1, keepdim=True) > 0.0
            conditioned = conditioned / conditioned.sum(
                dim=1,
                keepdim=True,
            ).clamp(min=1e-12)
            coord_weights = torch.where(
                has_candidate,
                conditioned,
                weights,
            )

        mean = torch.sum(
            coord_weights.unsqueeze(-1) * location,
            dim=1,
        )
        component_variance = (
            scale.square()
            * float(config["student_df"])
            / (float(config["student_df"]) - 2.0)
        )
        within_second_moment = torch.sum(
            coord_weights * component_variance.sum(dim=-1),
            dim=1,
        )
        between_second_moment = torch.sum(
            coord_weights
            * (location - mean.unsqueeze(1)).square().sum(dim=-1),
            dim=1,
        )
        radial_scale = (
            within_second_moment + between_second_moment
        ).clamp(min=1e-8).sqrt()
        entropy = -torch.sum(
            weights * weights.clamp(min=1e-12).log(),
            dim=1,
        )

        result = {
            "coords": mean,
            "building": pred_building,
            "floor": pred_floor,
            "radial_scale": radial_scale,
            "within_second_moment": within_second_moment,
            "between_second_moment": between_second_moment,
            "candidate_indices": candidate_indices,
            "candidate_weights": weights,
            "candidate_logits_corrected": corrected_logits,
            "candidate_logits_learned": learned_candidate_logits,
            "component_locations": location,
            "component_scales": scale,
            "component_scales_uncalibrated": base_scale,
            "radio_energy": radio_energy,
            "style_offset": style["style_offset"],
            "style_scale": style["style_scale"],
            "anchor_entropy": entropy,
            "effective_anchor_count": torch.exp(entropy),
            "top1_probability": weights.max(dim=1).values,
        }
        if target_coords is not None:
            component_lp = model.student_log_prob(
                target_coords,
                location,
                scale,
            )
            result["nll"] = -torch.logsumexp(
                torch.log(weights.clamp(min=1e-12))
                + component_lp,
                dim=1,
            )
        return result
