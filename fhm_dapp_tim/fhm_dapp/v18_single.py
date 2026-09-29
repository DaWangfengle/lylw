"""V18 single-member geometry/radio anchor posterior.

V18 deliberately defines a complete probabilistic localizer with one trained
checkpoint. It combines the strongest mechanisms from V9 and V17 without an
ensemble dependency:

* V9 bounded positive-affine invariant AP-set representation;
* V9 same-building/floor Delaunay geometry for retrieval targets and local
  residual transforms;
* V17 device-adversarial/style-augmentation training;
* V17 learned candidate radio-evidence reranking from detection probability,
  robust RSS center and robust RSS scale;
* one anchor-mixture Student-t predictive distribution.

There is no member axis, consensus operator, member weighting, or ensemble
hyperparameter in this module.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import build_delaunay_geometry
from fhm_dapp_tim.fhm_dapp.model import AnchorStudentTModel
from fhm_dapp_tim.fhm_dapp.train_anchor import _ap_building_owner


VISIBLE_THRESHOLD = -0.999

V18_CONFIG = {
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "max_visible": 64,
    "dropout": 0.10,
    "student_df": 4.0,
    "min_scale_m": 0.25,
    "max_residual_m": 30.0,
    "residual_experts": 1,
    "scan_feature_mode": "bounded_maximal_invariant",
    "prototype_metric": "none",
    "candidate_reranker": True,
    "candidate_reranker_mode": "summary",
    "reranker_anchor_decoder": False,
    "batch_size": 96,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "maximum_epochs": 80,
    "minimum_epochs": 20,
    "patience": 10,
    "minimum_improvement_nll": 1e-3,
    "rss_offset_aug_db": 4.0,
    "rss_scale_aug": 0.08,
    "ap_dropout": 0.10,
    "domain_adversarial_weight": 0.05,
    "retrieval_weight": 0.25,
    "hierarchy_weight": 0.50,
    "self_target_weight": 0.50,
    "coordinate_weight": 0.10,
    "reranker_weight": 0.50,
    "train_candidates": 48,
    "inference_top_k": 64,
    "posterior_temperature": 1.0,
    "condition_building": False,
    "deterministic_building": True,
}


def validate_v18_config(config: Mapping[str, object] = V18_CONFIG) -> None:
    """Fail closed if a configuration silently reintroduces an ensemble."""

    if int(config["residual_experts"]) != 1:
        raise ValueError("V18 is defined with exactly one residual expert")
    if str(config["scan_feature_mode"]) != "bounded_maximal_invariant":
        raise ValueError("V18 requires the bounded maximal-invariant encoder")
    if str(config["prototype_metric"]) != "none":
        raise ValueError(
            "V18 keeps raw RSS out of the neural retrieval representation; "
            "radio evidence enters through the candidate reranker"
        )
    if not bool(config["candidate_reranker"]):
        raise ValueError("V18 requires the single-model radio-evidence reranker")
    if int(config["inference_top_k"]) <= 0:
        raise ValueError("inference_top_k must be positive")
    if int(config["train_candidates"]) <= 0:
        raise ValueError("train_candidates must be positive")


def delaunay_neighbor_targets(
    geometry: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert V9 closed one-ring targets into non-self neighbour targets.

    AnchorStudentTModel.training_loss already assigns explicit mass to the true
    anchor via self_target_weight. V9 geometry includes the anchor in its
    closed one-ring, so V18 removes that duplicate self mass and renormalizes
    the remaining Delaunay neighbours.

    A singleton stratum has no geometric neighbour. In that exceptional case
    the anchor itself is retained as a one-point fallback; together with the
    explicit self target this remains a valid normalized target and avoids
    inventing a cross-floor/cross-building edge.
    """

    indices = torch.as_tensor(geometry["neighbor_indices"], dtype=torch.long)
    weights = torch.as_tensor(geometry["neighbor_weights"], dtype=torch.float32)
    if indices.shape != weights.shape or indices.ndim != 2:
        raise ValueError(
            "geometry neighbour indices/weights must be matching matrices"
        )
    anchors = torch.arange(len(indices), dtype=torch.long).unsqueeze(1)
    active = (weights > 0.0) & (indices != anchors)
    output_weights = torch.where(active, weights, torch.zeros_like(weights))
    totals = output_weights.sum(dim=1, keepdim=True)
    singleton = totals.squeeze(1) <= 0.0

    output_indices = indices.clone()
    for row in range(len(output_indices)):
        valid = torch.nonzero(active[row], as_tuple=False).squeeze(1)
        if len(valid) > 0:
            fallback = output_indices[row, int(valid[0])]
            output_indices[row, ~active[row]] = fallback
        else:
            output_indices[row].fill_(row)
            output_weights[row].zero_()
            output_weights[row, 0] = 1.0

    non_singleton = ~singleton
    if bool(non_singleton.any()):
        output_weights[non_singleton] = (
            output_weights[non_singleton]
            / totals[non_singleton].clamp(min=1e-12)
        )
    if not torch.allclose(
        output_weights.sum(dim=1),
        torch.ones(len(output_weights), dtype=output_weights.dtype),
        atol=1e-6,
    ):
        raise RuntimeError("V18 Delaunay targets are not normalized")
    return output_indices, output_weights


def build_v18_model(
    *,
    memory: Mapping[str, torch.Tensor],
    n_aps: int,
    n_buildings: int,
    n_devices: int,
    device: torch.device,
    config: Mapping[str, object] = V18_CONFIG,
) -> tuple[AnchorStudentTModel, dict[str, object], torch.Tensor, torch.Tensor]:
    """Build the one-checkpoint V18 model and its geometry targets."""

    validate_v18_config(config)
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
        raise KeyError(f"anchor memory misses V18 fields: {sorted(missing)}")

    geometry = build_delaunay_geometry(
        memory["coords"], memory["building"], memory["floor"]
    )
    neighbour_indices, neighbour_weights = delaunay_neighbor_targets(geometry)
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


def v18_training_loss(
    model: AnchorStudentTModel,
    rss: torch.Tensor,
    target_coords: torch.Tensor,
    target_anchor: torch.Tensor,
    device_ids: torch.Tensor,
    neighbour_indices: torch.Tensor,
    neighbour_weights: torch.Tensor,
    *,
    config: Mapping[str, object] = V18_CONFIG,
) -> dict[str, torch.Tensor]:
    """Return the V18 single-member objective."""

    validate_v18_config(config)
    return model.training_loss(
        rss,
        target_coords,
        target_anchor,
        neighbour_indices,
        neighbour_weights,
        retrieval_weight=float(config["retrieval_weight"]),
        hierarchy_weight=float(config["hierarchy_weight"]),
        self_target_weight=float(config["self_target_weight"]),
        coordinate_weight=float(config["coordinate_weight"]),
        device_ids=device_ids,
        domain_adversarial_weight=float(config["domain_adversarial_weight"]),
        rss_offset_aug_db=float(config["rss_offset_aug_db"]),
        rss_scale_aug=float(config["rss_scale_aug"]),
        reranker_weight=float(config["reranker_weight"]),
        train_candidates=int(config["train_candidates"]),
        ap_dropout=float(config["ap_dropout"]),
    )
