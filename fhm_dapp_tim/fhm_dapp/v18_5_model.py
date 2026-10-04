"""V18.5 JRT-SP model.

The learned model is a single spatial posterior plus Student-t residual mixture.
Radio evidence is applied only after temperature calibration through the exact
KL trust projection.  Geometry is used for support/targets/residual geometry,
never as an inference correction energy.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
import torch.nn.functional as F

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import build_delaunay_geometry
from fhm_dapp_tim.fhm_dapp.dapp import anchor_ap_reliability
from fhm_dapp_tim.fhm_dapp.model import (
    AnchorStudentTModel,
    SparseAPSetEncoder,
    _masked_lower_median,
    _positive_affine_components,
    gradient_reverse,
)
from fhm_dapp_tim.fhm_dapp.v18_single import delaunay_neighbor_targets
from fhm_dapp_tim.fhm_dapp.v18_5_core import (
    cap_anisotropic_residual_transform,
    masked_softmax,
    posterior_mean,
    radio_trust_projection,
    spatial_energy_distance,
    temperature_base_posterior,
)
from fhm_dapp_tim.fhm_dapp.v18_5_radio import (
    TemperatureRadioReference,
    profile_radio_nuisance,
    radio_compatibility,
)


V185_CONFIG: dict[str, object] = {
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
    "candidate_reranker": True,
    "candidate_reranker_mode": "summary",
    "invariant_retrieval_k": 32,
    "prototype_retrieval_k": 32,
    "floor_retrieval_k": 16,
    "geometry_seed_k": 8,
    "support_ceiling": 128,
    "lambda_spatial_grid": (0.00, 0.10, 0.25, 0.50),
    "temperature_grid": (0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00),
    "epsilon_kl_grid": (0.00, 0.025, 0.050, 0.100),
    "scale_grid": (0.50, 0.75, 1.00, 1.25, 1.50, 2.00),
    "radio_profile_steps": 6,
    "radio_profile_learning_rate": 0.20,
    "radio_profile_offset_max": 0.20,
    "radio_profile_log_scale_max": 0.25,
    "radio_profile_offset_weight": 0.10,
    "radio_profile_log_scale_weight": 0.10,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "maximum_epochs": 60,
    "minimum_epochs": 12,
    "energy_tie_margin_m": 0.05,
    "rss_offset_aug_db": 4.0,
    "rss_scale_aug": 0.08,
    "ap_dropout": 0.10,
    "anchor_weight": 1.0,
    "coordinate_weight": 1.0,
    "invariant_aux_weight": 0.25,
    "hierarchy_weight": 0.50,
    "domain_adversarial_weight": 0.05,
}


def validate_v185_config(
    config: Mapping[str, object] = V185_CONFIG,
) -> None:
    if tuple(map(float, config["lambda_spatial_grid"])) != (
        0.0, 0.10, 0.25, 0.50
    ):
        raise ValueError("V18.5 lambda grid changed")
    if tuple(map(float, config["epsilon_kl_grid"])) != (
        0.0, 0.025, 0.050, 0.100
    ):
        raise ValueError("V18.5 epsilon grid changed")
    if tuple(map(float, config["temperature_grid"])) != (
        0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00
    ):
        raise ValueError("V18.5 temperature grid changed")
    if tuple(map(float, config["scale_grid"])) != (
        0.50, 0.75, 1.00, 1.25, 1.50, 2.00
    ):
        raise ValueError("V18.5 scale grid changed")
    if int(config["residual_experts"]) != 1:
        raise ValueError("V18.5 uses one residual expert")
    if str(config["scan_feature_mode"]) != "bounded_maximal_invariant":
        raise ValueError("V18.5 requires bounded maximal-invariant encoder")
    if not bool(config["candidate_reranker"]):
        raise ValueError("V18.5 requires one learned candidate posterior")
    if str(config["candidate_reranker_mode"]) != "summary":
        raise ValueError("V18.5 registers the summary candidate reranker")
    if float(config["max_residual_m"]) != 30.0:
        raise ValueError("V18.5 residual cap is 30 m")
    if any(float(v) < 0.0 for v in config["lambda_spatial_grid"]):
        raise ValueError("lambda_spatial must be nonnegative")
    if any(float(v) < 0.0 for v in config["epsilon_kl_grid"]):
        raise ValueError("epsilon_KL must be nonnegative")
    primary = (
        int(config["invariant_retrieval_k"])
        + int(config["prototype_retrieval_k"])
        + int(config["floor_retrieval_k"])
    )
    if int(config["support_ceiling"]) < primary:
        raise ValueError("support ceiling is below primary proposal quotas")


class V185SparseAPSetEncoder(SparseAPSetEncoder):
    """Sparse-AP encoder whose visibility is supplied explicitly."""

    def _tokens_masked(
        self,
        rss: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        ap_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rss.ndim != 2 or rss.shape[1] != self.n_aps:
            raise ValueError("rss must be [batch,n_aps]")
        if observed_mask.shape != rss.shape or observed_mask.dtype != torch.bool:
            raise ValueError("observed_mask must be bool with rss shape")
        visible = observed_mask
        if self.training and float(ap_dropout) > 0.0:
            visible = visible & (torch.rand_like(rss) >= float(ap_dropout))

        token_count = min(self.max_visible, self.n_aps)
        ranked = rss.masked_fill(~visible, torch.finfo(rss.dtype).min)
        top_rss, top_ids = torch.topk(
            ranked, k=token_count, dim=1, largest=True, sorted=True
        )
        valid = torch.gather(visible, 1, top_ids)

        negative_inf = torch.finfo(rss.dtype).min
        strongest = rss.masked_fill(~visible, negative_inf).max(dim=1).values
        strongest = torch.where(
            visible.any(dim=1),
            strongest,
            torch.zeros_like(strongest),
        )
        median, absolute_deviation, mean_deviation, maximal_full = (
            _positive_affine_components(rss, visible)
        )
        robust_scale = _masked_lower_median(
            absolute_deviation, visible, empty_value=float("nan")
        )
        robust_scale = torch.where(
            robust_scale > 1e-3, robust_scale, mean_deviation
        )
        robust_scale = torch.nan_to_num(
            robust_scale, nan=0.05
        ).clamp_min(0.05)

        visible_count = visible.sum(dim=1).clamp(min=1)
        rank = torch.arange(
            token_count, device=rss.device, dtype=rss.dtype
        ).unsqueeze(0)
        rank = rank / (
            visible_count.to(rss.dtype).unsqueeze(1) - 1
        ).clamp(min=1)
        count_feature = (
            torch.log1p(visible_count.to(rss.dtype))
            / math.log1p(max(1, self.n_aps))
        ).unsqueeze(1).expand(-1, token_count)

        safe_rss = torch.where(valid, top_rss, torch.zeros_like(top_rss))
        centered = safe_rss - median.unsqueeze(1)
        standardized = (
            centered / robust_scale.unsqueeze(1)
        ).clamp(min=-6.0, max=6.0)
        maximal = torch.gather(maximal_full, 1, top_ids)
        bounded = maximal / (1.0 + maximal.abs())
        strongest_bounded = bounded[:, :1]
        features = torch.stack(
            [
                bounded,
                bounded - strongest_bounded,
                rank,
                count_feature,
            ],
            dim=-1,
        )
        features = torch.where(
            valid.unsqueeze(-1), features, torch.zeros_like(features)
        )

        known = self.seen_ap_mask[top_ids]
        mapped = torch.where(
            known & valid,
            top_ids,
            torch.full_like(top_ids, self.n_aps),
        )
        tokens = self.ap_embedding(mapped) + self.feature_encoder(features)
        padding_mask = ~valid
        no_visible = padding_mask.all(dim=1)
        if bool(no_visible.any()):
            padding_mask = padding_mask.clone()
            tokens = tokens.clone()
            padding_mask[no_visible, 0] = False
            tokens[no_visible, 0] = self.ap_embedding.weight[self.n_aps]
        return tokens, padding_mask

    def forward(
        self,
        rss: torch.Tensor,
        observed_mask: torch.Tensor,
        ap_dropout: float = 0.0,
    ) -> torch.Tensor:
        tokens, padding = self._tokens_masked(
            rss, observed_mask, ap_dropout=ap_dropout
        )
        tokens = self.transformer(tokens, src_key_padding_mask=padding)
        scores = self.pool_score(tokens).squeeze(-1)
        scores = scores.masked_fill(
            padding, torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return F.normalize(self.output(pooled), dim=-1)


class V185AnchorModel(AnchorStudentTModel):
    """Single V18.5 learned posterior with explicit-mask input methods."""

    def __init__(
        self,
        *,
        local_residual_transform: torch.Tensor,
        seen_ap_mask: torch.Tensor,
        max_residual_m: float = 30.0,
        **kwargs,
    ) -> None:
        capped = cap_anisotropic_residual_transform(
            local_residual_transform,
            max_residual_m=max_residual_m,
        )
        super().__init__(
            seen_ap_mask=seen_ap_mask,
            local_residual_transform=capped,
            max_residual_m=max_residual_m,
            prototype_metric="none",
            **kwargs,
        )
        old = self.encoder
        self.encoder = V185SparseAPSetEncoder(
            n_aps=old.n_aps,
            d_model=old.d_model,
            num_heads=int(kwargs.get("num_heads", 4)),
            num_layers=int(kwargs.get("num_layers", 2)),
            max_visible=old.max_visible,
            dropout=float(kwargs.get("dropout", 0.10)),
            seen_ap_mask=seen_ap_mask,
            scan_feature_mode="bounded_maximal_invariant",
        )

    def retrieval_logits_v185(
        self,
        rss: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        ap_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(
            rss, observed_mask, ap_dropout=ap_dropout
        )
        return self.retrieval_logits_from_embedding(embedding), embedding

    @staticmethod
    def augment_scan_style_v185(
        rss: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        offset_std_db: float,
        scale_std: float,
    ) -> torch.Tensor:
        if observed_mask.shape != rss.shape or observed_mask.dtype != torch.bool:
            raise ValueError("observed_mask must be bool with rss shape")
        if float(offset_std_db) <= 0.0 and float(scale_std) <= 0.0:
            return rss
        values = torch.where(
            observed_mask, rss, torch.full_like(rss, float("nan"))
        )
        median = _masked_lower_median(
            values, observed_mask, empty_value=0.0
        ).unsqueeze(1)
        offset = torch.randn(
            len(rss), 1, device=rss.device, dtype=rss.dtype
        ) * (float(offset_std_db) / 40.0)
        gain = 1.0 + torch.randn(
            len(rss), 1, device=rss.device, dtype=rss.dtype
        ) * float(scale_std)
        gain = gain.clamp(0.75, 1.25)
        augmented = median + gain * (rss - median) + offset
        augmented = augmented.clamp(-1.0, 1.0)
        return torch.where(observed_mask, augmented, rss)

    def candidate_evidence_features_v185(
        self,
        rss: torch.Tensor,
        observed_mask: torch.Tensor,
        embedding: torch.Tensor,
        candidate_indices: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        query = rss.float()
        query_mask = observed_mask.float().unsqueeze(1)
        query_values = query.unsqueeze(1)

        probability = self.anchor_detection_prob[candidate_indices].clamp(
            min=0.02, max=0.98
        )
        center = self.anchor_rss_center[candidate_indices]
        scale = self.anchor_rss_scale[candidate_indices]
        intersection = (query_mask * probability).sum(dim=2)
        query_count = query_mask.sum(dim=2).clamp_min(1.0)
        anchor_count = probability.sum(dim=2).clamp_min(1.0)
        union = query_count + anchor_count - intersection
        jaccard = intersection / union.clamp_min(1.0)
        query_coverage = intersection / query_count
        anchor_coverage = intersection / anchor_count

        present_log_support = (
            query_mask * probability.log()
        ).sum(dim=2) / query_count
        present_support = present_log_support.exp()
        missing_conflict = (
            (1.0 - query_mask) * probability
        ).sum(dim=2) / anchor_count

        shared_weight = query_mask * probability
        shared_mass = shared_weight.sum(dim=2).clamp_min(1e-4)
        query_mean = (
            shared_weight * query_values
        ).sum(dim=2) / shared_mass
        center_mean = (
            shared_weight * center
        ).sum(dim=2) / shared_mass
        q_delta = query_values - query_mean.unsqueeze(2)
        c_delta = center - center_mean.unsqueeze(2)
        covariance = (shared_weight * q_delta * c_delta).sum(dim=2)
        q_var = (shared_weight * q_delta.square()).sum(dim=2)
        c_var = (shared_weight * c_delta.square()).sum(dim=2)
        correlation = covariance / (
            q_var * c_var
        ).clamp_min(1e-8).sqrt()
        enough = (query_mask * (probability >= 0.20)).sum(dim=2) >= 3.0
        correlation = torch.where(
            enough,
            correlation.clamp(-1.0, 1.0),
            torch.zeros_like(correlation),
        )
        affine_residual = 1.0 - correlation.square()
        standardized_residual = (
            shared_weight
            * (query_values - center).abs()
            / (scale + 0.05)
        ).sum(dim=2) / shared_mass
        standardized_residual = torch.tanh(
            torch.log1p(standardized_residual) / 2.0
        )

        neural = torch.gather(
            self.retrieval_logits_from_embedding(embedding),
            1,
            candidate_indices,
        )
        sample_count = (
            torch.log1p(self.anchor_sample_count[candidate_indices])
            / math.log(32.0)
        ).clamp(max=1.5)
        return torch.stack(
            [
                self._candidate_standardize(coarse_logits) / 5.0,
                self._candidate_standardize(neural) / 5.0,
                jaccard,
                query_coverage,
                anchor_coverage,
                present_support,
                missing_conflict,
                correlation,
                affine_residual,
                standardized_residual,
                sample_count,
            ],
            dim=2,
        )

    def rerank_candidate_logits_v185(
        self,
        rss: torch.Tensor,
        observed_mask: torch.Tensor,
        embedding: torch.Tensor,
        candidate_indices: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        features = self.candidate_evidence_features_v185(
            rss,
            observed_mask,
            embedding,
            candidate_indices,
            coarse_logits,
        )
        return coarse_logits + self.evidence_reranker(features).squeeze(-1)

    @torch.no_grad()
    def initialize_anchor_keys_v185(
        self,
        prototype_rss: torch.Tensor,
        prototype_observed_mask: torch.Tensor,
        *,
        batch_size: int = 256,
    ) -> None:
        was_training = self.training
        self.eval()
        values = []
        for start in range(0, len(prototype_rss), int(batch_size)):
            values.append(
                self.encoder(
                    prototype_rss[start:start + batch_size].to(
                        self.anchor_keys.device
                    ),
                    prototype_observed_mask[start:start + batch_size].to(
                        self.anchor_keys.device
                    ),
                )
            )
        self.anchor_keys.copy_(torch.cat(values, dim=0))
        self.train(was_training)


def _standardize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1, keepdim=True).sqrt()
    return centered / scale.clamp_min(1e-4)


def prototype_retrieval_score_v185(
    model: V185AnchorModel,
    rss: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    """Explicit-mask proposal distance over observed-union AP channels."""

    prototype = model.prototype_rss.float()
    prototype_mask = model.v185_prototype_observed_mask
    q = rss.float()
    qm = observed_mask
    pm = prototype_mask

    qm_f = qm.float()
    pm_f = pm.float()
    overlap = qm_f @ pm_f.t()
    union = (
        qm_f.sum(dim=1, keepdim=True)
        + pm_f.sum(dim=1).unsqueeze(0)
        - overlap
    ).clamp_min(1.0)

    q_obs = torch.where(qm, q, torch.zeros_like(q))
    p_obs = torch.where(pm, prototype, torch.zeros_like(prototype))
    shared_q2 = q_obs.square() @ pm_f.t()
    shared_p2 = qm_f @ p_obs.square().t()
    cross = q_obs @ p_obs.t()
    shared_squared = (shared_q2 + shared_p2 - 2.0 * cross).clamp_min(0.0)

    mismatch_count = union - overlap
    distance2 = (shared_squared + mismatch_count) / union
    return -distance2.clamp_min(0.0).sqrt()


def _floor_anchor_score(
    model: V185AnchorModel,
    invariant_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    floor_logits = model._marginal_logits(
        invariant_logits, model.anchor_floors
    )
    floor_logp = F.log_softmax(floor_logits, dim=1)
    anchor_score = torch.gather(
        floor_logp,
        1,
        model.anchor_floors.unsqueeze(0).expand(
            len(invariant_logits), -1
        ),
    )
    return _standardize(invariant_logits) + anchor_score, floor_logp


def _scatter_membership(
    indices: torch.Tensor,
    active: torch.Tensor,
    *,
    anchors: int,
) -> torch.Tensor:
    counts = torch.zeros(
        len(indices),
        int(anchors),
        dtype=torch.int32,
        device=indices.device,
    )
    counts.scatter_add_(1, indices, active.to(counts.dtype))
    return counts > 0


def _select_support(
    pool: torch.Tensor,
    mandatory: torch.Tensor,
    priority: torch.Tensor,
    *,
    ceiling: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool((mandatory.sum(dim=1) > int(ceiling)).any()):
        raise RuntimeError("mandatory support exceeds ceiling")
    finite = torch.where(pool, priority, torch.zeros_like(priority))
    span = (
        finite.max(dim=1, keepdim=True).values
        - finite.min(dim=1, keepdim=True).values
        + 10.0
    )
    ranked = (
        priority
        + mandatory.to(priority.dtype) * span
    ).masked_fill(~pool, float("-inf"))
    width = int(pool.sum(dim=1).clamp(max=int(ceiling)).max())
    if width <= 0:
        raise RuntimeError("empty candidate support")
    selected = torch.topk(ranked, width, dim=1).indices
    return selected, torch.gather(pool, 1, selected)


def candidate_support_v185(
    model: V185AnchorModel,
    invariant_logits: torch.Tensor,
    prototype_score: torch.Tensor,
    *,
    force_indices: torch.Tensor | None = None,
    force_weights: torch.Tensor | None = None,
    config: Mapping[str, object] = V185_CONFIG,
) -> dict[str, torch.Tensor]:
    validate_v185_config(config)
    if invariant_logits.shape != prototype_score.shape:
        raise ValueError("proposal matrices must align")
    batch, anchors = invariant_logits.shape
    inv_k = min(int(config["invariant_retrieval_k"]), anchors)
    proto_k = min(int(config["prototype_retrieval_k"]), anchors)
    floor_k = min(int(config["floor_retrieval_k"]), anchors)
    seed_k = min(int(config["geometry_seed_k"]), anchors)
    ceiling = min(int(config["support_ceiling"]), anchors)

    inv_indices = torch.topk(invariant_logits, inv_k, dim=1).indices
    proto_indices = torch.topk(prototype_score, proto_k, dim=1).indices
    floor_score, floor_logp = _floor_anchor_score(model, invariant_logits)
    floor_indices = torch.topk(floor_score, floor_k, dim=1).indices

    primary = torch.zeros(
        batch, anchors, dtype=torch.bool, device=invariant_logits.device
    )
    primary.scatter_(1, inv_indices, True)
    primary.scatter_(1, proto_indices, True)
    primary.scatter_(1, floor_indices, True)

    seed = torch.cat(
        [inv_indices[:, :seed_k], proto_indices[:, :seed_k]], dim=1
    )
    geom_idx = model.v185_geometry_neighbors[seed].reshape(batch, -1)
    geom_active = (
        model.v185_geometry_neighbor_weights[seed] > 0
    ).reshape(batch, -1)
    geometry = _scatter_membership(
        geom_idx, geom_active, anchors=anchors
    )
    natural_pool = primary | geometry

    priority = torch.maximum(
        _standardize(invariant_logits),
        _standardize(prototype_score),
    )
    priority = torch.maximum(priority, _standardize(floor_score))
    tie = torch.arange(
        anchors, device=priority.device, dtype=priority.dtype
    ).unsqueeze(0) * 1e-7
    priority = priority + tie

    natural_indices, natural_mask = _select_support(
        natural_pool, primary, priority, ceiling=ceiling
    )
    if (force_indices is None) != (force_weights is None):
        raise ValueError("force indices/weights must be supplied together")
    if force_indices is None:
        pool = natural_pool
        mandatory = primary
    else:
        forced = _scatter_membership(
            force_indices,
            force_weights > 0,
            anchors=anchors,
        )
        pool = natural_pool | forced
        mandatory = primary | forced
    indices, mask = _select_support(
        pool, mandatory, priority, ceiling=ceiling
    )
    return {
        "candidate_indices": indices,
        "candidate_mask": mask,
        "natural_candidate_indices": natural_indices,
        "natural_candidate_mask": natural_mask,
        "natural_size": natural_mask.sum(dim=1),
        "natural_pool_size_before_ceiling": natural_pool.sum(dim=1),
        "natural_truncated": natural_pool.sum(dim=1) > ceiling,
        "invariant_indices": inv_indices,
        "prototype_indices": proto_indices,
        "floor_indices": floor_indices,
        "floor_log_probability": floor_logp,
    }


def base_posterior_v185(
    model: V185AnchorModel,
    rss: torch.Tensor,
    observed_mask: torch.Tensor,
    embedding: torch.Tensor,
    invariant_logits: torch.Tensor,
    support: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    indices = support["candidate_indices"]
    mask = support["candidate_mask"].bool()
    coarse = torch.gather(invariant_logits, 1, indices)
    logits = model.rerank_candidate_logits_v185(
        rss,
        observed_mask,
        embedding,
        indices,
        coarse,
    )
    return {
        "candidate_indices": indices,
        "candidate_mask": mask,
        "candidate_logits": logits,
        "probability": masked_softmax(logits, mask),
    }


def _weighted_target_mass(
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    match = (
        candidate_indices.unsqueeze(2) == target_indices.unsqueeze(1)
    ).to(target_weights.dtype)
    return (
        torch.sum(match * target_weights.unsqueeze(1), dim=2)
        * candidate_mask.to(target_weights.dtype)
    )


def _support_recall(
    indices: torch.Tensor,
    mask: torch.Tensor,
    target_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    covered = (
        (indices.unsqueeze(2) == target_indices.unsqueeze(1))
        & mask.unsqueeze(2)
    ).any(dim=1)
    return torch.sum(
        covered.to(target_weights.dtype) * target_weights, dim=1
    )


def training_spatial_scale_m(
    memory: Mapping[str, torch.Tensor],
) -> float:
    geometry = build_delaunay_geometry(
        memory["coords"], memory["building"], memory["floor"]
    )
    indices, weights = delaunay_neighbor_targets(geometry)
    anchor = torch.arange(len(indices)).unsqueeze(1)
    active = (weights > 0) & (indices != anchor)
    source = memory["coords"].unsqueeze(1).expand(
        -1, indices.shape[1], -1
    )
    target = memory["coords"][indices]
    distance = torch.linalg.vector_norm(source - target, dim=2)
    positive = distance[active & (distance > 0)]
    if positive.numel() == 0:
        raise ValueError("training geometry has no positive edge")
    return float(torch.median(positive))


def build_v185_model(
    *,
    memory: Mapping[str, torch.Tensor],
    n_aps: int,
    n_buildings: int,
    n_devices: int,
    spatial_scale_m: float,
    device: torch.device,
    config: Mapping[str, object] = V185_CONFIG,
) -> tuple[V185AnchorModel, dict[str, object], torch.Tensor, torch.Tensor]:
    validate_v185_config(config)
    geometry = build_delaunay_geometry(
        memory["coords"], memory["building"], memory["floor"]
    )
    neighbours, neighbour_weights = delaunay_neighbor_targets(geometry)
    required = (
        min(int(config["invariant_retrieval_k"]), len(memory["coords"]))
        + min(int(config["prototype_retrieval_k"]), len(memory["coords"]))
        + min(int(config["floor_retrieval_k"]), len(memory["coords"]))
        + int((neighbour_weights > 0).sum(dim=1).max())
    )
    if required > int(config["support_ceiling"]):
        raise ValueError(
            f"support capacity contract failed: {required}>"
            f"{int(config['support_ceiling'])}"
        )

    detection = memory["detection_prob"]
    owners = []
    for ap in range(int(n_aps)):
        score = []
        for building_id in range(int(n_buildings)):
            active = memory["building"] == building_id
            score.append(
                detection[active, ap].sum()
                if bool(active.any())
                else detection.new_tensor(0.0)
            )
        owners.append(int(torch.stack(score).argmax()))
    owner = torch.tensor(owners, dtype=torch.long)

    model = V185AnchorModel(
        n_aps=int(n_aps),
        anchor_coords=memory["coords"],
        anchor_buildings=memory["building"],
        anchor_floors=memory["floor"],
        seen_ap_mask=memory["seen_ap_mask"],
        prototype_rss=memory["prototype_rss"],
        ap_building_owner=owner,
        n_devices=int(n_devices),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        max_visible=int(config["max_visible"]),
        dropout=float(config["dropout"]),
        student_df=float(config["student_df"]),
        min_scale_m=float(config["min_scale_m"]),
        residual_experts=1,
        scan_feature_mode="bounded_maximal_invariant",
        anchor_detection_prob=memory["detection_prob"],
        anchor_rss_center=memory["rss_center"],
        anchor_rss_scale=memory["rss_scale"],
        anchor_sample_count=memory["counts"].float(),
        candidate_reranker=True,
        candidate_reranker_mode="summary",
        reranker_anchor_decoder=False,
        local_residual_transform=geometry["residual_transform"],
        max_residual_m=float(config["max_residual_m"]),
    ).to(device)
    model.register_buffer(
        "v185_prototype_observed_mask",
        memory["prototype_observed_mask"].to(device),
        persistent=True,
    )
    model.register_buffer(
        "v185_ap_reliability",
        anchor_ap_reliability(
            memory["detection_prob"].float(),
            memory["rss_center"].float(),
            memory["rss_scale"].float(),
        ).to(device),
        persistent=False,
    )
    model.register_buffer(
        "v185_geometry_neighbors", neighbours.to(device), persistent=False
    )
    model.register_buffer(
        "v185_geometry_neighbor_weights",
        neighbour_weights.to(device),
        persistent=False,
    )
    model.register_buffer(
        "v185_spatial_scale_m",
        torch.tensor(
            float(spatial_scale_m), dtype=torch.float32, device=device
        ),
        persistent=False,
    )
    model.initialize_anchor_keys_v185(
        memory["prototype_rss"],
        memory["prototype_observed_mask"],
    )
    return model, geometry, neighbours.to(device), neighbour_weights.to(device)


def v185_training_loss(
    model: V185AnchorModel,
    rss: torch.Tensor,
    observed_mask: torch.Tensor,
    target_coords: torch.Tensor,
    target_anchor: torch.Tensor,
    device_ids: torch.Tensor,
    neighbour_indices: torch.Tensor,
    neighbour_weights: torch.Tensor,
    *,
    lambda_spatial: float,
    config: Mapping[str, object] = V185_CONFIG,
) -> dict[str, torch.Tensor]:
    validate_v185_config(config)
    if float(lambda_spatial) not in set(map(float, config["lambda_spatial_grid"])):
        raise ValueError("lambda_spatial outside V18.5 grid")

    model_rss = model.augment_scan_style_v185(
        rss,
        observed_mask,
        offset_std_db=float(config["rss_offset_aug_db"]),
        scale_std=float(config["rss_scale_aug"]),
    )
    invariant_logits, embedding = model.retrieval_logits_v185(
        model_rss,
        observed_mask,
        ap_dropout=float(config["ap_dropout"]),
    )
    prototype_score = prototype_retrieval_score_v185(
        model, model_rss, observed_mask
    )
    target_indices = neighbour_indices[target_anchor]
    target_weights = neighbour_weights[target_anchor].to(invariant_logits)

    natural = candidate_support_v185(
        model,
        invariant_logits.detach(),
        prototype_score.detach(),
        config=config,
    )
    forced = candidate_support_v185(
        model,
        invariant_logits.detach(),
        prototype_score.detach(),
        force_indices=target_indices,
        force_weights=target_weights,
        config=config,
    )
    base = base_posterior_v185(
        model,
        model_rss,
        observed_mask,
        embedding,
        invariant_logits,
        forced,
    )
    q = base["probability"]
    candidate_indices = base["candidate_indices"]
    candidate_mask = base["candidate_mask"]

    target_mass = _weighted_target_mass(
        candidate_indices,
        candidate_mask,
        target_indices,
        target_weights,
    )
    retained = target_mass.sum(dim=1)
    if bool((retained < 1.0 - 1e-5).any()):
        raise RuntimeError("forced support lost positive target mass")
    target_mass = target_mass / retained.unsqueeze(1).clamp_min(1e-12)
    anchor_nll = -torch.sum(
        target_mass * q.clamp_min(1e-12).log(), dim=1
    ).mean()

    target_location, target_scale = model._components(
        embedding, target_indices
    )
    coordinate_logp = model.student_log_prob(
        target_coords, target_location, target_scale
    )
    coordinate_nll = -torch.sum(
        target_weights * coordinate_logp, dim=1
    ).mean()

    spatial_loss = spatial_energy_distance(
        q,
        model.anchor_coords[candidate_indices],
        target_weights,
        model.anchor_coords[target_indices],
        spatial_scale=model.v185_spatial_scale_m,
        predicted_mask=candidate_mask,
        target_mask=target_weights > 0,
    ).mean()

    invariant_logp = F.log_softmax(invariant_logits, dim=1)
    invariant_aux = -torch.sum(
        target_weights
        * torch.gather(invariant_logp, 1, target_indices),
        dim=1,
    ).mean()
    building_logits = model._marginal_logits(
        invariant_logits, model.anchor_buildings
    )
    floor_logits = model._marginal_logits(
        invariant_logits, model.anchor_floors
    )
    hierarchy = (
        0.25
        * F.cross_entropy(
            building_logits, model.anchor_buildings[target_anchor]
        )
        + F.cross_entropy(
            floor_logits, model.anchor_floors[target_anchor]
        )
    )
    domain = anchor_nll.new_zeros(())
    valid_device = device_ids >= 0
    if model.device_head is not None and bool(valid_device.any()):
        domain = F.cross_entropy(
            model.device_head(
                gradient_reverse(embedding[valid_device])
            ),
            device_ids[valid_device],
        )

    total = (
        float(config["anchor_weight"]) * anchor_nll
        + float(config["coordinate_weight"]) * coordinate_nll
        + float(lambda_spatial) * spatial_loss
        + float(config["invariant_aux_weight"]) * invariant_aux
        + float(config["hierarchy_weight"]) * hierarchy
        + float(config["domain_adversarial_weight"]) * domain
    )
    return {
        "loss": total,
        "anchor_nll": anchor_nll.detach(),
        "coordinate_nll": coordinate_nll.detach(),
        "spatial_energy_loss": spatial_loss.detach(),
        "spatial_contribution": (
            spatial_loss.detach() * float(lambda_spatial)
        ),
        "invariant_aux_loss": invariant_aux.detach(),
        "hierarchy_loss": hierarchy.detach(),
        "domain_loss": domain.detach(),
        "natural_recall_mass": _support_recall(
            natural["natural_candidate_indices"],
            natural["natural_candidate_mask"],
            target_indices,
            target_weights,
        ).mean().detach(),
        "natural_support_size": natural["natural_size"].float().mean().detach(),
        "forced_support_size": candidate_mask.sum(dim=1).float().mean().detach(),
    }


def v185_distribution(
    model: V185AnchorModel,
    rss: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    radio_memory: Mapping[str, torch.Tensor],
    radio_reference: TemperatureRadioReference | None,
    temperature: float,
    epsilon_kl: float,
    scale_multiplier: float,
    config: Mapping[str, object] = V185_CONFIG,
) -> dict[str, torch.Tensor]:
    """Full V18.5 predictive distribution and registered posterior mean."""

    validate_v185_config(config)
    if float(temperature) not in set(map(float, config["temperature_grid"])):
        raise ValueError("temperature outside V18.5 grid")
    if float(epsilon_kl) not in set(map(float, config["epsilon_kl_grid"])):
        raise ValueError("epsilon outside V18.5 grid")
    if float(scale_multiplier) not in set(map(float, config["scale_grid"])):
        raise ValueError("scale outside V18.5 grid")

    logits, embedding = model.retrieval_logits_v185(
        rss, observed_mask
    )
    prototype = prototype_retrieval_score_v185(
        model, rss, observed_mask
    )
    support = candidate_support_v185(
        model, logits.detach(), prototype.detach(), config=config
    )
    base = base_posterior_v185(
        model,
        rss,
        observed_mask,
        embedding,
        logits,
        support,
    )
    q_t = temperature_base_posterior(
        base["probability"],
        base["candidate_mask"],
        temperature=float(temperature),
    )
    correction = None
    final = q_t

    if float(epsilon_kl) > 0.0:
        if radio_reference is None:
            raise ValueError("positive epsilon requires a T-matched reference")
        raw = profile_radio_nuisance(
            rss,
            observed_mask,
            base["candidate_indices"],
            radio_memory,
            q_t,
            ap_reliability=model.v185_ap_reliability,
            degrees_freedom=float(config["student_df"]),
            steps=int(config["radio_profile_steps"]),
            learning_rate=float(config["radio_profile_learning_rate"]),
            offset_max=float(config["radio_profile_offset_max"]),
            log_scale_max=float(config["radio_profile_log_scale_max"]),
            offset_weight=float(config["radio_profile_offset_weight"]),
            log_scale_weight=float(config["radio_profile_log_scale_weight"]),
        )
        compatibility = radio_compatibility(
            raw["profiled_objective"],
            radio_reference,
            temperature=float(temperature),
        )
        correction = radio_trust_projection(
            q_t,
            raw["energy"],
            compatibility,
            epsilon_kl=float(epsilon_kl),
            mask=base["candidate_mask"],
        )
        correction.update(raw)
        final = correction["posterior"]

    location, component_scale = model._components(
        embedding, base["candidate_indices"]
    )
    component_scale = component_scale * float(scale_multiplier)
    point = posterior_mean(
        final,
        location,
        base["candidate_mask"],
    )
    return {
        **base,
        "base_probability": base["probability"],
        "temperature_probability": q_t,
        "probability": final,
        "component_location": location,
        "component_scale": component_scale,
        "posterior_mean": point,
        "correction": correction,
        "temperature": final.new_tensor(float(temperature)),
        "epsilon_kl": final.new_tensor(float(epsilon_kl)),
        "scale_multiplier": final.new_tensor(float(scale_multiplier)),
    }


__all__ = [
    "V185_CONFIG",
    "V185AnchorModel",
    "base_posterior_v185",
    "build_v185_model",
    "candidate_support_v185",
    "prototype_retrieval_score_v185",
    "training_spatial_scale_m",
    "v185_distribution",
    "v185_training_loss",
    "validate_v185_config",
]
