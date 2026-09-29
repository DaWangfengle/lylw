"""Minimal joint member-anchor localization primitives for v3.

The point method has four operations only:

1. every frozen member returns a positive categorical distribution on the
   complete, globally aligned anchor dictionary;
2. the uniform member-anchor joint distribution receives one shared
   KL-regularized generalized-loss update;
3. the joint distribution is projected exactly onto at most K anchor
   locations; and
4. the projected anchor marginal is decoded by its spatial expectation.

No coordinate residual head, per-member support, epsilon completion,
divergence-order parameter, reliability learner, or gating rule is present.
"""

from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fhm_dapp_tim.fhm_dapp.dapp import robust_hurdle_anchor_energy
from fhm_dapp_tim.fhm_dapp.model import SparseAPSetEncoder


RADIO_STUDENT_DF = 4.0


class MinimalAnchorModel(nn.Module):
    """One sparse-AP encoder and one full-dictionary categorical head."""

    MODEL_SCHEMA = "minimal_anchor_categorical_v3"

    def __init__(
        self,
        *,
        n_aps: int,
        anchor_coords: torch.Tensor,
        anchor_buildings: torch.Tensor,
        anchor_floors: torch.Tensor,
        seen_ap_mask: torch.Tensor,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_visible: int = 64,
        dropout: float = 0.1,
        visible_threshold: float = -0.75,
        scan_feature_mode: str = "legacy",
    ) -> None:
        super().__init__()
        if len(anchor_coords) <= 0 or anchor_coords.shape[1:] != (2,):
            raise ValueError("anchor_coords must have shape [anchors,2]")
        if anchor_buildings.shape != (len(anchor_coords),):
            raise ValueError("anchor_buildings must contain one value per anchor")
        if anchor_floors.shape != (len(anchor_coords),):
            raise ValueError("anchor_floors must contain one value per anchor")
        self.encoder = SparseAPSetEncoder(
            n_aps=int(n_aps),
            d_model=int(d_model),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            max_visible=int(max_visible),
            dropout=float(dropout),
            visible_threshold=float(visible_threshold),
            seen_ap_mask=seen_ap_mask,
            scan_feature_mode=str(scan_feature_mode),
        )
        self.register_buffer(
            "anchor_coords", anchor_coords.to(dtype=torch.float32), persistent=True)
        self.register_buffer(
            "anchor_buildings",
            anchor_buildings.to(dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "anchor_floors", anchor_floors.to(dtype=torch.long), persistent=True)
        self.anchor_keys = nn.Parameter(
            F.normalize(torch.randn(len(anchor_coords), int(d_model)), dim=-1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    @property
    def n_anchors(self) -> int:
        return int(self.anchor_coords.shape[0])

    @torch.no_grad()
    def initialize_anchor_keys(
        self,
        prototype_rss: torch.Tensor,
        batch_size: int = 256,
    ) -> None:
        if prototype_rss.shape[0] != self.n_anchors:
            raise ValueError("prototype_rss must contain one row per anchor")
        was_training = self.training
        self.eval()
        keys = []
        for start in range(0, len(prototype_rss), int(batch_size)):
            values = prototype_rss[start:start + int(batch_size)].to(
                self.anchor_keys.device)
            keys.append(self.encoder(values))
        self.anchor_keys.copy_(torch.cat(keys, dim=0))
        self.train(was_training)

    def retrieval_logits(
        self,
        rss: torch.Tensor,
        ap_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(rss, ap_dropout=float(ap_dropout))
        keys = F.normalize(self.anchor_keys, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * embedding @ keys.t(), embedding


def load_minimal_v3_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[MinimalAnchorModel, dict]:
    """Load only checkpoints produced by the dedicated minimal v3 model."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("model_schema") != MinimalAnchorModel.MODEL_SCHEMA:
        raise ValueError(f"not a dedicated minimal-v3 checkpoint: {path}")
    state = checkpoint["model_state"]
    model = MinimalAnchorModel(
        n_aps=int(checkpoint["stats"]["n_aps"]),
        anchor_coords=state["anchor_coords"],
        anchor_buildings=state["anchor_buildings"],
        anchor_floors=state["anchor_floors"],
        seen_ap_mask=state["encoder.seen_ap_mask"],
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        max_visible=int(config["max_visible"]),
        dropout=float(config["dropout"]),
        visible_threshold=float(config.get("visible_threshold", -0.75)),
        scan_feature_mode=str(config["scan_feature_mode"]),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def joint_member_anchor_update(
    member_logits: torch.Tensor,
    anchor_loss: torch.Tensor,
    loss_temperature: float,
) -> torch.Tensor:
    r"""Return the closed-form generalized posterior on member-anchor pairs.

    With uniform member mass and member distributions ``r_m(a)``, the result
    uniquely minimizes

    .. math::
        E_q[\ell(A)] + \lambda KL(q(M,A)\Vert \rho(M)r_M(A)).

    It is normalized jointly over both the member and anchor axes.  Posterior
    member masses are therefore implied by this single update rather than
    learned by a separate reliability mechanism.
    """
    if member_logits.ndim != 3:
        raise ValueError("member_logits must have shape [members,batch,anchors]")
    members, batch, anchors = member_logits.shape
    if min(members, batch, anchors) <= 0:
        raise ValueError("member_logits dimensions must be non-empty")
    if anchor_loss.shape != (batch, anchors):
        raise ValueError("anchor_loss must have shape [batch,anchors]")
    if not member_logits.is_floating_point() or not anchor_loss.is_floating_point():
        raise TypeError("member_logits and anchor_loss must be floating point")
    if not bool(torch.isfinite(member_logits).all()):
        raise ValueError("member_logits must be finite")
    if not bool(torch.isfinite(anchor_loss).all()):
        raise ValueError("anchor_loss must be finite")
    temperature = float(loss_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("loss_temperature must be finite and positive")
    log_member_anchor_prior = (
        F.log_softmax(member_logits, dim=2) - math.log(float(members)))
    score = log_member_anchor_prior - anchor_loss.unsqueeze(0) / temperature
    normalizer = torch.logsumexp(
        score.permute(1, 0, 2).reshape(batch, members * anchors),
        dim=1,
    )
    return score - normalizer.unsqueeze(0).unsqueeze(2)


def anchor_log_marginal(log_joint: torch.Tensor) -> torch.Tensor:
    """Marginalize a normalized ``[members,batch,anchors]`` log joint."""
    if log_joint.ndim != 3 or not log_joint.is_floating_point():
        raise ValueError("log_joint must be floating [members,batch,anchors]")
    if not bool(torch.isfinite(log_joint).all()):
        raise ValueError("log_joint must be finite")
    return torch.logsumexp(log_joint, dim=0)


def topk_joint_reverse_kl_projection(
    log_joint: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Project the joint exactly onto distributions using at most K anchors.

    The support constraint applies to the anchor coordinate, not the member.
    For a fixed anchor support the reverse-KL optimum is the renormalized joint;
    hence the global optimum retains the K largest anchor-marginal masses.
    """
    log_marginal = anchor_log_marginal(log_joint)
    candidate_count = _positive_integer(top_k, "top_k")
    if candidate_count > log_marginal.shape[1]:
        raise ValueError("top_k cannot exceed the anchor count")
    _selected_marginal, indices = torch.topk(
        log_marginal, k=candidate_count, dim=1, largest=True, sorted=True)
    gather_indices = indices.unsqueeze(0).expand(log_joint.shape[0], -1, -1)
    selected_log_joint = torch.gather(log_joint, 2, gather_indices)
    batch = log_joint.shape[1]
    normalizer = torch.logsumexp(
        selected_log_joint.permute(1, 0, 2).reshape(
            batch, log_joint.shape[0] * candidate_count),
        dim=1,
    )
    joint_weights = torch.exp(
        selected_log_joint - normalizer.unsqueeze(0).unsqueeze(2))
    return indices, joint_weights


def discrete_energy_score(
    probability: torch.Tensor,
    support_coords: torch.Tensor,
    target_coords: torch.Tensor,
) -> torch.Tensor:
    r"""Multivariate energy score for a finite spatial distribution.

    ``support_coords`` may be shared ``[K,2]`` or row-specific ``[B,K,2]``.
    The returned score is proper and remains defined for a discrete predictive
    distribution, unlike a continuous density log score.
    """
    if probability.ndim != 2 or not probability.is_floating_point():
        raise ValueError("probability must be floating [batch,support]")
    batch, support = probability.shape
    if target_coords.shape != (batch, 2):
        raise ValueError("target_coords must have shape [batch,2]")
    if support_coords.ndim == 2:
        if support_coords.shape != (support, 2):
            raise ValueError("shared support_coords must have shape [support,2]")
        locations = support_coords.unsqueeze(0).expand(batch, -1, -1)
    elif support_coords.shape == (batch, support, 2):
        locations = support_coords
    else:
        raise ValueError("support_coords must be [support,2] or [batch,support,2]")
    if not bool(torch.isfinite(probability).all()) or bool((probability < 0.0).any()):
        raise ValueError("probability must be finite and non-negative")
    totals = probability.sum(dim=1)
    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5):
        raise ValueError("probability rows must sum to one")
    target_distance = torch.linalg.vector_norm(
        locations - target_coords.unsqueeze(1), dim=2)
    pair_distance = torch.linalg.vector_norm(
        locations.unsqueeze(2) - locations.unsqueeze(1), dim=3)
    first = torch.sum(probability * target_distance, dim=1)
    second = 0.5 * torch.einsum(
        "bi,bij,bj->b", probability, pair_distance, probability)
    return first - second


def joint_spatial_moments(
    joint_weights: torch.Tensor,
    support_coords: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return exact candidate/member covariance decomposition."""
    if joint_weights.ndim != 3 or not joint_weights.is_floating_point():
        raise ValueError("joint_weights must be floating [members,batch,support]")
    members, batch, support = joint_weights.shape
    if support_coords.shape != (batch, support, 2):
        raise ValueError("support_coords must have shape [batch,support,2]")
    totals = joint_weights.sum(dim=(0, 2))
    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5):
        raise ValueError("joint weights must sum to one per row")
    member_mass = joint_weights.sum(dim=2)
    conditional = joint_weights / member_mass.clamp_min(1e-30).unsqueeze(2)
    member_mean = torch.einsum("ebk,bkd->ebd", conditional, support_coords)
    mean = torch.einsum("eb,ebd->bd", member_mass, member_mean)
    candidate_delta = support_coords.unsqueeze(0) - member_mean.unsqueeze(2)
    candidate_covariance = torch.einsum(
        "eb,ebk,ebki,ebkj->bij",
        member_mass,
        conditional,
        candidate_delta,
        candidate_delta,
    )
    member_delta = member_mean - mean.unsqueeze(0)
    member_covariance = torch.einsum(
        "eb,ebi,ebj->bij", member_mass, member_delta, member_delta)
    covariance = candidate_covariance + member_covariance
    return {
        "mean": mean,
        "covariance": covariance,
        "candidate_covariance": candidate_covariance,
        "member_covariance": member_covariance,
        "member_mass": member_mass,
        "member_mean": member_mean,
    }


@torch.no_grad()
def full_dictionary_radio_loss(
    query_values: torch.Tensor,
    detection_probability: torch.Tensor,
    rss_center: torch.Tensor,
    rss_mad: torch.Tensor,
    *,
    anchor_chunk_size: int = 64,
) -> torch.Tensor:
    """Evaluate one fixed proper composite radio loss for every anchor."""
    if query_values.ndim != 2 or not query_values.is_floating_point():
        raise ValueError("query_values must be a floating [batch,aps] tensor")
    if not (
        detection_probability.ndim == 2
        and detection_probability.shape == rss_center.shape == rss_mad.shape
    ):
        raise ValueError("radio-memory tensors must share shape [anchors,aps]")
    if detection_probability.shape[1] != query_values.shape[1]:
        raise ValueError("query and radio memory must share the AP dimension")
    chunk_size = _positive_integer(anchor_chunk_size, "anchor_chunk_size")
    query_mask = query_values > -0.999
    equal_ap_weight = torch.ones(
        query_values.shape[1],
        device=query_values.device,
        dtype=query_values.dtype,
    )
    losses = []
    for start in range(0, detection_probability.shape[0], chunk_size):
        stop = min(start + chunk_size, detection_probability.shape[0])
        losses.append(robust_hurdle_anchor_energy(
            detection_prob=detection_probability[start:stop],
            rss_center=rss_center[start:stop],
            rss_scale=rss_mad[start:stop],
            query_values=query_values,
            query_mask=query_mask,
            reliability=equal_ap_weight,
            distribution="student_t",
            degrees_freedom=RADIO_STUDENT_DF,
            score="joint_log",
            visible_factor=1.0,
            invisible_factor=1.0,
        ))
    return torch.cat(losses, dim=1)


def minimal_joint_posterior(
    member_logits: torch.Tensor,
    anchor_loss: torch.Tensor,
    *,
    loss_temperature: float,
    top_k: int,
) -> dict[str, torch.Tensor]:
    """Compose the joint KL update and exact sparse-anchor projection."""
    log_joint = joint_member_anchor_update(
        member_logits, anchor_loss, loss_temperature)
    indices, joint_weights = topk_joint_reverse_kl_projection(log_joint, top_k)
    return {
        "log_joint": log_joint,
        "log_anchor_marginal": anchor_log_marginal(log_joint),
        "candidate_indices": indices,
        "joint_candidate_weights": joint_weights,
        "candidate_weights": joint_weights.sum(dim=0),
    }


def simple_member_anchor_nll(
    model: MinimalAnchorModel,
    rss: torch.Tensor,
    target_anchor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The sole v3 training objective: categorical anchor log loss."""
    if rss.ndim != 2 or target_anchor.shape != (len(rss),):
        raise ValueError("rss and target_anchor must have shapes [B,A] and [B]")
    logits, _embedding = model.retrieval_logits(rss)
    anchor_nll = F.cross_entropy(logits, target_anchor)
    return {"loss": anchor_nll, "anchor_nll": anchor_nll.detach()}
