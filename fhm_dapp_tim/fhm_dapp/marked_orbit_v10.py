"""Mechanism primitives for the V10 marked-orbit experiment.

The module has three responsibilities only:

1. expose the canonical ``[AP, (mark, detection)]`` scan representation and
   its positive-affine quotient/orbit decomposition;
2. add one centered, AP-identity-aware detection statistic to the existing
   sparse invariant set encoder; and
3. marginalize a shared two-dimensional affine nuisance under a fixed
   quadrature prior in the same Gibbs/KL update as the member--anchor law.

Dataset roles, checkpoints, selection, and outcome-access gates are kept in
the registered experiment driver.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fhm_dapp_tim.fhm_dapp.dapp import robust_hurdle_anchor_energy
from fhm_dapp_tim.fhm_dapp.model import (
    AnchorStudentTModel,
    SparseAPSetEncoder,
    _positive_affine_components,
)


VISIBLE_THRESHOLD = -0.999


def proper_marked_energy_on_support(
    query: torch.Tensor,
    anchor_indices: torch.Tensor,
    fields: Mapping[str, torch.Tensor],
    *,
    degrees_freedom: float = 4.0,
    reliability: torch.Tensor | None = None,
    style_offset: float | torch.Tensor | None = None,
    style_scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the fixed-AP proper joint hurdle log score on a support.

    Unlike V9's channel-balanced composite loss, every AP is normalized by
    the same fixed AP-universe weight.  This is exactly the negative joint log
    density of the declared Bernoulli/conditional-Student-t marked model.
    """

    if query.ndim != 2 or not query.is_floating_point():
        raise ValueError("query must be floating [batch,aps]")
    if anchor_indices.ndim != 2 or len(anchor_indices) != len(query):
        raise ValueError("anchor_indices must be [batch,K]")
    required = ("detection_probability", "rss_center", "rss_scale")
    if any(name not in fields for name in required):
        raise KeyError("marked memory misses a required anchor--AP field")
    probability = fields["detection_probability"][anchor_indices]
    center = fields["rss_center"][anchor_indices]
    scale_field = fields["rss_scale"]
    if scale_field.ndim == 1:
        scale = scale_field.view(1, 1, -1).expand_as(center)
    else:
        scale = scale_field[anchor_indices]
    if reliability is None:
        reliability = torch.ones(
            query.shape[1], device=query.device, dtype=query.dtype
        )
    visible = query > VISIBLE_THRESHOLD
    result = robust_hurdle_anchor_energy(
        probability,
        center,
        scale,
        query,
        visible.to(query.dtype),
        reliability,
        style_offset=style_offset,
        style_scale=style_scale,
        distribution="student_t",
        degrees_freedom=float(degrees_freedom),
        score="joint_log",
    )
    if not isinstance(result, torch.Tensor):
        raise TypeError("proper marked score unexpectedly returned diagnostics")
    return result


def marked_scan_tensor(
    rss: torch.Tensor, *, visible_threshold: float = VISIBLE_THRESHOLD
) -> torch.Tensor:
    """Return ``[batch, AP, 2]`` with channels ``(observed mark, detection)``.

    The mark channel is zero when an AP is not detected.  It is a numerical
    placeholder whose meaning is determined entirely by the second channel.
    """

    if rss.ndim != 2 or not rss.is_floating_point():
        raise ValueError("rss must be a floating [batch,aps] tensor")
    if not bool(torch.isfinite(rss).all()):
        raise ValueError("rss must be finite")
    detected = rss > float(visible_threshold)
    mark = torch.where(detected, rss, torch.zeros_like(rss))
    return torch.stack((mark, detected.to(rss.dtype)), dim=2)


def orbit_decompose_scan(
    rss: torch.Tensor, *, visible_threshold: float = VISIBLE_THRESHOLD
) -> dict[str, torch.Tensor]:
    """Decompose a marked scan into detection, invariant shape, and orbit.

    The location coordinate is the deterministic lower visible median and the
    positive scale coordinate is the mean absolute deviation around it.
    Singleton and constant visible subvectors have scale zero and invariant
    zero; their visible marks are still exactly reconstructed by the location
    coordinate.
    """

    marked = marked_scan_tensor(rss, visible_threshold=visible_threshold)
    detected = marked[..., 1].bool()
    location, _absolute_deviation, scale, invariant = (
        _positive_affine_components(rss, detected)
    )
    finite_scale = torch.where(
        torch.isfinite(scale), scale, torch.zeros_like(scale)
    )
    bounded = invariant / (1.0 + invariant.abs())
    bounded = torch.where(
        detected, bounded, torch.full_like(bounded, -1.0)
    )
    visible_count = detected.sum(dim=1)
    degenerate = (visible_count < 2) | (finite_scale <= 0.0)
    return {
        "marked_tensor": marked,
        "detection": detected,
        "invariant": invariant,
        "bounded_invariant": bounded,
        "orbit_location": location,
        "orbit_scale": finite_scale,
        "visible_count": visible_count,
        "degenerate": degenerate,
    }


def reconstruct_visible_marks(
    decomposition: Mapping[str, torch.Tensor], *, missing_placeholder: float = 0.0
) -> torch.Tensor:
    """Reconstruct all visible marks from a V10 orbit decomposition."""

    required = ("detection", "invariant", "orbit_location", "orbit_scale")
    if any(name not in decomposition for name in required):
        raise KeyError("orbit decomposition is incomplete")
    detected = decomposition["detection"]
    invariant = decomposition["invariant"]
    location = decomposition["orbit_location"]
    scale = decomposition["orbit_scale"]
    if detected.shape != invariant.shape or detected.dtype != torch.bool:
        raise ValueError("detection and invariant must be matching AP tensors")
    if location.shape != (len(invariant),) or scale.shape != (len(invariant),):
        raise ValueError("orbit coordinates must contain one value per scan")
    reconstructed = location[:, None] + scale[:, None] * invariant
    return torch.where(
        detected,
        reconstructed,
        torch.full_like(reconstructed, float(missing_placeholder)),
    )


class ExplicitMarkedInvariantEncoder(SparseAPSetEncoder):
    """Sparse invariant mark encoder plus one centered detection statistic.

    The detection statistic shares the AP identity embedding with the visible
    mark tokens.  No second transformer or learned mixture gate is introduced.
    """

    def __init__(
        self,
        n_aps: int,
        *,
        detection_prevalence: torch.Tensor,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_visible: int = 64,
        dropout: float = 0.1,
        visible_threshold: float = VISIBLE_THRESHOLD,
        seen_ap_mask: torch.Tensor | None = None,
        scan_feature_mode: str = "bounded_maximal_invariant",
    ) -> None:
        prevalence = torch.as_tensor(detection_prevalence, dtype=torch.float32)
        if prevalence.shape != (int(n_aps),):
            raise ValueError("detection_prevalence must contain one value per AP")
        if not bool(torch.isfinite(prevalence).all()) or bool(
            ((prevalence < 0.0) | (prevalence > 1.0)).any()
        ):
            raise ValueError("detection_prevalence must be finite in [0,1]")
        super().__init__(
            n_aps=n_aps,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_visible=max_visible,
            dropout=dropout,
            visible_threshold=visible_threshold,
            seen_ap_mask=seen_ap_mask,
            scan_feature_mode=scan_feature_mode,
        )
        self.register_buffer(
            "detection_prevalence", prevalence, persistent=True
        )
        self.detection_norm = nn.LayerNorm(self.d_model)

    def detection_summary(
        self,
        detected: torch.Tensor,
        *,
        expected_prevalence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the centered AP-identity detection representation."""

        if detected.ndim != 2 or detected.shape[1] != self.n_aps:
            raise ValueError("detected must have shape [batch,n_aps]")
        if detected.dtype != torch.bool:
            raise TypeError("detected must be boolean")
        expected = (
            self.detection_prevalence
            if expected_prevalence is None
            else torch.as_tensor(
                expected_prevalence,
                device=detected.device,
                dtype=self.detection_prevalence.dtype,
            )
        )
        if expected.shape != (self.n_aps,):
            raise ValueError("expected_prevalence must contain one value per AP")
        ap_ids = torch.arange(self.n_aps, device=detected.device)
        mapped_ids = torch.where(
            self.seen_ap_mask,
            ap_ids,
            torch.full_like(ap_ids, self.n_aps),
        )
        identity = self.ap_embedding(mapped_ids)
        centered = detected.to(identity.dtype) - expected.to(identity.dtype)
        summary = centered @ identity / math.sqrt(float(self.n_aps))
        return self.detection_norm(summary)

    def forward(self, rss: torch.Tensor, ap_dropout: float = 0.0) -> torch.Tensor:
        if rss.ndim != 2 or rss.shape[1] != self.n_aps:
            raise ValueError(
                f"expected RSS tensor [batch, {self.n_aps}], got {rss.shape}"
            )
        if not 0.0 <= float(ap_dropout) < 1.0:
            raise ValueError("ap_dropout must lie in [0,1)")

        detected = rss > self.visible_threshold
        dropout_value = float(ap_dropout) if self.training else 0.0
        if dropout_value > 0.0:
            keep = torch.rand_like(rss) >= dropout_value
            detected = detected & keep
        missing_value = min(-1.0, self.visible_threshold - 1e-3)
        augmented = torch.where(
            detected, rss, torch.full_like(rss, missing_value)
        )

        # The parent token builder sees the exact same dropped occurrence set;
        # no second AP-dropout draw is made.
        tokens, padding_mask = self._tokens(augmented, ap_dropout=0.0)
        tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        scores = self.pool_score(tokens).squeeze(-1)
        scores = scores.masked_fill(
            padding_mask, torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=1)
        mark_summary = torch.sum(tokens * weights.unsqueeze(-1), dim=1)

        expected = self.detection_prevalence * (1.0 - dropout_value)
        occurrence_summary = self.detection_summary(
            detected, expected_prevalence=expected
        )
        pooled = (mark_summary + occurrence_summary) / math.sqrt(2.0)
        return F.normalize(self.output(pooled), dim=-1)


def install_explicit_marked_encoder(
    model: AnchorStudentTModel,
    detection_prevalence: torch.Tensor,
) -> AnchorStudentTModel:
    """Replace a fresh model's sparse encoder before anchor-key initialization."""

    old = model.encoder
    if not isinstance(old, SparseAPSetEncoder):
        raise TypeError("model does not contain the registered sparse AP encoder")
    layer = old.transformer.layers[0]
    dropout = float(layer.dropout.p)
    replacement = ExplicitMarkedInvariantEncoder(
        n_aps=old.n_aps,
        detection_prevalence=detection_prevalence,
        d_model=old.d_model,
        num_heads=int(layer.self_attn.num_heads),
        num_layers=len(old.transformer.layers),
        max_visible=old.max_visible,
        dropout=dropout,
        visible_threshold=old.visible_threshold,
        seen_ap_mask=old.seen_ap_mask.detach().clone(),
        scan_feature_mode=old.scan_feature_mode,
    )
    device = next(model.parameters()).device
    model.encoder = replacement.to(device=device)
    return model


def tensor_legendre_uniform_quadrature(
    *,
    order: int = 5,
    offset_bounds: Sequence[float] = (-0.20, 0.20),
    log_gain_bounds: Sequence[float] = (-0.25, 0.25),
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | int | tuple[float, float]]:
    """Return a normalized tensor Gauss-Legendre rule for a uniform prior."""

    count = int(order)
    if count < 1:
        raise ValueError("quadrature order must be positive")

    def bounds(value: Sequence[float], name: str) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain lower and upper bounds")
        lower, upper = map(float, value)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"{name} bounds are invalid")
        return lower, upper

    offset = bounds(offset_bounds, "offset")
    log_gain = bounds(log_gain_bounds, "log_gain")
    base_node, base_weight = np.polynomial.legendre.leggauss(count)

    def transform(interval: tuple[float, float]) -> np.ndarray:
        lower, upper = interval
        return (lower + upper) / 2.0 + (upper - lower) / 2.0 * base_node

    offset_node = transform(offset)
    gain_node = transform(log_gain)
    nodes = np.stack(
        np.meshgrid(offset_node, gain_node, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    # For a uniform expectation, each one-dimensional Legendre weight is
    # divided by two; interval length cancels with uniform density.
    weights = np.outer(base_weight / 2.0, base_weight / 2.0).reshape(-1)
    target_device = torch.device("cpu") if device is None else torch.device(device)
    node_tensor = torch.as_tensor(nodes, device=target_device, dtype=dtype)
    weight_tensor = torch.as_tensor(weights, device=target_device, dtype=dtype)
    weight_tensor = weight_tensor / weight_tensor.sum()
    return {
        "nodes": node_tensor,
        "weights": weight_tensor,
        "order_per_axis": count,
        "offset_bounds": offset,
        "log_gain_bounds": log_gain,
    }


def affine_marked_energy_grid(
    query: torch.Tensor,
    anchor_indices: torch.Tensor,
    fields: Mapping[str, torch.Tensor],
    quadrature_nodes: torch.Tensor,
    *,
    degrees_freedom: float = 4.0,
    reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate marked energy at every shared affine nuisance node."""

    nodes = torch.as_tensor(
        quadrature_nodes, device=query.device, dtype=query.dtype
    )
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("quadrature_nodes must have shape [nodes,2]")
    if len(nodes) == 0 or not bool(torch.isfinite(nodes).all()):
        raise ValueError("quadrature_nodes must be nonempty and finite")
    values = []
    for offset, log_gain in nodes:
        values.append(
            proper_marked_energy_on_support(
                query,
                anchor_indices,
                fields,
                degrees_freedom=degrees_freedom,
                reliability=reliability,
                style_offset=offset,
                style_scale=torch.exp(log_gain),
            )
        )
    return torch.stack(values, dim=2)


def marginal_affine_marked_update(
    prior_joint: torch.Tensor,
    energy_grid: torch.Tensor,
    quadrature_weights: torch.Tensor,
    *,
    lambda_value: float,
) -> dict[str, torch.Tensor]:
    """Marginalize one shared affine nuisance in the Gibbs/KL update.

    ``prior_joint`` is ``[B,E,K]`` and ``energy_grid`` is ``[B,K,H]``.  The
    returned member--anchor law is the exact marginal of the registered
    quadrature-discrete joint law ``q(m,a,h|x)``.
    """

    if prior_joint.ndim != 3 or not prior_joint.is_floating_point():
        raise ValueError("prior_joint must be floating [batch,members,K]")
    if energy_grid.ndim != 3 or energy_grid.shape[:2] != (
        prior_joint.shape[0],
        prior_joint.shape[2],
    ):
        raise ValueError("energy_grid must have shape [batch,K,nodes]")
    weights = torch.as_tensor(
        quadrature_weights,
        device=prior_joint.device,
        dtype=prior_joint.dtype,
    )
    if weights.shape != (energy_grid.shape[2],):
        raise ValueError("quadrature_weights and energy nodes disagree")
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
        raise ValueError("quadrature weights must be finite and positive")
    weights = weights / weights.sum()
    temperature = float(lambda_value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("lambda_value must be positive and finite")

    energy = energy_grid.to(device=prior_joint.device, dtype=prior_joint.dtype)
    if not bool(torch.isfinite(energy).all()):
        raise ValueError("energy_grid must be finite")
    log_kernel = weights.log()[None, None, :] - energy / temperature
    log_marginal_factor = torch.logsumexp(log_kernel, dim=2)
    log_member_anchor = prior_joint.clamp_min(1e-30).log()
    log_member_anchor = log_member_anchor + log_marginal_factor[:, None, :]
    flat = log_member_anchor.reshape(len(prior_joint), -1)
    member_anchor = torch.softmax(flat, dim=1).reshape_as(prior_joint)

    anchor_prior = prior_joint.sum(dim=1).clamp_min(1e-30)
    log_nuisance = weights.log()[None, :] + torch.logsumexp(
        anchor_prior.log()[:, :, None] - energy / temperature,
        dim=1,
    )
    nuisance = torch.softmax(log_nuisance, dim=1)
    return {
        "joint_probability": member_anchor,
        "nuisance_probability": nuisance,
        "log_marginal_factor": log_marginal_factor,
    }


def marginal_affine_update_on_support(
    prior_joint: torch.Tensor,
    query: torch.Tensor,
    anchor_indices: torch.Tensor,
    fields: Mapping[str, torch.Tensor],
    quadrature: Mapping[str, torch.Tensor | int | tuple[float, float]],
    *,
    lambda_value: float,
    degrees_freedom: float = 4.0,
    reliability: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate the affine grid and return the marginalized update."""

    nodes = quadrature.get("nodes")
    weights = quadrature.get("weights")
    if not isinstance(nodes, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("quadrature nodes and weights are required")
    energy = affine_marked_energy_grid(
        query,
        anchor_indices,
        fields,
        nodes,
        degrees_freedom=degrees_freedom,
        reliability=reliability,
    )
    result = marginal_affine_marked_update(
        prior_joint,
        energy,
        weights,
        lambda_value=lambda_value,
    )
    result["energy_grid"] = energy
    return result


def nuisance_posterior_diagnostics(
    nuisance_probability: torch.Tensor,
    quadrature_nodes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return interpretable per-scan affine posterior summaries."""

    probability = torch.as_tensor(nuisance_probability)
    nodes = torch.as_tensor(
        quadrature_nodes, device=probability.device, dtype=probability.dtype
    )
    if probability.ndim != 2 or nodes.shape != (probability.shape[1], 2):
        raise ValueError("nuisance probabilities and nodes disagree")
    if not torch.allclose(
        probability.sum(dim=1),
        torch.ones(len(probability), device=probability.device),
        atol=1e-5,
    ):
        raise ValueError("nuisance probabilities must be normalized")
    mean_offset = probability @ nodes[:, 0]
    mean_log_gain = probability @ nodes[:, 1]
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(dim=1)
    return {
        "mean_offset": mean_offset,
        "mean_log_gain": mean_log_gain,
        "mean_gain": probability @ torch.exp(nodes[:, 1]),
        "entropy": entropy,
        "effective_nodes": torch.exp(entropy),
    }
