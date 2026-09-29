"""Consensus operators for a single-projection DAPP ensemble.

These functions contain no labels and no optimizer.  They reduce several
checkpoint posteriors and measurement memories to one common anchor model,
which can then be passed through the unchanged DAPP objective exactly once.
"""

from __future__ import annotations

import math

import torch


def _validate_alpha_consensus_inputs(
    initial_weights: torch.Tensor,
    candidate_indices: torch.Tensor,
    member_weights: torch.Tensor,
    anchor_count: int,
    top_k: int,
    epsilon: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate inputs and return long IDs plus normalized member weights."""
    if initial_weights.ndim != 3:
        raise ValueError("initial_weights must have shape [M, B, K]")
    if not initial_weights.is_floating_point():
        raise ValueError("initial_weights must be floating point")
    if any(size <= 0 for size in initial_weights.shape):
        raise ValueError("initial_weights dimensions must be non-empty")
    if candidate_indices.shape != initial_weights.shape:
        raise ValueError("candidate_indices must match initial_weights")
    if candidate_indices.is_floating_point() or candidate_indices.is_complex():
        raise ValueError("candidate_indices must contain integer anchor IDs")
    if member_weights.shape != initial_weights.shape[:2]:
        raise ValueError("member_weights must have shape [M, B]")
    if anchor_count <= 0 or not 0 < top_k <= anchor_count:
        raise ValueError("top_k must be in [1, anchor_count]")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not math.isfinite(float(alpha)) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    if not torch.isfinite(initial_weights).all():
        raise ValueError("initial_weights must be finite")
    if (initial_weights < 0).any():
        raise ValueError("initial_weights must be non-negative")
    if not torch.isfinite(member_weights).all():
        raise ValueError("member_weights must be finite")
    if (member_weights < 0).any():
        raise ValueError("member_weights must be non-negative")

    indices = candidate_indices.to(
        device=initial_weights.device, dtype=torch.long)
    if (indices < 0).any() or (indices >= anchor_count).any():
        raise ValueError("candidate anchor IDs must be in [0, anchor_count)")
    if indices.shape[2] > 1:
        member_sorted = torch.sort(indices, dim=2).values
        if (member_sorted[:, :, 1:] == member_sorted[:, :, :-1]).any():
            raise ValueError(
                "candidate anchor IDs must be unique within each member/sample")

    weights = member_weights.to(
        device=initial_weights.device, dtype=initial_weights.dtype)
    weight_sums = weights.sum(dim=0, keepdim=True)
    if not torch.isfinite(weight_sums).all() or (weight_sums <= 0).any():
        raise ValueError("member_weights must have a positive sum per sample")
    return indices, weights / weight_sums


def _alpha_consensus_logits(
    values: torch.Tensor,
    normalized_member_weights: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Evaluate the weighted power-mean score along the member axis."""
    expanded_weights = normalized_member_weights.unsqueeze(2)
    if alpha == 0.0:
        return torch.sum(expanded_weights * torch.log(values), dim=0)
    power_mean = torch.sum(
        expanded_weights * values.pow(float(alpha)), dim=0)
    return torch.log(power_mean.clamp(min=1e-30)) / float(alpha)


def _dense_alpha_consensus_prior_validated(
    initial_weights: torch.Tensor,
    candidate_indices: torch.Tensor,
    normalized_member_weights: torch.Tensor,
    anchor_count: int,
    top_k: int,
    epsilon: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense reference after input validation (also used for safe fallback)."""
    members, batch_size, _ = initial_weights.shape
    dense = torch.full(
        (members, batch_size, anchor_count),
        float(epsilon),
        dtype=initial_weights.dtype,
        device=initial_weights.device,
    )
    dense.scatter_(
        2,
        candidate_indices,
        initial_weights.clamp(min=float(epsilon)),
    )
    # The epsilon-completed member opinions are probability distributions,
    # not merely positive scores.  Normalising here makes the executable
    # operator exactly match the alpha-divergence barycenter stated in the
    # paper.  The sparse implementation below applies the same normalisation
    # without materialising [M,B,N].
    dense = dense / dense.sum(dim=2, keepdim=True)
    consensus_logits = _alpha_consensus_logits(
        dense, normalized_member_weights, alpha)
    top_logits, top_indices = torch.topk(
        consensus_logits, k=top_k, dim=1, largest=True, sorted=True)
    return top_indices, torch.softmax(top_logits, dim=1)


def dense_alpha_consensus_prior(
    initial_weights: torch.Tensor,
    candidate_indices: torch.Tensor,
    member_weights: torch.Tensor,
    anchor_count: int,
    top_k: int,
    epsilon: float = 1e-6,
    alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense full-dictionary reference for the alpha-consensus operator.

    This implementation is intentionally retained as an executable definition
    and a correctness oracle for the sparse union implementation.
    """
    indices, normalized = _validate_alpha_consensus_inputs(
        initial_weights,
        candidate_indices,
        member_weights,
        anchor_count,
        top_k,
        epsilon,
        alpha,
    )
    return _dense_alpha_consensus_prior_validated(
        initial_weights,
        indices,
        normalized,
        anchor_count,
        top_k,
        epsilon,
        alpha,
    )


def alpha_consensus_prior(
    initial_weights: torch.Tensor,
    candidate_indices: torch.Tensor,
    member_weights: torch.Tensor,
    anchor_count: int,
    top_k: int,
    epsilon: float = 1e-6,
    alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse exact alpha-divergence barycenter on global anchor identities.

    ``alpha=0`` is the reverse-KL/geometric opinion pool. Values in ``(0, 1]``
    form the normalized weighted power-mean path toward the arithmetic pool.
    Scores are normally evaluated only on each sample's union of member
    supports.  An anchor outside that union has the analytically known
    all-``epsilon`` score.  If the retained boundary does not strictly exceed
    this score, that sample falls back to the dense executable definition; the
    optimization therefore changes cost, not the consensus operator.

    Args:
        initial_weights: member posteriors ``[M, B, K]``.
        candidate_indices: matching global anchor IDs ``[M, B, K]``.
        member_weights: label-free reliability weights ``[M, B]``.
        anchor_count: size of the shared global anchor dictionary.
        top_k: retained consensus support size.
        epsilon: probability floor for anchors absent from a member's Top-K.
        alpha: power order in ``[0, 1]``.
    """
    indices, normalized = _validate_alpha_consensus_inputs(
        initial_weights,
        candidate_indices,
        member_weights,
        anchor_count,
        top_k,
        epsilon,
        alpha,
    )
    members, batch_size, member_support = initial_weights.shape
    union_width = members * member_support

    # A support union cannot supply top_k entries in this case, so every sample
    # necessarily requires the exact dense path.
    if top_k > union_width:
        return _dense_alpha_consensus_prior_validated(
            initial_weights,
            indices,
            normalized,
            anchor_count,
            top_k,
            epsilon,
            alpha,
        )

    # Build all per-sample unions at once.  Sorting gives deterministic global
    # anchor order and an inverse map from every member candidate to its union.
    flat_indices = indices.permute(1, 0, 2).reshape(batch_size, union_width)
    sorted_indices, sort_order = torch.sort(flat_indices, dim=1)
    unique_mask = torch.ones_like(sorted_indices, dtype=torch.bool)
    unique_mask[:, 1:] = sorted_indices[:, 1:] != sorted_indices[:, :-1]
    sorted_union_rank = unique_mask.cumsum(dim=1) - 1
    union_count = unique_mask.sum(dim=1)

    union_indices = torch.full_like(sorted_indices, -1)
    batch_rows = torch.arange(
        batch_size, device=indices.device)[:, None].expand_as(sorted_indices)
    union_indices[
        batch_rows[unique_mask], sorted_union_rank[unique_mask]
    ] = sorted_indices[unique_mask]

    flat_union_rank = torch.empty_like(sorted_union_rank)
    flat_union_rank.scatter_(1, sort_order, sorted_union_rank)
    candidate_union_rank = flat_union_rank.reshape(
        batch_size, members, member_support).permute(1, 0, 2)

    # Accumulate only each observed support value's increment above its
    # member-specific normalised epsilon background.  This is algebraically
    # identical to an
    # [M,B,N] dense fill, but needs only [B,MK] temporary storage rather than
    # either the global dictionary or a local [M,B,MK] tensor.
    candidate_values = initial_weights.clamp(min=float(epsilon))
    completion_mass = (
        candidate_values.sum(dim=2, keepdim=True)
        + float(anchor_count - member_support) * float(epsilon)
    )
    candidate_values = candidate_values / completion_mass
    epsilon_values = torch.full_like(candidate_values, float(epsilon))
    epsilon_values = epsilon_values / completion_mass
    if alpha == 0.0:
        member_background = torch.log(epsilon_values)
        background = torch.sum(
            normalized.unsqueeze(2) * member_background, dim=0
        )[:, :1]
        increments = normalized.unsqueeze(2) * (
            torch.log(candidate_values) - member_background
        )
        union_logits = background.expand(batch_size, union_width).clone()
        union_logits.scatter_add_(
            1,
            flat_union_rank,
            increments.permute(1, 0, 2).reshape(batch_size, union_width),
        )
    else:
        alpha_value = float(alpha)
        member_background = epsilon_values.pow(alpha_value)
        background = torch.sum(
            normalized.unsqueeze(2) * member_background, dim=0
        )[:, :1]
        increments = normalized.unsqueeze(2) * (
            candidate_values.pow(alpha_value) - member_background
        )
        power_mean = background.expand(batch_size, union_width).clone()
        power_mean.scatter_add_(
            1,
            flat_union_rank,
            increments.permute(1, 0, 2).reshape(batch_size, union_width),
        )
        union_logits = torch.log(power_mean.clamp(min=1e-30)) / alpha_value
    valid_union = (
        torch.arange(union_width, device=indices.device)[None, :]
        < union_count[:, None]
    )
    union_logits = union_logits.masked_fill(~valid_union, -torch.inf)
    top_logits, top_union_positions = torch.topk(
        union_logits, k=top_k, dim=1, largest=True, sorted=True)
    top_indices = torch.gather(union_indices, 1, top_union_positions)
    top_weights = torch.softmax(top_logits, dim=1)

    outside_boundary = _alpha_consensus_logits(
        epsilon_values[:, :, :1], normalized, alpha).squeeze(1)
    safe = (union_count == anchor_count) | (
        (union_count >= top_k) & (top_logits[:, -1] > outside_boundary)
    )

    fallback = torch.nonzero(~safe, as_tuple=False).squeeze(1)
    if fallback.numel() > 0:
        dense_indices, dense_weights = _dense_alpha_consensus_prior_validated(
            initial_weights.index_select(1, fallback),
            indices.index_select(1, fallback),
            normalized.index_select(1, fallback),
            anchor_count,
            top_k,
            epsilon,
            alpha,
        )
        top_indices.index_copy_(0, fallback, dense_indices)
        top_weights.index_copy_(0, fallback, dense_weights)
    return top_indices, top_weights


def geometric_consensus_prior(
    initial_weights: torch.Tensor,
    candidate_indices: torch.Tensor,
    member_weights: torch.Tensor,
    anchor_count: int,
    top_k: int,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for the reverse-KL endpoint (alpha=0)."""
    return alpha_consensus_prior(
        initial_weights,
        candidate_indices,
        member_weights,
        anchor_count,
        top_k,
        epsilon=epsilon,
        alpha=0.0,
    )


def weighted_component_mean(
    components: torch.Tensor,
    member_weights: torch.Tensor,
) -> torch.Tensor:
    """Euclidean barycenter of member component coordinates/statistics."""
    if components.ndim < 3:
        raise ValueError("components must have leading dimensions [M, B, ...]")
    if member_weights.shape != components.shape[:2]:
        raise ValueError("member_weights must match component [M, B] axes")
    expansion = (None,) * (components.ndim - 2)
    return torch.sum(member_weights[(...,) + expansion] * components, dim=0)


def moment_match_measurements(
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    member_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match a consensus Bernoulli/conditional-RSS measurement model.

    Detection probabilities are linearly pooled. Conditional RSS centers and
    scales are matched under detection-weighted first and second moments, so a
    member that regards an AP as absent cannot dominate its conditional RSS.
    """
    if not (
        detection_prob.shape == rss_center.shape == rss_scale.shape
        and detection_prob.ndim == 4
    ):
        raise ValueError("measurement tensors must share shape [M, B, K, A]")
    if member_weights.shape != detection_prob.shape[:2]:
        raise ValueError("member_weights must have shape [M, B]")
    normalized = member_weights / member_weights.sum(
        dim=0, keepdim=True).clamp(min=1e-12)
    expanded = normalized[:, :, None, None]
    detection = torch.sum(expanded * detection_prob, dim=0)
    detected_mass = torch.sum(expanded * detection_prob, dim=0).clamp(min=1e-6)
    center = torch.sum(
        expanded * detection_prob * rss_center, dim=0
    ) / detected_mass
    second_moment = torch.sum(
        expanded
        * detection_prob
        * (rss_scale.square() + rss_center.square()),
        dim=0,
    ) / detected_mass
    scale = torch.sqrt(
        (second_moment - center.square()).clamp(min=1e-6))
    return detection, center, scale


def wasserstein_pool_measurements(
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    member_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Pool robust location--scale memories by a conditional W2 barycenter.

    For a common one-dimensional location--scale base distribution with finite
    second moment (the final candidate uses Student-t with four degrees of
    freedom), the 2-Wasserstein barycenter has the weighted-average quantile
    function.  Its location and scale are therefore weighted arithmetic means.
    We additionally condition member weights on AP visibility, so a member that
    regards an AP as absent cannot determine its conditional RSS memory.

    ``rss_center`` and ``rss_scale`` are robust median/MAD summaries.  Unlike
    :func:`moment_match_measurements`, this function does not reinterpret MAD
    as a Gaussian standard deviation or add between-member center variance.
    """
    if not (
        detection_prob.shape == rss_center.shape == rss_scale.shape
        and detection_prob.ndim == 4
    ):
        raise ValueError("measurement tensors must share shape [M, B, K, A]")
    if member_weights.shape != detection_prob.shape[:2]:
        raise ValueError("member_weights must have shape [M, B]")
    if not bool(torch.isfinite(detection_prob).all()) or bool(
        (detection_prob < 0.0).any()
    ):
        raise ValueError("detection_prob must be finite and non-negative")
    if not bool(torch.isfinite(rss_center).all()):
        raise ValueError("rss_center must be finite")
    if not bool(torch.isfinite(rss_scale).all()) or bool(
        (rss_scale < 0.0).any()
    ):
        raise ValueError("rss_scale must be finite and non-negative")

    normalized = member_weights / member_weights.sum(
        dim=0, keepdim=True).clamp(min=1e-12)
    expanded = normalized[:, :, None, None]
    visible_mass = expanded * detection_prob
    detection = visible_mass.sum(dim=0)
    denominator = detection.clamp(min=1e-6)
    center = (visible_mass * rss_center).sum(dim=0) / denominator
    scale = (visible_mass * rss_scale).sum(dim=0) / denominator
    return detection, center, scale.clamp(min=1e-6)
