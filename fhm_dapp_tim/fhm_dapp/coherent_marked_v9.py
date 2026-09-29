"""Coherent V9 member--anchor front and single marked projection.

This module contains only mechanism-level mathematics.  Dataset roles,
checkpoint provenance, and outcome-access gates live in ``uji_v9_experiment``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay, QhullError

from fhm_dapp_tim.fhm_dapp.dapp import robust_hurdle_anchor_energy
from fhm_dapp_tim.fhm_dapp.model import (
    AnchorStudentTModel,
    bounded_positive_affine_maximal_invariant,
)


VISIBLE_THRESHOLD = -0.999


def maximal_invariant_scan(
    rss: torch.Tensor, *, visible_threshold: float = VISIBLE_THRESHOLD
) -> torch.Tensor:
    """Return the bounded maximal-invariant marked scan.

    Visibility is kept as a separate event through the reserved ``-1`` value.
    For at most ``A`` visible values the standardized magnitude is at most
    ``A``; hence the bounded representative is strictly greater than ``-1``
    and remains distinguishable from the missing sentinel.
    """

    if rss.ndim != 2 or not rss.is_floating_point():
        raise ValueError("rss must be a floating [rows,aps] tensor")
    visible = rss > float(visible_threshold)
    transformed = bounded_positive_affine_maximal_invariant(rss, visible)
    if bool((transformed[visible] <= -1.0).any()):
        raise RuntimeError("a visible invariant mark collided with the sentinel")
    return transformed


def _intrinsic_adjacency(xy: np.ndarray) -> tuple[list[set[int]], int, str]:
    """Build a deterministic Delaunay/adjacent graph in intrinsic rank."""

    count = len(xy)
    adjacency = [set() for _ in range(count)]
    if count <= 1:
        return adjacency, 0, "singleton"
    centered = xy - xy.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    tolerance = max(xy.shape) * np.finfo(np.float64).eps
    tolerance *= float(singular[0]) if len(singular) else 1.0
    rank = int(np.sum(singular > tolerance))
    if rank < 2 or count < 3:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0] if len(vh) else np.asarray([1.0, 0.0])
        projection = centered @ direction
        order = np.lexsort((np.arange(count), projection))
        for left, right in zip(order[:-1], order[1:]):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
        return adjacency, min(rank, 1), "intrinsic_rank_one_adjacency"
    try:
        triangulation = Delaunay(xy)
    except QhullError:
        # Numerically near-collinear coordinates use the same intrinsic rule;
        # no outcome-dependent k is introduced as a rescue parameter.
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        projection = centered @ vh[0]
        order = np.lexsort((np.arange(count), projection))
        for left, right in zip(order[:-1], order[1:]):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
        return adjacency, 1, "qhull_rank_one_fallback"
    for simplex in np.asarray(triangulation.simplices, dtype=np.int64):
        for left_pos in range(len(simplex)):
            for right_pos in range(left_pos + 1, len(simplex)):
                left = int(simplex[left_pos])
                right = int(simplex[right_pos])
                adjacency[left].add(right)
                adjacency[right].add(left)
    return adjacency, 2, "delaunay"


def build_delaunay_geometry(
    anchor_coords: torch.Tensor,
    anchor_buildings: torch.Tensor,
    anchor_floors: torch.Tensor,
) -> dict[str, object]:
    """Construct fixed soft targets and anisotropic residual transforms.

    All quantities are functions of the anchor dictionary alone.  Outcome
    predictions and validation losses cannot enter this construction.
    """

    coords = torch.as_tensor(anchor_coords, dtype=torch.float64).cpu()
    buildings = torch.as_tensor(anchor_buildings, dtype=torch.long).cpu()
    floors = torch.as_tensor(anchor_floors, dtype=torch.long).cpu()
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("anchor_coords must have shape [anchors,2]")
    if buildings.shape != (len(coords),) or floors.shape != (len(coords),):
        raise ValueError("anchor strata must contain one value per coordinate")
    if len(coords) == 0 or not bool(torch.isfinite(coords).all()):
        raise ValueError("anchor coordinates must be nonempty and finite")

    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (building, floor) in enumerate(zip(buildings, floors)):
        strata[(int(building), int(floor))].append(index)

    global_adjacency = [set() for _ in range(len(coords))]
    stratum_diagnostics: dict[str, dict[str, object]] = {}
    coords_np = coords.numpy()
    for stratum, global_indices in sorted(strata.items()):
        local_xy = coords_np[np.asarray(global_indices, dtype=np.int64)]
        adjacency, rank, method = _intrinsic_adjacency(local_xy)
        for local_index, neighbours in enumerate(adjacency):
            source = global_indices[local_index]
            global_adjacency[source].update(global_indices[j] for j in neighbours)
        edge_count = sum(len(value) for value in adjacency) // 2
        stratum_diagnostics[f"{stratum[0]}:{stratum[1]}"] = {
            "anchors": len(global_indices),
            "intrinsic_rank": rank,
            "graph_method": method,
            "undirected_edges": edge_count,
        }

    local_indices: list[list[int]] = []
    local_weights: list[np.ndarray] = []
    transforms = np.zeros((len(coords), 2, 2), dtype=np.float64)
    bandwidths = np.zeros(len(coords), dtype=np.float64)
    degrees = np.zeros(len(coords), dtype=np.int64)
    for anchor, neighbours_set in enumerate(global_adjacency):
        neighbours = sorted(neighbours_set)
        degrees[anchor] = len(neighbours)
        if not neighbours:
            local_indices.append([anchor])
            local_weights.append(np.asarray([1.0], dtype=np.float64))
            continue
        displacement = coords_np[np.asarray(neighbours)] - coords_np[anchor]
        distance = np.linalg.norm(displacement, axis=1)
        positive = distance[distance > 0.0]
        if len(positive) == 0:
            raise RuntimeError("distinct anchors share zero geometric distance")
        bandwidth = float(np.median(positive))
        bandwidths[anchor] = bandwidth
        neighbour_weight = np.exp(
            -np.square(distance) / (2.0 * bandwidth * bandwidth)
        )
        closed_weight = np.concatenate(
            [np.asarray([1.0], dtype=np.float64), neighbour_weight]
        )
        closed_weight /= closed_weight.sum()
        local_indices.append([anchor, *neighbours])
        local_weights.append(closed_weight)

        covariance = np.einsum(
            "i,ij,ik->jk", neighbour_weight, displacement, displacement
        ) / neighbour_weight.sum()
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        eigenvalue = np.clip(eigenvalue, 0.0, None)
        transforms[anchor] = (
            eigenvector * np.sqrt(eigenvalue)[None, :]
        ) @ eigenvector.T

    width = max(map(len, local_indices))
    padded_indices = np.zeros((len(coords), width), dtype=np.int64)
    padded_weights = np.zeros((len(coords), width), dtype=np.float64)
    for anchor, (indices, weights) in enumerate(zip(local_indices, local_weights)):
        padded_indices[anchor, : len(indices)] = indices
        padded_indices[anchor, len(indices) :] = anchor
        padded_weights[anchor, : len(indices)] = weights

    return {
        "neighbor_indices": torch.from_numpy(padded_indices).long(),
        "neighbor_weights": torch.from_numpy(padded_weights).float(),
        "residual_transform": torch.from_numpy(transforms).float(),
        "bandwidth_m": torch.from_numpy(bandwidths).float(),
        "degree": torch.from_numpy(degrees).long(),
        "diagnostics": {
            "definition": "same-building-floor Delaunay closed one-ring",
            "maximum_closed_neighborhood": int(width),
            "minimum_degree": int(degrees.min()),
            "median_degree": float(np.median(degrees)),
            "maximum_degree": int(degrees.max()),
            "strata": stratum_diagnostics,
        },
    }


def geometry_joint_front_loss(
    model: AnchorStudentTModel,
    rss: torch.Tensor,
    target_coords: torch.Tensor,
    target_anchor: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_weights: torch.Tensor,
    *,
    ap_dropout: float = 0.10,
) -> dict[str, torch.Tensor]:
    """One joint anchor/coordinate log score identified by fixed geometry."""

    logits, embedding = model.retrieval_logits(rss, ap_dropout=ap_dropout)
    candidates = neighbor_indices[target_anchor]
    weight = neighbor_weights[target_anchor].to(logits)
    if candidates.shape != weight.shape:
        raise ValueError("geometry target indices and weights disagree")
    if bool((weight < 0.0).any()) or not torch.allclose(
        weight.sum(dim=1), torch.ones(len(weight), device=weight.device), atol=1e-5
    ):
        raise ValueError("each geometry target must be a probability vector")
    log_anchor = torch.gather(F.log_softmax(logits, dim=1), 1, candidates)
    location, scale = model._components(embedding, candidates)
    log_coordinate = model.student_log_prob(target_coords, location, scale)
    row_retrieval = -(weight * log_anchor).sum(dim=1)
    row_coordinate = -(weight * log_coordinate).sum(dim=1)
    row_joint = row_retrieval + row_coordinate
    return {
        "loss": row_joint.mean(),
        "joint_nll": row_joint.mean(),
        "retrieval_nll": row_retrieval.mean(),
        "coordinate_nll": row_coordinate.mean(),
    }


def uniform_member_anchor_prior(
    member_logits: torch.Tensor, *, members: int | None = None
) -> torch.Tensor:
    """Return ``p0(row,member,anchor)=r_m(anchor|x)/E``."""

    if member_logits.ndim != 3 or not member_logits.is_floating_point():
        raise ValueError("member_logits must be floating [members,rows,anchors]")
    count = member_logits.shape[0] if members is None else int(members)
    if not 1 <= count <= member_logits.shape[0]:
        raise ValueError("members is outside the available panel")
    probability = F.softmax(member_logits[:count], dim=2)
    return probability.permute(1, 0, 2) / float(count)


def topk_prior_projection(
    full_joint: torch.Tensor, *, top_k: int
) -> dict[str, torch.Tensor | int]:
    """Project once onto ensemble Top-K anchor support by reverse KL."""

    if full_joint.ndim != 3 or not full_joint.is_floating_point():
        raise ValueError("full_joint must be floating [rows,members,anchors]")
    width = int(top_k)
    if not 1 <= width <= full_joint.shape[2]:
        raise ValueError("top_k is outside the anchor dictionary")
    marginal = full_joint.sum(dim=1)
    _probability, indices = torch.topk(marginal, width, dim=1, sorted=True)
    selected = torch.gather(
        full_joint,
        2,
        indices[:, None, :].expand(-1, full_joint.shape[1], -1),
    )
    retained = selected.sum(dim=(1, 2))
    projected = selected / retained[:, None, None].clamp_min(1e-30)
    return {
        "anchor_indices": indices,
        "joint_probability": projected,
        "retained_mass": retained,
        "reverse_kl": -torch.log(retained.clamp_min(1e-30)),
        "top_k": width,
    }


def _support_fields(
    anchor_indices: torch.Tensor,
    fields: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return probability, center, scale


def marked_energy_on_support(
    query: torch.Tensor,
    anchor_indices: torch.Tensor,
    fields: Mapping[str, torch.Tensor],
    *,
    degrees_freedom: float = 4.0,
    reliability: torch.Tensor | None = None,
    style_offset: float | torch.Tensor | None = None,
    style_scale: float | torch.Tensor | None = None,
    visible_factor: float = 1.0,
    invisible_factor: float = 1.0,
) -> torch.Tensor:
    """Evaluate the registered channel-balanced marked energy."""

    probability, center, scale = _support_fields(anchor_indices, fields)
    visible = query > VISIBLE_THRESHOLD
    if reliability is None:
        reliability = torch.ones(query.shape[1], device=query.device, dtype=query.dtype)
    return robust_hurdle_anchor_energy(
        probability,
        center,
        scale,
        query,
        visible.to(query.dtype),
        reliability,
        style_offset=style_offset,
        style_scale=style_scale,
        visible_factor=float(visible_factor),
        invisible_factor=float(invisible_factor),
        distribution="student_t",
        degrees_freedom=float(degrees_freedom),
        score="balanced_log",
    )


def profile_shared_affine_marked_energy(
    query: torch.Tensor,
    anchor_indices: torch.Tensor,
    fields: Mapping[str, torch.Tensor],
    anchor_prior: torch.Tensor,
    *,
    degrees_freedom: float = 4.0,
    reliability: torch.Tensor | None = None,
    steps: int = 10,
    learning_rate: float = 0.20,
    offset_max: float = 0.20,
    log_scale_max: float = 0.25,
    offset_weight: float = 0.10,
    log_scale_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Profile one scan-level affine nuisance under the Top-K prior.

    The nuisance is common to all anchors and is fitted before lambda is
    applied.  Unlike the old helper, the objective is the registered
    channel-balanced robust-Hurdle energy.
    """

    if anchor_prior.shape != anchor_indices.shape:
        raise ValueError("anchor_prior and anchor_indices must have equal shape")
    rows = len(query)
    raw_offset = torch.zeros(
        rows, device=query.device, dtype=query.dtype, requires_grad=True
    )
    raw_log_scale = torch.zeros_like(raw_offset, requires_grad=True)
    optimizer = torch.optim.Adam(
        [raw_offset, raw_log_scale], lr=float(learning_rate)
    )
    normalized_prior = anchor_prior / anchor_prior.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-30)

    def parameters() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offset = float(offset_max) * torch.tanh(raw_offset)
        log_gain = float(log_scale_max) * torch.tanh(raw_log_scale)
        return offset, log_gain, torch.exp(log_gain)

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        offset, log_gain, gain = parameters()
        energy = marked_energy_on_support(
            query,
            anchor_indices,
            fields,
            degrees_freedom=degrees_freedom,
            reliability=reliability,
            style_offset=offset,
            style_scale=gain,
        )
        objective = (normalized_prior * energy).sum(dim=1)
        objective = objective + float(offset_weight) * offset.square()
        objective = objective + float(log_scale_weight) * log_gain.square()
        objective.mean().backward()
        optimizer.step()
    with torch.no_grad():
        offset, _log_gain, gain = parameters()
        energy = marked_energy_on_support(
            query,
            anchor_indices,
            fields,
            degrees_freedom=degrees_freedom,
            reliability=reliability,
            style_offset=offset,
            style_scale=gain,
        )
    return energy.detach(), {
        "style_offset": offset.detach(),
        "style_scale": gain.detach(),
    }


def marked_kl_update(
    prior_joint: torch.Tensor,
    marked_energy: torch.Tensor,
    *,
    lambda_value: float,
) -> torch.Tensor:
    """Closed-form minimizer of expected energy plus lambda times KL."""

    if prior_joint.ndim != 3:
        raise ValueError("prior_joint must have shape [rows,members,K]")
    if marked_energy.shape != (prior_joint.shape[0], prior_joint.shape[2]):
        raise ValueError("marked_energy must have shape [rows,K]")
    temperature = float(lambda_value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("lambda_value must be positive and finite")
    log_weight = prior_joint.clamp_min(1e-30).log()
    log_weight = log_weight - marked_energy[:, None, :] / temperature
    flat = log_weight.reshape(len(log_weight), -1)
    return torch.softmax(flat, dim=1).reshape_as(prior_joint)


def common_student_t_randomness(
    rows: int,
    *,
    draws: int = 64,
    degrees_freedom: float = 4.0,
    seed: int = 9511,
) -> dict[str, torch.Tensor]:
    """Create two fixed independent draw streams for paired comparisons."""

    if int(rows) <= 0 or int(draws) <= 1:
        raise ValueError("rows must be positive and draws must exceed one")
    rng = np.random.default_rng(int(seed))
    shape_u = (int(rows), int(draws))
    shape_z = (int(rows), int(draws), 2)
    eps = np.finfo(np.float32).eps
    return {
        "uniform_1": torch.from_numpy(
            rng.uniform(eps, 1.0 - eps, size=shape_u).astype(np.float32)
        ),
        "noise_1": torch.from_numpy(
            rng.standard_t(float(degrees_freedom), size=shape_z).astype(np.float32)
        ),
        "uniform_2": torch.from_numpy(
            rng.uniform(eps, 1.0 - eps, size=shape_u).astype(np.float32)
        ),
        "noise_2": torch.from_numpy(
            rng.standard_t(float(degrees_freedom), size=shape_z).astype(np.float32)
        ),
        "draws": int(draws),
        "degrees_freedom": float(degrees_freedom),
        "seed": int(seed),
    }


def sample_continuous_mixture(
    probability: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    uniform: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Transform fixed uniforms/noises into draws from a Student-t mixture."""

    if probability.ndim != 2 or location.shape != (*probability.shape, 2):
        raise ValueError("probability/location must have shapes [B,S]/[B,S,2]")
    if scale.shape != location.shape:
        raise ValueError("scale and location must have equal shape")
    if uniform.ndim != 2 or noise.shape != (*uniform.shape, 2):
        raise ValueError("uniform/noise must have shapes [B,D]/[B,D,2]")
    if len(uniform) != len(probability):
        raise ValueError("randomness and predictive rows disagree")
    normalized = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-30)
    cumulative = torch.cumsum(normalized, dim=1)
    cumulative[:, -1] = 1.0
    component = torch.searchsorted(
        cumulative.contiguous(), uniform.to(cumulative).contiguous(), right=False
    ).clamp(max=probability.shape[1] - 1)
    gather = component.unsqueeze(2).expand(-1, -1, 2)
    selected_location = torch.gather(location, 1, gather)
    selected_scale = torch.gather(scale, 1, gather)
    return selected_location + selected_scale * noise.to(selected_scale)


def continuous_energy_score_mc(
    probability: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    randomness: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unbiased paired Monte-Carlo energy score for the full mixture."""

    first = sample_continuous_mixture(
        probability,
        location,
        scale,
        randomness["uniform_1"],
        randomness["noise_1"],
    )
    second = sample_continuous_mixture(
        probability,
        location,
        scale,
        randomness["uniform_2"],
        randomness["noise_2"],
    )
    data_term = 0.5 * (
        torch.linalg.vector_norm(first - target[:, None, :], dim=2).mean(dim=1)
        + torch.linalg.vector_norm(second - target[:, None, :], dim=2).mean(dim=1)
    )
    spread_term = 0.5 * torch.linalg.vector_norm(first - second, dim=2).mean(dim=1)
    return data_term - spread_term, first, second


def spatial_median_from_draws(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    maximum_iterations: int = 64,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Deterministic row-wise Weiszfeld approximation to the Bayes action."""

    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 2:
        raise ValueError("draw tensors must share shape [rows,draws,2]")
    points = torch.cat([first, second], dim=1)
    action = points.mean(dim=1)
    for _ in range(int(maximum_iterations)):
        distance = torch.linalg.vector_norm(points - action[:, None, :], dim=2)
        inverse = distance.clamp_min(1e-7).reciprocal()
        proposal = (inverse[:, :, None] * points).sum(dim=1)
        proposal = proposal / inverse.sum(dim=1, keepdim=True)
        if bool(
            (torch.linalg.vector_norm(proposal - action, dim=1) <= tolerance).all()
        ):
            action = proposal
            break
        action = proposal
    return action


def simultaneous_paired_upper_bounds(
    reference: np.ndarray,
    comparators: Sequence[np.ndarray],
    *,
    confidence: float = 0.95,
    bootstrap_repetitions: int = 5000,
    seed: int = 9512,
) -> dict[str, object]:
    """Max-centered paired-bootstrap upper bounds for several differences."""

    base = np.asarray(reference, dtype=np.float64)
    others = [np.asarray(value, dtype=np.float64) for value in comparators]
    if base.ndim != 1 or not others or any(value.shape != base.shape for value in others):
        raise ValueError("paired group-loss vectors must be nonempty and aligned")
    if len(base) < 2 or not np.isfinite(base).all() or any(
        not np.isfinite(value).all() for value in others
    ):
        raise ValueError("paired group-loss vectors must be finite with at least two groups")
    differences = np.stack([base - value for value in others], axis=1)
    mean = differences.mean(axis=0)
    rng = np.random.default_rng(int(seed))
    maxima = np.empty(int(bootstrap_repetitions), dtype=np.float64)
    for start in range(0, int(bootstrap_repetitions), 250):
        stop = min(start + 250, int(bootstrap_repetitions))
        index = rng.integers(0, len(base), size=(stop - start, len(base)))
        sampled = differences[index].mean(axis=1) - mean[None, :]
        maxima[start:stop] = sampled.max(axis=1)
    critical = float(np.quantile(maxima, float(confidence)))
    upper = mean + critical
    return {
        "mean_differences": mean.tolist(),
        "simultaneous_upper_bounds": upper.tolist(),
        "critical_value": critical,
        "confidence": float(confidence),
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "seed": int(seed),
        "coordinate_groups": int(len(base)),
    }


def select_information_saturated_E(
    profiled_group_losses: Mapping[int, np.ndarray],
    *,
    tolerance: float,
    minimum_later_values: int = 2,
    bootstrap_repetitions: int = 5000,
    seed: int = 9512,
) -> dict[str, object]:
    """Select the first preregistered member prefix with confirmed saturation."""

    ordered = sorted(map(int, profiled_group_losses))
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise ValueError("profiled member counts must be consecutive")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    audits = []
    selected: int | None = None
    for position, E in enumerate(ordered):
        later = ordered[position + 1 :]
        if len(later) < int(minimum_later_values):
            audits.append(
                {"E": E, "later_E": later, "eligible": False, "reason": "fewer_than_two_later_values"}
            )
            continue
        bounds = simultaneous_paired_upper_bounds(
            np.asarray(profiled_group_losses[E]),
            [np.asarray(profiled_group_losses[value]) for value in later],
            bootstrap_repetitions=bootstrap_repetitions,
            seed=int(seed) + E,
        )
        eligible = bool(
            max(bounds["simultaneous_upper_bounds"]) <= float(tolerance)
        )
        audits.append({"E": E, "later_E": later, "eligible": eligible, **bounds})
        if eligible and selected is None:
            selected = E
    return {
        "selected_E": selected,
        "saturated": selected is not None,
        "requires_extension": selected is None,
        "tested_E": ordered,
        "tolerance": float(tolerance),
        "minimum_later_values": int(minimum_later_values),
        "audits": audits,
    }
