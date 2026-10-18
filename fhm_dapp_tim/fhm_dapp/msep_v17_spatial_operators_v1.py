"""Auditable mathematical operators for V17 spatial-identifiability experiments.

The frozen phase must use canonical anchor order, preserve the original V17
continuous Student-t components, and treat virtual midpoint components as a
compact measure over N physical cells plus E unique undirected midpoint edges.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import student_t_logpdf


def build_doubly_stochastic_floor_graph(
    coords: np.ndarray | torch.Tensor,
    buildings: np.ndarray | torch.Tensor,
    floors: np.ndarray | torch.Tensor,
    *,
    count: int = 8,
    temperature_m: float = 8.0,
    sinkhorn_iterations: int = 20000,
    tolerance: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build symmetric doubly-stochastic W and unique same-BF midpoint edges."""
    xy = np.asarray(torch.as_tensor(coords).detach().cpu(), dtype=np.float64)
    b = np.asarray(torch.as_tensor(buildings).detach().cpu(), dtype=np.int64)
    f = np.asarray(torch.as_tensor(floors).detach().cpu(), dtype=np.int64)
    if xy.ndim != 2 or xy.shape[1] != 2 or b.shape != (len(xy),) or f.shape != (len(xy),):
        raise ValueError("coords/buildings/floors must have shapes [N,2], [N], [N]")
    if len(xy) < 1 or not np.isfinite(xy).all():
        raise ValueError("graph requires at least one finite coordinate")
    if count < 1 or temperature_m <= 0 or sinkhorn_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid graph hyperparameters")

    n = len(xy)
    adjacency = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(adjacency, 1.0)
    edge_set: set[tuple[int, int]] = set()
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(b.tolist(), f.tolist())):
        strata[key].append(i)

    for members in strata.values():
        if len(members) <= 1:
            continue
        block = xy[members]
        delta = block[:, None, :] - block[None, :, :]
        distance = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))
        for local_i, global_i in enumerate(members):
            order = np.argsort(distance[local_i] + np.eye(len(members))[local_i] * 1e30)
            neighbours = [int(j) for j in order if int(j) != local_i][:min(count, len(members) - 1)]
            for local_j in neighbours:
                global_j = members[local_j]
                weight = math.exp(-float(distance[local_i, local_j]) / temperature_m)
                adjacency[global_i, global_j] = max(adjacency[global_i, global_j], weight)
                adjacency[global_j, global_i] = max(adjacency[global_j, global_i], weight)
                edge_set.add((min(global_i, global_j), max(global_i, global_j)))

    scale = np.ones(n, dtype=np.float64)
    for _ in range(sinkhorn_iterations):
        row_mass = scale * (adjacency @ scale)
        if not np.isfinite(row_mass).all() or np.any(row_mass <= 0):
            raise RuntimeError("symmetric Sinkhorn scaling encountered invalid row mass")
        updated = scale / np.sqrt(row_mass)
        if float(np.max(np.abs(updated - scale))) <= tolerance:
            scale = updated
            break
        scale = updated
    weights = scale[:, None] * adjacency * scale[None, :]
    row_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    col_error = float(np.max(np.abs(weights.sum(axis=0) - 1.0)))
    if max(row_error, col_error) > 2e-6:
        raise RuntimeError(
            f"floor graph failed doubly-stochastic check: row={row_error:.3g}, col={col_error:.3g}"
        )
    if not np.allclose(weights, weights.T, atol=2e-10, rtol=2e-10):
        raise RuntimeError("floor graph lost symmetry")
    for i in range(n):
        nz = np.flatnonzero(weights[i] > 0)
        if np.any((b[nz] != b[i]) | (f[nz] != f[i])):
            raise RuntimeError("floor graph contains cross-building/floor edges")

    sparse_rows, sparse_cols = np.nonzero(weights > 0)
    indices = torch.as_tensor(np.stack([sparse_rows, sparse_cols]), dtype=torch.long)
    values = torch.as_tensor(weights[sparse_rows, sparse_cols], dtype=torch.float32)
    graph = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    edges = torch.as_tensor(sorted(edge_set), dtype=torch.long).reshape(-1, 2)
    return graph, edges


def graph_diffuse_logits(
    logits: torch.Tensor,
    graph_w: torch.Tensor,
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    """Apply P(alpha)=(1-alpha)I+alpha W to canonical anchor logits."""
    if logits.ndim != 2 or graph_w.ndim != 2 or graph_w.shape != (logits.shape[1], logits.shape[1]):
        raise ValueError("logit and graph shapes are incompatible")
    if graph_w.layout != torch.sparse_coo:
        raise ValueError("graph_w must be sparse COO")
    w = graph_w.to(device=logits.device, dtype=logits.dtype).coalesce()
    a = torch.as_tensor(alpha, dtype=logits.dtype, device=logits.device)
    if bool((a < 0).detach()) or bool((a > 1).detach()):
        raise ValueError("alpha must lie in [0,1]")
    neighbour_logits = torch.sparse.mm(w, logits.transpose(0, 1)).transpose(0, 1)
    return (1.0 - a) * logits + a * neighbour_logits


def layer_mass_preserving_softmax(
    raw_base_scores: torch.Tensor,
    within_layer_logits: torch.Tensor,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    """Preserve BF masses from RAW scores (T=1, theta=0), never calibrated logits.

    Only the within-BF conditional distribution comes from within_layer_logits.
    """
    if raw_base_scores.shape != within_layer_logits.shape or raw_base_scores.ndim != 2:
        raise ValueError("raw base scores and within-layer logits must have equal [batch, anchors] shapes")
    classes = torch.as_tensor(class_ids, dtype=torch.long, device=raw_base_scores.device)
    if classes.shape != (raw_base_scores.shape[1],) or bool((classes < 0).any()):
        raise ValueError("class_ids must be nonnegative with one entry per anchor")
    raw_q = torch.softmax(raw_base_scores, dim=-1)
    unique = torch.unique(classes, sorted=True)
    class_mass = torch.zeros(
        (len(raw_base_scores), int(unique.max().item()) + 1),
        dtype=raw_q.dtype, device=raw_q.device,
    )
    class_mass.scatter_add_(1, classes[None, :].expand(len(raw_q), -1), raw_q)
    output = torch.zeros_like(raw_q)
    for cls in unique.tolist():
        mask = classes == int(cls)
        output[:, mask] = class_mass[:, int(cls), None] * torch.softmax(
            within_layer_logits[:, mask], dim=-1
        )
    return output


def combine_retrieval_and_g13_logits(
    retrieval_scores: torch.Tensor,
    evidence: torch.Tensor,
    temperature: float | torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Complete anchor logit s/T + theta_visibility*z_visibility + theta_strength*z_strength."""
    if evidence.shape != (*retrieval_scores.shape, 2):
        raise ValueError("evidence must have shape [batch, anchors, 2]")
    t = torch.as_tensor(temperature, dtype=retrieval_scores.dtype, device=retrieval_scores.device)
    coeff = torch.as_tensor(theta, dtype=retrieval_scores.dtype, device=retrieval_scores.device)
    if t.numel() != 1 or bool((t <= 0).detach()) or coeff.shape != (2,):
        raise ValueError("temperature must be positive and theta must have shape [2]")
    return retrieval_scores / t + (evidence * coeff).sum(dim=-1)


def full_logit_graph_diffusion(
    scores: torch.Tensor,
    evidence: torch.Tensor,
    theta: torch.Tensor,
    temperature: float | torch.Tensor,
    graph_w: torch.Tensor,
    alpha_prior: float | torch.Tensor,
    alpha_obs: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complete undiffused logits and dual-channel diffused logits."""
    if evidence.ndim != 3 or evidence.shape[:2] != scores.shape or evidence.shape[2] != 2:
        raise ValueError("evidence must have shape [batch, anchors, 2]")
    theta_t = torch.as_tensor(theta, dtype=scores.dtype, device=scores.device)
    if theta_t.shape != (2,):
        raise ValueError("theta must contain visibility and strength coefficients")
    t = torch.as_tensor(temperature, dtype=scores.dtype, device=scores.device)
    if t.numel() != 1 or bool((t <= 0).detach()):
        raise ValueError("temperature must be positive")
    prior = scores / t
    obs = (evidence * theta_t).sum(dim=-1)
    base = prior + obs
    diffused = graph_diffuse_logits(prior, graph_w, alpha_prior) + graph_diffuse_logits(
        obs, graph_w, alpha_obs
    )
    return base, diffused


def log_partition_gap(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """log Z(after)-log Z(before), nonpositive for full-logit doubly-stochastic diffusion."""
    if before.shape != after.shape:
        raise ValueError("before and after logits must have equal shapes")
    return torch.logsumexp(after, dim=-1) - torch.logsumexp(before, dim=-1)


def interpolate_g13_midpoint_memory(
    detection_probability: torch.Tensor,
    conditional_mean: torch.Tensor,
    conditional_variance: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    gamma_space: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Interpolate FIT-only G13 statistics at unique undirected edge midpoints.

    At omega_i=omega_j=0.5:
      pi_v = (pi_i+pi_j)/2
      mu_v = (mu_i+mu_j)/2
      var_v = (var_i+var_j)/2 + gamma_space*(1-sum omega^2)
            = (var_i+var_j)/2 + gamma_space/2.
    Variance is not upper-clipped after adding gamma; gamma is in normalized RSS^2 units.
    """
    if not math.isfinite(float(gamma_space)) or gamma_space < 0:
        raise ValueError("gamma_space must be finite and nonnegative")
    if detection_probability.ndim != 2:
        raise ValueError("G13 detection_probability must have shape [N, AP]")
    if conditional_mean.shape != detection_probability.shape or conditional_variance.shape != detection_probability.shape:
        raise ValueError("G13 pi, mean, and variance arrays must have identical shapes")
    if not torch.isfinite(detection_probability).all() or bool(
        ((detection_probability <= 0) | (detection_probability >= 1)).any()
    ):
        raise ValueError("G13 detection probabilities must be finite and strictly between 0 and 1")
    if not torch.isfinite(conditional_mean).all() or not torch.isfinite(conditional_variance).all():
        raise ValueError("G13 conditional statistics must be finite")
    if bool((conditional_variance <= 0).any()):
        raise ValueError("G13 conditional variances must be positive")

    edges = torch.as_tensor(edge_index, dtype=torch.long, device=detection_probability.device).reshape(-1, 2)
    n = detection_probability.shape[0]
    if edges.numel() and bool(((edges < 0) | (edges >= n)).any()):
        raise ValueError("midpoint edge index outside G13 anchor memory")
    i, j = edges[:, 0], edges[:, 1]
    pi_v = 0.5 * (detection_probability[i] + detection_probability[j])
    mu_v = 0.5 * (conditional_mean[i] + conditional_mean[j])
    var_v = 0.5 * (conditional_variance[i] + conditional_variance[j]) + 0.5 * float(gamma_space)
    return {
        "detection_probability": pi_v,
        "conditional_mean": mu_v,
        "conditional_variance": var_v,
    }


def g13_continuous_field_features(
    query_rss: torch.Tensor,
    seen_ap_mask: torch.Tensor,
    detection_probability: torch.Tensor,
    conditional_mean: torch.Tensor,
    conditional_variance: torch.Tensor,
    *,
    n_physical: int,
) -> torch.Tensor:
    """G13 evidence for physical candidates followed by virtual candidates.

    The stabilization mean/RMS is computed ONLY from the first n_physical
    candidates, then applied unchanged to virtual candidates. Therefore appending
    virtual points cannot alter any parent G13 feature: beta=0 remains an exact
    posterior identity. Memory tensors have shape [N+E, seen_APs].
    """
    if query_rss.ndim != 2 or seen_ap_mask.ndim != 1:
        raise ValueError("query_rss and seen_ap_mask must have shapes [B, AP] and [AP]")
    seen = seen_ap_mask.to(query_rss.device).bool()
    if seen.shape[0] != query_rss.shape[1] or not bool(seen.any()):
        raise ValueError("seen AP mask does not match query RSS schema")
    x = query_rss[:, seen].float()
    pi = detection_probability.to(device=x.device, dtype=x.dtype)
    mu = conditional_mean.to(device=x.device, dtype=x.dtype)
    var = conditional_variance.to(device=x.device, dtype=x.dtype)
    if pi.ndim != 2 or pi.shape != mu.shape or pi.shape != var.shape or pi.shape[1] != x.shape[1]:
        raise ValueError("G13 memory must have shape [N+E, seen_APs]")
    if not 1 <= int(n_physical) <= pi.shape[0]:
        raise ValueError("n_physical must identify a nonempty physical prefix")
    if not torch.isfinite(pi).all() or bool(((pi <= 0) | (pi >= 1)).any()):
        raise ValueError("G13 pi must be finite and in (0,1)")
    if not torch.isfinite(mu).all() or not torch.isfinite(var).all() or bool((var <= 0).any()):
        raise ValueError("G13 means/variances must be finite and variances positive")

    detected = x > -0.999
    vis = torch.where(detected[:, None, :], pi.log()[None, :, :],
                      torch.log1p(-pi)[None, :, :]).mean(dim=2)
    diff = (x[:, None, :] - mu[None, :, :]).square() / (4.0 * var[None, :, :])
    log_strength = -2.5 * torch.log1p(diff) - 0.5 * var.log()[None, :, :]
    strength = (detected[:, None, :].to(x.dtype) * log_strength).sum(dim=2)
    strength = strength / detected.sum(dim=1).clamp_min(1).to(x.dtype)[:, None]
    strength = torch.where(detected.any(dim=1)[:, None], strength, torch.zeros_like(strength))

    def stabilize_against_physical(value: torch.Tensor) -> torch.Tensor:
        reference = value[:, :n_physical]
        mean = reference.mean(dim=1, keepdim=True)
        scale = torch.sqrt((reference - mean).square().mean(dim=1, keepdim=True) + 0.20**2)
        return (value - mean) / scale

    features = torch.stack((
        stabilize_against_physical(vis),
        stabilize_against_physical(strength),
    ), dim=2)
    if not torch.isfinite(features).all():
        raise RuntimeError("nonfinite G13 continuous-field evidence")
    return features


def interpolate_midpoint_centers(
    anchor_coords: torch.Tensor,
    parent_centers: torch.Tensor,
    edge_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return geometric midpoint and residual-interpolated Student-t centers.

    The primary center is y_v + 0.5*((mu_i-y_i)+(mu_j-y_j)); the pure geometric
    midpoint is diagnostic only. With canonical parent residuals, the primary
    expression algebraically equals 0.5*(mu_i+mu_j).
    """
    if anchor_coords.ndim != 2 or anchor_coords.shape[1] != 2:
        raise ValueError("anchor_coords must have shape [N,2]")
    if parent_centers.ndim != 3 or parent_centers.shape[1:] != anchor_coords.shape:
        raise ValueError("parent_centers must have shape [B,N,2]")
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=parent_centers.device).reshape(-1, 2)
    coords = anchor_coords.to(device=parent_centers.device, dtype=parent_centers.dtype)
    if edges.numel() and bool(((edges < 0) | (edges >= len(coords))).any()):
        raise ValueError("midpoint edge index outside anchor coordinates")
    i, j = edges[:, 0], edges[:, 1]
    geometric = 0.5 * (coords[i] + coords[j])
    residual = parent_centers - coords[None, :, :]
    residual_center = geometric[None, :, :] + 0.5 * (residual[:, i, :] + residual[:, j, :])
    return {
        "geometric_centers": geometric[None, :, :].expand(len(parent_centers), -1, -1),
        "residual_interpolated_centers": residual_center,
    }


def interpolate_midpoint_scales(parent_scales: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate continuous Student-t component scales at edge midpoints."""
    if parent_scales.ndim != 3 or parent_scales.shape[-1] != 2:
        raise ValueError("parent_scales must have shape [B,N,2]")
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=parent_scales.device).reshape(-1, 2)
    if edges.numel() and bool(((edges < 0) | (edges >= parent_scales.shape[1])).any()):
        raise ValueError("midpoint edge index outside parent scales")
    i, j = edges[:, 0], edges[:, 1]
    return 0.5 * (parent_scales[:, i, :] + parent_scales[:, j, :])


def cell_lse_expansion(
    parent_logits: torch.Tensor,
    virtual_logits: torch.Tensor,
    edge_index: torch.Tensor,
    beta: float,
    parent_centers: torch.Tensor,
    parent_scales: torch.Tensor,
    virtual_centers: torch.Tensor,
    virtual_scales: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    preserve_class_mass: bool = False,
    raw_base_scores: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    """Unit-measure Cell LSE and compact [B,N+E] physical/virtual mixture.

    Every cell i allocates (1-beta) to its physical endpoint and beta/degree_i
    to each incident midpoint. A shared edge midpoint is owned by both endpoint
    cells; its final component weight is the sum of both cell allocations.
    No padded component slots are returned. At beta=0 the function bypasses
    LSE and returns exact parent logits, parent weights, centers, and scales.
    """
    if not math.isfinite(float(beta)) or not 0 <= float(beta) < 1:
        raise ValueError("beta must be finite and lie in [0,1)")
    if parent_logits.ndim != 2:
        raise ValueError("parent_logits must have shape [B,N]")
    batch, n = parent_logits.shape
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=parent_logits.device).reshape(-1, 2)
    e = len(edges)
    if virtual_logits.shape != (batch, e):
        raise ValueError("virtual_logits shape does not match edge count")
    if not torch.isfinite(parent_logits).all() or not torch.isfinite(virtual_logits).all():
        raise ValueError("parent and virtual complete logits must be finite")
    if parent_centers.shape != (batch, n, 2) or parent_scales.shape != (batch, n, 2):
        raise ValueError("parent centers/scales must have shape [B,N,2]")
    if virtual_centers.shape != (batch, e, 2) or virtual_scales.shape != (batch, e, 2):
        raise ValueError("virtual centers/scales must have shape [B,E,2]")
    if not torch.isfinite(parent_centers).all() or not torch.isfinite(virtual_centers).all():
        raise ValueError("component centers must be finite")
    if not torch.isfinite(parent_scales).all() or not torch.isfinite(virtual_scales).all():
        raise ValueError("component scales must be finite")
    if bool((parent_scales <= 0).any()) or bool((virtual_scales <= 0).any()):
        raise ValueError("component scales must be strictly positive")
    classes = torch.as_tensor(class_ids, dtype=torch.long, device=parent_logits.device)
    if classes.shape != (n,) or bool((classes < 0).any()):
        raise ValueError("class_ids must have one nonnegative entry per parent cell")
    if preserve_class_mass:
        if raw_base_scores is None or raw_base_scores.shape != parent_logits.shape:
            raise ValueError("BF-mass-preserving mode requires raw unscaled base scores [B,N]")
        if not torch.isfinite(raw_base_scores).all():
            raise ValueError("raw unscaled base scores must be finite")

    # Canonicalize each undirected edge without reordering edge IDs, because the
    # virtual-logit/center/scale tensors use exactly this row order.
    edges = torch.sort(edges, dim=1).values
    edge_list = [tuple(map(int, pair)) for pair in edges.detach().cpu().tolist()]
    if any(i == j or i < 0 or j >= n for i, j in edge_list):
        raise ValueError("midpoint edges must connect two distinct valid anchors")
    if len(set(edge_list)) != len(edge_list):
        raise ValueError("edge_index must contain each undirected midpoint edge exactly once")
    class_values = classes.detach().cpu().tolist()
    incident: list[list[int]] = [[] for _ in range(n)]
    for edge_id, (i, j) in enumerate(edge_list):
        if class_values[i] != class_values[j]:
            raise ValueError("midpoint edge crosses building-floor class")
        incident[i].append(edge_id)
        incident[j].append(edge_id)

    if beta == 0.0:
        cell_logits = parent_logits
        cell_internal_weights = tuple(
            parent_logits.new_ones((batch, 1)) for _ in range(n)
        )
    else:
        logits_by_cell = []
        weights_by_cell = []
        for i, edge_ids in enumerate(incident):
            degree = len(edge_ids)
            if degree == 0:
                logits_by_cell.append(parent_logits[:, i])
                weights_by_cell.append(parent_logits.new_ones((batch, 1)))
                continue
            endpoint = parent_logits[:, i] + math.log1p(-float(beta))
            virtual = virtual_logits[:, edge_ids] + math.log(float(beta) / degree)
            terms = torch.cat((endpoint[:, None], virtual), dim=1)
            logits_by_cell.append(torch.logsumexp(terms, dim=1))
            weights_by_cell.append(torch.softmax(terms, dim=1))
        cell_logits = torch.stack(logits_by_cell, dim=1)
        cell_internal_weights = tuple(weights_by_cell)

    if preserve_class_mass:
        cell_q = layer_mass_preserving_softmax(raw_base_scores, cell_logits, classes)
    else:
        cell_q = torch.softmax(cell_logits, dim=-1)

    parent_component_weights = torch.stack(
        [cell_q[:, i] * cell_internal_weights[i][:, 0] for i in range(n)], dim=1
    )
    cell_center_rows = []
    if beta == 0.0:
        # Exact identity path: no virtual slots exist in per-cell weights.
        cell_center_rows = [parent_centers[:, i, :] for i in range(n)]
        virtual_component_weights = parent_logits.new_zeros((batch, e))
    else:
        for i in range(n):
            weights_i = cell_internal_weights[i]
            center_i = weights_i[:, :1] * parent_centers[:, i, :]
            for slot, edge_id in enumerate(incident[i], start=1):
                center_i = center_i + weights_i[:, slot:slot + 1] * virtual_centers[:, edge_id, :]
            cell_center_rows.append(center_i)
        edge_component_weights = []
        for edge_id, (i, j) in enumerate(edge_list):
            slot_i = incident[i].index(edge_id) + 1
            slot_j = incident[j].index(edge_id) + 1
            weight_i = cell_q[:, i] * cell_internal_weights[i][:, slot_i]
            weight_j = cell_q[:, j] * cell_internal_weights[j][:, slot_j]
            edge_component_weights.append(weight_i + weight_j)
        if e:
            virtual_component_weights = torch.stack(edge_component_weights, dim=1)
        else:
            virtual_component_weights = parent_logits.new_empty((batch, 0))
    expanded_weights = torch.cat((parent_component_weights, virtual_component_weights), dim=1)
    expanded_centers = torch.cat((parent_centers, virtual_centers), dim=1)
    expanded_scales = torch.cat((parent_scales, virtual_scales), dim=1)
    # A virtual midpoint's edge[0] is a valid representative ID for BF masking:
    # edge endpoints are asserted to have the same building-floor class.
    expanded_ids_1d = torch.cat((
        torch.arange(n, dtype=torch.long, device=parent_logits.device),
        edges[:, 0],
    ))
    expanded_ids = expanded_ids_1d[None, :].expand(batch, -1)
    expanded_is_virtual = torch.cat((
        torch.zeros(n, dtype=torch.bool, device=parent_logits.device),
        torch.ones(e, dtype=torch.bool, device=parent_logits.device),
    ))
    expanded_edge_ids = torch.cat((
        torch.full((n,), -1, dtype=torch.long, device=parent_logits.device),
        torch.arange(e, dtype=torch.long, device=parent_logits.device),
    ))
    return {
        "cell_logits": cell_logits,
        "cell_probabilities": cell_q,
        "cell_internal_weights": cell_internal_weights,
        "cell_centers": torch.stack(cell_center_rows, dim=1),
        "expanded_weights": expanded_weights,
        "expanded_centers": expanded_centers,
        "expanded_scales": expanded_scales,
        "expanded_ids": expanded_ids,
        "expanded_is_virtual": expanded_is_virtual,
        "expanded_edge_ids": expanded_edge_ids,
        "cell_classes": classes,
    }


def joint_bf_xy_nll(
    q: torch.Tensor,
    candidate_ids: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    target_xy: torch.Tensor,
    target_building: torch.Tensor,
    target_floor: torch.Tensor,
    anchor_building: torch.Tensor,
    anchor_floor: torch.Tensor,
) -> torch.Tensor:
    """Per-row -log p(B,F,x,y|RSS), with strict zero-weight and true-BF masking."""
    if q.shape != candidate_ids.shape or q.shape != centers.shape[:2] or centers.shape != scales.shape:
        raise ValueError("incompatible joint NLL tensors")
    if target_xy.shape != (len(q), 2):
        raise ValueError("target_xy must have shape [B,2]")
    if not torch.isfinite(q).all() or bool((q < 0).any()):
        raise ValueError("mixture weights must be finite and nonnegative")
    if not torch.isfinite(centers).all() or not torch.isfinite(scales).all() or bool((scales <= 0).any()):
        raise ValueError("mixture centers/scales must be finite and scales strictly positive")
    ids = candidate_ids.long()
    cls_b = anchor_building.to(ids.device)[ids]
    cls_f = anchor_floor.to(ids.device)[ids]
    true_b = torch.as_tensor(target_building, device=ids.device).long()[:, None]
    true_f = torch.as_tensor(target_floor, device=ids.device).long()[:, None]
    mask = (cls_b == true_b) & (cls_f == true_f)
    valid_mask = mask & (q > 0)
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError("positive-weight true building-floor component is absent from candidate support")
    lp = student_t_logpdf(centers, scales, target_xy)
    log_q = q.clamp_min(torch.finfo(q.dtype).tiny).log()
    terms = (log_q + lp).masked_fill(~valid_mask, -torch.inf)
    return -torch.logsumexp(terms, dim=-1)


def replace_unmasked_nll_with_joint_bf_xy(
    base_loss: torch.Tensor,
    original_unmasked_nll: torch.Tensor,
    joint_bf_xy_nll_loss: torch.Tensor,
) -> torch.Tensor:
    """Replace the old NLL term at weight 1.0; never add a second NLL."""
    if any(value.numel() != 1 for value in (base_loss, original_unmasked_nll, joint_bf_xy_nll_loss)):
        raise ValueError("loss composition inputs must be scalar tensors")
    return base_loss - original_unmasked_nll + joint_bf_xy_nll_loss


def add_transport_risk_increment(
    loss: torch.Tensor,
    transport_risk_per_row: torch.Tensor,
) -> torch.Tensor:
    """Add the locked +0.30 mean component-center transport risk, retaining coord loss."""
    if loss.numel() != 1 or transport_risk_per_row.ndim != 1:
        raise ValueError("loss must be scalar and transport risk must have shape [batch]")
    if not torch.isfinite(transport_risk_per_row).all() or bool((transport_risk_per_row < 0).any()):
        raise ValueError("transport risks must be finite and nonnegative")
    return loss + 0.30 * transport_risk_per_row.mean()


def component_center_transport_risk(
    q: torch.Tensor,
    centers: torch.Tensor,
    target_xy: torch.Tensor,
) -> torch.Tensor:
    """W1 center risk: sum_i q_i ||mu_i-y||_2 for a Dirac target."""
    if q.shape != centers.shape[:2] or centers.shape != (len(q), q.shape[1], 2):
        raise ValueError("incompatible transport-risk tensors")
    if target_xy.shape != (len(q), 2):
        raise ValueError("target_xy must have shape [B,2]")
    if not torch.isfinite(q).all() or bool((q < 0).any()):
        raise ValueError("mixture weights must be finite and nonnegative")
    if not torch.isfinite(centers).all():
        raise ValueError("component centers must be finite")
    distance = torch.linalg.vector_norm(centers - target_xy[:, None, :], dim=-1)
    return (q * distance).sum(dim=-1)
