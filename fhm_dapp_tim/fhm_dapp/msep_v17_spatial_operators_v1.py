"""Mathematically explicit operators for the V17 spatial-identifiability study.

This module is intentionally independent of dataset loading and training.  It
contains the invariants that must be tested before the experimental runner is
allowed to score any result.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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
    """Build symmetric, doubly-stochastic W and undirected same-BF midpoint edges.

    W is sparse COO. Its nonzero support never crosses a building-floor stratum.
    A symmetric kNN graph is given positive self-loops and symmetrically Sinkhorn
    scaled as W = diag(x) A diag(x), preserving symmetry and hence both row and
    column stochasticity. The separate edge list excludes self-loops.
    """
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

    # Symmetric diagonal scaling solves x_i (A x)_i = 1.
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
    """Apply P(alpha)=(1-alpha)I+alpha W to the anchor-logit axis."""
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
    base_logits: torch.Tensor,
    within_layer_logits: torch.Tensor,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    """Diffuse within BF classes while preserving each class's original total mass."""
    if base_logits.shape != within_layer_logits.shape or base_logits.ndim != 2:
        raise ValueError("base and diffused logits must have equal [batch, anchors] shapes")
    classes = torch.as_tensor(class_ids, dtype=torch.long, device=base_logits.device)
    if classes.shape != (base_logits.shape[1],):
        raise ValueError("class_ids must have one entry per anchor")
    base_q = torch.softmax(base_logits, dim=-1)
    unique = torch.unique(classes, sorted=True)
    class_mass = torch.zeros(
        (len(base_logits), int(unique.max().item()) + 1),
        dtype=base_q.dtype, device=base_q.device,
    )
    class_mass.scatter_add_(1, classes[None, :].expand(len(base_q), -1), base_q)
    output = torch.zeros_like(base_q)
    for cls in unique.tolist():
        mask = classes == int(cls)
        output[:, mask] = class_mass[:, int(cls), None] * torch.softmax(
            within_layer_logits[:, mask], dim=-1
        )
    return output


def full_logit_graph_diffusion(
    scores: torch.Tensor,
    evidence: torch.Tensor,
    theta: torch.Tensor,
    temperature: float | torch.Tensor,
    graph_w: torch.Tensor,
    alpha_prior: float | torch.Tensor,
    alpha_obs: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (undiffused total logits, dual-channel diffused total logits)."""
    if evidence.ndim != 3 or evidence.shape[:2] != scores.shape or evidence.shape[2] != 2:
        raise ValueError("evidence must have shape [batch, anchors, 2]")
    theta_t = torch.as_tensor(theta, dtype=scores.dtype, device=scores.device)
    if theta_t.shape != (2,):
        raise ValueError("theta must contain visibility and strength coefficients")
    t = torch.as_tensor(temperature, dtype=scores.dtype, device=scores.device)
    if bool((t <= 0).detach()):
        raise ValueError("temperature must be positive")
    prior = scores / t
    obs = (evidence * theta_t).sum(dim=-1)
    base = prior + obs
    diffused = graph_diffuse_logits(prior, graph_w, alpha_prior) + graph_diffuse_logits(
        obs, graph_w, alpha_obs
    )
    return base, diffused


def log_partition_gap(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """log Z(after)-log Z(before); nonpositive under doubly-stochastic full-logit diffusion."""
    if before.shape != after.shape:
        raise ValueError("before and after logits must have equal shapes")
    return torch.logsumexp(after, dim=-1) - torch.logsumexp(before, dim=-1)


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
    base_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Measure-preserving cell LSE with symmetric midpoint ownership.

    Every parent cell has unit reference measure: endpoint weight (1-beta) and
    total virtual weight beta, evenly split over incident midpoint edges. Each
    midpoint is included in both endpoint cells by design. The output expanded
    mixture retains endpoint/virtual components rather than collapsing away
    within-cell uncertainty.
    """
    if not 0 <= float(beta) < 1:
        raise ValueError("beta must lie in [0,1)")
    if parent_logits.ndim != 2:
        raise ValueError("parent_logits must have shape [batch, anchors]")
    batch, n = parent_logits.shape
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=parent_logits.device).reshape(-1, 2)
    e = len(edges)
    if virtual_logits.shape != (batch, e):
        raise ValueError("virtual_logits shape does not match edge count")
    if parent_centers.shape != (batch, n, 2) or parent_scales.shape != (batch, n, 2):
        raise ValueError("parent centers/scales must have shape [batch, anchors, 2]")
    if virtual_centers.shape != (batch, e, 2) or virtual_scales.shape != (batch, e, 2):
        raise ValueError("virtual centers/scales must have shape [batch, edges, 2]")
    cls = torch.as_tensor(class_ids, dtype=torch.long, device=parent_logits.device)
    if cls.shape != (n,):
        raise ValueError("class_ids must have one entry per parent cell")

    incident: list[list[int]] = [[] for _ in range(n)]
    class_values = cls.detach().cpu().tolist()
    for edge_id, (i, j) in enumerate(edges.detach().cpu().tolist()):
        if i == j or i < 0 or j < 0 or i >= n or j >= n:
            raise ValueError("invalid midpoint edge")
        if int(class_values[i]) != int(class_values[j]):
            raise ValueError("midpoint edge crosses building-floor class")
        incident[i].append(edge_id)
        incident[j].append(edge_id)
    max_degree = max((len(x) for x in incident), default=0)
    incident_idx = torch.full((n, max_degree), -1, dtype=torch.long, device=parent_logits.device)
    for i, row in enumerate(incident):
        if row:
            incident_idx[i, :len(row)] = torch.as_tensor(row, dtype=torch.long, device=parent_logits.device)
    degree = torch.as_tensor([len(x) for x in incident], dtype=parent_logits.dtype, device=parent_logits.device)
    if max_degree:
        gathered_score = virtual_logits[:, incident_idx.clamp_min(0)]
        valid = incident_idx >= 0
        log_beta = parent_logits.new_tensor(-torch.inf) if beta == 0 else parent_logits.new_tensor(math.log(beta))
        log_share = torch.where(
            degree > 0,
            log_beta - degree.clamp_min(1).log(),
            parent_logits.new_tensor(-torch.inf),
        )
        virtual_terms = gathered_score + log_share[None, :, None]
        virtual_terms = virtual_terms.masked_fill(~valid[None, :, :], -torch.inf)
    else:
        valid = torch.zeros((n, 0), dtype=torch.bool, device=parent_logits.device)
        gathered_score = parent_logits.new_empty((batch, n, 0))
        virtual_terms = gathered_score
    if beta == 0:
        endpoint_log_weight = parent_logits.new_zeros((n,))
    else:
        endpoint_log_weight = torch.where(
            degree > 0,
            parent_logits.new_full((n,), math.log1p(-beta)),
            parent_logits.new_zeros((n,)),
        )
    endpoint_terms = parent_logits + endpoint_log_weight[None, :]
    terms = torch.cat([endpoint_terms[:, :, None], virtual_terms], dim=-1)
    cell_logits = torch.logsumexp(terms, dim=-1)
    internal_weights = torch.softmax(terms, dim=-1)
    if preserve_class_mass:
        reference = base_logits if base_logits is not None else parent_logits
        if reference.shape != parent_logits.shape:
            raise ValueError("base_logits must match parent_logits")
        cell_q = layer_mass_preserving_softmax(reference, cell_logits, cls)
    else:
        cell_q = torch.softmax(cell_logits, dim=-1)

    parent_mu = parent_centers[:, :, None, :]
    parent_sigma = parent_scales[:, :, None, :]
    if max_degree:
        gathered_mu = virtual_centers[:, incident_idx.clamp_min(0)]
        gathered_sigma = virtual_scales[:, incident_idx.clamp_min(0)]
        gathered_mu = gathered_mu.masked_fill(~valid[None, :, :, None], 0.0)
        gathered_sigma = gathered_sigma.masked_fill(~valid[None, :, :, None], 0.0)
        all_mu = torch.cat([parent_mu, gathered_mu], dim=2)
        all_sigma = torch.cat([parent_sigma, gathered_sigma], dim=2)
    else:
        all_mu, all_sigma = parent_mu, parent_sigma
    expanded_weights = (cell_q[:, :, None] * internal_weights).reshape(batch, -1)
    expanded_centers = all_mu.expand(-1, -1, -1, -1).reshape(batch, -1, 2)
    expanded_scales = all_sigma.expand(-1, -1, -1, -1).reshape(batch, -1, 2)
    expanded_ids = torch.arange(n, device=parent_logits.device)[None, :, None].expand(
        batch, -1, all_mu.shape[2]
    ).reshape(batch, -1)
    return {
        "cell_logits": cell_logits,
        "cell_probabilities": cell_q,
        "internal_weights": internal_weights,
        "cell_centers": (internal_weights[:, :, :, None] * all_mu).sum(dim=2),
        "expanded_weights": expanded_weights,
        "expanded_centers": expanded_centers,
        "expanded_scales": expanded_scales,
        "expanded_ids": expanded_ids,
        "cell_classes": cls,
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
    """Per-row -log p(B,F,x,y|RSS), retaining only the true BF in the numerator."""
    if q.shape != candidate_ids.shape or q.shape != centers.shape[:2] or centers.shape != scales.shape:
        raise ValueError("incompatible joint NLL tensors")
    if target_xy.shape != (len(q), 2):
        raise ValueError("target_xy must have shape [batch,2]")
    ids = candidate_ids.long()
    cls_b = anchor_building.to(ids.device)[ids]
    cls_f = anchor_floor.to(ids.device)[ids]
    true_b = torch.as_tensor(target_building, device=ids.device).long()[:, None]
    true_f = torch.as_tensor(target_floor, device=ids.device).long()[:, None]
    mask = (cls_b == true_b) & (cls_f == true_f)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("true building-floor is absent from candidate support")
    lp = student_t_logpdf(centers, scales, target_xy)
    return -torch.logsumexp(
        (q.clamp_min(1e-30).log() + lp).masked_fill(~mask, -torch.inf), dim=-1
    )


def component_center_transport_risk(
    q: torch.Tensor,
    centers: torch.Tensor,
    target_xy: torch.Tensor,
) -> torch.Tensor:
    """Closed-form W1 center risk: sum_i q_i ||mu_i-y||_2 for a Dirac target."""
    if q.shape != centers.shape[:2] or centers.shape != (len(q), q.shape[1], 2):
        raise ValueError("incompatible transport-risk tensors")
    if target_xy.shape != (len(q), 2):
        raise ValueError("target_xy must have shape [batch,2]")
    distance = torch.linalg.vector_norm(centers - target_xy[:, None, :], dim=-1)
    return (q * distance).sum(dim=-1)


