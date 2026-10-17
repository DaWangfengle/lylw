"""Small, independently testable operators for MSEP joint-location ablations.

The anchor represents a reference component, not a ground-truth latent class.
No query truth is used in inference. The only labels here occur in training
losses and retrospective development metrics.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.msep_single_posterior import mixture_loss, posterior


def same_floor_neighbors(coords, buildings, floors, *, count=8, temperature_m=8.0):
    """Fixed-width neighbor targets with NO cross-floor positive mass.

    A singleton stratum uses its own anchor as the only feasible target.
    For strata with 2..count anchors, padding indices have exactly zero weight.
    """
    n = len(coords)
    if n < 2 or buildings.shape != (n,) or floors.shape != (n,):
        raise ValueError("invalid reference geometry")
    if count < 1 or temperature_m <= 0:
        raise ValueError("invalid neighbor hyperparameters")
    k = min(count, n - 1)
    d = torch.cdist(coords.float(), coords.float())
    same = ((buildings[:, None] == buildings[None, :])
            & (floors[:, None] == floors[None, :]))
    diagonal = torch.eye(n, dtype=torch.bool, device=coords.device)
    allowed = same & ~diagonal
    d = d.masked_fill(~allowed, float("inf"))
    values, indices = d.topk(k, largest=False, dim=1)
    valid = torch.isfinite(values)
    # Softmax over valid slots only; avoid all -inf -> NaN.
    logits = (-values / temperature_m).masked_fill(~valid, -1e9)
    weights = torch.softmax(logits, dim=1) * valid.to(logits.dtype)
    missing = ~valid.any(dim=1)
    if missing.any():
        indices[missing] = torch.nonzero(missing).flatten()[:, None]
        weights[missing] = 0.0
        weights[missing, 0] = 1.0
    else:
        weights = weights / weights.sum(1, keepdim=True)
    # Padding must never point to another floor, even with zero weight.
    indices = torch.where(valid | missing[:, None], indices,
                          torch.arange(n, device=coords.device)[:, None])
    if not torch.all(buildings[indices] == buildings[:, None]):
        raise AssertionError("cross-building neighbor")
    if not torch.all(floors[indices] == floors[:, None]):
        raise AssertionError("cross-floor neighbor")
    return indices.long(), weights.float()


def neighbor_contamination(indices, weights, buildings, floors):
    """Mass assigned to wrong building/floor by neighbor soft labels."""
    if indices.shape != weights.shape:
        raise ValueError("neighbor shapes differ")
    same = ((buildings[indices] == buildings[:, None])
            & (floors[indices] == floors[:, None]))
    return float((weights * (~same).to(weights.dtype)).sum().item()
                 / weights.sum().clamp_min(1e-8).item())


def class_structure(buildings, floors):
    pair = torch.stack([buildings.long(), floors.long()], dim=1)
    unique, anchor_class = torch.unique(pair, dim=0, return_inverse=True)
    return unique, anchor_class


def student_t_logpdf(centers, scales, y, df=4.0):
    if df <= 2:
        raise ValueError("df must exceed 2")
    sigma = scales.clamp_min(.05)
    z = (y[:, None, :] - centers) / sigma
    c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - .5 * math.log(df * math.pi)
    return (c - sigma.log() - .5 * (df + 1) * torch.log1p(z.square() / df)).sum(-1)


def mask_self_when_supported(scores, ids, target_anchor, anchor_class):
    """Exclude own reference only if its building-floor has another anchor."""
    counts = torch.bincount(anchor_class, minlength=int(anchor_class.max()) + 1)
    eligible = counts[anchor_class[target_anchor]] > 1
    self_mask = (ids == target_anchor[:, None]) & eligible[:, None]
    return scores.masked_fill(self_mask, -1e9), int((~eligible).sum().item())


def full_joint_loss(q, ids, centers, scales, target_xy,
                    target_class, anchor_class, *, point_weight=.30,
                    kl_reference=None, kl_weight=.02):
    """-log p(B,F,x,y|RSS), plus class-conditional Euclidean point risk.

    Neither a nearest-anchor class label nor a true-class-conditioned inference
    decision is used. The full normalizer includes *all* building/floor anchors.
    """
    if q.shape != ids.shape or q.shape != centers.shape[:2]:
        raise ValueError("incompatible candidate geometry")
    cls = anchor_class[ids]
    correct = cls == target_class[:, None]
    if not bool(correct.any(dim=1).all()):
        raise ValueError("true building-floor absent from candidate support")
    lp = student_t_logpdf(centers, scales, target_xy)
    terms = q.clamp_min(1e-30).log() + lp
    joint_nll = -torch.logsumexp(terms.masked_fill(~correct, -torch.inf), dim=1)
    within = q * correct.to(q.dtype)
    within = within / within.sum(1, keepdim=True).clamp_min(1e-30)
    point = torch.linalg.vector_norm((within[:, :, None] * centers).sum(1)
                                     - target_xy, dim=1)
    loss = joint_nll + point_weight * point
    if kl_reference is not None:
        kl = (q * (q.clamp_min(1e-30).log()
                   - kl_reference.clamp_min(1e-30).log())).sum(1)
        loss = loss + kl_weight * kl
    return loss.mean()


def refit_loss(kind, q, ids, centers, scales, xy, target_class,
               anchor_class, *, prior):
    if kind == "xy":
        return mixture_loss(q, centers, scales, xy, kl_reference=prior,
                            kl_weight=.02)
    if kind == "joint":
        return full_joint_loss(q, ids, centers, scales, xy, target_class,
                               anchor_class, kl_reference=prior)
    raise ValueError("unknown objective")


def calibrate_probability(pack, *, mode, steps=110):
    """CAL-fit only. NLL+.30 Euclidean MAE, sites equally weighted.

    Temperature is free in [.25,3], evidence coefficients in [0,1.5].
    No gate multiplication or frozen zero-coefficient absorbing state.
    """
    scores, ev = pack["scores"], pack["z"]
    use_t = mode in ("temperature", "joint")
    use_ev = mode in ("g13_fixed", "joint")
    if not (use_t or use_ev):
        raise ValueError("unsupported calibration")
    log_t = torch.nn.Parameter(torch.tensor(math.log(.9 if use_t else .6),
                                            device=scores.device))
    theta = torch.nn.Parameter(torch.tensor([.2, .2], device=scores.device))
    parameters = ([log_t] if use_t else []) + ([theta] if use_ev else [])
    opt = torch.optim.Adam(parameters, lr=.035)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        t = log_t.exp().clamp(.25, 3.) if use_t else scores.new_tensor(.6)
        logits = scores / t
        if use_ev:
            logits = logits + (ev * theta.clamp(0, 1.5)).sum(-1)
        q = torch.softmax(logits, dim=1)
        rows = mixture_loss(q, pack["xy"], pack["sigma"], pack["target"],
                            reduction="none", kl_weight=0)
        loss = (rows * pack["weight"]).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_t.clamp_(math.log(.25), math.log(3.))
            theta.clamp_(0, 1.5)
    return {"temperature": float(log_t.exp().item()) if use_t else .6,
            "theta": [float(x) for x in theta.detach()] if use_ev else [0., 0.]}


def probability(scores, z, params):
    t = float(params["temperature"])
    theta = torch.as_tensor(params["theta"], dtype=scores.dtype, device=scores.device)
    if t <= 0 or theta.shape != (2,):
        raise ValueError("invalid posterior parameters")
    return torch.softmax(scores / t + (z * theta).sum(-1), dim=-1)


def site_mean(values, keys):
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(tuple(key), []).append(i)
    return float(np.mean([np.asarray(values)[ix].mean() for ix in groups.values()]))


def spatial_metrics(q, ids, xy, y, buildings, floors, anchor_class,
                    class_pairs, *, keys):
    """Metrics for both legal decisions: full mean and predicted-class mean."""
    cls = anchor_class[ids]
    masses = torch.zeros((len(q), len(class_pairs)), dtype=q.dtype, device=q.device)
    masses.scatter_add_(1, cls, q)
    predicted_id = masses.argmax(1)
    predicted_bf = class_pairs[predicted_id]
    correct = ((predicted_bf[:, 0] == buildings)
               & (predicted_bf[:, 1] == floors)).float()
    full_point = (q[:, :, None] * xy).sum(1)
    chosen = (cls == predicted_id[:, None])
    conditioned = q * chosen.float()
    conditioned = conditioned / conditioned.sum(1, keepdim=True).clamp_min(1e-30)
    conditional_point = (conditioned[:, :, None] * xy).sum(1)
    err_full = torch.linalg.vector_norm(full_point - y, dim=1).detach().cpu().numpy()
    err_cond = torch.linalg.vector_norm(conditional_point - y, dim=1).detach().cpu().numpy()
    hits = correct.detach().cpu().numpy()
    return {
        "macro_mae_full_m": site_mean(err_full, keys),
        "macro_mae_predicted_bf_m": site_mean(err_cond, keys),
        "macro_bf_accuracy": site_mean(hits, keys),
        "row_bf_accuracy": float(hits.mean()),
        "site_p95_full_m": float(np.quantile(
            [np.mean(err_full[[i for i, x in enumerate(keys) if x == k]])
             for k in dict.fromkeys(keys)], .95)),
    }, {"err_full": err_full.tolist(), "err_predicted_bf": err_cond.tolist(),
        "bf_correct": hits.tolist()}
