"""MSEP-Loc single-member anchor posterior operators.

Every operator is defined on the same *candidate anchor identities* and
preserves the original V17 continuous Student-t component parameters.
No test targets are required by any predictor or reliability feature.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fhm_dapp_tim.fhm_dapp.uji_v17_g13_multiscan_observation import (
    build_fit_observation_memory,
    observation_features,
)

HISTORICAL_NEURAL_SCALE = 1.0
HISTORICAL_PROTOTYPE_WEIGHT = 32.0 / 3.0
HISTORICAL_ANCHOR_T = 0.6


def anchor_site_keys(coords: np.ndarray, building: np.ndarray, floor: np.ndarray) -> list[tuple]:
    """Physical site is (building, floor, rounded 4-decimal original XY)."""
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(building) or len(coords) != len(floor):
        raise ValueError("site inputs have inconsistent shapes")
    return [(int(b), int(f), round(float(x), 4), round(float(y), 4)) for (x, y), b, f in zip(coords, building, floor)]


def split_physical_sites(keys: list[tuple], *, seed: int = 17301,
                         fractions: tuple[float, float, float] = (.10, .10, .10)) -> dict[str, np.ndarray]:
    """FIT/DEV/CAL/TEST site-disjoint; floor/building-stratified with FIT reserve.

    Caller must train on FIT only and only fit calibration on CAL. Tiny strata
    can be FIT-only; each global role must have >=1 physical site.
    """
    if len(fractions) != 3 or not all(0 < x < 1 for x in fractions) or sum(fractions) >= 1:
        raise ValueError("invalid dev/cal/test fractions")
    groups: dict[tuple, list[tuple]] = {}
    for key in sorted(set(keys)):
        groups.setdefault(key[:2], []).append(key)
    rng = np.random.default_rng(seed)
    roles: dict[str, set[tuple]] = {name: set() for name in ("fit", "dev", "cal", "test")}
    for _, sites in sorted(groups.items()):
        sites = sorted(sites)
        order = rng.permutation(len(sites))
        n = len(sites)
        counts = [int(np.floor(n * f)) for f in fractions]
        # At least one FIT reference per building/floor and never invent role sites.
        for role, indices in zip(("dev", "cal", "test"), np.split(order[:sum(counts)], np.cumsum(counts)[:-1])):
            roles[role].update(sites[int(j)] for j in indices)
        roles["fit"].update(sites[int(j)] for j in order[sum(counts):])
    if not all(roles[name] for name in roles):
        raise ValueError("physical split has an empty global role; increase site count")
    if len(set.union(*roles.values())) != len(set(keys)):
        raise AssertionError("site split lost physical sites")
    if any(roles[x] & roles[y] for x in roles for y in roles if x != y):
        raise AssertionError("physical site leaked across roles")
    return {name: np.flatnonzero(np.fromiter((k in selected for k in keys), dtype=bool, count=len(keys)))
            for name, selected in roles.items()}


def candidate_evidence(model, rss: torch.Tensor, memory: Mapping[str, torch.Tensor],
                       k: int | None = None, *, excluded_anchor: torch.Tensor | None = None):
    """Compute the V17-R3 atomic prior then G13 evidence on same IDs.

    Optional exclusion is TRAIN-only leave-own-anchor-out; never used on
    DEV/CAL/TEST. When k=None all FIT anchors remain in support.
    """
    logits, embedding = model.retrieval_logits(
        rss, neural_scale=HISTORICAL_NEURAL_SCALE,
        prototype_distance_weight=HISTORICAL_PROTOTYPE_WEIGHT)
    if excluded_anchor is not None:
        if excluded_anchor.shape != (len(rss),):
            raise ValueError("excluded_anchor must have one id per query")
        logits = logits.scatter(1, excluded_anchor.long()[:, None], -1e9)
    n = logits.shape[1]
    width = n if k is None or k <= 0 else min(int(k), n)
    scores, ids = torch.topk(logits, k=width, dim=1, sorted=True)
    # A masked training anchor MUST not be selected, even if only one other exists.
    if excluded_anchor is not None and width >= n:
        raise ValueError("leave-self-out requires k < number of anchors")
    xy, sigma = model._components(embedding, ids)
    features = observation_features(rss, ids, model.encoder.seen_ap_mask, memory)
    stats = reliability_features(rss, ids, memory)
    return scores, ids, xy, sigma, features, stats


def reliability_features(rss: torch.Tensor, ids: torch.Tensor,
                         memory: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Only FIT-derived or current-query-derived statistics, all bounded [0,1]."""
    counts = memory["fit_rows_per_anchor"].to(rss.device)[ids].float()
    detection = memory["detection_probability"].to(rss.device)[ids]
    variance = memory["conditional_variance"].to(rss.device)[ids]
    seen = memory.get("seen_ap_mask", None)
    if seen is None:
        raise ValueError("memory must carry exact model-seen AP mask")
    seen = seen.to(rss.device).bool()
    query = rss[:, seen]
    visible = query > -0.999
    coverage = visible.float().mean(dim=1, keepdim=True).expand_as(counts)
    shrinkage = counts / (counts + 4.0)
    # Bernoulli uncertainty is greatest at p=1/2, zero near certain detection.
    visibility_confidence = (2.0 * detection - 1.0).square().mean(dim=2)
    strength_confidence = (1.0 / (1.0 + 20.0 * variance)).mean(dim=2)
    return torch.stack((shrinkage, coverage, visibility_confidence, strength_confidence), dim=2)


class UnifiedAnchorPosterior(nn.Module):
    """G13-compatible dynamic evidence reliability; zero gate exactly equals G13.

    Raw global coefficients constrained to [0,1.5] after every optimization step.
    Gate is low-rank shared across locations; no learned anchor-ID table.
    """
    def __init__(self, theta=(.8, .9)) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        self.gate = nn.Linear(4, 2, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def clamp_(self):
        with torch.no_grad():
            self.theta.clamp_(0, 1.5)
            self.gate.weight.clamp_(-3, 3)
            self.gate.bias.clamp_(-3, 3)

    def forward(self, scores: torch.Tensor, evidence: torch.Tensor,
                reliability: torch.Tensor, *, temperature: float = HISTORICAL_ANCHOR_T) -> torch.Tensor:
        if scores.shape != evidence.shape[:2] or evidence.shape[-1] != 2 or reliability.shape != (*scores.shape, 4):
            raise ValueError("posterior tensor dimensions are inconsistent")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        dynamic = 1.0 + 0.75 * torch.tanh(self.gate(reliability))
        eta = self.theta.clamp(0, 1.5)[None, None, :] * dynamic
        return torch.softmax(scores / temperature + (eta * evidence).sum(dim=-1), dim=-1)


def posterior(scores: torch.Tensor, evidence: torch.Tensor, *, mode: str,
              temperature: float = HISTORICAL_ANCHOR_T, theta=(0., 0.),
              reliability: torch.Tensor | None = None,
              unified: UnifiedAnchorPosterior | None = None) -> torch.Tensor:
    if mode == "raw":
        return torch.softmax(scores, dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mode == "temperature":
        return torch.softmax(scores / temperature, dim=-1)
    if mode == "g13":
        weights = torch.as_tensor(theta, dtype=scores.dtype, device=scores.device)
        if weights.shape != (2,) or bool((weights < 0).any()) or bool((weights > 1.5).any()):
            raise ValueError("G13 coefficients must be in [0,1.5]")
        return torch.softmax(scores / temperature + (evidence * weights).sum(-1), dim=-1)
    if mode == "unified" and unified is not None and reliability is not None:
        return unified(scores, evidence, reliability, temperature=temperature)
    raise ValueError(f"unknown or incompletely specified posterior mode: {mode}")


def mixture_loss(weights: torch.Tensor, centers: torch.Tensor, scales: torch.Tensor,
                 target: torch.Tensor, *, student_df: float = 4.0,
                 mae_weight: float = .30, kl_reference: torch.Tensor | None = None,
                 kl_weight: float = .05, reduction: str = "mean") -> torch.Tensor:
    """Exact independent-axis Student-t mixture NLL + Euclidean point risk.

    Spatial labeling supervises the continuous mixture, not a fictitious
    'correct anchor' class. For training, leave query's own FIT anchor out.
    """
    if student_df <= 2 or weights.shape != centers.shape[:2] or centers.shape != scales.shape or target.shape != (len(weights), 2):
        raise ValueError("invalid Student-t mixture dimensions")
    z = (target[:, None, :] - centers) / scales.clamp_min(.05)
    df = float(student_df)
    const = math.lgamma((df+1)/2) - math.lgamma(df/2) - .5*math.log(df*math.pi)
    lp = (const - scales.clamp_min(.05).log() - .5*(df+1)*torch.log1p(z.square()/df)).sum(-1)
    nll = -torch.logsumexp(weights.clamp_min(1e-30).log() + lp, dim=1)
    point = torch.linalg.vector_norm(torch.sum(weights[:, :, None]*centers, dim=1)-target, dim=1)
    result = nll + mae_weight*point
    if kl_reference is not None:
        result = result + kl_weight*(weights*(weights.clamp_min(1e-30).log()-kl_reference.clamp_min(1e-30).log())).sum(1)
    if reduction == "none":
        return result
    if reduction != "mean":
        raise ValueError("reduction must be mean or none")
    return result.mean()


def energy_score_mc(weights: torch.Tensor, centers: torch.Tensor, scales: torch.Tensor,
                    target: torch.Tensor, *, draws: int = 32, seed: int = 19602) -> torch.Tensor:
    """Full predictive Student-t energy score, common Monte-Carlo stream.

    Produces per-scan ES. Independent Student-t df4 per coordinate matches
    the legacy diagonal Student-t component convention.
    """
    if draws < 2:
        raise ValueError("draws must be >=2")
    rng = np.random.default_rng(seed)
    b, k = weights.shape
    # Using the same uniform/t random arrays across modes reduces comparison noise.
    uni = torch.as_tensor(rng.uniform(size=(2, b, draws)), dtype=weights.dtype, device=weights.device)
    t = torch.as_tensor(rng.standard_t(df=4.0, size=(2, b, draws, 2)), dtype=weights.dtype, device=weights.device)
    cdf = torch.cumsum(weights, dim=1).contiguous()
    sims = []
    for stream in range(2):
        chosen = torch.searchsorted(cdf, uni[stream].contiguous()).clamp(max=k-1)
        mu = torch.gather(centers, 1, chosen[:, :, None].expand(-1, -1, 2))
        sig = torch.gather(scales, 1, chosen[:, :, None].expand(-1, -1, 2))
        sims.append(mu + sig*t[stream])
    return (torch.linalg.vector_norm(sims[0]-target[:, None, :], dim=-1).mean(1)
            - .5*torch.linalg.vector_norm(sims[0]-sims[1], dim=-1).mean(1))


def site_metrics(err: np.ndarray, es: np.ndarray, keys: list[tuple]) -> dict:
    """Site-macro estimand: equal contribution per unique physical site."""
    if len(err) != len(es) or len(err) != len(keys) or not len(keys):
        raise ValueError("metric rows and site keys mismatch")
    grouped: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        grouped.setdefault(key, []).append(i)
    site_err = np.asarray([np.mean(err[idx]) for idx in grouped.values()])
    site_es = np.asarray([np.mean(es[idx]) for idx in grouped.values()])
    return {"site_macro_mae_m": float(site_err.mean()), "site_macro_es_m": float(site_es.mean()),
            "row_mae_m": float(err.mean()), "row_es_m": float(es.mean()),
            "site_p90_mae_m": float(np.quantile(site_err, .90)),
            "site_p95_mae_m": float(np.quantile(site_err, .95)),
            "sites": len(grouped), "rows": len(keys)}