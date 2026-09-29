"""Predictive-distribution and conformal-risk utilities for AC-RHAP.

The point localizer produces one projected anchor posterior shared by all
members.  We retain every member's coordinate Student-t component under that
posterior instead of collapsing uncertainty into an ad-hoc scalar.  The
resulting finite mixture has the existing AC-RHAP coordinate as its exact
mean, while its covariance contains within-component scale, between-anchor
ambiguity, and between-member disagreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import beta as beta_distribution


@dataclass(frozen=True)
class PredictiveMoments:
    """First two moments and optional log density of a Student-t mixture."""

    mean: torch.Tensor
    covariance: torch.Tensor
    within_covariance: torch.Tensor
    between_candidate_covariance: torch.Tensor
    between_member_covariance: torch.Tensor
    root_trace_scale: torch.Tensor
    log_prob: torch.Tensor | None = None


def _normalized_mixture_weights(
    member_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> torch.Tensor:
    if member_weights.ndim != 2:
        raise ValueError("member_weights must have shape [members, batch]")
    if candidate_weights.ndim != 2:
        raise ValueError("candidate_weights must have shape [batch, candidates]")
    if member_weights.shape[1] != candidate_weights.shape[0]:
        raise ValueError("member and candidate batch dimensions differ")
    if torch.any(member_weights < 0.0) or torch.any(candidate_weights < 0.0):
        raise ValueError("mixture weights must be nonnegative")
    weights = member_weights.unsqueeze(2) * candidate_weights.unsqueeze(0)
    totals = weights.sum(dim=(0, 2), keepdim=True)
    if not torch.all(torch.isfinite(weights)) or torch.any(totals <= 0.0):
        raise ValueError("every batch row must have finite positive weight mass")
    return weights / totals


def student_t_mixture_moments(
    component_locations: torch.Tensor,
    component_scales: torch.Tensor,
    member_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
    degrees_freedom: float,
    target: torch.Tensor | None = None,
) -> PredictiveMoments:
    """Return exact moments of the member-by-candidate Student-t mixture.

    Args:
        component_locations: ``[M,B,K,2]`` component centers in metres.
        component_scales: matching independent-axis Student-t scales.
        member_weights: label-free member weights ``[M,B]``.
        candidate_weights: projected posterior ``[B,K]``.
        degrees_freedom: common Student-t degrees of freedom; must exceed two.
        target: optional coordinates ``[B,2]`` for exact mixture log density.

    The component family is a product of two univariate Student-t densities,
    matching ``AnchorStudentTModel.student_log_prob``.  The returned full
    covariance is a moment summary of the finite mixture; the mixture itself
    is not claimed to be Gaussian or elliptically Student-t.
    """
    if component_locations.ndim != 4 or component_locations.shape[-1] != 2:
        raise ValueError("component_locations must have shape [M,B,K,2]")
    if component_scales.shape != component_locations.shape:
        raise ValueError("component_scales must match component_locations")
    if not torch.all(torch.isfinite(component_locations)):
        raise ValueError("component locations must be finite")
    if not torch.all(torch.isfinite(component_scales)) or torch.any(
        component_scales <= 0.0
    ):
        raise ValueError("component scales must be finite and positive")
    if not math.isfinite(float(degrees_freedom)) or (
        float(degrees_freedom) <= 2.0
    ):
        raise ValueError("degrees_freedom must exceed two for finite covariance")
    members, batch, candidates, _ = component_locations.shape
    if member_weights.shape != (members, batch):
        raise ValueError("member_weights shape does not match components")
    if candidate_weights.shape != (batch, candidates):
        raise ValueError("candidate_weights shape does not match components")

    mixture_weights = _normalized_mixture_weights(
        member_weights, candidate_weights)
    member_probabilities = member_weights / member_weights.sum(
        dim=0, keepdim=True)
    candidate_probabilities = candidate_weights / candidate_weights.sum(
        dim=1, keepdim=True)
    mean = torch.sum(
        mixture_weights.unsqueeze(-1) * component_locations,
        dim=(0, 2),
    )
    variance_factor = float(degrees_freedom) / (
        float(degrees_freedom) - 2.0)
    component_covariance = torch.diag_embed(
        component_scales.square() * variance_factor)
    within_covariance = torch.sum(
        mixture_weights.unsqueeze(-1).unsqueeze(-1) * component_covariance,
        dim=(0, 2),
    )

    # Hierarchical law of total covariance.  Candidate means first average
    # members with rho; the member term is disagreement around that candidate
    # mean, and the candidate term is ambiguity among those candidate means.
    candidate_mean = torch.sum(
        member_probabilities.unsqueeze(-1).unsqueeze(-1)
        * component_locations,
        dim=0,
    )
    member_centered = component_locations - candidate_mean.unsqueeze(0)
    between_member_covariance = torch.sum(
        mixture_weights.unsqueeze(-1).unsqueeze(-1)
        * (
            member_centered.unsqueeze(-1)
            * member_centered.unsqueeze(-2)
        ),
        dim=(0, 2),
    )
    candidate_centered = candidate_mean - mean.unsqueeze(1)
    between_candidate_covariance = torch.sum(
        candidate_probabilities.unsqueeze(-1).unsqueeze(-1)
        * (
            candidate_centered.unsqueeze(-1)
            * candidate_centered.unsqueeze(-2)
        ),
        dim=1,
    )
    covariance = (
        within_covariance
        + between_candidate_covariance
        + between_member_covariance
    )
    within_covariance = 0.5 * (
        within_covariance + within_covariance.transpose(-1, -2))
    between_candidate_covariance = 0.5 * (
        between_candidate_covariance
        + between_candidate_covariance.transpose(-1, -2))
    between_member_covariance = 0.5 * (
        between_member_covariance
        + between_member_covariance.transpose(-1, -2))
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    # sqrt(trace(Sigma)) = sqrt(E ||Y-EY||^2), not std(||Y-EY||).
    root_trace_scale = covariance.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1).sqrt()

    log_prob = None
    if target is not None:
        if target.shape != (batch, 2):
            raise ValueError("target must have shape [B,2]")
        if not torch.all(torch.isfinite(target)):
            raise ValueError("target must be finite")
        df = float(degrees_freedom)
        z = (
            target.unsqueeze(0).unsqueeze(2) - component_locations
        ) / component_scales
        normalizer = (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * math.log(df * math.pi)
        )
        component_log_prob = (
            normalizer
            - torch.log(component_scales)
            - 0.5 * (df + 1.0) * torch.log1p(z.square() / df)
        ).sum(dim=-1)
        negative_infinity = torch.full_like(
            mixture_weights, float("-inf"))
        log_weights = torch.where(
            mixture_weights > 0.0,
            torch.log(mixture_weights),
            negative_infinity,
        )
        log_prob = torch.logsumexp(
            log_weights + component_log_prob, dim=(0, 2))
    return PredictiveMoments(
        mean=mean,
        covariance=covariance,
        within_covariance=within_covariance,
        between_candidate_covariance=between_candidate_covariance,
        between_member_covariance=between_member_covariance,
        root_trace_scale=root_trace_scale,
        log_prob=log_prob,
    )


def mahalanobis_scores(
    target: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    eigenvalue_floor_m2: float = 1e-6,
) -> np.ndarray:
    """Compute stable covariance-normalized radial nonconformity scores."""
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if target.shape != mean.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target and mean must have shape [n,2]")
    if covariance.shape != (len(target), 2, 2):
        raise ValueError("covariance must have shape [n,2,2]")
    if eigenvalue_floor_m2 <= 0.0:
        raise ValueError("eigenvalue_floor_m2 must be positive")
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, float(eigenvalue_floor_m2))
    inverse = np.einsum(
        "nij,nj,nkj->nik", eigenvectors, 1.0 / eigenvalues, eigenvectors)
    residual = target - mean
    squared = np.einsum("ni,nij,nj->n", residual, inverse, residual)
    return np.sqrt(np.maximum(squared, 0.0))


def split_conformal_quantile(
    calibration_scores: np.ndarray,
    miscoverage: float,
) -> float:
    """Finite-sample split-conformal order statistic.

    Returns the ``ceil((n+1)(1-alpha))``-th smallest score.  If that rank is
    ``n+1``, the requested level has no finite distribution-free radius and
    the function returns ``+inf`` rather than silently clipping the rank.
    Under exchangeability, this preserves marginal coverage at least
    ``1-alpha``.
    """
    scores = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    if len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("calibration_scores must be finite and nonempty")
    if not 0.0 < float(miscoverage) < 1.0:
        raise ValueError("miscoverage must lie in (0,1)")
    rank = int(math.ceil((len(scores) + 1) * (1.0 - miscoverage)))
    if rank > len(scores):
        # The only distribution-free set attaining this finite-sample level
        # is the unbounded set.  Returning +inf preserves the guarantee and
        # prevents a silent, anti-conservative clip to the sample maximum.
        return float("inf")
    rank = max(rank, 1)
    return float(np.partition(scores, rank - 1)[rank - 1])


def covariance_ellipse_area(
    covariance: np.ndarray,
    radius: float | np.ndarray,
    *,
    eigenvalue_floor_m2: float = 1e-6,
) -> np.ndarray:
    """Area of ``(y-mu)^T Sigma^-1 (y-mu) <= radius^2`` in square metres."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 3 or covariance.shape[1:] != (2, 2):
        raise ValueError("covariance must have shape [n,2,2]")
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (covariance + np.swapaxes(covariance, 1, 2)))
    eigenvalues = np.maximum(eigenvalues, float(eigenvalue_floor_m2))
    return math.pi * np.asarray(radius, dtype=np.float64) ** 2 * np.sqrt(
        eigenvalues[:, 0] * eigenvalues[:, 1])


def risk_coverage_curve(
    errors_m: np.ndarray,
    uncertainty: np.ndarray,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> list[dict[str, float | int]]:
    """Selective-localization risk when retaining least-uncertain scans."""
    errors = np.asarray(errors_m, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if len(errors) == 0 or len(errors) != len(uncertainty):
        raise ValueError("errors and uncertainty must be nonempty and aligned")
    order = np.argsort(uncertainty, kind="stable")
    rows: list[dict[str, float | int]] = []
    for coverage in coverages:
        if not 0.0 < float(coverage) <= 1.0:
            raise ValueError("coverages must lie in (0,1]")
        retained = max(1, int(math.ceil(float(coverage) * len(errors))))
        chosen = order[:retained]
        selected_errors = errors[chosen]
        rows.append({
            "target_coverage": float(coverage),
            "retained": int(retained),
            "empirical_coverage": float(retained / len(errors)),
            "mae_m": float(selected_errors.mean()),
            "rmse_m": float(np.sqrt(np.mean(selected_errors ** 2))),
            "p90_m": float(np.quantile(selected_errors, 0.90)),
            "max_uncertainty": float(uncertainty[chosen].max()),
        })
    return rows


def clopper_pearson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(
        beta_distribution.ppf(tail, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta_distribution.ppf(
            1.0 - tail, successes + 1, trials - successes))
    return lower, upper


def _one_sided_clopper_pearson_upper(
    successes: int,
    trials: int,
    delta: float,
) -> float:
    """Exact one-sided upper confidence bound for a Bernoulli mean."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if successes == trials:
        return 1.0
    return float(beta_distribution.ppf(
        1.0 - float(delta), successes + 1, trials - successes))


def _designed_uncertainty_thresholds(
    design_uncertainty: np.ndarray,
    target_retentions: tuple[float, ...],
) -> list[float]:
    design = np.asarray(design_uncertainty, dtype=np.float64).reshape(-1)
    if len(design) == 0 or not np.all(np.isfinite(design)):
        raise ValueError("design uncertainty must be finite and nonempty")
    thresholds = []
    for retention in target_retentions:
        if not 0.0 < float(retention) <= 1.0:
            raise ValueError("target_retentions must lie in (0,1]")
        thresholds.append(float(np.quantile(
            design, float(retention), method="higher")))
    return sorted(set(thresholds))


def certify_unsafe_acceptance(
    design_uncertainty: np.ndarray,
    certification_uncertainty: np.ndarray,
    certification_errors_m: np.ndarray,
    *,
    error_limit_m: float,
    risk_bound: float,
    familywise_delta: float = 0.05,
    target_retentions: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 1.0),
) -> dict:
    """PAC-select an abstention threshold with unsafe-acceptance control.

    Thresholds are designed from an uncertainty-only split.  A disjoint
    certification split evaluates the bounded loss

    ``1{uncertainty <= threshold and localization_error > error_limit}``.

    One-sided Clopper--Pearson bounds and a Bonferroni correction make the
    selected threshold valid with probability at least
    ``1-familywise_delta`` over the certification draw.  The guarantee is for
    the *marginal unsafe-acceptance probability*, not conditional error among
    accepted scans.
    """
    design = np.asarray(design_uncertainty, dtype=np.float64).reshape(-1)
    cert_u = np.asarray(
        certification_uncertainty, dtype=np.float64).reshape(-1)
    cert_e = np.asarray(certification_errors_m, dtype=np.float64).reshape(-1)
    if len(design) == 0 or len(cert_u) == 0 or len(cert_u) != len(cert_e):
        raise ValueError("design and aligned certification samples are required")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(cert_u)):
        raise ValueError("uncertainties must be finite")
    if not np.all(np.isfinite(cert_e)) or np.any(cert_e < 0.0):
        raise ValueError("certification errors must be finite and nonnegative")
    if error_limit_m <= 0.0:
        raise ValueError("error_limit_m must be positive")
    if not 0.0 < risk_bound < 1.0:
        raise ValueError("risk_bound must lie in (0,1)")
    if not 0.0 < familywise_delta < 1.0:
        raise ValueError("familywise_delta must lie in (0,1)")
    thresholds = _designed_uncertainty_thresholds(
        design, target_retentions)
    per_threshold_delta = familywise_delta / max(len(thresholds), 1)
    rows = []
    for threshold in thresholds:
        accepted = cert_u <= threshold
        unsafe = accepted & (cert_e > float(error_limit_m))
        unsafe_count = int(unsafe.sum())
        upper = _one_sided_clopper_pearson_upper(
            unsafe_count, len(cert_u), per_threshold_delta)
        rows.append({
            "threshold": threshold,
            "certification_acceptance": float(accepted.mean()),
            "unsafe_count": unsafe_count,
            "certification_rows": int(len(cert_u)),
            "unsafe_rate": float(unsafe.mean()),
            "simultaneous_upper_bound": upper,
            "eligible": bool(upper <= risk_bound),
        })
    eligible = [row for row in rows if row["eligible"]]
    selected = max(eligible, key=lambda row: row["threshold"]) if eligible else {
        "threshold": float("-inf"),
        "certification_acceptance": 0.0,
        "unsafe_count": 0,
        "certification_rows": int(len(cert_u)),
        "unsafe_rate": 0.0,
        "simultaneous_upper_bound": 0.0,
        "eligible": True,
    }
    return {
        "loss": "accepted_and_error_above_limit",
        "error_limit_m": float(error_limit_m),
        "risk_bound": float(risk_bound),
        "familywise_delta": float(familywise_delta),
        "familywise_confidence": float(1.0 - familywise_delta),
        "candidate_thresholds": int(len(thresholds)),
        "per_candidate_delta": float(per_threshold_delta),
        "multiple_testing": "Bonferroni one-sided Clopper-Pearson",
        "selected": selected,
        "candidates": rows,
    }


def certify_group_unsafe_acceptance(
    design_uncertainty: np.ndarray,
    certification_uncertainty: np.ndarray,
    certification_errors_m: np.ndarray,
    certification_groups: np.ndarray,
    *,
    error_limit_m: float,
    risk_bound: float,
    familywise_delta: float = 0.05,
    target_retentions: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 1.0),
) -> dict:
    """Certify group-level probability of any unsafe accepted scan.

    Candidate thresholds use uncertainty from design groups only.  For each
    fixed threshold the certification Bernoulli trial is one entire group,
    with loss one iff *any* scan in that group is both accepted and above the
    localization-error limit.  The Clopper--Pearson guarantee therefore
    requires iid deployment groups, while allowing arbitrary dependence among
    scans inside a group.
    """
    design = np.asarray(design_uncertainty, dtype=np.float64).reshape(-1)
    cert_u = np.asarray(
        certification_uncertainty, dtype=np.float64).reshape(-1)
    cert_e = np.asarray(certification_errors_m, dtype=np.float64).reshape(-1)
    cert_g = np.asarray(certification_groups).reshape(-1)
    if (
        len(cert_u) == 0
        or len(cert_u) != len(cert_e)
        or len(cert_u) != len(cert_g)
    ):
        raise ValueError(
            "aligned certification uncertainty, errors, and groups are required")
    if not np.all(np.isfinite(cert_u)) or not np.all(np.isfinite(design)):
        raise ValueError("uncertainties must be finite")
    if not np.all(np.isfinite(cert_e)) or np.any(cert_e < 0.0):
        raise ValueError("certification errors must be finite and nonnegative")
    if error_limit_m <= 0.0:
        raise ValueError("error_limit_m must be positive")
    if not 0.0 < risk_bound < 1.0:
        raise ValueError("risk_bound must lie in (0,1)")
    if not 0.0 < familywise_delta < 1.0:
        raise ValueError("familywise_delta must lie in (0,1)")
    unique_groups = np.unique(cert_g)
    if len(unique_groups) == 0:
        raise ValueError("at least one certification group is required")

    thresholds = _designed_uncertainty_thresholds(
        design, target_retentions)
    per_threshold_delta = familywise_delta / max(len(thresholds), 1)
    rows = []
    for threshold in thresholds:
        accepted = cert_u <= threshold
        unsafe_scan = accepted & (cert_e > float(error_limit_m))
        group_unsafe = np.asarray([
            bool(np.any(unsafe_scan[cert_g == group]))
            for group in unique_groups
        ])
        group_accepted = np.asarray([
            bool(np.any(accepted[cert_g == group]))
            for group in unique_groups
        ])
        unsafe_count = int(group_unsafe.sum())
        upper = _one_sided_clopper_pearson_upper(
            unsafe_count, len(unique_groups), per_threshold_delta)
        rows.append({
            "threshold": threshold,
            "certification_scan_acceptance": float(accepted.mean()),
            "certification_group_acceptance": float(group_accepted.mean()),
            "unsafe_groups": unsafe_count,
            "certification_groups": int(len(unique_groups)),
            "certification_scans": int(len(cert_u)),
            "unsafe_group_rate": float(group_unsafe.mean()),
            "simultaneous_upper_bound": upper,
            "eligible": bool(upper <= risk_bound),
        })
    eligible = [row for row in rows if row["eligible"]]
    selected = max(eligible, key=lambda row: row["threshold"]) if eligible else {
        "threshold": float("-inf"),
        "certification_scan_acceptance": 0.0,
        "certification_group_acceptance": 0.0,
        "unsafe_groups": 0,
        "certification_groups": int(len(unique_groups)),
        "certification_scans": int(len(cert_u)),
        "unsafe_group_rate": 0.0,
        "simultaneous_upper_bound": 0.0,
        "eligible": True,
    }
    return {
        "loss": "group_any_accepted_scan_above_error_limit",
        "validity_unit": "group",
        "error_limit_m": float(error_limit_m),
        "risk_bound": float(risk_bound),
        "familywise_delta": float(familywise_delta),
        "familywise_confidence": float(1.0 - familywise_delta),
        "candidate_thresholds": int(len(thresholds)),
        "per_candidate_delta": float(per_threshold_delta),
        "multiple_testing": "Bonferroni one-sided Clopper-Pearson",
        "assumption": (
            "iid deployment groups; arbitrary dependence is allowed within "
            "each group"),
        "selected": selected,
        "candidates": rows,
    }
