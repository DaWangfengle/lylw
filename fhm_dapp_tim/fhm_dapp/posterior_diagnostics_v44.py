"""Exact, non-selective information diagnostics for a joint member-anchor law."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from fhm_dapp_tim.fhm_dapp.minimal_v3 import joint_spatial_moments


DEFAULT_CREDIBLE_LEVELS = (0.50, 0.80, 0.90, 0.95)


def joint_information_diagnostics(
    joint_weights: torch.Tensor,
    support_xy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply total covariance and entropy identities without approximation."""
    moments = joint_spatial_moments(joint_weights, support_xy)
    candidate_covariance = moments["candidate_covariance"]
    member_covariance = moments["member_covariance"]
    total_covariance = moments["covariance"]
    # Registered v4.4 components are point masses at anchor coordinates.
    # Keep the zero term explicit so the three-term model boundary is audited
    # without inventing an unregistered local-noise component.
    component_covariance = torch.zeros_like(total_covariance)
    candidate_trace = torch.diagonal(
        candidate_covariance, dim1=1, dim2=2
    ).sum(dim=1)
    member_trace = torch.diagonal(member_covariance, dim1=1, dim2=2).sum(dim=1)
    total_trace = torch.diagonal(total_covariance, dim1=1, dim2=2).sum(dim=1)
    if not torch.allclose(
        total_trace, candidate_trace + member_trace, atol=1e-5, rtol=1e-5
    ):
        raise RuntimeError("joint covariance trace decomposition is inconsistent")
    member_mass = moments["member_mass"]
    candidate_mass = joint_weights.sum(dim=0)
    member_entropy = -torch.sum(
        member_mass * torch.log(member_mass.clamp_min(1e-30)), dim=0
    )
    candidate_entropy = -torch.sum(
        candidate_mass * torch.log(candidate_mass.clamp_min(1e-30)), dim=1
    )
    member_eigenvalues = torch.linalg.eigvalsh(member_covariance)
    candidate_eigenvalues = torch.linalg.eigvalsh(candidate_covariance)
    return {
        **moments,
        "component_covariance": component_covariance,
        "component_covariance_trace_m2": torch.zeros_like(total_trace),
        "candidate_covariance_trace_m2": candidate_trace,
        "member_covariance_trace_m2": member_trace,
        "total_covariance_trace_m2": total_trace,
        "member_trace_fraction": member_trace / total_trace.clamp_min(1e-30),
        "effective_member_count": torch.exp(member_entropy),
        "effective_candidate_count": torch.exp(candidate_entropy),
        "member_covariance_eigenvalues_m2": member_eigenvalues,
        "candidate_covariance_eigenvalues_m2": candidate_eigenvalues,
    }


def radial_credible_balls(
    probability: torch.Tensor,
    support: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: Iterable[float] = DEFAULT_CREDIBLE_LEVELS,
) -> dict[str, torch.Tensor]:
    """Return posterior-mass radii about the posterior mean and coverage.

    This geometric definition remains meaningful when a true coordinate is not
    itself an anchor in the training dictionary.
    """
    if probability.ndim != 2 or not probability.is_floating_point():
        raise ValueError("probability must be floating [batch,support]")
    batch, candidates = probability.shape
    if support.ndim != 3 or support.shape[:2] != (batch, candidates):
        raise ValueError("support must have shape [batch,support,dimension]")
    if target.shape != (batch, support.shape[2]):
        raise ValueError("target must have shape [batch,dimension]")
    if not bool(torch.isfinite(probability).all()) or bool((probability < 0).any()):
        raise ValueError("probability must be finite and non-negative")
    totals = probability.sum(dim=1)
    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5):
        raise ValueError("probability rows must sum to one")
    requested = tuple(float(value) for value in levels)
    if not requested or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in requested
    ):
        raise ValueError("credible levels must lie strictly between zero and one")
    if len(set(requested)) != len(requested):
        raise ValueError("credible levels must be unique")

    mean = torch.sum(probability.unsqueeze(2) * support, dim=1)
    support_distance = torch.linalg.vector_norm(
        support - mean.unsqueeze(1), dim=2
    )
    ordered_distance, order = torch.sort(support_distance, dim=1, stable=True)
    ordered_probability = torch.gather(probability, 1, order)
    cumulative = torch.cumsum(ordered_probability, dim=1)
    target_distance = torch.linalg.vector_norm(target - mean, dim=1)
    output: dict[str, torch.Tensor] = {
        "posterior_mean": mean,
        "target_distance_from_mean_m": target_distance,
    }
    for level in requested:
        threshold = probability.new_full((batch, 1), level)
        position = torch.searchsorted(cumulative.contiguous(), threshold).squeeze(1)
        position = position.clamp(max=candidates - 1)
        radius = ordered_distance.gather(1, position.unsqueeze(1)).squeeze(1)
        label = f"{level:.2f}"
        output[f"credible_radius_{label}_m"] = radius
        output[f"credible_coverage_{label}"] = target_distance <= (
            radius + 1e-6
        )
    return output
