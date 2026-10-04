"""V18.5 mathematical core.

V18.5 replaces the bounded V18.4 trust tilt with the exact row-wise convex
program

    minimize_p  <p,E>
    subject to  KL(p || q) <= budget.

The dual exponential tilt is not capped.  When the budget reaches the finite
KL limit, the exact solution is q conditioned on the minimum-energy set.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.v18_4_core import (
    cap_anisotropic_residual_transform,
    empirical_survival_reliability,
    fit_empirical_reference,
    masked_softmax,
    normalize_probability,
    spatial_energy_distance,
    temperature_base_posterior,
)

DEFAULT_BISECTION_ITERATIONS = 80
DEFAULT_BRACKET_DOUBLINGS = 80
NUMERICAL_TOLERANCE = 1e-12


def kl_p_to_q(
    p: torch.Tensor,
    q: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Row-wise KL(p||q); q-zero coordinates cannot receive p mass."""

    if p.shape != q.shape or p.shape != mask.shape or p.ndim != 2:
        raise ValueError("p, q and mask must share shape [batch,support]")
    mask = mask.bool()
    p_n = normalize_probability(p.double(), mask)
    q_n = normalize_probability(q.double(), mask)
    impossible = mask & (p_n > 0.0) & (q_n <= 0.0)
    safe_p = p_n.clamp_min(torch.finfo(torch.float64).tiny)
    safe_q = q_n.clamp_min(torch.finfo(torch.float64).tiny)
    term = torch.where(
        mask & (p_n > 0.0),
        p_n * (safe_p.log() - safe_q.log()),
        torch.zeros_like(p_n),
    )
    result = term.sum(dim=1)
    return torch.where(
        impossible.any(dim=1),
        torch.full_like(result, float("inf")),
        result,
    )


def _tilt_one(
    q: torch.Tensor,
    energy: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Stable exponential tilt on one strictly-positive support."""

    if not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("finite beta must be nonnegative")
    shifted = energy - energy.min()
    logw = q.log() - float(beta) * shifted
    return torch.softmax(logw, dim=0)


def _kl_one(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    positive = p > 0.0
    return torch.sum(
        torch.where(
            positive,
            p * (p.clamp_min(torch.finfo(p.dtype).tiny).log() - q.log()),
            torch.zeros_like(p),
        )
    )


def exact_kl_energy_projection(
    base_probability: torch.Tensor,
    energy: torch.Tensor,
    *,
    epsilon_kl: torch.Tensor | float,
    mask: torch.Tensor | None = None,
    bisection_iterations: int = DEFAULT_BISECTION_ITERATIONS,
    maximum_bracket_doublings: int = DEFAULT_BRACKET_DOUBLINGS,
    tolerance: float = NUMERICAL_TOLERANCE,
) -> dict[str, torch.Tensor]:
    """Solve the V18.5 KL-constrained energy projection exactly per row.

    For a nonconstant energy and an active finite constraint,

        p_beta(j) proportional to q(j) exp(-beta E(j))

    with beta >= 0 selected so KL(p_beta||q)=budget.  beta has no scientific
    upper bound.  If the budget reaches the limiting KL, the exact optimum is
    q conditioned on the full minimum-energy tie set.
    """

    if base_probability.shape != energy.shape or base_probability.ndim != 2:
        raise ValueError("base_probability and energy must share [B,S] shape")
    if mask is None:
        mask = torch.ones_like(base_probability, dtype=torch.bool)
    if mask.shape != base_probability.shape:
        raise ValueError("mask shape mismatch")
    if int(bisection_iterations) < 64:
        raise ValueError("V18.5 requires at least 64 bisection iterations")
    if int(maximum_bracket_doublings) <= 0:
        raise ValueError("maximum_bracket_doublings must be positive")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    mask = mask.bool()
    q_input = torch.where(
        mask, base_probability, torch.zeros_like(base_probability)
    )
    if not bool(torch.isfinite(q_input).all()) or bool((q_input < 0.0).any()):
        raise ValueError("base probability must be finite and nonnegative")
    if bool((q_input.sum(dim=1) <= 0.0).any()):
        raise ValueError("each row needs positive valid base mass")
    valid_energy = torch.where(mask, energy, torch.zeros_like(energy))
    if not bool(torch.isfinite(valid_energy).all()):
        raise ValueError("valid energy must be finite")

    q = normalize_probability(q_input.double(), mask)
    e = valid_energy.double()
    budget = torch.as_tensor(
        epsilon_kl, dtype=torch.float64, device=q.device
    )
    if budget.ndim == 0:
        budget = budget.expand(len(q))
    if budget.shape != (len(q),):
        raise ValueError("epsilon_kl must be scalar or [batch]")
    if not bool(torch.isfinite(budget).all()) or bool((budget < 0.0).any()):
        raise ValueError("epsilon_kl must be finite and nonnegative")

    posterior = torch.zeros_like(q)
    beta = torch.zeros(len(q), dtype=torch.float64, device=q.device)
    beta_infinite = torch.zeros(len(q), dtype=torch.bool, device=q.device)
    limiting_kl = torch.zeros(len(q), dtype=torch.float64, device=q.device)
    minimum_energy_mass = torch.ones(
        len(q), dtype=torch.float64, device=q.device
    )
    constraint_active = torch.zeros(len(q), dtype=torch.bool, device=q.device)

    for row in range(len(q)):
        row_mask = mask[row]
        positive = row_mask & (q[row] > 0.0)
        qr = q[row, positive]
        er = e[row, positive]
        if qr.numel() == 0:
            raise RuntimeError("normalized row lost all positive support")

        emin = torch.min(er)
        minimum = er == emin
        q_min = qr[minimum].sum()
        limit = -torch.log(q_min)
        limiting_kl[row] = limit
        minimum_energy_mass[row] = q_min

        original_row = q[row].clone()
        if float(budget[row]) == 0.0 or bool((er == er[0]).all()):
            posterior[row] = original_row
            continue

        if float(budget[row]) >= float(limit) - float(tolerance):
            conditioned = torch.zeros_like(qr)
            conditioned[minimum] = qr[minimum] / q_min
            posterior[row, positive] = conditioned
            beta_infinite[row] = True
            constraint_active[row] = bool(
                float(budget[row]) < float(limit) + float(tolerance)
            )
            continue

        target = float(budget[row])
        low = 0.0
        high = 1.0
        high_kl = float(_kl_one(_tilt_one(qr, er, high), qr))
        doublings = 0
        while high_kl < target:
            low = high
            high *= 2.0
            doublings += 1
            if doublings > int(maximum_bracket_doublings):
                raise RuntimeError(
                    "failed to bracket exact KL dual; refusing to clip beta"
                )
            high_kl = float(_kl_one(_tilt_one(qr, er, high), qr))

        for _ in range(int(bisection_iterations)):
            mid = 0.5 * (low + high)
            value = float(_kl_one(_tilt_one(qr, er, mid), qr))
            if value <= target:
                low = mid
            else:
                high = mid

        beta[row] = low
        posterior[row, positive] = _tilt_one(qr, er, low)
        constraint_active[row] = True

    posterior = torch.where(mask, posterior, torch.zeros_like(posterior))
    realized = kl_p_to_q(posterior, q, mask).clamp_min(0.0)
    if bool((realized > budget + 5e-10).any()):
        raise RuntimeError("exact KL projection exceeded its row budget")

    expected_before = torch.sum(q * e, dim=1)
    expected_after = torch.sum(posterior * e, dim=1)
    if bool((expected_after > expected_before + 1e-10).any()):
        raise RuntimeError("KL projection increased expected correction energy")

    # Preserve exact input identity for zero-budget rows when the caller
    # already supplied a normalized probability vector.
    input_mass = q_input.double().sum(dim=1)
    normalized_input = (input_mass - 1.0).abs() <= (
        4.0 * torch.finfo(base_probability.dtype).eps
    )
    zero_budget = budget == 0.0
    identity_rows = zero_budget & normalized_input
    output = posterior.to(base_probability.dtype)
    if bool(identity_rows.any()):
        output = torch.where(identity_rows[:, None], q_input, output)
        realized = torch.where(
            identity_rows,
            torch.zeros_like(realized),
            realized,
        )

    return {
        "posterior": output,
        "dual_beta": beta,
        "dual_beta_infinite": beta_infinite,
        "realized_kl": realized,
        "epsilon_kl_effective": budget,
        "limiting_kl": limiting_kl,
        "constraint_active": constraint_active,
        "minimum_energy_mass": minimum_energy_mass,
        "expected_energy_before": expected_before,
        "expected_energy_after": expected_after,
    }


def radio_trust_projection(
    q_temperature: torch.Tensor,
    radio_energy: torch.Tensor,
    radio_compatibility: torch.Tensor,
    *,
    epsilon_kl: float,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the V18.5 compatibility-scaled radio-only trust region."""

    if radio_compatibility.shape != (len(q_temperature),):
        raise ValueError("radio_compatibility must have one value per row")
    if not bool(torch.isfinite(radio_compatibility).all()):
        raise ValueError("radio compatibility must be finite")
    if bool(
        ((radio_compatibility < 0.0) | (radio_compatibility > 1.0)).any()
    ):
        raise ValueError("radio compatibility must lie in [0,1]")
    if not math.isfinite(float(epsilon_kl)) or float(epsilon_kl) < 0.0:
        raise ValueError("epsilon_kl must be finite and nonnegative")
    effective = radio_compatibility.double() * float(epsilon_kl)
    result = exact_kl_energy_projection(
        q_temperature,
        radio_energy,
        epsilon_kl=effective,
        mask=mask,
    )
    result["radio_compatibility"] = radio_compatibility
    result["epsilon_kl_max"] = effective.new_tensor(float(epsilon_kl))
    return result


def posterior_mean(
    probability: torch.Tensor,
    component_location: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Registered V18.5 point summary."""

    if component_location.shape != (*probability.shape, 2):
        raise ValueError("component_location must be [B,S,2]")
    p = normalize_probability(probability, mask).to(component_location)
    return torch.sum(p.unsqueeze(-1) * component_location, dim=1)


def paired_standard_error(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("paired values must be finite 1-D with at least 2 groups")
    return float(array.std(ddof=1) / math.sqrt(len(array)))


def paired_one_se_set(
    group_losses: Mapping[str, Sequence[float] | np.ndarray],
    *,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Return the preregistered paired one-standard-error near-optimal set."""

    if not group_losses:
        raise ValueError("group_losses cannot be empty")
    arrays = {
        str(key): np.asarray(value, dtype=np.float64)
        for key, value in group_losses.items()
    }
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all cells must have aligned group-loss vectors")
    shape = next(iter(shapes))
    if len(shape) != 1 or shape[0] < 2:
        raise ValueError("group-loss vectors must be 1-D with >=2 groups")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("group losses must be finite")

    means = {key: float(value.mean()) for key, value in arrays.items()}
    best = min(means, key=lambda key: (means[key], key))
    reference = arrays[best]
    records: dict[str, dict[str, float | bool]] = {}
    retained: list[str] = []
    for key, value in arrays.items():
        difference = value - reference
        mean_difference = float(difference.mean())
        se = 0.0 if key == best else paired_standard_error(difference)
        keep = mean_difference <= se + float(tolerance)
        if keep:
            retained.append(key)
        records[key] = {
            "mean_loss": means[key],
            "mean_difference_vs_best": mean_difference,
            "paired_se": se,
            "retained": bool(keep),
        }
    return {
        "best": best,
        "retained": tuple(sorted(retained)),
        "records": records,
    }


def select_joint_calibration(
    cells: Sequence[Mapping[str, object]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """V18.5 paired NLL one-SE -> ES one-SE -> deterministic choice."""

    if not cells:
        raise ValueError("calibration cells cannot be empty")
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    if len(by_id) != len(cells):
        raise ValueError("cell_id values must be unique")

    nll_gate = paired_one_se_set(
        {
            key: np.asarray(cell["group_nll"], dtype=np.float64)
            for key, cell in by_id.items()
        },
        tolerance=tolerance,
    )
    nll_ids = set(nll_gate["retained"])
    es_gate = paired_one_se_set(
        {
            key: np.asarray(by_id[key]["group_energy_score"], dtype=np.float64)
            for key in nll_ids
        },
        tolerance=tolerance,
    )
    final_ids = list(es_gate["retained"])

    def choice_key(key: str) -> tuple:
        cell = by_id[key]
        return (
            float(cell["posterior_mean_mae_m"]),
            float(cell["epsilon_kl"]),
            abs(math.log(float(cell["temperature"]))),
            abs(math.log(float(cell["scale_multiplier"]))),
            float(cell["temperature"]),
            float(cell["scale_multiplier"]),
            key,
        )

    selected = min(final_ids, key=choice_key)
    return {
        "raw_nll_best": nll_gate["best"],
        "nll_one_se": nll_gate,
        "raw_es_best_within_nll_set": es_gate["best"],
        "es_one_se": es_gate,
        "selected_cell_id": selected,
        "selected_cell": dict(by_id[selected]),
    }


__all__ = [
    "cap_anisotropic_residual_transform",
    "empirical_survival_reliability",
    "exact_kl_energy_projection",
    "fit_empirical_reference",
    "kl_p_to_q",
    "masked_softmax",
    "paired_one_se_set",
    "posterior_mean",
    "radio_trust_projection",
    "select_joint_calibration",
    "spatial_energy_distance",
    "temperature_base_posterior",
]
