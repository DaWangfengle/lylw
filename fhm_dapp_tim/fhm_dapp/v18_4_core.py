"""Mathematical core for V18.4.

The functions in this module are model-agnostic.  They implement the frozen
mathematical contracts for:

- masked posterior normalization;
- robust training-derived energy scaling;
- empirical-CDF reliability;
- KL-bounded exponential posterior correction;
- debiased entropic spatial transport;
- component-center weighted geometric median.

No function learns an evidence gate or uses evaluation labels.
"""

from __future__ import annotations

import math

import torch


DEFAULT_SINKHORN_ITERATIONS = 40
DEFAULT_TRUST_BISECTION_ITERATIONS = 40
NUMERICAL_EPS = 1e-12


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax over valid support, returning exact zero on padding."""

    if logits.shape != mask.shape:
        raise ValueError("logits and mask must have identical shape")
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("every posterior row must contain valid support")
    masked = logits.masked_fill(~mask, float("-inf"))
    probability = torch.softmax(masked, dim=1)
    return torch.where(mask, probability, torch.zeros_like(probability))


def normalize_probability(
    probability: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize nonnegative weights on an optional validity mask."""

    if probability.ndim != 2:
        raise ValueError("probability must be [batch,support]")
    if mask is None:
        mask = torch.ones_like(probability, dtype=torch.bool)
    if mask.shape != probability.shape:
        raise ValueError("mask shape mismatch")
    mask = mask.bool()
    if bool((probability < 0.0).any()):
        raise ValueError("probability cannot be negative")
    value = torch.where(mask, probability, torch.zeros_like(probability))
    total = value.sum(dim=1, keepdim=True)
    if bool((total <= 0.0).any()):
        raise ValueError("every probability row needs positive valid mass")
    return value / total


def robust_energy_scale(
    energy: torch.Tensor,
    mask: torch.Tensor,
    *,
    minimum_scale: float = 1e-4,
) -> torch.Tensor:
    """Median across scans of within-scan energy MAD.

    This is intended to be fitted on training/fitting scans and then frozen.
    Each scan contributes one MAD irrespective of its number of candidates.
    """

    if energy.shape != mask.shape:
        raise ValueError("energy and mask must have identical shape")
    if energy.ndim != 2:
        raise ValueError("energy must be [rows,support]")
    row_mads = []
    for row_energy, row_mask in zip(energy, mask.bool()):
        values = row_energy[row_mask]
        if len(values) == 0:
            raise ValueError("energy row has no valid candidates")
        median = torch.median(values)
        mad = torch.median((values - median).abs())
        row_mads.append(mad)
    scale = torch.median(torch.stack(row_mads))
    return scale.clamp(min=float(minimum_scale))


def normalize_energy(
    energy: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """Shift by scan minimum and divide by a frozen global robust scale.

    Additive energy shifts do not change an exponentially tilted posterior.
    Invalid padding is returned as zero and must remain masked downstream.
    """

    if energy.shape != mask.shape:
        raise ValueError("energy and mask must have identical shape")
    mask = mask.bool()
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("every energy row needs valid candidates")
    scale_tensor = torch.as_tensor(
        scale,
        dtype=energy.dtype,
        device=energy.device,
    ).clamp(min=1e-8)
    valid_energy = energy.masked_fill(~mask, float("inf"))
    row_min = valid_energy.min(dim=1, keepdim=True).values
    normalized = (energy - row_min) / scale_tensor
    return torch.where(mask, normalized, torch.zeros_like(normalized))


def fit_empirical_reference(values: torch.Tensor) -> torch.Tensor:
    """Return a sorted finite 1-D reference for label-free reliability."""

    flat = values.detach().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if len(flat) < 2:
        raise ValueError("empirical reliability reference needs >=2 values")
    return torch.sort(flat).values


def empirical_survival_reliability(
    values: torch.Tensor,
    sorted_reference: torch.Tensor,
) -> torch.Tensor:
    """Smoothed empirical survival probability in (0,1).

    F_hat(x) = (rank_right(x) + 0.5) / (n + 1)
    r(x) = 1 - F_hat(x).

    The half-count smoothing prevents exact 0/1 reliability while preserving
    monotonicity: larger mismatch can never receive larger reliability.
    """

    reference = sorted_reference.reshape(-1)
    if len(reference) < 2:
        raise ValueError("reference needs >=2 samples")
    if bool((reference[1:] < reference[:-1]).any()):
        raise ValueError("reference must be sorted")
    rank = torch.searchsorted(
        reference.to(device=values.device, dtype=values.dtype),
        values,
        right=True,
    ).to(values.dtype)
    n = float(len(reference))
    cdf = (rank + 0.5) / (n + 1.0)
    return (1.0 - cdf).clamp(min=0.0, max=1.0)


def _tilted_posterior(
    base_probability: torch.Tensor,
    energy: torch.Tensor,
    rho: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    q = normalize_probability(base_probability, mask)
    log_q = torch.where(
        mask,
        q.clamp(min=NUMERICAL_EPS).log(),
        torch.full_like(q, float("-inf")),
    )
    while rho.ndim < q.ndim:
        rho = rho.unsqueeze(-1)
    log_weight = log_q - rho * energy
    return masked_softmax(log_weight, mask)


def kl_p_to_q(
    p: torch.Tensor,
    q: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Row-wise KL(p || q) on valid support."""

    p = normalize_probability(p, mask)
    q = normalize_probability(q, mask)
    log_p = p.clamp(min=NUMERICAL_EPS).log()
    log_q = q.clamp(min=NUMERICAL_EPS).log()
    term = torch.where(
        mask,
        p * (log_p - log_q),
        torch.zeros_like(p),
    )
    return term.sum(dim=1)


def trust_region_posterior_correction(
    base_probability: torch.Tensor,
    correction_energy: torch.Tensor,
    *,
    epsilon_kl: float,
    mask: torch.Tensor | None = None,
    bisection_iterations: int = DEFAULT_TRUST_BISECTION_ITERATIONS,
) -> dict[str, torch.Tensor]:
    """Apply the maximum correction in rho in [0,1] satisfying KL(p||q).

    The operation is deterministic and has no learned routing parameters.
    """

    if base_probability.shape != correction_energy.shape:
        raise ValueError("base probability and energy shape mismatch")
    if float(epsilon_kl) < 0.0:
        raise ValueError("epsilon_kl cannot be negative")
    if mask is None:
        mask = torch.ones_like(base_probability, dtype=torch.bool)
    mask = mask.bool()
    q = normalize_probability(base_probability, mask)

    zero_rho = torch.zeros(len(q), device=q.device, dtype=q.dtype)
    one_rho = torch.ones_like(zero_rho)
    full = _tilted_posterior(q, correction_energy, one_rho, mask)
    full_kl = kl_p_to_q(full, q, mask)

    if float(epsilon_kl) == 0.0:
        return {
            "posterior": q,
            "rho": zero_rho,
            "realized_kl": torch.zeros_like(full_kl),
            "full_correction_kl": full_kl,
        }

    feasible_full = full_kl <= float(epsilon_kl) + 1e-10
    low = torch.zeros_like(zero_rho)
    high = torch.ones_like(zero_rho)

    for _ in range(int(bisection_iterations)):
        mid = 0.5 * (low + high)
        candidate = _tilted_posterior(q, correction_energy, mid, mask)
        divergence = kl_p_to_q(candidate, q, mask)
        feasible = divergence <= float(epsilon_kl)
        low = torch.where(feasible, mid, low)
        high = torch.where(feasible, high, mid)

    rho = torch.where(feasible_full, one_rho, low)
    posterior = _tilted_posterior(q, correction_energy, rho, mask)
    realized = kl_p_to_q(posterior, q, mask)
    return {
        "posterior": posterior,
        "rho": rho,
        "realized_kl": realized,
        "full_correction_kl": full_kl,
    }


def pairwise_spatial_cost(
    first_coords: torch.Tensor,
    second_coords: torch.Tensor,
    *,
    spatial_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Dimensionless Euclidean ground cost."""

    if first_coords.ndim != 3 or second_coords.ndim != 3:
        raise ValueError("coordinates must be [batch,support,2]")
    if first_coords.shape[0] != second_coords.shape[0]:
        raise ValueError("coordinate batches differ")
    if first_coords.shape[2] != 2 or second_coords.shape[2] != 2:
        raise ValueError("V18.4 spatial coordinates must be 2-D")
    scale = torch.as_tensor(
        spatial_scale,
        dtype=first_coords.dtype,
        device=first_coords.device,
    ).clamp(min=1e-8)
    delta = (
        first_coords.unsqueeze(2)
        - second_coords.unsqueeze(1)
    )
    return torch.linalg.vector_norm(delta, dim=3) / scale


def _sinkhorn_regularized_cost(
    first_probability: torch.Tensor,
    second_probability: torch.Tensor,
    cost: torch.Tensor,
    first_mask: torch.Tensor,
    second_mask: torch.Tensor,
    *,
    epsilon_ot: float,
    iterations: int,
) -> torch.Tensor:
    """Balanced entropy-regularized OT objective in log domain."""

    if float(epsilon_ot) <= 0.0:
        raise ValueError("epsilon_ot must be positive")
    a = normalize_probability(first_probability, first_mask)
    b = normalize_probability(second_probability, second_mask)

    if cost.shape != (len(a), a.shape[1], b.shape[1]):
        raise ValueError("cost matrix shape mismatch")
    pair_mask = first_mask.unsqueeze(2) & second_mask.unsqueeze(1)

    # Keep every intermediate finite.  Using +/-inf for padded support is
    # algebraically convenient but creates undefined (-inf)-(-inf) branches in
    # autograd even when the corresponding forward entries are later masked.
    # A very negative finite sentinel is equivalent for the valid Sinkhorn
    # system because invalid pairs are explicitly zeroed in the transport plan.
    negative_sentinel = torch.as_tensor(
        -1.0e6,
        dtype=cost.dtype,
        device=cost.device,
    )
    log_a = torch.where(
        first_mask,
        a.clamp(min=NUMERICAL_EPS).log(),
        torch.zeros_like(a),
    )
    log_b = torch.where(
        second_mask,
        b.clamp(min=NUMERICAL_EPS).log(),
        torch.zeros_like(b),
    )
    log_kernel = torch.where(
        pair_mask,
        -cost / float(epsilon_ot),
        torch.full_like(cost, negative_sentinel),
    )

    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)

    for _ in range(int(iterations)):
        next_u = log_a - torch.logsumexp(
            log_kernel + log_v.unsqueeze(1),
            dim=2,
        )
        log_u = torch.where(
            first_mask,
            next_u,
            torch.zeros_like(next_u),
        )
        next_v = log_b - torch.logsumexp(
            log_kernel + log_u.unsqueeze(2),
            dim=1,
        )
        log_v = torch.where(
            second_mask,
            next_v,
            torch.zeros_like(next_v),
        )

    log_plan = (
        log_u.unsqueeze(2)
        + log_kernel
        + log_v.unsqueeze(1)
    )
    plan = torch.exp(log_plan)
    plan = torch.where(
        pair_mask,
        plan,
        torch.zeros_like(plan),
    )

    transport = (plan * cost).sum(dim=(1, 2))
    # Never form 0 * (-inf) on invalid padding.  torch.where evaluates
    # both branches before selection, so masking only the final product can
    # leave NaNs in autograd even when the forward value is finite.
    safe_log_plan = torch.where(
        pair_mask,
        log_plan,
        torch.zeros_like(log_plan),
    )
    entropy = torch.where(
        pair_mask,
        plan * (safe_log_plan - 1.0),
        torch.zeros_like(plan),
    ).sum(dim=(1, 2))
    return transport + float(epsilon_ot) * entropy


def spatial_sinkhorn_divergence(
    predicted_probability: torch.Tensor,
    predicted_coords: torch.Tensor,
    target_probability: torch.Tensor,
    target_coords: torch.Tensor,
    *,
    spatial_scale: torch.Tensor | float,
    predicted_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    epsilon_ot: float = 0.10,
    iterations: int = DEFAULT_SINKHORN_ITERATIONS,
) -> torch.Tensor:
    """Row-wise debiased entropic transport divergence."""

    if predicted_mask is None:
        predicted_mask = predicted_probability > 0.0
    if target_mask is None:
        target_mask = target_probability > 0.0
    predicted_mask = predicted_mask.bool()
    target_mask = target_mask.bool()

    cross_cost = pairwise_spatial_cost(
        predicted_coords,
        target_coords,
        spatial_scale=spatial_scale,
    )
    predicted_self_cost = pairwise_spatial_cost(
        predicted_coords,
        predicted_coords,
        spatial_scale=spatial_scale,
    )
    target_self_cost = pairwise_spatial_cost(
        target_coords,
        target_coords,
        spatial_scale=spatial_scale,
    )

    cross = _sinkhorn_regularized_cost(
        predicted_probability,
        target_probability,
        cross_cost,
        predicted_mask,
        target_mask,
        epsilon_ot=epsilon_ot,
        iterations=iterations,
    )
    self_predicted = _sinkhorn_regularized_cost(
        predicted_probability,
        predicted_probability,
        predicted_self_cost,
        predicted_mask,
        predicted_mask,
        epsilon_ot=epsilon_ot,
        iterations=iterations,
    )
    self_target = _sinkhorn_regularized_cost(
        target_probability,
        target_probability,
        target_self_cost,
        target_mask,
        target_mask,
        epsilon_ot=epsilon_ot,
        iterations=iterations,
    )
    divergence = cross - 0.5 * self_predicted - 0.5 * self_target
    # The true Sinkhorn divergence is nonnegative.  Clamp only numerical
    # round-off below zero.
    return divergence.clamp(min=0.0)


def _weighted_geometric_median_row(
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_iter: int,
    tolerance: float,
) -> torch.Tensor:
    weights = weights / weights.sum().clamp(min=NUMERICAL_EPS)
    current = torch.sum(weights.unsqueeze(1) * points, dim=0)
    eps = max(float(tolerance) * 0.1, 1e-9)

    for _ in range(int(max_iter)):
        delta = points - current.unsqueeze(0)
        distance = torch.linalg.vector_norm(delta, dim=1)
        coincident = distance <= eps

        if bool(coincident.any()):
            w0 = weights[coincident].sum()
            nonzero = ~coincident
            if not bool(nonzero.any()):
                return current
            nz_distance = distance[nonzero].clamp(min=eps)
            nz_weights = weights[nonzero]
            residual = torch.sum(
                nz_weights.unsqueeze(1)
                * delta[nonzero]
                / nz_distance.unsqueeze(1),
                dim=0,
            )
            residual_norm = torch.linalg.vector_norm(residual)
            if float(residual_norm) <= float(w0) + float(tolerance):
                return current
            inverse = nz_weights / nz_distance
            weiszfeld = torch.sum(
                inverse.unsqueeze(1) * points[nonzero],
                dim=0,
            ) / inverse.sum().clamp(min=eps)
            eta = torch.clamp(
                w0 / residual_norm.clamp(min=eps),
                min=0.0,
                max=1.0,
            )
            updated = (1.0 - eta) * weiszfeld + eta * current
        else:
            inverse = weights / distance.clamp(min=eps)
            updated = torch.sum(
                inverse.unsqueeze(1) * points,
                dim=0,
            ) / inverse.sum().clamp(min=eps)

        movement = torch.linalg.vector_norm(updated - current)
        current = updated
        if float(movement) <= float(tolerance):
            break
    return current


def weighted_component_geometric_median(
    probability: torch.Tensor,
    component_locations: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    max_iter: int = 128,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Posterior-weighted geometric median of component centers."""

    if probability.ndim != 2:
        raise ValueError("probability must be [batch,support]")
    if (
        component_locations.ndim != 3
        or component_locations.shape[:2] != probability.shape
        or component_locations.shape[2] != 2
    ):
        raise ValueError("component locations must be [batch,support,2]")
    if mask is None:
        mask = probability > 0.0
    normalized = normalize_probability(probability, mask)
    result = []
    for row in range(len(normalized)):
        valid = mask[row].bool()
        result.append(
            _weighted_geometric_median_row(
                component_locations[row, valid],
                normalized[row, valid],
                max_iter=max_iter,
                tolerance=tolerance,
            )
        )
    return torch.stack(result, dim=0)


def tail_mass_diagnostics(
    probability: torch.Tensor,
    component_locations: torch.Tensor,
    *,
    target_coords: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    radii_m: tuple[float, ...] = (10.0, 20.0, 40.0),
) -> dict[str, torch.Tensor]:
    """Per-row truth/mode far mass and GM spatial leverage."""

    if mask is None:
        mask = probability > 0.0
    p = normalize_probability(probability, mask)
    gm = weighted_component_geometric_median(
        p,
        component_locations,
        mask=mask,
    )
    map_index = p.argmax(dim=1)
    rows = torch.arange(len(p), device=p.device)
    map_location = component_locations[rows, map_index]

    mode_distance = torch.linalg.vector_norm(
        component_locations - map_location.unsqueeze(1),
        dim=2,
    )
    result: dict[str, torch.Tensor] = {}
    for radius in radii_m:
        result[f"mode_far_mass_{int(radius)}m"] = torch.sum(
            p * (mode_distance > float(radius)).to(p.dtype),
            dim=1,
        )

    if target_coords is not None:
        truth_distance = torch.linalg.vector_norm(
            component_locations - target_coords.unsqueeze(1),
            dim=2,
        )
        for radius in radii_m:
            result[f"truth_far_mass_{int(radius)}m"] = torch.sum(
                p * (truth_distance > float(radius)).to(p.dtype),
                dim=1,
            )

    result["gm_spatial_leverage_m"] = torch.sum(
        p
        * torch.linalg.vector_norm(
            component_locations - gm.unsqueeze(1),
            dim=2,
        ),
        dim=1,
    )
    result["geometric_median"] = gm
    return result
