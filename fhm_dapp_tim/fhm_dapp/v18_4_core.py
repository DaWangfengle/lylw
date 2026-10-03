"""Mathematical core for V18.4.

The functions in this module are model-agnostic.  They implement the frozen
mathematical contracts for:

- masked posterior normalization;
- robust training-derived energy scaling;
- empirical-CDF reliability;
- KL-bounded exponential posterior correction;
- exact half-energy-distance spatial supervision;
- component-center weighted geometric median.

No function learns an evidence gate or uses evaluation labels.
"""

from __future__ import annotations

import math

import torch


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
    value = torch.where(mask, probability, torch.zeros_like(probability))
    if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()):
        raise ValueError("valid probability must be finite and nonnegative")
    total = value.sum(dim=1, keepdim=True)
    if bool((total <= 0.0).any()):
        raise ValueError("every probability row needs positive valid mass")
    return value / total


def temperature_base_posterior(
    probability: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Calibrate an already normalized learned base posterior before tilting."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    normalize_probability(probability, mask)  # Validate only; log normalization is shift-invariant.
    if float(temperature) == 1.0:
        return torch.where(mask, probability, torch.zeros_like(probability))
    log_q = torch.where(
        probability > 0.0,
        probability.clamp(min=torch.finfo(probability.dtype).tiny).log(),
        torch.full_like(probability, float("-inf")),
    )
    return masked_softmax(log_q / float(temperature), mask)


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
        torch.where(q > 0.0, q.clamp(min=torch.finfo(q.dtype).tiny).log(),
                    torch.full_like(q, float("-inf"))),
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
    log_p = p.clamp(min=torch.finfo(p.dtype).tiny).log()
    log_q = q.clamp(min=torch.finfo(q.dtype).tiny).log()
    term = torch.where(
        mask,
        p * (log_p - log_q),
        torch.zeros_like(p),
    )
    term = torch.where(mask & (p > 0.0) & (q == 0.0), torch.full_like(term, float("inf")), term)
    return term.sum(dim=1)


def trust_region_posterior_correction(
    base_probability: torch.Tensor,
    correction_energy: torch.Tensor,
    *,
    epsilon_kl: torch.Tensor | float,
    mask: torch.Tensor | None = None,
    bisection_iterations: int = DEFAULT_TRUST_BISECTION_ITERATIONS,
) -> dict[str, torch.Tensor]:
    """Maximum rho in [0,1] under a scalar or [batch] KL(p||q) budget.

    Bisection uses float64 arithmetic. Zero-budget rows preserve an already
    normalized input bitwise, including exact zero padding. Returned KL is
    evaluated on the actual returned posterior, after any dtype conversion.
    """
    if base_probability.shape != correction_energy.shape:
        raise ValueError("base probability and energy shape mismatch")
    if int(bisection_iterations) <= 0:
        raise ValueError("bisection_iterations must be positive")
    if mask is None:
        mask = torch.ones_like(base_probability, dtype=torch.bool)
    mask = mask.bool()
    q_input = torch.where(mask, base_probability, torch.zeros_like(base_probability))
    q = normalize_probability(q_input.double(), mask)
    energy = torch.where(mask, correction_energy, torch.zeros_like(correction_energy)).double()
    if not bool(torch.isfinite(energy).all()):
        raise ValueError("valid correction energy must be finite")
    budget = torch.as_tensor(epsilon_kl, device=q.device, dtype=torch.float64)
    if budget.ndim == 0:
        budget = budget.expand(len(q))
    elif budget.shape != (len(q),):
        raise ValueError("epsilon_kl must be scalar or one value per row")
    if not bool(torch.isfinite(budget).all()) or bool((budget < 0.0).any()):
        raise ValueError("epsilon_kl must be finite and nonnegative")

    zero = torch.zeros(len(q), device=q.device, dtype=q.dtype)
    one = torch.ones_like(zero)
    full = _tilted_posterior(q, energy, one, mask)
    full_kl = kl_p_to_q(full, q, mask).clamp(min=0.0)
    low, high = zero, one
    for _ in range(int(bisection_iterations)):
        mid = 0.5 * (low + high)
        candidate = _tilted_posterior(q, energy, mid, mask)
        feasible = kl_p_to_q(candidate, q, mask) <= budget
        low = torch.where(feasible, mid, low)
        high = torch.where(feasible, high, mid)
    rho = torch.where(full_kl <= budget, one, low)
    rho = torch.where(budget == 0.0, zero, rho)
    posterior = _tilted_posterior(q, energy, rho, mask).to(base_probability.dtype)
    near_unit_mass = (q_input.double().sum(dim=1) - 1.0).abs() <= (
        4.0 * torch.finfo(base_probability.dtype).eps
    )
    identity = torch.where(near_unit_mass.unsqueeze(1), q_input, q.to(base_probability.dtype))
    posterior = torch.where((budget == 0.0).unsqueeze(1), identity, posterior)
    realized = kl_p_to_q(posterior.double(), q, mask).clamp(min=0.0)
    realized = torch.where(budget == 0.0, zero, realized)
    return {
        "posterior": posterior,
        "rho": rho,
        "realized_kl": realized,
        "full_correction_kl": full_kl,
        "epsilon_kl_max": budget.max(),
        "epsilon_kl_effective": budget,
    }


def reliability_bounded_posterior_correction(
    base_probability: torch.Tensor,
    radio_energy: torch.Tensor,
    geometry_energy: torch.Tensor,
    *,
    radio_reliability: torch.Tensor,
    geometry_reliability: torch.Tensor,
    epsilon_kl: float,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Separate empirical evidence direction from reliability-scaled budget."""
    if radio_energy.shape != base_probability.shape or geometry_energy.shape != base_probability.shape:
        raise ValueError("source energy and posterior shapes differ")
    if not math.isfinite(float(epsilon_kl)) or float(epsilon_kl) < 0.0:
        raise ValueError("maximum epsilon_kl must be finite and nonnegative")
    rr = radio_reliability.to(base_probability)
    rg = geometry_reliability.to(base_probability)
    for reliability in (rr, rg):
        if reliability.shape != (len(base_probability),):
            raise ValueError("reliability must have one value per row")
        if not bool(torch.isfinite(reliability).all()) or bool(((reliability < 0.0) | (reliability > 1.0)).any()):
            raise ValueError("reliability must be finite and in [0,1]")
    total = rr + rg
    active = total > NUMERICAL_EPS
    denominator = total.clamp(min=NUMERICAL_EPS)
    wr = torch.where(active, rr / denominator, torch.zeros_like(rr))
    wg = torch.where(active, rg / denominator, torch.zeros_like(rg))
    reliability = torch.where(active, torch.maximum(rr, rg), torch.zeros_like(rr))
    if mask is None:
        mask = torch.ones_like(base_probability, dtype=torch.bool)
    radio = torch.where(mask, radio_energy, torch.zeros_like(radio_energy))
    geometry = torch.where(mask, geometry_energy, torch.zeros_like(geometry_energy))
    direction = wr.unsqueeze(1) * radio + wg.unsqueeze(1) * geometry
    effective = reliability.double() * float(epsilon_kl)
    corrected = trust_region_posterior_correction(
        base_probability, direction, epsilon_kl=effective, mask=mask,
    )
    return {
        **corrected,
        "epsilon_kl_max": effective.new_tensor(float(epsilon_kl)),
        "trust_reliability": reliability,
        "radio_reliability": rr,
        "geometry_reliability": rg,
        "radio_direction_weight": wr,
        "geometry_direction_weight": wg,
        "combined_energy": direction,
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


def spatial_energy_distance(
    predicted_probability: torch.Tensor,
    predicted_coords: torch.Tensor,
    target_probability: torch.Tensor,
    target_coords: torch.Tensor,
    *,
    spatial_scale: torch.Tensor | float,
    predicted_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact row-wise half energy distance on dimensionless discrete supports.

    D = E_qt[d] - 0.5 E_qq[d] - 0.5 E_tt[d], d=Euclidean/s_C.
    Float64 accumulation limits cancellation; no solver, entropy coefficient,
    or nonnegativity clamp changes the supplied discrete objective. Padding is
    removed before normalization and distance computation, even if it is NaN.
    """
    if predicted_mask is None:
        predicted_mask = predicted_probability > 0.0
    if target_mask is None:
        target_mask = target_probability > 0.0
    predicted_mask, target_mask = predicted_mask.bool(), target_mask.bool()
    q = normalize_probability(predicted_probability.double(), predicted_mask)
    t = normalize_probability(target_probability.double(), target_mask)
    if predicted_coords.shape != (*q.shape, 2) or target_coords.shape != (*t.shape, 2):
        raise ValueError("coordinate and probability support shapes differ")
    x = torch.where(predicted_mask.unsqueeze(2), predicted_coords.double(), 0.0)
    y = torch.where(target_mask.unsqueeze(2), target_coords.double(), 0.0)
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("valid spatial coordinates must be finite")
    scale = torch.as_tensor(spatial_scale, device=x.device, dtype=torch.float64)
    if scale.ndim != 0 or not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
        raise ValueError("spatial_scale must be a finite positive scalar")
    cross = pairwise_spatial_cost(x, y, spatial_scale=scale)
    self_q = pairwise_spatial_cost(x, x, spatial_scale=scale)
    self_t = pairwise_spatial_cost(y, y, spatial_scale=scale)
    return (
        (q.unsqueeze(2) * t.unsqueeze(1) * cross).sum(dim=(1, 2))
        - 0.5 * (q.unsqueeze(2) * q.unsqueeze(1) * self_q).sum(dim=(1, 2))
        - 0.5 * (t.unsqueeze(2) * t.unsqueeze(1) * self_t).sum(dim=(1, 2))
    )


def cap_anisotropic_residual_transform(
    transform: torch.Tensor,
    *,
    max_residual_m: float,
) -> torch.Tensor:
    """Preserve local axes/anisotropy while capping the operator norm."""
    if transform.shape[-2:] != (2, 2) or not bool(torch.isfinite(transform).all()):
        raise ValueError("residual transforms must be finite [...,2,2] matrices")
    bound = float(max_residual_m)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("max_residual_m must be finite and positive")
    largest = torch.linalg.svdvals(transform.double()).amax(dim=-1)
    multiplier = torch.clamp(bound / largest.clamp(min=torch.finfo(torch.float64).tiny), max=1.0)
    return transform * multiplier.to(transform).unsqueeze(-1).unsqueeze(-1)


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
