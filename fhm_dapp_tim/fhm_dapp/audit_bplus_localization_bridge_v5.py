"""Development-only bridge from B+ measurement structure to localization.

The audit reuses one already-trained raw M0 member in each registered v4.4
coordinate fold.  It does not retrain the front end and it never evaluates the
registered outer-test, confirmation-coordinate, or confirmation-phone roles.

For each fold, the categorical front distribution on measured anchors is
extended to the registered development-coordinate geometry by interpolation
of log probabilities.  On that common candidate support, the audit compares

1. the transported front distribution;
2. one KL update with a nearest-coordinate marked-hurdle field;
3. one KL update with a continuous marked-hurdle field; and
4. the same continuous field with one scan-level detection random effect.

The update temperature, and the random-effect scale when applicable, are
chosen for every held-out fold using the other folds only.  This is a bridge
audit: a positive result licenses a later same-budget sparse-front experiment;
it is not a final model-selection or confirmation artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords, resolve_path
from fhm_dapp_tim.fhm_dapp.audit_sparse_radiomap_v5 import (
    LocationKey,
    _global_detection_background,
    _natural_parameter_predictions,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.device_robust_v4 import (
    VISIBLE_THRESHOLD,
    load_device_robust_checkpoint,
)
from fhm_dapp_tim.fhm_dapp.protocol_v4 import (
    coordinate_axis_partition,
    validate_registry,
)


def _csv_tuple(text: str, cast) -> tuple:
    values = tuple(cast(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _parameter_label(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _summary(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not np.isfinite(vector).all():
        raise ValueError("summary requires one finite non-empty vector")
    count = max(1, int(math.ceil(0.10 * len(vector))))
    return {
        "mean": float(vector.mean()),
        "standard_deviation": float(vector.std(ddof=0)),
        "median": float(np.median(vector)),
        "p90": float(np.quantile(vector, 0.90)),
        "p95": float(np.quantile(vector, 0.95)),
        "cvar90": float(np.sort(vector)[-count:].mean()),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
    }


def candidate_transport_matrix(
    anchor_keys: list[LocationKey],
    candidate_keys: list[LocationKey],
    *,
    neighbors: int,
    inverse_distance_power: float,
) -> np.ndarray:
    """Return a geometry-only interpolation operator from anchors to candidates."""

    if int(neighbors) <= 0:
        raise ValueError("neighbors must be positive")
    if not math.isfinite(float(inverse_distance_power)) or inverse_distance_power <= 0:
        raise ValueError("inverse_distance_power must be finite and positive")
    if len(set(anchor_keys)) != len(anchor_keys):
        raise ValueError("anchor keys must be unique")
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("candidate keys must be unique")
    matrix = np.zeros((len(candidate_keys), len(anchor_keys)), dtype=np.float64)
    anchor_xy = np.asarray([[key[2], key[3]] for key in anchor_keys], dtype=np.float64)
    strata: dict[tuple[int, int], np.ndarray] = {}
    for stratum in sorted({key[:2] for key in anchor_keys}):
        strata[stratum] = np.asarray(
            [index for index, key in enumerate(anchor_keys) if key[:2] == stratum],
            dtype=np.int64,
        )
    anchor_index = {key: index for index, key in enumerate(anchor_keys)}
    for row, key in enumerate(candidate_keys):
        exact = anchor_index.get(key)
        if exact is not None:
            matrix[row, exact] = 1.0
            continue
        local = strata.get(key[:2])
        if local is None or len(local) == 0:
            raise ValueError(f"candidate stratum {key[:2]} has no measured anchor")
        distance = np.linalg.norm(
            anchor_xy[local] - np.asarray(key[2:4], dtype=np.float64)[None, :],
            axis=1,
        )
        order = np.argsort(distance, kind="stable")[: min(int(neighbors), len(local))]
        chosen = local[order]
        weight = 1.0 / np.maximum(distance[order], 1e-6) ** float(
            inverse_distance_power
        )
        matrix[row, chosen] = weight / weight.sum()
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("candidate transport rows are not normalized")
    return matrix.astype(np.float32)


def _ap_fallbacks(
    center: np.ndarray,
    scale: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    aps = center.shape[1]
    fallback_center = np.full(aps, -1.0, dtype=np.float64)
    fallback_scale = np.full(aps, 0.15, dtype=np.float64)
    for ap in range(aps):
        valid = available[:, ap] & np.isfinite(center[:, ap])
        if np.any(valid):
            fallback_center[ap] = float(np.median(center[valid, ap]))
            valid_scale = scale[valid, ap]
            valid_scale = valid_scale[np.isfinite(valid_scale) & (valid_scale > 0.0)]
            if len(valid_scale):
                fallback_scale[ap] = float(np.median(valid_scale))
    return fallback_center, np.clip(fallback_scale, 0.05, 0.50)


def build_candidate_fields(
    *,
    rss: torch.Tensor,
    phone_ids: np.ndarray,
    coordinate_keys: list[LocationKey],
    training_indices: np.ndarray,
    anchor_keys: list[LocationKey],
    radio_memory: dict,
    candidate_keys: list[LocationKey],
    neighbors: int,
    inverse_distance_power: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    """Construct nearest and continuous two-field memories on one candidate grid."""

    raw_probability = radio_memory["detection_prob"].to(torch.float64)
    device_count = radio_memory["device_counts"].to(torch.float64)
    background = _global_detection_background(
        rss, phone_ids, coordinate_keys, training_indices
    ).to(torch.float64)
    probability = (
        raw_probability * device_count[:, None] + background[None, :]
    ) / (device_count[:, None] + 1.0)
    probability_np = probability.clamp(1e-5, 1.0 - 1e-5).numpy()
    center = radio_memory["rss_center"].to(torch.float64).numpy()
    scale = radio_memory["rss_scale"].to(torch.float64).numpy()
    available = raw_probability.numpy() > 0.0
    fallback_center, fallback_scale = _ap_fallbacks(center, scale, available)
    prediction = _natural_parameter_predictions(
        candidate_keys,
        anchor_keys,
        probability_np,
        center,
        available,
        neighbors=int(neighbors),
        inverse_distance_power=float(inverse_distance_power),
    )
    result: dict[str, dict[str, torch.Tensor]] = {}
    fallback_counts = {}
    for source_name, output_name in (("nearest", "nearest"), ("smooth", "continuous")):
        candidate_center = prediction[f"{source_name}_center"].copy()
        missing = ~np.isfinite(candidate_center)
        candidate_center[missing] = np.broadcast_to(
            fallback_center[None, :], candidate_center.shape
        )[missing]
        fallback_counts[output_name] = {
            "candidate_ap_cells": int(missing.size),
            "fallback_cells": int(missing.sum()),
            "fallback_fraction": float(missing.mean()),
        }
        result[output_name] = {
            "detection_probability": torch.from_numpy(
                prediction[f"{source_name}_probability"].astype(np.float32)
            ),
            "rss_center": torch.from_numpy(candidate_center.astype(np.float32)),
            "rss_scale": torch.from_numpy(fallback_scale.astype(np.float32)),
        }
    return result, {
        "conditional_center_fallback": fallback_counts,
        "scale_estimator": "AP-wise median support scale, fixed over space",
        "probability_shrinkage": "one equal-phone global-background pseudo-device",
    }


def marked_hurdle_loss(
    query: torch.Tensor,
    detection_probability: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    *,
    degrees_freedom: float,
    scan_effect_tau: float,
    quadrature_order: int,
    candidate_chunk_size: int,
) -> torch.Tensor:
    """Return fixed-dictionary joint NLL/J for the marked-hurdle field."""

    if query.ndim != 2 or not query.is_floating_point():
        raise ValueError("query must be floating [batch,aps]")
    if detection_probability.shape != rss_center.shape:
        raise ValueError("detection probability and RSS center shapes differ")
    if detection_probability.shape[1] != query.shape[1]:
        raise ValueError("query and field AP dimensions differ")
    if rss_scale.shape != (query.shape[1],):
        raise ValueError("rss_scale must contain one spatially fixed value per AP")
    nu = float(degrees_freedom)
    tau = float(scan_effect_tau)
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError("degrees_freedom must be finite and positive")
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("scan_effect_tau must be finite and nonnegative")
    if int(candidate_chunk_size) <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    if tau > 0.0 and int(quadrature_order) < 3:
        raise ValueError("quadrature_order must be at least three when tau is positive")

    visible = query > VISIBLE_THRESHOLD
    visible_float = visible.to(dtype=query.dtype).unsqueeze(1)
    hidden_float = (~visible).to(dtype=query.dtype).unsqueeze(1)
    scale = rss_scale.to(device=query.device, dtype=query.dtype).clamp_min(1e-4)
    amplitude_constant = (
        math.lgamma(0.5 * nu)
        + 0.5 * math.log(nu * math.pi)
        - math.lgamma(0.5 * (nu + 1.0))
    )
    if tau > 0.0:
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(int(quadrature_order))
        nodes = torch.as_tensor(
            math.sqrt(2.0) * tau * nodes_np,
            device=query.device,
            dtype=query.dtype,
        )
        log_weights = torch.as_tensor(
            np.log(weights_np) - 0.5 * math.log(math.pi),
            device=query.device,
            dtype=query.dtype,
        )
    outputs = []
    candidates = len(detection_probability)
    for start in range(0, candidates, int(candidate_chunk_size)):
        stop = min(start + int(candidate_chunk_size), candidates)
        probability = detection_probability[start:stop].to(
            device=query.device, dtype=query.dtype
        ).clamp(1e-6, 1.0 - 1e-6)
        eta = torch.logit(probability)
        if tau == 0.0:
            detection_log_likelihood = (
                visible_float * F.logsigmoid(eta).unsqueeze(0)
                + hidden_float * F.logsigmoid(-eta).unsqueeze(0)
            ).sum(dim=2)
        else:
            quadrature_terms = []
            for node, log_weight in zip(nodes, log_weights):
                shifted = eta + node
                conditional_log_likelihood = (
                    visible_float * F.logsigmoid(shifted).unsqueeze(0)
                    + hidden_float * F.logsigmoid(-shifted).unsqueeze(0)
                ).sum(dim=2)
                quadrature_terms.append(conditional_log_likelihood + log_weight)
            detection_log_likelihood = torch.logsumexp(
                torch.stack(quadrature_terms, dim=0), dim=0
            )
        center = rss_center[start:stop].to(device=query.device, dtype=query.dtype)
        standardized = (
            query.unsqueeze(1) - center.unsqueeze(0)
        ) / scale.view(1, 1, -1)
        amplitude_nll = (
            visible_float
            * (
                amplitude_constant
                + torch.log(scale).view(1, 1, -1)
                + 0.5 * (nu + 1.0) * torch.log1p(standardized.square() / nu)
            )
        ).sum(dim=2)
        outputs.append(
            (-detection_log_likelihood + amplitude_nll) / float(query.shape[1])
        )
    loss = torch.cat(outputs, dim=1)
    if not bool(torch.isfinite(loss).all()):
        raise RuntimeError("marked-hurdle loss contains non-finite values")
    return loss


def _distribution_rows(
    probability: torch.Tensor,
    support_xy: torch.Tensor,
    support_building: torch.Tensor,
    support_floor: torch.Tensor,
    target_xy: torch.Tensor,
    target_building: torch.Tensor,
    target_floor: torch.Tensor,
    *,
    pair_distance_3d: torch.Tensor,
    floor_height_m: float,
    true_candidate: torch.Tensor | None,
    credible_levels: tuple[float, ...],
) -> dict[str, np.ndarray]:
    if probability.ndim != 2 or probability.shape[1] != len(support_xy):
        raise ValueError("probability and support shapes disagree")
    probability = probability / probability.sum(dim=1, keepdim=True)
    prediction = probability @ support_xy
    error = torch.linalg.vector_norm(prediction - target_xy, dim=1)
    support_3d = torch.cat(
        (
            support_xy,
            float(floor_height_m) * support_floor.to(support_xy.dtype).unsqueeze(1),
        ),
        dim=1,
    )
    target_3d = torch.cat(
        (
            target_xy,
            float(floor_height_m) * target_floor.to(target_xy.dtype).unsqueeze(1),
        ),
        dim=1,
    )
    first = torch.sum(
        probability * torch.cdist(target_3d, support_3d, p=2), dim=1
    )
    second = 0.5 * torch.sum((probability @ pair_distance_3d) * probability, dim=1)
    centered = support_xy.unsqueeze(0) - prediction.unsqueeze(1)
    trace = torch.sum(probability * centered.square().sum(dim=2), dim=1)
    floor_probability = torch.zeros(
        (len(probability), int(support_floor.max()) + 1),
        device=probability.device,
        dtype=probability.dtype,
    )
    floor_probability.scatter_add_(
        1, support_floor.unsqueeze(0).expand(len(probability), -1), probability
    )
    building_probability = torch.zeros(
        (len(probability), int(support_building.max()) + 1),
        device=probability.device,
        dtype=probability.dtype,
    )
    building_probability.scatter_add_(
        1, support_building.unsqueeze(0).expand(len(probability), -1), probability
    )
    result = {
        "error_2d_m": error.detach().cpu().numpy().astype(np.float64),
        "energy_score_3d_m": (first - second).detach().cpu().numpy().astype(np.float64),
        "covariance_trace_m2": trace.detach().cpu().numpy().astype(np.float64),
        "floor_correct": (
            floor_probability.argmax(dim=1) == target_floor
        ).detach().cpu().numpy().astype(np.float64),
        "building_correct": (
            building_probability.argmax(dim=1) == target_building
        ).detach().cpu().numpy().astype(np.float64),
    }
    if true_candidate is not None:
        selected = probability.gather(1, true_candidate.unsqueeze(1)).squeeze(1)
        result["categorical_nll_nats"] = (
            -torch.log(selected.clamp_min(1e-30))
        ).detach().cpu().numpy().astype(np.float64)
    radial_distance = torch.linalg.vector_norm(centered, dim=2)
    order = torch.argsort(radial_distance, dim=1, stable=True)
    ordered_distance = torch.gather(radial_distance, 1, order)
    cumulative = torch.cumsum(torch.gather(probability, 1, order), dim=1)
    for level in credible_levels:
        threshold = torch.full(
            (len(probability), 1),
            float(level),
            device=probability.device,
            dtype=probability.dtype,
        )
        index = torch.searchsorted(cumulative.contiguous(), threshold).squeeze(1)
        index = index.clamp(max=probability.shape[1] - 1)
        radius = ordered_distance.gather(1, index.unsqueeze(1)).squeeze(1)
        result[f"coverage_{int(round(100 * level))}"] = (
            error <= radius + 1e-7
        ).detach().cpu().numpy().astype(np.float64)
        result[f"radius_{int(round(100 * level))}_m"] = (
            radius.detach().cpu().numpy().astype(np.float64)
        )
    return result


def _append_rows(target: dict[str, list[np.ndarray]], values: dict[str, np.ndarray]) -> None:
    for name, vector in values.items():
        target.setdefault(name, []).append(np.asarray(vector, dtype=np.float64))


def _finish_rows(values: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(parts) for name, parts in values.items()}


def _group_metric(
    values: np.ndarray,
    row_indices: np.ndarray,
    coordinate_keys: list[LocationKey],
    phone_ids: np.ndarray,
) -> dict:
    vector = np.asarray(values, dtype=np.float64)
    rows = np.asarray(row_indices, dtype=np.int64)
    if vector.shape != (len(rows),) or not np.isfinite(vector).all():
        raise ValueError("metric must contain one finite value per evaluation row")
    coordinate_groups: dict[LocationKey, list[float]] = {}
    phone_groups: dict[int, dict[LocationKey, list[float]]] = {}
    for local, row in enumerate(rows):
        key = coordinate_keys[int(row)]
        phone = int(phone_ids[int(row)])
        coordinate_groups.setdefault(key, []).append(float(vector[local]))
        phone_groups.setdefault(phone, {}).setdefault(key, []).append(
            float(vector[local])
        )
    coordinate_values = np.asarray(
        [np.mean(group) for group in coordinate_groups.values()], dtype=np.float64
    )
    per_phone = {
        str(phone): float(
            np.mean([np.mean(group) for group in coordinate_group.values()])
        )
        for phone, coordinate_group in sorted(phone_groups.items())
    }
    phone_values = np.asarray(list(per_phone.values()), dtype=np.float64)
    return {
        "row": _summary(vector),
        "coordinate_macro": _summary(coordinate_values),
        "equal_phone_coordinate_macro_mean": float(phone_values.mean()),
        "per_phone_coordinate_macro_mean": per_phone,
        "represented_rows": int(len(rows)),
        "represented_coordinates": int(len(coordinate_values)),
        "represented_phones": int(len(phone_values)),
    }


def summarize_rows(
    values: dict[str, np.ndarray],
    row_indices: np.ndarray,
    coordinate_keys: list[LocationKey],
    phone_ids: np.ndarray,
) -> dict:
    return {
        name: _group_metric(vector, row_indices, coordinate_keys, phone_ids)
        for name, vector in values.items()
    }


def _candidate_arrays(
    candidate_keys: list[LocationKey], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[LocationKey, int]]:
    xy = torch.as_tensor(
        [[key[2], key[3]] for key in candidate_keys],
        device=device,
        dtype=torch.float32,
    )
    building = torch.as_tensor(
        [key[0] for key in candidate_keys], device=device, dtype=torch.long
    )
    floor = torch.as_tensor(
        [key[1] for key in candidate_keys], device=device, dtype=torch.long
    )
    return xy, building, floor, {key: i for i, key in enumerate(candidate_keys)}


@torch.no_grad()
def evaluate_fold(
    *,
    fold: int,
    checkpoint_path: Path,
    registry: dict,
    rss: torch.Tensor,
    raw_xy: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    phone_ids: np.ndarray,
    coordinate_keys: list[LocationKey],
    candidate_keys: list[LocationKey],
    lambda_grid: tuple[float, ...],
    tau_grid: tuple[float, ...],
    neighbors: int,
    inverse_distance_power: float,
    degrees_freedom: float,
    quadrature_order: int,
    batch_size: int,
    candidate_chunk_size: int,
    credible_levels: tuple[float, ...],
    device: torch.device,
) -> dict:
    model, checkpoint = load_device_robust_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    checks = {
        "selection_authority": checkpoint.get("selection_authority") is True,
        "registry": checkpoint.get("registry_sha256") == registry["registry_sha256"],
        "raw_M0": config.get("mechanism") == "M0",
        "coordinate_role": config.get("role_axis") == "coordinate",
        "fold": int(config.get("outer_fold", -1)) == int(fold),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"checkpoint failed bridge validation: {', '.join(failed)}")
    evaluation_rows = checkpoint["role_indices"]["structure_development"].cpu().numpy()
    forbidden_rows = np.sort(
        np.concatenate(
            (
                checkpoint["role_indices"]["outer_test"].cpu().numpy(),
                checkpoint["role_indices"]["device_confirmation_quarantine"].cpu().numpy(),
            )
        )
    )
    if len(np.intersect1d(evaluation_rows, forbidden_rows)):
        raise RuntimeError("bridge evaluation overlaps a forbidden role")
    training_indices = checkpoint["training_indices"].cpu().numpy()
    if len(np.intersect1d(training_indices, evaluation_rows)):
        raise RuntimeError("front training rows overlap bridge evaluation rows")
    anchor_keys = [
        (int(key[0]), int(key[1]), float(key[2]), float(key[3]))
        for key in checkpoint["anchor_coordinate_keys"]
    ]
    if len(anchor_keys) != model.n_anchors:
        raise ValueError("checkpoint anchor identities disagree with the model")
    transport = torch.from_numpy(
        candidate_transport_matrix(
            anchor_keys,
            candidate_keys,
            neighbors=neighbors,
            inverse_distance_power=inverse_distance_power,
        )
    ).to(device)
    fields_cpu, field_diagnostics = build_candidate_fields(
        rss=rss,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        training_indices=training_indices,
        anchor_keys=anchor_keys,
        radio_memory=checkpoint["radio_memory"],
        candidate_keys=candidate_keys,
        neighbors=neighbors,
        inverse_distance_power=inverse_distance_power,
    )
    fields = {
        name: {key: value.to(device) for key, value in memory.items()}
        for name, memory in fields_cpu.items()
    }
    origin = checkpoint["coord_origin"].to(device=device, dtype=torch.float32)
    candidate_xy, candidate_building, candidate_floor, candidate_index = (
        _candidate_arrays(candidate_keys, device)
    )
    # UJI coordinates are large projected map coordinates.  All spatial
    # moments and pairwise distances must be evaluated after translation to a
    # local origin; otherwise float32 cdist suffers catastrophic cancellation.
    candidate_xy = candidate_xy - origin.unsqueeze(0)
    candidate_support_3d = torch.cat(
        (
            candidate_xy,
            float(config["floor_height_m"])
            * candidate_floor.to(candidate_xy.dtype).unsqueeze(1),
        ),
        dim=1,
    )
    candidate_pair = torch.cdist(candidate_support_3d, candidate_support_3d, p=2)
    anchor_xy = model.anchor_coords.to(device)
    anchor_building = model.anchor_buildings.to(device)
    anchor_floor = model.anchor_floors.to(device)
    anchor_support_3d = torch.cat(
        (
            anchor_xy,
            float(config["floor_height_m"])
            * anchor_floor.to(anchor_xy.dtype).unsqueeze(1),
        ),
        dim=1,
    )
    anchor_pair = torch.cdist(anchor_support_3d, anchor_support_3d, p=2)
    output_parts: dict[str, dict[str, list[np.ndarray]]] = {}
    configurations: dict[str, dict] = {
        "front_anchor": {"variant": "front_anchor"},
        "front_grid": {"variant": "front_grid"},
    }
    for value in lambda_grid:
        label = _parameter_label(value)
        configurations[f"nearest__lambda_{label}"] = {
            "variant": "nearest",
            "lambda": float(value),
        }
        configurations[f"continuous__lambda_{label}"] = {
            "variant": "continuous",
            "lambda": float(value),
        }
        for tau in tau_grid:
            tau_label = _parameter_label(tau)
            configurations[f"continuous_u__tau_{tau_label}__lambda_{label}"] = {
                "variant": "continuous_u",
                "lambda": float(value),
                "tau": float(tau),
            }
    for name in configurations:
        output_parts[name] = {}

    rss_device = rss.to(device)
    for start in range(0, len(evaluation_rows), int(batch_size)):
        rows = evaluation_rows[start : start + int(batch_size)]
        query = rss_device[rows]
        logits = model.retrieval_logits(query)[0]
        anchor_probability = torch.softmax(logits, dim=1)
        candidate_log_prior = F.log_softmax(
            F.log_softmax(logits, dim=1) @ transport.t(), dim=1
        )
        candidate_probability = torch.exp(candidate_log_prior)
        target_xy = (
            torch.as_tensor(raw_xy[rows], device=device, dtype=torch.float32)
            - origin.unsqueeze(0)
        )
        target_building = torch.as_tensor(building[rows], device=device, dtype=torch.long)
        target_floor = torch.as_tensor(floor[rows], device=device, dtype=torch.long)
        true_candidate = torch.as_tensor(
            [candidate_index[coordinate_keys[int(row)]] for row in rows],
            device=device,
            dtype=torch.long,
        )
        _append_rows(
            output_parts["front_anchor"],
            _distribution_rows(
                anchor_probability,
                anchor_xy,
                anchor_building,
                anchor_floor,
                target_xy,
                target_building,
                target_floor,
                pair_distance_3d=anchor_pair,
                floor_height_m=float(config["floor_height_m"]),
                true_candidate=None,
                credible_levels=credible_levels,
            ),
        )
        _append_rows(
            output_parts["front_grid"],
            _distribution_rows(
                candidate_probability,
                candidate_xy,
                candidate_building,
                candidate_floor,
                target_xy,
                target_building,
                target_floor,
                pair_distance_3d=candidate_pair,
                floor_height_m=float(config["floor_height_m"]),
                true_candidate=true_candidate,
                credible_levels=credible_levels,
            ),
        )
        losses = {}
        for field_name in ("nearest", "continuous"):
            memory = fields[field_name]
            losses[field_name] = marked_hurdle_loss(
                query,
                memory["detection_probability"],
                memory["rss_center"],
                memory["rss_scale"],
                degrees_freedom=degrees_freedom,
                scan_effect_tau=0.0,
                quadrature_order=quadrature_order,
                candidate_chunk_size=candidate_chunk_size,
            )
        for tau in tau_grid:
            memory = fields["continuous"]
            losses[f"continuous_u_{_parameter_label(tau)}"] = marked_hurdle_loss(
                query,
                memory["detection_probability"],
                memory["rss_center"],
                memory["rss_scale"],
                degrees_freedom=degrees_freedom,
                scan_effect_tau=float(tau),
                quadrature_order=quadrature_order,
                candidate_chunk_size=candidate_chunk_size,
            )
        for name, specification in configurations.items():
            variant = specification["variant"]
            if variant in ("front_anchor", "front_grid"):
                continue
            loss_key = (
                variant
                if variant != "continuous_u"
                else f"continuous_u_{_parameter_label(specification['tau'])}"
            )
            log_probability = candidate_log_prior - losses[loss_key] / float(
                specification["lambda"]
            )
            probability = torch.softmax(log_probability, dim=1)
            _append_rows(
                output_parts[name],
                _distribution_rows(
                    probability,
                    candidate_xy,
                    candidate_building,
                    candidate_floor,
                    target_xy,
                    target_building,
                    target_floor,
                    pair_distance_3d=candidate_pair,
                    floor_height_m=float(config["floor_height_m"]),
                    true_candidate=true_candidate,
                    credible_levels=credible_levels,
                ),
            )
        processed = int(min(start + batch_size, len(evaluation_rows)))
        if start == 0 or processed == len(evaluation_rows) or (
            (start // int(batch_size)) + 1
        ) % 10 == 0:
            print(
                json.dumps(
                    {
                        "bridge_fold": int(fold),
                        "processed_rows": processed,
                        "total_rows": int(len(evaluation_rows)),
                    }
                ),
                flush=True,
            )
    row_outputs = {name: _finish_rows(parts) for name, parts in output_parts.items()}
    summaries = {
        name: summarize_rows(values, evaluation_rows, coordinate_keys, phone_ids)
        for name, values in row_outputs.items()
    }
    return {
        "fold": int(fold),
        "checkpoint": str(checkpoint_path),
        "evaluation_rows": evaluation_rows,
        "row_outputs": row_outputs,
        "summaries": summaries,
        "configurations": configurations,
        "anchors": int(model.n_anchors),
        "candidate_coordinates": int(len(candidate_keys)),
        "field_diagnostics": field_diagnostics,
    }


def _select_other_fold(
    folds: dict[int, dict],
    held_out_fold: int,
    variant: str,
) -> tuple[str, dict]:
    eligible = {
        name: specification
        for name, specification in folds[held_out_fold]["configurations"].items()
        if specification["variant"] == variant
    }
    if not eligible:
        raise ValueError(f"no configurations for variant {variant}")
    other_folds = [fold for fold in sorted(folds) if fold != held_out_fold]
    if not other_folds:
        other_folds = [held_out_fold]
    scores = {}
    for name in eligible:
        scores[name] = float(
            np.mean(
                [
                    folds[fold]["summaries"][name]["energy_score_3d_m"][
                        "coordinate_macro"
                    ]["mean"]
                    for fold in other_folds
                ]
            )
        )
    def tie_key(name: str) -> tuple[float, float, float]:
        specification = eligible[name]
        return (
            scores[name],
            float(specification.get("tau", 0.0)),
            -float(specification["lambda"]),
        )
    selected = min(eligible, key=tie_key)
    return selected, {
        "selected_configuration": selected,
        "selected_parameters": eligible[selected],
        "selection_folds": other_folds,
        "selection_metric": "mean coordinate-macro 3-D energy score",
        "selection_score": scores[selected],
        "all_selection_scores": scores,
    }


def _paired_coordinate_delta(
    left: np.ndarray,
    right: np.ndarray,
    rows: np.ndarray,
    coordinate_keys: list[LocationKey],
) -> dict:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    groups: dict[LocationKey, list[float]] = {}
    for local, row in enumerate(rows):
        groups.setdefault(coordinate_keys[int(row)], []).append(float(difference[local]))
    coordinate_delta = np.asarray(
        [np.mean(group) for group in groups.values()], dtype=np.float64
    )
    return {
        "left_minus_right_coordinate_delta": _summary(coordinate_delta),
        "fraction_left_better": float(np.mean(coordinate_delta < 0.0)),
        "fraction_left_worse": float(np.mean(coordinate_delta > 0.0)),
        "coordinates": int(len(coordinate_delta)),
    }


def run_bridge(
    *,
    train_data: str | Path,
    registry_path: str | Path,
    checkpoint_root: str | Path,
    folds: tuple[int, ...],
    seed: int,
    lambda_grid: tuple[float, ...],
    tau_grid: tuple[float, ...],
    neighbors: int,
    inverse_distance_power: float,
    degrees_freedom: float,
    quadrature_order: int,
    batch_size: int,
    candidate_chunk_size: int,
    credible_levels: tuple[float, ...],
    device_name: str,
) -> dict:
    data_path = resolve_path(str(train_data)).resolve()
    registry_file = resolve_path(str(registry_path)).resolve()
    checkpoint_directory = resolve_path(str(checkpoint_root)).resolve()
    registry = validate_registry(registry_file, train_data=data_path)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        _timestamp_t,
        stats,
    ) = load_dataset(data_path, device="cpu")
    raw_xy = raw_coords(coords_norm, building_t, stats)
    building = building_t.numpy()
    floor_values = floor_t.numpy()
    phone_ids = phone_t.numpy()
    coordinate_keys = location_keys(raw_xy, building, floor_values)
    development_phones = tuple(map(int, registry["device_development_phone_ids"]))
    partition = coordinate_axis_partition(
        coordinate_keys, phone_ids, development_phones
    )
    candidate_keys = sorted(partition["development_core"])
    fold_results = {}
    seen_evaluation_rows: set[int] = set()
    for fold in folds:
        checkpoint_path = (
            checkpoint_directory / f"fold{int(fold)}_seed{int(seed)}.pt"
        ).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        result = evaluate_fold(
            fold=int(fold),
            checkpoint_path=checkpoint_path,
            registry=registry,
            rss=rss,
            raw_xy=raw_xy,
            building=building,
            floor=floor_values,
            phone_ids=phone_ids,
            coordinate_keys=coordinate_keys,
            candidate_keys=candidate_keys,
            lambda_grid=lambda_grid,
            tau_grid=tau_grid,
            neighbors=neighbors,
            inverse_distance_power=inverse_distance_power,
            degrees_freedom=degrees_freedom,
            quadrature_order=quadrature_order,
            batch_size=batch_size,
            candidate_chunk_size=candidate_chunk_size,
            credible_levels=credible_levels,
            device=device,
        )
        overlap = seen_evaluation_rows.intersection(map(int, result["evaluation_rows"]))
        if overlap:
            raise RuntimeError("bridge folds have overlapping evaluation rows")
        seen_evaluation_rows.update(map(int, result["evaluation_rows"]))
        fold_results[int(fold)] = result

    selected_by_fold: dict[int, dict[str, str]] = {}
    selection_records: dict[int, dict] = {}
    pooled_rows = []
    pooled_outputs: dict[str, dict[str, list[np.ndarray]]] = {
        "front_anchor": {},
        "front_grid": {},
        "nearest": {},
        "continuous": {},
        "continuous_u": {},
    }
    for fold in sorted(fold_results):
        selected_by_fold[fold] = {
            "front_anchor": "front_anchor",
            "front_grid": "front_grid",
        }
        selection_records[fold] = {}
        for variant in ("nearest", "continuous", "continuous_u"):
            name, record = _select_other_fold(fold_results, fold, variant)
            selected_by_fold[fold][variant] = name
            selection_records[fold][variant] = record
        pooled_rows.append(fold_results[fold]["evaluation_rows"])
        for variant, name in selected_by_fold[fold].items():
            _append_rows(
                pooled_outputs[variant], fold_results[fold]["row_outputs"][name]
            )
    pooled_row_indices = np.concatenate(pooled_rows)
    finished_outputs = {
        variant: _finish_rows(parts) for variant, parts in pooled_outputs.items()
    }
    pooled_summary = {
        variant: summarize_rows(
            values, pooled_row_indices, coordinate_keys, phone_ids
        )
        for variant, values in finished_outputs.items()
    }
    paired = {}
    comparisons = (
        ("front_grid", "front_anchor"),
        ("nearest", "front_grid"),
        ("continuous", "nearest"),
        ("continuous_u", "continuous"),
    )
    for left, right in comparisons:
        paired[f"{left}_vs_{right}"] = {
            metric: _paired_coordinate_delta(
                finished_outputs[left][metric],
                finished_outputs[right][metric],
                pooled_row_indices,
                coordinate_keys,
            )
            for metric in ("error_2d_m", "energy_score_3d_m")
        }
    serializable_folds = {}
    for fold, result in sorted(fold_results.items()):
        serializable_folds[str(fold)] = {
            "checkpoint": result["checkpoint"],
            "anchors": result["anchors"],
            "candidate_coordinates": result["candidate_coordinates"],
            "evaluation_rows": int(len(result["evaluation_rows"])),
            "field_diagnostics": result["field_diagnostics"],
            "selected_configurations": selected_by_fold[fold],
            "selection": selection_records[fold],
            "selected_summaries": {
                variant: result["summaries"][name]
                for variant, name in selected_by_fold[fold].items()
            },
        }
    return {
        "artifact_status": "v5_bplus_development_localization_bridge",
        "selection_authority": False,
        "bridge_only": True,
        "protected_coordinate_confirmation_evaluated": False,
        "protected_confirmation_phones_evaluated": False,
        "registered_outer_test_outcomes_evaluated": False,
        "train_data": str(data_path),
        "registry": str(registry_file),
        "registry_sha256": registry["registry_sha256"],
        "checkpoint_root": str(checkpoint_directory),
        "device": str(device),
        "development_candidate_coordinates": int(len(candidate_keys)),
        "evaluation_rows": int(len(pooled_row_indices)),
        "evaluation_folds": list(map(int, folds)),
        "front_seed": int(seed),
        "design": {
            "front": "registered raw M0 categorical member; no retraining",
            "front_continuation": (
                "IDW interpolation of anchor log probabilities on the registered "
                "development-coordinate geometry"
            ),
            "measurement_law": (
                "separate Bernoulli occurrence and univariate Student-t visible mark"
            ),
            "nearest_control": "nearest-coordinate occurrence and mark fields",
            "continuous_field": "IDW of Bernoulli logits and conditional RSS centers",
            "scan_effect": "one Gaussian scan intercept integrated by Gauss-Hermite",
            "lambda_grid": list(map(float, lambda_grid)),
            "tau_grid": list(map(float, tau_grid)),
            "hyperparameter_selection": (
                "each fold selected using mean coordinate-macro energy score on all "
                "other included folds"
            ),
            "neighbors": int(neighbors),
            "inverse_distance_power": float(inverse_distance_power),
            "student_degrees_freedom": float(degrees_freedom),
            "quadrature_order": int(quadrature_order),
            "credible_levels": list(map(float, credible_levels)),
        },
        "pooled_selected_summary": pooled_summary,
        "paired_coordinate_comparisons": paired,
        "folds": serializable_folds,
        "interpretation_contract": {
            "positive_bridge": (
                "The continuous field must improve paired coordinate localization or "
                "energy score over the nearest field without a material calibration loss."
            ),
            "scan_effect_admission": (
                "The scan effect is retained only if it improves cross-fitted energy "
                "or localization and does not concentrate harm in specific phones."
            ),
            "scope": (
                "A positive bridge licenses same-budget sparse-front retraining.  It "
                "does not select a final model or validate 25/50/75 percent budgets."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge B+ continuous measurement fields to localization"
    )
    parser.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv"
    )
    parser.add_argument(
        "--registry", default="evidence/v44/registry_v4_4.json"
    )
    parser.add_argument(
        "--checkpoint-root", default="models/v44/mechanism/coordinate/M0"
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--lambda-grid", default="0.25,0.5,1,2,4,8")
    parser.add_argument("--tau-grid", default="0.4,0.8,1.2")
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--inverse-distance-power", type=float, default=2.0)
    parser.add_argument("--student-df", type=float, default=4.0)
    parser.add_argument("--quadrature-order", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-chunk-size", type=int, default=64)
    parser.add_argument("--credible-levels", default="0.5,0.8,0.9,0.95")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    folds = _csv_tuple(args.folds, int)
    if len(set(folds)) != len(folds) or any(fold not in range(5) for fold in folds):
        raise ValueError("--folds must contain unique registered folds in 0..4")
    lambda_grid = _csv_tuple(args.lambda_grid, float)
    tau_grid = _csv_tuple(args.tau_grid, float)
    credible_levels = _csv_tuple(args.credible_levels, float)
    if any(not math.isfinite(value) or value <= 0.0 for value in lambda_grid):
        raise ValueError("all lambda values must be finite and positive")
    if any(not math.isfinite(value) or value <= 0.0 for value in tau_grid):
        raise ValueError("all tau values must be finite and positive")
    if any(not 0.0 < value < 1.0 for value in credible_levels):
        raise ValueError("credible levels must lie strictly inside (0,1)")
    payload = run_bridge(
        train_data=args.train_data,
        registry_path=args.registry,
        checkpoint_root=args.checkpoint_root,
        folds=folds,
        seed=args.seed,
        lambda_grid=lambda_grid,
        tau_grid=tau_grid,
        neighbors=args.neighbors,
        inverse_distance_power=args.inverse_distance_power,
        degrees_freedom=args.student_df,
        quadrature_order=args.quadrature_order,
        batch_size=args.batch_size,
        candidate_chunk_size=args.candidate_chunk_size,
        credible_levels=credible_levels,
        device_name=args.device,
    )
    output = resolve_path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bridge artifact: {output}")
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact_status": payload["artifact_status"],
                "output": str(output),
                "evaluation_rows": payload["evaluation_rows"],
                "paired_coordinate_comparisons": payload[
                    "paired_coordinate_comparisons"
                ],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
