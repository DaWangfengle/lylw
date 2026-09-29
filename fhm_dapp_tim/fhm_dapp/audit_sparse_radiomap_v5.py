"""Development-only feasibility audit for a sparse continuous radio map.

This script is deliberately not a localization experiment.  It geometrically
thins the v4.4 development-coordinate core, builds the equal-phone radio map
only at retained coordinates, and predicts measurement distributions at the
unmeasured development coordinates.  Retained coordinates are selected by
farthest-point sampling within each building/floor stratum without consulting
RSS values or localization outcomes.

Two estimators are compared:

* a piecewise-constant nearest-reference-point field; and
* inverse-distance interpolation of Bernoulli natural parameters and visible
  RSS centers.

The audit asks whether the continuous-field premise has predictive signal.  It
must not be used to select a final localization model or to inspect any of the
protected v4.4 coordinate/phone confirmation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords, resolve_path
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.device_robust_v4 import (
    build_device_balanced_anchor_memory,
)
from fhm_dapp_tim.fhm_dapp.protocol_v4 import (
    coordinate_axis_partition,
    validate_registry,
)


LocationKey = tuple[int, int, float, float]


def _summary(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not np.isfinite(vector).all():
        raise ValueError("summary requires one finite non-empty vector")
    return {
        "mean": float(vector.mean()),
        "standard_deviation": float(vector.std(ddof=0)),
        "q10": float(np.quantile(vector, 0.10)),
        "median": float(np.quantile(vector, 0.50)),
        "q90": float(np.quantile(vector, 0.90)),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
    }


def _stable_rank(seed: int, salt: str, key: LocationKey) -> bytes:
    return hashlib.sha256(
        f"{int(seed)}|{salt}|{repr(key)}".encode("utf-8")
    ).digest()


def _farthest_point_sample(
    keys: list[LocationKey],
    count: int,
    *,
    seed: int,
) -> set[LocationKey]:
    """Choose a geometry-only space-filling subset with deterministic ties."""

    if not keys:
        raise ValueError("cannot sample an empty coordinate stratum")
    target = int(count)
    if not 1 <= target <= len(keys):
        raise ValueError("sample count is outside the coordinate stratum")
    ordered = sorted(keys)
    if target == len(ordered):
        return set(ordered)
    xy = np.asarray([[key[2], key[3]] for key in ordered], dtype=np.float64)
    first = min(
        range(len(ordered)),
        key=lambda index: _stable_rank(seed, "fps-start", ordered[index]),
    )
    selected = [first]
    minimum_squared_distance = np.sum((xy - xy[first]) ** 2, axis=1)
    minimum_squared_distance[first] = -np.inf
    while len(selected) < target:
        maximum = float(np.max(minimum_squared_distance))
        tied = np.flatnonzero(
            np.isclose(minimum_squared_distance, maximum, rtol=0.0, atol=1e-12)
        )
        next_index = min(
            map(int, tied),
            key=lambda index: _stable_rank(
                seed, f"fps-tie-{len(selected)}", ordered[index]
            ),
        )
        selected.append(next_index)
        squared_distance = np.sum((xy - xy[next_index]) ** 2, axis=1)
        minimum_squared_distance = np.minimum(
            minimum_squared_distance, squared_distance
        )
        minimum_squared_distance[np.asarray(selected, dtype=np.int64)] = -np.inf
    return {ordered[index] for index in selected}


def _space_filling_support(
    keys: set[LocationKey],
    fraction: float,
    *,
    seed: int,
) -> set[LocationKey]:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must lie in (0,1)")
    strata: dict[tuple[int, int], list[LocationKey]] = {}
    for key in keys:
        strata.setdefault(key[:2], []).append(key)
    support: set[LocationKey] = set()
    for stratum, values in sorted(strata.items()):
        if len(values) < 2:
            support.update(values)
            continue
        count = int(round(len(values) * float(fraction)))
        count = min(len(values) - 1, max(1, count))
        support.update(
            _farthest_point_sample(
                values,
                count,
                seed=int(seed) + 1009 * int(stratum[0]) + 9176 * int(stratum[1]),
            )
        )
    if not support or support == keys:
        raise RuntimeError("space-filling split did not leave support and evaluation")
    return support


def _global_detection_background(
    rss: torch.Tensor,
    phone_ids: np.ndarray,
    coordinate_keys: list[LocationKey],
    indices: np.ndarray,
) -> torch.Tensor:
    """Equal-phone/equal-coordinate AP detection background."""

    selected = np.asarray(indices, dtype=np.int64)
    phone_summaries = []
    for phone in sorted(map(int, np.unique(phone_ids[selected]))):
        phone_rows = selected[phone_ids[selected] == phone]
        coordinate_summaries = []
        for key in sorted({coordinate_keys[int(row)] for row in phone_rows}):
            rows = np.asarray(
                [row for row in phone_rows if coordinate_keys[int(row)] == key],
                dtype=np.int64,
            )
            coordinate_summaries.append((rss[rows] > -0.999).float().mean(dim=0))
        phone_summaries.append(torch.stack(coordinate_summaries).mean(dim=0))
    if not phone_summaries:
        raise ValueError("global detection background has no represented phones")
    return torch.stack(phone_summaries).mean(dim=0)


def _natural_parameter_predictions(
    evaluation_keys: list[LocationKey],
    support_keys: list[LocationKey],
    support_probability: np.ndarray,
    support_center: np.ndarray,
    support_center_available: np.ndarray,
    *,
    neighbors: int,
    inverse_distance_power: float,
) -> dict[str, np.ndarray]:
    """Predict Bernoulli probabilities and conditional RSS centers."""

    probability = np.asarray(support_probability, dtype=np.float64)
    center = np.asarray(support_center, dtype=np.float64)
    available = np.asarray(support_center_available, dtype=bool)
    if probability.shape != center.shape or center.shape != available.shape:
        raise ValueError("support radio fields must share [anchors,aps]")
    eta = np.log(probability) - np.log1p(-probability)
    count = len(evaluation_keys)
    aps = probability.shape[1]
    nearest_probability = np.empty((count, aps), dtype=np.float64)
    smooth_probability = np.empty((count, aps), dtype=np.float64)
    nearest_center = np.full((count, aps), np.nan, dtype=np.float64)
    smooth_center = np.full((count, aps), np.nan, dtype=np.float64)
    nearest_distance = np.empty(count, dtype=np.float64)

    support_strata: dict[tuple[int, int], np.ndarray] = {}
    for stratum in sorted({key[:2] for key in support_keys}):
        support_strata[stratum] = np.asarray(
            [index for index, key in enumerate(support_keys) if key[:2] == stratum],
            dtype=np.int64,
        )
    support_xy = np.asarray(
        [[key[2], key[3]] for key in support_keys], dtype=np.float64
    )
    for row, key in enumerate(evaluation_keys):
        local = support_strata.get(key[:2])
        if local is None or len(local) == 0:
            raise RuntimeError(f"evaluation stratum {key[:2]} has no support point")
        xy = np.asarray([key[2], key[3]], dtype=np.float64)
        distance = np.linalg.norm(support_xy[local] - xy[None, :], axis=1)
        order = np.argsort(distance, kind="stable")
        nearest = int(local[int(order[0])])
        nearest_distance[row] = float(distance[int(order[0])])
        nearest_probability[row] = probability[nearest]
        k = min(int(neighbors), len(local))
        chosen = local[order[:k]]
        chosen_distance = distance[order[:k]]
        weight = 1.0 / np.maximum(chosen_distance, 1e-6) ** float(
            inverse_distance_power
        )
        weight /= weight.sum()
        smooth_eta = weight @ eta[chosen]
        smooth_probability[row] = 1.0 / (1.0 + np.exp(-smooth_eta))

        # Conditional RSS is only defined where an AP was observed at a
        # support point.  Use AP-specific nearest/IDW fields and report the
        # resulting event coverage separately.
        local_available = available[local]
        for ap in range(aps):
            valid_position = np.flatnonzero(local_available[:, ap])
            if len(valid_position) == 0:
                continue
            valid_local = local[valid_position]
            valid_distance = distance[valid_position]
            valid_order = np.argsort(valid_distance, kind="stable")
            nearest_ap = int(valid_local[int(valid_order[0])])
            nearest_center[row, ap] = center[nearest_ap, ap]
            rss_k = min(int(neighbors), len(valid_local))
            rss_chosen = valid_local[valid_order[:rss_k]]
            rss_distance = valid_distance[valid_order[:rss_k]]
            rss_weight = 1.0 / np.maximum(rss_distance, 1e-6) ** float(
                inverse_distance_power
            )
            rss_weight /= rss_weight.sum()
            smooth_center[row, ap] = float(rss_weight @ center[rss_chosen, ap])
    return {
        "nearest_probability": nearest_probability,
        "smooth_probability": smooth_probability,
        "nearest_center": nearest_center,
        "smooth_center": smooth_center,
        "nearest_distance_m": nearest_distance,
    }


def _macro_mean(
    values: np.ndarray,
    row_indices: np.ndarray,
    keys: list[LocationKey],
    phones: np.ndarray,
) -> dict[str, float]:
    """Return coordinate macro and equal-phone/coordinate macro means."""

    vector = np.asarray(values, dtype=np.float64)
    selected = np.asarray(row_indices, dtype=np.int64)
    if vector.shape != (len(selected),) or not np.isfinite(vector).all():
        raise ValueError("macro input must be one finite value per selected row")
    by_coordinate: dict[LocationKey, list[float]] = {}
    by_phone_coordinate: dict[int, dict[LocationKey, list[float]]] = {}
    for local, global_row in enumerate(selected):
        key = keys[int(global_row)]
        phone = int(phones[int(global_row)])
        value = float(vector[local])
        by_coordinate.setdefault(key, []).append(value)
        by_phone_coordinate.setdefault(phone, {}).setdefault(key, []).append(value)
    coordinate_values = np.asarray(
        [np.mean(group) for group in by_coordinate.values()], dtype=np.float64
    )
    phone_values = np.asarray(
        [
            np.mean([np.mean(group) for group in coordinate_groups.values()])
            for coordinate_groups in by_phone_coordinate.values()
        ],
        dtype=np.float64,
    )
    return {
        "row_mean": float(vector.mean()),
        "coordinate_macro_mean": float(coordinate_values.mean()),
        "equal_phone_coordinate_macro_mean": float(phone_values.mean()),
        "represented_phones": int(len(phone_values)),
        "represented_coordinates": int(len(coordinate_values)),
    }


def _paired_coordinate_delta(
    smooth_values: np.ndarray,
    nearest_values: np.ndarray,
    row_indices: np.ndarray,
    keys: list[LocationKey],
) -> dict[str, float | dict[str, float]]:
    """Summarize smooth-minus-nearest differences at the coordinate unit."""

    smooth = np.asarray(smooth_values, dtype=np.float64)
    nearest = np.asarray(nearest_values, dtype=np.float64)
    selected = np.asarray(row_indices, dtype=np.int64)
    if smooth.shape != nearest.shape or smooth.shape != (len(selected),):
        raise ValueError("paired values must have one entry per selected row")
    grouped: dict[LocationKey, list[float]] = {}
    for local, global_row in enumerate(selected):
        grouped.setdefault(keys[int(global_row)], []).append(
            float(smooth[local] - nearest[local])
        )
    coordinate_delta = np.asarray(
        [np.mean(values) for values in grouped.values()], dtype=np.float64
    )
    return {
        "coordinate_delta": _summary(coordinate_delta),
        "coordinate_fraction_positive": float(np.mean(coordinate_delta > 0.0)),
        "coordinate_fraction_negative": float(np.mean(coordinate_delta < 0.0)),
        "coordinate_fraction_zero": float(np.mean(coordinate_delta == 0.0)),
    }


def _evaluate_design(
    rss: torch.Tensor,
    raw_xy: np.ndarray,
    building: np.ndarray,
    floor_values: np.ndarray,
    phones: np.ndarray,
    keys: list[LocationKey],
    development_rows: np.ndarray,
    support_keys_set: set[LocationKey],
    *,
    neighbors: int,
    inverse_distance_power: float,
) -> dict:
    support_indices = np.asarray(
        [row for row in development_rows if keys[int(row)] in support_keys_set],
        dtype=np.int64,
    )
    evaluation_indices = np.asarray(
        [row for row in development_rows if keys[int(row)] not in support_keys_set],
        dtype=np.int64,
    )
    if len(support_indices) == 0 or len(evaluation_indices) == 0:
        raise RuntimeError("sparse radio-map design has an empty role")
    origin = raw_xy[support_indices].mean(axis=0)
    memory, _ = build_device_balanced_anchor_memory(
        rss,
        raw_xy - origin,
        building,
        floor_values,
        phones,
        support_indices,
        coordinate_keys=keys,
    )
    support_keys = list(memory["coordinate_keys"])
    raw_probability_t = memory["detection_prob"].to(dtype=torch.float64)
    background_t = _global_detection_background(
        rss, phones, keys, support_indices
    ).to(dtype=torch.float64)
    device_count_t = memory["device_counts"].to(dtype=torch.float64)
    probability_t = (
        raw_probability_t * device_count_t[:, None] + background_t[None, :]
    ) / (device_count_t[:, None] + 1.0)
    probability = probability_t.clamp(1e-5, 1.0 - 1e-5).numpy()
    center = memory["rss_center"].to(dtype=torch.float64).numpy()
    center_available = raw_probability_t.numpy() > 0.0

    evaluation_keys = sorted(
        {keys[int(row)] for row in evaluation_indices}
    )
    prediction = _natural_parameter_predictions(
        evaluation_keys,
        support_keys,
        probability,
        center,
        center_available,
        neighbors=neighbors,
        inverse_distance_power=inverse_distance_power,
    )
    evaluation_key_index = {
        key: index for index, key in enumerate(evaluation_keys)
    }
    prediction_row = np.asarray(
        [evaluation_key_index[keys[int(row)]] for row in evaluation_indices],
        dtype=np.int64,
    )
    observed = (rss[evaluation_indices] > -0.999).numpy()
    observed_float = observed.astype(np.float64)

    metrics: dict[str, dict] = {}
    row_values: dict[str, dict[str, np.ndarray]] = {}
    for name in ("nearest", "smooth"):
        p = prediction[f"{name}_probability"][prediction_row]
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        log_score = np.mean(
            observed_float * np.log(p)
            + (1.0 - observed_float) * np.log1p(-p),
            axis=1,
        )
        brier = np.mean((observed_float - p) ** 2, axis=1)
        predicted_center = prediction[f"{name}_center"][prediction_row]
        supported = observed & np.isfinite(predicted_center)
        query_dbm = (rss[evaluation_indices].numpy() + 1.0) * 40.0 - 110.0
        center_dbm = (predicted_center + 1.0) * 40.0 - 110.0
        absolute_error = np.abs(query_dbm - center_dbm)
        supported_count = supported.sum(axis=1)
        row_rss_mae = np.divide(
            np.where(supported, absolute_error, 0.0).sum(axis=1),
            supported_count,
            out=np.zeros(len(evaluation_indices), dtype=np.float64),
            where=supported_count > 0,
        )
        supported_rows = supported_count > 0
        if not np.any(supported_rows):
            raise RuntimeError("no evaluation scan has a supported visible AP")
        metrics[name] = {
            "detection_log_score_nats_per_ap": _macro_mean(
                log_score, evaluation_indices, keys, phones
            ),
            "detection_brier_per_ap": _macro_mean(
                brier, evaluation_indices, keys, phones
            ),
            "visible_rss_mae_dbm": _macro_mean(
                row_rss_mae[supported_rows],
                evaluation_indices[supported_rows],
                keys,
                phones,
            ),
            "visible_event_prediction_coverage": float(
                supported.sum() / max(1, observed.sum())
            ),
            "scan_with_at_least_one_rss_prediction_fraction": float(
                supported_rows.mean()
            ),
        }
        row_values[name] = {
            "detection_log_score": log_score,
            "detection_brier": brier,
            "visible_rss_mae": row_rss_mae,
            "rss_supported_rows": supported_rows,
        }

    nearest_supported = row_values["nearest"]["rss_supported_rows"].astype(bool)
    smooth_supported = row_values["smooth"]["rss_supported_rows"].astype(bool)
    if not np.array_equal(nearest_supported, smooth_supported):
        raise RuntimeError("nearest and smooth RSS comparisons have unequal support")
    paired_coordinate_deltas = {
        "detection_log_score_nats_per_ap": _paired_coordinate_delta(
            row_values["smooth"]["detection_log_score"],
            row_values["nearest"]["detection_log_score"],
            evaluation_indices,
            keys,
        ),
        "detection_brier_per_ap": _paired_coordinate_delta(
            row_values["smooth"]["detection_brier"],
            row_values["nearest"]["detection_brier"],
            evaluation_indices,
            keys,
        ),
        "visible_rss_mae_dbm": _paired_coordinate_delta(
            row_values["smooth"]["visible_rss_mae"][smooth_supported],
            row_values["nearest"]["visible_rss_mae"][nearest_supported],
            evaluation_indices[smooth_supported],
            keys,
        ),
    }

    nearest_log = metrics["nearest"]["detection_log_score_nats_per_ap"]
    smooth_log = metrics["smooth"]["detection_log_score_nats_per_ap"]
    nearest_brier = metrics["nearest"]["detection_brier_per_ap"]
    smooth_brier = metrics["smooth"]["detection_brier_per_ap"]
    nearest_rss = metrics["nearest"]["visible_rss_mae_dbm"]
    smooth_rss = metrics["smooth"]["visible_rss_mae_dbm"]
    return {
        "support_coordinates": int(len(support_keys_set)),
        "evaluation_coordinates": int(len(evaluation_keys)),
        "support_rows": int(len(support_indices)),
        "evaluation_rows": int(len(evaluation_indices)),
        "evaluation_to_support_nearest_distance_m": _summary(
            prediction["nearest_distance_m"]
        ),
        "metrics": metrics,
        "paired_coordinate_smooth_minus_nearest": paired_coordinate_deltas,
        "smooth_minus_nearest": {
            "detection_log_score_nats_per_ap": float(
                smooth_log["equal_phone_coordinate_macro_mean"]
                - nearest_log["equal_phone_coordinate_macro_mean"]
            ),
            "detection_brier_per_ap": float(
                smooth_brier["equal_phone_coordinate_macro_mean"]
                - nearest_brier["equal_phone_coordinate_macro_mean"]
            ),
            "visible_rss_mae_dbm": float(
                smooth_rss["equal_phone_coordinate_macro_mean"]
                - nearest_rss["equal_phone_coordinate_macro_mean"]
            ),
        },
    }


def run_audit(
    train_data: str | Path,
    registry_path: str | Path,
    *,
    fractions: tuple[float, ...],
    seeds: tuple[int, ...],
    neighbors: int,
    inverse_distance_power: float,
) -> dict:
    data_path = resolve_path(str(train_data)).resolve()
    registry_file = resolve_path(str(registry_path)).resolve()
    registry = validate_registry(registry_file, train_data=data_path)
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
    phones = phone_t.numpy()
    keys = location_keys(raw_xy, building, floor_values)
    development_phones = tuple(map(int, registry["device_development_phone_ids"]))
    coordinate_partition = coordinate_axis_partition(
        keys, phones, development_phones
    )
    development_keys = set(coordinate_partition["development_core"])
    development_rows = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if int(phone) in development_phones and keys[row] in development_keys
        ],
        dtype=np.int64,
    )

    designs: dict[str, dict] = {}
    pooled_by_fraction: dict[str, dict] = {}
    for fraction in fractions:
        fraction_deltas: dict[str, list[float]] = {
            "detection_log_score_nats_per_ap": [],
            "detection_brier_per_ap": [],
            "visible_rss_mae_dbm": [],
        }
        distance_medians = []
        for seed in seeds:
            support = _space_filling_support(
                development_keys, float(fraction), seed=int(seed)
            )
            result = _evaluate_design(
                rss,
                raw_xy,
                building,
                floor_values,
                phones,
                keys,
                development_rows,
                support,
                neighbors=neighbors,
                inverse_distance_power=inverse_distance_power,
            )
            name = f"fraction_{fraction:.3f}_seed_{int(seed)}"
            designs[name] = result
            distance_medians.append(
                result["evaluation_to_support_nearest_distance_m"]["median"]
            )
            for metric, value in result["smooth_minus_nearest"].items():
                fraction_deltas[metric].append(float(value))
        pooled_by_fraction[f"{fraction:.3f}"] = {
            "replicates": int(len(seeds)),
            "median_nearest_support_distance_m": _summary(
                np.asarray(distance_medians, dtype=np.float64)
            ),
            "smooth_minus_nearest": {
                metric: _summary(np.asarray(values, dtype=np.float64))
                for metric, values in fraction_deltas.items()
            },
        }

    strata_counts: dict[str, int] = {}
    for key in development_keys:
        label = f"building_{key[0]}_floor_{key[1]}"
        strata_counts[label] = strata_counts.get(label, 0) + 1
    return {
        "artifact_status": "v5_sparse_continuous_radiomap_feasibility_audit",
        "selection_authority": False,
        "localization_predictions_or_errors_computed": False,
        "protected_coordinate_confirmation_evaluated": False,
        "protected_confirmation_phones_evaluated": False,
        "training_data": str(data_path),
        "registry": str(registry_file),
        "registry_sha256": registry["registry_sha256"],
        "development_phones": list(development_phones),
        "development_rows": int(len(development_rows)),
        "development_coordinates": int(len(development_keys)),
        "development_coordinates_by_building_floor": strata_counts,
        "design": {
            "support_selection": (
                "geometry-only farthest-point sampling within building/floor"
            ),
            "retained_coordinate_fractions": list(map(float, fractions)),
            "geometric_replicate_seeds": list(map(int, seeds)),
            "continuous_detection_field": (
                "inverse-distance interpolation of smoothed Bernoulli logits"
            ),
            "continuous_visible_rss_field": (
                "AP-specific inverse-distance interpolation of visible RSS centers"
            ),
            "neighbors": int(neighbors),
            "inverse_distance_power": float(inverse_distance_power),
            "nearest_control": (
                "nearest coordinate for detection and nearest AP-observed coordinate "
                "for conditional RSS"
            ),
        },
        "pooled_by_fraction": pooled_by_fraction,
        "designs": designs,
        "interpretation_contract": {
            "positive_continuous_field_signal": (
                "A useful field must improve detection log score (positive delta), "
                "reduce Brier score (negative delta), and not materially worsen "
                "visible-RSS MAE across geometric replicates."
            ),
            "scope": (
                "This tests spatial measurement predictability only.  It neither "
                "selects a localization model nor demonstrates localization benefit."
            ),
            "geometry_boundary": (
                "The support design knows the development coordinate geometry but "
                "does not use RSS values; this corresponds to planning a sparse "
                "survey over a known candidate floor grid."
            ),
        },
    }


def _csv_tuple(text: str, cast) -> tuple:
    values = tuple(cast(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the predictive premise of a sparse continuous radio map"
    )
    parser.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv"
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--fractions", default="0.25,0.50,0.75")
    parser.add_argument("--seeds", default="5101,5102,5103")
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--inverse-distance-power", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.neighbors < 1:
        raise ValueError("--neighbors must be positive")
    if args.inverse_distance_power <= 0.0:
        raise ValueError("--inverse-distance-power must be positive")
    fractions = _csv_tuple(args.fractions, float)
    seeds = _csv_tuple(args.seeds, int)
    payload = run_audit(
        args.train_data,
        args.registry,
        fractions=fractions,
        seeds=seeds,
        neighbors=args.neighbors,
        inverse_distance_power=args.inverse_distance_power,
    )
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.output is not None:
        output = resolve_path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite audit artifact: {output}")
        output.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
