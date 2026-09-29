"""Same-split classical device-heterogeneity baselines for DR-JAM v4.4."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords, resolve_path
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.final_v44 import _confirmation_roles
from fhm_dapp_tim.fhm_dapp.freeze_confirmation_v43 import (
    validate_confirmation_manifest,
)
from fhm_dapp_tim.fhm_dapp.protocol_v4 import (
    MEMBER_VISIBLE_THRESHOLD,
    PROTOCOL_VERSION,
    coordinate_axis_partition,
    coordinate_roles_for_development_phones,
    device_epoch_selection_roles,
    sha256_indices,
    validate_registry,
)


METHODS = ("raw_wknn", "diff_wknn", "rank_wknn")
K_GRID = (1, 3, 5, 8, 12, 20)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _visible_centered_feature(values: np.ndarray) -> np.ndarray:
    """Mask-aware offset-invariant representation for SSD/DIFF WKNN."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("RSS features must have shape [rows,aps]")
    visible = x > float(MEMBER_VISIBLE_THRESHOLD)
    count = visible.sum(axis=1, keepdims=True)
    center = np.divide(
        np.where(visible, x, 0.0).sum(axis=1, keepdims=True),
        np.maximum(count, 1),
    )
    centered = np.where(visible, x - center, 0.0)
    return np.concatenate((visible.astype(np.float64), centered), axis=1)


def _rank_feature(values: np.ndarray) -> np.ndarray:
    """Mask plus normalized within-visible-set RSS rank."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("RSS features must have shape [rows,aps]")
    visible = x > float(MEMBER_VISIBLE_THRESHOLD)
    rank = np.zeros_like(x, dtype=np.float64)
    for row in range(len(x)):
        indices = np.flatnonzero(visible[row])
        if len(indices) == 0:
            continue
        order = indices[np.argsort(x[row, indices], kind="stable")]
        if len(order) == 1:
            rank[row, order] = 1.0
        else:
            rank[row, order] = np.arange(len(order), dtype=np.float64) / float(
                len(order) - 1
            )
    return np.concatenate((visible.astype(np.float64), rank), axis=1)


def _features(values: np.ndarray, method: str) -> np.ndarray:
    if method == "raw_wknn":
        return np.asarray(values, dtype=np.float64)
    if method == "diff_wknn":
        return _visible_centered_feature(values)
    if method == "rank_wknn":
        return _rank_feature(values)
    raise ValueError(f"unknown baseline: {method}")


def _coordinate_prototypes(
    rss: np.ndarray,
    raw_xy: np.ndarray,
    phones: np.ndarray,
    keys: list[tuple],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """One equal-phone fingerprint prototype per protocol coordinate group."""
    grouped: dict[tuple, list[int]] = {}
    for row in np.asarray(indices, dtype=np.int64):
        grouped.setdefault(keys[int(row)], []).append(int(row))
    if not grouped:
        raise ValueError("baseline training role contains no coordinate")
    prototypes = []
    targets = []
    ordered_keys = sorted(grouped)
    for key in ordered_keys:
        rows = np.asarray(grouped[key], dtype=np.int64)
        phone_summaries = []
        for phone in sorted(map(int, np.unique(phones[rows]))):
            phone_rows = rows[phones[rows] == phone]
            phone_summaries.append(np.median(rss[phone_rows], axis=0))
        prototypes.append(np.mean(np.stack(phone_summaries), axis=0))
        targets.append(raw_xy[rows].mean(axis=0))
    return np.stack(prototypes), np.stack(targets), ordered_keys


def _predict_k_grid(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    k_grid: tuple[int, ...] = K_GRID,
    chunk_size: int = 256,
) -> dict[int, np.ndarray]:
    if max(k_grid) > len(train_x):
        raise ValueError("baseline k exceeds the prototype count")
    outputs = {
        int(k): np.empty((len(test_x), 2), dtype=np.float64) for k in k_grid
    }
    train = np.asarray(train_x, dtype=np.float64)
    train_norm = np.sum(train * train, axis=1)
    for start in range(0, len(test_x), int(chunk_size)):
        stop = min(start + int(chunk_size), len(test_x))
        block = np.asarray(test_x[start:stop], dtype=np.float64)
        distance2 = (
            np.sum(block * block, axis=1)[:, None]
            + train_norm[None, :]
            - 2.0 * block @ train.T
        )
        np.maximum(distance2, 0.0, out=distance2)
        # Stable full sorting is inexpensive for <=679 prototypes and fixes
        # exact-distance ties by prototype order.
        neighbors = np.argsort(distance2, axis=1, kind="stable")[:, : max(k_grid)]
        local_distance = np.take_along_axis(distance2, neighbors, axis=1)
        weights = 1.0 / np.maximum(np.sqrt(local_distance), 1e-8)
        neighbor_targets = train_y[neighbors]
        weighted_sum = np.cumsum(neighbor_targets * weights[:, :, None], axis=1)
        weight_sum = np.cumsum(weights, axis=1)
        for k in k_grid:
            outputs[int(k)][start:stop] = weighted_sum[:, int(k) - 1] / weight_sum[
                :, int(k) - 1, None
            ]
    return outputs


def _group_macro_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    keys: list[tuple],
    indices: np.ndarray,
) -> tuple[float, dict[tuple, float], np.ndarray]:
    errors = np.linalg.norm(predictions - targets, axis=1)
    grouped: dict[tuple, list[float]] = {}
    for row, error in zip(np.asarray(indices, dtype=np.int64), errors):
        grouped.setdefault(keys[int(row)], []).append(float(error))
    per_group = {key: float(np.mean(values)) for key, values in grouped.items()}
    return float(np.mean(list(per_group.values()))), per_group, errors


def select_baseline_k(
    coordinate_by_group: Mapping[int, Mapping[object, float]],
    device_by_phone: Mapping[int, Mapping[int, float]],
    *,
    coordinate_margin_m: float = 0.50,
    device_tie_m: float = 0.10,
) -> dict:
    candidates = tuple(sorted(map(int, coordinate_by_group)))
    if set(candidates) != set(map(int, device_by_phone)):
        raise ValueError("coordinate and device k grids differ")
    coordinate_units = None
    phone_units = None
    coordinate_macro = {}
    device_macro = {}
    for k in candidates:
        current_coordinates = tuple(sorted(coordinate_by_group[k], key=str))
        current_phones = tuple(sorted(device_by_phone[k]))
        if coordinate_units is None:
            coordinate_units = current_coordinates
            phone_units = current_phones
        if current_coordinates != coordinate_units or current_phones != phone_units:
            raise ValueError("baseline selection units are not paired")
        coordinate_macro[k] = float(
            np.mean([coordinate_by_group[k][key] for key in coordinate_units])
        )
        device_macro[k] = float(
            np.mean([device_by_phone[k][phone] for phone in phone_units])
        )
    coordinate_best = min(coordinate_macro.values())
    eligible = [
        k
        for k in candidates
        if coordinate_macro[k] <= coordinate_best + float(coordinate_margin_m)
    ]
    device_best = min(device_macro[k] for k in eligible)
    practically_tied = [
        k for k in eligible if device_macro[k] <= device_best + float(device_tie_m)
    ]
    selected = min(practically_tied)
    return {
        "selected_k": selected,
        "coordinate_eligible_k": eligible,
        "device_practically_tied_k": practically_tied,
        "coordinate_group_macro_mae_m": {str(k): coordinate_macro[k] for k in candidates},
        "device_phone_coordinate_group_macro_mae_m": {
            str(k): device_macro[k] for k in candidates
        },
    }


def _load_data(data_path: Path):
    data = load_dataset(data_path, dataset_type="UJI")
    raw_xy = raw_coords(data[1], data[2], data[8])
    building = data[2].numpy()
    floor = data[3].numpy()
    phones = data[6].numpy().astype(np.int64, copy=False)
    keys = location_keys(raw_xy, building, floor)
    return data[0].numpy(), raw_xy, phones, keys


def tune_baselines(args: argparse.Namespace) -> dict:
    data_path = resolve_path(args.train_data).resolve()
    registry_path = Path(args.registry).resolve()
    registry = validate_registry(registry_path, train_data=data_path)
    rss, raw_xy, phones, keys = _load_data(data_path)
    development = tuple(map(int, registry["device_development_phone_ids"]))
    core = coordinate_axis_partition(keys, phones, development)["development_core"]
    identifiable = tuple(map(int, registry["development_identifiable_phone_ids"]))
    coordinate_scores = {
        method: {k: {} for k in K_GRID} for method in METHODS
    }
    device_scores = {method: {k: {} for k in K_GRID} for method in METHODS}
    raw_audit = {method: {"coordinate": {}, "device": {}} for method in METHODS}
    started = time.perf_counter()

    for fold in range(len(registry["coordinate_development_audit"])):
        roles = coordinate_roles_for_development_phones(
            keys, phones, development, fold
        )
        train_idx = np.sort(
            np.concatenate((roles["member_fit"], roles["early_stop"]))
        ).astype(np.int64)
        test_idx = roles["structure_development"]
        prototypes, prototype_xy, _prototype_keys = _coordinate_prototypes(
            rss, raw_xy, phones, keys, train_idx
        )
        for method in METHODS:
            predictions = _predict_k_grid(
                _features(prototypes, method),
                prototype_xy,
                _features(rss[test_idx], method),
                chunk_size=int(args.chunk_size),
            )
            raw_audit[method]["coordinate"][str(fold)] = {}
            for k, predicted in predictions.items():
                macro, per_group, _errors = _group_macro_errors(
                    predicted, raw_xy[test_idx], keys, test_idx
                )
                raw_audit[method]["coordinate"][str(fold)][str(k)] = macro
                overlap = set(coordinate_scores[method][k]) & set(per_group)
                if overlap:
                    raise RuntimeError("a core coordinate was validated twice")
                coordinate_scores[method][k].update(per_group)

    for held_out in identifiable:
        training_phones = tuple(
            phone for phone in development if phone != held_out
        )
        roles = device_epoch_selection_roles(
            keys,
            phones,
            held_out,
            training_phones=training_phones,
            allowed_coordinate_keys=core,
        )
        train_idx = roles["refit_train"]
        test_idx = roles["matched_test"]
        prototypes, prototype_xy, _prototype_keys = _coordinate_prototypes(
            rss, raw_xy, phones, keys, train_idx
        )
        for method in METHODS:
            predictions = _predict_k_grid(
                _features(prototypes, method),
                prototype_xy,
                _features(rss[test_idx], method),
                chunk_size=int(args.chunk_size),
            )
            raw_audit[method]["device"][str(held_out)] = {}
            for k, predicted in predictions.items():
                macro, _per_group, _errors = _group_macro_errors(
                    predicted, raw_xy[test_idx], keys, test_idx
                )
                device_scores[method][k][held_out] = macro
                raw_audit[method]["device"][str(held_out)][str(k)] = macro

    expected_groups = int(
        registry["coordinate_axis"]["development_core_coordinate_groups"]
    )
    decisions = {}
    for method in METHODS:
        if any(len(coordinate_scores[method][k]) != expected_groups for k in K_GRID):
            raise RuntimeError("baseline coordinate cross-fitting is incomplete")
        decisions[method] = select_baseline_k(
            coordinate_scores[method], device_scores[method]
        )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite baseline freeze: {output}")
    payload = {
        "artifact_status": "registered_v44_baseline_freeze",
        "selection_authority": True,
        "protocol_version": PROTOCOL_VERSION,
        "registry_path": str(registry_path),
        "registry_sha256": registry["registry_sha256"],
        "training_data_sha256": _sha256_file(data_path),
        "methods": list(METHODS),
        "k_grid": list(K_GRID),
        "representation_contract": {
            "radio_map": (
                "one coordinate prototype; median rows within phone then "
                "equal-phone mean"
            ),
            "raw_wknn": "fixed-width normalized RSS including missing sentinel",
            "diff_wknn": (
                "visibility mask plus visible-mean-centered RSS; Euclidean "
                "distance is an offset-invariant SSD/DIFF representation"
            ),
            "rank_wknn": "visibility mask plus normalized stable visible RSS ranks",
            "weight": "inverse Euclidean distance",
        },
        "selection_rule": {
            "coordinate_guard_m": 0.50,
            "device_practical_tie_m": 0.10,
            "tie_break": "smallest k",
            "coordinate_unit": "exact development-core coordinate",
            "device_unit": (
                "coordinate group within phone, followed by equal-phone macro"
            ),
            "feeds_back_into_dr_jam": False,
        },
        "decisions": decisions,
        "raw_fold_and_phone_macro_scores": raw_audit,
        "paired_selection_units": {
            "coordinate_group": {
                method: {
                    str(k): {
                        json.dumps(list(key), separators=(",", ":")): value
                        for key, value in coordinate_scores[method][k].items()
                    }
                    for k in K_GRID
                }
                for method in METHODS
            },
            "phone": {
                method: {
                    str(k): {
                        str(phone): value
                        for phone, value in device_scores[method][k].items()
                    }
                    for k in K_GRID
                }
                for method in METHODS
            },
        },
        "confirmation_roles_evaluated": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "decisions": decisions,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return payload


def evaluate_baseline_confirmation(args: argparse.Namespace) -> dict:
    data_path = resolve_path(args.train_data).resolve()
    registry_path = Path(args.registry).resolve()
    registry = validate_registry(registry_path, train_data=data_path)
    freeze_path = Path(args.baseline_freeze).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("artifact_status") != "registered_v44_baseline_freeze":
        raise ValueError("baseline artifact has no v4.4 freeze authority")
    if freeze.get("selection_authority") is not True:
        raise ValueError("baseline freeze selection authority is false")
    if freeze.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("baseline freeze protocol mismatch")
    if freeze.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("baseline freeze registry mismatch")
    manifest_path = Path(args.confirmation_manifest).resolve()
    manifest = validate_confirmation_manifest(
        manifest_path, train_data=data_path, registry_path=registry_path
    )
    rss, raw_xy, phones, keys = _load_data(data_path)
    development = set(map(int, registry["device_development_phone_ids"]))
    core = coordinate_axis_partition(keys, phones, development)["development_core"]
    train_idx = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if int(phone) in development and keys[row] in core
        ],
        dtype=np.int64,
    )
    if sha256_indices(train_idx) != manifest["training_contract"]["fit_role"][
        "row_index_sha256"
    ]:
        raise ValueError("baseline final radio map differs from the frozen core")
    roles = _confirmation_roles(
        keys=keys, phones=phones, registry=registry, manifest=manifest
    )
    prototypes, prototype_xy, _prototype_keys = _coordinate_prototypes(
        rss, raw_xy, phones, keys, train_idx
    )
    results = {}
    started = time.perf_counter()
    for method in METHODS:
        selected_k = int(freeze["decisions"][method]["selected_k"])
        method_results = {}
        train_features = _features(prototypes, method)
        for role_name, indices in roles.items():
            prediction = _predict_k_grid(
                train_features,
                prototype_xy,
                _features(rss[indices], method),
                k_grid=(selected_k,),
                chunk_size=int(args.chunk_size),
            )[selected_k]
            macro, per_group, errors = _group_macro_errors(
                prediction, raw_xy[indices], keys, indices
            )
            per_phone = {}
            for phone in sorted(map(int, np.unique(phones[indices]))):
                mask = phones[indices] == phone
                phone_rows = indices[mask]
                phone_macro, _phone_groups, phone_errors = _group_macro_errors(
                    prediction[mask], raw_xy[phone_rows], keys, phone_rows
                )
                per_phone[str(phone)] = {
                    "rows": int(mask.sum()),
                    "coordinate_groups": len({keys[int(row)] for row in phone_rows}),
                    "coordinate_group_macro_mae_m": phone_macro,
                    "row_mae_m": float(phone_errors.mean()),
                }
            method_results[role_name] = {
                "rows": len(indices),
                "coordinate_groups": len(per_group),
                "coordinate_group_macro_mae_m": macro,
                "row_mae_m": float(errors.mean()),
                "phone_coordinate_group_macro_mae_m": (
                    float(
                        np.mean(
                            [
                                value["coordinate_group_macro_mae_m"]
                                for value in per_phone.values()
                            ]
                        )
                    )
                    if per_phone
                    else None
                ),
                "per_phone": per_phone,
                "row_indices": indices.tolist(),
                "errors_m": errors.tolist(),
                "predicted_xy_m": prediction.tolist(),
            }
        results[method] = {
            "selected_k": selected_k,
            "roles": method_results,
        }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite baseline confirmation: {output}"
        )
    payload = {
        "artifact_status": "registered_v44_baseline_two_axis_confirmation",
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "baseline_freeze_sha256": _sha256_file(freeze_path),
        "confirmation_manifest_sha256": _sha256_file(manifest_path),
        "role_row_sha256": {
            name: sha256_indices(indices) for name, indices in roles.items()
        },
        "results": results,
        "protected_roles_opened_once": True,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "artifact_status": payload["artifact_status"],
                "summaries": {
                    method: {
                        role: {
                            key: value
                            for key, value in role_result.items()
                            if key
                            not in {
                                "row_indices",
                                "errors_m",
                                "predicted_xy_m",
                                "per_phone",
                            }
                        }
                        for role, role_result in method_result["roles"].items()
                    }
                    for method, method_result in results.items()
                },
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DR-JAM v4.4 classical baselines")
    children = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv"
    )
    common.add_argument(
        "--registry", default="fhm_dapp_tim/evidence/v44/registry_v4_4.json"
    )
    common.add_argument("--chunk-size", type=int, default=256)
    tune = children.add_parser("tune", parents=[common])
    tune.add_argument(
        "--output",
        default="fhm_dapp_tim/evidence/v44/baseline_selection_v4_4.json",
    )
    tune.set_defaults(handler=tune_baselines)
    confirm = children.add_parser("confirm", parents=[common])
    confirm.add_argument(
        "--baseline-freeze",
        default="fhm_dapp_tim/evidence/v44/baseline_selection_v4_4.json",
    )
    confirm.add_argument(
        "--confirmation-manifest",
        default="fhm_dapp_tim/evidence/v44/confirmation_roles_v4_4.json",
    )
    confirm.add_argument(
        "--output",
        default="fhm_dapp_tim/evidence/v44/baseline_confirmation_v4_4.json",
    )
    confirm.set_defaults(handler=evaluate_baseline_confirmation)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
