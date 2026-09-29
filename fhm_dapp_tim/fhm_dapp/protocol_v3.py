"""Nested reference-location-group protocol for the minimal v3 method.

The primary estimand is interpolation to unseen physical reference-location
groups in the same acquisition domain.  Outer folds are stratified by
building and floor.  Within every outer-training fold, independent groups are
assigned to member fitting, member early stopping, and structure development.
No role ever shares a rounded coordinate key with another role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from fhm_dapp_tim.fhm_dapp.anchor import (
    location_keys,
    raw_coords,
    resolve_path,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset


PROTOCOL_VERSION = "uji_nested_group_v3"
OUTER_FOLDS = 5
SPLIT_SEED = 20260804
EARLY_STOP_FRACTION_OF_OUTER_TRAIN = 0.125
STRUCTURE_FRACTION_OF_OUTER_TRAIN = 0.20
MEMBER_SEEDS = tuple(range(3101, 3109))
FINAL_MEMBER_CANDIDATES = (3, 4, 5, 6, 7)
RANK_DIAGNOSTIC_E = 2
SATURATION_REFERENCE_E = 8
OUTER_MEMBER_STARTS = (0, 3, 6, 1, 4)
TOP_K_CANDIDATES = (3, 4, 8, 16, 32, 64)
LOSS_TEMPERATURES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


LocationKey = tuple[int, int, float, float]


def cyclic_member_indices(
    member_count: int,
    start: int,
    pool_size: int = SATURATION_REFERENCE_E,
) -> tuple[int, ...]:
    """Return the registered identity-neutral cyclic member subset."""
    if isinstance(member_count, bool) or not 1 <= int(member_count) <= int(pool_size):
        raise ValueError("member_count must lie in [1,pool_size]")
    return tuple(
        (int(start) + offset) % int(pool_size)
        for offset in range(int(member_count))
    )


def _stable_digest(seed: int, salt: str, key: LocationKey) -> bytes:
    payload = (
        f"{int(seed)}|{salt}|{int(key[0])}|{int(key[1])}|"
        f"{float(key[2]):.4f}|{float(key[3]):.4f}"
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _strata(keys: Iterable[LocationKey]) -> dict[tuple[int, int], list[LocationKey]]:
    result: dict[tuple[int, int], list[LocationKey]] = {}
    for key in dict.fromkeys(keys):
        result.setdefault(key[:2], []).append(key)
    return result


def outer_group_assignment(
    keys: list[LocationKey],
    *,
    folds: int = OUTER_FOLDS,
    seed: int = SPLIT_SEED,
) -> dict[LocationKey, int]:
    """Assign every location group to one balanced stratified outer fold."""
    if isinstance(folds, bool) or int(folds) < 2:
        raise ValueError("folds must be an integer of at least two")
    fold_count = int(folds)
    assignment: dict[LocationKey, int] = {}
    for stratum, values in sorted(_strata(keys).items()):
        ordered = sorted(
            values,
            key=lambda key: _stable_digest(seed, f"outer-{stratum}", key),
        )
        # The stratum-specific rotation prevents every small remainder from
        # accumulating in fold zero while preserving deterministic balance.
        rotation = int.from_bytes(
            hashlib.sha256(
                f"{seed}|rotation|{stratum}".encode("utf-8")
            ).digest()[:4],
            "little",
        ) % fold_count
        for position, key in enumerate(ordered):
            assignment[key] = (position + rotation) % fold_count
    if len(assignment) != len(set(keys)):
        raise RuntimeError("outer assignment is not exhaustive over groups")
    return assignment


def _inner_counts(
    group_count: int,
    early_stop_fraction: float,
    structure_fraction: float,
) -> tuple[int, int]:
    if group_count <= 0:
        return 0, 0
    if not 0.0 < early_stop_fraction < 1.0:
        raise ValueError("early_stop_fraction must lie in (0,1)")
    if not 0.0 < structure_fraction < 1.0:
        raise ValueError("structure_fraction must lie in (0,1)")
    if early_stop_fraction + structure_fraction >= 1.0:
        raise ValueError("inner holdout fractions must sum to less than one")
    if group_count < 3:
        # Tiny generic-test strata remain train-only.  UJI strata are much
        # larger; the protocol audit records any role missing a stratum.
        return 0, 0
    stop_count = max(1, int(round(group_count * early_stop_fraction)))
    structure_count = max(1, int(round(group_count * structure_fraction)))
    while stop_count + structure_count >= group_count:
        if structure_count >= stop_count and structure_count > 1:
            structure_count -= 1
        elif stop_count > 1:
            stop_count -= 1
        else:
            break
    if stop_count + structure_count >= group_count:
        return 0, 0
    return stop_count, structure_count


def nested_outer_roles(
    keys: list[LocationKey],
    outer_fold: int,
    *,
    folds: int = OUTER_FOLDS,
    seed: int = SPLIT_SEED,
    early_stop_fraction: float = EARLY_STOP_FRACTION_OF_OUTER_TRAIN,
    structure_fraction: float = STRUCTURE_FRACTION_OF_OUTER_TRAIN,
) -> dict[str, np.ndarray]:
    """Return row indices for fit/stop/structure/outer-test roles."""
    fold_count = int(folds)
    if isinstance(outer_fold, bool) or not 0 <= int(outer_fold) < fold_count:
        raise ValueError("outer_fold must index one registered fold")
    fold = int(outer_fold)
    outer_assignment = outer_group_assignment(keys, folds=fold_count, seed=seed)
    role_by_key: dict[LocationKey, str] = {}
    for key, assigned_fold in outer_assignment.items():
        if assigned_fold == fold:
            role_by_key[key] = "outer_test"

    remaining = [key for key in dict.fromkeys(keys) if key not in role_by_key]
    for stratum, values in sorted(_strata(remaining).items()):
        ordered = sorted(
            values,
            key=lambda key: _stable_digest(
                seed, f"inner-fold-{fold}-{stratum}", key),
        )
        stop_count, structure_count = _inner_counts(
            len(ordered), early_stop_fraction, structure_fraction)
        stop_end = stop_count
        structure_end = stop_end + structure_count
        for key in ordered[:stop_end]:
            role_by_key[key] = "early_stop"
        for key in ordered[stop_end:structure_end]:
            role_by_key[key] = "structure_development"
        for key in ordered[structure_end:]:
            role_by_key[key] = "member_fit"

    role_names = (
        "member_fit",
        "early_stop",
        "structure_development",
        "outer_test",
    )
    indices = {
        role: np.asarray(
            [row for row, key in enumerate(keys) if role_by_key[key] == role],
            dtype=np.int64,
        )
        for role in role_names
    }
    _validate_roles(keys, indices)
    return indices


def _validate_roles(
    keys: list[LocationKey],
    roles: dict[str, np.ndarray],
) -> None:
    row_sets = {name: set(map(int, values)) for name, values in roles.items()}
    names = list(row_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if row_sets[left] & row_sets[right]:
                raise ValueError(f"row overlap between {left} and {right}")
    all_rows = set().union(*row_sets.values())
    if all_rows != set(range(len(keys))):
        raise ValueError("nested roles are not row-exhaustive")
    key_sets = {
        name: {keys[row] for row in values}
        for name, values in row_sets.items()
    }
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if key_sets[left] & key_sets[right]:
                raise ValueError(
                    f"reference-location leakage between {left} and {right}")


def _sha256_indices(values: np.ndarray) -> str:
    normalized = np.asarray(values, dtype="<i8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def protocol_audit(
    keys: list[LocationKey],
    timestamps: np.ndarray,
    *,
    folds: int = OUTER_FOLDS,
    seed: int = SPLIT_SEED,
) -> dict:
    """Build a label-free audit of every registered nested fold."""
    timestamp_values = np.asarray(timestamps)
    if timestamp_values.shape != (len(keys),):
        raise ValueError("timestamps must contain one value per row")
    fold_rows = []
    outer_test_group_counts: Counter[LocationKey] = Counter()
    for fold in range(int(folds)):
        roles = nested_outer_roles(keys, fold, folds=folds, seed=seed)
        role_payload = {}
        for role, indices in roles.items():
            role_keys = {keys[int(row)] for row in indices}
            strata = sorted({key[:2] for key in role_keys})
            times = timestamp_values[indices]
            role_payload[role] = {
                "rows": int(len(indices)),
                "coordinate_groups": int(len(role_keys)),
                "building_floor_strata": len(strata),
                "row_index_sha256": _sha256_indices(indices),
                "timestamp_min": (
                    float(np.min(times)) if len(times) else None),
                "timestamp_median": (
                    float(np.median(times)) if len(times) else None),
                "timestamp_max": (
                    float(np.max(times)) if len(times) else None),
            }
        for key in {keys[int(row)] for row in roles["outer_test"]}:
            outer_test_group_counts[key] += 1
        fold_rows.append({"outer_fold": fold, "roles": role_payload})
    unique_keys = set(keys)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "primary_estimand": (
            "unseen reference-location groups within the acquisition domain"),
        "outer_folds": int(folds),
        "split_seed": int(seed),
        "rows": len(keys),
        "coordinate_groups": len(unique_keys),
        "building_floor_strata": len({key[:2] for key in unique_keys}),
        "folds": fold_rows,
        "outer_test_partition": {
            "every_group_exactly_once": bool(
                set(outer_test_group_counts) == unique_keys
                and set(outer_test_group_counts.values()) == {1}
            ),
            "minimum_group_appearances": min(
                outer_test_group_counts.values(), default=0),
            "maximum_group_appearances": max(
                outer_test_group_counts.values(), default=0),
        },
        "selection_grid": {
            "member_count_E": list(FINAL_MEMBER_CANDIDATES),
            "rank_diagnostic_E": RANK_DIAGNOSTIC_E,
            "saturation_reference_E": SATURATION_REFERENCE_E,
            "top_k": list(TOP_K_CANDIDATES),
            "loss_temperature_lambda": list(LOSS_TEMPERATURES),
            "member_seeds": list(MEMBER_SEEDS),
            "outer_member_starts": list(OUTER_MEMBER_STARTS),
        },
    }


def validate_protocol_audit(audit: dict) -> None:
    """Fail closed if the registered real-data partition is incomplete."""
    if audit.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol audit version mismatch")
    partition = audit.get("outer_test_partition", {})
    if partition.get("every_group_exactly_once") is not True:
        raise ValueError("outer folds do not partition coordinate groups exactly once")
    if (
        int(partition.get("minimum_group_appearances", 0)) != 1
        or int(partition.get("maximum_group_appearances", 0)) != 1
    ):
        raise ValueError("outer coordinate-group appearance count is not one")
    expected_strata = int(audit.get("building_floor_strata", 0))
    if expected_strata <= 0:
        raise ValueError("protocol audit contains no building-floor strata")
    folds = audit.get("folds", [])
    if len(folds) != int(audit.get("outer_folds", -1)):
        raise ValueError("protocol audit fold count mismatch")
    required_roles = {
        "member_fit",
        "early_stop",
        "structure_development",
        "outer_test",
    }
    for fold in folds:
        roles = fold.get("roles", {})
        if set(roles) != required_roles:
            raise ValueError("protocol audit is missing a nested role")
        for role, values in roles.items():
            if int(values.get("rows", 0)) <= 0:
                raise ValueError(f"{role} is empty in a registered fold")
            if int(values.get("coordinate_groups", 0)) <= 0:
                raise ValueError(f"{role} has no coordinate groups")
            if int(values.get("building_floor_strata", 0)) != expected_strata:
                raise ValueError(
                    f"{role} does not cover every building-floor stratum")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the nested v3 protocol")
    parser.add_argument(
        "--data", default="data/UJIndoorLoc/trainingData.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data_path = resolve_path(args.data).resolve()
    (
        _rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        _phone_t,
        timestamp_t,
        stats,
    ) = load_dataset(data_path, dataset_type="UJI")
    xy = raw_coords(coords_norm, building_t, stats)
    keys = location_keys(
        xy, building_t.numpy(), floor_t.numpy())
    audit = protocol_audit(keys, timestamp_t.numpy())
    validate_protocol_audit(audit)
    audit["source_path"] = str(data_path)
    output = Path(args.output)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    _write_json(output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
