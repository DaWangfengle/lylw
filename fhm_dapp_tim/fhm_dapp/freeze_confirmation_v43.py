"""Freeze orthogonal coordinate/device confirmation roles for DR-JAM v4.4.

This artifact is deliberately label-blind with respect to model outcomes: it
uses phone IDs and coordinate identities only.  All confirmation phones and
all confirmation coordinates remain outside the common final fit at the same
time.  Their intersection is explicitly a dual-shift stress role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords, resolve_path
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.protocol_v4 import (
    PROTOCOL_VERSION,
    coordinate_axis_partition,
    sha256_indices,
    sha256_location_keys,
    validate_registry,
)


ARTIFACT_STATUS = "registered_v44_orthogonal_confirmation_roles"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _role(indices: np.ndarray, keys: list[tuple]) -> dict:
    values = np.asarray(indices, dtype=np.int64)
    return {
        "rows": int(len(values)),
        "coordinate_groups": int(len({keys[int(row)] for row in values})),
        "coordinate_group_sha256": sha256_location_keys(
            keys[int(row)] for row in values
        ),
        "row_index_sha256": sha256_indices(values),
    }


def build_confirmation_manifest(
    *,
    train_data: str | Path,
    registry_path: str | Path,
) -> dict:
    data_path = resolve_path(str(train_data)).resolve()
    registry = validate_registry(registry_path, train_data=data_path)
    (
        _rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        _timestamp_t,
        stats,
    ) = load_dataset(data_path, dataset_type="UJI")
    raw_xy = raw_coords(coords_norm, building_t, stats)
    building = building_t.numpy()
    floor = floor_t.numpy()
    phones = phone_t.numpy().astype(np.int64, copy=False)
    keys = location_keys(raw_xy, building, floor)

    development = tuple(map(int, registry["device_development_phone_ids"]))
    confirmation = tuple(map(
        int, registry["device_confirmation_selection"]["phone_ids"]
    ))
    coordinate_partition = coordinate_axis_partition(
        keys, phones, development
    )
    core_keys = set(coordinate_partition["development_core"])
    confirmation_keys = set(
        coordinate_partition["coordinate_confirmation"]
    )
    train_indices = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if int(phone) in development and keys[row] in core_keys
        ],
        dtype=np.int64,
    )
    coordinate_test_indices = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if int(phone) in development and keys[row] in confirmation_keys
        ],
        dtype=np.int64,
    )
    quarantine_indices = np.flatnonzero(np.isin(phones, confirmation)).astype(np.int64)
    role_sets = [
        set(map(int, values))
        for values in (train_indices, coordinate_test_indices, quarantine_indices)
    ]
    if any(
        role_sets[left] & role_sets[right]
        for left in range(len(role_sets))
        for right in range(left + 1, len(role_sets))
    ):
        raise RuntimeError("final fit and confirmation axes overlap")
    if set().union(*role_sets) != set(range(len(keys))):
        raise RuntimeError("final fit and two confirmation axes are not exhaustive")

    per_phone: dict[str, dict] = {}
    primary_phones = []
    for phone in confirmation:
        phone_rows = np.flatnonzero(phones == phone).astype(np.int64)
        matched = np.asarray(
            [row for row in phone_rows if keys[int(row)] in core_keys],
            dtype=np.int64,
        )
        dual_shift = np.asarray(
            [row for row in phone_rows if keys[int(row)] in confirmation_keys],
            dtype=np.int64,
        )
        novel_coordinate = np.asarray(
            [
                row
                for row in phone_rows
                if (
                    keys[int(row)] not in core_keys
                    and keys[int(row)] not in confirmation_keys
                )
            ],
            dtype=np.int64,
        )
        if len(matched) + len(dual_shift) + len(novel_coordinate) != len(phone_rows):
            raise RuntimeError(f"confirmation phone {phone} is not row-exhaustive")
        if len(matched):
            primary_phones.append(phone)
        per_phone[str(phone)] = {
            "all_rows": _role(phone_rows, keys),
            "matched_device_test": _role(matched, keys),
            "device_plus_registered_coordinate_shift_test": _role(
                dual_shift, keys
            ),
            "device_plus_confirmation_only_coordinate_stress": _role(
                novel_coordinate, keys
            ),
        }

    payload = {
        "artifact_status": ARTIFACT_STATUS,
        "protocol_version": PROTOCOL_VERSION,
        "created_without_localization_predictions_or_errors": True,
        "registry_path": str(Path(registry_path).resolve()),
        "registry_sha256": registry["registry_sha256"],
        "training_data": str(data_path),
        "training_data_sha256": _sha256_file(data_path),
        "training_contract": {
            "fit_phone_ids": list(development),
            "all_confirmation_phones_simultaneously_excluded": True,
            "all_coordinate_confirmation_groups_simultaneously_excluded": True,
            "fit_role": _role(train_indices, keys),
            "fixed_epoch_rule": (
                "for each selected seed, integer median of selected_epoch over "
                "the five coordinate-development folds for the frozen member mechanism"
            ),
            "uses_confirmation_rows_for_epoch_selection": False,
        },
        "coordinate_confirmation_contract": {
            "coordinate_group_count": len(confirmation_keys),
            "coordinate_group_sha256": sha256_location_keys(confirmation_keys),
            "development_phone_test_role": _role(
                coordinate_test_indices, keys
            ),
            "primary_aggregation": (
                "mean rows within coordinate group, then equal-coordinate macro"
            ),
            "selection_access_before_freeze": False,
        },
        "confirmation_contract": {
            "quarantined_phone_ids": list(confirmation),
            "quarantine_role": _role(quarantine_indices, keys),
            "primary_matched_device_phone_ids": primary_phones,
            "primary_aggregation": (
                "mean rows within coordinate, mean coordinates within phone, "
                "then equal-phone macro over matched-device roles"
            ),
            "dual_shift_aggregation": (
                "registered coordinate shift and confirmation-only coordinate "
                "stress are reported separately; descriptive only"
            ),
            "per_phone": per_phone,
        },
    }
    payload["artifact_sha256"] = _digest(payload)
    return payload


def validate_confirmation_manifest(
    path: str | Path,
    *,
    train_data: str | Path,
    registry_path: str | Path,
) -> dict:
    artifact_path = Path(path).resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("artifact_status") != ARTIFACT_STATUS:
        raise ValueError("confirmation-role artifact has no v4.4 authority")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("confirmation-role protocol mismatch")
    if payload.get("created_without_localization_predictions_or_errors") is not True:
        raise ValueError("confirmation-role provenance declaration is absent")
    if payload.get("artifact_sha256") != _digest(payload):
        raise ValueError("confirmation-role artifact digest mismatch")
    registry = validate_registry(registry_path, train_data=train_data)
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("confirmation roles and registry differ")
    data_path = resolve_path(str(train_data)).resolve()
    if payload.get("training_data_sha256") != _sha256_file(data_path):
        raise ValueError("confirmation source data changed")
    expected = build_confirmation_manifest(
        train_data=data_path, registry_path=registry_path
    )
    if payload != expected:
        raise ValueError("confirmation roles no longer reproduce exactly")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze orthogonal confirmation roles for DR-JAM v4.4"
    )
    parser.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv"
    )
    parser.add_argument(
        "--registry", default="fhm_dapp_tim/evidence/v44/registry_v4_4.json"
    )
    parser.add_argument(
        "--output",
        default="fhm_dapp_tim/evidence/v44/confirmation_roles_v4_4.json",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmation roles: {output}")
    payload = build_confirmation_manifest(
        train_data=args.train_data, registry_path=args.registry
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "artifact_sha256": payload["artifact_sha256"],
        "primary_matched_device_phone_ids": payload["confirmation_contract"][
            "primary_matched_device_phone_ids"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
