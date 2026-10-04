"""Registered training-only data roles for V18.5.

V18.5 intentionally uses a fresh preregistered grouped-coordinate split seed.
This is not claimed to restore benchmark blindness; it reduces direct reuse of
the exact V18.4 development/calibration partition while keeping the same
outcome-free hull-preserving split law.
"""

from __future__ import annotations

import json
from pathlib import Path

from fhm_dapp_tim.fhm_dapp.v18_4_protocol import (
    DEVELOPMENT_FRACTION,
    CALIBRATION_FRACTION,
    RolePartition,
    audit_roles,
    construct_roles,
    coordinate_keys,
    row_indices_for_roles,
)
from fhm_dapp_tim.fhm_dapp.v18_5_datasets import (
    DATA_ROOT,
    load_tampere,
    load_tuji1,
    load_uji,
    load_uts,
)

PROTOCOL_VERSION = "v18_5_multidataset_roles_v1_20261005"

ROLE_SEEDS = {
    "UJIIndoorLoc": 185501,
    "UTSIndoorLoc": 185502,
    "Tampere": 185503,
    "TUJI1": 185504,
}

TUJI_LODO_ROLE_SEEDS = {
    "A12": 185551,
    "POCO": 185552,
    "S20": 185553,
    "S7": 185554,
    "tabS7": 185555,
}


def roles_for_training_data(data) -> RolePartition:
    if data.dataset not in ROLE_SEEDS:
        raise ValueError(f"no V18.5 role seed for {data.dataset}")
    return construct_roles(data, seed=ROLE_SEEDS[data.dataset])


def prepare_manifest(data_root: Path = DATA_ROOT) -> dict[str, object]:
    datasets: dict[str, object] = {}
    for dataset in ("UJIIndoorLoc", "UTSIndoorLoc", "Tampere", "TUJI1"):
        if dataset == "UJIIndoorLoc":
            data = load_uji("train", data_root=data_root)[0]
        elif dataset == "UTSIndoorLoc":
            data = load_uts("train", data_root=data_root)[0]
        elif dataset == "Tampere":
            data = load_tampere("train", data_root=data_root)[0]
        else:
            data = load_tuji1("train", data_root=data_root)[0]
        roles = roles_for_training_data(data)
        audit = audit_roles(data, roles)
        if any(audit["missing_hull_support"].values()):
            raise RuntimeError(f"{dataset}: fit role lost hull support")
        datasets[dataset] = {
            "seed": int(ROLE_SEEDS[dataset]),
            "source_split": "train",
            "audit": audit,
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "development_fraction": DEVELOPMENT_FRACTION,
        "calibration_fraction": CALIBRATION_FRACTION,
        "scientific_note": (
            "V18.5 uses newly preregistered role seeds after V18.4 mechanism "
            "development. This reduces exact split reuse but is not described "
            "as restoring blind benchmark access."
        ),
        "datasets": datasets,
        "tuji1_lodo_role_seeds": {
            key: int(value)
            for key, value in TUJI_LODO_ROLE_SEEDS.items()
        },
    }


def main() -> None:
    print(json.dumps(
        prepare_manifest(), indent=2, sort_keys=True, allow_nan=False
    ))


__all__ = [
    "CALIBRATION_FRACTION",
    "DEVELOPMENT_FRACTION",
    "PROTOCOL_VERSION",
    "ROLE_SEEDS",
    "TUJI_LODO_ROLE_SEEDS",
    "RolePartition",
    "audit_roles",
    "coordinate_keys",
    "construct_roles",
    "prepare_manifest",
    "roles_for_training_data",
    "row_indices_for_roles",
]


if __name__ == "__main__":
    main()
