"""Registered data-role construction for V18.4.

Only dataset training partitions are used here. Official/provided evaluation
splits are intentionally not opened by protocol preparation.

Roles are complete physical-coordinate groups and are nested as:

training source = selection_refit + calibration
selection_refit = fit + development

Convex-hull support coordinates are always retained in fit. This protects
geometry from artificial extrapolation caused by role splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull, QhullError

try:
    from .v18_4_datasets import (
        CanonicalFingerprintData,
        DATA_ROOT,
        load_tampere,
        load_tuji1,
        load_uji,
        load_uts,
    )
except ImportError:
    from v18_4_datasets import (
        CanonicalFingerprintData,
        DATA_ROOT,
        load_tampere,
        load_tuji1,
        load_uji,
        load_uts,
    )


PROTOCOL_VERSION = "v18_4_multidataset_roles_v1_20261002"
DEVELOPMENT_FRACTION = 0.15
CALIBRATION_FRACTION = 0.15
ROLE_SEEDS = {
    "UJIIndoorLoc": 184401,
    "UTSIndoorLoc": 184402,
    "Tampere": 184403,
    "TUJI1": 184404,
}

TUJI_LODO_ROLE_SEEDS = {
    "A12": 184451,
    "POCO": 184452,
    "S20": 184453,
    "S7": 184454,
    "tabS7": 184455,
}

CoordinateKey = tuple[int, int, float, float]


@dataclass(frozen=True)
class RolePartition:
    fit: frozenset[CoordinateKey]
    development: frozenset[CoordinateKey]
    calibration: frozenset[CoordinateKey]

    @property
    def selection_refit(self) -> frozenset[CoordinateKey]:
        return self.fit | self.development

    @property
    def all_groups(self) -> frozenset[CoordinateKey]:
        return self.selection_refit | self.calibration


def coordinate_keys(data: CanonicalFingerprintData) -> list[CoordinateKey]:
    return [
        (
            int(building),
            int(floor),
            round(float(x), 4),
            round(float(y), 4),
        )
        for (x, y), building, floor in zip(
            data.coords_m,
            data.building,
            data.floor,
        )
    ]


def _stable_stratum_seed(
    seed: int,
    stratum: tuple[int, int],
) -> int:
    encoded = f"{int(seed)}:{int(stratum[0])}:{int(stratum[1])}".encode()
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _support_indices(points: np.ndarray) -> set[int]:
    """Return deterministic extremal points required to preserve the hull."""

    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n <= 3:
        return set(range(n))
    centered = points - points.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered, tol=1e-9))
    if rank >= 2:
        try:
            hull = ConvexHull(points)
            return set(map(int, hull.vertices.tolist()))
        except QhullError:
            pass

    spread = np.ptp(points, axis=0)
    axis = int(np.argmax(spread))
    return {
        int(np.argmin(points[:, axis])),
        int(np.argmax(points[:, axis])),
    }


def construct_roles(
    data: CanonicalFingerprintData,
    *,
    seed: int,
    development_fraction: float = DEVELOPMENT_FRACTION,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> RolePartition:
    if development_fraction < 0 or calibration_fraction < 0:
        raise ValueError("role fractions cannot be negative")
    if development_fraction + calibration_fraction >= 0.5:
        raise ValueError("development + calibration must remain below 0.5")

    unique_keys = list(dict.fromkeys(coordinate_keys(data)))
    strata: dict[tuple[int, int], list[CoordinateKey]] = {}
    for key in unique_keys:
        strata.setdefault(key[:2], []).append(key)

    fit: set[CoordinateKey] = set()
    development: set[CoordinateKey] = set()
    calibration: set[CoordinateKey] = set()

    for stratum, keys in sorted(strata.items()):
        ordered = sorted(keys)
        points = np.asarray(
            [[key[2], key[3]] for key in ordered],
            dtype=np.float64,
        )
        support_local = _support_indices(points)
        support = {ordered[index] for index in support_local}
        interior = [
            key
            for index, key in enumerate(ordered)
            if index not in support_local
        ]

        rng = np.random.default_rng(_stable_stratum_seed(seed, stratum))
        if interior:
            permutation = rng.permutation(len(interior))
            interior = [interior[int(index)] for index in permutation]

        n_groups = len(ordered)
        desired_dev = int(round(n_groups * development_fraction))
        desired_cal = int(round(n_groups * calibration_fraction))
        if n_groups >= 10:
            desired_dev = max(1, desired_dev)
            desired_cal = max(1, desired_cal)

        available = len(interior)
        n_cal = min(desired_cal, available)
        available -= n_cal
        n_dev = min(desired_dev, available)

        calibration.update(interior[:n_cal])
        development.update(interior[n_cal:n_cal + n_dev])
        fit.update(support)
        fit.update(interior[n_cal + n_dev:])

    result = RolePartition(
        fit=frozenset(fit),
        development=frozenset(development),
        calibration=frozenset(calibration),
    )
    if result.fit & result.development:
        raise RuntimeError("fit/development overlap")
    if result.fit & result.calibration:
        raise RuntimeError("fit/calibration overlap")
    if result.development & result.calibration:
        raise RuntimeError("development/calibration overlap")
    if result.all_groups != frozenset(unique_keys):
        raise RuntimeError("role partition does not reconstruct training groups")
    return result


def row_indices_for_roles(
    data: CanonicalFingerprintData,
    roles: RolePartition,
) -> dict[str, np.ndarray]:
    keys = coordinate_keys(data)
    sets = {
        "fit": roles.fit,
        "development": roles.development,
        "selection_refit": roles.selection_refit,
        "calibration": roles.calibration,
    }
    return {
        name: np.asarray(
            [
                index
                for index, key in enumerate(keys)
                if key in key_set
            ],
            dtype=np.int64,
        )
        for name, key_set in sets.items()
    }


def _strata(keys: Iterable[CoordinateKey]) -> set[tuple[int, int]]:
    return {key[:2] for key in keys}


def audit_roles(
    data: CanonicalFingerprintData,
    roles: RolePartition,
) -> dict[str, object]:
    rows = row_indices_for_roles(data, roles)
    training_strata = _strata(roles.all_groups)
    role_groups = {
        "fit": roles.fit,
        "development": roles.development,
        "selection_refit": roles.selection_refit,
        "calibration": roles.calibration,
    }
    missing_strata = {
        name: sorted(training_strata - _strata(groups))
        for name, groups in role_groups.items()
    }

    all_by_stratum: dict[tuple[int, int], list[CoordinateKey]] = {}
    for key in sorted(roles.all_groups):
        all_by_stratum.setdefault(key[:2], []).append(key)

    missing_hull_support: dict[str, int] = {}
    for stratum, keys in sorted(all_by_stratum.items()):
        points = np.asarray(
            [[key[2], key[3]] for key in keys],
            dtype=np.float64,
        )
        support = {keys[index] for index in _support_indices(points)}
        missing_hull_support[f"{stratum[0]}:{stratum[1]}"] = int(
            len(support - roles.fit)
        )

    return {
        "rows": {name: int(len(indices)) for name, indices in rows.items()},
        "coordinate_groups": {
            name: int(len(groups))
            for name, groups in role_groups.items()
        },
        "training_strata": [
            list(value) for value in sorted(training_strata)
        ],
        "missing_strata": {
            name: [list(value) for value in values]
            for name, values in missing_strata.items()
        },
        "missing_hull_support": missing_hull_support,
        "exact_union": bool(
            roles.fit | roles.development | roles.calibration
            == roles.all_groups
        ),
        "pairwise_disjoint": bool(
            not (roles.fit & roles.development)
            and not (roles.fit & roles.calibration)
            and not (roles.development & roles.calibration)
        ),
    }


def prepare_manifest(data_root: Path = DATA_ROOT) -> dict[str, object]:
    datasets = {}
    for name in ("UJIIndoorLoc", "UTSIndoorLoc", "Tampere", "TUJI1"):
        if name == "UJIIndoorLoc":
            data = load_uji("train", data_root=data_root)[0]
        elif name == "UTSIndoorLoc":
            data = load_uts("train", data_root=data_root)[0]
        elif name == "Tampere":
            data = load_tampere("train", data_root=data_root)[0]
        else:
            data = load_tuji1("train", data_root=data_root)[0]

        roles = construct_roles(data, seed=ROLE_SEEDS[name])
        audit = audit_roles(data, roles)
        if any(audit["missing_hull_support"].values()):
            raise RuntimeError(f"{name}: missing convex-hull support")
        datasets[name] = {
            "seed": int(ROLE_SEEDS[name]),
            "source_split": "train",
            "evaluation_status": (
                "development_only_preexposed_official"
                if name == "UJIIndoorLoc"
                else "provided_test_reserved_no_model_selection"
            ),
            "audit": audit,
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "development_fraction": DEVELOPMENT_FRACTION,
        "calibration_fraction": CALIBRATION_FRACTION,
        "datasets": datasets,
        "tuji1_cross_device_protocol": {
            "devices": ["A12", "POCO", "S20", "S7", "tabS7"],
            "role_seeds": {
                key: int(value)
                for key, value in TUJI_LODO_ROLE_SEEDS.items()
            },
            "rule": (
                "five leave-one-device-out evaluations: build AP dictionary "
                "and fit on four devices' training scans; evaluate held-out "
                "device only after global V18.4 method hyperparameters freeze"
            ),
        },
    }


def main() -> None:
    manifest = prepare_manifest()
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
