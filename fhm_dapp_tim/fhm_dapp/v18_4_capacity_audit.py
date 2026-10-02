"""Static candidate-capacity audit for V18.4.

This audit uses coordinates only.  It computes the widest positive non-self
Delaunay neighborhood for every registered training role and for each TUJI1
leave-one-device-out training pool.

The registered primary proposal quotas are 32 invariant + 32 prototype +
16 hierarchy/floor = 80.  A computational ceiling L_max is acceptable only if
it can contain those 80 primary proposals and the widest forced positive
Delaunay target simultaneously.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import Delaunay, QhullError

try:
    from .v18_4_datasets import (
        DATA_ROOT,
        CanonicalFingerprintData,
        load_tampere,
        load_tuji1,
        load_uji,
        load_uts,
    )
    from .v18_4_protocol import (
        ROLE_SEEDS,
        construct_roles,
        coordinate_keys,
    )
except ImportError:
    MODULE_DIR = Path(__file__).resolve().parent
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))
    from v18_4_datasets import (
        DATA_ROOT,
        CanonicalFingerprintData,
        load_tampere,
        load_tuji1,
        load_uji,
        load_uts,
    )
    from v18_4_protocol import (
        ROLE_SEEDS,
        construct_roles,
        coordinate_keys,
    )


PRIMARY_PROPOSAL_QUOTA = 32 + 32 + 16
REGISTERED_L_MAX = 128


def _delaunay_adjacency(points: np.ndarray) -> list[set[int]]:
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    adjacency = [set() for _ in range(n)]
    if n <= 1:
        return adjacency
    if n == 2:
        adjacency[0].add(1)
        adjacency[1].add(0)
        return adjacency

    centered = points - points.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered, tol=1e-9))
    if rank < 2:
        # Degenerate/collinear geometry: connect adjacent points along the
        # principal coordinate axis, matching the local-neighbor intent
        # without inventing dense all-to-all positives.
        spread = np.ptp(points, axis=0)
        axis = int(np.argmax(spread))
        order = np.argsort(points[:, axis], kind="stable")
        for left, right in zip(order[:-1], order[1:]):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
        return adjacency

    try:
        triangulation = Delaunay(points)
    except QhullError:
        triangulation = Delaunay(points, qhull_options="QJ")

    for simplex in np.asarray(triangulation.simplices, dtype=np.int64):
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                first = int(simplex[i])
                second = int(simplex[j])
                adjacency[first].add(second)
                adjacency[second].add(first)
    return adjacency


def geometry_width(keys: set[tuple[int, int, float, float]]) -> dict[str, object]:
    strata: dict[tuple[int, int], list[tuple[int, int, float, float]]] = {}
    for key in sorted(keys):
        strata.setdefault(key[:2], []).append(key)

    widths: dict[str, int] = {}
    hist: dict[int, int] = {}
    for stratum, values in sorted(strata.items()):
        points = np.asarray(
            [[value[2], value[3]] for value in values],
            dtype=np.float64,
        )
        adjacency = _delaunay_adjacency(points)
        maximum = max((len(row) for row in adjacency), default=0)
        widths[f"{stratum[0]}:{stratum[1]}"] = int(maximum)
        for row in adjacency:
            degree = int(len(row))
            hist[degree] = hist.get(degree, 0) + 1

    maximum = max(widths.values(), default=0)
    return {
        "coordinate_groups": int(len(keys)),
        "stratum_max_positive_nonself": widths,
        "maximum_positive_nonself": int(maximum),
        "degree_histogram": {
            str(key): int(value)
            for key, value in sorted(hist.items())
        },
        "required_capacity_with_primary_quota": int(
            PRIMARY_PROPOSAL_QUOTA + maximum
        ),
        "registered_l_max": REGISTERED_L_MAX,
        "capacity_margin": int(
            REGISTERED_L_MAX - PRIMARY_PROPOSAL_QUOTA - maximum
        ),
        "capacity_sufficient": bool(
            PRIMARY_PROPOSAL_QUOTA + maximum <= REGISTERED_L_MAX
        ),
    }


def _role_keys(
    data: CanonicalFingerprintData,
    dataset_name: str,
) -> dict[str, set[tuple[int, int, float, float]]]:
    roles = construct_roles(data, seed=ROLE_SEEDS[dataset_name])
    return {
        "fit": set(roles.fit),
        "selection_refit": set(roles.selection_refit),
        "full_training": set(roles.all_groups),
    }


def _load_training(name: str, data_root: Path) -> CanonicalFingerprintData:
    if name == "UJIIndoorLoc":
        return load_uji("train", data_root=data_root)[0]
    if name == "UTSIndoorLoc":
        return load_uts("train", data_root=data_root)[0]
    if name == "Tampere":
        return load_tampere("train", data_root=data_root)[0]
    if name == "TUJI1":
        return load_tuji1("train", data_root=data_root)[0]
    raise ValueError(name)


def capacity_manifest(data_root: Path = DATA_ROOT) -> dict[str, object]:
    datasets = {}
    for name in ("UJIIndoorLoc", "UTSIndoorLoc", "Tampere", "TUJI1"):
        data = _load_training(name, data_root)
        roles = _role_keys(data, name)
        datasets[name] = {
            role: geometry_width(keys)
            for role, keys in roles.items()
        }

    devices = ("A12", "POCO", "S20", "S7", "tabS7")
    lodo = {}
    for held_out in devices:
        training_devices = tuple(
            device for device in devices if device != held_out
        )
        data = load_tuji1(
            "train",
            devices=training_devices,
            data_root=data_root,
        )[0]
        keys = set(coordinate_keys(data))
        lodo[held_out] = {
            "training_devices": list(training_devices),
            **geometry_width(keys),
        }

    manifest = {
        "audit_version": "v18_4_capacity_v1_20261002",
        "primary_proposal_quota": PRIMARY_PROPOSAL_QUOTA,
        "registered_l_max": REGISTERED_L_MAX,
        "datasets": datasets,
        "tuji1_leave_one_device_out": lodo,
    }

    failures = []
    for dataset_name, roles in datasets.items():
        for role_name, record in roles.items():
            if not record["capacity_sufficient"]:
                failures.append(f"{dataset_name}:{role_name}")
    for device, record in lodo.items():
        if not record["capacity_sufficient"]:
            failures.append(f"TUJI1:LODO:{device}")
    manifest["failed_capacity_contracts"] = failures
    manifest["all_capacity_contracts_pass"] = not failures
    return manifest


def main() -> None:
    print(
        json.dumps(
            capacity_manifest(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
