"""TUJI1 data adapter and pre-result protocol registrar for external V9.

This module deliberately contains no neural training or test prediction path.
It makes the TUJI1 CSV contract explicit, constructs the training-only nested
selection roles, and freezes an auditable registry before a later V9 runner is
allowed to use the author testing split.

The adapter preserves the V9 mathematical interface:
``rss`` is normalized to ``[-1, 1]`` with the missing event at ``-1``;
``raw_xy`` remains in metres; a coordinate key contains the one building and
floor identifiers together with an exact six-decimal physical coordinate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from fhm_dapp_tim.fhm_dapp.audit_bplus_sparse50_fem_v5 import (
    _hull_local_indices,
    geometry_support_with_hull,
)
from fhm_dapp_tim.fhm_dapp.experiment_v7 import support_fit_stop_roles
from fhm_dapp_tim.fhm_dapp.uji_v9_experiment import (
    CANDIDATES,
    COMMON_FRONT,
    CONCENTRATION_GRID,
    E_GRID,
    HISTORICAL_FRONT,
    K_GRID,
    LAMBDA_GRID,
    MEMBER_SEEDS,
)


PROTOCOL_VERSION = "tuji_v9_external_replication_v1_20260812"
SELECTION_GEOMETRY_SEED = 9611
EARLY_STOP_SEED = 9612
SCORING_SEED = 9613
PAIRED_INFERENCE_SEED = 9614
SELECTION_REFIT_FRACTION = 0.80
EARLY_STOP_FRACTION = 0.15
VISIBLE_THRESHOLD = -0.999
MAX_VISIBLE = int(COMMON_FRONT["max_visible"])
BOOTSTRAP_REPETITIONS = 5000
STUDENT_DF = float(COMMON_FRONT["student_df"])

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "TUJI1" / "DATASET"
DEFAULT_EVIDENCE = ROOT / "evidence" / "v9_tuji"
DEFAULT_DESIGN = DEFAULT_EVIDENCE / "TUJI1_V9_EXTERNAL_PROTOCOL_V1.md"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "registry_tuji_v9.json"
DEFAULT_FREEZE = DEFAULT_EVIDENCE / "selection_freeze_tuji_v9.json"
DEFAULT_TEST_CACHE = DEFAULT_EVIDENCE / "author_test_cache_tuji_v9.pt"
DEFAULT_RESULT = DEFAULT_EVIDENCE / "author_test_result_tuji_v9.json"

CoordinateKey = tuple[int, int, float, float]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_json_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key_json(key: CoordinateKey) -> list[int | float]:
    return [int(key[0]), int(key[1]), float(key[2]), float(key[3])]


def _sha256_keys(keys: Sequence[CoordinateKey] | set[CoordinateKey]) -> str:
    ordered = [_key_json(key) for key in sorted(keys)]
    return hashlib.sha256(_canonical_bytes(ordered)).hexdigest()


def _sha256_indices(rows: Sequence[int] | np.ndarray) -> str:
    value = np.asarray(rows, dtype=np.int64)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _rows_for_keys(
    data_keys: Sequence[CoordinateKey], selected_keys: set[CoordinateKey]
) -> np.ndarray:
    return np.flatnonzero(
        np.asarray([key in selected_keys for key in data_keys], dtype=bool)
    ).astype(np.int64, copy=False)


def _role_record(
    keys: set[CoordinateKey], data_keys: Sequence[CoordinateKey]
) -> dict[str, object]:
    rows = _rows_for_keys(data_keys, keys)
    return {
        "coordinate_count": int(len(keys)),
        "row_count": int(len(rows)),
        "coordinate_keys": [_key_json(key) for key in sorted(keys)],
        "coordinate_key_sha256": _sha256_keys(keys),
        "row_index_sha256": _sha256_indices(rows),
    }


def _decode_key(value: Sequence[object]) -> CoordinateKey:
    if len(value) != 4:
        raise ValueError("a coordinate key must have four values")
    return (int(value[0]), int(value[1]), float(value[2]), float(value[3]))


def _write_json_new(path: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _matrix(path: str | Path, *, expected_columns: int) -> np.ndarray:
    """Read a headerless numeric CSV and reject a silently dropped first row."""

    matrix = np.loadtxt(Path(path), delimiter=",", dtype=np.float64, ndmin=2)
    if matrix.ndim != 2 or matrix.shape[1] != int(expected_columns):
        raise ValueError(
            f"{path} must be a headerless [rows,{expected_columns}] matrix; "
            f"got {tuple(matrix.shape)}"
        )
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{path} contains a non-finite value")
    return matrix


def _device_label_map(path: str | Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"Device", "Label"}:
        raise ValueError("Device_labels.csv must have exactly Device,Label columns")
    mapping: dict[int, str] = {}
    for row in rows:
        label = int(row["Label"])
        name = str(row["Device"]).strip()
        if label in mapping or not name:
            raise ValueError("Device_labels.csv has a duplicate or empty device label")
        mapping[label] = name
    if sorted(mapping) != list(range(1, len(mapping) + 1)):
        raise ValueError("TUJI device labels must be contiguous one-based identifiers")
    if len(mapping) != 5:
        raise ValueError(f"TUJI V9 expects the five published devices, got {len(mapping)}")
    return mapping


def _coordinate_bounds_from_stats(stats: Mapping[str, object]) -> dict[str, float]:
    raw = stats.get("building_stats")
    if not isinstance(raw, Mapping):
        raise ValueError("reference stats have no building_stats")
    bounds = raw.get(0, raw.get("0"))
    if not isinstance(bounds, Mapping):
        raise ValueError("reference stats have no bounds for the TUJI building")
    required = ("x_min", "x_max", "y_min", "y_max")
    if set(required) - set(bounds):
        raise ValueError("reference coordinate bounds are incomplete")
    result = {name: float(bounds[name]) for name in required}
    if not result["x_max"] > result["x_min"] or not result["y_max"] > result["y_min"]:
        raise ValueError("reference coordinate bounds are degenerate")
    return result


def _normalized_rss(raw_rss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not bool(np.equal(raw_rss, np.round(raw_rss)).all()):
        raise ValueError("TUJI RSS values must be integer dBm or sentinel 100")
    missing = raw_rss == 100.0
    observed = raw_rss[~missing]
    if len(observed) == 0 or float(observed.min()) < -110.0 or float(observed.max()) > 0.0:
        raise ValueError("TUJI observed RSS falls outside the documented dBm range")
    converted = raw_rss.astype(np.float32, copy=True)
    converted[missing] = -110.0
    converted = np.clip(converted, -110.0, -30.0)
    normalized = (converted + 110.0) / 80.0 * 2.0 - 1.0
    if not bool(np.isfinite(normalized).all()):
        raise RuntimeError("RSS normalization produced a non-finite value")
    if not bool(np.equal(normalized[missing], -1.0).all()):
        raise RuntimeError("missing RSS did not map exactly to the V9 sentinel")
    return normalized.astype(np.float32, copy=False), missing


def load_tuji_split(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    split: str,
    reference_stats: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load one author-provided TUJI1 split into the V9 data contract.

    ``reference_stats`` must come from the author training split when loading
    testing data; no test coordinate bound may be fitted at inference time.
    """

    name = str(split).strip().lower()
    suffix_by_split = {"training": "training", "testing": "testing"}
    if name not in suffix_by_split:
        raise ValueError("split must be exactly 'training' or 'testing'")
    suffix = suffix_by_split[name]
    root = Path(data_dir).resolve()
    rss_path = root / f"RSS_{suffix}.csv"
    coordinates_path = root / f"Coordinates_{suffix}.csv"
    devices_path = root / "Device_labels.csv"
    for path in (rss_path, coordinates_path, devices_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_rss = _matrix(rss_path, expected_columns=310)
    labels = _matrix(coordinates_path, expected_columns=6)
    if len(raw_rss) != len(labels):
        raise ValueError(
            f"TUJI {name} RSS/coordinate row mismatch: {len(raw_rss)} vs {len(labels)}"
        )
    if not bool(np.equal(labels[:, 2:5], 0.0).all()):
        raise ValueError("TUJI V9 is a 2-D experiment: label columns 3--5 must be zero")
    device_map = _device_label_map(devices_path)
    device_labels = labels[:, 5]
    if not bool(np.equal(device_labels, np.round(device_labels)).all()):
        raise ValueError("TUJI device labels must be integer-valued")
    device_labels = device_labels.astype(np.int64, copy=False)
    if set(map(int, np.unique(device_labels))) != set(device_map):
        raise ValueError("TUJI coordinate device labels do not match Device_labels.csv")

    normalized_rss, missing = _normalized_rss(raw_rss)
    raw_xy64 = labels[:, :2].astype(np.float64, copy=False)
    if reference_stats is None:
        bounds = {
            "x_min": float(raw_xy64[:, 0].min()),
            "x_max": float(raw_xy64[:, 0].max()),
            "y_min": float(raw_xy64[:, 1].min()),
            "y_max": float(raw_xy64[:, 1].max()),
        }
        bounds_source = "this_split"
    else:
        bounds = _coordinate_bounds_from_stats(reference_stats)
        bounds_source = "author_training_reference_stats"
    x_norm = 2.0 * (raw_xy64[:, 0] - bounds["x_min"]) / (
        bounds["x_max"] - bounds["x_min"]
    ) - 1.0
    y_norm = 2.0 * (raw_xy64[:, 1] - bounds["y_min"]) / (
        bounds["y_max"] - bounds["y_min"]
    ) - 1.0
    coords_norm = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    raw_xy = raw_xy64.astype(np.float32)
    keys: list[CoordinateKey] = [
        (0, 0, round(float(x), 6), round(float(y), 6))
        for x, y in raw_xy64
    ]
    if not keys or len(set(keys)) == 0:
        raise RuntimeError("TUJI split has no coordinate key")

    # Use zero-based IDs internally, while retaining the published one-based
    # label/name mapping in stats for auditability.
    phone = device_labels - 1
    stats: dict[str, object] = {
        "dataset_type": "TUJI1",
        "n_aps": int(raw_rss.shape[1]),
        "wap_cols": list(range(int(raw_rss.shape[1]))),
        "n_buildings": 1,
        "n_floors": 1,
        "building_stats": {0: bounds},
        "coordinate_bounds_source": bounds_source,
        "rss_missing_sentinel_raw": 100,
        "rss_missing_sentinel_normalized": -1.0,
        "phone_label_to_name": {
            str(label): device_map[label] for label in sorted(device_map)
        },
        "phone_mapping": {
            device_map[label]: int(label - 1) for label in sorted(device_map)
        },
    }
    return {
        "rss": torch.from_numpy(normalized_rss).float(),
        "coords_norm": torch.from_numpy(coords_norm).float(),
        "raw_xy": raw_xy,
        "building_t": torch.zeros(len(keys), dtype=torch.long),
        "floor_t": torch.zeros(len(keys), dtype=torch.long),
        "building": np.zeros(len(keys), dtype=np.int64),
        "floor": np.zeros(len(keys), dtype=np.int64),
        "phone_t": torch.from_numpy(phone.astype(np.int64, copy=False)).long(),
        "phone": phone.astype(np.int64, copy=False),
        "timestamp_t": torch.zeros(len(keys), dtype=torch.long),
        "keys": keys,
        "stats": stats,
        "raw_file_paths": {
            "rss": str(rss_path),
            "coordinates": str(coordinates_path),
            "device_labels": str(devices_path),
        },
        "raw_visible_count": (~missing).sum(axis=1).astype(np.int64),
    }


def load_tuji_author_splits(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[dict[str, object], dict[str, object]]:
    """Load author training first and force testing to reuse its bounds."""

    training = load_tuji_split(data_dir, split="training")
    testing = load_tuji_split(
        data_dir, split="testing", reference_stats=training["stats"]
    )
    if int(training["stats"]["n_aps"]) != int(testing["stats"]["n_aps"]):
        raise RuntimeError("TUJI train/test AP dimensions disagree")
    return training, testing


def _strata(keys: set[CoordinateKey]) -> dict[tuple[int, int], set[CoordinateKey]]:
    groups: dict[tuple[int, int], set[CoordinateKey]] = defaultdict(set)
    for key in keys:
        groups[(int(key[0]), int(key[1]))].add(key)
    return dict(groups)


def _nearest_support(
    query: set[CoordinateKey], reference: set[CoordinateKey]
) -> dict[str, object]:
    query_by_stratum = _strata(query)
    reference_by_stratum = _strata(reference)
    values: list[float] = []
    detail: dict[str, object] = {}
    for stratum, query_keys in sorted(query_by_stratum.items()):
        reference_keys = reference_by_stratum.get(stratum)
        if not reference_keys:
            raise RuntimeError(f"reference misses stratum {stratum}")
        query_xy = np.asarray([[key[2], key[3]] for key in sorted(query_keys)])
        reference_xy = np.asarray(
            [[key[2], key[3]] for key in sorted(reference_keys)]
        )
        distances, _ = cKDTree(reference_xy).query(query_xy, k=1)
        values.extend(np.asarray(distances, dtype=np.float64).tolist())
        detail[f"{stratum[0]}:{stratum[1]}"] = {
            "coordinates": int(len(distances)),
            "mean_m": float(np.mean(distances)),
            "maximum_m": float(np.max(distances)),
        }
    vector = np.asarray(values, dtype=np.float64)
    return {
        "coordinates": int(len(vector)),
        "mean_m": float(vector.mean()),
        "median_m": float(np.median(vector)),
        "q90_m": float(np.quantile(vector, 0.90)),
        "maximum_m": float(vector.max()),
        "per_stratum": detail,
    }


def _coordinate_device_summary(
    data: Mapping[str, object], selected_keys: set[CoordinateKey] | None = None
) -> dict[str, object]:
    keys = data["keys"]
    phone = np.asarray(data["phone"], dtype=np.int64)
    groups: dict[CoordinateKey, list[int]] = defaultdict(list)
    for key, device in zip(keys, phone):
        if selected_keys is None or key in selected_keys:
            groups[key].append(int(device))
    if not groups:
        raise RuntimeError("selected role has no coordinate group")
    device_per_coordinate = Counter(len(set(values)) for values in groups.values())
    scans_per_coordinate = Counter(len(values) for values in groups.values())
    all_devices = len(np.unique(phone))
    return {
        "coordinates": int(len(groups)),
        "rows": int(sum(map(len, groups.values()))),
        "device_count_distribution": {
            str(key): int(value) for key, value in sorted(device_per_coordinate.items())
        },
        "scan_count_distribution": {
            str(key): int(value) for key, value in sorted(scans_per_coordinate.items())
        },
        "coordinates_with_all_published_devices": int(
            sum(len(set(values)) == all_devices for values in groups.values())
        ),
        "coordinates_with_at_least_two_devices": int(
            sum(len(set(values)) >= 2 for values in groups.values())
        ),
    }


def _visibility_summary(data: Mapping[str, object]) -> dict[str, object]:
    count = np.asarray(data["raw_visible_count"], dtype=np.int64)
    return {
        "rows": int(len(count)),
        "minimum": int(count.min()),
        "median": float(np.median(count)),
        "q95": float(np.quantile(count, 0.95)),
        "maximum": int(count.max()),
        "above_max_visible": int((count > MAX_VISIBLE).sum()),
        "above_max_visible_fraction": float((count > MAX_VISIBLE).mean()),
        "max_visible": MAX_VISIBLE,
    }


def audit_tuji_author_splits(
    training: Mapping[str, object], testing: Mapping[str, object]
) -> dict[str, object]:
    """Audit the author splits without producing any model outcome."""

    train_keys = set(training["keys"])
    test_keys = set(testing["keys"])
    overlap = train_keys & test_keys
    if overlap:
        raise RuntimeError(f"TUJI author train/test coordinate leakage: {len(overlap)}")
    if int(training["stats"]["n_aps"]) != int(testing["stats"]["n_aps"]):
        raise RuntimeError("TUJI author splits have unequal AP dimensions")
    train_devices = set(map(int, np.asarray(training["phone"])))
    test_devices = set(map(int, np.asarray(testing["phone"])))
    if train_devices != test_devices:
        raise RuntimeError("TUJI author train/test device support differs")
    return {
        "status": "passed_no_model_outcomes",
        "author_train": {
            "rows": int(len(training["keys"])),
            "coordinate_count": int(len(train_keys)),
            "n_aps": int(training["stats"]["n_aps"]),
            "device_rows": {
                str(key): int(value)
                for key, value in sorted(Counter(map(int, training["phone"])).items())
            },
            "coordinate_device_summary": _coordinate_device_summary(training),
            "visibility": _visibility_summary(training),
        },
        "author_test": {
            "rows": int(len(testing["keys"])),
            "coordinate_count": int(len(test_keys)),
            "n_aps": int(testing["stats"]["n_aps"]),
            "device_rows": {
                str(key): int(value)
                for key, value in sorted(Counter(map(int, testing["phone"])).items())
            },
            "coordinate_device_summary": _coordinate_device_summary(testing),
            "visibility": _visibility_summary(testing),
        },
        "coordinate_intersection_count": 0,
        "test_to_train_nearest_support_m": _nearest_support(test_keys, train_keys),
        "test_coordinate_bounds_source": testing["stats"]["coordinate_bounds_source"],
    }


def construct_selection_roles(
    all_keys: set[CoordinateKey],
) -> tuple[dict[str, set[CoordinateKey]], dict[str, object]]:
    """Create training-only selection roles; author testing is never split here."""

    if not all_keys:
        raise ValueError("all_keys must be nonempty")
    selection_refit, selection_details = geometry_support_with_hull(
        all_keys, SELECTION_REFIT_FRACTION, seed=SELECTION_GEOMETRY_SEED
    )
    selection_refit = set(selection_refit)
    development = set(all_keys) - selection_refit
    nested, early_details = support_fit_stop_roles(
        selection_refit,
        seed=EARLY_STOP_SEED,
        early_stop_fraction=EARLY_STOP_FRACTION,
    )
    roles = {
        "member_fit": set(nested["member_fit"]),
        "early_stop": set(nested["early_stop"]),
        "selection_refit": selection_refit,
        "development": development,
        "final_refit": set(all_keys),
    }
    return roles, {
        "selection_refit": selection_details,
        "early_stop": early_details,
        "final_refit": "all author training coordinates after selection freeze",
    }


def _role_diagnostics(
    role_keys: set[CoordinateKey], data: Mapping[str, object]
) -> dict[str, object]:
    selected = np.asarray([key in role_keys for key in data["keys"]], dtype=bool)
    phone = np.asarray(data["phone"], dtype=np.int64)[selected]
    return {
        "coordinates": int(len(role_keys)),
        "rows": int(selected.sum()),
        "phone_row_counts": {
            str(key): int(value) for key, value in sorted(Counter(map(int, phone)).items())
        },
        "coordinate_device_summary": _coordinate_device_summary(data, role_keys),
    }


def audit_selection_roles(
    roles: Mapping[str, set[CoordinateKey]], data: Mapping[str, object]
) -> dict[str, object]:
    """Fail closed on leakage, role-union error, hull loss, or no device LOO."""

    all_keys = set(data["keys"])
    intersections = {
        "member_fit_early_stop": len(roles["member_fit"] & roles["early_stop"]),
        "selection_refit_development": len(
            roles["selection_refit"] & roles["development"]
        ),
    }
    if any(intersections.values()):
        raise RuntimeError(f"TUJI coordinate leakage detected: {intersections}")
    if roles["member_fit"] | roles["early_stop"] != roles["selection_refit"]:
        raise RuntimeError("member_fit and early_stop do not reconstruct selection_refit")
    if roles["selection_refit"] | roles["development"] != all_keys:
        raise RuntimeError("selection_refit and development do not reconstruct training")
    if roles["final_refit"] != all_keys:
        raise RuntimeError("final_refit must contain every author training coordinate")

    source_pairs = (
        ("author_training", all_keys, "selection_refit", roles["selection_refit"]),
        ("selection_refit", roles["selection_refit"], "member_fit", roles["member_fit"]),
    )
    hull_checks: dict[str, int] = {}
    for source_name, source, reference_name, reference in source_pairs:
        missing = 0
        for stratum, stratum_keys in _strata(source).items():
            ordered = sorted(stratum_keys)
            xy = np.asarray([[key[2], key[3]] for key in ordered], dtype=np.float64)
            hull = {ordered[index] for index in _hull_local_indices(xy)}
            missing += len(hull - reference)
        key = f"{source_name}_hull_missing_from_{reference_name}"
        hull_checks[key] = int(missing)
        if missing:
            raise RuntimeError(f"{reference_name} misses {missing} required hull coordinates")

    # The radio selector requires at least one anchor observed by two phones;
    # the count is otherwise reported rather than thresholded post hoc.
    for name in ("member_fit", "selection_refit", "final_refit"):
        summary = _coordinate_device_summary(data, roles[name])
        if int(summary["coordinates_with_at_least_two_devices"]) < 1:
            raise RuntimeError(f"{name} has no multi-device anchor for radio LOO")

    return {
        "status": "passed_no_model_outcomes",
        "grouping": "complete (building=0,floor=0,x,y) coordinate groups",
        "disjoint_intersections": intersections,
        "exact_union_identities": True,
        "hull_checks": hull_checks,
        "roles": {
            name: _role_diagnostics(set(keys), data) for name, keys in roles.items()
        },
        "development_to_selection_refit": _nearest_support(
            roles["development"], roles["selection_refit"]
        ),
    }


def _source_manifest(data_dir: Path) -> dict[str, dict[str, object]]:
    files = {
        "rss_training": data_dir / "RSS_training.csv",
        "coordinates_training": data_dir / "Coordinates_training.csv",
        "rss_testing": data_dir / "RSS_testing.csv",
        "coordinates_testing": data_dir / "Coordinates_testing.csv",
        "device_labels": data_dir / "Device_labels.csv",
        "matlab_reference": data_dir / "TUJI1.mat",
    }
    manifest: dict[str, dict[str, object]] = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest[name] = {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    return manifest


def _code_manifest() -> dict[str, dict[str, str]]:
    paths = {
        "tuji_protocol": Path(__file__).resolve(),
        "v9_experiment": ROOT / "fhm_dapp" / "uji_v9_experiment.py",
        "v9_marked_math": ROOT / "fhm_dapp" / "coherent_marked_v9.py",
        "marked_memory": ROOT / "fhm_dapp" / "experiment_v71.py",
        "front_model": ROOT / "fhm_dapp" / "model.py",
    }
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _registry_without_hash(registry: Mapping[str, object]) -> dict[str, object]:
    value = copy.deepcopy(dict(registry))
    value.pop("registry_sha256", None)
    return value


def register_protocol(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    design_path: str | Path = DEFAULT_DESIGN,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict[str, object]:
    """Write a self-hashed TUJI registry only if no result artifact exists."""

    source = Path(data_dir).resolve()
    design = Path(design_path).resolve()
    destination = Path(output).resolve()
    if not design.is_file():
        raise FileNotFoundError(design)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite registry: {destination}")
    planned_outputs = (DEFAULT_FREEZE.resolve(), DEFAULT_TEST_CACHE.resolve(), DEFAULT_RESULT.resolve())
    existing = [str(path) for path in planned_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "cannot register after a planned result artifact already exists: "
            + ", ".join(existing)
        )

    training, testing = load_tuji_author_splits(source)
    data_audit = audit_tuji_author_splits(training, testing)
    roles, split_details = construct_selection_roles(set(training["keys"]))
    role_audit = audit_selection_roles(roles, training)
    source_manifest = _source_manifest(source)
    registry: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_status": (
            "external-site V9 replication on TUJI1 after a data-eligibility audit; "
            "no model predictions or outcome scores have been accessed"
        ),
        "created_after_tuji_test_metadata_access": True,
        "test_exposure_before_protocol": {
            "accessed": [
                "file schemas and hashes",
                "true-coordinate group counts and train/test disjointness",
                "device-label coverage",
                "visible-AP count distribution and token-cap compatibility",
            ],
            "not_accessed": [
                "neural predictions",
                "point errors or MAE",
                "proper scores or likelihood scores",
                "parameter, candidate, or member-count outcomes",
            ],
            "model_outcome_accesses_before_freeze": 0,
        },
        "selection_authority_for_v9_uji": False,
        "may_modify_v9_uji": False,
        "comparison_status": "external_replication_after_eligibility_audit",
        "design": str(design),
        "design_sha256": _sha256_file(design),
        "source_files": source_manifest,
        "code_sources": _code_manifest(),
        "data_contract": {
            "csv_header": "none; first row is a valid scan",
            "n_aps": int(training["stats"]["n_aps"]),
            "raw_missing_sentinel": 100,
            "normalized_missing_sentinel": -1.0,
            "visible_threshold": VISIBLE_THRESHOLD,
            "max_visible_tokens": MAX_VISIBLE,
            "coordinate_schema": "(x,y,0,0,0,one_based_device_label)",
            "physical_strata": "one building and one floor, both encoded as zero",
            "query_device_label_use": "forbidden; train device labels are offline radio-memory grouping only",
            "test_coordinate_use": "forbidden until post-freeze scoring",
        },
        "author_split_audit": data_audit,
        "training_coordinate_sha256": _sha256_keys(set(training["keys"])),
        "roles": {
            name: _role_record(set(keys), training["keys"])
            for name, keys in roles.items()
        },
        "split_details": split_details,
        "split_audit": role_audit,
        "constants": {
            "selection_geometry_seed": SELECTION_GEOMETRY_SEED,
            "early_stop_seed": EARLY_STOP_SEED,
            "scoring_seed": SCORING_SEED,
            "paired_inference_seed": PAIRED_INFERENCE_SEED,
            "selection_refit_fraction": SELECTION_REFIT_FRACTION,
            "early_stop_fraction": EARLY_STOP_FRACTION,
            "member_seeds": list(MEMBER_SEEDS),
            "student_df": STUDENT_DF,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        },
        "front": copy.deepcopy(COMMON_FRONT),
        "historical_front": copy.deepcopy(HISTORICAL_FRONT),
        "candidates": list(CANDIDATES),
        "joint_grid": {
            "E": list(E_GRID),
            "K": list(K_GRID),
            "lambda": list(LAMBDA_GRID),
            "member_rule": "preregistered_seed_prefix",
        },
        "radio_memory": {
            "concentration_grid": list(CONCENTRATION_GRID),
            "degrees_freedom_grid": [STUDENT_DF],
            "selection_role": "selection_refit",
            "selection_outcome_access": "none: radio observations and device IDs only",
        },
        "selection_and_refit_contract": {
            "selection_role": "development",
            "final_refit_role": "all author training coordinates",
            "author_testing_role": "one final external evaluation only",
            "author_testing_refit_required": True,
            "author_testing_allowed_before_selection_freeze": False,
            "author_testing_model_outcome_accesses_before_freeze": 0,
            "final_refit_uses_author_testing_labels": False,
        },
        "evaluation_contract": {
            "primary": "coordinate-macro continuous-mixture energy score",
            "secondary": "coordinate-macro horizontal MAE",
            "bootstrap_unit": "physical test coordinate",
            "floor_and_3d_reporting": "not applicable: TUJI labels are one 2-D plane",
            "test_query_inputs": "RSS_testing.csv only",
            "post_prediction_labels": "coordinates for score; device label for prespecified stratification only",
        },
        "planned_output_paths": {
            "selection_freeze": str(DEFAULT_FREEZE.resolve()),
            "author_test_cache": str(DEFAULT_TEST_CACHE.resolve()),
            "author_test_result": str(DEFAULT_RESULT.resolve()),
        },
    }
    registry["registry_sha256"] = _canonical_json_sha256(
        _registry_without_hash(registry)
    )
    _write_json_new(destination, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, object]:
    """Verify registry, protocol, data and code hashes before execution."""

    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a TUJI V9 registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(
        _registry_without_hash(registry)
    ):
        raise ValueError("TUJI V9 registry self-hash failed")
    design = Path(str(registry["design"]))
    if _sha256_file(design) != registry.get("design_sha256"):
        raise ValueError("protocol design changed after registration")
    source_files = registry.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValueError("registry has no source-file manifest")
    for name, payload in source_files.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid source manifest entry {name}")
        if _sha256_file(str(payload["path"])) != payload["sha256"]:
            raise ValueError(f"source data changed after registration: {name}")
    sources = registry.get("code_sources")
    if not isinstance(sources, Mapping):
        raise ValueError("registry has no code-source manifest")
    for name, payload in sources.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid code manifest entry {name}")
        if _sha256_file(str(payload["path"])) != payload["sha256"]:
            raise ValueError(f"code changed after registration: {name}")
    return registry


def registered_role_keys(registry: Mapping[str, object], role: str) -> set[CoordinateKey]:
    roles = registry.get("roles")
    if not isinstance(roles, Mapping) or role not in roles:
        raise ValueError(f"unknown TUJI role {role}")
    record = roles[role]
    if not isinstance(record, Mapping):
        raise ValueError(f"invalid role record {role}")
    keys = {_decode_key(value) for value in record["coordinate_keys"]}
    if _sha256_keys(keys) != record["coordinate_key_sha256"]:
        raise ValueError(f"role key hash failed: {role}")
    return keys


def registered_rows(
    registry: Mapping[str, object], role: str, data: Mapping[str, object]
) -> np.ndarray:
    keys = registered_role_keys(registry, role)
    rows = _rows_for_keys(data["keys"], keys)
    record = registry["roles"][role]
    if _sha256_indices(rows) != record["row_index_sha256"]:
        raise ValueError(f"role row hash failed: {role}")
    return rows


def verify_pre_result_contract(
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, object]:
    """Re-audit all pre-result conditions without loading a model or scoring test."""

    registry = load_registry(registry_path)
    source_files = registry["source_files"]
    train_rss = Path(str(source_files["rss_training"]["path"]))
    training, _testing = load_tuji_author_splits(train_rss.parent)
    roles = {
        name: registered_role_keys(registry, name)
        for name in ("member_fit", "early_stop", "selection_refit", "development", "final_refit")
    }
    for name in roles:
        registered_rows(registry, name, training)
    role_audit = audit_selection_roles(roles, training)
    planned = registry["planned_output_paths"]
    existing = [
        str(path)
        for path in map(Path, planned.values())
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "pre-result contract rejects existing planned output artifacts: "
            + ", ".join(existing)
        )
    return {
        "status": "passed_no_model_outcomes",
        "registry_sha256": registry["registry_sha256"],
        "training_role_audit": role_audit,
        "author_testing_model_outcome_accesses_before_freeze": registry[
            "selection_and_refit_contract"
        ]["author_testing_model_outcome_accesses_before_freeze"],
        "planned_outputs_absent": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    audit = command.add_parser("audit", help="read-only TUJI data qualification audit")
    audit.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    roles = command.add_parser(
        "audit-roles", help="read-only audit of training-only coordinate roles"
    )
    roles.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    register = command.add_parser("register", help="write a new self-hashed registry")
    register.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    register.add_argument("--design", default=str(DEFAULT_DESIGN))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))
    check = command.add_parser("check-contract", help="verify frozen pre-result contract")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        training, testing = load_tuji_author_splits(args.data_dir)
        result: Mapping[str, object] = audit_tuji_author_splits(training, testing)
    elif args.command == "audit-roles":
        training = load_tuji_split(args.data_dir, split="training")
        roles, split_details = construct_selection_roles(set(training["keys"]))
        result = {
            "status": "passed_no_model_outcomes",
            "split_details": split_details,
            "role_audit": audit_selection_roles(roles, training),
        }
    elif args.command == "register":
        result = register_protocol(
            data_dir=args.data_dir, design_path=args.design, output=args.output
        )
    elif args.command == "check-contract":
        result = verify_pre_result_contract(args.registry)
    else:  # pragma: no cover - argparse enforces valid commands.
        raise RuntimeError(f"unknown command {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
