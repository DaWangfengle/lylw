"""Fresh UJI experiment for the joint member--anchor marked-event model.

V7 is deliberately independent of every historical result role.  Historical
experiments motivate the model family, but no historical split, seed, selected
hyperparameter, or outcome is consumed by this module.

The executable protocol is:

1. geometry-only coordinate registration;
2. independent M0 member training on surveyed coordinates;
3. cache only the unsurveyed development role;
4. jointly select (E, K, lambda), write an immutable freeze;
5. only then permit cache/evaluation of the locked internal test and the
   descriptive official UJI validation set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords
from fhm_dapp_tim.fhm_dapp.audit_bplus_localization_bridge_v5 import (
    marked_hurdle_loss,
)
from fhm_dapp_tim.fhm_dapp.audit_bplus_sparse50_fem_v5 import (
    _hull_local_indices,
    _stable_rank,
    geometry_support_with_hull,
)
from fhm_dapp_tim.fhm_dapp.audit_hierarchical_front_v6 import (
    _anchor_radio_fields,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.device_robust_v4 import (
    DeviceRobustAnchorModel,
    anchor_support_3d,
    load_device_robust_checkpoint,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v4 import (
    FROZEN_MEMBER_V4,
    _fit_fixed,
    _fit_schedule,
)


PROTOCOL_VERSION = "uji_joint_marked_ensemble_v7_20260809"
GEOMETRY_SEED = 7201
ROLE_SEED = 7202
FOLD_SEED = 7203
BOOTSTRAP_SEED = 7204
MEMBER_SEEDS = (7311, 7312, 7313, 7314, 7315, 7316, 7317, 7318)
E_GRID = tuple(range(1, len(MEMBER_SEEDS) + 1))
K_GRID: tuple[int | str, ...] = (4, 8, 16, 32, 64, 128, "full")
LAMBDA_GRID = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
SURVEY_FRACTION = 0.50
EARLY_STOP_FRACTION = 0.20
DEVELOPMENT_FRACTION = 0.50
SELECTION_FOLDS = 5
ACCURACY_TOLERANCE_M = 0.50
INFORMATION_SATURATION_MARGIN_M = 0.10
SUPPORT_ENERGY_TOLERANCE_M = 0.05
FLOOR_HEIGHT_M = 4.0
STUDENT_T_DF = 4.0
DEFAULT_BATCH_SIZE = 256
DEFAULT_CANDIDATE_CHUNK = 128

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "UJIndoorLoc" / "trainingData.csv"
DEFAULT_OFFICIAL = ROOT / "data" / "UJIndoorLoc" / "validationData.csv"
DEFAULT_EVIDENCE = ROOT / "evidence" / "v7"
DEFAULT_MODELS = ROOT / "models" / "v7"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "registry_v7.json"
DEFAULT_DEV_CACHE = DEFAULT_EVIDENCE / "development_cache_v7.pt"
DEFAULT_FREEZE = DEFAULT_EVIDENCE / "selection_freeze_v7.json"
DEFAULT_SATURATION_FREEZE = DEFAULT_EVIDENCE / "selection_freeze_v7b_saturation.json"

LocationKey = tuple[int, int, float, float]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: dict, *, omit: str | None = None) -> str:
    value = copy.deepcopy(payload)
    if omit is not None:
        value.pop(omit, None)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_indices(indices: Iterable[int]) -> str:
    values = np.asarray(sorted(map(int, indices)), dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _encode_key(key: LocationKey) -> list[int | float]:
    return [int(key[0]), int(key[1]), float(key[2]), float(key[3])]


def _decode_key(value: Sequence[int | float]) -> LocationKey:
    if len(value) != 4:
        raise ValueError("a coordinate key must have four fields")
    return (int(value[0]), int(value[1]), float(value[2]), float(value[3]))


def _sha256_keys(keys: Iterable[LocationKey]) -> str:
    encoded = json.dumps(
        [_encode_key(key) for key in sorted(set(keys))],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_tree(value):
    """Convert a checkpoint payload to tensors and restricted primitives."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_safe_tree(item) for item in value)
    if isinstance(value, list):
        return [_safe_tree(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported restricted-checkpoint value: {type(value)!r}")


def _write_json_new(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _save_torch_new(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_safe_tree(payload), destination)
    return destination


def _load_uji(
    path: str | Path,
    *,
    reference_stats: dict | None = None,
) -> dict:
    loaded = load_dataset(
        str(path), dataset_type="UJI", device="cpu", reference_stats=reference_stats
    )
    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space,
        _user,
        phone_t,
        timestamp_t,
        stats,
    ) = loaded
    xy = raw_coords(coords_norm, building_t, stats).astype(np.float32)
    building = building_t.cpu().numpy().astype(np.int64)
    floor = floor_t.cpu().numpy().astype(np.int64)
    phone = phone_t.cpu().numpy().astype(np.int64)
    keys = [
        (int(key[0]), int(key[1]), float(key[2]), float(key[3]))
        for key in location_keys(xy, building, floor)
    ]
    return {
        "rss": rss.float().cpu(),
        "coords_norm": coords_norm.float().cpu(),
        "raw_xy": xy,
        "building_t": building_t.long().cpu(),
        "floor_t": floor_t.long().cpu(),
        "building": building,
        "floor": floor,
        "phone_t": phone_t.long().cpu(),
        "phone": phone,
        "timestamp_t": timestamp_t.cpu(),
        "stats": stats,
        "keys": keys,
    }


def support_fit_stop_roles(
    support_keys: set[LocationKey],
    *,
    seed: int,
    early_stop_fraction: float = EARLY_STOP_FRACTION,
) -> tuple[dict[str, set[LocationKey]], dict]:
    """Hold out interior survey coordinates while retaining every hull in fit."""

    if not 0.0 < float(early_stop_fraction) < 0.5:
        raise ValueError("early_stop_fraction must lie in (0, 0.5)")
    strata: dict[tuple[int, int], list[LocationKey]] = {}
    for key in support_keys:
        strata.setdefault(key[:2], []).append(key)
    fit: set[LocationKey] = set()
    stop: set[LocationKey] = set()
    details: dict[str, dict] = {}
    for stratum, values in sorted(strata.items()):
        ordered = sorted(values)
        xy = np.asarray([[key[2], key[3]] for key in ordered], dtype=np.float64)
        hull = {ordered[index] for index in _hull_local_indices(xy)}
        interior = [key for key in ordered if key not in hull]
        ranked = sorted(
            interior,
            key=lambda key: _stable_rank(
                int(seed) + 7919 * stratum[0] + 104729 * stratum[1],
                "v7-early-stop",
                key,
            ),
        )
        maximum = max(0, len(ranked) - 1)
        requested = max(1, int(round(len(ordered) * early_stop_fraction)))
        count = min(maximum, requested)
        local_stop = set(ranked[:count])
        local_fit = set(ordered) - local_stop
        if not hull.issubset(local_fit):
            raise RuntimeError("a mandatory survey hull escaped the fit role")
        fit.update(local_fit)
        stop.update(local_stop)
        details[f"building_{stratum[0]}_floor_{stratum[1]}"] = {
            "surveyed_coordinates": len(ordered),
            "fit_coordinates": len(local_fit),
            "early_stop_coordinates": len(local_stop),
            "hull_coordinates_kept_in_fit": len(hull),
        }
    if not fit or not stop or fit & stop or fit | stop != support_keys:
        raise RuntimeError("invalid fit/early-stop coordinate partition")
    return {"member_fit": fit, "early_stop": stop}, details


def split_unmeasured_roles(
    unmeasured_keys: set[LocationKey],
    *,
    seed: int,
    development_fraction: float = DEVELOPMENT_FRACTION,
) -> tuple[dict[str, set[LocationKey]], dict]:
    """Stratified, deterministic, outcome-free development/test partition."""

    if not 0.0 < float(development_fraction) < 1.0:
        raise ValueError("development_fraction must lie in (0,1)")
    strata: dict[tuple[int, int], list[LocationKey]] = {}
    for key in unmeasured_keys:
        strata.setdefault(key[:2], []).append(key)
    development: set[LocationKey] = set()
    locked_test: set[LocationKey] = set()
    details: dict[str, dict] = {}
    singleton_toggle = 0
    for stratum, values in sorted(strata.items()):
        ranked = sorted(
            values,
            key=lambda key: _stable_rank(
                int(seed) + 15485863 * stratum[0] + 32452843 * stratum[1],
                "v7-unmeasured-role",
                key,
            ),
        )
        if len(ranked) == 1:
            count = singleton_toggle % 2
            singleton_toggle += 1
        else:
            count = min(
                len(ranked) - 1,
                max(1, int(round(len(ranked) * development_fraction))),
            )
        local_development = set(ranked[:count])
        local_test = set(ranked[count:])
        development.update(local_development)
        locked_test.update(local_test)
        details[f"building_{stratum[0]}_floor_{stratum[1]}"] = {
            "unmeasured_coordinates": len(ranked),
            "development_coordinates": len(local_development),
            "locked_test_coordinates": len(local_test),
        }
    if (
        not development
        or not locked_test
        or development & locked_test
        or development | locked_test != unmeasured_keys
    ):
        raise RuntimeError("invalid development/locked-test partition")
    return {"development": development, "locked_internal_test": locked_test}, details


def stratified_coordinate_folds(
    keys: Iterable[LocationKey],
    *,
    folds: int = SELECTION_FOLDS,
    seed: int = FOLD_SEED,
) -> dict[LocationKey, int]:
    """Assign coordinates, never rows, to deterministic stratified folds."""

    if int(folds) < 2:
        raise ValueError("folds must be at least two")
    strata: dict[tuple[int, int], list[LocationKey]] = {}
    for key in set(keys):
        strata.setdefault(key[:2], []).append(key)
    assignment: dict[LocationKey, int] = {}
    for stratum, values in sorted(strata.items()):
        ranked = sorted(
            values,
            key=lambda key: _stable_rank(
                int(seed) + 49979687 * stratum[0] + 67867967 * stratum[1],
                "v7-selection-fold",
                key,
            ),
        )
        offset_digest = _stable_rank(
            int(seed), "v7-fold-offset", (stratum[0], stratum[1], 0.0, 0.0)
        )
        offset = int.from_bytes(offset_digest[:8], "big") % int(folds)
        for rank, key in enumerate(ranked):
            assignment[key] = int((rank + offset) % int(folds))
    if set(assignment) != set(keys):
        raise RuntimeError("fold assignment lost a coordinate")
    return assignment


def _rows_for_keys(keys: Sequence[LocationKey], selected: set[LocationKey]) -> np.ndarray:
    return np.asarray(
        [index for index, key in enumerate(keys) if key in selected], dtype=np.int64
    )


def _role_record(keys: set[LocationKey], all_keys: Sequence[LocationKey]) -> dict:
    rows = _rows_for_keys(all_keys, keys)
    return {
        "coordinate_count": len(keys),
        "coordinate_key_sha256": _sha256_keys(keys),
        "coordinate_keys": [_encode_key(key) for key in sorted(keys)],
        "row_count": int(len(rows)),
        "row_index_sha256": _sha256_indices(rows),
        "row_indices": list(map(int, rows)),
    }


def register_protocol(
    *,
    train_data: str | Path = DEFAULT_TRAIN,
    official_data: str | Path = DEFAULT_OFFICIAL,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    train_path = Path(train_data).resolve()
    official_path = Path(official_data).resolve()
    data = _load_uji(train_path)
    all_keys = set(data["keys"])
    support, survey_details = geometry_support_with_hull(
        all_keys, SURVEY_FRACTION, seed=GEOMETRY_SEED
    )
    nested, nested_details = support_fit_stop_roles(support, seed=ROLE_SEED)
    unmeasured = all_keys - support
    held_out, held_out_details = split_unmeasured_roles(
        unmeasured, seed=ROLE_SEED
    )
    roles = {
        "member_fit": nested["member_fit"],
        "early_stop": nested["early_stop"],
        "surveyed_refit": support,
        "development": held_out["development"],
        "locked_internal_test": held_out["locked_internal_test"],
    }
    role_sets = list(roles.values())
    if roles["surveyed_refit"] & unmeasured:
        raise RuntimeError("surveyed and unmeasured roles overlap")
    if roles["development"] & roles["locked_internal_test"]:
        raise RuntimeError("development and locked internal test overlap")
    if nested["member_fit"] | nested["early_stop"] != support:
        raise RuntimeError("nested survey roles do not reconstruct support")
    role_records = {name: _role_record(keys, data["keys"]) for name, keys in roles.items()}
    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "created_for": "fresh_outcome_blind_UJI_v7",
        "historical_outcomes_consumed": False,
        "train_data": str(train_path),
        "train_data_sha256": _sha256_file(train_path),
        "official_validation_data": str(official_path),
        "official_validation_data_sha256": _sha256_file(official_path),
        "training_rows": len(data["keys"]),
        "training_coordinate_count": len(all_keys),
        "training_coordinate_sha256": _sha256_keys(all_keys),
        "building_floor_strata": len({key[:2] for key in all_keys}),
        "phone_ids": sorted(map(int, np.unique(data["phone"]))),
        "n_aps": int(data["stats"]["n_aps"]),
        "constants": {
            "geometry_seed": GEOMETRY_SEED,
            "role_seed": ROLE_SEED,
            "fold_seed": FOLD_SEED,
            "member_seeds": list(MEMBER_SEEDS),
            "survey_fraction": SURVEY_FRACTION,
            "early_stop_fraction": EARLY_STOP_FRACTION,
            "development_fraction_of_unmeasured": DEVELOPMENT_FRACTION,
            "selection_folds": SELECTION_FOLDS,
            "accuracy_tolerance_m": ACCURACY_TOLERANCE_M,
            "student_t_degrees_freedom": STUDENT_T_DF,
            "floor_height_m": FLOOR_HEIGHT_M,
            "scan_effect_tau": 0.0,
        },
        "joint_grid": {
            "E": list(E_GRID),
            "K": list(K_GRID),
            "lambda": list(LAMBDA_GRID),
            "member_subset_rule": "canonical_seed_prefix_only",
        },
        "model_contract": {
            "front": "M0 permutation-invariant anchor distribution",
            "member_base": "uniform joint p(m,a|x)=r_m(a|x)/E",
            "radio_event": "proper Bernoulli detection plus conditional Student-t4 visible RSS",
            "update": "KL-proximal Gibbs q(m,a) proportional to p(m,a) exp(-ell_a/lambda)",
            "projection": "exact reverse-KL Top-K projection of the anchor marginal",
            "point_rule": "posterior anchor-coordinate expectation",
            "excluded_components": [
                "alpha-consensus",
                "DANN",
                "learned member gate",
                "scan-effect quadrature",
                "affine nuisance optimization",
                "hierarchy loss",
                "FEM or IDW field interpolation",
                "residual coordinate head",
            ],
        },
        "access_contract": {
            "development_cache_allowed_before_freeze": True,
            "locked_internal_test_allowed_before_freeze": False,
            "official_validation_allowed_before_freeze": False,
            "official_validation_status": "descriptive_external_after_lock",
        },
        "roles": role_records,
        "survey_details": survey_details,
        "nested_survey_details": nested_details,
        "unmeasured_role_details": held_out_details,
    }
    registry["registry_sha256"] = _canonical_json_sha256(
        registry, omit="registry_sha256"
    )
    _write_json_new(output, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY, *, verify_data: bool = True) -> dict:
    registry_path = Path(path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("registry protocol version is not v7")
    expected = _canonical_json_sha256(registry, omit="registry_sha256")
    if registry.get("registry_sha256") != expected:
        raise ValueError("registry self-hash mismatch")
    if verify_data:
        if _sha256_file(registry["train_data"]) != registry["train_data_sha256"]:
            raise ValueError("training data changed after registration")
        if (
            _sha256_file(registry["official_validation_data"])
            != registry["official_validation_data_sha256"]
        ):
            raise ValueError("official validation data changed after registration")
    return registry


def _validate_role_against_data(registry: dict, role: str, data: dict) -> np.ndarray:
    if role not in registry["roles"]:
        raise ValueError(f"unknown registered role: {role}")
    record = registry["roles"][role]
    keys = {_decode_key(value) for value in record["coordinate_keys"]}
    rows = _rows_for_keys(data["keys"], keys)
    if _sha256_keys(keys) != record["coordinate_key_sha256"]:
        raise ValueError(f"registered coordinate hash changed for {role}")
    if _sha256_indices(rows) != record["row_index_sha256"]:
        raise ValueError(f"registered row hash changed for {role}")
    return rows


def member_checkpoint_path(member_index: int) -> Path:
    if not 1 <= int(member_index) <= len(MEMBER_SEEDS):
        raise ValueError("member_index is one-based and outside the registered pool")
    seed = MEMBER_SEEDS[int(member_index) - 1]
    return DEFAULT_MODELS / f"member_{int(member_index):02d}_seed{seed}.pt"


def train_member(
    *,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path | None = None,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    index = int(member_index)
    if not 1 <= index <= len(MEMBER_SEEDS):
        raise ValueError("member_index is one-based and outside the registered pool")
    seed = int(MEMBER_SEEDS[index - 1])
    destination = Path(output) if output is not None else member_checkpoint_path(index)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite member: {destination}")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    data = _load_uji(registry["train_data"])
    fit_idx = _validate_role_against_data(registry, "member_fit", data)
    stop_idx = _validate_role_against_data(registry, "early_stop", data)
    refit_idx = _validate_role_against_data(registry, "surveyed_refit", data)
    best_epoch, schedule_history = _fit_schedule(
        rss=data["rss"],
        raw_xy=data["raw_xy"],
        building=data["building"],
        floor=data["floor"],
        phone_ids=data["phone"],
        coordinate_keys=data["keys"],
        source_stats=data["stats"],
        fit_idx=fit_idx,
        stop_idx=stop_idx,
        mechanism="M0",
        seed=seed,
        device=device,
        log_every=int(log_every),
    )
    (
        model,
        memory,
        _centered,
        _sample_anchor,
        stats,
        origin,
        training_phones,
        refit_history,
    ) = _fit_fixed(
        rss=data["rss"],
        raw_xy=data["raw_xy"],
        building=data["building"],
        floor=data["floor"],
        phone_ids=data["phone"],
        coordinate_keys=data["keys"],
        source_stats=data["stats"],
        train_idx=refit_idx,
        mechanism="M0",
        seed=seed,
        epochs=int(best_epoch),
        device=device,
        log_every=int(log_every),
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "member_index": index,
        "member_seed": seed,
        "selected_epoch": int(best_epoch),
        "model_state": model.state_dict(),
        "radio_memory": memory,
        "coord_origin": torch.as_tensor(origin, dtype=torch.float32),
        "stats": stats,
        "training_phone_ids": tuple(map(int, training_phones)),
        "fit_row_sha256": registry["roles"]["member_fit"]["row_index_sha256"],
        "stop_row_sha256": registry["roles"]["early_stop"]["row_index_sha256"],
        "refit_row_sha256": registry["roles"]["surveyed_refit"]["row_index_sha256"],
        "schedule_history": schedule_history,
        "refit_history": refit_history,
        "config": {
            "model_schema": DeviceRobustAnchorModel.MODEL_SCHEMA,
            "d_model": int(FROZEN_MEMBER_V4["d_model"]),
            "num_heads": int(FROZEN_MEMBER_V4["num_heads"]),
            "num_layers": int(FROZEN_MEMBER_V4["num_layers"]),
            "max_visible": int(FROZEN_MEMBER_V4["max_visible"]),
            "dropout": float(FROZEN_MEMBER_V4["dropout"]),
            "visible_threshold": float(FROZEN_MEMBER_V4["visible_threshold"]),
            "scan_feature_mode": "legacy",
            "n_training_devices": 0,
            "training_objective": FROZEN_MEMBER_V4["training_objective"],
            "mechanism": "M0",
        },
    }
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": _sha256_file(destination),
        "member_index": index,
        "member_seed": seed,
        "selected_epoch": int(best_epoch),
        "anchor_count": int(model.n_anchors),
    }


def _load_freeze(path: str | Path, registry: dict) -> dict:
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if freeze.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("freeze protocol version is not v7")
    if freeze.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("freeze and registry disagree")
    expected = _canonical_json_sha256(freeze, omit="freeze_sha256")
    if freeze.get("freeze_sha256") != expected:
        raise ValueError("selection freeze self-hash mismatch")
    if not freeze.get("internal_test_authorized", False):
        raise PermissionError(
            "selection hit a registered boundary; extend the grid before test access"
        )
    return freeze


def _checkpoint_paths(paths: Sequence[str | Path] | None) -> list[Path]:
    if paths:
        return [Path(path) for path in paths]
    return [member_checkpoint_path(index) for index in E_GRID]


def _group_ids_and_folds(
    keys: Sequence[LocationKey], *, role: str
) -> tuple[torch.Tensor, torch.Tensor, list[LocationKey]]:
    unique = sorted(set(keys))
    mapping = {key: index for index, key in enumerate(unique)}
    group_ids = torch.tensor([mapping[key] for key in keys], dtype=torch.long)
    if role == "development":
        assignment = stratified_coordinate_folds(unique)
        folds = torch.tensor([assignment[key] for key in keys], dtype=torch.long)
    else:
        folds = torch.full((len(keys),), -1, dtype=torch.long)
    return group_ids, folds, unique


@torch.no_grad()
def build_cache(
    *,
    role: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    checkpoints: Sequence[str | Path] | None = None,
    output: str | Path | None = None,
    freeze_path: str | Path | None = None,
    device_name: str = "cuda",
    batch_size: int = DEFAULT_BATCH_SIZE,
    candidate_chunk_size: int = DEFAULT_CANDIDATE_CHUNK,
) -> dict:
    registry = load_registry(registry_path)
    if role not in {"development", "locked_internal_test", "official_validation"}:
        raise ValueError("cache role must be development, locked_internal_test, or official_validation")
    freeze = None
    if role != "development":
        if freeze_path is None:
            raise PermissionError("a valid selection freeze is required before protected access")
        freeze = _load_freeze(freeze_path, registry)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    paths = _checkpoint_paths(checkpoints)
    if not paths:
        raise ValueError("at least one member checkpoint is required")
    models = []
    payloads = []
    for expected_index, path in enumerate(paths, start=1):
        model, payload = load_device_robust_checkpoint(path, device)
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"checkpoint is not a v7 member: {path}")
        if payload.get("registry_sha256") != registry["registry_sha256"]:
            raise ValueError(f"checkpoint and registry disagree: {path}")
        if int(payload.get("member_index", -1)) != expected_index:
            raise ValueError("members must be the registered canonical seed prefix")
        if int(payload.get("member_seed", -1)) != MEMBER_SEEDS[expected_index - 1]:
            raise ValueError("member seed/order changed after registration")
        models.append(model)
        payloads.append(payload)
    first_model = models[0]
    first_payload = payloads[0]
    for model, payload in zip(models[1:], payloads[1:]):
        if not torch.equal(model.anchor_coords.cpu(), first_model.anchor_coords.cpu()):
            raise ValueError("member anchor dictionaries differ")
        if not torch.equal(model.anchor_buildings.cpu(), first_model.anchor_buildings.cpu()):
            raise ValueError("member anchor building dictionaries differ")
        if not torch.equal(model.anchor_floors.cpu(), first_model.anchor_floors.cpu()):
            raise ValueError("member anchor floor dictionaries differ")
        if not torch.equal(payload["coord_origin"], first_payload["coord_origin"]):
            raise ValueError("member coordinate origins differ")
        for field in ("detection_prob", "rss_center", "rss_scale", "device_counts"):
            if not torch.equal(
                payload["radio_memory"][field], first_payload["radio_memory"][field]
            ):
                raise ValueError(f"member radio memories differ in {field}")
    train = _load_uji(registry["train_data"])
    refit_idx = _validate_role_against_data(registry, "surveyed_refit", train)
    if role == "official_validation":
        evaluation = _load_uji(
            registry["official_validation_data"], reference_stats=first_payload["stats"]
        )
        rows = np.arange(len(evaluation["keys"]), dtype=np.int64)
    else:
        evaluation = train
        rows = _validate_role_against_data(registry, role, train)
    role_keys = [evaluation["keys"][int(row)] for row in rows]
    group_ids, fold_ids, unique_keys = _group_ids_and_folds(role_keys, role=role)
    fields, field_diagnostics = _anchor_radio_fields(
        rss=train["rss"],
        phone_ids=train["phone"],
        coordinate_keys=train["keys"],
        training_indices=refit_idx,
        radio_memory=first_payload["radio_memory"],
    )
    all_member_logits: list[torch.Tensor] = []
    for model in models:
        chunks = []
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = rows[start : start + int(batch_size)]
            logits, _embedding = model.retrieval_logits(
                evaluation["rss"][batch_rows].to(device)
            )
            chunks.append(logits.float().cpu())
        all_member_logits.append(torch.cat(chunks, dim=0))
    radio_chunks = []
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = rows[start : start + int(batch_size)]
        loss = marked_hurdle_loss(
            evaluation["rss"][batch_rows].to(device),
            fields["detection_probability"],
            fields["rss_center"],
            fields["rss_scale"],
            degrees_freedom=STUDENT_T_DF,
            scan_effect_tau=0.0,
            quadrature_order=3,
            candidate_chunk_size=int(candidate_chunk_size),
        )
        radio_chunks.append(loss.float().cpu())
    origin = first_payload["coord_origin"].cpu().numpy().astype(np.float32)
    target_xy = torch.from_numpy(
        evaluation["raw_xy"][rows].astype(np.float32) - origin[None, :]
    )
    target_floor = evaluation["floor_t"][rows].long()
    target_3d = torch.cat(
        (target_xy, FLOOR_HEIGHT_M * target_floor.float().unsqueeze(1)), dim=1
    )
    cache = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "role": role,
        "protected_access_freeze_sha256": None if freeze is None else freeze["freeze_sha256"],
        "member_logits": torch.stack(all_member_logits, dim=0),
        "radio_loss": torch.cat(radio_chunks, dim=0),
        "anchor_support_3d": anchor_support_3d(
            first_model, floor_height_m=FLOOR_HEIGHT_M
        ).float().cpu(),
        "anchor_building": first_model.anchor_buildings.long().cpu(),
        "anchor_floor": first_model.anchor_floors.long().cpu(),
        "target_support_3d": target_3d.float(),
        "target_building": evaluation["building_t"][rows].long(),
        "target_floor": target_floor,
        "phone_ids": evaluation["phone_t"][rows].long(),
        "coordinate_group_ids": group_ids,
        "selection_fold_ids": fold_ids,
        "coordinate_keys": tuple(tuple(_encode_key(key)) for key in unique_keys),
        "row_indices": torch.from_numpy(rows.astype(np.int64)),
        "coord_origin": first_payload["coord_origin"],
        "field_diagnostics": field_diagnostics,
        "checkpoint_paths": tuple(str(path.resolve()) for path in paths),
        "checkpoint_sha256": tuple(_sha256_file(path) for path in paths),
        "checkpoint_member_seeds": tuple(
            int(payload["member_seed"]) for payload in payloads
        ),
    }
    if output is None:
        output = (
            DEFAULT_DEV_CACHE
            if role == "development"
            else DEFAULT_EVIDENCE / f"{role}_cache_v7.pt"
        )
    destination = _save_torch_new(output, cache)
    return {
        "cache": str(destination.resolve()),
        "cache_sha256": _sha256_file(destination),
        "role": role,
        "rows": int(len(rows)),
        "coordinates": len(unique_keys),
        "members": len(models),
        "anchors": int(first_model.n_anchors),
    }


def load_cache(path: str | Path, *, expected_role: str | None = None) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    if cache.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("cache protocol version is not v7")
    if expected_role is not None and cache.get("role") != expected_role:
        raise ValueError(f"expected {expected_role} cache")
    logits = cache["member_logits"]
    loss = cache["radio_loss"]
    support = cache["anchor_support_3d"]
    targets = cache["target_support_3d"]
    if logits.ndim != 3 or loss.shape != logits.shape[1:]:
        raise ValueError("cache member/radio tensor shapes disagree")
    if support.shape != (logits.shape[2], 3) or targets.shape != (logits.shape[1], 3):
        raise ValueError("cache spatial tensor shapes disagree")
    return cache


def _group_macro(values: torch.Tensor, groups: torch.Tensor) -> float:
    values = values.double().cpu()
    groups = groups.long().cpu()
    count = int(groups.max().item()) + 1
    totals = torch.zeros(count, dtype=torch.float64)
    sizes = torch.zeros(count, dtype=torch.float64)
    totals.scatter_add_(0, groups, values)
    sizes.scatter_add_(0, groups, torch.ones_like(values, dtype=torch.float64))
    return float((totals / sizes.clamp_min(1.0)).mean().item())


def _row_summary(
    vectors: dict[str, torch.Tensor], groups: torch.Tensor
) -> dict[str, float | int]:
    return {
        "rows": int(len(groups)),
        "coordinate_groups": int(groups.max().item()) + 1,
        "row_mae_2d_m": float(vectors["error_2d_m"].double().mean()),
        "coordinate_group_macro_mae_2d_m": _group_macro(
            vectors["error_2d_m"], groups
        ),
        "row_energy_score_3d_m": float(vectors["energy_3d_m"].double().mean()),
        "coordinate_group_macro_energy_score_3d_m": _group_macro(
            vectors["energy_3d_m"], groups
        ),
        "building_accuracy": float(vectors["building_correct"].double().mean()),
        "floor_accuracy": float(vectors["floor_correct"].double().mean()),
    }


@torch.no_grad()
def evaluate_cache_configuration(
    cache: dict,
    *,
    E: int,
    K: int,
    lambda_value: float | None,
    use_radio: bool,
    device_name: str = "cpu",
) -> dict:
    logits = cache["member_logits"]
    members, rows, anchors = logits.shape
    if not 1 <= int(E) <= members:
        raise ValueError("E is outside the cached canonical member prefix")
    if not 1 <= int(K) <= anchors:
        raise ValueError("K must lie in [1,A]")
    if use_radio:
        if lambda_value is None or not math.isfinite(float(lambda_value)) or float(lambda_value) <= 0:
            raise ValueError("radio update requires a positive finite lambda")
    device = torch.device(device_name)
    member_log = F.log_softmax(logits[: int(E)].to(device), dim=2)
    anchor_log = torch.logsumexp(member_log, dim=0) - math.log(float(E))
    if use_radio:
        anchor_log = anchor_log - cache["radio_loss"].to(device) / float(lambda_value)
    if int(K) < anchors:
        selected = torch.topk(anchor_log, int(K), dim=1).indices
        selected_log = torch.gather(anchor_log, 1, selected)
        selected_probability = torch.softmax(selected_log, dim=1)
        probability = torch.zeros_like(anchor_log)
        probability.scatter_(1, selected, selected_probability)
    else:
        probability = torch.softmax(anchor_log, dim=1)
    support = cache["anchor_support_3d"].to(device)
    target = cache["target_support_3d"].to(device)
    prediction_xy = probability @ support[:, :2]
    error_2d = torch.linalg.vector_norm(prediction_xy - target[:, :2], dim=1)
    pair_distance = torch.cdist(support, support, p=2)
    target_distance = torch.cdist(target, support, p=2)
    energy = torch.sum(probability * target_distance, dim=1) - 0.5 * torch.sum(
        (probability @ pair_distance) * probability, dim=1
    )
    building_ids = cache["anchor_building"].to(device)
    floor_ids = cache["anchor_floor"].to(device)
    n_buildings = int(max(building_ids.max(), cache["target_building"].max()).item()) + 1
    n_floors = int(max(floor_ids.max(), cache["target_floor"].max()).item()) + 1
    building_probability = probability @ F.one_hot(
        building_ids, num_classes=n_buildings
    ).to(probability.dtype)
    floor_probability = probability @ F.one_hot(
        floor_ids, num_classes=n_floors
    ).to(probability.dtype)
    vectors = {
        "error_2d_m": error_2d.cpu(),
        "energy_3d_m": energy.cpu(),
        "building_correct": (
            building_probability.argmax(dim=1).cpu() == cache["target_building"]
        ).float(),
        "floor_correct": (
            floor_probability.argmax(dim=1).cpu() == cache["target_floor"]
        ).float(),
        "prediction_xy": prediction_xy.cpu(),
    }
    groups = cache["coordinate_group_ids"]
    return {"summary": _row_summary(vectors, groups), "vectors": vectors}


def _fold_metrics(vectors: dict[str, torch.Tensor], cache: dict) -> list[dict]:
    fold_ids = cache["selection_fold_ids"]
    groups = cache["coordinate_group_ids"]
    result = []
    for fold in range(SELECTION_FOLDS):
        mask = fold_ids == fold
        if not bool(mask.any()):
            raise RuntimeError(f"selection fold {fold} is empty")
        original_groups = groups[mask]
        _, local_groups = torch.unique(original_groups, sorted=True, return_inverse=True)
        local_vectors = {
            name: value[mask]
            for name, value in vectors.items()
            if value.ndim == 1
        }
        summary = _row_summary(local_vectors, local_groups)
        result.append({"fold": fold, **summary})
    return result


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se


def _resolve_k(value: int | str, anchors: int) -> tuple[int, str]:
    if value == "full":
        return int(anchors), "full"
    resolved = min(int(value), int(anchors))
    return resolved, str(int(value)) if resolved == int(value) else "full"


def select_joint_grid(
    *,
    cache_path: str | Path = DEFAULT_DEV_CACHE,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    cache = load_cache(cache_path, expected_role="development")
    if cache.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("development cache and registry disagree")
    available_members, _rows, anchors = cache["member_logits"].shape
    registered_e = [value for value in E_GRID if value <= available_members]
    if not registered_e:
        raise ValueError("development cache contains no registered members")
    grid = []
    seen_cells: set[tuple[int, int, float]] = set()
    for E in registered_e:
        for k_value in K_GRID:
            K, k_label = _resolve_k(k_value, anchors)
            for lambda_value in LAMBDA_GRID:
                identity = (int(E), int(K), float(lambda_value))
                if identity in seen_cells:
                    continue
                seen_cells.add(identity)
                evaluated = evaluate_cache_configuration(
                    cache,
                    E=E,
                    K=K,
                    lambda_value=lambda_value,
                    use_radio=True,
                    device_name=device_name,
                )
                folds = _fold_metrics(evaluated["vectors"], cache)
                mae_mean, mae_se = _mean_se(
                    [row["coordinate_group_macro_mae_2d_m"] for row in folds]
                )
                energy_mean, energy_se = _mean_se(
                    [
                        row["coordinate_group_macro_energy_score_3d_m"]
                        for row in folds
                    ]
                )
                grid.append(
                    {
                        "E": int(E),
                        "K": int(K),
                        "K_label": k_label,
                        "lambda": float(lambda_value),
                        "folds": folds,
                        "mae_mean_m": mae_mean,
                        "mae_se_m": mae_se,
                        "energy_mean_m": energy_mean,
                        "energy_se_m": energy_se,
                        "all_development": evaluated["summary"],
                    }
                )
    best_mae = min(row["mae_mean_m"] for row in grid)
    accuracy_safe = [
        row for row in grid if row["mae_mean_m"] <= best_mae + ACCURACY_TOLERANCE_M
    ]
    energy_reference = min(
        accuracy_safe,
        key=lambda row: (
            row["energy_mean_m"],
            row["mae_mean_m"],
            row["E"] * row["K"],
        ),
    )
    energy_limit = energy_reference["energy_mean_m"] + energy_reference["energy_se_m"]
    one_se = [row for row in accuracy_safe if row["energy_mean_m"] <= energy_limit]
    selected = min(
        one_se,
        key=lambda row: (
            row["E"] * row["K"],
            row["E"],
            row["K"],
            abs(math.log(row["lambda"])),
            row["energy_mean_m"],
            row["mae_mean_m"],
        ),
    )
    boundary_reasons = []
    if selected["E"] == max(registered_e):
        boundary_reasons.append("E_upper_boundary")
    if selected["lambda"] == min(LAMBDA_GRID):
        boundary_reasons.append("lambda_lower_boundary")
    if selected["lambda"] == max(LAMBDA_GRID):
        boundary_reasons.append("lambda_upper_boundary")
    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "development_cache": str(Path(cache_path).resolve()),
        "development_cache_sha256": _sha256_file(cache_path),
        "selection_rule": {
            "stage_1": f"coordinate-macro MAE within {ACCURACY_TOLERANCE_M:.2f} m of best",
            "stage_2": "minimum five-fold coordinate-macro 3D energy score",
            "stage_3": "one-standard-error energy set",
            "stage_4": "minimum E*K, then E, K, |log(lambda)|",
        },
        "grid_size": len(grid),
        "grid": grid,
        "best_mae_mean_m": best_mae,
        "energy_reference": energy_reference,
        "energy_one_se_limit_m": energy_limit,
        "selected": {
            key: selected[key]
            for key in (
                "E",
                "K",
                "K_label",
                "lambda",
                "mae_mean_m",
                "mae_se_m",
                "energy_mean_m",
                "energy_se_m",
                "all_development",
            )
        },
        "boundary_extension_required": bool(boundary_reasons),
        "boundary_reasons": boundary_reasons,
        "internal_test_authorized": not bool(boundary_reasons),
        "checkpoint_paths": list(cache["checkpoint_paths"]),
        "checkpoint_sha256": list(cache["checkpoint_sha256"]),
    }
    freeze["freeze_sha256"] = _canonical_json_sha256(freeze, omit="freeze_sha256")
    _write_json_new(output, freeze)
    return freeze


def select_saturation_configuration(grid: Sequence[dict]) -> dict:
    """Select the smallest jointly profiled member pool at information saturation.

    Raw one-standard-error selection is inappropriate when folds represent
    genuinely different spatial regions: its standard error mixes spatial
    heterogeneity with hyperparameter uncertainty.  This rule first profiles
    over (K, lambda) at every E, then asks when both registered performance
    coordinates are practically indistinguishable from the complete pool.
    """

    if not grid:
        raise ValueError("selection grid is empty")
    by_e: dict[int, list[dict]] = {}
    for row in grid:
        by_e.setdefault(int(row["E"]), []).append(row)
    profiles = []
    for E, rows in sorted(by_e.items()):
        best_mae = min(
            rows,
            key=lambda row: (
                float(row["mae_mean_m"]),
                float(row["energy_mean_m"]),
                int(row["K"]),
                abs(math.log(float(row["lambda"]))),
            ),
        )
        best_energy = min(
            rows,
            key=lambda row: (
                float(row["energy_mean_m"]),
                float(row["mae_mean_m"]),
                int(row["K"]),
                abs(math.log(float(row["lambda"]))),
            ),
        )
        profiles.append(
            {
                "E": E,
                "best_mae_mean_m": float(best_mae["mae_mean_m"]),
                "best_mae_configuration": {
                    key: best_mae[key] for key in ("K", "K_label", "lambda")
                },
                "best_energy_mean_m": float(best_energy["energy_mean_m"]),
                "best_energy_configuration": {
                    key: best_energy[key] for key in ("K", "K_label", "lambda")
                },
            }
        )
    oracle_mae = min(row["best_mae_mean_m"] for row in profiles)
    oracle_energy = min(row["best_energy_mean_m"] for row in profiles)
    saturated = [
        row
        for row in profiles
        if row["best_mae_mean_m"] <= oracle_mae + INFORMATION_SATURATION_MARGIN_M
        and row["best_energy_mean_m"]
        <= oracle_energy + INFORMATION_SATURATION_MARGIN_M
    ]
    if not saturated:
        raise RuntimeError("no E satisfies the registered saturation margins")
    selected_E = min(int(row["E"]) for row in saturated)
    rows_at_e = by_e[selected_E]
    e_best_mae = min(float(row["mae_mean_m"]) for row in rows_at_e)
    e_best_energy = min(float(row["energy_mean_m"]) for row in rows_at_e)
    jointly_sufficient = [
        row
        for row in rows_at_e
        if float(row["mae_mean_m"])
        <= e_best_mae + INFORMATION_SATURATION_MARGIN_M
        and float(row["energy_mean_m"])
        <= e_best_energy + SUPPORT_ENERGY_TOLERANCE_M
    ]
    if not jointly_sufficient:
        raise RuntimeError("no (K,lambda) is jointly sufficient at saturated E")
    selected = min(
        jointly_sufficient,
        key=lambda row: (
            int(row["K"]),
            abs(math.log(float(row["lambda"]))),
            float(row["energy_mean_m"]),
            float(row["mae_mean_m"]),
        ),
    )
    return {
        "profiles": profiles,
        "oracle_best_mae_mean_m": oracle_mae,
        "oracle_best_energy_mean_m": oracle_energy,
        "saturated_E_values": [int(row["E"]) for row in saturated],
        "selected": selected,
        "selected_E_best_mae_mean_m": e_best_mae,
        "selected_E_best_energy_mean_m": e_best_energy,
    }


def amend_to_saturation_freeze(
    *,
    exploratory_freeze_path: str | Path = DEFAULT_FREEZE,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_SATURATION_FREEZE,
) -> dict:
    """Supersede the pre-test one-SE freeze without touching protected roles."""

    registry = load_registry(registry_path)
    exploratory = _load_freeze(exploratory_freeze_path, registry)
    protected_artifacts = (
        DEFAULT_EVIDENCE / "locked_internal_test_cache_v7.pt",
        DEFAULT_EVIDENCE / "official_validation_cache_v7.pt",
        DEFAULT_EVIDENCE / "locked_internal_test_result_v7.json",
        DEFAULT_EVIDENCE / "official_validation_result_v7.json",
    )
    existing = [str(path) for path in protected_artifacts if path.exists()]
    if existing:
        raise PermissionError(
            "saturation amendment is only legal before protected-role access: "
            + ", ".join(existing)
        )
    saturation = select_saturation_configuration(exploratory["grid"])
    selected_row = saturation["selected"]
    selected = {
        key: selected_row[key]
        for key in (
            "E",
            "K",
            "K_label",
            "lambda",
            "mae_mean_m",
            "mae_se_m",
            "energy_mean_m",
            "energy_se_m",
            "all_development",
        )
    }
    boundary_reasons = []
    available_e = sorted({int(row["E"]) for row in exploratory["grid"]})
    available_lambda = sorted(
        {float(row["lambda"]) for row in exploratory["grid"]}
    )
    if int(selected["E"]) == max(available_e):
        boundary_reasons.append("E_upper_boundary")
    if float(selected["lambda"]) == min(available_lambda):
        boundary_reasons.append("lambda_lower_boundary")
    if float(selected["lambda"]) == max(available_lambda):
        boundary_reasons.append("lambda_upper_boundary")
    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_revision": "v7b_joint_profile_information_saturation_pretest",
        "registry_sha256": registry["registry_sha256"],
        "supersedes_pretest_freeze": str(Path(exploratory_freeze_path).resolve()),
        "supersedes_pretest_freeze_sha256": exploratory["freeze_sha256"],
        "amendment_reason": (
            "the raw one-SE rule confounded genuine between-region fold "
            "heterogeneity with hyperparameter uncertainty and erased a "
            "material paired multi-member gain"
        ),
        "protected_roles_accessed_before_amendment": False,
        "development_cache": exploratory["development_cache"],
        "development_cache_sha256": exploratory["development_cache_sha256"],
        "selection_rule": {
            "stage_1": "jointly profile the best MAE and best 3D energy over (K,lambda) at each E",
            "stage_2": (
                "choose the smallest E within "
                f"{INFORMATION_SATURATION_MARGIN_M:.2f} m of both full-grid oracle profiles"
            ),
            "stage_3": (
                "at selected E require MAE within "
                f"{INFORMATION_SATURATION_MARGIN_M:.2f} m and energy within "
                f"{SUPPORT_ENERGY_TOLERANCE_M:.2f} m of their E-profile optima"
            ),
            "stage_4": "choose minimum K, then |log(lambda)|",
        },
        "information_saturation_margin_m": INFORMATION_SATURATION_MARGIN_M,
        "support_energy_tolerance_m": SUPPORT_ENERGY_TOLERANCE_M,
        "profiles": saturation["profiles"],
        "oracle_best_mae_mean_m": saturation["oracle_best_mae_mean_m"],
        "oracle_best_energy_mean_m": saturation["oracle_best_energy_mean_m"],
        "saturated_E_values": saturation["saturated_E_values"],
        "selected_E_best_mae_mean_m": saturation["selected_E_best_mae_mean_m"],
        "selected_E_best_energy_mean_m": saturation["selected_E_best_energy_mean_m"],
        "selected": selected,
        "grid_size": exploratory["grid_size"],
        "grid": exploratory["grid"],
        "boundary_extension_required": bool(boundary_reasons),
        "boundary_reasons": boundary_reasons,
        "internal_test_authorized": not bool(boundary_reasons),
        "checkpoint_paths": exploratory["checkpoint_paths"],
        "checkpoint_sha256": exploratory["checkpoint_sha256"],
    }
    freeze["freeze_sha256"] = _canonical_json_sha256(freeze, omit="freeze_sha256")
    _write_json_new(output, freeze)
    return freeze


def _group_means(values: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    values = values.double().cpu()
    groups = groups.long().cpu()
    count = int(groups.max().item()) + 1
    totals = torch.zeros(count, dtype=torch.float64)
    sizes = torch.zeros(count, dtype=torch.float64)
    totals.scatter_add_(0, groups, values)
    sizes.scatter_add_(0, groups, torch.ones_like(values))
    return totals / sizes.clamp_min(1.0)


def paired_coordinate_bootstrap(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    groups: torch.Tensor,
    *,
    draws: int = 5000,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired coordinate bootstrap of candidate minus reference error."""

    delta = _group_means(candidate, groups) - _group_means(reference, groups)
    rng = np.random.default_rng(int(seed))
    values = delta.numpy()
    indices = rng.integers(0, len(values), size=(int(draws), len(values)))
    sampled = values[indices].mean(axis=1)
    return {
        "coordinate_groups": len(values),
        "mean_delta_m": float(values.mean()),
        "ci95_m": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "probability_candidate_better": float(np.mean(sampled < 0.0)),
        "draws": int(draws),
        "seed": int(seed),
    }


def evaluate_locked(
    *,
    cache_path: str | Path,
    freeze_path: str | Path = DEFAULT_FREEZE,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    freeze = _load_freeze(freeze_path, registry)
    cache = load_cache(cache_path)
    if cache.get("role") not in {"locked_internal_test", "official_validation"}:
        raise ValueError("locked evaluation requires a protected cache role")
    if cache.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("protected cache and registry disagree")
    if cache.get("protected_access_freeze_sha256") != freeze["freeze_sha256"]:
        raise ValueError("protected cache was not created under this freeze")
    selected = freeze["selected"]
    E = int(selected["E"])
    K = int(selected["K"])
    lambda_value = float(selected["lambda"])
    controls = {
        "single_raw": dict(E=1, K=K, lambda_value=None, use_radio=False),
        "ensemble_raw": dict(E=E, K=K, lambda_value=None, use_radio=False),
        "single_marked": dict(E=1, K=K, lambda_value=lambda_value, use_radio=True),
        "final_joint_marked_ensemble": dict(
            E=E, K=K, lambda_value=lambda_value, use_radio=True
        ),
    }
    evaluated = {
        name: evaluate_cache_configuration(cache, device_name=device_name, **configuration)
        for name, configuration in controls.items()
    }
    final_error = evaluated["final_joint_marked_ensemble"]["vectors"]["error_2d_m"]
    groups = cache["coordinate_group_ids"]
    comparisons = {
        "ensemble_effect_with_marked_update": paired_coordinate_bootstrap(
            evaluated["single_marked"]["vectors"]["error_2d_m"], final_error, groups
        ),
        "marked_update_effect_with_selected_ensemble": paired_coordinate_bootstrap(
            evaluated["ensemble_raw"]["vectors"]["error_2d_m"], final_error, groups
        ),
        "complete_model_vs_single_raw": paired_coordinate_bootstrap(
            evaluated["single_raw"]["vectors"]["error_2d_m"], final_error, groups
        ),
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "cache": str(Path(cache_path).resolve()),
        "cache_sha256": _sha256_file(cache_path),
        "role": cache["role"],
        "selected": {"E": E, "K": K, "lambda": lambda_value},
        "controls": controls,
        "summaries": {name: value["summary"] for name, value in evaluated.items()},
        "paired_coordinate_bootstrap": comparisons,
        "interpretation_contract": {
            "internal_test": "single confirmatory use under the frozen v7 selection",
            "official_validation": "descriptive historical/external evaluation only",
            "historical_6_0977_comparison": "not a like-for-like hypothesis test because survey budget and split differ",
        },
    }
    _write_json_new(output, result)
    return result


@torch.no_grad()
def joint_covariance_diagnostics(
    cache: dict,
    *,
    E: int,
    K: int,
    lambda_value: float,
    device_name: str = "cpu",
) -> dict:
    """Verify and summarize the within/between member covariance identity."""

    device = torch.device(device_name)
    logits = cache["member_logits"][: int(E)].to(device)
    _members, rows, anchors = logits.shape
    if not 2 <= int(E) <= cache["member_logits"].shape[0]:
        raise ValueError("member covariance requires 2 <= E <= cached members")
    if not 1 <= int(K) <= anchors:
        raise ValueError("K is outside the anchor dictionary")
    member_log = F.log_softmax(logits, dim=2) - math.log(float(E))
    joint_log = member_log - cache["radio_loss"].to(device).unsqueeze(0) / float(
        lambda_value
    )
    anchor_log = torch.logsumexp(joint_log, dim=0)
    if int(K) < anchors:
        selected = torch.topk(anchor_log, int(K), dim=1).indices
    else:
        selected = torch.arange(anchors, device=device).expand(rows, anchors)
    gather_index = selected.unsqueeze(0).expand(int(E), rows, int(K))
    selected_joint_log = torch.gather(joint_log, 2, gather_index)
    joint_weight = torch.softmax(
        selected_joint_log.permute(1, 0, 2).reshape(rows, -1), dim=1
    ).reshape(rows, int(E), int(K))
    support_xy = cache["anchor_support_3d"][:, :2].to(device)
    selected_xy = support_xy[selected]
    member_mass = joint_weight.sum(dim=2)
    conditional = joint_weight / member_mass.unsqueeze(2).clamp_min(1e-12)
    member_mean = torch.sum(
        conditional.unsqueeze(3) * selected_xy.unsqueeze(1), dim=2
    )
    overall_mean = torch.sum(member_mass.unsqueeze(2) * member_mean, dim=1)
    within = torch.sum(
        joint_weight
        * torch.sum(
            (selected_xy.unsqueeze(1) - member_mean.unsqueeze(2)).square(), dim=3
        ),
        dim=(1, 2),
    )
    between = torch.sum(
        member_mass
        * torch.sum((member_mean - overall_mean.unsqueeze(1)).square(), dim=2),
        dim=1,
    )
    total = torch.sum(
        joint_weight
        * torch.sum(
            (selected_xy.unsqueeze(1) - overall_mean[:, None, None, :]).square(),
            dim=3,
        ),
        dim=(1, 2),
    )
    residual = (total - within - between).abs()
    fraction = between / total.clamp_min(1e-12)
    member_entropy = -torch.sum(
        member_mass * torch.log(member_mass.clamp_min(1e-12)), dim=1
    )

    def summarize(vector: torch.Tensor) -> dict:
        vector = vector.double().cpu()
        return {
            "mean": float(vector.mean()),
            "median": float(vector.median()),
            "q10": float(torch.quantile(vector, 0.10)),
            "q90": float(torch.quantile(vector, 0.90)),
        }

    return {
        "rows": rows,
        "E": int(E),
        "K": int(K),
        "lambda": float(lambda_value),
        "trace_within_member_m2": summarize(within),
        "trace_between_member_m2": summarize(between),
        "trace_total_m2": summarize(total),
        "between_fraction_of_total": summarize(fraction),
        "effective_member_count": summarize(torch.exp(member_entropy)),
        "mean_member_masses": [
            float(value) for value in member_mass.double().mean(dim=0).cpu()
        ],
        "covariance_identity_max_abs_residual_m2": float(residual.max().cpu()),
        "covariance_identity_mean_abs_residual_m2": float(residual.mean().cpu()),
    }


def diagnose_locked_members(
    *,
    cache_path: str | Path,
    freeze_path: str | Path = DEFAULT_SATURATION_FREEZE,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path,
    device_name: str = "cuda",
) -> dict:
    """Post-lock member panel; descriptive only and never used for selection."""

    registry = load_registry(registry_path)
    freeze = _load_freeze(freeze_path, registry)
    cache = load_cache(cache_path)
    if cache.get("role") not in {"locked_internal_test", "official_validation"}:
        raise ValueError("member diagnostics require a protected cache")
    if cache.get("protected_access_freeze_sha256") != freeze["freeze_sha256"]:
        raise ValueError("cache was created under a different freeze")
    selected = freeze["selected"]
    E = int(selected["E"])
    K = int(selected["K"])
    lambda_value = float(selected["lambda"])
    individual = []
    individual_errors = []
    for index in range(E):
        member_cache = dict(cache)
        member_cache["member_logits"] = cache["member_logits"][index : index + 1]
        raw = evaluate_cache_configuration(
            member_cache,
            E=1,
            K=K,
            lambda_value=None,
            use_radio=False,
            device_name=device_name,
        )
        marked = evaluate_cache_configuration(
            member_cache,
            E=1,
            K=K,
            lambda_value=lambda_value,
            use_radio=True,
            device_name=device_name,
        )
        individual.append(
            {
                "member_index": index + 1,
                "member_seed": int(cache["checkpoint_member_seeds"][index]),
                "raw": raw["summary"],
                "marked": marked["summary"],
            }
        )
        individual_errors.append(marked["vectors"]["error_2d_m"])
    final = evaluate_cache_configuration(
        cache,
        E=E,
        K=K,
        lambda_value=lambda_value,
        use_radio=True,
        device_name=device_name,
    )
    mean_individual_error = torch.stack(individual_errors, dim=0).mean(dim=0)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "role": cache["role"],
        "diagnostic_status": "post_lock_descriptive_not_used_for_selection",
        "selected": {"E": E, "K": K, "lambda": lambda_value},
        "individual_members": individual,
        "individual_marked_coordinate_macro_mae_mean_m": float(
            np.mean(
                [
                    row["marked"]["coordinate_group_macro_mae_2d_m"]
                    for row in individual
                ]
            )
        ),
        "individual_marked_coordinate_macro_mae_range_m": [
            float(
                min(
                    row["marked"]["coordinate_group_macro_mae_2d_m"]
                    for row in individual
                )
            ),
            float(
                max(
                    row["marked"]["coordinate_group_macro_mae_2d_m"]
                    for row in individual
                )
            ),
        ],
        "ensemble_vs_random_selected_member": paired_coordinate_bootstrap(
            mean_individual_error,
            final["vectors"]["error_2d_m"],
            cache["coordinate_group_ids"],
        ),
        "joint_covariance": joint_covariance_diagnostics(
            cache,
            E=E,
            K=K,
            lambda_value=lambda_value,
            device_name=device_name,
        ),
    }
    _write_json_new(output, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    register.add_argument("--official-data", default=str(DEFAULT_OFFICIAL))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))

    train = subparsers.add_parser("train-member")
    train.add_argument("--member-index", type=int, required=True)
    train.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train.add_argument("--output")
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=5)

    cache = subparsers.add_parser("cache")
    cache.add_argument(
        "--role",
        choices=("development", "locked_internal_test", "official_validation"),
        required=True,
    )
    cache.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    cache.add_argument("--checkpoints", nargs="*")
    cache.add_argument("--output")
    cache.add_argument("--freeze")
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    cache.add_argument(
        "--candidate-chunk-size", type=int, default=DEFAULT_CANDIDATE_CHUNK
    )

    select = subparsers.add_parser("select")
    select.add_argument("--cache", default=str(DEFAULT_DEV_CACHE))
    select.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    select.add_argument("--output", default=str(DEFAULT_FREEZE))
    select.add_argument("--device", default="cuda")

    amend = subparsers.add_parser("amend-saturation")
    amend.add_argument("--exploratory-freeze", default=str(DEFAULT_FREEZE))
    amend.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    amend.add_argument("--output", default=str(DEFAULT_SATURATION_FREEZE))

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--cache", required=True)
    evaluate.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    evaluate.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda")

    diagnose = subparsers.add_parser("diagnose-members")
    diagnose.add_argument("--cache", required=True)
    diagnose.add_argument("--freeze", default=str(DEFAULT_SATURATION_FREEZE))
    diagnose.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "register":
        result = register_protocol(
            train_data=args.train_data,
            official_data=args.official_data,
            output=args.output,
        )
        compact = {
            "registry": str(Path(args.output).resolve()),
            "registry_sha256": result["registry_sha256"],
            "role_counts": {
                name: {
                    "coordinates": record["coordinate_count"],
                    "rows": record["row_count"],
                }
                for name, record in result["roles"].items()
            },
        }
    elif args.command == "train-member":
        compact = train_member(
            member_index=args.member_index,
            registry_path=args.registry,
            output=args.output,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "cache":
        compact = build_cache(
            role=args.role,
            registry_path=args.registry,
            checkpoints=args.checkpoints,
            output=args.output,
            freeze_path=args.freeze,
            device_name=args.device,
            batch_size=args.batch_size,
            candidate_chunk_size=args.candidate_chunk_size,
        )
    elif args.command == "select":
        result = select_joint_grid(
            cache_path=args.cache,
            registry_path=args.registry,
            output=args.output,
            device_name=args.device,
        )
        compact = {
            "freeze": str(Path(args.output).resolve()),
            "selected": result["selected"],
            "boundary_extension_required": result["boundary_extension_required"],
            "boundary_reasons": result["boundary_reasons"],
            "internal_test_authorized": result["internal_test_authorized"],
        }
    elif args.command == "amend-saturation":
        result = amend_to_saturation_freeze(
            exploratory_freeze_path=args.exploratory_freeze,
            registry_path=args.registry,
            output=args.output,
        )
        compact = {
            "freeze": str(Path(args.output).resolve()),
            "selection_revision": result["selection_revision"],
            "selected": result["selected"],
            "saturated_E_values": result["saturated_E_values"],
            "boundary_extension_required": result["boundary_extension_required"],
            "boundary_reasons": result["boundary_reasons"],
            "internal_test_authorized": result["internal_test_authorized"],
        }
    elif args.command == "evaluate":
        result = evaluate_locked(
            cache_path=args.cache,
            freeze_path=args.freeze,
            registry_path=args.registry,
            output=args.output,
            device_name=args.device,
        )
        compact = {
            "output": str(Path(args.output).resolve()),
            "role": result["role"],
            "selected": result["selected"],
            "summaries": result["summaries"],
        }
    else:
        result = diagnose_locked_members(
            cache_path=args.cache,
            freeze_path=args.freeze,
            registry_path=args.registry,
            output=args.output,
            device_name=args.device,
        )
        compact = {
            "output": str(Path(args.output).resolve()),
            "role": result["role"],
            "selected": result["selected"],
            "individual_marked_coordinate_macro_mae_mean_m": result[
                "individual_marked_coordinate_macro_mae_mean_m"
            ],
            "individual_marked_coordinate_macro_mae_range_m": result[
                "individual_marked_coordinate_macro_mae_range_m"
            ],
            "ensemble_vs_random_selected_member": result[
                "ensemble_vs_random_selected_member"
            ],
            "joint_covariance": result["joint_covariance"],
        }
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
