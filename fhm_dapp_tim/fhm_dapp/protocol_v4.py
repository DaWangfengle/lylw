"""Pre-registered data roles for DR-JAM v4.

The v4.4 protocol has two globally quarantined, independent axes:

* one coordinate-group panel is absent from every development fit and score;
* one phone panel is absent from every development fit and score.

All selections are made inside the remaining phone-by-coordinate core.  Five
internal coordinate folds cover that core exactly once for validation; they
are not outer tests.  This distinction is essential: pooling validation from
conventional outer folds would let an outer-test coordinate re-enter another
fold's development data and would not support one globally selected method.

Four phone IDs are selected by a label-blind hash and quarantined from every
mechanism and structural decision.  They are evaluated only after the complete
method is frozen.  ``validationData.csv`` is deliberately absent from this
module: it is a known historical benchmark and has no v4 selection authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords, resolve_path
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.protocol_v3 import outer_group_assignment


PROTOCOL_VERSION = "uji_dr_jam_v4_4_20260809"
SPLIT_SEED = 20260802
COORDINATE_CONFIRMATION_FOLDS = 5
COORDINATE_CONFIRMATION_FOLD = 0
COORDINATE_DEVELOPMENT_FOLDS = 5
COORDINATE_DEVELOPMENT_SEED = SPLIT_SEED + 44
COORDINATE_EARLY_STOP_FRACTION = 0.125
# Backward-compatible internal name.  These are development cross-fitting
# folds, not outer tests; the global coordinate confirmation is common to all.
COORDINATE_OUTER_FOLDS = COORDINATE_DEVELOPMENT_FOLDS
DEVICE_CONFIRMATION_COUNT = 4
DEVICE_EARLY_STOP_FRACTION = 0.125
MECHANISM_MEMBER_SEEDS = (4101, 4102, 4103)
STRUCTURAL_MEMBER_SEEDS = tuple(range(4101, 4109))
# Backward-compatible public name for callers that need the full trainable pool.
MEMBER_SEEDS = STRUCTURAL_MEMBER_SEEDS
DANN_GAMMAS = (0.03, 0.10, 0.30)
DANN_TUNING_PHONE_COUNT = 3
MEMBER_VISIBLE_THRESHOLD = -0.75
MEMBER_MAX_VISIBLE = 64

# This is a mechanism diagnostic, not structural tuning.  Independent E=1
# runs are paired across seeds.  No sparse projection is applied and the radio
# loss has its registered natural scale of mean nats/AP.
MECHANISM_DIAGNOSTIC = {
    "member_count_E": 1,
    "top_k": "full_dictionary",
    "loss_temperature_lambda": 1.0,
    "aggregate_across_seeds": "paired_phone_or_coordinate_group_mean",
    "radio_score": "joint_hurdle_log_score_fixed_J_nats_per_ap",
}

STRUCTURAL_GRID = {
    "member_count_E": (2, 3, 4, 5, 6, 7, 8),
    "top_k": (4, 8, 16, 32, 64),
    "loss_temperature_lambda": (0.25, 0.5, 1.0, 2.0, 4.0),
}


LocationKey = tuple[int, int, float, float]


@dataclass(frozen=True)
class MechanismSpec:
    identifier: str
    member_variant: str
    scan_feature_mode: str
    radio_variant: str
    dann_gamma: float


MECHANISMS: Mapping[str, MechanismSpec] = {
    "M0": MechanismSpec("M0", "raw", "legacy", "diagonal", 0.0),
    "M1-g003": MechanismSpec("M1-g003", "dann", "legacy", "diagonal", 0.03),
    "M1-g010": MechanismSpec("M1-g010", "dann", "legacy", "diagonal", 0.10),
    "M1-g030": MechanismSpec("M1-g030", "dann", "legacy", "diagonal", 0.30),
    "M2": MechanismSpec(
        "M2", "maximal_invariant", "maximal_invariant", "diagonal", 0.0
    ),
    # M2 and M3 intentionally share the exact same trained member.  Only the
    # analytically marginalized radio score changes.
    "M3": MechanismSpec(
        "M3", "maximal_invariant", "maximal_invariant", "rank2", 0.0
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_indices(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def sha256_location_keys(keys: Iterable[LocationKey]) -> str:
    """Hash an unordered coordinate-group set with a canonical encoding."""
    rows = sorted(
        (
            int(key[0]),
            int(key[1]),
            format(float(key[2]), ".4f"),
            format(float(key[3]), ".4f"),
        )
        for key in set(keys)
    )
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def registry_digest(payload: Mapping) -> str:
    body = dict(payload)
    body.pop("registry_sha256", None)
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _stable_digest(seed: int, salt: str, value: object) -> bytes:
    return hashlib.sha256(f"{int(seed)}|{salt}|{value}".encode("utf-8")).digest()


def _as_phone_array(phone_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(phone_ids, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("phone_ids must be one non-empty vector")
    return values


def coordinate_axis_partition(
    keys: list[LocationKey],
    phone_ids: np.ndarray,
    development_phones: Iterable[int],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, frozenset[LocationKey]]:
    """Freeze the one global coordinate confirmation before any selection.

    The universe is formed only from development-phone coordinates, so the
    protected coordinate estimand is observable without using a quarantined
    phone.  A building/floor-stratified hash assignment selects one of five
    panels.  The returned sets are independent of RSS values and outcomes.
    """
    phones = _as_phone_array(phone_ids)
    if len(keys) != len(phones):
        raise ValueError("keys and phone_ids must have equal length")
    allowed = set(map(int, development_phones))
    if not allowed:
        raise ValueError("development_phones must be non-empty")
    observed = set(map(int, np.unique(phones)))
    if not allowed <= observed:
        raise ValueError("development_phones contain an absent phone")
    universe = list(
        dict.fromkeys(
            keys[row]
            for row, phone in enumerate(phones)
            if int(phone) in allowed
        )
    )
    assignment = outer_group_assignment(
        universe,
        folds=COORDINATE_CONFIRMATION_FOLDS,
        seed=int(seed),
    )
    confirmation = frozenset(
        key
        for key, fold in assignment.items()
        if int(fold) == COORDINATE_CONFIRMATION_FOLD
    )
    core = frozenset(set(universe) - set(confirmation))
    if not confirmation or not core or confirmation & core:
        raise RuntimeError("coordinate confirmation/core partition is invalid")
    if confirmation | core != set(universe):
        raise RuntimeError("coordinate partition is not exhaustive")
    return {
        "development_core": core,
        "coordinate_confirmation": confirmation,
    }


def identifiable_phone_ids(
    keys: list[LocationKey], phone_ids: np.ndarray
) -> tuple[int, ...]:
    """Phones with at least one coordinate observed by another phone."""
    phones = _as_phone_array(phone_ids)
    if len(keys) != len(phones):
        raise ValueError("keys and phone_ids must have equal length")
    key_sets = {
        int(phone): {keys[int(row)] for row in np.flatnonzero(phones == phone)}
        for phone in np.unique(phones)
    }
    result = []
    for phone in sorted(key_sets):
        other_keys: set[LocationKey] = set()
        for other, values in key_sets.items():
            if other != phone:
                other_keys.update(values)
        if key_sets[phone] & other_keys:
            result.append(phone)
    return tuple(result)


def select_device_confirmation_phones(
    keys: list[LocationKey],
    phone_ids: np.ndarray,
    *,
    count: int = DEVICE_CONFIRMATION_COUNT,
    seed: int = SPLIT_SEED,
) -> tuple[int, ...]:
    """Select confirmation phones by hash without consulting any outcome."""
    eligible = identifiable_phone_ids(keys, phone_ids)
    if isinstance(count, bool) or not 1 <= int(count) < len(eligible):
        raise ValueError("confirmation count must leave development phones")
    ordered = sorted(
        eligible,
        key=lambda phone: _stable_digest(seed, "device-confirm", phone),
    )
    return tuple(int(phone) for phone in ordered[: int(count)])


def select_dann_tuning_phones(
    development_phone_ids: Iterable[int],
    identifiable_phone_ids_: Iterable[int],
    *,
    count: int = DANN_TUNING_PHONE_COUNT,
    seed: int = SPLIT_SEED,
) -> tuple[int, ...]:
    """Select the DANN-gamma tuning panel without localization outcomes."""
    development = set(map(int, development_phone_ids))
    eligible = development & set(map(int, identifiable_phone_ids_))
    if isinstance(count, bool) or not 1 <= int(count) < len(eligible):
        raise ValueError("DANN tuning count must leave mechanism-selection phones")
    ordered = sorted(
        eligible,
        key=lambda phone: _stable_digest(seed, "dann-tuning", phone),
    )
    return tuple(ordered[: int(count)])


def matched_coordinate_lopo_roles(
    keys: list[LocationKey],
    phone_ids: np.ndarray,
    held_out_phone: int,
    *,
    training_phones: Iterable[int] | None = None,
    allowed_coordinate_keys: Iterable[LocationKey] | None = None,
) -> dict[str, np.ndarray]:
    """Return train, identifiable test, stress-test and quarantined rows.

    ``matched_test`` contains held-out-phone rows whose coordinate is present
    among the allowed training phones.  ``combined_shift_test`` contains the
    remaining held-out rows and must never be called a pure device effect.
    """
    phones = _as_phone_array(phone_ids)
    if len(keys) != len(phones):
        raise ValueError("keys and phone_ids must have equal length")
    held_out = int(held_out_phone)
    observed = set(map(int, np.unique(phones)))
    if held_out not in observed:
        raise ValueError("held_out_phone is absent")
    allowed = (
        observed - {held_out}
        if training_phones is None
        else set(map(int, training_phones))
    )
    if held_out in allowed:
        raise ValueError("held-out phone cannot be a training phone")
    if not allowed <= observed:
        raise ValueError("training_phones contain an absent phone")

    allowed_keys = (
        set(keys)
        if allowed_coordinate_keys is None
        else set(allowed_coordinate_keys)
    )
    if not allowed_keys:
        raise ValueError("allowed_coordinate_keys must be non-empty")
    if not allowed_keys <= set(keys):
        raise ValueError("allowed_coordinate_keys contain an absent coordinate")

    train = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if int(phone) in allowed and keys[row] in allowed_keys
        ],
        dtype=np.int64,
    )
    train_keys = {keys[int(row)] for row in train}
    matched = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if (
                int(phone) == held_out
                and keys[row] in allowed_keys
                and keys[row] in train_keys
            )
        ],
        dtype=np.int64,
    )
    combined = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if (
                int(phone) == held_out
                and keys[row] in allowed_keys
                and keys[row] not in train_keys
            )
        ],
        dtype=np.int64,
    )
    coordinate_quarantined = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if (
                (int(phone) == held_out or int(phone) in allowed)
                and keys[row] not in allowed_keys
            )
        ],
        dtype=np.int64,
    )
    quarantined = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if int(phone) != held_out and int(phone) not in allowed
        ],
        dtype=np.int64,
    )
    roles = {
        "refit_train": train,
        "matched_test": matched,
        "combined_shift_test": combined,
        "coordinate_confirmation_quarantine": coordinate_quarantined,
        "quarantined": quarantined,
    }
    _validate_row_partition(len(keys), roles)
    return roles


def device_epoch_selection_roles(
    keys: list[LocationKey],
    phone_ids: np.ndarray,
    held_out_phone: int,
    *,
    training_phones: Iterable[int] | None = None,
    allowed_coordinate_keys: Iterable[LocationKey] | None = None,
    stop_fraction: float = DEVICE_EARLY_STOP_FRACTION,
    seed: int = SPLIT_SEED,
) -> dict[str, np.ndarray]:
    """Split allowed training coordinates for epoch selection, then refit all."""
    if not 0.0 < float(stop_fraction) < 0.5:
        raise ValueError("stop_fraction must lie in (0,0.5)")
    base = matched_coordinate_lopo_roles(
        keys,
        phone_ids,
        held_out_phone,
        training_phones=training_phones,
        allowed_coordinate_keys=allowed_coordinate_keys,
    )
    refit = base["refit_train"]
    unique_keys = list(dict.fromkeys(keys[int(row)] for row in refit))
    strata: dict[tuple[int, int], list[LocationKey]] = {}
    for key in unique_keys:
        strata.setdefault(key[:2], []).append(key)
    stop_keys: set[LocationKey] = set()
    for stratum, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda key: _stable_digest(
                seed, f"device-stop-{int(held_out_phone)}-{stratum}", key
            ),
        )
        count = max(1, int(round(len(ordered) * float(stop_fraction))))
        if count >= len(ordered):
            count = max(0, len(ordered) - 1)
        stop_keys.update(ordered[:count])
    early_stop = np.asarray(
        [row for row in refit if keys[int(row)] in stop_keys], dtype=np.int64
    )
    member_fit = np.asarray(
        [row for row in refit if keys[int(row)] not in stop_keys], dtype=np.int64
    )
    if len(member_fit) == 0 or len(early_stop) == 0:
        raise ValueError("device epoch-selection split is empty")
    if {keys[int(row)] for row in member_fit} & {
        keys[int(row)] for row in early_stop
    }:
        raise RuntimeError("coordinate leakage in device epoch selection")
    return {**base, "member_fit": member_fit, "early_stop": early_stop}


def coordinate_roles_for_development_phones(
    keys: list[LocationKey],
    phone_ids: np.ndarray,
    development_phones: Iterable[int],
    outer_fold: int,
    *,
    folds: int = COORDINATE_OUTER_FOLDS,
    seed: int = SPLIT_SEED,
) -> dict[str, np.ndarray]:
    """Map global confirmation plus inner-development roles to global rows.

    ``outer_test`` is retained as an internal compatibility name for the one
    global coordinate confirmation panel.  It is identical in every internal
    fold.  ``structure_development`` partitions the non-protected core exactly
    once across folds; it is the only coordinate selection-evaluation role.
    """
    phones = _as_phone_array(phone_ids)
    allowed = set(map(int, development_phones))
    fold_count = int(folds)
    if isinstance(outer_fold, bool) or not 0 <= int(outer_fold) < fold_count:
        raise ValueError("outer_fold must index one development fold")
    fold = int(outer_fold)
    partition = coordinate_axis_partition(
        keys, phones, allowed, seed=int(seed)
    )
    core = set(partition["development_core"])
    confirmation = set(partition["coordinate_confirmation"])
    core_assignment = outer_group_assignment(
        list(core),
        folds=fold_count,
        seed=COORDINATE_DEVELOPMENT_SEED,
    )
    validation = {
        key for key, assigned in core_assignment.items() if int(assigned) == fold
    }
    training_keys = set(core) - validation

    stop_keys: set[LocationKey] = set()
    strata: dict[tuple[int, int], list[LocationKey]] = {}
    for key in training_keys:
        strata.setdefault(key[:2], []).append(key)
    for stratum, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda key: _stable_digest(
                COORDINATE_DEVELOPMENT_SEED,
                f"coordinate-stop-{fold}-{stratum}",
                key,
            ),
        )
        count = max(
            1,
            int(round(len(ordered) * COORDINATE_EARLY_STOP_FRACTION)),
        )
        if count >= len(ordered):
            count = max(0, len(ordered) - 1)
        stop_keys.update(ordered[:count])
    fit_keys = training_keys - stop_keys
    if not validation or not stop_keys or not fit_keys:
        raise ValueError("coordinate development fold contains an empty role")

    role_by_key = {
        **{key: "outer_test" for key in confirmation},
        **{key: "structure_development" for key in validation},
        **{key: "early_stop" for key in stop_keys},
        **{key: "member_fit" for key in fit_keys},
    }
    role_names = (
        "member_fit",
        "early_stop",
        "structure_development",
        "outer_test",
    )
    roles = {
        name: np.asarray(
            [
                row
                for row, phone in enumerate(phones)
                if int(phone) in allowed and role_by_key[keys[row]] == name
            ],
            dtype=np.int64,
        )
        for name in role_names
    }
    roles["device_confirmation_quarantine"] = np.asarray(
        [row for row, phone in enumerate(phones) if int(phone) not in allowed],
        dtype=np.int64,
    )
    _validate_row_partition(len(keys), roles)
    key_sets = {
        name: {keys[int(row)] for row in roles[name]}
        for name in role_names
    }
    for index, left in enumerate(role_names):
        for right in role_names[index + 1 :]:
            if key_sets[left] & key_sets[right]:
                raise RuntimeError(
                    f"coordinate-group leakage between {left} and {right}"
                )
    return roles


def _validate_row_partition(row_count: int, roles: Mapping[str, np.ndarray]) -> None:
    sets = {name: set(map(int, values)) for name, values in roles.items()}
    names = list(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if sets[left] & sets[right]:
                raise ValueError(f"row overlap between {left} and {right}")
    if set().union(*sets.values()) != set(range(int(row_count))):
        raise ValueError("roles are not row-exhaustive")


def build_registry(train_data: str | Path) -> dict:
    """Create the label-blind v4 registry from ``trainingData.csv`` only."""
    data_path = resolve_path(str(train_data)).resolve()
    if data_path.name.lower() != "trainingdata.csv":
        raise ValueError("v4 registry accepts UJI trainingData.csv only")
    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        timestamp_t,
        stats,
    ) = load_dataset(data_path, dataset_type="UJI")
    raw_xy = raw_coords(coords_norm, building_t, stats)
    building = building_t.numpy()
    floor = floor_t.numpy()
    phones = phone_t.numpy()
    keys = location_keys(raw_xy, building, floor)
    confirmation = select_device_confirmation_phones(keys, phones)
    observed = tuple(sorted(map(int, np.unique(phones))))
    development = tuple(phone for phone in observed if phone not in confirmation)
    global_identifiable = identifiable_phone_ids(keys, phones)
    global_nonidentifiable = tuple(
        phone for phone in observed if phone not in global_identifiable
    )

    coordinate_partition = coordinate_axis_partition(
        keys, phones, development
    )
    coordinate_core = set(coordinate_partition["development_core"])
    coordinate_confirmation = set(
        coordinate_partition["coordinate_confirmation"]
    )

    # Device identifiability must be recomputed after *both* confirmation axes
    # are quarantined.  A phone can share coordinates only with a quarantined
    # phone or only at a protected coordinate; neither identifies a pure device
    # effect inside the development core.
    development_rows = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if int(phone) in development and keys[row] in coordinate_core
        ],
        dtype=np.int64,
    )
    development_identifiable = identifiable_phone_ids(
        [keys[int(row)] for row in development_rows],
        phones[development_rows],
    )
    development_nonidentifiable = tuple(
        phone for phone in development if phone not in development_identifiable
    )
    dann_tuning = select_dann_tuning_phones(
        development, development_identifiable, count=DANN_TUNING_PHONE_COUNT
    )
    mechanism_selection = tuple(
        phone
        for phone in development
        if phone in development_identifiable and phone not in dann_tuning
    )

    visible_counts = (rss > float(MEMBER_VISIBLE_THRESHOLD)).sum(dim=1)
    maximum_visible = int(visible_counts.max())
    if maximum_visible > MEMBER_MAX_VISIBLE:
        raise ValueError(
            "registered maximal-invariant input would truncate visible APs: "
            f"maximum={maximum_visible}, limit={MEMBER_MAX_VISIBLE}"
        )

    coordinate_audit = []
    coordinate_validation_counts: dict[LocationKey, int] = {
        key: 0 for key in coordinate_core
    }
    common_coordinate_confirmation_rows: np.ndarray | None = None
    for fold in range(COORDINATE_OUTER_FOLDS):
        roles = coordinate_roles_for_development_phones(
            keys, phones, development, fold
        )
        if common_coordinate_confirmation_rows is None:
            common_coordinate_confirmation_rows = roles["outer_test"]
        elif not np.array_equal(
            common_coordinate_confirmation_rows, roles["outer_test"]
        ):
            raise RuntimeError(
                "global coordinate confirmation differs across development folds"
            )
        for key in {keys[int(row)] for row in roles["structure_development"]}:
            coordinate_validation_counts[key] += 1
        coordinate_audit.append(
            {
                "outer_fold": fold,
                "roles": {
                    name: {
                        "rows": int(len(indices)),
                        "coordinate_groups": int(
                            len({keys[int(row)] for row in indices})
                        ),
                        "coordinate_group_sha256": sha256_location_keys(
                            keys[int(row)] for row in indices
                        ),
                        "row_index_sha256": sha256_indices(indices),
                    }
                    for name, indices in roles.items()
                },
            }
        )
    if set(coordinate_validation_counts.values()) != {1}:
        raise RuntimeError(
            "development-core coordinates are not validated exactly once"
        )

    device_audit = []
    for held_out in development:
        training_phones = tuple(
            phone for phone in development if phone != held_out
        )
        roles = device_epoch_selection_roles(
            keys,
            phones,
            held_out,
            training_phones=training_phones,
            allowed_coordinate_keys=coordinate_core,
        )
        role_identifiable = bool(len(roles["matched_test"]))
        expected_identifiable = held_out in development_identifiable
        if role_identifiable != expected_identifiable:
            raise RuntimeError(
                "development identifiability and matched-test role disagree "
                f"for phone {held_out}"
            )
        device_audit.append(
            {
                "held_out_phone": held_out,
                "identifiable": role_identifiable,
                "roles": {
                    name: {
                        "rows": int(len(indices)),
                        "coordinate_groups": int(
                            len({keys[int(row)] for row in indices})
                        ),
                        "coordinate_group_sha256": sha256_location_keys(
                            keys[int(row)] for row in indices
                        ),
                        "row_index_sha256": sha256_indices(indices),
                    }
                    for name, indices in roles.items()
                },
            }
        )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_without_localization_predictions_or_errors": True,
        "official_validation_selection_authority": False,
        "training_data": str(data_path),
        "training_data_sha256": _sha256_file(data_path),
        "rows": len(keys),
        "coordinate_groups": len(set(keys)),
        "coordinate_axis": {
            "method": (
                "building/floor-stratified SHA-256 assignment; one fixed "
                "panel globally quarantined before all selections"
            ),
            "seed": SPLIT_SEED,
            "panel_count": COORDINATE_CONFIRMATION_FOLDS,
            "confirmation_panel": COORDINATE_CONFIRMATION_FOLD,
            "development_crossfit_folds": COORDINATE_DEVELOPMENT_FOLDS,
            "development_crossfit_seed": COORDINATE_DEVELOPMENT_SEED,
            "development_core_coordinate_groups": len(coordinate_core),
            "development_core_coordinate_group_sha256": (
                sha256_location_keys(coordinate_core)
            ),
            "coordinate_confirmation_groups": len(coordinate_confirmation),
            "coordinate_confirmation_group_sha256": (
                sha256_location_keys(coordinate_confirmation)
            ),
            "selection_contract": (
                "each development-core coordinate is selection validation "
                "in exactly one internal fold; confirmation coordinates "
                "never enter any development fit, early stop, or score"
            ),
        },
        "observed_phone_ids": list(observed),
        "global_identifiable_phone_ids": list(global_identifiable),
        "global_nonidentifiable_phone_ids": list(global_nonidentifiable),
        "development_identifiable_phone_ids": list(development_identifiable),
        "development_combined_shift_phone_ids": list(
            development_nonidentifiable
        ),
        "device_confirmation_selection": {
            "method": "sha256(seed|device-confirm|phone), first identifiable IDs",
            "seed": SPLIT_SEED,
            "count": DEVICE_CONFIRMATION_COUNT,
            "phone_ids": list(confirmation),
        },
        "device_development_phone_ids": list(development),
        "mechanism_member_seeds": list(MECHANISM_MEMBER_SEEDS),
        "structural_member_seeds": list(STRUCTURAL_MEMBER_SEEDS),
        "mechanisms": {
            name: asdict(spec) for name, spec in MECHANISMS.items()
        },
        "dann_gamma_grid": list(DANN_GAMMAS),
        "nested_device_development": {
            "dann_tuning_selection": (
                "sha256(seed|dann-tuning|phone), first identifiable development IDs"
            ),
            "dann_tuning_phone_ids": list(dann_tuning),
            "mechanism_selection_phone_ids": list(mechanism_selection),
            "combined_shift_stress_phone_ids": list(
                development_nonidentifiable
            ),
            "order": [
                "select DANN gamma inside the phone-by-coordinate development core",
                "freeze gamma",
                "select M0/M1/M2/M3 inside the same development core",
                "freeze mechanism",
                "jointly select E,K,lambda inside coordinate cross-fitting",
                "refit once on the full development core",
                "evaluate coordinate and phone confirmations by shift stratum",
            ],
        },
        "mechanism_diagnostic": dict(MECHANISM_DIAGNOSTIC),
        "common_estimation_contract": {
            "anchor_radio_memory": "equal_phone_within_coordinate_v4",
            "member_sampling": "equal_phone_equal_coordinate_equal_row",
            "member_score": "exact_full_dictionary_3d_energy_score",
            "energy_approximation_width_L": None,
            "energy_score_is_exact": True,
            "member_visible_threshold": MEMBER_VISIBLE_THRESHOLD,
            "member_max_visible": MEMBER_MAX_VISIBLE,
            "observed_maximum_visible_aps": maximum_visible,
            "member_invariant_location": "visible_median",
            "member_invariant_scale": (
                "mean_absolute_deviation_about_visible_median"
            ),
            "member_invariant_clipping": None,
            "radio_score": (
                "joint_hurdle_log_score_divided_by_fixed_dictionary_width_J"
            ),
        },
        "structural_grid": {
            name: list(values) for name, values in STRUCTURAL_GRID.items()
        },
        "acceptance_gates": {
            "coordinate_mae_noninferiority_margin_m": 0.50,
            "device_macro_mae_noninferiority_margin_m": 0.50,
            "one_sided_confidence": 0.95,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": SPLIT_SEED + 43,
            "interval_unit": "coordinate_group_or_phone paired bootstrap",
        },
        "selection_rules": {
            "dann_gamma": {
                "coordinate_safety_gate": (
                    "coordinate-group macro MAE no more than 0.50 m above "
                    "the best gamma"
                ),
                "primary": (
                    "minimum equal-phone macro of within-phone "
                    "coordinate-group macro matched MAE after averaging "
                    "paired seeds within phone"
                ),
                "practical_tie_m": 0.05,
                "tie_break": (
                    "smallest gamma within 0.05 m of the eligible device-MAE best"
                ),
                "uncertainty": (
                    "paired phone and coordinate-group bootstrap intervals "
                    "are reported but do not replace the frozen point rule"
                ),
            },
            "mechanism": {
                "utility_gates": (
                    "one-sided 95% paired-bootstrap upper bounds for coordinate "
                    "and device macro MAE are each at most 0.50 m above the "
                    "best standard control M0/M1"
                ),
                "primary": (
                    "minimum equal-phone macro of within-phone "
                    "coordinate-group macro matched MAE after averaging "
                    "paired seeds within phone"
                ),
                "practical_tie_m": 0.10,
                "scientific_secondary": (
                    "within 0.10 m of the eligible device-MAE best, minimize "
                    "phone-macro true-anchor joint radio log score"
                ),
                "complexity_tie_break": ["M0", "M2", "M3", "M1"],
            },
            "structure": {
                "joint_parameters": [
                    "member_count_E", "top_k", "loss_temperature_lambda"
                ],
                "primary": "coordinate-group macro 3-D energy score",
                "accuracy_guard": (
                    "coordinate-group macro MAE within 0.50 m of the grid best"
                ),
                "saturation_rule": (
                    "among configurations within one internal-fold standard "
                    "error of the best energy score, minimize E*K; then E, "
                    "then K, then |log(lambda)|"
                ),
                "member_subsets": (
                    "nested registered seed prefixes; all-subset curves are "
                    "diagnostic only"
                ),
            },
        },
        "coordinate_development_audit": coordinate_audit,
        "device_development_audit": device_audit,
        "timestamp_dtype": str(timestamp_t.numpy().dtype),
    }
    payload["registry_sha256"] = registry_digest(payload)
    return payload


def validate_registry(path: str | Path, *, train_data: str | Path) -> dict:
    registry_path = Path(path).resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("v4 registry protocol mismatch")
    if payload.get("official_validation_selection_authority") is not False:
        raise ValueError("official validation was given selection authority")
    if payload.get("created_without_localization_predictions_or_errors") is not True:
        raise ValueError("v4.4 registry provenance declaration is absent")
    if payload.get("registry_sha256") != registry_digest(payload):
        raise ValueError("v4 registry digest mismatch")
    data_path = resolve_path(str(train_data)).resolve()
    if data_path.name.lower() != "trainingdata.csv":
        raise ValueError("v4 training must use trainingData.csv")
    if payload.get("training_data_sha256") != _sha256_file(data_path):
        raise ValueError("training data changed after v4 registration")
    confirmation = set(map(
        int, payload["device_confirmation_selection"]["phone_ids"]
    ))
    development = set(map(int, payload["device_development_phone_ids"]))
    identifiable = set(map(
        int, payload["development_identifiable_phone_ids"]
    ))
    stress = set(map(
        int, payload["development_combined_shift_phone_ids"]
    ))
    nested = payload["nested_device_development"]
    dann = set(map(int, nested["dann_tuning_phone_ids"]))
    mechanism = set(map(int, nested["mechanism_selection_phone_ids"]))
    if confirmation & development:
        raise ValueError("confirmation and development phones overlap")
    if identifiable | stress != development or identifiable & stress:
        raise ValueError("development identifiability partition is invalid")
    if dann & mechanism or dann | mechanism != identifiable:
        raise ValueError("DANN and mechanism panels do not partition identifiable development")
    coordinate_axis = payload.get("coordinate_axis", {})
    if int(coordinate_axis.get("development_crossfit_folds", -1)) != len(
        payload.get("coordinate_development_audit", [])
    ):
        raise ValueError("coordinate development fold count is invalid")
    coordinate_rows = payload["coordinate_development_audit"]
    confirmation_hashes = {
        row["roles"]["outer_test"]["row_index_sha256"]
        for row in coordinate_rows
    }
    confirmation_key_hashes = {
        row["roles"]["outer_test"]["coordinate_group_sha256"]
        for row in coordinate_rows
    }
    if len(confirmation_hashes) != 1 or len(confirmation_key_hashes) != 1:
        raise ValueError("coordinate confirmation is not global across folds")
    if next(iter(confirmation_key_hashes)) != coordinate_axis.get(
        "coordinate_confirmation_group_sha256"
    ):
        raise ValueError("coordinate confirmation key hash is inconsistent")
    validation_groups = sum(
        int(row["roles"]["structure_development"]["coordinate_groups"])
        for row in coordinate_rows
    )
    if validation_groups != int(
        coordinate_axis.get("development_core_coordinate_groups", -1)
    ):
        raise ValueError(
            "coordinate cross-fitting does not cover the development core once"
        )
    audit = {
        int(row["held_out_phone"]): row
        for row in payload["device_development_audit"]
    }
    for phone in identifiable:
        if not audit[phone]["identifiable"]:
            raise ValueError(f"selection phone {phone} has no matched test role")
        if int(audit[phone]["roles"]["matched_test"]["rows"]) <= 0:
            raise ValueError(f"selection phone {phone} has an empty matched test")
    for phone in stress:
        if audit[phone]["identifiable"]:
            raise ValueError(f"stress phone {phone} unexpectedly has a matched test")
    for row in payload["device_development_audit"]:
        if int(
            row["roles"]["coordinate_confirmation_quarantine"][
                "coordinate_groups"
            ]
        ) <= 0:
            raise ValueError("device development did not quarantine coordinates")
    contract = payload["common_estimation_contract"]
    if int(contract["observed_maximum_visible_aps"]) > int(
        contract["member_max_visible"]
    ):
        raise ValueError("maximal-invariant input is truncated by max_visible")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the DR-JAM v4 protocol")
    parser.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v4 registry: {output}")
    payload = build_registry(args.train_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "protocol_version": payload["protocol_version"],
                "registry_sha256": payload["registry_sha256"],
                "rows": payload["rows"],
                "coordinate_groups": payload["coordinate_groups"],
                "device_development_phone_ids": payload[
                    "device_development_phone_ids"
                ],
                "device_confirmation_phone_ids": payload[
                    "device_confirmation_selection"
                ]["phone_ids"],
                "dann_tuning_phone_ids": payload["nested_device_development"][
                    "dann_tuning_phone_ids"
                ],
                "mechanism_selection_phone_ids": payload[
                    "nested_device_development"
                ]["mechanism_selection_phone_ids"],
                "development_combined_shift_phone_ids": payload[
                    "development_combined_shift_phone_ids"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
