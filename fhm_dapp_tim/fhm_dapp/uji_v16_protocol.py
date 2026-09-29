"""Draft coordinate-group development protocol for the single V16 law.

This module fixes data roles and information-budget constants without reading
localization predictions.  Registration remains a later step: the execution
driver and all hashes must be complete before any V16 training is authorized.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from fhm_dapp_tim.fhm_dapp.audit_bplus_sparse50_fem_v5 import (
    _hull_local_indices,
    _stable_rank,
)
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _load_uji,
    _role_record,
    _sha256_file,
    support_fit_stop_roles,
)


PROTOCOL_VERSION = "uji_v16_orbit_barycentric_crossfit_v1_20260817"
FOLD_COUNT = 5
FOLD_SEED = 16101
EARLY_STOP_SEEDS = tuple(16201 + value for value in range(FOLD_COUNT))
MEMBER_SEEDS = tuple(16301 + value for value in range(12))
E_GRID = tuple(range(1, len(MEMBER_SEEDS) + 1))
ELIGIBLE_E_GRID = tuple(range(2, len(MEMBER_SEEDS) + 1))
K_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
K_UPPER_EXTENSION = (256,)
LAMBDA_GRID = tuple(float(2.0**power) for power in range(-12, 4))
LAMBDA_LOWER_EXTENSION = tuple(float(2.0**power) for power in range(-16, -12))
LAMBDA_UPPER_EXTENSION = tuple(float(2.0**power) for power in range(4, 8))
EARLY_STOP_FRACTION = 0.15
SELECTION_PARTITION_SEED = 16401
OPERATING_TUNING_FRACTION = 2.0 / 3.0
MAE_NONINFERIORITY_MARGIN_M = 0.25
SATURATION_TOLERANCE_FRACTION = 0.01
SATURATION_MINIMUM_LATER_E = 2
BOOTSTRAP_REPETITIONS = 5000
SATURATION_INFERENCE_SEED = 16402
SCORING_SEED = 16403

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "evidence" / "v16_uji"
DEFAULT_MODELS = ROOT / "models" / "v16_uji_crossfit"
DEFAULT_FINAL_MODELS = ROOT / "models" / "v16_uji_final"
DEFAULT_TRAIN = ROOT / "data" / "UJIndoorLoc" / "trainingData.csv"
DEFAULT_OFFICIAL = ROOT / "data" / "UJIndoorLoc" / "validationData.csv"
DEFAULT_MECHANISM_DESIGN = (
    DEFAULT_EVIDENCE / "DESIGN_V16_ORBIT_BARYCENTRIC_CHART_DRAFT.md"
)
DEFAULT_PROTOCOL_DESIGN = (
    DEFAULT_EVIDENCE / "DESIGN_V16_UJI_DEVELOPMENT_PROTOCOL_DRAFT.md"
)
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "registry_v16_uji_v1.json"
DEFAULT_SUMMARY = DEFAULT_EVIDENCE / "oof_summary_v16.json"
DEFAULT_FREEZE = DEFAULT_EVIDENCE / "selection_freeze_v16.json"
DEFAULT_PROTOCOL_SOURCE = Path(__file__).resolve()
DEFAULT_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v16_experiment.py"
DEFAULT_MODEL_SOURCE = ROOT / "fhm_dapp" / "orbit_barycentric_v16.py"
DEFAULT_MODEL_BASE_SOURCE = ROOT / "fhm_dapp" / "identified_kernel_v12.py"
DEFAULT_MARKED_SOURCE = ROOT / "fhm_dapp" / "marked_orbit_v10.py"
DEFAULT_GEOMETRY_SOURCE = ROOT / "fhm_dapp" / "coherent_marked_v9.py"
DEFAULT_ANCHOR_SOURCE = ROOT / "fhm_dapp" / "anchor.py"
DEFAULT_MEMORY_SOURCE = ROOT / "fhm_dapp" / "experiment_v71.py"
DEFAULT_RADIO_ENERGY_SOURCE = ROOT / "fhm_dapp" / "dapp.py"
DEFAULT_ENCODER_SOURCE = ROOT / "fhm_dapp" / "model.py"
DEFAULT_SPLIT_PRIMITIVE_SOURCE = ROOT / "fhm_dapp" / "experiment_v7.py"
DEFAULT_HULL_SOURCE = ROOT / "fhm_dapp" / "audit_bplus_sparse50_fem_v5.py"
DEFAULT_METRIC_SOURCE = ROOT / "fhm_dapp" / "uji_v9_experiment.py"
DEFAULT_PROTOCOL_TEST = ROOT.parent / "tests" / "test_uji_v16_protocol.py"
DEFAULT_EXECUTION_TEST = ROOT.parent / "tests" / "test_uji_v16_experiment.py"
DEFAULT_MECHANISM_TEST = ROOT.parent / "tests" / "test_orbit_barycentric_v16.py"


def crossfit_checkpoint_path(fold: int, member_index: int) -> Path:
    return (
        DEFAULT_MODELS
        / f"fold_{int(fold)}"
        / f"member_{int(member_index):02d}_seed{MEMBER_SEEDS[int(member_index)-1]}.pt"
    )


def fold_radio_path(fold: int) -> Path:
    return DEFAULT_EVIDENCE / f"radio_memory_fold_{int(fold)}.pt"


def fold_cache_path(fold: int) -> Path:
    return DEFAULT_EVIDENCE / f"oof_cache_fold_{int(fold)}.pt"


def fold_grid_path(fold: int) -> Path:
    return DEFAULT_EVIDENCE / f"oof_grid_fold_{int(fold)}.pt"


def load_uji_v16(path: str | Path) -> dict:
    """Load RSS while preserving exact raw-CSV float64 coordinate keys."""

    source = Path(path).resolve()
    data = _load_uji(source)
    frame = pd.read_csv(
        source,
        usecols=["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"],
    )
    if len(frame) != len(data["keys"]):
        raise ValueError("V16 raw-coordinate row count differs from RSS loader")
    raw_xy = frame[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=np.float64)
    building = frame["BUILDINGID"].to_numpy(dtype=np.int64)
    floor = frame["FLOOR"].to_numpy(dtype=np.int64)
    if not np.array_equal(building, data["building"]):
        raise ValueError("V16 raw-coordinate building labels changed")
    if not np.array_equal(floor, data["floor"]):
        raise ValueError("V16 raw-coordinate floor labels changed")
    if not bool(np.isfinite(raw_xy).all()):
        raise ValueError("V16 raw CSV coordinates are not finite")
    data["raw_xy"] = raw_xy
    data["keys"] = [
        (int(b), int(f), float(x), float(y))
        for b, f, (x, y) in zip(building, floor, raw_xy)
    ]
    data["coordinate_precision_contract"] = "raw_csv_float64_before_centering"
    return data


def _keys_by_stratum(keys: Iterable[tuple]) -> dict[tuple[int, int], list[tuple]]:
    result: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for key in set(keys):
        result[(int(key[0]), int(key[1]))].append(key)
    return {stratum: sorted(values) for stratum, values in sorted(result.items())}


def build_crossfit_roles(all_keys: set[tuple]) -> tuple[set[tuple], list[dict], dict]:
    """Build deterministic hull-supported five-fold coordinate roles."""

    support_core: set[tuple] = set()
    oof_sets = [set() for _ in range(FOLD_COUNT)]
    stratum_details: dict[str, dict] = {}
    for stratum, ordered in _keys_by_stratum(all_keys).items():
        xy = np.asarray([[key[2], key[3]] for key in ordered], dtype=np.float64)
        hull = {ordered[int(index)] for index in _hull_local_indices(xy)}
        support_core.update(hull)
        interior = [key for key in ordered if key not in hull]
        ranked = sorted(
            interior,
            key=lambda key: _stable_rank(
                FOLD_SEED + 7919 * stratum[0] + 104729 * stratum[1],
                "v16-oof-fold",
                key,
            ),
        )
        offset = int.from_bytes(
            _stable_rank(
                FOLD_SEED,
                "v16-fold-offset",
                (stratum[0], stratum[1], 0.0, 0.0),
            )[:8],
            "big",
        ) % FOLD_COUNT
        for rank, key in enumerate(ranked):
            oof_sets[(rank + offset) % FOLD_COUNT].add(key)
        stratum_details[f"building_{stratum[0]}_floor_{stratum[1]}"] = {
            "coordinates": len(ordered),
            "support_core_hull_coordinates": len(hull),
            "interior_coordinates": len(interior),
            "oof_coordinates_by_fold": [
                sum(key[:2] == stratum for key in values) for values in oof_sets
            ],
        }

    interior_union = set().union(*oof_sets)
    if support_core & interior_union or support_core | interior_union != all_keys:
        raise RuntimeError("V16 support core and OOF interior do not partition coordinates")
    for left in range(FOLD_COUNT):
        for right in range(left):
            if oof_sets[left] & oof_sets[right]:
                raise RuntimeError("V16 OOF folds overlap")

    folds = []
    for fold, query in enumerate(oof_sets):
        refit = all_keys - query
        nested, nested_details = support_fit_stop_roles(
            refit,
            seed=EARLY_STOP_SEEDS[fold],
            early_stop_fraction=EARLY_STOP_FRACTION,
        )
        if support_core - nested["member_fit"]:
            raise RuntimeError("a V16 support-core coordinate escaped member fit")
        if nested["member_fit"] & nested["early_stop"]:
            raise RuntimeError("V16 member-fit and early-stop roles overlap")
        if nested["member_fit"] | nested["early_stop"] != refit:
            raise RuntimeError("V16 nested roles do not reconstruct fold refit")
        folds.append(
            {
                "fold": fold,
                "member_fit_keys": nested["member_fit"],
                "early_stop_keys": nested["early_stop"],
                "fold_refit_keys": refit,
                "oof_query_keys": query,
                "nested_details": nested_details,
            }
        )
    return support_core, folds, stratum_details


def partition_selection_roles(
    interior_keys: set[tuple],
) -> tuple[set[tuple], set[tuple], dict]:
    """Split pooled OOF coordinates into tuning and E-saturation roles."""

    operating: set[tuple] = set()
    saturation: set[tuple] = set()
    details: dict[str, dict[str, int]] = {}
    for stratum, ordered in _keys_by_stratum(interior_keys).items():
        ranked = sorted(
            ordered,
            key=lambda key: _stable_rank(
                SELECTION_PARTITION_SEED
                + 8191 * stratum[0]
                + 131071 * stratum[1],
                "v16-operating-vs-saturation",
                key,
            ),
        )
        # One position in each deterministic block of three is held for the
        # member-saturation decision; the stratum-specific offset avoids a
        # coordinate-order convention becoming a role convention.
        offset_digest = hashlib.sha256(
            f"{SELECTION_PARTITION_SEED}:{stratum[0]}:{stratum[1]}".encode()
        ).digest()
        saturation_position = int.from_bytes(offset_digest[:8], "big") % 3
        for rank, key in enumerate(ranked):
            if rank % 3 == saturation_position:
                saturation.add(key)
            else:
                operating.add(key)
        details[f"building_{stratum[0]}_floor_{stratum[1]}"] = {
            "interior_coordinates": len(ranked),
            "operating_tuning_coordinates": sum(key in operating for key in ranked),
            "member_saturation_coordinates": sum(key in saturation for key in ranked),
        }

    if operating & saturation or operating | saturation != interior_keys:
        raise RuntimeError("V16 selection roles do not partition OOF coordinates")
    if not operating or not saturation:
        raise RuntimeError("V16 selection role is empty")
    return operating, saturation, details


def split_audit(
    train_data: str | Path = DEFAULT_TRAIN,
    official_data: str | Path = DEFAULT_OFFICIAL,
) -> dict:
    """Return a prediction-free audit of V16 coordinate roles."""

    train = load_uji_v16(train_data)
    official = load_uji_v16(official_data)
    train_keys = set(train["keys"])
    official_keys = set(official["keys"])
    core, folds, strata = build_crossfit_roles(train_keys)
    interior = train_keys - core
    operating, saturation, selection_strata = partition_selection_roles(interior)
    queries = [payload["oof_query_keys"] for payload in folds]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "training_rows": len(train["keys"]),
        "training_coordinates": len(train_keys),
        "support_core_coordinates": len(core),
        "interior_oof_coordinates": len(interior),
        "fold_oof_coordinates": [len(value) for value in queries],
        "fold_refit_coordinates": [len(payload["fold_refit_keys"]) for payload in folds],
        "fold_member_fit_coordinates": [len(payload["member_fit_keys"]) for payload in folds],
        "fold_early_stop_coordinates": [len(payload["early_stop_keys"]) for payload in folds],
        "operating_tuning_coordinates": len(operating),
        "member_saturation_coordinates": len(saturation),
        "official_rows": len(official["keys"]),
        "official_coordinates": len(official_keys),
        "official_overlap_coordinates": len(train_keys & official_keys),
        "official_overlap_rows": sum(key in train_keys for key in official["keys"]),
        "official_strict_unseen_rows": sum(key not in train_keys for key in official["keys"]),
        "strata": strata,
        "selection_strata": selection_strata,
        "outcomes_read": False,
        "training_authorized": False,
    }


def _record_from_keys(keys: set[tuple], all_keys: list[tuple]) -> dict:
    return _role_record(keys, all_keys)


def _without_hash(payload: Mapping[str, object], field: str = "registry_sha256") -> dict:
    value = copy.deepcopy(dict(payload))
    value.pop(field, None)
    return value


def _source_manifest() -> dict[str, Path]:
    return {
        "mechanism_design": DEFAULT_MECHANISM_DESIGN,
        "protocol_design": DEFAULT_PROTOCOL_DESIGN,
        "protocol_source": DEFAULT_PROTOCOL_SOURCE,
        "execution_source": DEFAULT_EXECUTION_SOURCE,
        "model_source": DEFAULT_MODEL_SOURCE,
        "model_base_source": DEFAULT_MODEL_BASE_SOURCE,
        "marked_source": DEFAULT_MARKED_SOURCE,
        "geometry_source": DEFAULT_GEOMETRY_SOURCE,
        "anchor_source": DEFAULT_ANCHOR_SOURCE,
        "memory_source": DEFAULT_MEMORY_SOURCE,
        "radio_energy_source": DEFAULT_RADIO_ENERGY_SOURCE,
        "encoder_source": DEFAULT_ENCODER_SOURCE,
        "split_primitive_source": DEFAULT_SPLIT_PRIMITIVE_SOURCE,
        "hull_source": DEFAULT_HULL_SOURCE,
        "metric_source": DEFAULT_METRIC_SOURCE,
        "protocol_test": DEFAULT_PROTOCOL_TEST,
        "execution_test": DEFAULT_EXECUTION_TEST,
        "mechanism_test": DEFAULT_MECHANISM_TEST,
    }


def register_protocol(
    *,
    train_data: str | Path = DEFAULT_TRAIN,
    official_data: str | Path = DEFAULT_OFFICIAL,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    """Freeze the complete development-only V16 execution contract once."""

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    sources = _source_manifest()
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V16 registration source is missing: {missing}")
    train_path = Path(train_data).resolve()
    official_path = Path(official_data).resolve()
    train = load_uji_v16(train_path)
    official = load_uji_v16(official_path)
    train_keys = set(train["keys"])
    official_keys = set(official["keys"])
    core, folds, strata = build_crossfit_roles(train_keys)
    interior = train_keys - core
    operating, saturation, selection_strata = partition_selection_roles(interior)

    fold_records = []
    for payload in folds:
        fold_records.append(
            {
                "fold": payload["fold"],
                "member_fit": _record_from_keys(payload["member_fit_keys"], train["keys"]),
                "early_stop": _record_from_keys(payload["early_stop_keys"], train["keys"]),
                "fold_refit": _record_from_keys(payload["fold_refit_keys"], train["keys"]),
                "oof_query": _record_from_keys(payload["oof_query_keys"], train["keys"]),
                "nested_details": payload["nested_details"],
            }
        )
    allowed_outputs = {
        "crossfit_checkpoints": [
            [
                str(crossfit_checkpoint_path(fold, index).resolve())
                for index in range(1, len(MEMBER_SEEDS) + 1)
            ]
            for fold in range(FOLD_COUNT)
        ],
        "fold_radio_memories": [
            str(fold_radio_path(fold).resolve()) for fold in range(FOLD_COUNT)
        ],
        "fold_caches": [
            str(fold_cache_path(fold).resolve()) for fold in range(FOLD_COUNT)
        ],
        "fold_grids": [
            str(fold_grid_path(fold).resolve()) for fold in range(FOLD_COUNT)
        ],
        "oof_summary": str(DEFAULT_SUMMARY.resolve()),
        "selection_freeze": str(DEFAULT_FREEZE.resolve()),
    }
    flat_outputs = [
        *[value for fold in allowed_outputs["crossfit_checkpoints"] for value in fold],
        *allowed_outputs["fold_radio_memories"],
        *allowed_outputs["fold_caches"],
        *allowed_outputs["fold_grids"],
        allowed_outputs["oof_summary"],
        allowed_outputs["selection_freeze"],
    ]
    existing = [value for value in flat_outputs if Path(value).exists()]
    if existing:
        raise FileExistsError(f"V16 registered output already exists: {existing[:3]}")

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_status": "new single-model V16 cross-fitted UJI development experiment",
        "development_only": True,
        "training_authorized_after_registry_freeze": True,
        "official_validation_prediction_authorized": False,
        "official_validation_selection_authority": False,
        "cross_dataset_confirmation_claim_authorized": False,
        "historical_results_may_select_v16": False,
        "v15_checkpoints_eligible": False,
        "train_data": str(train_path),
        "train_data_sha256": _sha256_file(train_path),
        "official_data_audit_only": str(official_path),
        "official_data_sha256": _sha256_file(official_path),
        "coordinate_precision_contract": "raw CSV float64 keys before centering; only centered differences become float32",
        "split_contract": {
            "unit": "exact physical coordinate (building,floor,longitude,latitude)",
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
            "early_stop_seeds": list(EARLY_STOP_SEEDS),
            "early_stop_fraction": EARLY_STOP_FRACTION,
            "support_core": _record_from_keys(core, train["keys"]),
            "folds": fold_records,
            "strata": strata,
        },
        "selection_roles": {
            "partition_seed": SELECTION_PARTITION_SEED,
            "operating_tuning": _record_from_keys(operating, train["keys"]),
            "member_saturation": _record_from_keys(saturation, train["keys"]),
            "strata": selection_strata,
        },
        "official_overlap_audit": {
            "official_rows": len(official["keys"]),
            "official_coordinates": len(official_keys),
            "overlap_coordinates": len(train_keys & official_keys),
            "overlap_rows": sum(key in train_keys for key in official["keys"]),
            "strict_unseen_rows": sum(key not in train_keys for key in official["keys"]),
            "predictions_computed": 0,
        },
        "model_contract": {
            "main_model": "V16 orbit-factorized marked barycentric chart",
            "model_candidates": 1,
            "control_panels": 0,
            "fallbacks": 0,
            "learned_coordinate_decoder_parameters": 0,
            "posterior_corrections": 1,
            "student_df": 4.0,
        },
        "selection_contract": {
            "E_grid": list(E_GRID),
            "eligible_E_grid": list(ELIGIBLE_E_GRID),
            "K_grid": list(K_GRID),
            "K_upper_extension_if_boundary_improves": list(K_UPPER_EXTENSION),
            "lambda_grid": list(LAMBDA_GRID),
            "lambda_lower_extension_if_boundary_improves": list(LAMBDA_LOWER_EXTENSION),
            "lambda_upper_extension_if_boundary_improves": list(LAMBDA_UPPER_EXTENSION),
            "operating_tuning_rule": "within each E, one-coordinate-group-SE energy platform and 0.25 m MAE guardrail; minimize (K,energy,MAE,lambda)",
            "saturation_tolerance_fraction_of_E1_tuning_energy": SATURATION_TOLERANCE_FRACTION,
            "saturation_minimum_later_E": SATURATION_MINIMUM_LATER_E,
            "saturation_rule": "global max-centered paired coordinate bootstrap over every E<E'; select first E with at least two later values and all upper bounds <= tolerance",
            "budget_censored_rule": "if no E saturates through 12, select minimum member-saturation energy and make no saturation claim",
            "retained_mass_is_diagnostic_only": True,
            "mae_margin_m": MAE_NONINFERIORITY_MARGIN_M,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": SATURATION_INFERENCE_SEED,
        },
        "optimization_contract": {
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.10,
            "ap_dropout": 0.10,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip": 5.0,
            "maximum_epochs": 60,
            "minimum_epochs": 12,
            "patience": 8,
            "minimum_improvement": 1e-3,
        },
        "member_seeds": list(MEMBER_SEEDS),
        "allowed_outputs": allowed_outputs,
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in sources.items()
        },
        "outcome_access_at_registration": {
            "v16_oof_predictions": 0,
            "v16_official_predictions": 0,
            "historical_official_outcomes_known": True,
        },
    }
    registry["registry_sha256"] = _canonical_json_sha256(registry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
    )
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("registry_sha256") != _canonical_json_sha256(_without_hash(payload)):
        raise ValueError("V16 registry self-hash failed")
    for name, record in payload["sources"].items():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V16 registered source changed: {name}")
    if _sha256_file(payload["train_data"]) != payload["train_data_sha256"]:
        raise ValueError("V16 training data changed")
    if _sha256_file(payload["official_data_audit_only"]) != payload["official_data_sha256"]:
        raise ValueError("V16 official audit file changed")
    if payload.get("development_only") is not True:
        raise ValueError("V16 development registry changed phase")
    if payload.get("official_validation_prediction_authorized") is not False:
        raise ValueError("V16 development registry gained official access")
    return payload


def decode_keys(record: Mapping[str, object]) -> set[tuple[int, int, float, float]]:
    return {
        (int(value[0]), int(value[1]), float(value[2]), float(value[3]))
        for value in record["coordinate_keys"]
    }


def fold_role_keys(registry: Mapping[str, object], fold: int, role: str) -> set[tuple]:
    index = int(fold)
    if not 0 <= index < FOLD_COUNT:
        raise ValueError("V16 fold is outside the registered range")
    if role not in {"member_fit", "early_stop", "fold_refit", "oof_query"}:
        raise ValueError("unknown V16 fold role")
    record = registry["split_contract"]["folds"][index]
    if int(record["fold"]) != index:
        raise ValueError("V16 fold order changed")
    return decode_keys(record[role])


def selection_role_keys(registry: Mapping[str, object], role: str) -> set[tuple]:
    if role not in {"operating_tuning", "member_saturation"}:
        raise ValueError("unknown V16 selection role")
    return decode_keys(registry["selection_roles"][role])
