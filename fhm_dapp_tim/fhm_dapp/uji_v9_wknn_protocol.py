"""Freeze and audit the V9 post-lock WKNN comparator protocol.

This module deliberately has no command that tunes ``k`` or predicts a V9
query role.  It freezes provenance and verifies data/preprocessing contracts
before a separately reviewed execution implementation is allowed to exist.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fhm_dapp_tim.fhm_dapp import baselines_v44, v9_wknn_common
from fhm_dapp_tim.fhm_dapp.v9_wknn_common import (
    coordinate_prototypes,
    features,
    predict_k_grid,
)
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    simultaneous_paired_upper_bounds,
)
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _load_uji,
    _sha256_file,
    _sha256_indices,
    _sha256_keys,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.uji_v9_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    PROTOCOL_VERSION as V9_PROTOCOL_VERSION,
    _registered_role_keys,
    _registered_rows,
    load_registry,
)


PROTOCOL_VERSION = "uji_v9_postlock_wknn_comparator_v2_20260811"
METHODS = ("raw_wknn", "diff_wknn", "rank_wknn")
K_GRID = (1, 3, 5, 8, 12, 20)
VISIBILITY_THRESHOLD = -0.75
NUMERICAL_TIE_TOLERANCE_M = 1e-12
BOOTSTRAP_REPETITIONS = 5000
ENERGY_BOOTSTRAP_SEED = 9911
MAE_BOOTSTRAP_SEED = 9912
EXPECTED_ROLE_COUNTS = {
    "member_fit": (506, 10763),
    "early_stop": (89, 1902),
    "selection_refit": (595, 12665),
    "development": (151, 3139),
    "outer_refit": (746, 15804),
    "outer_test": (187, 4133),
}
EXPECTED_TRAINING_COORDINATES = 933
EXPECTED_OUTER_STRATA = 13

DEFAULT_DOCUMENT = DEFAULT_EVIDENCE / "POSTLOCK_WKNN_COMPARATOR_PROTOCOL_V2.md"
DEFAULT_PROTOCOL = DEFAULT_EVIDENCE / "postlock_wknn_protocol_v2.json"
DEFAULT_AUDIT = DEFAULT_EVIDENCE / "postlock_wknn_protocol_audit_v2.json"
DEFAULT_V9_POSTLOCK_AUDIT = (
    DEFAULT_EVIDENCE / "outer_test_postlock_audit_v9_uji.json"
)
DEFAULT_V1_PROTOCOL = DEFAULT_EVIDENCE / "postlock_wknn_protocol_v1.json"
PLANNED_DEVELOPMENT = (
    DEFAULT_EVIDENCE / "postlock_wknn_development_selection_v2.json"
)
PLANNED_OUTER = DEFAULT_EVIDENCE / "postlock_wknn_outer_comparator_v2.json"
PLANNED_EXECUTION_MODULE = (
    Path(__file__).resolve().parent / "uji_v9_wknn_comparator_v2.py"
)

REQUIRED_STATUS = {
    "created_after_v9_outer_test": True,
    "selection_authority_for_v9": False,
    "may_modify_v9": False,
    "comparison_status": "post_lock_comparator",
}


def _without_hash(payload: dict, field: str) -> dict:
    value = copy.deepcopy(payload)
    value.pop(field, None)
    return value


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _validate_status(payload: Mapping[str, object]) -> None:
    for field, expected in REQUIRED_STATUS.items():
        _require(payload.get(field) == expected, f"invalid scientific status: {field}")
    _require(payload.get("not_a_preregistration") is True, "missing post-lock disclosure")
    _require(
        payload.get("internal_v9_variant_expansion_allowed") is False,
        "protocol admits an unregistered V9 variant",
    )


def select_k_by_coordinate_macro(
    scores: Mapping[int, float],
    *,
    k_grid: Sequence[int] = K_GRID,
    tie_tolerance_m: float = NUMERICAL_TIE_TOLERANCE_M,
) -> int:
    """Select the exact development-risk minimum with a fixed numeric tie rule."""

    expected = tuple(map(int, k_grid))
    _require(tuple(sorted(map(int, scores))) == tuple(sorted(expected)), "k grid differs")
    values = {int(k): float(scores[int(k)]) for k in expected}
    _require(all(np.isfinite(list(values.values()))), "k scores must be finite")
    minimum = min(values.values())
    tied = [
        k
        for k in expected
        if values[k] <= minimum + float(tie_tolerance_m)
    ]
    return int(min(tied))


def _validate_existing_v9_result(path: Path, registry: dict) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    _require(result.get("protocol_version") == V9_PROTOCOL_VERSION, "not a V9 result")
    _require(
        result.get("registry_sha256") == registry["registry_sha256"],
        "V9 result registry mismatch",
    )
    _require(result.get("outer_test_access_number") == 1, "unexpected V9 access count")
    stored = result.get("result_sha256")
    _require(
        stored == _canonical_json_sha256(_without_hash(result, "result_sha256")),
        "V9 result self-hash failed",
    )
    return result


def _validate_existing_postlock_audit(path: Path, registry: dict) -> dict:
    audit = json.loads(path.read_text(encoding="utf-8"))
    _require(
        audit.get("protocol_version") == V9_PROTOCOL_VERSION,
        "not a V9 post-lock audit",
    )
    _require(
        audit.get("registry_sha256") == registry["registry_sha256"],
        "V9 post-lock audit registry mismatch",
    )
    stored = audit.get("audit_sha256")
    _require(
        stored == _canonical_json_sha256(_without_hash(audit, "audit_sha256")),
        "V9 post-lock audit self-hash failed",
    )
    return audit


def freeze_protocol(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    document_path: str | Path = DEFAULT_DOCUMENT,
    v9_result_path: str | Path = DEFAULT_RESULT,
    v9_postlock_audit_path: str | Path = DEFAULT_V9_POSTLOCK_AUDIT,
    output: str | Path = DEFAULT_PROTOCOL,
) -> dict:
    """Write the immutable machine-readable post-lock protocol."""

    registry_file = Path(registry_path).resolve()
    document = Path(document_path).resolve()
    result_file = Path(v9_result_path).resolve()
    postlock_file = Path(v9_postlock_audit_path).resolve()
    for required in (registry_file, document, result_file, postlock_file):
        if not required.exists():
            raise FileNotFoundError(required)
    registry = load_registry(registry_file)
    result = _validate_existing_v9_result(result_file, registry)
    postlock = _validate_existing_postlock_audit(postlock_file, registry)
    _require(tuple(v9_wknn_common.METHODS) == METHODS, "V2 methods changed")
    _require(tuple(v9_wknn_common.K_GRID) == K_GRID, "V2 k grid changed")
    _require(
        float(v9_wknn_common.VISIBILITY_THRESHOLD) == VISIBILITY_THRESHOLD,
        "V2 visibility threshold changed",
    )
    _require(tuple(baselines_v44.METHODS) == METHODS, "V4.4 method lineage changed")
    _require(tuple(baselines_v44.K_GRID) == K_GRID, "V4.4 k-grid lineage changed")
    _require(
        float(baselines_v44.MEMBER_VISIBLE_THRESHOLD) == VISIBILITY_THRESHOLD,
        "V4.4 visibility-threshold lineage changed",
    )
    role_contract = {}
    for role, (coordinates, rows) in EXPECTED_ROLE_COUNTS.items():
        record = registry["roles"][role]
        _require(record["coordinate_count"] == coordinates, f"{role} coordinates changed")
        _require(record["row_count"] == rows, f"{role} rows changed")
        role_contract[role] = {
            "coordinate_count": coordinates,
            "row_count": rows,
            "coordinate_key_sha256": record["coordinate_key_sha256"],
            "row_index_sha256": record["row_index_sha256"],
        }
    v9_groups = postlock["coordinate_group_uncertainty"]["mae_2d_m"][
        "coordinate_groups"
    ]
    _require(v9_groups == 187, "V9 post-lock audit does not contain 187 groups")

    baseline_source = Path(v9_wknn_common.__file__).resolve()
    historical_source = Path(baselines_v44.__file__).resolve()
    statistics_source = Path(inspect.getsourcefile(simultaneous_paired_upper_bounds)).resolve()
    payload = {
        "artifact_status": "postlock_wknn_comparator_protocol_frozen",
        "protocol_version": PROTOCOL_VERSION,
        **REQUIRED_STATUS,
        "not_a_preregistration": True,
        "not_independent_external_confirmation": True,
        "internal_v9_variant_expansion_allowed": False,
        "all_named_methods_must_be_reported": True,
        "scientific_scope": (
            "three traditional deterministic comparators only; no V9 selection or "
            "attribution authority"
        ),
        "protocol_document": {
            "path": str(document),
            "sha256": _sha256_file(document),
        },
        "v9_registry": {
            "path": str(registry_file),
            "file_sha256": _sha256_file(registry_file),
            "registry_sha256": registry["registry_sha256"],
            "protocol_version": registry["protocol_version"],
        },
        "post_lock_timing_boundary": {
            "v9_outer_result_path": str(result_file),
            "v9_outer_result_file_sha256": _sha256_file(result_file),
            "v9_outer_result_self_hash": result["result_sha256"],
            "v9_outer_test_access_number": 1,
            "v9_postlock_audit_path": str(postlock_file),
            "v9_postlock_audit_file_sha256": _sha256_file(postlock_file),
            "v9_postlock_audit_self_hash": postlock["audit_sha256"],
            "new_wknn_outcomes_opened_when_written": False,
        },
        "data_contract": {
            "dataset_path": registry["train_data"],
            "dataset_sha256": registry["train_data_sha256"],
            "official_validation_in_scope": False,
            "group_key": ["building", "floor", "longitude", "latitude"],
            "roles": role_contract,
            "development_reference_role": "selection_refit",
            "development_query_role": "development",
            "final_reference_role": "outer_refit",
            "final_query_role": "outer_test",
        },
        "source_contract": {
            "baseline_source": str(baseline_source),
            "baseline_source_sha256": _sha256_file(baseline_source),
            "reused_functions": [
                "coordinate_prototypes",
                "features",
                "predict_k_grid",
                "group_macro_errors",
            ],
            "numeric_precision": "float64 for reference RSS and coordinate aggregation",
            "historical_v44_source": str(historical_source),
            "historical_v44_source_sha256": _sha256_file(historical_source),
            "historical_method_lineage": (
                "raw/diff/rank representation semantics and global WKNN only"
            ),
            "statistics_source": str(statistics_source),
            "statistics_source_sha256": _sha256_file(statistics_source),
            "statistics_function": "simultaneous_paired_upper_bounds",
        },
        "reference_fingerprint_contract": {
            "unit": "one prototype per exact registered physical coordinate",
            "within_device": "per-AP median over reference rows",
            "between_device": "unweighted arithmetic mean of device medians",
            "reference_phone_id_use": "prototype aggregation only",
            "query_phone_id_use": "forbidden",
            "prototype_target": "registered raw 2-D coordinate",
        },
        "representation_contract": {
            "methods": list(METHODS),
            "raw_wknn": "520 normalized RSS values including -1.0 missing sentinel",
            "diff_wknn": (
                "visibility mask concatenated with visible-mean-centered RSS"
            ),
            "rank_wknn": (
                "visibility mask concatenated with stable normalized visible ranks"
            ),
            "visibility_rule_for_diff_and_rank": "normalized_RSS > -0.75",
            "visibility_threshold": VISIBILITY_THRESHOLD,
            "representation_applied_after_prototype_aggregation": True,
        },
        "neighbor_contract": {
            "candidate_scope": "all reference prototypes globally",
            "query_building_or_floor_conditioning": False,
            "query_device_or_user_conditioning": False,
            "distance": "Euclidean in the named representation",
            "ordering": "stable full sort; sorted coordinate key breaks exact ties",
            "weight": "1 / max(distance, 1e-8)",
        },
        "selection_contract": {
            "selected_separately_for_every_method": True,
            "k_grid": list(K_GRID),
            "objective": "development coordinate-group macro 2-D Euclidean error",
            "rule": "minimum objective",
            "numeric_tie_tolerance_m": NUMERICAL_TIE_TOLERANCE_M,
            "tie_break": "smallest k",
            "uses_device_risk_guard": False,
            "retains_every_grid_result": True,
        },
        "evaluation_contract": {
            "primary_metric": "coordinate-group macro 2-D Euclidean error",
            "descriptive_metric": "row-micro 2-D Euclidean error",
            "degenerate_energy_score_equals_point_error": True,
            "outer_test_prediction_passes": 1,
            "feeds_back_into_v9": False,
        },
        "paired_inference_contract": {
            "unit": "exact outer_test physical-coordinate group",
            "coordinate_groups": 187,
            "difference_order": [
                "v9_minus_raw_wknn",
                "v9_minus_diff_wknn",
                "v9_minus_rank_wknn",
            ],
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "resampling": "uniform coordinate-group resampling with replacement",
            "confidence": 0.95,
            "family_control": "simultaneous one-sided upper bounds over 3 methods",
            "energy_seed": ENERGY_BOOTSTRAP_SEED,
            "point_error_seed": MAE_BOOTSTRAP_SEED,
            "negative_difference_favors": "V9",
        },
        "planned_artifacts": {
            "execution_module": str(PLANNED_EXECUTION_MODULE.resolve()),
            "execution_module_status": "must_not_exist_before_protocol_review",
            "development_selection": str(PLANNED_DEVELOPMENT.resolve()),
            "outer_comparator": str(PLANNED_OUTER.resolve()),
        },
        "predecessor": {
            "protocol_path": str(DEFAULT_V1_PROTOCOL.resolve()),
            "protocol_file_sha256": _sha256_file(DEFAULT_V1_PROTOCOL),
            "disposition": "pre_execution_audit_rejected_no_wknn_outcomes_opened",
            "rejection_reason": (
                "float32 aggregation of metre-scale reference coordinates caused "
                "a demonstrated 3.5 m target displacement"
            ),
        },
    }
    payload["protocol_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(output, payload)
    return payload


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict:
    protocol_path = Path(path).resolve()
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "wrong protocol")
    _require(
        payload.get("artifact_status")
        == "postlock_wknn_comparator_protocol_frozen",
        "protocol is not frozen",
    )
    _validate_status(payload)
    _require(
        payload.get("protocol_sha256")
        == _canonical_json_sha256(_without_hash(payload, "protocol_sha256")),
        "protocol self-hash failed",
    )
    for record_name, path_field, hash_field in (
        ("document", "path", "sha256"),
        ("registry", "path", "file_sha256"),
    ):
        record_key = "protocol_document" if record_name == "document" else "v9_registry"
        record = payload[record_key]
        _require(
            _sha256_file(record[path_field]) == record[hash_field],
            f"pinned {record_name} changed",
        )
    source = payload["source_contract"]
    _require(
        _sha256_file(source["baseline_source"])
        == source["baseline_source_sha256"],
        "pinned baseline source changed",
    )
    _require(
        _sha256_file(source["historical_v44_source"])
        == source["historical_v44_source_sha256"],
        "pinned V4.4 lineage source changed",
    )
    _require(
        _sha256_file(source["statistics_source"])
        == source["statistics_source_sha256"],
        "pinned statistics source changed",
    )
    boundary = payload["post_lock_timing_boundary"]
    _require(
        _sha256_file(boundary["v9_outer_result_path"])
        == boundary["v9_outer_result_file_sha256"],
        "pinned V9 result changed",
    )
    _require(
        _sha256_file(boundary["v9_postlock_audit_path"])
        == boundary["v9_postlock_audit_file_sha256"],
        "pinned V9 post-lock audit changed",
    )
    return payload


def _audit_preprocessing_contract() -> dict:
    _require(tuple(v9_wknn_common.METHODS) == METHODS, "method family differs")
    _require(tuple(v9_wknn_common.K_GRID) == K_GRID, "k grid differs")
    _require(
        float(v9_wknn_common.VISIBILITY_THRESHOLD) == VISIBILITY_THRESHOLD,
        "visibility threshold differs",
    )
    _require(
        float(baselines_v44.MEMBER_VISIBLE_THRESHOLD) == VISIBILITY_THRESHOLD,
        "historical visibility threshold differs",
    )
    fixture = np.asarray(
        [[-1.0, -0.70, -0.20], [-1.0, -0.60, -0.10]], dtype=np.float64
    )
    expected_diff = np.asarray(
        [
            [0.0, 1.0, 1.0, 0.0, -0.25, 0.25],
            [0.0, 1.0, 1.0, 0.0, -0.25, 0.25],
        ],
        dtype=np.float64,
    )
    expected_rank = np.asarray(
        [
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _require(np.array_equal(features(fixture, "raw_wknn"), fixture), "raw changed")
    _require(
        np.allclose(features(fixture, "diff_wknn"), expected_diff),
        "diff representation differs",
    )
    _require(
        np.array_equal(features(fixture, "rank_wknn"), expected_rank),
        "rank representation differs",
    )
    signature = inspect.signature(predict_k_grid)
    _require(
        tuple(signature.parameters)
        == ("train_x", "train_y", "test_x", "k_grid", "chunk_size"),
        "WKNN predictor signature changed",
    )
    predictor_source = inspect.getsource(predict_k_grid).lower()
    _require("building" not in predictor_source, "predictor contains building logic")
    _require("floor" not in predictor_source, "predictor contains floor logic")
    prototype_rss = np.asarray(
        [[-1.0, -0.20], [-0.60, -0.40], [-0.20, -0.80]], dtype=np.float64
    )
    prototype_xy = np.asarray([[1.0, 2.0]] * 3, dtype=np.float64)
    prototype_phones = np.asarray([1, 1, 2], dtype=np.int64)
    prototype_keys = [(0, 0, 1.0, 2.0)] * 3
    device_balanced, target, keys = coordinate_prototypes(
        prototype_rss,
        prototype_xy,
        prototype_phones,
        prototype_keys,
        np.arange(3, dtype=np.int64),
    )
    expected_device_balanced = np.asarray([[-0.50, -0.55]], dtype=np.float64)
    _require(
        np.allclose(device_balanced, expected_device_balanced),
        "coordinate prototype is not device balanced",
    )
    _require(np.array_equal(target, [[1.0, 2.0]]), "prototype target changed")
    _require(keys == [(0, 0, 1.0, 2.0)], "prototype key changed")
    selected = select_k_by_coordinate_macro(
        {1: 2.0, 3: 1.0, 5: 1.0, 8: 1.2, 12: 1.3, 20: 1.4}
    )
    _require(selected == 3, "development tie rule changed")
    return {
        "methods": list(METHODS),
        "k_grid": list(K_GRID),
        "visibility_threshold": VISIBILITY_THRESHOLD,
        "fixture_raw_sha256": _array_sha256(features(fixture, "raw_wknn")),
        "fixture_diff_sha256": _array_sha256(features(fixture, "diff_wknn")),
        "fixture_rank_sha256": _array_sha256(features(fixture, "rank_wknn")),
        "predictor_parameters": list(signature.parameters),
        "predictor_has_building_logic": False,
        "predictor_has_floor_logic": False,
        "device_balanced_prototype_fixture_sha256": _array_sha256(
            device_balanced
        ),
        "selection_fixture_selected_k": selected,
    }


def audit_protocol(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    output: str | Path = DEFAULT_AUDIT,
) -> dict:
    """Audit identity and preprocessing only; never compute WKNN predictions."""

    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    registry = load_registry(protocol["v9_registry"]["path"])
    data = _load_uji(registry["train_data"])
    keys_by_role = {}
    rows_by_role = {}
    role_audit = {}
    for role, (expected_coordinates, expected_rows) in EXPECTED_ROLE_COUNTS.items():
        keys = _registered_role_keys(registry, role)
        rows = _registered_rows(registry, role, data)
        _require(len(keys) == expected_coordinates, f"{role} coordinate count failed")
        _require(len(rows) == expected_rows, f"{role} row count failed")
        _require(
            _sha256_keys(keys) == registry["roles"][role]["coordinate_key_sha256"],
            f"{role} coordinate identity failed",
        )
        _require(
            _sha256_indices(rows) == registry["roles"][role]["row_index_sha256"],
            f"{role} row identity failed",
        )
        keys_by_role[role] = keys
        rows_by_role[role] = rows
        role_audit[role] = {
            "coordinate_count": len(keys),
            "row_count": len(rows),
            "coordinate_key_sha256": _sha256_keys(keys),
            "row_index_sha256": _sha256_indices(rows),
            "building_floor_strata": len({key[:2] for key in keys}),
        }

    _require(
        keys_by_role["member_fit"].isdisjoint(keys_by_role["early_stop"]),
        "inner coordinate roles overlap",
    )
    _require(
        keys_by_role["member_fit"] | keys_by_role["early_stop"]
        == keys_by_role["selection_refit"],
        "inner coordinate roles do not recover selection_refit",
    )
    _require(
        keys_by_role["selection_refit"].isdisjoint(keys_by_role["development"]),
        "selection/development coordinate roles overlap",
    )
    _require(
        keys_by_role["selection_refit"] | keys_by_role["development"]
        == keys_by_role["outer_refit"],
        "selection/development roles do not recover outer_refit",
    )
    _require(
        keys_by_role["outer_refit"].isdisjoint(keys_by_role["outer_test"]),
        "outer coordinate roles overlap",
    )
    all_keys = set(data["keys"])
    _require(len(all_keys) == EXPECTED_TRAINING_COORDINATES, "training groups changed")
    _require(
        keys_by_role["outer_refit"] | keys_by_role["outer_test"] == all_keys,
        "outer roles do not recover the training coordinates",
    )
    _require(
        all(
            role_audit[role]["building_floor_strata"] == EXPECTED_OUTER_STRATA
            for role in EXPECTED_ROLE_COUNTS
        ),
        "a role lost a building-floor stratum",
    )
    row_sets = {
        role: set(map(int, rows)) for role, rows in rows_by_role.items()
    }
    _require(
        row_sets["member_fit"].isdisjoint(row_sets["early_stop"]),
        "inner row roles overlap",
    )
    _require(
        row_sets["member_fit"] | row_sets["early_stop"]
        == row_sets["selection_refit"],
        "inner row roles do not recover selection_refit",
    )
    _require(
        row_sets["selection_refit"].isdisjoint(row_sets["development"]),
        "selection/development row roles overlap",
    )
    _require(
        row_sets["selection_refit"] | row_sets["development"]
        == row_sets["outer_refit"],
        "selection/development rows do not recover outer_refit",
    )
    _require(
        row_sets["outer_refit"].isdisjoint(row_sets["outer_test"]),
        "outer row roles overlap",
    )
    _require(
        row_sets["outer_refit"] | row_sets["outer_test"]
        == set(range(len(data["keys"]))),
        "outer row roles do not recover all training rows",
    )

    rss = data["rss"].numpy().astype(np.float64, copy=False)
    phone = data["phone"]
    prototype_audit = {}
    for role in ("selection_refit", "outer_refit"):
        prototypes, targets, prototype_keys = coordinate_prototypes(
            rss,
            data["raw_xy"],
            phone,
            data["keys"],
            rows_by_role[role],
        )
        expected_keys = sorted(keys_by_role[role])
        _require(prototype_keys == expected_keys, f"{role} prototype keys differ")
        expected_xy = np.asarray([key[2:] for key in expected_keys], dtype=np.float64)
        max_target_delta = float(np.abs(targets - expected_xy).max())
        _require(
            np.allclose(targets, expected_xy, rtol=0.0, atol=1e-3),
            f"{role} prototype targets differ (max abs delta={max_target_delta:.6f} m)",
        )
        _require(prototypes.shape == (len(expected_keys), 520), "prototype shape differs")
        _require(np.isfinite(prototypes).all(), "prototype contains nonfinite values")
        prototype_audit[role] = {
            "coordinate_prototypes": len(prototype_keys),
            "feature_width": int(prototypes.shape[1]),
            "prototype_rss_sha256": _array_sha256(prototypes),
            "prototype_target_sha256": _array_sha256(targets),
            "uses_reference_phone_only": True,
        }

    preprocessing = _audit_preprocessing_contract()
    planned = protocol["planned_artifacts"]
    planned_absence = {
        name: not Path(planned[name]).exists()
        for name in ("execution_module", "development_selection", "outer_comparator")
    }
    _require(all(planned_absence.values()), "a planned outcome artifact already exists")

    payload = {
        "artifact_status": "postlock_wknn_protocol_audit_passed",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_path": str(protocol_file),
        "protocol_file_sha256": _sha256_file(protocol_file),
        "protocol_self_hash": protocol["protocol_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "passed": True,
        "outcome_access": {
            "new_wknn_predictions_computed": False,
            "new_wknn_errors_computed": False,
            "development_selection_performed": False,
            "outer_test_comparator_performed": False,
            "outer_test_rows_loaded_for_identity_audit": True,
            "outer_test_features_transformed_for_wknn": False,
            "existing_v9_outcome_artifacts_used_for_provenance_only": True,
        },
        "scientific_status": REQUIRED_STATUS,
        "role_identity_audit": role_audit,
        "nested_role_relations": {
            "member_fit_disjoint_early_stop": True,
            "member_fit_union_early_stop_equals_selection_refit": True,
            "selection_refit_disjoint_development": True,
            "selection_refit_union_development_equals_outer_refit": True,
            "outer_refit_disjoint_outer_test": True,
            "outer_refit_union_outer_test_equals_all_933_coordinates": True,
            "all_corresponding_row_relations_passed": True,
        },
        "reference_prototype_audit": prototype_audit,
        "preprocessing_contract_audit": preprocessing,
        "planned_artifacts_absent": planned_absence,
        "inference_contract": {
            "simultaneous_family_size": 3,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "energy_seed": ENERGY_BOOTSTRAP_SEED,
            "point_error_seed": MAE_BOOTSTRAP_SEED,
            "statistics_function_importable": callable(
                simultaneous_paired_upper_bounds
            ),
        },
    }
    payload["audit_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(output, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    freeze = children.add_parser("freeze-protocol")
    freeze.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    freeze.add_argument("--document", default=str(DEFAULT_DOCUMENT))
    freeze.add_argument("--v9-result", default=str(DEFAULT_RESULT))
    freeze.add_argument(
        "--v9-postlock-audit", default=str(DEFAULT_V9_POSTLOCK_AUDIT)
    )
    freeze.add_argument("--output", default=str(DEFAULT_PROTOCOL))
    audit = children.add_parser("audit")
    audit.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    audit.add_argument("--output", default=str(DEFAULT_AUDIT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "freeze-protocol":
        payload = freeze_protocol(
            registry_path=args.registry,
            document_path=args.document,
            v9_result_path=args.v9_result,
            v9_postlock_audit_path=args.v9_postlock_audit,
            output=args.output,
        )
        summary = {
            "output": str(Path(args.output).resolve()),
            "protocol_sha256": payload["protocol_sha256"],
            "new_wknn_outcomes_opened": False,
        }
    elif args.command == "audit":
        payload = audit_protocol(protocol_path=args.protocol, output=args.output)
        summary = {
            "output": str(Path(args.output).resolve()),
            "audit_sha256": payload["audit_sha256"],
            "passed": payload["passed"],
            "new_wknn_outcomes_opened": False,
        }
    else:  # pragma: no cover - argparse makes this unreachable.
        raise ValueError(args.command)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
