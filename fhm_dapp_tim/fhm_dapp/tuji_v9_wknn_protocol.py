"""Freeze and audit the TUJI1 V9 post-lock deterministic WKNN comparison.

This module deliberately has no author-test prediction implementation.  Its
only job is to make the comparison boundary auditable before an execution
module is written.  In particular, it binds the already-locked V9 external
result without granting a baseline any authority over V9 selection or claims.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.tuji_v9_protocol import (
    DEFAULT_REGISTRY,
    load_registry,
    load_tuji_split,
    registered_role_keys,
    registered_rows,
)
from fhm_dapp_tim.fhm_dapp.tuji_v9_wknn_common import (
    K_GRID,
    METHODS,
    VISIBILITY_THRESHOLD,
    coordinate_prototypes,
    features,
    predict_k_grid,
)


PROTOCOL_VERSION = "tuji1_v9_wknn_post_lock_v1_20260813"
BOOTSTRAP_REPETITIONS = 5000
PAIRED_BOOTSTRAP_SEED = 9911

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v9_tuji"
DEFAULT_DOCUMENT = EVIDENCE / "TUJI1_V9_POSTLOCK_WKNN_PROTOCOL_V1.md"
DEFAULT_PROTOCOL = EVIDENCE / "tuji1_postlock_wknn_protocol_v1.json"
DEFAULT_AUDIT = EVIDENCE / "tuji1_postlock_wknn_protocol_audit_v1.json"
DEFAULT_V9_RESULT = EVIDENCE / "author_test_result_tuji_v9_v5_scoring_signature.json"
DEFAULT_COMMON = ROOT / "fhm_dapp" / "tuji_v9_wknn_common.py"
DEFAULT_EXECUTION_MODULE = ROOT / "fhm_dapp" / "tuji_v9_wknn_comparator_v1.py"
DEFAULT_AMENDMENT = EVIDENCE / "tuji1_wknn_postlock_execution_amendment_v1.json"
DEFAULT_DEVELOPMENT = EVIDENCE / "tuji1_wknn_postlock_development_selection_v1.json"
DEFAULT_OUTER = EVIDENCE / "tuji1_wknn_postlock_outer_comparator_v1.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _without_hash(payload: Mapping[str, object], field: str) -> dict:
    result = copy.deepcopy(dict(payload))
    result.pop(field, None)
    return result


def _array_sha256(values: np.ndarray) -> str:
    value = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def select_k_by_coordinate_macro(
    scores: Mapping[int, float], *, expected: Sequence[int] = K_GRID
) -> int:
    """Choose the smallest `k` attaining the fixed coordinate-macro minimum."""

    expected_grid = tuple(map(int, expected))
    received = tuple(sorted(map(int, scores)))
    _require(received == tuple(sorted(expected_grid)), "k grid differs from protocol")
    values = {int(k): float(scores[int(k)]) for k in expected_grid}
    _require(all(np.isfinite(value) for value in values.values()), "non-finite k score")
    minimum = min(values.values())
    return min(k for k in expected_grid if values[k] == minimum)


def _role_record(registry: Mapping[str, object], role: str) -> dict[str, object]:
    record = registry["roles"][role]
    _require(isinstance(record, Mapping), f"invalid registry role: {role}")
    return {
        "coordinate_count": int(record["coordinate_count"]),
        "row_count": int(record["row_count"]),
        "coordinate_key_sha256": str(record["coordinate_key_sha256"]),
        "row_index_sha256": str(record["row_index_sha256"]),
    }


def _v9_result(path: str | Path, registry: Mapping[str, object]) -> dict:
    result_path = Path(path).resolve()
    _require(result_path.is_file(), f"frozen TUJI V9 result is absent: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require(
        result.get("result_sha256")
        == _canonical_json_sha256(_without_hash(result, "result_sha256")),
        "frozen TUJI V9 result self-hash failed",
    )
    _require(
        result.get("registry_sha256") == registry["registry_sha256"],
        "frozen TUJI V9 result registry mismatch",
    )
    _require(
        result.get("author_test_access_number") == 1,
        "unexpected V9 author-test access number",
    )
    _require(
        result.get("comparison_status") == "external_replication_after_eligibility_audit",
        "frozen result has an unexpected comparison status",
    )
    cache_path = Path(str(result["author_test_cache"])).resolve()
    _require(cache_path.is_file(), "frozen V9 test cache is absent")
    _require(
        _sha256_file(cache_path) == result["author_test_cache_sha256"],
        "frozen V9 test cache changed",
    )
    metrics = result.get("metrics")
    _require(isinstance(metrics, Mapping), "frozen result metrics missing")
    _require(
        int(result.get("coordinate_group_count", -1)) == 431,
        "unexpected TUJI author-test coordinate-group count",
    )
    for field in ("coordinate_group_energy_score_2d_m", "coordinate_group_mae_2d_m"):
        values = np.asarray(result.get(field), dtype=np.float64)
        _require(values.shape == (431,) and np.isfinite(values).all(), f"invalid frozen V9 {field}")
    return result


def _planned_paths() -> dict[str, Path]:
    return {
        "execution_module": DEFAULT_EXECUTION_MODULE.resolve(),
        "execution_amendment": DEFAULT_AMENDMENT.resolve(),
        "development_selection": DEFAULT_DEVELOPMENT.resolve(),
        "outer_comparator": DEFAULT_OUTER.resolve(),
    }


def _require_future_artifacts_absent() -> None:
    present = [str(path) for path in _planned_paths().values() if path.exists()]
    _require(not present, "future post-lock artifacts already exist: " + ", ".join(present))


def freeze_protocol(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    v9_result_path: str | Path = DEFAULT_V9_RESULT,
    document_path: str | Path = DEFAULT_DOCUMENT,
    output: str | Path = DEFAULT_PROTOCOL,
) -> dict:
    """Write the machine-readable protocol before any comparator code exists."""

    destination = Path(output).resolve()
    _require(not destination.exists(), f"protocol already exists: {destination}")
    _require(not Path(DEFAULT_AUDIT).exists(), "protocol audit already exists")
    _require_future_artifacts_absent()
    registry = load_registry(registry_path)
    result = _v9_result(v9_result_path, registry)
    document = Path(document_path).resolve()
    common = Path(DEFAULT_COMMON).resolve()
    _require(document.is_file(), f"protocol document is absent: {document}")
    _require(common.is_file(), f"baseline primitive is absent: {common}")
    source_files = registry["source_files"]
    _require(isinstance(source_files, Mapping), "TUJI registry source manifest missing")
    status = {
        "created_after_v9_author_test": True,
        "selection_authority_for_v9": False,
        "may_modify_v9": False,
        "scientific_change": False,
        "comparison_status": "post_lock_comparator",
    }
    payload: dict[str, object] = {
        "artifact_status": "tuji1_v9_postlock_wknn_protocol_frozen",
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "scope": {
            "methods": "three deterministic conventional WKNN comparators only",
            "does_not_test": [
                "V9 internal component attribution",
                "V9 member-count saturation",
                "a new independent blind external test",
            ],
        },
        "protocol_document": {
            "path": str(document),
            "sha256": _sha256_file(document),
        },
        "v9_registry": {
            "path": str(Path(registry_path).resolve()),
            "file_sha256": _sha256_file(registry_path),
            "registry_sha256": registry["registry_sha256"],
        },
        "post_lock_timing_boundary": {
            "v9_author_test_result_path": str(Path(v9_result_path).resolve()),
            "v9_author_test_result_file_sha256": _sha256_file(v9_result_path),
            "v9_author_test_result_self_hash": result["result_sha256"],
            "v9_author_test_access_number": 1,
            "baseline_author_test_accesses_before_protocol": 0,
            "baseline_author_test_accesses_planned": 1,
        },
        "source_contract": {
            "baseline_primitive": str(common),
            "baseline_primitive_sha256": _sha256_file(common),
            "execution_module_status": "must_not_exist_before_protocol_audit",
            "execution_module_planned_path": str(DEFAULT_EXECUTION_MODULE.resolve()),
        },
        "data_contract": {
            "rss_normalization": "TUJI V9: 100 -> -110 dBm; clip [-110,-30]; affine map to [-1,1]",
            "missing_sentinel": -1.0,
            "visibility_rule": "normalized RSS > -0.999",
            "n_aps": 310,
            "coordinate_schema": "two-dimensional physical metres; fixed building=floor=0",
            "roles": {
                "selection_refit": _role_record(registry, "selection_refit"),
                "development": _role_record(registry, "development"),
                "final_refit": _role_record(registry, "final_refit"),
            },
            "author_test_sources": {
                name: {
                    "path": str(record["path"]),
                    "sha256": str(record["sha256"]),
                }
                for name, record in source_files.items()
                if name in {"rss_testing", "coordinates_testing", "device_labels"}
            },
        },
        "baseline_contract": {
            "methods": list(METHODS),
            "k_grid": list(K_GRID),
            "raw_wknn": "310 normalized RSS values including -1 missing sentinel",
            "diff_wknn": "visibility mask plus visible-mean-centred normalized RSS",
            "rank_wknn": "visibility mask plus stable normalized visible ranks",
            "visibility_threshold": float(VISIBILITY_THRESHOLD),
            "reference_prototype": {
                "within_device": "per-AP median over reference scans",
                "between_device": "unweighted arithmetic mean of available device medians",
                "target": "registered 2-D physical coordinate",
            },
            "predictor": {
                "distance": "global Euclidean feature distance",
                "weights": "1 / max(distance, 1e-8)",
                "building_or_floor_conditioning": False,
                "query_device_or_user_conditioning": False,
            },
        },
        "selection_contract": {
            "reference_role": "selection_refit",
            "query_role": "development",
            "objective": "coordinate-macro 2-D MAE",
            "one_k_per_named_method": True,
            "tie_rule": "smallest k at exact minimum",
            "outer_test_used_for_selection": False,
        },
        "comparison_contract": {
            "reference_role": "final_refit",
            "query_role": "official_author_test",
            "test_prediction_passes": 1,
            "unit": "exact physical-coordinate group",
            "group_order": "lexicographically sorted (building,floor,x,y)",
            "metrics": ["coordinate-macro 2-D MAE", "coordinate-macro energy score"],
            "deterministic_energy": "degenerate predictive distribution; energy score equals Euclidean point error",
            "difference_order": [f"v9_minus_{method}" for method in METHODS],
            "negative_difference_favors": "V9",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
            "confidence": 0.95,
            "family_control": "max-centred simultaneous upper bounds across all three named methods",
        },
        "planned_artifacts": {name: str(path) for name, path in _planned_paths().items()},
    }
    payload["protocol_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict:
    """Load and validate the immutable post-lock comparison protocol."""

    protocol_path = Path(path).resolve()
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    _require(payload.get("artifact_status") == "tuji1_v9_postlock_wknn_protocol_frozen", "protocol status is invalid")
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "wrong TUJI WKNN protocol")
    _require(
        payload.get("protocol_sha256")
        == _canonical_json_sha256(_without_hash(payload, "protocol_sha256")),
        "protocol self-hash failed",
    )
    status = payload.get("status")
    _require(isinstance(status, Mapping), "protocol status missing")
    expected_status = {
        "created_after_v9_author_test": True,
        "selection_authority_for_v9": False,
        "may_modify_v9": False,
        "scientific_change": False,
        "comparison_status": "post_lock_comparator",
    }
    for name, expected in expected_status.items():
        _require(status.get(name) == expected, f"post-lock status changed: {name}")
    document = payload["protocol_document"]
    _require(
        _sha256_file(document["path"]) == document["sha256"],
        "protocol document changed",
    )
    registry_record = payload["v9_registry"]
    _require(
        _sha256_file(registry_record["path"]) == registry_record["file_sha256"],
        "V9 registry file changed",
    )
    registry = load_registry(registry_record["path"])
    _require(registry["registry_sha256"] == registry_record["registry_sha256"], "V9 registry self-hash changed")
    result_record = payload["post_lock_timing_boundary"]
    _require(
        _sha256_file(result_record["v9_author_test_result_path"])
        == result_record["v9_author_test_result_file_sha256"],
        "frozen V9 result file changed",
    )
    result = _v9_result(result_record["v9_author_test_result_path"], registry)
    _require(
        result["result_sha256"] == result_record["v9_author_test_result_self_hash"],
        "frozen V9 result self-hash changed",
    )
    source = payload["source_contract"]
    _require(
        _sha256_file(source["baseline_primitive"]) == source["baseline_primitive_sha256"],
        "baseline primitive changed",
    )
    _require(tuple(payload["baseline_contract"]["methods"]) == METHODS, "method family changed")
    _require(tuple(payload["baseline_contract"]["k_grid"]) == K_GRID, "k grid changed")
    _require(
        float(payload["baseline_contract"]["visibility_threshold"]) == VISIBILITY_THRESHOLD,
        "visibility threshold changed",
    )
    return payload


def _audit_preprocessing_contract() -> dict:
    fixture = np.asarray(
        [
            [-1.0, -0.999, -0.5, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -0.75, -0.25, -0.25],
        ],
        dtype=np.float64,
    )
    expected_diff = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -0.25, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, -1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        ]
    )
    expected_rank = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0],
        ]
    )
    _require(np.array_equal(features(fixture, "raw_wknn"), fixture), "raw representation changed")
    _require(np.allclose(features(fixture, "diff_wknn"), expected_diff), "diff representation changed")
    _require(np.array_equal(features(fixture, "rank_wknn"), expected_rank), "rank representation changed")
    proto, target, keys = coordinate_prototypes(
        np.asarray([[-0.1, -0.3], [-0.5, -0.7], [-0.2, -0.4], [-0.6, -0.8]]),
        np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
        np.asarray([0, 0, 1, 1]),
        [(0, 0, 1.0, 2.0)] * 4,
        np.arange(4),
    )
    _require(keys == [(0, 0, 1.0, 2.0)], "prototype fixture key changed")
    _require(np.allclose(proto, [[-0.35, -0.55]]), "device-balanced prototype changed")
    _require(np.allclose(target, [[1.0, 2.0]]), "prototype target changed")
    signature = inspect.signature(predict_k_grid)
    _require(tuple(signature.parameters) == ("train_x", "train_y", "test_x", "k_grid", "chunk_size"), "predictor signature changed")
    predictor_source = inspect.getsource(predict_k_grid)
    _require("building" not in predictor_source and "floor" not in predictor_source, "predictor conditions physical strata")
    _require(select_k_by_coordinate_macro({1: 2.0, 3: 1.0, 5: 1.0, 8: 4.0, 12: 5.0, 20: 6.0, 32: 7.0}) == 3, "tie rule changed")
    return {
        "raw_fixture_sha256": _array_sha256(features(fixture, "raw_wknn")),
        "diff_fixture_sha256": _array_sha256(features(fixture, "diff_wknn")),
        "rank_fixture_sha256": _array_sha256(features(fixture, "rank_wknn")),
        "device_balanced_fixture_sha256": _array_sha256(proto),
        "visibility_threshold": VISIBILITY_THRESHOLD,
        "predictor_signature": list(signature.parameters),
        "predictor_has_building_or_floor_conditioning": False,
        "tie_fixture_selected_k": 3,
    }


def audit_protocol(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    output: str | Path = DEFAULT_AUDIT,
) -> dict:
    """Audit all pre-execution guarantees without parsing TUJI test labels."""

    destination = Path(output).resolve()
    _require(not destination.exists(), f"audit already exists: {destination}")
    protocol = load_protocol(protocol_path)
    _require_future_artifacts_absent()
    registry = load_registry(protocol["v9_registry"]["path"])
    training_path = Path(str(registry["source_files"]["rss_training"]["path"])).resolve()
    training = load_tuji_split(training_path.parent, split="training")
    role_names = ("member_fit", "early_stop", "selection_refit", "development", "final_refit")
    keys = {name: registered_role_keys(registry, name) for name in role_names}
    rows = {name: registered_rows(registry, name, training) for name in role_names}
    _require(keys["member_fit"] | keys["early_stop"] == keys["selection_refit"], "inner roles do not recover selection refit")
    _require(keys["selection_refit"].isdisjoint(keys["development"]), "selection/development coordinates overlap")
    _require(keys["selection_refit"] | keys["development"] == keys["final_refit"], "selection/development do not recover final refit")
    _require(set(rows["member_fit"]) | set(rows["early_stop"]) == set(rows["selection_refit"]), "inner rows do not recover selection refit")
    _require(set(rows["selection_refit"]).isdisjoint(rows["development"]), "selection/development rows overlap")
    _require(set(rows["selection_refit"]) | set(rows["development"]) == set(rows["final_refit"]), "selection/development rows do not recover final refit")
    prototype, target, prototype_keys = coordinate_prototypes(
        training["rss"], training["raw_xy"], training["phone"], training["keys"], rows["selection_refit"]
    )
    _require(prototype.shape == (1081, 310), "selection reference prototype shape changed")
    _require(target.shape == (1081, 2), "selection reference target shape changed")
    _require(set(prototype_keys) == keys["selection_refit"], "selection prototype keys changed")
    preprocessing = _audit_preprocessing_contract()
    role_summary = {
        name: {
            "coordinates": int(len(keys[name])),
            "rows": int(len(rows[name])),
            "coordinate_key_sha256": registry["roles"][name]["coordinate_key_sha256"],
            "row_index_sha256": registry["roles"][name]["row_index_sha256"],
        }
        for name in ("selection_refit", "development", "final_refit")
    }
    payload: dict[str, object] = {
        "artifact_status": "tuji1_v9_postlock_wknn_protocol_audit_passed",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_file_sha256": _sha256_file(protocol_path),
        "protocol_self_hash": protocol["protocol_sha256"],
        "status": dict(protocol["status"]),
        "pre_execution_state": {
            "development_selection_performed": False,
            "baseline_author_test_comparator_performed": False,
            "author_test_labels_loaded_for_wknn_audit": False,
            "author_test_features_transformed_for_wknn_audit": False,
            "future_execution_artifacts_absent": True,
        },
        "role_audit": role_summary,
        "role_relationships": {
            "member_fit_union_early_stop_equals_selection_refit": True,
            "selection_refit_disjoint_development": True,
            "selection_refit_union_development_equals_final_refit": True,
        },
        "selection_reference": {
            "prototype_shape": list(prototype.shape),
            "target_shape": list(target.shape),
            "prototype_feature_sha256": _array_sha256(prototype),
            "target_sha256": _array_sha256(target),
        },
        "preprocessing_contract": preprocessing,
        "frozen_v9_boundary": dict(protocol["post_lock_timing_boundary"]),
    }
    payload["audit_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    freeze = children.add_parser("freeze-protocol")
    freeze.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    freeze.add_argument("--v9-result", default=str(DEFAULT_V9_RESULT))
    freeze.add_argument("--document", default=str(DEFAULT_DOCUMENT))
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
            v9_result_path=args.v9_result,
            document_path=args.document,
            output=args.output,
        )
        print(json.dumps({"protocol": str(Path(args.output).resolve()), "protocol_sha256": payload["protocol_sha256"]}))
    else:
        payload = audit_protocol(protocol_path=args.protocol, output=args.output)
        print(json.dumps({"audit": str(Path(args.output).resolve()), "audit_sha256": payload["audit_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
