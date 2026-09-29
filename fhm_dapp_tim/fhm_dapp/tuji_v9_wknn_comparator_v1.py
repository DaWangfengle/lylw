"""Execute the frozen TUJI1 V9 post-lock deterministic WKNN comparison.

The module has two stages.  ``development`` only reads registered author
training rows to choose one fixed ``k`` per named baseline.  ``outer`` then
performs one post-lock prediction pass on official test RSS, loads test labels
only after those predictions exist in memory, and compares the locked WKNN
losses with the already frozen V9 external result.

This is a comparator, not a V9 variant: every produced artifact carries the
post-lock status flags and no function may change V9 selection or outcomes.
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

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import simultaneous_paired_upper_bounds
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.tuji_v9_experiment import (
    load_tuji_test_labels_only,
    load_tuji_test_rss_only,
)
from fhm_dapp_tim.fhm_dapp.tuji_v9_protocol import (
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
    group_macro_errors,
    predict_k_grid,
)
from fhm_dapp_tim.fhm_dapp.tuji_v9_wknn_protocol import (
    BOOTSTRAP_REPETITIONS,
    DEFAULT_AMENDMENT,
    DEFAULT_AUDIT,
    DEFAULT_DEVELOPMENT,
    DEFAULT_EXECUTION_MODULE,
    DEFAULT_OUTER,
    DEFAULT_PROTOCOL,
    PAIRED_BOOTSTRAP_SEED,
    PROTOCOL_VERSION,
    load_protocol,
    select_k_by_coordinate_macro,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _without_hash(payload: Mapping[str, object], field: str) -> dict:
    result = copy.deepcopy(dict(payload))
    result.pop(field, None)
    return result


def _module_path() -> Path:
    return Path(__file__).resolve()


def _key_sha256(keys: Sequence[tuple]) -> str:
    encoded = [[int(key[0]), int(key[1]), float(key[2]), float(key[3])] for key in keys]
    canonical = json.dumps(encoded, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _planned_paths(protocol: Mapping[str, object]) -> dict[str, Path]:
    planned = protocol.get("planned_artifacts")
    _require(isinstance(planned, Mapping), "protocol planned artifact manifest missing")
    result = {name: Path(str(planned[name])).resolve() for name in ("execution_module", "execution_amendment", "development_selection", "outer_comparator")}
    _require(result["execution_module"] == _module_path(), "execution module path differs from frozen protocol")
    _require(result["execution_amendment"] == Path(DEFAULT_AMENDMENT).resolve(), "amendment path differs from frozen protocol")
    _require(result["development_selection"] == Path(DEFAULT_DEVELOPMENT).resolve(), "development output path differs from frozen protocol")
    _require(result["outer_comparator"] == Path(DEFAULT_OUTER).resolve(), "outer output path differs from frozen protocol")
    return result


def _require_path(path: str | Path, expected: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    _require(resolved == expected, f"{label} path differs from frozen protocol")
    _require(not resolved.exists(), f"{label} already exists: {resolved}")
    return resolved


def _load_audit(protocol: Mapping[str, object], path: str | Path = DEFAULT_AUDIT) -> dict:
    audit_path = Path(path).resolve()
    _require(audit_path.is_file(), f"protocol audit is absent: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _require(audit.get("artifact_status") == "tuji1_v9_postlock_wknn_protocol_audit_passed", "protocol audit status differs")
    _require(audit.get("protocol_version") == PROTOCOL_VERSION, "protocol audit version differs")
    _require(audit.get("protocol_self_hash") == protocol["protocol_sha256"], "audit protocol self-hash differs")
    _require(audit.get("protocol_file_sha256") == _sha256_file(audit["protocol_path"]), "audit protocol file hash differs")
    _require(audit.get("audit_sha256") == _canonical_json_sha256(_without_hash(audit, "audit_sha256")), "protocol audit self-hash failed")
    state = audit.get("pre_execution_state")
    _require(isinstance(state, Mapping), "protocol audit state missing")
    for field in ("development_selection_performed", "baseline_author_test_comparator_performed", "author_test_labels_loaded_for_wknn_audit", "author_test_features_transformed_for_wknn_audit"):
        _require(state.get(field) is False, f"protocol audit pre-execution field changed: {field}")
    return audit


def freeze_execution_amendment(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    audit_path: str | Path = DEFAULT_AUDIT,
    output: str | Path = DEFAULT_AMENDMENT,
) -> dict:
    """Freeze the exact comparator source after protocol audit and before data work."""

    destination = Path(output).resolve()
    _require(destination == Path(DEFAULT_AMENDMENT).resolve(), "amendment output path differs")
    _require(not destination.exists(), f"execution amendment already exists: {destination}")
    protocol = load_protocol(protocol_path)
    paths = _planned_paths(protocol)
    _require(paths["execution_module"].is_file(), "frozen execution module is absent")
    _require(not paths["development_selection"].exists(), "development selection exists before amendment")
    _require(not paths["outer_comparator"].exists(), "outer comparator exists before amendment")
    audit = _load_audit(protocol, audit_path)
    status = dict(protocol["status"])
    payload: dict[str, object] = {
        "artifact_status": "tuji1_v9_wknn_postlock_execution_amendment_frozen",
        "amendment_version": "V1-A1",
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "protocol": {
            "path": str(Path(protocol_path).resolve()),
            "file_sha256": _sha256_file(protocol_path),
            "self_hash": protocol["protocol_sha256"],
            "document_path": protocol["protocol_document"]["path"],
            "document_sha256": protocol["protocol_document"]["sha256"],
        },
        "audit": {
            "path": str(Path(audit_path).resolve()),
            "file_sha256": _sha256_file(audit_path),
            "self_hash": audit["audit_sha256"],
        },
        "source_contract": {
            "execution_module": str(paths["execution_module"]),
            "execution_module_sha256": _sha256_file(paths["execution_module"]),
            "baseline_primitive": protocol["source_contract"]["baseline_primitive"],
            "baseline_primitive_sha256": protocol["source_contract"]["baseline_primitive_sha256"],
        },
        "output_paths": {
            "development_selection": str(paths["development_selection"]),
            "outer_comparator": str(paths["outer_comparator"]),
        },
        "scientific_change": False,
        "purpose": "freeze comparator implementation and output boundary; no V9 selection authority",
    }
    payload["execution_amendment_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def load_execution_amendment(
    protocol: Mapping[str, object], *, path: str | Path = DEFAULT_AMENDMENT
) -> dict:
    """Load the immutable V1-A1 execution amendment and validate all bindings."""

    amendment_path = Path(path).resolve()
    payload = json.loads(amendment_path.read_text(encoding="utf-8"))
    _require(payload.get("artifact_status") == "tuji1_v9_wknn_postlock_execution_amendment_frozen", "wrong execution amendment")
    _require(payload.get("amendment_version") == "V1-A1", "wrong amendment version")
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "amendment protocol version differs")
    _require(payload.get("execution_amendment_sha256") == _canonical_json_sha256(_without_hash(payload, "execution_amendment_sha256")), "amendment self-hash failed")
    _require(payload.get("status") == protocol["status"], "amendment post-lock status changed")
    pinned = payload["protocol"]
    _require(pinned["self_hash"] == protocol["protocol_sha256"], "amendment protocol self-hash differs")
    _require(_sha256_file(pinned["path"]) == pinned["file_sha256"], "amendment protocol file changed")
    _require(_sha256_file(pinned["document_path"]) == pinned["document_sha256"], "amendment protocol document changed")
    audit = _load_audit(protocol, payload["audit"]["path"])
    pinned_audit = payload["audit"]
    _require(_sha256_file(pinned_audit["path"]) == pinned_audit["file_sha256"], "amendment audit file changed")
    _require(audit["audit_sha256"] == pinned_audit["self_hash"], "amendment audit self-hash differs")
    paths = _planned_paths(protocol)
    source = payload["source_contract"]
    _require(source["execution_module"] == str(paths["execution_module"]), "amendment execution module path differs")
    _require(_sha256_file(source["execution_module"]) == source["execution_module_sha256"], "execution module changed")
    _require(source["baseline_primitive"] == protocol["source_contract"]["baseline_primitive"], "baseline primitive path differs")
    _require(_sha256_file(source["baseline_primitive"]) == source["baseline_primitive_sha256"], "baseline primitive changed")
    outputs = payload["output_paths"]
    _require(outputs["development_selection"] == str(paths["development_selection"]), "development output path differs")
    _require(outputs["outer_comparator"] == str(paths["outer_comparator"]), "outer output path differs")
    return payload


def _assert_global_predictor_contract() -> dict:
    _require(tuple(METHODS) == ("raw_wknn", "diff_wknn", "rank_wknn"), "method family differs")
    _require(tuple(K_GRID) == (1, 3, 5, 8, 12, 20, 32), "k grid differs")
    _require(VISIBILITY_THRESHOLD == -0.999, "TUJI visibility event changed")
    signature = inspect.signature(predict_k_grid)
    _require(tuple(signature.parameters) == ("train_x", "train_y", "test_x", "k_grid", "chunk_size"), "predictor signature differs")
    source = inspect.getsource(predict_k_grid)
    _require("building" not in source and "floor" not in source, "predictor conditions physical strata")
    return {
        "methods": list(METHODS),
        "k_grid": list(K_GRID),
        "visibility_threshold": VISIBILITY_THRESHOLD,
        "predictor_signature": list(signature.parameters),
        "building_or_floor_conditioning": False,
    }


def _validate_execution_contract(
    *, protocol_path: str | Path = DEFAULT_PROTOCOL, amendment_path: str | Path = DEFAULT_AMENDMENT
) -> tuple[dict, dict, dict]:
    protocol = load_protocol(protocol_path)
    amendment = load_execution_amendment(protocol, path=amendment_path)
    registry = load_registry(protocol["v9_registry"]["path"])
    registry_record = protocol["v9_registry"]
    _require(registry["registry_sha256"] == registry_record["registry_sha256"], "registry self-hash changed")
    _require(_sha256_file(registry_record["path"]) == registry_record["file_sha256"], "registry file changed")
    _assert_global_predictor_contract()
    return protocol, amendment, registry


def check_execution_contract(
    *, protocol_path: str | Path = DEFAULT_PROTOCOL, amendment_path: str | Path = DEFAULT_AMENDMENT
) -> dict:
    """Read-only contract check required before either computational stage."""

    protocol, amendment, registry = _validate_execution_contract(protocol_path=protocol_path, amendment_path=amendment_path)
    return {
        "status": "passed",
        "protocol_sha256": protocol["protocol_sha256"],
        "amendment_sha256": amendment["execution_amendment_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "execution_module_sha256": amendment["source_contract"]["execution_module_sha256"],
        "baseline_primitive_sha256": amendment["source_contract"]["baseline_primitive_sha256"],
        "predictor": _assert_global_predictor_contract(),
    }


def _training_data(registry: Mapping[str, object]) -> dict:
    source = registry["source_files"]["rss_training"]
    return load_tuji_split(Path(str(source["path"])).resolve().parent, split="training")


def _reference(
    data: Mapping[str, object], registry: Mapping[str, object], role: str
) -> tuple[np.ndarray, np.ndarray, list[tuple], np.ndarray]:
    rows = registered_rows(registry, role, data)
    prototype, targets, keys = coordinate_prototypes(
        data["rss"], data["raw_xy"], data["phone"], data["keys"], rows
    )
    _require(set(keys) == registered_role_keys(registry, role), f"{role} prototype keys differ")
    return prototype, targets, keys, rows


def _group_vector(per_group: Mapping[tuple, float], ordered: Sequence[tuple]) -> np.ndarray:
    _require(tuple(sorted(per_group)) == tuple(ordered), "coordinate-group order differs")
    return np.asarray([per_group[key] for key in ordered], dtype=np.float64)


def run_development_selection(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    output: str | Path = DEFAULT_DEVELOPMENT,
) -> dict:
    """Choose one predeclared k per baseline using only author training roles."""

    protocol, amendment, registry = _validate_execution_contract(protocol_path=protocol_path, amendment_path=amendment_path)
    paths = _planned_paths(protocol)
    destination = _require_path(output, paths["development_selection"], "development selection")
    _require(not paths["outer_comparator"].exists(), "outer comparator exists before development selection")
    data = _training_data(registry)
    reference, reference_targets, reference_keys, reference_rows = _reference(data, registry, "selection_refit")
    query_rows = registered_rows(registry, "development", data)
    query_keys = tuple(sorted(registered_role_keys(registry, "development")))
    _require(set(reference_keys).isdisjoint(query_keys), "development query leaks reference coordinates")
    query_rss = np.asarray(data["rss"], dtype=np.float64)[query_rows]
    query_targets = np.asarray(data["raw_xy"], dtype=np.float64)[query_rows]
    method_payload: dict[str, object] = {}
    for method in METHODS:
        predictions = predict_k_grid(features(reference, method), reference_targets, features(query_rss, method), k_grid=K_GRID)
        scores: dict[int, float] = {}
        for k in K_GRID:
            macro, _groups, _errors = group_macro_errors(predictions[k], query_targets, data["keys"], query_rows)
            scores[int(k)] = float(macro)
        selected = select_k_by_coordinate_macro(scores)
        method_payload[method] = {
            "k_scores_coordinate_macro_mae_2d_m": {str(k): scores[k] for k in K_GRID},
            "selected_k": int(selected),
            "selected_development_coordinate_macro_mae_2d_m": float(scores[selected]),
        }
    payload: dict[str, object] = {
        "artifact_status": "tuji1_v9_wknn_postlock_development_selection_v1",
        "protocol_version": PROTOCOL_VERSION,
        "status": dict(protocol["status"]),
        "protocol_sha256": protocol["protocol_sha256"],
        "execution_amendment_sha256": amendment["execution_amendment_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "reference_role": "selection_refit",
        "query_role": "development",
        "reference_coordinate_count": int(len(reference_keys)),
        "reference_row_count": int(len(reference_rows)),
        "query_coordinate_count": int(len(query_keys)),
        "query_row_count": int(len(query_rows)),
        "query_coordinate_key_sha256": _key_sha256(query_keys),
        "methods": method_payload,
        "selection_rule": "smallest k at minimum coordinate-macro 2-D MAE",
        "author_test_rss_loaded": False,
        "author_test_labels_loaded": False,
    }
    payload["development_selection_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def load_development_selection(
    protocol: Mapping[str, object], amendment: Mapping[str, object], *, path: str | Path = DEFAULT_DEVELOPMENT
) -> dict:
    """Validate and load the fixed per-baseline development choices."""

    expected = _planned_paths(protocol)["development_selection"]
    selection_path = Path(path).resolve()
    _require(selection_path == expected, "development selection path differs")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    _require(payload.get("artifact_status") == "tuji1_v9_wknn_postlock_development_selection_v1", "wrong development artifact")
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "development protocol version differs")
    _require(payload.get("status") == protocol["status"], "development post-lock status differs")
    _require(payload.get("protocol_sha256") == protocol["protocol_sha256"], "development protocol hash differs")
    _require(payload.get("execution_amendment_sha256") == amendment["execution_amendment_sha256"], "development amendment hash differs")
    _require(payload.get("development_selection_sha256") == _canonical_json_sha256(_without_hash(payload, "development_selection_sha256")), "development self-hash failed")
    methods = payload.get("methods")
    _require(isinstance(methods, Mapping) and tuple(methods) == METHODS, "development method family differs")
    for method in METHODS:
        record = methods[method]
        _require(isinstance(record, Mapping), f"development record missing: {method}")
        scores = {int(k): float(value) for k, value in record["k_scores_coordinate_macro_mae_2d_m"].items()}
        selected = select_k_by_coordinate_macro(scores)
        _require(int(record["selected_k"]) == selected, f"development selected k differs: {method}")
        _require(np.isclose(float(record["selected_development_coordinate_macro_mae_2d_m"]), scores[selected]), f"development selected score differs: {method}")
    _require(payload.get("author_test_rss_loaded") is False and payload.get("author_test_labels_loaded") is False, "development touched author test")
    return payload


def _load_v9_group_losses(protocol: Mapping[str, object], *, ordered_keys: Sequence[tuple]) -> tuple[np.ndarray, np.ndarray, dict]:
    boundary = protocol["post_lock_timing_boundary"]
    path = Path(str(boundary["v9_author_test_result_path"])).resolve()
    _require(_sha256_file(path) == boundary["v9_author_test_result_file_sha256"], "frozen V9 result file changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    _require(result.get("result_sha256") == boundary["v9_author_test_result_self_hash"], "frozen V9 result self-hash differs")
    energy = np.asarray(result["coordinate_group_energy_score_2d_m"], dtype=np.float64)
    mae = np.asarray(result["coordinate_group_mae_2d_m"], dtype=np.float64)
    _require(energy.shape == (len(ordered_keys),), "V9 energy group length differs")
    _require(mae.shape == (len(ordered_keys),), "V9 MAE group length differs")
    _require(np.isfinite(energy).all() and np.isfinite(mae).all(), "V9 group losses are non-finite")
    return energy, mae, result


def run_outer_comparator(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    development_selection_path: str | Path = DEFAULT_DEVELOPMENT,
    output: str | Path = DEFAULT_OUTER,
) -> dict:
    """Run the single locked RSS-prediction and post-lock paired comparison."""

    protocol, amendment, registry = _validate_execution_contract(protocol_path=protocol_path, amendment_path=amendment_path)
    paths = _planned_paths(protocol)
    destination = _require_path(output, paths["outer_comparator"], "outer comparator")
    selection = load_development_selection(protocol, amendment, path=development_selection_path)
    data = _training_data(registry)
    reference, reference_targets, reference_keys, reference_rows = _reference(data, registry, "final_refit")
    query = load_tuji_test_rss_only(registry)
    query_rss = np.asarray(query["rss"], dtype=np.float64)
    predictions: dict[str, np.ndarray] = {}
    for method in METHODS:
        selected_k = int(selection["methods"][method]["selected_k"])
        predictions[method] = predict_k_grid(
            features(reference, method), reference_targets, features(query_rss, method), k_grid=(selected_k,)
        )[selected_k]
    # Test labels are intentionally opened only after all RSS-derived predictions exist.
    labels = load_tuji_test_labels_only(registry, expected_rows=int(query["rows"]))
    query_indices = np.arange(int(query["rows"]), dtype=np.int64)
    ordered_keys = tuple(sorted(set(labels["keys"])))
    _require(len(ordered_keys) == 431, "author-test coordinate group count differs")
    _require(set(reference_keys).isdisjoint(ordered_keys), "author test leaks final reference coordinates")
    v9_energy, v9_mae, v9_result = _load_v9_group_losses(protocol, ordered_keys=ordered_keys)
    method_payload: dict[str, object] = {}
    energy_vectors: list[np.ndarray] = []
    mae_vectors: list[np.ndarray] = []
    for method in METHODS:
        macro, per_group, row_errors = group_macro_errors(predictions[method], labels["raw_xy"], labels["keys"], query_indices)
        group_vector = _group_vector(per_group, ordered_keys)
        _require(np.isclose(macro, group_vector.mean()), f"outer macro metric differs: {method}")
        method_payload[method] = {
            "selected_k": int(selection["methods"][method]["selected_k"]),
            "coordinate_macro_mae_2d_m": float(macro),
            "coordinate_macro_energy_score_2d_m": float(macro),
            "coordinate_group_mae_2d_m": group_vector.tolist(),
            "coordinate_group_energy_score_2d_m": group_vector.tolist(),
            "row_mae_2d_m": float(np.mean(row_errors)),
        }
        energy_vectors.append(group_vector)
        mae_vectors.append(group_vector)
    energy_pairs = simultaneous_paired_upper_bounds(
        v9_energy, energy_vectors, confidence=0.95, bootstrap_repetitions=BOOTSTRAP_REPETITIONS, seed=PAIRED_BOOTSTRAP_SEED
    )
    mae_pairs = simultaneous_paired_upper_bounds(
        v9_mae, mae_vectors, confidence=0.95, bootstrap_repetitions=BOOTSTRAP_REPETITIONS, seed=PAIRED_BOOTSTRAP_SEED
    )
    difference_order = [f"v9_minus_{method}" for method in METHODS]
    energy_pairs["difference_order"] = difference_order
    mae_pairs["difference_order"] = difference_order
    payload: dict[str, object] = {
        "artifact_status": "tuji1_v9_wknn_postlock_outer_comparator_v1",
        "protocol_version": PROTOCOL_VERSION,
        "status": dict(protocol["status"]),
        "comparison_status": "post_lock_comparator",
        "protocol_sha256": protocol["protocol_sha256"],
        "execution_amendment_sha256": amendment["execution_amendment_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "development_selection_path": str(Path(development_selection_path).resolve()),
        "development_selection_file_sha256": _sha256_file(development_selection_path),
        "development_selection_self_hash": selection["development_selection_sha256"],
        "reference_role": "final_refit",
        "reference_coordinate_count": int(len(reference_keys)),
        "reference_row_count": int(len(reference_rows)),
        "query_role": "official_author_test",
        "baseline_author_test_access_number": 1,
        "author_test_prediction_passes": 1,
        "author_test_labels_loaded_after_predictions": True,
        "coordinate_group_count": int(len(ordered_keys)),
        "coordinate_group_order": "lexicographically sorted (building,floor,x,y)",
        "coordinate_group_key_sha256": _key_sha256(ordered_keys),
        "v9_author_test_result_path": protocol["post_lock_timing_boundary"]["v9_author_test_result_path"],
        "v9_author_test_result_self_hash": v9_result["result_sha256"],
        "v9_coordinate_macro_energy_score_2d_m": float(np.mean(v9_energy)),
        "v9_coordinate_macro_mae_2d_m": float(np.mean(v9_mae)),
        "methods": method_payload,
        "paired_v9_minus_wknn_energy": energy_pairs,
        "paired_v9_minus_wknn_mae": mae_pairs,
        "reporting_rule": "all three named methods are reported; simultaneous upper bounds govern family claims",
        "metric_note": "WKNN is deterministic; its degenerate predictive energy score equals its Euclidean point error",
    }
    payload["outer_comparator_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    amendment = children.add_parser("freeze-amendment")
    amendment.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    amendment.add_argument("--audit", default=str(DEFAULT_AUDIT))
    amendment.add_argument("--output", default=str(DEFAULT_AMENDMENT))
    check = children.add_parser("check-contract")
    check.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    check.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    development = children.add_parser("development")
    development.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    development.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    development.add_argument("--output", default=str(DEFAULT_DEVELOPMENT))
    outer = children.add_parser("outer")
    outer.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    outer.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    outer.add_argument("--development-selection", default=str(DEFAULT_DEVELOPMENT))
    outer.add_argument("--output", default=str(DEFAULT_OUTER))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "freeze-amendment":
        payload = freeze_execution_amendment(protocol_path=args.protocol, audit_path=args.audit, output=args.output)
        print(json.dumps({"amendment": str(Path(args.output).resolve()), "execution_amendment_sha256": payload["execution_amendment_sha256"]}))
    elif args.command == "check-contract":
        print(json.dumps(check_execution_contract(protocol_path=args.protocol, amendment_path=args.amendment)))
    elif args.command == "development":
        payload = run_development_selection(protocol_path=args.protocol, amendment_path=args.amendment, output=args.output)
        print(json.dumps({"development_selection": str(Path(args.output).resolve()), "development_selection_sha256": payload["development_selection_sha256"]}))
    else:
        payload = run_outer_comparator(protocol_path=args.protocol, amendment_path=args.amendment, development_selection_path=args.development_selection, output=args.output)
        print(json.dumps({"outer_comparator": str(Path(args.output).resolve()), "outer_comparator_sha256": payload["outer_comparator_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
