"""Executable V2 post-lock WKNN comparator for the frozen V9 experiment.

The module intentionally separates three phases:

1. freeze a naming-only execution amendment after code review;
2. select one ``k`` per named baseline on ``selection_refit -> development``;
3. run one locked comparator on ``outer_refit -> outer_test``.

It has no operation that changes V9 or creates a new V9 internal variant.
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
    _registered_role_keys,
    _registered_rows,
    load_registry,
)
from fhm_dapp_tim.fhm_dapp.uji_v9_wknn_protocol import (
    BOOTSTRAP_REPETITIONS,
    ENERGY_BOOTSTRAP_SEED,
    K_GRID,
    MAE_BOOTSTRAP_SEED,
    METHODS,
    PROTOCOL_VERSION,
    VISIBILITY_THRESHOLD,
    load_protocol,
    select_k_by_coordinate_macro,
)
from fhm_dapp_tim.fhm_dapp.v9_wknn_common import (
    coordinate_prototypes,
    features,
    group_macro_errors,
    predict_k_grid,
)


EXECUTION_VERSION = "uji_v9_wknn_postlock_execution_v2_a1_20260811"
AMENDMENT_DOCUMENT = (
    DEFAULT_EVIDENCE / "POSTLOCK_WKNN_EXECUTION_AMENDMENT_V2_A1.md"
)
DEFAULT_AMENDMENT = (
    DEFAULT_EVIDENCE / "wknn_postlock_execution_amendment_v2_a1.json"
)
DEFAULT_DEVELOPMENT = (
    DEFAULT_EVIDENCE / "wknn_postlock_development_selection_v2.json"
)
DEFAULT_OUTER = DEFAULT_EVIDENCE / "wknn_postlock_outer_comparator_v2.json"
LOCATION_KEY_FIELDS = ("building", "floor", "longitude", "latitude")


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _without_hash(payload: dict, field: str) -> dict:
    value = copy.deepcopy(payload)
    value.pop(field, None)
    return value


def _encode_key(key: tuple) -> list[int | float]:
    return [int(key[0]), int(key[1]), float(key[2]), float(key[3])]


def _key_json(key: tuple) -> str:
    return json.dumps(_encode_key(key), separators=(",", ":"), allow_nan=False)


def _module_path() -> Path:
    return Path(__file__).resolve()


def _planned_original_paths(protocol: dict) -> dict[str, Path]:
    planned = protocol["planned_artifacts"]
    return {
        "development_selection": Path(planned["development_selection"]).resolve(),
        "outer_comparator": Path(planned["outer_comparator"]).resolve(),
    }


def _new_execution_paths() -> dict[str, Path]:
    return {
        "development_selection": DEFAULT_DEVELOPMENT.resolve(),
        "outer_comparator": DEFAULT_OUTER.resolve(),
    }


def _validate_output_path(path: str | Path, expected: Path, label: str) -> Path:
    destination = Path(path).resolve()
    _require(destination == expected.resolve(), f"{label} output path is not frozen")
    _require(destination.name.startswith("wknn_postlock_"), "missing result prefix")
    _require(not destination.exists(), f"refusing to overwrite {label}: {destination}")
    return destination


def freeze_execution_amendment(
    *,
    protocol_path: str | Path | None = None,
    document_path: str | Path = AMENDMENT_DOCUMENT,
    output: str | Path = DEFAULT_AMENDMENT,
) -> dict:
    """Freeze the naming-only amendment after executable-code review."""

    protocol = load_protocol(protocol_path) if protocol_path is not None else load_protocol()
    document = Path(document_path).resolve()
    destination = Path(output).resolve()
    _require(document.exists(), f"amendment document is missing: {document}")
    _require(destination == DEFAULT_AMENDMENT.resolve(), "amendment path is not frozen")
    _require(not destination.exists(), f"refusing to overwrite amendment: {destination}")
    original = _planned_original_paths(protocol)
    replacement = _new_execution_paths()
    _require(
        all(not path.exists() for path in original.values()),
        "an original V2 planned result artifact already exists",
    )
    _require(
        all(not path.exists() for path in replacement.values()),
        "a V2-A1 result artifact already exists",
    )
    source = _module_path()
    payload = {
        "artifact_status": "wknn_postlock_execution_naming_amendment_frozen",
        "execution_version": EXECUTION_VERSION,
        "scientific_status": {
            "created_after_v9_outer_test": True,
            "selection_authority_for_v9": False,
            "may_modify_v9": False,
            "comparison_status": "post_lock_comparator",
            "scientific_change": False,
        },
        "protocol": {
            "path": str(Path(protocol_path).resolve())
            if protocol_path is not None
            else str((DEFAULT_EVIDENCE / "postlock_wknn_protocol_v2.json").resolve()),
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_document_path": protocol["protocol_document"]["path"],
            "protocol_document_sha256": protocol["protocol_document"]["sha256"],
        },
        "amendment_document": {
            "path": str(document),
            "sha256": _sha256_file(document),
        },
        "execution_module": {
            "path": str(source),
            "sha256": _sha256_file(source),
        },
        "common_primitive": {
            "path": protocol["source_contract"]["baseline_source"],
            "sha256": protocol["source_contract"]["baseline_source_sha256"],
        },
        "sole_change": "artifact output names only",
        "original_v2_paths": {name: str(path) for name, path in original.items()},
        "replacement_paths": {name: str(path) for name, path in replacement.items()},
        "required_absence_before_execution": {
            **{f"original_{name}": True for name in original},
            **{f"replacement_{name}": True for name in replacement},
        },
    }
    payload["amendment_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def load_execution_amendment(
    *,
    protocol: dict,
    path: str | Path = DEFAULT_AMENDMENT,
) -> dict:
    """Validate the post-review naming amendment and all execution hashes."""

    amendment_path = Path(path).resolve()
    payload = json.loads(amendment_path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_status")
        == "wknn_postlock_execution_naming_amendment_frozen",
        "not a frozen execution amendment",
    )
    _require(payload.get("execution_version") == EXECUTION_VERSION, "wrong execution amendment")
    _require(
        payload.get("amendment_sha256")
        == _canonical_json_sha256(_without_hash(payload, "amendment_sha256")),
        "execution amendment self-hash failed",
    )
    status = payload.get("scientific_status", {})
    _require(status.get("created_after_v9_outer_test") is True, "timing status changed")
    _require(status.get("selection_authority_for_v9") is False, "selection authority changed")
    _require(status.get("may_modify_v9") is False, "V9 modification authority changed")
    _require(status.get("comparison_status") == "post_lock_comparator", "comparison label changed")
    _require(status.get("scientific_change") is False, "amendment changes science")
    pinned = payload["protocol"]
    _require(pinned["protocol_version"] == protocol["protocol_version"], "protocol version changed")
    _require(pinned["protocol_sha256"] == protocol["protocol_sha256"], "protocol hash changed")
    _require(
        _sha256_file(pinned["protocol_document_path"])
        == pinned["protocol_document_sha256"],
        "protocol document changed",
    )
    amendment_document = payload["amendment_document"]
    _require(
        _sha256_file(amendment_document["path"]) == amendment_document["sha256"],
        "amendment document changed",
    )
    module = payload["execution_module"]
    _require(module["path"] == str(_module_path()), "wrong execution module path")
    _require(_sha256_file(module["path"]) == module["sha256"], "execution module changed")
    common = payload["common_primitive"]
    _require(
        common["path"] == protocol["source_contract"]["baseline_source"],
        "common primitive path changed",
    )
    _require(
        _sha256_file(common["path"]) == common["sha256"],
        "common primitive changed",
    )
    original = _planned_original_paths(protocol)
    replacement = _new_execution_paths()
    _require(
        payload["original_v2_paths"] == {name: str(value) for name, value in original.items()},
        "original V2 paths changed",
    )
    _require(
        payload["replacement_paths"]
        == {name: str(value) for name, value in replacement.items()},
        "replacement paths changed",
    )
    return payload


def _validate_execution_contract(
    *, amendment_path: str | Path = DEFAULT_AMENDMENT,
) -> tuple[dict, dict, dict]:
    protocol = load_protocol()
    amendment = load_execution_amendment(protocol=protocol, path=amendment_path)
    registry = load_registry(protocol["v9_registry"]["path"])
    _require(
        registry["registry_sha256"] == protocol["v9_registry"]["registry_sha256"],
        "registry internal self-hash changed",
    )
    _require(
        _sha256_file(protocol["v9_registry"]["path"])
        == protocol["v9_registry"]["file_sha256"],
        "registry file changed",
    )
    return protocol, amendment, registry


def _assert_global_predictor_contract() -> dict:
    """Verify behavior, not merely a textual description, of global WKNN."""

    signature = inspect.signature(predict_k_grid)
    _require(
        tuple(signature.parameters)
        == ("train_x", "train_y", "test_x", "k_grid", "chunk_size"),
        "predictor signature admits conditional labels",
    )
    source = inspect.getsource(predict_k_grid).lower()
    _require("building" not in source and "floor" not in source, "predictor conditions candidates")
    train_x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    train_y = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float64)
    result = predict_k_grid(
        train_x,
        train_y,
        np.asarray([[1.0]], dtype=np.float64),
        k_grid=(1, 3),
        chunk_size=1,
    )
    _require(np.allclose(result[1], [[10.0, 0.0]]), "exact-match weight changed")
    _require(np.allclose(result[3], [[10.0, 0.0]]), "inverse-distance aggregation changed")
    return {
        "predictor_parameters": list(signature.parameters),
        "candidate_scope": "global",
        "building_or_floor_conditioning": False,
        "exact_match_fixture_passed": True,
        "inverse_distance_fixture_passed": True,
    }


def check_execution_contract(
    *, amendment_path: str | Path = DEFAULT_AMENDMENT,
) -> dict:
    """Read-only contract check for the second audit gate."""

    protocol, amendment, registry = _validate_execution_contract(
        amendment_path=amendment_path
    )
    predictor = _assert_global_predictor_contract()
    replacement = _new_execution_paths()
    return {
        "execution_version": EXECUTION_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "amendment_sha256": amendment["amendment_sha256"],
        "execution_module_sha256": _sha256_file(_module_path()),
        "registry_sha256": registry["registry_sha256"],
        "common_primitive_sha256": protocol["source_contract"]["baseline_source_sha256"],
        "predictor_contract": predictor,
        "replacement_outputs": {name: str(path) for name, path in replacement.items()},
        "new_wknn_predictions_computed": False,
        "new_wknn_errors_computed": False,
    }


def _role_rows(
    registry: dict,
    data: dict,
    protocol: dict,
    role: str,
) -> tuple[np.ndarray, tuple[tuple, ...]]:
    rows = _registered_rows(registry, role, data)
    keys = _registered_role_keys(registry, role)
    contract = protocol["data_contract"]["roles"][role]
    _require(len(rows) == contract["row_count"], f"{role} row count changed")
    _require(len(keys) == contract["coordinate_count"], f"{role} coordinate count changed")
    _require(
        _sha256_indices(rows) == contract["row_index_sha256"],
        f"{role} row identity changed",
    )
    _require(
        _sha256_keys(keys) == contract["coordinate_key_sha256"],
        f"{role} coordinate identity changed",
    )
    ordered = tuple(sorted(keys))
    return rows, ordered


def _group_error_vector(
    per_group: Mapping[tuple, float],
    ordered_keys: Sequence[tuple],
) -> np.ndarray:
    _require(set(per_group) == set(ordered_keys), "group identities are not aligned")
    values = np.asarray([per_group[key] for key in ordered_keys], dtype=np.float64)
    _require(np.isfinite(values).all(), "group error vector is nonfinite")
    return values


def _public_group_map(per_group: Mapping[tuple, float]) -> dict[str, float]:
    return {_key_json(key): float(value) for key, value in sorted(per_group.items())}


def _role_reference(
    *,
    data: dict,
    rows: np.ndarray,
    expected_keys: Sequence[tuple],
) -> tuple[np.ndarray, np.ndarray]:
    prototypes, targets, prototype_keys = coordinate_prototypes(
        data["rss"].numpy(),
        data["raw_xy"],
        data["phone"],
        data["keys"],
        rows,
    )
    _require(tuple(prototype_keys) == tuple(expected_keys), "reference prototype keys changed")
    _require(prototypes.shape == (len(expected_keys), 520), "reference shape changed")
    _require(np.isfinite(prototypes).all() and np.isfinite(targets).all(), "bad reference")
    return prototypes, targets


def run_development_selection(
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    output: str | Path = DEFAULT_DEVELOPMENT,
) -> dict:
    """Select one development-risk-minimizing ``k`` per named representation."""

    protocol, amendment, registry = _validate_execution_contract(
        amendment_path=amendment_path
    )
    destination = _validate_output_path(
        output,
        Path(amendment["replacement_paths"]["development_selection"]),
        "development selection",
    )
    data = _load_uji(registry["train_data"])
    reference_rows, reference_keys = _role_rows(
        registry, data, protocol, "selection_refit"
    )
    query_rows, query_keys = _role_rows(registry, data, protocol, "development")
    _require(set(reference_keys).isdisjoint(query_keys), "development query leaks reference coordinates")
    prototypes, prototype_targets = _role_reference(
        data=data, rows=reference_rows, expected_keys=reference_keys
    )
    query_rss = data["rss"].numpy()[query_rows]
    query_targets = np.asarray(data["raw_xy"][query_rows], dtype=np.float64)
    method_payload = {}
    for method in METHODS:
        predictions = predict_k_grid(
            features(prototypes, method),
            prototype_targets,
            features(query_rss, method),
            k_grid=K_GRID,
        )
        per_k = {}
        macro_scores = {}
        for k in K_GRID:
            macro, per_group, row_errors = group_macro_errors(
                predictions[k], query_targets, data["keys"], query_rows
            )
            group_values = _group_error_vector(per_group, query_keys)
            _require(np.isclose(macro, group_values.mean()), "macro metric changed")
            macro_scores[int(k)] = float(macro)
            per_k[str(k)] = {
                "coordinate_macro_mae_2d_m": float(macro),
                "row_micro_mae_2d_m": float(np.mean(row_errors)),
                "coordinate_group_errors_2d_m": group_values.tolist(),
                "coordinate_group_errors_by_key": _public_group_map(per_group),
            }
        selected = select_k_by_coordinate_macro(macro_scores, k_grid=K_GRID)
        method_payload[method] = {
            "selected_k": int(selected),
            "development_coordinate_macro_mae_2d_m_by_k": {
                str(k): macro_scores[k] for k in K_GRID
            },
            "all_k_results": per_k,
        }
    _require(tuple(method_payload) == METHODS, "not all named methods were evaluated")
    payload = {
        "artifact_status": "wknn_postlock_development_selection_v2",
        "execution_version": EXECUTION_VERSION,
        "comparison_status": "post_lock_comparator",
        "selection_authority_for_v9": False,
        "may_modify_v9": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "amendment_sha256": amendment["amendment_sha256"],
        "execution_module_sha256": _sha256_file(_module_path()),
        "registry_sha256": registry["registry_sha256"],
        "training_data_sha256": registry["train_data_sha256"],
        "reference_role": "selection_refit",
        "query_role": "development",
        "reference_row_index_sha256": _sha256_indices(reference_rows),
        "query_row_index_sha256": _sha256_indices(query_rows),
        "reference_coordinate_key_sha256": _sha256_keys(reference_keys),
        "query_coordinate_key_sha256": _sha256_keys(query_keys),
        "coordinate_group_keys": [_encode_key(key) for key in query_keys],
        "reference_prototype_count": len(reference_keys),
        "representation_methods": list(METHODS),
        "k_grid": list(K_GRID),
        "numeric_tie_tolerance_m": 1e-12,
        "tie_break": "smallest k",
        "uses_device_risk_guard": False,
        "candidate_scope": "global",
        "query_building_or_floor_conditioning": False,
        "query_device_or_user_conditioning": False,
        "query_phone_id_accessed": False,
        "outer_test_features_transformed": False,
        "methods": method_payload,
    }
    payload["development_selection_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def load_development_selection(
    *,
    protocol: dict,
    amendment: dict,
    registry: dict,
    path: str | Path = DEFAULT_DEVELOPMENT,
) -> dict:
    selection_path = Path(path).resolve()
    _require(
        selection_path == Path(amendment["replacement_paths"]["development_selection"]),
        "development selection path changed",
    )
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_status") == "wknn_postlock_development_selection_v2",
        "not a V2 development selection",
    )
    _require(
        payload.get("development_selection_sha256")
        == _canonical_json_sha256(_without_hash(payload, "development_selection_sha256")),
        "development selection self-hash failed",
    )
    for field, expected in (
        ("execution_version", EXECUTION_VERSION),
        ("comparison_status", "post_lock_comparator"),
        ("selection_authority_for_v9", False),
        ("may_modify_v9", False),
        ("protocol_sha256", protocol["protocol_sha256"]),
        ("amendment_sha256", amendment["amendment_sha256"]),
        ("registry_sha256", registry["registry_sha256"]),
    ):
        _require(payload.get(field) == expected, f"development selection changed: {field}")
    _require(
        payload.get("execution_module_sha256") == _sha256_file(_module_path()),
        "development selection executable changed",
    )
    _require(tuple(payload.get("representation_methods", ())) == METHODS, "methods changed")
    _require(tuple(payload.get("k_grid", ())) == K_GRID, "k grid changed")
    for method in METHODS:
        record = payload["methods"].get(method)
        _require(record is not None, f"missing method: {method}")
        scores = {
            int(k): float(value)
            for k, value in record["development_coordinate_macro_mae_2d_m_by_k"].items()
        }
        _require(
            int(record["selected_k"]) == select_k_by_coordinate_macro(scores),
            f"selection rule changed for {method}",
        )
    return payload


def _load_v9_outer_group_losses(
    *,
    protocol: dict,
    registry: dict,
    ordered_outer_keys: Sequence[tuple],
) -> tuple[np.ndarray, np.ndarray, dict]:
    boundary = protocol["post_lock_timing_boundary"]
    result_path = Path(boundary["v9_outer_result_path"])
    _require(
        _sha256_file(result_path) == boundary["v9_outer_result_file_sha256"],
        "frozen V9 outer result changed",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require(result.get("registry_sha256") == registry["registry_sha256"], "V9 result registry changed")
    _require(result.get("outer_test_access_number") == 1, "V9 access count changed")
    _require(
        result.get("result_sha256")
        == _canonical_json_sha256(_without_hash(result, "result_sha256")),
        "V9 result self-hash failed",
    )
    energy = np.asarray(result["coordinate_group_energy_score_2d_m"], dtype=np.float64)
    mae = np.asarray(result["coordinate_group_mae_2d_m"], dtype=np.float64)
    _require(energy.shape == (len(ordered_outer_keys),), "V9 energy group length changed")
    _require(mae.shape == (len(ordered_outer_keys),), "V9 MAE group length changed")
    _require(np.isfinite(energy).all() and np.isfinite(mae).all(), "V9 group loss is nonfinite")
    metrics = result["metrics"]
    _require(
        np.isclose(energy.mean(), float(metrics["coordinate_macro_energy_score_2d_m"])),
        "V9 energy aggregate changed",
    )
    _require(
        np.isclose(mae.mean(), float(metrics["coordinate_macro_mae_2d_m"])),
        "V9 MAE aggregate changed",
    )
    return energy, mae, result


def run_outer_comparator(
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    development_selection_path: str | Path = DEFAULT_DEVELOPMENT,
    output: str | Path = DEFAULT_OUTER,
) -> dict:
    """Run the one locked V2-A1 comparator after development selection."""

    protocol, amendment, registry = _validate_execution_contract(
        amendment_path=amendment_path
    )
    destination = _validate_output_path(
        output,
        Path(amendment["replacement_paths"]["outer_comparator"]),
        "outer comparator",
    )
    selection = load_development_selection(
        protocol=protocol,
        amendment=amendment,
        registry=registry,
        path=development_selection_path,
    )
    data = _load_uji(registry["train_data"])
    reference_rows, reference_keys = _role_rows(registry, data, protocol, "outer_refit")
    query_rows, query_keys = _role_rows(registry, data, protocol, "outer_test")
    _require(set(reference_keys).isdisjoint(query_keys), "outer query leaks reference coordinates")
    prototypes, prototype_targets = _role_reference(
        data=data, rows=reference_rows, expected_keys=reference_keys
    )
    query_rss = data["rss"].numpy()[query_rows]
    query_targets = np.asarray(data["raw_xy"][query_rows], dtype=np.float64)
    v9_energy, v9_mae, v9_result = _load_v9_outer_group_losses(
        protocol=protocol,
        registry=registry,
        ordered_outer_keys=query_keys,
    )
    method_payload = {}
    group_energy = []
    group_mae = []
    for method in METHODS:
        selected_k = int(selection["methods"][method]["selected_k"])
        prediction = predict_k_grid(
            features(prototypes, method),
            prototype_targets,
            features(query_rss, method),
            k_grid=(selected_k,),
        )[selected_k]
        macro, per_group, row_errors = group_macro_errors(
            prediction, query_targets, data["keys"], query_rows
        )
        group_values = _group_error_vector(per_group, query_keys)
        _require(np.isclose(macro, group_values.mean()), "outer macro metric changed")
        group_energy.append(group_values)
        group_mae.append(group_values)
        method_payload[method] = {
            "selected_k": selected_k,
            "row_micro_mae_2d_m": float(np.mean(row_errors)),
            "coordinate_macro_mae_2d_m": float(macro),
            "coordinate_macro_energy_score_2d_m": float(macro),
            "degenerate_energy_score_equals_point_error": True,
            "coordinate_group_errors_2d_m": group_values.tolist(),
            "coordinate_group_errors_by_key": _public_group_map(per_group),
        }
    _require(tuple(method_payload) == METHODS, "not all named methods were reported")
    energy_pairs = simultaneous_paired_upper_bounds(
        v9_energy,
        group_energy,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=ENERGY_BOOTSTRAP_SEED,
    )
    mae_pairs = simultaneous_paired_upper_bounds(
        v9_mae,
        group_mae,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=MAE_BOOTSTRAP_SEED,
    )
    difference_order = [f"v9_minus_{method}" for method in METHODS]
    energy_pairs["difference_order"] = difference_order
    mae_pairs["difference_order"] = difference_order
    payload = {
        "artifact_status": "wknn_postlock_outer_comparator_v2",
        "execution_version": EXECUTION_VERSION,
        "comparison_status": "post_lock_comparator",
        "selection_authority_for_v9": False,
        "may_modify_v9": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "amendment_sha256": amendment["amendment_sha256"],
        "execution_module_sha256": _sha256_file(_module_path()),
        "registry_sha256": registry["registry_sha256"],
        "training_data_sha256": registry["train_data_sha256"],
        "development_selection_path": str(Path(development_selection_path).resolve()),
        "development_selection_file_sha256": _sha256_file(development_selection_path),
        "development_selection_self_hash": selection["development_selection_sha256"],
        "reference_role": "outer_refit",
        "query_role": "outer_test",
        "reference_row_index_sha256": _sha256_indices(reference_rows),
        "query_row_index_sha256": _sha256_indices(query_rows),
        "reference_coordinate_key_sha256": _sha256_keys(reference_keys),
        "query_coordinate_key_sha256": _sha256_keys(query_keys),
        "coordinate_group_keys": [_encode_key(key) for key in query_keys],
        "coordinate_group_order": "sorted registered (building,floor,longitude,latitude)",
        "coordinate_group_count": len(query_keys),
        "reference_prototype_count": len(reference_keys),
        "representation_methods": list(METHODS),
        "candidate_scope": "global",
        "query_building_or_floor_conditioning": False,
        "query_device_or_user_conditioning": False,
        "query_phone_id_accessed": False,
        "v9_outer_result_path": protocol["post_lock_timing_boundary"]["v9_outer_result_path"],
        "v9_outer_result_self_hash": v9_result["result_sha256"],
        "v9_outer_alignment": (
            "V9 cache group order and comparator order are both sorted registered "
            "physical-coordinate keys"
        ),
        "methods": method_payload,
        "paired_v9_minus_wknn_energy": energy_pairs,
        "paired_v9_minus_wknn_mae": mae_pairs,
        "reporting_rule": "all three named methods are reported; simultaneous bounds govern family claims",
    }
    payload["outer_comparator_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    amendment = children.add_parser("freeze-amendment")
    amendment.add_argument("--protocol")
    amendment.add_argument("--document", default=str(AMENDMENT_DOCUMENT))
    amendment.add_argument("--output", default=str(DEFAULT_AMENDMENT))
    check = children.add_parser("check-contract")
    check.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    development = children.add_parser("development")
    development.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    development.add_argument("--output", default=str(DEFAULT_DEVELOPMENT))
    outer = children.add_parser("outer")
    outer.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    outer.add_argument("--development-selection", default=str(DEFAULT_DEVELOPMENT))
    outer.add_argument("--output", default=str(DEFAULT_OUTER))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "freeze-amendment":
        payload = freeze_execution_amendment(
            protocol_path=args.protocol,
            document_path=args.document,
            output=args.output,
        )
        summary = {
            "output": str(Path(args.output).resolve()),
            "amendment_sha256": payload["amendment_sha256"],
            "new_wknn_outcomes_opened": False,
        }
    elif args.command == "check-contract":
        summary = check_execution_contract(amendment_path=args.amendment)
    elif args.command == "development":
        payload = run_development_selection(
            amendment_path=args.amendment,
            output=args.output,
        )
        summary = {
            "output": str(Path(args.output).resolve()),
            "selected_k": {
                method: payload["methods"][method]["selected_k"] for method in METHODS
            },
        }
    elif args.command == "outer":
        payload = run_outer_comparator(
            amendment_path=args.amendment,
            development_selection_path=args.development_selection,
            output=args.output,
        )
        summary = {
            "output": str(Path(args.output).resolve()),
            "methods": list(METHODS),
            "comparison_status": payload["comparison_status"],
        }
    else:  # pragma: no cover - argparse makes this unreachable.
        raise ValueError(args.command)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

