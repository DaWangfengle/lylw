"""Mechanical A2 continuation for TUJI1's frozen post-lock WKNN protocol.

V1 correctly completed training-only development selection, but its loader
mistook canonical JSON key order for the semantic baseline order.  This module
does not change a baseline, a grid, a selected ``k``, a metric, or V9.  It
only validates that the *set* of method names equals the frozen family before
using the already self-hashed V1 development artifact.  V1 and A1 remain
immutable evidence.
"""

from __future__ import annotations

import argparse
import copy
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
from fhm_dapp_tim.fhm_dapp.tuji_v9_protocol import load_registry
from fhm_dapp_tim.fhm_dapp.tuji_v9_wknn_common import K_GRID, METHODS, features, group_macro_errors, predict_k_grid
from fhm_dapp_tim.fhm_dapp.tuji_v9_wknn_protocol import (
    BOOTSTRAP_REPETITIONS,
    DEFAULT_DEVELOPMENT,
    DEFAULT_OUTER,
    DEFAULT_PROTOCOL,
    PAIRED_BOOTSTRAP_SEED,
    PROTOCOL_VERSION,
    load_protocol,
    select_k_by_coordinate_macro,
)
from fhm_dapp_tim.fhm_dapp import tuji_v9_wknn_comparator_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v9_tuji"
DEFAULT_A2 = EVIDENCE / "tuji1_wknn_postlock_execution_amendment_v1_a2_method_order.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _without_hash(payload: Mapping[str, object], field: str) -> dict:
    result = copy.deepcopy(dict(payload))
    result.pop(field, None)
    return result


def _module_path() -> Path:
    return Path(__file__).resolve()


def _load_v1_development_selection(protocol: Mapping[str, object], *, path: str | Path = DEFAULT_DEVELOPMENT) -> dict:
    """Validate V1's actual data semantics without relying on JSON key order."""

    expected = v1._planned_paths(protocol)["development_selection"]
    selection_path = Path(path).resolve()
    _require(selection_path == expected, "V1 development selection path differs")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    _require(payload.get("artifact_status") == "tuji1_v9_wknn_postlock_development_selection_v1", "wrong V1 development artifact")
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "V1 development protocol version differs")
    _require(payload.get("status") == protocol["status"], "V1 development post-lock status differs")
    _require(payload.get("protocol_sha256") == protocol["protocol_sha256"], "V1 development protocol hash differs")
    _require(payload.get("development_selection_sha256") == _canonical_json_sha256(_without_hash(payload, "development_selection_sha256")), "V1 development self-hash failed")
    methods = payload.get("methods")
    _require(isinstance(methods, Mapping), "V1 development method map missing")
    _require(set(methods) == set(METHODS) and len(methods) == len(METHODS), "V1 development method family differs")
    for method in METHODS:
        record = methods[method]
        _require(isinstance(record, Mapping), f"V1 development record missing: {method}")
        scores = {int(k): float(value) for k, value in record["k_scores_coordinate_macro_mae_2d_m"].items()}
        selected = select_k_by_coordinate_macro(scores)
        _require(int(record["selected_k"]) == selected, f"V1 development selected k differs: {method}")
        _require(np.isclose(float(record["selected_development_coordinate_macro_mae_2d_m"]), scores[selected]), f"V1 development selected score differs: {method}")
    _require(payload.get("author_test_rss_loaded") is False and payload.get("author_test_labels_loaded") is False, "V1 development touched author test")
    return payload


def freeze_a2(
    *, protocol_path: str | Path = DEFAULT_PROTOCOL, output: str | Path = DEFAULT_A2
) -> dict:
    """Freeze an A2 code-only correction after V1 selection and before test access."""

    destination = Path(output).resolve()
    _require(destination == Path(DEFAULT_A2).resolve(), "A2 output path differs")
    _require(not destination.exists(), f"A2 already exists: {destination}")
    protocol = load_protocol(protocol_path)
    a1 = v1.load_execution_amendment(protocol)
    paths = v1._planned_paths(protocol)
    _require(not paths["outer_comparator"].exists(), "outer comparator exists before A2")
    selection = _load_v1_development_selection(protocol)
    _require(_module_path().is_file(), "A2 execution module is absent")
    payload: dict[str, object] = {
        "artifact_status": "tuji1_v9_wknn_postlock_execution_amendment_v1_a2_frozen",
        "amendment_version": "V1-A2-method-order",
        "protocol_version": PROTOCOL_VERSION,
        "status": dict(protocol["status"]),
        "scientific_change": False,
        "mechanical_change": True,
        "correction": {
            "fault": "V1 loader required insertion order tuple(methods)==METHODS after canonical JSON serialization sorted mapping keys",
            "fix": "validate exact method-key set and cardinality; retain frozen METHODS order only for computation and reporting",
            "changes_to_grid_or_scores": False,
            "changes_to_selected_k": False,
            "changes_to_v9": False,
        },
        "test_access_boundary": {
            "baseline_author_test_accesses_before_a2": 0,
            "baseline_author_test_rss_or_labels_loaded_by_a2": False,
        },
        "protocol": {
            "path": str(Path(protocol_path).resolve()),
            "file_sha256": _sha256_file(protocol_path),
            "self_hash": protocol["protocol_sha256"],
        },
        "a1_lineage": {
            "path": str(Path(v1.DEFAULT_AMENDMENT).resolve()),
            "file_sha256": _sha256_file(v1.DEFAULT_AMENDMENT),
            "self_hash": a1["execution_amendment_sha256"],
            "execution_module": a1["source_contract"]["execution_module"],
            "execution_module_sha256": a1["source_contract"]["execution_module_sha256"],
        },
        "development_selection": {
            "path": str(Path(DEFAULT_DEVELOPMENT).resolve()),
            "file_sha256": _sha256_file(DEFAULT_DEVELOPMENT),
            "self_hash": selection["development_selection_sha256"],
            "author_test_rss_loaded": False,
            "author_test_labels_loaded": False,
            "selected_k": {method: int(selection["methods"][method]["selected_k"]) for method in METHODS},
        },
        "source_contract": {
            "a2_execution_module": str(_module_path()),
            "a2_execution_module_sha256": _sha256_file(_module_path()),
            "baseline_primitive": a1["source_contract"]["baseline_primitive"],
            "baseline_primitive_sha256": a1["source_contract"]["baseline_primitive_sha256"],
        },
        "output_path": str(paths["outer_comparator"]),
    }
    payload["execution_amendment_a2_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def load_a2(protocol: Mapping[str, object], *, path: str | Path = DEFAULT_A2) -> dict:
    """Validate the code-only A2 lineage and its no-test-access boundary."""

    amendment_path = Path(path).resolve()
    payload = json.loads(amendment_path.read_text(encoding="utf-8"))
    _require(payload.get("artifact_status") == "tuji1_v9_wknn_postlock_execution_amendment_v1_a2_frozen", "wrong A2 artifact")
    _require(payload.get("amendment_version") == "V1-A2-method-order", "wrong A2 version")
    _require(payload.get("protocol_version") == PROTOCOL_VERSION, "A2 protocol version differs")
    _require(payload.get("execution_amendment_a2_sha256") == _canonical_json_sha256(_without_hash(payload, "execution_amendment_a2_sha256")), "A2 self-hash failed")
    _require(payload.get("status") == protocol["status"], "A2 post-lock status differs")
    _require(payload.get("scientific_change") is False and payload.get("mechanical_change") is True, "A2 change classification differs")
    boundary = payload.get("test_access_boundary")
    _require(isinstance(boundary, Mapping), "A2 test boundary missing")
    _require(boundary.get("baseline_author_test_accesses_before_a2") == 0, "A2 access count differs")
    _require(boundary.get("baseline_author_test_rss_or_labels_loaded_by_a2") is False, "A2 touched test data")
    pinned_protocol = payload["protocol"]
    _require(pinned_protocol["self_hash"] == protocol["protocol_sha256"], "A2 protocol self-hash differs")
    _require(_sha256_file(pinned_protocol["path"]) == pinned_protocol["file_sha256"], "A2 protocol file changed")
    a1 = v1.load_execution_amendment(protocol)
    lineage = payload["a1_lineage"]
    _require(lineage["self_hash"] == a1["execution_amendment_sha256"], "A2 A1 self-hash differs")
    _require(_sha256_file(lineage["path"]) == lineage["file_sha256"], "A2 A1 file changed")
    _require(_sha256_file(lineage["execution_module"]) == lineage["execution_module_sha256"], "A2 V1 module changed")
    source = payload["source_contract"]
    _require(source["a2_execution_module"] == str(_module_path()), "A2 module path differs")
    _require(_sha256_file(source["a2_execution_module"]) == source["a2_execution_module_sha256"], "A2 module changed")
    _require(_sha256_file(source["baseline_primitive"]) == source["baseline_primitive_sha256"], "A2 primitive changed")
    selection = _load_v1_development_selection(protocol, path=payload["development_selection"]["path"])
    pinned_selection = payload["development_selection"]
    _require(_sha256_file(pinned_selection["path"]) == pinned_selection["file_sha256"], "A2 development file changed")
    _require(selection["development_selection_sha256"] == pinned_selection["self_hash"], "A2 development self-hash differs")
    _require(pinned_selection["selected_k"] == {method: int(selection["methods"][method]["selected_k"]) for method in METHODS}, "A2 selected k differs")
    _require(Path(payload["output_path"]).resolve() == v1._planned_paths(protocol)["outer_comparator"], "A2 outer path differs")
    return payload


def _validate_a2_contract(
    *, protocol_path: str | Path = DEFAULT_PROTOCOL, amendment_path: str | Path = DEFAULT_A2
) -> tuple[dict, dict, dict, dict]:
    protocol = load_protocol(protocol_path)
    a2 = load_a2(protocol, path=amendment_path)
    registry = load_registry(protocol["v9_registry"]["path"])
    selection = _load_v1_development_selection(protocol)
    v1._assert_global_predictor_contract()
    return protocol, a2, registry, selection


def check_a2_contract(
    *, protocol_path: str | Path = DEFAULT_PROTOCOL, amendment_path: str | Path = DEFAULT_A2
) -> dict:
    protocol, a2, registry, selection = _validate_a2_contract(protocol_path=protocol_path, amendment_path=amendment_path)
    return {
        "status": "passed",
        "protocol_sha256": protocol["protocol_sha256"],
        "a2_sha256": a2["execution_amendment_a2_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "selection_sha256": selection["development_selection_sha256"],
        "selected_k": {method: int(selection["methods"][method]["selected_k"]) for method in METHODS},
        "baseline_author_test_accesses_before_outer": 0,
    }


def run_outer_comparator_a2(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    amendment_path: str | Path = DEFAULT_A2,
    development_selection_path: str | Path = DEFAULT_DEVELOPMENT,
    output: str | Path = DEFAULT_OUTER,
) -> dict:
    """Execute the same frozen comparator after A2's mapping-only repair."""

    protocol, a2, registry, selection = _validate_a2_contract(protocol_path=protocol_path, amendment_path=amendment_path)
    _require(Path(development_selection_path).resolve() == Path(DEFAULT_DEVELOPMENT).resolve(), "A2 development path differs")
    destination = Path(output).resolve()
    expected = v1._planned_paths(protocol)["outer_comparator"]
    _require(destination == expected, "A2 outer output path differs")
    _require(not destination.exists(), f"outer comparator already exists: {destination}")
    data = v1._training_data(registry)
    reference, reference_targets, reference_keys, reference_rows = v1._reference(data, registry, "final_refit")
    query = load_tuji_test_rss_only(registry)
    query_rss = np.asarray(query["rss"], dtype=np.float64)
    predictions: dict[str, np.ndarray] = {}
    for method in METHODS:
        selected_k = int(selection["methods"][method]["selected_k"])
        predictions[method] = predict_k_grid(
            features(reference, method), reference_targets, features(query_rss, method), k_grid=(selected_k,)
        )[selected_k]
    # This follows the original locked order: test labels are loaded only now.
    labels = load_tuji_test_labels_only(registry, expected_rows=int(query["rows"]))
    query_indices = np.arange(int(query["rows"]), dtype=np.int64)
    ordered_keys = tuple(sorted(set(labels["keys"])))
    _require(len(ordered_keys) == 431, "author-test coordinate group count differs")
    _require(set(reference_keys).isdisjoint(ordered_keys), "author test leaks final reference coordinates")
    v9_energy, v9_mae, v9_result = v1._load_v9_group_losses(protocol, ordered_keys=ordered_keys)
    method_payload: dict[str, object] = {}
    comparator_vectors: list[np.ndarray] = []
    for method in METHODS:
        macro, per_group, row_errors = group_macro_errors(predictions[method], labels["raw_xy"], labels["keys"], query_indices)
        group_vector = v1._group_vector(per_group, ordered_keys)
        _require(np.isclose(macro, group_vector.mean()), f"outer macro metric differs: {method}")
        method_payload[method] = {
            "selected_k": int(selection["methods"][method]["selected_k"]),
            "coordinate_macro_mae_2d_m": float(macro),
            "coordinate_macro_energy_score_2d_m": float(macro),
            "coordinate_group_mae_2d_m": group_vector.tolist(),
            "coordinate_group_energy_score_2d_m": group_vector.tolist(),
            "row_mae_2d_m": float(np.mean(row_errors)),
        }
        comparator_vectors.append(group_vector)
    energy_pairs = simultaneous_paired_upper_bounds(v9_energy, comparator_vectors, confidence=0.95, bootstrap_repetitions=BOOTSTRAP_REPETITIONS, seed=PAIRED_BOOTSTRAP_SEED)
    mae_pairs = simultaneous_paired_upper_bounds(v9_mae, comparator_vectors, confidence=0.95, bootstrap_repetitions=BOOTSTRAP_REPETITIONS, seed=PAIRED_BOOTSTRAP_SEED)
    difference_order = [f"v9_minus_{method}" for method in METHODS]
    energy_pairs["difference_order"] = difference_order
    mae_pairs["difference_order"] = difference_order
    payload: dict[str, object] = {
        "artifact_status": "tuji1_v9_wknn_postlock_outer_comparator_v1_a2",
        "protocol_version": PROTOCOL_VERSION,
        "status": dict(protocol["status"]),
        "comparison_status": "post_lock_comparator",
        "execution_amendment_a2_sha256": a2["execution_amendment_a2_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "a1_development_selection_path": str(Path(development_selection_path).resolve()),
        "a1_development_selection_file_sha256": _sha256_file(development_selection_path),
        "a1_development_selection_self_hash": selection["development_selection_sha256"],
        "mechanical_correction": "A2 validates method-name set rather than JSON insertion order; selected values are inherited unchanged",
        "reference_role": "final_refit",
        "reference_coordinate_count": int(len(reference_keys)),
        "reference_row_count": int(len(reference_rows)),
        "query_role": "official_author_test",
        "baseline_author_test_access_number": 1,
        "author_test_prediction_passes": 1,
        "author_test_labels_loaded_after_predictions": True,
        "coordinate_group_count": int(len(ordered_keys)),
        "coordinate_group_order": "lexicographically sorted (building,floor,x,y)",
        "coordinate_group_key_sha256": v1._key_sha256(ordered_keys),
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
    freeze = children.add_parser("freeze-a2")
    freeze.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    freeze.add_argument("--output", default=str(DEFAULT_A2))
    check = children.add_parser("check-contract")
    check.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    check.add_argument("--amendment", default=str(DEFAULT_A2))
    outer = children.add_parser("outer")
    outer.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    outer.add_argument("--amendment", default=str(DEFAULT_A2))
    outer.add_argument("--development-selection", default=str(DEFAULT_DEVELOPMENT))
    outer.add_argument("--output", default=str(DEFAULT_OUTER))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "freeze-a2":
        payload = freeze_a2(protocol_path=args.protocol, output=args.output)
        print(json.dumps({"a2": str(Path(args.output).resolve()), "execution_amendment_a2_sha256": payload["execution_amendment_a2_sha256"]}))
    elif args.command == "check-contract":
        print(json.dumps(check_a2_contract(protocol_path=args.protocol, amendment_path=args.amendment)))
    else:
        payload = run_outer_comparator_a2(protocol_path=args.protocol, amendment_path=args.amendment, development_selection_path=args.development_selection, output=args.output)
        print(json.dumps({"outer_comparator": str(Path(args.output).resolve()), "outer_comparator_sha256": payload["outer_comparator_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
