"""Freeze the one-shot V17-R3 outer-confirmation protocol.

This module may inspect only already-frozen development artifacts and the
pre-outcome role metadata in the R0 registry.  It never indexes outer-test rows
and never opens the official UJI validation file.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_protocol import (
    load_registry as load_r0_registry,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2f_protocol import (
    load_registry as load_r2f_registry,
)


PROTOCOL_VERSION = "uji_v17_r3_outer_confirmation_v1_20260822"
PANEL_MEMBER_COUNT = 25
MEMBER_SEEDS = tuple(range(17301, 17326))
CYCLIC_GENERATOR = 7
SUBSET_OFFSET_ORDER = tuple(
    (CYCLIC_GENERATOR * index) % PANEL_MEMBER_COUNT
    for index in range(PANEL_MEMBER_COUNT)
)

SELECTED_NAME = "selected_E16_L192_K192_A1"
HISTORICAL_NAME = "historical_E05_L024_K024_A0p25"
CONFIGURATIONS = {
    SELECTED_NAME: {"E": 16, "L": 192, "K": 192, "alpha": 1.0},
    HISTORICAL_NAME: {"E": 5, "L": 24, "K": 24, "alpha": 0.25},
}
BETA_NEURAL = 1.0
BETA_PROTOTYPE = 32.0 / 3.0
CONSENSUS_EPSILON = 1e-6
STUDENT_DF = 4.0
PREDICTIVE_DRAWS_PER_STREAM = 64
PREDICTIVE_SEED = 17931
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 17932
CONFIDENCE = 0.95
OUTER_REFIT_ANCHORS = 746
OUTER_TEST_ROWS = 4066
OUTER_TEST_COORDINATES = 187
COMPONENT_BATCH_SIZE = 32

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v17_acrhap_mother"
SOURCE_MODEL_R0 = ROOT / "models" / "v17_acrhap_mother_r0"
SOURCE_MODEL_R2 = ROOT / "models" / "v17_acrhap_mother_r2_panel"
FINAL_MODEL_DIR = ROOT / "models" / "v17_acrhap_mother_r3_outer_refit"

DEFAULT_DESIGN = EVIDENCE / "DESIGN_V17_R3_OUTER_CONFIRMATION.md"
DEFAULT_REGISTRY = EVIDENCE / "registry_v17_r3_outer_confirmation.json"
DEFAULT_OUTER_REFIT_CSV = EVIDENCE / "outer_refit_v17_r3.csv"
DEFAULT_RESULT = EVIDENCE / "outer_result_v17_r3_confirmation.json"
DEFAULT_R0_REGISTRY = EVIDENCE / "registry_v17_r0.json"
DEFAULT_R2F_REGISTRY = EVIDENCE / "registry_v17_r2f_member_closure.json"
DEFAULT_R2F_RESULT = EVIDENCE / "development_result_v17_r2f_member_closure.json"
DEFAULT_R2F_CACHE = EVIDENCE / "development_cache_v17_r2f_panel25.pt"
DEFAULT_R2F_A2 = EVIDENCE / "amendment_v17_r2f_a2_operator_continuity.json"

DEFAULT_PROTOCOL_SOURCE = Path(__file__).resolve()
DEFAULT_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r3.py"
DEFAULT_PROTOCOL_TEST = ROOT.parent / "tests" / "test_uji_v17_mother_r3.py"
DEFAULT_R0_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_protocol.py"
DEFAULT_R0_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r0.py"
DEFAULT_R2F_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2f_protocol.py"
DEFAULT_R2F_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2f.py"
DEFAULT_R2F_A1_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2f_runtime_a1.py"
DEFAULT_R2F_A2_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2f_runtime_a2.py"
DEFAULT_CONSENSUS_SOURCE = ROOT / "fhm_dapp" / "consensus.py"
DEFAULT_SCORE_SOURCE = ROOT / "fhm_dapp" / "coherent_marked_v9.py"
DEFAULT_MODEL_SOURCE = ROOT / "fhm_dapp" / "model.py"
DEFAULT_ANCHOR_SOURCE = ROOT / "fhm_dapp" / "anchor.py"
DEFAULT_TRAIN_SOURCE = ROOT / "fhm_dapp" / "train_anchor_v2.py"
DEFAULT_ENSEMBLE_SOURCE = ROOT / "fhm_dapp" / "projected_ensemble.py"
DEFAULT_DATA_SOURCE = ROOT / "fhm_dapp" / "data.py"
DEFAULT_UJI_SOURCE = ROOT / "fhm_dapp" / "uji_v16_protocol.py"
DEFAULT_ADAPTER_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r0_runtime_a2.py"
DEFAULT_HASH_SOURCE = ROOT / "fhm_dapp" / "experiment_v7.py"


def source_checkpoint_path(member_index: int) -> Path:
    index = int(member_index)
    if not 1 <= index <= PANEL_MEMBER_COUNT:
        raise ValueError("member index is outside the registered panel")
    directory = SOURCE_MODEL_R0 if index <= 5 else SOURCE_MODEL_R2
    return directory / f"member_{index:02d}_seed{MEMBER_SEEDS[index - 1]}.pt"


def final_checkpoint_path(member_index: int) -> Path:
    index = int(member_index)
    if not 1 <= index <= PANEL_MEMBER_COUNT:
        raise ValueError("member index is outside the registered panel")
    return FINAL_MODEL_DIR / f"member_{index:02d}_seed{MEMBER_SEEDS[index - 1]}.pt"


def balanced_member_subsets(E: int) -> tuple[tuple[int, ...], ...]:
    """Return 25 outcome-independent cyclic subsets on zero-based member IDs."""

    count = int(E)
    if not 1 <= count < PANEL_MEMBER_COUNT:
        raise ValueError("R3 repeated realizations require 1 <= E < 25")
    offsets = SUBSET_OFFSET_ORDER[:count]
    subsets = tuple(
        tuple((start + offset) % PANEL_MEMBER_COUNT for offset in offsets)
        for start in range(PANEL_MEMBER_COUNT)
    )
    if any(len(set(subset)) != count for subset in subsets):
        raise RuntimeError("a cyclic R3 subset contains duplicate members")
    appearances = [
        sum(member in subset for subset in subsets)
        for member in range(PANEL_MEMBER_COUNT)
    ]
    if appearances != [count] * PANEL_MEMBER_COUNT:
        raise RuntimeError("R3 member subsets are not exactly balanced")
    return subsets


def _load_self_hashed_json(path: str | Path, field: str) -> dict:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get(field) != _canonical_json_sha256(payload, omit=field):
        raise ValueError(f"self-hash failed for {source}")
    return payload


def _source_manifest() -> dict[str, Path]:
    return {
        "design": DEFAULT_DESIGN,
        "protocol_source": DEFAULT_PROTOCOL_SOURCE,
        "execution_source": DEFAULT_EXECUTION_SOURCE,
        "protocol_test": DEFAULT_PROTOCOL_TEST,
        "r0_protocol_source": DEFAULT_R0_PROTOCOL_SOURCE,
        "r0_execution_source": DEFAULT_R0_EXECUTION_SOURCE,
        "r2f_protocol_source": DEFAULT_R2F_PROTOCOL_SOURCE,
        "r2f_execution_source": DEFAULT_R2F_EXECUTION_SOURCE,
        "r2f_a1_source": DEFAULT_R2F_A1_SOURCE,
        "r2f_a2_source": DEFAULT_R2F_A2_SOURCE,
        "consensus_source": DEFAULT_CONSENSUS_SOURCE,
        "score_source": DEFAULT_SCORE_SOURCE,
        "model_source": DEFAULT_MODEL_SOURCE,
        "anchor_source": DEFAULT_ANCHOR_SOURCE,
        "train_source": DEFAULT_TRAIN_SOURCE,
        "ensemble_source": DEFAULT_ENSEMBLE_SOURCE,
        "data_source": DEFAULT_DATA_SOURCE,
        "uji_source": DEFAULT_UJI_SOURCE,
        "adapter_source": DEFAULT_ADAPTER_SOURCE,
        "hash_source": DEFAULT_HASH_SOURCE,
    }


def _binding(path: str | Path, *, self_hash_field: str | None = None) -> dict:
    source = Path(path).resolve()
    record = {"path": str(source), "file_sha256": _sha256_file(source)}
    if self_hash_field is not None:
        payload = _load_self_hashed_json(source, self_hash_field)
        record[self_hash_field] = payload[self_hash_field]
    return record


def _load_source_panel() -> tuple[list[dict], dict]:
    cache = torch.load(DEFAULT_R2F_CACHE, map_location="cpu", weights_only=True)
    cache_paths = tuple(Path(value).resolve() for value in cache["member_paths"])
    cache_hashes = tuple(str(value) for value in cache["member_sha256"])
    if len(cache_paths) != PANEL_MEMBER_COUNT or len(cache_hashes) != PANEL_MEMBER_COUNT:
        raise ValueError("R2F cache does not contain the complete 25-member panel")
    records = []
    for index in range(1, PANEL_MEMBER_COUNT + 1):
        path = source_checkpoint_path(index).resolve()
        digest = _sha256_file(path)
        if path != cache_paths[index - 1] or digest != cache_hashes[index - 1]:
            raise ValueError(f"R2F cache/source checkpoint mismatch for member {index}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        seed = MEMBER_SEEDS[index - 1]
        if payload.get("member_index") != index or payload.get("member_seed") != seed:
            raise ValueError(f"source member identity changed for member {index}")
        split_contract = payload.get("split_contract", {})
        outer_access = split_contract.get(
            "outer_labels_consumed",
            split_contract.get("outer_test_predictions"),
        )
        official_access = split_contract.get(
            "official_labels_consumed",
            split_contract.get("official_validation_predictions"),
        )
        if outer_access not in (False, 0):
            raise ValueError(f"source member {index} consumed outer labels")
        if official_access not in (False, 0):
            raise ValueError(f"source member {index} consumed official labels")
        epoch = int(payload.get("epoch", 0))
        if epoch <= 0:
            raise ValueError(f"source member {index} has no selected epoch")
        records.append(
            {
                "member_index": index,
                "seed": seed,
                "selected_epoch": epoch,
                "path": str(path),
                "file_sha256": digest,
            }
        )
        del payload
    contract = {
        "path": str(DEFAULT_R2F_CACHE.resolve()),
        "file_sha256": _sha256_file(DEFAULT_R2F_CACHE),
        "member_paths_match": True,
        "member_hashes_match": True,
        "source_members": PANEL_MEMBER_COUNT,
    }
    del cache
    gc.collect()
    return records, contract


def build_registry() -> dict:
    r0 = load_r0_registry(DEFAULT_R0_REGISTRY)
    r2f_registry = load_r2f_registry(DEFAULT_R2F_REGISTRY)
    r2f_result = _load_self_hashed_json(DEFAULT_R2F_RESULT, "result_sha256")
    a2 = _load_self_hashed_json(DEFAULT_R2F_A2, "amendment_sha256")

    decision = r2f_result.get("registered_decision", {})
    selected = decision.get("registered_selection", {})
    expected = CONFIGURATIONS[SELECTED_NAME]
    for field in ("E", "L", "K", "alpha"):
        if selected.get(field) != expected[field]:
            raise ValueError(f"R2F selected {field} changed")
    if decision.get("member_information_saturated") is not True:
        raise ValueError("R2F did not close member information")
    if decision.get("support_information_saturated") is not True:
        raise ValueError("R2F did not close support information")
    if r2f_result.get("outer_test_predictions") != 0:
        raise ValueError("R2F accessed the registered outer test")
    if r2f_result.get("official_validation_predictions") != 0:
        raise ValueError("R2F accessed official validation")
    if r2f_result.get("execution_amendment", {}).get("amendment_sha256") != a2.get(
        "amendment_sha256"
    ):
        raise ValueError("R2F result is not bound to the final A2 execution")

    outer_refit = copy.deepcopy(r0["roles"]["outer_refit"])
    outer_test = copy.deepcopy(r0["roles"]["outer_test"])
    if outer_refit["coordinate_count"] != OUTER_REFIT_ANCHORS:
        raise ValueError("outer-refit anchor count changed")
    if outer_test["row_count"] != OUTER_TEST_ROWS:
        raise ValueError("outer-test row count changed")
    if outer_test["coordinate_count"] != OUTER_TEST_COORDINATES:
        raise ValueError("outer-test coordinate count changed")
    if set(map(tuple, outer_refit["coordinate_keys"])) & set(
        map(tuple, outer_test["coordinate_keys"])
    ):
        raise ValueError("outer-refit and outer-test coordinates overlap")

    source_members, cache_contract = _load_source_panel()
    subset_contract = {}
    for name, configuration in CONFIGURATIONS.items():
        subsets = balanced_member_subsets(configuration["E"])
        subset_contract[name] = {
            "realizations": len(subsets),
            "subsets_zero_based": [list(value) for value in subsets],
            "members_per_realization": configuration["E"],
            "appearances_per_panel_member": configuration["E"],
            "losses_averaged_not_predictive_distributions": True,
        }

    allowed_outputs = {
        "outer_refit_csv": str(DEFAULT_OUTER_REFIT_CSV.resolve()),
        "outer_refit_checkpoints": [
            str(final_checkpoint_path(index).resolve())
            for index in range(1, PANEL_MEMBER_COUNT + 1)
        ],
        "outer_result": str(DEFAULT_RESULT.resolve()),
    }
    flat_outputs = [
        allowed_outputs["outer_refit_csv"],
        *allowed_outputs["outer_refit_checkpoints"],
        allowed_outputs["outer_result"],
    ]
    if len(set(flat_outputs)) != len(flat_outputs):
        raise ValueError("R3 output paths are not unique")
    if any(Path(value).exists() for value in flat_outputs):
        raise FileExistsError("an R3 output already exists before registration")

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_outer_prediction",
        "created_after_r2f_development_outcome": True,
        "selection_authority_for_v17_r2f": False,
        "may_modify_v17_r2f_model": False,
        "outer_test_prediction_authorized": True,
        "official_validation_prediction_authorized": False,
        "outcome_access_at_registration": {
            "r3_outer_test_predictions": 0,
            "r3_official_validation_predictions": 0,
        },
        "scientific_question": (
            "Does the frozen E16 information-saturated mother law improve "
            "coordinate-macro predictive risk over the frozen historical E5 "
            "law on the untouched registered outer test?"
        ),
        "selected_model": {
            **copy.deepcopy(CONFIGURATIONS[SELECTED_NAME]),
            "beta_neural": BETA_NEURAL,
            "beta_prototype": BETA_PROTOTYPE,
            "consensus_epsilon": CONSENSUS_EPSILON,
            "student_t_degrees_freedom": STUDENT_DF,
            "member_weight": "inverse_radial_predictive_variance",
            "joint_law": "q(m,a|x)=w_m(x)c_a(x)",
            "point_action": "exact_mixture_mean",
            "consensus_execution": "deterministic_dense_executable_definition",
            "no_marked_update": True,
            "no_candidate_reranker": True,
            "no_fallback": True,
            "no_building_or_floor_conditioning": True,
        },
        "historical_comparator": {
            **copy.deepcopy(CONFIGURATIONS[HISTORICAL_NAME]),
            "beta_neural": BETA_NEURAL,
            "beta_prototype": BETA_PROTOTYPE,
            "consensus_epsilon": CONSENSUS_EPSILON,
            "student_t_degrees_freedom": STUDENT_DF,
            "member_weight": "inverse_radial_predictive_variance",
            "joint_law": "q(m,a|x)=w_m(x)c_a(x)",
            "point_action": "exact_mixture_mean",
            "same_outer_refit_member_panel": True,
            "historical_6p0977446758_m_value_is_context_only": True,
        },
        "training_contract": {
            "panel_members": PANEL_MEMBER_COUNT,
            "source_members": source_members,
            "outer_refit_rule": (
                "reinitialize with identical seed and fit exact source-selected "
                "epoch count on every registered outer_refit row"
            ),
            "epoch_reselection": False,
            "epoch_scaling_for_refit_size": False,
            "only_training_data_change": "selection_refit_to_outer_refit",
            "training_algorithm": "exact_V17_R0_refit",
            "outer_test_consumed_in_training": False,
        },
        "member_subset_estimand": {
            "panel_members": PANEL_MEMBER_COUNT,
            "cyclic_generator": CYCLIC_GENERATOR,
            "offset_order": list(SUBSET_OFFSET_ORDER),
            "configurations": subset_contract,
            "favorable_seed_selection": False,
            "panel_is_experimental_replication_not_predictor_size": True,
        },
        "estimand": {
            "primary": "continuous_2d_energy_score_coordinate_macro_m",
            "point_endpoint": "exact_mixture_mean_coordinate_macro_mae_m",
            "aggregation_order": (
                "average 25 realization losses within scan; average scans "
                "within exact physical coordinate; average 187 coordinates"
            ),
            "difference_orientation": "selected_E16_minus_historical_E5",
            "predictive_draws_per_stream": PREDICTIVE_DRAWS_PER_STREAM,
            "predictive_seed": PREDICTIVE_SEED,
            "common_random_numbers": True,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "familywise_endpoints": ["energy_score_2d_m", "exact_mixture_mean_mae_m"],
            "confirmation_rule": "simultaneous_upper_bound_below_zero",
            "outer_outcome_cannot_change_model": True,
        },
        "runtime": {
            "component_batch_size": COMPONENT_BATCH_SIZE,
            "single_outer_access_command": True,
            "query_path_argument": False,
            "configuration_argument": False,
            "result_overwrite": False,
        },
        "data_contract": {
            "train_data": r0["train_data"],
            "train_data_sha256": r0["train_data_sha256"],
            "outer_refit": outer_refit,
            "outer_test": outer_test,
            "outer_refit_outer_test_coordinate_overlap": 0,
            "official_validation_path": r0["official_data_audit_only"],
            "official_validation_sha256": r0["official_data_sha256"],
            "official_validation_sealed": True,
        },
        "bindings": {
            "r0_registry": {
                **_binding(DEFAULT_R0_REGISTRY),
                "registry_sha256": r0["registry_sha256"],
            },
            "r2f_registry": {
                **_binding(DEFAULT_R2F_REGISTRY),
                "registry_sha256": r2f_registry["registry_sha256"],
            },
            "r2f_result": _binding(DEFAULT_R2F_RESULT, self_hash_field="result_sha256"),
            "r2f_a2": _binding(DEFAULT_R2F_A2, self_hash_field="amendment_sha256"),
            "r2f_cache": cache_contract,
        },
        "sources": {
            name: {"path": str(path.resolve()), "file_sha256": _sha256_file(path)}
            for name, path in _source_manifest().items()
        },
        "allowed_outputs": allowed_outputs,
    }
    registry["registry_sha256"] = _canonical_json_sha256(registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("registry_sha256") != _canonical_json_sha256(
        payload, omit="registry_sha256"
    ):
        raise ValueError("R3 registry self-hash failed")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("R3 protocol version changed")
    if payload.get("status") != "frozen_before_outer_prediction":
        raise ValueError("R3 registry was not frozen before outer prediction")
    if payload.get("selection_authority_for_v17_r2f") is not False:
        raise ValueError("R3 cannot select or modify R2F")
    if payload.get("may_modify_v17_r2f_model") is not False:
        raise ValueError("R3 cannot modify the selected model")
    if payload.get("outer_test_prediction_authorized") is not True:
        raise ValueError("R3 registry does not authorize its one outer access")
    if payload.get("official_validation_prediction_authorized") is not False:
        raise ValueError("R3 must keep official validation sealed")
    for name, record in payload["sources"].items():
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"R3 source binding changed: {name}")
    for name, record in payload["bindings"].items():
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"R3 artifact binding changed: {name}")
    for name, configuration in CONFIGURATIONS.items():
        registered = payload["member_subset_estimand"]["configurations"][name]
        expected = [list(value) for value in balanced_member_subsets(configuration["E"])]
        if registered["subsets_zero_based"] != expected:
            raise ValueError(f"R3 subset contract changed: {name}")
    return payload


def register(output: str | Path = DEFAULT_REGISTRY) -> dict:
    destination = Path(output).resolve()
    if destination != DEFAULT_REGISTRY.resolve():
        raise ValueError("R3 registry path changed")
    payload = build_registry()
    _write_json_new(destination, payload)
    return {
        "registry": str(destination),
        "registry_sha256": payload["registry_sha256"],
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("register")
    freeze.add_argument("--output", default=str(DEFAULT_REGISTRY))
    check = subparsers.add_parser("check")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "register":
        result = register(args.output)
    else:
        registry = load_registry(args.registry)
        result = {
            "registry": str(Path(args.registry).resolve()),
            "registry_sha256": registry["registry_sha256"],
            "outer_test_predictions": registry["outcome_access_at_registration"][
                "r3_outer_test_predictions"
            ],
            "official_validation_predictions": registry[
                "outcome_access_at_registration"
            ]["r3_official_validation_predictions"],
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
