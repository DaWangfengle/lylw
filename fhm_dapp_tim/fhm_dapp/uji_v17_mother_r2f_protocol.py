"""Register V17-R2F: close member information after R2E closed support."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2d_protocol import (
    load_registry as load_r2d_registry,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2e import load_cache as load_r2e_cache
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2e_protocol import (
    load_registry as load_r2e_registry,
)


PROTOCOL_VERSION = "uji_v17_r2f_member_closure_v1_20260822"
BASE_MEMBER_COUNT = 20
PANEL_MEMBER_COUNT = 25
ADDITIONAL_MEMBER_INDICES = tuple(range(21, 26))
ADDITIONAL_MEMBER_SEEDS = tuple(range(17321, 17326))
MEMBER_GRID = tuple(range(16, 26))
FINAL_MEMBER_CEILING_WITH_TWO_LATER_VALUES = 23
ANCHOR_COUNT = 595
SUPPORT_L_GRID = (192, 595)
SUPPORT_K = 192
CONSENSUS_ALPHA = 1.0
CONSENSUS_EPSILON = 1e-6
CYCLIC_GENERATOR = 7
SUBSET_OFFSET_ORDER = tuple(
    (CYCLIC_GENERATOR * index) % PANEL_MEMBER_COUNT
    for index in range(PANEL_MEMBER_COUNT)
)
HISTORICAL_REFERENCE = "E05_L024_K024_A0p25"
R2E_GLOBAL_REFERENCE = "E20_L595_K192_A1"
R2E_COMPACT_REFERENCE = "E20_L192_K192_A1"
ENERGY_EQUIVALENCE_MARGIN_M = 0.05
MAE_NONINFERIORITY_MARGIN_M = 0.10
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 17826
CONFIDENCE = 0.95
SCORING_BATCH_SIZE = 128
R2E_PARITY_TOLERANCE_M = 1e-5

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v17_acrhap_mother"
MODEL_DIR = ROOT / "models" / "v17_acrhap_mother_r2_panel"
DEFAULT_DESIGN = EVIDENCE / "DESIGN_V17_R2F_MEMBER_CLOSURE.md"
DEFAULT_R2E_REGISTRY = EVIDENCE / "registry_v17_r2e_closure.json"
DEFAULT_R2E_CACHE = EVIDENCE / "development_cache_v17_r2e_panel20.pt"
DEFAULT_R2E_RESULT = EVIDENCE / "development_result_v17_r2e_closure.json"
DEFAULT_CACHE = EVIDENCE / "development_cache_v17_r2f_panel25.pt"
DEFAULT_RESULT = EVIDENCE / "development_result_v17_r2f_member_closure.json"
DEFAULT_REGISTRY = EVIDENCE / "registry_v17_r2f_member_closure.json"

DEFAULT_PROTOCOL_SOURCE = Path(__file__).resolve()
DEFAULT_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2f.py"
DEFAULT_PROTOCOL_TEST = ROOT.parent / "tests" / "test_uji_v17_mother_r2f.py"
DEFAULT_R2E_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2e_protocol.py"
DEFAULT_R2E_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2e.py"
DEFAULT_R2D_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2d_protocol.py"
DEFAULT_R0_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r0.py"
DEFAULT_R0_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_protocol.py"
DEFAULT_ANCHOR_SOURCE = ROOT / "fhm_dapp" / "anchor.py"
DEFAULT_MODEL_SOURCE = ROOT / "fhm_dapp" / "model.py"
DEFAULT_TRAIN_SOURCE = ROOT / "fhm_dapp" / "train_anchor_v2.py"
DEFAULT_ENSEMBLE_SOURCE = ROOT / "fhm_dapp" / "projected_ensemble.py"
DEFAULT_DATA_SOURCE = ROOT / "fhm_dapp" / "data.py"
DEFAULT_UJI_SOURCE = ROOT / "fhm_dapp" / "uji_v16_protocol.py"
DEFAULT_ADAPTER_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r0_runtime_a2.py"
DEFAULT_R2C_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2c.py"
DEFAULT_CONSENSUS_SOURCE = ROOT / "fhm_dapp" / "consensus.py"
DEFAULT_SCORE_SOURCE = ROOT / "fhm_dapp" / "coherent_marked_v9.py"
DEFAULT_HASH_SOURCE = ROOT / "fhm_dapp" / "experiment_v7.py"


def checkpoint_path(member_index: int) -> Path:
    index = int(member_index)
    if index not in ADDITIONAL_MEMBER_INDICES:
        raise ValueError("R2F creates only members 21 through 25")
    seed = ADDITIONAL_MEMBER_SEEDS[index - ADDITIONAL_MEMBER_INDICES[0]]
    return MODEL_DIR / f"member_{index:02d}_seed{seed}.pt"


def variant_name(E: int, L: int) -> str:
    return f"E{int(E):02d}_L{int(L):03d}_K{SUPPORT_K:03d}_A1"


def configurations() -> tuple[tuple[int, int], ...]:
    return tuple((E, L) for E in MEMBER_GRID for L in SUPPORT_L_GRID)


CONFIGURATIONS = configurations()
VARIANTS = tuple(variant_name(*value) for value in CONFIGURATIONS)


def balanced_member_subsets(E: int) -> tuple[tuple[int, ...], ...]:
    count = int(E)
    if count not in MEMBER_GRID:
        raise ValueError("E is outside the R2F member grid")
    if count == PANEL_MEMBER_COUNT:
        return (tuple(range(PANEL_MEMBER_COUNT)),)
    offsets = SUBSET_OFFSET_ORDER[:count]
    return tuple(
        tuple((start + offset) % PANEL_MEMBER_COUNT for offset in offsets)
        for start in range(PANEL_MEMBER_COUNT)
    )


def _without_hash(payload: Mapping[str, object]) -> dict:
    value = copy.deepcopy(dict(payload))
    value.pop("registry_sha256", None)
    return value


def _load_self_hashed(path: str | Path, field: str) -> dict:
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
        "r2e_protocol_source": DEFAULT_R2E_PROTOCOL_SOURCE,
        "r2e_execution_source": DEFAULT_R2E_EXECUTION_SOURCE,
        "r2d_protocol_source": DEFAULT_R2D_PROTOCOL_SOURCE,
        "r0_execution_source": DEFAULT_R0_EXECUTION_SOURCE,
        "r0_protocol_source": DEFAULT_R0_PROTOCOL_SOURCE,
        "anchor_source": DEFAULT_ANCHOR_SOURCE,
        "model_source": DEFAULT_MODEL_SOURCE,
        "training_source": DEFAULT_TRAIN_SOURCE,
        "ensemble_source": DEFAULT_ENSEMBLE_SOURCE,
        "data_source": DEFAULT_DATA_SOURCE,
        "uji_source": DEFAULT_UJI_SOURCE,
        "adapter_source": DEFAULT_ADAPTER_SOURCE,
        "r2c_execution_source": DEFAULT_R2C_EXECUTION_SOURCE,
        "consensus_source": DEFAULT_CONSENSUS_SOURCE,
        "score_source": DEFAULT_SCORE_SOURCE,
        "hash_source": DEFAULT_HASH_SOURCE,
    }


def _stagewise_r2e_evidence(result: Mapping[str, object]) -> dict:
    decision = result["registered_decision"]
    if decision.get("registered_selection") is not None:
        raise ValueError("R2E already selected an operating point")
    if decision.get("requires_member_panel_extension") is not True:
        raise ValueError("R2E did not require member-panel extension")
    if decision.get("support_boundary_closed_by_exact_endpoint") is not True:
        raise ValueError("R2E did not close finite support")
    if decision.get("requires_support_grid_extension") is not False:
        raise ValueError("R2E unexpectedly requests support extension")
    if decision.get("eligible_global_energy_minimum") != R2E_GLOBAL_REFERENCE:
        raise ValueError("R2E global reference changed")
    platform = tuple(decision.get("energy_equivalence_platform", ()))
    expected = {R2E_GLOBAL_REFERENCE, R2E_COMPACT_REFERENCE}
    if set(platform) != expected or len(platform) != len(expected):
        raise ValueError("R2E compact/exact energy platform changed")
    if any("_K192_" not in name for name in platform):
        raise ValueError("R2E does not license fixing K=192")
    exact = result.get("exact_support_endpoint", {})
    if exact != {"anchors": 595, "K": 595, "L": 595, "truncation": False}:
        raise ValueError("R2E exact-support endpoint changed")

    energy_comparisons = result["simultaneous_inference"][
        "energy_candidate_minus_eligible_global_minimum"
    ]["comparisons"]
    removed_low_members = {}
    for E in range(10, 16):
        profile = decision["profiled_best_by_E"][str(E)]
        name = profile["variant"]
        upper = energy_comparisons[name]["simultaneous_upper_bound_m"]
        if upper <= ENERGY_EQUIVALENCE_MARGIN_M:
            raise ValueError(f"R2E does not license removing E={E}")
        removed_low_members[str(E)] = {
            "variant": name,
            "simultaneous_upper_bound_m": upper,
        }
    return {
        "r2e_global_reference": R2E_GLOBAL_REFERENCE,
        "r2e_energy_platform": list(platform),
        "removed_E_10_through_15": removed_low_members,
        "fixed_K": SUPPORT_K,
        "retained_L": list(SUPPORT_L_GRID),
        "finite_support_closed": True,
    }


def register_protocol(
    *,
    r2e_registry_path: str | Path = DEFAULT_R2E_REGISTRY,
    r2e_cache_path: str | Path = DEFAULT_R2E_CACHE,
    r2e_result_path: str | Path = DEFAULT_R2E_RESULT,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    reserved = [
        *(checkpoint_path(index).resolve() for index in ADDITIONAL_MEMBER_INDICES),
        DEFAULT_CACHE.resolve(),
        DEFAULT_RESULT.resolve(),
    ]
    existing = [str(path) for path in reserved if path.exists()]
    if existing:
        raise FileExistsError(f"R2F output already exists: {existing}")

    r2e_file = Path(r2e_registry_path).resolve()
    r2e_cache_file = Path(r2e_cache_path).resolve()
    r2e_result_file = Path(r2e_result_path).resolve()
    r2e_registry = load_r2e_registry(r2e_file)
    r2e_result = _load_self_hashed(r2e_result_file, "result_sha256")
    if r2e_result.get("registry_sha256") != r2e_registry["registry_sha256"]:
        raise ValueError("R2E result and registry differ")
    if r2e_result.get("cache_sha256") != _sha256_file(r2e_cache_file):
        raise ValueError("R2E result and cache differ")
    for key in ("outer_test_predictions", "official_validation_predictions"):
        if r2e_result.get(key) != 0:
            raise ValueError(f"R2E {key} forbids R2F")
    old_cache = load_r2e_cache(r2e_cache_file, registry=r2e_registry)
    old_paths = tuple(Path(value).resolve() for value in old_cache["member_paths"])
    old_hashes = tuple(old_cache["member_sha256"])
    if len(old_paths) != BASE_MEMBER_COUNT:
        raise ValueError("R2F requires the complete twenty-member R2E cache")
    if tuple(_sha256_file(path) for path in old_paths) != old_hashes:
        raise ValueError("an existing panel checkpoint changed")
    if old_cache["member_neural_logits"].shape[2] != ANCHOR_COUNT:
        raise ValueError("registered anchor count changed")
    stagewise = _stagewise_r2e_evidence(r2e_result)

    r2d_registry = load_r2d_registry(r2e_registry["bindings"]["r2d_registry"]["path"])
    r0_binding = r2d_registry["bindings"]["r0_registry"]
    r2c_binding = r2e_registry["bindings"]["r2c_result"]
    sources = _source_manifest()
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"R2F source is missing: {missing}")

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_r2f_member_closure_registered",
        "created_after_r2e_development_outcome": True,
        "selection_authority_for_r0_through_r2e": False,
        "may_modify_r0_through_r2e": False,
        "outer_test_prediction_authorized": False,
        "official_validation_prediction_authorized": False,
        "scientific_question": (
            "what minimum exchangeable-member information budget is practically "
            "equivalent after finite anchor support has been closed"
        ),
        "panel_contract": {
            "base_members": BASE_MEMBER_COUNT,
            "target_members": PANEL_MEMBER_COUNT,
            "additional_member_indices": list(ADDITIONAL_MEMBER_INDICES),
            "additional_member_seeds": list(ADDITIONAL_MEMBER_SEEDS),
            "only_exchangeable_factor_changed": "random_seed",
            "training_algorithm": "exact_V17_R0_member_fit_early_stop_selection_refit",
            "member_selection_from_development": False,
        },
        "frozen_predictive_law": {
            "beta_neural": r2e_registry["frozen_atomic_and_consensus_law"]["beta_neural"],
            "beta_prototype": r2e_registry["frozen_atomic_and_consensus_law"]["beta_prototype"],
            "consensus_alpha": CONSENSUS_ALPHA,
            "consensus_epsilon": CONSENSUS_EPSILON,
            "member_weight": "inverse_radial_variance_normalized_within_subset",
            "member_top_L": "exact_reverse_KL_projection_and_renormalization",
            "consensus_top_K": "exact_reverse_KL_projection_and_renormalization",
            "final_joint_law": "member_weight_times_consensus_anchor_probability",
            "decoder": "exact_continuous_mixture_mean",
            "student_t_components": "frozen_member_specific_location_and_scale",
            "new_mechanism": False,
        },
        "stagewise_r2e_evidence": stagewise,
        "member_closure_grid": {
            "E": list(MEMBER_GRID),
            "L": list(SUPPORT_L_GRID),
            "K": SUPPORT_K,
            "alpha": CONSENSUS_ALPHA,
            "configurations": [
                {"name": variant_name(E, L), "E": E, "L": L, "K": SUPPORT_K}
                for E, L in CONFIGURATIONS
            ],
            "removed_after_r2e": {"E": list(range(10, 16)), "K": [24, 48, 96, 384, 595]},
            "finite_support_sentinel_L": ANCHOR_COUNT,
        },
        "member_subset_design": {
            "member_indices": list(range(PANEL_MEMBER_COUNT)),
            "cyclic_generator": CYCLIC_GENERATOR,
            "offset_order": list(SUBSET_OFFSET_ORDER),
            "nested_across_E": True,
            "each_member_appears_exactly_E_times_for_E_less_than_25": True,
            "member_subsets": {
                str(E): [list(value) for value in balanced_member_subsets(E)]
                for E in MEMBER_GRID
            },
            "favorable_seed_selection": False,
        },
        "estimand": {
            "primary": "continuous_2d_energy_score_coordinate_macro_m",
            "point_guard": "exact_mixture_mean_coordinate_macro_mae_m",
            "aggregation_order": (
                "average registered balanced member subsets within scan; average "
                "scans within exact coordinate; average coordinates"
            ),
            "common_random_numbers": "reuse the frozen R2B two 64-draw streams",
        },
        "selection_rule": {
            "fixed_mae_reference": HISTORICAL_REFERENCE,
            "mae_noninferiority_margin_m": MAE_NONINFERIORITY_MARGIN_M,
            "energy_equivalence_margin_m": ENERGY_EQUIVALENCE_MARGIN_M,
            "simultaneous_method": "studentized_max_t_paired_coordinate_bootstrap",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "final_requires_E_at_most": FINAL_MEMBER_CEILING_WITH_TWO_LATER_VALUES,
            "information_budget": "E*L+K",
            "member_boundary_rule": (
                "if no platform point has E<=23, report unresolved and forbid outer test"
            ),
            "support_boundary_rule": "595 anchors is exact and cannot be extended",
        },
        "continuity_gate": {
            "configurations": [R2E_COMPACT_REFERENCE, R2E_GLOBAL_REFERENCE],
            "members": "members 1 through 20",
            "reference": "R2E coordinate-group energy and MAE vectors",
            "maximum_absolute_coordinate_macro_endpoint_tolerance_m": R2E_PARITY_TOLERANCE_M,
        },
        "data_contract": copy.deepcopy(r2e_registry["data_contract"]),
        "current_panel": [
            {"member_index": index, "path": str(path), "sha256": sha}
            for index, (path, sha) in enumerate(zip(old_paths, old_hashes), start=1)
        ],
        "allowed_outputs": {
            "additional_checkpoints": [
                str(checkpoint_path(index).resolve())
                for index in ADDITIONAL_MEMBER_INDICES
            ],
            "development_cache": str(DEFAULT_CACHE.resolve()),
            "development_result": str(DEFAULT_RESULT.resolve()),
        },
        "bindings": {
            "r2e_registry": {
                "path": str(r2e_file),
                "file_sha256": _sha256_file(r2e_file),
                "registry_sha256": r2e_registry["registry_sha256"],
            },
            "r2e_cache": {"path": str(r2e_cache_file), "file_sha256": _sha256_file(r2e_cache_file)},
            "r2e_result": {
                "path": str(r2e_result_file),
                "file_sha256": _sha256_file(r2e_result_file),
                "result_sha256": r2e_result["result_sha256"],
            },
            "r0_registry": copy.deepcopy(r0_binding),
            "r2c_result": copy.deepcopy(r2c_binding),
        },
        "runtime": {"scoring_batch_size": SCORING_BATCH_SIZE},
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in sources.items()
        },
        "outcome_access_at_registration": {
            "r2e_development_outcome_known": True,
            "r2f_development_target_scores_computed": 0,
            "outer_test_predictions": 0,
            "official_validation_predictions": 0,
        },
    }
    registry["registry_sha256"] = _canonical_json_sha256(registry)
    _write_json_new(destination, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    source = Path(path).resolve()
    registry = json.loads(source.read_text(encoding="utf-8"))
    if registry.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V17-R2F registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(_without_hash(registry)):
        raise ValueError("V17-R2F registry self-hash failed")
    for flag in ("outer_test_prediction_authorized", "official_validation_prediction_authorized"):
        if registry.get(flag) is not False:
            raise ValueError(f"V17-R2F illegally authorizes {flag}")
    for record in registry["sources"].values():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17-R2F source changed: {record['path']}")
    for record in registry["bindings"].values():
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"V17-R2F binding changed: {record['path']}")
    for record in registry["current_panel"]:
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17-R2F current member changed: {record['path']}")
    if _sha256_file(registry["data_contract"]["train_data"]) != registry["data_contract"]["train_data_sha256"]:
        raise ValueError("V17-R2F training data changed")
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--r2e-registry", default=str(DEFAULT_R2E_REGISTRY))
    register.add_argument("--r2e-cache", default=str(DEFAULT_R2E_CACHE))
    register.add_argument("--r2e-result", default=str(DEFAULT_R2E_RESULT))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))
    check = sub.add_parser("check")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "register":
        payload = register_protocol(
            r2e_registry_path=args.r2e_registry,
            r2e_cache_path=args.r2e_cache,
            r2e_result_path=args.r2e_result,
            output=args.output,
        )
    else:
        payload = load_registry(args.registry)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

