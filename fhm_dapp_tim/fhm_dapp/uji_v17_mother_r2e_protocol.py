"""Register V17-R2E: jointly close the member and finite-support boundaries."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2d import load_cache as load_r2d_cache
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2d_protocol import (
    load_registry as load_r2d_registry,
)


PROTOCOL_VERSION = "uji_v17_r2e_information_closure_v1_20260822"
ORIGINAL_NUMERICAL_BLOCK = 10
BASE_MEMBER_COUNT = 15
PANEL_MEMBER_COUNT = 20
ADDITIONAL_MEMBER_INDICES = tuple(range(16, 21))
ADDITIONAL_MEMBER_SEEDS = tuple(range(17316, 17321))
MEMBER_GRID = tuple(range(10, 21))
FINAL_MEMBER_CEILING_WITH_TWO_LATER_VALUES = 18
ANCHOR_COUNT = 595
SUPPORT_L_GRID = (48, 96, 192, 384, 595)
SUPPORT_K_GRID = (24, 48, 96, 192, 384, 595)
CONSENSUS_ALPHA = 1.0
CONSENSUS_EPSILON = 1e-6
CYCLIC_GENERATOR = 7
SUBSET_OFFSET_ORDER = tuple(
    (CYCLIC_GENERATOR * index) % PANEL_MEMBER_COUNT
    for index in range(PANEL_MEMBER_COUNT)
)
HISTORICAL_REFERENCE = "E05_L024_K024_A0p25"
R2D_CONTINUITY_REFERENCE = "E15_L192_K048_A1"
ENERGY_EQUIVALENCE_MARGIN_M = 0.05
MAE_NONINFERIORITY_MARGIN_M = 0.10
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 17825
CONFIDENCE = 0.95
SCORING_BATCH_SIZE = 128
R2D_PARITY_TOLERANCE_M = 1e-5
DENSE_CONSENSUS_FROM_L = 384

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v17_acrhap_mother"
MODEL_DIR = ROOT / "models" / "v17_acrhap_mother_r2_panel"
DEFAULT_DESIGN = EVIDENCE / "DESIGN_V17_R2E_INFORMATION_CLOSURE.md"
DEFAULT_R2D_REGISTRY = EVIDENCE / "registry_v17_r2d_boundary.json"
DEFAULT_R2D_CACHE = EVIDENCE / "development_cache_v17_r2d_panel15.pt"
DEFAULT_R2D_RESULT = EVIDENCE / "development_result_v17_r2d_boundary.json"
DEFAULT_CACHE = EVIDENCE / "development_cache_v17_r2e_panel20.pt"
DEFAULT_RESULT = EVIDENCE / "development_result_v17_r2e_closure.json"
DEFAULT_REGISTRY = EVIDENCE / "registry_v17_r2e_closure.json"

DEFAULT_PROTOCOL_SOURCE = Path(__file__).resolve()
DEFAULT_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2e.py"
DEFAULT_PROTOCOL_TEST = ROOT.parent / "tests" / "test_uji_v17_mother_r2e.py"
DEFAULT_R2D_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2d_protocol.py"
DEFAULT_R2D_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2d.py"
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
        raise ValueError("R2E creates only members 16 through 20")
    seed = ADDITIONAL_MEMBER_SEEDS[index - ADDITIONAL_MEMBER_INDICES[0]]
    return MODEL_DIR / f"member_{index:02d}_seed{seed}.pt"


def variant_name(E: int, L: int, K: int) -> str:
    return f"E{int(E):02d}_L{int(L):03d}_K{int(K):03d}_A1"


def configurations() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (E, L, K)
        for E in MEMBER_GRID
        for L in SUPPORT_L_GRID
        for K in SUPPORT_K_GRID
        if K <= L
    )


CONFIGURATIONS = configurations()
VARIANTS = tuple(variant_name(*value) for value in CONFIGURATIONS)


def balanced_member_subsets(E: int) -> tuple[tuple[int, ...], ...]:
    count = int(E)
    if count not in MEMBER_GRID:
        raise ValueError("E is outside the R2E member grid")
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
        "r2d_protocol_source": DEFAULT_R2D_PROTOCOL_SOURCE,
        "r2d_execution_source": DEFAULT_R2D_EXECUTION_SOURCE,
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


def _stagewise_r2d_evidence(result: Mapping[str, object]) -> dict:
    metrics = result["metrics"]
    decision = result["registered_decision"]
    if decision.get("registered_selection") is not None:
        raise ValueError("R2D already selected an operating point")
    if decision.get("requires_member_panel_extension") is not True:
        raise ValueError("R2D did not require member-panel extension")
    if decision.get("eligible_global_energy_minimum") != R2D_CONTINUITY_REFERENCE:
        raise ValueError("R2D boundary minimum changed")
    if decision.get("energy_equivalence_platform") != [R2D_CONTINUITY_REFERENCE]:
        raise ValueError("R2D energy platform changed")

    removed_support_profiles = {}
    for E in range(10, 16):
        prefix = f"E{E:02d}_"
        records = {
            name: record
            for name, record in metrics.items()
            if name.startswith(prefix)
        }
        overall = min(
            record["energy_score_2d_m"]["coordinate_macro_m"]
            for record in records.values()
        )
        best_k16 = min(
            record["energy_score_2d_m"]["coordinate_macro_m"]
            for name, record in records.items()
            if "_K016_" in name
        )
        best_l24 = min(
            record["energy_score_2d_m"]["coordinate_macro_m"]
            for name, record in records.items()
            if f"E{E:02d}_L024_" in name
        )
        gaps = {
            "best_K16_minus_within_E_best_energy_m": best_k16 - overall,
            "best_L24_minus_within_E_best_energy_m": best_l24 - overall,
        }
        if min(gaps.values()) <= ENERGY_EQUIVALENCE_MARGIN_M:
            raise ValueError(f"R2D does not license low-support pruning at E={E}")
        removed_support_profiles[str(E)] = gaps

    low_member_profiles = {}
    global_energy = metrics[R2D_CONTINUITY_REFERENCE]["energy_score_2d_m"][
        "coordinate_macro_m"
    ]
    for E in range(5, 10):
        profile = decision["profiled_best_by_E"][str(E)]
        gap = profile["energy_score_coordinate_macro_m"] - global_energy
        if gap <= ENERGY_EQUIVALENCE_MARGIN_M:
            raise ValueError(f"R2D does not license removing E={E}")
        low_member_profiles[str(E)] = {
            "variant": profile["variant"],
            "energy_minus_r2d_global_m": gap,
        }

    comparisons = result["simultaneous_inference"][
        "energy_candidate_minus_eligible_global_minimum"
    ]["comparisons"]
    near_boundary = {}
    for name in (
        "E13_L096_K048_A1",
        "E14_L096_K048_A1",
        "E15_L096_K048_A1",
    ):
        near_boundary[name] = {
            "mean_difference_m": comparisons[name]["mean_difference_m"],
            "simultaneous_upper_bound_m": comparisons[name][
                "simultaneous_upper_bound_m"
            ],
        }
    return {
        "r2d_global_boundary_variant": R2D_CONTINUITY_REFERENCE,
        "r2d_platform_was_boundary_only": True,
        "removed_low_member_profiles": low_member_profiles,
        "removed_low_support_profiles": removed_support_profiles,
        "near_member_boundary": near_boundary,
    }


def _unlabelled_support_audit(
    cache: Mapping[str, object], beta_n: float, beta_p: float
) -> dict:
    neural = cache["member_neural_logits"]
    distance = cache["prototype_distance"]
    blocks = (neural[:ORIGINAL_NUMERICAL_BLOCK], neural[ORIGINAL_NUMERICAL_BLOCK:])
    probability = torch.cat(
        [
            torch.softmax(
                float(beta_n) * block
                - float(beta_p) * distance.unsqueeze(0),
                dim=2,
            )
            for block in blocks
        ],
        dim=0,
    )
    audit = {}
    for L in SUPPORT_L_GRID:
        mass = torch.topk(probability, L, dim=2).values.sum(dim=2)
        audit[str(L)] = {
            "mean_retained_mass": float(mass.mean()),
            "q10_retained_mass": float(torch.quantile(mass, 0.10)),
            "median_retained_mass": float(torch.quantile(mass, 0.50)),
            "q90_retained_mass": float(torch.quantile(mass, 0.90)),
            "mean_reverse_kl_nats": float((-mass.clamp_min(1e-30).log()).mean()),
        }
    if not math.isclose(audit[str(ANCHOR_COUNT)]["mean_retained_mass"], 1.0, abs_tol=1e-6):
        raise ValueError("full registered support does not retain unit mass")
    return audit


def register_protocol(
    *,
    r2d_registry_path: str | Path = DEFAULT_R2D_REGISTRY,
    r2d_cache_path: str | Path = DEFAULT_R2D_CACHE,
    r2d_result_path: str | Path = DEFAULT_R2D_RESULT,
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
        raise FileExistsError(f"R2E output already exists: {existing}")

    r2d_file = Path(r2d_registry_path).resolve()
    r2d_cache_file = Path(r2d_cache_path).resolve()
    r2d_result_file = Path(r2d_result_path).resolve()
    r2d = load_r2d_registry(r2d_file)
    r2d_result = _load_self_hashed(r2d_result_file, "result_sha256")
    if r2d_result.get("registry_sha256") != r2d["registry_sha256"]:
        raise ValueError("R2D result and registry differ")
    if r2d_result.get("cache_sha256") != _sha256_file(r2d_cache_file):
        raise ValueError("R2D result and cache differ")
    for key in ("outer_test_predictions", "official_validation_predictions"):
        if r2d_result.get(key) != 0:
            raise ValueError(f"R2D {key} forbids R2E")

    old_cache = load_r2d_cache(r2d_cache_file, registry=r2d)
    old_paths = tuple(Path(value).resolve() for value in old_cache["member_paths"])
    old_hashes = tuple(old_cache["member_sha256"])
    if len(old_paths) != BASE_MEMBER_COUNT:
        raise ValueError("R2E requires the complete fifteen-member R2D cache")
    if tuple(_sha256_file(path) for path in old_paths) != old_hashes:
        raise ValueError("an existing panel checkpoint changed")
    if old_cache["member_neural_logits"].shape[2] != ANCHOR_COUNT:
        raise ValueError("registered anchor count changed")

    r2c_binding = r2d["bindings"]["r2c_result"]
    r2c_result_file = Path(r2c_binding["path"]).resolve()
    if _sha256_file(r2c_result_file) != r2c_binding["file_sha256"]:
        raise ValueError("R2C historical reference changed")
    stagewise = _stagewise_r2d_evidence(r2d_result)
    beta_n = r2d["frozen_atomic_and_consensus_law"]["beta_neural"]
    beta_p = r2d["frozen_atomic_and_consensus_law"]["beta_prototype"]
    support_audit = _unlabelled_support_audit(old_cache, beta_n, beta_p)
    sources = _source_manifest()
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"R2E source is missing: {missing}")

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_r2e_information_closure_registered",
        "created_after_r2d_development_outcome": True,
        "selection_authority_for_r0_through_r2d": False,
        "may_modify_r0_through_r2d": False,
        "development_prediction_authorized_after_member_completion": True,
        "outer_test_prediction_authorized": False,
        "official_validation_prediction_authorized": False,
        "scientific_question": (
            "what minimum joint member/support information budget is practically "
            "equivalent to the best law when the finite 595-anchor support is closed"
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
        "frozen_atomic_and_consensus_law": {
            "beta_neural": beta_n,
            "beta_prototype": beta_p,
            "consensus_alpha": CONSENSUS_ALPHA,
            "consensus_epsilon": CONSENSUS_EPSILON,
            "member_weight": "inverse_radial_variance_normalized_within_subset",
            "member_top_L": "exact_reverse_KL_projection_and_renormalization",
            "consensus_top_K": "exact_reverse_KL_projection_and_renormalization",
            "final_joint_law": "member_weight_times_consensus_anchor_probability",
            "decoder": "exact_continuous_mixture_mean",
            "numerical_block_contract": [
                "members_1_through_10_in_original_R2C_softmax_block",
                "members_11_through_15_in_original_R2D_extension_block",
                "members_16_through_20_in_new_R2E_extension_block",
            ],
            "dense_consensus_from_L": DENSE_CONSENSUS_FROM_L,
            "dense_consensus_is_exact_executable_definition_not_a_new_operator": True,
        },
        "stagewise_r2d_evidence": stagewise,
        "joint_closure_grid": {
            "E": list(MEMBER_GRID),
            "L": list(SUPPORT_L_GRID),
            "K": list(SUPPORT_K_GRID),
            "constraint": "K<=L",
            "alpha": CONSENSUS_ALPHA,
            "configurations": [
                {"name": variant_name(E, L, K), "E": E, "L": L, "K": K}
                for E, L, K in CONFIGURATIONS
            ],
            "removed_after_r2d": {"E": list(range(5, 10)), "L": [24], "K": [16]},
            "added_boundary": {
                "members": list(ADDITIONAL_MEMBER_INDICES),
                "L": [384, 595],
                "K": [96, 192, 384, 595],
            },
            "finite_support_closed": True,
            "exact_no_truncation_endpoint": {"L": ANCHOR_COUNT, "K": ANCHOR_COUNT},
        },
        "member_subset_design": {
            "member_indices": list(range(PANEL_MEMBER_COUNT)),
            "cyclic_generator": CYCLIC_GENERATOR,
            "offset_order": list(SUBSET_OFFSET_ORDER),
            "rule": (
                "for E<20 use all twenty cyclic translations of the first E offsets; "
                "for E=20 use the full panel once"
            ),
            "nested_across_E": True,
            "each_member_appears_exactly_E_times_for_E_less_than_20": True,
            "member_subsets": {
                str(E): [list(value) for value in balanced_member_subsets(E)]
                for E in MEMBER_GRID
            },
            "favorable_seed_selection": False,
        },
        "unlabelled_support_audit_on_existing_fifteen_member_panel": {
            "uses_target_coordinates_or_losses": False,
            "hard_mass_threshold": None,
            "diagnostic_only": True,
            "diagnostic": support_audit,
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
            "fixed_mae_reference_source": "frozen R2C coordinate-group vector",
            "mae_noninferiority_margin_m": MAE_NONINFERIORITY_MARGIN_M,
            "energy_equivalence_margin_m": ENERGY_EQUIVALENCE_MARGIN_M,
            "simultaneous_method": "studentized_max_t_paired_coordinate_bootstrap",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "final_requires_E_at_least": min(MEMBER_GRID),
            "final_requires_E_at_most": FINAL_MEMBER_CEILING_WITH_TWO_LATER_VALUES,
            "information_budget": "E*L+K",
            "rule": (
                "retain configurations passing simultaneous 0.10 m MAE noninferiority; "
                "form a simultaneous 0.05 m energy-equivalence platform around the "
                "eligible global energy minimum; among platform points with E<=18, "
                "choose minimum E*L+K, then E, L, K, mean energy, and registered order"
            ),
            "member_boundary_rule": (
                "if no platform point has E<=18, report unresolved and forbid outer test"
            ),
            "support_boundary_rule": (
                "the 595-anchor exact endpoint closes support; no larger L or K exists"
            ),
        },
        "continuity_gate": {
            "configuration": R2D_CONTINUITY_REFERENCE,
            "members": "members 1 through 15",
            "reference": "R2D coordinate-group energy and MAE vectors",
            "maximum_absolute_coordinate_macro_endpoint_tolerance_m": R2D_PARITY_TOLERANCE_M,
        },
        "data_contract": copy.deepcopy(r2d["data_contract"]),
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
            "r2d_registry": {
                "path": str(r2d_file),
                "file_sha256": _sha256_file(r2d_file),
                "registry_sha256": r2d["registry_sha256"],
            },
            "r2d_cache": {
                "path": str(r2d_cache_file),
                "file_sha256": _sha256_file(r2d_cache_file),
            },
            "r2d_result": {
                "path": str(r2d_result_file),
                "file_sha256": _sha256_file(r2d_result_file),
                "result_sha256": r2d_result["result_sha256"],
            },
            "r2c_result": {
                "path": str(r2c_result_file),
                "file_sha256": _sha256_file(r2c_result_file),
                "result_sha256": r2c_binding["result_sha256"],
            },
        },
        "runtime": {"scoring_batch_size": SCORING_BATCH_SIZE},
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in sources.items()
        },
        "outcome_access_at_registration": {
            "r2d_development_outcome_known": True,
            "r2e_development_target_scores_computed": 0,
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
        raise ValueError("not a V17-R2E registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(
        _without_hash(registry)
    ):
        raise ValueError("V17-R2E registry self-hash failed")
    for flag in ("outer_test_prediction_authorized", "official_validation_prediction_authorized"):
        if registry.get(flag) is not False:
            raise ValueError(f"V17-R2E illegally authorizes {flag}")
    for record in registry["sources"].values():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17-R2E source changed: {record['path']}")
    for record in registry["bindings"].values():
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"V17-R2E binding changed: {record['path']}")
    for record in registry["current_panel"]:
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17-R2E current member changed: {record['path']}")
    if _sha256_file(registry["data_contract"]["train_data"]) != registry[
        "data_contract"
    ]["train_data_sha256"]:
        raise ValueError("V17-R2E training data changed")
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--r2d-registry", default=str(DEFAULT_R2D_REGISTRY))
    register.add_argument("--r2d-cache", default=str(DEFAULT_R2D_CACHE))
    register.add_argument("--r2d-result", default=str(DEFAULT_R2D_RESULT))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))
    check = sub.add_parser("check")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "register":
        payload = register_protocol(
            r2d_registry_path=args.r2d_registry,
            r2d_cache_path=args.r2d_cache,
            r2d_result_path=args.r2d_result,
            output=args.output,
        )
    else:
        payload = load_registry(args.registry)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

