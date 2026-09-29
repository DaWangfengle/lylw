"""Register V17-R2C: joint member/support/consensus information saturation."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r1_protocol import (
    load_registry as load_r1_registry,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2b_protocol import (
    load_registry as load_r2b_registry,
)


PROTOCOL_VERSION = "uji_v17_r2c_joint_information_saturation_v1_20260822"
MEMBER_GRID = tuple(range(1, 11))
FINAL_MINIMUM_MEMBERS = 2
MEMBER_CEILING_WITH_TWO_LATER_VALUES = 8
SUPPORT_L_GRID = (12, 24, 48, 96)
SUPPORT_K_GRID = (8, 16, 24, 48)
CONSENSUS_ALPHA_GRID = (0.0, 0.25, 0.5, 1.0)
CONSENSUS_EPSILON = 1e-6
SUBSET_OFFSET_ORDER = (0, 1, 3, 7, 2, 5, 8, 4, 6, 9)
HISTORICAL_CONFIGURATION = (5, 24, 24, 0.25)
ENERGY_EQUIVALENCE_MARGIN_M = 0.05
MAE_NONINFERIORITY_MARGIN_M = 0.10
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 17822
CONFIDENCE = 0.95
SCORING_BATCH_SIZE = 128
SUPPORT_AUDIT_REPRODUCTION_TOLERANCE = 1e-5

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v17_acrhap_mother"
DEFAULT_DESIGN = EVIDENCE / "DESIGN_V17_R2_IDENTIFIABLE_FRONT.md"
DEFAULT_R2B_REGISTRY = EVIDENCE / "registry_v17_r2b_atomic.json"
DEFAULT_R2B_RESULT = EVIDENCE / "development_result_v17_r2b_atomic.json"
DEFAULT_R2B_CACHE = EVIDENCE / "development_cache_v17_r2b_atomic.pt"
DEFAULT_R1_REGISTRY = EVIDENCE / "registry_v17_r1_backend.json"
DEFAULT_R1_RESULT = EVIDENCE / "development_result_v17_r1_backend.json"
DEFAULT_RESULT = EVIDENCE / "development_result_v17_r2c_joint.json"
DEFAULT_REGISTRY = EVIDENCE / "registry_v17_r2c_joint.json"

DEFAULT_PROTOCOL_SOURCE = Path(__file__).resolve()
DEFAULT_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2c.py"
DEFAULT_PROTOCOL_TEST = ROOT.parent / "tests" / "test_uji_v17_mother_r2c.py"
DEFAULT_R2B_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2b_protocol.py"
DEFAULT_R2B_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2b.py"
DEFAULT_CONSENSUS_SOURCE = ROOT / "fhm_dapp" / "consensus.py"
DEFAULT_SCORE_SOURCE = ROOT / "fhm_dapp" / "coherent_marked_v9.py"
DEFAULT_HASH_SOURCE = ROOT / "fhm_dapp" / "experiment_v7.py"


def alpha_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def variant_name(E: int, L: int, K: int, alpha: float) -> str:
    return f"E{int(E):02d}_L{int(L):03d}_K{int(K):03d}_A{alpha_token(alpha)}"


def configurations() -> tuple[tuple[int, int, int, float], ...]:
    values = []
    for E in MEMBER_GRID:
        alphas = (0.25,) if E == 1 else CONSENSUS_ALPHA_GRID
        for L in SUPPORT_L_GRID:
            for K in SUPPORT_K_GRID:
                if K > L:
                    continue
                for alpha in alphas:
                    values.append((E, L, K, alpha))
    return tuple(values)


CONFIGURATIONS = configurations()
VARIANTS = tuple(variant_name(*value) for value in CONFIGURATIONS)
HISTORICAL_VARIANT = variant_name(*HISTORICAL_CONFIGURATION)


def balanced_member_subsets(E: int) -> tuple[tuple[int, ...], ...]:
    count = int(E)
    if count not in MEMBER_GRID:
        raise ValueError("E is outside the R2C member grid")
    if count == 10:
        return (tuple(range(10)),)
    offsets = SUBSET_OFFSET_ORDER[:count]
    return tuple(
        tuple((start + offset) % 10 for offset in offsets)
        for start in range(10)
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
        "r2b_protocol_source": DEFAULT_R2B_PROTOCOL_SOURCE,
        "r2b_execution_source": DEFAULT_R2B_EXECUTION_SOURCE,
        "consensus_source": DEFAULT_CONSENSUS_SOURCE,
        "score_source": DEFAULT_SCORE_SOURCE,
        "hash_source": DEFAULT_HASH_SOURCE,
    }


def _unlabelled_support_audit(cache: Mapping[str, object], beta_n: float, beta_p: float) -> dict:
    probability = torch.softmax(
        float(beta_n) * cache["member_neural_logits"]
        - float(beta_p) * cache["prototype_distance"].unsqueeze(0),
        dim=2,
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
    return audit


def register_protocol(
    *,
    r2b_registry_path: str | Path = DEFAULT_R2B_REGISTRY,
    r2b_result_path: str | Path = DEFAULT_R2B_RESULT,
    r2b_cache_path: str | Path = DEFAULT_R2B_CACHE,
    r1_registry_path: str | Path = DEFAULT_R1_REGISTRY,
    r1_result_path: str | Path = DEFAULT_R1_RESULT,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if DEFAULT_RESULT.exists():
        raise FileExistsError(DEFAULT_RESULT)

    r2b_registry_file = Path(r2b_registry_path).resolve()
    r2b_result_file = Path(r2b_result_path).resolve()
    r2b_cache_file = Path(r2b_cache_path).resolve()
    r1_registry_file = Path(r1_registry_path).resolve()
    r1_result_file = Path(r1_result_path).resolve()
    r2b = load_r2b_registry(r2b_registry_file)
    r2b_result = _load_self_hashed(r2b_result_file, "result_sha256")
    r1 = load_r1_registry(r1_registry_file)
    r1_result = _load_self_hashed(r1_result_file, "result_sha256")
    if r2b_result.get("registry_sha256") != r2b["registry_sha256"]:
        raise ValueError("R2B result and registry differ")
    if r2b_result.get("cache_sha256") != _sha256_file(r2b_cache_file):
        raise ValueError("R2B cache binding changed")
    selected = r2b_result.get("registered_selection", {})
    if selected.get("variant") != "bn_x1__bp_x1":
        raise ValueError("R2C requires the registered R2B historical coefficient selection")
    if selected.get("selected_on_registered_grid_boundary") is not False:
        raise ValueError("R2B coefficient selection lies on a grid boundary")
    for payload, label in ((r2b_result, "R2B"), (r1_result, "R1")):
        if payload.get("outer_test_predictions") != 0:
            raise ValueError(f"{label} outer-test access forbids R2C")
        if payload.get("official_validation_predictions") != 0:
            raise ValueError(f"{label} official access forbids R2C")

    cache = torch.load(r2b_cache_file, map_location="cpu", weights_only=True)
    if cache.get("registry_sha256") != r2b["registry_sha256"]:
        raise ValueError("R2B cache and registry differ")
    support_audit = _unlabelled_support_audit(
        cache, selected["beta_neural"], selected["beta_prototype"]
    )
    sources = _source_manifest()
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"R2C source is missing: {missing}")

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_r2c_joint_information_saturation_registered",
        "created_after_r2b_development_outcome": True,
        "selection_authority_for_r0_r1_or_r2b": False,
        "may_modify_r0_r1_or_r2b": False,
        "outer_test_prediction_authorized": False,
        "official_validation_prediction_authorized": False,
        "scientific_question": (
            "what joint member/support/consensus information budget is sufficient "
            "after freezing the identifiable atomic probability law"
        ),
        "frozen_atomic_law": {
            "beta_neural": selected["beta_neural"],
            "beta_prototype": selected["beta_prototype"],
            "source_variant": selected["variant"],
            "decoder": "R1-retained exact continuous-mixture mean D0",
            "marked_update": "excluded by R1",
        },
        "joint_grid": {
            "E": list(MEMBER_GRID),
            "L": list(SUPPORT_L_GRID),
            "K": list(SUPPORT_K_GRID),
            "constraint": "K<=L",
            "consensus_alpha": list(CONSENSUS_ALPHA_GRID),
            "E1_alpha_rule": "alpha=0.25 only because alpha is unidentified for one member",
            "consensus_epsilon": CONSENSUS_EPSILON,
            "configurations": [
                {"name": variant_name(E, L, K, alpha), "E": E, "L": L, "K": K, "alpha": alpha}
                for E, L, K, alpha in CONFIGURATIONS
            ],
        },
        "member_subset_design": {
            "member_indices": list(range(10)),
            "offset_order": list(SUBSET_OFFSET_ORDER),
            "rule": (
                "for E<10 use all ten cyclic translations of the first E offsets; "
                "for E=10 use the full panel once"
            ),
            "nested_across_E": True,
            "each_member_appears_exactly_E_times_for_E_less_than_10": True,
            "member_subsets": {
                str(E): [list(value) for value in balanced_member_subsets(E)]
                for E in MEMBER_GRID
            },
            "favorable_seed_selection": False,
        },
        "fixed_inherited_mechanisms": {
            "member_weight": "inverse_radial_variance_normalized_within_subset",
            "member_top_L": "exact_reverse_KL_projection_and_renormalization",
            "consensus": "epsilon_completed_alpha_divergence_barycenter",
            "consensus_top_K": "exact_reverse_KL_projection_and_renormalization",
            "final_joint_law": "member_weight_times_consensus_anchor_probability",
            "student_t_degrees_freedom": 4.0,
        },
        "unlabelled_support_audit_at_registration": {
            "uses_target_coordinates_or_losses": False,
            "hard_mass_threshold": None,
            "evaluation_reproduction_tolerance": SUPPORT_AUDIT_REPRODUCTION_TOLERANCE,
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
            "historical_balanced_reference": HISTORICAL_VARIANT,
            "mae_noninferiority_margin_m": MAE_NONINFERIORITY_MARGIN_M,
            "energy_equivalence_margin_m": ENERGY_EQUIVALENCE_MARGIN_M,
            "simultaneous_method": "studentized_max_t_paired_coordinate_bootstrap",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "final_requires_E_at_least": FINAL_MINIMUM_MEMBERS,
            "final_requires_E_at_most": MEMBER_CEILING_WITH_TWO_LATER_VALUES,
            "information_budget": "E*L+K",
            "information_budget_interpretation": (
                "E*L retained member-anchor probability entries enter consensus, "
                "and K shared consensus-anchor probabilities define the factorized "
                "joint law; this is a representation budget, not a runtime claim"
            ),
            "rule": (
                "retain configurations passing simultaneous 0.10 m MAE noninferiority; "
                "form a simultaneous 0.05 m energy-equivalence platform around the "
                "eligible global energy minimum; among platform points with 2<=E<=8, "
                "choose minimum E*L+K, then E, L, K, mean energy, distance of alpha "
                "from 0.25, and registered order"
            ),
            "boundary_rule": (
                "if no platform point has E<=8, or the provisional selection uses "
                "maximum L or maximum K, report unsaturated and require a later "
                "registered extension; never call the current ceiling optimal"
            ),
        },
        "parity_gate": {
            "configuration": HISTORICAL_VARIANT,
            "member_subset": [0, 1, 2, 3, 4],
            "reference": "R1 P0D0 coordinate-macro MAE from its frozen row error vector",
            "absolute_coordinate_macro_mae_tolerance_m": 1e-4,
            "rowwise_difference_reporting": (
                "report maximum and mean absolute row-error differences as numerical "
                "diagnostics; do not require bitwise row parity across the R1 one-batch "
                "and R2B full-support batched cache-construction paths"
            ),
        },
        "bindings": {
            "r2b_registry": {
                "path": str(r2b_registry_file),
                "file_sha256": _sha256_file(r2b_registry_file),
                "registry_sha256": r2b["registry_sha256"],
            },
            "r2b_result": {
                "path": str(r2b_result_file),
                "file_sha256": _sha256_file(r2b_result_file),
                "result_sha256": r2b_result["result_sha256"],
            },
            "r2b_cache": {
                "path": str(r2b_cache_file),
                "file_sha256": _sha256_file(r2b_cache_file),
            },
            "r1_registry": {
                "path": str(r1_registry_file),
                "file_sha256": _sha256_file(r1_registry_file),
                "registry_sha256": r1["registry_sha256"],
            },
            "r1_result": {
                "path": str(r1_result_file),
                "file_sha256": _sha256_file(r1_result_file),
                "result_sha256": r1_result["result_sha256"],
            },
        },
        "allowed_outputs": {"development_result": str(DEFAULT_RESULT.resolve())},
        "runtime": {"scoring_batch_size": SCORING_BATCH_SIZE},
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in sources.items()
        },
        "outcome_access_at_registration": {
            "r2b_development_outcome_known": True,
            "r2c_target_based_scores_computed": 0,
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
        raise ValueError("not a V17-R2C registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(_without_hash(registry)):
        raise ValueError("V17-R2C registry self-hash failed")
    for flag in ("outer_test_prediction_authorized", "official_validation_prediction_authorized"):
        if registry.get(flag) is not False:
            raise ValueError(f"V17-R2C illegally authorizes {flag}")
    for record in registry["sources"].values():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17-R2C source changed: {record['path']}")
    for record in registry["bindings"].values():
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"V17-R2C binding changed: {record['path']}")
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--r2b-registry", default=str(DEFAULT_R2B_REGISTRY))
    register.add_argument("--r2b-result", default=str(DEFAULT_R2B_RESULT))
    register.add_argument("--r2b-cache", default=str(DEFAULT_R2B_CACHE))
    register.add_argument("--r1-registry", default=str(DEFAULT_R1_REGISTRY))
    register.add_argument("--r1-result", default=str(DEFAULT_R1_RESULT))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))
    check = sub.add_parser("check")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "register":
        payload = register_protocol(
            r2b_registry_path=args.r2b_registry,
            r2b_result_path=args.r2b_result,
            r2b_cache_path=args.r2b_cache,
            r1_registry_path=args.r1_registry,
            r1_result_path=args.r1_result,
            output=args.output,
        )
    else:
        payload = load_registry(args.registry)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
