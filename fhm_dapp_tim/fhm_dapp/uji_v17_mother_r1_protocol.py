"""Register the V17-R1 frozen-front backend identification experiment."""

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
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_protocol import (
    load_registry as load_r0_registry,
)


PROTOCOL_VERSION = "uji_v17_r1_backend_identification_v1_20260821"
VARIANTS = ("P0D0", "P0D1", "N0D0", "N0D1", "N1D0", "N1D1")
POSTERIOR_VARIANTS = ("P0", "N0", "N1")
MEMBER_COUNT = 5
TOP_L = 24
TOP_K = 24
CONSENSUS_ALPHA = 0.25
PROXIMAL_ALPHA = 0.25
LAMBDA_VALUE = 0.50
QUADRATURE_ORDER = 5
OFFSET_BOUNDS = (-0.20, 0.20)
LOG_GAIN_BOUNDS = (-0.25, 0.25)
PROFILE_STEPS = 10
PROFILE_LEARNING_RATE = 0.20
PROFILE_OFFSET_WEIGHT = 0.10
PROFILE_LOG_GAIN_WEIGHT = 0.10
STUDENT_DF = 4.0
PREDICTIVE_DRAWS_PER_STREAM = 64
PREDICTIVE_SEED = 17621
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 17622
R0_PARITY_TOLERANCE_M = 5e-5
MAE_NONINFERIORITY_MARGIN_M = 0.10
CACHE_BATCH_SIZE = 4096
SCORING_BATCH_SIZE = 256
MARGINAL_BATCH_SIZE = 128

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v17_acrhap_mother"
DEFAULT_DESIGN = EVIDENCE / "DESIGN_V17_R1_BACKEND_IDENTIFICATION.md"
DEFAULT_R0_REGISTRY = EVIDENCE / "registry_v17_r0.json"
DEFAULT_R0_RESULT = EVIDENCE / "development_result_v17_r0.json"
DEFAULT_R0_DIAGNOSIS = EVIDENCE / "diagnosis_v17_r0_gate.json"
DEFAULT_A2_AMENDMENT = EVIDENCE / "amendment_v17_r0_a2_coordinate_contract.json"
DEFAULT_REGISTRY = EVIDENCE / "registry_v17_r1_backend.json"
DEFAULT_CACHE = EVIDENCE / "development_cache_v17_r1_backend.pt"
DEFAULT_RESULT = EVIDENCE / "development_result_v17_r1_backend.json"

DEFAULT_PROTOCOL_SOURCE = Path(__file__).resolve()
DEFAULT_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r1.py"
DEFAULT_PROTOCOL_TEST = ROOT.parent / "tests" / "test_uji_v17_mother_r1.py"
DEFAULT_A2_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r0_runtime_a2.py"
DEFAULT_CANDIDATE_SOURCE = ROOT / "fhm_dapp" / "candidates.py"
DEFAULT_CONSENSUS_SOURCE = ROOT / "fhm_dapp" / "consensus.py"
DEFAULT_DAPP_SOURCE = ROOT / "fhm_dapp" / "dapp.py"
DEFAULT_PROJECTED_SOURCE = ROOT / "fhm_dapp" / "projected_ensemble.py"
DEFAULT_COHERENT_SOURCE = ROOT / "fhm_dapp" / "coherent_marked_v9.py"
DEFAULT_ORBIT_SOURCE = ROOT / "fhm_dapp" / "marked_orbit_v10.py"


def _without_hash(payload: Mapping[str, object]) -> dict:
    value = copy.deepcopy(dict(payload))
    value.pop("registry_sha256", None)
    return value


def _load_self_hashed_json(path: Path, hash_field: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get(hash_field) != _canonical_json_sha256(value, omit=hash_field):
        raise ValueError(f"self-hash failed for {path}")
    return value


def _source_manifest() -> dict[str, Path]:
    return {
        "design": DEFAULT_DESIGN,
        "protocol_source": DEFAULT_PROTOCOL_SOURCE,
        "execution_source": DEFAULT_EXECUTION_SOURCE,
        "protocol_test": DEFAULT_PROTOCOL_TEST,
        "runtime_coordinate_adapter": DEFAULT_A2_SOURCE,
        "candidate_source": DEFAULT_CANDIDATE_SOURCE,
        "consensus_source": DEFAULT_CONSENSUS_SOURCE,
        "dapp_source": DEFAULT_DAPP_SOURCE,
        "projected_ensemble_source": DEFAULT_PROJECTED_SOURCE,
        "predictive_action_source": DEFAULT_COHERENT_SOURCE,
        "quadrature_source": DEFAULT_ORBIT_SOURCE,
    }


def register_protocol(
    *,
    r0_registry_path: str | Path = DEFAULT_R0_REGISTRY,
    r0_result_path: str | Path = DEFAULT_R0_RESULT,
    r0_diagnosis_path: str | Path = DEFAULT_R0_DIAGNOSIS,
    a2_amendment_path: str | Path = DEFAULT_A2_AMENDMENT,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    cache_path = DEFAULT_CACHE.resolve()
    result_path = DEFAULT_RESULT.resolve()
    existing = [str(path) for path in (cache_path, result_path) if path.exists()]
    if existing:
        raise FileExistsError(f"R1 output already exists: {existing}")

    r0_registry_file = Path(r0_registry_path).resolve()
    r0_result_file = Path(r0_result_path).resolve()
    r0_diagnosis_file = Path(r0_diagnosis_path).resolve()
    a2_amendment_file = Path(a2_amendment_path).resolve()
    r0_registry = load_r0_registry(r0_registry_file)
    r0_result = _load_self_hashed_json(r0_result_file, "result_sha256")
    r0_diagnosis = _load_self_hashed_json(
        r0_diagnosis_file, "diagnostic_sha256"
    )
    a2_amendment = _load_self_hashed_json(
        a2_amendment_file, "amendment_sha256"
    )
    if r0_result.get("registry_sha256") != r0_registry["registry_sha256"]:
        raise ValueError("R0 result and registry differ")
    if r0_result.get("query_role") != "development":
        raise ValueError("R1 requires the completed R0 development result")
    if r0_result["reconstruction_gate"].get("passed") is not False:
        raise ValueError("R1 is registered only after the failed R0 gate")
    if r0_result.get("outer_test_predictions") != 0:
        raise ValueError("R0 outer-test access forbids R1 registration")
    if r0_result.get("official_validation_predictions") != 0:
        raise ValueError("R0 official access forbids R1 registration")
    if r0_diagnosis.get("interpretation_authority") != (
        "descriptive_only_not_model_selection"
    ):
        raise ValueError("R0 diagnosis authority changed")
    if a2_amendment.get("scientific_change") is not False:
        raise ValueError("R0 coordinate adapter is not scientifically neutral")

    sources = _source_manifest()
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"R1 source is missing: {missing}")
    checkpoints = [
        Path(value).resolve()
        for value in r0_registry["allowed_outputs"]["r0_checkpoints"]
    ]
    if len(checkpoints) != MEMBER_COUNT or any(
        not path.exists() for path in checkpoints
    ):
        raise FileNotFoundError("R1 requires all five frozen R0 members")

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_r1_backend_identification_registered",
        "scientific_status": (
            "iterative UJI development registered after the R0 outcome; "
            "not independent confirmation"
        ),
        "created_after_r0_development_outcome": True,
        "r0_gate_known_failed": True,
        "selection_authority_for_r0": False,
        "may_modify_r0": False,
        "r1_development_prediction_authorized": True,
        "outer_test_prediction_authorized": False,
        "official_validation_prediction_authorized": False,
        "data_contract": {
            "selection_refit_csv": r0_registry["allowed_outputs"]["selection_refit_csv"],
            "selection_refit_csv_sha256": _sha256_file(
                r0_registry["allowed_outputs"]["selection_refit_csv"]
            ),
            "development_csv": r0_registry["allowed_outputs"]["development_csv"],
            "development_csv_sha256": _sha256_file(
                r0_registry["allowed_outputs"]["development_csv"]
            ),
            "development_role": r0_registry["roles"]["development"],
            "sealed_roles": ["outer_test", "official_validation"],
        },
        "frozen_checkpoints": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in checkpoints
        ],
        "fixed_model_contract": {
            "members": MEMBER_COUNT,
            "L": TOP_L,
            "K": TOP_K,
            "neural_scale": 0.75,
            "prototype_distance_weight": 8.0,
            "posterior_temperature": 0.75,
            "consensus_alpha": CONSENSUS_ALPHA,
            "consensus_epsilon": 1e-6,
            "proximal_alpha": PROXIMAL_ALPHA,
            "lambda": LAMBDA_VALUE,
            "member_weight_mode": "inverse_scale",
            "measurement_pool_mode": "shared",
            "ap_reliability_power": 0.0,
            "visible_factor": 1.0,
            "invisible_factor": 1.0,
            "hurdle_score": "balanced_log",
            "hurdle_distribution": "student_t",
            "student_df": STUDENT_DF,
            "candidate_building_conditioning": False,
        },
        "factorial_contract": {
            "variants": list(VARIANTS),
            "posterior_variants": list(POSTERIOR_VARIANTS),
            "N0": {
                "nuisance": "ten_step_alpha_proximal_envelope_profile",
                "steps": PROFILE_STEPS,
                "learning_rate": PROFILE_LEARNING_RATE,
                "offset_bounds": list(OFFSET_BOUNDS),
                "log_gain_bounds": list(LOG_GAIN_BOUNDS),
                "offset_weight": PROFILE_OFFSET_WEIGHT,
                "log_gain_weight": PROFILE_LOG_GAIN_WEIGHT,
            },
            "N1": {
                "nuisance": "uniform_affine_loss_marginalization",
                "quadrature_order_per_axis": QUADRATURE_ORDER,
                "nodes": QUADRATURE_ORDER ** 2,
                "offset_bounds": list(OFFSET_BOUNDS),
                "log_gain_bounds": list(LOG_GAIN_BOUNDS),
                "effective_loss": "-lambda*log E_h exp(-loss/lambda)",
            },
            "D0": "exact_continuous_mixture_mean",
            "D1": "deterministic_Weiszfeld_spatial_median_of_128_CRN_draws",
            "P0": "unchanged_five_member_alpha_consensus_prior",
        },
        "predictive_contract": {
            "draws_per_stream": PREDICTIVE_DRAWS_PER_STREAM,
            "streams": 2,
            "random_seed": PREDICTIVE_SEED,
            "energy_score": "unbiased_two_stream_monte_carlo",
            "same_randomness_for_every_variant": True,
        },
        "estimand": {
            "primary_probability_metric": "coordinate_macro_energy_score_2d_m",
            "primary_action_metric": "coordinate_macro_euclidean_mae_2d_m",
            "indivisible_unit": "exact physical coordinate",
            "row_metrics": "descriptive_only",
        },
        "inference_contract": {
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "familywise_confidence": 0.95,
            "common_coordinate_group_resamples": True,
            "r0_profile_mean_parity_tolerance_m": R0_PARITY_TOLERANCE_M,
            "mae_noninferiority_margin_m": MAE_NONINFERIORITY_MARGIN_M,
            "multiplicity_families": {
                "energy": ["N0-P0", "N1-P0", "N1-N0"],
                "mae": [
                    "P0D1-P0D0",
                    "N0D0-P0D0",
                    "N0D1-P0D0",
                    "N1D0-P0D0",
                    "N1D1-P0D0",
                    "N0D1-N0D0",
                    "N1D1-N1D0",
                    "N1-N0_action_main_effect",
                    "D1-D0_decoder_main_effect",
                ],
            },
            "cache_batch_size": CACHE_BATCH_SIZE,
            "marginal_batch_size": MARGINAL_BATCH_SIZE,
            "scoring_batch_size": SCORING_BATCH_SIZE,
        },
        "decision_gates": {
            "marked_inclusion": (
                "a marked posterior must improve energy versus P0 under the "
                "simultaneous upper bound and have a non-inferior action MAE"
            ),
            "marginalization_claim": (
                "simultaneous upper bound of N1-minus-N0 energy below zero"
            ),
            "spatial_median_claim": (
                "simultaneous upper bound of D1-minus-D0 MAE below zero"
            ),
            "posterior_selection_rule": (
                "retain P0 if neither marked law is eligible; use the sole "
                "eligible marked law if exactly one is eligible; if both "
                "are eligible, replace historical N0 by N1 only when the "
                "simultaneous N1-minus-N0 energy upper bound is below zero"
            ),
            "decoder_selection_rule": (
                "replace D0 by D1 only when both the simultaneous D1-minus-D0 "
                "decoder-main-effect MAE upper bound and the selected-"
                "posterior-specific D1-minus-D0 upper bound are below zero"
            ),
            "failure_action": "stop_without_rescue_component",
        },
        "allowed_outputs": {
            "development_cache": str(cache_path),
            "development_result": str(result_path),
        },
        "bindings": {
            "r0_registry": {
                "path": str(r0_registry_file),
                "file_sha256": _sha256_file(r0_registry_file),
                "registry_sha256": r0_registry["registry_sha256"],
            },
            "r0_result": {
                "path": str(r0_result_file),
                "file_sha256": _sha256_file(r0_result_file),
                "result_sha256": r0_result["result_sha256"],
            },
            "r0_diagnosis": {
                "path": str(r0_diagnosis_file),
                "file_sha256": _sha256_file(r0_diagnosis_file),
                "diagnostic_sha256": r0_diagnosis["diagnostic_sha256"],
            },
            "r0_a2_amendment": {
                "path": str(a2_amendment_file),
                "file_sha256": _sha256_file(a2_amendment_file),
                "amendment_sha256": a2_amendment["amendment_sha256"],
            },
        },
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in sources.items()
        },
        "outcome_access_at_registration": {
            "r0_development_outcome_known": True,
            "r1_development_predictions": 0,
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
        raise ValueError("not a V17 R1 registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(
        _without_hash(registry)
    ):
        raise ValueError("V17 R1 registry self-hash failed")
    if registry.get("outer_test_prediction_authorized") is not False:
        raise ValueError("V17 R1 illegally authorizes outer prediction")
    if registry.get("official_validation_prediction_authorized") is not False:
        raise ValueError("V17 R1 illegally authorizes official prediction")
    for record in registry["sources"].values():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17 R1 source changed: {record['path']}")
    for field in ("selection_refit_csv", "development_csv"):
        path = registry["data_contract"][field]
        if _sha256_file(path) != registry["data_contract"][f"{field}_sha256"]:
            raise ValueError(f"V17 R1 data changed: {field}")
    for record in registry["frozen_checkpoints"]:
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("V17 R1 checkpoint changed")
    for field in ("r0_registry", "r0_result", "r0_diagnosis", "r0_a2_amendment"):
        record = registry["bindings"][field]
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"V17 R1 input binding changed: {field}")
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--r0-registry", default=str(DEFAULT_R0_REGISTRY))
    register.add_argument("--r0-result", default=str(DEFAULT_R0_RESULT))
    register.add_argument("--r0-diagnosis", default=str(DEFAULT_R0_DIAGNOSIS))
    register.add_argument("--a2-amendment", default=str(DEFAULT_A2_AMENDMENT))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))
    check = subparsers.add_parser("check")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "register":
        payload = register_protocol(
            r0_registry_path=args.r0_registry,
            r0_result_path=args.r0_result,
            r0_diagnosis_path=args.r0_diagnosis,
            a2_amendment_path=args.a2_amendment,
            output=args.output,
        )
    else:
        payload = load_registry(args.registry)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
