"""Register V17-R2B: calibrate the identifiable atomic retrieval law."""

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
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r1_protocol import (
    load_registry as load_r1_registry,
)
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2a_protocol import (
    load_registry as load_r2a_registry,
)


PROTOCOL_VERSION = "uji_v17_r2b_identifiable_atomic_law_v1_20260822"
MEMBER_COUNT = 10
ANCHOR_COUNT = 595
HISTORICAL_BETA_NEURAL = 1.0
HISTORICAL_BETA_PROTOTYPE = 32.0 / 3.0
LOG_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
BETA_GRID = tuple(
    (HISTORICAL_BETA_NEURAL * neural_multiplier,
     HISTORICAL_BETA_PROTOTYPE * prototype_multiplier)
    for neural_multiplier in LOG_MULTIPLIERS
    for prototype_multiplier in LOG_MULTIPLIERS
)
STUDENT_DF = 4.0
PREDICTIVE_DRAWS_PER_STREAM = 64
PREDICTIVE_SEED = 17721
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 17722
MAE_NONINFERIORITY_MARGIN_M = 0.10
COMPONENT_BATCH_SIZE = 64
SCORING_BATCH_SIZE = 128

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v17_acrhap_mother"
DEFAULT_DESIGN = EVIDENCE / "DESIGN_V17_R2_IDENTIFIABLE_FRONT.md"
DEFAULT_R2A_REGISTRY = EVIDENCE / "registry_v17_r2a_panel.json"
DEFAULT_R1_REGISTRY = EVIDENCE / "registry_v17_r1_backend.json"
DEFAULT_R1_RESULT = EVIDENCE / "development_result_v17_r1_backend.json"
DEFAULT_CACHE = EVIDENCE / "development_cache_v17_r2b_atomic.pt"
DEFAULT_RESULT = EVIDENCE / "development_result_v17_r2b_atomic.json"
DEFAULT_REGISTRY = EVIDENCE / "registry_v17_r2b_atomic.json"

DEFAULT_PROTOCOL_SOURCE = Path(__file__).resolve()
DEFAULT_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2b.py"
DEFAULT_PROTOCOL_TEST = ROOT.parent / "tests" / "test_uji_v17_mother_r2b.py"
DEFAULT_R2A_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2a_protocol.py"
DEFAULT_R2A_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2a.py"
DEFAULT_MODEL_SOURCE = ROOT / "fhm_dapp" / "model.py"
DEFAULT_ENSEMBLE_SOURCE = ROOT / "fhm_dapp" / "projected_ensemble.py"
DEFAULT_SCORE_SOURCE = ROOT / "fhm_dapp" / "coherent_marked_v9.py"
DEFAULT_ADAPTER_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r0_runtime_a2.py"
DEFAULT_DATA_SOURCE = ROOT / "fhm_dapp" / "data.py"
DEFAULT_UJI_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v16_protocol.py"
DEFAULT_HASH_SOURCE = ROOT / "fhm_dapp" / "experiment_v7.py"


def beta_name(beta_neural: float, beta_prototype: float) -> str:
    neural_multiplier = float(beta_neural) / HISTORICAL_BETA_NEURAL
    prototype_multiplier = float(beta_prototype) / HISTORICAL_BETA_PROTOTYPE
    return f"bn_x{neural_multiplier:g}__bp_x{prototype_multiplier:g}"


HISTORICAL_VARIANT = beta_name(
    HISTORICAL_BETA_NEURAL, HISTORICAL_BETA_PROTOTYPE
)
VARIANTS = tuple(beta_name(*value) for value in BETA_GRID)


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
        "r2a_protocol_source": DEFAULT_R2A_PROTOCOL_SOURCE,
        "r2a_execution_source": DEFAULT_R2A_EXECUTION_SOURCE,
        "model_source": DEFAULT_MODEL_SOURCE,
        "ensemble_source": DEFAULT_ENSEMBLE_SOURCE,
        "score_source": DEFAULT_SCORE_SOURCE,
        "adapter_source": DEFAULT_ADAPTER_SOURCE,
        "data_source": DEFAULT_DATA_SOURCE,
        "uji_protocol_source": DEFAULT_UJI_PROTOCOL_SOURCE,
        "hash_source": DEFAULT_HASH_SOURCE,
    }


def register_protocol(
    *,
    r2a_registry_path: str | Path = DEFAULT_R2A_REGISTRY,
    r1_registry_path: str | Path = DEFAULT_R1_REGISTRY,
    r1_result_path: str | Path = DEFAULT_R1_RESULT,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    for artifact in (DEFAULT_CACHE, DEFAULT_RESULT):
        if artifact.exists():
            raise FileExistsError(artifact)

    r2a_file = Path(r2a_registry_path).resolve()
    r1_file = Path(r1_registry_path).resolve()
    r1_result_file = Path(r1_result_path).resolve()
    r2a = load_r2a_registry(r2a_file)
    r1 = load_r1_registry(r1_file)
    r1_result = _load_self_hashed(r1_result_file, "result_sha256")
    if r1_result.get("registered_selection", {}).get("variant") != "P0D0":
        raise ValueError("R2B requires the registered R1 P0D0 selection")
    if r1_result.get("outer_test_predictions") != 0:
        raise ValueError("R1 outer-test access forbids R2B")
    if r1_result.get("official_validation_predictions") != 0:
        raise ValueError("R1 official access forbids R2B")

    base = [Path(item["path"]).resolve() for item in r2a["base_checkpoints"]]
    added = [
        Path(value).resolve()
        for value in r2a["allowed_outputs"]["additional_checkpoints"]
    ]
    checkpoints = base + added
    if len(checkpoints) != MEMBER_COUNT or any(not path.exists() for path in checkpoints):
        raise FileNotFoundError("the registered ten-member R2A panel is incomplete")

    selection_data = load_uji_v16(r1["data_contract"]["selection_refit_csv"])
    development_data = load_uji_v16(r1["data_contract"]["development_csv"])
    selection_keys = set(map(tuple, selection_data["keys"]))
    development_keys = set(map(tuple, development_data["keys"]))
    if selection_keys & development_keys:
        raise ValueError("R2B selection-refit/development coordinates overlap")
    expected_development = r1["data_contract"]["development_role"]
    if len(development_data["rss"]) != expected_development["row_count"]:
        raise ValueError("R2B development row role changed")
    if len(development_keys) != expected_development["coordinate_count"]:
        raise ValueError("R2B development coordinate role changed")
    if len(selection_keys) != ANCHOR_COUNT:
        raise ValueError("R2B selection-refit anchor dictionary changed")

    sources = _source_manifest()
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"R2B source is missing: {missing}")

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_r2b_identifiable_atomic_law_registered",
        "created_after_r1_development_outcome": True,
        "selection_authority_for_r0_or_r1": False,
        "may_modify_r0_or_r1": False,
        "outer_test_prediction_authorized": False,
        "official_validation_prediction_authorized": False,
        "scientific_question": (
            "which identifiable neural/prototype evidence coefficients improve "
            "the untruncated atomic member law"
        ),
        "identifiability_contract": {
            "raw_retrieval_law": "softmax(beta_neural*u-beta_prototype*d)",
            "beta_neural": "neural_scale/posterior_temperature",
            "beta_prototype": "prototype_weight/posterior_temperature",
            "forbidden_redundant_parameterization": [
                "neural_scale", "prototype_weight", "posterior_temperature"
            ],
            "candidate_reranker": False,
            "support_before_scoring": ANCHOR_COUNT,
            "top_L_or_top_K_before_scoring": False,
            "member_consensus_before_scoring": False,
        },
        "grid": {
            "historical_beta_neural": HISTORICAL_BETA_NEURAL,
            "historical_beta_prototype": HISTORICAL_BETA_PROTOTYPE,
            "multipliers": list(LOG_MULTIPLIERS),
            "variants": [
                {
                    "name": beta_name(beta_neural, beta_prototype),
                    "beta_neural": beta_neural,
                    "beta_prototype": beta_prototype,
                }
                for beta_neural, beta_prototype in BETA_GRID
            ],
        },
        "estimand": {
            "primary": "continuous_2d_energy_score_coordinate_macro_m",
            "point_guard": "exact_full_mixture_mean_coordinate_macro_mae_m",
            "aggregation_order": (
                "average ten member losses within scan; average scans within "
                "exact physical coordinate; average coordinate groups"
            ),
            "seed_members_are_independent_observations": False,
            "student_t_degrees_freedom": STUDENT_DF,
            "draws_per_stream": PREDICTIVE_DRAWS_PER_STREAM,
            "predictive_seed": PREDICTIVE_SEED,
            "common_random_numbers_across_members_and_variants": True,
        },
        "selection_rule": {
            "reference": HISTORICAL_VARIANT,
            "family": [name for name in VARIANTS if name != HISTORICAL_VARIANT],
            "mae_noninferiority_margin_m": MAE_NONINFERIORITY_MARGIN_M,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
            "common_coordinate_group_resamples": True,
            "rule": (
                "among variants whose simultaneous MAE upper bound versus the "
                "historical reference is <=0.10 m, identify the smallest mean "
                "energy score; replace the historical reference only when that "
                "variant's family-wise simultaneous energy upper bound is <0; "
                "otherwise retain the historical coefficients"
            ),
            "tie_break": (
                "smaller squared log-distance to the historical coefficients, "
                "then VARIANTS order"
            ),
            "boundary_extension_authorized": False,
        },
        "data_contract": {
            "selection_refit_csv": r1["data_contract"]["selection_refit_csv"],
            "selection_refit_csv_sha256": _sha256_file(
                r1["data_contract"]["selection_refit_csv"]
            ),
            "development_csv": r1["data_contract"]["development_csv"],
            "development_csv_sha256": _sha256_file(
                r1["data_contract"]["development_csv"]
            ),
            "development_role": r1["data_contract"]["development_role"],
            "selection_refit_coordinate_count": len(selection_keys),
            "selection_refit_development_coordinate_overlap": 0,
            "outer_test_role_sealed": True,
            "official_validation_sealed": True,
        },
        "frozen_checkpoints": [
            {"member_index": index, "path": str(path), "sha256": _sha256_file(path)}
            for index, path in enumerate(checkpoints, start=1)
        ],
        "bindings": {
            "r2a_registry": {
                "path": str(r2a_file),
                "file_sha256": _sha256_file(r2a_file),
                "registry_sha256": r2a["registry_sha256"],
            },
            "r1_registry": {
                "path": str(r1_file),
                "file_sha256": _sha256_file(r1_file),
                "registry_sha256": r1["registry_sha256"],
            },
            "r1_result": {
                "path": str(r1_result_file),
                "file_sha256": _sha256_file(r1_result_file),
                "result_sha256": r1_result["result_sha256"],
            },
        },
        "runtime": {
            "component_batch_size": COMPONENT_BATCH_SIZE,
            "scoring_batch_size": SCORING_BATCH_SIZE,
        },
        "allowed_outputs": {
            "development_cache": str(DEFAULT_CACHE.resolve()),
            "development_result": str(DEFAULT_RESULT.resolve()),
        },
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in sources.items()
        },
        "outcome_access_at_registration": {
            "r1_development_outcome_known": True,
            "r2b_development_predictions": 0,
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
        raise ValueError("not a V17-R2B registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(
        _without_hash(registry)
    ):
        raise ValueError("V17-R2B registry self-hash failed")
    for flag in ("outer_test_prediction_authorized", "official_validation_prediction_authorized"):
        if registry.get(flag) is not False:
            raise ValueError(f"V17-R2B illegally authorizes {flag}")
    for record in registry["sources"].values():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17-R2B source changed: {record['path']}")
    for field in ("selection_refit_csv", "development_csv"):
        if _sha256_file(registry["data_contract"][field]) != registry[
            "data_contract"
        ][f"{field}_sha256"]:
            raise ValueError(f"V17-R2B data changed: {field}")
    for record in registry["frozen_checkpoints"]:
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("V17-R2B checkpoint changed")
    for record in registry["bindings"].values():
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"V17-R2B binding changed: {record['path']}")
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--r2a-registry", default=str(DEFAULT_R2A_REGISTRY))
    register.add_argument("--r1-registry", default=str(DEFAULT_R1_REGISTRY))
    register.add_argument("--r1-result", default=str(DEFAULT_R1_RESULT))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))
    check = sub.add_parser("check")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "register":
        payload = register_protocol(
            r2a_registry_path=args.r2a_registry,
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
