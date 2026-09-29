"""Register the V17-R0 AC-RHAP mother-model reconstruction.

Registration reads geometry and role labels only.  It never trains a model,
constructs a prediction, or admits the official UJI validation file as a
selection source.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from fhm_dapp_tim.fhm_dapp.audit_bplus_sparse50_fem_v5 import (
    geometry_support_with_hull,
)
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _decode_key,
    _role_record,
    _sha256_file,
    _sha256_keys,
    _write_json_new,
    support_fit_stop_roles,
)
from fhm_dapp_tim.fhm_dapp.uji_v9_experiment import audit_coordinate_roles
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16


PROTOCOL_VERSION = "uji_v17_acrhap_mother_r0_v1_20260821"

OUTER_GEOMETRY_SEED = 17101
INNER_GEOMETRY_SEED = 17102
EARLY_STOP_SEED = 17103
MEMBER_SEEDS = tuple(range(17301, 17313))
R0_MEMBER_COUNT = 5
R0_BOOTSTRAP_REPETITIONS = 5000
R0_BOOTSTRAP_SEED = 17502
OUTER_REFIT_FRACTION = 0.80
SELECTION_REFIT_FRACTION = 0.80
EARLY_STOP_FRACTION = 0.15

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "UJIndoorLoc" / "trainingData.csv"
DEFAULT_OFFICIAL = ROOT / "data" / "UJIndoorLoc" / "validationData.csv"
DEFAULT_EVIDENCE = ROOT / "evidence" / "v17_acrhap_mother"
DEFAULT_MODELS = ROOT / "models" / "v17_acrhap_mother_r0"
DEFAULT_DESIGN = DEFAULT_EVIDENCE / "DESIGN_V17_ACRHAP_MOTHER_R0.md"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "registry_v17_r0.json"
DEFAULT_SELECTION_CSV = DEFAULT_EVIDENCE / "selection_refit_v17_r0.csv"
DEFAULT_DEVELOPMENT_CSV = DEFAULT_EVIDENCE / "development_v17_r0.csv"
DEFAULT_RESULT = DEFAULT_EVIDENCE / "development_result_v17_r0.json"

DEFAULT_PROTOCOL_SOURCE = Path(__file__).resolve()
DEFAULT_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r0.py"
DEFAULT_PROTOCOL_TEST = ROOT.parent / "tests" / "test_uji_v17_mother.py"
DEFAULT_TRAINING_SOURCE = ROOT / "fhm_dapp" / "train_anchor_v2.py"
DEFAULT_MODEL_SOURCE = ROOT / "fhm_dapp" / "model.py"
DEFAULT_ANCHOR_SOURCE = ROOT / "fhm_dapp" / "anchor.py"
DEFAULT_CONSENSUS_SOURCE = ROOT / "fhm_dapp" / "consensus.py"
DEFAULT_DAPP_SOURCE = ROOT / "fhm_dapp" / "dapp.py"
DEFAULT_ENSEMBLE_SOURCE = ROOT / "fhm_dapp" / "projected_ensemble.py"

LocationKey = tuple[int, int, float, float]


def checkpoint_path(member_index: int) -> Path:
    index = int(member_index)
    if not 1 <= index <= len(MEMBER_SEEDS):
        raise ValueError("V17 member index is outside the registered budget")
    return DEFAULT_MODELS / f"member_{index:02d}_seed{MEMBER_SEEDS[index - 1]}.pt"


def _registry_without_hash(payload: Mapping[str, object]) -> dict:
    value = copy.deepcopy(dict(payload))
    value.pop("registry_sha256", None)
    return value


def construct_coordinate_roles(
    all_keys: set[LocationKey],
) -> tuple[dict[str, set[LocationKey]], dict]:
    """Build deterministic nested roles with geometry-only hull protection."""

    if not all_keys:
        raise ValueError("all_keys must be nonempty")
    outer_refit, outer_details = geometry_support_with_hull(
        all_keys,
        OUTER_REFIT_FRACTION,
        seed=OUTER_GEOMETRY_SEED,
    )
    outer_test = set(all_keys) - outer_refit
    selection_refit, selection_details = geometry_support_with_hull(
        outer_refit,
        SELECTION_REFIT_FRACTION,
        seed=INNER_GEOMETRY_SEED,
    )
    development = outer_refit - selection_refit
    nested, early_details = support_fit_stop_roles(
        selection_refit,
        seed=EARLY_STOP_SEED,
        early_stop_fraction=EARLY_STOP_FRACTION,
    )
    roles = {
        "member_fit": set(nested["member_fit"]),
        "early_stop": set(nested["early_stop"]),
        "selection_refit": set(selection_refit),
        "development": set(development),
        "outer_refit": set(outer_refit),
        "outer_test": set(outer_test),
    }
    return roles, {
        "outer": outer_details,
        "inner_selection": selection_details,
        "early_stop": early_details,
    }


def _source_manifest() -> dict[str, Path]:
    return {
        "design": DEFAULT_DESIGN,
        "protocol_source": DEFAULT_PROTOCOL_SOURCE,
        "execution_source": DEFAULT_EXECUTION_SOURCE,
        "protocol_test": DEFAULT_PROTOCOL_TEST,
        "historical_training_source": DEFAULT_TRAINING_SOURCE,
        "model_source": DEFAULT_MODEL_SOURCE,
        "anchor_source": DEFAULT_ANCHOR_SOURCE,
        "consensus_source": DEFAULT_CONSENSUS_SOURCE,
        "dapp_source": DEFAULT_DAPP_SOURCE,
        "ensemble_source": DEFAULT_ENSEMBLE_SOURCE,
    }


def register_protocol(
    *,
    train_data: str | Path = DEFAULT_TRAIN,
    official_data: str | Path = DEFAULT_OFFICIAL,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    """Freeze R0 roles, sources, outputs, and the no-test-access boundary."""

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    train_path = Path(train_data).resolve()
    official_path = Path(official_data).resolve()
    sources = _source_manifest()
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V17 R0 source is missing: {missing}")
    if not train_path.exists() or not official_path.exists():
        raise FileNotFoundError("registered UJI source file is missing")

    data = load_uji_v16(train_path)
    all_keys = set(data["keys"])
    roles, split_details = construct_coordinate_roles(all_keys)
    diagnostics = audit_coordinate_roles(roles, data)

    allowed_outputs = {
        "selection_refit_csv": str(DEFAULT_SELECTION_CSV.resolve()),
        "development_csv": str(DEFAULT_DEVELOPMENT_CSV.resolve()),
        "r0_checkpoints": [
            str(checkpoint_path(index).resolve())
            for index in range(1, R0_MEMBER_COUNT + 1)
        ],
        "development_result": str(DEFAULT_RESULT.resolve()),
    }
    flat_outputs = [
        allowed_outputs["selection_refit_csv"],
        allowed_outputs["development_csv"],
        *allowed_outputs["r0_checkpoints"],
        allowed_outputs["development_result"],
    ]
    existing = [value for value in flat_outputs if Path(value).exists()]
    if existing:
        raise FileExistsError(f"registered V17 output already exists: {existing[:3]}")

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_r0_mother_reconstruction_registered",
        "scientific_status": (
            "iterative UJI development created after historical official and "
            "V4.4-V16 outcomes; not independent confirmation"
        ),
        "created_after_v16_official_result": True,
        "historical_official_outcome_known": True,
        "historical_official_row_mae_m": 6.097744675813586,
        "selection_authority_for_v16": False,
        "may_modify_v16": False,
        "official_validation_selection_authority": False,
        "official_validation_prediction_authorized": False,
        "outer_test_prediction_authorized": False,
        "r0_development_prediction_authorized": True,
        "r1_execution_authorized": False,
        "train_data": str(train_path),
        "train_data_sha256": _sha256_file(train_path),
        "official_data_audit_only": str(official_path),
        "official_data_sha256": _sha256_file(official_path),
        "training_rows": int(len(data["keys"])),
        "training_coordinates": int(len(all_keys)),
        "training_coordinate_sha256": _sha256_keys(all_keys),
        "n_aps": int(data["stats"]["n_aps"]),
        "phone_ids": sorted(map(int, np.unique(data["phone"]))),
        "roles": {
            name: _role_record(keys, data["keys"])
            for name, keys in roles.items()
        },
        "split_details": split_details,
        "split_diagnostics": diagnostics,
        "constants": {
            "outer_geometry_seed": OUTER_GEOMETRY_SEED,
            "inner_geometry_seed": INNER_GEOMETRY_SEED,
            "early_stop_seed": EARLY_STOP_SEED,
            "member_seeds": list(MEMBER_SEEDS),
            "r0_member_count": R0_MEMBER_COUNT,
            "outer_refit_fraction": OUTER_REFIT_FRACTION,
            "selection_refit_fraction": SELECTION_REFIT_FRACTION,
            "early_stop_fraction": EARLY_STOP_FRACTION,
            "r0_bootstrap_repetitions": R0_BOOTSTRAP_REPETITIONS,
            "r0_bootstrap_seed": R0_BOOTSTRAP_SEED,
        },
        "estimand": {
            "target": "unseen physical coordinate within UJI acquisition domain",
            "indivisible_unit": "exact raw-CSV (building,floor,longitude,latitude)",
            "primary_aggregation": "coordinate macro",
            "row_metrics_status": "descriptive historical bridge",
        },
        "r0_model_contract": {
            "mechanism": "historical AC-RHAP mother model",
            "E": R0_MEMBER_COUNT,
            "L": 24,
            "K": 24,
            "alpha_consensus": 0.25,
            "alpha_proximal": 0.25,
            "prior_weight_lambda": 0.50,
            "member_weight_mode": "inverse_scale",
            "measurement_pool_mode": "shared",
            "hurdle_score": "balanced_log",
            "hurdle_distribution": "student_t",
            "student_df": 4.0,
            "affine_profile_steps": 10,
            "point_action": "exact_mixture_mean",
            "no_building_or_floor_conditioning": True,
        },
        "access_contract": {
            "training_roles": ["member_fit", "early_stop", "selection_refit"],
            "r0_query_role": "development",
            "sealed_roles": ["outer_test", "official_validation"],
            "r0_failure_action": "stop_and_diagnose_without_rescue_component",
            "r1_requires_separate_registry": True,
        },
        "allowed_outputs": allowed_outputs,
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in sources.items()
        },
        "outcome_access_at_registration": {
            "v17_development_predictions": 0,
            "v17_outer_predictions": 0,
            "v17_official_predictions": 0,
        },
    }
    registry["registry_sha256"] = _canonical_json_sha256(registry)
    _write_json_new(destination, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    """Fail closed if any registered R0 source, data file, or role changed."""

    source = Path(path).resolve()
    registry = json.loads(source.read_text(encoding="utf-8"))
    if registry.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V17 R0 registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(
        _registry_without_hash(registry)
    ):
        raise ValueError("V17 R0 registry self-hash failed")
    if _sha256_file(registry["train_data"]) != registry["train_data_sha256"]:
        raise ValueError("V17 R0 training data changed")
    if (
        _sha256_file(registry["official_data_audit_only"])
        != registry["official_data_sha256"]
    ):
        raise ValueError("V17 R0 official audit file changed")
    for record in registry["sources"].values():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17 R0 frozen source changed: {record['path']}")
    if registry.get("official_validation_prediction_authorized") is not False:
        raise ValueError("V17 R0 illegally authorizes official prediction")
    if registry.get("outer_test_prediction_authorized") is not False:
        raise ValueError("V17 R0 illegally authorizes outer prediction")
    if registry.get("r1_execution_authorized") is not False:
        raise ValueError("V17 R0 illegally authorizes R1")
    return registry


def registered_role_keys(
    registry: Mapping[str, object], role: str
) -> set[LocationKey]:
    roles = registry.get("roles")
    if not isinstance(roles, Mapping) or role not in roles:
        raise ValueError(f"unknown V17 role: {role}")
    record = roles[role]
    if not isinstance(record, Mapping):
        raise TypeError("role record must be a mapping")
    keys = {_decode_key(value) for value in record["coordinate_keys"]}
    if _sha256_keys(keys) != record["coordinate_key_sha256"]:
        raise ValueError(f"coordinate hash failed for V17 role {role}")
    return keys


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    register.add_argument("--official-data", default=str(DEFAULT_OFFICIAL))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))
    check = sub.add_parser("check")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "register":
        payload = register_protocol(
            train_data=args.train_data,
            official_data=args.official_data,
            output=args.output,
        )
        print(json.dumps({
            "registry": str(Path(args.output).resolve()),
            "registry_sha256": payload["registry_sha256"],
            "role_coordinates": {
                name: record["coordinate_count"]
                for name, record in payload["roles"].items()
            },
            "role_rows": {
                name: record["row_count"]
                for name, record in payload["roles"].items()
            },
        }, ensure_ascii=False, indent=2))
        return 0
    payload = load_registry(args.registry)
    print(json.dumps({
        "registry": str(Path(args.registry).resolve()),
        "registry_sha256": payload["registry_sha256"],
        "valid": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
