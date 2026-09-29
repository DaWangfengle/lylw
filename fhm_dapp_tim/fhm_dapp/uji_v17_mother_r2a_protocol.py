"""Register R2A: extend the frozen AC-RHAP seed panel from five to ten."""

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
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r1_protocol import (
    load_registry as load_r1_registry,
)


PROTOCOL_VERSION = "uji_v17_r2a_exchangeable_panel_v1_20260821"
BASE_MEMBER_COUNT = 5
PANEL_MEMBER_COUNT = 10
ADDITIONAL_MEMBER_SEEDS = tuple(range(17306, 17311))
ADDITIONAL_MEMBER_INDICES = tuple(range(6, 11))

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v17_acrhap_mother"
MODEL_DIR = ROOT / "models" / "v17_acrhap_mother_r2_panel"
DEFAULT_DESIGN = EVIDENCE / "DESIGN_V17_R2_IDENTIFIABLE_FRONT.md"
DEFAULT_R0_REGISTRY = EVIDENCE / "registry_v17_r0.json"
DEFAULT_R1_REGISTRY = EVIDENCE / "registry_v17_r1_backend.json"
DEFAULT_R1_RESULT = EVIDENCE / "development_result_v17_r1_backend.json"
DEFAULT_R1_AMENDMENT = EVIDENCE / "amendment_v17_r1_a1_query_retention.json"
DEFAULT_REGISTRY = EVIDENCE / "registry_v17_r2a_panel.json"

DEFAULT_PROTOCOL_SOURCE = Path(__file__).resolve()
DEFAULT_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r2a.py"
DEFAULT_PROTOCOL_TEST = ROOT.parent / "tests" / "test_uji_v17_mother_r2a.py"
DEFAULT_R0_EXECUTION_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_r0.py"
DEFAULT_R0_PROTOCOL_SOURCE = ROOT / "fhm_dapp" / "uji_v17_mother_protocol.py"
DEFAULT_ANCHOR_SOURCE = ROOT / "fhm_dapp" / "anchor.py"
DEFAULT_MODEL_SOURCE = ROOT / "fhm_dapp" / "model.py"
DEFAULT_TRAIN_SOURCE = ROOT / "fhm_dapp" / "train_anchor_v2.py"


def checkpoint_path(member_index: int) -> Path:
    index = int(member_index)
    if index not in ADDITIONAL_MEMBER_INDICES:
        raise ValueError("R2A creates only members 6 through 10")
    seed = ADDITIONAL_MEMBER_SEEDS[index - 6]
    return MODEL_DIR / f"member_{index:02d}_seed{seed}.pt"


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
        "r0_execution_source": DEFAULT_R0_EXECUTION_SOURCE,
        "r0_protocol_source": DEFAULT_R0_PROTOCOL_SOURCE,
        "anchor_source": DEFAULT_ANCHOR_SOURCE,
        "model_source": DEFAULT_MODEL_SOURCE,
        "training_source": DEFAULT_TRAIN_SOURCE,
    }


def register_protocol(
    *,
    r0_registry_path: str | Path = DEFAULT_R0_REGISTRY,
    r1_registry_path: str | Path = DEFAULT_R1_REGISTRY,
    r1_result_path: str | Path = DEFAULT_R1_RESULT,
    r1_amendment_path: str | Path = DEFAULT_R1_AMENDMENT,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    new_checkpoints = [checkpoint_path(index).resolve() for index in ADDITIONAL_MEMBER_INDICES]
    existing = [str(path) for path in new_checkpoints if path.exists()]
    if existing:
        raise FileExistsError(f"R2A checkpoint already exists: {existing}")

    r0_file = Path(r0_registry_path).resolve()
    r1_file = Path(r1_registry_path).resolve()
    result_file = Path(r1_result_path).resolve()
    amendment_file = Path(r1_amendment_path).resolve()
    r0 = load_r0_registry(r0_file)
    r1 = load_r1_registry(r1_file)
    result = _load_self_hashed(result_file, "result_sha256")
    amendment = _load_self_hashed(amendment_file, "amendment_sha256")
    if result.get("registry_sha256") != r1["registry_sha256"]:
        raise ValueError("R1 result and registry differ")
    if result.get("registered_selection", {}).get("variant") != "P0D0":
        raise ValueError("R2A requires the registered R1 P0D0 selection")
    if result.get("outer_test_predictions") != 0:
        raise ValueError("R1 outer-test access forbids R2A")
    if result.get("official_validation_predictions") != 0:
        raise ValueError("R1 official access forbids R2A")
    if result.get("runtime_amendment", {}).get("amendment_sha256") != amendment[
        "amendment_sha256"
    ]:
        raise ValueError("R1 result is not bound to its runtime amendment")
    if amendment.get("scientific_change") is not False:
        raise ValueError("R1 runtime amendment changed scientific content")

    base_paths = [Path(value).resolve() for value in r0["allowed_outputs"]["r0_checkpoints"]]
    if len(base_paths) != BASE_MEMBER_COUNT or any(not path.exists() for path in base_paths):
        raise FileNotFoundError("the frozen five-member R0 panel is incomplete")
    r1_hashes = tuple(item["sha256"] for item in r1["frozen_checkpoints"])
    if tuple(_sha256_file(path) for path in base_paths) != r1_hashes:
        raise ValueError("R0/R1 five-member panel binding changed")

    sources = _source_manifest()
    missing_sources = [str(path) for path in sources.values() if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"R2A source is missing: {missing_sources}")

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_r2a_exchangeable_panel_registered",
        "scientific_status": (
            "iterative panel extension after R1; training only; no R2 development localization outcome"
        ),
        "created_after_r1_development_outcome": True,
        "selection_authority_for_r0_or_r1": False,
        "may_modify_r0_or_r1": False,
        "development_localization_prediction_authorized": False,
        "outer_test_prediction_authorized": False,
        "official_validation_prediction_authorized": False,
        "panel_contract": {
            "base_members": BASE_MEMBER_COUNT,
            "target_members": PANEL_MEMBER_COUNT,
            "base_member_indices": list(range(1, 6)),
            "additional_member_indices": list(ADDITIONAL_MEMBER_INDICES),
            "additional_member_seeds": list(ADDITIONAL_MEMBER_SEEDS),
            "only_exchangeable_factor_changed": "random_seed",
            "training_algorithm": "exact_V17_R0_member_fit_early_stop_selection_refit",
            "member_selection_from_development": False,
        },
        "data_contract": {
            "train_data": r0["train_data"],
            "train_data_sha256": _sha256_file(r0["train_data"]),
            "roles": {
                name: r0["roles"][name]
                for name in ("member_fit", "early_stop", "selection_refit")
            },
            "development_role_sealed": True,
            "outer_test_role_sealed": True,
            "official_validation_sealed": True,
        },
        "base_checkpoints": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in base_paths
        ],
        "allowed_outputs": {
            "additional_checkpoints": [str(path) for path in new_checkpoints],
        },
        "bindings": {
            "r0_registry": {
                "path": str(r0_file),
                "file_sha256": _sha256_file(r0_file),
                "registry_sha256": r0["registry_sha256"],
            },
            "r1_registry": {
                "path": str(r1_file),
                "file_sha256": _sha256_file(r1_file),
                "registry_sha256": r1["registry_sha256"],
            },
            "r1_result": {
                "path": str(result_file),
                "file_sha256": _sha256_file(result_file),
                "result_sha256": result["result_sha256"],
            },
            "r1_runtime_amendment": {
                "path": str(amendment_file),
                "file_sha256": _sha256_file(amendment_file),
                "amendment_sha256": amendment["amendment_sha256"],
            },
        },
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in sources.items()
        },
        "outcome_access_at_registration": {
            "r1_development_outcome_known": True,
            "r2_development_localization_predictions": 0,
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
        raise ValueError("not a V17-R2A registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(
        _without_hash(registry)
    ):
        raise ValueError("V17-R2A registry self-hash failed")
    for flag in (
        "development_localization_prediction_authorized",
        "outer_test_prediction_authorized",
        "official_validation_prediction_authorized",
    ):
        if registry.get(flag) is not False:
            raise ValueError(f"V17-R2A illegally authorizes {flag}")
    for record in registry["sources"].values():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17-R2A source changed: {record['path']}")
    if _sha256_file(registry["data_contract"]["train_data"]) != registry[
        "data_contract"
    ]["train_data_sha256"]:
        raise ValueError("V17-R2A training data changed")
    for record in registry["base_checkpoints"]:
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("V17-R2A base checkpoint changed")
    for record in registry["bindings"].values():
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"V17-R2A binding changed: {record['path']}")
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--r0-registry", default=str(DEFAULT_R0_REGISTRY))
    register.add_argument("--r1-registry", default=str(DEFAULT_R1_REGISTRY))
    register.add_argument("--r1-result", default=str(DEFAULT_R1_RESULT))
    register.add_argument("--r1-amendment", default=str(DEFAULT_R1_AMENDMENT))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))
    check = sub.add_parser("check")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "register":
        payload = register_protocol(
            r0_registry_path=args.r0_registry,
            r1_registry_path=args.r1_registry,
            r1_result_path=args.r1_result,
            r1_amendment_path=args.r1_amendment,
            output=args.output,
        )
    else:
        payload = load_registry(args.registry)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
