"""Scientifically neutral recovery runtime for the failed V17-R3 attempt.

The registered execution stored retained-mass diagnostics as [members,batch]
blocks and attempted to concatenate a two-row final batch with 32-row blocks.
This adapter flattens only that unused diagnostic output and records the replay
as the second outer execution attempt.
"""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as registered_r3
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r3_protocol import (
    CONFIGURATIONS,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    OUTER_TEST_COORDINATES,
    OUTER_TEST_ROWS,
    PANEL_MEMBER_COUNT,
    load_registry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMENDMENT = (
    ROOT
    / "evidence"
    / "v17_acrhap_mother"
    / "amendment_v17_r3_a1_diagnostic_flatten.json"
)
RUNTIME_PROTOCOL = "uji_v17_r3_runtime_a1_diagnostic_flatten_v1_20260822"


def _load_amendment(path: str | Path = DEFAULT_AMENDMENT) -> dict:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("amendment_sha256") != _canonical_json_sha256(
        payload, omit="amendment_sha256"
    ):
        raise ValueError("R3-A1 amendment self-hash failed")
    if payload.get("runtime_protocol") != RUNTIME_PROTOCOL:
        raise ValueError("R3-A1 runtime protocol changed")
    if payload.get("scientific_change") is not False:
        raise ValueError("R3-A1 must remain scientifically neutral")
    if payload.get("selection_authority_for_v17_r2f") is not False:
        raise ValueError("R3-A1 cannot select or modify R2F")
    if payload.get("may_modify_v17_r3_estimand") is not False:
        raise ValueError("R3-A1 cannot modify the R3 estimand")
    if payload.get("outer_test_execution_attempts_before_retry") != 1:
        raise ValueError("R3-A1 must follow exactly one failed outer attempt")
    if payload.get("first_attempt_outcomes_inspected") is not False:
        raise ValueError("R3-A1 retry is forbidden after outcome inspection")
    if payload.get("retry_authorized") is not True:
        raise ValueError("R3-A1 retry was not authorized")
    for name, record in payload["bindings"].items():
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"R3-A1 binding changed: {name}")
    if Path(payload["bindings"]["runtime_source"]["path"]).resolve() != Path(
        __file__
    ).resolve():
        raise ValueError("R3-A1 runtime path changed")
    registry = load_registry(payload["bindings"]["registered_registry"]["path"])
    if registry["registry_sha256"] != payload["bindings"]["registered_registry"][
        "registry_sha256"
    ]:
        raise ValueError("R3-A1 registry identity changed")
    registered_paths = registry["allowed_outputs"]["outer_refit_checkpoints"]
    checkpoint_records = payload["outer_refit_checkpoints"]
    if len(checkpoint_records) != PANEL_MEMBER_COUNT:
        raise ValueError("R3-A1 does not bind all final members")
    for expected_path, record in zip(registered_paths, checkpoint_records):
        if Path(expected_path).resolve() != Path(record["path"]).resolve():
            raise ValueError("R3-A1 final member path order changed")
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError("R3-A1 final member checkpoint changed")
    return payload


def _flatten_diagnostic_retained_mass(
    original_member_top_l,
    *args,
    **kwargs,
):
    prior, indices, radial, retained = original_member_top_l(*args, **kwargs)
    return prior, indices, radial, retained.reshape(-1)


@contextmanager
def _diagnostic_flatten_adapter(amendment: dict):
    original_member_top_l = registered_r3._member_top_l
    original_write_json_new = registered_r3._write_json_new

    def amended_member_top_l(*args, **kwargs):
        return _flatten_diagnostic_retained_mass(
            original_member_top_l, *args, **kwargs
        )

    def amended_write_json_new(path, payload):
        enriched = copy.deepcopy(payload)
        enriched["runtime_protocol"] = RUNTIME_PROTOCOL
        enriched["execution_amendment"] = {
            "amendment_sha256": amendment["amendment_sha256"],
            "scientific_change": False,
            "diagnostic_only_change": "flatten_retained_mass_blocks_before_cat",
            "predictive_law_changed": False,
            "loss_or_inference_changed": False,
            "first_attempt_result_file_created": False,
            "first_attempt_metrics_printed": False,
            "first_attempt_outcomes_inspected": False,
        }
        enriched["outer_test_access_number"] = 2
        enriched["outer_test_execution_attempts"] = 2
        enriched["successful_outer_result_writes"] = 1
        enriched["result_sha256"] = _canonical_json_sha256(
            enriched, omit="result_sha256"
        )
        return original_write_json_new(path, enriched)

    registered_r3._member_top_l = amended_member_top_l
    registered_r3._write_json_new = amended_write_json_new
    try:
        yield
    finally:
        registered_r3._member_top_l = original_member_top_l
        registered_r3._write_json_new = original_write_json_new


def check_contract(
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    *,
    require_result_absent: bool = True,
) -> dict:
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registered_registry"]["path"]
    registry = load_registry(registry_path)
    result_path = Path(registry["allowed_outputs"]["outer_result"])
    if require_result_absent and result_path.exists():
        raise FileExistsError("R3 result exists before the authorized A1 replay")
    members = registered_r3.check_members(registry_path)
    if members["all_complete"] is not True:
        raise RuntimeError("R3-A1 requires all 25 final members")
    if _sha256_file(registry["allowed_outputs"]["outer_refit_csv"]) != amendment[
        "bindings"
    ]["outer_refit_csv"]["file_sha256"]:
        raise ValueError("R3-A1 outer-refit CSV changed")
    return {
        "runtime_protocol": RUNTIME_PROTOCOL,
        "amendment_sha256": amendment["amendment_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "complete_members": members["complete_members"],
        "result_absent": not result_path.exists(),
        "outer_test_execution_attempts_before_retry": 1,
        "first_attempt_outcomes_inspected": False,
        "official_validation_predictions": 0,
        "scientific_change": False,
    }


@torch.no_grad()
def evaluate(
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    *,
    device_name: str = "cuda",
) -> dict:
    contract = check_contract(amendment_path, require_result_absent=True)
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registered_registry"]["path"]
    registry = load_registry(registry_path)
    with _diagnostic_flatten_adapter(amendment):
        registered_r3.evaluate_outer(
            registry_path=registry_path,
            device_name=device_name,
        )
    result_path = Path(registry["allowed_outputs"]["outer_result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        **contract,
        "result": str(result_path.resolve()),
        "result_sha256": result["result_sha256"],
        "outer_test_execution_attempts": result["outer_test_execution_attempts"],
        "successful_outer_result_writes": result["successful_outer_result_writes"],
        "metrics": result["metrics"],
        "paired_familywise_inference": result["paired_familywise_inference"],
        "endpoint_conclusions": result["endpoint_conclusions"],
    }


def check_result(
    amendment_path: str | Path = DEFAULT_AMENDMENT,
) -> dict:
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registered_registry"]["path"]
    registry = load_registry(registry_path)
    source = Path(registry["allowed_outputs"]["outer_result"])
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("result_sha256") != _canonical_json_sha256(
        result, omit="result_sha256"
    ):
        raise ValueError("R3-A1 result self-hash failed")
    execution = result.get("execution_amendment", {})
    checks = {
        "runtime": result.get("runtime_protocol") == RUNTIME_PROTOCOL,
        "amendment": execution.get("amendment_sha256")
        == amendment["amendment_sha256"],
        "scientific_change_false": execution.get("scientific_change") is False,
        "predictive_law_unchanged": execution.get("predictive_law_changed") is False,
        "loss_and_inference_unchanged": execution.get("loss_or_inference_changed")
        is False,
        "first_result_absent": execution.get("first_attempt_result_file_created")
        is False,
        "first_outcomes_uninspected": execution.get(
            "first_attempt_outcomes_inspected"
        )
        is False,
        "two_attempts": result.get("outer_test_access_number") == 2
        and result.get("outer_test_execution_attempts") == 2,
        "one_result_write": result.get("successful_outer_result_writes") == 1,
        "outer_rows": result.get("outer_test_rows") == OUTER_TEST_ROWS,
        "outer_groups": result.get("outer_test_coordinate_groups")
        == OUTER_TEST_COORDINATES,
        "official_sealed": result.get("official_validation_predictions") == 0
        and result.get("official_validation_path_opened") is False,
        "both_configurations": tuple(result.get("metrics", {}))
        == tuple(CONFIGURATIONS),
        "family_size_two": result.get("paired_familywise_inference", {}).get(
            "family_size"
        )
        == 2,
        "realizations_complete": all(
            len(result.get("realization_metrics", {}).get(name, ()))
            == PANEL_MEMBER_COUNT
            for name in CONFIGURATIONS
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"R3-A1 result contract failed: {checks}")
    return {
        "result": str(source.resolve()),
        "result_file_sha256": _sha256_file(source),
        "result_sha256": result["result_sha256"],
        "checks": checks,
        "metrics": result["metrics"],
        "paired_familywise_inference": result["paired_familywise_inference"],
        "endpoint_conclusions": result["endpoint_conclusions"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract = subparsers.add_parser("check-contract")
    contract.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    score = subparsers.add_parser("evaluate")
    score.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    score.add_argument("--device", default="cuda")
    result_check = subparsers.add_parser("check-result")
    result_check.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    args = parser.parse_args(argv)
    if args.command == "check-contract":
        result = check_contract(args.amendment)
    elif args.command == "evaluate":
        result = evaluate(args.amendment, device_name=args.device)
    else:
        result = check_result(args.amendment)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

