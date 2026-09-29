"""Scientifically neutral R1 runtime adapter for retained consensus queries.

The registered R1 cache builder compacted each candidate dictionary before
copying the shared seen-AP query.  The registered compact representation does
not include ``query``, so the first cache attempt stopped with ``KeyError``
before any artifact or metric was written.  This adapter preserves that exact
input tensor across compaction and records the amendment in cache/result
metadata.  It changes no numerical operator, hyperparameter, role, or output
path.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r1 as registered_r1
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r1_protocol import (
    DEFAULT_CACHE,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    PROTOCOL_VERSION,
    load_registry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMENDMENT = (
    ROOT
    / "evidence"
    / "v17_acrhap_mother"
    / "amendment_v17_r1_a1_query_retention.json"
)
ADAPTER_PROTOCOL = "uji_v17_r1_runtime_a1_query_retention_v1_20260821"
_ORIGINAL_COMPACT = registered_r1._compact_consensus_candidates
_ORIGINAL_SAVE = registered_r1._save_torch_new
_ORIGINAL_WRITE = registered_r1._write_json_new


def _load_amendment(path: str | Path = DEFAULT_AMENDMENT) -> dict:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("adapter_protocol") != ADAPTER_PROTOCOL:
        raise ValueError("not the registered V17-R1-A1 amendment")
    if payload.get("amendment_sha256") != _canonical_json_sha256(
        payload, omit="amendment_sha256"
    ):
        raise ValueError("V17-R1-A1 amendment self-hash failed")
    if payload.get("scientific_change") is not False:
        raise ValueError("V17-R1-A1 must remain scientifically neutral")
    for name in ("adapter_source", "registered_execution", "registered_protocol"):
        record = payload["bindings"][name]
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17-R1-A1 binding changed: {name}")
    adapter_record = payload["bindings"]["adapter_source"]
    if Path(adapter_record["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("V17-R1-A1 adapter path changed")
    registry_record = payload["bindings"]["registry"]
    if _sha256_file(registry_record["path"]) != registry_record["file_sha256"]:
        raise ValueError("V17-R1 registry file changed after A1")
    registry = load_registry(registry_record["path"])
    if registry["registry_sha256"] != registry_record["registry_sha256"]:
        raise ValueError("V17-R1 registry self-hash changed after A1")
    if payload["failed_attempt"]["development_metrics_computed"] != 0:
        raise ValueError("A1 cannot follow an inspected R1 metric")
    if payload["failed_attempt"]["artifacts_written"] != 0:
        raise ValueError("A1 cannot replace an existing R1 artifact")
    return payload


@contextlib.contextmanager
def _installed_cache_adapter(amendment: dict) -> Iterator[None]:
    if registered_r1._compact_consensus_candidates is not _ORIGINAL_COMPACT:
        raise RuntimeError("R1 compact-candidate primitive was already changed")
    if registered_r1._save_torch_new is not _ORIGINAL_SAVE:
        raise RuntimeError("R1 restricted saver was already changed")

    def retain_query(candidate: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        query = candidate["query"]
        compact = _ORIGINAL_COMPACT(candidate)
        compact["query"] = query
        return compact

    def save_with_amendment(path, payload):
        audited = copy.deepcopy(payload)
        audited["runtime_amendment"] = {
            "adapter_protocol": ADAPTER_PROTOCOL,
            "amendment_sha256": amendment["amendment_sha256"],
            "scientific_change": False,
            "change": "retain_existing_query_tensor_across_candidate_compaction",
        }
        return _ORIGINAL_SAVE(path, audited)

    registered_r1._compact_consensus_candidates = retain_query
    registered_r1._save_torch_new = save_with_amendment
    try:
        yield
    finally:
        registered_r1._save_torch_new = _ORIGINAL_SAVE
        registered_r1._compact_consensus_candidates = _ORIGINAL_COMPACT


@contextlib.contextmanager
def _installed_result_audit(amendment: dict) -> Iterator[None]:
    if registered_r1._write_json_new is not _ORIGINAL_WRITE:
        raise RuntimeError("R1 JSON writer was already changed")

    def write_with_amendment(path, payload):
        audited = copy.deepcopy(payload)
        audited["runtime_amendment"] = {
            "adapter_protocol": ADAPTER_PROTOCOL,
            "amendment_sha256": amendment["amendment_sha256"],
            "scientific_change": False,
            "cache_is_transitively_bound_by_sha256": True,
        }
        audited["result_sha256"] = _canonical_json_sha256(
            audited, omit="result_sha256"
        )
        return _ORIGINAL_WRITE(path, audited)

    registered_r1._write_json_new = write_with_amendment
    try:
        yield
    finally:
        registered_r1._write_json_new = _ORIGINAL_WRITE


def check_contract(
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    *,
    require_cache_absent: bool = True,
) -> dict:
    amendment = _load_amendment(amendment_path)
    registry = load_registry(amendment["bindings"]["registry"]["path"])
    cache = Path(registry["allowed_outputs"]["development_cache"])
    result = Path(registry["allowed_outputs"]["development_result"])
    if require_cache_absent and cache.exists():
        raise FileExistsError(cache)
    if result.exists():
        raise FileExistsError(result)

    marker = torch.arange(3)
    candidate = {
        "baseline": marker,
        "radial_scale": marker,
        "candidate_indices": marker,
        "predicted_building": marker,
        "predicted_floor": marker,
        "initial_weights": marker,
        "embedding": marker,
        "query": marker,
    }
    with _installed_cache_adapter(amendment):
        compact = registered_r1._compact_consensus_candidates(candidate)
    if compact["query"] is not marker:
        raise RuntimeError("A1 copied or altered the registered query tensor")
    if set(compact) != set(candidate):
        raise RuntimeError("A1 changed the compact-candidate field set")
    return {
        "adapter_protocol": ADAPTER_PROTOCOL,
        "amendment_sha256": amendment["amendment_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "scientific_change": False,
        "query_tensor_identity_preserved": True,
        "cache_exists": cache.exists(),
        "result_exists": result.exists(),
        "development_metrics_computed_before_amendment": 0,
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def build_cache(
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    device_name: str = "cuda",
) -> dict:
    check_contract(amendment_path, require_cache_absent=True)
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registry"]["path"]
    with _installed_cache_adapter(amendment):
        return registered_r1.build_cache(
            registry_path=registry_path,
            output=DEFAULT_CACHE,
            device_name=device_name,
        )


def evaluate(
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    device_name: str = "cuda",
) -> dict:
    check_contract(amendment_path, require_cache_absent=False)
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registry"]["path"]
    cache = torch.load(DEFAULT_CACHE, map_location="cpu", weights_only=True)
    audit = cache.get("runtime_amendment", {})
    if audit.get("amendment_sha256") != amendment["amendment_sha256"]:
        raise ValueError("R1 cache is not bound to A1")
    with _installed_result_audit(amendment):
        registered_r1.evaluate(
            registry_path=registry_path,
            cache_path=DEFAULT_CACHE,
            output=DEFAULT_RESULT,
            device_name=device_name,
        )
    payload = json.loads(DEFAULT_RESULT.read_text(encoding="utf-8"))
    if payload.get("result_sha256") != _canonical_json_sha256(
        payload, omit="result_sha256"
    ):
        raise RuntimeError("A1 result audit self-hash failed")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-contract")
    check.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    cache = sub.add_parser("build-cache")
    cache.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    cache.add_argument("--device", default="cuda")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    evaluate_parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "check-contract":
        result = check_contract(args.amendment)
    elif args.command == "build-cache":
        result = build_cache(
            amendment_path=args.amendment,
            device_name=args.device,
        )
    else:
        result = evaluate(
            amendment_path=args.amendment,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
