"""A3-isolated execution adapter for the frozen TUJI1 V9 experiment.

This module does not edit V1, V2, or V3.  It executes the unchanged V2
mathematics with A2's compatible loaders, while retaining an unredirected V2
contract checker for predecessor validation.  V4 owns all new paths and all
new outcome artifacts.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from fhm_dapp_tim.fhm_dapp import tuji_v9_experiment as v1
from fhm_dapp_tim.fhm_dapp import tuji_v9_memory_safe_v2 as v2
from fhm_dapp_tim.fhm_dapp import tuji_v9_memory_safe_v3_callfix as v3
from fhm_dapp_tim.fhm_dapp.tuji_v9_protocol import DEFAULT_REGISTRY, PROTOCOL_VERSION, load_registry


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v9_tuji"
V1_CONTRACT = EVIDENCE / "execution_contract_tuji_v9.json"
V2_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v2_memory_safe.json"
V3_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v3_memory_safe_callfix.json"
A1_AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A1_MEMORY_SAFE.md"
A2_AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A2_CALL_SIGNATURE.md"
A3_AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A3_RUNTIME_CONTRACT_ISOLATION.md"
DEFAULT_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v4_runtime_isolation.json"
DEFAULT_FREEZE = EVIDENCE / "selection_freeze_tuji_v9_v4_runtime_isolation.json"
DEFAULT_TEST_CACHE = EVIDENCE / "author_test_cache_tuji_v9_v4_runtime_isolation.pt"
DEFAULT_RESULT = EVIDENCE / "author_test_result_tuji_v9_v4_runtime_isolation.json"
FINAL_MODELS = ROOT / "models" / "v9_tuji_final_v4_runtime_isolation"

EXECUTION_CONTRACT_VERSION = "tuji_v9_execution_contract_v4_runtime_isolation_20260812"
EXTENSION_MEMBER_COUNTS = v2.EXTENSION_MEMBER_COUNTS
_ORIGINAL_V2_LOAD_EXECUTION_CONTRACT = v2.load_execution_contract


def _candidate(value: str) -> str:
    name = str(value)
    if name not in v1.CANDIDATES:
        raise ValueError(f"unknown V9 candidate: {name}")
    return name


def _require_path(path: str | Path, expected: str | Path, *, label: str) -> Path:
    result, frozen = Path(path).resolve(), Path(expected).resolve()
    if result != frozen:
        raise ValueError(f"{label} must be the frozen protocol path: {frozen}")
    return result


def _cache_path(candidate: str, members: int) -> Path:
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_cache_{_candidate(candidate)}_through_E{count}_v4_runtime_isolation.pt"


def _grid_path(candidate: str, members: int) -> Path:
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_grid_{_candidate(candidate)}_through_E{count}_v4_runtime_isolation.pt"


def _grid_summary_path(candidate: str, members: int) -> Path:
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_grid_{_candidate(candidate)}_through_E{count}_v4_runtime_isolation.json"


def _extension_report_path(members: int) -> Path:
    return EVIDENCE / f"selection_extension_required_through_E{int(members)}_tuji_v9_v4.json"


def _final_checkpoint_path(candidate: str, index: int) -> Path:
    member = int(index)
    if not 1 <= member <= len(v1.MEMBER_SEEDS):
        raise ValueError("member index is outside the preregistered panel")
    return FINAL_MODELS / _candidate(candidate) / f"member_{member:02d}_seed{v1.MEMBER_SEEDS[member - 1]}.pt"


def _final_radio_path(candidate: str) -> Path:
    return EVIDENCE / f"radio_memory_final_{_candidate(candidate)}_v4_runtime_isolation.pt"


def _future_output_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {
        "selection_freeze": DEFAULT_FREEZE,
        "author_test_cache": DEFAULT_TEST_CACHE,
        "author_test_result": DEFAULT_RESULT,
    }
    for candidate in v1.CANDIDATES:
        paths[f"final_radio_{candidate}"] = _final_radio_path(candidate)
        for index in range(1, len(v1.MEMBER_SEEDS) + 1):
            paths[f"final_{candidate}_{index}"] = _final_checkpoint_path(candidate, index)
        for count in EXTENSION_MEMBER_COUNTS:
            paths[f"cache_{candidate}_E{count}"] = _cache_path(candidate, count)
            paths[f"grid_{candidate}_E{count}"] = _grid_path(candidate, count)
            paths[f"grid_summary_{candidate}_E{count}"] = _grid_summary_path(candidate, count)
    for count in EXTENSION_MEMBER_COUNTS[:-1]:
        paths[f"extension_report_E{count}"] = _extension_report_path(count)
    return paths


def _source_paths() -> dict[str, Path]:
    module_dir = Path(__file__).resolve().parent
    return {
        "v4_runtime_isolation_executor": Path(__file__).resolve(),
        "v3_callfix_executor": module_dir / "tuji_v9_memory_safe_v3_callfix.py",
        "v2_memory_safe_executor": module_dir / "tuji_v9_memory_safe_v2.py",
        "v1_executor": module_dir / "tuji_v9_experiment.py",
        "tuji_protocol": module_dir / "tuji_v9_protocol.py",
        "uji_v9_core": module_dir / "uji_v9_experiment.py",
        "marked_math": module_dir / "coherent_marked_v9.py",
        "marked_memory": module_dir / "experiment_v71.py",
        "safe_artifact_primitives": module_dir / "experiment_v7.py",
        "front_model": module_dir / "model.py",
        "anchor_memory": module_dir / "anchor.py",
        "front_training": module_dir / "train_anchor.py",
        "execution_amendment_a1": A1_AMENDMENT,
        "execution_amendment_a2": A2_AMENDMENT,
        "execution_amendment_a3": A3_AMENDMENT,
        "v1_execution_contract": V1_CONTRACT,
        "v2_execution_contract": V2_CONTRACT,
        "v3_execution_contract": V3_CONTRACT,
    }


def _manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {name: {"path": str(path.resolve()), "sha256": v1._sha256_file(path)} for name, path in paths.items()}


def _load_frozen_v2_contract(registry_path: str | Path) -> dict[str, object]:
    """Use V2's original checker even after V4 redirects V2 runtime globals."""

    prior_default = v2.DEFAULT_CONTRACT
    try:
        v2.DEFAULT_CONTRACT = V2_CONTRACT
        return _ORIGINAL_V2_LOAD_EXECUTION_CONTRACT(V2_CONTRACT, registry_path=registry_path)
    finally:
        v2.DEFAULT_CONTRACT = prior_default


def _load_failed_v3_contract(registry_path: str | Path) -> dict[str, object]:
    payload = json.loads(V3_CONTRACT.read_text(encoding="utf-8"))
    if payload.get("execution_contract_version") != v3.EXECUTION_CONTRACT_VERSION:
        raise ValueError("unexpected V3 contract version")
    if payload.get("execution_contract_sha256") != v1._canonical_json_sha256(payload, omit="execution_contract_sha256"):
        raise ValueError("V3 failed-record self-hash mismatch")
    if Path(str(payload["registry"])).resolve() != Path(registry_path).resolve():
        raise ValueError("V3 failed record uses another registry")
    for name, record in payload["execution_sources"].items():
        if v1._sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V3 failed-record source changed: {name}")
    produced = [name for name, path in payload["future_output_paths"].items() if Path(str(path)).exists()]
    if produced:
        raise RuntimeError("V3 failed attempt unexpectedly has output artifacts: " + ", ".join(produced))
    return payload


def _validate_predecessors(registry_path: str | Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    registry_file = Path(registry_path).resolve()
    v1_contract = v1.load_execution_contract(V1_CONTRACT, registry_path=registry_file)
    v2_contract = _load_frozen_v2_contract(registry_file)
    v3_contract = _load_failed_v3_contract(registry_file)
    if v2_contract["v1_execution_contract"]["execution_contract_sha256"] != v1_contract["execution_contract_sha256"]:
        raise ValueError("V2 does not identify the frozen V1 antecedent")
    if v3_contract["v1_execution_contract"]["execution_contract_sha256"] != v1_contract["execution_contract_sha256"]:
        raise ValueError("V3 does not identify the frozen V1 antecedent")
    return v1_contract, v2_contract, v3_contract


def freeze_execution_contract(*, registry_path: str | Path = DEFAULT_REGISTRY, output: str | Path = DEFAULT_CONTRACT) -> dict[str, object]:
    registry_file = Path(registry_path).resolve()
    destination = _require_path(output, DEFAULT_CONTRACT, label="V4 execution contract")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite execution contract: {destination}")
    registry = load_registry(registry_file)
    v1_contract, v2_contract, v3_contract = _validate_predecessors(registry_file)
    prior = v2._selection_artifact_manifest(registry=registry, v1_contract=v1_contract)
    existing = [str(path.resolve()) for path in _future_output_paths().values() if path.exists()]
    if existing:
        raise FileExistsError("V4 contract must precede all V4 outcomes: " + ", ".join(existing))
    contract: dict[str, object] = {
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "registry": str(registry_file),
        "registry_sha256": registry["registry_sha256"],
        "design": str(Path(str(registry["design"])).resolve()),
        "design_sha256": registry["design_sha256"],
        "comparison_status": "external_replication_after_eligibility_audit",
        "selection_authority_for_v9_uji": False,
        "may_modify_v9_uji": False,
        "scientific_change": False,
        "amendment_chain": [
            {"id": "A1-memory-safe-cache", "path": str(A1_AMENDMENT.resolve()), "sha256": v1._sha256_file(A1_AMENDMENT)},
            {"id": "A2-keyword-loader-callfix", "path": str(A2_AMENDMENT.resolve()), "sha256": v1._sha256_file(A2_AMENDMENT)},
            {"id": "A3-runtime-contract-isolation", "path": str(A3_AMENDMENT.resolve()), "sha256": v1._sha256_file(A3_AMENDMENT)},
        ],
        "v1_execution_contract": {"path": str(V1_CONTRACT.resolve()), "sha256": v1._sha256_file(V1_CONTRACT), "execution_contract_sha256": v1_contract["execution_contract_sha256"]},
        "v2_failed_execution": {"path": str(V2_CONTRACT.resolve()), "sha256": v1._sha256_file(V2_CONTRACT), "execution_contract_sha256": v2_contract["execution_contract_sha256"], "created_outcomes": False, "read_author_test_rss": False, "read_author_test_labels": False, "failure": "keyword-only loader called positionally before development-cache serialization"},
        "v3_failed_audit": {"path": str(V3_CONTRACT.resolve()), "sha256": v1._sha256_file(V3_CONTRACT), "execution_contract_sha256": v3_contract["execution_contract_sha256"], "created_outcomes": False, "read_author_test_rss": False, "read_author_test_labels": False, "failure": "runtime redirection recursively selected V3 loader while auditing V2 antecedent"},
        "preexisting_v1_selection_artifacts": prior,
        "test_model_outcome_accesses_before_v4_freeze": 0,
        "memory_safe_execution": {"member_residency": "one member at a time", "query_chunk_size": v2.QUERY_CHUNK_SIZE, "scoring_chunk_size": v2.SCORING_CHUNK_SIZE, "scientific_parameters_changed": False},
        "loader_compatibility": {"selection": "A2 exact V1 keyword-only ABI", "final": "A2 V2-final schema-compatible reconstruction", "runtime_contract_isolation": "A3 captured original V2 checker with temporary frozen V2 contract path"},
        "test_gate": {"selection_commands": "author_training_only", "author_test_cache": "requires V4 selection freeze, V4 final refit panel, and V4 final marked memory", "author_test_query_input": "RSS_testing.csv only", "author_test_labels": "only after prediction cache exists"},
        "selection_grid": {"E": list(v1.E_GRID), "K": list(v1.K_GRID), "lambda": list(v1.LAMBDA_GRID), "member_seeds": list(v1.MEMBER_SEEDS), "extensions": list(EXTENSION_MEMBER_COUNTS), "member_rule": "preregistered V1 seed prefix"},
        "execution_sources": _manifest(_source_paths()),
        "future_output_paths": {name: str(path.resolve()) for name, path in _future_output_paths().items()},
    }
    contract["execution_contract_sha256"] = v1._canonical_json_sha256(contract, omit="execution_contract_sha256")
    v1._write_json_new(destination, contract)
    return contract


def load_execution_contract(path: str | Path = DEFAULT_CONTRACT, *, registry_path: str | Path | None = None) -> dict[str, object]:
    contract_file = _require_path(path, DEFAULT_CONTRACT, label="V4 execution contract")
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION or contract.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V4 TUJI execution contract")
    if contract.get("execution_contract_sha256") != v1._canonical_json_sha256(contract, omit="execution_contract_sha256"):
        raise ValueError("V4 execution-contract self-hash failed")
    registry_file = Path(str(contract["registry"])).resolve()
    if registry_path is not None and Path(registry_path).resolve() != registry_file:
        raise ValueError("caller registry differs from V4 contract")
    registry = load_registry(registry_file)
    if registry["registry_sha256"] != contract["registry_sha256"] or registry["design_sha256"] != contract["design_sha256"]:
        raise ValueError("V4 registry or design hash mismatch")
    for name, record in contract["execution_sources"].items():
        if v1._sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V4 source changed after freeze: {name}")
    v1_contract, v2_contract, v3_contract = _validate_predecessors(registry_file)
    if v1_contract["execution_contract_sha256"] != contract["v1_execution_contract"]["execution_contract_sha256"] or v2_contract["execution_contract_sha256"] != contract["v2_failed_execution"]["execution_contract_sha256"] or v3_contract["execution_contract_sha256"] != contract["v3_failed_audit"]["execution_contract_sha256"]:
        raise ValueError("frozen predecessor identity mismatch")
    if v2._selection_artifact_manifest(registry=registry, v1_contract=v1_contract) != contract["preexisting_v1_selection_artifacts"]:
        raise ValueError("immutable V1 selection artifacts changed")
    return contract


def _install_v2_runtime() -> None:
    """Route V2's unchanged computation only after V4 contract validation."""

    v2.DEFAULT_CONTRACT = DEFAULT_CONTRACT
    v2.DEFAULT_FREEZE = DEFAULT_FREEZE
    v2.DEFAULT_TEST_CACHE = DEFAULT_TEST_CACHE
    v2.DEFAULT_RESULT = DEFAULT_RESULT
    v2.FINAL_MODELS = FINAL_MODELS
    v2._v2_cache_path = _cache_path
    v2._v2_grid_path = _grid_path
    v2._v2_grid_summary_path = _grid_summary_path
    v2._extension_report_path = _extension_report_path
    v2._final_checkpoint_path = _final_checkpoint_path
    v2._final_radio_path = _final_radio_path
    v2._load_one_front = v3._load_one_front_v3
    v2._load_v2_final_front = v3._load_v2_final_front_v3
    v2.load_execution_contract = load_execution_contract


def audit_execution_contract(*, registry_path: str | Path = DEFAULT_REGISTRY, contract_path: str | Path = DEFAULT_CONTRACT) -> dict[str, object]:
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    _install_v2_runtime()
    base = v2.audit_execution_contract(registry_path=registry_path, contract_path=contract_path)
    source = inspect.getsource(v3._load_one_front_v3)
    if "registry=registry" not in source or "contract=v1_contract" not in source or "device=torch.device" not in source:
        raise RuntimeError("A2 selection loader source no longer exposes V1 keyword ABI")
    registry = load_registry(registry_path)
    v1_contract = v1.load_execution_contract(V1_CONTRACT, registry_path=registry_path)
    model, payload = v3._load_one_front_v3(v1.selection_checkpoint_path("geometry_invariant", 1), registry=registry, expected_contract_sha256=str(v1_contract["execution_contract_sha256"]), expected_phase="selection", expected_candidate="geometry_invariant", expected_role="selection_refit", expected_index=1)
    anchors = int(model.n_anchors)
    del model, payload
    gc.collect()
    return {"status": "passed_runtime_isolation_pre_outcome_audit", "registry_sha256": contract["registry_sha256"], "v1_execution_contract_sha256": contract["v1_execution_contract"]["execution_contract_sha256"], "v2_failed_execution_contract_sha256": contract["v2_failed_execution"]["execution_contract_sha256"], "v3_failed_audit_contract_sha256": contract["v3_failed_audit"]["execution_contract_sha256"], "v4_execution_contract_sha256": contract["execution_contract_sha256"], "v1_selection_artifacts_verified": len(contract["preexisting_v1_selection_artifacts"]), "v4_future_outputs_absent": True, "one_member_gpu_residency": True, "author_test_cache_requires_freeze_before_rss": True, "author_test_cache_does_not_load_labels": True, "selection_loader_keyword_abi_checked": True, "selection_loader_cpu_anchors": anchors, "underlying_v2_audit_status": base["status"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-execution"); freeze.add_argument("--registry", default=str(DEFAULT_REGISTRY)); freeze.add_argument("--output", default=str(DEFAULT_CONTRACT))
    audit = commands.add_parser("audit-execution"); audit.add_argument("--registry", default=str(DEFAULT_REGISTRY)); audit.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    extension = commands.add_parser("train-v1-extension-member"); extension.add_argument("--candidate", choices=v1.CANDIDATES, required=True); extension.add_argument("--member-index", type=int, required=True); extension.add_argument("--registry", default=str(DEFAULT_REGISTRY)); extension.add_argument("--contract", default=str(DEFAULT_CONTRACT)); extension.add_argument("--device", default="cuda"); extension.add_argument("--log-every", type=int, default=5)
    cache = commands.add_parser("build-development-cache"); cache.add_argument("--candidate", choices=v1.CANDIDATES, required=True); cache.add_argument("--members", type=int, default=v1.INITIAL_MEMBER_COUNT); cache.add_argument("--registry", default=str(DEFAULT_REGISTRY)); cache.add_argument("--contract", default=str(DEFAULT_CONTRACT)); cache.add_argument("--device", default="cuda")
    grid = commands.add_parser("grid"); grid.add_argument("--candidate", choices=v1.CANDIDATES, required=True); grid.add_argument("--members", type=int, default=v1.INITIAL_MEMBER_COUNT); grid.add_argument("--registry", default=str(DEFAULT_REGISTRY)); grid.add_argument("--contract", default=str(DEFAULT_CONTRACT)); grid.add_argument("--device", default="cuda")
    select = commands.add_parser("freeze-selection"); select.add_argument("--members", type=int, default=v1.INITIAL_MEMBER_COUNT); select.add_argument("--registry", default=str(DEFAULT_REGISTRY)); select.add_argument("--contract", default=str(DEFAULT_CONTRACT)); select.add_argument("--output", default=str(DEFAULT_FREEZE))
    final = commands.add_parser("train-final-member"); final.add_argument("--member-index", type=int, required=True); final.add_argument("--registry", default=str(DEFAULT_REGISTRY)); final.add_argument("--contract", default=str(DEFAULT_CONTRACT)); final.add_argument("--freeze", default=str(DEFAULT_FREEZE)); final.add_argument("--device", default="cuda"); final.add_argument("--log-every", type=int, default=5)
    radio = commands.add_parser("fit-final-radio"); radio.add_argument("--registry", default=str(DEFAULT_REGISTRY)); radio.add_argument("--contract", default=str(DEFAULT_CONTRACT)); radio.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    test = commands.add_parser("cache-author-test"); test.add_argument("--registry", default=str(DEFAULT_REGISTRY)); test.add_argument("--contract", default=str(DEFAULT_CONTRACT)); test.add_argument("--freeze", default=str(DEFAULT_FREEZE)); test.add_argument("--device", default="cuda")
    evaluate = commands.add_parser("evaluate-author-test"); evaluate.add_argument("--registry", default=str(DEFAULT_REGISTRY)); evaluate.add_argument("--contract", default=str(DEFAULT_CONTRACT)); evaluate.add_argument("--freeze", default=str(DEFAULT_FREEZE)); evaluate.add_argument("--cache", default=str(DEFAULT_TEST_CACHE)); evaluate.add_argument("--output", default=str(DEFAULT_RESULT)); evaluate.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-execution":
        result = freeze_execution_contract(registry_path=args.registry, output=args.output)
    elif args.command == "audit-execution":
        result = audit_execution_contract(registry_path=args.registry, contract_path=args.contract)
    else:
        load_execution_contract(args.contract, registry_path=args.registry)
        _install_v2_runtime()
        if args.command == "train-v1-extension-member":
            result = v2.train_v1_selection_extension_member(candidate=args.candidate, member_index=args.member_index, registry_path=args.registry, contract_path=args.contract, device_name=args.device, log_every=args.log_every)
        elif args.command == "build-development-cache":
            result = v2.build_development_cache(candidate=args.candidate, members=args.members, registry_path=args.registry, contract_path=args.contract, device_name=args.device)
        elif args.command == "grid":
            result = v2.evaluate_development_grid(candidate=args.candidate, members=args.members, registry_path=args.registry, contract_path=args.contract, device_name=args.device)
        elif args.command == "freeze-selection":
            result = v2.select_and_freeze(members=args.members, registry_path=args.registry, contract_path=args.contract, output=args.output)
        elif args.command == "train-final-member":
            result = v2.train_final_member(member_index=args.member_index, registry_path=args.registry, contract_path=args.contract, freeze_path=args.freeze, device_name=args.device, log_every=args.log_every)
        elif args.command == "fit-final-radio":
            result = v2.fit_final_radio_memory(registry_path=args.registry, contract_path=args.contract, freeze_path=args.freeze)
        elif args.command == "cache-author-test":
            result = v2.cache_author_test(registry_path=args.registry, contract_path=args.contract, freeze_path=args.freeze, device_name=args.device)
        elif args.command == "evaluate-author-test":
            result = v2.evaluate_author_test(registry_path=args.registry, contract_path=args.contract, freeze_path=args.freeze, cache_path=args.cache, output=args.output, device_name=args.device)
        else:  # pragma: no cover
            raise RuntimeError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
