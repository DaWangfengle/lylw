"""A2-compatible, memory-safe TUJI1 V9 execution adapter.

V1 is immutable and its eight-member cache attempt failed only because all
members were simultaneously resident on a 6 GiB GPU.  V2 correctly scheduled
one member at a time, but it called V1's keyword-only checkpoint loader with a
positional device argument before any cache was written.  V3 retains V2's
mathematics and scheduling, puts every output under a new V3 contract/path, and
installs two explicit ABI-compatible loaders at runtime.  It never modifies
V1 or V2 source files.
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
from fhm_dapp_tim.fhm_dapp.tuji_v9_protocol import (
    DEFAULT_REGISTRY,
    PROTOCOL_VERSION,
    load_registry,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v9_tuji"
V1_CONTRACT = EVIDENCE / "execution_contract_tuji_v9.json"
V2_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v2_memory_safe.json"
A1_AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A1_MEMORY_SAFE.md"
A2_AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A2_CALL_SIGNATURE.md"
DEFAULT_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v3_memory_safe_callfix.json"
DEFAULT_FREEZE = EVIDENCE / "selection_freeze_tuji_v9_v3_memory_safe_callfix.json"
DEFAULT_TEST_CACHE = EVIDENCE / "author_test_cache_tuji_v9_v3_memory_safe_callfix.pt"
DEFAULT_RESULT = EVIDENCE / "author_test_result_tuji_v9_v3_memory_safe_callfix.json"
FINAL_MODELS = ROOT / "models" / "v9_tuji_final_v3_memory_safe_callfix"

EXECUTION_CONTRACT_VERSION = "tuji_v9_execution_contract_v3_memory_safe_callfix_20260812"
EXTENSION_MEMBER_COUNTS = v2.EXTENSION_MEMBER_COUNTS


def _candidate(value: str) -> str:
    name = str(value)
    if name not in v1.CANDIDATES:
        raise ValueError(f"unknown V9 candidate: {name}")
    return name


def _require_path(path: str | Path, expected: str | Path, *, label: str) -> Path:
    result = Path(path).resolve()
    frozen = Path(expected).resolve()
    if result != frozen:
        raise ValueError(f"{label} must be the frozen protocol path: {frozen}")
    return result


def _cache_path(candidate: str, members: int) -> Path:
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_cache_{_candidate(candidate)}_through_E{count}_v3_memory_safe_callfix.pt"


def _grid_path(candidate: str, members: int) -> Path:
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_grid_{_candidate(candidate)}_through_E{count}_v3_memory_safe_callfix.pt"


def _grid_summary_path(candidate: str, members: int) -> Path:
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_grid_{_candidate(candidate)}_through_E{count}_v3_memory_safe_callfix.json"


def _extension_report_path(members: int) -> Path:
    return EVIDENCE / f"selection_extension_required_through_E{int(members)}_tuji_v9_v3_callfix.json"


def _final_checkpoint_path(candidate: str, index: int) -> Path:
    member = int(index)
    if not 1 <= member <= len(v1.MEMBER_SEEDS):
        raise ValueError("member index is outside the preregistered panel")
    return FINAL_MODELS / _candidate(candidate) / f"member_{member:02d}_seed{v1.MEMBER_SEEDS[member - 1]}.pt"


def _final_radio_path(candidate: str) -> Path:
    return EVIDENCE / f"radio_memory_final_{_candidate(candidate)}_v3_memory_safe_callfix.pt"


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
        "v3_callfix_executor": Path(__file__).resolve(),
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
        "v1_execution_contract": V1_CONTRACT,
        "v2_execution_contract": V2_CONTRACT,
    }


def _manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": v1._sha256_file(path)}
        for name, path in paths.items()
    }


def _validate_predecessors(registry_path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    """Validate frozen V1/V2 antecedents before creating or using V3."""

    registry_file = Path(registry_path).resolve()
    v1_contract = v1.load_execution_contract(V1_CONTRACT, registry_path=registry_file)
    v2_contract = v2.load_execution_contract(V2_CONTRACT, registry_path=registry_file)
    if v2_contract["v1_execution_contract"]["execution_contract_sha256"] != v1_contract["execution_contract_sha256"]:
        raise ValueError("V2 no longer identifies the frozen V1 antecedent")
    unexpected_v2 = [
        name
        for name, path in v2_contract["future_output_paths"].items()
        if Path(str(path)).exists()
    ]
    if unexpected_v2:
        raise RuntimeError("V2 failed attempt unexpectedly has output artifacts: " + ", ".join(unexpected_v2))
    return v1_contract, v2_contract


def freeze_execution_contract(
    *, registry_path: str | Path = DEFAULT_REGISTRY, output: str | Path = DEFAULT_CONTRACT
) -> dict[str, object]:
    """Freeze V3 after both predecessor contracts but before any V3 outcome."""

    registry_file = Path(registry_path).resolve()
    destination = _require_path(output, DEFAULT_CONTRACT, label="V3 execution contract")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite execution contract: {destination}")
    registry = load_registry(registry_file)
    v1_contract, v2_contract = _validate_predecessors(registry_file)
    prior = v2._selection_artifact_manifest(registry=registry, v1_contract=v1_contract)
    existing = [str(path.resolve()) for path in _future_output_paths().values() if path.exists()]
    if existing:
        raise FileExistsError("V3 contract must precede all V3 outcomes: " + ", ".join(existing))
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
            {
                "id": "TUJI-V9-EXEC-A1-memory-safe-cache",
                "path": str(A1_AMENDMENT.resolve()),
                "sha256": v1._sha256_file(A1_AMENDMENT),
            },
            {
                "id": "TUJI-V9-EXEC-A2-keyword-loader-callfix",
                "path": str(A2_AMENDMENT.resolve()),
                "sha256": v1._sha256_file(A2_AMENDMENT),
            },
        ],
        "v1_execution_contract": {
            "path": str(V1_CONTRACT.resolve()),
            "sha256": v1._sha256_file(V1_CONTRACT),
            "execution_contract_sha256": v1_contract["execution_contract_sha256"],
        },
        "v2_failed_execution": {
            "path": str(V2_CONTRACT.resolve()),
            "sha256": v1._sha256_file(V2_CONTRACT),
            "execution_contract_sha256": v2_contract["execution_contract_sha256"],
            "created_cache_artifact": False,
            "read_author_test_rss": False,
            "read_author_test_labels": False,
            "failure": "TypeError from passing V1 keyword-only checkpoint-loader device argument positionally before cache serialization",
        },
        "preexisting_v1_selection_artifacts": prior,
        "test_model_outcome_accesses_before_v3_freeze": 0,
        "memory_safe_execution": {
            "member_residency": "one member at a time",
            "member_checkpoint_source": "verified immutable V1 selection checkpoint",
            "query_chunk_size": v2.QUERY_CHUNK_SIZE,
            "scoring_chunk_size": v2.SCORING_CHUNK_SIZE,
            "scientific_parameters_changed": False,
        },
        "loader_compatibility": {
            "selection": "call frozen V1 load_front_checkpoint with registry=, contract=, device= keyword arguments",
            "final": "reconstruct V2 final checkpoint from its unchanged model_state/config then apply V2 final payload validation",
            "read_only_interface_check": "one V1 selection checkpoint, CPU only, no test RSS or labels",
        },
        "test_gate": {
            "selection_commands": "author_training_only",
            "author_test_cache": "requires V3 selection freeze, V3 final refit panel, and V3 final marked memory",
            "author_test_query_input": "RSS_testing.csv only",
            "author_test_labels": "only after prediction cache exists, for score/grouping/device stratification",
        },
        "selection_grid": {
            "E": list(v1.E_GRID),
            "K": list(v1.K_GRID),
            "lambda": list(v1.LAMBDA_GRID),
            "member_seeds": list(v1.MEMBER_SEEDS),
            "extensions": list(EXTENSION_MEMBER_COUNTS),
            "member_rule": "preregistered V1 seed prefix",
        },
        "execution_sources": _manifest(_source_paths()),
        "future_output_paths": {name: str(path.resolve()) for name, path in _future_output_paths().items()},
    }
    contract["execution_contract_sha256"] = v1._canonical_json_sha256(contract, omit="execution_contract_sha256")
    v1._write_json_new(destination, contract)
    return contract


def load_execution_contract(
    path: str | Path = DEFAULT_CONTRACT,
    *, registry_path: str | Path | None = None,
) -> dict[str, object]:
    contract_file = _require_path(path, DEFAULT_CONTRACT, label="V3 execution contract")
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
        raise ValueError("not a TUJI V9 memory-safe V3 call-fix execution contract")
    if contract.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("V3 protocol version mismatch")
    if contract.get("execution_contract_sha256") != v1._canonical_json_sha256(contract, omit="execution_contract_sha256"):
        raise ValueError("V3 execution-contract self-hash failed")
    frozen_registry = Path(str(contract["registry"])).resolve()
    if registry_path is not None and Path(registry_path).resolve() != frozen_registry:
        raise ValueError("caller registry differs from V3 contract")
    registry = load_registry(frozen_registry)
    if registry["registry_sha256"] != contract.get("registry_sha256") or registry["design_sha256"] != contract.get("design_sha256"):
        raise ValueError("V3 registry or design hash mismatch")
    records = contract.get("execution_sources")
    if not isinstance(records, Mapping):
        raise ValueError("V3 source manifest missing")
    for name, record in records.items():
        if not isinstance(record, Mapping) or v1._sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V3 source changed after freeze: {name}")
    v1_contract, v2_contract = _validate_predecessors(frozen_registry)
    if v1_contract["execution_contract_sha256"] != contract["v1_execution_contract"]["execution_contract_sha256"]:
        raise ValueError("V1 contract identity mismatch")
    if v2_contract["execution_contract_sha256"] != contract["v2_failed_execution"]["execution_contract_sha256"]:
        raise ValueError("V2 contract identity mismatch")
    current_selection = v2._selection_artifact_manifest(registry=registry, v1_contract=v1_contract)
    if current_selection != contract["preexisting_v1_selection_artifacts"]:
        raise ValueError("immutable V1 selection artifacts changed")
    return contract


def _load_one_front_v3(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    expected_contract_sha256: str,
    expected_phase: str,
    expected_candidate: str,
    expected_role: str,
    expected_index: int,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Exact V1 selection-loader ABI, used one resident member at a time."""

    v1_contract = v1.load_execution_contract(V1_CONTRACT)
    if str(v1_contract["execution_contract_sha256"]) != str(expected_contract_sha256):
        raise ValueError("stream requested an unexpected selection contract")
    model, payload = v1.load_front_checkpoint(
        path, registry=registry, contract=v1_contract, device=torch.device("cpu")
    )
    if (
        payload.get("registry_sha256") != registry["registry_sha256"]
        or payload.get("execution_contract_sha256") != expected_contract_sha256
        or payload.get("phase") != expected_phase
        or payload.get("candidate") != expected_candidate
        or payload.get("refit_role") != expected_role
        or int(payload.get("member_index", -1)) != int(expected_index)
        or int(payload.get("member_seed", -1)) != int(v1.MEMBER_SEEDS[expected_index - 1])
    ):
        raise ValueError(f"invalid V1 selection checkpoint: {path}")
    return model, payload


def _reconstruct_v2_final_model(
    path: str | Path, *, registry: Mapping[str, object], contract: Mapping[str, object]
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Model reconstruction equivalent to V1 loader, admitting only V2 final schema."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("checkpoint_schema") != "tuji_coherent_marked_front_v9_v2_memory_safe"
        or config.get("model_schema") != "coherent_marked_front_v9"
        or payload.get("registry_sha256") != registry["registry_sha256"]
        or payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]
    ):
        raise ValueError(f"not a compatible V2 final checkpoint: {path}")
    state = payload["model_state"]
    candidate = _candidate(str(payload["candidate"]))
    geometry = None
    if candidate == "geometry_invariant":
        geometry = v1.build_delaunay_geometry(
            state["anchor_coords"], state["anchor_buildings"], state["anchor_floors"]
        )
    model = v1.AnchorStudentTModel(
        n_aps=int(state["prototype_rss"].shape[1]),
        anchor_coords=state["anchor_coords"],
        anchor_buildings=state["anchor_buildings"],
        anchor_floors=state["anchor_floors"],
        seen_ap_mask=state["encoder.seen_ap_mask"],
        prototype_rss=state["prototype_rss"],
        ap_building_owner=state["ap_building_owner"],
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        max_visible=int(config["max_visible"]),
        dropout=float(config["dropout"]),
        visible_threshold=float(config["visible_threshold"]),
        student_df=float(config["student_df"]),
        min_scale_m=float(config["min_scale_m"]),
        max_residual_m=float(v1.HISTORICAL_FRONT["max_residual_m"]),
        scan_feature_mode=str(config["scan_feature_mode"]),
        prototype_metric=str(config["prototype_metric"]),
        local_residual_transform=None if geometry is None else geometry["residual_transform"],
    )
    model.load_state_dict(state, strict=True)
    model.to(torch.device("cpu")).eval()
    return model, payload


def _load_v2_final_front_v3(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    freeze: Mapping[str, object],
    expected_index: int,
) -> tuple[torch.nn.Module, dict[str, object]]:
    model, payload = _reconstruct_v2_final_model(path, registry=registry, contract=contract)
    selected = freeze["selected"]
    if (
        payload.get("freeze_sha256") != freeze["freeze_sha256"]
        or payload.get("phase") != "final"
        or payload.get("refit_role") != "final_refit"
        or payload.get("candidate") != selected["candidate"]
        or int(payload.get("member_index", -1)) != int(expected_index)
        or int(payload.get("member_seed", -1)) != int(v1.MEMBER_SEEDS[expected_index - 1])
    ):
        raise ValueError(f"invalid V3 final checkpoint: {path}")
    return model, payload


def _install_v2_runtime() -> None:
    """Redirect immutable V2 mathematics to V3 paths and fixed loaders."""

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
    v2._load_one_front = _load_one_front_v3
    v2._load_v2_final_front = _load_v2_final_front_v3
    v2.load_execution_contract = load_execution_contract


def audit_execution_contract(
    *, registry_path: str | Path = DEFAULT_REGISTRY, contract_path: str | Path = DEFAULT_CONTRACT
) -> dict[str, object]:
    """Pre-outcome V3 audit, including an actual CPU ABI check of member 1."""

    contract = load_execution_contract(contract_path, registry_path=registry_path)
    _install_v2_runtime()
    v2_audit = v2.audit_execution_contract(registry_path=registry_path, contract_path=contract_path)
    source = inspect.getsource(_load_one_front_v3)
    if "registry=registry" not in source or "contract=v1_contract" not in source or "device=torch.device" not in source:
        raise RuntimeError("V3 selection loader does not expose the exact V1 keyword ABI")
    registry = load_registry(registry_path)
    v1_contract = v1.load_execution_contract(V1_CONTRACT, registry_path=registry_path)
    model, payload = _load_one_front_v3(
        v1.selection_checkpoint_path("geometry_invariant", 1),
        registry=registry,
        expected_contract_sha256=str(v1_contract["execution_contract_sha256"]),
        expected_phase="selection",
        expected_candidate="geometry_invariant",
        expected_role="selection_refit",
        expected_index=1,
    )
    anchors = int(model.n_anchors)
    del model, payload
    gc.collect()
    return {
        "status": "passed_memory_safe_callfix_pre_outcome_audit",
        "registry_sha256": contract["registry_sha256"],
        "v1_execution_contract_sha256": contract["v1_execution_contract"]["execution_contract_sha256"],
        "v2_failed_execution_contract_sha256": contract["v2_failed_execution"]["execution_contract_sha256"],
        "v3_execution_contract_sha256": contract["execution_contract_sha256"],
        "v1_selection_artifacts_verified": len(contract["preexisting_v1_selection_artifacts"]),
        "v3_future_outputs_absent": True,
        "one_member_gpu_residency": True,
        "author_test_cache_requires_freeze_before_rss": True,
        "author_test_cache_does_not_load_labels": True,
        "selection_loader_keyword_abi_checked": True,
        "selection_loader_cpu_anchors": anchors,
        "v2_audit_status": v2_audit["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-execution")
    freeze.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    freeze.add_argument("--output", default=str(DEFAULT_CONTRACT))
    audit = commands.add_parser("audit-execution")
    audit.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    audit.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    extension = commands.add_parser("train-v1-extension-member")
    extension.add_argument("--candidate", choices=v1.CANDIDATES, required=True)
    extension.add_argument("--member-index", type=int, required=True)
    extension.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    extension.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    extension.add_argument("--device", default="cuda")
    extension.add_argument("--log-every", type=int, default=5)
    cache = commands.add_parser("build-development-cache")
    cache.add_argument("--candidate", choices=v1.CANDIDATES, required=True)
    cache.add_argument("--members", type=int, default=v1.INITIAL_MEMBER_COUNT)
    cache.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    cache.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    cache.add_argument("--device", default="cuda")
    grid = commands.add_parser("grid")
    grid.add_argument("--candidate", choices=v1.CANDIDATES, required=True)
    grid.add_argument("--members", type=int, default=v1.INITIAL_MEMBER_COUNT)
    grid.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    grid.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    grid.add_argument("--device", default="cuda")
    select = commands.add_parser("freeze-selection")
    select.add_argument("--members", type=int, default=v1.INITIAL_MEMBER_COUNT)
    select.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    select.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    select.add_argument("--output", default=str(DEFAULT_FREEZE))
    final = commands.add_parser("train-final-member")
    final.add_argument("--member-index", type=int, required=True)
    final.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    final.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    final.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    final.add_argument("--device", default="cuda")
    final.add_argument("--log-every", type=int, default=5)
    radio = commands.add_parser("fit-final-radio")
    radio.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    radio.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    radio.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    test_cache = commands.add_parser("cache-author-test")
    test_cache.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    test_cache.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    test_cache.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    test_cache.add_argument("--device", default="cuda")
    evaluate = commands.add_parser("evaluate-author-test")
    evaluate.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    evaluate.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    evaluate.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    evaluate.add_argument("--cache", default=str(DEFAULT_TEST_CACHE))
    evaluate.add_argument("--output", default=str(DEFAULT_RESULT))
    evaluate.add_argument("--device")
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
        else:  # pragma: no cover - argparse closes this branch.
            raise RuntimeError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
