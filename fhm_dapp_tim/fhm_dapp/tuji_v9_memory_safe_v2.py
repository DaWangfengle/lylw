"""Memory-safe, auditable continuation of the frozen TUJI1 V9 experiment.

This is the execution-only A1 amendment to ``tuji_v9_experiment``.  It keeps
the original V1 source and all V1 selection artifacts immutable, but streams
one already-frozen member through the GPU at a time.  Its only purpose is to
avoid the V1 cache step's eight-model simultaneous-residency OOM on a 6 GiB
GPU.  It changes no data role, front candidate, learned weight, grid, random
seed, statistical rule, or author-test access gate.
"""

from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import tuji_v9_experiment as v1
from fhm_dapp_tim.fhm_dapp.tuji_v9_protocol import (
    BOOTSTRAP_REPETITIONS,
    DEFAULT_FREEZE,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    DEFAULT_TEST_CACHE,
    PAIRED_INFERENCE_SEED,
    PROTOCOL_VERSION,
    SCORING_SEED,
    STUDENT_DF,
    _normalized_rss,
    load_registry,
    registered_role_keys,
    registered_rows,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v9_tuji"
V1_CONTRACT = EVIDENCE / "execution_contract_tuji_v9.json"
AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A1_MEMORY_SAFE.md"
DEFAULT_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v2_memory_safe.json"
FINAL_MODELS = ROOT / "models" / "v9_tuji_final_v2_memory_safe"

EXECUTION_CONTRACT_VERSION = "tuji_v9_execution_contract_v2_memory_safe_20260812"
QUERY_CHUNK_SIZE = 16
SCORING_CHUNK_SIZE = 16
EXTENSION_MEMBER_COUNTS = (8, 10, 12)

LocationKey = tuple[int, int, float, float]


def _candidate(candidate: str) -> str:
    name = str(candidate)
    if name not in v1.CANDIDATES:
        raise ValueError(f"unknown V9 candidate: {name}")
    return name


def _member_index(index: int) -> int:
    value = int(index)
    if not 1 <= value <= len(v1.MEMBER_SEEDS):
        raise ValueError("member index is outside the preregistered panel")
    return value


def _require_path(path: str | Path, expected: str | Path, *, label: str) -> Path:
    result = Path(path).resolve()
    fixed = Path(expected).resolve()
    if result != fixed:
        raise ValueError(f"{label} must be the frozen protocol path: {fixed}")
    return result


def _v2_cache_path(candidate: str, members: int) -> Path:
    name = _candidate(candidate)
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_cache_{name}_through_E{count}_v2_memory_safe.pt"


def _v2_grid_path(candidate: str, members: int) -> Path:
    name = _candidate(candidate)
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_grid_{name}_through_E{count}_v2_memory_safe.pt"


def _v2_grid_summary_path(candidate: str, members: int) -> Path:
    name = _candidate(candidate)
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_grid_{name}_through_E{count}_v2_memory_safe.json"


def _extension_report_path(members: int) -> Path:
    return EVIDENCE / f"selection_extension_required_through_E{int(members)}_tuji_v9_v2.json"


def _final_checkpoint_path(candidate: str, index: int) -> Path:
    name = _candidate(candidate)
    member = _member_index(index)
    return FINAL_MODELS / name / f"member_{member:02d}_seed{v1.MEMBER_SEEDS[member - 1]}.pt"


def _final_radio_path(candidate: str) -> Path:
    return EVIDENCE / f"radio_memory_final_{_candidate(candidate)}_v2_memory_safe.pt"


def _future_output_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {
        "selection_freeze": Path(DEFAULT_FREEZE),
        "author_test_cache": Path(DEFAULT_TEST_CACHE),
        "author_test_result": Path(DEFAULT_RESULT),
    }
    for candidate in v1.CANDIDATES:
        paths[f"final_radio_{candidate}"] = _final_radio_path(candidate)
        for index in range(1, len(v1.MEMBER_SEEDS) + 1):
            paths[f"final_{candidate}_{index}"] = _final_checkpoint_path(candidate, index)
        for count in EXTENSION_MEMBER_COUNTS:
            paths[f"cache_{candidate}_E{count}"] = _v2_cache_path(candidate, count)
            paths[f"grid_{candidate}_E{count}"] = _v2_grid_path(candidate, count)
            paths[f"grid_summary_{candidate}_E{count}"] = _v2_grid_summary_path(candidate, count)
    for count in EXTENSION_MEMBER_COUNTS[:-1]:
        paths[f"extension_report_E{count}"] = _extension_report_path(count)
    return paths


def _source_paths() -> dict[str, Path]:
    module_dir = Path(__file__).resolve().parent
    return {
        "v2_memory_safe_executor": Path(__file__).resolve(),
        "v1_executor": module_dir / "tuji_v9_experiment.py",
        "tuji_protocol": module_dir / "tuji_v9_protocol.py",
        "uji_v9_core": module_dir / "uji_v9_experiment.py",
        "marked_math": module_dir / "coherent_marked_v9.py",
        "marked_memory": module_dir / "experiment_v71.py",
        "safe_artifact_primitives": module_dir / "experiment_v7.py",
        "front_model": module_dir / "model.py",
        "anchor_memory": module_dir / "anchor.py",
        "front_training": module_dir / "train_anchor.py",
        "execution_amendment": AMENDMENT,
        "v1_execution_contract": V1_CONTRACT,
    }


def _manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": v1._sha256_file(path)}
        for name, path in paths.items()
    }


def _load_v1_contract(registry_path: str | Path) -> dict[str, object]:
    """Validate V1 source before V2 reuses any V1 selection output."""

    return v1.load_execution_contract(V1_CONTRACT, registry_path=registry_path)


def _selection_artifact_manifest(
    *, registry: Mapping[str, object], v1_contract: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    required_hash = str(v1_contract["execution_contract_sha256"])
    for candidate in v1.CANDIDATES:
        for index in range(1, v1.INITIAL_MEMBER_COUNT + 1):
            path = v1.selection_checkpoint_path(candidate, index)
            if not path.is_file():
                raise FileNotFoundError(f"V1 selection member missing: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload.get("registry_sha256") != registry["registry_sha256"]
                or payload.get("execution_contract_sha256") != required_hash
                or payload.get("candidate") != candidate
                or int(payload.get("member_index", -1)) != index
                or payload.get("phase") != "selection"
                or payload.get("refit_role") != "selection_refit"
            ):
                raise ValueError(f"invalid V1 selection checkpoint: {path}")
            output[f"selection_{candidate}_{index}"] = {
                "path": str(path.resolve()), "sha256": v1._sha256_file(path)
            }
        radio = v1.radio_memory_path(candidate, "selection")
        if not radio.is_file():
            raise FileNotFoundError(f"V1 selection radio memory missing: {radio}")
        payload = v1._load_radio_memory(
            radio,
            registry=registry,
            contract=v1_contract,
            candidate=candidate,
            phase="selection",
        )
        if payload.get("refit_role") != "selection_refit":
            raise ValueError("V1 radio memory role mismatch")
        output[f"selection_radio_{candidate}"] = {
            "path": str(radio.resolve()), "sha256": v1._sha256_file(radio)
        }
    return output


def freeze_execution_contract(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_CONTRACT,
) -> dict[str, object]:
    """Freeze the A1 memory-safe executor before any development outcome."""

    registry_file = Path(registry_path).resolve()
    destination = _require_path(output, DEFAULT_CONTRACT, label="V2 execution contract")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite execution contract: {destination}")
    registry = load_registry(registry_file)
    v1_contract = _load_v1_contract(registry_file)
    prior = _selection_artifact_manifest(registry=registry, v1_contract=v1_contract)
    existing = [
        str(path.resolve()) for path in _future_output_paths().values() if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "V2 execution contract must precede every V2 outcome artifact: "
            + ", ".join(existing)
        )
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
        "amendment_id": "TUJI-V9-EXEC-A1-memory-safe-cache",
        "amendment_status": "post-failure pre-outcome execution-only correction",
        "v1_execution_contract": {
            "path": str(V1_CONTRACT.resolve()),
            "sha256": v1._sha256_file(V1_CONTRACT),
            "execution_contract_sha256": v1_contract["execution_contract_sha256"],
        },
        "preexisting_v1_selection_artifacts": prior,
        "test_model_outcome_accesses_before_v2_freeze": 0,
        "failed_v1_cache_attempt": {
            "created_cache_artifact": False,
            "read_author_test_rss": False,
            "read_author_test_labels": False,
            "reason": "eight members simultaneously resident on a 6 GiB GPU exhausted memory before cache serialization",
        },
        "memory_safe_execution": {
            "member_residency": "one member at a time",
            "member_checkpoint_source": "verified immutable V1 selection checkpoint",
            "query_chunk_size": QUERY_CHUNK_SIZE,
            "scoring_chunk_size": SCORING_CHUNK_SIZE,
            "device": "GPU one member at a time; outputs immediately copied to CPU",
            "scientific_parameters_changed": False,
        },
        "test_gate": {
            "selection_commands": "author_training_only",
            "author_test_cache": "requires V2 selection freeze, V2 final refit panel, and V2 final marked memory",
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
        "future_output_paths": {
            name: str(path.resolve()) for name, path in _future_output_paths().items()
        },
    }
    contract["execution_contract_sha256"] = v1._canonical_json_sha256(
        contract, omit="execution_contract_sha256"
    )
    v1._write_json_new(destination, contract)
    return contract


def load_execution_contract(
    path: str | Path = DEFAULT_CONTRACT,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, object]:
    contract_file = _require_path(path, DEFAULT_CONTRACT, label="V2 execution contract")
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
        raise ValueError("not TUJI V9 memory-safe V2 execution contract")
    if contract.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("V2 protocol version mismatch")
    if contract.get("execution_contract_sha256") != v1._canonical_json_sha256(
        contract, omit="execution_contract_sha256"
    ):
        raise ValueError("V2 execution contract self-hash failed")
    frozen_registry = Path(str(contract["registry"])).resolve()
    if registry_path is not None and Path(registry_path).resolve() != frozen_registry:
        raise ValueError("caller registry differs from V2 contract")
    registry = load_registry(frozen_registry)
    if registry["registry_sha256"] != contract.get("registry_sha256"):
        raise ValueError("V2 registry hash mismatch")
    if registry["design_sha256"] != contract.get("design_sha256"):
        raise ValueError("V2 design hash mismatch")
    records = contract.get("execution_sources")
    if not isinstance(records, Mapping):
        raise ValueError("V2 source manifest missing")
    for name, record in records.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid V2 source record: {name}")
        if v1._sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V2 source changed after freeze: {name}")
    v1_record = contract["v1_execution_contract"]
    if v1._sha256_file(v1_record["path"]) != v1_record["sha256"]:
        raise ValueError("V1 execution contract changed after V2 freeze")
    verified_v1 = _load_v1_contract(frozen_registry)
    if verified_v1["execution_contract_sha256"] != v1_record["execution_contract_sha256"]:
        raise ValueError("V1 execution-contract identity mismatch")
    for name, record in contract["preexisting_v1_selection_artifacts"].items():
        if v1._sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V1 selection artifact changed: {name}")
    return contract


def audit_execution_contract(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_CONTRACT,
) -> dict[str, object]:
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    existing = [name for name, path in _future_output_paths().items() if path.exists()]
    if existing:
        raise FileExistsError("V2 pre-outcome audit found artifacts: " + ", ".join(existing))
    source = inspect.getsource(cache_author_test)
    if source.index("load_selection_freeze") > source.index("_load_test_rss_only"):
        raise RuntimeError("V2 author-test RSS load is not guarded by selection freeze")
    if "_load_test_labels" in source:
        raise RuntimeError("V2 author-test cache must not load test labels")
    return {
        "status": "passed_memory_safe_pre_outcome_audit",
        "registry_sha256": contract["registry_sha256"],
        "v1_execution_contract_sha256": contract["v1_execution_contract"]["execution_contract_sha256"],
        "v2_execution_contract_sha256": contract["execution_contract_sha256"],
        "v1_selection_artifacts_verified": len(contract["preexisting_v1_selection_artifacts"]),
        "v2_future_outputs_absent": True,
        "one_member_gpu_residency": True,
        "author_test_cache_requires_freeze_before_rss": True,
        "author_test_cache_does_not_load_labels": True,
    }


def _v1_context(
    *, registry_path: str | Path, contract_path: str | Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    registry = load_registry(registry_path)
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    v1_contract = _load_v1_contract(registry_path)
    if v1_contract["execution_contract_sha256"] != contract["v1_execution_contract"]["execution_contract_sha256"]:
        raise ValueError("V1 contract differs from V2's frozen antecedent")
    return registry, contract, v1_contract


def _selection_paths(candidate: str, members: int) -> list[Path]:
    name = _candidate(candidate)
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return [v1.selection_checkpoint_path(name, index) for index in range(1, count + 1)]


def _load_one_front(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    expected_contract_sha256: str,
    expected_phase: str,
    expected_candidate: str,
    expected_role: str,
    expected_index: int,
) -> tuple[torch.nn.Module, dict[str, object]]:
    model, payload = v1.load_front_checkpoint(path, torch.device("cpu"))
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError(f"checkpoint registry mismatch: {path}")
    if payload.get("execution_contract_sha256") != expected_contract_sha256:
        raise ValueError(f"checkpoint execution-contract mismatch: {path}")
    if payload.get("phase") != expected_phase:
        raise ValueError(f"checkpoint phase mismatch: {path}")
    if payload.get("candidate") != expected_candidate:
        raise ValueError(f"checkpoint candidate mismatch: {path}")
    if payload.get("refit_role") != expected_role:
        raise ValueError(f"checkpoint refit-role mismatch: {path}")
    if int(payload.get("member_index", -1)) != int(expected_index):
        raise ValueError("member order is not the registered seed prefix")
    if int(payload.get("member_seed", -1)) != int(v1.MEMBER_SEEDS[expected_index - 1]):
        raise ValueError("checkpoint member seed mismatch")
    return model, payload


@torch.no_grad()
def _stream_panel_predictions(
    *,
    paths: Sequence[Path],
    query: torch.Tensor,
    registry: Mapping[str, object],
    expected_contract_sha256: str,
    phase: str,
    candidate: str,
    role: str,
    device: torch.device,
) -> dict[str, object]:
    """Load, evaluate, and release one member at a time.

    This is the sole A1 difference from V1's simultaneous resident panel.
    """

    name = _candidate(candidate)
    if not paths:
        raise ValueError("a member panel cannot be empty")
    rows = int(len(query))
    member_count = len(paths)
    logits_cache: torch.Tensor | None = None
    location_cache: torch.Tensor | None = None
    scale_cache: torch.Tensor | None = None
    first_keys = None
    first_origin = None
    anchor_coords = None
    anchor_buildings = None
    anchor_floors = None
    for panel_index, path in enumerate(paths, start=1):
        model, payload = _load_one_front(
            path,
            registry=registry,
            expected_contract_sha256=expected_contract_sha256,
            expected_phase=phase,
            expected_candidate=name,
            expected_role=role,
            expected_index=panel_index,
        )
        if first_keys is None:
            first_keys = payload["anchor_coordinate_keys"]
            first_origin = payload["coord_origin"]
            anchors = int(model.n_anchors)
            logits_cache = torch.empty(member_count, rows, anchors)
            location_cache = torch.empty(member_count, rows, anchors, 2)
            scale_cache = torch.empty_like(location_cache)
            anchor_coords = model.anchor_coords.detach().cpu()
            anchor_buildings = model.anchor_buildings.detach().cpu()
            anchor_floors = model.anchor_floors.detach().cpu()
        else:
            if payload["anchor_coordinate_keys"] != first_keys:
                raise ValueError("members do not share an anchor dictionary")
            if not torch.equal(payload["coord_origin"], first_origin):
                raise ValueError("members do not share a coordinate origin")
        model.to(device).eval()
        anchors = int(model.n_anchors)
        for start in range(0, rows, QUERY_CHUNK_SIZE):
            stop = min(start + QUERY_CHUNK_SIZE, rows)
            values = query[start:stop].to(device)
            logits, embedding = v1._front_logits_embedding(model, values, name)
            anchor_index = torch.arange(
                anchors, device=device, dtype=torch.long
            ).unsqueeze(0).expand(len(values), -1)
            location, scale = model._components(embedding, anchor_index)
            logits_cache[panel_index - 1, start:stop] = logits.detach().cpu()
            location_cache[panel_index - 1, start:stop] = location.detach().cpu()
            scale_cache[panel_index - 1, start:stop] = scale.detach().cpu()
            del values, logits, embedding, anchor_index, location, scale
        model.to(torch.device("cpu"))
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    if any(value is None for value in (logits_cache, location_cache, scale_cache, first_keys, first_origin, anchor_coords, anchor_buildings, anchor_floors)):
        raise RuntimeError("memory-safe panel prediction did not initialize")
    return {
        "query": query.float().cpu(),
        "member_logits": logits_cache,
        "member_locations": location_cache,
        "member_scales": scale_cache,
        "anchor_coords": anchor_coords,
        "anchor_buildings": anchor_buildings,
        "anchor_floors": anchor_floors,
        "anchor_coordinate_keys": first_keys,
        "coord_origin": first_origin,
        "randomness": v1.common_student_t_randomness(
            rows,
            draws=v1.SCORING_DRAWS,
            degrees_freedom=STUDENT_DF,
            seed=SCORING_SEED,
        ),
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "member_residency": "sequential_gpu_one_member_at_a_time",
    }


def _coordinate_groups(keys: Sequence[LocationKey]) -> tuple[torch.Tensor, tuple[LocationKey, ...]]:
    unique = tuple(sorted(set(keys)))
    mapping = {key: index for index, key in enumerate(unique)}
    return torch.tensor([mapping[key] for key in keys], dtype=torch.long), unique


def build_development_cache(
    *,
    candidate: str,
    members: int = v1.INITIAL_MEMBER_COUNT,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_CONTRACT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Build a training-only V2 cache with sequential GPU member residency."""

    registry, contract, v1_contract = _v1_context(
        registry_path=registry_path, contract_path=contract_path
    )
    name = _candidate(candidate)
    count = int(members)
    destination = _v2_cache_path(name, count)
    if destination.exists():
        raise FileExistsError(destination)
    paths = _selection_paths(name, count)
    data = v1._load_training_data(registry)
    rows = registered_rows(registry, "development", data)
    radio_path = v1.radio_memory_path(name, "selection")
    radio = v1._load_radio_memory(
        radio_path,
        registry=registry,
        contract=v1_contract,
        candidate=name,
        phase="selection",
    )
    represented = v1._represented_rss(data["rss"], name)
    cache = _stream_panel_predictions(
        paths=paths,
        query=represented[rows],
        registry=registry,
        expected_contract_sha256=str(v1_contract["execution_contract_sha256"]),
        phase="selection",
        candidate=name,
        role="selection_refit",
        device=torch.device(device_name),
    )
    if tuple(radio["memory"]["anchor_keys"]) != tuple(cache["anchor_coordinate_keys"]):
        raise ValueError("V1 marked memory and V1 member anchors are not aligned")
    origin = cache["coord_origin"].numpy().astype(np.float32)
    role_keys = [data["keys"][int(row)] for row in rows]
    group_ids, group_keys = _coordinate_groups(role_keys)
    payload: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema": "tuji_v9_development_cache_v2_memory_safe",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "v1_execution_contract_sha256": v1_contract["execution_contract_sha256"],
        "candidate": name,
        "phase": "selection",
        "role": "development",
        "members": count,
        "role_coordinate_sha256": registry["roles"]["development"]["coordinate_key_sha256"],
        "row_index_sha256": registry["roles"]["development"]["row_index_sha256"],
        "row_indices": torch.from_numpy(rows).long(),
        "coordinate_group_ids": group_ids,
        "coordinate_group_keys": group_keys,
        "target_xy": torch.from_numpy(
            (data["raw_xy"][rows] - origin[None, :]).astype(np.float32)
        ),
        "target_building": torch.zeros(len(rows), dtype=torch.long),
        "target_floor": torch.zeros(len(rows), dtype=torch.long),
        "radio_fields": radio["memory"]["fields"],
        "radio_degrees_freedom": radio["memory"]["degrees_freedom"],
        "radio_memory_path": str(radio_path.resolve()),
        "radio_memory_sha256": v1._sha256_file(radio_path),
        "member_paths": tuple(str(path.resolve()) for path in paths),
        "member_sha256": tuple(v1._sha256_file(path) for path in paths),
        "test_model_outcome_accesses": 0,
        **cache,
    }
    v1._save_torch_new(destination, payload)
    return {
        "cache": str(destination.resolve()),
        "sha256": v1._sha256_file(destination),
        "candidate": name,
        "members": count,
        "rows": int(len(rows)),
        "coordinate_groups": len(group_keys),
        "anchors": int(cache["member_logits"].shape[2]),
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "test_model_outcome_accesses": 0,
    }


def _load_development_cache(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    candidate: str,
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("cache_schema") != "tuji_v9_development_cache_v2_memory_safe"
    ):
        raise ValueError("not a V2 memory-safe TUJI development cache")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("development cache registry mismatch")
    if payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]:
        raise ValueError("development cache execution-contract mismatch")
    if payload.get("candidate") != candidate or payload.get("role") != "development":
        raise ValueError("development cache candidate/role mismatch")
    role = registry["roles"]["development"]
    if payload.get("role_coordinate_sha256") != role["coordinate_key_sha256"]:
        raise ValueError("development cache coordinate hash mismatch")
    if payload.get("row_index_sha256") != role["row_index_sha256"]:
        raise ValueError("development cache row hash mismatch")
    return payload


def evaluate_development_grid(
    *,
    candidate: str,
    members: int = v1.INITIAL_MEMBER_COUNT,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_CONTRACT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Evaluate the unchanged joint V9 grid on training-only development data."""

    registry, contract, _v1_contract = _v1_context(
        registry_path=registry_path, contract_path=contract_path
    )
    name = _candidate(candidate)
    count = int(members)
    source = _v2_cache_path(name, count)
    destination = _v2_grid_path(name, count)
    summary_destination = _v2_grid_summary_path(name, count)
    if destination.exists() or summary_destination.exists():
        raise FileExistsError("refusing to overwrite V2 development grid")
    cache = _load_development_cache(source, registry=registry, contract=contract, candidate=name)
    available = int(cache["member_logits"].shape[0])
    if available != count:
        raise ValueError("cache member count differs from requested grid")
    configurations: list[dict[str, object]] = []
    inherited = None
    first_new_e = 1
    if count > v1.INITIAL_MEMBER_COUNT:
        previous = count - 2
        base_path = _v2_grid_path(name, previous)
        base = _load_grid(base_path, registry=registry, contract=contract, candidate=name)
        base_cache = _load_development_cache(
            _v2_cache_path(name, previous), registry=registry, contract=contract, candidate=name
        )
        if tuple(base_cache["member_sha256"]) != tuple(cache["member_sha256"][:previous]):
            raise ValueError("extension cache did not preserve V1 member prefix")
        for key in ("role_coordinate_sha256", "row_index_sha256", "radio_memory_sha256"):
            if base_cache[key] != cache[key]:
                raise ValueError(f"extension changed frozen field: {key}")
        configurations = list(base["configurations"])
        first_new_e = previous + 1
        inherited = {
            "path": str(base_path.resolve()),
            "sha256": v1._sha256_file(base_path),
            "available_members": previous,
        }
    device = torch.device(device_name)
    for e_count in range(first_new_e, available + 1):
        full_joint = v1.uniform_member_anchor_prior(cache["member_logits"], members=e_count)
        for top_k in v1.K_GRID:
            projection = v1.topk_prior_projection(full_joint, top_k=top_k)
            marked, nuisance = v1._marked_energy_for_projection(
                cache, projection, candidate=name, device=device, batch_size=SCORING_CHUNK_SIZE
            )
            for lambda_value in v1.LAMBDA_GRID:
                metrics = v1._configuration_metrics(
                    cache,
                    projection,
                    marked,
                    members=e_count,
                    lambda_value=float(lambda_value),
                    device=device,
                    evaluation_batch_size=SCORING_CHUNK_SIZE,
                )
                configurations.append(metrics)
                print(
                    json.dumps(
                        {
                            "candidate": name,
                            "E": e_count,
                            "K": top_k,
                            "lambda": lambda_value,
                            "macro_energy_m": metrics["coordinate_macro_energy_score_2d_m"],
                            "macro_mae_m": metrics["coordinate_macro_mae_2d_m"],
                        }, allow_nan=False
                    ), flush=True,
                )
            del marked, projection
            if nuisance and device.type == "cuda":
                torch.cuda.empty_cache()
        del full_joint
    payload: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "grid_schema": "tuji_v9_joint_development_grid_v2_memory_safe",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "candidate": name,
        "cache": str(source.resolve()),
        "cache_sha256": v1._sha256_file(source),
        "available_members": available,
        "inherited_base_grid": inherited,
        "coordinate_group_keys": cache["coordinate_group_keys"],
        "configurations": configurations,
        "scoring_chunk_size": SCORING_CHUNK_SIZE,
        "test_model_outcome_accesses": 0,
    }
    v1._save_torch_new(destination, payload)
    profiled = []
    for e_count in range(1, available + 1):
        selected = min(
            (row for row in configurations if int(row["E"]) == e_count),
            key=lambda row: float(row["coordinate_macro_energy_score_2d_m"]),
        )
        profiled.append({key: value for key, value in selected.items() if not isinstance(value, torch.Tensor)})
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "grid_schema": "tuji_v9_joint_development_grid_v2_memory_safe_summary",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "candidate": name,
        "grid": str(destination.resolve()),
        "grid_sha256": v1._sha256_file(destination),
        "cache_sha256": payload["cache_sha256"],
        "available_members": available,
        "profiled_by_E": profiled,
        "scoring_chunk_size": SCORING_CHUNK_SIZE,
        "test_model_outcome_accesses": 0,
    }
    v1._write_json_new(summary_destination, summary)
    return summary


def _load_grid(
    path: str | Path,
    *, registry: Mapping[str, object], contract: Mapping[str, object], candidate: str
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("grid_schema") != "tuji_v9_joint_development_grid_v2_memory_safe"
    ):
        raise ValueError("not a V2 TUJI development grid")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V2 grid registry mismatch")
    if payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]:
        raise ValueError("V2 grid execution-contract mismatch")
    if payload.get("candidate") != candidate:
        raise ValueError("V2 grid candidate mismatch")
    if v1._sha256_file(payload["cache"]) != payload["cache_sha256"]:
        raise ValueError("V2 development cache changed after grid scoring")
    return payload


def _profile_grid(grid: Mapping[str, object]) -> dict[int, dict[str, object]]:
    profile: dict[int, dict[str, object]] = {}
    for e_count in range(1, int(grid["available_members"]) + 1):
        entries = [row for row in grid["configurations"] if int(row["E"]) == e_count]
        if not entries:
            raise RuntimeError(f"grid misses E={e_count}")
        profile[e_count] = min(
            entries, key=lambda row: float(row["coordinate_macro_energy_score_2d_m"])
        )
    return profile


def select_and_freeze(
    *,
    members: int = v1.INITIAL_MEMBER_COUNT,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_CONTRACT,
    output: str | Path = DEFAULT_FREEZE,
) -> dict[str, object]:
    """Apply V9's unchanged candidate and information-saturation rule."""

    registry, contract, _v1_contract = _v1_context(
        registry_path=registry_path, contract_path=contract_path
    )
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    destination = _require_path(output, DEFAULT_FREEZE, label="selection freeze")
    if destination.exists():
        raise FileExistsError(destination)
    grids = {
        candidate: _load_grid(
            _v2_grid_path(candidate, count), registry=registry, contract=contract, candidate=candidate
        ) for candidate in v1.CANDIDATES
    }
    profiles = {candidate: _profile_grid(grid) for candidate, grid in grids.items()}
    common_max_e = min(max(profile) for profile in profiles.values())
    geometry = profiles["geometry_invariant"][common_max_e]
    historical = profiles["historical_affine"][common_max_e]
    structural_tolerance = 0.01 * float(
        profiles["historical_affine"][1]["coordinate_macro_energy_score_2d_m"]
    )
    energy_comparison = v1.simultaneous_paired_upper_bounds(
        geometry["group_energy_score_2d_m"].numpy(),
        [historical["group_energy_score_2d_m"].numpy()],
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=PAIRED_INFERENCE_SEED,
    )
    mae_comparison = v1.simultaneous_paired_upper_bounds(
        geometry["group_mae_2d_m"].numpy(),
        [historical["group_mae_2d_m"].numpy()],
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=PAIRED_INFERENCE_SEED + 100,
    )
    geometry_mae_eligible = bool(
        mae_comparison["simultaneous_upper_bounds"][0] <= v1.MAE_NONINFERIORITY_MARGIN_M
    )
    geometry_energy_eligible = bool(
        energy_comparison["mean_differences"][0] <= 0.0
        or energy_comparison["simultaneous_upper_bounds"][0] <= structural_tolerance
    )
    chosen_candidate = (
        "geometry_invariant" if geometry_mae_eligible and geometry_energy_eligible else "historical_affine"
    )
    chosen_profile = profiles[chosen_candidate]
    saturation_tolerance = 0.01 * float(
        chosen_profile[1]["coordinate_macro_energy_score_2d_m"]
    )
    saturation = v1.select_information_saturated_E(
        {e_count: row["group_energy_score_2d_m"].numpy() for e_count, row in chosen_profile.items()},
        tolerance=saturation_tolerance,
        minimum_later_values=2,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=PAIRED_INFERENCE_SEED + 200,
    )
    report: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_freeze_schema": "tuji_v9_selection_freeze_v2_memory_safe",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "candidate_comparison_at_common_E": common_max_e,
        "structural_energy_tolerance_m": structural_tolerance,
        "energy_comparison_geometry_minus_historical": energy_comparison,
        "mae_comparison_geometry_minus_historical": mae_comparison,
        "geometry_mae_noninferior": geometry_mae_eligible,
        "geometry_energy_equivalent_or_better": geometry_energy_eligible,
        "nominated_candidate": chosen_candidate,
        "saturation": saturation,
        "grid_files": {
            candidate: {
                "path": str(_v2_grid_path(candidate, count).resolve()),
                "sha256": v1._sha256_file(_v2_grid_path(candidate, count)),
            } for candidate in v1.CANDIDATES
        },
        "author_test_model_outcome_accesses_before_freeze": 0,
        "memory_safe_execution_only": True,
    }
    if not saturation["saturated"] and count < max(EXTENSION_MEMBER_COUNTS):
        next_count = count + 2
        extension = {
            **report,
            "status": "extension_required",
            "next_member_indices": list(range(count + 1, next_count + 1)),
            "next_member_count": next_count,
            "test_model_outcome_accesses": 0,
        }
        v1._write_json_new(_extension_report_path(count), extension)
        return extension
    if saturation["saturated"]:
        selected_e = int(saturation["selected_E"])
        selection_basis = "confirmed_information_saturation"
        saturation_claim = True
        amendment = None
    else:
        selected_e = min(
            chosen_profile,
            key=lambda e_count: (
                float(chosen_profile[e_count]["coordinate_macro_energy_score_2d_m"]), int(e_count)
            ),
        )
        selection_basis = "budget_censored_development_risk_minimum"
        saturation_claim = False
        amendment = {
            "id": "TUJI-V9-A1-terminal-selection",
            "timing": "after development before author test",
            "rule": "at E=12 choose global profiled development proper-risk minimum without a saturation claim",
        }
    selected_row = chosen_profile[int(selected_e)]
    freeze: dict[str, object] = {
        **report,
        "status": "selection_frozen",
        "selection_basis": selection_basis,
        "saturation_claim": saturation_claim,
        "protocol_amendment": amendment,
        "selected": {
            "candidate": chosen_candidate,
            "E": int(selected_e),
            "K": int(selected_row["K"]),
            "lambda": float(selected_row["lambda"]),
            "development_coordinate_macro_energy_score_2d_m": float(selected_row["coordinate_macro_energy_score_2d_m"]),
            "development_coordinate_macro_mae_2d_m": float(selected_row["coordinate_macro_mae_2d_m"]),
            "member_indices": list(range(1, int(selected_e) + 1)),
            "member_seeds": list(v1.MEMBER_SEEDS[: int(selected_e)]),
        },
    }
    freeze["freeze_sha256"] = v1._canonical_json_sha256(freeze, omit="freeze_sha256")
    v1._write_json_new(destination, freeze)
    return freeze


def load_selection_freeze(
    path: str | Path = DEFAULT_FREEZE,
    *, registry: Mapping[str, object], contract: Mapping[str, object]
) -> dict[str, object]:
    freeze_path = _require_path(path, DEFAULT_FREEZE, label="selection freeze")
    if not freeze_path.exists():
        raise PermissionError("final refit and author-test access require V2 selection freeze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("protocol_version") != PROTOCOL_VERSION
        or freeze.get("selection_freeze_schema") != "tuji_v9_selection_freeze_v2_memory_safe"
        or freeze.get("status") != "selection_frozen"
    ):
        raise ValueError("not a frozen V2 TUJI selection record")
    if freeze.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("selection freeze registry mismatch")
    if freeze.get("execution_contract_sha256") != contract["execution_contract_sha256"]:
        raise ValueError("selection freeze execution-contract mismatch")
    if freeze.get("freeze_sha256") != v1._canonical_json_sha256(freeze, omit="freeze_sha256"):
        raise ValueError("selection freeze self-hash failed")
    if freeze.get("author_test_model_outcome_accesses_before_freeze") != 0:
        raise ValueError("author test access preceded selection freeze")
    for record in freeze["grid_files"].values():
        if v1._sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("frozen V2 development grid changed")
    return freeze


def train_v1_selection_extension_member(
    *,
    candidate: str,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_CONTRACT,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict[str, object]:
    """Train only an extension member explicitly required by the V2 report.

    The member itself remains a V1 selection artifact: its training machinery,
    weights and checkpoint schema are unchanged.  V2 subsequently streams it
    with the same verified V1 contract as the original 1--8 prefix.
    """

    registry, _contract, _v1_contract = _v1_context(
        registry_path=registry_path, contract_path=contract_path
    )
    name = _candidate(candidate)
    index = _member_index(member_index)
    if index <= v1.INITIAL_MEMBER_COUNT:
        raise ValueError("V2 extension command is only valid for members 9--12")
    prior_count = 8 if index <= 10 else 10
    report_path = _extension_report_path(prior_count)
    if not report_path.is_file():
        raise PermissionError("registered extension report is required before member training")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "extension_required"
        or index not in report.get("next_member_indices", [])
        or report.get("registry_sha256") != registry["registry_sha256"]
    ):
        raise ValueError("extension member is not authorized by its frozen report")
    return v1.train_selection_member(
        candidate=name,
        member_index=index,
        registry_path=registry_path,
        contract_path=V1_CONTRACT,
        device_name=device_name,
        log_every=log_every,
    )


def _v2_checkpoint_payload(
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    candidate: str,
    index: int,
    selected_epoch: int,
    model: torch.nn.Module,
    memory: Mapping[str, object],
    origin: np.ndarray,
    selection_payload: Mapping[str, object],
    refit_history: Sequence[Mapping[str, object]],
    geometry: Mapping[str, object] | None,
    freeze: Mapping[str, object],
) -> dict[str, object]:
    keys = tuple(sorted(registered_role_keys(registry, "final_refit")))
    if len(keys) != int(model.n_anchors):
        raise RuntimeError("final refit role and final anchor dictionary disagree")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_schema": "tuji_coherent_marked_front_v9_v2_memory_safe",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "phase": "final",
        "refit_role": "final_refit",
        "candidate": _candidate(candidate),
        "member_index": int(index),
        "member_seed": int(v1.MEMBER_SEEDS[index - 1]),
        "selected_epoch": int(selected_epoch),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "coord_origin": torch.as_tensor(origin, dtype=torch.float32),
        "stats": dict(memory.get("stats", {})),
        "anchor_coordinate_keys": keys,
        "selection_checkpoint": str(
            v1.selection_checkpoint_path(candidate, index).resolve()
        ),
        "selection_checkpoint_sha256": v1._sha256_file(
            v1.selection_checkpoint_path(candidate, index)
        ),
        "selection_checkpoint_execution_contract_sha256": selection_payload[
            "execution_contract_sha256"
        ],
        "schedule_history": list(selection_payload["schedule_history"]),
        "refit_history": list(refit_history),
        "geometry_diagnostics": (
            None if geometry is None else dict(geometry["diagnostics"])
        ),
        "freeze_sha256": freeze["freeze_sha256"],
        "config": {
            **v1.COMMON_FRONT,
            "model_schema": "coherent_marked_front_v9",
            "scan_feature_mode": (
                "bounded_maximal_invariant"
                if candidate == "geometry_invariant"
                else "legacy"
            ),
            "prototype_metric": (
                "none" if candidate == "geometry_invariant" else "l2"
            ),
            "visible_threshold": v1.VISIBLE_THRESHOLD,
            "historical_front": dict(v1.HISTORICAL_FRONT),
        },
    }


def train_final_member(
    *,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_CONTRACT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict[str, object]:
    """Refit one frozen selected member on all author-training coordinates."""

    registry, contract, v1_contract = _v1_context(
        registry_path=registry_path, contract_path=contract_path
    )
    freeze = load_selection_freeze(freeze_path, registry=registry, contract=contract)
    selected = freeze["selected"]
    index = _member_index(member_index)
    if index not in selected["member_indices"]:
        raise ValueError("member is outside frozen deployment prefix")
    candidate = str(selected["candidate"])
    destination = _final_checkpoint_path(candidate, index)
    if destination.exists():
        raise FileExistsError(destination)
    selection_path = v1.selection_checkpoint_path(candidate, index)
    selection_model, selection_payload = _load_one_front(
        selection_path,
        registry=registry,
        expected_contract_sha256=str(v1_contract["execution_contract_sha256"]),
        expected_phase="selection",
        expected_candidate=candidate,
        expected_role="selection_refit",
        expected_index=index,
    )
    del selection_model
    gc.collect()
    epoch = int(selection_payload["selected_epoch"])
    data = v1._load_training_data(registry)
    rows = registered_rows(registry, "final_refit", data)
    device = torch.device(device_name)
    model, memory, origin, history, geometry = v1._fit_fixed(
        data,
        rows,
        candidate=candidate,
        seed=int(v1.MEMBER_SEEDS[index - 1]),
        epochs=epoch,
        device=device,
        log_every=log_every,
    )
    memory = dict(memory)
    memory["stats"] = data["stats"]
    payload = _v2_checkpoint_payload(
        registry=registry,
        contract=contract,
        candidate=candidate,
        index=index,
        selected_epoch=epoch,
        model=model,
        memory=memory,
        origin=origin,
        selection_payload=selection_payload,
        refit_history=history,
        geometry=geometry,
        freeze=freeze,
    )
    payload["training_seconds"] = None
    # Capture duration only after the deterministic model work is complete;
    # it is operational metadata and never participates in selection.
    v1._save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "sha256": v1._sha256_file(destination),
        "candidate": candidate,
        "member_index": index,
        "selected_epoch": epoch,
        "anchors": int(model.n_anchors),
        "test_model_outcome_accesses": 0,
    }


def _load_v2_final_front(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    freeze: Mapping[str, object],
    expected_index: int,
) -> tuple[torch.nn.Module, dict[str, object]]:
    model, payload = v1.load_front_checkpoint(path, torch.device("cpu"))
    selected = freeze["selected"]
    if (
        payload.get("checkpoint_schema") != "tuji_coherent_marked_front_v9_v2_memory_safe"
        or payload.get("registry_sha256") != registry["registry_sha256"]
        or payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]
        or payload.get("freeze_sha256") != freeze["freeze_sha256"]
        or payload.get("phase") != "final"
        or payload.get("refit_role") != "final_refit"
        or payload.get("candidate") != selected["candidate"]
        or int(payload.get("member_index", -1)) != int(expected_index)
        or int(payload.get("member_seed", -1)) != int(v1.MEMBER_SEEDS[expected_index - 1])
    ):
        raise ValueError(f"invalid V2 final checkpoint: {path}")
    return model, payload


def fit_final_radio_memory(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_CONTRACT,
    freeze_path: str | Path = DEFAULT_FREEZE,
) -> dict[str, object]:
    """Fit the frozen candidate's final marked memory using all train scans."""

    registry, contract, _v1_contract = _v1_context(
        registry_path=registry_path, contract_path=contract_path
    )
    freeze = load_selection_freeze(freeze_path, registry=registry, contract=contract)
    candidate = str(freeze["selected"]["candidate"])
    destination = _final_radio_path(candidate)
    if destination.exists():
        raise FileExistsError(destination)
    data = v1._load_training_data(registry)
    rows = registered_rows(registry, "final_refit", data)
    represented = v1._represented_rss(data["rss"], candidate)
    anchor_keys = tuple(sorted(registered_role_keys(registry, "final_refit")))
    memory = v1.fit_hierarchical_marked_memory(
        rss=represented,
        phone_ids=data["phone"],
        coordinate_keys=data["keys"],
        training_indices=rows,
        anchor_keys=anchor_keys,
        detection_concentration_grid=v1.CONCENTRATION_GRID,
        center_concentration_grid=v1.CONCENTRATION_GRID,
        scale_concentration_grid=v1.CONCENTRATION_GRID,
        degrees_freedom_grid=(STUDENT_DF,),
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "radio_memory_schema": "tuji_hierarchical_marked_memory_v2_memory_safe",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "candidate": candidate,
        "phase": "final",
        "refit_role": "final_refit",
        "role_coordinate_sha256": registry["roles"]["final_refit"]["coordinate_key_sha256"],
        "memory": memory,
        "outcome_access": "none: author training RSS and training device IDs only",
    }
    v1._save_torch_new(destination, payload)
    chosen = memory["radio_only_selection"]
    return {
        "radio_memory": str(destination.resolve()),
        "sha256": v1._sha256_file(destination),
        "candidate": candidate,
        "anchors": len(anchor_keys),
        "selected_detection_concentration": chosen["detection"]["selected_concentration"],
        "selected_center_concentration": chosen["conditional_mark"]["selected_center_concentration"],
        "selected_scale_concentration": chosen["conditional_mark"]["selected_scale_concentration"],
        "test_model_outcome_accesses": 0,
    }


def _load_final_radio(
    path: str | Path,
    *, registry: Mapping[str, object], contract: Mapping[str, object], freeze: Mapping[str, object]
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("radio_memory_schema") != "tuji_hierarchical_marked_memory_v2_memory_safe"
        or payload.get("registry_sha256") != registry["registry_sha256"]
        or payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]
        or payload.get("freeze_sha256") != freeze["freeze_sha256"]
        or payload.get("candidate") != freeze["selected"]["candidate"]
        or payload.get("phase") != "final"
        or payload.get("refit_role") != "final_refit"
    ):
        raise ValueError("invalid V2 final marked memory")
    return payload


@torch.no_grad()
def _stream_final_predictions(
    *,
    paths: Sequence[Path],
    query: torch.Tensor,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    freeze: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    """V2 final-panel analogue of the selection cache's sequential stream."""

    candidate = str(freeze["selected"]["candidate"])
    rows = int(len(query))
    count = len(paths)
    logits_cache: torch.Tensor | None = None
    location_cache: torch.Tensor | None = None
    scale_cache: torch.Tensor | None = None
    first_keys = None
    first_origin = None
    anchor_coords = None
    anchor_buildings = None
    anchor_floors = None
    for index, path in enumerate(paths, start=1):
        model, payload = _load_v2_final_front(
            path, registry=registry, contract=contract, freeze=freeze, expected_index=index
        )
        if first_keys is None:
            first_keys = payload["anchor_coordinate_keys"]
            first_origin = payload["coord_origin"]
            anchors = int(model.n_anchors)
            logits_cache = torch.empty(count, rows, anchors)
            location_cache = torch.empty(count, rows, anchors, 2)
            scale_cache = torch.empty_like(location_cache)
            anchor_coords = model.anchor_coords.detach().cpu()
            anchor_buildings = model.anchor_buildings.detach().cpu()
            anchor_floors = model.anchor_floors.detach().cpu()
        else:
            if payload["anchor_coordinate_keys"] != first_keys:
                raise ValueError("final members do not share anchor dictionary")
            if not torch.equal(payload["coord_origin"], first_origin):
                raise ValueError("final members do not share coordinate origin")
        model.to(device).eval()
        anchors = int(model.n_anchors)
        for start in range(0, rows, QUERY_CHUNK_SIZE):
            stop = min(start + QUERY_CHUNK_SIZE, rows)
            values = query[start:stop].to(device)
            logits, embedding = v1._front_logits_embedding(model, values, candidate)
            anchor_index = torch.arange(anchors, device=device, dtype=torch.long).unsqueeze(0).expand(len(values), -1)
            location, scale = model._components(embedding, anchor_index)
            logits_cache[index - 1, start:stop] = logits.detach().cpu()
            location_cache[index - 1, start:stop] = location.detach().cpu()
            scale_cache[index - 1, start:stop] = scale.detach().cpu()
            del values, logits, embedding, anchor_index, location, scale
        model.to(torch.device("cpu"))
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    if any(value is None for value in (logits_cache, location_cache, scale_cache, first_keys, first_origin, anchor_coords, anchor_buildings, anchor_floors)):
        raise RuntimeError("final sequential prediction did not initialize")
    return {
        "query": query.float().cpu(),
        "member_logits": logits_cache,
        "member_locations": location_cache,
        "member_scales": scale_cache,
        "anchor_coords": anchor_coords,
        "anchor_buildings": anchor_buildings,
        "anchor_floors": anchor_floors,
        "anchor_coordinate_keys": first_keys,
        "coord_origin": first_origin,
        "randomness": v1.common_student_t_randomness(
            rows, draws=v1.SCORING_DRAWS, degrees_freedom=STUDENT_DF, seed=SCORING_SEED
        ),
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "member_residency": "sequential_gpu_one_member_at_a_time",
    }


def _load_test_rss_only(registry: Mapping[str, object]) -> dict[str, object]:
    """Read the registered author-test query RSS and no labels whatsoever."""

    source_files = registry["source_files"]
    record = source_files["rss_testing"]
    path = Path(str(record["path"])).resolve()
    if v1._sha256_file(path) != record["sha256"]:
        raise ValueError("author test RSS changed after registration")
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    expected_aps = int(registry["data_contract"]["n_aps"])
    if values.ndim != 2 or values.shape[1] != expected_aps:
        raise ValueError("author test RSS violates frozen AP schema")
    normalized, missing = _normalized_rss(values)
    return {
        "rss": torch.from_numpy(normalized).float(),
        "raw_visible_count": (~missing).sum(axis=1).astype(np.int64),
        "rows": int(len(values)),
        "rss_path": str(path),
    }


def cache_author_test(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_CONTRACT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Generate the sole label-free author-test prediction cache after freeze."""

    registry, contract, _v1_contract = _v1_context(
        registry_path=registry_path, contract_path=contract_path
    )
    freeze = load_selection_freeze(freeze_path, registry=registry, contract=contract)
    destination = Path(DEFAULT_TEST_CACHE).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    selected = freeze["selected"]
    candidate = str(selected["candidate"])
    members = int(selected["E"])
    paths = [_final_checkpoint_path(candidate, index) for index in range(1, members + 1)]
    radio_path = _final_radio_path(candidate)
    radio = _load_final_radio(radio_path, registry=registry, contract=contract, freeze=freeze)
    # The freeze and final artifacts above are deliberately checked before the
    # following label-free RSS loader is allowed to open the author test file.
    query_data = _load_test_rss_only(registry)
    represented = v1._represented_rss(query_data["rss"], candidate)
    cache = _stream_final_predictions(
        paths=paths,
        query=represented,
        registry=registry,
        contract=contract,
        freeze=freeze,
        device=torch.device(device_name),
    )
    if tuple(radio["memory"]["anchor_keys"]) != tuple(cache["anchor_coordinate_keys"]):
        raise ValueError("final marked memory and final member anchors are not aligned")
    raw_visible = query_data["raw_visible_count"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema": "tuji_v9_author_test_prediction_cache_v2_memory_safe",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "candidate": candidate,
        "members": members,
        "rows": int(query_data["rows"]),
        "query_source": "RSS_testing.csv only",
        "test_labels_loaded_during_cache": False,
        "raw_visible_count": torch.from_numpy(raw_visible).long(),
        "raw_visible_above_64": int((raw_visible > int(v1.COMMON_FRONT["max_visible"])).sum()),
        "radio_fields": radio["memory"]["fields"],
        "radio_degrees_freedom": radio["memory"]["degrees_freedom"],
        "radio_memory_path": str(radio_path.resolve()),
        "radio_memory_sha256": v1._sha256_file(radio_path),
        "member_paths": tuple(str(path.resolve()) for path in paths),
        "member_sha256": tuple(v1._sha256_file(path) for path in paths),
        **cache,
    }
    v1._save_torch_new(destination, payload)
    return {
        "author_test_cache": str(destination),
        "sha256": v1._sha256_file(destination),
        "candidate": candidate,
        "members": members,
        "rows": int(query_data["rows"]),
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "test_labels_loaded_during_cache": False,
        "raw_visible_above_64": payload["raw_visible_above_64"],
    }


def _load_author_test_cache(
    path: str | Path,
    *, registry: Mapping[str, object], contract: Mapping[str, object], freeze: Mapping[str, object]
) -> dict[str, object]:
    source = _require_path(path, DEFAULT_TEST_CACHE, label="author-test prediction cache")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("cache_schema") != "tuji_v9_author_test_prediction_cache_v2_memory_safe"
        or payload.get("registry_sha256") != registry["registry_sha256"]
        or payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]
        or payload.get("freeze_sha256") != freeze["freeze_sha256"]
        or payload.get("candidate") != freeze["selected"]["candidate"]
        or int(payload.get("members", -1)) != int(freeze["selected"]["E"])
        or payload.get("test_labels_loaded_during_cache") is not False
    ):
        raise ValueError("invalid V2 author-test prediction cache")
    return payload


def _load_test_labels(
    registry: Mapping[str, object], *, expected_rows: int
) -> dict[str, object]:
    """Final labels, available only after a verified prediction cache exists."""

    source_files = registry["source_files"]
    coordinate = source_files["coordinates_testing"]
    device_map = source_files["device_labels"]
    coordinate_path = Path(str(coordinate["path"])).resolve()
    device_path = Path(str(device_map["path"])).resolve()
    if v1._sha256_file(coordinate_path) != coordinate["sha256"]:
        raise ValueError("author test coordinate labels changed after registration")
    if v1._sha256_file(device_path) != device_map["sha256"]:
        raise ValueError("published device map changed after registration")
    labels = np.loadtxt(coordinate_path, delimiter=",", dtype=np.float64)
    if labels.ndim != 2 or labels.shape != (int(expected_rows), 6):
        raise ValueError("author test coordinate labels violate frozen shape")
    if not bool(np.equal(labels[:, 2:5], 0.0).all()):
        raise ValueError("author test is not the registered single 2-D plane")
    one_based_phone = labels[:, 5]
    if not bool(np.equal(one_based_phone, np.round(one_based_phone)).all()):
        raise ValueError("test device labels must be integral")
    phones = one_based_phone.astype(np.int64) - 1
    if set(map(int, np.unique(phones))) != {0, 1, 2, 3, 4}:
        raise ValueError("test device labels violate registered five-device map")
    keys: list[LocationKey] = [
        (0, 0, round(float(x), 6), round(float(y), 6)) for x, y in labels[:, :2]
    ]
    return {"raw_xy": labels[:, :2].astype(np.float32), "phones": phones, "keys": keys}


def _covariance_trace_diagnostics(
    cache: Mapping[str, object], projection: Mapping[str, object], posterior: torch.Tensor,
    *, members: int, group_ids: torch.Tensor, group_count: int
) -> dict[str, object]:
    """Within-component/candidate/member trace decomposition on final posterior."""

    indices = projection["anchor_indices"]
    if not isinstance(indices, torch.Tensor):
        raise TypeError("projection anchor indices missing")
    locations, scales = v1._support_components(cache, indices, members=int(members))
    member_mass = posterior.sum(dim=2)
    conditional = posterior / member_mass.unsqueeze(2).clamp_min(1e-30)
    member_mean = (conditional.unsqueeze(3) * locations).sum(dim=2)
    mixture_mean = (member_mass.unsqueeze(2) * member_mean).sum(dim=1)
    component_trace = scales.square().sum(dim=3) * (STUDENT_DF / (STUDENT_DF - 2.0))
    within = (posterior * component_trace).sum(dim=(1, 2))
    candidate_delta = locations - member_mean[:, :, None, :]
    candidate = (posterior * candidate_delta.square().sum(dim=3)).sum(dim=(1, 2))
    member_delta = member_mean - mixture_mean[:, None, :]
    between = (member_mass * member_delta.square().sum(dim=2)).sum(dim=1)
    total = within + candidate + between
    return {
        "coordinate_macro_within_component_covariance_trace_m2": float(
            v1._group_vector(within, group_ids, group_count).mean()
        ),
        "coordinate_macro_candidate_ambiguity_covariance_trace_m2": float(
            v1._group_vector(candidate, group_ids, group_count).mean()
        ),
        "coordinate_macro_member_disagreement_covariance_trace_m2": float(
            v1._group_vector(between, group_ids, group_count).mean()
        ),
        "coordinate_macro_total_predictive_covariance_trace_m2": float(
            v1._group_vector(total, group_ids, group_count).mean()
        ),
    }


def _device_macro_mae(
    row_mae: torch.Tensor, keys: Sequence[LocationKey], phones: np.ndarray
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for phone in sorted(map(int, np.unique(phones))):
        rows = np.flatnonzero(phones == phone)
        local_keys = [keys[int(index)] for index in rows]
        groups, group_keys = _coordinate_groups(local_keys)
        group_mae = v1._group_vector(
            row_mae[torch.from_numpy(rows).long()], groups, len(group_keys)
        )
        result[str(phone + 1)] = {
            "rows": int(len(rows)),
            "coordinate_groups": int(len(group_keys)),
            "coordinate_macro_mae_2d_m": float(group_mae.mean()),
        }
    return result


def evaluate_author_test(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_CONTRACT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    cache_path: str | Path = DEFAULT_TEST_CACHE,
    output: str | Path = DEFAULT_RESULT,
    device_name: str | None = None,
) -> dict[str, object]:
    """Score the one final author-test cache exactly once at coordinate level."""

    registry, contract, _v1_contract = _v1_context(
        registry_path=registry_path, contract_path=contract_path
    )
    freeze = load_selection_freeze(freeze_path, registry=registry, contract=contract)
    destination = _require_path(output, DEFAULT_RESULT, label="author-test result")
    if destination.exists():
        raise FileExistsError(destination)
    cache = _load_author_test_cache(
        cache_path, registry=registry, contract=contract, freeze=freeze
    )
    labels = _load_test_labels(registry, expected_rows=int(cache["rows"]))
    group_ids, group_keys = _coordinate_groups(labels["keys"])
    origin = cache["coord_origin"].numpy().astype(np.float32)
    target_xy = torch.from_numpy((labels["raw_xy"] - origin[None, :]).astype(np.float32))
    scoring_cache = {
        **cache,
        "target_xy": target_xy,
        "target_building": torch.zeros(len(target_xy), dtype=torch.long),
        "target_floor": torch.zeros(len(target_xy), dtype=torch.long),
        "coordinate_group_ids": group_ids,
        "coordinate_group_keys": group_keys,
    }
    selected = freeze["selected"]
    full_joint = v1.uniform_member_anchor_prior(
        scoring_cache["member_logits"], members=int(selected["E"])
    )
    projection = v1.topk_prior_projection(full_joint, top_k=int(selected["K"]))
    device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    marked, nuisance = v1._marked_energy_for_projection(
        scoring_cache,
        projection,
        candidate=str(selected["candidate"]),
        device=device,
        batch_size=SCORING_CHUNK_SIZE,
    )
    metrics = v1._configuration_metrics(
        scoring_cache,
        projection,
        marked,
        members=int(selected["E"]),
        lambda_value=float(selected["lambda"]),
        device=device,
        evaluation_batch_size=SCORING_CHUNK_SIZE,
    )
    posterior = v1.marked_kl_update(
        projection["joint_probability"], marked, lambda_value=float(selected["lambda"])
    )
    covariance = _covariance_trace_diagnostics(
        scoring_cache,
        projection,
        posterior,
        members=int(selected["E"]),
        group_ids=group_ids,
        group_count=len(group_keys),
    )
    raw_visible = cache["raw_visible_count"].numpy()
    result: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "result_schema": "tuji_v9_author_test_result_v2_memory_safe",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "freeze": str(Path(freeze_path).resolve()),
        "freeze_sha256": freeze["freeze_sha256"],
        "author_test_cache": str(Path(cache_path).resolve()),
        "author_test_cache_sha256": v1._sha256_file(cache_path),
        "author_test_access_number": 1,
        "comparison_status": "external_replication_after_eligibility_audit",
        "selected": selected,
        "metrics": {
            key: value for key, value in metrics.items() if not isinstance(value, torch.Tensor)
        },
        "covariance_diagnostics": covariance,
        "coordinate_group_count": len(group_keys),
        "coordinate_group_energy_score_2d_m": metrics["group_energy_score_2d_m"].tolist(),
        "coordinate_group_mae_2d_m": metrics["group_mae_2d_m"].tolist(),
        "device_stratified_coordinate_macro_mae_2d_m": _device_macro_mae(
            metrics["row_mae_vector"], labels["keys"], labels["phones"]
        ),
        "top64_diagnostic": {
            "rows": int(len(raw_visible)),
            "raw_visible_above_64": int((raw_visible > int(v1.COMMON_FRONT["max_visible"])).sum()),
            "raw_visible_above_64_fraction": float((raw_visible > int(v1.COMMON_FRONT["max_visible"])).mean()),
            "maximum_raw_visible": int(raw_visible.max()),
            "mean_discarded_visible_aps_if_capped": float(
                np.maximum(raw_visible - int(v1.COMMON_FRONT["max_visible"]), 0).mean()
            ),
        },
        "affine_nuisance_summary": (
            None if not nuisance else {
                "mean_offset": float(nuisance["style_offset"].mean()),
                "mean_scale": float(nuisance["style_scale"].mean()),
            }
        ),
        "metric_note": (
            "all headline metrics are coordinate-macro 2-D; V2 only changes member/query scheduling, not model or score definitions"
        ),
    }
    result["result_sha256"] = v1._canonical_json_sha256(result, omit="result_sha256")
    v1._write_json_new(destination, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_execution = commands.add_parser("freeze-execution")
    freeze_execution.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    freeze_execution.add_argument("--output", default=str(DEFAULT_CONTRACT))
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

    selection = commands.add_parser("freeze-selection")
    selection.add_argument("--members", type=int, default=v1.INITIAL_MEMBER_COUNT)
    selection.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    selection.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    selection.add_argument("--output", default=str(DEFAULT_FREEZE))

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
    elif args.command == "train-v1-extension-member":
        result = train_v1_selection_extension_member(
            candidate=args.candidate, member_index=args.member_index,
            registry_path=args.registry, contract_path=args.contract,
            device_name=args.device, log_every=args.log_every,
        )
    elif args.command == "build-development-cache":
        result = build_development_cache(
            candidate=args.candidate, members=args.members, registry_path=args.registry,
            contract_path=args.contract, device_name=args.device,
        )
    elif args.command == "grid":
        result = evaluate_development_grid(
            candidate=args.candidate, members=args.members, registry_path=args.registry,
            contract_path=args.contract, device_name=args.device,
        )
    elif args.command == "freeze-selection":
        result = select_and_freeze(
            members=args.members, registry_path=args.registry,
            contract_path=args.contract, output=args.output,
        )
    elif args.command == "train-final-member":
        result = train_final_member(
            member_index=args.member_index, registry_path=args.registry,
            contract_path=args.contract, freeze_path=args.freeze,
            device_name=args.device, log_every=args.log_every,
        )
    elif args.command == "fit-final-radio":
        result = fit_final_radio_memory(
            registry_path=args.registry, contract_path=args.contract, freeze_path=args.freeze,
        )
    elif args.command == "cache-author-test":
        result = cache_author_test(
            registry_path=args.registry, contract_path=args.contract,
            freeze_path=args.freeze, device_name=args.device,
        )
    elif args.command == "evaluate-author-test":
        result = evaluate_author_test(
            registry_path=args.registry, contract_path=args.contract,
            freeze_path=args.freeze, cache_path=args.cache, output=args.output,
            device_name=args.device,
        )
    else:  # pragma: no cover - argparse closes this branch.
        raise RuntimeError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
