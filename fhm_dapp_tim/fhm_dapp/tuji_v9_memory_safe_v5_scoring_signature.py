"""A4-compatible execution of the sealed TUJI1 V9 protocol.

V5 preserves V1--V4 and rewires only V2's two obsolete calls to
``_configuration_metrics``.  The three required cache-derived arguments are
passed explicitly, exactly as in frozen V1.  All V5 outputs have new paths and
are bound to a V5 contract.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import tuji_v9_experiment as v1
from fhm_dapp_tim.fhm_dapp import tuji_v9_memory_safe_v2 as v2
from fhm_dapp_tim.fhm_dapp import tuji_v9_memory_safe_v3_callfix as v3
from fhm_dapp_tim.fhm_dapp import tuji_v9_memory_safe_v4_runtime_isolation as v4
from fhm_dapp_tim.fhm_dapp.tuji_v9_protocol import DEFAULT_REGISTRY, PROTOCOL_VERSION, load_registry


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v9_tuji"
V1_CONTRACT = EVIDENCE / "execution_contract_tuji_v9.json"
V2_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v2_memory_safe.json"
V3_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v3_memory_safe_callfix.json"
V4_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v4_runtime_isolation.json"
A1_AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A1_MEMORY_SAFE.md"
A2_AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A2_CALL_SIGNATURE.md"
A3_AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A3_RUNTIME_CONTRACT_ISOLATION.md"
A4_AMENDMENT = EVIDENCE / "TUJI1_V9_EXECUTION_AMENDMENT_A4_SCORING_SIGNATURE.md"
DEFAULT_CONTRACT = EVIDENCE / "execution_contract_tuji_v9_v5_scoring_signature.json"
DEFAULT_FREEZE = EVIDENCE / "selection_freeze_tuji_v9_v5_scoring_signature.json"
DEFAULT_TEST_CACHE = EVIDENCE / "author_test_cache_tuji_v9_v5_scoring_signature.pt"
DEFAULT_RESULT = EVIDENCE / "author_test_result_tuji_v9_v5_scoring_signature.json"
FINAL_MODELS = ROOT / "models" / "v9_tuji_final_v5_scoring_signature"

EXECUTION_CONTRACT_VERSION = "tuji_v9_execution_contract_v5_scoring_signature_20260812"
EXTENSION_MEMBER_COUNTS = v2.EXTENSION_MEMBER_COUNTS


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
    return EVIDENCE / f"development_cache_{_candidate(candidate)}_through_E{count}_v5_scoring_signature.pt"


def _grid_path(candidate: str, members: int) -> Path:
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_grid_{_candidate(candidate)}_through_E{count}_v5_scoring_signature.pt"


def _grid_summary_path(candidate: str, members: int) -> Path:
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    return EVIDENCE / f"development_grid_{_candidate(candidate)}_through_E{count}_v5_scoring_signature.json"


def _extension_report_path(members: int) -> Path:
    return EVIDENCE / f"selection_extension_required_through_E{int(members)}_tuji_v9_v5.json"


def _final_checkpoint_path(candidate: str, index: int) -> Path:
    member = int(index)
    if not 1 <= member <= len(v1.MEMBER_SEEDS):
        raise ValueError("member index is outside the preregistered panel")
    return FINAL_MODELS / _candidate(candidate) / f"member_{member:02d}_seed{v1.MEMBER_SEEDS[member - 1]}.pt"


def _final_radio_path(candidate: str) -> Path:
    return EVIDENCE / f"radio_memory_final_{_candidate(candidate)}_v5_scoring_signature.pt"


def _future_output_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {"selection_freeze": DEFAULT_FREEZE, "author_test_cache": DEFAULT_TEST_CACHE, "author_test_result": DEFAULT_RESULT}
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
    d = Path(__file__).resolve().parent
    return {
        "v5_scoring_signature_executor": Path(__file__).resolve(), "v4_runtime_isolation_executor": d / "tuji_v9_memory_safe_v4_runtime_isolation.py", "v3_callfix_executor": d / "tuji_v9_memory_safe_v3_callfix.py", "v2_memory_safe_executor": d / "tuji_v9_memory_safe_v2.py", "v1_executor": d / "tuji_v9_experiment.py", "tuji_protocol": d / "tuji_v9_protocol.py", "uji_v9_core": d / "uji_v9_experiment.py", "marked_math": d / "coherent_marked_v9.py", "marked_memory": d / "experiment_v71.py", "safe_artifact_primitives": d / "experiment_v7.py", "front_model": d / "model.py", "anchor_memory": d / "anchor.py", "front_training": d / "train_anchor.py", "amendment_a1": A1_AMENDMENT, "amendment_a2": A2_AMENDMENT, "amendment_a3": A3_AMENDMENT, "amendment_a4": A4_AMENDMENT, "v1_contract": V1_CONTRACT, "v2_contract": V2_CONTRACT, "v3_contract": V3_CONTRACT, "v4_contract": V4_CONTRACT,
    }


def _manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {name: {"path": str(path.resolve()), "sha256": v1._sha256_file(path)} for name, path in paths.items()}


def _validate_v4_predecessor(registry_path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    v4_contract = v4.load_execution_contract(V4_CONTRACT, registry_path=registry_path)
    cache_path = Path(str(v4_contract["future_output_paths"]["cache_geometry_invariant_E8"])).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError("sealed V4 geometry cache is missing")
    unexpected = [name for name, path in v4_contract["future_output_paths"].items() if name != "cache_geometry_invariant_E8" and Path(str(path)).exists()]
    if unexpected:
        raise RuntimeError("V4 grid failure unexpectedly created additional output: " + ", ".join(unexpected))
    return v4_contract, {"path": str(cache_path), "sha256": v1._sha256_file(cache_path), "rows": 1366, "coordinate_groups": 270, "author_test_accesses": 0}


def freeze_execution_contract(*, registry_path: str | Path = DEFAULT_REGISTRY, output: str | Path = DEFAULT_CONTRACT) -> dict[str, object]:
    registry_file = Path(registry_path).resolve()
    destination = _require_path(output, DEFAULT_CONTRACT, label="V5 execution contract")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite execution contract: {destination}")
    registry = load_registry(registry_file)
    v4_contract, v4_cache = _validate_v4_predecessor(registry_file)
    v1_contract = v1.load_execution_contract(V1_CONTRACT, registry_path=registry_file)
    prior = v2._selection_artifact_manifest(registry=registry, v1_contract=v1_contract)
    existing = [str(path.resolve()) for path in _future_output_paths().values() if path.exists()]
    if existing:
        raise FileExistsError("V5 contract must precede all V5 outcomes: " + ", ".join(existing))
    contract: dict[str, object] = {
        "execution_contract_version": EXECUTION_CONTRACT_VERSION, "protocol_version": PROTOCOL_VERSION, "registry": str(registry_file), "registry_sha256": registry["registry_sha256"], "design": str(Path(str(registry["design"])).resolve()), "design_sha256": registry["design_sha256"], "comparison_status": "external_replication_after_eligibility_audit", "selection_authority_for_v9_uji": False, "may_modify_v9_uji": False, "scientific_change": False,
        "amendment_chain": [{"id": "A1", "path": str(A1_AMENDMENT.resolve()), "sha256": v1._sha256_file(A1_AMENDMENT)}, {"id": "A2", "path": str(A2_AMENDMENT.resolve()), "sha256": v1._sha256_file(A2_AMENDMENT)}, {"id": "A3", "path": str(A3_AMENDMENT.resolve()), "sha256": v1._sha256_file(A3_AMENDMENT)}, {"id": "A4", "path": str(A4_AMENDMENT.resolve()), "sha256": v1._sha256_file(A4_AMENDMENT)}],
        "v1_execution_contract": {"path": str(V1_CONTRACT.resolve()), "sha256": v1._sha256_file(V1_CONTRACT), "execution_contract_sha256": v1_contract["execution_contract_sha256"]},
        "v4_predecessor": {"path": str(V4_CONTRACT.resolve()), "sha256": v1._sha256_file(V4_CONTRACT), "execution_contract_sha256": v4_contract["execution_contract_sha256"], "valid_training_only_cache": v4_cache, "failed_grid": {"created_grid": False, "read_author_test_rss": False, "read_author_test_labels": False, "reason": "V2 scoring call omitted V1's required target_xy/group_ids/group_keys keyword-only arguments"}},
        "preexisting_v1_selection_artifacts": prior, "test_model_outcome_accesses_before_v5_freeze": 0,
        "memory_safe_execution": {"member_residency": "one member at a time", "query_chunk_size": v2.QUERY_CHUNK_SIZE, "scoring_chunk_size": v2.SCORING_CHUNK_SIZE, "scientific_parameters_changed": False},
        "scoring_abi": {"helper": "v1._configuration_metrics", "arguments": ["target_xy=cache[target_xy]", "group_ids=cache[coordinate_group_ids]", "group_keys=cache[coordinate_group_keys]"], "development_and_author_test": True},
        "test_gate": {"selection_commands": "author_training_only", "author_test_cache": "requires V5 selection freeze, V5 final refit panel, and V5 final marked memory", "author_test_query_input": "RSS_testing.csv only", "author_test_labels": "only after prediction cache exists"},
        "selection_grid": {"E": list(v1.E_GRID), "K": list(v1.K_GRID), "lambda": list(v1.LAMBDA_GRID), "member_seeds": list(v1.MEMBER_SEEDS), "extensions": list(EXTENSION_MEMBER_COUNTS), "member_rule": "preregistered V1 seed prefix"},
        "execution_sources": _manifest(_source_paths()), "future_output_paths": {name: str(path.resolve()) for name, path in _future_output_paths().items()},
    }
    contract["execution_contract_sha256"] = v1._canonical_json_sha256(contract, omit="execution_contract_sha256")
    v1._write_json_new(destination, contract)
    return contract


def load_execution_contract(path: str | Path = DEFAULT_CONTRACT, *, registry_path: str | Path | None = None) -> dict[str, object]:
    file = _require_path(path, DEFAULT_CONTRACT, label="V5 execution contract")
    contract = json.loads(file.read_text(encoding="utf-8"))
    if contract.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION or contract.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V5 TUJI execution contract")
    if contract.get("execution_contract_sha256") != v1._canonical_json_sha256(contract, omit="execution_contract_sha256"):
        raise ValueError("V5 execution-contract self-hash failed")
    registry_file = Path(str(contract["registry"])).resolve()
    if registry_path is not None and Path(registry_path).resolve() != registry_file:
        raise ValueError("caller registry differs from V5 contract")
    registry = load_registry(registry_file)
    if registry["registry_sha256"] != contract["registry_sha256"] or registry["design_sha256"] != contract["design_sha256"]:
        raise ValueError("V5 registry or design mismatch")
    for name, record in contract["execution_sources"].items():
        if v1._sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V5 source changed after freeze: {name}")
    v4_contract, v4_cache = _validate_v4_predecessor(registry_file)
    if v4_contract["execution_contract_sha256"] != contract["v4_predecessor"]["execution_contract_sha256"] or v4_cache != contract["v4_predecessor"]["valid_training_only_cache"]:
        raise ValueError("V4 predecessor changed after V5 freeze")
    v1_contract = v1.load_execution_contract(V1_CONTRACT, registry_path=registry_file)
    if v1_contract["execution_contract_sha256"] != contract["v1_execution_contract"]["execution_contract_sha256"]:
        raise ValueError("V1 contract identity mismatch")
    if v2._selection_artifact_manifest(registry=registry, v1_contract=v1_contract) != contract["preexisting_v1_selection_artifacts"]:
        raise ValueError("immutable V1 selection artifacts changed")
    return contract


def _evaluate_development_grid_v5(*, candidate: str, members: int = v1.INITIAL_MEMBER_COUNT, registry_path: str | Path = DEFAULT_REGISTRY, contract_path: str | Path = DEFAULT_CONTRACT, device_name: str = "cuda") -> dict[str, object]:
    registry, contract, _ = v2._v1_context(registry_path=registry_path, contract_path=contract_path)
    name, count = _candidate(candidate), int(members)
    source, destination, summary_destination = _cache_path(name, count), _grid_path(name, count), _grid_summary_path(name, count)
    if destination.exists() or summary_destination.exists():
        raise FileExistsError("refusing to overwrite V5 development grid")
    cache = v2._load_development_cache(source, registry=registry, contract=contract, candidate=name)
    available = int(cache["member_logits"].shape[0])
    if available != count:
        raise ValueError("cache member count differs from requested grid")
    configurations: list[dict[str, object]] = []
    inherited = None
    first_new_e = 1
    if count > v1.INITIAL_MEMBER_COUNT:
        previous = count - 2
        base_path = _grid_path(name, previous)
        base = v2._load_grid(base_path, registry=registry, contract=contract, candidate=name)
        base_cache = v2._load_development_cache(_cache_path(name, previous), registry=registry, contract=contract, candidate=name)
        if tuple(base_cache["member_sha256"]) != tuple(cache["member_sha256"][:previous]):
            raise ValueError("extension cache did not preserve V1 member prefix")
        for key in ("role_coordinate_sha256", "row_index_sha256", "radio_memory_sha256"):
            if base_cache[key] != cache[key]:
                raise ValueError(f"extension changed frozen field: {key}")
        configurations, first_new_e = list(base["configurations"]), previous + 1
        inherited = {"path": str(base_path.resolve()), "sha256": v1._sha256_file(base_path), "available_members": previous}
    device = torch.device(device_name)
    for e_count in range(first_new_e, available + 1):
        full_joint = v1.uniform_member_anchor_prior(cache["member_logits"], members=e_count)
        for top_k in v1.K_GRID:
            if top_k > full_joint.shape[2]:
                continue
            projection = v1.topk_prior_projection(full_joint, top_k=top_k)
            marked, nuisance = v1._marked_energy_for_projection(cache, projection, candidate=name, device=device, batch_size=v2.SCORING_CHUNK_SIZE)
            for lambda_value in v1.LAMBDA_GRID:
                metrics = v1._configuration_metrics(cache, projection, marked, members=e_count, lambda_value=float(lambda_value), target_xy=cache["target_xy"], group_ids=cache["coordinate_group_ids"], group_keys=cache["coordinate_group_keys"], device=device, evaluation_batch_size=v2.SCORING_CHUNK_SIZE)
                configurations.append(metrics)
                print(json.dumps({"candidate": name, "E": e_count, "K": top_k, "lambda": lambda_value, "macro_energy_m": metrics["coordinate_macro_energy_score_2d_m"], "macro_mae_m": metrics["coordinate_macro_mae_2d_m"]}, allow_nan=False), flush=True)
            del marked, projection
            if nuisance and device.type == "cuda":
                torch.cuda.empty_cache()
        del full_joint
    payload: dict[str, object] = {"protocol_version": PROTOCOL_VERSION, "grid_schema": "tuji_v9_joint_development_grid_v2_memory_safe", "registry_sha256": registry["registry_sha256"], "execution_contract_sha256": contract["execution_contract_sha256"], "candidate": name, "cache": str(source.resolve()), "cache_sha256": v1._sha256_file(source), "available_members": available, "inherited_base_grid": inherited, "coordinate_group_keys": cache["coordinate_group_keys"], "configurations": configurations, "scoring_chunk_size": v2.SCORING_CHUNK_SIZE, "test_model_outcome_accesses": 0}
    v1._save_torch_new(destination, payload)
    profiled = []
    for e_count in range(1, available + 1):
        chosen = min((row for row in configurations if int(row["E"]) == e_count), key=lambda row: float(row["coordinate_macro_energy_score_2d_m"]))
        profiled.append({key: value for key, value in chosen.items() if not isinstance(value, torch.Tensor)})
    summary = {"protocol_version": PROTOCOL_VERSION, "grid_schema": "tuji_v9_joint_development_grid_v2_memory_safe_summary", "registry_sha256": registry["registry_sha256"], "execution_contract_sha256": contract["execution_contract_sha256"], "candidate": name, "grid": str(destination.resolve()), "grid_sha256": v1._sha256_file(destination), "cache_sha256": payload["cache_sha256"], "available_members": available, "profiled_by_E": profiled, "scoring_chunk_size": v2.SCORING_CHUNK_SIZE, "test_model_outcome_accesses": 0}
    v1._write_json_new(summary_destination, summary)
    return summary


def _evaluate_author_test_v5(*, registry_path: str | Path = DEFAULT_REGISTRY, contract_path: str | Path = DEFAULT_CONTRACT, freeze_path: str | Path = DEFAULT_FREEZE, cache_path: str | Path = DEFAULT_TEST_CACHE, output: str | Path = DEFAULT_RESULT, device_name: str | None = None) -> dict[str, object]:
    registry, contract, _ = v2._v1_context(registry_path=registry_path, contract_path=contract_path)
    freeze = v2.load_selection_freeze(freeze_path, registry=registry, contract=contract)
    destination = _require_path(output, DEFAULT_RESULT, label="author-test result")
    if destination.exists():
        raise FileExistsError(destination)
    cache = v2._load_author_test_cache(cache_path, registry=registry, contract=contract, freeze=freeze)
    labels = v2._load_test_labels(registry, expected_rows=int(cache["rows"]))
    group_ids, group_keys = v2._coordinate_groups(labels["keys"])
    origin = cache["coord_origin"].numpy().astype(np.float32)
    target_xy = torch.from_numpy((labels["raw_xy"] - origin[None, :]).astype(np.float32))
    scoring_cache = {**cache, "target_xy": target_xy, "target_building": torch.zeros(len(target_xy), dtype=torch.long), "target_floor": torch.zeros(len(target_xy), dtype=torch.long), "coordinate_group_ids": group_ids, "coordinate_group_keys": group_keys}
    selected = freeze["selected"]
    full_joint = v1.uniform_member_anchor_prior(scoring_cache["member_logits"], members=int(selected["E"]))
    projection = v1.topk_prior_projection(full_joint, top_k=int(selected["K"]))
    device = torch.device(device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    marked, nuisance = v1._marked_energy_for_projection(scoring_cache, projection, candidate=str(selected["candidate"]), device=device, batch_size=v2.SCORING_CHUNK_SIZE)
    metrics = v1._configuration_metrics(scoring_cache, projection, marked, members=int(selected["E"]), lambda_value=float(selected["lambda"]), target_xy=scoring_cache["target_xy"], group_ids=scoring_cache["coordinate_group_ids"], group_keys=scoring_cache["coordinate_group_keys"], device=device, evaluation_batch_size=v2.SCORING_CHUNK_SIZE)
    posterior = v1.marked_kl_update(projection["joint_probability"], marked, lambda_value=float(selected["lambda"]))
    covariance = v2._covariance_trace_diagnostics(scoring_cache, projection, posterior, members=int(selected["E"]), group_ids=group_ids, group_count=len(group_keys))
    raw_visible = cache["raw_visible_count"].numpy()
    result: dict[str, object] = {"protocol_version": PROTOCOL_VERSION, "result_schema": "tuji_v9_author_test_result_v2_memory_safe", "registry_sha256": registry["registry_sha256"], "execution_contract_sha256": contract["execution_contract_sha256"], "freeze": str(Path(freeze_path).resolve()), "freeze_sha256": freeze["freeze_sha256"], "author_test_cache": str(Path(cache_path).resolve()), "author_test_cache_sha256": v1._sha256_file(cache_path), "author_test_access_number": 1, "comparison_status": "external_replication_after_eligibility_audit", "selected": selected, "metrics": {key: value for key, value in metrics.items() if not isinstance(value, torch.Tensor)}, "covariance_diagnostics": covariance, "coordinate_group_count": len(group_keys), "coordinate_group_energy_score_2d_m": metrics["group_energy_score_2d_m"].tolist(), "coordinate_group_mae_2d_m": metrics["group_mae_2d_m"].tolist(), "device_stratified_coordinate_macro_mae_2d_m": v2._device_macro_mae(metrics["row_mae_vector"], labels["keys"], labels["phones"]), "top64_diagnostic": {"rows": int(len(raw_visible)), "raw_visible_above_64": int((raw_visible > int(v1.COMMON_FRONT["max_visible"])).sum()), "raw_visible_above_64_fraction": float((raw_visible > int(v1.COMMON_FRONT["max_visible"])).mean()), "maximum_raw_visible": int(raw_visible.max()), "mean_discarded_visible_aps_if_capped": float(np.maximum(raw_visible - int(v1.COMMON_FRONT["max_visible"]), 0).mean())}, "affine_nuisance_summary": None if not nuisance else {"mean_offset": float(nuisance["style_offset"].mean()), "mean_scale": float(nuisance["style_scale"].mean())}, "metric_note": "all headline metrics are coordinate-macro 2-D; V5 changes only mechanical scorer argument passing"}
    result["result_sha256"] = v1._canonical_json_sha256(result, omit="result_sha256")
    v1._write_json_new(destination, result)
    return result


def _install_v2_runtime() -> None:
    v2.DEFAULT_CONTRACT, v2.DEFAULT_FREEZE, v2.DEFAULT_TEST_CACHE, v2.DEFAULT_RESULT, v2.FINAL_MODELS = DEFAULT_CONTRACT, DEFAULT_FREEZE, DEFAULT_TEST_CACHE, DEFAULT_RESULT, FINAL_MODELS
    v2._v2_cache_path, v2._v2_grid_path, v2._v2_grid_summary_path, v2._extension_report_path, v2._final_checkpoint_path, v2._final_radio_path = _cache_path, _grid_path, _grid_summary_path, _extension_report_path, _final_checkpoint_path, _final_radio_path
    v2._load_one_front, v2._load_v2_final_front, v2.load_execution_contract = v3._load_one_front_v3, v3._load_v2_final_front_v3, load_execution_contract
    v2.evaluate_development_grid, v2.evaluate_author_test = _evaluate_development_grid_v5, _evaluate_author_test_v5


def audit_execution_contract(*, registry_path: str | Path = DEFAULT_REGISTRY, contract_path: str | Path = DEFAULT_CONTRACT) -> dict[str, object]:
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    _install_v2_runtime()
    base = v2.audit_execution_contract(registry_path=registry_path, contract_path=contract_path)
    development = inspect.getsource(_evaluate_development_grid_v5)
    author_test = inspect.getsource(_evaluate_author_test_v5)
    for source in (development, author_test):
        if "target_xy=" not in source or "group_ids=" not in source or "group_keys=" not in source:
            raise RuntimeError("V5 scorer does not explicitly satisfy V1 keyword-only ABI")
    registry = load_registry(registry_path)
    v1_contract = v1.load_execution_contract(V1_CONTRACT, registry_path=registry_path)
    model, payload = v3._load_one_front_v3(v1.selection_checkpoint_path("geometry_invariant", 1), registry=registry, expected_contract_sha256=str(v1_contract["execution_contract_sha256"]), expected_phase="selection", expected_candidate="geometry_invariant", expected_role="selection_refit", expected_index=1)
    anchors = int(model.n_anchors)
    del model, payload
    gc.collect()
    return {"status": "passed_scoring_signature_pre_outcome_audit", "registry_sha256": contract["registry_sha256"], "v4_execution_contract_sha256": contract["v4_predecessor"]["execution_contract_sha256"], "v5_execution_contract_sha256": contract["execution_contract_sha256"], "v1_selection_artifacts_verified": len(contract["preexisting_v1_selection_artifacts"]), "v5_future_outputs_absent": True, "v4_geometry_cache_sealed_not_reused_for_v5_selection": True, "one_member_gpu_residency": True, "author_test_cache_requires_freeze_before_rss": True, "author_test_cache_does_not_load_labels": True, "selection_loader_cpu_anchors": anchors, "scoring_keyword_abi_checked_for_development_and_author_test": True, "underlying_v2_audit_status": base["status"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
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
        load_execution_contract(args.contract, registry_path=args.registry); _install_v2_runtime()
        if args.command == "train-v1-extension-member": result = v2.train_v1_selection_extension_member(candidate=args.candidate, member_index=args.member_index, registry_path=args.registry, contract_path=args.contract, device_name=args.device, log_every=args.log_every)
        elif args.command == "build-development-cache": result = v2.build_development_cache(candidate=args.candidate, members=args.members, registry_path=args.registry, contract_path=args.contract, device_name=args.device)
        elif args.command == "grid": result = _evaluate_development_grid_v5(candidate=args.candidate, members=args.members, registry_path=args.registry, contract_path=args.contract, device_name=args.device)
        elif args.command == "freeze-selection": result = v2.select_and_freeze(members=args.members, registry_path=args.registry, contract_path=args.contract, output=args.output)
        elif args.command == "train-final-member": result = v2.train_final_member(member_index=args.member_index, registry_path=args.registry, contract_path=args.contract, freeze_path=args.freeze, device_name=args.device, log_every=args.log_every)
        elif args.command == "fit-final-radio": result = v2.fit_final_radio_memory(registry_path=args.registry, contract_path=args.contract, freeze_path=args.freeze)
        elif args.command == "cache-author-test": result = v2.cache_author_test(registry_path=args.registry, contract_path=args.contract, freeze_path=args.freeze, device_name=args.device)
        elif args.command == "evaluate-author-test": result = _evaluate_author_test_v5(registry_path=args.registry, contract_path=args.contract, freeze_path=args.freeze, cache_path=args.cache, output=args.output, device_name=args.device)
        else: raise RuntimeError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
