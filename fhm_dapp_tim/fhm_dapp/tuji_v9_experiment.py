"""Frozen-gated external TUJI1 execution of the coherent V9 mechanism.

This module is intentionally separate from :mod:`uji_v9_experiment`: UJI V9
artifacts must remain immutable.  It uses the same two complete V9 front
candidates and the same joint ``(E, K, lambda)`` selection mathematics, but
with TUJI1's author-provided train/test split and the pre-result registry in
``evidence/v9_tuji``.

There are two independent gates before author-test inference:

1. a self-hashed TUJI data/protocol registry; and
2. a self-hashed execution contract that binds this adapter and its local
   mechanism dependencies.

Selection commands load *only* author training data.  ``cache-author-test``
is the sole command allowed to read ``RSS_testing.csv`` for prediction; it
first requires the immutable selection freeze and final refit artifacts.
Coordinates and device labels are loaded only by ``evaluate-author-test``
after the prediction cache exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    build_delaunay_geometry,
    common_student_t_randomness,
    continuous_energy_score_mc,
    marked_energy_on_support,
    marked_kl_update,
    profile_shared_affine_marked_energy,
    select_information_saturated_E,
    simultaneous_paired_upper_bounds,
    spatial_median_from_draws,
    topk_prior_projection,
    uniform_member_anchor_prior,
)
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _save_torch_new,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.experiment_v71 import (
    fit_hierarchical_marked_memory,
)
from fhm_dapp_tim.fhm_dapp.model import AnchorStudentTModel
from fhm_dapp_tim.fhm_dapp.tuji_v9_protocol import (
    BOOTSTRAP_REPETITIONS,
    DEFAULT_DATA_DIR,
    DEFAULT_DESIGN,
    DEFAULT_FREEZE,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    DEFAULT_TEST_CACHE,
    PAIRED_INFERENCE_SEED,
    PROTOCOL_VERSION,
    SCORING_SEED,
    STUDENT_DF,
    VISIBLE_THRESHOLD,
    _normalized_rss,
    load_registry,
    load_tuji_split,
    registered_role_keys,
    registered_rows,
)
from fhm_dapp_tim.fhm_dapp.uji_v9_experiment import (
    AFFINE_PROFILE,
    CANDIDATES,
    COMMON_FRONT,
    CONCENTRATION_GRID,
    E_GRID,
    HISTORICAL_FRONT,
    INITIAL_MEMBER_COUNT,
    K_GRID,
    LAMBDA_GRID,
    MAE_NONINFERIORITY_MARGIN_M,
    MEMBER_SEEDS,
    SCORING_DRAWS,
    _fit_fixed,
    _fit_schedule,
    _front_logits_embedding,
    _represented_rss,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "evidence" / "v9_tuji"
DEFAULT_EXECUTION_CONTRACT = DEFAULT_EVIDENCE / "execution_contract_tuji_v9.json"
DEFAULT_SELECTION_MODELS = ROOT / "models" / "v9_tuji_selection"
DEFAULT_FINAL_MODELS = ROOT / "models" / "v9_tuji_final"

EXECUTION_CONTRACT_VERSION = "tuji_v9_execution_contract_v1_20260812"
EXTENSION_MEMBER_COUNTS = (8, 10, 12)

LocationKey = tuple[int, int, float, float]


def _candidate_check(candidate: str) -> str:
    name = str(candidate)
    if name not in CANDIDATES:
        raise ValueError(f"unknown V9 candidate: {name}")
    return name


def _require_protocol_path(
    path: str | Path, expected: str | Path, *, label: str
) -> Path:
    """Reject alternate artifact paths that would evade the frozen boundary."""

    resolved = Path(path).resolve()
    required = Path(expected).resolve()
    if resolved != required:
        raise ValueError(f"{label} must be the protocol path: {required}")
    return resolved


def _member_check(member_index: int) -> int:
    index = int(member_index)
    if not 1 <= index <= len(MEMBER_SEEDS):
        raise ValueError("member index is outside the registered seed panel")
    return index


def selection_checkpoint_path(candidate: str, member_index: int) -> Path:
    name = _candidate_check(candidate)
    index = _member_check(member_index)
    return DEFAULT_SELECTION_MODELS / name / (
        f"member_{index:02d}_seed{MEMBER_SEEDS[index - 1]}.pt"
    )


def final_checkpoint_path(candidate: str, member_index: int) -> Path:
    name = _candidate_check(candidate)
    index = _member_check(member_index)
    return DEFAULT_FINAL_MODELS / name / (
        f"member_{index:02d}_seed{MEMBER_SEEDS[index - 1]}.pt"
    )


def radio_memory_path(candidate: str, phase: str) -> Path:
    name = _candidate_check(candidate)
    if phase not in {"selection", "final"}:
        raise ValueError("radio-memory phase must be selection or final")
    return DEFAULT_EVIDENCE / f"radio_memory_{phase}_{name}.pt"


def development_cache_path(candidate: str, members: int) -> Path:
    name = _candidate_check(candidate)
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"development cache member count must be one of {EXTENSION_MEMBER_COUNTS}")
    return DEFAULT_EVIDENCE / f"development_cache_{name}_through_E{count}.pt"


def grid_result_path(candidate: str, members: int) -> Path:
    name = _candidate_check(candidate)
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"grid member count must be one of {EXTENSION_MEMBER_COUNTS}")
    return DEFAULT_EVIDENCE / f"development_grid_{name}_through_E{count}.pt"


def grid_summary_path(candidate: str, members: int) -> Path:
    name = _candidate_check(candidate)
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"grid member count must be one of {EXTENSION_MEMBER_COUNTS}")
    return DEFAULT_EVIDENCE / f"development_grid_{name}_through_E{count}.json"


def extension_report_path(members: int) -> Path:
    count = int(members)
    return DEFAULT_EVIDENCE / f"selection_extension_required_through_E{count}_tuji_v9.json"


def _execution_source_paths() -> dict[str, Path]:
    """Every local source directly imported by this TUJI executor."""

    module_dir = Path(__file__).resolve().parent
    return {
        "tuji_executor": Path(__file__).resolve(),
        "tuji_protocol": module_dir / "tuji_v9_protocol.py",
        "uji_v9_core": module_dir / "uji_v9_experiment.py",
        "marked_math": module_dir / "coherent_marked_v9.py",
        "marked_memory": module_dir / "experiment_v71.py",
        "safe_artifact_primitives": module_dir / "experiment_v7.py",
        "front_model": module_dir / "model.py",
        "anchor_memory": module_dir / "anchor.py",
        "front_training": module_dir / "train_anchor.py",
    }


def _source_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _all_future_output_paths() -> dict[str, Path]:
    outputs: dict[str, Path] = {
        "selection_freeze": Path(DEFAULT_FREEZE),
        "author_test_cache": Path(DEFAULT_TEST_CACHE),
        "author_test_result": Path(DEFAULT_RESULT),
    }
    for candidate in CANDIDATES:
        for index in range(1, len(MEMBER_SEEDS) + 1):
            outputs[f"selection_{candidate}_{index}"] = selection_checkpoint_path(
                candidate, index
            )
            outputs[f"final_{candidate}_{index}"] = final_checkpoint_path(
                candidate, index
            )
        for phase in ("selection", "final"):
            outputs[f"radio_{phase}_{candidate}"] = radio_memory_path(
                candidate, phase
            )
        for members in EXTENSION_MEMBER_COUNTS:
            outputs[f"cache_{candidate}_E{members}"] = development_cache_path(
                candidate, members
            )
            outputs[f"grid_{candidate}_E{members}"] = grid_result_path(
                candidate, members
            )
            outputs[f"grid_summary_{candidate}_E{members}"] = grid_summary_path(
                candidate, members
            )
    for members in EXTENSION_MEMBER_COUNTS[:-1]:
        outputs[f"extension_report_E{members}"] = extension_report_path(members)
    return outputs


def _contract_without_hash(contract: Mapping[str, object]) -> dict[str, object]:
    value = copy.deepcopy(dict(contract))
    value.pop("execution_contract_sha256", None)
    return value


def freeze_execution_contract(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_EXECUTION_CONTRACT,
) -> dict[str, object]:
    """Freeze the newly implemented TUJI runner before any model outcome."""

    registry_file = Path(registry_path).resolve()
    destination = _require_protocol_path(
        output, DEFAULT_EXECUTION_CONTRACT, label="execution contract"
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite execution contract: {destination}")
    registry = load_registry(registry_file)
    existing = [
        str(path.resolve())
        for path in _all_future_output_paths().values()
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "execution contract must precede every model artifact; found: "
            + ", ".join(existing)
        )
    contract: dict[str, object] = {
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "registry": str(registry_file),
        "registry_sha256": registry["registry_sha256"],
        "design": str(Path(str(registry["design"])).resolve()),
        "design_sha256": registry["design_sha256"],
        "created_after_tuji_test_metadata_access": True,
        "selection_authority_for_v9_uji": False,
        "may_modify_v9_uji": False,
        "comparison_status": "external_replication_after_eligibility_audit",
        "scientific_change": False,
        "test_model_outcome_accesses_before_execution_freeze": 0,
        "test_gate": {
            "selection_commands": "author_training_only",
            "author_test_cache": (
                "requires selection freeze, frozen final candidate, final refit "
                "members, and final marked memory"
            ),
            "author_test_query_input": "RSS_testing.csv only",
            "author_test_labels": (
                "Coordinates_testing.csv and device labels may be loaded only "
                "after author-test cache exists, for scoring/grouping"
            ),
        },
        "execution_sources": _source_manifest(_execution_source_paths()),
        "future_output_paths": {
            name: str(path.resolve())
            for name, path in _all_future_output_paths().items()
        },
        "selection_grid": {
            "initial_members": INITIAL_MEMBER_COUNT,
            "permitted_extensions": list(EXTENSION_MEMBER_COUNTS),
            "E": list(E_GRID),
            "K": list(K_GRID),
            "lambda": list(LAMBDA_GRID),
            "member_seeds": list(MEMBER_SEEDS),
            "member_rule": "preregistered_seed_prefix",
        },
        "metrics": {
            "primary": "coordinate_macro_continuous_mixture_energy_2d",
            "secondary": "coordinate_macro_spatial_median_mae_2d",
            "test_bootstrap_unit": "physical coordinate",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "paired_seed": PAIRED_INFERENCE_SEED,
            "scoring_seed": SCORING_SEED,
            "scoring_draws_per_half": SCORING_DRAWS,
        },
    }
    contract["execution_contract_sha256"] = _canonical_json_sha256(
        contract, omit="execution_contract_sha256"
    )
    _write_json_new(destination, contract)
    return contract


def load_execution_contract(
    path: str | Path = DEFAULT_EXECUTION_CONTRACT,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, object]:
    """Validate the immutable runner gate and every local source hash."""

    contract_file = Path(path).resolve()
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
        raise ValueError("not a TUJI V9 execution contract")
    if contract.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("execution contract protocol version mismatch")
    stored = contract.get("execution_contract_sha256")
    if stored != _canonical_json_sha256(contract, omit="execution_contract_sha256"):
        raise ValueError("execution contract self-hash failed")
    expected_registry = Path(str(contract["registry"])).resolve()
    if registry_path is not None and Path(registry_path).resolve() != expected_registry:
        raise ValueError("caller registry path differs from frozen execution contract")
    registry = load_registry(expected_registry)
    if registry["registry_sha256"] != contract.get("registry_sha256"):
        raise ValueError("execution contract registry hash mismatch")
    if registry["design_sha256"] != contract.get("design_sha256"):
        raise ValueError("execution contract design hash mismatch")
    sources = contract.get("execution_sources")
    if not isinstance(sources, Mapping):
        raise ValueError("execution contract source manifest missing")
    for name, record in sources.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid execution-source record: {name}")
        if _sha256_file(str(record["path"])) != record["sha256"]:
            raise ValueError(f"execution source changed after freeze: {name}")
    return contract


def audit_execution_contract(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_EXECUTION_CONTRACT,
) -> dict[str, object]:
    """Read-only implementation audit before selection training begins."""

    contract = load_execution_contract(contract_path, registry_path=registry_path)
    existing = [
        name for name, path in _all_future_output_paths().items() if path.exists()
    ]
    if existing:
        raise FileExistsError("pre-training audit found model artifacts: " + ", ".join(existing))
    cache_source = inspect.getsource(cache_author_test)
    if cache_source.index("load_selection_freeze") > cache_source.index(
        "load_tuji_test_rss_only"
    ):
        raise RuntimeError("author-test RSS load is not preceded by selection freeze")
    if "load_tuji_test_labels_only" in cache_source:
        raise RuntimeError("author-test cache must not load coordinate/device labels")
    return {
        "status": "passed_pre_training_no_model_outcomes",
        "registry_sha256": contract["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "future_outputs_absent": True,
        "author_test_cache_requires_freeze_before_rss_load": True,
        "author_test_cache_does_not_load_labels": True,
        "selection_commands_are_training_only": True,
    }


def _data_dir_from_registry(registry: Mapping[str, object]) -> Path:
    source = registry["source_files"]
    if not isinstance(source, Mapping):
        raise ValueError("registry source manifest missing")
    train = source["rss_training"]
    if not isinstance(train, Mapping):
        raise ValueError("training RSS source record missing")
    return Path(str(train["path"])).resolve().parent


def _load_training_data(registry: Mapping[str, object]) -> dict[str, object]:
    """Only selection/final training commands may call this helper."""

    return load_tuji_split(_data_dir_from_registry(registry), split="training")


def _checkpoint_payload(
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    phase: str,
    refit_role: str,
    candidate: str,
    member_index: int,
    selected_epoch: int,
    model: AnchorStudentTModel,
    memory: Mapping[str, object],
    origin: np.ndarray,
    schedule_history: Sequence[Mapping[str, object]],
    refit_history: Sequence[Mapping[str, object]],
    geometry: Mapping[str, object] | None,
) -> dict[str, object]:
    keys = tuple(sorted(registered_role_keys(registry, refit_role)))
    if len(keys) != model.n_anchors:
        raise RuntimeError("registered refit role and front anchors disagree")
    index = _member_check(member_index)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_schema": "tuji_coherent_marked_front_v9",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "phase": str(phase),
        "refit_role": str(refit_role),
        "candidate": _candidate_check(candidate),
        "member_index": index,
        "member_seed": int(MEMBER_SEEDS[index - 1]),
        "selected_epoch": int(selected_epoch),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "coord_origin": torch.as_tensor(origin, dtype=torch.float32),
        "stats": dict(memory.get("stats", {})),
        "anchor_coordinate_keys": keys,
        "schedule_history": list(schedule_history),
        "refit_history": list(refit_history),
        "geometry_diagnostics": (
            None if geometry is None else dict(geometry["diagnostics"])
        ),
        "config": {
            **COMMON_FRONT,
            "model_schema": "coherent_marked_front_v9",
            "scan_feature_mode": (
                "bounded_maximal_invariant"
                if candidate == "geometry_invariant"
                else "legacy"
            ),
            "prototype_metric": (
                "none" if candidate == "geometry_invariant" else "l2"
            ),
            "visible_threshold": VISIBLE_THRESHOLD,
            "historical_front": dict(HISTORICAL_FRONT),
        },
    }


def train_selection_member(
    *,
    candidate: str,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_EXECUTION_CONTRACT,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict[str, object]:
    """Select one epoch schedule and refit that member on selection-refit."""

    registry = load_registry(registry_path)
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    name = _candidate_check(candidate)
    index = _member_check(member_index)
    destination = selection_checkpoint_path(name, index)
    if destination.exists():
        raise FileExistsError(destination)
    data = _load_training_data(registry)
    fit_rows = registered_rows(registry, "member_fit", data)
    stop_rows = registered_rows(registry, "early_stop", data)
    refit_rows = registered_rows(registry, "selection_refit", data)
    device = torch.device(device_name)
    seed = int(MEMBER_SEEDS[index - 1])
    started = time.perf_counter()
    selected_epoch, schedule = _fit_schedule(
        data,
        fit_rows,
        stop_rows,
        candidate=name,
        seed=seed,
        device=device,
        log_every=log_every,
    )
    model, memory, origin, refit, geometry = _fit_fixed(
        data,
        refit_rows,
        candidate=name,
        seed=seed,
        epochs=selected_epoch,
        device=device,
        log_every=log_every,
    )
    memory = dict(memory)
    memory["stats"] = data["stats"]
    payload = _checkpoint_payload(
        registry=registry,
        contract=contract,
        phase="selection",
        refit_role="selection_refit",
        candidate=name,
        member_index=index,
        selected_epoch=selected_epoch,
        model=model,
        memory=memory,
        origin=origin,
        schedule_history=schedule,
        refit_history=refit,
        geometry=geometry,
    )
    payload["training_seconds"] = float(time.perf_counter() - started)
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "candidate": name,
        "member_index": index,
        "member_seed": seed,
        "selected_epoch": int(selected_epoch),
        "anchors": int(model.n_anchors),
        "training_seconds": payload["training_seconds"],
        "test_model_outcome_accesses": 0,
    }


def load_front_checkpoint(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    device: torch.device,
) -> tuple[AnchorStudentTModel, dict[str, object]]:
    """Load only a checkpoint made under this TUJI execution contract."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("checkpoint_schema") != "tuji_coherent_marked_front_v9"
        or config.get("model_schema") != "coherent_marked_front_v9"
    ):
        raise ValueError(f"not a TUJI coherent V9 checkpoint: {path}")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("checkpoint registry mismatch")
    if payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]:
        raise ValueError("checkpoint execution-contract mismatch")
    state = payload["model_state"]
    candidate = _candidate_check(str(payload["candidate"]))
    geometry = None
    if candidate == "geometry_invariant":
        geometry = build_delaunay_geometry(
            state["anchor_coords"], state["anchor_buildings"], state["anchor_floors"]
        )
    model = AnchorStudentTModel(
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
        max_residual_m=float(HISTORICAL_FRONT["max_residual_m"]),
        scan_feature_mode=str(config["scan_feature_mode"]),
        prototype_metric=str(config["prototype_metric"]),
        local_residual_transform=(
            None if geometry is None else geometry["residual_transform"]
        ),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def _load_member_panel(
    paths: Sequence[Path],
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    expected_phase: str,
    expected_candidate: str,
    expected_role: str,
    device: torch.device,
) -> tuple[list[AnchorStudentTModel], list[dict[str, object]]]:
    if not paths:
        raise ValueError("at least one member checkpoint is required")
    models: list[AnchorStudentTModel] = []
    payloads: list[dict[str, object]] = []
    for expected_index, path in enumerate(paths, start=1):
        model, payload = load_front_checkpoint(
            path, registry=registry, contract=contract, device=device
        )
        if payload.get("phase") != expected_phase:
            raise ValueError(f"checkpoint phase mismatch: {path}")
        if payload.get("refit_role") != expected_role:
            raise ValueError(f"checkpoint refit-role mismatch: {path}")
        if payload.get("candidate") != expected_candidate:
            raise ValueError(f"checkpoint candidate mismatch: {path}")
        if int(payload.get("member_index", -1)) != expected_index:
            raise ValueError("member panel must use the preregistered prefix order")
        if int(payload.get("member_seed", -1)) != int(MEMBER_SEEDS[expected_index - 1]):
            raise ValueError("member seed order changed")
        models.append(model)
        payloads.append(payload)
    first_keys = payloads[0]["anchor_coordinate_keys"]
    first_origin = payloads[0]["coord_origin"]
    for payload in payloads[1:]:
        if payload["anchor_coordinate_keys"] != first_keys:
            raise ValueError("member anchor dictionaries are not aligned")
        if not torch.equal(payload["coord_origin"], first_origin):
            raise ValueError("member coordinate origins are not aligned")
    return models, payloads


def fit_radio_memory(
    *,
    candidate: str,
    phase: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_EXECUTION_CONTRACT,
    freeze_path: str | Path = DEFAULT_FREEZE,
) -> dict[str, object]:
    """Fit device-hierarchical marked memory without localization outcomes."""

    registry = load_registry(registry_path)
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    name = _candidate_check(candidate)
    if phase == "selection":
        role = "selection_refit"
    elif phase == "final":
        freeze = load_selection_freeze(
            freeze_path, registry=registry, contract=contract
        )
        if freeze["selected"]["candidate"] != name:
            raise ValueError("final radio memory must use the frozen candidate")
        role = "final_refit"
    else:
        raise ValueError("phase must be selection or final")
    destination = radio_memory_path(name, phase)
    if destination.exists():
        raise FileExistsError(destination)
    data = _load_training_data(registry)
    rows = registered_rows(registry, role, data)
    represented = _represented_rss(data["rss"], name)
    anchor_keys = tuple(sorted(registered_role_keys(registry, role)))
    memory = fit_hierarchical_marked_memory(
        rss=represented,
        phone_ids=data["phone"],
        coordinate_keys=data["keys"],
        training_indices=rows,
        anchor_keys=anchor_keys,
        detection_concentration_grid=CONCENTRATION_GRID,
        center_concentration_grid=CONCENTRATION_GRID,
        scale_concentration_grid=CONCENTRATION_GRID,
        degrees_freedom_grid=(STUDENT_DF,),
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "radio_memory_schema": "tuji_hierarchical_marked_memory_v9",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "candidate": name,
        "phase": phase,
        "refit_role": role,
        "role_coordinate_sha256": registry["roles"][role]["coordinate_key_sha256"],
        "memory": memory,
        "outcome_access": "none: RSS and training device IDs only",
    }
    _save_torch_new(destination, payload)
    radio = memory["radio_only_selection"]
    return {
        "radio_memory": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "candidate": name,
        "phase": phase,
        "anchors": len(anchor_keys),
        "selected_detection_concentration": radio["detection"]["selected_concentration"],
        "selected_center_concentration": radio["conditional_mark"]["selected_center_concentration"],
        "selected_scale_concentration": radio["conditional_mark"]["selected_scale_concentration"],
        "test_model_outcome_accesses": 0,
    }


def _load_radio_memory(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    candidate: str,
    phase: str,
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("radio_memory_schema") != "tuji_hierarchical_marked_memory_v9"
    ):
        raise ValueError("not TUJI V9 marked memory")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("radio-memory registry mismatch")
    if payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]:
        raise ValueError("radio-memory execution-contract mismatch")
    if payload.get("candidate") != candidate or payload.get("phase") != phase:
        raise ValueError("radio-memory candidate/phase mismatch")
    return payload


def _coordinate_groups(
    keys: Sequence[LocationKey],
) -> tuple[torch.Tensor, tuple[LocationKey, ...]]:
    unique = tuple(sorted(set(keys)))
    mapping = {key: index for index, key in enumerate(unique)}
    return torch.tensor([mapping[key] for key in keys], dtype=torch.long), unique


def _build_prediction_cache(
    *,
    query: torch.Tensor,
    models: Sequence[AnchorStudentTModel],
    payloads: Sequence[Mapping[str, object]],
    candidate: str,
    device: torch.device,
) -> dict[str, object]:
    """Create label-free front outputs for development or final test queries."""

    if not models:
        raise ValueError("prediction cache needs at least one member")
    name = _candidate_check(candidate)
    anchors = int(models[0].n_anchors)
    member_count = len(models)
    rows = int(len(query))
    logits_cache = torch.empty(member_count, rows, anchors)
    location_cache = torch.empty(member_count, rows, anchors, 2)
    scale_cache = torch.empty_like(location_cache)
    width = int(COMMON_FRONT["evaluation_batch_size"])
    for member_index, model in enumerate(models):
        for start in range(0, rows, width):
            stop = min(start + width, rows)
            values = query[start:stop].to(device)
            logits, embedding = _front_logits_embedding(model, values, name)
            all_anchor = torch.arange(
                anchors, device=device, dtype=torch.long
            ).unsqueeze(0).expand(len(values), -1)
            location, scale = model._components(embedding, all_anchor)
            logits_cache[member_index, start:stop] = logits.cpu()
            location_cache[member_index, start:stop] = location.cpu()
            scale_cache[member_index, start:stop] = scale.cpu()
    return {
        "query": query.float().cpu(),
        "member_logits": logits_cache,
        "member_locations": location_cache,
        "member_scales": scale_cache,
        "anchor_coords": models[0].anchor_coords.detach().cpu(),
        "anchor_buildings": models[0].anchor_buildings.detach().cpu(),
        "anchor_floors": models[0].anchor_floors.detach().cpu(),
        "anchor_coordinate_keys": payloads[0]["anchor_coordinate_keys"],
        "coord_origin": payloads[0]["coord_origin"],
        "randomness": common_student_t_randomness(
            rows,
            draws=SCORING_DRAWS,
            degrees_freedom=STUDENT_DF,
            seed=SCORING_SEED,
        ),
    }


def build_development_cache(
    *,
    candidate: str,
    members: int = INITIAL_MEMBER_COUNT,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_EXECUTION_CONTRACT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Cache training-only development predictions for one member prefix."""

    registry = load_registry(registry_path)
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    name = _candidate_check(candidate)
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    destination = development_cache_path(name, count)
    if destination.exists():
        raise FileExistsError(destination)
    paths = [selection_checkpoint_path(name, index) for index in range(1, count + 1)]
    device = torch.device(device_name)
    models, payloads = _load_member_panel(
        paths,
        registry=registry,
        contract=contract,
        expected_phase="selection",
        expected_candidate=name,
        expected_role="selection_refit",
        device=device,
    )
    radio_file = radio_memory_path(name, "selection")
    radio = _load_radio_memory(
        radio_file,
        registry=registry,
        contract=contract,
        candidate=name,
        phase="selection",
    )
    if tuple(radio["memory"]["anchor_keys"]) != tuple(payloads[0]["anchor_coordinate_keys"]):
        raise ValueError("radio memory and member anchors are not aligned")
    data = _load_training_data(registry)
    rows = registered_rows(registry, "development", data)
    represented = _represented_rss(data["rss"], name)
    cache = _build_prediction_cache(
        query=represented[rows],
        models=models,
        payloads=payloads,
        candidate=name,
        device=device,
    )
    origin = payloads[0]["coord_origin"].numpy().astype(np.float32)
    keys = [data["keys"][int(row)] for row in rows]
    group_ids, group_keys = _coordinate_groups(keys)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema": "tuji_v9_development_cache",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
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
        "radio_fields": radio["memory"]["fields"],
        "radio_degrees_freedom": radio["memory"]["degrees_freedom"],
        "radio_memory_path": str(radio_file.resolve()),
        "radio_memory_sha256": _sha256_file(radio_file),
        "member_paths": tuple(str(path.resolve()) for path in paths),
        "member_sha256": tuple(_sha256_file(path) for path in paths),
        **cache,
    }
    _save_torch_new(destination, payload)
    return {
        "cache": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "candidate": name,
        "members": count,
        "rows": int(len(rows)),
        "coordinate_groups": len(group_keys),
        "anchors": int(models[0].n_anchors),
        "test_model_outcome_accesses": 0,
    }


def load_development_cache(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    candidate: str,
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("cache_schema") != "tuji_v9_development_cache"
    ):
        raise ValueError("not a TUJI V9 development cache")
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


def _group_vector(
    row_values: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    if row_values.ndim != 1 or group_ids.shape != row_values.shape:
        raise ValueError("row values and group IDs must be aligned vectors")
    total = torch.zeros(group_count, dtype=torch.float64)
    count = torch.zeros(group_count, dtype=torch.float64)
    total.scatter_add_(0, group_ids.long(), row_values.double())
    count.scatter_add_(0, group_ids.long(), torch.ones_like(row_values).double())
    if bool((count <= 0).any()):
        raise RuntimeError("an empty physical-coordinate group entered scoring")
    return (total / count).float()


def _support_components(
    cache: Mapping[str, object],
    anchor_indices: torch.Tensor,
    *,
    members: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(members)
    all_locations = cache["member_locations"][:count].permute(1, 0, 2, 3)
    all_scales = cache["member_scales"][:count].permute(1, 0, 2, 3)
    gather = anchor_indices[:, None, :, None].expand(-1, count, -1, 2)
    return torch.gather(all_locations, 2, gather), torch.gather(all_scales, 2, gather)


def _marked_energy_for_projection(
    cache: Mapping[str, object],
    projection: Mapping[str, torch.Tensor | int],
    *,
    candidate: str,
    device: torch.device,
    batch_size: int = 96,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    name = _candidate_check(candidate)
    indices = projection["anchor_indices"]
    joint = projection["joint_probability"]
    if not isinstance(indices, torch.Tensor) or not isinstance(joint, torch.Tensor):
        raise TypeError("projection tensors are missing")
    anchor_prior = joint.sum(dim=1)
    fields = {key: value.to(device) for key, value in cache["radio_fields"].items()}
    output: list[torch.Tensor] = []
    offsets: list[torch.Tensor] = []
    gains: list[torch.Tensor] = []
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        query = cache["query"][start:stop].to(device)
        local = indices[start:stop].to(device)
        if name == "geometry_invariant":
            energy = marked_energy_on_support(
                query, local, fields, degrees_freedom=STUDENT_DF
            )
        else:
            energy, nuisance = profile_shared_affine_marked_energy(
                query,
                local,
                fields,
                anchor_prior[start:stop].to(device),
                degrees_freedom=STUDENT_DF,
                **AFFINE_PROFILE,
            )
            offsets.append(nuisance["style_offset"].detach().cpu())
            gains.append(nuisance["style_scale"].detach().cpu())
        output.append(energy.detach().cpu())
    diagnostics: dict[str, torch.Tensor] = {}
    if offsets:
        diagnostics = {
            "style_offset": torch.cat(offsets),
            "style_scale": torch.cat(gains),
        }
    return torch.cat(output), diagnostics


def _uncertainty_trace_components(
    posterior: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace decomposition: Student-t within + candidate + member dispersion."""

    if posterior.ndim != 3 or locations.shape[:3] != posterior.shape:
        raise ValueError("posterior and location component dimensions disagree")
    if scales.shape != locations.shape:
        raise ValueError("component scales and locations disagree")
    member_mass = posterior.sum(dim=2)
    conditional = posterior / member_mass.unsqueeze(2).clamp_min(1e-30)
    member_mean = (conditional.unsqueeze(3) * locations).sum(dim=2)
    mixture_mean = (member_mass.unsqueeze(2) * member_mean).sum(dim=1)
    df = float(STUDENT_DF)
    component_variance_trace = scales.square().sum(dim=3) * (df / (df - 2.0))
    within = (posterior * component_variance_trace).sum(dim=(1, 2))
    candidate_delta = locations - member_mean[:, :, None, :]
    candidate = (
        posterior * candidate_delta.square().sum(dim=3)
    ).sum(dim=(1, 2))
    member_delta = member_mean - mixture_mean[:, None, :]
    between = (member_mass * member_delta.square().sum(dim=2)).sum(dim=1)
    return within, candidate, between


def _configuration_metrics(
    cache: Mapping[str, object],
    projection: Mapping[str, torch.Tensor | int],
    marked_energy: torch.Tensor,
    *,
    members: int,
    lambda_value: float,
    target_xy: torch.Tensor,
    group_ids: torch.Tensor,
    group_keys: Sequence[LocationKey],
    device: torch.device,
    evaluation_batch_size: int = 256,
) -> dict[str, object]:
    """Score a complete continuous posterior using physical-coordinate means."""

    prior = projection["joint_probability"]
    indices = projection["anchor_indices"]
    retained = projection["retained_mass"]
    reverse_kl = projection["reverse_kl"]
    if not all(isinstance(value, torch.Tensor) for value in (prior, indices, retained, reverse_kl)):
        raise TypeError("projection tensors are incomplete")
    posterior = marked_kl_update(prior, marked_energy, lambda_value=float(lambda_value))
    locations, scales = _support_components(cache, indices, members=int(members))
    rows = len(posterior)
    if target_xy.shape != (rows, 2):
        raise ValueError("target coordinates do not match cached query rows")
    energy_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    mean_parts: list[torch.Tensor] = []
    entropy_parts: list[torch.Tensor] = []
    coverage50_parts: list[torch.Tensor] = []
    coverage90_parts: list[torch.Tensor] = []
    within_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    member_parts: list[torch.Tensor] = []
    for start in range(0, rows, int(evaluation_batch_size)):
        stop = min(start + int(evaluation_batch_size), rows)
        probability = posterior[start:stop].reshape(stop - start, -1).to(device)
        location = locations[start:stop].reshape(stop - start, -1, 2).to(device)
        scale = scales[start:stop].reshape(stop - start, -1, 2).to(device)
        randomness = {
            key: (
                value[start:stop].to(device)
                if isinstance(value, torch.Tensor) and value.ndim > 0
                else value
            )
            for key, value in cache["randomness"].items()
        }
        row_energy, first, second = continuous_energy_score_mc(
            probability,
            location,
            scale,
            target_xy[start:stop].to(device),
            randomness,
        )
        action = spatial_median_from_draws(first, second)
        mixture_mean = (probability[:, :, None] * location).sum(dim=1)
        sampled = torch.cat((first, second), dim=1)
        sampled_radius = torch.linalg.vector_norm(
            sampled - mixture_mean[:, None, :], dim=2
        )
        truth_radius = torch.linalg.vector_norm(
            target_xy[start:stop].to(device) - mixture_mean, dim=1
        )
        coverage50 = truth_radius <= torch.quantile(sampled_radius, 0.50, dim=1)
        coverage90 = truth_radius <= torch.quantile(sampled_radius, 0.90, dim=1)
        local_posterior = posterior[start:stop].to(device)
        local_locations = locations[start:stop].to(device)
        local_scales = scales[start:stop].to(device)
        within, candidate, member = _uncertainty_trace_components(
            local_posterior, local_locations, local_scales
        )
        energy_parts.append(row_energy.detach().cpu())
        action_parts.append(action.detach().cpu())
        mean_parts.append(mixture_mean.detach().cpu())
        entropy_parts.append(
            -(probability * probability.clamp_min(1e-30).log()).sum(dim=1).detach().cpu()
        )
        coverage50_parts.append(coverage50.float().detach().cpu())
        coverage90_parts.append(coverage90.float().detach().cpu())
        within_parts.append(within.detach().cpu())
        candidate_parts.append(candidate.detach().cpu())
        member_parts.append(member.detach().cpu())
    row_energy = torch.cat(energy_parts)
    action = torch.cat(action_parts)
    mixture_mean = torch.cat(mean_parts)
    row_mae = torch.linalg.vector_norm(action - target_xy, dim=1)
    row_mean_error = torch.linalg.vector_norm(mixture_mean - target_xy, dim=1)
    group_count = len(group_keys)
    group_energy = _group_vector(row_energy, group_ids, group_count)
    group_mae = _group_vector(row_mae, group_ids, group_count)
    group_coverage50 = _group_vector(torch.cat(coverage50_parts), group_ids, group_count)
    group_coverage90 = _group_vector(torch.cat(coverage90_parts), group_ids, group_count)
    group_within = _group_vector(torch.cat(within_parts), group_ids, group_count)
    group_candidate = _group_vector(torch.cat(candidate_parts), group_ids, group_count)
    group_member = _group_vector(torch.cat(member_parts), group_ids, group_count)
    entropy = torch.cat(entropy_parts)
    return {
        "E": int(members),
        "K": int(indices.shape[1]),
        "lambda": float(lambda_value),
        "coordinate_macro_energy_score_2d_m": float(group_energy.mean()),
        "coordinate_macro_mae_2d_m": float(group_mae.mean()),
        "row_energy_score_2d_m": float(row_energy.mean()),
        "row_mae_2d_m": float(row_mae.mean()),
        "row_mixture_mean_error_2d_m": float(row_mean_error.mean()),
        "mean_retained_prior_mass": float(retained.mean()),
        "q10_retained_prior_mass": float(torch.quantile(retained, 0.10)),
        "mean_truncation_reverse_kl_nat": float(reverse_kl.mean()),
        "mean_posterior_entropy_nat": float(entropy.mean()),
        "coordinate_macro_empirical_radial_50_coverage": float(group_coverage50.mean()),
        "coordinate_macro_empirical_radial_90_coverage": float(group_coverage90.mean()),
        "coordinate_macro_within_component_covariance_trace_m2": float(group_within.mean()),
        "coordinate_macro_candidate_ambiguity_covariance_trace_m2": float(group_candidate.mean()),
        "coordinate_macro_member_disagreement_covariance_trace_m2": float(group_member.mean()),
        "group_energy_score_2d_m": group_energy,
        "group_mae_2d_m": group_mae,
        "group_coverage50": group_coverage50,
        "group_coverage90": group_coverage90,
        "group_within_trace_m2": group_within,
        "group_candidate_trace_m2": group_candidate,
        "group_member_trace_m2": group_member,
        "row_mae_vector": row_mae,
        "spatial_median_xy_centered": action,
    }


def evaluate_development_grid(
    *,
    candidate: str,
    members: int = INITIAL_MEMBER_COUNT,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_EXECUTION_CONTRACT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Evaluate every registered ``(E,K,lambda)`` on development only."""

    registry = load_registry(registry_path)
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    name = _candidate_check(candidate)
    count = int(members)
    source = development_cache_path(name, count)
    destination = grid_result_path(name, count)
    summary_destination = grid_summary_path(name, count)
    if destination.exists() or summary_destination.exists():
        raise FileExistsError("refusing to overwrite a TUJI development grid")
    cache = load_development_cache(source, registry=registry, contract=contract, candidate=name)
    available = int(cache["member_logits"].shape[0])
    if available != count:
        raise ValueError("cache member count mismatches requested grid")
    base_grid = None
    configurations: list[dict[str, object]] = []
    first_new_e = 1
    if count > INITIAL_MEMBER_COUNT:
        previous = count - 2
        base_path = grid_result_path(name, previous)
        base_grid = _load_grid(base_path, registry=registry, contract=contract, candidate=name)
        if int(base_grid["available_members"]) != previous:
            raise ValueError("extension base grid has the wrong member count")
        base_cache = load_development_cache(
            development_cache_path(name, previous),
            registry=registry,
            contract=contract,
            candidate=name,
        )
        if tuple(base_cache["member_sha256"]) != tuple(cache["member_sha256"][:previous]):
            raise ValueError("extension cache did not preserve the member prefix")
        for key in ("role_coordinate_sha256", "row_index_sha256", "radio_memory_sha256"):
            if base_cache[key] != cache[key]:
                raise ValueError(f"extension changed registered development field: {key}")
        configurations = list(base_grid["configurations"])
        first_new_e = previous + 1
    device = torch.device(device_name)
    started = time.perf_counter()
    for e_count in range(first_new_e, available + 1):
        full_joint = uniform_member_anchor_prior(cache["member_logits"], members=e_count)
        for top_k in K_GRID:
            if top_k > full_joint.shape[2]:
                continue
            projection = topk_prior_projection(full_joint, top_k=top_k)
            marked, nuisance = _marked_energy_for_projection(
                cache, projection, candidate=name, device=device
            )
            for lambda_value in LAMBDA_GRID:
                metrics = _configuration_metrics(
                    cache,
                    projection,
                    marked,
                    members=e_count,
                    lambda_value=float(lambda_value),
                    target_xy=cache["target_xy"],
                    group_ids=cache["coordinate_group_ids"],
                    group_keys=cache["coordinate_group_keys"],
                    device=device,
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
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )
            del marked, projection
            if nuisance and device.type == "cuda":
                torch.cuda.empty_cache()
        del full_joint
    payload: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "grid_schema": "tuji_v9_joint_development_grid",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "candidate": name,
        "cache": str(source.resolve()),
        "cache_sha256": _sha256_file(source),
        "available_members": available,
        "inherited_base_grid": (
            None
            if base_grid is None
            else {
                "path": str(grid_result_path(name, count - 2).resolve()),
                "sha256": _sha256_file(grid_result_path(name, count - 2)),
                "available_members": count - 2,
            }
        ),
        "coordinate_group_keys": cache["coordinate_group_keys"],
        "configurations": configurations,
        "elapsed_seconds": float(time.perf_counter() - started),
        "test_model_outcome_accesses": 0,
    }
    _save_torch_new(destination, payload)
    profiled = []
    for e_count in range(1, available + 1):
        selected = min(
            (row for row in configurations if int(row["E"]) == e_count),
            key=lambda row: float(row["coordinate_macro_energy_score_2d_m"]),
        )
        profiled.append(
            {key: value for key, value in selected.items() if not isinstance(value, torch.Tensor)}
        )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "grid_schema": "tuji_v9_joint_development_grid_summary",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "candidate": name,
        "grid": str(destination.resolve()),
        "grid_sha256": _sha256_file(destination),
        "cache_sha256": payload["cache_sha256"],
        "available_members": available,
        "profiled_by_E": profiled,
        "elapsed_seconds": payload["elapsed_seconds"],
        "test_model_outcome_accesses": 0,
    }
    _write_json_new(summary_destination, summary)
    return summary


def _load_grid(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    candidate: str,
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("grid_schema") != "tuji_v9_joint_development_grid"
    ):
        raise ValueError("not a TUJI V9 development grid")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("grid registry mismatch")
    if payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]:
        raise ValueError("grid execution-contract mismatch")
    if payload.get("candidate") != candidate:
        raise ValueError("grid candidate mismatch")
    if _sha256_file(payload["cache"]) != payload["cache_sha256"]:
        raise ValueError("development cache changed after grid scoring")
    return payload


def _profile_grid(grid: Mapping[str, object]) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for e_count in range(1, int(grid["available_members"]) + 1):
        rows = [row for row in grid["configurations"] if int(row["E"]) == e_count]
        if not rows:
            raise RuntimeError(f"grid misses E={e_count}")
        result[e_count] = min(
            rows, key=lambda row: float(row["coordinate_macro_energy_score_2d_m"])
        )
    return result


def _selection_freeze_without_hash(payload: Mapping[str, object]) -> dict[str, object]:
    value = copy.deepcopy(dict(payload))
    value.pop("freeze_sha256", None)
    return value


def select_and_freeze(
    *,
    members: int = INITIAL_MEMBER_COUNT,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_EXECUTION_CONTRACT,
    output: str | Path = DEFAULT_FREEZE,
) -> dict[str, object]:
    """Compare bundled candidates, test E saturation, and freeze once."""

    registry = load_registry(registry_path)
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    count = int(members)
    if count not in EXTENSION_MEMBER_COUNTS:
        raise ValueError(f"members must be one of {EXTENSION_MEMBER_COUNTS}")
    destination = _require_protocol_path(
        output, DEFAULT_FREEZE, label="selection freeze"
    )
    if destination.exists():
        raise FileExistsError(destination)
    grids = {
        candidate: _load_grid(
            grid_result_path(candidate, count),
            registry=registry,
            contract=contract,
            candidate=candidate,
        )
        for candidate in CANDIDATES
    }
    profiles = {candidate: _profile_grid(grid) for candidate, grid in grids.items()}
    common_max_e = min(max(profile) for profile in profiles.values())
    geometry = profiles["geometry_invariant"][common_max_e]
    historical = profiles["historical_affine"][common_max_e]
    structural_tolerance = 0.01 * float(
        profiles["historical_affine"][1]["coordinate_macro_energy_score_2d_m"]
    )
    energy_comparison = simultaneous_paired_upper_bounds(
        geometry["group_energy_score_2d_m"].numpy(),
        [historical["group_energy_score_2d_m"].numpy()],
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=PAIRED_INFERENCE_SEED,
    )
    mae_comparison = simultaneous_paired_upper_bounds(
        geometry["group_mae_2d_m"].numpy(),
        [historical["group_mae_2d_m"].numpy()],
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=PAIRED_INFERENCE_SEED + 100,
    )
    geometry_mae_eligible = bool(
        mae_comparison["simultaneous_upper_bounds"][0] <= MAE_NONINFERIORITY_MARGIN_M
    )
    geometry_energy_equivalent_or_better = bool(
        energy_comparison["mean_differences"][0] <= 0.0
        or energy_comparison["simultaneous_upper_bounds"][0] <= structural_tolerance
    )
    selected_candidate = (
        "geometry_invariant"
        if geometry_mae_eligible and geometry_energy_equivalent_or_better
        else "historical_affine"
    )
    selected_profile = profiles[selected_candidate]
    saturation_tolerance = 0.01 * float(
        selected_profile[1]["coordinate_macro_energy_score_2d_m"]
    )
    saturation = select_information_saturated_E(
        {e_count: row["group_energy_score_2d_m"].numpy() for e_count, row in selected_profile.items()},
        tolerance=saturation_tolerance,
        minimum_later_values=2,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=PAIRED_INFERENCE_SEED + 200,
    )
    selection_report: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_freeze_schema": "tuji_v9_selection_freeze",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "candidate_comparison_at_common_E": common_max_e,
        "structural_energy_tolerance_m": structural_tolerance,
        "energy_comparison_geometry_minus_historical": energy_comparison,
        "mae_comparison_geometry_minus_historical": mae_comparison,
        "geometry_mae_noninferior": geometry_mae_eligible,
        "geometry_energy_equivalent_or_better": geometry_energy_equivalent_or_better,
        "nominated_candidate": selected_candidate,
        "saturation": saturation,
        "grid_files": {
            candidate: {
                "path": str(grid_result_path(candidate, count).resolve()),
                "sha256": _sha256_file(grid_result_path(candidate, count)),
            }
            for candidate in CANDIDATES
        },
        "author_test_model_outcome_accesses_before_freeze": 0,
    }
    if not saturation["saturated"] and count < max(EXTENSION_MEMBER_COUNTS):
        next_count = count + 2
        report = {
            **selection_report,
            "status": "extension_required",
            "next_member_indices": list(range(count + 1, next_count + 1)),
            "next_member_count": next_count,
            "test_model_outcome_accesses": 0,
        }
        _write_json_new(extension_report_path(count), report)
        return report
    if saturation["saturated"]:
        selected_e = int(saturation["selected_E"])
        selection_basis = "confirmed_information_saturation"
        saturation_claim = True
        amendment = None
    else:
        selected_e = min(
            selected_profile,
            key=lambda value: (
                float(selected_profile[value]["coordinate_macro_energy_score_2d_m"]),
                int(value),
            ),
        )
        selection_basis = "budget_censored_development_risk_minimum"
        saturation_claim = False
        amendment = {
            "id": "TUJI-V9-A1-terminal-selection",
            "timing": "after_development_before_author_test",
            "reason": "registered panel reaches E=12 without confirmed saturation",
            "rule": "select global profiled development proper-risk minimum; make no saturation claim",
            "maximum_registered_E": max(EXTENSION_MEMBER_COUNTS),
        }
    selected_row = selected_profile[int(selected_e)]
    freeze: dict[str, object] = {
        **selection_report,
        "status": "selection_frozen",
        "selection_basis": selection_basis,
        "saturation_claim": saturation_claim,
        "protocol_amendment": amendment,
        "selected": {
            "candidate": selected_candidate,
            "E": int(selected_e),
            "K": int(selected_row["K"]),
            "lambda": float(selected_row["lambda"]),
            "development_coordinate_macro_energy_score_2d_m": float(selected_row["coordinate_macro_energy_score_2d_m"]),
            "development_coordinate_macro_mae_2d_m": float(selected_row["coordinate_macro_mae_2d_m"]),
            "member_indices": list(range(1, int(selected_e) + 1)),
            "member_seeds": list(MEMBER_SEEDS[: int(selected_e)]),
        },
    }
    freeze["freeze_sha256"] = _canonical_json_sha256(freeze, omit="freeze_sha256")
    _write_json_new(destination, freeze)
    return freeze


def load_selection_freeze(
    path: str | Path = DEFAULT_FREEZE,
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, object]:
    freeze_file = _require_protocol_path(
        path, DEFAULT_FREEZE, label="selection freeze"
    )
    if not freeze_file.exists():
        raise PermissionError(
            "final refit and author-test access are forbidden before TUJI selection freeze"
        )
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    if (
        freeze.get("protocol_version") != PROTOCOL_VERSION
        or freeze.get("selection_freeze_schema") != "tuji_v9_selection_freeze"
        or freeze.get("status") != "selection_frozen"
    ):
        raise ValueError("not a frozen TUJI V9 selection record")
    if freeze.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("selection freeze registry mismatch")
    if freeze.get("execution_contract_sha256") != contract["execution_contract_sha256"]:
        raise ValueError("selection freeze execution-contract mismatch")
    if freeze.get("freeze_sha256") != _canonical_json_sha256(
        freeze, omit="freeze_sha256"
    ):
        raise ValueError("selection freeze self-hash failed")
    if freeze.get("author_test_model_outcome_accesses_before_freeze") != 0:
        raise ValueError("author test was accessed before selection freeze")
    for record in freeze["grid_files"].values():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("frozen development grid changed")
    return freeze


def train_final_member(
    *,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_EXECUTION_CONTRACT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict[str, object]:
    """Refit a frozen selected member on all 1,351 author-training coordinates."""

    registry = load_registry(registry_path)
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    freeze = load_selection_freeze(freeze_path, registry=registry, contract=contract)
    index = _member_check(member_index)
    selected = freeze["selected"]
    if index not in selected["member_indices"]:
        raise ValueError("member is outside the frozen deployment prefix")
    candidate = str(selected["candidate"])
    destination = final_checkpoint_path(candidate, index)
    if destination.exists():
        raise FileExistsError(destination)
    selection_path = selection_checkpoint_path(candidate, index)
    _selection_model, selection_payload = load_front_checkpoint(
        selection_path, registry=registry, contract=contract, device=torch.device("cpu")
    )
    epoch = int(selection_payload["selected_epoch"])
    data = _load_training_data(registry)
    rows = registered_rows(registry, "final_refit", data)
    device = torch.device(device_name)
    started = time.perf_counter()
    model, memory, origin, history, geometry = _fit_fixed(
        data,
        rows,
        candidate=candidate,
        seed=int(MEMBER_SEEDS[index - 1]),
        epochs=epoch,
        device=device,
        log_every=log_every,
    )
    memory = dict(memory)
    memory["stats"] = data["stats"]
    payload = _checkpoint_payload(
        registry=registry,
        contract=contract,
        phase="final",
        refit_role="final_refit",
        candidate=candidate,
        member_index=index,
        selected_epoch=epoch,
        model=model,
        memory=memory,
        origin=origin,
        schedule_history=selection_payload["schedule_history"],
        refit_history=history,
        geometry=geometry,
    )
    payload["selection_checkpoint"] = str(selection_path.resolve())
    payload["selection_checkpoint_sha256"] = _sha256_file(selection_path)
    payload["freeze_sha256"] = freeze["freeze_sha256"]
    payload["training_seconds"] = float(time.perf_counter() - started)
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "candidate": candidate,
        "member_index": index,
        "selected_epoch": epoch,
        "anchors": int(model.n_anchors),
        "training_seconds": payload["training_seconds"],
    }


def load_tuji_test_rss_only(registry: Mapping[str, object]) -> dict[str, object]:
    """Load query RSS alone; deliberately do not open coordinates/devices."""

    sources = registry["source_files"]
    record = sources["rss_testing"]
    path = Path(str(record["path"])).resolve()
    if _sha256_file(path) != record["sha256"]:
        raise ValueError("author test RSS changed after registration")
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != int(registry["data_contract"]["n_aps"]):
        raise ValueError("TUJI author test RSS does not meet the frozen 310-AP contract")
    normalized, missing = _normalized_rss(raw)
    return {
        "rss": torch.from_numpy(normalized).float(),
        "raw_visible_count": (~missing).sum(axis=1).astype(np.int64),
        "rows": int(len(raw)),
        "rss_path": str(path),
    }


def cache_author_test(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_EXECUTION_CONTRACT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
) -> dict[str, object]:
    """The only prediction command that may read author-test RSS.

    Selection freeze is checked *before* ``RSS_testing.csv`` is opened.  This
    function does not receive or load test coordinates or test device labels.
    """

    registry = load_registry(registry_path)
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    freeze = load_selection_freeze(freeze_path, registry=registry, contract=contract)
    destination = Path(DEFAULT_TEST_CACHE)
    if destination.exists():
        raise FileExistsError(destination)
    selected = freeze["selected"]
    candidate = str(selected["candidate"])
    members = int(selected["E"])
    paths = [final_checkpoint_path(candidate, index) for index in range(1, members + 1)]
    device = torch.device(device_name)
    models, payloads = _load_member_panel(
        paths,
        registry=registry,
        contract=contract,
        expected_phase="final",
        expected_candidate=candidate,
        expected_role="final_refit",
        device=device,
    )
    radio_file = radio_memory_path(candidate, "final")
    radio = _load_radio_memory(
        radio_file,
        registry=registry,
        contract=contract,
        candidate=candidate,
        phase="final",
    )
    if tuple(radio["memory"]["anchor_keys"]) != tuple(payloads[0]["anchor_coordinate_keys"]):
        raise ValueError("final radio memory and final member anchors are not aligned")
    query_data = load_tuji_test_rss_only(registry)
    represented = _represented_rss(query_data["rss"], candidate)
    cache = _build_prediction_cache(
        query=represented,
        models=models,
        payloads=payloads,
        candidate=candidate,
        device=device,
    )
    raw_visible = query_data["raw_visible_count"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema": "tuji_v9_author_test_prediction_cache",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "candidate": candidate,
        "members": members,
        "rows": int(query_data["rows"]),
        "query_source": "RSS_testing.csv only",
        "test_labels_loaded_during_cache": False,
        "raw_visible_count": torch.from_numpy(raw_visible).long(),
        "raw_visible_above_64": int((raw_visible > int(COMMON_FRONT["max_visible"])).sum()),
        "radio_fields": radio["memory"]["fields"],
        "radio_degrees_freedom": radio["memory"]["degrees_freedom"],
        "radio_memory_path": str(radio_file.resolve()),
        "radio_memory_sha256": _sha256_file(radio_file),
        "member_paths": tuple(str(path.resolve()) for path in paths),
        "member_sha256": tuple(_sha256_file(path) for path in paths),
        **cache,
    }
    _save_torch_new(destination, payload)
    return {
        "author_test_cache": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "candidate": candidate,
        "members": members,
        "rows": int(query_data["rows"]),
        "test_labels_loaded_during_cache": False,
        "raw_visible_above_64": payload["raw_visible_above_64"],
    }


def load_author_test_cache(
    path: str | Path,
    *,
    registry: Mapping[str, object],
    contract: Mapping[str, object],
    freeze: Mapping[str, object],
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("cache_schema") != "tuji_v9_author_test_prediction_cache"
    ):
        raise ValueError("not a TUJI author-test prediction cache")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("author-test cache registry mismatch")
    if payload.get("execution_contract_sha256") != contract["execution_contract_sha256"]:
        raise ValueError("author-test cache execution-contract mismatch")
    if payload.get("freeze_sha256") != freeze["freeze_sha256"]:
        raise ValueError("author-test cache selection-freeze mismatch")
    if payload.get("candidate") != freeze["selected"]["candidate"]:
        raise ValueError("author-test cache candidate mismatch")
    if int(payload.get("members", -1)) != int(freeze["selected"]["E"]):
        raise ValueError("author-test cache member count mismatch")
    if payload.get("test_labels_loaded_during_cache") is not False:
        raise ValueError("author-test cache violates its label-access rule")
    return payload


def load_tuji_test_labels_only(
    registry: Mapping[str, object],
    *,
    expected_rows: int,
) -> dict[str, object]:
    """Load final score labels after prediction cache creation only."""

    sources = registry["source_files"]
    coordinate_record = sources["coordinates_testing"]
    device_record = sources["device_labels"]
    coordinate_path = Path(str(coordinate_record["path"])).resolve()
    device_path = Path(str(device_record["path"])).resolve()
    if _sha256_file(coordinate_path) != coordinate_record["sha256"]:
        raise ValueError("author test coordinates changed after registration")
    if _sha256_file(device_path) != device_record["sha256"]:
        raise ValueError("TUJI device-label map changed after registration")
    labels = np.loadtxt(coordinate_path, delimiter=",", dtype=np.float64)
    if labels.ndim != 2 or labels.shape != (int(expected_rows), 6):
        raise ValueError("author test coordinate labels do not meet the frozen shape")
    if not bool(np.equal(labels[:, 2:5], 0.0).all()):
        raise ValueError("TUJI author test is not the registered single 2-D plane")
    device_label = labels[:, 5]
    if not bool(np.equal(device_label, np.round(device_label)).all()):
        raise ValueError("author test device labels must be integral")
    phones = device_label.astype(np.int64) - 1
    if set(map(int, np.unique(phones))) != {0, 1, 2, 3, 4}:
        raise ValueError("author test device labels do not match the frozen five-device map")
    xy = labels[:, :2].astype(np.float32)
    keys: list[LocationKey] = [
        (0, 0, round(float(x), 6), round(float(y), 6)) for x, y in labels[:, :2]
    ]
    return {"raw_xy": xy, "phones": phones, "keys": keys}


def _device_macro_mae(
    row_mae: torch.Tensor,
    keys: Sequence[LocationKey],
    phones: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for phone in sorted(map(int, np.unique(phones))):
        selected = np.flatnonzero(phones == phone)
        local_keys = [keys[int(index)] for index in selected]
        group_ids, group_keys = _coordinate_groups(local_keys)
        local = row_mae[torch.from_numpy(selected).long()]
        macro = _group_vector(local, group_ids, len(group_keys))
        output[str(phone + 1)] = {
            "rows": int(len(selected)),
            "coordinate_groups": int(len(group_keys)),
            "coordinate_macro_mae_2d_m": float(macro.mean()),
        }
    return output


def evaluate_author_test(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    contract_path: str | Path = DEFAULT_EXECUTION_CONTRACT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    cache_path: str | Path = DEFAULT_TEST_CACHE,
    output: str | Path = DEFAULT_RESULT,
    device_name: str | None = None,
) -> dict[str, object]:
    """Score a frozen author-test prediction cache exactly once."""

    registry = load_registry(registry_path)
    contract = load_execution_contract(contract_path, registry_path=registry_path)
    freeze = load_selection_freeze(freeze_path, registry=registry, contract=contract)
    destination = _require_protocol_path(
        output, DEFAULT_RESULT, label="author-test result"
    )
    if destination.exists():
        raise FileExistsError(destination)
    source = _require_protocol_path(
        cache_path, DEFAULT_TEST_CACHE, label="author-test prediction cache"
    )
    cache = load_author_test_cache(
        source, registry=registry, contract=contract, freeze=freeze
    )
    labels = load_tuji_test_labels_only(registry, expected_rows=int(cache["rows"]))
    group_ids, group_keys = _coordinate_groups(labels["keys"])
    origin = cache["coord_origin"].numpy().astype(np.float32)
    target = torch.from_numpy((labels["raw_xy"] - origin[None, :]).astype(np.float32))
    selected = freeze["selected"]
    full_joint = uniform_member_anchor_prior(
        cache["member_logits"], members=int(selected["E"])
    )
    projection = topk_prior_projection(full_joint, top_k=int(selected["K"]))
    score_device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    marked, nuisance = _marked_energy_for_projection(
        cache, projection, candidate=str(selected["candidate"]), device=score_device
    )
    metrics = _configuration_metrics(
        cache,
        projection,
        marked,
        members=int(selected["E"]),
        lambda_value=float(selected["lambda"]),
        target_xy=target,
        group_ids=group_ids,
        group_keys=group_keys,
        device=score_device,
    )
    row_mae = metrics["row_mae_vector"]
    raw_visible = cache["raw_visible_count"].numpy()
    result: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "result_schema": "tuji_v9_author_test_result",
        "registry_sha256": registry["registry_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "freeze": str(Path(freeze_path).resolve()),
        "freeze_sha256": freeze["freeze_sha256"],
        "author_test_cache": str(source.resolve()),
        "author_test_cache_sha256": _sha256_file(source),
        "author_test_access_number": 1,
        "comparison_status": "external_replication_after_eligibility_audit",
        "selected": selected,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if not isinstance(value, torch.Tensor)
        },
        "coordinate_group_count": len(group_keys),
        "coordinate_group_energy_score_2d_m": metrics["group_energy_score_2d_m"].tolist(),
        "coordinate_group_mae_2d_m": metrics["group_mae_2d_m"].tolist(),
        "device_stratified_coordinate_macro_mae_2d_m": _device_macro_mae(
            row_mae, labels["keys"], labels["phones"]
        ),
        "top64_diagnostic": {
            "rows": int(len(raw_visible)),
            "raw_visible_above_64": int((raw_visible > int(COMMON_FRONT["max_visible"])).sum()),
            "raw_visible_above_64_fraction": float(
                (raw_visible > int(COMMON_FRONT["max_visible"])).mean()
            ),
            "maximum_raw_visible": int(raw_visible.max()),
            "mean_discarded_visible_aps_if_capped": float(
                np.maximum(raw_visible - int(COMMON_FRONT["max_visible"]), 0).mean()
            ),
        },
        "affine_nuisance_summary": (
            None
            if not nuisance
            else {
                "mean_offset": float(nuisance["style_offset"].mean()),
                "mean_scale": float(nuisance["style_scale"].mean()),
            }
        ),
        "metric_note": (
            "energy and MAE are coordinate-macro 2-D metrics; empirical radial "
            "coverage uses 128 common-random-number Student-t draws per scan"
        ),
    }
    result["result_sha256"] = _canonical_json_sha256(result, omit="result_sha256")
    _write_json_new(destination, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    freeze_execution = command.add_parser(
        "freeze-execution", help="write self-hashed TUJI executor contract"
    )
    freeze_execution.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    freeze_execution.add_argument("--output", default=str(DEFAULT_EXECUTION_CONTRACT))
    audit = command.add_parser("audit-execution", help="read-only pre-training runner audit")
    audit.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    audit.add_argument("--contract", default=str(DEFAULT_EXECUTION_CONTRACT))

    train = command.add_parser("train-selection-member")
    train.add_argument("--candidate", choices=CANDIDATES, required=True)
    train.add_argument("--member-index", type=int, required=True)
    train.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train.add_argument("--contract", default=str(DEFAULT_EXECUTION_CONTRACT))
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=5)

    radio = command.add_parser("fit-radio")
    radio.add_argument("--candidate", choices=CANDIDATES, required=True)
    radio.add_argument("--phase", choices=("selection", "final"), required=True)
    radio.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    radio.add_argument("--contract", default=str(DEFAULT_EXECUTION_CONTRACT))
    radio.add_argument("--freeze", default=str(DEFAULT_FREEZE))

    cache = command.add_parser("build-development-cache")
    cache.add_argument("--candidate", choices=CANDIDATES, required=True)
    cache.add_argument("--members", type=int, default=INITIAL_MEMBER_COUNT)
    cache.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    cache.add_argument("--contract", default=str(DEFAULT_EXECUTION_CONTRACT))
    cache.add_argument("--device", default="cuda")

    grid = command.add_parser("grid")
    grid.add_argument("--candidate", choices=CANDIDATES, required=True)
    grid.add_argument("--members", type=int, default=INITIAL_MEMBER_COUNT)
    grid.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    grid.add_argument("--contract", default=str(DEFAULT_EXECUTION_CONTRACT))
    grid.add_argument("--device", default="cuda")

    select = command.add_parser("freeze-selection")
    select.add_argument("--members", type=int, default=INITIAL_MEMBER_COUNT)
    select.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    select.add_argument("--contract", default=str(DEFAULT_EXECUTION_CONTRACT))
    select.add_argument("--output", default=str(DEFAULT_FREEZE))

    final = command.add_parser("train-final-member")
    final.add_argument("--member-index", type=int, required=True)
    final.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    final.add_argument("--contract", default=str(DEFAULT_EXECUTION_CONTRACT))
    final.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    final.add_argument("--device", default="cuda")
    final.add_argument("--log-every", type=int, default=5)

    author_cache = command.add_parser("cache-author-test")
    author_cache.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    author_cache.add_argument("--contract", default=str(DEFAULT_EXECUTION_CONTRACT))
    author_cache.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    author_cache.add_argument("--device", default="cuda")

    evaluate = command.add_parser("evaluate-author-test")
    evaluate.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    evaluate.add_argument("--contract", default=str(DEFAULT_EXECUTION_CONTRACT))
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
    elif args.command == "train-selection-member":
        result = train_selection_member(
            candidate=args.candidate,
            member_index=args.member_index,
            registry_path=args.registry,
            contract_path=args.contract,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "fit-radio":
        result = fit_radio_memory(
            candidate=args.candidate,
            phase=args.phase,
            registry_path=args.registry,
            contract_path=args.contract,
            freeze_path=args.freeze,
        )
    elif args.command == "build-development-cache":
        result = build_development_cache(
            candidate=args.candidate,
            members=args.members,
            registry_path=args.registry,
            contract_path=args.contract,
            device_name=args.device,
        )
    elif args.command == "grid":
        result = evaluate_development_grid(
            candidate=args.candidate,
            members=args.members,
            registry_path=args.registry,
            contract_path=args.contract,
            device_name=args.device,
        )
    elif args.command == "freeze-selection":
        result = select_and_freeze(
            members=args.members,
            registry_path=args.registry,
            contract_path=args.contract,
            output=args.output,
        )
    elif args.command == "train-final-member":
        result = train_final_member(
            member_index=args.member_index,
            registry_path=args.registry,
            contract_path=args.contract,
            freeze_path=args.freeze,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "cache-author-test":
        result = cache_author_test(
            registry_path=args.registry,
            contract_path=args.contract,
            freeze_path=args.freeze,
            device_name=args.device,
        )
    elif args.command == "evaluate-author-test":
        result = evaluate_author_test(
            registry_path=args.registry,
            contract_path=args.contract,
            freeze_path=args.freeze,
            cache_path=args.cache,
            output=args.output,
            device_name=args.device,
        )
    else:  # pragma: no cover - argparse closes this branch.
        raise RuntimeError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
