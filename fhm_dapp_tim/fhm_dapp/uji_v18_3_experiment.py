"""Registered V18.3 hierarchical robust-consensus UJI experiment.

Scientific status
-----------------
V18.3 is a one-checkpoint journal-method model.  It keeps the successful V18.2
multi-source candidate union but removes the collapsed learned six-way gate.

V18.3 uses deterministic hierarchical reliability, train-time evidence-family
 dropout, and disagreement-aware posterior softening.  It does not use V9/V17
checkpoints, teacher outputs, knowledge distillation, or an inference ensemble.

The completed V18.2 confirmation and the official UJI validation benchmark have
already informed model development.  Therefore this protocol makes only a
sequential, newly registered internal confirmation claim.  It does not claim a
new independent external test.  The official validation file remains a
pre-exposed descriptive benchmark and is never used for tuning.

All roles remain complete physical-coordinate groups with building-floor hull
support.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.audit_bplus_sparse50_fem_v5 import (
    _hull_local_indices,
    geometry_support_with_hull,
)
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness,
    continuous_energy_score_mc,
)
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _decode_key,
    _encode_key,
    _load_uji,
    _role_record,
    _rows_for_keys,
    _save_torch_new,
    _sha256_file,
    _sha256_indices,
    _sha256_keys,
    _write_json_new,
    support_fit_stop_roles,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import (
    _remap_training_devices,
    _training_reference_stats,
)
from fhm_dapp_tim.fhm_dapp.v18_3_single import (
    EVIDENCE_FAMILIES,
    EVIDENCE_SOURCES,
    V183_CONFIG,
    StructuredEvidenceAnchorModel,
    build_v183_model,
    validate_v183_config,
    v183_predict,
    v183_training_loss,
)


PROTOCOL_VERSION = "uji_v18_3_hierarchical_robust_consensus_v1_20261001"
OUTER_GEOMETRY_SEED = 18331
INNER_GEOMETRY_SEED = 18332
EARLY_STOP_SEED = 18333
MODEL_SEED = 18431
STABILITY_SEEDS = (18431, 18432, 18433, 18434, 18435)
SCORING_SEED = 18631
OUTER_REFIT_FRACTION = 0.80
SELECTION_REFIT_FRACTION = 0.80
EARLY_STOP_FRACTION = 0.15
PREDICTIVE_DRAWS = 64
CALIBRATION_TEMPERATURE_GRID = (
    1.00,
    1.25,
    1.50,
    2.00,
    2.50,
    3.00,
)
CALIBRATION_SCALE_GRID = (
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
)
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_TRAIN = REPO_ROOT / "data" / "UJIndoorLoc" / "trainingData.csv"
DEFAULT_BENCHMARK = REPO_ROOT / "data" / "UJIndoorLoc" / "validationData.csv"
DEFAULT_EVIDENCE = ROOT / "evidence" / "v18_3"
DEFAULT_MODELS = ROOT / "models" / "v18_3"
DEFAULT_SHARDS = DEFAULT_EVIDENCE / "role_shards"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "registry_v18_3.json"
DEFAULT_SELECTION_CHECKPOINT = DEFAULT_MODELS / "selection.pt"
DEFAULT_CALIBRATION_RESULT = DEFAULT_EVIDENCE / "calibration_result_v18_3.json"
DEFAULT_FREEZE = DEFAULT_EVIDENCE / "selection_freeze_v18_3.json"
DEFAULT_STABILITY_RESULT = DEFAULT_EVIDENCE / "stability_result_v18_3.json"
DEFAULT_CONFIRMATION_CHECKPOINT = DEFAULT_MODELS / "confirmation.pt"
DEFAULT_CONFIRMATION_LOCK = DEFAULT_EVIDENCE / "confirmation_consumed.lock"
DEFAULT_CONFIRMATION_RESULT = DEFAULT_EVIDENCE / "confirmation_result_v18_3.json"
DEFAULT_FINAL_CHECKPOINT = DEFAULT_MODELS / "final.pt"
DEFAULT_BENCHMARK_RESULT = DEFAULT_EVIDENCE / "benchmark_result_v18_3.json"

LocationKey = tuple[int, int, float, float]


def _set_seed(seed: int) -> None:
    """Set a fail-closed deterministic runtime."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _strata(keys: set[LocationKey]) -> dict[tuple[int, int], set[LocationKey]]:
    output: dict[tuple[int, int], set[LocationKey]] = {}
    for key in keys:
        output.setdefault(key[:2], set()).add(key)
    return output


def construct_coordinate_roles(
    all_keys: set[LocationKey],
) -> tuple[dict[str, set[LocationKey]], dict]:
    """Create a fresh nested V18.3 coordinate protocol."""

    if not all_keys:
        raise ValueError("all_keys must be nonempty")
    outer_refit, outer_details = geometry_support_with_hull(
        all_keys,
        OUTER_REFIT_FRACTION,
        seed=OUTER_GEOMETRY_SEED,
    )
    confirmation = all_keys - outer_refit
    selection_refit, selection_details = geometry_support_with_hull(
        outer_refit,
        SELECTION_REFIT_FRACTION,
        seed=INNER_GEOMETRY_SEED,
    )
    calibration = outer_refit - selection_refit
    nested, early_details = support_fit_stop_roles(
        selection_refit,
        seed=EARLY_STOP_SEED,
        early_stop_fraction=EARLY_STOP_FRACTION,
    )
    roles = {
        "member_fit": nested["member_fit"],
        "early_stop": nested["early_stop"],
        "selection_refit": selection_refit,
        "calibration": calibration,
        "outer_refit": outer_refit,
        "confirmation": confirmation,
    }
    return roles, {
        "outer": outer_details,
        "selection": selection_details,
        "early_stop": early_details,
    }


def audit_coordinate_roles(
    roles: Mapping[str, set[LocationKey]],
    all_keys: set[LocationKey],
) -> dict:
    """Fail on coordinate leakage, union errors, stratum loss, or hull loss."""

    checks = {
        "member_fit_early_stop": len(
            roles["member_fit"] & roles["early_stop"]
        ),
        "selection_refit_calibration": len(
            roles["selection_refit"] & roles["calibration"]
        ),
        "outer_refit_confirmation": len(
            roles["outer_refit"] & roles["confirmation"]
        ),
    }
    if any(checks.values()):
        raise RuntimeError(f"V18.3 coordinate leakage: {checks}")
    if roles["member_fit"] | roles["early_stop"] != roles["selection_refit"]:
        raise RuntimeError("fit/stop do not reconstruct selection_refit")
    if roles["selection_refit"] | roles["calibration"] != roles["outer_refit"]:
        raise RuntimeError("selection/calibration do not reconstruct outer_refit")
    if roles["outer_refit"] | roles["confirmation"] != all_keys:
        raise RuntimeError("outer_refit/confirmation do not reconstruct trainingData")

    expected_strata = set(_strata(all_keys))
    for name, keys in roles.items():
        if set(_strata(keys)) != expected_strata:
            raise RuntimeError(f"{name} loses a building-floor stratum")

    hull_checks: dict[str, int] = {}
    for source_name, reference_name in (
        ("all", "outer_refit"),
        ("outer_refit", "selection_refit"),
        ("selection_refit", "member_fit"),
    ):
        source = all_keys if source_name == "all" else roles[source_name]
        reference = roles[reference_name]
        missing = 0
        for keys in _strata(source).values():
            ordered = sorted(keys)
            xy = np.asarray(
                [[key[2], key[3]] for key in ordered],
                dtype=np.float64,
            )
            hull = {ordered[i] for i in _hull_local_indices(xy)}
            missing += len(hull - reference)
        hull_checks[
            f"{source_name}_hull_missing_from_{reference_name}"
        ] = missing
        if missing:
            raise RuntimeError(
                f"{reference_name} misses {missing} mandatory hull coordinates"
            )

    return {
        "coordinate_grouping": "complete (building,floor,longitude,latitude)",
        "disjoint_intersections": checks,
        "exact_union_identities": True,
        "all_roles_keep_all_building_floor_strata": True,
        "hull_checks": hull_checks,
    }


def _shard_payload(
    data: dict,
    rows: np.ndarray,
    *,
    role: str,
    source_sha256: str,
) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "role": role,
        "source_sha256": source_sha256,
        "row_indices": torch.from_numpy(np.asarray(rows, dtype=np.int64)),
        "row_index_sha256": _sha256_indices(rows),
        "rss": data["rss"][rows].float().cpu(),
        "raw_xy": torch.from_numpy(
            np.asarray(data["raw_xy"][rows], dtype=np.float32)
        ),
        "building_t": data["building_t"][rows].long().cpu(),
        "floor_t": data["floor_t"][rows].long().cpu(),
        "phone_t": data["phone_t"][rows].long().cpu(),
        "timestamp_t": data["timestamp_t"][rows].cpu(),
        "stats": data["stats"],
        "keys": [_encode_key(data["keys"][int(row)]) for row in rows],
    }


def prepare_protocol(
    *,
    train_data: str | Path = DEFAULT_TRAIN,
    registry_path: str | Path = DEFAULT_REGISTRY,
    shard_dir: str | Path = DEFAULT_SHARDS,
) -> dict:
    """Register all internal roles without reading the exposed benchmark."""

    validate_v183_config(V183_CONFIG)
    destination = Path(registry_path)
    if destination.exists():
        raise FileExistsError(destination)

    train_path = Path(train_data).resolve()
    source_sha = _sha256_file(train_path)
    data = _load_uji(train_path)
    all_keys = set(data["keys"])
    roles, split_details = construct_coordinate_roles(all_keys)
    diagnostics = audit_coordinate_roles(roles, all_keys)

    shard_records: dict[str, dict] = {}
    shard_root = Path(shard_dir)
    for role, keys in roles.items():
        rows = _rows_for_keys(data["keys"], keys)
        shard_path = shard_root / f"{role}.pt"
        _save_torch_new(
            shard_path,
            _shard_payload(
                data,
                rows,
                role=role,
                source_sha256=source_sha,
            ),
        )
        shard_records[role] = {
            "path": str(shard_path.resolve()),
            "sha256": _sha256_file(shard_path),
            "row_count": int(len(rows)),
            "row_index_sha256": _sha256_indices(rows),
            "coordinate_count": len(keys),
            "coordinate_key_sha256": _sha256_keys(keys),
        }

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_status": (
            "newly registered sequential V18.3 internal coordinate confirmation; "
            "prior V18.2 outcomes informed model development; official "
            "validationData is pre-exposed and descriptive only"
        ),
        "train_data": str(train_path),
        "train_data_sha256": source_sha,
        "training_rows": int(len(data["keys"])),
        "training_coordinate_count": len(all_keys),
        "training_coordinate_sha256": _sha256_keys(all_keys),
        "n_aps": int(data["stats"]["n_aps"]),
        "phone_ids": sorted(map(int, np.unique(data["phone"]))),
        "roles": {
            name: _role_record(keys, data["keys"])
            for name, keys in roles.items()
        },
        "role_shards": shard_records,
        "split_details": split_details,
        "split_diagnostics": diagnostics,
        "constants": {
            "outer_geometry_seed": OUTER_GEOMETRY_SEED,
            "inner_geometry_seed": INNER_GEOMETRY_SEED,
            "early_stop_seed": EARLY_STOP_SEED,
            "model_seed": MODEL_SEED,
            "stability_seeds": list(STABILITY_SEEDS),
            "scoring_seed": SCORING_SEED,
            "outer_refit_fraction": OUTER_REFIT_FRACTION,
            "selection_refit_fraction": SELECTION_REFIT_FRACTION,
            "early_stop_fraction": EARLY_STOP_FRACTION,
            "predictive_draws": PREDICTIVE_DRAWS,
            "calibration_temperature_grid": list(
                CALIBRATION_TEMPERATURE_GRID
            ),
            "calibration_scale_grid": list(CALIBRATION_SCALE_GRID),
        },
        "model_config": dict(V183_CONFIG),
        "single_member_contract": {
            "checkpoint_count_at_inference": 1,
            "residual_experts": 1,
            "teacher_model": False,
            "knowledge_distillation": False,
            "member_axis": False,
            "ensemble": False,
        },
        "evaluation_contract": {
            "confirmation_is_new_registered_trainingData_holdout": True,
            "confirmation_is_sequential_not_independent_external": True,
            "confirmation_is_one_shot": True,
            "official_validation_is_pre_exposed": True,
            "official_validation_is_descriptive_only": True,
        },
    }
    registry["registry_sha256"] = _canonical_json_sha256(registry)
    _write_json_new(destination, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18.3 registry")
    if payload.get("registry_sha256") != _canonical_json_sha256(
        payload,
        omit="registry_sha256",
    ):
        raise ValueError("V18.3 registry self-hash failed")
    if payload.get("model_config") != V183_CONFIG:
        raise ValueError("V18.3 model configuration drifted")
    validate_v183_config(payload["model_config"])
    return payload


def load_role_shard(registry: dict, role: str) -> dict:
    if role not in registry["role_shards"]:
        raise ValueError(f"unknown V18.3 role: {role}")
    record = registry["role_shards"][role]
    path = Path(record["path"])
    if _sha256_file(path) != record["sha256"]:
        raise ValueError(f"V18.3 shard hash failed for {role}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"not a V18.3 shard: {role}")
    if payload.get("role") != role:
        raise ValueError(f"V18.3 shard role mismatch: {role}")
    if payload.get("source_sha256") != registry["train_data_sha256"]:
        raise ValueError(f"V18.3 shard source mismatch: {role}")
    rows = payload["row_indices"].cpu().numpy().astype(np.int64)
    if _sha256_indices(rows) != record["row_index_sha256"]:
        raise ValueError(f"V18.3 shard row hash failed: {role}")
    return payload


def combine_role_shards(
    first: dict,
    second: dict,
    *,
    role: str,
) -> dict:
    if first["source_sha256"] != second["source_sha256"]:
        raise ValueError("cannot combine V18.3 shards from different sources")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "role": role,
        "source_sha256": first["source_sha256"],
        "rss": torch.cat([first["rss"], second["rss"]], dim=0),
        "raw_xy": torch.cat([first["raw_xy"], second["raw_xy"]], dim=0),
        "building_t": torch.cat(
            [first["building_t"], second["building_t"]], dim=0
        ),
        "floor_t": torch.cat([first["floor_t"], second["floor_t"]], dim=0),
        "phone_t": torch.cat([first["phone_t"], second["phone_t"]], dim=0),
        "timestamp_t": torch.cat(
            [first["timestamp_t"], second["timestamp_t"]], dim=0
        ),
        "stats": first["stats"],
        "keys": list(first["keys"]) + list(second["keys"]),
    }


def _shard_arrays(shard: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        shard["raw_xy"].cpu().numpy().astype(np.float32),
        shard["building_t"].cpu().numpy().astype(np.int64),
        shard["floor_t"].cpu().numpy().astype(np.int64),
    )


def _decode_shard_keys(shard: dict) -> list[LocationKey]:
    return [_decode_key(value) for value in shard["keys"]]


def _build_training_state(
    shard: dict,
    *,
    device: torch.device,
) -> tuple[
    StructuredEvidenceAnchorModel,
    dict,
    np.ndarray,
    torch.Tensor,
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    dict[int, int],
    dict,
]:
    raw_xy, building, floor = _shard_arrays(shard)
    rows = np.arange(len(raw_xy), dtype=np.int64)
    origin = raw_xy.mean(axis=0).astype(np.float32)
    centered_np = (raw_xy - origin[None, :]).astype(np.float32)
    rss = shard["rss"].float().cpu()

    memory, sample_anchor = build_anchor_memory(
        rss,
        centered_np,
        building,
        floor,
        rows,
    )
    if np.any(sample_anchor < 0):
        raise RuntimeError("V18.3 training row escaped the anchor dictionary")

    phone = shard["phone_t"].cpu().numpy().astype(np.int64)
    _mapped, device_mapping = _remap_training_devices(phone, rows)
    model, geometry, neighbours, neighbour_weights = build_v183_model(
        memory=memory,
        n_aps=int(shard["stats"]["n_aps"]),
        n_buildings=int(building.max()) + 1,
        n_devices=len(device_mapping),
        device=device,
        config=V183_CONFIG,
    )
    stats = _training_reference_stats(
        shard["stats"],
        raw_xy,
        building,
        rows,
    )
    return (
        model,
        memory,
        sample_anchor,
        torch.from_numpy(centered_np).float(),
        origin,
        neighbours,
        neighbour_weights,
        device_mapping,
        stats,
    )


def _loader(
    shard: dict,
    centered: torch.Tensor,
    sample_anchor: np.ndarray,
    device_mapping: Mapping[int, int],
    *,
    seed: int,
) -> DataLoader:
    phone = shard["phone_t"].cpu().numpy().astype(np.int64)
    mapped = torch.tensor(
        [device_mapping[int(value)] for value in phone],
        dtype=torch.long,
    )
    dataset = TensorDataset(
        shard["rss"].float(),
        centered,
        torch.from_numpy(sample_anchor).long(),
        mapped,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(V183_CONFIG["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _coordinate_macro(
    values: torch.Tensor,
    keys: Sequence[LocationKey],
) -> tuple[float, float, int]:
    groups: dict[LocationKey, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    grouped = torch.stack(
        [
            values[torch.as_tensor(indices, dtype=torch.long)].mean()
            for indices in groups.values()
        ]
    ).double()
    se = (
        float(grouped.std(unbiased=True) / math.sqrt(len(grouped)))
        if len(grouped) > 1
        else 0.0
    )
    return float(grouped.mean()), se, len(grouped)


def _nearest_anchor_recall(
    model: StructuredEvidenceAnchorModel,
    target: torch.Tensor,
    target_building: torch.Tensor,
    target_floor: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(
        model.anchor_coords.unsqueeze(0) - target.unsqueeze(1),
        dim=2,
    )
    same = (
        model.anchor_buildings.unsqueeze(0) == target_building.unsqueeze(1)
    ) & (
        model.anchor_floors.unsqueeze(0) == target_floor.unsqueeze(1)
    )
    nearest = distance.masked_fill(~same, float("inf")).argmin(dim=1)
    return (
        candidate_indices == nearest.unsqueeze(1)
    ).any(dim=1).float()


def evaluate_shard(
    model: StructuredEvidenceAnchorModel,
    shard: dict,
    origin: np.ndarray,
    *,
    posterior_temperature: float,
    scale_multiplier: float,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    model.eval()
    raw_xy, _building, _floor = _shard_arrays(shard)
    centered = torch.from_numpy(
        (raw_xy - origin[None, :]).astype(np.float32)
    )
    keys = _decode_shard_keys(shard)

    values: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "nll",
            "error",
            "energy",
            "building_error",
            "floor_error",
            "entropy",
            "effective",
            "top1",
            "radial",
            "within_sd",
            "between_sd",
            "radio_energy",
            "style_offset_abs",
            "style_scale",
            "global_disagreement",
            "disagreement_temperature",
            "radio_mismatch",
        )
    }
    final_recall: dict[int, list[torch.Tensor]] = {
        k: [] for k in (8, 16, 32, 64)
    }
    source_recall: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "invariant",
            "prototype",
            "floor",
            "union",
        )
    }
    family_weight_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in EVIDENCE_FAMILIES
    }
    family_entropy_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in EVIDENCE_FAMILIES
    }
    family_disagreement_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in EVIDENCE_FAMILIES
    }
    source_entropy_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in EVIDENCE_SOURCES
    }
    within_family_parts: dict[str, list[torch.Tensor]] = {
        "representation_invariant": [],
        "representation_learned": [],
        "radio_prototype": [],
        "radio_marked": [],
        "structure_floor": [],
        "structure_geometry": [],
    }
    consensus_parts: dict[str, list[torch.Tensor]] = {
        "global_disagreement": [],
        "disagreement_temperature": [],
        "radio_mismatch": [],
    }
    coverage_parts: dict[float, list[torch.Tensor]] = {
        level: [] for level in COVERAGE_LEVELS
    }
    radius_parts: dict[float, list[torch.Tensor]] = {
        level: [] for level in COVERAGE_LEVELS
    }

    for start in range(0, len(raw_xy), int(batch_size)):
        stop = min(start + int(batch_size), len(raw_xy))
        rss = shard["rss"][start:stop].to(device)
        target = centered[start:stop].to(device)
        target_building = shard["building_t"][start:stop].to(device)
        target_floor = shard["floor_t"][start:stop].to(device)

        result = v183_predict(
            model,
            rss,
            posterior_temperature=float(posterior_temperature),
            scale_multiplier=float(scale_multiplier),
            target_coords=target,
            config=V183_CONFIG,
        )
        for name in (
            "nll",
            "coords",
            "candidate_weights",
            "component_locations",
            "component_scales",
            "family_weights",
            "global_disagreement",
            "disagreement_temperature",
            "radio_mismatch",
        ):
            if not bool(torch.isfinite(result[name]).all()):
                raise RuntimeError(f"non-finite V18.3 prediction tensor: {name}")

        error = torch.linalg.vector_norm(
            result["coords"] - target,
            dim=1,
        )
        values["nll"].append(result["nll"].cpu())
        values["error"].append(error.cpu())
        values["building_error"].append(
            (result["building"] != target_building).float().cpu()
        )
        values["floor_error"].append(
            (result["floor"] != target_floor).float().cpu()
        )
        values["entropy"].append(result["anchor_entropy"].cpu())
        values["effective"].append(result["effective_anchor_count"].cpu())
        values["top1"].append(result["top1_probability"].cpu())
        values["radial"].append(result["radial_scale"].cpu())
        values["within_sd"].append(
            result["within_second_moment"].sqrt().cpu()
        )
        values["between_sd"].append(
            result["between_second_moment"].sqrt().cpu()
        )
        values["radio_energy"].append(
            (
                result["union_probability"]
                * result["radio_energy_union"]
            ).sum(dim=1).cpu()
        )
        values["style_offset_abs"].append(
            result["style_offset"].abs().cpu()
        )
        values["style_scale"].append(result["style_scale"].cpu())
        values["global_disagreement"].append(
            result["global_disagreement"].cpu()
        )
        values["disagreement_temperature"].append(
            result["disagreement_temperature"].cpu()
        )
        values["radio_mismatch"].append(result["radio_mismatch"].cpu())

        randomness = common_student_t_randomness(
            len(target),
            draws=PREDICTIVE_DRAWS,
            degrees_freedom=float(V183_CONFIG["student_df"]),
            seed=SCORING_SEED + start,
        )
        energy, first, second = continuous_energy_score_mc(
            result["candidate_weights"],
            result["component_locations"],
            result["component_scales"],
            target,
            randomness,
        )
        values["energy"].append(energy.cpu())
        draws = torch.cat([first, second], dim=1)
        draw_radius = torch.linalg.vector_norm(
            draws - result["coords"].unsqueeze(1),
            dim=2,
        )

        for cutoff in final_recall:
            width = min(cutoff, result["candidate_indices"].shape[1])
            final_recall[cutoff].append(
                _nearest_anchor_recall(
                    model,
                    target,
                    target_building,
                    target_floor,
                    result["candidate_indices"][:, :width],
                ).cpu()
            )
        source_recall["invariant"].append(
            _nearest_anchor_recall(
                model,
                target,
                target_building,
                target_floor,
                result["retrieval_invariant_indices"],
            ).cpu()
        )
        source_recall["prototype"].append(
            _nearest_anchor_recall(
                model,
                target,
                target_building,
                target_floor,
                result["retrieval_prototype_indices"],
            ).cpu()
        )
        source_recall["floor"].append(
            _nearest_anchor_recall(
                model,
                target,
                target_building,
                target_floor,
                result["retrieval_floor_indices"],
            ).cpu()
        )
        source_recall["union"].append(
            _nearest_anchor_recall(
                model,
                target,
                target_building,
                target_floor,
                result["union_candidate_indices"],
            ).cpu()
        )

        for source_index, source_name in enumerate(EVIDENCE_SOURCES):
            source_entropy_parts[source_name].append(
                result["source_entropy_union"][:, source_index].cpu()
            )
        for family_index, family_name in enumerate(EVIDENCE_FAMILIES):
            family_weight_parts[family_name].append(
                result["family_weights"][:, family_index].cpu()
            )
            family_entropy_parts[family_name].append(
                result["family_entropy"][:, family_index].cpu()
            )
            family_disagreement_parts[family_name].append(
                result["family_disagreement"][:, family_index].cpu()
            )

        within = result["within_family_weights"].cpu()
        within_family_parts["representation_invariant"].append(
            within[:, 0, 0]
        )
        within_family_parts["representation_learned"].append(
            within[:, 0, 1]
        )
        within_family_parts["radio_prototype"].append(within[:, 1, 0])
        within_family_parts["radio_marked"].append(within[:, 1, 1])
        within_family_parts["structure_floor"].append(within[:, 2, 0])
        within_family_parts["structure_geometry"].append(within[:, 2, 1])

        for level in COVERAGE_LEVELS:
            radius = torch.quantile(draw_radius, float(level), dim=1)
            coverage_parts[level].append(
                (error <= radius).float().cpu()
            )
            radius_parts[level].append(radius.cpu())

    joined = {
        name: torch.cat(parts)
        for name, parts in values.items()
    }
    macro_nll, nll_se, groups = _coordinate_macro(joined["nll"], keys)
    macro_mae, mae_se, _ = _coordinate_macro(joined["error"], keys)
    macro_energy, energy_se, _ = _coordinate_macro(joined["energy"], keys)
    macro_building, _, _ = _coordinate_macro(joined["building_error"], keys)
    macro_floor, _, _ = _coordinate_macro(joined["floor_error"], keys)

    error_np = joined["error"].double().numpy()
    radial_np = joined["radial"].double().numpy()
    corr = (
        float(np.corrcoef(error_np, radial_np)[0, 1])
        if np.std(error_np) > 0.0 and np.std(radial_np) > 0.0
        else 0.0
    )

    metrics = {
        "coordinate_macro_mixture_nll": macro_nll,
        "coordinate_macro_mixture_nll_se": nll_se,
        "coordinate_macro_mae_m": macro_mae,
        "coordinate_macro_mae_se_m": mae_se,
        "coordinate_macro_energy_score_m": macro_energy,
        "coordinate_macro_energy_score_se_m": energy_se,
        "coordinate_macro_building_error": macro_building,
        "coordinate_macro_floor_error": macro_floor,
        "mean_anchor_entropy_nats": float(joined["entropy"].mean()),
        "mean_effective_anchor_count": float(joined["effective"].mean()),
        "mean_top1_probability": float(joined["top1"].mean()),
        "mean_radial_scale_m": float(joined["radial"].mean()),
        "mean_within_component_radial_sd_m": float(joined["within_sd"].mean()),
        "mean_between_anchor_radial_sd_m": float(joined["between_sd"].mean()),
        "mean_posterior_radio_energy": float(joined["radio_energy"].mean()),
        "mean_abs_profiled_style_offset": float(
            joined["style_offset_abs"].mean()
        ),
        "mean_profiled_style_scale": float(joined["style_scale"].mean()),
        "mean_global_evidence_disagreement": float(
            joined["global_disagreement"].mean()
        ),
        "mean_disagreement_temperature": float(
            joined["disagreement_temperature"].mean()
        ),
        "mean_radio_mismatch": float(joined["radio_mismatch"].mean()),
        "error_radial_scale_correlation": corr,
        "coordinate_groups": groups,
        "rows": int(len(raw_xy)),
    }
    for cutoff, parts in final_recall.items():
        metrics[f"final_nearest_anchor_recall_at_{cutoff}"] = float(
            torch.cat(parts).mean()
        )
    for name, parts in source_recall.items():
        metrics[f"{name}_nearest_anchor_recall"] = float(
            torch.cat(parts).mean()
        )
    for name in EVIDENCE_SOURCES:
        metrics[f"mean_source_entropy_{name}"] = float(
            torch.cat(source_entropy_parts[name]).mean()
        )
    for name in EVIDENCE_FAMILIES:
        metrics[f"mean_family_weight_{name}"] = float(
            torch.cat(family_weight_parts[name]).mean()
        )
        metrics[f"mean_family_entropy_{name}"] = float(
            torch.cat(family_entropy_parts[name]).mean()
        )
        metrics[f"mean_family_disagreement_{name}"] = float(
            torch.cat(family_disagreement_parts[name]).mean()
        )
    for name, parts in within_family_parts.items():
        metrics[f"mean_within_family_weight_{name}"] = float(
            torch.cat(parts).mean()
        )

    disagreement_np = joined["global_disagreement"].double().numpy()
    radio_mismatch_np = joined["radio_mismatch"].double().numpy()
    radio_weight_np = torch.cat(
        family_weight_parts["radio"]
    ).double().numpy()
    metrics["error_evidence_disagreement_correlation"] = (
        float(np.corrcoef(error_np, disagreement_np)[0, 1])
        if np.std(error_np) > 0.0 and np.std(disagreement_np) > 0.0
        else 0.0
    )
    metrics["radio_mismatch_radio_weight_correlation"] = (
        float(np.corrcoef(radio_mismatch_np, radio_weight_np)[0, 1])
        if np.std(radio_mismatch_np) > 0.0 and np.std(radio_weight_np) > 0.0
        else 0.0
    )
    for level in COVERAGE_LEVELS:
        label = int(round(100 * level))
        metrics[f"predictive_radial_coverage_{label}"] = float(
            torch.cat(coverage_parts[level]).mean()
        )
        metrics[f"mean_predictive_radius_{label}_m"] = float(
            torch.cat(radius_parts[level]).mean()
        )
    return metrics


def _train_epoch(
    model: StructuredEvidenceAnchorModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    neighbours: torch.Tensor,
    neighbour_weights: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for rss, target, anchor, device_id in loader:
        rss = rss.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        anchor = anchor.to(device, non_blocking=True)
        device_id = device_id.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        terms = v183_training_loss(
            model,
            rss,
            target,
            anchor,
            device_id,
            neighbours,
            neighbour_weights,
            config=V183_CONFIG,
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(V183_CONFIG["gradient_clip"]),
        )
        optimizer.step()
        batch = len(rss)
        count += batch
        for name, value in terms.items():
            if value.ndim == 0:
                if not bool(torch.isfinite(value)):
                    raise RuntimeError(f"non-finite V18.3 training term: {name}")
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach()
                ) * batch
    return {name: value / count for name, value in totals.items()}


def select_epoch_from_history(history: Sequence[dict]) -> int:
    eligible = [
        item
        for item in history
        if int(item["epoch"]) >= int(V183_CONFIG["minimum_epochs"])
    ]
    if not eligible:
        raise ValueError("history contains no eligible V18.3 epoch")
    best_energy = min(
        float(item["early_stop"]["coordinate_macro_energy_score_m"])
        for item in eligible
    )
    shortlist = [
        item
        for item in eligible
        if float(item["early_stop"]["coordinate_macro_energy_score_m"])
        <= best_energy + float(V183_CONFIG["energy_tie_margin_m"])
    ]
    selected = min(
        shortlist,
        key=lambda item: (
            float(item["early_stop"]["coordinate_macro_mae_m"]),
            float(item["early_stop"]["coordinate_macro_energy_score_m"]),
            int(item["epoch"]),
        ),
    )
    return int(selected["epoch"])


def select_epoch(
    fit_shard: dict,
    stop_shard: dict,
    *,
    device: torch.device,
    log_every: int = 5,
) -> tuple[int, list[dict]]:
    _set_seed(MODEL_SEED)
    (
        model,
        _memory,
        sample_anchor,
        centered,
        origin,
        neighbours,
        neighbour_weights,
        mapping,
        _stats,
    ) = _build_training_state(fit_shard, device=device)
    loader = _loader(
        fit_shard,
        centered,
        sample_anchor,
        mapping,
        seed=MODEL_SEED,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(V183_CONFIG["learning_rate"]),
        weight_decay=float(V183_CONFIG["weight_decay"]),
    )

    history: list[dict] = []
    best_running = float("inf")
    stale = 0
    for epoch in range(1, int(V183_CONFIG["maximum_epochs"]) + 1):
        train_metrics = _train_epoch(
            model,
            loader,
            optimizer,
            neighbours,
            neighbour_weights,
            device=device,
        )
        stop_metrics = evaluate_shard(
            model,
            stop_shard,
            origin,
            posterior_temperature=1.0,
            scale_multiplier=1.0,
            device=device,
        )
        eligible = epoch >= int(V183_CONFIG["minimum_epochs"])
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "early_stop": stop_metrics,
            "eligible_for_selection": eligible,
        }
        history.append(record)

        if eligible:
            energy = float(stop_metrics["coordinate_macro_energy_score_m"])
            if energy < best_running - float(
                V183_CONFIG["minimum_improvement_energy_m"]
            ):
                best_running = energy
                stale = 0
            else:
                stale += 1

        if epoch == 1 or epoch % int(log_every) == 0:
            print(json.dumps(record, allow_nan=False), flush=True)
        if eligible and stale >= int(V183_CONFIG["patience"]):
            break
    return select_epoch_from_history(history), history


def fit_fixed(
    shard: dict,
    *,
    epochs: int,
    device: torch.device,
    log_every: int = 5,
    seed: int = MODEL_SEED,
) -> tuple[
    StructuredEvidenceAnchorModel,
    dict,
    np.ndarray,
    dict,
    list[dict],
]:
    if int(epochs) < int(V183_CONFIG["minimum_epochs"]):
        raise ValueError("V18.3 refit epoch violates minimum_epochs")
    _set_seed(int(seed))
    (
        model,
        memory,
        sample_anchor,
        centered,
        origin,
        neighbours,
        neighbour_weights,
        mapping,
        stats,
    ) = _build_training_state(shard, device=device)
    loader = _loader(
        shard,
        centered,
        sample_anchor,
        mapping,
        seed=int(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(V183_CONFIG["learning_rate"]),
        weight_decay=float(V183_CONFIG["weight_decay"]),
    )
    history: list[dict] = []
    for epoch in range(1, int(epochs) + 1):
        metrics = _train_epoch(
            model,
            loader,
            optimizer,
            neighbours,
            neighbour_weights,
            device=device,
        )
        history.append({"epoch": epoch, **metrics})
        if epoch == 1 or epoch == int(epochs) or epoch % int(log_every) == 0:
            print(
                json.dumps(
                    {
                        "refit_epoch": epoch,
                        "total_epochs": int(epochs),
                        **metrics,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
    memory = dict(memory)
    memory["stats"] = stats
    return model, memory, origin, mapping, history


def _checkpoint_payload(
    *,
    registry: dict,
    phase: str,
    refit_role: str,
    selected_epoch: int,
    model: StructuredEvidenceAnchorModel,
    memory: dict,
    origin: np.ndarray,
    device_mapping: Mapping[int, int],
    selection_history: list[dict] | None,
    refit_history: list[dict],
) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "phase": phase,
        "refit_role": refit_role,
        "selected_epoch": int(selected_epoch),
        "model_seed": MODEL_SEED,
        "model_state": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "coord_origin": torch.as_tensor(origin, dtype=torch.float32),
        "stats": memory["stats"],
        "config": dict(V183_CONFIG),
        "device_id_mapping": {
            str(raw): int(mapped)
            for raw, mapped in device_mapping.items()
        },
        "selection_history": selection_history,
        "refit_history": refit_history,
        "single_member_contract": {
            "checkpoint_count": 1,
            "residual_experts": 1,
            "teacher_model": False,
            "ensemble": False,
        },
    }


def train_selection(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_SELECTION_CHECKPOINT,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)

    fit_shard = load_role_shard(registry, "member_fit")
    stop_shard = load_role_shard(registry, "early_stop")
    refit_shard = load_role_shard(registry, "selection_refit")
    device = torch.device(device_name)
    started = time.perf_counter()
    epoch, selection_history = select_epoch(
        fit_shard,
        stop_shard,
        device=device,
        log_every=log_every,
    )
    model, memory, origin, mapping, refit_history = fit_fixed(
        refit_shard,
        epochs=epoch,
        device=device,
        log_every=log_every,
    )
    payload = _checkpoint_payload(
        registry=registry,
        phase="selection",
        refit_role="selection_refit",
        selected_epoch=epoch,
        model=model,
        memory=memory,
        origin=origin,
        device_mapping=mapping,
        selection_history=selection_history,
        refit_history=refit_history,
    )
    payload["training_seconds"] = float(time.perf_counter() - started)
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "selected_epoch": epoch,
        "anchors": int(model.n_anchors),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": payload["training_seconds"],
    }


def load_v183_checkpoint(
    path: str | Path,
    *,
    registry: dict,
    device: torch.device,
    expected_phase: str,
) -> tuple[StructuredEvidenceAnchorModel, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18.3 checkpoint")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18.3 checkpoint registry mismatch")
    if payload.get("phase") != expected_phase:
        raise ValueError("V18.3 checkpoint phase mismatch")
    if payload.get("config") != V183_CONFIG:
        raise ValueError("V18.3 checkpoint config mismatch")

    state = payload["model_state"]
    memory = {
        "prototype_rss": state["prototype_rss"],
        "detection_prob": state["anchor_detection_prob"],
        "rss_center": state["anchor_rss_center"],
        "rss_scale": state["anchor_rss_scale"],
        "coords": state["anchor_coords"],
        "building": state["anchor_buildings"],
        "floor": state["anchor_floors"],
        "counts": state["anchor_sample_count"].long(),
    }
    model, _geometry, _neighbours, _weights = build_v183_model(
        memory=memory,
        n_aps=int(payload["stats"]["n_aps"]),
        n_buildings=int(state["anchor_buildings"].max().item()) + 1,
        n_devices=len(payload.get("device_id_mapping", {})),
        device=device,
        config=V183_CONFIG,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def collect_calibration_cache(
    model: StructuredEvidenceAnchorModel,
    shard: dict,
    origin: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    raw_xy, _building, _floor = _shard_arrays(shard)
    centered = torch.from_numpy(
        (raw_xy - origin[None, :]).astype(np.float32)
    )
    output = {
        "log_probability": [],
        "location": [],
        "scale": [],
        "target": [],
    }
    for start in range(0, len(raw_xy), int(batch_size)):
        stop = min(start + int(batch_size), len(raw_xy))
        result = v183_predict(
            model,
            shard["rss"][start:stop].to(device),
            posterior_temperature=1.0,
            scale_multiplier=1.0,
            target_coords=centered[start:stop].to(device),
            config=V183_CONFIG,
        )
        output["log_probability"].append(
            result["candidate_weights"].clamp(min=1e-12).log().cpu()
        )
        output["location"].append(result["component_locations"].cpu())
        output["scale"].append(
            result["component_scales_uncalibrated"].cpu()
        )
        output["target"].append(centered[start:stop].cpu())
    return {
        name: torch.cat(parts)
        for name, parts in output.items()
    }


def _student_log_prob(
    target: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    df = float(V183_CONFIG["student_df"])
    z = (target.unsqueeze(1) - location) / scale
    normalizer = (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
    )
    return (
        normalizer
        - torch.log(scale)
        - 0.5 * (df + 1.0) * torch.log1p(z.square() / df)
    ).sum(dim=2)


def calibration_macro_nll(
    cache: dict,
    keys: Sequence[LocationKey],
    *,
    posterior_temperature: float,
    scale_multiplier: float,
) -> float:
    logp = cache["log_probability"] / float(posterior_temperature)
    weights = torch.softmax(logp, dim=1)
    component_lp = _student_log_prob(
        cache["target"],
        cache["location"],
        cache["scale"] * float(scale_multiplier),
    )
    nll = -torch.logsumexp(
        weights.clamp(min=1e-12).log() + component_lp,
        dim=1,
    )
    mean, _se, _groups = _coordinate_macro(nll, keys)
    return mean


def fit_calibration_grid(
    cache: dict,
    keys: Sequence[LocationKey],
) -> dict:
    surface = []
    for temperature in CALIBRATION_TEMPERATURE_GRID:
        for scale in CALIBRATION_SCALE_GRID:
            surface.append(
                {
                    "posterior_temperature": float(temperature),
                    "scale_multiplier": float(scale),
                    "coordinate_macro_mixture_nll": calibration_macro_nll(
                        cache,
                        keys,
                        posterior_temperature=temperature,
                        scale_multiplier=scale,
                    ),
                }
            )
    selected = min(
        surface,
        key=lambda item: (
            item["coordinate_macro_mixture_nll"],
            abs(math.log(item["posterior_temperature"]))
            + abs(math.log(item["scale_multiplier"])),
        ),
    )
    t = float(selected["posterior_temperature"])
    s = float(selected["scale_multiplier"])
    return {
        "posterior_temperature": t,
        "scale_multiplier": s,
        "best_coordinate_macro_mixture_nll": float(
            selected["coordinate_macro_mixture_nll"]
        ),
        "temperature_on_grid_boundary": t
        in {
            min(CALIBRATION_TEMPERATURE_GRID),
            max(CALIBRATION_TEMPERATURE_GRID),
        },
        "scale_on_grid_boundary": s
        in {
            min(CALIBRATION_SCALE_GRID),
            max(CALIBRATION_SCALE_GRID),
        },
        "surface": surface,
    }


def calibrate_and_freeze(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    checkpoint_path: str | Path = DEFAULT_SELECTION_CHECKPOINT,
    result_path: str | Path = DEFAULT_CALIBRATION_RESULT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    if Path(result_path).exists() or Path(freeze_path).exists():
        raise FileExistsError("V18.3 calibration/freeze already exists")

    device = torch.device(device_name)
    model, checkpoint = load_v183_checkpoint(
        checkpoint_path,
        registry=registry,
        device=device,
        expected_phase="selection",
    )
    calibration_shard = load_role_shard(registry, "calibration")
    origin = checkpoint["coord_origin"].cpu().numpy().astype(np.float32)
    cache = collect_calibration_cache(
        model,
        calibration_shard,
        origin,
        device=device,
    )
    calibration = fit_calibration_grid(
        cache,
        _decode_shard_keys(calibration_shard),
    )
    metrics = evaluate_shard(
        model,
        calibration_shard,
        origin,
        posterior_temperature=calibration["posterior_temperature"],
        scale_multiplier=calibration["scale_multiplier"],
        device=device,
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "selection_checkpoint_sha256": _sha256_file(checkpoint_path),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "calibration": calibration,
        "calibration_metrics": metrics,
        "confirmation_accessed": False,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(result_path, result)

    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "selection_checkpoint_sha256": result["selection_checkpoint_sha256"],
        "calibration_result_sha256": result["result_sha256"],
        "frozen_selected_epoch": int(checkpoint["selected_epoch"]),
        "frozen_model_config": dict(V183_CONFIG),
        "frozen_posterior_temperature": float(
            calibration["posterior_temperature"]
        ),
        "frozen_scale_multiplier": float(
            calibration["scale_multiplier"]
        ),
        "confirmation_access_authorized": True,
    }
    freeze["freeze_sha256"] = _canonical_json_sha256(freeze)
    _write_json_new(freeze_path, freeze)
    return {"calibration": result, "freeze": freeze}


def load_freeze(
    path: str | Path = DEFAULT_FREEZE,
    *,
    registry: dict,
) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18.3 freeze")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18.3 freeze registry mismatch")
    if payload.get("freeze_sha256") != _canonical_json_sha256(
        payload,
        omit="freeze_sha256",
    ):
        raise ValueError("V18.3 freeze self-hash failed")
    if payload.get("frozen_model_config") != V183_CONFIG:
        raise ValueError("V18.3 frozen config drifted")
    return payload


def _stability_summary(records: Sequence[dict]) -> dict:
    names = (
        "coordinate_macro_mae_m",
        "coordinate_macro_energy_score_m",
        "coordinate_macro_mixture_nll",
        "final_nearest_anchor_recall_at_8",
        "final_nearest_anchor_recall_at_64",
    )
    output = {}
    for name in names:
        values = np.asarray(
            [float(item["metrics"][name]) for item in records],
            dtype=np.float64,
        )
        output[name] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return output


def run_stability_panel(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    output: str | Path = DEFAULT_STABILITY_RESULT,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    freeze = load_freeze(freeze_path, registry=registry)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)

    fit = load_role_shard(registry, "member_fit")
    stop = load_role_shard(registry, "early_stop")
    device = torch.device(device_name)
    records = []
    for seed in STABILITY_SEEDS:
        model, _memory, origin, _mapping, history = fit_fixed(
            fit,
            epochs=int(freeze["frozen_selected_epoch"]),
            device=device,
            log_every=int(freeze["frozen_selected_epoch"]) + 1,
            seed=int(seed),
        )
        metrics = evaluate_shard(
            model,
            stop,
            origin,
            posterior_temperature=float(
                freeze["frozen_posterior_temperature"]
            ),
            scale_multiplier=float(freeze["frozen_scale_multiplier"]),
            device=device,
        )
        records.append(
            {
                "seed": int(seed),
                "metrics": metrics,
                "last_train_metrics": history[-1],
            }
        )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "seeds": list(map(int, STABILITY_SEEDS)),
        "ensemble_constructed": False,
        "best_seed_selected": False,
        "records": records,
        "summary": _stability_summary(records),
        "used_for_model_selection": False,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(destination, result)
    return result


def train_confirmation_model(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    output: str | Path = DEFAULT_CONFIRMATION_CHECKPOINT,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    freeze = load_freeze(freeze_path, registry=registry)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    outer_refit = load_role_shard(registry, "outer_refit")
    device = torch.device(device_name)
    started = time.perf_counter()
    model, memory, origin, mapping, history = fit_fixed(
        outer_refit,
        epochs=int(freeze["frozen_selected_epoch"]),
        device=device,
        log_every=log_every,
    )
    payload = _checkpoint_payload(
        registry=registry,
        phase="confirmation",
        refit_role="outer_refit",
        selected_epoch=int(freeze["frozen_selected_epoch"]),
        model=model,
        memory=memory,
        origin=origin,
        device_mapping=mapping,
        selection_history=None,
        refit_history=history,
    )
    payload["freeze_sha256"] = freeze["freeze_sha256"]
    payload["training_seconds"] = float(time.perf_counter() - started)
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "anchors": int(model.n_anchors),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": payload["training_seconds"],
    }


def _create_lock(
    path: str | Path,
    *,
    registry_sha256: str,
    freeze_sha256: str,
    checkpoint_sha256: str,
) -> dict:
    destination = Path(path)
    if destination.exists():
        raise PermissionError("V18.3 confirmation has already been consumed")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry_sha256,
        "freeze_sha256": freeze_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "confirmation_consumed": True,
        "note": "Written before loading the confirmation shard.",
    }
    payload["lock_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def evaluate_confirmation_once(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    checkpoint_path: str | Path = DEFAULT_CONFIRMATION_CHECKPOINT,
    lock_path: str | Path = DEFAULT_CONFIRMATION_LOCK,
    output: str | Path = DEFAULT_CONFIRMATION_RESULT,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    freeze = load_freeze(freeze_path, registry=registry)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if Path(output).exists():
        raise FileExistsError(output)

    device = torch.device(device_name)
    model, checkpoint = load_v183_checkpoint(
        checkpoint_path,
        registry=registry,
        device=device,
        expected_phase="confirmation",
    )
    if checkpoint.get("freeze_sha256") != freeze["freeze_sha256"]:
        raise ValueError("V18.3 confirmation checkpoint freeze mismatch")

    lock = _create_lock(
        lock_path,
        registry_sha256=registry["registry_sha256"],
        freeze_sha256=freeze["freeze_sha256"],
        checkpoint_sha256=_sha256_file(checkpoint_path),
    )
    confirmation = load_role_shard(registry, "confirmation")
    origin = checkpoint["coord_origin"].cpu().numpy().astype(np.float32)
    metrics = evaluate_shard(
        model,
        confirmation,
        origin,
        posterior_temperature=float(
            freeze["frozen_posterior_temperature"]
        ),
        scale_multiplier=float(freeze["frozen_scale_multiplier"]),
        device=device,
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "confirmation_lock_sha256": lock["lock_sha256"],
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "confirmation_metrics": metrics,
        "one_shot_internal_confirmation": True,
        "single_member": True,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(output, result)
    return result


def load_confirmation_result(
    path: str | Path,
    *,
    registry: dict,
    freeze: dict,
) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18.3 confirmation result")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18.3 confirmation registry mismatch")
    if payload.get("freeze_sha256") != freeze["freeze_sha256"]:
        raise ValueError("V18.3 confirmation freeze mismatch")
    if payload.get("result_sha256") != _canonical_json_sha256(
        payload,
        omit="result_sha256",
    ):
        raise ValueError("V18.3 confirmation result self-hash failed")
    if payload.get("one_shot_internal_confirmation") is not True:
        raise ValueError("V18.3 confirmation result lacks one-shot marker")
    return payload


def train_final(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    confirmation_result_path: str | Path = DEFAULT_CONFIRMATION_RESULT,
    output: str | Path = DEFAULT_FINAL_CHECKPOINT,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    freeze = load_freeze(freeze_path, registry=registry)
    if not Path(confirmation_result_path).exists():
        raise PermissionError(
            "V18.3 final refit is allowed only after one-shot confirmation"
        )
    confirmation_record = load_confirmation_result(
        confirmation_result_path,
        registry=registry,
        freeze=freeze,
    )
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)

    outer_refit = load_role_shard(registry, "outer_refit")
    confirmation = load_role_shard(registry, "confirmation")
    full = combine_role_shards(
        outer_refit,
        confirmation,
        role="full_train",
    )
    if len(set(_decode_shard_keys(full))) != int(
        registry["training_coordinate_count"]
    ):
        raise RuntimeError("V18.3 full refit does not reconstruct trainingData")

    device = torch.device(device_name)
    started = time.perf_counter()
    model, memory, origin, mapping, history = fit_fixed(
        full,
        epochs=int(freeze["frozen_selected_epoch"]),
        device=device,
        log_every=log_every,
    )
    payload = _checkpoint_payload(
        registry=registry,
        phase="final",
        refit_role="full_train",
        selected_epoch=int(freeze["frozen_selected_epoch"]),
        model=model,
        memory=memory,
        origin=origin,
        device_mapping=mapping,
        selection_history=None,
        refit_history=history,
    )
    payload["freeze_sha256"] = freeze["freeze_sha256"]
    payload["confirmation_result_sha256"] = confirmation_record[
        "result_sha256"
    ]
    payload["confirmation_result_file_sha256"] = _sha256_file(
        confirmation_result_path
    )
    payload["training_seconds"] = float(time.perf_counter() - started)
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "anchors": int(model.n_anchors),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": payload["training_seconds"],
    }


def _benchmark_as_shard(data: dict) -> dict:
    return {
        "rss": data["rss"].float().cpu(),
        "raw_xy": torch.from_numpy(
            np.asarray(data["raw_xy"], dtype=np.float32)
        ),
        "building_t": data["building_t"].long().cpu(),
        "floor_t": data["floor_t"].long().cpu(),
        "phone_t": data["phone_t"].long().cpu(),
        "timestamp_t": data["timestamp_t"].cpu(),
        "stats": data["stats"],
        "keys": [_encode_key(key) for key in data["keys"]],
    }


def evaluate_preexposed_benchmark(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    checkpoint_path: str | Path = DEFAULT_FINAL_CHECKPOINT,
    benchmark_data: str | Path = DEFAULT_BENCHMARK,
    output: str | Path = DEFAULT_BENCHMARK_RESULT,
    device_name: str = "cuda",
) -> dict:
    """Evaluate the already exposed official validation set descriptively."""

    registry = load_registry(registry_path)
    freeze = load_freeze(freeze_path, registry=registry)
    if Path(output).exists():
        raise FileExistsError(output)
    model, checkpoint = load_v183_checkpoint(
        checkpoint_path,
        registry=registry,
        device=torch.device(device_name),
        expected_phase="final",
    )
    if checkpoint.get("freeze_sha256") != freeze["freeze_sha256"]:
        raise ValueError("V18.3 final checkpoint freeze mismatch")

    benchmark_path = Path(benchmark_data).resolve()
    benchmark = _load_uji(
        benchmark_path,
        reference_stats=checkpoint["stats"],
    )
    shard = _benchmark_as_shard(benchmark)
    metrics = evaluate_shard(
        model,
        shard,
        checkpoint["coord_origin"].cpu().numpy().astype(np.float32),
        posterior_temperature=float(
            freeze["frozen_posterior_temperature"]
        ),
        scale_multiplier=float(freeze["frozen_scale_multiplier"]),
        device=torch.device(device_name),
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "final_checkpoint_sha256": _sha256_file(checkpoint_path),
        "benchmark_data_sha256": _sha256_file(benchmark_path),
        "benchmark_metrics": metrics,
        "benchmark_was_pre_exposed_before_v18_3": True,
        "blind_confirmation_claim": False,
        "used_for_model_selection": False,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V18.3 adaptive multi-evidence UJI experiment"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--shard-dir", default=str(DEFAULT_SHARDS))

    p = sub.add_parser("train-selection")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--output", default=str(DEFAULT_SELECTION_CHECKPOINT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=5)

    p = sub.add_parser("calibrate-freeze")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--checkpoint", default=str(DEFAULT_SELECTION_CHECKPOINT))
    p.add_argument("--result", default=str(DEFAULT_CALIBRATION_RESULT))
    p.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("stability")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    p.add_argument("--output", default=str(DEFAULT_STABILITY_RESULT))
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("train-confirmation")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    p.add_argument("--output", default=str(DEFAULT_CONFIRMATION_CHECKPOINT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=5)

    p = sub.add_parser("confirm")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    p.add_argument("--checkpoint", default=str(DEFAULT_CONFIRMATION_CHECKPOINT))
    p.add_argument("--lock", default=str(DEFAULT_CONFIRMATION_LOCK))
    p.add_argument("--output", default=str(DEFAULT_CONFIRMATION_RESULT))
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("train-final")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    p.add_argument(
        "--confirmation-result",
        default=str(DEFAULT_CONFIRMATION_RESULT),
    )
    p.add_argument("--output", default=str(DEFAULT_FINAL_CHECKPOINT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=5)

    p = sub.add_parser("benchmark")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    p.add_argument("--checkpoint", default=str(DEFAULT_FINAL_CHECKPOINT))
    p.add_argument("--benchmark-data", default=str(DEFAULT_BENCHMARK))
    p.add_argument("--output", default=str(DEFAULT_BENCHMARK_RESULT))
    p.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare_protocol(
            train_data=args.train_data,
            registry_path=args.registry,
            shard_dir=args.shard_dir,
        )
    elif args.command == "train-selection":
        payload = train_selection(
            registry_path=args.registry,
            output=args.output,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "calibrate-freeze":
        payload = calibrate_and_freeze(
            registry_path=args.registry,
            checkpoint_path=args.checkpoint,
            result_path=args.result,
            freeze_path=args.freeze,
            device_name=args.device,
        )
    elif args.command == "stability":
        payload = run_stability_panel(
            registry_path=args.registry,
            freeze_path=args.freeze,
            output=args.output,
            device_name=args.device,
        )
    elif args.command == "train-confirmation":
        payload = train_confirmation_model(
            registry_path=args.registry,
            freeze_path=args.freeze,
            output=args.output,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "confirm":
        payload = evaluate_confirmation_once(
            registry_path=args.registry,
            freeze_path=args.freeze,
            checkpoint_path=args.checkpoint,
            lock_path=args.lock,
            output=args.output,
            device_name=args.device,
        )
    elif args.command == "train-final":
        payload = train_final(
            registry_path=args.registry,
            freeze_path=args.freeze,
            confirmation_result_path=args.confirmation_result,
            output=args.output,
            device_name=args.device,
            log_every=args.log_every,
        )
    else:
        payload = evaluate_preexposed_benchmark(
            registry_path=args.registry,
            freeze_path=args.freeze,
            checkpoint_path=args.checkpoint,
            benchmark_data=args.benchmark_data,
            output=args.output,
            device_name=args.device,
        )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
