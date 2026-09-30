"""V18.1 registered UJI experiment with a sealed external confirmation.

V18.1 treats all of trainingData as development material because V18.0 already
revealed one internal outer split. The still-unopened official validation file
is reserved as the only one-shot external confirmation set.

Five formal stages:
1. prepare: split trainingData by physical coordinate and materialize role shards;
2. train-selection: select an epoch by Energy Score, then refit one checkpoint;
3. calibrate-freeze: fit only posterior temperature and scale multiplier;
4. train-final: refit one checkpoint on all trainingData;
5. external: create a consumption lock, then open the official validation file once.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from fhm_dapp_tim.fhm_dapp.v18_1_single import (
    V181_CONFIG,
    build_v181_model,
    validate_v181_config,
    v181_predict,
    v181_training_loss,
)


PROTOCOL_VERSION = "uji_v18_1_single_geometry_radio_calibrated_20260930"
GEOMETRY_SEED = 18111
EARLY_STOP_SEED = 18112
MODEL_SEED = 18311
SCORING_SEED = 18511
SELECTION_REFIT_FRACTION = 0.80
EARLY_STOP_FRACTION = 0.15
PREDICTIVE_DRAWS = 64
CALIBRATION_TEMPERATURE_GRID = (
    0.50,
    0.75,
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
DEFAULT_EXTERNAL = REPO_ROOT / "data" / "UJIndoorLoc" / "validationData.csv"
DEFAULT_EVIDENCE = ROOT / "evidence" / "v18_1"
DEFAULT_MODELS = ROOT / "models" / "v18_1"
DEFAULT_SHARDS = DEFAULT_EVIDENCE / "role_shards"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "registry_v18_1.json"
DEFAULT_SELECTION_CHECKPOINT = DEFAULT_MODELS / "selection.pt"
DEFAULT_CALIBRATION_RESULT = DEFAULT_EVIDENCE / "calibration_result_v18_1.json"
DEFAULT_FREEZE = DEFAULT_EVIDENCE / "selection_freeze_v18_1.json"
DEFAULT_FINAL_CHECKPOINT = DEFAULT_MODELS / "final.pt"
DEFAULT_EXTERNAL_LOCK = DEFAULT_EVIDENCE / "external_validation_consumed.lock"
DEFAULT_EXTERNAL_RESULT = DEFAULT_EVIDENCE / "external_result_v18_1.json"

LocationKey = tuple[int, int, float, float]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _strata(keys: set[LocationKey]) -> dict[tuple[int, int], set[LocationKey]]:
    result: dict[tuple[int, int], set[LocationKey]] = {}
    for key in keys:
        result.setdefault(key[:2], set()).add(key)
    return result


def construct_coordinate_roles(
    all_keys: set[LocationKey],
) -> tuple[dict[str, set[LocationKey]], dict]:
    """Create V18.1 development roles using only trainingData coordinates."""

    if not all_keys:
        raise ValueError("all_keys must be nonempty")
    selection_refit, selection_details = geometry_support_with_hull(
        all_keys,
        SELECTION_REFIT_FRACTION,
        seed=GEOMETRY_SEED,
    )
    calibration = all_keys - selection_refit
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
        "full_train": set(all_keys),
    }
    return roles, {
        "selection": selection_details,
        "early_stop": early_details,
    }


def audit_coordinate_roles(
    roles: Mapping[str, set[LocationKey]],
    all_keys: set[LocationKey],
) -> dict:
    """Fail on coordinate leakage, union errors, hull loss, or stratum loss."""

    if roles["member_fit"] & roles["early_stop"]:
        raise RuntimeError("member_fit overlaps early_stop")
    if roles["selection_refit"] & roles["calibration"]:
        raise RuntimeError("selection_refit overlaps calibration")
    if roles["member_fit"] | roles["early_stop"] != roles["selection_refit"]:
        raise RuntimeError("fit/stop roles do not reconstruct selection_refit")
    if roles["selection_refit"] | roles["calibration"] != all_keys:
        raise RuntimeError("selection/calibration do not reconstruct trainingData")
    if roles["full_train"] != all_keys:
        raise RuntimeError("full_train does not equal all trainingData coordinates")

    expected_strata = set(_strata(all_keys))
    for name, keys in roles.items():
        if set(_strata(keys)) != expected_strata:
            raise RuntimeError(f"{name} loses a building-floor stratum")

    hull_checks: dict[str, int] = {}
    for source_name, reference_name in (
        ("all", "selection_refit"),
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
            hull = {ordered[index] for index in _hull_local_indices(xy)}
            missing += len(hull - reference)
        hull_checks[f"{source_name}_hull_missing_from_{reference_name}"] = missing
        if missing:
            raise RuntimeError(
                f"{reference_name} misses {missing} mandatory hull coordinates"
            )

    return {
        "coordinate_grouping": "complete (building,floor,longitude,latitude)",
        "member_fit_early_stop_intersection": 0,
        "selection_refit_calibration_intersection": 0,
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
    """Register internal roles and materialize physically separate role files.

    This stage reads trainingData only. The sealed external validation file is
    neither opened nor hashed here.
    """

    validate_v181_config(V181_CONFIG)
    registry_destination = Path(registry_path)
    if registry_destination.exists():
        raise FileExistsError(registry_destination)

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
        destination = shard_root / f"{role}.pt"
        payload = _shard_payload(
            data,
            rows,
            role=role,
            source_sha256=source_sha,
        )
        _save_torch_new(destination, payload)
        shard_records[role] = {
            "path": str(destination.resolve()),
            "sha256": _sha256_file(destination),
            "row_count": int(len(rows)),
            "row_index_sha256": _sha256_indices(rows),
            "coordinate_count": len(keys),
            "coordinate_key_sha256": _sha256_keys(keys),
        }

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_status": (
            "V18.1 development uses trainingData only; the official validation "
            "file remains sealed until the external command after freeze and "
            "final refit."
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
            "geometry_seed": GEOMETRY_SEED,
            "early_stop_seed": EARLY_STOP_SEED,
            "model_seed": MODEL_SEED,
            "scoring_seed": SCORING_SEED,
            "selection_refit_fraction": SELECTION_REFIT_FRACTION,
            "early_stop_fraction": EARLY_STOP_FRACTION,
            "predictive_draws": PREDICTIVE_DRAWS,
            "calibration_temperature_grid": list(
                CALIBRATION_TEMPERATURE_GRID
            ),
            "calibration_scale_grid": list(CALIBRATION_SCALE_GRID),
        },
        "model_config": dict(V181_CONFIG),
        "single_member_contract": {
            "checkpoint_count_at_inference": 1,
            "residual_experts": 1,
            "member_axis": False,
            "consensus_operator": False,
        },
        "access_contract": {
            "trainingData_is_development_material": True,
            "external_validation_must_not_be_opened_before_freeze": True,
            "external_validation_must_not_be_opened_before_final_refit": True,
            "external_evaluation_is_one_shot": True,
        },
    }
    registry["registry_sha256"] = _canonical_json_sha256(registry)
    _write_json_new(registry_destination, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18.1 registry")
    if payload.get("registry_sha256") != _canonical_json_sha256(
        payload,
        omit="registry_sha256",
    ):
        raise ValueError("V18.1 registry self-hash failed")
    if payload.get("model_config") != V181_CONFIG:
        raise ValueError("V18.1 model configuration drifted")
    validate_v181_config(payload["model_config"])
    return payload


def load_role_shard(registry: dict, role: str) -> dict:
    """Load one physically materialized role and verify its hash chain."""

    if role not in registry["role_shards"]:
        raise ValueError(f"unknown V18.1 role: {role}")
    record = registry["role_shards"][role]
    path = Path(record["path"])
    if _sha256_file(path) != record["sha256"]:
        raise ValueError(f"V18.1 shard hash failed for {role}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"not a V18.1 shard: {role}")
    if payload.get("role") != role:
        raise ValueError(f"V18.1 shard role mismatch: {role}")
    if payload.get("source_sha256") != registry["train_data_sha256"]:
        raise ValueError(f"V18.1 shard source mismatch: {role}")
    rows = payload["row_indices"].cpu().numpy().astype(np.int64)
    if _sha256_indices(rows) != record["row_index_sha256"]:
        raise ValueError(f"V18.1 shard row hash failed: {role}")
    if len(rows) != int(record["row_count"]):
        raise ValueError(f"V18.1 shard row count failed: {role}")
    return payload


def _shard_arrays(shard: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_xy = shard["raw_xy"].cpu().numpy().astype(np.float32)
    building = shard["building_t"].cpu().numpy().astype(np.int64)
    floor = shard["floor_t"].cpu().numpy().astype(np.int64)
    return raw_xy, building, floor


def _build_training_state(
    shard: dict,
    *,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
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
        raise RuntimeError("V18.1 training row escaped the anchor dictionary")

    phone = shard["phone_t"].cpu().numpy().astype(np.int64)
    _mapped, device_mapping = _remap_training_devices(phone, rows)
    model, geometry, neighbours, neighbour_weights = build_v181_model(
        memory=memory,
        n_aps=int(shard["stats"]["n_aps"]),
        n_buildings=int(building.max()) + 1,
        n_devices=len(device_mapping),
        device=device,
        config=V181_CONFIG,
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
        batch_size=int(V181_CONFIG["batch_size"]),
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


def _decode_shard_keys(shard: dict) -> list[LocationKey]:
    return [_decode_key(value) for value in shard["keys"]]


def _nearest_anchor_recall(
    model: torch.nn.Module,
    target: torch.Tensor,
    target_building: torch.Tensor,
    target_floor: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    cutoffs: Sequence[int] = (8, 16, 32, 64),
) -> dict[int, torch.Tensor]:
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
    result: dict[int, torch.Tensor] = {}
    for cutoff in cutoffs:
        width = min(int(cutoff), candidate_indices.shape[1])
        result[int(cutoff)] = (
            candidate_indices[:, :width] == nearest.unsqueeze(1)
        ).any(dim=1).float()
    return result


@torch.no_grad()
def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"non-finite V18.1 tensor: {name}")


def evaluate_shard(
    model: torch.nn.Module,
    shard: dict,
    origin: np.ndarray,
    *,
    posterior_temperature: float,
    scale_multiplier: float,
    device: torch.device,
    batch_size: int = 192,
) -> dict:
    """Evaluate one role with probability, calibration, and retrieval diagnostics."""

    model.eval()
    raw_xy, _building_np, _floor_np = _shard_arrays(shard)
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
        )
    }
    recall_parts: dict[int, list[torch.Tensor]] = {
        cutoff: [] for cutoff in (8, 16, 32, 64)
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

        result = v181_predict(
            model,
            rss,
            top_k=min(int(V181_CONFIG["inference_top_k"]), model.n_anchors),
            posterior_temperature=float(posterior_temperature),
            scale_multiplier=float(scale_multiplier),
            target_coords=target,
            config=V181_CONFIG,
        )
        for name in (
            "nll",
            "coords",
            "candidate_weights",
            "component_locations",
            "component_scales",
            "radio_energy",
            "radial_scale",
        ):
            _finite_tensor(name, result[name])

        error = torch.linalg.vector_norm(
            result["coords"] - target,
            dim=1,
        )
        building_error = (
            result["building"] != target_building
        ).float()
        floor_error = (
            result["floor"] != target_floor
        ).float()

        randomness = common_student_t_randomness(
            len(target),
            draws=PREDICTIVE_DRAWS,
            degrees_freedom=float(V181_CONFIG["student_df"]),
            seed=SCORING_SEED + start,
        )
        energy, first, second = continuous_energy_score_mc(
            result["candidate_weights"],
            result["component_locations"],
            result["component_scales"],
            target,
            randomness,
        )
        draws = torch.cat([first, second], dim=1)
        draw_radius = torch.linalg.vector_norm(
            draws - result["coords"].unsqueeze(1),
            dim=2,
        )

        values["nll"].append(result["nll"].cpu())
        values["error"].append(error.cpu())
        values["energy"].append(energy.cpu())
        values["building_error"].append(building_error.cpu())
        values["floor_error"].append(floor_error.cpu())
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
                result["candidate_weights"] * result["radio_energy"]
            ).sum(dim=1).cpu()
        )
        values["style_offset_abs"].append(
            result["style_offset"].abs().cpu()
        )
        values["style_scale"].append(result["style_scale"].cpu())

        recalls = _nearest_anchor_recall(
            model,
            target,
            target_building,
            target_floor,
            result["candidate_indices"],
        )
        for cutoff, recall in recalls.items():
            recall_parts[cutoff].append(recall.cpu())

        for level in COVERAGE_LEVELS:
            radius = torch.quantile(
                draw_radius,
                float(level),
                dim=1,
            )
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
    if np.std(error_np) > 0.0 and np.std(radial_np) > 0.0:
        error_radial_corr = float(np.corrcoef(error_np, radial_np)[0, 1])
    else:
        error_radial_corr = 0.0

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
        "mean_within_component_radial_sd_m": float(
            joined["within_sd"].mean()
        ),
        "mean_between_anchor_radial_sd_m": float(
            joined["between_sd"].mean()
        ),
        "mean_posterior_radio_energy": float(
            joined["radio_energy"].mean()
        ),
        "mean_abs_profiled_style_offset": float(
            joined["style_offset_abs"].mean()
        ),
        "mean_profiled_style_scale": float(
            joined["style_scale"].mean()
        ),
        "error_radial_scale_correlation": error_radial_corr,
        "coordinate_groups": groups,
        "rows": int(len(raw_xy)),
    }
    for cutoff, parts in recall_parts.items():
        metrics[f"nearest_same_stratum_anchor_recall_at_{cutoff}"] = float(
            torch.cat(parts).mean()
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
    model: torch.nn.Module,
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
        terms = v181_training_loss(
            model,
            rss,
            target,
            anchor,
            device_id,
            neighbours,
            neighbour_weights,
            config=V181_CONFIG,
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(V181_CONFIG["gradient_clip"]),
        )
        optimizer.step()
        batch = len(rss)
        count += batch
        for name, value in terms.items():
            if value.ndim == 0:
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach()
                ) * batch
    return {
        name: value / count
        for name, value in totals.items()
    }


def select_epoch_from_history(history: Sequence[dict]) -> int:
    """Choose an eligible near-best Energy Score epoch with an MAE guardrail."""

    minimum = int(V181_CONFIG["minimum_epochs"])
    eligible = [
        record
        for record in history
        if int(record["epoch"]) >= minimum
    ]
    if not eligible:
        raise ValueError("history contains no epoch eligible for selection")
    best_energy = min(
        float(record["early_stop"]["coordinate_macro_energy_score_m"])
        for record in eligible
    )
    margin = float(V181_CONFIG["energy_tie_margin_m"])
    shortlist = [
        record
        for record in eligible
        if float(
            record["early_stop"]["coordinate_macro_energy_score_m"]
        ) <= best_energy + margin
    ]
    selected = min(
        shortlist,
        key=lambda record: (
            float(record["early_stop"]["coordinate_macro_mae_m"]),
            float(record["early_stop"]["coordinate_macro_energy_score_m"]),
            int(record["epoch"]),
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
        lr=float(V181_CONFIG["learning_rate"]),
        weight_decay=float(V181_CONFIG["weight_decay"]),
    )

    history: list[dict] = []
    best_running = float("inf")
    stale = 0
    for epoch in range(1, int(V181_CONFIG["maximum_epochs"]) + 1):
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
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "early_stop": stop_metrics,
            "eligible_for_selection": (
                epoch >= int(V181_CONFIG["minimum_epochs"])
            ),
        }
        history.append(record)

        if record["eligible_for_selection"]:
            energy = float(
                stop_metrics["coordinate_macro_energy_score_m"]
            )
            if energy < best_running - float(
                V181_CONFIG["minimum_improvement_energy_m"]
            ):
                best_running = energy
                stale = 0
            else:
                stale += 1

        if epoch == 1 or epoch % int(log_every) == 0:
            print(json.dumps(record, allow_nan=False), flush=True)

        if (
            epoch >= int(V181_CONFIG["minimum_epochs"])
            and stale >= int(V181_CONFIG["patience"])
        ):
            break

    selected_epoch = select_epoch_from_history(history)
    return selected_epoch, history


def fit_fixed(
    shard: dict,
    *,
    epochs: int,
    device: torch.device,
    log_every: int = 5,
) -> tuple[torch.nn.Module, dict, np.ndarray, dict, list[dict]]:
    if int(epochs) < int(V181_CONFIG["minimum_epochs"]):
        raise ValueError("V18.1 refit epoch count violates minimum_epochs")
    _set_seed(MODEL_SEED)
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
        seed=MODEL_SEED,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(V181_CONFIG["learning_rate"]),
        weight_decay=float(V181_CONFIG["weight_decay"]),
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
    model: torch.nn.Module,
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
        "config": dict(V181_CONFIG),
        "device_id_mapping": {
            str(raw): int(mapped)
            for raw, mapped in device_mapping.items()
        },
        "selection_history": selection_history,
        "refit_history": refit_history,
        "single_member_contract": {
            "checkpoint_count": 1,
            "member_axis": False,
            "residual_experts": 1,
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
        "training_seconds": payload["training_seconds"],
    }


def load_v181_checkpoint(
    path: str | Path,
    *,
    registry: dict,
    device: torch.device,
    expected_phase: str,
) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18.1 checkpoint")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18.1 checkpoint registry mismatch")
    if payload.get("phase") != expected_phase:
        raise ValueError("V18.1 checkpoint phase mismatch")
    if payload.get("config") != V181_CONFIG:
        raise ValueError("V18.1 checkpoint configuration mismatch")

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
    model, _geometry, _neighbours, _weights = build_v181_model(
        memory=memory,
        n_aps=int(payload["stats"]["n_aps"]),
        n_buildings=int(state["anchor_buildings"].max().item()) + 1,
        n_devices=len(payload.get("device_id_mapping", {})),
        device=device,
        config=V181_CONFIG,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def collect_calibration_cache(
    model: torch.nn.Module,
    shard: dict,
    origin: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 192,
) -> dict:
    raw_xy, _building, _floor = _shard_arrays(shard)
    centered = torch.from_numpy(
        (raw_xy - origin[None, :]).astype(np.float32)
    )
    output = {
        "logits": [],
        "location": [],
        "scale": [],
        "target": [],
    }
    for start in range(0, len(raw_xy), int(batch_size)):
        stop = min(start + int(batch_size), len(raw_xy))
        result = v181_predict(
            model,
            shard["rss"][start:stop].to(device),
            top_k=min(int(V181_CONFIG["inference_top_k"]), model.n_anchors),
            posterior_temperature=1.0,
            scale_multiplier=1.0,
            target_coords=centered[start:stop].to(device),
            config=V181_CONFIG,
        )
        output["logits"].append(
            result["candidate_logits_corrected"].cpu()
        )
        output["location"].append(
            result["component_locations"].cpu()
        )
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
    *,
    degrees_freedom: float,
) -> torch.Tensor:
    df = float(degrees_freedom)
    z = (target.unsqueeze(1) - location) / scale
    normalizer = (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
    )
    value = (
        normalizer
        - torch.log(scale)
        - 0.5 * (df + 1.0) * torch.log1p(z.square() / df)
    )
    return value.sum(dim=-1)


def calibration_macro_nll(
    cache: dict,
    keys: Sequence[LocationKey],
    *,
    posterior_temperature: float,
    scale_multiplier: float,
) -> float:
    weights = torch.softmax(
        cache["logits"] / float(posterior_temperature),
        dim=1,
    )
    component_lp = _student_log_prob(
        cache["target"],
        cache["location"],
        cache["scale"] * float(scale_multiplier),
        degrees_freedom=float(V181_CONFIG["student_df"]),
    )
    nll = -torch.logsumexp(
        torch.log(weights.clamp(min=1e-12)) + component_lp,
        dim=1,
    )
    mean, _se, _groups = _coordinate_macro(nll, keys)
    return mean


def fit_calibration_grid(
    cache: dict,
    keys: Sequence[LocationKey],
) -> dict:
    surface: list[dict] = []
    for temperature in CALIBRATION_TEMPERATURE_GRID:
        for scale in CALIBRATION_SCALE_GRID:
            score = calibration_macro_nll(
                cache,
                keys,
                posterior_temperature=temperature,
                scale_multiplier=scale,
            )
            surface.append(
                {
                    "posterior_temperature": float(temperature),
                    "scale_multiplier": float(scale),
                    "coordinate_macro_mixture_nll": float(score),
                }
            )
    selected = min(
        surface,
        key=lambda item: (
            item["coordinate_macro_mixture_nll"],
            abs(math.log(item["posterior_temperature"]))
            + abs(math.log(item["scale_multiplier"])),
            item["posterior_temperature"],
            item["scale_multiplier"],
        ),
    )
    selected_temperature = float(selected["posterior_temperature"])
    selected_scale = float(selected["scale_multiplier"])
    return {
        "posterior_temperature": selected_temperature,
        "scale_multiplier": selected_scale,
        "best_coordinate_macro_mixture_nll": float(
            selected["coordinate_macro_mixture_nll"]
        ),
        "temperature_on_grid_boundary": selected_temperature
        in {
            min(CALIBRATION_TEMPERATURE_GRID),
            max(CALIBRATION_TEMPERATURE_GRID),
        },
        "scale_on_grid_boundary": selected_scale
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
    calibration_result_path: str | Path = DEFAULT_CALIBRATION_RESULT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    result_destination = Path(calibration_result_path)
    freeze_destination = Path(freeze_path)
    if result_destination.exists() or freeze_destination.exists():
        raise FileExistsError("V18.1 calibration result/freeze already exists")

    device = torch.device(device_name)
    model, checkpoint = load_v181_checkpoint(
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
        "selection_checkpoint": str(Path(checkpoint_path).resolve()),
        "selection_checkpoint_sha256": _sha256_file(checkpoint_path),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "calibration": calibration,
        "calibration_metrics": metrics,
        "calibration_role_is_internal_development": True,
        "external_validation_accessed": False,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(result_destination, result)

    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "selection_checkpoint_sha256": result[
            "selection_checkpoint_sha256"
        ],
        "calibration_result_sha256": result["result_sha256"],
        "frozen_selected_epoch": int(checkpoint["selected_epoch"]),
        "frozen_model_config": dict(V181_CONFIG),
        "frozen_posterior_temperature": float(
            calibration["posterior_temperature"]
        ),
        "frozen_scale_multiplier": float(
            calibration["scale_multiplier"]
        ),
        "external_validation_access_authorized_after_final_refit": True,
    }
    freeze["freeze_sha256"] = _canonical_json_sha256(freeze)
    _write_json_new(freeze_destination, freeze)
    return {
        "calibration": result,
        "freeze": freeze,
    }


def load_freeze(
    path: str | Path = DEFAULT_FREEZE,
    *,
    registry: dict,
) -> dict:
    source = Path(path)
    if not source.exists():
        raise PermissionError("V18.1 external access is forbidden before freeze")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18.1 freeze")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18.1 freeze registry mismatch")
    if payload.get("freeze_sha256") != _canonical_json_sha256(
        payload,
        omit="freeze_sha256",
    ):
        raise ValueError("V18.1 freeze self-hash failed")
    if payload.get("frozen_model_config") != V181_CONFIG:
        raise ValueError("V18.1 frozen model config drifted")
    if int(payload["frozen_selected_epoch"]) < int(
        V181_CONFIG["minimum_epochs"]
    ):
        raise ValueError("V18.1 freeze selected an ineligible early epoch")
    return payload


def train_final(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    output: str | Path = DEFAULT_FINAL_CHECKPOINT,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    freeze = load_freeze(freeze_path, registry=registry)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)

    full_shard = load_role_shard(registry, "full_train")
    device = torch.device(device_name)
    started = time.perf_counter()
    model, memory, origin, mapping, history = fit_fixed(
        full_shard,
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
    payload["training_seconds"] = float(time.perf_counter() - started)
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "selected_epoch": int(freeze["frozen_selected_epoch"]),
        "anchors": int(model.n_anchors),
        "training_seconds": payload["training_seconds"],
    }


def _external_as_shard(data: dict) -> dict:
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


def evaluate_external_once(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    checkpoint_path: str | Path = DEFAULT_FINAL_CHECKPOINT,
    external_data: str | Path = DEFAULT_EXTERNAL,
    lock_path: str | Path = DEFAULT_EXTERNAL_LOCK,
    output: str | Path = DEFAULT_EXTERNAL_RESULT,
    device_name: str = "cuda",
) -> dict:
    """Consume the sealed external file exactly once.

    The lock is written before any external file read. A crash after that point
    still counts as consuming the external confirmation set.
    """

    registry = load_registry(registry_path)
    freeze = load_freeze(freeze_path, registry=registry)
    lock_destination = Path(lock_path)
    result_destination = Path(output)
    if lock_destination.exists():
        raise PermissionError("V18.1 external validation has already been consumed")
    if result_destination.exists():
        raise FileExistsError(result_destination)
    final_path = Path(checkpoint_path)
    if not final_path.exists():
        raise FileNotFoundError("V18.1 final checkpoint is required before external access")

    device = torch.device(device_name)
    model, checkpoint = load_v181_checkpoint(
        final_path,
        registry=registry,
        device=device,
        expected_phase="final",
    )
    if checkpoint.get("freeze_sha256") != freeze["freeze_sha256"]:
        raise ValueError("V18.1 final checkpoint freeze mismatch")
    if int(checkpoint["selected_epoch"]) != int(
        freeze["frozen_selected_epoch"]
    ):
        raise ValueError("V18.1 final checkpoint epoch mismatch")

    # This is the irreversible access boundary.
    lock_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "final_checkpoint_sha256": _sha256_file(final_path),
        "external_consumed": True,
        "note": (
            "Written before opening the external validation file. If the "
            "evaluation crashes after this lock, do not silently rerun it."
        ),
    }
    lock_payload["lock_sha256"] = _canonical_json_sha256(lock_payload)
    _write_json_new(lock_destination, lock_payload)

    external_path = Path(external_data).resolve()
    external = _load_uji(
        external_path,
        reference_stats=checkpoint["stats"],
    )
    external_sha = _sha256_file(external_path)
    shard = _external_as_shard(external)
    origin = checkpoint["coord_origin"].cpu().numpy().astype(np.float32)
    metrics = evaluate_shard(
        model,
        shard,
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
        "external_lock_sha256": lock_payload["lock_sha256"],
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": lock_payload[
            "final_checkpoint_sha256"
        ],
        "external_data_sha256": external_sha,
        "external_rows": int(len(external["keys"])),
        "external_coordinate_count": len(set(external["keys"])),
        "frozen_posterior_temperature": float(
            freeze["frozen_posterior_temperature"]
        ),
        "frozen_scale_multiplier": float(
            freeze["frozen_scale_multiplier"]
        ),
        "external_metrics": metrics,
        "single_member": True,
        "external_evaluation_count_for_this_protocol": 1,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(result_destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V18.1 single-member UJI experiment"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    prepare.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    prepare.add_argument("--shard-dir", default=str(DEFAULT_SHARDS))

    train_sel = sub.add_parser("train-selection")
    train_sel.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train_sel.add_argument("--output", default=str(DEFAULT_SELECTION_CHECKPOINT))
    train_sel.add_argument("--device", default="cuda")
    train_sel.add_argument("--log-every", type=int, default=5)

    calibrate = sub.add_parser("calibrate-freeze")
    calibrate.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    calibrate.add_argument(
        "--checkpoint",
        default=str(DEFAULT_SELECTION_CHECKPOINT),
    )
    calibrate.add_argument(
        "--calibration-result",
        default=str(DEFAULT_CALIBRATION_RESULT),
    )
    calibrate.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    calibrate.add_argument("--device", default="cuda")

    train_fin = sub.add_parser("train-final")
    train_fin.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train_fin.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    train_fin.add_argument("--output", default=str(DEFAULT_FINAL_CHECKPOINT))
    train_fin.add_argument("--device", default="cuda")
    train_fin.add_argument("--log-every", type=int, default=5)

    external = sub.add_parser("external")
    external.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    external.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    external.add_argument(
        "--checkpoint",
        default=str(DEFAULT_FINAL_CHECKPOINT),
    )
    external.add_argument(
        "--external-data",
        default=str(DEFAULT_EXTERNAL),
    )
    external.add_argument("--lock", default=str(DEFAULT_EXTERNAL_LOCK))
    external.add_argument("--output", default=str(DEFAULT_EXTERNAL_RESULT))
    external.add_argument("--device", default="cuda")

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
            calibration_result_path=args.calibration_result,
            freeze_path=args.freeze,
            device_name=args.device,
        )
    elif args.command == "train-final":
        payload = train_final(
            registry_path=args.registry,
            freeze_path=args.freeze,
            output=args.output,
            device_name=args.device,
            log_every=args.log_every,
        )
    else:
        payload = evaluate_external_once(
            registry_path=args.registry,
            freeze_path=args.freeze,
            checkpoint_path=args.checkpoint,
            external_data=args.external_data,
            lock_path=args.lock,
            output=args.output,
            device_name=args.device,
        )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
