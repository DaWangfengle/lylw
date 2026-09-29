"""Registered nested UJI experiment for V18 single-member localization.

V18 is a fresh one-checkpoint experiment. Historical V9/V17 results motivate
its mechanism, but no historical checkpoint, selected ensemble size, consensus
parameter, or outcome is consumed.

Protocol:
1. Register leakage-controlled physical-coordinate roles inside trainingData.
2. Select an epoch using member_fit -> early_stop only.
3. Refit exactly one V18 checkpoint on selection_refit for that epoch count.
4. Evaluate development once and write an immutable freeze.
5. Refit exactly one final checkpoint on outer_refit for the frozen epoch count.
6. Only after the freeze, evaluate outer_test once.

The official UJI validation split is deliberately outside this main experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _decode_key,
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
from fhm_dapp_tim.fhm_dapp.audit_bplus_sparse50_fem_v5 import (
    _hull_local_indices,
    geometry_support_with_hull,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import (
    _remap_training_devices,
    _training_reference_stats,
)
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness,
    continuous_energy_score_mc,
)
from fhm_dapp_tim.fhm_dapp.v18_single import (
    V18_CONFIG,
    build_v18_model,
    validate_v18_config,
    v18_training_loss,
)


PROTOCOL_VERSION = "uji_v18_single_geometry_radio_v1_20260929"
GEOMETRY_SEED = 18101
INNER_GEOMETRY_SEED = 18102
EARLY_STOP_SEED = 18103
MODEL_SEED = 18301
SCORING_SEED = 18501
OUTER_REFIT_FRACTION = 0.80
SELECTION_REFIT_FRACTION = 0.80
EARLY_STOP_FRACTION = 0.15
FLOOR_HEIGHT_M = 4.0
PREDICTIVE_DRAWS = 64

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "UJIndoorLoc" / "trainingData.csv"
DEFAULT_EVIDENCE = ROOT / "evidence" / "v18_single"
DEFAULT_MODELS = ROOT / "models" / "v18_single"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "registry_v18.json"
DEFAULT_SELECTION_CHECKPOINT = DEFAULT_MODELS / "selection.pt"
DEFAULT_DEVELOPMENT_RESULT = DEFAULT_EVIDENCE / "development_result_v18.json"
DEFAULT_FREEZE = DEFAULT_EVIDENCE / "selection_freeze_v18.json"
DEFAULT_FINAL_CHECKPOINT = DEFAULT_MODELS / "final.pt"
DEFAULT_OUTER_RESULT = DEFAULT_EVIDENCE / "outer_test_result_v18.json"

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


def _registry_without_hash(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("registry_sha256", None)
    return value


def _checkpoint_sha(path: str | Path) -> str:
    return _sha256_file(Path(path))


def _strata(keys: set[LocationKey]) -> dict[tuple[int, int], set[LocationKey]]:
    result: dict[tuple[int, int], set[LocationKey]] = {}
    for key in keys:
        result.setdefault(key[:2], set()).add(key)
    return result


def construct_coordinate_roles(
    all_keys: set[LocationKey],
) -> tuple[dict[str, set[LocationKey]], dict]:
    """Create the fixed V18 nested coordinate roles with hull coverage."""

    if not all_keys:
        raise ValueError("all_keys must be nonempty")
    outer_refit, outer_details = geometry_support_with_hull(
        all_keys, OUTER_REFIT_FRACTION, seed=GEOMETRY_SEED
    )
    outer_test = set(all_keys) - outer_refit
    selection_refit, selection_details = geometry_support_with_hull(
        outer_refit, SELECTION_REFIT_FRACTION, seed=INNER_GEOMETRY_SEED
    )
    development = outer_refit - selection_refit
    nested, early_details = support_fit_stop_roles(
        selection_refit,
        seed=EARLY_STOP_SEED,
        early_stop_fraction=EARLY_STOP_FRACTION,
    )
    roles = {
        "member_fit": nested["member_fit"],
        "early_stop": nested["early_stop"],
        "selection_refit": selection_refit,
        "development": development,
        "outer_refit": outer_refit,
        "outer_test": outer_test,
    }
    return roles, {
        "outer": outer_details,
        "inner_selection": selection_details,
        "early_stop": early_details,
    }


def audit_coordinate_roles(
    roles: Mapping[str, set[LocationKey]],
    data: Mapping[str, object],
) -> dict:
    """Fail on coordinate leakage, union errors, hull loss or stratum loss."""

    all_keys = set(data["keys"])
    intersections = {
        "member_fit_early_stop": len(roles["member_fit"] & roles["early_stop"]),
        "selection_refit_development": len(
            roles["selection_refit"] & roles["development"]
        ),
        "outer_refit_outer_test": len(roles["outer_refit"] & roles["outer_test"]),
    }
    if any(intersections.values()):
        raise RuntimeError(f"coordinate leakage detected: {intersections}")
    if roles["member_fit"] | roles["early_stop"] != roles["selection_refit"]:
        raise RuntimeError("fit/stop roles do not reconstruct selection_refit")
    if roles["selection_refit"] | roles["development"] != roles["outer_refit"]:
        raise RuntimeError("selection/development do not reconstruct outer_refit")
    if roles["outer_refit"] | roles["outer_test"] != all_keys:
        raise RuntimeError("outer roles do not reconstruct training coordinates")

    all_strata = set(_strata(all_keys))
    for name, keys in roles.items():
        if set(_strata(keys)) != all_strata:
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
            xy = np.asarray([[key[2], key[3]] for key in ordered], dtype=np.float64)
            hull = {ordered[index] for index in _hull_local_indices(xy)}
            missing += len(hull - reference)
        hull_checks[f"{source_name}_hull_missing_from_{reference_name}"] = missing
        if missing:
            raise RuntimeError(
                f"{reference_name} misses {missing} mandatory hull coordinates"
            )
    return {
        "coordinate_grouping": "complete (building,floor,longitude,latitude)",
        "disjoint_intersections": intersections,
        "exact_union_identities": True,
        "all_roles_keep_all_building_floor_strata": True,
        "hull_checks": hull_checks,
    }


def register_protocol(
    *,
    train_data: str | Path = DEFAULT_TRAIN,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    validate_v18_config(V18_CONFIG)
    train_path = Path(train_data).resolve()
    data = _load_uji(train_path)
    roles, split_details = construct_coordinate_roles(set(data["keys"]))
    diagnostics = audit_coordinate_roles(roles, data)
    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_status": (
            "fresh single-member V18; nested coordinate split inside "
            "trainingData.csv; no historical outcome or official validation "
            "file is consumed"
        ),
        "train_data": str(train_path),
        "train_data_sha256": _sha256_file(train_path),
        "training_rows": int(len(data["keys"])),
        "training_coordinate_count": int(len(set(data["keys"]))),
        "training_coordinate_sha256": _sha256_keys(set(data["keys"])),
        "n_aps": int(data["stats"]["n_aps"]),
        "phone_ids": sorted(map(int, np.unique(data["phone"]))),
        "roles": {
            name: _role_record(set(keys), data["keys"])
            for name, keys in roles.items()
        },
        "split_details": split_details,
        "split_diagnostics": diagnostics,
        "constants": {
            "geometry_seed": GEOMETRY_SEED,
            "inner_geometry_seed": INNER_GEOMETRY_SEED,
            "early_stop_seed": EARLY_STOP_SEED,
            "model_seed": MODEL_SEED,
            "scoring_seed": SCORING_SEED,
            "outer_refit_fraction": OUTER_REFIT_FRACTION,
            "selection_refit_fraction": SELECTION_REFIT_FRACTION,
            "early_stop_fraction": EARLY_STOP_FRACTION,
            "floor_height_m": FLOOR_HEIGHT_M,
            "predictive_draws": PREDICTIVE_DRAWS,
        },
        "model_config": dict(V18_CONFIG),
        "single_member_contract": {
            "checkpoint_count_at_inference": 1,
            "residual_experts": 1,
            "member_axis": False,
            "consensus_operator": False,
        },
        "access_contract": {
            "development_allowed_before_freeze": True,
            "outer_test_allowed_before_freeze": False,
            "official_validation_in_scope": False,
        },
    }
    registry["registry_sha256"] = _canonical_json_sha256(registry)
    _write_json_new(output, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18 registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(
        _registry_without_hash(registry)
    ):
        raise ValueError("V18 registry self-hash failed")
    if _sha256_file(registry["train_data"]) != registry["train_data_sha256"]:
        raise ValueError("V18 training file changed after registration")
    if registry.get("model_config") != V18_CONFIG:
        raise ValueError("V18 model configuration changed after registration")
    if registry.get("access_contract", {}).get("official_validation_in_scope") is not False:
        raise ValueError("V18 main experiment illegally admits official validation")
    return registry


def _registered_role_keys(registry: dict, role: str) -> set[LocationKey]:
    if role not in registry["roles"]:
        raise ValueError(f"unknown V18 role: {role}")
    record = registry["roles"][role]
    keys = {_decode_key(value) for value in record["coordinate_keys"]}
    if _sha256_keys(keys) != record["coordinate_key_sha256"]:
        raise ValueError(f"coordinate hash failed for role {role}")
    return keys


def _registered_rows(registry: dict, role: str, data: dict) -> np.ndarray:
    rows = _rows_for_keys(data["keys"], _registered_role_keys(registry, role))
    if _sha256_indices(rows) != registry["roles"][role]["row_index_sha256"]:
        raise ValueError(f"row hash failed for role {role}")
    return rows


def _build_training_state(
    data: dict,
    rows: np.ndarray,
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
    raw_xy = np.asarray(data["raw_xy"], dtype=np.float64)
    origin = raw_xy[rows].mean(axis=0).astype(np.float32)
    centered = (raw_xy - origin[None, :]).astype(np.float32)
    memory, sample_anchor = build_anchor_memory(
        data["rss"], centered, data["building"], data["floor"], rows
    )
    if np.any(sample_anchor[rows] < 0):
        raise RuntimeError("V18 training row escaped the anchor dictionary")
    device_ids, device_mapping = _remap_training_devices(data["phone"], rows)
    n_buildings = int(np.max(data["building"])) + 1
    model, geometry, neighbours, neighbour_weights = build_v18_model(
        memory=memory,
        n_aps=int(data["stats"]["n_aps"]),
        n_buildings=n_buildings,
        n_devices=len(device_mapping),
        device=device,
        config=V18_CONFIG,
    )
    stats = _training_reference_stats(
        data["stats"], raw_xy, data["building"], rows
    )
    return (
        model,
        memory,
        sample_anchor,
        torch.from_numpy(centered).float(),
        origin,
        neighbours,
        neighbour_weights,
        device_mapping,
        stats,
    )


def _loader(
    data: dict,
    centered: torch.Tensor,
    sample_anchor: np.ndarray,
    rows: np.ndarray,
    device_mapping: Mapping[int, int],
    *,
    seed: int,
) -> DataLoader:
    mapped = torch.tensor(
        [device_mapping[int(data["phone"][index])] for index in rows],
        dtype=torch.long,
    )
    dataset = TensorDataset(
        data["rss"][rows].float(),
        centered[rows],
        torch.from_numpy(sample_anchor[rows]).long(),
        mapped,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(V18_CONFIG["batch_size"]),
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


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data: dict,
    centered: torch.Tensor,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    model.eval()
    nll_parts: list[torch.Tensor] = []
    error_parts: list[torch.Tensor] = []
    building_errors: list[torch.Tensor] = []
    floor_errors: list[torch.Tensor] = []
    entropy_parts: list[torch.Tensor] = []
    radial_parts: list[torch.Tensor] = []
    energy_parts: list[torch.Tensor] = []

    for start in range(0, len(rows), int(batch_size)):
        selected = rows[start : start + int(batch_size)]
        rss = data["rss"][selected].to(device)
        target = centered[selected].to(device)
        result = model.predict(
            rss,
            top_k=min(int(V18_CONFIG["inference_top_k"]), model.n_anchors),
            target_coords=target,
            posterior_temperature=float(V18_CONFIG["posterior_temperature"]),
            condition_building=bool(V18_CONFIG["condition_building"]),
            deterministic_building=bool(V18_CONFIG["deterministic_building"]),
        )
        weights = result["candidate_weights"]
        nll_parts.append(result["nll"].cpu())
        error_parts.append(
            torch.linalg.vector_norm(result["coords"] - target, dim=1).cpu()
        )
        building_errors.append(
            (result["building"].cpu() != data["building_t"][selected]).float()
        )
        floor_errors.append(
            (result["floor"].cpu() != data["floor_t"][selected]).float()
        )
        entropy_parts.append(
            (-(weights * weights.clamp(min=1e-12).log()).sum(dim=1)).cpu()
        )
        radial_parts.append(result["radial_scale"].cpu())
        randomness = common_student_t_randomness(
            len(selected),
            draws=PREDICTIVE_DRAWS,
            degrees_freedom=float(V18_CONFIG["student_df"]),
            seed=SCORING_SEED + start,
        )
        energy, _, _ = continuous_energy_score_mc(
            weights.cpu(),
            result["component_locations"].cpu(),
            result["component_scales"].cpu(),
            target.cpu(),
            randomness,
        )
        energy_parts.append(energy)

    nll = torch.cat(nll_parts)
    error = torch.cat(error_parts)
    building_error = torch.cat(building_errors)
    floor_error = torch.cat(floor_errors)
    entropy = torch.cat(entropy_parts)
    radial = torch.cat(radial_parts)
    energy = torch.cat(energy_parts)
    keys = [data["keys"][int(row)] for row in rows]
    macro_nll, nll_se, groups = _coordinate_macro(nll, keys)
    macro_mae, mae_se, _ = _coordinate_macro(error, keys)
    macro_building, _, _ = _coordinate_macro(building_error, keys)
    macro_floor, _, _ = _coordinate_macro(floor_error, keys)
    macro_energy, energy_se, _ = _coordinate_macro(energy, keys)
    return {
        "coordinate_macro_mixture_nll": macro_nll,
        "coordinate_macro_mixture_nll_se": nll_se,
        "coordinate_macro_mae_m": macro_mae,
        "coordinate_macro_mae_se_m": mae_se,
        "coordinate_macro_building_error": macro_building,
        "coordinate_macro_floor_error": macro_floor,
        "coordinate_macro_energy_score_m": macro_energy,
        "coordinate_macro_energy_score_se_m": energy_se,
        "mean_anchor_entropy_nats": float(entropy.mean()),
        "mean_radial_scale_m": float(radial.mean()),
        "coordinate_groups": groups,
        "rows": int(len(rows)),
    }


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
        terms = v18_training_loss(
            model,
            rss,
            target,
            anchor,
            device_id,
            neighbours,
            neighbour_weights,
            config=V18_CONFIG,
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(V18_CONFIG["gradient_clip"])
        )
        optimizer.step()
        batch = len(rss)
        count += batch
        for name, value in terms.items():
            if value.ndim == 0:
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch
    return {name: value / count for name, value in totals.items()}


def select_epoch(
    data: dict,
    fit_rows: np.ndarray,
    stop_rows: np.ndarray,
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
        _origin,
        neighbours,
        neighbour_weights,
        device_mapping,
        _stats,
    ) = _build_training_state(data, fit_rows, device=device)
    loader = _loader(
        data,
        centered,
        sample_anchor,
        fit_rows,
        device_mapping,
        seed=MODEL_SEED,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(V18_CONFIG["learning_rate"]),
        weight_decay=float(V18_CONFIG["weight_decay"]),
    )
    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, int(V18_CONFIG["maximum_epochs"]) + 1):
        train = _train_epoch(
            model,
            loader,
            optimizer,
            neighbours,
            neighbour_weights,
            device=device,
        )
        stop = evaluate_model(model, data, centered, stop_rows, device=device)
        score = float(stop["coordinate_macro_mixture_nll"])
        improved = score < best - float(V18_CONFIG["minimum_improvement_nll"])
        if improved:
            best = score
            best_epoch = epoch
            stale = 0
        elif epoch >= int(V18_CONFIG["minimum_epochs"]):
            stale += 1
        record = {
            "epoch": epoch,
            "train": train,
            "early_stop": stop,
            "best_epoch": best_epoch,
            "best_coordinate_macro_mixture_nll": best,
        }
        history.append(record)
        if epoch == 1 or improved or epoch % int(log_every) == 0:
            print(json.dumps(record, allow_nan=False), flush=True)
        if (
            epoch >= int(V18_CONFIG["minimum_epochs"])
            and stale >= int(V18_CONFIG["patience"])
        ):
            break
    if best_epoch <= 0:
        raise RuntimeError("V18 epoch selection failed")
    return best_epoch, history


def fit_fixed(
    data: dict,
    rows: np.ndarray,
    *,
    epochs: int,
    device: torch.device,
    log_every: int = 5,
) -> tuple[torch.nn.Module, dict, np.ndarray, dict, list[dict]]:
    _set_seed(MODEL_SEED)
    (
        model,
        memory,
        sample_anchor,
        centered,
        origin,
        neighbours,
        neighbour_weights,
        device_mapping,
        stats,
    ) = _build_training_state(data, rows, device=device)
    loader = _loader(
        data,
        centered,
        sample_anchor,
        rows,
        device_mapping,
        seed=MODEL_SEED,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(V18_CONFIG["learning_rate"]),
        weight_decay=float(V18_CONFIG["weight_decay"]),
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
                    {"refit_epoch": epoch, "total_epochs": int(epochs), **metrics},
                    allow_nan=False,
                ),
                flush=True,
            )
    memory = dict(memory)
    memory["stats"] = stats
    return model, memory, origin, device_mapping, history


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
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "coord_origin": torch.as_tensor(origin, dtype=torch.float32),
        "stats": memory["stats"],
        "config": dict(V18_CONFIG),
        "device_id_mapping": {
            str(raw): int(mapped) for raw, mapped in device_mapping.items()
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
    data = _load_uji(registry["train_data"])
    fit_rows = _registered_rows(registry, "member_fit", data)
    stop_rows = _registered_rows(registry, "early_stop", data)
    refit_rows = _registered_rows(registry, "selection_refit", data)
    device = torch.device(device_name)
    started = time.perf_counter()
    epoch, selection_history = select_epoch(
        data, fit_rows, stop_rows, device=device, log_every=log_every
    )
    model, memory, origin, mapping, refit_history = fit_fixed(
        data,
        refit_rows,
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
        "sha256": _checkpoint_sha(destination),
        "selected_epoch": epoch,
        "anchors": int(model.n_anchors),
        "training_seconds": payload["training_seconds"],
    }


def load_v18_checkpoint(
    path: str | Path,
    *,
    registry: dict,
    device: torch.device,
    expected_phase: str,
) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18 checkpoint")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18 checkpoint registry mismatch")
    if payload.get("phase") != expected_phase:
        raise ValueError("V18 checkpoint phase mismatch")
    if payload.get("config") != V18_CONFIG:
        raise ValueError("V18 checkpoint configuration mismatch")

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
    model, _geometry, _neighbours, _weights = build_v18_model(
        memory=memory,
        n_aps=int(payload["stats"]["n_aps"]),
        n_buildings=int(state["anchor_buildings"].max().item()) + 1,
        n_devices=len(payload.get("device_id_mapping", {})),
        device=device,
        config=V18_CONFIG,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def evaluate_role(
    *,
    role: str,
    checkpoint_path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    if role not in {"development", "outer_test"}:
        raise ValueError("role must be development or outer_test")
    phase = "selection" if role == "development" else "final"
    if role == "outer_test":
        load_freeze(DEFAULT_FREEZE, registry=registry)
    device = torch.device(device_name)
    model, payload = load_v18_checkpoint(
        checkpoint_path,
        registry=registry,
        device=device,
        expected_phase=phase,
    )
    data = _load_uji(registry["train_data"])
    rows = _registered_rows(registry, role, data)
    origin = payload["coord_origin"].cpu().numpy().astype(np.float32)
    centered = torch.from_numpy(
        (np.asarray(data["raw_xy"], dtype=np.float32) - origin[None, :]).astype(
            np.float32
        )
    )
    return evaluate_model(model, data, centered, rows, device=device)


def evaluate_development_and_freeze(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    checkpoint_path: str | Path = DEFAULT_SELECTION_CHECKPOINT,
    result_path: str | Path = DEFAULT_DEVELOPMENT_RESULT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    if Path(result_path).exists() or Path(freeze_path).exists():
        raise FileExistsError("V18 development result/freeze already exists")
    metrics = evaluate_role(
        role="development",
        checkpoint_path=checkpoint_path,
        registry_path=registry_path,
        device_name=device_name,
    )
    _model, payload = load_v18_checkpoint(
        checkpoint_path,
        registry=registry,
        device=torch.device("cpu"),
        expected_phase="selection",
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "selection_checkpoint": str(Path(checkpoint_path).resolve()),
        "selection_checkpoint_sha256": _checkpoint_sha(checkpoint_path),
        "development_metrics": metrics,
        "decision_rule": (
            "V18 has no development hyperparameter search; development is a "
            "post-training diagnostic only. The model/configuration and epoch "
            "count were fixed without development outcomes."
        ),
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(result_path, result)
    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "selection_checkpoint_sha256": result["selection_checkpoint_sha256"],
        "development_result_sha256": result["result_sha256"],
        "frozen_selected_epoch": int(payload["selected_epoch"]),
        "frozen_model_config": dict(V18_CONFIG),
        "outer_test_access_authorized": True,
    }
    freeze["freeze_sha256"] = _canonical_json_sha256(freeze)
    _write_json_new(freeze_path, freeze)
    return {"development": result, "freeze": freeze}


def load_freeze(
    path: str | Path = DEFAULT_FREEZE,
    *,
    registry: dict,
) -> dict:
    source = Path(path)
    if not source.exists():
        raise PermissionError("V18 outer-test access is forbidden before freeze")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V18 freeze")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18 freeze registry mismatch")
    if payload.get("freeze_sha256") != _canonical_json_sha256(
        payload, omit="freeze_sha256"
    ):
        raise ValueError("V18 freeze self-hash failed")
    if payload.get("frozen_model_config") != V18_CONFIG:
        raise ValueError("V18 frozen configuration changed")
    if payload.get("outer_test_access_authorized") is not True:
        raise PermissionError("V18 outer-test access is not authorized")
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
    data = _load_uji(registry["train_data"])
    rows = _registered_rows(registry, "outer_refit", data)
    device = torch.device(device_name)
    started = time.perf_counter()
    model, memory, origin, mapping, history = fit_fixed(
        data,
        rows,
        epochs=int(freeze["frozen_selected_epoch"]),
        device=device,
        log_every=log_every,
    )
    payload = _checkpoint_payload(
        registry=registry,
        phase="final",
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
        "sha256": _checkpoint_sha(destination),
        "selected_epoch": int(freeze["frozen_selected_epoch"]),
        "anchors": int(model.n_anchors),
        "training_seconds": payload["training_seconds"],
    }


def evaluate_outer(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    checkpoint_path: str | Path = DEFAULT_FINAL_CHECKPOINT,
    output: str | Path = DEFAULT_OUTER_RESULT,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    freeze = load_freeze(DEFAULT_FREEZE, registry=registry)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    metrics = evaluate_role(
        role="outer_test",
        checkpoint_path=checkpoint_path,
        registry_path=registry_path,
        device_name=device_name,
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "final_checkpoint": str(Path(checkpoint_path).resolve()),
        "final_checkpoint_sha256": _checkpoint_sha(checkpoint_path),
        "outer_test_metrics": metrics,
        "single_member": True,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V18 single-member UJI experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register")
    register.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))

    train_sel = sub.add_parser("train-selection")
    train_sel.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train_sel.add_argument("--output", default=str(DEFAULT_SELECTION_CHECKPOINT))
    train_sel.add_argument("--device", default="cuda")
    train_sel.add_argument("--log-every", type=int, default=5)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    freeze.add_argument("--checkpoint", default=str(DEFAULT_SELECTION_CHECKPOINT))
    freeze.add_argument("--result", default=str(DEFAULT_DEVELOPMENT_RESULT))
    freeze.add_argument("--freeze", dest="freeze_path", default=str(DEFAULT_FREEZE))
    freeze.add_argument("--device", default="cuda")

    train_fin = sub.add_parser("train-final")
    train_fin.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train_fin.add_argument("--freeze", dest="freeze_path", default=str(DEFAULT_FREEZE))
    train_fin.add_argument("--output", default=str(DEFAULT_FINAL_CHECKPOINT))
    train_fin.add_argument("--device", default="cuda")
    train_fin.add_argument("--log-every", type=int, default=5)

    outer = sub.add_parser("outer")
    outer.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    outer.add_argument("--checkpoint", default=str(DEFAULT_FINAL_CHECKPOINT))
    outer.add_argument("--output", default=str(DEFAULT_OUTER_RESULT))
    outer.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.command == "register":
        payload = register_protocol(train_data=args.train_data, output=args.output)
    elif args.command == "train-selection":
        payload = train_selection(
            registry_path=args.registry,
            output=args.output,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "freeze":
        payload = evaluate_development_and_freeze(
            registry_path=args.registry,
            checkpoint_path=args.checkpoint,
            result_path=args.result,
            freeze_path=args.freeze_path,
            device_name=args.device,
        )
    elif args.command == "train-final":
        payload = train_final(
            registry_path=args.registry,
            freeze_path=args.freeze_path,
            output=args.output,
            device_name=args.device,
            log_every=args.log_every,
        )
    else:
        payload = evaluate_outer(
            registry_path=args.registry,
            checkpoint_path=args.checkpoint,
            output=args.output,
            device_name=args.device,
        )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
