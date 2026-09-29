"""Train one matched DR-JAM-v4.4 anchor member.

This executable trains only on UJI ``trainingData.csv`` under a frozen v4
registry.  It supports coordinate-development and device-development roles.
The historical official validation file is never accepted here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

# Must be set before the first CUDA linear-algebra context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from fhm_dapp_tim.fhm_dapp.anchor import (
    location_keys,
    raw_coords,
    resolve_path,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.device_robust_v4 import (
    DeviceRobustAnchorModel,
    anchor_support_3d,
    build_device_balanced_anchor_memory,
    device_group_sampling_weights,
    estimate_device_effect_covariance,
    spatial_energy_member_loss,
)
from fhm_dapp_tim.fhm_dapp.protocol_v4 import (
    DANN_GAMMAS,
    MEMBER_MAX_VISIBLE,
    MEMBER_SEEDS,
    MEMBER_VISIBLE_THRESHOLD,
    MECHANISM_MEMBER_SEEDS,
    MECHANISMS,
    PROTOCOL_VERSION,
    coordinate_roles_for_development_phones,
    coordinate_axis_partition,
    device_epoch_selection_roles,
    sha256_indices,
    sha256_location_keys,
    validate_registry,
)


FROZEN_MEMBER_V4 = {
    "model_schema": DeviceRobustAnchorModel.MODEL_SCHEMA,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "max_visible": MEMBER_MAX_VISIBLE,
    "visible_threshold": MEMBER_VISIBLE_THRESHOLD,
    "dropout": 0.10,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "maximum_epochs": 80,
    "minimum_epochs": 20,
    "early_stop_patience": 10,
    "minimum_energy_improvement_m": 1e-3,
    "validation_batch_size": 128,
    "floor_height_m": 4.0,
    "distance_normalizer_m": 1.0,
    "training_objective": "exact_full_dictionary_3d_energy_score",
    "sampler": "equal_phone_equal_coordinate_equal_row",
    "radio_memory_estimator": "equal_phone_within_coordinate_v4",
    "radio_score": "joint_hurdle_log_score_fixed_J_nats_per_ap",
    "cuda_attention_kernel": "math_only",
    "tf32": False,
    "deterministic_algorithms": "strict",
    "custom_rss_augmentation": False,
    "ap_dropout": 0.0,
}


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _configure_cuda_reproducibility() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_reference_stats(
    stats: dict,
    raw_xy: np.ndarray,
    building: np.ndarray,
    train_idx: np.ndarray,
) -> dict:
    result = copy.deepcopy(stats)
    building_stats: dict[int, dict[str, float]] = {}
    for building_id in sorted(np.unique(building).tolist()):
        points = raw_xy[train_idx][building[train_idx] == int(building_id)]
        if len(points) == 0:
            raise ValueError(f"training role has no rows for building {building_id}")
        building_stats[int(building_id)] = {
            "x_min": float(points[:, 0].min()),
            "x_max": float(points[:, 0].max()),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
        }
    result["building_stats"] = building_stats
    return result


def _member_training_spec(mechanism: str) -> dict:
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown registered mechanism: {mechanism}")
    if mechanism == "M3":
        raise ValueError("M3 reuses the paired M2 member; do not retrain it")
    spec = MECHANISMS[mechanism]
    gamma = float(spec.dann_gamma)
    if spec.member_variant == "dann" and gamma not in DANN_GAMMAS:
        raise ValueError("DANN gamma is outside the registered grid")
    return {
        "mechanism": mechanism,
        "member_variant": spec.member_variant,
        "scan_feature_mode": spec.scan_feature_mode,
        "dann_gamma": gamma,
        "stores_rank2_device_effect": spec.member_variant == "maximal_invariant",
    }


def _build_model_and_memory(
    *,
    rss: torch.Tensor,
    raw_xy: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    phone_ids: np.ndarray,
    coordinate_keys: list[tuple],
    source_stats: dict,
    train_idx: np.ndarray,
    mechanism: str,
    device: torch.device,
    max_visible: int | None = None,
) -> tuple[
    DeviceRobustAnchorModel,
    dict[str, object],
    torch.Tensor,
    np.ndarray,
    torch.Tensor,
    dict,
    np.ndarray,
    tuple[int, ...],
]:
    spec = _member_training_spec(mechanism)
    origin = raw_xy[train_idx].mean(axis=0).astype(np.float32)
    centered_xy = (raw_xy - origin).astype(np.float32)
    stats = _training_reference_stats(
        source_stats, raw_xy, building, train_idx
    )
    memory, sample_anchor = build_device_balanced_anchor_memory(
        rss,
        centered_xy,
        building,
        floor,
        phone_ids,
        train_idx,
        coordinate_keys=coordinate_keys,
    )
    if np.any(sample_anchor[train_idx] < 0):
        raise RuntimeError("some training rows are absent from the anchor dictionary")
    training_phones = tuple(sorted(map(int, np.unique(phone_ids[train_idx]))))
    phone_to_label = {phone: index for index, phone in enumerate(training_phones)}
    device_labels = torch.full((len(rss),), -1, dtype=torch.long)
    for row in train_idx:
        device_labels[int(row)] = phone_to_label[int(phone_ids[int(row)])]
    n_devices = len(training_phones) if spec["member_variant"] == "dann" else 0
    seen_ap_mask = (memory["prototype_rss"] > -0.999).any(dim=0)
    visible_capacity = (
        int(FROZEN_MEMBER_V4["max_visible"])
        if max_visible is None
        else int(max_visible)
    )
    if visible_capacity <= 0:
        raise ValueError("max_visible must be a positive integer")
    model = DeviceRobustAnchorModel(
        n_aps=int(stats["n_aps"]),
        anchor_coords=memory["coords"],
        anchor_buildings=memory["building"],
        anchor_floors=memory["floor"],
        seen_ap_mask=seen_ap_mask,
        n_devices=n_devices,
        d_model=FROZEN_MEMBER_V4["d_model"],
        num_heads=FROZEN_MEMBER_V4["num_heads"],
        num_layers=FROZEN_MEMBER_V4["num_layers"],
        max_visible=visible_capacity,
        dropout=FROZEN_MEMBER_V4["dropout"],
        visible_threshold=FROZEN_MEMBER_V4["visible_threshold"],
        scan_feature_mode=spec["scan_feature_mode"],
    ).to(device)
    model.initialize_anchor_keys(memory["prototype_rss"].to(device))
    return (
        model,
        memory,
        torch.from_numpy(centered_xy).float(),
        sample_anchor,
        device_labels,
        stats,
        origin,
        training_phones,
    )


def _training_loader(
    *,
    rss: torch.Tensor,
    sample_anchor: np.ndarray,
    device_labels: torch.Tensor,
    phone_ids: np.ndarray,
    coordinate_keys: list[tuple],
    train_idx: np.ndarray,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        rss[train_idx].float(),
        torch.from_numpy(sample_anchor[train_idx]).long(),
        device_labels[train_idx].long(),
    )
    weights = device_group_sampling_weights(
        phone_ids, coordinate_keys, train_idx
    )
    generator = torch.Generator().manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_idx),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=FROZEN_MEMBER_V4["batch_size"],
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _pair_distance(model: DeviceRobustAnchorModel) -> torch.Tensor:
    support = anchor_support_3d(
        model, floor_height_m=FROZEN_MEMBER_V4["floor_height_m"]
    )
    return torch.cdist(support, support, p=2)


def _train_one_epoch(
    *,
    model: DeviceRobustAnchorModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pair_distance_m: torch.Tensor,
    dann_gamma: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "spatial_energy_score_m": 0.0,
        "domain_loss": 0.0,
    }
    rows = 0
    for batch_rss, batch_anchor, batch_device in loader:
        batch_rss = batch_rss.to(device, non_blocking=True)
        batch_anchor = batch_anchor.to(device, non_blocking=True)
        batch_device = batch_device.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        values = spatial_energy_member_loss(
            model,
            batch_rss,
            batch_anchor,
            pair_distance_m=pair_distance_m,
            distance_normalizer_m=FROZEN_MEMBER_V4["distance_normalizer_m"],
            device_labels=batch_device if dann_gamma > 0.0 else None,
            domain_adversarial_weight=dann_gamma,
            floor_height_m=FROZEN_MEMBER_V4["floor_height_m"],
        )
        values["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), FROZEN_MEMBER_V4["gradient_clip"]
        )
        optimizer.step()
        count = len(batch_rss)
        rows += count
        for key in totals:
            totals[key] += float(values[key].detach().cpu()) * count
    if rows == 0:
        raise RuntimeError("training loader produced no rows")
    return {key: value / rows for key, value in totals.items()}


@torch.no_grad()
def evaluate_member_distribution(
    model: DeviceRobustAnchorModel,
    rss: torch.Tensor,
    centered_xy: torch.Tensor,
    building: np.ndarray,
    floor: np.ndarray,
    indices: np.ndarray,
    *,
    coordinate_keys: list[tuple],
    batch_size: int | None = None,
) -> dict[str, float]:
    """Score the member prior without a radio update or sparse projection."""
    if len(indices) == 0:
        raise ValueError("evaluation indices cannot be empty")
    model.eval()
    device = next(model.parameters()).device
    width = int(batch_size or FROZEN_MEMBER_V4["validation_batch_size"])
    anchor_support = anchor_support_3d(
        model, floor_height_m=FROZEN_MEMBER_V4["floor_height_m"]
    ).to(device)
    anchor_pair = torch.cdist(anchor_support, anchor_support, p=2)
    anchor_xy = model.anchor_coords.to(device)
    total_energy = 0.0
    total_error = 0.0
    correct_floor = 0
    correct_building = 0
    total_rows = 0
    row_energy = []
    row_error = []
    for start in range(0, len(indices), width):
        selected = indices[start : start + width]
        query = rss[selected].to(device)
        target_xy = centered_xy[selected].to(device)
        target_floor = torch.as_tensor(floor[selected], device=device)
        target_building = torch.as_tensor(building[selected], device=device)
        target_support = torch.cat(
            (
                target_xy,
                FROZEN_MEMBER_V4["floor_height_m"]
                * target_floor.to(target_xy.dtype).unsqueeze(1),
            ),
            dim=1,
        )
        logits, _embedding = model.retrieval_logits(query)
        probability = torch.softmax(logits, dim=1)
        first = torch.sum(
            probability * torch.cdist(target_support, anchor_support, p=2), dim=1
        )
        second = 0.5 * torch.sum(
            (probability @ anchor_pair) * probability, dim=1
        )
        prediction = probability @ anchor_xy
        error = torch.linalg.vector_norm(prediction - target_xy, dim=1)
        floor_probability = torch.zeros(
            (len(query), int(model.anchor_floors.max()) + 1), device=device
        )
        floor_probability.scatter_add_(
            1,
            model.anchor_floors.unsqueeze(0).expand(len(query), -1),
            probability,
        )
        building_probability = torch.zeros(
            (len(query), int(model.anchor_buildings.max()) + 1), device=device
        )
        building_probability.scatter_add_(
            1,
            model.anchor_buildings.unsqueeze(0).expand(len(query), -1),
            probability,
        )
        energy = first - second
        total_energy += float(energy.sum().cpu())
        total_error += float(error.sum().cpu())
        row_energy.append(energy.cpu())
        row_error.append(error.cpu())
        correct_floor += int((floor_probability.argmax(1) == target_floor).sum())
        correct_building += int(
            (building_probability.argmax(1) == target_building).sum()
        )
        total_rows += len(query)
    energy_values = torch.cat(row_energy)
    error_values = torch.cat(row_error)
    group_rows: dict[tuple, list[int]] = {}
    for local, row in enumerate(np.asarray(indices, dtype=np.int64)):
        group_rows.setdefault(coordinate_keys[int(row)], []).append(local)
    group_energy = torch.stack(
        [energy_values[torch.as_tensor(rows)].mean() for rows in group_rows.values()]
    )
    group_error = torch.stack(
        [error_values[torch.as_tensor(rows)].mean() for rows in group_rows.values()]
    )
    return {
        "energy_score_3d_m": total_energy / total_rows,
        "mae_2d_m": total_error / total_rows,
        "coordinate_group_macro_energy_score_3d_m": float(group_energy.mean()),
        "coordinate_group_macro_mae_2d_m": float(group_error.mean()),
        "coordinate_groups": len(group_rows),
        "floor_accuracy": correct_floor / total_rows,
        "building_accuracy": correct_building / total_rows,
        "rows": int(total_rows),
    }


def _fit_schedule(
    *,
    rss: torch.Tensor,
    raw_xy: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    phone_ids: np.ndarray,
    coordinate_keys: list[tuple],
    source_stats: dict,
    fit_idx: np.ndarray,
    stop_idx: np.ndarray,
    mechanism: str,
    seed: int,
    device: torch.device,
    log_every: int,
    max_visible: int | None = None,
) -> tuple[int, list[dict]]:
    _set_seed(seed)
    (
        model,
        _memory,
        centered,
        sample_anchor,
        device_labels,
        _stats,
        _origin,
        _training_phones,
    ) = _build_model_and_memory(
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        source_stats=source_stats,
        train_idx=fit_idx,
        mechanism=mechanism,
        device=device,
        max_visible=max_visible,
    )
    loader = _training_loader(
        rss=rss,
        sample_anchor=sample_anchor,
        device_labels=device_labels,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        train_idx=fit_idx,
        seed=seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_MEMBER_V4["learning_rate"],
        weight_decay=FROZEN_MEMBER_V4["weight_decay"],
    )
    pair = _pair_distance(model)
    gamma = float(_member_training_spec(mechanism)["dann_gamma"])
    best_energy = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, FROZEN_MEMBER_V4["maximum_epochs"] + 1):
        fit_metrics = _train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            pair_distance_m=pair,
            dann_gamma=gamma,
            device=device,
        )
        stop_metrics = evaluate_member_distribution(
            model,
            rss,
            centered,
            building,
            floor,
            stop_idx,
            coordinate_keys=coordinate_keys,
        )
        improved = stop_metrics["coordinate_group_macro_energy_score_3d_m"] < (
            best_energy - FROZEN_MEMBER_V4["minimum_energy_improvement_m"]
        )
        if improved:
            best_energy = float(
                stop_metrics["coordinate_group_macro_energy_score_3d_m"]
            )
            best_epoch = epoch
            stale = 0
        elif epoch >= FROZEN_MEMBER_V4["minimum_epochs"]:
            stale += 1
        record = {
            "epoch": epoch,
            "fit": fit_metrics,
            "early_stop": stop_metrics,
            "best_epoch": best_epoch,
            "best_energy_score_3d_m": best_energy,
        }
        history.append(record)
        if epoch == 1 or improved or epoch % int(log_every) == 0:
            print(json.dumps(record, allow_nan=False), flush=True)
        if (
            epoch >= FROZEN_MEMBER_V4["minimum_epochs"]
            and stale >= FROZEN_MEMBER_V4["early_stop_patience"]
        ):
            break
    if best_epoch <= 0:
        raise RuntimeError("epoch selection found no finite energy score")
    return best_epoch, history


def _fit_fixed(
    *,
    rss: torch.Tensor,
    raw_xy: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    phone_ids: np.ndarray,
    coordinate_keys: list[tuple],
    source_stats: dict,
    train_idx: np.ndarray,
    mechanism: str,
    seed: int,
    epochs: int,
    device: torch.device,
    log_every: int,
    max_visible: int | None = None,
) -> tuple[DeviceRobustAnchorModel, dict, torch.Tensor, np.ndarray, dict, np.ndarray, tuple[int, ...], list[dict]]:
    if int(epochs) <= 0:
        raise ValueError("fixed epochs must be positive")
    _set_seed(seed)
    (
        model,
        memory,
        centered,
        sample_anchor,
        device_labels,
        stats,
        origin,
        training_phones,
    ) = _build_model_and_memory(
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        source_stats=source_stats,
        train_idx=train_idx,
        mechanism=mechanism,
        device=device,
        max_visible=max_visible,
    )
    loader = _training_loader(
        rss=rss,
        sample_anchor=sample_anchor,
        device_labels=device_labels,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        train_idx=train_idx,
        seed=seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_MEMBER_V4["learning_rate"],
        weight_decay=FROZEN_MEMBER_V4["weight_decay"],
    )
    pair = _pair_distance(model)
    gamma = float(_member_training_spec(mechanism)["dann_gamma"])
    history = []
    for epoch in range(1, int(epochs) + 1):
        metrics = _train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            pair_distance_m=pair,
            dann_gamma=gamma,
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
    return (
        model,
        memory,
        centered,
        sample_anchor,
        stats,
        origin,
        training_phones,
        history,
    )


def _roles_from_registry(
    *,
    role_axis: str,
    outer_fold: int | None,
    held_out_phone: int | None,
    keys: list[tuple],
    phone_ids: np.ndarray,
    registry: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    development = tuple(map(int, registry["device_development_phone_ids"]))
    if role_axis == "coordinate":
        if outer_fold is None:
            raise ValueError("coordinate role requires --outer-fold")
        roles = coordinate_roles_for_development_phones(
            keys, phone_ids, development, int(outer_fold)
        )
        registered = registry["coordinate_development_audit"][int(outer_fold)]
        for name, indices in roles.items():
            expected = registered["roles"][name]["row_index_sha256"]
            if sha256_indices(indices) != expected:
                raise ValueError(f"coordinate role changed after registration: {name}")
        refit = np.sort(
            np.concatenate((roles["member_fit"], roles["early_stop"]))
        ).astype(np.int64)
        return roles, {
            "fit": roles["member_fit"],
            "stop": roles["early_stop"],
            "refit": refit,
            "selection_evaluation": roles["structure_development"],
            "forbidden_evaluation": roles["outer_test"],
            "fold_identifier": int(outer_fold),
        }
    if role_axis == "device-development":
        if held_out_phone is None:
            raise ValueError("device-development requires --held-out-phone")
        held_out = int(held_out_phone)
        if held_out not in development:
            raise ValueError("held-out phone is not in the development set")
        training_phones = tuple(phone for phone in development if phone != held_out)
        coordinate_partition = coordinate_axis_partition(
            keys, phone_ids, development
        )
        roles = device_epoch_selection_roles(
            keys,
            phone_ids,
            held_out,
            training_phones=training_phones,
            allowed_coordinate_keys=coordinate_partition["development_core"],
        )
        registered_rows = {
            int(row["held_out_phone"]): row
            for row in registry["device_development_audit"]
        }[held_out]
        for name, indices in roles.items():
            expected = registered_rows["roles"][name]["row_index_sha256"]
            if sha256_indices(indices) != expected:
                raise ValueError(f"device role changed after registration: {name}")
        forbidden = np.sort(
            np.concatenate(
                (
                    roles["quarantined"],
                    roles["coordinate_confirmation_quarantine"],
                )
            )
        ).astype(np.int64)
        return roles, {
            "fit": roles["member_fit"],
            "stop": roles["early_stop"],
            "refit": roles["refit_train"],
            "selection_evaluation": roles["matched_test"],
            "forbidden_evaluation": forbidden,
            "fold_identifier": held_out,
        }
    raise ValueError("role_axis must be coordinate or device-development")


def train_member(args: argparse.Namespace) -> dict:
    torch.set_num_threads(max(1, min(int(args.threads), torch.get_num_threads())))
    if torch.cuda.is_available():
        _configure_cuda_reproducibility()
        torch.set_float32_matmul_precision("highest")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    seed = int(args.seed)
    if seed not in MEMBER_SEEDS:
        raise ValueError("seed is outside the registered v4 member set")
    spec = _member_training_spec(args.mechanism)
    data_path = resolve_path(args.train_data).resolve()
    registry = validate_registry(args.registry, train_data=data_path)
    output = Path(args.output)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v4 checkpoint: {output}")

    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        timestamp_t,
        source_stats,
    ) = load_dataset(data_path, dataset_type="UJI")
    raw_xy = raw_coords(coords_norm, building_t, source_stats)
    building = building_t.numpy()
    floor = floor_t.numpy()
    phone_ids = phone_t.numpy()
    keys = location_keys(raw_xy, building, floor)
    roles, training_roles = _roles_from_registry(
        role_axis=args.role_axis,
        outer_fold=args.outer_fold,
        held_out_phone=args.held_out_phone,
        keys=keys,
        phone_ids=phone_ids,
        registry=registry,
    )
    started = time.perf_counter()
    pilot_epochs = int(args.engineering_pilot_epochs or 0)
    if pilot_epochs > 0:
        selected_epoch = pilot_epochs
        schedule_history: list[dict] = []
        # A pilot skips epoch search but still uses the same refit population
        # as a formal member.  Otherwise device matched-test identities can
        # refer to anchors that occur only in the registered early-stop part.
        refit_idx = training_roles["refit"]
        selection_authority = False
        artifact_status = "engineering_pilot_no_selection_authority"
    else:
        selected_epoch, schedule_history = _fit_schedule(
            rss=rss,
            raw_xy=raw_xy,
            building=building,
            floor=floor,
            phone_ids=phone_ids,
            coordinate_keys=keys,
            source_stats=source_stats,
            fit_idx=training_roles["fit"],
            stop_idx=training_roles["stop"],
            mechanism=args.mechanism,
            seed=seed,
            device=device,
            log_every=int(args.log_every),
        )
        refit_idx = training_roles["refit"]
        selection_authority = True
        artifact_status = "registered_development_member"

    (
        model,
        memory,
        centered,
        sample_anchor,
        stats,
        origin,
        training_phones,
        refit_history,
    ) = _fit_fixed(
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor,
        phone_ids=phone_ids,
        coordinate_keys=keys,
        source_stats=source_stats,
        train_idx=refit_idx,
        mechanism=args.mechanism,
        seed=seed,
        epochs=selected_epoch,
        device=device,
        log_every=int(args.log_every),
    )
    device_effect = None
    if spec["stores_rank2_device_effect"]:
        estimate = estimate_device_effect_covariance(
            rss, sample_anchor, phone_ids, refit_idx
        )
        device_effect = {
            "covariance": estimate.covariance.cpu(),
            "phone_ids": list(estimate.phone_ids),
            "effects": estimate.effects.cpu(),
            "observation_counts": list(estimate.observation_counts),
            "shared_anchor_counts": list(estimate.shared_anchor_counts),
            "estimation_indices_sha256": sha256_indices(refit_idx),
        }

    config = {
        **FROZEN_MEMBER_V4,
        **spec,
        "training_protocol_version": PROTOCOL_VERSION,
        "role_axis": args.role_axis,
        "outer_fold": args.outer_fold,
        "held_out_phone": args.held_out_phone,
        "n_training_devices": (
            len(training_phones) if spec["member_variant"] == "dann" else 0
        ),
        "selected_epoch": int(selected_epoch),
        "seed_role": (
            "mechanism_and_structure"
            if seed in MECHANISM_MEMBER_SEEDS
            else "post_mechanism_structure_expansion_only"
        ),
    }
    checkpoint = {
        "artifact_status": artifact_status,
        "selection_authority": selection_authority,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "radio_memory": {
            key: value.detach().cpu()
            for key, value in memory.items()
            if key
            in {
                "detection_prob",
                "rss_center",
                "rss_scale",
                "counts",
                "device_counts",
            }
        },
        "device_effect": device_effect,
        "config": config,
        "stats": stats,
        "coord_origin": torch.as_tensor(origin, dtype=torch.float32),
        "anchor_coordinate_keys": [
            list(key) for key in memory["coordinate_keys"]
        ],
        "anchor_coordinate_keys_sha256": sha256_location_keys(
            memory["coordinate_keys"]
        ),
        "seed": seed,
        "training_phone_ids": list(training_phones),
        "training_indices": torch.as_tensor(refit_idx, dtype=torch.long),
        "training_indices_sha256": sha256_indices(refit_idx),
        "role_indices": {
            name: torch.as_tensor(indices, dtype=torch.long)
            for name, indices in roles.items()
        },
        "role_indices_sha256": {
            name: sha256_indices(indices) for name, indices in roles.items()
        },
        "schedule_history": schedule_history,
        "refit_history": refit_history,
        "selection_metrics": None,
        "forbidden_metrics": None,
        "source_sha256": _sha256_file(data_path),
        "registry_path": str(Path(args.registry).resolve()),
        "registry_sha256": registry["registry_sha256"],
        "timestamp_dtype": str(timestamp_t.numpy().dtype),
        "training_seconds": float(time.perf_counter() - started),
        "centered_coordinate_shape": list(centered.shape),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    payload = {
        "output": str(output),
        "artifact_status": artifact_status,
        "mechanism": args.mechanism,
        "role_axis": args.role_axis,
        "fold_identifier": training_roles["fold_identifier"],
        "seed": seed,
        "device": str(device),
        "anchors": model.n_anchors,
        "selected_epoch": int(selected_epoch),
        "training_rows": int(len(refit_idx)),
        "training_seconds": checkpoint["training_seconds"],
        "selection_evaluated": False,
        "forbidden_evaluated": False,
    }
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one DR-JAM-v4 member")
    parser.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv"
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--role-axis", choices=("coordinate", "device-development"), required=True
    )
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--held-out-phone", type=int, default=None)
    parser.add_argument(
        "--mechanism",
        choices=tuple(name for name in MECHANISMS if name != "M3"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--engineering-pilot-epochs",
        type=int,
        default=None,
        help="fixed short run; artifact is irrevocably marked no-selection-authority",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_member(args)


if __name__ == "__main__":
    main()
