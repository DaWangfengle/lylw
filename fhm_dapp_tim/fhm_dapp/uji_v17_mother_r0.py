"""Execute the registered V17-R0 AC-RHAP mother reconstruction."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _rows_for_keys,
    _save_torch_new,
    _sha256_file,
    _sha256_indices,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.model import AnchorStudentTModel
from fhm_dapp_tim.fhm_dapp.projected_ensemble import ProjectedEnsemble
from fhm_dapp_tim.fhm_dapp.train_anchor import (
    _ap_building_owner,
    _neighbor_targets,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import (
    FROZEN_TRAINING,
    _remap_training_devices,
    _set_seed,
    _training_reference_stats,
)
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_protocol import (
    DEFAULT_DEVELOPMENT_CSV,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    DEFAULT_SELECTION_CSV,
    MEMBER_SEEDS,
    PROTOCOL_VERSION,
    R0_BOOTSTRAP_REPETITIONS,
    R0_BOOTSTRAP_SEED,
    R0_MEMBER_COUNT,
    checkpoint_path,
    load_registry,
    registered_role_keys,
)


EVALUATION_BATCH_SIZE = 4096
MAXIMUM_EPOCHS = 80
PATIENCE = 10
BOOTSTRAP_REPETITIONS = R0_BOOTSTRAP_REPETITIONS
BOOTSTRAP_SEED = R0_BOOTSTRAP_SEED


def _registered_rows(registry: dict, role: str, data: dict) -> np.ndarray:
    rows = _rows_for_keys(data["keys"], registered_role_keys(registry, role))
    if _sha256_indices(rows) != registry["roles"][role]["row_index_sha256"]:
        raise ValueError(f"row hash failed for V17 role {role}")
    return rows


def _validate_output(registry: dict, kind: str, path: Path) -> None:
    allowed = registry["allowed_outputs"]
    if kind == "checkpoint":
        expected = {Path(value).resolve() for value in allowed["r0_checkpoints"]}
        if path.resolve() not in expected:
            raise ValueError("checkpoint path is outside the V17 registry")
    else:
        field = {
            "selection_refit_csv": "selection_refit_csv",
            "development_csv": "development_csv",
            "development_result": "development_result",
        }[kind]
        if path.resolve() != Path(allowed[field]).resolve():
            raise ValueError(f"{kind} path is outside the V17 registry")
    if path.exists():
        raise FileExistsError(path)


def _materialize_csv(
    source: Path,
    destination: Path,
    rows: np.ndarray,
) -> None:
    """Copy registered CSV rows byte-for-byte in original row order."""

    if destination.exists():
        raise FileExistsError(destination)
    lines = source.read_bytes().splitlines(keepends=True)
    if len(lines) < 2 or len(lines) - 1 <= int(rows.max(initial=-1)):
        raise ValueError("source CSV line count does not match registered rows")
    ordered = np.asarray(rows, dtype=np.int64)
    if ordered.ndim != 1 or len(np.unique(ordered)) != len(ordered):
        raise ValueError("registered CSV rows must be unique and one-dimensional")
    if len(ordered) > 1 and bool(np.any(ordered[1:] <= ordered[:-1])):
        raise ValueError("registered CSV rows must retain source-file order")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(lines[0])
        for index in ordered:
            handle.write(lines[int(index) + 1])


def materialize_roles(
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict:
    registry = load_registry(registry_path)
    data = load_uji_v16(registry["train_data"])
    outputs = {
        "selection_refit": Path(registry["allowed_outputs"]["selection_refit_csv"]),
        "development": Path(registry["allowed_outputs"]["development_csv"]),
    }
    for role, destination in outputs.items():
        _validate_output(registry, f"{role}_csv", destination)
    records = {}
    for role, destination in outputs.items():
        rows = _registered_rows(registry, role, data)
        _materialize_csv(Path(registry["train_data"]), destination, rows)
        derived = load_uji_v16(destination)
        expected_keys = registered_role_keys(registry, role)
        if set(derived["keys"]) != expected_keys:
            raise RuntimeError(f"materialized V17 {role} keys changed")
        if len(derived["keys"]) != len(rows):
            raise RuntimeError(f"materialized V17 {role} row count changed")
        records[role] = {
            "path": str(destination.resolve()),
            "sha256": _sha256_file(destination),
            "rows": int(len(rows)),
            "coordinates": int(len(expected_keys)),
            "source_row_index_sha256": _sha256_indices(rows),
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "materialized": records,
    }


def _build_member(
    data: dict,
    rows: np.ndarray,
    *,
    device: torch.device,
) -> tuple[
    AnchorStudentTModel,
    dict,
    np.ndarray,
    torch.Tensor,
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    dict,
    dict[int, int],
]:
    raw_xy = np.asarray(data["raw_xy"], dtype=np.float64)
    origin = raw_xy[rows].mean(axis=0).astype(np.float32)
    centered = (raw_xy - origin[None, :]).astype(np.float32)
    memory, sample_anchor = build_anchor_memory(
        data["rss"],
        centered,
        data["building"],
        data["floor"],
        rows,
    )
    if np.any(sample_anchor[rows] < 0):
        raise RuntimeError("V17 training row escaped the anchor dictionary")
    device_ids, device_mapping = _remap_training_devices(data["phone"], rows)
    seen = (memory["prototype_rss"] > -0.999).any(dim=0)
    owner = _ap_building_owner(
        memory["prototype_rss"],
        memory["building"],
        int(data["stats"]["n_buildings"]),
    )
    model = AnchorStudentTModel(
        n_aps=int(data["stats"]["n_aps"]),
        anchor_coords=memory["coords"],
        anchor_buildings=memory["building"],
        anchor_floors=memory["floor"],
        seen_ap_mask=seen,
        prototype_rss=memory["prototype_rss"],
        ap_building_owner=owner,
        n_devices=len(device_mapping),
        d_model=FROZEN_TRAINING["d_model"],
        num_heads=FROZEN_TRAINING["num_heads"],
        num_layers=FROZEN_TRAINING["num_layers"],
        max_visible=FROZEN_TRAINING["max_visible"],
        dropout=FROZEN_TRAINING["dropout"],
        student_df=FROZEN_TRAINING["student_df"],
        max_residual_m=FROZEN_TRAINING["max_residual_m"],
        residual_experts=FROZEN_TRAINING["residual_experts"],
        scan_feature_mode=FROZEN_TRAINING["scan_feature_mode"],
        prototype_metric=FROZEN_TRAINING["prototype_metric"],
        candidate_reranker=FROZEN_TRAINING["candidate_reranker"],
    ).to(device)
    model.initialize_anchor_keys(memory["prototype_rss"].to(device))
    neighbours, neighbour_weights = _neighbor_targets(
        memory["coords"],
        memory["building"],
        neighbor_count=FROZEN_TRAINING["target_neighbors"],
        temperature_m=FROZEN_TRAINING["spatial_temperature_m"],
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
        neighbours.to(device),
        neighbour_weights.to(device),
        stats,
        device_mapping,
    )


def _loader(
    data: dict,
    centered: torch.Tensor,
    sample_anchor: np.ndarray,
    rows: np.ndarray,
    device_ids: torch.Tensor,
    *,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        data["rss"][rows].float(),
        centered[rows],
        torch.from_numpy(sample_anchor[rows]).long(),
        device_ids,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=FROZEN_TRAINING["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def _train_epoch(
    model: AnchorStudentTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    neighbours: torch.Tensor,
    neighbour_weights: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, list[float]] = {}
    for batch_rss, batch_xy, batch_anchor, batch_device in loader:
        batch_rss = batch_rss.to(device)
        batch_xy = batch_xy.to(device)
        batch_anchor = batch_anchor.to(device)
        batch_device = batch_device.to(device)
        optimizer.zero_grad(set_to_none=True)
        terms = model.training_loss(
            batch_rss,
            batch_xy,
            batch_anchor,
            neighbours,
            neighbour_weights,
            retrieval_weight=FROZEN_TRAINING["retrieval_weight"],
            hierarchy_weight=FROZEN_TRAINING["hierarchy_weight"],
            self_target_weight=FROZEN_TRAINING["self_target_weight"],
            coordinate_weight=FROZEN_TRAINING["coordinate_weight"],
            device_ids=batch_device,
            domain_adversarial_weight=FROZEN_TRAINING[
                "domain_adversarial_weight"
            ],
            rss_offset_aug_db=FROZEN_TRAINING["rss_offset_aug_db"],
            rss_scale_aug=FROZEN_TRAINING["rss_scale_aug"],
            train_candidates=FROZEN_TRAINING["train_candidates"],
            ap_dropout=FROZEN_TRAINING["ap_dropout"],
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), FROZEN_TRAINING["gradient_clip"]
        )
        optimizer.step()
        for name, value in terms.items():
            totals.setdefault(name, []).append(float(value.detach().cpu()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


@torch.no_grad()
def _evaluate_member(
    model: AnchorStudentTModel,
    data: dict,
    centered: torch.Tensor,
    rows: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions = []
    for start in range(0, len(rows), 512):
        block_rows = rows[start : start + 512]
        result = model.predict(
            data["rss"][block_rows].to(device),
            top_k=FROZEN_TRAINING["inference_top_k"],
            neural_scale=FROZEN_TRAINING["inference_neural_scale"],
            prototype_distance_weight=FROZEN_TRAINING[
                "inference_prototype_weight"
            ],
            posterior_temperature=FROZEN_TRAINING["posterior_temperature"],
        )
        predictions.append(result["coords"].detach().cpu())
    prediction = torch.cat(predictions)
    target = centered[rows]
    error = torch.linalg.vector_norm(prediction - target, dim=1).numpy()
    grouped: dict[tuple, list[float]] = {}
    for index, value in zip(rows, error):
        grouped.setdefault(data["keys"][int(index)], []).append(float(value))
    group_error = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    return {
        "coordinate_macro_mae_m": float(group_error.mean()),
        "row_mae_m": float(error.mean()),
        "median_error_m": float(np.median(error)),
        "p90_error_m": float(np.quantile(error, 0.90)),
        "coordinates": int(len(group_error)),
        "rows": int(len(error)),
    }


def _fit_schedule(
    data: dict,
    fit_rows: np.ndarray,
    stop_rows: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    log_every: int,
) -> tuple[int, list[dict]]:
    _set_seed(seed)
    (
        model,
        _memory,
        sample_anchor,
        centered,
        _origin,
        neighbours,
        neighbour_weights,
        _stats,
        device_mapping,
    ) = _build_member(data, fit_rows, device=device)
    device_ids = torch.tensor(
        [device_mapping[int(data["phone"][index])] for index in fit_rows],
        dtype=torch.long,
    )
    loader = _loader(
        data, centered, sample_anchor, fit_rows, device_ids, seed=seed
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_TRAINING["learning_rate"],
        weight_decay=FROZEN_TRAINING["weight_decay"],
    )
    best = math.inf
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        train = _train_epoch(
            model,
            loader,
            optimizer,
            neighbours,
            neighbour_weights,
            device=device,
        )
        stop = _evaluate_member(
            model, data, centered, stop_rows, device=device
        )
        score = float(stop["coordinate_macro_mae_m"])
        improved = score < best - FROZEN_TRAINING["minimum_improvement_m"]
        if improved:
            best = score
            best_epoch = epoch
            stale = 0
        elif epoch >= FROZEN_TRAINING["minimum_epochs"]:
            stale += 1
        record = {
            "epoch": epoch,
            "train": train,
            "early_stop": stop,
            "best_epoch": best_epoch,
            "best_coordinate_macro_mae_m": best,
        }
        history.append(record)
        if epoch == 1 or improved or epoch % int(log_every) == 0:
            print(json.dumps(record, allow_nan=False), flush=True)
        if (
            epoch >= FROZEN_TRAINING["minimum_epochs"]
            and stale >= PATIENCE
        ):
            break
    if best_epoch <= 0:
        raise RuntimeError("V17 R0 epoch selection failed")
    return best_epoch, history


def _fit_refit(
    data: dict,
    rows: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    log_every: int,
) -> tuple[AnchorStudentTModel, dict, np.ndarray, dict[int, int], list[dict]]:
    _set_seed(seed)
    (
        model,
        _memory,
        sample_anchor,
        centered,
        origin,
        neighbours,
        neighbour_weights,
        stats,
        device_mapping,
    ) = _build_member(data, rows, device=device)
    device_ids = torch.tensor(
        [device_mapping[int(data["phone"][index])] for index in rows],
        dtype=torch.long,
    )
    loader = _loader(data, centered, sample_anchor, rows, device_ids, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_TRAINING["learning_rate"],
        weight_decay=FROZEN_TRAINING["weight_decay"],
    )
    history = []
    for epoch in range(1, int(epochs) + 1):
        train = _train_epoch(
            model,
            loader,
            optimizer,
            neighbours,
            neighbour_weights,
            device=device,
        )
        history.append({"epoch": epoch, **train})
        if epoch == 1 or epoch == int(epochs) or epoch % int(log_every) == 0:
            print(json.dumps({
                "refit_epoch": epoch,
                "refit_total_epochs": int(epochs),
                **train,
            }, allow_nan=False), flush=True)
    return model, stats, origin, device_mapping, history


def train_member(
    *,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    index = int(member_index)
    if not 1 <= index <= R0_MEMBER_COUNT:
        raise ValueError("R0 trains only the registered five-member prefix")
    destination = Path(registry["allowed_outputs"]["r0_checkpoints"][index - 1])
    if destination.resolve() != checkpoint_path(index).resolve():
        raise ValueError("V17 checkpoint naming contract changed")
    _validate_output(registry, "checkpoint", destination)
    data = load_uji_v16(registry["train_data"])
    fit_rows = _registered_rows(registry, "member_fit", data)
    stop_rows = _registered_rows(registry, "early_stop", data)
    refit_rows = _registered_rows(registry, "selection_refit", data)
    device = torch.device(device_name)
    seed = MEMBER_SEEDS[index - 1]
    started = time.perf_counter()
    selected_epoch, schedule = _fit_schedule(
        data,
        fit_rows,
        stop_rows,
        seed=seed,
        device=device,
        log_every=log_every,
    )
    model, stats, origin, device_mapping, refit_history = _fit_refit(
        data,
        refit_rows,
        seed=seed,
        epochs=selected_epoch,
        device=device,
        log_every=log_every,
    )
    config = copy.deepcopy(FROZEN_TRAINING)
    config.update({
        "training_protocol_version": PROTOCOL_VERSION,
        "model_schema": "v17_r0_historical_acrhap_member",
        "n_devices": len(device_mapping),
        # ProjectedEnsemble receives the materialized selection-refit CSV;
        # every row in that registered derivative is a radio-memory row.
        "train_all": True,
        "split_strategy": "registered_selection_refit_v17",
        "validation_fraction": 0.0,
        "calibration_fraction": 0.0,
        "epochs": int(selected_epoch),
        "patience": PATIENCE,
        "lr": FROZEN_TRAINING["learning_rate"],
        "target_neighbors": FROZEN_TRAINING["target_neighbors"],
        "early_stop_aggregation": "coordinate_macro_mae",
    })
    role_keys = tuple(sorted(registered_role_keys(registry, "selection_refit")))
    if len(role_keys) != model.n_anchors:
        raise RuntimeError("V17 refit anchors do not equal selection coordinates")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "phase": "r0_selection_refit",
        "member_index": index,
        "member_seed": seed,
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "config": config,
        "stats": stats,
        "coord_origin": np.asarray(origin, dtype=np.float32),
        "conformal_quantiles": {},
        "epoch": int(selected_epoch),
        "val_mae": float(schedule[selected_epoch - 1]["early_stop"]["row_mae_m"]),
        "seed": seed,
        "development_metrics": None,
        "confirmation_metrics": None,
        "split_contract": {
            "member_fit": registry["roles"]["member_fit"],
            "early_stop": registry["roles"]["early_stop"],
            "selection_refit": registry["roles"]["selection_refit"],
            "development_labels_consumed": False,
            "outer_labels_consumed": False,
            "official_labels_consumed": False,
        },
        "device_id_mapping": {
            str(raw): mapped for raw, mapped in device_mapping.items()
        },
        "anchor_coordinate_keys": role_keys,
        "schedule_history": schedule,
        "refit_history": refit_history,
        "training_seconds": float(time.perf_counter() - started),
    }
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": _sha256_file(destination),
        "member_index": index,
        "seed": seed,
        "selected_epoch": int(selected_epoch),
        "anchors": model.n_anchors,
        "training_seconds": payload["training_seconds"],
    }


def _coordinate_group_vector(
    errors: Sequence[float], keys: Sequence[tuple]
) -> tuple[np.ndarray, list[tuple]]:
    if len(errors) != len(keys):
        raise ValueError("errors and coordinate keys differ in length")
    grouped: dict[tuple, list[float]] = {}
    for key, error in zip(keys, errors):
        grouped.setdefault(tuple(key), []).append(float(error))
    ordered = sorted(grouped)
    return (
        np.asarray([np.mean(grouped[key]) for key in ordered], dtype=np.float64),
        ordered,
    )


def _paired_group_bootstrap(
    projected: np.ndarray,
    prior: np.ndarray,
) -> dict:
    if projected.shape != prior.shape or projected.ndim != 1:
        raise ValueError("paired group vectors must have equal one-dimensional shape")
    difference = projected - prior
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(difference)
    values = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPETITIONS, 500):
        stop = min(start + 500, BOOTSTRAP_REPETITIONS)
        draw = rng.integers(0, n, size=(stop - start, n))
        values[start:stop] = difference[draw].mean(axis=1)
    return {
        "mean_difference_m": float(difference.mean()),
        "ci95_m": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "probability_of_improvement": float(np.mean(values < 0.0)),
        "coordinates": int(n),
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
    }


def evaluate_r0(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    selection_csv = Path(registry["allowed_outputs"]["selection_refit_csv"])
    development_csv = Path(registry["allowed_outputs"]["development_csv"])
    destination = Path(registry["allowed_outputs"]["development_result"])
    _validate_output(registry, "development_result", destination)
    for role_path in (selection_csv, development_csv):
        if not role_path.exists():
            raise FileNotFoundError(role_path)
    checkpoints = [
        Path(value) for value in registry["allowed_outputs"]["r0_checkpoints"]
    ]
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V17 R0 checkpoint is missing: {missing}")

    ensemble = ProjectedEnsemble(
        checkpoints=checkpoints,
        anchor_data=selection_csv,
        device=device_name,
        dataset="UJI",
    )
    result = ensemble.evaluate(
        test_data=development_csv,
        test_partition="all",
        batch=EVALUATION_BATCH_SIZE,
        top_k=24,
        member_top_k=24,
        neural_scale=0.75,
        prototype_weight=8.0,
        posterior_temperature=0.75,
        candidate_building_conditioning=False,
        prior_weight=0.50,
        prior_alpha=0.25,
        style_offset_max=0.20,
        style_offset_weight=0.10,
        style_scale_max_log=0.25,
        style_scale_weight=0.10,
        reliability_power=0.0,
        member_weight_mode="inverse_scale",
        measurement_pool_mode="shared",
        mixture_visible_factor=1.0,
        mixture_invisible_factor=1.0,
        hurdle_distribution="student_t",
        hurdle_df=4.0,
        hurdle_score="balanced_log",
        projection_mode="alpha_prox_hurdle",
        steps=10,
        learning_rate=0.20,
        include_diagnostics=False,
        compute_predictive_distribution=False,
        variant_name="v17_r0_historical_acrhap_mother",
        consensus_prior_mode="power",
        consensus_prior_epsilon=1e-6,
        consensus_prior_alpha=0.25,
    )
    development = load_uji_v16(development_csv)
    prior_group, group_keys = _coordinate_group_vector(
        result["errors_m"]["baseline_consensus"], development["keys"]
    )
    projected_group, projected_keys = _coordinate_group_vector(
        result["errors_m"]["projected_consensus"], development["keys"]
    )
    if projected_keys != group_keys:
        raise RuntimeError("V17 paired coordinate group order changed")
    paired = _paired_group_bootstrap(projected_group, prior_group)
    passed = bool(projected_group.mean() <= prior_group.mean())
    wrapper = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_r0_development_complete",
        "registry": str(Path(registry_path).resolve()),
        "registry_sha256": registry["registry_sha256"],
        "scientific_status": registry["scientific_status"],
        "query_role": "development",
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
        "checkpoint_sha256": [
            _sha256_file(path) for path in checkpoints
        ],
        "selection_refit_csv_sha256": _sha256_file(selection_csv),
        "development_csv_sha256": _sha256_file(development_csv),
        "coordinate_macro": {
            "coordinates": int(len(group_keys)),
            "consensus_prior_mae_m": float(prior_group.mean()),
            "projected_acrhap_mae_m": float(projected_group.mean()),
            "paired_projected_minus_prior": paired,
        },
        "row_metrics": {
            "consensus_prior": result["baseline_consensus"],
            "projected_acrhap": result["projected_consensus"],
            "paired_row_bootstrap": result["paired_bootstrap_consensus"],
        },
        "reconstruction_gate": {
            "finite": bool(
                np.isfinite(prior_group).all()
                and np.isfinite(projected_group).all()
            ),
            "projected_not_worse_than_prior_coordinate_macro": passed,
            "passed": passed,
            "failure_action": "stop_and_diagnose_without_rescue_component",
        },
        "legacy_evaluation_payload": result,
    }
    wrapper["result_sha256"] = _canonical_json_sha256(wrapper)
    _write_json_new(destination, wrapper)
    return wrapper


def check_artifacts(
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict:
    registry = load_registry(registry_path)
    records = {}
    for field in ("selection_refit_csv", "development_csv"):
        path = Path(registry["allowed_outputs"][field])
        records[field] = {
            "exists": path.exists(),
            "sha256": _sha256_file(path) if path.exists() else None,
        }
    records["checkpoints"] = [
        {
            "path": str(Path(path).resolve()),
            "exists": Path(path).exists(),
            "sha256": _sha256_file(path) if Path(path).exists() else None,
        }
        for path in registry["allowed_outputs"]["r0_checkpoints"]
    ]
    result_path = Path(registry["allowed_outputs"]["development_result"])
    records["development_result"] = {
        "exists": result_path.exists(),
        "sha256": _sha256_file(result_path) if result_path.exists() else None,
    }
    return {
        "registry_sha256": registry["registry_sha256"],
        "artifacts": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    materialize = sub.add_parser("materialize-roles")
    materialize.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train = sub.add_parser("train-member")
    train.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train.add_argument("--member-index", type=int, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=5)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    evaluate.add_argument("--device", default="cuda")
    check = sub.add_parser("check-artifacts")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)

    if args.command == "materialize-roles":
        payload = materialize_roles(args.registry)
    elif args.command == "train-member":
        payload = train_member(
            member_index=args.member_index,
            registry_path=args.registry,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "evaluate":
        payload = evaluate_r0(
            registry_path=args.registry,
            device_name=args.device,
        )
        payload = {
            "status": payload["status"],
            "coordinate_macro": payload["coordinate_macro"],
            "row_metrics": payload["row_metrics"],
            "reconstruction_gate": payload["reconstruction_gate"],
            "result": str(Path(args.registry).resolve().parent / DEFAULT_RESULT.name),
        }
    else:
        payload = check_artifacts(args.registry)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
