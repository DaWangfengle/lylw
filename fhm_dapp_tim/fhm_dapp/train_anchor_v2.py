"""Train fresh UJI members under the corrected forward-temporal protocol.

This entry point is deliberately narrow.  The base learner is frozen and the
only repeated factor is the initialization/data-loader seed.  Development
labels are used solely for early stopping; confirmation labels are never
evaluated here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fhm_dapp_tim.fhm_dapp.anchor import (
    build_anchor_memory,
    grouped_protocol_split,
    location_keys,
    raw_coords,
    resolve_path,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.model import AnchorStudentTModel
from fhm_dapp_tim.fhm_dapp.train_anchor import (
    _ap_building_owner,
    _evaluate_model,
    _neighbor_targets,
)


PROTOCOL_VERSION = "uji_forward_temporal_v2"
VALIDATION_FRACTION = 0.10
CALIBRATION_FRACTION = 0.10

# These values describe the frozen base learner, not the ensemble variants.
# Keeping them in one literal makes accidental per-seed drift auditable.
FROZEN_TRAINING = {
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "max_visible": 64,
    "dropout": 0.10,
    "student_df": 4.0,
    "max_residual_m": 30.0,
    "residual_experts": 1,
    "scan_feature_mode": "legacy",
    "prototype_metric": "l2",
    "candidate_reranker": False,
    "candidate_reranker_mode": "summary",
    "reranker_anchor_decoder": False,
    "anchor_decoder_scale_m": 6.0,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "minimum_improvement_m": 0.01,
    "minimum_epochs": 30,
    "target_neighbors": 8,
    "spatial_temperature_m": 8.0,
    "retrieval_weight": 0.50,
    "hierarchy_weight": 1.0,
    "self_target_weight": 0.50,
    "coordinate_weight": 0.25,
    "domain_adversarial_weight": 0.10,
    "rss_offset_aug_db": 4.0,
    "rss_scale_aug": 0.08,
    "ap_dropout": 0.10,
    "train_candidates": 32,
    "inference_top_k": 24,
    "inference_neural_scale": 0.75,
    "inference_prototype_weight": 8.0,
    "posterior_temperature": 0.75,
    "condition_building": False,
    "deterministic_building": True,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Warn-only retains portability if an uncommon CUDA kernel has no fully
    # deterministic implementation, while recording the intended contract.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_indices(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _training_reference_stats(
    stats: dict,
    raw_xy: np.ndarray,
    building: np.ndarray,
    train_idx: np.ndarray,
) -> dict:
    """Return coordinate normalization metadata fitted on training rows only."""
    result = copy.deepcopy(stats)
    building_stats: dict[int, dict[str, float]] = {}
    for building_id in sorted(np.unique(building).tolist()):
        mask = building[train_idx] == int(building_id)
        points = raw_xy[train_idx][mask]
        if len(points) == 0:
            raise ValueError(
                f"training partition has no rows for building {building_id}")
        building_stats[int(building_id)] = {
            "x_min": float(points[:, 0].min()),
            "x_max": float(points[:, 0].max()),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
        }
    result["building_stats"] = building_stats
    result["normalization_fit_partition"] = "train"
    result["protocol_version"] = PROTOCOL_VERSION
    return result


def _remap_training_devices(
    phone: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[torch.Tensor, dict[int, int]]:
    raw_ids = sorted(int(value) for value in np.unique(phone[train_idx]))
    mapping = {raw_id: index for index, raw_id in enumerate(raw_ids)}
    remapped = torch.tensor(
        [mapping[int(phone[index])] for index in train_idx],
        dtype=torch.long,
    )
    return remapped, mapping


def _split_contract(
    data_path: Path,
    keys: list[tuple[int, int, float, float]],
    timestamps: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    calibration_idx: np.ndarray,
) -> dict:
    roles = {
        "train": train_idx,
        "development": validation_idx,
        "confirmation": calibration_idx,
    }
    role_payload = {}
    for name, indices in roles.items():
        groups = {keys[int(index)] for index in indices}
        role_payload[name] = {
            "rows": int(len(indices)),
            "coordinate_groups": int(len(groups)),
            "row_index_sha256": _sha256_indices(indices),
        }
    role_groups = [
        {keys[int(index)] for index in indices} for indices in roles.values()
    ]
    pairwise_disjoint = all(
        role_groups[left].isdisjoint(role_groups[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    if not pairwise_disjoint:
        raise AssertionError("coordinate groups cross experimental roles")
    if sum(len(indices) for indices in roles.values()) != len(keys):
        raise AssertionError("split is not row-exhaustive")
    return {
        "version": PROTOCOL_VERSION,
        "source_path": str(data_path),
        "source_sha256": _sha256_file(data_path),
        "timestamp_dtype": str(np.asarray(timestamps).dtype),
        "strategy": "forward_temporal_v2",
        "validation_fraction": VALIDATION_FRACTION,
        "calibration_fraction": CALIBRATION_FRACTION,
        "roles": role_payload,
        "row_exhaustive": True,
        "coordinate_group_pairwise_disjoint": True,
        "confirmation_labels_consumed_during_training": False,
    }


def train_member(args: argparse.Namespace) -> dict:
    _set_seed(int(args.seed))
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto"
        else "cpu"
    )
    torch.set_num_threads(max(1, min(int(args.threads), torch.get_num_threads())))

    data_path = resolve_path(args.train_data).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing v2 checkpoint: {output_path}")

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
    timestamps = timestamp_t.numpy()
    train_idx, validation_idx, calibration_idx = grouped_protocol_split(
        raw_xy,
        building,
        floor,
        VALIDATION_FRACTION,
        CALIBRATION_FRACTION,
        int(args.split_seed),
        timestamps=timestamps,
        strategy="forward_temporal_v2",
    )
    if min(len(train_idx), len(validation_idx), len(calibration_idx)) == 0:
        raise ValueError("all three protocol roles must be non-empty")

    coord_origin = raw_xy[train_idx].mean(axis=0).astype(np.float32)
    centered_xy = (raw_xy - coord_origin).astype(np.float32)
    stats = _training_reference_stats(
        source_stats, raw_xy, building, train_idx)
    memory, sample_anchor = build_anchor_memory(
        rss,
        centered_xy,
        building,
        floor,
        train_idx,
    )
    if np.any(sample_anchor[train_idx] < 0):
        raise ValueError("some training rows were not assigned to an anchor")

    training_device_ids, device_mapping = _remap_training_devices(
        phone_t.numpy(), train_idx)
    n_devices = len(device_mapping)
    seen_ap_mask = (memory["prototype_rss"] > -0.999).any(dim=0)
    ap_owner = _ap_building_owner(
        memory["prototype_rss"],
        memory["building"],
        int(stats["n_buildings"]),
    )

    model = AnchorStudentTModel(
        n_aps=int(stats["n_aps"]),
        anchor_coords=memory["coords"],
        anchor_buildings=memory["building"],
        anchor_floors=memory["floor"],
        seen_ap_mask=seen_ap_mask,
        prototype_rss=memory["prototype_rss"],
        ap_building_owner=ap_owner,
        n_devices=n_devices,
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

    neighbor_idx, neighbor_weights = _neighbor_targets(
        memory["coords"],
        memory["building"],
        neighbor_count=FROZEN_TRAINING["target_neighbors"],
        temperature_m=FROZEN_TRAINING["spatial_temperature_m"],
    )
    neighbor_idx = neighbor_idx.to(device)
    neighbor_weights = neighbor_weights.to(device)

    target_t = torch.from_numpy(centered_xy).float()
    anchor_t = torch.from_numpy(sample_anchor).long()
    train_dataset = TensorDataset(
        rss[train_idx].float(),
        target_t[train_idx],
        anchor_t[train_idx],
        training_device_ids,
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    loader = DataLoader(
        train_dataset,
        batch_size=FROZEN_TRAINING["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_TRAINING["learning_rate"],
        weight_decay=FROZEN_TRAINING["weight_decay"],
    )

    best_state = None
    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        epoch_losses = []
        for batch_rss, batch_xy, batch_anchor, batch_device in loader:
            batch_rss = batch_rss.to(device)
            batch_xy = batch_xy.to(device)
            batch_anchor = batch_anchor.to(device)
            batch_device = batch_device.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.training_loss(
                batch_rss,
                batch_xy,
                batch_anchor,
                neighbor_idx,
                neighbor_weights,
                retrieval_weight=FROZEN_TRAINING["retrieval_weight"],
                hierarchy_weight=FROZEN_TRAINING["hierarchy_weight"],
                self_target_weight=FROZEN_TRAINING["self_target_weight"],
                coordinate_weight=FROZEN_TRAINING["coordinate_weight"],
                device_ids=batch_device,
                domain_adversarial_weight=(
                    FROZEN_TRAINING["domain_adversarial_weight"]),
                rss_offset_aug_db=FROZEN_TRAINING["rss_offset_aug_db"],
                rss_scale_aug=FROZEN_TRAINING["rss_scale_aug"],
                train_candidates=FROZEN_TRAINING["train_candidates"],
                ap_dropout=FROZEN_TRAINING["ap_dropout"],
            )
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), FROZEN_TRAINING["gradient_clip"])
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation_metrics = _evaluate_model(
            model,
            rss[validation_idx].float(),
            target_t[validation_idx],
            batch_size=int(args.eval_batch_size),
            device=device,
            top_k=FROZEN_TRAINING["inference_top_k"],
            neural_scale=FROZEN_TRAINING["inference_neural_scale"],
            prototype_weight=FROZEN_TRAINING["inference_prototype_weight"],
            posterior_temperature=FROZEN_TRAINING["posterior_temperature"],
        )
        validation_mae = float(validation_metrics["mae"])
        improved = validation_mae < (
            best_mae - FROZEN_TRAINING["minimum_improvement_m"])
        if improved:
            best_mae = validation_mae
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            # Do not let a noisy initialization terminate before the common
            # representation has had a meaningful optimization horizon.
            stale_epochs = (
                stale_epochs + 1
                if epoch >= FROZEN_TRAINING["minimum_epochs"]
                else 0
            )
        record = {
            "epoch": epoch,
            "mean_training_loss": float(np.mean(epoch_losses)),
            "development_mae_m": validation_mae,
            "best_development_mae_m": best_mae,
        }
        history.append(record)
        if epoch == 1 or improved or epoch % int(args.log_every) == 0:
            print(json.dumps(record, allow_nan=False), flush=True)
        if (
            epoch >= FROZEN_TRAINING["minimum_epochs"]
            and stale_epochs >= int(args.patience)
        ):
            break

    if best_state is None:
        raise RuntimeError("training produced no finite development checkpoint")
    model.load_state_dict(best_state)
    final_validation_metrics = _evaluate_model(
        model,
        rss[validation_idx].float(),
        target_t[validation_idx],
        batch_size=int(args.eval_batch_size),
        device=device,
        top_k=FROZEN_TRAINING["inference_top_k"],
        neural_scale=FROZEN_TRAINING["inference_neural_scale"],
        prototype_weight=FROZEN_TRAINING["inference_prototype_weight"],
        posterior_temperature=FROZEN_TRAINING["posterior_temperature"],
    )

    keys = location_keys(raw_xy, building, floor)
    split_contract = _split_contract(
        data_path,
        keys,
        timestamps,
        train_idx,
        validation_idx,
        calibration_idx,
    )
    config = dict(FROZEN_TRAINING)
    config.update({
        "training_protocol_version": PROTOCOL_VERSION,
        "n_devices": n_devices,
        "train_all": False,
        "validation_fraction": VALIDATION_FRACTION,
        "calibration_fraction": CALIBRATION_FRACTION,
        "split_strategy": "forward_temporal_v2",
        "split_seed": int(args.split_seed),
        "epochs": int(args.max_epochs),
        "patience": int(args.patience),
        # Compatibility names used by the existing checkpoint loader/audits.
        "lr": FROZEN_TRAINING["learning_rate"],
        "target_neighbors": FROZEN_TRAINING["target_neighbors"],
    })
    checkpoint = {
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "config": config,
        "stats": stats,
        "coord_origin": coord_origin,
        "conformal_quantiles": {},
        "epoch": int(best_epoch),
        "val_mae": float(final_validation_metrics["mae"]),
        "seed": int(args.seed),
        "development_metrics": final_validation_metrics,
        "split_contract": split_contract,
        "device_id_mapping": {
            str(raw_id): mapped for raw_id, mapped in device_mapping.items()
        },
        "training_history": history,
        "training_seconds": float(time.perf_counter() - started),
        "confirmation_metrics": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    payload = {
        "output": str(output_path),
        "seed": int(args.seed),
        "device": str(device),
        "rows": split_contract["roles"],
        "anchors": int(memory["coords"].shape[0]),
        "features": int(stats["n_aps"]),
        "best_epoch": int(best_epoch),
        "development_metrics": final_validation_metrics,
        "training_seconds": checkpoint["training_seconds"],
        "confirmation_evaluated": False,
    }
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one fresh corrected-protocol UJI ensemble member")
    parser.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=5)
    train_member(parser.parse_args())


if __name__ == "__main__":
    main()
