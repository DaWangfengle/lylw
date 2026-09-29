# 从 train_anchor.py 提取，自包含 fhm_dapp_tim 实验
"""Anchor 训练辅助函数：路径解析、协议划分、锚点记忆与检查点加载.

从 train_anchor.py 提取的函数集合，仅依赖 fhm_dapp_tim 内部模块.
保留原始代码逻辑，不修改函数行为.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.model import AnchorStudentTModel
from fhm_dapp_tim.fhm_dapp.physics import (
    FixedAPCensoredLikelihood,
    load_sod_ap_metadata,
)
from fhm_dapp_tim.fhm_dapp.data import denorm_coords, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def raw_coords(
    coords_norm: torch.Tensor,
    building: torch.Tensor,
    stats: dict,
) -> np.ndarray:
    coords_np = coords_norm.numpy()
    building_np = building.numpy()
    x, y = denorm_coords(
        coords_np[:, 0], coords_np[:, 1], building_np, stats)
    return np.stack([x, y], axis=1).astype(np.float64)


def location_keys(
    coords: np.ndarray, building: np.ndarray, floor: np.ndarray
) -> list[tuple[int, int, float, float]]:
    return [
        (int(b), int(f), round(float(x), 4), round(float(y), 4))
        for (x, y), b, f in zip(coords, building, floor)
    ]


def grouped_protocol_split(
    coords: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    validation_fraction: float,
    calibration_fraction: float,
    seed: int,
    timestamps: np.ndarray | None = None,
    strategy: str = "random",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split exact reference points, stratified by building and floor.

    ``temporal`` is the frozen legacy protocol.  It intentionally reproduces
    the historical float32 timestamp quantization and assigns the middle time
    block to ``calibration`` and the latest block to ``validation``.

    ``forward_temporal_v2`` is the corrected protocol for new experiments.
    It uses float64 timestamp medians, keeps equal-time blocks intact, and
    orders the returned roles chronologically as train, validation
    (development), calibration (one-shot confirmation).
    """
    if strategy not in {"random", "temporal", "forward_temporal_v2"}:
        raise ValueError(f"unknown split strategy: {strategy}")
    if strategy in {"temporal", "forward_temporal_v2"} and timestamps is None:
        raise ValueError("temporal splitting requires timestamps")
    keys = location_keys(coords, building, floor)
    strata: dict[tuple[int, int], list[tuple[int, int, float, float]]] = {}
    for key in dict.fromkeys(keys):
        strata.setdefault(key[:2], []).append(key)

    rng = np.random.default_rng(seed)
    train_keys: set[tuple[int, int, float, float]] = set()
    validation_keys: set[tuple[int, int, float, float]] = set()
    calibration_keys: set[tuple[int, int, float, float]] = set()
    group_times: dict[tuple[int, int, float, float], float] = {}
    if timestamps is not None:
        timestamp_values = np.asarray(timestamps)
        if timestamp_values.shape != (len(keys),):
            raise ValueError("timestamps must contain one value per row")
        if strategy == "temporal":
            # Preserve the exact legacy partition for old checkpoints and
            # evidence even though the loader now retains timestamp precision.
            timestamp_values = timestamp_values.astype(np.float32)
        else:
            timestamp_values = timestamp_values.astype(np.float64)
        if not np.all(np.isfinite(timestamp_values)):
            raise ValueError("timestamps must be finite")
        time_values: dict[tuple[int, int, float, float], list[float]] = {}
        for key, timestamp in zip(keys, timestamp_values):
            time_values.setdefault(key, []).append(float(timestamp))
        group_times = {
            key: float(np.median(values)) for key, values in time_values.items()
        }
    for values in strata.values():
        values = list(values)
        if strategy in {"temporal", "forward_temporal_v2"}:
            values.sort(key=lambda key: group_times[key])
        else:
            rng.shuffle(values)
        n_groups = len(values)
        n_val = int(round(n_groups * validation_fraction))
        n_cal = int(round(n_groups * calibration_fraction))
        if n_groups >= 5:
            n_val = max(1, n_val)
            n_cal = max(1, n_cal)
        while n_val + n_cal >= n_groups and (n_val > 0 or n_cal > 0):
            if n_cal >= n_val and n_cal > 0:
                n_cal -= 1
            elif n_val > 0:
                n_val -= 1
        if strategy == "temporal":
            train_end = n_groups - n_val - n_cal
            calibration_end = n_groups - n_val
            train_keys.update(values[:train_end])
            calibration_keys.update(values[train_end:calibration_end])
            validation_keys.update(values[calibration_end:])
        elif strategy == "forward_temporal_v2":
            if n_val == 0 and n_cal == 0:
                train_keys.update(values)
                continue

            # Split only between exact time blocks.  This prevents arbitrary
            # file-order tie breaking from placing measurements taken at the
            # same time into different experimental roles.
            blocks: list[list[tuple[int, int, float, float]]] = []
            for key in values:
                if not blocks or group_times[blocks[-1][0]] != group_times[key]:
                    blocks.append([])
                blocks[-1].append(key)
            minimum_blocks = 1 + int(n_val > 0) + int(n_cal > 0)
            if len(blocks) < minimum_blocks:
                raise ValueError(
                    "forward_temporal_v2 cannot keep timestamp ties intact "
                    "while producing all requested roles"
                )

            prefix = np.concatenate((
                np.asarray([0], dtype=np.int64),
                np.cumsum([len(block) for block in blocks], dtype=np.int64),
            ))
            target_train = n_groups - n_val - n_cal
            candidates: list[tuple[tuple[int, ...], int, int]] = []
            first_min = 1
            first_max = len(blocks) - int(n_cal > 0)
            for first in range(first_min, first_max + 1):
                second_min = first + int(n_val > 0)
                second_max = len(blocks)
                for second in range(second_min, second_max + 1):
                    train_count = int(prefix[first])
                    validation_count = int(prefix[second] - prefix[first])
                    calibration_count = int(prefix[-1] - prefix[second])
                    if n_val > 0 and validation_count == 0:
                        continue
                    if n_cal > 0 and calibration_count == 0:
                        continue
                    score = (
                        abs(validation_count - n_val)
                        + abs(calibration_count - n_cal),
                        abs(train_count - target_train),
                        abs(validation_count - n_val),
                        abs(calibration_count - n_cal),
                        first,
                        second,
                    )
                    candidates.append((score, first, second))
            if not candidates:
                raise ValueError(
                    "forward_temporal_v2 found no valid timestamp-block split"
                )
            _, first, second = min(candidates, key=lambda row: row[0])
            train_keys.update(
                key for block in blocks[:first] for key in block)
            validation_keys.update(
                key for block in blocks[first:second] for key in block)
            calibration_keys.update(
                key for block in blocks[second:] for key in block)
        else:
            validation_keys.update(values[:n_val])
            calibration_keys.update(values[n_val:n_val + n_cal])
            train_keys.update(values[n_val + n_cal:])

    train_idx = np.array(
        [i for i, key in enumerate(keys) if key in train_keys], dtype=np.int64)
    validation_idx = np.array(
        [i for i, key in enumerate(keys) if key in validation_keys], dtype=np.int64)
    calibration_idx = np.array(
        [i for i, key in enumerate(keys) if key in calibration_keys], dtype=np.int64)
    return train_idx, validation_idx, calibration_idx


def build_anchor_memory(
    rss: torch.Tensor,
    centered_coords: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    keys = location_keys(centered_coords, building, floor)
    grouped: dict[tuple[int, int, float, float], list[int]] = {}
    for index in train_idx:
        grouped.setdefault(keys[int(index)], []).append(int(index))

    anchor_keys = sorted(grouped)
    key_to_anchor = {key: i for i, key in enumerate(anchor_keys)}
    sample_anchor = np.full(len(rss), -1, dtype=np.int64)
    prototypes = []
    detection_probabilities = []
    rss_centers = []
    rss_scales = []
    anchor_coords = []
    anchor_building = []
    anchor_floor = []
    counts = []
    for anchor_id, key in enumerate(anchor_keys):
        indices = grouped[key]
        sample_anchor[indices] = anchor_id
        anchor_rss = rss[indices]
        detected = anchor_rss > -0.999
        visible_values = torch.where(
            detected,
            anchor_rss,
            torch.full_like(anchor_rss, float("nan")),
        )
        visible_center = torch.nanmedian(
            visible_values, dim=0).values
        visible_center = torch.nan_to_num(visible_center, nan=-1.0)
        absolute_deviation = torch.where(
            detected,
            (anchor_rss - visible_center).abs(),
            torch.full_like(anchor_rss, float("nan")),
        )
        visible_scale = torch.nanmedian(
            absolute_deviation, dim=0).values
        visible_scale = torch.nan_to_num(
            visible_scale, nan=0.15).clamp(min=0.05)
        detection_probability = (
            detected.sum(dim=0).float() + 0.5
        ) / (len(indices) + 1.0)
        prototypes.append(torch.median(anchor_rss, dim=0).values)
        detection_probabilities.append(detection_probability)
        rss_centers.append(visible_center)
        rss_scales.append(visible_scale)
        anchor_coords.append(centered_coords[indices].mean(axis=0))
        anchor_building.append(building[indices[0]])
        anchor_floor.append(floor[indices[0]])
        counts.append(len(indices))

    memory = {
        "prototype_rss": torch.stack(prototypes),
        "detection_prob": torch.stack(detection_probabilities),
        "rss_center": torch.stack(rss_centers),
        "rss_scale": torch.stack(rss_scales),
        "coords": torch.tensor(np.asarray(anchor_coords), dtype=torch.float32),
        "building": torch.tensor(anchor_building, dtype=torch.long),
        "floor": torch.tensor(anchor_floor, dtype=torch.long),
        "counts": torch.tensor(counts, dtype=torch.long),
    }
    return memory, sample_anchor


def load_anchor_checkpoint(
    path: Path, device: torch.device
) -> tuple[AnchorStudentTModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    state = dict(checkpoint["model_state"])
    if "ap_building_owner" not in state:
        prototype_visible = state["prototype_rss"] > -0.999
        anchor_buildings = state["anchor_buildings"]
        building_counts = torch.stack(
            [
                prototype_visible[anchor_buildings == class_id].sum(dim=0)
                for class_id in range(int(anchor_buildings.max()) + 1)
            ],
            dim=0,
        )
        state["ap_building_owner"] = building_counts.argmax(dim=0)
    physics_model = None
    if "physics_model.ap_coords" in state:
        physics_model = FixedAPCensoredLikelihood(
            ap_coords=state["physics_model.ap_coords"],
            ap_frequencies=state["physics_model.ap_frequencies"],
            threshold_dbm=config.get("physics_threshold_dbm", -104.0),
            invisible_weight=config.get("physics_invisible_weight", 0.20),
        )
    model = AnchorStudentTModel(
        n_aps=checkpoint["stats"]["n_aps"],
        anchor_coords=state["anchor_coords"],
        anchor_buildings=state["anchor_buildings"],
        anchor_floors=state["anchor_floors"],
        seen_ap_mask=state["encoder.seen_ap_mask"],
        prototype_rss=state["prototype_rss"],
        ap_building_owner=state["ap_building_owner"],
        n_devices=config.get("n_devices", 0),
        physics_model=physics_model,
        physics_rerank_weight=config.get("physics_rerank_weight", 0.0),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_visible=config["max_visible"],
        dropout=config["dropout"],
        student_df=config["student_df"],
        max_residual_m=config["max_residual_m"],
        residual_experts=config.get("residual_experts", 1),
        scan_feature_mode=config.get("scan_feature_mode", "legacy"),
        prototype_metric=config.get("prototype_metric", "l2"),
        anchor_detection_prob=state.get("anchor_detection_prob"),
        anchor_rss_center=state.get("anchor_rss_center"),
        anchor_rss_scale=state.get("anchor_rss_scale"),
        anchor_sample_count=state.get("anchor_sample_count"),
        candidate_reranker=config.get("candidate_reranker", False),
        candidate_reranker_mode=config.get(
            "candidate_reranker_mode", "summary"),
        reranker_anchor_decoder=config.get(
            "reranker_anchor_decoder", False),
        anchor_decoder_scale_m=config.get("anchor_decoder_scale_m", 6.0),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


