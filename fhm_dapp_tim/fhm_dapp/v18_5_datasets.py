"""V18.5 canonical model inputs and explicit-mask anchor memory.

V18.5 treats missingness as a Boolean event.  The numeric RSS value is never
used to infer whether an AP was observed, so a real clipped -110 dBm sample may
legitimately map to -1.0 without colliding with missingness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.v18_4_datasets import (
    DATA_ROOT,
    RSS_CLIP_MAX_DBM,
    RSS_CLIP_MIN_DBM,
    CanonicalFingerprintData,
    DatasetReference,
    load_tampere as _load_tampere,
    load_tuji1 as _load_tuji1,
    load_uji as _load_uji,
    load_uts as _load_uts,
)


@dataclass(frozen=True)
class V185ModelInput:
    rss: torch.Tensor
    observed_mask: torch.Tensor

    def validate(self) -> None:
        if self.rss.ndim != 2 or self.observed_mask.shape != self.rss.shape:
            raise ValueError("rss/mask must share [rows,aps] shape")
        if self.observed_mask.dtype != torch.bool:
            raise ValueError("observed_mask must be bool")
        if not bool(torch.isfinite(self.rss).all()):
            raise ValueError("model RSS must be finite")


def model_input(
    data: CanonicalFingerprintData,
    *,
    device: torch.device | None = None,
    missing_fill: float = -1.0,
) -> V185ModelInput:
    """Convert canonical dBm data without inferring missingness from values."""

    data.validate()
    observed = np.isfinite(data.rss_dbm)
    rss = np.full(data.rss_dbm.shape, float(missing_fill), dtype=np.float32)
    clipped = np.clip(
        data.rss_dbm[observed],
        RSS_CLIP_MIN_DBM,
        RSS_CLIP_MAX_DBM,
    )
    rss[observed] = (
        (clipped - RSS_CLIP_MIN_DBM)
        / (RSS_CLIP_MAX_DBM - RSS_CLIP_MIN_DBM)
        * 2.0
        - 1.0
    ).astype(np.float32)
    result = V185ModelInput(
        rss=torch.as_tensor(rss, dtype=torch.float32, device=device),
        observed_mask=torch.as_tensor(observed, dtype=torch.bool, device=device),
    )
    result.validate()
    return result


def _coordinate_keys(
    coords: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
) -> list[tuple[int, int, float, float]]:
    return [
        (int(b), int(f), round(float(x), 4), round(float(y), 4))
        for (x, y), b, f in zip(coords, building, floor)
    ]


def _masked_median(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Vectorized per-AP median; memory construction is CPU-only."""
    if values.ndim != 2 or active.shape != values.shape:
        raise ValueError("masked median expects aligned matrices")
    masked = torch.where(
        active,
        values,
        torch.full_like(values, float("nan")),
    )
    return torch.nanmedian(masked, dim=0).values


def _masked_mad(
    values: torch.Tensor,
    active: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 2 or active.shape != values.shape:
        raise ValueError("masked MAD expects aligned matrices")
    deviation = torch.where(
        active,
        (values - center.unsqueeze(0)).abs(),
        torch.full_like(values, float("nan")),
    )
    return torch.nanmedian(deviation, dim=0).values


def build_v185_anchor_memory(
    rss: torch.Tensor,
    observed_mask: torch.Tensor,
    centered_coords: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    train_idx: Sequence[int] | np.ndarray,
    *,
    minimum_scale: float = 0.05,
) -> tuple[dict[str, torch.Tensor], np.ndarray, dict[str, object]]:
    """Build training-only marked memory with count-aware scale fallback."""

    if rss.ndim != 2 or observed_mask.shape != rss.shape:
        raise ValueError("rss and observed_mask must share [rows,aps] shape")
    if observed_mask.dtype != torch.bool:
        raise ValueError("observed_mask must be bool")
    if not bool(torch.isfinite(rss).all()):
        raise ValueError("rss must be finite")
    if not np.isfinite(centered_coords).all():
        raise ValueError("coordinates must be finite")
    minimum_scale = float(minimum_scale)
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive and finite")

    train_idx = np.asarray(train_idx, dtype=np.int64)
    if train_idx.ndim != 1 or len(train_idx) == 0:
        raise ValueError("train_idx must be a nonempty vector")
    if train_idx.min() < 0 or train_idx.max() >= len(rss):
        raise ValueError("train_idx outside data range")

    train_values = rss[torch.as_tensor(train_idx, dtype=torch.long, device=rss.device)]
    train_mask = observed_mask[
        torch.as_tensor(train_idx, dtype=torch.long, device=rss.device)
    ]
    ap_count = train_mask.sum(dim=0)
    ap_center = _masked_median(train_values, train_mask)
    ap_mad = _masked_mad(train_values, train_mask, ap_center)

    flattened = train_values[train_mask]
    if flattened.numel() == 0:
        raise ValueError("training memory has no observed RSS")
    global_center = torch.median(flattened)
    global_scale = torch.median((flattened - global_center).abs())
    if not bool(torch.isfinite(global_scale)) or float(global_scale) < minimum_scale:
        global_scale = rss.new_tensor(minimum_scale)

    keys = _coordinate_keys(centered_coords, building, floor)
    grouped: dict[tuple[int, int, float, float], list[int]] = {}
    train_set = set(map(int, train_idx.tolist()))
    for index in sorted(train_set):
        grouped.setdefault(keys[index], []).append(index)
    if not grouped:
        raise ValueError("no anchors found")

    anchor_keys = sorted(grouped)
    sample_anchor = np.full(len(rss), -1, dtype=np.int64)
    prototypes = []
    prototype_observed = []
    detection_probabilities = []
    rss_centers = []
    rss_scales = []
    local_observed_counts = []
    coords_out = []
    buildings_out = []
    floors_out = []
    counts = []

    local_scale_cells = 0
    ap_fallback_cells = 0
    global_fallback_cells = 0
    observed_scale_cells = 0

    for anchor_id, key in enumerate(anchor_keys):
        indices = grouped[key]
        sample_anchor[indices] = anchor_id
        index_t = torch.as_tensor(indices, dtype=torch.long, device=rss.device)
        values = rss[index_t]
        active = observed_mask[index_t]
        n_local = active.sum(dim=0)
        center = _masked_median(values, active)
        center_filled = torch.where(
            torch.isfinite(center),
            center,
            torch.where(torch.isfinite(ap_center), ap_center, torch.zeros_like(center)),
        )
        local_mad = _masked_mad(values, active, center_filled)

        local_ok = (
            (n_local >= 2)
            & torch.isfinite(local_mad)
            & (local_mad >= minimum_scale)
        )
        pooled_ok = (
            (~local_ok)
            & (ap_count >= 2)
            & torch.isfinite(ap_mad)
            & (ap_mad >= minimum_scale)
        )
        scale = torch.where(
            local_ok,
            local_mad,
            torch.where(
                pooled_ok,
                ap_mad,
                torch.full_like(local_mad, float(global_scale)),
            ),
        ).clamp_min(minimum_scale)

        local_scale_cells += int(local_ok.sum())
        ap_fallback_cells += int(pooled_ok.sum())
        global_fallback_cells += int((~local_ok & ~pooled_ok).sum())
        observed_scale_cells += int(scale.numel())

        prototype = center_filled.clone()
        detection = (n_local.float() + 0.5) / (len(indices) + 1.0)
        prototypes.append(prototype)
        prototype_observed.append(n_local > 0)
        detection_probabilities.append(detection)
        rss_centers.append(center_filled)
        rss_scales.append(scale)
        local_observed_counts.append(n_local)
        coords_out.append(centered_coords[indices].mean(axis=0))
        buildings_out.append(building[indices[0]])
        floors_out.append(floor[indices[0]])
        counts.append(len(indices))

    memory = {
        "prototype_rss": torch.stack(prototypes).float(),
        "prototype_observed_mask": torch.stack(prototype_observed).bool(),
        "detection_prob": torch.stack(detection_probabilities).float(),
        "rss_center": torch.stack(rss_centers).float(),
        "rss_scale": torch.stack(rss_scales).float(),
        "local_observed_count": torch.stack(local_observed_counts).long(),
        "ap_pooled_count": ap_count.long(),
        "ap_pooled_center": torch.nan_to_num(ap_center, nan=0.0).float(),
        "ap_pooled_scale": torch.where(
            torch.isfinite(ap_mad),
            ap_mad,
            torch.full_like(ap_mad, float(global_scale)),
        ).clamp_min(minimum_scale).float(),
        "global_rss_scale": global_scale.float(),
        "coords": torch.as_tensor(np.asarray(coords_out), dtype=torch.float32),
        "building": torch.as_tensor(buildings_out, dtype=torch.long),
        "floor": torch.as_tensor(floors_out, dtype=torch.long),
        "counts": torch.as_tensor(counts, dtype=torch.long),
        "seen_ap_mask": (ap_count > 0).bool(),
    }
    total = max(1, observed_scale_cells)
    diagnostics = {
        "anchors": int(len(anchor_keys)),
        "singleton_anchors": int(sum(value == 1 for value in counts)),
        "singleton_anchor_fraction": float(
            sum(value == 1 for value in counts) / len(anchor_keys)
        ),
        "local_scale_fraction": float(local_scale_cells / total),
        "ap_pooled_fallback_fraction": float(ap_fallback_cells / total),
        "global_fallback_fraction": float(global_fallback_cells / total),
        "global_rss_scale": float(global_scale),
        "training_observed_values": int(train_mask.sum()),
    }
    if (
        local_scale_cells + ap_fallback_cells + global_fallback_cells
        != observed_scale_cells
    ):
        raise RuntimeError("scale fallback accounting mismatch")
    return memory, sample_anchor, diagnostics


def load_uji(split: str, **kwargs):
    return _load_uji(split, **kwargs)


def load_uts(split: str, **kwargs):
    return _load_uts(split, **kwargs)


def load_tampere(split: str, **kwargs):
    return _load_tampere(split, **kwargs)


def load_tuji1(split: str, **kwargs):
    return _load_tuji1(split, **kwargs)


__all__ = [
    "DATA_ROOT",
    "CanonicalFingerprintData",
    "DatasetReference",
    "V185ModelInput",
    "build_v185_anchor_memory",
    "load_tampere",
    "load_tuji1",
    "load_uji",
    "load_uts",
    "model_input",
]
