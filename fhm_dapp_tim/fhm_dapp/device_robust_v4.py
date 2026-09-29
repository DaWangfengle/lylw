"""Device-robust member and radio-evidence primitives for DR-JAM v4.

This module is deliberately separate from the frozen v3.1 implementation.  It
adds one domain-specific mechanism: a positive-affine-invariant member input and
a rank-two device random effect in the radio-map score.  Existing v3 artifacts
must never be loaded as v4 checkpoints or silently recomputed through this code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fhm_dapp_tim.fhm_dapp.anchor import location_keys
from fhm_dapp_tim.fhm_dapp.minimal_v3 import MinimalAnchorModel
from fhm_dapp_tim.fhm_dapp.model import gradient_reverse


DEVICE_STUDENT_DF = 4.0
VISIBLE_THRESHOLD = -0.999


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _masked_midpoint_median(
    values: torch.Tensor, mask: torch.Tensor, *, empty_value: float
) -> torch.Tensor:
    """Deterministic midpoint median for masked row vectors on CPU or CUDA."""
    if values.ndim != 2 or mask.shape != values.shape or mask.dtype != torch.bool:
        raise ValueError("values and mask must be matching [batch,width] tensors")
    valid = mask & ~torch.isnan(values)
    ordered = torch.sort(
        torch.where(valid, values, torch.full_like(values, float("inf"))),
        dim=1,
        stable=True,
    ).values
    count = valid.sum(dim=1)
    lower_index = ((count - 1).clamp(min=0) // 2).unsqueeze(1)
    upper_index = (count.clamp(min=1) // 2).unsqueeze(1)
    lower = ordered.gather(1, lower_index).squeeze(1)
    upper = ordered.gather(1, upper_index).squeeze(1)
    median = 0.5 * (lower + upper)
    return torch.where(
        count > 0, median, torch.full_like(median, float(empty_value))
    )


class DeviceRobustAnchorModel(MinimalAnchorModel):
    """Minimal anchor model with an optional DANN control head.

    The proposed DR-JAM member uses ``scan_feature_mode='invariant'`` and does
    not use ``device_head``.  The head exists only so that the standard DANN
    control can use the identical encoder and anchor dictionary.
    """

    MODEL_SCHEMA = "device_robust_anchor_categorical_v43"

    def __init__(self, *, n_devices: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        device_count = int(n_devices)
        if device_count < 0:
            raise ValueError("n_devices cannot be negative")
        self.n_devices = device_count
        self.device_head = (
            nn.Linear(self.anchor_keys.shape[1], device_count)
            if device_count >= 2
            else None
        )


def load_device_robust_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[DeviceRobustAnchorModel, dict]:
    """Load a tensor-and-primitive-only v4 checkpoint.

    V4 checkpoints intentionally avoid arbitrary pickled Python objects, so
    PyTorch's restricted ``weights_only`` loader can be used here.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    if config.get("model_schema") != DeviceRobustAnchorModel.MODEL_SCHEMA:
        raise ValueError(f"not a DR-JAM-v4 checkpoint: {path}")
    state = checkpoint["model_state"]
    model = DeviceRobustAnchorModel(
        n_aps=int(checkpoint["stats"]["n_aps"]),
        anchor_coords=state["anchor_coords"],
        anchor_buildings=state["anchor_buildings"],
        anchor_floors=state["anchor_floors"],
        seen_ap_mask=state["encoder.seen_ap_mask"],
        n_devices=int(config.get("n_training_devices", 0)),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        max_visible=int(config["max_visible"]),
        dropout=float(config["dropout"]),
        visible_threshold=float(config.get("visible_threshold", -0.75)),
        scan_feature_mode=str(config["scan_feature_mode"]),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


def anchor_support_3d(
    model: MinimalAnchorModel,
    *,
    floor_height_m: float = 4.0,
) -> torch.Tensor:
    """Return one distinct spatial support for proper-score member training."""
    height = float(floor_height_m)
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("floor_height_m must be finite and positive")
    vertical = model.anchor_floors.to(model.anchor_coords.dtype).unsqueeze(1)
    return torch.cat((model.anchor_coords, height * vertical), dim=1)


def spatial_energy_member_loss(
    model: DeviceRobustAnchorModel,
    rss: torch.Tensor,
    target_anchor: torch.Tensor,
    *,
    pair_distance_m: torch.Tensor | None = None,
    distance_normalizer_m: torch.Tensor | float | None = None,
    device_labels: torch.Tensor | None = None,
    domain_adversarial_weight: float = 0.0,
    floor_height_m: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Train one anchor distribution with a 3-D spatial energy score.

    Energy score replaces, rather than supplements, the anchor log score.  The
    optional DANN term is used only by the matched standard control.
    """
    if rss.ndim != 2 or target_anchor.shape != (len(rss),):
        raise ValueError("rss and target_anchor must be [B,J] and [B]")
    if target_anchor.dtype != torch.long:
        raise TypeError("target_anchor must use torch.long indices")
    gamma = float(domain_adversarial_weight)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("domain_adversarial_weight must be finite and nonnegative")

    logits, embedding = model.retrieval_logits(rss)
    probability = torch.softmax(logits, dim=1)
    support = anchor_support_3d(model, floor_height_m=floor_height_m)
    if pair_distance_m is None:
        pair_distance = torch.cdist(support, support, p=2)
    else:
        pair_distance = pair_distance_m.to(
            device=probability.device, dtype=probability.dtype)
        if pair_distance.shape != (model.n_anchors, model.n_anchors):
            raise ValueError("pair_distance_m has the wrong anchor shape")
    target_distance = pair_distance[target_anchor]
    first = torch.sum(probability * target_distance, dim=1)
    second = 0.5 * torch.sum((probability @ pair_distance) * probability, dim=1)
    energy_m = first - second

    if distance_normalizer_m is None:
        positive = pair_distance[pair_distance > 0.0]
        if positive.numel() == 0:
            raise ValueError("anchor support must contain distinct locations")
        normalizer = positive.detach().median()
    else:
        normalizer = torch.as_tensor(
            distance_normalizer_m,
            device=probability.device,
            dtype=probability.dtype,
        )
        if normalizer.numel() != 1 or not bool(torch.isfinite(normalizer)):
            raise ValueError("distance_normalizer_m must be one finite scalar")
        if not bool(normalizer > 0.0):
            raise ValueError("distance_normalizer_m must be positive")
    spatial_loss = energy_m.mean() / normalizer

    domain_loss = spatial_loss.new_zeros(())
    if gamma > 0.0:
        if model.device_head is None:
            raise ValueError("DANN requires a model with at least two devices")
        if device_labels is None or device_labels.shape != (len(rss),):
            raise ValueError("DANN requires one device label per row")
        if device_labels.dtype != torch.long:
            raise TypeError("device_labels must use torch.long indices")
        domain_logits = model.device_head(gradient_reverse(embedding))
        domain_loss = F.cross_entropy(domain_logits, device_labels)
    loss = spatial_loss + gamma * domain_loss
    return {
        "loss": loss,
        "spatial_energy_score_m": energy_m.detach().mean(),
        "normalized_spatial_loss": spatial_loss.detach(),
        "domain_loss": domain_loss.detach(),
    }


def device_group_sampling_weights(
    phone_ids: np.ndarray | torch.Tensor,
    coordinate_keys: list[tuple] | np.ndarray,
    indices: np.ndarray,
) -> torch.Tensor:
    """Weights for equal-phone, equal-coordinate, equal-row sampling."""
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("indices must be one non-empty vector")
    phone = (
        phone_ids.detach().cpu().numpy()
        if isinstance(phone_ids, torch.Tensor)
        else np.asarray(phone_ids)
    )
    if len(phone) <= int(selected.max()):
        raise ValueError("phone_ids do not cover selected rows")
    if len(coordinate_keys) <= int(selected.max()):
        raise ValueError("coordinate_keys do not cover selected rows")

    group_rows: dict[tuple[object, object], list[int]] = {}
    phone_groups: dict[object, set[object]] = {}
    for local_index, row in enumerate(selected):
        device = phone[int(row)].item() if hasattr(phone[int(row)], "item") else phone[int(row)]
        coordinate = coordinate_keys[int(row)]
        if isinstance(coordinate, np.ndarray):
            coordinate = tuple(coordinate.tolist())
        elif isinstance(coordinate, list):
            coordinate = tuple(coordinate)
        key = (device, coordinate)
        group_rows.setdefault(key, []).append(local_index)
        phone_groups.setdefault(device, set()).add(coordinate)
    weights = np.empty(len(selected), dtype=np.float64)
    for (device, _coordinate), local_rows in group_rows.items():
        group_count = len(phone_groups[device])
        value = 1.0 / (float(group_count) * float(len(local_rows)))
        weights[np.asarray(local_rows, dtype=np.int64)] = value
    weights /= weights.sum()
    return torch.from_numpy(weights).to(dtype=torch.double)


def build_device_balanced_anchor_memory(
    rss: torch.Tensor,
    centered_coords: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    phone_ids: np.ndarray | torch.Tensor,
    train_idx: np.ndarray,
    *,
    coordinate_keys: list[tuple] | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """Build a radio map that gives each observed phone equal anchor weight.

    Row-level medians make phones with thousands of repeated scans define the
    nominal radio map.  Here each phone is summarized inside an anchor first;
    the anchor center and detection probability then average phone summaries.
    The diagonal scale describes within-phone noise, while cross-phone
    offset/gain variation is reserved for the rank-two random effect.
    """
    selected = np.asarray(train_idx, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("train_idx must be one non-empty vector")
    phones = (
        phone_ids.detach().cpu().numpy()
        if isinstance(phone_ids, torch.Tensor)
        else np.asarray(phone_ids)
    ).astype(np.int64, copy=False)
    if len(phones) != len(rss):
        raise ValueError("phone_ids must contain one value per RSS row")
    if centered_coords.shape != (len(rss), 2):
        raise ValueError("centered_coords must have shape [rows,2]")
    if coordinate_keys is None:
        keys = location_keys(centered_coords, building, floor)
    else:
        if len(coordinate_keys) != len(rss):
            raise ValueError("coordinate_keys must contain one key per row")
        keys = list(coordinate_keys)
    grouped: dict[tuple, list[int]] = {}
    for row in selected:
        grouped.setdefault(keys[int(row)], []).append(int(row))
    anchor_keys = sorted(grouped)
    sample_anchor = np.full(len(rss), -1, dtype=np.int64)
    prototypes = []
    detection_probabilities = []
    rss_centers = []
    rss_scales = []
    anchor_coords = []
    anchor_building = []
    anchor_floor = []
    counts = []
    device_counts = []
    for anchor_id, key in enumerate(anchor_keys):
        rows = grouped[key]
        sample_anchor[np.asarray(rows, dtype=np.int64)] = anchor_id
        phone_centers = []
        phone_detection = []
        phone_scales = []
        for phone in sorted(map(int, np.unique(phones[rows]))):
            phone_rows = [row for row in rows if int(phones[row]) == phone]
            block = rss[phone_rows]
            detected = block > VISIBLE_THRESHOLD
            visible_values = torch.where(
                detected, block, torch.full_like(block, float("nan"))
            )
            center = torch.nanmedian(visible_values, dim=0).values
            deviation = torch.where(
                detected,
                (block - center).abs(),
                torch.full_like(block, float("nan")),
            )
            scale = torch.nanmedian(deviation, dim=0).values
            phone_centers.append(center)
            phone_scales.append(scale)
            phone_detection.append(detected.float().mean(dim=0))
        center_stack = torch.stack(phone_centers)
        # Equal-phone mean is the population center implied by a zero-mean
        # device random effect.  The Student-t score handles residual tails.
        center = torch.nanmean(center_stack, dim=0)
        center = torch.nan_to_num(center, nan=-1.0)
        scale = torch.nanmedian(torch.stack(phone_scales), dim=0).values
        scale = torch.nan_to_num(scale, nan=0.15).clamp(min=0.05)
        detection = torch.stack(phone_detection).mean(dim=0)
        prototypes.append(center)
        detection_probabilities.append(detection)
        rss_centers.append(center)
        rss_scales.append(scale)
        anchor_coords.append(centered_coords[rows].mean(axis=0))
        anchor_building.append(building[rows[0]])
        anchor_floor.append(floor[rows[0]])
        counts.append(len(rows))
        device_counts.append(len(phone_centers))
    memory = {
        "prototype_rss": torch.stack(prototypes),
        "detection_prob": torch.stack(detection_probabilities),
        "rss_center": torch.stack(rss_centers),
        "rss_scale": torch.stack(rss_scales),
        "coords": torch.tensor(np.asarray(anchor_coords), dtype=torch.float32),
        "building": torch.tensor(anchor_building, dtype=torch.long),
        "floor": torch.tensor(anchor_floor, dtype=torch.long),
        "counts": torch.tensor(counts, dtype=torch.long),
        "device_counts": torch.tensor(device_counts, dtype=torch.long),
        # This is the exact protocol identity, not a key reconstructed from
        # float32 centered coordinates.  It intentionally remains a Python
        # list and is stored separately from tensor radio memory at checkpoint.
        "coordinate_keys": anchor_keys,
    }
    return memory, sample_anchor


@dataclass(frozen=True)
class DeviceEffectEstimate:
    """Training-role-only estimate of offset/gain random-effect covariance."""

    covariance: torch.Tensor
    phone_ids: tuple[int, ...]
    effects: torch.Tensor
    observation_counts: tuple[int, ...]
    shared_anchor_counts: tuple[int, ...]

    def validate(self) -> None:
        count = len(self.phone_ids)
        if count < 3:
            raise ValueError("at least three identifiable phones are required")
        if self.effects.shape != (count, 2):
            raise ValueError("effects must have shape [phones,2]")
        if self.covariance.shape != (2, 2):
            raise ValueError("covariance must have shape [2,2]")
        if len(self.observation_counts) != count:
            raise ValueError("observation count length mismatch")
        if len(self.shared_anchor_counts) != count:
            raise ValueError("shared-anchor count length mismatch")
        if not bool(torch.isfinite(self.effects).all()):
            raise ValueError("device effects must be finite")
        singular_values = torch.linalg.svdvals(self.effects)
        if (
            len(singular_values) < 2
            or bool(singular_values[0] <= 0.0)
            or bool(singular_values[1] <= 1e-8 * singular_values[0])
        ):
            raise ValueError("phone effects do not identify two affine directions")
        if not bool(torch.isfinite(self.covariance).all()):
            raise ValueError("device covariance must be finite")
        if not torch.allclose(
            self.covariance, self.covariance.t(), atol=1e-7, rtol=1e-7
        ):
            raise ValueError("device covariance must be symmetric")
        if bool(torch.linalg.eigvalsh(self.covariance).min() <= 0.0):
            raise ValueError("device covariance must be positive definite")


def student_scale_from_marginal_covariance(
    covariance: torch.Tensor, degrees_freedom: float
) -> torch.Tensor:
    """Convert a Student-t marginal covariance to its scale matrix.

    For ``t_nu(0, Sigma)``, ``Cov = nu/(nu-2) Sigma`` when ``nu>2``.
    The training-role device estimator returns an empirical covariance, so the
    low-rank likelihood must use ``Sigma=(nu-2)/nu Cov`` rather than silently
    inflating device variation.
    """
    nu = float(degrees_freedom)
    if not math.isfinite(nu) or nu <= 2.0:
        raise ValueError("a finite Student-t covariance requires degrees_freedom > 2")
    if covariance.shape != (2, 2) or not bool(torch.isfinite(covariance).all()):
        raise ValueError("device covariance must be one finite 2-by-2 matrix")
    return covariance * ((nu - 2.0) / nu)


def _huber_affine_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    weights = np.ones(len(x), dtype=np.float64)
    coefficient = np.zeros(2, dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    for _ in range(8):
        lhs = design.T @ (weights[:, None] * design) + 1e-10 * identity
        rhs = design.T @ (weights * y)
        coefficient = np.linalg.solve(lhs, rhs)
        residual = y - design @ coefficient
        center = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - center))
        if not np.isfinite(scale) or scale <= 1e-8:
            break
        cutoff = 1.345 * scale
        magnitude = np.abs(residual - center)
        weights = np.minimum(1.0, cutoff / np.maximum(magnitude, 1e-12))
    return coefficient


def estimate_device_effect_covariance(
    rss: torch.Tensor,
    sample_anchor: np.ndarray | torch.Tensor,
    phone_ids: np.ndarray | torch.Tensor,
    indices: np.ndarray,
    *,
    minimum_observations: int = 32,
    minimum_shared_anchors: int = 2,
    covariance_floor: float = 1e-6,
) -> DeviceEffectEstimate:
    """Estimate offset/gain variation from same-anchor cross-phone contrasts.

    Anchor-specific RSS is removed before estimating a phone effect, so spatial
    signal variation is not mistaken for device variation.  Phones without any
    shared anchor remain explicitly unidentifiable and are omitted from the
    random-effect covariance rather than assigned a fabricated effect.
    """
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("indices must be one non-empty vector")
    minimum_rows = _positive_integer(minimum_observations, "minimum_observations")
    minimum_groups = _positive_integer(
        minimum_shared_anchors, "minimum_shared_anchors")
    floor = float(covariance_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("covariance_floor must be finite and positive")

    values = rss.detach().cpu().numpy().astype(np.float64, copy=False)
    anchors = (
        sample_anchor.detach().cpu().numpy()
        if isinstance(sample_anchor, torch.Tensor)
        else np.asarray(sample_anchor)
    ).astype(np.int64, copy=False)
    phones = (
        phone_ids.detach().cpu().numpy()
        if isinstance(phone_ids, torch.Tensor)
        else np.asarray(phone_ids)
    ).astype(np.int64, copy=False)
    maximum = int(selected.max())
    if min(len(values), len(anchors), len(phones)) <= maximum:
        raise ValueError("input arrays do not cover selected rows")

    x_by_phone: dict[int, list[np.ndarray]] = {}
    y_by_phone: dict[int, list[np.ndarray]] = {}
    shared_by_phone: dict[int, int] = {}
    for anchor in np.unique(anchors[selected]):
        anchor_rows = selected[anchors[selected] == anchor]
        anchor_phones = np.unique(phones[anchor_rows])
        if len(anchor_phones) < 2:
            continue
        medians: dict[int, np.ndarray] = {}
        for phone in anchor_phones:
            phone_rows = anchor_rows[phones[anchor_rows] == phone]
            block = values[phone_rows].copy()
            block[block <= VISIBLE_THRESHOLD] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                medians[int(phone)] = np.nanmedian(block, axis=0)
        for phone in anchor_phones:
            own = medians[int(phone)]
            others = np.stack([
                medians[int(other)] for other in anchor_phones if other != phone
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                reference = np.nanmedian(others, axis=0)
            valid = np.isfinite(own) & np.isfinite(reference)
            if int(valid.sum()) < 2:
                continue
            reference_visible = reference[valid]
            centered_reference = reference_visible - np.median(reference_visible)
            residual = own[valid] - reference_visible
            x_by_phone.setdefault(int(phone), []).append(centered_reference)
            y_by_phone.setdefault(int(phone), []).append(residual)
            shared_by_phone[int(phone)] = shared_by_phone.get(int(phone), 0) + 1

    effect_phones = []
    effects = []
    observations = []
    shared_counts = []
    for phone in sorted(x_by_phone):
        x = np.concatenate(x_by_phone[phone])
        y = np.concatenate(y_by_phone[phone])
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        shared = shared_by_phone.get(phone, 0)
        if len(x) < minimum_rows or shared < minimum_groups:
            continue
        coefficient = _huber_affine_fit(x, y)
        if not np.isfinite(coefficient).all():
            continue
        effect_phones.append(int(phone))
        effects.append(coefficient)
        observations.append(int(len(x)))
        shared_counts.append(int(shared))
    if len(effects) < 3:
        raise ValueError("fewer than three phones have identifiable affine effects")

    effect_array = np.asarray(effects, dtype=np.float64)
    robust_center = np.median(effect_array, axis=0)
    robust_scale = 1.4826 * np.median(
        np.abs(effect_array - robust_center), axis=0)
    fallback_scale = np.std(effect_array, axis=0, ddof=1)
    robust_scale = np.where(
        robust_scale > 1e-8, robust_scale, np.maximum(fallback_scale, 1e-8))
    radial = np.linalg.norm(
        (effect_array - robust_center) / robust_scale, axis=1)
    weights = np.minimum(1.0, 2.5 / np.maximum(radial, 1e-12))
    weighted_center = np.sum(
        weights[:, None] * effect_array, axis=0) / weights.sum()
    effect_array = effect_array - weighted_center
    singular_values = np.linalg.svd(effect_array, compute_uv=False)
    if (
        len(singular_values) < 2
        or singular_values[0] <= 0.0
        or singular_values[1] <= 1e-8 * singular_values[0]
    ):
        raise ValueError("phone effects do not identify two affine directions")
    denominator = weights.sum() - np.square(weights).sum() / weights.sum()
    if denominator <= 0.0:
        raise ValueError("device effects do not identify a covariance")
    covariance = (
        effect_array.T @ (weights[:, None] * effect_array) / denominator)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= max(1e-12, 1e-8 * eigenvalues[-1]):
        raise ValueError("device-effect covariance is not rank-two identifiable")
    eigenvalues = np.maximum(eigenvalues, floor)
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    result = DeviceEffectEstimate(
        covariance=torch.from_numpy(covariance).to(dtype=torch.float32),
        phone_ids=tuple(effect_phones),
        effects=torch.from_numpy(effect_array).to(dtype=torch.float32),
        observation_counts=tuple(observations),
        shared_anchor_counts=tuple(shared_counts),
    )
    result.validate()
    return result


def _multivariate_student_nll(
    residual: torch.Tensor,
    scale: torch.Tensor,
    visible: torch.Tensor,
    centered_reference: torch.Tensor,
    covariance: torch.Tensor | None,
    degrees_freedom: float,
) -> torch.Tensor:
    """Multivariate Student-t negative log score on the visible subvector."""
    # residual/scale/centered_reference: [B,C,J], visible: [B,1,J]
    batch, anchors, _aps = residual.shape
    mask = visible.to(dtype=residual.dtype)
    count = mask.sum(dim=2).expand(batch, anchors)
    inverse_diagonal = mask / scale.square().clamp_min(1e-8)
    base_quadratic = torch.sum(inverse_diagonal * residual.square(), dim=2)
    log_determinant = torch.sum(
        mask * (2.0 * torch.log(scale.clamp_min(1e-4))), dim=2)

    if covariance is not None:
        marginal_covariance = covariance.to(
            device=residual.device, dtype=residual.dtype
        )
        if marginal_covariance.shape != (2, 2):
            raise ValueError("device covariance must have shape [2,2]")
        if not bool(torch.isfinite(marginal_covariance).all()):
            raise ValueError("device covariance must be finite")
        marginal_covariance = 0.5 * (
            marginal_covariance + marginal_covariance.t()
        )
        omega = student_scale_from_marginal_covariance(
            marginal_covariance, degrees_freedom
        )
        if bool(torch.linalg.eigvalsh(omega).min() <= 0.0):
            raise ValueError("device covariance must be positive definite")
        omega_inverse = torch.linalg.inv(omega)
        z = torch.stack((mask.expand(-1, anchors, -1), centered_reference), dim=3)
        weighted_z = inverse_diagonal.unsqueeze(3) * z
        gram = torch.einsum("bcji,bcjk->bcik", z, weighted_z)
        system = gram + omega_inverse.view(1, 1, 2, 2)
        weighted_residual = inverse_diagonal * residual
        cross = torch.einsum("bcji,bcj->bci", z, weighted_residual)
        solution = torch.linalg.solve(system, cross.unsqueeze(3)).squeeze(3)
        correction = torch.sum(cross * solution, dim=2)
        base_quadratic = (base_quadratic - correction).clamp_min(0.0)
        sign_omega, logdet_omega = torch.linalg.slogdet(omega)
        sign_system, logdet_system = torch.linalg.slogdet(system)
        if not bool(sign_omega > 0.0) or not bool((sign_system > 0.0).all()):
            raise ValueError("device random-effect systems must be positive definite")
        log_determinant = log_determinant + logdet_omega + logdet_system

    nu = residual.new_tensor(float(degrees_freedom))
    count_f = count.to(dtype=residual.dtype)
    constant = (
        torch.lgamma(0.5 * nu)
        - torch.lgamma(0.5 * (nu + count_f))
        + 0.5 * count_f * torch.log(nu * math.pi)
    )
    nll = (
        constant
        + 0.5 * log_determinant
        + 0.5 * (nu + count_f) * torch.log1p(base_quadratic / nu)
    )
    return nll


@torch.no_grad()
def device_marginal_radio_loss(
    query_values: torch.Tensor,
    detection_probability: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    *,
    device_covariance: torch.Tensor | None,
    degrees_freedom: float = DEVICE_STUDENT_DF,
    anchor_chunk_size: int = 64,
) -> torch.Tensor:
    """Evaluate a proper hurdle score with an optional rank-two device effect."""
    if query_values.ndim != 2 or not query_values.is_floating_point():
        raise ValueError("query_values must be floating [batch,aps]")
    if not (
        detection_probability.ndim == 2
        and detection_probability.shape == rss_center.shape == rss_scale.shape
    ):
        raise ValueError("radio-memory tensors must share [anchors,aps]")
    if detection_probability.shape[1] != query_values.shape[1]:
        raise ValueError("query and radio memory must share the AP dimension")
    nu = float(degrees_freedom)
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError("degrees_freedom must be finite and positive")
    chunk = _positive_integer(anchor_chunk_size, "anchor_chunk_size")
    visible = query_values > VISIBLE_THRESHOLD
    visible_expanded = visible.unsqueeze(1)
    outputs = []
    for start in range(0, len(detection_probability), chunk):
        stop = min(start + chunk, len(detection_probability))
        probability = detection_probability[start:stop].to(
            device=query_values.device, dtype=query_values.dtype).clamp(
                min=1e-6, max=1.0 - 1e-6)
        center = rss_center[start:stop].to(
            device=query_values.device, dtype=query_values.dtype)
        scale = rss_scale[start:stop].to(
            device=query_values.device, dtype=query_values.dtype).clamp_min(1e-4)

        detection = -(
            visible_expanded.to(query_values.dtype) * torch.log(probability).unsqueeze(0)
            + (~visible_expanded).to(query_values.dtype)
            * torch.log1p(-probability).unsqueeze(0)
        ).sum(dim=2)

        residual = query_values.unsqueeze(1) - center.unsqueeze(0)
        expanded_scale = scale.unsqueeze(0).expand(len(query_values), -1, -1)
        expanded_center = center.unsqueeze(0).expand(len(query_values), -1, -1)
        expanded_visible = visible_expanded.expand(-1, len(center), -1)
        visible_center = _masked_midpoint_median(
            expanded_center.reshape(-1, expanded_center.shape[2]),
            expanded_visible.reshape(-1, expanded_visible.shape[2]),
            empty_value=0.0,
        ).reshape(len(query_values), len(center))
        centered_reference = (
            expanded_center - visible_center.unsqueeze(2)
        ) * visible_expanded.to(query_values.dtype)
        amplitude = _multivariate_student_nll(
            residual=residual,
            scale=expanded_scale,
            visible=visible_expanded,
            centered_reference=centered_reference,
            covariance=device_covariance,
            degrees_freedom=nu,
        )
        # This is the joint hurdle log score divided by the fixed dictionary
        # width J.  Dividing the amplitude and detection terms by different,
        # outcome-dependent counts would no longer be a fixed rescaling of a
        # proper joint log score.
        outputs.append(
            (detection + amplitude) / float(query_values.shape[1])
        )
    return torch.cat(outputs, dim=1)
