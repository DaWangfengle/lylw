"""V7.1 core mechanisms: hierarchical marked memory and controlled projection.

This module intentionally contains no command that reads the already exposed
V7 protected roles.  It defines the next model family and its outcome-blind
estimators so that the design can be reviewed before a new experiment is
licensed.

The model remains one pipeline:

    member anchor distribution
      -> partially pooled marked-AP composite score
      -> one KL-proximal update
      -> mass-controlled reverse-KL projection and Bayes action.

No learned gate, DANN branch, hierarchy head, spatial interpolator, affine
nuisance optimizer, or scan random effect is introduced.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as scipy_student_t


PROTOCOL_VERSION = "uji_hierarchical_marked_ensemble_v7_1_20260809"
VISIBLE_THRESHOLD = -0.999
DEFAULT_STUDENT_DF = 4.0
DEFAULT_MASS_TOLERANCE = 0.05
DEFAULT_CONCENTRATION_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_DF_GRID = (3.0, 4.0, 6.0, 8.0, 16.0)


def _positive_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    grid = tuple(sorted(set(map(float, values))))
    if not grid or any(not math.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError(f"{name} must contain positive finite values")
    return grid


def student_t_mad_unit(degrees_freedom: float) -> float:
    """MAD of a unit-scale centered Student-t distribution."""

    nu = float(degrees_freedom)
    if not math.isfinite(nu) or nu <= 2.0:
        raise ValueError("degrees_freedom must exceed two")
    value = float(scipy_student_t.ppf(0.75, df=nu))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("failed to obtain a Student-t MAD calibration")
    return value


def mad_to_student_scale(
    mad: torch.Tensor,
    *,
    degrees_freedom: float,
    minimum_scale: float = 0.025,
    maximum_scale: float = 0.75,
) -> torch.Tensor:
    """Convert robust MAD to the scale parameter used by Student-t."""

    if not mad.is_floating_point():
        raise TypeError("mad must be a floating tensor")
    lower = float(minimum_scale)
    upper = float(maximum_scale)
    if not 0.0 < lower < upper:
        raise ValueError("scale bounds must satisfy 0 < minimum < maximum")
    return (mad / student_t_mad_unit(degrees_freedom)).clamp(lower, upper)


def collect_equal_phone_marked_statistics(
    *,
    rss: torch.Tensor,
    phone_ids: Sequence[int] | np.ndarray | torch.Tensor,
    coordinate_keys: Sequence[object],
    training_indices: Sequence[int] | np.ndarray | torch.Tensor,
    anchor_keys: Sequence[object] | None = None,
) -> dict:
    """Collect marked-AP summaries without letting repeated scans define a device.

    A phone first produces one detection frequency, one conditional RSS median,
    and one conditional MAD at each anchor.  Anchor centers then average the
    phone-level centers and anchor detection rates average phone-level rates.
    Consequently, a phone with many repeated scans cannot dominate either
    quantity.  The scale value is the median phone-level absolute residual from
    the equal-phone anchor center, and its reliability is counted across visible
    devices rather than repeated scans.
    """

    if rss.ndim != 2 or not rss.is_floating_point():
        raise ValueError("rss must be floating [rows,aps]")
    rows, aps = rss.shape
    if len(coordinate_keys) != rows:
        raise ValueError("coordinate_keys length must equal rss rows")
    phone_array = np.asarray(
        phone_ids.detach().cpu().numpy()
        if isinstance(phone_ids, torch.Tensor)
        else phone_ids,
        dtype=np.int64,
    )
    if phone_array.shape != (rows,):
        raise ValueError("phone_ids must have shape [rows]")
    index_array = np.asarray(
        training_indices.detach().cpu().numpy()
        if isinstance(training_indices, torch.Tensor)
        else training_indices,
        dtype=np.int64,
    )
    if index_array.ndim != 1 or len(index_array) == 0:
        raise ValueError("training_indices must be a nonempty vector")
    if len(np.unique(index_array)) != len(index_array):
        raise ValueError("training_indices must not contain duplicates")
    if int(index_array.min()) < 0 or int(index_array.max()) >= rows:
        raise ValueError("training_indices contain an out-of-range row")

    if anchor_keys is None:
        selected_keys = {coordinate_keys[int(index)] for index in index_array}
        try:
            ordered_anchor_keys = tuple(sorted(selected_keys))
        except TypeError:
            ordered_anchor_keys = tuple(
                sorted(selected_keys, key=lambda value: repr(value))
            )
    else:
        ordered_anchor_keys = tuple(anchor_keys)
        if len(set(ordered_anchor_keys)) != len(ordered_anchor_keys):
            raise ValueError("anchor_keys must be unique")
    if not ordered_anchor_keys:
        raise ValueError("at least one anchor is required")
    key_to_anchor = {key: index for index, key in enumerate(ordered_anchor_keys)}
    grouped: list[dict[int, list[int]]] = [dict() for _ in ordered_anchor_keys]
    for row in index_array:
        key = coordinate_keys[int(row)]
        if key not in key_to_anchor:
            continue
        phone = int(phone_array[int(row)])
        grouped[key_to_anchor[key]].setdefault(phone, []).append(int(row))
    if any(not group for group in grouped):
        raise ValueError("every requested anchor must contain a training row")

    dtype = torch.float64
    local_detection = torch.empty((len(grouped), aps), dtype=dtype)
    local_center = torch.full((len(grouped), aps), float("nan"), dtype=dtype)
    local_mad = torch.full((len(grouped), aps), float("nan"), dtype=dtype)
    anchor_device_count = torch.empty(len(grouped), dtype=dtype)
    visible_device_count = torch.zeros((len(grouped), aps), dtype=dtype)
    scale_degrees_freedom = torch.zeros((len(grouped), aps), dtype=dtype)
    detection_blocks: list[torch.Tensor] = []
    summary_blocks: list[dict[str, torch.Tensor]] = []
    rss_cpu = rss.detach().cpu().double()

    for anchor, phone_rows in enumerate(grouped):
        detection_rows = []
        center_rows = []
        count_rows = []
        rss_rows = []
        for phone in sorted(phone_rows):
            values = rss_cpu[phone_rows[phone]]
            visible = values > VISIBLE_THRESHOLD
            count = visible.sum(dim=0).double()
            detection_rows.append(visible.double().mean(dim=0))
            center = torch.nanmedian(
                torch.where(visible, values, torch.full_like(values, float("nan"))),
                dim=0,
            ).values
            center_rows.append(center)
            count_rows.append(count)
            rss_rows.append(values)
        detection_block = torch.stack(detection_rows)
        center_block = torch.stack(center_rows)
        count_block = torch.stack(count_rows)
        detection_blocks.append(detection_block)
        anchor_device_count[anchor] = len(phone_rows)
        local_detection[anchor] = detection_block.mean(dim=0)
        center_valid = count_block > 0.0
        visible_device_count[anchor] = center_valid.sum(dim=0)
        local_center[anchor] = torch.where(
            center_valid,
            center_block,
            torch.zeros_like(center_block),
        ).sum(dim=0) / visible_device_count[anchor].clamp_min(1.0)
        local_center[anchor, visible_device_count[anchor] == 0.0] = float("nan")
        predictive_mad_block = torch.full_like(center_block, float("nan"))
        for phone_index, values in enumerate(rss_rows):
            visible = values > VISIBLE_THRESHOLD
            absolute_residual = torch.abs(values - local_center[anchor].unsqueeze(0))
            predictive_mad_block[phone_index] = torch.nanmedian(
                torch.where(
                    visible,
                    absolute_residual,
                    torch.full_like(absolute_residual, float("nan")),
                ),
                dim=0,
            ).values
        scale_valid = center_valid & torch.isfinite(predictive_mad_block)
        # The target is a previously unseen device.  Scale reliability is thus
        # the residual degrees of freedom across devices, not the raw number of
        # repeated scans from one device.
        scale_degrees_freedom[anchor] = (
            visible_device_count[anchor] - 1.0
        ).clamp_min(0.0)
        local_mad[anchor] = torch.nanmedian(
            torch.where(
                scale_valid,
                predictive_mad_block,
                torch.full_like(predictive_mad_block, float("nan")),
            ),
            dim=0,
        ).values
        summary_blocks.append(
            {
                "center": center_block,
                "mad": predictive_mad_block,
                "visible_count": count_block,
                "rss": tuple(rss_rows),
            }
        )

    return {
        "anchor_keys": ordered_anchor_keys,
        "local_detection_probability": local_detection,
        "local_center": local_center,
        "local_mad": local_mad,
        "anchor_device_count": anchor_device_count,
        "visible_device_count": visible_device_count,
        "scale_degrees_freedom": scale_degrees_freedom,
        "phone_detection_blocks": tuple(detection_blocks),
        "phone_summary_blocks": tuple(summary_blocks),
        "estimator": "scan-to-phone robust summary followed by equal-phone aggregation",
    }


def derive_ap_global_bases(
    *,
    local_detection_probability: torch.Tensor,
    local_center: torch.Tensor,
    local_mad: torch.Tensor,
    anchor_device_count: torch.Tensor,
    visible_device_count: torch.Tensor,
    scale_degrees_freedom: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Derive robust AP-level prior centers from anchor-level summaries."""

    shape = local_detection_probability.shape
    if len(shape) != 2:
        raise ValueError("local radio summaries must have shape [anchors,aps]")
    anchors, aps = shape
    if local_center.shape != shape or local_mad.shape != shape:
        raise ValueError("detection, center, and MAD shapes disagree")
    if visible_device_count.shape != shape or scale_degrees_freedom.shape != shape:
        raise ValueError("radio information-count shapes disagree")
    if anchor_device_count.shape != (anchors,):
        raise ValueError("anchor_device_count must have shape [anchors]")
    if not (
        local_detection_probability.is_floating_point()
        and local_center.is_floating_point()
        and local_mad.is_floating_point()
    ):
        raise TypeError("radio summaries must be floating tensors")

    weights = anchor_device_count.to(local_detection_probability).clamp_min(0.0)
    probability = torch.sum(
        weights[:, None] * local_detection_probability, dim=0
    ) / weights.sum().clamp_min(1.0)
    probability = probability.clamp(1e-5, 1.0 - 1e-5)
    center_global = torch.full(
        (aps,), -1.0, dtype=local_center.dtype, device=local_center.device
    )
    mad_global = torch.full(
        (aps,), 0.15, dtype=local_mad.dtype, device=local_mad.device
    )
    for ap in range(aps):
        center_valid = (visible_device_count[:, ap] > 0) & torch.isfinite(
            local_center[:, ap]
        )
        if bool(center_valid.any()):
            center_global[ap] = torch.median(local_center[center_valid, ap])
        scale_valid = (scale_degrees_freedom[:, ap] > 0) & torch.isfinite(
            local_mad[:, ap]
        ) & (local_mad[:, ap] > 0.0)
        if bool(scale_valid.any()):
            mad_global[ap] = torch.median(local_mad[scale_valid, ap])
    return {
        "detection_probability": probability,
        "rss_center": center_global,
        "rss_mad": mad_global.clamp(0.01, 0.75),
    }


def partially_pooled_marked_memory(
    *,
    local_detection_probability: torch.Tensor,
    local_center: torch.Tensor,
    local_mad: torch.Tensor,
    anchor_device_count: torch.Tensor,
    visible_device_count: torch.Tensor,
    scale_degrees_freedom: torch.Tensor,
    global_detection_probability: torch.Tensor,
    global_center: torch.Tensor,
    global_mad: torch.Tensor,
    detection_concentration: float,
    center_concentration: float,
    scale_concentration: float,
    degrees_freedom: float = DEFAULT_STUDENT_DF,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Construct one reliability-adaptive hierarchical marked-AP memory.

    The three concentrations are empirical-Bayes pseudo-information, not
    downstream localization knobs.  They must be selected with held-out-phone
    radio prediction before any localization-development outcome is inspected.
    """

    shape = local_detection_probability.shape
    if len(shape) != 2:
        raise ValueError("local summaries must have shape [anchors,aps]")
    anchors, aps = shape
    if any(
        tensor.shape != shape
        for tensor in (
            local_center,
            local_mad,
            visible_device_count,
            scale_degrees_freedom,
        )
    ):
        raise ValueError("local marked-memory shapes disagree")
    if anchor_device_count.shape != (anchors,):
        raise ValueError("anchor_device_count must have shape [anchors]")
    if any(
        tensor.shape != (aps,)
        for tensor in (global_detection_probability, global_center, global_mad)
    ):
        raise ValueError("global AP bases must have shape [aps]")
    k_rho, k_mu, k_sigma = _positive_grid(
        (detection_concentration,), name="detection_concentration"
    )[0], _positive_grid(
        (center_concentration,), name="center_concentration"
    )[0], _positive_grid(
        (scale_concentration,), name="scale_concentration"
    )[0]
    dtype = local_detection_probability.dtype
    device = local_detection_probability.device
    global_rho = global_detection_probability.to(device=device, dtype=dtype)
    global_mu = global_center.to(device=device, dtype=dtype)
    global_mad_value = global_mad.to(device=device, dtype=dtype)
    device_information = anchor_device_count.to(device=device, dtype=dtype)[:, None]
    visible_information = visible_device_count.to(device=device, dtype=dtype)
    scale_information = scale_degrees_freedom.to(device=device, dtype=dtype)

    rho = (
        device_information * local_detection_probability
        + k_rho * global_rho[None, :]
    ) / (device_information + k_rho)
    rho = rho.clamp(1e-5, 1.0 - 1e-5)

    local_mu = torch.where(
        (visible_information > 0.0) & torch.isfinite(local_center),
        local_center,
        global_mu[None, :],
    )
    mu_weight = visible_information / (visible_information + k_mu)
    mu = mu_weight * local_mu + (1.0 - mu_weight) * global_mu[None, :]

    local_scale = mad_to_student_scale(
        torch.where(
            (scale_information > 0.0) & torch.isfinite(local_mad) & (local_mad >= 0),
            local_mad,
            global_mad_value[None, :],
        ),
        degrees_freedom=degrees_freedom,
    )
    global_scale = mad_to_student_scale(
        global_mad_value, degrees_freedom=degrees_freedom
    )
    scale_weight = scale_information / (scale_information + k_sigma)
    log_scale = (
        scale_weight * torch.log(local_scale)
        + (1.0 - scale_weight) * torch.log(global_scale[None, :])
    )
    scale = torch.exp(log_scale)
    fields = {
        "detection_probability": rho,
        "rss_center": mu,
        "rss_scale": scale,
    }
    diagnostics = {
        "anchors": anchors,
        "aps": aps,
        "detection_concentration": k_rho,
        "center_concentration": k_mu,
        "scale_concentration": k_sigma,
        "degrees_freedom": float(degrees_freedom),
        "mean_center_local_weight": float(mu_weight.double().mean()),
        "mean_scale_local_weight": float(scale_weight.double().mean()),
        "zero_visible_device_fraction": float(
            (visible_information == 0.0).double().mean()
        ),
        "zero_scale_information_fraction": float(
            (scale_information == 0.0).double().mean()
        ),
        "estimator": "equal-phone hierarchical partial pooling",
    }
    return fields, diagnostics


def select_detection_concentration_leave_one_phone_out(
    phone_detection_blocks: Sequence[torch.Tensor],
    global_detection_probability: torch.Tensor,
    *,
    concentration_grid: Iterable[float] = DEFAULT_CONCENTRATION_GRID,
) -> dict:
    """Empirical-Bayes detection shrinkage from equal-phone predictive score.

    Each block is [phones, aps] and stores one phone's detection frequency in
    each AP at one anchor.  Every held-out phone contributes equal average
    Bernoulli score regardless of its number of repeated scans.
    """

    grid = _positive_grid(concentration_grid, name="concentration_grid")
    if global_detection_probability.ndim != 1:
        raise ValueError("global_detection_probability must have shape [aps]")
    aps = len(global_detection_probability)
    usable = [
        block.double()
        for block in phone_detection_blocks
        if block.ndim == 2 and block.shape[1] == aps and len(block) >= 2
    ]
    if not usable:
        raise ValueError("at least one anchor with two phones is required")
    global_rho = global_detection_probability.double().clamp(1e-6, 1 - 1e-6)
    records = []
    for concentration in grid:
        phone_scores = []
        for block in usable:
            total = block.sum(dim=0)
            denominator = float(len(block) - 1) + concentration
            for held_out in block:
                probability = (
                    total - held_out + concentration * global_rho
                ) / denominator
                probability = probability.clamp(1e-6, 1.0 - 1e-6)
                score = held_out * torch.log(probability) + (
                    1.0 - held_out
                ) * torch.log1p(-probability)
                phone_scores.append(float(score.mean()))
        values = np.asarray(phone_scores, dtype=np.float64)
        records.append(
            {
                "concentration": concentration,
                "equal_phone_mean_log_score": float(values.mean()),
                "equal_phone_standard_error": float(
                    values.std(ddof=1) / math.sqrt(len(values))
                )
                if len(values) > 1
                else 0.0,
                "held_out_phone_anchor_units": int(len(values)),
            }
        )
    selected = max(
        records,
        key=lambda row: (
            row["equal_phone_mean_log_score"],
            -row["concentration"],
        ),
    )
    return {
        "selected_concentration": float(selected["concentration"]),
        "selection_rule": "maximum equal-phone leave-one-phone-out Bernoulli log score",
        "grid": records,
    }


def select_mark_hierarchy_leave_one_phone_out(
    phone_summary_blocks: Sequence[dict],
    global_center: torch.Tensor,
    global_mad: torch.Tensor,
    *,
    center_concentration_grid: Iterable[float] = DEFAULT_CONCENTRATION_GRID,
    scale_concentration_grid: Iterable[float] = DEFAULT_CONCENTRATION_GRID,
    degrees_freedom_grid: Iterable[float] = DEFAULT_DF_GRID,
) -> dict:
    """Select center/scale pooling and one global tail shape without location.

    A block contains `center`, `mad`, and `visible_count`, each [phones,aps],
    plus one [scans,aps] RSS tensor per phone.  One phone is held out at a time.
    Its visible scan marks are averaged inside the phone before phones are
    averaged, so repeated scans do not let one device dominate.  This directly
    scores the conditional mark model and never sees localization error.
    """

    mu_grid = _positive_grid(
        center_concentration_grid, name="center_concentration_grid"
    )
    sigma_grid = _positive_grid(
        scale_concentration_grid, name="scale_concentration_grid"
    )
    df_grid = _positive_grid(degrees_freedom_grid, name="degrees_freedom_grid")
    if any(value <= 2.0 for value in df_grid):
        raise ValueError("all Student-t degrees of freedom must exceed two")
    if global_center.ndim != 1 or global_mad.shape != global_center.shape:
        raise ValueError("global center and MAD must have shape [aps]")
    aps = len(global_center)
    blocks = []
    for block in phone_summary_blocks:
        if not {"center", "mad", "visible_count", "rss"} <= set(block):
            raise ValueError(
                "phone summary blocks require center/mad/visible_count/rss"
            )
        center = block["center"].double()
        mad = block["mad"].double()
        count = block["visible_count"].double()
        if center.ndim != 2 or center.shape[1] != aps or mad.shape != center.shape or count.shape != center.shape:
            raise ValueError("phone summary tensors must share [phones,aps]")
        rss_rows = tuple(value.double() for value in block["rss"])
        if len(rss_rows) != len(center) or any(
            value.ndim != 2 or value.shape[1] != aps or len(value) == 0
            for value in rss_rows
        ):
            raise ValueError("rss must contain one nonempty [scans,aps] tensor per phone")
        if len(center) >= 2:
            blocks.append((center, mad, count, rss_rows))
    if not blocks:
        raise ValueError("at least one anchor with two phone summaries is required")
    global_mu = global_center.double()
    global_mad_value = global_mad.double().clamp_min(1e-6)
    units_by_center_concentration = {}
    for k_mu in mu_grid:
        units = []
        for center, _mad, count, rss_rows in blocks:
            valid_center = count > 0.0
            for held_index in range(len(center)):
                other = torch.ones(len(center), dtype=torch.bool)
                other[held_index] = False
                other_center_valid = valid_center[other]
                other_count = other_center_valid.sum(dim=0).double()
                other_center_sum = torch.where(
                    other_center_valid,
                    center[other],
                    torch.zeros_like(center[other]),
                ).sum(dim=0)
                predicted_center = (
                    other_center_sum + k_mu * global_mu
                ) / (other_count + k_mu)
                loo_mad_rows = []
                for other_index in torch.nonzero(
                    other, as_tuple=False
                ).flatten().tolist():
                    other_rss = rss_rows[int(other_index)]
                    other_visible = other_rss > VISIBLE_THRESHOLD
                    absolute_residual = torch.abs(
                        other_rss - predicted_center.unsqueeze(0)
                    )
                    loo_mad_rows.append(
                        torch.nanmedian(
                            torch.where(
                                other_visible,
                                absolute_residual,
                                torch.full_like(absolute_residual, float("nan")),
                            ),
                            dim=0,
                        ).values
                    )
                loo_mad = torch.stack(loo_mad_rows)
                other_scale_valid = other_center_valid & torch.isfinite(loo_mad)
                scale_df = (
                    other_scale_valid.sum(dim=0).double() - 1.0
                ).clamp_min(0.0)
                local_mad = torch.nanmedian(
                    torch.where(
                        other_scale_valid,
                        loo_mad,
                        torch.full_like(loo_mad, float("nan")),
                    ),
                    dim=0,
                ).values
                local_mad = torch.where(
                    torch.isfinite(local_mad), local_mad, global_mad_value
                )
                held_rss = rss_rows[held_index]
                held_visible = held_rss > VISIBLE_THRESHOLD
                visible_location = torch.nonzero(held_visible, as_tuple=False)
                if len(visible_location) == 0:
                    continue
                held_ap = visible_location[:, 1]
                units.append(
                    {
                        "center": predicted_center,
                        "local_mad": local_mad,
                        "scale_df": scale_df,
                        "held_ap": held_ap,
                        "held_value": held_rss[held_visible],
                    }
                )
        if not units:
            raise RuntimeError("no held-out visible phone marks were available")
        units_by_center_concentration[k_mu] = units

    records = []
    for nu in df_grid:
        global_scale = mad_to_student_scale(
            global_mad_value, degrees_freedom=nu
        ).double()
        constant = (
            math.lgamma(0.5 * (nu + 1.0))
            - math.lgamma(0.5 * nu)
            - 0.5 * math.log(nu * math.pi)
        )
        for k_mu in mu_grid:
            prepared_units = []
            for unit in units_by_center_concentration[k_mu]:
                prepared_units.append(
                    {
                        **unit,
                        "local_scale": mad_to_student_scale(
                            unit["local_mad"], degrees_freedom=nu
                        ).double(),
                    }
                )
            for k_sigma in sigma_grid:
                unit_scores = []
                for unit in prepared_units:
                    scale_weight = unit["scale_df"] / (
                        unit["scale_df"] + k_sigma
                    )
                    predicted_scale = torch.exp(
                        scale_weight * torch.log(unit["local_scale"])
                        + (1.0 - scale_weight) * torch.log(global_scale)
                    )
                    selected_scale = predicted_scale[unit["held_ap"]]
                    residual = (
                        unit["held_value"] - unit["center"][unit["held_ap"]]
                    ) / selected_scale
                    log_density = (
                        constant
                        - torch.log(selected_scale)
                        - 0.5
                        * (nu + 1.0)
                        * torch.log1p(residual.square() / nu)
                    )
                    unit_scores.append(float(log_density.mean()))
                values = np.asarray(unit_scores, dtype=np.float64)
                if len(values) == 0:
                    raise RuntimeError("no held-out visible phone marks were scored")
                records.append(
                    {
                        "center_concentration": k_mu,
                        "scale_concentration": k_sigma,
                        "degrees_freedom": nu,
                        "equal_phone_mean_log_score": float(values.mean()),
                        "equal_phone_standard_error": float(
                            values.std(ddof=1) / math.sqrt(len(values))
                        )
                        if len(values) > 1
                        else 0.0,
                        "held_out_phone_anchor_units": int(len(values)),
                    }
                )
    selected = max(
        records,
        key=lambda row: (
            row["equal_phone_mean_log_score"],
            -row["center_concentration"],
            -row["scale_concentration"],
            -abs(math.log(row["degrees_freedom"] / DEFAULT_STUDENT_DF)),
        ),
    )
    return {
        "selected_center_concentration": float(selected["center_concentration"]),
        "selected_scale_concentration": float(selected["scale_concentration"]),
        "selected_degrees_freedom": float(selected["degrees_freedom"]),
        "selection_rule": "maximum equal-phone leave-one-phone-out Student-t visible-mark log score",
        "grid": records,
    }


def fit_hierarchical_marked_memory(
    *,
    rss: torch.Tensor,
    phone_ids: Sequence[int] | np.ndarray | torch.Tensor,
    coordinate_keys: Sequence[object],
    training_indices: Sequence[int] | np.ndarray | torch.Tensor,
    anchor_keys: Sequence[object] | None = None,
    detection_concentration_grid: Iterable[float] = DEFAULT_CONCENTRATION_GRID,
    center_concentration_grid: Iterable[float] = DEFAULT_CONCENTRATION_GRID,
    scale_concentration_grid: Iterable[float] = DEFAULT_CONCENTRATION_GRID,
    degrees_freedom_grid: Iterable[float] = DEFAULT_DF_GRID,
) -> dict:
    """Fit the complete V7.1 memory using radio observations and phone IDs only."""

    statistics = collect_equal_phone_marked_statistics(
        rss=rss,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        training_indices=training_indices,
        anchor_keys=anchor_keys,
    )
    base_arguments = {
        name: statistics[name]
        for name in (
            "local_detection_probability",
            "local_center",
            "local_mad",
            "anchor_device_count",
            "visible_device_count",
            "scale_degrees_freedom",
        )
    }
    global_bases = derive_ap_global_bases(**base_arguments)
    detection_selection = select_detection_concentration_leave_one_phone_out(
        statistics["phone_detection_blocks"],
        global_bases["detection_probability"],
        concentration_grid=detection_concentration_grid,
    )
    mark_selection = select_mark_hierarchy_leave_one_phone_out(
        statistics["phone_summary_blocks"],
        global_bases["rss_center"],
        global_bases["rss_mad"],
        center_concentration_grid=center_concentration_grid,
        scale_concentration_grid=scale_concentration_grid,
        degrees_freedom_grid=degrees_freedom_grid,
    )
    fields, diagnostics = partially_pooled_marked_memory(
        **base_arguments,
        global_detection_probability=global_bases["detection_probability"],
        global_center=global_bases["rss_center"],
        global_mad=global_bases["rss_mad"],
        detection_concentration=detection_selection["selected_concentration"],
        center_concentration=mark_selection["selected_center_concentration"],
        scale_concentration=mark_selection["selected_scale_concentration"],
        degrees_freedom=mark_selection["selected_degrees_freedom"],
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "fields": fields,
        "degrees_freedom": mark_selection["selected_degrees_freedom"],
        "anchor_keys": statistics["anchor_keys"],
        "global_bases": global_bases,
        "radio_only_selection": {
            "detection": detection_selection,
            "conditional_mark": mark_selection,
        },
        "information_counts": {
            "anchor_device_count": statistics["anchor_device_count"],
            "visible_device_count": statistics["visible_device_count"],
            "scale_degrees_freedom": statistics["scale_degrees_freedom"],
        },
        "diagnostics": diagnostics,
        "outcome_access": "none: no coordinates or localization errors enter this fit",
    }


def marked_marginal_composite_loss(
    query: torch.Tensor,
    detection_probability: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    *,
    degrees_freedom: float = DEFAULT_STUDENT_DF,
    candidate_chunk_size: int = 128,
) -> torch.Tensor:
    """Average marginal marked-AP proper score, returned as [rows,anchors].

    This is deliberately a composite score.  It does not assert conditional AP
    independence, so no scan-correlation component is required merely to make
    the KL-proximal decision update coherent.
    """

    if query.ndim != 2 or not query.is_floating_point():
        raise ValueError("query must be floating [rows,aps]")
    if detection_probability.ndim != 2 or detection_probability.shape != rss_center.shape:
        raise ValueError("detection probability and center must be [anchors,aps]")
    anchors, aps = detection_probability.shape
    if query.shape[1] != aps:
        raise ValueError("query and marked memory AP dimensions disagree")
    if rss_scale.shape not in {(aps,), (anchors, aps)}:
        raise ValueError("rss_scale must be [aps] or [anchors,aps]")
    if int(candidate_chunk_size) <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    nu = float(degrees_freedom)
    if not math.isfinite(nu) or nu <= 2.0:
        raise ValueError("degrees_freedom must exceed two")
    visible = query > VISIBLE_THRESHOLD
    observed = visible.to(query.dtype).unsqueeze(1)
    hidden = (~visible).to(query.dtype).unsqueeze(1)
    amplitude_constant = (
        math.lgamma(0.5 * nu)
        + 0.5 * math.log(nu * math.pi)
        - math.lgamma(0.5 * (nu + 1.0))
    )
    outputs = []
    for start in range(0, anchors, int(candidate_chunk_size)):
        stop = min(start + int(candidate_chunk_size), anchors)
        probability = detection_probability[start:stop].to(
            device=query.device, dtype=query.dtype
        ).clamp(1e-6, 1.0 - 1e-6)
        eta = torch.logit(probability)
        detection_log_likelihood = (
            observed * F.logsigmoid(eta).unsqueeze(0)
            + hidden * F.logsigmoid(-eta).unsqueeze(0)
        ).sum(dim=2)
        center = rss_center[start:stop].to(device=query.device, dtype=query.dtype)
        if rss_scale.ndim == 1:
            scale = rss_scale.to(device=query.device, dtype=query.dtype).view(1, -1)
        else:
            scale = rss_scale[start:stop].to(
                device=query.device, dtype=query.dtype
            )
        scale = scale.clamp_min(1e-5)
        standardized = (
            query.unsqueeze(1) - center.unsqueeze(0)
        ) / scale.unsqueeze(0)
        amplitude_nll = (
            observed
            * (
                amplitude_constant
                + torch.log(scale).unsqueeze(0)
                + 0.5
                * (nu + 1.0)
                * torch.log1p(standardized.square() / nu)
            )
        ).sum(dim=2)
        outputs.append(
            (-detection_log_likelihood + amplitude_nll) / float(aps)
        )
    result = torch.cat(outputs, dim=1)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("marked composite score contains non-finite values")
    return result


def joint_kl_proximal_update(
    member_logits: torch.Tensor,
    marked_loss: torch.Tensor,
    *,
    lambda_value: float,
    member_indices: Sequence[int] | None = None,
) -> dict[str, torch.Tensor | float | tuple[int, ...]]:
    """Apply one entropic decision update on the member-anchor joint space.

    The marked score is a second view of the same scan, so this is deliberately
    described as a KL-proximal/generalized-Bayes update rather than multiplication
    of conditionally independent likelihoods.
    """

    if member_logits.ndim != 3 or not member_logits.is_floating_point():
        raise ValueError("member_logits must be floating [members,rows,anchors]")
    total_members, rows, anchors = member_logits.shape
    if marked_loss.shape != (rows, anchors) or not marked_loss.is_floating_point():
        raise ValueError("marked_loss must be floating [rows,anchors]")
    temperature = float(lambda_value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("lambda_value must be positive and finite")
    if member_indices is None:
        selected_indices = tuple(range(total_members))
    else:
        selected_indices = tuple(map(int, member_indices))
        if not selected_indices or len(set(selected_indices)) != len(selected_indices):
            raise ValueError("member_indices must be nonempty and unique")
        if min(selected_indices) < 0 or max(selected_indices) >= total_members:
            raise ValueError("member_indices contain an out-of-range member")
    selected = member_logits[list(selected_indices)]
    members = len(selected_indices)
    member_log_base = F.log_softmax(selected, dim=2) - math.log(float(members))
    joint_log_unnormalized = member_log_base - marked_loss.to(selected).unsqueeze(0) / temperature
    joint_probability = torch.softmax(
        joint_log_unnormalized.permute(1, 0, 2).reshape(rows, -1), dim=1
    ).reshape(rows, members, anchors)
    anchor_probability = joint_probability.sum(dim=1)
    member_probability = joint_probability.sum(dim=2)
    return {
        "joint_probability": joint_probability,
        "anchor_probability": anchor_probability,
        "member_probability": member_probability,
        "member_indices": selected_indices,
        "lambda": temperature,
    }


def mass_constrained_reverse_kl_projection(
    probability: torch.Tensor,
    *,
    epsilon: float = DEFAULT_MASS_TOLERANCE,
) -> dict[str, torch.Tensor | float]:
    """Smallest-support reverse-KL projection retaining at least 1-epsilon.

    For each row, the selected support is the set of largest atoms whose mass
    first reaches 1-epsilon.  Renormalization is the exact reverse-KL projection
    on that support, with KL bounded by -log(1-epsilon).
    """

    if probability.ndim != 2 or not probability.is_floating_point():
        raise ValueError("probability must be floating [rows,atoms]")
    tolerance = float(epsilon)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("epsilon must lie in (0,1)")
    if bool((probability < 0.0).any()) or not bool(torch.isfinite(probability).all()):
        raise ValueError("probability must be finite and nonnegative")
    row_mass = probability.sum(dim=1, keepdim=True)
    if bool((row_mass <= 0.0).any()):
        raise ValueError("every probability row must have positive mass")
    normalized = probability / row_mass
    sorted_probability, sorted_index = torch.sort(
        normalized, dim=1, descending=True
    )
    cumulative = torch.cumsum(sorted_probability, dim=1)
    support_size = (cumulative < (1.0 - tolerance)).sum(dim=1) + 1
    ranks = torch.arange(
        normalized.shape[1], device=normalized.device
    ).unsqueeze(0)
    sorted_mask = ranks < support_size.unsqueeze(1)
    retained_sorted = torch.where(
        sorted_mask, sorted_probability, torch.zeros_like(sorted_probability)
    )
    retained_mass = retained_sorted.sum(dim=1)
    projected_sorted = retained_sorted / retained_mass.unsqueeze(1)
    projected = torch.zeros_like(normalized)
    projected.scatter_(1, sorted_index, projected_sorted)
    reverse_kl = -torch.log(retained_mass)
    bound = -math.log(1.0 - tolerance)
    if bool((retained_mass < 1.0 - tolerance - 1e-6).any()):
        raise RuntimeError("mass-constrained support failed its retention target")
    if bool((reverse_kl > bound + 1e-6).any()):
        raise RuntimeError("reverse-KL approximation bound was violated")
    return {
        "probability": projected,
        "support_size": support_size,
        "retained_mass": retained_mass,
        "reverse_kl": reverse_kl,
        "reverse_kl_bound": bound,
        "epsilon": tolerance,
    }


def project_joint_by_anchor_mass(
    joint_probability: torch.Tensor,
    *,
    epsilon: float = DEFAULT_MASS_TOLERANCE,
) -> dict[str, torch.Tensor | float]:
    """Project a member-anchor joint distribution using its anchor marginal."""

    if joint_probability.ndim != 3 or not joint_probability.is_floating_point():
        raise ValueError("joint_probability must be floating [rows,members,anchors]")
    if bool((joint_probability < 0.0).any()) or not bool(
        torch.isfinite(joint_probability).all()
    ):
        raise ValueError("joint_probability must be finite and nonnegative")
    joint = joint_probability / joint_probability.sum(
        dim=(1, 2), keepdim=True
    ).clamp_min(1e-12)
    anchor = joint.sum(dim=1)
    projection = mass_constrained_reverse_kl_projection(anchor, epsilon=epsilon)
    anchor_mask = projection["probability"] > 0.0
    projected_joint = torch.where(
        anchor_mask.unsqueeze(1), joint, torch.zeros_like(joint)
    )
    projected_joint = projected_joint / projected_joint.sum(
        dim=(1, 2), keepdim=True
    ).clamp_min(1e-12)
    if not torch.allclose(
        projected_joint.sum(dim=1), projection["probability"], atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("joint and anchor projections disagree")
    return {
        **projection,
        "joint_probability": projected_joint,
        "anchor_mask": anchor_mask,
    }


def weighted_geometric_median(
    probability: torch.Tensor,
    support: torch.Tensor,
    *,
    maximum_iterations: int = 64,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Weighted Weiszfeld Bayes action for Euclidean-distance loss."""

    if probability.ndim != 2 or support.ndim != 2:
        raise ValueError("probability and support must be [rows,atoms] and [atoms,d]")
    if probability.shape[1] != support.shape[0]:
        raise ValueError("probability and support atom counts disagree")
    if not probability.is_floating_point() or not support.is_floating_point():
        raise TypeError("probability and support must be floating")
    if int(maximum_iterations) <= 0 or float(tolerance) <= 0.0:
        raise ValueError("iteration limit and tolerance must be positive")
    if bool((probability < 0.0).any()) or not bool(torch.isfinite(probability).all()):
        raise ValueError("probability must be finite and nonnegative")
    row_mass = probability.sum(dim=1, keepdim=True)
    if bool((row_mass <= 0.0).any()):
        raise ValueError("every probability row must have positive mass")
    weights = probability / row_mass
    points = support.to(device=probability.device, dtype=probability.dtype)
    action = weights @ points
    for _iteration in range(int(maximum_iterations)):
        distance = torch.linalg.vector_norm(
            action.unsqueeze(1) - points.unsqueeze(0), dim=2
        ).clamp_min(1e-7)
        inverse = weights / distance
        updated = (inverse @ points) / inverse.sum(dim=1, keepdim=True).clamp_min(1e-12)
        maximum_change = torch.linalg.vector_norm(updated - action, dim=1).max()
        action = updated
        if float(maximum_change.detach().cpu()) <= float(tolerance):
            break
    # The ordinary Weiszfeld iteration is singular at a support atom.  Enforce
    # the exact subgradient condition there so dominant atoms are returned
    # exactly rather than as an epsilon-dependent nearby point.
    pair_delta = points.unsqueeze(0) - points.unsqueeze(1)
    pair_distance = torch.linalg.vector_norm(pair_delta, dim=2)
    safe_distance = pair_distance.clamp_min(1e-12)
    directions = pair_delta / safe_distance.unsqueeze(2)
    directions[pair_distance == 0.0] = 0.0
    atom_subgradient = torch.einsum("ri,ijd->rjd", weights, directions)
    atom_optimal = torch.linalg.vector_norm(atom_subgradient, dim=2) <= weights + 1e-7
    for row in range(len(action)):
        candidates = torch.nonzero(atom_optimal[row], as_tuple=False).flatten()
        if len(candidates) > 0:
            action[row] = points[int(candidates[0])]
    return action


def bayes_point_action(
    probability: torch.Tensor,
    support: torch.Tensor,
    *,
    loss: str,
) -> torch.Tensor:
    if loss == "squared_euclidean":
        normalized = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return normalized @ support.to(probability)
    if loss == "euclidean":
        return weighted_geometric_median(probability, support)
    raise ValueError("loss must be squared_euclidean or euclidean")


def joint_covariance_decomposition(
    joint_probability: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Decompose spatial covariance into candidate and member disagreement.

    The returned identity is

        total = within_member_candidate + between_member.

    Keeping the joint member-candidate distribution is what makes the second
    term identifiable; collapsing members first would erase it.
    """

    if joint_probability.ndim != 3 or support.ndim != 2:
        raise ValueError("joint_probability and support must be [R,E,A] and [A,d]")
    if joint_probability.shape[2] != support.shape[0]:
        raise ValueError("joint probability and support anchor counts disagree")
    if not joint_probability.is_floating_point() or not support.is_floating_point():
        raise TypeError("joint_probability and support must be floating")
    if bool((joint_probability < 0.0).any()) or not bool(
        torch.isfinite(joint_probability).all()
    ):
        raise ValueError("joint_probability must be finite and nonnegative")
    total_mass = joint_probability.sum(dim=(1, 2), keepdim=True)
    if bool((total_mass <= 0.0).any()):
        raise ValueError("every joint row must have positive mass")
    joint = joint_probability / total_mass
    points = support.to(joint)
    member_mass = joint.sum(dim=2)
    conditional = joint / member_mass.unsqueeze(2).clamp_min(1e-12)
    member_mean = torch.einsum("rea,ad->red", conditional, points)
    overall_mean = torch.einsum("re,red->rd", member_mass, member_mean)
    candidate_delta = points[None, None, :, :] - member_mean[:, :, None, :]
    within = torch.einsum(
        "rea,reai,reaj->rij", joint, candidate_delta, candidate_delta
    )
    member_delta = member_mean - overall_mean[:, None, :]
    between = torch.einsum(
        "re,rei,rej->rij", member_mass, member_delta, member_delta
    )
    total_delta = points[None, :, :] - overall_mean[:, None, :]
    anchor_probability = joint.sum(dim=1)
    total = torch.einsum(
        "ra,rai,raj->rij", anchor_probability, total_delta, total_delta
    )
    residual = total - within - between
    return {
        "within_member_candidate": within,
        "between_member": between,
        "total": total,
        "identity_residual": residual,
        "overall_mean": overall_mean,
        "member_mean": member_mean,
        "member_probability": member_mass,
        "conditional_anchor_probability": conditional,
    }


def balanced_member_subsets(
    total_members: int,
    subset_size: int,
    *,
    maximum_subsets: int | None = None,
    seed: int = 7511,
) -> tuple[tuple[int, ...], ...]:
    """Outcome-blind balanced subsets for an ensemble-size information curve.

    With the registered eight-member pool, all C(8,E) subsets are inexpensive
    and exactly balanced.  The deterministic greedy branch is available only
    for larger future pools where exhaustive enumeration is impractical.
    """

    members = int(total_members)
    size = int(subset_size)
    if not 1 <= size <= members:
        raise ValueError("subset_size must lie in [1,total_members]")
    candidates = list(itertools.combinations(range(members), size))
    if maximum_subsets is None or int(maximum_subsets) >= len(candidates):
        return tuple(candidates)
    limit = int(maximum_subsets)
    if limit <= 0:
        raise ValueError("maximum_subsets must be positive")

    def digest(subset: tuple[int, ...]) -> bytes:
        return hashlib.sha256(
            f"{int(seed)}|v71-balanced-subset|{subset!r}".encode("utf-8")
        ).digest()

    remaining = sorted(candidates, key=digest)
    selected: list[tuple[int, ...]] = []
    member_count = np.zeros(members, dtype=np.int64)
    pair_count = np.zeros((members, members), dtype=np.int64)
    while len(selected) < limit:
        best = min(
            remaining,
            key=lambda subset: (
                np.var(
                    member_count
                    + np.asarray([index in subset for index in range(members)])
                ),
                np.var(
                    pair_count
                    + np.asarray(
                        [
                            [i in subset and j in subset for j in range(members)]
                            for i in range(members)
                        ]
                    )
                ),
                digest(subset),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        for member in best:
            member_count[member] += 1
        for first in best:
            for second in best:
                pair_count[first, second] += 1
    return tuple(selected)


def summarize_member_subset_curve(records: Sequence[dict]) -> list[dict]:
    """Summarize subset variation instead of treating one seed prefix as truth."""

    grouped: dict[int, list[dict]] = {}
    for record in records:
        if not {"E", "subset", "mae_m", "energy_m"} <= set(record):
            raise ValueError("subset records require E/subset/mae_m/energy_m")
        subset = tuple(map(int, record["subset"]))
        E = int(record["E"])
        if len(subset) != E:
            raise ValueError("record E does not match its subset")
        grouped.setdefault(E, []).append(record)
    result = []
    for E, rows in sorted(grouped.items()):
        mae = np.asarray([float(row["mae_m"]) for row in rows], dtype=np.float64)
        energy = np.asarray([float(row["energy_m"]) for row in rows], dtype=np.float64)
        result.append(
            {
                "E": E,
                "subsets": len(rows),
                "mae_mean_m": float(mae.mean()),
                "mae_standard_deviation_m": float(mae.std(ddof=1)) if len(mae) > 1 else 0.0,
                "mae_range_m": [float(mae.min()), float(mae.max())],
                "energy_mean_m": float(energy.mean()),
                "energy_standard_deviation_m": float(energy.std(ddof=1)) if len(energy) > 1 else 0.0,
                "energy_range_m": [float(energy.min()), float(energy.max())],
                "interpretation": "member-count information averaged over outcome-blind subsets",
            }
        )
    return result


def select_member_count_and_lambda_one_standard_error(
    records: Sequence[dict],
    *,
    minimum_members: int = 2,
) -> dict:
    """Jointly select E and lambda without promoting member identity to a knob.

    Every record contains coordinate-level energy scores for one outcome-blind
    member subset.  Scores are first averaged over subsets for each coordinate;
    overlapping subsets therefore do not masquerade as independent samples.
    The smallest E within one coordinate-level standard error of the best pair
    is selected.  Within that E, lambda retains its proper-score ordering.
    """

    required = {"E", "subset", "lambda", "coordinate_energy_m"}
    grouped: dict[tuple[int, float], list[tuple[tuple[int, ...], np.ndarray]]] = {}
    coordinate_count: int | None = None
    for record in records:
        if not required <= set(record):
            raise ValueError(
                "selection records require E/subset/lambda/coordinate_energy_m"
            )
        E = int(record["E"])
        subset = tuple(map(int, record["subset"]))
        lambda_value = float(record["lambda"])
        if len(subset) != E or len(set(subset)) != E:
            raise ValueError("record subset must contain E unique members")
        if not math.isfinite(lambda_value) or lambda_value <= 0.0:
            raise ValueError("record lambda must be positive and finite")
        energy = np.asarray(record["coordinate_energy_m"], dtype=np.float64)
        if energy.ndim != 1 or len(energy) == 0 or not np.isfinite(energy).all():
            raise ValueError("coordinate_energy_m must be a finite nonempty vector")
        if coordinate_count is None:
            coordinate_count = len(energy)
        elif len(energy) != coordinate_count:
            raise ValueError("all configurations must score the same coordinates")
        grouped.setdefault((E, lambda_value), []).append((subset, energy))
    if not grouped:
        raise ValueError("at least one selection record is required")
    if int(minimum_members) < 2:
        raise ValueError("minimum_members must be at least two")

    subsets_by_E: dict[int, set[tuple[int, ...]]] = {}
    for (E, _lambda), rows in grouped.items():
        subsets = [subset for subset, _energy in rows]
        if len(set(subsets)) != len(subsets):
            raise ValueError("a member subset is duplicated within an E-lambda pair")
        current = set(subsets)
        if E in subsets_by_E and current != subsets_by_E[E]:
            raise ValueError("every lambda must use the same member subsets at fixed E")
        subsets_by_E[E] = current

    grid = []
    for (E, lambda_value), rows in sorted(grouped.items()):
        coordinate_average = np.stack([energy for _subset, energy in rows]).mean(axis=0)
        standard_error = (
            float(coordinate_average.std(ddof=1) / math.sqrt(len(coordinate_average)))
            if len(coordinate_average) > 1
            else 0.0
        )
        grid.append(
            {
                "E": E,
                "lambda": lambda_value,
                "balanced_subsets": len(rows),
                "coordinate_count": int(len(coordinate_average)),
                "mean_energy_m": float(coordinate_average.mean()),
                "coordinate_standard_error_m": standard_error,
            }
        )
    eligible_grid = [row for row in grid if row["E"] >= int(minimum_members)]
    if not eligible_grid:
        raise ValueError("no E satisfies minimum_members")
    best = min(
        eligible_grid,
        key=lambda row: (row["mean_energy_m"], row["E"], -row["lambda"]),
    )
    threshold = best["mean_energy_m"] + best["coordinate_standard_error_m"]
    saturated = [row for row in eligible_grid if row["mean_energy_m"] <= threshold]
    selected_E = min(row["E"] for row in saturated)
    selected = min(
        (row for row in saturated if row["E"] == selected_E),
        key=lambda row: (row["mean_energy_m"], -row["lambda"]),
    )
    return {
        "selected_E": int(selected["E"]),
        "selected_lambda": float(selected["lambda"]),
        "best_mean_pair": {"E": int(best["E"]), "lambda": float(best["lambda"])},
        "one_standard_error_threshold_m": float(threshold),
        "minimum_members": int(minimum_members),
        "selection_rule": (
            "average over outcome-blind subsets per coordinate; choose the "
            "smallest E within one best-pair coordinate SE, then the lowest "
            "energy lambda at that E"
        ),
        "grid": grid,
    }


def v71_model_contract() -> dict:
    """Machine-readable contract for review before any new empirical run."""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "front": "exchangeable independent-seed anchor distributions",
        "member_base": "uniform p(m,a|x)=r_m(a|x)/E",
        "marked_memory": {
            "occurrence": "anchor-AP Bernoulli probability with empirical-Bayes device pooling",
            "center": "anchor-AP RSS center with visible-device reliability pooling",
            "scale": "anchor-AP Student-t scale with residual-df reliability pooling",
            "tail": "one global Student-t degrees of freedom selected by radio-only held-out-phone score",
        },
        "score": "average marginal marked-AP proper score; no AP-independence claim",
        "update": "one KL-proximal q proportional to uniform member base times exp(-loss/lambda)",
        "projection": (
            "smallest support retaining 1-epsilon mass; exact reverse-KL "
            "renormalization with explicit approximation bound"
        ),
        "point_action": {
            "MAE": "weighted geometric median",
            "squared_error": "posterior mean",
        },
        "member_count_audit": "balanced outcome-blind subsets, not one canonical prefix curve",
        "localization_selection": (
            "joint (E,lambda) coordinate-energy one-standard-error rule; E>=2"
        ),
        "radio_only_parameters": ["kappa_rho", "kappa_mu", "kappa_sigma", "nu"],
        "retained_global_parameters": ["E", "lambda"],
        "approximation_tolerance": "epsilon, fixed before localization selection",
        "excluded_components": [
            "learned member gate",
            "DANN",
            "hierarchy head",
            "scan random effect",
            "affine nuisance optimizer",
            "FEM/IDW extension",
            "residual coordinate head",
        ],
        "protected_role_policy": (
            "V7 protected roles are already exposed and cannot confirm V7.1; "
            "a fresh external confirmation is required"
        ),
    }
