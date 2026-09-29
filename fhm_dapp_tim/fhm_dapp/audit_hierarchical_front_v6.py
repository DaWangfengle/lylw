"""Development-only audit of a topology-aware hierarchical front objective.

The experiment keeps the sparse survey, anchor dictionary, model architecture,
marked AP likelihood, KL update, and calibration protocol fixed.  It changes
only the proper scoring rule used to train the front posterior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords, resolve_path
from fhm_dapp_tim.fhm_dapp.audit_bplus_localization_bridge_v5 import (
    _ap_fallbacks,
    _append_rows,
    _distribution_rows,
    _finish_rows,
    _paired_coordinate_delta,
    _parameter_label,
    marked_hurdle_loss,
    summarize_rows,
)
from fhm_dapp_tim.fhm_dapp.audit_bplus_sparse50_fem_v5 import (
    _rows_for_keys,
    geometry_support_with_hull,
    internal_support_roles,
)
from fhm_dapp_tim.fhm_dapp.audit_sparse_radiomap_v5 import (
    LocationKey,
    _global_detection_background,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.protocol_v4 import (
    coordinate_axis_partition,
    validate_registry,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v4 import (
    FROZEN_MEMBER_V4,
    _build_model_and_memory,
    _configure_cuda_reproducibility,
    _fit_fixed,
    _fit_schedule,
    _set_seed,
    _training_loader,
    evaluate_member_distribution,
)


OBJECTIVES = ("joint_3d_energy", "hierarchical_topology")


def _csv_tuple(text: str, cast) -> tuple:
    values = tuple(cast(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


@dataclass(frozen=True)
class HierarchicalScoreGeometry:
    strata: tuple[tuple[int, int], ...]
    anchor_stratum: torch.Tensor
    anchor_indices: tuple[torch.Tensor, ...]
    pair_distance_m: tuple[torch.Tensor, ...]
    distance_scale_m: torch.Tensor

    def validate(self, anchors: int) -> None:
        if self.anchor_stratum.shape != (anchors,):
            raise ValueError("anchor_stratum has the wrong shape")
        if len(self.strata) != len(self.anchor_indices):
            raise ValueError("hierarchical strata and index blocks disagree")
        if len(self.strata) != len(self.pair_distance_m):
            raise ValueError("hierarchical strata and distance blocks disagree")
        if self.distance_scale_m.shape != (len(self.strata),):
            raise ValueError("distance scales have the wrong shape")
        if torch.any(self.distance_scale_m <= 0.0):
            raise ValueError("all within-stratum distance scales must be positive")
        covered = torch.cat(self.anchor_indices).sort().values
        expected = torch.arange(anchors, device=covered.device)
        if not torch.equal(covered, expected):
            raise ValueError("hierarchical strata do not partition the anchors")


def build_hierarchical_score_geometry(
    anchor_coords: torch.Tensor,
    anchor_buildings: torch.Tensor,
    anchor_floors: torch.Tensor,
) -> HierarchicalScoreGeometry:
    """Construct the fixed disjoint-floor geometry used by the proper score."""

    if anchor_coords.ndim != 2 or anchor_coords.shape[1] != 2:
        raise ValueError("anchor coordinates must have shape [A,2]")
    anchors = len(anchor_coords)
    if anchor_buildings.shape != (anchors,) or anchor_floors.shape != (anchors,):
        raise ValueError("anchor hierarchy arrays have the wrong shape")
    strata = tuple(
        sorted(
            {
                (int(building), int(floor))
                for building, floor in zip(
                    anchor_buildings.detach().cpu().tolist(),
                    anchor_floors.detach().cpu().tolist(),
                )
            }
        )
    )
    if not strata:
        raise ValueError("at least one hierarchy stratum is required")
    anchor_stratum = torch.full(
        (anchors,), -1, device=anchor_coords.device, dtype=torch.long
    )
    index_blocks = []
    pair_blocks = []
    scales = []
    for stratum_index, (building, floor) in enumerate(strata):
        indices = torch.nonzero(
            (anchor_buildings == int(building)) & (anchor_floors == int(floor)),
            as_tuple=False,
        ).flatten()
        if len(indices) == 0:
            raise RuntimeError("constructed an empty hierarchy stratum")
        anchor_stratum[indices] = int(stratum_index)
        pair = torch.cdist(anchor_coords[indices], anchor_coords[indices], p=2)
        nonzero = pair[pair > 1e-7]
        scale = nonzero.mean() if len(nonzero) else pair.new_tensor(1.0)
        index_blocks.append(indices)
        pair_blocks.append(pair)
        scales.append(scale)
    geometry = HierarchicalScoreGeometry(
        strata=strata,
        anchor_stratum=anchor_stratum,
        anchor_indices=tuple(index_blocks),
        pair_distance_m=tuple(pair_blocks),
        distance_scale_m=torch.stack(scales),
    )
    geometry.validate(anchors)
    return geometry


def hierarchical_proper_score_from_logits(
    logits: torch.Tensor,
    target_anchor: torch.Tensor,
    anchor_coords: torch.Tensor,
    geometry: HierarchicalScoreGeometry,
) -> dict[str, torch.Tensor]:
    """Chain a stratum log score with a normalized conditional energy score."""

    if logits.ndim != 2 or logits.shape[1] != len(anchor_coords):
        raise ValueError("logits and anchor dictionary disagree")
    if target_anchor.shape != (len(logits),):
        raise ValueError("target anchors have the wrong shape")
    log_probability = F.log_softmax(logits, dim=1)
    target_stratum = geometry.anchor_stratum[target_anchor]
    stratum_score = logits.new_empty(len(logits))
    conditional_energy_m = logits.new_empty(len(logits))
    for stratum_index, indices in enumerate(geometry.anchor_indices):
        rows = torch.nonzero(
            target_stratum == int(stratum_index), as_tuple=False
        ).flatten()
        if len(rows) == 0:
            continue
        stratum_score[rows] = -torch.logsumexp(
            log_probability[rows][:, indices], dim=1
        )
        conditional_probability = torch.softmax(
            logits[rows][:, indices], dim=1
        )
        target_xy = anchor_coords[target_anchor[rows]]
        support_xy = anchor_coords[indices]
        first = torch.sum(
            conditional_probability * torch.cdist(target_xy, support_xy, p=2),
            dim=1,
        )
        second = 0.5 * torch.sum(
            (conditional_probability @ geometry.pair_distance_m[stratum_index])
            * conditional_probability,
            dim=1,
        )
        conditional_energy_m[rows] = (first - second).clamp_min(0.0)
    conditional_normalized = conditional_energy_m / geometry.distance_scale_m[
        target_stratum
    ]
    total = stratum_score + conditional_normalized
    return {
        "loss": total.mean(),
        "stratum_log_score_nats": stratum_score.mean(),
        "conditional_energy_score_m": conditional_energy_m.mean(),
        "conditional_energy_score_normalized": conditional_normalized.mean(),
    }


def _train_hierarchical_one_epoch(
    *, model, loader, optimizer: torch.optim.Optimizer, geometry, device
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "stratum_log_score_nats": 0.0,
        "conditional_energy_score_m": 0.0,
        "conditional_energy_score_normalized": 0.0,
    }
    rows = 0
    for batch_rss, batch_anchor, _batch_device in loader:
        batch_rss = batch_rss.to(device, non_blocking=True)
        batch_anchor = batch_anchor.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model.retrieval_logits(batch_rss)[0]
        values = hierarchical_proper_score_from_logits(
            logits, batch_anchor, model.anchor_coords, geometry
        )
        values["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), FROZEN_MEMBER_V4["gradient_clip"]
        )
        optimizer.step()
        count = len(batch_rss)
        rows += count
        for name in totals:
            totals[name] += float(values[name].detach().cpu()) * count
    if rows == 0:
        raise RuntimeError("hierarchical training loader produced no rows")
    return {name: value / rows for name, value in totals.items()}


def _fit_hierarchical_schedule(
    *,
    rss,
    raw_xy,
    building,
    floor,
    phone_ids,
    coordinate_keys,
    source_stats,
    fit_idx,
    stop_idx,
    seed,
    device,
    log_every,
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
        mechanism="M0",
        device=device,
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
    geometry = build_hierarchical_score_geometry(
        model.anchor_coords, model.anchor_buildings, model.anchor_floors
    )
    best_energy = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, FROZEN_MEMBER_V4["maximum_epochs"] + 1):
        fit_metrics = _train_hierarchical_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            geometry=geometry,
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
        score = stop_metrics["coordinate_group_macro_energy_score_3d_m"]
        improved = score < (
            best_energy - FROZEN_MEMBER_V4["minimum_energy_improvement_m"]
        )
        if improved:
            best_energy = float(score)
            best_epoch = int(epoch)
            stale = 0
        elif epoch >= FROZEN_MEMBER_V4["minimum_epochs"]:
            stale += 1
        record = {
            "epoch": int(epoch),
            "fit": fit_metrics,
            "early_stop": stop_metrics,
            "best_epoch": int(best_epoch),
            "best_energy_score_3d_m": float(best_energy),
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
        raise RuntimeError("hierarchical epoch selection found no finite score")
    return best_epoch, history


def _fit_hierarchical_fixed(
    *,
    rss,
    raw_xy,
    building,
    floor,
    phone_ids,
    coordinate_keys,
    source_stats,
    train_idx,
    seed,
    epochs,
    device,
    log_every,
):
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
        mechanism="M0",
        device=device,
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
    geometry = build_hierarchical_score_geometry(
        model.anchor_coords, model.anchor_buildings, model.anchor_floors
    )
    history = []
    for epoch in range(1, int(epochs) + 1):
        metrics = _train_hierarchical_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            geometry=geometry,
            device=device,
        )
        history.append({"epoch": int(epoch), **metrics})
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


def _anchor_radio_fields(
    *, rss, phone_ids, coordinate_keys, training_indices, radio_memory
) -> tuple[dict[str, torch.Tensor], dict]:
    raw_probability = radio_memory["detection_prob"].to(torch.float64)
    device_count = radio_memory["device_counts"].to(torch.float64)
    background = _global_detection_background(
        rss, phone_ids, coordinate_keys, training_indices
    ).to(torch.float64)
    probability = (
        raw_probability * device_count[:, None] + background[None, :]
    ) / (device_count[:, None] + 1.0)
    probability_np = probability.clamp(1e-5, 1.0 - 1e-5).numpy()
    center = radio_memory["rss_center"].to(torch.float64).numpy()
    scale = radio_memory["rss_scale"].to(torch.float64).numpy()
    available = raw_probability.numpy() > 0.0
    fallback_center, fallback_scale = _ap_fallbacks(center, scale, available)
    completed_center = center.copy()
    missing = ~available
    completed_center[missing] = np.broadcast_to(
        fallback_center[None, :], completed_center.shape
    )[missing]
    return {
        "detection_probability": torch.from_numpy(probability_np.astype(np.float32)),
        "rss_center": torch.from_numpy(completed_center.astype(np.float32)),
        "rss_scale": torch.from_numpy(fallback_scale.astype(np.float32)),
    }, {
        "global_fallback_node_ap_fraction": float(missing.mean()),
        "detection_estimator": "equal-phone node rate with one background pseudo-count",
        "conditional_rss": "node center with AP-wise global fallback",
    }


@torch.no_grad()
def evaluate_anchor_panel(
    *,
    label,
    model,
    radio_memory,
    coord_origin,
    training_indices,
    evaluation_rows,
    rss,
    raw_xy,
    building,
    floor,
    phone_ids,
    coordinate_keys,
    lambda_values,
    degrees_freedom,
    batch_size,
    candidate_chunk_size,
    credible_levels,
) -> dict:
    device = next(model.parameters()).device
    fields_cpu, field_diagnostics = _anchor_radio_fields(
        rss=rss,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        training_indices=training_indices,
        radio_memory=radio_memory,
    )
    fields = {name: value.to(device) for name, value in fields_cpu.items()}
    anchor_xy = model.anchor_coords.to(device)
    anchor_building = model.anchor_buildings.to(device)
    anchor_floor = model.anchor_floors.to(device)
    height = float(FROZEN_MEMBER_V4["floor_height_m"])
    anchor_support_3d = torch.cat(
        (
            anchor_xy,
            height * anchor_floor.to(anchor_xy.dtype).unsqueeze(1),
        ),
        dim=1,
    )
    anchor_pair = torch.cdist(anchor_support_3d, anchor_support_3d, p=2)
    origin = torch.as_tensor(coord_origin, device=device, dtype=torch.float32)
    configurations = {"front_anchor": {"variant": "front_anchor"}}
    for value in lambda_values:
        name = f"anchor_full__lambda_{_parameter_label(value)}"
        configurations[name] = {"variant": "anchor_full", "lambda": float(value)}
    output_parts = {name: {} for name in configurations}
    rss_device = rss.to(device)
    for start in range(0, len(evaluation_rows), int(batch_size)):
        rows = evaluation_rows[start : start + int(batch_size)]
        query = rss_device[rows]
        logits = model.retrieval_logits(query)[0]
        log_prior = F.log_softmax(logits, dim=1)
        target_xy = (
            torch.as_tensor(raw_xy[rows], device=device, dtype=torch.float32)
            - origin.unsqueeze(0)
        )
        target_building = torch.as_tensor(
            building[rows], device=device, dtype=torch.long
        )
        target_floor = torch.as_tensor(floor[rows], device=device, dtype=torch.long)
        _append_rows(
            output_parts["front_anchor"],
            _distribution_rows(
                torch.exp(log_prior),
                anchor_xy,
                anchor_building,
                anchor_floor,
                target_xy,
                target_building,
                target_floor,
                pair_distance_3d=anchor_pair,
                floor_height_m=height,
                true_candidate=None,
                credible_levels=credible_levels,
            ),
        )
        loss = marked_hurdle_loss(
            query,
            fields["detection_probability"],
            fields["rss_center"],
            fields["rss_scale"],
            degrees_freedom=degrees_freedom,
            scan_effect_tau=0.0,
            quadrature_order=9,
            candidate_chunk_size=candidate_chunk_size,
        )
        for name, specification in configurations.items():
            if specification["variant"] == "front_anchor":
                continue
            probability = torch.softmax(
                log_prior - loss / float(specification["lambda"]), dim=1
            )
            _append_rows(
                output_parts[name],
                _distribution_rows(
                    probability,
                    anchor_xy,
                    anchor_building,
                    anchor_floor,
                    target_xy,
                    target_building,
                    target_floor,
                    pair_distance_3d=anchor_pair,
                    floor_height_m=height,
                    true_candidate=None,
                    credible_levels=credible_levels,
                ),
            )
        processed = min(start + int(batch_size), len(evaluation_rows))
        if start == 0 or processed == len(evaluation_rows) or (
            (start // int(batch_size)) + 1
        ) % 20 == 0:
            print(
                json.dumps(
                    {
                        "objective": label,
                        "anchor_evaluation_rows": int(processed),
                        "total_rows": int(len(evaluation_rows)),
                        "anchors": int(len(anchor_xy)),
                    }
                ),
                flush=True,
            )
    row_outputs = {name: _finish_rows(parts) for name, parts in output_parts.items()}
    summaries = {
        name: summarize_rows(values, evaluation_rows, coordinate_keys, phone_ids)
        for name, values in row_outputs.items()
    }
    return {
        "evaluation_rows": evaluation_rows,
        "row_outputs": row_outputs,
        "summaries": summaries,
        "configurations": configurations,
        "field_diagnostics": field_diagnostics,
    }


def select_anchor_lambda(panel: dict, lambda_grid: tuple[float, ...]) -> dict:
    scores = {}
    for value in lambda_grid:
        name = f"anchor_full__lambda_{_parameter_label(value)}"
        scores[name] = panel["summaries"][name]["energy_score_3d_m"][
            "coordinate_macro"
        ]["mean"]
    chosen = min(
        scores,
        key=lambda name: (
            scores[name],
            -float(panel["configurations"][name]["lambda"]),
        ),
    )
    return {
        "configuration": chosen,
        "lambda": float(panel["configurations"][chosen]["lambda"]),
        "calibration_energy_score_3d_m": float(scores[chosen]),
        "all_scores": {name: float(score) for name, score in scores.items()},
    }


def _schedule_for_objective(objective: str, **kwargs):
    if objective == "joint_3d_energy":
        return _fit_schedule(mechanism="M0", **kwargs)
    if objective == "hierarchical_topology":
        return _fit_hierarchical_schedule(**kwargs)
    raise ValueError(f"unknown front objective: {objective}")


def _fixed_for_objective(objective: str, **kwargs):
    if objective == "joint_3d_energy":
        return _fit_fixed(mechanism="M0", **kwargs)
    if objective == "hierarchical_topology":
        return _fit_hierarchical_fixed(**kwargs)
    raise ValueError(f"unknown front objective: {objective}")


def _score_geometry_diagnostics(model) -> dict:
    geometry = build_hierarchical_score_geometry(
        model.anchor_coords, model.anchor_buildings, model.anchor_floors
    )
    return {
        f"building_{building}_floor_{floor}": {
            "anchors": int(len(geometry.anchor_indices[index])),
            "mean_nonzero_pair_distance_m": float(
                geometry.distance_scale_m[index].detach().cpu()
            ),
        }
        for index, (building, floor) in enumerate(geometry.strata)
    }


def run_hierarchical_front_audit(
    *,
    train_data: str | Path,
    registry_path: str | Path,
    geometry_seed: int,
    front_seed: int,
    retained_fraction: float,
    early_stop_fraction: float,
    calibration_fraction: float,
    lambda_grid: tuple[float, ...],
    degrees_freedom: float,
    batch_size: int,
    candidate_chunk_size: int,
    credible_levels: tuple[float, ...],
    device_name: str,
    threads: int,
    log_every: int,
    checkpoint_output: str | Path | None,
) -> dict:
    torch.set_num_threads(max(1, min(int(threads), torch.get_num_threads())))
    if torch.cuda.is_available():
        _configure_cuda_reproducibility()
        torch.set_float32_matmul_precision("highest")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    data_path = resolve_path(str(train_data)).resolve()
    registry_file = resolve_path(str(registry_path)).resolve()
    registry = validate_registry(registry_file, train_data=data_path)
    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        _timestamp_t,
        source_stats,
    ) = load_dataset(data_path, device="cpu")
    raw_xy = raw_coords(coords_norm, building_t, source_stats)
    building = building_t.numpy()
    floor = floor_t.numpy()
    phone_ids = phone_t.numpy()
    coordinate_keys = location_keys(raw_xy, building, floor)
    development_phones = tuple(map(int, registry["device_development_phone_ids"]))
    partition = coordinate_axis_partition(
        coordinate_keys, phone_ids, development_phones
    )
    development_keys = set(partition["development_core"])
    support_keys, support_details = geometry_support_with_hull(
        development_keys, retained_fraction, seed=geometry_seed
    )
    evaluation_keys = development_keys - support_keys
    nested, nested_details = internal_support_roles(
        support_keys,
        seed=geometry_seed + 1_000_003,
        early_stop_fraction=early_stop_fraction,
        calibration_fraction=calibration_fraction,
    )
    fit_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, nested["fit"]
    )
    stop_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, nested["early_stop"]
    )
    calibration_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, nested["calibration"]
    )
    calibration_train_rows = _rows_for_keys(
        coordinate_keys,
        phone_ids,
        development_phones,
        nested["fit"] | nested["early_stop"],
    )
    support_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, support_keys
    )
    evaluation_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, evaluation_keys
    )
    row_roles = {
        "fit": fit_rows,
        "early_stop": stop_rows,
        "calibration": calibration_rows,
        "unmeasured_evaluation": evaluation_rows,
    }
    role_names = list(row_roles)
    for index, left in enumerate(role_names):
        for right in role_names[index + 1 :]:
            if len(np.intersect1d(row_roles[left], row_roles[right])):
                raise RuntimeError(f"hierarchical audit row roles overlap: {left}, {right}")

    objective_records = {}
    objective_rows = {}
    checkpoint_records = {}
    score_geometry = None
    checkpoint_stats = None
    for objective in OBJECTIVES:
        selected_epoch, schedule_history = _schedule_for_objective(
            objective,
            rss=rss,
            raw_xy=raw_xy,
            building=building,
            floor=floor,
            phone_ids=phone_ids,
            coordinate_keys=coordinate_keys,
            source_stats=source_stats,
            fit_idx=fit_rows,
            stop_idx=stop_rows,
            seed=front_seed,
            device=device,
            log_every=log_every,
        )
        (
            calibration_model,
            calibration_memory,
            _calibration_centered,
            _calibration_anchor,
            _calibration_stats,
            calibration_origin,
            _calibration_phones,
            calibration_history,
        ) = _fixed_for_objective(
            objective,
            rss=rss,
            raw_xy=raw_xy,
            building=building,
            floor=floor,
            phone_ids=phone_ids,
            coordinate_keys=coordinate_keys,
            source_stats=source_stats,
            train_idx=calibration_train_rows,
            seed=front_seed,
            epochs=selected_epoch,
            device=device,
            log_every=log_every,
        )
        calibration_panel = evaluate_anchor_panel(
            label=f"{objective}_calibration",
            model=calibration_model,
            radio_memory=calibration_memory,
            coord_origin=calibration_origin,
            training_indices=calibration_train_rows,
            evaluation_rows=calibration_rows,
            rss=rss,
            raw_xy=raw_xy,
            building=building,
            floor=floor,
            phone_ids=phone_ids,
            coordinate_keys=coordinate_keys,
            lambda_values=lambda_grid,
            degrees_freedom=degrees_freedom,
            batch_size=batch_size,
            candidate_chunk_size=candidate_chunk_size,
            credible_levels=credible_levels,
        )
        selected_lambda = select_anchor_lambda(calibration_panel, lambda_grid)
        calibration_selected_summary = calibration_panel["summaries"][
            selected_lambda["configuration"]
        ]
        del calibration_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        (
            final_model,
            final_memory,
            _final_centered,
            _final_anchor,
            final_stats,
            final_origin,
            final_phones,
            final_history,
        ) = _fixed_for_objective(
            objective,
            rss=rss,
            raw_xy=raw_xy,
            building=building,
            floor=floor,
            phone_ids=phone_ids,
            coordinate_keys=coordinate_keys,
            source_stats=source_stats,
            train_idx=support_rows,
            seed=front_seed,
            epochs=selected_epoch,
            device=device,
            log_every=log_every,
        )
        final_panel = evaluate_anchor_panel(
            label=f"{objective}_unmeasured",
            model=final_model,
            radio_memory=final_memory,
            coord_origin=final_origin,
            training_indices=support_rows,
            evaluation_rows=evaluation_rows,
            rss=rss,
            raw_xy=raw_xy,
            building=building,
            floor=floor,
            phone_ids=phone_ids,
            coordinate_keys=coordinate_keys,
            lambda_values=(selected_lambda["lambda"],),
            degrees_freedom=degrees_freedom,
            batch_size=batch_size,
            candidate_chunk_size=candidate_chunk_size,
            credible_levels=credible_levels,
        )
        full_name = f"anchor_full__lambda_{_parameter_label(selected_lambda['lambda'])}"
        objective_records[objective] = {
            "selected_epoch": int(selected_epoch),
            "selected_lambda": selected_lambda,
            "schedule": {
                "epochs_run": int(len(schedule_history)),
                "best_early_stop_energy_score_3d_m": float(
                    min(
                        record["early_stop"][
                            "coordinate_group_macro_energy_score_3d_m"
                        ]
                        for record in schedule_history
                    )
                ),
                "history": schedule_history,
            },
            "calibration": {
                "rows": int(len(calibration_rows)),
                "front_summary": calibration_panel["summaries"]["front_anchor"],
                "selected_full_summary": calibration_selected_summary,
                "refit_history": calibration_history,
            },
            "unmeasured_evaluation": {
                "front_summary": final_panel["summaries"]["front_anchor"],
                "full_summary": final_panel["summaries"][full_name],
                "field_diagnostics": final_panel["field_diagnostics"],
            },
            "final_refit_history": final_history,
        }
        objective_rows[objective] = {
            "front": final_panel["row_outputs"]["front_anchor"],
            "full": final_panel["row_outputs"][full_name],
        }
        checkpoint_records[objective] = {
            "model_state": {
                name: value.detach().cpu()
                for name, value in final_model.state_dict().items()
            },
            "radio_memory": {
                name: value.detach().cpu()
                for name, value in final_memory.items()
                if isinstance(value, torch.Tensor)
            },
            "coordinate_keys": [list(key) for key in final_memory["coordinate_keys"]],
            "coord_origin": torch.as_tensor(final_origin),
            "training_phone_ids": list(final_phones),
            "selected_epoch": int(selected_epoch),
            "selected_lambda": selected_lambda,
        }
        if score_geometry is None:
            score_geometry = _score_geometry_diagnostics(final_model)
            checkpoint_stats = final_stats
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparisons = {}
    for label, left_objective, left_stage, right_objective, right_stage in (
        (
            "hierarchical_front_vs_joint_front",
            "hierarchical_topology",
            "front",
            "joint_3d_energy",
            "front",
        ),
        (
            "hierarchical_full_vs_joint_full",
            "hierarchical_topology",
            "full",
            "joint_3d_energy",
            "full",
        ),
        (
            "joint_full_vs_joint_front",
            "joint_3d_energy",
            "full",
            "joint_3d_energy",
            "front",
        ),
        (
            "hierarchical_full_vs_hierarchical_front",
            "hierarchical_topology",
            "full",
            "hierarchical_topology",
            "front",
        ),
    ):
        comparisons[label] = {
            metric: _paired_coordinate_delta(
                objective_rows[left_objective][left_stage][metric],
                objective_rows[right_objective][right_stage][metric],
                evaluation_rows,
                coordinate_keys,
            )
            for metric in ("error_2d_m", "energy_score_3d_m")
        }

    checkpoint_path = None
    if checkpoint_output is not None:
        checkpoint_path = resolve_path(str(checkpoint_output)).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            raise FileExistsError(
                f"refusing to overwrite hierarchical checkpoint: {checkpoint_path}"
            )
        torch.save(
            {
                "artifact_status": "v6_hierarchical_front_development_comparison",
                "selection_authority": False,
                "objectives": checkpoint_records,
                "stats": checkpoint_stats,
                "training_indices": torch.as_tensor(support_rows),
                "geometry_seed": int(geometry_seed),
                "front_seed": int(front_seed),
                "registry_sha256": registry["registry_sha256"],
            },
            checkpoint_path,
        )

    return {
        "artifact_status": "v6_hierarchical_front_development_audit",
        "selection_authority": False,
        "protected_coordinate_confirmation_evaluated": False,
        "protected_confirmation_phones_evaluated": False,
        "registered_outer_test_outcomes_evaluated": False,
        "train_data": str(data_path),
        "registry": str(registry_file),
        "registry_sha256": registry["registry_sha256"],
        "device": str(device),
        "geometry_seed": int(geometry_seed),
        "front_seed": int(front_seed),
        "geometry": {
            "development_coordinates": int(len(development_keys)),
            "retained_coordinates": int(len(support_keys)),
            "unmeasured_coordinates": int(len(evaluation_keys)),
            "realized_retained_fraction": float(len(support_keys) / len(development_keys)),
            "support_by_stratum": support_details,
            "nested_support_roles": {
                name: int(len(values)) for name, values in nested.items()
            },
            "nested_by_stratum": nested_details,
            "row_roles": {name: int(len(rows)) for name, rows in row_roles.items()},
            "hierarchical_distance_scales": score_geometry,
        },
        "design": {
            "member_count_E": 1,
            "front_architecture_shared": True,
            "joint_baseline_score": "exact full-dictionary 3-D energy; floor height 4m",
            "hierarchical_score": (
                "negative log building-floor marginal plus within-stratum energy "
                "divided by the mean nonzero within-stratum anchor distance"
            ),
            "hierarchical_score_is_strictly_proper": True,
            "hierarchical_tunable_weight": None,
            "backend": "identical marked-AP generalized-Bayes update on anchors",
            "student_degrees_freedom": float(degrees_freedom),
            "lambda_grid": list(map(float, lambda_grid)),
            "lambda_selection": "nested calibration coordinate-macro 3-D energy score",
            "no_continuous_field": True,
            "no_top_k": True,
            "no_ensemble": True,
            "no_domain_adversarial_component": True,
        },
        "objectives": objective_records,
        "unmeasured_evaluation": {
            "rows": int(len(evaluation_rows)),
            "coordinates": int(len(evaluation_keys)),
            "paired_coordinate_comparisons": comparisons,
        },
        "training": {"checkpoint": str(checkpoint_path) if checkpoint_path else None},
        "interpretation_contract": {
            "hierarchical_admission": (
                "The hierarchical full method must improve floor accuracy without "
                "increasing either coordinate-macro MAE or coordinate-macro 3-D energy."
            ),
            "backend_admission": (
                "Within each front objective, the calibrated marked-AP update must not "
                "increase coordinate-macro MAE or 3-D energy."
            ),
            "scope": (
                "One development geometry and one member can reject a structural "
                "hypothesis, but cannot confirm the final model."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a topology-aware hierarchical front under the sparse50 protocol"
    )
    parser.add_argument("--train-data", default="data/UJIndoorLoc/trainingData.csv")
    parser.add_argument("--registry", default="evidence/v44/registry_v4_4.json")
    parser.add_argument("--geometry-seed", type=int, default=5201)
    parser.add_argument("--front-seed", type=int, default=4101)
    parser.add_argument("--retained-fraction", type=float, default=0.50)
    parser.add_argument("--early-stop-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument(
        "--lambda-grid", default="0.015625,0.03125,0.0625,0.125,0.25,0.5"
    )
    parser.add_argument("--student-df", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-chunk-size", type=int, default=64)
    parser.add_argument("--credible-levels", default="0.5,0.8,0.9,0.95")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-output", default=None)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lambda_grid = _csv_tuple(args.lambda_grid, float)
    credible_levels = _csv_tuple(args.credible_levels, float)
    if any(not math.isfinite(value) or value <= 0.0 for value in lambda_grid):
        raise ValueError("all lambda values must be finite and positive")
    if any(not 0.0 < value < 1.0 for value in credible_levels):
        raise ValueError("credible levels must lie inside (0,1)")
    output = resolve_path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite hierarchical artifact: {output}")
    if args.checkpoint_output is not None:
        checkpoint = resolve_path(args.checkpoint_output).resolve()
        if checkpoint.exists():
            raise FileExistsError(
                f"refusing to overwrite hierarchical checkpoint: {checkpoint}"
            )
    payload = run_hierarchical_front_audit(
        train_data=args.train_data,
        registry_path=args.registry,
        geometry_seed=args.geometry_seed,
        front_seed=args.front_seed,
        retained_fraction=args.retained_fraction,
        early_stop_fraction=args.early_stop_fraction,
        calibration_fraction=args.calibration_fraction,
        lambda_grid=lambda_grid,
        degrees_freedom=args.student_df,
        batch_size=args.batch_size,
        candidate_chunk_size=args.candidate_chunk_size,
        credible_levels=credible_levels,
        device_name=args.device,
        threads=args.threads,
        log_every=args.log_every,
        checkpoint_output=args.checkpoint_output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact_status": payload["artifact_status"],
                "output": str(output),
                "geometry": payload["geometry"],
                "objectives": {
                    name: {
                        "selected_epoch": record["selected_epoch"],
                        "selected_lambda": record["selected_lambda"],
                        "front": record["unmeasured_evaluation"]["front_summary"],
                        "full": record["unmeasured_evaluation"]["full_summary"],
                    }
                    for name, record in payload["objectives"].items()
                },
                "paired": payload["unmeasured_evaluation"][
                    "paired_coordinate_comparisons"
                ],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
