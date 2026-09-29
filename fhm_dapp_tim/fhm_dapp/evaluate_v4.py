"""Evaluate registered DR-JAM-v4.4 development members.

Only roles explicitly present in a v4 training checkpoint can be evaluated.
This development executable has no route to coordinate confirmation or the
historical official validation file; those require a later frozen-decision
evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords, resolve_path
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.device_robust_v4 import (
    anchor_support_3d,
    device_marginal_radio_loss,
    load_device_robust_checkpoint,
)
from fhm_dapp_tim.fhm_dapp.minimal_v3 import (
    anchor_log_marginal,
    joint_member_anchor_update,
    topk_joint_reverse_kl_projection,
)
from fhm_dapp_tim.fhm_dapp.posterior_diagnostics_v44 import (
    joint_information_diagnostics,
    radial_credible_balls,
)
from fhm_dapp_tim.fhm_dapp.protocol_v4 import (
    MECHANISM_MEMBER_SEEDS,
    MECHANISM_DIAGNOSTIC,
    PROTOCOL_VERSION,
    sha256_location_keys,
    validate_registry,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values, probability).cpu())


def _cvar(values: torch.Tensor, tail_fraction: float = 0.10) -> float:
    count = max(1, int(math.ceil(len(values) * float(tail_fraction))))
    return float(torch.topk(values, k=count, largest=True).values.mean().cpu())


def _summarize_vector(values: torch.Tensor, prefix: str) -> dict[str, float]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("metric vector must be non-empty")
    return {
        f"{prefix}_mean": float(values.mean().cpu()),
        f"{prefix}_median": _quantile(values, 0.50),
        f"{prefix}_p90": _quantile(values, 0.90),
        f"{prefix}_p95": _quantile(values, 0.95),
        f"{prefix}_cvar90": _cvar(values, 0.10),
    }


def _checkpoint_role(checkpoint: dict, requested: str) -> tuple[str, torch.Tensor]:
    role_axis = checkpoint["config"]["role_axis"]
    mapping = {
        ("coordinate", "selection"): "structure_development",
        ("coordinate", "early-stop"): "early_stop",
        ("device-development", "selection"): "matched_test",
        ("device-development", "early-stop"): "early_stop",
        ("device-development", "combined-shift"): "combined_shift_test",
    }
    role = mapping.get((role_axis, requested))
    if role is None:
        raise ValueError(
            f"evaluation role {requested!r} is not permitted for {role_axis}"
        )
    indices = checkpoint["role_indices"][role]
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
        raise ValueError("checkpoint role indices are malformed")
    if len(indices) == 0:
        raise ValueError(f"registered evaluation role is empty: {role}")
    return role, indices.to(dtype=torch.long)


def _validate_member_set(
    paths: list[Path],
    checkpoints: list[dict],
    mechanism: str,
    *,
    allow_engineering_pilot: bool,
) -> None:
    if len(paths) == 0:
        raise ValueError("at least one checkpoint is required")
    if len(paths) != int(MECHANISM_DIAGNOSTIC["member_count_E"]):
        raise ValueError("mechanism evaluator requires independent E=1 runs")
    expected_training_mechanism = "M2" if mechanism == "M3" else mechanism
    seeds = []
    first = checkpoints[0]
    for path, checkpoint in zip(paths, checkpoints):
        config = checkpoint.get("config", {})
        if config.get("training_protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"checkpoint protocol mismatch: {path}")
        if config.get("radio_memory_estimator") != "equal_phone_within_coordinate_v4":
            raise ValueError(f"checkpoint uses obsolete row-weighted radio memory: {path}")
        if config.get("radio_score") != "joint_hurdle_log_score_fixed_J_nats_per_ap":
            raise ValueError(f"checkpoint uses an improper mixed-denominator radio score: {path}")
        if config.get("cuda_attention_kernel") != "math_only":
            raise ValueError(f"checkpoint predates deterministic v4 attention: {path}")
        if config.get("tf32") is not False:
            raise ValueError(f"checkpoint does not freeze TF32 off: {path}")
        if config.get("mechanism") != expected_training_mechanism:
            raise ValueError(
                f"{mechanism} requires {expected_training_mechanism} members: {path}"
            )
        if checkpoint.get("selection_authority") is not True:
            if not allow_engineering_pilot:
                raise ValueError(f"engineering pilot has no selection authority: {path}")
        if checkpoint.get("source_sha256") != first.get("source_sha256"):
            raise ValueError("member source hashes differ")
        if checkpoint.get("registry_sha256") != first.get("registry_sha256"):
            raise ValueError("member registry hashes differ")
        if config.get("role_axis") != first["config"].get("role_axis"):
            raise ValueError("member role axes differ")
        if config.get("outer_fold") != first["config"].get("outer_fold"):
            raise ValueError("member coordinate folds differ")
        if config.get("held_out_phone") != first["config"].get("held_out_phone"):
            raise ValueError("member held-out phones differ")
        if checkpoint.get("training_indices_sha256") != first.get(
            "training_indices_sha256"
        ):
            raise ValueError("member training roles differ")
        seeds.append(int(checkpoint["seed"]))
    if len(set(seeds)) != len(seeds):
        raise ValueError("member seeds must be distinct")
    if any(seed not in MECHANISM_MEMBER_SEEDS for seed in seeds):
        raise ValueError("mechanism evaluation requires a registered diagnostic seed")


def _validate_shared_dictionary(models: list, checkpoints: list[dict]) -> None:
    first_model = models[0]
    first_memory = checkpoints[0]["radio_memory"]
    for model, checkpoint in zip(models[1:], checkpoints[1:]):
        for name in ("anchor_coords", "anchor_buildings", "anchor_floors"):
            if not torch.equal(getattr(model, name).cpu(), getattr(first_model, name).cpu()):
                raise ValueError(f"member anchor dictionaries differ: {name}")
        for name in (
            "detection_prob",
            "rss_center",
            "rss_scale",
            "counts",
            "device_counts",
        ):
            if not torch.equal(checkpoint["radio_memory"][name], first_memory[name]):
                raise ValueError(f"member radio memories differ: {name}")
        if not torch.equal(
            checkpoint["coord_origin"], checkpoints[0]["coord_origin"]
        ):
            raise ValueError("member coordinate origins differ")
        if checkpoint.get("anchor_coordinate_keys_sha256") != checkpoints[0].get(
            "anchor_coordinate_keys_sha256"
        ):
            raise ValueError("member protocol anchor identities differ")


def _device_covariance(checkpoints: list[dict], mechanism: str) -> torch.Tensor | None:
    if mechanism != "M3":
        return None
    first_effect = checkpoints[0].get("device_effect")
    if not first_effect or "covariance" not in first_effect:
        raise ValueError("M3 requires a training-role device covariance")
    covariance = first_effect["covariance"]
    for checkpoint in checkpoints[1:]:
        effect = checkpoint.get("device_effect")
        if not effect or not torch.equal(effect["covariance"], covariance):
            raise ValueError("M3 member device covariance estimates differ")
    return covariance


def _selected_joint(
    log_joint: torch.Tensor,
    anchor_support: torch.Tensor,
    top_k: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    anchors = log_joint.shape[2]
    if int(top_k) == anchors:
        log_marginal = anchor_log_marginal(log_joint)
        probability = torch.exp(log_marginal)
        indices = torch.arange(anchors, device=log_joint.device).unsqueeze(0)
        indices = indices.expand(log_joint.shape[1], -1)
        support = anchor_support.unsqueeze(0).expand(log_joint.shape[1], -1, -1)
        joint_weights = torch.exp(log_joint)
        retained_mass = torch.ones(
            log_joint.shape[1], device=log_joint.device, dtype=log_joint.dtype
        )
        return probability, indices, support, joint_weights, retained_mass
    indices, joint_weights = topk_joint_reverse_kl_projection(log_joint, int(top_k))
    probability = joint_weights.sum(dim=0)
    support = anchor_support[indices]
    selected_log_marginal = torch.gather(anchor_log_marginal(log_joint), 1, indices)
    retained_mass = torch.exp(selected_log_marginal).sum(dim=1)
    return probability, indices, support, joint_weights, retained_mass


def _energy_score_3d(
    probability: torch.Tensor,
    support: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    target_distance = torch.linalg.vector_norm(
        support - target.unsqueeze(1), dim=2
    )
    pair_distance = torch.linalg.vector_norm(
        support.unsqueeze(2) - support.unsqueeze(1), dim=3
    )
    first = torch.sum(probability * target_distance, dim=1)
    second = 0.5 * torch.einsum(
        "bi,bij,bj->b", probability, pair_distance, probability
    )
    return first - second


@torch.no_grad()
def evaluate_posterior(
    *,
    models: list,
    checkpoints: list[dict],
    rss: torch.Tensor,
    raw_xy: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    phone_ids: np.ndarray,
    coordinate_keys: list[tuple],
    indices: torch.Tensor,
    mechanism: str,
    top_k: int,
    loss_temperature: float,
    batch_size: int,
) -> dict:
    if len(indices) == 0:
        raise ValueError("evaluation role is empty")
    if not math.isfinite(float(loss_temperature)) or float(loss_temperature) <= 0:
        raise ValueError("loss_temperature must be finite and positive")
    device = next(models[0].parameters()).device
    anchor_count = models[0].n_anchors
    if not 1 <= int(top_k) <= anchor_count:
        raise ValueError("top_k must lie within the anchor dictionary")
    memory = checkpoints[0]["radio_memory"]
    covariance = _device_covariance(checkpoints, mechanism)
    anchor_support = anchor_support_3d(
        models[0], floor_height_m=float(checkpoints[0]["config"]["floor_height_m"])
    ).to(device)
    origin = checkpoints[0]["coord_origin"].numpy()
    centered_xy = torch.from_numpy((raw_xy - origin).astype(np.float32))
    height = float(checkpoints[0]["config"]["floor_height_m"])
    all_errors = []
    all_energy = []
    all_trace = []
    all_candidate_trace = []
    all_member_trace = []
    all_member_trace_fraction = []
    all_effective_members = []
    all_effective_anchors = []
    all_member_eigenvalues = []
    all_candidate_eigenvalues = []
    all_retained_mass = []
    all_credible: dict[str, list[torch.Tensor]] = {}
    all_member_mass = []
    all_predicted = []
    all_true_anchor_radio = []
    all_true_anchor_rows = []
    correct_floor = 0
    correct_building = 0
    selected_indices = indices.cpu().numpy().astype(np.int64, copy=False)
    encoded_anchor_keys = checkpoints[0].get("anchor_coordinate_keys")
    if not isinstance(encoded_anchor_keys, list):
        raise ValueError("checkpoint lacks protocol anchor identities")
    anchor_keys = [
        (int(key[0]), int(key[1]), float(key[2]), float(key[3]))
        for key in encoded_anchor_keys
    ]
    if len(anchor_keys) != models[0].n_anchors:
        raise ValueError("protocol anchor identities have the wrong length")
    if checkpoints[0].get("anchor_coordinate_keys_sha256") != sha256_location_keys(
        anchor_keys
    ):
        raise ValueError("protocol anchor identity hash mismatch")
    anchor_by_key = {key: index for index, key in enumerate(anchor_keys)}
    if len(anchor_by_key) != len(anchor_keys):
        raise ValueError("anchor dictionary contains duplicate coordinate keys")
    for start in range(0, len(selected_indices), int(batch_size)):
        rows = selected_indices[start : start + int(batch_size)]
        query = rss[rows].to(device)
        logits = torch.stack(
            [model.retrieval_logits(query)[0] for model in models], dim=0
        )
        radio_loss = device_marginal_radio_loss(
            query,
            memory["detection_prob"],
            memory["rss_center"],
            memory["rss_scale"],
            device_covariance=covariance,
        )
        true_anchor = [anchor_by_key.get(coordinate_keys[int(row)], -1) for row in rows]
        valid_true = [local for local, anchor in enumerate(true_anchor) if anchor >= 0]
        if valid_true:
            local_tensor = torch.as_tensor(valid_true, device=device, dtype=torch.long)
            anchor_tensor = torch.as_tensor(
                [true_anchor[local] for local in valid_true],
                device=device,
                dtype=torch.long,
            )
            all_true_anchor_radio.append(
                radio_loss[local_tensor, anchor_tensor].cpu()
            )
            all_true_anchor_rows.append(
                torch.as_tensor(rows[valid_true], dtype=torch.long)
            )
        log_joint = joint_member_anchor_update(
            logits, radio_loss, float(loss_temperature)
        )
        (
            probability,
            anchor_indices,
            support,
            joint_weights,
            retained_mass,
        ) = _selected_joint(
            log_joint, anchor_support, int(top_k)
        )
        prediction_xy = torch.sum(
            probability.unsqueeze(2) * support[:, :, :2], dim=1
        )
        target_xy = centered_xy[rows].to(device)
        target_floor = torch.as_tensor(floor[rows], device=device)
        target_building = torch.as_tensor(building[rows], device=device)
        target_support = torch.cat(
            (target_xy, height * target_floor.to(target_xy.dtype).unsqueeze(1)),
            dim=1,
        )
        errors = torch.linalg.vector_norm(prediction_xy - target_xy, dim=1)
        energy = _energy_score_3d(probability, support, target_support)
        information = joint_information_diagnostics(
            joint_weights, support[:, :, :2]
        )
        trace = information["total_covariance_trace_m2"]
        credible = radial_credible_balls(
            probability, support, target_support
        )
        chosen_floors = models[0].anchor_floors[anchor_indices]
        chosen_buildings = models[0].anchor_buildings[anchor_indices]
        floor_probability = torch.zeros(
            (len(rows), int(models[0].anchor_floors.max()) + 1), device=device
        )
        floor_probability.scatter_add_(1, chosen_floors, probability)
        building_probability = torch.zeros(
            (len(rows), int(models[0].anchor_buildings.max()) + 1), device=device
        )
        building_probability.scatter_add_(1, chosen_buildings, probability)
        correct_floor += int((floor_probability.argmax(1) == target_floor).sum())
        correct_building += int(
            (building_probability.argmax(1) == target_building).sum()
        )
        all_errors.append(errors.cpu())
        all_energy.append(energy.cpu())
        all_trace.append(trace.cpu())
        all_candidate_trace.append(
            information["candidate_covariance_trace_m2"].cpu()
        )
        all_member_trace.append(information["member_covariance_trace_m2"].cpu())
        all_member_trace_fraction.append(information["member_trace_fraction"].cpu())
        all_effective_members.append(information["effective_member_count"].cpu())
        all_effective_anchors.append(information["effective_candidate_count"].cpu())
        all_member_eigenvalues.append(
            information["member_covariance_eigenvalues_m2"].cpu()
        )
        all_candidate_eigenvalues.append(
            information["candidate_covariance_eigenvalues_m2"].cpu()
        )
        all_retained_mass.append(retained_mass.cpu())
        for name, values in credible.items():
            if name != "posterior_mean":
                all_credible.setdefault(name, []).append(values.cpu())
        all_member_mass.append(joint_weights.sum(dim=2).t().cpu())
        all_predicted.append((prediction_xy.cpu() + torch.from_numpy(origin)))
    errors = torch.cat(all_errors)
    energy = torch.cat(all_energy)
    trace = torch.cat(all_trace)
    candidate_trace = torch.cat(all_candidate_trace)
    member_trace = torch.cat(all_member_trace)
    member_trace_fraction = torch.cat(all_member_trace_fraction)
    effective_members = torch.cat(all_effective_members)
    effective_anchors = torch.cat(all_effective_anchors)
    member_eigenvalues = torch.cat(all_member_eigenvalues)
    candidate_eigenvalues = torch.cat(all_candidate_eigenvalues)
    retained_mass = torch.cat(all_retained_mass)
    credible_vectors = {
        name: torch.cat(values) for name, values in all_credible.items()
    }
    member_mass = torch.cat(all_member_mass)
    predicted = torch.cat(all_predicted)
    true_anchor_radio = (
        torch.cat(all_true_anchor_radio)
        if all_true_anchor_radio
        else torch.empty(0, dtype=errors.dtype)
    )
    true_anchor_rows = (
        torch.cat(all_true_anchor_rows)
        if all_true_anchor_rows
        else torch.empty(0, dtype=torch.long)
    )
    row_phone = torch.as_tensor(phone_ids[selected_indices], dtype=torch.long)
    per_phone = {}
    phone_row_means = []
    phone_coordinate_means = []
    radio_groups: dict[tuple, list[int]] = {}
    for local, global_row in enumerate(true_anchor_rows.tolist()):
        radio_groups.setdefault(coordinate_keys[int(global_row)], []).append(local)
    radio_group_means = (
        torch.stack(
            [
                true_anchor_radio[torch.as_tensor(rows)].mean()
                for rows in radio_groups.values()
            ]
        )
        if radio_groups
        else torch.empty(0, dtype=errors.dtype)
    )
    for phone in sorted(map(int, torch.unique(row_phone).tolist())):
        mask = row_phone == phone
        values = errors[mask]
        local_rows = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        phone_groups: dict[tuple, list[int]] = {}
        for local in local_rows:
            phone_groups.setdefault(
                coordinate_keys[int(selected_indices[local])], []
            ).append(local)
        phone_group_errors = torch.stack(
            [
                errors[torch.as_tensor(rows)].mean()
                for rows in phone_groups.values()
            ]
        )
        phone_group_energy = torch.stack(
            [
                energy[torch.as_tensor(rows)].mean()
                for rows in phone_groups.values()
            ]
        )
        phone_radio_groups: dict[tuple, list[int]] = {}
        for local, global_row in enumerate(true_anchor_rows.tolist()):
            if int(phone_ids[int(global_row)]) == phone:
                phone_radio_groups.setdefault(
                    coordinate_keys[int(global_row)], []
                ).append(local)
        phone_radio_macro = None
        if phone_radio_groups:
            phone_radio_macro = float(
                torch.stack(
                    [
                        true_anchor_radio[torch.as_tensor(rows)].mean()
                        for rows in phone_radio_groups.values()
                    ]
                ).mean()
            )
        phone_row_means.append(values.mean())
        phone_coordinate_means.append(phone_group_errors.mean())
        per_phone[str(phone)] = {
            "rows": int(mask.sum()),
            "coordinate_groups": len(phone_groups),
            **_summarize_vector(values, "error_m"),
            "energy_score_3d_m": float(energy[mask].mean()),
            "coordinate_group_macro_mae_m": float(phone_group_errors.mean()),
            "coordinate_group_macro_energy_score_3d_m": float(
                phone_group_energy.mean()
            ),
            "coordinate_group_macro_true_anchor_radio_nll_nats_per_ap": (
                phone_radio_macro
            ),
        }
    group_rows: dict[tuple, list[int]] = {}
    for local, global_row in enumerate(selected_indices):
        group_rows.setdefault(coordinate_keys[int(global_row)], []).append(local)
    group_mean_errors = torch.stack(
        [errors[torch.as_tensor(rows)].mean() for rows in group_rows.values()]
    )
    group_mean_energy = torch.stack(
        [energy[torch.as_tensor(rows)].mean() for rows in group_rows.values()]
    )
    group_metric_vectors = {
        "retained_anchor_mass": retained_mass,
        "effective_member_count": effective_members,
        "effective_anchor_count": effective_anchors,
        "candidate_covariance_trace_m2": candidate_trace,
        "member_covariance_trace_m2": member_trace,
        "member_trace_fraction": member_trace_fraction,
    }
    group_metric_means = {
        name: torch.stack(
            [values[torch.as_tensor(rows)].mean() for rows in group_rows.values()]
        ).mean()
        for name, values in group_metric_vectors.items()
    }
    credible_summary: dict[str, float] = {}
    for name, values in credible_vectors.items():
        numeric = values.to(dtype=torch.float32)
        if name.startswith("credible_coverage_"):
            label = name.removeprefix("credible_coverage_")
            credible_summary[f"empirical_credible_coverage_{label}"] = float(
                numeric.mean()
            )
            credible_summary[
                f"coordinate_group_macro_credible_coverage_{label}"
            ] = float(
                torch.stack(
                    [
                        numeric[torch.as_tensor(rows)].mean()
                        for rows in group_rows.values()
                    ]
                ).mean()
            )
        elif name.startswith("credible_radius_"):
            credible_summary[f"mean_{name}"] = float(numeric.mean())
        elif name == "target_distance_from_mean_m":
            credible_summary["mean_target_distance_from_mean_3d_m"] = float(
                numeric.mean()
            )
    summary = {
        "rows": int(len(errors)),
        "coordinate_groups": int(len(group_rows)),
        "member_count_E": len(models),
        "top_k": int(top_k),
        "loss_temperature_lambda": float(loss_temperature),
        "mechanism": mechanism,
        **_summarize_vector(errors, "error_m"),
        "energy_score_3d_m": float(energy.mean()),
        "coordinate_group_macro_mae_m": float(group_mean_errors.mean()),
        "coordinate_group_macro_energy_score_3d_m": float(
            group_mean_energy.mean()
        ),
        "phone_macro_row_mae_m": float(torch.stack(phone_row_means).mean()),
        "phone_coordinate_group_macro_mae_m": float(
            torch.stack(phone_coordinate_means).mean()
        ),
        # Compatibility alias now has the explicitly hierarchical meaning.
        "phone_macro_mae_m": float(torch.stack(phone_coordinate_means).mean()),
        "floor_accuracy": correct_floor / len(errors),
        "building_accuracy": correct_building / len(errors),
        "mean_predictive_variance_trace_m2": float(trace.mean()),
        "mean_component_covariance_trace_m2": 0.0,
        "mean_candidate_covariance_trace_m2": float(candidate_trace.mean()),
        "mean_member_covariance_trace_m2": float(member_trace.mean()),
        "mean_member_trace_fraction": float(member_trace_fraction.mean()),
        "mean_effective_member_count": float(effective_members.mean()),
        "mean_effective_anchor_count": float(effective_anchors.mean()),
        "mean_retained_anchor_mass": float(retained_mass.mean()),
        "coordinate_group_macro_candidate_covariance_trace_m2": float(
            group_metric_means["candidate_covariance_trace_m2"]
        ),
        "coordinate_group_macro_member_covariance_trace_m2": float(
            group_metric_means["member_covariance_trace_m2"]
        ),
        "coordinate_group_macro_member_trace_fraction": float(
            group_metric_means["member_trace_fraction"]
        ),
        "coordinate_group_macro_effective_member_count": float(
            group_metric_means["effective_member_count"]
        ),
        "coordinate_group_macro_effective_anchor_count": float(
            group_metric_means["effective_anchor_count"]
        ),
        "coordinate_group_macro_retained_anchor_mass": float(
            group_metric_means["retained_anchor_mass"]
        ),
        **credible_summary,
        "matched_true_anchor_rows": int(len(true_anchor_radio)),
        "true_anchor_radio_nll_nats_per_ap": (
            float(true_anchor_radio.mean()) if len(true_anchor_radio) else None
        ),
        "coordinate_group_macro_true_anchor_radio_nll_nats_per_ap": (
            float(radio_group_means.mean()) if len(radio_group_means) else None
        ),
        "per_phone": per_phone,
    }
    return {
        "summary": summary,
        "row_indices": torch.as_tensor(selected_indices, dtype=torch.long),
        "errors_m": errors,
        "energy_score_3d_m": energy,
        "predictive_variance_trace_m2": trace,
        "component_covariance_trace_m2": torch.zeros_like(trace),
        "candidate_covariance_trace_m2": candidate_trace,
        "member_covariance_trace_m2": member_trace,
        "member_trace_fraction": member_trace_fraction,
        "effective_member_count": effective_members,
        "effective_anchor_count": effective_anchors,
        "member_covariance_eigenvalues_m2": member_eigenvalues,
        "candidate_covariance_eigenvalues_m2": candidate_eigenvalues,
        "retained_anchor_mass": retained_mass,
        **credible_vectors,
        "member_posterior_mass": member_mass,
        "predicted_xy_m": predicted,
        "true_anchor_row_indices": true_anchor_rows,
        "true_anchor_radio_nll_nats_per_ap": true_anchor_radio,
    }


def run_evaluation(args: argparse.Namespace) -> dict:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    paths = [Path(value).resolve() for value in args.checkpoints]
    loaded = [load_device_robust_checkpoint(path, device) for path in paths]
    models = [item[0] for item in loaded]
    checkpoints = [item[1] for item in loaded]
    _validate_member_set(
        paths,
        checkpoints,
        args.mechanism,
        allow_engineering_pilot=bool(args.allow_engineering_pilot),
    )
    _validate_shared_dictionary(models, checkpoints)
    role_name, indices = _checkpoint_role(checkpoints[0], args.evaluation_role)
    for checkpoint in checkpoints[1:]:
        other_name, other_indices = _checkpoint_role(
            checkpoint, args.evaluation_role
        )
        if other_name != role_name or not torch.equal(indices, other_indices):
            raise ValueError("member evaluation roles differ")

    data_path = resolve_path(args.train_data).resolve()
    registry = validate_registry(args.registry, train_data=data_path)
    if checkpoints[0]["registry_sha256"] != registry["registry_sha256"]:
        raise ValueError("checkpoint and supplied v4 registry differ")
    if checkpoints[0]["source_sha256"] != _sha256_file(data_path):
        raise ValueError("checkpoint and supplied training data differ")
    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        _timestamp_t,
        stats,
    ) = load_dataset(data_path, dataset_type="UJI")
    raw_xy = raw_coords(coords_norm, building_t, stats)
    building = building_t.numpy()
    floor = floor_t.numpy()
    phone_ids = phone_t.numpy()
    keys = location_keys(raw_xy, building, floor)
    anchor_count = models[0].n_anchors
    top_k = anchor_count if args.top_k == "full" else int(args.top_k)
    if top_k != anchor_count:
        raise ValueError("mechanism diagnostic forbids sparse Top-K projection")
    if float(args.loss_temperature) != float(
        MECHANISM_DIAGNOSTIC["loss_temperature_lambda"]
    ):
        raise ValueError("mechanism diagnostic fixes lambda=1")
    started = time.perf_counter()
    result = evaluate_posterior(
        models=models,
        checkpoints=checkpoints,
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor,
        phone_ids=phone_ids,
        coordinate_keys=keys,
        indices=indices,
        mechanism=args.mechanism,
        top_k=top_k,
        loss_temperature=float(args.loss_temperature),
        batch_size=int(args.batch_size),
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v4 evaluation cache: {output}")
    payload = {
        "artifact_status": (
            "engineering_pilot_no_selection_authority"
            if any(c.get("selection_authority") is not True for c in checkpoints)
            else "registered_development_evaluation"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_role": role_name,
        "summary": result["summary"],
        "row_indices": result["row_indices"],
        "errors_m": result["errors_m"],
        "energy_score_3d_m": result["energy_score_3d_m"],
        "predictive_variance_trace_m2": result[
            "predictive_variance_trace_m2"
        ],
        "component_covariance_trace_m2": result[
            "component_covariance_trace_m2"
        ],
        "candidate_covariance_trace_m2": result[
            "candidate_covariance_trace_m2"
        ],
        "member_covariance_trace_m2": result["member_covariance_trace_m2"],
        "member_trace_fraction": result["member_trace_fraction"],
        "effective_member_count": result["effective_member_count"],
        "effective_anchor_count": result["effective_anchor_count"],
        "member_covariance_eigenvalues_m2": result[
            "member_covariance_eigenvalues_m2"
        ],
        "candidate_covariance_eigenvalues_m2": result[
            "candidate_covariance_eigenvalues_m2"
        ],
        "retained_anchor_mass": result["retained_anchor_mass"],
        **{
            name: values
            for name, values in result.items()
            if name.startswith("credible_")
            or name == "target_distance_from_mean_m"
        },
        "member_posterior_mass": result["member_posterior_mass"],
        "predicted_xy_m": result["predicted_xy_m"],
        "true_anchor_row_indices": result["true_anchor_row_indices"],
        "true_anchor_radio_nll_nats_per_ap": result[
            "true_anchor_radio_nll_nats_per_ap"
        ],
        "checkpoint_paths": [str(path) for path in paths],
        "checkpoint_sha256": {
            str(path): _sha256_file(path) for path in paths
        },
        "registry_sha256": registry["registry_sha256"],
        "source_sha256": _sha256_file(data_path),
        "elapsed_seconds": float(time.perf_counter() - started),
        "mechanism_diagnostic_contract": dict(MECHANISM_DIAGNOSTIC),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    printable = {
        "output": str(output),
        "artifact_status": payload["artifact_status"],
        "evaluation_role": role_name,
        "elapsed_seconds": payload["elapsed_seconds"],
        **payload["summary"],
    }
    print(json.dumps(printable, indent=2, allow_nan=False), flush=True)
    return printable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DR-JAM-v4 development roles")
    parser.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv"
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--mechanism", choices=("M0", "M1-g003", "M1-g010", "M1-g030", "M2", "M3"), required=True)
    parser.add_argument(
        "--evaluation-role",
        choices=("selection", "early-stop", "combined-shift"),
        default="selection",
    )
    parser.add_argument("--top-k", default="full")
    parser.add_argument(
        "--loss-temperature",
        type=float,
        default=float(MECHANISM_DIAGNOSTIC["loss_temperature_lambda"]),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-engineering-pilot", action="store_true")
    return parser


def main() -> None:
    run_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    main()
