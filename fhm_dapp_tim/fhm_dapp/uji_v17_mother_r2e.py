"""Execute the registered V17-R2E joint information-closure experiment."""

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

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as r0_execution
from fhm_dapp_tim.fhm_dapp.consensus import dense_alpha_consensus_prior
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _save_torch_new,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.projected_ensemble import ProjectedEnsemble
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_protocol import (
    load_registry as load_r0_registry,
    registered_role_keys,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r0_runtime_a2 import (
    _installed_adapters,
    _role_record,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2b_protocol import (
    COMPONENT_BATCH_SIZE,
    STUDENT_DF,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2c import (
    _group_vector,
    _joint_consensus_law,
    _member_top_l,
    _metric_record,
    _score_joint_law,
    studentized_max_t_family,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2d import (
    load_cache as load_r2d_cache,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2d_protocol import (
    load_registry as load_r2d_registry,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2e_protocol import (
    ADDITIONAL_MEMBER_INDICES,
    ADDITIONAL_MEMBER_SEEDS,
    ANCHOR_COUNT,
    BASE_MEMBER_COUNT,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CONFIDENCE,
    CONFIGURATIONS,
    CONSENSUS_ALPHA,
    CONSENSUS_EPSILON,
    DEFAULT_CACHE,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    DENSE_CONSENSUS_FROM_L,
    ENERGY_EQUIVALENCE_MARGIN_M,
    FINAL_MEMBER_CEILING_WITH_TWO_LATER_VALUES,
    HISTORICAL_REFERENCE,
    MAE_NONINFERIORITY_MARGIN_M,
    MEMBER_GRID,
    ORIGINAL_NUMERICAL_BLOCK,
    PANEL_MEMBER_COUNT,
    PROTOCOL_VERSION,
    R2D_CONTINUITY_REFERENCE,
    R2D_PARITY_TOLERANCE_M,
    SUPPORT_K_GRID,
    SUPPORT_L_GRID,
    VARIANTS,
    balanced_member_subsets,
    checkpoint_path,
    load_registry,
    variant_name,
)


def _validate_output(registry: Mapping[str, object], field: str, path: Path) -> None:
    allowed = Path(registry["allowed_outputs"][field]).resolve()
    if path.resolve() != allowed:
        raise ValueError(f"{field} path is outside the R2E registry")
    if path.exists():
        raise FileExistsError(path)


def _validate_checkpoint_output(registry: Mapping[str, object], index: int, path: Path) -> None:
    allowed = {
        Path(value).resolve()
        for value in registry["allowed_outputs"]["additional_checkpoints"]
    }
    if path.resolve() not in allowed or path.resolve() != checkpoint_path(index).resolve():
        raise ValueError("R2E checkpoint path is outside the registry")
    if path.exists():
        raise FileExistsError(path)


def _atomic_probability_panel(
    neural: torch.Tensor,
    distance: torch.Tensor,
    beta_neural: float,
    beta_prototype: float,
) -> torch.Tensor:
    """Preserve both frozen numerical blocks before adding the third block."""

    if neural.ndim != 3 or len(neural) != PANEL_MEMBER_COUNT:
        raise ValueError("R2E neural logits must contain the complete panel")
    if distance.shape != neural.shape[1:]:
        raise ValueError("R2E prototype distance is not aligned")
    blocks = (
        neural[:ORIGINAL_NUMERICAL_BLOCK],
        neural[ORIGINAL_NUMERICAL_BLOCK:BASE_MEMBER_COUNT],
        neural[BASE_MEMBER_COUNT:],
    )
    return torch.cat(
        [
            torch.softmax(
                float(beta_neural) * block
                - float(beta_prototype) * distance[None, :, :],
                dim=2,
            )
            for block in blocks
        ],
        dim=0,
    )


def _joint_consensus_law_memory_safe(
    member_prior: torch.Tensor,
    member_indices: torch.Tensor,
    member_radial: torch.Tensor,
    full_location: torch.Tensor,
    full_scale: torch.Tensor,
    subset: tuple[int, ...],
    *,
    K: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the exact dense definition only when sparse unions become larger."""

    if member_prior.shape[2] < DENSE_CONSENSUS_FROM_L:
        return _joint_consensus_law(
            member_prior,
            member_indices,
            member_radial,
            full_location,
            full_scale,
            subset,
            K=K,
            alpha=alpha,
        )
    subset_index = torch.tensor(subset, device=member_prior.device, dtype=torch.long)
    priors = member_prior.index_select(0, subset_index)
    indices = member_indices.index_select(0, subset_index)
    radial = member_radial.index_select(0, subset_index)
    member_weight = radial.square().clamp_min(1e-8).reciprocal()
    member_weight = member_weight / member_weight.sum(dim=0, keepdim=True).clamp_min(1e-30)
    consensus_indices, consensus_probability = dense_alpha_consensus_prior(
        priors,
        indices,
        member_weight,
        full_location.shape[2],
        int(K),
        epsilon=CONSENSUS_EPSILON,
        alpha=float(alpha),
    )
    locations = full_location.index_select(0, subset_index)
    scales = full_scale.index_select(0, subset_index)
    gather = consensus_indices[None, :, :, None].expand(len(subset), -1, -1, 2)
    selected_location = torch.gather(locations, 2, gather)
    selected_scale = torch.gather(scales, 2, gather)
    joint = member_weight.transpose(0, 1)[:, :, None] * consensus_probability[:, None, :]
    return (
        joint.reshape(len(consensus_probability), -1),
        selected_location.permute(1, 0, 2, 3).reshape(
            len(consensus_probability), -1, 2
        ),
        selected_scale.permute(1, 0, 2, 3).reshape(
            len(consensus_probability), -1, 2
        ),
    )


def _load_new_member_record(registry: Mapping[str, object], index: int) -> tuple[Path, dict]:
    path = checkpoint_path(index).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"member {index} protocol changed")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError(f"member {index} registry binding changed")
    if payload.get("member_index") != index:
        raise ValueError(f"member {index} identity changed")
    expected_seed = ADDITIONAL_MEMBER_SEEDS[index - ADDITIONAL_MEMBER_INDICES[0]]
    if payload.get("member_seed") != expected_seed:
        raise ValueError(f"member {index} seed changed")
    split = payload.get("split_contract", {})
    for key in (
        "development_localization_predictions",
        "outer_test_predictions",
        "official_validation_predictions",
    ):
        if split.get(key) != 0:
            raise ValueError(f"member {index} illegally accessed {key}")
    return path, payload


def train_member(
    *,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    index = int(member_index)
    if index not in ADDITIONAL_MEMBER_INDICES:
        raise ValueError("R2E trains only members 16 through 20")
    destination = checkpoint_path(index)
    _validate_checkpoint_output(registry, index, destination)
    r2d_registry = load_r2d_registry(registry["bindings"]["r2d_registry"]["path"])
    r0_registry = load_r0_registry(r2d_registry["bindings"]["r0_registry"]["path"])
    data = load_uji_v16(registry["data_contract"]["train_data"])
    fit_rows = r0_execution._registered_rows(r0_registry, "member_fit", data)
    stop_rows = r0_execution._registered_rows(r0_registry, "early_stop", data)
    refit_rows = r0_execution._registered_rows(r0_registry, "selection_refit", data)
    seed = ADDITIONAL_MEMBER_SEEDS[index - ADDITIONAL_MEMBER_INDICES[0]]
    device = torch.device(device_name)
    started = time.perf_counter()
    selected_epoch, schedule = r0_execution._fit_schedule(
        data,
        fit_rows,
        stop_rows,
        seed=seed,
        device=device,
        log_every=log_every,
    )
    model, stats, origin, device_mapping, refit_history = r0_execution._fit_refit(
        data,
        refit_rows,
        seed=seed,
        epochs=selected_epoch,
        device=device,
        log_every=log_every,
    )
    config = copy.deepcopy(FROZEN_TRAINING)
    config.update(
        {
            "training_protocol_version": PROTOCOL_VERSION,
            "model_schema": "v17_r2e_exchangeable_r0_member",
            "n_devices": len(device_mapping),
            "train_all": True,
            "split_strategy": "registered_selection_refit_v17",
            "validation_fraction": 0.0,
            "calibration_fraction": 0.0,
            "epochs": int(selected_epoch),
            "patience": r0_execution.PATIENCE,
            "lr": FROZEN_TRAINING["learning_rate"],
            "target_neighbors": FROZEN_TRAINING["target_neighbors"],
            "early_stop_aggregation": "coordinate_macro_mae",
        }
    )
    role_keys = tuple(sorted(registered_role_keys(r0_registry, "selection_refit")))
    if len(role_keys) != model.n_anchors:
        raise RuntimeError("R2E refit anchors do not equal registered coordinates")
    elapsed = float(time.perf_counter() - started)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "phase": "r2e_selection_refit_panel_extension",
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
            "member_fit": registry["data_contract"]["training_roles"]["member_fit"],
            "early_stop": registry["data_contract"]["training_roles"]["early_stop"],
            "selection_refit": registry["data_contract"]["training_roles"]["selection_refit"],
            "development_localization_predictions": 0,
            "outer_test_predictions": 0,
            "official_validation_predictions": 0,
        },
        "exchangeability_contract": {
            "same_algorithm_as_v17_r0": True,
            "only_changed_training_factor": "random_seed",
            "existing_panel_selected_by_outcome": False,
            "new_member_selected_by_outcome": False,
        },
        "device_id_mapping": {
            str(raw): mapped for raw, mapped in device_mapping.items()
        },
        "anchor_coordinate_keys": role_keys,
        "schedule_history": schedule,
        "refit_history": refit_history,
        "training_seconds": elapsed,
    }
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": _sha256_file(destination),
        "member_index": index,
        "seed": seed,
        "selected_epoch": int(selected_epoch),
        "anchors": int(model.n_anchors),
        "training_seconds": elapsed,
        "development_localization_predictions": 0,
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def check_members(registry_path: str | Path = DEFAULT_REGISTRY) -> dict:
    registry = load_registry(registry_path)
    records = []
    for index in ADDITIONAL_MEMBER_INDICES:
        path = checkpoint_path(index).resolve()
        record = {"member_index": index, "path": str(path), "exists": path.exists()}
        if path.exists():
            path, payload = _load_new_member_record(registry, index)
            record.update(
                {
                    "sha256": _sha256_file(path),
                    "seed": payload["member_seed"],
                    "selected_epoch": payload["epoch"],
                    "anchors": len(payload["anchor_coordinate_keys"]),
                }
            )
        records.append(record)
    return {
        "registry_sha256": registry["registry_sha256"],
        "additional_members": records,
        "all_complete": all(record["exists"] for record in records),
        "development_localization_predictions": 0,
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


@torch.no_grad()
def build_cache(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_CACHE,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    destination = Path(output).resolve()
    _validate_output(registry, "development_cache", destination)
    new_records = [
        _load_new_member_record(registry, index)
        for index in ADDITIONAL_MEMBER_INDICES
    ]
    old_paths = [Path(record["path"]) for record in registry["current_panel"]]
    new_paths = [path for path, _payload in new_records]
    checkpoints = old_paths + new_paths
    selection_csv = Path(registry["data_contract"]["selection_refit_csv"])
    development_csv = Path(registry["data_contract"]["development_csv"])
    selection_role = _role_record(selection_csv)
    development_role = _role_record(development_csv)
    device = torch.device(device_name)
    with _installed_adapters([selection_role, development_role]):
        ensemble = ProjectedEnsemble(
            checkpoints=checkpoints,
            anchor_data=selection_csv,
            device=device_name,
            dataset="UJI",
        )
        loaded = load_dataset(
            str(development_csv),
            dataset_type="UJI",
            device="cpu",
            reference_stats=ensemble.reference["stats"],
        )
    rss = loaded[0].float().cpu()
    exact = load_uji_v16(development_csv)
    expected_role = registry["data_contract"]["development_role"]
    if len(rss) != expected_role["row_count"] or not torch.equal(rss, exact["rss"]):
        raise ValueError("R2E development RSS or row count changed")
    if len(ensemble.checkpoints) != PANEL_MEMBER_COUNT:
        raise ValueError("R2E member count changed")

    r2d_registry = load_r2d_registry(registry["bindings"]["r2d_registry"]["path"])
    old_cache = load_r2d_cache(
        registry["bindings"]["r2d_cache"]["path"], registry=r2d_registry
    )
    models = [model for model, _payload in ensemble.checkpoints]
    reference_model = models[0]
    reference_seen = reference_model.encoder.seen_ap_mask.detach().cpu()
    reference_prototypes = reference_model.prototype_rss.detach().cpu()
    reference_coords = reference_model.anchor_coords.detach().cpu()
    for member_index, model in enumerate(models, start=1):
        if model.use_candidate_reranker:
            raise ValueError(f"member {member_index} unexpectedly uses a reranker")
        if model.residual_experts != 1:
            raise ValueError(f"member {member_index} is not an atomic residual law")
        if not math.isclose(float(model.student_df), STUDENT_DF, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"member {member_index} Student-t law changed")
        if model.n_anchors != ANCHOR_COUNT:
            raise ValueError(f"member {member_index} anchor count changed")
        if not torch.equal(model.encoder.seen_ap_mask.cpu(), reference_seen):
            raise ValueError("member seen-AP masks differ")
        if not torch.equal(model.prototype_rss.cpu(), reference_prototypes):
            raise ValueError("member prototype memories differ")
        if not torch.allclose(model.anchor_coords.cpu(), reference_coords, atol=1e-6):
            raise ValueError("member anchor coordinates differ")

    query = rss.to(device)
    neural_members = []
    location_members = []
    scale_members = []
    for member_index, model in enumerate(models[BASE_MEMBER_COUNT:], start=16):
        model.eval()
        neural_parts = []
        location_parts = []
        scale_parts = []
        for start in range(0, len(query), COMPONENT_BATCH_SIZE):
            batch = query[start : start + COMPONENT_BATCH_SIZE]
            embedding = model.encoder(batch)
            neural = model.retrieval_logits_from_embedding(embedding)
            indices = torch.arange(
                ANCHOR_COUNT, device=device, dtype=torch.long
            ).unsqueeze(0).expand(len(batch), -1)
            location, scale = model._components(embedding, indices)
            neural_parts.append(neural.cpu())
            location_parts.append(location.cpu())
            scale_parts.append(scale.cpu())
        neural_members.append(torch.cat(neural_parts))
        location_members.append(torch.cat(location_parts))
        scale_members.append(torch.cat(scale_parts))
        print(json.dumps({"cached_member": member_index, "rows": len(query), "anchors": ANCHOR_COUNT}))

    if not torch.allclose(reference_coords, old_cache["anchor_coordinates"], atol=1e-6):
        raise ValueError("R2E anchor coordinates differ from R2D")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "role": "development",
        "source_r2d_cache_sha256": registry["bindings"]["r2d_cache"]["file_sha256"],
        "development_csv_sha256": _sha256_file(development_csv),
        "selection_refit_csv_sha256": _sha256_file(selection_csv),
        "member_paths": tuple(str(path.resolve()) for path in checkpoints),
        "member_sha256": tuple(_sha256_file(path) for path in checkpoints),
        "coordinate_group_ids": old_cache["coordinate_group_ids"],
        "coordinate_group_keys": old_cache["coordinate_group_keys"],
        "target_xy": old_cache["target_xy"],
        "prototype_distance": old_cache["prototype_distance"],
        "member_neural_logits": torch.cat(
            [old_cache["member_neural_logits"], torch.stack(neural_members)], dim=0
        ),
        "member_locations": torch.cat(
            [old_cache["member_locations"], torch.stack(location_members)], dim=0
        ),
        "member_scales": torch.cat(
            [old_cache["member_scales"], torch.stack(scale_members)], dim=0
        ),
        "anchor_coordinates": old_cache["anchor_coordinates"],
        "coord_origin": old_cache["coord_origin"],
        "randomness": old_cache["randomness"],
        "construction_contract": {
            "members": PANEL_MEMBER_COUNT,
            "rows": int(len(rss)),
            "anchors": ANCHOR_COUNT,
            "old_fifteen_member_tensor_prefix_reused_exactly": True,
            "full_untruncated_support": True,
            "member_consensus_applied": False,
            "candidate_reranker_applied": False,
            "component_batch_size": COMPONENT_BATCH_SIZE,
            "development_scan_predictions": int(len(rss)),
            "member_laws_per_scan": PANEL_MEMBER_COUNT,
            "outer_test_predictions": 0,
            "official_validation_predictions": 0,
        },
    }
    _save_torch_new(destination, payload)
    return {
        "cache": str(destination),
        "cache_sha256": _sha256_file(destination),
        "rows": int(len(rss)),
        "coordinate_groups": len(payload["coordinate_group_keys"]),
        "members": PANEL_MEMBER_COUNT,
        "anchors": ANCHOR_COUNT,
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def load_cache(
    path: str | Path = DEFAULT_CACHE, *, registry: Mapping[str, object]
) -> dict:
    source = Path(path).resolve()
    if source != Path(registry["allowed_outputs"]["development_cache"]).resolve():
        raise ValueError("cache path is outside the R2E registry")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V17-R2E cache")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("R2E cache and registry differ")
    if payload.get("source_r2d_cache_sha256") != registry["bindings"]["r2d_cache"]["file_sha256"]:
        raise ValueError("R2E cache and R2D prefix differ")
    rows = registry["data_contract"]["development_role"]["row_count"]
    if payload["member_neural_logits"].shape != (PANEL_MEMBER_COUNT, rows, ANCHOR_COUNT):
        raise ValueError("R2E cache neural shape changed")
    expected_components = (PANEL_MEMBER_COUNT, rows, ANCHOR_COUNT, 2)
    if payload["member_locations"].shape != expected_components:
        raise ValueError("R2E cache location shape changed")
    if payload["member_scales"].shape != expected_components:
        raise ValueError("R2E cache scale shape changed")
    if tuple(_sha256_file(path) for path in payload["member_paths"]) != tuple(payload["member_sha256"]):
        raise ValueError("R2E cache member binding changed")
    r2d_registry = load_r2d_registry(registry["bindings"]["r2d_registry"]["path"])
    old = load_r2d_cache(registry["bindings"]["r2d_cache"]["path"], registry=r2d_registry)
    for key in ("member_neural_logits", "member_locations", "member_scales"):
        if not torch.equal(payload[key][:BASE_MEMBER_COUNT], old[key]):
            raise ValueError(f"R2E did not preserve the R2D {key} prefix")
    return payload


def _select_information_budget(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    energy_groups: Mapping[str, np.ndarray],
    mae_family: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    config_by_name = {
        variant_name(E, L, K): (E, L, K) for E, L, K in CONFIGURATIONS
    }
    eligible = [
        name
        for name in VARIANTS
        if mae_family["comparisons"][name]["simultaneous_upper_bound_m"]
        <= MAE_NONINFERIORITY_MARGIN_M
    ]
    if not eligible:
        raise RuntimeError("no R2E configuration passed the MAE gate")
    order = {name: index for index, name in enumerate(VARIANTS)}
    global_best = min(
        eligible,
        key=lambda name: (
            metrics[name]["energy_score_2d_m"]["coordinate_macro_m"],
            order[name],
        ),
    )
    differences = {
        name: energy_groups[name] - energy_groups[global_best]
        for name in eligible
        if name != global_best
    }
    if differences:
        energy_family = studentized_max_t_family(
            differences,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
            confidence=CONFIDENCE,
            orientation="candidate_minus_eligible_global_energy_minimum",
        )
    else:
        energy_family = {
            "difference_orientation": "candidate_minus_eligible_global_energy_minimum",
            "comparisons": {},
            "critical_max_t": 0.0,
            "confidence": CONFIDENCE,
            "family_size": 0,
            "coordinate_groups": len(energy_groups[global_best]),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "studentized": True,
            "common_coordinate_group_resamples": True,
        }
    platform = [global_best]
    platform.extend(
        name
        for name in eligible
        if name != global_best
        and energy_family["comparisons"][name]["simultaneous_upper_bound_m"]
        <= ENERGY_EQUIVALENCE_MARGIN_M
    )

    def information_key(name: str) -> tuple:
        E, L, K = config_by_name[name]
        return (
            E * L + K,
            E,
            L,
            K,
            metrics[name]["energy_score_2d_m"]["coordinate_macro_m"],
            order[name],
        )

    interior_members = [
        name
        for name in platform
        if config_by_name[name][0] <= FINAL_MEMBER_CEILING_WITH_TWO_LATER_VALUES
    ]
    selected = min(interior_members, key=information_key) if interior_members else None
    if selected is None:
        selected_payload = None
    else:
        E, L, K = config_by_name[selected]
        selected_payload = {
            "variant": selected,
            "E": E,
            "L": L,
            "K": K,
            "alpha": CONSENSUS_ALPHA,
            "information_budget_E_times_L_plus_K": E * L + K,
            "later_member_counts": [value for value in MEMBER_GRID if value > E],
            "exact_finite_support_endpoint_compared": True,
        }
    profiles = {}
    for E in MEMBER_GRID:
        candidates = [name for name in eligible if config_by_name[name][0] == E]
        if not candidates:
            profiles[str(E)] = None
            continue
        best = min(
            candidates,
            key=lambda name: (
                metrics[name]["energy_score_2d_m"]["coordinate_macro_m"],
                order[name],
            ),
        )
        profiles[str(E)] = {
            "variant": best,
            "energy_score_coordinate_macro_m": metrics[best]["energy_score_2d_m"][
                "coordinate_macro_m"
            ],
            "exact_mixture_mean_mae_m": metrics[best]["exact_mixture_mean_mae_m"][
                "coordinate_macro_m"
            ],
        }
    return (
        {
            "eligible_global_energy_minimum": global_best,
            "mae_eligible_variants": eligible,
            "energy_equivalence_platform": platform,
            "registered_selection": selected_payload,
            "member_information_saturated": selected is not None,
            "support_boundary_closed_by_exact_endpoint": True,
            "support_information_saturated": None if selected is None else True,
            "requires_member_panel_extension": selected is None,
            "requires_support_grid_extension": False,
            "may_register_later_outer_test_protocol": selected_payload is not None,
            "may_access_outer_test": False,
            "profiled_best_by_E": profiles,
        },
        energy_family,
    )


@torch.no_grad()
def evaluate(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    cache_path: str | Path = DEFAULT_CACHE,
    output: str | Path = DEFAULT_RESULT,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    destination = Path(output).resolve()
    _validate_output(registry, "development_result", destination)
    cache = load_cache(cache_path, registry=registry)
    device = torch.device(device_name)
    beta_n = registry["frozen_atomic_and_consensus_law"]["beta_neural"]
    beta_p = registry["frozen_atomic_and_consensus_law"]["beta_prototype"]
    full_probability = _atomic_probability_panel(
        cache["member_neural_logits"].to(device),
        cache["prototype_distance"].to(device),
        beta_n,
        beta_p,
    )
    full_location = cache["member_locations"].to(device)
    full_scale = cache["member_scales"].to(device)
    target = cache["target_xy"].to(device)
    randomness = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in cache["randomness"].items()
    }
    rows = len(target)
    row_energy = {name: torch.zeros(rows, dtype=torch.float64) for name in VARIANTS}
    row_mae = {name: torch.zeros(rows, dtype=torch.float64) for name in VARIANTS}
    retained_audit = {}
    continuity = None

    for L in SUPPORT_L_GRID:
        member_prior, member_indices, member_radial, retained_mass = _member_top_l(
            full_probability, full_location, full_scale, L
        )
        retained_audit[str(L)] = {
            "mean_retained_mass": float(retained_mass.mean()),
            "q10_retained_mass": float(torch.quantile(retained_mass, 0.10)),
            "median_retained_mass": float(torch.quantile(retained_mass, 0.50)),
            "q90_retained_mass": float(torch.quantile(retained_mass, 0.90)),
            "mean_reverse_kl_nats": float((-retained_mass.clamp_min(1e-30).log()).mean()),
        }
        for E in MEMBER_GRID:
            subsets = balanced_member_subsets(E)
            valid_k = tuple(K for K in SUPPORT_K_GRID if K <= L)
            maximum_k = max(valid_k)
            for subset in subsets:
                maximum_probability, maximum_location, maximum_scale = (
                    _joint_consensus_law_memory_safe(
                        member_prior,
                        member_indices,
                        member_radial,
                        full_location,
                        full_scale,
                        subset,
                        K=maximum_k,
                        alpha=CONSENSUS_ALPHA,
                    )
                )
                probability_grid = maximum_probability.reshape(rows, E, maximum_k)
                location_grid = maximum_location.reshape(rows, E, maximum_k, 2)
                scale_grid = maximum_scale.reshape(rows, E, maximum_k, 2)
                for K in valid_k:
                    probability = probability_grid[:, :, :K]
                    probability = probability / probability.sum(
                        dim=(1, 2), keepdim=True
                    ).clamp_min(1e-30)
                    probability = probability.reshape(rows, -1)
                    location = location_grid[:, :, :K, :].reshape(rows, -1, 2)
                    scale = scale_grid[:, :, :K, :].reshape(rows, -1, 2)
                    energy, mae = _score_joint_law(
                        probability, location, scale, target, randomness
                    )
                    name = variant_name(E, L, K)
                    divisor = float(len(subsets))
                    row_energy[name] += energy / divisor
                    row_mae[name] += mae / divisor
            print(json.dumps({
                "completed_L": L,
                "completed_E": E,
                "balanced_subsets": len(subsets),
                "K_values": list(valid_k),
                "consensus_execution": "dense_exact" if L >= DENSE_CONSENSUS_FROM_L else "sparse_exact",
            }))

        if L == 192:
            parity_prior, parity_indices, parity_radial, _parity_mass = _member_top_l(
                full_probability[:BASE_MEMBER_COUNT],
                full_location[:BASE_MEMBER_COUNT],
                full_scale[:BASE_MEMBER_COUNT],
                L,
            )
            probability, location, scale = _joint_consensus_law(
                parity_prior,
                parity_indices,
                parity_radial,
                full_location[:BASE_MEMBER_COUNT],
                full_scale[:BASE_MEMBER_COUNT],
                tuple(range(BASE_MEMBER_COUNT)),
                K=48,
                alpha=CONSENSUS_ALPHA,
            )
            parity_energy, parity_mae = _score_joint_law(
                probability, location, scale, target, randomness
            )
            continuity = {"energy": parity_energy, "mae": parity_mae}

    if continuity is None:
        raise RuntimeError("R2E continuity configuration was not evaluated")
    group_ids = cache["coordinate_group_ids"]
    group_count = len(cache["coordinate_group_keys"])
    r2d = json.loads(
        Path(registry["bindings"]["r2d_result"]["path"]).read_text(encoding="utf-8")
    )
    continuity_differences = {}
    for endpoint, key in (
        ("energy", "energy_score_2d_m"),
        ("mae", "exact_mixture_mean_mae_m"),
    ):
        observed = _group_vector(continuity[endpoint], group_ids, group_count).numpy()
        expected = np.asarray(
            r2d["coordinate_group_vectors"][key][R2D_CONTINUITY_REFERENCE],
            dtype=np.float64,
        )
        if observed.shape != expected.shape:
            raise ValueError("R2D/R2E continuity vectors have different shapes")
        absolute = np.abs(observed - expected)
        continuity_differences[endpoint] = {
            "r2e_coordinate_macro_m": float(observed.mean()),
            "r2d_coordinate_macro_m": float(expected.mean()),
            "absolute_coordinate_macro_difference_m": float(
                abs(observed.mean() - expected.mean())
            ),
            "maximum_absolute_group_difference_m": float(absolute.max(initial=0.0)),
            "mean_absolute_group_difference_m": float(absolute.mean()),
        }
    maximum_continuity = max(
        value["absolute_coordinate_macro_difference_m"]
        for value in continuity_differences.values()
    )
    continuity_gate = {
        "reference_variant": R2D_CONTINUITY_REFERENCE,
        "endpoints": continuity_differences,
        "maximum_absolute_coordinate_macro_endpoint_difference_m": maximum_continuity,
        "tolerance_m": R2D_PARITY_TOLERANCE_M,
        "passed": maximum_continuity <= R2D_PARITY_TOLERANCE_M,
    }
    if not continuity_gate["passed"]:
        raise RuntimeError("R2E failed the frozen R2D continuity gate")

    metrics = {}
    energy_groups = {}
    mae_groups = {}
    for name in VARIANTS:
        energy_record, energy_group = _metric_record(
            row_energy[name], group_ids, group_count
        )
        mae_record, mae_group = _metric_record(row_mae[name], group_ids, group_count)
        metrics[name] = {
            "energy_score_2d_m": energy_record,
            "exact_mixture_mean_mae_m": mae_record,
        }
        energy_groups[name] = energy_group
        mae_groups[name] = mae_group

    r2c = json.loads(
        Path(registry["bindings"]["r2c_result"]["path"]).read_text(encoding="utf-8")
    )
    historical_mae = np.asarray(
        r2c["coordinate_group_vectors"]["exact_mixture_mean_mae_m"][
            HISTORICAL_REFERENCE
        ],
        dtype=np.float64,
    )
    mae_family = studentized_max_t_family(
        {name: mae_groups[name] - historical_mae for name in VARIANTS},
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
        confidence=CONFIDENCE,
        orientation="candidate_minus_frozen_R2C_balanced_historical_configuration",
    )
    decision, energy_family = _select_information_budget(
        metrics, energy_groups, mae_family
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "cache_sha256": _sha256_file(cache_path),
        "metrics": metrics,
        "coordinate_group_vectors": {
            "energy_score_2d_m": {
                name: energy_groups[name].tolist() for name in VARIANTS
            },
            "exact_mixture_mean_mae_m": {
                name: mae_groups[name].tolist() for name in VARIANTS
            },
        },
        "simultaneous_inference": {
            "mae_candidate_minus_frozen_historical": mae_family,
            "energy_candidate_minus_eligible_global_minimum": energy_family,
        },
        "registered_decision": decision,
        "r2d_continuity_gate": continuity_gate,
        "retained_mass_diagnostic": retained_audit,
        "exact_support_endpoint": {
            "anchors": ANCHOR_COUNT,
            "L": ANCHOR_COUNT,
            "K": ANCHOR_COUNT,
            "truncation": False,
        },
        "development_scan_predictions": rows,
        "registered_configurations": len(VARIANTS),
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(destination, result)
    return result


def check_cache(
    *, registry_path: str | Path = DEFAULT_REGISTRY, cache_path: str | Path = DEFAULT_CACHE
) -> dict:
    registry = load_registry(registry_path)
    cache = load_cache(cache_path, registry=registry)
    return {
        "registry_sha256": registry["registry_sha256"],
        "cache_sha256": _sha256_file(cache_path),
        "members": cache["construction_contract"]["members"],
        "rows": cache["construction_contract"]["rows"],
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def check_result(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    result_path: str | Path = DEFAULT_RESULT,
) -> dict:
    registry = load_registry(registry_path)
    source = Path(result_path).resolve()
    if source != Path(registry["allowed_outputs"]["development_result"]).resolve():
        raise ValueError("result path is outside the R2E registry")
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("result_sha256") != _canonical_json_sha256(
        result, omit="result_sha256"
    ):
        raise ValueError("R2E result self-hash failed")
    if result.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("R2E result and registry differ")
    if result.get("cache_sha256") != _sha256_file(
        registry["allowed_outputs"]["development_cache"]
    ):
        raise ValueError("R2E result and cache differ")
    return {
        "registry_sha256": registry["registry_sha256"],
        "result_sha256": result["result_sha256"],
        "registered_decision": result["registered_decision"],
        "continuity_gate": result["r2d_continuity_gate"],
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-member")
    train.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train.add_argument("--member-index", type=int, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=5)
    members = sub.add_parser("check-members")
    members.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    cache = sub.add_parser("build-cache")
    cache.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    cache.add_argument("--output", default=str(DEFAULT_CACHE))
    cache.add_argument("--device", default="cuda")
    cache_check = sub.add_parser("check-cache")
    cache_check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    cache_check.add_argument("--cache", default=str(DEFAULT_CACHE))
    score = sub.add_parser("evaluate")
    score.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    score.add_argument("--cache", default=str(DEFAULT_CACHE))
    score.add_argument("--output", default=str(DEFAULT_RESULT))
    score.add_argument("--device", default="cuda")
    result_check = sub.add_parser("check-result")
    result_check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    result_check.add_argument("--result", default=str(DEFAULT_RESULT))
    args = parser.parse_args(argv)
    if args.command == "train-member":
        payload = train_member(
            member_index=args.member_index,
            registry_path=args.registry,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "check-members":
        payload = check_members(args.registry)
    elif args.command == "build-cache":
        payload = build_cache(
            registry_path=args.registry,
            output=args.output,
            device_name=args.device,
        )
    elif args.command == "check-cache":
        payload = check_cache(registry_path=args.registry, cache_path=args.cache)
    elif args.command == "evaluate":
        payload = evaluate(
            registry_path=args.registry,
            cache_path=args.cache,
            output=args.output,
            device_name=args.device,
        )
    else:
        payload = check_result(registry_path=args.registry, result_path=args.result)
    if args.command == "evaluate":
        payload = {
            "registry_sha256": payload["registry_sha256"],
            "result_sha256": payload["result_sha256"],
            "registered_decision": payload["registered_decision"],
            "continuity_gate": payload["r2d_continuity_gate"],
            "outer_test_predictions": 0,
            "official_validation_predictions": 0,
        }
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

