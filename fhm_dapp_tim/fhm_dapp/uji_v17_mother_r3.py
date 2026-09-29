"""Execute the frozen V17-R3 outer-refit and one-shot confirmation."""

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
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b_execution
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f as r2f_execution
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f_runtime_a1 as r2f_a1
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import common_student_t_randomness
from fhm_dapp_tim.fhm_dapp.consensus import dense_alpha_consensus_prior
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _save_torch_new,
    _sha256_file,
    _sha256_indices,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.projected_ensemble import ProjectedEnsemble
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r0_runtime_a2 import (
    _installed_adapters,
    _role_record,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2c import (
    _member_top_l,
    _metric_record,
    _score_joint_law,
    studentized_max_t_family,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r3_protocol import (
    BETA_NEURAL,
    BETA_PROTOTYPE,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    COMPONENT_BATCH_SIZE,
    CONFIDENCE,
    CONFIGURATIONS,
    CONSENSUS_EPSILON,
    DEFAULT_OUTER_REFIT_CSV,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    HISTORICAL_NAME,
    MEMBER_SEEDS,
    OUTER_REFIT_ANCHORS,
    OUTER_TEST_COORDINATES,
    OUTER_TEST_ROWS,
    PANEL_MEMBER_COUNT,
    PREDICTIVE_DRAWS_PER_STREAM,
    PREDICTIVE_SEED,
    PROTOCOL_VERSION,
    SELECTED_NAME,
    STUDENT_DF,
    balanced_member_subsets,
    final_checkpoint_path,
    load_registry,
)


def _validate_output(registry: Mapping[str, object], field: str, path: Path) -> None:
    allowed = registry["allowed_outputs"]
    if field == "outer_refit_checkpoint":
        expected = {Path(value).resolve() for value in allowed["outer_refit_checkpoints"]}
        if path.resolve() not in expected:
            raise ValueError("checkpoint path is outside the R3 registry")
    else:
        key = {"outer_refit_csv": "outer_refit_csv", "outer_result": "outer_result"}[field]
        if path.resolve() != Path(allowed[key]).resolve():
            raise ValueError(f"{field} path is outside the R3 registry")
    if path.exists():
        raise FileExistsError(path)


def _registered_rows(registry: Mapping[str, object], role: str, data: dict) -> np.ndarray:
    record = registry["data_contract"][role]
    rows = np.asarray(record["row_indices"], dtype=np.int64)
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise ValueError(f"R3 {role} indices are not unique and one-dimensional")
    if len(rows) > 1 and bool(np.any(rows[1:] <= rows[:-1])):
        raise ValueError(f"R3 {role} indices do not preserve source order")
    if _sha256_indices(rows) != record["row_index_sha256"]:
        raise ValueError(f"R3 {role} row hash failed")
    observed_keys = {data["keys"][int(index)] for index in rows}
    expected_keys = {tuple(value) for value in record["coordinate_keys"]}
    if observed_keys != expected_keys:
        raise ValueError(f"R3 {role} coordinate identities changed")
    if len(rows) != record["row_count"] or len(observed_keys) != record["coordinate_count"]:
        raise ValueError(f"R3 {role} role size changed")
    return rows


def materialize_outer_refit(
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict:
    registry = load_registry(registry_path)
    destination = Path(registry["allowed_outputs"]["outer_refit_csv"])
    _validate_output(registry, "outer_refit_csv", destination)
    data = load_uji_v16(registry["data_contract"]["train_data"])
    rows = _registered_rows(registry, "outer_refit", data)
    r0_execution._materialize_csv(
        Path(registry["data_contract"]["train_data"]), destination, rows
    )
    derived = load_uji_v16(destination)
    expected_keys = {tuple(value) for value in registry["data_contract"]["outer_refit"]["coordinate_keys"]}
    if set(derived["keys"]) != expected_keys or len(derived["keys"]) != len(rows):
        raise RuntimeError("materialized R3 outer-refit role changed")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "outer_refit_csv": str(destination.resolve()),
        "outer_refit_csv_sha256": _sha256_file(destination),
        "rows": int(len(rows)),
        "coordinates": int(len(expected_keys)),
        "outer_test_rows_consumed": 0,
        "official_validation_rows_consumed": 0,
    }


def _source_member_record(registry: Mapping[str, object], index: int) -> tuple[dict, dict]:
    record = registry["training_contract"]["source_members"][index - 1]
    if record["member_index"] != index or record["seed"] != MEMBER_SEEDS[index - 1]:
        raise ValueError(f"R3 source member order changed for member {index}")
    if _sha256_file(record["path"]) != record["file_sha256"]:
        raise ValueError(f"R3 source checkpoint changed for member {index}")
    payload = torch.load(record["path"], map_location="cpu", weights_only=True)
    if payload.get("epoch") != record["selected_epoch"]:
        raise ValueError(f"R3 source epoch changed for member {index}")
    return record, payload


def train_member(
    *,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    index = int(member_index)
    if not 1 <= index <= PANEL_MEMBER_COUNT:
        raise ValueError("R3 trains exactly members 1 through 25")
    destination = Path(registry["allowed_outputs"]["outer_refit_checkpoints"][index - 1])
    if destination.resolve() != final_checkpoint_path(index).resolve():
        raise ValueError("R3 final checkpoint naming contract changed")
    _validate_output(registry, "outer_refit_checkpoint", destination)
    outer_refit_csv = Path(registry["allowed_outputs"]["outer_refit_csv"])
    if not outer_refit_csv.exists():
        raise FileNotFoundError("materialize the R3 outer-refit role before training")

    source_record, source_payload = _source_member_record(registry, index)
    data = load_uji_v16(registry["data_contract"]["train_data"])
    refit_rows = _registered_rows(registry, "outer_refit", data)
    device = torch.device(device_name)
    seed = MEMBER_SEEDS[index - 1]
    epochs = int(source_record["selected_epoch"])
    started = time.perf_counter()
    model, stats, origin, device_mapping, history = r0_execution._fit_refit(
        data,
        refit_rows,
        seed=seed,
        epochs=epochs,
        device=device,
        log_every=log_every,
    )
    if model.n_anchors != OUTER_REFIT_ANCHORS:
        raise RuntimeError("R3 final member anchor count changed")
    config = copy.deepcopy(FROZEN_TRAINING)
    config.update(
        {
            "training_protocol_version": PROTOCOL_VERSION,
            "model_schema": "v17_r3_outer_refit_historical_acrhap_member",
            "n_devices": len(device_mapping),
            "train_all": True,
            "split_strategy": "registered_outer_refit_v17_r3",
            "validation_fraction": 0.0,
            "calibration_fraction": 0.0,
            "epochs": epochs,
            "patience": source_payload["config"].get("patience"),
            "lr": FROZEN_TRAINING["learning_rate"],
            "target_neighbors": FROZEN_TRAINING["target_neighbors"],
            "early_stop_aggregation": "coordinate_macro_mae",
            "epoch_selection_source": "frozen_inner_member_fit_to_early_stop",
        }
    )
    role_keys = tuple(
        tuple(value)
        for value in registry["data_contract"]["outer_refit"]["coordinate_keys"]
    )
    if len(role_keys) != model.n_anchors:
        raise RuntimeError("R3 final anchors do not equal registered outer-refit coordinates")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "phase": "r3_outer_refit",
        "member_index": index,
        "member_seed": seed,
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "config": config,
        "stats": stats,
        "coord_origin": np.asarray(origin, dtype=np.float32),
        "conformal_quantiles": {},
        "epoch": epochs,
        "val_mae": float(source_payload.get("val_mae", math.nan)),
        "seed": seed,
        "development_metrics": None,
        "confirmation_metrics": None,
        "source_selection_checkpoint": {
            "path": source_record["path"],
            "file_sha256": source_record["file_sha256"],
            "selected_epoch": epochs,
            "epoch_only_reused_no_weights_reused": True,
        },
        "split_contract": {
            "outer_refit": registry["data_contract"]["outer_refit"],
            "outer_refit_rows_consumed": int(len(refit_rows)),
            "outer_test_features_consumed": False,
            "outer_test_labels_consumed": False,
            "official_features_consumed": False,
            "official_labels_consumed": False,
        },
        "device_id_mapping": {
            str(raw): mapped for raw, mapped in device_mapping.items()
        },
        "anchor_coordinate_keys": role_keys,
        "schedule_history": [],
        "refit_history": history,
        "outer_refit_csv_sha256": _sha256_file(outer_refit_csv),
        "training_seconds": float(time.perf_counter() - started),
    }
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": _sha256_file(destination),
        "member_index": index,
        "seed": seed,
        "epochs": epochs,
        "anchors": model.n_anchors,
        "training_seconds": payload["training_seconds"],
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def check_members(registry_path: str | Path = DEFAULT_REGISTRY) -> dict:
    registry = load_registry(registry_path)
    outer_refit_csv = Path(registry["allowed_outputs"]["outer_refit_csv"])
    if not outer_refit_csv.exists():
        raise FileNotFoundError("R3 outer-refit CSV is absent")
    csv_hash = _sha256_file(outer_refit_csv)
    expected_keys = tuple(
        tuple(value) for value in registry["data_contract"]["outer_refit"]["coordinate_keys"]
    )
    records = []
    for index, value in enumerate(
        registry["allowed_outputs"]["outer_refit_checkpoints"], start=1
    ):
        path = Path(value)
        if not path.exists():
            records.append({"member_index": index, "exists": False})
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        source = registry["training_contract"]["source_members"][index - 1]
        checks = {
            "protocol": payload.get("protocol_version") == PROTOCOL_VERSION,
            "registry": payload.get("registry_sha256") == registry["registry_sha256"],
            "phase": payload.get("phase") == "r3_outer_refit",
            "identity": payload.get("member_index") == index
            and payload.get("member_seed") == MEMBER_SEEDS[index - 1],
            "epoch": payload.get("epoch") == source["selected_epoch"],
            "source_hash": payload.get("source_selection_checkpoint", {}).get(
                "file_sha256"
            )
            == source["file_sha256"],
            "outer_refit_csv": payload.get("outer_refit_csv_sha256") == csv_hash,
            "anchor_count": len(payload.get("anchor_coordinate_keys", ()))
            == OUTER_REFIT_ANCHORS,
            "anchor_identity": tuple(map(tuple, payload.get("anchor_coordinate_keys", ())))
            == expected_keys,
            "outer_test_unconsumed": payload.get("split_contract", {}).get(
                "outer_test_labels_consumed"
            )
            is False,
            "official_unconsumed": payload.get("split_contract", {}).get(
                "official_labels_consumed"
            )
            is False,
            "train_all": payload.get("config", {}).get("train_all") is True,
            "no_reranker": payload.get("config", {}).get("candidate_reranker")
            == FROZEN_TRAINING["candidate_reranker"],
        }
        if not all(checks.values()):
            raise ValueError(f"R3 final member {index} failed checks: {checks}")
        records.append(
            {
                "member_index": index,
                "exists": True,
                "file_sha256": _sha256_file(path),
                "seed": payload["member_seed"],
                "epochs": payload["epoch"],
                "checks": checks,
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "outer_refit_csv_sha256": csv_hash,
        "members": records,
        "complete_members": sum(record["exists"] for record in records),
        "all_complete": all(record["exists"] for record in records),
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def _dense_joint_law(
    member_prior: torch.Tensor,
    member_indices: torch.Tensor,
    member_radial: torch.Tensor,
    full_location: torch.Tensor,
    full_scale: torch.Tensor,
    subset: tuple[int, ...],
    *,
    top_k: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not subset:
        raise ValueError("R3 member subset cannot be empty")
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
        int(top_k),
        epsilon=CONSENSUS_EPSILON,
        alpha=float(alpha),
    )
    locations = full_location.index_select(0, subset_index)
    scales = full_scale.index_select(0, subset_index)
    gather = consensus_indices[None, :, :, None].expand(len(subset), -1, -1, 2)
    selected_location = torch.gather(locations, 2, gather)
    selected_scale = torch.gather(scales, 2, gather)
    joint = member_weight.transpose(0, 1)[:, :, None] * consensus_probability[:, None, :]
    rows = len(consensus_probability)
    return (
        joint.reshape(rows, -1),
        selected_location.permute(1, 0, 2, 3).reshape(rows, -1, 2),
        selected_scale.permute(1, 0, 2, 3).reshape(rows, -1, 2),
    )


def _validate_loaded_panel(ensemble: ProjectedEnsemble) -> None:
    if len(ensemble.checkpoints) != PANEL_MEMBER_COUNT:
        raise ValueError("R3 loaded panel size changed")
    models = [model for model, _payload in ensemble.checkpoints]
    reference = models[0]
    for index, model in enumerate(models, start=1):
        if model.n_anchors != OUTER_REFIT_ANCHORS:
            raise ValueError(f"R3 member {index} anchor count changed")
        if model.use_candidate_reranker:
            raise ValueError(f"R3 member {index} unexpectedly uses a reranker")
        if model.residual_experts != 1:
            raise ValueError(f"R3 member {index} is not an atomic residual law")
        if not math.isclose(float(model.student_df), STUDENT_DF, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"R3 member {index} Student-t law changed")
        if not torch.equal(model.encoder.seen_ap_mask.cpu(), reference.encoder.seen_ap_mask.cpu()):
            raise ValueError("R3 member seen-AP masks differ")
        if not torch.equal(model.prototype_rss.cpu(), reference.prototype_rss.cpu()):
            raise ValueError("R3 member prototype memories differ")
        if not torch.allclose(model.anchor_coords.cpu(), reference.anchor_coords.cpu(), atol=1e-6):
            raise ValueError("R3 member anchor coordinates differ")


def _subset_run_records(
    subset_energy: torch.Tensor,
    subset_mae: torch.Tensor,
    subsets: tuple[tuple[int, ...], ...],
    group_ids: torch.Tensor,
    group_count: int,
) -> list[dict]:
    records = []
    for index, subset in enumerate(subsets):
        energy, _ = _metric_record(subset_energy[index], group_ids, group_count)
        mae, _ = _metric_record(subset_mae[index], group_ids, group_count)
        records.append(
            {
                "realization": index + 1,
                "member_indices_one_based": [value + 1 for value in subset],
                "energy_score_coordinate_macro_m": energy["coordinate_macro_m"],
                "exact_mixture_mean_mae_coordinate_macro_m": mae["coordinate_macro_m"],
            }
        )
    return records


@torch.no_grad()
def evaluate_outer(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    destination = Path(registry["allowed_outputs"]["outer_result"])
    _validate_output(registry, "outer_result", destination)
    member_audit = check_members(registry_path)
    if member_audit["all_complete"] is not True:
        raise RuntimeError("all 25 R3 outer-refit members must exist before outer access")

    # This is the only line of the R3 workflow that materializes the registered
    # outer-test row indices.  No official-file path is accepted or opened.
    data = load_uji_v16(registry["data_contract"]["train_data"])
    outer_rows = _registered_rows(registry, "outer_test", data)
    if len(outer_rows) != OUTER_TEST_ROWS:
        raise ValueError("R3 outer-test row count changed at access")

    checkpoint_paths = [
        Path(value) for value in registry["allowed_outputs"]["outer_refit_checkpoints"]
    ]
    outer_refit_csv = Path(registry["allowed_outputs"]["outer_refit_csv"])
    outer_refit_role = _role_record(outer_refit_csv)
    with _installed_adapters([outer_refit_role]):
        ensemble = ProjectedEnsemble(
            checkpoints=checkpoint_paths,
            anchor_data=outer_refit_csv,
            device=device_name,
            dataset="UJI",
        )
    _validate_loaded_panel(ensemble)
    device = torch.device(device_name)
    models = [model for model, _payload in ensemble.checkpoints]
    for model in models:
        model.eval()

    rss = data["rss"][outer_rows].float().cpu()
    origin = np.asarray(ensemble.reference["coord_origin"], dtype=np.float64)
    target = torch.from_numpy(data["raw_xy"][outer_rows] - origin[None, :]).double()
    group_ids, group_keys = r2b_execution._coordinate_groups(
        [data["keys"][int(index)] for index in outer_rows]
    )
    expected_group_keys = tuple(
        tuple(value) for value in registry["data_contract"]["outer_test"]["coordinate_keys"]
    )
    if group_keys != expected_group_keys or len(group_keys) != OUTER_TEST_COORDINATES:
        raise ValueError("R3 outer-test coordinate group order changed")
    randomness = common_student_t_randomness(
        len(rss),
        draws=PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=STUDENT_DF,
        seed=PREDICTIVE_SEED,
    )
    randomness = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in randomness.items()
    }
    target_device = target.to(device)
    subsets_by_name = {
        name: balanced_member_subsets(configuration["E"])
        for name, configuration in CONFIGURATIONS.items()
    }
    subset_energy = {
        name: torch.zeros((len(subsets), len(rss)), dtype=torch.float64)
        for name, subsets in subsets_by_name.items()
    }
    subset_mae = {
        name: torch.zeros((len(subsets), len(rss)), dtype=torch.float64)
        for name, subsets in subsets_by_name.items()
    }
    retained_values: dict[str, list[torch.Tensor]] = {name: [] for name in CONFIGURATIONS}

    with r2f_a1._deterministic_algorithms():
        for start in range(0, len(rss), COMPONENT_BATCH_SIZE):
            stop = min(start + COMPONENT_BATCH_SIZE, len(rss))
            batch = rss[start:stop].to(device)
            prototype_distance = r2b_execution._prototype_distance(models[0], batch)
            neural_members = []
            location_members = []
            scale_members = []
            anchor_indices = torch.arange(
                OUTER_REFIT_ANCHORS, device=device, dtype=torch.long
            ).unsqueeze(0).expand(len(batch), -1)
            for model in models:
                embedding = model.encoder(batch)
                neural_members.append(model.retrieval_logits_from_embedding(embedding))
                location, scale = model._components(embedding, anchor_indices)
                location_members.append(location)
                scale_members.append(scale)
            neural = torch.stack(neural_members)
            location = torch.stack(location_members)
            scale = torch.stack(scale_members)
            full_probability = r2f_execution._atomic_probability_panel(
                neural, prototype_distance, BETA_NEURAL, BETA_PROTOTYPE
            )
            batch_randomness = {
                key: (
                    value[start:stop]
                    if isinstance(value, torch.Tensor) and value.ndim > 0
                    else value
                )
                for key, value in randomness.items()
            }
            for name, configuration in CONFIGURATIONS.items():
                member_prior, member_indices, member_radial, retained_mass = _member_top_l(
                    full_probability,
                    location,
                    scale,
                    configuration["L"],
                    student_df=STUDENT_DF,
                )
                retained_values[name].append(retained_mass.detach().cpu())
                for subset_index, subset in enumerate(subsets_by_name[name]):
                    law = _dense_joint_law(
                        member_prior,
                        member_indices,
                        member_radial,
                        location,
                        scale,
                        subset,
                        top_k=configuration["K"],
                        alpha=configuration["alpha"],
                    )
                    energy, mae = _score_joint_law(
                        *law,
                        target_device[start:stop],
                        batch_randomness,
                    )
                    subset_energy[name][subset_index, start:stop] = energy
                    subset_mae[name][subset_index, start:stop] = mae
            print(
                json.dumps(
                    {
                        "outer_rows_completed": stop,
                        "outer_rows_total": len(rss),
                        "configurations": list(CONFIGURATIONS),
                        "realizations_per_configuration": PANEL_MEMBER_COUNT,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

    group_count = len(group_keys)
    metrics = {}
    group_vectors: dict[str, dict[str, np.ndarray]] = {}
    realization_metrics = {}
    retained_mass_diagnostic = {}
    for name in CONFIGURATIONS:
        expected_energy_rows = subset_energy[name].mean(dim=0)
        expected_mae_rows = subset_mae[name].mean(dim=0)
        energy_record, energy_group = _metric_record(
            expected_energy_rows, group_ids, group_count
        )
        mae_record, mae_group = _metric_record(expected_mae_rows, group_ids, group_count)
        metrics[name] = {
            "energy_score_2d_m": energy_record,
            "exact_mixture_mean_mae_m": mae_record,
        }
        group_vectors[name] = {
            "energy_score_2d_m": energy_group,
            "exact_mixture_mean_mae_m": mae_group,
        }
        realization_metrics[name] = _subset_run_records(
            subset_energy[name],
            subset_mae[name],
            subsets_by_name[name],
            group_ids,
            group_count,
        )
        mass = torch.cat(retained_values[name]).double()
        retained_mass_diagnostic[name] = {
            "mean": float(mass.mean()),
            "q10": float(torch.quantile(mass, 0.10)),
            "median": float(torch.quantile(mass, 0.50)),
            "q90": float(torch.quantile(mass, 0.90)),
            "mean_reverse_kl_nats": float((-mass.clamp_min(1e-30).log()).mean()),
        }

    differences = {
        endpoint: group_vectors[SELECTED_NAME][endpoint]
        - group_vectors[HISTORICAL_NAME][endpoint]
        for endpoint in ("energy_score_2d_m", "exact_mixture_mean_mae_m")
    }
    inference = studentized_max_t_family(
        differences,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
        confidence=CONFIDENCE,
        orientation="selected_E16_minus_historical_E5",
    )
    endpoint_conclusions = {
        endpoint: {
            "simultaneous_upper_bound_below_zero": (
                inference["comparisons"][endpoint]["simultaneous_upper_bound_m"] < 0.0
            ),
            "model_selection_authority": False,
        }
        for endpoint in differences
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "comparison_status": "one_shot_registered_outer_confirmation",
        "selection_authority_for_v17_r2f": False,
        "may_modify_v17_r2f_model": False,
        "outer_refit_csv_sha256": _sha256_file(outer_refit_csv),
        "outer_refit_checkpoint_sha256": [
            _sha256_file(path) for path in checkpoint_paths
        ],
        "metrics": metrics,
        "coordinate_group_vectors": {
            name: {
                endpoint: values.tolist() for endpoint, values in endpoints.items()
            }
            for name, endpoints in group_vectors.items()
        },
        "realization_metrics": realization_metrics,
        "paired_familywise_inference": inference,
        "endpoint_conclusions": endpoint_conclusions,
        "retained_mass_diagnostic": retained_mass_diagnostic,
        "estimand_audit": {
            "losses_averaged_across_realizations": True,
            "predictive_distributions_averaged_across_realizations": False,
            "selected_members_per_predictor": CONFIGURATIONS[SELECTED_NAME]["E"],
            "historical_members_per_predictor": CONFIGURATIONS[HISTORICAL_NAME]["E"],
            "realizations_per_configuration": PANEL_MEMBER_COUNT,
            "favorable_subset_selected": False,
        },
        "outer_test_access_number": 1,
        "outer_test_rows": len(rss),
        "outer_test_coordinate_groups": group_count,
        "official_validation_predictions": 0,
        "official_validation_path_opened": False,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(destination, result)
    return result


def check_result(
    registry_path: str | Path = DEFAULT_REGISTRY,
    result_path: str | Path = DEFAULT_RESULT,
) -> dict:
    registry = load_registry(registry_path)
    source = Path(result_path).resolve()
    if source != Path(registry["allowed_outputs"]["outer_result"]).resolve():
        raise ValueError("R3 result path is outside the registry")
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("result_sha256") != _canonical_json_sha256(
        result, omit="result_sha256"
    ):
        raise ValueError("R3 result self-hash failed")
    checks = {
        "protocol": result.get("protocol_version") == PROTOCOL_VERSION,
        "registry": result.get("registry_sha256") == registry["registry_sha256"],
        "outer_access_once": result.get("outer_test_access_number") == 1,
        "outer_rows": result.get("outer_test_rows") == OUTER_TEST_ROWS,
        "coordinate_groups": result.get("outer_test_coordinate_groups")
        == OUTER_TEST_COORDINATES,
        "official_sealed": result.get("official_validation_predictions") == 0
        and result.get("official_validation_path_opened") is False,
        "no_selection_authority": result.get("selection_authority_for_v17_r2f")
        is False,
        "no_model_modification": result.get("may_modify_v17_r2f_model") is False,
        "both_configurations_reported": tuple(result.get("metrics", {}))
        == tuple(CONFIGURATIONS),
        "family_size_two": result.get("paired_familywise_inference", {}).get(
            "family_size"
        )
        == 2,
        "all_realizations_reported": all(
            len(result.get("realization_metrics", {}).get(name, ()))
            == PANEL_MEMBER_COUNT
            for name in CONFIGURATIONS
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"R3 result contract failed: {checks}")
    return {
        "result": str(source),
        "result_file_sha256": _sha256_file(source),
        "result_sha256": result["result_sha256"],
        "checks": checks,
        "metrics": result["metrics"],
        "paired_familywise_inference": result["paired_familywise_inference"],
        "endpoint_conclusions": result["endpoint_conclusions"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize-outer-refit")
    materialize.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train = subparsers.add_parser("train-member")
    train.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train.add_argument("--member-index", type=int, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=5)
    members = subparsers.add_parser("check-members")
    members.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    outer = subparsers.add_parser("evaluate-outer")
    outer.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    outer.add_argument("--device", default="cuda")
    result_check = subparsers.add_parser("check-result")
    result_check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    result_check.add_argument("--result", default=str(DEFAULT_RESULT))
    args = parser.parse_args(argv)
    if args.command == "materialize-outer-refit":
        result = materialize_outer_refit(args.registry)
    elif args.command == "train-member":
        result = train_member(
            member_index=args.member_index,
            registry_path=args.registry,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "check-members":
        result = check_members(args.registry)
    elif args.command == "evaluate-outer":
        result = evaluate_outer(registry_path=args.registry, device_name=args.device)
    else:
        result = check_result(args.registry, args.result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

