"""V17-G1: prospective, development-only geometry-support debiasing pilot.

Training is the ORIGINAL R0 refit, on SELECTION_REFIT (never OUTER_REFIT).
Epochs are fixed from prior R0 member_fit/early_stop decisions. R3 E16
scoring and balanced cyclic subsets are frozen. No new training loss/model.
Calibration and confirmation are distinct physical-coordinate sets.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_protocol as v17_protocol
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as r0
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f as r2f
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _rows_for_keys, _save_torch_new,
    _sha256_file, _sha256_indices, _sha256_keys, _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.projected_ensemble import ProjectedEnsemble
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r0_runtime_a2 import (
    _installed_adapters, _role_record,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA = ROOT.parent / "data" / "UJIndoorLoc" / "trainingData.csv"
HISTORICAL_REGISTRY = ROOT / "evidence/historical_reproduction/v17/registry_v17_reproduction.json"
EVIDENCE = ROOT / "evidence/v17_g1_geometry_support"
MODELS = ROOT / "models/v17_g1_geometry_support"
DESIGN = EVIDENCE / "DESIGN_V17_G1.md"
REGISTRY = EVIDENCE / "registry_v17_g1.json"
REFIT_CSV = EVIDENCE / "selection_refit_v17_g1.csv"
CALIBRATION = EVIDENCE / "calibration_v17_g1.json"
FREEZE = EVIDENCE / "freeze_v17_g1.json"
CONFIRMATION = EVIDENCE / "confirmation_v17_g1.json"


class G1Error(RuntimeError):
    pass


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, content: dict) -> dict:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(content)
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(path, result)
    return result


def _read_hashed(path: Path, hash_name: str) -> dict:
    content = _json(path)
    if content.get(hash_name) != _canonical_json_sha256(content, omit=hash_name):
        raise G1Error(f"self hash does not verify: {path}")
    return content


def member_path(index: int) -> Path:
    if not 1 <= int(index) <= g1.PANEL_SIZE:
        raise ValueError("member index outside 1..25")
    return MODELS / f"member_{index:02d}_seed{g1.SEEDS[index-1]}.pt"


def _roles(data: dict, historical: dict) -> tuple[dict[str, set], dict[str, np.ndarray]]:
    keys, _ = v17_protocol.construct_coordinate_roles(set(data["keys"]))
    rows = {role: _rows_for_keys(data["keys"], group) for role, group in keys.items()}
    for role, entry in historical["roles"].items():
        if _sha256_keys(keys[role]) != entry["coordinate_key_sha256"]:
            raise G1Error(f"historical V17 coordinate role changed: {role}")
        if _sha256_indices(rows[role]) != entry["row_index_sha256"]:
            raise G1Error(f"historical V17 row role changed: {role}")
    if len(keys["selection_refit"]) != g1.EXPECTED_SELECTION_ANCHORS:
        raise G1Error("selection_refit anchor count changed")
    if len(keys["development"]) != g1.EXPECTED_DEVELOPMENT_COORDINATES:
        raise G1Error("development coordinate count changed")
    return keys, rows


def preflight(device_name: str = "cuda") -> dict:
    for source in (TRAIN_DATA, HISTORICAL_REGISTRY, DESIGN):
        if not source.is_file():
            raise FileNotFoundError(source)
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise G1Error("CUDA was requested but unavailable")
    if _sha256_file(TRAIN_DATA) != g1.EXPECTED_TRAIN_SHA256:
        raise G1Error("registered raw data bytes changed")
    hist = _read_hashed(HISTORICAL_REGISTRY, "registry_sha256")
    if hist["training_data_sha256"] != g1.EXPECTED_TRAIN_SHA256:
        raise G1Error("historical V17 training input changed")
    if tuple(hist["member_seeds"]) != g1.SEEDS:
        raise G1Error("original V17 member panel differs")
    if tuple(r3p.MEMBER_SEEDS) != g1.SEEDS or r3p.PANEL_MEMBER_COUNT != 25:
        raise G1Error("historical R3 member panel mutated")
    if r3p.CONFIGURATIONS[r3p.SELECTED_NAME] != {
        "E": 16, "L": 192, "K": 192, "alpha": 1.0
    }:
        raise G1Error("R3 selected configuration mutated")
    if FROZEN_TRAINING["scan_feature_mode"] != "legacy":
        raise G1Error("R0 training has changed")
    return {
        "status": "ready", "protocol": g1.PROTOCOL,
        "train_sha256": g1.EXPECTED_TRAIN_SHA256,
        "historical_v17_registry_sha256": hist["registry_sha256"],
        "historical_v17_registry_file_sha256": _sha256_file(HISTORICAL_REGISTRY),
        "baseline": "unchanged_V17_R3_E16_with_25_balanced_subsets",
        "candidate": "inverse_geometry_support_atomic_anchor_prior",
        "no_outer_test_or_official_predictions": True,
        "historical_next_stage_authorized_remains": False,
    }


def prepare(device_name: str = "cuda") -> dict:
    status = preflight(device_name)
    historical = _read_hashed(HISTORICAL_REGISTRY, "registry_sha256")
    data = load_uji_v16(TRAIN_DATA)
    keys, rows = _roles(data, historical)
    cal, conf = g1.split_development_keys(keys["development"])
    if len(cal) + len(conf) != g1.EXPECTED_DEVELOPMENT_COORDINATES:
        raise G1Error("cal/confirm groups lost")
    config = {
        "protocol": g1.PROTOCOL,
        "hypothesis": "anchor_density_correction_reduces_coordinate_macro_risk",
        "registered_before_g1_prediction": True,
        "input_sha256": status["train_sha256"],
        "historical_v17_registry_sha256": historical["registry_sha256"],
        "historical_v17_registry_file_sha256": status["historical_v17_registry_file_sha256"],
        "design_sha256": _sha256_file(DESIGN),
        "source_sha256": {
            "g1_protocol": _sha256_file(Path(g1.__file__)),
            "g1_execution": _sha256_file(Path(__file__)),
            "r0_training": _sha256_file(Path(r0.__file__)),
            "r3_scoring": _sha256_file(Path(r3.__file__)),
        },
        "epochs": list(g1.EPOCHS), "seeds": list(g1.SEEDS),
        "epochs_from": "historical V17 member_fit/early_stop only, NOT outer outcome",
        "roles": {
            role: {"coordinate_key_sha256": _sha256_keys(keys[role]),
                   "row_index_sha256": _sha256_indices(rows[role]),
                   "coordinate_count": len(keys[role]), "rows": len(rows[role])}
            for role in ("member_fit", "early_stop", "selection_refit", "development")
        },
        "calibration": {
            "coordinate_key_sha256": _sha256_keys(cal), "groups": len(cal),
            "row_index_sha256": _sha256_indices(_rows_for_keys(data["keys"], cal))
        },
        "confirmation": {
            "coordinate_key_sha256": _sha256_keys(conf), "groups": len(conf),
            "row_index_sha256": _sha256_indices(_rows_for_keys(data["keys"], conf))
        },
        "gamma_grid": list(g1.GAMMAS), "radius_m": g1.RADIUS_M,
        "fold_seed": g1.FOLD_SEED, "placebo_seed": g1.PLACEBO_SEED,
        "bootstrap_seed": g1.BOOTSTRAP_SEED, "bootstrap_repetitions": g1.BOOTSTRAP_REPETITIONS,
        "selected_R3_configuration": copy.deepcopy(r3p.CONFIGURATIONS[r3p.SELECTED_NAME]),
        "allowed_outputs": {
            "selection_refit_csv": str(REFIT_CSV.resolve()),
            "models": [str(member_path(i).resolve()) for i in range(1,26)],
            "calibration": str(CALIBRATION.resolve()),
            "freeze": str(FREEZE.resolve()),
            "confirmation": str(CONFIRMATION.resolve()),
        },
        "outer_test_predictions_authorized": False,
        "official_validation_predictions_authorized": False,
        "external_predictions_authorized": False,
        "historical_gate_waiver": False,
    }
    config["registry_sha256"] = _canonical_json_sha256(config)
    if REGISTRY.exists():
        saved = _read_hashed(REGISTRY, "registry_sha256")
        if saved != config:
            raise G1Error("G1 registry/protocol/environment changed; do not overwrite")
    else:
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        _write_json_new(REGISTRY, config)
    if not REFIT_CSV.exists():
        r0._materialize_csv(TRAIN_DATA, REFIT_CSV, rows["selection_refit"])
    derived = load_uji_v16(REFIT_CSV)
    if set(derived["keys"]) != keys["selection_refit"] or len(derived["keys"]) != len(rows["selection_refit"]):
        raise G1Error("materialized selection refit differs from registration")
    return config


def train_member(index: int, device_name: str = "cuda") -> dict:
    registry = prepare(device_name)
    path = member_path(index)
    seed = g1.SEEDS[index - 1]
    epoch = g1.EPOCHS[index - 1]
    if path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not (checkpoint.get("registry_sha256") == registry["registry_sha256"]
                and checkpoint.get("member_index") == index
                and checkpoint.get("member_seed") == seed
                and checkpoint.get("epoch") == epoch
                and checkpoint.get("phase") == "g1_selection_refit"):
            raise G1Error(f"member {index} checkpoint mismatch")
        return {"member": index, "seed": seed, "epoch": epoch,
                "sha256": _sha256_file(path), "resumed": True}
    data = load_uji_v16(TRAIN_DATA)
    historical = _read_hashed(HISTORICAL_REGISTRY, "registry_sha256")
    keys, rows = _roles(data, historical)
    started = time.perf_counter()
    model, stats, origin, mapping, history = r0._fit_refit(
        data, rows["selection_refit"], seed=seed, epochs=epoch,
        device=torch.device(device_name), log_every=5,
    )
    if model.n_anchors != g1.EXPECTED_SELECTION_ANCHORS:
        raise G1Error("trained member anchor count changed")
    config = copy.deepcopy(FROZEN_TRAINING)
    config.update({
        "model_schema": "v17_g1_selection_refit_original_atomic_acrhap",
        "training_protocol_version": g1.PROTOCOL,
        "n_devices": len(mapping), "train_all": True,
        "split_strategy": "v17_registered_selection_refit",
        "validation_fraction": 0.0, "calibration_fraction": 0.0,
        "epochs": epoch, "patience": r0.PATIENCE,
        "lr": FROZEN_TRAINING["learning_rate"],
        "epoch_selection_source": "reused_historical_member_fit_early_stop",
    })
    payload = {
        "protocol_version": g1.PROTOCOL,
        "registry_sha256": registry["registry_sha256"],
        "phase": "g1_selection_refit", "member_index": index,
        "member_seed": seed,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config, "stats": stats,
        "coord_origin": np.asarray(origin, dtype=np.float32),
        "conformal_quantiles": {}, "epoch": epoch,
        "val_mae": 0.0, "seed": seed,
        "device_id_mapping": {str(k): v for k, v in mapping.items()},
        "anchor_coordinate_keys": tuple(sorted(keys["selection_refit"])),
        "refit_history": history,
        "selection_refit_csv_sha256": _sha256_file(REFIT_CSV),
        "training_seconds": time.perf_counter() - started,
        "split_contract": {
            "training_roles": ["selection_refit"],
            "development_features_consumed": False,
            "development_labels_consumed": False,
            "outer_test_features_consumed": False,
            "outer_test_labels_consumed": False,
            "official_features_consumed": False,
            "official_labels_consumed": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_torch_new(path, payload)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"member": index, "seed": seed, "epoch": epoch,
            "sha256": _sha256_file(path), "resumed": False,
            "training_seconds": payload["training_seconds"]}


def check_members() -> dict:
    if not REGISTRY.exists():
        raise G1Error("prepare is required")
    registry = _read_hashed(REGISTRY, "registry_sha256")
    records = []
    for i in range(1, 26):
        path = member_path(i)
        if not path.exists():
            return {"all_complete": False, "complete_members": i - 1, "missing": i}
        member = torch.load(path, map_location="cpu", weights_only=True)
        if not (member.get("phase") == "g1_selection_refit"
                and member.get("registry_sha256") == registry["registry_sha256"]
                and member.get("member_index") == i
                and member.get("member_seed") == g1.SEEDS[i-1]
                and member.get("epoch") == g1.EPOCHS[i-1]
                and len(member["anchor_coordinate_keys"]) == g1.EXPECTED_SELECTION_ANCHORS
                and member["split_contract"]["outer_test_labels_consumed"] is False):
            raise G1Error(f"invalid G1 member {i}")
        records.append({"member": i, "sha256": _sha256_file(path),
                        "epoch": member["epoch"]})
    return {"all_complete": True, "complete_members": len(records), "members": records}


def _evaluation_rows(role: str) -> tuple[dict, np.ndarray, list[tuple]]:
    if role not in ("calibration", "confirmation"):
        raise ValueError("G1 evaluator may only access development folds")
    registry = _read_hashed(REGISTRY, "registry_sha256")
    data = load_uji_v16(TRAIN_DATA)
    historical = _read_hashed(HISTORICAL_REGISTRY, "registry_sha256")
    keys, _ = _roles(data, historical)
    cal, conf = g1.split_development_keys(keys["development"])
    requested = cal if role == "calibration" else conf
    rows = _rows_for_keys(data["keys"], requested)
    if _sha256_keys(requested) != registry[role]["coordinate_key_sha256"]:
        raise G1Error(f"{role} key hash changed")
    if _sha256_indices(rows) != registry[role]["row_index_sha256"]:
        raise G1Error(f"{role} row hash changed")
    return data, rows, sorted(requested)


@torch.no_grad()
def _evaluate(role: str, variants: dict[str, tuple[float, bool]],
              device_name: str) -> dict:
    audit = check_members()
    if not audit["all_complete"]:
        raise G1Error("all 25 development-safe members must be complete")
    data, rows, registered_keys = _evaluation_rows(role)
    with _installed_adapters([_role_record(REFIT_CSV)]):
        ensemble = ProjectedEnsemble(
            checkpoints=[member_path(i) for i in range(1, 26)],
            anchor_data=REFIT_CSV, device=device_name, dataset="UJI",
        )
    device = torch.device(device_name)
    models = [model for model, _ in ensemble.checkpoints]
    for model in models:
        model.eval()
    reference = models[0]
    if len(models) != 25 or reference.n_anchors != g1.EXPECTED_SELECTION_ANCHORS:
        raise G1Error("loaded G1 panel is not 25 shared 595-anchor models")
    for model in models:
        if model.n_anchors != reference.n_anchors:
            raise G1Error("member anchor count mismatch")
        if not torch.allclose(model.anchor_coords, reference.anchor_coords, atol=1e-6):
            raise G1Error("member support geometry mismatch")
        if not torch.equal(model.anchor_buildings, reference.anchor_buildings):
            raise G1Error("member support building mismatch")
        if not torch.equal(model.anchor_floors, reference.anchor_floors):
            raise G1Error("member support floor mismatch")
    log_d = g1.support_log_density(
        reference.anchor_coords, reference.anchor_buildings, reference.anchor_floors
    )
    placebo = g1.placebo_log_density(
        log_d, reference.anchor_buildings, reference.anchor_floors
    )
    rss = data["rss"][rows].float().cpu()
    origin = np.asarray(ensemble.reference["coord_origin"], dtype=np.float64)
    target = torch.from_numpy(data["raw_xy"][rows] - origin[None, :]).double().to(device)
    group_ids, group_keys = r2b._coordinate_groups(
        [data["keys"][int(i)] for i in rows]
    )
    if group_keys != tuple(registered_keys):
        raise G1Error("group ordering/registered coordinates differ")
    subsets = r3p.balanced_member_subsets(16)
    if len(subsets) != 25:
        raise G1Error("R3 subset contract changed")
    randomness = r3.common_student_t_randomness(
        len(rss), draws=r3p.PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=r3p.STUDENT_DF, seed=g1.SCORING_SEED,
    )
    randomness = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in randomness.items()
    }
    energy = {key: torch.empty((25, len(rss)), dtype=torch.float64)
              for key in variants}
    mae = {key: torch.empty((25, len(rss)), dtype=torch.float64)
           for key in variants}
    count = reference.n_anchors
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0, len(rss), r3p.COMPONENT_BATCH_SIZE):
            stop = min(start + r3p.COMPONENT_BATCH_SIZE, len(rss))
            batch = rss[start:stop].to(device)
            pd = r2b._prototype_distance(reference, batch)
            logits, components, scales = [], [], []
            anchor_ids = torch.arange(count, device=device).unsqueeze(0).expand(len(batch), -1)
            for model in models:
                emb = model.encoder(batch)
                logits.append(model.retrieval_logits_from_embedding(emb))
                local, scale = model._components(emb, anchor_ids)
                components.append(local)
                scales.append(scale)
            p = r2f._atomic_probability_panel(
                torch.stack(logits), pd, r3p.BETA_NEURAL, r3p.BETA_PROTOTYPE
            )
            locations, sigma = torch.stack(components), torch.stack(scales)
            random_slice = {
                key: value[start:stop] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
                for key, value in randomness.items()
            }
            for name, (gamma, shuffled) in variants.items():
                prior = g1.corrected_probability(
                    p, placebo if shuffled else log_d, gamma
                )
                top_p, top_i, radial, _ = r3._member_top_l(
                    prior, locations, sigma, r3p.CONFIGURATIONS[r3p.SELECTED_NAME]["L"],
                    student_df=r3p.STUDENT_DF,
                )
                for j, subset in enumerate(subsets):
                    law = r3._dense_joint_law(
                        top_p, top_i, radial, locations, sigma, subset,
                        top_k=r3p.CONFIGURATIONS[r3p.SELECTED_NAME]["K"],
                        alpha=r3p.CONFIGURATIONS[r3p.SELECTED_NAME]["alpha"],
                    )
                    es, error = r3._score_joint_law(
                        *law, target[start:stop], random_slice
                    )
                    energy[name][j, start:stop] = es.detach().cpu()
                    mae[name][j, start:stop] = error.detach().cpu()
            print(json.dumps({"role": role, "completed": stop, "total": len(rss)}), flush=True)
    metrics, vectors, realization = {}, {}, {}
    for name in variants:
        mean_energy = energy[name].mean(dim=0)
        mean_mae = mae[name].mean(dim=0)
        er, ev = r3._metric_record(mean_energy, group_ids, len(group_keys))
        mr, mv = r3._metric_record(mean_mae, group_ids, len(group_keys))
        metrics[name] = {
            "energy_score_2d_m": float(er["coordinate_macro_m"]),
            "mae_2d_m": float(mr["coordinate_macro_m"]),
            "energy_row_mean_m": float(er["row_mean_m"]),
            "mae_row_mean_m": float(mr["row_mean_m"]),
        }
        vectors[name] = {
            "energy_score_2d_m": ev.tolist(),
            "exact_mixture_mean_mae_m": mv.tolist(),
        }
        realization[name] = r3._subset_run_records(
            energy[name], mae[name], subsets, group_ids, len(group_keys)
        )
    return {
        "protocol": g1.PROTOCOL, "registry_sha256": _read_hashed(REGISTRY, "registry_sha256")["registry_sha256"],
        "role": role, "rows": len(rows), "groups": len(group_keys),
        "group_key_sha256": _sha256_keys(set(group_keys)),
        "input_member_sha256": audit["members"],
        "estimand": "average_loss_over_25_fixed_balanced_E16_subsets",
        "do_not_interpret_as_one_fixed_16_member_deployment": True,
        "variants": {name: {"gamma": gamma, "permuted_placebo": bool(placebo_flag)}
                     for name, (gamma, placebo_flag) in variants.items()},
        "metrics": metrics, "coordinate_group_vectors": vectors,
        "realization_metrics": realization,
        "official_validation_predictions": 0,
        "outer_test_predictions": 0,
        "external_predictions": 0,
    }


def calibrate(device_name: str = "cuda") -> dict:
    prepare(device_name)
    if CALIBRATION.exists() or FREEZE.exists() or CONFIRMATION.exists():
        raise G1Error("calibration already exists; immutable one-shot stage")
    variants = {g1.gamma_name(gamma): (gamma, False) for gamma in g1.GAMMAS}
    result = _evaluate("calibration", variants, device_name)
    result = _write_once(CALIBRATION, result)
    chosen = g1.choose_gamma(result["metrics"])
    freeze = {
        "protocol": g1.PROTOCOL, "registry_sha256": result["registry_sha256"],
        "calibration_result_sha256": result["result_sha256"],
        "calibration_file_sha256": _sha256_file(CALIBRATION),
        "chosen_gamma": chosen, "selection": "min_mae_with_ES_guard",
        "confirmation_metrics_accessed": False,
        "outer_test_predictions": 0, "official_validation_predictions": 0,
    }
    freeze = _write_once(FREEZE, freeze)
    return {"chosen_gamma": chosen, "calibration": str(CALIBRATION),
            "freeze": str(FREEZE), "freeze_sha256": freeze["result_sha256"]}


def confirm(device_name: str = "cuda") -> dict:
    prepare(device_name)
    if CONFIRMATION.exists():
        raise G1Error("confirmation already frozen; cannot reuse as tuner")
    if not CALIBRATION.exists() or not FREEZE.exists():
        raise G1Error("calibration and frozen gamma required")
    cal = _read_hashed(CALIBRATION, "result_sha256")
    freeze = _read_hashed(FREEZE, "result_sha256")
    if (freeze["calibration_result_sha256"] != cal["result_sha256"]
            or freeze["calibration_file_sha256"] != _sha256_file(CALIBRATION)
            or freeze["registry_sha256"] != cal["registry_sha256"]):
        raise G1Error("calibration/freeze chain invalid")
    gamma = float(freeze["chosen_gamma"])
    if gamma not in g1.GAMMAS:
        raise G1Error("frozen gamma is not registered")
    variants = {"baseline": (0.0, False),
                "selected": (gamma, False),
                "placebo": (gamma, True)}
    result = _evaluate("confirmation", variants, device_name)
    vectors = result["coordinate_group_vectors"]
    endpoints = ("exact_mixture_mean_mae_m", "energy_score_2d_m")
    differences = {
        endpoint: np.asarray(vectors["selected"][endpoint], dtype=np.float64)
        - np.asarray(vectors["baseline"][endpoint], dtype=np.float64)
        for endpoint in endpoints
    }
    inference = r3.studentized_max_t_family(
        differences, repetitions=g1.BOOTSTRAP_REPETITIONS,
        confidence=g1.CONFIDENCE, seed=g1.BOOTSTRAP_SEED,
        orientation="geometry_correction_minus_frozen_R3_baseline",
    )
    delta_mae = float(np.mean(differences["exact_mixture_mean_mae_m"]))
    delta_es = float(np.mean(differences["energy_score_2d_m"]))
    placebo_delta = float(np.mean(
        np.asarray(vectors["placebo"]["exact_mixture_mean_mae_m"])
        - np.asarray(vectors["baseline"]["exact_mixture_mean_mae_m"])
    ))
    gate = g1.confirm_gate(
        delta_mae,
        float(inference["comparisons"]["exact_mixture_mean_mae_m"]["simultaneous_upper_bound_m"]),
        delta_es,
        float(inference["comparisons"]["energy_score_2d_m"]["simultaneous_upper_bound_m"]),
        placebo_delta,
    )
    result.update({
        "chosen_gamma": gamma,
        "calibration_file_sha256": _sha256_file(CALIBRATION),
        "freeze_file_sha256": _sha256_file(FREEZE),
        "paired_familywise_inference": inference,
        "delta_coordinate_macro_mae_m": delta_mae,
        "delta_coordinate_macro_energy_m": delta_es,
        "placebo_delta_mae_m": placebo_delta,
        "gate": gate,
        "decision": "GO_TO_EXTERNAL_VALIDATION_DESIGN" if gate["passed"] else "STOP_GEOMETRY_DENSITY_MECHANISM",
        "historical_next_stage_authorized": False,
        "warning": "Development-only confirmation; this is NOT untouched outer or external validation.",
    })
    return _write_once(CONFIRMATION, result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "prepare", "train-member", "check-members",
                 "calibrate", "confirm", "run-all"):
        sub = commands.add_parser(name)
        if name in {"preflight","prepare","train-member","calibrate","confirm","run-all"}:
            sub.add_argument("--device", default="cuda")
        if name == "train-member":
            sub.add_argument("--member-index", required=True, type=int)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        output = preflight(args.device)
    elif args.command == "prepare":
        output = prepare(args.device)
    elif args.command == "train-member":
        output = train_member(args.member_index, args.device)
    elif args.command == "check-members":
        output = check_members()
    elif args.command == "calibrate":
        output = calibrate(args.device)
    elif args.command == "confirm":
        output = confirm(args.device)
    else:
        prepare(args.device)
        for index in range(1, 26):
            print(json.dumps(train_member(index, args.device)), flush=True)
        output = {"members": check_members(), "calibration": calibrate(args.device)}
        output["confirmation"] = confirm(args.device)
    print(json.dumps(output, indent=2, allow_nan=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
