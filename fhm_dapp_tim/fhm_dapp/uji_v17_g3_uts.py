"""G3: UTS externally held-out test of V17 with fixed G1 density correction.

Never load UTS_test.csv until 25 *train-only* members and frozen manifest
are verified AND an irreversible test invocation lock has been written.
The test data is user-attested as not previously scored/observed; schema
adaptation is disclosed.  No UJI outer/development reuse or new tuning.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as r0
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f as r2f
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp import v18_4_datasets as ds
from fhm_dapp_tim.fhm_dapp.anchor import load_anchor_checkpoint, location_keys
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _save_torch_new, _sha256_file, _write_json_new
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA_TRAIN = REPO / "data/UTSIndoorLoc/UTS_training.csv"
# This path MUST NEVER be read, stat'ed, or hashed in preflight/train/freeze.
DATA_TEST = REPO / "data/UTSIndoorLoc/UTS_test.csv"
EVIDENCE = ROOT / "evidence/v17_g3_uts_frozen_transfer"
MODELS = ROOT / "models/v17_g3_uts_frozen_transfer"
DESIGN = EVIDENCE / "DESIGN_V17_G3_UTS.md"
MANIFEST = EVIDENCE / "manifest_v17_g3_uts.json"
FREEZE = EVIDENCE / "freeze_v17_g3_uts.json"
LOCK = EVIDENCE / "external_test_invocation.lock"
RESULT = EVIDENCE / "external_result_v17_g3_uts.json"
REPORT = EVIDENCE / "RUN_REPORT_V17_G3_UTS.md"
G1_REGISTRY = ROOT / "evidence/v17_g1_geometry_support/registry_v17_g1.json"
G1_FREEZE = ROOT / "evidence/v17_g1_geometry_support/freeze_v17_g1.json"
G1_CONFIRM = ROOT / "evidence/v17_g1_geometry_support/confirmation_v17_g1.json"
G2_RESULT = ROOT / "evidence/v17_g2_mechanism/G2_MECHANISM_AUDIT.json"

PROTOCOL = "v17_g3_uts_blind_external_fixed_geometry_transfer_20261008"
PANEL_SIZE = 25
SEEDS = g1.SEEDS
EPOCHS = g1.EPOCHS
GAMMA = 0.70
RADIUS_M = 10.0
PLACEBO_SEED = g1.PLACEBO_SEED
SCORING_SEED = g1.SCORING_SEED
BOOTSTRAP_SEED = g1.BOOTSTRAP_SEED
BOOTSTRAP_REPETITIONS = g1.BOOTSTRAP_REPETITIONS
CONFIDENCE = 0.95
DISCLOSURE = (
    "User attests UTS provided test labels/outcomes were never observed "
    "for method development; earlier schema/interface adapter checks are "
    "disclosed separately. This is one-shot, performance-blind-to-freeze "
    "provided-test evaluation, not proof of wholly unopened file bytes. "
    "UJI G1 historical STOP remains unchanged."
)


class G3Error(RuntimeError):
    pass


def _load_json(path: Path, self_hash: str) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get(self_hash) != _canonical_json_sha256(obj, omit=self_hash):
        raise G3Error(f"self-hash failed: {path}")
    return obj


def _write_once(path: Path, obj: dict, self_hash: str = "result_sha256") -> dict:
    if path.exists():
        raise FileExistsError(path)
    item = dict(obj)
    item[self_hash] = _canonical_json_sha256(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_new(path, item)
    return item


def member_path(index: int) -> Path:
    if not 1 <= int(index) <= PANEL_SIZE:
        raise ValueError("member index must be 1..25")
    return MODELS / f"member_{index:02d}_seed{SEEDS[index - 1]}.pt"


def _frozen_source_hashes() -> dict[str, str]:
    """Must never inspect provided UTS test or UJI official/outer labels."""
    sources = {
        "training_csv": DATA_TRAIN, "design": DESIGN,
        "g1_registry": G1_REGISTRY, "g1_freeze": G1_FREEZE,
        "g1_confirmation": G1_CONFIRM, "g2_result": G2_RESULT,
        "g3_source": Path(__file__),
        "g1_geometry_source": Path(g1.__file__),
        "v17_r0_source": Path(r0.__file__),
        "v17_r3_source": Path(r3.__file__),
        "v17_r3_protocol": Path(r3p.__file__),
        "v17_r2b_source": Path(r2b.__file__),
        "v17_r2f_source": Path(r2f.__file__),
        "v18_4_dataset_adapter": Path(ds.__file__),
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    reg = _load_json(G1_REGISTRY, "registry_sha256")
    gf = _load_json(G1_FREEZE, "result_sha256")
    conf = _load_json(G1_CONFIRM, "result_sha256")
    dg = _load_json(G2_RESULT, "result_sha256")
    if (reg.get("historical_gate_waiver") is not False
        or reg.get("outer_test_predictions_authorized") is not False
        or gf["registry_sha256"] != reg["registry_sha256"]
        or gf["chosen_gamma"] != GAMMA
        or conf["chosen_gamma"] != GAMMA
        or conf["decision"] != "STOP_GEOMETRY_DENSITY_MECHANISM"
        or conf["gate"]["passed"] is not False
        or dg["g1_formal_outcome_unchanged"]["status"] != conf["decision"]
        or reg["radius_m"] != RADIUS_M):
        raise G3Error("G1/G2 frozen source integrity or scientific outcome differs")
    return {key: _sha256_file(path) for key, path in sources.items()}


def training_data() -> tuple[dict, object]:
    """Train-only legacy V17 tensors from already-vetted canonical UTS loader."""
    canonical, reference = ds.load_uts("train")
    canonical.validate()
    return _as_r0_input(canonical, require_training=True), reference


def _as_r0_input(canonical, *, require_training: bool) -> dict:
    if canonical.dataset != "UTSIndoorLoc":
        raise G3Error("only UTS supported")
    if require_training and canonical.split != "train":
        raise G3Error("cannot use test in train-only stage")
    rss = torch.as_tensor(canonical.model_rss(), dtype=torch.float32).cpu()
    xy = np.asarray(canonical.coords_m, dtype=np.float64)
    b = np.asarray(canonical.building, dtype=np.int64)
    f = np.asarray(canonical.floor, dtype=np.int64)
    device = np.asarray(canonical.device, dtype=np.int64)
    if (len(rss) == 0 or not np.isfinite(xy).all()
        or not torch.isfinite(rss).all()):
        raise G3Error("invalid canonical RSS/geometry")
    if require_training and (len(rss) < 100 or b.min() < 0 or f.min() < 0):
        raise G3Error("training rows or building/floor labels invalid")
    keys = location_keys(xy, b, f)
    return {
        "rss": rss, "raw_xy": xy, "building": b, "floor": f,
        "phone": device, "keys": keys,
        "stats": {
            "n_aps": int(rss.shape[1]),
            "n_buildings": int(b.max(initial=-1)) + 1,
            "n_floors": int(f.max(initial=-1)) + 1,
            "adapted_from": "v18_4_datasets canonical UTS train-only adapter",
        },
    }


def _training_geometry(data: dict) -> dict:
    """Fail if R0 center+float32 rounding collapses distinct physical sites."""
    orig = data["keys"]
    xy = data["raw_xy"]
    origin = xy.mean(axis=0).astype(np.float32)
    centered = (xy - origin[None]).astype(np.float32)
    embedded = location_keys(centered, data["building"], data["floor"])
    forwards, reverse = {}, {}
    for a, b in zip(orig, embedded):
        if a in forwards and forwards[a] != b:
            raise G3Error("one UTS physical site maps to multiple R0 anchors")
        if b in reverse and reverse[b] != a:
            raise G3Error("two UTS physical sites collapse to one R0 anchor")
        forwards[a], reverse[b] = b, a
    count = len(forwards)
    if count < 192:
        raise G3Error("less than 192 training anchors: frozen R3 L192/K192 incompatible")
    by_stratum = {}
    for key in forwards:
        by_stratum.setdefault((key[0], key[1]), 0)
        by_stratum[(key[0], key[1])] += 1
    # Train/test coordinates can overlap in provided benchmarks: that's
    # NOT row leakage; training/test scans must be disjoint files.
    return {
        "rows": len(orig), "anchor_count": count,
        "rss_aps": data["rss"].shape[1],
        "building_count": data["stats"]["n_buildings"],
        "floor_count": data["stats"]["n_floors"],
        "stratum_anchor_counts": {
            f"B{k[0]}_F{k[1]}": value for k, value in sorted(by_stratum.items())
        },
        "origin_mean_xy": origin.astype(float).tolist(),
    }


def preflight() -> dict:
    source = _frozen_source_hashes()
    data, ref = training_data()
    training = _training_geometry(data)
    legacy_visible_minus_one = int((data["rss"] == -1.0).sum())
    # Extract visible-RSS collision statistics using TRAINING data only.
    canonical, _ = ds.load_uts("train")
    visible = np.isfinite(canonical.rss_dbm)
    observed_clipped = int(np.count_nonzero(
        visible & (canonical.rss_dbm <= ds.RSS_CLIP_MIN_DBM)
    ))
    return {
        "status": "TRAIN_ONLY_READY", "protocol": PROTOCOL, "disclosure": DISCLOSURE,
        "source_sha256": source, "train_geometry": training,
        "training_reference_aps": len(ref.ap_ids),
        "legacy_minus_one_values_include_unobserved_and_clipped": True,
        "legacy_minus_one_cell_count": legacy_visible_minus_one,
        "observed_RSS_clipped_to_minus_one_count": observed_clipped,
        "provided_test_accessed": False, "gpu_required_for_fit": True,
        "frozen_gamma": GAMMA, "radius_m": RADIUS_M,
        "member_seeds": list(SEEDS), "member_epochs": list(EPOCHS),
    }


def prepare() -> dict:
    state = preflight()
    config = {
        "protocol": PROTOCOL, "disclosure": DISCLOSURE,
        "source_sha256": state["source_sha256"],
        "train_geometry": state["train_geometry"],
        "frozen_gamma": GAMMA, "radius_m": RADIUS_M,
        "placebo_seed": PLACEBO_SEED, "scoring_seed": SCORING_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "member_seeds": list(SEEDS), "member_epochs": list(EPOCHS),
        "panel_members": PANEL_SIZE,
        "balanced_E": 16,
        "fixed_R3_config": dict(r3p.CONFIGURATIONS[r3p.SELECTED_NAME]),
        "fixed_V17_training": copy.deepcopy(FROZEN_TRAINING),
        "training_protocol": "all provided UTS_training rows, no development tuning",
        "provided_test_use": "only after freeze and irreversible lock",
        "provided_test_outcomes_blind_prior_to_freeze": "user-attested",
        "provided_test_file_sha256_before_freeze": None,
        "allowed_model_outputs": [str(member_path(i).relative_to(REPO)) for i in range(1, 26)],
    }
    config["manifest_sha256"] = _canonical_json_sha256(config)
    if MANIFEST.exists():
        saved = _load_json(MANIFEST, "manifest_sha256")
        if saved != config:
            raise G3Error("G3 manifest/source changed; do not continue")
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        _write_json_new(MANIFEST, config)
    return config


def _check_member(i: int, manifest: dict) -> dict:
    path = member_path(i)
    if not path.exists():
        raise FileNotFoundError(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("protocol_version") != PROTOCOL
        or saved.get("manifest_sha256") != manifest["manifest_sha256"]
        or saved.get("member_index") != i
        or saved.get("member_seed") != SEEDS[i - 1]
        or saved.get("epoch") != EPOCHS[i - 1]
        or saved.get("split_contract", {}).get("test_rows_accessed") is not False
        or len(saved.get("anchor_coordinate_keys", [])) !=
           manifest["train_geometry"]["anchor_count"]):
        raise G3Error(f"member {i} is not a frozen G3 training-only member")
    return {
        "member": i, "seed": saved["member_seed"],
        "epoch": saved["epoch"], "anchors": len(saved["anchor_coordinate_keys"]),
        "sha256": _sha256_file(path),
    }


def train_member(index: int, device_name: str = "cuda") -> dict:
    manifest = prepare()
    if FREEZE.exists() or LOCK.exists() or RESULT.exists():
        raise G3Error("no training permitted after freeze/test lock")
    path = member_path(index)
    if path.exists():
        return dict(_check_member(index, manifest), resumed=True)
    data, _ = training_data()
    train_idx = np.arange(len(data["keys"]), dtype=np.int64)
    started = time.perf_counter()
    model, stats, origin, mapping, history = r0._fit_refit(
        data, train_idx, seed=SEEDS[index - 1], epochs=EPOCHS[index - 1],
        device=torch.device(device_name), log_every=5,
    )
    anchor_keys = tuple(location_keys(
        model.anchor_coords.detach().cpu().numpy(),
        model.anchor_buildings.detach().cpu().numpy(),
        model.anchor_floors.detach().cpu().numpy(),
    ))
    if (len(anchor_keys) != manifest["train_geometry"]["anchor_count"]
        or len(set(anchor_keys)) != len(anchor_keys)):
        raise G3Error("V17 R0 learned anchor count differs from UTS train geometry")
    config = copy.deepcopy(FROZEN_TRAINING)
    config.update({
        "model_schema": "v17_g3_uts_original_atomic_acrhap",
        "training_protocol_version": PROTOCOL,
        "n_devices": len(mapping), "train_all": True,
        "split_strategy": "g3_full_uts_training",
        "validation_fraction": 0.0, "calibration_fraction": 0.0,
        "epochs": int(EPOCHS[index - 1]),
        "lr": FROZEN_TRAINING["learning_rate"],
    })
    payload = {
        "protocol_version": PROTOCOL, "manifest_sha256": manifest["manifest_sha256"],
        "phase": "g3_uts_train_only", "member_index": index,
        "member_seed": SEEDS[index - 1], "seed": SEEDS[index - 1],
        "epoch": EPOCHS[index - 1], "val_mae": 0.0,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "stats": stats, "coord_origin": np.asarray(origin, dtype=np.float32),
        "config": config, "conformal_quantiles": {},
        "device_id_mapping": {str(k): v for k, v in mapping.items()},
        "anchor_coordinate_keys": anchor_keys, "refit_history": history,
        "training_seconds": time.perf_counter() - started,
        "split_contract": {
            "training_source": "UTS_training.csv",
            "training_rows": int(len(train_idx)),
            "test_rows_accessed": False, "test_labels_consumed": False,
            "test_metrics_consumed": False, "training_only_epochs": list(EPOCHS),
            "outer_uji_consumed": False, "g1_development_consumed": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_torch_new(path, payload)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"member": index, "sha256": _sha256_file(path),
            "seed": SEEDS[index - 1], "epoch": EPOCHS[index - 1],
            "seconds": payload["training_seconds"], "resumed": False}


def check_members() -> dict:
    manifest = prepare()
    complete = []
    for i in range(1, 26):
        if not member_path(i).exists():
            return {"all_complete": False, "complete": len(complete), "first_missing": i}
        complete.append(_check_member(i, manifest))
    return {"all_complete": True, "complete": len(complete),
            "members": complete, "manifest_sha256": manifest["manifest_sha256"]}


def freeze() -> dict:
    manifest = prepare()
    if FREEZE.exists():
        existing = _load_json(FREEZE, "result_sha256")
        if existing["manifest_file_sha256"] != _sha256_file(MANIFEST):
            raise G3Error("saved freeze manifest no longer matches")
        if existing["members"] != check_members()["members"]:
            raise G3Error("checkpoint bytes differ from frozen hashes")
        return existing
    if LOCK.exists() or RESULT.exists():
        raise G3Error("test access begun without freeze")
    members = check_members()
    if not members["all_complete"]:
        raise G3Error("25 complete members required before freeze")
    content = {
        "protocol": PROTOCOL,
        "status": "FULL_TRAINING_MODEL_FROZEN_BEFORE_TEST",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": _sha256_file(MANIFEST),
        "members": members["members"],
        "gamma": GAMMA, "radius_m": RADIUS_M,
        "test_performance_seen": False, "test_file_accessed": False,
        "disclosure": DISCLOSURE,
    }
    return _write_once(FREEZE, content)


@torch.no_grad()
def _evaluate_test_once(device_name: str) -> dict:
    """ONLY call after external test invocation lock is already durable."""
    if not LOCK.exists() or RESULT.exists():
        raise G3Error("no locked one-shot evaluation available")
    locked = _load_json(LOCK, "result_sha256")
    frozen = _load_json(FREEZE, "result_sha256")
    if locked["freeze_file_sha256"] != _sha256_file(FREEZE):
        raise G3Error("test invocation lock/freeze differ")
    # FIRST access to provided UTS test rows: below this line only.
    train, reference = ds.load_uts("train")
    test, _ = ds.load_uts("test", reference=reference)
    data = _as_r0_input(test, require_training=False)
    if len(data["keys"]) < 2:
        raise G3Error("provided test has too few scans")
    if data["rss"].shape[1] != train.rss_dbm.shape[1]:
        raise G3Error("train/test AP dictionary width mismatch")
    if test.metadata.get("unseen_buildings") or test.metadata.get("unseen_floors"):
        raise G3Error("UTS official test contains unseen building/floor categories")
    if not torch.isfinite(data["rss"]).all():
        raise G3Error("test input model RSS nonfinite")
    models = []
    for i in range(1, 26):
        model, member = load_anchor_checkpoint(member_path(i), torch.device(device_name))
        if (member["manifest_sha256"] != frozen["manifest_sha256"]
            or model.n_anchors != frozen["members"][i-1]["anchors"]):
            raise G3Error("member data changed post-freeze")
        model.eval()
        models.append(model)
    ref_model = models[0]
    ref_xy = ref_model.anchor_coords
    ref_b, ref_f = ref_model.anchor_buildings, ref_model.anchor_floors
    for model in models:
        if not torch.allclose(model.anchor_coords, ref_xy, atol=1e-6):
            raise G3Error("member train anchors mismatch")
        if not torch.equal(model.anchor_buildings, ref_b):
            raise G3Error("member building support mismatch")
        if not torch.equal(model.anchor_floors, ref_f):
            raise G3Error("member floor support mismatch")
    log_density = g1.support_log_density(ref_xy, ref_b, ref_f, radius_m=RADIUS_M)
    placebo = g1.placebo_log_density(log_density, ref_b, ref_f, seed=PLACEBO_SEED)
    variants = {"baseline": (0.0, False),
                "g1_gamma_0p7": (GAMMA, False),
                "permuted_density": (GAMMA, True)}
    device = torch.device(device_name)
    rss = data["rss"].float()
    origin = np.asarray(
        torch.load(member_path(1), map_location="cpu", weights_only=True)["coord_origin"],
        dtype=np.float64
    )
    target = torch.from_numpy(data["raw_xy"] - origin[None, :]).double().to(device)
    groups, keys = r2b._coordinate_groups(data["keys"])
    subsets = r3p.balanced_member_subsets(16)
    randomness = r3.common_student_t_randomness(
        len(rss), draws=r3p.PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=r3p.STUDENT_DF, seed=SCORING_SEED
    )
    randomness = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in randomness.items()
    }
    mae = {key: torch.empty((25, len(rss)), dtype=torch.float64) for key in variants}
    es = {key: torch.empty((25, len(rss)), dtype=torch.float64) for key in variants}
    n_anchors = ref_model.n_anchors
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0, len(rss), r3p.COMPONENT_BATCH_SIZE):
            stop = min(start+r3p.COMPONENT_BATCH_SIZE, len(rss))
            batch = rss[start:stop].to(device)
            pd = r2b._prototype_distance(ref_model, batch)
            logits, locations, scales = [], [], []
            ids = torch.arange(n_anchors, device=device)[None].expand(len(batch), -1)
            for model in models:
                embedding = model.encoder(batch)
                logits.append(model.retrieval_logits_from_embedding(embedding))
                loc, scale = model._components(embedding, ids)
                locations.append(loc)
                scales.append(scale)
            prior = r2f._atomic_probability_panel(
                torch.stack(logits), pd, r3p.BETA_NEURAL, r3p.BETA_PROTOTYPE
            )
            locpanel = torch.stack(locations)
            scalepanel = torch.stack(scales)
            random_slice = {
                key: value[start:stop] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
                for key, value in randomness.items()
            }
            for name, (gamma, shuffled) in variants.items():
                p = g1.corrected_probability(
                    prior, placebo if shuffled else log_density, gamma
                )
                top_p, top_ids, radial, _ = r3._member_top_l(
                    p, locpanel, scalepanel,
                    r3p.CONFIGURATIONS[r3p.SELECTED_NAME]["L"],
                    student_df=r3p.STUDENT_DF
                )
                for j, subset in enumerate(subsets):
                    law = r3._dense_joint_law(
                        top_p, top_ids, radial, locpanel, scalepanel, subset,
                        top_k=r3p.CONFIGURATIONS[r3p.SELECTED_NAME]["K"],
                        alpha=r3p.CONFIGURATIONS[r3p.SELECTED_NAME]["alpha"],
                    )
                    energy, error = r3._score_joint_law(
                        *law, target[start:stop], random_slice
                    )
                    es[name][j, start:stop] = energy.detach().cpu()
                    mae[name][j, start:stop] = error.detach().cpu()
            print(json.dumps({"scored_rows": stop, "total_rows": len(rss)}), flush=True)
    metrics, vectors, realizations = {}, {}, {}
    for name in variants:
        er, ev = r3._metric_record(es[name].mean(dim=0), groups, len(keys))
        mr, mv = r3._metric_record(mae[name].mean(dim=0), groups, len(keys))
        metrics[name] = {
            "mae_2d_m": float(mr["coordinate_macro_m"]),
            "energy_score_2d_m": float(er["coordinate_macro_m"]),
            "mae_row_mean_m": float(mr["row_mean_m"]),
            "energy_row_mean_m": float(er["row_mean_m"]),
        }
        vectors[name] = {
            "exact_mixture_mean_mae_m": mv.tolist(),
            "energy_score_2d_m": ev.tolist(),
        }
        realizations[name] = r3._subset_run_records(
            es[name], mae[name], subsets, groups, len(keys)
        )
    differences = {
        name: np.asarray(vectors["g1_gamma_0p7"][name])
              - np.asarray(vectors["baseline"][name])
        for name in ("exact_mixture_mean_mae_m", "energy_score_2d_m")
    }
    inference = r3.studentized_max_t_family(
        differences, repetitions=BOOTSTRAP_REPETITIONS,
        confidence=CONFIDENCE, seed=BOOTSTRAP_SEED,
        orientation="fixed_G1_minus_original_V17_on_provided_UTS_test"
    )
    dmae = float(differences["exact_mixture_mean_mae_m"].mean())
    des = float(differences["energy_score_2d_m"].mean())
    placebo_mae = float(metrics["permuted_density"]["mae_2d_m"])
    baseline_mae = float(metrics["baseline"]["mae_2d_m"])
    gate = g1.confirm_gate(
        dmae,
        inference["comparisons"]["exact_mixture_mean_mae_m"]["simultaneous_upper_bound_m"],
        des,
        inference["comparisons"]["energy_score_2d_m"]["simultaneous_upper_bound_m"],
        placebo_mae - baseline_mae,
    )
    # Provided test coordinate density is post-outcome descriptive only. The
    # online method applies density to the TRAINING anchors, never query truth.
    return {
        "protocol": PROTOCOL, "status": "ONE_SHOT_EXTERNAL_TEST_COMPLETE",
        "disclosure": DISCLOSURE,
        "provided_test_data_sha256": _sha256_file(DATA_TEST),
        "test_rows": len(rss), "test_coordinate_groups": len(keys),
        "test_coordinate_keys": [list(key) for key in keys],
        "train_test_coordinate_overlap_count": len(set(train_key for train_key in
            location_keys(train.coords_m, train.building, train.floor)) & set(keys)),
        "metrics": metrics, "coordinate_group_vectors": vectors,
        "realization_metrics": realizations,
        "paired_familywise_inference": inference,
        "delta_coordinate_macro_mae_m": dmae,
        "delta_coordinate_macro_energy_m": des,
        "placebo_minus_baseline_mae_m": placebo_mae-baseline_mae,
        "fixed_transfer_gate": gate,
        "decision": "TRANSFER_SUPPORTED_ON_UTS" if gate["passed"]
                    else "TRANSFER_NOT_ESTABLISHED_ON_UTS",
        "historical_G1_decision": "STOP_GEOMETRY_DENSITY_MECHANISM",
        "test_access_before_freeze": False,
        "tuned_on_external_test": False, "trained_model_on_external_test": False,
        "model_count": PANEL_SIZE, "subset_count": len(subsets),
    }


def evaluate_test(device_name: str = "cuda") -> dict:
    if LOCK.exists() or RESULT.exists() or REPORT.exists():
        raise G3Error("external test already attempted: irreversible lock")
    manifest = prepare()
    frozen = freeze()
    if frozen["manifest_sha256"] != manifest["manifest_sha256"]:
        raise G3Error("freeze/manifest changed")
    # Durable, immutable lock FIRST, before the test loader is called.
    lock = _write_once(LOCK, {
        "protocol": PROTOCOL,
        "status": "PROVIDED_TEST_EVALUATION_STARTS_AFTER_FREEZE",
        "manifest_file_sha256": _sha256_file(MANIFEST),
        "freeze_file_sha256": _sha256_file(FREEZE),
        "freeze_sha256": frozen["result_sha256"],
        "member_sha256": [m["sha256"] for m in frozen["members"]],
        "test_opened_prior_to_lock": False,
        "warning": "This lock remains if evaluation fails; no silent rerun.",
    })
    result = _evaluate_test_once(device_name)
    result.update({
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": _sha256_file(MANIFEST),
        "freeze_file_sha256": _sha256_file(FREEZE),
        "lock_file_sha256": _sha256_file(LOCK),
        "lock_result_sha256": lock["result_sha256"],
        "checkpoint_sha256": [m["sha256"] for m in frozen["members"]],
        "source_sha256": manifest["source_sha256"],
    })
    result = _write_once(RESULT, result)
    REPORT.write_text(make_report(result), encoding="utf-8", newline="\n")
    return {
        "status": result["status"], "decision": result["decision"],
        "result_sha256": result["result_sha256"],
        "metrics": result["metrics"],
        "fixed_transfer_gate": result["fixed_transfer_gate"],
    }


def make_report(obj: dict) -> str:
    lines = [
        "# V17-G3: UTS one-shot fixed-G1 provided-test transfer",
        "",
        "**Training input:** UTS_training.csv only; 25 original V17 R0 members.",
        "**Evaluation:** one frozen provided UTS_test.csv attempt after immutable lock.",
        "**Provenance:** test outcomes are user-attested unseen; adapter/schema was known.",
        "**Not:** a claim that every file byte was never touched.",
        "", obj["disclosure"], "",
        f"Protocol: {obj['protocol']}",
        f"Test scans: {obj['test_rows']}; test coordinate groups: {obj['test_coordinate_groups']}",
        "",
        "| Variant | Coordinate macro MAE (m) | Energy Score (m) | Row MAE (m) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("baseline", "g1_gamma_0p7", "permuted_density"):
        rec = obj["metrics"][name]
        lines.append(
            f"| {name} | {rec['mae_2d_m']:.6f} | "
            f"{rec['energy_score_2d_m']:.6f} | {rec['mae_row_mean_m']:.6f} |"
        )
    lines.extend([
        "",
        f"- G1 minus baseline macro MAE: {obj['delta_coordinate_macro_mae_m']:+.6f} m",
        f"- G1 minus baseline macro ES: {obj['delta_coordinate_macro_energy_m']:+.6f} m",
        f"- Paired familywise MAE upper: {obj['paired_familywise_inference']['comparisons']['exact_mixture_mean_mae_m']['simultaneous_upper_bound_m']:+.6f} m",
        f"- Paired familywise ES upper: {obj['paired_familywise_inference']['comparisons']['energy_score_2d_m']['simultaneous_upper_bound_m']:+.6f} m",
        f"- Freeze gate: {obj['fixed_transfer_gate']['passed']}; decision: {obj['decision']}",
        "- The 25 cyclic E16 realizations share panel members and test coordinates; they are NOT independent replications.",
        "- UJI G1 formal decision remains STOP regardless of these test outcomes.",
        "- No post-test parameter selection or second attempt is authorized.",
        "", f"Immutable result self-SHA256: {obj['result_sha256']}", "",
    ])
    return "\n".join(lines)


def check() -> dict:
    manifest = _load_json(MANIFEST, "manifest_sha256")
    saved = prepare()
    if manifest != saved:
        raise G3Error("registered G3 manifest changed")
    frozen = _load_json(FREEZE, "result_sha256")
    if frozen["members"] != check_members()["members"]:
        raise G3Error("trained panel differs from frozen checkpoint hashes")
    if not LOCK.exists():
        return {"status": "FROZEN_NO_TEST_YET",
                "manifest_sha256": manifest["manifest_sha256"],
                "members": PANEL_SIZE}
    locked = _load_json(LOCK, "result_sha256")
    if locked["freeze_file_sha256"] != _sha256_file(FREEZE):
        raise G3Error("test lock no longer binds freeze")
    if not RESULT.exists():
        return {"status": "TEST_LOCK_EXISTS_WITHOUT_RESULT_STOP_DO_NOT_RERUN"}
    result = _load_json(RESULT, "result_sha256")
    if result["lock_file_sha256"] != _sha256_file(LOCK):
        raise G3Error("result lock binding mismatch")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != make_report(result):
        raise G3Error("report/render integrity mismatch")
    if result["historical_G1_decision"] != "STOP_GEOMETRY_DENSITY_MECHANISM":
        raise G3Error("historical G1 STOP overwritten")
    return {"status": "VERIFIED", "result_sha256": result["result_sha256"],
            "decision": result["decision"]}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("preflight", "prepare", "train-member", "check-members", "freeze", "evaluate-test", "check"):
        cmd = sub.add_parser(name)
        if name in ("train-member", "evaluate-test"):
            cmd.add_argument("--device", default="cuda")
        if name == "train-member":
            cmd.add_argument("--member-index", type=int, required=True)
    args = p.parse_args(argv)
    if args.command == "preflight":
        result = preflight()
    elif args.command == "prepare":
        result = prepare()
    elif args.command == "train-member":
        result = train_member(args.member_index, args.device)
    elif args.command == "check-members":
        result = check_members()
    elif args.command == "freeze":
        result = freeze()
    elif args.command == "evaluate-test":
        result = evaluate_test(args.device)
    else:
        result = check()
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
