"""V17 two-stage closeout: full 80-epoch paired training, then frozen posterior calibration.

One invocation runs one (dataset, seed) pair. No Joint refit is implemented.
The primary Stage-2 checkpoint is the lowest DEV-MAE epoch in [21, 80].
The legacy early-stop and unrestricted 80-epoch best checkpoints are retained
as predeclared sensitivity comparisons. CAL-fit and CAL-eval are site-disjoint.
No internal TEST role or official test labels are scored.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fhm_dapp_tim.fhm_dapp.msep_single_study import (
    _load, _seed, _prepare_fit, _pack, _json_safe,
)
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import (
    anchor_site_keys, split_physical_sites, energy_score_mc,
)
from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import (
    same_floor_neighbors, neighbor_contamination, class_structure,
    calibrate_probability, probability, spatial_metrics, site_mean,
    student_t_logpdf,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING


DATASETS = ("tuji", "uts", "uji")
SEEDS = (17301, 17302, 17303)
EPOCHS = 80
WARMUP_EPOCHS = 20
LEGACY_MIN_EPOCHS = 20
LEGACY_PATIENCE = 12
LEGACY_MIN_IMPROVEMENT_M = 0.01
PRIMARY_CHECKPOINT = "best_dev_epoch_21_80"
CHECKPOINT_POLICIES = (
    "legacy_early_stop",
    "best_dev_epoch_1_80",
    PRIMARY_CHECKPOINT,
)
CALIBRATION_POLICIES = (
    "raw",
    "temperature_only",
    "g13_fixed_t06",
    "temperature_plus_g13",
)
RAW_PARAMS = {"temperature": 1.0, "theta": [0.0, 0.0]}
CALIBRATION_MODE = {
    "temperature_only": "temperature",
    "g13_fixed_t06": "g13_fixed",
    "temperature_plus_g13": "joint",
}


def split_cal(rows, keys, seed):
    """Split CAL into site-disjoint parameter-fit and evaluation sites."""
    by_site = {}
    for index in rows:
        by_site.setdefault(keys[int(index)], []).append(int(index))
    sites = sorted(by_site)
    if len(sites) < 4:
        raise ValueError("CAL needs at least four independent physical sites")
    rng = np.random.default_rng(seed)
    selected = set(rng.permutation(len(sites))[:len(sites) // 2].tolist())
    fit, evaluation = [], []
    for position, site in enumerate(sites):
        (fit if position in selected else evaluation).extend(by_site[site])
    fit, evaluation = np.asarray(sorted(fit), dtype=np.int64), np.asarray(sorted(evaluation), dtype=np.int64)
    fit_sites = {keys[int(i)] for i in fit}
    evaluation_sites = {keys[int(i)] for i in evaluation}
    if fit_sites & evaluation_sites:
        raise AssertionError("CAL-fit/CAL-eval physical-site leakage")
    if set(fit) & set(evaluation):
        raise AssertionError("CAL-fit/CAL-eval row leakage")
    return fit, evaluation


def select_checkpoint_epochs(history, *, patience=LEGACY_PATIENCE,
                             minimum_epochs=LEGACY_MIN_EPOCHS,
                             minimum_improvement=LEGACY_MIN_IMPROVEMENT_M):
    """Pure audit helper reproducing the old early-stop rule on a full history."""
    if len(history) != EPOCHS:
        raise ValueError(f"expected exactly {EPOCHS} history rows, got {len(history)}")
    ordered = sorted(history, key=lambda row: int(row["epoch"]))
    expected = list(range(1, EPOCHS + 1))
    if [int(row["epoch"]) for row in ordered] != expected:
        raise ValueError("epoch history must contain every epoch 1..80 exactly once")
    scores = [float(row["dev_mae_m"]) for row in ordered]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("DEV MAE history contains nonfinite values")
    full_row = min(ordered, key=lambda row: float(row["dev_mae_m"]))
    post_warmup = [row for row in ordered if int(row["epoch"]) > WARMUP_EPOCHS]
    if not post_warmup:
        raise ValueError("no post-warmup epochs available")
    warm_row = min(post_warmup, key=lambda row: float(row["dev_mae_m"]))

    legacy_best_score = float("inf")
    legacy_best_epoch = 0
    legacy_stop_epoch = None
    for row in ordered:
        epoch = int(row["epoch"])
        score = float(row["dev_mae_m"])
        if score < legacy_best_score - minimum_improvement:
            legacy_best_score = score
            legacy_best_epoch = epoch
        if epoch >= minimum_epochs and epoch - legacy_best_epoch >= patience:
            legacy_stop_epoch = epoch
            break
    return {
        "legacy_early_stop": {
            "selected_epoch": legacy_best_epoch,
            "dev_mae_m": legacy_best_score,
            "simulated_stop_epoch": legacy_stop_epoch,
            "rule": "improvement >= 0.01m; stop when epoch>=20 and epoch-best_epoch>=12",
        },
        "best_dev_epoch_1_80": {
            "selected_epoch": int(full_row["epoch"]),
            "dev_mae_m": float(full_row["dev_mae_m"]),
            "simulated_stop_epoch": None,
            "rule": "minimum DEV MAE across epochs 1..80; exact minimum, no 0.01m threshold",
        },
        PRIMARY_CHECKPOINT: {
            "selected_epoch": int(warm_row["epoch"]),
            "dev_mae_m": float(warm_row["dev_mae_m"]),
            "simulated_stop_epoch": None,
            "rule": "minimum DEV MAE across epochs 21..80; exact minimum, no 0.01m threshold",
        },
    }


def _cpu_state_dict(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


@torch.no_grad()
def _dev_metrics(model, obs, rss, xy, keys, data, rows, anchor_class,
                 class_pairs, *, device, batch):
    pack = _pack(model, obs, rss, rows, xy, keys, k=None,
                 device=device, batch_size=batch)
    q = probability(pack["scores"], pack["z"], RAW_PARAMS)
    building = torch.as_tensor(np.asarray(data.building)[rows],
                               dtype=torch.long, device=device)
    floor = torch.as_tensor(np.asarray(data.floor)[rows],
                            dtype=torch.long, device=device)
    metrics, _ = spatial_metrics(
        q, pack["ids"], pack["xy"], pack["target"], building, floor,
        anchor_class, class_pairs, keys=pack["keys"])
    return {
        "dev_mae_m": float(metrics["macro_mae_full_m"]),
        "dev_mae_predicted_bf_m": float(metrics["macro_mae_predicted_bf_m"]),
        "dev_bf_accuracy_macro": float(metrics["macro_bf_accuracy"]),
        "dev_bf_accuracy_row": float(metrics["row_bf_accuracy"]),
        "dev_sites": int(metrics.get("n_sites", len(set(pack["keys"])))),
    }


def _train_fixed_80(model, rss, xy, sample_anchor, device_ids,
                    neighbor_indices, neighbor_weights, obs, roles, keys, data,
                    anchor_class, class_pairs, *, dataset_name, variant, seed,
                    device, batch):
    """Train exactly 80 epochs and retain only the three required CPU snapshots."""
    _seed(seed)  # Pair model/dropout randomness and shuffled row order across variants.
    fit_rows = roles["fit"]
    dataset = TensorDataset(
        rss[fit_rows], xy[fit_rows],
        torch.from_numpy(sample_anchor[fit_rows]).long(), device_ids[fit_rows])
    loader = DataLoader(
        dataset, batch_size=batch, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_TRAINING["learning_rate"],
        weight_decay=FROZEN_TRAINING["weight_decay"])

    history = []
    snapshots = {}
    full_best_score, full_best_epoch, full_best_state = float("inf"), None, None
    warm_best_score, warm_best_epoch, warm_best_state = float("inf"), None, None
    legacy_best_score, legacy_best_epoch, legacy_best_state = float("inf"), None, None
    legacy_stop_epoch = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = defaultdict(float)
        seen_rows = 0
        for x, y, anchor, domain in loader:
            x, y, anchor, domain = [tensor.to(device) for tensor in (x, y, anchor, domain)]
            optimizer.zero_grad(set_to_none=True)
            terms = model.training_loss(
                x, y, anchor, neighbor_indices, neighbor_weights,
                retrieval_weight=FROZEN_TRAINING["retrieval_weight"],
                hierarchy_weight=FROZEN_TRAINING["hierarchy_weight"],
                self_target_weight=FROZEN_TRAINING["self_target_weight"],
                coordinate_weight=FROZEN_TRAINING["coordinate_weight"],
                device_ids=domain, domain_adversarial_weight=0.0,
                rss_offset_aug_db=FROZEN_TRAINING["rss_offset_aug_db"],
                rss_scale_aug=FROZEN_TRAINING["rss_scale_aug"],
                train_candidates=FROZEN_TRAINING["train_candidates"],
                ap_dropout=FROZEN_TRAINING["ap_dropout"],
            )
            if not bool(torch.isfinite(terms["loss"])):
                raise RuntimeError(
                    f"nonfinite training loss: dataset={dataset_name}, seed={seed}, "
                    f"variant={variant}, epoch={epoch}")
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = int(len(x))
            seen_rows += count
            for name, value in terms.items():
                scalar = float(value.detach().cpu())
                if not math.isfinite(scalar):
                    raise RuntimeError(f"nonfinite {name} at epoch {epoch}")
                sums[name] += scalar * count
        if seen_rows != len(fit_rows):
            raise AssertionError("an epoch did not visit every FIT row exactly once")

        model.eval()
        dev = _dev_metrics(
            model, obs, rss, xy, keys, data, roles["dev"], anchor_class,
            class_pairs, device=device, batch=batch)
        score = dev["dev_mae_m"]
        row = {
            "dataset": dataset_name, "seed": seed, "variant": variant,
            "epoch": epoch, "fit_loss": sums["loss"] / seen_rows,
        }
        for name, total in sums.items():
            if name != "loss":
                row[f"fit_{name}"] = total / seen_rows
        row.update(dev)
        history.append(row)

        if score < full_best_score:
            full_best_score, full_best_epoch = score, epoch
            full_best_state = _cpu_state_dict(model)
        if epoch > WARMUP_EPOCHS and score < warm_best_score:
            warm_best_score, warm_best_epoch = score, epoch
            warm_best_state = _cpu_state_dict(model)

        # Simulate the original stopping rule but never actually break the 80-epoch loop.
        if legacy_stop_epoch is None:
            if score < legacy_best_score - LEGACY_MIN_IMPROVEMENT_M:
                legacy_best_score, legacy_best_epoch = score, epoch
                legacy_best_state = _cpu_state_dict(model)
            if (epoch >= LEGACY_MIN_EPOCHS
                    and epoch - int(legacy_best_epoch) >= LEGACY_PATIENCE):
                legacy_stop_epoch = epoch

        print(json.dumps({
            "stage": "epoch_end", "dataset": dataset_name, "seed": seed,
            "variant": variant, "epoch": epoch, "fit_loss": row["fit_loss"],
            "dev_mae_m": score, "dev_bf_accuracy_macro": dev["dev_bf_accuracy_macro"],
        }, allow_nan=False), flush=True)

    if len(history) != EPOCHS:
        raise AssertionError(f"expected {EPOCHS} epochs, got {len(history)}")
    snapshots = {
        "legacy_early_stop": legacy_best_state,
        "best_dev_epoch_1_80": full_best_state,
        PRIMARY_CHECKPOINT: warm_best_state,
    }
    if any(value is None for value in snapshots.values()):
        raise RuntimeError("one or more checkpoint policies failed to select a finite checkpoint")
    selected = select_checkpoint_epochs(history)
    selected["legacy_early_stop"]["simulated_stop_epoch"] = legacy_stop_epoch
    tracked_epochs = {
        "legacy_early_stop": legacy_best_epoch,
        "best_dev_epoch_1_80": full_best_epoch,
        PRIMARY_CHECKPOINT: warm_best_epoch,
    }
    for policy, epoch in tracked_epochs.items():
        if selected[policy]["selected_epoch"] != epoch:
            raise AssertionError(
                f"checkpoint-selection audit mismatch for {variant}/{policy}: "
                f"tracked={epoch}, replay={selected[policy]['selected_epoch']}")
    return history, snapshots, selected


@torch.no_grad()
def _metrics_for_pack(pack, params, data, rows, anchor_class, class_pairs,
                      *, seed, draws):
    q = probability(pack["scores"], pack["z"], params)
    device = q.device
    building = torch.as_tensor(np.asarray(data.building)[rows],
                               dtype=torch.long, device=device)
    floor = torch.as_tensor(np.asarray(data.floor)[rows],
                            dtype=torch.long, device=device)
    metrics, row = spatial_metrics(
        q, pack["ids"], pack["xy"], pack["target"], building, floor,
        anchor_class, class_pairs, keys=pack["keys"])
    log_density = student_t_logpdf(pack["xy"], pack["sigma"], pack["target"])
    anchor_class_for_row = anchor_class[pack["ids"]]
    true_bf = (
        (class_pairs[:, 0][None, :] == building[:, None])
        & (class_pairs[:, 1][None, :] == floor[:, None]))
    if not bool(true_bf.any(dim=1).all()):
        raise RuntimeError("an evaluation BF class has no FIT reference support")
    true_class_id = true_bf.to(torch.int32).argmax(1)
    correct_class_mask = anchor_class_for_row == true_class_id[:, None]
    joint_nll = -torch.logsumexp(
        (q.clamp_min(1e-30).log() + log_density)
        .masked_fill(~correct_class_mask, -torch.inf), dim=1)
    es = energy_score_mc(
        q, pack["xy"], pack["sigma"], pack["target"],
        draws=draws, seed=seed).detach().cpu().numpy()
    nll = joint_nll.detach().cpu().numpy()
    metrics.update({
        "macro_es_m": float(site_mean(es, pack["keys"])),
        "row_es_m": float(np.mean(es)),
        "macro_joint_nll": float(site_mean(nll, pack["keys"])),
        "n_scans": int(len(rows)),
        "n_sites": int(len(set(pack["keys"]))),
    })
    grouped = {}
    for index, key in enumerate(pack["keys"]):
        key_text = json.dumps(list(key), separators=(",", ":"))
        group = grouped.setdefault(key_text, {
            "site": list(key), "indices": [], "errors": [],
            "errors_predicted_bf": [], "bf": [], "es": [], "joint_nll": [],
        })
        group["indices"].append(index)
        group["errors"].append(float(row["err_full"][index]))
        group["errors_predicted_bf"].append(float(row["err_predicted_bf"][index]))
        group["bf"].append(float(row["bf_correct"][index]))
        group["es"].append(float(es[index]))
        group["joint_nll"].append(float(nll[index]))
    sites = {}
    for key_text, group in grouped.items():
        sites[key_text] = {
            "site": group["site"],
            "n_scans": len(group["indices"]),
            "mae_full_m": float(np.mean(group["errors"])),
            "mae_predicted_bf_m": float(np.mean(group["errors_predicted_bf"])),
            "bf_accuracy": float(np.mean(group["bf"])),
            "energy_score_m": float(np.mean(group["es"])),
            "joint_nll": float(np.mean(group["joint_nll"])),
        }
    return metrics, sites


def _site_deltas(reference, candidate):
    if set(reference) != set(candidate):
        raise AssertionError("paired per-site metrics do not have identical site keys")
    output = {}
    for key in sorted(reference):
        ref, cand = reference[key], candidate[key]
        output[key] = {
            "site": ref["site"],
            "n_scans": ref["n_scans"],
            "mae_delta_m": float(cand["mae_full_m"] - ref["mae_full_m"]),
            "energy_score_delta_m": float(cand["energy_score_m"] - ref["energy_score_m"]),
            "bf_accuracy_delta": float(cand["bf_accuracy"] - ref["bf_accuracy"]),
        }
    return output


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_one(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    if args.seed < 0 or args.batch < 1 or args.threads < 1 or args.es_draws < 2:
        raise ValueError("invalid seed, batch, thread count, or Energy Score draw count")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)

    _seed(args.seed)
    data = _load(args.dataset, Path(args.data_root) if args.data_root else None)
    keys = anchor_site_keys(data.coords_m, data.building, data.floor)
    roles = split_physical_sites(keys, seed=args.split_seed)
    cal_fit, cal_eval = split_cal(roles["cal"], keys, args.split_seed + 99)
    (rss, xy, origin, memory, sample_anchor, obs, device_ids, initial,
     old_neighbors, old_weights) = _prepare_fit(data, roles, device)
    floor_neighbors, floor_weights = same_floor_neighbors(
        memory["coords"], memory["building"], memory["floor"],
        count=FROZEN_TRAINING["target_neighbors"],
        temperature_m=FROZEN_TRAINING["spatial_temperature_m"])
    floor_neighbors, floor_weights = floor_neighbors.to(device), floor_weights.to(device)
    class_pairs, anchor_class = class_structure(
        memory["building"].to(device), memory["floor"].to(device))
    old_contamination = neighbor_contamination(
        old_neighbors.detach().cpu(), old_weights.detach().cpu(),
        memory["building"], memory["floor"])
    floor_contamination = neighbor_contamination(
        floor_neighbors.detach().cpu(), floor_weights.detach().cpu(),
        memory["building"], memory["floor"])
    bf_strata = set(zip(memory["building"].tolist(), memory["floor"].tolist()))
    single_bf_stratum = len(bf_strata) == 1
    same_floor_is_noop = (
        torch.equal(old_neighbors.detach().cpu(), floor_neighbors.detach().cpu())
        and torch.allclose(old_weights.detach().cpu(), floor_weights.detach().cpu(),
                           atol=1e-7, rtol=1e-7))
    if single_bf_stratum and not same_floor_is_noop:
        raise AssertionError(
            "single-BF dataset should make same_floor neighbor targets an exact no-op")

    site_counts = {
        role: {
            "rows": int(len(indices)),
            "sites": int(len({keys[int(i)] for i in indices})),
        }
        for role, indices in roles.items()
    }
    role_site_sha = {
        role: hashlib.sha256(json.dumps(
            sorted({keys[int(i)] for i in indices}),
            separators=(",", ":")).encode()).hexdigest()
        for role, indices in roles.items()
    }
    report = {
        "study": "V17 two-stage full-80-epoch and frozen posterior calibration closeout",
        "dataset": args.dataset, "seed": args.seed, "split_seed": args.split_seed,
        "device": str(device), "config": {
            "epochs_per_variant": EPOCHS, "batch": args.batch, "threads": args.threads,
            "es_draws": args.es_draws, "primary_checkpoint_policy": PRIMARY_CHECKPOINT,
            "legacy_min_epochs": LEGACY_MIN_EPOCHS, "legacy_patience": LEGACY_PATIENCE,
            "legacy_min_improvement_m": LEGACY_MIN_IMPROVEMENT_M,
            "calibration_fit_split_seed": args.split_seed + 99,
        },
        "data_protocol": {
            "role_isolation": "physical site (building, floor, XY rounded to four decimals)",
            "site_counts": site_counts, "role_site_key_sha256": role_site_sha,
            "cal_fit_sites": int(len({keys[int(i)] for i in cal_fit})),
            "cal_eval_sites": int(len({keys[int(i)] for i in cal_eval})),
            "cal_fit_eval_site_overlap": 0,
            "official_test_rows_scored": 0,
            "internal_test_role_rows_scored": 0,
        },
        "neighbor_audit": {
            "old_neighbor_wrong_bf_mass": old_contamination,
            "same_floor_neighbor_wrong_bf_mass": floor_contamination,
            "fit_bf_strata": len(bf_strata),
            "single_bf_stratum": single_bf_stratum,
            "same_floor_is_noop": same_floor_is_noop,
        },
        "stage1": {"training": {}, "cal_eval_checkpoint_metrics": {},
                   "primary_checkpoint_policy": PRIMARY_CHECKPOINT},
        "stage2": {"frozen_backbone_and_continuous_components": True,
                   "calibration_fit_role": "CAL-fit",
                   "evaluation_role": "CAL-eval",
                   "primary_checkpoint_policy": PRIMARY_CHECKPOINT,
                   "variants": {}},
        "limitations": [
            "CAL-eval is a site-disjoint internal comparison split, not a new blind official-test benchmark.",
            "The unrestricted best-DEV and legacy early-stop checkpoints are sensitivity diagnostics; CAL-eval is not used to select the primary checkpoint.",
            "Stage 2 changes posterior temperature and/or G13 coefficients only; encoder, candidate IDs, centers, and scales are fixed.",
            "Raw V17 and same-floor V17 use the same initial state, optimizer configuration, paired seed, and data-loader seed.",
            "No post-hoc Joint fine-tuning is implemented or run.",
        ],
    }

    site_diagnostics = {
        "stage1_cal_eval": {},
        "stage1_neighbor_deltas": {},
        "stage2_frozen_calibration": {},
    }
    epoch_rows = []
    selection_rows = []
    variants = [
        ("raw_neighbors", old_neighbors, old_weights, old_contamination),
        ("same_floor_neighbors", floor_neighbors, floor_weights, floor_contamination),
    ]
    trained = {}

    for variant, neighbor_indices, neighbor_weights, contamination in variants:
        model = copy.deepcopy(initial)
        history, states, selections = _train_fixed_80(
            model, rss, xy, sample_anchor, device_ids, neighbor_indices,
            neighbor_weights, obs, roles, keys, data, anchor_class, class_pairs,
            dataset_name=args.dataset, variant=variant, seed=args.seed,
            device=device, batch=args.batch)
        epoch_rows.extend(history)
        report["stage1"]["training"][variant] = {
            "epochs_completed": len(history),
            "all_epoch_history": history,
            "checkpoint_selection": selections,
            "neighbor_wrong_bf_mass": contamination,
        }
        trained[variant] = {"model": model, "states": states, "selections": selections}
        variant_site_metrics = {}
        report["stage1"]["cal_eval_checkpoint_metrics"][variant] = {}
        site_diagnostics["stage1_cal_eval"][variant] = {}
        for policy in CHECKPOINT_POLICIES:
            selection = selections[policy]
            model.load_state_dict(states[policy])
            model.eval()
            checkpoint_path = output / f"{variant}_{policy}.pt"
            torch.save({
                "state_dict": states[policy],
                "dataset": args.dataset, "seed": args.seed,
                "variant": variant, "checkpoint_policy": policy,
                "selected_epoch": selection["selected_epoch"],
                "dev_mae_m": selection["dev_mae_m"],
                "simulated_legacy_stop_epoch": selection["simulated_stop_epoch"],
                "epochs_trained_in_this_run": EPOCHS,
                "frozen_training_config": FROZEN_TRAINING,
                "no_joint_refit": True,
            }, checkpoint_path)
            eval_pack = _pack(
                model, obs, rss, cal_eval, xy, keys, k=None,
                device=device, batch_size=args.batch)
            metrics, sites = _metrics_for_pack(
                eval_pack, RAW_PARAMS, data, cal_eval, anchor_class, class_pairs,
                seed=args.seed + 900, draws=args.es_draws)
            report["stage1"]["cal_eval_checkpoint_metrics"][variant][policy] = {
                "selected_epoch": selection["selected_epoch"],
                "dev_mae_m": selection["dev_mae_m"],
                "metrics": metrics,
            }
            site_diagnostics["stage1_cal_eval"][variant][policy] = sites
            variant_site_metrics[policy] = sites
            selection_rows.append({
                "dataset": args.dataset, "seed": args.seed, "variant": variant,
                "checkpoint_policy": policy, "selected_epoch": selection["selected_epoch"],
                "dev_mae_m": selection["dev_mae_m"],
                "simulated_legacy_stop_epoch": selection["simulated_stop_epoch"],
                "is_primary_for_stage2": policy == PRIMARY_CHECKPOINT,
                "cal_eval_macro_mae_m": metrics["macro_mae_full_m"],
                "cal_eval_macro_es_m": metrics["macro_es_m"],
                "cal_eval_macro_bf_accuracy": metrics["macro_bf_accuracy"],
            })
            del eval_pack

    for policy in CHECKPOINT_POLICIES:
        site_diagnostics["stage1_neighbor_deltas"][policy] = _site_deltas(
            site_diagnostics["stage1_cal_eval"]["raw_neighbors"][policy],
            site_diagnostics["stage1_cal_eval"]["same_floor_neighbors"][policy])

    # Stage 2: use only the predeclared post-warmup best-DEV checkpoint.
    for variant, _, _, contamination in variants:
        model = trained[variant]["model"]
        model.load_state_dict(trained[variant]["states"][PRIMARY_CHECKPOINT])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        cal_fit_pack = _pack(
            model, obs, rss, cal_fit, xy, keys, k=None,
            device=device, batch_size=args.batch)
        cal_eval_pack = _pack(
            model, obs, rss, cal_eval, xy, keys, k=None,
            device=device, batch_size=args.batch)

        params_by_policy = {"raw": dict(RAW_PARAMS)}
        for policy in ("temperature_only", "g13_fixed_t06", "temperature_plus_g13"):
            params_by_policy[policy] = calibrate_probability(
                cal_fit_pack, mode=CALIBRATION_MODE[policy])

        policy_results = {}
        site_results = {}
        for policy in CALIBRATION_POLICIES:
            params = params_by_policy[policy]
            metrics, sites = _metrics_for_pack(
                cal_eval_pack, params, data, cal_eval, anchor_class, class_pairs,
                seed=args.seed + 900, draws=args.es_draws)
            policy_results[policy] = {"params": params, "metrics": metrics}
            site_results[policy] = sites
        raw_sites = site_results["raw"]
        for policy in CALIBRATION_POLICIES:
            if policy == "raw":
                site_diagnostics["stage2_frozen_calibration"].setdefault(variant, {})[policy] = {
                    "params": params_by_policy[policy], "sites": site_results[policy],
                    "delta_vs_raw": _site_deltas(raw_sites, raw_sites),
                }
            else:
                site_diagnostics["stage2_frozen_calibration"].setdefault(variant, {})[policy] = {
                    "params": params_by_policy[policy], "sites": site_results[policy],
                    "delta_vs_raw": _site_deltas(raw_sites, site_results[policy]),
                }
        report["stage2"]["variants"][variant] = {
            "checkpoint_policy": PRIMARY_CHECKPOINT,
            "checkpoint_epoch": trained[variant]["selections"][PRIMARY_CHECKPOINT]["selected_epoch"],
            "neighbor_wrong_bf_mass": contamination,
            "calibration": policy_results,
            "frozen_pack_invariant": {
                "same_candidate_ids_across_policies": True,
                "same_continuous_centers_and_scales_across_policies": True,
                "only_temperature_and_g13_coefficients_change": True,
            },
        }
        del cal_fit_pack, cal_eval_pack

    # Cross-check: Stage-1 post-warmup raw result and Stage-2 raw must be identical.
    for variant in ("raw_neighbors", "same_floor_neighbors"):
        a = report["stage1"]["cal_eval_checkpoint_metrics"][variant][PRIMARY_CHECKPOINT]["metrics"]
        b = report["stage2"]["variants"][variant]["calibration"]["raw"]["metrics"]
        for metric in ("macro_mae_full_m", "macro_es_m", "macro_bf_accuracy"):
            if not math.isclose(float(a[metric]), float(b[metric]), rel_tol=1e-7, abs_tol=1e-7):
                raise AssertionError(f"Stage-1/Stage-2 raw metric mismatch: {variant}/{metric}")

    report["elapsed_seconds"] = time.perf_counter() - started
    report["source_sha256"] = {
        path.name: _sha256(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("msep_single_study.py"),
            Path(__file__).with_name("msep_single_posterior.py"),
            Path(__file__).with_name("msep_joint_science_v1.py"),
            Path(__file__).with_name("model.py"),
            Path(__file__).with_name("train_anchor.py"),
            Path(__file__).with_name("train_anchor_v2.py"),
        )
    }
    result_path = output / "RESULT.json"
    result_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    site_path = output / "SITE_DIAGNOSTICS.json"
    site_path.write_text(
        json.dumps(_json_safe(site_diagnostics), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    with (output / "EPOCH_HISTORY.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)
    with (output / "CHECKPOINT_SELECTION.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)
    manifest = {
        "study": report["study"], "dataset": args.dataset, "seed": args.seed,
        "split_seed": args.split_seed, "device": str(device),
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "epochs_per_variant": EPOCHS, "variant_count": 2,
        "joint_refit_implemented": False, "joint_refit_run": False,
        "evaluation_role": "CAL-eval", "official_test_rows_scored": 0,
        "result_sha256": _sha256(result_path),
        "site_diagnostics_sha256": _sha256(site_path),
        "source_sha256": report["source_sha256"],
    }
    (output / "RUN_MANIFEST.json").write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(json.dumps({
        "stage": "complete", "dataset": args.dataset, "seed": args.seed,
        "output": str(output), "epochs_per_variant": EPOCHS,
        "primary_checkpoint_policy": PRIMARY_CHECKPOINT,
        "stage2_variants": list(report["stage2"]["variants"]),
        "official_test_rows_scored": 0,
    }, ensure_ascii=False), flush=True)
    return report


def _metric_row(stage, dataset, seed, variant, checkpoint_policy,
                calibration_policy, selected_epoch, params, metrics):
    theta = params.get("theta", [0.0, 0.0]) if params else [0.0, 0.0]
    return {
        "stage": stage, "dataset": dataset, "seed": seed, "variant": variant,
        "checkpoint_policy": checkpoint_policy, "calibration_policy": calibration_policy,
        "selected_epoch": selected_epoch,
        "temperature": params.get("temperature") if params else None,
        "theta_visibility": theta[0], "theta_strength": theta[1],
        "macro_mae_full_m": metrics.get("macro_mae_full_m"),
        "macro_mae_predicted_bf_m": metrics.get("macro_mae_predicted_bf_m"),
        "macro_es_m": metrics.get("macro_es_m"),
        "macro_bf_accuracy": metrics.get("macro_bf_accuracy"),
        "row_bf_accuracy": metrics.get("row_bf_accuracy"),
        "macro_joint_nll": metrics.get("macro_joint_nll"),
        "n_sites": metrics.get("n_sites"), "n_scans": metrics.get("n_scans"),
    }


def aggregate(args):
    root = Path(args.root).resolve()
    expected = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
    loaded = {}
    for dataset, seed in expected:
        path = root / f"{dataset}_seed{seed}" / "RESULT.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing completed run {path}; finish all nine paired runs before aggregation")
        loaded[(dataset, seed)] = json.loads(path.read_text(encoding="utf-8"))

    run_rows = []
    paired_rows = []
    site_rows = []
    summary_values = defaultdict(list)
    summary_epoch_values = defaultdict(list)

    for (dataset, seed), result in loaded.items():
        stage1 = result["stage1"]["cal_eval_checkpoint_metrics"]
        stage2 = result["stage2"]["variants"]
        for variant in ("raw_neighbors", "same_floor_neighbors"):
            for checkpoint_policy in CHECKPOINT_POLICIES:
                entry = stage1[variant][checkpoint_policy]
                metrics = entry["metrics"]
                row = _metric_row(
                    "stage1_raw_checkpoint", dataset, seed, variant,
                    checkpoint_policy, "raw", entry["selected_epoch"], RAW_PARAMS, metrics)
                run_rows.append(row)
                key = ("stage1_raw_checkpoint", dataset, variant, checkpoint_policy, "raw")
                summary_values[key].append(row)
                summary_epoch_values[key].append(entry["selected_epoch"])
        for variant in ("raw_neighbors", "same_floor_neighbors"):
            base = stage2[variant]
            for policy in CALIBRATION_POLICIES:
                entry = base["calibration"][policy]
                row = _metric_row(
                    "stage2_frozen_calibration", dataset, seed, variant,
                    PRIMARY_CHECKPOINT, policy, base["checkpoint_epoch"],
                    entry["params"], entry["metrics"])
                run_rows.append(row)
                key = ("stage2_frozen_calibration", dataset, variant, PRIMARY_CHECKPOINT, policy)
                summary_values[key].append(row)
                summary_epoch_values[key].append(base["checkpoint_epoch"])
                if policy != "raw":
                    raw = base["calibration"]["raw"]["metrics"]
                    paired_rows.append({
                        "stage": "stage2_calibration_vs_raw", "dataset": dataset,
                        "seed": seed, "variant": variant, "policy": policy,
                        "checkpoint_policy": PRIMARY_CHECKPOINT,
                        "mae_delta_m": entry["metrics"]["macro_mae_full_m"] - raw["macro_mae_full_m"],
                        "es_delta_m": entry["metrics"]["macro_es_m"] - raw["macro_es_m"],
                        "bf_accuracy_delta": entry["metrics"]["macro_bf_accuracy"] - raw["macro_bf_accuracy"],
                    })
        for checkpoint_policy in CHECKPOINT_POLICIES:
            raw = stage1["raw_neighbors"][checkpoint_policy]["metrics"]
            floor = stage1["same_floor_neighbors"][checkpoint_policy]["metrics"]
            paired_rows.append({
                "stage": "stage1_same_floor_vs_raw_neighbors", "dataset": dataset,
                "seed": seed, "variant": "same_floor_neighbors_minus_raw_neighbors",
                "policy": checkpoint_policy, "checkpoint_policy": checkpoint_policy,
                "mae_delta_m": floor["macro_mae_full_m"] - raw["macro_mae_full_m"],
                "es_delta_m": floor["macro_es_m"] - raw["macro_es_m"],
                "bf_accuracy_delta": floor["macro_bf_accuracy"] - raw["macro_bf_accuracy"],
            })

        diagnostics_path = root / f"{dataset}_seed{seed}" / "SITE_DIAGNOSTICS.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        for checkpoint_policy, sites in diagnostics["stage1_neighbor_deltas"].items():
            for site_key, item in sites.items():
                site_rows.append({
                    "stage": "stage1_same_floor_vs_raw_neighbors",
                    "dataset": dataset, "seed": seed, "variant": "same_floor_minus_raw",
                    "policy": checkpoint_policy, "site_key": site_key,
                    "n_scans": item["n_scans"], "mae_delta_m": item["mae_delta_m"],
                    "es_delta_m": item["energy_score_delta_m"],
                    "bf_accuracy_delta": item["bf_accuracy_delta"],
                })
        for variant, policies in diagnostics["stage2_frozen_calibration"].items():
            for policy, item in policies.items():
                for site_key, delta in item["delta_vs_raw"].items():
                    site_rows.append({
                        "stage": "stage2_calibration_vs_raw",
                        "dataset": dataset, "seed": seed, "variant": variant,
                        "policy": policy, "site_key": site_key,
                        "n_scans": delta["n_scans"], "mae_delta_m": delta["mae_delta_m"],
                        "es_delta_m": delta["energy_score_delta_m"],
                        "bf_accuracy_delta": delta["bf_accuracy_delta"],
                    })

    if run_rows:
        with (root / "RUN_LEVEL_METRICS.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
    if paired_rows:
        with (root / "PAIRED_DELTAS.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
            writer.writeheader()
            writer.writerows(paired_rows)
    if site_rows:
        with (root / "SITE_DELTAS_ALL_RUNS.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(site_rows[0].keys()))
            writer.writeheader()
            writer.writerows(site_rows)

    summary_rows = []
    for key, rows in sorted(summary_values.items()):
        stage, dataset, variant, checkpoint_policy, calibration_policy = key
        summary = {
            "stage": stage, "dataset": dataset, "variant": variant,
            "checkpoint_policy": checkpoint_policy, "calibration_policy": calibration_policy,
            "n_seeds": len(rows),
            "selected_epoch_mean": float(np.mean(summary_epoch_values[key])),
            "selected_epoch_sd": float(statistics.stdev(summary_epoch_values[key]))
            if len(summary_epoch_values[key]) > 1 else 0.0,
        }
        for metric in (
            "macro_mae_full_m", "macro_mae_predicted_bf_m", "macro_es_m",
            "macro_bf_accuracy", "row_bf_accuracy", "macro_joint_nll",
        ):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_sd"] = float(statistics.stdev(values)) if len(values) > 1 else (0.0 if values else None)
        summary_rows.append(summary)
    with (root / "AGGREGATE_SUMMARY.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = {
        "expected_runs": len(expected), "completed_runs": len(loaded),
        "datasets": list(DATASETS), "paired_seeds": list(SEEDS),
        "primary_checkpoint_policy": PRIMARY_CHECKPOINT,
        "calibration_policies": list(CALIBRATION_POLICIES),
        "files": [
            "RUN_LEVEL_METRICS.csv", "AGGREGATE_SUMMARY.csv",
            "PAIRED_DELTAS.csv", "SITE_DELTAS_ALL_RUNS.csv",
        ],
        "interpretation": [
            "All means and sample standard deviations summarize three paired seeds on the same internal CAL-eval sites.",
            "Negative MAE/ES delta means improvement; positive BF-accuracy delta means improvement.",
            "Site-level deltas are descriptive and are not independent-site significance tests.",
            "No internal TEST role or official test labels were evaluated.",
        ],
    }
    (root / "AGGREGATE_SUMMARY.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "aggregate_complete", "root": str(root),
                      "runs": len(loaded), "summary_rows": len(summary_rows),
                      "site_delta_rows": len(site_rows)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run one dataset/seed pair")
    run_parser.add_argument("--dataset", choices=DATASETS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--data-root", default=None)
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run_parser.add_argument("--split-seed", type=int, default=20261017)
    run_parser.add_argument("--batch", type=int, default=64)
    run_parser.add_argument("--threads", type=int, default=4)
    run_parser.add_argument("--es-draws", type=int, default=32)
    aggregate_parser = sub.add_parser("aggregate", help="aggregate all nine completed runs")
    aggregate_parser.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_one(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
