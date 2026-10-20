"""Zero-retraining composition evaluation for V17 Structures 3 + 1 + 2.

This module only loads frozen C/D/E checkpoints and fits calibration parameters
on CAL-fit. It never constructs an optimizer or scores internal/official TEST.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import msep_v17_spatial_operators_v1 as op
from . import msep_v17_spatial_structure_v1 as s3

REPO = s3.REPO
DEFAULT_INPUT = s3.DEFAULT_INPUT
DEFAULT_CHECKPOINT_ROOT = s3.DEFAULT_OUTPUT
DEFAULT_OUTPUT = REPO / "fhm_dapp_tim/evidence/v17_spatial_composition_v1"
ARMS = ("C", "D", "E")
S1_METHODS = ("1A_global", "1D_global")
METHOD_NAMES = (
    "joint_calibrated_base",
    "1A_global",
    "1D_global",
    "2A",
    "2B",
    "2A_beta_only_control",
    "combo_1A_2A",
    "combo_1A_2B",
    "combo_1A_2A_beta_only_control",
    "combo_1D_2A",
    "combo_1D_2B",
    "combo_1D_2A_beta_only_control",
)
TEST_COUNTS = {"internal_test_rows_scored": 0, "official_test_rows_scored": 0}
METRICS = (
    "macro_mae_full_m",
    "macro_mae_predicted_bf_m",
    "macro_es_m",
    "macro_joint_nll",
    "macro_bf_accuracy",
    "row_bf_accuracy",
    "site_p95_full_m",
    "site_p99_full_m",
    "worst_site_mae_m",
    "macro_transport_risk_m",
)
S1_SELECTION_POLICY = (
    "Select alpha_prior/alpha_obs for 1A/1D on CAL-fit with clean T/theta; "
    "freeze those alpha values for composition; select beta/gamma conditionally "
    "on the diffused parent logits using CAL-fit only."
)


def _json_write(path: Path, value, *, exclusive: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(s3._json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _csv_read(path: Path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_write(path: Path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_hashes():
    hashes = dict(s3.source_hashes())
    for path in (
        REPO / "V17_SPATIAL_COMPOSITION_DESIGN.md",
        REPO / "V17_SPATIAL_COMPOSITION_CODEX_RUN.txt",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required composition source is missing: {path}")
        hashes[str(path.relative_to(REPO)).replace("\\", "/")] = s3.sha(path)
    return dict(sorted(hashes.items()))


def _checkpoint_path(checkpoint_root: Path, dataset: str, seed: int, arm: str) -> Path:
    return (
        Path(checkpoint_root)
        / "trained"
        / f"{dataset}_seed{seed}"
        / arm
        / "best_dev_epoch_21_80.pt"
    )


def _verify_checkpoint(checkpoint_root: Path, dataset: str, seed: int, arm: str, phase_t_hashes):
    run_dir = Path(checkpoint_root) / "trained" / f"{dataset}_seed{seed}" / arm
    path = _checkpoint_path(checkpoint_root, dataset, seed, arm)
    result_path = run_dir / "RESULT.json"
    selection_path = run_dir / "CHECKPOINT_SELECTION.csv"
    for required in (path, result_path, selection_path):
        if not required.is_file():
            raise FileNotFoundError(f"required frozen checkpoint artifact is missing: {required}")
    digest = s3.sha(path)
    relative = str(path.relative_to(checkpoint_root)).replace("\\", "/")
    if relative not in phase_t_hashes:
        raise RuntimeError(f"Phase-T integrity review does not bind checkpoint: {relative}")
    if phase_t_hashes[relative] != digest:
        raise RuntimeError(f"Phase-T checkpoint hash mismatch: {relative}")
    result = s3.read_json(result_path)
    if (result.get("phase"), result.get("dataset"), result.get("seed"), result.get("backbone")) != (
        "T", dataset, seed, arm
    ):
        raise RuntimeError(f"checkpoint result identity mismatch: {result_path}")
    if result.get("checkpoint", {}).get("sha256") != digest:
        raise RuntimeError(f"RESULT.json checkpoint SHA mismatch: {result_path}")
    chosen = _csv_read(selection_path)
    if len(chosen) != 1 or chosen[0].get("checkpoint_sha256") != digest:
        raise RuntimeError(f"checkpoint selection record mismatch: {selection_path}")
    if int(chosen[0]["epoch"]) != int(result["selected_epoch"]):
        raise RuntimeError(f"checkpoint epoch mismatch: {selection_path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint.get("dataset"), checkpoint.get("seed"), checkpoint.get("arm")) != (dataset, seed, arm):
        raise RuntimeError(f"checkpoint payload identity mismatch: {path}")
    if int(checkpoint.get("selected_epoch", -1)) != int(result["selected_epoch"]):
        raise RuntimeError(f"checkpoint payload epoch mismatch: {path}")
    if checkpoint.get("frozen_training_config") != s3.FROZEN_TRAINING:
        raise RuntimeError(f"checkpoint training config mismatch: {path}")
    if int(checkpoint.get("epochs_trained_in_this_run", 80)) != 80:
        raise RuntimeError(f"checkpoint is not an 80-epoch frozen run: {path}")
    return {
        "path": relative,
        "sha256": digest,
        "selected_epoch": int(result["selected_epoch"]),
        "dataset": dataset,
        "seed": seed,
        "arm": arm,
    }


def _verify_source_manifest(manifest):
    current = _source_hashes()
    if current != manifest["source_sha256"]:
        changed = sorted(set(current) | set(manifest["source_sha256"]))
        changed = [p for p in changed if current.get(p) != manifest["source_sha256"].get(p)]
        raise RuntimeError(f"source changed after preflight; rerun preflight on a fresh output root: {changed[:20]}")
    review_path = Path(manifest["checkpoint_root"]) / "PHASE_T_INTEGRITY_REVIEW.json"
    if not review_path.is_file() or s3.sha(review_path) != manifest["phase_t_integrity_review_sha256"]:
        raise RuntimeError("Phase-T integrity review changed after preflight")


def preflight(args):
    started = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; rerun preflight/all explicitly with --device cpu")
    input_root = Path(args.input_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root = Path(args.output).resolve()
    if not (input_root / "INTEGRITY_REVIEW.json").is_file():
        raise FileNotFoundError(f"source integrity review missing under {input_root}")
    input_review = s3.read_json(input_root / "INTEGRITY_REVIEW.json")
    if input_review.get("status") != "PASS":
        raise RuntimeError("source/split root integrity review is not PASS; fail closed")
    if not (checkpoint_root / "EXECUTION_MANIFEST.json").is_file():
        raise FileNotFoundError(f"spatial execution manifest missing under {checkpoint_root}")
    phase_t_review_path = checkpoint_root / "PHASE_T_INTEGRITY_REVIEW.json"
    if not phase_t_review_path.is_file():
        raise FileNotFoundError(f"Phase-T integrity review missing: {phase_t_review_path}")
    saved_phase_t = s3.read_json(phase_t_review_path)
    if saved_phase_t.get("status") != "PASS" or saved_phase_t.get("completed") != 45:
        raise RuntimeError("existing Phase-T audit is not a complete PASS; fail closed")
    # This replays the existing study's frozen-input, split, result, and site reductions.
    recomputed_phase_t = s3.audit_results(checkpoint_root, phase="T", require_complete=True, write=False)
    if saved_phase_t.get("artifact_sha256") != recomputed_phase_t.get("artifact_sha256"):
        raise RuntimeError("saved Phase-T integrity hashes differ from a fresh read-only audit")
    phase_t_hashes = recomputed_phase_t["artifact_sha256"]
    source_hashes = _source_hashes()
    checkpoints, splits = {}, {}
    for dataset in s3.DATASETS:
        for seed in s3.SEEDS:
            ctx = s3.reconstruct(dataset, seed, input_root, torch.device("cpu"))
            run_key = f"{dataset}_seed{seed}"
            fit_xy = ctx["memory"]["coords"].detach().cpu().numpy()
            fit_edges = ctx["edges"].detach().cpu().numpy()
            if len(fit_edges):
                edge_lengths = np.linalg.norm(
                    fit_xy[fit_edges[:, 0]] - fit_xy[fit_edges[:, 1]], axis=1
                )
                edge_diagnostics = {
                    "length_p50_m": float(np.quantile(edge_lengths, .50)),
                    "length_p95_m": float(np.quantile(edge_lengths, .95)),
                    "length_max_m": float(edge_lengths.max()),
                    "count_gt_30m": int((edge_lengths > 30.0).sum()),
                    "fraction_gt_30m": float((edge_lengths > 30.0).mean()),
                }
            else:
                edge_diagnostics = {
                    "length_p50_m": 0.0, "length_p95_m": 0.0, "length_max_m": 0.0,
                    "count_gt_30m": 0, "fraction_gt_30m": 0.0,
                }
            splits[run_key] = {
                "cal_fit_key_sha256": s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_fit"]),
                "cal_eval_key_sha256": s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"]),
                "cal_fit_rows": int(len(ctx["cal_fit"])),
                "cal_eval_rows": int(len(ctx["cal_eval"])),
                "physical_anchors": int(len(ctx["memory"]["coords"])),
                "unique_midpoint_edges": int(len(ctx["edges"])),
                "graph_edge_diagnostics": edge_diagnostics,
            }
            for arm in ARMS:
                record = _verify_checkpoint(checkpoint_root, dataset, seed, arm, phase_t_hashes)
                checkpoints[record["path"]] = record["sha256"]
            del ctx
            gc.collect()
    manifest = {
        "status": "PASS",
        "study": "V17 zero-retraining composition of Structure 3 + Structure 1 + Structure 2",
        "input_root": str(input_root),
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "source_sha256": source_hashes,
        "checkpoint_sha256": dict(sorted(checkpoints.items())),
        "phase_t_integrity_review_sha256": s3.sha(phase_t_review_path),
        "splits": splits,
        "datasets": list(s3.DATASETS),
        "seeds": list(s3.SEEDS),
        "backbones": list(ARMS),
        "calibration_objective": "site-equal joint BF-XY NLL + 0.30 component-center transport risk",
        "graph_spec": {
            "builder": "build_doubly_stochastic_floor_graph",
            "nearest_neighbor_count": 8,
            "temperature_m": 8.0,
            "maximum_edge_distance_cutoff_m": None,
            "note": "Exact existing graph retained; edge-length diagnostics expose remote-neighbor risk rather than silently changing Structure 1.",
        },
        "selection_policy": S1_SELECTION_POLICY,
        "training_runs": 0,
        "optimizer_steps": 0,
        "retraining_performed": False,
        "evaluation_role": "CAL-eval",
        "data_access_note": (
            "Uses the existing reconstruct path and only indexes FIT, DEV, CAL-fit, and CAL-eval rows for "
            "features/scoring. TEST labels are never packed, predicted, or scored; TUJI only results_train is decoded."
        ),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_requested": args.device,
        "elapsed_seconds": time.perf_counter() - started,
        **TEST_COUNTS,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "EXECUTION_MANIFEST.json"
    if manifest_path.is_file():
        existing = s3.read_json(manifest_path)
        for field in ("status", "input_root", "checkpoint_root", "output_root", "device_requested", "source_sha256", "checkpoint_sha256", "phase_t_integrity_review_sha256", "splits"):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError(f"existing composition manifest differs in {field}; use a fresh output root")
        emit("preflight_reused", output=str(output_root), existing_runs=len(list((output_root / "runs").glob("*/*/RESULT.json"))) if (output_root / "runs").exists() else 0)
    else:
        non_manifest = [p for p in output_root.iterdir() if p.name != "EXECUTION_MANIFEST.json"]
        if non_manifest:
            raise RuntimeError(f"output root is nonempty but has no execution manifest: {output_root}")
        _json_write(manifest_path, manifest, exclusive=True)
    emit("preflight_complete", output=str(output_root), checkpoints=len(checkpoints), runs=len(splits) * len(ARMS),
         elapsed_seconds=manifest["elapsed_seconds"], **TEST_COUNTS)


def compose_parent_logits(pack, ctx, clean_params, *, s1_arm=None, s1_params=None):
    """Return clean parent logits or parent logits after the selected S1 channel operator."""
    if s1_arm is None:
        return s3.complete_logits(pack, clean_params)
    if s1_arm not in S1_METHODS or s1_params is None:
        raise ValueError("S1 composition requires 1A_global/1D_global and CAL-fit-selected parameters")
    alpha_prior = float(s1_params["alpha_prior"])
    alpha_obs = 0.0 if s1_arm == "1A_global" else float(s1_params["alpha_obs"])
    _, logits, _ = s3.s1_probability(
        pack, ctx, clean_params, alpha_prior, alpha_obs, "global"
    )
    return logits


def compose_cell_mixture(pack, ctx, clean_params, parent_logits, beta):
    """Expand the supplied physical parent logits with the existing S2 virtual field."""
    virtual_logits = op.combine_retrieval_and_g13_logits(
        pack["virtual_scores"],
        pack["virtual_z"],
        clean_params["temperature"],
        torch.as_tensor(clean_params["theta"], dtype=parent_logits.dtype, device=parent_logits.device),
    )
    return s3.fast_cell_mixture(
        parent_logits,
        virtual_logits,
        ctx["edges"],
        float(beta),
        pack["xy"],
        pack["sigma"],
        pack["virtual_xy"],
        pack["virtual_sigma"],
    )


def _grid_objective(table, beta, gamma):
    matches = [
        row for row in table
        if float(row["beta"]) == float(beta) and float(row["gamma_space"]) == float(gamma)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one CAL-fit grid row for beta={beta}, gamma={gamma}; got {len(matches)}")
    return float(matches[0]["objective"])


@torch.no_grad()
def _select_s2_and_combinations(ctx, model, fit_pack, clean, s1_selected):
    standalone_grid, combo_grids = [], {name: [] for name in S1_METHODS}
    for gamma in s3.GAMMAS:
        pack = fit_pack if float(gamma) == 0.0 else s3.make_pack(
            ctx, model, ctx["cal_fit"], virtual_gamma=float(gamma)
        )
        cache = s3.objective_cache(pack, ctx, virtual=True)
        parent_logits = {"base": compose_parent_logits(pack, ctx, clean)}
        for name in S1_METHODS:
            parent_logits[name] = compose_parent_logits(
                pack, ctx, clean, s1_arm=name, s1_params=s1_selected[name]
            )
        virtual_logits = op.combine_retrieval_and_g13_logits(
            pack["virtual_scores"],
            pack["virtual_z"],
            clean["temperature"],
            torch.as_tensor(clean["theta"], dtype=pack["scores"].dtype, device=pack["scores"].device),
        )
        for beta in s3.BETAS:
            for name, parent in parent_logits.items():
                cell = s3.fast_cell_mixture(
                    parent, virtual_logits, ctx["edges"], float(beta),
                    pack["xy"], pack["sigma"], pack["virtual_xy"], pack["virtual_sigma"],
                )
                objective = float(s3.objective_from_q(cell["q"], cache).detach().cpu())
                row = {
                    "beta": float(beta),
                    "gamma_space": float(gamma),
                    "objective": objective,
                }
                if name == "base":
                    standalone_grid.append(row)
                else:
                    combo_grids[name].append(row)
        if float(gamma) != 0.0:
            del pack, cache, parent_logits, virtual_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    best2 = min(standalone_grid, key=lambda row: (row["objective"], row["beta"], row["gamma_space"]))
    best0 = min(
        (row for row in standalone_grid if row["gamma_space"] == 0.0),
        key=lambda row: (row["objective"], row["beta"]),
    )
    combo = {}
    for name, table in combo_grids.items():
        best = min(table, key=lambda row: (row["objective"], row["beta"], row["gamma_space"]))
        best_gamma_zero = min(
            (row for row in table if row["gamma_space"] == 0.0),
            key=lambda row: (row["objective"], row["beta"]),
        )
        combo[name] = {
            "s1_method": name,
            "alpha_prior": float(s1_selected[name]["alpha_prior"]),
            "alpha_obs": float(s1_selected[name]["alpha_obs"]) if name == "1D_global" else 0.0,
            "2B": dict(best),
            "2A": {
                "beta": float(best["beta"]),
                "gamma_space": 0.0,
                "objective": _grid_objective(table, best["beta"], 0.0),
            },
            "2A_beta_only_control": dict(best_gamma_zero),
        }
    return {
        "structure2_grid": standalone_grid,
        "2B": dict(best2),
        "2A": {
            "beta": float(best2["beta"]),
            "gamma_space": 0.0,
            "objective": _grid_objective(standalone_grid, best2["beta"], 0.0),
        },
        "2A_beta_only_control": dict(best0),
        "combined": combo,
        "combined_grid": combo_grids,
        "selection_role": "CAL-fit",
        "selection_policy": S1_SELECTION_POLICY,
        "tie_break": "objective, then lower beta, then lower gamma_space",
    }


def _load_model(ctx, checkpoint_root, dataset, seed, arm, device):
    path = _checkpoint_path(checkpoint_root, dataset, seed, arm)
    payload = torch.load(path, map_location=device, weights_only=False)
    model = s3.enable_training_arm(ctx["initial"], arm, ctx["graph"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if arm == "E":
        native_alpha = float(model.raw_alpha.sigmoid().detach().cpu())
        if payload.get("alpha") is not None and abs(native_alpha - float(payload["alpha"])) > 1e-7:
            raise RuntimeError(f"E native alpha differs from checkpoint metadata: {path}")
    else:
        native_alpha = None
    return model, payload, native_alpha


@torch.no_grad()
def run_one(args, ctx, dataset, seed, arm, manifest):
    started = time.perf_counter()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root = Path(args.output).resolve()
    run_dir = output_root / "runs" / f"{dataset}_seed{seed}" / arm
    checkpoint = _checkpoint_path(checkpoint_root, dataset, seed, arm)
    relative_checkpoint = str(checkpoint.relative_to(checkpoint_root)).replace("\\", "/")
    checkpoint_sha = s3.sha(checkpoint)
    if manifest["checkpoint_sha256"].get(relative_checkpoint) != checkpoint_sha:
        raise RuntimeError(f"checkpoint changed after preflight: {relative_checkpoint}")
    if (run_dir / "RESULT.json").is_file():
        result = s3.read_json(run_dir / "RESULT.json")
        if result.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(f"completed result is bound to a different checkpoint: {run_dir}")
        emit("preserved_completed_composition_run", dataset=dataset, seed=seed, arm=arm)
        return
    model, payload, native_alpha = _load_model(ctx, checkpoint_root, dataset, seed, arm, args.device)
    selection_path = run_dir / "CAL_FIT_SELECTION.json"
    eval_lock_path = run_dir / "CAL_EVAL.lock"
    if run_dir.exists():
        if not selection_path.is_file() or not eval_lock_path.is_file():
            raise RuntimeError(
                f"incomplete run directory without frozen CAL-fit selection/lock: {run_dir}; preserve and inspect before repair"
            )
        selection = s3.read_json(selection_path)
        lock = s3.read_json(eval_lock_path)
        run_lock = s3.read_json(run_dir / "RUN.lock")
        if selection.get("selection_role") != "CAL-fit" or selection.get("cal_eval_not_used_for_selection") is not True:
            raise RuntimeError(f"resume selection is not a frozen CAL-fit selection: {selection_path}")
        if lock["selection_sha256"] != s3.sha(selection_path):
            raise RuntimeError(f"resume selection SHA mismatch: {eval_lock_path}")
        if lock["checkpoint_sha256"] != checkpoint_sha or run_lock["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError(f"resume checkpoint SHA mismatch: {run_dir}")
        if lock["device"] != args.device or run_lock["device"] != args.device:
            raise RuntimeError(f"resume device differs from frozen CAL-fit selection: {run_dir}")
        if lock["source_sha256"] != manifest["source_sha256"]:
            raise RuntimeError(f"resume source SHA mismatch: {eval_lock_path}")
        expected_fit_sha = s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_fit"])
        expected_eval_sha = s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"])
        if (lock["cal_fit_key_sha256"], run_lock["cal_fit_key_sha256"]) != (expected_fit_sha, expected_fit_sha):
            raise RuntimeError(f"resume CAL-fit key hash mismatch: {run_dir}")
        if (lock["cal_eval_key_sha256"], run_lock["cal_eval_key_sha256"]) != (expected_eval_sha, expected_eval_sha):
            raise RuntimeError(f"resume CAL-eval key hash mismatch: {run_dir}")
        if lock["cal_fit_rows"] != len(ctx["cal_fit"]) or run_lock["cal_fit_rows"] != len(ctx["cal_fit"]):
            raise RuntimeError(f"resume CAL-fit row count mismatch: {run_dir}")
        if lock["cal_eval_rows"] != len(ctx["cal_eval"]) or run_lock["cal_eval_rows"] != len(ctx["cal_eval"]):
            raise RuntimeError(f"resume CAL-eval row count mismatch: {run_dir}")
        if run_lock["source_sha256"] != manifest["source_sha256"]:
            raise RuntimeError(f"resume source SHA mismatch: {run_dir / 'RUN.lock'}")
        if any(lock.get(k) != 0 or run_lock.get(k) != 0 for k in TEST_COUNTS):
            raise RuntimeError(f"resume lock records nonzero TEST scoring: {run_dir}")
        emit("resume_locked_cal_eval", dataset=dataset, seed=seed, arm=arm,
             selection_sha256=lock["selection_sha256"])
    else:
        fit_pack = s3.make_pack(ctx, model, ctx["cal_fit"], virtual_gamma=0.0)
        clean, calibration_grid, calibration_objective = s3.grid_calibrate(fit_pack, ctx)
        s1_selected, s1_grids = s3.select_s1(fit_pack, ctx, clean)
        # Keep the requested global-normalized prior-only and dual-channel arms.
        s1_selected = {name: s1_selected[name] for name in S1_METHODS}
        s2 = _select_s2_and_combinations(ctx, model, fit_pack, clean, s1_selected)
        selection = {
            "dataset": dataset,
            "seed": seed,
            "backbone": arm,
            "checkpoint_sha256": checkpoint_sha,
            "selected_epoch": int(payload["selected_epoch"]),
            "joint_calibrated_base": clean,
            "calibration_objective": float(calibration_objective),
            "calibration_grid": calibration_grid,
            "structure1": s1_selected,
            "structure1_grids": {name: s1_grids[name] for name in S1_METHODS},
            **s2,
            "selection_role": "CAL-fit",
            "evaluation_role": "CAL-eval",
            "cal_eval_not_used_for_selection": True,
            "training_runs": 0,
            "optimizer_steps": 0,
            "retraining_performed": False,
            **TEST_COUNTS,
        }
        run_dir.mkdir(parents=True, exist_ok=False)
        s3.write_json(run_dir / "RUN.lock", {
            "dataset": dataset,
            "seed": seed,
            "backbone": arm,
            "checkpoint_relpath": relative_checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "selected_epoch": int(payload["selected_epoch"]),
            "device": args.device,
            "source_sha256": manifest["source_sha256"],
            "cal_fit_key_sha256": s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_fit"]),
            "cal_eval_key_sha256": s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"]),
            "cal_fit_rows": int(len(ctx["cal_fit"])),
            "cal_eval_rows": int(len(ctx["cal_eval"])),
            "graph_edge_diagnostics": manifest["splits"][f"{dataset}_seed{seed}"]["graph_edge_diagnostics"],
            "retraining_performed": False,
            "optimizer_steps": 0,
            **TEST_COUNTS,
        })
        s3.write_json(selection_path, selection)
        s3.write_json(eval_lock_path, {
            "selection_sha256": s3.sha(selection_path),
            "checkpoint_sha256": checkpoint_sha,
            "device": args.device,
            "source_sha256": manifest["source_sha256"],
            "cal_fit_key_sha256": s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_fit"]),
            "cal_eval_key_sha256": s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"]),
            "cal_fit_rows": int(len(ctx["cal_fit"])),
            "cal_eval_rows": int(len(ctx["cal_eval"])),
            "selection_role": "CAL-fit",
            "evaluation_role": "CAL-eval",
            "selection_frozen_before_cal_eval": True,
            **TEST_COUNTS,
        })
        del fit_pack
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    clean = selection["joint_calibrated_base"]
    s1_selected = selection["structure1"]
    s2_best = selection["2B"]
    s2_control = selection["2A_beta_only_control"]
    combo = selection["combined"]
    results, diagnostics = {}, {}

    def record(name, pack, q, params, *, centers=None, scales=None, ids=None, extra=None):
        metrics, sites = s3.score_mixture(
            pack, ctx, q, centers=centers, scales=scales, ids=ids, seed=seed + 900
        )
        results[name] = {
            "params": s3._json_safe(params),
            "metrics": s3._json_safe(metrics),
            **(extra or {}),
        }
        diagnostics[name] = sites

    def record_cell(name, pack, parent_logits, beta, params, group, s1_method=None):
        cell = compose_cell_mixture(pack, ctx, clean, parent_logits, beta)
        extra = {
            "group": group,
            "composition_order": (
                "S1 diffuses physical parent prior/observation channels, then S2 Cell LSE"
                if s1_method else "S2 virtual field and Cell LSE on clean parent logits"
            ),
        }
        if s1_method:
            extra["s1_method"] = s1_method
        record(
            name, pack, cell["q"], params,
            centers=cell["xy"], scales=cell["sigma"], ids=cell["ids"], extra=extra,
        )

    evaluation0 = s3.make_pack(ctx, model, ctx["cal_eval"], virtual_gamma=0.0)
    clean_parent = compose_parent_logits(evaluation0, ctx, clean)
    record(
        "joint_calibrated_base", evaluation0, torch.softmax(clean_parent, dim=1), clean,
        extra={"group": "S3", "composition_order": "clean joint calibration"},
    )
    eval_s1_logits = {}
    for name in S1_METHODS:
        params = s1_selected[name]
        parent = compose_parent_logits(evaluation0, ctx, clean, s1_arm=name, s1_params=params)
        eval_s1_logits[name] = parent
        _, _, q = s3.s1_probability(
            evaluation0, ctx, clean, float(params["alpha_prior"]),
            0.0 if name == "1A_global" else float(params["alpha_obs"]), "global",
        )
        record(name, evaluation0, q, params, extra={
            "group": "S3+S1",
            "composition_order": "post-hoc dual-channel S1 on physical parent logits",
            "native_checkpoint_alpha": native_alpha,
        })
    record_cell(
        "2A", evaluation0, clean_parent, float(selection["2A"]["beta"]),
        {**clean, **selection["2A"], "center_mode": "residual_interpolated"},
        "S3+S2",
    )
    record_cell(
        "2A_beta_only_control", evaluation0, clean_parent, float(s2_control["beta"]),
        {**clean, **s2_control, "center_mode": "residual_interpolated"},
        "S3+S2",
    )
    for name in S1_METHODS:
        arm_combo = combo[name]
        record_cell(
            f"combo_{name[:2]}_2A", evaluation0, eval_s1_logits[name],
            float(arm_combo["2A"]["beta"]),
            {
                **clean,
                **arm_combo["2A"],
                "s1_method": name,
                "alpha_prior": arm_combo["alpha_prior"],
                "alpha_obs": arm_combo["alpha_obs"],
                "center_mode": "residual_interpolated",
                "selection_policy": S1_SELECTION_POLICY,
            },
            "S3+S1+S2",
            s1_method=name,
        )
        record_cell(
            f"combo_{name[:2]}_2A_beta_only_control", evaluation0, eval_s1_logits[name],
            float(arm_combo["2A_beta_only_control"]["beta"]),
            {
                **clean,
                **arm_combo["2A_beta_only_control"],
                "s1_method": name,
                "alpha_prior": arm_combo["alpha_prior"],
                "alpha_obs": arm_combo["alpha_obs"],
                "center_mode": "residual_interpolated",
                "selection_policy": S1_SELECTION_POLICY,
            },
            "S3+S1+S2",
            s1_method=name,
        )
    if float(s2_best["gamma_space"]) == 0.0:
        record_cell(
            "2B", evaluation0, clean_parent, float(s2_best["beta"]),
            {**clean, **s2_best, "center_mode": "residual_interpolated"},
            "S3+S2",
        )
    for name in S1_METHODS:
        chosen = combo[name]["2B"]
        if float(chosen["gamma_space"]) == 0.0:
            record_cell(
                f"combo_{name[:2]}_2B", evaluation0, eval_s1_logits[name],
                float(chosen["beta"]),
                {
                    **clean, **chosen, "s1_method": name,
                    "alpha_prior": combo[name]["alpha_prior"],
                    "alpha_obs": combo[name]["alpha_obs"],
                    "center_mode": "residual_interpolated",
                    "selection_policy": S1_SELECTION_POLICY,
                },
                "S3+S1+S2",
                s1_method=name,
            )
    del evaluation0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    needed_gammas = set()
    if float(s2_best["gamma_space"]) > 0.0:
        needed_gammas.add(float(s2_best["gamma_space"]))
    for name in S1_METHODS:
        gamma = float(combo[name]["2B"]["gamma_space"])
        if gamma > 0.0:
            needed_gammas.add(gamma)
    for gamma in sorted(needed_gammas):
        evaluation = s3.make_pack(ctx, model, ctx["cal_eval"], virtual_gamma=gamma)
        parent_clean = compose_parent_logits(evaluation, ctx, clean)
        if float(s2_best["gamma_space"]) == gamma:
            record_cell(
                "2B", evaluation, parent_clean, float(s2_best["beta"]),
                {**clean, **s2_best, "center_mode": "residual_interpolated"},
                "S3+S2",
            )
        for name in S1_METHODS:
            chosen = combo[name]["2B"]
            if float(chosen["gamma_space"]) != gamma:
                continue
            parent = compose_parent_logits(
                evaluation, ctx, clean, s1_arm=name, s1_params=s1_selected[name]
            )
            record_cell(
                f"combo_{name[:2]}_2B", evaluation, parent, float(chosen["beta"]),
                {
                    **clean, **chosen, "s1_method": name,
                    "alpha_prior": combo[name]["alpha_prior"],
                    "alpha_obs": combo[name]["alpha_obs"],
                    "center_mode": "residual_interpolated",
                    "selection_policy": S1_SELECTION_POLICY,
                },
                "S3+S1+S2",
                s1_method=name,
            )
        del evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if set(results) != set(METHOD_NAMES):
        raise RuntimeError(f"composition method set incomplete: missing={sorted(set(METHOD_NAMES)-set(results))}; extra={sorted(set(results)-set(METHOD_NAMES))}")
    result = {
        "phase": "COMPOSITION",
        "dataset": dataset,
        "seed": seed,
        "backbone": arm,
        "selected_epoch": int(payload["selected_epoch"]),
        "checkpoint_relpath": relative_checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "device": args.device,
        "selection_sha256": s3.sha(selection_path),
        "cal_fit_key_sha256": s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_fit"]),
        "cal_eval_key_sha256": s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"]),
        "cal_fit_rows": int(len(ctx["cal_fit"])),
        "cal_eval_rows": int(len(ctx["cal_eval"])),
        "methods": results,
        "physical_anchors": int(len(ctx["memory"]["coords"])),
        "unique_virtual_edges": int(len(ctx["edges"])),
        "graph_edge_diagnostics": manifest["splits"][f"{dataset}_seed{seed}"]["graph_edge_diagnostics"],
        "native_checkpoint_alpha": native_alpha,
        "evaluation_role": "CAL-eval",
        "selection_role": "CAL-fit",
        "cal_eval_not_used_for_selection": True,
        "retraining_performed": False,
        "training_runs": 0,
        "optimizer_steps": 0,
        "elapsed_seconds": time.perf_counter() - started,
        **TEST_COUNTS,
    }
    _json_write(run_dir / "SITE_DIAGNOSTICS.json", diagnostics)
    _json_write(run_dir / "RESULT.json", result)
    emit("composition_run_complete", dataset=dataset, seed=seed, arm=arm,
         methods=len(results), elapsed_seconds=result["elapsed_seconds"], **TEST_COUNTS)
    del model, payload
    gc.collect()


def run(args):
    output_root = Path(args.output).resolve()
    manifest_path = output_root / "EXECUTION_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run requires a successful preflight; EXECUTION_MANIFEST.json is missing")
    manifest = s3.read_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise RuntimeError("composition preflight manifest is not PASS")
    if Path(args.input_root).resolve() != Path(manifest["input_root"]):
        raise RuntimeError("input-root differs from the preflight-locked source root")
    if Path(args.checkpoint_root).resolve() != Path(manifest["checkpoint_root"]):
        raise RuntimeError("checkpoint-root differs from the preflight-locked checkpoint root")
    if output_root != Path(manifest["output_root"]):
        raise RuntimeError("output root differs from the preflight-locked output root")
    if args.device != manifest["device_requested"]:
        raise RuntimeError("runtime device differs from preflight; use a fresh output root and rerun preflight")
    _verify_source_manifest(manifest)
    for relative, expected in manifest["checkpoint_sha256"].items():
        if s3.sha(Path(args.checkpoint_root).resolve() / relative) != expected:
            raise RuntimeError(f"checkpoint changed after preflight: {relative}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; rerun explicitly with --device cpu")
    torch.set_num_threads(4)
    for dataset in s3.DATASETS:
        for seed in s3.SEEDS:
            ctx = s3.reconstruct(dataset, seed, args.input_root, torch.device(args.device))
            for arm in ARMS:
                run_one(args, ctx, dataset, seed, arm, manifest)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del ctx
            gc.collect()
    audit_results(output_root, require_complete=True, write=False)


def _comparisons(methods):
    pairs = {("joint_calibrated_base", name) for name in methods if name != "joint_calibrated_base"}
    pairs.update({
        ("2A", "2B"),
        ("2A", "2A_beta_only_control"),
        ("1A_global", "combo_1A_2A"),
        ("1A_global", "combo_1A_2B"),
        ("1A_global", "combo_1A_2A_beta_only_control"),
        ("1D_global", "combo_1D_2A"),
        ("1D_global", "combo_1D_2B"),
        ("1D_global", "combo_1D_2A_beta_only_control"),
        ("2A", "combo_1A_2A"),
        ("2B", "combo_1A_2B"),
        ("2A", "combo_1D_2A"),
        ("2B", "combo_1D_2B"),
        ("combo_1A_2A", "combo_1A_2B"),
        ("combo_1D_2A", "combo_1D_2B"),
        ("combo_1A_2A", "combo_1A_2A_beta_only_control"),
        ("combo_1D_2A", "combo_1D_2A_beta_only_control"),
    })
    return sorted(pairs)


def audit_results(root, *, require_complete=True, write=False):
    root = Path(root)
    paths = sorted((root / "runs").glob("*/*/RESULT.json"))
    expected = len(s3.DATASETS) * len(s3.SEEDS) * len(ARMS)
    if require_complete and len(paths) != expected:
        raise RuntimeError(f"composition study incomplete: {len(paths)}/{expected} run results")
    max_reduction_error = 0.0
    expected_method_set = set(METHOD_NAMES)
    expected_identities = {
        (dataset, seed, arm)
        for dataset in s3.DATASETS for seed in s3.SEEDS for arm in ARMS
    }
    seen_identities = set()
    manifest_path = root / "EXECUTION_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"composition execution manifest missing: {manifest_path}")
    manifest = s3.read_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise RuntimeError("composition execution manifest is not PASS")
    _verify_source_manifest(manifest)
    if len(manifest.get("checkpoint_sha256", {})) != expected:
        raise RuntimeError("preflight manifest does not bind all 27 frozen checkpoints")
    for path in paths:
        result = s3.read_json(path)
        diagnostics = s3.read_json(path.with_name("SITE_DIAGNOSTICS.json"))
        lock = s3.read_json(path.with_name("CAL_EVAL.lock"))
        selection = s3.read_json(path.with_name("CAL_FIT_SELECTION.json"))
        run_lock = s3.read_json(path.with_name("RUN.lock"))
        identity = (result["dataset"], int(result["seed"]), result["backbone"])
        if identity not in expected_identities or identity in seen_identities:
            raise RuntimeError(f"unexpected or duplicate result identity: {identity} in {path}")
        seen_identities.add(identity)
        run_key = f"{result['dataset']}_seed{result['seed']}"
        if path.parent.name != result["backbone"] or path.parent.parent.name != run_key:
            raise RuntimeError(f"result identity disagrees with its run directory: {path}")
        if run_key not in manifest["splits"]:
            raise RuntimeError(f"run split is absent from preflight manifest: {run_key}")
        split = manifest["splits"][run_key]
        expected_split = (
            split["cal_fit_key_sha256"], split["cal_eval_key_sha256"],
            split["cal_fit_rows"], split["cal_eval_rows"],
            split["physical_anchors"], split["unique_midpoint_edges"],
        )
        actual_split = (
            result["cal_fit_key_sha256"], result["cal_eval_key_sha256"],
            result["cal_fit_rows"], result["cal_eval_rows"],
            result["physical_anchors"], result["unique_virtual_edges"],
        )
        if result.get("graph_edge_diagnostics") != split["graph_edge_diagnostics"]:
            raise RuntimeError(f"graph edge diagnostics differ from preflight manifest: {path}")
        if run_lock.get("graph_edge_diagnostics") != split["graph_edge_diagnostics"]:
            raise RuntimeError(f"RUN.lock graph diagnostics differ from preflight manifest: {path}")
        if actual_split != expected_split:
            raise RuntimeError(f"run split/anchor/edge identity differs from preflight manifest: {path}")
        assert result["evaluation_role"] == "CAL-eval" and result["selection_role"] == "CAL-fit"
        assert selection["selection_role"] == "CAL-fit" and selection["cal_eval_not_used_for_selection"] is True
        assert (selection["dataset"], selection["seed"], selection["backbone"]) == (
            result["dataset"], result["seed"], result["backbone"]
        )
        assert lock["selection_sha256"] == s3.sha(path.with_name("CAL_FIT_SELECTION.json"))
        assert result["selection_sha256"] == lock["selection_sha256"]
        assert result["checkpoint_sha256"] == lock["checkpoint_sha256"] == selection["checkpoint_sha256"]
        assert lock["selection_frozen_before_cal_eval"] is True
        assert lock["source_sha256"] == manifest["source_sha256"]
        assert lock["cal_fit_key_sha256"] == result["cal_fit_key_sha256"]
        assert lock["cal_eval_key_sha256"] == result["cal_eval_key_sha256"]
        assert lock["cal_fit_rows"] == result["cal_fit_rows"]
        assert lock["cal_eval_rows"] == result["cal_eval_rows"]
        assert run_lock["cal_fit_rows"] == result["cal_fit_rows"]
        assert run_lock["cal_eval_rows"] == result["cal_eval_rows"]
        assert run_lock["checkpoint_sha256"] == result["checkpoint_sha256"]
        assert run_lock["device"] == result["device"] == manifest["device_requested"]
        assert lock["device"] == result["device"]
        assert run_lock["selected_epoch"] == result["selected_epoch"]
        assert run_lock["cal_fit_key_sha256"] == result["cal_fit_key_sha256"]
        assert run_lock["cal_eval_key_sha256"] == result["cal_eval_key_sha256"]
        assert run_lock["source_sha256"] == manifest["source_sha256"]
        assert all(result.get(key) == 0 and lock.get(key) == 0 and run_lock.get(key) == 0 for key in TEST_COUNTS)
        relative_checkpoint = result["checkpoint_relpath"]
        if manifest["checkpoint_sha256"].get(relative_checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError(f"result checkpoint SHA differs from preflight manifest: {path}")
        actual_checkpoint = Path(manifest["checkpoint_root"]) / relative_checkpoint
        if not actual_checkpoint.is_file() or s3.sha(actual_checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError(f"frozen checkpoint changed after scoring: {actual_checkpoint}")
        assert result["retraining_performed"] is False and result["optimizer_steps"] == 0
        assert set(result["methods"]) == expected_method_set and set(diagnostics) == expected_method_set
        for name, entry in result["methods"].items():
            sites = list(diagnostics[name].values())
            assert sites and sum(int(site["n_scans"]) for site in sites) == entry["metrics"]["n_scans"]
            checks = (
                ("macro_mae_full_m", "mae_full_m"),
                ("macro_mae_predicted_bf_m", "mae_predicted_bf_m"),
                ("macro_es_m", "energy_score_m"),
                ("macro_joint_nll", "joint_nll"),
                ("macro_bf_accuracy", "bf_accuracy"),
                ("macro_transport_risk_m", "transport_risk_m"),
            )
            for metric, field in checks:
                error = abs(float(np.mean([site[field] for site in sites])) - float(entry["metrics"][metric]))
                if error >= 2e-5:
                    raise RuntimeError(f"site reduction mismatch: {path}/{name}/{metric}: {error}")
                max_reduction_error = max(max_reduction_error, error)
            site_mae = np.asarray([site["mae_full_m"] for site in sites], dtype=float)
            for metric, expected_value in (
                ("site_p95_full_m", float(np.quantile(site_mae, .95))),
                ("site_p99_full_m", float(np.quantile(site_mae, .99))),
                ("worst_site_mae_m", float(site_mae.max())),
            ):
                error = abs(float(entry["metrics"][metric]) - expected_value)
                if error >= 2e-5:
                    raise RuntimeError(f"tail metric mismatch: {path}/{name}/{metric}: {error}")
    if require_complete and seen_identities != expected_identities:
        raise RuntimeError(f"composition run identities incomplete: missing={sorted(expected_identities-seen_identities)}")
    record = {
        "status": "PASS",
        "study": "V17 zero-retraining composition",
        "completed_runs": len(paths),
        "expected_runs": expected,
        "methods_per_run": len(METHOD_NAMES),
        "maximum_site_reduction_error": max_reduction_error,
        "training_runs": 0,
        "optimizer_steps": 0,
        "retraining_performed": False,
        "evaluation_role": "CAL-eval",
        "selection_role": "CAL-fit",
        "source_sha256": manifest["source_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "phase_t_integrity_review_sha256": manifest["phase_t_integrity_review_sha256"],
        "splits": manifest["splits"],
        **TEST_COUNTS,
    }
    if write:
        _json_write(root / "INTEGRITY_REVIEW.json", record)
    return record


def aggregate(args):
    root = Path(args.output).resolve()
    audit = audit_results(root, require_complete=True, write=False)
    run_rows, paired_rows, site_rows = [], [], []
    grouped = {}
    for path in sorted((root / "runs").glob("*/*/RESULT.json")):
        result = s3.read_json(path)
        diagnostics = s3.read_json(path.with_name("SITE_DIAGNOSTICS.json"))
        for method, entry in result["methods"].items():
            row = {
                "phase": "COMPOSITION",
                "dataset": result["dataset"],
                "seed": result["seed"],
                "backbone": result["backbone"],
                "method": method,
                "selected_epoch": result["selected_epoch"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "params_json": json.dumps(entry["params"], sort_keys=True),
                **entry["metrics"],
            }
            run_rows.append(row)
            grouped.setdefault((result["dataset"], result["backbone"], method), []).append(row)
        for reference, candidate in _comparisons(result["methods"]):
            base = result["methods"][reference]["metrics"]
            cand = result["methods"][candidate]["metrics"]
            ref_sites, cand_sites = diagnostics[reference], diagnostics[candidate]
            if set(ref_sites) != set(cand_sites):
                raise RuntimeError(f"site-key mismatch: {path}/{reference}/{candidate}")
            site_delta = []
            for key in sorted(ref_sites):
                ref, value = ref_sites[key], cand_sites[key]
                delta = {
                    "site_key": key,
                    "site": json.dumps(ref["site"], separators=(",", ":")),
                    "n_scans": int(ref["n_scans"]),
                    "mae_delta_m": float(value["mae_full_m"] - ref["mae_full_m"]),
                    "es_delta_m": float(value["energy_score_m"] - ref["energy_score_m"]),
                    "joint_nll_delta": float(value["joint_nll"] - ref["joint_nll"]),
                    "bf_accuracy_delta": float(value["bf_accuracy"] - ref["bf_accuracy"]),
                }
                site_delta.append(delta)
                site_rows.append({
                    "dataset": result["dataset"], "seed": result["seed"], "backbone": result["backbone"],
                    "reference": reference, "candidate": candidate, **delta,
                })
            mae_values = np.asarray([row["mae_delta_m"] for row in site_delta], dtype=float)
            paired_rows.append({
                "dataset": result["dataset"], "seed": result["seed"], "backbone": result["backbone"],
                "reference": reference, "candidate": candidate,
                "mae_delta_m": float(cand["macro_mae_full_m"] - base["macro_mae_full_m"]),
                "es_delta_m": float(cand["macro_es_m"] - base["macro_es_m"]),
                "joint_nll_delta": float(cand["macro_joint_nll"] - base["macro_joint_nll"]),
                "bf_accuracy_delta": float(cand["macro_bf_accuracy"] - base["macro_bf_accuracy"]),
                "p95_delta_m": float(cand["site_p95_full_m"] - base["site_p95_full_m"]),
                "p99_delta_m": float(cand["site_p99_full_m"] - base["site_p99_full_m"]),
                "worst_site_delta_m": float(cand["worst_site_mae_m"] - base["worst_site_mae_m"]),
                "improved_sites": int((mae_values < -1e-7).sum()),
                "harmed_sites": int((mae_values > 1e-7).sum()),
                "max_single_site_harm_m": float(mae_values.max()),
                "n_sites": len(site_delta),
            })
    summary_rows = []
    for (dataset, backbone, method), rows in sorted(grouped.items()):
        if len(rows) != len(s3.SEEDS):
            raise RuntimeError(f"incomplete seed group: {dataset}/{backbone}/{method}")
        row = {"dataset": dataset, "backbone": backbone, "method": method, "n_seeds": len(rows)}
        for metric in METRICS:
            values = [float(item[metric]) for item in rows]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_sd"] = statistics.stdev(values)
        summary_rows.append(row)
    _csv_write(root / "RUN_LEVEL_METRICS.csv", run_rows)
    _csv_write(root / "AGGREGATE_SUMMARY.csv", summary_rows)
    _csv_write(root / "PAIRED_DELTAS.csv", paired_rows)
    _csv_write(root / "SITE_DELTAS_ALL_RUNS.csv", site_rows)
    _write_report(root, summary_rows, paired_rows, audit)
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "INTEGRITY_REVIEW.json":
            hashes[str(path.relative_to(root)).replace("\\", "/")] = s3.sha(path)
    audit.update({
        "run_metric_rows": len(run_rows),
        "aggregate_summary_rows": len(summary_rows),
        "paired_comparisons": len(paired_rows),
        "paired_site_rows": len(site_rows),
        "artifact_sha256": hashes,
    })
    _json_write(root / "INTEGRITY_REVIEW.json", audit)
    emit("composition_aggregate_complete", run_metric_rows=len(run_rows),
         summary_rows=len(summary_rows), paired_comparisons=len(paired_rows),
         paired_site_rows=len(site_rows), **TEST_COUNTS)


def _write_report(root, summaries, paired, audit):
    lines = [
        "# V17 Spatial Composition — Zero-Retraining Results",
        "",
        "This is a retrospective CAL-eval comparison of frozen C/D/E checkpoints. It is not an independent blind test. "
        "No model retraining or optimizer steps were performed; internal and official TEST score counts are zero.",
        "",
        "## Audit and protocol",
        "",
        f"- Integrity status: {audit['status']}; completed runs: {audit['completed_runs']}/{audit['expected_runs']}.",
        "- CAL-fit selects clean T/theta, S1 alpha, standalone S2 beta/gamma, and combo beta/gamma conditional on frozen S1 alpha.",
        "- CAL-eval is used only after CAL_FIT_SELECTION.json and CAL_EVAL.lock are written.",
        "- For E checkpoints, the saved training-time graph layer remains active; S1 is an additional post-hoc graph operation.",
        "- Combination order: dual-channel S1 on physical parent logits, then S2 Cell LSE with normal interpolated-key/prototype virtual retrieval scores and G13 evidence.",
        "- All methods use the same physical CAL-eval rows and the same seed+900 Energy Score Monte Carlo seed within each run.",
        "- Exact per-run selected values are printed below; the corresponding full calibration grids and hashes are preserved in CAL_FIT_SELECTION.json.",
        "",
        "## Frozen graph edge diagnostics",
        "",
        "The graph is unchanged from the prior Structure 1 study: same-building/same-floor k=8 union graph, Sinkhorn-scaled, with no maximum-distance cutoff. These diagnostics make remote-neighbor risk visible rather than changing the operator mid-comparison.",
        "",
        "|Dataset|Seed|Physical anchors|Unique edges|Edge P50 m|Edge P95 m|Max edge m|Edges >30 m|Fraction >30 m|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    manifest = s3.read_json(Path(root) / "EXECUTION_MANIFEST.json")
    for run_key, split in sorted(manifest["splits"].items()):
        diag = split["graph_edge_diagnostics"]
        lines.append(
            f"|{run_key.rsplit('_seed', 1)[0]}|{run_key.rsplit('_seed', 1)[1]}|"
            f"{split['physical_anchors']}|{split['unique_midpoint_edges']}|"
            f"{diag['length_p50_m']:.3f}|{diag['length_p95_m']:.3f}|{diag['length_max_m']:.3f}|"
            f"{diag['count_gt_30m']}|{diag['fraction_gt_30m']:.4f}|"
        )
    lines.extend([
        "",
        "## Selected parameters (one row per frozen checkpoint)",
        "",
        "|Dataset|Seed|Backbone|T|theta visibility/strength|1A alpha prior/obs|1D alpha prior/obs|S2 2B beta/gamma|S1+S2 1A beta/gamma|S1+S2 1D beta/gamma|",
        "|---|---:|---|---:|---|---|---|---|---|---|",
    ])
    selection_paths = sorted((Path(root) / "runs").glob("*/*/CAL_FIT_SELECTION.json"))
    if len(selection_paths) != audit["expected_runs"]:
        raise RuntimeError(f"parameter report requires all {audit['expected_runs']} frozen selections")
    for selection_path in selection_paths:
        selected = s3.read_json(selection_path)
        s1a = selected["structure1"]["1A_global"]
        s1d = selected["structure1"]["1D_global"]
        c1a = selected["combined"]["1A_global"]["2B"]
        c1d = selected["combined"]["1D_global"]["2B"]
        theta = selected["joint_calibrated_base"]["theta"]
        lines.append(
            f"|{selected['dataset']}|{selected['seed']}|{selected['backbone']}|"
            f"{selected['joint_calibrated_base']['temperature']:.5f}|"
            f"{theta[0]:.4f}/{theta[1]:.4f}|"
            f"{s1a['alpha_prior']:.3f}/{s1a['alpha_obs']:.3f}|"
            f"{s1d['alpha_prior']:.3f}/{s1d['alpha_obs']:.3f}|"
            f"{selected['2B']['beta']:.3f}/{selected['2B']['gamma_space']:.3f}|"
            f"{c1a['beta']:.3f}/{c1a['gamma_space']:.3f}|"
            f"{c1d['beta']:.3f}/{c1d['gamma_space']:.3f}|"
        )
    lines.extend([
        "",
        "## Metrics by dataset, backbone, and method",
        "",
        "|Dataset|Backbone|Method|MAE m|ES m|Joint NLL|BF accuracy|P95 m|P99 m|Worst site m|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summaries:
        fmt = lambda name: f"{row[name + '_mean']:.5f} ± {row[name + '_sd']:.5f}"
        lines.append(
            f"|{row['dataset']}|{row['backbone']}|{row['method']}|"
            + "|".join(fmt(name) for name in (
                "macro_mae_full_m", "macro_es_m", "macro_joint_nll", "macro_bf_accuracy",
                "site_p95_full_m", "site_p99_full_m", "worst_site_mae_m",
            ))
            + "|"
        )
    lines.extend([
        "",
        "## Paired deltas and tail risk",
        "",
        "Negative MAE/ES/NLL deltas indicate improvement; positive BF-accuracy deltas indicate improvement. "
        "PAIRED_DELTAS.csv retains each seed and SITE_DELTAS_ALL_RUNS.csv retains each physical site.",
        "",
        "|Dataset|Backbone|Reference → candidate|ΔMAE m|ΔES m|ΔNLL|Max single-site MAE harm m|",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    groups = {}
    for row in paired:
        groups.setdefault((row["dataset"], row["backbone"], row["reference"], row["candidate"]), []).append(row)
    for key, rows in sorted(groups.items()):
        mean_sd = lambda field: (
            f"{statistics.mean(float(r[field]) for r in rows):+.5f} ± "
            f"{statistics.stdev(float(r[field]) for r in rows):.5f}"
        )
        lines.append(
            f"|{key[0]}|{key[1]}|{key[2]} → {key[3]}|{mean_sd('mae_delta_m')}|"
            f"{mean_sd('es_delta_m')}|{mean_sd('joint_nll_delta')}|"
            f"{max(float(r['max_single_site_harm_m']) for r in rows):.5f}|"
        )
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- Means and sample standard deviations summarize three seeds on the same previously observed CAL-eval site panel; they are not independent-site confidence intervals.",
        "- A lower NLL/ES does not guarantee a lower coordinate MAE. Read all metrics and per-site harms together.",
        "- S1 parameters are selected on the S1-only CAL-fit objective and frozen before conditional S2 selection in the combination. This is a progressive composition test, not a joint alpha/beta/gamma optimum.",
        "- The virtual candidate is generated by the existing interpolated-key and FIT RSS-prototype retrieval law. It is not assigned the average of endpoint complete logits.",
        "- The run does not prove generalization to a new geography. Any winning combination needs an independent location panel before a broad superiority claim.",
        "",
    ])
    (Path(root) / "RUN_REPORT_V17_SPATIAL_COMPOSITION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "all"):
        p = sub.add_parser(name)
        p.add_argument("--input-root", default=str(DEFAULT_INPUT))
        p.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
        p.add_argument("--output", default=str(DEFAULT_OUTPUT))
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p = sub.add_parser("aggregate")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "run":
        run(args)
    elif args.command == "aggregate":
        aggregate(args)
    elif args.command == "all":
        preflight(args)
        run(args)
        aggregate(args)


if __name__ == "__main__":
    main()
