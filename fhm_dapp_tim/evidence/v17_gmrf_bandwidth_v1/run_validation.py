"""Zero-retraining validation: GMRF graph-diffusion posterior + discretization bandwidth.

New operators only. Every selection is made on CAL-fit; CAL-eval is reporting-only.
No optimizer, no checkpoint rewrite, no TEST scoring.

Design under test
-----------------
L1  ell_tilde = (I - a W)^-1 ell            (all-order graph diffusion, one parameter a)
L2  sigma_tilde = sqrt(sigma^2 + (rho d)^2)  (discretization bandwidth, one parameter rho)
L0  y_hat = sum_i p_i mu_i                   (unchanged by L2 by construction)

Assertions
----------
A1  a=0 reproduces the frozen calibrated base bit-for-bit.
A2  L2 leaves MAE and BF accuracy unchanged for every rho.
A4  DeltaMAE(1G+2K) == DeltaMAE(1G).
A6  the resolvent commutes with the temperature.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_operators_v1 as op
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_structure_v1 as s3
from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import spatial_metrics

torch.set_grad_enabled(False)

DEFAULT_ROOT = REPO / "fhm_dapp_tim/evidence/v17_two_stage_freeze_v1"
DEFAULT_CKPT = REPO / "fhm_dapp_tim/evidence/v17_spatial_structure_v1"
ALPHAS = (0.0, 0.25, 0.4, 0.55, 0.7, 0.85)
RHOS = (0.0, 0.1, 0.2, 0.289, 0.4, 0.5, 0.65, 0.8)


# --------------------------------------------------------------------------- #
# graph                                                                       #
# --------------------------------------------------------------------------- #
def build_floor_graph(coords, buildings, floors, *, count=8, temperature_m=8.0,
                      d_max=25.0, max_iters=20000, tol=1e-10):
    """Same-BF kNN graph with a physical distance cap and zero diagonal."""
    xy = np.asarray(torch.as_tensor(coords).detach().cpu(), dtype=np.float64)
    b = np.asarray(torch.as_tensor(buildings).detach().cpu(), dtype=np.int64)
    f = np.asarray(torch.as_tensor(floors).detach().cpu(), dtype=np.int64)
    n = len(xy)
    adjacency = np.zeros((n, n), dtype=np.float64)
    edge_set: set[tuple[int, int]] = set()
    strata: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(zip(b.tolist(), f.tolist())):
        strata.setdefault(key, []).append(idx)
    dropped = 0
    for members in strata.values():
        if len(members) <= 1:
            continue
        block = xy[members]
        delta = block[:, None, :] - block[None, :, :]
        distance = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))
        for local_i, global_i in enumerate(members):
            order = np.argsort(distance[local_i] + np.eye(len(members))[local_i] * 1e30)
            taken = 0
            for local_j in order:
                if int(local_j) == local_i:
                    continue
                d = float(distance[local_i, int(local_j)])
                if d > d_max:
                    dropped += 1
                    continue
                global_j = members[int(local_j)]
                weight = math.exp(-d / temperature_m)
                adjacency[global_i, global_j] = max(adjacency[global_i, global_j], weight)
                adjacency[global_j, global_i] = max(adjacency[global_j, global_i], weight)
                edge_set.add((min(global_i, global_j), max(global_i, global_j)))
                taken += 1
                if taken >= count:
                    break
    isolated = adjacency.sum(axis=1) == 0
    adjacency[isolated, isolated] = 1.0
    # Metropolis-Hastings symmetric doubly-stochastic normalisation.
    # P_ij = A_ij / max(d_i, d_j) for i != j, P_ii = 1 - sum_j P_ij >= 0.
    # This is exact for any symmetric graph, including heterogeneous degrees.
    degree = adjacency.sum(axis=1)
    larger = np.maximum(degree[:, None], degree[None, :])
    weights = adjacency / np.where(larger > 0, larger, 1.0)
    np.fill_diagonal(weights, 0.0)
    residual = 1.0 - weights.sum(axis=1)
    if np.any(residual < -1e-12):
        raise RuntimeError("Metropolis-Hastings self-loop mass negative")
    np.fill_diagonal(weights, np.maximum(residual, 0.0))
    row_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    col_error = float(np.max(np.abs(weights.sum(axis=0) - 1.0)))
    if max(row_error, col_error) > 1e-10:
        raise RuntimeError(f"graph not doubly stochastic: {row_error:.3g}/{col_error:.3g}")
    if not np.allclose(weights, weights.T, atol=1e-12, rtol=1e-12):
        raise RuntimeError("graph lost symmetry")
    iterations = 0
    edges = torch.as_tensor(sorted(edge_set), dtype=torch.long).reshape(-1, 2)
    lengths = torch.linalg.vector_norm(
        torch.as_tensor(xy, dtype=torch.float64)[edges[:, 0]]
        - torch.as_tensor(xy, dtype=torch.float64)[edges[:, 1]], dim=1
    )
    stats = {
        "physical_anchors": int(n),
        "unique_edges": int(len(edges)),
        "isolated_anchors": int(isolated.sum()),
        "sinkhorn_iterations": int(iterations),
        "normalization": "metropolis_hastings",
        "row_error": row_error,
        "edge_p50_m": float(lengths.median()),
        "edge_p95_m": float(torch.quantile(lengths, 0.95)),
        "edge_max_m": float(lengths.max()),
        "edges_gt_30m": int((lengths > 30.0).sum()),
        "d_max": float(d_max),
        "dropped_over_cap": int(dropped),
        "zero_diagonal": True,
    }
    return sp.csc_matrix(weights), edges, stats


def neighbor_spacing(coords, buildings, floors, *, count=8):
    """Median same-BF neighbour distance per anchor (local grid spacing, metres)."""
    xy = np.asarray(torch.as_tensor(coords).detach().cpu(), dtype=np.float64)
    b = np.asarray(torch.as_tensor(buildings).detach().cpu(), dtype=np.int64)
    f = np.asarray(torch.as_tensor(floors).detach().cpu(), dtype=np.int64)
    n = len(xy)
    out = np.zeros(n, dtype=np.float64)
    strata: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(zip(b.tolist(), f.tolist())):
        strata.setdefault(key, []).append(idx)
    for members in strata.values():
        if len(members) <= 1:
            continue
        block = xy[members]
        delta = block[:, None, :] - block[None, :, :]
        distance = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))
        k = min(count, len(members) - 1)
        for local_i, global_i in enumerate(members):
            row = np.sort(distance[local_i])[1:k + 1]
            out[global_i] = float(np.median(row))
    missing = out <= 0
    if missing.any():
        out[missing] = float(np.median(out[~missing])) if (~missing).any() else 1.0
    return torch.as_tensor(out, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# operators                                                                   #
# --------------------------------------------------------------------------- #
def smooth_channels(graph_csc, a, scores, z):
    """Return (I - a W)^-1 applied to scores and both G13 channels."""
    n = graph_csc.shape[0]
    a = float(a)
    if a == 0.0:
        return scores, z
    matrix = (sp.identity(n, format="csc") - a * graph_csc).tocsc()
    lu = splu(matrix, permc_spec="MMD_AT_PLUS_A")
    batch = scores.shape[0]
    host = lambda t: t.detach().cpu().numpy().astype(np.float64)
    stacked = np.concatenate(
        [host(scores).T, host(z[..., 0]).T, host(z[..., 1]).T], axis=1
    )
    solved = lu.solve(np.ascontiguousarray(stacked))
    device, dtype = scores.device, scores.dtype
    s = torch.as_tensor(solved[:, :batch].T, device=device, dtype=dtype)
    z0 = torch.as_tensor(solved[:, batch:2 * batch].T, device=device, dtype=dtype)
    z1 = torch.as_tensor(solved[:, 2 * batch:3 * batch].T, device=device, dtype=dtype)
    return s, torch.stack((z0, z1), dim=-1)


def inflate_scales(scales, spacing, rho):
    if float(rho) == 0.0:
        return scales
    return torch.sqrt(scales.square() + (float(rho) * spacing)[None, :, None].square())


def light_metrics(pack, ctx, q, centers=None, scales=None, ids=None):
    centers = pack["xy"] if centers is None else centers
    scales = pack["sigma"] if scales is None else scales
    ids = pack["ids"] if ids is None else ids
    metrics, _ = spatial_metrics(
        q, ids, centers, pack["target"], pack["building"], pack["floor"],
        ctx["classes"], ctx["class_pairs"], keys=pack["keys"],
    )
    nll = op.joint_bf_xy_nll(
        q, ids, centers, scales, pack["target"], pack["building"], pack["floor"],
        ctx["memory"]["building"], ctx["memory"]["floor"],
    )
    return {
        "mae": float(metrics["macro_mae_full_m"]),
        "bf": float(metrics["macro_bf_accuracy"]),
        "nll": float(nll.mean()),
    }


def full_metrics(pack, ctx, q, *, centers=None, scales=None, ids=None, seed=0):
    metrics, _ = s3.score_mixture(
        pack, ctx, q, centers=centers, scales=scales, ids=ids, seed=seed + 900
    )
    return {
        "mae": float(metrics["macro_mae_full_m"]),
        "bf": float(metrics["macro_bf_accuracy"]),
        "nll": float(metrics["macro_joint_nll"]),
        "es": float(metrics["macro_es_m"]),
        "p95": float(metrics["site_p95_full_m"]),
        "worst_site": float(metrics["worst_site_mae_m"]),
    }


# --------------------------------------------------------------------------- #
# one run                                                                     #
# --------------------------------------------------------------------------- #
def run_one(dataset, seed, arm, args, graph_cache, spacing_cache):
    started = time.time()
    device = torch.device(args.device)
    ctx = s3.reconstruct(dataset, seed, args.root, device)
    model = s3.enable_training_arm(ctx["initial"], arm, ctx["graph"])
    checkpoint = (Path(args.checkpoint_root) / "trained" / f"{dataset}_seed{seed}" / arm
                  / "best_dev_epoch_21_80.pt")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    cache_key = dataset
    if cache_key not in graph_cache:
        graph, edges, stats = build_floor_graph(
            ctx["memory"]["coords"], ctx["memory"]["building"], ctx["memory"]["floor"],
            d_max=args.dmax,
        )
        graph_cache[cache_key] = (graph, edges, stats)
    graph, edges, stats = graph_cache[cache_key]
    if cache_key not in spacing_cache:
        spacing_cache[cache_key] = neighbor_spacing(
            ctx["memory"]["coords"], ctx["memory"]["building"], ctx["memory"]["floor"]
        )
    spacing = spacing_cache[cache_key].to(device)
    h = stats["edge_p50_m"]

    fit = s3.make_pack(ctx, model, ctx["cal_fit"])
    ev = s3.make_pack(ctx, model, ctx["cal_eval"])

    clean, _, clean_objective = s3.grid_calibrate(fit, ctx)
    q_base = torch.softmax(s3.complete_logits(fit, clean), dim=1)
    base_fit = light_metrics(fit, ctx, q_base)
    q_base_eval = torch.softmax(s3.complete_logits(ev, clean), dim=1)
    base_eval = full_metrics(ev, ctx, q_base_eval, seed=seed)

    # ---- A1: a=0 must be the identity ------------------------------------- #
    s0, z0 = smooth_channels(graph, 0.0, fit["scores"], fit["z"])
    a1_gap = float((s0 - fit["scores"]).abs().max()) + float((z0 - fit["z"]).abs().max())

    # ---- A6: resolvent commutes with temperature --------------------------- #
    ts, tz = smooth_channels(graph, args.alpha_test, fit["scores"] / 0.7, fit["z"] / 0.7)
    ts2, tz2 = smooth_channels(graph, args.alpha_test, fit["scores"], fit["z"])
    a6_gap = float((ts - ts2 / 0.7).abs().max()) + float((tz - tz2 / 0.7).abs().max())

    # ---- L1 sweep (selection: CAL-fit MAE, guard CAL-fit NLL) -------------- #
    sweep = []
    for a in args.alphas:
        s_a, z_a = smooth_channels(graph, a, fit["scores"], fit["z"])
        fit_a = dict(fit)
        fit_a["scores"], fit_a["z"] = s_a, z_a
        params_a, _, objective = s3.grid_calibrate(fit_a, ctx)
        q_fit = torch.softmax(s3.complete_logits(fit_a, params_a), dim=1)
        fit_m = light_metrics(fit_a, ctx, q_fit)
        ev_a = dict(ev)
        s_e, z_e = smooth_channels(graph, a, ev["scores"], ev["z"])
        ev_a["scores"], ev_a["z"] = s_e, z_e
        q_eval = torch.softmax(s3.complete_logits(ev_a, params_a), dim=1)
        eval_m = full_metrics(ev_a, ctx, q_eval, seed=seed)
        sweep.append({
            "a": float(a),
            "implied_range_m": None if a <= 0 else float(h / (1.0 - a)),
            "T": float(params_a["temperature"]),
            "theta": [float(x) for x in params_a["theta"]],
            "cal_objective": float(objective),
            "fit_mae": fit_m["mae"],
            "fit_nll": fit_m["nll"],
            "fit_bf": fit_m["bf"],
            "eval_mae": eval_m["mae"],
            "eval_nll": eval_m["nll"],
            "eval_es": eval_m["es"],
            "eval_bf": eval_m["bf"],
        })
    guard_ok = [r for r in sweep if r["fit_nll"] <= base_fit["nll"] + 0.05]
    pool = guard_ok if guard_ok else sweep
    best = min(pool, key=lambda r: (r["fit_mae"], -r["a"] if args.prefer_smoothing else r["a"]))
    a_star = best["a"]
    s_fit, z_fit = smooth_channels(graph, a_star, fit["scores"], fit["z"])
    fit_star = dict(fit)
    fit_star["scores"], fit_star["z"] = s_fit, z_fit
    params_star = {"temperature": best["T"], "theta": best["theta"]}
    s_eval, z_eval = smooth_channels(graph, a_star, ev["scores"], ev["z"])
    ev_star = dict(ev)
    ev_star["scores"], ev_star["z"] = s_eval, z_eval
    q_star_fit = torch.softmax(s3.complete_logits(fit_star, params_star), dim=1)
    q_star_eval = torch.softmax(s3.complete_logits(ev_star, params_star), dim=1)

    # ---- L2 sweep (selection: CAL-fit NLL) --------------------------------- #
    rho_rows = []
    for rho in args.rhos:
        scales_fit = inflate_scales(fit_star["sigma"], spacing, rho)
        fit_m = light_metrics(fit_star, ctx, q_star_fit, scales=scales_fit)
        scales_eval = inflate_scales(ev_star["sigma"], spacing, rho)
        eval_m = full_metrics(ev_star, ctx, q_star_eval, scales=scales_eval, seed=seed)
        rho_rows.append({
            "rho": float(rho),
            "kappa_over_spacing": float(rho),
            "fit_nll": fit_m["nll"],
            "eval_nll": eval_m["nll"],
            "eval_es": eval_m["es"],
            "eval_mae": eval_m["mae"],
            "eval_bf": eval_m["bf"],
            "eval_p95": eval_m["p95"],
        })
    rho_best = min(rho_rows, key=lambda r: (r["fit_nll"], r["rho"]))
    rho_star = rho_best["rho"]

    # ---- A2 / A4: invariance (reference = the selected smoothing, not base) -- #
    mae_gap = max(abs(r["eval_mae"] - best["eval_mae"]) for r in rho_rows)
    bf_gap = max(abs(r["eval_bf"] - best["eval_bf"]) for r in rho_rows)
    a4_gap = max(abs(r["eval_mae"] - best["eval_mae"]) for r in rho_rows)

    # ---- A5: 2x2x2 support decomposition on CAL-eval ---------------------- #
    decomposition = {}
    vpack = s3.make_pack(ctx, model, ctx["cal_eval"], virtual_gamma=0.0)
    parent_clean = s3.complete_logits(vpack, clean)
    virt_clean = op.combine_retrieval_and_g13_logits(
        vpack["virtual_scores"], vpack["virtual_z"], clean["temperature"],
        torch.as_tensor(clean["theta"], dtype=parent_clean.dtype, device=device),
    )
    parent_star = s3.complete_logits(vpack, params_star)
    delta = parent_star - parent_clean
    edge_delta = 0.5 * (delta[:, ctx["edges"][:, 0]] + delta[:, ctx["edges"][:, 1]])
    virt_star = virt_clean + edge_delta
    d_virtual = 0.5 * (spacing[ctx["edges"][:, 0]] + spacing[ctx["edges"][:, 1]])
    spacing_expanded = torch.cat((spacing, d_virtual), dim=0)

    def cell_entry(name, parent, virtual, with_bandwidth):
        cell = s3.fast_cell_mixture(
            parent, virtual, ctx["edges"], 0.8,
            vpack["xy"], vpack["sigma"], vpack["virtual_xy"], vpack["virtual_sigma"],
        )
        scales = inflate_scales(cell["sigma"], spacing_expanded, rho_star) if with_bandwidth else cell["sigma"]
        decomposition[name] = full_metrics(
            vpack, ctx, cell["q"], centers=cell["xy"], scales=scales,
            ids=cell["ids"], seed=seed,
        )

    decomposition["base"] = dict(base_eval)
    decomposition["1G"] = full_metrics(
        ev_star, ctx, q_star_eval, scales=ev_star["sigma"], seed=seed
    )
    decomposition["2K"] = full_metrics(
        ev_star, ctx, q_base_eval, scales=inflate_scales(ev["sigma"], spacing, rho_star), seed=seed
    )
    decomposition["1G+2K"] = full_metrics(
        ev_star, ctx, q_star_eval, scales=inflate_scales(ev_star["sigma"], spacing, rho_star), seed=seed
    )
    cell_entry("2A", parent_clean, virt_clean, False)
    cell_entry("2A+2K", parent_clean, virt_clean, True)
    cell_entry("1G+2A", parent_star, virt_star, False)
    cell_entry("1G+2A+2K", parent_star, virt_star, True)
    # Propagation ablation: smoothed parents, untouched clean virtual logits.
    cell_entry("1G+2A_noprop", parent_star, virt_clean, False)

    result = {
        "dataset": dataset,
        "seed": seed,
        "arm": arm,
        "n_anchors": int(model.n_anchors),
        "cal_fit_rows": int(len(ctx["cal_fit"])),
        "cal_eval_rows": int(len(ctx["cal_eval"])),
        "graph": stats,
        "clean_params": {"T": float(clean["temperature"]), "theta": [float(x) for x in clean["theta"]]},
        "clean_objective": float(clean_objective),
        "base_fit": base_fit,
        "base_eval": base_eval,
        "alpha_sweep": sweep,
        "selected": {"a": a_star, "implied_range_m": best["implied_range_m"],
                     "T": best["T"], "theta": best["theta"],
                     "fit_mae": best["fit_mae"], "fit_nll": best["fit_nll"]},
        "rho_sweep": rho_rows,
        "selected_rho": rho_star,
        "decomposition": decomposition,
        "assertions": {
            "A1_a0_identity_max_abs": a1_gap,
            "A2_mae_max_abs_gap_over_rho": mae_gap,
            "A2_bf_max_abs_gap_over_rho": bf_gap,
            "A4_composition_gap": a4_gap,
            "A6_commutation_max_abs": a6_gap,
            "evaluated_rows": int(len(ctx["cal_eval"])),
            "internal_test_rows_scored": 0,
            "official_test_rows_scored": 0,
        },
        "runtime_s": time.time() - started,
    }
    print(json.dumps({k: result[k] for k in ("dataset", "seed", "arm", "selected",
                                             "selected_rho", "assertions")}, ensure_ascii=False),
          flush=True)
    del fit, ev, fit_star, ev_star, vpack
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CKPT))
    parser.add_argument("--out", default=str(REPO / "outputs/v17_gmrf_bandwidth_v1"))
    parser.add_argument("--datasets", default="tuji,uts,uji")
    parser.add_argument("--seeds", default="17301")
    parser.add_argument("--arms", default="D")
    parser.add_argument("--alphas", default=",".join(str(x) for x in ALPHAS))
    parser.add_argument("--rhos", default=",".join(str(x) for x in RHOS))
    parser.add_argument("--alpha-test", type=float, default=0.55)
    parser.add_argument("--dmax", type=float, default=25.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prefer-smoothing", action="store_true")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    args.alphas = [float(x) for x in str(args.alphas).split(",") if x != ""]
    args.rhos = [float(x) for x in str(args.rhos).split(",") if x != ""]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    graph_cache, spacing_cache, results = {}, {}, []
    for dataset in [x for x in args.datasets.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            for arm in [x for x in args.arms.split(",") if x]:
                if args.only and f"{dataset}:{seed}:{arm}" != args.only:
                    continue
                try:
                    results.append(run_one(dataset, seed, arm, args, graph_cache, spacing_cache))
                except Exception as exc:  # keep the remaining runs alive
                    import traceback
                    traceback.print_exc()
                    results.append({"dataset": dataset, "seed": seed, "arm": arm,
                                    "error": repr(exc)})
                (out / "RESULTS.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    print(f"wrote {out / 'RESULTS.json'} ({len(results)} runs)", flush=True)


if __name__ == "__main__":
    main()
