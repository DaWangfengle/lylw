"""Hypothesis tests for the GMRF + discretization-bandwidth design (zero retraining).

H1  calibration risk term 0.30 sharpens the posterior and manufactures the NLL gap
H2  midpoint Cell LSE shifts the mean; mass-preserving and symmetric quadrature do not
H3  the transport risk is blind to the mean shift (Jensen gap)
H4  graph locality: the 77 m UJI edges matter for the resolvent
H5  bandwidth optimum vs the 1/sqrt(12) prediction and its seed stability
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for path in (str(REPO), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_validation as rv  # noqa: E402
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_operators_v1 as op  # noqa: E402
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_structure_v1 as s3  # noqa: E402
from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import spatial_metrics  # noqa: E402

torch.set_grad_enabled(False)
TINY = 1e-30
RHOS_FINE = tuple(round(0.05 * i, 2) for i in range(17))


# --------------------------------------------------------------------------- #
# calibration with a configurable risk weight                                  #
# --------------------------------------------------------------------------- #
def _parts(q, cache):
    lp, distances, weights = cache
    valid = (q > 0) & torch.isfinite(lp)
    log_q = q.clamp_min(TINY).log()
    terms = (log_q + lp).masked_fill(~valid, -torch.inf)
    nll = -torch.logsumexp(terms, dim=-1)
    risk = (q * distances).sum(-1)
    return nll, risk, weights


def calibrate(pack, ctx, risk_weight, coarse_t, coarse_theta):
    cache = s3.objective_cache(pack, ctx)
    table = []

    def evaluate(candidates, stage):
        for start in range(0, len(candidates), 8):
            values = candidates[start:start + 8]
            p = torch.as_tensor(np.ascontiguousarray(values), dtype=pack["scores"].dtype,
                                device=pack["scores"].device)
            logits = (pack["scores"][None] / p[:, 0, None, None]
                      + pack["z"][None, :, :, 0] * p[:, 1, None, None]
                      + pack["z"][None, :, :, 1] * p[:, 2, None, None])
            q = torch.softmax(logits, dim=-1)
            nll, risk, weights = _parts(q, cache)
            objective = ((nll + float(risk_weight) * risk) * weights[None, :]).sum(-1)
            for (t, v, s), o, n, r in zip(
                    values, objective.cpu().tolist(),
                    nll.mean(1).cpu().tolist(), risk.mean(1).cpu().tolist()):
                table.append(dict(stage=stage, temperature=t, theta=[v, s],
                                  objective=float(o), nll=float(n), risk=float(r)))

    evaluate(np.array(np.meshgrid(coarse_t, coarse_theta, coarse_theta)).T.reshape(-1, 3), "coarse")
    tie = lambda r: (r["objective"], r["temperature"], r["theta"][0], r["theta"][1])
    best = min(table, key=tie)
    ti = min(range(len(coarse_t)), key=lambda i: abs(coarse_t[i] - best["temperature"]))
    t_ref = [float(x) for x in np.geomspace(coarse_t[max(0, ti - 1)], coarse_t[min(len(coarse_t) - 1, ti + 1)], 21)]
    v_ref = [float(x) for x in np.arange(max(0.0, best["theta"][0] - .25), min(1.5, best["theta"][0] + .25) + 1e-9, .0625)]
    s_ref = [float(x) for x in np.arange(max(0.0, best["theta"][1] - .25), min(1.5, best["theta"][1] + .25) + 1e-9, .0625)]
    evaluate(np.array(np.meshgrid(t_ref, v_ref, s_ref)).T.reshape(-1, 3), "refine")
    best = min(table, key=tie)
    return {"temperature": float(best["temperature"]), "theta": [float(x) for x in best["theta"]]}, table


# --------------------------------------------------------------------------- #
# cell variants                                                                #
# --------------------------------------------------------------------------- #
def cell_components(pack, parent, virtual, edges, beta, preserve_cell_mass):
    """Cell LSE mixture, optionally with cell masses frozen at softmax(parent)."""
    batch, n = parent.shape
    device = parent.device
    edges = edges.long()
    degree = torch.bincount(edges.flatten(), minlength=n).to(parent.dtype)
    incident = [[] for _ in range(n)]
    for edge_id, (i, j) in enumerate(edges.detach().cpu().tolist()):
        incident[i].append(edge_id)
        incident[j].append(edge_id)
    internal, cell_logits = [], []
    for i in range(n):
        ids = incident[i]
        if not ids:
            internal.append(parent.new_ones((batch, 1)))
            cell_logits.append(parent[:, i])
            continue
        terms = torch.cat(
            [parent[:, i:i + 1] + math.log1p(-float(beta)),
             virtual[:, ids] + math.log(float(beta) / len(ids))], dim=1)
        cell_logits.append(torch.logsumexp(terms, dim=1))
        internal.append(torch.softmax(terms, dim=1))
    cell_logits = torch.stack(cell_logits, dim=1)
    cell_q = torch.softmax(parent, dim=1) if preserve_cell_mass else torch.softmax(cell_logits, dim=1)
    parent_w = torch.stack([cell_q[:, i] * internal[i][:, 0] for i in range(n)], dim=1)
    edge_w = []
    for edge_id, (i, j) in enumerate(edges.detach().cpu().tolist()):
        slot_i = incident[i].index(edge_id) + 1
        slot_j = incident[j].index(edge_id) + 1
        edge_w.append(cell_q[:, i] * internal[i][:, slot_i] + cell_q[:, j] * internal[j][:, slot_j])
    edge_w = torch.stack(edge_w, dim=1) if edge_w else parent.new_zeros((batch, 0))
    q = torch.cat((parent_w, edge_w), dim=1)
    centers = torch.cat((pack["xy"], pack["virtual_xy"]), dim=1)
    scales = torch.cat((pack["sigma"], pack["virtual_sigma"]), dim=1)
    ids = torch.cat((torch.arange(n, device=device), edges[:, 0]))[None].expand(batch, -1)
    return q, centers, scales, ids


def quadrature_components(pack, q, spacing, rho_q, K=4, rho_k=0.0):
    batch, n = q.shape
    angles = 2.0 * math.pi * torch.arange(K, device=q.device, dtype=q.dtype) / K
    unit = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)  # (K,2)
    offsets = rho_q * spacing[:, None, None] * unit[None]                # (n,K,2)
    centers = (pack["xy"][:, :, None, :] + offsets[None]).reshape(batch, n * K, 2)
    base_scales = rv.inflate_scales(pack["sigma"], spacing, rho_k) if rho_k else pack["sigma"]
    scales = base_scales[:, :, None, :].expand(batch, n, K, 2).reshape(batch, n * K, 2)
    weights = q[:, :, None].expand(batch, n, K).reshape(batch, n * K) / K
    ids = torch.arange(n, device=q.device).repeat_interleave(K)[None].expand(batch, -1)
    return weights, centers, scales, ids


def site_nll(pack, ctx, q, centers=None, scales=None, ids=None):
    centers = pack["xy"] if centers is None else centers
    scales = pack["sigma"] if scales is None else scales
    ids = pack["ids"] if ids is None else ids
    per_row = op.joint_bf_xy_nll(
        q, ids, centers, scales, pack["target"], pack["building"], pack["floor"],
        ctx["memory"]["building"], ctx["memory"]["floor"],
    ).detach().cpu().numpy()
    groups: dict[tuple, list[float]] = {}
    for value, key in zip(per_row, pack["keys"]):
        groups.setdefault(tuple(key), []).append(float(value))
    return {key: float(np.mean(values)) for key, values in groups.items()}


def risk(pack, q, centers=None):
    centers = pack["xy"] if centers is None else centers
    return float(op.component_center_transport_risk(q, centers, pack["target"]).mean())


# --------------------------------------------------------------------------- #
# one run                                                                      #
# --------------------------------------------------------------------------- #
def run_one(dataset, seed, arm, args, coarse_t, coarse_theta):
    device = torch.device(args.device)
    ctx = s3.reconstruct(dataset, seed, args.root, device)
    model = s3.enable_training_arm(ctx["initial"], arm, ctx["graph"])
    ckpt = Path(args.checkpoint_root) / "trained" / f"{dataset}_seed{seed}" / arm / "best_dev_epoch_21_80.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"], strict=True)
    model.eval()

    graph, edges, stats = rv.build_floor_graph(
        ctx["memory"]["coords"], ctx["memory"]["building"], ctx["memory"]["floor"], d_max=args.dmax)
    spacing = rv.neighbor_spacing(
        ctx["memory"]["coords"], ctx["memory"]["building"], ctx["memory"]["floor"]).to(device)
    fit = s3.make_pack(ctx, model, ctx["cal_fit"])
    ev = s3.make_pack(ctx, model, ctx["cal_eval"])
    vpack = s3.make_pack(ctx, model, ctx["cal_eval"], virtual_gamma=0.0)

    out = {"dataset": dataset, "seed": seed, "arm": arm, "graph": stats}

    # ---- H1: risk weight in calibration ---------------------------------- #
    h1 = {}
    for tag, w in (("w0.30", 0.30), ("w0.00", 0.00)):
        params, _ = calibrate(fit, ctx, w, coarse_t, coarse_theta)
        q_fit = torch.softmax(s3.complete_logits(fit, params), dim=1)
        q_eval = torch.softmax(s3.complete_logits(ev, params), dim=1)
        base = rv.full_metrics(ev, ctx, q_eval, seed=seed)
        curve = []
        for rho in RHOS_FINE:
            scales_fit = rv.inflate_scales(fit["sigma"], spacing, rho)
            fit_nll = float(op.joint_bf_xy_nll(
                q_fit, fit["ids"], fit["xy"], scales_fit, fit["target"], fit["building"], fit["floor"],
                ctx["memory"]["building"], ctx["memory"]["floor"]).mean())
            scales = rv.inflate_scales(ev["sigma"], spacing, rho)
            nll = float(op.joint_bf_xy_nll(
                q_eval, ev["ids"], ev["xy"], scales, ev["target"], ev["building"], ev["floor"],
                ctx["memory"]["building"], ctx["memory"]["floor"]).mean())
            metrics, _ = spatial_metrics(
                q_eval, ev["ids"], ev["xy"], ev["target"], ev["building"], ev["floor"],
                ctx["classes"], ctx["class_pairs"], keys=ev["keys"])
            curve.append({"rho": rho, "fit_nll": fit_nll, "nll": nll,
                          "mae": float(metrics["macro_mae_full_m"])})
        rho_star = min(curve, key=lambda r: r["fit_nll"])["rho"]
        h1[tag] = {
            "params": params, "base": base, "rho_curve": curve, "rho_star": rho_star,
            "nll_gain_at_rho_star": base["nll"] - min(r["nll"] for r in curve),
        }
    out["h1_calibration"] = h1

    # ---- H2: cell variants at the w=0.30 calibration ---------------------- #
    params = h1["w0.30"]["params"]
    parent = s3.complete_logits(vpack, params)
    virtual = op.combine_retrieval_and_g13_logits(
        vpack["virtual_scores"], vpack["virtual_z"], params["temperature"],
        torch.tensor(params["theta"], device=device))
    h2 = {}
    q0 = torch.softmax(parent, dim=1)
    h2["base"] = rv.full_metrics(vpack, ctx, q0, seed=seed)
    h2["base"]["risk"] = risk(vpack, q0)
    for name, preserve in (("2A_lse", False), ("2A_mass", True)):
        q, centers, scales, ids = cell_components(vpack, parent, virtual, ctx["edges"], 0.8, preserve)
        h2[name] = rv.full_metrics(vpack, ctx, q, centers=centers, scales=scales, ids=ids, seed=seed)
        h2[name]["risk"] = risk(vpack, q, centers)
    rho_q = 0.4
    for name, K, rho_k in (("2Q4", 4, 0.0), ("2Q8", 8, 0.0),
                           ("2Q4+2K", 4, h1["w0.30"]["rho_star"]),
                           ("2Q8+2K", 8, h1["w0.30"]["rho_star"])):
        q, centers, scales, ids = quadrature_components(vpack, q0, spacing, rho_q, K=K, rho_k=rho_k)
        h2[name] = rv.full_metrics(vpack, ctx, q, centers=centers, scales=scales, ids=ids, seed=seed)
        h2[name]["risk"] = risk(vpack, q, centers)
    # bandwidth reference
    q1 = q0
    h2["2K"] = rv.full_metrics(vpack, ctx, q1, scales=rv.inflate_scales(vpack["sigma"], spacing, h1["w0.30"]["rho_star"]), seed=seed)
    h2["2K"]["risk"] = risk(vpack, q1)
    out["h2_cells"] = h2

    # ---- H3: site-level NLL deltas for a bootstrap ----------------------- #
    q03 = torch.softmax(s3.complete_logits(ev, h1["w0.30"]["params"]), dim=1)
    base_sites = site_nll(ev, ctx, q03)
    k_sites = site_nll(ev, ctx, q03,
                       scales=rv.inflate_scales(ev["sigma"], spacing, h1["w0.30"]["rho_star"]))
    same = sorted(set(base_sites) & set(k_sites))
    out["h3_site_delta_nll"] = {json.dumps(list(k)): base_sites[k] - k_sites[k] for k in same}

    # ---- H4: graph locality on UJI --------------------------------------- #
    if dataset == "uji":
        graph_nocap, _, stats_nocap = rv.build_floor_graph(
            ctx["memory"]["coords"], ctx["memory"]["building"], ctx["memory"]["floor"], d_max=1e9)
        out["h4_cutoff"] = {"cap": {}, "nocap": {}, "graph_nocap": stats_nocap}
        for a in (0.0, 0.4):
            for tag, g in (("cap", graph), ("nocap", graph_nocap)):
                if a == 0.0:
                    params_a = params
                else:
                    s_fit, z_fit = rv.smooth_channels(g, a, fit["scores"], fit["z"])
                    fit_a = dict(fit); fit_a["scores"], fit_a["z"] = s_fit, z_fit
                    params_a, _ = calibrate(fit_a, ctx, 0.30, coarse_t, coarse_theta)
                s_ev, z_ev = rv.smooth_channels(g, a, ev["scores"], ev["z"])
                ev_a = dict(ev); ev_a["scores"], ev_a["z"] = s_ev, z_ev
                q_a = torch.softmax(s3.complete_logits(ev_a, params_a), dim=1)
                m = rv.light_metrics(ev_a, ctx, q_a)
                out["h4_cutoff"][tag][f"a={a}"] = {"mae": m["mae"], "nll": m["nll"], "T": params_a["temperature"]}

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(rv.DEFAULT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(rv.DEFAULT_CKPT))
    parser.add_argument("--out", default=str(REPO / "outputs/v17_gmrf_bandwidth_v1/hypotheses"))
    parser.add_argument("--datasets", default="tuji,uts,uji")
    parser.add_argument("--seeds", default="17301,17302,17303")
    parser.add_argument("--arms", default="D")
    parser.add_argument("--dmax", type=float, default=25.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coarse_t = tuple(float(x) for x in np.geomspace(.25, 3., 25))
    coarse_theta = tuple(float(x) for x in np.arange(0., 1.50001, .25))

    results = []
    for dataset in [x for x in args.datasets.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            for arm in [x for x in args.arms.split(",") if x]:
                key = f"{dataset}:{seed}:{arm}"
                if args.only and key != args.only:
                    continue
                started = time.time()
                try:
                    record = run_one(dataset, seed, arm, args, coarse_t, coarse_theta)
                    record["runtime_s"] = time.time() - started
                    results.append(record)
                    print(f"[{key}] ok {record['runtime_s']:.1f}s", flush=True)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    results.append({"dataset": dataset, "seed": seed, "arm": arm, "error": repr(exc)})
                    print(f"[{key}] FAILED {exc}", flush=True)
                (out / "HYPOTHESES.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out / "HYPOTHESES.json", flush=True)


if __name__ == "__main__":
    main()
