"""MSEP single-member causal pilot: floor neighbors x joint location x RSS evidence.

Run as: python -m fhm_dapp_tim.fhm_dapp.msep_joint_ablation_v1 --dataset uji
Scores ONLY split-internal development sites. The split's TEST role is NEVER
scored or used for selection. This is exploratory, not a new blind benchmark.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fhm_dapp_tim.fhm_dapp.msep_single_study import (
    _load, _seed, _prepare_fit, _train_base, _pack,
)
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import (
    candidate_evidence, split_physical_sites, anchor_site_keys, energy_score_mc,
)
from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import (
    same_floor_neighbors, neighbor_contamination, class_structure,
    mask_self_when_supported, refit_loss, calibrate_probability, probability,
    spatial_metrics, site_mean, student_t_logpdf,
)


def split_cal(rows, keys, seed):
    """Split CAL into site-disjoint parameter-fit and comparison sites."""
    by_site = {}
    for i in rows:
        by_site.setdefault(keys[int(i)], []).append(int(i))
    sites = sorted(by_site)
    if len(sites) < 4:
        raise ValueError("CAL needs at least four independent sites")
    rng = np.random.default_rng(seed)
    selected = set(rng.permutation(len(sites))[:len(sites)//2].tolist())
    fit, check = [], []
    for j, site in enumerate(sites):
        (fit if j in selected else check).extend(by_site[site])
    if set(fit) & set(check):
        raise AssertionError("CAL site leakage")
    return np.asarray(sorted(fit), dtype=np.int64), np.asarray(sorted(check), dtype=np.int64)


def fit_refit(base, *, kind, evidence, rss, xy, sample_anchor,
              obs, roles, anchor_class, epochs, batch, seed, device):
    """Common start/seed/budget; independent gradient updates for each policy."""
    model = copy.deepcopy(base)
    theta = torch.nn.Parameter(torch.tensor([.2, .2], device=device))
    groups = [{"params": model.parameters(), "lr": 3e-5}]
    if evidence:
        groups.append({"params": [theta], "lr": 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    fit = roles["fit"]
    ds = TensorDataset(rss[fit], xy[fit], torch.from_numpy(sample_anchor[fit]).long())
    torch.manual_seed(seed + 113)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 113)
    gen = torch.Generator().manual_seed(seed + 131)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=gen, num_workers=0)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y, a in loader:
            x, y, a = x.to(device), y.to(device), a.to(device)
            # All FIT reference modes stay available. Remove own component when
            # alternatives exist in the true BF stratum, else retain singleton.
            scores, ids, loc, scale, z, _ = candidate_evidence(model, x, obs, k=None)
            scores, _ = mask_self_when_supported(scores, ids, a, anchor_class)
            logits = scores / .6
            if evidence:
                logits = logits + (z * theta.clamp(0, 1.5)).sum(-1)
            q = torch.softmax(logits, dim=1)
            prior = torch.softmax(scores.detach() / .6, dim=1)
            loss = refit_loss(kind, q, ids, loc, scale, y, anchor_class[a],
                              anchor_class, prior=prior)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite refit loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + ([theta] if evidence else []), 5.)
            opt.step()
            with torch.no_grad():
                theta.clamp_(0., 1.5)
            total += float(loss.detach())
        history.append({"epoch": epoch + 1, "fit_loss_sum": total,
                        "batches": len(loader)})
        print(json.dumps({"stage": "refit", "kind": kind, "evidence": evidence,
                          **history[-1]}), flush=True)
    model.eval()
    return model, ([float(x) for x in theta.detach()] if evidence else [0., 0.]), history


@torch.no_grad()
def metrics_for_pack(pack, params, data, rows, anchor_class, class_pairs,
                     *, seed, draws=32):
    q = probability(pack["scores"], pack["z"], params)
    b = torch.as_tensor(np.asarray(data.building)[rows],
                        dtype=torch.long, device=q.device)
    f = torch.as_tensor(np.asarray(data.floor)[rows],
                        dtype=torch.long, device=q.device)
    m, row = spatial_metrics(q, pack["ids"], pack["xy"], pack["target"],
                             b, f, anchor_class, class_pairs, keys=pack["keys"])
    lp = student_t_logpdf(pack["xy"], pack["sigma"], pack["target"])
    cls = anchor_class[pack["ids"]]
    tcls = ((class_pairs[:, 0][None, :] == b[:, None])
            & (class_pairs[:, 1][None, :] == f[:, None])).to(torch.int32).argmax(1)
    true_mode = cls == tcls[:, None]
    joint_nll = -torch.logsumexp(
        (q.clamp_min(1e-30).log() + lp).masked_fill(~true_mode, -torch.inf), dim=1)
    es = energy_score_mc(q, pack["xy"], pack["sigma"], pack["target"],
                         draws=draws, seed=seed).cpu().numpy()
    m.update({"macro_es_m": site_mean(es, pack["keys"]),
              "macro_joint_nll": site_mean(joint_nll.cpu().numpy(), pack["keys"]),
              "n_scans": len(rows), "n_sites": len(set(pack["keys"]))})
    by_site = {}
    for i, key in enumerate(pack["keys"]):
        k = json.dumps(key)
        by_site.setdefault(k, {"errors": [], "errors_bf": [], "bf": [], "es": []})
        by_site[k]["errors"].append(row["err_full"][i])
        by_site[k]["errors_bf"].append(row["err_predicted_bf"][i])
        by_site[k]["bf"].append(row["bf_correct"][i])
        by_site[k]["es"].append(float(es[i]))
    return m, {k: {"mae_full": float(np.mean(v["errors"])),
                    "mae_predicted_bf": float(np.mean(v["errors_bf"])),
                    "bf_accuracy": float(np.mean(v["bf"])),
                    "es": float(np.mean(v["es"])), "n_scans": len(v["errors"])}
               for k, v in by_site.items()}


@torch.no_grad()
def pack_model(model, obs, rss, xy, keys, rows, batch, device):
    return _pack(model, obs, rss, rows, xy, keys, k=None, device=device,
                 batch_size=batch)


def main(args):
    if args.seed < 0 or args.base_epochs < 1 or args.joint_epochs < 1:
        raise ValueError("invalid training budgets")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _seed(args.seed)
    data = _load(args.dataset, Path(args.data_root) if args.data_root else None)
    site_keys = anchor_site_keys(data.coords_m, data.building, data.floor)
    roles = split_physical_sites(site_keys, seed=args.split_seed)
    cal_fit, cal_eval = split_cal(roles["cal"], site_keys, args.split_seed+99)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    rss, xy, _, memory, sample_anchor, obs, device_ids, initial, old_n, old_w = (
        _prepare_fit(data, roles, device))
    floor_n, floor_w = same_floor_neighbors(
        memory["coords"], memory["building"], memory["floor"], count=8,
        temperature_m=8.)
    class_pairs, anchor_class = class_structure(
        memory["building"].to(device), memory["floor"].to(device))
    report = {
        "study": "MSEP causal single-member v1; exploratory split-internal only",
        "dataset": args.dataset, "config": vars(args),
        "warning": "TEST NOT EVALUATED; previous experiment exposed historical outcomes",
        "site_counts": {k: len(set(site_keys[int(i)] for i in v))
                        for k, v in roles.items()},
        "cal_fit_sites": len(set(site_keys[int(i)] for i in cal_fit)),
        "cal_eval_sites": len(set(site_keys[int(i)] for i in cal_eval)),
        "neighbor_mass_wrong_bf_original": neighbor_contamination(
            old_n, old_w, memory["building"], memory["floor"]),
        "neighbor_mass_wrong_bf_new": neighbor_contamination(
            floor_n, floor_w, memory["building"], memory["floor"]),
        "variants": {}, "calibration": {},
    }
    by_site = {}
    all_bases = {}
    for name, neighbors, nw in [
        ("raw", old_n, old_w),
        ("floor_neighbors", floor_n.to(device), floor_w.to(device)),
    ]:
        _seed(args.seed)
        model = copy.deepcopy(initial)
        state = _train_base(model, rss, xy, sample_anchor, device_ids,
                            neighbors, nw, obs, roles, site_keys, device=device,
                            epochs=args.base_epochs, patience=12,
                            batch=args.batch, k=None, seed=args.seed)
        all_bases[name] = model
        pack = pack_model(model, obs, rss, xy, site_keys, cal_eval,
                          args.batch, device)
        value, sites = metrics_for_pack(
            pack, {"temperature": 1., "theta": [0., 0.]}, data,
            cal_eval, anchor_class, class_pairs, seed=args.seed+900,
            draws=args.es_draws)
        report["variants"][name] = {"metrics": value, "training": state}
        by_site[name] = sites
        print(json.dumps({"stage": "base_done", "variant": name,
                          "macro_mae_m": value["macro_mae_full_m"]}), flush=True)
    # Stage A: same raw backbone, exactly the same continuous components.
    caltrain_pack = pack_model(all_bases["raw"], obs, rss, xy, site_keys,
                               cal_fit, args.batch, device)
    raw_check_pack = pack_model(all_bases["raw"], obs, rss, xy, site_keys,
                                cal_eval, args.batch, device)
    policies = {
        "raw": {"temperature": 1., "theta": [0., 0.]},
        "temp_t06": {"temperature": .6, "theta": [0., 0.]},
        "temp_fit": calibrate_probability(caltrain_pack, mode="temperature"),
        "g13_t06": calibrate_probability(caltrain_pack, mode="g13_fixed"),
        "temp_plus_g13": calibrate_probability(caltrain_pack, mode="joint"),
    }
    for name, p in policies.items():
        met, sites = metrics_for_pack(raw_check_pack, p, data, cal_eval,
                                      anchor_class, class_pairs, seed=args.seed+900,
                                      draws=args.es_draws)
        report["calibration"][name] = {"params": p, "metrics": met}
        by_site["cal_"+name] = sites
    del caltrain_pack, raw_check_pack
    # Stage B: independent 10-epoch equal-budget refits from each selected
    # base. xy is original 2-D-only counterfactual; joint includes BF-XY.
    for base_name, base in all_bases.items():
        for kind, evidence in (("xy", False), ("joint", False), ("joint", True)):
            if evidence and args.skip_evidence_refit:
                continue
            tag = base_name + "_" + kind + ("_with_evidence" if evidence else "")
            joint_model, theta, history = fit_refit(
                base, kind=kind, evidence=evidence, rss=rss, xy=xy,
                sample_anchor=sample_anchor, obs=obs, roles=roles,
                anchor_class=anchor_class, epochs=args.joint_epochs,
                batch=args.batch, seed=args.seed, device=device)
            pack = pack_model(joint_model, obs, rss, xy, site_keys,
                              cal_eval, args.batch, device)
            params = {"temperature": .6, "theta": theta}
            met, sites = metrics_for_pack(pack, params, data, cal_eval,
                                          anchor_class, class_pairs,
                                          seed=args.seed+900, draws=args.es_draws)
            report["variants"][tag] = {
                "metrics": met, "theta": theta,
                "epochs": len(history), "last_fit_loss": history[-1]["fit_loss_sum"]}
            by_site[tag] = sites
            print(json.dumps({"stage": "variant_done", "variant": tag,
                              "macro_mae_m": met["macro_mae_full_m"],
                              "bf_accuracy": met["macro_bf_accuracy"]}), flush=True)
            if args.save_checkpoints:
                torch.save({"model": joint_model.state_dict(), "theta": theta,
                            "class_pairs": class_pairs.cpu(),
                            "anchor_class": anchor_class.cpu()},
                           output / (tag+".pt"))
            del pack, joint_model
    # Never write data/parameters/metrics derived from TEST role.
    (output/"RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    (output/"SITE_DIAGNOSTICS.json").write_text(
        json.dumps(by_site, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(json.dumps({"stage": "complete", "output": str(output),
                      "test_accesses": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tuji", "uts", "uji"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=17301)
    parser.add_argument("--split-seed", type=int, default=20261017)
    parser.add_argument("--base-epochs", type=int, default=80)
    parser.add_argument("--joint-epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--es-draws", type=int, default=32)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--skip-evidence-refit", action="store_true")
    main(parser.parse_args())
