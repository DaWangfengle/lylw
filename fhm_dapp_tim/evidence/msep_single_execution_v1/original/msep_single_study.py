"""MSEP-Loc single-member full experimental protocol (FIT/DEV/CAL/TEST).

Run from repository root:
python -m fhm_dapp_tim.fhm_dapp.msep_single_study --dataset tuji --output ...
No Git operations; no pre-trained checkpoints or official-test labels required.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fhm_dapp_tim.fhm_dapp.v18_4_datasets import load_tuji1, load_uts, load_uji
from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.model import AnchorStudentTModel
from fhm_dapp_tim.fhm_dapp.train_anchor import _neighbor_targets, _ap_building_owner
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING
from fhm_dapp_tim.fhm_dapp.uji_v17_g10_evidence_corrected_law import discrete_energy_loss
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import (
    HISTORICAL_ANCHOR_T, UnifiedAnchorPosterior, anchor_site_keys,
    build_fit_observation_memory, candidate_evidence, energy_score_mc,
    mixture_loss, posterior, site_metrics, split_physical_sites,
)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _load(dataset, root):
    kw = {"data_root": root} if root else {}
    if dataset == "tuji":
        return load_tuji1("train", **kw)[0]
    if dataset == "uts":
        return load_uts("train", **kw)[0]
    if dataset == "uji":
        return load_uji("train", **kw)[0]
    raise ValueError(f"unknown dataset {dataset}")


def _site_weights(keys):
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return torch.tensor([1.0 / counts[key] / len(counts) for key in keys], dtype=torch.float32)


def _pack(model, obs, rss, rows, coords, keys, *, k, device, batch_size):
    model.eval()
    accumulated = {n: [] for n in ("scores", "ids", "xy", "sigma", "z", "reliability")}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            block = rows[start:start+batch_size]
            vals = candidate_evidence(model, rss[block].to(device), obs, k=k)
            for n, v in zip(accumulated, vals):
                accumulated[n].append(v.detach())
    out = {n: torch.cat(v, dim=0) for n, v in accumulated.items()}
    out["target"] = coords[rows].to(device)
    out["keys"] = [keys[int(i)] for i in rows]
    out["weight"] = _site_weights(out["keys"]).to(device)
    return out


def _weighted_objective(prob, pack, *, reg_to=None, reg_weight=0.0, l2=0.0):
    row = mixture_loss(prob, pack["xy"], pack["sigma"], pack["target"],
                       kl_reference=reg_to, kl_weight=reg_weight, reduction="none")
    return (pack["weight"] * row).sum() + l2


def _fit_temperature(pack, *, initial=.6, steps=90, lr=.045):
    log_t = torch.nn.Parameter(torch.tensor(math.log(initial), device=pack["scores"].device))
    opt = torch.optim.Adam([log_t], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        p = torch.softmax(pack["scores"] / log_t.exp().clamp(.25, 3.), dim=1)
        objective = _weighted_objective(p, pack)
        objective.backward()
        opt.step()
        with torch.no_grad():
            log_t.clamp_(math.log(.25), math.log(3.))
    return float(log_t.exp().item())


def _fit_g13(pack, *, steps=100, lr=.04):
    theta = torch.nn.Parameter(torch.tensor([.8, .9], device=pack["scores"].device))
    opt = torch.optim.Adam([theta], lr=lr)
    prior = posterior(pack["scores"], pack["z"], mode="temperature")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        prob = posterior(pack["scores"], pack["z"], mode="g13", theta=theta)
        obj = _weighted_objective(prob, pack, reg_to=prior, reg_weight=.05)
        obj.backward()
        opt.step()
        with torch.no_grad():
            theta.clamp_(0., 1.5)
    return theta.detach().tolist()


def _fit_g13_historical_surrogate(pack, *, epochs=24, batch=12):
    """Original G13 CAL objective, adapted to a single member/one dataset.

    Per-anchor component centers replace the historical E5 member-averaged
    centers. Streaming mini-batches avoid materializing CAL x K x K in GPU RAM.
    This is the original G13 *calibration law*, not the original E5 panel.
    """
    theta = torch.nn.Parameter(torch.zeros(2, dtype=pack["scores"].dtype,
                                            device=pack["scores"].device))
    optimizer = torch.optim.Adam([theta], lr=.05)
    n = len(pack["scores"])
    with torch.no_grad():
        baseline = torch.softmax(pack["scores"] / HISTORICAL_ANCHOR_T, dim=1)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, n, batch):
            end = min(n,start+batch)
            q = posterior(pack["scores"][start:end],pack["z"][start:end],
                          mode="g13",theta=theta)
            c = pack["xy"][start:end]
            y = pack["target"][start:end]
            es = discrete_energy_loss(q,c,y)
            point = torch.linalg.vector_norm((q[:,:,None]*c).sum(1)-y,dim=1)
            prior = baseline[start:end]
            kl = (q*(q.clamp_min(1e-30).log()-prior.clamp_min(1e-30).log())).sum(1)
            objective = ((es+.30*point)/10. + .10*kl)
            (pack["weight"][start:end]*objective).sum().backward()
        optimizer.step()
        with torch.no_grad():
            theta.clamp_(0.,1.5)
    return theta.detach().tolist()


def _fit_unified_frozen(pack, theta, *, steps=100, lr=.025):
    head = UnifiedAnchorPosterior(theta=theta).to(pack["scores"].device)
    head.theta.requires_grad_(False)  # exactly the same global G13 coefficients
    opt = torch.optim.Adam(head.gate.parameters(), lr=lr)
    base = posterior(pack["scores"], pack["z"], mode="g13", theta=theta)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        prob = posterior(pack["scores"], pack["z"], mode="unified", unified=head,
                         reliability=pack["reliability"])
        penalty = .01*(head.gate.weight.square().sum()+head.gate.bias.square().sum())
        loss = _weighted_objective(prob, pack, reg_to=base, reg_weight=.05, l2=penalty)
        loss.backward()
        opt.step()
        head.clamp_()
    head.eval()
    return head


def _prepare_fit(data, roles, device):
    fit = roles["fit"]
    raw = data.model_rss()
    rss = torch.from_numpy(raw).float()
    xy_np = np.asarray(data.coords_m, dtype=np.float32)
    origin = xy_np[fit].mean(axis=0).astype(np.float32)
    centered = (xy_np - origin).astype(np.float32)
    xy = torch.from_numpy(centered).float()
    memory, sample_anchor = build_anchor_memory(rss, centered, data.building, data.floor, fit)
    if np.any(sample_anchor[fit] < 0):
        raise RuntimeError("FIT row has no anchor")
    if len(memory["coords"]) < 2:
        raise RuntimeError("at least two FIT anchors are needed")
    seen = (memory["prototype_rss"] > -0.999).any(dim=0)
    obs = build_fit_observation_memory(rss[fit], torch.from_numpy(sample_anchor[fit]),
                                       len(memory["coords"]), seen)
    obs["seen_ap_mask"] = seen
    obs = {k: v.to(device) for k, v in obs.items()}
    unique_devices = sorted(set(int(x) for x in data.device[fit] if int(x) >= 0))
    devmap = {x: i for i, x in enumerate(unique_devices)}
    device_ids = torch.tensor([devmap.get(int(x), -1) for x in data.device], dtype=torch.long)
    num_devices = max(len(unique_devices), 1)
    ap_owner = _ap_building_owner(memory["prototype_rss"], memory["building"],
                                   int(np.max(data.building[fit]))+1)
    kwargs = dict(
        n_aps=rss.shape[1], anchor_coords=memory["coords"],
        anchor_buildings=memory["building"], anchor_floors=memory["floor"],
        seen_ap_mask=seen, prototype_rss=memory["prototype_rss"],
        ap_building_owner=ap_owner, n_devices=num_devices,
        d_model=FROZEN_TRAINING["d_model"], num_heads=FROZEN_TRAINING["num_heads"],
        num_layers=FROZEN_TRAINING["num_layers"], max_visible=FROZEN_TRAINING["max_visible"],
        dropout=FROZEN_TRAINING["dropout"], student_df=FROZEN_TRAINING["student_df"],
        max_residual_m=FROZEN_TRAINING["max_residual_m"],
        residual_experts=FROZEN_TRAINING["residual_experts"],
        scan_feature_mode=FROZEN_TRAINING["scan_feature_mode"],
        prototype_metric=FROZEN_TRAINING["prototype_metric"],
        candidate_reranker=False,
    )
    model = AnchorStudentTModel(**kwargs).to(device)
    model.initialize_anchor_keys(memory["prototype_rss"].to(device))
    neighbors, nw = _neighbor_targets(memory["coords"], memory["building"],
                    neighbor_count=FROZEN_TRAINING["target_neighbors"],
                    temperature_m=FROZEN_TRAINING["spatial_temperature_m"])
    return rss, xy, origin, memory, sample_anchor, obs, device_ids, model, neighbors.to(device), nw.to(device)


def _dev_mae(model, obs, rss, xy, keys, devrows, *, k, device, batch):
    pack = _pack(model, obs, rss, devrows, xy, keys, k=k, device=device, batch_size=batch)
    q = posterior(pack["scores"], pack["z"], mode="raw")
    error = torch.linalg.vector_norm((q[:, :, None]*pack["xy"]).sum(1)-pack["target"], dim=1)
    return float((pack["weight"]*error).sum().item())


def _train_base(model, rss, xy, sample_anchor, device_ids, neighbor, nw,
                obs, roles, keys, *, device, epochs, patience, batch, k, seed):
    fit = roles["fit"]
    dataset = TensorDataset(rss[fit], xy[fit], torch.from_numpy(sample_anchor[fit]).long(), device_ids[fit])
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, generator=gen, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=FROZEN_TRAINING["learning_rate"],
                            weight_decay=FROZEN_TRAINING["weight_decay"])
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        total, batches = 0., 0
        for x, y, a, d in loader:
            x, y, a, d = [t.to(device) for t in (x, y, a, d)]
            opt.zero_grad(set_to_none=True)
            terms = model.training_loss(
                x, y, a, neighbor, nw,
                retrieval_weight=FROZEN_TRAINING["retrieval_weight"],
                hierarchy_weight=FROZEN_TRAINING["hierarchy_weight"],
                self_target_weight=FROZEN_TRAINING["self_target_weight"],
                coordinate_weight=FROZEN_TRAINING["coordinate_weight"],
                device_ids=d, domain_adversarial_weight=0.0,
                rss_offset_aug_db=FROZEN_TRAINING["rss_offset_aug_db"],
                rss_scale_aug=FROZEN_TRAINING["rss_scale_aug"],
                train_candidates=FROZEN_TRAINING["train_candidates"],
                ap_dropout=FROZEN_TRAINING["ap_dropout"],
            )
            if not torch.isfinite(terms["loss"]):
                raise RuntimeError("nonfinite base training loss")
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            total += float(terms["loss"].detach()); batches += 1
        score = _dev_mae(model, obs, rss, xy, keys, roles["dev"], k=k, device=device, batch=batch)
        history.append({"epoch": epoch, "fit_loss": total/batches, "dev_mae": score})
        if score < best - .01:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch >= 20 and epoch-best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("base learner has no finite DEV checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_dev_mae": best, "history": history}


def _joint_refit(base, rss, xy, sample_anchor, obs, roles, keys,
                 *, device, epochs, batch, k, seed):
    """True gradient flow through encoder, anchor posterior and Student-t head.

    The same FIT data trained the base, so this is leave-SELF-anchor-out
    training, not a completely unseen-site episodic meta-learning claim.
    """
    model = copy.deepcopy(base)
    head = UnifiedAnchorPosterior().to(device)
    fit = roles["fit"]
    ds = TensorDataset(rss[fit], xy[fit], torch.from_numpy(sample_anchor[fit]).long())
    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed+101), num_workers=0)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 3e-5},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    history = []
    n = int(base.n_anchors)
    if n < 3:
        raise ValueError("joint leave-own-anchor-out requires >=3 reference sites")
    for epoch in range(1, epochs+1):
        model.train(); head.train()
        total = 0.; steps = 0
        for x, y, a in loader:
            x, y, a = x.to(device), y.to(device), a.to(device)
            scores, ids, loc, scale, ev, rel = candidate_evidence(
                model, x, obs, k=min(n-1, k or n-1), excluded_anchor=a)
            p = head(scores, ev, rel)
            prior = torch.softmax(scores.detach()/HISTORICAL_ANCHOR_T, dim=1)
            penalty = .002*(head.gate.weight.square().sum()+head.gate.bias.square().sum())
            loss = mixture_loss(p, loc, scale, y, kl_reference=prior, kl_weight=.02) + penalty
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite joint-training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()), 5.)
            opt.step(); head.clamp_()
            total += float(loss.detach()); steps += 1
        history.append({"epoch": epoch, "fit_leave_self_loss": total/max(steps,1)})
    model.eval(); head.eval()
    return model, head, history


def _evaluate(modes, model, obs, rss, xy, keys, rows, memory,
              *, k, device, batch, draws, seed, site_sink=None):
    outcome = {name: {"err": [], "es": [], "nll": [], "mass": [], "coverage": [], "floor": [], "building": [], "joint_correct": [], "top1_near": [], "entropy": [], "top10_near": [], "nearest_center": []}
               for name in modes if not name.startswith("_")}
    for start in range(0, len(rows), batch):
        block = rows[start:start+batch]
        with torch.no_grad():
            scores, ids, loc, scales, z, rel = candidate_evidence(model, rss[block].to(device), obs, k=k)
            y = xy[block].to(device)
            anchor_xy = memory["coords"].to(device)[ids]
            anchor_b = memory["building"].to(device)[ids]
            anchor_f = memory["floor"].to(device)[ids]
            btruth = torch.tensor(np.asarray(modes["_data"].building)[block], device=device)
            ftruth = torch.tensor(np.asarray(modes["_data"].floor)[block], device=device)
            within = ((anchor_b == btruth[:, None]) & (anchor_f == ftruth[:, None])
                      & (torch.linalg.vector_norm(anchor_xy-y[:, None, :],dim=-1) <= 5.0))
            center_nearest = torch.linalg.vector_norm(loc-y[:,None,:],dim=2).min(1).values
            for name, spec in modes.items():
                if name.startswith("_"):
                    continue
                if spec["mode"] == "unified":
                    prob = posterior(scores,z,mode="unified",reliability=rel,
                                     unified=spec["head"],temperature=spec.get("temperature",.6))
                else:
                    prob = posterior(scores,z,mode=spec["mode"],
                                     temperature=spec.get("temperature",.6),theta=spec.get("theta",(0.,0.)))
                point = (prob[:, :, None]*loc).sum(1)
                err = torch.linalg.vector_norm(point-y,dim=1)
                es = energy_score_mc(prob,loc,scales,y,draws=draws,seed=seed+start)
                nll = mixture_loss(prob,loc,scales,y,mae_weight=0.,reduction="none")
                mass = (prob*within.float()).sum(1)
                floor_ok = torch.stack([(prob[i]*(anchor_f[i]==ftruth[i]).float()).sum()
                                        for i in range(len(block))])
                # Joint building-floor posterior; never use true floor in inference.
                b_ok = torch.stack([(prob[i]*(anchor_b[i]==btruth[i]).float()).sum()
                                    for i in range(len(block))])
                # Joint-class accuracy, not the probability assigned to the true class.
                correct = []
                for j in range(len(block)):
                    classes, inverse = torch.unique(anchor_b[j] * 10000 + anchor_f[j], return_inverse=True)
                    class_mass = torch.zeros(len(classes), dtype=prob.dtype, device=device)
                    class_mass.scatter_add_(0, inverse, prob[j])
                    guess = classes[class_mass.argmax()]
                    correct.append((guess == btruth[j] * 10000 + ftruth[j]).float())
                joint_correct = torch.stack(correct)
                top1_near = within.gather(1, prob.argmax(1)[:,None]).squeeze(1).float()
                ranked = prob.topk(k=min(10,prob.shape[1]),dim=1).indices
                top10_near = within.gather(1,ranked).any(1).float()
                entropy = -(prob*prob.clamp_min(1e-30).log()).sum(1)
                result = outcome[name]
                for label, t in (("err",err),("es",es),("nll",nll),("mass",mass),
                                 ("coverage",within.any(1).float()),("floor",floor_ok),
                                 ("building",b_ok),("joint_correct",joint_correct),
                                 ("top1_near",top1_near),("top10_near",top10_near),
                                 ("entropy",entropy),("nearest_center",center_nearest)):
                    result[label].append(t.detach().cpu().numpy())
    reports = {}
    site_keys = [keys[int(i)] for i in rows]
    for name, data in outcome.items():
        a = {k: np.concatenate(v) for k,v in data.items()}
        metric = site_metrics(a["err"], a["es"], site_keys)
        metric.update({"row_nll": float(a["nll"].mean()),
                       "correct_neighborhood_mass": float(a["mass"].mean()),
                       "correct_neighborhood_coverage": float(a["coverage"].mean()),
                       "true_floor_probability": float(a["floor"].mean()),
                       "true_building_probability": float(a["building"].mean()),
                       "building_floor_joint_accuracy": float(a["joint_correct"].mean()),
                       "correct_neighborhood_top1_rate": float(a["top1_near"].mean()),
                       "correct_neighborhood_top10_rate": float(a["top10_near"].mean()),
                       "anchor_entropy_nats": float(a["entropy"].mean()),
                       "nearest_component_center_error_m_oracle": float(a["nearest_center"].mean())})
        reports[name] = metric
        if site_sink is not None:
            groups = {}
            for idx, key in enumerate(site_keys):
                groups.setdefault(key, []).append(idx)
            site_sink[name] = [
                {"site":list(key),"scan_count":len(idx),
                 **{field:float(np.mean(a[field][idx])) for field in
                    ("err","es","nll","mass","coverage","floor","building","joint_correct",
                     "top1_near","top10_near","entropy","nearest_center")}}
                for key, idx in sorted(groups.items())
            ]
    return reports


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
                          else "cpu" if args.device=="auto" else args.device)
    data = _load(args.dataset, Path(args.data_root) if args.data_root else None)
    keys = anchor_site_keys(data.coords_m, data.building, data.floor)
    roles = split_physical_sites(keys, seed=args.split_seed)
    rss, xy, origin, memory, sample_anchor, obs, device_ids, base, neighbors, nw = _prepare_fit(data,roles,device)
    n = base.n_anchors
    k = None if args.support=="all" else min(args.k,n)
    if len(set(keys[int(x)] for x in roles["fit"])) != n:
        raise AssertionError("FIT anchor IDs do not biject with site keys")
    base_history = _train_base(base,rss,xy,sample_anchor,device_ids,neighbors,nw,obs,
                               roles,keys,device=device,epochs=args.base_epochs,
                               patience=args.patience,batch=args.batch,k=k,seed=args.seed)
    cal = _pack(base,obs,rss,roles["cal"],xy,keys,k=k,device=device,batch_size=args.batch)
    fitted_temperature = _fit_temperature(cal)
    theta_hist = _fit_g13_historical_surrogate(cal)
    theta = _fit_g13(cal)
    frozen_head = _fit_unified_frozen(cal,theta)
    del cal
    joint, joint_head, joint_history = _joint_refit(base,rss,xy,sample_anchor,obs,roles,keys,
                                                   device=device,epochs=args.joint_epochs,
                                                   batch=args.batch,k=k,seed=args.seed)
    modes = {"_data": data,
             "v17_single_raw": {"mode":"raw"},
             "v17_single_temp_fit": {"mode":"temperature","temperature":fitted_temperature},
             "v17_single_temp_t06": {"mode":"temperature","temperature":.6},
             "v17_single_g13_historical_cal": {"mode":"g13","theta":theta_hist},
             "v17_single_g13": {"mode":"g13","theta":theta},
             "v17_single_unified_frozen": {"mode":"unified","head":frozen_head}}
    evaluation_role = "dev" if args.smoke else "test"
    site_diagnostics = {}
    base_result = _evaluate(modes,base,obs,rss,xy,keys,roles[evaluation_role],memory,
                            k=k,device=device,batch=args.batch,draws=args.es_draws,
                            seed=args.seed+900,site_sink=site_diagnostics)
    # Explicitly isolate what changed in the backbone during joint fine tuning.
    modes_joint = {"_data":data,
                   "unified_joint_no_evidence": {"mode":"temperature","temperature":.6},
                   "unified_joint": {"mode":"unified","head":joint_head}}
    joint_result = _evaluate(modes_joint,joint,obs,rss,xy,keys,roles[evaluation_role],memory,
                             k=k,device=device,batch=args.batch,draws=args.es_draws,
                             seed=args.seed+900,site_sink=site_diagnostics)
    result = {
        "protocol":"msep-single-anchor-v1", "dataset":args.dataset,
        "hypotheses":{
            "stage1":"single-member support/weighting diagnostics",
            "stage2":"raw vs temperature vs G13 vs learned reliability posterior (frozen backbone)",
            "stage3":"FIT leave-own-anchor-out joint gradients through V17 encoder and Student-t head",
        },
        "role_isolation":"physical site (building,floor,XY rounded to 4 decimals)",
        "role_count": {role:{"rows":len(idx),"sites":len(set(keys[int(i)] for i in idx))} for role,idx in roles.items()},
        "input_ap_count":len(data.ap_ids), "fit_anchor_count":int(n),
        "fit_scan_count_quantiles":np.quantile(memory["counts"].cpu().numpy(),[0,.5,.9,1]).tolist(),
        "base_training":base_history,"joint_fit_history":joint_history,
        "calibration":{"fitted_temperature":fitted_temperature,"g13_historical_surrogate_coefficients":theta_hist,
                       "g13_full_nll_coefficients":theta,
                       "unified_frozen_gate":frozen_head.gate.state_dict()["weight"].detach().cpu().tolist()},
        "evaluation_role":evaluation_role,
        "dev_smoke_metrics" if args.smoke else "test_metrics":{**base_result,**joint_result},
        "limitations":[
            "These TEST labels are newly held out by this script but raw dataset families have historical exposure.",
            "Single-member FIT training includes original V17 retrieval targets; joint stage excludes query's own anchor only.",
            "Within-FIT leave-self-out is not wholly unseen-site meta-learning: base encoder saw all FIT sites.",
            "The unified reliability gate is a testable low-dimensional mechanism, not a proven calibrated posterior.",
            "No architecture/parameter may be selected based on test_metrics; all hypotheses frozen prior to run.",
            "The G13 score is a composite likelihood of the same RSS, not independent sensor data.",
            "A true neighborhood is an audit-only label-derived quantity; never an inference input.",
        ],
        "config":vars(args),
    }
    (output/"RESULT.json").write_text(json.dumps(_json_safe(result),ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"SITE_DIAGNOSTICS.json").write_text(
        json.dumps(_json_safe({"evaluation_role":evaluation_role,
                                "audit_only_not_inference":True,
                                "by_model":site_diagnostics}),ensure_ascii=False),encoding="utf-8")
    torch.save({"base_model":base.state_dict(),"joint_model":joint.state_dict(),
                "frozen_head":frozen_head.state_dict(),"joint_head":joint_head.state_dict(),
                "fit_anchor_coords":memory["coords"],"fit_anchor_building":memory["building"],
                "fit_anchor_floor":memory["floor"],"reference_origin":origin,
                "ap_ids":data.ap_ids,"role_site_keys":{k:[keys[int(i)] for i in v] for k,v in roles.items() if (not args.smoke or k!="test")},
                "observation_memory":{k:v.cpu() for k,v in obs.items()},"config":vars(args)},
               output/"CHECKPOINTS.pt")
    metric_key = "dev_smoke_metrics" if args.smoke else "test_metrics"
    print(json.dumps({"output":str(output),"evaluation_role":evaluation_role,
                      metric_key:result[metric_key]},ensure_ascii=False,indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,choices=("uji","uts","tuji"))
    p.add_argument("--output",required=True)
    p.add_argument("--data-root",default=None)
    p.add_argument("--device",default="auto")
    p.add_argument("--seed",type=int,default=17301)
    p.add_argument("--split-seed",type=int,default=20261016)
    p.add_argument("--base-epochs",type=int,default=80)
    p.add_argument("--patience",type=int,default=12)
    p.add_argument("--joint-epochs",type=int,default=10)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--support",choices=("all","topk"),default="all")
    p.add_argument("--k",type=int,default=192)
    p.add_argument("--es-draws",type=int,default=32)
    p.add_argument("--smoke",action="store_true",help="DEV-only diagnostics; do not evaluate TEST")
    args = p.parse_args()
    if args.base_epochs<1 or args.joint_epochs<1 or args.batch<1 or args.k<2:
        p.error("invalid training, batch, or candidate size")
    run(args)


if __name__ == "__main__":
    main()