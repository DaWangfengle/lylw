"""G4 UTS *TRAINING-FILE ONLY* prospective observable-support pilot.

Question: Can posterior-weighted train-anchor support predict WHEN frozen
G1 gamma=.70 helps, and does calibration-only selective application pass
a physically disjoint internal confirmation? This is NOT a new external test
or a causal training-anchor-density intervention.

5 original V17 R0 members; freeze coordinate split before fitting. One
calibration-based threshold and one locked confirmation. Never read
UTS_test.csv, UJI outer/official, Tampere or TUJI test files.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from fhm_dapp_tim.fhm_dapp import uji_v17_g3_uts as g3
from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as r0
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.anchor import load_anchor_checkpoint, location_keys
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _save_torch_new, _sha256_file, _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA_TRAIN = REPO / "data/UTSIndoorLoc/UTS_training.csv"
EVIDENCE = ROOT / "evidence/v17_g4_observable_support"
MODELS = ROOT / "models/v17_g4_observable_support"
DESIGN = EVIDENCE / "DESIGN_V17_G4.md"
MANIFEST = EVIDENCE / "manifest_v17_g4.json"
CALIBRATION = EVIDENCE / "calibration_v17_g4.json"
CONFIRM_LOCK = EVIDENCE / "internal_confirmation.lock"
CONFIRMATION = EVIDENCE / "confirmation_v17_g4.json"
REPORT = EVIDENCE / "RUN_REPORT_V17_G4.md"

PROTOCOL = "v17_g4_uts_train_internal_observable_support_v1_20261009"
SPLIT_SEED = 19401
SCORING_SEED = 19402
BOOTSTRAP_SEED = 19403
PANEL = 5
SEEDS = g1.SEEDS[:PANEL]
EPOCHS = g1.EPOCHS[:PANEL]
GAMMA = 0.70
RADIUS_M = 10.0
L = 192
K = 192
ALPHA = 1.0
CAL_QUANTILES = (0.25, 0.50, 0.75)
CAL_MIN_GAIN_M = 0.05
CAL_ES_TOL_M = 0.02
CONF_MIN_GAIN_M = 0.15
BOOTSTRAP_REPS = 5000
BATCH = 24


class G4Error(RuntimeError):
    pass


def _read(path: Path, field: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get(field) != _canonical_json_sha256(data, omit=field):
        raise G4Error(f"failed {field} self hash: {path}")
    return data


def _write(path: Path, data: dict, field: str = "result_sha256") -> dict:
    if path.exists():
        raise FileExistsError(path)
    content = dict(data)
    content[field] = _canonical_json_sha256(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_new(path, content)
    return content


def _source_hashes() -> dict:
    paths = {
        "training_csv": DATA_TRAIN,
        "design": DESIGN,
        "g4_code": Path(__file__),
        "g3_adapter_code": Path(g3.__file__),
        "v17_training": Path(r0.__file__),
        "v17_score": Path(r3.__file__),
        "g1_density": Path(g1.__file__),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def member_path(i: int) -> Path:
    if not 1 <= int(i) <= PANEL:
        raise G4Error("G4 member index must be 1..5")
    return MODELS / f"member_{i:02d}_seed{SEEDS[i - 1]}.pt"


def split_keys(keys: Sequence[tuple], seed: int = SPLIT_SEED) -> dict[str, set[tuple]]:
    """Physical-site-disjoint 65% fit / ~17.5% cal / rest confirmation.

    Sort by stable SHA256 to avoid Python process hash randomization. Always
    keep at least one site/stratum per role when the stratum permits it.
    """
    strata = defaultdict(set)
    for key in keys:
        if len(key) != 4:
            raise G4Error("coordinate key shape differs")
        strata[(key[0], key[1])].add(tuple(key))
    roles = {"fit": set(), "calibration": set(), "confirmation": set()}
    for sf, sites in sorted(strata.items()):
        ordered = sorted(sites, key=lambda k: (
            hashlib.sha256(f"{seed}/{k!r}".encode("utf-8")).digest(), k
        ))
        n = len(ordered)
        if n < 6:
            raise G4Error(f"too few physical anchors to split stratum {sf}: {n}")
        n_fit = max(1, min(n - 2, int(math.floor(0.65 * n))))
        n_cal = max(1, min(n - n_fit - 1, (n - n_fit) // 2))
        roles["fit"].update(ordered[:n_fit])
        roles["calibration"].update(ordered[n_fit:n_fit+n_cal])
        roles["confirmation"].update(ordered[n_fit+n_cal:])
    if len(set.union(*roles.values())) != len(set(keys)):
        raise G4Error("role union mismatch")
    if any(roles[a] & roles[b] for a,b in (
        ("fit", "calibration"), ("fit", "confirmation"),
        ("calibration", "confirmation")
    )):
        raise G4Error("physical-coordinate leakage between roles")
    return roles


def rows_for_role(data: dict, roles: dict, role: str) -> np.ndarray:
    sites = roles[role]
    rows = np.asarray([i for i,k in enumerate(data["keys"]) if k in sites],dtype=np.int64)
    if len(rows) < 2 or set(data["keys"][int(k)] for k in rows) != sites:
        raise G4Error(f"non-exhaustive/missing rows for {role}")
    return rows


def _data_roles() -> tuple[dict,dict[str,set[tuple]],dict[str,np.ndarray]]:
    data, _ = g3.training_data()  # calls the canonical UTS *train* loader only
    roles = split_keys(data["keys"])
    rowmap = {r:rows_for_role(data,roles,r) for r in roles}
    if sum(len(v) for v in rowmap.values()) != len(data["keys"]):
        raise G4Error("incomplete training-file role partition")
    if len(roles["fit"]) < K:
        raise G4Error("fewer than K192 fitted anchors")
    return data, roles, rowmap


def preflight() -> dict:
    data, roles, rowmap = _data_roles()
    return {
        "status": "UTS_TRAIN_ONLY_READY",
        "protocol": PROTOCOL,
        "source_sha256": _source_hashes(),
        "training_rows": len(data["keys"]),
        "role_groups": {k:len(v) for k,v in roles.items()},
        "role_rows": {k:len(v) for k,v in rowmap.items()},
        "coordinates_disjoint": True,
        "panel": PANEL, "seeds": list(SEEDS), "epochs": list(EPOCHS),
        "gamma": GAMMA, "radius_m": RADIUS_M,
        "predictive_budget": {"E":PANEL,"L":L,"K":K,"alpha":ALPHA},
        "provided_test_files_accessed": False,
    }


def prepare() -> dict:
    info=preflight()
    data, roles, _ = _data_roles()
    recipe = {
        "protocol":PROTOCOL, "scientific_question":"Observable posterior support predicts fixed G1 benefit?",
        "source_sha256":info["source_sha256"],
        "coordinate_keys": {k:[list(x) for x in sorted(v)] for k,v in roles.items()},
        "role_rows":info["role_rows"],
        "role_groups":info["role_groups"],
        "panel":PANEL,"seeds":list(SEEDS),"epochs":list(EPOCHS),
        "fixed_gamma":GAMMA,"density_radius_m":RADIUS_M,
        "L":L,"K":K,"alpha":ALPHA,"score_seed":SCORING_SEED,
        "bootstrap_seed":BOOTSTRAP_SEED,"bootstrap_repetitions":BOOTSTRAP_REPS,
        "calibration_quantiles":list(CAL_QUANTILES),
        "calibration_min_gain_m":CAL_MIN_GAIN_M,"calibration_es_tolerance_m":CAL_ES_TOL_M,
        "confirmation_gain_threshold_m":CONF_MIN_GAIN_M,
        "fixed_v17_training":copy.deepcopy(FROZEN_TRAINING),
        "no_provided_external_test_access":True,
    }
    recipe["manifest_sha256"] = _canonical_json_sha256(recipe)
    if MANIFEST.exists():
        stored=_read(MANIFEST,"manifest_sha256")
        if stored != recipe:
            raise G4Error("immutable G4 scientific manifest or sources changed")
        return stored
    _write_json_new(MANIFEST,recipe)
    return recipe


def train_member(i: int,device_name: str="cuda") -> dict:
    if CALIBRATION.exists() or CONFIRM_LOCK.exists() or CONFIRMATION.exists():
        raise G4Error("training forbidden after calibration/confirmation")
    manifest=prepare()
    path=member_path(i)
    if path.exists():
        item=torch.load(path,map_location="cpu",weights_only=True)
        if item.get("manifest_sha256")!=manifest["manifest_sha256"]:
            raise G4Error("existing member mismatched manifest")
        return {"member":i,"sha256":_sha256_file(path),"reused":True}
    data,_,rows=_data_roles()
    started=time.perf_counter()
    model,stats,origin,mapping,history=r0._fit_refit(
        data,rows["fit"],seed=SEEDS[i-1],epochs=EPOCHS[i-1],
        device=torch.device(device_name),log_every=10
    )
    if int(model.n_anchors)!=manifest["role_groups"]["fit"]:
        raise G4Error("fitted model anchors do not match physical fit groups")
    config=copy.deepcopy(FROZEN_TRAINING)
    config.update({
        "training_protocol_version":PROTOCOL,"model_schema":"v17_g4_five_member_pilot",
        "n_devices":len(mapping),"train_all":True,
        "epochs":EPOCHS[i-1],"lr":FROZEN_TRAINING["learning_rate"],
        "validation_fraction":0.,"calibration_fraction":0.,
    })
    payload={
        "protocol":PROTOCOL,"manifest_sha256":manifest["manifest_sha256"],
        "member_index":i,"member_seed":SEEDS[i-1],"epoch":EPOCHS[i-1],
        "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "config":config,"stats":stats,
        "coord_origin":np.asarray(origin,dtype=np.float32),
        "device_id_mapping":{str(k):v for k,v in mapping.items()},
        "split_contract":{"train_source":"UTS_training.csv",
                         "fit_coordinates_only":True,
                         "test_files_accessed":False},
        "refit_history":history,
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    _save_torch_new(path,payload)
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return {"member":i,"sha256":_sha256_file(path),
            "seconds":time.perf_counter()-started,"reused":False}


def _load_models(device_name: str, manifest: dict):
    models=[];hashes=[];origins=[]
    for i in range(1,PANEL+1):
        path=member_path(i)
        if not path.is_file():raise G4Error(f"member {i} missing")
        model,ckpt=load_anchor_checkpoint(path,torch.device(device_name))
        if (ckpt.get("manifest_sha256")!=manifest["manifest_sha256"] or
            ckpt.get("member_seed")!=SEEDS[i-1] or
            ckpt.get("epoch")!=EPOCHS[i-1]):
            raise G4Error(f"member {i} wrong scientific identity")
        models.append(model.eval());hashes.append(_sha256_file(path))
        origins.append(np.asarray(ckpt["coord_origin"],dtype=np.float32))
    ref=models[0]
    for m,origin in zip(models,origins):
        if not (torch.equal(m.anchor_coords,ref.anchor_coords) and
                torch.equal(m.anchor_buildings,ref.anchor_buildings) and
                torch.equal(m.anchor_floors,ref.anchor_floors) and
                torch.equal(m.prototype_rss,ref.prototype_rss) and
                np.array_equal(origin,origins[0])):
            raise G4Error("the five models use inconsistent physical training support")
    if ref.n_anchors!=manifest["role_groups"]["fit"]:
        raise G4Error("anchor count changed")
    return models,hashes,origins[0]


def check_members() -> dict:
    manifest=prepare()
    complete=[]
    for i in range(1,PANEL+1):
        if not member_path(i).exists():
            return {"all_complete":False,"complete":len(complete),"first_missing":i}
        c=torch.load(member_path(i),map_location="cpu",weights_only=True)
        if c.get("manifest_sha256")!=manifest["manifest_sha256"] or c.get("member_index")!=i:
            raise G4Error(f"member {i} manifest or ID changed")
        complete.append({"member":i,"sha256":_sha256_file(member_path(i))})
    return {"all_complete":True,"complete":len(complete),"members":complete}


def posterior_observable(prior: torch.Tensor, log_density: torch.Tensor) -> torch.Tensor:
    """Only test-time prior over fitted anchors plus FIT-anchor density; no labels."""
    if prior.ndim!=3 or prior.shape[-1]!=len(log_density):
        raise G4Error("posterior/support dimensions mismatch")
    avg=prior.mean(dim=0)
    value=(avg*log_density.to(device=avg.device,dtype=avg.dtype)[None,:]).sum(dim=-1)
    if not torch.isfinite(value).all():
        raise G4Error("nonfinite observable support")
    return value


def mixture_prior(prior:torch.Tensor,log_density:torch.Tensor,
                  *, mode:str,observable:torch.Tensor|None=None,
                  threshold:float|None=None) -> torch.Tensor:
    if mode=="baseline":return prior
    corrected=g1.corrected_probability(prior,log_density,GAMMA)
    if mode=="g1":return corrected
    if mode=="placebo":
        raise G4Error("placebo has a separate precomputed shuffled log density")
    if mode=="selective":
        if observable is None or threshold is None or not math.isfinite(threshold):
            raise G4Error("selective action requires precomputed label-free threshold")
        mask=(observable<=threshold)[None,:,None]
        return torch.where(mask,corrected,prior)
    raise G4Error(f"unknown policy {mode}")


def _cache_role(models, data:dict, indices:np.ndarray, device:torch.device):
    """Compute shared base logits/components once; all alternatives reuse it."""
    ref=models[0]
    n=len(indices)
    anchors=ref.n_anchors
    logd=g1.support_log_density(
        ref.anchor_coords.detach().cpu(),ref.anchor_buildings.detach().cpu(),
        ref.anchor_floors.detach().cpu(),radius_m=RADIUS_M
    )
    shuffled=g1.placebo_log_density(
        logd,ref.anchor_buildings.detach().cpu(),ref.anchor_floors.detach().cpu(),
        seed=g1.PLACEBO_SEED,
    )
    logits=torch.empty((PANEL,n,anchors),dtype=torch.float32)
    locations=torch.empty((PANEL,n,anchors,2),dtype=torch.float32)
    scales=torch.empty_like(locations)
    features=torch.empty(n,dtype=torch.float64)
    for start in range(0,n,BATCH):
        stop=min(start+BATCH,n);index=indices[start:stop]
        rss=data["rss"][index].float().to(device)
        d=r2b._prototype_distance(ref,rss)
        coords=torch.arange(anchors,device=device)[None].expand(len(index),-1)
        neur=[];locs=[];scal=[]
        for model in models:
            emb=model.encoder(rss)
            neur.append(model.retrieval_logits_from_embedding(emb))
            location,scale=model._components(emb,coords)
            locs.append(location);scal.append(scale)
        prior=torch.softmax(
            r3p.BETA_NEURAL*torch.stack(neur)-r3p.BETA_PROTOTYPE*d[None,:,:],
            dim=2
        )
        logits[:,start:stop]=prior.detach().cpu()
        locations[:,start:stop]=torch.stack(locs).detach().cpu()
        scales[:,start:stop]=torch.stack(scal).detach().cpu()
        features[start:stop]=posterior_observable(prior,logd).double().detach().cpu()
    return {"prior":logits,"location":locations,"scale":scales,
            "support":features,"logdensity":logd,"shuffled":shuffled}


def _group_means(rows:np.ndarray, data:dict, values:np.ndarray):
    keymap=defaultdict(list)
    for i,idx in enumerate(rows):
        keymap[data["keys"][int(idx)]].append(float(values[i]))
    return {k:float(np.mean(v)) for k,v in keymap.items()}


def select_calibration_policy(metrics:dict,feature_by_group:dict) -> dict:
    """Calibrate threshold + action using calibration labels only.

    Candidate set fixed: baseline, global gamma, bottom 25/50/75% of
    baseline posterior expected log anchor density. Select smallest MAE
    IF it beats calibration baseline by at least .05m and does not worsen
    ES by >.02m; otherwise fall back to baseline. Tie is deterministic.
    """
    base=metrics["baseline"]
    admissible=[]
    for name in ("g1","gate_q25","gate_q50","gate_q75"):
        m=metrics[name]
        if (m["mae_2d_m"] <= base["mae_2d_m"]-CAL_MIN_GAIN_M and
                m["energy_score_2d_m"] <= base["energy_score_2d_m"]+CAL_ES_TOL_M):
            admissible.append(name)
    candidate=min(admissible,key=lambda x:(metrics[x]["mae_2d_m"],x)) if admissible else "baseline"
    thresholds={f"gate_q{int(q*100)}":float(np.quantile(
        list(feature_by_group.values()),q)) for q in CAL_QUANTILES}
    return {"policy":candidate,"threshold":thresholds.get(candidate),
            "thresholds":thresholds,"feasible_policies":admissible,
            "fallback_baseline":not bool(admissible),
            "calibration_rule_frozen":True}


@torch.no_grad()
def _evaluate_role(role:str,models,data:dict,rows:np.ndarray,
                   origin:np.ndarray,selected:dict|None=None):
    device=next(models[0].parameters()).device
    cache=_cache_role(models,data,rows,device)
    features=cache["support"].numpy()
    feature_by_group=_group_means(rows,data,features)
    if role=="calibration":
        thresholds={f"gate_q{int(q*100)}":float(np.quantile(
            list(feature_by_group.values()),q)) for q in CAL_QUANTILES}
        variants=("baseline","g1","placebo","gate_q25","gate_q50","gate_q75")
    else:
        if selected is None:raise G4Error("confirmation requires frozen choice")
        thresholds={selected["policy"]:selected["threshold"]} if selected["threshold"] is not None else {}
        variants=tuple(dict.fromkeys(("baseline","g1","placebo",selected["policy"])))
    n=len(rows)
    gid,gkeys=r2b._coordinate_groups([data["keys"][int(x)] for x in rows])
    target=torch.from_numpy(
        data["raw_xy"][rows]-np.asarray(origin,dtype=np.float64)[None,:]
    ).double().to(device)
    rng=r3.common_student_t_randomness(
        n,draws=r3p.PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=r3p.STUDENT_DF,seed=SCORING_SEED
    )
    rng={k:(v.to(device) if isinstance(v,torch.Tensor) else v)
         for k,v in rng.items()}
    row_mae={k:torch.empty(n,dtype=torch.float64) for k in variants}
    row_es={k:torch.empty(n,dtype=torch.float64) for k in variants}
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0,n,BATCH):
            stop=min(start+BATCH,n)
            p=cache["prior"][:,start:stop].to(device)
            loc=cache["location"][:,start:stop].to(device)
            scale=cache["scale"][:,start:stop].to(device)
            obs=cache["support"][start:stop].to(device)
            random_slice={k:(v[start:stop] if isinstance(v,torch.Tensor)
                           and v.ndim>0 else v) for k,v in rng.items()}
            for name in variants:
                if name=="placebo":
                    p1=g1.corrected_probability(p,cache["shuffled"],GAMMA)
                elif name.startswith("gate_q"):
                    p1=mixture_prior(p,cache["logdensity"],mode="selective",
                                    observable=obs,threshold=thresholds[name])
                else:
                    p1=mixture_prior(p,cache["logdensity"],mode=name)
                top,ids,radial,_=r3._member_top_l(p1,loc,scale,L,
                                                  student_df=r3p.STUDENT_DF)
                law=r3._dense_joint_law(
                    top,ids,radial,loc,scale,tuple(range(PANEL)),
                    top_k=K,alpha=ALPHA
                )
                energy,error=r3._score_joint_law(
                    *law,target[start:stop],random_slice)
                row_mae[name][start:stop]=error.detach().cpu()
                row_es[name][start:stop]=energy.detach().cpu()
    metrics={}; vectors={}
    for name in variants:
        m,mv=r3._metric_record(row_mae[name],gid,len(gkeys))
        e,ev=r3._metric_record(row_es[name],gid,len(gkeys))
        metrics[name]={"mae_2d_m":m["coordinate_macro_m"],
                       "energy_score_2d_m":e["coordinate_macro_m"],
                       "row_mae_m":m["row_mean_m"],
                       "row_es_m":e["row_mean_m"]}
        vectors[name]={"mae":mv.tolist(),"es":ev.tolist()}
    feature_vector=np.asarray([feature_by_group[key] for key in gkeys])
    gain=np.asarray(vectors["baseline"]["mae"])-np.asarray(vectors["g1"]["mae"])
    corr=spearmanr(feature_vector,gain).statistic
    if not math.isfinite(float(corr)):corr=None
    return {"role":role,"coordinate_keys":[list(x) for x in gkeys],
            "metrics":metrics,"coordinate_vectors":vectors,
            "observable_expected_log_support_by_coordinate":feature_vector.tolist(),
            "observable_support_vs_fixed_g1_gain_spearman":corr,
            "thresholds_applied":thresholds,"rows":n,"coordinates":len(gkeys),
            "role_contains_no_training_coordinate":True}


def calibrate(device_name:str="cuda") -> dict:
    if CALIBRATION.exists() or CONFIRM_LOCK.exists() or CONFIRMATION.exists():
        raise G4Error("calibration already written or confirmation already accessed")
    manifest=prepare();members=check_members()
    if not members["all_complete"]:raise G4Error("all five models required")
    models,hashes,origin=_load_models(device_name,manifest)
    data,roles,rows=_data_roles()
    result=_evaluate_role("calibration",models,data,rows["calibration"],origin)
    features=dict(zip(
        map(tuple,result["coordinate_keys"]),
        result["observable_expected_log_support_by_coordinate"]
    ))
    selection=select_calibration_policy(result["metrics"],features)
    result.update({
        "protocol":PROTOCOL,"manifest_sha256":manifest["manifest_sha256"],
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "member_sha256":hashes,"selected_policy":selection,
        "confirmation_scores_accessed":False,
        "external_test_scores_accessed":False,
        "warning":"G3 UTS official test previously opened, but this calibration uses ONLY UTS_training."
    })
    return _write(CALIBRATION,result)


def confirm(device_name:str="cuda") -> dict:
    if CONFIRM_LOCK.exists() or CONFIRMATION.exists() or REPORT.exists():
        raise G4Error("G4 confirmation attempted; no retuning or rescore")
    manifest=prepare();cal=_read(CALIBRATION,"result_sha256")
    if cal["manifest_sha256"]!=manifest["manifest_sha256"]:
        raise G4Error("calibration/manifest source mismatch")
    models,hashes,origin=_load_models(device_name,manifest)
    if hashes!=cal["member_sha256"]:
        raise G4Error("members changed after calibration")
    # This internal confirmation is a one-shot check; never open external test.
    lock=_write(CONFIRM_LOCK,{
        "protocol":PROTOCOL,"calibration_file_sha256":_sha256_file(CALIBRATION),
        "selected_policy":cal["selected_policy"],
        "manifest_sha256":manifest["manifest_sha256"],
        "member_sha256":hashes,"confirmation_opened_before_lock":False,
    })
    data,_,rows=_data_roles()
    result=_evaluate_role(
        "confirmation",models,data,rows["confirmation"],origin,
        selected=cal["selected_policy"]
    )
    policy=cal["selected_policy"]["policy"]
    diffs={
        "mae":np.asarray(result["coordinate_vectors"][policy]["mae"])
             -np.asarray(result["coordinate_vectors"]["baseline"]["mae"]),
        "es":np.asarray(result["coordinate_vectors"][policy]["es"])
             -np.asarray(result["coordinate_vectors"]["baseline"]["es"]),
    }
    if policy=="baseline":
        inference={"status":"baseline_selected_no_candidate_effect",
                   "comparisons":{"mae":{"simultaneous_upper_bound_m":0.0},
                                  "es":{"simultaneous_upper_bound_m":0.0}}}
    else:
        inference=r3.studentized_max_t_family(
            diffs,repetitions=BOOTSTRAP_REPS,
            confidence=.95,seed=BOOTSTRAP_SEED,
            orientation="calibration_frozen_policy_minus_V17_5member_baseline"
        )
    dmae=float(diffs["mae"].mean())
    des=float(diffs["es"].mean())
    pass_gate=(policy!="baseline" and dmae<=-CONF_MIN_GAIN_M and
        inference["comparisons"]["mae"]["simultaneous_upper_bound_m"]<0 and
        inference["comparisons"]["es"]["simultaneous_upper_bound_m"]<=0)
    result.update({
        "protocol":PROTOCOL,
        "status":"TRAINING_FILE_INTERNAL_CONFIRMATION_COMPLETE",
        "selected_policy":cal["selected_policy"],
        "decision":"PROMISING_INTERNAL_PILOT" if pass_gate else "NO_INTERNAL_PROMOTION",
        "confirmation_gate":{
            "passed":pass_gate,"delta_mae_m":dmae,"delta_es_m":des,
            "min_gain_m":CONF_MIN_GAIN_M,
            "mae_simultaneous_upper_m":inference["comparisons"]["mae"]["simultaneous_upper_bound_m"],
            "es_simultaneous_upper_m":inference["comparisons"]["es"]["simultaneous_upper_bound_m"],
        },
        "inference":inference,
        "calibration_file_sha256":_sha256_file(CALIBRATION),
        "confirmation_lock_sha256":_sha256_file(CONFIRM_LOCK),
        "confirmation_lock_result_sha256":lock["result_sha256"],
        "member_sha256":hashes,
        "G3_original_external_decision":"TRANSFER_NOT_ESTABLISHED_ON_UTS",
        "other_external_test_data_used":False,
        "warning":"This is an exploratory UTS TRAIN-file pilot: not independent of G3 method history and not a Q1 generalization claim.",
    })
    result=_write(CONFIRMATION,result)
    REPORT.write_text(make_report(cal,result),encoding="utf-8",newline="\n")
    return {"status":result["status"],"decision":result["decision"],
            "selected_policy":policy,"delta_mae_m":dmae,
            "delta_es_m":des,"result_sha256":result["result_sha256"]}


def make_report(cal:dict,conf:dict)->str:
    selection=cal["selected_policy"]; selected=selection["policy"]
    c=conf["confirmation_gate"]
    out=[
        "# V17-G4 | UTS training-only observable support pilot",
        "",
        "UTS_training.csv **only**. UTS_test.csv, Tampere/TUJI provided tests "
        "and UJI outer/official are forbidden for this pilot.",
        "",
        "**Design:** same 5 frozen V17 members, 3 physically disjoint coordinate roles; "
        "query-observable posterior expected log anchor support; "
        "calibration-selected threshold/identity; internal one-shot confirmation.",
        "",
        f"Calibration selection: **{selected}**; threshold={selection['threshold']}",
        f"Calibration fixed G1 support-vs-gain Spearman: "
        f"{cal['observable_support_vs_fixed_g1_gain_spearman']}",
        f"Confirmation fixed G1 support-vs-gain Spearman: "
        f"{conf['observable_support_vs_fixed_g1_gain_spearman']}",
        "",
        "| Confirmation variant | Coordinate macro MAE (m) | Energy Score (m) |",
        "|---|---:|---:|"
    ]
    for name in conf["metrics"]:
        rec=conf["metrics"][name]
        out.append(f"| {name} | {rec['mae_2d_m']:.6f} | "
                   f"{rec['energy_score_2d_m']:.6f} |")
    out += [
        "",f"Selected policy minus baseline MAE: {c['delta_mae_m']:+.6f}m",
        f"Selected policy minus baseline ES: {c['delta_es_m']:+.6f}m",
        f"One-shot internal gate: {c['passed']}; **{conf['decision']}**",
        "",
        "All thresholds selected only on CALIBRATION coordinates, not confirmation labels. "
        "Rank correlation is descriptive and not causal proof.",
        "Five members are a screening pilot, not directly comparable to 25-member G3.",
        "This study does not rehabilitate G1's original STOP or reverse G3 UTS external STOP.",
        "A positive internal pilot requires new independent verification; "
        "no further UTS_test tuning or scoring is authorized.",
        "",f"Result self hash: {conf['result_sha256']}",""
    ]
    return "\n".join(out)


def check() -> dict:
    manifest=prepare()
    members=check_members()
    if not members["all_complete"]:raise G4Error("five members missing")
    if not CALIBRATION.exists():
        return {"status":"PRE_CALIBRATION","members":PANEL}
    cal=_read(CALIBRATION,"result_sha256")
    if cal["manifest_file_sha256"]!=_sha256_file(MANIFEST):
        raise G4Error("calibration source binding failed")
    if cal["member_sha256"]!=[m["sha256"] for m in members["members"]]:
        raise G4Error("members changed after calibration")
    if not CONFIRM_LOCK.exists():
        return {"status":"CALIBRATED_NOT_CONFIRMED",
                "selected_policy":cal["selected_policy"]["policy"]}
    lock=_read(CONFIRM_LOCK,"result_sha256")
    if lock["calibration_file_sha256"]!=_sha256_file(CALIBRATION):
        raise G4Error("internal lock/calibration binding mismatch")
    if not CONFIRMATION.exists():
        return {"status":"CONFIRM_LOCK_EXISTS_WITHOUT_RESULT_STOP"}
    conf=_read(CONFIRMATION,"result_sha256")
    if (conf["confirmation_lock_sha256"]!=_sha256_file(CONFIRM_LOCK) or
        conf["member_sha256"]!=cal["member_sha256"] or
        conf["selected_policy"]!=cal["selected_policy"]):
        raise G4Error("confirmation provenance or policy changed")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8")!=make_report(cal,conf):
        raise G4Error("G4 report changed")
    return {"status":"VERIFIED","decision":conf["decision"],
            "result_sha256":conf["result_sha256"]}


def main(argv:Sequence[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("preflight","prepare","train-member","check-members","calibrate","confirm","check"):
        s=sub.add_parser(name)
        if name in ("train-member","calibrate","confirm"):
            s.add_argument("--device",default="cuda")
        if name=="train-member":s.add_argument("--member-index",type=int,required=True)
    a=p.parse_args(argv)
    if a.command=="preflight":out=preflight()
    elif a.command=="prepare":out=prepare()
    elif a.command=="train-member":out=train_member(a.member_index,a.device)
    elif a.command=="check-members":out=check_members()
    elif a.command=="calibrate":out=calibrate(a.device)
    elif a.command=="confirm":out=confirm(a.device)
    else:out=check()
    print(json.dumps(out,indent=2,allow_nan=False,default=str),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
