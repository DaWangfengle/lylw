"""G8-A: exact member-vs-anchor probability temperature attribution.

NO training, NO temperature search, NO blind-test claims. Reuse previously
observed G6 TUJI, G4 UTS and G5 Tampere internal roles, original E5
checkpoints, Student-t law, and G6 scoring common random numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g4_observable_support as g4
from fhm_dapp_tim.fhm_dapp import uji_v17_g5_bayes_action as g5
from fhm_dapp_tim.fhm_dapp import uji_v17_g6_calibration as g6
from fhm_dapp_tim.fhm_dapp import uji_v17_g7_frozen_transfer as g7
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import common_student_t_randomness
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _sha256_file, _write_json_new,
)

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"evidence/v17_g8_hierarchical_attribution"
DESIGN=EVIDENCE/"DESIGN_V17_G8.md"
MANIFEST=EVIDENCE/"manifest_v17_g8.json"
LOCK=EVIDENCE/"attribution_v17_g8.lock"
RESULT=EVIDENCE/"attribution_v17_g8.json"
REPORT=EVIDENCE/"RUN_REPORT_V17_G8.md"
PROTOCOL="v17_g8_member_anchor_temperature_attribution_20261010"
DOMAINS=("tuji","uts","tampere")
ROLES={"tuji":("calibration","confirmation"),
       "uts":("calibration","confirmation"),
       "tampere":("confirmation",)}
VARIANTS=("base","member","anchor","both")
T=0.6
E=5
K=192
BATCH=16
SEED=g6.SCORE_SEED
DF=r3p.STUDENT_DF
TOL=3e-6


class G8Error(RuntimeError):
    pass


def _hashed(record:dict,field:str="result_sha256")->dict:
    obj=dict(record)
    obj[field]=_canonical_json_sha256(obj)
    return obj


def _write(path:Path,record:dict,field:str="result_sha256")->dict:
    if path.exists():
        raise FileExistsError(f"G8 immutable file exists: {path}")
    obj=_hashed(record,field)
    path.parent.mkdir(parents=True,exist_ok=True)
    _write_json_new(path,obj)
    return obj


def _read(path:Path,field:str="result_sha256")->dict:
    obj=json.loads(path.read_text(encoding="utf-8"))
    if obj.get(field)!=_canonical_json_sha256(obj,omit=field):
        raise G8Error(f"G8 evidence hash changed: {path}")
    return obj


def _source_hashes()->dict:
    files={
        "design":DESIGN,"g8":Path(__file__),
        "g8_tests":ROOT.parent/"tests/test_uji_v17_g8_hierarchical_attribution.py",
        "g7":Path(g7.__file__),"g6":Path(g6.__file__),
        "g5":Path(g5.__file__),"g4":Path(g4.__file__),
        "r3":Path(r3.__file__),"r2b":Path(r2b.__file__),
        "r3_protocol":Path(r3p.__file__),
        "v9_sampler":Path(common_student_t_randomness.__code__.co_filename),
    }
    return {name:_sha256_file(path) for name,path in files.items()}


def _load_data(domain:str)->dict:
    """Only already-exposed TRAIN-internal roles; no historical loss JSON."""
    if domain in ("uts","tampere"):
        old=g7._data_and_old_protocol(domain) # checks original G4/G5 weights
        return old
    if domain!="tuji":
        raise G8Error("only three fixed historically observed training domains")
    if not g6.MANIFEST.exists() or not g6.CALIBRATION.exists():
        raise G8Error("G6 TUJI source manifest and CAL evidence are required")
    manifest=g6._read(g6.MANIFEST,"manifest_sha256")
    if g6._source_hashes()!=manifest["source_sha256"]:
        raise G8Error("G6 original code/data identity changed")
    cal=g6._read(g6.CALIBRATION)
    if (cal["manifest_sha256"]!=manifest["manifest_sha256"]
        or cal["selected_policy"]["policy"]!="T0.6_S0.5"):
        raise G8Error("G6 CAL source policy identity changed")
    data,groups,rows=g6._training() # scipy loads TUJI results_train variable only
    if (manifest["role_coordinate_keys"]!=
        {k:[list(v) for v in sorted(sites)] for k,sites in groups.items()}):
        raise G8Error("G6 TUJI original TRAIN role split changed")
    ck=[g6.member_path(i) for i in range(1,E+1)]
    sha=g7._hash_local_checkpoints(ck)
    if sha!=cal["member_sha256"]:
        raise G8Error("G6 member checkpoint bytes differ from frozen CAL")
    info={
        "historical_protocol":g6.PROTOCOL,
        "historical_manifest_sha256":_sha256_file(g6.MANIFEST),
        "historical_calibration_sha256":_sha256_file(g6.CALIBRATION),
        "checkpoints":sha,
        "anchor_count":len(groups["fit"]),
        "ap_width":int(data["rss"].shape[1]),
        "role_coordinate_keys":{r:[list(x) for x in sorted(groups[r])]
                                for r in ROLES["tuji"]},
        "role_rows":{r:len(rows[r]) for r in ROLES["tuji"]},
        "test_variable_deserialized":False,
        "old_confirmation_labels_previously_observed":True,
    }
    return {"data":data,"groups":groups,"rows":rows,
            "roles":ROLES["tuji"],"original":info}


def _pack(domain:str)->dict:
    p=_load_data(domain)
    if domain!="tuji":
        p["original"]["role_coordinate_keys"]={
            role:[list(x) for x in sorted(p["groups"][role])]
            for role in ROLES[domain]
        }
    # Role membership is checked again at actual scoring, not merely by row count.
    for role in ROLES[domain]:
        assert p["rows"][role].dtype.kind in "iu"
        if set(p["data"]["keys"][int(i)] for i in p["rows"][role])!=p["groups"][role]:
            raise G8Error(f"{domain}/{role}: changed physical site membership")
        if p["groups"][role]&p["groups"]["fit"]:
            raise G8Error(f"{domain}/{role}: FIT physical location leakage")
    return p


def preflight()->dict:
    scenarios={d:_pack(d)["original"] for d in DOMAINS}
    if (g6.r3p.PREDICTIVE_DRAWS_PER_STREAM!=64
        or g6.L!=192 or g6.K!=192 or g6.ALPHA!=1.0
        or g6.PANEL!=5 or DF!=4.0):
        raise G8Error("original E5/L192/K192 Student-t protocol changed")
    return {
        "protocol":PROTOCOL,"status":"THREE_ORIGINAL_E5_PANELS_READY",
        "source_sha256":_source_hashes(),"domains":scenarios,
        "temperature":T,"variants":list(VARIANTS),
        "L":g6.L,"K":K,"E":E,"alpha":g6.ALPHA,
        "draws_per_stream":r3p.PREDICTIVE_DRAWS_PER_STREAM,
        "score_seed":SEED,"all_test_labels_previously_seen":True,
        "official_provided_test_access":False,"new_training":False,
    }


def prepare()->dict:
    info=preflight()
    plan={
        "protocol":PROTOCOL,"source_sha256":info["source_sha256"],
        "model_and_role_provenance":info["domains"],
        "roles":{d:list(ROLES[d]) for d in DOMAINS},
        "variants":list(VARIANTS),"temperature":T,
        "E":E,"L":g6.L,"K":K,"alpha":g6.ALPHA,
        "df":DF,"draws_per_stream":info["draws_per_stream"],
        "score_seed":SEED,
        "no_candidate_selection":True,
        "no_training_no_new_test_access":True,
        "old_domain_confirmations_previously_viewed":True,
        "decision_rule":"descriptive primary layer and interaction decomposition only",
    }
    plan["manifest_sha256"]=_canonical_json_sha256(plan)
    if MANIFEST.is_file():
        old=_read(MANIFEST,"manifest_sha256")
        if old!=plan:raise G8Error("G8 frozen plan or sources changed")
        return old
    _write_json_new(MANIFEST,plan)
    return plan


def _tempered(x:torch.Tensor,temp:float)->torch.Tensor:
    if temp==1.:return x
    return torch.softmax(x.clamp_min(1e-30).log()/temp,dim=-1)


def hierarchical_laws(joint:torch.Tensor, members:int=E, anchors:int=K
                      )->dict[str,torch.Tensor]:
    """Factor independent member and anchor marginals from original R3 law.

    The BOTH law is scored with G6's *exact* published global temperature,
    and is independently compared against member^1/T x anchor^1/T.
    """
    if joint.ndim!=2 or joint.shape[1]!=members*anchors:
        raise G8Error("joint law must contain exactly E*K member-anchor entries")
    if not torch.isfinite(joint).all() or (joint<0).any():
        raise G8Error("invalid R3 mixture probabilities")
    n=len(joint);p=joint.reshape(n,members,anchors)
    member=p.sum(dim=2);anchor=p.sum(dim=1)
    reconstructed=member[:,:,None]*anchor[:,None,:]
    if not torch.allclose(p,reconstructed,atol=TOL,rtol=TOL):
        raise G8Error("R3 joint member*anchor factorization does not hold")
    if not torch.allclose(p.sum(dim=(1,2)),torch.ones(n,device=p.device,dtype=p.dtype),
                          atol=TOL,rtol=TOL):
        raise G8Error("joint law is not normalized")
    wm=_tempered(member,T);ca=_tempered(anchor,T)
    manual_both=(wm[:,:,None]*ca[:,None,:]).reshape(n,-1)
    exact_both=g6.calibrate_joint_law(
        joint,torch.zeros(n,members*anchors,2,device=joint.device,dtype=joint.dtype),
        torch.ones(n,members*anchors,2,device=joint.device,dtype=joint.dtype),
        T,1.
    )[0]
    if not torch.allclose(manual_both,exact_both,atol=TOL,rtol=TOL):
        raise G8Error("factorial BOTH does not reproduce G6 global temperature")
    return {
        "base":joint,
        "member":(wm[:,:,None]*anchor[:,None,:]).reshape(n,-1),
        "anchor":(member[:,:,None]*ca[:,None,:]).reshape(n,-1),
        "both":exact_both,
    }


def decompose_trace(q:torch.Tensor,loc:torch.Tensor,scale:torch.Tensor,
                    members:int=E,anchors:int=K)->dict[str,torch.Tensor]:
    """Exact trace law of total variance under original independent-axis t(df4).

    Two nesting directions are deliberately saved: a unique attribution of
    all between-component variance to 'member' vs 'anchor' is impossible
    without choosing conditioning order.
    """
    n=len(q);p=q.reshape(n,members,anchors).double()
    centres=loc.reshape(n,members,anchors,2).double()
    sigma=scale.reshape(n,members,anchors,2).double()
    if DF<=2:raise G8Error("Student-t variance requires df>2")
    wm=p.sum(dim=2);ca=p.sum(dim=1)
    centre=(p[...,None]*centres).sum(dim=(1,2))
    inner=(p*(DF/(DF-2.)*sigma.square().sum(dim=-1))).sum(dim=(1,2))
    total_between=(p*((centres-centre[:,None,None,:]).square().sum(dim=-1))).sum(dim=(1,2))
    average_by_anchor=(wm[:,:,None,None]*centres).sum(dim=1)
    anchor_between=(ca*((average_by_anchor-centre[:,None,:]).square().sum(dim=-1))).sum(dim=1)
    members_within_anchor=(p*((centres-average_by_anchor[:,None,:,:]).square().sum(dim=-1))).sum(dim=(1,2))
    average_by_member=(ca[:,None,:,None]*centres).sum(dim=2)
    member_between=(wm*((average_by_member-centre[:,None,:]).square().sum(dim=-1))).sum(dim=1)
    anchors_within_member=(p*((centres-average_by_member[:,:,None,:]).square().sum(dim=-1))).sum(dim=(1,2))
    if not (torch.allclose(total_between,anchor_between+members_within_anchor,atol=1e-5,rtol=1e-6)
            and torch.allclose(total_between,member_between+anchors_within_member,atol=1e-5,rtol=1e-6)):
        raise G8Error("hierarchical law-of-total-variance identity failed")
    return {
        "within_student_t_trace_m2":inner,
        "between_centres_trace_m2":total_between,
        "total_predictive_trace_m2":inner+total_between,
        "anchor_between_given_member_average_m2":anchor_between,
        "member_within_anchor_m2":members_within_anchor,
        "member_between_given_anchor_average_m2":member_between,
        "anchor_within_member_m2":anchors_within_member,
    }


def _load_models(domain:str,manifest:dict,device_name:str):
    if domain=="tuji":
        g6manifest=g6._read(g6.MANIFEST,"manifest_sha256")
        models,origin,hashes=g6._load_models(device_name,g6manifest)
    else:
        models,origin=g7._load_historical_models(
            domain,{"models_and_roles":manifest["model_and_role_provenance"]},
            device_name)
        hashes=g7._hash_local_checkpoints([
            g4.member_path(i) if domain=="uts" else g5.member_path(i)
            for i in range(1,E+1)])
    if hashes!=manifest["model_and_role_provenance"][domain]["checkpoints"]:
        raise G8Error("original panel sha mismatch")
    return models,origin


@torch.no_grad()
def score_role(models:list,data:dict,rows:np.ndarray,origin:np.ndarray)->dict:
    device=next(models[0].parameters()).device
    n=len(rows)
    keys=[data["keys"][int(i)] for i in rows]
    gid,unique=r2b._coordinate_groups(keys)
    target=torch.as_tensor(data["raw_xy"][rows]-origin[None,:],
                           dtype=torch.float64,device=device)
    rnd=common_student_t_randomness(n,draws=r3p.PREDICTIVE_DRAWS_PER_STREAM,
                                    degrees_freedom=DF,seed=SEED)
    loss={v:torch.empty(n,dtype=torch.float64) for v in VARIANTS}
    energy={v:torch.empty(n,dtype=torch.float64) for v in VARIANTS}
    record_trace={v:{name:torch.empty(n,dtype=torch.float64) for name in (
        "within_student_t_trace_m2","between_centres_trace_m2",
        "total_predictive_trace_m2","anchor_between_given_member_average_m2",
        "member_within_anchor_m2","member_between_given_anchor_average_m2",
        "anchor_within_member_m2")} for v in ("base","both")}
    ref=models[0]
    if int(ref.n_anchors)<K:raise G8Error("fewer than K192 original FIT anchors")
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0,n,BATCH):
            end=min(start+BATCH,n)
            ids=rows[start:end]
            rss=data["rss"][ids].float().to(device)
            distance=r2b._prototype_distance(ref,rss)
            indices=torch.arange(ref.n_anchors,device=device)[None,:].expand(end-start,-1)
            logits=[];locations=[];scales=[]
            for model in models:
                emb=model.encoder(rss)
                logits.append(model.retrieval_logits_from_embedding(emb))
                loc,scale=model._components(emb,indices)
                locations.append(loc);scales.append(scale)
            priors=torch.softmax(
                r3p.BETA_NEURAL*torch.stack(logits)
                -r3p.BETA_PROTOTYPE*distance[None,:,:],dim=-1)
            top,ix,rad,_=r3._member_top_l(
                priors,torch.stack(locations),torch.stack(scales),g6.L,
                student_df=DF)
            p,xy,sigma=r3._dense_joint_law(
                top,ix,rad,torch.stack(locations),torch.stack(scales),
                tuple(range(E)),top_k=K,alpha=g6.ALPHA)
            weights=hierarchical_laws(p,E,K)
            random=g6._random_slice(rnd,start,end,device)
            for name,q in weights.items():
                es,mae=r3._score_joint_law(q,xy,sigma,target[start:end],random)
                loss[name][start:end]=mae.double().detach().cpu()
                energy[name][start:end]=es.double().detach().cpu()
                if name in record_trace:
                    traces=decompose_trace(q,xy,sigma,E,K)
                    for col,value in traces.items():
                        record_trace[name][col][start:end]=value.detach().cpu()
    metric={};vectors={}
    for name in VARIANTS:
        mae,mvec=r3._metric_record(loss[name],gid,len(unique))
        es,evec=r3._metric_record(energy[name],gid,len(unique))
        metric[name]={"mae_2d_m":float(mae["coordinate_macro_m"]),
                      "energy_score_2d_m":float(es["coordinate_macro_m"]),
                      "row_mae_m":float(mae["row_mean_m"]),
                      "row_es_m":float(es["row_mean_m"])}
        vectors[name]={"mae":mvec.tolist(),"es":evec.tolist()}
    trace={}
    for policy,rec in record_trace.items():
        trace[policy]={}
        for label,value in rec.items():
            summary,mvec=r3._metric_record(value,gid,len(unique))
            trace[policy][label]={"mean":float(summary["coordinate_macro_m"]),
                                  "vector":mvec.tolist()}
    effects=factorial_effects(metric)
    return {"rows":n,"coordinate_groups":len(unique),
            "coordinate_keys":[list(k) for k in unique],
            "candidate_metrics":metric,"coordinate_vectors":vectors,
            "trace_decomposition":trace,"factorial_effects":effects,
            "monte_carlo_shared_random_draws":2*r3p.PREDICTIVE_DRAWS_PER_STREAM}


def factorial_effects(metrics:dict)->dict:
    result={}
    for key in ("mae_2d_m","energy_score_2d_m"):
        b=metrics["base"][key]
        mem=metrics["member"][key]-b
        anch=metrics["anchor"][key]-b
        both=metrics["both"][key]-b
        interaction=both-mem-anch
        result[key]={
            "base":float(b),"member_delta":float(mem),
            "anchor_delta":float(anch),"both_delta":float(both),
            "factorial_interaction":float(interaction),
            "member_fraction_of_both":float(mem/both) if abs(both)>1e-8 else None,
            "anchor_fraction_of_both":float(anch/both) if abs(both)>1e-8 else None,
        }
    return result


def _overview(result:dict)->str:
    lines=[
        "# G8-A | Member vs anchor probability sharpening: descriptive attribution",
        "",
        "**No training or new blind confirmation.** This uses already-viewed "
        "TRAIN-internal TUJI / UTS / Tampere roles. No p-values or model promotion.",
        "",
        "All four variants preserve original member centre/Student-t scale;",
        "T=0.6 is fixed, independently for member and anchor weights.",
        "Both-level factorized temperature must equal G6 whole-joint T=0.6",
        "up to documented floating-point tolerance.",
        "",
        "| Domain / historical role | Method | Macro MAE m | ES m |",
        "|---|---|---:|---:|",
    ]
    for domain in DOMAINS:
        for role,one in result["domains"][domain]["roles"].items():
            for name in VARIANTS:
                m=one["candidate_metrics"][name]
                lines.append(f"| {domain} / {role} | {name} | "
                             f"{m['mae_2d_m']:.6f} | "
                             f"{m['energy_score_2d_m']:.6f} |")
    lines.extend(["","## Factorial decomposition (historical confirmations only)",""])
    for domain in DOMAINS:
        fx=result["domains"][domain]["roles"]["confirmation"]["factorial_effects"]
        for score in ("mae_2d_m","energy_score_2d_m"):
            x=fx[score]
            lines.append(
                f"- {domain} {score}: member {x['member_delta']:+.6f}m; "
                f"anchor {x['anchor_delta']:+.6f}m; "
                f"both {x['both_delta']:+.6f}m; "
                f"interaction {x['factorial_interaction']:+.6f}m"
            )
    lines.extend([
        "",
        "Full per-coordinate effects and BOTH/BASE trace decompositions are in JSON.",
        "Both alternative nesting orders are reported for variance; splitting",
        "cross-level centre disagreement into unique causal shares is not justified.",
        "Descriptive contrasts do not establish attribution by intervention in",
        "the training process; they isolate the fixed R3 post-prediction operator.",
        "Negative-transfer locations, ES worsening and historical exposure persist.",
        "",
        f"Self hash: {result['result_sha256']}",""
    ])
    return "\n".join(lines)


def run(device_name:str="cuda")->dict:
    if LOCK.exists() or RESULT.exists():
        raise G8Error("G8 run already attempted: preserve lock, do not repeat")
    manifest=prepare()
    lock=_write(LOCK,{"protocol":PROTOCOL,
                     "manifest_file_sha256":_sha256_file(MANIFEST),
                     "historically_viewed_scoring_roles":True,
                     "four_prespecified_variants":list(VARIANTS)})
    out={}
    for domain in DOMAINS:
        scenario=_pack(domain)
        models,origin=_load_models(domain,manifest,device_name)
        scored={}
        for role in ROLES[domain]:
            scored[role]=score_role(models,scenario["data"],scenario["rows"][role],origin)
        out[domain]={"roles":scored,"provenance":manifest["model_and_role_provenance"][domain]}
        del models
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result=_write(RESULT,{
        "protocol":PROTOCOL,"status":"DESCRIPTIVE_FACTORIAL_ATTRIBUTION_COMPLETE",
        "domains":out,"temperature":T,"variants":list(VARIANTS),
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "lock_file_sha256":_sha256_file(LOCK),
        "lock_result_sha256":lock["result_sha256"],
        "blind_external_test":False,"retrained_networks":False,
        "warning":"All roles have previously exposed labels; diagnostic only. "
                  "No target-domain policy selection. Do not call this independent validation.",
    })
    REPORT.write_text(_overview(result),encoding="utf-8",newline="\n")
    return {"status":result["status"],"sha256":result["result_sha256"],
            "confirm_effects":{
                d:result["domains"][d]["roles"]["confirmation"]["factorial_effects"]
                for d in DOMAINS}}


def check()->dict:
    if not MANIFEST.exists():return {"status":"UNPREPARED"}
    manifest=prepare()
    if not LOCK.exists():return {"status":"FROZEN_READY"}
    lock=_read(LOCK)
    if lock["manifest_file_sha256"]!=_sha256_file(MANIFEST):
        raise G8Error("G8 lock frozen provenance changed")
    if not RESULT.exists():return {"status":"G8_LOCK_WITHOUT_RESULT_STOP"}
    result=_read(RESULT)
    if (result["lock_result_sha256"]!=lock["result_sha256"]
        or result["lock_file_sha256"]!=_sha256_file(LOCK)
        or result["manifest_file_sha256"]!=_sha256_file(MANIFEST)):
        raise G8Error("G8 result provenance changed")
    if not REPORT.is_file() or REPORT.read_text(encoding="utf-8")!=_overview(result):
        raise G8Error("G8 report not reproducible from frozen JSON")
    return {"status":"VERIFIED","science_status":result["status"],
            "result_sha256":result["result_sha256"]}


def main(argv:Sequence[str]|None=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="command",required=True)
    for name in ("preflight","prepare","run","check"):
        p=sub.add_parser(name)
        if name=="run":p.add_argument("--device",default="cuda")
    args=ap.parse_args(argv)
    obj=(preflight() if args.command=="preflight"
         else prepare() if args.command=="prepare"
         else run(args.device) if args.command=="run" else check())
    print(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
