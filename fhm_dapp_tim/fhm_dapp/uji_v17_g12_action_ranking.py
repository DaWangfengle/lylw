"""G12: disentangle V17 spatial ranking from Euclidean-MAE point action.

Zero new training / no model selection. G5 already tried medians on the
untempered Tampere posterior; G12 tests the *stronger frozen G8 anchor-T=.6*
law across TUJI/UTS/Tampere, while decomposing retrieval stages within the
same original E5 networks. Continuous geometric median is the existing
V9 Weiszfeld approximation, with action draws independent from risk and ES.

Historical CAL/CONFIRM outcomes are all exposed. These are retrospective
descriptive scientific mechanisms, NOT blind estimates or Q1 promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g8_hierarchical_attribution as g8
from fhm_dapp_tim.fhm_dapp import uji_v17_g5_bayes_action as g5
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness, sample_continuous_mixture,
    spatial_median_from_draws,
)
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _sha256_file, _write_json_new,
)

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"evidence/v17_g12_action_ranking"
DESIGN=EVIDENCE/"DESIGN_V17_G12.md"
MANIFEST=EVIDENCE/"manifest_v17_g12.json"
LOCK=EVIDENCE/"run_v17_g12.lock"
RESULT=EVIDENCE/"diagnosis_v17_g12.json"
REPORT=EVIDENCE/"RUN_REPORT_V17_G12.md"
PROTOCOL="v17_g12_frozen_anchor_action_and_rank_chain_20261010"
DOMAINS=("tuji","uts","tampere")
ROLES=g8.ROLES
E,L,K=5,192,192
BATCH=16
RADIUS=10.0
TEMPERATURE=0.6
ACTION_SEED=19801
RISK_SEED=19802
ACTION_DRAWS=128
RISK_DRAWS=128
MEDIAN_ITERATIONS=64
MEDIAN_TOL=1e-5
ACTIONS=("mean","component_median","continuous_median")
STAGES=("prototype_only","neural_only","fused_equal_members",
        "fused_radial_members","topL_radial_members","R3_consensus_K192")
DIAGNOSTICS=("site_near_mass","site_top1_in10m","site_best_near_rank")
PREDICTIVE_ES_SEED=g8.SEED


class G12Error(RuntimeError):
    pass


def _write(path:Path,object:dict,field:str="result_sha256")->dict:
    if path.exists():
        raise FileExistsError(f"G12 immutable file already exists: {path}")
    value=dict(object)
    value[field]=_canonical_json_sha256(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    _write_json_new(path,value)
    return value


def _read(path:Path,field:str="result_sha256")->dict:
    record=json.loads(path.read_text(encoding="utf-8"))
    if record.get(field)!=_canonical_json_sha256(record,omit=field):
        raise G12Error(f"G12 evidence hash changed: {path}")
    return record


def _source_hashes()->dict:
    files={
        "design":DESIGN,"g12":Path(__file__),
        "tests":ROOT.parent/"tests/test_uji_v17_g12_action_ranking.py",
        "g8":Path(g8.__file__),"g5":Path(g5.__file__),
        "r2b":Path(r2b.__file__),"r3":Path(r3.__file__),
        "r3_protocol":Path(r3p.__file__),
        "sampler":Path(sample_continuous_mixture.__code__.co_filename),
    }
    return {key:_sha256_file(path) for key,path in files.items()}


def preflight()->dict:
    if (g8.E!=E or g8.K!=K or g8.g6.L!=L or g8.g6.ALPHA!=1.0
        or g8.T!=TEMPERATURE or r3p.STUDENT_DF!=4.0
        or r3p.PREDICTIVE_DRAWS_PER_STREAM!=64):
        raise G12Error("frozen G8 E5/L192/K192/alpha1/T.6/Student-t changed")
    sources={domain:g8._pack(domain)["original"] for domain in DOMAINS}
    return {"status":"FROZEN_E5_RANK_AND_ACTION_READY",
            "protocol":PROTOCOL,
            "source_sha256":_source_hashes(),
            "model_and_role_provenance":sources,
            "no_training":True,"no_official_or_provided_test":True,
            "historical_label_exposure":True}


def prepare()->dict:
    checked=preflight()
    manifest={
        "protocol":PROTOCOL,"source_sha256":checked["source_sha256"],
        "model_and_role_provenance":checked["model_and_role_provenance"],
        "roles":{d:list(ROLES[d]) for d in DOMAINS},
        "actions":list(ACTIONS),"ranking_stages":list(STAGES),
        "batch":BATCH,"E":E,"L":L,"K":K,"alpha":1.0,
        "anchor_temperature":TEMPERATURE,"student_df":4.0,
        "ranking_hit_radius_m":RADIUS,
        "action_seed":ACTION_SEED,"risk_seed":RISK_SEED,
        "action_draws_per_stream":ACTION_DRAWS,
        "risk_draws_per_stream":RISK_DRAWS,
        "median_iterations":MEDIAN_ITERATIONS,
        "median_tolerance_m":MEDIAN_TOL,
        "energy_seed":PREDICTIVE_ES_SEED,
        "energy_draws_per_stream":r3p.PREDICTIVE_DRAWS_PER_STREAM,
        "shared_energy_for_all_actions":True,
        "diagnostics_posthoc_only":True,"no_truth_gating":True,
        "all_train_internal_confirmations_previously_seen":True,
        "no_new_model_fit_or_hyperparameter_selection":True,
    }
    manifest["manifest_sha256"]=_canonical_json_sha256(manifest)
    if MANIFEST.exists():
        if _read(MANIFEST,"manifest_sha256")!=manifest:
            raise G12Error("G12 frozen plan/source mismatch")
        return manifest
    _write_json_new(MANIFEST,manifest)
    return manifest


def _draws_for_batch(records:dict,start:int,end:int,device:torch.device)->dict:
    return {key:(value[start:end].to(device)
                 if isinstance(value,torch.Tensor) and value.ndim>0 else value)
            for key,value in records.items()}


@torch.no_grad()
def actions_from_frozen_law(
    joint:torch.Tensor,xy:torch.Tensor,sigma:torch.Tensor,
    action_draws:dict
)->dict[str,torch.Tensor]:
    """Same posterior, three point decisions. No label is an argument."""
    if (joint.ndim!=2 or joint.shape[1]!=E*K
        or xy.shape!=(*joint.shape,2) or sigma.shape!=xy.shape
        or not torch.isfinite(joint).all()
        or not torch.isfinite(xy).all() or not torch.isfinite(sigma).all()
        or (sigma<=0).any() or (joint<0).any()):
        raise G12Error("invalid frozen mixture law")
    if not torch.allclose(joint.sum(dim=1),torch.ones(len(joint),device=joint.device),
                          atol=4e-6,rtol=4e-6):
        raise G12Error("posterior not normalized")
    # Preserve G8 exact float32 summation of E*K mean locations.
    mean=(joint[:,:,None]*xy).sum(1)
    component=g5.weighted_component_geometric_median(
        joint,xy,iterations=MEDIAN_ITERATIONS,tolerance=MEDIAN_TOL)
    first=sample_continuous_mixture(
        joint,xy,sigma,action_draws["uniform_1"],action_draws["noise_1"])
    second=sample_continuous_mixture(
        joint,xy,sigma,action_draws["uniform_2"],action_draws["noise_2"])
    continuous=spatial_median_from_draws(
        first,second,maximum_iterations=MEDIAN_ITERATIONS,tolerance=MEDIAN_TOL)
    if (not torch.isfinite(component).all() or
        not torch.isfinite(continuous).all()):
        raise G12Error("nonfinite frozen-law point decisions")
    return {"mean":mean,"component_median":component,
            "continuous_median":continuous}


@torch.no_grad()
def predicted_point_risk(
    actions:dict[str,torch.Tensor],joint:torch.Tensor,
    xy:torch.Tensor,sigma:torch.Tensor,risk_draws:dict
)->dict[str,torch.Tensor]:
    """Independent MC estimate of expected Euclidean loss UNDER THE MODEL.

    NOT the true location risk. No query label. If median risk improves but
    measured MAE worsens, posterior mismatch is a plausible explanation.
    """
    f=sample_continuous_mixture(
        joint,xy,sigma,risk_draws["uniform_1"],risk_draws["noise_1"])
    s=sample_continuous_mixture(
        joint,xy,sigma,risk_draws["uniform_2"],risk_draws["noise_2"])
    return {name:.5*(torch.linalg.vector_norm(
                f.double()-a[:,None,:].double(),dim=-1).mean(1)+
                torch.linalg.vector_norm(
                s.double()-a[:,None,:].double(),dim=-1).mean(1))
            for name,a in actions.items()}


def ranking_probability_panels(
    distance:torch.Tensor,neural_logits:torch.Tensor,
    full_prior:torch.Tensor,top_prob:torch.Tensor,top_ids:torch.Tensor,
    radial:torch.Tensor,consensus_ids:torch.Tensor,
    consensus_prior:torch.Tensor
)->dict[str,torch.Tensor]:
    """Label-free aligned [B,A] stage maps; not modified V17 decisions.

    STAGES are mechanism counterfactual ranking maps:
    original prototype similarity; neural-only member mean;
    mixed fused posterior with equal and radial member weight;
    truncated Top-L radial-weight pool; actual R3 consensus Top-K.
    Neural-only vs prototype-only are ALTERNATIVES not temporal steps.
    """
    e,b,a=neural_logits.shape
    if (e!=E or distance.shape!=(b,a)
        or full_prior.shape!=(E,b,a)
        or top_prob.shape!=(E,b,L)
        or top_ids.shape!=(E,b,L)
        or radial.shape!=(E,b)
        or consensus_ids.shape!=(b,K)
        or consensus_prior.shape!=(b,K)):
        raise G12Error("mismatched full retrieval and R3 consensus dimensions")
    if ((top_ids<0).any() or (top_ids>=a).any()
        or (consensus_ids<0).any() or (consensus_ids>=a).any()):
        raise G12Error("invalid global FIT anchor IDs")
    wm=radial.square().clamp_min(1e-8).reciprocal()
    wm=wm/wm.sum(0,keepdim=True).clamp_min(1e-30)
    proto=torch.softmax(-r3p.BETA_PROTOTYPE*distance,dim=1)
    neural=torch.softmax(r3p.BETA_NEURAL*neural_logits,dim=2).mean(0)
    equal=full_prior.mean(0)
    radial_full=(wm[:,:,None]*full_prior).sum(0)
    sparse=torch.zeros_like(equal)
    sparse.scatter_add_(1,top_ids.permute(1,0,2).reshape(b,E*L),
                        (wm[:,:,None]*top_prob).permute(1,0,2).reshape(b,E*L))
    sparse=sparse/sparse.sum(1,keepdim=True).clamp_min(1e-30)
    consensus=torch.zeros_like(equal)
    consensus.scatter_(1,consensus_ids,consensus_prior)
    maps=dict(zip(STAGES,(proto,neural,equal,radial_full,sparse,consensus)))
    for name,prob in maps.items():
        if not torch.isfinite(prob).all() or (prob<0).any() or not torch.allclose(
            prob.sum(1),torch.ones(b,device=prob.device),atol=5e-6,rtol=5e-6):
            raise G12Error(f"invalid stage posterior {name}")
    return maps


def ranking_label_diagnostics(
    panels:dict[str,torch.Tensor],
    anchor_xy:torch.Tensor,anchor_build:torch.Tensor,
    anchor_floor:torch.Tensor,true_xy:torch.Tensor,
    true_build:torch.Tensor,true_floor:torch.Tensor
)->dict[str,torch.Tensor]:
    """Retrospective labels only; NEVER passed into any inference path."""
    b,a=true_xy.shape[0],anchor_xy.shape[0]
    near=(torch.cdist(true_xy,anchor_xy)<=RADIUS+1e-5)
    near &= (anchor_build[None,:]==true_build[:,None])
    near &= (anchor_floor[None,:]==true_floor[:,None])
    out={"near_reference_exists":near.any(1).float()}
    for name,prob in panels.items():
        if prob.shape!=(b,a):
            raise G12Error(f"wrong stage shape: {name}")
        order=torch.argsort(prob,dim=1,stable=True,descending=True)
        top=order[:,0]
        near_sorted=near.gather(1,order)
        first=torch.where(near_sorted,torch.arange(
            1,a+1,device=near.device)[None,:].expand(b,-1),
            torch.full_like(order,a+1)).min(1).values
        out[name+"_top1_in10m"]=near.gather(1,top[:,None]).squeeze(1).float()
        out[name+"_near10_mass"]=(prob*near).sum(1)
        out[name+"_best_near_rank"]=torch.where(
            first<=a,first.double(),
            torch.full((b,),float("nan"),device=near.device,dtype=torch.float64))
    return out


def _site_vector(row:torch.Tensor,gid:torch.Tensor,sites:int)->list:
    x=row.detach().cpu().double()
    g=gid.detach().cpu().long()
    mask=torch.isfinite(x)
    sum_=torch.zeros(sites,dtype=torch.float64)
    count=torch.zeros(sites,dtype=torch.float64)
    sum_.scatter_add_(0,g,torch.where(mask,x,torch.zeros_like(x)))
    count.scatter_add_(0,g,mask.double())
    return [float(a/b) if b else None for a,b in
            zip(sum_.tolist(),count.tolist())]


def _summary(vectors:dict[str,list])->dict:
    out={}
    for name,values in vectors.items():
        x=np.asarray([v for v in values if v is not None],dtype=np.float64)
        out[name]={"site_mean":float(x.mean()) if len(x) else None,
                   "n_finite_sites":len(x),"n_sites":len(values)}
        if name.endswith("_mae_m") and len(x):
            out[name].update({
                "p90_m":float(np.quantile(x,.9)),
                "p95_m":float(np.quantile(x,.95)),
                "max_m":float(np.max(x))})
    return out


@torch.no_grad()
def score_role(models:list,data:dict,rows:np.ndarray,origin:np.ndarray)->dict:
    ref=models[0]
    device=next(ref.parameters()).device
    n=len(rows)
    gid,ordered=r2b._coordinate_groups([data["keys"][int(i)] for i in rows])
    target=torch.as_tensor(data["raw_xy"][rows]-origin[None,:],
                           dtype=torch.float64,device=device)
    truth_b=torch.as_tensor(data["building"][rows],dtype=torch.long,device=device)
    truth_f=torch.as_tensor(data["floor"][rows],dtype=torch.long,device=device)
    a=int(ref.n_anchors)
    if a<K:raise G12Error("FIT anchors fewer than Top-K192")
    anchor_xy=ref.anchor_coords.to(device).float()
    anchor_build=ref.anchor_buildings.to(device).long()
    anchor_floor=ref.anchor_floors.to(device).long()
    action_rng=common_student_t_randomness(n,draws=ACTION_DRAWS,
                      degrees_freedom=r3p.STUDENT_DF,seed=ACTION_SEED)
    risk_rng=common_student_t_randomness(n,draws=RISK_DRAWS,
                      degrees_freedom=r3p.STUDENT_DF,seed=RISK_SEED)
    es_rng=common_student_t_randomness(n,
                      draws=r3p.PREDICTIVE_DRAWS_PER_STREAM,
                      degrees_freedom=r3p.STUDENT_DF,seed=PREDICTIVE_ES_SEED)
    output:dict[str,torch.Tensor]={}
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0,n,BATCH):
            end=min(start+BATCH,n)
            b=end-start
            rss=data["rss"][rows[start:end]].float().to(device)
            proto=r2b._prototype_distance(ref,rss)
            all_ids=torch.arange(a,device=device)[None,:].expand(b,-1)
            logits=[];loc=[];scales=[]
            for model in models:
                enc=model.encoder(rss)
                logits.append(model.retrieval_logits_from_embedding(enc))
                x,s=model._components(enc,all_ids)
                loc.append(x);scales.append(s)
            logit=torch.stack(logits)
            full_prior=torch.softmax(
                r3p.BETA_NEURAL*logit-r3p.BETA_PROTOTYPE*proto[None,:,:],
                dim=2)
            top,ix,rad,_=r3._member_top_l(
                full_prior,torch.stack(loc),torch.stack(scales),
                L,student_df=r3p.STUDENT_DF)
            joint,xy,sigma=r3._dense_joint_law(
                top,ix,rad,torch.stack(loc),torch.stack(scales),
                tuple(range(E)),top_k=K,alpha=g8.g6.ALPHA)
            w=rad.square().clamp_min(1e-8).reciprocal()
            w=w/w.sum(0,keepdim=True).clamp_min(1e-30)
            ids,c=r3.dense_alpha_consensus_prior(
                top,ix,w,a,K,epsilon=r3.CONSENSUS_EPSILON,
                alpha=g8.g6.ALPHA)
            if not torch.allclose(c,joint.reshape(b,E,K).sum(1),
                                  atol=3e-6,rtol=3e-6):
                raise G12Error("R3 exact K192 anchor marginal changed")
            q=g8.hierarchical_laws(joint,E,K)["anchor"]
            dec=actions_from_frozen_law(q,xy,sigma,
                _draws_for_batch(action_rng,start,end,device))
            risk=predicted_point_risk(dec,q,xy,sigma,
                _draws_for_batch(risk_rng,start,end,device))
            # ES and posterior are IDENTICAL for all decisions.
            es,old_mean=r3._score_joint_law(
                q,xy,sigma,target[start:end],
                _draws_for_batch(es_rng,start,end,device))
            if not torch.allclose(
                old_mean,torch.linalg.vector_norm(
                    dec["mean"].double()-target[start:end],dim=1).cpu(),
                    atol=1e-6,rtol=1e-6):
                raise G12Error("G8 original mean/MAE reconstruction changed")
            measures={"shared_student_t_es_m":es.to(device),
                      "action_shift_component_m":torch.linalg.vector_norm(
                          dec["component_median"]-dec["mean"],dim=1),
                      "action_shift_continuous_m":torch.linalg.vector_norm(
                          dec["continuous_median"]-dec["mean"],dim=1)}
            for name,xy_dec in dec.items():
                measures[name+"_mae_m"]=torch.linalg.vector_norm(
                    xy_dec.double()-target[start:end],dim=1)
                measures[name+"_model_risk_m"]=risk[name]
            panels=ranking_probability_panels(
                proto,logit,full_prior,top,ix,rad,ids,c)
            # The sharpened anchor marginal is rank preserving.
            original_top=panels["R3_consensus_K192"].argmax(1)
            sharpened=c.pow(1/TEMPERATURE).argmax(1)
            if not torch.equal(original_top,ids.gather(1,sharpened[:,None]).squeeze(1)):
                raise G12Error("Anchor T0.6 changed anchor Top-1 identity")
            diagnostics=ranking_label_diagnostics(
                panels,anchor_xy,anchor_build,anchor_floor,
                target[start:end].float(),truth_b[start:end],truth_f[start:end])
            measures.update(diagnostics)
            for name,value in measures.items():
                if name not in output:
                    output[name]=torch.empty(n,dtype=torch.float64)
                output[name][start:end]=value.detach().cpu().double()
    site={name:_site_vector(values,gid,len(ordered)) for name,values in output.items()}
    stats=_summary(site)
    delta={k:[a-b for a,b in zip(site[k+"_mae_m"],site["mean_mae_m"])]
           for k in ("component_median","continuous_median")}
    pairs={}
    for name,ds in delta.items():
        pairs[name+"_minus_mean"]={
            "mean_delta_mae_m":float(np.mean(ds)),
            "improved_sites":sum(x<0 for x in ds),
            "worsened_sites":sum(x>0 for x in ds),
            "worst_harm_m":max(ds),
            "best_gain_m":-min(ds),
            "coordinate_vector_delta_mae_m":ds,
        }
    # Descriptive stage rescue/loss uses true labels; no runtime gate.
    movements={}
    for before,after in zip(STAGES[:-1],STAGES[1:]):
        old=site[before+"_top1_in10m"]
        new=site[after+"_top1_in10m"]
        movements[before+"__to__"+after]={
            "both_correct":sum(a>=.5 and b>=.5 for a,b in zip(old,new)),
            "lost_correct":sum(a>=.5 and b<.5 for a,b in zip(old,new)),
            "rescued_correct":sum(a<.5 and b>=.5 for a,b in zip(old,new)),
            "both_wrong":sum(a<.5 and b<.5 for a,b in zip(old,new)),
        }
    return {"rows":n,"physical_sites":len(ordered),
            "physical_coordinate_keys":[list(x) for x in ordered],
            "summary":stats,"site_vectors":site,
            "decision_effects":pairs,
            "ranking_stage_transitions_posthoc":movements,
            "oracle_labels_only_for_scoring_diagnostics":True,
            "random_streams_independent":True}


def _report(record:dict)->str:
    lines=["# G12 | Frozen posterior: ranking vs MAE point decision","",
           "**Retrospective G8 anchor-only T=.6 across original five TRAIN-internal roles.**",
           "G5 already used geometric medians on untempered Tampere;",
           "G12 does not claim a novel median algorithm.","",
           "| Domain / role | Mean MAE m | Discrete centre median m | "
           "Continuous t-median m | ES (all decisions) m |",
           "|---|---:|---:|---:|---:|"]
    for domain in DOMAINS:
        for role,x in record["by_domain"][domain].items():
            s=x["summary"]
            lines.append(f"| {domain}/{role} | {s['mean_mae_m']['site_mean']:.6f} | "
                f"{s['component_median_mae_m']['site_mean']:.6f} | "
                f"{s['continuous_median_mae_m']['site_mean']:.6f} | "
                f"{s['shared_student_t_es_m']['site_mean']:.6f} |")
    lines.extend(["","## Scientific interpretation boundaries","",
        "- Point actions share EXACTLY one frozen 960-component posterior.",
        "- The training-free continuous median is approximate: 2x128 fixed",
        "  Student-t draws, Weiszfeld 64 iterations, independent MC risk",
        "  sample and the original common-random-numbers ES stream.",
        "- Predicted model risk is NOT the realized Euclidean position MAE.",
        "- Median may lower its own posterior expected risk while worsening",
        "  actual positional error if the posterior is misspecified.",
        "- Prototype/neural-only stage probabilities are analytical",
        "  counterfactuals, not stages of a deployed cascaded predictor.",
        "- Near10m/true-floor labels are only posthoc scientific diagnosis.",
        "- Historical CAL/CONFIRM outcomes have ALREADY been observed;",
        "  no new blind testing, significance-based promotion or Q1 claim.","",
        f"Self hash: {record['result_sha256']}",""])
    return "\n".join(lines)


def run(device_name:str="cuda")->dict:
    if LOCK.exists() or RESULT.exists():
        raise G12Error("formal experiment already attempted: no rerun")
    manifest=prepare()
    locked=_write(LOCK,{
        "protocol":PROTOCOL,
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "no_parameter_training":True,
        "historical_outcome_labels_exposed":True})
    by_domain={}
    for domain in DOMAINS:
        source=g8._pack(domain)
        models,origin=g8._load_models(
            domain,{"model_and_role_provenance":manifest["model_and_role_provenance"]},
            device_name)
        by_domain[domain]={}
        for role in ROLES[domain]:
            by_domain[domain][role]=score_role(
                models,source["data"],source["rows"][role],origin)
        del models
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result=_write(RESULT,{
        "protocol":PROTOCOL,"status":"FROZEN_T06_ACTION_AND_RANK_DIAGNOSIS",
        "by_domain":by_domain,
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "lock_file_sha256":_sha256_file(LOCK),
        "lock_self_sha256":locked["result_sha256"],
        "no_original_network_updates":True,
        "no_train_calibration_or_confirm_fit":True,
        "official_test_opened":False,
        "historical_confirmation_labels_already_seen":True,
    })
    REPORT.write_text(_report(result),encoding="utf-8",newline="\n")
    return {"status":result["status"],"result_sha256":result["result_sha256"],
            "mean_minus_median_by_domain":{
                domain:{role:{
                    key:value["mean_delta_mae_m"]
                    for key,value in by_domain[domain][role]["decision_effects"].items()}
                    for role in by_domain[domain]} for domain in DOMAINS}}


def check()->dict:
    if not MANIFEST.exists():return {"status":"UNPREPARED"}
    prepare()
    if not LOCK.exists():return {"status":"PLAN_FROZEN"}
    lock=_read(LOCK)
    if lock["manifest_file_sha256"]!=_sha256_file(MANIFEST):
        raise G12Error("lock manifest mismatch")
    if not RESULT.exists():return {"status":"LOCK_WITHOUT_RESULT_STOP"}
    result=_read(RESULT)
    if (result["lock_file_sha256"]!=_sha256_file(LOCK)
        or result["lock_self_sha256"]!=lock["result_sha256"]
        or result["manifest_file_sha256"]!=_sha256_file(MANIFEST)):
        raise G12Error("result file provenance mismatch")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8")!=_report(result):
        raise G12Error("G12 report mismatch")
    return {"status":"VERIFIED","result_sha256":result["result_sha256"]}


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for command in ("preflight","prepare","run","check"):
        cmd=sub.add_parser(command)
        if command=="run":cmd.add_argument("--device",default="cuda")
    cmd=p.parse_args(argv)
    v=preflight() if cmd.command=="preflight" else (
        prepare() if cmd.command=="prepare" else
        run(cmd.device) if cmd.command=="run" else check())
    print(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
