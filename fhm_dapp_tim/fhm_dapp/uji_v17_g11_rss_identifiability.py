"""G11: FIT-only RSS distinguishability vs frozen V17 spatial ranking.

One retrospective, no-training experiment on original TRAIN-internal roles.
No official/provided test data, no new tuning on exposed CAL or CONFIRM.
Raw RSS L2 (the original R3 FIT prototype distance), a fixed 5NN rule,
and independently constructed AP-visibility Brier distance are compared
with unmodified E5 Anchor-only T=.6 using a common site-macro protocol.

The label-facing hard-negative margin is DIAGNOSTIC ONLY and cannot be
used in the prediction rules. A missed raw prototype is not proof that
RSS is inherently uninformative; a correct raw prototype is not proof
that a future learned method can achieve oracle error on new locations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g8_hierarchical_attribution as g8
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _sha256_file, _write_json_new,
)

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"evidence/v17_g11_rss_identifiability"
DESIGN=EVIDENCE/"DESIGN_V17_G11.md"
MANIFEST=EVIDENCE/"manifest_v17_g11.json"
LOCK=EVIDENCE/"diagnostic_v17_g11.lock"
RESULT=EVIDENCE/"rss_identifiability_v17_g11.json"
REPORT=EVIDENCE/"RUN_REPORT_V17_G11.md"
PROTOCOL="v17_g11_raw_rss_vs_learned_ranking_train_internal_20261010"
DOMAINS=("tuji","uts","tampere")
ROLES=g8.ROLES
BATCH=16
E,L,K=5,192,192
NEAR=10.0
FAR=20.0
AP_MISSING=-1.0
AP_EPS=1e-3
KNN=5
KNN_EPS=1e-3
BETA_PRIOR=.5
METRICS=("raw_l2_1nn","raw_l2_5nn","ap_visibility_1nn","v17_anchor_t06")
ANALYSES=("raw_l2_top1_in_10m","raw_l2_5nn_in_10m",
          "ap_visibility_top1_in_10m","v17_top1_in_10m",
          "raw_near_far_margin","ap_near_far_margin",
          "raw_near_far_positive","ap_near_far_positive",
          "near_fit_anchor_exists","v17_near_10m_mass",
          "v17_best_anchor_rank","ap_visible_fraction")


class G11Error(RuntimeError):
    pass


def _write(path:Path,value:dict,field:str="result_sha256")->dict:
    if path.exists():
        raise FileExistsError(f"G11 artifact already exists: {path}")
    record=dict(value)
    record[field]=_canonical_json_sha256(record)
    path.parent.mkdir(parents=True,exist_ok=True)
    _write_json_new(path,record)
    return record


def _read(path:Path,field:str="result_sha256")->dict:
    record=json.loads(path.read_text(encoding="utf-8"))
    if record.get(field)!=_canonical_json_sha256(record,omit=field):
        raise G11Error(f"G11 immutable evidence mismatch: {path}")
    return record


def _source_hashes()->dict:
    names={"design":DESIGN,"g11":Path(__file__),
           "g11_tests":ROOT.parent/"tests/test_uji_v17_g11_rss_identifiability.py",
           "g8":Path(g8.__file__),"r2b":Path(r2b.__file__),
           "r3":Path(r3.__file__),"r3_protocol":Path(r3p.__file__)}
    return {name:_sha256_file(path) for name,path in names.items()}


def preflight()->dict:
    source={d:g8._pack(d)["original"] for d in DOMAINS}
    if (g8.T!=.6 or g8.E!=E or g8.K!=K or g8.g6.L!=L
        or g8.g6.ALPHA!=1.0 or r3p.STUDENT_DF!=4.0):
        raise G11Error("E5/L192/K192/alpha1/dF4 V17 source changed")
    return {
        "protocol":PROTOCOL,"status":"FIT_ONLY_RSS_COMPARATORS_READY",
        "source_sha256":_source_hashes(),
        "historical_e5":source,
        "roles":{d:list(ROLES[d]) for d in DOMAINS},
        "official_test_opened":False,
        "historical_evaluation_labels_already_seen":True,
        "original_checkpoints_retrained":False,
    }


def prepare()->dict:
    checked=preflight()
    value={
        "protocol":PROTOCOL,"source_sha256":checked["source_sha256"],
        "model_and_role_provenance":checked["historical_e5"],
        "roles":checked["roles"],"domains":list(DOMAINS),
        "batch":BATCH,"ensemble":E,"top_l":L,"top_k":K,
        "temperature":g8.T,"near_radius_m":NEAR,"far_radius_m":FAR,
        "ap_present_when":"model RSS > -0.999 (original FIT rule); clipped -110 dBm indistinguishable from missing",
        "missing_rss_value":AP_MISSING,"ap_missing_epsilon":AP_EPS,
        "visibility_beta_prior_each_side":BETA_PRIOR,
        "baseline_5nn_count":KNN,"baseline_5nn_eps":KNN_EPS,
        "scores":list(METRICS),"descriptive_measures":list(ANALYSES),
        "macro":"average within each physical coordinate then average coordinates",
        "ranking":"all FIT anchors, no true-building/floor filtering",
        "margin":"best clearly-wrong (different floor/building or XY>=20m)"
                 " minus best correct (same floor/building XY<=10m)",
        "uncertainty":"no ES for raw deterministic point predictors; no probability model fit",
        "no_new_training_no_model_selection":True,
        "old_confirmation_labels_already_viewed":True,
        "no_official_test":True,
    }
    value["manifest_sha256"]=_canonical_json_sha256(value)
    if MANIFEST.exists():
        if _read(MANIFEST,"manifest_sha256")!=value:
            raise G11Error("G11 source or preregistered method changed")
        return value
    _write_json_new(MANIFEST,value)
    return value


def fit_site_ap_frequency(
    fit_rss:torch.Tensor, fit_group:torch.Tensor, anchors:int,
    seen_ap_mask:torch.Tensor,
)->torch.Tensor:
    """Reproduce original FIT per-anchor AP detection probabilities exactly.

    Uses the original FIT rule RSS > -0.999 and Beta(.5,.5) smoothing.
    The channel is already in the frozen anchor checkpoint, not new sensing.
    """
    if (fit_rss.ndim!=2 or fit_group.shape!=(len(fit_rss),)
        or seen_ap_mask.shape!=(fit_rss.shape[1],)
        or anchors<=0 or len(fit_rss)<=0):
        raise G11Error("invalid FIT fingerprint / AP visibility dimensions")
    if (fit_group.min()<0 or fit_group.max()>=anchors
        or seen_ap_mask.sum()<1 or not torch.isfinite(fit_rss).all()):
        raise G11Error("invalid FIT anchor/AP membership")
    obs=(fit_rss[:,seen_ap_mask]>AP_MISSING+AP_EPS).float()
    counts=torch.bincount(fit_group,minlength=anchors).float()
    if (counts<=0).any():
        raise G11Error("FIT has reference anchor without scans")
    accumulation=torch.zeros((anchors,obs.shape[1]),dtype=torch.float32)
    accumulation.index_add_(0,fit_group,obs)
    out=(accumulation+BETA_PRIOR)/(counts[:,None]+2*BETA_PRIOR)
    if (out<=0).any() or (out>=1).any():
        raise G11Error("Beta-smoothed presence must lie strictly within (0,1)")
    return out


def map_fit_rows_to_model_anchors(
    fit_rows:np.ndarray, data:dict, ref, origin:np.ndarray
)->tuple[torch.Tensor,tuple[tuple,...]]:
    """Prove a one-to-one key -> checkpoint anchor matching, never assume order.

    Location labels are used ONLY to construct the original FIT memory,
    never to filter prediction anchors using the unknown query label.
    """
    fit_keys=[data["keys"][int(i)] for i in fit_rows]
    gid,sorted_keys=r2b._coordinate_groups(fit_keys)
    a=int(ref.n_anchors)
    if len(sorted_keys)!=a:
        raise G11Error("FIT unique physical keys != checkpoint anchors")
    expected_xy=torch.tensor([[float(k[2]),float(k[3])] for k in sorted_keys],
                             dtype=torch.float32)
    expected_xy-=torch.as_tensor(origin,dtype=torch.float32)[None,:]
    reference_xy=ref.anchor_coords.detach().cpu().float()
    expected_build=torch.tensor([int(k[0]) for k in sorted_keys])
    expected_floor=torch.tensor([int(k[1]) for k in sorted_keys])
    ab=ref.anchor_buildings.detach().cpu().long()
    af=ref.anchor_floors.detach().cpu().long()
    if (reference_xy.shape!=(a,2) or ab.shape!=(a,) or af.shape!=(a,)):
        raise G11Error("FIT anchor geometry changed")
    matched=(expected_build[:,None]==ab[None,:])&(
        expected_floor[:,None]==af[None,:])
    dist=torch.cdist(expected_xy,reference_xy)
    dist.masked_fill_(~matched,float("inf"))
    indices=dist.argmin(dim=1)
    errors=dist.gather(1,indices[:,None]).squeeze(1)
    if (len(torch.unique(indices))!=a or
        not torch.isfinite(errors).all() or errors.max()>0.03):
        raise G11Error("FIT key-to-anchor bijection/coordinates not verified")
    return indices[gid],sorted_keys


def visibility_brier_distance(
    query:torch.Tensor, ref_frequency:torch.Tensor
)->torch.Tensor:
    """Exact per-AP mean squared Bernoulli forecast, efficient GEMM.

    Assumes query already contains only the model-seen AP columns and
    in [-1,1]. Query and FIT frequency never receive true location labels.
    """
    if (query.ndim!=2 or ref_frequency.ndim!=2
        or query.shape[1]!=ref_frequency.shape[1]):
        raise G11Error("AP visibility distance dimension mismatch")
    z=(query>AP_MISSING+AP_EPS).float()
    p=ref_frequency.float()
    n=z.shape[1]
    return ((z.square().sum(1,keepdim=True)
            -2*z@p.T+p.square().sum(1)[None,:])/n).clamp_min(0.)


def rss_knn_predictions(distance:torch.Tensor,
                        anchor_xy:torch.Tensor)->dict[str,torch.Tensor]:
    """Unsupervised non-oracle global-anchor 1NN and inverse-square 5NN."""
    if (distance.ndim!=2 or distance.shape[1]!=len(anchor_xy)
        or len(anchor_xy)<KNN or not torch.isfinite(distance).all()):
        raise G11Error("raw RSS reference scores invalid")
    ids=torch.argsort(distance,dim=1,stable=True)[:,:KNN]
    dist=distance.gather(1,ids).clamp_min(0)
    weights=(dist+KNN_EPS).pow(-2)
    weights=weights/weights.sum(1,keepdim=True)
    coords=anchor_xy[ids]
    return {
        "raw_l2_1nn":coords[:,0],
        "raw_l2_5nn":(weights[:,:,None]*coords).sum(dim=1),
        "raw_l2_top1_id":ids[:,0],
        "raw_l2_5nn_ids":ids,
    }


def distinguishability(
    true_xy:torch.Tensor,true_build:torch.Tensor,true_floor:torch.Tensor,
    anchor_xy:torch.Tensor,anchor_build:torch.Tensor,anchor_floor:torch.Tensor,
    raw:torch.Tensor,ap_distance:torch.Tensor,
    raw_top:torch.Tensor,ap_top:torch.Tensor,r3_top:torch.Tensor,
    consensus_ids:torch.Tensor,consensus_mass:torch.Tensor
)->dict[str,torch.Tensor]:
    """Label-facing metrics only; no value returns to prediction functions."""
    n,a=raw.shape
    if (ap_distance.shape!=(n,a) or consensus_ids.shape!=(n,K)
        or consensus_mass.shape!=(n,K)):
        raise G11Error("diagnostic geometry mismatch")
    same=(true_build[:,None]==anchor_build[None,:])&(
        true_floor[:,None]==anchor_floor[None,:])
    geometric=torch.cdist(true_xy,anchor_xy)
    near=same&(geometric<=NEAR+1e-5)
    far=(~same)|(geometric>=FAR-1e-5)
    def near_indicator(top):
        return near.gather(1,top[:,None]).squeeze(1).float()
    output={
        "near_fit_anchor_exists":near.any(1).float(),
        "raw_l2_top1_in_10m":near_indicator(raw_top),
        "ap_visibility_top1_in_10m":near_indicator(ap_top),
        "v17_top1_in_10m":near_indicator(r3_top),
        "raw_l2_5nn_in_10m":torch.zeros(n,device=raw.device),
        "v17_near_10m_mass":(
            consensus_mass*near.gather(1,consensus_ids)).sum(1),
        "ap_visible_fraction":torch.full((n,),float("nan"),device=raw.device),
    }
    def margin(dist):
        near_min=dist.masked_fill(~near,float("inf")).min(1).values
        far_min=dist.masked_fill(~far,float("inf")).min(1).values
        gap=far_min-near_min
        return torch.where(torch.isfinite(gap),gap,
                           torch.full_like(gap,float("nan")))
    for prefix,dist in (("raw",raw),("ap",ap_distance)):
        gap=margin(dist)
        output[prefix+"_near_far_margin"]=gap
        output[prefix+"_near_far_positive"]=torch.where(
            torch.isfinite(gap),(gap>0).float(),
            torch.full_like(gap,float("nan")))
    # Probability ranking of best geometrically correct anchor in the
    # original final consensus; null if no such anchor in Top-K.
    match=near.gather(1,consensus_ids)
    order=torch.arange(K,device=raw.device)[None,:].expand(n,-1)
    rank=torch.where(match,order+1,torch.full_like(order,K+1)).min(1).values
    output["v17_best_anchor_rank"]=torch.where(
        rank<=K,rank.float(),torch.full_like(rank.float(),float("nan")))
    return output


def _site_vector(row:torch.Tensor,gid:torch.Tensor,count:int)->list:
    x=row.detach().cpu().double()
    group=gid.detach().cpu().long()
    finite=torch.isfinite(x)
    totals=torch.zeros(count,dtype=torch.float64)
    counts=torch.zeros(count,dtype=torch.float64)
    totals.scatter_add_(0,group,torch.where(finite,x,0.))
    counts.scatter_add_(0,group,finite.double())
    return [float(s/n) if n else None for s,n in
            zip(totals.tolist(),counts.tolist())]


def _metric_summaries(vectors:dict[str,list])->dict:
    result={}
    for name,vec in vectors.items():
        valid=np.asarray([x for x in vec if x is not None],dtype=np.float64)
        result[name]={
            "macro_mean":float(valid.mean()) if valid.size else None,
            "eligible_sites":int(valid.size),"total_sites":len(vec),
        }
        if name.endswith("_point_mae_m") and valid.size:
            result[name].update({
                "p90_mae_m":float(np.quantile(valid,.9)),
                "p95_mae_m":float(np.quantile(valid,.95)),
                "max_mae_m":float(valid.max())})
    return result


@torch.no_grad()
def score_role(models:list,data:dict,rows:np.ndarray,origin:np.ndarray,
               fit_freq:torch.Tensor)->dict:
    """No training, no confirmation-conditioned choice, no oracle floor."""
    ref=models[0];device=next(ref.parameters()).device
    target=torch.as_tensor(data["raw_xy"][rows]-origin[None,:],
                           device=device,dtype=torch.float64)
    true_b=torch.as_tensor(data["building"][rows],dtype=torch.long,device=device)
    true_f=torch.as_tensor(data["floor"][rows],dtype=torch.long,device=device)
    row_keys=[data["keys"][int(i)] for i in rows]
    gid,ordered=r2b._coordinate_groups(row_keys)
    vectors:dict[str,torch.Tensor]={}
    n=len(rows);a=ref.n_anchors
    anchor_xy=ref.anchor_coords.to(device).float()
    ab=ref.anchor_buildings.to(device).long()
    af=ref.anchor_floors.to(device).long()
    seen=ref.encoder.seen_ap_mask.to(device).bool()
    freq=fit_freq.to(device)
    if (freq.shape!=(a,int(seen.sum())) or not torch.isfinite(target).all()):
        raise G11Error("FIT fingerprint presence geometry invalid")
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0,n,BATCH):
            end=min(start+BATCH,n);b=end-start
            rss=data["rss"][rows[start:end]].float().to(device)
            raw=r2b._prototype_distance(ref,rss)
            ap=visibility_brier_distance(rss[:,seen],freq)
            knn=rss_knn_predictions(raw,anchor_xy)
            ap_top=torch.argsort(ap,dim=1,stable=True)[:,0]
            ids=torch.arange(a,device=device)[None,:].expand(b,-1)
            logit=[];centres=[];sigma=[]
            for model in models:
                embedding=model.encoder(rss)
                logit.append(model.retrieval_logits_from_embedding(embedding))
                xy,sc=model._components(embedding,ids)
                centres.append(xy);sigma.append(sc)
            prior=torch.softmax(
                r3p.BETA_NEURAL*torch.stack(logit)
                -r3p.BETA_PROTOTYPE*raw[None,:,:],dim=-1)
            top,idx,rad,_=r3._member_top_l(
                prior,torch.stack(centres),torch.stack(sigma),
                L,student_df=r3p.STUDENT_DF)
            joint,loc,_=r3._dense_joint_law(
                top,idx,rad,torch.stack(centres),torch.stack(sigma),
                tuple(range(E)),top_k=K,alpha=g8.g6.ALPHA)
            member_w=rad.square().clamp_min(1e-8).reciprocal()
            member_w=member_w/member_w.sum(0,keepdim=True).clamp_min(1e-30)
            aid,c=r3.dense_alpha_consensus_prior(
                top,idx,member_w,int(a),K,
                epsilon=r3.CONSENSUS_EPSILON,alpha=g8.g6.ALPHA)
            if not torch.allclose(
                c,joint.reshape(b,E,K).sum(1),rtol=3e-6,atol=3e-6):
                raise G11Error("original V17 R3 anchor law not reproduced")
            q=g8.hierarchical_laws(joint,E,K)["anchor"]
            point=(q[:,:,None]*loc).sum(1)
            decisions={
                "raw_l2_1nn":knn["raw_l2_1nn"],
                "raw_l2_5nn":knn["raw_l2_5nn"],
                "ap_visibility_1nn":anchor_xy[ap_top],
                "v17_anchor_t06":point,
            }
            diag=distinguishability(
                target[start:end].float(),true_b[start:end],true_f[start:end],
                anchor_xy,ab,af,raw,ap,knn["raw_l2_top1_id"],
                ap_top,aid[:,0],aid,c)
            match=(true_b[start:end,None]==ab[None,:])&(
                true_f[start:end,None]==af[None,:])
            distance=torch.cdist(target[start:end].float(),anchor_xy)
            diag["raw_l2_5nn_in_10m"]=(
                match.gather(1,knn["raw_l2_5nn_ids"])&
                (distance.gather(1,knn["raw_l2_5nn_ids"])<=NEAR+1e-5)
            ).any(1).float()
            diag["ap_visible_fraction"]=(
                (rss[:,seen]>AP_MISSING+AP_EPS).float().mean(1))
            for name,xy in decisions.items():
                diag[name+"_point_mae_m"]=torch.linalg.vector_norm(
                    xy.double()-target[start:end],dim=1)
            for name,metric in diag.items():
                if name not in vectors:
                    vectors[name]=torch.empty(n,dtype=torch.float64)
                vectors[name][start:end]=metric.detach().cpu().double()
    sites={name:_site_vector(vec,gid,len(ordered))
           for name,vec in vectors.items()}
    summaries=_metric_summaries(sites)
    # Label-facing conditional groups are posthoc. A G9-style true
    # location risk condition is NEVER introduced into the predictors.
    conditional={}
    for key in ("raw_l2_top1_in_10m","ap_visibility_top1_in_10m",
                "v17_top1_in_10m"):
        indicator=sites[key]
        hit=[i for i,x in enumerate(indicator) if x is not None and x>=.5]
        miss=[i for i,x in enumerate(indicator) if x is not None and x<.5]
        name="v17_anchor_t06_point_mae_m"
        conditional[key]={
            "majority_hit_sites":len(hit),"majority_miss_sites":len(miss),
            "v17_mae_hit_sites_m":float(np.mean([sites[name][i] for i in hit]))
                                   if hit else None,
            "v17_mae_miss_sites_m":float(np.mean([sites[name][i] for i in miss]))
                                   if miss else None,
        }
    return {
        "rows":n,"physical_sites":len(ordered),
        "physical_coordinate_keys":[list(v) for v in ordered],
        "summary":summaries,"site_vectors":sites,
        "conditional_descriptive_only":conditional,
        "label_use":"posthoc ranking diagnostics and error metrics ONLY",
    }


def _report(result:dict)->str:
    lines=[
        "# G11 | Can observed RSS distinguish the V17 wrong spatial modes?",
        "",
        "FIT-only reference signals, fixed original five-member checkpoints.",
        "Historical CAL/CONFIRM only. **Not prospective/independent test.**",
        "",
        "| Domain/role | V17 Anchor MAE m | Raw RSS 1NN MAE m | "
        "Raw RSS 5NN MAE m | AP visibility 1NN MAE m | "
        "Raw 1NN@10m | V17 Top1@10m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for d in DOMAINS:
        for role,record in result["by_domain"][d].items():
            s=record["summary"]
            def v(name):
                n=s[name]["macro_mean"]
                return "n/a" if n is None else f"{n:.4f}"
            lines.append(
                f"| {d}/{role} | {v('v17_anchor_t06_point_mae_m')} | "
                f"{v('raw_l2_1nn_point_mae_m')} | "
                f"{v('raw_l2_5nn_point_mae_m')} | "
                f"{v('ap_visibility_1nn_point_mae_m')} | "
                f"{v('raw_l2_top1_in_10m')} | "
                f"{v('v17_top1_in_10m')} |")
    lines.extend(["",
        "## Scientific boundary","",
        "1. All three point baselines rank ALL original FIT anchors; no true",
        "   floor/building filtering or CAL/CONFIRM parameter fitting.",
        "2. Raw RSS L2 uses the EXACT same model-seen AP L2 prototype law",
        "   already used upstream in R3: it is an information benchmark,",
        "   not a novel independent sensor observation.",
        "3. AP-visibility Brier uses a new FIT-only reference frequency",
        "   channel, but it comes from the SAME RSS scans; observed values",
        "   clipped to -1 are indistinguishable from missing AP.",
        "4. Margin uses TRUE location, with positives <=10m same floor",
        "   and clearly wrong >=20m or different building/floor.",
        "   It is for hindsight diagnosis only, never a deployed gate.",
        "5. A wrong 1NN does not establish Bayes irreducibility; a correct",
        "   1NN does not prove V17 learned representation discards all",
        "   useful RSS information. Neither is an oracle performance bound.",
        "6. Deterministic RSS baselines are point estimates: continuous",
        "   Student-t Energy Score cannot be fairly compared to them.",
        "7. All observed site outcomes were seen in G4–G10, so new independent",
        "   folds are necessary for Q1 generalization claims.","",
        f"Result self hash: {result['result_sha256']}",""])
    return "\n".join(lines)


def run(device_name:str="cuda")->dict:
    if LOCK.exists() or RESULT.exists():
        raise G11Error("one formal G11 run only; existing lock/result")
    manifest=prepare()
    lock=_write(LOCK,{
        "protocol":PROTOCOL,"manifest_file_sha256":_sha256_file(MANIFEST),
        "one_retrospective_execution":True,
        "old_confirmation_labels_already_seen":True,
    })
    by_domain={}
    for domain in DOMAINS:
        data=g8._pack(domain)
        models,origin=g8._load_models(
            domain,{"model_and_role_provenance":manifest["model_and_role_provenance"]},
            device_name)
        ref=models[0]
        fit_rows=data["rows"]["fit"]
        group_index,keys=map_fit_rows_to_model_anchors(
            fit_rows,data["data"],ref,origin)
        seen=ref.encoder.seen_ap_mask.detach().cpu().bool()
        fit_rss=data["data"]["rss"][fit_rows].detach().cpu().float()
        frequencies=fit_site_ap_frequency(
            fit_rss,group_index,int(ref.n_anchors),seen)
        original_detection=ref.anchor_detection_prob.detach().cpu()[:,seen]
        if (original_detection.shape!=frequencies.shape or
            not torch.allclose(original_detection,frequencies,atol=1e-6,rtol=1e-6)):
            raise G11Error("FIT AP detection reconstruction differs from checkpoint memory")
        frequencies=original_detection
        by_domain[domain]={}
        for role in ROLES[domain]:
            by_domain[domain][role]=score_role(
                models,data["data"],data["rows"][role],origin,frequencies)
        del models
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result=_write(RESULT,{
        "protocol":PROTOCOL,"status":"TRAIN_INTERNAL_FIXED_RSS_IDENTIFIABILITY",
        "by_domain":by_domain,
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "lock_file_sha256":_sha256_file(LOCK),
        "lock_self_sha256":lock["result_sha256"],
        "no_fitting_on_cal_or_confirm":True,
        "original_frozen_models_only":True,
        "official_test_accessed":False,
        "previously_exposed_roles":True,
    })
    REPORT.write_text(_report(result),encoding="utf-8",newline="\n")
    return {"status":result["status"],"result_sha256":result["result_sha256"],
            "summary":{
                d:{role:r["summary"] for role,r in by_domain[d].items()}
                for d in DOMAINS}}


def check()->dict:
    if not MANIFEST.exists():return {"status":"NO_PREPARED_PLAN"}
    prepare()
    if not LOCK.exists():return {"status":"PLAN_FROZEN"}
    lock=_read(LOCK)
    if lock["manifest_file_sha256"]!=_sha256_file(MANIFEST):
        raise G11Error("lock manifest mismatch")
    if not RESULT.exists():return {"status":"LOCK_WITHOUT_RESULT_STOP"}
    result=_read(RESULT)
    if (result["lock_file_sha256"]!=_sha256_file(LOCK)
        or result["lock_self_sha256"]!=lock["result_sha256"]
        or result["manifest_file_sha256"]!=_sha256_file(MANIFEST)):
        raise G11Error("result provenance mismatch")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8")!=_report(result):
        raise G11Error("report does not match frozen JSON")
    return {"status":"VERIFIED","result_sha256":result["result_sha256"]}


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("preflight","prepare","run","check"):
        subcommand=sub.add_parser(name)
        if name=="run":
            subcommand.add_argument("--device",default="cuda")
    args=p.parse_args(argv)
    record=(preflight() if args.command=="preflight"
            else prepare() if args.command=="prepare"
            else run(args.device) if args.command=="run" else check())
    print(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
