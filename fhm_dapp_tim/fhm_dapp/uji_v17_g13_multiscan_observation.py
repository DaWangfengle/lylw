"""G13: FIT-only multi-scan AP-detection and conditional-RSS evidence pilot.

One frozen two-coefficient observation model pooled on previously seen
TUJI/UTS CAL, tested descriptively on previously seen CONFIRM. No Tampere,
no official test, no new neural model training. UJIIndoorLoc is explicitly
reserved for a separate future independent protocol (different E5 panel).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g8_hierarchical_attribution as g8
from fhm_dapp_tim.fhm_dapp import uji_v17_g10_evidence_corrected_law as g10
from fhm_dapp_tim.fhm_dapp import uji_v17_g11_rss_identifiability as g11
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import common_student_t_randomness
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _sha256_file, _write_json_new,
)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"evidence/v17_g13_multiscan_observation"
DESIGN=OUT/"DESIGN_V17_G13.md"
MANIFEST=OUT/"manifest_v17_g13.json"
LOCK=OUT/"pilot_v17_g13.lock"
COEFFICIENTS=OUT/"coefficients_v17_g13.json"
RESULT=OUT/"evaluation_v17_g13.json"
REPORT=OUT/"RUN_REPORT_V17_G13.md"
PROTOCOL="g13_multiscan_fit_bernoulli_conditional_student_observation_v1"
DOMAINS=("tuji","uts")
ROLES=("calibration","confirmation")
VARIANTS=("base","anchor_t06","visibility","strength","joint")
E,L,K=5,192,192
TEMP=.6
KAPPA=4.
BETA=.5
DF_RSS=4.
SIGMA_FLOOR=.06          # normalized model RSS (~3.3 dBm), avoid overprecision
BATCH=12
TRAIN_BATCH=24
EPOCHS=24
LR=.05
COEFF_CAP=1.5
SEED=20261012
POINT_WEIGHT=.30
KL_WEIGHT=.10
SCALE_M=10.
MC_SEED=g8.SEED


class G13Error(RuntimeError):
    pass


def _record(value:dict,field:str="result_sha256")->dict:
    result=dict(value)
    result[field]=_canonical_json_sha256(result)
    return result


def _write(path:Path,value:dict,field:str="result_sha256")->dict:
    if path.exists():
        raise FileExistsError(f"immutable G13 artifact already exists: {path}")
    obj=_record(value,field)
    path.parent.mkdir(parents=True,exist_ok=True)
    _write_json_new(path,obj)
    return obj


def _read(path:Path,field:str="result_sha256")->dict:
    obj=json.loads(path.read_text(encoding="utf-8"))
    if obj.get(field)!=_canonical_json_sha256(obj,omit=field):
        raise G13Error(f"immutable G13 artifact hash mismatch: {path}")
    return obj


def _source_hashes()->dict:
    files={
        "G13_design":DESIGN,
        "G13_science":Path(__file__),
        "G13_tests":ROOT.parent/"tests/test_uji_v17_g13_multiscan_observation.py",
        "G8_model_protocol":Path(g8.__file__),
        "G10_existing_surrogate":Path(g10.__file__),
        "G11_FIT_to_anchor_bijection":Path(g11.__file__),
        "R3_original_law":Path(r3.__file__),
        "R2B_original_prototypes":Path(r2b.__file__),
        "R3_original_protocol":Path(r3p.__file__),
    }
    return {k:_sha256_file(v) for k,v in files.items()}


def _pack(domain:str)->dict:
    if domain not in DOMAINS:
        raise G13Error("G13 is ONLY TUJI+UTS multi-scan; no Tampere")
    p=g8._pack(domain)  # historic FIT/CAL/CONFIRM physical sites; no scores
    if any(len(p["rows"][role])==0 for role in ("fit",)+ROLES):
        raise G13Error("missing physical role rows")
    return p


def preflight()->dict:
    sources={d:_pack(d)["original"] for d in DOMAINS}
    if (g8.E!=E or g8.g6.L!=L or g8.K!=K or g8.T!=TEMP
        or r3p.STUDENT_DF!=4. or r3p.PREDICTIVE_DRAWS_PER_STREAM!=64):
        raise G13Error("historical V17 original protocol changed")
    return {
        "protocol":PROTOCOL,"status":"G13_FIT_ONLY_TWO_DOMAIN_READY",
        "domains":list(DOMAINS),"roles":list(ROLES),
        "original_E5_panel_provenance":sources,
        "source_sha256":_source_hashes(),
        "uiji_policy":"UJIIndoorLoc only after a separate truly independent protocol; do not access here",
        "tampere_in_study":False,
        "old_CONFIRM_labels_already_seen":True,
        "no_original_training_or_official_test":True,
    }


def prepare()->dict:
    info=preflight()
    config={
        "protocol":PROTOCOL,"source_sha256":info["source_sha256"],
        "original_E5_panel_provenance":info["original_E5_panel_provenance"],
        "domains":list(DOMAINS),"roles":list(ROLES),
        "variants":list(VARIANTS),"fixed_temp":TEMP,
        "multi_scan":{"missing_cutoff_model_rss":-0.999,
            "global_reference_strength":"detected observations only",
            "ap_bernoulli_beta_base":BETA,"per_anchor_shrinkage_pseudoscans":KAPPA,
            "rss_student_df":DF_RSS,"normalized_min_rss_sigma":SIGMA_FLOOR,
            "model_seen_aps_only":True,
            "not_independent_scans_assumption":"repeated scans correlated; descriptive pilot"},
        "features":"per query K192 candidate demeaned standard-deviation-stabilized "
                   "log Bernoulli detection and conditional Student-t RSS likelihood",
        "training":{"train_on":"historical TUJI/UTS CAL only",
            "sites_equal_weight":True,"domains_equal_weight":True,
            "initial_coefficients":[0.,0.],"nonnegative_cap":COEFF_CAP,
            "seed":SEED,"optimizer":"Adam","lr":LR,"epochs":EPOCHS,
            "batch":TRAIN_BATCH,
            "loss":"Dirac component-centre ES + 0.30 point Euclidean MAE, "
                   "divided by 10m, +0.10 KL to anchor-only T0.6"},
        "evaluation":{"roles":["historical confirmation only"],
            "continuous_student_df":r3p.STUDENT_DF,
            "continuous_ES_MC_seed":MC_SEED,
            "draws_each_of_two_streams":r3p.PREDICTIVE_DRAWS_PER_STREAM,
            "preserve_member_centres_scales":True},
        "primary_comparison":"joint minus anchor_t06, MAE+ES, p90/p95, worsening sites",
        "descriptive_screen":"TUJI and UTS each >=0.15m MAE gain and ES nonworsening",
        "scientific_status":"historical exposed domain evidence, NOT independent test",
        "tampere_excluded":True,"uji_reserved_untouched":True,
    }
    config["manifest_sha256"]=_canonical_json_sha256(config)
    if MANIFEST.is_file():
        if _read(MANIFEST,"manifest_sha256")!=config:
            raise G13Error("immutable scientific manifest differs")
        return config
    _write_json_new(MANIFEST,config)
    return config


def build_fit_observation_memory(
    fit_rss:torch.Tensor, fit_anchor_ids:torch.Tensor,
    anchor_count:int,seen:torch.Tensor,
    kappa:float=KAPPA,
)->dict[str,torch.Tensor]:
    """Shrink multi-scan per-anchor Bernoulli visibility and conditional RSS.

    Missing is NOT an observed weak RSS. FIT alone supplies site prototypes.
    Global priors use FIT rows only. There are no validation/target labels.
    """
    x=fit_rss.detach().cpu().float()
    ids=fit_anchor_ids.detach().cpu().long()
    seen=seen.detach().cpu().bool()
    if (x.ndim!=2 or ids.shape!=(len(x),) or seen.shape!=(x.shape[1],)
        or anchor_count<1 or len(x)<anchor_count or not torch.isfinite(x).all()
        or not bool(seen.any()) or float(kappa)<=0.):
        raise G13Error("invalid FIT observation inputs")
    if ids.min()<0 or ids.max()>=anchor_count:
        raise G13Error("FIT indices outside original anchor dictionary")
    x=x[:,seen]
    mask=(x>-0.999)
    binary=mask.float()
    a=anchor_count;p=x.shape[1]
    counts=torch.bincount(ids,minlength=a).float()
    if (counts<=0).any():
        raise G13Error("FIT anchor lacks scans")
    detections=torch.zeros(a,p)
    detections.index_add_(0,ids,binary)
    global_vis=(detections.sum(0)+BETA)/(float(len(x))+2*BETA)
    # A hierarchical Beta prior with strength kappa; global rate learned
    # from FIT itself, not an unrelated uniform 50% AP prior.
    prob=(detections+kappa*global_vis[None,:])/(counts[:,None]+kappa)
    prob=prob.clamp(1e-5,1.-1e-5)
    sum_x=torch.zeros(a,p)
    sum_sq=torch.zeros(a,p)
    measured=torch.where(mask,x,0.)
    sum_x.index_add_(0,ids,measured)
    sum_sq.index_add_(0,ids,measured.square())
    glob_d=detections.sum(0)
    glob_mu=measured.sum(0)/glob_d.clamp_min(1)
    glob_sq=measured.square().sum(0)/glob_d.clamp_min(1)
    glob_var=(glob_sq-glob_mu.square()).clamp_min(SIGMA_FLOOR**2)
    # Conditional RSS posterior shrinkage, not a global [-1] imputation.
    mu=(sum_x+kappa*glob_mu[None,:])/(detections+kappa)
    centred_ss=(sum_sq-2*mu*sum_x+mu.square()*detections)
    # Prior shifts keep a single-scan anchor from having zero variance.
    prior_ss=kappa*(glob_var[None,:]+(glob_mu[None,:]-mu).square())
    variance=((centred_ss+prior_ss)/(detections+kappa)).clamp(
        min=SIGMA_FLOOR**2,max=1.)
    if not (torch.isfinite(prob).all() and torch.isfinite(mu).all()
            and torch.isfinite(variance).all()):
        raise G13Error("nonfinite FIT observation memory")
    return {
        "detection_probability":prob,
        "conditional_mean":mu,
        "conditional_variance":variance,
        "fit_rows_per_anchor":counts,
        "detections_per_anchor":detections,
        "model_ap_count":torch.tensor(p),
        "global_detection_probability":global_vis,
    }


def _stabilize(x:torch.Tensor)->torch.Tensor:
    centred=x-x.mean(dim=1,keepdim=True)
    # Stability floor prevents amplifying essentially identical candidate
    # scores into artificially high-confidence contrast.
    return centred/torch.sqrt(
        centred.square().mean(dim=1,keepdim=True)+0.20**2)


def observation_features(
    query_rss:torch.Tensor, candidate_ids:torch.Tensor,
    seen:torch.Tensor, memory:dict[str,torch.Tensor],
)->torch.Tensor:
    """Per candidate, independent Bernoulli detect and conditional-t RSS terms.

    No query building/floor/XY, device identifier, or target labels used.
    The candidate identity is the original R3 Top-K ID, never an oracle ID.
    """
    if query_rss.ndim!=2 or candidate_ids.ndim!=2:
        raise G13Error("invalid observation query geometry")
    device=query_rss.device
    seen=seen.to(device).bool()
    x=query_rss[:,seen].float()
    ids=candidate_ids.long()
    if (len(x)!=len(ids) or x.shape[1]!=
        int(memory["model_ap_count"]) or ids.min()<0 or
        ids.max()>=len(memory["detection_probability"])):
        raise G13Error("query AP schema / candidate IDs mismatch")
    z=x>-0.999
    pi=memory["detection_probability"].to(device)[ids]
    mu=memory["conditional_mean"].to(device)[ids]
    var=memory["conditional_variance"].to(device)[ids]
    vis=torch.where(z[:,None,:],pi.log(),torch.log1p(-pi)).mean(dim=2)
    # Student-t log p(observed RSS | detected, anchor), common constants
    # cancel across anchors. Visibility term handles absent APs separately.
    diff=(x[:,None,:]-mu).square()/(DF_RSS*var)
    log_strength=-(DF_RSS+1.)/2.*torch.log1p(diff)-.5*var.log()
    sig=(z[:,None,:]*log_strength).sum(dim=2)/z.sum(dim=1).clamp_min(1)[:,None]
    sig=torch.where(z.any(1)[:,None],sig,torch.zeros_like(sig))
    features=torch.stack([_stabilize(vis),_stabilize(sig)],dim=2)
    if not torch.isfinite(features).all():
        raise G13Error("nonfinite observation likelihood features")
    return features


def probability(c:torch.Tensor,features:torch.Tensor,
                coeff:torch.Tensor)->torch.Tensor:
    if (c.ndim!=2 or features.shape!=(len(c),c.shape[1],2)
        or coeff.shape!=(2,) or (coeff<0).any() or (coeff>COEFF_CAP+1e-6).any()):
        raise G13Error("invalid G13 fixed anchor evidence coefficients")
    logit=c.clamp_min(1e-30).log()/TEMP+(features*coeff).sum(-1)
    result=torch.softmax(logit,dim=1)
    if not torch.isfinite(result).all():
        raise G13Error("nonfinite observation-updated posterior")
    return result


@torch.no_grad()
def extract(models:list,data:dict,rows:np.ndarray,origin:np.ndarray,
            memory:dict, scoring:bool=False)->dict:
    """Evaluate original frozen R3 exactly, then query FIT observation model."""
    ref=models[0];device=next(ref.parameters()).device
    row_keys=[data["keys"][int(i)] for i in rows]
    gid,keys=r2b._coordinate_groups(row_keys)
    y=torch.as_tensor(data["raw_xy"][rows]-origin[None,:],
                      device=device,dtype=torch.float64)
    seen=ref.encoder.seen_ap_mask.detach().to(device).bool()
    if (seen.sum()!=memory["model_ap_count"] or len(rows)<1 or
        int(ref.n_anchors)<K):
        raise G13Error("FIT observation AP mapping mismatch")
    accum={k:[] for k in ("c","member","features","centres",
                          "component_p","component_xy","component_scale")}
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0,len(rows),BATCH):
            end=min(start+BATCH,len(rows))
            b=end-start
            rss=data["rss"][rows[start:end]].float().to(device)
            proto=r2b._prototype_distance(ref,rss)
            idx=torch.arange(ref.n_anchors,device=device)[None,:].expand(b,-1)
            logits=[];xy_all=[];scale_all=[]
            for model in models:
                embedding=model.encoder(rss)
                logits.append(model.retrieval_logits_from_embedding(embedding))
                location,scale=model._components(embedding,idx)
                xy_all.append(location);scale_all.append(scale)
            priors=torch.softmax(
                r3p.BETA_NEURAL*torch.stack(logits)-
                r3p.BETA_PROTOTYPE*proto[None,:,:],dim=2)
            top,ix,rad,_=r3._member_top_l(
                priors,torch.stack(xy_all),torch.stack(scale_all),L,
                student_df=r3p.STUDENT_DF)
            joint,locations,sigma=r3._dense_joint_law(
                top,ix,rad,torch.stack(xy_all),torch.stack(scale_all),
                tuple(range(E)),top_k=K,alpha=g8.g6.ALPHA)
            wm=joint.reshape(b,E,K).sum(2)
            c=joint.reshape(b,E,K).sum(1)
            old_w=rad.square().clamp_min(1e-8).reciprocal()
            old_w=old_w/old_w.sum(dim=0,keepdim=True).clamp_min(1e-30)
            candidate_ids,independent_c=r3.dense_alpha_consensus_prior(
                top,ix,old_w,int(ref.n_anchors),K,
                epsilon=r3.CONSENSUS_EPSILON,alpha=g8.g6.ALPHA)
            if not torch.allclose(c,independent_c,rtol=3e-6,atol=3e-6):
                raise G13Error("original R3 marginal/candidate alignment failure")
            features=observation_features(rss,candidate_ids,seen,memory)
            selected_centres=(
                wm[:,:,None,None]*locations.reshape(b,E,K,2)).sum(dim=1)
            for k,value in (
                ("c",c),("member",wm),("features",features),
                ("centres",selected_centres),
                ("component_p",joint),("component_xy",locations),
                ("component_scale",sigma)):
                if not scoring and k.startswith("component_"):
                    continue
                accum[k].append(value.detach().cpu())
    return {**{k:torch.cat(v,dim=0) for k,v in accum.items() if v},
            "target":y.detach().cpu(),"gid":gid.detach().cpu(),
            "keys":[list(key) for key in keys],"rows":len(rows),
            "sites":len(keys)}


def _row_weights(gid:torch.Tensor,sites:int)->torch.Tensor:
    size=torch.bincount(gid.long(),minlength=sites).float()
    if (size<=0).any():raise G13Error("empty CAL physical site")
    return 1./(size[gid.long()]*sites)


def fit(cal:dict[str,dict],device_name:str)->dict:
    """Exactly one pooled, nonnegative 2-scalar fit on CAL only."""
    torch.manual_seed(SEED)
    device=torch.device(device_name)
    raw=torch.nn.Parameter(torch.zeros(2,device=device))
    optimizer=torch.optim.Adam([raw],lr=LR)
    losses=[]
    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        total_loss=0.
        for domain in DOMAINS:
            p=cal[domain]
            rw=_row_weights(p["gid"],p["sites"]).to(device)
            for start in range(0,p["rows"],TRAIN_BATCH):
                end=min(start+TRAIN_BATCH,p["rows"])
                c=p["c"][start:end].to(device)
                z=p["features"][start:end].to(device)
                means=p["centres"][start:end].to(device)
                y=p["target"][start:end].to(device).float()
                theta=raw.clamp(0.,COEFF_CAP)
                q=probability(c,z,theta)
                baseline=torch.softmax(c.clamp_min(1e-30).log()/TEMP,dim=1)
                # Existing G10 surrogate: exact Dirac-mixture proper energy,
                # not the continuous Student-t ES scored at confirmation.
                es=g10.discrete_energy_loss(q,means,y)
                point=torch.linalg.vector_norm(
                    (q[:,:,None]*means).sum(1)-y,dim=1)
                kl=(q*(q.clamp_min(1e-30).log()-
                       baseline.clamp_min(1e-30).log())).sum(1)
                loss=((es+POINT_WEIGHT*point)/SCALE_M+
                      KL_WEIGHT*kl)
                term=.5*(loss*rw[start:end]).sum()
                if not torch.isfinite(term):
                    raise G13Error("nonfinite train-only CAL objective")
                term.backward()
                total_loss+=float(term.detach().cpu())
        optimizer.step()
        with torch.no_grad():
            raw.clamp_(0.,COEFF_CAP)
        losses.append(total_loss)
    theta=raw.detach().cpu().tolist()
    return {
        "protocol":PROTOCOL,
        "coefficients":{"visibility":theta[0],"strength":theta[1]},
        "train_loss_by_epoch":losses,
        "epochs":EPOCHS,"lr":LR,
        "fitting_roles":{d:{"CAL_rows":cal[d]["rows"],
                            "CAL_sites":cal[d]["sites"]} for d in DOMAINS},
        "no_confirmation_fitting":True,
        "train_loss_not_continuous_Student_t_ES":True,
        "source_candidate_anchor_only_T":TEMP,
    }


def _variant_probabilities(pack:dict,start:int,end:int,theta:torch.Tensor,
                           device:torch.device)->dict[str,torch.Tensor]:
    original=pack["component_p"][start:end].to(device)
    f=pack["features"][start:end].to(device)
    p=original.reshape(end-start,E,K)
    w=p.sum(2);c=p.sum(1)
    # Keep original BASE float32 exact, use original G8 anchor-only law.
    anchor=g8.hierarchical_laws(original,E,K)["anchor"]
    zero=torch.zeros_like(theta)
    qv=probability(c,f,torch.stack((theta[0],zero[1])))
    qs=probability(c,f,torch.stack((zero[0],theta[1])))
    qj=probability(c,f,theta)
    def joint(q):
        return (w[:,:,None]*q[:,None,:]).reshape(end-start,E*K)
    if not torch.allclose(
        joint(probability(c,f,zero)),anchor,atol=3e-6,rtol=3e-6):
        raise G13Error("zero learned evidence fails to reproduce strong baseline")
    return {"base":original,"anchor_t06":anchor,"visibility":joint(qv),
            "strength":joint(qs),"joint":joint(qj)}


def score(pack:dict,theta:list[float],device_name:str)->dict:
    device=torch.device(device_name)
    coeff=torch.tensor(theta,dtype=torch.float32,device=device)
    n=pack["rows"]
    rnd=common_student_t_randomness(
        n,draws=r3p.PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=r3p.STUDENT_DF,seed=MC_SEED)
    scores={v:{"mae":torch.empty(n,dtype=torch.float64),
               "es":torch.empty(n,dtype=torch.float64)} for v in VARIANTS}
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0,n,BATCH):
            end=min(start+BATCH,n)
            q=_variant_probabilities(pack,start,end,coeff,device)
            loc=pack["component_xy"][start:end].to(device)
            sigma=pack["component_scale"][start:end].to(device)
            target=pack["target"][start:end].to(device)
            random=g8.g6._random_slice(rnd,start,end,device)
            for name,prior in q.items():
                es,mae=r3._score_joint_law(prior,loc,sigma,target,random)
                scores[name]["mae"][start:end]=mae.double().detach().cpu()
                scores[name]["es"][start:end]=es.double().detach().cpu()
    grouped=pack["gid"];sites=pack["sites"]
    out={}
    for name in VARIANTS:
        mae,site_mae=r3._metric_record(scores[name]["mae"],grouped,sites)
        es,site_es=r3._metric_record(scores[name]["es"],grouped,sites)
        ma=site_mae.tolist();ee=site_es.tolist()
        out[name]={
            "macro_mae_m":float(mae["coordinate_macro_m"]),
            "macro_ES_m":float(es["coordinate_macro_m"]),
            "p90_site_mae_m":float(np.quantile(ma,.9)),
            "p95_site_mae_m":float(np.quantile(ma,.95)),
            "worst_site_mae_m":max(ma),
            "site_mae_m":ma,"site_ES_m":ee,
        }
    old=np.asarray(out["anchor_t06"]["site_mae_m"])
    new=np.asarray(out["joint"]["site_mae_m"])
    old_es=np.asarray(out["anchor_t06"]["site_ES_m"])
    new_es=np.asarray(out["joint"]["site_ES_m"])
    return {
        "rows":n,"physical_sites":sites,"physical_coordinate_keys":pack["keys"],
        "methods":out,
        "joint_minus_anchor":{
            "delta_mae_m":float((new-old).mean()),
            "delta_ES_m":float((new_es-old_es).mean()),
            "sites_improve":int((new<old).sum()),
            "sites_worsen":int((new>old).sum()),
            "largest_added_site_error_m":float((new-old).max()),
        },
        "MC_seed":MC_SEED,"draws_per_stream":r3p.PREDICTIVE_DRAWS_PER_STREAM,
    }


def _report(result:dict)->str:
    lines=[
        "# G13 | Multi-scan AP visibility × conditional RSS evidence",
        "",
        "**Retrospective TUJI/UTS only; all CONFIRM labels previously exposed.**",
        "No Tampere, no official UJIIndoorLoc test, no E5 retraining.",
        "Only two pooled coefficients learned on original CAL.",
        "",
        "| Domain | Policy | Macro MAE m | Full Student-t ES m | P90 m | P95 m |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        for name in VARIANTS:
            v=result["confirm"][domain]["methods"][name]
            lines.append(f"| {domain} | {name} | {v['macro_mae_m']:.6f} | "
                         f"{v['macro_ES_m']:.6f} | "
                         f"{v['p90_site_mae_m']:.6f} | {v['p95_site_mae_m']:.6f} |")
    lines.extend(["","## FIT-scan support and learned evidence coefficients",""])
    for domain in DOMAINS:
        st=result["fit_observation_stats"][domain]
        lines.append(f"- {domain}: FIT scans per reference site "
                     f"min/median/max={st['scan_min']}/{st['scan_median']}/{st['scan_max']}; "
                     f"sites with >=2 scans={st['sites_at_least_two']}/{st['sites']}")
    lines.append(f"- learned coefficients: {result['coefficients']['coefficients']}")
    lines.append("")
    lines.append(f"**{result['status']}** (DESCRIPTIVE screening, not promotion).")
    lines.append("")
    lines.append("AP detection clipping at -1, conditional independence,")
    lines.append("FIT scan correlation, heterogeneous devices and reused")
    lines.append("historical CONFIRM outcomes limit causal/generalization claims.")
    lines.append("UJIIndoorLoc needs a separate, unviewed prospective protocol.")
    lines.append("")
    lines.append(f"Result SHA: {result['result_sha256']}")
    lines.append("")
    return "\n".join(lines)


def run(device_name:str="cuda")->dict:
    if LOCK.exists() or RESULT.exists() or COEFFICIENTS.exists():
        raise G13Error("G13 already locked or fitted: no repeated confirmation")
    manifest=prepare()
    lock=_write(LOCK,{
        "protocol":PROTOCOL,"manifest_file_sha256":_sha256_file(MANIFEST),
        "before_any_CONFIRM_scoring":True,
        "multi_scan_domains":list(DOMAINS)})
    packs={}
    cal={}
    stats={}
    for domain in DOMAINS:
        p=_pack(domain)
        models,origin=g8._load_models(
            domain,{"model_and_role_provenance":
                    manifest["original_E5_panel_provenance"]},device_name)
        ref=models[0]
        fit_ids=p["rows"]["fit"]
        fit_rss=p["data"]["rss"][fit_ids]
        fit_group,_=g11.map_fit_rows_to_model_anchors(
            fit_ids,p["data"],ref,origin)
        memory=build_fit_observation_memory(
            fit_rss,fit_group,int(ref.n_anchors),
            ref.encoder.seen_ap_mask.detach().cpu())
        counts=memory["fit_rows_per_anchor"].tolist()
        stats[domain]={
            "sites":len(counts),"scan_min":int(min(counts)),
            "scan_median":float(np.median(counts)),"scan_max":int(max(counts)),
            "sites_at_least_two":sum(x>=2 for x in counts),
            "sites_at_least_five":sum(x>=5 for x in counts),
            "model_seen_APs":int(memory["model_ap_count"]),
        }
        # CAL ONLY until fit() is finalized: no confirmation feature or
        # target cached, selected, or used in the optimizer.
        cal[domain]=extract(models,p["data"],p["rows"]["calibration"],
                            origin,memory,scoring=False)
        packs[domain]={"data":p["data"],"rows":p["rows"],
                       "models":models,"origin":origin,"memory":memory}
    learned=fit(cal,device_name)
    frozen=_write(COEFFICIENTS,{
        **learned,"lock_file_sha256":_sha256_file(LOCK),
        "manifest_file_sha256":_sha256_file(MANIFEST)})
    coeff=[frozen["coefficients"]["visibility"],frozen["coefficients"]["strength"]]
    confirm={}
    for domain in DOMAINS:
        p=packs[domain]
        eval_data=extract(p["models"],p["data"],p["rows"]["confirmation"],
                          p["origin"],p["memory"],scoring=True)
        confirm[domain]=score(eval_data,coeff,device_name)
        del p["models"]
        if torch.cuda.is_available():torch.cuda.empty_cache()
    passed=all(
        confirm[d]["joint_minus_anchor"]["delta_mae_m"]<=-.15
        and confirm[d]["joint_minus_anchor"]["delta_ES_m"]<=0
        for d in DOMAINS)
    result=_write(RESULT,{
        "protocol":PROTOCOL,
        "status":("PROMISING_RETROSPECTIVE_EVIDENCE" if passed
                  else "NO_RETROSPECTIVE_ADVANCEMENT"),
        "descriptive_screen_passed":passed,
        "confirm":confirm,
        "fit_observation_stats":stats,
        "coefficients":frozen,
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "lock_file_sha256":_sha256_file(LOCK),
        "coefficients_file_sha256":_sha256_file(COEFFICIENTS),
        "not_new_independent_confirmation":True,
        "no_tampere":True,"UJIIndoorLoc_reserved_for_future":True,
        "no_original_E5_training":True,
    })
    REPORT.write_text(_report(result),encoding="utf-8",newline="\n")
    return {
        "status":result["status"],"result_sha256":result["result_sha256"],
        "learned_coefficients":frozen["coefficients"],
        "domain_comparisons":{d:confirm[d]["joint_minus_anchor"] for d in DOMAINS},
    }


def check()->dict:
    if not MANIFEST.exists():return {"status":"UNPREPARED"}
    prepare()
    if not LOCK.exists():return {"status":"FROZEN_READY"}
    lock=_read(LOCK)
    if lock["manifest_file_sha256"]!=_sha256_file(MANIFEST):
        raise G13Error("lock provenance mismatch")
    if not RESULT.exists():
        return {"status":"G13_LOCK_WITHOUT_RESULT_STOP"}
    fitted=_read(COEFFICIENTS)
    result=_read(RESULT)
    if (result["coefficients_file_sha256"]!=_sha256_file(COEFFICIENTS)
        or result["manifest_file_sha256"]!=_sha256_file(MANIFEST)
        or result["lock_file_sha256"]!=_sha256_file(LOCK)
        or fitted["manifest_file_sha256"]!=_sha256_file(MANIFEST)):
        raise G13Error("G13 score or coefficient identity changed")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8")!=_report(result):
        raise G13Error("G13 report does not reproduce result")
    return {"status":"VERIFIED","result_sha256":result["result_sha256"],
            "scientific_status":result["status"]}


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="command",required=True)
    for action in ("preflight","prepare","run","check"):
        sub=subs.add_parser(action)
        if action=="run":sub.add_argument("--device",default="cuda")
    args=parser.parse_args(argv)
    payload=(preflight() if args.command=="preflight" else
             prepare() if args.command=="prepare" else
             run(args.device) if args.command=="run" else check())
    print(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
