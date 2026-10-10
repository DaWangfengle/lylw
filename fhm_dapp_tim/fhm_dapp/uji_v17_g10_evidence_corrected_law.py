"""G10: tiny, cross-domain RSS-evidence correction to frozen V17 anchor law.

Exactly ONE prespecified fit on historical TUJI+UTS CAL physical sites,
then evaluation on historical TUJI+UTS+Tampere CONFIRM. No FIT-label
recalibration, no temperature search, no test data, no prospective claims.
Original E5 models, centres, Student-t scales and R3 support are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from fhm_dapp_tim.fhm_dapp import uji_v17_g8_hierarchical_attribution as g8
from fhm_dapp_tim.fhm_dapp import uji_v17_g9_anchor_support_diagnostic as g9
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import common_student_t_randomness
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _sha256_file, _write_json_new,
)

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"evidence/v17_g10_evidence_corrected_law"
DESIGN=EVIDENCE/"DESIGN_V17_G10.md"
MANIFEST=EVIDENCE/"manifest_v17_g10.json"
LOCK=EVIDENCE/"train_v17_g10.lock"
FIT=EVIDENCE/"fitted_coefficients_v17_g10.json"
RESULT=EVIDENCE/"evaluation_v17_g10.json"
REPORT=EVIDENCE/"RUN_REPORT_V17_G10.md"
PROTOCOL="v17_g10_cross_domain_evidence_corrected_anchor_law_20261010"
DOMAINS=("tuji","uts","tampere")
LEARN_DOMAINS=("tuji","uts")
FEATURE_NAMES=("rss_prototype_agreement","member_topL_support",
               "member_prior_log_agreement","local_spatial_probability_coherence")
NFEATURES=len(FEATURE_NAMES)
E,L,K=5,192,192
T=0.6
RADIUS_M=10.0
CAP=1.5
SEED=20261010
EPOCHS=24
LR=0.06
TRAIN_BATCH=24
INFER_BATCH=16
POINT_WEIGHT=0.30
KL_WEIGHT=0.10
SCALE_M=10.0
VARIANTS=("base","anchor_t06","evidence_correction")


class G10Error(RuntimeError):
    pass


def _write(path:Path,record:dict,field:str="result_sha256")->dict:
    if path.exists():
        raise FileExistsError(f"G10 immutable artifact exists: {path}")
    value=dict(record);value[field]=_canonical_json_sha256(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    _write_json_new(path,value)
    return value


def _read(path:Path,field:str="result_sha256")->dict:
    value=json.loads(path.read_text(encoding="utf-8"))
    if value.get(field)!=_canonical_json_sha256(value,omit=field):
        raise G10Error(f"immutable evidence hash mismatch: {path}")
    return value


def _source_sha()->dict:
    paths={"design":DESIGN,"g10":Path(__file__),
           "tests":ROOT.parent/"tests/test_uji_v17_g10_evidence_corrected_law.py",
           "g9":Path(g9.__file__),"g8":Path(g8.__file__),
           "r3":Path(r3.__file__),"r2b":Path(r2b.__file__),
           "r3_protocol":Path(r3p.__file__),
           "sampler":Path(common_student_t_randomness.__code__.co_filename)}
    return {name:_sha256_file(path) for name,path in paths.items()}


def preflight()->dict:
    orig={name:g8._pack(name)["original"] for name in DOMAINS}
    if (g8.T!=T or g8.E!=E or g8.K!=K or g8.g6.L!=L
        or g8.g6.ALPHA!=1.0 or r3p.STUDENT_DF!=4.0
        or r3p.PREDICTIVE_DRAWS_PER_STREAM!=64 or g8.SEED!=19602):
        raise G10Error("original frozen G8/G9 member/score protocol changed")
    if not all("calibration" in g8.ROLES[d] for d in LEARN_DOMAINS):
        raise G10Error("only two original disjoint historical CAL roles may train G10")
    return {
        "protocol":PROTOCOL,
        "status":"ORIGINAL_E5_CAL_ONLY_LEARNING_READY",
        "panel_provenance":orig,"source_sha256":_source_sha(),
        "learner_domains":list(LEARN_DOMAINS),
        "assessment_domains":list(DOMAINS),
        "all_assessment_labels_historically_viewed":True,
        "official_or_provided_test_access":False,
        "frozen_base_networks_retrained":False,
    }


def prepare()->dict:
    checked=preflight()
    manifest={
        "protocol":PROTOCOL,"source_sha256":checked["source_sha256"],
        "panels":checked["panel_provenance"],
        "learner_domains":list(LEARN_DOMAINS),
        "learner_role":"calibration",
        "assessment_domains":list(DOMAINS),
        "assessment_role":"confirmation",
        "feature_names":list(FEATURE_NAMES),
        "model":"shared four-feature bounded log posterior correction",
        "anchor_temperature":T,"radius_m":RADIUS_M,"theta_cap":CAP,
        "seed":SEED,"epochs":EPOCHS,"learning_rate":LR,
        "train_batch":TRAIN_BATCH,"infer_batch":INFER_BATCH,
        "point_weight":POINT_WEIGHT,"kl_weight":KL_WEIGHT,
        "loss_scale_m":SCALE_M,"E":E,"L":L,"K":K,
        "continuous_test_score_random_seed":g8.SEED,
        "scoring_draws_per_stream":r3p.PREDICTIVE_DRAWS_PER_STREAM,
        "scoring_variants":list(VARIANTS),
        "no_training_on_fit_or_confirmation":True,
        "tampere_never_used_to_fit_or_select":True,
        "history_exposed_not_independent_confirmation":True,
        "no_validation_based_model_selection":True,
    }
    manifest["manifest_sha256"]=_canonical_json_sha256(manifest)
    if MANIFEST.exists():
        if _read(MANIFEST,"manifest_sha256")!=manifest:
            raise G10Error("G10 source and prespecified hyperparameters differ")
        return manifest
    _write_json_new(MANIFEST,manifest)
    return manifest


def _row_zscore(x:torch.Tensor)->torch.Tensor:
    x=x-x.mean(dim=1,keepdim=True)
    return (x/x.square().mean(dim=1,keepdim=True).add(1e-4).sqrt()).clamp(-3,3)


def label_free_features(
    prototypes:torch.Tensor, full_priors:torch.Tensor,
    member_top_indices:torch.Tensor, consensus_indices:torch.Tensor,
    consensus_probability:torch.Tensor, anchor_coords:torch.Tensor,
    anchor_building:torch.Tensor, anchor_floor:torch.Tensor,
)->torch.Tensor:
    """NO true labels or CAL/CONFIRM coordinates accepted in this API.

    Values are per-query standardized, bounded, shared across datasets
    with differing AP counts, reference densities and coordinate origins.
    """
    n,k=consensus_indices.shape
    m=full_priors.shape[0]
    a=anchor_coords.shape[0]
    if (m!=E or k!=K or member_top_indices.shape!=(E,n,L)
        or prototypes.shape!=(n,a) or full_priors.shape!=(E,n,a)
        or anchor_coords.shape!=(a,2)
        or anchor_building.shape!=(a,) or anchor_floor.shape!=(a,)
        or consensus_probability.shape!=(n,k)):
        raise G10Error("wrong G10 label-free feature tensor geometry")
    if ((consensus_indices<0).any() or (consensus_indices>=a).any()
        or (member_top_indices<0).any() or (member_top_indices>=a).any()):
        raise G10Error("candidate global index invalid")
    if not torch.isfinite(consensus_probability).all():
        raise G10Error("nonfinite consensus evidence")
    prototype=prototypes.gather(1,consensus_indices).float()
    rss_agreement=_row_zscore(-prototype)
    support=(member_top_indices[:,:,:,None] ==
             consensus_indices[None,:,None,:]).any(dim=2).float().mean(dim=0)
    support=(2.0*(support-support.mean(dim=1,keepdim=True))).clamp(-3,3)
    ids=consensus_indices[None,:,:].expand(E,n,k)
    member_log=full_priors.gather(2,ids).clamp_min(1e-30).log()
    member_disagreement=member_log.float().std(dim=0,unbiased=False)
    member_agreement=_row_zscore(-member_disagreement)
    xy=anchor_coords[consensus_indices].float()
    building=anchor_building[consensus_indices]
    floor=anchor_floor[consensus_indices]
    pair_distance=torch.cdist(xy,xy)
    valid=(pair_distance<=RADIUS_M) & (
        building[:,:,None]==building[:,None,:]) & (
        floor[:,:,None]==floor[:,None,:])
    near_mass=(valid.float()*consensus_probability[:,None,:]).sum(dim=2)
    neighbor_count=valid.sum(dim=2).clamp_min(1)
    # Density-corrected local probability coherence; not simply number
    # of reference anchors clustered in a dense corridor.
    local=torch.log(near_mass.clamp_min(1e-30)/neighbor_count)
    local_coherence=_row_zscore(local)
    feature=torch.stack((rss_agreement,support,member_agreement,
                         local_coherence),dim=-1).float()
    if feature.shape!=(n,K,NFEATURES) or not torch.isfinite(feature).all():
        raise G10Error("invalid calibrated evidence features")
    return feature


def anchor_posterior(consensus:torch.Tensor,features:torch.Tensor,
                     bounded_theta:torch.Tensor)->torch.Tensor:
    if (consensus.ndim!=2 or features.shape!=(*consensus.shape,NFEATURES)
        or bounded_theta.shape!=(NFEATURES,)):
        raise G10Error("posterior feature/parameter shape mismatch")
    logits=consensus.clamp_min(1e-30).log()/T
    logits=logits+torch.einsum("nkf,f->nk",features,bounded_theta)
    q=torch.softmax(logits,dim=1)
    if not torch.isfinite(q).all():
        raise G10Error("evidence correction produced non-finite probabilities")
    return q


def original_joint(member:torch.Tensor,candidate:torch.Tensor)->torch.Tensor:
    if member.ndim!=2 or member.shape[1]!=E or candidate.shape!=(len(member),K):
        raise G10Error("original member / shared anchor shapes changed")
    return (member[:,:,None]*candidate[:,None,:]).reshape(len(member),E*K)


@torch.no_grad()
def extract(models:list,data:dict,rows:np.ndarray,origin:np.ndarray,
            scoring:bool=False)->dict:
    """Read frozen R3 panel once; each row carries original physical key.

    Training extraction is only invoked with original calibration row IDs.
    """
    device=next(models[0].parameters()).device
    ref=models[0]
    n=len(rows)
    keys=[data["keys"][int(i)] for i in rows]
    gid,unique=r2b._coordinate_groups(keys)
    target=torch.as_tensor(data["raw_xy"][rows]-origin[None,:],
                           dtype=torch.float64,device=device)
    out={name:[] for name in ("c","wm","features","centres","component_probability",
                             "component_location","component_scale")}
    origin_tensor=torch.as_tensor(origin,dtype=torch.float32,device=device)
    if (int(ref.n_anchors)<K or not torch.isfinite(target).all()
        or ref.anchor_coords.shape!=(ref.n_anchors,2)):
        raise G10Error("old FIT memory invalid")
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0,n,INFER_BATCH):
            end=min(start+INFER_BATCH,n);b=end-start
            rss=data["rss"][rows[start:end]].float().to(device)
            proto=r2b._prototype_distance(ref,rss)
            ids=torch.arange(ref.n_anchors,device=device)[None,:].expand(b,-1)
            logits=[];locations=[];scales=[]
            for model in models:
                embedding=model.encoder(rss)
                logits.append(model.retrieval_logits_from_embedding(embedding))
                loc,sigma=model._components(embedding,ids)
                locations.append(loc);scales.append(sigma)
            prior=torch.softmax(r3p.BETA_NEURAL*torch.stack(logits)
                -r3p.BETA_PROTOTYPE*proto[None,:,:],dim=-1)
            top,ix,rad,_=r3._member_top_l(
                prior,torch.stack(locations),torch.stack(scales),L,
                student_df=r3p.STUDENT_DF)
            joint,xy,sigma=r3._dense_joint_law(
                top,ix,rad,torch.stack(locations),torch.stack(scales),
                tuple(range(E)),top_k=K,alpha=g8.g6.ALPHA)
            wm=joint.reshape(b,E,K).sum(dim=2)
            c=joint.reshape(b,E,K).sum(dim=1)
            mr=rad.square().clamp_min(1e-8).reciprocal()
            mr=mr/mr.sum(dim=0,keepdim=True).clamp_min(1e-30)
            aid,probs=r3.dense_alpha_consensus_prior(
                top,ix,mr,int(ref.n_anchors),K,
                epsilon=r3.CONSENSUS_EPSILON,alpha=g8.g6.ALPHA)
            if not torch.allclose(c,probs,rtol=3e-6,atol=3e-6):
                raise G10Error("G10 independent consensus marginal differs from R3")
            gathered=torch.gather(
                torch.stack(locations).permute(1,0,2,3),2,
                aid[:,None,:,None].expand(b,E,K,2)).reshape(b,E*K,2)
            if not torch.allclose(xy,gathered,atol=1e-5,rtol=1e-6):
                raise G10Error("G10 R3 component identity changed")
            feats=label_free_features(
                proto,prior,ix,aid,c,ref.anchor_coords.to(device),
                ref.anchor_buildings.to(device),ref.anchor_floors.to(device))
            centre=(wm[:,:,None,None]*
                    xy.reshape(b,E,K,2)).sum(dim=1)
            if not torch.allclose(joint,original_joint(wm,c),atol=3e-6,rtol=3e-6):
                raise G10Error("original member-anchor law was not factorizable")
            for name,item in (("c",c),("wm",wm),("features",feats),
                              ("centres",centre),
                              ("component_probability",joint if scoring else torch.empty(0)),
                              ("component_location",xy if scoring else torch.empty(0)),
                              ("component_scale",sigma if scoring else torch.empty(0))):
                if scoring or name not in ("component_probability","component_location","component_scale"):
                    out[name].append(item.detach().cpu().contiguous())
    result={name:torch.cat(v,dim=0) for name,v in out.items() if v}
    result.update({"target":target.detach().cpu(),"gid":gid,
                   "keys":[list(k) for k in unique],
                   "rows":n,"sites":len(unique)})
    return result


def _row_weights(group:torch.Tensor,sites:int)->torch.Tensor:
    cnt=torch.bincount(group,minlength=sites).float()
    if (cnt<=0).any():raise G10Error("empty CAL physical site")
    return (1.0/cnt[group])/(sites*len(LEARN_DOMAINS))


def bounded_parameters(raw:torch.Tensor)->torch.Tensor:
    return CAP*torch.tanh(raw)


def discrete_energy_loss(q:torch.Tensor,centres:torch.Tensor,
                         truth:torch.Tensor)->torch.Tensor:
    """Proper energy score on the Dirac mixture at frozen member means.

    This surrogate is NOT the continuous Student-t Energy Score used to
    evaluate final deployment predictions. It is differentiable and
    introduces no trainable component locations or scales.
    """
    dist_y=torch.linalg.vector_norm(centres-truth[:,None,:],dim=2)
    pair=torch.cdist(centres,centres)
    return (q*dist_y).sum(dim=1)-.5*torch.einsum("bi,bij,bj->b",q,pair,q)


def calibration_objective(q:torch.Tensor,c:torch.Tensor,centres:torch.Tensor,
                          truth:torch.Tensor)->torch.Tensor:
    baseline=torch.softmax(c.clamp_min(1e-30).log()/T,dim=1)
    energy=discrete_energy_loss(q,centres,truth)
    point=torch.linalg.vector_norm(
        (q[:,:,None]*centres).sum(dim=1)-truth,dim=1)
    kl=(q*(q.clamp_min(1e-30).log()-baseline.clamp_min(1e-30).log())).sum(1)
    return (energy+POINT_WEIGHT*point)/SCALE_M+KL_WEIGHT*kl


def fit_once(cal:dict[str,dict],device_name:str)->dict:
    """One 4-parameter pooled fit; no validation or best-epoch selection."""
    torch.manual_seed(SEED)
    dev=torch.device(device_name)
    raw=nn.Parameter(torch.zeros(NFEATURES,dtype=torch.float32,device=dev))
    optim=torch.optim.Adam([raw],lr=LR)
    summary=[]
    for epoch in range(EPOCHS):
        optim.zero_grad(set_to_none=True)
        total=0.
        # Loss averaged by physical sites, then the two domains equally.
        for domain in LEARN_DOMAINS:
            pack=cal[domain]
            weights=_row_weights(pack["gid"],pack["sites"])
            for start in range(0,pack["rows"],TRAIN_BATCH):
                end=min(start+TRAIN_BATCH,pack["rows"])
                feat=pack["features"][start:end].to(dev)
                c=pack["c"][start:end].to(dev)
                centres=pack["centres"][start:end].to(dev)
                y=pack["target"][start:end].to(dev).float()
                q=anchor_posterior(c,feat,bounded_parameters(raw))
                loss=(calibration_objective(q,c,centres,y)*
                      weights[start:end].to(dev)).sum()
                if not torch.isfinite(loss):
                    raise G10Error("nonfinite G10 training objective")
                loss.backward()
                total+=float(loss.detach().cpu())
        optim.step()
        summary.append(float(total))
    learned=bounded_parameters(raw).detach().cpu().tolist()
    if not all(math.isfinite(v) for v in learned):
        raise G10Error("nonfinite learned coefficients")
    return {"feature_names":list(FEATURE_NAMES),
            "raw_coefficients":raw.detach().cpu().tolist(),
            "bounded_coefficients":learned,
            "train_loss_by_epoch":summary,
            "epochs":EPOCHS,"seed":SEED,
            "learner_domains":list(LEARN_DOMAINS),
            "calibration_sites":{d:int(cal[d]["sites"]) for d in LEARN_DOMAINS},
            "calibration_rows":{d:int(cal[d]["rows"]) for d in LEARN_DOMAINS},
            "no_confirmation_gradients":True}


def score_confirmation(pack:dict,theta:list[float],device_name:str)->dict:
    dev=torch.device(device_name)
    n=pack["rows"]
    # Preserve the original R3 float32 law; multiplying rounded marginals
    # a second time changes the baseline and zero-coefficient comparator.
    original_probability=pack["component_probability"]
    coeff=torch.tensor(theta,dtype=torch.float32,device=dev)
    rnd=common_student_t_randomness(
        n,draws=r3p.PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=r3p.STUDENT_DF,seed=g8.SEED)
    scores={v:{"mae":torch.empty(n,dtype=torch.float64),
               "es":torch.empty(n,dtype=torch.float64)} for v in VARIANTS}
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0,n,INFER_BATCH):
            end=min(start+INFER_BATCH,n)
            original=original_probability[start:end].to(dev)
            w=original.reshape(end-start,E,K).sum(dim=2)
            c=original.reshape(end-start,E,K).sum(dim=1)
            f=pack["features"][start:end].to(dev)
            locations=pack["component_location"][start:end].to(dev)
            scales=pack["component_scale"][start:end].to(dev)
            target=pack["target"][start:end].to(dev)
            frozen=g8.hierarchical_laws(original,E,K)["anchor"]
            corrected=original_joint(w,anchor_posterior(c,f,coeff))
            if not torch.count_nonzero(coeff) and not torch.equal(corrected,frozen):
                raise G10Error("zero coefficients do not exactly reproduce Anchor T=.6")
            laws={"base":original,"anchor_t06":frozen,
                  "evidence_correction":corrected}
            random=g8.g6._random_slice(rnd,start,end,dev)
            for name,prob in laws.items():
                es,mae=r3._score_joint_law(prob,locations,scales,target,random)
                scores[name]["mae"][start:end]=mae.detach().cpu().double()
                scores[name]["es"][start:end]=es.detach().cpu().double()
    gid=pack["gid"]
    out={}
    for name in VARIANTS:
        mae,vm=r3._metric_record(scores[name]["mae"],gid,pack["sites"])
        es,ve=r3._metric_record(scores[name]["es"],gid,pack["sites"])
        mvec=vm.tolist();evec=ve.tolist()
        out[name]={
            "coordinate_macro_mae_m":float(mae["coordinate_macro_m"]),
            "coordinate_macro_es_m":float(es["coordinate_macro_m"]),
            "coordinate_p90_mae_m":float(np.quantile(mvec,.9)),
            "coordinate_p95_mae_m":float(np.quantile(mvec,.95)),
            "row_mean_mae_m":float(mae["row_mean_m"]),
            "row_mean_es_m":float(es["row_mean_m"]),
            "coordinate_vector_mae_m":mvec,
            "coordinate_vector_es_m":evec,
        }
    old=np.asarray(out["anchor_t06"]["coordinate_vector_mae_m"])
    new=np.asarray(out["evidence_correction"]["coordinate_vector_mae_m"])
    oldes=np.asarray(out["anchor_t06"]["coordinate_vector_es_m"])
    newes=np.asarray(out["evidence_correction"]["coordinate_vector_es_m"])
    comparisons={
        "learned_minus_anchor_mae_m":float((new-old).mean()),
        "learned_minus_anchor_es_m":float((newes-oldes).mean()),
        "sites_mae_improved":int((new<old).sum()),
        "sites_mae_worsened":int((new>old).sum()),
        "sites_es_improved":int((newes<oldes).sum()),
        "sites_es_worsened":int((newes>oldes).sum()),
        "largest_site_mae_harm_m":float((new-old).max()),
        "largest_site_mae_gain_m":float((old-new).max()),
    }
    return {"rows":n,"physical_sites":pack["sites"],
            "physical_coordinate_keys":pack["keys"],
            "methods":out,"relative_to_frozen_anchor":comparisons,
            "mc_seed":g8.SEED,"draws_per_stream":r3p.PREDICTIVE_DRAWS_PER_STREAM}


def _report(result:dict)->str:
    lines=[
        "# G10: learned RSS/spatial evidence correction — retrospective pilot","",
        "**TRAIN ONLY:** Original TUJI + UTS CAL physical locations.",
        "**No fitting:** Original E5 FIT rows, any CONFIRM rows, all Tampere rows.",
        "**No independent confirmation:** All assessed domain confirmation",
        "labels were seen in G4–G9; no p-values or method promotion.","",
        "| CONFIRM domain | Method | MAE m | ES m | P90 error m | P95 error m |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        record=result["confirmation"][domain]
        for name in VARIANTS:
            x=record["methods"][name]
            lines.append(
                f"| {domain} | {name} | "
                f"{x['coordinate_macro_mae_m']:.6f} | "
                f"{x['coordinate_macro_es_m']:.6f} | "
                f"{x['coordinate_p90_mae_m']:.6f} | "
                f"{x['coordinate_p95_mae_m']:.6f} |")
    lines.extend(["","## Frozen Anchor-only contrasts",""])
    for domain in DOMAINS:
        x=result["confirmation"][domain]["relative_to_frozen_anchor"]
        lines.append(
            f"- {domain}: ΔMAE {x['learned_minus_anchor_mae_m']:+.6f}m; "
            f"ΔES {x['learned_minus_anchor_es_m']:+.6f}m; "
            f"MAE improve/harm {x['sites_mae_improved']}/{x['sites_mae_worsened']}; "
            f"worst harm +{x['largest_site_mae_harm_m']:.4f}m")
    lines.extend(["",
        "Four shared coefficients were learned only from two original CAL",
        "roles using a fixed Dirac-mixture Energy Score + point-MAE + KL",
        "surrogate. This train loss is NOT the continuous Student-t ES.",
        "The three assessment domains were scored with paired continuous",
        "Student-t 64+64 MC draws and original G8 sampling seed.",
        "No validation selection, no adaptive temperature, no original",
        "member-centre/scale update, no new blind holdout. A future truly",
        "independent experiment is required before Q1/generalization claims.",
        "",
        f"Result self hash: {result['result_sha256']}",""])
    return "\n".join(lines)


def run(device_name:str="cuda")->dict:
    if LOCK.exists() or FIT.exists() or RESULT.exists():
        raise G10Error("G10 fitting already attempted; no repetitions")
    manifest=prepare()
    lock=_write(LOCK,{
        "protocol":PROTOCOL,"manifest_file_sha256":_sha256_file(MANIFEST),
        "CAL_only_training":list(LEARN_DOMAINS),
        "confirmation_previous_exposure":True,
        "single_run":True,
    })
    cal={}
    # Strict causal order: do not load CONFIRM labels until the final
    # four fitted coefficients have been immutably saved.
    for domain in LEARN_DOMAINS:
        data=g8._pack(domain)
        models,origin=g8._load_models(
            domain,{"model_and_role_provenance":manifest["panels"]},device_name)
        cal[domain]=extract(models,data["data"],data["rows"]["calibration"],
                            origin,scoring=False)
        del models
        if torch.cuda.is_available():torch.cuda.empty_cache()
    learned=fit_once(cal,device_name)
    fitted=_write(FIT,{
        "protocol":PROTOCOL,"manifest_file_sha256":_sha256_file(MANIFEST),
        "lock_file_sha256":_sha256_file(LOCK),
        "fit":learned,"calibration_only":True,
        "tampere_label_free":True,"no_confirmation_seen_at_fit":True,
    })
    del cal
    confirmation={}
    for domain in DOMAINS:
        data=g8._pack(domain)
        models,origin=g8._load_models(
            domain,{"model_and_role_provenance":manifest["panels"]},device_name)
        pack=extract(models,data["data"],data["rows"]["confirmation"],
                     origin,scoring=True)
        confirmation[domain]=score_confirmation(
            pack,learned["bounded_coefficients"],device_name)
        del models,pack
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result=_write(RESULT,{
        "protocol":PROTOCOL,
        "status":"ONE_POOLED_CAL_FIT_THREE_RETROSPECTIVE_CONFIRMS",
        "confirmation":confirmation,
        "fit_self_sha256":fitted["result_sha256"],
        "fit_file_sha256":_sha256_file(FIT),
        "lock_file_sha256":_sha256_file(LOCK),
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "trained_original_member_networks":False,
        "old_confirm_labels_seen_historically":True,
        "official_test_used":False,
        "future_prospective_reproduction_needed":True,
    })
    REPORT.write_text(_report(result),encoding="utf-8",newline="\n")
    return {"status":result["status"],"result_sha256":result["result_sha256"],
            "fit_coefficients":learned["bounded_coefficients"],
            "assessment":{domain:confirmation[domain]["relative_to_frozen_anchor"]
                          for domain in DOMAINS}}


def check()->dict:
    if not MANIFEST.exists():return {"status":"UNPREPARED"}
    prepare()
    if not LOCK.exists():return {"status":"PLAN_FROZEN"}
    lock=_read(LOCK)
    if lock["manifest_file_sha256"]!=_sha256_file(MANIFEST):
        raise G10Error("G10 locked plan changed")
    if not FIT.exists():return {"status":"LOCK_WITHOUT_FIT_STOP"}
    fit=_read(FIT)
    if (fit["manifest_file_sha256"]!=_sha256_file(MANIFEST)
        or fit["lock_file_sha256"]!=_sha256_file(LOCK)):
        raise G10Error("fitted coefficients provenance changed")
    if not RESULT.exists():return {"status":"FITTED_WITHOUT_RESULT_STOP"}
    result=_read(RESULT)
    if (result["fit_self_sha256"]!=fit["result_sha256"]
        or result["fit_file_sha256"]!=_sha256_file(FIT)
        or result["lock_file_sha256"]!=_sha256_file(LOCK)
        or result["manifest_file_sha256"]!=_sha256_file(MANIFEST)):
        raise G10Error("result provenance changed")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8")!=_report(result):
        raise G10Error("report not reproducible")
    return {"status":"VERIFIED","scientific_status":result["status"],
            "result_sha256":result["result_sha256"]}


def main(argv:Sequence[str]|None=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="command",required=True)
    for name in ("preflight","prepare","run","check"):
        p=sub.add_parser(name)
        if name=="run":p.add_argument("--device",default="cuda")
    ns=ap.parse_args(argv)
    output=(preflight() if ns.command=="preflight"
            else prepare() if ns.command=="prepare"
            else run(ns.device) if ns.command=="run" else check())
    print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
