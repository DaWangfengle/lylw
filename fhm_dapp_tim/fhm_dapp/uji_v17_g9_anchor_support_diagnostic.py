"""G9 retrospective mechanism test: anchor geometry, retrieval survival, probability mass.

Three original E5 panels; TRAIN-only historically exposed CAL/CONFIRM rows.
NO retraining, no new policy search, no official test, no statistical promotion.
Near-target diagnostics use labels ONLY for retrospective explanations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g8_hierarchical_attribution as g8
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _sha256_file, _write_json_new,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/"evidence/v17_g9_anchor_support_diagnostic"
DESIGN = EVIDENCE/"DESIGN_V17_G9.md"
MANIFEST = EVIDENCE/"manifest_v17_g9.json"
LOCK = EVIDENCE/"diagnosis_v17_g9.lock"
RESULT = EVIDENCE/"diagnosis_v17_g9.json"
REPORT = EVIDENCE/"RUN_REPORT_V17_G9.md"
PROTOCOL = "v17_g9_train_internal_anchor_support_attribution_20261010"
DOMAINS = g8.DOMAINS
ROLES = g8.ROLES
RADII_M = (5.0, 10.0)
T = 0.6
E, L, K = 5, 192, 192
BATCH = 16
EPS = 1e-5
MISSING = 1e8


class G9Error(RuntimeError):
    pass


def _write(path:Path, value:dict, field:str="result_sha256")->dict:
    if path.exists():
        raise FileExistsError(f"immutable G9 artifact already exists: {path}")
    item=dict(value)
    item[field]=_canonical_json_sha256(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_new(path,item)
    return item


def _read(path:Path,field:str="result_sha256")->dict:
    obj=json.loads(path.read_text(encoding="utf-8"))
    if obj.get(field)!=_canonical_json_sha256(obj,omit=field):
        raise G9Error(f"evidence self hash mismatch: {path}")
    return obj


def _source_hashes()->dict:
    paths={
        "design":DESIGN,
        "g9":Path(__file__),
        "g9_tests":ROOT.parent/"tests/test_uji_v17_g9_anchor_support_diagnostic.py",
        "g8":Path(g8.__file__),
        "r3":Path(r3.__file__),
        "r2b":Path(r2b.__file__),
        "r3p":Path(r3p.__file__),
    }
    return {k:_sha256_file(v) for k,v in paths.items()}


def preflight()->dict:
    domains={k:g8._pack(k)["original"] for k in DOMAINS}
    if (g8.T!=T or g8.E!=E or g8.K!=K or g8.g6.L!=L
        or g8.g6.ALPHA!=1.0 or r3p.STUDENT_DF!=4.0):
        raise G9Error("original R3 E5/L192/K192/alpha1 scientific settings changed")
    return {
        "protocol":PROTOCOL,"status":"G9_PRECHECKED",
        "domains":domains,"source_sha256":_source_hashes(),
        "radius_m":list(RADII_M),
        "temperature":T,
        "old_confirm_roles_previously_seen":True,
        "no_retraining":True,"no_provided_or_official_test":True,
    }


def prepare()->dict:
    check=preflight()
    manifest={
        "protocol":PROTOCOL,"source_sha256":check["source_sha256"],
        "original_panels_and_roles":check["domains"],
        "roles":{k:list(ROLES[k]) for k in DOMAINS},
        "radius_m":list(RADII_M),"temperature":T,
        "E":E,"L":L,"K":K,"alpha":g8.g6.ALPHA,
        "allowed_aggregate":"physical-coordinate macro; mean over rows per site",
        "diagnostic_data":"TRAIN-only historical CAL/CONFIRM, no new holdout",
        "no_training_no_candidate_selection":True,
        "support_definitions":"geometry: same building+floor FIT anchor XY; "
            "Top-L: union of five original member top192; "
            "Top-K: final 192 consensus anchor IDs; "
            "mass: original/post T0.6 consensus marginal on same-floor "
            "anchors within 5/10m true XY",
        "scoring_policy":"base and fixed anchor-only T0.6 means; no ES draw",
    }
    manifest["manifest_sha256"]=_canonical_json_sha256(manifest)
    if MANIFEST.exists():
        if _read(MANIFEST,"manifest_sha256")!=manifest:
            raise G9Error("G9 frozen source/protocol/roles changed")
        return manifest
    _write_json_new(MANIFEST,manifest)
    return manifest


def stage_metrics(
    anchor_xy:torch.Tensor, anchor_building:torch.Tensor, anchor_floor:torch.Tensor,
    true_xy:torch.Tensor, true_building:torch.Tensor, true_floor:torch.Tensor,
    top_l_indices:torch.Tensor, consensus_indices:torch.Tensor,
    consensus_prob:torch.Tensor, fixed_temperature:float=T,
    component_location:torch.Tensor|None=None,
)->dict[str,torch.Tensor]:
    """Per-row, label-facing descriptive diagnostic; never a prediction feature.

    Anchor coordinates and true coordinates must share the same centered
    metric frame. Candidate truth matching is *always* same-building+floor.
    Residual-adjusted component centres are a distinct alternative support
    and must never be conflated with dictionary anchor coordinate support.
    """
    b,n=consensus_indices.shape
    a=anchor_xy.shape[0]
    if (n!=K or top_l_indices.ndim!=3 or top_l_indices.shape!=(E,b,L)
        or anchor_xy.shape!=(a,2) or true_xy.shape!=(b,2)
        or anchor_building.shape!=(a,) or anchor_floor.shape!=(a,)
        or true_building.shape!=(b,) or true_floor.shape!=(b,)
        or consensus_prob.shape!=(b,K)):
        raise G9Error("stage diagnostic input dimensions changed")
    if (not torch.isfinite(consensus_prob).all()
        or (consensus_prob<0).any()
        or not torch.allclose(consensus_prob.sum(1),
                               torch.ones(b,device=consensus_prob.device,
                                          dtype=consensus_prob.dtype),atol=EPS,rtol=EPS)):
        raise G9Error("invalid consensus posterior")
    same=(anchor_building[None,:]==true_building[:,None]) & (
        anchor_floor[None,:]==true_floor[:,None])
    distance=torch.linalg.vector_norm(anchor_xy[None,:,:]-true_xy[:,None,:],dim=2)
    same_distance=torch.where(same,distance,torch.full_like(distance,float("inf")))
    fit_nearest=same_distance.min(dim=1).values
    same_floor_fit=same.any(dim=1)
    # The union contains FIVE member Top-L anchors, not five separately
    # counted virtual anchors. Duplicates are merged by global dictionary ID.
    union=torch.zeros((b,a),dtype=torch.bool,device=anchor_xy.device)
    union.scatter_(1,top_l_indices.permute(1,0,2).reshape(b,E*L),True)
    selected=torch.zeros_like(union)
    selected.scatter_(1,consensus_indices,True)
    if (top_l_indices.min()<0 or top_l_indices.max()>=a
        or consensus_indices.min()<0 or consensus_indices.max()>=a):
        raise G9Error("retrieval ID outside frozen FIT anchor memory")
    union_nearest=torch.where(union,same_distance,float("inf")).min(dim=1).values
    selected_nearest=torch.where(selected,same_distance,float("inf")).min(dim=1).values
    picked_distance=same_distance.gather(1,consensus_indices)
    top1=consensus_indices[:,0]
    # TOP-K marginal from the exact original R3 dense alpha consensus.
    sharp=torch.softmax(consensus_prob.clamp_min(1e-30).log()/fixed_temperature,dim=1)
    if component_location is not None:
        if component_location.shape!=(b,E*K,2):
            raise G9Error("residual-adjusted component centre dimensions changed")
        diff=torch.linalg.vector_norm(
            component_location.reshape(b,E,K,2)-true_xy[:,None,None,:],dim=-1)
        valid=same.gather(1,consensus_indices)
        component_nearest=torch.where(valid[:,None,:],diff,float("inf")).amin(dim=(1,2))
    else:
        component_nearest=torch.full_like(fit_nearest,float("inf"))
    out={
        "nearest_fit_m":fit_nearest,"nearest_member_union_m":union_nearest,
        "nearest_consensus_m":selected_nearest,
        "nearest_residual_centre_m":component_nearest,
        "same_floor_fit_exists":same_floor_fit.double(),
        "consensus_top1_matches_floor":same.gather(1,top1[:,None]).squeeze(1).double(),
        "consensus_top1_anchor_distance_m":picked_distance[:,0],
        "union_anchor_count":union.sum(dim=1).double(),
        "consensus_selected_outside_union_count":(
            selected&~union).sum(dim=1).double(),
    }
    for radius in RADII_M:
        key=f"r{int(radius)}"
        within=same_distance<=radius+EPS
        chosen=within.gather(1,consensus_indices)
        out[f"{key}_fit_exists"]=within.any(dim=1).double()
        union_hit=(within&union).any(dim=1)
        consensus_hit=chosen.any(dim=1)
        out[f"{key}_union_exists"]=union_hit.double()
        out[f"{key}_consensus_exists"]=consensus_hit.double()
        out[f"{key}_consensus_and_union_exists"]=(union_hit&consensus_hit).double()
        out[f"{key}_base_mass"]=(consensus_prob*chosen).sum(dim=1).double()
        out[f"{key}_anchor_t_mass"]=(sharp*chosen).sum(dim=1).double()
        if component_location is not None:
            out[f"{key}_residual_centre_exists"]=(component_nearest<=radius+EPS).double()
        out[f"{key}_top1_inside"]=chosen[:,0].double()
    return out


def _mean_vector(vals:torch.Tensor,gid:torch.Tensor,count:int)->list:
    """Physical site mean over finite observations; null if no valid row.

    A single missing Top-K candidate must not poison other valid RSS
    observations from the same physical coordinate.
    """
    x=vals.detach().cpu().double()
    groups=gid.detach().cpu().long()
    finite=torch.isfinite(x)
    total=torch.zeros(count,dtype=torch.float64)
    number=torch.zeros(count,dtype=torch.float64)
    total.scatter_add_(0,groups,torch.where(finite,x,0.))
    number.scatter_add_(0,groups,finite.double())
    return [float(v/n) if n else None for v,n in zip(total.tolist(),number.tolist())]


def _summary(site:dict, count:int)->dict:
    result={}
    for name,vec in site.items():
        defined=[float(x) for x in vec if x is not None]
        result[name]={
            "mean":float(np.mean(defined)) if defined else None,
            "defined_sites":len(defined),"total_sites":count,
        }
    for radius in RADII_M:
        tag=f"r{int(radius)}"
        fit=site[tag+"_fit_exists"]
        union=site[tag+"_union_exists"]
        cons=site[tag+"_consensus_exists"]
        # Average *fraction of rows* per physical location, not oracle hit.
        # A site is eligible if it is geometrically supported in FIT.
        denom=sum(1 for x in fit if x>0.5)
        result[tag+"_retrieval_recall_given_fit"]=(
            sum(x for x,f in zip(union,fit) if f>0.5)/denom if denom else None)
        result[tag+"_consensus_recall_given_fit"]=(
            sum(x for x,f in zip(cons,fit) if f>0.5)/denom if denom else None)
        selected_denom=sum(union)
        both=site[tag+"_consensus_and_union_exists"]
        result[tag+"_consensus_recall_given_union"]=(
            sum(both)/selected_denom if selected_denom else None)
    return result


@torch.no_grad()
def score_role(models:list,data:dict,rows:np.ndarray,origin:np.ndarray)->dict:
    """Reuse unmodified R3 retrieval; recompute consensus IDs independently.

    This run generates its own base/anchor-only exact MEAN errors, not old
    result files. No ES RNG because ES is not the G9 scientific question.
    """
    device=next(models[0].parameters()).device
    ref=models[0]
    n=len(rows)
    keys=[data["keys"][int(i)] for i in rows]
    gid,unique=r2b._coordinate_groups(keys)
    values:dict[str,torch.Tensor]={}
    true=torch.as_tensor(data["raw_xy"][rows]-origin[None,:],
                         device=device,dtype=torch.float32)
    bg=torch.as_tensor(data["building"][rows],dtype=torch.long,device=device)
    fl=torch.as_tensor(data["floor"][rows],dtype=torch.long,device=device)
    if not (torch.isfinite(true).all() and int(ref.n_anchors)>=K):
        raise G9Error("invalid centered target/anchor geometry")
    # Must use the same original FIT dictionary frame as r3._dense_joint_law.
    anchor_xy=ref.anchor_coords.to(device).float()
    ab=ref.anchor_buildings.to(device).long()
    af=ref.anchor_floors.to(device).long()
    for start in range(0,n,BATCH):
        end=min(start+BATCH,n);ids=rows[start:end];b=end-start
        rss=data["rss"][ids].float().to(device)
        prototype_distance=r2b._prototype_distance(ref,rss)
        indices=torch.arange(ref.n_anchors,device=device)[None,:].expand(b,-1)
        logits=[];centres=[];scales=[]
        for model in models:
            enc=model.encoder(rss)
            logits.append(model.retrieval_logits_from_embedding(enc))
            loc,sigma=model._components(enc,indices)
            centres.append(loc);scales.append(sigma)
        priors=torch.softmax(
            r3p.BETA_NEURAL*torch.stack(logits)
            -r3p.BETA_PROTOTYPE*prototype_distance[None,:,:],dim=-1)
        top,ix,rad,_=r3._member_top_l(
            priors,torch.stack(centres),torch.stack(scales),
            L,student_df=r3p.STUDENT_DF)
        p,loc,_=r3._dense_joint_law(
            top,ix,rad,torch.stack(centres),torch.stack(scales),
            tuple(range(E)),top_k=K,alpha=g8.g6.ALPHA)
        wm=rad.square().clamp_min(1e-8).reciprocal()
        wm=wm/wm.sum(dim=0,keepdim=True).clamp_min(1e-30)
        anchor_ids,c=r3.dense_alpha_consensus_prior(
            top,ix,wm,int(ref.n_anchors),K,
            epsilon=r3.CONSENSUS_EPSILON,alpha=g8.g6.ALPHA)
        # Abort if original R3 joint differs from the separately decoded
        # consensus anchor marginal. Never silently switch R3 operators.
        if not torch.allclose(p.reshape(b,E,K).sum(1),c,atol=3e-6,rtol=3e-6):
            raise G9Error("G9 consensus probability differs from frozen R3 law")
        reconstructed=(
            torch.stack(centres).permute(1,0,2,3)
            [:, :, :, :] # original member coordinates in same model frame
        )
        # Full component centre array has the original member-major/anchor-
        # minor order. Verify each final selected ID by exact gather.
        expected=torch.gather(
            reconstructed,2,
            anchor_ids[:,None,:,None].expand(b,E,K,2)
        ).reshape(b,E*K,2)
        if not torch.allclose(expected,loc,atol=1e-5,rtol=1e-6):
            raise G9Error("consensus index/location alignment failure")
        diag=stage_metrics(anchor_xy,ab,af,true[start:end],bg[start:end],
                           fl[start:end],ix,anchor_ids,c,
                           component_location=loc)
        # Compare exactly two fixed decisions to make source/diagnosis links.
        mean_base=(p[:,:,None]*loc).sum(1)
        sharp=torch.softmax(c.clamp_min(1e-30).log()/T,dim=1)
        means=(wm.transpose(0,1)[:,:,None,None]*
               sharp[:,None,:,None]*
               loc.reshape(b,E,K,2)).sum(dim=(1,2))
        diag["base_point_mae_m"]=torch.linalg.vector_norm(
            mean_base.double()-true[start:end].double(),dim=1)
        diag["anchor_t_point_mae_m"]=torch.linalg.vector_norm(
            means.double()-true[start:end].double(),dim=1)
        for name,metric in diag.items():
            if name not in values:
                values[name]=torch.empty(n,dtype=torch.float64)
            values[name][start:end]=metric.detach().cpu().double()
    site={name:_mean_vector(v,gid,len(unique)) for name,v in values.items()}
    summary=_summary(site,len(unique))
    return {
        "rows":n,"coordinate_groups":len(unique),
        "coordinate_keys":[list(k) for k in unique],
        "summary":summary,
        "site_vectors":site,
        "label_use":"retrospective analysis only; NEVER used as prediction feature",
    }


def _report(result:dict)->str:
    lines=[
        "# G9 | Anchor candidate survival and rank-vs-support diagnosis",
        "",
        "Retrospective FIT-only frozen E5 inference on previously exposed roles.",
        "**No new blind holdout; no fitted reliability model.**",
        "",
        "| Domain / historical role | Geometric FIT@5m | Top-L@5m | "
        "Consensus@5m | Base mass@5m | Anchor-T mass@5m | "
        "Base MAE m | Anchor-T MAE m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        for role,record in result["by_domain"][domain].items():
            s=record["summary"]
            def v(field):
                n=s[field]["mean"]
                return "n/a" if n is None else f"{n:.4f}"
            lines.append(
                f"| {domain}/{role} | {v('r5_fit_exists')} | "
                f"{v('r5_union_exists')} | {v('r5_consensus_exists')} | "
                f"{v('r5_base_mass')} | {v('r5_anchor_t_mass')} | "
                f"{v('base_point_mae_m')} | {v('anchor_t_point_mae_m')} |"
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The full FIT dictionary is an *anchor-geometry reference*, not a"
        " Bayes-optimal predictor or full continuous model support.",
        "Top-L union and consensus Top-K describe selected anchor IDs,"
        " not residual-adjusted continuous centres.",
        "Both geometric anchor support and learned residual-centre support"
        " are saved separately, with floor/building matching by anchor ID.",
        "A correct-neighbourhood mass is defined with held-out location labels"
        " and can NEVER be used as a deployable gating feature.",
        "Large mass of a wrong anchor does NOT by itself identify"
        " why the retrieval model assigned it weight.",
        "All results are descriptive for historical roles; no promotion,"
        " no new optimizer, no test access, no p-value.",
        "",
        f"Self hash: {result['result_sha256']}","",
    ])
    return "\n".join(lines)


def run(device_name:str="cuda")->dict:
    if LOCK.exists() or RESULT.exists():
        raise G9Error("G9 already locked or run; no repeats")
    manifest=prepare()
    lock=_write(LOCK,{
        "protocol":PROTOCOL,
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "old_labels_previously_exposed":True,
        "metric_radius_m":list(RADII_M),
    })
    by_domain={}
    for domain in DOMAINS:
        data=g8._pack(domain)
        models,origin=g8._load_models(
            domain,{"model_and_role_provenance":manifest["original_panels_and_roles"]},
            device_name)
        by_domain[domain]={}
        with r3.r2f_a1._deterministic_algorithms():
            for role in ROLES[domain]:
                by_domain[domain][role]=score_role(
                    models,data["data"],data["rows"][role],origin)
        del models
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result=_write(RESULT,{
        "protocol":PROTOCOL,
        "status":"RETROSPECTIVE_SUPPORT_VS_WEIGHTING_DIAGNOSTIC",
        "by_domain":by_domain,
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "lock_file_sha256":_sha256_file(LOCK),
        "lock_result_sha256":lock["result_sha256"],
        "radii_m":list(RADII_M),
        "no_model_training":True,"official_test_accessed":False,
        "old_confirmation_labels_previously_viewed":True,
    })
    REPORT.write_text(_report(result),encoding="utf-8",newline="\n")
    return {"status":result["status"],"result_sha256":result["result_sha256"],
            "role_summary":{
                domain:{role:r["summary"] for role,r in v.items()}
                for domain,v in by_domain.items()}}


def check()->dict:
    if not MANIFEST.exists():return {"status":"NO_FROZEN_PLAN"}
    prepare()
    if not LOCK.exists():return {"status":"PLAN_FROZEN"}
    lock=_read(LOCK)
    if lock["manifest_file_sha256"]!=_sha256_file(MANIFEST):
        raise G9Error("locked manifest changed")
    if not RESULT.exists():return {"status":"LOCK_WITHOUT_RESULT_STOP"}
    result=_read(RESULT)
    if (result["manifest_file_sha256"]!=_sha256_file(MANIFEST)
        or result["lock_file_sha256"]!=_sha256_file(LOCK)
        or result["lock_result_sha256"]!=lock["result_sha256"]):
        raise G9Error("result provenance differs")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8")!=_report(result):
        raise G9Error("report differs from result")
    return {"status":"VERIFIED","scientific_status":result["status"],
            "result_sha256":result["result_sha256"]}


def main(argv:Sequence[str]|None=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="command",required=True)
    for cmd in ("preflight","prepare","run","check"):
        item=sub.add_parser(cmd)
        if cmd=="run":item.add_argument("--device",default="cuda")
    ns=ap.parse_args(argv)
    answer=(preflight() if ns.command=="preflight"
            else prepare() if ns.command=="prepare"
            else run(ns.device) if ns.command=="run"
            else check())
    print(json.dumps(answer,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
