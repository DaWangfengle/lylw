"""G5-A: preregistered Tampere TRAIN-only V17 posterior Bayes action pilot.

No external provided test is opened. The predictive law and all five model
weights are COMMON to all decisions. Compare frozen exact mixture mean against
weighted component-center median and sampled continuous Student-t median.
Only 75/25 physical-coordinate-stratified train/confirmation; NO tuning.
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

from fhm_dapp_tim.fhm_dapp import v18_4_datasets as ds
from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as r0
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness,
    sample_continuous_mixture,
    spatial_median_from_draws,
)
from fhm_dapp_tim.fhm_dapp.anchor import load_anchor_checkpoint, location_keys
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _save_torch_new, _sha256_file, _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EVIDENCE = ROOT / "evidence/v17_g5_bayes_action"
MODELS = ROOT / "models/v17_g5_bayes_action"
DESIGN = EVIDENCE / "DESIGN_V17_G5.md"
MANIFEST = EVIDENCE / "manifest_v17_g5.json"
FREEZE = EVIDENCE / "freeze_v17_g5.json"
LOCK = EVIDENCE / "internal_confirmation.lock"
RESULT = EVIDENCE / "confirmation_v17_g5.json"
REPORT = EVIDENCE / "RUN_REPORT_V17_G5.md"
TRAIN_RSS = ds.DATA_ROOT / "Tampere/Training_rss.csv"
TRAIN_XYZ = ds.DATA_ROOT / "Tampere/Training_coordinates.csv"

PROTOCOL = "v17_g5_tampere_train_only_bayes_action_v1_20261009"
SPLIT_SEED = 19501
BOOTSTRAP_SEED = 19502
ACTION_SAMPLING_SEED = 19503
RISK_SAMPLING_SEED = 19504
ENERGY_SAMPLING_SEED = 19505
PANEL = 5
SEEDS = g1.SEEDS[:PANEL]
EPOCHS = g1.EPOCHS[:PANEL]
L, K, ALPHA = 192, 192, 1.0
FIT_FRACTION = 0.75
BATCH = 16
ACTION_DRAWS_PER_STREAM = 128
RISK_DRAWS_PER_STREAM = 128
MEDIAN_ITERATIONS = 64
MEDIAN_TOLERANCE_M = 1e-5
MIN_GAIN_M = 0.15
BOOTSTRAP_REPS = 5000
VARIANTS = ("mean", "component_median", "continuous_median")


class G5Error(RuntimeError):
    pass


def _write(path: Path, value: dict, field: str = "result_sha256") -> dict:
    if path.exists():
        raise FileExistsError(f"immutable G5 artifact already exists: {path}")
    record = dict(value)
    record[field] = _canonical_json_sha256(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_new(path, record)
    return record


def _read(path: Path, field: str = "result_sha256") -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get(field) != _canonical_json_sha256(value, omit=field):
        raise G5Error(f"G5 integrity failure: {path}")
    return value


def _source_hashes() -> dict:
    sources = {
        "training_rss": TRAIN_RSS,
        "training_xyz": TRAIN_XYZ,
        "g5_design": DESIGN,
        "g5_code": Path(__file__),
        "tampere_adapter": Path(ds.__file__),
        "v17_training": Path(r0.__file__),
        "v17_score": Path(r3.__file__),
        "v17_score_protocol": Path(r3p.__file__),
        "v17_anchor_loader": Path(load_anchor_checkpoint.__code__.co_filename),
        "v9_continuous_sampler_and_median": Path(spatial_median_from_draws.__code__.co_filename),
    }
    return {k:_sha256_file(v) for k,v in sources.items()}


def _as_r0_input(canonical) -> dict:
    if canonical.dataset != "Tampere" or canonical.split != "train":
        raise G5Error("G5 uses Tampere training files only")
    canonical.validate()
    rss = torch.as_tensor(canonical.model_rss(),dtype=torch.float32)
    coords = np.asarray(canonical.coords_m,dtype=np.float64)
    building = np.asarray(canonical.building,dtype=np.int64)
    floor = np.asarray(canonical.floor,dtype=np.int64)
    # Tampere has no handset column. Existing adapter uses -1 unknown; V17
    # train-device remapping safely assigns one constant device category.
    phone = np.asarray(canonical.device,dtype=np.int64)
    if len(coords)<100 or not np.isfinite(coords).all() or not torch.isfinite(rss).all():
        raise G5Error("Tampere training RSS or locations invalid")
    if np.any(building<0) or np.any(floor<0):
        raise G5Error("unseen train building or floor is illegal")
    return {
        "rss":rss,"raw_xy":coords,"building":building,"floor":floor,
        "phone":phone,"keys":location_keys(coords,building,floor),
        "stats":{"n_aps":int(rss.shape[1]),
                 "n_buildings":int(building.max())+1,
                 "n_floors":int(floor.max())+1,
                 "adapted_from":"V18.4 Tampere TRAIN only; z is floor category"},
    }


def _roles(data:dict) -> tuple[dict[str,set[tuple]],dict[str,np.ndarray]]:
    by_floor=defaultdict(set)
    for k in data["keys"]:
        by_floor[(int(k[0]),int(k[1]))].add(k)
    groups={"fit":set(),"confirmation":set()}
    for stratum,sites in sorted(by_floor.items()):
        if len(sites)<4:
            raise G5Error(f"too few sites for fit/confirmation on stratum {stratum}")
        ordered=sorted(sites,key=lambda k:(
            hashlib.sha256(f"{SPLIT_SEED}|{k!r}".encode("utf-8")).digest(),k
        ))
        cut=max(2,min(len(ordered)-2,int(math.floor(FIT_FRACTION*len(ordered)))))
        groups["fit"].update(ordered[:cut])
        groups["confirmation"].update(ordered[cut:])
    if groups["fit"]&groups["confirmation"] or (
        groups["fit"]|groups["confirmation"]!=set(data["keys"])
    ):
        raise G5Error("training and confirmation physical coordinate leakage")
    rows={r:np.asarray([i for i,k in enumerate(data["keys"]) if k in sites],dtype=np.int64)
          for r,sites in groups.items()}
    if len(groups["fit"])<max(L,K) or not all(len(x)>0 for x in rows.values()):
        raise G5Error("Tampere FIT anchors or confirmation rows insufficient")
    if sum(len(v) for v in rows.values())!=len(data["keys"]):
        raise G5Error("train-file partition is not row exhaustive")
    # Fail early if R0's float32 centering + four-decimal anchor dictionary
    # merges distinct physical locations or divides the same physical site.
    origin=data["raw_xy"][rows["fit"]].mean(axis=0).astype(np.float32)
    xy=(data["raw_xy"]-origin[None,:]).astype(np.float32)
    transformed=location_keys(xy,data["building"],data["floor"])
    fw,rev={},{}
    for i in rows["fit"]:
        a,b=data["keys"][int(i)],transformed[int(i)]
        if (a in fw and fw[a]!=b) or (b in rev and rev[b]!=a):
            raise G5Error("R0 centered float32 creates physical-site collision")
        fw[a],rev[b]=b,a
    if len(fw)!=len(groups["fit"]):
        raise G5Error("FIT reference anchor dictionary collapsed")
    return groups,rows


def _training() -> tuple[dict,dict[str,set[tuple]],dict[str,np.ndarray]]:
    canonical,_reference=ds.load_tampere("train")
    data=_as_r0_input(canonical)
    groups,rows=_roles(data)
    return data,groups,rows


def member_path(index:int) -> Path:
    if not 1<=int(index)<=PANEL:
        raise G5Error("members must be 1..5")
    return MODELS / f"member_{index:02d}_seed{SEEDS[index-1]}.pt"


def preflight() -> dict:
    data,groups,rows=_training()
    return {
        "status":"TAMPERE_TRAIN_ONLY_READY","protocol":PROTOCOL,
        "source_sha256":_source_hashes(),
        "train_rows":len(data["keys"]),"train_aps":int(data["rss"].shape[1]),
        "train_coordinate_groups":len(set(data["keys"])),
        "floor_count":len(set(data["floor"].tolist())),
        "role_groups":{k:len(v) for k,v in groups.items()},
        "role_rows":{k:len(v) for k,v in rows.items()},
        "physical_site_overlap":len(groups["fit"]&groups["confirmation"]),
        "frozen_seeds":list(SEEDS),"frozen_epochs":list(EPOCHS),
        "model_count":PANEL,"L":L,"K":K,"alpha":ALPHA,
        "action_draws_per_stream":ACTION_DRAWS_PER_STREAM,
        "other_provided_test_files_accessed":False,
    }


def prepare() -> dict:
    info=preflight()
    _,groups,_=_training()
    record={
        "protocol":PROTOCOL,"scientific_question":"Euclidean-MAE Bayes action vs mixture mean",
        "source_sha256":info["source_sha256"],
        "role_coordinate_keys":{k:[list(x) for x in sorted(s)] for k,s in groups.items()},
        "role_groups":info["role_groups"],"role_rows":info["role_rows"],
        "model_count":PANEL,"seeds":list(SEEDS),"epochs":list(EPOCHS),
        "L":L,"K":K,"alpha":ALPHA,"split_seed":SPLIT_SEED,
        "train_fraction":FIT_FRACTION,
        "variants":list(VARIANTS),
        "action_sampling_seed":ACTION_SAMPLING_SEED,
        "action_draws_per_stream":ACTION_DRAWS_PER_STREAM,
        "risk_sampling_seed":RISK_SAMPLING_SEED,
        "risk_draws_per_stream":RISK_DRAWS_PER_STREAM,
        "median_iterations":MEDIAN_ITERATIONS,
        "median_tolerance_m":MEDIAN_TOLERANCE_M,
        "energy_sampling_seed":ENERGY_SAMPLING_SEED,
        "bootstrap_seed":BOOTSTRAP_SEED,
        "bootstrap_repetitions":BOOTSTRAP_REPS,
        "minimum_mae_gain_m":MIN_GAIN_M,
        "training_recipe":copy.deepcopy(FROZEN_TRAINING),
        "provided_test_use":"forbidden (Tampere Test_rss/coordinates, TUJI test, UJI official, UTS_test)",
    }
    record["manifest_sha256"]=_canonical_json_sha256(record)
    if MANIFEST.exists():
        stored=_read(MANIFEST,"manifest_sha256")
        if stored!=record:
            raise G5Error("G5 frozen manifest or train source differs")
        return stored
    _write_json_new(MANIFEST,record)
    return record


def train_member(i:int,device_name:str="cuda") -> dict:
    if FREEZE.exists() or LOCK.exists() or RESULT.exists():
        raise G5Error("G5 model training forbidden after freeze")
    manifest=prepare()
    path=member_path(i)
    if path.exists():
        existing=torch.load(path,map_location="cpu",weights_only=True)
        if existing.get("manifest_sha256")!=manifest["manifest_sha256"] or existing.get("member_index")!=i:
            raise G5Error("incompatible preexisting G5 member")
        return {"member":i,"sha256":_sha256_file(path),"reused":True}
    data,groups,rows=_training()
    started=time.perf_counter()
    model,stats,origin,device_mapping,history=r0._fit_refit(
        data,rows["fit"],seed=SEEDS[i-1],epochs=EPOCHS[i-1],
        device=torch.device(device_name),log_every=10
    )
    if int(model.n_anchors)!=len(groups["fit"]):
        raise G5Error("trained R0 anchor memory does not match FIT sites")
    config=copy.deepcopy(FROZEN_TRAINING)
    config.update({
        "model_schema":"v17_g5_tampere_original_atomic",
        "training_protocol_version":PROTOCOL,
        "n_devices":len(device_mapping),
        "train_all":True,"epochs":EPOCHS[i-1],
        "validation_fraction":0.0,"calibration_fraction":0.0,
    })
    payload={
        "protocol":PROTOCOL,"manifest_sha256":manifest["manifest_sha256"],
        "member_index":i,"member_seed":SEEDS[i-1],"epoch":EPOCHS[i-1],
        "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "config":config,"stats":stats,
        "coord_origin":np.asarray(origin,dtype=np.float32),
        "device_id_mapping":{str(k):v for k,v in device_mapping.items()},
        "refit_history":history,
        "data_provenance":{"fit_coordinate_only":True,
            "training_rss_file":"Tampere/Training_rss.csv",
            "training_coordinates_file":"Tampere/Training_coordinates.csv",
            "provided_test_accessed":False},
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    _save_torch_new(path,payload)
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return {"member":i,"seed":SEEDS[i-1],"epochs":EPOCHS[i-1],
            "sha256":_sha256_file(path),"seconds":time.perf_counter()-started,
            "reused":False}


def check_members() -> dict:
    manifest=prepare()
    members=[]
    for i in range(1,PANEL+1):
        p=member_path(i)
        if not p.is_file():
            return {"all_complete":False,"complete":len(members),"first_missing":i}
        obj=torch.load(p,map_location="cpu",weights_only=True)
        if (obj.get("manifest_sha256")!=manifest["manifest_sha256"] or
            obj.get("member_index")!=i or obj.get("member_seed")!=SEEDS[i-1]
            or obj.get("epoch")!=EPOCHS[i-1]):
            raise G5Error(f"member {i} ID/training contract mismatched")
        members.append({"index":i,"sha256":_sha256_file(p)})
    if len({v["sha256"] for v in members})!=PANEL:
        raise G5Error("some distinct-seed members share identical checkpoint hashes")
    return {"all_complete":True,"members":members,
            "manifest_sha256":manifest["manifest_sha256"]}


def freeze() -> dict:
    manifest=prepare()
    if LOCK.exists() or RESULT.exists():
        raise G5Error("cannot freeze after confirmation begins")
    members=check_members()
    if not members["all_complete"]:
        raise G5Error("all five members must be trained before freeze")
    frozen={
        "protocol":PROTOCOL,"status":"FULL_FIVE_MEMBER_PANEL_FROZEN",
        "manifest_sha256":manifest["manifest_sha256"],
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "members":members["members"],"confirmation_outcomes_observed":False,
        "provided_test_accessed":False
    }
    if FREEZE.exists():
        existing=_read(FREEZE)
        if existing!=dict(frozen,result_sha256=_canonical_json_sha256(frozen)):
            raise G5Error("frozen five-member checkpoint hash mismatch")
        return existing
    return _write(FREEZE,frozen)


def _load_models(device_name:str,frozen:dict):
    models=[];origins=[]
    for i in range(1,PANEL+1):
        path=member_path(i)
        if _sha256_file(path)!=frozen["members"][i-1]["sha256"]:
            raise G5Error("checkpoint byte change post-freeze")
        model,ckpt=load_anchor_checkpoint(path,torch.device(device_name))
        if ckpt["manifest_sha256"]!=frozen["manifest_sha256"]:
            raise G5Error("wrong checkpoint manifest")
        models.append(model.eval())
        origins.append(np.asarray(ckpt["coord_origin"],dtype=np.float64))
    ref=models[0]
    for m,origin in zip(models,origins):
        if (not torch.equal(ref.anchor_coords,m.anchor_coords)
            or not torch.equal(ref.anchor_buildings,m.anchor_buildings)
            or not torch.equal(ref.anchor_floors,m.anchor_floors)
            or not torch.equal(ref.prototype_rss,m.prototype_rss)
            or not np.array_equal(origin,origins[0])):
            raise G5Error("members have inconsistent TRAIN/FIT anchors or coordinate origin")
    return models,origins[0]


def weighted_component_geometric_median(
    probability:torch.Tensor,centres:torch.Tensor, *,
    iterations:int=MEDIAN_ITERATIONS,
    tolerance:float=MEDIAN_TOLERANCE_M
) -> torch.Tensor:
    """Discrete mixture-centre geometric median: separate diagnostic only.

    Exact continuous Student-t risk involves scale; this omits it on purpose.
    """
    if probability.ndim!=2 or centres.shape!=(*probability.shape,2):
        raise G5Error("invalid component median tensor dimensions")
    if not torch.isfinite(probability).all() or (probability<0).any():
        raise G5Error("negative/non-finite mixture weights")
    if not torch.isfinite(centres).all() or probability.sum(dim=1).min()<=0:
        raise G5Error("non-finite centres or empty probability")
    p=probability/probability.sum(dim=1,keepdim=True)
    action=(p[...,None]*centres).sum(dim=1)
    for _ in range(int(iterations)):
        dist=torch.linalg.vector_norm(centres-action[:,None,:],dim=-1)
        # Damp exact collisions; preserve baseline fallback for degenerate distributions.
        w=p/dist.clamp_min(1e-7)
        proposal=(w[...,None]*centres).sum(dim=1)/w.sum(dim=1,keepdim=True)
        if bool((torch.linalg.vector_norm(proposal-action,dim=1)<=tolerance).all()):
            action=proposal;break
        action=proposal
    return action


def three_actions(
    probability:torch.Tensor,location:torch.Tensor,scale:torch.Tensor,
    action_randomness:dict,
) -> dict[str,torch.Tensor]:
    if probability.ndim!=2 or location.shape!=(*probability.shape,2):
        raise G5Error("mixture dimensions invalid")
    if scale.shape!=location.shape or not torch.isfinite(scale).all() or (scale<=0).any():
        raise G5Error("student-t mixture scales must be positive finite")
    p=probability/probability.sum(dim=1,keepdim=True).clamp_min(1e-30)
    mean=(p[...,None]*location).sum(dim=1)
    centres=weighted_component_geometric_median(p,location)
    first=sample_continuous_mixture(
        p,location,scale,action_randomness["uniform_1"],action_randomness["noise_1"]
    )
    second=sample_continuous_mixture(
        p,location,scale,action_randomness["uniform_2"],action_randomness["noise_2"]
    )
    continuous=spatial_median_from_draws(
        first,second,maximum_iterations=MEDIAN_ITERATIONS,
        tolerance=MEDIAN_TOLERANCE_M,
    )
    return {"mean":mean,"component_median":centres,"continuous_median":continuous}


def _randomness_slice(payload:dict,start:int,stop:int,device:torch.device)->dict:
    return {k:(v[start:stop].to(device) if isinstance(v,torch.Tensor)
                and v.ndim>0 else v) for k,v in payload.items()}


def _evaluate_confirmation(device_name:str,frozen:dict) -> dict:
    # This function is never called before irreversible lock is created.
    if not LOCK.exists() or RESULT.exists():
        raise G5Error("confirmation lock absent or result already exists")
    manifest=prepare()
    data,groups,rows=_training() # only Tampere Training_rss + Training_coordinates
    models,origin=_load_models(device_name,frozen)
    device=torch.device(device_name)
    idx=rows["confirmation"]
    n=len(idx)
    coordinate_ids,ordered_keys=r2b._coordinate_groups([data["keys"][int(x)] for x in idx])
    actual=torch.as_tensor(
        data["raw_xy"][idx]-origin[None,:],dtype=torch.float64
    ).to(device)
    random_action=common_student_t_randomness(
        n,draws=ACTION_DRAWS_PER_STREAM,
        degrees_freedom=r3p.STUDENT_DF,seed=ACTION_SAMPLING_SEED
    )
    random_risk=common_student_t_randomness(
        n,draws=RISK_DRAWS_PER_STREAM,
        degrees_freedom=r3p.STUDENT_DF,seed=RISK_SAMPLING_SEED
    )
    random_energy=common_student_t_randomness(
        n,draws=r3p.PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=r3p.STUDENT_DF,seed=ENERGY_SAMPLING_SEED
    )
    row_loss={v:torch.empty(n,dtype=torch.float64) for v in VARIANTS}
    predicted_risk={v:torch.empty(n,dtype=torch.float64) for v in VARIANTS}
    energy=torch.empty(n,dtype=torch.float64)
    decision_shift=torch.empty(n,dtype=torch.float64)
    ref=models[0]
    anchors=ref.n_anchors
    for start in range(0,n,BATCH):
        end=min(start+BATCH,n)
        ids=idx[start:end]
        rss=data["rss"][ids].float().to(device)
        prototype=r2b._prototype_distance(ref,rss)
        anchor_idx=torch.arange(anchors,device=device)[None].expand(len(ids),-1)
        logits=[];loc=[];scale=[]
        with torch.no_grad():
            for model in models:
                embed=model.encoder(rss)
                logits.append(model.retrieval_logits_from_embedding(embed))
                centre,spread=model._components(embed,anchor_idx)
                loc.append(centre);scale.append(spread)
            prior=torch.softmax(
                r3p.BETA_NEURAL*torch.stack(logits)-
                r3p.BETA_PROTOTYPE*prototype[None,:,:],dim=2
            )
            top,top_indices,radial,_=r3._member_top_l(
                prior,torch.stack(loc),torch.stack(scale),
                L,student_df=r3p.STUDENT_DF
            )
            prob,centre,spread=r3._dense_joint_law(
                top,top_indices,radial,torch.stack(loc),torch.stack(scale),
                tuple(range(PANEL)),top_k=K,alpha=ALPHA
            )
            p=prob/prob.sum(dim=1,keepdim=True).clamp_min(1e-30)
            actions=three_actions(
                p,centre,spread,_randomness_slice(random_action,start,end,device)
            )
            if not torch.isfinite(torch.stack(list(actions.values()))).all():
                raise G5Error("non-finite Bayes actions")
            draw_risk=_randomness_slice(random_risk,start,end,device)
            draw1=sample_continuous_mixture(
                p,centre,spread,draw_risk["uniform_1"],draw_risk["noise_1"]
            )
            draw2=sample_continuous_mixture(
                p,centre,spread,draw_risk["uniform_2"],draw_risk["noise_2"]
            )
            for name,action in actions.items():
                row_loss[name][start:end]=torch.linalg.vector_norm(
                    action.double()-actual[start:end],dim=-1
                ).detach().cpu()
                predicted_risk[name][start:end]=(
                    .5*(torch.linalg.vector_norm(
                        draw1.double()-action[:,None,:].double(),dim=-1
                    ).mean(dim=1)+torch.linalg.vector_norm(
                        draw2.double()-action[:,None,:].double(),dim=-1
                    ).mean(dim=1))
                ).detach().cpu()
            decision_shift[start:end]=torch.linalg.vector_norm(
                actions["continuous_median"]-actions["mean"],dim=-1
            ).double().detach().cpu()
            energy[start:end]=r3._score_joint_law(
                p,centre,spread,actual[start:end],
                _randomness_slice(random_energy,start,end,device)
            )[0].detach().cpu()
        print(json.dumps({"confirmation_scored_rows":end,"total_rows":n}),
              flush=True)
    metrics={};vectors={}
    for name in VARIANTS:
        summary,vec=r3._metric_record(row_loss[name],coordinate_ids,len(ordered_keys))
        metrics[name]={"coordinate_macro_mae_m":float(summary["coordinate_macro_m"]),
                       "row_mae_m":float(summary["row_mean_m"])}
        vectors[name]=vec.tolist()
    es_record,es_groups=r3._metric_record(energy,coordinate_ids,len(ordered_keys))
    risk_record={}
    for name in VARIANTS:
        rr,_=r3._metric_record(predicted_risk[name],coordinate_ids,len(ordered_keys))
        risk_record[name]=float(rr["coordinate_macro_m"])
    shift,_=r3._metric_record(decision_shift,coordinate_ids,len(ordered_keys))
    differences={
        "component_median_minus_mean":np.asarray(vectors["component_median"])
                                      -np.asarray(vectors["mean"]),
        "continuous_median_minus_mean":np.asarray(vectors["continuous_median"])
                                      -np.asarray(vectors["mean"])
    }
    inference=r3.studentized_max_t_family(
        differences,repetitions=BOOTSTRAP_REPS,confidence=.95,
        seed=BOOTSTRAP_SEED,
        orientation="frozen_median_action_minus_original_mean_on_train_only_Tampere_confirmation"
    )
    delta=float(differences["continuous_median_minus_mean"].mean())
    upper=float(inference["comparisons"]["continuous_median_minus_mean"]["simultaneous_upper_bound_m"])
    pass_gate=(delta<=-MIN_GAIN_M and upper<0)
    return {
        "protocol":PROTOCOL,"status":"TRAINING_FILE_INTERNAL_CONFIRMATION_COMPLETE",
        "decision":"PROMISING_ACTION_ALIGNMENT" if pass_gate else "NO_INTERNAL_PROMOTION",
        "role":"confirmation",
        "rows":n,"coordinates":len(ordered_keys),
        "coordinate_keys":[list(x) for x in ordered_keys],
        "metrics":metrics,"coordinate_loss_vectors":vectors,
        "common_energy_score":{
            "coordinate_macro_energy_m":float(es_record["coordinate_macro_m"]),
            "row_energy_m":float(es_record["row_mean_m"]),
            "coordinate_energy_vector":es_groups.tolist(),
            "all_actions_have_identical_predictive_law":True,
            "energy_difference_between_actions_exactly_zero":True
        },
        "expected_posterior_euclidean_risk_m":risk_record,
        "coordinate_macro_median_mean_displacement_m":float(shift["coordinate_macro_m"]),
        "paired_familywise_inference":inference,
        "gate":{"passed":bool(pass_gate),"min_mae_gain_m":MIN_GAIN_M,
                "mean_diff_m":delta,"maxT_upper_m":upper},
        "manifest_sha256":manifest["manifest_sha256"],
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "freeze_file_sha256":_sha256_file(FREEZE),
        "lock_file_sha256":_sha256_file(LOCK),
        "member_sha256":[x["sha256"] for x in frozen["members"]],
        "physical_train_confirmation_overlap":len(groups["fit"]&groups["confirmation"]),
        "provided_test_files_accessed":False,
        "warning":"No architecture change; Bayes action is known decision theory, and the sampler/Weiszfeld routine already existed in V9. 5-member pilot only. Spatially correlated coordinates; 5000x coordinate bootstrap may underestimate dependence."
    }


def confirm(device_name:str="cuda") -> dict:
    if LOCK.exists() or RESULT.exists():
        raise G5Error("G5 confirmation already attempted; no rerun")
    frozen=freeze()
    locked=_write(LOCK,{
        "protocol":PROTOCOL,
        "status":"INTERNAL_CONFIRMATION_STARTS_AFTER_MODEL_FREEZE",
        "freeze_file_sha256":_sha256_file(FREEZE),
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "member_sha256":[x["sha256"] for x in frozen["members"]],
        "test_file_accessed":False,
        "warning":"Keep this file if evaluation fails. Never remove to retry."
    })
    result=_evaluate_confirmation(device_name,frozen)
    result["lock_result_sha256"]=locked["result_sha256"]
    result=_write(RESULT,result)
    REPORT.write_text(make_report(result),encoding="utf-8",newline="\n")
    return {"status":result["status"],"decision":result["decision"],
            "gate":result["gate"],"result_sha256":result["result_sha256"]}


def make_report(result:dict) -> str:
    out=[
        "# V17 G5-A | Tampere TRAIN-only posterior action alignment",
        "",
        "No provided external test used. One frozen five-member V17 predictive distribution; only point action changes.",
        "",
        "| Point action | Coordinate-macro MAE (m) |",
        "|---|---:|"
    ]
    for name in VARIANTS:
        out.append(f"| {name} | {result['metrics'][name]['coordinate_macro_mae_m']:.6f} |")
    inf=result["paired_familywise_inference"]["comparisons"]
    gate=result["gate"]
    out.extend([
        "",
        f"Primary continuous-median minus exact-mixture-mean MAE: {gate['mean_diff_m']:+.6f} m",
        f"95% simultaneous maxT upper: {gate['maxT_upper_m']:+.6f} m",
        f"Fixed minimum improvement: {gate['min_mae_gain_m']:.2f} m",
        f"Decision: **{result['decision']}**",
        "",
        f"Secondary component-centre-median minus mean: {inf['component_median_minus_mean']['mean_difference_m']:+.6f} m",
        f"Shared energy score: {result['common_energy_score']['coordinate_macro_energy_m']:.6f} m",
        "Energy Score is identical across point actions by construction, not an independent improvement.",
        f"Independent-draw expected predictive Euclidean risk: {result['expected_posterior_euclidean_risk_m']}",
        f"Mean geometric-median point displacement: {result['coordinate_macro_median_mean_displacement_m']:.6f} m",
        "",
        "Both median algorithms use no query labels. Student-t continuous-median is Monte Carlo approximate;",
        "the V9 code already implements sampling and Weiszfeld median—do not call this method novel.",
        "All experimental comparisons use identical five V17 member weights and the E5/L192/K192/A1 posterior.",
        "Do not compare absolute error with G3/G4's different panels/role splits.",
        "Internal confirmation coordinates can be spatially dependent; external tests remain untouched.",
        f"Result self hash: {result['result_sha256']}",""
    ])
    return "\n".join(out)


def check() -> dict:
    manifest=prepare()
    frozen=freeze() if not LOCK.exists() else _read(FREEZE)
    if frozen["manifest_sha256"]!=manifest["manifest_sha256"]:
        raise G5Error("freeze differs from manifest")
    if frozen["manifest_file_sha256"]!=_sha256_file(MANIFEST):
        raise G5Error("manifest bytes changed")
    current=check_members()
    if not current["all_complete"] or current["members"]!=frozen["members"]:
        raise G5Error("frozen member bytes changed")
    if not LOCK.exists():
        return {"status":"FROZEN_PRE_CONFIRMATION"}
    lock=_read(LOCK)
    if lock["freeze_file_sha256"]!=_sha256_file(FREEZE):
        raise G5Error("confirmation lock no longer matches freeze")
    if not RESULT.exists():
        return {"status":"CONFIRMATION_LOCK_WITHOUT_RESULT_STOP"}
    result=_read(RESULT)
    if (result["freeze_file_sha256"]!=_sha256_file(FREEZE)
        or result["lock_file_sha256"]!=_sha256_file(LOCK)
        or result["lock_result_sha256"]!=lock["result_sha256"]):
        raise G5Error("confirmation provenance invalid")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8")!=make_report(result):
        raise G5Error("G5 rendered report does not match frozen result")
    return {"status":"VERIFIED","decision":result["decision"],
            "result_sha256":result["result_sha256"]}


def main(argv:Sequence[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    cmd=parser.add_subparsers(dest="command",required=True)
    for name in ("preflight","prepare","train-member","check-members","freeze","confirm","check"):
        p=cmd.add_parser(name)
        if name in ("train-member","confirm"):
            p.add_argument("--device",default="cuda")
        if name=="train-member":
            p.add_argument("--member-index",type=int,required=True)
    args=parser.parse_args(argv)
    if args.command=="preflight":v=preflight()
    elif args.command=="prepare":v=prepare()
    elif args.command=="train-member":v=train_member(args.member_index,args.device)
    elif args.command=="check-members":v=check_members()
    elif args.command=="freeze":v=freeze()
    elif args.command=="confirm":v=confirm(args.device)
    else:v=check()
    print(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=str),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
