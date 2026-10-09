"""G6: TUJI1 train-variable-only calibration of V17 mixture location weights and scales.

Three physical-site-disjoint TRAIN-variable roles: FIT / CAL / CONFIRM.
No provided-test label/variable is deserialized. Model and R3 posterior
remain frozen; compare global joint-weight temperature and Student-t scale.
This is a preregistered *pilot*, not external validation or a Q1 invention.
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
from scipy.io import loadmat

from fhm_dapp_tim.fhm_dapp import v18_4_datasets as ds
from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as r0
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness, sample_continuous_mixture,
)
from fhm_dapp_tim.fhm_dapp.anchor import load_anchor_checkpoint, location_keys
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _save_torch_new, _sha256_file, _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EVIDENCE = ROOT / "evidence/v17_g6_distribution_calibration"
MODELS = ROOT / "models/v17_g6_distribution_calibration"
DESIGN = EVIDENCE / "DESIGN_V17_G6.md"
MANIFEST = EVIDENCE / "manifest_v17_g6.json"
CALIBRATION = EVIDENCE / "calibration_v17_g6.json"
LOCK = EVIDENCE / "confirmation_v17_g6.lock"
RESULT = EVIDENCE / "confirmation_v17_g6.json"
REPORT = EVIDENCE / "RUN_REPORT_V17_G6.md"
MAT_DIR = ds.DATA_ROOT / "TUJI1"

PROTOCOL = "v17_g6_tuji1_train_var_only_dist_calibration_v1_20261009"
SPLIT_SEED = 19601
SCORE_SEED = 19602
BOOT_SEED = 19603
PANEL = 5
SEEDS = tuple(g1.SEEDS[:PANEL])
EPOCHS = tuple(g1.EPOCHS[:PANEL])
L, K, ALPHA = 192, 192, 1.0
BATCH = 16
BOOT_REPS = 5000
CAL_MIN_MAE_GAIN_M = 0.10
CAL_MIN_ES_GAIN_M = 0.02
CONF_MIN_MAE_GAIN_M = 0.15
T_VALUES = (1.0, 0.8, 0.6)
S_VALUES = (1.0, 0.75, 0.5)
CANDIDATES = tuple((t,s) for t in T_VALUES for s in S_VALUES)
ROLE_WEIGHTS = (0.70, 0.15, 0.15)  # deterministic site-stratified
COVERAGE_LEVELS = (0.50,0.80,0.90)
BASE = "T1_S1"
EXPECTED_DEVICES = tuple(ds.TUJI_DEVICE_FILES)


class G6Error(RuntimeError):
    pass


def candidate_name(t:float,s:float) -> str:
    if (t,s) not in CANDIDATES:
        raise G6Error("candidate outside frozen temperature/scale grid")
    return f"T{t:g}_S{s:g}"


def _write(path:Path,record:dict,field:str="result_sha256") -> dict:
    if path.exists():raise FileExistsError(path)
    obj=dict(record);obj[field]=_canonical_json_sha256(obj)
    path.parent.mkdir(parents=True,exist_ok=True)
    _write_json_new(path,obj)
    return obj


def _read(path:Path,field:str="result_sha256") -> dict:
    obj=json.loads(path.read_text(encoding="utf-8"))
    if obj.get(field)!=_canonical_json_sha256(obj,omit=field):
        raise G6Error(f"immutable G6 evidence hash mismatch: {path}")
    return obj


def _source_hashes() -> dict:
    sources={
        "design":DESIGN,"g6_code":Path(__file__),"v17_r0":Path(r0.__file__),
        "v17_r3":Path(r3.__file__),"v17_r3_protocol":Path(r3p.__file__),
        "canonical_adapter":Path(ds.__file__),
        "v17_anchor":Path(load_anchor_checkpoint.__code__.co_filename),
        "v17_r2b":Path(r2b.__file__),
        "frozen_training_recipe":ROOT / "fhm_dapp/train_anchor_v2.py",
        "historical_seed_source":Path(g1.__file__),
        "g6_tests":REPO / "tests/test_uji_v17_g6_calibration.py",
        "v9_sampling":Path(sample_continuous_mixture.__code__.co_filename),
    }
    for dev,name in ds.TUJI_DEVICE_FILES.items():
        sources[f"train_container_{dev}"]=MAT_DIR/name
    return {k:_sha256_file(v) for k,v in sources.items()}


def load_tuji_train_variable_only() -> ds.CanonicalFingerprintData:
    """Decode exclusively results_train variable from each mixed MAT container.

    Each physical MAT file also contains results_test. Read access to the
    *container* is unavoidable; the test variable is not requested/decoded,
    test labels are NEVER consulted. Do not call ds.load_tuji1, which
    deserializes both variable trees via unfiltered scipy.loadmat.
    """
    rows=[];seen_ids=set()
    for device_name in EXPECTED_DEVICES:
        source=MAT_DIR/ds.TUJI_DEVICE_FILES[device_name]
        payload=loadmat(source,squeeze_me=True,struct_as_record=False,
                        variable_names=("results_train",))
        if "results_test" in payload:
            raise G6Error("test variable unexpectedly deserialized")
        if "results_train" not in payload:
            raise G6Error(f"no train variable in TUJI MAT {device_name}")
        bundle=payload["results_train"]
        scans=np.asarray(bundle.Wifi,dtype=object).reshape(-1).tolist()
        for scan in scans:
            rows.append((device_name,scan))
            for ap,_ in ds._tuji_measurements(scan):
                seen_ids.add(ds._tuji_ap_id(ap))
    if not rows or not seen_ids:
        raise G6Error("TUJI training variable empty")
    ap_ids=tuple(sorted(seen_ids,key=ds._tuji_ap_sort_key))
    reference=ds._reference(
        dataset="TUJI1",ap_ids=ap_ids,
        building_values=["0"]*len(rows),floor_values=["0"]*len(rows),
        device_values=list(EXPECTED_DEVICES),
        user_values=["unknown"]*len(rows),
        extra={"training_device_files":list(EXPECTED_DEVICES),
               "loaded_mat_variables":["results_train"],
               "container_also_stores_test_variable":True},
    )
    pos={k:i for i,k in enumerate(ap_ids)}
    rss=np.full((len(rows),len(ap_ids)),np.nan,dtype=np.float32)
    xy=np.empty((len(rows),2),dtype=np.float32)
    phone=np.empty(len(rows),dtype=np.int64)
    for i,(name,scan) in enumerate(rows):
        xy[i]=np.asarray(scan.coords,dtype=np.float32).reshape(2)
        phone[i]=reference.device_mapping[name]
        for ap,value in ds._tuji_measurements(scan):
            v=float(value);col=pos[ds._tuji_ap_id(ap)]
            if not math.isfinite(v):
                continue
            if np.isfinite(rss[i,col]):
                rss[i,col]=max(float(rss[i,col]),v)
            else:
                rss[i,col]=v
    data=ds.CanonicalFingerprintData(
        dataset="TUJI1",split="train",rss_dbm=rss,coords_m=xy,
        building=np.zeros(len(rows),dtype=np.int64),
        floor=np.zeros(len(rows),dtype=np.int64),
        device=phone,user=np.full(len(rows),-1,dtype=np.int64),
        timestamp_s=np.full(len(rows),np.nan,dtype=np.float64),
        ap_ids=ap_ids,metadata={
            "mat_variable_only":"results_train",
            "files_physically_contain_results_test":True,
            "mat_file_count":len(EXPECTED_DEVICES),
            "test_data_decoded":False,
        }
    )
    data.validate()
    return data


def _as_r0_input(canonical:ds.CanonicalFingerprintData) -> dict:
    if canonical.dataset!="TUJI1" or canonical.split!="train":
        raise G6Error("only TUJI1 training variable allowed")
    raw=np.asarray(canonical.coords_m,dtype=np.float64)
    rss=torch.from_numpy(canonical.model_rss()).float()
    building=np.asarray(canonical.building,dtype=np.int64)
    floor=np.asarray(canonical.floor,dtype=np.int64)
    phone=np.asarray(canonical.device,dtype=np.int64)
    if len(raw)<300 or np.any(phone<0) or not torch.isfinite(rss).all():
        raise G6Error("invalid TUJI training row count, phone IDs, or model RSS")
    return {"rss":rss,"raw_xy":raw,"building":building,"floor":floor,
            "phone":phone,"keys":location_keys(raw,building,floor),
            "stats":{"n_aps":rss.shape[1],"n_buildings":1,"n_floors":1,
                     "adapted_from":"TUJI1 results_train only"}}


def split_physical_keys(keys:Sequence[tuple],seed:int=SPLIT_SEED
                       ) -> dict[str,set[tuple]]:
    strata=defaultdict(set)
    for k in keys:
        if len(k)!=4:raise G6Error("physical-key shape mismatch")
        strata[(int(k[0]),int(k[1]))].add(tuple(k))
    roles={"fit":set(),"calibration":set(),"confirmation":set()}
    for sf,locations in sorted(strata.items()):
        ordered=sorted(locations,key=lambda k:(
            hashlib.sha256(f"{seed}|{k!r}".encode("utf-8")).digest(),k
        ))
        count=len(ordered)
        if count<15:
            raise G6Error(f"too few unique physical positions on {sf}: {count}")
        a=max(1,min(count-2,int(math.floor(ROLE_WEIGHTS[0]*count))))
        b=max(1,min(count-a-1,int(math.floor(ROLE_WEIGHTS[1]*count))))
        roles["fit"].update(ordered[:a])
        roles["calibration"].update(ordered[a:a+b])
        roles["confirmation"].update(ordered[a+b:])
    if (set.union(*roles.values())!=set(keys)
        or any(roles[x]&roles[y] for x,y in
               (("fit","calibration"),("fit","confirmation"),("calibration","confirmation"))):
        raise G6Error("train/cal/confirm site overlap or omission")
    return roles


def _roles(data:dict):
    roles=split_physical_keys(data["keys"])
    rows={name:np.asarray([i for i,k in enumerate(data["keys"]) if k in sites],
                          dtype=np.int64) for name,sites in roles.items()}
    if len(roles["fit"])<max(L,K):
        raise G6Error("TUJI FIT has fewer than 192 unique anchors")
    if sum(len(x) for x in rows.values())!=len(data["keys"]):
        raise G6Error("TUJI train rows not exhaustively partitioned")
    origin=data["raw_xy"][rows["fit"]].mean(axis=0).astype(np.float32)
    centred=(data["raw_xy"]-origin[None,:]).astype(np.float32)
    alias={}
    reverse={}
    # Verify the entire original-to-centred float32 key dictionary, not
    # merely FIT: otherwise a confirmation site could alias a FIT anchor.
    transformed=location_keys(centred,data["building"],data["floor"])
    for k,c in zip(data["keys"],transformed):
        if (k in alias and alias[k]!=c) or (c in reverse and reverse[c]!=k):
            raise G6Error("R0 centring collapses or divides a physical site")
        alias[k]=c;reverse[c]=k
    if len(alias)!=len(set(data["keys"])):
        raise G6Error("R0 centring breaks cross-role physical coordinate isolation")
    return roles,rows


def _training():
    data=_as_r0_input(load_tuji_train_variable_only())
    roles,rows=_roles(data)
    return data,roles,rows


def member_path(i:int) -> Path:
    if not 1<=int(i)<=PANEL:raise G6Error("G6 member must be 1..5")
    return MODELS/f"member_{i:02d}_seed{SEEDS[i-1]}.pt"


def preflight() -> dict:
    data,roles,rows=_training()
    return {
        "protocol":PROTOCOL,"status":"TUJI_RESULTS_TRAIN_VARIABLE_ONLY_READY",
        "source_sha256":_source_hashes(),
        "train_rows":len(data["keys"]),"unique_train_sites":len(set(data["keys"])),
        "ap_count":int(data["rss"].shape[1]),
        "role_site_counts":{k:len(v) for k,v in roles.items()},
        "role_row_counts":{k:len(v) for k,v in rows.items()},
        "role_device_rows":{k:{str(i):int(np.count_nonzero(
            data["phone"][idx]==i)) for i in range(len(EXPECTED_DEVICES))}
            for k,idx in rows.items()},
        "physical_site_overlap":0,
        "mat_containers_physically_include_provided_test":True,
        "results_test_deserialized":False,
        "seeds":list(SEEDS),"epochs":list(EPOCHS),
        "candidates":[candidate_name(*c) for c in CANDIDATES],
    }


def prepare() -> dict:
    info=preflight();_,roles,_=_training()
    record={
        "protocol":PROTOCOL,"source_sha256":info["source_sha256"],
        "role_coordinate_keys":{k:[list(x) for x in sorted(v)] for k,v in roles.items()},
        "role_site_counts":info["role_site_counts"],
        "role_row_counts":info["role_row_counts"],
        "role_device_rows":info["role_device_rows"],
        "seeds":list(SEEDS),"epochs":list(EPOCHS),
        "model_count":PANEL,"L":L,"K":K,"alpha":ALPHA,
        "grid":[{"name":candidate_name(t,s),"T":t,"scale":s}
                for t,s in CANDIDATES],
        "selection_rule":{
            "min_cal_mae_gain_m":CAL_MIN_MAE_GAIN_M,
            "min_cal_es_gain_m":CAL_MIN_ES_GAIN_M,
            "primary":"lowest calibration coordinate-macro MAE among eligible",
            "fallback":BASE,
            "placebo_not_eligible":True,
        },
        "confirmation_gate":{
            "min_mae_gain_m":CONF_MIN_MAE_GAIN_M,
            "studentized_maxT_bootstrap_repetitions":BOOT_REPS,
            "joint_family":["mae","es"],"confidence":.95,
            "mae_upper_lt_0":True,"es_upper_le_0":True,
        },
        "score_seed":SCORE_SEED,"bootstrap_seed":BOOT_SEED,
        "coverage_levels":list(COVERAGE_LEVELS),
        "frozen_v17_training":copy.deepcopy(FROZEN_TRAINING),
        "mat_test_variable_never_loaded":True,
    }
    record["manifest_sha256"]=_canonical_json_sha256(record)
    if MANIFEST.exists():
        current=_read(MANIFEST,"manifest_sha256")
        if record!=current:raise G6Error("source, data, design or role split changed after manifest")
        return current
    _write_json_new(MANIFEST,record)
    return record


def train_member(i:int,device_name:str="cuda") -> dict:
    if CALIBRATION.exists() or LOCK.exists() or RESULT.exists():
        raise G6Error("training forbidden after calibration")
    manifest=prepare();destination=member_path(i)
    if destination.exists():
        ck=torch.load(destination,map_location="cpu",weights_only=True)
        if ck.get("manifest_sha256")!=manifest["manifest_sha256"] or ck.get("member_index")!=i:
            raise G6Error("incompatible existing member checkpoint")
        return {"member":i,"sha256":_sha256_file(destination),"reused":True}
    data,roles,rows=_training()
    started=time.perf_counter()
    model,stats,origin,mapping,history=r0._fit_refit(
        data,rows["fit"],seed=SEEDS[i-1],epochs=EPOCHS[i-1],
        device=torch.device(device_name),log_every=10
    )
    if int(model.n_anchors)!=len(roles["fit"]):
        raise G6Error("R0 anchor dictionary count mismatch")
    config=copy.deepcopy(FROZEN_TRAINING)
    config.update({"model_schema":"v17_g6_tuji_original_atomic",
                   "training_protocol_version":PROTOCOL,
                   "train_all":True,"n_devices":len(mapping),
                   "epochs":EPOCHS[i-1],
                   "validation_fraction":0.,"calibration_fraction":0.})
    payload={
        "protocol":PROTOCOL,"manifest_sha256":manifest["manifest_sha256"],
        "member_index":i,"member_seed":SEEDS[i-1],"epoch":EPOCHS[i-1],
        "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "config":config,"stats":stats,
        "coord_origin":np.asarray(origin,dtype=np.float32),
        "device_id_mapping":{str(k):v for k,v in mapping.items()},
        "refit_history":history,
        "data_provenance":{"variable":"results_train","fit_only":True,
            "test_variable_loaded":False}
    }
    destination.parent.mkdir(parents=True,exist_ok=True)
    _save_torch_new(destination,payload)
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return {"member":i,"sha256":_sha256_file(destination),
            "seconds":time.perf_counter()-started,"reused":False}


def check_members() -> dict:
    manifest=prepare();out=[]
    for i in range(1,PANEL+1):
        p=member_path(i)
        if not p.is_file():return {"all_complete":False,"first_missing":i,"members":out}
        ck=torch.load(p,map_location="cpu",weights_only=True)
        if (ck.get("manifest_sha256")!=manifest["manifest_sha256"]
            or ck.get("member_seed")!=SEEDS[i-1] or ck.get("epoch")!=EPOCHS[i-1]
            or ck.get("member_index")!=i):
            raise G6Error(f"member {i} identity mismatch")
        out.append({"index":i,"sha256":_sha256_file(p)})
    if len(set(v["sha256"] for v in out))!=PANEL:
        raise G6Error("identical member checkpoints")
    return {"all_complete":True,"members":out}


def _load_models(device_name:str,manifest:dict):
    models=[];origins=[];hashes=[]
    for i in range(1,PANEL+1):
        p=member_path(i)
        model,ck=load_anchor_checkpoint(p,torch.device(device_name))
        if ck.get("manifest_sha256")!=manifest["manifest_sha256"]:
            raise G6Error("model checkpoint not from frozen G6 manifest")
        models.append(model.eval())
        origins.append(np.asarray(ck["coord_origin"],dtype=np.float64))
        hashes.append(_sha256_file(p))
    ref=models[0]
    for model,origin in zip(models,origins):
        if (not torch.equal(model.anchor_coords,ref.anchor_coords)
            or not torch.equal(model.anchor_buildings,ref.anchor_buildings)
            or not torch.equal(model.anchor_floors,ref.anchor_floors)
            or not torch.equal(model.prototype_rss,ref.prototype_rss)
            or not np.array_equal(origin,origins[0])):
            raise G6Error("five members do not share identical training memory and origin")
    if ref.n_anchors!=manifest["role_site_counts"]["fit"]:
        raise G6Error("number of fitted anchors differs from frozen role count")
    return models,origins[0],hashes


def calibrate_joint_law(p:torch.Tensor,loc:torch.Tensor,scale:torch.Tensor,
                        temperature:float,scale_factor:float):
    """Pure, label-free posterior calibration of two distinct uncertainty levels."""
    if (temperature,scale_factor) not in CANDIDATES:
        raise G6Error("unregistered calibrator pair")
    if p.ndim!=2 or loc.shape!=(*p.shape,2) or scale.shape!=loc.shape:
        raise G6Error("G6 invalid mixture law shapes")
    if not torch.isfinite(p).all() or not torch.isfinite(scale).all() or (p<0).any():
        raise G6Error("invalid mixture probabilities or scales")
    if temperature==1.:q=p  # exact bitwise V17 baseline
    else:q=torch.softmax(p.clamp_min(1e-30).log()/temperature,dim=1)
    sigma=scale if scale_factor==1. else scale*scale_factor
    return q,loc,sigma


def select_policy(metrics:dict) -> dict:
    base=metrics[BASE]
    eligible=[]
    for pair in CANDIDATES:
        name=candidate_name(*pair)
        if name==BASE:continue
        rec=metrics[name]
        if (rec["mae_2d_m"]<=base["mae_2d_m"]-CAL_MIN_MAE_GAIN_M
            and rec["energy_score_2d_m"]<=base["energy_score_2d_m"]-CAL_MIN_ES_GAIN_M):
            eligible.append(name)
    selected=min(eligible,key=lambda name:(metrics[name]["mae_2d_m"],
                                           metrics[name]["energy_score_2d_m"],name)) if eligible else BASE
    t,s=next(pair for pair in CANDIDATES if candidate_name(*pair)==selected)
    return {"policy":selected,"T":t,"scale":s,"eligible":eligible,
            "fell_back_to_baseline":not bool(eligible),
            "frozen_calibration_only":True}


def _random_slice(random:dict,start:int,end:int,device:torch.device) -> dict:
    return {k:(v[start:end].to(device) if isinstance(v,torch.Tensor) and v.ndim>0
               else v) for k,v in random.items()}


@torch.no_grad()
def _score_role(role:str,models,data:dict,row_ids:np.ndarray,origin:np.ndarray,
                selected:dict|None=None) -> dict:
    device=next(models[0].parameters()).device
    n=len(row_ids)
    group_ids,group_keys=r2b._coordinate_groups([data["keys"][int(i)] for i in row_ids])
    xy=torch.as_tensor(data["raw_xy"][row_ids]-origin[None,:],dtype=torch.float64).to(device)
    random=common_student_t_randomness(
        n,draws=r3p.PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=r3p.STUDENT_DF,seed=SCORE_SEED
    )
    if role=="calibration":
        pairs=CANDIDATES
    else:
        if selected is None:raise G6Error("confirmation requires frozen calibration")
        # The two one-factor ablations are descriptive, not eligible to replace
        # the frozen policy; all parameters fixed from calibration.
        pairs=tuple(dict.fromkeys((
            (1.,1.),(selected["T"],selected["scale"]),
            (selected["T"],1.),(1.,selected["scale"]),
        )))
    names=[candidate_name(*v) for v in pairs]
    losses={k:torch.empty(n,dtype=torch.float64) for k in names}
    energy={k:torch.empty(n,dtype=torch.float64) for k in names}
    calibration_diag={k:{"predicted_radius":torch.empty(n,dtype=torch.float64),
                         "observed_error":torch.empty(n,dtype=torch.float64),
                         "coverage":{str(level):torch.empty(n,dtype=torch.float64)
                                     for level in COVERAGE_LEVELS}}
                      for k in names if k in (BASE,(selected or {}).get("policy"))}
    ref=models[0];anchors=ref.n_anchors
    with r3.r2f_a1._deterministic_algorithms():
        for start in range(0,n,BATCH):
            end=min(n,start+BATCH);ids=row_ids[start:end]
            rss=data["rss"][ids].float().to(device)
            prototype=r2b._prototype_distance(ref,rss)
            anchor_ix=torch.arange(anchors,device=device)[None,:].expand(len(ids),-1)
            logits=[];loc=[];sigma=[]
            for model in models:
                embedding=model.encoder(rss)
                logits.append(model.retrieval_logits_from_embedding(embedding))
                m,s=model._components(embedding,anchor_ix)
                loc.append(m);sigma.append(s)
            prior=torch.softmax(
                r3p.BETA_NEURAL*torch.stack(logits)
                -r3p.BETA_PROTOTYPE*prototype[None,:,:],dim=-1
            )
            top,idx,radial,_=r3._member_top_l(
                prior,torch.stack(loc),torch.stack(sigma),
                L,student_df=r3p.STUDENT_DF
            )
            p,locations,scales=r3._dense_joint_law(
                top,idx,radial,torch.stack(loc),torch.stack(sigma),
                tuple(range(PANEL)),top_k=K,alpha=ALPHA
            )
            rand=_random_slice(random,start,end,device)
            for (t,s),name in zip(pairs,names):
                q,centre,sig=calibrate_joint_law(p,locations,scales,t,s)
                e,mae=r3._score_joint_law(q,centre,sig,xy[start:end],rand)
                losses[name][start:end]=mae.detach().cpu().double()
                energy[name][start:end]=e.detach().cpu().double()
                if name in calibration_diag:
                    action=(q[:,:,None]*centre).sum(dim=1)
                    draw1=sample_continuous_mixture(q,centre,sig,
                        rand["uniform_1"],rand["noise_1"])
                    draw2=sample_continuous_mixture(q,centre,sig,
                        rand["uniform_2"],rand["noise_2"])
                    distances=torch.linalg.vector_norm(
                        torch.cat((draw1,draw2),dim=1)-action[:,None,:],dim=-1)
                    diag=calibration_diag[name]
                    diag["predicted_radius"][start:end]=distances.mean(dim=1).double().cpu()
                    observed=torch.linalg.vector_norm(
                        action.double()-xy[start:end],dim=-1)
                    diag["observed_error"][start:end]=observed.cpu()
                    for level in COVERAGE_LEVELS:
                        threshold=torch.quantile(distances,level,dim=1)
                        diag["coverage"][str(level)][start:end]=(
                            observed<=threshold.double()).double().cpu()
        metrics={};vectors={};by_device={}
        for name in names:
            m,mv=r3._metric_record(losses[name],group_ids,len(group_keys))
            e,ev=r3._metric_record(energy[name],group_ids,len(group_keys))
            metrics[name]={"mae_2d_m":float(m["coordinate_macro_m"]),
                           "energy_score_2d_m":float(e["coordinate_macro_m"]),
                           "row_mae_m":float(m["row_mean_m"]),
                           "row_es_m":float(e["row_mean_m"])}
            vectors[name]={"mae":mv.tolist(),"es":ev.tolist()}
            by_device[name]={}
            for device_id in sorted(set(int(v) for v in data["phone"][row_ids])):
                mask=torch.as_tensor(data["phone"][row_ids]==device_id,dtype=torch.bool)
                by_device[name][str(device_id)]={
                    "rows":int(mask.sum()),
                    "row_mae_m":float(losses[name][mask].mean()),
                    "row_es_m":float(energy[name][mask].mean()),
                }
        reliability={}
        for name,diag in calibration_diag.items():
            observed=r3._metric_record(diag["observed_error"],group_ids,len(group_keys))[0]
            predicted=r3._metric_record(diag["predicted_radius"],group_ids,len(group_keys))[0]
            reliability[name]={
                "macro_predicted_radius_m":float(predicted["coordinate_macro_m"]),
                "macro_observed_error_m":float(observed["coordinate_macro_m"]),
                "coverage":{level:float(r3._metric_record(
                    vec,group_ids,len(group_keys))[0]["coordinate_macro_m"])
                    for level,vec in diag["coverage"].items()},
            }
    return {"role":role,"rows":n,"coordinate_groups":len(group_keys),
            "coordinate_keys":[list(k) for k in group_keys],
            "candidate_metrics":metrics,"coordinate_vectors":vectors,
            "row_metrics_by_device":by_device,
            "reliability":reliability,"random_draws_per_stream":r3p.PREDICTIVE_DRAWS_PER_STREAM}


def calibrate(device_name:str="cuda") -> dict:
    if CALIBRATION.exists() or LOCK.exists() or RESULT.exists():
        raise G6Error("calibration already attempted")
    manifest=prepare()
    complete=check_members()
    if not complete["all_complete"]:raise G6Error("five members required")
    models,origin,sha=_load_models(device_name,manifest)
    data,_,rows=_training()
    result=_score_role("calibration",models,data,rows["calibration"],origin)
    policy=select_policy(result["candidate_metrics"])
    result.update({
        "protocol":PROTOCOL,"manifest_sha256":manifest["manifest_sha256"],
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "member_sha256":sha,"selected_policy":policy,
        "results_test_deserialized":False,"confirmation_labels_used":False,
    })
    return _write(CALIBRATION,result)


def confirm(device_name:str="cuda") -> dict:
    if LOCK.exists() or RESULT.exists():
        raise G6Error("confirmation already attempted")
    manifest=prepare()
    cal=_read(CALIBRATION)
    members=check_members()
    if not members["all_complete"]:
        raise G6Error("five original members not complete at confirmation")
    sha=[m["sha256"] for m in members["members"]]
    if sha!=cal["member_sha256"] or cal["manifest_sha256"]!=manifest["manifest_sha256"]:
        raise G6Error("manifest/members changed since calibration")
    policy=cal["selected_policy"]
    lock=_write(LOCK,{
        "protocol":PROTOCOL,"manifest_file_sha256":_sha256_file(MANIFEST),
        "calibration_file_sha256":_sha256_file(CALIBRATION),
        "member_sha256":sha,
        "frozen_policy":policy,"confirmation_outcome_seen":False,
    })
    models,origin,modelsha=_load_models(device_name,manifest)
    if modelsha!=sha:raise G6Error("models changed between lock and confirmation")
    data,_,rows=_training()
    result=_score_role("confirmation",models,data,rows["confirmation"],origin,selected=policy)
    name=policy["policy"];base=result["coordinate_vectors"][BASE]
    v=result["coordinate_vectors"][name]
    differences={
        "mae":np.asarray(v["mae"],dtype=np.float64)-np.asarray(base["mae"],dtype=np.float64),
        "es":np.asarray(v["es"],dtype=np.float64)-np.asarray(base["es"],dtype=np.float64)
    }
    if name==BASE:
        inference={"selected_baseline_no_change":True,
                   "comparisons":{"mae":{"simultaneous_upper_bound_m":0.},
                                  "es":{"simultaneous_upper_bound_m":0.}}}
    else:
        inference=r3.studentized_max_t_family(
            differences,repetitions=BOOT_REPS,confidence=.95,seed=BOOT_SEED,
            orientation="frozen_G6_candidate_minus_V17_baseline"
        )
    dmae=float(differences["mae"].mean())
    des=float(differences["es"].mean())
    um=float(inference["comparisons"]["mae"]["simultaneous_upper_bound_m"])
    ue=float(inference["comparisons"]["es"]["simultaneous_upper_bound_m"])
    passed=(name!=BASE and dmae<=-CONF_MIN_MAE_GAIN_M and um<0 and ue<=0)
    site_effect=differences["mae"]
    result.update({
        "protocol":PROTOCOL,"selected_policy":policy,
        "status":"TRAIN_VARIABLE_INTERNAL_CONFIRMATION_COMPLETE",
        "decision":"PROMISING_JOINT_CALIBRATION" if passed else "NO_INTERNAL_PROMOTION",
        "gate":{"passed":bool(passed),"min_mae_gain_m":CONF_MIN_MAE_GAIN_M,
                "mae_delta_m":dmae,"es_delta_m":des,"mae_upper_m":um,"es_upper_m":ue},
        "paired_inference":inference,
        "descriptive":{"mae_win_sites":int(np.count_nonzero(site_effect<-1e-9)),
                       "mae_loss_sites":int(np.count_nonzero(site_effect>1e-9)),
                       "mae_ties":int(np.count_nonzero(abs(site_effect)<=1e-9)),
                       "median_mae_delta_m":float(np.median(site_effect)),
                       "coordinate_error_p90_m":{
                           v:float(np.quantile(result["coordinate_vectors"][v]["mae"],.9))
                           for v in (BASE,name)}},
        "manifest_sha256":manifest["manifest_sha256"],
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "calibration_file_sha256":_sha256_file(CALIBRATION),
        "lock_file_sha256":_sha256_file(LOCK),
        "lock_result_sha256":lock["result_sha256"],
        "member_sha256":sha,
        "provided_test_variable_deserialized":False,
        "warning":"Mat container files physically include results_test; only results_train variables were deserialized. This is not a blind external test. Spatial dependence may inflate bootstrap precision."
    })
    result=_write(RESULT,result)
    REPORT.write_text(make_report(result),encoding="utf-8",newline="\n")
    return {"decision":result["decision"],"gate":result["gate"],
            "result_sha256":result["result_sha256"]}


def make_report(result:dict) -> str:
    out=[
        "# G6 | TUJI1 training-variable-only two-axis distribution calibration",
        "",
        "5-member frozen original V17; FIT/CAL/CONFIRM split by physical coordinates.",
        "TUJI MAT files include results_test but only results_train is parsed. No external benchmark used.",
        "",
        f"Frozen CAL-selected policy: {result['selected_policy']['policy']}",
        "",
        "| CONFIRM variant | Macro MAE (m) | Energy Score (m) |",
        "|---|---:|---:|",
    ]
    for name,m in result["candidate_metrics"].items():
        out.append(f"| {name} | {m['mae_2d_m']:.6f} | {m['energy_score_2d_m']:.6f} |")
    gate=result["gate"]
    out.extend([
        "",
        f"Selected-baseline MAE: {gate['mae_delta_m']:+.6f} m",
        f"Selected-baseline ES: {gate['es_delta_m']:+.6f} m",
        f"95% familywise MAE upper: {gate['mae_upper_m']:+.6f} m",
        f"95% familywise ES upper: {gate['es_upper_m']:+.6f} m",
        f"Gate passed: {gate['passed']} — **{result['decision']}**",
        "",
        "The two one-factor variants are mechanistic ablations, NOT alternative selected policies.",
        "Baseline vs selected nominal coverage and predicted/observed radius are descriptive; Student-t sample size is finite.",
        "Joint weight temperature is NOT a demonstrated physical sample-density prior correction.",
        "G6 is a non-novel, exploratory calibration pilot only; do not compare absolute MAE with E16/25 historical benchmarks.",
        "No new independent external provided-test result. No G1/G3/G4/G5 result overturned.",
        f"Result hash: {result['result_sha256']}",""
    ])
    return "\n".join(out)


def check() -> dict:
    manifest=prepare();members=check_members()
    if not members["all_complete"]:raise G6Error("missing frozen 5-member panel")
    if not CALIBRATION.exists():
        return {"status":"PRE_CALIBRATION","complete_members":5}
    cal=_read(CALIBRATION)
    if cal["manifest_sha256"]!=manifest["manifest_sha256"] or cal["member_sha256"]!=[m["sha256"] for m in members["members"]]:
        raise G6Error("calibration checkpoint or manifest mismatch")
    if not LOCK.exists():
        return {"status":"CALIBRATION_FROZEN_PRE_CONFIRMATION","policy":cal["selected_policy"]["policy"]}
    lock=_read(LOCK)
    if lock["calibration_file_sha256"]!=_sha256_file(CALIBRATION) or lock["frozen_policy"]!=cal["selected_policy"]:
        raise G6Error("lock/calibration mismatch")
    if not RESULT.exists():return {"status":"CONFIRM_LOCK_WITHOUT_RESULT_STOP"}
    result=_read(RESULT)
    if (result["lock_file_sha256"]!=_sha256_file(LOCK)
        or result["calibration_file_sha256"]!=_sha256_file(CALIBRATION)
        or result["selected_policy"]!=cal["selected_policy"]):
        raise G6Error("confirmation provenance broken")
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8")!=make_report(result):
        raise G6Error("G6 report differs from frozen confirmation")
    return {"status":"VERIFIED","decision":result["decision"],
            "result_sha256":result["result_sha256"]}


def main(argv:Sequence[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for command in ("preflight","prepare","train-member","check-members","calibrate","confirm","check"):
        x=sub.add_parser(command)
        if command in ("train-member","calibrate","confirm"):
            x.add_argument("--device",default="cuda")
        if command=="train-member":
            x.add_argument("--member-index",type=int,required=True)
    args=p.parse_args(argv)
    if args.command=="preflight":v=preflight()
    elif args.command=="prepare":v=prepare()
    elif args.command=="train-member":v=train_member(args.member_index,args.device)
    elif args.command=="check-members":v=check_members()
    elif args.command=="calibrate":v=calibrate(args.device)
    elif args.command=="confirm":v=confirm(args.device)
    else:v=check()
    print(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=str),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
