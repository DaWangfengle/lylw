"""G5 pre-manifest data check and post-freeze FIT-only GPU smoke test."""
from __future__ import annotations
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import scipy
import torch
from fhm_dapp_tim.fhm_dapp import uji_v17_g5_bayes_action as g5

AUDIT=Path(__file__).resolve().parent

def emit(name,value):
    with (AUDIT/name).open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),flush=True)

def bijection(first,second):
    forward,reverse={},{}
    for a,b in zip(first,second):
        assert a not in forward or forward[a]==b,"physical site split by coordinate conversion"
        assert b not in reverse or reverse[b]==a,"coordinate conversion merged physical sites"
        forward[a],reverse[b]=b,a
    assert len(forward)==len(reverse)==len(set(first))
    return len(forward)

def role_hash(keys):
    raw=json.dumps([list(k) for k in sorted(keys)],ensure_ascii=False,
                   separators=(",",":")).encode("utf-8")
    import hashlib
    return hashlib.sha256(raw).hexdigest()

def preparation():
    assert not g5.MANIFEST.exists(),"review must precede manifest"
    data,groups,rows=g5._training()
    raw=np.loadtxt(g5.TRAIN_XYZ,delimiter=",",ndmin=2)
    canonical,reference=g5.ds.load_tampere("train")
    raw_keys=g5.location_keys(raw[:,:2],data["building"],data["floor"])
    raw_bijection=bijection(raw_keys,data["keys"])
    origin=data["raw_xy"][rows["fit"]].mean(axis=0).astype(np.float32)
    centered=(data["raw_xy"]-origin[None,:]).astype(np.float32)
    centered_keys=g5.location_keys(centered,data["building"],data["floor"])
    centered_bijection=bijection(data["keys"],centered_keys)
    assert all(set(rows[a]).isdisjoint(rows[b]) for a,b in [("fit","confirmation")])
    assert set(rows["fit"])|set(rows["confirmation"])==set(range(len(raw)))
    assert (g5.SEEDS,g5.EPOCHS)==((17301,17302,17303,17304,17305),(65,69,56,67,62))
    assert (g5.L,g5.K,g5.ALPHA,g5.PANEL,g5.r3p.STUDENT_DF)==(192,192,1.,5,4.)
    assert (g5.ACTION_SAMPLING_SEED,g5.RISK_SAMPLING_SEED,g5.ENERGY_SAMPLING_SEED,
            g5.BOOTSTRAP_SEED)==(19503,19504,19505,19502)
    assert (g5.ACTION_DRAWS_PER_STREAM,g5.RISK_DRAWS_PER_STREAM,
            g5.MEDIAN_ITERATIONS,g5.MEDIAN_TOLERANCE_M,g5.MIN_GAIN_M,
            g5.BOOTSTRAP_REPS)==(128,128,64,1e-5,.15,5000)
    assert g5.r3p.PREDICTIVE_DRAWS_PER_STREAM==64
    assert len(groups["fit"])>=192
    assert torch.cuda.is_available(),"CUDA required for the registered run"
    prop=torch.cuda.get_device_properties(0)
    floors={str(f):{role:sum(k[1]==f for k in keys)
                   for role,keys in groups.items()}
            for f in sorted(set(data["floor"].tolist()))}
    emit("PREPARATION_REVIEW.json",{
        "status":"PRE_MANIFEST_DATA_AND_ENVIRONMENT_VERIFIED",
        "environment":{"python":sys.version,"python_executable":sys.executable,
                       "torch":torch.__version__,"cuda":torch.version.cuda,
                       "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,
                       "platform":platform.platform(),"gpu":prop.name,
                       "gpu_total_bytes":prop.total_memory,
                       "cuda_free_total_bytes":list(torch.cuda.mem_get_info())},
        "source_sha256":g5._source_hashes(),
        "rows":len(raw),"aps":data["rss"].shape[1],
        "role_groups":{r:len(k) for r,k in groups.items()},
        "role_rows":{r:len(k) for r,k in rows.items()},
        "role_sorted_keys_sha256":{r:role_hash(k) for r,k in groups.items()},
        "physical_overlap":len(groups["fit"]&groups["confirmation"]),
        "row_overlap":len(set(rows["fit"])&set(rows["confirmation"])),
        "raw_to_adapter_coordinate_key_bijection":raw_bijection,
        "adapter_to_centered_float32_key_bijection_all_roles":centered_bijection,
        "fit_anchor_count":len(set(centered_keys[int(i)] for i in rows["fit"])),
        "fit_origin_float32":origin.tolist(),"by_floor":floors,
        "floor_height_mapping":reference.floor_mapping,
        "unknown_device_ids":np.unique(data["phone"]).tolist(),
        "xy_bounds_m":[raw[:,:2].min(axis=0).tolist(),raw[:,:2].max(axis=0).tolist()],
        "rss_model_bounds":[float(data["rss"].min()),float(data["rss"].max())],
        "query_target_dimension":2,"z_is_floor_category":True,
        "no_confirmation_prediction_or_error_computed":True})

def gpu_smoke():
    assert g5.FREEZE.exists() and not g5.LOCK.exists() and not g5.RESULT.exists()
    start=time.perf_counter()
    frozen=g5._read(g5.FREEZE)
    manifest=g5.prepare()
    hashes_before=[g5._sha256_file(g5.member_path(i)) for i in range(1,6)]
    data,groups,rows=g5._training()
    models,origin=g5._load_models("cuda",frozen)
    histories=[]
    for i in range(1,6):
        obj=torch.load(g5.member_path(i),map_location="cpu",weights_only=True)
        assert len(obj["refit_history"])==g5.EPOCHS[i-1]
        assert [item["epoch"] for item in obj["refit_history"]]==list(range(1,g5.EPOCHS[i-1]+1))
        assert obj["member_seed"]==g5.SEEDS[i-1]
        assert obj["epoch"]==g5.EPOCHS[i-1]
        assert obj["config"]["epochs"]==g5.EPOCHS[i-1]
        assert obj["data_provenance"]["fit_coordinate_only"] is True
        histories.append({"member":i,"seed":obj["member_seed"],"epochs":obj["epoch"],
                          "history_epochs":len(obj["refit_history"]),"sha256":hashes_before[i-1]})
    ref=models[0]
    assert ref.n_anchors==len(groups["fit"])
    for model in models:
        assert float(model.student_df)==4. and model.residual_experts==1
        assert not model.use_candidate_reranker
        assert torch.equal(model.encoder.seen_ap_mask,ref.encoder.seen_ap_mask)
    device=torch.device("cuda")
    ids=rows["fit"][:g5.BATCH]
    rss=data["rss"][ids].to(device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        prototype=g5.r2b._prototype_distance(ref,rss)
        indices=torch.arange(ref.n_anchors,device=device)[None].expand(len(ids),-1)
        logits=[];location=[];scale=[]
        for model in models:
            embed=model.encoder(rss)
            logits.append(model.retrieval_logits_from_embedding(embed))
            centre,spread=model._components(embed,indices)
            location.append(centre);scale.append(spread)
        loc,sc=torch.stack(location),torch.stack(scale)
        prior=torch.softmax(g5.r3p.BETA_NEURAL*torch.stack(logits)
                            -g5.r3p.BETA_PROTOTYPE*prototype[None,:,:],dim=2)
        top,topids,radial,_=g5.r3._member_top_l(prior,loc,sc,g5.L,student_df=4.)
        p,c,s=g5.r3._dense_joint_law(top,topids,radial,loc,sc,tuple(range(5)),
                                   top_k=g5.K,alpha=g5.ALPHA)
        assert p.shape==(len(ids),960)
        assert torch.allclose(p.sum(1),torch.ones(len(ids),device=device),atol=1e-5,rtol=0)
        random=g5.common_student_t_randomness(len(ids),draws=128,seed=19503)
        acts=g5.three_actions(p,c,s,g5._randomness_slice(random,0,len(ids),device))
        assert all(torch.isfinite(x).all() for x in acts.values())
        assert torch.equal(acts["mean"],torch.sum(p[:,:,None]*c,dim=1))
        risk=g5.common_student_t_randomness(len(ids),draws=128,seed=19504)
        risk=g5._randomness_slice(risk,0,len(ids),device)
        draws=g5.sample_continuous_mixture(p,c,s,risk["uniform_1"],risk["noise_1"])
        for action in acts.values():
            assert torch.isfinite(torch.linalg.vector_norm(draws-action[:,None,:],dim=-1)).all()
        energy=g5.common_student_t_randomness(len(ids),draws=64,seed=19505)
        # Artificial zero target exercises the CUDA scorer without scoring a real query label.
        shared_es,mae=g5.r3._score_joint_law(p,c,s,torch.zeros(len(ids),2,device=device,dtype=torch.float64),
                                          g5._randomness_slice(energy,0,len(ids),device))
        assert torch.isfinite(shared_es).all()
        assert torch.equal(mae,torch.linalg.vector_norm(acts["mean"].double(),dim=1).cpu())
    torch.cuda.synchronize()
    assert all(parameter.grad is None for model in models for parameter in model.parameters())
    hashes_after=[g5._sha256_file(g5.member_path(i)) for i in range(1,6)]
    assert hashes_after==hashes_before
    emit("FROZEN_FIT_GPU_SMOKE.json",{
        "status":"FROZEN_FIT_ONLY_GPU_COMPATIBILITY_VERIFIED","seconds":time.perf_counter()-start,
        "fit_scan_rows":ids.tolist(),"confirmation_scans_used":0,
        "confirmation_labels_scored":0,"scorer_target":"synthetic all-zero target only",
        "joint_law_shape":list(p.shape),"action_draws_per_stream":128,
        "probability_sums_bounds":[float(p.sum(1).min()),float(p.sum(1).max())],
        "original_r3_mean_bitwise_equal":True,"shared_fit_anchors_and_origin":True,
        "all_fixed_epochs_complete":True,"all_model_grads_none":True,
        "checkpoints_unchanged":True,"members":histories,
        "origin":origin.tolist(),"manifest_sha256":manifest["manifest_sha256"],
        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved()})

if __name__=="__main__":
    {"preparation":preparation,"gpu-smoke":gpu_smoke}[sys.argv[1]]()
