"""G6 structural/engineering review only; never scores CAL/CONFIRM labels."""
import copy
import hashlib
import json
import platform
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
import torch
from fhm_dapp_tim.fhm_dapp import uji_v17_g6_calibration as g6
AUDIT = Path(__file__).resolve().parent

def key_sha(keys):
    return hashlib.sha256(json.dumps([list(k) for k in sorted(keys)],
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def smoke(models, data, fit_rows, origin):
    device = torch.device("cuda")
    ids = fit_rows[:16]
    rss = data["rss"][ids].float().to(device)
    ref = models[0]
    with torch.no_grad(), g6.r3.r2f_a1._deterministic_algorithms():
        prototype = g6.r2b._prototype_distance(ref, rss)
        anchor_ix = torch.arange(ref.n_anchors,device=device)[None,:].expand(len(ids),-1)
        logits=[];centres=[];scales=[]
        for model in models:
            model.eval()
            embedding=model.encoder(rss)
            logits.append(model.retrieval_logits_from_embedding(embedding))
            m,s=model._components(embedding,anchor_ix)
            centres.append(m);scales.append(s)
        prior=torch.softmax(g6.r3p.BETA_NEURAL*torch.stack(logits)
            -g6.r3p.BETA_PROTOTYPE*prototype[None,:,:],dim=-1)
        locations=torch.stack(centres);sigma=torch.stack(scales)
        top,ix,rad,_=g6.r3._member_top_l(prior,locations,sigma,g6.L,student_df=g6.r3p.STUDENT_DF)
        p,loc,scale=g6.r3._dense_joint_law(top,ix,rad,locations,sigma,
            tuple(range(5)),top_k=g6.K,alpha=g6.ALPHA)
        assert p.shape==(len(ids),960)
        assert torch.allclose(p.sum(1),torch.ones(len(ids),device=device),atol=1e-5)
        assert torch.isfinite(p).all() and torch.isfinite(loc).all()
        assert torch.isfinite(scale).all() and (scale>0).all()
        random=g6._random_slice(g6.common_student_t_randomness(
            len(ids),draws=64,degrees_freedom=4.,seed=g6.SCORE_SEED),0,len(ids),device)
        synthetic_target=torch.zeros(len(ids),2,dtype=torch.float64,device=device)
        original_mean=(p[:,:,None]*loc).sum(dim=1)
        original_es,original_mae=g6.r3._score_joint_law(p,loc,scale,synthetic_target,random)
        originals=g6.sample_continuous_mixture(p,loc,scale,random["uniform_1"],random["noise_1"])
        mae_by_t={}
        for t,s in g6.CANDIDATES:
            q,centre,sig=g6.calibrate_joint_law(p,loc,scale,t,s)
            es,mae=g6.r3._score_joint_law(q,centre,sig,synthetic_target,random)
            assert torch.isfinite(es).all() and torch.isfinite(mae).all()
            if t in mae_by_t:assert torch.equal(mae_by_t[t],mae)
            else:mae_by_t[t]=mae
            if (t,s)==(1.,1.):
                assert q is p and centre is loc and sig is scale
                assert torch.equal((q[:,:,None]*centre).sum(1),original_mean)
                assert torch.equal(es,original_es) and torch.equal(mae,original_mae)
                assert torch.equal(g6.sample_continuous_mixture(
                    q,centre,sig,random["uniform_1"],random["noise_1"]),originals)
        assert all(param.grad is None for model in models for param in model.parameters())
    torch.cuda.synchronize()
    return {"input":"first 16 FIT RSS rows; all scorer targets are synthetic zeros",
            "query_rows":len(ids),"joint_components":960,"candidate_count":9,
            "draws_per_stream":64,"paired_common_randomness":True,
            "baseline_law_mean_MAE_ES_draws_bitwise_original_R3":True,
            "same_T_all_s_MAE_bitwise_equal":True,"finite_probabilities_centres_scales_scores":True,
            "network_gradients_none":True,"CAL_CONFIRM_predictions_or_errors_computed":False,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved()}

def main(mode):
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    canonical=g6.load_tuji_train_variable_only()
    data=g6._as_r0_input(canonical)
    roles,rows=g6._roles(data)
    if mode=="before_manifest":
        assert not g6.MANIFEST.exists()
        raw=[]
        mat_review=[]
        for name in g6.EXPECTED_DEVICES:
            payload=loadmat(g6.MAT_DIR/g6.ds.TUJI_DEVICE_FILES[name],squeeze_me=True,
                            struct_as_record=False,variable_names=("results_train",))
            assert "results_test" not in payload
            scans=np.asarray(payload["results_train"].Wifi,dtype=object).reshape(-1).tolist()
            raw.extend(np.asarray(scan.coords,dtype=np.float64).reshape(2) for scan in scans)
            mat_review.append({"device":name,"train_rows":len(scans),
                              "returned_keys":sorted(payload),"results_test_returned":False})
        raw=np.asarray(raw)
        raw_keys=g6.location_keys(raw,data["building"],data["floor"])
        assert raw_keys==data["keys"]
        origin=data["raw_xy"][rows["fit"]].mean(0).astype(np.float32)
        centred=(data["raw_xy"]-origin[None,:]).astype(np.float32)
        centred_keys=g6.location_keys(centred,data["building"],data["floor"])
        forward={};reverse={}
        for a,b in zip(raw_keys,centred_keys):
            assert a not in forward or forward[a]==b
            assert b not in reverse or reverse[b]==a
            forward[a]=b;reverse[b]=a
        model,*rest=g6.r0._build_member(data,rows["fit"],device=torch.device("cuda"))
        models=[model]+[copy.deepcopy(model) for _ in range(4)]
        report={"stage":mode,"environment":{"python":sys.version,"python_executable":sys.executable,
            "torch":torch.__version__,"cuda_runtime":torch.version.cuda,
            "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,
            "platform":platform.platform(),"gpu":torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes":torch.cuda.get_device_properties(0).total_memory},
            "mat_variable_review":mat_review,"coordinate_identity":{
                **canonical.metadata,"raw_original_keys_equal_canonical_keys":True,
                "original_to_centred_all_role_bijection":True,"original_sites":len(forward),
                "centred_sites":len(reverse),"coordinate_origin_fit_only":origin.tolist(),
                "raw_xy_bounds":[raw.min(0).tolist(),raw.max(0).tolist()]},
            "role_key_sha256":{name:key_sha(keys) for name,keys in roles.items()},
            "role_sites":{name:len(keys) for name,keys in roles.items()},
            "role_rows":{name:len(ix) for name,ix in rows.items()},
            "rss_dbm_visible_bounds":[float(np.nanmin(canonical.rss_dbm)),float(np.nanmax(canonical.rss_dbm))],
            "model_rss_bounds":[float(data["rss"].min()),float(data["rss"].max())],
            "ap_count":len(canonical.ap_ids),
            "ap_dictionary_sha256":hashlib.sha256(json.dumps(canonical.ap_ids,
                separators=(",",":")).encode()).hexdigest(),
            "model_weights":"fresh untrained original R0 FIT-memory models, not saved",
            "CUDA_smoke":smoke(models,data,rows["fit"],origin)}
        output=AUDIT/"PRE_MANIFEST_REVIEW.json"
    elif mode=="trained_panel":
        manifest=g6.prepare()
        models,origin,hashes=g6._load_models("cuda",manifest)
        members=[]
        for i in range(1,6):
            ck=torch.load(g6.member_path(i),map_location="cpu",weights_only=True)
            history=ck["refit_history"]
            assert [entry["epoch"] for entry in history]==list(range(1,g6.EPOCHS[i-1]+1))
            assert ck["data_provenance"]=={
                "variable":"results_train","fit_only":True,"test_variable_loaded":False}
            assert ck["config"]["epochs"]==g6.EPOCHS[i-1]
            members.append({"member":i,"seed":g6.SEEDS[i-1],"epoch":g6.EPOCHS[i-1],
                            "history_entries":len(history),"sha256":hashes[i-1],
                            "data_provenance":ck["data_provenance"]})
        before={k:g6._sha256_file(p) for k,p in (
            ("manifest",g6.MANIFEST),("source",Path(g6.__file__)))}
        smoke_record=smoke(models,data,rows["fit"],origin)
        assert hashes==[g6._sha256_file(g6.member_path(i)) for i in range(1,6)]
        assert before=={k:g6._sha256_file(p) for k,p in (
            ("manifest",g6.MANIFEST),("source",Path(g6.__file__)))}
        assert all(torch.equal(model.encoder.seen_ap_mask,models[0].encoder.seen_ap_mask) for model in models)
        report={"stage":mode,"members":members,"all_five_complete_distinct_hashes":len(set(hashes))==5,
                "identical_FIT_anchors_buildings_floors_prototype_RSS_origin_seen_AP_mask":True,
                "coordinate_origin":origin.tolist(),"manifest_and_source_unchanged":True,
                "CUDA_smoke":smoke_record}
        output=AUDIT/"TRAINED_PANEL_REVIEW.json"
    else:raise ValueError(mode)
    with output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"status":"ENGINEERING_REVIEW_PASSED","stage":mode,"output":str(output),
        "CUDA_smoke":report["CUDA_smoke"]},ensure_ascii=False))
if __name__=="__main__":
    main(sys.argv[1])
