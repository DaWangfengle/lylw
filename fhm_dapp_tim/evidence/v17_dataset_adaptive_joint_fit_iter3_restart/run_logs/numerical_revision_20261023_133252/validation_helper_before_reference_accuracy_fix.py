"""Validate only frozen clean CAL-fit rows before any formal continuation."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(r"D:\liuyanlunwen")
sys.path.insert(0,str(ROOT))
from fhm_dapp_tim.fhm_dapp import msep_v17_dataset_adaptive_joint_fit_v1 as joint

OUT=ROOT/"fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart"
AUDIT=OUT/"run_logs/numerical_revision_20261023_133252"
DEST=AUDIT/(sys.argv[1] if len(sys.argv)>1 else "VALIDATION_CLEAN_ANCHORS.json")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def save(records,status):
    with DEST.open("x",encoding="utf-8") as handle:
        json.dump({"status":status,"core_source_sha256":sha(Path(joint.__file__)),
                   "validation_script_sha256":sha(Path(__file__)),"records":records,
                   "scope":"Frozen CAL-fit clean rows only, keeping original 2x9 batch shape; no selection or CAL-eval",
                   "internal_test_rows_scored":0,"official_test_rows_scored":0,
                   "checkpoint_count_completed":len({(r['dataset'],r['seed'],r['backbone']) for r in records if r['passed']})},
                  handle,ensure_ascii=False,indent=2,allow_nan=False)


@torch.no_grad()
def main():
    assert not DEST.exists()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    search=json.loads((OUT/"SPATIAL_SCALE_SEARCH_SPACE.json").read_text(encoding="utf-8"))["outer_geometry_candidates"]
    records=[]
    for dataset in ("uts","uji"):
        for seed in (17301,17302,17303):
            ctx=joint.s3.reconstruct(dataset,seed,Path(joint.DEFAULT_INPUT),torch.device("cuda"))
            fit=joint._safe_rows(ctx,ctx["cal_fit"],"CAL-fit")
            assert not set(map(int,fit))&set(map(int,ctx["cal_eval"]))
            coords=ctx["memory"]["coords"].cpu().numpy().astype(np.float64,copy=False)
            building=ctx["memory"]["building"].cpu().numpy()
            floor=ctx["memory"]["floor"].cpu().numpy()
            edges=ctx["edges"].cpu().numpy()
            lengths=np.linalg.norm(coords[edges[:,0]]-coords[edges[:,1]],axis=1)
            geometries=[]
            for geom in search[f"{dataset}_seed{seed}"]["geometry_candidates"]:
                graph,kept,diag=joint.build_cutoff_graph(coords,building,floor,edges,geom["r_max_s1_m"],geom["tau_graph_m"])
                selected_edges=edges[lengths<=geom["r_max_s2_m"]]
                support=geom["support_diagnostics"]
                assert len(kept)==support["retained_s1_edges"]
                assert len(selected_edges)==support["retained_virtual_edges"]>=math.ceil(.45*len(edges))
                endpoint=joint.normalized_virtual_endpoint_weights(selected_edges,coords,geom["tau_virt_m"])
                geometries.append((geom,graph.to("cuda"),selected_edges,endpoint))
            for arm in "CDE":
                model,_,_=joint.comp._load_model(ctx,Path(joint.DEFAULT_CHECKPOINT_ROOT),dataset,seed,arm,"cuda")
                checkpoint=joint.comp._checkpoint_path(Path(joint.DEFAULT_CHECKPOINT_ROOT),dataset,seed,arm)
                identity=sha(checkpoint)
                pack=joint.s3.make_pack(ctx,model,fit,virtual_gamma=0.)
                saved=json.loads((joint.comp.DEFAULT_OUTPUT/"runs"/f"{dataset}_seed{seed}"/arm/"CAL_FIT_SELECTION.json").read_text(encoding="utf-8"))
                assert saved["checkpoint_sha256"]==identity
                clean=saved["joint_calibrated_base"]
                params={"geometry_id":"legacy_unbounded_8m","R_max_S1_m":math.inf,"tau_graph_m":8.,
                        "R_max_S2_m":math.inf,"tau_virt_m":math.inf,"temperature":clean["temperature"],
                        "theta":clean["theta"],"alpha_prior":0.,"alpha_obs":0.,"beta":0.}
                mix,sub,_=joint._make_probability(pack,ctx,params)
                metrics,_=joint.s3.spatial_metrics(mix["q"],mix["ids"],mix["xy"],sub["target"],sub["building"],sub["floor"],
                                                 ctx["classes"],ctx["class_pairs"],keys=sub["keys"])
                cls=ctx["classes"][mix["ids"]]
                masses=torch.zeros((len(mix["q"]),len(ctx["class_pairs"])),device="cuda",dtype=mix["q"].dtype)
                masses.scatter_add_(1,cls,mix["q"])
                target_cls=joint._target_class_ids(sub,ctx)
                tm=masses.gather(1,target_cls[:,None]).squeeze(1)
                base_bfce=float(((-tm.clamp_min(torch.finfo(tm.dtype).tiny).log())*sub["weight"]).sum().cpu())
                candidates=joint._calibration_candidates(params)
                signatures=list(dict.fromkeys(tuple(c[k] for k in ("temperature","theta_v","theta_s","alpha_prior","alpha_obs")) for c in candidates))
                clean_indices=[i for i,c in enumerate(candidates) if c["temperature"]==params["temperature"]
                               and [c["theta_v"],c["theta_s"]]==params["theta"]
                               and c["alpha_prior"]==c["alpha_obs"]==c["beta"]==0.]
                assert len(clean_indices)==1
                original=candidates[clean_indices[0]]
                signature=tuple(original[k] for k in ("temperature","theta_v","theta_s","alpha_prior","alpha_obs"))
                group=signatures.index(signature)
                pair=signatures[(group//2)*2:(group//2)*2+2]
                batch=[c for c in candidates if tuple(c[k] for k in ("temperature","theta_v","theta_s","alpha_prior","alpha_obs")) in pair]
                ix=next(i for i,c in enumerate(batch) if c["candidate_id"]==original["candidate_id"])
                model_records=[]
                for geom,graph,selected_edges,endpoint in geometries:
                    values=joint.evaluate_calibration_grid(pack,ctx,graph,selected_edges,geom,batch,endpoint_cache=endpoint)[ix]
                    dm=values["coordinate_mae_m"]-metrics["macro_mae_full_m"]
                    da=values["bf_accuracy"]-metrics["macro_bf_accuracy"]
                    dc=values["bf_cross_entropy"]-base_bfce
                    passed=(dm<=1e-6 and da>=-1e-12 and values["bf_cross_entropy"]<=base_bfce*1.02+.005)
                    record={"dataset":dataset,"seed":seed,"backbone":arm,"geometry_id":geom["geometry_id"],
                            "passed":passed,"reference_mae":metrics["macro_mae_full_m"],"grid_mae":values["coordinate_mae_m"],
                            "mae_delta_m":dm,"reference_bf_accuracy":metrics["macro_bf_accuracy"],
                            "grid_bf_accuracy":values["bf_accuracy"],"bf_accuracy_delta":da,
                            "bfce_delta":dc,"checkpoint_sha256":identity,"cal_fit_key_sha256":joint.s3.key_sha(pack["keys"]),
                            "cal_fit_rows":len(fit),"s1_edges":geom["support_diagnostics"]["retained_s1_edges"],"s2_edges":len(selected_edges)}
                    records.append(record); model_records.append(record)
                    if not passed:
                        save(records,"FAILED_CLEAN_ANCHOR_VALIDATION")
                        raise RuntimeError(json.dumps(record,ensure_ascii=False))
                assert sha(checkpoint)==identity and all(p.grad is None for p in model.parameters())
                print(json.dumps({"stage":"clean_anchor_checkpoint_validated","dataset":dataset,"seed":seed,"backbone":arm,
                                  "geometries":len(model_records),"mae_delta_range":[min(r["mae_delta_m"] for r in model_records),max(r["mae_delta_m"] for r in model_records)]}),flush=True)
                del pack,model,mix,sub
        torch.cuda.empty_cache()
    assert len(records)==216
    save(records,"PASS_18_CHECKPOINTS_216_CLEAN_ANCHORS")
    print(json.dumps({"status":"PASS","checkpoints":18,"clean_anchors":len(records)}),flush=True)


if __name__=="__main__":
    main()
