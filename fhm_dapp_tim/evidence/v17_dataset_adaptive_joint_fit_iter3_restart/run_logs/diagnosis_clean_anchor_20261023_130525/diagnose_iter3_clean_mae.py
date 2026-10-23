"""Isolated numerical diagnosis. Never run/aggregate/refit the frozen experiment."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(r"D:\liuyanlunwen")
sys.path.insert(0, str(ROOT))
from fhm_dapp_tim.fhm_dapp import msep_v17_dataset_adaptive_joint_fit_v1 as joint

OUT = ROOT / "fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart"
AUDIT = OUT / "run_logs/diagnosis_clean_anchor_20261023_130525"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def macro(errors, keys, *, double=False):
    values = errors.detach().cpu().numpy()
    if double:
        values = values.astype(np.float64)
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    return float(np.mean([values[ix].mean() for ix in groups.values()]))


@torch.no_grad()
def diagnose(pack, ctx, graph, edges, endpoint, geometry, candidates, clean_index, label):
    center = candidates[clean_index]
    theta = pack["scores"].new_tensor([center["theta_v"], center["theta_s"]])
    _, logits = joint.op.full_logit_graph_diffusion(pack["scores"], pack["z"], theta,
        center["temperature"], graph, 0., 0.)
    refq = torch.softmax(logits, -1)
    refpoint = (refq[..., None]*pack["xy"]).sum(1)
    referr = torch.linalg.vector_norm(refpoint-pack["target"], dim=-1)
    n = refq.shape[1]
    native = inspect.unwrap(joint.evaluate_calibration_grid)
    source, start = inspect.getsourcelines(native)
    capture_line = start + next(i for i, line in enumerate(source)
                                if 'err=torch.linalg.vector_norm(point-target' in line)
    signatures = []
    for candidate in candidates:
        signature = tuple(candidate[k] for k in ("temperature", "theta_v", "theta_s", "alpha_prior", "alpha_obs"))
        if signature not in signatures:
            signatures.append(signature)
    clean_signature = tuple(center[k] for k in ("temperature", "theta_v", "theta_s", "alpha_prior", "alpha_obs"))
    ci = signatures.index(clean_signature)
    assert len(signatures) <= 2
    bi = sorted({candidate["beta"] for candidate in candidates}).index(0.)
    records = {}
    sums = {}
    details = []
    maxima = {k: 0. for k in ("parent_logit_gap", "q_gap", "virtual_mass", "point_gap_m", "point_gap_same_q_m")}

    def trace(frame, event, arg):
        if frame.f_code is native.__code__ and event == "line" and frame.f_lineno == capture_line:
            loc = frame.f_locals
            sl = loc["sl"]
            q, centers, target = loc["q"], loc["sub_centers"], loc["target"]
            actual_point = loc["point"][ci, bi]
            actual_q = q[ci, bi]
            rpoint = refpoint[sl]
            padded_refq = torch.cat((refq[sl], torch.zeros_like(actual_q[:, n:])), dim=-1)
            altq = q.clone()
            altq[ci, bi] = padded_refq
            same_q_einsum = torch.einsum("ckbm,bmd->ckbd", altq, centers)[ci, bi]
            points = {
                "reference_mul_sum_physical": rpoint,
                "reference_q_mul_sum_padded": (padded_refq[..., None]*centers).sum(1),
                "reference_q_grid_einsum": same_q_einsum,
                "grid_q_mul_sum_padded": (actual_q[..., None]*centers).sum(1),
                "grid_native_einsum": actual_point,
                "reference_q_float64_point": (refq[sl].double()[..., None]*pack["xy"][sl].double()).sum(1),
                "grid_q_float64_point": (actual_q.double()[..., None]*centers.double()).sum(1),
            }
            errors = {name: torch.linalg.vector_norm(point-target.to(point.dtype), dim=-1)
                      for name, point in points.items()}
            native_errors = torch.linalg.vector_norm(loc["point"]-target[None,None,:,:],dim=-1)
            weights = pack["weight"][sl][None,None,:]
            for name, error in errors.items():
                records.setdefault(name, []).append(error.detach().cpu())
                replaced = native_errors.clone()
                replaced[ci,bi] = error.to(replaced.dtype)
                partial = ((replaced*weights).sum(-1))[ci,bi]
                if name not in sums:
                    sums[name] = partial.new_zeros(())
                sums[name] += partial
            maxima["parent_logit_gap"] = max(maxima["parent_logit_gap"],float((loc["parent"][ci]-logits[sl]).abs().max()))
            maxima["q_gap"] = max(maxima["q_gap"],float((actual_q[:,:n]-refq[sl]).abs().max()))
            maxima["virtual_mass"] = max(maxima["virtual_mass"],float(actual_q[:,n:].sum(-1).abs().max()))
            maxima["point_gap_m"] = max(maxima["point_gap_m"],float((actual_point-rpoint).norm(dim=-1).max()))
            maxima["point_gap_same_q_m"] = max(maxima["point_gap_same_q_m"],float((same_q_einsum-rpoint).norm(dim=-1).max()))
            details.append({"start":sl.start,"stop":sl.stop,"component_count":q.shape[-1],"group_beta_rows":list(q.shape[:2]),
                            "softmax_sum_error":float((actual_q.sum(-1)-1).abs().max())})
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        rows = joint.evaluate_calibration_grid(pack, ctx, graph, edges,
            geometry, candidates, endpoint_cache=endpoint)
    finally:
        sys.settrace(previous)
    assert records
    actual = rows[clean_index]["coordinate_mae_m"]
    assert actual == float(sums["grid_native_einsum"]), (actual,float(sums["grid_native_einsum"]))
    table = {}
    for name, errors in records.items():
        error = torch.cat(errors)
        table[name] = {"original_reference_site_macro":macro(error,pack["keys"]),
                       "site_macro_float64":macro(error,pack["keys"],double=True),
                       "grid_weighted_float32":float(sums[name]),
                       "delta_from_reference_row":macro(error,pack["keys"])-macro(referr,pack["keys"])}
    ref_macro = macro(referr,pack["keys"])
    result = {"label":label,"dtype":str(pack["scores"].dtype),"device":str(pack["scores"].device),
              "rows":len(refq),"physical_components":n,"virtual_components":len(edges),
              "reference_mae":ref_macro,"grid_mae":actual,"grid_minus_reference_mae":actual-ref_macro,
              "gate_threshold":1e-6,"original_mae_guard_passes":actual <= ref_macro+1e-6,
              "maxima":maxima,"arithmetic_counterfactuals":table,"row_chunks":details,
              "per_scan_error_differences":{name:(torch.cat(errors).double()-referr.cpu().double()).tolist()
                                            for name,errors in records.items()},
              "clean_parameters":center}
    print(json.dumps({k:result[k] for k in ("label","dtype","device","reference_mae","grid_mae","grid_minus_reference_mae","maxima")}),flush=True)
    return result


def synthetic(device, dtype):
    generator = torch.Generator().manual_seed(131)
    b,n,e = 413,1052,2500
    coords = torch.randn(n,2,generator=generator)*20+50
    pairs = []
    for distance in (1,2,3):
        pairs.extend((i,i+distance) for i in range(n-distance))
    edges = np.asarray(pairs[:e],dtype=np.int64)
    def move(value):
        return value.to(device=device,dtype=dtype) if value.is_floating_point() else value.to(device)
    coords=move(coords)
    xy=coords[None].expand(b,-1,-1).contiguous()
    edge_tensor=torch.as_tensor(edges,device=device)
    virtual=(xy[:,edge_tensor[:,0]]+xy[:,edge_tensor[:,1]])/2
    pack={"scores":move(torch.randn(b,n,generator=generator)),"z":move(torch.randn(b,n,2,generator=generator)),
          "virtual_scores":move(torch.randn(b,e,generator=generator)),"virtual_z":move(torch.randn(b,e,2,generator=generator)),
          "xy":xy,"sigma":torch.ones_like(xy),"virtual_xy":virtual,"virtual_sigma":torch.ones_like(virtual),
          "target":move(torch.randn(b,2,generator=generator)*5+45),
          "building":torch.zeros(b,dtype=torch.long,device=device),"floor":torch.zeros(b,dtype=torch.long,device=device),
          "keys":[i//5 for i in range(b)]}
    counts={key:pack["keys"].count(key) for key in set(pack["keys"])}
    pack["weight"]=move(torch.tensor([1/(len(counts)*counts[key]) for key in pack["keys"]]))
    ctx={"memory":{"coords":coords,"building":torch.zeros(n,dtype=torch.long,device=device),"floor":torch.zeros(n,dtype=torch.long,device=device)},
         "edges":edge_tensor,"classes":torch.zeros(n,dtype=torch.long,device=device),
         "class_pairs":torch.tensor([[0,0]],device=device)}
    graph=torch.eye(n,device=device,dtype=dtype).to_sparse().coalesce()
    endpoint=joint.normalized_virtual_endpoint_weights(edges,coords.cpu().numpy(),math.inf)
    candidates=[]
    for group in range(2):
        for beta in joint.BETAS:
            candidates.append({"candidate_id":f"synthetic_{group}_{beta}","temperature":.9805950440880841,
                "theta_v":1.5,"theta_s":1.5,"alpha_prior":0.,"alpha_obs":group*.1,"beta":beta})
    return diagnose(pack,ctx,graph,edges,endpoint,{"tau_virt_m":math.inf},candidates,0,f"synthetic_{device}_{dtype}")


@torch.no_grad()
def isolated_real():
    seed=17301
    device=torch.device("cuda")
    ctx=joint.s3.reconstruct("uts",seed,Path(joint.DEFAULT_INPUT),device)
    fit=joint._safe_rows(ctx,ctx["cal_fit"],"CAL-fit")
    assert not set(map(int,fit)) & set(map(int,ctx["cal_eval"]))
    model,payload,_=joint.comp._load_model(ctx,Path(joint.DEFAULT_CHECKPOINT_ROOT),"uts",seed,"C","cuda")
    checkpoint=joint.comp._checkpoint_path(Path(joint.DEFAULT_CHECKPOINT_ROOT),"uts",seed,"C")
    identity=sha(checkpoint)
    pack=joint.s3.make_pack(ctx,model,fit,virtual_gamma=0.)
    # Read the already frozen CAL-fit parameter identity; no fitting or search.
    saved=json.loads((joint.comp.DEFAULT_OUTPUT/'runs/uts_seed17301/C/CAL_FIT_SELECTION.json').read_text(encoding='utf-8'))
    assert saved['checkpoint_sha256']==identity
    config=saved['joint_calibrated_base']
    full=joint._calibration_candidates(config)
    clean=[i for i,c in enumerate(full) if c['temperature']==config['temperature']
           and [c['theta_v'],c['theta_s']]==config['theta']
           and c['alpha_prior']==c['alpha_obs']==c['beta']==0.]
    assert clean==[2754],clean
    signatures=list(dict.fromkeys(tuple(c[k] for k in ('temperature','theta_v','theta_s','alpha_prior','alpha_obs')) for c in full))
    clean_signature=tuple(full[clean[0]][k] for k in ('temperature','theta_v','theta_s','alpha_prior','alpha_obs'))
    group=signatures.index(clean_signature)
    pair=signatures[(group//2)*2:(group//2)*2+2]
    candidates=[c for c in full if tuple(c[k] for k in ('temperature','theta_v','theta_s','alpha_prior','alpha_obs')) in pair]
    clean_index=next(i for i,c in enumerate(candidates) if c['candidate_id']==full[clean[0]]['candidate_id'])
    search=json.loads((OUT/'SPATIAL_SCALE_SEARCH_SPACE.json').read_text(encoding='utf-8'))
    geometry=search['outer_geometry_candidates']['uts_seed17301']['geometry_candidates'][0]
    coords=ctx['memory']['coords'].cpu().numpy().astype(np.float64,copy=False)
    building=ctx['memory']['building'].cpu().numpy(); floor=ctx['memory']['floor'].cpu().numpy()
    all_edges=ctx['edges'].cpu().numpy()
    graph,_,_=joint.build_cutoff_graph(coords,building,floor,all_edges,geometry['r_max_s1_m'],geometry['tau_graph_m'])
    lengths=np.linalg.norm(coords[all_edges[:,0]]-coords[all_edges[:,1]],axis=1)
    edges=all_edges[lengths<=geometry['r_max_s2_m']]
    endpoint=joint.normalized_virtual_endpoint_weights(edges,coords,geometry['tau_virt_m'])
    result=diagnose(pack,ctx,graph.to(device),edges,endpoint,geometry,candidates,clean_index,'isolated_uts_17301_C_CAL_fit')
    result.update(checkpoint_sha256=identity,checkpoint_file_unchanged=sha(checkpoint)==identity,
                  fit_key_sha256=joint.s3.key_sha(pack['keys']),diagnostic_candidates=len(candidates),
                  formal_run_retried=False,cal_eval_rows_scored=0,internal_test_rows_scored=0,official_test_rows_scored=0,
                  model_parameter_grad_none=all(p.grad is None for p in model.parameters()))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('synthetic','isolated-real'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    if args.mode=='synthetic':
        results=[synthetic(device,dtype) for device,dtype in (('cpu',torch.float32),('cuda',torch.float32),('cuda',torch.float64))]
    else:
        results=[isolated_real()]
    target=AUDIT/(args.mode+'_RESULT.json')
    with target.open('x',encoding='utf-8') as handle:
        json.dump({'mode':args.mode,'results':results,'scientific_source_modified':False},handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')
    print(str(target),flush=True)


if __name__=='__main__':
    main()
